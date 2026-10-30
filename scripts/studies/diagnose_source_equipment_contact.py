"""Explain sealed contact counts without treating image proximity as grasp or support."""

import argparse
from collections import Counter
import itertools
import os
from pathlib import Path
import signal
import time
import xml.etree.ElementTree as ET

os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')

import mujoco
import numpy as np
from PIL import Image

from weft.shared_world import SharedWorldEnv
from check_equipment_camera_bridge import ROOT, now, pose_points, read, sha, statistics, write
from check_equipment_perspective_bridge import perspective, signed_point_box


CONTRACT = {
    'method': 'source-hand-model-depth-and-contact-diagnosis-v1', 'maximum_seconds': 60,
    'maximum_output_bytes': 64 * 1024 * 1024, 'maximum_replayed_actions': 60,
    'source_confidence_threshold': .5, 'image_proximity_px': 3., 'large_3d_gap_model_units': .05,
    'replay': 'One exact frozen authored-reference action replay; no regenerated controller actions, policy training or source inference',
    'camera_alternatives': 'Reuse only four previously sealed authored perspective distances, with unchanged body/floor/root',
    'ray_probe': 'Unapplied source-image wrist ray versus authored box, with modeled elbow and forearm length fixed; not a recovered pose or contact',
    'clocks': 'Substep contact geometry belongs to the pre-integration configuration; source association uses declared frame/FPS pose clock only',
    'scope': 'Construction/model consistency and simulator mechanics; no observed grasp, support, biomechanics or physical validation',
}
HANDS = {'left': {'wrist': 15, 'pinky': 17, 'index': 19, 'thumb': 21},
         'right': {'wrist': 16, 'pinky': 18, 'index': 20, 'thumb': 22}}


def mask_cells(row, directory):
    bitmap = np.asarray(Image.open(directory / row['raster_path'])) > 0
    y, x = np.nonzero(bitmap)
    x0, y0, x1, y1 = row['box']['box_xyxy_halfopen']
    width, height = bitmap.shape[1], bitmap.shape[0]
    return np.column_stack((x0 + x * (x1 - x0) / width, y0 + y * (y1 - y0) / height,
                            x0 + (x + 1) * (x1 - x0) / width, y0 + (y + 1) * (y1 - y0) / height))


def image_gap(pixel, cells):
    delta = np.maximum(np.maximum(cells[:, :2] - pixel, pixel - cells[:, 2:]), 0.)
    return float(np.sqrt(np.min(np.sum(delta ** 2, axis=1))))


def ray_probe(pixel, wrist, elbow, camera, center, half):
    origin = np.array([0., -camera['distance_model_units'], camera['world_anchor_model_units'][2]])
    direction = np.array([(pixel[0] - camera['pixel_origin'][0]) / camera['focal_px'], 1.,
                          -(pixel[1] - camera['pixel_origin'][1]) / camera['focal_px']])
    lo, hi = .05, float('inf')
    for axis in range(3):
        if abs(direction[axis]) < 1e-12:
            if not center[axis] - half[axis] <= origin[axis] <= center[axis] + half[axis]:
                return {'status': 'source_wrist_ray_misses_solid_box'}
        else:
            limits = sorted(((center[axis] - half[axis] - origin[axis]) / direction[axis],
                             (center[axis] + half[axis] - origin[axis]) / direction[axis]))
            lo, hi = max(lo, limits[0]), min(hi, limits[1])
    if hi < lo:
        return {'status': 'source_wrist_ray_misses_solid_box'}
    length = float(np.linalg.norm(wrist - elbow))
    nearest = origin + np.clip(np.dot(wrist - origin, direction) / np.dot(direction, direction), lo, hi) * direction
    offset = origin - elbow
    a, b, c = np.dot(direction, direction), 2 * np.dot(offset, direction), np.dot(offset, offset) - length ** 2
    discriminant = b * b - 4 * a * c
    roots = [] if discriminant < 0 else [(-b - np.sqrt(discriminant)) / (2 * a), (-b + np.sqrt(discriminant)) / (2 * a)]
    feasible = [value for value in roots if lo - 1e-9 <= value <= hi + 1e-9]
    return {'status': 'fixed_elbow_length_feasible' if feasible else 'requires_elbow_or_arm_pose_change',
            'camera_depth_interval_model_units': [float(lo), float(hi)], 'original_forearm_length_model_units': length,
            'closest_box_ray_point_model_units': nearest.tolist(), 'required_xyz_correction_model_units': (nearest - wrist).tolist(),
            'minimum_wrist_correction_model_units': float(np.linalg.norm(nearest - wrist)),
            'feasible_ray_depths_model_units': list(map(float, feasible))}


def replay(scene, saved):
    env = SharedWorldEnv(scene)
    original_step = mujoco.mj_step
    contacts, balance, agreement = [], [], []
    target = scene['task']['object_goal']['target_object_id']
    gid, body = env.model.geom(target).id, env.model.body(target).id
    joint = env.model.joint(target + '_joint').id
    q, v = env.model.jnt_qposadr[joint], env.model.jnt_dofadr[joint]
    counter = 0

    def observed_step(model, data, *args, **kwargs):
        nonlocal counter
        before = float(data.time)
        original_step(model, data, *args, **kwargs)
        counter += 1
        total_force = np.zeros(3)
        for index in range(data.ncon):
            contact = data.contact[index]
            if gid not in (contact.geom1, contact.geom2):
                continue
            force = np.zeros(6)
            mujoco.mj_contactForce(model, data, index, force)
            direction = 1. if contact.geom2 == gid else -1.
            world_force = direction * contact.frame.reshape(3, 3).T @ force[:3]
            total_force += world_force
            other = int(contact.geom1 if contact.geom2 == gid else contact.geom2)
            if other not in env._human_geom_to_name:
                continue
            name = env._human_geom_to_name[other]
            source_start = env.source_start + before
            source_end = env.source_start + float(data.time)
            contacts.append({'physics_substep': counter, 'control_step': (counter - 1) // scene['substeps'] + 1,
                'simulation_interval_s': [before, float(data.time)], 'source_assumed_pose_clock_interval_s': [source_start, source_end],
                'source_frame_bracket_assumed_clock': [int(np.floor(source_start * 30 + 1e-8)), int(np.ceil(source_end * 30 - 1e-8))],
                'human_geom': name, 'body_segment': name.replace('_elbow_geom', '_forearm').replace('_shoulder_geom', '_upper_arm').replace('_knee_geom', '_shin').replace('_hip_geom', '_thigh'),
                'target': target, 'distance_world': float(contact.dist), 'position_world_preintegration': contact.pos.tolist(),
                'normal_force_simulator_units': float(force[0]), 'force_on_target_world': world_force.tolist(),
                'normal_toward_target_world': (direction * contact.frame[:3]).tolist(),
                'target_position_postintegration': data.qpos[q:q + 3].tolist(),
                'target_velocity_postintegration': data.qvel[v:v + 3].tolist(),
                'wrist_positions_preintegration': {side: data.site_xpos[model.site(side + '_wrist_site').id].tolist() for side in HANDS}})
        balance.append(float(np.linalg.norm(total_force - data.qfrc_constraint[v:v + 3])))

    try:
        env.reset(seed=saved['seed'])
        mujoco.mj_step = observed_step
        for index, frozen in enumerate(saved['frames']):
            if index >= CONTRACT['maximum_replayed_actions'] or np.any(np.asarray(frozen['action'])[22:] != 0):
                raise ValueError('Replay exceeds frozen action/zero-equipment-force contract')
            _, reward, terminated, truncated, info = env.step(frozen['action'])
            state = env.snapshot()
            errors = [abs(reward - frozen['reward'])]
            for name, point in frozen['state']['human']['joints'].items():
                errors.append(float(np.max(np.abs(np.asarray(point) - state['human']['joints'][name]))))
            for actual, expected in zip(state['objects'], frozen['state']['objects']):
                for key in ('position', 'velocity', 'quaternion', 'angular_velocity'):
                    errors.append(float(np.max(np.abs(np.asarray(actual[key]) - expected[key]))))
            if (max(errors) > 1e-9 or terminated != frozen['terminated'] or truncated != frozen['truncated']
                    or info['human_goal_object_contacts_total'] != frozen['info']['human_goal_object_contacts_total']):
                raise ValueError('Instrumented replay differs from sealed control-boundary states')
            agreement.append(max(errors))
        if len(contacts) != saved['frames'][-1]['info']['human_goal_object_contacts_total']:
            raise ValueError('Substep trace does not explain the complete sealed contact count')
    finally:
        mujoco.mj_step = original_step
        env.close()
    return contacts, {'replayed_control_steps': len(agreement), 'physics_substeps': counter,
                      'maximum_boundary_state_error': max(agreement),
                      'maximum_world_force_vs_generalized_constraint_force_error': max(balance),
                      'contact_force_direction_convention': 'Force on target = contact-frame force rotated to world, positive on geom2; checked against target generalized constraint force'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--integration', type=Path, required=True)
    parser.add_argument('--body-run', type=Path, required=True)
    parser.add_argument('--mask-evidence', type=Path, required=True)
    parser.add_argument('--camera-study', type=Path, required=True)
    parser.add_argument('--body-projection-study', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    started, cpu = time.monotonic(), time.process_time()
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError('60-second diagnostic bound')))
    signal.alarm(60)
    out = args.out.resolve(); out.mkdir(parents=True, exist_ok=False)
    write(out / 'pre-run.json', {'created_at': now(), 'contract': CONTRACT, 'inputs': {k: str(v.resolve()) for k, v in vars(args).items()}})
    manifests, bindings = {}, {}

    def bound(directory, name):
        directory = directory.resolve()
        if directory not in manifests: manifests[directory] = read(directory / 'manifest.json')
        path = directory / name
        digest = sha(path)
        if manifests[directory]['files'].get(name) != digest:
            raise ValueError('Sealed input changed: ' + str(path))
        bindings[str(path.relative_to(ROOT))] = {'sha256': digest, 'bytes': path.stat().st_size,
                                               'manifest_sha256': sha(directory / 'manifest.json')}
        return path

    scene = read(bound(args.integration, 'candidate/scene.json'))
    authored = read(bound(args.integration, 'authored_reference-rollout.json'))
    coast = read(bound(args.integration, 'coast-rollout.json'))
    evidence = read(bound(args.body_run, 'evidence.json'))
    human = read(bound(args.body_run, 'human.json'))
    xml = bound(args.body_run, 'human.xml').read_text()
    masks = read(bound(args.mask_evidence, 'masks/mask-evidence.json'))
    camera_models = read(bound(args.camera_study, 'candidates.json'))
    source_slots = read(bound(args.camera_study, 'source-slot-alignment.json'))
    projection = read(bound(args.body_projection_study, 'body-reprojection.json'))
    if scene['human_spec'] != human or evidence['source_sha256'] != human['source_sha256']:
        raise ValueError('Candidate changed the original body/source contract')
    target = scene['task']['object_goal']['target_object_id']
    object_spec = next(obj for obj in scene['objects'] if obj['id'] == target)
    camera = object_spec['provenance']['camera_hypothesis']
    center = np.asarray(object_spec['provenance']['geometry_model_units']['center'])
    half = np.asarray(object_spec['provenance']['geometry_model_units']['half_extents'])
    source_pose = {row['frame']: row for row in evidence['frames']}
    source_masks = {row['frame']: row for row in masks['observations'] if row['object_id'] == target}
    world_points = {}
    for row in projection['points']:
        world_points.setdefault(row['frame'], {})[row['joint']] = np.asarray(row['world_model_units'])
    reference_indices = {int(row['evidence_id'].rsplit(':', 1)[1]): i for i, row in enumerate(human['reference_evidence'])}
    contacts, replay_verification = replay(scene, authored)
    if replay_verification['maximum_world_force_vs_generalized_constraint_force_error'] > 1e-6:
        raise ValueError('Contact force direction decomposition failed its generalized-force check')
    tree = ET.fromstring(xml)
    ET.SubElement(tree.find('worldbody'), 'geom', name='study_target', type='box',
                  pos=' '.join(map(str, center)), size=' '.join(map(str, half)), contype='0', conaffinity='0', density='0')
    model = mujoco.MjModel.from_xml_string(ET.tostring(tree, encoding='unicode')); data = mujoco.MjData(model)
    point_rows, pose_rows, alternative_rows = [], [], []
    slots = []
    for frame in range(evidence['decoded_frames']):
        pose, mask, ref = source_pose.get(frame), source_masks[frame], reference_indices.get(frame)
        slot = dict(source_slots['frames'][frame]); slot['target_mask_evidence_id'] = mask['id']; slot['target_mask_phase'] = mask['phase']
        slot['target_identity_status'] = mask['identity']['status']; slots.append(slot)
        cells = None
        if mask['mask_status'] == 'decoded':
            bound(args.mask_evidence, 'masks/' + mask['raster_path'])
            cells = mask_cells(mask, args.mask_evidence / 'masks')
        landmarks = pose.get('landmarks') if pose else None
        raw_world = pose.get('world_landmarks') if pose else None
        raw_hips = None
        if isinstance(raw_world, list) and len(raw_world) == 33:
            raw_hips = np.mean([[raw_world[i][key] for key in ('x', 'y', 'z')] for i in (23, 24)], axis=0)
        points = None
        if ref is not None:
            points = pose_points(model, data, human, ref)
            for name, point in points.items():
                if np.max(np.abs(point - world_points[frame][name])) > 1e-10:
                    raise ValueError('Unchanged saved body kinematics differ from prior verified projection study')
        for side, names in HANDS.items():
            hand_rows = []
            for name, index in names.items():
                row = {'frame': frame, 'side': side, 'landmark': name, 'landmark_index': index,
                       'pose_status': pose['status'] if pose else 'not_sampled', 'source_evidence_id': pose['id'] if pose else None,
                       'assumed_pose_time_s': pose['time_s'] if pose else None, 'decoder_time_s': pose.get('decoder_timestamp_s') if pose else None,
                       'mask_evidence_id': mask['id'], 'mask_phase': mask['phase'], 'image_px': None, 'confidence': None,
                       'confident_image_estimate': False, 'source_mask_gap_px': None, 'raw_world_hip_centered_model': None}
                if isinstance(landmarks, list) and len(landmarks) == 33:
                    lm = landmarks[index]
                    pixel = np.array([lm['x'] * evidence['width'], lm['y'] * evidence['height']])
                    confidence = min(lm['visibility'], lm['presence'])
                    confident = (pose['status'] in ('tracked', 'tracked_partial') and pose.get('pose_count') == 1
                                 and confidence >= .5 and 0 <= lm['x'] <= 1 and 0 <= lm['y'] <= 1)
                    row.update(image_px=pixel.tolist(), confidence=confidence, confident_image_estimate=confident,
                               source_mask_gap_px=image_gap(pixel, cells) if cells is not None else None,
                               raw_image_depth_image_widths=lm.get('z'))
                    if raw_hips is not None:
                        relative = np.array([raw_world[index][key] for key in ('x', 'y', 'z')]) - raw_hips
                        row['raw_world_hip_centered_model'] = relative[[0, 2, 1]].tolist()
                        row['raw_world_hip_centered_model'][2] *= -1
                point_rows.append(row); hand_rows.append(row)
            if points is None:
                continue
            wrist, elbow = points[side + '_wrist'], points[side + '_elbow']
            source = hand_rows[0]
            projected, _, valid = perspective(np.asarray([wrist]), camera['distance_model_units'], camera['original_scale_at_anchor_model_units_per_pixel'],
                                               np.array(camera['pixel_origin']), camera['world_anchor_model_units'][2])
            closest = np.zeros(6)
            distance = float(mujoco.mj_geomDistance(model, data, model.geom(side + '_elbow_geom').id, model.geom('study_target').id, 10., closest))
            image_estimates = [row for row in hand_rows if row['confident_image_estimate'] and row['source_mask_gap_px'] is not None]
            result = {'frame': frame, 'side': side, 'reference_index': ref,
                      'modeled_wrist_model_units': wrist.tolist(), 'modeled_elbow_model_units': elbow.tolist(),
                      'wrist_point_box_signed_gap_model_units': signed_point_box(wrist, center, half),
                      'forearm_capsule_box_signed_gap_model_units': distance,
                      'closest_forearm_box_points_model_units': closest.reshape(2, 3).tolist(),
                      'modeled_wrist_projected_px': projected[0].tolist() if valid[0] else None,
                      'modeled_wrist_mask_gap_px': image_gap(projected[0], cells) if cells is not None and valid[0] else None,
                      'source_wrist': source, 'minimum_confident_hand_landmark_mask_gap_px': min((row['source_mask_gap_px'] for row in image_estimates), default=None)}
            if source['image_px'] is not None and valid[0]:
                result['wrist_image_reprojection_error_px'] = float(np.linalg.norm(projected[0] - source['image_px']))
                result['ray_feasibility'] = ray_probe(np.array(source['image_px']), wrist, elbow, camera, center, half)
            if source['raw_world_hip_centered_model'] is not None:
                raw = np.asarray(source['raw_world_hip_centered_model']) + points['pelvis']
                result.update(raw_world_wrist_in_body_origin=raw.tolist(), modeled_minus_raw_world_wrist=(wrist - raw).tolist(),
                              raw_world_wrist_box_signed_gap_model_units=signed_point_box(raw, center, half))
            pose_rows.append(result)
            for candidate in camera_models['candidates']:
                proxy = candidate['equipment'][target]
                alternative_rows.append({'camera_hypothesis': candidate['id'], 'frame': frame, 'side': side,
                    'wrist_box_signed_gap_model_units': signed_point_box(wrist, np.asarray(proxy['center_model_units']), np.asarray(proxy['half_extents_model_units'])),
                    'authored_box_center_depth_model_units': proxy['center_model_units'][1]})
    summaries = {}
    for side in HANDS:
        rows = [row for row in pose_rows if row['side'] == side]
        near = [row for row in rows if row['minimum_confident_hand_landmark_mask_gap_px'] is not None
                and row['minimum_confident_hand_landmark_mask_gap_px'] <= CONTRACT['image_proximity_px']]
        summaries[side] = {'reference_poses': len(rows),
            'confident_source_hand_near_mask_poses': len(near),
            'image_near_but_wrist_gap_over_005': sum(row['wrist_point_box_signed_gap_model_units'] > .05 for row in near),
            'image_near_but_forearm_gap_over_005': sum(row['forearm_capsule_box_signed_gap_model_units'] > .05 for row in near),
            'wrist_box_gap_model_units': statistics([row['wrist_point_box_signed_gap_model_units'] for row in rows]),
            'forearm_box_gap_model_units': statistics([row['forearm_capsule_box_signed_gap_model_units'] for row in rows]),
            'wrist_image_reprojection_error_px': statistics([row['wrist_image_reprojection_error_px'] for row in rows if 'wrist_image_reprojection_error_px' in row]),
            'modeled_minus_raw_world_wrist_depth': statistics([row['modeled_minus_raw_world_wrist'][1] for row in rows if 'modeled_minus_raw_world_wrist' in row]),
            'confident_wrist_ray_probe_statuses': dict(Counter(row['ray_feasibility']['status'] for row in rows if row['source_wrist']['confident_image_estimate'] and 'ray_feasibility' in row))}
    alternative_summary = {candidate['id']: {side: statistics([row['wrist_box_signed_gap_model_units'] for row in alternative_rows
        if row['camera_hypothesis'] == candidate['id'] and row['side'] == side]) for side in HANDS} for candidate in camera_models['candidates']}
    initial_world = np.array(object_spec['position'])
    rollout_summary = {}
    for name, saved in [('coast', coast), ('authored_reference', authored)]:
        trace = []
        for index, row in enumerate(saved['frames']):
            obj = next(obj for obj in row['state']['objects'] if obj['id'] == target)
            trace.append({'control_step': index + 1, 'source_assumed_time_s': row['state']['source_time_s'],
                          'target_displacement_world': (np.array(obj['position']) - initial_world).tolist(),
                          'target_velocity_world': obj['velocity'], 'cumulative_target_contact_records': row['info']['human_goal_object_contacts_total'],
                          'boundary_human_contacts': [contact for contact in row['state']['contact_details'] if contact['kind'] == 'human_object']})
        rollout_summary[name] = {'trace': trace, 'final_info': saved['frames'][-1]['info']}
    impulses = np.sum([np.asarray(row['force_on_target_world']) * (row['simulation_interval_s'][1] - row['simulation_interval_s'][0]) for row in contacts], axis=0)
    report = {'created_at': now(), 'status': 'diagnosis_complete_no_runtime_or_source_repair', 'contract': CONTRACT,
              'source_frames': len(slots), 'body_references': len(reference_indices), 'source_pose_status_counts': dict(Counter(row['pose_status'] for row in slots)),
              'target_mask_phase_counts': dict(Counter(row['target_mask_phase'] for row in slots)),
              'contact_trace': {'records': len(contacts), 'body_segments': dict(Counter(row['body_segment'] for row in contacts)),
                  'control_steps': dict(Counter(row['control_step'] for row in contacts)),
                  'simulation_interval_s': [contacts[0]['simulation_interval_s'][0], contacts[-1]['simulation_interval_s'][1]],
                  'source_assumed_interval_s': [contacts[0]['source_assumed_pose_clock_interval_s'][0], contacts[-1]['source_assumed_pose_clock_interval_s'][1]],
                  'normal_force_simulator_units': statistics([row['normal_force_simulator_units'] for row in contacts]),
                  'summed_human_contact_impulse_on_target_world': impulses.tolist(),
                  'scope': 'Substep contact records from one short forearm collision interval, not separate grasps or observed physical trials'},
              'replay_verification': replay_verification, 'hand_geometry': summaries, 'existing_camera_alternatives': alternative_summary,
              'recommended_first_repair': 'Constrained full-arm reference alignment against source image landmarks and existing model depth, preserving body shape/root/floor; audit elbow and wrist together before any support/grasp controller.',
              'repair_scope': 'New bounded pose candidate only, never overwrite raw image/world landmarks; preserve failures and all source references. A hand-support controller is not justified by the current proxy/forearm collision.',
              'limitations': ['Image landmark proximity to a proposed silhouette does not identify physical depth, hand contact, force or support.',
                  'The solid box includes empty handle/cavity space and the human model has no hand/finger collision geometry.',
                  'Raw model-world depth and the articulated reference are both inferred; their discrepancy does not establish which is physically correct.',
                  'The exact-action replay covers only source-assumed frame2 through62, not the full exercise; kinematic comparisons retain all121 references/257slots.',
                  'Source wrists/pinky/index/thumb landmarks and confidence are model estimates; unavailable/low-confidence slots remain explicit.',
                  'Ray feasibility tests source-image rays plus fixed elbow/length only; successful intersection would not validate joints, anatomy, contact or dynamics.'],
              'cpu_seconds': time.process_time() - cpu, 'wall_seconds': time.monotonic() - started,
              'paid_calls': 0, 'inference_calls': 0, 'video_decode_calls': 0, 'runtime_source_sha256': sha(ROOT / 'src/weft/shared_world.py')}
    write(out / 'contact-substeps.json', {'contacts': contacts, 'verification': replay_verification})
    write(out / 'source-slots.json', {'frames': slots})
    write(out / 'source-hand-landmarks.json', {'observations': point_rows})
    write(out / 'reference-hand-geometry.json', {'poses': pose_rows})
    write(out / 'camera-alternative-gaps.json', {'comparisons': alternative_rows, 'scope': 'Existing sealed hypotheses only; no new camera/body/equipment fit'})
    write(out / 'saved-rollout-comparison.json', rollout_summary)
    write(out / 'report.json', report); write(out / 'input-bindings.json', bindings)
    (out / Path(__file__).name).write_text(Path(__file__).read_text())
    for path, value in bindings.items():
        if sha(ROOT / path) != value['sha256']: raise ValueError('Input changed during diagnostic')
    if sum(path.stat().st_size for path in out.iterdir() if path.is_file()) > CONTRACT['maximum_output_bytes']:
        raise ValueError('Diagnostic output exceeds 64 MiB')
    write(out / 'manifest.json', {'created_at': now(), 'files': {path.name: sha(path) for path in sorted(out.iterdir()) if path.is_file()}})
    signal.alarm(0)
    print({'output': str(out), 'records': len(contacts), 'segments': report['contact_trace']['body_segments'],
           'replay_error': replay_verification['maximum_boundary_state_error'], 'wall_seconds': report['wall_seconds'], 'report_sha256': sha(out / 'report.json')})


if __name__ == '__main__':
    main()
