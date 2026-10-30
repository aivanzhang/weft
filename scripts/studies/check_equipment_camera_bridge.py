"""Read-only source/body camera feasibility; no inferred depth or contact is promoted."""

import argparse
from collections import Counter
from datetime import datetime
from fractions import Fraction
import hashlib
import itertools
import json
import os
from pathlib import Path
import signal
import time
import xml.etree.ElementTree as ET
from zoneinfo import ZoneInfo

os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')

import mujoco
import numpy as np
from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[2]
JOINTS = {'left_shoulder': 11, 'right_shoulder': 12, 'left_elbow': 13, 'right_elbow': 14,
          'left_wrist': 15, 'right_wrist': 16, 'left_hip': 23, 'right_hip': 24,
          'left_knee': 25, 'right_knee': 26, 'left_ankle': 27, 'right_ankle': 28}
BASE_ROTATION = np.array([[1., 0., 0.], [0., 0., -1.], [0., 1., 0.]])
CONTRACT = {
    'method': 'fixed-body-equipment-camera-feasibility-v1',
    'maximum_seconds': 60, 'maximum_output_bytes': 64 * 1024 * 1024,
    'maximum_frames': 900, 'minimum_image_confidence': .5,
    'fit_scope': 'construction_model_estimates_only',
    'maximum_camera_rotation_degrees': 15., 'maximum_camera_evaluations': 512,
    'camera_fit': 'One global proper rotation; fixed torso scale and frame-origin pelvis anchor; no translation, scale, timing, or per-frame fit',
    'camera_search': 'Rotation-vector grid at 7.5 degrees inside a 15-degree ball; deterministic coordinate refinement to 0.05859375 degrees',
    'fit_loss': 'Unweighted squared pixel residual of eligible twelve joints and pelvis; all other finite residuals retained',
    'equipment_shape': 'Axis-aligned solid box collision proxy; width and height from first common silhouette bounds times original body scale; depth equals width',
    'equipment_support': 'Authored box bottom at unchanged floor; no observed contact',
    'baseline_depth': 'Authored y=0; no depth search to improve overlap',
    'alternative_depth': 'Solve fixed-floor box-center projection if nonsingular; reject |x| or |y| > 2 model units, never clamp',
    'maximum_equipment_xy_model_units': 2.,
    'preinitialization': 'No mask backfill; candidate at body start requires explicit static extrapolation from first common frame',
    'dynamics': 'None; zero start velocities/mass/contact/material would be additional authored simulation priors',
}


def read(path):
    return json.loads(path.read_text())


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n')


def now():
    return datetime.now(ZoneInfo('America/Los_Angeles')).isoformat()


def statistics(values):
    data = np.asarray(values, dtype=float)
    return {'count': len(data), 'rmse': float(np.sqrt(np.mean(data ** 2))) if len(data) else None,
            'median': float(np.median(data)) if len(data) else None,
            'p95': float(np.quantile(data, .95)) if len(data) else None,
            'maximum': float(data.max()) if len(data) else None,
            'minimum': float(data.min()) if len(data) else None}


def rotation_vector(vector):
    vector = np.asarray(vector, dtype=float)
    angle = np.linalg.norm(vector)
    if angle < 1e-15:
        return np.eye(3)
    x, y, z = vector / angle
    cross = np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])
    return np.eye(3) + np.sin(angle) * cross + (1 - np.cos(angle)) * (cross @ cross)


def project(points, rotation, origin, anchor, scale):
    return (np.asarray(points) - anchor) @ rotation[:2].T / scale + origin


def camera_fit(world, pixels, origin, anchor, scale):
    best, loss, evaluations = np.zeros(3), float('inf'), 0
    bound = np.deg2rad(CONTRACT['maximum_camera_rotation_degrees'])

    def evaluate(vector):
        nonlocal best, loss, evaluations
        if np.linalg.norm(vector) > bound + 1e-12:
            return False
        if evaluations >= CONTRACT['maximum_camera_evaluations']:
            return False
        evaluations += 1
        residual = project(world, BASE_ROTATION @ rotation_vector(vector), origin, anchor, scale) - pixels
        value = float(np.mean(np.sum(residual ** 2, axis=1)))
        if value < loss - 1e-12:
            best, loss = vector.copy(), value
            return True
        return False

    evaluate(np.zeros(3))
    for vector in itertools.product(np.deg2rad([-15., -7.5, 0., 7.5, 15.]), repeat=3):
        evaluate(np.asarray(vector))
    for degrees in [3.75, 1.875, .9375, .46875, .234375, .1171875, .05859375]:
        for _ in range(8):
            improved = False
            for axis in range(3):
                for sign in (-1, 1):
                    trial = best.copy()
                    trial[axis] += np.deg2rad(degrees) * sign
                    improved = evaluate(trial) or improved
            if not improved:
                break
    rotation = BASE_ROTATION @ rotation_vector(best)
    assert np.allclose(rotation @ rotation.T, np.eye(3), atol=1e-12)
    assert abs(np.linalg.det(rotation) - 1) < 1e-12
    return rotation, {'rotation_vector_radians': best.tolist(), 'rotation_degrees': float(np.rad2deg(np.linalg.norm(best))),
                      'evaluations': evaluations, 'fit_rmse_px': float(np.sqrt(loss)),
                      'at_bound': bool(np.rad2deg(np.linalg.norm(best)) > 14.9),
                      'search_scope': 'Bounded deterministic search; not proof of a global camera optimum'}


def pose_points(model, data, human, index):
    mujoco.mj_resetData(model, data)
    for name, value in human['reference'][index]['joints'].items():
        data.qpos[model.jnt_qposadr[model.joint(name).id]] = value
    for axis, value in zip('xyz', human['root_reference'][index]['translation']):
        data.qpos[model.jnt_qposadr[model.joint('root_' + axis).id]] = value
    mujoco.mj_forward(model, data)
    return {name: data.site_xpos[model.site(name + '_site').id].copy()
            for name in [*JOINTS, 'pelvis']}


def image_point(pose, name, width, height):
    if pose is None or pose.get('pose_count') != 1:
        return None, None, 'pose_unavailable'
    landmarks = pose.get('landmarks')
    if not isinstance(landmarks, list) or len(landmarks) != 33:
        return None, None, 'landmarks_unavailable'
    indices = [23, 24] if name == 'pelvis' else [JOINTS[name]]
    rows = [landmarks[index] for index in indices]
    values = np.array([[row.get(k, np.nan) for k in ('x', 'y', 'visibility', 'presence')] for row in rows])
    if not np.isfinite(values).all():
        return None, None, 'invalid_landmark'
    pixel = values[:, :2].mean(axis=0) * [width, height]
    confidence = float(values[:, 2:].min())
    if np.any(values[:, :2] < 0) or np.any(values[:, :2] > 1):
        status = 'outside_source_image'
    elif confidence < CONTRACT['minimum_image_confidence']:
        status = 'low_confidence'
    elif pose.get('status') not in ('tracked', 'tracked_partial'):
        status = 'unsupported_pose_status'
    else:
        status = 'eligible_construction_estimate'
    return pixel, confidence, status


def box_projection(center, half, rotation, origin, anchor, scale):
    pixel = project(np.asarray([center]), rotation, origin, anchor, scale)[0]
    extent = np.abs(rotation[:2]) @ half / scale
    return np.concatenate((pixel - extent, pixel + extent))


def proxy_hypothesis(observation, rotation, origin, anchor, scale, floor, baseline):
    bounds = np.asarray(observation['silhouette']['foreground_bounds_px'], dtype=float)
    center_px = (bounds[:2] + bounds[2:]) / 2
    width, height = bounds[2:] - bounds[:2]
    half = np.array([width, width, height]) * scale / 2
    center = np.array([(center_px[0] - origin[0]) * scale, 0., floor + half[2]])
    status = 'authored_fixed_depth_floor_proxy'
    ground_matrix = rotation[:2, :2]
    singular = np.linalg.svd(ground_matrix, compute_uv=False)
    if not baseline:
        if singular[-1] < 1e-6:
            return {'status': 'rejected_singular_floor_projection', 'ground_singular_values': singular.tolist()}
        rhs = (center_px - origin) * scale - rotation[:2, 2] * (center[2] - anchor[2])
        center[:2] = np.linalg.solve(ground_matrix, rhs)
        status = ('authored_floor_depth_solution' if np.max(np.abs(center[:2])) <= CONTRACT['maximum_equipment_xy_model_units']
                  else 'rejected_depth_or_lateral_bound')
    projected = box_projection(center, half, rotation, origin, anchor, scale)
    return {'status': status, 'source_frame': observation['frame'], 'source_evidence_id': observation['id'],
            'center_model_units': center.tolist(), 'half_extents_model_units': half.tolist(),
            'shape': 'solid_axis_aligned_box_proxy_not_recovered_kettlebell',
            'input_silhouette_bounds_px': bounds.tolist(), 'projected_bounds_px': projected.tolist(),
            'bbox_edge_error_px': (projected - bounds).tolist(),
            'bbox_edge_rmse_px': float(np.sqrt(np.mean((projected - bounds) ** 2))),
            'bbox_center_error_px': float(np.linalg.norm((projected[:2] + projected[2:] - bounds[:2] - bounds[2:]) / 2)),
            'bottom_at_original_floor': bool(abs(center[2] - half[2] - floor) < 1e-12),
            'ground_singular_values': singular.tolist(),
            'source_depth_observed': False, 'source_support_observed': False,
            'source_identity_verified': False, 'mass_or_material_estimated': False}


def overlap_checks(xml_text, human, proxies, frames_to_indices):
    tree = ET.fromstring(xml_text)
    world = tree.find('worldbody')
    admitted = {name: row for name, row in proxies.items() if not row['status'].startswith('rejected')}
    for name, row in admitted.items():
        ET.SubElement(world, 'geom', name='study_' + name, type='box',
                      pos=' '.join(map(str, row['center_model_units'])),
                      size=' '.join(map(str, row['half_extents_model_units'])),
                      contype='0', conaffinity='0', density='0')
    model = mujoco.MjModel.from_xml_string(ET.tostring(tree, encoding='unicode'))
    data = mujoco.MjData(model)
    human_geoms = [i for i in range(model.ngeom) if model.geom_bodyid[i] != 0]
    results = []
    for frame, index in frames_to_indices.items():
        points = pose_points(model, data, human, index)
        for name, proxy in admitted.items():
            distances = [{'body_geom': model.geom(i).name,
                          'signed_distance_model_units': float(mujoco.mj_geomDistance(model, data, i, model.geom('study_' + name).id, 10., None))}
                         for i in human_geoms]
            nearest = min(distances, key=lambda row: row['signed_distance_model_units'])
            penetrating = [row for row in distances if row['signed_distance_model_units'] < -1e-8]
            results.append({'source_frame': frame, 'reference_index': index, 'object_id': name,
                            'nearest_body_geometry': nearest, 'penetrating_body_geometries': penetrating,
                            'initial_overlap_rejected': bool(penetrating),
                            'wrist_positions_model_units': {k: points[k].tolist() for k in ('left_wrist', 'right_wrist')},
                            'scope': 'Exact collision-proxy separation at unchanged source poses; neither physical contact nor dynamics'})
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--body-run', type=Path, required=True)
    parser.add_argument('--equipment-evidence', type=Path, required=True)
    parser.add_argument('--equipment', nargs='+', default=['sam-2', 'sam-3'])
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    started, cpu_started = time.monotonic(), time.process_time()
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError('60-second geometry study bound')))
    signal.alarm(CONTRACT['maximum_seconds'])
    out, body, equipment = args.out.resolve(), args.body_run.resolve(), args.equipment_evidence.resolve()
    out.mkdir(parents=True, exist_ok=False)
    write(out / 'pre-run.json', {'created_at': now(), 'contract': CONTRACT,
                                'body_run': str(body), 'equipment_evidence': str(equipment), 'equipment_ids': args.equipment})
    bindings = {}

    def bound(path, sealed_root):
        manifest = read(sealed_root / 'manifest.json')
        relative = path.relative_to(sealed_root).as_posix()
        value = sha(path)
        if manifest['files'].get(relative) != value:
            raise ValueError('Used input differs from its sealed manifest: ' + str(path))
        bindings[str(path.relative_to(ROOT))] = {'sha256': value, 'bytes': path.stat().st_size,
                                               'manifest_sha256': sha(sealed_root / 'manifest.json')}
        return path

    human = read(bound(body / 'human.json', body))
    evidence = read(bound(body / 'evidence.json', body))
    root = read(bound(body / 'root-motion.json', body))
    saved = read(bound(body / 'human-reference.json', body))
    xml_text = bound(body / 'human.xml', body).read_text()
    masks = read(bound(equipment / 'masks/mask-evidence.json', equipment))
    mapping = read(bound(equipment / 'inputs/preparation/frame-mapping-and-pixels.json', equipment))
    scene = read(bound(equipment / 'evidence/scene-observations.json', equipment))
    if human['profile'] != 'exercise-v3' or 'camera_reference' in human:
        raise ValueError('Study requires the original uncalibrated mobile body')
    if not human['source_sha256'] == evidence['source_sha256'] == root['source_sha256'] == mapping['source_sha256']:
        raise ValueError('Original source identities differ')
    if not 1 <= len(mapping['frames']) <= CONTRACT['maximum_frames']:
        raise ValueError('Source frame count outside bounded study')
    count, width, height = len(mapping['frames']), evidence['width'], evidence['height']
    time_base = float(Fraction(mapping['time_base']))
    if (count != evidence['decoded_frames'] or len(masks['frames']) != count
            or len(masks['observations']) != 3 * count or len(scene['observations']) != 3 * count):
        raise ValueError('Complete original/derivative source slots must agree')
    for index, row in enumerate(mapping['frames']):
        if (row['source_frame'] != index or row['relative_frame'] != index
                or row['source_yuv_sha256'] != row['derivative_yuv_sha256']
                or row['source_native_pts'] != row['derivative_native_pts']
                or row['source_decoder_pts'] != row['derivative_decoder_pts']):
            raise ValueError('Frame identity/pixel/clock mapping is not exact')
    observations = {(row['frame'], row['object_id']): row for row in masks['observations']}
    if len(observations) != len(masks['observations']):
        raise ValueError('Duplicate segmentation slots')
    for row in masks['observations']:
        mapped = mapping['frames'][row['frame']]
        original = row['original_source']
        if (not row['id'].startswith(mapping['derivative_sha256'] + ':')
                or original['sha256'] != mapping['source_sha256'] or original['frame'] != row['frame']
                or abs(original['time_s'] - mapped['source_native_pts'] * time_base) > 1e-9
                or abs(row['time_s'] - mapped['derivative_decoder_pts'] * time_base) > 1e-9):
            raise ValueError('Mask source/native/decoder linkage differs')
    poses = {row['frame']: row for row in evidence['frames']}
    reference_indices = {int(row['evidence_id'].rsplit(':', 1)[1]): index
                         for index, row in enumerate(human['reference_evidence'])}
    if len(reference_indices) != len(human['reference']):
        raise ValueError('Body reference identity must be one-to-one')
    scale = root['scale']['model_units_per_pixel']
    origin = np.array(root['origin']['pelvis_image_px'])
    anchor = np.array([0., 0., human['mobility']['base_height']])
    floor = human['mobility']['floor_height']
    model = mujoco.MjModel.from_xml_string(xml_text)
    data = mujoco.MjData(model)
    point_rows, frame_rows, snapshots = [], [], {}
    fit_world, fit_pixels = [], []
    for frame in range(count):
        pose, index = poses.get(frame), reference_indices.get(frame)
        mapped = mapping['frames'][frame]
        frame_rows.append({'frame': frame, 'pose_evidence_id': pose['id'] if pose else None,
                           'pose_status': pose['status'] if pose else 'not_sampled', 'body_reference_index': index,
                           'pose_assumed_time_s': pose['time_s'] if pose else None,
                           'pose_decoder_time_s': pose['decoder_timestamp_s'] if pose else None,
                           'original_native_time_s': mapped['source_native_pts'] * time_base,
                           'original_decoder_time_s': mapped['source_decoder_pts'] * time_base,
                           'equipment': {name: {'status': observations[(frame, name)]['mask_status'],
                                               'phase': observations[(frame, name)]['phase'],
                                               'evidence_id': observations[(frame, name)]['id'],
                                               'decoder_time_s': observations[(frame, name)]['time_s']}
                                         for name in args.equipment}})
        if index is None:
            continue
        reference = human['reference_evidence'][index]
        if (reference['evidence_id'] != pose['id'] or reference['time_s'] != pose['time_s']
                or human['reference'][index]['time_s'] != reference['time_s']
                or root['reference'][index]['evidence_id'] != pose['id']
                or human['root_reference'][index]['translation'] != root['reference'][index]['translation']):
            raise ValueError('Body/source reference identity, clock or root translation differs')
        points = snapshots[frame] = pose_points(model, data, human, index)
        for name, world in points.items():
            pixel, confidence, status = image_point(pose, name, width, height)
            point_rows.append({'frame': frame, 'evidence_id': pose['id'], 'joint': name, 'world_model_units': world.tolist(),
                               'image_px': pixel.tolist() if pixel is not None else None,
                               'source_confidence': confidence, 'status': status})
            if status == 'eligible_construction_estimate':
                fit_world.append(world); fit_pixels.append(pixel)
    if len(fit_world) < 32:
        raise ValueError('Insufficient construction correspondences for bounded camera fit')
    parity = []
    for reference_index, frame in zip(saved['source_indices'], saved['frames']):
        source_frame = int(human['reference_evidence'][reference_index]['evidence_id'].rsplit(':', 1)[1])
        parity.extend(np.linalg.norm(snapshots[source_frame][name] - point)
                      for name, point in frame['human']['joints'].items() if name in snapshots[source_frame])
    if max(parity) > 1e-10:
        raise ValueError('Saved MuJoCo model no longer reproduces frozen reference joints')
    fitted, fit = camera_fit(np.array(fit_world), np.array(fit_pixels), origin, anchor, scale)
    cameras = {'existing_root_projection': BASE_ROTATION, 'bounded_fixed_scale_rotation': fitted}
    camera_reports = {}
    for name, rotation in cameras.items():
        eligible, finite, pelvis, low = [], [], [], []
        for row in point_rows:
            projected = project([row['world_model_units']], rotation, origin, anchor, scale)[0]
            error = None if row['image_px'] is None else float(np.linalg.norm(projected - row['image_px']))
            row[name] = {'projected_px': projected.tolist(), 'error_px': error}
            if error is not None:
                finite.append(error)
                if row['status'] == 'eligible_construction_estimate': eligible.append(error)
                else: low.append(error)
                if row['joint'] == 'pelvis': pelvis.append(error)
        camera_reports[name] = {'world_to_camera_rotation': rotation.tolist(), 'pixel_origin': origin.tolist(),
                                'world_anchor_model_units': anchor.tolist(), 'model_units_per_pixel': scale,
                                'eligible_residual_px': statistics(eligible), 'all_finite_residual_px': statistics(finite),
                                'nonfit_finite_residual_px': statistics(low), 'pelvis_residual_px': statistics(pelvis),
                                'calibration': 'conditional_authored_camera_hypothesis_not_measured',
                                'fixed_fields': ['body_geometry', 'source_joint_references', 'root_translation', 'root_origin',
                                                 'torso_scale', 'floor', 'gravity', 'source_ids', 'source_clocks']}
    camera_reports['bounded_fixed_scale_rotation']['fit'] = fit
    common = [frame for frame in reference_indices
              if all(observations[(frame, name)]['mask_status'] == 'decoded' for name in args.equipment)]
    first_common, body_first = min(common), min(reference_indices)
    equipment_rows, equipment_reports, proxies = [], {}, {}
    for camera_name, rotation in cameras.items():
        proxies[camera_name] = {name: proxy_hypothesis(observations[(first_common, name)], rotation, origin, anchor,
                                                     scale, floor, camera_name == 'existing_root_projection')
                                for name in args.equipment}
    for name in args.equipment:
        valid = [observations[(frame, name)] for frame in range(count) if observations[(frame, name)]['mask_status'] == 'decoded']
        bounds = np.array([row['silhouette']['foreground_bounds_px'] for row in valid])
        center = (bounds[:, :2] + bounds[:, 2:]) / 2
        baseline_floor_v = float(project([[0., 0., floor]], BASE_ROTATION, origin, anchor, scale)[0, 1])
        free_bottom_z = anchor[2] - (bounds[:, 3] - origin[1]) * scale
        equipment_reports[name] = {
            'returned_frames': len(valid), 'preinitialization_frames': [frame for frame in range(count) if observations[(frame, name)]['phase'] == 'pre_initialization'],
            'silhouette_box_center_displacement_from_first_px': statistics(np.linalg.norm(center - center[0], axis=1)),
            'silhouette_width_px': statistics(bounds[:, 2] - bounds[:, 0]),
            'silhouette_height_px': statistics(bounds[:, 3] - bounds[:, 1]),
            'baseline_floor_projected_v_px': baseline_floor_v,
            'baseline_floor_minus_silhouette_bottom_px': statistics(baseline_floor_v - bounds[:, 3]),
            'baseline_free_image_fit_box_bottom_z_model_units': statistics(free_bottom_z),
            'identity_status_counts': dict(Counter(row['identity']['status'] for row in valid)),
            'no_measured_3d_stationarity': True,
        }
        for frame in range(count):
            row = observations[(frame, name)]
            summary = {k: row[k] for k in ('frame', 'id', 'object_id', 'time_s', 'original_source', 'phase', 'mask_status', 'identity')}
            summary['silhouette'] = row.get('silhouette')
            if row['mask_status'] == 'decoded':
                b = np.array(row['silhouette']['foreground_bounds_px'])
                summary['camera_hypotheses'] = {}
                for camera_name, proxy in ((k, v[name]) for k, v in proxies.items()):
                    if 'projected_bounds_px' in proxy:
                        error = np.array(proxy['projected_bounds_px']) - b
                        summary['camera_hypotheses'][camera_name] = {'fixed_proxy_bbox_edge_error_px': error.tolist(),
                                                                 'fixed_proxy_bbox_edge_rmse_px': float(np.sqrt(np.mean(error ** 2)))}
            equipment_rows.append(summary)
    overlap = {name: overlap_checks(xml_text, human, values, {body_first: reference_indices[body_first], first_common: reference_indices[first_common]})
               for name, values in proxies.items()}
    for name, checks in overlap.items():
        for object_id, proxy in proxies[name].items():
            proxy['collision_checks'] = [row for row in checks if row['object_id'] == object_id]
            proxy['any_initial_pose_overlap'] = any(row['initial_overlap_rejected'] for row in proxy['collision_checks'])
            proxy['construction_status'] = ('rejected' if proxy['status'].startswith('rejected') or proxy['any_initial_pose_overlap']
                                            else 'authored_proxy_only_not_source_calibrated')
    selected_frames = sorted({first_common, 64, 128, 192, count - 1})
    panels = []
    colors = {'existing_root_projection': '#E97F35', 'bounded_fixed_scale_rotation': '#13B6BC'}
    for frame in selected_frames:
        if frame not in snapshots:
            continue
        preview = equipment / 'masks' / masks['frames'][frame]['preview']
        bound(preview, equipment)
        source = Image.open(preview).convert('RGB')
        if source.size != (width, height):
            raise ValueError('Saved preview dimensions differ from source')
        canvas = Image.new('RGB', (width * 3, height + 96), '#FFFFFF')
        draw = ImageDraw.Draw(canvas)
        for column, label in enumerate(['Saved source preview', *cameras]):
            canvas.paste(source, (column * width, 64))
            draw.text((column * width + 8, 8), f'Frame {frame}: {label}', fill='#172F30')
            if column:
                rotation = cameras[label]
                for row in point_rows:
                    if row['frame'] != frame:
                        continue
                    x, y = row[label]['projected_px']
                    draw.ellipse((column * width + x - 3, 64 + y - 3, column * width + x + 3, 64 + y + 3), fill=colors[label])
                    if row['image_px'] is not None:
                        u, v = row['image_px']
                        draw.line((column * width + x, 64 + y, column * width + u, 64 + v), fill=colors[label], width=1)
                for proxy in proxies[label].values():
                    if 'projected_bounds_px' in proxy:
                        x0, y0, x1, y1 = proxy['projected_bounds_px']
                        draw.rectangle((column * width + x0, 64 + y0, column * width + x1, 64 + y1), outline=colors[label], width=2)
            for object_id in args.equipment:
                b = observations[(frame, object_id)].get('silhouette')
                if b:
                    x0, y0, x1, y1 = b['foreground_bounds_px']
                    draw.rectangle((column * width + x0, 64 + y0, column * width + x1, 64 + y1), outline='#B046BE', width=1)
        draw.text((8, 35), 'Dots: unchanged body projection. Lines: model-image residuals. Magenta: proposed masks. Boxes: authored floor proxies.', fill='#172F30')
        draw.text((8, height + 75), 'Construction estimates only. Fixed scale, body, root and floor. No calibration, measured depth, grasp, support, or physical validation.', fill='#172F30')
        name = f'comparison-{frame:04d}.png'
        canvas.save(out / name)
        panels.append(name)
    write(out / 'input-bindings.json', bindings)
    write(out / 'source-slot-alignment.json', {'frames': frame_rows, 'mapping_sha256': sha(equipment / 'inputs/preparation/frame-mapping-and-pixels.json'),
                                            'rule': 'Exact source frame/pixel identity; native, decoder and assumed pose clocks remain separate'})
    write(out / 'body-reprojection.json', {'camera_hypotheses': camera_reports, 'points': point_rows})
    write(out / 'equipment-reprojection.json', {'objects': equipment_reports, 'observations': equipment_rows})
    write(out / 'proxy-hypotheses.json', {'scope': 'Study hypotheses only; not runtime scene or measured reconstruction',
                                       'body_start_frame': body_first, 'equipment_construction_frame': first_common,
                                       'preinitialization_extrapolation_required': body_first < first_common,
                                       'hypotheses': proxies})
    (out / 'check_equipment_camera_bridge.py').write_text(Path(__file__).read_text())
    report = {'created_at': now(), 'status': 'read_only_feasibility_complete_no_calibration_or_runtime_promotion',
              'contract': CONTRACT, 'source_sha256': human['source_sha256'], 'submitted_source_sha256': mapping['derivative_sha256'],
              'source_frames': count, 'body_source_sampled_frames': len(poses), 'body_reference_poses': len(reference_indices),
              'source_pose_status_counts': dict(Counter(row['pose_status'] for row in frame_rows)),
              'body_reference_point_count': len(point_rows), 'point_status_counts': dict(Counter(row['status'] for row in point_rows)),
              'saved_snapshot_parity': {'joint_comparisons': len(parity), 'maximum_error_model_units': float(max(parity))},
              'camera_hypotheses': camera_reports, 'equipment': equipment_reports,
              'body_first_frame': body_first, 'first_common_frame': first_common,
              'overlap_checks': overlap, 'panels': panels, 'paid_calls': 0, 'new_inference_calls': 0, 'video_decode_calls': 0,
              'cpu_seconds': time.process_time() - cpu_started, 'wall_seconds': time.monotonic() - started,
              'limitations': [
                  'Source pose and mask proposals are construction estimates, not ground truth; a lower fit error does not verify camera accuracy.',
                  'All source slots and low-confidence finite residuals remain; unsupported source poses have no generated body reference.',
                  'Weak-perspective rays leave one depth degree of freedom. Floor support closes it only as an authored prior, often ill-conditioned.',
                  'A solid box includes a kettlebell handle hole and does not model hands/fingers or a measured grasp; it is only a collision proxy.',
                  'No observed camera intrinsics, metric scale, support plane, depth, mass, material, contact, force or physical validation exists.',
                  'A proxy at body frame2 requires explicit static extrapolation across equipment pre-initialization frames; no mask is backfilled.',
                  'Zero external forces and initial velocities are controller/dynamics assumptions, not inferred source forces or motion.',
                  'No body, floor, scale, origin, gravity, source reference or runtime file was changed; no simulated episode was run.',
              ]}
    write(out / 'report.json', report)
    total = sum(path.stat().st_size for path in out.rglob('*') if path.is_file())
    if total > CONTRACT['maximum_output_bytes']:
        raise ValueError('Study output exceeds the declared 64 MB limit')
    for path_string, binding in bindings.items():
        if sha(ROOT / path_string) != binding['sha256']:
            raise ValueError('Source changed during read-only study')
    seal = {'schema': 1, 'created_at': now(), 'files': {path.relative_to(out).as_posix(): sha(path)
                                                     for path in sorted(out.rglob('*')) if path.is_file()}}
    write(out / 'manifest.json', seal)
    signal.alarm(0)
    print(json.dumps({'output': str(out), 'report_sha256': sha(out / 'report.json'),
                      'bytes': sum(path.stat().st_size for path in out.rglob('*') if path.is_file()),
                      'wall_seconds': report['wall_seconds'], 'camera_fit': fit,
                      'camera_eligible_rmse_px': {name: value['eligible_residual_px']['rmse'] for name, value in camera_reports.items()},
                      'initial_overlap_count': {name: sum(row['initial_overlap_rejected'] for row in rows) for name, rows in overlap.items()}}))


if __name__ == '__main__':
    main()
