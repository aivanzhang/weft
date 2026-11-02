"""Audit every saved joint-trajectory trial, original image slot and unchanged floor gate."""

import argparse
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
import time

import numpy as np

from verify_aspset_depth_branches import CAMERAS, EVIDENCE, JOINTS, audit_images, project, read, summary, verify_artifacts
from weft.io import canonical_hash, digest, write_json
from weft.spatial_pose import LIMB_ENDPOINTS, rotation_zyx


BASE = Path('artifacts/aspset-partial-limb-body-lab-01/run')
PREVIOUS = Path('artifacts/aspset-distal-uncertainty-body-lab-01/run')
CURRENT = Path('artifacts/aspset-joint-pixel-body-lab-01/run')
SEALS = ('0254ca5f0bf4e97b7ec3d220170bfbd13f61c2076a14a55b93a15b2e1fc1b945',
         '672234299b8645616a8fb2b4787591774aa4bea9e9623d3af7a6915bc5bdf449',
         '64710fc56cd47a02253b397a3840e5ded177b95e71c3f4e5e4f9e392d9cb8674')


def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists() or args.out.parent.resolve() != Path('artifacts').resolve():
        raise ValueError('Use a new artifact directory')
    started = time.monotonic(); verified = [verify_artifacts(p) for p in (BASE, PREVIOUS, CURRENT)]
    assert tuple(row['manifest_sha256'] for row in verified) == SEALS
    source_files = ['evidence.json', 'source.json', 'camera.json', 'ground.json', 'camera-reference.json',
                    'two-view-triangulation.json', 'two-view-root.json']
    source_files += [str(p.relative_to(CURRENT)) for p in (CURRENT/'inputs').rglob('*') if p.is_file()]
    for name in source_files: assert (BASE/name).read_bytes() == (PREVIOUS/name).read_bytes() == (CURRENT/name).read_bytes(), name
    old, body = [read(p/'human.json') for p in (BASE, CURRENT)]
    original, audit = [read(p/'two-view-body.json') for p in (BASE, CURRENT)]
    native_before, native_previous, native_after = [read(p/'human-reference.json')['frames'] for p in (BASE, PREVIOUS, CURRENT)]
    assert len(native_before) == len(native_previous) == len(native_after) == len(body['reference']) == 47
    assert len(audit['frames']) == len(original['frames']) == 188
    for key in ('lengths', 'body_frame', 'root_reference', 'mobility', 'motor_limits', 'dt'):
        assert old[key] == body[key], key
    for key in ('ground', 'root_refinement'): assert old['camera_reference'][key] == body['camera_reference'][key]
    for key in ('dimensions', 'limb_position_fit', 'partial_segments', 'group_coverage'):
        assert audit[key] == original[key], key
    pin = body['camera_reference']['body_refinement']['joint_pixel_fit']; settings = read(CURRENT/'joint-pixel-uncertainty.json')
    assert settings == pin['uncertainty'] == audit['joint_pixels']['uncertainty']
    assert [settings[k] for k in ('image_std_fraction', 'angle_prior_std_rad', 'correction_speed_std_rad_s')] == [.0025, .3, 4.5]
    assert pin['input_reference_sha256'] == canonical_hash(old['reference'])
    assert pin['input_evidence_sha256'] == canonical_hash(old['reference_evidence'])
    assert pin['input_audit_sha256'] == canonical_hash(original)
    assert canonical_hash(audit) == body['camera_reference']['body_refinement']['audit_sha256']
    restored_reference, restored_meta = deepcopy(body['reference']), deepcopy(body['reference_evidence'])
    cameras = [read(CURRENT/name) for name in CAMERAS]
    sources = [{p['frame']: p for p in read(CURRENT/name)['frames']} for name in EVIDENCE]
    predictions = read(CURRENT/'two-view-triangulation.json')
    frames = [int(meta['evidence_id'].rsplit(':', 1)[1]) for meta in old['reference_evidence']]
    times = np.array([meta['time_s'] for meta in old['reference_evidence']]); gaps = np.diff(times)
    assert np.allclose(times, np.array(frames)/50, atol=0, rtol=0)
    parents = [np.array(old['body_frame']['rotation']) @ rotation_zyx([r['joints'][k] for k in ('torso_twist', 'torso_lateral', 'torso_flex')]) for r in old['reference']]
    roots = np.array(old['camera_reference']['origin_world_m'])+np.array([r['translation'] for r in old['root_reference']])
    offset = np.r_[body['camera_reference']['origin_world_m'][:2], body['camera_reference']['ground']['height_m']]
    rows = []; trial_count = constraints = complete = weak = changed = 0
    for group, indices in LIMB_ENDPOINTS.items():
        record = audit['joint_pixel_trajectories'][group]; assert record['status'] == 'fitted'
        fit = record['fit']; assert fit['method'] == 'time-coupled-joint-pixels-v1'
        assert fit['depth_measured'] is fit['observations_modified'] is fit['uncertainty_calibrated'] is False
        assert fit['uncertainty'] == settings
        names = (group+'_twist', group+'_lateral', group, JOINTS[indices[1]])
        angles = np.array([[r['joints'][name] for name in names] for r in old['reference']])
        leg = group.endswith('hip'); side = -1 if group.startswith('left_') else 1; sign = -1 if leg else 1
        size = [old['lengths'][k] for k in (('thigh', 'shin') if leg else ('upper_arm', 'forearm'))]
        origins = np.array([root+parent @ [side*old['lengths']['hip_width' if leg else 'shoulder_width']/2, 0., 0. if leg else old['lengths']['torso']]
                            for root, parent in zip(roots, parents, strict=True)])
        np.testing.assert_allclose(fit['fixed_origins_world_m'], origins, atol=1e-12, rtol=0)
        np.testing.assert_allclose(fit['fixed_parent_rotations'], parents, atol=1e-12, rtol=0)
        assert fit['fixed_lengths_m'] == size and fit['source_times_s'] == times.tolist()
        np.testing.assert_array_equal(fit['input_angles_rad'], angles)
        observations = []
        for i, frame in enumerate(frames):
            detail = audit['frames'][frame]['groups'][group]; before = original['frames'][frame]['groups'][group]
            assert {k: v for k, v in detail.items() if k != 'joint_pixel_attempt'} == before
            complete += 'length_fit' in before
            attempt = detail['joint_pixel_attempt']; expected_slots = []
            for source_index in indices[1:]:
                statuses = predictions['frames'][frame]['joints'][source_index]['observation_statuses']
                assert attempt['observation_statuses'][str(source_index)] == statuses
                available = [v for v, status in enumerate(statuses) if status == 'available']
                assert attempt['view_indices'][str(source_index)] == available
                expected_slots.extend((source_index, v) for v in available)
            assert [(v['source_landmark_index'], v['view_index']) for v in attempt['views']] == expected_slots
            for view in attempt['views']:
                v = view['view_index']; pose = sources[v][frame]; source_index = view['source_landmark_index']; point = pose['landmarks'][source_index]
                assert view['joint_index'] == list(indices[1:]).index(source_index)
                assert view['source_evidence_id'] == pose['id'] and view['camera_sha256'] == canonical_hash(cameras[v])
                assert view['pixel'] == [point['x']*cameras[v]['width'], point['y']*cameras[v]['height']]
                assert view['confidence'] == min(point['visibility'], point['presence']) >= .5
                rotation = np.array(cameras[v]['frames'][frame]['R']); u, _, vt = np.linalg.svd(rotation)
                np.testing.assert_allclose(view['solver_rotation'], u @ vt, atol=1e-12, rtol=0)
                assert view['maximum_rotation_element_change'] == float(np.max(abs(u @ vt-rotation))) <= 1e-5
            observations.append(attempt['views']); constraints += len(attempt['views'])
        active = np.array([bool(v) for v in observations]); edges = np.where(active[:-1] & active[1:] & (gaps <= .25), 1/(4.5*gaps)**2, 0.)
        assert fit['active_frames'] == active.tolist()
        np.testing.assert_array_equal(fit['temporal_edge_weights'], edges)

        def frame_score(i, delta):
            current = angles[i]+delta; rotation = parents[i] @ rotation_zyx(current[:3]); bend = current[3]
            middle = origins[i]+rotation @ [0., 0., -size[0]]
            distal = middle+rotation @ [0., sign*size[1]*np.sin(bend), -size[1]*np.cos(bend)]
            points = np.array([middle, distal]); errors = []; residuals = []
            for view in observations[i]:
                camera = cameras[view['view_index']]
                difference = project(points[view['joint_index']], camera, frames[i], view['solver_rotation'])-view['pixel']
                errors.append(float(np.linalg.norm(difference)))
                residuals.extend(np.sqrt(view['confidence'])*difference/(.0025*max(camera['width'], camera['height'])))
            return points, errors, np.array(residuals)

        def loss(delta):
            data = sum(float((residual := frame_score(i, d)[2]) @ residual) for i, d in enumerate(delta))
            prior = float(np.sum(delta**2))/.3**2
            temporal = float(np.sum(edges[:, None]*np.diff(delta, axis=0)**2))
            return {'objective': .5*(data+prior+temporal), 'image_chi2': data,
                    'angle_prior_chi2': prior, 'correction_speed_chi2': temporal}

        for i in range(47):
            derivative = np.column_stack([(frame_score(i, e)[2]-frame_score(i, -e)[2])/2e-6 for e in np.eye(4)*1e-6])
            eigenvalues = np.maximum(np.linalg.eigvalsh(derivative.T @ derivative), 0.)*.3**2
            np.testing.assert_allclose(eigenvalues, fit['initial_image_to_prior_eigenvalues'][i], atol=1e-6, rtol=1e-5)
            assert fit['weak_image_directions'][i] == np.count_nonzero(eigenvalues <= 1.)
            weak += bool(active[i] and np.any(eigenvalues <= 1.))
        current = np.zeros_like(angles)
        for iteration in fit['iterations']:
            assert 1 <= iteration['iteration'] <= 12 and len(iteration['trials']) <= 4
            assert abs(loss(current)['objective']-iteration['objective_before']) < 1e-8
            for trial in iteration['trials']:
                trial_count += 1; delta = np.array(trial['correction_rad'])
                assert delta.shape == angles.shape and np.max(abs(delta)) <= .30 and np.max(abs(delta-current)) <= .10+1e-12
                assert np.all((angles+delta)[:, 3] >= 0) and np.all((angles+delta)[:, 3] <= 2.6)
                assert np.all(delta[~active] == 0.)
                try: values = loss(delta)
                except AssertionError:
                    assert trial['rejection'] == 'camera_projection_domain' and not trial['accepted']; continue
                assert trial['rejection'] is None
                for key, value in values.items(): assert abs(value-trial[key]) < 1e-8, (group, key)
                if trial['accepted']:
                    assert trial['objective'] < iteration['objective_before']-1e-12; current = delta
        np.testing.assert_array_equal(current, fit['correction_rad'])
        np.testing.assert_allclose(angles+current, fit['output_angles_rad'], atol=1e-12, rtol=0)
        assert fit['maximum_angle_change_rad'] == np.max(abs(current)) <= .3
        for label, delta, native in [('before', np.zeros_like(angles), native_before), ('after', current, native_after)]:
            for key, value in loss(delta).items(): assert abs(value-fit[label][key]) < 1e-8, (group, label, key)
            for i, d in enumerate(delta):
                points, errors, _ = frame_score(i, d)
                np.testing.assert_allclose(points, fit[label]['points_world_m'][i], atol=1e-12, rtol=0)
                np.testing.assert_allclose(errors, fit[label]['errors_px'][i], atol=1e-9, rtol=0)
                np.testing.assert_allclose(points, [np.array(native[i]['human']['joints'][JOINTS[j]])+offset for j in indices[1:]], atol=1e-10, rtol=0)
        for i, (meta, reference) in enumerate(zip(body['reference_evidence'], body['reference'], strict=True)):
            record = meta['geometry_joint_pixels'][group]; altered = np.any(current[i] != 0); changed += bool(altered)
            assert record['input_angles_rad'] == angles[i].tolist() and record['correction_rad'] == current[i].tolist()
            for k, name in enumerate(names):
                assert abs(reference['joints'][name]-angles[i, k]-current[i, k]) < 1e-12
                assert record['input_joint_confidence'][k] == old['reference_evidence'][i]['joint_confidence'][name]
                assert meta['joint_confidence'][name] == (0. if altered else record['input_joint_confidence'][k])
                restored_reference[i]['joints'][name] = angles[i, k].item()
                restored_meta[i]['joint_confidence'][name] = record['input_joint_confidence'][k]
        rows.append({'group': group, 'status': fit['status'], 'before': loss(np.zeros_like(angles)), 'after': loss(current),
                     'iterations': len(fit['iterations']), 'maximum_angle_change_rad': fit['maximum_angle_change_rad']})
    for meta in restored_meta:
        meta.pop('geometry_joint_pixels'); meta['geometry_method'] = 'camera-body-length-fit-v2'
    assert restored_reference == old['reference'] and restored_meta == old['reference_evidence']
    joint_summary = audit['joint_pixels']
    assert joint_summary['changed_group_poses'] == changed and joint_summary['source_image_constraints'] == constraints
    assert joint_summary['weak_image_group_poses'] == weak
    for label in ('before', 'after'): assert abs(joint_summary['image_chi2_'+label]-sum(r[label]['image_chi2'] for r in rows)) < 1e-8
    images, image_summary = audit_images(cameras, sources, native_before, native_after, offset)
    by_frame = {frame: i for i, frame in enumerate(frames)}; view_names = ('left', 'mid', 'right'); joint_indices = {v: k for k, v in JOINTS.items()}
    for row in images:
        row['v2_before_px'] = None
        if row['status'] != 'compared': continue
        view = view_names.index(row['view']); frame = row['frame']; camera = cameras[view]
        point = sources[view][frame]['landmarks'][joint_indices[row['joint']]]
        target = [point['x']*camera['width'], point['y']*camera['height']]
        world = np.array(native_previous[by_frame[frame]]['human']['joints'][row['joint']])+offset
        row['v2_before_px'] = float(np.linalg.norm(project(world, camera, frame)-target))
    for view in view_names:
        known = [r for r in images if r['view'] == view and r['status'] == 'compared']
        image_summary[view]['v2_before'] = summary([r['v2_before_px'] for r in known])
        image_summary[view]['regressed_from_v2'] = sum(r['after_px'] > r['v2_before_px']+1e-8 for r in known)
    floors = [read(p/'reference-feasibility.json') for p in (BASE, PREVIOUS, CURRENT)]
    assert floors[0]['contract'] == floors[1]['contract'] == floors[2]['contract']
    for checks in zip(*(f['checks'] for f in floors), strict=True):
        for key in ('kind', 'time_s', 'source_evidence_ids', 'bracketing_reference_indices'): assert checks[0][key] == checks[1][key] == checks[2][key]
    for floor in floors:
        assert len(floor['checks']) == 158
        assert all(r['passes_floor_limit'] == (r['max_penetration_model_units'] <= .025) for r in floor['checks'])
        for kind, key in [('source_pose', 'failed_source_poses'), ('control_grid', 'failed_control_grid_poses')]:
            assert sum(not r['passes_floor_limit'] for r in floor['checks'] if r['kind'] == kind) == floor['summary'][key]
    report = {'status': 'saved_joint_trajectories_verified', 'verifications': verified, 'source_files_equal': len(source_files),
        'trajectories': rows, 'retained_trial_losses_checked': trial_count, 'unchanged_initial_complete_limb_fits': complete,
        'joint_pixels': joint_summary, 'full_source_image_slots': len(images),
        'image_status_counts': dict(Counter(r['status'] for r in images)), 'image_comparison': image_summary,
        'floor_partial': floors[0]['summary'], 'floor_v2': floors[1]['summary'], 'floor_after': floors[2]['summary'],
        'remaining_positive_penetration_checks': sum(r['max_penetration_model_units'] > 0 for r in floors[2]['checks']),
        'failed_checks': [r for r in floors[2]['checks'] if not r['passes_floor_limit']],
        'root_floor_shape_forces_unchanged': True, 'source_observations_unchanged': True, 'source_candidate_promoted': False,
        'physical_validation': 'unverified', 'scope': 'Construction image fit and sampled-floor tolerance, not physical depth/contact accuracy or learning',
        'new_optimizer_inference_native_physics_api_calls': 0, 'seconds': time.monotonic()-started}
    args.out.mkdir()
    for name, value in [('report.json', report), ('source-images.json', images)]: write_json(args.out/name, value)
    (args.out/'worker.py').write_bytes(Path(__file__).read_bytes())
    helper = Path(__file__).with_name('verify_aspset_depth_branches.py'); (args.out/helper.name).write_bytes(helper.read_bytes())
    write_json(args.out/'manifest.json', {'schema': 1, 'files': {p.name: digest(p) for p in sorted(args.out.iterdir())}})
    print(json.dumps({'verification': verify_artifacts(args.out), 'trial_losses': trial_count, 'floor_after': floors[2]['summary'], 'image_comparison': image_summary, 'seconds': report['seconds']}))


if __name__ == '__main__':
    main()
