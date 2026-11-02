"""Audit frozen distal-pixel proposals, native upper bones and every source image slot."""

import argparse
from collections import Counter
import json
from pathlib import Path
import time

import numpy as np

from weft.io import canonical_hash, digest, write_json
from weft.pipeline import verify_artifacts
from weft.spatial_pose import LIMB_ENDPOINTS, rotation_zyx


JOINTS = {11: 'left_shoulder', 12: 'right_shoulder', 13: 'left_elbow', 14: 'right_elbow',
          15: 'left_wrist', 16: 'right_wrist', 23: 'left_hip', 24: 'right_hip',
          25: 'left_knee', 26: 'right_knee', 27: 'left_ankle', 28: 'right_ankle'}
CAMERAS = ['camera.json', 'inputs/two-view-root/camera.json', 'inputs/two-view-root/extra-0/camera.json']
EVIDENCE = ['evidence.json', 'inputs/two-view-root/evidence.json', 'inputs/two-view-root/extra-0/evidence.json']


def read(path):
    return json.loads(path.read_text())


def summary(values):
    return {'count': len(values), 'mean': float(np.mean(values)), 'rmse': float(np.sqrt(np.mean(np.square(values)))),
            'maximum': max(values)} if values else {'count': 0, 'mean': None, 'rmse': None, 'maximum': None}


def project(point, camera, frame, rotation=None):
    transform = camera['frames'][frame]
    q = np.array(transform['R'] if rotation is None else rotation) @ point+transform['t']
    assert .2 <= q[2] <= 100
    xy = q[:2]/q[2]; radial = camera['radial_k1']; r2 = float(xy @ xy)
    assert 1+3*radial*r2 > 0
    return (np.array(camera['K']) @ np.r_[xy*(1+radial*r2), 1.])[:2]


def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists() or args.out.parent.resolve() != Path('artifacts').resolve():
        raise ValueError('Use a new artifact directory')
    started = time.monotonic()
    baseline = Path('artifacts/aspset-partial-limb-body-lab-01/run')
    candidate = Path('artifacts/aspset-distal-pixel-body-lab-01/run')
    verified = [verify_artifacts(p) for p in (baseline, candidate)]
    assert [v['manifest_sha256'] for v in verified] == [
        '0254ca5f0bf4e97b7ec3d220170bfbd13f61c2076a14a55b93a15b2e1fc1b945',
        'da5aca806abd17cf9e8c09cc895e0134151cc4530a900e72ba0f25656cbe0909']
    source_files = ['evidence.json', 'source.json', 'camera.json', 'ground.json', 'camera-reference.json',
                    'two-view-triangulation.json', 'two-view-root.json']
    source_files += [str(p.relative_to(candidate)) for p in (candidate/'inputs').rglob('*') if p.is_file()]
    for name in source_files:
        assert (baseline/name).read_bytes() == (candidate/name).read_bytes(), name
    old, new = [read(p/'human.json') for p in (baseline, candidate)]
    old_audit, audit = [read(p/'two-view-body.json') for p in (baseline, candidate)]
    for key in ('lengths', 'mobility', 'body_frame', 'root_reference', 'motor_limits'):
        assert old[key] == new[key], key
    for key in ('ground', 'root_refinement'):
        assert old['camera_reference'][key] == new['camera_reference'][key]
    assert old_audit['dimensions'] == audit['dimensions'] and old_audit['limb_position_fit'] == audit['limb_position_fit']
    assert audit['partial_pixels']['partial_reference_sha256'] == canonical_hash(old['reference'])
    before, after = [read(p/'human-reference.json')['frames'] for p in (baseline, candidate)]
    assert len(before) == len(after) == len(new['reference']) == 47
    predictions = read(candidate/'two-view-triangulation.json')
    assert len(predictions['frames']) == len(audit['frames']) == 188
    cameras = [read(candidate/name) for name in CAMERAS]
    sources = [{p['frame']: p for p in read(candidate/name)['frames']} for name in EVIDENCE]
    offset = np.r_[new['camera_reference']['origin_world_m'][:2], new['camera_reference']['ground']['height_m']]
    rows = []; complete = trials = 0; previous = {}
    for i, (a, b, meta) in enumerate(zip(before, after, new['reference_evidence'], strict=True)):
        frame = int(meta['evidence_id'].rsplit(':', 1)[1]); stamp = frame/50
        assert a['source_time_s'] == b['source_time_s'] == meta['time_s'] == stamp
        assert meta['geometry_fit'] == old['reference_evidence'][i]['geometry_fit']
        changed = set()
        torso = np.array(old['body_frame']['rotation']) @ rotation_zyx([
            old['reference'][i]['joints'][k] for k in ('torso_twist', 'torso_lateral', 'torso_flex')])
        for group, indices in LIMB_ENDPOINTS.items():
            detail, former = [v['frames'][frame]['groups'][group] for v in (audit, old_audit)]
            attempt = detail.get('distal_pixel_attempt')
            if attempt is None or not attempt['applied']:
                previous.pop(group, None)
                if 'length_fit' in detail:
                    assert detail['length_fit'] == former['length_fit']; complete += 1
                continue
            fit = attempt['fit']; names = (group+'_twist', group+'_lateral', group, JOINTS[indices[1]])
            changed.update(names); distal = JOINTS[indices[-1]]
            assert former['status'] == 'multi_view_proximal_monocular_distal_and_twist'
            assert detail['status'] == 'multi_view_proximal_image_constrained_distal'
            assert not fit['depth_measured'] and not fit['observations_modified']
            assert all(meta['joint_confidence'][name] == 0. for name in names)
            for name in (group, names[-1]):
                np.testing.assert_allclose(a['human']['joints'][name], b['human']['joints'][name], atol=1e-12)
            origin = np.array(a['human']['joints'][names[-1]])+offset
            old_angles = [old['reference'][i]['joints'][name] for name in names]
            rotation = torso @ rotation_zyx(old_angles[:3]); length = new['lengths']['shin' if group.endswith('hip') else 'forearm']
            sign = -1 if group.endswith('hip') else 1
            np.testing.assert_allclose(fit['input_rotation_world'], rotation, atol=1e-12)
            np.testing.assert_allclose(fit['fixed_knee_world_m'], origin, atol=1e-12)
            assert fit['input_bend_rad'] == old_angles[3] and fit['fixed_distal_length_m'] == length and fit['bend_sign'] == sign
            assert attempt['input_angles_rad'] == old_angles
            assert attempt['observation_statuses'] == predictions['frames'][frame]['joints'][indices[-1]]['observation_statuses']
            available = [v for v, status in enumerate(attempt['observation_statuses']) if status == 'available']
            assert available == attempt['view_indices'] == meta['geometry_partial_pixels'][group]['view_indices']
            assert available == [v['view_index'] for v in attempt['views']]
            for j, item in enumerate(attempt['views']):
                v = item['view_index']; source = sources[v][frame]; point = source['landmarks'][indices[-1]]
                assert item['source_evidence_id'] == source['id'] and item['camera_sha256'] == canonical_hash(cameras[v])
                assert item['pixel'] == [point['x']*cameras[v]['width'], point['y']*cameras[v]['height']]
                assert item['confidence'] == min(point['visibility'], point['presence']) == fit['source_confidence'][j]
                assert .5 <= item['confidence'] <= 1 and fit['source_pixels'][j] == item['pixel']
                original_rotation = np.array(cameras[v]['frames'][frame]['R']); u, _, vt = np.linalg.svd(original_rotation)
                np.testing.assert_allclose(item['solver_rotation'], u @ vt, atol=1e-12)
                assert np.max(abs(np.array(item['solver_rotation'])-original_rotation)) == item['maximum_rotation_element_change'] <= 1e-5
            prior = previous[group][1] if group in previous and 0 < stamp-previous[group][0] <= .25 else None
            assert fit['previous_correction_rad'] == prior
            previous[group] = (stamp, fit['correction_rad'])

            def evaluate(delta):
                point = origin+rotation @ rotation_zyx([delta[0], 0., 0.]) @ rotation_zyx([0., 0., sign*(old_angles[3]+delta[1])]) @ [0., 0., -length]
                errors = [float(np.linalg.norm(project(point, cameras[v['view_index']], frame, v['solver_rotation'])-v['pixel'])) for v in attempt['views']]
                sse = sum(v['confidence']*(error/max(cameras[v['view_index']]['width'], cameras[v['view_index']]['height']))**2
                          for v, error in zip(attempt['views'], errors, strict=True))
                cost = .5*sse+.5*1e-3*float(delta @ delta)
                if prior is not None: cost += .5*5e-4*float((delta-prior) @ (delta-prior))
                return point, errors, sse, cost

            current = np.zeros(2)
            for iteration in fit['iterations']:
                assert 1 <= iteration['iteration'] <= 12 and len(iteration['trials']) <= 4
                assert abs(evaluate(current)[3]-iteration['objective_before']) < 1e-12
                for trial in iteration['trials']:
                    delta = np.array(trial['correction_rad']); trials += 1
                    assert np.max(abs(delta)) <= .30 and np.max(abs(delta-current)) <= .10+1e-12
                    assert 0 <= old_angles[3]+delta[1] <= 2.6
                    try: _, _, sse, cost = evaluate(delta)
                    except AssertionError:
                        assert trial['rejection'] and not trial['accepted']; continue
                    assert trial['rejection'] is None
                    assert abs(sse-trial['pixel_sse']) < 1e-12 and abs(cost-trial['objective']) < 1e-12
                    if trial['accepted']:
                        assert cost < iteration['objective_before']; current = delta
            zero = evaluate(np.zeros(2)); final = evaluate(np.array(fit['correction_rad']))
            for label, evaluated, snapshot in [('before', zero, a), ('after', final, b)]:
                np.testing.assert_allclose(evaluated[0], np.array(snapshot['human']['joints'][distal])+offset, atol=1e-12)
                np.testing.assert_allclose(evaluated[1], fit['pixel_errors_'+label], atol=1e-9)
                assert abs(evaluated[2]-fit['pixel_sse_'+label]) < 1e-12 and abs(evaluated[3]-fit['objective_'+label]) < 1e-12
            assert final[2] <= zero[2]+1e-15
            rows.append({'frame': frame, 'group': group, 'views': available, 'status': fit['status'], 'correction_rad': fit['correction_rad'],
                'pixel_sse_before': zero[2], 'pixel_sse_after': final[2], 'pixel_errors_before': zero[1], 'pixel_errors_after': final[1]})
        assert {k: v for k, v in old['reference'][i]['joints'].items() if k not in changed} == {
            k: v for k, v in new['reference'][i]['joints'].items() if k not in changed}
    assert len(rows) == audit['partial_pixels']['fitted_group_poses']
    images = []; image_summary = {}; by_frame = {round(v['source_time_s']*50): i for i, v in enumerate(before)}
    for view, camera, poses in zip(('left', 'mid', 'right'), cameras, sources, strict=True):
        known = []
        for frame in range(188):
            for joint, name in JOINTS.items():
                row = {'frame': frame, 'view': view, 'joint': name, 'status': 'source_pose_unsampled', 'before_px': None, 'after_px': None}
                images.append(row); pose = poses.get(frame)
                if pose is None: continue
                point = pose['landmarks'][joint]; row['status'] = 'low_confidence_or_outside'
                if not 0 <= point['x'] <= 1 or not 0 <= point['y'] <= 1 or min(point['visibility'], point['presence']) < .5: continue
                row['status'] = 'body_reference_unavailable'
                if frame not in by_frame: continue
                index = by_frame[frame]; target = np.array([point['x']*camera['width'], point['y']*camera['height']])
                for label, snapshots in [('before_px', before), ('after_px', after)]:
                    world = np.array(snapshots[index]['human']['joints'][name])+offset
                    row[label] = float(np.linalg.norm(project(world, camera, frame)-target))
                row['status'] = 'compared'; known.append(row)
        image_summary[view] = {'before': summary([r['before_px'] for r in known]), 'after': summary([r['after_px'] for r in known]),
            'regressed_points': sum(r['after_px'] > r['before_px']+1e-8 for r in known)}
    floors = [read(p/'reference-feasibility.json') for p in (baseline, candidate)]
    assert floors[0]['contract'] == floors[1]['contract']
    for a, b in zip(floors[0]['checks'], floors[1]['checks'], strict=True):
        for key in ('kind', 'time_s', 'source_evidence_ids', 'bracketing_reference_indices'): assert a[key] == b[key]
    for floor in floors:
        assert len(floor['checks']) == 158
        assert all(r['passes_floor_limit'] == (r['max_penetration_model_units'] <= .025) for r in floor['checks'])
        assert sum(not r['passes_floor_limit'] for r in floor['checks'] if r['kind'] == 'source_pose') == floor['summary']['failed_source_poses']
        assert sum(not r['passes_floor_limit'] for r in floor['checks'] if r['kind'] == 'control_grid') == floor['summary']['failed_control_grid_poses']
    report = {'status': 'saved_distal_pixels_verified', 'verifications': verified, 'source_files_equal': len(source_files),
        'pixel_fits_checked': len(rows), 'retained_trial_losses_checked': trials, 'unchanged_complete_limb_fits': complete,
        'pixel_sse_before': sum(r['pixel_sse_before'] for r in rows), 'pixel_sse_after': sum(r['pixel_sse_after'] for r in rows),
        'full_source_image_slots': len(images), 'image_status_counts': dict(Counter(r['status'] for r in images)), 'image_comparison': image_summary,
        'floor_before': floors[0]['summary'], 'floor_after': floors[1]['summary'], 'failed_checks': [r for r in floors[1]['checks'] if not r['passes_floor_limit']],
        'root_floor_shape_forces_unchanged': True, 'source_observations_unchanged': True, 'source_candidate_promoted': False,
        'physical_validation': 'unverified', 'scope': 'Construction pixel fit and fixed sampled-floor consistency; not recovered distal depth or physical accuracy',
        'new_inference_solve_physics_api_calls': 0, 'seconds': time.monotonic()-started}
    args.out.mkdir()
    for name, value in [('report.json', report), ('pixel-fits.json', rows), ('source-images.json', images)]: write_json(args.out/name, value)
    (args.out/'worker.py').write_bytes(Path(__file__).read_bytes())
    write_json(args.out/'manifest.json', {'schema': 1, 'files': {p.name: digest(p) for p in sorted(args.out.iterdir())}})
    print(json.dumps({'verification': verify_artifacts(args.out), 'pixel_fits': len(rows), 'trials': trials, 'complete_fits': complete,
        'floor_before': report['floor_before'], 'floor_after': report['floor_after'], 'image_comparison': image_summary, 'seconds': report['seconds']}))


if __name__ == '__main__':
    main()
