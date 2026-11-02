"""Verify frozen fixed-length proposals and compare every original image slot; no solve or physics."""

import argparse
from collections import Counter
import json
from pathlib import Path
import time

import numpy as np

from weft.io import digest, write_json
from weft.pipeline import verify_artifacts
from weft.spatial_pose import LIMB_ENDPOINTS, rotation_zyx


JOINTS = {11: 'left_shoulder', 12: 'right_shoulder', 13: 'left_elbow', 14: 'right_elbow',
          15: 'left_wrist', 16: 'right_wrist', 23: 'left_hip', 24: 'right_hip',
          25: 'left_knee', 26: 'right_knee', 27: 'left_ankle', 28: 'right_ankle'}


def read(path):
    return json.loads(path.read_text())


def summary(values):
    array = np.array(values)
    return {'count': len(values), 'mean': float(array.mean()), 'rmse': float(np.sqrt(np.mean(array**2))),
            'maximum': float(array.max())} if len(values) else {'count': 0, 'mean': None, 'rmse': None, 'maximum': None}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists() or args.out.parent.resolve() != Path('artifacts').resolve():
        raise ValueError('Use a new artifact directory')
    start = time.monotonic()
    baseline = Path('artifacts/aspset-third-camera-body-lab-01/run')
    candidate = Path('artifacts/aspset-fixed-limb-body-lab-01/run')
    verified = [verify_artifacts(p) for p in (baseline, candidate)]
    assert [v['manifest_sha256'] for v in verified] == ['fe2b84b7b8f7c26ab1f912f87962a2e7ef13eb58d88bcc035911514e98365857',
        'e3988f89a7638a3a34f87fc71ab97f5f284b145bdb1bfa16081470c13b8aa3f4']
    for name in ('evidence.json', 'source.json', 'camera.json', 'ground.json', 'camera-reference.json', 'two-view-triangulation.json', 'two-view-root.json'):
        assert (baseline/name).read_bytes() == (candidate/name).read_bytes(), name
    old, new = [read(p/'human.json') for p in (baseline, candidate)]
    for key in ('lengths', 'mobility', 'body_frame', 'root_reference', 'motor_limits'):
        assert old[key] == new[key], key
    assert old['camera_reference']['ground'] == new['camera_reference']['ground']
    assert old['camera_reference']['root_refinement'] == new['camera_reference']['root_refinement']
    before, after = [read(p/'human-reference.json')['frames'] for p in (baseline, candidate)]
    assert len(before) == len(after) == len(new['reference']) == 47
    pred = read(candidate/'two-view-triangulation.json')
    audit = read(candidate/'two-view-body.json'); fit_contract = audit['contract']['limb_fit']
    assert fit_contract['maximum_angle_change_rad'] == .3 and fit_contract['maximum_update_rad'] == .1
    offset = np.r_[new['camera_reference']['origin_world_m'][:2], new['camera_reference']['ground']['height_m']]
    limb_rows = []; trials = 0; last = {}
    for index, (a, b, meta) in enumerate(zip(before, after, new['reference_evidence'], strict=True)):
        frame = int(meta['evidence_id'].rsplit(':', 1)[1]); stamp = frame/50
        assert a['source_time_s'] == b['source_time_s'] == meta['time_s'] == stamp
        torso = np.array(new['body_frame']['rotation']) @ rotation_zyx([new['reference'][index]['joints'][k] for k in ('torso_twist', 'torso_lateral', 'torso_flex')])
        for group, indices in LIMB_ENDPOINTS.items():
            detail = audit['frames'][frame]['groups'][group]; fit = detail.get('length_fit')
            if fit is None:
                last.pop(group, None); continue
            sign = -1 if group.endswith('hip') else 1
            names = [JOINTS[j] for j in indices]
            points = np.array([pred['frames'][frame]['joints'][j]['world_point_m'] for j in indices[1:]])
            original = np.array([a['human']['joints'][name] for name in names[1:]])+offset
            proposed = np.array([b['human']['joints'][name] for name in names[1:]])+offset
            origin = np.array(a['human']['joints'][group])+offset
            assert np.allclose(np.array(fit['targets_parent_m']) @ torso.T+origin, points, atol=1e-12)
            assert np.allclose(np.array(fit['before_parent_m']) @ torso.T+origin, original, atol=1e-12)
            assert np.allclose(np.array(fit['after_parent_m']) @ torso.T+origin, proposed, atol=1e-12)
            angle_names = (group+'_twist', group+'_lateral', group, names[1])
            assert np.allclose(fit['input_angles_rad'], [old['reference'][index]['joints'][k] for k in angle_names], atol=1e-12)
            assert fit['output_angles_rad'] == [new['reference'][index]['joints'][k] for k in angle_names]
            expected_previous = last[group][1] if group in last and stamp-last[group][0] <= .25 else None
            assert fit['previous_correction_rad'] == expected_previous
            last[group] = (stamp, fit['angle_correction_rad'])
            previous = np.zeros(4) if expected_previous is None else np.array(expected_previous)
            current = np.zeros(4)
            for iteration in fit['iterations']:
                assert 1 <= iteration['iteration'] <= 12 and len(iteration['trials']) <= 4
                for trial in iteration['trials']:
                    change = np.array(trial['angle_correction_rad'])
                    assert np.max(abs(change)) <= .3 and np.max(abs(change-current)) <= .1+1e-12
                    angles = np.array(fit['input_angles_rad'])+change
                    first_length, second_length = fit['fixed_lengths_m']
                    rotation = rotation_zyx(angles[:3])
                    knee = rotation @ [0., 0., -first_length]
                    ankle = knee+rotation @ rotation_zyx([0., 0., sign*angles[3]]) @ [0., 0., -second_length]
                    position = float(np.sum((np.array([knee, ankle])-fit['targets_parent_m'])**2))
                    objective = .5*position+.5*1e-3*float(change @ change)
                    if expected_previous is not None: objective += .5*5e-4*float((change-previous) @ (change-previous))
                    assert abs(position-trial['position_sse_m2']) < 1e-12 and abs(objective-trial['objective']) < 1e-12
                    if trial['accepted']:
                        assert objective < iteration['objective_before']
                        current = change
                    trials += 1
            old_sse, new_sse = [float(np.sum((value-points)**2)) for value in (original, proposed)]
            assert abs(old_sse-fit['position_sse_before_m2']) < 1e-12
            assert abs(new_sse-fit['position_sse_after_m2']) < 1e-12 and new_sse <= old_sse+1e-12
            limb_rows.append({'frame': frame, 'group': group, 'before_sse_m2': old_sse, 'after_sse_m2': new_sse,
                              'status': fit['status'], 'maximum_angle_change_rad': fit['maximum_angle_change_rad']})
    images = []; by_frame = {round(v['source_time_s']*50): i for i, v in enumerate(before)}
    camera_names = ['camera.json', 'inputs/two-view-root/camera.json', 'inputs/two-view-root/extra-0/camera.json']
    evidence_names = ['evidence.json', 'inputs/two-view-root/evidence.json', 'inputs/two-view-root/extra-0/evidence.json']
    image_summary = {}
    for view, camera_name, evidence_name in zip(('left', 'mid', 'right'), camera_names, evidence_names, strict=True):
        camera = read(candidate/camera_name); poses = {p['frame']: p for p in read(candidate/evidence_name)['frames']}
        K = np.array(camera['K']); known = []
        for frame in range(188):
            for joint, name in JOINTS.items():
                row = {'frame': frame, 'view': view, 'joint': name, 'status': 'source_pose_unsampled', 'before_px': None, 'after_px': None}
                images.append(row); pose = poses.get(frame)
                if pose is None: continue
                point = pose['landmarks'][joint]
                row['status'] = 'low_confidence_or_outside'
                if not 0 <= point['x'] <= 1 or not 0 <= point['y'] <= 1 or min(point['visibility'], point['presence']) < .5: continue
                row['status'] = 'body_reference_unavailable'
                if frame not in by_frame: continue
                index = by_frame[frame]; target = np.array([point['x']*3840, point['y']*2160])
                for label, snapshots in [('before_px', before), ('after_px', after)]:
                    world = np.array(snapshots[index]['human']['joints'][name])+offset
                    cam = np.array(camera['frames'][frame]['R']) @ world+camera['frames'][frame]['t']
                    assert .2 <= cam[2] <= 100 and camera['radial_k1'] == 0
                    pixel = K @ cam
                    row[label] = float(np.linalg.norm(pixel[:2]/pixel[2]-target))
                row['status'] = 'compared'; known.append(row)
        image_summary[view] = {'before': summary([r['before_px'] for r in known]), 'after': summary([r['after_px'] for r in known]),
                               'regressed_points': sum(r['after_px'] > r['before_px']+1e-8 for r in known)}
    floors = [read(p/'reference-feasibility.json') for p in (baseline, candidate)]
    assert floors[0]['contract'] == floors[1]['contract']
    for floor in floors:
        assert len(floor['checks']) == 158
        assert all(r['passes_floor_limit'] == (r['max_penetration_model_units'] <= .025) for r in floor['checks'])
        assert sum(not r['passes_floor_limit'] for r in floor['checks'] if r['kind'] == 'source_pose') == floor['summary']['failed_source_poses']
        assert sum(not r['passes_floor_limit'] for r in floor['checks'] if r['kind'] == 'control_grid') == floor['summary']['failed_control_grid_poses']
    report = {'status': 'saved_fit_verified_still_infeasible_between_source_poses', 'verifications': verified,
        'supported_limb_checks': len(limb_rows), 'retained_trial_losses_checked': trials,
        'position_sse_before_m2': sum(r['before_sse_m2'] for r in limb_rows), 'position_sse_after_m2': sum(r['after_sse_m2'] for r in limb_rows),
        'full_source_image_slots': len(images), 'image_status_counts': dict(Counter(r['status'] for r in images)), 'image_comparison': image_summary,
        'floor_before': floors[0]['summary'], 'floor_after': floors[1]['summary'],
        'failed_interpolated_checks': [r for r in floors[1]['checks'] if not r['passes_floor_limit']],
        'root_floor_shape_forces_unchanged': True, 'source_observations_unchanged': True,
        'physical_validation': 'unverified', 'scope': 'Construction fit and fixed sampled-floor consistency, not contact, dynamics, learning or independent accuracy',
        'new_inference_solve_physics_api_calls': 0, 'seconds': time.monotonic()-start}
    args.out.mkdir()
    for name, value in [('report.json', report), ('limbs.json', limb_rows), ('source-images.json', images)]: write_json(args.out/name, value)
    (args.out/'worker.py').write_bytes(Path(__file__).read_bytes())
    write_json(args.out/'manifest.json', {'schema': 1, 'files': {p.name: digest(p) for p in sorted(args.out.iterdir())}})
    print(json.dumps({'verification': verify_artifacts(args.out), 'limbs': len(limb_rows), 'trials': trials, 'image_comparison': image_summary, 'seconds': report['seconds']}))


if __name__ == '__main__':
    main()
