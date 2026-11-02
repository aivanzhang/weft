"""Audit saved third-view construction without new inference, optimization or physics."""

import argparse
from collections import Counter
import json
from pathlib import Path
import time

import numpy as np

from weft.io import canonical_hash, digest, write_json
from weft.pipeline import verify_artifacts


JOINTS = {11: 'left_shoulder', 12: 'right_shoulder', 13: 'left_elbow', 14: 'right_elbow',
          15: 'left_wrist', 16: 'right_wrist', 23: 'left_hip', 24: 'right_hip',
          25: 'left_knee', 26: 'right_knee', 27: 'left_ankle', 28: 'right_ankle'}


def read(path):
    return json.loads(path.read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists() or args.out.parent.resolve() != Path('artifacts').resolve():
        raise ValueError('Use a new artifact directory')
    start = time.monotonic()
    baseline = Path('artifacts/aspset-camera-body-lab-02/run')
    candidate = Path('artifacts/aspset-third-camera-body-lab-01/run')
    ground = Path('artifacts/aspset-manual-ground-01')
    inputs = Path('artifacts/aspset-third-camera-input-01')
    roots = [baseline, candidate, ground, inputs]
    pins = ['99ac31f00f82f9c9a7e347aa8e6bc3b17196f523683911db331515a7629fde29',
            'fe2b84b7b8f7c26ab1f912f87962a2e7ef13eb58d88bcc035911514e98365857',
            'bdf11ee7ddd7b74fce3420c6ee36902acc35b9c57d15117536671bf6745f23f2',
            '8abfc496985b8538d07ae7b33d741dd33a21e658842f2076818fbd477e28a928']
    verifications = {str(root): verify_artifacts(root) for root in roots}
    assert [verifications[str(root)]['manifest_sha256'] for root in roots] == pins
    for name in ('evidence.json', 'source.json', 'camera.json', 'ground.json', 'camera-reference.json',
                 'inputs/two-view-root/config.json', 'inputs/two-view-root/camera.json', 'inputs/two-view-root/evidence.json'):
        assert (baseline/name).read_bytes() == (candidate/name).read_bytes(), name
    assert (candidate/'ground.json').read_bytes() == (ground/'ground-0.json').read_bytes()
    evidence = [read(candidate/name) for name in ('evidence.json', 'inputs/two-view-root/evidence.json',
                                                 'inputs/two-view-root/extra-0/evidence.json')]
    cameras = [read(candidate/name) for name in ('camera.json', 'inputs/two-view-root/camera.json',
                                                'inputs/two-view-root/extra-0/camera.json')]
    poses = [{row['frame']: row for row in value['frames']} for value in evidence]
    predicted = read(candidate/'two-view-triangulation.json')
    assert predicted['source_frames'] == len(predicted['frames']) == 188 and predicted['joint_slots'] == 6204
    assert [row['evidence_sha256'] for row in predicted['views']] == [canonical_hash(v) for v in evidence]
    assert [row['camera_sha256'] for row in predicted['views']] == [canonical_hash(v) for v in cameras]
    projections = 0; max_error_difference = 0.; max_rotation_adjustment = 0.; statuses = Counter()
    for frame, row in enumerate(predicted['frames']):
        assert row['frame'] == frame and row['time_s'] == frame/50 and len(row['joints']) == 33
        assert row['source_evidence_ids'] == [p[frame]['id'] if frame in p else None for p in poses]
        for i, correction in enumerate(row['solver_rotations']):
            R = np.array(correction['R']); original = np.array(cameras[i]['frames'][frame]['R'])
            change = float(np.max(np.abs(R-original)))
            assert change == correction['maximum_element_change'] and change <= 1e-5
            assert np.max(np.abs(R @ R.T-np.eye(3))) < 1e-12 and abs(np.linalg.det(R)-1) < 1e-12
            max_rotation_adjustment = max(max_rotation_adjustment, change)
        for joint, estimate in enumerate(row['joints']):
            supported = []; pixels = {}
            for i, stream in enumerate(poses):
                pose = stream.get(frame)
                if pose is None: continue
                assert pose['pose_count'] == 1 and pose['status'] in ('tracked', 'tracked_partial', 'low_confidence')
                point = pose['landmarks'][joint]
                if 0 <= point['x'] <= 1 and 0 <= point['y'] <= 1 and .5 <= min(point['visibility'], point['presence']) <= max(point['visibility'], point['presence']) <= 1:
                    supported.append(i); pixels[i] = np.array([point['x']*3840, point['y']*2160])
            assert estimate['contributing_view_indices'] == supported
            statuses[estimate['status']] += 1
            if estimate['status'] != 'estimated':
                assert estimate['world_point_m'] is None
                continue
            assert len(supported) >= 2 and estimate['optimization_status'] == 'converged'
            errors = []
            for i in supported:
                camera = cameras[i]; R = np.array(row['solver_rotations'][i]['R'])
                position = R @ estimate['world_point_m']+camera['frames'][frame]['t']
                assert .2 <= position[2] <= 100 and camera['radial_k1'] == 0
                projection = np.array(camera['K']) @ position
                errors.append(float(np.linalg.norm(projection[:2]/projection[2]-pixels[i])))
                projections += 1
            max_error_difference = max(max_error_difference, float(np.max(np.abs(np.array(errors)-estimate['reprojection_px']))))
            assert max(errors) <= .01*3840
    assert dict(statuses) == predicted['joint_status_counts'] and max_error_difference < 1e-8
    reports, residuals = {}, []
    for name, path in [('two_view', baseline), ('three_view', candidate)]:
        body = read(path/'human.json'); floor = read(path/'reference-feasibility.json')
        assert len(body['reference']) == len(body['reference_evidence']) == 47
        assert floor['contract']['floor_penetration_limit_model_units'] == .025
        failures = [row for row in floor['checks'] if not row['passes_floor_limit']]
        assert len(floor['checks']) == 158
        assert all(row['passes_floor_limit'] == (row['max_penetration_model_units'] <= .025) for row in floor['checks'])
        assert len([row for row in failures if row['kind'] == 'source_pose']) == floor['summary']['failed_source_poses']
        assert len([row for row in failures if row['kind'] == 'control_grid']) == floor['summary']['failed_control_grid_poses']
        assert max(row['max_penetration_model_units'] for row in floor['checks']) == floor['summary']['max_penetration_model_units']
        reports[name] = {'floor': floor['summary'], 'lengths': body['lengths'], 'failed_checks': failures,
                         'body_coverage': read(path/'two-view-body.json')['group_coverage']}
        if name != 'three_view': continue
        binding = body['camera_reference']
        offset = np.r_[binding['origin_world_m'][:2], binding['ground']['height_m']]
        snapshots = read(path/'human-reference.json')['frames']
        for meta, snapshot in zip(body['reference_evidence'], snapshots, strict=True):
            frame = int(meta['evidence_id'].rsplit(':', 1)[1]); row = predicted['frames'][frame]
            assert snapshot['source_time_s'] == meta['time_s'] == frame/50
            for joint, label in JOINTS.items():
                estimated = row['joints'][joint]; model = np.array(snapshot['human']['joints'][label])+offset
                image_point = estimated['world_point_m']
                residuals.append({'frame': frame, 'joint': label, 'image_status': estimated['status'],
                    'contributing_view_indices': estimated['contributing_view_indices'], 'image_estimate_m': image_point,
                    'model_point_m': model.tolist(), 'model_image_distance_m': None if image_point is None else float(np.linalg.norm(model-image_point)),
                    'model_height_above_ground_m': float(model[2]-binding['ground']['height_m']),
                    'image_height_above_ground_m': None if image_point is None else image_point[2]-binding['ground']['height_m']})
    assert [row['evidence_id'] for row in read(baseline/'human.json')['reference_evidence']] == [row['evidence_id'] for row in body['reference_evidence']]
    report = {'status': 'saved_checks_verified_candidate_unpromoted', 'verifications': verifications,
        'source_slots': 188, 'joint_slots': 6204, 'projections_checked': projections,
        'maximum_projection_difference_px': max_error_difference, 'maximum_rotation_element_adjustment': max_rotation_adjustment,
        'joint_status_counts': dict(statuses), 'comparisons': reports, 'model_image_comparison_slots': len(residuals),
        'scope': 'Construction agreement and fixed floor checks only; image estimates are not independent physical truth',
        'physical_validation': 'unverified', 'default_changed': False, 'reference_inputs': [],
        'inference_or_physics_or_api_calls': 0, 'seconds': time.monotonic()-start}
    args.out.mkdir()
    write_json(args.out/'report.json', report)
    write_json(args.out/'model-image-joints.json', residuals)
    (args.out/'worker.py').write_bytes(Path(__file__).read_bytes())
    write_json(args.out/'manifest.json', {'schema': 1, 'files': {p.name: digest(p) for p in sorted(args.out.iterdir())}})
    print(json.dumps({'verification': verify_artifacts(args.out), 'projections_checked': projections,
        'maximum_projection_difference_px': max_error_difference, 'seconds': report['seconds']}))


if __name__ == '__main__':
    main()
