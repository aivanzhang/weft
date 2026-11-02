"""Audit frozen partial-limb transport and all image slots without new solving or physics."""

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
PARTIAL = 'multi_view_proximal_monocular_distal_and_twist'


def read(path):
    return json.loads(path.read_text())


def summary(values):
    return {'count': len(values), 'mean': float(np.mean(values)), 'rmse': float(np.sqrt(np.mean(np.square(values)))),
            'maximum': max(values)} if values else {'count': 0, 'mean': None, 'rmse': None, 'maximum': None}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists() or args.out.parent.resolve() != Path('artifacts').resolve():
        raise ValueError('Use a new artifact directory')
    start = time.monotonic()
    baseline = Path('artifacts/aspset-fixed-limb-body-lab-01/run')
    candidate = Path('artifacts/aspset-partial-limb-body-lab-01/run')
    verified = [verify_artifacts(p) for p in (baseline, candidate)]
    assert [v['manifest_sha256'] for v in verified] == [
        'e3988f89a7638a3a34f87fc71ab97f5f284b145bdb1bfa16081470c13b8aa3f4',
        '0254ca5f0bf4e97b7ec3d220170bfbd13f61c2076a14a55b93a15b2e1fc1b945']
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
    assert old_audit['dimensions'] == audit['dimensions']
    assert old_audit['limb_position_fit'] == audit['limb_position_fit']
    assert audit['partial_segments']['complete_limb_reference_sha256'] == canonical_hash(old['reference'])
    before, after = [read(p/'human-reference.json')['frames'] for p in (baseline, candidate)]
    assert len(before) == len(after) == len(new['reference']) == 47
    predictions = read(candidate/'two-view-triangulation.json')
    assert len(predictions['frames']) == len(audit['frames']) == 188
    parent = new['camera_reference']['body_refinement']['input_body']
    offset = np.r_[new['camera_reference']['origin_world_m'][:2], new['camera_reference']['ground']['height_m']]
    partial_rows = []; complete = 0
    camera_names = ['camera.json', 'inputs/two-view-root/camera.json', 'inputs/two-view-root/extra-0/camera.json']
    evidence_names = ['evidence.json', 'inputs/two-view-root/evidence.json', 'inputs/two-view-root/extra-0/evidence.json']
    sources = [{p['frame']: p for p in read(candidate/name)['frames']} for name in evidence_names]
    for i, (a, b, meta) in enumerate(zip(before, after, new['reference_evidence'], strict=True)):
        frame = int(meta['evidence_id'].rsplit(':', 1)[1]); stamp = frame/50
        assert a['source_time_s'] == b['source_time_s'] == meta['time_s'] == stamp
        assert meta['geometry_fit'] == old['reference_evidence'][i]['geometry_fit']
        changed = set()
        for group, indices in LIMB_ENDPOINTS.items():
            detail, previous = [v['frames'][frame]['groups'][group] for v in (audit, old_audit)]
            if detail['status'] != PARTIAL:
                assert detail == previous
                if 'length_fit' in detail: complete += 1
                continue
            names = [group+'_twist', group+'_lateral', group]
            distal = JOINTS[indices[1]]; changed.update(names)
            assert previous['status'] == 'monocular_prior_missing_points' and 'length_fit' not in detail
            assert detail['point_statuses'][-1] != 'estimated'
            assert new['reference'][i]['joints'][distal] == parent['reference'][i]['joints'][distal]
            assert all(meta['joint_confidence'][name] == 0. for name in names)
            assert meta['geometry_components'][group] == {'proximal_direction': 'multi_view_estimate',
                'axial_twist': 'transported_monocular_prior', 'distal_flexion': 'monocular_prior', 'distal_endpoint': 'unavailable'}
            points = np.array([predictions['frames'][frame]['joints'][j]['world_point_m'] for j in indices[:2]])
            assert np.allclose(detail['proximal_world_points_m'], points, atol=1e-12)
            target = points[1]-points[0]; target /= np.linalg.norm(target)
            proximal = np.subtract(b['human']['joints'][distal], b['human']['joints'][group]); proximal /= np.linalg.norm(proximal)
            np.testing.assert_allclose(proximal, target, atol=1e-10)
            prior = rotation_zyx([parent['reference'][i]['joints'][name] for name in names])
            proposed = rotation_zyx([new['reference'][i]['joints'][name] for name in names])
            minimum_swing = float(np.arccos(np.clip(prior[:, 2] @ proposed[:, 2], -1., 1.)))
            actual_swing = float(np.arccos(np.clip((np.trace(proposed @ prior.T)-1)/2, -1., 1.)))
            assert abs(actual_swing-minimum_swing) < 1e-7
            assert abs(actual_swing-detail['segment_swing_radians']) < 1e-7
            contributors = {str(j): predictions['frames'][frame]['joints'][j]['contributing_view_indices'] for j in indices[:2]}
            assert detail['joint_view_indices'] == meta['geometry_view_indices'][group] == contributors
            confidence = min(sources[view][frame]['landmarks'][j][key] for j in indices[:2]
                             for view in contributors[str(j)] for key in ('visibility', 'presence'))
            assert detail['proximal_image_confidence'] == confidence
            before_direction = np.subtract(a['human']['joints'][distal], a['human']['joints'][group])
            before_direction /= np.linalg.norm(before_direction)
            partial_rows.append({'frame': frame, 'group': group, 'minimum_swing_rad': actual_swing,
                'before_direction_error_rad': float(np.arccos(np.clip(before_direction @ target, -1., 1.))),
                'after_direction_error_rad': float(np.arccos(np.clip(proximal @ target, -1., 1.))),
                'contributors': contributors, 'retained_distal_bend_rad': new['reference'][i]['joints'][distal],
                'before_angles_rad': [old['reference'][i]['joints'][name] for name in names],
                'after_angles_rad': [new['reference'][i]['joints'][name] for name in names]})
        assert {k: v for k, v in old['reference'][i]['joints'].items() if k not in changed} == {
            k: v for k, v in new['reference'][i]['joints'].items() if k not in changed}
    assert len(partial_rows) == audit['partial_segments']['applied_group_poses']
    images = []; by_frame = {round(v['source_time_s']*50): i for i, v in enumerate(before)}; image_summary = {}
    for view, camera_name, poses in zip(('left', 'mid', 'right'), camera_names, sources, strict=True):
        camera = read(candidate/camera_name); K = np.array(camera['K']); known = []
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
                    cam = np.array(camera['frames'][frame]['R']) @ world+camera['frames'][frame]['t']
                    assert .2 <= cam[2] <= 100 and camera['radial_k1'] == 0
                    pixel = K @ cam; row[label] = float(np.linalg.norm(pixel[:2]/pixel[2]-target))
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
    report = {'status': 'saved_partial_transport_verified', 'verifications': verified, 'source_files_equal': len(source_files),
        'partial_limb_checks': len(partial_rows), 'unchanged_complete_limb_fits': complete,
        'full_source_image_slots': len(images), 'image_status_counts': dict(Counter(r['status'] for r in images)), 'image_comparison': image_summary,
        'floor_before': floors[0]['summary'], 'floor_after': floors[1]['summary'],
        'failed_checks': [r for r in floors[1]['checks'] if not r['passes_floor_limit']],
        'root_floor_shape_forces_unchanged': True, 'source_observations_unchanged': True, 'source_candidate_promoted': False,
        'physical_validation': 'unverified', 'scope': 'Construction consistency only; missing endpoints/bends/twist remain unmeasured',
        'new_inference_solve_physics_api_calls': 0, 'seconds': time.monotonic()-start}
    args.out.mkdir()
    for name, value in [('report.json', report), ('partial-limbs.json', partial_rows), ('source-images.json', images)]: write_json(args.out/name, value)
    (args.out/'worker.py').write_bytes(Path(__file__).read_bytes())
    write_json(args.out/'manifest.json', {'schema': 1, 'files': {p.name: digest(p) for p in sorted(args.out.iterdir())}})
    print(json.dumps({'verification': verify_artifacts(args.out), 'partial_limbs': len(partial_rows), 'complete_fits': complete,
        'floor_before': report['floor_before'], 'floor_after': report['floor_after'], 'image_comparison': image_summary, 'seconds': report['seconds']}))


if __name__ == '__main__':
    main()
