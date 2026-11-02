"""Independently audit retained depth alternatives against the frozen v2 source run."""

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import time

import numpy as np

from weft.io import canonical_hash, digest, write_json
from weft.spatial_pose import LIMB_ENDPOINTS


BASELINE = Path('artifacts/aspset-distal-uncertainty-body-lab-01/run')
CANDIDATE = Path('artifacts/aspset-depth-branches-body-lab-01/run')
SEALS = ('672234299b8645616a8fb2b4787591774aa4bea9e9623d3af7a6915bc5bdf449',
         '6deb3c0b9f5171b531f06d6443f4bcb6df6a7b0491ad62f0bf3ee612353d6402')
JOINTS = {11: 'left_shoulder', 12: 'right_shoulder', 13: 'left_elbow', 14: 'right_elbow',
          15: 'left_wrist', 16: 'right_wrist', 23: 'left_hip', 24: 'right_hip',
          25: 'left_knee', 26: 'right_knee', 27: 'left_ankle', 28: 'right_ankle'}
CAMERAS = ['camera.json', 'inputs/two-view-root/camera.json', 'inputs/two-view-root/extra-0/camera.json']
EVIDENCE = ['evidence.json', 'inputs/two-view-root/evidence.json', 'inputs/two-view-root/extra-0/evidence.json']


def read(path):
    return json.loads(path.read_text())


def verify_artifacts(directory):
    # Avoid the pipeline's video/provider imports in a saved-data-only verifier.
    files = read(directory/'manifest.json')['files']
    for name, expected in files.items():
        path = directory/name
        assert path.resolve().is_relative_to(directory.resolve()) and not path.is_symlink()
        assert digest(path) == expected, path
    return {'manifest_sha256': digest(directory/'manifest.json'), 'verified_files': len(files)}


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


def source_ray(pixel, camera, frame, rotation):
    # A cubic inversion checks the production Newton inversion independently.
    xy = np.linalg.solve(np.array(camera['K']), np.r_[pixel, 1.])[:2]
    radius = float(np.linalg.norm(xy)); k = camera['radial_k1']
    if radius > 4 or k < 0 and radius >= 2/3*math.sqrt(-1/(3*k)):
        return None
    if k and radius:
        roots = np.roots([k, 0., 1., -radius])
        accepted = [float(r.real) for r in roots if abs(r.imag) < 1e-10 and r.real >= 0 and 1+3*k*r.real**2 > 0]
        assert len(accepted) == 1
        xy *= accepted[0]/radius
    direction = rotation.T @ np.r_[xy, 1.]
    return -rotation.T @ camera['frames'][frame]['t'], direction/np.linalg.norm(direction)


def temporal_context(body, native, index, group, distal):
    stamp = body['reference_evidence'][index]['time_s']
    eligible = [i for i, row in enumerate(body['reference_evidence'])
                if row['geometry_sources'][group] in ('two_view_estimate', 'multi_view_estimate')
                and group in row['geometry_fit']]
    left = [i for i in eligible if body['reference_evidence'][i]['time_s'] < stamp]
    right = [i for i in eligible if body['reference_evidence'][i]['time_s'] >= stamp]
    result = {'status': 'missing_bracketing_anchors', 'anchors': [], 'direction': None}
    if not left or not right:
        return result
    middle = group.rsplit('_', 1)[0]+('_knee' if group.endswith('hip') else '_elbow')
    for i in (left[-1], right[0]):
        meta = body['reference_evidence'][i]; joints = native[i]['human']['joints']
        vector = np.array(joints[distal])-joints[middle]; vector /= np.linalg.norm(vector)
        result['anchors'].append({'reference_index': i, 'evidence_id': meta['evidence_id'],
                                  'time_s': meta['time_s'], 'direction': vector.tolist()})
    a, b = result['anchors']
    if not 0 < stamp-a['time_s'] <= .25 or not 0 < b['time_s']-stamp <= .25:
        result['status'] = 'anchor_gap'
    elif np.dot(a['direction'], b['direction']) < 0:
        result['status'] = 'inconsistent_anchor_directions'
    else:
        fraction = (stamp-a['time_s'])/(b['time_s']-a['time_s'])
        vector = (1-fraction)*np.array(a['direction'])+fraction*np.array(b['direction'])
        result.update(status='smooth_motion_assumption', direction=(vector/np.linalg.norm(vector)).tolist())
    return result


def audit_branches(fit, attempt, cameras, frame, anchor):
    rotation = np.array(fit['input_rotation_world']); origin = np.array(fit['fixed_knee_world_m'])
    length, bend, sign = [fit[k] for k in ('fixed_distal_length_m', 'input_bend_rad', 'bend_sign')]
    assert fit['depth_measured'] is False and fit['method'] == 'temporal-distal-ray-sphere-v1'
    before = origin+rotation @ [0., sign*length*math.sin(bend), -length*math.cos(bend)]
    np.testing.assert_allclose(before, fit['before_world_m'], atol=1e-12, rtol=0)
    views = attempt['views']

    def score(point):
        errors = [float(np.linalg.norm(project(point, cameras[v['view_index']], frame, v['solver_rotation'])-v['pixel'])) for v in views]
        sse = sum(v['confidence']*(e/max(cameras[v['view_index']]['width'], cameras[v['view_index']]['height']))**2
                  for v, e in zip(views, errors, strict=True))
        distance = None if anchor is None else float(np.linalg.norm((point-origin)/length-anchor))
        return errors, sse, distance

    errors, before_sse, before_distance = score(before)
    np.testing.assert_allclose(errors, fit['pixel_errors_before'], atol=1e-9, rtol=0)
    assert abs(before_sse-fit['pixel_sse_before']) < 1e-12
    if anchor is None: assert fit['temporal_distance_before'] is None
    else: assert abs(before_distance-fit['temporal_distance_before']) < 1e-10
    endpoints = []; rows = []; raw_roots = 0; misses = 0
    for local, (view, ray) in enumerate(zip(views, fit['rays'], strict=True)):
        assert ray['view_index'] == local
        prepared = source_ray(view['pixel'], cameras[view['view_index']], frame, np.array(view['solver_rotation']))
        if prepared is None:
            assert ray['status'] == 'distortion_not_invertible' and not ray['roots']
            continue
        center, direction = prepared; offset = center-origin
        coefficients = [float(direction @ direction), 2*float(offset @ direction), float(offset @ offset)-length**2]
        discriminant = (coefficients[1]**2-4*coefficients[0]*coefficients[2])/4
        assert abs(discriminant-ray['discriminant']) < 1e-10
        if discriminant < 0:
            assert ray['status'] == 'no_sphere_intersection' and not ray['roots']; misses += 1
            continue
        roots = sorted(np.roots(coefficients)); assert ray['status'] == 'intersections_retained'
        for depth, saved in zip(roots, ray['roots'], strict=True):
            assert not np.iscomplexobj(depth); raw_roots += 1
            point = center+depth*direction
            np.testing.assert_allclose(point, saved['world_m'], atol=1e-9, rtol=0)
            assert abs(depth-saved['distance']) < 1e-9
            if depth <= 0:
                assert saved['rejection'] == 'nonforward_root' and saved['candidate_index'] is None
                continue
            duplicates = [i for i, p in enumerate(endpoints) if np.linalg.norm(point-p) <= 1e-8]
            if duplicates:
                assert saved['candidate_index'] == duplicates[0] and saved['rejection'] == 'duplicate_endpoint'
                continue
            n = len(endpoints); endpoints.append(point)
            assert saved['candidate_index'] == n and saved['rejection'] is None
            candidate = fit['candidates'][n]
            np.testing.assert_allclose(point, candidate['world_m'], atol=1e-9, rtol=0)
            local_point = rotation.T @ (point-origin)/length
            output_bend = math.atan2(math.hypot(local_point[0], local_point[1]), -local_point[2])
            unknown = math.hypot(local_point[0], local_point[1]) <= 1e-8
            twist = 0. if unknown else math.atan2(-sign*local_point[0], sign*local_point[1])
            np.testing.assert_allclose([twist, output_bend-bend], candidate['correction_rad'], atol=1e-8, rtol=0)
            assert abs(output_bend-candidate['output_bend_rad']) < 1e-8 and unknown == candidate['axial_twist_unobservable']
            rejection = 'hinge_range' if output_bend > 2.6 else None
            values = (None, None, None)
            if rejection is None:
                try: values = score(point)
                except AssertionError: rejection = 'projection_outside_envelope'
            errors, sse, distance = values
            if errors is not None:
                np.testing.assert_allclose(errors, candidate['pixel_errors'], atol=1e-7, rtol=0)
                assert abs(sse-candidate['pixel_sse']) < 1e-12
                if anchor is None: assert candidate['temporal_distance'] is None
                else: assert abs(distance-candidate['temporal_distance']) < 1e-8
                if sse >= before_sse-1e-12: rejection = 'no_all_view_image_improvement'
                elif anchor is None: rejection = 'missing_temporal_anchor'
                elif distance > .5 or distance >= before_distance-1e-10: rejection = 'insufficient_temporal_agreement'
            else:
                assert all(candidate[k] is None for k in ('pixel_errors', 'pixel_sse', 'temporal_distance'))
            assert rejection == candidate['rejection']
            rows.append({'candidate_index': n, 'world_m': point.tolist(), 'pixel_sse': sse,
                         'temporal_distance': distance, 'rejection': rejection})
    assert len(endpoints) == len(fit['candidates']) <= 8
    ranked = sorted((r['temporal_distance'], r['candidate_index']) for r in rows
                    if r['rejection'] in (None, 'insufficient_temporal_agreement'))
    selected = None; gap = None; status = 'missing_temporal_anchor' if anchor is None else 'no_qualified_branch'
    if anchor is not None and ranked and rows[ranked[0][1]]['rejection'] is None:
        gap = ranked[1][0]-ranked[0][0] if len(ranked) > 1 else None
        status = 'ambiguous_branches' if gap is not None and gap < .25 else 'selected_temporal_hypothesis'
        if status == 'selected_temporal_hypothesis': selected = ranked[0][1]
    assert fit['selected_index'] == selected and fit['status'] == status
    if gap is None: assert fit['runner_up_gap_lengths'] is None
    else: assert abs(gap-fit['runner_up_gap_lengths']) < 1e-8
    return {'frame': frame, 'status': status, 'selected_index': selected, 'source_rays': len(views),
            'raw_roots': raw_roots, 'sphere_misses': misses, 'candidates': rows}


def audit_images(cameras, sources, before, after, offset):
    images = []; per_view = {}; by_frame = {round(v['source_time_s']*50): i for i, v in enumerate(before)}
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
        per_view[view] = {'before': summary([r['before_px'] for r in known]), 'after': summary([r['after_px'] for r in known]),
                          'regressed_points': sum(r['after_px'] > r['before_px']+1e-8 for r in known)}
    return images, per_view


def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists() or args.out.parent.resolve() != Path('artifacts').resolve():
        raise ValueError('Use a new artifact directory')
    started = time.monotonic(); verified = [verify_artifacts(p) for p in (BASELINE, CANDIDATE)]
    assert tuple(v['manifest_sha256'] for v in verified) == SEALS
    prior_audit = verify_artifacts(Path('artifacts/aspset-distal-uncertainty-audit-01'))
    assert prior_audit['manifest_sha256'] == 'ccfc1361e4c0ccda6ff691abe0c2549c6b6a9691605c20ea69e50b0f493cbdff'
    source_files = ['evidence.json', 'source.json', 'camera.json', 'ground.json', 'camera-reference.json',
                    'two-view-triangulation.json', 'two-view-root.json', 'partial-pixel-uncertainty.json']
    source_files += [str(p.relative_to(CANDIDATE)) for p in (CANDIDATE/'inputs').rglob('*') if p.is_file()]
    for name in source_files: assert (BASELINE/name).read_bytes() == (CANDIDATE/name).read_bytes(), name
    old, new = [read(p/'human.json') for p in (BASELINE, CANDIDATE)]
    old_audit, audit = [read(p/'two-view-body.json') for p in (BASELINE, CANDIDATE)]
    before, after = [read(p/'human-reference.json')['frames'] for p in (BASELINE, CANDIDATE)]
    assert len(before) == len(after) == len(new['reference']) == 47
    for key in ('lengths', 'mobility', 'body_frame', 'root_reference', 'motor_limits'):
        assert old[key] == new[key], key
    for key in ('ground', 'root_refinement'): assert old['camera_reference'][key] == new['camera_reference'][key]
    for key in ('dimensions', 'limb_position_fit', 'partial_pixels'): assert old_audit[key] == audit[key], key
    branch_summary = audit['depth_branches']; binding = new['camera_reference']['body_refinement']
    assert branch_summary == new['reconstruction_coverage']['two_view_body']['depth_branches']
    assert binding['depth_branch_reference_sha256'] == branch_summary['before_branch_reference_sha256'] == canonical_hash(old['reference'])
    contract = binding['depth_branch_contract']
    expected = {'maximum_views': 4, 'maximum_roots': 8, 'maximum_body_seconds': 20., 'maximum_anchor_gap_s': .25,
                'minimum_anchor_direction_dot': 0., 'maximum_axial_change_rad': math.pi, 'maximum_bend_rad': 2.6,
                'maximum_temporal_distance_lengths': .5, 'minimum_branch_gap_lengths': .25,
                'duplicate_endpoint_tolerance_m': 1e-8, 'minimum_pixel_improvement': 1e-12,
                'minimum_temporal_improvement': 1e-10, 'floor_selection': False, 'depth_measured': False}
    assert all(contract[k] == v for k, v in expected.items()) and branch_summary['depth_measured'] is False
    cameras = [read(CANDIDATE/name) for name in CAMERAS]
    sources = [{p['frame']: p for p in read(CANDIDATE/name)['frames']} for name in EVIDENCE]
    offset = np.r_[new['camera_reference']['origin_world_m'][:2], new['camera_reference']['ground']['height_m']]
    rows = []; complete = 0; retained_trials = 0
    for i, meta in enumerate(new['reference_evidence']):
        frame = int(meta['evidence_id'].rsplit(':', 1)[1]); changed = set()
        assert before[i]['source_time_s'] == after[i]['source_time_s'] == meta['time_s'] == frame/50
        assert meta['geometry_fit'] == old['reference_evidence'][i]['geometry_fit']
        for group, indices in LIMB_ENDPOINTS.items():
            detail = audit['frames'][frame]['groups'][group]; previous = old_audit['frames'][frame]['groups'][group]
            assert {k: v for k, v in detail.items() if k != 'depth_branch_attempt'} == previous
            if 'length_fit' in detail: complete += 1
            attempt = detail.get('depth_branch_attempt')
            if not attempt: continue
            pixels = detail['distal_pixel_attempt']; fit = attempt['fit']; distal = JOINTS[indices[-1]]
            retained_trials += sum(len(it['trials']) for it in pixels['fit']['iterations'])
            names = (group+'_twist', group+'_lateral', group, JOINTS[indices[1]])
            assert attempt['input_angles_rad'] == [old['reference'][i]['joints'][k] for k in names]
            assert attempt['source_view_indices'] == pixels['view_indices']
            for v in pixels['views']:
                source = sources[v['view_index']][frame]; point = source['landmarks'][indices[-1]]; camera = cameras[v['view_index']]
                assert v['source_evidence_id'] == source['id'] and v['camera_sha256'] == canonical_hash(camera)
                assert v['pixel'] == [point['x']*camera['width'], point['y']*camera['height']]
                assert v['confidence'] == min(point['visibility'], point['presence']) >= .5
            for key, previous_key in [('input_rotation_world', 'output_rotation_world'), ('input_bend_rad', 'output_bend_rad'),
                                      ('fixed_knee_world_m', 'fixed_knee_world_m'), ('fixed_distal_length_m', 'fixed_distal_length_m')]:
                assert fit[key] == pixels['fit'][previous_key]
            context = temporal_context(old, before, i, group, distal)
            assert context['status'] == attempt['temporal']['status']
            for a, b in zip(context['anchors'], attempt['temporal']['anchors'], strict=True):
                for key in ('reference_index', 'evidence_id', 'time_s'): assert a[key] == b[key]
                np.testing.assert_allclose(a['direction'], b['direction'], atol=1e-10, rtol=0)
            if context['direction'] is None: assert attempt['temporal']['direction'] is fit['anchor_direction'] is None
            else:
                np.testing.assert_allclose(context['direction'], attempt['temporal']['direction'], atol=1e-10, rtol=0)
                np.testing.assert_allclose(context['direction'], fit['anchor_direction'], atol=1e-10, rtol=0)
            row = audit_branches(fit, pixels, cameras, frame, context['direction']); row['group'] = group; rows.append(row)
            for name in (group, names[-1]):
                np.testing.assert_allclose(before[i]['human']['joints'][name], after[i]['human']['joints'][name], atol=1e-12, rtol=0)
            if row['selected_index'] is None:
                assert group not in meta['geometry_depth_branches']
                continue
            selected = fit['candidates'][row['selected_index']]; changed.update(names)
            np.testing.assert_allclose(np.array(after[i]['human']['joints'][distal])+offset, selected['world_m'], atol=1e-9, rtol=0)
            assert all(meta['joint_confidence'][k] == 0. for k in names)
        assert {k: v for k, v in old['reference'][i]['joints'].items() if k not in changed} == {
            k: v for k, v in new['reference'][i]['joints'].items() if k not in changed}
    assert len(audit['frames']) == len(old_audit['frames']) == 188
    assert branch_summary['attempted_group_poses'] == len(rows) == 24
    assert branch_summary['selected_group_poses'] == sum(r['selected_index'] is not None for r in rows)
    assert branch_summary['candidate_branches'] == sum(len(r['candidates']) for r in rows)
    assert branch_summary['status_counts'] == dict(Counter(r['status'] for r in rows))
    images, image_summary = audit_images(cameras, sources, before, after, offset)
    floors = [read(p/'reference-feasibility.json') for p in (BASELINE, CANDIDATE)]
    assert floors[0]['contract'] == floors[1]['contract']
    for a, b in zip(floors[0]['checks'], floors[1]['checks'], strict=True):
        for key in ('kind', 'time_s', 'source_evidence_ids', 'bracketing_reference_indices'): assert a[key] == b[key]
    for floor in floors:
        assert len(floor['checks']) == 158
        assert all(r['passes_floor_limit'] == (r['max_penetration_model_units'] <= .025) for r in floor['checks'])
        for kind, key in [('source_pose', 'failed_source_poses'), ('control_grid', 'failed_control_grid_poses')]:
            assert sum(not r['passes_floor_limit'] for r in floor['checks'] if r['kind'] == kind) == floor['summary'][key]
    report = {'status': 'saved_depth_branches_verified', 'verifications': verified, 'prior_v2_audit': prior_audit,
        'source_files_equal': len(source_files), 'unchanged_v2_pixel_fits': len(rows), 'unchanged_v2_trial_losses': retained_trials,
        'unchanged_complete_limb_fits': complete, 'depth_branches': branch_summary,
        'source_rays': sum(r['source_rays'] for r in rows), 'raw_roots': sum(r['raw_roots'] for r in rows),
        'sphere_misses': sum(r['sphere_misses'] for r in rows),
        'candidate_rejections': dict(Counter(c['rejection'] for r in rows for c in r['candidates'])),
        'full_source_image_slots': len(images), 'image_status_counts': dict(Counter(r['status'] for r in images)),
        'image_comparison': image_summary, 'floor_before': floors[0]['summary'], 'floor_after': floors[1]['summary'],
        'failed_checks': [r for r in floors[1]['checks'] if not r['passes_floor_limit']],
        'root_floor_shape_forces_unchanged': True, 'source_observations_unchanged': True, 'source_candidate_promoted': False,
        'physical_validation': 'unverified', 'scope': 'Nominal alternatives and assumed smooth motion, not measured depth or contact',
        'optimizer_inference_native_physics_api_calls': 0, 'seconds': time.monotonic()-started,
        'projection_method': 'Independent saved-camera pinhole/radial equations; no production proposer or image solver'}
    args.out.mkdir()
    for name, value in [('report.json', report), ('depth-branches.json', rows), ('source-images.json', images)]: write_json(args.out/name, value)
    (args.out/'worker.py').write_bytes(Path(__file__).read_bytes())
    write_json(args.out/'manifest.json', {'schema': 1, 'files': {p.name: digest(p) for p in sorted(args.out.iterdir())}})
    print(json.dumps({'verification': verify_artifacts(args.out), 'branches': branch_summary, 'seconds': report['seconds']}))


if __name__ == '__main__':
    main()
