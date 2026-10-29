"""Fit a complete-source fixed-length hypothesis without changing image observations."""

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
import tomllib

import cv2
import numpy as np

from weft.benchmark import seal_suite
from weft.constrained_stereo import CONTRACT, fit_stereo_lengths
from weft.io import digest, local_time, write_json
from weft.multiview_pose import _observation
from weft.pipeline import _implementation_files


ROOT = Path(__file__).resolve().parents[2]
POSES = ROOT/'artifacts/freemocap-full-pose-01'
JOINTS = ROOT/'artifacts/freemocap-camera-space-joints-01'
SOURCE = ROOT/'artifacts/freemocap-full-source-01'
MAJOR = (11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28)
SEGMENTS = {'shoulder_width': (11, 12), 'hip_width': (23, 24),
    'left_upper_arm': (11, 13), 'right_upper_arm': (12, 14), 'left_forearm': (13, 15), 'right_forearm': (14, 16),
    'left_thigh': (23, 25), 'right_thigh': (24, 26), 'left_shin': (25, 27), 'right_shin': (26, 28)}
COEFFICIENTS = {name: {str(a): 1., str(b): -1.} for name, (a, b) in SEGMENTS.items()}
COEFFICIENTS['torso_midpoints'] = {'11': .5, '12': .5, '23': -.5, '24': -.5}


def read(path):
    return json.loads(path.read_text())


def verified(root, name):
    path = root/name
    if digest(path) != read(root/'manifest.json')['files'][name]:
        raise ValueError('Frozen length-fit input changed: '+name)
    return path


def worker(out, index):
    pre = read(out.parent/'pre-run.json'); prepared = read(out.parent/'prepared-observations.json')
    if digest(out.parent/'prepared-observations.json') != pre['prepared_observations_sha256']:
        raise ValueError('Prepared image observations changed')
    out.mkdir(exist_ok=False); started = time.perf_counter(); count = 0; statuses = Counter()
    with (out/'frame-proposals.jsonl').open('x') as journal:
        for row in prepared:
            if (row['frame']//2)%2 != index-1:
                continue
            result = {'frame': row['frame'], 'time_s': row['time_s'], 'joint_indices': row['joint_indices'],
                'observation_statuses': row['observation_statuses'], 'status': 'insufficient_constraint_support', 'fit': None}
            if len(row['joint_indices']) >= 2 and row['constraints']:
                try:
                    tick = time.perf_counter()
                    fit = fit_stereo_lengths(row['pixels'], pre['cameras'], row['constraints'])
                    result.update(status='fit_proposed', fit=fit, fit_seconds=time.perf_counter()-tick)
                except ValueError as error:
                    result.update(status='unsupported_initial_geometry', reason=str(error))
            journal.write(json.dumps(result, allow_nan=False)+'\n'); journal.flush()
            statuses[result['status']] += 1; count += 1
            if time.perf_counter()-started > 60:
                raise TimeoutError('Length fitting exceeded the per-worker work envelope')
    write_json(out/'report.json', {'status': 'assigned_samples_complete', 'sample_count': count,
        'statuses': dict(statuses), 'runtime_seconds': time.perf_counter()-started})


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--compare-prepared', type=Path)
    parser.add_argument('--worker', type=int, choices=(1, 2)); args = parser.parse_args(); out = args.out.resolve()
    if args.worker:
        worker(out, args.worker); return
    if out.parent != ROOT/'artifacts':
        raise ValueError('Use a new top-level artifact directory')
    views = [read(verified(POSES, f'view-{v}/evidence.json')) for v in (1, 2)]
    lookup = [{r['frame']: r for r in v['frames']} for v in views]
    baseline = read(verified(JOINTS, 'camera-space-joints.json'))['frames']
    calibration = tomllib.loads(verified(SOURCE, 'member-04-freemocap_sample_data_camera_calibration.toml').read_text())
    if len(baseline) != 1108 or any(len(v['frames']) != 554 for v in views):
        raise ValueError('Complete source coverage changed')
    cameras = []
    for i in (0, 1):
        c = calibration[f'cam_{i}']
        if any(c['distortions'][j] != 0 for j in range(1, 5)):
            raise ValueError('Only the declared single radial-distortion term is supported')
        cameras.append({'matrix': c['matrix'], 'radial': c['distortions'][0],
            'rotation': cv2.Rodrigues(np.array(c['rotation']))[0].tolist(),
            'translation': (np.array(c['translation'])*.001).tolist(), 'width': 720, 'height': 1280})
    parameters = {}; length_evidence = []
    for name, coefficients in COEFFICIENTS.items():
        values = []
        for frame in range(400, 622, 2):
            joints = baseline[frame]['joints']
            supported = all(joints[int(j)]['status'] == 'estimated' for j in coefficients)
            length = None
            if supported:
                delta = sum(weight*np.array(joints[int(j)]['world_point_m']) for j, weight in coefficients.items())
                length = float(np.linalg.norm(delta)); values.append(length)
            length_evidence.append({'frame': frame, 'segment': name, 'raw_supported': supported, 'length_m': length})
        if len(values) < 16:
            raise ValueError('Opening interval lacks enough original segment support: '+name)
        parameters[name] = {'length_m': float(np.median(values)), 'count': len(values), 'coefficients': coefficients,
            'p95_absolute_deviation_m': float(np.percentile(np.abs(np.array(values)-np.median(values)), 95)),
            'quality': 'inferred_from_construction_images_not_measured_anatomy'}
    prepared = []
    for frame in range(0, 1108, 2):
        observations = {j: [_observation(v.get(frame), j, 720, 1280) for v in lookup] for j in MAJOR}
        indices = [j for j in MAJOR if all(p[0] is not None for p in observations[j])]
        constraints = []
        for name, p in parameters.items():
            if all(int(j) in indices for j in p['coefficients']):
                constraints.append({'name': name, 'length_m': p['length_m'],
                    'coefficients': [p['coefficients'].get(str(j), 0.) for j in indices]})
        prepared.append({'frame': frame, 'time_s': frame/30, 'joint_indices': indices,
            'pixels': [[observations[j][v][0].tolist() for j in indices] for v in (0, 1)],
            'observation_statuses': {str(j): [p[1] for p in observations[j]] for j in MAJOR}, 'constraints': constraints})
    previous = None
    if args.compare_prepared:
        prior = args.compare_prepared.resolve()
        if (prepared != read(verified(prior, 'prepared-observations.json'))
                or parameters != read(verified(prior, 'frozen-lengths.json'))
                or length_evidence != read(verified(prior, 'length-construction-slots.json'))
                or cameras != read(verified(prior, 'pre-run.json'))['cameras']):
            raise ValueError('Solver comparison changed frozen observations, lengths or cameras')
        previous = {'path': str(prior.relative_to(ROOT)), 'manifest_sha256': digest(prior/'manifest.json')}
    runtime = {n: digest(p) for n, p in _implementation_files().items()}
    out.mkdir(exist_ok=False)
    for source in (Path(__file__), ROOT/'src/weft/constrained_stereo.py'):
        shutil.copyfile(source, out/source.name)
    write_json(out/'length-construction-slots.json', length_evidence)
    write_json(out/'frozen-lengths.json', parameters)
    write_json(out/'prepared-observations.json', prepared)
    write_json(out/'pre-run.json', {'created_at': local_time(), 'contract': dict(CONTRACT), 'cameras': cameras,
        'hypothesis': 'Fixed segment lengths couple independent two-view joints and may reduce inconsistent geometry; image observations and raw rejection statuses remain unchanged.',
        'input_manifests': {p.name: digest(p/'manifest.json') for p in (POSES, JOINTS, SOURCE)},
        'runtime_hashes': runtime, 'prepared_observations_sha256': digest(out/'prepared-observations.json'),
        'matched_previous_inputs': previous,
        'parameter_construction_frames': {'first': 400, 'last_inclusive': 620, 'stride': 2},
        'comparison': 'All source slots retained; later active frames 622–1058 form a descriptive development comparison, not a fresh final holdout.',
        'missing_joints': 'Require original usable image observations in both views; no temporal interpolation, monocular world prior, crop proposal or label swap.',
        'numerical_support': 'Solver converged, all fitted lengths within 1e-5 m, and each joint within the unchanged 1% image-size reprojection limit in both cameras.',
        'retained_support_check': 'Count every originally supported major joint lost, and every previously unsupported joint gained; no aggregate improvement waives losses.',
        'maximum_source_frames': 1108, 'samples': 554, 'workers': 2, 'worker_timeout_seconds': 65,
        'maximum_output_bytes': 64000000, 'new_video_decodes': 0, 'new_pose_inference': 0,
        'reserved_camera_opened': False, 'projected_paid_spend_usd': 0,
        'scope': 'Camera-0 coordinate proposals only. No floor, calibrated gravity, body, muscles, dynamics, policy, admission or default change.'})
    started = time.perf_counter()
    def run(index):
        with (out/f'worker-{index}.stdout').open('w') as stdout, (out/f'worker-{index}.stderr').open('w') as stderr:
            try:
                p = subprocess.run([sys.executable, str(out/Path(__file__).name), '--worker', str(index), '--out', str(out/f'worker-{index}')],
                    cwd=ROOT, stdout=stdout, stderr=stderr, timeout=65, check=False)
                return {'exit_code': p.returncode, 'timeout': False}
            except subprocess.TimeoutExpired:
                return {'exit_code': None, 'timeout': True}
    with ThreadPoolExecutor(max_workers=2) as pool:
        jobs = dict(zip(('worker-1', 'worker-2'), pool.map(run, (1, 2)), strict=True))
    complete = all(j['exit_code'] == 0 for j in jobs.values())
    if runtime != {n: digest(p) for n, p in _implementation_files().items()}:
        raise ValueError('Runtime changed during constrained fitting')
    results = {}
    for name in jobs:
        file = out/name/'frame-proposals.jsonl'
        if file.exists():
            for line in file.read_text().splitlines():
                row = json.loads(line)
                if row['frame'] in results:
                    raise ValueError('Duplicate source frame proposal')
                results[row['frame']] = row
    complete = complete and sorted(results) == list(range(0, 1108, 2))
    frames = []; gained = []; lost = []; status_counts = Counter(); numerical = Counter(); groups = {}
    for group in ('all_samples', 'length_construction', 'later_active_comparison'):
        groups[group] = {'samples': 0, 'raw_supported_major_joints': 0, 'proposed_supported_major_joints': 0,
            'raw_complete_body_samples': 0, 'proposed_complete_body_samples': 0, 'gained_joints': 0, 'lost_joints': 0}
    for frame in range(1108):
        row = results.get(frame); fit = row['fit'] if row else None
        new_support = set()
        if row:
            status_counts[row['status']] += 1
        if fit:
            numerical[fit['status']] += 1
            if fit['status'] == 'converged' and fit['length_constraints_satisfied']:
                new_support = {j for j, ok in zip(row['joint_indices'], fit['joint_image_limits_satisfied'], strict=True) if ok}
        old_support = {j for j in MAJOR if baseline[frame]['joints'][j]['status'] == 'estimated'}
        proposals = []
        lookup_index = {j: i for i, j in enumerate(row['joint_indices'])} if row else {}
        for j in range(33):
            index = lookup_index.get(j)
            proposal = {'joint': j, 'raw_status': baseline[frame]['joints'][j]['status'], 'point_proposal_m': None,
                'status': 'not_targeted' if j not in MAJOR else 'unsampled' if frame%2 else 'unavailable_two_view_or_solver',
                'numerically_supported': j in new_support}
            if fit and index is not None:
                proposal.update(status='constrained_two_view_proposal' if j in new_support else 'rejected_constraint_proposal',
                    point_proposal_m=fit['point_proposals_m'][index],
                    reprojection_px=[fit['proposed_reprojection_px'][v][index] for v in (0, 1)])
            proposals.append(proposal)
        frames.append({'frame': frame, 'time_s': frame/30, 'source_evidence_ids': baseline[frame]['source_evidence_ids'],
            'fit_status': row['status'] if row else 'unsampled' if frame%2 else 'worker_result_unavailable',
            'joint_proposals': proposals, 'raw_supported_major_joints': len(old_support), 'proposed_supported_major_joints': len(new_support)})
        if frame%2:
            continue
        gain, loss = sorted(new_support-old_support), sorted(old_support-new_support)
        gained.extend({'frame': frame, 'joint': j} for j in gain); lost.extend({'frame': frame, 'joint': j} for j in loss)
        for group in ('all_samples', 'length_construction' if 400 <= frame <= 620 else 'later_active_comparison' if 622 <= frame <= 1058 else None):
            if group is None:
                continue
            g = groups[group]; g['samples'] += 1
            g['raw_supported_major_joints'] += len(old_support); g['proposed_supported_major_joints'] += len(new_support)
            g['raw_complete_body_samples'] += len(old_support) == 12; g['proposed_complete_body_samples'] += len(new_support) == 12
            g['gained_joints'] += len(gain); g['lost_joints'] += len(loss)
    write_json(out/'complete-joint-proposals.json', {'world_axes': 'camera0_x_right_y_down_z_forward', 'frames': frames,
        'original_joint_evidence': str(JOINTS.relative_to(ROOT)/'camera-space-joints.json'), 'floor': None,
        'physical_validation': 'unverified', 'observations_modified': False})
    write_json(out/'all-support-changes.json', {'gained': gained, 'lost': lost})
    times = [r['fit_seconds'] for r in results.values() if r.get('fit_seconds') is not None]
    size = sum(p.stat().st_size for p in out.rglob('*') if p.is_file())
    report = {'created_at': local_time(), 'status': 'complete_source_length_candidate' if complete else 'length_study_failed',
        'jobs': jobs, 'source_frames': len(frames), 'joint_slots': 33*len(frames), 'sample_statuses': dict(status_counts),
        'numerical_statuses': dict(numerical), 'groups': groups, 'no_original_image_support_lost': not lost,
        'per_fit_seconds': {'median': float(np.median(times)), 'p95': float(np.percentile(times, 95)), 'maximum': max(times)} if times else None,
        'runtime_seconds': time.perf_counter()-started, 'output_bytes': size, 'new_video_decodes': 0,
        'new_pose_inference': 0, 'paid_calls': 0, 'reserved_camera_opened': False, 'body_built': False,
        'repair_applied': False, 'runtime_unchanged': True,
        'limitations': ['Constant lengths are imposed, not independently validated anatomy.',
            'Partial or lower reprojection error does not prove correct joint centers, labels, depth, floor or contact.',
            'Source/calibration/clock assumptions and all original missing slots/failures remain.']}
    write_json(out/'report.json', report)
    print(json.dumps({'status': report['status'], 'sample_statuses': report['sample_statuses'], 'numerical_statuses': report['numerical_statuses'],
        'groups': groups, 'runtime_seconds': report['runtime_seconds'], 'per_fit_seconds': report['per_fit_seconds'],
        'verification': seal_suite(out)}))
    if not complete or size > 64000000:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
