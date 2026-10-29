"""Compare complete-source three-view estimates without changing saved observations."""

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

import cv2
import numpy as np
from PIL import Image, ImageDraw

from weft.benchmark import seal_suite
from weft.exercise_vision import CONNECTIONS
from weft.io import digest, local_time, write_json
from weft.multiray_pose import CONTRACT, triangulate_pixels
from weft.multiview_pose import _observation, _triangulate_rays
from weft.pipeline import _implementation_files


ROOT = Path(__file__).resolve().parents[2]
MAJOR = (11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28)
SEGMENTS = {'shoulder_width': (11, 12), 'hip_width': (23, 24),
    'left_upper_arm': (11, 13), 'right_upper_arm': (12, 14), 'left_forearm': (13, 15),
    'right_forearm': (14, 16), 'left_thigh': (23, 25), 'right_thigh': (24, 26),
    'left_shin': (25, 27), 'right_shin': (26, 28)}
INPUTS = ('freemocap-full-pose-01', 'freemocap-third-comparison-01',
          'freemocap-fixed-length-02', 'freemocap-third-predictions-01', 'freemocap-camera-space-joints-01')


def read(path):
    return json.loads(path.read_text())


def write_large(path, value):
    payload = json.dumps(value, separators=(',', ':'), allow_nan=False).encode()
    existing = sum(p.stat().st_size for p in path.parent.rglob('*') if p.is_file())
    if existing+len(payload) > 96000000:
        raise ValueError('Compact output would exceed the study envelope')
    with path.open('xb') as stream:
        stream.write(payload)


def verified(root, filename):
    path = root/filename
    if digest(path) != read(root/'manifest.json')['files'][filename]:
        raise ValueError('Frozen source changed: '+filename)
    return path


def project(points, camera):
    return cv2.projectPoints(np.array(points, dtype=float), cv2.Rodrigues(np.array(camera['rotation']))[0],
        np.array(camera['translation']), np.array(camera['matrix']),
        np.array([camera['radial'], 0., 0., 0., 0.]))[0].reshape(-1, 2)


def controls():
    views = []
    for center, axis in [([-.65, 0., .1], [0., -.1, .02]),
                         ([.65, 0., .15], [0., .1, -.02]), ([0., -.45, .05], [.08, 0., .01])]:
        R = cv2.Rodrigues(np.array(axis))[0]
        views.append({'matrix': [[800., 0., 360.], [0., 820., 640.], [0., 0., 1.]],
            'radial': -.12, 'rotation': R.tolist(), 'translation': (-R @ center).tolist(),
            'width': 720, 'height': 1280})
    rng = np.random.default_rng(904); rows = []
    for index in range(96):
        truth = rng.uniform([-.35, -.65, 2.4], [.35, .65, 4.5])
        observation = np.array([project(truth[None, :], c)[0] for c in views])+rng.normal(0., .75, (3, 2))
        for cohort in ('independent_noise', 'third_view_conflict', 'shared_3d_bias'):
            pixels = observation.copy()
            if cohort == 'third_view_conflict': pixels[2] += [80., 100.]
            if cohort == 'shared_3d_bias':
                pixels = np.array([project((truth+[.04, 0., 0.])[None, :], c)[0] for c in views])
            rows.append({'index': index, 'cohort': cohort, 'truth_m': truth.tolist(), 'pixels': pixels.tolist()})
    return views, rows


def grade_controls(cameras, rows):
    results = []
    old_cameras = []
    for c in cameras[:2]:
        K = np.array(c['matrix']); R = np.array(c['rotation']); t = np.array(c['translation'])
        old_cameras.append((K, c['radial'], R, t, c['width'], c['height'], np.linalg.inv(K), -t @ R))
    for row in rows:
        baseline = _triangulate_rays(row['pixels'][:2], old_cameras)
        candidate = triangulate_pixels(row['pixels'], cameras)
        errors = {name: float(np.linalg.norm(np.array(value['world_point_m'])-row['truth_m']))
                  if value['status'] == 'estimated' else None for name, value in [('baseline', baseline), ('candidate', candidate)]}
        results.append({'index': row['index'], 'cohort': row['cohort'], 'baseline': baseline,
                        'candidate': candidate, 'true_3d_errors_m': errors})
    return results, summarize_controls(results)


def summarize_controls(results):
    summary = {}
    for name in ('independent_noise', 'third_view_conflict', 'shared_3d_bias'):
        group = [r for r in results if r['cohort'] == name]
        paired = [r['true_3d_errors_m'] for r in group if all(x is not None for x in r['true_3d_errors_m'].values())]
        summary[name] = {'cases': len(group), 'baseline_estimated': sum(r['baseline']['status'] == 'estimated' for r in group),
            'candidate_estimated': sum(r['candidate']['status'] == 'estimated' for r in group),
            'candidate_statuses': dict(Counter(r['candidate']['status'] for r in group)), 'shared_cases': len(paired),
            'shared_3d_rmse_m': {method: float(np.sqrt(np.mean([r[method]**2 for r in paired]))) if paired else None
                                 for method in ('baseline', 'candidate')}}
    return summary


def worker(out, index):
    pre = read(out.parent/'pre-run.json'); prepared = read(out.parent/'prepared-observations.json')
    assert digest(out.parent/'prepared-observations.json') == pre['prepared_sha256']
    out.mkdir(exist_ok=False); started = time.perf_counter(); fits = 0; durations = []
    with (out/'all-sample-estimates.jsonl').open('x') as journal:
        for sample in prepared:
            if (sample['frame']//2) % 2 != index-1:
                continue
            estimates = []
            for row in sample['joints']:
                available = [i for i, p in enumerate(row['pixels']) if p is not None]
                results = {}
                for method, selected in [('two_view_pixel', [0, 1] if all(i in available for i in (0, 1)) else []),
                                         ('all_available_views', available)]:
                    if len(selected) < 2:
                        results[method] = {'status': 'image_evidence_unavailable', 'world_point_m': None,
                                           'used_views': selected, 'view_count': len(selected)}
                        continue
                    tick = time.perf_counter()
                    result = triangulate_pixels([row['pixels'][i] for i in selected], [pre['cameras'][i] for i in selected])
                    durations.append(time.perf_counter()-tick); fits += 1
                    results[method] = result | {'used_views': selected}
                estimates.append({'joint': row['joint'], 'observation_statuses': row['statuses'], **results})
            journal.write(json.dumps({'frame': sample['frame'], 'joints': estimates}, allow_nan=False)+'\n'); journal.flush()
            if time.perf_counter()-started > 60 or (out/'all-sample-estimates.jsonl').stat().st_size > 32000000:
                raise TimeoutError('Worker reached its declared time/output envelope')
    write_json(out/'report.json', {'status': 'assigned_samples_complete', 'fits': fits,
        'runtime_seconds': time.perf_counter()-started, 'per_fit_seconds': {
            'median': float(np.median(durations)), 'p95': float(np.percentile(durations, 95)), 'max': max(durations)}})


def draw_panel(out, frame, rows, cameras, prepared, layout):
    canvas = Image.new('RGB', (1500, 720), 'white'); draw = ImageDraw.Draw(canvas)
    draw.text((12, 10), f'Frame {frame}: all available camera observations; inferred joints, not measured anatomy or a physics body', fill='black')
    draw.text((12, 28), 'Cyan = original image observations; orange = accepted joint projections; missing/conflicting joints are not filled', fill='black')
    frozen = ROOT/'artifacts/freemocap-camera-space-joints-01'
    paths = [frozen/f'view-{i}-source-{frame:04d}.png' for i in (1, 2)]
    paths += [ROOT/'artifacts/freemocap-third-comparison-01'/f'view-3-source-{frame:04d}.png']
    points = {i: np.array(r['all_available_views']['world_point_m']) for i, r in enumerate(rows)
              if i in MAJOR and r['all_available_views']['status'] == 'estimated'}
    for v, path in enumerate(paths):
        im = Image.open(path).convert('RGB'); assert im.size == (720, 1280)
        image = im.resize((340, 604)); canvas.paste(image, (v*340, 80))
        draw.text((v*340+8, 59), f'Cam{v+1}: source observations and projections', fill='black')
        for joint in MAJOR:
            original = prepared['joints'][joint]['pixels'][v]
            if original is not None:
                x, y = np.array(original)*340/720+[v*340, 80]
                draw.ellipse((x-3, y-3, x+3, y+3), outline='cyan', width=2)
            if joint in points:
                x, y = project(points[joint][None, :], cameras[v])[0]*340/720+[v*340, 80]
                if v*340 <= x < (v+1)*340 and 80 <= y < 684:
                    draw.line((x-4, y, x+4, y), fill='orange', width=2)
                    draw.line((x, y-4, x, y+4), fill='orange', width=2)
    transform, center, scale = layout
    screen = {j: (transform @ p-center)*scale+np.array([1260., 370.]) for j, p in points.items()}
    draw.text((1035, 80), f'Camera-space geometry: {len(points)}/12 major joints', fill='black')
    draw.text((1035, 100), 'No floor, gravity alignment, body or muscle estimate', fill='black')
    for a, b in CONNECTIONS:
        if a in screen and b in screen:
            draw.line((*screen[a], *screen[b]), fill='#214F50', width=5)
    for j, (x, y) in screen.items():
        draw.ellipse((x-5, y-5, x+5, y+5), fill='#965644'); draw.text((x+6, y-4), str(j), fill='black')
    canvas.save(out/f'three-view-joints-{frame:04d}.png')


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--recover-fits', type=Path)
    parser.add_argument('--worker', type=int, choices=(1, 2)); args = parser.parse_args(); out = args.out.resolve()
    if args.worker:
        worker(out, args.worker); return
    if out.parent != ROOT/'artifacts': raise ValueError('Use a new top-level artifact directory')
    roots = {n: ROOT/'artifacts'/n for n in INPUTS}
    third_report = read(verified(roots['freemocap-third-comparison-01'], 'report.json'))
    if third_report['status'] != 'frozen_third_view_comparison_complete' or not third_report['third_camera_opened']:
        raise ValueError('Third-view evidence must first finish its frozen comparison')
    views = [read(verified(roots['freemocap-full-pose-01'], f'view-{i}/evidence.json')) for i in (1, 2)]
    views += [read(verified(roots['freemocap-third-comparison-01'], 'pose/evidence.json'))]
    for view in views:
        if (view['frame_count'] != 1108 or view['decoded_frames'] != 1108 or view['fps'] != 30
                or view['width'] != 720 or view['height'] != 1280
                or [r['frame'] for r in view['frames']] != list(range(0, 1108, 2))):
            raise ValueError('Complete packaged source coverage changed')
        for r in view['frames']:
            if r['id'] != f"{view['source_sha256']}:frame:{r['frame']}" or abs(r['time_s']-r['frame']/30) > 1e-7 or abs(r['decoder_timestamp_s']-r['time_s']) > 1e-7:
                raise ValueError('Source observation identity or clock changed')
    cameras = read(verified(roots['freemocap-fixed-length-02'], 'pre-run.json'))['cameras']
    third = read(verified(roots['freemocap-third-predictions-01'], 'protocol.json'))['camera']
    cameras.append({k: third[k] for k in ('matrix', 'radial', 'rotation')} |
                   {'translation': third['translation_m'], 'width': 720, 'height': 1280})
    lookups = [{r['frame']: r for r in v['frames']} for v in views]; prepared = []
    for frame in range(0, 1108, 2):
        joints = []
        for j in range(33):
            obs = [_observation(v[frame], j, 720, 1280) for v in lookups]
            joints.append({'joint': j, 'pixels': [p.tolist() if p is not None else None for p, _ in obs],
                           'statuses': [s for _, s in obs]})
        prepared.append({'frame': frame, 'time_s': frame/30,
            'source_evidence_ids': [v[frame]['id'] for v in lookups], 'joints': joints})
    out.mkdir(exist_ok=False)
    for path in (Path(__file__), ROOT/'src/weft/multiray_pose.py', ROOT/'src/weft/constrained_stereo.py'):
        shutil.copyfile(path, out/path.name)
    runtime = {n: digest(p) for n, p in _implementation_files().items()}
    control_cameras, control_inputs = controls()
    recovery = None; previous = None
    if args.recover_fits:
        previous = args.recover_fits.resolve()
        old = read(verified(previous, 'pre-run.json'))
        if (old['runtime_hashes'] != runtime or old['cameras'] != cameras
                or old['input_manifests'] != {n: digest(p/'manifest.json') for n, p in roots.items()}
                or old['contract'] != json.loads(json.dumps(dict(CONTRACT)))
                or read(verified(previous, 'prepared-observations.json')) != prepared
                or read(verified(previous, 'prepared-controls.json')) != {'cameras': control_cameras, 'cases': control_inputs}
                or read(verified(previous, 'report.json'))['status'] != 'output_size_limit_exceeded_after_completed_fits'):
            raise ValueError('Recovery must reuse exact completed inputs and runtime')
        recovery = {'path': str(previous.relative_to(ROOT)), 'manifest_sha256': digest(previous/'manifest.json'),
                    'new_source_fits': 0, 'new_control_fits': 0}
    write_large(out/'prepared-controls.json', {'cameras': control_cameras, 'cases': control_inputs})
    write_large(out/'prepared-observations.json', prepared)
    write_json(out/'pre-run.json', {'created_at': local_time(), 'contract': dict(CONTRACT), 'cameras': cameras,
        'hypothesis': 'Additional original image observations can constrain depth without fixed limb-length priors; all available views must support every accepted estimate.',
        'input_manifests': {n: digest(p/'manifest.json') for n, p in roots.items()},
        'runtime_hashes': runtime, 'prepared_sha256': digest(out/'prepared-observations.json'),
        'controls_sha256': digest(out/'prepared-controls.json'), 'recovery': recovery, 'maximum_source_frames': 1108,
        'maximum_joint_slots': 36564, 'maximum_source_fits': 36564, 'control_fits': 288,
        'workers': 2, 'worker_timeout_seconds': 65, 'maximum_output_bytes': 96000000,
        'new_inference_calls': 0, 'new_video_decodes': 0, 'projected_paid_spend_usd': 0,
        'third_camera_status': 'Already exposed development evidence; now a construction input, never independent validation.',
        'source_comparison': 'Report every gain/loss, all incomplete joints and two-view versus three-view support. Fitted reprojection and length stability are diagnostics, not accuracy scores.',
        'controls': 'Independent pixel noise; one conflicting camera; coherent 4-cm wrong landmark in every view. Known geometry only, not physical truth.',
        'scope': 'No body, floor, policy, muscle, default integration, threshold change, label swap or temporal interpolation.'})
    started = time.perf_counter()
    def run(index):
        with (out/f'worker-{index}.stdout').open('w') as stdout, (out/f'worker-{index}.stderr').open('w') as stderr:
            try:
                p = subprocess.run([sys.executable, str(out/Path(__file__).name), '--worker', str(index), '--out', str(out/f'worker-{index}')],
                    cwd=ROOT, stdout=stdout, stderr=stderr, timeout=65, check=False)
                return {'exit_code': p.returncode, 'timeout': False}
            except subprocess.TimeoutExpired:
                return {'exit_code': None, 'timeout': True}
    if previous is None:
        control_results, control_summary = grade_controls(control_cameras, control_inputs)
        with ThreadPoolExecutor(max_workers=2) as pool:
            jobs = dict(zip(('worker-1', 'worker-2'), pool.map(run, (1, 2)), strict=True))
    else:
        control_results = read(verified(previous, 'all-control-results.json'))
        control_summary = summarize_controls(control_results)
        jobs = {}
        for name in ('worker-1', 'worker-2'):
            saved = read(verified(previous, name+'/report.json'))
            if saved['status'] != 'assigned_samples_complete': raise ValueError('Cannot reuse an incomplete worker')
            (out/name).mkdir()
            for filename in ('report.json', 'all-sample-estimates.jsonl'):
                shutil.copyfile(verified(previous, name+'/'+filename), out/name/filename)
            jobs[name] = {'exit_code': 0, 'timeout': False, 'reused_completed_results': True}
    write_large(out/'all-control-results.json', control_results)
    if any(j['exit_code'] != 0 for j in jobs.values()):
        write_json(out/'report.json', {'status': 'worker_failure', 'jobs': jobs, 'controls': control_summary,
                                      'body_built': False, 'paid_calls': 0})
        print(json.dumps(seal_suite(out))); raise SystemExit(1)
    samples = {}
    for name in jobs:
        for line in (out/name/'all-sample-estimates.jsonl').read_text().splitlines():
            row = json.loads(line); assert row['frame'] not in samples; samples[row['frame']] = row
    assert sorted(samples) == list(range(0, 1108, 2))
    baseline = read(verified(roots['freemocap-camera-space-joints-01'], 'camera-space-joints.json'))['frames']
    all_rows = []; changes = []; counts = {k: Counter() for k in ('raw_two_view', 'two_view_pixel', 'all_available_views')}
    accepted_views = Counter(); segments = []; timings = [read(out/name/'report.json') for name in jobs]
    for frame in range(1108):
        row = samples.get(frame)
        if row is None:
            row = {'frame': frame, 'joints': [{'joint': j, 'observation_statuses': ['pose_not_sampled']*3,
                **{m: {'status': 'pose_not_sampled', 'world_point_m': None, 'used_views': [], 'view_count': 0}
                   for m in ('two_view_pixel', 'all_available_views')}} for j in range(33)]}
        row['time_s'] = frame/30; row['source_evidence_ids'] = [f"{v['source_sha256']}:frame:{frame}" if frame%2 == 0 else None for v in views]
        for joint, r in enumerate(row['joints']):
            r['raw_two_view'] = baseline[frame]['joints'][joint]
            for method in counts: counts[method][r[method]['status']] += 1
            if r['all_available_views']['status'] == 'estimated': accepted_views[str(r['all_available_views']['used_views'])] += 1
            before = r['raw_two_view']['status'] == 'estimated'
            for method in ('two_view_pixel', 'all_available_views'):
                after = r[method]['status'] == 'estimated'
                changes.append({'frame': frame, 'joint': joint, 'method': method, 'was_supported': before,
                                'is_supported': after, 'change': 'gain' if after and not before else 'loss' if before and not after else 'unchanged'})
        for name, (a, b) in SEGMENTS.items():
            values = {}
            for method in counts:
                ends = [row['joints'][i][method] for i in (a, b)]
                values[method] = float(np.linalg.norm(np.array(ends[0]['world_point_m'])-ends[1]['world_point_m'])) if all(e['status'] == 'estimated' for e in ends) else None
            segments.append({'frame': frame, 'segment': name, 'lengths_m': values})
        all_rows.append(row)
    write_large(out/'complete-joint-estimates.json', {'frames': all_rows, 'world_axes': 'camera_0_x_right_y_down_z_forward',
        'floor': None, 'physical_validation': 'unverified', 'observations_modified': False})
    write_large(out/'all-support-changes.json', changes); write_large(out/'all-segment-slots.json', segments)
    summary = {}
    for method in counts:
        subset = [r for r in changes if r['method'] == method and r['joint'] in MAJOR]
        summary[method] = {'joint_statuses': dict(counts[method]),
            'major_joint_estimates': sum(r['joints'][j][method]['status'] == 'estimated' for r in all_rows for j in MAJOR),
            'complete_major_joint_samples': sum(all(r['joints'][j][method]['status'] == 'estimated' for j in MAJOR) for r in all_rows),
            'major_gains': sum(r['change'] == 'gain' for r in subset), 'major_losses': sum(r['change'] == 'loss' for r in subset)}
    layout_transform = np.array([[.8660254, 0., -.5], [.17101, .939693, .296198]])
    displayed = np.array([layout_transform @ np.array(all_rows[f]['joints'][j]['all_available_views']['world_point_m'])
                         for f in (550, 824) for j in MAJOR if all_rows[f]['joints'][j]['all_available_views']['status'] == 'estimated'])
    low, high = displayed.min(axis=0), displayed.max(axis=0); scale = min(400/max(high[0]-low[0], 1e-6), 510/max(high[1]-low[1], 1e-6))
    for frame in (550, 824):
        if previous is None:
            draw_panel(out, frame, all_rows[frame]['joints'], cameras, prepared[frame//2], (layout_transform, (low+high)/2, scale))
        else:
            shutil.copyfile(verified(previous, f'three-view-joints-{frame:04d}.png'), out/f'three-view-joints-{frame:04d}.png')
    assert runtime == {n: digest(p) for n, p in _implementation_files().items()}
    total_bytes = sum(p.stat().st_size for p in out.rglob('*') if p.is_file())
    if total_bytes > 96000000: raise ValueError('Output size exceeds the declared envelope')
    report = {'created_at': local_time(), 'status': 'complete_three_view_development_comparison', 'source_frames': 1108,
        'joint_slots': 36564, 'methods': summary, 'accepted_view_combinations': dict(accepted_views),
        'controls': control_summary, 'jobs': jobs, 'workers': timings, 'runtime_seconds': time.perf_counter()-started,
        'output_bytes': total_bytes, 'recovery': recovery, 'new_source_fits': 0 if previous else sum(r['fits'] for r in timings),
        'new_control_fits': 0 if previous else len(control_results), 'third_camera_status': 'exposed_construction_evidence',
        'body_built': False, 'candidate_promoted': False, 'floor': None, 'new_pose_inference_calls': 0,
        'new_video_decodes': 0, 'paid_calls': 0, 'physical_validation': 'unverified',
        'limitations': ['Every available qualified view participates; conflicting views are not dropped to manufacture support.',
            'Two-view-only estimates are labeled separately; no single-view depth or missing-sample interpolation.',
            'All three cameras now participate in construction. Their image fit is not an independent accuracy measurement.',
            'Coherent landmark bias cannot be diagnosed by geometric consistency alone; no anatomy/contact/physical claim.']}
    write_json(out/'report.json', report)
    print(json.dumps({'methods': summary, 'controls': control_summary, 'runtime_seconds': report['runtime_seconds'], 'verification': seal_suite(out)}))


if __name__ == '__main__':
    main()
