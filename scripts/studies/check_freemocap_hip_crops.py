"""Compare six frozen crop proposals with saved original two-view hip observations."""

import argparse
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
from weft.exercise_vision import _detector, _draw_pose, model_provenance, summarize_result
from weft.io import digest, local_time, write_json
from weft.multiview_pose import _observation, _triangulate_rays
from weft.pipeline import _implementation_files
from weft.pose_crop import BODY_JOINTS, CONTRACT, map_crop_pose, plan_crop


ROOT = Path(__file__).resolve().parents[2]
IMAGES = ROOT/'artifacts/freemocap-hip-inspection-01'
POSES = ROOT/'artifacts/freemocap-full-pose-01'
JOINTS = ROOT/'artifacts/freemocap-camera-space-joints-01'
SOURCE = ROOT/'artifacts/freemocap-full-source-01'
MODEL = ROOT/'.local/models/pose_landmarker_full.task'
SELECTED = (416, 550, 966)


def read(path):
    return json.loads(path.read_text())


def verified(root, name):
    path = root/name
    if digest(path) != read(root/'manifest.json')['files'][name]:
        raise ValueError('Frozen crop input changed: '+name)
    return path


def worker(out, view):
    pre = read(out.parent/'pre-run.json')
    if model_provenance(MODEL, variant='full') != pre['model']:
        raise ValueError('Pinned model changed')
    evidence = read(verified(POSES, f'view-{view}/evidence.json'))
    lookup = {r['frame']: r for r in evidence['frames']}
    out.mkdir(exist_ok=False); cv2.setNumThreads(1)
    started = time.perf_counter(); results = []
    mp, detector = _detector(MODEL, 'image')
    try:
        with (out/'proposals.jsonl').open('x') as journal:
            for plan in pre['crop_plans'][str(view)]:
                frame = plan['frame']; image = cv2.imread(str(verified(IMAGES, f'view-{view}-source-{frame:04d}.png')))
                x0, y0, x1, y1 = plan['box_xyxy']; pixels = image[y0:y1, x0:x1]
                if not cv2.imwrite(str(out/f'crop-input-{frame:04d}.png'), pixels):
                    raise ValueError('Crop input export failed')
                tick = time.perf_counter()
                raw = summarize_result(detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB,
                    data=cv2.cvtColor(pixels, cv2.COLOR_BGR2RGB))))
                elapsed = time.perf_counter()-tick
                proposal = map_crop_pose(raw, plan, evidence, lookup[frame], elapsed)
                row = {'frame': frame, 'plan': plan, 'raw_crop_result': raw, 'proposal': proposal}
                journal.write(json.dumps(row, allow_nan=False)+'\n'); journal.flush(); results.append(row)
                _draw_pose(image, proposal).save(out/f'crop-proposal-{frame:04d}.png')
    finally:
        detector.close()
    write_json(out/'proposals-before-comparison.json', results)
    write_json(out/'report.json', {'status': 'three_crops_frozen', 'view': view, 'inference_calls': len(results),
        'runtime_seconds': time.perf_counter()-started, 'new_video_decodes': 0, 'paid_calls': 0})


def hip_error(joints):
    errors = [e for j in (23, 24) for e in (joints[j].get('reprojection_px') or [])]
    return float(np.sqrt(np.mean(np.square(errors)))) if len(errors) == 4 else None


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--worker', type=int, choices=(1, 2)); args = parser.parse_args(); out = args.out.resolve()
    if args.worker:
        worker(out, args.worker); return
    if out.parent != ROOT/'artifacts':
        raise ValueError('Use a new top-level artifact directory')
    evidence = [read(verified(POSES, f'view-{v}/evidence.json')) for v in (1, 2)]
    baseline = read(verified(JOINTS, 'camera-space-joints.json'))['frames']
    poses = [{r['frame']: r for r in view['frames']} for view in evidence]
    plans = {str(v): [plan_crop(evidence[v-1], f) for f in SELECTED] for v in (1, 2)}
    for view in (1, 2):
        for frame in SELECTED:
            verified(IMAGES, f'view-{view}-source-{frame:04d}.png')
    calibration = tomllib.loads(verified(SOURCE, 'member-04-freemocap_sample_data_camera_calibration.toml').read_text())
    cameras = []
    for i in (0, 1):
        c = calibration[f'cam_{i}']; K = np.array(c['matrix']); R = cv2.Rodrigues(np.array(c['rotation']))[0]; t = np.array(c['translation'])*.001
        cameras.append((K, c['distortions'][0], R, t, 720, 1280, np.linalg.inv(K), -t @ R))
    runtime = {n: digest(p) for n, p in _implementation_files().items()}
    out.mkdir(exist_ok=False); shutil.copyfile(__file__, out/Path(__file__).name)
    write_json(out/'pre-run.json', {'created_at': local_time(), 'selected_frames': list(SELECTED),
        'selection': 'Same frozen diagnostic cases as hip inspection: worst crossed-only, standing control, worst raw hip error. Not a representative or independent holdout.',
        'hypothesis': 'A tighter source-supported image crop can improve hip localization under viewpoint/occlusion without label swaps or geometry fitting.',
        'crop_plans': plans, 'crop_contract_unchanged': dict(CONTRACT), 'runtime_hashes': runtime,
        'input_manifests': {p.name: digest(p/'manifest.json') for p in (IMAGES, POSES, JOINTS, SOURCE)},
        'model': model_provenance(MODEL, variant='full'), 'mode': 'IMAGE', 'identity': 'raw',
        'maximum_inference_calls': 6, 'workers': 2, 'worker_timeout_seconds': 60,
        'maximum_output_bytes': 16000000, 'projected_paid_spend_usd': 0,
        'scope': 'Diagnostic proposals only. No complete-source repair, body admission, clock/camera change, default change or threshold relaxation. All original evidence remains.'})
    started = time.perf_counter()
    def run(view):
        with (out/f'view-{view}.stdout').open('w') as stdout, (out/f'view-{view}.stderr').open('w') as stderr:
            try:
                p = subprocess.run([sys.executable, str(out/Path(__file__).name), '--worker', str(view), '--out', str(out/f'view-{view}')],
                    cwd=ROOT, stdout=stdout, stderr=stderr, timeout=60, check=False)
                return {'exit_code': p.returncode, 'timeout': False}
            except subprocess.TimeoutExpired:
                return {'exit_code': None, 'timeout': True}
    with ThreadPoolExecutor(max_workers=2) as pool:
        jobs = dict(zip(('view-1', 'view-2'), pool.map(run, (1, 2)), strict=True))
    complete = all(j['exit_code'] == 0 for j in jobs.values())
    if runtime != {n: digest(p) for n, p in _implementation_files().items()}:
        raise ValueError('Runtime changed during inference')
    comparisons = []; new_rays = 0
    if complete:
        proposed = [{r['frame']: r['proposal'] for r in read(out/f'view-{v}/proposals-before-comparison.json')} for v in (1, 2)]
        for frame in SELECTED:
            revised = []
            for j in range(33):
                observed = [_observation(view[frame], j, 720, 1280) for view in proposed]
                if all(p[0] is not None for p in observed):
                    row = _triangulate_rays([p[0] for p in observed], cameras); new_rays += 1
                else:
                    row = {'status': 'image_evidence_unavailable', 'world_point_m': None, 'reprojection_px': None}
                revised.append(row | {'observation_statuses': [p[1] for p in observed]})
            old = baseline[frame]['joints']
            gained = [j for j in BODY_JOINTS if old[j]['status'] != 'estimated' and revised[j]['status'] == 'estimated']
            lost = [j for j in BODY_JOINTS if old[j]['status'] == 'estimated' and revised[j]['status'] != 'estimated']
            supported = [r for r in baseline if all(r['joints'][j]['status'] == 'estimated' for j in (23, 24))]
            before = [r['frame'] for r in supported if 0 < (frame-r['frame'])/30 <= .2]
            after = [r['frame'] for r in supported if 0 < (r['frame']-frame)/30 <= .2]
            shift = {str(j): [float(np.linalg.norm(np.array([proposed[v][frame]['landmarks'][j][k]-poses[v][frame]['landmarks'][j][k]
                for k in ('x', 'y')])*[720, 1280])) if len(proposed[v][frame]['landmarks']) == 33 else None for v in (0, 1)] for j in (23, 24)}
            comparisons.append({'frame': frame, 'original_joints': old, 'proposed_joints': revised,
                'raw_hip_reprojection_rms_px': hip_error(old), 'crop_hip_reprojection_rms_px': hip_error(revised),
                'raw_major_joint_support': sum(old[j]['status'] == 'estimated' for j in BODY_JOINTS),
                'crop_major_joint_support': sum(revised[j]['status'] == 'estimated' for j in BODY_JOINTS),
                'gained_major_joints': gained, 'lost_major_joints': lost, 'hip_image_displacement_px': shift,
                'root_anchor_frames_within_existing_limit': [max(before) if before else None, min(after) if after else None],
                'proposed_torso_supported': all(revised[j]['status'] == 'estimated' for j in (11, 12, 23, 24)),
                'original_pose_statuses': [view[frame]['status'] for view in poses],
                'crop_pose_statuses': [view[frame]['status'] for view in proposed], 'repair_applied': False})
        write_json(out/'all-selected-comparisons.json', comparisons)
    size = sum(p.stat().st_size for p in out.rglob('*') if p.is_file())
    summary = [{k: r[k] for k in ('frame', 'raw_hip_reprojection_rms_px', 'crop_hip_reprojection_rms_px',
        'raw_major_joint_support', 'crop_major_joint_support', 'gained_major_joints', 'lost_major_joints',
        'proposed_torso_supported', 'root_anchor_frames_within_existing_limit', 'crop_pose_statuses')} for r in comparisons]
    report = {'created_at': local_time(), 'status': 'six_crop_diagnostic_complete' if complete else 'crop_diagnostic_failed',
        'jobs': jobs, 'comparisons': summary, 'new_inference_calls': 6 if complete else None,
        'new_ray_pairs': new_rays, 'new_video_decodes': 0, 'output_bytes': size,
        'runtime_seconds': time.perf_counter()-started, 'runtime_unchanged': True,
        'paid_calls': 0, 'reserved_camera_opened': False, 'repair_applied': False,
        'limitations': ['Selected development cases do not establish a complete sequence repair or anatomical accuracy.',
            'Larger geometric support can still be wrong; frozen originals, camera/floor failures and missing slots remain.',
            'No bilateral swap, root interpolation, floor alignment, body or policy is generated.']}
    write_json(out/'report.json', report)
    print(json.dumps({'status': report['status'], 'comparisons': summary, 'runtime_seconds': report['runtime_seconds'],
        'new_ray_pairs': new_rays, 'verification': seal_suite(out)}))
    if not complete or size > 16000000:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
