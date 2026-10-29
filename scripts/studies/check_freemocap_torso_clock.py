"""Compare clock offsets on a fixed, fully observed torso cohort without retiming."""

import argparse
import json
from pathlib import Path
import shutil
import time
import tomllib

import cv2
import numpy as np

from weft.benchmark import seal_suite
from weft.io import digest, local_time, write_json
from weft.multiview_pose import _observation, _triangulate_rays


ROOT = Path(__file__).resolve().parents[2]
POSES = ROOT/'artifacts/freemocap-full-pose-01'
JOINTS = ROOT/'artifacts/freemocap-camera-space-joints-01'
SOURCE = ROOT/'artifacts/freemocap-full-source-01'
OFFSETS = [-6, -4, -2, 0, 2, 4, 6]
FRAMES = list(range(406, 1054, 2))
TORSO = (11, 12, 23, 24)


def read(path):
    return json.loads(path.read_text())


def verified(root, name):
    path = root/name
    if digest(path) != read(root/'manifest.json')['files'][name]:
        raise ValueError('Frozen input changed: '+name)
    return path


def summarize(rows):
    values = [e for r in rows for e in r['reprojection_px']]
    return {'joint_slots': len(rows), 'estimated_slots': sum(r['status'] == 'estimated' for r in rows),
        'rmse_px': float(np.sqrt(np.mean(np.square(values)))), 'median_px': float(np.median(values)),
        'p95_px': float(np.percentile(values, 95)), 'maximum_px': max(values)}


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--out', type=Path, required=True)
    out = parser.parse_args().out.resolve()
    if out.parent != ROOT/'artifacts':
        raise ValueError('Use a new top-level artifact directory')
    baseline = read(verified(JOINTS, 'camera-space-joints.json'))['frames']
    views = [{r['frame']: r for r in read(verified(POSES, f'view-{i}/evidence.json'))['frames']} for i in (1, 2)]
    supplied = tomllib.loads(verified(SOURCE, 'member-04-freemocap_sample_data_camera_calibration.toml').read_text())
    cameras = []
    for i in (0, 1):
        c = supplied[f'cam_{i}']; K = np.array(c['matrix']); R = cv2.Rodrigues(np.array(c['rotation']))[0]; t = np.array(c['translation'])*.001
        cameras.append((K, c['distortions'][0], R, t, 720, 1280, np.linalg.inv(K), -t @ R))
    out.mkdir(exist_ok=False); shutil.copyfile(__file__, out/Path(__file__).name)
    write_json(out/'pre-run.json', {'created_at': local_time(), 'input_manifests': {p.name: digest(p/'manifest.json') for p in (POSES, JOINTS, SOURCE)},
        'hypothesis': 'A temporal mismatch rather than low confidence or a side-label swap may explain persistent active-sequence torso ray disagreement.',
        'frames': FRAMES, 'joints': list(TORSO), 'cam2_frame_offsets': OFFSETS, 'fps': 30,
        'positive_offset': 'Compare Cam1 frame f with Cam2 frame f+offset; no source-clock correction is applied.',
        'fixed_cohort': '324 Cam1 sampled moments times four torso joints. Every candidate must retain all 1,296 observed joint pairs; no smaller-subset winner.',
        'blocks': [[406, 620], [622, 836], [838, 1052]],
        'maximum_new_ray_pairs': 7776, 'maximum_seconds': 15,
        'no_retiming': True, 'no_interpolation': True, 'new_video_decodes': 0, 'new_pose_inference': 0,
        'reserved_camera_opened': False, 'paid_calls': 0,
        'evidence_boundary': 'Previously exposed construction image estimates. Better ray agreement alone is not verified synchronization or anatomy.'})
    started = time.perf_counter(); rows = []; summaries = []
    for offset in OFFSETS:
        compared = []
        for sample, frame in enumerate(FRAMES):
            for joint in TORSO:
                pixels = [_observation(views[i].get(frame+(offset if i else 0)), joint, 720, 1280)[0] for i in (0, 1)]
                if any(p is None for p in pixels):
                    raise ValueError('The fixed observed torso cohort is incomplete; do not compare a smaller subset')
                estimate = baseline[frame]['joints'][joint] if offset == 0 else _triangulate_rays(pixels, cameras)
                if estimate.get('reprojection_px') is None:
                    raise ValueError('The fixed torso cohort has an unscorable ray pair')
                old = baseline[frame]['joints'][joint]['reprojection_px']
                compared.append({'source_frame': frame, 'source_time_s': frame/30, 'cam2_frame': frame+offset,
                    'cam2_time_s': (frame+offset)/30, 'joint': joint, 'offset_frames': offset,
                    'block': sample//108, 'status': estimate['status'], 'reprojection_px': estimate['reprojection_px'],
                    'baseline_reprojection_px': old,
                    'paired_error_worsened': bool(np.mean(np.square(estimate['reprojection_px'])) > np.mean(np.square(old))+1e-10)})
            if time.perf_counter()-started > 15:
                raise TimeoutError('Fixed-cohort torso clock diagnostic exceeded its work envelope')
        if len(compared) != 1296:
            raise ValueError('Fixed cohort size changed')
        summaries.append({'offset_frames': offset, 'offset_seconds': offset/30, 'all': summarize(compared),
            'blocks': [summarize([r for r in compared if r['block'] == block]) for block in range(3)],
            'paired_regressions': sum(r['paired_error_worsened'] for r in compared)})
        rows.extend(compared)
    write_json(out/'all-offset-joint-slots.json', rows)
    report = {'created_at': local_time(), 'status': 'torso_clock_diagnostic_complete_no_retiming', 'results': summaries,
        'joint_slots_each_offset': 1296, 'all_slots': len(rows), 'new_ray_pairs': 7776,
        'best_rmse_offset_each_block': [min(summaries, key=lambda r: r['blocks'][block]['rmse_px'])['offset_frames'] for block in range(3)],
        'runtime_seconds': time.perf_counter()-started, 'retiming_applied': False, 'new_inference_calls': 0,
        'new_video_decodes': 0, 'reserved_camera_opened': False, 'paid_calls': 0,
        'limitations': ['Changing anatomical image estimates can mimic temporal disagreement.',
            'Only torso consistency is tested; no body, floor, physical calibration or synchronization is promoted.',
            'Every offset retains the same complete observation cohort and the original source timestamps.']}
    write_json(out/'report.json', report)
    print(json.dumps({'summaries': [{'offset': r['offset_frames'], 'rmse': r['all']['rmse_px'],
        'estimated_slots': r['all']['estimated_slots'], 'regressions': r['paired_regressions'],
        'block_rmse': [b['rmse_px'] for b in r['blocks']]} for r in summaries],
        'runtime_seconds': report['runtime_seconds'], 'verification': seal_suite(out)}))


if __name__ == '__main__':
    main()
