"""Retain both same-side and crossed stereo joint assignments without choosing repairs."""

import argparse
from collections import Counter
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
PAIRS = {'shoulders': (11, 12), 'elbows': (13, 14), 'wrists': (15, 16),
         'hips': (23, 24), 'knees': (25, 26), 'ankles': (27, 28)}


def read(path):
    return json.loads(path.read_text())


def verified(root, name):
    path = root/name
    if digest(path) != read(root/'manifest.json')['files'][name]:
        raise ValueError('Frozen input changed: '+name)
    return path


def error(result):
    values = [e for r in result for e in (r.get('reprojection_px') or [])]
    return float(np.sqrt(np.mean(np.square(values)))) if len(values) == 4 else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', required=True, type=Path)
    out = parser.parse_args().out.resolve()
    if out.parent != ROOT/'artifacts':
        raise ValueError('Use a new top-level artifact directory')
    baseline = read(verified(JOINTS, 'camera-space-joints.json'))['frames']
    views = [read(verified(POSES, f'view-{i}/evidence.json')) for i in (1, 2)]
    lookup = [{r['frame']: r for r in v['frames']} for v in views]
    supplied = tomllib.loads(verified(SOURCE, 'member-04-freemocap_sample_data_camera_calibration.toml').read_text())
    cameras = []
    for i in (0, 1):
        c = supplied[f'cam_{i}']; K = np.array(c['matrix']); R = cv2.Rodrigues(np.array(c['rotation']))[0]; t = np.array(c['translation'])*.001
        cameras.append((K, c['distortions'][0], R, t, 720, 1280, np.linalg.inv(K), -t @ R))
    if len(baseline) != 1108 or any(len(v['frames']) != 554 for v in views):
        raise ValueError('Complete construction interval changed')
    out.mkdir(exist_ok=False)
    shutil.copyfile(__file__, out/Path(__file__).name)
    write_json(out/'pre-run.json', {'created_at': local_time(), 'input_manifests': {p.name: digest(p/'manifest.json') for p in (POSES, JOINTS, SOURCE)},
        'hypothesis': 'Cross-camera left/right disagreement may explain some inconsistent joint rays; compare fixed alternatives before considering a repair.',
        'cohort': 'All 554 sampled moments and all six bilateral pairs; compare alternatives only when all four original 2D observations are available.',
        'alternatives': ['Original same-side associations, reusing saved ray results', 'Crossed Cam2 associations; no coordinate movement'],
        'admission': 'No repair or automatic hypothesis selection. Smaller reprojection error alone cannot identify anatomical side or physical correctness.',
        'all_source_frames': 1108, 'maximum_new_ray_pairs': 554*6*2, 'maximum_seconds': 15,
        'new_inference_calls': 0, 'new_video_decodes': 0, 'reserved_camera_opened': False, 'paid_calls': 0})
    started = time.perf_counter(); rows = []; counters = {name: Counter() for name in PAIRS}; new_pairs = 0
    for frame in range(0, 1108, 2):
        for name, pair in PAIRS.items():
            observations = [[_observation(view.get(frame), j, 720, 1280) for j in pair] for view in lookup]
            original = [baseline[frame]['joints'][j] for j in pair]
            row = {'frame': frame, 'time_s': frame/30, 'pair': name, 'indices': list(pair),
                'original': original, 'observation_statuses': [[p[1] for p in view] for view in observations]}
            if any(p[0] is None for view in observations for p in view):
                row.update(status='incomplete_common_observations', crossed=None)
                counters[name]['incomplete_common_observations'] += 1
            else:
                crossed = [_triangulate_rays([observations[0][side][0], observations[1][side ^ 1][0]], cameras) for side in (0, 1)]
                new_pairs += 2
                old_ok = all(r['status'] == 'estimated' for r in original)
                new_ok = all(r['status'] == 'estimated' for r in crossed)
                old_error, new_error = error(original), error(crossed)
                result = 'both_supported' if old_ok and new_ok else 'raw_only_supported' if old_ok else 'cross_only_supported' if new_ok else 'neither_supported'
                counters[name][result] += 1
                if old_error is not None and new_error is not None:
                    counters[name]['cross_lower_reprojection' if new_error < old_error else 'raw_lower_or_equal_reprojection'] += 1
                row.update(status='compared', crossed=crossed, support=result,
                    original_reprojection_rmse_px=old_error, crossed_reprojection_rmse_px=new_error,
                    original_pair_separation_m=None if not old_ok else float(np.linalg.norm(np.subtract(*[r['world_point_m'] for r in original]))),
                    crossed_pair_separation_m=None if not new_ok else float(np.linalg.norm(np.subtract(*[r['world_point_m'] for r in crossed]))))
            rows.append(row)
        if time.perf_counter()-started > 15:
            raise TimeoutError('Joint-association diagnostic exceeded its work envelope')
    write_json(out/'all-sampled-pair-alternatives.json', rows)
    hip_rows = [r for r in rows if r['pair'] == 'hips' and r['status'] == 'compared' and 400 <= r['frame'] < 1060]
    worst = max(hip_rows, key=lambda r: r['original_reprojection_rmse_px'] or 0.)
    crossed_help = sorted([r for r in hip_rows if r['support'] == 'cross_only_supported'],
        key=lambda r: (r['original_reprojection_rmse_px'] or 0.)-(r['crossed_reprojection_rmse_px'] or 0.), reverse=True)
    report = {'created_at': local_time(), 'status': 'association_diagnostic_complete_no_repair',
        'sampled_pair_slots': len(rows), 'all_source_frames_preserved_in_parent': 1108, 'new_ray_pairs': new_pairs,
        'counts_by_pair': {name: dict(c) for name, c in counters.items()},
        'worst_active_hip_frame': worst['frame'], 'cross_only_active_hip_frames': [r['frame'] for r in crossed_help],
        'runtime_seconds': time.perf_counter()-started, 'repair_applied': False, 'new_inference_calls': 0,
        'new_video_frames_decoded': 0, 'reserved_camera_opened': False, 'paid_calls': 0,
        'limitations': ['Ray agreement does not identify correct anatomical labels, joint centers or hidden depth.',
            'A crossed pair can have plausible rays but incorrect shape; no available-case improvement promotes it.',
            'Complete interval, original hypotheses and all unavailable pair slots remain unchanged.']}
    write_json(out/'report.json', report)
    print(json.dumps({'counts': report['counts_by_pair'], 'worst_active_hip_frame': worst['frame'],
        'cross_only_active_hip_frames': report['cross_only_active_hip_frames'], 'runtime_seconds': report['runtime_seconds'], 'verification': seal_suite(out)}))


if __name__ == '__main__':
    main()
