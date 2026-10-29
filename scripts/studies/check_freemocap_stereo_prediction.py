"""Predict each board corner without using that corner to place the rigid board."""

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
from weft.rigid_correspondence import fit_rigid_points


ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT/'artifacts/freemocap-full-evidence-01'
CHECK = ROOT/'artifacts/freemocap-full-board-check-01'
ORIGINAL = ROOT/'artifacts/freemocap-board-check-01'
SOURCE = ROOT/'artifacts/freemocap-full-source-01'


def read(path):
    return json.loads(path.read_text())


def verified(root, name):
    path = root/name
    if digest(path) != read(root/'manifest.json')['files'][name]:
        raise ValueError('Frozen input changed: '+name)
    return path


def describe(values):
    return {'count': len(values), 'rmse_px': float(np.sqrt(np.mean(np.square(values)))),
            'p95_px': float(np.percentile(values, 95)), 'maximum_px': float(max(values))}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', required=True, type=Path)
    out = parser.parse_args().out.resolve()
    if out.parent != ROOT/'artifacts':
        raise ValueError('Use a new top-level artifacts directory')
    source = read(verified(CHECK, 'full-supplied-all-prefix-comparisons.json'))
    limits = read(verified(ORIGINAL, 'pre-run.json'))['contract']
    calibration = tomllib.loads(verified(SOURCE, 'member-04-freemocap_sample_data_camera_calibration.toml').read_text())
    model = cv2.aruco.CharucoBoard((7, 5), .058, .0464,
        cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)).getChessboardCorners().astype(float)
    out.mkdir(exist_ok=False)
    shutil.copyfile(__file__, out/Path(__file__).name)
    shutil.copyfile(ROOT/'src/weft/rigid_correspondence.py', out/'rigid_correspondence.py')
    write_json(out/'pre-run.json', {
        'created_at': local_time(), 'method': 'leave-one-corner-out-stereo-board-v1',
        'hypothesis': 'A fixed-scale board placed from other stereo corners predicts omitted corners despite an unstable monocular board pose.',
        'input_manifests': {p.name: digest(p/'manifest.json') for p in (EVIDENCE, CHECK, ORIGINAL, SOURCE)},
        'cohort': 'All 500 paired prefix slots, all 18 previously supported poses, all 234 shared corners. Omit each target from placement in turn; no corner selection.',
        'unchanged_numeric_limits': limits,
        'evidence_boundary': 'Development consistency on previously exposed construction evidence, not a fresh holdout or independent calibration truth.',
        'original_check': 'Preserved failed; this different stereo prediction diagnostic does not replace its method, result or scorer.',
        'maximum_seconds': 10, 'maximum_predictions': 234, 'reserved_camera_opened': False,
        'new_decodes': 0, 'new_detector_calls': 0, 'paid_calls': 0,
    })
    started = time.perf_counter()
    predictions, frames = [], []
    for row in source:
        frames.append({'frame': row['frame'], 'time_s': row['time_s'], 'status': row['status']})
        if row['status'] != 'compared':
            continue
        ids = row['source_ids']
        for index, identifier in enumerate(ids):
            training = [i for i in range(len(ids)) if i != index]
            training_ids = [ids[i] for i in training]
            fit = fit_rigid_points(model[training_ids], [row['triangulated_corners_m'][i] for i in training])
            point = model[identifier] @ np.array(fit['rotation']).T+fit['translation']
            pixels, depths = [], []
            for view in (0, 1):
                c = calibration[f'cam_{view}']
                r, t = np.array(c['rotation']), np.array(c['translation'])*.001
                pixels.append(cv2.projectPoints(point[None, :], r, t, np.array(c['matrix']), np.array(c['distortions']))[0].reshape(2).tolist())
                depths.append(float((cv2.Rodrigues(r)[0] @ point+t)[2]))
            predictions.append({'frame': row['frame'], 'time_s': row['time_s'], 'corner_id': identifier,
                'training_ids': training_ids, 'predicted_world_m': point.tolist(),
                'predicted_pixels': pixels, 'depths_m': depths, 'scale_applied': fit['scale_applied']})
            if time.perf_counter()-started > 10:
                raise TimeoutError('Stereo prediction exceeded its work envelope')
    if len(predictions) != 234 or len(frames) != 500:
        raise ValueError('Full comparison coverage changed')
    write_json(out/'predictions-before-grading.json', {'source_frames': frames, 'predictions': predictions})
    prediction_hash = digest(out/'predictions-before-grading.json')
    # Omitted pixels are loaded only after every prediction has been frozen.
    observations = [read(verified(EVIDENCE, f'view-{i}-detections.json')) for i in (1, 2)]
    errors, rows = [[], []], []
    for prediction in predictions:
        row = dict(prediction); row['errors_px'] = []
        for view in (0, 1):
            observed = {p['id']: p['pixel'] for p in observations[view][prediction['frame']]['corners']}[prediction['corner_id']]
            error = float(np.linalg.norm(np.array(prediction['predicted_pixels'][view])-observed))
            errors[view].append(error); row['errors_px'].append(error)
        rows.append(row)
    summaries = [describe(values) for values in errors]
    low, high = limits['camera_depth_bounds_m']
    checks = {
        'complete_original_shared_cohort': len(rows) == 234,
        'enough_board_poses': len({r['frame'] for r in rows}) >= limits['minimum_compared_poses'],
        'all_targets_omitted': all(r['corner_id'] not in r['training_ids'] for r in rows),
        'all_positive_bounded_depths': all(low <= d <= high for r in rows for d in r['depths_m']),
        'camera0_p95': summaries[0]['p95_px'] <= limits['camera0_fit_p95_limit_px'],
        'camera1_p95': summaries[1]['p95_px'] <= limits['cross_view_p95_limit_px'],
        'both_maximum': max(s['maximum_px'] for s in summaries) <= limits['cross_view_max_limit_px'],
    }
    if digest(out/'predictions-before-grading.json') != prediction_hash:
        raise ValueError('Predictions changed during grading')
    write_json(out/'graded-predictions.json', rows)
    report = {
        'created_at': local_time(), 'status': 'development_stereo_prediction_passed' if all(checks.values()) else 'development_stereo_prediction_failed',
        'checks': checks, 'views': summaries, 'source_slots': 500, 'compared_poses': 18,
        'unavailable_poses': 482, 'predictions': 234, 'prediction_sha256': prediction_hash,
        'runtime_seconds': time.perf_counter()-started, 'original_monocular_check_still_failed': True,
        'calibration_changed': False, 'scale_changed': False, 'reserved_camera_opened': False,
        'new_decodes': 0, 'new_detector_calls': 0, 'paid_calls': 0,
        'limitations': ['Omitted corners are unseen by each placement fit, but the recording and camera estimates are not independent physical truth.',
            'This does not verify ground/gravity, anatomy, changed actions or all source-frame coverage.',
            'Neither a default calibration promotion nor a waiver of the original failed monocular check.'],
    }
    write_json(out/'report.json', report)
    print(json.dumps({'status': report['status'], 'checks': checks, 'views': summaries,
        'runtime_seconds': report['runtime_seconds'], 'verification': seal_suite(out)}))


if __name__ == '__main__':
    main()
