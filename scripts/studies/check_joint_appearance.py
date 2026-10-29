"""Grade a frozen experimental association mode against retained development evidence."""

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import shutil

import cv2

from weft.benchmark import seal_suite
from weft.generic_vision import evaluate_manual, track_video
from weft.io import digest, local_time, write_json
from weft.joint_tracking import JOINT_CONTRACT


CASES = [
    ('triple', 'data/balls/triple-collision-240p.webm',
     'artifacts/source-interval-final-08/triple-collision-interval/run/appearance',
     'artifacts/interval-source-grades-01/normalized-labels.json'),
    ('disks', 'data/balls/air-table-collision.webm',
     'artifacts/appearance-guards-e2e-06/air-table-collision/run/appearance',
     'data/balls/air-table-collision.manual.json'),
    ('pool', 'data/balls/quick-pool-break.webm',
     'artifacts/appearance-guards-e2e-06/quick-pool-break/run/appearance',
     'data/balls/quick-pool-break.manual.json'),
    ('electrostatic', 'data/balls/bouncing-ball.webm',
     'artifacts/color-appearance-01/electrostatic',
     'data/general/annotations/electrostatic.manual.json'),
    ('pendulum', 'data/general/elastic-pendulum.webm',
     'artifacts/color-appearance-01/pendulum',
     'data/general/annotations/pendulum.manual.json'),
    ('grayscale', 'data/general/rolling-ball-grayscale.avi',
     'artifacts/source-interval-final-08/rolling-grayscale-interval/run/appearance', None),
]


def accepted(trace):
    return [(row['frame'], row['object_id'], row['x_px'], row['y_px'])
            for row in trace['observations'] if row['status'] == 'tracked']


def run_case(case, out):
    name, video, baseline_dir, labels_path = case
    baseline_path = Path(baseline_dir) / 'appearance-tracks.json'
    config_path = Path(baseline_dir) / 'tracking-config.json'
    baseline = json.loads(baseline_path.read_text())
    config = json.loads(config_path.read_text())
    config['association_mode'] = 'joint'
    candidate = track_video(video, out / name, config)
    inputs = {str(path): digest(path) for path in (video, baseline_path, config_path)}
    report = {'case': name, 'inputs': inputs,
              'candidate_seconds': candidate['seconds'], 'baseline_seconds': baseline['seconds'],
              'candidate_statuses': dict(Counter(row['status'] for row in candidate['observations'])),
              'baseline_statuses': dict(Counter(row['status'] for row in baseline['observations'])),
              'unchanged_accepted_positions': accepted(candidate) == accepted(baseline)}
    if labels_path:
        labels = json.loads(Path(labels_path).read_text())
        inputs[labels_path] = digest(labels_path)
        shutil.copyfile(labels_path, out / name / 'manual-labels.json')
        report['baseline'] = evaluate_manual(baseline, labels)
        report['candidate'] = evaluate_manual(candidate, labels)
    write_json(out / name / 'comparison.json', report)
    return report


def run(out):
    out.mkdir(parents=True, exist_ok=False)
    cv2.setNumThreads(1)
    paths = [Path(__file__), *[Path('src/weft') / name for name in (
        'joint_tracking.py', 'generic_vision.py', 'appearance_consistency.py',
        'scene_vision.py', 'video_interval.py', 'video_io.py')],
        Path('tests/test_joint_tracking.py'), Path('tests/test_generic_vision.py')]
    hashes = {str(path): digest(path) for path in paths}
    for path in paths:
        shutil.copyfile(path, out / path.name)
    write_json(out / 'pre-evaluation-freeze.json', {
        'created_at': local_time(), 'hashes': hashes, 'contract': dict(JOINT_CONTRACT),
        'cases': CASES, 'workers': 2, 'external_api_usd': 0,
        'scope': 'Exposed development labels, not final holdouts; unchanged source intervals, boxes and gates. Only association_mode changes.'})
    with ThreadPoolExecutor(max_workers=2) as pool:
        reports = list(pool.map(lambda case: run_case(case, out), CASES))
    if hashes != {str(path): digest(path) for path in paths}:
        raise ValueError('Source changed during study; preserve evidence but do not promote')
    write_json(out / 'report.json', {'created_at': local_time(), 'cases': reports, 'hashes': hashes,
        'external_api_usd': 0, 'scope': 'Tracking comparison only; no physical identity, reconstruction or transfer claim'})
    print(json.dumps(seal_suite(out)))
    fields = ('accepted_visible_labels', 'visible_labels', 'false_visible_labels',
              'accepted_rmse_source_px', 'accepted_max_error_source_px', 'total_tracked')
    for report in reports:
        print(json.dumps({'case': report['case'], 'unchanged': report['unchanged_accepted_positions'],
            **{side: {key: report[side][key] for key in fields}
               for side in ('baseline', 'candidate') if side in report}}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', required=True, type=Path)
    run(parser.parse_args().out)
