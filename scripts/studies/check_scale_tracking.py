"""Frozen opt-in scale tracking on two existing real sources, never a physics fit."""

from collections import Counter
import json
from pathlib import Path
import shutil
import signal
import time

from weft.benchmark import seal_suite
from weft.generic_vision import evaluate_manual, track_video
from weft.io import digest, local_time, write_json
from weft.pipeline import verify_artifacts
from weft.scale_tracking import SCALE_CONTRACT


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'artifacts/scale-tracking-01'


def main():
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError('90-second study bound')))
    signal.alarm(90)
    OUT.mkdir(exist_ok=False)
    files = ['src/weft/scale_tracking.py', 'src/weft/generic_vision.py', 'src/weft/appearance_consistency.py',
             'src/weft/scene_vision.py', 'src/weft/video_interval.py', 'src/weft/video_io.py',
             'tests/test_scale_tracking.py', 'tests/test_generic_vision.py', 'scripts/studies/check_scale_tracking.py']
    hashes = {name: digest(ROOT / name) for name in files}
    for name in files:
        path = OUT / 'implementation' / name
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, path)
    cases = {
        'cylinder': {'video': 'data/general/bicone-cylinder-240p.webm',
                     'bundle': 'artifacts/general-cylinder-tracking-01',
                     'tracks': 'tracking/appearance-tracks.json', 'config': 'tracking/tracking-config.json',
                     'labels': 'artifacts/general-cylinder-source-review-01/manual.json'},
        'toy-van': {'video': 'data/general/corgi-hand-crank-van.webm',
                    'bundle': 'artifacts/general-object-e2e-01',
                    'tracks': 'lab/run/appearance/appearance-tracks.json',
                    'config': 'lab/run/appearance/tracking-config.json', 'labels': 'data/general/corgi-hand-crank-van.manual.json'},
    }
    write_json(OUT / 'pre-run.json', {'created_at': local_time(), 'contract': SCALE_CONTRACT, 'cases': cases,
        'analytic_tests': {'scale': 15, 'generic_existing': 27}, 'implementation_hashes': hashes,
        'changes': 'Only scale_mode=pyramid; original thresholds, intervals, IDs and labels retained',
        'maximum_seconds': 90, 'new_api_calls': 0, 'cost_usd': 0,
        'scope': 'Known development sources; fixed-size baselines reused, no tuning or environment promotion'})
    summaries = {}
    for name, case in cases.items():
        out = OUT / name
        out.mkdir()
        baseline_root = ROOT / case['bundle']
        verification = verify_artifacts(baseline_root)
        paths = {'config': baseline_root / case['config'], 'baseline': baseline_root / case['tracks'],
                 'labels': ROOT / case['labels']}
        input_hashes = {key: digest(path) for key, path in paths.items()}
        for key, path in paths.items():
            shutil.copyfile(path, out / (key + '.json'))
        config = json.loads(paths['config'].read_text()) | {'scale_mode': 'pyramid'}
        labels = json.loads(paths['labels'].read_text())
        if name == 'cylinder':
            labels['annotations'] = [row for row in labels['annotations'] if row['phase'] != 'initialization']
        else:
            labels = {'source_sha256': labels['source_sha256'], 'coordinate_system': 'original_source_pixels',
                'annotations': [{'frame': row['frame'], 'object_id': labels['object_id'],
                    'visible': row['center_px'] is not None, 'center_xy': row['center_px'],
                    'tolerance_px': labels['center_uncertainty_radius_px'], 'source_status': row['visibility']}
                    for row in labels['observations'] if row['frame'] != config['initialization_frame']]}
        write_json(out / 'grading-labels.json', labels)
        write_json(out / 'input-verification.json', {'baseline': verification, 'hashes': input_hashes})
        start = time.monotonic()
        candidate = track_video(ROOT / case['video'], out / 'tracking', config)
        baseline = json.loads(paths['baseline'].read_text())
        grades = {key: evaluate_manual(tracks, labels) for key, tracks in [('baseline', baseline), ('candidate', candidate)]}
        for grade in grades.values():
            grade['within_source_tolerance'] = sum(row.get('error_source_px', float('inf')) <= row['tolerance_px'] for row in grade['rows'])
            grade['ambiguous_source_returned'] = sum(row.get('source_status') == 'ambiguous_source_crossfade' and row['status'] == 'tracked' for row in grade['rows'])
            grade['scope'] += '; ambiguous crossfade is not verified physical absence; interpret returned ambiguous slots separately'
        write_json(out / 'grades.json', grades)
        assert all(digest(path) == input_hashes[key] for key, path in paths.items())
        summaries[name] = {'seconds': time.monotonic() - start,
            'baseline': {key: value for key, value in grades['baseline'].items() if key not in ('rows', 'scope')},
            'candidate': {key: value for key, value in grades['candidate'].items() if key not in ('rows', 'scope')},
            'candidate_statuses': dict(Counter(row['status'] for row in candidate['observations'])),
            'decoded_frames': candidate['decoded_frames'], 'inputs_unchanged': True, 'promoted': False}
        print(json.dumps({name: summaries[name]}), flush=True)
    assert hashes == {name: digest(ROOT / name) for name in files}
    write_json(OUT / 'report.json', {'created_at': local_time(), 'cases': summaries, 'implementation_unchanged': True,
        'new_api_calls': 0, 'cost_usd': 0, 'scope': 'Image observations only; no physical shape/depth/forces, simulator or learned policy recovered'})
    signal.alarm(0)
    print(json.dumps(seal_suite(OUT)), flush=True)


if __name__ == '__main__':
    main()
