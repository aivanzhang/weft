"""Compare one predeclared local model across complete retained development clips."""

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np

from weft.benchmark import seal_suite
from weft.errors import UnsupportedVideo
from weft.exercise_vision import inspect_exercise, model_provenance
from weft.io import digest, local_time, write_json
from weft.pipeline import _implementation_files, verify_artifacts


ROOT = Path(__file__).resolve().parents[2]
MODEL = ROOT/'.local/models/pose_landmarker_full.task'
CASES = {
    'jog-drone1': ('artifacts/multiview-jog-source-01/drone1.mp4', 'artifacts/multiview-jog-image-01/run', 'image', 'raw'),
    'jog-drone2': ('artifacts/multiview-jog-extra-views-02/drone2.mp4', 'artifacts/multiview-jog-drone2-image-01', 'image', 'raw'),
    'fullbody-squat': ('data/exercise/bodyweight-squats-danielflefil-500.mp4', 'artifacts/fullbody-exercise-01/baseline/run', 'video', 'temporal'),
    'pushup': ('data/exercise/kettlebell-pushup.webm', 'artifacts/mobile-exercise-e2e-10/kettlebell-pushup/run', 'video', 'temporal'),
}
JOINTS = (11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28)


def read(path):
    return json.loads(path.read_text())


def image_point(row, joint, width, height):
    if row is None or row.get('pose_count') != 1 or len(row.get('landmarks', [])) != 33: return None
    point = row['landmarks'][joint]
    if not (0 <= point['x'] <= 1 and 0 <= point['y'] <= 1 and .5 <= min(point['presence'], point['visibility'])
            and max(point['presence'], point['visibility']) <= 1): return None
    return np.array([point['x']*width, point['y']*height])


def worker(name, out):
    video_name, baseline_name, mode, identity = CASES[name]
    video, baseline = ROOT/video_name, ROOT/baseline_name
    pinned = read(out.parent/'pre-run.json')['cases'][name]
    if digest(video) != pinned['video_sha256'] or verify_artifacts(baseline) != pinned['baseline_verification']:
        raise ValueError('Candidate source differs from predeclared study')
    out.mkdir(exist_ok=False)
    write_json(out/'pre-run.json', {'created_at': local_time(), 'case': name, 'inputs': pinned,
        'model': model_provenance(MODEL, variant='full'), 'parent_plan_sha256': digest(out.parent/'pre-run.json'),
        'maximum_inference_seconds': 180, 'maximum_process_seconds': 210, 'paid_requests': 0})
    tick = time.monotonic(); error = None
    try:
        evidence = inspect_exercise(video, out, MODEL, pose_mode=mode, pose_identity=identity)
        status = 'inference_completed'
    except UnsupportedVideo as failure:
        evidence = read(out/'evidence.json'); error = str(failure)
        status = 'raw_evidence_retained_legacy_gate_rejected'
    report = {'created_at': local_time(), 'case': name, 'status': status, 'legacy_error': error,
              'pose_statuses': dict(Counter(row['status'] for row in evidence['frames'])),
              'decoded_frames': evidence['decoded_frames'], 'sampled_frames': len(evidence['frames']),
              'quality': evidence['quality'], 'runtime_seconds': time.monotonic()-tick, 'paid_requests': 0,
              'body_built': False, 'policy_trained': False}
    if digest(video) != pinned['video_sha256']: raise RuntimeError('Source video changed during inference')
    write_json(out/'report.json', report)
    print(json.dumps({'report': report, 'verification': seal_suite(out)}, indent=2))


def compare(old, new):
    if any(old[key] != new[key] for key in ('source_sha256', 'fps', 'width', 'height', 'decoded_frames')):
        raise ValueError('Model comparison requires identical source, decode and dimensions')
    if [(r['id'], r['frame'], r['time_s'], r['decoder_timestamp_s']) for r in old['frames']] != [
            (r['id'], r['frame'], r['time_s'], r['decoder_timestamp_s']) for r in new['frames']]:
        raise ValueError('Model comparison sample identities or clocks differ')
    indexed = [{row['frame']: row for row in evidence['frames']} for evidence in (old, new)]
    rows, availability, differences = [], Counter(), []
    for frame in range(old['decoded_frames']):
        for joint in JOINTS:
            first, second = [image_point(source.get(frame), joint, old['width'], old['height']) for source in indexed]
            kind = ('both' if first is not None else 'full_only') if second is not None else ('lite_only' if first is not None else 'neither')
            distance = float(np.linalg.norm(first-second)) if first is not None and second is not None else None
            if distance is not None: differences.append(distance)
            availability[kind] += 1
            rows.append({'frame': frame, 'joint': joint, 'availability': kind, 'between_model_image_distance_px': distance})
    model_timing = {}
    for label, evidence in [('lite_historical', old), ('full_current', new)]:
        timing = [row['inference_seconds'] for row in evidence['frames']]
        model_timing[label] = {'median_seconds': float(np.median(timing)), 'p95_seconds': float(np.percentile(timing, 95)),
                              'total_inference_stage_seconds': evidence['elapsed_seconds'],
                              'pose_statuses': dict(Counter(row['status'] for row in evidence['frames'])),
                              'quality_passes': evidence['quality']['passes'], 'identity': evidence.get('pose_identity')}
    return {'decoded_frames': old['decoded_frames'], 'sampled_frames': len(old['frames']), 'joint_slots': len(rows),
        'availability': dict(availability), 'timing': model_timing,
        'between_model_image_distance_px': {'count': len(differences), 'mean': float(np.mean(differences)) if differences else None,
                                          'maximum': max(differences) if differences else None},
        'scope': 'Availability and disagreement, not accuracy; timing compares historical runs to bounded concurrent current workers', 'rows': rows}


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--worker', choices=CASES); args = parser.parse_args()
    out = args.out.resolve()
    if args.worker: worker(args.worker, out); return
    out.mkdir(parents=True, exist_ok=False)
    implementation = _implementation_files()
    hashes = {name: digest(path) for name, path in implementation.items()}
    for name, source in implementation.items():
        target = out/'implementation'/name; target.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(source, target)
    shutil.copyfile(__file__, out/Path(__file__).name)
    shutil.copyfile(ROOT/'tests/test_pose_models.py', out/'test_pose_models.py')
    shutil.copyfile(ROOT/'.local/full-pose-model-tests.xml', out/'model-tests.xml')
    cases = {}
    for name, (video, baseline, mode, identity) in CASES.items():
        saved = read(ROOT/baseline/'evidence.json')
        checksum = digest(ROOT/video)
        if saved['source_sha256'] != checksum or saved['model']['running_mode'].lower() != mode or saved['pose_identity']['mode'] != identity:
            raise ValueError('Comparison settings do not match saved baseline')
        cases[name] = {'video': video, 'video_sha256': checksum, 'baseline': baseline,
                       'baseline_verification': verify_artifacts(ROOT/baseline), 'mode': mode, 'identity': identity}
    write_json(out/'pre-run.json', {'created_at': local_time(), 'cases': cases, 'implementation': hashes,
        'model': model_provenance(MODEL, variant='full'), 'workers': 2, 'process_seconds_per_case': 210,
        'candidate': 'Exactly one predeclared Full model, same modes/sample clocks/identity settings as Lite baselines',
        'no_auto_retry_or_best_frame_selection': True, 'paid_requests': 0, 'projected_external_cost_usd': 0,
        'scope': 'Four complete development inputs, three movement types; no final holdout, anatomy or dynamics validation',
        'excluded_from_inference': ['camera calibration', 'provider body joints', 'drone3 comparison images', 'floor scores', 'manual labels'],
        'default_change': False, 'new_body_or_policy': False})
    started = time.monotonic()

    def launch(name):
        with (out/f'{name}.log').open('x') as stream:
            try:
                process = subprocess.run([sys.executable, str(Path(__file__).resolve()), '--worker', name, '--out', str(out/name)],
                                         cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, timeout=210)
                return {'case': name, 'exit_code': process.returncode}
            except subprocess.TimeoutExpired:
                return {'case': name, 'exit_code': 124, 'status': 'process_deadline_exceeded_partial_evidence_retained'}
    with ThreadPoolExecutor(max_workers=2) as executor: outcomes = list(executor.map(launch, CASES))
    results = {}
    for outcome in outcomes:
        name = outcome['case']
        if outcome['exit_code'] != 0: results[name] = outcome; continue
        verify_artifacts(out/name)
        result = compare(read(ROOT/CASES[name][1]/'evidence.json'), read(out/name/'evidence.json'))
        write_json(out/f'{name}-comparison.json', result)
        results[name] = {key: value for key, value in result.items() if key != 'rows'}
    if implementation != _implementation_files() or any(digest(path) != hashes[name] for name, path in implementation.items()):
        raise RuntimeError('Implementation changed while candidate workers were active')
    for value in cases.values():
        if digest(ROOT/value['video']) != value['video_sha256'] or verify_artifacts(ROOT/value['baseline']) != value['baseline_verification']:
            raise RuntimeError('Baseline/source changed during comparison')
    if model_provenance(MODEL, variant='full') != read(out/'pre-run.json')['model']: raise RuntimeError('Model changed during comparison')
    report = {'created_at': local_time(), 'outcomes': outcomes, 'results': results, 'runtime_seconds': time.monotonic()-started,
              'paid_requests': 0, 'default_change': False, 'new_body_or_policy': False}
    write_json(out/'report.json', report)
    print(json.dumps({'report': report, 'verification': seal_suite(out)}, indent=2))
    if any(row['exit_code'] != 0 for row in outcomes): raise SystemExit(1)


if __name__ == '__main__':
    main()
