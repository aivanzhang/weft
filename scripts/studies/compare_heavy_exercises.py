"""Qualify one larger pose model across the complete five-exercise cohort."""

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import json
import os
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
from weft.limb_fidelity import grade_limb_projection, load_annotations
from weft.pipeline import _implementation_files, verify_artifacts
from weft.root_motion import build_root_reference
from check_crop_diversity import CLIPS, LIMB_LABELS, PAIR_LABELS, deltas, limb_predictions
from compare_pose_models import JOINTS, image_point
from grade_pose_identity import grade as grade_identity


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / 'artifacts/selective-crop-videos-03'
MODEL_SOURCE = ROOT / 'artifacts/heavy-pose-source-02'
MODEL = ROOT / '.local/models/pose_landmarker_heavy.task'


def read(path):
    return json.loads(path.read_text())


def worker(out, clip):
    pre = read(out / 'pre-run.json')
    case = pre['cases'][clip]
    video = ROOT / case['video']
    if digest(video) != case['source_sha256'] or model_provenance(MODEL) != pre['model']:
        raise ValueError('Pinned input or model changed')
    target = out / clip
    target.mkdir()
    started = time.monotonic()
    quality_error = None
    try:
        evidence = inspect_exercise(video, target, MODEL, pose_mode='image', pose_identity='raw')
    except UnsupportedVideo as error:
        quality_error = str(error)
        evidence = read(target / 'evidence.json')
    baseline = read(SOURCE / clip / 'whole_image/evidence.json')
    clocks = lambda e: [(row['id'], row['frame'], row['time_s'], row['decoder_timestamp_s']) for row in e['frames']]
    if (clocks(evidence) != clocks(baseline)
            or any(evidence[key] != baseline[key] for key in ('decoded_frames', 'source_sha256', 'fps', 'width', 'height'))):
        raise ValueError('Candidate must retain the complete matched source and sampling')
    write_json(target / 'pose-quality.json', evidence['quality'])
    audit = build_root_reference(evidence)
    write_json(target / 'root-motion.json', audit)
    times = [row['inference_seconds'] for row in evidence['frames']]
    report = {'created_at': local_time(), 'status': 'complete_heavy_image_inference', 'clip': clip,
        'decoded_frames': evidence['decoded_frames'], 'sampled_frames': len(evidence['frames']),
        'usable': evidence['quality']['tracked_frames'], 'quality_error': quality_error,
        'pose_statuses': dict(Counter(row['status'] for row in evidence['frames'])),
        'root_status': audit['status'], 'root_errors': audit['quality']['errors'],
        'long_intervals': len(audit['quality']['long_reference_intervals']),
        'unfilled_root_gaps': sum(row['status'] == 'unfilled' for row in audit['quality']['gaps']),
        'maximum_gap_s': max((row['seconds'] for row in audit['quality']['long_reference_intervals']), default=0),
        'inference_p50_ms': float(np.percentile(times, 50) * 1000),
        'inference_p95_ms': float(np.percentile(times, 95) * 1000),
        'wall_seconds': time.monotonic() - started, 'new_local_calls': len(evidence['frames']),
        'body_constructed': False, 'paid_requests': 0}
    write_json(target / 'report.json', report)
    print(json.dumps({'report': report, 'verification': seal_suite(target)}, indent=2))


def compare_availability(before, after):
    indices = [{row['frame']: row for row in evidence['frames']} for evidence in (before, after)]
    counts, rows = Counter(), []
    for frame in range(before['decoded_frames']):
        for joint in JOINTS:
            a, b = [image_point(index.get(frame), joint, before['width'], before['height']) for index in indices]
            kind = ('both' if a is not None else 'heavy_only') if b is not None else ('full_only' if a is not None else 'neither')
            counts[kind] += 1
            rows.append({'frame': frame, 'joint': joint, 'availability': kind,
                'between_model_distance_px': float(np.linalg.norm(a - b)) if a is not None and b is not None else None})
    return {'counts': dict(counts), 'source_joint_slots': len(rows), 'rows': rows,
            'scope': 'All source-frame/joint slots including unsampled frames; disagreement is not accuracy'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--worker', choices=CLIPS)
    args = parser.parse_args()
    out = args.out.resolve()
    if args.worker:
        worker(out, args.worker)
        return
    if out.is_relative_to(SOURCE) or out.is_relative_to(MODEL_SOURCE):
        raise ValueError('Use a fresh output outside saved source bundles')
    verified = {str(path): verify_artifacts(path) for path in (SOURCE, MODEL_SOURCE)}
    cases = {}
    for clip in CLIPS:
        baseline = read(SOURCE / clip / 'whole_image/evidence.json')
        video = ROOT / 'data/exercise' / (clip + '.webm')
        if digest(video) != baseline['source_sha256'] or baseline['model']['running_mode'] != 'IMAGE' or baseline['pose_identity']['mode'] != 'raw':
            raise ValueError('Comparison requires matched raw IMAGE-mode source evidence')
        cases[clip] = {'video': str(video.relative_to(ROOT)), 'source_sha256': baseline['source_sha256'],
            'baseline_usable': baseline['quality']['tracked_frames'], 'samples': len(baseline['frames']),
            'decoded_frames': baseline['decoded_frames']}
    implementation = {name: digest(path) for name, path in _implementation_files().items()}
    labels = {str(path): digest(path) for path in (PAIR_LABELS, LIMB_LABELS)}
    out.mkdir(parents=True, exist_ok=False)
    for name, path in _implementation_files().items():
        target = out / 'implementation-source' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
    for name in ('compare_heavy_exercises.py', 'check_crop_diversity.py', 'compare_pose_models.py', 'grade_pose_identity.py'):
        shutil.copyfile(Path(__file__).with_name(name), out / name)
    shutil.copyfile(ROOT / '.local/heavy-pose-model-tests-01.xml', out / 'model-tests.xml')
    pre = {'created_at': local_time(), 'cases': cases, 'model': model_provenance(MODEL, variant='heavy'),
        'source_verifications': verified, 'implementation': implementation, 'labels': labels,
        'maximum_workers': 2, 'process_seconds_per_case': 210, 'maximum_local_calls': sum(x['samples'] for x in cases.values()),
        'automatic_retries': 0, 'paid_requests': 0, 'projected_external_cost_usd': 0,
        'scope': 'One Heavy candidate on five complete development videos; no crop, relabeling or per-frame model selection',
        'excluded_from_construction': ['manual comparison labels', 'body/floor scores', 'Meta outputs'],
        'default_changed': False, 'body_or_policy_built': False}
    write_json(out / 'pre-run.json', pre)

    def run_case(clip):
        with (out / (clip + '.log')).open('x') as log:
            result = subprocess.run([sys.executable, str(Path(__file__)), '--out', str(out), '--worker', clip],
                cwd=ROOT, env=os.environ | {'MPLCONFIGDIR': str(ROOT / '.local/matplotlib')},
                stdout=log, stderr=subprocess.STDOUT, timeout=210)
        if result.returncode:
            return clip, {'status': 'worker_failed', 'returncode': result.returncode}
        return clip, read(out / clip / 'report.json')

    started = time.monotonic()
    with ThreadPoolExecutor(max_workers=2) as pool:
        outcomes = dict(pool.map(run_case, CLIPS))
    if any(value['status'] == 'worker_failed' for value in outcomes.values()):
        write_json(out / 'report.json', {'status': 'partial_failed_study', 'outcomes': outcomes, 'paid_requests': 0})
        print(json.dumps({'verification': seal_suite(out), 'outcomes': outcomes}, indent=2))
        raise SystemExit(1)
    if any(digest(Path(path)) != checksum for path, checksum in labels.items()):
        raise ValueError('Frozen development labels changed')
    annotations = {'identity': read(PAIR_LABELS), 'limbs': load_annotations(LIMB_LABELS, project_root=ROOT)}
    scores, availability = {}, {}
    for mode in ('full', 'heavy'):
        observations = {clip: read(SOURCE / clip / 'whole_image/evidence.json') if mode == 'full'
            else read(out / clip / 'evidence.json') for clip in CLIPS}
        predicted, matching = limb_predictions(annotations['limbs'], observations)
        scores[mode] = {'identity': grade_identity(annotations['identity'], observations),
            'limbs': grade_limb_projection(annotations['limbs'], predicted)}
        write_json(out / (mode + '-limb-predictions.json'), {'predictions': predicted, 'matching': matching})
    for clip in CLIPS:
        result = compare_availability(read(SOURCE / clip / 'whole_image/evidence.json'), read(out / clip / 'evidence.json'))
        write_json(out / (clip + '-availability.json'), result)
        availability[clip] = {key: value for key, value in result.items() if key != 'rows'}
    changes = {kind: deltas(scores['full'][kind]['observations'], scores['heavy'][kind]['observations'],
        'passes' if kind == 'identity' else 'within_tolerance') for kind in ('identity', 'limbs')}
    write_json(out / 'scores.json', {'scores': scores, 'changes': changes})
    if (implementation != {name: digest(path) for name, path in _implementation_files().items()}
            or any(verify_artifacts(Path(path)) != value for path, value in verified.items())
            or model_provenance(MODEL) != pre['model']):
        raise ValueError('Runtime, source bundle or model changed')
    report = {'status': 'complete_heavy_comparison', 'created_at': local_time(), 'outcomes': outcomes,
        'availability': availability, 'summaries': {mode: {kind: result['summary'] for kind, result in groups.items()}
            for mode, groups in scores.items()},
        'changes': {kind: {key: value for key, value in result.items() if key != 'paired_rows'} for kind, result in changes.items()},
        'wall_seconds': time.monotonic() - started, 'new_local_calls': sum(case['samples'] for case in cases.values()),
        'paid_requests': 0, 'default_changed': False, 'scope': pre['scope']}
    write_json(out / 'report.json', report)
    print(json.dumps({'outcomes': outcomes, 'summaries': report['summaries'], 'verification': seal_suite(out)}, indent=2))


if __name__ == '__main__':
    main()
