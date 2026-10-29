"""Compare complete IMAGE and VIDEO crop sequences without selecting a winner."""

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import cv2

from weft.benchmark import seal_suite
from weft.exercise_vision import ASSUMPTIONS, _detector, _draw_pose, _save_previews, model_provenance, summarize_result
from weft.io import canonical_hash, digest, local_time, write_json
from weft.limb_fidelity import grade_limb_projection, load_annotations
from weft.pipeline import _implementation_files, verify_artifacts
from weft.pose_crop import map_crop_pose
from weft.root_motion import build_root_reference
from check_crop_diversity import CLIPS, LIMB_LABELS, MODEL, PAIR_LABELS, deltas, limb_predictions
from check_selective_crop_videos import decoded_images, finish_evidence
from grade_pose_identity import grade as grade_identity


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / 'artifacts/jacks-pose-envelope-01'
BASE = ROOT / 'artifacts/selective-crop-videos-03'
VIDEO = ROOT / 'data/exercise/jumping-jacks-burpees.webm'
CLIP = 'jumping-jacks-burpees'
METHOD = 'whole-interval-fixed-crop-tracking-comparison-v1'


def read(path):
    return json.loads(path.read_text())


def worker(out, mode):
    pre = read(out / 'pre-run.json')
    original = read(SOURCE / 'inputs/context.json')
    if digest(VIDEO) != original['source_sha256'] or model_provenance(MODEL) != pre['model']:
        raise ValueError('Source bytes or pinned model changed')
    cached = {row['frame']: row for row in map(json.loads, (SOURCE / 'crop-observations.jsonl').read_text().splitlines())}
    if sorted(cached) != pre['cached_image_frames']:
        raise ValueError('Cached IMAGE observations changed')
    target = out / mode
    target.mkdir()
    indexed = {row['frame']: row for row in original['frames']}
    evidence = {key: deepcopy(value) for key, value in original.items()
                if key not in ('reference', 'quality', 'lengths', 'pose_identity', 'elapsed_seconds')}
    evidence['frames'] = []
    evidence['model'].update(running_mode=mode.upper(), temporal_tracking=mode == 'video')
    evidence['assumptions'] = ASSUMPTIONS[:] + [
        'One fixed crop is defined by original whole-image poses, not manual labels or body scores.',
        'VIDEO mode tracks regions across sampled frames; IMAGE mode infers each sample independently.'
    ]
    evidence['image_sampling'] = {'method': METHOD, 'mode': mode, 'box_xyxy': pre['box_xyxy'],
        'context_sha256': canonical_hash(original), 'source_plan_sha256': pre['source_plan_sha256'],
        'physical_identity_verified': False}
    previews, preview_times, new_calls, reuse = [], [], 0, []
    preview_frames = set(list(indexed)[::max(1, math.ceil(len(indexed) / 48))])
    started = time.monotonic()
    detector = None
    try:
        mp, detector = _detector(MODEL, mode)
        with (target / 'observations.jsonl').open('x') as journal:
            for frame, stamp, bgr in decoded_images(VIDEO, original, started):
                if time.monotonic() - started > 80:
                    raise TimeoutError('Fixed-crop tracking exceeded 80 seconds')
                if frame not in indexed:
                    continue
                source_row = indexed[frame]
                if stamp != source_row['decoder_timestamp_s']:
                    raise ValueError('Original source decoder clock changed')
                if frame == 0 and not cv2.imwrite(str(target / 'background.jpg'), bgr):
                    raise OSError('Cannot preserve source preview')
                if mode == 'image' and frame in cached:
                    raw = cached[frame]['raw_crop_result']
                    row = deepcopy(cached[frame]['proposed'])
                    reuse.append({'frame': frame, 'cached_proposal_sha256': canonical_hash(row)})
                    reused = True
                else:
                    left, top, right, bottom = pre['box_xyxy']
                    pixels = cv2.cvtColor(bgr[top:bottom, left:right], cv2.COLOR_BGR2RGB)
                    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=pixels)
                    tick = time.monotonic()
                    result = detector.detect(image) if mode == 'image' else detector.detect_for_video(image, round(source_row['time_s'] * 1000))
                    raw = summarize_result(result)
                    row = map_crop_pose(raw, {'frame': frame, 'box_xyxy': pre['box_xyxy']}, original,
                        source_row, time.monotonic() - tick)
                    new_calls += 1
                    reused = False
                evidence['frames'].append(row)
                journal.write(json.dumps({'frame': frame, 'reused': reused,
                    'raw_crop_result': raw, 'observation': row}, allow_nan=False) + '\n')
                journal.flush()
                if frame in preview_frames:
                    previews.append(_draw_pose(bgr, row))
                    preview_times.append(row['time_s'])
                if frame in pre['inspection_frames']:
                    _draw_pose(bgr, row).save(target / f'pose-{frame:04d}.png')
        if [row['frame'] for row in evidence['frames']] != list(indexed):
            raise ValueError('Complete original sampling must be preserved')
        quality_error = finish_evidence(evidence, target, evidence['frames'])
        _save_previews(previews, preview_times, target)
        audit = build_root_reference(evidence)
        write_json(target / 'root-motion.json', audit)
        write_json(target / 'cache-reuse.json', reuse)
        report = {'status': 'complete_fixed_crop_sequence', 'created_at': local_time(), 'mode': mode,
            'sampled_frames': len(evidence['frames']), 'source_frames_decoded': original['decoded_frames'],
            'new_inference_calls': new_calls, 'reused_image_calls': len(reuse),
            'usable': evidence['quality']['tracked_frames'], 'quality_error': quality_error,
            'status_counts': dict(Counter(row['status'] for row in evidence['frames'])),
            'root_status': audit['status'], 'root_errors': audit['quality']['errors'],
            'long_intervals': len(audit['quality']['long_reference_intervals']),
            'maximum_gap_s': max((row['seconds'] for row in audit['quality']['long_reference_intervals']), default=0),
            'unfilled_root_gaps': sum(row['status'] == 'unfilled' for row in audit['quality']['gaps']),
            'wall_seconds': time.monotonic() - started, 'paid_requests': 0}
        write_json(target / 'report.json', report)
    finally:
        if detector is not None:
            detector.close()
    print(json.dumps({'report': report, 'verification': seal_suite(target)}, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--worker', choices=('image', 'video'))
    args = parser.parse_args()
    out = args.out.resolve()
    if args.worker:
        worker(out, args.worker)
        return
    if out.is_relative_to(SOURCE) or out.is_relative_to(BASE):
        raise ValueError('Use a fresh output outside original studies')
    verified = {str(path): verify_artifacts(path) for path in (SOURCE, BASE)}
    context = read(SOURCE / 'inputs/context.json')
    cached = [json.loads(line)['frame'] for line in (SOURCE / 'crop-observations.jsonl').read_text().splitlines()]
    plan = read(SOURCE / 'inputs/plan.json')
    if digest(VIDEO) != context['source_sha256']:
        raise ValueError('Source bytes changed')
    out.mkdir(parents=True, exist_ok=False)
    implementation = {name: digest(path) for name, path in _implementation_files().items()}
    for name, path in _implementation_files().items():
        target = out / 'implementation-source' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
    for name in ('check_fixed_crop_tracking.py', 'check_selective_crop_videos.py', 'check_crop_diversity.py', 'grade_pose_identity.py'):
        shutil.copyfile(Path(__file__).with_name(name), out / name)
    labels = {str(path): digest(path) for path in (PAIR_LABELS, LIMB_LABELS)}
    pre = {'created_at': local_time(), 'method': METHOD, 'source_verifications': verified,
        'implementation': implementation, 'model': model_provenance(MODEL), 'box_xyxy': plan['box_xyxy'],
        'source_plan_sha256': canonical_hash(plan), 'cached_image_frames': sorted(cached),
        'maximum_new_calls': 2 * len(context['frames']) - len(cached), 'maximum_workers': 2,
        'process_timeout_seconds': 90, 'comparison_labels': labels,
        'inspection_frames': [40, 106, 112, 118, 124, 178, 184, 188, 190, 194, 200, 272, 276, 520, 526],
        'scope': 'Both complete conditions frozen before label comparison; no post-hoc per-frame selection',
        'paid_requests': 0, 'default_changed': False}
    write_json(out / 'pre-run.json', pre)

    def run_worker(mode):
        with (out / f'{mode}-worker.log').open('x') as log:
            process = subprocess.run([sys.executable, str(Path(__file__)), '--out', str(out), '--worker', mode],
                cwd=ROOT, env=os.environ | {'MPLCONFIGDIR': str(ROOT / '.local/matplotlib')},
                stdout=log, stderr=subprocess.STDOUT, timeout=90)
        if process.returncode:
            raise RuntimeError(f'{mode} worker failed ({process.returncode}); inspect its log')
        return mode, read(out / mode / 'report.json')

    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            outcomes = dict(pool.map(run_worker, ('image', 'video')))
        if any(digest(Path(path)) != checksum for path, checksum in labels.items()):
            raise ValueError('Frozen development labels changed')
        annotations = {'identity': read(PAIR_LABELS), 'limbs': load_annotations(LIMB_LABELS, project_root=ROOT)}
        scores = {}
        for mode in ('before', 'image', 'video'):
            observations = {clip: read(BASE / clip / 'selected/evidence.json') for clip in CLIPS}
            observations[CLIP] = read(SOURCE / 'candidate/evidence.json') if mode == 'before' else read(out / mode / 'evidence.json')
            predicted, matching = limb_predictions(annotations['limbs'], observations)
            scores[mode] = {'identity': grade_identity(annotations['identity'], observations),
                'limbs': grade_limb_projection(annotations['limbs'], predicted)}
            write_json(out / f'{mode}-limb-predictions.json', {'predictions': predicted, 'matching': matching})
        changes = {mode: {kind: deltas(scores['before'][kind]['observations'], scores[mode][kind]['observations'],
            'passes' if kind == 'identity' else 'within_tolerance') for kind in ('identity', 'limbs')}
            for mode in ('image', 'video')}
        write_json(out / 'scores.json', {'scores': scores, 'changes': changes})
        if (any(verify_artifacts(Path(path)) != value for path, value in verified.items())
                or implementation != {name: digest(path) for name, path in _implementation_files().items()}
                or digest(VIDEO) != context['source_sha256'] or model_provenance(MODEL) != pre['model']):
            raise ValueError('Frozen source, runtime or model changed')
        report = {'status': 'complete_tracking_comparison', 'created_at': local_time(), 'outcomes': outcomes,
            'summaries': {mode: {kind: result['summary'] for kind, result in groups.items()} for mode, groups in scores.items()},
            'changes': {mode: {kind: {key: value for key, value in result.items() if key != 'paired_rows'}
                for kind, result in group.items()} for mode, group in changes.items()},
            'paid_requests': 0, 'default_changed': False, 'scope': pre['scope']}
    except Exception as error:
        write_json(out / 'report.json', {'status': 'failed_tracking_comparison', 'error': str(error), 'paid_requests': 0})
        seal_suite(out)
        raise
    write_json(out / 'report.json', report)
    print(json.dumps({'outcomes': outcomes, 'summaries': report['summaries'], 'verification': seal_suite(out)}, indent=2))


if __name__ == '__main__':
    main()
