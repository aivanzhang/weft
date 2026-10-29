"""Inspect every unused source frame inside the remaining reference gaps."""

import argparse
from collections import Counter
from copy import deepcopy
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import cv2

from weft.benchmark import seal_suite
from weft.exercise_vision import _detector, _draw_pose, model_provenance, summarize_result
from weft.io import canonical_hash, digest, local_time, write_json
from weft.pipeline import _implementation_files, verify_artifacts
from weft.pose_crop import map_crop_pose
from weft.root_motion import build_root_reference
from check_selective_crop_videos import decoded_images, finish_evidence


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / 'artifacts/jacks-pose-envelope-01'
VIDEO = ROOT / 'data/exercise/jumping-jacks-burpees.webm'
MODEL = ROOT / '.local/models/pose_landmarker_full.task'
METHOD = 'unused-gap-source-frame-probe-v1'


def read(path):
    return json.loads(path.read_text())


def plan_samples(evidence, root):
    cached = {row['frame'] for row in evidence['frames']}
    intervals, missing = [], set()
    for row in root['quality']['long_reference_intervals']:
        left, right = [int(row[key].rsplit(':', 1)[1])
                       for key in ('left_evidence_id', 'right_evidence_id')]
        selected = sorted(set(range(left + 1, right)) - cached)
        intervals.append({'left': left, 'right': right, 'new_frames': selected})
        missing.update(selected)
    if not 1 <= len(missing) <= 64 or len(cached | missing) > 900:
        raise ValueError('Keep every cached sample and bound new gap samples to 64')
    return {'method': METHOD, 'intervals': intervals, 'new_frames': sorted(missing),
            'input_evidence_sha256': canonical_hash(evidence),
            'root_audit_sha256': canonical_hash(root), 'conditions': ['whole', 'crop']}


def worker(out):
    pre = read(out / 'pre-run.json')
    current = read(SOURCE / 'candidate/evidence.json')
    original = read(SOURCE / 'inputs/context.json')
    plan = pre['plan']
    if plan_samples(current, read(SOURCE / 'after-root-motion.json')) != plan:
        raise ValueError('Source-bound gap plan changed')
    cached = {row['frame']: row for row in current['frames']}
    requested = set(plan['new_frames'])
    rows = {mode: {} for mode in plan['conditions']}
    started = time.monotonic()
    detector = None
    try:
        mp, detector = _detector(MODEL, 'image')
        with (out / 'new-observations.jsonl').open('x') as journal:
            for frame, stamp, bgr in decoded_images(VIDEO, current, started):
                if time.monotonic() - started > 80:
                    raise TimeoutError('Gap inference exceeded its 80-second local bound')
                if frame in cached and stamp != cached[frame]['decoder_timestamp_s']:
                    raise ValueError('Cached source decoder clock changed')
                if frame not in requested:
                    continue
                if abs(stamp - frame / current['fps']) > 1 / current['fps']:
                    raise ValueError('New decoder timestamp disagrees with source-frame time')
                identity = {'frame': frame, 'time_s': frame / current['fps'],
                            'decoder_timestamp_s': stamp,
                            'id': f"{current['source_sha256']}:frame:{frame}"}
                for mode in plan['conditions']:
                    tick = time.monotonic()
                    x1, y1, x2, y2 = pre['box_xyxy']
                    pixels = bgr if mode == 'whole' else bgr[y1:y2, x1:x2]
                    raw = summarize_result(detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB,
                        data=cv2.cvtColor(pixels, cv2.COLOR_BGR2RGB))))
                    elapsed = time.monotonic() - tick
                    if mode == 'whole':
                        row = raw | identity | {'inference_seconds': elapsed}
                    else:
                        crop = {'frame': frame, 'box_xyxy': pre['box_xyxy']}
                        row = map_crop_pose(raw, crop, current, identity, elapsed)
                        row['image_sampling'] = {'method': METHOD, 'crop': crop,
                            'plan_sha256': canonical_hash(plan), 'prior_observation': None}
                    rows[mode][frame] = row
                    journal.write(json.dumps({'condition': mode, 'frame': frame,
                        'raw_result': raw, 'observation': row}, allow_nan=False) + '\n')
                    journal.flush()
                    _draw_pose(bgr, row).save(out / f'{mode}-{frame:04d}.png')
    finally:
        if detector is not None:
            detector.close()
    results = {}
    for mode in plan['conditions']:
        if set(rows[mode]) != requested:
            raise ValueError('Every planned sample must be retained, including failed poses')
        candidate = deepcopy(current)
        candidate['frames'] = sorted(candidate['frames'] + list(rows[mode].values()), key=lambda row: row['frame'])
        candidate['inference_stride'] = None
        candidate['sample_rate_fps'] = None
        candidate['densification'] = {'method': METHOD, 'plan_sha256': canonical_hash(plan),
            'condition': mode, 'new_frames': plan['new_frames'], 'cached_samples': len(cached),
            'interpolated_observations': 0, 'scope': 'New pixel estimates, not verified anatomy or contact'}
        if any(row != cached[row['frame']] for row in candidate['frames'] if row['frame'] in cached):
            raise ValueError('An existing source observation changed')
        output = out / mode
        output.mkdir()
        raw_rows = sorted(deepcopy(original['frames']) + list(rows[mode].values()), key=lambda row: row['frame'])
        quality_error = finish_evidence(candidate, output, raw_rows)
        shutil.copyfile(SOURCE / 'candidate/background.jpg', output / 'background.jpg')
        root = build_root_reference(candidate)
        write_json(output / 'root-motion.json', root)
        results[mode] = {'usable': candidate['quality']['tracked_frames'],
            'sampled_frames': len(candidate['frames']), 'quality_error': quality_error,
            'new_statuses': dict(Counter(row['status'] for row in rows[mode].values())),
            'root_status': root['status'], 'root_errors': root['quality']['errors'],
            'long_intervals': len(root['quality']['long_reference_intervals']),
            'maximum_gap_s': max((row['seconds'] for row in root['quality']['long_reference_intervals']), default=0),
            'unfilled_root_gaps': sum(row['status'] == 'unfilled' for row in root['quality']['gaps']),
            'verification': seal_suite(output)}
    write_json(out / 'construction.json', {'created_at': local_time(), 'results': results,
        'wall_seconds': time.monotonic() - started, 'new_local_calls': 2 * len(requested),
        'cached_samples_unchanged': len(cached), 'source_frames_decoded': current['decoded_frames'],
        'paid_requests': 0, 'body_constructed': False})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--worker', action='store_true')
    args = parser.parse_args()
    out = args.out.resolve()
    if args.worker:
        worker(out)
        return
    if out.is_relative_to(SOURCE):
        raise ValueError('Use a new directory outside saved source evidence')
    verified = verify_artifacts(SOURCE)
    current = read(SOURCE / 'candidate/evidence.json')
    if digest(VIDEO) != current['source_sha256']:
        raise ValueError('Original video bytes changed')
    implementation = {name: digest(path) for name, path in _implementation_files().items()}
    provenance = model_provenance(MODEL)
    if any(current['model'].get(key) != value for key, value in provenance.items()):
        raise ValueError('Pinned pose model changed')
    plan = plan_samples(current, read(SOURCE / 'after-root-motion.json'))
    out.mkdir(parents=True, exist_ok=False)
    for name in ('check_jacks_gap_samples.py', 'check_selective_crop_videos.py', 'check_crop_diversity.py', 'grade_pose_identity.py'):
        shutil.copyfile(Path(__file__).with_name(name), out / name)
    for name, path in _implementation_files().items():
        target = out / 'implementation-source' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
    write_json(out / 'pre-run.json', {'created_at': local_time(), 'source_verification': verified,
        'plan': plan, 'box_xyxy': read(SOURCE / 'inputs/plan.json')['box_xyxy'],
        'model': provenance, 'implementation': implementation, 'process_timeout_seconds': 90,
        'maximum_local_calls': 2 * len(plan['new_frames']), 'paid_requests': 0,
        'selection': 'Every unused source frame inside every unsupported reference interval',
        'exclusions': ['manual labels', 'body/floor scores', 'Meta outputs'],
        'scope': 'Two prespecified diagnostic conditions; no automatic winner or default promotion'})
    try:
        with (out / 'worker.log').open('x') as log:
            process = subprocess.run([sys.executable, str(Path(__file__)), '--out', str(out), '--worker'],
                cwd=ROOT, env=os.environ | {'MPLCONFIGDIR': str(ROOT / '.local/matplotlib')},
                stdout=log, stderr=subprocess.STDOUT, timeout=90)
        if process.returncode:
            raise RuntimeError(f'Gap worker failed ({process.returncode}); inspect worker.log')
        if (verify_artifacts(SOURCE) != verified or digest(VIDEO) != current['source_sha256']
                or model_provenance(MODEL) != provenance
                or implementation != {name: digest(path) for name, path in _implementation_files().items()}):
            raise ValueError('Source evidence, video, model or runtime changed')
        report = read(out / 'construction.json') | {'status': 'gap_samples_compared', 'default_changed': False}
    except Exception as error:
        write_json(out / 'report.json', {'status': 'failed_gap_probe', 'error': str(error), 'paid_requests': 0})
        seal_suite(out)
        raise
    write_json(out / 'report.json', report)
    print(json.dumps({'report': report, 'verification': seal_suite(out)}, indent=2))


if __name__ == '__main__':
    main()
