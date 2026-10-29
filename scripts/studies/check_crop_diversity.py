"""Matched whole-image/crop screening on frozen exercise-label frames, not full runs."""

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import cv2
import numpy as np

from weft.benchmark import seal_suite
from weft.camera_reference import JOINTS
from weft.exercise_vision import _detector, _draw_pose, model_provenance, summarize_result
from weft.io import digest, local_time, write_json
from weft.limb_fidelity import grade_limb_projection, load_annotations
from weft.pipeline import _implementation_files, verify_artifacts
from weft.pose_crop import CONTRACT, map_crop_pose, plan_crop
from weft.video_io import webm_video_frame_count
from grade_pose_identity import grade as grade_identity


ROOT = Path(__file__).resolve().parents[2]
CLIPS = ('single-leg-squat', 'squat', 'kettlebell-pushup', 'jumping-jacks-burpees', 'lunge-twist')
MODEL = ROOT/'.local/models/pose_landmarker_full.task'
PAIR_LABELS = ROOT/'data/pose-identity-labels/annotations.json'
LIMB_LABELS = ROOT/'data/limb-orientation/annotations.json'


def read(path):
    return json.loads(path.read_text())


def baseline(clip):
    study = 'exercise-body-e2e-02' if clip == 'lunge-twist' else 'spatial-human-e2e-04'
    return ROOT/'artifacts'/study/clip/'run'


def baseline_bundle(clip):
    path = baseline(clip)
    for parent in (path, path.parent, path.parent.parent):
        manifest = parent/'manifest.json'
        if not manifest.exists():
            continue
        files = read(manifest)['files']
        if all(str((path/name).relative_to(parent)) in files for name in ('evidence.json', 'pose-landmarks.json')):
            return parent
    raise ValueError('Original raw observations must be covered by a run or parent-study seal')


def infer(out, clip):
    inputs = out/'inputs'; frozen = verify_artifacts(inputs)
    case = read(inputs/'cases.json')[clip]
    evidence = read(inputs/(clip+'.json'))
    by_frame = {row['frame']: row for row in evidence['frames']}
    targets = {row['frame']: row for row in case['targets']}
    video = ROOT/case['video']
    if digest(video) != case['source_sha256'] or model_provenance(MODEL) != read(out/'pre-run.json')['model']:
        raise ValueError('Source/model differs from the frozen screening plan')
    target = out/clip; target.mkdir()
    begin, calls, decoded = time.monotonic(), 0, 0
    cap, detector = cv2.VideoCapture(str(video)), None
    outputs = {name: [] for name in ('whole_image', 'crop_image')}
    proposal_rows = []
    try:
        mp, detector = _detector(MODEL, 'image')
        with (target/'observations.jsonl').open('x') as journal:
            for frame in range(evidence['decoded_frames']):
                ok, bgr = cap.read()
                if not ok or bgr.shape[:2] != (evidence['height'], evidence['width']):
                    raise ValueError('The complete source interval or image dimensions changed')
                decoded += 1
                stamp = float(cap.get(cv2.CAP_PROP_POS_MSEC))/1000
                if time.monotonic()-begin > 90:
                    raise TimeoutError('Diversity screen exceeded its local time budget')
                if frame not in targets:
                    continue
                original = by_frame[frame]
                if abs(stamp-original['decoder_timestamp_s']) > 1e-6:
                    raise ValueError('Decoder timestamp differs from the original sampled observation')
                if not cv2.imwrite(str(target/f'source-{frame:04d}.png'), bgr):
                    raise OSError('Cannot retain original source frame')
                width = min(960, evidence['width'])
                image = cv2.resize(bgr, (width, round(evidence['height']*width/evidence['width'])), interpolation=cv2.INTER_AREA)
                tick = time.monotonic()
                whole = summarize_result(detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB,
                    data=cv2.cvtColor(image, cv2.COLOR_BGR2RGB))))
                whole['inference_seconds'] = time.monotonic()-tick; calls += 1
                whole.update({key: original[key] for key in ('id', 'frame', 'time_s', 'decoder_timestamp_s')})
                outputs['whole_image'].append(whole)
                _draw_pose(bgr, whole).save(target/f'whole-{frame:04d}.png')
                plan = targets[frame]
                raw = None
                if plan['status'] == 'planned':
                    left, top, right, bottom = plan['crop']['box_xyxy']
                    image = bgr[top:bottom, left:right]
                    if not cv2.imwrite(str(target/f'crop-input-{frame:04d}.png'), image):
                        raise OSError('Cannot retain exact crop pixels')
                    tick = time.monotonic()
                    raw = summarize_result(detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB,
                        data=cv2.cvtColor(image, cv2.COLOR_BGR2RGB))))
                    elapsed = time.monotonic()-tick; calls += 1
                    cropped = map_crop_pose(raw, plan['crop'], evidence, original, elapsed)
                    cropped['crop_proposal'] = plan['crop']
                    _draw_pose(bgr, cropped).save(target/f'crop-{frame:04d}.png')
                else:
                    cropped = {key: original[key] for key in ('id', 'frame', 'time_s', 'decoder_timestamp_s')}
                    cropped.update(status='crop_context_unavailable', pose_count=0, landmarks=[], world_landmarks=[],
                                   reason=plan['reason'], inference_seconds=0.)
                outputs['crop_image'].append(cropped)
                row = {'frame': frame, 'whole_image': whole, 'raw_crop_result': raw, 'crop_image': cropped}
                proposal_rows.append(row)
                journal.write(json.dumps(row, allow_nan=False)+'\n'); journal.flush()
            if cap.read()[0] or len(proposal_rows) != len(targets):
                raise ValueError('Source or selected frame coverage differs from the frozen plan')
        if verify_artifacts(inputs) != frozen or digest(video) != case['source_sha256']:
            raise ValueError('A frozen construction input changed')
        for mode, frames in outputs.items():
            exported = {key: evidence[key] for key in ('schema_version', 'source_sha256', 'width', 'height', 'fps', 'frame_count', 'decoded_frames')}
            exported.update(frames=frames, model=model_provenance(MODEL) | {'running_mode': 'IMAGE',
                'mediapipe_version': importlib.metadata.version('mediapipe'), 'maximum_pose_hypotheses': 2,
                'temporal_tracking': False},
                scope='Selected development frames only; not a complete inference run or simulation input')
            write_json(target/(mode+'.json'), exported)
        write_json(target/'report.json', {'status': 'matched_image_proposals_frozen', 'created_at': local_time(),
            'clip': clip, 'decoded_frames': decoded, 'selected_frames': len(targets), 'inference_calls': calls,
            'crop_plans': dict(Counter(row['status'] for row in case['targets'])),
            'whole_image_statuses': dict(Counter(row['status'] for row in outputs['whole_image'])),
            'crop_image_statuses': dict(Counter(row['status'] for row in outputs['crop_image'])),
            'wall_seconds': time.monotonic()-begin, 'paid_requests': 0,
            'scope': 'Source-pinned selected-frame screen; no label coordinates, body construction or promotion'})
        seal_suite(target)
    finally:
        cap.release()
        if detector is not None: detector.close()


def limb_predictions(labels, evidence_by_clip):
    result, matching = [], []
    for label in labels['frames']:
        evidence = evidence_by_clip[label['clip_id']]
        if (evidence['source_sha256'] != label['source_sha256']
                or evidence['width'] != label['width'] or evidence['height'] != label['height']):
            raise ValueError('Limb labels and original image domain differ')
        sample = min(evidence['frames'], key=lambda row: (abs(row['time_s']-label['timestamp_seconds']), row['frame']))
        offset = sample['time_s']-label['timestamp_seconds']
        near = abs(offset) <= 1/evidence['fps']+1e-8
        segments = {}; points = sample.get('landmarks', [])
        if near and sample['status'] in ('tracked', 'tracked_partial') and len(points) == 33:
            for name, endpoints in labels['segment_definitions'].items():
                indices = [JOINTS[joint] for joint in endpoints]
                if all(min(points[i]['visibility'], points[i]['presence']) >= .5
                       and -.05 <= points[i]['x'] <= 1.05 and -.05 <= points[i]['y'] <= 1.05 for i in indices):
                    a, b = (points[i] for i in indices)
                    segments[name] = {'axis_xy': [(b['x']-a['x'])*label['width'], (b['y']-a['y'])*label['height']]}
        result.append({'id': label['id'], 'source_sha256': label['source_sha256'], 'segments': segments})
        matching.append({'id': label['id'], 'sample_frame': sample['frame'], 'time_offset_s': offset, 'nearby': near})
    return result, matching


def deltas(before, after, pass_key):
    gains, losses, pairs = [], [], []
    for old, new in zip(before, after, strict=True):
        part = old.get('pair', old.get('segment'))
        if old['id'] != new['id'] or part != new.get('pair', new.get('segment')):
            raise ValueError('Comparison slots changed')
        item = {'id': old['id'], 'clip_id': old['clip_id'], 'part': part}
        if not old[pass_key] and new[pass_key]: gains.append(item)
        if old[pass_key] and not new[pass_key]: losses.append(item)
        if old['angular_error_degrees'] is not None and new['angular_error_degrees'] is not None:
            pairs.append(item | {'before': old['angular_error_degrees'], 'after': new['angular_error_degrees']})
    return {'new_passes': gains, 'regressions': losses, 'paired_scored': len(pairs),
        'paired_mean_before_deg': float(np.mean([row['before'] for row in pairs])) if pairs else None,
        'paired_mean_after_deg': float(np.mean([row['after'] for row in pairs])) if pairs else None,
        'paired_rows': pairs}


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--worker', choices=CLIPS); args = parser.parse_args(); out = args.out.resolve()
    if args.worker:
        infer(out, args.worker); return
    sources = [baseline_bundle(clip) for clip in CLIPS]
    if any(out.is_relative_to(path) for path in sources):
        raise ValueError('Output must be outside all sealed originals')
    verified = {str(path): verify_artifacts(path) for path in sources}
    pair_labels, limb_labels = read(PAIR_LABELS), load_annotations(LIMB_LABELS, project_root=ROOT)
    for label in pair_labels['frames']:
        if digest(ROOT/label['image_path']) != label['image_sha256']:
            raise ValueError('Frozen source label image changed')
    implementation = {name: digest(path) for name, path in _implementation_files().items()}
    out.mkdir(parents=True, exist_ok=False); (out/'inputs').mkdir()
    for path in (Path(__file__), Path(__file__).with_name('grade_pose_identity.py')):
        shutil.copyfile(path, out/path.name)
    for name, path in _implementation_files().items():
        target = out/'implementation-source'/name; target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
    cases = {}
    for clip in CLIPS:
        evidence = read(baseline(clip)/'evidence.json')
        evidence['frames'] = read(baseline(clip)/'pose-landmarks.json')['frames']
        video = ROOT/f'data/exercise/{clip}.webm'
        if digest(video) != evidence['source_sha256'] or not all('identity_raw' not in row for row in evidence['frames']):
            raise ValueError('Raw original evidence or video identity differs')
        if webm_video_frame_count(video) != evidence['decoded_frames']:
            raise ValueError('Encoded source count differs from complete retained observations')
        targets, matches = {}, []
        for label in [*pair_labels['frames'], *limb_labels['frames']]:
            if label['clip_id'] != clip: continue
            sample = min(evidence['frames'], key=lambda row: (abs(row['time_s']-label['timestamp_seconds']), row['frame']))
            if abs(sample['time_s']-label['timestamp_seconds']) > 1/evidence['fps']+1e-8:
                raise ValueError('Frozen label lacks a nearby original sampled frame')
            frame = sample['frame']; matches.append({'id': label['id'], 'label_frame': label['frame_index'], 'sample_frame': frame})
            if frame in targets: continue
            try:
                targets[frame] = {'frame': frame, 'status': 'planned', 'crop': plan_crop(evidence, frame)}
            except ValueError as error:
                targets[frame] = {'frame': frame, 'status': 'unavailable_context', 'reason': str(error)}
        cases[clip] = {'video': str(video.relative_to(ROOT)), 'source_sha256': evidence['source_sha256'],
            'original_decoded_frames': evidence['decoded_frames'], 'targets': [targets[k] for k in sorted(targets)],
            'label_to_sample_mapping': matches, 'baseline_directory': str(baseline(clip)),
            'baseline_verification': verified[str(baseline_bundle(clip))]}
        write_json(out/'inputs'/(clip+'.json'), evidence)
    write_json(out/'inputs/cases.json', cases)
    write_json(out/'inputs/report.json', {'status': 'crop_and_frame_selection_frozen', 'created_at': local_time(),
        'contract': dict(CONTRACT), 'selection': 'Union of frozen identity/limb label frame IDs, mapped to original nearest sampled frame',
        'label_coordinates_excluded': True, 'candidate_output_not_available': True})
    seal_suite(out/'inputs')
    maximum = sum(1+(p['status'] == 'planned') for case in cases.values() for p in case['targets'])
    write_json(out/'pre-run.json', {'created_at': local_time(), 'inputs': verified, 'implementation': implementation,
        'model': model_provenance(MODEL), 'mediapipe_version': importlib.metadata.version('mediapipe'),
        'labels': {str(path): digest(path) for path in (PAIR_LABELS, LIMB_LABELS)}, 'maximum_inference_calls': maximum,
        'maximum_workers': 2, 'process_timeout_seconds_each': 95, 'projected_external_cost_usd': 0,
        'scope': 'Cheap selected-frame development screen before complete-video reconstruction; not a final holdout',
        'arms': ['whole-image Full IMAGE', 'fixed original-temporal-crop Full IMAGE or explicit crop-context failure'],
        'no_auto_selection_or_promotion': True, 'default_change': False, 'paid_requests': 0})
    def run(clip):
        with (out/(clip+'.log')).open('x') as stream:
            try:
                process = subprocess.run([sys.executable, str(out/Path(__file__).name), '--out', str(out), '--worker', clip],
                    cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, timeout=95,
                    env=os.environ | {'MPLCONFIGDIR': str(ROOT/'.local/matplotlib')})
                return clip, process.returncode
            except subprocess.TimeoutExpired:
                return clip, None
    tick = time.monotonic()
    with ThreadPoolExecutor(max_workers=2) as pool: outcomes = dict(pool.map(run, CLIPS))
    if any(code != 0 for code in outcomes.values()):
        write_json(out/'report.json', {'status': 'screen_inference_failed', 'outcomes': outcomes, 'paid_requests': 0})
        print(json.dumps(seal_suite(out))); raise SystemExit(1)
    constructions = {clip: verify_artifacts(out/clip) for clip in CLIPS}
    scores = {}
    for mode in ('whole_image', 'crop_image'):
        evidences = {clip: read(out/clip/(mode+'.json')) for clip in CLIPS}
        poses = grade_identity(pair_labels, evidences)
        predictions, matching = limb_predictions(limb_labels, evidences)
        limbs = grade_limb_projection(limb_labels, predictions)
        write_json(out/(mode+'-limb-predictions.json'), {'predictions': predictions, 'time_matching': matching})
        scores[mode] = {'identity': poses, 'limbs': limbs}
    changes = {'identity': deltas(scores['whole_image']['identity']['observations'], scores['crop_image']['identity']['observations'], 'passes'),
               'limbs': deltas(scores['whole_image']['limbs']['observations'], scores['crop_image']['limbs']['observations'], 'within_tolerance')}
    if any(verify_artifacts(Path(path)) != value for path, value in verified.items()):
        raise ValueError('An original source bundle changed')
    if implementation != {name: digest(path) for name, path in _implementation_files().items()}:
        raise ValueError('Implementation changed during the screen')
    if any(digest(Path(path)) != expected for path, expected in read(out/'pre-run.json')['labels'].items()):
        raise ValueError('Comparison labels changed')
    write_json(out/'scores.json', {'scores': scores, 'changes': changes})
    reports = {clip: read(out/clip/'report.json') for clip in CLIPS}
    report = {'status': 'matched_crop_screen_complete', 'created_at': local_time(), 'constructions': constructions,
        'case_reports': reports, 'score_summaries': {mode: {name: value['summary'] for name, value in groups.items()} for mode, groups in scores.items()},
        'changes': {name: {key: value for key, value in rows.items() if key != 'paired_rows'} for name, rows in changes.items()},
        'wall_seconds': time.monotonic()-tick, 'paid_requests': 0, 'default_changed': False,
        'scope': 'Fixed-label source-image development screen, not body/contact accuracy or full-video crop qualification'}
    write_json(out/'report.json', report)
    print(json.dumps({'report': report, 'verification': seal_suite(out)}, indent=2))


if __name__ == '__main__':
    main()
