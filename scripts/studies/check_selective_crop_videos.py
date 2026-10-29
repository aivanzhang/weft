"""Qualify selective crops on complete sampled clips using original Full-model context."""

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import importlib.metadata
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import cv2
import numpy as np

from weft.benchmark import seal_suite
from weft.errors import UnsupportedVideo
from weft.exercise_vision import (ASSUMPTIONS, _detector, _draw_pose, _save_previews,
    finalize_evidence, model_provenance, resolve_pose_identity, summarize_result)
from weft.io import canonical_hash, digest, local_time, write_json
from weft.limb_fidelity import grade_limb_projection, load_annotations
from weft.pipeline import _implementation_files, verify_artifacts
from weft.pose_crop import map_crop_pose
from weft.pose_crop_selection import CONTRACT, apply_selective_crop, assess_selective_crop, plan_selective_crop
from weft.video_io import webm_video_frame_count
from check_crop_diversity import CLIPS, LIMB_LABELS, MODEL, PAIR_LABELS, deltas, limb_predictions
from grade_pose_identity import grade as grade_identity


ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT/'artifacts/crop-diversity-screen-01'
PROCESS_SECONDS = 120


def read(path):
    return json.loads(path.read_text())


def decoded_images(video, evidence, started):
    cap = cv2.VideoCapture(str(video))
    try:
        fields = [('width', cv2.CAP_PROP_FRAME_WIDTH), ('height', cv2.CAP_PROP_FRAME_HEIGHT),
                  ('fps', cv2.CAP_PROP_FPS), ('frame_count', cv2.CAP_PROP_FRAME_COUNT)]
        if not cap.isOpened() or any(cap.get(prop) != evidence[key] for key, prop in fields):
            raise ValueError('Decoder metadata differs from the original source')
        for index in range(evidence['decoded_frames']+1):
            if time.monotonic()-started > PROCESS_SECONDS-10:
                raise TimeoutError('Complete-video crop study exceeded its local time bound')
            ok, image = cap.read()
            if index == evidence['decoded_frames']:
                if ok: raise ValueError('Source contains frames beyond its verified interval')
                break
            if not ok or image.shape[:2] != (evidence['height'], evidence['width']):
                raise ValueError('Source ended early or changed image dimensions')
            stamp = float(cap.get(cv2.CAP_PROP_POS_MSEC))/1000
            if not math.isfinite(stamp): raise ValueError('Missing decoder timestamp')
            yield index, stamp, image
    finally:
        cap.release()


def finish_evidence(evidence, directory, originals):
    write_json(directory/'pose-landmarks.json', {'frames': originals, 'model': evidence['model'], 'scope': evidence['scope']})
    resolve_pose_identity(evidence, directory, 'raw')
    quality_error = None
    try:
        finalize_evidence(evidence, directory)
    except UnsupportedVideo as error:
        quality_error = str(error)
    write_json(directory/'pose-quality.json', evidence['quality'])
    write_json(directory/'report.json', {'status': 'complete_pose_evidence', 'quality_error': quality_error,
        'sampled_frames': len(evidence['frames']), 'source_sha256': evidence['source_sha256'],
        'scope': 'Saved pose observations, not a constructed or physically verified body'})
    return quality_error


def infer_case(out, clip):
    started = time.monotonic()
    pre = read(out/'pre-run.json'); case = read(out/'inputs/cases.json')[clip]
    template = read(out/'inputs'/(clip+'.json')); video = ROOT/case['video']
    if digest(video) != template['source_sha256'] or model_provenance(MODEL) != pre['model']:
        raise ValueError('Pinned source or model changed')
    original_clocks = {row['frame']: row for row in template['frames']}
    cached = read(CACHE/clip/'whole_image.json')
    if (any(cached['model'].get(k) != value for k, value in pre['model'].items())
            or cached['model']['mediapipe_version'] != pre['mediapipe_version']):
        raise ValueError('Cached inference uses a different model or runtime')
    cached_rows = {row['frame']: row for row in cached['frames']}
    directory = out/clip; directory.mkdir()
    whole_dir, selected_dir = directory/'whole_image', directory/'selected'
    whole_dir.mkdir(); selected_dir.mkdir()
    evidence = {key: deepcopy(value) for key, value in template.items() if key not in
        ('frames', 'quality', 'reference', 'lengths', 'origin_time_s', 'elapsed_seconds', 'pose_identity', 'model', 'assumptions')}
    evidence.update(frames=[], assumptions=ASSUMPTIONS[:], model=pre['model'] | {
        'running_mode': 'IMAGE', 'temporal_tracking': False, 'maximum_pose_hypotheses': 2,
        'mediapipe_version': pre['mediapipe_version']},
        scope='Complete original sampled interval; Full IMAGE-mode development observations, not physical truth')
    evidence['assumptions'].append('All sampled frames use independent IMAGE inference; original Full-model neighbors alone define crop context.')
    previews, stamps, cache_matches, new_calls = [], [], [], 0
    targets = sorted(original_clocks)
    preview_frames = set(targets[::max(1, math.ceil(len(targets)/48))])
    width = min(960, evidence['width']); size = (width, round(evidence['height']*width/evidence['width']))
    detector = None
    try:
        mp, detector = _detector(MODEL, 'image')
        reused = pre.get('reused_whole_run')
        if reused:
            source = Path(reused)/clip/'whole_image'
            manifest = read(Path(reused)/'manifest.json')['files']
            for path in source.iterdir():
                if not path.is_file() or manifest.get(str(path.relative_to(reused))) != digest(path):
                    raise ValueError('Reused whole-image artifact is not covered by the verified study')
                shutil.copyfile(path, whole_dir/path.name)
            evidence = read(whole_dir/'evidence.json')
            cache_matches = [row['frame'] for row in evidence['frames']]
            for row in evidence['frames']:
                original = original_clocks[row['frame']]
                if any(row[key] != original[key] for key in ('id', 'frame', 'time_s', 'decoder_timestamp_s')):
                    raise ValueError('Reused whole-image evidence changed source identity or clocks')
        else:
            with (whole_dir/'new-observations.jsonl').open('x') as journal:
                for frame, stamp, bgr in decoded_images(video, evidence, started):
                    if frame == 0 and not cv2.imwrite(str(whole_dir/'background.jpg'), bgr):
                        raise OSError('Cannot preserve source preview')
                    if frame not in original_clocks: continue
                    original = original_clocks[frame]
                    if abs(stamp-original['decoder_timestamp_s']) > 1e-6:
                        raise ValueError('Decoder clock differs from original evidence')
                    resized = cv2.resize(bgr, size, interpolation=cv2.INTER_AREA)
                    if frame in cached_rows:
                        stored = cv2.imread(str(CACHE/clip/f'source-{frame:04d}.png'))
                        if stored is None or not np.array_equal(stored, bgr):
                            raise ValueError('Cached source pixels do not match this exact frame')
                        row = deepcopy(cached_rows[frame]); cache_matches.append(frame)
                    else:
                        tick = time.monotonic()
                        row = summarize_result(detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB,
                            data=cv2.cvtColor(resized, cv2.COLOR_BGR2RGB))))
                        row.update({key: original[key] for key in ('id', 'frame', 'time_s', 'decoder_timestamp_s')})
                        row['inference_seconds'] = time.monotonic()-tick; new_calls += 1
                        journal.write(json.dumps(row, allow_nan=False)+'\n'); journal.flush()
                    if any(row[key] != original[key] for key in ('id', 'frame', 'time_s', 'decoder_timestamp_s')):
                        raise ValueError('Whole-image inference changed source identity or clock')
                    evidence['frames'].append(row)
                    if frame in preview_frames:
                        previews.append(_draw_pose(bgr, row)); stamps.append(row['time_s'])
        if [row['frame'] for row in evidence['frames']] != targets or cache_matches != (targets if reused else sorted(cached_rows)):
            raise ValueError('Complete original sampling or cached coverage was not preserved')
        if not reused: evidence['elapsed_seconds'] = time.monotonic()-started
        quality_before = finish_evidence(evidence, whole_dir, evidence['frames'])
        _save_previews(previews, stamps, whole_dir)
        whole_verification = seal_suite(whole_dir)
        plans = {row['frame']: plan_selective_crop(evidence, row) for row in evidence['frames']}
        eligible = {frame: plan for frame, plan in plans.items() if plan['eligible']}
        bound_exceeded = len(eligible) > CONTRACT['maximum_target_frames']
        write_json(directory/'crop-plan.json', {'created_at': local_time(), 'plans': list(plans.values()),
            'eligible_frames': sorted(eligible), 'bound_exceeded': bound_exceeded,
            'context': 'Original complete Full IMAGE-mode sequence; no repaired anchors or comparison labels',
            'baseline_verification': whole_verification})
        proposals, decisions, observations, crop_calls = {}, {}, {}, 0
        originals = {row['frame']: row for row in evidence['frames']}
        # Context changes invalidate old crop results even when the target frame is the same.
        old_crops = {row['frame']: row for row in read(CACHE/clip/'crop_image.json')['frames']}
        raw_cache = {row['frame']: row for row in map(json.loads, (CACHE/clip/'observations.jsonl').read_text().splitlines())}
        crop_reuse = []
        selected_previews, selected_times = [], []
        with (directory/'crop-observations.jsonl').open('x') as journal:
            for frame, stamp, bgr in decoded_images(video, evidence, started):
                if frame not in originals: continue
                original = originals[frame]
                if abs(stamp-original['decoder_timestamp_s']) > 1e-6:
                    raise ValueError('Crop decode changed original source clock')
                if frame in eligible and not bound_exceeded:
                    plan = eligible[frame]; crop = plan['crop']
                    x1, y1, x2, y2 = crop['box_xyxy']; pixels = bgr[y1:y2, x1:x2]
                    old = old_crops.get(frame, {})
                    if old.get('crop_proposal') == crop:
                        saved = cv2.imread(str(CACHE/clip/f'crop-input-{frame:04d}.png'))
                        if saved is None or not np.array_equal(saved, pixels):
                            raise ValueError('Matching cached crop plan has different pixels')
                        proposed = deepcopy(old); raw = raw_cache[frame]['raw_crop_result']; crop_reuse.append(frame)
                    else:
                        tick = time.monotonic()
                        raw = summarize_result(detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB,
                            data=cv2.cvtColor(pixels, cv2.COLOR_BGR2RGB))))
                        proposed = map_crop_pose(raw, crop, evidence, original, time.monotonic()-tick)
                        proposed['crop_proposal'] = crop; crop_calls += 1
                    proposals[frame] = proposed
                    decisions[frame] = assess_selective_crop(evidence, original, proposed, plan)
                    observations[frame] = {'frame': frame, 'raw_crop_result': raw, 'proposed': proposed, 'decision': decisions[frame]}
                    journal.write(json.dumps(observations[frame], allow_nan=False)+'\n'); journal.flush()
                    for name, image in (('source', bgr), ('crop-input', pixels)):
                        if not cv2.imwrite(str(directory/f'{name}-{frame:04d}.png'), image):
                            raise OSError('Cannot retain crop source pixels')
                    _draw_pose(bgr, original).save(directory/f'before-{frame:04d}.png')
                    _draw_pose(bgr, proposed).save(directory/f'proposal-{frame:04d}.png')
                if frame in preview_frames:
                    row = proposals[frame] if decisions.get(frame, {}).get('accepted') else original
                    selected_previews.append(_draw_pose(bgr, row)); selected_times.append(row['time_s'])
        selected = apply_selective_crop(evidence, proposals, plans, decisions)
        quality_after = finish_evidence(selected, selected_dir, evidence['frames'])
        shutil.copyfile(whole_dir/'background.jpg', selected_dir/'background.jpg')
        _save_previews(selected_previews, selected_times, selected_dir)
        write_json(directory/'decisions.json', list(decisions.values()))
        accepted = selected['pose_reinference']['changed_frames']
        for original, row in zip(evidence['frames'], selected['frames'], strict=True):
            if original['frame'] not in accepted and row != original:
                raise ValueError('An unselected original pose changed')
        if (digest(video) != evidence['source_sha256'] or model_provenance(MODEL) != pre['model']
                or verify_artifacts(whole_dir) != whole_verification):
            raise ValueError('A source, model or frozen whole-image baseline changed')
        selected_verification = seal_suite(selected_dir)
        report = {'status': 'complete_video_selection_frozen', 'created_at': local_time(), 'clip': clip,
            'source_frames': evidence['decoded_frames'], 'sampled_frames': len(targets),
            'whole_new_calls': new_calls, 'whole_cached_frames': cache_matches,
            'whole_inference_origin': reused or 'this study plus matched selected-frame cache',
            'eligible_frames': sorted(eligible), 'crop_bound_exceeded': bound_exceeded,
            'crop_new_calls': crop_calls, 'crop_cached_frames': crop_reuse, 'accepted_frames': accepted,
            'decision_reasons': dict(Counter(reason for row in decisions.values() for reason in row['reasons'])),
            'whole_quality_error': quality_before, 'selected_quality_error': quality_after,
            'whole_quality': evidence['quality'], 'selected_quality': selected['quality'],
            'unselected_rows_unchanged': len(targets)-len(accepted),
            'whole_verification': whole_verification, 'selected_verification': selected_verification,
            'wall_seconds': time.monotonic()-started, 'paid_requests': 0,
            'scope': 'Complete sampled-image development comparison; no body construction or independent physical accuracy'}
        write_json(directory/'report.json', report)
        seal_suite(directory)
    finally:
        if detector is not None: detector.close()


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--worker', choices=CLIPS)
    parser.add_argument('--reuse-whole', type=Path, help='Recover complete whole-image observations from a sealed study')
    args = parser.parse_args(); out = args.out.resolve()
    if args.worker:
        infer_case(out, args.worker); return
    if out.is_relative_to(CACHE): raise ValueError('Use a new output outside immutable cached proposals')
    cache_verification = verify_artifacts(CACHE)
    if digest(CACHE/'check_crop_diversity.py') != digest(Path(__file__).with_name('check_crop_diversity.py')):
        raise ValueError('Cached image preprocessing implementation changed')
    implementation = {name: digest(path) for name, path in _implementation_files().items()}
    prior = read(CACHE/'pre-run.json')
    if implementation['exercise_vision.py'] != prior['implementation']['exercise_vision.py']:
        raise ValueError('Cached pose model parsing/detector implementation changed')
    reused = args.reuse_whole.resolve() if args.reuse_whole else None
    reuse_verification = None
    if reused:
        if out.is_relative_to(reused): raise ValueError('Output must not alter the saved full-image study')
        reuse_verification = verify_artifacts(reused)
        reused_pre = read(reused/'pre-run.json')
        if (reused_pre['model'] != model_provenance(MODEL)
                or reused_pre['mediapipe_version'] != importlib.metadata.version('mediapipe')
                or reused_pre['implementation']['exercise_vision.py'] != implementation['exercise_vision.py']):
            raise ValueError('Reused complete-image parser/model/runtime changed')
    cases = read(CACHE/'inputs/cases.json')
    expected_new = 0
    for clip, case in cases.items():
        evidence = read(CACHE/'inputs'/(clip+'.json')); video = ROOT/case['video']
        frames = [row['frame'] for row in evidence['frames']]
        if (len(frames) > 900 or evidence['decoded_frames'] > 1800 or evidence['decoded_frames']/evidence['fps'] > 60
                or frames != list(range(0, evidence['decoded_frames'], evidence['inference_stride']))
                or digest(video) != evidence['source_sha256'] or webm_video_frame_count(video) != evidence['decoded_frames']):
            raise ValueError('Original complete source sampling or limits changed')
        expected_new += len(frames)-len(read(CACHE/clip/'whole_image.json')['frames'])
        if reused:
            saved = read(reused/clip/'whole_image/evidence.json')
            if (saved['source_sha256'] != evidence['source_sha256']
                    or [row['frame'] for row in saved['frames']] != frames
                    or saved['decoded_frames'] != evidence['decoded_frames']
                    or saved['pose_identity']['mode'] != 'raw' or saved.get('pose_reinference')
                    or saved['model']['running_mode'] != 'IMAGE' or saved['model']['temporal_tracking'] is not False
                    or saved['model']['maximum_pose_hypotheses'] != 2):
                raise ValueError('Reused whole-image observations are incomplete or already modified')
    out.mkdir(parents=True, exist_ok=False); (out/'inputs').mkdir()
    for clip in CLIPS: shutil.copyfile(CACHE/'inputs'/(clip+'.json'), out/'inputs'/(clip+'.json'))
    write_json(out/'inputs/cases.json', {clip: {'video': cases[clip]['video']} for clip in CLIPS})
    write_json(out/'inputs/report.json', {'status': 'complete_source_inputs_frozen',
        'created_at': local_time(), 'source': str(CACHE), 'cached_verification': cache_verification,
        'scope': 'Original source clocks and sampling only; historical poses do not supply new crop context'})
    for path in (Path(__file__), Path(__file__).with_name('check_crop_diversity.py'), Path(__file__).with_name('grade_pose_identity.py')):
        shutil.copyfile(path, out/path.name)
    for name, path in _implementation_files().items():
        target = out/'implementation-source'/name; target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
    write_json(out/'pre-run.json', {'created_at': local_time(), 'cached_verification': cache_verification,
        'implementation': implementation, 'model': model_provenance(MODEL), 'mediapipe_version': importlib.metadata.version('mediapipe'),
        'contract': dict(CONTRACT), 'maximum_workers': 2, 'process_timeout_seconds': PROCESS_SECONDS,
        'new_whole_image_calls': 0 if reused else expected_new,
        'reused_whole_run': str(reused) if reused else None, 'reused_whole_verification': reuse_verification,
        'maximum_new_crop_calls': len(CLIPS)*CONTRACT['maximum_target_frames'],
        'paid_requests': 0, 'new_cloud_cost_usd': 0, 'labels': {str(path): digest(path) for path in (PAIR_LABELS, LIMB_LABELS)},
        'selection': 'Every original sampled frame; all eligible source-only retries before comparison scoring',
        'scope': 'Five already-exposed development sources; not independent holdout or anatomical validation'})
    seal_suite(out/'inputs')
    started = time.monotonic()
    def worker(clip):
        with (out/(clip+'.log')).open('x') as log:
            result = subprocess.run([sys.executable, str(Path(__file__)), '--out', str(out), '--worker', clip],
                cwd=ROOT, env=os.environ | {'MPLCONFIGDIR': str(ROOT/'.local/matplotlib')},
                stdout=log, stderr=subprocess.STDOUT, timeout=PROCESS_SECONDS)
        return clip, result.returncode
    with ThreadPoolExecutor(max_workers=2) as pool:
        outcomes = dict(pool.map(worker, CLIPS))
    write_json(out/'worker-results.json', outcomes)
    if any(outcomes.values()):
        write_json(out/'report.json', {'status': 'incomplete_workers', 'outcomes': outcomes, 'paid_requests': 0})
        seal_suite(out)
        raise RuntimeError('Preserved failed workers; do not rerun successful inference')
    reports = {clip: read(out/clip/'report.json') for clip in CLIPS}
    write_json(out/'construction.json', {'created_at': local_time(), 'status': 'all_candidates_frozen_before_scoring',
        'cases': {clip: verify_artifacts(out/clip) for clip in CLIPS}})
    if any(digest(Path(path)) != value for path, value in read(out/'pre-run.json')['labels'].items()):
        raise ValueError('Fixed development labels changed')
    labels = {'identity': read(PAIR_LABELS), 'limbs': load_annotations(LIMB_LABELS, project_root=ROOT)}
    scores = {}
    for mode in ('whole_image', 'selected'):
        observations = {clip: read(out/clip/mode/'evidence.json') for clip in CLIPS}
        predicted, matching = limb_predictions(labels['limbs'], observations)
        scores[mode] = {'identity': grade_identity(labels['identity'], observations),
                        'limbs': grade_limb_projection(labels['limbs'], predicted)}
        write_json(out/(mode+'-limb-predictions.json'), {'predictions': predicted, 'time_matching': matching})
    changes = {kind: deltas(scores['whole_image'][kind]['observations'], scores['selected'][kind]['observations'],
        'passes' if kind == 'identity' else 'within_tolerance') for kind in ('identity', 'limbs')}
    write_json(out/'scores.json', {'scores': scores, 'changes': changes})
    if (verify_artifacts(CACHE) != cache_verification
            or (reused and verify_artifacts(reused) != reuse_verification)
            or implementation != {name: digest(path) for name, path in _implementation_files().items()}):
        raise ValueError('Cached input or frozen runtime changed')
    report = {'status': 'complete_video_selective_crop_compared', 'created_at': local_time(),
        'wall_seconds': time.monotonic()-started, 'cases': reports,
        'summaries': {mode: {name: group['summary'] for name, group in kinds.items()} for mode, kinds in scores.items()},
        'changes': {kind: {key: value for key, value in group.items() if key != 'paired_rows'} for kind, group in changes.items()},
        'new_inference_calls': sum(row['whole_new_calls']+row['crop_new_calls'] for row in reports.values()),
        'paid_requests': 0, 'default_changed': False, 'body_construction': False,
        'scope': 'Complete sampled-image development comparison with original Full-model context; physical validation unavailable'}
    write_json(out/'report.json', report)
    print(json.dumps({'verification': seal_suite(out), 'summaries': report['summaries'],
        'accepted_frames': {clip: row['accepted_frames'] for clip, row in reports.items()},
        'quality_errors': {clip: row['selected_quality_error'] for clip, row in reports.items()},
        'new_inference_calls': report['new_inference_calls'], 'wall_seconds': report['wall_seconds']}, indent=2))


if __name__ == '__main__': main()
