"""Infer additional IMAGE-mode samples while preserving every cached observation."""

import argparse
from copy import deepcopy
import importlib.metadata
import json
import math
from pathlib import Path
import time

import cv2

from weft.benchmark import seal_suite
from weft.errors import UnsupportedVideo
from weft.exercise_vision import (_detector, _draw_pose, _save_previews, finalize_evidence,
                                 model_provenance, resolve_pose_identity, summarize_result)
from weft.io import canonical_hash, digest, local_time, write_json
from weft.pipeline import verify_artifacts


METHOD = 'cached-image-pose-densification-v1'


def _plan(evidence, source_hash, provenance, version, stride):
    if (evidence.get('profile') != 'exercise-pose-v1' or type(evidence.get('schema_version')) is not int or evidence.get('schema_version') != 1
            or evidence.get('source_sha256') != source_hash):
        raise ValueError('Densification requires the exact original exercise video and pose evidence')
    model = evidence.get('model', {})
    if (model.get('running_mode') != 'IMAGE' or model.get('temporal_tracking') is not False
            or model.get('maximum_pose_hypotheses') != 2 or model.get('mediapipe_version') != version
            or any(model.get(key) != value for key, value in provenance.items())):
        raise ValueError('Only identical pinned IMAGE-mode models and runtime versions can reuse cached poses')
    identity = evidence.get('pose_identity', {})
    if identity.get('mode') != 'raw' or identity.get('repaired_frames') != 0 or evidence.get('pose_reinference'):
        raise ValueError('Densification requires original raw identities')
    count, fps = evidence.get('frame_count'), evidence.get('fps')
    if (type(count) is not int or not 10 <= count <= 1800
            or type(fps) not in (int, float) or not math.isfinite(fps) or not 1 <= fps <= 240 or count/fps > 60
            or evidence.get('decoded_frames') != count):
        raise ValueError('Densification requires a complete constant-frame-rate source within existing bounds')
    if type(stride) is not int or not 1 <= stride <= 1800:
        raise ValueError('Requested stride must be an integer from 1 to 1800')
    frames = evidence.get('frames')
    if not isinstance(frames, list) or not 1 <= len(frames) <= 900:
        raise ValueError('Cached poses exceed the 1–900 sample bound')
    cached, previous = {}, -1
    for row in frames:
        if not isinstance(row, dict):
            raise ValueError('Cached observations must be records')
        index = row.get('frame')
        stamps = [row.get(key) for key in ('time_s', 'decoder_timestamp_s')]
        if (type(index) is not int or not previous < index < count
                or row.get('id') != f'{source_hash}:frame:{index}' or 'identity_raw' in row
                or any(type(value) not in (int, float) or not math.isfinite(value) or abs(value-index/fps) > 1e-7 for value in stamps)):
            raise ValueError('Cached evidence must retain original unique frame identities and exact decoder clocks')
        cached[index] = row
        previous = index
    selected = sorted(set(range(0, count, stride)) | set(cached))
    missing = sorted(set(selected)-set(cached))
    if len(selected) > 900:
        raise ValueError('Requested union exceeds 900 poses; no cached observation will be dropped')
    if not missing:
        raise ValueError('Requested frames are already cached; no inference is needed')
    return cached, selected, missing


def densify_image_poses(run, video, model, out, *, stride=1):
    run, video, model, out = (Path(path).resolve() for path in (run, video, model, out))
    if out.exists() or out.is_relative_to(run):
        raise ValueError('Use a new output outside the immutable cached run')
    verified = verify_artifacts(run)
    evidence_path = run / 'evidence.json'
    manifest = json.loads((run / 'manifest.json').read_text())
    if evidence_path.stat().st_size > 32*1024*1024 or manifest['files'].get('evidence.json') != digest(evidence_path):
        raise ValueError('Cached evidence must be bounded and sealed')
    if not video.is_file() or video.stat().st_size > 100_000_000:
        raise ValueError('Densification video must be local and at most 100 MB')
    source_hash, provenance = digest(video), model_provenance(model)
    version = importlib.metadata.version('mediapipe')
    original = json.loads(evidence_path.read_text())
    cached, selected, missing = _plan(original, source_hash, provenance, version, stride)
    cap = cv2.VideoCapture(str(video))
    count, width, height, fps = (cap.get(key) for key in (cv2.CAP_PROP_FRAME_COUNT, cv2.CAP_PROP_FRAME_WIDTH,
                                                      cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
    if not cap.isOpened() or any(value != original[key] for key, value in
            (('frame_count', count), ('width', width), ('height', height), ('fps', fps))):
        cap.release()
        raise ValueError('Decoder dimensions, count or clock differ from the cached source')
    width, height, count = int(width), int(height), int(count)
    if not 64 <= width <= 4096 or not 64 <= height <= 2160:
        cap.release()
        raise ValueError('Source resolution exceeds the existing pose envelope')
    out.mkdir(parents=True)
    write_json(out / 'pre-run.json', {'method': METHOD, 'created_at': local_time(), 'source_sha256': source_hash,
        'cached_run': str(run), 'cached_verification': verified, 'cached_evidence_sha256': canonical_hash(original),
        'model': provenance, 'mediapipe_version': version, 'requested_stride': stride,
        'selected_frames': selected, 'cached_frames': sorted(cached), 'new_frames': missing,
        'maximum_inference_calls': len(missing), 'maximum_wall_seconds': 90, 'paid_requests': 0,
        'scope': 'Additional image observations only; no interpolation, relabeling or reference/ground fitting'})
    started = time.perf_counter()
    evidence = deepcopy(original)
    evidence['frames'] = []
    evidence.pop('reference', None)
    evidence.pop('quality', None)
    evidence.update(inference_stride=stride if selected == list(range(0, count, stride)) else None,
        sample_rate_fps=fps/stride if selected == list(range(0, count, stride)) else None,
        densification={'method': METHOD, 'input_evidence_sha256': canonical_hash(original),
            'requested_stride': stride, 'cached_samples': len(cached), 'new_samples': len(missing),
            'selection': 'Union of every cached sample and requested stride; exact source-frame times'})
    selected_set, preview_frames = set(selected), set(selected[::max(1, math.ceil(len(selected)/48))])
    preview, preview_times, inferred = [], [], []
    resized_width = min(960, width)
    size = (resized_width, round(height*resized_width/width))
    detector, failure, quality_error, decoded = None, None, None, 0
    try:
        mp, detector = _detector(model, 'image')
        with (out / 'new-observations.jsonl').open('x') as journal:
            for frame in range(count+1):
                if time.perf_counter()-started > 90:
                    raise TimeoutError('Dense pose inference exceeded its 90-second local budget')
                ok, bgr = cap.read()
                if not ok:
                    if frame != count: raise ValueError('Source ended before its complete retained interval')
                    break
                if frame == count: raise ValueError('Source exceeds its declared full interval')
                decoded += 1
                stamp = float(cap.get(cv2.CAP_PROP_POS_MSEC))/1000
                if bgr.shape[:2] != (height, width) or not math.isfinite(stamp) or abs(stamp-frame/fps) > 1e-7:
                    raise ValueError('Source resolution or decoder clock changed')
                if frame == 0 and not cv2.imwrite(str(out/'background.jpg'), bgr):
                    raise OSError('Source preview could not be written')
                if frame not in selected_set: continue
                resized = cv2.resize(bgr, size, interpolation=cv2.INTER_AREA)
                if frame in cached:
                    row = deepcopy(cached[frame])
                else:
                    begin = time.perf_counter()
                    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(resized, cv2.COLOR_BGR2RGB))
                    row = summarize_result(detector.detect(image)) | {'frame': frame, 'time_s': frame/fps,
                        'decoder_timestamp_s': stamp, 'id': f'{source_hash}:frame:{frame}',
                        'inference_seconds': time.perf_counter()-begin}
                    journal.write(json.dumps(row, allow_nan=False)+'\n')
                    journal.flush()
                    inferred.append(frame)
                evidence['frames'].append(row)
                if frame in preview_frames:
                    preview.append(_draw_pose(resized, row)); preview_times.append(row['time_s'])
        if digest(video) != source_hash or verify_artifacts(run) != verified or model_provenance(model) != provenance:
            raise ValueError('Source, cached input or model changed during inference')
        if inferred != missing or any(canonical_hash(row) != canonical_hash(cached[row['frame']])
                for row in evidence['frames'] if row['frame'] in cached):
            raise ValueError('Cached samples changed or inference did not cover exactly the missing frames')
    except Exception as error:
        failure = {'type': type(error).__name__, 'message': str(error)}
        raise
    finally:
        if detector is not None: detector.close()
        cap.release()
        evidence.update(decoded_frames=decoded, elapsed_seconds=time.perf_counter()-started)
        write_json(out / 'pose-landmarks.json', {'frames': evidence['frames'], 'model': evidence['model'], 'scope': evidence['scope']})
        write_json(out / 'evidence.json', evidence)
        _save_previews(preview, preview_times, out)
        if failure:
            write_json(out / 'report.json', {'status': 'failed_densification', 'failure': failure,
                'new_observations': len(inferred), 'decoded_frames': decoded, 'paid_requests': 0})
    resolve_pose_identity(evidence, out, 'raw')
    try:
        finalize_evidence(evidence, out)
    except UnsupportedVideo as error:
        quality_error = str(error)
    report = {'created_at': local_time(), 'status': 'completed_densification', 'wall_seconds': time.perf_counter()-started,
        'decoded_frames': decoded, 'sampled_frames': len(selected), 'reused_observations': len(cached),
        'new_inference_calls': len(inferred), 'cached_samples_unchanged': True, 'source_and_model_unchanged': True,
        'whole_body_quality_passes': evidence['quality']['passes'], 'whole_body_quality_error': quality_error,
        'paid_requests': 0, 'scope': 'Raw image evidence; completion does not qualify a whole-body simulation'}
    write_json(out / 'report.json', report)
    return report


def main():
    parser = argparse.ArgumentParser()
    for key in ('run', 'video', 'model', 'out'): parser.add_argument('--'+key, type=Path, required=True)
    parser.add_argument('--stride', type=int, default=1)
    args = parser.parse_args()
    try:
        report = densify_image_poses(args.run, args.video, args.model, args.out, stride=args.stride)
    except Exception:
        if (args.out/'report.json').is_file() and not (args.out/'manifest.json').exists(): seal_suite(args.out)
        raise
    print(json.dumps({'report': report, 'verification': seal_suite(args.out)}, indent=2))


if __name__ == '__main__':
    main()
