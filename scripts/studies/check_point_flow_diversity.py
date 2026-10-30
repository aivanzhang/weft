"""Compare immutable-seed image flow with saved descriptor histories on complete sources."""

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import shutil
import time

import cv2
import numpy as np
from PIL import Image, ImageDraw

from check_surface_feature_diversity import CASES, load_case, source_mask
from check_track_shape_diversity import seal, summary
from weft.io import digest, local_time, write_json
from weft.pipeline import verify_artifacts
from weft.point_flow import CONTRACT as FLOW_CONTRACT, OpticalPointHistories
from weft.track_shape import CONTRACT as SHAPE_CONTRACT, reconstruct_track_shape


ROOT = Path(__file__).resolve().parents[2]
FEATURES = ROOT/'artifacts/surface-feature-diversity-01'
HISTORIES = ROOT/'artifacts/feature-track-diversity-01'
CONTRACT = {'point_flow': dict(FLOW_CONTRACT), 'shape': dict(SHAPE_CONTRACT),
    'maximum_workers': 2, 'maximum_case_seconds': 60, 'maximum_case_bytes': 96*1024**2,
    'maximum_total_bytes': 384*1024**2, 'shape_windows': [8, 32],
    'seed_rule': 'All saved unique SIFT locations at the original declared initialization; no reseeding or revival.',
    'shape_window_rule': 'Nonoverlapping original-frame-zero windows at each fixed length; retain every suffix.',
    'control': {'frames': 75, 'missing_mask_frames': [31, 32], 'minimum_final_tracks': 6,
        'maximum_point_error_px': 3., 'p95_point_error_px': 1.},
    'new_paid_calls': 0, 'new_source_feature_detection': False, 'new_pose_or_segmentation_inference': False,
    'new_physics': False, 'defaults_changed': False, 'source_scope': 'Five exposed views of four recordings.',
    'meaning': 'Longer image histories are not independently verified physical correspondence or geometry.'}


def read(path):
    return json.loads(path.read_text())


def shape_windows(journal, out, binding):
    histories = {obj['id']: [] for obj in binding['objects']}
    for line in journal.open():
        frame = json.loads(line)
        for obj in frame['objects']:
            histories[obj['object_id']].append((frame['frame'], frame['time_s'], obj['flow']))
    counts = {}; candidates = 0
    with (out/'shape-windows.jsonl').open('x') as stream:
        for length in CONTRACT['shape_windows']:
            statuses = Counter()
            for identifier, rows in histories.items():
                for start in range(0, len(rows), length):
                    group = rows[start:start+length]
                    ids = sorted({r['seed_id'] for _, _, flow in group if flow for r in flow['rows']})
                    points = np.full((len(group), len(ids), 2), np.nan)
                    for index, (_, _, flow) in enumerate(group):
                        if flow:
                            for row in flow['rows']:
                                if row['point_px'] is not None:points[index, row['seed_id']] = row['point_px']
                    result = reconstruct_track_shape(points, frame_ids=[r[0] for r in group],
                        times_s=[r[1] for r in group], track_ids=ids, image_size=binding['size'], scope='construction')
                    statuses[result['status']] += 1; candidates += result['status'] == 'conditional_shape'
                    stream.write(json.dumps({'object_id': identifier, 'requested_window_frames': length,
                        'result': result}, separators=(',', ':'), allow_nan=False)+'\n')
            counts[str(length)] = dict(statuses)
    return {'by_window_length': counts, 'conditional_windows': candidates,
        'world_promoted': False, 'physical_accuracy_verified': False}


def process_case(name, out):
    case = load_case(name, ROOT/CASES[name]); binding = read(FEATURES/name/'source-binding.json')
    out.mkdir(); write_json(out/'source-binding.json', binding)
    write_json(out/'input-binding.json', {'mask_source': CASES[name],
        'mask_manifest_sha256': digest(case['folder']/'manifest.json'),
        'feature_manifest_sha256': digest(FEATURES/'manifest.json'),
        'history_manifest_sha256': digest(HISTORIES/'manifest.json')})
    capture = cv2.VideoCapture(str(case['source'])); crop = binding['preview_crop_xyxy']; scale = binding['preview_scale']
    fps = (len(case['times'])-1)/(case['times'][-1]-case['times'][0])
    writer = cv2.VideoWriter(str(out/'point-histories.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), fps, (1280, 460))
    if not capture.isOpened() or not writer.isOpened():raise RuntimeError('Video codec unavailable')
    trackers = {}; seeds = {}; original_previews = {}; counts = Counter(); timings = []; errors = []
    by_object = {obj['id']: [] for obj in case['objects']}; previews = set(np.rint(np.linspace(0, len(case['times'])-1, 6)).astype(int))
    started = time.perf_counter(); decoded = 0
    try:
        with (FEATURES/name/'all-frame-tracks.jsonl').open() as features, (HISTORIES/name/'all-frame-tracklets.jsonl').open() as baseline, (out/'all-frame-points.jsonl').open('x') as journal:
            for source_line, baseline_line in zip(features, baseline, strict=True):
                old, prior = json.loads(source_line), json.loads(baseline_line)
                frame = old['frame']; stamp = old['time_s']; ok, bgr = capture.read()
                if (not ok or frame != decoded or stamp != case['times'][frame]
                        or prior['frame'] != frame or prior['time_s'] != stamp
                        or abs(capture.get(cv2.CAP_PROP_POS_MSEC)/1000-stamp) > 1e-6):
                    raise ValueError('Source frame/clock coverage changed')
                decoded += 1; gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
                preview = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)).crop(crop)
                preview = preview.resize((round(preview.width*scale), round(preview.height*scale)))
                panel = Image.new('RGB', (1280, 460), '#F5EADC'); panel.paste(preview, (640, 30)); draw = ImageDraw.Draw(panel)
                output = {'frame': frame, 'time_s': stamp, 'objects': []}
                for object_index, obj in enumerate(old['objects']):
                    identifier = obj['object_id']; result = None; tick = time.perf_counter()
                    reference = prior['objects'][object_index]
                    if reference['object_id'] != identifier:raise ValueError('Object identity order changed')
                    original_anchors = {r['reference_id']: r['point_px'] for r in reference['tracks']['rows']
                        if r['reference_id'] is not None} if reference['tracks'] else {}
                    if obj['features'] is not None:
                        region = source_mask(case, case['slots'][(frame, identifier)])
                        if identifier not in trackers:
                            declared = next(o for o in case['objects'] if o['id'] == identifier)['initialization']['frame']
                            if frame != declared:raise ValueError('Initialization frame shifted')
                            locator = obj['features']
                            with np.load(FEATURES/name/locator['file'], allow_pickle=False) as archive:
                                selected = archive[locator['prefix']+'_selected_indices'].astype(int)
                                points = archive[locator['prefix']+'_keypoints'][selected, :2]
                            seeds[identifier] = {'frame': frame, 'source_indices': selected.tolist(), 'points_px': points.tolist(), 'features': locator}
                            tracker = OpticalPointHistories(gray, region, points, frame=frame, time_s=stamp)
                            trackers[identifier] = tracker; result = tracker.initialization
                            original_previews[identifier] = preview.copy()
                        else:result = trackers[identifier].advance(gray, region, frame=frame, time_s=stamp)
                        timings.append((time.perf_counter()-tick)*1000)
                    elif identifier in trackers:raise ValueError('Post-initialization source feature slot missing')
                    row = {k: obj[k] for k in ('object_id', 'source_observation_id', 'source_identity', 'mask_status', 'status')}
                    row['flow'] = result; row['baseline_opening_anchors'] = len(original_anchors); row['shared_point_differences'] = []
                    if result:
                        if object_index == 0:panel.paste(original_previews[identifier], (0, 30))
                        counts.update(r['status'] for r in result['rows'])
                        for point in result['rows']:
                            if point['point_px'] is None:continue
                            source_id = seeds[identifier]['source_indices'][point['seed_id']]
                            if source_id in original_anchors:
                                error = float(np.linalg.norm(np.asarray(point['point_px'])-original_anchors[source_id])); errors.append(error)
                                row['shared_point_differences'].append({'seed_id': point['seed_id'], 'source_feature_id': source_id,
                                    'baseline_point_px': original_anchors[source_id], 'distance_px': error,
                                    'meaning': 'Method disagreement, not ground-truth error.'})
                            if point['seed_id'] < 24:
                                start = (np.asarray(point['original_point_px'])-crop[:2])*scale+[0, 30]
                                end = (np.asarray(point['point_px'])-crop[:2])*scale+[640, 30]
                                draw.line([tuple(start), tuple(end)], fill='#00A8C0', width=1)
                                draw.ellipse((*tuple(end-2), *tuple(end+2)), fill='#00A8C0')
                    by_object[identifier].append({'frame': frame, 'time_s': stamp,
                        'active_original_points': result['active_points'] if result else 0,
                        'baseline_opening_anchors': len(original_anchors)})
                    output['objects'].append(row)
                journal.write(json.dumps(output, separators=(',', ':'), allow_nan=False)+'\n')
                draw.text((8, 8), f'{name}: immutable seed image', fill='#214F50')
                draw.text((648, 8), f'Frame {frame} | {stamp:.3f}s', fill='#214F50')
                draw.text((8, 432), ', '.join(f'{o["object_id"]}: {o["flow"]["active_points"] if o["flow"] else 0} active original points' for o in output['objects']), fill='#214F50')
                draw.text((8, 449), 'First 24 seed IDs drawn. Ended points never resume. Not verified physical identity, anatomy or source-edit continuity.', fill='#214F50')
                writer.write(cv2.cvtColor(np.asarray(panel), cv2.COLOR_RGB2BGR))
                if frame in previews:panel.save(out/f'preview-{frame:04d}.jpg', quality=92)
                if time.perf_counter()-started > CONTRACT['maximum_case_seconds']:raise TimeoutError('Case deadline exceeded')
                if journal.tell() > CONTRACT['maximum_case_bytes']:raise ValueError('Case byte envelope exceeded')
            if decoded != len(case['times']) or capture.read()[0]:raise ValueError('Incomplete source retention')
    finally:
        capture.release(); writer.release()
    write_json(out/'seeds.json', seeds)
    shape = shape_windows(out/'all-frame-points.jsonl', out, binding)
    totals = {}
    for identifier, values in by_object.items():
        initialization = next(o for o in case['objects'] if o['id'] == identifier)['initialization']['frame']
        later = [r for r in values if r['frame'] > initialization]
        totals[identifier] = {'later_frames': len(later),
            'frames_with_original_points': sum(r['active_original_points'] > 0 for r in later),
            'frames_with_six_original_points': sum(r['active_original_points'] >= 6 for r in later),
            'baseline_frames_with_opening_anchors': sum(r['baseline_opening_anchors'] > 0 for r in later),
            'last_frame_points': values[-1]['active_original_points'], 'per_frame': values}
    report = {'status': 'complete', 'source_frames': decoded, 'source_object_slots': decoded*len(case['objects']),
        'seed_slots': sum(len(s['source_indices']) for s in seeds.values()), 'status_counts': dict(counts),
        'by_object': totals, 'method_disagreement_px': summary(errors), 'tracking_ms': summary(timings),
        'shape': shape, 'seconds': time.perf_counter()-started,
        'new_pose_or_segmentation_inference': False, 'new_paid_calls': 0, 'new_world': False}
    write_json(out/'report.json', report); seal(out)
    return report


def synthetic(out):
    out.mkdir(); source_dir = HISTORIES/'synthetic'
    with Image.open(source_dir/'original-texture.png') as im:original = np.asarray(im).copy()
    with Image.open(source_dir/'original-mask.png') as im:mask = np.asarray(im).copy()
    inputs = [json.loads(line) for line in (source_dir/'all-frame-controls.jsonl').open()]
    seeds = [r for r in inputs[0]['tracks']['rows'] if r['track_id'] is not None]
    points = np.array([r['point_px'] for r in seeds])
    trackers = {name: OpticalPointHistories(original, mask, points, frame=0, time_s=0.) for name in ('continuous', 'missing_masks')}
    errors = {name: [] for name in trackers}; last = {}; counts = Counter()
    write_json(out/'input-binding.json', {'manifest_sha256': digest(HISTORIES/'manifest.json'),
        'input_control': str((source_dir/'all-frame-controls.jsonl').relative_to(ROOT)), 'seed_points_px': points.tolist()})
    with (out/'known-warp-points.jsonl').open('x') as stream:
        for item in inputs:
            frame = item['frame']; matrix = np.asarray(item['transform'])
            image = cv2.warpAffine(original, matrix[:2], original.shape[::-1], borderValue=128)
            region = cv2.warpAffine(mask, matrix[:2], original.shape[::-1], flags=cv2.INTER_NEAREST)
            truth = (points-.5) @ matrix[:2, :2].T+matrix[:2, 2]+.5
            for name, tracker in trackers.items():
                current_mask = np.zeros_like(region) if name == 'missing_masks' and frame in CONTRACT['control']['missing_mask_frames'] else region
                result = tracker.initialization if frame == 0 else tracker.advance(image, current_mask, frame=frame, time_s=frame*.04)
                distances = [{'seed_id': row['seed_id'], 'error_px': float(np.linalg.norm(np.asarray(row['point_px'])-truth[row['seed_id']])) if row['point_px'] is not None else None} for row in result['rows']]
                errors[name].extend(row['error_px'] for row in distances if row['error_px'] is not None)
                stream.write(json.dumps({'condition': name, 'frame': frame, 'transform': matrix.tolist(), 'flow': result,
                    'known_point_errors': distances}, separators=(',', ':'), allow_nan=False)+'\n')
                last[name] = result['active_points']; counts[name] += 1
    metrics = {name: summary(values) for name, values in errors.items()}
    gates = {'continuous_final_coverage': last['continuous'] >= CONTRACT['control']['minimum_final_tracks'],
        'missing_mask_tracks_stay_ended': last['missing_masks'] == 0,
        'point_accuracy': all(v['count'] > 0 and v['p95'] <= 1. and v['maximum'] <= 3. for v in metrics.values())}
    report = {'status': 'complete', 'conditions': dict(counts), 'last_active_points': last,
        'point_error_px': metrics, 'gates': gates, 'passes': all(gates.values()),
        'meaning': 'Known planar software warp; not real physical correspondence or 3D shape.'}
    write_json(out/'report.json', report); seal(out)
    return report


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--out', type=Path, required=True)
    out = parser.parse_args().out.resolve()
    if out.parent != ROOT/'artifacts':raise ValueError('Use a new top-level study directory')
    input_checks = {str(p.relative_to(ROOT)): verify_artifacts(p) for p in (FEATURES, HISTORIES)}
    out.mkdir(exist_ok=False)
    used = [Path(__file__), Path(__file__).with_name('check_surface_feature_diversity.py'),
        Path(__file__).with_name('check_track_shape_diversity.py'), ROOT/'src/weft/point_flow.py',
        ROOT/'src/weft/track_shape.py', ROOT/'src/weft/feature_tracks.py', ROOT/'tests/test_point_flow.py']
    hashes = {str(p.relative_to(ROOT)): digest(p) for p in used}
    for path in used:shutil.copyfile(path, out/path.name)
    write_json(out/'pre-run.json', {'created_at': local_time(), 'contract': CONTRACT,
        'implementation': hashes, 'input_checks': input_checks, 'opencv_version': cv2.__version__,
        'test_report_sha256': digest(ROOT/'.local/point-flow-tests-05.xml'),
        'cost': 'Local CPU, two workers, zero paid calls/downloads/recurring resources; at most 60 s per real case.'})
    cv2.setNumThreads(1); started = time.perf_counter()
    try:
        with ThreadPoolExecutor(max_workers=CONTRACT['maximum_workers']) as pool:
            jobs = {name: pool.submit(process_case, name, out/name) for name in sorted(CASES)}
            controlled = pool.submit(synthetic, out/'synthetic')
            cases = {name: job.result() for name, job in jobs.items()}; controls = controlled.result()
        if any(digest(ROOT/name) != expected for name, expected in hashes.items()):raise ValueError('Used source changed during study')
        size = sum(p.stat().st_size for p in out.rglob('*') if p.is_file())
        if size > CONTRACT['maximum_total_bytes']:raise ValueError('Total output byte bound exceeded')
        report = {'created_at': local_time(), 'status': 'complete', 'contract': CONTRACT, 'cases': cases,
            'synthetic': controls, 'seconds': time.perf_counter()-started, 'bytes_before_report_and_manifest': size,
            'new_paid_calls': 0, 'new_world': False}
    except Exception as error:
        write_json(out/'report.json', {'status': 'failed', 'failure': {'type': type(error).__name__, 'message': str(error)}})
        seal(out); raise
    write_json(out/'report.json', report)
    sections = ''.join(f'<h2>{name}</h2><video controls preload="metadata" src="{name}/point-histories.mp4"></video><p><a href="{name}/report.json">Coverage and disagreements</a> · <a href="{name}/all-frame-points.jsonl">Every seed and source frame</a> · <a href="{name}/shape-windows.jsonl">All reconstruction windows</a></p>' for name in cases)
    (out/'index.html').write_text('<!doctype html><html lang="en"><meta charset="utf-8"><title>Weft · original-point histories</title><style>body{font:16px system-ui;max-width:1280px;margin:30px auto;padding:0 20px;line-height:1.5}video{max-width:100%}</style><h1>Longer histories for original image points</h1><p>Local flow proposes movement. Forward/backward checks, local appearance and an unchanged original image patch validate each proposal. Missing or rejected points end permanently.</p><p><strong>Appearance agreement is not verified physical identity, anatomy, depth or continuity across edits.</strong> Source/object identity uncertainty remains. No world is promoted from these studies.</p>'+sections+'<p><a href="synthetic/report.json">Known-warp drift controls</a> · <a href="pre-run.json">Frozen inputs and contract</a> · <a href="report.json">Full report</a></p></html>')
    seal(out)
    print(json.dumps({'status': 'complete', 'seconds': report['seconds'], 'cases': {k: {'frames': v['source_frames'], 'shape': v['shape']} for k,v in cases.items()}, 'synthetic': controls, 'manifest_sha256': digest(out/'manifest.json')}))


if __name__ == '__main__':main()
