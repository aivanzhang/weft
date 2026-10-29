"""One fixed feature method across complete saved masks; no physics or new inference."""

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from hashlib import sha256
import json
from pathlib import Path
import shutil
import time

import cv2
import numpy as np
from PIL import Image, ImageDraw

from weft.benchmark import seal_suite
from weft.io import digest, local_time, write_json
from weft.pipeline import _implementation_files
from weft.surface_features import CONTRACT as FEATURE_CONTRACT, detect_surface_features, match_surface_features


ROOT = Path(__file__).resolve().parents[2]
CASES = {
    'ramp-side': 'artifacts/spatial-ramp-sandbox-02',
    'ramp-front': 'artifacts/meta-phys101-ramp-masks-01',
    'toy-van': 'artifacts/decoded-masks-01/toy-van/decoded',
    'three-balls': 'artifacts/decoded-masks-01/three-ball/decoded',
    'exercise': 'artifacts/meta-exercise-silhouette-masks-02',
}
CONTRACT = {'method': FEATURE_CONTRACT, 'maximum_workers': 2, 'maximum_case_seconds': 60,
    'maximum_case_bytes': 96 * 1024**2, 'maximum_frames_per_case': 500, 'feature_batch_frames': 32,
    'reference_rule': 'Declared initialization only; never chain, replace, refresh or shift to an easier frame',
    'preview_reference_ids': 'First 24 raw detector indices; every slot remains in the journal',
    'new_paid_calls': 0, 'new_inference': False, 'new_physics': False, 'default_changes': False,
    'scope': 'Five views of four exposed scene recordings; two ramp views are the same event',
    'physical_point_identity_verified': False, 'rigidity_verified': False}


def read(path):
    return json.loads(path.read_text())


def verified(root, relative, manifest):
    path = root/relative
    if digest(path) != manifest['files'][relative]:
        raise ValueError(f'Source artifact changed: {relative}')
    return path


def load_case(name, folder):
    manifest = read(folder/'manifest.json')
    if name == 'ramp-side':
        evidence = read(verified(folder, 'side-image-evidence.json', manifest))
        source = ROOT/'artifacts/phys101-constraints-source-01/member-02-Camera_2.mp4'
        objects = [{'id': 'green-box', 'initialization': {'frame': 0}}]
        rows = [{'frame': row['frame'], 'time_s': row['time_s'], 'object_id': 'green-box',
            'id': f'{row["source_sha256"]}:frame:{row["frame"]}:color-component',
            'mask_status': 'decoded', 'raster_path': row['mask'],
            'box': {'box_xyxy_halfopen': [400, 350, 800, 750]},
            'identity': {'physical_identity_verified': False, 'status': 'source_color_component'}} for row in evidence['rows']]
        times = [row['time_s'] for row in rows]; size = [1920, 1080]; source_hash = evidence['rows'][0]['source_sha256']
    else:
        evidence = read(verified(folder, 'mask-evidence.json', manifest))
        original = read(verified(folder, 'original-evidence.json', manifest))
        source = verified(folder, 'source.mp4', manifest)
        objects, rows, times = original['objects'], evidence['observations'], original['time']['timestamps_s']
        size = [original['source']['width_px'], original['source']['height_px']]
        source_hash = original['source']['submitted_video_sha256']
    if digest(source) != source_hash or not 1 <= len(times) <= CONTRACT['maximum_frames_per_case']:
        raise ValueError('Source identity or frame envelope changed')
    slots = {(row['frame'], row['object_id']): row for row in rows}
    if len(slots) != len(rows) or set(slots) != {(frame, obj['id']) for frame in range(len(times)) for obj in objects}:
        raise ValueError('Every source object/frame slot must be retained')
    if any(abs(row['time_s']-times[row['frame']]) > 1e-9 for row in rows):
        raise ValueError('Mask/source clocks disagree')
    return {'name': name, 'folder': folder, 'manifest': manifest, 'source': source, 'source_sha256': source_hash,
        'objects': objects, 'rows': rows, 'slots': slots, 'times': times, 'size': size}


def source_mask(case, row):
    width, height = case['size']; output = np.zeros((height, width), np.uint8)
    if row['mask_status'] != 'decoded':
        return output
    path = verified(case['folder'], row['raster_path'], case['manifest'])
    with Image.open(path) as image:
        raster = np.asarray(image).copy()
    if raster.ndim != 2 or not np.isin(raster, [0, 1, 255]).all():
        raise ValueError('Expected an unchanged binary mask raster')
    if row.get('raster_sha256') and sha256(raster.tobytes()).hexdigest() != row['raster_sha256']:
        raise ValueError('Decoded mask-pixel hash changed')
    x0, y0, x1, y1 = row['box']['box_xyxy_halfopen']
    if any(type(v) is not int for v in (x0, y0, x1, y1)) or not 0 <= x0 < x1 <= width or not 0 <= y0 < y1 <= height:
        raise ValueError('Invalid source mask cell mapping')
    # Map source pixel centers into the saved half-open raster cells without interpolation.
    columns = np.floor((np.arange(x1-x0)+.5)*raster.shape[1]/(x1-x0)).astype(int)
    rows = np.floor((np.arange(y1-y0)+.5)*raster.shape[0]/(y1-y0)).astype(int)
    output[y0:y1, x0:x1] = (raster[np.ix_(rows, columns)] != 0).astype(np.uint8)*255
    return output


def view_transform(case):
    boxes = [row['box']['box_xyxy_halfopen'] for row in case['rows'] if row['mask_status'] == 'decoded']
    bounds = np.asarray(boxes)
    x0 = max(0, int(bounds[:, 0].min())-24); y0 = max(0, int(bounds[:, 1].min())-24)
    x1 = min(case['size'][0], int(bounds[:, 2].max())+24); y1 = min(case['size'][1], int(bounds[:, 3].max())+24)
    scale = min(640/(x1-x0), 400/(y1-y0))
    return [x0, y0, x1, y1], scale


def process_case(case, out):
    out.mkdir(); batch = {}; batch_index = None; references = {}; reference_previews = {}
    first_counts = {}; row_counts = Counter(); pairs_per_object = {obj['id']: [] for obj in case['objects']}
    times_ms = []; frames = 0; feature_slots = 0; status = 'complete'; failure = None
    crop, display_scale = view_transform(case)
    write_json(out/'source-binding.json', {'source': str(case['source'].relative_to(ROOT)), 'sha256': case['source_sha256'],
        'source_manifest_sha256': digest(case['folder']/'manifest.json'), 'size': case['size'], 'times_s': case['times'],
        'objects': case['objects'], 'preview_crop_xyxy': crop, 'preview_scale': display_scale})
    write_json(out/'original-mask-slots.json', case['rows'])
    capture = cv2.VideoCapture(str(case['source']))
    fps = (len(case['times'])-1)/(case['times'][-1]-case['times'][0]) if len(case['times']) > 1 else 1.
    writer = cv2.VideoWriter(str(out/'appearance-tracks.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), fps, (1280, 460))
    if not capture.isOpened() or not writer.isOpened():
        raise RuntimeError('Source decoder or track-video encoder unavailable')
    selected_previews = set(np.rint(np.linspace(0, len(case['times'])-1, 6)).astype(int))
    started = time.perf_counter()

    def flush():
        nonlocal batch
        if batch:
            np.savez_compressed(out/f'features-{batch_index:04d}.npz', **batch)
            batch = {}

    try:
        with (out/'all-frame-tracks.jsonl').open('x') as stream:
            for frame, stamp in enumerate(case['times']):
                ok, bgr = capture.read()
                if not ok or bgr.shape[:2] != (case['size'][1], case['size'][0]):
                    raise ValueError('Complete source frame unavailable')
                decoded_stamp = capture.get(cv2.CAP_PROP_POS_MSEC)/1000
                if abs(decoded_stamp-stamp) > 1e-6:
                    raise ValueError(f'Source clock mismatch at frame {frame}')
                next_batch = frame//CONTRACT['feature_batch_frames']
                if next_batch != batch_index:
                    flush(); batch_index = next_batch
                gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
                current_image = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)).crop(crop)
                current_image = current_image.resize((round(current_image.width*display_scale), round(current_image.height*display_scale)))
                panel = Image.new('RGB', (1280, 460), '#F5EADC'); draw = ImageDraw.Draw(panel)
                panel.paste(current_image, (640, 30)); trace = {'frame': frame, 'time_s': stamp, 'objects': []}
                display_lines = 0
                for object_index, obj in enumerate(case['objects']):
                    identifier = obj['id']; observation = case['slots'][(frame, identifier)]
                    row = {'object_id': identifier, 'source_observation_id': observation['id'],
                        'source_identity': observation['identity'], 'mask_status': observation['mask_status'],
                        'status': 'before_initialization', 'matches': None, 'features': None}
                    if frame >= obj['initialization']['frame']:
                        tick = time.perf_counter()
                        if observation['mask_status'] != 'decoded':
                            features = detect_surface_features(gray, np.zeros_like(gray)); row['status'] = 'mask_unavailable'
                        else:
                            try:
                                features = detect_surface_features(gray, source_mask(case, observation)); row['status'] = 'features_observed'
                            except ValueError as error:
                                if 'crop pixel budget' not in str(error):raise
                                features = detect_surface_features(gray, np.zeros_like(gray))
                                row.update(status='feature_region_exceeds_bound', rejection=str(error))
                        prefix = f'f{frame:05d}_o{object_index}'
                        for key in ('keypoints', 'descriptors', 'selected_indices', 'aliases'):
                            batch[f'{prefix}_{key}'] = np.asarray(features[key])
                        row['features'] = {'file': f'features-{batch_index:04d}.npz', 'prefix': prefix, 'audit': features['audit']}
                        if frame == obj['initialization']['frame']:
                            references[identifier] = features; reference_previews[identifier] = current_image.copy()
                            first_counts[identifier] = {'raw': len(features['keypoints']), 'unique': len(features['selected_indices']),
                                'frame': frame, 'feature_file': row['features']}
                        if identifier not in references:
                            raise ValueError('Missing declared initialization; no later-frame substitution')
                        matches = match_surface_features(references[identifier], features)
                        row['matches'] = matches; feature_slots += len(matches['rows'])
                        if not references[identifier]['selected_indices']:
                            row['status'] = 'reference_has_no_features'
                        pairs_per_object[identifier].append({'frame': frame, 'time_s': stamp, 'pairs': matches['appearance_pairs']})
                        row_counts.update(match['status'] for match in matches['rows'])
                        times_ms.append((time.perf_counter()-tick)*1000)
                        if object_index == 0:panel.paste(reference_previews[identifier], (0, 30))
                        for match in matches['rows'][:24]:
                            if match['status'] != 'appearance_pair':continue
                            start = (np.asarray(match['reference_point_px'])-crop[:2])*display_scale + [0, 30]
                            end = (np.asarray(match['current_point_px'])-crop[:2])*display_scale + [640, 30]
                            draw.line([tuple(start), tuple(end)], fill='#00A8C0', width=1)
                            for x, y in (start, end):draw.ellipse((x-2, y-2, x+2, y+2), fill='#00D7FF')
                            display_lines += 1
                    trace['objects'].append(row)
                stream.write(json.dumps(trace, separators=(',', ':'), allow_nan=False)+'\n')
                draw.text((8, 8), f'{case["name"]} | fixed initialization reference', fill='#214F50')
                draw.text((648, 8), f'Source frame {frame} | {stamp:.3f}s', fill='#214F50')
                counts = ', '.join(f'{obj["object_id"]}: {obj["matches"]["appearance_pairs"] if obj["matches"] else 0}' for obj in trace['objects'])
                draw.text((8, 433), f'Appearance pairs: {counts}. Display: first 24 reference IDs per object ({display_lines} lines).', fill='#214F50')
                draw.text((8, 448), 'Not verified physical point identity, 3D shape, camera calibration or anatomy. All source slots and failures retained.', fill='#214F50')
                writer.write(cv2.cvtColor(np.asarray(panel), cv2.COLOR_RGB2BGR))
                if frame in selected_previews:panel.save(out/f'preview-{frame:04d}.jpg', quality=92)
                frames += 1
                if time.perf_counter()-started > CONTRACT['maximum_case_seconds']:
                    raise TimeoutError('Feature extraction exceeded its case deadline')
            if capture.read()[0]:raise ValueError('Unexpected source suffix')
    except Exception as error:
        failure = {'type': type(error).__name__, 'error': str(error)}; status = 'failed'
        write_json(out/'failure.json', failure)
    finally:
        flush(); capture.release(); writer.release()
    summary = {}
    for obj in case['objects']:
        identifier = obj['id']; pairs = pairs_per_object[identifier]
        after = [row for row in pairs if row['frame'] > obj['initialization']['frame']]
        summary[identifier] = {'reference_features': first_counts.get(identifier),
            'later_frames': len(after), 'later_frames_with_pairs': sum(row['pairs'] > 0 for row in after),
            'later_frames_with_at_least_four_pairs': sum(row['pairs'] >= 4 for row in after),
            'later_pairs': sum(row['pairs'] for row in after), 'per_frame': pairs}
    bytes_used = sum(path.stat().st_size for path in out.rglob('*') if path.is_file())
    if bytes_used > CONTRACT['maximum_case_bytes']:
        status = 'failed'; failure = {'type': 'OutputBound', 'error': 'Case export exceeds byte envelope'}
        write_json(out/'failure.json', failure)
    report = {'created_at': local_time(), 'status': status, 'failure': failure,
        'frames_decoded': frames, 'source_frames': len(case['times']), 'source_object_slots': len(case['rows']),
        'feature_slots': feature_slots, 'matching_status_counts': dict(row_counts), 'by_object': summary,
        'elapsed_seconds': time.perf_counter()-started, 'bytes_before_report_and_manifest': bytes_used,
        'feature_step_ms': {'count': len(times_ms), 'median': float(np.median(times_ms)) if times_ms else None,
            'p95': float(np.percentile(times_ms, 95)) if times_ms else None},
        'new_paid_calls': 0, 'physical_point_identity_verified': False, 'rigidity_verified': False,
        'limitations': ['First-frame reference and masks are supplied, not automatic object discovery.',
            'Descriptor support can include adjacent texture outside the inset mask.',
            'Repeated features, occlusion, nonrigidity and source edits can defeat physical correspondence.',
            'Zero matches do not imply no object; matched features do not establish anatomy or camera calibration.']}
    write_json(out/'report.json', report); seal_suite(out)
    return report


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--out', type=Path, required=True)
    out = parser.parse_args().out.resolve()
    if out.parent != ROOT/'artifacts':raise ValueError('Use a new top-level study directory')
    cases = [load_case(name, ROOT/path) for name, path in CASES.items()]
    out.mkdir(exist_ok=False); shutil.copyfile(__file__, out/Path(__file__).name)
    runtime = _implementation_files()
    for name in ('surface_features.py', 'regional_features.py'):shutil.copyfile(runtime[name], out/name)
    shutil.copyfile(ROOT/'tests/test_surface_features.py', out/'test_surface_features.py')
    write_json(out/'pre-run.json', {'created_at': local_time(), 'contract': CONTRACT,
        'sources': [{key: case[key] for key in ('name', 'source_sha256', 'size', 'times', 'objects')} for case in cases],
        'input_manifest_hashes': {case['name']: digest(case['folder']/'manifest.json') for case in cases},
        'implementation': {name: digest(path) for name, path in runtime.items()},
        'test_report_sha256': digest(ROOT/'.local/surface-features-tests-01.xml')})
    cv2.setNumThreads(1); started = time.perf_counter(); results = {}
    with ThreadPoolExecutor(max_workers=2) as pool:
        pending = {pool.submit(process_case, case, out/case['name']): case['name'] for case in cases}
        for future in as_completed(pending):
            name = pending[future]
            try:results[name] = future.result()
            except Exception as error:results[name] = {'status': 'failed', 'failure': {'type': type(error).__name__, 'error': str(error)}}
    summary = {'created_at': local_time(), 'status': 'complete' if all(row['status']=='complete' for row in results.values()) else 'failed',
        'contract': CONTRACT, 'cases': results, 'elapsed_seconds': time.perf_counter()-started,
        'new_paid_calls': 0, 'new_model_inference': False, 'new_environment': False}
    write_json(out/'report.json', summary)
    links = ''.join(f'<h2>{name}</h2><p>{row["status"]}: {row.get("frames_decoded",0)} source frames.</p><video controls preload="metadata" src="{name}/appearance-tracks.mp4"></video><p><a href="{name}/report.json">Counts and limitations</a> · <a href="{name}/all-frame-tracks.jsonl">Every frame and point slot</a></p>' for name,row in sorted(results.items()))
    (out/'index.html').write_text(f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>Weft · automatic appearance correspondences</title>
<style>body{{font:17px system-ui;line-height:1.5;max-width:1280px;margin:30px auto;padding:0 20px}}video{{max-width:100%}}</style>
<h1>Automatic features across different videos</h1><p>The same bounded method matches each frame directly to the declared initialization. Saved masks supply regions. No hand-marked feature points, chained transforms or new model calls.</p>
<p><strong>Appearance evidence, not verified 3D identity or physics.</strong> Missing/ambiguous matches remain. These are five views of four exposed scene recordings, not unseen-video validation.</p>{links}<p><a href="pre-run.json">Frozen method and inputs</a> · <a href="report.json">Complete results</a></p></html>''')
    seal = seal_suite(out)
    print(json.dumps({'status': summary['status'], 'seconds': summary['elapsed_seconds'],
        'cases': {name: {'status': row['status'], 'frames': row.get('frames_decoded'), 'feature_slots': row.get('feature_slots'),
            'objects': {identifier: {k:v for k,v in obj.items() if k not in ('per_frame','reference_features')} for identifier,obj in row.get('by_object',{}).items()},
            'failure': row.get('failure')} for name,row in results.items()}, 'verification': seal}))


if __name__ == '__main__':main()
