"""Track saved descriptors across complete videos; retain breaks and original conflicts."""

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import shutil
import time

import cv2
import numpy as np
from PIL import Image, ImageDraw

from weft.benchmark import seal_suite
from weft.feature_tracks import AppearanceTracklets, CONTRACT as TRACK_CONTRACT
from weft.io import digest, local_time, write_json
from weft.pipeline import _implementation_files, verify_artifacts
from weft.surface_features import CONTRACT as FEATURE_CONTRACT, detect_surface_features


ROOT = Path(__file__).resolve().parents[2]
INPUT = ROOT/'artifacts/surface-feature-diversity-01'
CONTRACT = {'tracker': TRACK_CONTRACT, 'maximum_workers': 2, 'maximum_case_seconds': 60,
    'maximum_case_bytes': 96*1024**2, 'new_paid_calls': 0, 'new_model_inference': False,
    'new_physics': False, 'default_changes': False, 'source_scope': 'Same five views of four exposed recordings',
    'synthetic': {'frames': 75, 'seed': 981, 'image_size': [360, 300],
        'rotation_degrees_per_frame': 2.5, 'missing_mask_frames': [31, 32],
        'point_error_p95_px': 1., 'point_error_max_px': 3.},
    'preview': 'First 24 raw current IDs; solid cyan adjacent links, magenta original-reference conflicts',
    'interpretation': 'More links are not verified physical identity, accurate geometry, anatomy or edit continuity'}


def read(path):
    return json.loads(path.read_text())


def load_features(archive, locator):
    result = {'method': FEATURE_CONTRACT['method'], **{key: archive[locator['prefix']+'_'+key]
        for key in ('keypoints', 'descriptors', 'selected_indices', 'aliases')}}
    for key in ('selected_indices', 'aliases'):result[key] = result[key].tolist()
    return result


def summarize(values):
    if not values:return {'count': 0, 'median': None, 'p95': None, 'maximum': None}
    return {'count': len(values), 'median': float(np.median(values)),
        'p95': float(np.percentile(values, 95)), 'maximum': float(np.max(values))}


def process_case(name, out):
    directory = INPUT/name; binding = read(directory/'source-binding.json')
    source = ROOT/binding['source']
    if digest(source) != binding['sha256']:raise ValueError('Source changed')
    out.mkdir(); write_json(out/'source-binding.json', binding)
    write_json(out/'input-binding.json', {'study_manifest_sha256': digest(INPUT/'manifest.json'),
        'case_manifest_sha256': digest(directory/'manifest.json'),
        'all_frame_tracks_sha256': digest(directory/'all-frame-tracks.jsonl')})
    capture = cv2.VideoCapture(str(source)); crop = binding['preview_crop_xyxy']; scale = binding['preview_scale']
    fps = (len(binding['times_s'])-1)/(binding['times_s'][-1]-binding['times_s'][0])
    writer = cv2.VideoWriter(str(out/'tracklets.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), fps, (1280, 460))
    if not capture.isOpened() or not writer.isOpened():raise RuntimeError('Video decoder/encoder unavailable')
    trackers = {}; previous_rows = {}; previous_image = None; counts = Counter(); timings = []
    by_object = {obj['id']: [] for obj in binding['objects']}; active_peak = 0; frames = 0
    archive = None; archive_name = None; failure = None; status = 'complete'; raw_slots = 0
    selected_previews = set(np.rint(np.linspace(0, len(binding['times_s'])-1, 6)).astype(int))
    started = time.perf_counter()
    try:
        with (directory/'all-frame-tracks.jsonl').open() as original, (out/'all-frame-tracklets.jsonl').open('x') as journal:
            for frame, line in enumerate(original):
                old = json.loads(line); ok, image = capture.read()
                if not ok or old['frame'] != frame or abs(capture.get(cv2.CAP_PROP_POS_MSEC)/1000-old['time_s']) > 1e-6:
                    raise ValueError('Source frames or clocks changed')
                preview = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)).crop(crop)
                preview = preview.resize((round(preview.width*scale), round(preview.height*scale)))
                panel = Image.new('RGB', (1280, 460), '#F5EADC'); draw = ImageDraw.Draw(panel)
                if previous_image is not None:panel.paste(previous_image, (0, 30))
                panel.paste(preview, (640, 30)); output = {'frame': frame, 'time_s': old['time_s'], 'objects': []}
                for obj in old['objects']:
                    identifier = obj['object_id']; locator = obj['features']
                    row = {key: obj[key] for key in ('object_id', 'source_observation_id', 'source_identity', 'mask_status', 'status')}
                    row['features'] = locator; row['tracks'] = None
                    if locator is not None:
                        if archive_name != locator['file']:
                            if archive is not None:archive.close()
                            archive = np.load(directory/locator['file'], allow_pickle=False); archive_name = locator['file']
                        features = load_features(archive, locator); tick = time.perf_counter()
                        if identifier not in trackers:
                            trackers[identifier] = AppearanceTracklets(features, frame=frame, time_s=old['time_s'])
                            result = trackers[identifier].initialization
                        else:
                            result = trackers[identifier].advance(features, frame=frame, time_s=old['time_s'])
                            if result['original_matches'] != obj['matches']:
                                raise ValueError('Original-reference comparison changed')
                        timings.append((time.perf_counter()-tick)*1000)
                        row['tracks'] = result; raw_slots += len(result['rows'])
                        counts.update(item['link_status'] for item in result['rows'])
                        active_peak = max(active_peak, result['active_tracks'])
                        direct_pairs = obj['matches']['appearance_pairs']
                        by_object[identifier].append({'frame': frame, 'time_s': old['time_s'],
                            'direct_reference_pairs': direct_pairs,
                            **{key: result[key] for key in ('active_tracks', 'linked_tracks', 'anchored_tracks', 'conflicted_tracks', 'total_track_ids')},
                            'longest_active_age': max((r['age_frames'] or 0 for r in result['rows']), default=0)})
                        previous = previous_rows.get(identifier, {})
                        for point in result['rows'][:24]:
                            end = (np.asarray(point['point_px'])-crop[:2])*scale+[640, 30]
                            if point['link_status'] == 'adjacent_pair' and point['previous_track_id'] in previous:
                                start = (np.asarray(previous[point['previous_track_id']])-crop[:2])*scale+[0, 30]
                                draw.line([tuple(start), tuple(end)], fill='#00A8C0', width=1)
                                draw.ellipse((end[0]-2, end[1]-2, end[0]+2, end[1]+2), fill='#00D7FF')
                            elif point['link_status'] == 'reference_conflict':
                                draw.ellipse((end[0]-4, end[1]-4, end[0]+4, end[1]+4), outline='#DA2684', width=2)
                        previous_rows[identifier] = {r['track_id']: r['point_px'] for r in result['rows'] if r['track_id'] is not None}
                    output['objects'].append(row)
                journal.write(json.dumps(output, separators=(',', ':'), allow_nan=False)+'\n')
                draw.text((8, 8), f'{name} | preceding frame {frame-1 if frame else "unavailable"}', fill='#214F50')
                draw.text((648, 8), f'Source frame {frame} | {old["time_s"]:.3f}s', fill='#214F50')
                summary = ', '.join(f'{r["object_id"]}: {r["tracks"]["linked_tracks"] if r["tracks"] else 0} links' for r in output['objects'])
                draw.text((8, 433), f'{summary}. First 24 raw current IDs displayed; complete points and breaks retained.', fill='#214F50')
                draw.text((8, 448), 'Image continuity only. Not verified physical identity, 3D motion, anatomy or continuity across edits.', fill='#214F50')
                writer.write(cv2.cvtColor(np.asarray(panel), cv2.COLOR_RGB2BGR))
                if frame in selected_previews:panel.save(out/f'preview-{frame:04d}.jpg', quality=92)
                previous_image = preview; frames += 1
                if time.perf_counter()-started > CONTRACT['maximum_case_seconds']:raise TimeoutError('Case deadline exceeded')
                if journal.tell() > CONTRACT['maximum_case_bytes']:raise ValueError('Journal byte bound exceeded')
            if capture.read()[0]:raise ValueError('Unretained source suffix')
    except Exception as error:
        status = 'failed'; failure = {'type': type(error).__name__, 'error': str(error)}
        write_json(out/'failure.json', failure)
    finally:
        capture.release(); writer.release()
        if archive is not None:archive.close()
    summaries = {}
    for identifier, values in by_object.items():
        later = values[1:]
        summaries[identifier] = {'later_frames': len(later),
            'later_frames_with_direct_reference_pairs': sum(v['direct_reference_pairs'] > 0 for v in later),
            'later_frames_with_any_adjacent_link': sum(v['linked_tracks'] > 0 for v in later),
            'later_frames_with_four_adjacent_links': sum(v['linked_tracks'] >= 4 for v in later),
            'later_frames_with_original_anchor': sum(v['anchored_tracks'] > 0 for v in later),
            'later_frames_without_direct_but_with_adjacent_links': sum(v['direct_reference_pairs'] == 0 and v['linked_tracks'] > 0 for v in later),
            'original_reference_conflict_points': sum(v['conflicted_tracks'] for v in values),
            'total_track_ids': values[-1]['total_track_ids'] if values else 0,
            'maximum_active_age': max((v['longest_active_age'] for v in values), default=0), 'per_frame': values}
    size = sum(path.stat().st_size for path in out.rglob('*') if path.is_file())
    if size > CONTRACT['maximum_case_bytes']:
        status = 'failed'; failure = {'type': 'OutputBound', 'error': 'Case output exceeds byte envelope'}
    report = {'created_at': local_time(), 'status': status, 'failure': failure, 'frames': frames,
        'source_frames': len(binding['times_s']), 'source_object_slots': len(binding['times_s'])*len(binding['objects']),
        'raw_current_point_slots': raw_slots, 'status_counts': dict(counts), 'peak_active_tracks': active_peak,
        'by_object': summaries, 'tracking_step_ms': summarize(timings), 'seconds': time.perf_counter()-started,
        'bytes_before_report_and_manifest': size, 'new_inference': False, 'new_physics': False, 'new_paid_calls': 0,
        'scope': 'Unverified appearance tracklets; breaks retained, tiny balls remain featureless, source edits remain unqualified'}
    write_json(out/'report.json', report); seal_suite(out)
    return report


def synthetic(out):
    out.mkdir(); config = CONTRACT['synthetic']; width, height = config['image_size']
    source = np.full((height, width), 128, np.uint8); rng = np.random.default_rng(config['seed'])
    source[65:235, 80:280] = cv2.GaussianBlur(rng.integers(0, 256, (170, 200), np.uint8), (0, 0), .8)
    mask = np.zeros_like(source); mask[65:235, 80:280] = 255
    Image.fromarray(source).save(out/'original-texture.png'); Image.fromarray(mask).save(out/'original-mask.png')
    features = detect_surface_features(source, mask); tracker = AppearanceTracklets(features, frame=0, time_s=0.)
    initial = tracker.initialization; original = features['keypoints'][:, :2].copy()
    previous = {row['track_id']: row['point_px'] for row in initial['rows'] if row['track_id'] is not None}
    previous_transform = np.eye(3); errors = []; anchor_errors = []; frames = []
    with (out/'all-frame-controls.jsonl').open('x') as journal:
        for frame in range(config['frames']):
            matrix = cv2.getRotationMatrix2D((width/2, height/2), frame*config['rotation_degrees_per_frame'], 1.)
            transform = np.vstack([matrix, [0., 0., 1.]])
            image = cv2.warpAffine(source, matrix, (width, height), borderValue=128)
            region = cv2.warpAffine(mask, matrix, (width, height), flags=cv2.INTER_NEAREST)
            if frame in config['missing_mask_frames']:region[:] = 0
            result = initial if frame == 0 else tracker.advance(detect_surface_features(image, region), frame=frame, time_s=frame*.04)
            step = transform @ np.linalg.inv(previous_transform); graded = []
            for row in result['rows']:
                item = {'current_id': row['current_id'], 'adjacent_error_px': None, 'original_anchor_error_px': None}
                if row['link_status'] == 'adjacent_pair':
                    point = step @ np.r_[np.asarray(previous[row['track_id']])-.5, 1.]
                    error = float(np.linalg.norm(np.asarray(row['point_px'])-(point[:2]/point[2]+.5)))
                    item['adjacent_error_px'] = error; errors.append(error)
                if row['reference_id'] is not None:
                    point = transform @ np.r_[original[row['reference_id']]-.5, 1.]
                    error = float(np.linalg.norm(np.asarray(row['point_px'])-(point[:2]/point[2]+.5)))
                    item['original_anchor_error_px'] = error; anchor_errors.append(error)
                graded.append(item)
            journal.write(json.dumps({'frame': frame, 'transform': transform.tolist(), 'tracks': result, 'known_point_errors': graded}, separators=(',', ':'))+'\n')
            frames.append({'frame': frame, 'linked': result['linked_tracks'], 'anchored': result['anchored_tracks'],
                'active': result['active_tracks'], 'conflicts': result['conflicted_tracks']})
            previous = {row['track_id']: row['point_px'] for row in result['rows'] if row['track_id'] is not None}
            previous_transform = transform
    accuracy = {'adjacent_point_error_px': summarize(errors), 'original_anchor_error_px': summarize(anchor_errors)}
    passes = all(row['count'] > 0 and row['p95'] <= config['point_error_p95_px'] and row['maximum'] <= config['point_error_max_px'] for row in accuracy.values())
    report = {'created_at': local_time(), 'status': 'complete', 'analytic_accuracy_passes': passes,
        'config': config, 'accuracy': accuracy, 'frames': frames,
        'scope': 'Known planar warp, not real-video validation; blank masks force breaks, identical-looking replacements remain undetectable'}
    write_json(out/'report.json', report); seal_suite(out)
    return report


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--out', type=Path, required=True)
    out = parser.parse_args().out.resolve()
    if out.parent != ROOT/'artifacts':raise ValueError('Use a new top-level study directory')
    original = read(INPUT/'report.json'); input_seal = verify_artifacts(INPUT)
    out.mkdir(exist_ok=False)
    for path in (Path(__file__), ROOT/'src/weft/feature_tracks.py', ROOT/'tests/test_feature_tracks.py'):
        shutil.copyfile(path, out/path.name)
    write_json(out/'pre-run.json', {'created_at': local_time(), 'contract': CONTRACT,
        'input_seal': input_seal, 'implementation': {name: digest(path) for name, path in _implementation_files().items()},
        'test_report_sha256': digest(ROOT/'.local/feature-track-tests-01.xml')})
    cv2.setNumThreads(1); started = time.perf_counter(); cases = {}
    with ThreadPoolExecutor(max_workers=CONTRACT['maximum_workers']) as pool:
        pending = {pool.submit(process_case, name, out/name): name for name in original['cases']}
        for future in as_completed(pending):
            name = pending[future]
            try:cases[name] = future.result()
            except Exception as error:cases[name] = {'status': 'failed', 'failure': {'type': type(error).__name__, 'error': str(error)}}
    controlled = synthetic(out/'synthetic')
    result = {'created_at': local_time(), 'status': 'complete' if all(c['status'] == 'complete' for c in cases.values()) else 'failed',
        'contract': CONTRACT, 'cases': cases, 'synthetic': controlled, 'seconds': time.perf_counter()-started,
        'new_model_inference': False, 'new_physics': False, 'new_paid_calls': 0, 'new_environment': False}
    write_json(out/'report.json', result)
    links = ''.join(f'<h2>{name}</h2><p>{row["status"]}: {row.get("frames", 0)} source frames.</p><video controls preload="metadata" src="{name}/tracklets.mp4"></video><p><a href="{name}/report.json">Complete counts and limits</a> · <a href="{name}/all-frame-tracklets.jsonl">Every point, link, break and conflict</a></p>' for name,row in sorted(cases.items()))
    (out/'index.html').write_text(f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>Weft · appearance histories</title>
<style>body{{font:17px system-ui;line-height:1.5;max-width:1280px;margin:30px auto;padding:0 20px}}video{{max-width:100%}}</style>
<h1>Short appearance histories across different videos</h1><p>Neighboring frames can retain matches after the opening appearance changes. The original reference remains fixed; contradictory claims split histories. Missing frames are never bridged.</p>
<p><strong>Not verified physical identity, 3D motion, anatomy or continuity across edits.</strong> Small featureless balls remain unresolved. These are five views of four already exposed scenes.</p>{links}
<p><a href="synthetic/report.json">Known-warp drift control</a> · <a href="pre-run.json">Frozen method</a> · <a href="report.json">Complete results</a></p></html>''')
    seal = seal_suite(out)
    print(json.dumps({'status': result['status'], 'seconds': result['seconds'], 'seal': seal,
        'cases': {name: {obj: {k:v for k,v in values.items() if k != 'per_frame'} for obj,values in row.get('by_object',{}).items()} for name,row in cases.items()},
        'synthetic': controlled['accuracy'], 'analytic_accuracy_passes': controlled['analytic_accuracy_passes']}))


if __name__ == '__main__':main()
