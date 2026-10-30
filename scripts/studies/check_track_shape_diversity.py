"""Attempt relative shape on fixed complete-source windows without filling track gaps."""

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import shutil
import time

import numpy as np
from PIL import Image, ImageDraw

from weft.io import digest, local_time, write_json
from weft.pipeline import verify_artifacts
from weft.track_shape import CONTRACT as SHAPE_CONTRACT, reconstruct_track_shape


ROOT = Path(__file__).resolve().parents[2]
INPUT = ROOT/'artifacts/feature-track-diversity-01'
CONTRACT = {'shape': dict(SHAPE_CONTRACT), 'window_frames': 8, 'window_stride': 8,
    'window_origin': 'Original source frame zero, including pre-initialization and the short suffix.',
    'maximum_workers': 2, 'maximum_case_seconds': 30, 'maximum_case_bytes': 16*1024**2,
    'maximum_total_bytes': 96*1024**2, 'new_paid_calls': 0, 'new_model_inference': False,
    'new_video_decode': False, 'new_physics': False, 'default_changes': False,
    'source_scope': 'Five already exposed views of four recordings; not fresh validation.',
    'control_seeds': list(range(701, 709)), 'control_noise_px': [0., .1, .25],
    'control_distance_rms_limit': .03, 'maximum_workload_repeats': 20,
    'interpretation': 'Conditional relative visible point shape; not anatomy, a collision envelope or measured physics.'}


def read(path):
    return json.loads(path.read_text())


def seal(directory):
    write_json(directory/'manifest.json', {'schema': 1, 'created_at': local_time(),
        'files': {str(p.relative_to(directory)): digest(p) for p in sorted(directory.rglob('*')) if p.is_file()}})
    return verify_artifacts(directory)


def summary(values):
    return {'count': len(values), 'median': float(np.median(values)) if values else None,
        'p95': float(np.percentile(values, 95)) if values else None,
        'maximum': float(max(values)) if values else None}


def window_input(frames, identifier):
    objects = [next(o for o in f['objects'] if o['object_id'] == identifier) for f in frames]
    rows = [{r['track_id']: r['point_px'] for r in o['tracks']['rows'] if r['track_id'] is not None}
        if o['tracks'] else {} for o in objects]
    track_ids = sorted(set().union(*(set(r) for r in rows)))
    points = np.full((len(frames), len(track_ids), 2), np.nan)
    for index, row in enumerate(rows):
        for column, track in enumerate(track_ids):
            if track in row:points[index, column] = row[track]
    return objects, track_ids, points


def preview(window, path, title):
    result = window['result']; candidate = result['candidate']
    image = Image.new('RGB', (1100, 500), 'white'); draw = ImageDraw.Draw(image)
    draw.text((18, 12), title, fill='black')
    draw.text((18, 32), f"{result['status']} | frames {result['frame_ids'][0]}-{result['frame_ids'][-1]} | {len(result['complete_track_ids'])} complete tracks", fill='black')
    included = [window['track_ids'].index(t) for t in result['complete_track_ids']]
    observed = np.asarray(window['points_px'], dtype=float)[:, included]
    if result['affine_fit']:
        proposed = np.asarray((candidate or result['affine_fit'])['predicted_points_px'])
        low = np.minimum(observed.min(axis=(0, 1)), proposed.min(axis=(0, 1)))
        high = np.maximum(observed.max(axis=(0, 1)), proposed.max(axis=(0, 1)))
        scale = 380/max(float((high-low).max()), 1.)
        for f in (0, len(observed)-1):
            origin = np.array([25+f//(len(observed)-1)*530, 85])
            for actual, predicted in zip(observed[f], proposed[f], strict=True):
                a = (actual-low)*scale+origin; b = (predicted-low)*scale+origin
                draw.ellipse((*tuple(a-3), *tuple(a+3)), outline='#008DA8', width=2)
                draw.line([tuple(b-3), tuple(b+3)], fill='#B96521', width=2)
                draw.line([(b[0]-3, b[1]+3), (b[0]+3, b[1]-3)], fill='#B96521', width=2)
            draw.text(tuple(origin-[0, 20]), f"Source frame {result['frame_ids'][f]}", fill='black')
    draw.text((18, 468), 'Cyan: original complete-track observations. Orange: fitted projection; not independent accuracy.', fill='black')
    draw.text((18, 484), 'Missing and partial tracks remain in the journals. A small image error does not establish true 3D shape.', fill='black')
    image.save(path)


def process_case(name, out):
    directory = INPUT/name; binding = read(directory/'source-binding.json')
    out.mkdir(); write_json(out/'source-binding.json', binding)
    journal = directory/'all-frame-tracklets.jsonl'
    write_json(out/'input-binding.json', {'input_manifest_sha256': digest(INPUT/'manifest.json'),
        'case_manifest_sha256': digest(directory/'manifest.json'),
        'all_frame_tracklets': str(journal.relative_to(ROOT)), 'sha256': digest(journal)})
    ids = [o['id'] for o in binding['objects']]
    counts = {identifier: Counter() for identifier in ids}; timings = []; table = []
    raw_points = 0; frames_seen = 0; fit_slots = 0; window_points = 0; selected_preview = None
    started = time.perf_counter(); pending = []
    with journal.open() as source, (out/'windows.jsonl').open('x') as target:
        def consume(frames):
            nonlocal fit_slots, window_points, selected_preview
            for identifier in ids:
                objects, tracks, points = window_input(frames, identifier)
                tick = time.perf_counter()
                result = reconstruct_track_shape(points, frame_ids=[f['frame'] for f in frames],
                    times_s=[f['time_s'] for f in frames], track_ids=tracks,
                    image_size=binding['size'], scope='construction')
                timings.append((time.perf_counter()-tick)*1000)
                counts[identifier][result['status']] += 1
                fit_slots += len(result['complete_track_ids'])*len(frames)
                window_points += result['coverage']['observed_point_slots']
                row = {'object_id': identifier, 'frame_ids': result['frame_ids'], 'times_s': result['times_s'],
                    'track_ids': tracks, 'points_px': [[p.tolist() if np.isfinite(p).all() else [None, None]
                        for p in frame] for frame in points],
                    'source_object_slots': [{k: v for k, v in obj.items() if k != 'tracks'} for obj in objects],
                    'result': result}
                target.write(json.dumps(row, separators=(',', ':'), allow_nan=False)+'\n')
                table.append({'object_id': identifier, 'first_frame': frames[0]['frame'],
                    'last_frame': frames[-1]['frame'], 'status': result['status'],
                    'complete_tracks': len(result['complete_track_ids']),
                    'third_singular_ratio': result['third_singular_ratio'],
                    'affine_rms_px': result['affine_fit']['errors']['rms_px'] if result['affine_fit'] else None,
                    'candidate_rms_px': result['candidate']['errors']['rms_px'] if result['candidate'] else None})
                if result['affine_fit'] and (selected_preview is None or
                        result['status'] == 'conditional_shape' and selected_preview['result']['status'] != 'conditional_shape'):
                    selected_preview = row
                if target.tell() > CONTRACT['maximum_case_bytes']:
                    raise ValueError('Case byte bound exceeded')
                if time.perf_counter()-started > CONTRACT['maximum_case_seconds']:
                    raise TimeoutError('Case time bound exceeded')

        for line in source:
            frame = json.loads(line)
            if frame['frame'] != frames_seen or frame['time_s'] != binding['times_s'][frames_seen]:
                raise ValueError('Source frame/clock mismatch')
            if [o['object_id'] for o in frame['objects']] != ids:
                raise ValueError('Source object order or coverage changed')
            raw_points += sum(len(o['tracks']['rows']) if o['tracks'] else 0 for o in frame['objects'])
            pending.append(frame); frames_seen += 1
            if len(pending) == CONTRACT['window_frames']:
                consume(pending); pending = []
        if pending:consume(pending)
    if frames_seen != len(binding['times_s']):raise ValueError('Source suffix missing')
    if selected_preview is not None:preview(selected_preview, out/'projection-check.png', name)
    report = {'status': 'complete', 'source_frames': frames_seen, 'object_slots': frames_seen*len(ids),
        'raw_input_point_slots': raw_points, 'distinct_observed_point_slots': window_points,
        'complete_track_point_slots': fit_slots, 'by_object': {k: dict(v) for k, v in counts.items()},
        'windows': table, 'function_ms': summary(timings), 'seconds': time.perf_counter()-started,
        'new_world': False, 'new_paid_calls': 0}
    write_json(out/'report.json', report); seal(out)
    return report


def control(seed, frames=16, count=32):
    rng = np.random.default_rng(seed)
    truth = rng.uniform(-1, 1, (count, 3))*[1, .8, .6]
    cameras = []
    for t in np.linspace(-1, 1, frames):
        a, b, c = .65*t, .5*np.sin(2*t), .3*np.cos(2*t)
        x = np.array([[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]])
        y = np.array([[np.cos(b), 0, np.sin(b)], [0, 1, 0], [-np.sin(b), 0, np.cos(b)]])
        z = np.array([[np.cos(c), -np.sin(c), 0], [np.sin(c), np.cos(c), 0], [0, 0, 1]])
        cameras.append((z @ y @ x)[:2]*(90+20*t))
    centers = np.array([[320+i, 240-i] for i in range(frames)])
    pixels = np.einsum('fij,pj->fpi', cameras, truth)+centers[:, None, :]
    return truth, pixels


def distances(points):
    points = np.asarray(points)-np.mean(points, axis=0)
    points /= np.sqrt(np.mean(np.sum(points**2, axis=1)))
    delta = points[:, None, :]-points[None, :, :]
    return np.linalg.norm(delta, axis=2)[np.triu_indices(len(points), 1)]


def fit_control(points):
    return reconstruct_track_shape(points, frame_ids=list(range(len(points))),
        times_s=(np.arange(len(points))*.04).tolist(), track_ids=list(range(points.shape[1])),
        image_size=[640, 480], scope='construction')


def run_controls(out):
    out.mkdir(); cases = []; timing = []
    with (out/'controls.jsonl').open('x') as journal:
        for seed in CONTRACT['control_seeds']:
            truth, points = control(seed)
            variants = [(f'rigid-noise-{noise}', points+np.random.default_rng(seed+900).normal(0, noise, points.shape))
                for noise in CONTRACT['control_noise_px']]
            plane = np.repeat(points[:1], len(points), axis=0)
            mismatch = points.copy(); mismatch[5, 0] += [70, -45]
            variants += [('no_depth_motion', plane), ('wrong_correspondence', mismatch),
                ('two_view_orientations', np.tile(points[[0, -1]], (8, 1, 1))),
                ('coherent_scale_deformation', (points-[320, 240])*np.linspace(.8, 1.2, len(points))[:, None, None]+[320, 240])]
            for name, observed in variants:
                result = fit_control(observed)
                error = None
                if name.startswith('rigid-noise') and result['candidate'] is not None:
                    error = float(np.sqrt(np.mean((distances(result['candidate']['points_relative'])-distances(truth))**2)))
                row = {'seed': seed, 'case': name, 'truth_points': truth.tolist(),
                    'observed_px': observed.tolist(), 'result': result, 'distance_rms_error': error,
                    'physical_rigidity_false_by_construction': name == 'coherent_scale_deformation'}
                journal.write(json.dumps(row, separators=(',', ':'), allow_nan=False)+'\n')
                cases.append({k: row[k] for k in ('seed', 'case', 'distance_rms_error')} |
                    {'status': result['status'], 'reprojection_rms_px': result['candidate']['errors']['rms_px'] if result['candidate'] else None})
        _, largest = control(1901, frames=32, count=128)
        for _ in range(CONTRACT['maximum_workload_repeats']):
            tick = time.perf_counter(); result = fit_control(largest)
            timing.append((time.perf_counter()-tick)*1000)
            if result['status'] != 'conditional_shape':raise ValueError('Maximum-workload control failed')
    report = {'status': 'complete', 'cases': cases, 'maximum_workload_ms': summary(timing),
        'interpretation': 'Known geometry software controls; coherent scale deformation can pass the camera model.'}
    write_json(out/'report.json', report); seal(out)
    return report


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--out', type=Path, required=True)
    out = parser.parse_args().out.resolve()
    if out.parent != ROOT/'artifacts':raise ValueError('Use a new top-level artifact directory')
    input_check = verify_artifacts(INPUT)
    names = sorted(read(INPUT/'report.json')['cases'])
    out.mkdir(exist_ok=False)
    sources = [Path(__file__), ROOT/'src/weft/track_shape.py', ROOT/'tests/test_track_shape.py',
        ROOT/'src/weft/io.py', ROOT/'src/weft/pipeline.py']
    hashes = {str(p.relative_to(ROOT)): digest(p) for p in sources}
    for path in sources:shutil.copyfile(path, out/path.name)
    write_json(out/'pre-run.json', {'created_at': local_time(), 'contract': CONTRACT,
        'input_manifest_sha256': digest(INPUT/'manifest.json'), 'input_check': input_check,
        'implementation': hashes, 'numpy_version': np.__version__,
        'test_report_sha256': digest(ROOT/'.local/track-shape-tests-01.xml'),
        'cost': 'Existing local CPU only; zero new paid requests or recurring resources; under 30 s per case.',
        'references': ['https://www.ri.cmu.edu/pub_files/pub2/tomasi_c_1991_3/tomasi_c_1991_3.pdf',
            'https://www.ri.cmu.edu/pub_files/pub1/poelman_conrad_1993_1/poelman_conrad_1993_1.pdf']})
    started = time.perf_counter()
    try:
        with ThreadPoolExecutor(max_workers=CONTRACT['maximum_workers']) as pool:
            jobs = {name: pool.submit(process_case, name, out/name) for name in names}
            control_job = pool.submit(run_controls, out/'synthetic')
            cases = {name: job.result() for name, job in jobs.items()}
            controlled = control_job.result()
        if any(digest(ROOT/name) != expected for name, expected in hashes.items()):
            raise ValueError('Used implementation changed during the study')
        size = sum(p.stat().st_size for p in out.rglob('*') if p.is_file())
        if size > CONTRACT['maximum_total_bytes']:raise ValueError('Total output bound exceeded')
        report = {'created_at': local_time(), 'status': 'complete', 'contract': CONTRACT,
            'cases': cases, 'synthetic': controlled, 'seconds': time.perf_counter()-started,
            'bytes_before_report_and_manifest': size, 'new_world': False, 'new_paid_calls': 0}
    except Exception as error:
        report = {'created_at': local_time(), 'status': 'failed', 'failure': {
            'type': type(error).__name__, 'message': str(error)}, 'seconds': time.perf_counter()-started}
        write_json(out/'report.json', report); seal(out); raise
    write_json(out/'report.json', report)
    sections = []
    for name, result in cases.items():
        rows = ''.join(f'<tr><td>{r["object_id"]}</td><td>{r["first_frame"]}–{r["last_frame"]}</td><td>{r["complete_tracks"]}</td><td>{r["status"]}</td></tr>' for r in result['windows'])
        picture = f'<img src="{name}/projection-check.png" alt="Original and fitted point projections">' if (out/name/'projection-check.png').exists() else ''
        sections.append(f'<h2>{name}</h2>{picture}<p><a href="{name}/windows.jsonl">All windows, source slots and proposals</a> · <a href="../feature-track-diversity-01/{name}/tracklets.mp4">Original tracked-video preview</a></p><table><tr><th>Object</th><th>Frames</th><th>Complete tracks</th><th>Result</th></tr>{rows}</table>')
    (out/'index.html').write_text('<!doctype html><html lang="en"><meta charset="utf-8"><title>Weft · conditional track shape</title><style>body{font:16px system-ui;max-width:1100px;margin:30px auto;padding:0 20px;line-height:1.5}img{max-width:100%}td,th{padding:5px 12px;text-align:left}</style><h1>Can these tracks constrain a 3D shape?</h1><p>Fixed eight-frame windows retain the entire source, including initialization gaps and short suffixes. Only complete point histories are fitted; nothing is interpolated.</p><p><strong>A conditional point cloud is not a verified rigid object, anatomy, hidden shape, calibrated scale or a new simulator.</strong> Camera assumptions and depth reflection remain explicit. These are exposed development recordings.</p>'+''.join(sections)+'<h2>Known geometry controls</h2><p><a href="synthetic/report.json">Control results and bounded CPU timings</a> · <a href="pre-run.json">Frozen contract and inputs</a> · <a href="report.json">Full report</a></p></html>')
    seal(out)
    print(json.dumps({'status': report['status'], 'seconds': report['seconds'],
        'cases': {k: v['by_object'] for k, v in cases.items()}, 'manifest_sha256': digest(out/'manifest.json')}))


if __name__ == '__main__':
    main()
