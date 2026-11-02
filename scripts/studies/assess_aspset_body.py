"""Compare a frozen camera body and recorded episode with assessment-only ASPset joints."""

import argparse
from collections import Counter
import html
import json
from pathlib import Path
import sys
import time
import zipfile

import numpy as np

from weft.camera_reference import JOINTS, WORLD_CONTRACT, compare_world_frame
from weft.io import canonical_hash, digest, local_time, write_json


MAX_BYTES = 32 << 20
MAX_FILES = 512
CAMERA_TO_NOMINAL = np.array([[1., 0., 0.], [0., 0., 1.], [0., -1., 0.]])


def safe_path(value):
    path = Path(value).absolute()
    if '..' in path.parts or any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError('Assessment paths cannot traverse or follow symlinks')
    return path.resolve()


def read(path, maximum=8 << 20):
    if path.is_symlink() or not path.is_file() or path.stat().st_size > maximum:
        raise ValueError(f'Expected a bounded regular file: {path}')
    return json.loads(path.read_bytes())


def verify(root):
    manifest = read(root/'manifest.json', 256 << 10)
    files = manifest.get('files')
    if not isinstance(files, dict) or not 1 <= len(files) <= MAX_FILES or 'manifest.json' in files:
        raise ValueError('Expected a sealed package without a manifest self entry')
    total = 0
    for name, expected in files.items():
        path = root/name
        if (not isinstance(expected, str) or Path(name).is_absolute() or '..' in Path(name).parts
                or any(p.is_symlink() for p in (path, *path.parents)) or not path.is_file()
                or path.stat().st_size > 64 << 20 or digest(path) != expected):
            raise ValueError(f'Package member differs: {name}')
        total += path.stat().st_size
        if total > 400 << 20:
            raise ValueError('Input package exceeds the aggregate byte bound')
    return {'manifest_sha256': digest(root/'manifest.json'), 'members': len(files)}


def stats(values):
    if not values:
        return {'count': 0, 'mean': None, 'rmse': None, 'p95': None, 'maximum': None}
    values = np.asarray(values, dtype=float)
    return {'count': len(values), 'mean': float(np.mean(values)), 'rmse': float(np.sqrt(np.mean(values**2))),
            'p95': float(np.percentile(values, 95)), 'maximum': float(np.max(values))}


def frame_index(stamp, fps, count):
    if type(stamp) not in (int, float) or not np.isfinite(stamp):
        raise ValueError('Prediction requires a finite source timestamp')
    index = round(stamp*fps)
    if not 0 <= index < count or abs(stamp-index/fps) > 1e-8:
        raise ValueError('Prediction is not on the original source grid; no time fitting is allowed')
    return index


def validate_source_identity(body, camera, source, reference_source, assumptions):
    if (reference_source.get('source_video_unit') != f"{source.get('subject')}-{source.get('clip')}"
            or reference_source.get('reference_for_construction') is not False
            or reference_source.get('source_video_manifest_sha256') != assumptions.get('source_manifest_sha256')
            or not reference_source.get('source_video_manifest_sha256')
            or source.get('sha256') != camera.get('source_sha256')
            or source.get('sha256') != body.get('source_sha256') or not source.get('sha256')):
        raise ValueError('Published reference, camera and frozen body must identify the same original source unit')


def validate_fixed_camera(camera):
    count, fps = camera['frame_count'], camera['fps']
    frames = camera['frames']
    if (camera.get('units') != 'm' or camera.get('world_axes') != 'right_handed_z_up'
            or len(frames) != count or not frames or frames[0]['t'] != [0., 0., 0.]
            or any(row['frame'] != i or abs(row['time_s']-i/fps) > 1e-8
                   or row['R'] != frames[0]['R'] or row['t'] != frames[0]['t'] for i,row in enumerate(frames))):
        raise ValueError('ASPset assessment requires the fixed metric left-camera frame and original clock')


def summarize(rows, roots):
    result = {'joint_slots': len(rows), 'root_distance_mm': stats(roots),
              'absolute_statuses': dict(Counter(row['absolute_status'] for row in rows)),
              'root_relative_statuses': dict(Counter(row['root_relative_status'] for row in rows))}
    for name in ('absolute', 'root_relative'):
        key = name+'_distance_mm'
        result[key] = stats([row[key] for row in rows if row[key] is not None])
        result[name+'_by_joint_mm'] = {joint: stats([row[key] for row in rows if row['joint'] == joint and row[key] is not None]) for joint in JOINTS}
    return result


def render_report(report, comparisons_href='joint-comparisons.json'):
    labels = {'triangulated': '3D joints from camera images', 'constructed_body': 'Reconstructed body poses',
              'recorded_simulation': 'Recorded simulated motion'}
    def number(value):
        return 'unavailable' if value is None else f'{value:.2f}'
    def table(rows, caption):
        body = ''.join('<tr>'+''.join('<td>'+html.escape(str(v))+'</td>' for v in row)+'</tr>' for row in rows)
        return ('<table><caption>'+html.escape(caption)+'</caption><thead><tr><th scope="col">Stage</th>'
                '<th scope="col">Joint comparisons</th><th scope="col">Position error (mm RMSE)</th></tr></thead><tbody>'+body+'</tbody></table>')
    matched = [[labels[k], v['count'], number(v['rmse'])] for k,v in report['matched_absolute_distance_mm'].items()]
    available = [[labels[k], v['absolute_distance_mm']['count'], number(v['absolute_distance_mm']['rmse'])] for k,v in report['summaries'].items()]
    return ('<!doctype html><html lang="en"><meta charset="utf-8"><title>Weft reference comparison</title><main>'
        '<h1>How closely does the simulation match?</h1><p>The saved body and recording are compared with a separate published 3D joint reference. Neither input was changed.</p>'
        '<h2>Compare the same points</h2><p>Each row uses exactly the same available joints and timestamps. Lower error means closer agreement with this reference.</p>'
        +table(matched, 'Matched comparison — the fair comparison between stages')+
        '<p>RMSE means root mean square error; larger mistakes carry more weight. 100 mm equals 10 cm. Positions use the original coordinates, without fitting an alignment.</p>'
        '<h2>All available results</h2><p>These rows cover different numbers of frames and joints. Missing reconstructed poses remain missing; simulated motion is sampled at every recorded frame.</p>'
        +table(available, 'Full coverage — do not compare these rows as equal-sized samples')+
        '<h2>What this does not prove</h2><p>The published reference has unverified capture and joint-definition accuracy. This is not a physical-accuracy certificate.</p>'
        '<p>The recording uses programmed control and explicit body-position assistance. Ground support, equipment contact, muscle activity and unfamiliar subjects remain unverified.</p>'
        '<p><a href="report.json">Details: per-joint errors, body-position error, relative pose and worst frames</a> · '
        '<a href="'+html.escape(comparisons_href, quote=True)+'">Every original joint slot</a></p></main></html>\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('run', 'reference', 'camera-inputs', 'recording', 'out'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--episode', type=int, required=True, help='Zero-based recorded start/reset interval, selected before assessment')
    args = parser.parse_args(); started = time.monotonic()
    run, reference, inputs, recording, out = [safe_path(p) for p in (args.run, args.reference, args.camera_inputs, args.recording, args.out)]
    roots = (run, reference, inputs, recording)
    if out.exists() or any(out.is_relative_to(p) or p.is_relative_to(out) for p in roots):
        raise ValueError('Use a new assessment output outside all inputs')
    pins = {str(p): verify(p) for p in roots}
    body, native, triangulation = read(run/'human.json'), read(run/'human-reference.json'), read(run/'two-view-triangulation.json')
    camera = read(run/'camera.json'); count, fps = camera['frame_count'], camera['fps']
    metadata = read(reference/'report.json')
    if (type(count) is not int or not 1 <= count <= 1000 or not 1 <= fps <= 240
            or (metadata['units'], metadata['point_rate'], metadata['frame_count'], metadata['first_frame'], metadata['last_frame']) != ('mm', fps, count, 1, count)
            or metadata['source_sha256'] != digest(reference/'reference.c3d')):
        raise ValueError('Published reference and original source clock disagree')
    validate_fixed_camera(camera)
    source = read(reference/'source.json')
    source_input = read(run/'source.json')
    assumptions = read(inputs/'reuse-assumptions.json')
    validate_source_identity(body, camera, source_input, source, assumptions)
    if (run/'camera.json').read_bytes() != (inputs/'camera-left.json').read_bytes():
        raise ValueError('Assessment coordinates require the exact construction camera input')
    array_path = reference/'published-reference.npz'
    if array_path.stat().st_size > 1 << 20:
        raise ValueError('Published arrays exceed the input bound')
    with zipfile.ZipFile(array_path) as archive:
        if set(archive.namelist()) != {'points.npy', 'frame_ids.npy'} or len(archive.infolist()) != 2 or sum(i.file_size for i in archive.infolist()) > 1 << 20:
            raise ValueError('Published array layout exceeds bounds')
    with np.load(array_path, allow_pickle=False) as archive:
        points, ids = archive['points'].copy(), archive['frame_ids'].copy()
    labels = metadata['labels']
    if (points.shape != (count, len(labels), 5) or len(labels) != len(set(labels)) or not set(JOINTS) <= set(labels)
            or not np.array_equal(ids, np.arange(1, count+1)) or not np.isfinite(points).all()):
        raise ValueError('Published point identity/layout differs from its protocol')
    transform = assumptions['world_transform']
    rotation = np.asarray(transform['R']) @ CAMERA_TO_NOMINAL
    if (transform['direction'] != 'original_world_to_conditional_ground_world' or transform['scale'] != 1
            or transform['t_m'] != [0., 0., 0.] or rotation.shape != (3, 3)
            or not np.allclose(rotation @ rotation.T, np.eye(3), atol=1e-10, rtol=0)
            or not np.allclose(np.asarray(camera['frames'][0]['R']), rotation.T, atol=1e-10, rtol=0)):
        raise ValueError('Declared publisher-to-world mapping differs from the frozen left camera')
    world = points[:,:,:3].astype(float)*.001 @ rotation.T
    reference_rows = [{name: world[i, labels.index(name)].tolist() if points[i, labels.index(name), 3] >= 0 else None for name in JOINTS} for i in range(count)]
    offset = np.r_[body['camera_reference']['origin_world_m'][:2], body['camera_reference']['ground']['height_m']]
    predictions = {'triangulated': {}, 'constructed_body': {}, 'recorded_simulation': {}}
    if len(triangulation['frames']) != count:
        raise ValueError('Triangulated source slots changed')
    for i, row in enumerate(triangulation['frames']):
        if row['frame'] != i or len(row['joints']) != 33:
            raise ValueError('Triangulated frame/joint identity changed')
        predictions['triangulated'][i] = {name: row['joints'][j]['world_point_m'] if row['joints'][j]['status'] == 'estimated' else None for name,j in JOINTS.items()}
    for row in native['frames']:
        index = frame_index(row['source_time_s'], fps, count)
        if index in predictions['constructed_body']:
            raise ValueError('Duplicate body reference frame')
        predictions['constructed_body'][index] = {name: (np.asarray(row['human']['joints'][name])+offset).tolist() for name in JOINTS}
    if len(predictions['constructed_body']) != len(body['reference']):
        raise ValueError('Saved native body preview omits source references')
    scene = read(recording/'scene.json')['engine_scene']
    if scene['human_spec'] != body:
        raise ValueError('Recording uses a different body')
    coupling = scene['camera_coupling']; local = coupling['world_to_local']; placement = coupling['shared_placement']
    if (placement != {'origin': [0., 0., 0.], 'scale': 1.} or local['rotation'] != np.eye(3).tolist()
            or local['scale'] != 1 or not np.allclose(local['translation'], -offset, atol=1e-12, rtol=0)):
        raise ValueError('Recorded world requires a different explicit placement transform')
    journal = recording/'events.jsonl'
    if journal.stat().st_size > 16 << 20:
        raise ValueError('Recorded journal exceeds the assessment bound')
    body_hash, coupling_hash = canonical_hash(body), canonical_hash(coupling)
    episode = -1; controllers = Counter(); event_count = 0
    with journal.open() as stream:
        for expected_index, line in enumerate(stream):
            if expected_index >= 5000 or len(line) > 1 << 20:
                raise ValueError('Recorded journal exceeds its event/line bound')
            event = json.loads(line)
            if event['index'] != expected_index or event['type'] not in ('start', 'reset', 'step'):
                raise ValueError('Recorded event identity/type is unsupported')
            if event['type'] in ('start', 'reset'):
                episode += 1
            if episode != args.episode:
                continue
            snapshot = event['snapshot']; state = snapshot['engine_state']
            if snapshot['body_sha256'] != body_hash or snapshot['camera_coupling_sha256'] != coupling_hash:
                raise ValueError('Recorded snapshot binding changed')
            index = frame_index(state['source_time_s'], fps, count)
            if index in predictions['recorded_simulation']:
                raise ValueError('Selected episode has duplicate source frames')
            predictions['recorded_simulation'][index] = {name: (np.asarray(state['human']['joints'][name])+offset).tolist() for name in JOINTS}
            controllers[event.get('controller', event['type'])] += 1; event_count += 1
    if not event_count:
        raise ValueError('Selected recorded episode is unavailable')
    rows, summaries = {}, {}
    for mode, sequence in predictions.items():
        mode_rows, root_errors = [], []
        for index in range(count):
            predicted = sequence.get(index, dict.fromkeys(JOINTS))
            result = compare_world_frame(predicted, reference_rows[index])
            if result['hip_midpoint_distance_mm'] is not None:
                root_errors.append(result['hip_midpoint_distance_mm'])
            for name, values in result['joints'].items():
                mode_rows.append({'frame': index, 'reference_frame': index+1, 'time_s': index/fps, 'joint': name,
                                  'prediction_world_m': predicted[name], 'reference_world_m': reference_rows[index][name], **values})
        rows[mode] = mode_rows; summaries[mode] = summarize(mode_rows, root_errors)
    common = [i for i, values in enumerate(rows['constructed_body']) if all(rows[m][i]['absolute_status'] == 'compared' for m in rows)]
    matched = {m: stats([values[i]['absolute_distance_mm'] for i in common]) for m, values in rows.items()}
    worst = {m: sorted([v for v in values if v['absolute_distance_mm'] is not None], key=lambda v: v['absolute_distance_mm'], reverse=True)[:12] for m,values in rows.items()}
    report = {'profile': 'aspset-frozen-body-assessment-v1', 'created_at': local_time(), 'contract': WORLD_CONTRACT,
        'partition': 'exposed_development', 'physical_validation': 'unverified', 'accepted': None,
        'reference_quality': metadata['scope'], 'reference_frame_mapping': 'C3D frame 1 maps source frame 0; no time fit',
        'runtime': {'python': sys.version.split()[0], 'numpy': np.__version__},
        'reference_sha256': metadata['source_sha256'], 'frame_count': count, 'fps': fps, 'joint_names': list(JOINTS),
        'source_manifests': pins, 'world_mapping': {'publisher_mm_to_world_rotation': rotation.tolist(), 'scale': .001, 'translation_m': [0.,0.,0.]},
        'selected_episode': args.episode, 'selected_episode_events': event_count, 'selected_episode_controllers': dict(controllers),
        'summaries': summaries, 'matched_available_joint_slots': len(common), 'matched_absolute_distance_mm': matched,
        'worst_errors': worst, 'new_inference_optimization_physics_calls': 0, 'seconds': time.monotonic()-started,
        'limitations': ['Reference accuracy and joint-definition equivalence are not independently measured',
                       'Sparse reconstructed poses and dense recorded states have different coverage; use the matched subset to compare',
                       'Root-relative distance removes translation only; absolute error is retained separately',
                       'No floor, contact, muscle activation, unseen-subject or physical-transfer validation',
                       'Recorded episode uses explicit root assistance and authored control']}
    if time.monotonic()-started > 25:
        raise TimeoutError('Saved assessment exceeded its deadline')
    out.mkdir(parents=True)
    write_json(out/'report.json', report); write_json(out/'joint-comparisons.json', rows)
    for source_path, name in [(Path(__file__), 'assess_aspset_body.py'), (reference/'report.json', 'reference-metadata.json'),
                              (reference/'source.json', 'reference-source.json'), (reference/'published-reference.npz', 'published-reference.npz'),
                              (inputs/'reuse-assumptions.json', 'coordinate-assumptions.json')]:
        (out/name).write_bytes(source_path.read_bytes())
    import weft.camera_reference
    import weft.io
    (out/'camera_reference.py').write_bytes(Path(weft.camera_reference.__file__).read_bytes())
    (out/'io.py').write_bytes(Path(weft.io.__file__).read_bytes())
    (out/'index.html').write_text(render_report(report))
    files = {p.name: digest(p) for p in sorted(out.iterdir())}
    if len(files) > 32 or sum((out/n).stat().st_size for n in files) > MAX_BYTES:
        raise ValueError('Assessment output exceeds its budget')
    write_json(out/'manifest.json', {'schema_version': 1, 'files': files})
    print(json.dumps({'verification': verify(out), 'summaries': {m:{k:v[k] for k in ('joint_slots','absolute_distance_mm','root_relative_distance_mm','root_distance_mm')} for m,v in summaries.items()}, 'matched_available_joint_slots':len(common),'seconds':report['seconds']}))


if __name__ == '__main__':
    main()
