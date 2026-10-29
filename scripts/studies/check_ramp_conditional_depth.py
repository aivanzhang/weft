"""One source-corner depth hypothesis, then complete fixed-material two-view checks."""

import argparse
from copy import deepcopy
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

import cv2
import numpy as np
from PIL import Image, ImageDraw

from check_ramp_side_camera import CONTRACT as SIDE_CONTRACT, projected_centroid, read, summarize, verified
from evaluate_ramp_repeat import exact_replay
from weft.affine_camera import project_affine_points
from weft.benchmark import seal_suite
from weft.io import digest, local_time, write_json
from weft.material_fit import coast_candidate, score_states
from weft.orthographic_shape import conditional_axis_scale
from weft.pipeline import _implementation_files
from weft.primitive_geometry import silhouette_overlap
from weft.recording import Recorder, export_recording, load_recording
from weft.spatial_rigid import corners, rotation, validate_scene


ROOT = Path(__file__).resolve().parents[2]
CAMERA = ROOT / 'artifacts/ramp-side-camera-01'
MATERIAL = ROOT / 'artifacts/spatial-material-fit-01'
MASKS = ROOT / 'artifacts/meta-phys101-ramp-masks-01'
FRONT = ROOT / 'artifacts/phys101-ramp-source-01/member-01-Camera_1.mp4'
CONTRACT = {
    'scope': 'First-trial development; one opening-corner conditional depth, no motion fitting',
    'changed_dimension': 'green-box local y depth only',
    'unchanged': ['mass', 'friction', 'gravity', 'contact solver', 'initial poses and velocities',
                  'other dimensions', 'front camera', 'controls', 'task'],
    'camera_assumption': 'scaled orthographic square pixels; retain both depth-axis assignments',
    'factor_bounds': [.25, 4.], 'side_checks': SIDE_CONTRACT,
    'controls': ['right 4', 'forward 4', 'brake 4', 'reset', 'quit'],
    'maximum_simulation_seconds': 10, 'maximum_control_seconds': 30,
    'maximum_bytes': 48 * 1024**2, 'maximum_seconds_before_verification': 45,
    'new_paid_calls': 0, 'new_model_inference': False, 'new_training': False,
    'repeat_evidence_used': False, 'selection_or_promotion': False,
}


def grid(events):
    return {8 + i//2: event['snapshot'] for i, event in enumerate(events) if i % 2 == 0}


def compare_side(events, cameras, labels, observations, world_points):
    times = np.array([8*1001/30000 + event['snapshot']['time_s'] for event in events])
    output = {}
    for name, camera in cameras.items():
        points = np.asarray(world_points[name]); center = points.mean(axis=0)
        scale = np.sqrt(np.mean(np.sum((points-center)**2, axis=1)))
        inverse = np.linalg.pinv(np.column_stack([(points-center)/scale, np.ones(len(points))]))
        rows = []
        for observation in observations:
            row = {'frame': observation['frame'], 'source_time_s': observation['time_s'],
                'observed_centroid_px': observation['centroid_px'], 'status': 'before_simulation_initialization',
                'centroid_error_px': None, 'mask_overlap': None}
            if observation['time_s'] + 1e-9 >= times[0]:
                index = int(np.argmin(abs(times-observation['time_s'])))
                mismatch = float(times[index]-observation['time_s'])
                if abs(mismatch) > SIDE_CONTRACT['maximum_pair_clock_error_s']:
                    raise ValueError('Source-clock pairing exceeds the original envelope')
                obj = next(obj for obj in events[index]['snapshot']['objects'] if obj['id'] == 'green-box')
                vertices = corners(obj); pixels = project_affine_points(vertices, camera['matrix_px'])
                centroid = projected_centroid(pixels)
                weights = np.column_stack([(vertices-center)/scale, np.ones(8)]) @ inverse
                with Image.open(verified(CAMERA, 'inputs/masks/' + Path(observation['mask']).name)) as image:
                    mask = np.asarray(image)
                    overlap = silhouette_overlap(mask, [400, 350, 800, 750], pixels)
                row.update(status='compared', event_index=index, simulation_source_time_s=float(times[index]),
                    pair_clock_error_s=mismatch, projected_corners_px=pixels.tolist(),
                    predicted_silhouette_centroid_px=centroid.tolist(),
                    centroid_error_px=float(np.linalg.norm(centroid-observation['centroid_px'])),
                    mask_overlap=overlap,
                    annotation_corner_bound_px_per_axis=float(labels['coordinate_uncertainty_px_per_axis'] * np.abs(weights).sum(axis=1).max()))
            rows.append(row)
        output[name] = rows
    return output


def render_side(out, original, candidate):
    source = verified(CAMERA, 'inputs/source.mp4')
    capture = cv2.VideoCapture(str(source))
    writer = cv2.VideoWriter(str(out/'side-comparison.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), 25, (1600, 470))
    if not capture.isOpened() or not writer.isOpened():
        raise RuntimeError('Source decoder or side encoder unavailable')
    views = [(f'{condition} {direction}', rows[direction])
        for direction in ('depth-positive', 'depth-negative') for condition, rows in [('original', original), ('conditional', candidate)]]
    try:
        for frame in range(37):
            ok, bgr = capture.read()
            if not ok or abs(capture.get(cv2.CAP_PROP_POS_MSEC)/1000-frame/25) > 1e-7:
                raise ValueError('Side source clock or coverage changed')
            crop = Image.fromarray(cv2.cvtColor(bgr[350:750, 400:800], cv2.COLOR_BGR2RGB))
            panel = Image.new('RGB', (1600, 470), '#F5EADC'); draw = ImageDraw.Draw(panel)
            for column, (name, rows) in enumerate(views):
                source_crop = crop.copy(); overlay = ImageDraw.Draw(source_crop); row = rows[frame]
                detail = 'No simulated state before initialization'
                if row['status'] == 'compared':
                    pixels = np.asarray(row['projected_corners_px'])-[400, 350]
                    hull = cv2.convexHull(pixels.astype(np.float32)).reshape(-1, 2).tolist()
                    overlay.line([tuple(p) for p in hull+[hull[0]]], fill='#00D7FF', width=2)
                    detail = f'{row["centroid_error_px"]:.1f}px | overlap {row["mask_overlap"]["iou"]:.3f}'
                panel.paste(source_crop, (column*400, 30))
                draw.text((column*400+5, 8), name, fill='#214F50')
                draw.text((column*400+5, 435), detail, fill='#214F50')
            draw.text((6, 453), f'Frame {frame} / 36 | {frame/25:.2f}s | Camera-dependent depth; unchanged materials. Not physical validation.', fill='#214F50')
            writer.write(cv2.cvtColor(np.asarray(panel), cv2.COLOR_RGB2BGR))
            if frame in (7, 20, 26, 29, 36):
                panel.save(out/f'side-{frame:04d}.png')
        if capture.read()[0]:
            raise ValueError('Unexpected side source suffix')
    finally:
        capture.release(); writer.release()


def run(out):
    started = time.perf_counter(); cv2.setNumThreads(1)
    baseline = load_recording(MATERIAL/'selected-coast')
    construction = read(verified(MATERIAL, 'construction-inputs.json'))
    observations, projection = construction['observations'], construction['projection']
    if digest(FRONT) != projection['source_sha256']:
        raise ValueError('Front source binding changed')
    scene = deepcopy(baseline['scene']); objects = {obj['id']: obj for obj in scene['objects']}
    obj = objects['green-box']; direction = rotation(obj['quaternion'])[:, 1]
    cameras = read(verified(CAMERA, 'camera-fits.json'))
    labels = read(verified(CAMERA, 'inputs/source-corners.json'))
    side_observations = read(verified(CAMERA, 'inputs/side-image-evidence.json'))['rows']
    source_world = read(verified(CAMERA, 'construction-world-points.json'))
    proposals = {name: conditional_axis_scale(camera['matrix_px'], direction=direction,
        center=obj['position'], scope='construction') for name, camera in cameras.items()}
    if not all(proposal['available'] for proposal in proposals.values()):
        raise ValueError('The opening corners do not admit the bounded conditional shape')
    factor = proposals['depth-positive']['factor']
    if any(abs(proposal['factor']-factor) > 1e-10 for proposal in proposals.values()):
        raise ValueError('Depth assignments disagree on the dimension proposal')
    if any(proposal['image_axis_scale_ratio'] > SIDE_CONTRACT['orthographic_axis_scale_ratio_max']
            or abs(proposal['row_cosine']) > SIDE_CONTRACT['orthographic_abs_row_cosine_max'] for proposal in proposals.values()):
        raise ValueError('One-axis change does not pass the declared camera assumption screen')
    before = obj['size'][1]; obj['size'][1] *= factor
    obj['provenance']['conditional_depth'] = {'source_labels_sha256': digest(CAMERA/'inputs/source-corners.json'),
        'original_depth': before, 'factor': factor, 'status': 'unverified_camera_assumption'}
    scene['assumptions'].append('Green depth changes under a scaled-orthographic opening-corner hypothesis; mass stays fixed so inertia changes. Not measured geometry.')
    scene = validate_scene(scene)
    candidate = out/'candidate'; candidate.mkdir()
    write_json(candidate/'scene.json', scene)
    write_json(candidate/'report.json', {'status': 'frozen_conditional_geometry', 'physical_validation': 'unverified',
        'depth_before': before, 'depth_after': obj['size'][1], 'factor': factor, 'camera_proposals': proposals})
    seal_suite(candidate)
    write_json(out/'construction-inputs.json', construction)
    write_json(out/'camera-proposals.json', proposals)
    points = {}
    center = np.asarray(obj['position'])
    for name, original in source_world.items():
        relative = np.asarray(original)-center
        changed = relative + (factor-1)*np.outer(relative@direction, direction) + center
        points[name] = changed.tolist()
        expected = np.asarray(cameras[name]['fitted_points_px'])
        np.testing.assert_allclose(project_affine_points(changed, proposals[name]['matrix_px']), expected, atol=1e-9)
    write_json(out/'conditional-world-points.json', points)
    runtime = _implementation_files()
    for name in ('orthographic_shape.py', 'affine_camera.py', 'material_fit.py', 'primitive_geometry.py', 'spatial_rigid.py'):
        shutil.copyfile(runtime[name], out/name)
    for name in ('check_ramp_side_camera.py', 'evaluate_ramp_repeat.py'):
        shutil.copyfile(Path(__file__).with_name(name), out/name)
    write_json(out/'pre-run.json', {'created_at': local_time(), 'contract': CONTRACT,
        'source_camera_manifest_sha256': digest(CAMERA/'manifest.json'),
        'material_manifest_sha256': digest(MATERIAL/'manifest.json'),
        'front_mask_manifest_sha256': digest(MASKS/'manifest.json'),
        'candidate_scene_sha256': digest(candidate/'scene.json'),
        'camera_proposals_sha256': digest(out/'camera-proposals.json'),
        'implementation': {name: digest(path) for name, path in runtime.items()},
        'test_report_sha256': digest(ROOT/'.local/orthographic-shape-tests-01.xml')})
    result = coast_candidate(scene, observations, projection, initial_frame=8, steps_per_frame=2,
        max_seconds=CONTRACT['maximum_simulation_seconds'])
    events = result.pop('events'); result.pop('states')
    recorder = Recorder(out/'candidate-coast', scene, profile='spatial-rigid')
    recorder.start(events[0]['snapshot'], seed=1729)
    try:
        for event in events[1:]:
            recorder.append(event)
    finally:
        recorder.close()
    export = export_recording(out/'candidate-coast')
    exact_states = exact_replay(out/'candidate-coast')
    baseline_states, candidate_states = grid(baseline['events']), grid(events)
    original = score_states(observations, [{'frame': f, 'snapshot': s} for f, s in baseline_states.items()], projection)
    write_json(out/'front-center-comparison.json', {'original': original, 'conditional': result})
    side = compare_side(events, proposals, labels, side_observations, points)
    baseline_side = read(verified(CAMERA, 'all-source-slots.json'))
    write_json(out/'all-side-slots.json', side)
    front_masks = read(verified(MASKS, 'mask-evidence.json'))['observations']; mask_rows = []
    for observed in front_masks:
        identifier = {'sam-1': 'wood-block', 'sam-2': 'green-box'}[observed['object_id']]
        row = {'source_id': observed['id'], 'frame': observed['frame'], 'object_id': identifier,
            'mask_status': observed['mask_status'], 'original': None, 'conditional': None}
        if observed['frame'] in baseline_states and observed['mask_status'] == 'decoded':
            with Image.open(verified(MASKS, observed['raster_path'])) as image:
                mask = np.asarray(image)
                for name, states in [('original', baseline_states), ('conditional', candidate_states)]:
                    moving = next(obj for obj in states[observed['frame']]['objects'] if obj['id'] == identifier)
                    pixels = project_affine_points(corners(moving), projection['matrix_px'])
                    row[name] = silhouette_overlap(mask, observed['box']['box_xyxy_halfopen'], pixels)
        mask_rows.append(row)
    write_json(out/'all-front-mask-slots.json', mask_rows)
    deltas = []
    for before_row, after_row in zip(original['rows'], result['rows']):
        if before_row['source_id'] != after_row['source_id']:
            raise ValueError('Front slot identity changed')
        deltas.append({'source_id': before_row['source_id'], 'object_id': before_row['object_id'],
            'frame': before_row['frame'], 'original_error_px': before_row['error_px'],
            'conditional_error_px': after_row['error_px'],
            'change_px': after_row['error_px']-before_row['error_px'] if before_row['error_px'] is not None else None})
    write_json(out/'all-front-center-deltas.json', deltas)
    execution = subprocess.run([sys.executable, '-m', 'weft', 'scene-sandbox', '--run', str(candidate),
        '--record', str(out/'candidate-controls')], input='\n'.join(CONTRACT['controls'])+'\n', text=True,
        capture_output=True, timeout=CONTRACT['maximum_control_seconds'])
    (out/'candidate-controls.log').write_text(execution.stdout+execution.stderr)
    if execution.returncode:
        raise RuntimeError('Candidate terminal controls failed; see saved log')
    control_states = exact_replay(out/'candidate-controls')
    render_side(out, baseline_side, side)
    summary = {'depth_before': before, 'depth_after': obj['size'][1], 'depth_factor': factor,
        'front': {'original_rmse_px': original['rmse_px'], 'conditional_rmse_px': result['rmse_px'],
            'slots_each': len(deltas), 'compared_each': result['compared'],
            'improved': sum(row['change_px'] is not None and row['change_px'] < -1e-9 for row in deltas),
            'worsened': sum(row['change_px'] is not None and row['change_px'] > 1e-9 for row in deltas),
            'unchanged': sum(row['change_px'] is not None and abs(row['change_px']) <= 1e-9 for row in deltas)},
        'side': {name: {'original': summarize(baseline_side[name]), 'conditional': summarize(rows)} for name, rows in side.items()},
        'front_mask_overlap': {identifier: {condition: float(np.mean([row[condition]['iou'] for row in mask_rows
            if row['object_id'] == identifier and row[condition] is not None])) for condition in ('original', 'conditional')}
            for identifier in ('wood-block', 'green-box')}}
    elapsed = time.perf_counter()-started
    if elapsed > CONTRACT['maximum_seconds_before_verification']:
        raise TimeoutError('Conditional-depth study exceeded its runtime envelope')
    if sum(path.stat().st_size for path in out.rglob('*') if path.is_file()) > CONTRACT['maximum_bytes']:
        raise ValueError('Conditional-depth study exceeded its output envelope')
    report = {'created_at': local_time(), 'status': 'complete_conditional_depth_comparison', 'contract': CONTRACT,
        'results': summary, 'elapsed_seconds': elapsed, 'new_source_decodes': 37, 'promotion': False,
        'new_paid_calls': 0, 'physical_validation': 'unverified', 'new_coast_states_exact': exact_states,
        'new_control_states_exact': control_states, 'coast_video_frames': export['video_frames'],
        'limitations': ['One source-corner shape assumption; no motion-driven parameter search or repeat-trial reuse.',
            'Fixed mass with changed depth changes cuboid inertia; density and geometry remain unmeasured.',
            'The baseline was fitted to this first trial; this comparison is development, not a fresh holdout.',
            'Passing the camera screen cannot establish anatomy, collision accuracy or physical changed-action outcomes.',
            'All side-axis alternatives and source-slot failures remain visible; no default or existing lab changes.']}
    (out/'index.html').write_text(f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>Weft · conditional box depth</title>
<style>body{{font:17px system-ui;line-height:1.5;max-width:1500px;margin:30px auto;padding:0 20px}}video{{max-width:100%}}pre{{white-space:pre-wrap}}</style>
<h1>Test a camera-derived box depth</h1><p>Only green-box depth changes: {before:.5f} → {obj['size'][1]:.5f} scene widths. This uses an explicit camera assumption, not a measured dimension. All materials, mass, initial state and controls stay fixed.</p>
<p>Front-view center RMS: {original['rmse_px']:.2f} → {result['rmse_px']:.2f}px. Every individual change is retained; no candidate is promoted.</p>
<video controls preload="metadata" src="side-comparison.mp4"></video>
<p><a href="candidate-coast/replay.html">Rotate the new 3D replay</a> · <a href="candidate-controls/replay.html">Recorded controls and reset</a> · <a href="report.json">All results</a> · <a href="all-front-center-deltas.json">Every front-view change</a> · <a href="all-side-slots.json">Every side-view slot</a></p>
<h2>Interact locally</h2><pre>.venv-lab/bin/python -m weft scene-sandbox --run {candidate.relative_to(ROOT)} --record artifacts/my-conditional-depth-session</pre>
<p>Use a new recording directory. Try <code>right 4</code>, <code>forward 4</code>, <code>brake 4</code>, <code>reset</code>, <code>quit</code>. Virtual controls are not validated physical interventions.</p></html>''')
    write_json(out/'report.json', report)
    return report


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--out', type=Path, required=True)
    out = parser.parse_args().out.resolve()
    if out.parent != ROOT/'artifacts':
        raise ValueError('Use a new top-level artifact directory')
    out.mkdir(exist_ok=False); shutil.copyfile(__file__, out/Path(__file__).name)
    try:
        report = run(out)
    except Exception as error:
        failure = {'created_at': local_time(), 'status': 'failed', 'type': type(error).__name__, 'error': str(error)}
        write_json(out/'failure.json', failure); write_json(out/'report.json', failure); seal_suite(out)
        raise
    result = deepcopy(report['results'])
    for variants in result['side'].values():
        for value in variants.values():
            value.pop('final')
    print(json.dumps({'results': result, 'seconds': report['elapsed_seconds'],
        'exact_coast_states': report['new_coast_states_exact'], 'exact_control_states': report['new_control_states_exact'],
        'verification': seal_suite(out)}))


if __name__ == '__main__':
    main()
