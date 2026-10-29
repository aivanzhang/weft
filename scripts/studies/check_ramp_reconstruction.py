"""Freeze source-prefix ramp hypotheses, then compare every later object/frame slot."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from html import escape
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import time

import cv2
import numpy as np
from PIL import Image, ImageDraw

from weft.benchmark import seal_suite
from weft.io import canonical_hash, digest, local_time, write_json
from weft.pipeline import _implementation_files, verify_artifacts
from weft.ramp_inference import affine_ramp_camera, fit_motion_prefix, gravity_hypotheses, image_to_scene, scene_to_image
from weft.recording import Recorder, export_recording
from weft.scene import SCENE_UNITS, SceneEnv, validate_scene


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / 'artifacts/phys101-ramp-source-01'
EVIDENCE = ROOT / 'artifacts/meta-phys101-ramp-evidence-01'
MASKS = ROOT / 'artifacts/meta-phys101-ramp-masks-01'
LABELS = ROOT / 'artifacts/phys101-ramp-tracking-01/inputs/labels-20.json'
OBJECTS = {'sam-1': 'wood-block', 'sam-2': 'green-box'}
REPRESENTATIVE = 'friction-025-mass-1'
CONTRACT = {
    'source_sha256': 'a93fa02bf9c53c019ca8870a9532234cce6196c62efb5210a4105afffa44d710',
    'source_frames': 44, 'initial_frame': 8, 'fit_frames': list(range(8, 19)),
    'later_frames': list(range(19, 44)), 'pre_initialization_frames': list(range(8)),
    'ramp_line_px': [[3, 244], [713, 488]], 'table_line_px': [[742, 503], [1886, 518]],
    'geometry_scope': 'Source-specific choices from frame 8 only; ramp Hough endpoints and two visible table-top segments',
    'geometry_uncertainty': 'Pixel-level line selection is approximate; no calibrated camera or inferred true collision surfaces',
    'ramp_degrees': 20, 'friction_candidates': [.2, .25, .3], 'green_relative_masses': [.25, 1, 4],
    'source_steps_per_frame': 2, 'representative': REPRESENTATIVE, 'max_workers': 2,
    'source_panels': [8, 16, 23, 27, 31, 43], 'max_wall_seconds': 60,
    'paid_calls': 0, 'training': False, 'promotion': False,
    'scope': 'Exposed development video; later frames are excluded from fitting, not an untouched final holdout',
}
LIMITS = [
    'One affine plane is assumed for the ramp, table and object centers; perspective/depth are not calibrated.',
    'Wood/green shapes are effective 2D boxes, not recovered 3D geometry. Initial heights impose support contact.',
    'Friction and normalized gravity remain a family with identical analytic ramp acceleration; metric scale is unknown.',
    'The same pair friction is assumed for ramp and table, with zero restitution; target mass is unknown.',
    'Support endpoints are selected in the initialization image. The visible gap is retained, not joined by an invisible surface.',
    'Wood initial velocity is the tangent component of an eleven-frame fit; green starts at rest.',
    'Model IDs after proximity are unresolved; trajectory comparison keeps original IDs and is not physical identity proof.',
    'Frames 0–7 have no simulation or returned tracks and remain explicit. The source video is not trimmed.',
    'Virtual controls, boundary walls and parking target are authored. No observed changed-action or training claim.',
    'Existing source-only labels are development checks and are not used for fitting or selecting a candidate.',
]


def read(path):
    return json.loads(path.read_text())


def build_scene(camera, fitted, hypothesis, mass, masks, stamp):
    angle = math.radians(CONTRACT['ramp_degrees'])
    ramp = image_to_scene(camera, CONTRACT['ramp_line_px'])
    table = image_to_scene(camera, CONTRACT['table_line_px'])
    thickness = .016
    friction = math.sqrt(hypothesis['pair_friction'])
    common = {'shape': 'box', 'mass': 1, 'velocity': [0, 0], 'angular_velocity': 0, 'friction': friction}
    objects, diagnostics = [], {}
    for model_id, line, orientation in [('sam-1', ramp, angle), ('sam-2', table, 0.)]:
        row = next(r for r in masks['observations'] if r['frame'] == 8 and r['object_id'] == model_id)
        raster = np.asarray(Image.open(MASKS / row['raster_path']))
        yy, xx = np.nonzero(raster)
        left, top, right, bottom = row['box']['box_xyxy_halfopen']
        pixels = np.column_stack([left + (xx + .5) * (right - left) / raster.shape[1],
                                  top + (yy + .5) * (bottom - top) / raster.shape[0]])
        points = image_to_scene(camera, pixels)
        tangent = np.array([math.cos(orientation), math.sin(orientation)])
        normal = np.array([-tangent[1], tangent[0]])
        position = np.asarray(fitted['position']) if model_id == 'sam-1' else image_to_scene(camera, row['box']['center_px'])
        width = float(np.ptp(points @ tangent))
        support_distance = float((line[0] - position) @ normal)
        height = 2 * (support_distance - 1e-5)
        if width < .016 or height < .016:
            raise ValueError('Observed effective box falls below scene size bounds')
        velocity = tangent * float(np.asarray(fitted['velocity']) @ tangent) if model_id == 'sam-1' else np.zeros(2)
        obj = common | {'id': OBJECTS[model_id], 'label': 'Wooden block' if model_id == 'sam-1' else 'Green target box',
            'role': 'active' if model_id == 'sam-1' else 'passive', 'angle': orientation,
            'position': position.tolist(), 'velocity': velocity.tolist(), 'size': [width, height],
            'mass': 1 if model_id == 'sam-1' else mass, 'color': '#AB7E48' if model_id == 'sam-1' else '#4A762E',
            'provenance': {'source_observation': row['id'], 'geometry': 'effective mask-width/support-height hypothesis',
                'mass': 'relative hypothesis, not measured', 'friction': 'shared assumed pair coefficient',
                'initial_source_time_s': stamp}}
        objects.append(obj)
        diagnostics[obj['id']] = {'mask_extent_scene_units': [width, float(np.ptp(points @ normal))],
            'effective_collision_size': obj['size'], 'imposed_initial_support_gap': 1e-5,
            'initial_center_source_px': scene_to_image(camera, position).tolist(),
            'discarded_normal_velocity': float(np.asarray(fitted['velocity']) @ normal) if model_id == 'sam-1' else None}
    for name, line, orientation in [('ramp', ramp, angle), ('table', table, 0.)]:
        down = np.array([-math.sin(orientation), math.cos(orientation)])
        objects.append(common | {'id': name, 'label': f'Fixed {name}', 'role': 'static', 'angle': orientation,
            'position': (line.mean(axis=0) + thickness / 2 * down).tolist(),
            'size': [float(np.linalg.norm(line[1] - line[0])), thickness], 'color': '#665F59',
            'provenance': {'geometry': 'initial-image top-line hypothesis; authored thickness'}})
    scene = validate_scene({'schema_version': 2, 'profile': 'planar-rigid-v2', 'units': deepcopy(SCENE_UNITS),
        'width': 1, 'height': camera['scene_height'], 'dt': 1001 / 60000, 'gravity': [0, hypothesis['gravity_scene_widths_per_s2']],
        'damping': 1, 'restitution': 0, 'boundary_friction': friction, 'objects': objects,
        'controlled_id': 'wood-block', 'target': [.85, .1],
        'task': {'position_tolerance': .025, 'speed_tolerance': .05, 'max_steps': 900, 'acceleration': 2},
        'relationships': [{'type': 'supported_by', 'objects': [body, support], 'status': 'assumed'}
                          for body, support in [('wood-block', 'ramp'), ('green-box', 'table')]], 'assumptions': LIMITS[:]})
    return scene, diagnostics


def simulate(case):
    name, scene, destination = case
    env = SceneEnv(scene)
    recorder = Recorder(destination, scene) if name == REPRESENTATIVE else None
    states, elapsed, contacts, capped = [], [], set(), []
    env.reset(seed=1729, options={'jitter': False})
    try:
        state = env.snapshot()
        states.append({'frame': 8, 'snapshot': state})
        if recorder:
            recorder.start(state, seed=1729)
        for step in range(1, 71):
            tick = time.perf_counter()
            _, reward, terminated, truncated, _ = env.step(0)
            state = env.snapshot()
            elapsed.append((time.perf_counter() - tick) * 1000)
            contacts.update(tuple(pair) for pair in state['contacts'])
            if any(state[key] for key in ('speed_limited', 'angular_speed_limited', 'substep_limited')):
                capped.append(step)
            if recorder:
                recorder.append({'type': 'step', 'action': 0, 'reward': reward, 'terminated': terminated,
                                 'truncated': truncated, 'snapshot': state})
            if step % 2 == 0:
                states.append({'frame': 8 + step // 2, 'snapshot': state})
            if terminated or truncated:
                raise ValueError('Unexpected early simulation termination')
    finally:
        if recorder:
            recorder.close()
        env.close()
    return name, {'states': states, 'contacts': sorted(contacts), 'capped_steps': capped,
                  'step_snapshot_ms': {'p95': float(np.percentile(elapsed, 95)), 'max': max(elapsed), 'samples': len(elapsed)}}


def summarize(rows):
    errors = [r['error_source_px'] for r in rows if r['error_source_px'] is not None]
    return {'slots': len(rows), 'compared': len(errors), 'unavailable': len(rows) - len(errors),
            'rmse_source_px': float(np.sqrt(np.mean(np.square(errors)))) if errors else None,
            'max_source_px': max(errors) if errors else None}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    out = args.out.resolve()
    if out.parent != ROOT / 'artifacts':
        raise ValueError('Use a new top-level artifacts directory')
    before = {str(path.relative_to(ROOT)): verify_artifacts(path) for path in (SOURCE, EVIDENCE, MASKS)}
    video = SOURCE / 'member-01-Camera_1.mp4'
    if digest(video) != CONTRACT['source_sha256']:
        raise ValueError('Source identity differs')
    evidence, masks = read(EVIDENCE / 'segmentation-evidence.json'), read(MASKS / 'mask-evidence.json')
    if evidence['source']['submitted_video_sha256'] != CONTRACT['source_sha256'] or masks['source'] != evidence['source']:
        raise ValueError('Evidence is not bound to the source')
    if len(evidence['observations']) != 88:
        raise ValueError('Expected every original object/frame slot')
    out.mkdir(exist_ok=False)
    shutil.copyfile(__file__, out / Path(__file__).name)
    implementation = {name: digest(path) for name, path in _implementation_files().items()}
    for name, path in _implementation_files().items():
        target = out / 'implementation-source' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
    write_json(out / 'pre-run.json', {'created_at': local_time(), 'contract': CONTRACT, 'inputs': before,
                                    'implementation': implementation, 'label_file_sha256': digest(LABELS)})
    started = time.perf_counter()
    camera = affine_ramp_camera(1920, 1080, CONTRACT['ramp_line_px'], CONTRACT['table_line_px'], 20)
    prefix = [r for r in evidence['observations'] if r['object_id'] == 'sam-1' and r['frame'] in CONTRACT['fit_frames']]
    fitted = fit_motion_prefix([r['time_s'] for r in prefix], image_to_scene(camera, [r['geometry']['center_px'] for r in prefix]))
    family = gravity_hypotheses(fitted['acceleration'], 20, CONTRACT['friction_candidates'])
    write_json(out / 'construction.json', {'camera': camera, 'motion_fit': fitted, 'gravity_family': family,
        'input_observations': prefix, 'geometry_frame': 8, 'limitations': LIMITS})
    cases, frozen = [], {}
    for hypothesis in family:
        for mass in CONTRACT['green_relative_masses']:
            name = f"friction-{round(100 * hypothesis['pair_friction']):03d}-mass-{mass:g}"
            scene, diagnostics = build_scene(camera, fitted, hypothesis, mass, masks, fitted['start_time_s'])
            path = out / name
            path.mkdir()
            write_json(path / 'scene.json', scene)
            write_json(path / 'report.json', {'status': 'frozen_unverified_hypothesis', 'physics_ready': False,
                'created_at': local_time(), 'hypothesis': hypothesis, 'green_relative_mass': mass,
                'geometry': diagnostics, 'source_sha256': CONTRACT['source_sha256'], 'contract_sha256': canonical_hash(CONTRACT)})
            frozen[name] = seal_suite(path)
            cases.append((name, scene, out / 'representative-coast'))
    write_json(out / 'construction-frozen.json', {'created_at': local_time(), 'scenarios': frozen})
    with ThreadPoolExecutor(max_workers=2) as pool:
        simulated = dict(pool.map(simulate, cases))
    summaries, all_rows = {}, {}
    labels = read(LABELS)
    if labels['source_sha256'] != CONTRACT['source_sha256']:
        raise ValueError('Label identity differs')
    for name, result in simulated.items():
        states = {row['frame']: row['snapshot'] for row in result['states']}
        rows = []
        for observed in evidence['observations']:
            frame, object_id = observed['frame'], OBJECTS[observed['object_id']]
            state = states.get(frame)
            body = next((r for r in state['objects'] if r['id'] == object_id), None) if state else None
            predicted = scene_to_image(camera, body['position']).tolist() if body else None
            actual = observed['geometry']['center_px'] if observed['geometry'] is not None else None
            error = math.dist(predicted, actual) if predicted is not None and actual is not None else None
            rows.append({'frame': frame, 'time_s': observed['time_s'], 'source_id': observed['id'], 'object_id': object_id,
                'partition': 'pre_initialization' if frame < 8 else 'construction' if frame <= 18 else 'later_comparison',
                'observed_center_px': actual, 'predicted_center_px': predicted, 'error_source_px': error,
                'identity': observed['identity']})
        manual = []
        for label in labels['annotations']:
            row = next(r for r in rows if r['frame'] == label['frame'] and r['object_id'] == label['object_id'])
            error = math.dist(row['predicted_center_px'], label['center_xy']) if row['predicted_center_px'] else None
            manual.append({'frame': label['frame'], 'object_id': label['object_id'], 'partition': row['partition'],
                           'error_source_px': error, 'within_12px_uncertainty': error is not None and error <= 12})
        summaries[name] = {'later': {obj: summarize([r for r in rows if r['object_id'] == obj and r['partition'] == 'later_comparison'])
                                   for obj in OBJECTS.values()}, 'manual': manual, 'contacts': result['contacts'],
                           'capped_steps': result['capped_steps'], 'step_snapshot_ms': result['step_snapshot_ms']}
        all_rows[name] = rows
    write_json(out / 'all-slots.json', all_rows)
    write_json(out / 'simulation-states.json', simulated)
    write_json(out / 'comparison.json', summaries)
    from build_ramp_sandbox import verify_replay
    exported = export_recording(out / 'representative-coast')
    replayed = verify_replay(out / 'representative-coast')
    commands = 'status\nright 6\ncoast 6\nbrake 6\nreset\nquit\n'
    (out / 'terminal-input.txt').write_text(commands)
    with (out / 'terminal.stdout').open('x') as stdout, (out / 'terminal.stderr').open('x') as stderr:
        subprocess.run([sys.executable, '-m', 'weft', 'scene-sandbox', '--run', str(out / REPRESENTATIVE),
            '--record', str(out / 'terminal-session')], input=commands, text=True, stdout=stdout, stderr=stderr,
            cwd=ROOT, check=True, timeout=20)
    terminal_states = verify_replay(out / 'terminal-session')
    capture = cv2.VideoCapture(str(video))
    panels = []
    states = {row['frame']: row['snapshot'] for row in simulated[REPRESENTATIVE]['states']}
    try:
        for frame in range(44):
            ok, bgr = capture.read()
            if not ok:
                raise ValueError('Original source ended early')
            if frame not in CONTRACT['source_panels']:
                continue
            image = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(image)
            for line in (CONTRACT['ramp_line_px'], CONTRACT['table_line_px']):
                draw.line([tuple(p) for p in line], fill='#FFD050', width=4)
            for body in states[frame]['objects']:
                if body['id'] not in OBJECTS.values():
                    continue
                w, h = np.asarray(body['size']) / 2
                a = body['angle']
                rotation = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
                corners = np.array([[-w, -h], [w, -h], [w, h], [-w, h]]) @ rotation.T + body['position']
                pixels = scene_to_image(camera, corners)
                draw.line([tuple(p) for p in np.vstack([pixels, pixels[0]])], fill='#FF4949', width=4)
            for row in all_rows[REPRESENTATIVE]:
                if row['frame'] != frame or row['observed_center_px'] is None:
                    continue
                x, y = row['observed_center_px']
                draw.ellipse((x-7, y-7, x+7, y+7), outline='#00D7FF', width=3)
            panel = Image.new('RGB', (960, 600), '#F5EADC')
            panel.paste(image.resize((960, 540)), (0, 40))
            caption = ImageDraw.Draw(panel)
            caption.text((12, 8), f'Frame {frame} | red: simulated boxes; cyan: saved tracked centers; yellow: assumed supports', fill='#214F50')
            caption.text((12, 582), 'Fixed representative: friction .25 / relative mass 1. Source: Wu et al., Physics101 (2016). Unverified physics.', fill='#214F50')
            filename = f'frame-{frame:04d}.png'
            panel.save(out / filename)
            panels.append(filename)
        if capture.read()[0]:
            raise ValueError('Original source contains unreported frames')
    finally:
        capture.release()
    if {name: digest(path) for name, path in _implementation_files().items()} != implementation:
        raise ValueError('Implementation changed during the study')
    if digest(video) != CONTRACT['source_sha256'] or digest(LABELS) != read(out / 'pre-run.json')['label_file_sha256']:
        raise ValueError('Source or labels changed')
    for name in frozen:
        if verify_artifacts(out / name) != frozen[name]:
            raise ValueError('Frozen scenario changed')
    duration = time.perf_counter() - started
    if duration > CONTRACT['max_wall_seconds']:
        raise ValueError('Study exceeded its time bound')
    report = {'status': 'development_comparison_complete_not_promoted', 'created_at': local_time(),
        'source_frames': 44, 'objects': 2, 'hypotheses': len(cases), 'slots_per_hypothesis': 88,
        'later_slots_per_hypothesis': 50, 'pre_initialization_unavailable_slots': 16,
        'all_candidates_frozen_before_grading': True, 'physics_ready': False, 'training': False, 'paid_calls': 0,
        'runtime_seconds': duration, 'results': summaries, 'representative_coast_states': replayed,
        'representative_video_frames': exported['video_frames'], 'terminal_exact_replay_states': terminal_states,
        'limitations': LIMITS, 'performance_scope': 'Local simulator step/snapshot only; no browser or server measurement'}
    write_json(out / 'report.json', report)
    rows_html = ''.join(f'<tr><td>{escape(name)}</td><td>{value["later"]["wood-block"]["rmse_source_px"]:.1f}</td>'
        f'<td>{value["later"]["green-box"]["rmse_source_px"]:.1f}</td><td><a href="{escape(name)}/scene.json">Scene</a></td></tr>'
        for name, value in summaries.items())
    figures = ''.join(f'<figure><img src="{filename}" alt="Source and simulated boxes at frame {filename}"></figure>' for filename in panels)
    (out / 'index.html').write_text(f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>Weft source-derived ramp comparison</title><style>body{{font:17px system-ui;line-height:1.5;background:#F5EADC;color:#214F50;max-width:1060px;margin:32px auto;padding:0 20px}}a{{color:#965644}}img,video{{max-width:100%}}td,th{{padding:6px 14px;text-align:left}}figure{{margin:16px 0}}pre{{white-space:pre-wrap;overflow-wrap:anywhere}}</style>
<h1>Does the simulation follow the video?</h1><p>Eleven opening tracked positions set the motion. All nine friction/mass hypotheses are frozen before comparing the remaining 25 frames. None is promoted as accurate.</p>
<p>These are effective 2D boxes with assumed contacts, not reconstructed 3D objects. The first eight source frames have no simulation. After the objects approach, model identity is unresolved.</p>
<h2>1. Watch the complete source</h2><video controls preload="metadata" src="../phys101-ramp-source-01/member-01-Camera_1.mp4"></video>
<h2>2. Inspect the same source frames</h2><p>The red boxes are the fixed middle hypothesis, not a best-result selection. The camera mapping is approximate.</p>{figures}
<h2>3. Compare every hypothesis</h2><p>Later-frame center error in original source pixels; lower is closer. Both objects retain all 25 comparison slots. Tracked centers are model estimates, not physical truth.</p>
<table><thead><tr><th>Friction / target mass</th><th>Wood RMSE</th><th>Green RMSE</th><th>Inputs</th></tr></thead><tbody>{rows_html}</tbody></table>
<p><a href="construction.json">Construction and uncertainty</a> · <a href="all-slots.json">All source slots</a> · <a href="comparison.json">Contact and label checks</a> · <a href="report.json">Run report</a></p>
<h2>4. Replay or interact</h2><p><a href="representative-coast/replay.html">Recorded coast</a> · <a href="terminal-session/replay.html">Recorded controls and reset</a></p>
<pre>.venv-lab/bin/python -m weft scene-sandbox --run artifacts/{escape(out.name)}/{REPRESENTATIVE} --record artifacts/my-source-ramp-session</pre>
<p>Use a new recording folder. Type <code>right 6</code>, <code>coast 6</code>, <code>brake 6</code>, <code>reset</code> or <code>quit</code>. Each command advances only the requested steps. Open <code>current-frame.png</code> for the current view; quitting exports the video and step-by-step replay.</p></html>''')
    print(json.dumps({'report': str(out / 'report.json'), 'seal': seal_suite(out), 'runtime_seconds': duration}))


if __name__ == '__main__':
    main()
