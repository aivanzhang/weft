"""Retain source-derived front geometry while exposing previously hidden lateral hypotheses."""

import argparse
import ast
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
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
from weft.ramp_inference import scene_to_image
from weft.recording import Recorder, export_recording, load_recording, render_snapshot
from weft.spatial_rigid import PROFILE, UNITS, SpatialRigidEnv, validate_scene


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'artifacts/source-ramp-reconstruction-01'
SOURCE = ROOT / 'artifacts/phys101-constraints-source-01'
INSPECTION = ROOT / 'artifacts/phys101-constraints-inspection-02'
EVIDENCE = ROOT / 'artifacts/meta-phys101-ramp-evidence-01'
OFFSETS = {'negative-offset': -.015, 'centered': 0., 'positive-offset': .015}
REPRESENTATIVE = 'positive-offset'
CONTRACT = {'source_frame_start': 8, 'source_frame_end': 43, 'source_frame_steps': 2,
    'lateral_offsets_scene_units': OFFSETS, 'representative': REPRESENTATIVE,
    'box_depths_scene_units': {'wood-block': .04, 'green-box': .08, 'ramp': .35, 'table': .55},
    'table_top_height': .10, 'contact': {'time_constant': .01, 'damping_ratio': 1.},
    'torsional_friction': .001, 'rolling_friction': .0001,
    'side_initial_color_roi': [530, 465, 95, 105], 'side_search_roi': [400, 350, 400, 400],
    'side_hue_tolerance': 10, 'side_minimum_saturation': 35, 'side_minimum_value': 30,
    'side_area_ratio': [.4, 2.5], 'side_maximum_centroid_jump_px': 100,
    'maximum_workers': 2, 'maximum_case_seconds': 30, 'paid_requests': 0, 'training': False,
    'scope': 'Source-linked 3D hypotheses; added depth/contact parameters are not fitted or physically validated'}
LIMITS = [
    'Front geometry and opening motion reuse the frozen 2D construction; no source was retimed or trimmed.',
    'Depth dimensions and lateral offsets are explicit alternatives, not measured 3D geometry.',
    'Pair friction remains .25 and normalized gravity is inherited; physical scale, target mass and material properties remain unknown.',
    'MuJoCo contact friction uses component-wise maxima for equal-priority shapes, not the 2D engine product rule.',
    'Contact softness, torsional/rolling friction and zero initial spin are authored hypotheses; impact response is not calibrated.',
    'Front projection ignores lateral depth; lower front-image error alone cannot validate a 3D scene.',
    'Source cameras have different frame rates; nearest-time pairing uses provider-declared alignment, not newly calibrated synchronization.',
    'Side-view color masks and silhouette angles are image estimates, not measured rigid pose or true yaw.',
    'The separate freefall camera is not calibrated to the ramp; its image gravity cannot automatically supply the ramp scale.',
    'The fresh second trial is not inspected or used for construction or grading in this study.',
    'Controls and the parking task are authored virtual forces; no real changed-action trial or learned policy is claimed.',
]


def read(path):
    return json.loads(path.read_text())


def lift_scene(planar, offset):
    table = next(o for o in planar['objects'] if o['id'] == 'table')
    plane_table_y = table['position'][1] - table['size'][1]/2
    objects = []
    for old in planar['objects']:
        angle = old['angle']
        objects.append({'id': old['id'], 'label': old['label'], 'role': old['role'], 'shape': 'box',
            'position': [old['position'][0], .3 + (offset if old['id'] == 'wood-block' else 0),
                         CONTRACT['table_top_height'] + plane_table_y - old['position'][1]],
            'size': [old['size'][0], CONTRACT['box_depths_scene_units'][old['id']], old['size'][1]],
            'quaternion': [math.cos(angle/2), 0, math.sin(angle/2), 0],
            'velocity': [old['velocity'][0], 0, -old['velocity'][1]], 'angular_velocity': [0, 0, 0],
            'mass': old['mass'], 'friction': [old['friction']**2, CONTRACT['torsional_friction'], CONTRACT['rolling_friction']],
            'color': '#B8ADA0' if old['id'] == 'table' else old['color'],
            'provenance': {'front_geometry': old['provenance'], 'depth': 'authored alternative',
                'lateral_offset': offset if old['id'] == 'wood-block' else 0, 'mass_and_materials': 'unverified inherited hypotheses'}})
    wood = next(o for o in objects if o['id'] == 'wood-block')
    scene = {'schema_version': 1, 'profile': PROFILE, 'units': deepcopy(UNITS), 'bounds': [1, .6, .6],
        'dt': planar['dt'], 'gravity': [0, 0, -planar['gravity'][1]], 'objects': objects,
        'contact': deepcopy(CONTRACT['contact']), 'controlled_id': 'wood-block',
        'target': [.82, .30, CONTRACT['table_top_height'] + wood['size'][2]/2],
        'task': {'position_tolerance': .025, 'speed_tolerance': .05, 'max_steps': 900, 'acceleration': 5},
        'assumptions': LIMITS[:]}
    return validate_scene(scene), plane_table_y


def replay(path):
    saved = load_recording(path)
    env = SpatialRigidEnv(saved['scene'])
    try:
        for event in saved['events']:
            if event['type'] in {'start', 'reset'}:
                env.reset(seed=event['seed'], options={'jitter': False})
            else:
                _, reward, terminated, truncated, _ = env.step(event['action'])
                if (reward, terminated, truncated) != (event['reward'], event['terminated'], event['truncated']):
                    raise ValueError('Spatial recording transition differs on exact replay')
            if env.snapshot() != event['snapshot']:
                raise ValueError('Spatial recording state differs on exact replay')
        return len(saved['events'])
    finally:
        env.close()


def run_case(item):
    name, scene, out = item
    existing = out / f'{name}-coast'
    if existing.exists():
        saved = load_recording(existing)
        if saved['scene'] != scene or len(saved['events']) != 71 or saved['events'][0]['type'] != 'start' or any(
                e['type'] != 'step' or e['action'] != 0 for e in saved['events'][1:]):
            raise ValueError('Reusable coast recording differs from the frozen task')
        states = [{'frame': 8+i//2, 'snapshot': event['snapshot']} for i, event in enumerate(saved['events']) if i%2 == 0]
        contacts = sorted({tuple(pair) for event in saved['events'] for pair in event['snapshot']['contacts']})
        return name, {'states': states, 'contacts': contacts, 'exact_replay_states': 71,
            'video_frames': 71, 'max_substeps': max(e['snapshot']['substeps'] for e in saved['events']),
            'step_snapshot_ms': None, 'reused_completed_recording': True,
            'replay_evidence': 'Prior completed final image follows successful exact replay in the sealed original runner'}
    env = SpatialRigidEnv(scene)
    env.reset(seed=1729, options={'jitter': False})
    recorder = Recorder(out / f'{name}-coast', scene, profile='spatial-rigid')
    recorder.start(env.snapshot(), seed=1729)
    states = [{'frame': 8, 'snapshot': env.snapshot()}]
    timings, contacts, substeps = [], set(), []
    started = time.perf_counter()
    try:
        for step in range(1, 71):
            tick = time.perf_counter()
            _, reward, terminated, truncated, _ = env.step(0)
            state = env.snapshot()
            timings.append((time.perf_counter() - tick)*1000)
            recorder.append({'type': 'step', 'action': 0, 'reward': reward, 'terminated': terminated,
                             'truncated': truncated, 'snapshot': state})
            contacts.update(tuple(pair) for pair in state['contacts'])
            substeps.append(state['substeps'])
            if step % 2 == 0:
                states.append({'frame': 8 + step//2, 'snapshot': state})
            if terminated or truncated:
                raise ValueError('Spatial hypothesis terminated before the complete source interval')
            if time.perf_counter() - started > CONTRACT['maximum_case_seconds']:
                raise TimeoutError('Spatial case exceeded its wall-clock bound')
    finally:
        recorder.close()
        env.close()
    export = export_recording(out / f'{name}-coast')
    exact = replay(out / f'{name}-coast')
    Image.fromarray(render_snapshot(states[-1]['snapshot'], scene, width=1280)).save(out / f'{name}-final.png')
    return name, {'states': states, 'contacts': sorted(contacts), 'exact_replay_states': exact,
                  'video_frames': export['video_frames'], 'max_substeps': max(substeps),
                  'step_snapshot_ms': {'p95': float(np.percentile(timings, 95)), 'max': max(timings), 'samples': len(timings)}}


def side_measurements(out):
    source = SOURCE / 'member-02-Camera_2.mp4'
    capture = cv2.VideoCapture(str(source))
    rows, first_area, hue, previous = [], None, None, None
    masks = out / 'side-masks';masks.mkdir()
    try:
        for frame in range(37):
            ok, bgr = capture.read()
            if not ok or bgr.shape[:2] != (1080, 1920):
                raise ValueError('Side source dimensions or frame count differ')
            stamp = capture.get(cv2.CAP_PROP_POS_MSEC)/1000
            if not math.isclose(stamp, frame/25, abs_tol=1e-6):
                raise ValueError('Side source clock differs')
            hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
            if hue is None:
                x, y, w, h = CONTRACT['side_initial_color_roi']
                crop = hsv[y:y+h, x:x+w]
                eligible = (crop[:, :, 1] >= CONTRACT['side_minimum_saturation']) & (crop[:, :, 2] >= CONTRACT['side_minimum_value'])
                if eligible.sum() < 100:
                    raise ValueError('Insufficient initial source color support')
                hue = float(np.median(crop[:, :, 0][eligible]))
            x, y, w, h = CONTRACT['side_search_roi'];region = hsv[y:y+h, x:x+w]
            difference = np.abs(region[:, :, 0].astype(float) - hue)
            binary = ((np.minimum(difference, 180-difference) <= CONTRACT['side_hue_tolerance'])
                & (region[:, :, 1] >= CONTRACT['side_minimum_saturation']) & (region[:, :, 2] >= CONTRACT['side_minimum_value'])).astype(np.uint8)
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((3, 3), dtype=np.uint8))
            count, labels, stats, centers = cv2.connectedComponentsWithStats(binary, connectivity=8)
            if first_area is None:
                selected = [i for i in range(1, count) if stats[i, cv2.CC_STAT_AREA] >= 1000]
            else:
                selected = [i for i in range(1, count) if first_area*.4 <= stats[i, cv2.CC_STAT_AREA] <= first_area*2.5]
            row = {'frame': frame, 'time_s': stamp, 'source_sha256': digest(source) if frame == 0 else rows[0]['source_sha256'],
                   'status': 'ambiguous' if len(selected) > 1 else 'missing', 'centroid_px': None, 'edge_angle_mod90_deg': None}
            if len(selected) == 1:
                index = selected[0]
                center = centers[index] + [x+.5, y+.5]
                if previous is not None and np.linalg.norm(center - previous) > CONTRACT['side_maximum_centroid_jump_px']:
                    row['status'] = 'motion_inconsistent'
                else:
                    mask = (labels == index).astype(np.uint8)
                    yy, xx = np.nonzero(mask)
                    rect = cv2.minAreaRect(np.column_stack([xx+x+.5, yy+y+.5]).astype(np.float32))
                    angle = (float(rect[2]) + 45) % 90 - 45
                    filename = f'side-masks/{frame:04d}.png'
                    Image.fromarray(mask*255).save(out / filename)
                    if first_area is None:
                        first_area = int(stats[index, cv2.CC_STAT_AREA])
                    row.update(status='color_component', centroid_px=center.tolist(), edge_angle_mod90_deg=angle,
                               foreground_pixels=int(mask.sum()), mask=filename, box_corners_px=cv2.boxPoints(rect).tolist())
                    previous = center
            rows.append(row)
        if capture.read()[0]:
            raise ValueError('Side source contains unreported frames')
    finally:
        capture.release()
    result = {'method': 'one initial source-color region; bounded components; no tracking model or silhouette pose claim',
        'initial_hue': hue, 'initial_foreground_area': first_area, 'contract': {k: v for k, v in CONTRACT.items() if k.startswith('side_')},
        'rows': rows, 'scope': 'Approximate image evidence, not calibrated 3D position or rigid-body yaw'}
    write_json(out / 'side-image-evidence.json', result)
    return result


def main():
    parser = argparse.ArgumentParser();parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--reuse-completed', type=Path)
    args = parser.parse_args();out = args.out.resolve()
    if out.parent != ROOT / 'artifacts':
        raise ValueError('Use a new top-level artifacts directory')
    source_verification = {str(p.relative_to(ROOT)): verify_artifacts(p) for p in (BASE, SOURCE, INSPECTION, EVIDENCE)}
    out.mkdir(exist_ok=False);shutil.copyfile(__file__, out / Path(__file__).name)
    implementation = {name: digest(path) for name, path in _implementation_files().items()}
    reused = None
    if args.reuse_completed:
        previous = args.reuse_completed.resolve()
        verified = verify_artifacts(previous)
        old = read(previous / 'pre-run.json')
        if old['contract'] != CONTRACT or old['source_verification'] != source_verification:
            raise ValueError('Reusable spatial cases have different inputs or assumptions')
        changed = {name for name in implementation if implementation[name] != old['implementation'].get(name)}
        if changed != {'spatial_rigid.py'}:
            raise ValueError('Reuse permits only the recorded quaternion-validation correction')
        def unchanged_code(path):
            module = ast.parse(path.read_text())
            return {getattr(node, 'name', f'top-{i}'): ast.dump(node, include_attributes=False)
                    for i, node in enumerate(module.body) if getattr(node, 'name', '') != 'validate_snapshot'}
        prior_code = previous / 'implementation-source/spatial_rigid.py'
        if unchanged_code(prior_code) != unchanged_code(ROOT / 'src/weft/spatial_rigid.py'):
            raise ValueError('Spatial dynamics or rendering changed; cannot reuse completed simulation')
        reused = {'source': str(previous), 'verification': verified, 'changed_function': 'validate_snapshot',
                  'scope': 'Only quaternion roundoff validation changes; full dynamics/source/rendering ASTs match', 'cases': []}
        for name in OFFSETS:
            old_recording = previous / f'{name}-coast'
            if not (old_recording / 'manifest.json').exists() or not (previous / f'{name}-final.png').exists():
                continue
            verify_artifacts(old_recording)
            shutil.copytree(old_recording, out / f'{name}-coast')
            shutil.copyfile(previous / f'{name}-final.png', out / f'{name}-final.png')
            reused['cases'].append(name)
    for name, path in _implementation_files().items():
        target = out / 'implementation-source' / name;target.parent.mkdir(parents=True, exist_ok=True);shutil.copyfile(path, target)
    write_json(out / 'pre-run.json', {'created_at': local_time(), 'contract': CONTRACT, 'source_verification': source_verification,
        'implementation': implementation, 'reused_completed': reused, 'reserved_repeat_opened': False, 'paid_calls': 0})
    started = time.perf_counter();cv2.setNumThreads(1)
    planar = read(BASE / 'friction-025-mass-1/scene.json')
    camera = read(BASE / 'construction.json')['camera']
    cases, seals = [], {}
    for name, offset in OFFSETS.items():
        scene, plane_table_y = lift_scene(planar, offset)
        path = out / name;path.mkdir();write_json(path / 'scene.json', scene)
        write_json(path / 'report.json', {'status': 'frozen_source_linked_spatial_hypothesis', 'created_at': local_time(),
            'physical_validation': 'unverified', 'physics_ready': False, 'lateral_offset': offset,
            'front_parent_scene_sha256': canonical_hash(planar), 'depth_provenance': 'authored alternatives'})
        seals[name] = seal_suite(path);cases.append((name, scene, out))
    write_json(out / 'construction-frozen.json', {'created_at': local_time(), 'scenarios': seals})
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = dict(pool.map(run_case, cases))
    side = side_measurements(out)
    evidence = read(EVIDENCE / 'segmentation-evidence.json')
    comparisons = {}
    all_slots = {}
    for name, result in results.items():
        states = {r['frame']: r['snapshot'] for r in result['states']}
        rows = []
        for row in evidence['observations']:
            identifier = {'sam-1': 'wood-block', 'sam-2': 'green-box'}[row['object_id']]
            state = states.get(row['frame'])
            obj = next((o for o in state['objects'] if o['id'] == identifier), None) if state else None
            predicted = scene_to_image(camera, [obj['position'][0], CONTRACT['table_top_height'] + plane_table_y - obj['position'][2]]).tolist() if obj else None
            actual = row['geometry']['center_px'] if row['geometry'] is not None else None
            rows.append({'frame': row['frame'], 'time_s': row['time_s'], 'object_id': identifier, 'source_id': row['id'],
                'identity': row['identity'], 'predicted_center_px': predicted, 'observed_center_px': actual,
                'error_source_px': math.dist(predicted, actual) if predicted is not None and actual is not None else None,
                'partition': 'pre_initialization' if row['frame'] < 8 else 'construction' if row['frame'] <= 18 else 'later_comparison'})
        all_slots[name] = rows
        summary = {}
        for identifier in ('wood-block', 'green-box'):
            errors = [r['error_source_px'] for r in rows if r['object_id'] == identifier and r['partition'] == 'later_comparison']
            if len(errors) != 25 or any(e is None for e in errors):
                raise ValueError('Later-frame comparison lost source slots')
            first = next(o for o in states[8]['objects'] if o['id'] == identifier)
            last = next(o for o in states[43]['objects'] if o['id'] == identifier)
            summary[identifier] = {'later_slots': 25, 'front_rmse_source_px': float(np.sqrt(np.mean(np.square(errors)))),
                'lateral_displacement_scene_units': last['position'][1] - first['position'][1],
                'final_quaternion_wxyz': last['quaternion'], 'final_world_spin': last['angular_velocity']}
        comparisons[name] = summary
    write_json(out / 'all-slots.json', all_slots);write_json(out / 'simulation-states.json', results)
    write_json(out / 'front-comparison.json', comparisons)
    side_times = [r['time_s'] for r in side['rows']]
    mappings = [{'front_frame': frame, 'front_time_s': frame*1001/30000,
        'side_frame': int(np.argmin(np.abs(np.asarray(side_times) - frame*1001/30000)))} for frame in range(44)]
    for row in mappings:
        row['side_time_s'] = side_times[row['side_frame']];row['absolute_time_error_s'] = abs(row['front_time_s'] - row['side_time_s'])
    write_json(out / 'source-time-pairs.json', {'provider_declared_alignment': True, 'fitted_time_offset': None, 'pairs': mappings,
        'scope': 'Nearest source timestamps; different cameras are not paired by frame number'})
    commands = 'status\nright 6\nforward 6\nback 6\nbrake 6\nreset\nquit\n'
    (out / 'terminal-input.txt').write_text(commands)
    with (out / 'terminal.stdout').open('x') as stdout, (out / 'terminal.stderr').open('x') as stderr:
        subprocess.run([sys.executable, '-m', 'weft', 'scene-sandbox', '--run', str(out / REPRESENTATIVE),
            '--record', str(out / 'terminal-session')], input=commands, text=True, stdout=stdout, stderr=stderr,
            cwd=ROOT, check=True, timeout=30)
    terminal_replay = replay(out / 'terminal-session')
    for name in seals:
        if verify_artifacts(out / name) != seals[name]:raise ValueError('Frozen 3D scenario changed')
    if {name: digest(path) for name, path in _implementation_files().items()} != implementation:
        raise ValueError('Runtime changed during the spatial study')
    report = {'status': 'source_linked_spatial_hypotheses_complete_unverified', 'created_at': local_time(),
        'runtime_seconds': time.perf_counter() - started, 'cases': {name: {k: v for k, v in r.items() if k != 'states'} for name, r in results.items()},
        'front_comparison': comparisons, 'object_frame_slots_per_case': 88, 'terminal_exact_replay_states': terminal_replay,
        'side_image_slots': len(side['rows']), 'side_returned_image_components': sum(r['status'] == 'color_component' for r in side['rows']),
        'source_time_pair_max_error_s': max(r['absolute_time_error_s'] for r in mappings),
        'reserved_repeat_opened': False, 'paid_calls': 0, 'training': False, 'physical_validation': 'unverified', 'limitations': LIMITS}
    write_json(out / 'report.json', report)
    sections = ''.join(f'<section><h2>{name.replace("-", " ")}</h2><img src="{name}-final.png" alt="Three views of final simulated state">'
        f'<p><a href="{name}-coast/replay.html">Play and inspect every 3D state</a> · <a href="{name}/scene.json">Scene assumptions</a></p>'
        f'<p>Later front-image center RMSE: wood {comparisons[name]["wood-block"]["front_rmse_source_px"]:.1f} px, '
        f'green {comparisons[name]["green-box"]["front_rmse_source_px"]:.1f} px. This does not grade lateral motion.</p></section>' for name in OFFSETS)
    (out / 'index.html').write_text(f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>Weft · 3D rigid-object sandbox</title><style>body{{font:17px system-ui;line-height:1.5;background:#F5EADC;color:#214F50;max-width:1160px;margin:30px auto;padding:0 20px}}a{{color:#965644}}img,video{{max-width:100%}}section{{border-top:1px solid #D8C9BA;padding:18px 0}}pre{{white-space:pre-wrap;overflow-wrap:anywhere;background:#FBF7F1;padding:16px}}</style>
<h1>One world, now with sideways motion</h1><p>The second source view shows motion and rotation across the tabletop. A front-view 2D world cannot represent all of it. These 3D hypotheses add that motion, with visible assumptions.</p>
<p><strong>Not a validated reconstruction.</strong> Depth, lateral offset, contact response and physical scale remain uncertain. Nothing here replaces the existing exercise or planar sandboxes.</p>
<h2>1. Inspect the additional source view</h2><video controls preload="metadata" src="../phys101-constraints-source-01/member-02-Camera_2.mp4"></video>
<p>Original side camera: 25 fps; original front camera: about 29.97 fps. Compare times, not frame numbers. <a href="source-time-pairs.json">Timestamp pairs</a> · <a href="side-image-evidence.json">Approximate side-image evidence</a>.</p>
<h2>2. Compare the assumptions</h2><p>Only the block's starting sideways offset differs: −.015, 0 or +.015 scene widths. All three retain the same front-view construction, masses, gravity and materials. No best-result selection or automatic promotion.</p>{sections}
<h2>3. Try the controls and record</h2><pre>.venv-lab/bin/python -m weft scene-sandbox --run artifacts/{out.name}/{REPRESENTATIVE} --record artifacts/my-3d-ramp-session</pre>
<p>Use a new recording folder. <code>right 6</code> moves along the table, <code>forward 6</code> and <code>back 6</code> push sideways, <code>brake 6</code> slows the block, <code>reset</code> restarts and <code>quit</code> exports your replay. Up/down apply bounded vertical forces; they are not automatic flight.</p>
<p>Open the recording folder's <code>current-frame.png</code> to see front, top and 3D views. Nothing moves while you type. <a href="terminal-session/replay.html">Saved controls/reset example</a>.</p>
<p><a href="report.json">Complete report</a> · <a href="all-slots.json">Every source comparison slot</a>. Wu et al., Physics101 (2016); local/private research sources. No new paid call or learned policy.</p></html>''')
    print(json.dumps({'runtime_seconds': report['runtime_seconds'], 'verification': seal_suite(out),
                      'front_comparison': comparisons, 'side_returned': report['side_returned_image_components']}))


if __name__ == '__main__':
    main()
