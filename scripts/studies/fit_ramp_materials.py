"""Fit explicit contact alternatives on the exposed first trial, never the reserved repeat."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import itertools
import json
import math
from pathlib import Path
import shutil
import time

import cv2
import numpy as np
from PIL import Image, ImageDraw

from weft.benchmark import seal_suite
from weft.io import canonical_hash, digest, local_time, write_json
from weft.material_fit import coast_candidate, score_states, surface_hypothesis, validate_inputs
from weft.pipeline import _implementation_files, verify_artifacts
from weft.ramp_inference import scene_to_image
from weft.recording import Recorder, export_recording, load_recording
from weft.spatial_rigid import corners


ROOT = Path(__file__).resolve().parents[2]
PARENT = ROOT / 'artifacts/spatial-ramp-sandbox-02'
CONSTRUCTION = ROOT / 'artifacts/source-ramp-reconstruction-01'
EVIDENCE = ROOT / 'artifacts/meta-phys101-ramp-evidence-01'
SOURCE = ROOT / 'artifacts/phys101-ramp-source-01/member-01-Camera_1.mp4'
CONTRACT = {
    'scope': 'All previously exposed first-trial observations are construction evidence; no final-holdout claim',
    'initial_frame': 8, 'last_frame': 43, 'steps_per_frame': 2,
    'coarse_actor_friction': [0., .125, .25], 'coarse_target_friction': [.05, .2, .5],
    'coarse_mass_ratios': [.1, .3, 1., 3.],
    'bounds_actor_friction': [0., .25], 'bounds_target_friction': [0., .5], 'bounds_log10_mass_ratio': [-1., math.log10(4)],
    'refinement_rounds': 5, 'initial_coordinate_steps': [.0625, .1, .3],
    'maximum_candidates': 80, 'maximum_workers': 2, 'maximum_wall_seconds': 90, 'maximum_case_seconds': 10,
    'near_best_rmse_allowance_px': 3., 'seed': 1729, 'reserved_repeat_opened': False,
    'fixed': ['geometry', 'projection', 'gravity', 'ramp coefficient .25', 'initial state',
              'lateral offset .015', 'contact softness', 'torsional/rolling friction', 'source clocks', 'original IDs'],
    'fit_objective': 'Equal-weight squared source-image center errors for all available object/frame slots from frame 8',
    'new_paid_calls': 0, 'new_model_inference': False, 'training': False,
}


def read(path):
    return json.loads(path.read_text())


def key(parameters):
    return tuple(round(float(value), 10) for value in parameters)


def material_values(parameters):
    return {'actor_table_friction': parameters[0], 'target_table_friction': parameters[1],
            'target_actor_mass_ratio': 10**parameters[2]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    out = parser.parse_args().out.resolve()
    if out.parent != ROOT / 'artifacts':
        raise ValueError('Use a new top-level artifacts directory')
    verification = {str(path.relative_to(ROOT)): verify_artifacts(path) for path in (PARENT, CONSTRUCTION, EVIDENCE)}
    parent = load_recording(PARENT / 'positive-offset-coast')
    scene = parent['scene']
    camera = read(CONSTRUCTION / 'construction.json')['camera']
    planar = read(CONSTRUCTION / 'friction-025-mass-1/scene.json')
    table = next(obj for obj in planar['objects'] if obj['id'] == 'table')
    table_y = table['position'][1]-table['size'][1]/2
    def project(point):
        return scene_to_image(camera, [point[0], table_y+.10-point[2]])
    origin = project([0, 0, 0])
    matrix = np.column_stack([project(np.eye(3)[axis])-origin for axis in range(3)]+[origin])
    source_hash = digest(SOURCE)
    projection = {'scope': 'construction', 'source_sha256': source_hash, 'image_size': [1920, 1080],
                  'matrix_px': matrix.tolist(), 'calibration': 'unverified_affine_hypothesis'}
    observations = []
    for row in read(EVIDENCE / 'segmentation-evidence.json')['observations']:
        observations.append({'source_id': row['id'], 'frame': row['frame'], 'time_s': row['time_s'],
            'object_id': {'sam-1': 'wood-block', 'sam-2': 'green-box'}[row['object_id']],
            'center_px': (row.get('geometry') or {}).get('center_px'), 'identity': deepcopy(row['identity']),
            'scope': 'construction'})
    validate_inputs(scene, observations, projection, 8, 2)
    out.mkdir(exist_ok=False)
    shutil.copyfile(__file__, out / Path(__file__).name)
    implementation = {name: digest(path) for name, path in _implementation_files().items()}
    original_implementation = read(PARENT / 'pre-run.json')['implementation']
    changed = [name for name in implementation if implementation[name] != original_implementation.get(name)]
    for name in changed:
        target = out / 'implementation-source' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(_implementation_files()[name], target)
    write_json(out / 'construction-inputs.json', {'observations': observations, 'projection': projection, 'scene': scene})
    write_json(out / 'pre-run.json', {'created_at': local_time(), 'contract': CONTRACT,
        'source_verification': verification, 'source_sha256': source_hash, 'implementation': implementation,
        'changed_source_copies': changed, 'unchanged_source_reference': str(PARENT.relative_to(ROOT))})
    (out / 'candidates').mkdir()
    baseline_states = [{'frame': 8+index//2, 'snapshot': event['snapshot']}
        for index, event in enumerate(parent['events']) if index%2 == 0]
    baseline = score_states(observations, baseline_states, projection)
    baseline.update(name='baseline', parameters=[.25, .25, 0.], material_values=material_values([.25, .25, 0.]),
        reused_recording=str((PARENT/'positive-offset-coast').relative_to(ROOT)), scope='construction_fit_only')
    write_json(out / 'baseline.json', baseline)
    cache = {key([.25, .25, 0.]): baseline}
    started = time.perf_counter()
    cv2.setNumThreads(1)
    def evaluate(item):
        number, parameters = item
        name = f'case-{number:03d}';folder = out / 'candidates' / name
        folder.mkdir()
        values = material_values(parameters)
        candidate = surface_hypothesis(scene, actor_id='wood-block', target_id='green-box', support_id='table',
            actor_friction=values['actor_table_friction'], target_friction=values['target_table_friction'],
            mass_ratio=values['target_actor_mass_ratio'])
        write_json(folder / 'scene.json', candidate)
        write_json(folder / 'inputs.json', {'parameters': list(parameters), 'material_values': values,
            'construction_sha256': digest(out / 'construction-inputs.json'), 'scope': 'construction_only'})
        try:
            result = coast_candidate(candidate, observations, projection, initial_frame=8, steps_per_frame=2,
                                     max_seconds=CONTRACT['maximum_case_seconds'])
            recorder = Recorder(folder / 'recording', candidate, profile='spatial-rigid')
            recorder.start(result['events'][0]['snapshot'], seed=1729)
            try:
                for event in result['events'][1:]:
                    recorder.append(event)
            finally:
                recorder.close()
            result.pop('events');result.pop('states')
            result.update(status='complete', name=name, parameters=list(parameters), material_values=values)
        except (ValueError, RuntimeError, TimeoutError) as error:
            result = {'status': 'rejected', 'name': name, 'parameters': list(parameters), 'material_values': values,
                      'error_type': type(error).__name__, 'error': str(error), 'rmse_px': None}
        write_json(folder / 'report.json', result)
        result['manifest'] = seal_suite(folder)
        return key(parameters), result
    with ThreadPoolExecutor(max_workers=2) as pool:
        def batch(values, phase):
            unique = sorted({key(parameters) for parameters in values}-cache.keys())
            if len(cache)+len(unique)>CONTRACT['maximum_candidates'] or time.perf_counter()-started>CONTRACT['maximum_wall_seconds']:
                raise RuntimeError('Frozen material-search work envelope exceeded')
            offset = len(cache)-1
            for parameters, result in pool.map(evaluate, [(offset+i, p) for i,p in enumerate(unique)]):
                cache[parameters] = result
            write_json(out / f'{phase}.json', {'scope': 'construction_only', 'new_candidates': len(unique),
                'cumulative_candidates': len(cache), 'results': [{k:r.get(k) for k in ('name','status','parameters','material_values','rmse_px')} for r in cache.values()]})
            print(json.dumps({'phase': phase, 'new_candidates': len(unique), 'total': len(cache)}), flush=True)
        batch(itertools.product(CONTRACT['coarse_actor_friction'], CONTRACT['coarse_target_friction'],
              [math.log10(m) for m in CONTRACT['coarse_mass_ratios']]), 'coarse')
        def best():
            return min((row for row in cache.values() if row['rmse_px'] is not None), key=lambda row:(row['rmse_px'],row['parameters']))
        bounds = [CONTRACT['bounds_actor_friction'], CONTRACT['bounds_target_friction'], CONTRACT['bounds_log10_mass_ratio']]
        for iteration in range(CONTRACT['refinement_rounds']):
            current = best()['parameters'];neighbors = []
            for axis, delta in enumerate(CONTRACT['initial_coordinate_steps']):
                for direction in (-1, 1):
                    point = current[:];point[axis] = min(bounds[axis][1], max(bounds[axis][0], point[axis]+direction*delta/(2**iteration)))
                    neighbors.append(point)
            batch(neighbors, f'refine-{iteration+1}')
        selected = deepcopy(best())
        write_json(out / 'construction-frozen.json', {'created_at': local_time(), 'selected': selected['name'],
            'parameters': selected['parameters'], 'material_values': selected['material_values'],
            'scene_sha256': digest(out/'candidates'/selected['name']/'scene.json') if selected['name']!='baseline' else digest(PARENT/'positive-offset/scene.json'),
            'candidate_report_sha256': digest(out/'candidates'/selected['name']/'report.json') if selected['name']!='baseline' else digest(out/'baseline.json'),
            'scope': 'Construction-selected candidate, not a validated environment', 'reserved_repeat_opened': False})
        ablations = []
        for axis in range(3):
            point = [.25, .25, 0.];point[axis] = selected['parameters'][axis];ablations.append(point)
        batch(ablations, 'one-change-ablations')
    if selected['name']=='baseline':
        raise RuntimeError('No improved candidate; keep the unchanged baseline and search evidence')
    selected_folder = out / 'candidates' / selected['name']
    selected_recording = load_recording(selected_folder/'recording')
    for name, original in (('baseline-coast',PARENT/'positive-offset-coast'),('selected-coast',selected_folder/'recording')):
        target = out/name;target.mkdir()
        for filename in ('scene.json','events.jsonl','recording.json'):
            shutil.copyfile(original/filename,target/filename)
        export_recording(target)
    selected_states = {8+i//2:event['snapshot'] for i,event in enumerate(selected_recording['events']) if i%2==0}
    baseline_by_frame = {row['frame']:row['snapshot'] for row in baseline_states}
    capture = cv2.VideoCapture(str(SOURCE))
    writer = cv2.VideoWriter(str(out/'source-comparison.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),30000/1001,(1920,590))
    if not writer.isOpened():
        raise RuntimeError('Comparison video writer unavailable')
    selected_frames = [8,18,26,30,34,43]
    try:
        for frame in range(44):
            ok,bgr = capture.read()
            if not ok or bgr.shape[:2]!=(1080,1920):raise ValueError('Original source frame unavailable')
            if not math.isclose(capture.get(cv2.CAP_PROP_POS_MSEC)/1000, frame*1001/30000,abs_tol=1e-7):
                raise ValueError('Source clock changed')
            source_image = Image.fromarray(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)).resize((960,540))
            panel = Image.new('RGB',(1920,590),'#F5EADC');draw = ImageDraw.Draw(panel)
            for column,(label,states) in enumerate((('Original materials',baseline_by_frame),('Construction-fitted materials',selected_states))):
                image = source_image.copy();overlay = ImageDraw.Draw(image)
                if frame in states:
                    for obj in states[frame]['objects']:
                        if obj['role']=='static':continue
                        points = np.column_stack([corners(obj),np.ones(8)])@matrix.T/2
                        hull = cv2.convexHull(points.astype(np.float32)).reshape(-1,2).tolist()
                        overlay.line([tuple(p) for p in hull+[hull[0]]],fill='#00D7FF',width=2)
                        center = (matrix@np.r_[obj['position'],1])/2
                        overlay.ellipse((center[0]-3,center[1]-3,center[0]+3,center[1]+3),fill='#00D7FF')
                panel.paste(image,(column*960,30))
                draw.text((column*960+10,9),f'{label} | source frame {frame} | {frame*1001/30000:.3f}s',fill='#214F50')
            draw.text((10,574),'Cyan: simulated geometry. All first-trial frames are construction evidence. Not independently validated. Wu et al., Physics101 (2016).',fill='#214F50')
            writer.write(cv2.cvtColor(np.asarray(panel),cv2.COLOR_RGB2BGR))
            if frame in selected_frames:panel.save(out/f'frame-{frame:04d}.png')
        if capture.read()[0]:raise ValueError('Source has unreported extra frames')
    finally:
        capture.release();writer.release()
    comparable = sorted((row for row in cache.values() if row['rmse_px'] is not None), key=lambda r:r['rmse_px'])
    near = [row for row in comparable if row['rmse_px']<=selected['rmse_px']+CONTRACT['near_best_rmse_allowance_px']]
    report = {'status':'material_hypothesis_fit_complete_unverified','created_at':local_time(),
        'scope':CONTRACT['scope'],'baseline_rmse_px':baseline['rmse_px'],'selected':selected,
        'evaluated_candidates':len(cache)-1,'reused_baseline':1,'rejected_candidates':sum(row['rmse_px'] is None for row in cache.values()),
        'slots_per_candidate':len(observations),'selected_compared_slots':selected['compared'],
        'ablation_results':[cache[key(p)] for p in ablations],
        'near_best_scope':'Within 3 construction-RMSE pixels of the pre-ablation selection, not a confidence interval',
        'near_best':[{'name':row['name'],'material_values':row['material_values'],'rmse_px':row['rmse_px']} for row in near],
        'search_and_exports_seconds':time.perf_counter()-started,'reserved_repeat_opened':False,
        'new_paid_calls':0,'new_model_inference':False,'training':False,'physical_validation':'unverified',
        'limitations':['The first trial is now entirely construction data; old labels are not independent evaluation.',
            'Source model identity remains unresolved after proximity; original IDs are retained.',
            'Camera, geometry, gravity, depth offset and contact response remain hypotheses.',
            'Friction and mass estimates depend on those fixed assumptions; no unique physical identification claim.',
            'Only source-view centers are fitted; side-view rotation and changed-action predictions are unvalidated.']}
    write_json(out/'report.json',report)
    materials=selected['material_values']
    rows=''.join(f'<tr><td>{row["name"]}</td><td>{row["material_values"]["actor_table_friction"]:.4f}</td><td>{row["material_values"]["target_table_friction"]:.4f}</td><td>{row["material_values"]["target_actor_mass_ratio"]:.4f}</td><td>{row["rmse_px"]:.2f}</td></tr>' for row in comparable)
    (out/'index.html').write_text(f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>Weft · contact-material construction fit</title><style>body{{font:17px system-ui;line-height:1.5;background:#F5EADC;color:#214F50;max-width:1200px;margin:30px auto;padding:0 20px}}a{{color:#965644}}video,img{{max-width:100%}}table{{border-collapse:collapse}}td,th{{text-align:left;padding:6px 12px;border-bottom:1px solid #D8C9BA}}pre{{white-space:pre-wrap;overflow-wrap:anywhere;background:#FBF7F1;padding:16px}}</style>
<h1>Separate materials change the collision</h1><p>The original model assumes equal sliding friction. This construction fit changes only the two table-contact coefficients and target mass. Geometry, gravity, initial motion, camera and ramp friction stay fixed.</p>
<p><strong>Fitted to the exposed first trial—not an independent accuracy result.</strong> The reserved repeat remains unopened. Centers come from saved Meta tracks; unresolved identities and missing frames are retained.</p>
<p>Construction center error: {baseline['rmse_px']:.2f} → {selected['rmse_px']:.2f} pixels RMSE. Selected wood/table friction {materials['actor_table_friction']:.4f}, target/table friction {materials['target_table_friction']:.4f}, target/wood mass ratio {materials['target_actor_mass_ratio']:.4f}. These are conditional hypotheses, not measured properties.</p>
<video controls preload="metadata" src="source-comparison.mp4"></video>
<p><a href="baseline-coast/replay.html">Inspect original physics</a> · <a href="selected-coast/replay.html">Inspect fitted physics in 3D</a> · <a href="report.json">All results and limitations</a></p>
<h2>Interact and record</h2><pre>.venv-lab/bin/python -m weft scene-sandbox --run {selected_folder.relative_to(ROOT)} --record artifacts/my-fitted-ramp-session</pre>
<p>Try <code>right 6</code>, <code>forward 6</code>, <code>brake 6</code>, <code>reset</code> and <code>quit</code>. Use a new recording folder. These virtual interventions are not verified physical predictions.</p>
<h2>All completed construction candidates</h2><p>Lower error means a closer fit to this one source. Nearby fits do not prove unique friction or mass.</p>
<table><thead><tr><th>Candidate</th><th>Wood/table</th><th>Target/table</th><th>Mass ratio</th><th>Pixel RMSE</th></tr></thead><tbody>{rows}</tbody></table>
<p>Public source: Wu et al., Physics101 (2016). No new model inference, paid request or training. Every attempted candidate retains its scene, complete slots, result and recording.</p></html>''')
    if implementation!={name:digest(path) for name,path in _implementation_files().items()}:
        raise ValueError('Runtime changed during material fitting')
    print(json.dumps({'baseline_rmse_px':baseline['rmse_px'],'selected_rmse_px':selected['rmse_px'],
        'selected':selected['name'],'material_values':materials,'candidates':len(cache)-1,'near_best':len(near),
        'runtime_seconds':report['search_and_exports_seconds'],'verification':seal_suite(out)}))


if __name__ == '__main__':
    main()
