"""Run the frozen first-trial materials on a reserved repeat, with prefix-only initialization."""

import argparse
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
from weft.io import digest, local_time, write_json
from weft.pipeline import _implementation_files, verify_artifacts
from weft.ramp_transfer import initialize_prefix
from weft.recording import Recorder, export_recording, load_recording
from weft.spatial_rigid import SpatialRigidEnv, corners


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = ROOT/'artifacts/phys101-repeat-protocol-01'
FIT = ROOT/'artifacts/spatial-material-fit-01'
EVIDENCE = ROOT/'artifacts/meta-phys101-repeat-evidence-01'
LABELS = ROOT/'artifacts/phys101-repeat-labels-01'
CAMERA = ROOT/'artifacts/phys101-repeat-camera-check-01'
SOURCE = ROOT/'artifacts/phys101-constraints-source-01/member-03-Camera_1.mp4'


def read(path):
    return json.loads(path.read_text())


def exact_replay(path):
    saved=load_recording(path);env=SpatialRigidEnv(saved['scene'])
    try:
        for event in saved['events']:
            if event['type'] in ('start','reset'):
                env.reset(seed=event['seed'],options={'jitter':False})
            else:
                _,reward,terminated,truncated,_=env.step(event['action'])
                if (reward,terminated,truncated)!=(event['reward'],event['terminated'],event['truncated']):
                    raise ValueError('Recorded transition differs on exact replay')
            if env.snapshot()!=event['snapshot']:
                raise ValueError('Recorded full state differs on exact replay')
    finally:
        env.close()
    return len(saved['events'])


def metrics(values):
    return {'count':len(values),'rmse_px':float(np.sqrt(np.mean(np.square(values)))) if values else None,
            'p95_px':float(np.percentile(values,95)) if values else None,'maximum_px':max(values) if values else None}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True)
    out=parser.parse_args().out.resolve()
    if out.parent!=ROOT/'artifacts':raise ValueError('Use a new top-level artifacts directory')
    verified={str(path.relative_to(ROOT)):verify_artifacts(path) for path in (PROTOCOL,FIT,EVIDENCE,LABELS,CAMERA)}
    protocol=read(PROTOCOL/'protocol.json')
    if any(digest(ROOT/path)!=expected for path,expected in protocol['code'].items()):
        raise ValueError('Frozen constructor, dynamics or evidence adapter changed')
    if read(CAMERA/'report.json')['status']!='passed' or digest(SOURCE)!=protocol['source_sha256']:
        raise ValueError('The frozen same-camera/source prerequisites fail')
    freeze=read(FIT/'construction-frozen.json')
    parent_scene_path=FIT/'candidates'/freeze['selected']/'scene.json'
    if digest(parent_scene_path)!=protocol['selected']['scene_sha256']:
        raise ValueError('Selected material hypothesis changed after freeze')
    projection=deepcopy(read(FIT/'construction-inputs.json')['projection'])
    projection['source_sha256']=protocol['source_sha256']
    evidence=read(EVIDENCE/'segmentation-evidence.json')['observations']
    def source_row(row):
        return {'source_id':row['id'],'frame':row['frame'],'time_s':row['time_s'],
            'object_id':{'sam-1':'wood-block','sam-2':'green-box'}[row['object_id']],
            'center_px':(row.get('geometry') or {}).get('center_px'),'identity':deepcopy(row['identity']),
            'scope':'construction'}
    prefix=[source_row(row) for row in evidence if row['frame']<=18]
    scene,initialization=initialize_prefix(read(parent_scene_path),prefix,projection,
        actor_id='wood-block',target_id='green-box',ramp_id='ramp',support_id='table')
    baseline=deepcopy(scene)
    for obj in baseline['objects']:
        obj['friction'][0]=.25
        if obj['id']=='green-box':obj['mass']=next(o['mass'] for o in baseline['objects'] if o['id']=='wood-block')
    baseline['assumptions'].append('Paired baseline restores only original sliding coefficients and target mass; prefix initial state is identical.')
    out.mkdir(exist_ok=False);shutil.copyfile(__file__,out/Path(__file__).name)
    implementation={name:digest(path) for name,path in _implementation_files().items()}
    write_json(out/'pre-run.json',{'created_at':local_time(),'source_verification':verified,'protocol_sha256':digest(PROTOCOL/'protocol.json'),
        'implementation':implementation,'maximum_workers':2,'maximum_case_seconds':10,'paid_calls':0})
    write_json(out/'prefix-only-inputs.json',{'projection':projection,'observations':prefix})
    write_json(out/'initialization.json',initialization)
    for name,value in (('baseline',baseline),('candidate',scene)):
        folder=out/name;folder.mkdir()
        write_json(folder/'scene.json',value)
        write_json(folder/'report.json',{'created_at':local_time(),'scope':'Frozen repeat scene before later-frame grading',
            'source_sha256':protocol['source_sha256'],'source_initialization_frames':[8,18],
            'parent_material_scene_sha256':digest(parent_scene_path),'physical_validation':'unverified','training':False})
        seal_suite(folder)
    write_json(out/'prediction-inputs-frozen.json',{'created_at':local_time(),
        'baseline_scene_sha256':digest(out/'baseline/scene.json'),'candidate_scene_sha256':digest(out/'candidate/scene.json'),
        'source_frames':47,'initial_frame':8,'steps_per_frame':2,'comparison_frames':list(range(19,47)),
        'material_refit':False,'evaluation_used_by_constructor':False})
    cv2.setNumThreads(1);started=time.perf_counter()
    def simulate(item):
        name,value=item;env=SpatialRigidEnv(value);env.reset(seed=1729,options={'jitter':False})
        recorder=Recorder(out/f'{name}-coast',value,profile='spatial-rigid');recorder.start(env.snapshot(),seed=1729)
        states={8:env.snapshot()};tick=time.perf_counter()
        try:
            for step in range(1,77):
                _,reward,terminated,truncated,_=env.step(0);state=env.snapshot()
                recorder.append({'type':'step','action':0,'reward':reward,'terminated':terminated,'truncated':truncated,'snapshot':state})
                if step%2==0:states[8+step//2]=state
                if terminated or truncated:raise ValueError('Repeat simulation ended before its complete source interval')
                if time.perf_counter()-tick>10:raise TimeoutError('Repeat simulation exceeds its frozen work bound')
        finally:
            recorder.close();env.close()
        write_json(out/f'{name}-states.json',states)
        export=export_recording(out/f'{name}-coast');exact=exact_replay(out/f'{name}-coast')
        write_json(out/f'{name}-execution.json',{'frames':export['video_frames'],'exact_replay_states':exact,'source_frames':len(states)})
        return name,states
    with ThreadPoolExecutor(max_workers=2) as pool:
        simulations=dict(pool.map(simulate,[('baseline',baseline),('candidate',scene)]))
    matrix=np.asarray(projection['matrix_px'])
    def position(states,frame,identifier):
        if frame not in states:return None
        obj=next(obj for obj in states[frame]['objects'] if obj['id']==identifier)
        return (matrix@np.r_[obj['position'],1.]).tolist()
    labels=read(LABELS/'labels.json')['rows']
    all_slots,results={},{}
    for name,states in simulations.items():
        slots=[]
        for original in evidence:
            row=source_row(original);row.pop('scope')
            row['phase']='pre_initialization' if row['frame']<8 else 'prefix_initialization' if row['frame']<=18 else 'reserved_repeat_comparison'
            predicted=position(states,row['frame'],row['object_id'])
            row.update(predicted_center_px=predicted,error_px=math.dist(predicted,row['center_px']) if predicted is not None and row['center_px'] is not None else None)
            slots.append(row)
        later=[row for row in slots if row['phase']=='reserved_repeat_comparison']
        available=[row['error_px'] for row in later if row['error_px'] is not None]
        manual=[]
        for label in labels:
            predicted=position(states,label['frame'],label['object_id'])
            observation=next(row for row in slots if row['frame']==label['frame'] and row['object_id']==label['object_id'])
            manual.append(deepcopy(label)|{'predicted_center_px':predicted,
                'simulation_error_px':math.dist(predicted,label['center_px']),
                'model_error_px':math.dist(observation['center_px'],label['center_px']) if observation['center_px'] else None})
        summary=metrics(available)
        gate=(len(available)==56 and summary['p95_px']<=12 and summary['maximum_px']<=24
              and len(manual)==6 and all(row['simulation_error_px']<=12 for row in manual))
        results[name]={'later_model_center_errors':summary,'later_slots':len(later),
            'by_object':{identifier:metrics([row['error_px'] for row in later if row['object_id']==identifier and row['error_px'] is not None]) for identifier in ('wood-block','green-box')},
            'manual_source_labels':manual,'manual_within_uncertainty':sum(row['simulation_error_px']<=12 for row in manual),
            'model_within_manual_uncertainty':sum(row['model_error_px'] is not None and row['model_error_px']<=12 for row in manual),
            'frozen_center_motion_gate_passed':gate,'physical_validation':'unverified'}
        all_slots[name]=slots
    write_json(out/'all-slots.json',all_slots)
    write_json(out/'comparison-frozen.json',{'created_at':local_time(),'results':results,'thresholds_changed':False,
        'construction_refitted_after_comparison':False,'scope':'Prospective repeated-trial image-center comparison, not general physical validation'})
    commands='right 6\nforward 6\nback 6\nbrake 6\nreset\nquit\n'
    write_json(out/'terminal-plan.json',{'commands':commands.splitlines(),'scope':'Virtual controls, not real interventions'})
    command=[sys.executable,'-m','weft','scene-sandbox','--run',str(out/'candidate'),'--record',str(out/'terminal-session')]
    terminal=subprocess.run(command,input=commands,text=True,capture_output=True,cwd=ROOT,timeout=30)
    (out/'terminal.stdout').write_text(terminal.stdout);(out/'terminal.stderr').write_text(terminal.stderr)
    if terminal.returncode:raise RuntimeError('Repeat control/recording console failed; output retained')
    terminal_states=exact_replay(out/'terminal-session')
    capture=cv2.VideoCapture(str(SOURCE));writer=cv2.VideoWriter(str(out/'source-comparison.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),30000/1001,(1920,590))
    if not writer.isOpened():raise RuntimeError('Comparison video writer unavailable')
    try:
        for frame in range(47):
            ok,bgr=capture.read()
            if not ok or bgr.shape[:2]!=(1080,1920):raise ValueError('Original repeat frame unavailable')
            if not math.isclose(capture.get(cv2.CAP_PROP_POS_MSEC)/1000,frame*1001/30000,abs_tol=1e-7):raise ValueError('Repeat source clock differs')
            source_image=Image.fromarray(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)).resize((960,540))
            panel=Image.new('RGB',(1920,590),'#F5EADC');draw=ImageDraw.Draw(panel)
            for column,name in enumerate(('baseline','candidate')):
                image=source_image.copy();overlay=ImageDraw.Draw(image)
                if frame in simulations[name]:
                    for obj in simulations[name][frame]['objects']:
                        if obj['role']=='static':continue
                        points=np.column_stack([corners(obj),np.ones(8)])@matrix.T/2
                        hull=cv2.convexHull(points.astype(np.float32)).reshape(-1,2).tolist()
                        overlay.line([tuple(point) for point in hull+[hull[0]]],fill='#00D7FF',width=2)
                for label in labels:
                    if label['frame']==frame:
                        x,y=np.asarray(label['center_px'])/2;overlay.ellipse((x-6,y-6,x+6,y+6),outline='#E75832',width=2)
                panel.paste(image,(column*960,30));draw.text((column*960+10,9),f'{name.title()} | source frame {frame} | {frame*1001/30000:.3f}s',fill='#214F50')
            draw.text((10,574),'Cyan: frozen simulated geometry. Orange: source-only label ±12 pixels. Prefix 8–18 initializes motion; frames 19–46 test the frozen materials.',fill='#214F50')
            writer.write(cv2.cvtColor(np.asarray(panel),cv2.COLOR_RGB2BGR))
            if frame in (8,18,26,36,46):panel.save(out/f'frame-{frame:04d}.png')
        if capture.read()[0]:raise ValueError('Repeat source has extra frames')
    finally:
        capture.release();writer.release()
    report={'status':'frozen_repeat_evaluation_complete','created_at':local_time(),'source_verification':verified,
        'initialization':initialization,'results':results,'source_frames':47,'slots_per_condition':94,
        'exact_replay_states':154+terminal_states,'runtime_seconds':time.perf_counter()-started,
        'material_refit':False,'new_paid_calls_in_evaluation':0,'tracking_call_cost_usd':.01,
        'physical_validation':'unverified','scope':'Single repeated-trial source-center check; no 3D/anatomy/action-transfer claim',
        'reserve_status':'Now exposed once under the frozen protocol; never relabel as an untouched future holdout',
        'limitations':['Same object/camera/ramp, not broad diversity','Original model IDs remain unresolved after proximity',
            'Six manual centers are approximate nonexpert image labels','3D depth/rotation, actual materials and changed-action predictions remain unverified']}
    write_json(out/'report.json',report)
    row_html=''.join(f'<tr><td>{name}</td><td>{row["later_model_center_errors"]["rmse_px"]:.2f}</td><td>{row["later_model_center_errors"]["p95_px"]:.2f}</td><td>{row["later_model_center_errors"]["maximum_px"]:.2f}</td><td>{row["manual_within_uncertainty"]}/6</td><td>{row["frozen_center_motion_gate_passed"]}</td></tr>' for name,row in results.items())
    (out/'index.html').write_text(f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>Weft · reserved ramp trial</title><style>body{{font:17px system-ui;line-height:1.5;background:#F5EADC;color:#214F50;max-width:1200px;margin:30px auto;padding:0 20px}}a{{color:#965644}}video,img{{max-width:100%}}td,th{{text-align:left;padding:6px 12px;border-bottom:1px solid #D8C9BA}}pre{{white-space:pre-wrap;overflow-wrap:anywhere;background:#FBF7F1;padding:16px}}</style>
<h1>Does the fitted physics transfer to a second trial?</h1><p>Materials, geometry, gravity and camera were frozen before this repeat was inspected. Only opening position and speed use frames 8–18. Frames 19–46 test the frozen model; no settings or thresholds changed afterward.</p>
<video controls preload="metadata" src="source-comparison.mp4"></video>
<table><thead><tr><th>Condition</th><th>Later RMSE px</th><th>p95 px</th><th>Maximum px</th><th>Source labels</th><th>Frozen center check</th></tr></thead><tbody>{row_html}</tbody></table>
<p>The source-label check uses six points fixed before viewing predictions, each with 12-pixel uncertainty. All 94 object/frame slots remain, including 16 pre-initialization gaps. <strong>This is one image-motion test, not verified 3D physics or general-video reconstruction.</strong></p>
<p><a href="candidate-coast/replay.html">Inspect candidate in 3D</a> · <a href="baseline-coast/replay.html">Inspect baseline</a> · <a href="terminal-session/replay.html">Controls and reset</a> · <a href="report.json">Full evidence and limitations</a> · <a href="../meta-phys101-repeat-live-01/index.html">Meta tracking viewer</a></p>
<h2>Interact and record</h2><pre>.venv-lab/bin/python -m weft scene-sandbox --run {out.relative_to(ROOT)}/candidate --record artifacts/my-repeat-ramp-session</pre>
<p>Use a new folder; try <code>right 6</code>, <code>forward 6</code>, <code>brake 6</code>, <code>reset</code>, <code>quit</code>. Virtual interventions remain unverified against physical trials.</p>
<p>Source: Wu et al., Physics101 (2016). One Meta request cost $0.01; evaluation makes no further API call or training run. The repeat is now exposed and cannot be reused as an untouched holdout.</p></html>''')
    if implementation!={name:digest(path) for name,path in _implementation_files().items()}:raise ValueError('Runtime changed during evaluation')
    print(json.dumps({'results':{name:{'model':row['later_model_center_errors'],'manual_passes':row['manual_within_uncertainty'],'gate':row['frozen_center_motion_gate_passed']} for name,row in results.items()},
        'exact_states':report['exact_replay_states'],'runtime_seconds':report['runtime_seconds'],'verification':seal_suite(out)}))


if __name__=='__main__':
    main()
