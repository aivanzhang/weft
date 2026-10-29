"""Check frozen initial-shape candidates against the complete exposed first trial."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from hashlib import sha256
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

from evaluate_ramp_repeat import exact_replay
from weft.benchmark import seal_suite
from weft.io import digest, local_time, write_json
from weft.material_fit import coast_candidate, score_states
from weft.pipeline import _implementation_files
from weft.primitive_geometry import silhouette_overlap
from weft.recording import Recorder, export_recording, load_recording
from weft.spatial_rigid import corners, rotation


ROOT=Path(__file__).resolve().parents[2]
INITIAL=ROOT/'artifacts/initial-ramp-geometry-02'
MATERIAL=ROOT/'artifacts/spatial-material-fit-01'
MASKS=ROOT/'artifacts/meta-phys101-ramp-masks-01'
SOURCE=ROOT/'artifacts/phys101-ramp-source-01/member-01-Camera_1.mp4'
CONTRACT={
    'scope':'Complete exposed first-trial construction comparison; no independent accuracy claim',
    'conditions':['original','width-only','conditional-depth'],
    'initial_frame':8,'source_frames':44,'steps_per_frame':2,
    'maximum_workers':2,'maximum_case_seconds':10,'maximum_simulation_seconds':30,
    'fixed':['camera','initial state','height','orientation','green dimensions','mass','friction','gravity','solver'],
    'fitting':False,'selection':False,'repeat_evidence_used':False,'training':False,'new_paid_calls':0,
    'comparison':'All center slots, all saved mask slots, contacts and 3D yaw; retain every regression',
    'controls':['right 4','forward 4','brake 4','reset','quit'],
}


def read(path):
    return json.loads(path.read_text())


def verified(root,relative):
    path=root/relative
    if digest(path)!=read(root/'manifest.json')['files'][relative]:raise ValueError(f'Frozen input changed: {relative}')
    return path


def state_grid(events):
    return {8+i//2:event['snapshot'] for i,event in enumerate(events) if i%2==0}


def projected(obj,matrix):
    return np.column_stack([corners(obj),np.ones(8)])@matrix.T


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True)
    out=parser.parse_args().out.resolve()
    if out.parent!=ROOT/'artifacts':raise ValueError('Use a new top-level artifact directory')
    construction=read(verified(MATERIAL,'construction-inputs.json'))
    observations,projection=construction['observations'],construction['projection'];matrix=np.asarray(projection['matrix_px'])
    if digest(SOURCE)!=projection['source_sha256']:raise ValueError('Original source changed')
    baseline=load_recording(MATERIAL/'selected-coast')
    if baseline['scene']!=read(verified(INITIAL,'inputs/scene.json')):raise ValueError('Shape baseline differs from the saved material recording')
    runtime=_implementation_files()
    if digest(runtime['spatial_rigid.py'])!=read(MATERIAL/'pre-run.json')['implementation']['spatial_rigid.py']:
        raise ValueError('Saved baseline uses different dynamics')
    scenes={name:read(verified(INITIAL,f'{name}/scene.json')) for name in CONTRACT['conditions'][1:]}
    masks=read(verified(MASKS,'mask-evidence.json'))['observations'];rasters={}
    for row in masks:
        if row['mask_status']!='decoded':continue
        path=verified(MASKS,row['raster_path'])
        with Image.open(path) as image:raster=np.asarray(image).copy()
        if sha256(raster.tobytes()).hexdigest()!=row['raster_sha256']:raise ValueError('Mask raster changed')
        rasters[row['id']]=raster
    out.mkdir(exist_ok=False);shutil.copyfile(__file__,out/Path(__file__).name)
    write_json(out/'pre-run.json',{'created_at':local_time(),'contract':CONTRACT,
        'source_sha256':digest(SOURCE),'initial_manifest_sha256':digest(INITIAL/'manifest.json'),
        'material_manifest_sha256':digest(MATERIAL/'manifest.json'),'mask_manifest_sha256':digest(MASKS/'manifest.json'),
        'implementation':{name:digest(path) for name,path in runtime.items()},
        'candidate_scene_hashes':{name:digest(INITIAL/name/'scene.json') for name in scenes},
        'reused_baseline_recording':str((MATERIAL/'selected-coast').relative_to(ROOT)),
        'reused_baseline_exact_audit':'.local/material-transfer-final-verification-01.json'})
    write_json(out/'construction-inputs.json',{'observations':observations,'projection':projection,'mask_slots':masks})
    for name,scene in scenes.items():
        folder=out/name;folder.mkdir();write_json(folder/'scene.json',scene)
        write_json(folder/'report.json',{'status':'frozen_geometry_hypothesis','physical_validation':'unverified',
            'source':str((INITIAL/name/'scene.json').relative_to(ROOT)),'fit':False})
        seal_suite(folder)
    cv2.setNumThreads(1);started=time.perf_counter()
    def simulate(item):
        name,scene=item
        result=coast_candidate(scene,observations,projection,initial_frame=8,steps_per_frame=2,max_seconds=10)
        recorder=Recorder(out/f'{name}-coast',scene,profile='spatial-rigid')
        recorder.start(result['events'][0]['snapshot'],seed=1729)
        try:
            for event in result['events'][1:]:recorder.append(event)
        finally:recorder.close()
        states=state_grid(result.pop('events'));result.pop('states')
        write_json(out/f'{name}-center-comparison.json',result)
        export=export_recording(out/f'{name}-coast')
        replay=exact_replay(out/f'{name}-coast')
        return name,{'states':states,'center':result,'video_frames':export['video_frames'],'exact_replay_states':replay}
    with ThreadPoolExecutor(max_workers=2) as pool:simulations=dict(pool.map(simulate,scenes.items()))
    simulation_seconds=time.perf_counter()-started
    if simulation_seconds>30:raise TimeoutError('Geometry simulation exceeded its frozen work bound')
    states=state_grid(baseline['events'])
    simulations={'original':{'states':states,'center':score_states(observations,[{'frame':f,'snapshot':s} for f,s in states.items()],projection),
        'recording_reused':True},**simulations}
    results={};shape_rows={};center_deltas={}
    for name,simulation in simulations.items():
        rows=[]
        for source in masks:
            identifier={'sam-1':'wood-block','sam-2':'green-box'}[source['object_id']]
            row={'source_id':source['id'],'frame':source['frame'],'time_s':source['time_s'],'object_id':identifier,
                'identity':deepcopy(source['identity']),'mask_status':source['mask_status'],'mask_iou':None,
                'simulated_yaw_deg':None,'width_ratio':None,'height_ratio':None}
            if source['frame'] in simulation['states']:
                obj=next(obj for obj in simulation['states'][source['frame']]['objects'] if obj['id']==identifier)
                pixels=projected(obj,matrix);orientation=rotation(obj['quaternion'])
                row['simulated_yaw_deg']=math.degrees(math.atan2(orientation[1,0],orientation[0,0]))
                if source['id'] in rasters:
                    box=np.asarray(source['box']['box_xyxy_halfopen'])
                    overlap=silhouette_overlap(rasters[source['id']],box,pixels)
                    ratios=np.ptp(pixels,axis=0)/(box[2:]-box[:2])
                    row.update(mask_iou=overlap['iou'],width_ratio=float(ratios[0]),height_ratio=float(ratios[1]))
            rows.append(row)
        shape_rows[name]=rows
        impacts=[f for f,s in simulation['states'].items() if ['green-box','wood-block'] in s['contacts']]
        by_object={}
        for identifier in ('wood-block','green-box'):
            available=[r for r in rows if r['object_id']==identifier and r['mask_iou'] is not None]
            by_object[identifier]={'center':simulation['center']['by_object'][identifier],
                'mask_slots':len(available),'mean_mask_iou':float(np.mean([r['mask_iou'] for r in available])),
                'final':available[-1]}
        results[name]={'rmse_px':simulation['center']['rmse_px'],'slots':simulation['center']['slots'],
            'compared':simulation['center']['compared'],'by_object':by_object,
            'source_grid_impact_frames':impacts,'video_frames':simulation.get('video_frames'),
            'exact_replay_states':simulation.get('exact_replay_states'),'baseline_reused':name=='original'}
        if name!='original':
            deltas=[]
            for original,changed in zip(simulations['original']['center']['rows'],simulation['center']['rows']):
                if original['source_id']!=changed['source_id']:raise ValueError('Source slot order changed')
                deltas.append({'source_id':original['source_id'],'frame':original['frame'],'object_id':original['object_id'],
                    'original_error_px':original['error_px'],'candidate_error_px':changed['error_px'],
                    'error_change_px':changed['error_px']-original['error_px'] if original['error_px'] is not None else None})
            center_deltas[name]=deltas
            values=[r['error_change_px'] for r in deltas if r['error_change_px'] is not None]
            results[name]['paired_center_changes']={'improved':sum(v<-1e-8 for v in values),
                'worsened':sum(v>1e-8 for v in values),'unchanged':sum(abs(v)<=1e-8 for v in values),
                'maximum_regression_px':max(values)}
    write_json(out/'all-mask-slots.json',shape_rows);write_json(out/'all-center-deltas.json',center_deltas)
    for name in scenes:
        commands='\n'.join(CONTRACT['controls'])+'\n'
        execution=subprocess.run([sys.executable,'-m','weft','scene-sandbox','--run',str(out/name),
            '--record',str(out/f'{name}-controls')],input=commands,text=True,capture_output=True,timeout=30)
        (out/f'{name}-controls.log').write_text(execution.stdout+execution.stderr)
        if execution.returncode:raise RuntimeError(f'Terminal controls failed for {name}; see saved log')
        results[name]['control_replay_states']=exact_replay(out/f'{name}-controls')
    capture=cv2.VideoCapture(str(SOURCE))
    writer=cv2.VideoWriter(str(out/'source-comparison.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),30000/1001,(1920,410))
    if not writer.isOpened():raise RuntimeError('Comparison encoder unavailable')
    try:
        for frame in range(44):
            ok,bgr=capture.read()
            if not ok or bgr.shape[:2]!=(1080,1920):raise ValueError('Original source frame unavailable')
            if abs(capture.get(cv2.CAP_PROP_POS_MSEC)/1000-frame*1001/30000)>1e-7:raise ValueError('Source clock changed')
            original=Image.fromarray(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)).resize((640,360))
            panel=Image.new('RGB',(1920,410),'#F5EADC');draw=ImageDraw.Draw(panel)
            for column,(name,simulation) in enumerate(simulations.items()):
                image=original.copy();overlay=ImageDraw.Draw(image)
                if frame in simulation['states']:
                    for obj in simulation['states'][frame]['objects']:
                        if obj['role']=='static':continue
                        pixels=projected(obj,matrix)/3;hull=cv2.convexHull(pixels.astype(np.float32)).reshape(-1,2).tolist()
                        overlay.line([tuple(p) for p in hull+[hull[0]]],fill='#00D7FF',width=2)
                panel.paste(image,(column*640,30));draw.text((column*640+8,9),f'{name} | frame {frame} | {frame*1001/30000:.3f}s',fill='#214F50')
            draw.text((8,394),'Cyan: simulated cuboids. Fixed materials; construction comparison only. Source: Wu et al., Physics101 (2016).',fill='#214F50')
            writer.write(cv2.cvtColor(np.asarray(panel),cv2.COLOR_RGB2BGR))
            if frame in (8,18,26,30,34,43):panel.save(out/f'frame-{frame:04d}.png')
        if capture.read()[0]:raise ValueError('Unexpected source suffix')
    finally:capture.release();writer.release()
    report={'created_at':local_time(),'status':'geometry_motion_comparison_complete_unverified','contract':CONTRACT,
        'results':results,'simulation_and_coast_exports_seconds':simulation_seconds,'source_frames':44,
        'new_simulated_coast_states':142,'new_recorded_control_states':sum(v.get('control_replay_states',0) for v in results.values()),
        'new_paid_calls':0,'fitting':False,'promotion':False,
        'limitations':['All source observations are exposed development evidence, not an untouched holdout.',
            'The conditional depth uses assumed scale and provider volume, not an independent dimension measurement.',
            'Green depth, camera, identity through contact and real-world changed actions remain unverified.',
            'Mask overlap and centers compare to model estimates; 3D yaw and contacts are simulated, not measured.']}
    write_json(out/'report.json',report)
    summary=''.join(f'<tr><td>{name}</td><td>{value["rmse_px"]:.2f}</td><td>{value["by_object"]["wood-block"]["mean_mask_iou"]:.3f}</td><td>{value["by_object"]["green-box"]["mean_mask_iou"]:.3f}</td></tr>' for name,value in results.items())
    links=''.join(f'<li>{name}: <a href="{name}-coast/replay.html">3D coast replay</a> · <a href="{name}-controls/replay.html">Recorded controls and reset</a><pre>.venv-lab/bin/python -m weft scene-sandbox --run {out.relative_to(ROOT)}/{name} --record artifacts/my-{name}-session</pre></li>' for name in scenes)
    (out/'index.html').write_text(f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>Weft · shape and collision comparison</title>
<style>body{{font:17px system-ui;line-height:1.5;background:#F5EADC;color:#214F50;max-width:1200px;margin:30px auto;padding:0 20px}}a{{color:#965644}}video,img{{max-width:100%}}table{{border-collapse:collapse}}th,td{{text-align:left;padding:6px 14px;border-bottom:1px solid #D8C9BA}}pre{{white-space:pre-wrap;overflow-wrap:anywhere;background:#FBF7F1;padding:12px}}</style>
<h1>Do the new shapes improve the collision?</h1><p>Both candidates come from the opening frame. Camera, starting motion, mass and friction stay unchanged. The original recording is reused; neither candidate is refitted to this comparison.</p>
<p><strong>Construction evidence, not independent physical validation.</strong> Better opening outlines need not predict better motion. All source slots and regressions remain available.</p>
<video controls preload="metadata" src="source-comparison.mp4"></video>
<table><thead><tr><th>Shape</th><th>Center error, px</th><th>Wood mask overlap</th><th>Green mask overlap</th></tr></thead><tbody>{summary}</tbody></table>
<p>Lower center error and higher mask overlap mean closer agreement with saved model estimates, not measured physical truth. Depth remains conditional.</p>
<h2>Inspect, interact and record</h2><ul>{links}</ul><p>Use a new recording folder. Try <code>right 4</code>, <code>forward 4</code>, <code>brake 4</code>, <code>reset</code>, <code>quit</code>. Virtual controls have not been verified against physical interventions.</p>
<p><a href="report.json">All results</a> · <a href="all-center-deltas.json">Every paired center change</a> · <a href="all-mask-slots.json">Every mask slot</a> · <a href="../initial-ramp-geometry-02/index.html">Opening shape fit</a> · <a href="../spatial-material-fit-01/selected-coast/replay.html">Original 3D replay</a></p></html>''')
    print(json.dumps({'results':{name:{'rmse_px':r['rmse_px'],'paired_changes':r.get('paired_center_changes'),
        'mean_mask_iou':{obj:v['mean_mask_iou'] for obj,v in r['by_object'].items()},
        'exact_coast_states':r.get('exact_replay_states'),'exact_control_states':r.get('control_replay_states')} for name,r in results.items()},
        'simulation_seconds':simulation_seconds,'verification':seal_suite(out)}))


if __name__=='__main__':main()
