"""Inspect bounded clock hypotheses on a fixed corner cohort without retiming source."""

import argparse
import json
import math
from pathlib import Path
import shutil
import time
import tomllib

import cv2
import numpy as np

from weft.benchmark import seal_suite
from weft.io import digest, local_time, write_json


ROOT=Path(__file__).resolve().parents[2]
CHECK=ROOT/'artifacts/freemocap-board-check-01'
POSES=ROOT/'artifacts/freemocap-board-diagnostics-01'
SOURCE=ROOT/'artifacts/freemocap-test-source-01'
OFFSETS=[-3,-2,-1,-.5,0,.5,1,2,3]


def read(path):return json.loads(path.read_text())


def verified(root,name):
    path=root/name
    if digest(path)!=read(root/'manifest.json')['files'][name]:raise ValueError('Frozen diagnostic input changed')
    return path


def stats(values):
    return {'count':len(values),'rmse_px':float(np.sqrt(np.mean(np.square(values)))) if values else None,
        'p95_px':float(np.percentile(values,95)) if values else None,'maximum_px':max(values) if values else None}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True)
    out=parser.parse_args().out.resolve()
    if out.parent!=ROOT/'artifacts':raise ValueError('Use a new top-level artifacts directory')
    observed=read(verified(CHECK,'view-2-detections.json'))
    poses=read(verified(POSES,'report.json'))['poses'];original=read(verified(CHECK,'all-prefix-comparisons.json'))
    cameras=tomllib.loads(verified(SOURCE,'member-04-freemocap_test_data_camera_calibration.toml').read_text())
    camera=cameras['cam_1'];K=np.asarray(camera['matrix']);dist=np.asarray(camera['distortions'])
    rotation=np.asarray(camera['rotation']);translation=np.asarray(camera['translation'])*.001
    model=cv2.aruco.CharucoBoard((7,5),.058,.0464,cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)).getChessboardCorners().astype(float)
    lookup={row['frame']:{p['id']:p['pixel'] for p in row['corners']} for row in observed}
    cohort=[]
    for pose in poses:
        frame=pose['frame'];solution=pose['planar_solutions'][0]
        world=model@cv2.Rodrigues(np.asarray(solution['rotation']))[0].T+np.asarray(solution['translation_m'])
        projected=cv2.projectPoints(world,rotation,translation,K,dist)[0].reshape(-1,2)
        source=next(row for row in original if row['frame']==frame)
        for identifier in source['source_ids']:
            cohort.append({'source_frame':frame,'source_time_s':frame/6,'corner_id':identifier,'predicted_cam2_px':projected[identifier].tolist(),
                'baseline_error_px':float(np.linalg.norm(projected[identifier]-lookup[frame][identifier]))})
    if len(cohort)!=49:raise ValueError('Fixed original comparison cohort changed')
    out.mkdir(exist_ok=False);shutil.copyfile(__file__,out/Path(__file__).name)
    write_json(out/'pre-run.json',{'created_at':local_time(),'frame_offsets':OFFSETS,'fps':6,'fixed_cohort':cohort,
        'positive_offset':'Compare Cam1 frame f with Cam2 frame f+offset; not a verified physical clock correction.',
        'source_manifests':{p.name:digest(p/'manifest.json') for p in (CHECK,POSES,SOURCE)},
        'interpolation':'For half-frame offsets only, linearly interpolate a corner visible at both adjacent observed frames; never fill missing endpoints.',
        'eligibility':'All original 49 corner slots must be available; paired subset summaries never choose an offset.',
        'maximum_seconds':3,'retiming_applied':False,'new_detection_or_pose_fit':False,'reserved_camera_opened':False,'paid_calls':0})
    started=time.perf_counter();results=[];all_slots=[]
    for offset in OFFSETS:
        rows=[];paired_old=[];paired_new=[]
        for source in cohort:
            target=source['source_frame']+offset;lo=math.floor(target);hi=math.ceil(target);identifier=source['corner_id']
            first=lookup.get(lo,{}).get(identifier);last=lookup.get(hi,{}).get(identifier)
            value=None if first is None or last is None else (np.asarray(first)*(1-(target-lo))+np.asarray(last)*(target-lo)).tolist()
            error=None if value is None else math.dist(source['predicted_cam2_px'],value)
            row={**source,'frame_offset':offset,'target_frame':target,'target_time_s':target/6,'endpoint_frames':[lo,hi],
                'observed_or_interpolated_cam2_px':value,'available':value is not None,'error_px':error}
            rows.append(row)
            if error is not None:paired_old.append(source['baseline_error_px']);paired_new.append(error)
        results.append({'frame_offset':offset,'time_offset_s':offset/6,'slots':49,'available':len(paired_new),
            'complete_original_cohort':len(paired_new)==49,'paired_baseline':stats(paired_old),'candidate':stats(paired_new),
            'paired_regressions':sum(new>old+1e-8 for old,new in zip(paired_old,paired_new))})
        all_slots.extend(rows)
    if time.perf_counter()-started>3:raise TimeoutError('Clock diagnostic exceeded its work envelope')
    write_json(out/'all-offset-slots.json',all_slots)
    report={'created_at':local_time(),'status':'clock_diagnostic_complete_no_retiming','results':results,
        'runtime_seconds':time.perf_counter()-started,'fixed_corner_slots_per_offset':49,'all_slots':len(all_slots),
        'eligible_nonzero_offsets':[r['frame_offset'] for r in results if r['frame_offset'] and r['complete_original_cohort']],
        'retiming_applied':False,'camera_changed':False,'original_failed_check_preserved':True,'reserved_camera_opened':False,
        'new_video_frames_decoded':0,'new_corner_detection_calls':0,'new_pose_solutions':0,'paid_calls':0,
        'limitations':['Different available-corner subsets cannot establish a timing improvement.',
            'Linear interpolation is a diagnostic approximation, not observed motion or a source-clock change.',
            'The four-pose original cohort still fails coverage; no offset alone qualifies the source calibration.']}
    write_json(out/'report.json',report)
    print(json.dumps({'results':results,'eligible_nonzero_offsets':report['eligible_nonzero_offsets'],'verification':seal_suite(out)}))


if __name__=='__main__':main()
