"""Frozen full-cycle baseline, source-only diagnostics, and gated optional RL checks."""
import argparse
from copy import deepcopy
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time

import numpy as np

from weft.benchmark import seal_suite
from weft.human import HumanEnv
from weft.io import canonical_hash, digest, local_time, write_json
from weft.pipeline import verify_artifacts
from weft.whole_body_feasibility import audit_reference_feasibility


VIDEO=Path('data/exercise/bodyweight-squats-danielflefil-500.mp4')
VIDEO_SHA='ec636e28b4711887ecd9ff7014b64801d16055faf2f2b0bf6ed9221ea8280118'
LABELS=VIDEO.with_suffix('.manual.json')
PACKAGE=Path('artifacts/exercise-fullbody-source-01')
BRIDGE=PACKAGE/'candidate-02/conversion-verification.json'
MODEL=Path('.local/models/pose_landmarker_lite.task')


def load(path):
    return json.loads(Path(path).read_text())


def prepare(out):
    out.mkdir(parents=True,exist_ok=False)
    assert digest(VIDEO)==VIDEO_SHA
    labels,bridge=load(LABELS),load(BRIDGE)
    assert bridge['source_sha256']==labels['source_sha256'] and bridge['derivative_sha256']==VIDEO_SHA
    assert bridge['decoded_frames']==68 and bridge['all_rgb_pixels_and_timestamps_equal']
    paths=[VIDEO,LABELS,VIDEO.with_suffix('.source.json'),MODEL,BRIDGE,PACKAGE/'manifest.json',
           Path('requirements-lab.lock'),*sorted(Path('src/weft').glob('*.py'))]
    hashes={str(path):digest(path) for path in paths}
    for path in paths:
        if path in (VIDEO,MODEL):
            continue
        target=out/'freeze'/path;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(path,target)
    command=['.venv-lab/bin/python','-m','weft','lab','--video',str(VIDEO),'--out',str(out/'baseline'),
        '--profile','exercise','--pose-model',str(MODEL),'--pose-mode','video','--pose-identity','temporal',
        '--body-model','mobile','--episodes','100','--eval-episodes','100','--seed','1729','--no-serve',
        '--source-note','Complete 68-frame official 500px GIF cycle, lossless RGB MP4 derivative; '
        'source-only labels and provenance in artifacts/exercise-fullbody-source-01; not raw-camera timing or calibrated physical observations']
    write_json(out/'pre-run-freeze.json',{'created_at':local_time(),'command':command,'input_hashes':hashes,
        'source_package_verification':verify_artifacts(PACKAGE),'maximum_wall_seconds':240,'full_source_frames':68,
        'source_trim':False,'external_api_usd':0,
        'conditional_whole_body_training':{'requires':'baseline ready and full-reference floor feasible',
            'horizon_steps':20,'generations':8,'population':16,'seed':1729,'manual_recording_steps':12},
        'source_label_matching':{'nearest_source_frame_bound':1,'tie':'earlier','all_five_slots_retained':True,
                                 'model_terminal_time_clamping':False},
        'foot_proxy_scope':'Source heel-toe midpoint estimates may be compared as anonymous point sets; '
                           'not ankle/sole equivalence, anatomical side assignment, or force validation'})


def execute(out):
    settings=load(out/'pre-run-freeze.json')
    if (out/'baseline').exists():
        raise FileExistsError('Do not rerun or overwrite a baseline; use --finalize or a new directory')
    assert all(digest(path)==value for path,value in settings['input_hashes'].items())
    started=time.monotonic()
    with (out/'baseline-command.log').open('w') as log:
        process=subprocess.Popen(settings['command'],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        try:
            code=process.wait(timeout=240);timed_out=False
        except subprocess.TimeoutExpired:
            os.killpg(process.pid,signal.SIGTERM)
            try:
                code=process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid,signal.SIGKILL);code=process.wait()
            timed_out=True
    write_json(out/'baseline-command-result.json',{'created_at':local_time(),'returncode':code,'timed_out':timed_out,
        'seconds':time.monotonic()-started,'pid':process.pid,'ownership':'This study process group only'})


def image_point(frame, indices, width, height):
    landmarks=frame.get('landmarks')
    if not isinstance(landmarks,list) or len(landmarks)!=33:
        return {'status':'missing_landmarks','point_px':None,'confidence':None}
    points=[landmarks[index] for index in indices]
    try:
        xy=np.array([[p['x'],p['y']] for p in points],float)
        confidence=min(float(p[k]) for p in points for k in ['visibility','presence'])
    except (KeyError,TypeError,ValueError):
        return {'status':'invalid_landmarks','point_px':None,'confidence':None}
    if not np.isfinite(xy).all() or not math.isfinite(confidence):
        return {'status':'nonfinite_landmarks','point_px':None,'confidence':None}
    return {'status':'model_image_estimate','point_px':(xy.mean(axis=0)*[width,height]).tolist(),
            'confidence':confidence,'outside_actual_image':bool(np.any(xy<0) or np.any(xy>1)),
            'source_landmark_indices':indices,'scope':'Inferred image location, not measured anatomy or contact'}


def grade_source(out):
    run=out/'baseline/run'
    labels,bridge=load(LABELS),load(BRIDGE)
    assert labels['selected_frames']==[0,13,33,53,67]
    assert labels['source_sha256']==bridge['source_sha256'] and bridge['derivative_sha256']==VIDEO_SHA
    assert bridge['decoded_frames']==68 and all(row['source_rgb_sha256']==row['decoded_rgb_sha256'] for row in bridge['frames'])
    evidence=load(run/'pose-landmarks.json') if (run/'pose-landmarks.json').exists() else {'frames':[]}
    root=load(run/'root-motion.json') if (run/'root-motion.json').exists() else None
    spec=load(run/'human.json') if (run/'human.json').exists() else None
    env=HumanEnv(spec) if spec else None
    first=env.reference_snapshot(0) if env else None
    width,height=bridge['dimensions']
    rows=[]
    try:
        for label in labels['observations']:
            row={'label':deepcopy(label),'gif_source_sha256':labels['source_sha256'],'derivative_source_sha256':VIDEO_SHA}
            nearest=min(evidence['frames'],key=lambda r:(abs(r['frame']-label['frame']),r['frame'])) if evidence['frames'] else None
            matched=nearest is not None and abs(nearest['frame']-label['frame'])<=1
            usable=matched and nearest['status'] in ['tracked','tracked_partial'] and nearest.get('pose_count',1)==1
            row.update(prediction_frame=nearest['frame'] if matched else None,
                       prediction_time_s=nearest['time_s'] if matched else None,
                       prediction_time_offset_s=nearest['time_s']-label['time_ms']/1000 if matched else None,
                       prediction_status=nearest['status'] if matched else 'no_sample_within_one_source_frame')
            row['raw_pose_root_comparison']={'status':'unavailable'}
            row['source_foot_proxy_comparison']={'status':'unavailable','anatomical_side_assignment':None}
            if usable:
                pelvis=image_point(nearest,[23,24],width,height)
                shoulder=image_point(nearest,[11,12],width,height)
                row['raw_pose_root_comparison']={'pelvis':pelvis,'shoulder':shoulder}
                if pelvis['point_px'] is not None:
                    error=math.dist(pelvis['point_px'],label['image_pelvis_midpoint_px'])
                    row['raw_pose_root_comparison'].update(status='compared_image_estimates',pelvis_error_px=error,
                        within_subjective_pelvis_allowance=error<=label['pelvis_uncertainty_radius_px'])
                if shoulder['point_px'] is not None:
                    row['raw_pose_root_comparison']['shoulder_error_px']=math.dist(shoulder['point_px'],label['shoulder_midpoint_px'])
                proxies=[{'proxy_id':'landmarks_29_31_midpoint',**image_point(nearest,[29,31],width,height)},
                         {'proxy_id':'landmarks_30_32_midpoint',**image_point(nearest,[30,32],width,height)}]
                foot={'status':'proxy_estimates_retained','predictions':proxies,'manual_sole_proxies':label['foot_contact_proxies'],
                    'anatomical_side_assignment':None,
                    'scope':'Heel-to-toe midpoint versus visible sole-boundary proxy: different definitions; anonymous proximity only, no foot/contact pass claim'}
                if all(p['point_px'] is not None for p in proxies):
                    distances=[[math.dist(p['point_px'],manual['point_px']) for manual in label['foot_contact_proxies']] for p in proxies]
                    foot['anonymous_distance_matrix_px']=distances
                    foot['unordered_pair_mean_error_px']=min((distances[0][0]+distances[1][1])/2,(distances[0][1]+distances[1][0])/2)
                    foot['both_pairing_mean_errors_px']=[(distances[0][0]+distances[1][1])/2,(distances[0][1]+distances[1][0])/2]
                row['source_foot_proxy_comparison']=foot
            row['model_reference_projection']={'status':'no_body_model','anatomical_sole_comparison':'unavailable_model_has_no_feet_or_soles'}
            timestamp=label['time_ms']/1000
            if env:
                times=np.array([r['time_s'] for r in spec['reference']])
                if timestamp<times[0]-1e-9 or timestamp>times[-1]+1e-9:
                    row['model_reference_projection']['status']='outside_reference_time_coverage_no_endpoint_clamp'
                else:
                    index=int(np.argmin(abs(times-timestamp)))
                    if abs(times[index]-timestamp)>1/bridge['fps']+1e-9:
                        row['model_reference_projection']['status']='no_reference_within_one_source_frame'
                    elif not root or root.get('scale') is None:
                        row['model_reference_projection']['status']='missing_root_projection_assumption'
                    else:
                        state=env.reference_snapshot(index)
                        origin=np.array(root['origin']['pelvis_image_px'])
                        base=np.array(first['human']['joints']['pelvis'])
                        scale=root['scale']['model_units_per_pixel']
                        def project(point):
                            delta=np.array(point)-base
                            return (origin+np.array([delta[0],-delta[2]])/scale).tolist()
                        projection={name:project(state['human']['joints'][name]) for name in
                                    ['pelvis','neck','left_knee','right_knee','left_ankle','right_ankle']}
                        row['model_reference_projection'].update(status='authored_origin_anchored_projection',
                            reference_index=index,reference_time_s=float(times[index]),reference_time_offset_s=float(times[index]-timestamp),
                            projected_points_px=projection,
                            authored_floor_image_y_if_same_projection=float(origin[1]+base[2]/scale),
                            scope='Uncalibrated x,-z projection anchored at inferred first pelvis; model base height/floor screen plane are authored, not measured. Ankle endpoints are not sole proxies.')
            rows.append(row)
    finally:
        if env:env.close()
    paired=[r for r in rows if 'pelvis_error_px' in r['raw_pose_root_comparison']]
    result={'schema_version':1,'matching':'Nearest raw source sample within one frame; earlier ties. Body labels outside reference interval remain unavailable.',
        'label_method':labels['label_method'],'original_label_sha256':digest(LABELS),'conversion_bridge_sha256':digest(BRIDGE),
        'gif_to_mp4_identity_preserved':True,'rows':rows,'expected_label_slots':5,
        'raw_pose_pelvis_comparisons':len(paired),
        'raw_pose_pelvis_within_subjective_allowance':sum(r['raw_pose_root_comparison']['within_subjective_pelvis_allowance'] for r in paired),
        'raw_pose_pelvis_rmse_px':math.sqrt(np.mean([r['raw_pose_root_comparison']['pelvis_error_px']**2 for r in paired])) if paired else None,
        'available_model_projection_slots':sum(r['model_reference_projection']['status']=='authored_origin_anchored_projection' for r in rows),
        'scope':'Source-only approximate image comparisons; no camera calibration, anatomical foot identity, physical contact, forces, balance or biomechanical validation'}
    assert len(result['rows'])==5
    write_json(out/'source-grading.json',result)
    shutil.copyfile(LABELS,out/'original-source-labels.json');shutil.copyfile(BRIDGE,out/'source-conversion-bridge.json')
    return result


def finalize(out):
    if (out/'report.json').exists() or (out/'manifest.json').exists():
        raise FileExistsError('Study already finalized; preserve it')
    freeze=load(out/'pre-run-freeze.json')
    assert all(digest(path)==value for path,value in freeze['input_hashes'].items())
    status=load(out/'baseline/status.json')
    if status['status'] not in ['ready','failed','cancelled']:
        raise RuntimeError('Wait for the owned baseline worker to reach a terminal state')
    run=out/'baseline/run'
    baseline_verification=verify_artifacts(run) if (run/'manifest.json').exists() else None
    grading=grade_source(out)
    spec=load(run/'human.json') if (run/'human.json').exists() else None
    floor=None
    if spec:
        if (out/'baseline-feasibility.json').exists():
            floor=load(out/'baseline-feasibility.json');assert floor['model_sha256']==canonical_hash(spec)
        else:
            floor=audit_reference_feasibility(spec);write_json(out/'baseline-feasibility.json',floor)
    additional={'status':'not_run','reason':'Baseline or full-reference floor precondition failed',
                'predeclared_settings':freeze['conditional_whole_body_training']}
    if status['status']=='ready' and floor and floor['status']=='feasible':
        from weft.whole_body_learning import train_whole_body
        from weft.whole_body_console import console
        learning=train_whole_body(spec,out/'whole-body-learning',seed=1729,generations=8,population=16,horizon_steps=20,max_seconds=120)
        commands=['motor root_x .1 4','release 4','follow 4','quit']
        pending,output=iter(commands),[]
        recording=console(out/'whole-body-learning',out/'sessions/manual',seed=1729,
                          input_fn=lambda _:next(pending),output_fn=output.append)
        write_json(out/'manual-recording-output.json',output)
        additional={'status':'completed','learning':learning,'recording':recording,'commands':commands,
                    'scope':'Explicit virtual-motor control, not human balance or physiological action'}
    assert all(digest(path)==value for path,value in freeze['input_hashes'].items())
    if baseline_verification:
        assert baseline_verification==verify_artifacts(run)
    report={'schema_version':1,'created_at':local_time(),'baseline_command':load(out/'baseline-command-result.json'),
        'baseline_status':status['status'],'baseline_verification':baseline_verification,
        'decoded_source_frames':load(run/'evidence.json').get('decoded_frames') if (run/'evidence.json').exists() else None,
        'source_grading_summary':{k:v for k,v in grading.items() if k not in ['rows']},
        'floor_status':floor['status'] if floor else 'no_body_model','floor_summary':floor['summary'] if floor else None,
        'baseline_authored_servo_evaluation':load(run/'human-evaluation.json') if (run/'human-evaluation.json').exists() else None,
        'baseline_single_joint_evaluation':load(run/'rl-evaluation.json') if (run/'rl-evaluation.json').exists() else None,
        'additional_whole_body':additional,'external_api_usd':0,
        'scope':'Fresh full-cycle development source, unchanged baseline and source-only checks; single-joint success is not whole-body contact feasibility or physical validation'}
    shutil.copyfile(Path(__file__),out/'check_fullbody_exercise.py')
    write_json(out/'report.json',report)
    print(json.dumps({'baseline':status['status'],'floor':report['floor_status'],'floor_summary':report['floor_summary'],
        'grading':report['source_grading_summary'],'extra_training':additional['status'],'verification':seal_suite(out)},indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--out',type=Path,required=True)
    mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--prepare',action='store_true')
    mode.add_argument('--run',action='store_true')
    mode.add_argument('--finalize',action='store_true')
    args=parser.parse_args()
    if args.prepare:prepare(args.out)
    elif args.run:execute(args.out);finalize(args.out)
    else:finalize(args.out)
