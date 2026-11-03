#!/usr/bin/env python3
"""Frozen, keyless three-video development evaluation using ordinary portal builders."""

import argparse
from copy import deepcopy
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import resource
import shutil
import signal
import subprocess
import sys
import threading
import time

PROFILE = 'unfamiliar-video-evaluation-v1'
SETTINGS = {
    'model': 'Lite', 'pose_mode': 'video', 'pose_identity': 'raw',
    'max_pose_hypotheses': 2, 'pose_sampling': 'bounded', 'native_timing': True,
    'anchor_profile': 'sealed-pose-visible-chain-tracking-anchor-v2',
    'modeling_scope': 'body_with_authored_ground_equipment_unmodeled',
    'root_assistance': False, 'seed': 37, 'training': False, 'retries': 0,
}
BOUNDS = {'case_seconds': 90, 'rss_bytes': 1536 << 20, 'output_bytes': 650_000_000,
          'max_decoded_frames': 500, 'upload_bytes': 37_000_000, 'max_controls': 500,
          'max_native_substeps': 8000, 'api_calls': 0}
INTENT = {'schema_version': 1, 'profile': 'portal-exercise-intent-v1', 'inventory': 'unknown',
          'output': 'simulation', 'basis': 'not_declared', 'absence_verified': False}


def digest(path):
    h = sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    raw = json.dumps(value, sort_keys=True, indent=2, allow_nan=False).encode() + b'\n'
    if len(raw) > 4 << 20:
        raise ValueError('Evaluation metadata exceeds 4 MiB')
    Path(path).write_bytes(raw)


def validate_plan(plan):
    if type(plan) is not dict or set(plan) != {'profile', 'settings', 'bounds', 'model', 'cases', 'scope'}:
        raise ValueError('Closed evaluation plan required')
    if plan['profile'] != PROFILE or plan['settings'] != SETTINGS or plan['bounds'] != BOUNDS:
        raise ValueError('Frozen settings or bounds differ')
    # Canonical comparison rejects bool/int aliases in the fixed protocol.
    for name, expected in [('settings', SETTINGS), ('bounds', BOUNDS)]:
        if json.dumps(plan[name], sort_keys=True) != json.dumps(expected, sort_keys=True):
            raise ValueError('Frozen typed settings differ')
    if type(plan['cases']) is not list or len(plan['cases']) != 3:
        raise ValueError('Exactly three predetermined cases are required')
    ids = []
    for case in plan['cases']:
        if type(case) is not dict or set(case) != {'id', 'original', 'original_sha256', 'upload', 'upload_sha256', 'provenance'}:
            raise ValueError('Closed source case required')
        if type(case['id']) is not str or not re.fullmatch('[a-z][a-z0-9-]{0,40}', case['id']):
            raise ValueError('Unsafe case id')
        ids.append(case['id'])
        for key in ['original_sha256', 'upload_sha256']:
            if type(case[key]) is not str or not re.fullmatch('[a-f0-9]{64}', case[key]):
                raise ValueError('Exact source pins required')
        if any(type(case[k]) is not str or not Path(case[k]).is_absolute() for k in ['original', 'upload', 'provenance']):
            raise ValueError('Absolute local source paths required')
    if len(set(ids)) != 3 or len({c['upload_sha256'] for c in plan['cases']}) != 3:
        raise ValueError('Three distinct sources required')
    model = plan['model']
    if type(model) is not dict or set(model) != {'path', 'sha256'} or not Path(model['path']).is_absolute() or not re.fullmatch('[a-f0-9]{64}', model['sha256']):
        raise ValueError('Pinned local pose model required')
    return deepcopy(plan)


def summary(rows):
    return {'denominator': len(rows), 'preflight_ready': sum(r.get('preflight_ready') is True for r in rows),
            'body_reconstructed': sum(r.get('body_reconstructed') is True for r in rows),
            'world_ready': sum(r.get('world_ready') is True for r in rows),
            'lab_ready': sum(r.get('lab_ready') is True for r in rows),
            'playable': sum(r.get('playable') is True for r in rows),
            'full_authored_rollout': sum(r.get('full_authored_rollout') is True for r in rows),
            'task_success': sum(r.get('task_success') is True for r in rows),
            'failed_or_refused': sum(r.get('status') != 'complete' for r in rows),
            'api_calls': 0, 'api_cost_usd': 0,
            'physical_accuracy': 'not_evaluated', 'independent_holdout': False,
            'scope': 'Normalized uploads; conditional body/authored ground only. No equipment recovery, training, rendering or physical validation.'}


def _tree(root):
    return {str(p.relative_to(root)): digest(p) for p in sorted(Path(root).rglob('*')) if p.is_file() and '__pycache__' not in p.parts}


def _check_inputs(case, model):
    for path_key, pin_key in [('original', 'original_sha256'), ('upload', 'upload_sha256')]:
        p = Path(case[path_key])
        if p.is_symlink() or not p.is_file() or p.stat().st_size > BOUNDS['upload_bytes'] or digest(p) != case[pin_key]:
            raise ValueError('Source file or identity changed')
    if Path(case['upload']).suffix.lower() != '.mp4' or digest(model['path']) != model['sha256']:
        raise ValueError('MP4 upload or model identity differs')


def _child(plan_path, index, out, runtime):
    started = time.monotonic(); deadline = started + BOUNDS['case_seconds']; out = Path(out)
    sys.path.insert(0, str(runtime)); os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
    plan = validate_plan(json.loads(Path(plan_path).read_bytes())); case = plan['cases'][index]
    row = {'id': case['id'], 'status': 'failed', 'stage': 'admission', 'api_calls': 0,
           'api_cost_usd': 0, 'settings': SETTINGS, 'source': case, 'physics_steps': 0,
           'preflight_ready': False, 'body_reconstructed': False, 'world_ready': False,
           'lab_ready': False, 'playable': False, 'full_authored_rollout': False, 'task_success': False}
    write(out/'progress.json', row)
    resource.setrlimit(resource.RLIMIT_CPU, (85, 88))
    resource.setrlimit(resource.RLIMIT_FSIZE, (128 << 20, 128 << 20))
    def expired(*_):
        raise TimeoutError('Frozen 90-second case deadline exceeded')
    signal.signal(signal.SIGALRM, expired); signal.setitimer(signal.ITIMER_REAL, BOUNDS['case_seconds'])
    def guard():
        while True:
            peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == 'darwin' else 1024)
            if peak > BOUNDS['rss_bytes']:
                write(out/'supervisor-failure.json', {'reason': 'RSS bound exceeded', 'peak_rss_bytes': peak}); os._exit(125)
            time.sleep(.1)
    threading.Thread(target=guard, daemon=True).start()
    session = None
    def remain(maximum):
        seconds = min(maximum, deadline - time.monotonic())
        if seconds < 1: raise TimeoutError('No bounded stage time remains')
        return seconds
    def stage(name):
        row['stage'] = name; write(out/'progress.json', row)
    # Standard-library audit observes every Python socket attempt, including imported providers.
    def audit(event, args):
        if event in ('socket.connect', 'socket.getaddrinfo', 'socket.bind'):
            raise PermissionError('Evaluation workers prohibit network/API access')
    sys.addaudithook(audit)
    try:
        _check_inputs(case, plan['model'])
        from weft.exercise_perception_pipeline import prepare_exercise_perception_preflight, validate_perception_preflight
        from weft.person_inventory import make_conditional_person_inventory
        from weft.shared_world_human import construct_human_only_run
        from weft.shared_exercise_lab import prepare_shared_exercise_lab, SharedExerciseSession
        stage('preflight'); tick = time.monotonic()
        result = prepare_exercise_perception_preflight(case['upload'], out/'preflight',
            pose_model=plan['model']['path'], max_seconds=remain(60), anchor_profile=SETTINGS['anchor_profile'])
        row['preflight_seconds'] = time.monotonic()-tick; row['preflight_result'] = result; row['preflight_ready'] = True
        stage('preflight_readmission')
        admitted = validate_perception_preflight(case['upload'], out/'preflight', deadline=min(deadline, time.monotonic()+30))
        row['preflight_manifest_sha256'] = admitted['manifest_sha256']
        inventory = make_conditional_person_inventory(case['upload_sha256'], INTENT); write(out/'inventory.json', inventory)
        stage('world'); tick = time.monotonic()
        built = construct_human_only_run(out/'preflight/body_source', inventory, out/'world', deadline=deadline, max_seconds=remain(30))
        row.update(world_ready=True, world_result=built, world_seconds=time.monotonic()-tick)
        stage('lab'); tick = time.monotonic()
        lab = prepare_shared_exercise_lab(out/'world', out/'lab', root_assistance=False, max_seconds=remain(30))
        row.update(lab_ready=True, lab_result=lab, lab_seconds=time.monotonic()-tick)
        stage('authored_rollout'); tick = time.monotonic(); session = SharedExerciseSession(out/'lab'); session.reset(SETTINGS['seed'])
        task = session.exercise_task; write(out/'task.json', task)
        horizon = task['interval']['horizon_steps']; substeps = session.scene['substeps']
        if horizon > BOUNDS['max_controls'] or horizon * substeps > BOUNDS['max_native_substeps']:
            raise ValueError('Frozen rollout work bound exceeded')
        count = 0; reward_sum = 0.; terminated = truncated = False
        with (out/'actions.jsonl').open('x') as journal:
            while not session.done:
                remain(1); action = session.reference_action(); reward, terminated, truncated, info = session.step(action)
                count += 1; reward_sum += reward; row['playable'] = info.get('valid_substeps', 0) > 0
                row['physics_steps'] = info.get('observed_substeps', count*substeps)
                journal.write(json.dumps({'step': count, 'action': action, 'reward': reward, 'terminated': terminated, 'truncated': truncated, 'metrics': info}, allow_nan=False)+'\n')
                if count > horizon: raise ValueError('Episode exceeded exact task horizon')
        snapshot = session.snapshot(); write(out/'final-state.json', snapshot); metrics = session.env.metrics()
        row.update(controls=count, native_substeps=count*substeps, metrics=metrics, mean_reward=reward_sum/max(1,count),
                   rollout_seconds=time.monotonic()-tick, full_authored_rollout=metrics.get('completed_interval') is True,
                   task_success=metrics.get('success') is True, status='complete', stage='complete')
    except Exception as error:
        row['error'] = type(error).__name__+': '+str(error)
    finally:
        if session is not None: session.env.close()
        signal.setitimer(signal.ITIMER_REAL, 0)
        body = out/'preflight/body_source'; row['body_reconstructed'] = (body/'manifest.json').is_file() and (body/'human.json').is_file() and not (body/'failure.json').exists()
        for name in ['manifest.json', 'failure.json', 'root-motion.json', 'evidence.json']:
            p = body/name
            if p.is_file(): row.setdefault('body_artifact_sha256', {})[name] = digest(p)
        try: _check_inputs(case, plan['model']); row['inputs_unchanged'] = True
        except Exception as e: row.update(status='failed', inputs_unchanged=False, final_input_error=str(e))
        row.update(seconds=time.monotonic()-started, peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin' else 1024))
        write(out/'result.json', row)
    return 0


def run(plan_path, out):
    plan_path = Path(plan_path).resolve(); plan = validate_plan(json.loads(plan_path.read_bytes()))
    out = Path(out).resolve(); out.mkdir(parents=True, exist_ok=False)
    repo = Path(__file__).resolve().parents[2]; runtime = out/'runtime'; (runtime/'weft').mkdir(parents=True)
    for source in sorted((repo/'src/weft').rglob('*')):
        relative = source.relative_to(repo/'src/weft')
        if source.is_file() and '__pycache__' not in relative.parts and (source.suffix == '.py' or relative.parts[0]=='assets'):
            if source.is_symlink(): raise ValueError('Runtime source symlink refused')
            target = runtime/'weft'/relative; target.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(source, target)
    runtime_pins = _tree(runtime); write(out/'runtime-pins.json', runtime_pins); shutil.copyfile(__file__, out/'runner.py')
    write(out/'plan.json', plan); rows=[]; start=time.monotonic()
    for index, case in enumerate(plan['cases']):
        case_out=out/case['id']; case_out.mkdir()
        env={key: os.environ[key] for key in ['PATH','TMPDIR','HOME','SYSTEMROOT'] if key in os.environ}
        env.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',VECLIB_MAXIMUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
        with (case_out/'worker.log').open('xb') as log:
            p=subprocess.Popen([sys.executable,str(out/'runner.py'),'--child',str(index),'--plan',str(out/'plan.json'),'--out',str(case_out),'--runtime',str(runtime)],env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            began=time.monotonic(); violation=None
            while p.poll() is None:
                if time.monotonic()-began > BOUNDS['case_seconds']+5: violation='Supervisor wall bound exceeded'
                if sum(f.stat().st_size for f in case_out.rglob('*') if f.is_file()) > BOUNDS['output_bytes']: violation='Output byte bound exceeded'
                if violation: os.killpg(p.pid,signal.SIGKILL); break
                time.sleep(.25)
            code=p.wait(timeout=5)
        if (case_out/'result.json').is_file(): row=json.loads((case_out/'result.json').read_bytes())
        else: row={'id':case['id'],'status':'failed','stage':'worker','error':violation or 'Worker exited without complete result'}
        row.update(worker_exit_code=code,worker_reaped=True,output_bytes=sum(f.stat().st_size for f in case_out.rglob('*') if f.is_file()))
        if code or violation: row.update(status='failed',supervisor_error=violation or 'Nonzero worker exit')
        write(case_out/'supervisor-result.json',row); rows.append(row); print(json.dumps({'id':case['id'],'status':row['status'],'stage':row.get('stage'),'error':row.get('error'),'seconds':row.get('seconds')}),flush=True)
    result={'profile':PROFILE,'summary':summary(rows),'cases':rows,'seconds':time.monotonic()-start,'runtime_unchanged':_tree(runtime)==runtime_pins,'runtime_pins_sha256':digest(out/'runtime-pins.json'),'plan_sha256':digest(out/'plan.json'),'runner_sha256':digest(out/'runner.py'),'all_workers_reaped':all(r['worker_reaped'] for r in rows)}
    write(out/'result.json',result); return result


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--plan',required=True);parser.add_argument('--out',required=True);parser.add_argument('--child',type=int);parser.add_argument('--runtime')
    args=parser.parse_args()
    if args.child is not None:sys.exit(_child(args.plan,args.child,args.out,Path(args.runtime)))
    run(args.plan,args.out)
