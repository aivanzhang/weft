"""Read-only, bounded custody proof for failed Studio jobs."""

from contextlib import contextmanager
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import selectors
import sqlite3
import stat
import subprocess
import time

from avatar_assets.studio_jobs import MAX_JSON, _digest, _local_time, _require


MAX_FILES = 20000
MAX_RECEIPTS = 512
MAX_RECEIPT_BYTES = 4 * 1024**2


def _receipt(path,limit=MAX_JSON):
    _require(path.resolve()==path and path.is_file(),'Custody receipt path is unsafe or missing')
    fd=os.open(path,os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(fd,'rb') as stream:
        info=os.fstat(stream.fileno())
        _require(stat.S_ISREG(info.st_mode) and info.st_size<=limit,'Custody receipt is unsafe or oversized')
        raw=stream.read(limit+1)
    _require(len(raw)<=limit,'Custody receipt grew beyond its bound')
    value=json.loads(raw,parse_constant=lambda _:(_ for _ in ()).throw(ValueError('Nonfinite custody JSON')))
    _require(isinstance(value,dict),'Invalid custody receipt')
    return value,{'path':str(path),'sha256':hashlib.sha256(raw).hexdigest(),'bytes':len(raw)}


@contextmanager
def custody_locks(paths):
    """Existing locks fence cooperating supervisors and direct pipeline executions without new files."""
    streams=[]
    try:
        for path in paths:
            _require(path.resolve()==path and path.is_file(),'Existing custody lock is missing or unsafe')
            fd=os.open(path,os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
            stream=os.fdopen(fd,'rb');streams.append(stream)
            _require(stat.S_ISREG(os.fstat(fd).st_mode),'Custody lock is not a regular file')
            try:fcntl.flock(fd,fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:raise ValueError('A supervisor or pipeline still owns its lock; no release') from None
        yield
    finally:
        for stream in reversed(streams):stream.close()


def _process_snapshot():
    """Inspect PID/session groups without command arguments, environment or signals to workers."""
    command=['/bin/ps','-axo','pid=,ppid=,pgid=']
    process=subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
                             env={'PATH':'/usr/bin:/bin','LC_ALL':'C'})
    data=bytearray();deadline=time.monotonic()+2
    try:
        with selectors.DefaultSelector() as selector:
            os.set_blocking(process.stdout.fileno(),False)
            selector.register(process.stdout,selectors.EVENT_READ)
            while selector.get_map():
                remaining=deadline-time.monotonic()
                _require(remaining>0,'OS custody inspection exceeded two seconds')
                for key,_ in selector.select(remaining):
                    chunk=os.read(key.fd,65536)
                    if not chunk:selector.unregister(key.fileobj)
                    else:
                        data.extend(chunk)
                        _require(len(data)<=512*1024,'OS custody snapshot exceeds512KiB')
        _require(process.wait(timeout=max(.01,deadline-time.monotonic()))==0,'OS custody inspection failed')
    finally:
        if process.poll() is None:
            process.kill();process.wait(timeout=1)
        process.stdout.close()
    rows=[]
    for line in data.decode('ascii').splitlines():
        fields=line.split()
        _require(len(fields)==3 and all(field.isdigit() for field in fields),'Invalid OS custody snapshot')
        rows.append(tuple(int(field) for field in fields))
    _require(rows and len(rows)<=MAX_FILES,'OS process inventory is empty or oversized')
    return rows,hashlib.sha256(data).hexdigest()


def assert_absent(pids):
    rows,checksum=_process_snapshot()
    matched=[{'pid':pid,'parent_pid':parent,'process_group':group} for pid,parent,group in rows
             if pid in pids or parent in pids or group in pids]
    _require(not matched,'A recorded owner, child or process group is still present; no release')
    return {'recorded_local':_local_time(time.time()),'known_pid_count':len(pids),
            'os_snapshot_sha256':checksum,'matched_processes':[],
            'scope':'Recorded PIDs, direct children and process groups are absent; no worker was signalled.'}


def collect(root,output,invocation,owner_pid,owner_identity,*,operation='execute'):
    root,output,invocation=map(Path,(root,output,invocation))
    _require(output.resolve()==output and output.is_dir() and output.is_relative_to(root/'outputs'),'Unsafe inspected job output')
    _require(invocation.resolve()==invocation and invocation.is_dir() and invocation.is_relative_to(root/'.local'),'Unsafe inspected invocation')
    owner,_=_receipt(invocation/'owner.json')
    _require(owner.get('pid')==owner_pid and owner.get('start_identity')==owner_identity,
             'Sealed supervisor owner differs from the retained lease')
    child,_=_receipt(invocation/'worker-job.json')
    _require(child.get('process_exit_verified') is True
             and type(child.get('exit_code')) is int and child['exit_code'] in (0,1) and child.get('status')=='finished'
             and child.get('manual_stop_required') is False,'Controller child has no sealed safe exit receipt')
    terminal_paths={invocation/'result.json',invocation/'supervisor-result.json'}
    paths={invocation/'owner.json',invocation/'worker-job.json',invocation/'invocation.json',*terminal_paths}
    deadline=time.monotonic()+8;files=0
    for folder in (output,invocation):
        for directory,dirs,names in os.walk(folder,followlinks=False):
            _require(time.monotonic()<deadline,'Custody receipt inventory exceeded eight seconds')
            files+=len(dirs)+len(names);_require(files<=MAX_FILES,'Custody file inventory exceeds its bound')
            for name in dirs+names:
                _require(not (Path(directory)/name).is_symlink(),'Custody inventory contains a symlink')
            for name in names:
                if name=='job.json' or name.endswith(('-handle.json','-monitor.json','-process.json','-job.json')):
                    paths.add(Path(directory)/name)
                    _require(len(paths)<=MAX_RECEIPTS,'Too many custody receipts')
    records=[];pids={owner_pid};closed=set();exits={};handles=[];total=0
    for path in sorted(paths):
        value,record=_receipt(path)
        total+=record['bytes'];_require(total<=MAX_RECEIPT_BYTES,'Custody receipt bytes exceed4MiB')
        record['path']=str(path.relative_to(root))
        if path in terminal_paths:
            _require(value.get('status')=='needs_inspection' and value.get('safe_to_release') is False,
                     'Controller or supervisor did not seal the inspected terminal result')
            records.append({**record,'kind':'terminal_result'});continue
        if path==invocation/'invocation.json':
            _require(value.get('root')==str(root) and value.get('invocation_dir')==str(invocation)
                     and value.get('token')==owner.get('token') and operation in ('execute','plan')
                     and value.get('operation')==operation,
                     'Invocation receipt differs from inspected ownership')
            records.append({**record,'kind':'invocation'});continue
        pid=value.get('pid')
        _require(type(pid) is int and pid>0,'Custody receipt has no known process owner')
        pids.add(pid)
        kind='owner' if path==invocation/'owner.json' else 'handle' if path.name.endswith('-handle.json') else 'exit'
        if kind=='handle':
            monitor=path.name[:-len('-handle.json')]+'-monitor.json'
            _require(value.get('monitor_checkpoint')==monitor,'Handle does not bind its exact monitor receipt')
            handles.append((path.with_name(monitor),pid))
        if kind=='exit':
            _require(value.get('status')=='finished' and type(value.get('exit_code')) is int
                     and value.get('manual_stop_required') is False
                     and (value.get('process_exit_verified') is True or
                          path.name.endswith('-process.json') and 'process_exit_verified' not in value),
                     'Unsealed or unresolved process receipt; no inspection release')
            closed.add(pid);exits[path]=pid
        records.append({**record,'kind':kind,'pid':pid,'exit_code':value.get('exit_code') if kind=='exit' else None})
    _require(pids-{owner_pid}<=closed,'A recorded handle lacks a terminal process receipt')
    _require(all(exits.get(path)==pid for path,pid in handles),'A handle lacks its matching terminal monitor receipt')
    return {'receipts':records,'known_pids':sorted(pids),'receipt_bytes':total,
            'process_check':assert_absent(pids),'bounds':{'inventory_entries':MAX_FILES,
                'receipts':MAX_RECEIPTS,'receipt_bytes':MAX_RECEIPT_BYTES,'inventory_seconds':8,'os_snapshot_seconds':2},
            'scope':'Local source-reuse custody only. No accepted assets, remote-task completion, refunded spend or replay authorization.'}


def recheck(root,proof):
    root=Path(root)
    for record in proof['receipts']:
        path=root/record['path']
        _require(path.is_relative_to(root),'Custody receipt escaped the tool root')
        _,current=_receipt(path)
        _require(current['bytes']==record['bytes'] and current['sha256']==record['sha256'],
                 'Custody receipt changed during inspection')
    return assert_absent(set(proof['known_pids']))


def _file_pin(path, limit):
    _require(path.resolve()==path and not any(p.is_symlink() for p in (path,*path.parents)),
             'Remote custody path is unsafe')
    fd=os.open(path,os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(fd,'rb') as stream:
        before=os.fstat(stream.fileno())
        _require(stat.S_ISREG(before.st_mode) and 0<before.st_size<=limit,'Remote custody file exceeds its bound')
        checksum=hashlib.sha256();size=0
        while block:=stream.read(256*1024):
            size+=len(block);_require(size<=limit,'Remote custody file grew beyond its bound');checksum.update(block)
        after=os.fstat(stream.fileno())
    _require(size==before.st_size and (before.st_size,before.st_mtime_ns)==(after.st_size,after.st_mtime_ns),
             'Remote custody file changed during read')
    return {'path':str(path),'bytes':size,'sha256':checksum.hexdigest()}


def _entries(path,limit):
    _require(path.resolve()==path and path.is_dir(),'Remote custody directory is unsafe')
    names=[]
    for index,item in enumerate(path.iterdir()):
        _require(index<limit and not item.is_symlink(),'Remote custody inventory is unsafe or oversized')
        names.append(item.name)
    return sorted(names)


def include_previous_plans(root,invocation,proof,request_digest):
    """A finished planner can still have an escaped child; retain every prior PID for the final OS check."""
    root,invocation=map(Path,(root,invocation))
    names=[name for name in _entries(invocation.parent,64) if name.startswith('run-')]
    _require(1<=len(names)<=8 and names==[f'run-{i:04d}' for i in range(1,len(names)+1)]
             and names[-1]==invocation.name,'Unexpected prior Studio invocation history')
    records=list(proof['receipts']);pids=set(proof['known_pids']);size=proof['receipt_bytes']

    def read(path,kind):
        nonlocal size
        value,pin=_receipt(path);size+=pin['bytes']
        _require(len(records)<MAX_RECEIPTS and size<=MAX_RECEIPT_BYTES,'Prior plan custody exceeds its bound')
        records.append({**pin,'path':str(path.relative_to(root)),'kind':kind})
        return value

    for name in names[:-1]:
        folder=invocation.parent/name
        owner=read(folder/'owner.json','prior_plan_owner')
        meta=read(folder/'invocation.json','prior_plan_invocation')
        _require(meta.get('root')==str(root) and meta.get('invocation_dir')==str(folder)
                 and meta.get('job_id')==invocation.parent.name and meta.get('request_digest')==request_digest
                 and meta.get('operation')=='plan' and meta.get('token')==owner.get('token'),
                 'Prior invocation is not the same immutable planning request')
        _require(type(owner.get('pid')) is int and owner['pid']>0
                 and isinstance(owner.get('start_identity'),str) and 0<len(owner['start_identity'])<=256,
                 'Prior planner has no process identity')
        pids.add(owner['pid'])
        child=read(folder/'worker-job.json','prior_plan_exit')
        monitor=read(folder/'worker-monitor.json','prior_plan_exit')
        handle=read(folder/'worker-handle.json','prior_plan_handle')
        _require(type(child.get('pid')) is int and child['pid']>0 and monitor.get('pid')==child['pid']
                 and handle.get('pid')==child['pid'] and handle.get('monitor_checkpoint')=='worker-monitor.json',
                 'Prior planner child/handle identity changed')
        for row in (child,monitor):
            _require(row.get('status')=='finished' and type(row.get('exit_code')) is int and row['exit_code']==0
                     and row.get('process_exit_verified') is True and row.get('manual_stop_required') is False,
                     'Prior planner child is not safely closed')
        pids.add(child['pid'])
        results=[read(folder/name,'prior_plan_terminal') for name in ('result.json','supervisor-result.json')]
        _require(results[0].get('status') in ('planned','awaiting_authorization')
                 and results[1].get('status')==results[0]['status']
                 and all(row.get('safe_to_release') is True for row in results),
                 'Prior planning result is not safely terminal')
    return {**proof,'receipts':records,'known_pids':sorted(pids),'receipt_bytes':size,
            'previous_planning_invocations':len(names)-1}


def collect_generation(root,output,request):
    """Admit only completed, fully granted text mesh+texture jobs; never contact a provider."""
    from avatar_assets.meshy import public_value
    root,output=map(Path,(root,output))
    _require(output.resolve()==output and output.is_relative_to(root/'outputs'), 'Unsafe remote job output')
    _require(request.get('reuse_model') is None and request.get('reference') is None,
             'Fresh inspection currently supports text generation without references only')
    generation=output/'generation';records=[];receipt_bytes=0;started=time.monotonic()

    def read(path):
        nonlocal receipt_bytes
        _require(time.monotonic()-started<15 and len(records)<300,'Remote receipt inventory exceeded its bound')
        value,pin=_receipt(path);receipt_bytes+=pin['bytes']
        _require(receipt_bytes<=MAX_RECEIPT_BYTES,'Remote receipts exceed4MiB')
        records.append(pin)
        return value

    pipeline_request=read(output/'request.json')
    _require('agent_backend' not in pipeline_request,
             'API-backed agent custody needs separate inspection; this legacy release cannot clear uncertain API calls')
    _require(all(pipeline_request.get(k)==request.get(k) for k in
                 ('prompt','style','custom_actions','preset','reference','reuse_model'))
             and pipeline_request.get('id')==output.name and pipeline_request.get('max_credits')==35,
             'Fresh pipeline request differs from Studio')
    plan=read(generation/'plan.json');config=read(generation/'inputs/config.json')
    _require(plan.get('schema_version')==1 and plan.get('id')==output.name+'-mesh'
             and plan.get('model')=='meshy-7.1' and plan.get('max_credits')==35
             and plan.get('estimated_credits')==35 and plan.get('estimated_paid_calls')==2
             and plan.get('automatic_paid_retries')==0 and plan.get('config')==config
             and config==read(output/'generation-config.json')
             and config.get('id')==plan['id'] and config.get('rig')=='none' and config.get('states')==[]
             and config.get('generation',{}).get('mode')=='text',
             'Fresh release requires the exact unrigged35-credit text mesh+texture plan')
    plan_pin=next(p for p in records if p['path']==str(generation/'plan.json'))
    _require(read(output/'generation-receipt.json')=={'plan_sha256':plan_pin['sha256']},
             'Generation plan receipt changed')
    source_names=['meshy.py','meshy_access.py']
    _require(_entries(generation/'sources',2)==source_names,'Generation source inventory changed')
    source_rows=[]
    for name in source_names:
        pin=_file_pin(generation/'sources'/name,MAX_JSON);records.append(pin)
        source_rows.append({'path':name,'bytes':pin['bytes'],'sha256':pin['sha256']})
    _require(plan.get('sources')==source_rows,'Generation implementation snapshots changed')
    _require(not (output/'delivery.json').exists(),'A completed delivery cannot be released as failed')
    stage_dir=generation/'stages'
    _require(_entries(stage_dir,2)==['mesh','texture'],
             'Unexpected, missing or unfinished remote stage')
    ledger=root.parents[1]/'.local/meshy/ledger.sqlite3'
    ledger_pin=_file_pin(ledger,16*1024**2)
    _require(ledger.stat().st_uid==os.getuid() and stat.S_IMODE(ledger.stat().st_mode)==0o600,
             'Canonical ledger is not owner-only')
    _require(not any(Path(str(ledger)+suffix).exists() for suffix in ('-wal','-journal','-shm')),
             'Canonical ledger has an active journal; inspect later')
    db=sqlite3.connect(ledger.as_uri()+'?mode=ro&immutable=1',uri=True,timeout=.2)
    db.row_factory=sqlite3.Row
    try:
        _require(db.execute('SELECT count(*) FROM calls').fetchone()[0]<=10000,'Canonical call inventory exceeds bound')
        bound=db.execute('SELECT output,plan_hash,credit_limit FROM jobs WHERE id=?',(plan['id'],)).fetchone()
        grant=db.execute('SELECT credits,plan_hash,approval FROM budget_grants WHERE job=?',(plan['id'],)).fetchone()
        calls=[dict(row) for row in db.execute('SELECT * FROM calls WHERE job=? ORDER BY step',(plan['id'],))]
    finally:db.close()
    _require(_file_pin(ledger,16*1024**2)==ledger_pin,'Canonical ledger changed during inspection')
    _require(bound is not None and tuple(bound)==(str(generation),_digest(plan),35)
             and grant is not None and grant['credits']==35 and grant['plan_hash']==_digest(plan),
             'Canonical plan or exact job grant differs')
    _require(isinstance(grant['approval'],str) and len(grant['approval'].encode())<=8192,'Invalid canonical approval')
    approval=json.loads(grant['approval'])
    _require(isinstance(approval,dict) and approval.get('decision')=='approved'
             and approval.get('job_id')==plan['id'] and approval.get('plan_digest')==_digest(plan)
             and type(approval.get('additional_credits')) is int and approval['additional_credits']==35
             and isinstance(approval.get('user_reply'),str) and 0<len(approval['user_reply'])<=4000,
             'Canonical job approval is missing or changed')
    _require([c['step'] for c in calls]==['mesh','texture'],'Canonical remote stages differ from the plan')
    summaries=[];download_bytes=0
    for call,credits in zip(calls,(25,10)):
        _require(call['state']=='SUCCEEDED' and call['credits']==credits
                 and call['endpoint']=='/openapi/v2/text-to-3d'
                 and isinstance(call['task_id'],str) and 0<len(call['task_id'])<=128,
                 'Remote task is not fully successful with its exact charge')
        _require(all(isinstance(call[k],str) and len(call[k].encode())<=MAX_JSON for k in ('request','response')),
                 'Canonical remote payload exceeds its bound')
        reserved,response=json.loads(call['request']),json.loads(call['response'])
        _require(isinstance(reserved,dict) and isinstance(response,dict)
                 and response.get('id')==call['task_id'] and response.get('status')=='SUCCEEDED'
                 and type(response.get('consumed_credits')) is int and response['consumed_credits']==credits
                 and reserved.get('endpoint')==call['endpoint'] and reserved.get('estimated_credits')==credits
                 and reserved.get('request_sha256')==_digest(reserved.get('body')),
                 'Canonical successful task/request accounting differs')
        gen=config['generation']
        expected_body=({'ai_model':plan['model'],'model_type':'standard','geometry_resolution':gen['geometry_resolution'],
                        'should_remesh':True,'target_polycount':gen['target_polycount'],'topology':'triangle',
                        'pose_mode':gen['pose_mode'],'target_formats':['glb'],'mode':'preview','prompt':config['prompt']}
                       if call['step']=='mesh' else
                       {'mode':'refine','ai_model':plan['model'],'preview_task_id':calls[0]['task_id'],'enable_pbr':True,
                        'texture_resolution':gen['texture_resolution'],'texture_prompt':config['style'],'target_formats':['glb']})
        _require(reserved['body']==expected_body,'Submitted remote body differs from the approved plan')
        folder=stage_dir/call['step']
        names=_entries(folder,125)
        status_names=sorted(n for n in names if n.startswith('status-') and n.endswith('.json'))
        _require(1<=len(status_names)<=120
                 and status_names==[f'status-{i:03d}.json' for i in range(len(status_names))]
                 and set(names)=={'request-000.json','submission-000.json','download-000.json','downloads','tools',*status_names},
                 'Remote stage has retries, failures or unrecognized receipts')
        saved=read(folder/'request-000.json')
        _require({k:v for k,v in saved.items() if k!='execution_sources'}==reserved,
                 'Saved remote request differs from canonical reservation')
        submission=read(folder/'submission-000.json')
        _require(submission.get('task_id')==call['task_id'] and submission.get('estimated_credits')==credits,
                 'Remote submission identity differs')
        final=None
        executed=[saved,submission]
        for index,name in enumerate(status_names):
            observed=read(folder/name);task=observed.get('task')
            executed.append(observed)
            _require(isinstance(task,dict) and task.get('id')==call['task_id']
                     and task.get('status') in ('PENDING','IN_PROGRESS','SUCCEEDED'),
                     'Remote history contains failed, uncertain or unrelated task')
            _require(index==len(status_names)-1 or task['status']!='SUCCEEDED',
                     'Remote history continued after terminal success')
            final=task
        _require(final==public_value(response),'Latest remote success differs from canonical response')
        fetched=read(folder/'download-000.json')
        executed.append(fetched)
        source_hashes={p['path']:p['sha256'] for p in source_rows}
        _require(all(row.get('execution_sources')==source_hashes for row in executed),
                 'Remote receipt implementation differs from the approved plan')
        tool_names=[source_hashes[name][:16]+'-'+name for name in source_names]
        _require(_entries(folder/'tools',2)==sorted(tool_names),'Remote execution source inventory changed')
        for name,tool_name in zip(source_names,tool_names):
            pin=_file_pin(folder/'tools'/tool_name,MAX_JSON);records.append(pin)
            _require(pin['sha256']==source_hashes[name],'Remote execution source bytes changed')
        downloads=folder/'downloads';manifest=read(downloads/'manifest.json')
        _require(manifest.get('schema_version')==1 and manifest.get('kind')=='meshy_original_downloads'
                 and manifest.get('status')=='downloaded_not_visually_accepted'
                 and manifest.get('task_id')==call['task_id'] and manifest.get('stage')==call['step']
                 and manifest.get('plan_sha256')==plan_pin['sha256']
                 and isinstance(manifest.get('downloads'),dict) and manifest['downloads']==fetched.get('files'),
                 'Downloaded remote manifest differs from task/plan/receipt')
        expected={'model.glb'} | ({'thumbnail.png'} if response.get('thumbnail_url') else set())
        _require(isinstance(response.get('model_urls'),dict) and isinstance(response['model_urls'].get('glb'),str)
                 and set(manifest['downloads'])==expected
                 and set(_entries(downloads,3))==expected|{'manifest.json'},
                 'Remote download inventory is incomplete or has extra files')
        measured=[]
        for name in sorted(expected):
            pin=_file_pin(downloads/name,64*1024**2);download_bytes+=pin['bytes']
            _require(download_bytes<=256*1024**2,'Remote downloads exceed256MiB')
            record=manifest['downloads'][name]
            _require(isinstance(record,dict) and record.get('bytes')==pin['bytes'] and record.get('sha256')==pin['sha256'],
                     'Remote download bytes/hash changed')
            measured.append({'path':name,'bytes':pin['bytes'],'sha256':pin['sha256']});records.append(pin)
        _require(manifest.get('artifacts')==measured,'Remote artifact inventory differs from actual downloads')
        summaries.append({'stage':call['step'],'task_id':call['task_id'],'status':'SUCCEEDED',
                          'credits':credits,'downloads':measured})
    cost=read(output/'generation-cost.json')
    _require(cost=={'estimated_credits':35,'paid_submissions':2,'provider_reported_credits':35,
                   'reserved_credits':35,'usd_cost':None},'Pipeline generation cost receipt differs')
    return {'schema_version':1,'policy':'completed-text-generation-custody-v1','plan_id':plan['id'],
            'plan_digest':_digest(plan),'grant_credits':35,'approval_sha256':hashlib.sha256(grant['approval'].encode()).hexdigest(),
            'stages':summaries,'ledger':ledger_pin,'files':records,'receipt_bytes':receipt_bytes,
            'download_bytes':download_bytes,'new_provider_calls':0,'spent_credits_preserved':35,
            'scope':'Saved successful remote tasks and exact downloaded bytes; no live poll, refund, quality or replay approval.'}


def recheck_generation(proof):
    _require(proof.get('policy')=='completed-text-generation-custody-v1','Unknown remote custody proof')
    ledger=Path(proof['ledger']['path'])
    _require(not any(Path(str(ledger)+suffix).exists() for suffix in ('-wal','-journal','-shm')),
             'Canonical ledger journal appeared during inspection')
    for pin in [proof['ledger'],*proof['files']]:
        _require(_file_pin(Path(pin['path']),64*1024**2)==pin,'Remote evidence changed during inspection')


def collect_api_planning(root,output,request,binding_path):
    """Settle custody, not semantics: only one completed API planner on an explicit reused source is admitted."""
    from avatar_assets.agent_backend import validate_settings
    from avatar_assets.openai_runner import MODEL,PRICING,digest,reserve_microdollars
    from avatar_assets.studio_worker import _validate_prepared_request
    root,output,binding_path=map(Path,(root,output,binding_path))
    _require(output.resolve()==output and output.is_relative_to(root/'outputs')
             and request.get('reuse_model') is not None and request.get('reference') is None,
             'API planning release requires explicit source reuse without a reference')
    allowed={'request.json','input-receipt.json','estimate.json','status.json','pipeline.lock','sources','inputs','events','agents'}
    _require(set(_entries(output,16))==allowed,'API release cannot cover a plan, native output or generation work')
    records=[];total=0;started=time.monotonic()

    def read(path,limit=MAX_JSON):
        nonlocal total
        _require(time.monotonic()-started<15 and len(records)<160,'API custody inventory exceeds its bound')
        value,pin=_receipt(path,limit);total+=pin['bytes']
        _require(total<=8*1024**2,'API custody receipts exceed8MiB')
        records.append(pin)
        return value

    prepared=read(output/'request.json')
    backend=validate_settings(prepared.get('agent_backend'))
    _validate_prepared_request(prepared,request,output,agent_backend=backend)
    request_hash=records[-1]['sha256']
    binding=read(binding_path)
    _require(binding=={'schema_version':1,'studio_request_digest':_digest(request),'pipeline_request_sha256':request_hash},
             'API pipeline request differs from its private Studio binding')
    inputs=read(output/'input-receipt.json')
    _require(inputs.get('request_sha256')==request_hash and isinstance(inputs.get('sources'),list)
             and 1<=len(inputs['sources'])<=128,'API input receipt is missing its source binding')
    source_names=set();source_bytes=0;source_pins={}
    for row in inputs['sources']:
        _require(isinstance(row,dict) and set(row)=={'path','bytes','sha256'} and isinstance(row['path'],str)
                 and len(row['path'])<=200 and not Path(row['path']).is_absolute() and '..' not in Path(row['path']).parts
                 and Path(row['path']).suffix=='.py' and row['path'] not in source_names,'Invalid API source snapshot inventory')
        pin=_file_pin(output/'sources'/row['path'],MAX_JSON);source_bytes+=pin['bytes']
        _require(source_bytes<=16*1024**2 and {'path':row['path'],'bytes':pin['bytes'],'sha256':pin['sha256']}==row,
                 'API source snapshot changed')
        source_names.add(row['path']);source_pins[row['path']]=pin;records.append(pin)
    original=request['reuse_model'];selected=prepared['reuse_model']
    _require(set(_entries(output/'inputs',1))=={'source-model.glb'},'API selected input inventory changed')
    source=_file_pin(Path(original['path']),64*1024**2)
    copied=_file_pin(output/selected['path'],64*1024**2)
    info=Path(original['path']).stat()
    _require((info.st_size,info.st_mtime_ns,info.st_dev,info.st_ino)==
             (original['bytes'],original['mtime_ns'],original['device'],original['inode'])
             and source['sha256']==copied['sha256']==selected['sha256']
             and source['bytes']==copied['bytes']==selected['bytes'],'API reused source bytes or identity changed')
    records.extend((source,copied))
    status=read(output/'status.json')
    _require(status.get('status')=='checkpointed_error' and status.get('automatic_retry') is False,
             'API planning did not preserve a non-retrying failure')
    agents=output/'agents'
    _require(_entries(agents,3)==['planner','runner.json','runner.lock'],'Only one API planning call can be released here')
    settings=read(agents/'runner.json')
    expected={'schema_version':1,'execution_kind':'openai_responses_api','model':MODEL,
              'max_calls':backend['max_calls'],'timeout':backend['timeout'],'max_output_tokens':backend['max_output_tokens'],
              'job_id':output.name,'job_digest':request_hash,'automatic_retries':0,'tools':[],'store':False,
              'service_tier':'default','reasoning':{'effort':'none'},'truncation':'disabled','pricing':PRICING,
              'reservation_microdollars_per_call':reserve_microdollars(backend['max_output_tokens'])}
    _require(all(settings.get(k)==value for k,value in expected.items())
             and set(settings)==set(expected)|{'implementation_sha256','reservation_policy'}
             and 'openai_runner.py' in source_pins
             and settings['implementation_sha256']==source_pins['openai_runner.py']['sha256'],
             'API runner settings differ from the saved request or implementation')
    ledger=root/'.local/openai/ledger.sqlite3';ledger_pin=_file_pin(ledger,64*1024**2)
    _require(ledger.stat().st_uid==os.getuid() and stat.S_IMODE(ledger.stat().st_mode)==0o600,
             'Canonical API ledger is not owner-only')
    _require(not any(Path(str(ledger)+suffix).exists() for suffix in ('-wal','-journal','-shm')),
             'Canonical API ledger has an active journal')
    db=sqlite3.connect(ledger.as_uri()+'?mode=ro&immutable=1',uri=True,timeout=.2);db.row_factory=sqlite3.Row
    try:
        _require(db.execute('SELECT count(*) FROM calls').fetchone()[0]<=10000,'API ledger inventory exceeds bound')
        budget=db.execute('SELECT ceiling,approval FROM budget WHERE id=1').fetchone()
        grant=db.execute('SELECT digest,calls,ceiling,approval FROM jobs WHERE id=?',(output.name,)).fetchone()
        calls=[dict(row) for row in db.execute('SELECT * FROM calls WHERE job=? ORDER BY role',(output.name,))]
    finally:db.close()
    _require(_file_pin(ledger,64*1024**2)==ledger_pin,'API ledger changed during inspection')
    _require(budget is not None and grant is not None and grant['digest']==request_hash
             and type(grant['calls']) is int and grant['calls']==backend['max_calls']
             and type(budget['ceiling']) is int and type(grant['ceiling']) is int
             and 1<=grant['ceiling']<=budget['ceiling']<=1000000000,'API allowance differs from the exact request')
    approvals=[]
    for row in (budget,grant):
        _require(isinstance(row['approval'],str) and len(row['approval'].encode())<=8192,'Invalid API spending approval')
        approval=json.loads(row['approval'])
        _require(isinstance(approval,dict) and approval.get('decision')=='approved'
                 and isinstance(approval.get('user_reply'),str) and 0<len(approval['user_reply'])<=4000,
                 'API spending approval is missing')
        approvals.append(hashlib.sha256(row['approval'].encode()).hexdigest())
    _require(len(calls)==1 and calls[0]['role']=='planner' and calls[0]['state']=='settled',
             'API calls are uncertain, missing or outside this completed planning scope')
    call=calls[0]
    _require(type(call['reserved']) is int and call['reserved']==settings['reservation_microdollars_per_call']
             and type(call['actual']) is int and 0<=call['actual']<=call['reserved']<=grant['ceiling'],
             'API settled cost or reservation differs')
    folder=agents/'planner';names=['answer.json','payload-metadata.json','prompt.txt','receipt.json','request.json','response.json','schema.json']
    _require(_entries(folder,7)==names,'API planner files are missing, extra or unsafe')
    receipt=read(folder/'receipt.json');call_request=read(folder/'request.json');schema=read(folder/'schema.json',64*1024)
    payload=read(folder/'payload-metadata.json');answer=read(folder/'answer.json');response=read(folder/'response.json',4*1024**2)
    prompt_pin=_file_pin(folder/'prompt.txt',128*1024);records.append(prompt_pin)
    artifact_rows=[{'path':name,'bytes':pin['bytes'],'sha256':pin['sha256']} for name in names if name!='receipt.json'
                   for pin in records if pin['path']==str(folder/name)]
    _require(receipt.get('schema_version')==1 and receipt.get('status')=='succeeded'
             and receipt.get('execution_kind')=='openai_responses_api' and receipt.get('model')==MODEL
             and receipt.get('tool_activity') is False and receipt.get('artifacts')==artifact_rows
             and receipt.get('answer_sha256')==next(p['sha256'] for p in records if p['path']==str(folder/'answer.json')),
             'API response receipt or artifact hashes changed')
    _require(call_request=={'role':'planner','prompt_sha256':prompt_pin['sha256'],'schema_sha256':digest(schema),
                           'settings_sha256':digest(settings),'images':[]}
             and call['request_digest']==digest(call_request),'API call request does not match its reservation')
    _require(response.get('status')=='completed' and response.get('model')==MODEL
             and response.get('service_tier')=='default' and isinstance(response.get('id'),str)
             and response['id'].startswith('resp_') and len(response['id'])<=256
             and response['id']==call['response_id']==receipt.get('response_id'),
             'API completed response identity changed')
    _require(isinstance(call['usage'],str) and len(call['usage'].encode())<=16384,'API usage exceeds its bound')
    usage=response.get('usage');_require(isinstance(usage,dict) and usage==json.loads(call['usage'])==receipt.get('usage'),
                                        'API usage differs from settled accounting')
    inp,out=usage.get('input_tokens'),usage.get('output_tokens')
    _require(type(inp) is int and 0<=inp<=1050000 and type(out) is int and 0<=out<=backend['max_output_tokens']
             and type(usage.get('total_tokens')) is int and usage['total_tokens']==inp+out,'Invalid settled API token counts')
    amount=math.ceil(inp*(5 if inp>272000 else 2.5)+out*(22.5 if inp>272000 else 15))
    _require(call['actual']==receipt.get('actual_microdollars_upper_estimate')==amount
             and receipt.get('reserved_microdollars')==call['reserved'],'API settled token cost differs')
    messages=response.get('output')
    _require(isinstance(messages,list) and len(messages)==1 and isinstance(messages[0],dict)
             and messages[0].get('type')=='message' and messages[0].get('role')=='assistant'
             and messages[0].get('status')=='completed','API response contains unowned tool activity')
    content=messages[0].get('content')
    _require(isinstance(content,list) and len(content)==1 and isinstance(content[0],dict)
             and content[0].get('type')=='output_text' and isinstance(content[0].get('text'),str)
             and len(content[0]['text'].encode())<=MAX_JSON and json.loads(content[0]['text'])==answer,
             'API preserved answer differs from the completed response')
    prompt=(folder/'prompt.txt').read_text()
    actual_payload={'model':MODEL,'instructions':payload.get('instructions'),'input':[{'role':'user','content':[{'type':'input_text','text':prompt}]}],
                    'text':{'format':{'type':'json_schema','name':'avatar_planner','schema':schema,'strict':True}},
                    'max_output_tokens':backend['max_output_tokens'],'reasoning':{'effort':'none'},'store':False,'tools':[],
                    'tool_choice':'none','service_tier':'default','truncation':'disabled'}
    encoded=json.dumps(actual_payload,sort_keys=True,ensure_ascii=False,allow_nan=False,separators=(',',':')).encode()
    _require(payload.get('endpoint')=='https://api.openai.com/v1/responses' and payload.get('sha256')==digest(actual_payload)
             and payload.get('bytes')==len(encoded),'API payload metadata differs from saved inputs')
    return {'schema_version':1,'policy':'completed-api-planning-custody-v1','job_id':output.name,
            'request_sha256':request_hash,'ledger':ledger_pin,'files':records,'receipt_bytes':total,
            'approval_sha256':approvals,'calls':[{'role':'planner','state':'settled','response_id':call['response_id'],
                'reserved_microdollars':call['reserved'],'actual_microdollars':amount,'usage':usage}],
            'call_ceiling_preserved':grant['calls'],'dollar_ceiling_microdollars_preserved':grant['ceiling'],
            'new_provider_calls':0,'scope':'Completed API transport/accounting only; rejected semantics are preserved, not accepted or retried.'}


def recheck_api_planning(proof):
    _require(proof.get('policy')=='completed-api-planning-custody-v1','Unknown API custody proof')
    ledger=Path(proof['ledger']['path'])
    _require(not any(Path(str(ledger)+suffix).exists() for suffix in ('-wal','-journal','-shm')),
             'API ledger journal appeared during inspection')
    for pin in [proof['ledger'],*proof['files']]:
        _require(_file_pin(Path(pin['path']),64*1024**2)==pin,'API custody evidence changed during inspection')
