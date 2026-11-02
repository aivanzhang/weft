"""Owned pipeline child; standing grants only derive from an existing owner policy."""

import argparse
from contextlib import contextmanager
import json
import hashlib
import fcntl
import os
from pathlib import Path
import signal
import sys
import threading
import time

from avatar_assets.studio_jobs import (ROOT, StudioJobs, _digest, _json, _private_directory,
    _require, _safe_text, _write_private, process_identity)


WALL_SECONDS = 5400
CONTROLLER_RSS_MB = 1024
CONTROL_OUTPUT_MB = 64
PIPELINE_OUTPUT_BYTES = 2*1024**3
PIPELINE_LIMITS = {'max_repairs':1,'max_agent_calls':5,'agent_timeout':240,'agent_model':None}
MEDIA_SPECS = {
    'draft':{'size':256,'portrait':512,'seconds':2,'samples':4},
    'preview':{'size':720,'portrait':1600,'seconds':4,'samples':8},
    'muse':{'size':720,'portrait':1600,'seconds':8,'samples':8},
}


@contextmanager
def _owner_lock(path):
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd,'a') as stream:
        deadline = time.monotonic()+2
        while True:
            try:
                fcntl.flock(stream,fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic()>deadline:
                    raise ValueError('Another process still holds the studio worker lock') from None
                time.sleep(.05)
        yield


def _current(jobs,job_id,token):
    with jobs._db() as db:
        row=jobs._row(db,job_id);data=json.loads(row['data'])
        active=db.execute('SELECT job,token FROM active WHERE id=1').fetchone()
    _require(active and tuple(active)==(job_id,token) and data.get('token')==token,'Studio worker lease changed')
    _require(row['status'] in ('planning','running','cancel_requested'),'Studio supervisor has not claimed this invocation')
    request=json.loads(row['request'])
    _require(_digest(request)==row['request_digest'],'Studio immutable request changed')
    _require(_json(jobs.jobs_dir/job_id/'request.json')==request,'Studio request receipt changed')
    invocation=jobs.state_dir/data['invocation_dir'];receipt=_json(invocation/'invocation.json')
    _require(receipt.get('token')==token and receipt.get('request_digest')==row['request_digest'],'Studio invocation receipt changed')
    for source in receipt['sources']:
        saved=invocation/source['path'];live=Path(__file__).parent/Path(source['path']).name
        _require(hashlib.sha256(saved.read_bytes()).hexdigest()==source['sha256']
                 and hashlib.sha256(live.read_bytes()).hexdigest()==source['sha256'],
                 'Studio worker backend changed during this invocation; preserve it and inspect before resuming')
    return row,request,data


def _validate_prepared_request(prepared,request,output,*,agent_backend=None):
    # A self-consistent pipeline receipt does not bind it to the request shown in Studio.
    limits = PIPELINE_LIMITS
    if agent_backend is not None:
        from avatar_assets.agent_backend import pipeline_limits
        limits = pipeline_limits(agent_backend)
    expected={'schema_version':1,'id':output.name,'prompt':request['prompt'],'style':request['style'],
              'custom_actions':request['custom_actions'],'states':[],'preset':request['preset'],
              'media_spec':MEDIA_SPECS[request['preset']],
              'max_credits':0 if request['reuse_model'] else 35,**limits}
    if agent_backend is not None:
        expected['agent_backend'] = agent_backend
    extra={'continuation'} if request.get('continuation_from') else set()
    _require(isinstance(prepared,dict) and set(prepared)==set(expected)|{'reference','reuse_model','created'}|extra,
             'Prepared pipeline request fields differ from the Studio contract')
    for field,value in expected.items():
        _require(_digest(prepared[field])==_digest(value),
                 'Prepared pipeline request differs from the immutable Studio request: '+field)
    if extra:
        from avatar_assets.pipeline_continuation import verify
        _require(prepared['continuation'].get('parent')=='outputs/studio-'+request['continuation_from'],
                 'Prepared continuation parent differs from Studio')
        verify(output.parents[1],output,prepared)
    for field,name in (('reference','reference'),('reuse_model','source-model')):
        selected,actual=request[field],prepared[field]
        if selected is None:
            _require(actual is None,'Prepared pipeline selection differs from Studio: '+field)
            continue
        _require(isinstance(actual,dict) and set(actual)=={'path','original_path','bytes','sha256'},
                 'Prepared pipeline selection fields differ from Studio: '+field)
        expected_path='inputs/'+name+Path(selected['path']).suffix.lower()
        _require(actual['original_path']==selected['path'] and actual['path']==expected_path
                 and type(actual['bytes']) is int and actual['bytes']==selected['bytes']
                 and isinstance(actual['sha256'],str) and len(actual['sha256'])==64
                 and all(c in '0123456789abcdef' for c in actual['sha256']),
                 'Prepared pipeline selection differs from Studio: '+field)


def run_child(jobs,job_id,token,*,pipeline_module=None):
    """Injection is for explicit analytic fixtures; CLI always imports the real pipeline."""
    if pipeline_module is None:
        from avatar_assets import pipeline as pipeline_module
        provenance='actual_pipeline'
    else:
        provenance='analytic_fixture'
    row,request,data=_current(jobs,job_id,token)
    backend = None
    limits = PIPELINE_LIMITS
    if provenance == 'actual_pipeline':
        from avatar_assets.agent_backend import server_settings, pipeline_limits
        backend = server_settings(jobs.root)
        limits = {**pipeline_limits(backend), 'agent_backend': backend}
    invocation=jobs.state_dir/data['invocation_dir']
    cancel_path=invocation/'cancel.json'

    def cancelled():
        if not cancel_path.exists():return False
        value=_json(cancel_path)
        _require(value.get('token')==token,'Cancellation receipt belongs to a different invocation')
        return True

    output=jobs.resolve_job(job_id)
    if cancelled() and not (output/'delivery.json').exists():
        return {'status':'cancelled','safe_to_release':True,'provenance':provenance}
    def verify_selected():
        for selected in (request['reference'],request['reuse_model']):
            if selected:
                path=Path(selected['path'])
                _require(not any(p.is_symlink() for p in (path,*path.parents)),'Selected input path changed')
                info=path.stat()
                _require((info.st_size,info.st_mtime_ns,info.st_dev,info.st_ino)==
                         (selected['bytes'],selected['mtime_ns'],selected['device'],selected['inode']),
                         'Selected input changed during worker preparation; prepare a new immutable request')
    if not output.exists():
        verify_selected()
        pipeline_module.prepare(request['prompt'],output,style=request['style'],custom_actions=request['custom_actions'],
            preset=request['preset'],reference=request['reference']['path'] if request['reference'] else None,
            reuse_model=request['reuse_model']['path'] if request['reuse_model'] else None,
            states=[],max_credits=0 if request['reuse_model'] else 35,**limits,
            **({'continuation_from':jobs.resolve_job(request['continuation_from'])}
               if request.get('continuation_from') else {}))
        verify_selected()
    prepared=_json(output/'request.json')
    _validate_prepared_request(prepared,request,output,agent_backend=backend)
    request_sha=hashlib.sha256((output/'request.json').read_bytes()).hexdigest()
    binding={'schema_version':1,'studio_request_digest':row['request_digest'],'pipeline_request_sha256':request_sha}
    binding_path=jobs.jobs_dir/job_id/'pipeline-request.json'
    if binding_path.exists():
        _require(_json(binding_path)==binding,'Prepared pipeline request changed after its Studio binding')
    if not binding_path.exists():
        _write_private(binding_path,binding)
    if backend is not None:
        from avatar_assets.agent_backend import readiness
        if request.get('continuation_from'):
            try:
                jobs.authorize_retry(job_id,token)
            except ValueError as error:
                return {'status':'awaiting_authorization','safe_to_release':True,'provenance':provenance,
                        'reason':_safe_text(str(error),800),'pipeline_request_sha256':request_sha}
        ready = readiness(jobs.root, job_id=output.name, job_digest=request_sha, settings=backend)
        if not ready['can_run']:
            return {'status':'awaiting_authorization','safe_to_release':True,'provenance':provenance,
                    'reason':ready['reason'],'pipeline_request_sha256':request_sha}
    if data['operation']=='execute':
        _require(data.get('expected_plan_id')==jobs._plan_id(row),'Prepared plan changed after Start was requested')
        authorization=jobs._authorization(row)
        if not authorization['can_start']:
            return {'status':'awaiting_authorization','safe_to_release':True,'provenance':provenance,
                    'reason':authorization['reason']}
    pipeline=pipeline_module.Pipeline(output)
    _require(_digest(pipeline.request)==_digest(prepared)
             and hashlib.sha256((output/'request.json').read_bytes()).hexdigest()==request_sha,
             'Prepared pipeline request changed during Studio validation')
    result=pipeline.run(execute=data['operation']=='execute',
        plan_only=data['operation']=='plan',cancel_requested=cancelled)
    _require(isinstance(result,dict),'Pipeline returned no structured result')
    status=result.get('status')
    if status in ('awaiting_generation_authorization','awaiting_authorization'):
        status='awaiting_authorization'
    elif status in ('planned','prepared','prepared_no_paid_call','plan_ready'):
        status='planned'
    _require(status in ('planned','awaiting_authorization','complete','unsupported','cancelled','checkpointed'),
             'Unknown pipeline status requires inspection')
    return {'status':status,'safe_to_release':True,'provenance':provenance,
            'pipeline_status':result.get('status'),'reason':_safe_text(result.get('reason'),800),
            'plan_id':jobs._plan_id(row)}


def supervise(jobs,job_id,token):
    from avatar_assets.runner import _output_bytes,run_bounded
    with _owner_lock(jobs.state_dir/'worker.lock'), _owner_lock(jobs.jobs_dir/job_id/'worker.lock'):
        row,request,data=jobs._worker_claim(job_id,token,os.getpid(),process_identity(os.getpid()))
        invocation=jobs.state_dir/data['invocation_dir']
        _write_private(invocation/'owner.json',{'pid':os.getpid(),'start_identity':process_identity(os.getpid()),
                                               'token':token,'claimed_at':time.time()})
        done=threading.Event();monitor_errors=[];output_limit_requested=[]

        def heartbeat():
            while not done.wait(2):
                try:
                    output=jobs.resolve_job(job_id)
                    output_size=_output_bytes(output) if output.exists() else 0
                    jobs._heartbeat(job_id,token,retained_bytes=output_size+_output_bytes(jobs.jobs_dir/job_id),output_bytes=output_size)
                    if output_size>PIPELINE_OUTPUT_BYTES:
                        if not output_limit_requested:
                            output_limit_requested.append(time.time())
                            jobs.cancel_cooperative(job_id)
                except Exception as error:
                    if len(monitor_errors)<8:monitor_errors.append(type(error).__name__)

        thread=threading.Thread(target=heartbeat,name='studio-heartbeat',daemon=True)
        thread.start()
        command=[sys.executable,'-m','avatar_assets.studio_worker','--root',str(jobs.root),
                 '--state-dir',str(jobs.state_dir),'--job',job_id,'--token',token,'--child']
        try:
            report=run_bounded(command,invocation/'worker.log',invocation,timeout=WALL_SECONDS,
                               memory_mb=CONTROLLER_RSS_MB,output_mb=CONTROL_OUTPUT_MB)
            _write_private(invocation/'worker-job.json',report)
            result=_json(invocation/'result.json')
            safe=(report.get('exit_code')==0 and not report.get('failure') and isinstance(result,dict)
                  and result.get('safe_to_release') is True)
            status=result['status'] if safe else 'needs_inspection'
            error=(result.get('reason') if safe else 'Worker did not seal a safe terminal receipt; inspect preserved controller/native/provider handles before resuming.')
            if output_limit_requested and status!='complete':
                status='needs_inspection';safe=False
                error='Aggregate output exceeded its bound; cooperative stop requested and evidence preserved.'
            _write_private(invocation/'supervisor-result.json',{'status':status,'safe_to_release':safe,
                'monitor_errors':monitor_errors,'output_bound_cancellation_requested':bool(output_limit_requested),
                'scope':'Controller RSS is bounded separately; native/agent children retain their own existing bounds. A hard supervisor failure leaves ownership uncertain, never automatically replayed.'})
            output=jobs.resolve_job(job_id)
            output_size=_output_bytes(output) if output.exists() else 0
            jobs._heartbeat(job_id,token,retained_bytes=output_size+_output_bytes(jobs.jobs_dir/job_id),output_bytes=output_size)
            jobs._worker_complete(job_id,token,status,safe_to_release=safe,error=error)
        except BaseException as error:
            try:
                jobs._worker_complete(job_id,token,'needs_inspection',safe_to_release=False,
                                      error='Supervisor failed; inspect preserved worker and native handles. '+type(error).__name__)
            except Exception:
                pass
            raise
        finally:
            done.set();thread.join(timeout=3)
    jobs._dispatch()


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',required=True)
    parser.add_argument('--state-dir',required=True)
    parser.add_argument('--job',required=True)
    parser.add_argument('--token',required=True)
    parser.add_argument('--child',action='store_true')
    args=parser.parse_args(argv)
    _require(Path(args.root).resolve()==ROOT,'Worker root must identify this installed avatar tool')
    _require(len(args.token)==32 and all(c in '0123456789abcdef' for c in args.token),'Invalid worker token')
    jobs=StudioJobs(root=args.root,state_dir=args.state_dir)
    if not args.child:
        supervise(jobs,args.job,args.token)
        return
    _,_,data=_current(jobs,args.job,args.token)
    result_path=jobs.state_dir/data['invocation_dir']/'result.json'
    try:
        result=run_child(jobs,args.job,args.token)
    except BaseException as error:
        _write_private(result_path,{'status':'needs_inspection','safe_to_release':False,
            'error_type':type(error).__name__,'reason':_safe_text(str(error),800),'provenance':'actual_pipeline'})
        raise
    _write_private(result_path,result)


if __name__=='__main__':
    main()
