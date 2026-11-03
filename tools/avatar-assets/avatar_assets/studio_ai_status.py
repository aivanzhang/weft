"""Small read-only stopped-job summaries; never reconcile charges or expose provider output."""
from contextlib import closing
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sqlite3
import stat
import time

from avatar_assets.safety import require

STOPPED = {'needs_inspection', 'failed', 'unsupported'}
UNVERIFIED = {'code': 'ai_status_unverified', 'verified': False, 'automatic_retry': False,
              'message': 'AI call records could not be verified. Saved work is preserved; no automatic retry will start.',
              'charge': {'status': 'unverified', 'currency': 'USD'}}


def _regular(path, maximum, *, private=False):
    require(path.resolve()==path and not path.is_symlink(), 'Unsafe status evidence path')
    info=path.stat()
    require(stat.S_ISREG(info.st_mode) and 0<info.st_size<=maximum, 'Status evidence exceeds bounds')
    if private:
        require(info.st_uid==os.getuid() and stat.S_IMODE(info.st_mode)==0o600,'Ledger is not owner-only')
    return info


def _read(path, maximum):
    _regular(path,maximum)
    with os.fdopen(os.open(path,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK),'rb') as stream:
        raw=stream.read(maximum+1)
    require(len(raw)<=maximum,'Status evidence grew beyond bounds')
    value=json.loads(raw,parse_constant=lambda _: (_ for _ in ()).throw(ValueError('Nonfinite status data')))
    require(isinstance(value,dict),'Status evidence must be an object')
    return value,{'path':path.name,'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}


def failure_snapshot(root,folder,binding_path,studio_request_digest,*,status,operation):
    """Exact request/ledger/receipt binding, not source scans or a new execution decision."""
    if status not in STOPPED:return None
    root,folder,binding_path=map(Path,(root,folder,binding_path))
    try:
        if not (folder/'request.json').exists() and not (folder/'request.json').is_symlink():return None
        prepared,request_pin=_read(folder/'request.json',64*1024)
        if prepared.get('agent_backend') is None:return None
        require(folder.parent==root/'outputs' and folder.resolve()==folder,'Invalid job output')
        binding,_=_read(binding_path,4096)
        require(binding=={'schema_version':1,'studio_request_digest':studio_request_digest,
                          'pipeline_request_sha256':request_pin['sha256']},'Prepared request is unbound')
        inputs,_=_read(folder/'input-receipt.json',64*1024)
        require(inputs.get('request_sha256')==request_pin['sha256'],'Input request digest differs')
        cap=prepared.get('max_agent_calls')
        require(type(cap) is int and 3<=cap<=12 and prepared['agent_backend'].get('provider')=='openai'
                and prepared['agent_backend'].get('max_calls')==cap,'Invalid API call bound')
        ledger=root/'.local/openai/ledger.sqlite3';info=_regular(ledger,64*1024**2,private=True)
        with closing(sqlite3.connect(ledger.as_uri()+'?mode=ro',uri=True,timeout=.05)) as db:
            db.row_factory=sqlite3.Row;db.execute('PRAGMA query_only=ON')
            deadline=time.monotonic()+.05;db.set_progress_handler(lambda: int(time.monotonic()>deadline),1000)
            db.execute('BEGIN')
            grant=db.execute('SELECT digest,calls FROM jobs WHERE id=?',(folder.name,)).fetchone()
            sizes=db.execute('SELECT length(CAST(role AS BLOB)),length(CAST(request_digest AS BLOB)),length(CAST(response_id AS BLOB)),length(CAST(usage AS BLOB)) FROM calls WHERE job=? LIMIT 13',(folder.name,)).fetchall()
            require(len(sizes)<=cap and all(all(size is None or type(size) is int and size<=limit
                    for size,limit in zip(row,(32,64,256,16384))) for row in sizes),'API row exceeds display bound')
            calls=[dict(row) for row in db.execute('SELECT role,request_digest,reserved,state,response_id,actual,usage FROM calls WHERE job=? LIMIT 13',(folder.name,))]
        after=ledger.stat()
        require((info.st_dev,info.st_ino,info.st_size,info.st_mtime_ns)==
                (after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns),'Ledger changed while reading')
        require(grant is not None and grant['digest']==request_pin['sha256'] and grant['calls']==cap
                and len(calls)<=cap,'API grant/call scope differs')
        if not calls:return None
        def order(row):
            role=row['role']
            require(isinstance(role,str) and re.fullmatch(r'planner|(?:author|reviewer)-[0-2]',role),'Unknown AI call role')
            return 0 if role=='planner' else 1+int(role[-1])*2+int(role.startswith('reviewer'))
        calls.sort(key=order);call=calls[-1]
        uncertain=[row for row in calls if row['state']=='reserved_uncertain']
        require(not uncertain or uncertain==[call],'Multiple or nonterminal uncertain calls need inspection')
        from avatar_assets.openai_runner import digest,call_exposure
        role=call['role'];agent=folder/'agents'/role
        settings,_=_read(folder/'agents/runner.json',16*1024)
        require(settings.get('job_id')==folder.name and settings.get('job_digest')==request_pin['sha256']
                and settings.get('max_calls')==cap,'Runner request binding differs')
        request,call_pin=_read(agent/'request.json',16*1024)
        receipt,_=_read(agent/'receipt.json',64*1024)
        require(request.get('role')==role and request.get('settings_sha256')==digest(settings)
                and digest(request)==call['request_digest'],'AI call request differs')
        artifacts=receipt.get('artifacts')
        require(isinstance(artifacts,list) and len(artifacts)<=32
                and [row for row in artifacts if row.get('path')=='request.json']==[call_pin]
                and receipt.get('execution_kind')=='openai_responses_api'
                and receipt.get('reserved_microdollars')==call['reserved'],'AI call receipt binding differs')
        cost=call_exposure(call)
        base={'verified':True,'role':role,'automatic_retry':False,'source':'exact_job_ledger_and_receipt'}
        if call['state']=='reserved_uncertain':
            require(receipt.get('status')=='failed_or_uncertain' and receipt.get('automatic_resubmission_allowed') is False
                    and call['actual'] is None and call['usage'] is None,'Uncertain receipt differs from accounting')
            elapsed=receipt.get('elapsed_seconds');duration=''
            if type(elapsed) in (int,float) and math.isfinite(elapsed) and 0<=elapsed<=3600:duration=f' after {round(elapsed)} seconds'
            return {**base,'code':'ai_charge_unknown',
                    'message':f'AI request stopped{duration} without a verified result. Its charge is unknown; ${cost/1000000:.6f} remains reserved. Saved work is preserved; no automatic retry will start.',
                    'charge':{'status':'unknown','reserved_microdollars':cost,'actual_microdollars':None,'currency':'USD'}}
        charge={'status':'settled','actual_microdollars':cost,'currency':'USD'}
        if receipt.get('status')=='failed_or_uncertain':
            require(receipt.get('automatic_resubmission_allowed') is False,'Invalid failed-response receipt')
            return {**base,'code':'ai_response_unusable','charge':charge,
                    'message':'AI returned a response, but it could not be used. Its usage cost is recorded, not unknown. Saved work is preserved; no automatic retry will start.'}
        require(receipt.get('status')=='succeeded' and receipt.get('response_id')==call['response_id']
                and receipt.get('actual_microdollars_upper_estimate')==cost
                and receipt.get('usage')==json.loads(call['usage']),'Settled receipt differs from accounting')
        answer_rows=[row for row in artifacts if row.get('path')=='answer.json']
        require(len(answer_rows)==1,'Successful AI answer has no artifact binding')
        if answer_rows[0]['bytes']<=64*1024:
            answer,pin=_read(agent/'answer.json',64*1024)
            require(pin==answer_rows[0] and pin['sha256']==receipt.get('answer_sha256'),'AI answer changed')
            if answer.get('status')=='unsupported' or answer.get('verdict')=='unsupported':
                return {**base,'code':'ai_declined','charge':charge,
                        'message':'AI could not provide a supported result for this request. Its usage cost is recorded. Saved work is preserved; no automatic retry will start.'}
        if role=='planner' and operation=='plan' and not (folder/'plan.json').exists():
            require(not (folder/'plan.json').is_symlink(),'Unsafe plan path')
            return {**base,'code':'ai_plan_incomplete','charge':charge,
                    'message':'AI planning stopped after a received response without a usable plan. Its usage cost is recorded. Saved work is preserved; no automatic retry will start.'}
        return None
    except (OSError,ValueError,TypeError,KeyError,AttributeError,sqlite3.Error,OverflowError,RecursionError):
        return dict(UNVERIFIED)
