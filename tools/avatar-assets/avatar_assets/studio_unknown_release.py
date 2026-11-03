"""Release only exited local ownership; unknown provider charges remain reserved."""

from contextlib import closing
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sqlite3
import stat

from avatar_assets.studio_jobs import _digest, _require
from avatar_assets.studio_inspection import _receipt, _file_pin, _entries, recheck

POLICY = 'unknown-api-author-custody-v1'
TRANSITION = 'release_local_owner_preserve_uncertain_charge'
MAX_RELEASE_BYTES = 512 * 1024


def _private(path, limit):
    pin = _file_pin(path, limit)
    info = path.stat()
    _require(info.st_uid == os.getuid() and stat.S_IMODE(info.st_mode) == 0o600,
             'Custody state must be owner-only')
    return pin


def _ledger(root, output):
    from avatar_assets.openai_grant_closure import _inventory, _grant
    path = root/'.local/openai/ledger.sqlite3'
    pin = _private(path, 64*1024**2)
    _require(not any(Path(str(path)+suffix).exists() for suffix in ('-wal','-shm','-journal')),
             'Unknown-call ledger has an active journal')
    with closing(sqlite3.connect(path.as_uri()+'?mode=ro&immutable=1', uri=True, timeout=.2)) as db:
        db.row_factory = sqlite3.Row
        grant = db.execute('SELECT * FROM jobs WHERE id=?', (output.name,)).fetchone()
        _require(grant is not None, 'Unknown-call grant is missing')
        _require(not db.execute('SELECT 1 FROM retirements WHERE parent=?', (output.name,)).fetchone(),
                 'Unknown-call job cannot be retired')
        if db.execute("SELECT 1 FROM sqlite_master WHERE name='grant_closures'").fetchone():
            _require(not db.execute('SELECT 1 FROM grant_closures WHERE parent=?',(output.name,)).fetchone(),
                     'Unknown-call grant is already closed')
        calls = _inventory(db, output.name)
        result = {'ledger':pin, 'grant':_grant(dict(grant)), 'call_rows':calls}
    _require(_file_pin(path,64*1024**2)==pin,'Unknown-call ledger changed during inspection')
    return result


def _failed_stream_trace(author, call, settings, status):
    from avatar_assets.openai_runner import stream_client_request_id
    trace,pin=_receipt(author/'transport.json',4096)
    base={key:value for key,value in call.items() if key!='client_request_id'}
    expected=stream_client_request_id(settings['job_id'],'author-0',base)
    _require(set(trace)=={'schema_version','transport','status','client_request_id','server_request_id',
                         'response_id','wire_bytes','events','last_sequence','elapsed_seconds','error_code'}
             and trace.get('schema_version')==1 and trace.get('transport')=='sse-v1'
             and trace.get('status')=='failed_or_uncertain' and trace.get('error_code')=='unsupported_event'
             and call.get('client_request_id')==trace.get('client_request_id')==expected
             and status.get('error')=='OpenAI stream unsupported_event; reservation retained, no automatic retry',
             'Unknown stream trace differs from its failed request checkpoint')
    _require(type(trace['wire_bytes']) is int and 0<=trace['wire_bytes']<=16*1024**2
             and type(trace['events']) is int and 1<=trace['events']<=65536
             and type(trace['last_sequence']) is int and 0<=trace['last_sequence']<=65536
             and type(trace['elapsed_seconds']) in (int,float) and math.isfinite(trace['elapsed_seconds'])
             and 0<=trace['elapsed_seconds']<=settings['timeout']+30,
             'Unknown stream trace counters exceed bounds')
    for field,pattern in (('response_id',r'resp_[A-Za-z0-9_-]{1,250}'),
                          ('server_request_id',r'[A-Za-z0-9_.:-]{1,512}')):
        _require(trace[field] is None or isinstance(trace[field],str) and re.fullmatch(pattern,trace[field]),
                 'Unknown stream trace identity is invalid')
    return {'trace_sha256':pin['sha256'],'client_request_id':expected,
            'observed_response_id':trace['response_id'],'error_code':trace['error_code'],
            'response_id_is_not_settlement':True}


def collect_unknown(root, output, request, binding_path):
    """Authenticate one settled planner and one unanswered author; no provider requests."""
    from avatar_assets.agent_backend import validate_settings
    from avatar_assets.agent_runner import _call_artifacts
    from avatar_assets.openai_runner import digest, call_exposure, call_model, reserve_microdollars
    from avatar_assets.pipeline_continuation import _inventory, _role
    from avatar_assets.studio_worker import _validate_prepared_request
    from avatar_assets.safety import artifacts
    root,output,binding_path=map(Path,(root,output,binding_path))
    _require(output.resolve()==output and output.parent==root/'outputs' and request.get('reuse_model')
             and not request.get('continuation_from'),'Unknown-call release requires fresh explicit local source reuse')
    allowed={'request.json','input-receipt.json','estimate.json','status.json','pipeline.lock','sources','inputs',
             'events','agents','plan.json','inspection','quality-policy.json'}
    _require(set(_entries(output,16))==allowed,'Unknown-call release cannot cover extra or remote generation work')
    inventory=_inventory(output)
    _require(len(inventory)<=1024 and sum(row['bytes'] for row in inventory)<=512*1024**2,
             'Unknown-call custody inventory exceeds its bound')
    prepared,request_pin=_receipt(output/'request.json')
    backend=validate_settings(prepared.get('agent_backend'))
    streaming=backend.get('transport')=='sse-v1'
    _require(prepared.get('max_credits')==0 and not prepared.get('continuation'),
             'Unknown-call release covers fresh source reuse only')
    _validate_prepared_request(prepared,request,output,agent_backend=backend)
    binding,binding_pin=_receipt(binding_path)
    _require(binding=={'schema_version':1,'studio_request_digest':_digest(request),
                      'pipeline_request_sha256':request_pin['sha256']},'Unknown-call private request binding changed')
    inputs,_=_receipt(output/'input-receipt.json')
    _require(inputs.get('request_sha256')==request_pin['sha256'] and isinstance(inputs.get('sources'),list)
             and 1<=len(inputs['sources'])<=128,'Unknown-call source receipt changed')
    _require(artifacts(output/'sources')==inputs['sources'],'Unknown-call source snapshots changed')
    source_pins={row['path']:row for row in inputs['sources']}
    pins=[binding_pin];identities=[]
    for field in ('reuse_model','reference'):
        original,selected=request.get(field),prepared.get(field)
        _require((original is None)==(selected is None),'Unknown-call input selection changed')
        if original is None:continue
        path=Path(original['path']);copied=output/selected['path']
        _require(copied.parent==output/'inputs' and copied.resolve()==copied,'Unknown-call input escaped snapshot')
        a,b=_file_pin(path,64*1024**2),_file_pin(copied,64*1024**2)
        info=path.stat();identity={key:original[key] for key in ('bytes','mtime_ns','device','inode')}
        _require((info.st_size,info.st_mtime_ns,info.st_dev,info.st_ino)==tuple(identity.values())
                 and a['sha256']==b['sha256']==selected['sha256'] and a['bytes']==b['bytes']==selected['bytes'],
                 'Unknown-call original input identity or bytes changed')
        pins.append(a);identities.append({'path':str(path),**identity})
    status,_=_receipt(output/'status.json')
    _require(status.get('status')=='checkpointed_error' and status.get('stage')=='pipeline'
             and status.get('automatic_retry') is False
             and status.get('error')==('OpenAI stream unsupported_event; reservation retained, no automatic retry' if streaming
                 else 'OpenAI transport failed; reservation retained, no automatic retry'),
             'Unknown-call release requires the exact non-retrying transport checkpoint')
    agents=output/'agents'
    _require(_entries(agents,4)==['author-0','planner','runner.json','runner.lock'],
             'Unknown-call release covers exactly one planner and first author')
    settings,_=_receipt(agents/'runner.json')
    _require(settings.get('execution_kind')=='openai_responses_api' and settings.get('job_id')==output.name
             and settings.get('job_digest')==request_pin['sha256'] and settings.get('model')==backend['model']
             and settings.get('max_calls')==backend['max_calls'] and settings.get('timeout')==backend['timeout']
             and settings.get('max_output_tokens')==backend['max_output_tokens']
             and settings.get('reasoning')=={'effort':backend.get('reasoning_effort','none')}
             and settings.get('automatic_retries')==0 and settings.get('store') is False and settings.get('tools')==[]
             and settings.get('implementation_sha256')==source_pins.get('openai_runner.py',{}).get('sha256'),
             'Unknown-call runner settings differ from immutable request/source')
    if streaming:
        from avatar_assets.openai_models import price_profile
        _require(settings.get('transport')=='sse-v1'
                 and 'openai_stream.py' in source_pins and 'openai_models.py' in source_pins
                 and settings.get('transport_implementation_sha256')==source_pins['openai_stream.py']['sha256']
                 and settings.get('model_profile_implementation_sha256')==source_pins['openai_models.py']['sha256']
                 and settings.get('pricing')==price_profile(backend['model']),
                 'Unknown stream transport or model profile source pin changed')
    else:
        _require('transport' not in settings,'Unknown-call transport differs from saved request')
    account=_ledger(root,output);grant=account['grant']
    _require(grant['digest']==request_pin['sha256'] and grant['calls']==backend['max_calls'],
             'Unknown-call grant differs from request')
    rows=[]
    for item in account['call_rows']:
        row=dict(item['call'])
        if item['profile_binding'] is not None:row['model_profile']=json.loads(item['profile_binding'])
        rows.append(row)
    _require(len(rows)==2 and [row['role'] for row in rows]==['author-0','planner']
             and rows[0]['state']=='reserved_uncertain' and rows[1]['state']=='settled',
             'Unknown-call inventory is not one settled planner and one uncertain author')
    _require(all(call_model(row)==backend['model'] and row['reserved']==settings['reservation_microdollars_per_call']
                 ==reserve_microdollars(backend['max_output_tokens'],model=backend['model']) for row in rows)
             and sum(call_exposure(row) for row in rows)<=grant['ceiling'],'Unknown-call model/reservation differs')
    answer=_role(output,'planner',rows[1],settings)
    _require(_receipt(output/'plan.json')[0]==answer,'Saved paid plan differs from its authenticated response')
    author=agents/'author-0';receipt,_=_receipt(author/'receipt.json');call,_=_receipt(author/'request.json')
    schema,_=_receipt(author/'schema.json',64*1024);prompt=_file_pin(author/'prompt.txt',160*1024)
    images=call.get('images')
    _require(isinstance(images,list) and len(images)<=8,'Unknown author images exceed bound')
    expected_names={'payload-metadata.json','prompt.txt','receipt.json','request.json','schema.json'}
    if streaming:expected_names.add('transport.json')
    total=0
    for index,item in enumerate(images):
        _require(isinstance(item,dict) and set(item)=={'bytes','sha256','suffix'} and item['suffix'] in ('.png','.jpg','.jpeg','.webp'),
                 'Invalid unknown author image binding')
        name=f'image-{index:02d}'+item['suffix'];expected_names.add(name)
        pin=_file_pin(author/name,8*1024**2);total+=pin['bytes']
        _require(pin['bytes']==item['bytes'] and pin['sha256']==item['sha256'] and total<=32*1024**2,
                 'Unknown author image changed or exceeds bound')
    _require(set(_entries(author,14))==expected_names,'Unknown author has response, answer or extra files')
    expected_call={'role':'author-0','prompt_sha256':prompt['sha256'],'schema_sha256':digest(schema),
                   'settings_sha256':digest(settings),'images':images}
    stream_proof=None
    if streaming:
        from avatar_assets.openai_runner import stream_client_request_id
        expected_call['client_request_id']=stream_client_request_id(output.name,'author-0',expected_call)
        stream_proof=_failed_stream_trace(author,call,settings,status)
    _require(call==expected_call and digest(call)==rows[0]['request_digest'],
             'Unknown author request differs from held charge')
    _require(receipt.get('status')=='failed_or_uncertain' and receipt.get('execution_kind')=='openai_responses_api'
             and receipt.get('automatic_resubmission_allowed') is False and receipt.get('reserved_microdollars')==rows[0]['reserved']
             and receipt.get('response_id') is None and receipt.get('usage') is None
             and receipt.get('artifacts')==_call_artifacts(author),'Unknown author receipt is not an unanswered held request')
    inspection=output/'inspection';manifest,_=_receipt(inspection/'manifest.json')
    _require(manifest.get('status')=='needs_visual_review' and manifest.get('kind')=='generated_geometry_preview'
             and manifest.get('provider_calls')==0 and manifest.get('source_glb_sha256')==prepared['reuse_model']['sha256']
             and manifest.get('artifacts')==artifacts(inspection),'Unknown-call inspection custody changed')
    native,_=_receipt(inspection/'native-inspection.json')
    _require(native.get('source_unchanged') is True and native.get('source_sha256')==prepared['reuse_model']['sha256']
             and _file_pin(inspection/'original.glb',64*1024**2)['sha256']==prepared['reuse_model']['sha256'],
             'Unknown-call inspected source differs')
    native_job,_=_receipt(inspection/'job.json')
    _require(native_job.get('status')=='finished' and native_job.get('exit_code')==0
             and native_job.get('process_exit_verified') is True and native_job.get('manual_stop_required') is False,
             'Unknown-call native inspection has no successful exit')
    result={'policy':POLICY,'job_id':output.name,'request_sha256':request_pin['sha256'],**account,
            **({'stream':stream_proof} if stream_proof else {}),
            'output':str(output),'output_inventory':inventory,'files':pins,'input_identities':identities,
            'uncertain_call':{'role':'author-0','request_digest':rows[0]['request_digest'],
                'reserved_microdollars':rows[0]['reserved'],'state':'reserved_uncertain','response_id':None,'actual_microdollars':None}}
    recheck_unknown(result)
    return result


def recheck_unknown(proof, *, ledger=True):
    from avatar_assets.pipeline_continuation import _inventory
    _require(proof.get('policy')==POLICY,'Unknown custody policy')
    _require(_inventory(Path(proof['output']))==proof['output_inventory'],'Unknown-call preserved output changed')
    for pin in proof['files']+([proof['ledger']] if ledger else []):
        _require(_file_pin(Path(pin['path']),64*1024**2)==pin,'Unknown-call preserved custody changed')
    for row in proof['input_identities']:
        info=Path(row['path']).stat()
        _require((info.st_size,info.st_mtime_ns,info.st_dev,info.st_ino)==
                 (row['bytes'],row['mtime_ns'],row['device'],row['inode']),'Unknown-call original input identity changed')
    if ledger:
        _require(not any(Path(proof['ledger']['path']+suffix).exists() for suffix in ('-wal','-shm','-journal')),
                 'Unknown-call ledger journal appeared')


def verify_release(root, path, expected_sha256, *, parent_job_id, parent_digest):
    """Validate the canonical release without opening a provider ledger transaction."""
    root,path=Path(root).resolve(),Path(path)
    _require(isinstance(parent_job_id,str) and re.fullmatch(r'studio-[0-9a-f]{24}',parent_job_id),
             'Invalid unknown-call parent ID')
    job_id=parent_job_id.removeprefix('studio-')
    binding,_=_receipt(root/'.local/studio-registry.json')
    state=Path(binding['state_dir'])
    _require(state==root/'.local/studio' and binding=={'schema_version':1,'state_dir':str(state)},
             'Unknown-call release requires the canonical Studio registry')
    _require(path.parent==state/'jobs'/job_id and re.fullmatch(r'uncertain-owner-release-[0-9a-f]{32}\.json',path.name),
             'Invalid canonical unknown-call release path')
    pin=_private(path,MAX_RELEASE_BYTES);record,_=_receipt(path,MAX_RELEASE_BYTES)
    _require(pin['sha256']==expected_sha256 and record.get('job_id')==job_id
             and record.get('requested_transition')==TRANSITION and record.get('old_job_restart_allowed') is False
             and record.get('no_dispatch') is True and record.get('no_cost_reset') is True,
             'Unknown-call release receipt changed')
    proof=record['proof'];unknown=proof['unknown_provider']
    _require(unknown.get('job_id')==parent_job_id and unknown.get('request_sha256')==parent_digest
             and unknown.get('output')==str(root/'outputs'/parent_job_id),'Unknown-call release request binding changed')
    dbpath=state/'jobs.sqlite3';_private(dbpath,64*1024**2)
    with closing(sqlite3.connect(dbpath.as_uri()+'?mode=ro',uri=True,timeout=.2)) as db:
        db.row_factory=sqlite3.Row
        row=db.execute('SELECT * FROM jobs WHERE id=?',(job_id,)).fetchone()
        _require(row is not None and row['status']=='needs_inspection' and row['output']==parent_job_id
                 and row['request_digest']==record['request_digest'],'Unknown-call canonical job differs')
        request=json.loads(row['request']);data=json.loads(row['data'])
        _require(_digest(request)==row['request_digest'] and _receipt(state/'jobs'/job_id/'request.json')[0]==request
                 and data.get('uncertain_owner_release')=={'path':str(path.relative_to(state)),'sha256':expected_sha256},
                 'Unknown-call canonical release marker differs')
        _require(not db.execute('SELECT 1 FROM active').fetchone()
                 and not db.execute("SELECT 1 FROM jobs WHERE status='queued'").fetchone(),
                 'Unknown-call release verification requires no active or queued owner')
    recheck(root,proof['local_process_custody']);recheck_unknown(unknown,ledger=False)
    return record
