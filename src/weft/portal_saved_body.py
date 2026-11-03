"""Explicit saved-body approximation children; no inference or inventory-absence claims."""
from hashlib import sha256
import fcntl
import json
import math
import os
from pathlib import Path
import re
import time
from uuid import uuid4

from weft.io import canonical_hash, local_time
from weft.lab_worker import atomic_json
from weft.shared_world_packaging import read_package_file, write_package_files

PROFILE = 'portal-saved-body-v1'
DIRECTORY = 'saved-body'
MODELING_SCOPE = 'body_authored_ground_only'
MAX_JOB_BYTES = 650_000_000
ACTIVE = {'queued','running','cancelling','preflighting','analyzing','preparing_tracking','tracking','building','loading'}
TERMINAL = {'ready','failed','cancelled','interrupted','needs_input'}
SCOPE = ('Body and authored-ground approximation from saved evidence. Original inventory is preserved; '
         'equipment, obstacles, other people and their contacts are omitted, not absent. Ground and anatomy '
         'remain unverified priors. No AI calls, learned policy or physical validation.')
ARTIFACTS = {'request.json','start.json','inventory.json','receipt.json','report.json','failure.json',
             'world/manifest.json','world/reconstruction.json','world/reference-feasibility.json',
             'world/scene.json','world/task.json','world/inventory.json','world/source-binding-verification.json'}
IMPLEMENTATIONS = ('portal_saved_body.py','person_inventory.py','shared_world_human.py',
                   'exercise_perception_pipeline.py','portal_inventory.py','portal_task_jobs.py')


def _require(value,message):
    if not value: raise ValueError('Saved body: '+message)


def _check(deadline,child=None):
    _require(type(deadline) in (int,float) and math.isfinite(deadline),'finite deadline required')
    if time.monotonic() >= deadline: raise TimeoutError('Saved-body aggregate deadline exceeded')
    if child is not None and any((child/n).exists() or (child/n).is_symlink() for n in ('cancel-request.json',DIRECTORY+'/cancel-request.json')):
        raise InterruptedError('Cancelled; original source and partial results preserved')


def _json(raw,maximum=128<<10):
    from weft.spatial_scene_inputs import _json as strict
    return strict(raw,maximum)


def _bytes(value):
    raw=(json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()
    _require(len(raw)<=128<<10,'small receipt exceeds128KiB');return raw


def _read(root,name,maximum=128<<10,pin=None):
    return read_package_file(root,name,max_bytes=maximum,expected_sha256=pin)


def _directory(path):
    path=Path(path).absolute()
    _require('..' not in path.parts and path.is_dir() and not any(p.is_symlink() for p in (path,*path.parents)),'safe existing directory required')
    return path


def validate_request(value):
    _require(type(value) is dict and set(value)=={'request_id','preflight_manifest_sha256','modeling_scope'},'closed scope request required')
    for key,length in [('request_id',32),('preflight_manifest_sha256',64)]:
        _require(type(value[key]) is str and re.fullmatch('[0-9a-f]{'+str(length)+'}',value[key]),'exact '+key+' required')
    _require(value['modeling_scope']==MODELING_SCOPE,'explicit authored body/ground scope required')
    return dict(value)


def _metadata(parent,deadline):
    from weft.portal_inventory import validate_intent
    parent=_directory(parent)
    _require(re.fullmatch('[0-9a-f]{32}',parent.name),'exact parent ID required')
    files={n:_read(parent,n) for n in ('job.json','status.json','perception/status.json','perception/preflight/manifest.json')}
    job,state,perception,manifest=(_json(files[n]) for n in files)
    _require(job.get('id')==parent.name and job.get('kind','video')=='video'
             and job.get('mode') in {'auto','exercise'} and state.get('route')=='exercise',
             'resolved original exercise upload required')
    _require(type(job.get('source_sha256')) is str and re.fullmatch('[0-9a-f]{64}',job['source_sha256']),'original source hash required')
    _require(job.get('source_file')=='source.mp4','original fixed MP4 required')
    _require(state.get('status') not in ACTIVE and perception.get('status') not in ACTIVE,'parent work is active')
    _require(not (parent/'cancel-request.json').exists() and not (parent/'cancel-request.json').is_symlink(),'parent cancellation retained')
    for n in ('equipment/status.json','moving-equipment/status.json','muse/status.json','checks/status.json','repairs/status.json'):
        if (parent/n).exists() or (parent/n).is_symlink():
            files[n]=_read(parent,n);_require(_json(files[n]).get('status') not in ACTIVE,'parent work is active')
    _require(type(job.get('exercise_intent')) is dict,'original uploaded intent required; no invented inventory')
    intent=validate_intent(job['exercise_intent'])
    _require(manifest.get('profile')=='exercise-perception-preflight-v1' and manifest.get('status')=='local_ready'
             and type(manifest.get('files')) is dict and len(manifest['files'])<=64
             and type(manifest.get('children')) is dict and set(manifest['children'])=={'body_seed','body_source'},'complete retained local body preflight required')
    for name,pin in manifest['children'].items():
        _require(type(pin) is str and re.fullmatch('[0-9a-f]{64}',pin),'child seal required')
        n='perception/preflight/'+name+'/manifest.json';files[n]=_read(parent,n,128<<10,pin)
    for name in ('metadata.json','result.json'):
        _require(name in manifest['files'],'sealed preflight metadata required')
        _require(type(manifest['files'][name]) is str and re.fullmatch('[0-9a-f]{64}',manifest['files'][name]),'exact preflight metadata seal required')
        n='perception/preflight/'+name;files[n]=_read(parent,n,128<<10,manifest['files'][name])
    metadata=_json(files['perception/preflight/metadata.json']);result=_json(files['perception/preflight/result.json'])
    _require(metadata.get('source_sha256')==result.get('source_sha256')==job['source_sha256']
             and result.get('status')=='local_ready' and result.get('simulation_ready') is False,'saved preflight source identity differs')
    body_manifest=_json(files['perception/preflight/body_source/manifest.json'])
    _require(body_manifest.get('profile')=='exercise-body-source-v1' and body_manifest.get('status')=='construction_only'
             and body_manifest.get('simulation_ready') is False and type(body_manifest.get('files')) is dict
             and len(body_manifest['files'])<=511 and 'failure.json' not in body_manifest['files'],'unfailed sealed construction body required')
    for relative in ('perception/preflight/failure.json','perception/preflight/body_source/failure.json'):
        _require(not (parent/relative).exists() and not (parent/relative).is_symlink(),'saved body/preflight failure is preserved')
    _check(deadline)
    return job,intent,manifest,files


def availability(parent):
    """Bounded metadata only; full source and body admission belongs to the worker."""
    try:
        job,intent,manifest,files=_metadata(parent,time.monotonic()+2)
        return {'available':True,'reason':None,'preflight_manifest_sha256':sha256(files['perception/preflight/manifest.json']).hexdigest(),
                'body_manifest_sha256':manifest['children']['body_source'],'modeling_scope':MODELING_SCOPE,'original_exercise_intent':intent,'scope':SCOPE}
    except (ValueError,OSError,TimeoutError) as error:
        return {'available':False,'reason':str(error)[:300],'preflight_manifest_sha256':None,'body_manifest_sha256':None,'scope':SCOPE}


def queue_saved_body_child(root,parent_id,payload,*,lease_fd,existing_jobs,stored_bytes,reserved_bytes,pending_jobs,deadline):
    from weft.portal import MAX_JOBS,MAX_STORAGE
    _check(deadline);root=_directory(root);payload=validate_request(payload)
    _require(type(parent_id) is str and re.fullmatch('[0-9a-f]{32}',parent_id),'exact parent ID required')
    _require(type(lease_fd) is int and not (root/'portal.lock').is_symlink(),'existing portal lease required')
    descriptor,lock=os.fstat(lease_fd),(root/'portal.lock').stat()
    _require((descriptor.st_dev,descriptor.st_ino)==(lock.st_dev,lock.st_ino),'lease belongs to another portal')
    fcntl.flock(lease_fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
    parent=root/parent_id;job,intent,manifest,snapshots=_metadata(parent,deadline)
    _require(type(existing_jobs) is dict and len(existing_jobs)<=MAX_JOBS and existing_jobs.get(parent_id)==job,'bounded parent inventory differs')
    pins={n:sha256(v).hexdigest() for n,v in snapshots.items()}
    _require(payload['preflight_manifest_sha256']==pins['perception/preflight/manifest.json'],'requested preflight seal changed')
    identity=canonical_hash({'parent_id':parent_id,'parent_files':pins,'modeling_scope':MODELING_SCOPE})
    for key,other in existing_jobs.items():
        _check(deadline)
        if other.get('request_id')==payload['request_id'] or other.get('saved_body_identity')==identity:
            _require(other.get('saved_body_identity')==identity and type(key) is str and re.fullmatch('[0-9a-f]{32}',key),'request ID already binds another action')
            raw=_read(root/key,DIRECTORY+'/request.json',pin=other.get('saved_body_request_sha256'));saved=_json(raw)
            _require(saved.get('identity')==identity and saved.get('parent_files')==pins,'existing child request changed; no retry')
            return other
    _require(all(type(v) is int and v>=0 for v in (stored_bytes,reserved_bytes,pending_jobs)) and pending_jobs<4
             and len(existing_jobs)<MAX_JOBS and stored_bytes+reserved_bytes+MAX_JOB_BYTES<=MAX_STORAGE,'portal cannot reserve this complete bounded child')
    child_id=uuid4().hex;stage=root/('.saved-body-'+child_id);stage.mkdir(mode=0o700)
    request={'schema_version':1,'profile':PROFILE,'child_id':child_id,'parent_id':parent_id,'parent_files':pins,'payload':payload,
             'identity':identity,'original_exercise_intent':intent,'body_manifest_sha256':manifest['children']['body_source'],'api_calls_authorized':0}
    raw=_bytes(request)
    child={'id':child_id,'kind':'saved_body','name':'Body and ground approximation','parent_id':parent_id,'mode':'exercise','ai':False,
           'source_sha256':job['source_sha256'],'request_id':payload['request_id'],'saved_body_identity':identity,
           'saved_body_request_sha256':sha256(raw).hexdigest(),'created_at':local_time()}
    try:
        write_package_files(stage,{'job.json':_bytes(child),'status.json':_bytes({'status':'queued','stage':'saved_body','route':'exercise','scope':SCOPE}),
            DIRECTORY+'/request.json':raw,**{DIRECTORY+'/parent/'+n:v for n,v in snapshots.items()}})
        _require(_metadata(parent,deadline)[3]==snapshots,'parent changed before publication');_check(deadline)
        os.rename(stage,root/child_id);return child
    except Exception as error:
        write_package_files(stage,{DIRECTORY+'/publication-failure.json':_bytes({'error':str(error)[:1000],'preserved':True})});raise


def read_status(child):return _json(_read(child,'status.json',32<<10))

def reserved_growth(child):return MAX_JOB_BYTES if read_status(child).get('status') in ACTIVE else 0

def summary(child):
    job=_json(_read(child,'job.json'));state=read_status(child)
    result={**state,'parent_id':job.get('parent_id'),'scope':SCOPE,'api_calls':0,'inference_calls':0,'artifacts':state.get('artifacts',[])}
    if state.get('report_sha256'):result['report']=_json(_read(child,DIRECTORY+'/report.json',pin=state['report_sha256']))
    return result


def _bound(child,deadline):
    job_raw=_read(child,'job.json');job=_json(job_raw);raw=_read(child,DIRECTORY+'/request.json',pin=job.get('saved_body_request_sha256'));request=_json(raw)
    _require(job.get('kind')=='saved_body' and job.get('id')==child.name and request.get('child_id')==child.name
             and request.get('profile')==PROFILE and type(request.get('parent_id')) is str
             and re.fullmatch('[0-9a-f]{32}',request['parent_id']) and job.get('parent_id')==request['parent_id'],'child/parent binding differs')
    payload=validate_request(request['payload']);parent=_directory(child.parent/request['parent_id'])
    _require(parent.parent==child.parent,'parent must remain inside the portal')
    parent_job,intent,manifest,snapshots=_metadata(parent,deadline);pins={n:sha256(v).hexdigest() for n,v in snapshots.items()}
    identity=canonical_hash({'parent_id':parent.name,'parent_files':pins,'modeling_scope':MODELING_SCOPE})
    _require(pins==request.get('parent_files') and identity==request.get('identity')==job.get('saved_body_identity')
             and intent==request.get('original_exercise_intent') and manifest['children']['body_source']==request.get('body_manifest_sha256')
             and payload['preflight_manifest_sha256']==pins['perception/preflight/manifest.json']
             and parent_job['source_sha256']==job.get('source_sha256'),'queued source/scope changed')
    for n,pin in pins.items():_check(deadline,child);_read(child,DIRECTORY+'/parent/'+n,128<<10,pin)
    return job,job_raw,request,raw,parent,snapshots


def execute_job(child,*,project=None):
    """One supervised keyless worker; never re-infer or resume a partial construction."""
    from weft.portal_task_jobs import _tree
    child=_directory(child);work=child/DIRECTORY;state=read_status(child)
    if state.get('status') in TERMINAL:return 0 if state['status']=='ready' else 1
    started=time.monotonic();deadline=started+120;stage='admission';prior=(work/'start.json').exists() or (work/'start.json').is_symlink()
    try:
        _check(deadline,child);_require(state.get('status')=='queued' and not prior,'partial saved-body job cannot retry')
        write_package_files(work,{'start.json':_bytes({'profile':PROFILE,'started_at':local_time(),'api_calls':0,'retry':False})})
        atomic_json(child/'status.json',{'status':'running','stage':stage,'route':'exercise','scope':SCOPE})
        job,job_raw,request,request_raw,parent,snapshots=_bound(child,deadline)
        before=_tree(parent/'perception/preflight',deadline)
        code={n:Path(__file__).with_name(n).read_bytes() for n in IMPLEMENTATIONS}
        write_package_files(work,{'implementation/'+n:v for n,v in code.items()})
        from weft.exercise_perception_pipeline import validate_perception_preflight
        admitted=validate_perception_preflight(parent/'source.mp4',parent/'perception/preflight',deadline=min(deadline,time.monotonic()+30))
        _require(admitted['manifest_sha256']==request['payload']['preflight_manifest_sha256'],'admitted preflight changed')
        from weft.person_inventory import make_conditional_person_inventory
        inventory=make_conditional_person_inventory(job['source_sha256'],request['original_exercise_intent'])
        write_package_files(work,{'inventory.json':_bytes(inventory)})
        stage='construction';atomic_json(child/'status.json',{'status':'running','stage':stage,'route':'exercise','scope':SCOPE})
        from weft.shared_world_human import construct_human_only_run
        built=construct_human_only_run(parent/'perception/preflight/body_source',inventory,work/'world',deadline=deadline,max_seconds=30)
        from weft.shared_world_packaging import read_package_manifest
        world_raw,world_files=read_package_manifest(work/'world')
        _require(sha256(world_raw).hexdigest()==built['manifest_sha256'],'constructed world seal differs')
        feasibility=_json(_read(work/'world','reference-feasibility.json',4<<20,world_files['reference-feasibility.json']),4<<20)
        _check(deadline,child);stage='lab';atomic_json(child/'status.json',{'status':'running','stage':stage,'route':'exercise','scope':SCOPE})
        from weft.shared_exercise_lab import prepare_shared_exercise_lab
        remaining=min(60,deadline-time.monotonic());_require(remaining>=1,'no bounded lab time remains')
        # Match the primary exercise route: explicit disclosed root assistance, unchanged gates.
        lab=prepare_shared_exercise_lab(work/'world',child/'lab',root_assistance=True,max_seconds=remaining)
        config=_json(_read(child/'lab','lab-config.json',64<<10));lab_manifest=_read(child/'lab/run','manifest.json',1<<20)
        _require(sha256(lab_manifest).hexdigest()==config['manifest_sha256']==lab['manifest_sha256'],'prepared lab seal differs')
        terminal={'status':'ready','lab':'lab'};report={'status':'ready','lab_manifest_sha256':config['manifest_sha256'],
            'reference_feasibility':built['reference_feasibility'],'reference_feasible':feasibility.get('status')=='feasible',
            'task_success':False,'readiness':'manual_lab_admitted_not_reference_success'}
        stage='publication';_check(deadline,child)
        _require(_tree(parent/'perception/preflight',deadline)==before and _metadata(parent,deadline)[3]==snapshots,'original preflight changed')
        _read(parent,'source.mp4',37_000_000,job['source_sha256']);_check(deadline,child)
        _require(_read(child,'job.json')==job_raw and _read(child,DIRECTORY+'/request.json')==request_raw
                 and all(Path(__file__).with_name(n).read_bytes()==v for n,v in code.items()),'request/runtime changed')
        _require(sum(x[2] for x in _tree(child,deadline).values())+(512<<10)<=MAX_JOB_BYTES,'job finalization exceeds storage bound')
        receipt={'profile':PROFILE,'parent_id':parent.name,'parent_files':request['parent_files'],'body_manifest_sha256':request['body_manifest_sha256'],
                 'source_sha256':job['source_sha256'],'world_manifest_sha256':built['manifest_sha256'],'inventory':inventory,'scope':SCOPE,
                 'runtime_sha256':{n:sha256(v).hexdigest() for n,v in code.items()},'api_calls':0,'inference_calls':0,'source_modified':False}
        report.update(profile=PROFILE,source_parent_id=parent.name,inventory=inventory,scope=SCOPE,api_calls=0,inference_calls=0,learned_policy_available=False,physical_validation='unverified')
        rr,pr=_bytes(receipt),_bytes(report);write_package_files(work,{'receipt.json':rr,'report.json':pr});_check(deadline,child)
        artifacts=[DIRECTORY+'/'+n for n in sorted(ARTIFACTS) if (work/n).is_file()]
        atomic_json(child/'status.json',{**terminal,'stage':'complete','route':'exercise','report_sha256':sha256(pr).hexdigest(),'receipt_sha256':sha256(rr).hexdigest(),
            'artifacts':artifacts,'seconds':time.monotonic()-started,'scope':SCOPE,'api_calls':0,'inference_calls':0})
        return 0 if terminal['status']=='ready' else 1
    except Exception as error:
        failure={'status':'failed','stage':stage,'reason':f'{type(error).__name__}: {error}'[:1000],'scope':SCOPE,'retry':False,'partial_results_preserved':True,'api_calls':0,'inference_calls':0}
        if not (work/'failure.json').exists():write_package_files(work,{'failure.json':_bytes(failure)})
        if not (work/'report.json').exists():write_package_files(work,{'report.json':_bytes(failure)})
        report_raw=_read(work,'report.json')
        terminal='cancelled' if isinstance(error,InterruptedError) else 'interrupted' if prior or state.get('status')=='running' else 'failed'
        atomic_json(child/'status.json',{'status':terminal,'stage':stage,'route':'exercise','error':failure['reason'],'scope':SCOPE,'report_sha256':sha256(report_raw).hexdigest(),
            'artifacts':[DIRECTORY+'/'+n for n in sorted(ARTIFACTS) if (work/n).is_file()],'retry':False})
        return 1
