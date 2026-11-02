"""Immutable native-failure continuation; no model call, recipe edit or budget mutation."""
from contextlib import ExitStack, closing
import fcntl
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import stat
import time

from avatar_assets.safety import local_file, read_json, require, sha256, write_json

POLICY='terminal-native-repair-continuation-v1'
MAX_FILES=512
MAX_BYTES=256*1024**2
INTENT=('prompt','style','states','custom_actions','preset','media_spec')


def _inventory(parent):
    started=time.monotonic(); files=[]; total=0; entries=0; pending=[parent]
    while pending:
        with os.scandir(pending.pop()) as children:
            for entry in children:
                entries+=1
                require(entries<=2048 and time.monotonic()-started<20 and not entry.is_symlink(),
                        'Continuation inventory exceeded bounds or contains links')
                info=entry.stat(follow_symlinks=False);path=Path(entry.path)
                if stat.S_ISDIR(info.st_mode):pending.append(path);continue
                require(stat.S_ISREG(info.st_mode),'Unsupported continuation parent entry')
                total+=info.st_size
                require(len(files)<MAX_FILES and total<=MAX_BYTES and info.st_size<=64*1024**2,
                        'Continuation parent exceeds512 files/256MiB/64MiB per file')
                digest=hashlib.sha256();read=0
                with path.open('rb') as stream:
                    while True:
                        chunk=stream.read(min(1024**2,info.st_size-read+1))
                        if not chunk:break
                        read+=len(chunk)
                        require(read<=info.st_size and time.monotonic()-started<20,'Continuation file grew or exceeded deadline')
                        digest.update(chunk)
                require(read==info.st_size and path.stat().st_mtime_ns==info.st_mtime_ns,'Continuation file changed while hashing')
                files.append({'path':str(path.relative_to(parent)),'bytes':info.st_size,'sha256':digest.hexdigest()})
    return sorted(files,key=lambda row:row['path'])


def _parent(root,value):
    parent=Path(value)
    require(parent.is_absolute() and parent.resolve()==parent and parent.parent==Path(root)/'outputs'
            and parent.is_dir(),'Continuation requires a direct local output folder without symlinks')
    return parent


def _ledger(root,parent,request,*,successor=None):
    from avatar_assets.openai_runner import call_exposure
    path=local_file(Path(root)/'.local/openai/ledger.sqlite3',64*1024**2)
    info=path.stat()
    require(info.st_uid==os.getuid() and stat.S_IMODE(info.st_mode)==0o600,'Continuation ledger is not owner-only')
    with closing(sqlite3.connect(path.as_uri()+'?mode=ro',uri=True,timeout=.2)) as db:
        db.row_factory=sqlite3.Row;db.execute('PRAGMA query_only=ON');db.execute('BEGIN')
        grant=db.execute('SELECT id,digest,calls,ceiling,approval FROM jobs WHERE id=?',(parent.name,)).fetchone()
        rows=[dict(row) for row in db.execute('SELECT * FROM calls WHERE job=? ORDER BY role LIMIT 13',(parent.name,))]
        retired=db.execute('SELECT * FROM retirements WHERE parent=?',(parent.name,)).fetchone()
        child_grant=(db.execute('SELECT digest,calls,ceiling,approval FROM jobs WHERE id=?',(successor['id'],)).fetchone()
                     if successor is not None else None)
    require(grant is not None and grant['digest']==sha256(parent/'request.json')
            and grant['calls']==request['max_agent_calls'],'Continuation parent grant differs')
    require(1<=len(rows)<=grant['calls']<=12 and all(row['state']=='settled' for row in rows),
            'Continuation refuses uncertain or missing parent calls')
    actual=sum(call_exposure(row) for row in rows)
    if retired is not None:
        require(successor is not None and retired['successor']==successor['id'],
                'Continuation parent is retired into a different or unspecified successor')
        require(type(retired['actual']) is int and retired['actual']==actual
                and child_grant is not None and child_grant['digest']==successor['digest']
                and child_grant['calls']==successor['calls'] and type(child_grant['ceiling']) is int
                and 1<=child_grant['ceiling']<=1000000000
                and child_grant['approval']==retired['approval'],'Continuation successor retirement accounting differs')
        require(isinstance(retired['approval'],str) and len(retired['approval'].encode())<=8192,
                'Invalid continuation retirement approval')
        approval=json.loads(retired['approval'])
        require(isinstance(approval,dict) and approval.get('decision')=='approved'
                and approval.get('operation')=='retire_and_continue'
                and approval.get('parent_job_id')==parent.name
                and approval.get('successor_job_id')==successor['id']
                and type(approval.get('additional_calls')) is int and approval['additional_calls']==successor['calls']
                and isinstance(approval.get('user_reply'),str) and 0<len(approval['user_reply'])<=4000,
                'Continuation retirement lacks exact successor approval')
    return {'grant':dict(grant),'calls':rows}


def _role(parent,role,row,settings):
    from avatar_assets.agent_runner import _call_artifacts,_validate
    from avatar_assets.openai_runner import digest,_decode
    folder=parent/'agents'/role
    receipt=read_json(folder/'receipt.json');request=read_json(folder/'request.json')
    answer=read_json(folder/'answer.json');schema=read_json(folder/'schema.json')
    require(receipt.get('status')=='succeeded' and receipt.get('execution_kind')=='openai_responses_api'
            and receipt.get('artifacts')==_call_artifacts(folder)
            and receipt.get('answer_sha256')==sha256(folder/'answer.json')
            and request.get('role')==role and request.get('settings_sha256')==digest(settings)
            and request.get('prompt_sha256')==sha256(folder/'prompt.txt')
            and request.get('schema_sha256')==digest(schema)
            and row['request_digest']==digest(request) and row['response_id']==receipt.get('response_id')
            and row['actual']==receipt.get('actual_microdollars_upper_estimate')
            and row['reserved']==receipt.get('reserved_microdollars')
            and json.loads(row['usage'])==receipt.get('usage'),'Continuation agent receipt changed')
    response=_decode(local_file(folder/'response.json',4*1024**2).read_bytes())
    require(response.get('id')==row['response_id'] and response.get('status')=='completed'
            and response.get('model')==settings['model'] and response.get('usage')==receipt['usage'],
            'Continuation provider response differs')
    output=response.get('output')
    require(isinstance(output,list) and len(output)==1 and output[0].get('type')=='message'
            and len(output[0].get('content',[]))==1 and output[0]['content'][0].get('type')=='output_text'
            and _decode(output[0]['content'][0]['text'].encode())==answer,'Continuation answer differs from provider text')
    _validate(answer,schema)
    return answer


def _match(request,parent_request):
    require(all(request.get(key)==parent_request.get(key) for key in INTENT),'Continuation changes original prompt/style/actions/output intent')
    for field in ('reuse_model','reference'):
        child,old=request.get(field),parent_request.get(field)
        require((child is None)==(old is None) and (child is None or
                (child['sha256'],child['bytes'])==(old['sha256'],old['bytes'])),
                'Continuation source/reference differs')
    require(request.get('reuse_model') is not None and request['max_credits']==0,
            'Continuation requires exact explicit source reuse and no new Meshy credits')


def _read_context(child,request):
    binding=request['continuation']
    require(isinstance(binding,dict) and set(binding)=={'schema_version','policy','parent','receipt','sha256'}
            and binding['schema_version']==1 and binding['policy']==POLICY
            and binding['receipt']=='continuation/receipt.json' and isinstance(binding['parent'],str)
            and binding['parent'].startswith('outputs/') and len(Path(binding['parent']).parts)==2,
            'Invalid continuation binding')
    receipt=read_json(child/binding['receipt'])
    require(sha256(child/binding['receipt'])==binding['sha256'] and receipt.get('policy')==POLICY
            and receipt.get('schema_version')==1 and 1<=receipt.get('depth',0)<=8,'Continuation receipt changed')
    context=read_json(child/'continuation/context.json')
    require(receipt['context_sha256']==sha256(child/'continuation/context.json')
            and set(context)=={'plan','previous_answer','findings'},'Continuation context changed')
    return receipt,context


def collect(root,parent):
    """Verify one terminal parent and capture its latest complete failed-native context."""
    from avatar_assets.pipeline import Pipeline
    from avatar_assets.pipeline_prompts import validate_plan,_author_design,finish_config
    from avatar_assets.author_geometry import author_diagnostics
    root=Path(root);parent=_parent(root,parent)
    with ExitStack() as stack:
        for name in ('pipeline.lock','agents/runner.lock'):
            handle=stack.enter_context(local_file(parent/name,1024).open('rb'))
            try:fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:raise ValueError('Continuation parent still has a live owner') from None
        inventory=_inventory(parent)
        pinned={row['path']:row for row in inventory}
        request=read_json(parent/'request.json');input_receipt=read_json(parent/'input-receipt.json')
        terminal=read_json(parent/'unsupported.json');status=read_json(parent/'status.json')
        require(terminal.get('status')==status.get('status')=='unsupported'
                and terminal.get('stage')==status.get('stage')=='repair_budget'
                and terminal['request_sha256']==input_receipt['request_sha256']==sha256(parent/'request.json'),
                'Continuation parent is not a bound terminal repair-budget result')
        require(request.get('agent_backend',{}).get('provider')=='openai'
                and type(request.get('max_repairs')) is int and 0<=request['max_repairs']<=2
                and request.get('reuse_model') is not None,'Unsupported continuation parent contract')
        for row in input_receipt['sources']:
            require(pinned.get('sources/'+row['path'])=={**row,'path':'sources/'+row['path']},'Continuation original source snapshots changed')
        for field in ('reuse_model','reference'):
            item=request.get(field)
            if item:
                require(pinned.get(item['path'])=={key:item[key] for key in ('path','bytes','sha256')},'Continuation selected input changed')
        for row in inventory:
            if row['path'].endswith('/job.json'):
                job=read_json(parent/row['path'])
                require(job.get('process_exit_verified') is True and not job.get('manual_stop_required'),
                        'Continuation native process custody is unresolved')
        ledger=_ledger(root,parent,request);calls={row['role']:row for row in ledger['calls']}
        allowed={'planner'}|{f'{role}-{attempt}' for role in ('author','reviewer') for attempt in range(request['max_repairs']+1)}
        require(set(calls)<=allowed and {f'author-{attempt}' for attempt in range(request['max_repairs']+1)}<=set(calls),
                'Continuation parent has unexpected or missing author roles')
        folders={path.name for path in (parent/'agents').iterdir() if path.is_dir()}
        require(folders==set(calls),'Continuation agent folders/call ledger disagree')
        settings=read_json(parent/'agents/runner.json')
        require(settings.get('job_id')==parent.name and settings.get('job_digest')==sha256(parent/'request.json')
                and settings.get('max_calls')==request['max_agent_calls']
                and settings.get('implementation_sha256')==pinned['sources/openai_runner.py']['sha256'],
                'Continuation parent runner binding changed')
        answers={role:_role(parent,role,row,settings) for role,row in calls.items()}
        plan=read_json(parent/'plan.json');validate_plan(plan,request)
        depth=1
        if 'continuation' in request:
            prior,prior_context=_read_context(parent,request)
            require(prior_context['plan']==plan and 'planner' not in calls,'Continuation inherited plan changed')
            depth=prior['depth']+1
        else:
            require(answers.get('planner')==plan,'Continuation plan differs from completed planner answer')
        require(depth<=8,'Continuation chain reached its eight-job bound')
        attempt=request['max_repairs'];role='author-'+str(attempt)
        require(role in answers and 'reviewer-'+str(attempt) not in answers,'Continuation requires the latest failed native author attempt')
        answer=answers[role];design=_author_design(answer)
        pack=parent/f'model-{attempt}';manifest=read_json(pack/'manifest.json')
        pack_rows=[{**row,'path':row['path'][len(pack.name)+1:]} for row in inventory
                   if row['path'].startswith(pack.name+'/') and row['path']!=pack.name+'/manifest.json']
        require(manifest.get('status')=='failed' and manifest.get('artifacts')==pack_rows,'Continuation failed pack custody changed')
        failure=read_json(parent/f'stage-failure-model-{attempt}.json')
        require(failure['manifest_sha256']==sha256(pack/'manifest.json'),'Continuation stage failure receipt changed')
        require(sha256(parent/f'finish-{attempt}.json')==sha256(pack/'sources/finish-design.json'),
                'Continuation authored recipe differs from native snapshot')
        findings={'validation_failure':failure['error'],'measurements':Pipeline.__new__(Pipeline)._diagnostics(pack)}
        require('native_failure' in findings['measurements'],'Continuation needs actionable authenticated native diagnostics')
        inspection=read_json(parent/'inspection/manifest.json')
        native=inspection['native'];source=request['reuse_model']
        require(inspection.get('source_glb_sha256')==source['sha256'],'Continuation inspection belongs to another source')
        recipe=finish_config(answer,request,plan,{'path':str(parent/source['path']),'sha256':source['sha256']},attempt,
            quality_policy=read_json(parent/'quality-policy.json'),
            bone_parents={bone['name']:bone['parent'] for rig in native.get('rigs',[]) for bone in rig['bones']})
        require(recipe==read_json(parent/f'finish-{attempt}.json'),'Continuation recipe was edited outside the author contract')
        geometry=author_diagnostics(design,parent/source['path'],source['sha256'],native)
        require(geometry==read_json(parent/f'author-geometry-{attempt}.json'),'Continuation geometry diagnostics changed')
        findings['measurements']['author_geometry']=geometry
        require(findings==terminal.get('findings'),'Continuation terminal findings differ from authenticated evidence')
        require(_inventory(parent)==inventory,'Continuation parent changed during inspection')
        return {'parent':str(parent.relative_to(root)),'parent_request':request,'inventory':inventory,
                'ledger':ledger,'depth':depth,'context':{'plan':plan,'previous_answer':answer,'findings':findings}}


def save(child,request,captured):
    _match(request,captured['parent_request'])
    folder=Path(child)/'continuation';folder.mkdir(mode=0o700)
    write_json(folder/'context.json',captured['context'])
    receipt={'schema_version':1,'policy':POLICY,'parent':captured['parent'],
             'parent_request_sha256':next(row['sha256'] for row in captured['inventory'] if row['path']=='request.json'),
             'inventory':captured['inventory'],'ledger':captured['ledger'],'depth':captured['depth'],
             'context_sha256':sha256(folder/'context.json'),'new_planner_calls':0,'parent_mutated':False}
    write_json(folder/'receipt.json',receipt)
    return {'schema_version':1,'policy':POLICY,'parent':captured['parent'],
            'receipt':'continuation/receipt.json','sha256':sha256(folder/'receipt.json')}


def verify(root,child,request):
    root,child=Path(root),Path(child)
    receipt,context=_read_context(child,request)
    parent=_parent(root,root/request['continuation']['parent'])
    with ExitStack() as stack:
        for name in ('pipeline.lock','agents/runner.lock'):
            handle=stack.enter_context(local_file(parent/name,1024).open('rb'))
            try:fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:raise ValueError('Continuation parent still has a live owner') from None
        require(receipt['parent']==request['continuation']['parent'] and _inventory(parent)==receipt['inventory'],
                'Continuation parent artifacts changed')
        parent_request=read_json(parent/'request.json')
        require(sha256(parent/'request.json')==receipt['parent_request_sha256'],'Continuation parent request changed')
        _match(request,parent_request)
        require(_ledger(root,parent,parent_request,successor={'id':child.name,'digest':sha256(child/'request.json'),
                'calls':request['max_agent_calls']})==receipt['ledger'],'Continuation parent accounting changed')
    return context
