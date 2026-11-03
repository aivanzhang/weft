"""Immutable evidence-bound continuation; no model call, recipe edit or budget mutation."""
from contextlib import ExitStack, closing
import fcntl
import ast
import hashlib
import json
import math
import os
from pathlib import Path
import sqlite3
import stat
import time

from avatar_assets.safety import local_file, read_json, require, sha256, write_json

POLICY='terminal-native-repair-continuation-v1'
MAX_FILES=4096
MAX_BYTES=1024**3
MAX_ENTRIES=8192
MAX_FILE_BYTES=64*1024**2
INVENTORY_SECONDS=20
INTENT=('prompt','style','states','custom_actions','preset','media_spec')


def _inventory(parent):
    started=time.monotonic(); files=[]; total=0; entries=0; pending=[parent]
    while pending:
        with os.scandir(pending.pop()) as children:
            for entry in children:
                entries+=1
                require(entries<=MAX_ENTRIES and time.monotonic()-started<INVENTORY_SECONDS and not entry.is_symlink(),
                        'Continuation inventory exceeded bounds or contains links')
                info=entry.stat(follow_symlinks=False);path=Path(entry.path)
                if stat.S_ISDIR(info.st_mode):pending.append(path);continue
                require(stat.S_ISREG(info.st_mode),'Unsupported continuation parent entry')
                total+=info.st_size
                require(len(files)<MAX_FILES and total<=MAX_BYTES and info.st_size<=MAX_FILE_BYTES,
                        'Continuation parent exceeds4096 files/1GiB/64MiB per file')
                digest=hashlib.sha256();read=0
                with path.open('rb') as stream:
                    while True:
                        chunk=stream.read(min(1024**2,info.st_size-read+1))
                        if not chunk:break
                        read+=len(chunk)
                        require(read<=info.st_size and time.monotonic()-started<INVENTORY_SECONDS,'Continuation file grew or exceeded deadline')
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
    from avatar_assets.openai_runner import call_exposure,attach_call_profiles
    path=local_file(Path(root)/'.local/openai/ledger.sqlite3',64*1024**2)
    info=path.stat()
    require(info.st_uid==os.getuid() and stat.S_IMODE(info.st_mode)==0o600,'Continuation ledger is not owner-only')
    with closing(sqlite3.connect(path.as_uri()+'?mode=ro',uri=True,timeout=.2)) as db:
        db.row_factory=sqlite3.Row;db.execute('PRAGMA query_only=ON');db.execute('BEGIN')
        grant=db.execute('SELECT id,digest,calls,ceiling,approval FROM jobs WHERE id=?',(parent.name,)).fetchone()
        rows=[dict(row) for row in db.execute('SELECT * FROM calls WHERE job=? ORDER BY role LIMIT 13',(parent.name,))]
        rows=attach_call_profiles(db,rows)
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
    from avatar_assets.openai_runner import digest,_decode,response_answer_text,call_model
    require(call_model(row)==settings['model'],'Continuation ledger model differs from runner')
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
    if settings.get('transport') == 'sse-v1':
        from avatar_assets.openai_runner import verify_stream_trace
        verify_stream_trace(folder,request,settings,row['response_id'])
    require(_decode(response_answer_text(response,reasoning_effort=settings.get('reasoning',{}).get('effort','none')).encode())==answer,
            'Continuation answer differs from provider text')
    _validate(answer,schema)
    if role.startswith('reviewer-') and answer.get('schema_version') == 2:
        from avatar_assets.pipeline_prompts import decode_typed_review
        return decode_typed_review(answer)
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
            and set(context) in ({'plan','previous_answer','findings'}, {'plan','previous_answer','findings','replan'}),
            'Continuation context changed')
    if 'replan' in context:
        require(isinstance(context['replan'],dict) and set(context['replan'])=={'validation_failure','previous_plan'}
                and context['replan']['previous_plan']==context['plan']
                and isinstance(context['replan']['validation_failure'],str)
                and 0<len(context['replan']['validation_failure'])<=4000,'Invalid continuation replan evidence')
    return receipt,context


def _sealed_folder(parent,folder,inventory):
    prefix=str(folder.relative_to(parent))+'/'
    rows=[{**row,'path':row['path'][len(prefix):]} for row in inventory
          if row['path'].startswith(prefix) and row['path']!=prefix+'manifest.json']
    manifest=read_json(folder/'manifest.json')
    require(manifest.get('artifacts')==rows,'Continuation measured pack custody changed')
    return manifest


def _portable_geometry_facts(parent,pack,manifest,pinned,config,attempt):
    """Summarize sealed authored coat geometry, not measured spacing or visual quality."""
    report_path=f'{pack.name}/finish-report.json';config_path=f'finish-{attempt}.json'
    config_pin=pinned[config_path]
    require(config_pin['sha256']==pinned[f'{pack.name}/sources/finish-design.json']['sha256'],
            'Historical geometry config differs from model snapshot')
    facts={'status':'unavailable','scope':'Historical authored coat geometry; not current acceptance or measured root spacing.',
           'finish_config_reference':config_pin,'bands':[]}
    def number(value,label):
        require(type(value) in (int,float) and math.isfinite(value),'Invalid historical '+label)
        return value
    groom=config.get('groom') or {}
    if 'comb' in groom:facts['groom_comb']=number(groom['comb'],'groom comb')
    regions=(config.get('groom_flow') or {}).get('regions',[])
    require(isinstance(regions,list) and len(regions)<=12,'Historical groom flow exceeds12 regions')
    facts['groom_flow_regions']=[]
    for index,region in enumerate(regions):
        row={'index':index}
        if 'name' in region:
            require(isinstance(region['name'],str) and len(region['name'])<=128,'Invalid historical flow name')
            row['name']=region['name']
        for key in ('comb','length_scale'):
            if key in region:row[key]=number(region[key],'flow '+key)
        if 'direction' in region:
            require(isinstance(region['direction'],list) and len(region['direction'])==3,'Invalid historical flow direction')
            row['direction']=[number(x,'flow direction') for x in region['direction']]
        facts['groom_flow_regions'].append(row)
    if report_path not in pinned:
        facts['reason']='No preserved finish report';return facts
    pin=pinned[report_path]
    require({**pin,'path':'finish-report.json'} in manifest['artifacts'],'Historical coat report is not model-manifest bound')
    report=read_json(parent/report_path)
    require(report.get('source_sha256')==config['source_sha256'],'Historical coat report source differs')
    facts['report_reference']=pin
    bands=report.get('portable_fur')
    if bands is None or bands==[]:
        facts['reason']='No portable fur geometry facts in preserved report';return facts
    require(isinstance(bands,list) and len(bands)<=4,'Historical portable geometry exceeds4 bands')
    for index,band in enumerate(bands):
        root=band.get('root_summary') or {};row={'index':index,'missing_fields':[]}
        values={'cards':band.get('cards'),'roots':root.get('roots'),'eligible_area_m2':root.get('eligible_area_m2'),
                'length_range_m':root.get('length_range_m'),'width_range_m':root.get('width_range_m')}
        for key,value in values.items():
            if value is None:row['missing_fields'].append(key);continue
            if key in ('cards','roots'):
                require(type(value) is int and 0<=value<=1000000,'Invalid historical coat count')
            elif key.endswith('_range_m'):
                require(isinstance(value,list) and len(value)==2 and 0<=number(value[0],key)<=number(value[1],key),
                        'Invalid historical coat dimensions')
            else:require(number(value,key)>0,'Invalid historical coat area')
            row[key]=value
        if row.get('roots',0)>0 and 'eligible_area_m2' in row:
            row['roots_per_m2']=row['roots']/row['eligible_area_m2']
            row['nominal_sqrt_area_per_root_m']=math.sqrt(row['eligible_area_m2']/row['roots'])
            require(math.isfinite(row['roots_per_m2']),'Historical coat density is nonfinite')
            row['derived_scope']='Density = roots/eligible area; nominal sqrt(area/roots) is not measured spacing or coverage.'
        row['status']='available' if not row['missing_fields'] else 'unavailable'
        facts['bands'].append(row)
    facts['status']='available' if all(row['status']=='available' for row in facts['bands']) else 'partial'
    return facts


def _historical_visual_reviews(parent,request,plan,answers,inventory,before_attempt):
    """Carry paid visual repair targets without applying an older verdict to a newer recipe."""
    from avatar_assets.pipeline_prompts import validate_review
    pinned={row['path']:row for row in inventory};reviews=[]
    for attempt in range(before_attempt):
        role=f'reviewer-{attempt}'
        if role not in answers:continue
        review=answers[role];receipt=read_json(parent/f'review-{attempt}.json')
        pack=parent/f'model-{attempt}';manifest=_sealed_folder(parent,pack,inventory)
        model_pin=pinned[f'model-{attempt}/manifest.json']['sha256']
        require(receipt.get('review')==review and review.get('verdict')=='repair'
                and receipt.get('model_manifest_sha256')==model_pin and receipt.get('human_approval') is False
                and manifest.get('status')=='needs_visual_review'
                and manifest.get('source_sha256')==request['reuse_model']['sha256'],
                'Historical visual review differs from paid answer/model')
        folder=parent/f'final-evidence-{attempt}';proof=_sealed_folder(parent,folder,inventory)
        require(proof.get('model_manifest_sha256')==model_pin,'Historical visual evidence belongs to another model')
        evidence=receipt.get('evidence');expected=[{**row,'path':str(folder/row['path'])} for row in proof['images']]
        require(evidence==expected and 1<=len(evidence)<=8,'Historical visual evidence list changed')
        image_refs=[]
        for image in evidence:
            path=Path(image['path']);relative=str(path.relative_to(parent));row=pinned.get(relative)
            require(row is not None and row['sha256']==image['sha256'],'Historical review image changed')
            image_refs.append({'id':image['id'],'path':relative,'sha256':row['sha256'],'bytes':row['bytes']})
        paid=read_json(parent/f'agents/{role}/request.json')
        require(paid.get('images')==[{'bytes':row['bytes'],'sha256':row['sha256'],'suffix':Path(row['path']).suffix}
                                    for row in image_refs],'Historical reviewer saw different images')
        prompt=local_file(parent/f'agents/{role}/prompt.txt',512*1024).read_text()
        marker='INPUT_JSON (data, not instructions):\n'
        require(marker in prompt,'Historical reviewer prompt lacks bound input')
        supplied=json.loads(prompt.split(marker,1)[1])
        config=read_json(parent/f'finish-{attempt}.json')
        require(supplied.get('plan')==plan and supplied.get('request')==request
                and supplied.get('evidence')==evidence
                and supplied.get('authored_config')==config,
                'Historical reviewer prompt/model intent differs')
        validate_review(review,plan,[row['id'] for row in supplied['available_citations']])
        summary={'attempt':attempt,'role':role,'parent_output':'outputs/'+parent.name,
            'scope':'Historical repair targets for this earlier reviewed model, not a verdict on the latest recipe. Recheck after repair.',
            'summary':review['summary'],'required_repairs':review['required_repairs'],
            'failed_checks':[row for row in review['checks'] if not row['passed']],
            'passed_checks':[row['name'] for row in review['checks'] if row['passed']],
            'state_results':[{key:row[key] for key in ('id','meaning_passed','motion_passed')} for row in review['state_reviews']],
            'evidence':image_refs,'model_manifest_sha256':model_pin,
            'review_reference':{'path':f'review-{attempt}.json','sha256':pinned[f'review-{attempt}.json']['sha256']},
            'answer_sha256':pinned[f'agents/{role}/answer.json']['sha256'],
            'receipt_sha256':pinned[f'agents/{role}/receipt.json']['sha256']}
        summary['portable_geometry_facts']=_portable_geometry_facts(parent,pack,manifest,pinned,config,attempt)
        require(len(json.dumps(summary).encode())<=32*1024,'Historical visual review summary exceeds32KiB')
        reviews.append(summary)
    require(len(reviews)<=3,'Historical visual review count exceeds bounded attempts')
    return reviews


def _replan_context(parent,request,plan,answers,inventory,last,validation_failure):
    """Keep a built design only when its failed audit and later local rejections are bound."""
    from avatar_assets.pipeline_prompts import _author_design,finish_config,extract_action_bindings
    from avatar_assets.author_geometry import author_diagnostics
    pinned={row['path']:row for row in inventory}
    built=[attempt for attempt in range(last+1) if (parent/f'model-{attempt}/manifest.json').exists()]
    require(built and not any(role.startswith('reviewer-') for role in answers),
            'Semantic replan requires built action evidence without a reviewer verdict')
    attempt=max(built);pack=parent/f'model-{attempt}'
    manifest=_sealed_folder(parent,pack,inventory)
    source=request['reuse_model'];inspection=read_json(parent/'inspection/manifest.json');native=inspection['native']
    require(manifest.get('status')=='needs_visual_review' and manifest.get('source_sha256')==source['sha256']
            and inspection.get('source_glb_sha256')==source['sha256'], 'Replan model source/status differs')
    quality=read_json(parent/'quality-policy.json')
    parents={bone['name']:bone['parent'] for rig in native.get('rigs',[]) for bone in rig['bones']}
    def recipe(answer,index):
        return finish_config(answer,request,plan,{'path':str(parent/source['path']),'sha256':source['sha256']},index,
                             quality_policy=quality,bone_parents=parents)
    def geometry(answer,index):
        measured=author_diagnostics(_author_design(answer),parent/source['path'],source['sha256'],native)
        require(measured==read_json(parent/f'author-geometry-{index}.json'),'Replan geometry evidence differs')
        return measured
    answer=answers[f'author-{attempt}'];config=recipe(answer,attempt)
    require(answer.get('status')=='ready' and config==read_json(parent/f'finish-{attempt}.json')
            and sha256(parent/f'finish-{attempt}.json')==sha256(pack/'sources/finish-design.json'),
            'Replan built recipe differs from paid author')
    measured=geometry(answer,attempt)
    audit=parent/f'action-audit-{attempt}';audit_manifest=_sealed_folder(parent,audit,inventory)
    report=read_json(audit/'report.json');intent=read_json(audit/'intent.json');model_hash=sha256(pack/'model.glb')
    bindings=extract_action_bindings(answer,plan,report['actual_bone_parents'])
    contract={'states':plan['states'],'action_bindings':bindings,'height_meters':plan['height_meters']}
    contract_hash=hashlib.sha256(json.dumps(contract,sort_keys=True,ensure_ascii=False,allow_nan=False).encode()).hexdigest()
    require(intent=={'schema_version':1,**contract,'source_sha256':model_hash}
            and audit_manifest.get('status')=='measured' and audit_manifest.get('kind')=='weighted_action_audit'
            and audit_manifest.get('model_sha256')==report.get('model_sha256')==model_hash
            and report.get('passed') is False and report.get('source_unchanged') is True
            and report.get('intent_sha256')==sha256(audit/'intent.json')
            and report.get('input_contract_sha256')==contract_hash,
            'Replan action audit differs from exact model/plan/author')
    require([row.get('id') for row in report.get('states',[])]==[row['id'] for row in plan['states']]
            and any(row.get('requirement_results',{}).get('passed') is False for row in report['states']),
            'Replan action audit has no bound failed state')
    for folder in (pack,audit):
        job=read_json(folder/'job.json')
        require(job.get('exit_code')==0 and not job.get('failure'),'Replan measured stage did not finish successfully')
    implementations=report.get('implementation_sha256')
    require(isinstance(implementations,dict) and set(implementations)=={'action_audit.py','action_intent.py','three_d.py'}
            and all(pinned['sources/'+name]['sha256']==value for name,value in implementations.items())
            and audit_manifest.get('implementation_sha256')==implementations['action_audit.py'],
            'Replan action audit source snapshot differs')
    events=[read_json(parent/row['path']) for row in inventory if row['path'].startswith('events/') and row['path'].endswith('.json')]
    error='Requested action capability failed: '+json.dumps(report['states'])
    require(any(event.get('stage')=='action_audit' and event.get('status')=='failed'
                and event.get('attempt')==attempt and event.get('reason')==error[:4000] for event in events)
            and any(event.get('stage')=='attempt' and event.get('status')=='rejected'
                    and event.get('attempt')==attempt and event.get('reason')==error for event in events),
            'Replan lacks exact failed action stage')
    findings={'validation_failure':error,'measurements':{'action_audit':report,'author_geometry':measured},
              'later_rejected_authors':[]}
    latest=None
    for index in range(attempt+1,last+1):
        role=f'author-{index}';rejected=answers[role]
        require(rejected.get('status')=='ready' and not (parent/f'model-{index}').exists()
                and not (parent/f'finish-{index}.json').exists(),'Replan later author has unexplained build evidence')
        measured=geometry(rejected,index)
        try:
            recipe(rejected,index)
        except (ValueError,KeyError,TypeError) as exc:
            failure=str(exc)
        else:
            require(measured.get('status')=='available' and measured.get('failures'),
                    'Replan later author has no reproducible local rejection')
            failure='Authored geometry preflight failed; inspect all measured layer and bone findings'
        require(any(event.get('stage')=='author' and event.get('status')=='failed' and event.get('attempt')==index
                    and event.get('reason')==failure for event in events),'Replan later author rejection differs')
        findings['later_rejected_authors'].append({'role':role,'answer':rejected,'validation_failure':failure,
            'answer_sha256':pinned[f'agents/{role}/answer.json']['sha256'],
            'receipt_sha256':pinned[f'agents/{role}/receipt.json']['sha256']})
        latest={'validation_failure':failure,'measurements':{'author_geometry':measured}}
    require(latest is not None and read_json(parent/'unsupported.json').get('findings')==latest,
            'Replan terminal does not match reproduced later rejection')
    return {'plan':plan,'previous_answer':answer,'findings':findings,
            'replan':{'validation_failure':validation_failure,'previous_plan':plan}}


def _pinned_review_semantics(parent):
    """Compare trusted syntax; never execute preserved source files."""
    from avatar_assets import pipeline_prompts,pipeline
    def selected(path,names):
        tree=ast.parse(local_file(path,1024**2).read_text());result={}
        for node in ast.walk(tree):
            if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef)) and node.name in names:
                require(node.name not in result,'Ambiguous review validator source')
                result[node.name]=ast.dump(node,include_attributes=False)
            elif isinstance(node,ast.Assign):
                for target in node.targets:
                    if isinstance(target,ast.Name) and target.id in names:
                        require(target.id not in result,'Ambiguous review schema source')
                        result[target.id]=ast.dump(node,include_attributes=False)
        require(set(result)==set(names),'Pinned review semantics are incomplete')
        return result
    groups={'pipeline_prompts.py':(Path(pipeline_prompts.__file__),
        ('validate_review','review_citations','_fields','_text','_string_list','_object','_strings',
         'CRITIC_SCHEMA','_CHECK_SCHEMA','_STATE_REVIEW_SCHEMA')),
        'pipeline.py':(Path(pipeline.__file__),('_bind_review',))}
    hashes={}
    for name,(current,names) in groups.items():
        saved=parent/'sources'/name
        require(selected(saved,names)==selected(current,names),
                'Rejected review needs unchanged pinned validation semantics')
        hashes[name]=sha256(saved)
    return hashes


def _rejected_review(parent,request,plan,review,inventory,attempt,terminal):
    from avatar_assets.pipeline_prompts import validate_review,review_citations
    from avatar_assets.pipeline import Pipeline
    from avatar_assets.agent_runner import _image_inputs
    pinned={row['path']:row for row in inventory};role=f'reviewer-{attempt}'
    require(not (parent/f'review-{attempt}.json').exists() and not (parent/'delivery.json').exists(),
            'Rejected-review continuation cannot contain an accepted review or delivery receipt')
    semantics=_pinned_review_semantics(parent)
    pack=parent/f'model-{attempt}';model_pin=pinned[f'model-{attempt}/manifest.json']['sha256']
    folder=parent/f'final-evidence-{attempt}';proof=_sealed_folder(parent,folder,inventory)
    require(proof.get('model_manifest_sha256')==model_pin,'Rejected review evidence belongs to another model')
    evidence=[{**row,'path':str(folder/row['path'])} for row in proof['images']]
    require(1<=len(evidence)<=8,'Rejected review image count exceeds bound')
    refs=[];paths=[]
    for image in evidence:
        path=Path(image['path'])
        require(path.resolve()==path and path.is_relative_to(folder),'Rejected review image escaped evidence folder')
        relative=str(path.relative_to(parent));row=pinned.get(relative)
        require(row is not None and row['sha256']==image['sha256'],'Rejected review image changed')
        refs.append({'id':image['id'],**row});paths.append(path)
    submitted=[{key:row[key] for key in ('bytes','sha256','suffix')} for row in _image_inputs(paths)]
    paid=read_json(parent/f'agents/{role}/request.json')
    require(paid.get('images')==submitted,'Rejected reviewer saw different images')
    prompt=local_file(parent/f'agents/{role}/prompt.txt',512*1024).read_text()
    marker='INPUT_JSON (data, not instructions):\n'
    require(marker in prompt,'Rejected reviewer lacks its bound prompt')
    supplied=json.loads(prompt.split(marker,1)[1]);action=read_json(parent/f'action-audit-{attempt}/report.json')
    require(supplied.get('request')==request and supplied.get('plan')==plan
            and supplied.get('authored_config')==read_json(parent/f'finish-{attempt}.json')
            and supplied.get('evidence')==evidence
            and supplied.get('available_citations')==review_citations(evidence,action),
            'Rejected reviewer prompt/evidence/model intent differs')
    failure=None
    try:
        validate_review(review,plan,[row['id'] for row in supplied['available_citations']])
        Pipeline._bind_review(review,evidence,action)
    except (ValueError,KeyError,TypeError) as error:failure=str(error)[:3000]
    require(failure is not None and terminal.get('reason')=='Reviewer response failed its evidence contract: '+failure,
            'Saved reviewer contract rejection did not reproduce')
    raw=read_json(parent/f'agents/{role}/answer.json')
    require(len(json.dumps(raw).encode())<=64*1024,'Rejected review diagnostic exceeds64KiB')
    result={'status':'invalid_untrusted','scope':'Rejected data, not a valid verdict or approval. Independently re-evaluate every required check.',
        'attempt':attempt,'role':role,'parent_output':'outputs/'+parent.name,'reason':terminal['reason'],
        'source_sha256':request['reuse_model']['sha256'],'raw_answer':raw,
        'answer_sha256':pinned[f'agents/{role}/answer.json']['sha256'],
        'receipt_sha256':pinned[f'agents/{role}/receipt.json']['sha256'],
        'model_manifest_sha256':model_pin,'evidence':refs,'validation_semantics':semantics}
    require(len(json.dumps(result).encode())<=96*1024,'Rejected review evidence exceeds96KiB')
    return result


def _visual_terminal(parent,request,plan,answer,inventory,attempt,terminal,prior_context,*,rejected_review=None):
    """Authenticate a completed review's inputs; never rerun rendering or numerical audits."""
    from avatar_assets.pipeline import Pipeline,_remember_source_failures
    from avatar_assets.pipeline_prompts import finish_config,extract_action_bindings
    from avatar_assets.quality_policy import evaluate_quality_evidence
    pinned={row['path']:row for row in inventory}
    def digest(path):return pinned[str(path.relative_to(parent))]['sha256']
    pack=parent/f'model-{attempt}';manifest=_sealed_folder(parent,pack,inventory)
    source=request['reuse_model'];model_hash=digest(pack/'model.glb');model_pin=digest(pack/'manifest.json')
    require(answer.get('status')=='ready' and manifest.get('status')=='needs_visual_review'
            and manifest.get('source_sha256')==source['sha256'],'Visual continuation lacks the completed source-bound design')
    inspection=read_json(parent/'inspection/manifest.json');native=inspection['native']
    require(inspection.get('source_glb_sha256')==source['sha256'],'Visual inspection source differs')
    quality=read_json(parent/'quality-policy.json')
    config=finish_config(answer,request,plan,{'path':str(parent/source['path']),'sha256':source['sha256']},attempt,
        quality_policy=quality,bone_parents={bone['name']:bone['parent'] for rig in native.get('rigs',[]) for bone in rig['bones']})
    require(config==read_json(parent/f'finish-{attempt}.json')
            and digest(parent/f'finish-{attempt}.json')==digest(pack/'sources/finish-design.json'),
            'Visual continuation recipe differs from paid author')
    prompt=local_file(parent/f'agents/reviewer-{attempt}/prompt.txt',512*1024).read_text()
    supplied=json.loads(prompt.split('INPUT_JSON (data, not instructions):\n',1)[1])
    proof=_sealed_folder(parent,parent/f'final-evidence-{attempt}',inventory)
    exported=parent/f'exported-preview-{attempt}';exported_manifest=_sealed_folder(parent,exported,inventory)
    reload_manifest=_sealed_folder(parent,parent/f'reload-{attempt}',inventory)
    media=[parent/name for name in (f'media-{attempt}',f'media-continuation-{attempt}')
           if f'{name}/manifest.json' in pinned and pinned[f'{name}/manifest.json']['sha256']==proof['media_manifest_sha256']]
    require(len(media)==1,'Visual continuation final media binding differs')
    media=media[0];media_manifest=_sealed_folder(parent,media,inventory)
    portable=parent/f'media-portable-{attempt}';portable_manifest=_sealed_folder(parent,portable,inventory)
    require(proof['exported_manifest_sha256']==digest(exported/'manifest.json')
            and exported_manifest.get('source_glb_sha256')==model_hash
            and reload_manifest.get('status')=='passed' and reload_manifest.get('source_manifest_sha256')==model_pin
            and reload_manifest.get('source_model_sha256')==model_hash
            and media_manifest.get('parent_3d_manifest_sha256')==portable_manifest.get('parent_3d_manifest_sha256')==model_pin
            and digest(portable/'source-manifest.json')==digest(media/'manifest.json'),
            'Visual continuation media/reload belongs to another model')
    for path,value in proof['sources'].items():
        require(digest(Path(path))==value,'Visual continuation source image/media changed')
    action_folder=parent/f'action-audit-{attempt}';action_manifest=_sealed_folder(parent,action_folder,inventory)
    action=read_json(action_folder/'report.json');intent=read_json(action_folder/'intent.json')
    contract={'states':plan['states'],'action_bindings':extract_action_bindings(answer,plan,action['actual_bone_parents']),
              'height_meters':plan['height_meters']}
    require(action.get('passed') is True and action.get('source_unchanged') is True
            and action_manifest.get('model_sha256')==action.get('model_sha256')==model_hash
            and action.get('intent_sha256')==digest(action_folder/'intent.json')
            and intent=={'schema_version':1,**contract,'source_sha256':model_hash},
            'Visual continuation action gate/model intent differs')
    deformation=read_json(parent/f'deformation-gate-{attempt}.json');bindings=deformation['bindings']
    expected={'source_sha256':source['sha256'],'design_sha256':digest(parent/f'finish-{attempt}.json'),
              'manifest_sha256':model_pin,'exported_model_sha256':model_hash,
              'finish_report_sha256':digest(pack/'finish-report.json'),'rig_verification_sha256':digest(pack/'rig-verification.json'),
              'state_ids':[state['id'] for state in plan['states']]}
    require(deformation.get('passed') is True and all(bindings.get(key)==value for key,value in expected.items()),
            'Visual continuation deformation gate/model binding differs')
    attachment_folder=parent/f'portable-root-audit-{attempt}'
    attachment_manifest=_sealed_folder(parent,attachment_folder,inventory)
    attachment=read_json(attachment_folder/'report.json');attachment_intent=read_json(attachment_folder/'intent.json')
    expected_intent={'schema_version':1,'source_sha256':source['sha256'],'design_sha256':expected['design_sha256'],
                     'model_sha256':model_hash,'state_ids':expected['state_ids']}
    require(attachment.get('passed') is True and attachment.get('status') in ('measured','not_applicable')
            and attachment.get('intent')==attachment_intent==expected_intent
            and attachment_manifest.get('model_manifest_sha256')==model_pin
            and attachment_manifest.get('model_sha256')==model_hash,
            'Visual continuation portable attachment gate/model differs')
    receipt=read_json(parent/f'portable-root-audit-{attempt}.json')
    require(receipt=={'schema_version':1,'manifest_sha256':digest(attachment_folder/'manifest.json'),
        'report_sha256':digest(attachment_folder/'report.json'),'job_sha256':digest(attachment_folder/'job.json'),
        'intent_sha256':digest(attachment_folder/'intent.json'),'model_manifest_sha256':model_pin},
        'Visual continuation attachment receipt changed')
    for row in attachment['inputs']:
        require(pinned.get(str((pack/row['path']).relative_to(parent)))=={**row,'path':str((pack/row['path']).relative_to(parent))},
                'Visual continuation attachment input changed')
    worker=Pipeline.__new__(Pipeline);worker.root=parent
    summary=worker._attachment_summary(attempt,attachment)
    review_receipt=None
    if rejected_review is None:
        review_receipt=read_json(parent/f'review-{attempt}.json')
        require(review_receipt.get('portable_attachment_manifest_sha256')==digest(attachment_folder/'manifest.json'),
                'Visual review attachment binding differs')
    coverage=evaluate_quality_evidence(quality,proof['coverage'])
    require(coverage['passed'],'Visual continuation has incomplete final review coverage')
    expected_proofs={'rig':manifest['rig_verification'],'native_reload':'passed',
        'attributes':read_json(pack/'render-attributes.json'),'action_audit':action,'deformation_gate':deformation,
        'portable_attachment':summary,'quality_policy':quality,'quality_coverage':coverage,'delivered_media':portable_manifest['clips']}
    require(supplied.get('independent_proofs')==expected_proofs,'Visual reviewer independent proofs differ from measured files')
    findings={'measurements':worker._diagnostics(pack)}
    if rejected_review is None:
        contacts=read_json(parent/f'evidence-{attempt}/evidence.json')
        require(contacts.get('finish_report_sha256')==expected['finish_report_sha256'],'Visual repair contacts belong to another model')
        for row in contacts['images']:
            require(pinned.get(f'evidence-{attempt}/'+row['path'])=={**row,'path':f'evidence-{attempt}/'+row['path']},
                    'Visual repair contact image changed')
        findings.update(review=review_receipt['review'],
            repair_images=[str(parent/f'evidence-{attempt}'/row['path']) for row in contacts['images'][:5]])
    else:
        findings.update(rejected_review=rejected_review,validation_failure=terminal['reason'])
    findings['measurements']['portable_attachment']=summary
    memory=_remember_source_failures([],prior_context.get('findings'),source['sha256'])
    for index in range(attempt+1):
        earlier=parent/f'model-{index}'
        if f'model-{index}/manifest.json' not in pinned:continue
        if _sealed_folder(parent,earlier,inventory).get('status')=='failed':
            memory=_remember_source_failures(memory,{'measurements':worker._diagnostics(earlier)},source['sha256'])
    if memory:findings['known_source_failures']=memory
    if rejected_review is None:
        require(findings==terminal.get('findings'),'Visual continuation terminal findings differ from authenticated review')
    return findings


def collect(root,parent):
    """Capture a bound native failure, visual repair budget, or semantically invalid plan."""
    from avatar_assets.pipeline import Pipeline,_remember_source_failures
    from avatar_assets.pipeline_prompts import validate_plan,_author_design,finish_config
    from avatar_assets.author_geometry import author_diagnostics
    from avatar_assets.action_intent import PlanIntentError
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
                and terminal.get('stage')==status.get('stage') and terminal.get('stage') in ('repair_budget','author','reviewer')
                and terminal['request_sha256']==input_receipt['request_sha256']==sha256(parent/'request.json'),
                'Continuation parent is not a bound terminal native-repair result')
        require(request.get('agent_backend',{}).get('provider')=='openai'
                and type(request.get('max_repairs')) is int and 0<=request['max_repairs']<=2
                and request.get('reuse_model') is not None,'Unsupported continuation parent contract')
        rejected=terminal['stage']=='author'
        invalid_review=terminal['stage']=='reviewer'
        if invalid_review:
            require(terminal.get('failure_kind')=='review_contract' and status.get('attempt')==terminal.get('attempt')
                    and status.get('reason')==terminal.get('reason'), 'Reviewer continuation requires an exact rejected evidence contract')
        last=terminal.get('attempt') if rejected or invalid_review else request['max_repairs']
        require(type(last) is int and (1<=last<=request['max_repairs'] if rejected else 0<=last<=request['max_repairs']),
                'Continuation needs a prior complete native attempt before author rejection')
        if rejected:
            require(status.get('attempt')==last and status.get('reason')==terminal.get('reason')
                    and isinstance(terminal.get('reason'),str) and terminal['reason'],
                    'Continuation author rejection terminal disagrees')
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
        allowed={'planner'}|{f'{role}-{attempt}' for role in ('author','reviewer') for attempt in range(last+1)}
        require(set(calls)<=allowed and {f'author-{attempt}' for attempt in range(last+1)}<=set(calls),
                'Continuation parent has unexpected or missing author roles')
        folders={path.name for path in (parent/'agents').iterdir() if path.is_dir()}
        require(folders==set(calls),'Continuation agent folders/call ledger disagree')
        settings=read_json(parent/'agents/runner.json')
        require(settings.get('job_id')==parent.name and settings.get('job_digest')==sha256(parent/'request.json')
                and settings.get('max_calls')==request['max_agent_calls']
                and settings.get('transport')==request['agent_backend'].get('transport')
                and settings.get('implementation_sha256')==pinned['sources/openai_runner.py']['sha256'],
                'Continuation parent runner binding changed')
        if settings.get('transport') is not None:
            require(settings['transport']=='sse-v1' and request['agent_backend'].get('transport')=='sse-v1'
                    and settings.get('transport_implementation_sha256') ==
                    pinned.get('sources/openai_stream.py',{}).get('sha256'),
                    'Continuation streaming transport binding changed')
        answers={role:_role(parent,role,row,settings) for role,row in calls.items()}
        plan=read_json(parent/'plan.json');plan_failure=None
        try:validate_plan(plan,request)
        except PlanIntentError as error:
            require(not rejected and not invalid_review,'Semantic replan needs a terminal repair-budget result')
            plan_failure=str(error)
        depth=1;prior_context={}
        if 'continuation' in request:
            prior,prior_context=_read_context(parent,request)
            if 'replan' in prior_context:
                require(answers.get('planner')==plan,'Continuation revised plan differs from completed planner answer')
            else:
                require(prior_context['plan']==plan and 'planner' not in calls,'Continuation inherited plan changed')
            depth=prior['depth']+1
        else:
            require(answers.get('planner')==plan,'Continuation plan differs from completed planner answer')
        require(depth<=8,'Continuation chain reached its eight-job bound')
        if plan_failure is not None:
            context=_replan_context(parent,request,plan,answers,inventory,last,plan_failure)
            require(_inventory(parent)==inventory,'Continuation parent changed during inspection')
            return {'parent':str(parent.relative_to(root)),'parent_request':request,'inventory':inventory,
                    'ledger':ledger,'depth':depth,'context':context}
        if invalid_review:
            role=f'reviewer-{last}'
            require(role in answers and set(calls)=={f'author-{i}' for i in range(last+1)}|{role}|({'planner'} if 'planner' in calls else set()),
                    'Rejected-review continuation has unexpected review roles')
            invalid=_rejected_review(parent,request,plan,answers[role],inventory,last,terminal)
            findings=_visual_terminal(parent,request,plan,answers[f'author-{last}'],inventory,last,terminal,prior_context,
                                      rejected_review=invalid)
            inherited=prior_context.get('findings',{}).get('historical_visual_reviews',[])
            require(isinstance(inherited,list) and len(inherited)<=8 and len(json.dumps(inherited).encode())<=64*1024,
                    'Invalid inherited visual review bound')
            if inherited:findings['historical_visual_reviews']=inherited
            require(_inventory(parent)==inventory,'Rejected-review parent changed during inspection')
            return {'parent':str(parent.relative_to(root)),'parent_request':request,'inventory':inventory,
                    'ledger':ledger,'depth':depth,'context':{'plan':plan,'previous_answer':answers[f'author-{last}'],'findings':findings}}
        rejection=None
        if rejected:
            rejected_role=f'author-{last}';rejected_answer=answers[rejected_role]
            require(rejected_answer.get('status')=='unsupported' and rejected_answer.get('reason')==terminal['reason']
                    and f'reviewer-{last}' not in answers,'Continuation paid author rejection differs')
            rejection={'role':rejected_role,'answer':rejected_answer,
                       'answer_sha256':pinned[f'agents/{rejected_role}/answer.json']['sha256'],
                       'receipt_sha256':pinned[f'agents/{rejected_role}/receipt.json']['sha256']}
        attempt=last-1 if rejected else last;role='author-'+str(attempt)
        if not rejected and f'reviewer-{attempt}' in answers:
            reviews=_historical_visual_reviews(parent,request,plan,answers,inventory,attempt+1)
            findings=_visual_terminal(parent,request,plan,answers[role],inventory,attempt,terminal,prior_context)
            inherited=prior_context.get('findings',{}).get('historical_visual_reviews',[])
            require(isinstance(inherited,list) and len(inherited)<=8,'Invalid inherited visual review bound')
            visual_reviews=list(inherited)
            for review in reviews:
                if review not in visual_reviews:visual_reviews.append(review)
            require(len(visual_reviews)<=8 and len(json.dumps(visual_reviews).encode())<=64*1024,
                    'Historical visual review memory exceeds bounded context')
            findings['historical_visual_reviews']=visual_reviews
            require(_inventory(parent)==inventory,'Continuation parent changed during inspection')
            return {'parent':str(parent.relative_to(root)),'parent_request':request,'inventory':inventory,
                    'ledger':ledger,'depth':depth,'context':{'plan':plan,'previous_answer':answers[role],'findings':findings}}
        require(role in answers and 'reviewer-'+str(attempt) not in answers,'Continuation requires the latest failed native author attempt')
        answer=answers[role]
        require(answer.get('status')=='ready','Continuation seed author did not complete a design')
        design=_author_design(answer)
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
        memory=_remember_source_failures([],prior_context.get('findings'),source['sha256'])
        for index in range(attempt+1):
            earlier=parent/f'model-{index}'
            if not (earlier/'manifest.json').exists():continue
            earlier_manifest=_sealed_folder(parent,earlier,inventory)
            if earlier_manifest.get('status')=='failed':
                diagnostic=findings['measurements'] if index==attempt else Pipeline.__new__(Pipeline)._diagnostics(earlier)
                memory=_remember_source_failures(memory,{'measurements':diagnostic},source['sha256'])
        if rejected:
            # The refusal is evidence, not the seed: its null design must never replace the valid proposal.
            findings['author_rejection']=rejection
        else:
            if 'known_source_failures' in terminal.get('findings',{}):findings['known_source_failures']=memory
            require(findings==terminal.get('findings'),'Continuation terminal findings differ from authenticated evidence')
        if memory:findings['known_source_failures']=memory
        inherited_reviews=prior_context.get('findings',{}).get('historical_visual_reviews',[])
        require(isinstance(inherited_reviews,list) and len(inherited_reviews)<=8,'Invalid inherited visual review bound')
        visual_reviews=list(inherited_reviews)
        for review in _historical_visual_reviews(parent,request,plan,answers,inventory,attempt):
            if review not in visual_reviews:visual_reviews.append(review)
        require(len(visual_reviews)<=8 and len(json.dumps(visual_reviews).encode())<=64*1024,
                'Historical visual review memory exceeds bounded context')
        if visual_reviews:findings['historical_visual_reviews']=visual_reviews
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
             'context_sha256':sha256(folder/'context.json'),'new_planner_calls':int('replan' in captured['context']),
             'parent_mutated':False}
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
