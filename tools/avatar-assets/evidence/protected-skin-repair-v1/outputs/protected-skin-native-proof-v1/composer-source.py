"""Explicit, provenance-bound finishing/root-translation patches over a passing rig."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re

from avatar_assets.author_contract import AUTHOR_SCHEMA_V2, AUTHOR_SCHEMA_V2_SKIN_CONSISTENT, decode_typed_author
from avatar_assets.agent_runner import _validate
from avatar_assets.safety import local_file, read_json, require, sha256, write_json

POLICY = 'protected-skin-root-translation-v1'
PROTECTED = frozenset(('authored_rig','weight_constraints','part_rigs','attachment','action_bindings',
                       'smoothing','surface_smoothing','material_fur','portable_fringe'))
FINISH_FIELDS = tuple(key for key in AUTHOR_SCHEMA_V2['$defs']['design']['properties']
                      if key not in PROTECTED and key != 'authored_actions')
MAX_BYTES = 128*1024


def digest(value):
    raw=json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
    require(len(raw)<=256*1024,'Protected repair data exceeds bound')
    return hashlib.sha256(raw).hexdigest()


def _object(properties):
    return {'type':'object','properties':properties,'required':list(properties),'additionalProperties':False}


def protected_projection(answer,root_bone,root_state_ids=()):
    design=decode_typed_author(answer);require(design is not None,'Protected baseline needs a ready design')
    result={key:deepcopy(value) for key,value in design.items() if key not in FINISH_FIELDS and key!='authored_actions'}
    actions=deepcopy(design['authored_actions'])
    for action in actions:
        for key in action['keys']:
            translations=key.get('translations_m',{})
            if action['id'] in root_state_ids:translations.pop(root_bone,None)
            key['translations_m']=translations
    result['authored_actions']=actions
    return result


def patch_schema(baseline):
    answer=baseline['baseline_answer'];decode_typed_author(answer)
    definitions=deepcopy(AUTHOR_SCHEMA_V2_SKIN_CONSISTENT['$defs'])
    properties=definitions['design']['properties']
    updates=_object({name:properties[name] for name in FINISH_FIELDS})
    variants=[]
    allowed_states={r['state_id'] for r in baseline['failed_root_trajectory_requirements']}
    for action in answer['design']['authored_actions']:
        if action['id'] not in allowed_states:continue
        count=len(action['keys'])
        variants.append(_object({'state_id':{'type':'string','const':action['id']},
            'values':{'type':'array','minItems':count,'maxItems':count,'items':
                      {'type':'array','minItems':3,'maxItems':3,'items':{'type':'number','minimum':-2,'maximum':2}}}}))
    translations={'anyOf':[{'type':'array','maxItems':len(variants),'items':{'anyOf':variants}}, {'type':'null'}]} if variants else {'type':'null'}
    patch=_object({'finish_updates':updates,
        'remove_finish_fields':{'type':'array','maxItems':len(FINISH_FIELDS),'items':{'type':'string','enum':list(FINISH_FIELDS)}},
        'root_translations':translations})
    schema=_object({'schema_version':{'type':'integer','const':3},'status':{'type':'string','enum':['ready','unsupported']},
        'reason':{'type':'string','minLength':0,'maxLength':4000},'patch':{'anyOf':[patch,{'type':'null'}]}})
    # Only finishing definitions belong to this transport; protected rig fields are not editable.
    needed=set()
    def refs(node):
        if isinstance(node,dict):
            if '$ref' in node:
                name=node['$ref'].split('/')[-1]
                if name not in needed:needed.add(name);refs(definitions[name])
            for value in node.values():refs(value)
        elif isinstance(node,list):
            for value in node:refs(value)
    refs(schema)
    if needed:schema['$defs']={key:value for key,value in definitions.items() if key in needed}
    return schema


def compose_patch(raw,baseline):
    _validate(raw,patch_schema(baseline));require(len(json.dumps(raw).encode())<=MAX_BYTES,'Repair patch exceeds128KiB')
    if raw['status']=='unsupported':
        require(raw['patch'] is None and raw['reason'].strip(),'Unsupported patch needs reason and patch:null')
        return {'schema_version':2,'status':'unsupported','reason':raw['reason'],'design':None}
    require(raw['patch'] is not None,'Ready repair needs a patch')
    original=baseline['baseline_answer'];effective=deepcopy(original);effective['reason']=raw['reason']
    patch=raw['patch'];remove=patch['remove_finish_fields']
    require(len(remove)==len(set(remove)),'Duplicate removed finishing field')
    for name,value in patch['finish_updates'].items():
        require(name not in remove or value is None,'A finishing field cannot be updated and removed')
        if value is not None:effective['design'][name]=deepcopy(value)
    for name in remove:effective['design'][name]=None
    by_id={a['id']:a for a in effective['design']['authored_actions']};seen=set()
    for change in patch['root_translations'] or []:
        state=change['state_id'];require(state not in seen,'Duplicate root translation state');seen.add(state)
        action=by_id[state]
        require(action.get('interpolation')=='LINEAR','Root translation patch needs already-LINEAR keys')
        for key,xyz in zip(action['keys'],change['values']):
            entries=deepcopy(key.get('translations_m') or [])
            matches=[row for row in entries if row['bone']==baseline['root_bone']]
            require(len(matches)<=1,'Duplicate baseline root translation')
            if matches:matches[0]['xyz']=deepcopy(xyz)
            else:entries.append({'bone':baseline['root_bone'],'xyz':deepcopy(xyz)})
            key['translations_m']=entries
    decode_typed_author(effective)
    allowed_states={r['state_id'] for r in baseline['failed_root_trajectory_requirements']}
    require(protected_projection(effective,baseline['root_bone'],allowed_states)==protected_projection(original,baseline['root_bone'],allowed_states),
            'Patch changed protected skin, geometry, bindings or non-root motion')
    return effective


def eligible_scope(answer,review,plan):
    """Only measured-passing skin with appearance or whole-body translation defects qualifies."""
    if answer.get('schema_version')!=2 or answer.get('status')!='ready':return None
    design=decode_typed_author(answer);rig=design.get('authored_rig')
    if not isinstance(rig,dict):return None
    roots=[bone['name'] for bone in rig['bones'] if bone['parent'] is None]
    if len(roots)!=1:return None
    root=roots[0];checks={row['name']:row for row in review.get('checks',[])}
    if review.get('verdict')!='repair' or set(checks)!={'identity','appearance','framing','deformation'}:return None
    if any(checks[key].get('passed') is not True for key in ('identity','framing','deformation')):return None
    states={row['id']:row for row in plan['states']};bindings={row['state_id']:row for row in design.get('action_bindings',[])}
    reviews=review.get('state_reviews',[])
    if len(reviews)!=len(states) or {row['id'] for row in reviews}!=set(states):return None
    failed=[]
    for row in reviews:
        if row.get('motion_passed') is not True:return None
        requirements={r['id']:r for r in states[row['id']].get('requirements',[])}
        roles={r['id']:r for r in bindings.get(row['id'],{}).get('roles',[])}
        rejected=[r for r in row.get('requirement_reviews',[]) if r.get('passed') is False]
        if row.get('meaning_passed') is not True and not rejected:return None
        for rejection in rejected:
            requirement=requirements.get(rejection['id'],{})
            if (requirement.get('kind') not in ('travel','rhythm','return_pose')
                    or roles.get(requirement.get('role'),{}).get('root_bone')!=root):return None
            failed.append({'state_id':row['id'],'requirement_id':rejection['id'],'kind':requirement['kind']})
    if checks['appearance'].get('passed') is True and not failed:return None
    actions=design['authored_actions']
    if {a['id'] for a in actions}!=set(states) or any(a.get('interpolation')!='LINEAR' for a in actions if a['id'] in {r['state_id'] for r in failed}):return None
    return {'root_bone':root,'allowed_scope':'finishing_and_existing_key_global_root_translation',
            'failed_root_trajectory_requirements':failed,'fixed_skin_and_articulation':True}


def _paid_answer(parent,role,record):
    """Budget custody remains with the caller; independently bind provider text and prompt bytes."""
    from avatar_assets.agent_runner import _call_artifacts
    from avatar_assets.openai_runner import _decode,response_answer_text
    folder=parent/'agents'/role
    for name in ('receipt.json','request.json','answer.json','schema.json','prompt.txt','response.json'):
        record(f'agents/{role}/'+name,4*1024*1024 if name=='response.json' else 512*1024)
    receipt=read_json(folder/'receipt.json');request=read_json(folder/'request.json')
    answer=read_json(folder/'answer.json');schema=read_json(folder/'schema.json')
    require(receipt.get('status')=='succeeded' and receipt.get('execution_kind')=='openai_responses_api'
            and receipt.get('artifacts')==_call_artifacts(folder)
            and receipt.get('answer_sha256')==sha256(folder/'answer.json')
            and request.get('role')==role and request.get('prompt_sha256')==sha256(folder/'prompt.txt'),
            'Protected baseline paid response custody changed')
    response=_decode((folder/'response.json').read_bytes())
    settings=read_json(parent/'agents/runner.json');record('agents/runner.json')
    require(response.get('status')=='completed' and response.get('id')==receipt.get('response_id')
            and response.get('model')==settings['model'] and response.get('usage')==receipt.get('usage'),
            'Protected baseline provider result differs')
    require(_decode(response_answer_text(response,reasoning_effort=settings.get('reasoning',{}).get('effort','none')).encode())==answer,
            'Protected baseline answer differs from provider text')
    _validate(answer,schema)
    return answer


def _baseline_data(root,parent,attempt,source_sha,plan,*,depth=0):
    from avatar_assets import pipeline_prompts as prompts
    from avatar_assets import repair_visual_inputs
    require(depth<=8,'Protected composition ancestry exceeds8')
    require(parent.parent==root/'outputs' and parent.resolve()==parent,'Protected baseline is outside output ancestry')
    refs={}
    def record(name,maximum=1024*1024):
        relative=Path(name);require(not relative.is_absolute() and '..' not in relative.parts,'Protected evidence path escaped')
        path=local_file(parent/relative,maximum)
        require(path==parent/relative,'Protected evidence path is linked')
        refs[name]={'bytes':path.stat().st_size,'sha256':sha256(path)}
        return path
    def load(name):return read_json(record(name))
    request=load('request.json');saved_plan=load('plan.json')
    require(saved_plan==plan and request.get('reuse_model',{}).get('sha256')==source_sha,
            'Protected baseline plan or source differs')
    raw=_paid_answer(parent,f'author-{attempt}',record)
    answer=verify_materialized(parent,attempt,raw,_depth=depth+1) if raw.get('schema_version')==3 else raw
    if answer.get('schema_version')!=2:return None
    raw_review=_paid_answer(parent,f'reviewer-{attempt}',record)
    review=prompts.decode_typed_review(raw_review) if raw_review.get('schema_version')==2 else raw_review
    receipt=load(f'review-{attempt}.json');pack=parent/f'model-{attempt}'
    manifest=load(f'model-{attempt}/manifest.json')
    require(receipt['review']==review and receipt.get('human_approval') is False
            and receipt['model_manifest_sha256']==sha256(pack/'manifest.json')
            and manifest.get('status')=='needs_visual_review' and manifest.get('source_sha256')==source_sha,
            'Protected baseline review/model differs')
    repair_visual_inputs._review_evidence(parent,source_sha,attempt,review,receipt['evidence'])
    config=load(f'finish-{attempt}.json');snapshot=load(f'model-{attempt}/sources/finish-design.json')
    require(config==snapshot and sha256(parent/f'finish-{attempt}.json')==sha256(pack/'sources/finish-design.json'),
            'Protected baseline recipe snapshot differs')
    artifacts_by_path={row['path']:row for row in manifest.get('artifacts',[])}
    for name in ('sources/finish-design.json','finish-report.json','rig-verification.json','model.glb'):
        path=record(f'model-{attempt}/'+name,64*1024*1024 if name=='model.glb' else 1024*1024)
        require(artifacts_by_path.get(name)=={'path':name,'bytes':path.stat().st_size,'sha256':sha256(path)},
                'Protected baseline model artifact changed')
    quality=load('quality-policy.json');native=load('inspection/manifest.json')['native']
    source={'path':str(parent/request['reuse_model']['path']),'sha256':source_sha}
    require(prompts.finish_config(answer,request,plan,source,attempt,quality_policy=quality,
        bone_parents={b['name']:b['parent'] for rig in native.get('rigs',[]) for b in rig['bones']})==config,
        'Protected baseline paid design differs from finished recipe')
    deformation=load(f'deformation-gate-{attempt}.json');bindings=deformation.get('bindings',{})
    require(deformation.get('passed') is True and all(bindings.get(k)==v for k,v in {
        'source_sha256':source_sha,'design_sha256':sha256(parent/f'finish-{attempt}.json'),
        'manifest_sha256':sha256(pack/'manifest.json'),'exported_model_sha256':sha256(pack/'model.glb'),
        'finish_report_sha256':sha256(pack/'finish-report.json'),'rig_verification_sha256':sha256(pack/'rig-verification.json'),
        'state_ids':[s['id'] for s in plan['states']]}.items()), 'Protected baseline lacks bound passing deformation')
    action=load(f'action-audit-{attempt}/report.json');attachment=load(f'portable-root-audit-{attempt}/report.json')
    require(action.get('passed') is True and action.get('source_unchanged') is True
            and action.get('model_sha256')==sha256(pack/'model.glb')
            and attachment.get('passed') is True and attachment.get('intent',{}).get('source_sha256')==source_sha
            and attachment['intent'].get('model_sha256')==sha256(pack/'model.glb'),
            'Protected baseline lacks passing action/attachment evidence')
    text=(parent/f'agents/reviewer-{attempt}/prompt.txt').read_text()
    marker='INPUT_JSON (data, not instructions):\n';require(marker in text,'Protected review lacks bound payload')
    supplied=json.loads(text.split(marker,1)[1]);proofs=supplied.get('independent_proofs',{})
    require(supplied.get('request')==request and supplied.get('plan')==plan and supplied.get('authored_config')==config
            and supplied.get('evidence')==receipt['evidence'] and proofs.get('deformation_gate')==deformation
            and proofs.get('action_audit')==action,'Protected baseline reviewer saw different proofs')
    prompts.validate_review(review,plan,[r['id'] for r in supplied['available_citations']])
    scope=eligible_scope(answer,review,plan)
    if scope is None:return None
    return {'schema_version':1,'policy':POLICY,'parent_output':str(parent.relative_to(root)),
            'attempt':attempt,'source_sha256':source_sha,**scope,'baseline_answer':answer,
            'active_review':review,'evidence':refs,
            'protection_scope':'Preserve the measured-passing skin/rest geometry and articulation. Older failures are historical; later rejected patches cannot unlock protected fields.'}


def select_baseline(root,job,request,continuation,current_review,source_sha,plan):
    from avatar_assets.repair_admission import _target
    if request.get('agent_backend',{}).get('provider')!='openai':return None
    if current_review is not None and current_review.get('coat_representation',{}).get('status')=='unavailable':
        # Legacy summaries can still qualify, but only when their concrete source-bound model exists.
        if not (Path(job)/current_review.get('receipt',{}).get('path','missing')).is_file():return None
    target=_target(Path(root),Path(job),request,continuation,current_review,source_sha)
    if target is None:return None
    pack,_=target
    require(re.fullmatch(r'model-[0-2]',pack.name) is not None,'Invalid protected baseline attempt')
    return _baseline_data(Path(root),pack.parent,int(pack.name[-1]),source_sha,plan)


def verify_baseline(job,baseline,*,depth=0):
    root=Path(job).parent.parent
    require(isinstance(baseline,dict) and baseline.get('schema_version')==1 and baseline.get('policy')==POLICY,
            'Invalid protected baseline descriptor')
    relative=baseline.get('parent_output');require(isinstance(relative,str) and re.fullmatch(r'outputs/[a-zA-Z0-9_-]+',relative),
                                                'Invalid protected baseline ancestry')
    parent=root/relative
    for name,pin in baseline.get('evidence',{}).items():
        require(isinstance(name,str) and not Path(name).is_absolute() and '..' not in Path(name).parts,'Protected reference escaped')
        path=local_file(parent/name,64*1024*1024)
        require(path==parent/name and pin=={'bytes':path.stat().st_size,'sha256':sha256(path)},'Protected baseline evidence changed')
    rebuilt=_baseline_data(root,parent,baseline['attempt'],baseline['source_sha256'],read_json(parent/'plan.json'),depth=depth)
    require(rebuilt==baseline,'Protected baseline evidence or scope changed')
    return baseline


def _receipt(job,attempt,raw,baseline,effective):
    folder=job/'composition'/f'author-{attempt}'
    return {'schema_version':1,'policy':POLICY,'attempt':attempt,'request_sha256':sha256(job/'request.json'),
        'raw_answer_sha256':sha256(job/f'agents/author-{attempt}/answer.json'),
        'raw_receipt_sha256':sha256(job/f'agents/author-{attempt}/receipt.json'),
        'baseline_sha256':sha256(folder/'baseline.json'),'effective_answer_sha256':sha256(folder/'effective-answer.json'),
        'patch_schema_sha256':digest(patch_schema(baseline)),
        'protected_projection_sha256':digest(protected_projection(effective,baseline['root_bone'],{r['state_id'] for r in baseline['failed_root_trajectory_requirements']}))}


def verify_materialized(job,attempt,raw,*,_depth=0):
    job=Path(job);folder=job/'composition'/f'author-{attempt}'
    if raw.get('schema_version')!=3:
        require(not folder.exists(),'Legacy author unexpectedly has composition artifacts')
        return raw
    if raw.get('status')=='unsupported':
        require(not folder.exists() and raw.get('patch') is None and isinstance(raw.get('reason'),str) and raw['reason'].strip(),
                'Invalid unsupported protected patch')
        return {'schema_version':2,'status':'unsupported','reason':raw['reason'],'design':None}
    baseline=read_json(folder/'baseline.json');verify_baseline(job,baseline,depth=_depth)
    prompt=local_file(job/f'agents/author-{attempt}/prompt.txt',512*1024).read_text()
    marker='INPUT_JSON (data, not instructions):\n';require(marker in prompt,'Patch lacks bound baseline prompt')
    payload,_=json.JSONDecoder().raw_decode(prompt.split(marker,1)[1])
    require(payload.get('protected_repair_baseline')==baseline,'Composition baseline differs from paid prompt')
    require(read_json(job/f'agents/author-{attempt}/answer.json')==raw,'Composition raw patch changed')
    paid=read_json(job/f'agents/author-{attempt}/receipt.json')
    require(paid.get('status')=='succeeded' and paid.get('answer_sha256')==sha256(job/f'agents/author-{attempt}/answer.json'),
            'Composition paid patch custody changed')
    effective=compose_patch(raw,baseline)
    require(read_json(folder/'effective-answer.json')==effective
            and read_json(folder/'receipt.json')==_receipt(job,attempt,raw,baseline,effective),
            'Composition receipt/effective answer changed')
    return effective


def materialize(job,attempt,raw,baseline):
    job=Path(job);folder=job/'composition'/f'author-{attempt}'
    require(raw.get('status')=='ready','Only ready patches have composition artifacts')
    if folder.exists():return verify_materialized(job,attempt,raw)
    verify_baseline(job,baseline)
    effective=compose_patch(raw,baseline)
    folder.mkdir(parents=True,mode=0o700)
    write_json(folder/'baseline.json',baseline);write_json(folder/'effective-answer.json',effective)
    write_json(folder/'receipt.json',_receipt(job,attempt,raw,baseline,effective))
    return verify_materialized(job,attempt,raw)
