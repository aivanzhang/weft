"""Explicit, immutable appearance preference; not approval of a particular asset."""
from contextlib import closing
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import re
import sqlite3
import stat

from avatar_assets.safety import local_file, read_json, require, sha256

MODE = 'preserve_current'
SNAPSHOT = 'inputs/appearance-preference.json'
BASE_FIELDS = {'parent_output', 'attempt', 'source_sha256', 'model_manifest_sha256',
               'design_sha256', 'review_sha256', 'descriptor_sha256'}
PREFERENCE = 'Preserve current stable appearance and finish the requested motion.'


def _bytes(value):
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+'\n').encode()


def _hash(value):
    require(isinstance(value, str) and re.fullmatch(r'[0-9a-f]{64}', value) is not None,
            'Invalid appearance policy hash')


def validate_policy(value):
    require(isinstance(value, dict) and set(value)=={'schema_version','mode','baseline','authorization'}
            and type(value['schema_version']) is int and value['schema_version']==1 and value['mode']==MODE,
            'Invalid appearance policy')
    baseline=value['baseline']
    require(isinstance(baseline, dict) and set(baseline)==BASE_FIELDS
            and isinstance(baseline['parent_output'], str)
            and re.fullmatch(r'outputs/[A-Za-z0-9_-]+', baseline['parent_output']) is not None
            and type(baseline['attempt']) is int and 0<=baseline['attempt']<=2,
            'Invalid appearance baseline')
    for name in BASE_FIELDS-{'parent_output','attempt'}: _hash(baseline[name])
    auth=value['authorization']
    require(isinstance(auth, dict) and set(auth)=={'path','sha256'} and isinstance(auth['path'], str)
            and not Path(auth['path']).is_absolute() and '..' not in Path(auth['path']).parts
            and re.fullmatch(r'\.local/(?:[A-Za-z0-9_-]+/)+jobs/[0-9a-f]{24}/appearance-preference\.json', auth['path']) is not None,
            'Invalid appearance preference receipt path')
    _hash(auth['sha256'])
    require(len(_bytes(value))<=8192, 'Appearance policy exceeds bound')
    return deepcopy(value)


def validate_transition(request, parent_request):
    old=parent_request.get('appearance_policy'); new=request.get('appearance_policy')
    if old is not None:
        validate_policy(old)
        require(new==old, 'Continuation cannot remove or change its appearance policy')
    if new is not None:
        validate_policy(new)
        require(request.get('reuse_model', {}).get('sha256')==new['baseline']['source_sha256']
                and request.get('max_credits')==0, 'Appearance policy source or generation allowance differs')


def _receipt(raw, policy):
    require(len(raw)<=16384 and hashlib.sha256(raw).hexdigest()==policy['authorization']['sha256'],
            'Appearance preference receipt changed')
    value=json.loads(raw)
    require(isinstance(value, dict) and set(value)=={'schema_version','operation','requested_via',
            'user_preference','human_asset_approval','parent_job_id','child_job_id','parent_request_sha256','baseline'}
            and type(value['schema_version']) is int and value['schema_version']==1
            and value['operation']=='preserve_current_appearance' and value['requested_via']=='explicit_retry_payload'
            and value['user_preference']==PREFERENCE and value['human_asset_approval'] is False
            and value['baseline']==policy['baseline'], 'Invalid appearance preference authorization')
    for name in ('parent_job_id','child_job_id'):
        require(isinstance(value[name],str) and re.fullmatch(r'[0-9a-f]{24}',value[name]) is not None,
                'Invalid appearance preference job identity')
    require(value['parent_job_id']!=value['child_job_id']
            and Path(policy['authorization']['path']).parent.name==value['child_job_id'],
            'Appearance preference child differs')
    _hash(value['parent_request_sha256'])
    return value


def _private(path, maximum):
    path=local_file(path,maximum); info=path.stat()
    require(info.st_uid==os.getuid() and stat.S_IMODE(info.st_mode)==0o600,
            'Appearance preference origin must be owner-only')
    return path


def verify_origin(root, value):
    """Action-time authority comes from the canonical explicit Retry, never a model reply."""
    root=Path(root).resolve(); policy=validate_policy(value)
    binding=read_json(_private(root/'.local/studio-registry.json',16384))
    require(isinstance(binding,dict) and set(binding)=={'schema_version','state_dir'}
            and binding['schema_version']==1, 'Appearance preference registry binding differs')
    state=Path(binding['state_dir'])
    require(state.is_absolute() and state.resolve()==state and state.is_relative_to(root/'.local'),
            'Appearance preference registry escaped')
    path=_private(root/policy['authorization']['path'],16384)
    receipt=_receipt(path.read_bytes(),policy)
    require(path==state/'jobs'/receipt['child_job_id']/'appearance-preference.json',
            'Appearance preference is outside the canonical registry')
    saved=read_json(_private(path.parent/'request.json',1024**2))
    dbpath=_private(state/'jobs.sqlite3',32*1024**2)
    with closing(sqlite3.connect(dbpath.as_uri()+'?mode=ro',uri=True,timeout=.2)) as db:
        db.execute('PRAGMA query_only=ON')
        row=db.execute('SELECT request,request_digest FROM jobs WHERE id=?',(receipt['child_job_id'],)).fetchone()
        parent=db.execute('SELECT data FROM jobs WHERE id=?',(receipt['parent_job_id'],)).fetchone()
    canonical=json.dumps(saved,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
    require(row is not None and json.loads(row[0])==saved and row[1]==hashlib.sha256(canonical).hexdigest()
            and saved.get('appearance_policy')==policy and saved.get('continuation_from')==receipt['parent_job_id']
            and parent is not None and json.loads(parent[0]).get('retry_child')==receipt['child_job_id'],
            'Appearance preference lacks its canonical explicit Retry')
    return receipt


def build_policy(root, parent, captured, *, parent_job_id, child_job_id, receipt_path):
    """Caller holds custody and rechecks the captured terminal inventory before saving."""
    from avatar_assets.repair_composition import _baseline_data, digest
    from avatar_assets.repair_visual_inputs import _lineage_inventory
    root,parent=Path(root),Path(parent)
    require(parent==root/'outputs'/('studio-'+parent_job_id)
            and captured['parent']==str(parent.relative_to(root))
            and captured['parent_request'].get('appearance_policy') is None,
            'Appearance preference requires its original explicit transition')
    history=captured['context']['findings'].get('historical_visual_reviews',[])
    require(isinstance(history,list) and 1<=len(history)<=8,'No bounded reviewed appearance baseline')
    descriptor=None
    for summary in reversed(history):
        if not {'identity','framing','deformation'}<=set(summary.get('passed_checks',[])):
            continue
        if not summary.get('state_results') or not all(row.get('motion_passed') is True for row in summary['state_results']):
            continue
        relative=summary.get('parent_output'); attempt=summary.get('attempt')
        if relative==captured['parent']:
            ancestor=parent; pins={row['path']:row for row in captured['inventory']}
        else:
            ancestor,pins,_=_lineage_inventory(root,parent,captured['parent_request'],relative)
        descriptor=_baseline_data(root,ancestor,attempt,captured['parent_request']['reuse_model']['sha256'],
                                  captured['context']['plan'])
        require(descriptor is not None, 'Latest stable appearance is not eligible for motion-only repair')
        require(all(pins.get(name)=={'path':name,**pin} for name,pin in descriptor['evidence'].items()),
                'Appearance baseline differs from captured ancestry')
        break
    require(descriptor is not None, 'No eligible stable appearance baseline')
    attempt=descriptor['attempt']; refs=descriptor['evidence']
    baseline={'parent_output':descriptor['parent_output'],'attempt':attempt,'source_sha256':descriptor['source_sha256'],
        'model_manifest_sha256':refs[f'model-{attempt}/manifest.json']['sha256'],
        'design_sha256':refs[f'finish-{attempt}.json']['sha256'],
        'review_sha256':refs[f'review-{attempt}.json']['sha256'],'descriptor_sha256':digest(descriptor)}
    receipt={'schema_version':1,'operation':'preserve_current_appearance','requested_via':'explicit_retry_payload',
        'user_preference':PREFERENCE,'human_asset_approval':False,'parent_job_id':parent_job_id,
        'child_job_id':child_job_id,'parent_request_sha256':sha256(parent/'request.json'),'baseline':baseline}
    policy=validate_policy({'schema_version':1,'mode':MODE,'baseline':baseline,
        'authorization':{'path':receipt_path,'sha256':hashlib.sha256(_bytes(receipt)).hexdigest()}})
    _receipt(_bytes(receipt),policy)
    return policy,receipt


def snapshot_policy(root, out, value, parent_request):
    root,out=Path(root),Path(out); policy=validate_policy(value)
    origin=verify_origin(root,policy)
    validate_transition({'appearance_policy':policy,'reuse_model':parent_request.get('reuse_model'),
                         'max_credits':parent_request.get('max_credits')},parent_request)
    if parent_request.get('appearance_policy') is None:
        require(out.name=='studio-'+origin['child_job_id'] and parent_request['id']=='studio-'+origin['parent_job_id']
                and sha256(root/'outputs'/parent_request['id']/'request.json')==origin['parent_request_sha256'],
                'Appearance preference transition belongs to another parent or child')
    raw=(root/policy['authorization']['path']).read_bytes()
    with (out/SNAPSHOT).open('xb') as stream: stream.write(raw)
    return policy


def verify_policy(root, job, request):
    """Portable proof remains readable without the private Studio registry."""
    from avatar_assets.pipeline_continuation import INTENT, _read_context
    from avatar_assets.repair_visual_inputs import _lineage_inventory, _bound
    root,job=Path(root),Path(job); policy=validate_policy(request.get('appearance_policy'))
    require(job.parent==root/'outputs' and job.resolve()==job and request.get('id')==job.name
            and request.get('reuse_model',{}).get('sha256')==policy['baseline']['source_sha256']
            and request.get('max_credits')==0, 'Appearance policy request or source differs')
    preference=_receipt(local_file(job/SNAPSHOT,16384).read_bytes(),policy)
    receipt,context=_read_context(job,request)
    origin='outputs/studio-'+preference['child_job_id']
    if origin==str(job.relative_to(root)):
        require(receipt['parent']=='outputs/studio-'+preference['parent_job_id']
                and receipt['parent_request_sha256']==preference['parent_request_sha256'],
                'Appearance preference parent request differs')
    else:
        ancestor,pins,saved=_lineage_inventory(root,job,request,origin)
        require(saved.get('appearance_policy')==policy
                and sha256(_bound(ancestor,pins,SNAPSHOT,16384))==policy['authorization']['sha256'],
                'Inherited appearance preference differs from its original transition')
    baseline=policy['baseline']; ancestor,pins,saved=_lineage_inventory(root,job,request,baseline['parent_output'])
    attempt=baseline['attempt']
    for name,field in ((f'model-{attempt}/manifest.json','model_manifest_sha256'),
                       (f'finish-{attempt}.json','design_sha256'),(f'review-{attempt}.json','review_sha256')):
        require(sha256(_bound(ancestor,pins,name))==baseline[field],'Appearance baseline reference changed')
    require(all(request.get(key)==saved.get(key) for key in INTENT)
            and saved.get('reuse_model',{}).get('sha256')==baseline['source_sha256']
            and context['plan']==read_json(_bound(ancestor,pins,'plan.json')),
            'Appearance baseline changes the source, plan or requested intent')
    for name in ('reference','reuse_model'):
        current,old=request.get(name),saved.get(name)
        require((current is None)==(old is None) and (current is None or
                (current['sha256'],current['bytes'])==(old['sha256'],old['bytes'])),
                'Appearance baseline source/reference differs')
    return policy
