"""Separate-asset scene drafts; never dispatch generation, merge meshes or infer a split."""
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import time

from avatar_assets.action_intent import validate_custom_actions
from avatar_assets.safety import artifacts, local_file, new_directory, read_json, require, sha256, write_json

ROOT=Path(__file__).resolve().parents[1]
POLICY='independent-avatar-assets-v1'
SPACE='gltf_y_up_right_handed_meters'
MAX_ASSETS=3
MAX_JSON_BYTES=64*1024
MAX_SOURCE_BYTES=64*1024**2
PRESETS=('draft','preview','muse')


class UnsupportedSceneError(ValueError):
    def __init__(self,code,message):
        self.code=code
        super().__init__(message)


def _canonical(value):
    raw=json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()
    require(len(raw)<=MAX_JSON_BYTES,'Scene JSON exceeds64KiB')
    return raw


def _id(value,label='subject'):
    require(isinstance(value,str) and re.fullmatch(r'[a-z][a-z0-9_-]{0,19}',value) is not None,
            'Invalid '+label+' id; use1–20 lowercase ASCII letters, digits, hyphens or underscores')
    return value


def _text(value,limit,label,empty=False):
    require(isinstance(value,str) and len(value)<=limit and '\0' not in value
            and (empty or value.strip()),'Invalid bounded '+label)
    return value


def _hash(value):
    require(isinstance(value,str) and re.fullmatch(r'[0-9a-f]{64}',value) is not None,'Invalid SHA256')


def _path(value,suffix):
    require(isinstance(value,str) and 0<len(value)<=512 and not Path(value).is_absolute()
            and '..' not in Path(value).parts and '\\' not in value and '\0' not in value
            and len(Path(value).parts)>=2 and Path(value).parts[0] in ('outputs','references')
            and Path(value).suffix==suffix,'Require a project-relative '+suffix+' artifact path')


def _placement(value):
    if value is None:return None
    require(isinstance(value,dict) and set(value)=={'translation_m','rotation_xyzw','uniform_scale'},
            'Placement needs explicit translation, unit quaternion and positive uniform scale')
    for name,count,bound in (('translation_m',3,100),('rotation_xyzw',4,1)):
        row=value[name]
        require(isinstance(row,list) and len(row)==count and all(type(v) in (int,float) and math.isfinite(v)
                and abs(v)<=bound for v in row),'Invalid placement '+name)
    require(abs(math.sqrt(sum(v*v for v in value['rotation_xyzw']))-1)<=1e-6,'Placement quaternion must be unit length')
    scale=value['uniform_scale']
    require(type(scale) in (int,float) and math.isfinite(scale) and .001<=scale<=100,
            'Placement disallows reflection, shear or nonuniform scale')
    return deepcopy(value)


def validate_explicit_assets(assets):
    """Preserve subject-specific wording/order; declared ownership is not mesh inspection."""
    require(isinstance(assets,list) and 2<=len(assets)<=MAX_ASSETS,'A separate-assets scene needs2–3 subjects')
    _canonical(assets)
    result=[];ids=set();sources=set();paths=set()
    required={'id','name','prompt','style','custom_actions','preset'}
    for asset in assets:
        require(isinstance(asset,dict) and required<=set(asset)
                and set(asset)<=required|{'reference_id','source','placement'},'Invalid explicit asset fields')
        identifier=_id(asset['id'])
        require(identifier not in ids,'Duplicate subject id');ids.add(identifier)
        _text(asset['name'],120,'subject name');_text(asset['prompt'],4000,'subject prompt')
        _text(asset['style'],800,'subject style',empty=True)
        actions=validate_custom_actions(asset['custom_actions'])
        require(asset['preset'] in PRESETS,'Unsupported child preset')
        reference=asset.get('reference_id')
        require(reference is None or isinstance(reference,str) and re.fullmatch(r'[A-Za-z0-9_-]{1,64}',reference) is not None,
                'Invalid reference id; controllers resolve approved references')
        source=asset.get('source',{'mode':'generate'})
        require(isinstance(source,dict),'Invalid subject source')
        if source.get('mode')=='generate':
            require(set(source)=={'mode'},'Fresh generation cannot silently reuse an artifact')
        elif source.get('mode')=='reuse':
            require(set(source)=={'mode','path','sha256','subject_ids'},'Reuse needs exact path/hash and declared subject ownership')
            _path(source['path'],'.glb');_hash(source['sha256'])
            require(isinstance(source['subject_ids'],list) and 1<=len(source['subject_ids'])<=MAX_ASSETS,
                    'Invalid source subject ownership')
            for name in source['subject_ids']:_id(name)
            require(len(set(source['subject_ids']))==len(source['subject_ids']),'Duplicate source subject ownership')
            if source['subject_ids']!=[identifier]:
                raise UnsupportedSceneError('fused_or_foreign_source','A reused source must belong only to this subject; automatic splitting or hidden-surface repair is unsupported')
            if source['sha256'] in sources or source['path'] in paths:
                raise UnsupportedSceneError('duplicate_source','Independent subjects cannot reuse the same source hash or model path')
            sources.add(source['sha256']);paths.add(source['path'])
        else:
            raise UnsupportedSceneError('source_route','Only independent generation or explicitly single-subject reuse is supported; fused-source decomposition is not implemented')
        result.append({**deepcopy(asset),'custom_actions':actions,'reference_id':reference,
                       'source':deepcopy(source),'placement':_placement(asset.get('placement'))})
    return result


def validate_scene(data):
    require(isinstance(data,dict) and {'title','assets'}<=set(data)
            and set(data)<={'schema_version','id','title','space','assets','relationships'},'Invalid scene request fields')
    _canonical(data)
    require(type(data.get('schema_version',1)) is int and data.get('schema_version',1)==1,'Unsupported scene schema')
    _text(data['title'],120,'scene title')
    require(data.get('space',SPACE)==SPACE,'Scene placement requires explicit glTF right-handed Y-up meters')
    assets=validate_explicit_assets(data['assets']);ids={asset['id'] for asset in assets}
    relationships=data.get('relationships',[])
    require(isinstance(relationships,list) and len(relationships)<=MAX_ASSETS-1,'At most two scene relationships')
    seen=set();parents={}
    for relation in relationships:
        require(isinstance(relation,dict) and set(relation)=={'id','kind','parent','child','parent_anchor','child_anchor','user_request'},
                'Invalid scene relationship fields')
        name=_id(relation['id'],'relationship');require(name not in seen,'Duplicate relationship id');seen.add(name)
        require(relation['kind'] in ('attachment','interaction'),'Unsupported relationship kind')
        require(isinstance(relation['parent'],str) and isinstance(relation['child'],str)
                and relation['parent'] in ids and relation['child'] in ids and relation['parent']!=relation['child'],
                'Relationship must reference two different declared subjects')
        require(relation['child'] not in parents,'A child cannot have multiple attachment/interaction parents')
        parents[relation['child']]=relation['parent']
        for key in ('parent_anchor','child_anchor'):
            require(relation[key] is None or isinstance(relation[key],str) and re.fullmatch(r'[A-Za-z][A-Za-z0-9_-]{0,63}',relation[key]) is not None,
                    'Anchor names are declarations, not recovered bones')
        _text(relation['user_request'],600,'relationship request')
    for child in parents:
        visited=set();current=child
        while current in parents:
            require(current not in visited,'Scene relationship cycle');visited.add(current);current=parents[current]
    normalized={'schema_version':1,'title':data['title'],'space':SPACE,'assets':assets,'relationships':deepcopy(relationships)}
    identifier=data.get('id','s-'+hashlib.sha256(_canonical(normalized)).hexdigest()[:18]);_id(identifier,'scene')
    result={'schema_version':1,'id':identifier,**{k:v for k,v in normalized.items() if k!='schema_version'}}
    _canonical(result)
    return result


def child_requests(scene):
    """Independent request drafts; no Pipeline.prepare, provider, ledger or reference resolution."""
    scene=validate_scene(scene)
    return [{'subject_id':asset['id'],'name':asset['name'],
             'output_key':'scene-'+scene['id']+'-'+asset['id'],
             'request':{key:deepcopy(asset[key]) for key in ('prompt','style','custom_actions','preset','reference_id')},
             'source':deepcopy(asset['source'])} for asset in scene['assets']]


def plan_scene(scene):
    scene=validate_scene(scene);children=child_requests(scene)
    generated=sum(asset['source']['mode']=='generate' for asset in scene['assets'])
    relationships=[{**row,'status':'unsupported','reason':
        'Separate child assets are planned, but cross-asset anchor binding, attachment motion and interaction verification are not implemented.'}
        for row in scene['relationships']]
    return {'schema_version':1,'policy':POLICY,'status':'planned_separate_assets',
        'request_sha256':hashlib.sha256(_canonical(scene)).hexdigest(),'scene':scene,'children':children,
        'deliverables':[{'subject_id':row['subject_id'],'kind':'independently_rigged_glb',
                         'relative_output':'outputs/'+row['output_key'],'status':'not_created'} for row in children],
        'scene_manifest':{'space':SPACE,'placements':[{ 'subject_id':a['id'],'transform':a['placement']} for a in scene['assets']],
                          'relationships':relationships,'status':'references_only_no_combined_renderer'},
        'complete':False,'execution_ready':False,'paid_calls':0,'credits_reserved':0,
        'aggregate_estimate':{'fresh_subjects':generated,'reused_source_count':len(children)-generated,
            'new_meshy_credits':35*generated,'expected_ai_calls':3*len(children),
            'scope':'Current child geometry-plus-texture plan and one planner/author/critic per child; not a grant, provider price quote or guaranteed call count.',
            'automatic_paid_geometry_retries':False,'suggested_concurrency':1,
            'execution':'Re-estimate under the current server limits and shared project ledger before each explicit start.'},
        'requirements':['Estimate and authorize aggregate child generation/AI cost before dispatch; a scene draft grants no spending.',
                        'Run each child through the existing source, rig, action, attachment, media and critic gates.',
                        'Verify that each final asset contains only its requested subject; declarations and distinct hashes do not prove semantic separation.',
                        'Resolve and verify reference IDs before child preparation; no image is silently reused across subjects.'],
        'unsupported':['No automatic decomposition of fused geometry, hidden-surface repair, welding or identity substitution.',
                       'No synchronized scene timeline, attachment solve, physical interaction or combined scene acceptance.']}


def validate_deliveries(scene, deliveries):
    """Check reference metadata only; callers still authenticate complete child deliveries."""
    scene=validate_scene(scene);assets={a['id']:a for a in scene['assets']}
    require(isinstance(deliveries,list) and len(deliveries)==len(assets),'Each subject needs exactly one child delivery reference')
    _canonical(deliveries);seen=set();models=set();sources=set();paths=set();result=[]
    required={'subject_id','prompt','style','custom_actions','source_sha256','model_path','model_sha256','delivery_path','delivery_sha256'}
    for row in deliveries:
        require(isinstance(row,dict) and set(row)==required and row['subject_id'] in assets and row['subject_id'] not in seen,
                'Invalid or duplicate subject delivery reference')
        seen.add(row['subject_id']);asset=assets[row['subject_id']]
        require(all(row[key]==asset[key] for key in ('prompt','style','custom_actions')),'Child delivery changed its subject prompt/style/actions')
        for key in ('source_sha256','model_sha256','delivery_sha256'):_hash(row[key])
        _path(row['model_path'],'.glb');_path(row['delivery_path'],'.json')
        require(row['source_sha256'] not in sources and row['model_sha256'] not in models and row['model_path'] not in paths,
                'Independent assets require distinct source and model hashes/paths')
        if asset['source']['mode']=='reuse':require(row['source_sha256']==asset['source']['sha256'],'Delivered source differs from explicit reuse')
        sources.add(row['source_sha256']);models.add(row['model_sha256']);paths.add(row['model_path']);result.append(deepcopy(row))
    return {'schema_version':1,'policy':POLICY,'status':'references_validated_only','complete':False,
            'scene_id':scene['id'],'assets':result,'scope':'Metadata identity/intent uniqueness only; no file, geometry, rig, appearance or scene verification.'}


def prepare(config_path, output, *, root=ROOT):
    """Save a new immutable draft and code/config snapshots without creating child jobs."""
    root=Path(root).resolve();config_path=local_file(config_path,MAX_JSON_BYTES)
    raw=config_path.read_bytes();scene=validate_scene(read_json(config_path));plan=plan_scene(scene)
    output=Path(output)
    require(not any(p.is_symlink() for p in (output,*output.parents))
            and output.resolve().is_relative_to(root/'outputs') and output.resolve()!=root/'outputs',
            'Scene draft output must be a fresh tool output folder')
    require(not output.exists(),'Scene draft output already exists; use a new folder')
    started=time.monotonic();source_checks=[];total=0
    for asset in scene['assets']:
        source=asset['source']
        if source['mode']!='reuse':continue
        path=local_file(root/source['path'],MAX_SOURCE_BYTES);before=path.stat()
        require(path==root/source['path'],'Reused source escaped or is linked')
        checksum=hashlib.sha256()
        with path.open('rb') as stream:
            while block:=stream.read(1024**2):
                checksum.update(block);total+=len(block)
                require(total<=MAX_ASSETS*MAX_SOURCE_BYTES and time.monotonic()-started<10,'Scene source hashing exceeded bounds')
        after=path.stat()
        require((before.st_size,before.st_mtime_ns)==(after.st_size,after.st_mtime_ns)
                and checksum.hexdigest()==source['sha256'],'Reused source changed or hash differs')
        source_checks.append({'subject_id':asset['id'],'path':source['path'],'bytes':before.st_size,
                              'sha256':source['sha256'],'scope':'Byte custody only; single-subject ownership still needs inspection.'})
    out=new_directory(output);(out/'inputs').mkdir();(out/'sources').mkdir();(out/'children').mkdir()
    (out/'inputs/config.json').write_bytes(raw)
    write_json(out/'scene-request.json',scene);write_json(out/'plan.json',plan)
    for child in plan['children']:
        folder=out/'children'/child['subject_id'];folder.mkdir();write_json(folder/'request.json',child)
    source_files=[]
    for module in ('scene_requests.py','action_intent.py','safety.py'):
        original=Path(__file__).parent/module;target=out/'sources'/module;shutil.copyfile(original,target)
        source_files.append({'path':'sources/'+module,'sha256':sha256(target),'bytes':target.stat().st_size})
    receipt={'schema_version':1,'policy':POLICY,'status':'planned_separate_assets','config_sha256':hashlib.sha256(raw).hexdigest(),
             'request_sha256':sha256(out/'scene-request.json'),'plan_sha256':sha256(out/'plan.json'),
             'source_checks':source_checks,'sources':source_files,'child_jobs_created':0,'paid_calls':0,'credits_reserved':0,
             'input_sources_modified':False,'complete':False,'execution_ready':False}
    write_json(out/'receipt.json',receipt);write_json(out/'manifest.json',{'schema_version':1,'kind':'independent_asset_scene_draft',
        'policy':POLICY,'status':'planned_separate_assets','artifacts':artifacts(out)})
    return {**receipt,'output':str(out),'manifest_sha256':sha256(out/'manifest.json')}
