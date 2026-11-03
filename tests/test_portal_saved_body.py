"""Metadata queue custody plus explicit analytic worker fixtures; no inference or source study."""
from copy import deepcopy
from hashlib import sha256
import json
import os
from pathlib import Path
import time
import pytest
from weft import portal_saved_body as jobs
from weft.io import canonical_hash
from weft.portal_inventory import exercise_intent
from weft.person_inventory import make_conditional_person_inventory,make_person_inventory,validate_person_inventory,CONDITIONAL_PROFILE


def raw(value):return json.dumps(value,sort_keys=True).encode()
def write(path,value):path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(value if isinstance(value,bytes) else raw(value))
def pin(path):return sha256(path.read_bytes()).hexdigest()
def seal(root,**extra):
    members={p.relative_to(root).as_posix():pin(p) for p in root.rglob('*') if p.is_file() and p.name!='manifest.json'}
    write(root/'manifest.json',{'files':members,**extra});return pin(root/'manifest.json')


@pytest.fixture
def setup(tmp_path):
    root=tmp_path/'portal';root.mkdir();fd=os.open(root/'portal.lock',os.O_RDWR|os.O_CREAT,0o600)
    parent=root/('a'*32);parent.mkdir();write(parent/'source.mp4',b'analytic media fixture, not a video')
    job={'id':parent.name,'mode':'exercise','source_file':'source.mp4','source_sha256':pin(parent/'source.mp4'),'exercise_intent':exercise_intent('unknown','simulation')}
    write(parent/'job.json',job);write(parent/'status.json',{'status':'needs_input'})
    write(parent/'perception/status.json',{'status':'awaiting_consent'})
    preflight=parent/'perception/preflight';body=preflight/'body_source';seed=preflight/'body_seed'
    write(body/'body-source.json',{'fixture':'analytical body'});body_pin=seal(body,profile='exercise-body-source-v1',status='construction_only',simulation_ready=False)
    write(seed/'scene.json',{'fixture':'analytical seed'});seed_pin=seal(seed)
    write(preflight/'metadata.json',{'source_sha256':job['source_sha256']})
    write(preflight/'result.json',{'status':'local_ready','simulation_ready':False,'source_sha256':job['source_sha256']})
    write(preflight/'manifest.json',{'profile':'exercise-perception-preflight-v1','status':'local_ready','children':{'body_source':body_pin,'body_seed':seed_pin},'files':{n:pin(preflight/n) for n in ('metadata.json','result.json')}})
    payload={'request_id':'b'*32,'preflight_manifest_sha256':pin(preflight/'manifest.json'),'modeling_scope':jobs.MODELING_SCOPE}
    options={'lease_fd':fd,'existing_jobs':{parent.name:job},'stored_bytes':0,'reserved_bytes':0,'pending_jobs':0,'deadline':time.monotonic()+10}
    yield root,parent,payload,options
    os.close(fd)


def queue(setup):
    root,parent,payload,options=setup
    child=jobs.queue_saved_body_child(root,parent.name,payload,**options)
    return root/child['id'],child


def fake_workers(monkeypatch,setup,child,*,feasible=True):
    from weft import exercise_perception_pipeline as pipeline,shared_world_human as builder,shared_exercise_lab as lab
    calls=[];root,parent,payload,options=setup
    def validate(video,preflight,**kwargs):
        assert (child/'saved-body/start.json').is_file();calls.append('validate')
        return {'manifest_sha256':pin(preflight/'manifest.json')}
    def build(body,inventory,out,**kwargs):
        calls.append('build');assert inventory['original_exercise_intent']['inventory']=='unknown'
        write(out/'reference-feasibility.json',{'status':'feasible' if feasible else 'infeasible','summary':{'fixture':True}})
        write(out/'scene.json',{'fixture':'not source-qualified'})
        return {'manifest_sha256':seal(out),'reference_feasibility':{'status':'feasible' if feasible else 'infeasible','summary':{'fixture':True}}}
    def prepare(world,out,**kwargs):
        calls.append('lab');write(out/'run/scene.json',{'fixture':'not source-qualified'});manifest=seal(out/'run')
        write(out/'lab-config.json',{'manifest_sha256':manifest});return {'manifest_sha256':manifest}
    monkeypatch.setattr(pipeline,'validate_perception_preflight',validate)
    monkeypatch.setattr(builder,'construct_human_only_run',build)
    monkeypatch.setattr(lab,'prepare_shared_exercise_lab',prepare)
    return calls


def test_availability_is_metadata_only_and_preserves_unknown(setup,monkeypatch):
    root,parent,payload,options=setup;original=jobs._read;reads=[]
    def tracked(root,name,*a,**k):reads.append(name);return original(root,name,*a,**k)
    monkeypatch.setattr(jobs,'_read',tracked)
    got=jobs.availability(parent)
    assert got['available'] and got['preflight_manifest_sha256']==payload['preflight_manifest_sha256']
    assert got['original_exercise_intent']['inventory']=='unknown'
    assert all(n.endswith('.json') for n in reads)
    assert not any(n.endswith('human.json') for n in reads)


def test_unsealed_metadata_never_advertises_available(setup):
    _,parent,_,_=setup;path=parent/'perception/preflight/manifest.json'
    value=json.loads(path.read_bytes());value['files']['metadata.json']=None;write(path,value)
    assert jobs.availability(parent)['available'] is False


def test_queue_idempotent_before_capacity_and_no_parent_mutation(setup):
    root,parent,payload,options=setup;before={str(p):p.read_bytes() for p in parent.rglob('*') if p.is_file()}
    child,job=queue(setup);kwargs={**options,'existing_jobs':{**options['existing_jobs'],job['id']:job},'stored_bytes':10**10}
    assert jobs.queue_saved_body_child(root,parent.name,payload,**kwargs)==job
    assert job['kind']=='saved_body' and job['ai'] is False and 'source_file' not in job
    assert jobs.reserved_growth(child)==650_000_000
    assert jobs.read_status(child)['route']=='exercise'
    assert all(Path(n).read_bytes()==v for n,v in before.items())


@pytest.mark.parametrize('change',['scope','extra','pin','parent_path','active','body_failed','linked','quota','request_collision'])
def test_queue_negative_custody_and_request_bounds(setup,change):
    root,parent,payload,options=setup
    if change=='scope':payload['modeling_scope']='verified_ground_only'
    if change=='extra':payload['ignore_floor']=True
    if change=='pin':payload['preflight_manifest_sha256']='0'*64
    if change=='parent_path':parent=Path('../'+parent.name)
    if change=='active':write(parent/'perception/status.json',{'status':'running'})
    if change=='body_failed':write(parent/'perception/preflight/body_source/failure.json',{})
    if change=='linked':(parent/'perception/preflight/body_source/failure.json').symlink_to(root/'absent')
    if change=='quota':options['stored_bytes']=2_000_000_000
    if change=='request_collision':options['existing_jobs']['c'*32]={'request_id':payload['request_id']}
    with pytest.raises(ValueError):jobs.queue_saved_body_child(root,str(parent) if change=='parent_path' else parent.name,payload,**options)
    assert not any(p.name.startswith('.saved-body-') for p in root.iterdir())


@pytest.mark.parametrize('feasible',[True,False])
def test_worker_reuses_saved_body_and_keeps_floor_failure_visible_without_new_gate(setup,monkeypatch,feasible):
    child,job=queue(setup);calls=fake_workers(monkeypatch,setup,child,feasible=feasible)
    before={str(p):p.read_bytes() for p in setup[1].rglob('*') if p.is_file()}
    assert jobs.execute_job(child)==0
    assert calls==['validate','build','lab']
    assert jobs.read_status(child)['route']=='exercise'
    report=jobs.summary(child)['report']
    assert report['reference_feasible'] is feasible and report['task_success'] is False
    assert report['inventory']['profile']==CONDITIONAL_PROFILE and report['inventory']['original_exercise_intent']['inventory']=='unknown'
    assert report['api_calls']==report['inference_calls']==0
    assert all(Path(n).read_bytes()==v for n,v in before.items())
    assert jobs.execute_job(child)==0 and calls==['validate','build','lab']


@pytest.mark.parametrize('change',['parent','body_bytes','source','request_path','cancel','started'])
def test_worker_refuses_changed_or_partial_inputs_and_never_retries(setup,monkeypatch,change):
    child,job=queue(setup);calls=fake_workers(monkeypatch,setup,child);parent=setup[1]
    if change=='parent':write(parent/'job.json',{**setup[3]['existing_jobs'][parent.name],'exercise_intent':exercise_intent('equipment_or_obstacles')})
    if change=='body_bytes':write(parent/'perception/preflight/body_source/manifest.json',{})
    if change=='source':write(parent/'source.mp4',b'changed')
    if change=='request_path':
        r=json.loads((child/'saved-body/request.json').read_bytes());r['parent_id']='../'+parent.name;write(child/'saved-body/request.json',r)
        j=json.loads((child/'job.json').read_bytes());j['saved_body_request_sha256']=pin(child/'saved-body/request.json');write(child/'job.json',j)
    if change=='cancel':write(child/'cancel-request.json',{})
    if change=='started':write(child/'saved-body/start.json',{'partial':True})
    assert jobs.execute_job(child)==1
    state=jobs.read_status(child);assert state['status'] in {'failed','cancelled','interrupted'}
    assert jobs.execute_job(child)==1
    if change!='source':assert calls==[]
    assert jobs.read_status(child)['route']=='exercise'
    assert (child/'saved-body/failure.json').is_file()


@pytest.mark.parametrize('original',['unknown','equipment_or_obstacles','ground_only'])
def test_conditional_scope_preserves_original_inventory_and_v1_unchanged(original):
    intent=exercise_intent(original);inventory=make_conditional_person_inventory('a'*64,intent)
    assert inventory['original_exercise_intent']==intent and inventory['absence_verified'] is False
    assert validate_person_inventory(inventory,source_sha256='a'*64)==inventory
    assert inventory['original_exercise_intent_sha256']==canonical_hash(intent)
    old=make_person_inventory('a'*64)
    assert old=={'schema_version':1,'profile':'human-only-inventory-v1','source_sha256':'a'*64,'basis':'user_declared','selected_entity_id':'human','support':'ground_only','task_relevant_equipment':'none_declared','task_relevant_obstacles':'none_declared','absence_verified':False}


@pytest.mark.parametrize('field,value',[('absence_verified',True),('source_sha256','b'*64),('schema_version',True),('omitted_contacts','none'),('original_exercise_intent_sha256','0'*64)])
def test_conditional_inventory_tamper_rejects(field,value):
    inv=make_conditional_person_inventory('a'*64,exercise_intent('unknown'));inv[field]=value
    with pytest.raises(ValueError):validate_person_inventory(inv,source_sha256='a'*64)


from test_shared_world_human import analytical_body, body


def test_actual_conditional_constructor_and_ordinary_lab_keep_full_body_and_scope(body,tmp_path):
    from weft.shared_world_human import construct_human_only_run
    from weft.shared_exercise_lab import prepare_shared_exercise_lab
    original=(body/'human.json').read_bytes();source=json.loads((body/'body-source.json').read_bytes())['source_sha256']
    inventory=make_conditional_person_inventory(source,exercise_intent('equipment_or_obstacles'))
    world=tmp_path/'world';built=construct_human_only_run(body,inventory,world)
    scene=json.loads((world/'scene.json').read_bytes())
    assert scene['inventory']==inventory and scene['objects']==[]
    assert (world/'human.json').read_bytes()==original and 'omitted, not absent' in scene['assumptions'][0]
    assert json.loads((world/'reconstruction.json').read_bytes())['mode']=='conditional_body_ground'
    prepared=prepare_shared_exercise_lab(world,tmp_path/'lab')
    assert prepared['manifest_sha256']==pin(tmp_path/'lab/run/manifest.json')
    saved=json.loads((tmp_path/'lab/run/scene.json').read_bytes())
    assert saved==scene and saved['inventory']['original_exercise_intent']['inventory']=='equipment_or_obstacles'
    assert prepared['learned_policy_available'] is False
