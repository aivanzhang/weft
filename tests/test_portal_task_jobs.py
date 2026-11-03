"""Portal orchestration over explicit sealed analytic stubs; no source/native qualification."""

from copy import deepcopy
from hashlib import sha256
import json
import os
from pathlib import Path
import time
from types import SimpleNamespace

import pytest

from weft import portal_task_jobs as jobs
from weft.io import canonical_hash
from test_portal_task_contract import create, learning, outputs


def raw(value): return json.dumps(value,sort_keys=True).encode()
def write(path,value): path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(value if isinstance(value,bytes) else raw(value))
def pin(path): return sha256(path.read_bytes()).hexdigest()
def seal(root):
    members={p.relative_to(root).as_posix():pin(p) for p in root.rglob('*') if p.is_file() and p.name!='manifest.json'}
    write(root/'manifest.json',{'schema_version':1,'files':members});return pin(root/'manifest.json')


def task():
    return {'binding':{'scene_sha256':'4'*64,'family_sha256':'6'*64,'selection_sha256':'7'*64},
        'action_contract':{'names':['human','object_x']},'interval':{'horizon_steps':10},
        'scorer_sha256':{n:'8'*64 for n in ('moving_equipment_task.py','shared_exercise_task.py','whole_body.py')}}


@pytest.fixture
def setup(tmp_path):
    root=tmp_path/'portal';root.mkdir();fd=os.open(root/'portal.lock',os.O_RDWR|os.O_CREAT,0o600)
    def parent(operation='create'):
        identity=('a' if operation=='create' else 'b')*32;folder=root/identity;folder.mkdir()
        job={'id':identity,'kind':'lab','name':'Analytic parent','source_sha256':'9'*64}
        write(folder/'job.json',job);write(folder/'status.json',{'status':'ready','lab':'lab'})
        lab=folder/'lab';run=lab/'run';run.mkdir(parents=True)
        t=task(); marker={'source_sha256':'9'*64,'scene_sha256':'4'*64,'task_sha256':canonical_hash(t)}
        profile='shared_world_exploration' if operation=='create' else 'moving_equipment_task'
        write(run/('exploration-lab.json' if operation=='create' else 'task-lab.json'),marker)
        if operation!='create':write(run/'placement-task.json',t)
        manifest=seal(run);write(lab/'lab-config.json',{'control_profile':profile,'scene_sha256':'4'*64,'task_sha256':canonical_hash(t),'manifest_sha256':manifest})
        return folder,job
    yield root,fd,parent
    os.close(fd)


def queue(setup,operation='create',**options):
    root,fd,build=setup;parent,job=build(operation)
    request=create() if operation=='create' else learning()
    request['parent_manifest_sha256' if operation=='create' else 'task_sha256']=pin(parent/'lab/run/manifest.json') if operation=='create' else canonical_hash(task())
    fn=jobs.create_task_child if operation=='create' else jobs.create_learning_child
    kwargs={'lease_fd':fd,'existing_jobs':{parent.name:job},'stored_bytes':0,'reserved_bytes':0,'pending_jobs':0,'deadline':time.monotonic()+10,**options}
    child=fn(root,parent.name,request,**kwargs)
    return parent,root/child['id'],request,kwargs


def fake_workers(monkeypatch,child):
    from weft import moving_equipment_task_lab as labs,moving_equipment_learning as train,moving_equipment_evaluation as evaluation
    calls=[];admission=SimpleNamespace(task=task())
    def create_lab(source,parameters,out,**options):
        assert (child/'task-job/start.json').is_file();calls.append('create')
        run=out/'run';run.mkdir(parents=True);write(run/'placement-task.json',task());manifest=seal(run)
        write(out/'lab-config.json',{'manifest_sha256':manifest,'scene_sha256':'4'*64})
        return {'manifest_sha256':manifest,'status':'ready'}
    def load(*args,**kwargs):
        assert (child/'task-job/start.json').is_file();calls.append('admit');return admission
    def protocol(value,**settings):
        assert value is admission;calls.append('protocol');return {'fixture':'prospectively_fixed',**settings}
    def train_run(value,out,**settings):
        assert value is admission;calls.append('train')
        assert (child/'task-job/evaluation-protocol.json').is_file()
        out.mkdir();result,nominal,_=outputs();nominal['source']['task_sha256']=canonical_hash(task())
        for r in nominal['comparisons']:r['trial']['metrics']['task_sha256']=canonical_hash(task())
        write(out/'result.json',result);write(out/'evaluation.json',nominal);write(out/'policy.json',{'fixture':'sealed_unused_policy'})
        return {'manifest_sha256':seal(out)}
    def evaluate(value,protocol,policy,out,**settings):
        assert value is admission;calls.append('evaluate');out.mkdir()
        _,_,report=outputs();report['source']['task_sha256']=canonical_hash(task())
        write(out/'result.json',report);seal(out);return report
    monkeypatch.setattr(labs,'prepare_moving_equipment_task_lab',create_lab)
    monkeypatch.setattr(labs,'load_moving_equipment_task_lab',load)
    monkeypatch.setattr(train,'train_moving_equipment_policy',train_run)
    monkeypatch.setattr(evaluation,'build_changed_start_protocol',protocol)
    monkeypatch.setattr(evaluation,'evaluate_changed_starts',evaluate)
    return calls


def test_queue_only_small_metadata_never_copies_source_or_runs_native(setup,monkeypatch):
    parent,child,request,options=queue(setup)
    assert not (child/'lab').exists() and not (child/'source.mp4').exists()
    assert jobs.read_status(child)['status']=='queued'
    assert all(p.stat().st_size<=1<<20 for p in child.rglob('*') if p.is_file())
    assert jobs.reserved_growth(child)==650_000_000
    assert jobs.summary(child)['automatic_policy_installation'] is False


def test_create_worker_keeps_parent_and_publishes_separate_manual_lab(setup,monkeypatch):
    parent,child,_,_=queue(setup);before={str(p):pin(p) for p in parent.rglob('*') if p.is_file()}
    calls=fake_workers(monkeypatch,child)
    assert jobs.execute_job(child)==0 and calls==['create']
    state=jobs.read_status(child);assert state['status']=='ready' and state['lab']=='lab'
    assert not jobs.summary(child)['report']['learned_policy_available']
    assert {str(p):pin(p) for p in parent.rglob('*') if p.is_file()}==before
    assert jobs.reserved_growth(child)==0


def test_training_and_evaluation_share_one_admission_and_prospective_protocol(setup,monkeypatch):
    parent,child,_,_=queue(setup,'train_evaluate');before={str(p):pin(p) for p in parent.rglob('*') if p.is_file()}
    calls=fake_workers(monkeypatch,child)
    assert jobs.execute_job(child)==0 and calls==['admit','protocol','train','evaluate']
    state=jobs.read_status(child);assert state['status']=='complete' and 'lab' not in state
    report=jobs.summary(child)['report'];assert not report['policy_installed']
    assert report['evaluation']['requested_starts']==3 and report['evaluation']['excluded_or_refused_starts']==1
    assert report['training']['selected_kind']=='authored_unchanged'
    assert {str(p):pin(p) for p in parent.rglob('*') if p.is_file()}==before


def test_idempotent_unchanged_action_never_creates_second_child(setup):
    root,fd,_=setup;parent,child,request,options=queue(setup);saved=json.loads((child/'job.json').read_bytes())
    options['existing_jobs'][saved['id']]=saved
    assert jobs.create_task_child(root,parent.name,request,**options)==saved
    changed=deepcopy(request);changed['task']['hold_duration_s']=.7
    with pytest.raises(ValueError,match='different action'):jobs.create_task_child(root,parent.name,changed,**options)
    assert len([p for p in root.iterdir() if p.is_dir()])==2


@pytest.mark.parametrize('option,value',[('stored_bytes',1_500_000_000),('pending_jobs',4)])
def test_capacity_refusal_before_child_publication(setup,option,value):
    with pytest.raises(ValueError,match='reserve'):queue(setup,**{option:value})
    assert len([p for p in setup[0].iterdir() if p.is_dir()])==1


def test_parent_change_after_queue_refuses_before_heavy_work(setup,monkeypatch):
    parent,child,_,_=queue(setup);calls=fake_workers(monkeypatch,child)
    write(parent/'status.json',{'status':'ready','lab':'lab','changed':True})
    assert jobs.execute_job(child)==1 and not calls
    assert jobs.read_status(child)['status']=='failed'


def test_started_interrupted_child_is_not_reexecuted(setup,monkeypatch):
    _,child,_,_=queue(setup);calls=fake_workers(monkeypatch,child)
    write(child/'task-job/start.json',{'preserved':'old attempt'})
    assert jobs.execute_job(child)==1 and not calls
    assert jobs.read_status(child)['status']=='interrupted'
    assert json.loads((child/'task-job/start.json').read_bytes())=={'preserved':'old attempt'}


def test_cancel_before_start_never_admits_or_creates_lab(setup,monkeypatch):
    _,child,_,_=queue(setup);calls=fake_workers(monkeypatch,child)
    write(child/'cancel-request.json',{})
    assert jobs.execute_job(child)==1 and not calls
    assert jobs.read_status(child)['status']=='cancelled'
    assert not (child/'task-job/start.json').exists()


def test_failed_training_retains_original_failure_and_never_evaluates(setup,monkeypatch):
    from weft import moving_equipment_learning as train
    _,child,_,_=queue(setup,'train_evaluate');calls=fake_workers(monkeypatch,child)
    def fail(value,out,**settings):
        out.mkdir();write(out/'failure.json',{'status':'failed','failure':'original trial limit','control_steps':3});raise RuntimeError('worker saw failure')
    monkeypatch.setattr(train,'train_moving_equipment_policy',fail)
    assert jobs.execute_job(child)==1 and calls==['admit','protocol']
    assert jobs.summary(child)['report']['failure']=='original trial limit'
    before=pin(child/'task-job/training/failure.json');assert jobs.execute_job(child)==1
    assert pin(child/'task-job/training/failure.json')==before


def test_source_mutation_during_worker_prevents_success_publication(setup,monkeypatch):
    from weft import moving_equipment_task_lab as labs
    parent,child,_,_=queue(setup);fake_workers(monkeypatch,child);original=labs.prepare_moving_equipment_task_lab
    def changed(*args,**kwargs):
        result=original(*args,**kwargs);write(parent/'lab/run/unexpected.json',{});return result
    monkeypatch.setattr(labs,'prepare_moving_equipment_task_lab',changed)
    assert jobs.execute_job(child)==1 and jobs.read_status(child)['status']=='failed'
    assert (child/'lab/run/manifest.json').exists()


def test_summary_checks_small_report_pin_without_source_admission(setup,monkeypatch):
    _,child,_,_=queue(setup);fake_workers(monkeypatch,child);assert jobs.execute_job(child)==0
    write(child/'task-job/report.json',{'status':'fake success'})
    with pytest.raises(ValueError):jobs.summary(child)


@pytest.mark.parametrize('form',['absolute','relative_escape'])
def test_coordinated_parent_path_rewrite_refuses_before_admission(setup,monkeypatch,form):
    parent,child,_,_=queue(setup);calls=fake_workers(monkeypatch,child)
    request=json.loads((child/'task-job/request.json').read_bytes());job=json.loads((child/'job.json').read_bytes())
    request['parent_id']=str(parent) if form=='absolute' else '../'+parent.name
    request['action_identity']=jobs._identity(request['operation'],request['payload'],request['parent_id'],request['parent_files'])
    write(child/'task-job/request.json',request)
    job.update(parent_id=request['parent_id'],task_action_identity=request['action_identity'],task_request_sha256=pin(child/'task-job/request.json'))
    write(child/'job.json',job)
    assert jobs.execute_job(child)==1 and not calls
    assert 'exact portal project ID' in jobs.read_status(child)['error']


def test_cancel_during_final_readback_stops_before_report_publication(setup,monkeypatch):
    _,child,_,_=queue(setup);fake_workers(monkeypatch,child);original=jobs._tree
    def scan(path,deadline):
        result=original(path,deadline)
        if path==child:write(child/'cancel-request.json',{})
        return result
    monkeypatch.setattr(jobs,'_tree',scan)
    assert jobs.execute_job(child)==1
    assert jobs.read_status(child)['status']=='cancelled'
    assert jobs.summary(child)['report']['status']=='failed'
    assert not (child/'task-job/receipt.json').exists()


def test_finite_trial_artifacts_include_refusals_not_arbitrary_paths():
    assert {'training/settings.json','training/generation-4.json','evaluation/start-16.json',
            'evaluation/nominal-start.json','evaluation/plan.json'} <= jobs.ARTIFACTS
    assert not {'training/generation-5.json','evaluation/start-17.json','../job.json','implementation/scorer.py'} & jobs.ARTIFACTS


def test_actual_analytic_queued_create_train_evaluate(tmp_path,monkeypatch):
    """Real geometry/scorer/learning; only original source/body admission is fixture-stubbed."""
    import test_root_motion
    from test_moving_equipment_proposal import packet
    from test_moving_depth_family import settings,choice
    from weft.moving_depth_family import prepare_moving_depth_family,materialize_moving_depth_selection
    from weft.shared_world_exploration import prepare_shared_world_exploration
    root=tmp_path/'actual-portal';root.mkdir();parent=root/('c'*32);parent.mkdir()
    fd=os.open(root/'portal.lock',os.O_RDWR|os.O_CREAT,0o600)
    try:
        monkeypatch.setattr(test_root_motion,'HASH',sha256(b'Analytical bytes; full body admission is stubbed in this fixture.').hexdigest())
        inputs,_=packet(monkeypatch,raw_tracks=1,image_shift=(300,190))
        family=prepare_moving_depth_family(inputs,settings())
        bundle=materialize_moving_depth_selection(family,choice(family,['raw-0']))
        prepared=prepare_shared_world_exploration(inputs,bundle,'candidate-1',parent/'lab',depth_family=family)
        scene=json.loads((parent/'lab/run/scene.json').read_bytes())
        parent_job={'id':parent.name,'kind':'lab','name':'Analytic source'}
        write(parent/'job.json',parent_job);write(parent/'status.json',{'status':'ready','lab':'lab'})
        inventory={parent.name:parent_job}
        def options():return dict(lease_fd=fd,existing_jobs=inventory,stored_bytes=0,reserved_bytes=0,pending_jobs=0,deadline=time.monotonic()+10)
        request={'request_id':'d'*32,'parent_manifest_sha256':prepared['manifest_sha256'],'task':{
            'target_object_id':'raw-0','target_position_world':scene['objects'][0]['position'],
            'position_tolerance_world':.05,'speed_tolerance_world_s':.1,'hold_duration_s':.04}}
        created=jobs.create_task_child(root,parent.name,request,**options());task_child=root/created['id']
        assert jobs.execute_job(task_child)==0,jobs.read_status(task_child)
        inventory[created['id']]=created
        request={'request_id':'e'*32,'task_sha256':jobs.summary(task_child)['report']['task_sha256'],
            'training':{'seed':1729,'generations':1,'population':4,'max_seconds':10},
            'evaluation':{'evaluation_seeds':[4101,4102],'position_jitter':.025,'velocity_jitter':.1,'max_seconds':10}}
        trained=jobs.create_learning_child(root,task_child.name,request,**options());report_child=root/trained['id']
        assert jobs.execute_job(report_child)==0,jobs.read_status(report_child)
        report=jobs.summary(report_child)
        assert report['status']=='complete' and report['report']['policy_installed'] is False
        assert report['report']['evaluation']['requested_starts']==2
        assert 'task-job/training/generation-1.json' in report['artifacts']
        assert 'task-job/evaluation/start-2.json' in report['artifacts']
    finally:os.close(fd)
