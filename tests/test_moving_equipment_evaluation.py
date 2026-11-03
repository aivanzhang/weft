"""Analytic reset-variant checks; original sealed source admission is explicitly fixture-stubbed."""

from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from test_moving_equipment_task import example
from test_moving_equipment_learning import policy_fixture
from weft.io import canonical_hash
from weft import moving_equipment_evaluation as evaluation
from weft import moving_equipment_learning as learning


@pytest.fixture
def admitted(example, monkeypatch):
    scene, bundle, task = deepcopy(example)
    value = SimpleNamespace(scene=scene, task=task, bundle=bundle, manifest_sha256="a"*64,
                            parent_manifest_sha256="b"*64, runtime_sha256={})
    def check(candidate):
        if candidate is not value: raise ValueError("Sealed admission required")
        return candidate
    monkeypatch.setattr(evaluation, "_admitted", check)
    monkeypatch.setattr(learning, "_admitted", check)
    return value


def protocol(admitted):
    return evaluation.build_changed_start_protocol(admitted, evaluation_seeds=[101, 102], position_jitter=.005, velocity_jitter=.01)


def test_actual_changed_starts_matched_full_intervals_and_immutable_template(admitted, tmp_path, monkeypatch):
    original = canonical_hash([admitted.scene, admitted.task, admitted.bundle])
    plan = protocol(admitted); policy = policy_fixture(admitted); raw = learning._bytes(policy)
    created=[]; cls=evaluation._ChangedStartPlacementEnv
    class Watched(cls):
        def __init__(self,*args):
            assert (tmp_path/'run/policy.json').read_bytes()==raw
            super().__init__(*args); created.append(self)
        def close(self): self.closed=True; super().close()
    monkeypatch.setattr(evaluation,'_ChangedStartPlacementEnv',Watched)
    result=evaluation.evaluate_changed_starts(admitted,plan,raw,tmp_path/'run',max_seconds=10)
    assert len(created)==1 and created[0].closed
    assert result['admitted_changed_starts']==2 and result['excluded_or_refused_starts']==0
    assert result['control_steps']<=result['maximum_control_steps']==2*4*admitted.task['interval']['horizon_steps']
    assert result['final_holdout_independence']=='not_established'
    starts=[json.loads((tmp_path/f'run/start-{i}.json').read_bytes()) for i in (1,2)]
    nominal=json.loads((tmp_path/'run/nominal-start.json').read_bytes())
    assert len({canonical_hash(nominal),*(row['actual_start_sha256'] for row in starts)})==3
    for row in starts:
        assert row['reset_receipt']['initial_angular_offset_rmse_rad']==0
        assert row['reset_receipt']['initial_root_offset_rmse_model_units']==0
        assert [t['controller'] for t in row['trials']]==list(evaluation.CONTROLLERS)
        for trial in row['trials']:
            if trial['metrics']['completed_interval']:
                assert trial['metrics']['elapsed_s']==pytest.approx(.45)
                assert trial['metrics']['observed_substeps']==16*14
    assert original==canonical_hash([admitted.scene,admitted.task,admitted.bundle])
    manifest=json.loads((tmp_path/'run/manifest.json').read_bytes())
    assert all(sha256((tmp_path/'run'/p).read_bytes()).hexdigest()==pin for p,pin in manifest['files'].items())


@pytest.mark.parametrize('seeds,position,velocity', [([1],.01,.01),([1,1],.01,.01),([True,2],.01,.01),
    ([1,2],0,0),([1,2],.051,.01),([1,2],.01,float('nan')),([1,2],.01,True)])
def test_protocol_bounds(admitted,seeds,position,velocity):
    with pytest.raises(ValueError): evaluation.build_changed_start_protocol(admitted,evaluation_seeds=seeds,position_jitter=position,velocity_jitter=velocity)


@pytest.mark.parametrize('damage',['source','reset','human','interval','scope','extra','version'])
def test_protocol_tampering(admitted,damage):
    value=protocol(admitted)
    if damage=='source': value['source']['task_sha256']='0'*64
    elif damage=='reset': value['reset_options']['jitter']=False
    elif damage=='human': value['reset_options']['jitter_scale']=.1
    elif damage=='interval': value['interval']['horizon_steps']-=1
    elif damage=='scope': value['scope']='independent physical validation'
    elif damage=='version': value['schema_version']=True
    else: value['fallback_override']=True
    with pytest.raises(ValueError): evaluation.validate_changed_start_protocol(value,admitted)


def test_policy_seed_and_bytes_refuse_before_output(admitted,tmp_path):
    p=policy_fixture(admitted);p['training_seed']=101
    with pytest.raises(ValueError,match='training seed'):
        evaluation.evaluate_changed_starts(admitted,protocol(admitted),learning._bytes(p),tmp_path/'same')
    assert not (tmp_path/'same').exists()
    for raw in (b'{"x":1,"x":2}',b'{"x":NaN}', b'x'*65537):
        with pytest.raises(ValueError): evaluation.evaluate_changed_starts(admitted,protocol(admitted),raw,tmp_path/'bad')
    assert not (tmp_path/'bad').exists()


def test_fallback_and_nominal_are_excluded_without_rollouts(admitted,tmp_path,monkeypatch):
    cls=evaluation._ChangedStartPlacementEnv
    def reset(self,*,seed=None,options=None):
        observation,info=self._engine.reset(seed=seed,options={'jitter':False})
        info['object_perturbations']={'raw-0':{'fallback_unperturbed':seed==101}}
        return observation,info
    monkeypatch.setattr(cls,'reset',reset)
    monkeypatch.setattr(cls,'step',lambda *_a:pytest.fail('Excluded start cannot step'))
    result=evaluation.evaluate_changed_starts(admitted,protocol(admitted),learning._bytes(policy_fixture(admitted)),tmp_path/'excluded')
    assert result['admitted_changed_starts']==0 and result['control_steps']==0
    statuses=[json.loads((tmp_path/f'excluded/start-{i}.json').read_bytes())['status'] for i in (1,2)]
    assert statuses==['excluded_fallback','excluded_duplicate_or_nominal']
    assert all(row['successes']==0 and row['requested_starts']==2 for row in result['comparisons'])


def test_failed_reset_is_retained_not_replaced(admitted,tmp_path,monkeypatch):
    calls=[]
    def reset(self,*,seed=None,options=None): calls.append(seed);raise ValueError('original clearance refusal')
    monkeypatch.setattr(evaluation._ChangedStartPlacementEnv,'reset',reset)
    result=evaluation.evaluate_changed_starts(admitted,protocol(admitted),learning._bytes(policy_fixture(admitted)),tmp_path/'refused')
    assert calls==[101,102] and result['excluded_or_refused_starts']==2
    assert all(json.loads((tmp_path/f'refused/start-{i}.json').read_bytes())['status']=='reset_refused' for i in (1,2))


def test_partial_failure_retains_policy_and_last_metrics(admitted,tmp_path,monkeypatch):
    original=evaluation._check;calls=0
    def fail(deadline):
        nonlocal calls
        calls+=1
        if calls==9: raise TimeoutError('bounded fixture stop')
        original(deadline)
    monkeypatch.setattr(evaluation,'_check',fail)
    with pytest.raises(TimeoutError):
        evaluation.evaluate_changed_starts(admitted,protocol(admitted),learning._bytes(policy_fixture(admitted)),tmp_path/'partial')
    failure=json.loads((tmp_path/'partial/failure.json').read_bytes())
    assert failure['control_steps']>0 and failure['last_metrics']['success'] is False
    assert (tmp_path/'partial/policy.json').is_file() and (tmp_path/'partial/manifest.json').is_file()


def test_aggregate_full_interval_reservation(admitted):
    admitted.task['interval']['horizon_steps']=1800
    with pytest.raises(ValueError,match='20000'):
        evaluation.build_changed_start_protocol(admitted,evaluation_seeds=[1,2,3])


def test_saved_wrapper_single_admission_and_input_isolation(tmp_path,monkeypatch):
    import weft.moving_equipment_task_lab as lab
    root=tmp_path/'lab';root.mkdir();(root/'lab-config.json').write_text('{}')
    p=tmp_path/'protocol.json';p.write_text('{}');w=tmp_path/'policy.json';w.write_text('{}')
    calls=[];marker=object()
    def load(run,config,**kwargs): calls.append((run,kwargs));return marker
    monkeypatch.setattr(lab,'load_moving_equipment_task_lab',load)
    def evaluate(a,protocol,raw,out,**kwargs):
        assert a is marker and raw==b'{}' and 0<kwargs['max_seconds']<=60
        return {'status':'complete'}
    monkeypatch.setattr(evaluation,'evaluate_changed_starts',evaluate)
    result=evaluation.evaluate_saved_changed_starts(root,p,w,tmp_path/'result',max_seconds=90)
    assert result['source_admissions']==1 and len(calls)==1 and calls[0][1]['max_seconds']<=30
    with pytest.raises(ValueError,match='isolated'): evaluation.evaluate_saved_changed_starts(root,p,w,root/'out')
    assert len(calls)==1


def test_saved_wrapper_authored_protocol_after_single_admission(admitted,tmp_path,monkeypatch):
    import weft.moving_equipment_task_lab as lab
    root=tmp_path/'lab';root.mkdir();(root/'lab-config.json').write_text('{}')
    policy=tmp_path/'policy.json';policy.write_text('{}');calls=[]
    def load(*args,**kwargs): calls.append('load');return admitted
    monkeypatch.setattr(lab,'load_moving_equipment_task_lab',load)
    def evaluate(a,p,raw,out,**kwargs):
        assert calls==['load'] and p['evaluation_seeds']==[101,102]
        assert evaluation.validate_changed_start_protocol(p,a)==p
        calls.append('evaluate');return {'status':'complete'}
    monkeypatch.setattr(evaluation,'evaluate_changed_starts',evaluate)
    result=evaluation.evaluate_saved_changed_starts(root,None,policy,tmp_path/'result',evaluation_seeds=[101,102])
    assert calls==['load','evaluate'] and result['source_admissions']==1
    for protocol_file,seeds in ((None,None),(policy,[101,102])):
        with pytest.raises(ValueError,match='exactly one'):
            evaluation.evaluate_saved_changed_starts(root,protocol_file,policy,tmp_path/'bad',evaluation_seeds=seeds)
    assert calls==['load','evaluate']


def test_matched_reset_drift_preserves_failure_instead_of_comparing(admitted,tmp_path,monkeypatch):
    original=evaluation._ChangedStartPlacementEnv.reset;count=0
    def reset(self,**kwargs):
        nonlocal count
        result=original(self,**kwargs);count+=1
        if count==2: result[1]['fixture_changed']=True
        return result
    monkeypatch.setattr(evaluation._ChangedStartPlacementEnv,'reset',reset)
    with pytest.raises(ValueError,match='Matched controller reset'):
        evaluation.evaluate_changed_starts(admitted,protocol(admitted),learning._bytes(policy_fixture(admitted)),tmp_path/'drift')
    failure=json.loads((tmp_path/'drift/failure.json').read_bytes())
    assert len(failure['starts'][0]['trials'])==1
    assert not (tmp_path/'drift/result.json').exists()


def test_final_runtime_read_overrun_cannot_publish_success(tmp_path,monkeypatch):
    clock=[0.];calls=[];dummy=object()
    monkeypatch.setattr(evaluation.time,'monotonic',lambda:clock[0])
    monkeypatch.setattr(evaluation,'_context',lambda _: (dummy,{},{}))
    plan={'evaluation_seeds':[101,102],'maximum_control_steps':8,'source':{}}
    monkeypatch.setattr(evaluation,'validate_changed_start_protocol',lambda *_:plan)
    monkeypatch.setattr(evaluation,'_policy',lambda *_:{'training_seed':1,'weights':[1.]})
    def runtime(*_args):
        calls.append(1)
        if len(calls)==2:clock[0]=61.
        return {},{},{}
    monkeypatch.setattr(learning,'_runtime',runtime)
    monkeypatch.setattr(learning,'_Feedback',lambda *_:None)
    engine=SimpleNamespace(data=SimpleNamespace(qpos=np.zeros(1),qvel=np.zeros(1)),reset=lambda **_:None)
    class FakeEnv:
        _metrics=None
        _engine=engine
        def __init__(self,*_):pass
        def reset(self,**_):raise ValueError('fixture reset refusal')
        def close(self):pass
    monkeypatch.setattr(evaluation,'_ChangedStartPlacementEnv',FakeEnv)
    dummy=SimpleNamespace(bundle={})
    with pytest.raises(TimeoutError,match='aggregate deadline'):
        evaluation.evaluate_changed_starts(dummy,plan,b'{}',tmp_path/'late')
    assert len(calls)==2 and not (tmp_path/'late/result.json').exists()
    assert json.loads((tmp_path/'late/failure.json').read_bytes())['status']=='failed'
