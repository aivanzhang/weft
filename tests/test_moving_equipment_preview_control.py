"""Isolated feature math and real analytic scorer; source admission uses an explicit fixture stub."""
from copy import deepcopy
from hashlib import sha256
import json
from types import SimpleNamespace

import numpy as np
import pytest

from test_moving_equipment_learning import admitted,policy_fixture
from test_moving_equipment_task import example
from weft import moving_equipment_preview_control as preview
from weft import moving_equipment_learning as learning
from weft.moving_equipment_task import MovingEquipmentTaskEnv


def fake_control(monkeypatch,*,time=.009,steps=0,terminal=.007,linear=False):
    def position(t):
        if linear:return np.array([2*t,-3*t])
        return np.array([min(t,.01)*10-max(0,t-.01)*20,1.])
    def state():
        q=np.array([.1,-.2]);v=np.array([.3,-.4]);target=position(time);vel=np.array([2.,-3.]) if linear else np.array([10. if time<.01 else -20.,0.])
        f=np.column_stack([4*(target-q),5*(vel-v),np.array([1.,2.])])/30
        return q,v,target,vel,f
    engine=SimpleNamespace(_initialized=True,_done=False,steps=steps,data=SimpleNamespace(time=time),
        _human_state=state,_target_at=lambda t:(position(t),np.array([999.,999.])),_kd=np.array([5.,5.]),human_limits=np.array([30.,30.]),
        human_motor_count=2,group_indices=np.array([1,2]))
    env=SimpleNamespace(_engine=engine,task={})
    scene={'dt':.03,'task':{'horizon_steps':2,'native_interval':{'full_control_steps':1,'terminal_duration_s':terminal}}}
    monkeypatch.setattr(learning,'_Feedback',lambda *args:lambda:np.array([.11,.22,.7,-.4,.1]))
    obj=preview._PreviewControl(env,scene,{'weights':[1.]*15},{'fixture':'not_source_admission'})
    return obj,engine


def test_uneven_knot_reverses_velocity_only_not_position_bias_object_or_gains(monkeypatch):
    controller,engine=fake_control(monkeypatch)
    before=engine._human_state();action=controller();row=controller.last_diagnostic
    assert row['original_instantaneous_target_qvel'][0]==10
    assert row['target_qvel'][0]==pytest.approx((.1-.029*20-.09)/.03)
    assert row['target_qpos']==before[2].tolist()
    assert np.array_equal(np.asarray(row['features'])[:,[0,2]],before[-1][:,[0,2]])
    assert action[2:].tolist()==[.7,-.4,.1]
    assert np.array_equal(controller._weights,np.ones((4,3)))
    assert engine.data.time==.009 and engine.steps==0


def test_terminal_partial_interval_uses_original_short_duration(monkeypatch):
    controller,engine=fake_control(monkeypatch,time=.008,steps=1,terminal=.007)
    controller();row=controller.last_diagnostic
    assert row['action_duration_s']==.007
    assert row['target_qvel'][0]==pytest.approx((0.-.08)/.007)
    assert row['next_target_qpos']==pytest.approx([0.,1.])
    engine._done=True
    with pytest.raises(RuntimeError,match='live reset'):controller()


def test_constant_velocity_trajectory_preserves_original_feature_matrix(monkeypatch):
    controller,engine=fake_control(monkeypatch,linear=True)
    controller();row=controller.last_diagnostic
    assert np.asarray(row['features'])==pytest.approx(engine._human_state()[-1],abs=1e-14)
    detached=controller.contract;detached['fixture']='mutated'
    assert controller.contract['fixture']=='not_source_admission'
    row['action'][0]=99;assert controller.last_diagnostic['action'][0]!=99


@pytest.mark.parametrize('first,last,duration',[([1],[1],0),([1],[1],float('nan')),([True],[1],.1),([1],[1,2],.1),([1],[float('inf')],.1)])
def test_invalid_horizon_features_reject(first,last,duration):
    with pytest.raises(ValueError):preview.interval_average_velocity(first,last,duration)


@pytest.mark.parametrize('value',[[0]*29, [[0]*1000], np.zeros((2,2)), np.zeros(29), [10**10000], ['bad']])
def test_oversized_or_malformed_vector_rejects_before_array_conversion(value,monkeypatch):
    monkeypatch.setattr(preview.np,'asarray',lambda *a,**k:pytest.fail('No allocation before shape admission'))
    with pytest.raises(ValueError,match='bounded finite'):
        preview.interval_average_velocity(value,[0],.1)


def test_huge_duration_and_deadline_are_typed_refusals_before_work(tmp_path,monkeypatch):
    with pytest.raises(ValueError,match='duration'):
        preview.interval_average_velocity([0],[0],10**10000)
    monkeypatch.setattr(learning,'_context',lambda *a:pytest.fail('No admission before deadline validation'))
    with pytest.raises(ValueError,match='deadline'):
        preview.evaluate_preview_control(None,b'',tmp_path/'unused',base_policy_sha256='0'*64,
            baseline_metrics_raw=b'',baseline_metrics_sha256='0'*64,max_seconds=10**10000)


def test_real_prepared_controller_binds_new_profile_and_preserves_source(admitted):
    policy=learning._bytes(policy_fixture(admitted));pin=sha256(policy).hexdigest()
    before=(admitted.scene_bytes,admitted.task_bytes,admitted.bundle_bytes)
    env=MovingEquipmentTaskEnv(admitted.scene,admitted.task,admitted.bundle)
    try:
        controller=preview.prepare_preview_control(admitted,policy,env,base_policy_sha256=pin)
        assert controller.contract['profile']==preview.PROFILE and controller.contract['base_policy_sha256']==pin
        assert controller.contract['feature_profile']==preview.FEATURE_PROFILE
        with pytest.raises(RuntimeError):controller()
        env.reset(seed=1);original=learning._Feedback(env,admitted.scene,admitted.task,[1.]*15)()
        action=controller();count=env._engine.human_motor_count
        assert np.array_equal(action[count:],original[count:]) and np.all(np.abs(action)<=1)
        assert controller.contract['policy_installed'] is False
        assert np.array_equal(env._engine.data.ctrl,np.zeros(env._engine.model.nu))
        assert (admitted.scene_bytes,admitted.task_bytes,admitted.bundle_bytes)==before
    finally:env.close()


def baseline(admission):
    policy=policy_fixture(admission);env=MovingEquipmentTaskEnv(admission.scene,admission.task,admission.bundle)
    try:
        env.reset(seed=policy['training_seed']);feedback=learning._Feedback(env,admission.scene,admission.task,policy['weights'])
        while True:
            _,_,term,trunc,_=env.step(feedback())
            if term or trunc:break
        return learning._bytes(policy),learning._bytes(env.metrics())
    finally:env.close()


def evaluate(admitted,tmp_path,inputs):
    policy,metrics=inputs
    return preview.evaluate_preview_control(admitted,policy,tmp_path/'trial',base_policy_sha256=sha256(policy).hexdigest(),baseline_metrics_raw=metrics,baseline_metrics_sha256=sha256(metrics).hexdigest(),max_seconds=10)


def test_bounded_analytic_trial_keeps_all_native_substeps_and_original_baseline(admitted,tmp_path,monkeypatch):
    inputs=baseline(admitted);created=[];original=preview.MovingEquipmentTaskEnv
    class Counted(original):
        def __init__(self,*a,**k):super().__init__(*a,**k);created.append(self)
        def close(self):self.was_closed=True;super().close()
    monkeypatch.setattr(preview,'MovingEquipmentTaskEnv',Counted)
    result=evaluate(admitted,tmp_path,inputs)
    assert result['status']=='complete',result['failure']
    assert len(created)==1 and created[0].was_closed
    assert result['native_substeps']==admitted.task['interval']['horizon_steps']*admitted.scene['substeps']
    assert result['metrics']['elapsed_s']==pytest.approx(admitted.task['interval']['native_interval']['duration_s'])
    assert result['baseline_replays']==result['training_runs']==0 and result['policy_installed'] is False
    assert (tmp_path/'trial/baseline-metrics.json').read_bytes()==inputs[1]
    assert result['changed_start_evaluation']=='not_run'
    for name,pin in json.loads((tmp_path/'trial/manifest.json').read_bytes())['files'].items():assert sha256((tmp_path/'trial'/name).read_bytes()).hexdigest()==pin


def test_mismatched_policy_pin_cannot_create_model(admitted,tmp_path,monkeypatch):
    policy=learning._bytes(policy_fixture(admitted));metrics=learning._bytes({'task_sha256':learning.canonical_hash(admitted.task)})
    monkeypatch.setattr(preview,'MovingEquipmentTaskEnv',lambda *a,**k:pytest.fail('No model before policy admission'))
    with pytest.raises(ValueError,match='byte pins'):
        preview.evaluate_preview_control(admitted,policy,tmp_path/'trial',base_policy_sha256='0'*64,baseline_metrics_raw=metrics,baseline_metrics_sha256=sha256(metrics).hexdigest())
    assert not (tmp_path/'trial').exists()


def test_runtime_change_retains_unqualified_trial_without_replay(admitted,tmp_path,monkeypatch):
    inputs=baseline(admitted);original=learning._runtime;calls=[]
    def changed(*a,**k):
        hashes,versions,files=original(*a,**k);calls.append(1)
        if len(calls)==4:hashes={**hashes,'changed.py':'0'*64}
        return hashes,versions,files
    monkeypatch.setattr(learning,'_runtime',changed)
    result=evaluate(admitted,tmp_path,inputs)
    assert result['status']=='incomplete' and 'runtime changed' in result['failure']
    assert result['closed'] and result['model_compilations']==1 and not result['runtime_readback_complete']
