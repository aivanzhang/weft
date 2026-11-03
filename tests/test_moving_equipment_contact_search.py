"""Analytic coupled contact geometry; no source or dynamic-support qualification."""

from copy import deepcopy
import time

import numpy as np
import pytest

from test_moving_equipment_task import example
from test_moving_equipment_support import admission
from weft.io import canonical_hash
from weft.moving_equipment_task import build_moving_equipment_task
from weft import moving_equipment_support as support
import weft.moving_equipment_contact_search as search


def model(example):
    scene, _, task = example
    knots, times = support._schedule(scene, task)
    return search._Coupled(scene, task, support._Budget(time.monotonic()+10, 5000), knots, times)


def test_continuous_capsule_closest_point_gap_and_normal_jacobians(example):
    value = model(example)
    try:
        delta = np.array([.03,-.02,.01,.04, -.01,.02,-.04,.05])
        gid = value.supports['left']
        value.geometry.pose(.42, value.indices, delta)
        state = search._capsule_state(value.model,value.data,gid,value.columns,value.geometry.target)
        numeric_gap=np.empty(8);numeric_normal=np.empty((3,8))
        for i in range(8):
            high,low=delta.copy(),delta.copy();high[i]+=1e-6;low[i]-=1e-6
            value.geometry.pose(.42,value.indices,high)
            a=search._capsule_state(value.model,value.data,gid,value.columns,value.geometry.target)
            value.geometry.pose(.42,value.indices,low)
            b=search._capsule_state(value.model,value.data,gid,value.columns,value.geometry.target)
            numeric_gap[i]=(a['distance']-b['distance'])/2e-6
            numeric_normal[:,i]=(a['normal']-b['normal'])/2e-6
        assert np.allclose(state['gap_jacobian'],numeric_gap,atol=1e-8,rtol=1e-6)
        assert np.allclose(state['normal_jacobian'],numeric_normal,atol=1e-8,rtol=1e-6)
        assert np.array_equal(state['gap_jacobian'][4:],np.zeros(4))
    finally:value.close()


def test_native_sample_constraint_jacobian_includes_both_arms_and_midpoints(example):
    value=model(example)
    try:
        correction=np.zeros((len(value.knots),8))
        c,J,labels,hard=value.constraints(correction,'left',derivatives=True)
        assert set(x['time_s'] for x in labels)==set(value.times)
        assert len(c)==26*len(value.times) and J.shape==(len(c),8*len(value.knots))
        assert sum(hard)==16*len(value.times)
        kinds={x['kind'] for x in labels}
        assert 'nonpenetration:right_elbow_geom' in kinds and 'nonpenetration:left_elbow_geom' in kinds
        assert 'upward_normal_scaled' in kinds and 'support_gap' in kinds
        for column in (0,7,8,15):
            a,b=correction.copy(),correction.copy();a.flat[column]+=1e-6;b.flat[column]-=1e-6
            high,_,_,_=value.constraints(a,'left',derivatives=False)
            low,_,_,_=value.constraints(b,'left',derivatives=False)
            # Floor min-endpoint kinks are nonsmooth; all smooth rows and exact joint rows still match.
            smooth=np.array([not row['kind'].startswith('floor:') for row in labels])
            assert np.allclose(J[smooth,column],((high-low)/2e-6)[smooth],atol=2e-7,rtol=1e-5)
    finally:value.close()


def test_constraints_refuse_midinterval_elbow_violation_not_only_knot_bounds(example):
    value=model(example)
    try:
        correction=np.zeros((len(value.knots),8));correction[:,3]=-.5
        c,_,labels,hard=value.constraints(correction,'left',derivatives=False)
        elbow=[(v,r['time_s']) for v,r in zip(c,labels) if r['kind']=='joint_lower:left_elbow']
        reference_min=min(value.env._target_at(t)[0][value.indices[3]] for t in value.times)
        assert min(v for v,_ in elbow)==pytest.approx(reference_min-.5)
        assert any(t not in value.knots for _,t in elbow)
        assert all(h for h,r in zip(hard,labels) if r['kind'].startswith('joint_'))
    finally:value.close()


def test_search_keeps_two_coupled_alternatives_and_never_calls_old_fit(admission,monkeypatch):
    from weft import shared_world
    original=shared_world.SharedWorldEnv;created=[]
    class Watched(original):
        def __init__(self,*args,**kwargs):super().__init__(*args,**kwargs);created.append(self)
        def step(self,*args,**kwargs):pytest.fail('No physics steps')
        def reset(self,*args,**kwargs):pytest.fail('No reset/source pose mutation')
        def close(self):self.was_closed=True;super().close()
    monkeypatch.setattr(shared_world,'SharedWorldEnv',Watched)
    monkeypatch.setattr(support._Geometry,'solve',lambda *a:pytest.fail('Old point fits must not be rerun'))
    before=canonical_hash([admission.scene,admission.task,admission.bundle])
    result=search.search_moving_contacts(admission,max_seconds=10,max_iterations=1)
    assert result['status']=='complete' and result['physics_steps']==0
    assert not result['fixed_obstructions'] and result['runtime_readback_complete']
    assert all(row['iterations'] for row in result['alternatives'])
    assert len(created)==1 and created[0].was_closed and created[0].steps==0
    assert result['schedule']['native_boundary_times_s'][-1]==.45 and result['schedule']['subsampling'] is False
    assert [x['support_side'] for x in result['alternatives']]==['left','right']
    assert all(len(x['variables'])==8 for x in result['alternatives'])
    assert not result['qualification']['task_success'] and not result['qualification']['dynamic_contact_verified']
    assert result['native_forward_evaluations']<=24000 and result['unattempted_alternatives']==0
    assert canonical_hash([admission.scene,admission.task,admission.bundle])==before
    for alternative in result['alternatives']:
        if alternative['status']=='immutable_geometry_obstruction':continue
        assert [x['time_s'] for x in alternative['audit']]==result['schedule']['native_boundary_times_s']
        assert alternative['task_success'] is False and alternative['dynamic_contact_verified'] is False
    assert len(search._bytes(result))<=8<<20


def test_immutable_target_obstruction_is_explicit_not_dropped(admission,monkeypatch):
    # An authored center inside the world can still put its sphere through the floor.
    target=list(admission.task['goal']['position_world']);target[2]=admission.scene['objects'][0]['radius']/2
    goal=admission.task['goal']
    admission.task=build_moving_equipment_task(admission.scene,admission.bundle,candidate_id='candidate-1',
        target_object_id='raw-0',target_position_world=target,position_tolerance_world=goal['position_tolerance_world'],
        speed_tolerance_world_s=goal['speed_tolerance_world_s'],hold_duration_s=goal['terminal_consecutive_hold_s'])
    monkeypatch.setattr(search,'_proposal',lambda *a:pytest.fail('No search through fixed obstruction'))
    result=search.search_moving_contacts(admission,max_seconds=5,max_iterations=1)
    assert len(result['alternatives'])==2 and result['fixed_obstructions']
    assert all(x['status']=='immutable_geometry_obstruction' for x in result['alternatives'])
    assert all(any(v['kind']=='object_floor' for v in x['failed']) for x in result['fixed_obstructions'])


@pytest.mark.parametrize('options',[{'max_seconds':61},{'max_seconds':True},{'max_evaluations':24001},{'max_iterations':7}])
def test_bounds_reject_before_native(admission,monkeypatch,options):
    monkeypatch.setattr(search,'_Coupled',lambda *a:pytest.fail('No model with invalid bounds'))
    with pytest.raises(ValueError):search.search_moving_contacts(admission,**options)


def test_resource_exhaustion_keeps_unattempted_scope(admission):
    result=search.search_moving_contacts(admission,max_seconds=5,max_evaluations=1)
    assert result['status']=='incomplete_resource_bound' and result['native_forward_evaluations']==1
    assert result['unattempted_alternatives']==2 and result['physics_steps']==0
    assert not result['qualification']['task_success']


def test_restoration_slack_never_relaxes_joint_or_update_rows():
    calls=[]
    def solve(H,g,A,b,**kwargs):
        calls.append((H,g,A,b,kwargs))
        if len(calls)==1:return {'step':None,'status':'no_linear_step','report':{}}
        initial=kwargs['feasible_initial']
        assert np.min(A@initial-b)>=-1e-10
        assert A[0,-1]==1 and A[1,-1]==0
        assert np.all(A[2:-2,-1]==0)
        return {'step':initial.tolist(),'status':'feasible_restoration_only','report':{}}
    correction=np.zeros((2,8));values=np.array([-.2,.1]);matrix=np.zeros((2,16));matrix[0,0]=1;matrix[1,1]=1
    step,reports,slack=search._proposal(correction,values,matrix,np.array([False,True]),solve,support._Budget(time.monotonic()+5,1))
    assert len(step)==16 and slack>.2
    assert reports[1]['phase']=='search_only_restoration'


def test_runtime_and_admission_refuse_before_geometry(admission,monkeypatch):
    monkeypatch.setattr(search,'_Coupled',lambda *a:pytest.fail('No native after custody rejection'))
    with pytest.raises(ValueError):search.search_moving_contacts(object())
    admission.runtime_sha256['recording.py']='0'*64
    with pytest.raises(ValueError,match='runtime changed'):search.search_moving_contacts(admission)


def test_limits_immutable():
    with pytest.raises(TypeError):search.LIMITS['angle_correction_rad']=2


@pytest.mark.parametrize('phase',['runtime_readback','encoding'])
def test_final_deadline_cannot_publish_complete(admission,monkeypatch,phase):
    original_runtime,original_bytes=search._runtime,search._bytes
    calls=[]
    class FixedObstruction:
        names=['fixture']*8
        def __init__(self,*args):pass
        def fixed_obstructions(self):return [{'explicit_fixture_refusal':True}]
        def close(self):pass
    monkeypatch.setattr(search,'_Coupled',FixedObstruction)
    def runtime(value,scene,budget=None):
        calls.append(budget)
        if len(calls)==2 and phase=='runtime_readback':budget.deadline=time.monotonic()-1
        return original_runtime(value,scene,budget)
    monkeypatch.setattr(search,'_runtime',runtime)
    def encoded(result):
        raw=original_bytes(result)
        if phase=='encoding':calls[-1].deadline=time.monotonic()-1
        return raw
    monkeypatch.setattr(search,'_bytes',encoded)
    result=search.search_moving_contacts(admission,max_seconds=5,max_iterations=1)
    assert result['status']=='incomplete_resource_bound' and result['failure']
    assert result['runtime_readback_complete']==(phase=='encoding')
    assert len(result['alternatives'])==2 and all(x['status']=='immutable_geometry_obstruction' for x in result['alternatives'])
