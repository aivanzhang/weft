"""Analytic native geometry only; source admission is explicitly fixture-stubbed."""

from copy import deepcopy
from types import SimpleNamespace
import time

import numpy as np
import pytest

from test_moving_equipment_task import example
from test_moving_equipment_support import admission
from weft.io import canonical_hash
from weft import moving_equipment_contact_search as old
from weft import moving_equipment_support as support
from weft import moving_equipment_bilateral_contact as bilateral
from weft.contact_force_balance import solve_contact_force_balance


def model(example):
    scene, _, task = example
    knots, times = support._schedule(scene, task)
    return bilateral._Bilateral(scene, task, support._Budget(time.monotonic()+10, 5000), knots, times)


def test_bilateral_preserves_every_old_row_except_explicit_upward_hypothesis(example):
    value = model(example)
    try:
        x = np.zeros((len(value.knots), 8))
        a, J, labels, hard = old._Coupled.constraints(value, x, 'left', derivatives=True)
        b, K, names, fixed = value.constraints(x, derivatives=True)
        keep = [i for i,r in enumerate(labels) if r['kind'] != 'upward_normal_scaled']
        right = [i for i,r in enumerate(labels) if r['kind'] == 'nonpenetration:right_elbow_geom']
        assert np.array_equal(b[:len(keep)], a[keep])
        assert np.array_equal(K[:len(keep)], J[keep])
        assert np.array_equal(fixed[:len(keep)], hard[keep])
        assert np.array_equal(b[len(keep):], 1e-4+1e-8-a[right])
        assert np.array_equal(K[len(keep):], -J[right])
        assert len(b)==len(a) and sum(fixed)==sum(hard)
        assert len([r for r in names if r['kind']=='support_gap'])==2*len(value.times)
        assert not any(r['kind']=='upward_normal_scaled' for r in names)
        old._restoration_groups(names,fixed)
        for column in (0,7,8,15):
            hi,lo=x.copy(),x.copy();hi.flat[column]+=1e-6;lo.flat[column]-=1e-6
            high=value.constraints(hi,derivatives=False)[0];low=value.constraints(lo,derivatives=False)[0]
            smooth=np.array([not r['kind'].startswith('floor:') for r in names])
            assert np.allclose(K[smooth,column],((high-low)/2e-6)[smooth],atol=2e-7,rtol=1e-5)
    finally:value.close()


def test_independent_audit_preserves_floor_reference_collision_and_joint_checks(example):
    value=model(example)
    try:
        audit=value.audit(value.times[0],np.zeros(8))
        for side in ('left','right'):
            original=audit['original_single_contact_diagnostics'][side]['checks']
            assert audit['checks'][side]=={k:v for k,v in original.items() if k!='upward_normal' and not k.startswith('static_')}
            assert {'floor','reference_pointwise_conservative','reference_peak','joint_bounds','correction_bound',
                    'object_floor_clearance','object_world_containment','no_other_object_overlap','support_gap'}==set(audit['checks'][side])
        assert audit['force_balance'] is None
    finally:value.close()


def test_search_one_hypothesis_no_physics_no_old_single_contact_fit(admission,monkeypatch):
    from weft import shared_world
    original=shared_world.SharedWorldEnv;created=[]
    class Watched(original):
        def __init__(self,*a,**kw):super().__init__(*a,**kw);created.append(self)
        def step(self,*a,**kw):pytest.fail('No physics')
        def reset(self,*a,**kw):pytest.fail('No reset')
        def close(self):self.was_closed=True;super().close()
    monkeypatch.setattr(shared_world,'SharedWorldEnv',Watched)
    monkeypatch.setattr(support._Geometry,'solve',lambda *a:pytest.fail('No previous point fit'))
    monkeypatch.setattr(bilateral,'solve_contact_force_balance',lambda *a,**kw:pytest.fail('No force claim before geometry passes'))
    before=canonical_hash([admission.scene,admission.task,admission.bundle])
    result=bilateral.search_bilateral_contacts(admission,normal_force_cap=20,max_seconds=10,max_iterations=1)
    assert result['status']=='complete' and result['runtime_readback_complete']
    assert len(created)==1 and created[0].was_closed and created[0].steps==0
    assert result['hypothesis']['status']=='no_witness_in_bounded_search'
    assert [r['time_s'] for r in result['hypothesis']['audit']]==result['schedule']['native_boundary_times_s']
    assert result['physics_steps']==0 and not result['qualification']['task_success']
    assert result['normal_force_cap']['per_contact']==20
    assert canonical_hash([admission.scene,admission.task,admission.bundle])==before


@pytest.mark.parametrize('options',[{'normal_force_cap':None},{'normal_force_cap':True},{'normal_force_cap':float('nan')},
    {'normal_force_cap':10001},{'max_seconds':61},{'max_evaluations':24001},{'max_iterations':7}])
def test_invalid_bounds_refuse_before_native(admission,monkeypatch,options):
    monkeypatch.setattr(bilateral,'_Bilateral',lambda *a:pytest.fail('No native'))
    with pytest.raises(ValueError):bilateral.search_bilateral_contacts(admission,**{'normal_force_cap':20,**options})


def test_admission_and_runtime_reject_before_native(admission,monkeypatch):
    monkeypatch.setattr(bilateral,'_Bilateral',lambda *a:pytest.fail('No native'))
    with pytest.raises(ValueError):bilateral.search_bilateral_contacts(object(),normal_force_cap=20)
    admission.runtime_sha256['recording.py']='0'*64
    with pytest.raises(ValueError,match='runtime changed'):bilateral.search_bilateral_contacts(admission,normal_force_cap=20)


def test_native_budget_preserves_incomplete_not_a_negative_physical_claim(admission):
    result=bilateral.search_bilateral_contacts(admission,normal_force_cap=20,max_seconds=5,max_evaluations=1)
    assert result['status']=='incomplete_resource_bound' and result['native_forward_evaluations']==1
    assert not result['qualification']['global_infeasibility_claim']


def contact_fixture():
    import mujoco
    xml='''<mujoco><option gravity="0 0 -9.81"/><worldbody>
      <body name="left"><joint type="slide" axis="0 0 1"/><geom name="left" type="capsule" fromto="-.2 0 -.2 -.2 0 .2" size=".1" mass=".01" friction="1 .005 .0001" condim="3"/></body>
      <body name="right"><joint type="slide" axis="0 0 1"/><geom name="right" type="capsule" fromto=".2 0 -.2 .2 0 .2" size=".1" mass=".01" friction="1 .005 .0001" condim="3"/></body>
      <body name="object"><freejoint/><geom name="object" type="sphere" size=".1" mass="1" friction=".5 .005 .0001" condim="3"/></body>
    </worldbody></mujoco>'''
    m=mujoco.MjModel.from_xml_string(xml);d=mujoco.MjData(m)
    env=SimpleNamespace(_v=np.array([0,1]),human_limits=np.array([10.,10.]))
    def pose(stamp,indices,delta):
        d.qpos[:]=m.qpos0;d.qpos[indices]+=delta;d.qvel[:]=0;d.time=stamp;mujoco.mj_forward(m,d)
    g=SimpleNamespace(pose=pose,target=np.zeros(3),object={'radius':.1,'mass':1.},ogid=m.geom('object').id,
        scene={'gravity':[0.,0.,-9.81],'task':{'object_goal':{'virtual_acceleration':6.}}})
    value=bilateral._Bilateral.__new__(bilateral._Bilateral)
    value.model,value.data,value.env,value.geometry=m,d,env,g
    value.supports={side:m.geom(side).id for side in ('left','right')}
    value.columns=value.indices=np.array([0,1])
    value.force_pins={name:'a'*64 for name in ('source_sha256','scene_sha256','task_sha256')}
    return value


def test_actual_native_contact_frames_motor_sign_and_friction_make_static_witness():
    value=contact_fixture()
    h,surfaces=value.force_hypothesis(0.,np.zeros(2),10.)
    assert [c['normal_on_object_world'] for c in h['contacts']]==[[1.,0.,0.],[-1.,0.,0.]]
    assert all(c['sliding_friction']==[1.,1.] for c in h['contacts'])
    assert all(abs(r['gap_world'])<=1e-12 for r in surfaces)
    assert np.allclose(h['motor_limits']['matrix'],[[0,0,1,0,0,0],[0,0,0,0,0,1]])
    assert np.allclose(h['motor_limits']['bias'],[.0981,.0981])
    result=solve_contact_force_balance(h)
    assert result['status']=='feasible'
    assert result['witness']['motor_capacity']['status']=='supplied_linear_rows_checked'
    assert max(abs(v) for v in result['witness']['direct_force_world'])<=6+1e-8
    assert value.data.ncon >= 2
    assert all(c.friction[0]==1 and c.friction[1]==1 for c in value.data.contact)
    assert not result['source_authenticated']


def test_compiled_priority_rules_and_unsupported_pair_refusal():
    value=contact_fixture();m=value.model;a=value.supports['left'];b=value.geometry.ogid
    m.geom_priority[b]=2
    assert bilateral._sliding_friction(m,a,b)==[.5,.5]
    m.geom_condim[a]=1
    with pytest.raises(ValueError,match='sliding'):bilateral._sliding_friction(m,a,b)
    with pytest.raises(ValueError,match='explicit contact pairs'):
        bilateral._sliding_friction(SimpleNamespace(npair=1),0,1)


def test_force_requires_original_both_gap_gates():
    value=contact_fixture();value.geometry.target=np.array([0.,0.,1.])
    with pytest.raises(ValueError,match='contact-gap'):value.force_hypothesis(0.,np.zeros(2),10.)


def test_force_refuses_motor_capacity_without_changing_original_caps():
    value=contact_fixture();value.env.human_limits[:]=.01
    h,_=value.force_hypothesis(0.,np.zeros(2),10.)
    result=solve_contact_force_balance(h)
    assert result['status']=='refused' and result['witness'] is None
    assert h['motor_limits']['upper']==[.01,.01]


@pytest.mark.parametrize('phase',['runtime_readback','encoding'])
def test_final_deadline_does_not_publish_complete(admission,monkeypatch,phase):
    original_runtime,original_bytes=old._runtime,old._bytes
    budgets=[]
    class Obstructed:
        names=['analytic']*8
        def __init__(self,*args):pass
        def fixed_obstructions(self):return [{'analytic_fixed_obstruction':True}]
        def close(self):pass
    monkeypatch.setattr(bilateral,'_Bilateral',Obstructed)
    def runtime(value,scene,budget=None):
        budgets.append(budget)
        if len(budgets)==2 and phase=='runtime_readback':budget.deadline=time.monotonic()-1
        return original_runtime(value,scene,budget)
    def encode(result):
        raw=original_bytes(result)
        if phase=='encoding':budgets[-1].deadline=time.monotonic()-1
        return raw
    monkeypatch.setattr(old,'_runtime',runtime);monkeypatch.setattr(old,'_bytes',encode)
    result=bilateral.search_bilateral_contacts(admission,normal_force_cap=20,max_seconds=5)
    assert result['status']=='incomplete_resource_bound' and result['failure']
    assert result['hypothesis']['status']=='immutable_geometry_obstruction'


@pytest.mark.parametrize('field',['geom_priority','geom_condim','geom_solmix'])
def test_additional_compiled_friction_rules_are_immutable(example,field):
    value=model(example)
    getattr(value.model,field)[value.supports['left']]+=1
    with pytest.raises(ValueError,match='protected model parameters changed'):value.close()


@pytest.mark.parametrize('option',['normal_force_cap','max_seconds'])
def test_oversized_integer_refuses_without_float_overflow(admission,option):
    with pytest.raises(ValueError):
        bilateral.search_bilateral_contacts(admission,**{'normal_force_cap':20,option:10**400})
