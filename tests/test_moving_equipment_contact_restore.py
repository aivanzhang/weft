"""Pure linear fixtures; none are recovered source contact or physical qualification."""

import numpy as np
import pytest

from weft import moving_equipment_contact_restore as restore


def bounds(n, maximum=2.):
    return np.vstack((np.eye(n),-np.eye(n))), np.full(2*n,-maximum)


def test_blocking_other_arm_moves_even_below_larger_support_deficit():
    C, lower=bounds(2)
    result=restore.restore_contact_direction([[0,1]],[-.2],[[1,0]],[-1.],C,lower)
    assert result['status']=='linear_restoration_complete'
    assert result['step']==pytest.approx([1,.2],abs=1e-8)
    assert result['report']['collision_preservation_verified']
    assert not result['report']['nonlinear_contact_verified']
    assert not result['report']['task_success']


def test_correlated_native_rows_do_not_require_rows_squared_gram():
    C,lower=bounds(2)
    C=np.vstack((C,np.tile([1.,0.],(1000,1))))
    lower=np.r_[lower,np.zeros(1000)]
    r=restore.restore_contact_direction(np.tile([0,1.],(242,1)),np.full(242,-.2),
        np.tile([1.,0.],(242,1)),np.full(242,-1.),C,lower)
    assert r['status']=='linear_restoration_complete'
    assert np.min(C@r['step']-lower)>=-1e-8
    assert all(not i['rows_squared_allocation'] for s in r['report']['stages'] for i in s['iterations'])


def test_support_cannot_undo_resolved_collision():
    C,lower=bounds(1)
    r=restore.restore_contact_direction([[1]],[-.5],[[-1]],[-.5],C,lower)
    assert r['status']=='partial_restoration'
    assert r['step'][0]>=.5-1e-8
    assert r['report']['support_squared_deficit']>.99


def test_hard_bound_blocks_impossible_support_without_false_infeasibility():
    C,lower=bounds(1,.25)
    r=restore.restore_contact_direction(np.zeros((0,1)),[],[[1]],[-1],C,lower)
    assert r['status']=='partial_restoration'
    assert r['step']==pytest.approx([.25])
    assert r['report']['minimum_hard_residual']>=-1e-8
    assert 'infeasible' not in r['status']


def test_zero_width_hard_constraint_and_duplicates_remain_authoritative():
    C=np.array([[1.,0.],[-1.,0.],[1.,0.],[0.,1.],[0.,-1.]])
    lower=np.array([0.,0.,0.,-2.,-2.])
    r=restore.restore_contact_direction([[0,1]],[-.2],[[1,1]],[-1],C,lower)
    assert abs(r['step'][0])<=1e-8
    assert r['step'][1]>=1-1e-8
    assert r['report']['all_original_linear_rows_pass']


def test_row_permutation_does_not_change_qualified_direction():
    C,lower=bounds(2)
    args=([[0,1]],[-.2],[[1,0]],[-1])
    first=restore.restore_contact_direction(*args,C,lower)
    second=restore.restore_contact_direction(*args,C[::-1],lower[::-1])
    assert first['step']==pytest.approx(second['step'],abs=1e-8)


def test_qp_working_set_limit_is_explicit_and_feasible():
    C,lower=bounds(2,.25)
    x,r=restore._qp(np.eye(2),np.array([-1.,-2.]),C,lower,np.zeros(2),restore.time.monotonic()+2,1)
    assert r['status']=='working_set_limit'
    assert np.min(C@x-lower)>=-1e-8


@pytest.mark.parametrize('options',[{'max_seconds':11},{'max_seconds':True},{'max_iterations':25},{'max_working_set_steps':193}])
def test_resource_bounds(options):
    C,l=bounds(1)
    with pytest.raises(ValueError): restore.restore_contact_direction([[1]],[-1],[[1]],[-1],C,l,**options)


def test_nonzero_hard_anchor_is_explicitly_initialized_without_clamp():
    r=restore.restore_contact_direction([[1]],[-1],[[1]],[-1],[[1]],[.1])
    assert r['status']=='linear_restoration_complete'
    assert r['step'][0]>=1-1e-8
    assert r['report']['initial_hard_violations']==[{'row':0,'residual':-.1}]
    assert r['report']['stages'][0]['report']['original_hard_rows_verified']


def test_nonfinite_or_mismatched_rows_refuse():
    C,l=bounds(1)
    with pytest.raises(ValueError): restore.restore_contact_direction([[float('nan')]],[-1],[[1]],[-1],C,l)
    with pytest.raises(ValueError): restore.restore_contact_direction([[1,2]],[-1],[[1]],[-1],C,l)


def test_deadline_during_solver_preserves_feasible_partial_not_complete(monkeypatch):
    C,l=bounds(1)
    def expired(*args): raise TimeoutError('analytic deadline')
    monkeypatch.setattr(restore,'_qp',expired)
    r=restore.restore_contact_direction([[1]],[-1],[[1]],[-1],C,l)
    assert r['status']=='incomplete_resource_bound'
    assert r['step']==[0.]
    assert not r['report']['all_original_linear_rows_pass']


def test_final_deadline_cannot_publish_complete(monkeypatch):
    C,l=bounds(1); clock=[0.]
    monkeypatch.setattr(restore.time,'monotonic',lambda:clock[0])
    def phase(A,values,C,lower,start,*args):
        clock[0]=3.;return start,[]
    monkeypatch.setattr(restore,'_phase',phase)
    r=restore.restore_contact_direction([[1]],[0],[[1]],[0],C,l,max_seconds=2)
    assert r['status']=='incomplete_resource_bound'


def test_limits_immutable():
    with pytest.raises(TypeError): restore.LIMITS['variables']=1000


def test_object_and_oversized_arrays_refuse_before_coercion():
    class Untrusted:
        def __float__(self): raise AssertionError('must not coerce object payload')
    C,l=bounds(1)
    for matrix in (np.array([[Untrusted()]],dtype=object),np.zeros((8193,1)),[[Untrusted()]]):
        with pytest.raises(ValueError): restore.restore_contact_direction(matrix,[-1],[[1]],[-1],C,l)


def test_numerical_solver_failure_preserves_feasible_direction(monkeypatch):
    def failed(*args,**kwargs): raise np.linalg.LinAlgError('analytic singularity')
    monkeypatch.setattr(restore,'_qp',failed)
    C,l=bounds(1)
    r=restore.restore_contact_direction([[1]],[-.2],[[1]],[-1],C,l)
    assert r['status']=='partial_restoration'
    assert r['step']==[0.]
    assert all(s['iterations'][0]['status']=='numerical_linear_algebra_refusal' for s in r['report']['stages'])


@pytest.mark.parametrize('deficit',[5e-11,5e-9,.1])
def test_phase_one_preserves_exact_hard_row_at_small_and_real_deficits(deficit):
    r=restore.restore_contact_direction([[1]],[0],[[1]],[0],[[1],[-1]],[deficit,-1.])
    assert r['status']=='linear_restoration_complete'
    assert r['report']['initial_hard_violations'][0]['residual']==-deficit
    assert r['step'][0]>=deficit-1e-10 and r['step'][0]<=1.+1e-10
    assert r['report']['hard_feasibility_verified']
    assert r['report']['stages'][0]['phase']=='hard_feasibility_initialization'


def test_inconsistent_hard_anchor_returns_typed_refusal_not_global_infeasibility():
    r=restore.restore_contact_direction([[1]],[0],[[1]],[0],[[1],[-1]],[.1,.1])
    assert r['status']=='unsupported_hard_anchor' and r['step'] is None
    assert len(r['report']['initial_hard_violations'])==2
    assert not r['report']['hard_feasibility_verified']
    assert not r['report']['stages'][0]['report']['original_hard_rows_verified']


def test_phase_one_deadline_does_not_escape_as_zero_anchor_value_error(monkeypatch):
    def expired(*args): raise TimeoutError('phase one deadline')
    monkeypatch.setattr(restore,'_feasible_anchor',expired)
    r=restore.restore_contact_direction([[1]],[0],[[1]],[0],[[1]],[.1])
    assert r['status']=='incomplete_resource_bound'
    assert r['step'] is None and r['failure']=='phase one deadline'
    assert r['report']['initial_hard_violations']==[{'row':0,'residual':-.1}]
