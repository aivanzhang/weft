import numpy as np
import pytest
from copy import deepcopy
from hashlib import sha256
import json

from test_native_reference_inputs import views
from weft import global_limb_camera_fit as shared
from weft import constrained_limb_camera_fit as constrained
from weft.constrained_limb_camera_fit import solve_source_qp


def test_unconstrained_qp_matches_dense_solution():
    rng=np.random.default_rng(17);D=rng.normal(size=(20,12));H=D.T@D+np.eye(12)*.1;g=rng.normal(size=12)
    result=solve_source_qp(H,g,np.empty((0,12)),np.empty(0))
    np.testing.assert_allclose(result['step'],-np.linalg.solve(H,g),atol=1e-10)
    assert result['report']['optimality_verified']


def test_dependent_opposing_and_redundant_constraints_preserve_feasible_step():
    A=np.array([[1.,0.,0.],[-.5,1.,0.],[-.5,-1.,0.],[0.,0.,1.],[2.,0.,0.]])
    result=solve_source_qp(np.eye(3),np.array([-1.,-.2,-1.]),A,np.zeros(5))
    assert result['step'] is not None
    p=np.array(result['step']);assert np.min(A@p)>=-1e-8
    np.testing.assert_allclose(p,[0.,0.,1.],atol=1e-7)
    assert result['report']['global_nonlinear_infeasibility_claim'] is False


def test_infeasible_linear_box_returns_no_step_not_false_success():
    result=solve_source_qp(np.eye(2),np.zeros(2),np.array([[1.,0.],[-1.,0.]]),np.array([1.,1.]),maximum_sweeps=20)
    assert result['step'] is None
    assert result['report']['primal_maximum_violation']>1e-8
    assert not result['report']['global_nonlinear_infeasibility_claim']


def test_positive_floor_restoration_can_start_outside_the_linear_feasible_set():
    A=np.array([[1.,0.],[0.,1.],[-1.,0.],[0.,-1.],[1.,-1.]])
    b=np.array([.2,.1,-.3,-.3,-.05])
    result=solve_source_qp(np.eye(2),np.zeros(2),A,b)
    assert result['step'] is not None
    assert np.min(A@np.array(result['step'])-b)>=-1e-8
    np.testing.assert_allclose(result['step'],[.2,.1],atol=1e-8)


def test_constraint_units_and_temporal_difference_bounds_are_not_normalized_away():
    A=np.array([[1e4,0.],[-1e4,0.],[1.,-1.],[-1.,1.]])
    b=np.array([0.,-2000.,-.06,-.06])
    result=solve_source_qp(np.eye(2),np.array([-1.,-1.]),A,b)
    assert result['step'] is not None
    assert np.min(A@np.array(result['step'])-b)>=-1e-8


def test_rows_squared_matrix_is_not_needed_for_large_inventory():
    A=np.tile(np.array([[1.,0.,0.],[-1.,0.,0.]]),(2048,1));b=np.tile([0.,-1.],2048)
    result=solve_source_qp(np.eye(3),np.array([-.5,0.,0.]),A,b)
    assert result['step'] is not None
    assert result['report']['workspace_numeric_bytes']<2<<20
    assert not result['report']['rows_squared_allocation']


@pytest.mark.parametrize('kind',['shape','nonfinite','indefinite','asymmetric'])
def test_rejects_invalid_quadratic_contract(kind):
    H=np.eye(2);g=np.zeros(2);A=np.eye(2);b=np.zeros(2)
    if kind=='shape':A=np.eye(3)
    elif kind=='nonfinite':g[0]=np.nan
    elif kind=='indefinite':H[0,0]=-1
    else:H[0,1]=.1
    with pytest.raises(ValueError):solve_source_qp(H,g,A,b)


def test_rejects_ill_conditioned_hessian_without_adding_a_hidden_prior():
    with pytest.raises(ValueError):solve_source_qp(np.diag([1e-20,1.]),np.zeros(2),np.eye(2),np.zeros(2))


def test_feasible_progress_never_widens_original_primal_tolerance():
    A=np.array([[1.,0.],[0.,1.],[-1.,0.],[0.,-1.],[1.,-2.],[-2.,1.]])
    b=np.array([0.,0.,-1.,-1.,-.1,-.1])
    result=solve_source_qp(np.eye(2),np.array([-1.,-2.]),A,b,maximum_sweeps=1,feasible_initial=np.zeros(2))
    assert result['step'] is not None
    p=np.array(result['step']);assert np.min(A@p-b)>=-1e-8
    assert .5*p@p+np.array([-1.,-2.])@p<0
    if result['report']['feasible_progress_backoff']:
        assert result['report']['feasible_progress_backoff']['original_constraints_reverified']
        assert not result['report']['optimality_verified']


def test_unverified_initial_point_cannot_enable_progress_fallback():
    with pytest.raises(ValueError,match='feasible initial'):
        solve_source_qp(np.eye(2),np.zeros(2),np.eye(2),np.ones(2),feasible_initial=np.zeros(2))


def seed_fixture(args):
    human,times,angles,roots,target,confidence,scale,rotation,center,K=shared.source_fit._prepare(*args)
    result={'profile':shared.PROFILE,'contract':shared.CONTRACT,
        'bindings':dict(zip(('human','original_evidence','derived_evidence','dimensions'),(sha256(v).hexdigest() for v in args))),
        'camera':{'rotation':rotation.tolist(),'rotation_original':rotation.tolist(),'K':K.tolist(),'center':center.tolist()},
        'candidate':{'reference':deepcopy(human['reference']),'root_reference':deepcopy(human['root_reference']),
            'global_shape_factors':{'thigh':1.,'shin':1.},'lengths':deepcopy(human['lengths'])}}
    return json.dumps(result).encode()


def test_new_strategy_reuses_original_gates_and_native_clock(tmp_path):
    args=views(tmp_path);seed=seed_fixture(args)
    result=constrained.fit_constrained_limb_camera(*args,initial_candidate_bytes=seed,
        expected_candidate_sha256=sha256(seed).hexdigest(),maximum_seconds=2.,maximum_iterations=1)
    human=json.loads(args[0])
    assert result['profile']==constrained.PROFILE
    assert result['strategy']['seed_sha256']==sha256(seed).hexdigest()
    assert not result['strategy']['source_or_task_gates_changed']
    assert result['candidate']['root_reference']==human['root_reference']
    assert [r['time_s'] for r in result['candidate']['reference']]==[r['time_s'] for r in human['reference']]
    assert result['audit']['parameter_bounds']['passed']
    assert result['audit']['nonregression']
    assert result['audit']['inertial_force_settings_unchanged']
    assert result['strategy']['phases']


def test_strategy_cannot_mutate_shared_protected_state_or_replace_auditor(tmp_path):
    args=views(tmp_path);human,times,angles,roots,*rest=shared.source_fit._prepare(*args)
    class MutatingStrategy:
        profile='analytic-mutating-strategy'
        def seed(self,args,body,gauge):
            body['mobility']['floor_height']=10.
            self.owned=np.zeros((len(times),8))
            return {'shape':np.ones(2),'delta':self.owned,'rotation':gauge['rotation0']}
        def proposals(self,context):
            self.owned[:]=100.
            context['angles'][:]=100.
            context['baseline_states'][0]['gaps'][:]=-100.
            return []
        def receipt(self):return {}
    result=shared.fit_global_limb_camera(*args,maximum_seconds=2.,maximum_iterations=1,_strategy=MutatingStrategy())
    assert result['candidate']['reference']==human['reference']
    assert result['candidate']['root_reference']==human['root_reference']
    assert result['audit']['parameter_bounds']['passed']
    assert result['audit']['nonregression']


@pytest.mark.parametrize('change',['digest','root','upper_body','clock','shape','parameters'])
def test_seed_is_bound_to_original_source_not_new_relaxed_limits(tmp_path,change):
    args=views(tmp_path);value=json.loads(seed_fixture(args))
    if change=='root':value['candidate']['root_reference'][1]['translation'][2]+=.01
    elif change=='upper_body':value['candidate']['reference'][1]['joints']['torso_flex']+=.01
    elif change=='clock':value['candidate']['reference'][1]['time_s']+=.001
    elif change=='shape':value['candidate']['lengths']['torso']*=.99
    elif change=='parameters':value['candidate']['reference'][1]['joints']['left_hip']+=.2
    raw=json.dumps(value).encode();pin='f'*64 if change=='digest' else sha256(raw).hexdigest()
    with pytest.raises(ValueError):
        constrained.fit_constrained_limb_camera(*args,initial_candidate_bytes=raw,expected_candidate_sha256=pin,
                                              maximum_seconds=2.,maximum_iterations=1)


def test_explicit_no_strategy_keeps_default_profile_metadata_and_result(tmp_path):
    args=views(tmp_path)
    a=shared.fit_global_limb_camera(*args,maximum_seconds=2.,maximum_iterations=1)
    b=shared.fit_global_limb_camera(*args,maximum_seconds=2.,maximum_iterations=1,_strategy=None)
    a.pop('seconds');b.pop('seconds')
    assert a==b and a['profile']==shared.PROFILE and 'strategy' not in a
