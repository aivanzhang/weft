from copy import deepcopy
import json

import numpy as np
import pytest

from test_native_reference_inputs import views
from weft import global_limb_camera_fit as fit
from weft import root_leg_source_fit as source_fit


def test_shape_is_global_symmetric_and_does_not_change_authored_inertia_or_forces(tmp_path):
    human,*_ = source_fit._prepare(*views(tmp_path))
    original=deepcopy(human);engine=fit.SharedShapeGeometry(human)
    try:
        engine.set_shape([.95,1.05]);model=engine.env.model
        for side in ('left','right'):
            assert model.body_pos[model.body(side+'_knee_body').id,2] == pytest.approx(-human['lengths']['thigh']*.95)
            assert model.site_pos[model.site(side+'_ankle_site').id,2] == pytest.approx(-human['lengths']['shin']*1.05)
        assert engine.invariant()
        assert human==original
        engine.set_shape([1.,1.]);assert engine.invariant()
        for value in ([.949,1.],[[1.,1.],[1.,1.]],[np.nan,1.]):
            with pytest.raises(ValueError):engine.set_shape(value)
    finally:engine.close()


def test_shape_derivatives_match_actual_native_geometry_at_multiple_poses(tmp_path):
    human,times,angles,roots,*_=source_fit._prepare(*views(tmp_path))
    engine=fit.SharedShapeGeometry(human)
    try:
        for knot in (0,2):
            q=angles[knot].copy()
            q[list(source_fit.LEGS[0])]=[.2,-.15,.7,.8]
            q[list(source_fit.LEGS[1])]=[-.1,.2,.4,.6]
            engine.set_shape([.99,1.01]);state=engine.source_state(q,roots[knot],True)
            for axis in range(2):
                step=np.eye(2)[axis]*1e-6
                engine.set_shape(np.array([.99,1.01])+step);plus=engine.source_state(q,roots[knot])
                engine.set_shape(np.array([.99,1.01])-step);minus=engine.source_state(q,roots[knot])
                np.testing.assert_allclose((plus['absolute_points']-minus['absolute_points'])/2e-6,
                                           state['shape_point_jacobian'][:,:,axis],atol=1e-8)
        assert engine.invariant()
    finally:engine.close()


def test_schur_simultaneous_step_matches_full_dense_coupled_solution():
    rng=np.random.default_rng(73);design=rng.normal(size=(45,29));H=design.T@design+np.eye(29)*.1;g=rng.normal(size=29)
    a,b=fit._schur_step(H[:5,:5],H[5:,5:],H[:5,5:],g[:5],g[5:])
    np.testing.assert_allclose(np.r_[a,b],-np.linalg.solve(H,g),atol=1e-12)


def test_native_fit_preserves_source_clocks_roots_and_nonleg_axes(tmp_path):
    args=views(tmp_path);saved=tuple(args)
    result=fit.fit_global_limb_camera(*args,maximum_seconds=.5,maximum_iterations=1)
    assert args==saved
    human=json.loads(args[0]);candidate=result['candidate']
    assert candidate['root_reference']==human['root_reference']
    assert [r['time_s'] for r in candidate['reference']]==[r['time_s'] for r in human['reference']]
    protected=set(source_fit.NAMES)-{source_fit.NAMES[i] for i in fit.LEGS}
    for before,after in zip(human['reference'],candidate['reference']):
        assert all(before['joints'][name]==after['joints'][name] for name in protected)
    assert set(candidate['global_shape_factors'])=={'thigh','shin'}
    assert candidate['requires_fixed_inertia_shape_runtime']
    assert result['audit']['inertial_force_settings_unchanged']
    assert result['audit']['nonregression']
    assert result['audit']['parameter_bounds']['passed']
    assert result['installed'] is False and result['rollouts']==result['inference_calls']==0
    assert result['journal']
    json.dumps(result,allow_nan=False)


def test_independent_model_rejects_changed_inertial_fingerprint(tmp_path,monkeypatch):
    original=fit.SharedShapeGeometry;created=[]
    class DifferentAudit(original):
        def __init__(self,human):
            super().__init__(human);created.append(self)
            if len(created)==2:self.frozen['body_mass'][1]+=.1
    monkeypatch.setattr(fit,'SharedShapeGeometry',DifferentAudit)
    with pytest.raises(ValueError,match='inerti'):
        fit.fit_global_limb_camera(*views(tmp_path),maximum_seconds=.5,maximum_iterations=1)


def test_independent_floor_audit_prevents_candidate_qualification(tmp_path,monkeypatch):
    original=fit.SharedShapeGeometry;created=[]
    class ChangedFloorAudit(original):
        def __init__(self,human):
            super().__init__(human);created.append(self);self.is_audit=len(created)==2

        def source_state(self,*args,**kwargs):
            state=super().source_state(*args,**kwargs)
            if self.is_audit:state['gaps'][0]-=10.
            return state
    monkeypatch.setattr(fit,'SharedShapeGeometry',ChangedFloorAudit)
    result=fit.fit_global_limb_camera(*views(tmp_path),maximum_seconds=.5,maximum_iterations=1)
    assert result['status']=='rejected_unpromoted'
    assert not result['audit']['nonregression']
    assert not result['audit']['complete_original_floor_gate']
    assert result['installed'] is False


@pytest.mark.parametrize('change,failed_check',[("shape","shape_fraction"),("angle","angle_correction"),
    ("joint","authored_joint_limits"),("rate","native_correction_speed"),("camera","camera_rotation_bound"),
    ("clock","native_clock"),("rotation","proper_camera_rotation")])
def test_detached_parameter_audit_rechecks_each_original_bound(change,failed_check):
    shape=np.ones(2);angles=np.zeros((3,19));correction=np.zeros((3,8));times=np.array([0.,.04,.08]);rotation=np.eye(3)
    if change=='shape':shape[0]=.949
    elif change=='angle':correction[:,0]=.151
    elif change=='joint':correction[:,fit.LEGS.index(source_fit.NAMES.index('left_knee'))]=-.01
    elif change=='rate':correction[1,0]=.061
    elif change=='camera':rotation=fit._rotation([.151,0,0])
    elif change=='clock':times[1]=times[0]
    else:rotation[0,0]=.99
    report=fit.audit_global_parameters(shape,angles,correction,rotation,np.eye(3),times)
    assert not report['passed'] and not report['checks'][failed_check]


def test_detached_parameter_audit_rejects_proper_half_turn_even_when_skew_is_zero():
    report=fit.audit_global_parameters(np.ones(2),np.zeros((3,19)),np.zeros((3,8)),
                                      np.diag([1.,-1.,-1.]),np.eye(3),np.array([0.,.04,.08]))
    assert report['checks']['proper_camera_rotation']
    assert not report['checks']['camera_rotation_bound']
    assert report['camera_rotation_rad']==pytest.approx(np.pi)
    assert not report['passed']
