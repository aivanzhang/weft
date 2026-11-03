"""Analytical kinematics only; these checks do not establish real contact or task success."""
import time
from copy import deepcopy
import numpy as np
import pytest

from weft import source_contact_trajectory as module
from weft.shared_world_lab import authored_shared_world_scene


def test_fixed_original_camera_projection_jacobian():
    camera={'world_anchor_model_units':[0.,0.,.8],'distance_model_units':2.,'focal_px':240.,'pixel_origin':[320.,240.]}
    points=np.array([[1.2,2.1,1.3],[.7,2.4,.9]]);origin=np.array([1.,2.,0.]);scale=1.2
    _,jac,_=module.project(points,camera,origin,scale)
    for axis in range(3):
        change=np.zeros_like(points);change[:,axis]=1e-6
        numeric=(module.project(points+change,camera,origin,scale)[0]-module.project(points-change,camera,origin,scale)[0])/2e-6
        np.testing.assert_allclose(numeric,jac[:,:,axis],rtol=1e-7,atol=1e-7)


def test_target_envelope_keeps_original_endpoints_and_joint_root_rates():
    times=np.array([0.,.1,.2]);delta=np.zeros((3,7));delta[1,:4]=.1;delta[1,4]=.02
    assert module.bounded(delta,times)
    for index,value in [((0,0),.01),((1,0),.16),((1,4),.04),((2,1),.01)]:
        bad=delta.copy();bad[index]=value;assert not module.bounded(bad,times)
    assert not module.bounded(delta,[0.,0.,.2])


def test_signed_gap_and_endpoint_jacobians_use_exact_named_seven_dofs():
    scene=authored_shared_world_scene();before=deepcopy(scene);engine=module.ContactGeometry(scene)
    n=len(scene['human_spec']['reference']);delta=np.zeros((n,7));deadline=time.monotonic()+5
    state=engine.evaluate([[1,1.]],.1,delta,deadline,True)
    for k in range(7):
        change=delta.copy();change[1,k]=1e-6;plus=engine.evaluate([[1,1.]],.1,change,deadline)
        change[1,k]=-1e-6;minus=engine.evaluate([[1,1.]],.1,change,deadline)
        np.testing.assert_allclose((plus['points']-minus['points'])/2e-6,state['point_jac'][:,:,k],rtol=1e-5,atol=1e-6)
        numerical=(plus['gaps']-minus['gaps'])/2e-6
        np.testing.assert_allclose(numerical[engine.selected],state['gap_jac'][engine.selected,k],rtol=2e-4,atol=1e-5)
    assert scene==before and tuple(engine.names[i] for i in engine.indices)==module.VARIABLES


@pytest.mark.parametrize('seconds',[0.,31.,float('nan'),True])
def test_work_bounds_fail_before_loading_or_native_calls(tmp_path,seconds):
    with pytest.raises(ValueError):module.propose_contact_targets(tmp_path,max_seconds=seconds)


@pytest.mark.parametrize('key,value',[('focal_px',-1.),('distance_model_units',0.),('focal_px',float('nan'))])
def test_invalid_authored_camera_is_not_silently_used(key,value):
    camera={'world_anchor_model_units':[0.,0.,.8],'distance_model_units':2.,'focal_px':240.,'pixel_origin':[320.,240.]}
    camera[key]=value
    with pytest.raises(ValueError,match='positive finite'):
        module.project(np.zeros((1,3)),camera,np.zeros(3),1.)


@pytest.mark.parametrize('missing',['scene.json','exercise-task.json'])
def test_unsealed_scene_or_task_rejected_before_native_read(tmp_path,missing):
    import json
    names={'scene.json','exercise-task.json'}-{missing}
    (tmp_path/'manifest.json').write_text(json.dumps({'files':{name:'a'*64 for name in names}}))
    with pytest.raises(ValueError,match='both appear'):
        module.propose_contact_targets(tmp_path)
