from copy import deepcopy
import math

from gymnasium.utils.env_checker import check_env
import numpy as np
import pytest

from weft.spatial_rigid import PROFILE, UNITS, SpatialRigidEnv, validate_scene, validate_snapshot


def scene():
    common = {'shape': 'box', 'quaternion': [1, 0, 0, 0], 'velocity': [0, 0, 0], 'angular_velocity': [0, 0, 0],
              'mass': 1, 'friction': [.25, .001, .0001], 'color': '#AB7E48', 'provenance': {'scope': 'analytic fixture'}}
    return {'schema_version': 1, 'profile': PROFILE, 'units': deepcopy(UNITS), 'bounds': [1, .8, .8], 'dt': 1/120,
            'gravity': [0, 0, -9.81], 'contact': {'time_constant': .01, 'damping_ratio': 1},
            'objects': [common | {'id': 'block', 'label': 'Block', 'role': 'active', 'position': [.25, .4, .3], 'size': [.06, .05, .05]},
                        common | {'id': 'floor', 'label': 'Floor', 'role': 'static', 'position': [.5, .4, .02], 'size': [.98, .78, .04]}],
            'controlled_id': 'block', 'target': [.8, .6, .065],
            'task': {'position_tolerance': .025, 'speed_tolerance': .05, 'acceleration': 5, 'max_steps': 300},
            'assumptions': ['Analytic simulation fixture, not a reconstructed physical scene']}


def test_free_fall_and_contact_have_physical_dimensions_and_complete_snapshots():
    value = scene();env = SpatialRigidEnv(value);env.reset(seed=3)
    for _ in range(12):env.step(0)
    state = env.snapshot();body = state['objects'][0];duration = 12*value['dt']
    assert body['position'][2] == pytest.approx(.3 - .5*9.81*duration**2, abs=.0005)
    assert body['velocity'][2] == pytest.approx(-9.81*duration, abs=1e-10)
    for _ in range(48):env.step(0)
    state=env.snapshot();validate_snapshot(state, env.scene)
    assert ['block', 'floor'] in state['contacts']
    assert state['contact_count'] == len(state['contact_details']) > 0
    assert max(-c['distance'] for c in state['contact_details']) < .001
    assert sum(c['normal_force'] for c in state['contact_details']) == pytest.approx(9.81, abs=.02)
    assert state['objects'][0]['position'][2] == pytest.approx(.065, abs=.001)
    assert env.render().shape[2] == 3


def collision_scene(offset=0):
    value=scene();value['gravity']=[0,0,0];value['dt']=1/120
    body=value['objects'][0];body.update(position=[.25,.4+offset,.3],size=[.08,.08,.08],velocity=[.6,0,0],friction=[0,0,0])
    value['objects']=[body,deepcopy(body)|{'id':'target','label':'Target','role':'passive','position':[.5,.4,.3],'velocity':[0,0,0]}]
    return value


def test_off_center_collision_exposes_lateral_translation_and_world_spin():
    states=[]
    for offset in (0,.025):
        env=SpatialRigidEnv(collision_scene(offset));env.reset();contacts=set()
        for _ in range(72):
            env.step(0);contacts.update(tuple(pair) for pair in env.snapshot()['contacts'])
        assert ('block','target') in contacts
        state=env.snapshot();states.append(state)
        momentum=np.sum([np.array(o['velocity'])*o['mass'] for o in state['objects']],axis=0)
        np.testing.assert_allclose(momentum,[.6,0,0],atol=1e-10)
    assert abs(states[0]['objects'][1]['position'][1]-.4)<1e-6
    assert abs(states[0]['objects'][1]['angular_velocity'][2])<1e-4
    assert abs(states[1]['objects'][1]['position'][1]-.4)>.003
    assert abs(states[1]['objects'][1]['angular_velocity'][2])>.1


def test_seeded_reset_and_full_state_rl_contract_are_deterministic():
    env=SpatialRigidEnv(scene());check_env(env,skip_render_check=True)
    initial=env.reset(seed=7)[0];snapshot=env.snapshot()
    for action in [2,3,7,0]:env.step(action)
    restored=env.reset(seed=7)[0]
    np.testing.assert_array_equal(initial,restored)
    assert env.snapshot()==snapshot
    assert restored.shape==(13*len(env.scene['objects'])+4,)


def test_world_angular_velocity_survives_nonidentity_orientation():
    value=scene();value['objects'][0].update(quaternion=[math.sqrt(.5),0,math.sqrt(.5),0],angular_velocity=[.2,.3,.4])
    env=SpatialRigidEnv(value);env.reset()
    np.testing.assert_allclose(env.snapshot()['objects'][0]['angular_velocity'],[.2,.3,.4],atol=1e-12)


def test_lateral_controls_change_the_world_and_brake_is_a_bounded_force():
    value=scene();value['gravity']=[0,0,0];env=SpatialRigidEnv(value);env.reset()
    for _ in range(12):env.step(3)
    assert env.snapshot()['objects'][0]['position'][1]>.42
    _,_,_,_,info=env.step(7)
    assert np.linalg.norm(info['virtual_force'])<=value['task']['acceleration']*value['objects'][0]['mass']
    assert info['virtual_force'][1]<0


def test_small_ball_retains_rolling_spin_above_twenty_radians_per_second():
    value=scene();body=value['objects'][0];body.pop('size')
    body.update(shape='sphere',radius=.02,position=[.2,.4,.06001],velocity=[.6,0,0],angular_velocity=[0,30,0],friction=[.5,0,0])
    value['objects'][1]['friction']=[.5,0,0]
    env=SpatialRigidEnv(value);env.reset()
    for _ in range(48):env.step(0)
    body=env.snapshot()['objects'][0]
    assert body['position'][0] == pytest.approx(.44,abs=.001)
    assert body['velocity'][0] == pytest.approx(.6,abs=.002)
    assert body['angular_velocity'][1] == pytest.approx(30,abs=.1)


def test_full_state_observation_retains_spin_at_the_declared_limit():
    value=scene();value['gravity']=[0,0,0];body=value['objects'][0];body.pop('size')
    body.update(shape='sphere',radius=.02,angular_velocity=[0,200,0],friction=[0,0,0])
    env=SpatialRigidEnv(value);initial,_=env.reset()
    assert env.observation_space.contains(initial)
    assert initial[11] == 200
    observation,*_=env.step(0)
    assert env.observation_space.contains(observation)
    state=env.snapshot();validate_snapshot(state,env.scene)
    assert state['objects'][0]['angular_velocity'][1] == pytest.approx(200,abs=1e-10)
    state['objects'][0]['angular_velocity'][1]=200+2e-8
    with pytest.raises(ValueError):validate_snapshot(state,env.scene)


@pytest.mark.parametrize('degrees',[10,20])
def test_inclined_box_matches_static_and_sliding_analytic_motion(degrees):
    value=scene();angle=math.radians(degrees)
    tangent=np.array([math.cos(angle),0,-math.sin(angle)]);normal=np.array([math.sin(angle),0,math.cos(angle)])
    center=np.array([.45,.4,.2]);q=[math.cos(angle/2),0,math.sin(angle/2),0]
    value['objects'][0].update(position=(center-.18*tangent+.03501*normal).tolist(),quaternion=q,friction=[.25,0,0])
    value['objects'][1].update(id='ramp',position=center.tolist(),size=[.5,.4,.02],quaternion=q,friction=[.25,0,0])
    env=SpatialRigidEnv(value);env.reset();start=np.array(env.snapshot()['objects'][0]['position'])
    for _ in range(48):env.step(0)
    body=env.snapshot()['objects'][0];a=max(0,9.81*(math.sin(angle)-.25*math.cos(angle)))
    assert float((np.asarray(body['position'])-start)@tangent) == pytest.approx(.5*a*.4**2,abs=.002)
    assert float(np.asarray(body['velocity'])@tangent) == pytest.approx(a*.4,abs=.02)
    assert abs(float((np.asarray(body['position'])-start)@normal))<.001


@pytest.mark.parametrize('change',[
    lambda s:s.update(schema_version=True),lambda s:s['units'].update(length='meter'),
    lambda s:s['objects'][0].update(quaternion=[1,1,0,0]),lambda s:s['objects'][0].update(position=[0,.4,.3]),
    lambda s:s['objects'][1].update(velocity=[1,0,0]),lambda s:s['objects'][0].update(size=[.001,.1,.1]),
    lambda s:s['objects'][0].update(friction=[.2]),lambda s:s['objects'][0].update(id='<include file="secret"/>'),
    lambda s:s.update(gravity=[math.nan,0,0]),lambda s:s['contact'].update(time_constant=.0001),
])
def test_invalid_physical_or_unsafe_scene_contract_is_rejected(change):
    value=scene();change(value)
    with pytest.raises(ValueError):validate_scene(value)


def test_initial_intersections_and_runtime_speed_growth_are_rejected():
    value=scene();value['objects'][0]['position'][2]=.04
    with pytest.raises(ValueError,match='intersect'):SpatialRigidEnv(value).reset()
    value=scene();value['gravity']=[20,0,0];value['objects'][0]['velocity']=[3.99,0,0]
    env=SpatialRigidEnv(value);env.reset()
    with pytest.raises(RuntimeError,match='envelope'):env.step(0)


@pytest.mark.parametrize('field,value',[('quaternion',[0,0,0,0]),('angular_velocity',[0,0,math.inf]),('size',[.1,.1,.1])])
def test_recording_rejects_missing_orientation_nonfinite_spin_and_changed_geometry(field,value):
    env=SpatialRigidEnv(scene());env.reset();state=env.snapshot();state['objects'][0][field]=value
    with pytest.raises(ValueError):validate_snapshot(state,env.scene)
