from copy import deepcopy
from types import SimpleNamespace
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import pytest

from weft.errors import UnsupportedVideo
from weft.mobile_human import (MOBILITY_DEFAULTS, ROOT_AXES, MobileDynamics, add_mobile_actuators,
                               append_mobile_world, calibrate_floor, prepare_mobile, validate_mobility)


def spec():
    return {"schema_version": 3, "profile": "exercise-v3", "mobility": deepcopy(MOBILITY_DEFAULTS),
            "reference": [{"time_s": 2., "joints": {"left_knee": 0.}},
                          {"time_s": 2.25, "joints": {"left_knee": .4}},
                          {"time_s": 2.5, "joints": {"left_knee": .2}}],
            "root_reference": [{"time_s": 2., "translation": [0., 0., 0.]},
                               {"time_s": 2.25, "translation": [.2, 0., -.3]},
                               {"time_s": 2.5, "translation": [.4, 0., 0.]}]}


def model_xml(model_spec):
    root = ET.Element('mujoco')
    ET.SubElement(root, 'option', timestep='.002', gravity='0 0 -9.81')
    world = ET.SubElement(root, 'worldbody')
    parent = append_mobile_world(world, model_spec)
    body = ET.SubElement(parent, 'body', name='torso')
    ET.SubElement(body, 'joint', name='left_knee', type='hinge', axis='0 1 0')
    ET.SubElement(body, 'geom', name='body_sphere', type='sphere', size='.1', mass='1', contype='2', conaffinity='1')
    actuators = ET.SubElement(root, 'actuator')
    ET.SubElement(actuators, 'motor', name='left_knee_motor', joint='left_knee')
    add_mobile_actuators(actuators, model_spec)
    return ET.tostring(root, encoding='unicode')


def environment(model_spec=None):
    model_spec = spec() if model_spec is None else model_spec
    model = mujoco.MjModel.from_xml_string(model_xml(model_spec))
    data = mujoco.MjData(model)
    knee = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, 'left_knee')
    env = SimpleNamespace(model=model, data=data, human_spec=model_spec, joint_names=('left_knee',),
                          _q_array=np.array([model.jnt_qposadr[knee]]), _hold=np.zeros(1), hold_angles={})
    dynamics = MobileDynamics(env)
    dynamics.reset(data)
    mujoco.mj_forward(model, data)
    return env, dynamics


def advance(env, dynamics, action, steps=1):
    dynamics.prepare(action)
    for _ in range(steps):
        dynamics.apply(action)
        mujoco.mj_step(env.model, env.data)
    mujoco.mj_forward(env.model, env.data)


@pytest.mark.parametrize('mutation', [
    lambda d: d['mobility'].update(base_height=float('nan')),
    lambda d: d['mobility'].update(base_height=0),
    lambda d: d['mobility'].update(slide_limits=[-6, 6]),
    lambda d: d['mobility'].update(maximum_force=2001),
    lambda d: d['mobility'].update(root_mass=0),
    lambda d: d['root_reference'][1].update(time_s=2.6),
    lambda d: d['root_reference'][0].update(translation=[.1, 0., 0.]),
    lambda d: d['root_reference'][1].update(translation=[0., 0., 5.1]),
    lambda d: d['root_reference'][1].update(translation=[0., .1, 0.]),
])
def test_mobility_rejects_nonfinite_unbounded_or_misaligned_references(mutation):
    value = spec()
    mutation(value)
    with pytest.raises(ValueError):
        validate_mobility(value)


def test_mobile_model_has_real_independent_slide_axes_and_named_actuators():
    env, dynamics = environment()
    for axis, name in enumerate(ROOT_AXES):
        joint = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_JOINT, name)
        assert env.model.jnt_type[joint] == mujoco.mjtJoint.mjJNT_SLIDE
        np.testing.assert_array_equal(env.model.jnt_axis[joint], np.eye(3)[axis])
    assert dynamics.mass == pytest.approx(1.1)
    assert list(dynamics.actuators) == [1, 2, 3]
    assert list(dynamics.q) == [0, 1, 2]
    dynamics.prepare(7)
    dynamics.apply(7)
    assert env.data.ctrl[0] == 0
    assert env.data.ctrl[dynamics.actuators[0]] == pytest.approx(-1.1 * 6)
    assert env.data.ctrl[dynamics.actuators[2]] == pytest.approx(1.1 * 9.81)


def test_loaded_spec_cannot_bypass_root_reference_gap_guard():
    value = spec()
    for root, angular, timestamp in zip(value['root_reference'], value['reference'], [2., 2.3, 2.6]):
        root['time_s'] = angular['time_s'] = timestamp
    with pytest.raises(ValueError, match='interpolation horizon'):
        validate_mobility(value)


def test_unobserved_reference_depth_does_not_lock_simulated_depth_motion():
    env, dynamics = environment()
    env.data.qpos[dynamics.q[1]] = .2
    env.data.qvel[dynamics.v[1]] = .1
    dynamics.prepare(6)
    dynamics.apply(6)
    mujoco.mj_step(env.model, env.data)
    state = dynamics.snapshot(env.data)
    assert state['translation'][1] > .2 and state['linear_velocity'][1] > 0
    assert all(row['translation'][1] == 0 for row in env.human_spec['root_reference'])


def test_release_disables_every_motor_persistently_and_produces_floor_contact():
    env, dynamics = environment()
    env.data.ctrl[0] = 10
    advance(env, dynamics, 6, 800)
    state = dynamics.snapshot(env.data)
    assert not dynamics.support_enabled and dynamics.mode == 'released'
    assert np.all(env.data.ctrl == 0)
    assert state['translation'][2] < -.8
    assert state['contact_count'] > 0 and state['total_normal_force'] > 0
    assert state['max_penetration'] < .05
    advance(env, dynamics, 0)
    assert not dynamics.support_enabled and np.all(env.data.ctrl == 0)


def test_push_does_not_restore_released_support_and_restore_does_not_teleport():
    env, dynamics = environment()
    advance(env, dynamics, 6, 20)
    old_position = env.data.qpos.copy()
    dynamics.prepare(7)
    dynamics.apply(7)
    assert not dynamics.support_enabled and env.data.ctrl[dynamics.actuators[2]] == 0
    assert env.data.ctrl[dynamics.actuators[0]] < 0
    np.testing.assert_array_equal(env.data.qpos, old_position)
    advance(env, dynamics, 7, 20)
    assert env.data.qvel[dynamics.v[0]] < 0
    position = env.data.qpos.copy()
    dynamics.prepare(9)
    assert dynamics.support_enabled and dynamics.mode == 'hold'
    np.testing.assert_array_equal(env.data.qpos, position)
    np.testing.assert_array_equal(dynamics.hold, position[dynamics.q])
    np.testing.assert_array_equal(env._hold, position[env._q_array])


def test_reference_follow_uses_relative_clock_and_bounds_forces():
    env, dynamics = environment()
    env.data.time = .25
    dynamics.prepare(5)
    dynamics.apply(5)
    expected = dynamics.kp * np.array([.2, 0, -.3]) + dynamics.kd * np.array([.8, 0, 1.2]) + env.data.qfrc_bias[dynamics.v]
    np.testing.assert_allclose(env.data.ctrl[dynamics.actuators], np.clip(expected, -dynamics.limit, dynamics.limit))
    assert np.max(abs(env.data.ctrl[dynamics.actuators])) <= min(2000, dynamics.mass * 3 * 9.81)
    assert dynamics.support_enabled and dynamics.mode == 'follow'


@pytest.mark.parametrize('action', [0, 1, 2, 3, 4, 7, 8])
def test_manual_action_leaves_follow_and_captures_current_root_and_rotation(action):
    env, dynamics = environment()
    dynamics.prepare(5)
    env.data.qpos[dynamics.q] = [.2, .1, -.3]
    env.data.qpos[env._q_array] = [.7]
    dynamics.prepare(action)
    assert dynamics.mode == 'hold' and dynamics.support_enabled
    np.testing.assert_array_equal(dynamics.hold, [.2, .1, -.3])
    np.testing.assert_array_equal(env._hold, [.7])


def test_floor_uses_only_first_pose_and_reference_overlap_has_no_dynamic_force_claim(monkeypatch):
    monkeypatch.setattr('weft.human.human_xml', model_xml)
    original = spec()
    fitted = calibrate_floor(original)
    assert original['mobility']['base_height'] == 1
    assert fitted['mobility']['base_height'] == pytest.approx(.105)
    assert fitted['mobility']['floor_fit']['future_poses_used'] is False
    later_changed = spec()
    later_changed['root_reference'][1]['translation'][2] = -4
    assert calibrate_floor(later_changed)['mobility'] == fitted['mobility']
    env, dynamics = environment(fitted)
    data = mujoco.MjData(env.model)
    dynamics.reference(data, 1)
    mujoco.mj_forward(env.model, data)
    state = dynamics.snapshot(data)
    assert state['scope'] == 'kinematic_overlap_not_dynamic_force'
    assert not state['support_enabled']
    assert state['max_penetration'] > .2 and state['contact_count'] > 0
    assert state['total_normal_force'] == 0 and state['support_force'] == [0., 0., 0.]
    assert all(contact['normal_force'] == 0 for contact in state['contacts'])


def test_prepare_mobile_preserves_root_audit_and_rejects_insufficient_or_foreign_source():
    source = 'a' * 64
    spatial = spec()
    spatial.update(schema_version=2, profile='exercise-v2', source_sha256=source,
                   lengths={'torso': .5, 'thigh': .4, 'shin': .4},
                   body_frame={'scope': 'Root remains fixed'},
                   assumptions=['Pelvis position is anchored; no ground contacts', 'Depth remains uncalibrated'])
    del spatial['root_reference'], spatial['mobility']
    audit = {'status': 'ready', 'source_sha256': source, 'reference': deepcopy(spec()['root_reference'])}
    result, returned = prepare_mobile(spatial, {'source_sha256': source}, root_audit=audit)
    assert result['profile'] == 'exercise-v3' and result['reference'] == spatial['reference']
    assert returned == audit and 'mobility' not in spatial
    assert 'Pelvis position is anchored; no ground contacts' not in result['assumptions']
    assert 'Depth remains uncalibrated' in result['assumptions']
    assert 'Root remains fixed' not in result['body_frame']['scope']
    bad = dict(audit, status='insufficient_reference', quality={'errors': ['Missing origin']})
    with pytest.raises(UnsupportedVideo, match='Missing origin') as error:
        prepare_mobile(spatial, {'source_sha256': source}, root_audit=bad)
    assert error.value.root_motion_audit == bad
    with pytest.raises(ValueError, match='source identity'):
        prepare_mobile(spatial, {'source_sha256': 'b' * 64}, root_audit=audit)


def test_integrated_human_has_22_dynamic_axes_but_only_19_angular_metrics_and_muscle_pairs():
    from weft.human import HumanEnv
    from weft.spatial_pose import SPATIAL_JOINT_NAMES

    value = spec()
    value.update(dt=1 / 30, controller='bias_compensated_pd',
                 lengths={'torso': .5, 'upper_arm': .3, 'forearm': .25, 'thigh': .4,
                          'shin': .4, 'shoulder_width': .4, 'hip_width': .26},
                 assumptions=['Synthetic authored mobile model, not a source-video reconstruction'],
                 motor_limits={name: 120 if name.startswith('torso') else 30 for name in SPATIAL_JOINT_NAMES})
    for row in value['reference']:
        row['joints'] = {name: 0. for name in SPATIAL_JOINT_NAMES}
    value = calibrate_floor(value)
    env = HumanEnv(value)
    try:
        env.reset(seed=0, options={'jitter': False})
        initial = env.snapshot()
        assert env.model.nq == env.model.nv == 22
        assert len(initial['human']['angles']) == 19 and len(initial['human']['muscles']) == 38
        projected = env.reference_snapshot(1)
        np.testing.assert_allclose(np.subtract(projected['human']['joints']['pelvis'], initial['human']['joints']['pelvis']), [.2, 0, -.3], atol=1e-8)
        assert projected['human']['mobility']['scope'] == 'kinematic_overlap_not_dynamic_force'
        for _ in range(20):
            state = env.advance_physics(6)
        assert np.all(env.data.ctrl == 0)
        assert not state['human']['mobility']['support_enabled']
        assert state['human']['mobility']['contact_count'] > 0
        before = env.data.qpos.copy()
        env.mobility.prepare(9)
        np.testing.assert_array_equal(env.data.qpos, before)
    finally:
        env.close()
