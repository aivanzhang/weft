"""Authored tilted-plane geometry, not evidence of a physical camera calibration."""

from copy import deepcopy
import json
import math
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import pytest

from weft.human import HumanEnv, human_xml, validate_human
from weft.mobile_human import GROUND_SCOPE, calibrate_floor, ground_normal, validate_mobility
from test_whole_body_feasibility import model


def inclined(normal=(.3, .4, math.sqrt(.75))):
    value = model()
    value['mobility']['ground_plane'] = {
        'schema_version': 1, 'normal': list(normal), 'method': 'camera_relative_support_plane_v1',
        'source_sha256': value['source_sha256'], 'evidence_ids': [value['reference_evidence'][0]['evidence_id']],
        'provenance': 'Authored analytic plane; no measured camera or gravity.',
        'gravity_assumption': 'authored_normal_alignment_not_measured'}
    return value


def distances(env):
    normal = ground_normal(env.human_spec)
    result = {}
    for index in range(env.model.ngeom):
        if not env.model.geom_contype[index] & 2:
            continue
        center = env.data.geom_xpos[index]
        radius = env.model.geom_size[index, 0]
        points = [center]
        if env.model.geom_type[index] == mujoco.mjtGeom.mjGEOM_CAPSULE:
            axis = env.data.geom_xmat[index].reshape(3, 3)[:, 2]
            half = env.model.geom_size[index, 1]
            points = [center + half * axis, center - half * axis]
        result[index] = min(float(normal @ point - radius) for point in points)
    return result


def exact_first_pose(env):
    env.reset(options={'jitter': False})
    # Task reset clips angles away from limits; calibration uses the exact reference instead.
    for name, angle in env.human_spec['reference'][0]['joints'].items():
        env.data.qpos[env._q[name]] = angle
    mujoco.mj_forward(env.model, env.data)


def test_plane_normal_gravity_and_global_coordinates_match_analytic_geometry():
    value = inclined()
    original = deepcopy(value)
    env = HumanEnv(value)
    exact_first_pose(env)
    normal = ground_normal(value)
    np.testing.assert_allclose(env.model.opt.gravity, -9.81 * normal, atol=1e-12)
    floor = env.mobility.floor
    np.testing.assert_allclose(env.data.geom_xmat[floor].reshape(3, 3)[:, 2], normal, atol=1e-12)
    np.testing.assert_array_equal(env.data.geom_xpos[floor], [0, 0, 0])
    for index, axis in enumerate(('root_x', 'root_y', 'root_z')):
        joint = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_JOINT, axis)
        np.testing.assert_array_equal(env.model.jnt_axis[joint], np.eye(3)[index])
    assert value == original
    assert all(row['translation'][1] == 0 for row in value['root_reference'])


def test_released_body_accelerates_along_authored_normal_without_virtual_support():
    env = HumanEnv(inclined())
    exact_first_pose(env)
    env.mobility.prepare(6)
    env.mobility.apply(6)
    assert env.data.ncon == 0
    np.testing.assert_array_equal(env.data.ctrl, np.zeros(env.model.nu))
    mujoco.mj_step(env.model, env.data)
    np.testing.assert_allclose(env.data.qvel[env.mobility.v], env.model.opt.gravity * env.model.opt.timestep, atol=1e-10)


@pytest.mark.parametrize('normal', [(0., 0., 1.), (.6, 0., .8), (0., -.6, .8), (.3, .4, math.sqrt(.75))])
def test_first_pose_clearance_and_actual_contacts_use_signed_plane_distance(normal):
    value = inclined(normal)
    before = deepcopy(value)
    calibrated = calibrate_floor(value)
    assert value == before
    for key in ('root_reference', 'reference', 'lengths', 'source_sha256', 'reference_evidence'):
        assert calibrated[key] == value[key]
    fit = calibrated['mobility']['floor_fit']
    assert fit['method'] == 'first_pose_inclined_plane_clearance_v1'
    assert fit['future_poses_used'] is False
    assert fit['base_height_shift'] == pytest.approx((.005 - fit['lowest_surface_before']) / normal[2])
    env = HumanEnv(calibrated)
    exact_first_pose(env)
    assert min(distances(env).values()) == pytest.approx(.005, abs=1e-12)
    # Translating global z by 0.04/nz creates exactly 0.035 signed penetration.
    env.data.qpos[env.mobility.q[2]] -= .04 / normal[2]
    mujoco.mj_forward(env.model, env.data)
    expected = distances(env)
    contacts = [contact for contact in env.data.contact if env.mobility.floor in (contact.geom1, contact.geom2)]
    assert contacts
    for contact in contacts:
        other = contact.geom2 if contact.geom1 == env.mobility.floor else contact.geom1
        assert contact.dist == pytest.approx(expected[other], abs=1e-12)
    state = env.mobility.snapshot(env.data)
    assert state['max_penetration'] == pytest.approx(.035, abs=1e-12)
    np.testing.assert_allclose(state['floor_normal'], normal, atol=1e-12)
    assert state['floor_scope'] == GROUND_SCOPE
    json.dumps(state, allow_nan=False)


def test_sphere_can_set_first_pose_clearance_and_contacts():
    value = inclined((0., -.6, .8))
    value['reference'][0]['joints']['torso_flex'] = math.pi
    env = HumanEnv(calibrate_floor(value))
    exact_first_pose(env)
    index = min(distances(env), key=distances(env).get)
    assert mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_GEOM, index) == 'head_geom'
    assert distances(env)[index] == pytest.approx(.005, abs=1e-12)
    env.data.qpos[env.mobility.q[2]] -= .05
    mujoco.mj_forward(env.model, env.data)
    assert any(index in (contact.geom1, contact.geom2) and contact.dist == pytest.approx(-.035)
               for contact in env.data.contact)


def test_future_pose_penetration_is_retained_not_refitted_or_lifted(monkeypatch):
    value = inclined()
    value['root_reference'][1]['translation'][2] = -.2
    original = deepcopy(value)
    def forbidden(*args, **kwargs):
        raise AssertionError('Calibration must not advance dynamics')
    monkeypatch.setattr(mujoco, 'mj_step', forbidden)
    calibrated = calibrate_floor(value)
    assert calibrated['mobility']['floor_fit']['reference_scope'] == 'offset_only_orientation_is_supplied_separately'
    changed = deepcopy(value)
    changed['reference'][1]['joints']['left_hip'] = 1.2
    changed['root_reference'][1]['translation'][2] = -.5
    assert calibrate_floor(changed)['mobility'] == calibrated['mobility']
    assert value == original
    env = HumanEnv(calibrated)
    env.reset(options={'jitter': False})
    first = env.snapshot()
    later = env.reference_snapshot(1)
    assert later['human']['mobility']['max_penetration'] > .1
    assert later['human']['mobility']['translation'] == value['root_reference'][1]['translation']
    assert later['human']['mobility']['support_force'] == [0, 0, 0]
    assert env.snapshot() == first


@pytest.mark.parametrize('mutate', [
    lambda p: p.update(schema_version=True), lambda p: p.update(schema_version=2),
    lambda p: p.update(normal=[0, 0, .9]), lambda p: p.update(normal=[0, 0, -1]),
    lambda p: p.update(normal=[math.sqrt(.96), 0, .2]), lambda p: p.update(normal=[True, 0, 0]),
    lambda p: p.update(normal=[0, float('nan'), 1]), lambda p: p.update(normal=[0, 1]),
    lambda p: p.update(source_sha256='b' * 64), lambda p: p.update(evidence_ids=[]),
    lambda p: p.update(evidence_ids=['b' * 64 + ':frame:60']),
    lambda p: p.update(evidence_ids=['a' * 64 + ':frame:-1']),
    lambda p: p.update(evidence_ids=p['evidence_ids'] * 2),
    lambda p: p.update(evidence_ids=['a' * 64 + ':frame:' + str(i) for i in range(1801)]),
    lambda p: p.update(offset=.1), lambda p: p.update(provenance=' '),
    lambda p: p.update(provenance='x' * 2001), lambda p: p.update(method='measured'),
    lambda p: p.update(gravity_assumption='measured_gravity'),
])
def test_invalid_ground_contract_rejected(mutate):
    value = inclined()
    mutate(value['mobility']['ground_plane'])
    with pytest.raises(ValueError):
        validate_human(value)


def test_stale_normal_calibration_and_nonzero_reference_depth_rejected():
    value = calibrate_floor(inclined())
    value['mobility']['ground_plane']['normal'] = [0, 0, 1]
    with pytest.raises(ValueError, match='exact normal'):
        validate_mobility(value)
    value = inclined()
    value['root_reference'][1]['translation'][1] = .01
    with pytest.raises(ValueError, match='depth'):
        validate_human(value)
    value = calibrate_floor(inclined())
    value['mobility']['floor_fit']['base_height_shift'] += .1
    with pytest.raises(ValueError, match='recorded global-z shift'):
        validate_human(value)


def test_legacy_contract_xml_snapshot_and_calibration_remain_horizontal():
    value = model()
    assert validate_human(value) == value
    tree = ET.fromstring(human_xml(value))
    assert tree.find('option').attrib['gravity'] == '0 0 -9.81'
    assert 'quat' not in tree.find("./worldbody/geom[@name='mobile_floor']").attrib
    calibrated = calibrate_floor(value)
    assert calibrated['mobility']['floor_fit']['method'] == 'first_pose_collision_surface_clearance_v1'
    assert set(calibrated['mobility']['floor_fit']) == {
        'method', 'reference_time_s', 'future_poses_used', 'original_base_height',
        'lowest_surface_before', 'base_height_shift', 'lowest_surface_after'}
    legacy = HumanEnv(calibrated)
    legacy.reset(options={'jitter': False})
    snapshot = legacy.mobility.snapshot(legacy.data)
    assert 'floor_normal' not in snapshot and 'floor_scope' not in snapshot
    explicit = HumanEnv(calibrate_floor(inclined((0., 0., 1.))))
    explicit.reset(options={'jitter': False})
    np.testing.assert_array_equal(legacy.data.qpos, explicit.data.qpos)
    np.testing.assert_array_equal(legacy.data.geom_xpos, explicit.data.geom_xpos)
    np.testing.assert_array_equal(legacy.model.opt.gravity, explicit.model.opt.gravity)


def test_scene_discloses_inferred_plane_not_measured_gravity():
    value = inclined()
    value['assumptions'].append('Camera up is assumed gravity-up; scale is uncalibrated')
    env = HumanEnv(value)
    assert not any('camera up is assumed gravity-up' in text.lower() for text in env.scene['assumptions'])
    assert any('not measured ground or gravity' in text for text in env.scene['assumptions'])
    assert env.scene['human']['mobility']['gravity'] == 'authored normal alignment, not measured gravity'
    assert validate_human(validate_human(value)) == validate_human(value)
