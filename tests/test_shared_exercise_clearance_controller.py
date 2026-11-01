"""Analytical closest-point mechanics, not native collision or task-success evidence."""

from dataclasses import FrozenInstanceError, replace
from types import SimpleNamespace

import numpy as np
import pytest

from weft import shared_exercise_clearance_controller as clearance
from weft.spatial_pose import SPATIAL_JOINT_NAMES


def fixture(monkeypatch, *, scale=1., parts=(1,), root=True, gravity=(0., 0., -10.)):
    count = 22 if root else 19
    action = np.zeros(count)
    if root:
        action[19:] = [.125, -.25, -0.]
    nv = 28
    model = SimpleNamespace(nv=nv, opt=SimpleNamespace(gravity=np.array(gravity)),
                            geom_bodyid=np.array([1] + [2] * max(parts)))
    data = SimpleNamespace(qpos=np.arange(nv, dtype=float), qvel=np.zeros(nv),
                           ctrl=np.arange(22, dtype=float), qfrc_applied=np.zeros(nv),
                           xfrc_applied=np.zeros((3, 6)))
    names = SPATIAL_JOINT_NAMES + (('root_x', 'root_y', 'root_z') if root else ())
    engine = SimpleNamespace(model=model, data=data, scale=scale, human_mass=1.,
        human_limits=np.full(count, 10.), _v=np.arange(19), human_action_names=names,
        _human_geom_to_name={0: 'right_elbow_geom'}, _object_geometry_ids={'equipment': parts})
    env = SimpleNamespace(action_count=count, action_names=names, _engine=engine,
        scene={'profile': 'shared-world-v2', 'objects': [{'id': 'equipment', 'role': 'passive'}]},
        task={'profile': 'shared-exercise-task-v1', 'equipment_ids': ['equipment'], 'contact_requirements': []})
    calls = {'base': 0, 'distance': 0, 'jacobian': 0}
    gaps = {part: .02 * scale for part in parts}
    direction = np.array([1., 0., 0.])
    jac_h, jac_o = np.zeros((3, nv)), np.zeros((3, nv))
    jac_h[0, 0] = jac_h[0, 19] = scale
    jac_o[0, 22] = 1.

    def base():
        calls['base'] += 1
        return action.copy()

    def distance(m, d, human, part, maximum, fromto):
        assert m is model and d is data and human == 0
        calls['distance'] += 1
        gap = gaps[part]
        if gap < maximum:
            fromto[:3] = direction * gap
            fromto[3:] = 0.
        return min(gap, maximum)

    def jacobian(m, d, jac, angular, point, body):
        assert m is model and d is data and angular is None
        calls['jacobian'] += 1
        jac[:] = jac_h if body == 1 else jac_o

    env.authored_action = base
    monkeypatch.setattr(clearance.mujoco, 'mj_geomDistance', distance)
    monkeypatch.setattr(clearance.mujoco, 'mj_jac', jacobian)
    return SimpleNamespace(env=env, action=action, calls=calls, gaps=gaps, direction=direction,
                           jac_h=jac_h, jac_o=jac_o, model=model, data=data)


def test_positive_and_penetrating_signed_gaps_push_outward_without_state_mutation(monkeypatch):
    item = fixture(monkeypatch)
    before = {name: value.tobytes() for name, value in vars(item.data).items()}
    controller = clearance.prepare_clearance_controller(item.env)
    action, diagnostic = controller.sample()
    assert action[0] == pytest.approx(.04)
    assert diagnostic['reported_pairs'][0]['distance_gradient_model_units'][0] == 1.
    assert action[19:].tobytes() == item.action[19:].tobytes()
    item.gaps[1] = -.01
    penetrating, report = controller.sample()
    assert penetrating[0] == pytest.approx(.25)
    assert report['reported_pairs'][0]['normal_world'] == [1., 0., 0.]
    assert all(value.tobytes() == before[name] for name, value in vars(item.data).items())
    assert item.calls == {'base': 2, 'distance': 2, 'jacobian': 4}


def test_closing_equipment_and_root_motion_affect_response_but_not_root_actions(monkeypatch):
    item = fixture(monkeypatch)
    item.gaps[1] = .05
    controller = clearance.prepare_clearance_controller(item.env)
    assert controller().tobytes() == item.action.tobytes()
    item.data.qvel[22] = .4
    action, report = controller.sample()
    assert action[0] == pytest.approx(.12)
    assert report['reported_pairs'][0]['normal_speed_model_units_s'] == pytest.approx(-.4)
    item.data.qvel[19] = .4
    assert controller().tobytes() == item.action.tobytes()
    assert action[19:].tobytes() == item.action[19:].tobytes()


def test_receding_velocity_does_not_create_attraction(monkeypatch):
    item = fixture(monkeypatch)
    item.data.qvel[0] = 10.
    controller = clearance.prepare_clearance_controller(item.env)
    assert controller()[0] == pytest.approx(.04)
    item.gaps[1] = .05
    assert controller().tobytes() == item.action.tobytes()


def test_world_rotation_and_motor_limits_preserve_physical_torque_mapping(monkeypatch):
    item = fixture(monkeypatch)
    item.direction[:] = [0., 1., 0.]
    item.jac_h[:] = 0.; item.jac_h[1, 0] = -2.
    item.env._engine.human_limits[0] = 5.
    action, report = clearance.prepare_clearance_controller(item.env).sample()
    assert action[0] == pytest.approx(-.16)
    assert report['reported_pairs'][0]['distance_gradient_model_units'][0] == -2.


def test_relative_model_distances_and_world_jacobian_scale_are_explicit(monkeypatch):
    item = fixture(monkeypatch, scale=2.)
    action, report = clearance.prepare_clearance_controller(item.env).sample()
    assert action[0] == pytest.approx(.08)
    row = report['reported_pairs'][0]
    assert row['distance_model_units'] == .02
    assert row['distance_gradient_model_units'][0] == 1.
    assert row['force_body_weights'] == pytest.approx(.04)


def test_compound_duplicate_parts_do_not_multiply_gain_and_nearest_part_wins(monkeypatch):
    item = fixture(monkeypatch, parts=(1, 2, 3))
    item.gaps.update({1: .03, 2: .02, 3: .02})
    action, report = clearance.prepare_clearance_controller(item.env).sample()
    assert action[0] == pytest.approx(.04)
    assert report['active_pairs'] == 1 and report['distance_queries'] == 3 and report['jacobian_queries'] == 2
    assert report['reported_pairs'][0]['part_geom_id'] == 2


@pytest.mark.parametrize('kind', ['zero', 'disabled', 'far', 'saturated'])
def test_unchanged_action_is_bit_exact_and_detached(monkeypatch, kind):
    item = fixture(monkeypatch, gravity=(0., 0., 0.) if kind == 'zero' else (0., 0., -10.))
    item.action[3] = -0.
    if kind == 'far': item.gaps[1] = 1.
    if kind == 'saturated': item.action[0] = 1.
    config = clearance.ClearanceConfig(maximum_joint_residual=0. if kind == 'disabled' else .25)
    result, report = clearance.prepare_clearance_controller(item.env, config=config).sample()
    assert result.tobytes() == item.action.tobytes()
    assert not np.shares_memory(result, item.action)
    assert report['maximum_actual_joint_residual'] == 0.
    if kind in {'zero', 'disabled'}:
        assert item.calls['distance'] == item.calls['jacobian'] == 0


@pytest.mark.parametrize('gap', [0., 1e-11, float('nan')])
def test_unavailable_or_degenerate_normals_abstain_visibly(monkeypatch, gap):
    item = fixture(monkeypatch)
    item.gaps[1] = gap
    result, report = clearance.prepare_clearance_controller(item.env).sample()
    assert result.tobytes() == item.action.tobytes()
    assert report['status'] == 'unavailable_pairs' and report['unavailable_pairs'] == 1
    assert 'unavailable_reason' in report['reported_pairs'][0]


def test_unactuated_pair_never_uses_root_for_avoidance(monkeypatch):
    item = fixture(monkeypatch)
    item.jac_h[:, :19] = 0.
    action, report = clearance.prepare_clearance_controller(item.env).sample()
    assert action.tobytes() == item.action.tobytes()
    assert report['unactuated_pairs'] == 1


@pytest.mark.parametrize('root', [False, True])
def test_optional_base_policy_called_once_and_joint_residual_bounded(monkeypatch, root):
    item = fixture(monkeypatch, root=root)
    item.gaps[1] = -.1
    item.action[0] = .9
    base_calls = []
    def supplied():
        base_calls.append(True)
        return item.action.copy()
    controller = clearance.prepare_clearance_controller(item.env, base_action=supplied)
    action, report = controller.sample()
    assert len(base_calls) == 1 and item.calls['base'] == 0
    assert action[0] == 1. and len(action) == (22 if root else 19)
    assert np.all(abs(action[:19] - item.action[:19]) <= .25)
    assert report['maximum_actual_joint_residual'] == pytest.approx(.1)


@pytest.mark.parametrize('change', ['contact', 'static', 'camera', 'feet', 'channels', 'too_many_pairs'])
def test_unsupported_contracts_reject_before_geometry_queries(monkeypatch, change):
    item = fixture(monkeypatch)
    if change == 'contact': item.env.task['contact_requirements'] = [{'object_id': 'equipment'}]
    elif change == 'static': item.env.scene['objects'][0]['role'] = 'static'
    elif change == 'camera': item.env.scene['profile'] = 'shared-world-v6'
    elif change == 'feet': item.env.scene['profile'] = 'shared-world-v5'
    elif change == 'channels': item.env.action_names = tuple(reversed(item.env.action_names))
    else: item.env._engine._object_geometry_ids['equipment'] = tuple(range(1, 514))
    with pytest.raises(ValueError): clearance.prepare_clearance_controller(item.env)
    assert item.calls == {'base': 0, 'distance': 0, 'jacobian': 0}


def test_config_and_description_cannot_mutate_prepared_controller(monkeypatch):
    item = fixture(monkeypatch)
    config = clearance.ClearanceConfig()
    with pytest.raises(FrozenInstanceError): config.margin_model_units = .1
    controller = clearance.prepare_clearance_controller(item.env, config=config)
    description = controller.describe()
    description['settings']['margin_model_units'] = .1
    assert controller()[0] == pytest.approx(.04)
    for values in ({'margin_model_units': True}, {'lookahead_seconds': float('nan')},
                   {'maximum_joint_residual': .6}, {'query_radius_model_units': .025}):
        with pytest.raises(ValueError): replace(config, **values)


def test_bad_action_velocity_and_replaced_engine_fail_explicitly(monkeypatch):
    item = fixture(monkeypatch)
    controller = clearance.prepare_clearance_controller(item.env)
    item.action[0] = float('nan')
    with pytest.raises(ValueError, match='base action'): controller()
    item.action[0] = 0.; item.data.qvel[0] = float('nan')
    with pytest.raises(ValueError, match='velocity'): controller()
    item.env._engine.data = SimpleNamespace()
    with pytest.raises(RuntimeError, match='replacing'): controller()


@pytest.mark.parametrize('dtype', [np.float32, np.float64, np.int64])
def test_active_root_values_are_exact_and_floating_dtype_is_preserved(monkeypatch, dtype):
    item = fixture(monkeypatch)
    base = item.action.astype(dtype)
    action = clearance.prepare_clearance_controller(item.env, base_action=lambda: base.copy())()
    assert action[0] > 0
    np.testing.assert_array_equal(action[19:], base[19:])
    if np.issubdtype(dtype, np.floating):
        assert action.dtype == base.dtype and action[19:].tobytes() == base[19:].tobytes()
    else:
        assert action.dtype == np.float64


def test_float32_rounding_cannot_expand_residual_bound(monkeypatch):
    item = fixture(monkeypatch)
    base = item.action.astype(np.float32); base[0] = .1
    item.gaps[1] = -.1
    config = clearance.ClearanceConfig(maximum_joint_residual=.2)
    action = clearance.prepare_clearance_controller(item.env, base_action=lambda: base.copy(), config=config)()
    assert abs(float(action[0]) - float(base[0])) <= .2
    assert action[19:].tobytes() == base[19:].tobytes()


def test_finite_but_overflowing_velocity_is_not_returned_as_nan_action(monkeypatch):
    item = fixture(monkeypatch)
    item.jac_h[0, 0] = 1e200; item.data.qvel[0] = 1e200
    with np.errstate(over='ignore', invalid='ignore'):
        with pytest.raises(ValueError, match='relative speed'):
            clearance.prepare_clearance_controller(item.env)()
