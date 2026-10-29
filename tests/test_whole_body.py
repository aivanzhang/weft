"""Analytic direct-actuation checks, not evidence of human balance or transfer."""

import mujoco
import numpy as np
import pytest

from weft.human import build_human
from weft.mobile_human import ROOT_AXES
from weft.whole_body import WholeBodyEnv
from test_mobile_integration import evidence


@pytest.fixture
def spec(tmp_path):
    return build_human(evidence(), tmp_path, body_model='mobile')


@pytest.fixture
def env(spec):
    value = WholeBodyEnv(spec, horizon_steps=6)
    value.reset(seed=7, options={'jitter': False})
    yield value
    value.close()


def test_contract_covers_every_motor_without_mixing_root_into_muscles(env):
    assert len(env.action_names) == env.model.nu == 22
    assert env.action_names[-3:] == ROOT_AXES
    assert len(set(env._actuators)) == 22
    assert env.feature_matrix().shape == (22, 3)
    assert env.observation_space.contains(env.observation())
    assert env.observation().shape == (155,)
    assert env.scene['profile'] == 'exercise-imitation-v1'
    assert env.scene['action_contract']['hidden_assistance'] is False
    state = env.snapshot()
    assert state['human']['control_mode'] == 'direct_motor_actions'
    assert len(state['human']['angles']) == 19
    assert len(state['human']['muscles']) == 38
    assert not state['human']['mobility']['support_enabled']
    assert 'target_angle' not in state['human']


def test_zero_action_never_invokes_servo_or_bias_and_clears_external_forces(env, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('hidden assistance or expensive snapshot')
    monkeypatch.setattr(env.base, '_advance', forbidden)
    monkeypatch.setattr(env.base.mobility, 'prepare', forbidden)
    monkeypatch.setattr(env.base.mobility, 'apply', forbidden)
    monkeypatch.setattr(env.base, 'snapshot', forbidden)
    assert env.data.qfrc_bias[env._v[-1]] > 0
    env.data.qfrc_applied[:] = 3
    env.data.xfrc_applied[:] = 3
    env.step(np.zeros(22))
    np.testing.assert_array_equal(env.data.ctrl, 0)
    np.testing.assert_array_equal(env.data.qfrc_applied, 0)
    np.testing.assert_array_equal(env.data.xfrc_applied, 0)


def test_named_commands_are_held_for_all_eight_physics_ticks(env, monkeypatch):
    calls = []
    original = mujoco.mj_step
    def tracked(model, data):
        calls.append(data.ctrl.copy())
        original(model, data)
    monkeypatch.setattr(mujoco, 'mj_step', tracked)
    action = np.linspace(-.01, .01, 22)
    _, _, terminated, _, info = env.step(action)
    assert not terminated
    assert len(calls) == 8
    for values in calls:
        np.testing.assert_allclose(values[env._actuators], action * env.limits)
    np.testing.assert_allclose(info['applied_motor_forces'], action * env.limits)
    assert env.data.time == pytest.approx(env.dt)


def test_features_expose_bias_explicitly_and_target_uses_current_source_time(spec):
    env = WholeBodyEnv(spec, start_time_s=.2, horizon_steps=3)
    try:
        env.reset(options={'jitter': False})
        target, velocity = env.target_state()
        np.testing.assert_allclose(target, env.data.qpos[env._q])
        np.testing.assert_allclose(env.feature_matrix()[:, 2], env.data.qfrc_bias[env._v] / env.limits)
        np.testing.assert_allclose(env.observation()[44:66], target, atol=1e-6)
        np.testing.assert_allclose(env.observation()[66:88], velocity, atol=1e-6)
        _, _, _, _, info = env.step(np.zeros(22))
        assert info['source_time_s'] == pytest.approx(env.source_start + env.dt)
        np.testing.assert_allclose(env.target_state()[0], env._target_at(env.dt)[0])
    finally:
        env.close()


@pytest.mark.parametrize('action', [np.zeros(21), np.ones(22) * 1.01, np.full(22, np.nan),
                                    np.full(22, np.inf), np.zeros(22, dtype=bool),
                                    [False] + [0.] * 21, ['0'] * 22, 0])
def test_invalid_actions_rejected_before_any_state_mutation(env, action):
    before = (env.data.qpos.copy(), env.data.ctrl.copy(), env.data.time, env.steps)
    with pytest.raises(ValueError):
        env.step(action)
    np.testing.assert_array_equal(env.data.qpos, before[0])
    np.testing.assert_array_equal(env.data.ctrl, before[1])
    assert (env.data.time, env.steps) == before[2:]


def test_seeded_resets_reproduce_bounded_jitter_and_dynamics(env):
    initial, _ = env.reset(seed=42)
    after = env.step(np.zeros(22))[0]
    repeat, _ = env.reset(seed=42)
    np.testing.assert_array_equal(initial, repeat)
    np.testing.assert_array_equal(after, env.step(np.zeros(22))[0])
    different, _ = env.reset(seed=43)
    assert not np.array_equal(initial, different)
    target, _ = env.target_state()
    assert np.max(np.abs(env.data.qpos[env._q[:19]] - target[:19])) <= .01
    assert 0 <= env.data.qpos[env._q[-1]] - target[-1] <= .002


@pytest.mark.parametrize('kwargs', [{'horizon_steps': 0}, {'horizon_steps': 181}, {'horizon_steps': True},
                                    {'start_time_s': -.1}, {'start_time_s': float('nan')},
                                    {'start_time_s': .8, 'horizon_steps': 6}])
def test_invalid_or_uncovered_windows_fail(spec, kwargs):
    with pytest.raises(ValueError):
        WholeBodyEnv(spec, **kwargs)


@pytest.mark.parametrize('dt', [1 / 120, 1 / 20])
def test_other_valid_human_timesteps_cannot_change_the_30hz_contract(spec, dt, monkeypatch):
    spec['dt'] = dt
    def forbidden(*args, **kwargs):
        raise AssertionError('Reject before constructing a physics model')
    monkeypatch.setattr('weft.whole_body.HumanEnv', forbidden)
    with pytest.raises(ValueError, match='30-Hz'):
        WholeBodyEnv(spec, horizon_steps=6)


def test_metrics_are_full_body_rmse_independent_of_reward(env):
    angular, root, rewards = [], [], []
    for index in range(env.horizon_steps):
        _, reward, terminated, truncated, _ = env.step(np.zeros(22))
        assert not terminated
        assert truncated == (index == env.horizon_steps - 1)
        error = env.data.qpos[env._q] - env.target_state()[0]
        angular.append(np.mean(error[:19] ** 2))
        root.append(np.mean(error[19:] ** 2))
        rewards.append(reward)
    metrics = env.metrics()
    assert metrics['angular_rmse_rad'] == pytest.approx(np.sqrt(np.mean(angular)))
    assert metrics['root_rmse_model_units'] == pytest.approx(np.sqrt(np.mean(root)))
    assert metrics['return'] == pytest.approx(sum(rewards) / env.horizon_steps)
    assert metrics['metric_samples'] == env.horizon_steps
    assert metrics['completed_horizon']
    assert metrics['success'] == (metrics['angular_rmse_rad'] <= .35 and
                                  metrics['root_rmse_model_units'] <= .15 and
                                  metrics['max_floor_penetration_model_units'] <= .025)
    with pytest.raises(RuntimeError):
        env.step(np.zeros(22))


def test_floor_gate_does_not_short_circuit_physics_or_tracking_reward(env, monkeypatch):
    monkeypatch.setattr(env, '_floor_penetration', lambda: .03)
    for _ in range(env.horizon_steps):
        _, reward, terminated, _, _ = env.step(np.zeros(22))
        assert not terminated and reward >= 0
    assert env.metrics()['completed_horizon']
    assert not env.metrics()['success']
    assert 'floor_penetration' in env.metrics()['failure']


def test_finite_instability_cannot_pass_or_earn_unfinished_horizon(env, monkeypatch):
    monkeypatch.setattr(env, '_instability', lambda: 'velocity_envelope_exceeded')
    _, reward, terminated, truncated, _ = env.step(np.zeros(22))
    assert terminated and not truncated
    assert not env.metrics()['completed_horizon'] and not env.metrics()['success']
    assert 0 <= reward <= 1
    assert env.metrics()['return'] == pytest.approx(reward / env.horizon_steps)


def test_nonfinite_state_is_not_advanced_or_reported_as_valid_rmse(env, monkeypatch):
    env.step(np.zeros(22))
    samples = env.metrics()['metric_samples']
    env.data.qpos[0] = np.nan
    def forbidden(*args):
        raise AssertionError('MuJoCo must not silently reset nonfinite state')
    monkeypatch.setattr(mujoco, 'mj_step', forbidden)
    with pytest.raises(RuntimeError, match='Nonfinite'):
        env.step(np.zeros(22))
    metrics = env.metrics()
    assert metrics['failure'] == 'nonfinite_dynamics'
    assert metrics['metric_samples'] == samples
    assert metrics['angular_rmse_rad'] is None and metrics['root_rmse_model_units'] is None
    assert not metrics['tracking_metrics_valid'] and not metrics['success']


def test_root_depth_is_physical_and_directly_controlled_not_reference_inferred(env):
    assert all(row['translation'][1] == 0 for row in env.base.human_spec['root_reference'])
    action = np.zeros(22)
    action[-2] = .1
    env.step(action)
    assert env.data.qpos[env._q[-2]] > 0
    state = env.snapshot()['human']
    assert not state['mobility']['support_enabled']
    assert state['motor_forces'][-2] > 0
