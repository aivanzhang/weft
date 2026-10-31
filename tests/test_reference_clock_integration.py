"""Native-clock roundoff must not delay reference velocity or rewrite scored positions."""

from types import SimpleNamespace

import numpy as np
import pytest

from weft.shared_world import SharedWorldEnv
from weft.whole_body import WholeBodyEnv


def sampler(cls, *, start=0., substeps=16):
    env = object.__new__(cls)
    env.source_start = start
    env.dt = 1 / 30
    env.substeps = substeps
    env.model = SimpleNamespace(opt=SimpleNamespace(timestep=env.dt / substeps))
    env.data = SimpleNamespace(time=0.)
    env._times = np.array([start, start + 18 / 30, start + 20 / 30, start + 1.])
    env._reference = np.arange(1, 23)[:, None] * np.array([0., .01, -.02, .03])[None, :]
    env._slopes = np.diff(env._reference, axis=1) / np.diff(env._times)
    return env


def accumulated_time(steps, timestep):
    result = 0.
    for _ in range(steps):
        result += timestep
    return result


def original_position(env, elapsed):
    timestamp = float(np.clip(env.source_start + elapsed, env._times[0], env._times[-1]))
    index = min(len(env._times) - 2, max(0, int(np.searchsorted(env._times, timestamp, side="right")) - 1))
    return env._reference[:, index] + env._slopes[:, index] * (timestamp - env._times[index])


@pytest.mark.parametrize("cls", [WholeBodyEnv, SharedWorldEnv])
@pytest.mark.parametrize("substeps", [8, 16, 64])
@pytest.mark.parametrize("start", [0., .25])
def test_complete_control_knot_uses_right_derivative_without_changing_clock(cls, substeps, start):
    env = sampler(cls, start=start, substeps=substeps)
    for step, segment in ((18, 1), (20, 2)):
        env.data.time = accumulated_time(step * substeps, env.model.opt.timestep)
        native_time = env.data.time
        q, v = env.target_state()
        np.testing.assert_array_equal(v, env._slopes[:, segment])
        np.testing.assert_array_equal(q.view(np.uint64), original_position(env, native_time).view(np.uint64))
        assert env.data.time == native_time


@pytest.mark.parametrize("cls", [WholeBodyEnv, SharedWorldEnv])
def test_dense_positions_and_true_partial_interval_remain_exact(cls):
    env = sampler(cls, start=.25)
    queries = list(np.linspace(-.1, 1.1, 301)) + [.6 - 1e-7, .6 + 1e-7, 20 / 30 - 1e-7]
    for elapsed in queries:
        q, _ = env._target_at(elapsed)
        np.testing.assert_array_equal(q.view(np.uint64), original_position(env, elapsed).view(np.uint64))
    env.data.time = .6 - env.model.opt.timestep
    before = env.data.time
    _, v = env.target_state()
    np.testing.assert_array_equal(v, env._slopes[:, 0])
    assert env.data.time == before


@pytest.mark.parametrize("cls", [WholeBodyEnv, SharedWorldEnv])
def test_final_clock_roundoff_does_not_leave_a_terminal_velocity(cls):
    env = sampler(cls)
    env.data.time = accumulated_time(30 * env.substeps, env.model.opt.timestep)
    native_time = env.data.time
    q, v = env.target_state()
    np.testing.assert_array_equal(v, np.zeros(22))
    np.testing.assert_array_equal(q.view(np.uint64), original_position(env, native_time).view(np.uint64))
    assert env.data.time == native_time
