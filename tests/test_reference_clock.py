"""Analytic native-clock tests; no model construction or physics execution."""

from fractions import Fraction
import math
from types import SimpleNamespace

import numpy as np
import pytest

from weft import human
from weft.human import HumanEnv, reference_velocity_index


def accumulated_time(count, timestep):
    result = 0.
    for _ in range(count):
        result += timestep
    return result


def lookup(times, elapsed, timestep=1 / 480, source_start=0., timestamp=None):
    return reference_velocity_index(times, source_start + elapsed if timestamp is None else timestamp,
                                    elapsed=elapsed, timestep=timestep, source_start=source_start)


@pytest.mark.parametrize("substeps", [8, 9, 10, 12, 16, 24, 32, 48, 63, 64])
@pytest.mark.parametrize("origin", [Fraction(0), Fraction(1, 4), Fraction(2, 15), Fraction(3500)])
def test_repeated_positive_additions_match_independent_rational_knot_oracle(substeps, origin):
    boundaries = [0, 18, 20, 30, 1800]
    times = np.array([float(origin + Fraction(step, 30)) for step in boundaries])
    frozen = times.tobytes()
    elapsed, boundary = 0., 1
    timestep = (1 / 30) / substeps
    for step in range(1, 1800 * substeps + 1):
        elapsed += timestep
        if step == boundaries[boundary] * substeps:
            expected = boundary if boundary < len(boundaries) - 1 else None
            before = elapsed
            assert lookup(times, elapsed, timestep, float(origin)) == expected
            assert elapsed == before
            boundary += 1
    assert times.tobytes() == frozen


@pytest.mark.parametrize("direction", [-math.inf, math.inf])
def test_both_sides_of_a_native_knot_use_right_derivative(direction):
    elapsed = math.nextafter(.6, direction)
    assert lookup([0., .6, 2 / 3, 1.], elapsed) == 1


def test_source_start_addition_rounding_and_partial_native_boundary():
    origin, offset = 2 / 15, 1 / 6
    source_start = origin + offset
    elapsed = accumulated_time(16, 1 / 480)
    assert lookup([2 / 15, 1 / 3, .6], elapsed, source_start=source_start) == 1
    # An incomplete control step is still a legitimate native boundary, not a rounded control step.
    elapsed = accumulated_time(3, 1 / 480)
    assert lookup([0., float(Fraction(3, 480)), 1.], elapsed) == 1
    assert lookup([0., .6, 1.], .6 - 1 / 480) == 0
    assert lookup([0., .6, 1.], .6 - .5 / 480) == 0


def test_noncommensurate_knot_inside_accumulated_bound_is_not_promoted():
    timestep, count = 1 / 480, 115200
    elapsed = accumulated_time(count, timestep)
    native_knot = float(Fraction(count, 480))
    # This is within gamma_n*n*h, but much farther than the arithmetic-only lattice error.
    knot = native_knot + 1e-10
    n = math.ceil(math.nextafter(elapsed / timestep, math.inf))
    gamma = n * 2.**-53 / (1 - n * 2.**-53)
    assert 0 < knot - elapsed < min(1e-9, gamma * n * timestep)
    assert lookup([0., knot, knot + 1.], elapsed, timestep) == 0


def test_cap_never_advances_a_clock_more_than_one_nanosecond():
    timestep, count = 1 / 480, 1000000
    knot = count * timestep
    assert lookup([0., knot, knot + 1.], knot - 2e-9, timestep) == 0


def test_adjacent_knot_ambiguity_keeps_ordinary_lookup():
    knot = .6
    following = math.nextafter(knot, math.inf)
    elapsed = accumulated_time(288, 1 / 480)
    assert elapsed < knot
    assert lookup([0., knot, following, 1.], elapsed) == 0
    assert lookup([0., knot, following, 1.], knot) == 1


def test_large_reference_gap_does_not_expand_clock_tolerance():
    elapsed = accumulated_time(288, 1 / 480)
    assert lookup([0., .6, 3600.], elapsed) == 1
    assert lookup([0., .6, 3600.], .6 - 1e-7) == 0


def test_pre_source_start_clipped_queries_and_terminal_semantics():
    times = [.25, .85, 1.25]
    assert lookup(times, -.01, source_start=.25) == 0
    assert lookup(times, -.01, source_start=.25, timestamp=.25) == 0
    assert lookup(times, 0., source_start=math.nextafter(.85, -math.inf)) == 0
    assert lookup(times, .6 - .001, source_start=.25, timestamp=.85 - 1e-15) == 0
    assert lookup(times, 1., source_start=.25) is None
    assert lookup(times, 2., source_start=.25, timestamp=1.25) is None
    elapsed = accumulated_time(480, 1 / 480)
    assert elapsed < 1.
    assert lookup(times, elapsed, source_start=.25) is None


@pytest.mark.parametrize("field,value", [("elapsed", math.nan), ("elapsed", math.inf),
                                         ("timestep", 0.), ("timestep", -1.), ("timestep", math.inf),
                                         ("source_start", math.nan), ("timestamp", math.nan)])
def test_invalid_clock_inputs_reject(field, value):
    arguments = dict(timestamp=.6, elapsed=.6, timestep=1 / 480, source_start=0.)
    arguments[field] = value
    with pytest.raises(ValueError, match="finite clocks"):
        reference_velocity_index([0., .6, 1.], **arguments)


def test_lookup_does_not_scan_the_reference():
    class CountedTimes:
        reads = 0

        def __len__(self):
            return 1800

        def __getitem__(self, index):
            self.reads += 1
            return (index if index >= 0 else 1800 + index) / 30

    times = CountedTimes()
    assert lookup(times, accumulated_time(288, 1 / 480)) == 18
    assert times.reads <= 20


def servo_fixture(elapsed, controller="bias_compensated_pd"):
    env = object.__new__(HumanEnv)
    env.joint_names, env.selected_joint = ("left_elbow",), "left_elbow"
    env.action_space = SimpleNamespace(contains=lambda action: action in (4, 5), n=6)
    env.human_spec = {"controller": controller}
    env.mobility, env.targets, env.steps = None, {"left_elbow": 0.}, 0
    env._q = env._actuator_ids = {"left_elbow": 0}
    env._q_array = env._v_array = env._a_array = np.array([0])
    env._hold, env._motor_limits = np.array([0.]), np.array([1000.])
    env._reference_times = np.array([0., .6, 2 / 3, 1.])
    env._reference_matrix = np.array([[0., .01, -.02, .03]])
    env._reference_angles = {"left_elbow": env._reference_matrix[0]}
    env._reference_slopes = np.diff(env._reference_matrix) / np.diff(env._reference_times)
    env.model = SimpleNamespace(opt=SimpleNamespace(timestep=1 / 240))
    env.data = SimpleNamespace(time=elapsed, qpos=np.array([0.]), qvel=np.array([0.]),
                               qfrc_bias=np.array([0.]), ctrl=np.array([0.]))
    env.observation = lambda: np.array([0., 0.])
    return env


@pytest.mark.parametrize("controls,velocity_index", [(18, 1), (20, 2), (30, None)])
def test_human_servo_changes_only_derivative_at_native_knot(monkeypatch, controls, velocity_index):
    env = servo_fixture(accumulated_time(controls * 8, 1 / 240))
    timestamp = min(env._reference_times[0] + env.data.time, env._reference_times[-1])
    index = min(len(env._reference_times) - 2,
                max(0, int(np.searchsorted(env._reference_times, timestamp, side="right")) - 1))
    target = env._reference_matrix[:, index] + env._reference_slopes[:, index] * (timestamp - env._reference_times[index])
    velocity = env._reference_slopes[:, velocity_index] if velocity_index is not None else 0
    expected = 40 * target + 5 * velocity
    commands = []

    def native_stub(model, data):
        commands.append(data.ctrl.copy())
        data.time += model.opt.timestep

    monkeypatch.setattr(human.mujoco, "mj_step", native_stub)
    monkeypatch.setattr(human.mujoco, "mj_forward", lambda *_: None)
    env._advance(5, task=False)
    np.testing.assert_array_equal(commands[0].view(np.uint64), expected.view(np.uint64))
    assert len(commands) == 8


@pytest.mark.parametrize("controller,action", [("pd", 5), ("bias_compensated_pd", 4)])
def test_other_human_control_paths_never_use_derivative_helper(monkeypatch, controller, action):
    env = servo_fixture(.6, controller)
    monkeypatch.setattr(human.mujoco, "mj_step", lambda *_: None)
    monkeypatch.setattr(human.mujoco, "mj_forward", lambda *_: None)

    def unexpected(*args, **kwargs):
        raise AssertionError("Unrelated control path used the derivative correction")

    monkeypatch.setattr(human, "reference_velocity_index", unexpected)
    env._advance(action, task=False)
