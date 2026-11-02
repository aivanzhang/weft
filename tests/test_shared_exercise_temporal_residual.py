"""Analytical temporal-control invariants; no simulation or learning claims."""

from copy import deepcopy
from dataclasses import FrozenInstanceError
import hashlib
import json

import numpy as np
import pytest

from test_shared_exercise_equipment_feedback import fixture
from weft import shared_exercise_temporal_residual as actor


def weights(joint=14, values=(.2, -.1, .05)):
    result = np.zeros((19, 3))
    result[joint] = values
    return result


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_zero_schedule_preserves_every_base_byte_and_detaches_inputs(dtype):
    env, observe = fixture()
    base = np.linspace(-1., 1., 22).astype(dtype)
    base[[1, 19, 21]] = -0.
    observed = observe(elapsed=.375).astype(dtype)
    before = base.tobytes(), observed.tobytes()
    base.flags.writeable = observed.flags.writeable = False
    sampled, report = actor.prepare_temporal_joint_residual(env, np.zeros((19, 3))).sample(base, observed)
    assert sampled.tobytes() == before[0] and report["changed_joint_channels"] == []
    assert report["root_bytes_unchanged"] is True
    sampled[:] = .7
    assert (base.tobytes(), observed.tobytes()) == before


def test_schedule_acts_before_motion_and_uses_task_elapsed_not_reference_phase():
    env, observe = fixture()
    prepared = actor.prepare_temporal_joint_residual(env, weights())
    stationary = np.zeros(155)
    stationary[154] = .75
    sampled, report = prepared.sample(np.zeros(22), observe(body=stationary, elapsed=.125))
    assert sampled[14] == pytest.approx(.1) and np.count_nonzero(sampled) == 1
    assert report["basis"] == pytest.approx([.5, 0., 0.])
    assert report["reference_phase"] == .75 and report["elapsed_fraction"] == .125
    stationary[154] = 0.
    same, _ = prepared.sample(np.zeros(22), observe(body=stationary, elapsed=.125))
    assert sampled.tobytes() == same.tobytes()
    late, _ = prepared.sample(np.zeros(22), observe(body=stationary, elapsed=.5))
    assert late[14] == pytest.approx(-.1)


def test_basis_is_nonnegative_convex_and_has_zero_endpoint_value_and_slope():
    samples = np.asarray([actor.basis(float(p)) for p in np.linspace(0., 1., 401)])
    assert np.all(samples >= 0) and np.all(np.sum(samples, axis=1) <= 1. + 1e-14)
    assert np.array_equal(samples[0], np.zeros(3)) and np.array_equal(samples[-1], np.zeros(3))
    epsilon = 1e-5
    for endpoint in (0., 1.):
        inner = endpoint + epsilon if endpoint == 0. else endpoint - epsilon
        assert np.max(abs(actor.basis(inner) - actor.basis(endpoint))) / epsilon < .001
    for knot in (.25, .5, .75):
        left = (actor.basis(knot) - actor.basis(knot - epsilon)) / epsilon
        right = (actor.basis(knot + epsilon) - actor.basis(knot)) / epsilon
        assert np.max(abs(left - right)) < .002


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_all_phase_and_legal_clipping_bounds_with_root_byte_identity(dtype):
    env, observe = fixture()
    coefficients = np.tile([.25, -.25, .25], (19, 1))
    coefficients[::2] *= -1
    prepared = actor.prepare_temporal_joint_residual(env, coefficients)
    base = np.linspace(-1., 1., 22).astype(dtype)
    base[19:] = [-0., 1., -1.]
    original = base.tobytes()
    for phase in np.linspace(0., .999, 65):
        sampled, report = prepared.sample(base, observe(elapsed=float(phase)).astype(dtype))
        assert sampled.dtype == base.dtype and sampled[19:].tobytes() == base[19:].tobytes()
        assert np.max(abs(sampled[:19].astype(float) - base[:19].astype(float))) <= .25
        assert np.max(abs(sampled)) <= 1.
        assert len(json.dumps(report).encode()) < actor.MAX_REPORT_BYTES
    assert base.tobytes() == original


def test_prepared_weights_are_detached_and_reports_identify_the_exact_contract():
    env, observe = fixture()
    supplied = weights()
    expected = {"contract": actor.feature_contract(), "coefficients": supplied.tolist()}
    expected_sha = hashlib.sha256(json.dumps(expected, sort_keys=True, allow_nan=False,
        separators=(",", ":")).encode()).hexdigest()
    prepared = actor.prepare_temporal_joint_residual(env, supplied)
    supplied[:] = -.25
    action, report = prepared.sample(np.zeros(22), observe(elapsed=.25))
    assert action[14] == pytest.approx(.2) and report["parameter_sha256"] == expected_sha
    with pytest.raises(ValueError):
        prepared._weights.flags.writeable = True
    with pytest.raises(FrozenInstanceError):
        prepared._observation_size = 0
    report["basis"][:] = [0., 0., 0.]
    _, again = prepared.sample(np.zeros(22), observe(elapsed=.25))
    assert again["basis"] == [1., 0., 0.]


def test_preparation_and_sampling_never_query_or_change_live_state():
    env, observe = fixture()
    before = deepcopy((env.scene, env.task))
    def denied(*args, **kwargs):
        raise AssertionError("Live state accessed")
    env.step = env.reset = env.snapshot = env.authored_action = env.feature_matrix = denied
    prepared = actor.prepare_temporal_joint_residual(env, weights())
    observed = observe(elapsed=.375)
    sampled, report = prepared.sample(np.zeros(22), observed)
    assert sampled[14] == pytest.approx(.05) and report["pairing"] == "caller_responsibility_not_authenticated"
    assert (env.scene, env.task) == before


@pytest.mark.parametrize("invalid", [np.zeros((18, 3)), np.zeros((19, 4)), np.zeros((19, 3), dtype=bool),
    np.full((19, 3), np.nan), np.full((19, 3), .250001), [[False] * 3] * 19, [["0"] * 3] * 19,
    np.zeros((19, 3), dtype=object)])
def test_bad_coefficients_are_rejected_before_environment_access(invalid):
    with pytest.raises(ValueError):
        actor.prepare_temporal_joint_residual(object(), invalid)


@pytest.mark.parametrize("phase", [True, -.001, 1.001, float("nan"), float("inf"), "0.5"])
def test_invalid_basis_phase_is_rejected(phase):
    with pytest.raises(ValueError):
        actor.basis(phase)


@pytest.mark.parametrize("elapsed", [-.01, 1., float("nan"), float("inf")])
def test_terminal_or_invalid_observation_does_not_return_an_action(elapsed):
    env, observe = fixture()
    observed = observe().astype(float)
    observed[-1] = elapsed
    with pytest.raises(ValueError):
        actor.prepare_temporal_joint_residual(env, weights()).sample(np.zeros(22), observed)


def test_unsupported_required_contacts_and_observation_drift_fail_admission():
    env, _ = fixture()
    env.task["contact_requirements"] = [{"object_id": "object-0"}]
    with pytest.raises(ValueError):
        actor.prepare_temporal_joint_residual(env, weights())
    env.task["contact_requirements"] = []
    env.observation_size += 1
    with pytest.raises(ValueError):
        actor.prepare_temporal_joint_residual(env, weights())


@pytest.mark.parametrize("bad", [np.zeros(19), np.full(22, 1.01), np.zeros(22, dtype=np.int64)])
def test_wrong_or_illegal_base_actions_are_rejected(bad):
    env, observe = fixture()
    with pytest.raises(ValueError):
        actor.prepare_temporal_joint_residual(env, weights()).sample(bad, observe())
