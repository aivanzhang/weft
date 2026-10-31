"""Analytical camera actions and custody; native/model/network entrypoints are blocked."""

from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

from test_camera_shared_task import analytical_only, scene_fixture, _deny
from weft import camera_shared_policy as actor
from weft import camera_shared_task as task_api
from weft.io import canonical_hash


def environment(root=False, phase=.5):
    scene = scene_fixture()
    scene["bounds"] = [6., 6., 6.]
    scene["substeps"] = scene["coordinate_frame"]["clock_binding"]["physics_substeps"] = 9
    task = task_api.build_camera_shared_task(scene, root_assistance=root)
    count = 22 if root else 19
    env = SimpleNamespace(scene=scene, task=task, action_count=count,
        action_names=tuple(scene["action_contract"]["names"][:count]), scene_sha256=task["scene_sha256"],
        task_sha256=canonical_hash(task), evaluator_sha256=task["evaluator_sha256"], calls=0)
    env.features = np.tile([.2, .05, .02], (count, 1))
    env.observation = lambda: np.r_[np.zeros(155), phase]
    env.observation_contract = lambda: {"profile": task_api.OBSERVATION_PROFILE}

    def features():
        env.calls += 1
        return env.features.copy()

    env.feature_matrix = features
    return env


@pytest.mark.parametrize("root", [False, True])
def test_exact_authored_anchor_and_no_hot_path_hash_or_preparation(root, monkeypatch):
    env = environment(root)
    weights = np.zeros(actor.SHAPE)
    policy = actor.make_policy(env, weights, 1729, .5, actor.runtime_hashes(env.scene))
    action = actor.prepare_camera_shared_policy(policy, env)
    policy["feedback_coefficients"][0][0][0] = .25
    monkeypatch.setattr(actor, "runtime_hashes", _deny)
    monkeypatch.setattr(actor, "prepare_camera_shared_task", _deny)
    monkeypatch.setattr(actor, "canonical_hash", _deny)
    expected = np.clip(env.features.sum(axis=1), -1., 1.)
    assert action().tobytes() == expected.tobytes()
    assert env.calls == 1 and env.scene["learned_policy_available"] is False
    assert env.task["scope"]["learning"] == "unqualified"


def test_both_phase_and_state_change_actions_with_exact_unchanged_roots_and_bias():
    env = environment(True)
    groups = actor.joint_groups(env.action_names)
    weights = np.zeros(actor.SHAPE)
    weights[0, :, 0], weights[1, :, 1] = .2, -.1
    first, features, authored, _, phase = actor.action_sample(weights, groups, env)
    assert phase == .5 and env.calls == 1 and np.any(first[:19] != authored[:19])
    env.observation = lambda: np.r_[np.zeros(155), 1.]
    second = actor.action_sample(weights, groups, env)[0]
    assert not np.array_equal(first, second)
    env.features[:, 1] += .2
    third, _, baseline, residual, _ = actor.action_sample(weights, groups, env)
    assert not np.array_equal(second, third)
    assert third[19:].tobytes() == baseline[19:].tobytes()
    assert np.all(abs(residual) <= .25) and np.all(abs(third) <= 1.)
    assert np.all(abs(third[:19] - baseline[:19]) <= .25 + 1e-16)
    assert np.array_equal(features[:, 2], np.full(22, .02))


@pytest.mark.parametrize("field,value", [
    ("profile", "shared-exercise-policy-v4"), ("root_residual_channels", ["root_x"]),
    ("object_force_channels", ["object_x"]), ("gates_sha256", "0" * 64),
    ("body_sha256", "0" * 64), ("basis_profile", "invented"),
    ("observation_profile", "shared-exercise-observation-v1"), ("privileged_simulator_features", False),
])
def test_policy_cannot_change_authority_source_gates_or_privilege(field, value):
    env = environment(True)
    policy = actor.make_policy(env, np.zeros(actor.SHAPE), 1729, .5, actor.runtime_hashes(env.scene))
    policy[field] = value
    with pytest.raises(ValueError):
        actor.validate_camera_shared_policy(policy, env.scene, env.task)


def test_coefficients_phase_and_feature_bounds_reject_before_actions():
    env = environment()
    for value in (None, [[0.] * 19] * 2, np.zeros(actor.SHAPE).tolist() + [[]]):
        with pytest.raises(ValueError):
            actor.coefficients(value)
    for value in (True, 2**1000, float("nan"), .251):
        weights = np.zeros(actor.SHAPE).tolist()
        weights[0][0][0] = value
        with pytest.raises(ValueError):
            actor.coefficients(weights)
    groups = actor.joint_groups(env.action_names)
    for phase in (-.01, 1.01, float("nan")):
        env.observation = lambda: np.r_[np.zeros(155), phase]
        with pytest.raises(ValueError):
            actor.action_sample(np.zeros(actor.SHAPE), groups, env)
    assert env.calls == 0
    env.observation = lambda: np.zeros(156)
    env.features[0][0] = np.nan
    with pytest.raises(ValueError):
        actor.action_sample(np.zeros(actor.SHAPE), groups, env)
    env.features = np.full((19, 3), -(2**63), dtype=np.int64)
    with pytest.raises(ValueError, match="bounded existing feedback"):
        actor.action_sample(np.zeros(actor.SHAPE), groups, env)


def test_nonzero_coefficients_do_not_imply_realized_effect_or_improvement():
    env = environment(phase=0.)
    policy = actor.make_policy(env, np.full(actor.SHAPE, .2), 1729, .1, actor.runtime_hashes(env.scene))
    assert actor.selection_kind(policy) == "searched_phase_feedback_residuals"
    actual, _, authored, correction, _ = actor.action_sample(actor.coefficients(policy["feedback_coefficients"]),
                                                           actor.joint_groups(env.action_names), env)
    assert np.array_equal(actual, authored) and not np.any(correction)
    assert "nonzero coefficients alone" in policy["scope"]
    before = deepcopy(env.task)
    actor.validate_camera_shared_policy(policy, env.scene, env.task)
    assert env.task == before
