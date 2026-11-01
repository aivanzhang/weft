"""Analytical actor checks; scene admission and runtime identity providers are mocked."""

from copy import deepcopy
from types import SimpleNamespace
import socket
import subprocess

import mujoco
import numpy as np
import pytest

from weft import shared_exercise_feedback_residual as actor
from weft.io import canonical_hash
from weft.mobile_human import ROOT_AXES
from weft.spatial_pose import SPATIAL_JOINT_NAMES


PROFILES = (actor.POLICY_PROFILE, actor.SPLIT_POLICY_PROFILE)
HASHES = {"shared_exercise_task.py": "e" * 64, "shared_exercise_feedback_residual.py": "a" * 64}
VERSIONS = {"python": "analytical", "numpy": "analytical", "mujoco": "not executed"}
SPLIT_GROUPS = [0, 0, 0] + [1, 1, 1, 2, 3, 3, 3, 4] * 2


@pytest.fixture(autouse=True)
def no_native_network_or_process(monkeypatch):
    def forbidden(*_args, **_kwargs):
        pytest.fail("Actor contract checks cannot execute native, network or process work")

    for name in ("MjModel", "MjData", "mj_step", "mj_forward", "mj_resetData", "mju_quat2Mat"):
        monkeypatch.setattr(mujoco, name, forbidden)
    monkeypatch.setattr(socket, "socket", forbidden)
    monkeypatch.setattr(subprocess, "Popen", forbidden)


@pytest.fixture
def analytical(monkeypatch):
    def context(scene, task):
        assert scene["profile"] in ("shared-world-v2", "shared-world-v3")
        count = 22 if task["control"]["root_assistance"] == "explicit_3_axis" else 19
        return deepcopy(scene), deepcopy(task), count

    monkeypatch.setattr(actor, "_context", context)
    monkeypatch.setattr(actor, "runtime_hashes", lambda _scene: dict(HASHES))
    monkeypatch.setattr(actor, "_runtime_versions", lambda: dict(VERSIONS))

    def build(*, assistance=True, person=False, features=None):
        names = SPATIAL_JOINT_NAMES + (ROOT_AXES if assistance else ())
        scene = {"profile": "shared-world-v3" if person else "shared-world-v2",
                 "action_contract": {"names": list(names)}}
        task = {"control": {"root_assistance": "explicit_3_axis" if assistance else "disabled"},
                "equipment_ids": [] if person else ["object-a", "object-b"], "contact_requirements": []}
        matrix = np.full((len(names), 3), .1) if features is None else features
        env = SimpleNamespace(scene=scene, task=task, scene_sha256=canonical_hash(scene),
            task_sha256=canonical_hash(task), evaluator_sha256=HASHES["shared_exercise_task.py"],
            action_names=names, action_count=len(names), calls=0)

        def read():
            env.calls += 1
            return matrix.copy()

        env.feature_matrix = read
        env.observation_contract = lambda: {"profile": "shared-exercise-observation-v1"}
        return env

    return build


def policy_for(env, profile=actor.POLICY_PROFILE, weights=None):
    rank = {"profile": "shared-exercise-feasibility-rank-v1", "complete": True, "feasible": False,
            "max_normalized_excess": 1., "mean_normalized_excess": .1, "violated_terms": 1,
            "gate_terms": 2 * (5 + 4 * len(env.task["equipment_ids"])), "reward_score": .7}
    weights = np.zeros(actor.contract(profile)["coefficient_shape"]) if weights is None else weights
    return actor.make_policy(env, weights, 1729, .7, rank, dict(HASHES), profile=profile)


def test_contract_is_explicit_detached_and_v4_remains_default():
    assert actor.contract() == {"profile": actor.POLICY_PROFILE, "algorithm": actor.ALGORITHM,
        "actor_profile": actor.ACTOR_PROFILE, "groups": ("torso", "proximal", "distal"),
        "coefficient_shape": (3, 2), "scope": actor.SCOPE}
    split = actor.contract(actor.SPLIT_POLICY_PROFILE)
    assert split == {"profile": actor.SPLIT_POLICY_PROFILE, "algorithm": actor.SPLIT_ALGORITHM,
        "actor_profile": actor.SPLIT_ACTOR_PROFILE,
        "groups": ("torso", "arm_proximal", "arm_distal", "leg_proximal", "leg_distal"),
        "coefficient_shape": (5, 2), "scope": actor.SPLIT_SCOPE}
    split["groups"] = ("root",)
    assert actor.contract(actor.SPLIT_POLICY_PROFILE)["groups"] == actor.SPLIT_GROUP_NAMES


@pytest.mark.parametrize("profile", [None, True, "shared-exercise-policy-v3", "shared-exercise-policy-v6", [5, 2]])
def test_no_shape_or_alias_can_choose_a_profile(profile):
    with pytest.raises(ValueError, match="explicit supported"):
        actor.contract(profile)


@pytest.mark.parametrize("assistance", [False, True])
def test_split_mapping_is_named_exact_and_order_independent(assistance):
    names = SPATIAL_JOINT_NAMES + (ROOT_AXES if assistance else ())
    actual = actor.joint_groups(names, profile=actor.SPLIT_POLICY_PROFILE)
    assert actual.tolist() == SPLIT_GROUPS and not actual.flags.writeable
    reordered = tuple(reversed(SPATIAL_JOINT_NAMES)) + (ROOT_AXES if assistance else ())
    assert actor.joint_groups(reordered, profile=actor.SPLIT_POLICY_PROFILE).tolist() == SPLIT_GROUPS[::-1]
    assert actor.joint_groups(names).tolist() == [0, 0, 0] + [1, 1, 1, 2, 1, 1, 1, 2] * 2


@pytest.mark.parametrize("change", ["duplicate", "unknown", "root_joint", "root_tail", "force_channel"])
def test_split_rejects_ambiguous_or_extra_channel_identity(change):
    names = list(SPATIAL_JOINT_NAMES + ROOT_AXES)
    if change == "duplicate":
        names[1] = names[0]
    elif change == "unknown":
        names[3] = "left_shoulder_extra"
    elif change == "root_joint":
        names[3] = "root_x"
    elif change == "root_tail":
        names[-2:] = names[-2:][::-1]
    else:
        names[-1] = "object_force_z"
    with pytest.raises(ValueError, match="exact original"):
        actor.joint_groups(names, profile=actor.SPLIT_POLICY_PROFILE)


@pytest.mark.parametrize("profile", PROFILES)
def test_coefficients_enforce_selected_shape_bounds_and_detach(profile):
    count = actor.contract(profile)["coefficient_shape"][0]
    rows = [[.25, -.25] for _ in range(count)]
    result = actor.coefficients(rows, profile=profile)
    rows[0][0] = 0.
    assert result[0, 0] == .25 and not result.flags.writeable
    bad_values = (rows[:-1], rows + [[0., 0.]], [[0.]] * count, [[True, 0.]] * count,
                  [[.250001, 0.]] * count, [[float("nan"), 0.]] * count,
                  [[float("inf"), 0.]] * count, [[0j, 0.]] * count)
    for bad in bad_values:
        with pytest.raises(ValueError, match="coefficients"):
            actor.coefficients(bad, profile=profile)


@pytest.mark.parametrize("profile", PROFILES)
@pytest.mark.parametrize("assistance", [False, True])
def test_zero_weights_are_byte_exact_authored_and_read_features_once(analytical, profile, assistance):
    count = 22 if assistance else 19
    matrix = np.random.default_rng(1729).normal(0, .9, (count, 3))
    matrix[0] = -0.
    env = analytical(assistance=assistance, features=matrix)
    result, observed, authored, delta = actor.action_sample(
        actor.coefficients(np.zeros(actor.contract(profile)["coefficient_shape"]).tolist(), profile=profile),
        actor.joint_groups(env.action_names, profile=profile), env)
    expected = np.clip(matrix.sum(axis=1), -1., 1.)
    assert result.tobytes() == authored.tobytes() == expected.tobytes()
    assert env.calls == 1 and not np.any(delta) and observed.tobytes() == matrix.tobytes()
    result[:] = 0
    assert matrix.tobytes() == observed.tobytes()


@pytest.mark.parametrize("assistance", [False, True])
def test_tied_split_coefficients_reproduce_v4_numerics_exactly(analytical, assistance):
    count = 22 if assistance else 19
    matrix = np.random.default_rng(42).normal(0, 3., (count, 3))
    weights = np.asarray([[.11, -.07], [-.23, .19], [.05, -.13]])
    old = analytical(assistance=assistance, features=matrix)
    split = analytical(assistance=assistance, features=matrix)
    v4 = actor.action_sample(weights, actor.joint_groups(old.action_names), old)
    v5 = actor.action_sample(weights[[0, 1, 2, 1, 2]],
        actor.joint_groups(split.action_names, profile=actor.SPLIT_POLICY_PROFILE), split)
    assert all(a.tobytes() == b.tobytes() for a, b in zip(v4, v5))


def test_arm_and_leg_coefficients_change_only_their_named_joints(analytical):
    env = analytical(features=np.tile([.4, -.2, .1], (22, 1)))
    weights = np.zeros((5, 2)); weights[:, 0] = [0., .1, -.1, .2, -.2]
    action, _, authored, delta = actor.action_sample(weights,
        actor.joint_groups(env.action_names, profile=actor.SPLIT_POLICY_PROFILE), env)
    expected = np.asarray([0.] * 3 + [.04, .04, .04, -.04, .08, .08, .08, -.08] * 2)
    np.testing.assert_allclose(delta, expected, rtol=0., atol=1e-16)
    np.testing.assert_allclose(action[:19] - authored[:19], expected, rtol=0., atol=1e-16)
    assert action[19:].tobytes() == authored[19:].tobytes()


@pytest.mark.parametrize("value", [-1e6, -1., 1., 1e6])
def test_split_feedback_preserves_residual_action_and_root_bounds(analytical, value):
    env = analytical(features=np.tile([value, value, -value], (22, 1)))
    action, _, authored, delta = actor.action_sample(np.full((5, 2), .25),
        actor.joint_groups(env.action_names, profile=actor.SPLIT_POLICY_PROFILE), env)
    assert np.all(abs(delta) <= .25) and np.all(abs(action) <= 1.)
    assert np.all(abs(action[:19] - authored[:19]) <= .25 + 1e-16)
    assert action[19:].tobytes() == authored[19:].tobytes()


@pytest.mark.parametrize("person", [False, True])
@pytest.mark.parametrize("assistance", [False, True])
def test_explicit_split_policy_roundtrip_and_prepared_cache(analytical, monkeypatch, person, assistance):
    env = analytical(person=person, assistance=assistance)
    policy = policy_for(env, actor.SPLIT_POLICY_PROFILE, np.full((5, 2), .1))
    assert actor.validate_shared_exercise_feedback_policy(policy, env.scene, env.task) == policy
    prepared = actor.prepare_shared_exercise_feedback_policy(policy, env)
    policy["feedback_coefficients"][0][0] = -.25
    monkeypatch.setattr(actor, "runtime_hashes", lambda *_: pytest.fail("Hot-path runtime hash"))
    monkeypatch.setattr(actor, "_context", lambda *_: pytest.fail("Hot-path scene admission"))
    first = prepared(); second = prepared()
    assert env.calls == 2 and first.tobytes() == second.tobytes()
    np.testing.assert_allclose(first[:19], .32, rtol=0., atol=1e-16)
    if assistance:
        np.testing.assert_allclose(first[19:], .3, rtol=0., atol=1e-16)


@pytest.mark.parametrize("profile", PROFILES)
def test_make_policy_never_infers_profile_from_weights(analytical, profile):
    env = analytical()
    wrong = np.zeros((5, 2) if profile == actor.POLICY_PROFILE else (3, 2))
    with pytest.raises(ValueError, match="coefficients"):
        policy_for(env, profile, wrong)


@pytest.mark.parametrize("field,value", [
    ("profile", actor.POLICY_PROFILE), ("algorithm", actor.ALGORITHM),
    ("actor_profile", actor.ACTOR_PROFILE), ("groups", list(actor.GROUP_NAMES)),
    ("groups", ["torso", "leg_proximal", "leg_distal", "arm_proximal", "arm_distal"]),
    ("feedback_coefficients", [[0., 0.]] * 3), ("residual_bound", .5),
    ("runtime_sha256", {}), ("evaluator_sha256", "0" * 64), ("gates_sha256", "0" * 64),
    ("root_assistance", "disabled"), ("scope", actor.SCOPE), ("privileged_simulator_features", False),
])
def test_v5_rejects_v4_relabeling_and_changed_contract(analytical, field, value):
    env = analytical(); policy = policy_for(env, actor.SPLIT_POLICY_PROFILE)
    policy[field] = value
    with pytest.raises(ValueError):
        actor.validate_shared_exercise_feedback_policy(policy, env.scene, env.task)


def test_unknown_profile_rejects_before_context_or_features(analytical, monkeypatch):
    env = analytical(); policy = policy_for(env)
    policy["profile"] = "shared-exercise-policy-v6"
    monkeypatch.setattr(actor, "_context", lambda *_: pytest.fail("Unknown profile reached scene admission"))
    with pytest.raises(ValueError, match="explicit supported"):
        actor.prepare_shared_exercise_feedback_policy(policy, env)
    assert env.calls == 0


def test_existing_v4_policy_document_matches_prechange_golden(analytical):
    env = analytical()
    policy = policy_for(env)
    assert policy["profile"] == actor.POLICY_PROFILE
    assert actor.validate_shared_exercise_feedback_policy(policy, env.scene, env.task) == policy
    assert canonical_hash(policy) == "748bb1a3c3ff831a3f373b66be475de0a9646c7ae086618f31421ce3b7d41d4e"
