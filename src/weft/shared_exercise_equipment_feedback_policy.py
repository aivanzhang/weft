"""Explicit equipment-feedback policy artifacts; legacy dispatch remains unchanged."""

from copy import deepcopy
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from weft import shared_exercise_equipment_feedback as actor
from weft import shared_exercise_feedback_residual as base_actor
from weft.io import canonical_hash, digest
from weft.shared_exercise_feasibility_learning import RULE, _validate_rank
from weft.shared_exercise_lab import _json, _runtime_versions
from weft.shared_exercise_learning import _bytes, _integer, _sha
from weft.shared_exercise_selection import _hash
from weft.shared_exercise_task import GATES_SHA256, SharedExerciseTaskEnv


PROFILE = "shared-exercise-equipment-policy-v1"
ALGORITHM = "shared_exercise_equipment_feedback_feasibility_cem_v1"
POLICY_RUNTIME_FILES = (*base_actor.POLICY_RUNTIME_FILES, "shared_exercise_equipment_feedback.py",
                        "shared_exercise_equipment_feedback_policy.py", "shared_exercise_equipment_feedback_learning.py")
SCOPE = ("Training-only equipment-conditioned residual around the exact pinned v5 base. "
         "Privileged current simulator/reference features and existing joint/root authority are explicit; "
         "no equipment forces or task changes. Zero residual is v5 unchanged, not learned improvement. "
         "Original complete-interval feasibility ranking is unchanged. This is not a public/default policy, "
         "new-video generalization, measured contact or physical transfer.")
PAIRING = "serialized_current_observation_and_single_v5_feature_sample_v1"


def feature_contract():
    return {"profile": actor.PROFILE, "scope": actor.SCOPE, "groups": list(actor.GROUP_NAMES),
            "coefficient_shape": [8, 2], "coefficient_bound": actor.COEFFICIENT_BOUND,
            "residual_bound": actor.RESIDUAL_BOUND, "normalized_action_bounds": [-1., 1.],
            "normalizers": {"proximity_distance_model_units": actor.PROXIMITY_DISTANCE_MODEL_UNITS,
                            "joint_speed_rad_s": actor.JOINT_SPEED_RAD_S,
                            "closing_speed_model_units_s": actor.CLOSING_SPEED_MODEL_UNITS_S},
            "observation_profile": "shared-exercise-observation-v1", "human_prefix": 155,
            "equipment_fields": 17, "required_contacts": 0, "elapsed_fields": 1,
            "joint_channels": list(range(19)), "root_channels": [19, 20, 21], "equipment_force_channels": [],
            "current_state_pairing": PAIRING}


def runtime_hashes(scene):
    return {**base_actor.runtime_hashes(scene), **{name: digest(Path(__file__).with_name(name))
            for name in POLICY_RUNTIME_FILES[-3:]}}


def metadata_env(scene, task):
    """Reuse the task's pure metadata methods without constructing or stepping an engine."""
    env = SimpleNamespace(scene=scene, task=task, _scene=scene, _task=task, _static_support=False,
                          action_count=22, action_names=tuple(scene["action_contract"]["names"][:22]),
                          observation_size=156 + 17 * len(task["equipment_ids"]))
    env.action_contract = lambda: SharedExerciseTaskEnv.action_contract(env)
    env.observation_contract = lambda: SharedExerciseTaskEnv.observation_contract(env)
    return env


def admit_base(raw, expected_sha256, scene, task):
    _hash(expected_sha256, "base policy file")
    if type(raw) is not bytes or len(raw) > 64 * 1024 or _sha(raw) != expected_sha256:
        raise ValueError("Equipment policy requires exact bounded pinned v5 file bytes")
    base = _json(raw)
    if base.get("profile") != base_actor.SPLIT_POLICY_PROFILE:
        raise ValueError("Equipment policy requires an explicit v5 base, not another profile")
    base = base_actor.validate_shared_exercise_feedback_policy(base, scene, task)
    actor.prepare_equipment_feedback(metadata_env(scene, task), np.zeros((8, 2)))
    return base


def make_policy(env, coefficients, base_raw, source_manifest_sha256, score, rank, hashes):
    base = _json(base_raw)
    actor._coefficients(coefficients)
    return {"schema_version": 1, "profile": PROFILE, "algorithm": ALGORITHM,
            "scene_sha256": env.scene_sha256, "task_sha256": env.task_sha256,
            "evaluator_sha256": env.evaluator_sha256, "gates_sha256": GATES_SHA256,
            "source_manifest_sha256": source_manifest_sha256, "base_policy_file_sha256": _sha(base_raw),
            "base_policy_sha256": canonical_hash(base), "base_profile": base_actor.SPLIT_POLICY_PROFILE,
            "runtime_sha256": deepcopy(hashes), "runtime_versions": _runtime_versions(),
            "feature_contract": feature_contract(), "action_names": list(env.action_names),
            "equipment_feedback_coefficients": np.asarray(coefficients, dtype=float).tolist(),
            "root_assistance": "explicit_3_axis", "training_seed": base["training_seed"],
            "training_score": score, "optimization_rank": deepcopy(rank), "selection_objective": RULE,
            "privileged_simulator_features": True, "scope": SCOPE}


def _validated(policy, scene, task, base_raw, source_manifest_sha256):
    keys = {"schema_version", "profile", "algorithm", "scene_sha256", "task_sha256", "evaluator_sha256",
            "gates_sha256", "source_manifest_sha256", "base_policy_file_sha256", "base_policy_sha256",
            "base_profile", "runtime_sha256", "runtime_versions", "feature_contract", "action_names",
            "equipment_feedback_coefficients", "root_assistance", "training_seed", "training_score",
            "optimization_rank", "selection_objective", "privileged_simulator_features", "scope"}
    if (type(policy) is not dict or set(policy) != keys or type(policy["schema_version"]) is not int
            or policy["schema_version"] != 1 or policy["profile"] != PROFILE or policy["algorithm"] != ALGORITHM
            or len(_bytes(policy)) > 64 * 1024):
        raise ValueError("Equipment feedback policy differs from its explicit bounded artifact contract")
    _hash(source_manifest_sha256, "original construction manifest")
    base = admit_base(base_raw, policy["base_policy_file_sha256"], scene, task)
    hashes = runtime_hashes(scene)
    fixed = {"scene_sha256": canonical_hash(scene), "task_sha256": canonical_hash(task),
             "evaluator_sha256": hashes["shared_exercise_task.py"], "gates_sha256": GATES_SHA256,
             "source_manifest_sha256": source_manifest_sha256, "base_policy_sha256": canonical_hash(base),
             "base_profile": base_actor.SPLIT_POLICY_PROFILE, "runtime_sha256": hashes,
             "runtime_versions": _runtime_versions(), "feature_contract": feature_contract(),
             "action_names": list(scene["action_contract"]["names"][:22]), "root_assistance": "explicit_3_axis",
             "training_seed": base["training_seed"], "selection_objective": RULE,
             "privileged_simulator_features": True, "scope": SCOPE}
    if any(_bytes(policy[key]) != _bytes(value) for key, value in fixed.items()):
        raise ValueError("Equipment policy source, base, runtime, feature or task identities differ")
    actor._coefficients(policy["equipment_feedback_coefficients"])
    _integer(policy["training_seed"], "training seed", 0, 2**31 - 1)
    if type(policy["training_score"]) not in (int, float) or not math.isfinite(policy["training_score"]):
        raise ValueError("Equipment training score must be a finite original reward")
    _validate_rank(policy["optimization_rank"], scene, task, policy["training_score"])
    return deepcopy(policy), base


def validate_equipment_feedback_policy(policy, scene, task, *, base_policy_bytes, source_manifest_sha256):
    """Cold validation requires externally qualified source/base identities, not self-asserted provenance."""
    return _validated(policy, scene, task, base_policy_bytes, source_manifest_sha256)[0]


class PreparedEquipmentFeedbackPolicy:
    """The caller owns serialized environment commands and must not mutate the admitted model/task."""

    def __init__(self, env, base, coefficients):
        self._env, self._engine = env, env._engine
        self._identity = (env.scene_sha256, env.task_sha256, env.evaluator_sha256)
        self._base_weights = base_actor.coefficients(base["feedback_coefficients"], profile=base_actor.SPLIT_POLICY_PROFILE)
        self._base_groups = base_actor.joint_groups(env.action_names, profile=base_actor.SPLIT_POLICY_PROFILE)
        self._actor = actor.prepare_equipment_feedback(env, coefficients)

    def sample(self, expected_observation):
        env = self._env
        if (env._engine is not self._engine or (env.scene_sha256, env.task_sha256, env.evaluator_sha256) != self._identity
                or env._done or env.terminated or env.truncated or env._state_status != "current"):
            raise ValueError("Equipment policy requires the unchanged active serialized environment")
        current = env.observation()
        if (type(current) is not np.ndarray or current.dtype != np.dtype("float32")
                or current.shape != (self._actor._observation_size,) or not np.isfinite(current).all()
                or np.any(abs(current) > 1e6) or type(expected_observation) is not np.ndarray
                or expected_observation.dtype != current.dtype or expected_observation.shape != current.shape
                or expected_observation.tobytes() != current.tobytes()):
            raise ValueError("Equipment policy cannot reuse a stale or different returned observation")
        base, features, _, _ = base_actor.action_sample(self._base_weights, self._base_groups, env)
        if features.astype(np.float32).ravel().tobytes() != current[88:154].tobytes():
            raise ValueError("Current v5 features do not match the observed control boundary")
        action, report = self._actor.sample(base, current)
        if report["zero_coefficients"] and action.tobytes() != base.tobytes():
            raise ValueError("Zero equipment coefficients must preserve exact v5 action bytes")
        return action, {"observation": current.copy(), "base_action": base, "base_features": features.copy(), "actor": report}


def prepare_equipment_feedback_policy(policy, env, *, base_policy_bytes, source_manifest_sha256):
    policy, base = _validated(policy, env.scene, env.task, base_policy_bytes, source_manifest_sha256)
    if env.evaluator_sha256 != policy["evaluator_sha256"]:
        raise ValueError("Equipment policy environment evaluator differs from its frozen artifact")
    return PreparedEquipmentFeedbackPolicy(env, base, policy["equipment_feedback_coefficients"])
