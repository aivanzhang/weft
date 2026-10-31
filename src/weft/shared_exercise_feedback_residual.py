"""Versioned joint feedback corrections; authored root/bias and task gates stay fixed."""

from copy import deepcopy
import math
from pathlib import Path

import numpy as np

from weft.io import canonical_hash, digest
from weft.shared_exercise_feasibility_learning import RULE, _validate_rank
from weft.shared_exercise_lab import _runtime_versions
from weft.shared_exercise_learning import _bytes, _context, _groups, _integer
from weft.shared_exercise_residual_learning import POLICY_RUNTIME_FILES as BASE_RUNTIME_FILES
from weft.shared_exercise_task import GATES_SHA256
from weft.shared_world import HUMAN_ONLY_PROFILE


POLICY_PROFILE = "shared-exercise-policy-v4"
ALGORITHM = "shared_exercise_feedback_residual_feasibility_cem_v1"
ACTOR_PROFILE = "shared-exercise-bounded-feedback-residual-v1"
FEATURE_NAMES = ("normalized_position_feedback", "normalized_velocity_feedback")
GROUP_NAMES = ("torso", "proximal", "distal")
COEFFICIENT_BOUND = .25
FEATURE_BOUND = 1.
RESIDUAL_BOUND = .25
POLICY_RUNTIME_FILES = (*BASE_RUNTIME_FILES, "shared_exercise_selection.py",
                       "shared_exercise_feasibility_learning.py", "shared_exercise_feedback_residual.py",
                       "shared_exercise_feedback_learning.py")
SCOPE = ("Training-only feasibility search over six bounded joint position/velocity-feedback corrections. "
         "The existing normalized feedback features and motor groups are reused; this is not a new observed state. "
         "Authored reference/bias feedback and root assistance remain privileged and unchanged; no bias or root "
         "coefficient is learned and equipment forces remain forbidden. Zero coefficients are authored unchanged, "
         "not learned improvement. Fixed scene/reference/task gates and changed-start development evaluation "
         "do not establish recovered anatomy, physical transfer or unfamiliar-video generalization.")


def runtime_hashes(scene):
    names = (*POLICY_RUNTIME_FILES, "person_inventory.py") if scene["profile"] == HUMAN_ONLY_PROFILE else POLICY_RUNTIME_FILES
    return {name: digest(Path(__file__).with_name(name)) for name in names}


def coefficients(value):
    if (not isinstance(value, list) or len(value) != 3 or any(not isinstance(row, list) or len(row) != 2
            or any(type(v) not in (int, float) or not math.isfinite(v) or abs(v) > COEFFICIENT_BOUND for v in row)
            for row in value)):
        raise ValueError("Feedback residual requires finite 3-by-2 coefficients in [-.25,.25]")
    result = np.asarray(value, dtype=float)
    result.flags.writeable = False
    return result


def joint_groups(names):
    if len(names) not in (19, 22):
        raise ValueError("Feedback residual permits only nineteen joints and optional three root channels")
    result = _groups(names[:19])
    if np.any(result > 2):
        raise ValueError("Feedback residual cannot learn a root or equipment channel")
    result.flags.writeable = False
    return result


def action_sample(weights, groups, env):
    """Read features once for both action and optional training diagnostics."""
    features = np.asarray(env.feature_matrix())
    if (features.shape != (env.action_count, 3) or env.action_count not in (19, 22)
            or features.dtype.kind not in "fiu" or not np.isfinite(features).all()
            or np.any(abs(features) > 1e6)):
        raise ValueError("Feedback residual requires the finite bounded existing feature matrix")
    # This is the engine's exact authored arithmetic, including its bias column.
    authored = np.clip(features.sum(axis=1), -1., 1.)
    action = authored.copy()
    residual = np.zeros(19)
    if np.any(weights):
        bounded = np.clip(features[:19, :2], -FEATURE_BOUND, FEATURE_BOUND)
        residual = np.clip(np.sum(bounded * weights[groups], axis=1), -RESIDUAL_BOUND, RESIDUAL_BOUND)
        action[:19] = np.clip(authored[:19] + residual, -1., 1.)
    return action, features, authored, residual


def validate_shared_exercise_feedback_policy(policy, scene, task):
    scene, task, count = _context(scene, task)
    keys = {"schema_version", "profile", "algorithm", "actor_profile", "scene_sha256", "task_sha256",
            "evaluator_sha256", "gates_sha256", "runtime_sha256", "runtime_versions", "action_names",
            "feature_names", "groups", "feedback_coefficients", "coefficient_bound", "feature_bound",
            "residual_bound", "root_assistance", "training_seed", "training_score", "optimization_rank",
            "selection_objective", "privileged_simulator_features", "scope"}
    if (not isinstance(policy, dict) or set(policy) != keys or type(policy["schema_version"]) is not int
            or policy["schema_version"] != 1 or policy["profile"] != POLICY_PROFILE
            or policy["algorithm"] != ALGORITHM or policy["actor_profile"] != ACTOR_PROFILE
            or len(_bytes(policy)) > 64 * 1024):
        raise ValueError("Feedback residual policy differs from its versioned bounded contract")
    coefficients(policy["feedback_coefficients"])
    hashes = runtime_hashes(scene)
    if (policy["scene_sha256"] != canonical_hash(scene) or policy["task_sha256"] != canonical_hash(task)
            or policy["gates_sha256"] != GATES_SHA256 or policy["runtime_sha256"] != hashes
            or policy["evaluator_sha256"] != hashes["shared_exercise_task.py"]
            or policy["runtime_versions"] != _runtime_versions()):
        raise ValueError("Feedback residual source, task, gates or runtime differs")
    fixed = {"action_names": scene["action_contract"]["names"][:count], "feature_names": list(FEATURE_NAMES),
             "groups": list(GROUP_NAMES), "root_assistance": task["control"]["root_assistance"],
             "privileged_simulator_features": True, "selection_objective": RULE, "scope": SCOPE}
    if any(policy[name] != value for name, value in fixed.items()) or policy["privileged_simulator_features"] is not True:
        raise ValueError("Feedback residual cannot add channels, alter assistance or hide privileged features")
    for name, expected in (("coefficient_bound", COEFFICIENT_BOUND), ("feature_bound", FEATURE_BOUND),
                           ("residual_bound", RESIDUAL_BOUND)):
        if type(policy[name]) not in (float, int) or policy[name] != expected:
            raise ValueError("Feedback residual cannot change its fixed bounds")
    _integer(policy["training_seed"], "training seed", 0, 2**31 - 1)
    if type(policy["training_score"]) not in (int, float) or not math.isfinite(policy["training_score"]):
        raise ValueError("Feedback residual training score must be finite")
    _validate_rank(policy["optimization_rank"], scene, task, policy["training_score"])
    return deepcopy(policy)


def prepare_shared_exercise_feedback_policy(policy, env):
    policy = validate_shared_exercise_feedback_policy(policy, env.scene, env.task)
    if (policy["evaluator_sha256"] != env.evaluator_sha256
            or env.observation_contract()["profile"] != "shared-exercise-observation-v1"):
        raise ValueError("Feedback residual environment/evaluator differs from its policy")
    weights = coefficients(policy["feedback_coefficients"])
    groups = joint_groups(env.action_names)
    return lambda: action_sample(weights, groups, env)[0]


def policy_action(policy, env):
    return prepare_shared_exercise_feedback_policy(policy, env)()


def make_policy(env, weights, seed, score, rank, hashes):
    return {"schema_version": 1, "profile": POLICY_PROFILE, "algorithm": ALGORITHM, "actor_profile": ACTOR_PROFILE,
            "scene_sha256": env.scene_sha256, "task_sha256": env.task_sha256,
            "evaluator_sha256": env.evaluator_sha256, "gates_sha256": GATES_SHA256,
            "runtime_sha256": hashes, "runtime_versions": _runtime_versions(),
            "action_names": list(env.action_names), "feature_names": list(FEATURE_NAMES), "groups": list(GROUP_NAMES),
            "feedback_coefficients": weights.tolist(), "coefficient_bound": COEFFICIENT_BOUND,
            "feature_bound": FEATURE_BOUND, "residual_bound": RESIDUAL_BOUND,
            "root_assistance": env.task["control"]["root_assistance"], "training_seed": seed,
            "training_score": score, "optimization_rank": deepcopy(rank), "selection_objective": RULE,
            "privileged_simulator_features": True, "scope": SCOPE}
