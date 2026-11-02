"""Versioned joint feedback corrections; authored root/bias and task gates stay fixed."""

from copy import deepcopy
import math
from pathlib import Path

import numpy as np

from weft.io import canonical_hash, digest
from weft.mobile_human import ROOT_AXES
from weft.shared_exercise_feasibility_learning import RULE, _validate_rank
from weft.shared_exercise_lab import _runtime_versions
from weft.shared_exercise_learning import _bytes, _context, _groups, _integer
from weft.shared_exercise_residual_learning import POLICY_RUNTIME_FILES as BASE_RUNTIME_FILES
from weft.shared_exercise_task import GATES_SHA256
from weft.shared_world import HUMAN_ONLY_PROFILE
from weft.spatial_pose import SPATIAL_JOINT_NAMES


POLICY_PROFILE = "shared-exercise-policy-v4"
SPLIT_POLICY_PROFILE = "shared-exercise-policy-v5"
ALGORITHM = "shared_exercise_feedback_residual_feasibility_cem_v1"
SPLIT_ALGORITHM = "shared_exercise_split_limb_feedback_residual_feasibility_cem_v1"
ACTOR_PROFILE = "shared-exercise-bounded-feedback-residual-v1"
SPLIT_ACTOR_PROFILE = "shared-exercise-bounded-split-limb-feedback-residual-v1"
FEATURE_NAMES = ("normalized_position_feedback", "normalized_velocity_feedback")
GROUP_NAMES = ("torso", "proximal", "distal")
SPLIT_GROUP_NAMES = ("torso", "arm_proximal", "arm_distal", "leg_proximal", "leg_distal")
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
SPLIT_SCOPE = ("Training-only feasibility search over ten bounded joint position/velocity-feedback corrections. "
               "Arm and leg proximal/distal groups are separate; the existing normalized feedback features are "
               "unchanged. Authored reference/bias feedback and root assistance remain privileged and unchanged; "
               "no bias or root coefficient is learned and equipment forces remain forbidden. Zero coefficients "
               "are authored unchanged, not learned improvement. Fixed scene/reference/task gates and changed-start "
               "development evaluation do not establish recovered anatomy, physical transfer or unfamiliar-video "
               "generalization.")
_SPLIT_JOINT_GROUPS = {name: 0 if name.startswith("torso_") else 1 if "shoulder" in name else
                     2 if name.endswith("_elbow") else 3 if "hip" in name else 4
                     for name in SPATIAL_JOINT_NAMES}


def contract(profile=POLICY_PROFILE):
    if type(profile) is not str or profile not in (POLICY_PROFILE, SPLIT_POLICY_PROFILE):
        raise ValueError("Feedback residual requires an explicit supported policy profile")
    split = profile == SPLIT_POLICY_PROFILE
    return {"profile": profile, "algorithm": SPLIT_ALGORITHM if split else ALGORITHM,
            "actor_profile": SPLIT_ACTOR_PROFILE if split else ACTOR_PROFILE,
            "groups": SPLIT_GROUP_NAMES if split else GROUP_NAMES,
            "coefficient_shape": (5, 2) if split else (3, 2), "scope": SPLIT_SCOPE if split else SCOPE}


def runtime_hashes(scene):
    names = (*POLICY_RUNTIME_FILES, "person_inventory.py") if scene["profile"] == HUMAN_ONLY_PROFILE else POLICY_RUNTIME_FILES
    if "native_clock" in scene.get("human_spec", {}):
        names = (*names, "source_clock.py")
    if "native_interval" in scene.get("task", {}):
        names = (*names, "native_interval.py")
    return {name: digest(Path(__file__).with_name(name)) for name in names}


def coefficients(value, *, profile=POLICY_PROFILE):
    rows, columns = contract(profile)["coefficient_shape"]
    if (not isinstance(value, list) or len(value) != rows or any(not isinstance(row, list) or len(row) != columns
            or any(type(v) not in (int, float) or not math.isfinite(v) or abs(v) > COEFFICIENT_BOUND for v in row)
            for row in value)):
        raise ValueError(f"Feedback residual requires finite {rows}-by-{columns} coefficients in [-.25,.25]")
    result = np.asarray(value, dtype=float)
    result.flags.writeable = False
    return result


def joint_groups(names, *, profile=POLICY_PROFILE):
    contract(profile)
    if len(names) not in (19, 22):
        raise ValueError("Feedback residual permits only nineteen joints and optional three root channels")
    if profile == SPLIT_POLICY_PROFILE:
        if (any(type(name) is not str for name in names) or set(names[:19]) != set(SPATIAL_JOINT_NAMES)
                or len(names) == 22 and tuple(names[19:]) != ROOT_AXES):
            raise ValueError("Split-limb feedback requires the exact original joint and root channels")
        result = np.asarray([_SPLIT_JOINT_GROUPS[name] for name in names[:19]], dtype=int)
        result.flags.writeable = False
        return result
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
    keys = {"schema_version", "profile", "algorithm", "actor_profile", "scene_sha256", "task_sha256",
            "evaluator_sha256", "gates_sha256", "runtime_sha256", "runtime_versions", "action_names",
            "feature_names", "groups", "feedback_coefficients", "coefficient_bound", "feature_bound",
            "residual_bound", "root_assistance", "training_seed", "training_score", "optimization_rank",
            "selection_objective", "privileged_simulator_features", "scope"}
    if (not isinstance(policy, dict) or set(policy) != keys or type(policy["schema_version"]) is not int
            or policy["schema_version"] != 1 or len(_bytes(policy)) > 64 * 1024):
        raise ValueError("Feedback residual policy differs from its versioned bounded contract")
    spec = contract(policy["profile"])
    if policy["algorithm"] != spec["algorithm"] or policy["actor_profile"] != spec["actor_profile"]:
        raise ValueError("Feedback residual policy differs from its versioned bounded contract")
    coefficients(policy["feedback_coefficients"], profile=policy["profile"])
    scene, task, count = _context(scene, task)
    hashes = runtime_hashes(scene)
    if (policy["scene_sha256"] != canonical_hash(scene) or policy["task_sha256"] != canonical_hash(task)
            or policy["gates_sha256"] != GATES_SHA256 or policy["runtime_sha256"] != hashes
            or policy["evaluator_sha256"] != hashes["shared_exercise_task.py"]
            or policy["runtime_versions"] != _runtime_versions()):
        raise ValueError("Feedback residual source, task, gates or runtime differs")
    fixed = {"action_names": scene["action_contract"]["names"][:count], "feature_names": list(FEATURE_NAMES),
             "groups": list(spec["groups"]), "root_assistance": task["control"]["root_assistance"],
             "privileged_simulator_features": True, "selection_objective": RULE, "scope": spec["scope"]}
    if any(policy[name] != value for name, value in fixed.items()) or policy["privileged_simulator_features"] is not True:
        raise ValueError("Feedback residual cannot add channels, alter assistance or hide privileged features")
    if policy["profile"] == SPLIT_POLICY_PROFILE:
        joint_groups(fixed["action_names"], profile=policy["profile"])
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
    weights = coefficients(policy["feedback_coefficients"], profile=policy["profile"])
    groups = joint_groups(env.action_names, profile=policy["profile"])
    return lambda: action_sample(weights, groups, env)[0]


def policy_action(policy, env):
    return prepare_shared_exercise_feedback_policy(policy, env)()


def make_policy(env, weights, seed, score, rank, hashes, *, profile=POLICY_PROFILE):
    spec = contract(profile)
    coefficients(weights.tolist(), profile=profile)
    return {"schema_version": 1, "profile": profile, "algorithm": spec["algorithm"], "actor_profile": spec["actor_profile"],
            "scene_sha256": env.scene_sha256, "task_sha256": env.task_sha256,
            "evaluator_sha256": env.evaluator_sha256, "gates_sha256": GATES_SHA256,
            "runtime_sha256": hashes, "runtime_versions": _runtime_versions(),
            "action_names": list(env.action_names), "feature_names": list(FEATURE_NAMES), "groups": list(spec["groups"]),
            "feedback_coefficients": weights.tolist(), "coefficient_bound": COEFFICIENT_BOUND,
            "feature_bound": FEATURE_BOUND, "residual_bound": RESIDUAL_BOUND,
            "root_assistance": env.task["control"]["root_assistance"], "training_seed": seed,
            "training_score": score, "optimization_rank": deepcopy(rank), "selection_objective": RULE,
            "privileged_simulator_features": True, "scope": spec["scope"]}
