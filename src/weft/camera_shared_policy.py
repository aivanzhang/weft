"""External camera-task research policies; source/task qualification stays unchanged."""

from copy import deepcopy
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from weft.camera_shared_receipt import IDENTITIES
from weft.camera_shared_task import OBSERVATION_PROFILE, prepare_camera_shared_task
from weft.io import canonical_hash, digest
from weft.shared_exercise_feedback_residual import action_sample as feedback_sample, joint_groups
from weft.shared_exercise_lab import _runtime_versions, runtime_files
from weft.shared_exercise_learning import _bytes, _integer
from weft.shared_exercise_residual_learning import BASIS_PROFILE, phase_basis


PROFILE = "camera-shared-policy-v1"
ALGORITHM = "camera_shared_phase_feedback_cem_v1"
ACTOR_PROFILE = "camera-shared-phase-feedback-residual-v1"
GROUP_NAMES = ("torso", "proximal", "distal")
FEATURE_NAMES = ("normalized_position_feedback", "normalized_velocity_feedback", "declared_interval_elapsed_fraction")
SHAPE = (2, 3, 2)
COEFFICIENT_BOUND = RESIDUAL_BOUND = .25
FEATURE_BOUND = 1.
OBJECTIVE = "Original mean reward minus one for incomplete intervals; reset failures score -2 and cannot select a policy"
SCOPE = ("Unqualified research: bounded search over phase-conditioned joint feedback in the exact camera task. "
         "Reference, simulator feedback and bias are privileged; any authored root assistance stays unchanged. "
         "The source scene and task remain unqualified for learning. A searched external controller is not "
         "retroactive source qualification, learned locomotion, recovered support, anatomy or physical transfer. "
         "Zero coefficients are authored unchanged; nonzero coefficients alone do not prove useful learning.")
EXTRA_RUNTIME_FILES = (
    "camera_shared_policy.py", "camera_shared_learning.py", "shared_exercise_lab.py", "shared_world_packaging.py",
    "shared_exercise_feedback_residual.py", "shared_exercise_feasibility_learning.py", "shared_exercise_selection.py",
    "shared_exercise_residual_learning.py", "shared_exercise_learning.py", "shared_exercise_policies.py",
    "shared_exercise_feedback_learning.py", "whole_body_learning.py",
)


def runtime_hashes(scene):
    if not isinstance(scene, dict) or scene.get("profile") != "shared-world-v6":
        raise ValueError("Camera policy runtime requires the actual camera world")
    names = sorted(set(runtime_files(scene)) | set(EXTRA_RUNTIME_FILES))
    package = Path(__file__).parent
    if len(names) > 96 or sum((package / name).stat().st_size for name in names) > 16 << 20:
        raise ValueError("Camera policy runtime exceeds its bounded source closure")
    return {name: digest(package / name) for name in names}


def coefficients(value):
    if (not isinstance(value, list) or len(value) != 2 or any(not isinstance(phase, list) or len(phase) != 3
            or any(not isinstance(group, list) or len(group) != 2
                   or any(type(v) not in (int, float) or not -.25 <= v <= .25 or not math.isfinite(v) for v in group)
                   for group in phase) for phase in value)):
        raise ValueError("Camera feedback coefficients require finite 2-by-3-by-2 values in [-.25,.25]")
    result = np.asarray(value, dtype=float)
    result.flags.writeable = False
    return result


def action_sample(weights, groups, env):
    """Use one public phase and one existing feedback sample, never future observations."""
    observation = np.asarray(env.observation())
    if observation.shape != (156,) or observation.dtype.kind not in "fiu" or not np.isfinite(observation).all():
        raise ValueError("Camera policy requires the existing finite 156-value observation")
    phase = float(observation[-1])
    effective = np.tensordot(phase_basis(phase), weights, axes=1)
    features = np.asarray(env.feature_matrix())
    # Direct bounds reject signed-integer minima before the reused kernel's abs() check.
    if (features.shape != (env.action_count, 3) or features.dtype.kind not in "fiu" or not np.isfinite(features).all()
            or np.any(features < -1e6) or np.any(features > 1e6)):
        raise ValueError("Camera policy requires finite bounded existing feedback features")
    cached = SimpleNamespace(action_count=env.action_count, feature_matrix=lambda: features)
    action, features, authored, correction = feedback_sample(effective, groups, cached)
    return action, features, authored, correction, phase


def selection_kind(policy):
    if not isinstance(policy, dict) or policy.get("profile") != PROFILE:
        raise ValueError("Camera selection labels require a distinct camera policy")
    return "authored_unchanged" if not np.any(coefficients(policy.get("feedback_coefficients"))) else "searched_phase_feedback_residuals"


def _fixed(scene, task, hashes):
    count = 22 if task["control"]["root_assistance"] == "explicit_3_axis" else 19
    return {"schema_version": 1, "profile": PROFILE, "algorithm": ALGORITHM, "actor_profile": ACTOR_PROFILE,
        "scene_sha256": canonical_hash(scene), "task_sha256": canonical_hash(task),
        **{key: task[key] for key in IDENTITIES}, "gate_profile": task["gate_profile"], "gates_sha256": task["gates_sha256"],
        "runtime_sha256": hashes, "runtime_versions": _runtime_versions(), "observation_profile": OBSERVATION_PROFILE,
        "action_names": scene["action_contract"]["names"][:count], "groups": list(GROUP_NAMES),
        "feature_names": list(FEATURE_NAMES), "basis_profile": BASIS_PROFILE, "coefficient_shape": list(SHAPE),
        "coefficient_bound": COEFFICIENT_BOUND, "feature_bound": FEATURE_BOUND, "residual_bound": RESIDUAL_BOUND,
        "root_assistance": task["control"]["root_assistance"], "root_residual_channels": [], "object_force_channels": [],
        "privileged_simulator_features": True, "selection_objective": OBJECTIVE, "scope": SCOPE}


def validate_prepared_policy(policy, scene, task, *, hashes=None):
    """Validate against an already prepared camera context; this helper never compiles a model."""
    if (scene.get("profile") != "shared-world-v6" or scene.get("objects") != []
            or scene.get("learned_policy_available") is not False or task.get("profile") != "camera-shared-task-v1"
            or task.get("scene_sha256") != canonical_hash(scene) or task.get("scope", {}).get("learning") != "unqualified"):
        raise ValueError("Camera policy cannot replace its real scene or source/task qualification")
    hashes = runtime_hashes(scene) if hashes is None else hashes
    fixed = _fixed(scene, task, hashes)
    keys = {*fixed, "feedback_coefficients", "training_seed", "training_score"}
    if (not isinstance(policy, dict) or set(policy) != keys or len(_bytes(policy)) > 64 << 10
            or any(type(policy[key]) is not type(value) or policy[key] != value for key, value in fixed.items())):
        raise ValueError("Camera policy profile, runtime, source, task, action or privilege contract differs")
    coefficients(policy["feedback_coefficients"])
    _integer(policy["training_seed"], "training seed", 0, 2**31 - 1)
    score = policy["training_score"]
    if type(score) not in (int, float) or not -1e6 <= score <= 1 or not math.isfinite(score):
        raise ValueError("Camera policy requires its finite unchanged training score")
    return deepcopy(policy)


def validate_camera_shared_policy(policy, scene, task):
    actual, task = prepare_camera_shared_task(task, scene)
    return validate_prepared_policy(policy, actual, task)


def make_policy(env, weights, seed, score, hashes):
    return {**_fixed(env.scene, env.task, hashes), "feedback_coefficients": weights.tolist(),
            "training_seed": seed, "training_score": float(score)}


def prepare_camera_shared_policy(policy, env):
    """Freeze validation and coefficients once; actions do no source I/O or hashing."""
    policy = validate_camera_shared_policy(policy, env.scene, env.task)
    if (env.evaluator_sha256 != policy["evaluator_sha256"]
            or env.observation_contract()["profile"] != OBSERVATION_PROFILE
            or list(env.action_names) != policy["action_names"]):
        raise ValueError("Camera policy environment or observation identity differs")
    weights, groups = coefficients(policy["feedback_coefficients"]), joint_groups(env.action_names)
    return lambda: action_sample(weights, groups, env)[0]
