"""Bounded nominal-start gain search; changed-start and physical validation are unavailable."""

from copy import deepcopy
from hashlib import sha256
import json
import math
from pathlib import Path
import time

import numpy as np

from weft.io import canonical_hash
from weft.moving_equipment_task import MovingEquipmentTaskEnv
from weft.shared_world_action_replay import runtime_identity
from weft.shared_world_packaging import read_package_file, write_package_files
from weft.whole_body_learning import _action as grouped_feedback_action

PROFILE = "moving-equipment-feedback-policy-v1"
ALGORITHM = "nominal_moving_feedback_cem_v1"
MAX_CONTROL_STEPS = 20_000
MAX_OUTPUT_BYTES = 64 << 20
MAX_OUTPUT_FILES = 256
SCOPE = ("Exact-original-start feedback search for an authored virtual-force-assisted placement task. "
         "Different seeds do not create independent physical starts. Changed-start generalization, "
         "new-video accuracy, grasp and physical validation are unavailable.")
FEATURES = {"human": ["normalized_position_feedback", "normalized_velocity_feedback", "normalized_simulator_bias_force"],
    "object": ["4*authored_target_error/virtual_acceleration", "-2*velocity/virtual_acceleration", "-gravity/virtual_acceleration"],
    "other_object_targets": "original_simulated_initial_positions", "privileged_simulator_state": True}


def _integer(value, name, low, high):
    if type(value) is not int or not low <= value <= high:
        raise ValueError(f"{name} must be an integer in [{low},{high}]")


def _check(deadline):
    if time.monotonic() >= deadline:
        raise TimeoutError("Moving policy search reached its aggregate wall deadline")


def _admitted(value):
    from weft.moving_equipment_task_lab import validate_task_lab_admission
    return validate_task_lab_admission(value)


def _bytes(value, maximum=4 << 20):
    result = bytearray()
    try:
        for chunk in json.JSONEncoder(sort_keys=True, separators=(",", ":"), allow_nan=False).iterencode(value):
            encoded = chunk.encode()
            if len(result) + len(encoded) > maximum:
                raise ValueError("Moving learning document exceeds its byte bound")
            result.extend(encoded)
    except (TypeError, RecursionError) as error:
        raise ValueError("Moving learning requires bounded finite JSON") from error
    return bytes(result) + b"\n"


class _Output:
    def __init__(self, path):
        self.path, self.bytes, self.files = Path(path), 0, {}
        if self.path.exists() or self.path.is_symlink():
            raise ValueError("Choose a new moving learning output directory")
        self.path.mkdir(parents=True, exist_ok=False)

    def save(self, name, raw):
        reserve_bytes, reserve_files = (0, 0) if name in {"failure.json", "manifest.json"} else (256 << 10, 2)
        if len(self.files) >= MAX_OUTPUT_FILES - reserve_files or self.bytes + len(raw) > MAX_OUTPUT_BYTES - reserve_bytes:
            raise ValueError("Moving learning output reservation exceeded")
        write_package_files(self.path, {name: raw})
        self.files[name] = sha256(raw).hexdigest()
        self.bytes += len(raw)

    def json(self, name, value):
        self.save(name, _bytes(value))


def _runtime(admission, scene, deadline=None):
    identity = runtime_identity(scene)
    admitted_hashes = admission.runtime_sha256
    names = set(identity["runtime_sha256"]) | set(admitted_hashes) | {
        "moving_equipment_learning.py", "moving_equipment_task.py", "moving_equipment_task_lab.py",
        "whole_body_learning.py", "shared_exercise_task.py"}
    if len(names) > 160:
        raise ValueError("Moving learning runtime inventory exceeds 160 modules")
    files, hashes, total = {}, {}, 0
    for name in sorted(names):
        if deadline is not None:
            _check(deadline)
        if not isinstance(name, str) or Path(name).name != name or not name.endswith(".py"):
            raise ValueError("Moving learning runtime declaration is not a module basename")
        raw = read_package_file(Path(__file__).parent, name, max_bytes=2 << 20)
        total += len(raw)
        if total > 16 << 20:
            raise ValueError("Moving learning runtime exceeds 16 MiB")
        hashes[name] = sha256(raw).hexdigest()
        if name in admitted_hashes and hashes[name] != admitted_hashes[name]:
            raise ValueError("Task admission runtime changed before learning")
        files["implementation/" + name] = raw
    return hashes, identity["runtime_versions"], files


def _context(admission):
    admission = _admitted(admission)
    scene, task = admission.scene, admission.task
    contract = scene["action_contract"]
    dimension = 12 + 3 * len(contract["dynamic_object_ids"])
    return admission, scene, task, dimension


def _weights(value, dimension):
    if (not isinstance(value, list) or len(value) != dimension
            or any(type(x) not in (int, float) or not math.isfinite(x) or abs(x) > 3 for x in value)):
        raise ValueError("Moving policy requires bounded finite feedback weights")
    return np.asarray(value, dtype=float)


def _pins(admission, scene, task):
    return {"task_manifest_sha256": admission.manifest_sha256,
            "parent_manifest_sha256": admission.parent_manifest_sha256,
            "scene_sha256": canonical_hash(scene), "task_sha256": canonical_hash(task)}


def validate_moving_equipment_policy(policy, admission):
    admission, scene, task, dimension = _context(admission)
    _bytes(policy, 64 << 10)
    fields = {"schema_version", "profile", "algorithm", "source", "runtime_sha256", "runtime_versions",
              "action_names", "weights", "features", "training_seed", "training_score", "scope"}
    if (not isinstance(policy, dict) or set(policy) != fields or type(policy["schema_version"]) is not int
            or policy["schema_version"] != 1 or policy["profile"] != PROFILE or policy["algorithm"] != ALGORITHM):
        raise ValueError("Moving feedback policy contract differs")
    _weights(policy["weights"], dimension)
    _integer(policy["training_seed"], "training seed", 0, 2**31-1)
    if type(policy["training_score"]) not in (int, float) or not math.isfinite(policy["training_score"]):
        raise ValueError("Moving training score must be finite")
    hashes, versions, _ = _runtime(admission, scene)
    if (policy["source"] != _pins(admission, scene, task) or policy["runtime_sha256"] != hashes
            or policy["runtime_versions"] != versions or policy["action_names"] != scene["action_contract"]["names"]
            or policy["features"] != FEATURES or policy["scope"] != SCOPE):
        raise ValueError("Moving policy source, task, runtime or controls changed")
    return deepcopy(policy)


class _Feedback:
    def __init__(self, env, scene, task, weights):
        self.engine = env._engine
        self.weights = np.asarray(weights, dtype=float).copy()
        self.weights.flags.writeable = False
        self.human_count = scene["action_contract"]["human_motor_count"]
        self.ids = tuple(scene["action_contract"]["dynamic_object_ids"])
        self.count = len(scene["action_contract"]["names"])
        self.targets = {obj["id"]: np.asarray(obj["position"], dtype=float) for obj in scene["objects"]}
        self.targets[task["goal"]["object_id"]] = np.asarray(task["goal"]["position_world"], dtype=float)
        self.acceleration = float(scene["task"]["object_goal"]["virtual_acceleration"])
        self.gravity = np.asarray(scene["gravity"], dtype=float)

    def __call__(self):
        action = np.zeros(self.count)
        action[:self.human_count] = grouped_feedback_action(self.weights[:12].reshape(4, 3), self.engine)
        objects = {row["id"]: row for row in self.engine._objects_state()}
        for index, name in enumerate(self.ids):
            row = objects[name]
            features = np.stack((4 * (self.targets[name] - row["position"]),
                                 -2 * np.asarray(row["velocity"]), -self.gravity), axis=1) / self.acceleration
            start = self.human_count + 3 * index
            action[start:start+3] = np.clip(features @ self.weights[12+3*index:15+3*index], -1., 1.)
        return action


def prepare_moving_equipment_policy(policy, env, admission):
    """Validate once; the returned feedback function never scans files or hashes per action."""
    policy = validate_moving_equipment_policy(policy, admission)
    if not isinstance(env, MovingEquipmentTaskEnv) or (canonical_hash(env.scene) != policy["source"]["scene_sha256"]
            or canonical_hash(env.task) != policy["source"]["task_sha256"]):
        raise ValueError("Moving policy environment differs from its pinned task")
    return _Feedback(env, env.scene, env.task, policy["weights"])


def _score(metrics):
    return (metrics["mean_reward"] + 2 * int(metrics["success"]) - int(not metrics["completed_interval"])
            - .25 * len(metrics["failed_criteria"]))


def _trial(env, controller, weights, scene, task, seed, budget, deadline):
    row = {"controller": controller, "seed": seed, "reset": {"jitter": False}, "control_steps": 0,
           "physics_substeps": 0, "status": "running", "failure": None}
    reset_complete = False
    try:
        _check(deadline)
        env.reset(seed=seed)
        reset_complete = True
        feedback = _Feedback(env, scene, task, weights)
        rng = np.random.default_rng(seed)
        while True:
            _check(deadline)
            if budget["used"] >= budget["maximum"]:
                raise TimeoutError("Moving learning reached its aggregate control-step allowance")
            action = (rng.uniform(-1., 1., len(env.action_names)) if controller == "random" else feedback())
            budget["used"] += 1
            row["control_steps"] += 1
            _, _, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                break
        row.update(status="complete", metrics=env.metrics())
    except (RuntimeError, ValueError) as error:
        row.update(status="failed", failure=f"{type(error).__name__}: {error}"[:500])
        if not reset_complete:
            error.partial_trial = row
            raise
        if env._metrics is not None:
            row["metrics"] = env.metrics()
    except TimeoutError as error:
        error.partial_trial = row
        if env._metrics is not None:
            row["metrics"] = env.metrics()
            row["physics_substeps"] = row["metrics"]["observed_substeps"]
        raise
    metrics = row.get("metrics")
    row["physics_substeps"] = metrics["observed_substeps"] if metrics else 0
    row["score"] = _score(metrics) if metrics else -3.
    return row


def train_moving_equipment_policy(admission, out, *, seed=1729, generations=2, population=6, max_seconds=60):
    """One immutable admitted source, one simulator, bounded exact-start search and frozen-policy check."""
    started = time.monotonic()
    _integer(seed, "seed", 0, 2**31-1)
    _integer(generations, "generations", 1, 4)
    _integer(population, "population", 4, 12)
    if type(max_seconds) not in (int, float) or not math.isfinite(max_seconds) or not 1 <= max_seconds <= 60:
        raise ValueError("Moving learning requires a 1–60 second wall allowance")
    deadline = started + max_seconds
    admission, scene, task, dimension = _context(admission)
    horizon = task["interval"]["horizon_steps"]
    maximum = (generations * population + 4) * horizon
    if maximum > MAX_CONTROL_STEPS:
        raise ValueError("Moving training/baselines/evaluation exceed 20000 controls; interval is never shortened")
    hashes, versions, implementations = _runtime(admission, scene, deadline)
    output = _Output(out)
    env = None
    history, cache, evaluated = [], {}, []
    budget = {"used": 0, "maximum": maximum}
    failure = None
    try:
        for name, raw in implementations.items():
            _check(deadline);output.save(name, raw)
        for name, raw in (("scene.json", admission.scene_bytes), ("task.json", admission.task_bytes),
                          ("moving-proposals.json", admission.bundle_bytes)):
            _check(deadline);output.save(name, raw)
        settings = {"profile": PROFILE, "source": _pins(admission, scene, task), "seed": seed,
            "generations": generations, "population": population, "maximum_control_steps": maximum,
            "maximum_physics_substeps": maximum * scene["substeps"], "max_seconds": max_seconds,
            "workers": 1, "scope": SCOPE, "api_calls": 0, "inference_calls": 0,
            "selection_objective": "mean_reward + 2*success - incomplete - .25*failed_criterion_count",
            "reset": {"jitter": False}, "independent_changed_start_evaluation": "unavailable",
            "weight_bound": 3., "initial_std": .9, "minimum_std": .08,
            "candidate_anchors": ["zero", "authored_feedback", "previous_best"],
            "duplicate_nominal_policies": "reuse exact weight-vector results; no additional seed represents a new start"}
        output.json("settings.json", settings)
        env = MovingEquipmentTaskEnv(scene, task, admission.bundle)
        rng = np.random.default_rng(seed)
        mean, std = np.zeros(dimension), np.full(dimension, .9)
        best, best_score = np.zeros(dimension), -math.inf
        for generation in range(generations):
            proposals = np.clip(rng.normal(mean, std, (population, dimension)), -3., 3.)
            proposals[0], proposals[1] = 0., 1.
            if generation:
                proposals[2] = best
            rows = []
            for weights in proposals:
                _check(deadline)
                key = canonical_hash(weights.tolist())
                cached = key in cache
                if not cached:
                    cache[key] = _trial(env, "feedback", weights, scene, task, seed, budget, deadline)
                trial = cache[key]
                rows.append({"weights": weights.tolist(), "cached": cached, "trial": deepcopy(trial)})
                if trial["score"] > best_score:
                    best, best_score = weights.copy(), trial["score"]
            history.append({"generation": generation, "candidates": rows})
            output.json(f"generation-{generation+1}.json", history[-1])
            unique = {canonical_hash(row["weights"]): row for row in rows}
            elites = sorted(unique.values(), key=lambda row: -row["trial"]["score"])[:max(2, population//4)]
            array = np.asarray([row["weights"] for row in elites])
            mean, std = array.mean(axis=0), np.maximum(array.std(axis=0), .08)
        policy = {"schema_version": 1, "profile": PROFILE, "algorithm": ALGORITHM,
            "source": _pins(admission, scene, task), "runtime_sha256": hashes, "runtime_versions": versions,
            "action_names": list(env.action_names), "weights": best.tolist(), "features": deepcopy(FEATURES),
            "training_seed": seed, "training_score": best_score, "scope": SCOPE}
        policy_raw = _bytes(policy, 64 << 10)
        output.save("policy.json", policy_raw)
        policy_hash = sha256(policy_raw).hexdigest()
        # Frozen-policy replay is a reproducibility check on the same exposed start, not independent evaluation.
        for name, weights in (("zero", np.zeros(dimension)), ("authored_feedback", np.ones(dimension))):
            evaluated.append({"controller": name, "reused_training_trial": True,
                              "trial": deepcopy(cache[canonical_hash(weights.tolist())])})
        for name, weights in (("random", np.zeros(dimension)), ("frozen_policy", best)):
            evaluated.append({"controller": name, "reused_training_trial": False,
                "trial": _trial(env, name, weights, scene, task, seed, budget, deadline)})
        original = cache[canonical_hash(best.tolist())]
        repeated = evaluated[-1]["trial"]
        repeat_fields = ("status", "failure", "metrics", "score", "control_steps", "physics_substeps")
        if any(original.get(key) != repeated.get(key) for key in repeat_fields):
            raise ValueError("Frozen policy does not reproduce its exact-start training trial")
        if _runtime(admission, scene, deadline)[:2] != (hashes, versions):
            raise ValueError("Moving learning runtime changed before publication")
        if read_package_file(output.path, "policy.json", max_bytes=64 << 10) != policy_raw:
            raise ValueError("Frozen moving policy changed during nominal evaluation")
        _admitted(admission);_check(deadline)
        evaluation = {"profile": PROFILE, "policy_sha256": policy_hash, "comparisons": evaluated,
            "source": settings["source"], "scope": SCOPE, "independent_changed_start_evaluation": "unavailable",
            "baseline_starts": "same exact original start; zero/authored rows reused, not new evidence",
            "frozen_nominal_replay_matches_training": True,
            "selected_kind": "zero_unchanged" if np.all(best == 0) else "authored_unchanged" if np.all(best == 1) else "searched_feedback_gains",
            "selected_policy_success": bool(evaluated[-1]["trial"].get("metrics", {}).get("success", False))}
        output.json("evaluation.json", evaluation)
        result = {"status": "complete", "policy_sha256": policy_hash, "seconds": time.monotonic()-started,
            "control_steps": budget["used"], "maximum_control_steps": maximum, "scope": SCOPE,
            "independent_changed_start_evaluation": "unavailable", "api_calls": 0, "inference_calls": 0}
        output.json("result.json", result)
        output.json("manifest.json", {"schema_version": 1, "profile": PROFILE, "files": dict(output.files)})
        return {**result, "directory": str(output.path), "manifest_sha256": output.files["manifest.json"]}
    except Exception as error:
        failure = {"status": "failed", "failure": f"{type(error).__name__}: {error}"[:1000],
                   "control_steps": budget["used"], "completed_generations": len(history), "scope": SCOPE,
                   "partial_trial": getattr(error, "partial_trial", None), "evaluation_trials": evaluated}
        output.json("failure.json", failure)
        raise
    finally:
        if env is not None:
            env.close()


def train_saved_moving_equipment_policy(task_lab, out, *, seed=1729, generations=2, population=6, max_seconds=90):
    """Admit one sealed child once, then spend only the remaining aggregate wall allowance."""
    from weft.moving_equipment_task_lab import load_moving_equipment_task_lab
    from weft.shared_world_action_replay import _json

    started = time.monotonic()
    if type(max_seconds) not in (int, float) or not math.isfinite(max_seconds) or not 1 <= max_seconds <= 90:
        raise ValueError("Saved moving learning requires a 1–90 second total allowance")
    deadline = started + max_seconds
    source, target = Path(task_lab), Path(out)
    if ".." in source.parts or ".." in target.parts:
        raise ValueError("Saved moving learning paths cannot contain parent traversal")
    source, target = source.absolute(), target.absolute()
    if (target.exists() or target.is_relative_to(source) or source.is_relative_to(target)
            or any(path.is_symlink() for path in (source, target, *source.parents, *target.parents))):
        raise ValueError("Choose a new isolated, nonlinked output outside the task lab")
    config = _json(read_package_file(source, "lab-config.json", max_bytes=64 << 10))
    _check(deadline)
    admission = load_moving_equipment_task_lab(source / "run", config,
        max_seconds=min(30., deadline-time.monotonic()))
    remaining = min(60., deadline-time.monotonic())
    if remaining < 1:
        raise TimeoutError("Task admission exhausted the total learning allowance")
    result = train_moving_equipment_policy(admission, target, seed=seed, generations=generations,
        population=population, max_seconds=remaining)
    _check(deadline)
    return {**result, "aggregate_seconds": time.monotonic()-started, "maximum_aggregate_seconds": max_seconds,
            "source_admissions": 1}
