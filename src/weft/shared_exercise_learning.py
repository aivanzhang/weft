"""Bounded human-only feedback search; evaluation never changes the policy or task gates."""

from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np

from weft.io import canonical_hash, digest, local_time
from weft.recording import _atomic_json
from weft.shared_exercise_reset import human_jitter_options
from weft.shared_exercise_lab import (
    RUNTIME_FILES, _implementation_snapshot, _json, _runtime_versions, _verified_inputs, load_shared_exercise_lab,
)
from weft.shared_exercise_task import (
    GATES_SHA256, SharedExerciseTaskEnv, build_shared_exercise_task, prepare_shared_exercise_snapshot_validator,
    validate_shared_exercise_task,
)
from weft.shared_world import HUMAN_ONLY_PROFILE, validate_shared_world
from weft.shared_world_packaging import collect_shared_source_files, read_package_file, write_package_files
from weft.whole_body_learning import _action as grouped_feedback_action


POLICY_PROFILE = "shared-exercise-policy-v1"
ALGORITHM = "shared_exercise_feedback_cem_v1"
FEATURE_NAMES = ("normalized_position_feedback", "normalized_velocity_feedback", "normalized_simulator_bias_force")
POLICY_RUNTIME_FILES = (*RUNTIME_FILES, "whole_body_learning.py", "shared_exercise_learning.py")
MAX_CONTROL_STEPS = 20_000
MAX_EVALUATION_STEPS = 3_000
MAX_OUTPUT_BYTES = 128 * 1024 * 1024
TRAIN_RESETS = (human_jitter_options(0.), human_jitter_options(.05))
CONTROLLERS = ("coast", "random", "authored", "learned")
SCOPE = ("Reward-driven search over privileged reference-feedback gains in one authored simulator task. "
         "Coast and authored weights are explicit candidates; selecting authored unchanged is not learned improvement. "
         "The full fixed source interval is reused; changed reset states are not new videos, a temporal holdout, "
         "recovered grasp, measured muscle activation or physical transfer.")


def _bytes(value):
    return (json.dumps(value, indent=2, allow_nan=False) + "\n").encode()


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _integer(value, name, low, high):
    if type(value) is not int or not low <= value <= high:
        raise ValueError(f"{name} must be an integer in [{low},{high}]")


def _seconds(value, maximum):
    if type(value) not in (int, float) or not math.isfinite(value) or not 1 <= value <= maximum:
        raise ValueError(f"Local compute requires a finite 1–{maximum} second bound")


def _check(deadline):
    if time.monotonic() > deadline:
        raise TimeoutError("Shared exercise learning/evaluation exceeded its cooperative wall-time bound")


def _seed_plan(seed, count, excluded=()):
    rng, result = np.random.default_rng(seed), []
    for _ in range(64):
        value = int(rng.integers(0, 2**31))
        if value not in excluded and value not in result:
            result.append(value)
        if len(result) == count:
            return result
    raise ValueError("Could not construct disjoint bounded reset seeds")


def _groups(names):
    return np.asarray([3 if name.startswith("root_") else 0 if name.startswith("torso_") else
                       1 if "shoulder" in name or "hip" in name else 2 for name in names], dtype=int)


def _context(scene, task, *, allow_static_support=False):
    if isinstance(scene, dict) and scene.get("profile") == "shared-world-v4" and allow_static_support is not True:
        raise ValueError("Static-support learning requires a separately qualified policy contract")
    scene = validate_shared_world(scene)
    task = validate_shared_exercise_task(task, scene)
    count = 22 if task["control"]["root_assistance"] == "explicit_3_axis" else 19
    return scene, task, count


def _runtime_hashes(scene=None):
    package = Path(__file__).parent
    names = POLICY_RUNTIME_FILES
    if scene is not None and "native_clock" in scene.get("human_spec", {}):
        names = (*names, "source_clock.py")
    return {name: digest(package / name) for name in names}


def validate_shared_exercise_policy(policy, scene, task):
    scene, task, count = _context(scene, task)
    if scene["profile"] == HUMAN_ONLY_PROFILE:
        raise ValueError("Person-only scenes require the v2 residual learner, not the v1 gain policy")
    keys = {"schema_version", "profile", "algorithm", "scene_sha256", "task_sha256", "evaluator_sha256",
            "gates_sha256", "runtime_sha256", "runtime_versions", "action_names", "groups", "feature_names", "weights",
            "root_assistance", "training_seed", "training_score", "privileged_simulator_features", "scope"}
    if (not isinstance(policy, dict) or set(policy) != keys or type(policy["schema_version"]) is not int
            or policy["schema_version"] != 1 or policy["profile"] != POLICY_PROFILE or policy["algorithm"] != ALGORITHM
            or len(_bytes(policy)) > 64 * 1024):
        raise ValueError("Shared exercise policy differs from its bounded versioned contract")
    groups = ["torso", "proximal", "distal"] + (["root"] if count == 22 else [])
    weights = policy["weights"]
    if (not isinstance(weights, list) or len(weights) != len(groups)
            or any(not isinstance(row, list) or len(row) != 3 or
                   any(type(value) not in (int, float) or not math.isfinite(value) or abs(value) > 3 for value in row)
                   for row in weights)):
        raise ValueError("Exercise policy gains must be finite numeric group-by-3 weights in [-3,3]")
    expected = _runtime_hashes(scene)
    if (policy["scene_sha256"] != canonical_hash(scene) or policy["task_sha256"] != canonical_hash(task)
            or policy["gates_sha256"] != GATES_SHA256 or policy["runtime_sha256"] != expected
            or policy["evaluator_sha256"] != expected["shared_exercise_task.py"]
            or policy["runtime_versions"] != _runtime_versions()):
        raise ValueError("Exercise policy task, scene, evaluator or physics runtime differs")
    if (policy["action_names"] != scene["action_contract"]["names"][:count] or policy["groups"] != groups
            or policy["feature_names"] != list(FEATURE_NAMES)
            or policy["root_assistance"] != task["control"]["root_assistance"]
            or policy["privileged_simulator_features"] is not True or policy["scope"] != SCOPE):
        raise ValueError("Exercise policy cannot add object actions, assistance or hidden feature channels")
    _integer(policy["training_seed"], "training seed", 0, 2**31 - 1)
    if type(policy["training_score"]) not in (int, float) or not math.isfinite(policy["training_score"]):
        raise ValueError("Exercise policy training score must be finite")
    return deepcopy(policy)


class _Features:
    def __init__(self, env):
        self.env = env
        self.group_indices = _groups(env.action_names)
        self.group_indices.flags.writeable = False

    def feature_matrix(self):
        return self.env.feature_matrix()


def prepare_shared_exercise_policy(policy, env):
    """Pin a validated human-only policy once; actions never rehash the immutable scene."""
    policy = validate_shared_exercise_policy(policy, env.scene, env.task)
    if policy["evaluator_sha256"] != env.evaluator_sha256:
        raise ValueError("Exercise environment evaluator differs from its policy")
    weights, features = np.asarray(policy["weights"], dtype=float), _Features(env)
    weights.flags.writeable = False
    return lambda: grouped_feedback_action(weights, features)


def policy_action(policy, env):
    return prepare_shared_exercise_policy(policy, env)()


def _source(run, task_path, root_assistance, deadline, *, allow_static_support=False):
    if type(root_assistance) is not bool or task_path is not None and root_assistance:
        raise ValueError("Choose the supplied task or default-task assistance, not both")
    root = Path(run).resolve()
    root = root / "run" if (root / "run/manifest.json").is_file() else root
    manifest, files = _verified_inputs(root, deadline=deadline)
    if "scene.json" not in files:
        raise ValueError("Exercise learning requires a sealed shared-world scene")
    scene_raw = files["scene.json"]
    if _json(scene_raw).get("profile") == "shared-world-v4" and allow_static_support is not True:
        raise ValueError("Static-support learning requires a separately qualified policy contract")
    scene = validate_shared_world(_json(scene_raw))
    if "exercise-lab.json" in files:
        load_shared_exercise_lab(root, deadline=deadline)
    if task_path is not None:
        path = Path(task_path).absolute()
        task_raw = read_package_file(path.parent, path.name, max_bytes=64 * 1024)
        task = validate_shared_exercise_task(_json(task_raw), scene)
    elif "exercise-task.json" in files:
        task_raw = files["exercise-task.json"]
        task = validate_shared_exercise_task(_json(task_raw), scene)
        if root_assistance and task["control"]["root_assistance"] != "explicit_3_axis":
            raise ValueError("Do not override the sealed task's root assistance")
    else:
        if scene["profile"] == "shared-world-v4":
            raise ValueError("Static-support learning requires an explicit sealed support task")
        task = build_shared_exercise_task(scene, root_assistance=root_assistance)
        task_raw = _bytes(task)
    proof = collect_shared_source_files(scene, root, expected_manifest_sha256=_sha(manifest))
    return root, scene, task, scene_raw, task_raw, manifest, proof


def _output_path(source, out):
    source, out = Path(source).resolve(), Path(out).resolve()
    if out.exists():
        raise FileExistsError("Exercise learning/evaluation requires a new output directory")
    if out == source or out.is_relative_to(source) or source.is_relative_to(out):
        raise ValueError("Exercise learning/evaluation output must be outside its immutable input")
    return out


def _package(out, scene_raw, task_raw, manifest, proof, *, extra=None):
    implementation = _implementation_snapshot()
    reserved = {"manifest.json", "implementation", "scene.json", "exercise-task.json", "policy.json", "training.json",
                "settings.json", "evaluation.json", "failure.json", "input-manifest.json", "source-verification.json"}
    proof.validate_layout(reserved_paths=reserved)
    files = {**dict(proof.files), "scene.json": scene_raw, "exercise-task.json": task_raw,
             "input-manifest.json": manifest, "source-verification.json": _bytes(proof.report),
             **{"implementation/" + name: raw for name, raw in implementation.items()}, **(extra or {})}
    if len(files) > 512 or sum(map(len, files.values())) > MAX_OUTPUT_BYTES - 32 * 1024 * 1024:
        raise ValueError("Exercise learning source/runtime package exceeds its reserved output bound")
    out.mkdir(parents=True, mode=0o700, exist_ok=False)
    try:
        write_package_files(out, files)
    except Exception as error:
        _write(out, "failure.json", {"status": "failed", "stage": "input_packaging",
            "error": f"{type(error).__name__}: {error}"[:2000], "partial_evidence_preserved": True})
        _seal(out, "failed")
        raise
    return {name: _sha(raw) for name, raw in implementation.items()}


def _native_source_views(source, manifest, scene):
    from weft.shared_exercise_lab import native_source_input_names, native_source_views

    if "native_clock" not in scene.get("human_spec", {}):
        return {}
    hashes = _json(manifest)["files"]
    files = {name: read_package_file(source, name, max_bytes=64 * 1024 * 1024, expected_sha256=hashes[name])
             for name in native_source_input_names(scene) if name in hashes}
    return native_source_views(files, scene)


def _write(out, name, value):
    raw = _bytes(value)
    if len(raw) > 8 * 1024 * 1024:
        raise ValueError("Exercise learning metadata exceeds its 8MiB bound")
    _atomic_json(out / name, value)


def _seal(out, status):
    paths = [path for path in out.rglob("*") if path.is_file() and path != out / "manifest.json"]
    if len(paths) > 512 or sum(path.stat().st_size for path in paths) > MAX_OUTPUT_BYTES:
        raise ValueError("Exercise learning output exceeds its file/byte bound; partial evidence preserved")
    _write(out, "manifest.json", {"schema_version": 1, "status": status, "created_at": local_time(),
        "files": {path.relative_to(out).as_posix(): digest(path) for path in sorted(paths)}})
    return digest(out / "manifest.json")


def _failure(out, state_name, state, error):
    failure = {"status": "failed", "error": f"{type(error).__name__}: {error}"[:2000], "partial_evidence_preserved": True}
    _write(out, "failure.json", failure)
    try:
        _write(out, state_name, state)
        _seal(out, "failed")
    except (ValueError, OSError) as preservation_error:
        _write(out, "failure.json", failure | {"unsealed_partial_output": True,
            "checkpoint_error": str(preservation_error)[:2000]})


def _trial(env, controller, seed, reset, weights, deadline, *, trace=False, validator=None):
    row = {"seed": seed, "reset_options": deepcopy(reset), "initial_state_sha256": None,
           "status": "starting", "control_steps": 0, "return": 0., "metrics": None}
    try:
        _check(deadline)
        try:
            env.reset(seed=seed, options=reset)
        except (ValueError, RuntimeError) as error:
            return row | {"status": "reset_failed", "error": f"{type(error).__name__}: {error}"[:2000]}
        initial = env.snapshot()
        row.update(initial_state_sha256=canonical_hash(initial["engine_state"]), initial_metrics=env.metrics())
        if trace:
            row.update(initial_snapshot=initial, actions=[])
        features = _Features(env)
        rng = np.random.default_rng(seed ^ 0x5A17)
        validate = validator or prepare_shared_exercise_snapshot_validator(env.scene, env.task)
        for _ in range(env.horizon_steps):
            _check(deadline)
            action = (np.zeros(env.action_count) if controller == "coast" else
                      rng.uniform(-1, 1, env.action_count) if controller == "random" else
                      env.authored_action() if controller == "authored" else grouped_feedback_action(weights, features))
            _, reward, terminated, truncated, _ = env.step(action)
            row["control_steps"] += 1
            row["return"] += float(reward)
            if trace:
                snapshot = env.snapshot()
                validate(snapshot)
                row["actions"].append({"action": action.tolist(), "reward": float(reward), "terminated": terminated,
                                       "truncated": truncated, "snapshot_sha256": canonical_hash(snapshot)})
            if terminated or truncated:
                break
        metrics = env.metrics()
        row.update(status="completed" if metrics["completed_interval"] else "failed",
                   metrics=metrics, terminated=env.terminated, truncated=env.truncated)
        final = env.snapshot()
        validate(final)
        if trace:
            row["final_snapshot"] = final
        _check(deadline)
        return row
    except Exception as error:
        row.update(status="interrupted", error=f"{type(error).__name__}: {error}"[:2000])
        if row["initial_state_sha256"] is not None:
            row["metrics"] = env.metrics()
        error.partial_trial = row
        raise


def _score(trials):
    return float(np.mean([row["metrics"]["mean_reward"] - float(not row["metrics"]["completed_interval"])
                          if row["metrics"] is not None else -2. for row in trials]))


def _policy(scene, task, env, weights, seed, score, hashes):
    return {"schema_version": 1, "profile": POLICY_PROFILE, "algorithm": ALGORITHM,
            "scene_sha256": env.scene_sha256, "task_sha256": env.task_sha256,
            "evaluator_sha256": env.evaluator_sha256, "gates_sha256": GATES_SHA256,
            "runtime_sha256": hashes, "runtime_versions": _runtime_versions(),
            "action_names": list(env.action_names), "groups": ["torso", "proximal", "distal"] +
                (["root"] if env.action_count == 22 else []), "feature_names": list(FEATURE_NAMES),
            "weights": weights.tolist(), "root_assistance": task["control"]["root_assistance"],
            "training_seed": seed, "training_score": score, "privileged_simulator_features": True, "scope": SCOPE}


def train_shared_exercise_policy(run, out, *, task_path=None, root_assistance=False, seed=1729,
                                 generations=3, population=8, max_seconds=60):
    """Search gains using only two declared training starts; never evaluate or tune on later starts."""
    _integer(seed, "training seed", 0, 2**31 - 1)
    _integer(generations, "generations", 1, 8)
    _integer(population, "population", 4, 16)
    _seconds(max_seconds, 60)
    out = _output_path(run, out)
    started, deadline = time.monotonic(), time.monotonic() + max_seconds
    source, scene, task, scene_raw, task_raw, manifest, proof = _source(run, task_path, root_assistance, deadline)
    if scene["profile"] == HUMAN_ONLY_PROFILE:
        raise ValueError("Person-only scenes require the v2 residual learner, not the v1 gain trainer")
    maximum_steps = generations * population * 2 * task["interval"]["horizon_steps"]
    if maximum_steps > MAX_CONTROL_STEPS:
        raise ValueError("Training exceeds the 20000-control-step bound; never shorten the task interval silently")
    hashes = _runtime_hashes(scene)
    _check(deadline)
    _package(out, scene_raw, task_raw, manifest, proof, extra=_native_source_views(source, manifest, scene))
    training_seeds = _seed_plan(seed, 2)
    settings = {"schema_version": 1, "algorithm": ALGORITHM, "created_at": local_time(), "seed": seed,
        "scene_sha256": canonical_hash(scene), "task_sha256": canonical_hash(task), "gates_sha256": GATES_SHA256,
        "runtime_sha256": hashes, "source_manifest_sha256": _sha(manifest), "training_seeds": training_seeds,
        "training_reset_options": list(TRAIN_RESETS), "generations": generations, "population": population,
        "elite_count": max(2, population // 4), "initial_mean": 0., "initial_std": .9, "minimum_std": .08,
        "weight_bounds": [-3., 3.], "maximum_control_steps": maximum_steps,
        "maximum_physics_substeps": maximum_steps * scene["substeps"], "max_seconds": max_seconds,
        "workers": 1, "external_api_usd": 0, "horizon_steps": task["interval"]["horizon_steps"],
        "selection_objective": "mean(task mean_reward -1 for incomplete intervals); reset failure score=-2; loss=-score",
        "candidate_anchors": ["coast", "authored", "previous_best_after_first_generation"],
        "duplicate_candidates": "reuse identical weights on identical training seeds and reset options",
        "source_scope": task["source_scope"], "scope": SCOPE}
    _write(out, "settings.json", settings)
    training = {"status": "running", "history": [], "completed_generations": 0, "evaluated_episodes": 0,
                "control_steps": 0, "cache_hits": 0, "active_candidates": [], "failure": None}
    env = None
    try:
        env = SharedExerciseTaskEnv(scene, task)
        validator = prepare_shared_exercise_snapshot_validator(scene, task)
        shape = (4 if env.action_count == 22 else 3, 3)
        rng, mean, std = np.random.default_rng(seed), np.zeros(shape), np.full(shape, .9)
        best, best_score, cache = np.zeros(shape), -math.inf, {}
        authored_score = None
        for generation in range(generations):
            samples = np.clip(rng.normal(mean, std, (population, *shape)), -3, 3)
            samples[0], samples[1] = 0., 1.
            if generation:
                samples[2] = best
            rows = []
            training["active_candidates"] = rows
            for index, weights in enumerate(samples):
                _check(deadline)
                key = canonical_hash(weights.tolist())
                row = {"candidate": index, "weights": weights.tolist(), "trials": [], "complete": False}
                rows.append(row)
                if key in cache:
                    row.update(deepcopy(cache[key]), cached=True)
                    row["candidate"] = index
                    training["cache_hits"] += 1
                else:
                    for trial_seed, reset in zip(training_seeds, TRAIN_RESETS):
                        try:
                            trial = _trial(env, "learned", trial_seed, reset, weights, deadline, validator=validator)
                        except Exception as error:
                            if hasattr(error, "partial_trial"):
                                row["trials"].append(error.partial_trial)
                                training["control_steps"] += error.partial_trial["control_steps"]
                            raise
                        row["trials"].append(trial)
                        training["evaluated_episodes"] += 1
                        training["control_steps"] += trial["control_steps"]
                    score = _score(row["trials"])
                    row.update(score=score, loss=-score, complete=True, cached=False)
                    cache[key] = deepcopy(row)
                if index == 1:
                    authored_score = row["score"]
                if row["score"] > best_score:
                    best, best_score = weights.copy(), row["score"]
            unique = {canonical_hash(row["weights"]): row for row in rows}
            elites = sorted(unique.values(), key=lambda row: -row["score"])[:settings["elite_count"]]
            elite_weights = np.asarray([row["weights"] for row in elites])
            mean, std = elite_weights.mean(axis=0), np.maximum(elite_weights.std(axis=0), .08)
            training["history"].append({"generation": generation, "candidates": rows,
                "best_training_score": best_score, "next_mean": mean.tolist(), "next_std": std.tolist()})
            training.update(active_candidates=[], completed_generations=generation + 1)
            policy = _policy(scene, task, env, best, seed, best_score, hashes)
            validate_shared_exercise_policy(policy, scene, task)
            _write(out, "policy.json", policy)
            _write(out, "training.json", training)
        if _runtime_hashes(scene) != hashes:
            raise RuntimeError("Exercise policy/evaluator runtime changed during training")
        _check(deadline)
        training.update(status="trained", seconds=time.monotonic() - started, policy_sha256=digest(out / "policy.json"),
            selected_authored_baseline=bool(np.array_equal(best, np.ones(shape))),
            selected_coast_baseline=bool(np.array_equal(best, np.zeros(shape))),
            improvement_over_authored_training=best_score - authored_score,
            training_score_improved_over_authored=bool(best_score > authored_score),
            selected_feedback_kind="authored_unchanged" if np.array_equal(best, np.ones(shape)) else
                "coast_unchanged" if np.array_equal(best, np.zeros(shape)) else "searched_gains")
        _write(out, "training.json", training)
        return {"directory": str(out), "manifest_sha256": _seal(out, "trained"), **training}
    except Exception as error:
        training.update(status="failed", seconds=time.monotonic() - started,
                        failure=f"{type(error).__name__}: {error}"[:2000])
        _failure(out, "training.json", training, error)
        raise
    finally:
        if env is not None:
            env.close()


def evaluate_shared_exercise_policy(learning_dir, out, *, seed=924017, trials=3, jitter_scale=.10, max_seconds=30):
    """One independent changed-start comparison; policy bytes are frozen before any reset is observed."""
    _integer(seed, "evaluation seed", 0, 2**31 - 1)
    _integer(trials, "evaluation trials", 1, 3)
    _seconds(max_seconds, 30)
    if type(jitter_scale) not in (int, float) or not math.isfinite(jitter_scale) or not 0 < jitter_scale <= .25:
        raise ValueError("Changed-start evaluation jitter_scale must be explicit in (0,.25]")
    source, out = Path(learning_dir).resolve(), _output_path(learning_dir, out)
    started, deadline = time.monotonic(), time.monotonic() + max_seconds
    manifest, files = _verified_inputs(source, deadline=deadline)
    if _json(manifest).get("status") != "trained" or not {"policy.json", "training.json", "settings.json"} <= set(files):
        raise ValueError("Evaluation requires a completed sealed training run, never a failed checkpoint")
    scene_raw, task_raw, policy_raw = files["scene.json"], files["exercise-task.json"], files["policy.json"]
    scene, task, _ = _context(_json(scene_raw), _json(task_raw))
    hashes = _runtime_hashes(scene)
    policy = validate_shared_exercise_policy(_json(policy_raw), scene, task)
    settings, training = _json(files["settings.json"]), _json(files["training.json"])
    if (training.get("status") != "trained" or training.get("policy_sha256") != _sha(policy_raw)
            or settings.get("training_seeds") is None or seed == settings.get("seed")):
        raise ValueError("Evaluation must use the frozen selected policy and independent seed stream")
    maximum_steps = len(CONTROLLERS) * trials * task["interval"]["horizon_steps"]
    if maximum_steps > MAX_EVALUATION_STEPS:
        raise ValueError("Evaluation exceeds its full-interval control-step bound")
    seeds = _seed_plan(seed, trials, settings["training_seeds"])
    train_states = {trial["initial_state_sha256"] for generation in training["history"]
                    for row in generation["candidates"] for trial in row["trials"] if trial["initial_state_sha256"] is not None}
    proof = collect_shared_source_files(scene, source, expected_manifest_sha256=_sha(manifest))
    _check(deadline)
    _package(out, scene_raw, task_raw, manifest, proof,
             extra={**_native_source_views(source, manifest, scene), "policy.json": policy_raw})
    reset = human_jitter_options(float(jitter_scale))
    evaluation = {"schema_version": 1, "status": "running", "created_at": local_time(), "policy_sha256": _sha(policy_raw),
        "scene_sha256": canonical_hash(scene), "task_sha256": canonical_hash(task), "runtime_sha256": hashes,
        "seed": seed, "evaluation_seeds": seeds, "training_seeds": settings["training_seeds"], "reset_options": reset,
        "horizon_steps": task["interval"]["horizon_steps"], "maximum_control_steps": maximum_steps,
        "max_seconds": max_seconds, "workers": 1, "external_api_usd": 0, "control_steps": 0,
        "evaluation_feedback_to_optimizer": False, "source_scope": task["source_scope"],
        "scope": SCOPE, "controllers": {}, "initial_states": {}}
    _write(out, "settings.json", {key: value for key, value in evaluation.items()
                                  if key not in ("status", "control_steps", "controllers", "initial_states")})
    env = None
    try:
        env = SharedExerciseTaskEnv(scene, task)
        validator = prepare_shared_exercise_snapshot_validator(scene, task)
        weights = np.asarray(policy["weights"])
        for controller in CONTROLLERS:
            rows = []
            evaluation["controllers"][controller] = {"trials": rows}
            for trial_seed in seeds:
                try:
                    row = _trial(env, controller, trial_seed, reset, weights, deadline, trace=True, validator=validator)
                except Exception as error:
                    if hasattr(error, "partial_trial"):
                        rows.append(error.partial_trial)
                        evaluation["control_steps"] += error.partial_trial["control_steps"]
                    raise
                rows.append(row)
                evaluation["control_steps"] += row["control_steps"]
                initial_hash = row["initial_state_sha256"]
                if initial_hash is None:
                    continue
                if initial_hash in train_states:
                    raise ValueError("Changed-start evaluation repeated a training state; do not substitute a new seed")
                previous = evaluation["initial_states"].setdefault(str(trial_seed), initial_hash)
                if previous != initial_hash:
                    raise RuntimeError("Matched controllers did not receive identical seeded initial states")
            if len(set(evaluation["initial_states"].values())) != len(evaluation["initial_states"]):
                raise ValueError("Evaluation seeds did not produce distinct changed initial states")
            evaluation["controllers"][controller].update(successes=sum(bool(row["metrics"] and row["metrics"]["success"]) for row in rows),
                completed_intervals=sum(bool(row["metrics"] and row["metrics"]["completed_interval"]) for row in rows),
                reset_failures=sum(row["status"] == "reset_failed" for row in rows), mean_score=_score(rows))
            _write(out, "evaluation.json", evaluation)
        read_package_file(source, "policy.json", max_bytes=64 * 1024, expected_sha256=_sha(policy_raw))
        if _runtime_hashes(scene) != hashes:
            raise RuntimeError("Frozen policy or evaluator runtime changed during evaluation")
        _check(deadline)
        evaluation.update(status="evaluated", seconds=time.monotonic() - started,
                          independent_changed_states=len(evaluation["initial_states"]), policy_unchanged=True,
                          comparison_complete=all(row["initial_state_sha256"] is not None
                              for value in evaluation["controllers"].values() for row in value["trials"]),
                          physical_validation="unverified")
        _write(out, "evaluation.json", evaluation)
        return {"directory": str(out), "manifest_sha256": _seal(out, "evaluated"), **evaluation}
    except Exception as error:
        evaluation.update(status="failed", seconds=time.monotonic() - started, error=f"{type(error).__name__}: {error}"[:2000])
        _failure(out, "evaluation.json", evaluation, error)
        raise
    finally:
        if env is not None:
            env.close()
