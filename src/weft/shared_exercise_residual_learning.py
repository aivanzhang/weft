"""Learn bounded joint-action residuals without changing reference feedback, task gates or equipment."""

from copy import deepcopy
import math
from pathlib import Path
import time

import numpy as np

from weft.io import canonical_hash, digest, local_time
from weft.shared_exercise_lab import NATIVE_SOURCE_VIEWS, _json, _runtime_versions, _verified_inputs, native_source_views
from weft.shared_exercise_learning import (
    POLICY_RUNTIME_FILES as BASE_RUNTIME_FILES,
    _bytes, _context, _integer, _output_path, _package, _score, _seconds, _seed_plan, _sha, _source, _write,
)
from weft.shared_exercise_reset import human_jitter_options
from weft.shared_exercise_task import GATES_SHA256, SharedExerciseTaskEnv, prepare_shared_exercise_snapshot_validator
from weft.shared_world import FULLSPAN_PROFILES, HUMAN_ONLY_PROFILE, STATIC_SUPPORT_PROFILE
from weft.shared_world_packaging import collect_shared_source_files, read_package_file


POLICY_PROFILE = "shared-exercise-policy-v2"
ALGORITHM = "shared_exercise_phase_residual_cem_v1"
SUPPORT_POLICY_PROFILE = "shared-support-policy-v1"
SUPPORT_ALGORITHM = "shared_support_phase_residual_cem_v1"
BASIS_PROFILE = "shared-exercise-phase-basis-v1"
FEATURE_NAMES = ("authored_reference_feedback", "declared_interval_elapsed_fraction")
POLICY_RUNTIME_FILES = (*BASE_RUNTIME_FILES, "shared_exercise_residual_learning.py", "shared_exercise_policies.py")
RESIDUAL_BOUND = .25
MAX_CONTROL_STEPS = 8_640
MAX_EVALUATION_STEPS = 720
MAX_FULLSPAN_EVALUATION_STEPS = 21_600
MAX_OUTPUT_BYTES = 128 * 1024 * 1024
MAX_OUTPUT_FILES = 512
TRAIN_RESETS = (human_jitter_options(0.), human_jitter_options(.05))
CONTROLLERS = ("coast", "random", "authored", "learned")
SOURCE_VIEWS = frozenset({"source.json", "source.mp4", "source.webm", "source.mov", "source.m4v", "source.avi", "source.mkv"})
SCOPE = ("Reward-driven search over bounded phase-conditioned joint-action residuals on unchanged authored reference feedback. "
         "Reference targets, simulator bias forces and simulated state are privileged features; root assistance remains "
         "the task's explicit authored feedback and equipment forces remain forbidden. Zero residual selects authored "
         "unchanged, not learned improvement. Changed starts in the full fixed development interval are not new videos, "
         "a temporal holdout, recovered grasp, measured muscle activation or physical transfer.")
SUPPORT_SCOPE = ("Reward-driven search over bounded phase-conditioned joint-action residuals on unchanged authored reference feedback. "
                 "Reference targets, simulator bias forces and simulated state are privileged features; any explicit root assistance "
                 "remains unchanged and object-force channels remain forbidden. Static support geometry, declared load requirements, "
                 "task gates and reset rules stay fixed. Zero residual selects authored unchanged, not learned improvement. "
                 "Changed-start development evaluation is not a new video, a temporal holdout, recovered contact or grasp, "
                 "measured muscle activation, unassisted human balance or physical transfer.")


def _check(deadline):
    if time.monotonic() >= deadline:
        raise TimeoutError("Exercise residual learning/evaluation exceeded its cooperative wall-time bound")


def _runtime_hashes(scene=None):
    package = Path(__file__).parent
    names = POLICY_RUNTIME_FILES
    if scene is not None and scene["profile"] == HUMAN_ONLY_PROFILE:
        names = (*names, "person_inventory.py")
    elif scene is not None and scene["profile"] == STATIC_SUPPORT_PROFILE:
        names = (*names, "shared_support_task.py")
    if scene is not None and "native_clock" in scene.get("human_spec", {}):
        names = (*names, "source_clock.py")
    return {name: digest(package / name) for name in names}


def _scene_runtime_hashes(scene):
    return _runtime_hashes(scene) if (scene["profile"] in {HUMAN_ONLY_PROFILE, STATIC_SUPPORT_PROFILE}
                                     or "native_clock" in scene.get("human_spec", {})) else _runtime_hashes()


def _contract(scene, hashes):
    if scene["profile"] == STATIC_SUPPORT_PROFILE:
        from weft.shared_support_task import GATES_SHA256 as support_gates, evaluator_identity

        evaluator = evaluator_identity({name: hashes[name] for name in ("shared_exercise_task.py", "shared_support_task.py")})
        return {"profile": SUPPORT_POLICY_PROFILE, "algorithm": SUPPORT_ALGORITHM, "gates_sha256": support_gates,
                "evaluator_sha256": evaluator, "scope": SUPPORT_SCOPE, "observation_profile": "shared-support-observation-v1"}
    return {"profile": POLICY_PROFILE, "algorithm": ALGORITHM, "gates_sha256": GATES_SHA256,
            "evaluator_sha256": hashes["shared_exercise_task.py"], "scope": SCOPE,
            "observation_profile": "shared-exercise-observation-v1"}


def _coefficients(value):
    if (not isinstance(value, list) or len(value) != 2 or any(not isinstance(row, list) or len(row) != 19
            or any(type(item) not in (int, float) or not math.isfinite(item) or abs(item) > RESIDUAL_BOUND
                   for item in row) for row in value)):
        raise ValueError("Exercise residual coefficients require finite numeric 2-by-19 values in [-.25,.25]")
    result = np.asarray(value, dtype=float)
    result.flags.writeable = False
    return result


def phase_basis(phase):
    """Two fixed nonnegative weights sum to at most one, preserving the residual bound."""
    if type(phase) not in (int, float) or not math.isfinite(phase) or not 0 <= phase <= 1:
        raise ValueError("Residual phase must be the finite declared interval fraction in [0,1]")
    return np.asarray((max(0., 1. - abs(2. * phase - 1.)), max(0., 2. * phase - 1.)))


def _action(coefficients, env):
    authored = env.authored_action()
    # Preserve exact authored bytes, including signed zero, for the anchor and root channels.
    if not np.any(coefficients):
        return authored.copy()
    phase = float(env.observation()[-1])
    action = authored.copy()
    action[:19] = np.clip(authored[:19] + phase_basis(phase) @ coefficients, -1., 1.)
    return action


def validate_shared_exercise_residual_policy(policy, scene, task):
    support = isinstance(scene, dict) and scene.get("profile") == STATIC_SUPPORT_PROFILE
    support_policy = isinstance(policy, dict) and policy.get("profile") == SUPPORT_POLICY_PROFILE
    if support != support_policy:
        raise ValueError("Static-support learning requires a separately qualified support policy")
    scene, task, count = _context(scene, task, allow_static_support=True)
    hashes = _scene_runtime_hashes(scene)
    contract = _contract(scene, hashes)
    keys = {"schema_version", "profile", "algorithm", "scene_sha256", "task_sha256", "evaluator_sha256",
            "gates_sha256", "runtime_sha256", "runtime_versions", "action_names", "feature_names",
            "residual_coefficients", "residual_bound", "basis_profile", "root_assistance", "training_seed",
            "training_score", "privileged_simulator_features", "scope"}
    if (not isinstance(policy, dict) or set(policy) != keys or type(policy["schema_version"]) is not int
            or policy["schema_version"] != 1 or policy["profile"] != contract["profile"] or policy["algorithm"] != contract["algorithm"]
            or len(_bytes(policy)) > 64 * 1024):
        raise ValueError("Exercise residual policy differs from its bounded versioned contract")
    _coefficients(policy["residual_coefficients"])
    if (policy["scene_sha256"] != canonical_hash(scene) or policy["task_sha256"] != canonical_hash(task)
            or policy["gates_sha256"] != contract["gates_sha256"] or policy["runtime_sha256"] != hashes
            or policy["evaluator_sha256"] != contract["evaluator_sha256"]
            or policy["runtime_versions"] != _runtime_versions()):
        raise ValueError("Exercise residual policy task, scene, evaluator or runtime differs")
    if (policy["action_names"] != scene["action_contract"]["names"][:count]
            or policy["feature_names"] != list(FEATURE_NAMES) or policy["basis_profile"] != BASIS_PROFILE
            or type(policy["residual_bound"]) not in (int, float) or policy["residual_bound"] != RESIDUAL_BOUND
            or policy["root_assistance"] != task["control"]["root_assistance"]
            or policy["privileged_simulator_features"] is not True or policy["scope"] != contract["scope"]):
        raise ValueError("Exercise residual policy cannot add forces, change its basis or hide privileged features")
    _integer(policy["training_seed"], "training seed", 0, 2**31 - 1)
    if type(policy["training_score"]) not in (int, float) or not math.isfinite(policy["training_score"]):
        raise ValueError("Exercise residual policy training score must be finite")
    return deepcopy(policy)


def prepare_shared_exercise_residual_policy(policy, env):
    """Validate and detach once; the action path only reads current feedback and public phase."""
    policy = validate_shared_exercise_residual_policy(policy, env.scene, env.task)
    if (policy["evaluator_sha256"] != env.evaluator_sha256
            or env.observation_contract()["profile"] != _contract(env.scene, policy["runtime_sha256"])["observation_profile"]):
        raise ValueError("Exercise environment evaluator or observation contract differs from its residual policy")
    coefficients = _coefficients(policy["residual_coefficients"])
    return lambda: _action(coefficients, env)


def policy_action(policy, env):
    return prepare_shared_exercise_residual_policy(policy, env)()


def _policy(env, coefficients, seed, score, hashes):
    contract = _contract(env.scene, hashes)
    return {"schema_version": 1, "profile": contract["profile"], "algorithm": contract["algorithm"],
            "scene_sha256": env.scene_sha256, "task_sha256": env.task_sha256,
            "evaluator_sha256": env.evaluator_sha256, "gates_sha256": contract["gates_sha256"],
            "runtime_sha256": hashes, "runtime_versions": _runtime_versions(),
            "action_names": list(env.action_names), "feature_names": list(FEATURE_NAMES),
            "residual_coefficients": coefficients.tolist(), "residual_bound": RESIDUAL_BOUND,
            "basis_profile": BASIS_PROFILE, "root_assistance": env.task["control"]["root_assistance"],
            "training_seed": seed, "training_score": score, "privileged_simulator_features": True, "scope": contract["scope"]}


def _terminal_manifest(raw):
    document = _json(raw)
    if "status" in document and (not isinstance(document["status"], str)
                                 or document["status"] not in {"ready", "trained", "evaluated"}):
        raise ValueError("Residual learning requires a supported terminal source or historical absent status")


def source_view_names(scene=None):
    names = SOURCE_VIEWS
    if scene is not None and "native_clock" in scene.get("human_spec", {}):
        names |= NATIVE_SOURCE_VIEWS
    if scene is not None and scene["profile"] == HUMAN_ONLY_PROFILE:
        return names | {"inventory.json", "source-clock.json"}
    if scene is not None and scene["profile"] == STATIC_SUPPORT_PROFILE:
        return names | {"support-config.json", "inputs/support-config.json", "source-clock.json"}
    return names


def _source_views(source, manifest, scene=None):
    hashes = _json(manifest)["files"]
    names = sorted(source_view_names(scene) & set(hashes))
    if len([name for name in names if name in SOURCE_VIEWS and name != "source.json"]) > 1:
        raise ValueError("Residual learning source contains ambiguous original video views")
    files = {name: read_package_file(source, name, max_bytes=64 * 1024 * 1024,
                                    expected_sha256=hashes[name]) for name in names}
    if {"support-config.json", "inputs/support-config.json"} <= files.keys() and files["support-config.json"] != files["inputs/support-config.json"]:
        raise ValueError("Static-support configuration views differ")
    if scene is not None:
        native_source_views(files, scene)
    return files


def _input_unchanged(source, manifest, deadline, *, task_path=None, task_raw=None):
    read_package_file(source, "manifest.json", max_bytes=8 * 1024 * 1024, expected_sha256=_sha(manifest))
    for name, expected in _json(manifest)["files"].items():
        _check(deadline)
        read_package_file(source, name, max_bytes=64 * 1024 * 1024, expected_sha256=expected)
    if task_path is not None:
        path = Path(task_path).absolute()
        read_package_file(path.parent, path.name, max_bytes=64 * 1024, expected_sha256=_sha(task_raw))
    _check(deadline)


def _seal(out, status, deadline=None, *, maximum_bytes=None, maximum_files=None):
    maximum_bytes = MAX_OUTPUT_BYTES if maximum_bytes is None else maximum_bytes
    maximum_files = MAX_OUTPUT_FILES if maximum_files is None else maximum_files
    paths, size = [], 0
    for path in out.rglob("*"):
        if deadline is not None:
            _check(deadline)
        if path.is_symlink():
            raise ValueError("Residual output cannot seal symbolic links")
        if not path.is_file() or path == out / "manifest.json":
            continue
        paths.append(path)
        size += path.stat().st_size
        if len(paths) >= maximum_files or size > maximum_bytes:
            raise ValueError("Residual output exceeds its file/byte bound; partial evidence preserved")
    files = {}
    for path in sorted(paths):
        if deadline is not None:
            _check(deadline)
        files[path.relative_to(out).as_posix()] = digest(path)
    manifest = {"schema_version": 1, "status": status, "created_at": local_time(), "files": files}
    if size + len(_bytes(manifest)) > maximum_bytes:
        raise ValueError("Residual output manifest exceeds the aggregate byte bound")
    if deadline is not None:
        _check(deadline)
    _write(out, "manifest.json", manifest)
    if deadline is not None:
        _check(deadline)
    return digest(out / "manifest.json")


def _failure(out, state_name, state, error):
    state.update(status="failed", error=f"{type(error).__name__}: {error}"[:2000])
    failure = {"status": "failed", "error": state["error"], "partial_evidence_preserved": True}
    _write(out, "failure.json", failure)
    try:
        _write(out, state_name, state)
        _seal(out, "failed")
    except (ValueError, OSError) as preservation_error:
        _write(out, "failure.json", failure | {"checkpoint_error": str(preservation_error)[:2000],
                                               "unsealed_outputs_preserved": True})
        _write(out, "manifest.json", {"schema_version": 1, "status": "failed", "custody": "failure_receipt_only",
            "files": {"failure.json": digest(out / "failure.json")}, "unsealed_outputs_preserved": True})


def _initialize(out, scene_raw, task_raw, manifest, proof, extra):
    proof.validate_layout(reserved_paths=set(extra))
    try:
        return _package(out, scene_raw, task_raw, manifest, proof, extra=extra)
    except Exception as error:
        if out.exists():
            _failure(out, "failure.json", {"stage": "input_packaging"}, error)
        raise


class _Budget:
    def __init__(self, maximum):
        self.maximum, self.used = maximum, 0

    def reserve(self):
        if self.used >= self.maximum:
            raise ValueError("Residual control-action budget exhausted; no retries or hidden baseline actions")
        self.used += 1


def _trial(env, controller, seed, reset, coefficients, deadline, budget, *, trace=False, validator=None):
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
        rng = np.random.default_rng(seed ^ 0x5A17)
        validate = validator or prepare_shared_exercise_snapshot_validator(env.scene, env.task)
        for _ in range(env.horizon_steps):
            _check(deadline)
            action = (np.zeros(env.action_count) if controller == "coast" else
                      rng.uniform(-1., 1., env.action_count) if controller == "random" else
                      env.authored_action() if controller == "authored" else _action(coefficients, env))
            budget.reserve()
            row["control_steps"] += 1
            _, reward, terminated, truncated, _ = env.step(action)
            row["return"] += float(reward)
            if trace:
                snapshot = env.snapshot()
                validate(snapshot)
                row["actions"].append({"action": action.tolist(), "reward": float(reward), "terminated": terminated,
                                       "truncated": truncated, "snapshot_sha256": canonical_hash(snapshot)})
            if terminated or truncated:
                break
        metrics = env.metrics()
        if (not (env.terminated or env.truncated) or not (metrics["completed_interval"] or metrics["failure"] is not None)
                or metrics["completed_interval"] and row["control_steps"] != env.horizon_steps):
            raise ValueError("Residual trial must attempt the whole interval or preserve an explicit terminal failure")
        row.update(status="completed" if metrics["completed_interval"] else "failed", metrics=metrics,
                   terminated=env.terminated, truncated=env.truncated)
        final = env.snapshot()
        validate(final)
        row["final_snapshot_sha256"] = canonical_hash(final)
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


def train_shared_exercise_residual_policy(run, out, *, task_path=None, root_assistance=False,
                                        seed=1729, generations=6, population=12, max_seconds=60):
    """Search only joint residuals on two fixed starts; every anchor is inside the action budget."""
    _integer(seed, "training seed", 0, 2**31 - 1)
    _integer(generations, "generations", 1, 6)
    _integer(population, "population", 4, 12)
    _seconds(max_seconds, 60)
    out = _output_path(run, out)
    started = time.monotonic()
    deadline = started + max_seconds
    source, scene, task, scene_raw, task_raw, manifest, proof = _source(
        run, task_path, root_assistance, deadline, allow_static_support=True)
    _terminal_manifest(manifest)
    maximum_steps = generations * population * len(TRAIN_RESETS) * task["interval"]["horizon_steps"]
    if maximum_steps > MAX_CONTROL_STEPS:
        raise ValueError("Residual training exceeds 8640 actions; do not shorten the fixed task interval")
    hashes = _scene_runtime_hashes(scene)
    contract = _contract(scene, hashes)
    views = _source_views(source, manifest, scene)
    _check(deadline)
    _initialize(out, scene_raw, task_raw, manifest, proof, views)
    training_seeds = _seed_plan(seed, 2)
    settings = {"schema_version": 1, "algorithm": contract["algorithm"], "created_at": local_time(), "seed": seed,
        "scene_sha256": canonical_hash(scene), "task_sha256": canonical_hash(task), "gates_sha256": contract["gates_sha256"],
        "runtime_sha256": hashes, "source_manifest_sha256": _sha(manifest), "training_seeds": training_seeds,
        "source_views_sha256": {name: _sha(raw) for name, raw in views.items()},
        "training_reset_options": list(TRAIN_RESETS), "generations": generations, "population": population,
        "elite_count": max(2, population // 4), "initial_mean": 0., "initial_std": .08, "minimum_std": .02,
        "residual_bound": RESIDUAL_BOUND, "coefficient_shape": [2, 19], "basis_profile": BASIS_PROFILE,
        "maximum_control_steps": maximum_steps, "maximum_physics_substeps": maximum_steps * scene["substeps"],
        "max_seconds": max_seconds, "workers": 1, "external_api_usd": 0, "horizon_steps": task["interval"]["horizon_steps"],
        "selection_objective": "mean(task mean_reward -1 for incomplete intervals); reset failure score=-2; loss=-score",
        "candidate_anchors": ["zero_residual_authored", "previous_best_after_first_generation"],
        "action_accounting": "All anchors and candidates are in population; identical cached trials issue zero new actions; no retries",
        "root_residual_channels": [], "equipment_force_channels": [], "source_scope": task["source_scope"], "scope": contract["scope"]}
    training = {"status": "running", "history": [], "completed_generations": 0, "evaluated_episodes": 0,
                "control_steps": 0, "cache_hits": 0, "active_candidates": [], "failure": None}
    env, budget = None, _Budget(maximum_steps)
    try:
        _write(out, "settings.json", settings)
        env = SharedExerciseTaskEnv(scene, task)
        validator = prepare_shared_exercise_snapshot_validator(scene, task)
        rng = np.random.default_rng(seed)
        mean, std = np.zeros((2, 19)), np.full((2, 19), settings["initial_std"])
        best, best_score, cache, authored_score = mean.copy(), -math.inf, {}, None
        for generation in range(generations):
            samples = np.clip(rng.normal(mean, std, (population, 2, 19)), -RESIDUAL_BOUND, RESIDUAL_BOUND)
            samples[0] = 0.
            if generation:
                samples[1] = best
            rows = []
            training["active_candidates"] = rows
            for index, coefficients in enumerate(samples):
                _check(deadline)
                key = canonical_hash(coefficients.tolist())
                row = {"candidate": index, "residual_coefficients": coefficients.tolist(), "trials": [], "complete": False}
                rows.append(row)
                if key in cache:
                    row.update(deepcopy(cache[key]), candidate=index, cached=True)
                    training["cache_hits"] += 1
                else:
                    for trial_seed, reset in zip(training_seeds, TRAIN_RESETS, strict=True):
                        try:
                            trial = _trial(env, "learned", trial_seed, reset, coefficients, deadline, budget, validator=validator)
                        except Exception as error:
                            if hasattr(error, "partial_trial"):
                                row["trials"].append(error.partial_trial)
                            raise
                        row["trials"].append(trial)
                        training["evaluated_episodes"] += 1
                        training["control_steps"] = budget.used
                    score = _score(row["trials"])
                    row.update(score=score, loss=-score, complete=True, cached=False)
                    cache[key] = deepcopy(row)
                if index == 0:
                    authored_score = row["score"]
                if row["score"] > best_score:
                    best, best_score = coefficients.copy(), row["score"]
            unique = {canonical_hash(row["residual_coefficients"]): row for row in rows}
            elites = sorted(unique.values(), key=lambda row: -row["score"])[:settings["elite_count"]]
            elite_coefficients = np.asarray([row["residual_coefficients"] for row in elites])
            mean, std = elite_coefficients.mean(axis=0), np.maximum(elite_coefficients.std(axis=0), settings["minimum_std"])
            training["history"].append({"generation": generation, "candidates": rows,
                "best_training_score": best_score, "next_mean": mean.tolist(), "next_std": std.tolist()})
            training.update(active_candidates=[], completed_generations=generation + 1, control_steps=budget.used)
            policy = _policy(env, best, seed, best_score, hashes)
            validate_shared_exercise_residual_policy(policy, scene, task)
            _write(out, "policy.json", policy)
            _write(out, "training.json", training)
        _input_unchanged(source, manifest, deadline, task_path=task_path, task_raw=task_raw)
        if _scene_runtime_hashes(scene) != hashes:
            raise RuntimeError("Residual policy/evaluator runtime changed during training")
        _check(deadline)
        authored = not bool(np.any(best))
        training.update(status="trained", seconds=time.monotonic() - started, policy_sha256=digest(out / "policy.json"),
            selected_authored_baseline=authored, selected_coast_baseline=False,
            improvement_over_authored_training=best_score - authored_score,
            training_score_improved_over_authored=bool(best_score > authored_score),
            selected_feedback_kind="authored_unchanged" if authored else "searched_joint_residuals")
        _write(out, "training.json", training)
        return {"directory": str(out), "manifest_sha256": _seal(out, "trained", deadline), **training}
    except Exception as error:
        training.update(control_steps=budget.used, seconds=time.monotonic() - started,
                        failure=f"{type(error).__name__}: {error}"[:2000])
        _failure(out, "training.json", training, error)
        raise
    finally:
        if env is not None:
            env.close()


def _excluded_seeds(values):
    if not isinstance(values, (tuple, list)) or len(values) > 128:
        raise ValueError("Evaluation seed exclusions require a bounded list of at most 128 seeds")
    for seed in values:
        _integer(seed, "excluded seed", 0, 2**31 - 1)
    return sorted(set(values))


def evaluate_shared_exercise_residual_policy(learning_dir, out, *, seed=924017, trials=3,
                                           jitter_scale=.10, max_seconds=30, excluded_seeds=()):
    """Freeze selected policy bytes and fresh reset seeds before any comparison state is observed."""
    _integer(seed, "evaluation seed", 0, 2**31 - 1)
    _integer(trials, "evaluation trials", 1, 3)
    _seconds(max_seconds, 60)
    excluded = _excluded_seeds(excluded_seeds)
    if type(jitter_scale) not in (int, float) or not math.isfinite(jitter_scale) or not 0 < jitter_scale <= .25:
        raise ValueError("Changed-start evaluation jitter_scale must be explicit in (0,.25]")
    source, out = Path(learning_dir).resolve(), _output_path(learning_dir, out)
    started = time.monotonic()
    deadline = started + max_seconds
    manifest, files = _verified_inputs(source, deadline=deadline)
    required = {"policy.json", "training.json", "settings.json", "scene.json", "exercise-task.json"}
    if _json(manifest).get("status") != "trained" or not required <= set(files):
        raise ValueError("Residual evaluation requires completed sealed training, never a failed checkpoint")
    scene_raw, task_raw, policy_raw = files["scene.json"], files["exercise-task.json"], files["policy.json"]
    scene, task, _ = _context(_json(scene_raw), _json(task_raw), allow_static_support=True)
    fullspan = scene["profile"] in FULLSPAN_PROFILES
    _seconds(max_seconds, 60 if fullspan else 30)
    hashes = _scene_runtime_hashes(scene)
    contract = _contract(scene, hashes)
    policy = validate_shared_exercise_residual_policy(_json(policy_raw), scene, task)
    settings, training = _json(files["settings.json"]), _json(files["training.json"])
    training_seeds = settings.get("training_seeds")
    if (training.get("status") != "trained" or training.get("policy_sha256") != _sha(policy_raw)
            or not isinstance(training_seeds, list) or len(training_seeds) != 2
            or any(type(value) is not int or not 0 <= value < 2**31 for value in training_seeds)
            or len(set(training_seeds)) != 2 or seed == settings.get("seed")
            or any(settings.get(name) != policy[name] for name in ("scene_sha256", "task_sha256", "gates_sha256", "runtime_sha256"))):
        raise ValueError("Residual evaluation requires a frozen policy and independent seed stream")
    maximum_steps = len(CONTROLLERS) * trials * task["interval"]["horizon_steps"]
    maximum_allowed = MAX_FULLSPAN_EVALUATION_STEPS if fullspan else MAX_EVALUATION_STEPS
    if maximum_steps > maximum_allowed:
        raise ValueError(f"Residual evaluation exceeds {maximum_allowed} full-interval actions")
    seeds = _seed_plan(seed, trials, [*training_seeds, *excluded])
    train_states = {trial["initial_state_sha256"] for generation in training["history"]
                    for row in generation["candidates"] for trial in row["trials"] if trial["initial_state_sha256"] is not None}
    proof = collect_shared_source_files(scene, source, expected_manifest_sha256=_sha(manifest))
    views = _source_views(source, manifest, scene)
    reset = human_jitter_options(float(jitter_scale))
    evaluation = {"schema_version": 1, "status": "running", "created_at": local_time(), "policy_sha256": _sha(policy_raw),
        "scene_sha256": canonical_hash(scene), "task_sha256": canonical_hash(task), "runtime_sha256": hashes,
        "source_views_sha256": {name: _sha(raw) for name, raw in views.items()},
        "seed": seed, "evaluation_seeds": seeds, "training_seeds": training_seeds, "excluded_seeds": excluded,
        "reset_options": reset, "horizon_steps": task["interval"]["horizon_steps"], "maximum_control_steps": maximum_steps,
        "max_seconds": max_seconds, "workers": 1, "external_api_usd": 0, "control_steps": 0,
        "evaluation_feedback_to_optimizer": False, "source_scope": task["source_scope"], "scope": contract["scope"],
        "controllers": {}, "initial_states": {}}
    if fullspan:
        evaluation["evaluation_protocol"] = "shared-exercise-full-reference-evaluation-v1"
    freeze = {key: value for key, value in evaluation.items()
              if key not in ("status", "control_steps", "controllers", "initial_states")}
    _check(deadline)
    _initialize(out, scene_raw, task_raw, manifest, proof,
                {**views, "policy.json": policy_raw, "pre-evaluation-freeze.json": _bytes(freeze)})
    env, budget = None, _Budget(maximum_steps)
    try:
        _write(out, "settings.json", freeze)
        env = SharedExerciseTaskEnv(scene, task)
        validator = prepare_shared_exercise_snapshot_validator(scene, task)
        coefficients = _coefficients(policy["residual_coefficients"])
        for controller in CONTROLLERS:
            rows = []
            evaluation["controllers"][controller] = {"trials": rows}
            for trial_seed in seeds:
                try:
                    row = _trial(env, controller, trial_seed, reset, coefficients, deadline, budget,
                                 trace=True, validator=validator)
                except Exception as error:
                    if hasattr(error, "partial_trial"):
                        rows.append(error.partial_trial)
                    raise
                rows.append(row)
                evaluation["control_steps"] = budget.used
                initial_hash = row["initial_state_sha256"]
                if initial_hash is None:
                    continue
                if initial_hash in train_states:
                    raise ValueError("Residual evaluation repeated a training state; do not replace its seed")
                previous = evaluation["initial_states"].setdefault(str(trial_seed), initial_hash)
                if previous != initial_hash:
                    raise RuntimeError("Matched residual controllers received different seeded initial states")
            if len(set(evaluation["initial_states"].values())) != len(evaluation["initial_states"]):
                raise ValueError("Residual evaluation seeds did not produce distinct changed initial states")
            evaluation["controllers"][controller].update(
                successes=sum(bool(row["metrics"] and row["metrics"]["success"]) for row in rows),
                completed_intervals=sum(bool(row["metrics"] and row["metrics"]["completed_interval"]) for row in rows),
                reset_failures=sum(row["status"] == "reset_failed" for row in rows), mean_score=_score(rows))
            _write(out, "evaluation.json", evaluation)
        _input_unchanged(source, manifest, deadline)
        if _scene_runtime_hashes(scene) != hashes:
            raise RuntimeError("Frozen residual policy or evaluator runtime changed during evaluation")
        read_package_file(out, "policy.json", max_bytes=64 * 1024, expected_sha256=_sha(policy_raw))
        read_package_file(out, "pre-evaluation-freeze.json", max_bytes=64 * 1024, expected_sha256=_sha(_bytes(freeze)))
        _check(deadline)
        evaluation.update(status="evaluated", seconds=time.monotonic() - started,
            independent_changed_states=len(evaluation["initial_states"]), policy_unchanged=True,
            comparison_complete=all(row["initial_state_sha256"] is not None
                for summary in evaluation["controllers"].values() for row in summary["trials"]),
            physical_validation="unverified")
        _write(out, "evaluation.json", evaluation)
        return {"directory": str(out), "manifest_sha256": _seal(out, "evaluated", deadline), **evaluation}
    except Exception as error:
        evaluation.update(control_steps=budget.used, seconds=time.monotonic() - started)
        _failure(out, "evaluation.json", evaluation, error)
        raise
    finally:
        if env is not None:
            env.close()
