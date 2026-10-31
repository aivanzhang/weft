"""Bounded feedback learning, saved-only selection and independent changed-start evaluation."""

from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np

from weft.io import canonical_hash, digest, local_time
from weft import shared_exercise_feedback_residual as actor
from weft import shared_exercise_residual_learning as residual
from weft.shared_exercise_feasibility_learning import RULE, _rank_key, optimization_rank
from weft.shared_exercise_lab import _json, _runtime_versions, _verified_inputs
from weft.shared_exercise_learning import _bytes, _context, _integer, _output_path, _score, _seconds, _seed_plan, _sha, _source
from weft.shared_exercise_reset import human_jitter_options
from weft.shared_exercise_selection import _trial as validate_training_trial
from weft.shared_exercise_task import GATES_SHA256, SharedExerciseTaskEnv, prepare_shared_exercise_snapshot_validator
from weft.shared_world import FULLSPAN_PROFILES
from weft.shared_world_packaging import collect_shared_source_files, read_package_file, write_package_files


MAX_CONTROL_STEPS = residual.MAX_CONTROL_STEPS
MAX_EVALUATION_STEPS = residual.MAX_EVALUATION_STEPS
MAX_FULLSPAN_EVALUATION_STEPS = residual.MAX_FULLSPAN_EVALUATION_STEPS
MAX_OUTPUT_BYTES = residual.MAX_OUTPUT_BYTES
MAX_OUTPUT_FILES = residual.MAX_OUTPUT_FILES
MAX_DIAGNOSTIC_BYTES = 24 * 1024 * 1024
MAX_DIAGNOSTIC_FILE_BYTES = 8 * 1024 * 1024
MAX_DIAGNOSTIC_LINE_BYTES = 8192
METADATA_BYTES = 4 * 1024 * 1024
TERMINAL_RESERVE = 4 * 1024 * 1024
DIAGNOSTIC_PROFILE = "shared-exercise-control-boundary-feedback-v1"
SELECTION_PROFILE = "shared-exercise-feedback-selection-v1"
TRAIN_RESETS = residual.TRAIN_RESETS
CONTROLLERS = residual.CONTROLLERS


def _check(deadline):
    if time.monotonic() >= deadline:
        raise TimeoutError("Feedback learning/selection/evaluation reached its cooperative deadline")


def _encoded(value, maximum, deadline):
    raw = bytearray()
    for part in json.JSONEncoder(indent=2, allow_nan=False).iterencode(value):
        _check(deadline)
        block = part.encode()
        if len(raw) + len(block) + 1 > maximum:
            raise ValueError("Feedback metadata exceeds its encoding bound")
        raw.extend(block)
    raw.extend(b"\n")
    return bytes(raw)


class _Writer:
    def __init__(self, out, deadline):
        self.out, self.deadline = out, deadline
        self.sizes = {path.relative_to(out).as_posix(): path.stat().st_size for path in out.rglob("*") if path.is_file()}
        self.diagnostic_bytes = 0

    def _reserve(self, name, size, *, terminal=False, append=False):
        _check(self.deadline)
        reserve = 65536 if terminal else TERMINAL_RESERVE
        # Check temporary replacement bytes too; the old checkpoint remains until rename.
        if (sum(self.sizes.values()) + size + reserve > MAX_OUTPUT_BYTES
                or len(set(self.sizes) | {name}) + (0 if append else 1) >= MAX_OUTPUT_FILES):
            raise ValueError("Feedback output reservation exhausted before write")

    def write(self, name, value, *, terminal=False):
        if Path(name).name != name:
            raise ValueError("Feedback metadata must be a top-level owned file")
        raw = _encoded(value, METADATA_BYTES, self.deadline)
        self._reserve(name, len(raw), terminal=terminal)
        temporary = self.out / (name + ".tmp")
        with temporary.open("xb") as stream:
            stream.write(raw)
        os.replace(temporary, self.out / name)
        self.sizes[name] = len(raw)

    def diagnostic(self, name, value):
        raw = (json.dumps(value, separators=(",", ":"), allow_nan=False) + "\n").encode()
        if (len(raw) > MAX_DIAGNOSTIC_LINE_BYTES or self.sizes.get(name, 0) + len(raw) > MAX_DIAGNOSTIC_FILE_BYTES
                or self.diagnostic_bytes + len(raw) > MAX_DIAGNOSTIC_BYTES):
            raise ValueError("Training diagnostics exhausted their file/aggregate bound")
        self._reserve(name, len(raw), append=True)
        path = self.out / name
        if path.parent != self.out / "diagnostics":
            raise ValueError("Diagnostic escaped the owned output directory")
        path.parent.mkdir(exist_ok=True)
        with path.open("ab" if name in self.sizes else "xb") as stream:
            stream.write(raw)
        self.sizes[name] = self.sizes.get(name, 0) + len(raw)
        self.diagnostic_bytes += len(raw)

    def seal(self, status):
        files, size = {}, 0
        for path in sorted(self.out.rglob("*")):
            _check(self.deadline)
            if path.is_symlink():
                raise ValueError("Feedback output cannot seal symlinks")
            if not path.is_file() or path.name == "manifest.json" and path.parent == self.out:
                continue
            size += path.stat().st_size
            if len(files) >= MAX_OUTPUT_FILES - 2 or size > MAX_OUTPUT_BYTES - 65536:
                raise ValueError("Feedback terminal inventory exceeds its bound")
            files[path.relative_to(self.out).as_posix()] = digest(path)
        self.write("manifest.json", {"schema_version": 1, "status": status, "created_at": local_time(), "files": files}, terminal=True)
        return digest(self.out / "manifest.json")


class _Diagnostics:
    def __init__(self, writer, name):
        self.writer, self.name = writer, name
        self.attempts = self.results = 0

    def attempt(self, env, observation, action, features, authored, correction):
        if observation.ndim != 1 or len(observation) < 156 or not np.isfinite(observation).all():
            raise ValueError("Training diagnostic requires the existing bounded observation")
        self.writer.diagnostic(self.name, {"type": "attempt", "index": self.attempts + 1,
            "time_s": self.attempts * env.scene["dt"],
            "joint_position_error_rad": (observation[:19].astype(float) - observation[44:63]).tolist(),
            "feedback_features": features.tolist(), "action": action.tolist(),
            "authored_saturated_joints": np.flatnonzero(abs(authored[:19]) == 1.).tolist(),
            "residual_at_bound_joints": np.flatnonzero(abs(correction) == actor.RESIDUAL_BOUND).tolist()})
        self.attempts += 1

    def result(self, reward, terminated, truncated, info):
        self.writer.diagnostic(self.name, {"type": "result", "index": self.attempts, "reward": float(reward),
            "terminated": bool(terminated), "truncated": bool(truncated),
            **{key: info[key] for key in ("simulated_duration_s", "observed_substeps", "peak_angular_rmse_rad", "peak_root_rmse_model_units")}})
        self.results += 1

    def receipt(self):
        return {"profile": DIAGNOSTIC_PROFILE, "artifact": self.name if self.attempts else None,
            "sha256": digest(self.writer.out / self.name) if self.attempts else None,
            "attempts": self.attempts, "results": self.results,
            "scope": "Existing float32 observation errors and one pre-action feedback sample; reported peaks bound a control interval, not an exact peak substep"}


def _trial(env, controller, seed, reset, weights, deadline, budget, *, diagnostics=None, trace=False, validator=None):
    # The legacy trial hardcodes its phase actor; its AST/provenance contract remains untouched.
    row = {"seed": seed, "reset_options": deepcopy(reset), "initial_state_sha256": None,
           "status": "starting", "control_steps": 0, "return": 0., "metrics": None}
    try:
        _check(deadline)
        try:
            observation, _ = env.reset(seed=seed, options=reset)
        except (ValueError, RuntimeError) as error:
            row.update(status="reset_failed", error=f"{type(error).__name__}: {error}"[:2000])
            return row
        initial = env.snapshot()
        row.update(initial_state_sha256=canonical_hash(initial["engine_state"]), initial_metrics=env.metrics())
        if trace:
            row.update(initial_snapshot=initial, actions=[])
        rng, groups = np.random.default_rng(seed ^ 0x5A17), actor.joint_groups(env.action_names)
        validate = validator or prepare_shared_exercise_snapshot_validator(env.scene, env.task)
        for _ in range(env.horizon_steps):
            _check(deadline)
            if controller in ("authored", "learned"):
                action, features, authored, correction = actor.action_sample(np.zeros((3, 2)) if controller == "authored" else weights, groups, env)
            elif controller in ("coast", "random"):
                action = np.zeros(env.action_count) if controller == "coast" else rng.uniform(-1., 1., env.action_count)
            else:
                raise ValueError("Unknown feedback comparison controller")
            if budget.used >= budget.maximum:
                budget.reserve()
            if diagnostics is not None:
                diagnostics.attempt(env, observation, action, features, authored, correction)
            budget.reserve()
            row["control_steps"] += 1
            observation, reward, terminated, truncated, info = env.step(action)
            row["return"] += float(reward)
            if diagnostics is not None:
                diagnostics.result(reward, terminated, truncated, info)
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
            raise ValueError("Feedback trial must finish its full interval or retain an explicit terminal failure")
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
    finally:
        if diagnostics is not None:
            row["diagnostics"] = diagnostics.receipt()


def _selected(rows):
    eligible = [row for row in rows if row["optimization_rank"]["complete"]]
    if not eligible:
        raise ValueError("No complete training candidate; no feedback policy can be published")
    return min(eligible, key=lambda row: _rank_key(row["optimization_rank"]))


def _settings(scene, task, hashes, source_hash, seed, generations, population, max_seconds, views):
    maximum = generations * population * 2 * task["interval"]["horizon_steps"]
    return {"schema_version": 1, "algorithm": actor.ALGORITHM, "actor_profile": actor.ACTOR_PROFILE,
        "created_at": local_time(), "scene_sha256": canonical_hash(scene), "task_sha256": canonical_hash(task),
        "gates_sha256": GATES_SHA256, "runtime_sha256": hashes, "runtime_versions": _runtime_versions(),
        "source_manifest_sha256": source_hash, "source_views_sha256": {name: _sha(raw) for name, raw in views.items()},
        "seed": seed, "training_seeds": _seed_plan(seed, 2), "training_reset_options": list(TRAIN_RESETS),
        "generations": generations, "population": population, "elite_count": max(2, population // 4),
        "initial_mean": 0., "initial_std": .08, "minimum_std": .02,
        "coefficient_shape": [3, 2], "coefficient_bound": actor.COEFFICIENT_BOUND,
        "feature_bound": actor.FEATURE_BOUND, "residual_bound": actor.RESIDUAL_BOUND,
        "groups": list(actor.GROUP_NAMES), "feature_names": list(actor.FEATURE_NAMES),
        "candidate_anchors": ["zero_residual_authored", "previous_feasibility_best_after_first_generation"],
        "maximum_control_steps": maximum, "maximum_physics_substeps": maximum * scene["substeps"],
        "horizon_steps": task["interval"]["horizon_steps"], "max_seconds": max_seconds,
        "selection_objective": RULE, "root_residual_channels": [], "equipment_force_channels": [],
        "diagnostic_profile": DIAGNOSTIC_PROFILE, "maximum_diagnostic_bytes": MAX_DIAGNOSTIC_BYTES,
        "workers": 1, "external_api_usd": 0, "evaluation_used": False, "source_scope": task["source_scope"], "scope": actor.SCOPE}


def _failed(out, name, state, error):
    # Existing failure packaging keeps partial trials and overrides any nonterminal policy bytes.
    residual._failure(out, name, state, error)


def train_shared_exercise_feedback_policy(run, out, *, task_path=None, root_assistance=False,
                                        seed=1729, generations=2, population=8, max_seconds=60):
    _integer(seed, "training seed", 0, 2**31 - 1)
    _integer(generations, "generations", 1, 6)
    _integer(population, "population", 4, 12)
    _seconds(max_seconds, 60)
    out = _output_path(run, out)
    started, deadline = time.monotonic(), time.monotonic() + max_seconds
    source, scene, task, scene_raw, task_raw, manifest, proof = _source(run, task_path, root_assistance, deadline)
    residual._terminal_manifest(manifest)
    if any(part in {"evaluation.json", "pre-evaluation-freeze.json", "policy.json", "training.json"}
           for name in _json(manifest)["files"] for part in Path(name).parts):
        raise ValueError("Feedback training requires the original untrained construction, not prior policy/evaluation outcomes")
    maximum = generations * population * 2 * task["interval"]["horizon_steps"]
    if maximum > MAX_CONTROL_STEPS:
        raise ValueError("Feedback training exceeds 8640 actions; do not shorten the fixed task")
    hashes, views = actor.runtime_hashes(scene), residual._source_views(source, manifest, scene)
    settings = _settings(scene, task, hashes, _sha(manifest), seed, generations, population, max_seconds, views)
    residual._initialize(out, scene_raw, task_raw, manifest, proof, views)
    writer = _Writer(out, deadline)
    training = {"status": "running", "history": [], "completed_generations": 0, "evaluated_episodes": 0,
                "control_steps": 0, "cache_hits": 0, "active_candidates": [], "failure": None}
    env, budget = None, residual._Budget(maximum)
    try:
        writer.write("settings.json", settings)
        env = SharedExerciseTaskEnv(scene, task)
        validator = prepare_shared_exercise_snapshot_validator(scene, task)
        rng = np.random.default_rng(seed)
        mean, std, cache, origins = np.zeros((3, 2)), np.full((3, 2), .08), {}, {}
        selected, authored_score, initial_states = None, None, {}
        for generation in range(generations):
            samples = np.clip(rng.normal(mean, std, (population, 3, 2)), -actor.COEFFICIENT_BOUND, actor.COEFFICIENT_BOUND)
            samples[0] = 0.
            if generation:
                samples[1] = selected["feedback_coefficients"]
            rows = []
            training["active_candidates"] = rows
            for index, weights in enumerate(samples):
                _check(deadline)
                key = canonical_hash(weights.tolist())
                if key in cache:
                    row = deepcopy(cache[key]) | {"candidate": index, "cached": True}
                    training["cache_hits"] += 1
                    rows.append(row)
                else:
                    row = {"candidate": index, "feedback_coefficients": weights.tolist(), "trials": [], "complete": False, "cached": False}
                    rows.append(row)
                    for reset_index, (trial_seed, reset) in enumerate(zip(settings["training_seeds"], TRAIN_RESETS, strict=True)):
                        diag = _Diagnostics(writer, f"diagnostics/g{generation:02d}-c{index:02d}-r{reset_index}.jsonl")
                        training["evaluated_episodes"] += 1
                        try:
                            trial = _trial(env, "learned", trial_seed, reset, weights, deadline, budget, diagnostics=diag, validator=validator)
                        except Exception as error:
                            if hasattr(error, "partial_trial"):
                                row["trials"].append(error.partial_trial)
                            raise
                        row["trials"].append(trial)
                        training["control_steps"] = budget.used
                        if trial["initial_state_sha256"] is not None:
                            previous = initial_states.setdefault(trial_seed, trial["initial_state_sha256"])
                            if previous != trial["initial_state_sha256"]:
                                raise ValueError("Training candidates received different matched initial states")
                    score = _score(row["trials"])
                    row.update(score=score, loss=-score, complete=True, optimization_rank=optimization_rank(row["trials"]))
                    cache[key], origins[key] = deepcopy(row), {"generation": generation, "candidate": index}
                if index == 0:
                    authored_score = row["score"]
                writer.write("training.json", training)
            selected = _selected(list(cache.values()))
            eligible = [row for row in {canonical_hash(r["feedback_coefficients"]): r for r in rows}.values()
                        if row["optimization_rank"]["complete"]]
            elites = sorted(eligible, key=lambda row: _rank_key(row["optimization_rank"]))[:settings["elite_count"]]
            elite_weights = np.asarray([row["feedback_coefficients"] for row in elites])
            mean, std = elite_weights.mean(axis=0), np.maximum(elite_weights.std(axis=0), .02)
            training["history"].append({"generation": generation, "candidates": rows, "next_mean": mean.tolist(),
                "next_std": std.tolist(), "selected_optimization_rank": selected["optimization_rank"]})
            training.update(active_candidates=[], completed_generations=generation + 1, control_steps=budget.used)
        residual._input_unchanged(source, manifest, deadline, task_path=task_path, task_raw=task_raw)
        if actor.runtime_hashes(scene) != hashes:
            raise ValueError("Feedback source/runtime changed during training")
        weights = actor.coefficients(selected["feedback_coefficients"])
        policy = actor.make_policy(env, weights, seed, selected["score"], selected["optimization_rank"], hashes)
        actor.validate_shared_exercise_feedback_policy(policy, scene, task)
        writer.write("policy.json", policy)
        authored = not bool(np.any(weights))
        training.update(status="trained", seconds=time.monotonic() - started, policy_sha256=digest(out / "policy.json"),
            selected_authored_baseline=authored, selected_coast_baseline=False,
            selected_feedback_kind="authored_unchanged" if authored else "searched_feedback_residuals",
            improvement_over_authored_training=selected["score"] - authored_score,
            training_score_improved_over_authored=selected["score"] > authored_score,
            selected_candidate=origins[canonical_hash(weights.tolist())], selected_optimization_rank=selected["optimization_rank"],
            selection_objective=RULE, diagnostic_bytes=writer.diagnostic_bytes)
        writer.write("training.json", training)
        seal = writer.seal("trained")
        return {"directory": str(out), "manifest_sha256": seal, **training}
    except Exception as error:
        training.update(control_steps=budget.used, seconds=time.monotonic() - started, failure=f"{type(error).__name__}: {error}"[:2000])
        _failed(out, "training.json", training, error)
        raise
    finally:
        if env is not None:
            env.close()


def _verify_diagnostics(files, receipt, trial, weights, scene, count, deadline):
    keys = {"profile", "artifact", "sha256", "attempts", "results", "scope"}
    if (not isinstance(receipt, dict) or set(receipt) != keys or receipt["profile"] != DIAGNOSTIC_PROFILE
            or receipt["attempts"] != trial["control_steps"] or receipt["results"] != trial["control_steps"]):
        raise ValueError("Complete training diagnostics must cover every attempted control/result")
    if not trial["control_steps"]:
        if receipt["artifact"] is not None or receipt["sha256"] is not None:
            raise ValueError("Reset failure cannot fabricate action diagnostics")
        return None
    name = receipt["artifact"]
    if not isinstance(name, str) or Path(name).parent.as_posix() != "diagnostics" or name not in files:
        raise ValueError("Training diagnostic artifact is missing or outside its declared scope")
    raw = files[name]
    if len(raw) > MAX_DIAGNOSTIC_FILE_BYTES or _sha(raw) != receipt["sha256"]:
        raise ValueError("Training diagnostic identity/size differs")
    lines = raw.splitlines(keepends=True)
    if len(lines) != 2 * trial["control_steps"]:
        raise ValueError("Training diagnostics dropped or duplicated control rows")
    groups = actor.joint_groups(scene["action_contract"]["names"][:count])
    reward_sum, last_peak, last_root, last_time = 0., 0., 0., 0.
    for index in range(trial["control_steps"]):
        _check(deadline)
        pair = lines[2 * index:2 * index + 2]
        if any(len(line) > MAX_DIAGNOSTIC_LINE_BYTES or not line.endswith(b"\n") for line in pair):
            raise ValueError("Training diagnostic line is oversized or partial")
        attempt, result = map(_json, pair)
        attempt_keys = {"type", "index", "time_s", "joint_position_error_rad", "feedback_features", "action",
                        "authored_saturated_joints", "residual_at_bound_joints"}
        result_keys = {"type", "index", "reward", "terminated", "truncated", "simulated_duration_s",
                       "observed_substeps", "peak_angular_rmse_rad", "peak_root_rmse_model_units"}
        if (set(attempt) != attempt_keys or set(result) != result_keys or attempt["type"] != "attempt" or result["type"] != "result"
                or type(attempt["index"]) is not int or type(result["index"]) is not int
                or attempt["index"] != index + 1 or result["index"] != index + 1
                or type(attempt["time_s"]) not in (int, float) or attempt["time_s"] != index * scene["dt"]):
            raise ValueError("Training diagnostic control sequence differs")
        errors = attempt["joint_position_error_rad"]
        if (not isinstance(errors, list) or len(errors) != 19
                or any(type(v) not in (int, float) or not math.isfinite(v) or abs(v) > 2e6 for v in errors)):
            raise ValueError("Training diagnostic errors require nineteen finite bounded values")
        feature_env = SimpleNamespace(action_count=count, feature_matrix=lambda: np.asarray(attempt["feedback_features"]))
        action, _, authored, correction = actor.action_sample(weights, groups, feature_env)
        if (attempt["action"] != action.tolist()
                or attempt["authored_saturated_joints"] != np.flatnonzero(abs(authored[:19]) == 1.).tolist()
                or attempt["residual_at_bound_joints"] != np.flatnonzero(abs(correction) == actor.RESIDUAL_BOUND).tolist()):
            raise ValueError("Saved feedback action does not follow its exact features, coefficients and root bounds")
        for key in ("reward", "simulated_duration_s", "peak_angular_rmse_rad", "peak_root_rmse_model_units"):
            if type(result[key]) not in (int, float) or not math.isfinite(result[key]):
                raise ValueError("Training diagnostic result must be finite")
        _integer(result["observed_substeps"], "diagnostic observed substeps", index * scene["substeps"], (index + 1) * scene["substeps"])
        if (not 0 <= result["simulated_duration_s"] <= (index + 1) * scene["dt"] + 1e-8
                or result["simulated_duration_s"] < last_time - 1e-8
                or result["peak_angular_rmse_rad"] < last_peak or result["peak_root_rmse_model_units"] < last_root
                or any(type(result[key]) is not bool for key in ("terminated", "truncated"))
                or index < trial["control_steps"] - 1 and (result["terminated"] or result["truncated"])):
            raise ValueError("Training diagnostic clock/peak/terminal sequence differs")
        # Only a terminal failure can retain a partial or unavailable control-boundary clock.
        if (not (result["terminated"] or result["truncated"]) or trial["metrics"]["completed_interval"]) and (
                result["observed_substeps"] != (index + 1) * scene["substeps"]
                or not math.isclose(result["simulated_duration_s"], (index + 1) * scene["dt"], rel_tol=0, abs_tol=1e-8)):
            raise ValueError("Training diagnostic nonfailed control boundary is incomplete")
        reward_sum += result["reward"]
        last_peak, last_root = result["peak_angular_rmse_rad"], result["peak_root_rmse_model_units"]
        last_time = result["simulated_duration_s"]
    for key in ("simulated_duration_s", "observed_substeps", "peak_angular_rmse_rad", "peak_root_rmse_model_units"):
        if result[key] != trial["metrics"][key]:
            raise ValueError("Final training diagnostics differ from the original scored metrics")
    if (not math.isclose(reward_sum, trial["return"], rel_tol=0, abs_tol=1e-8)
            or result["terminated"] != trial["terminated"] or result["truncated"] != trial["truncated"]):
        raise ValueError("Training diagnostic return/termination differs")
    return name


def verify_shared_exercise_feedback_training(files, scene, task, *, deadline):
    """Verify saved search, matched starts, ranks and action arithmetic without physics."""
    scene, task, count = _context(scene, task)
    required = {"policy.json", "training.json", "settings.json", "scene.json", "exercise-task.json", "input-manifest.json"}
    if not required <= set(files) or _json(files["scene.json"]) != scene or _json(files["exercise-task.json"]) != task:
        raise ValueError("Feedback package must preserve its exact scene/task and original construction")
    policy = actor.validate_shared_exercise_feedback_policy(_json(files["policy.json"]), scene, task)
    for name, expected in policy["runtime_sha256"].items():
        if "implementation/" + name not in files or _sha(files["implementation/" + name]) != expected:
            raise ValueError("Feedback package does not preserve its executed runtime")
    settings, training = _json(files["settings.json"]), _json(files["training.json"])
    generations, population = settings.get("generations"), settings.get("population")
    _integer(generations, "generations", 1, 6)
    _integer(population, "population", 4, 12)
    _seconds(settings.get("max_seconds"), 60)
    maximum = generations * population * 2 * task["interval"]["horizon_steps"]
    if maximum > MAX_CONTROL_STEPS:
        raise ValueError("Feedback training metadata exceeds the original action bound")
    views = {name: files[name] for name in sorted(residual.source_view_names(scene) & set(files))}
    expected = _settings(scene, task, policy["runtime_sha256"], _sha(files["input-manifest.json"]), policy["training_seed"],
                         generations, population, settings["max_seconds"], views)
    expected["created_at"] = settings.get("created_at")
    if settings != expected or not isinstance(settings["created_at"], str):
        raise ValueError("Feedback settings differ from the frozen search and diagnostic contract")
    residual._terminal_manifest(files["input-manifest.json"])
    keys = {"status", "history", "completed_generations", "evaluated_episodes", "control_steps", "cache_hits",
            "active_candidates", "failure", "seconds", "policy_sha256", "selected_authored_baseline",
            "selected_coast_baseline", "selected_feedback_kind", "improvement_over_authored_training",
            "training_score_improved_over_authored", "selected_candidate", "selected_optimization_rank",
            "selection_objective", "diagnostic_bytes"}
    if (set(training) != keys or training["status"] != "trained" or training["failure"] is not None
            or training["active_candidates"] != [] or training["completed_generations"] != generations
            or len(training["history"]) != generations or training["policy_sha256"] != _sha(files["policy.json"])
            or type(training["seconds"]) not in (int, float) or not 0 <= training["seconds"] <= settings["max_seconds"]):
        raise ValueError("Feedback training is not a complete frozen terminal package")
    rng = np.random.default_rng(policy["training_seed"])
    mean, std, cache, origins = np.zeros((3, 2)), np.full((3, 2), .08), {}, {}
    selected, episodes, steps, hits, authored_score = None, 0, 0, 0, None
    initial_states, diagnostics = {}, set()
    for generation, record in enumerate(training["history"]):
        _check(deadline)
        if (set(record) != {"generation", "candidates", "next_mean", "next_std", "selected_optimization_rank"}
                or type(record["generation"]) is not int or record["generation"] != generation
                or not isinstance(record["candidates"], list) or len(record["candidates"]) != population):
            raise ValueError("Feedback generation sequence differs")
        samples = np.clip(rng.normal(mean, std, (population, 3, 2)), -actor.COEFFICIENT_BOUND, actor.COEFFICIENT_BOUND)
        samples[0] = 0.
        if generation:
            samples[1] = selected["feedback_coefficients"]
        for index, (row, weights) in enumerate(zip(record["candidates"], samples, strict=True)):
            row_keys = {"candidate", "feedback_coefficients", "trials", "complete", "cached", "score", "loss", "optimization_rank"}
            if (set(row) != row_keys or type(row["candidate"]) is not int or row["candidate"] != index
                    or row["feedback_coefficients"] != weights.tolist() or row["complete"] is not True
                    or not isinstance(row["trials"], list) or len(row["trials"]) != 2):
                raise ValueError("Feedback candidate differs from its reproducible generator")
            key = canonical_hash(weights.tolist())
            if key in cache:
                if row != cache[key] | {"candidate": index, "cached": True}:
                    raise ValueError("Feedback cache changed an earlier exact training trial")
                hits += 1
            else:
                if row["cached"] is not False:
                    raise ValueError("Feedback cache claimed an unevaluated candidate")
                for trial, seed, reset in zip(row["trials"], settings["training_seeds"], TRAIN_RESETS, strict=True):
                    stripped = {k: v for k, v in trial.items() if k != "diagnostics"}
                    validate_training_trial(stripped, scene, task, policy, seed, reset)
                    artifact = _verify_diagnostics(files, trial.get("diagnostics"), trial, weights, scene, count, deadline)
                    if artifact is not None:
                        if artifact in diagnostics:
                            raise ValueError("Different training episodes cannot reuse a diagnostic trace")
                        diagnostics.add(artifact)
                    episodes += 1
                    steps += trial["control_steps"]
                    if trial["initial_state_sha256"] is not None:
                        previous = initial_states.setdefault(seed, trial["initial_state_sha256"])
                        if previous != trial["initial_state_sha256"]:
                            raise ValueError("Feedback candidates changed their matched training starts")
                if row["score"] != _score(row["trials"]) or row["loss"] != -row["score"] or row["optimization_rank"] != optimization_rank(row["trials"]):
                    raise ValueError("Feedback candidate score/rank differs from unchanged training gates")
                cache[key], origins[key] = row, {"generation": generation, "candidate": index}
            if index == 0:
                authored_score = row["score"]
        selected = _selected(list(cache.values()))
        unique = {canonical_hash(row["feedback_coefficients"]): row for row in record["candidates"]}
        elite = sorted((row for row in unique.values() if row["optimization_rank"]["complete"]),
                       key=lambda row: _rank_key(row["optimization_rank"]))[:settings["elite_count"]]
        elite_weights = np.asarray([row["feedback_coefficients"] for row in elite])
        mean, std = elite_weights.mean(axis=0), np.maximum(elite_weights.std(axis=0), .02)
        if record["next_mean"] != mean.tolist() or record["next_std"] != std.tolist() or record["selected_optimization_rank"] != selected["optimization_rank"]:
            raise ValueError("Feedback next generation or selection differs from training-only feasibility ranking")
    selected_key = canonical_hash(selected["feedback_coefficients"])
    authored = not bool(np.any(selected["feedback_coefficients"]))
    fixed = {"evaluated_episodes": episodes, "control_steps": steps, "cache_hits": hits,
        "selected_candidate": origins[selected_key], "selected_optimization_rank": selected["optimization_rank"],
        "selection_objective": RULE, "selected_authored_baseline": authored, "selected_coast_baseline": False,
        "selected_feedback_kind": "authored_unchanged" if authored else "searched_feedback_residuals",
        "improvement_over_authored_training": selected["score"] - authored_score,
        "training_score_improved_over_authored": selected["score"] > authored_score,
        "diagnostic_bytes": sum(len(files[name]) for name in diagnostics)}
    if (any(training[key] != value for key, value in fixed.items()) or steps > maximum
            or fixed["diagnostic_bytes"] > MAX_DIAGNOSTIC_BYTES
            or diagnostics != {name for name in files if Path(name).parts[0] == "diagnostics"}
            or policy["feedback_coefficients"] != selected["feedback_coefficients"]
            or policy["training_score"] != selected["score"] or policy["optimization_rank"] != selected["optimization_rank"]):
        raise ValueError("Feedback selected policy, trace inventory or aggregate accounting differs")
    return {"unique_candidates": len(cache), "feasible_candidates": sum(row["optimization_rank"]["feasible"] for row in cache.values()),
            "selected_candidate": origins[selected_key], "selected_training_feasible": selected["optimization_rank"]["feasible"],
            "control_steps": steps, "evaluated_episodes": episodes, "policy_sha256": _sha(files["policy.json"]),
            "diagnostic_files": len(diagnostics), "diagnostic_bytes": fixed["diagnostic_bytes"]}


def verify_shared_exercise_feedback_package(manifest_raw, files, scene, task, *, deadline):
    manifest = _json(manifest_raw)
    if (manifest.get("status") != "trained" or "failure.json" in files or set(manifest.get("files", {})) != set(files)
            or any(_sha(files[name]) != expected for name, expected in manifest["files"].items())):
        raise ValueError("Feedback admission requires the complete successful training/selection seal")
    proof = verify_shared_exercise_feedback_training(files, scene, task, deadline=deadline)
    selection_names = {"selection.json", "selection-input-manifest.json"}
    if selection_names & set(files):
        if not selection_names <= set(files):
            raise ValueError("Feedback selection proof is incomplete")
        parent = _json(files["selection-input-manifest.json"])
        if (parent.get("status") != "trained" or set(parent.get("files", {})) != set(files) - selection_names
                or any(_sha(files[name]) != expected for name, expected in parent["files"].items())):
            raise ValueError("Feedback selection did not preserve the original exact training package")
        expected = _selection_receipt(proof, _sha(files["selection-input-manifest.json"]))
        if _json(files["selection.json"]) != expected:
            raise ValueError("Feedback saved-only selection differs from strict training proof")
    _check(deadline)
    return proof


def _selection_receipt(proof, parent_hash):
    return {"schema_version": 1, "profile": SELECTION_PROFILE,
        "selection_status": "retained_training_feasible_selection" if proof["selected_training_feasible"] else "retained_training_infeasible_selection",
        "training_manifest_sha256": parent_hash, **proof, "new_control_steps": 0, "evaluation_used": False,
        "scope": "Saved-only proof of the existing training-only feasible/least-violating selection; no new optimizer, physics or policy change"}


def select_shared_exercise_feedback_policy(learning_dir, out, *, max_seconds=30):
    _seconds(max_seconds, 30)
    source, out = Path(learning_dir).resolve(), _output_path(learning_dir, out)
    deadline = time.monotonic() + max_seconds
    manifest, files = _verified_inputs(source, deadline=deadline)
    if any(name in files for name in ("selection.json", "selection-input-manifest.json", "evaluation.json", "pre-evaluation-freeze.json")):
        raise ValueError("Feedback selection requires original training, not evaluation or chained selection")
    scene, task, _ = _context(_json(files["scene.json"]), _json(files["exercise-task.json"]))
    proof = verify_shared_exercise_feedback_package(manifest, files, scene, task, deadline=deadline)
    receipt = _selection_receipt(proof, _sha(manifest))
    copied = files | {"selection-input-manifest.json": manifest}
    if len(copied) + 3 >= MAX_OUTPUT_FILES or sum(map(len, copied.values())) + TERMINAL_RESERVE > MAX_OUTPUT_BYTES:
        raise ValueError("Feedback selection copy exceeds its reserved output bound")
    out.mkdir(mode=0o700)
    try:
        write_package_files(out, copied)
        writer = _Writer(out, deadline)
        writer.write("selection.json", receipt)
        residual._input_unchanged(source, manifest, deadline)
        if actor.runtime_hashes(scene) != _json(files["policy.json"])["runtime_sha256"]:
            raise ValueError("Feedback runtime changed during selection")
        seal = writer.seal("trained")
        return {"directory": str(out), "manifest_sha256": seal, **receipt}
    except Exception as error:
        _failed(out, "selection.json", receipt, error)
        raise


def evaluate_shared_exercise_feedback_policy(learning_dir, out, *, seed=924017, trials=3,
                                            jitter_scale=.10, max_seconds=30, excluded_seeds=()):
    """Compare the frozen selected actor with the same three baselines; never feed evaluation back."""
    _integer(seed, "evaluation seed", 0, 2**31 - 1)
    _integer(trials, "evaluation trials", 1, 3)
    _seconds(max_seconds, 60)
    excluded = residual._excluded_seeds(excluded_seeds)
    if type(jitter_scale) not in (int, float) or not math.isfinite(jitter_scale) or not 0 < jitter_scale <= .25:
        raise ValueError("Feedback evaluation jitter must be in (0,.25]")
    source, out = Path(learning_dir).resolve(), _output_path(learning_dir, out)
    started, deadline = time.monotonic(), time.monotonic() + max_seconds
    manifest, files = _verified_inputs(source, deadline=deadline)
    if "selection.json" not in files:
        raise ValueError("Feedback evaluation requires the strict saved-only selection stage")
    scene_raw, task_raw, policy_raw = files["scene.json"], files["exercise-task.json"], files["policy.json"]
    scene, task, _ = _context(_json(scene_raw), _json(task_raw))
    verify_shared_exercise_feedback_package(manifest, files, scene, task, deadline=deadline)
    policy = actor.validate_shared_exercise_feedback_policy(_json(policy_raw), scene, task)
    settings, training = _json(files["settings.json"]), _json(files["training.json"])
    training_seeds = settings["training_seeds"]
    if seed == settings["seed"]:
        raise ValueError("Feedback evaluation requires a distinct seed stream")
    fullspan = scene["profile"] in FULLSPAN_PROFILES
    _seconds(max_seconds, 60 if fullspan else 30)
    maximum = len(CONTROLLERS) * trials * task["interval"]["horizon_steps"]
    maximum_allowed = MAX_FULLSPAN_EVALUATION_STEPS if fullspan else MAX_EVALUATION_STEPS
    if maximum > maximum_allowed:
        raise ValueError("Feedback evaluation exceeds the existing complete-interval action bound")
    seeds = _seed_plan(seed, trials, [*training_seeds, *excluded])
    train_states = {trial["initial_state_sha256"] for generation in training["history"]
        for row in generation["candidates"] for trial in row["trials"] if trial["initial_state_sha256"] is not None}
    hashes = actor.runtime_hashes(scene)
    proof = collect_shared_source_files(scene, source, expected_manifest_sha256=_sha(manifest))
    views, reset = residual._source_views(source, manifest, scene), human_jitter_options(float(jitter_scale))
    evaluation = {"schema_version": 1, "status": "running", "created_at": local_time(), "policy_sha256": _sha(policy_raw),
        "scene_sha256": canonical_hash(scene), "task_sha256": canonical_hash(task), "runtime_sha256": hashes,
        "source_views_sha256": {name: _sha(raw) for name, raw in views.items()}, "seed": seed,
        "evaluation_seeds": seeds, "training_seeds": training_seeds, "excluded_seeds": excluded,
        "reset_options": reset, "horizon_steps": task["interval"]["horizon_steps"], "maximum_control_steps": maximum,
        "max_seconds": max_seconds, "workers": 1, "external_api_usd": 0, "control_steps": 0,
        "evaluation_feedback_to_optimizer": False, "source_scope": task["source_scope"], "scope": actor.SCOPE,
        "controllers": {}, "initial_states": {}}
    if fullspan:
        evaluation["evaluation_protocol"] = "shared-exercise-full-reference-evaluation-v1"
    freeze = {key: value for key, value in evaluation.items() if key not in ("status", "control_steps", "controllers", "initial_states")}
    residual._initialize(out, scene_raw, task_raw, manifest, proof,
        {**views, "policy.json": policy_raw, "pre-evaluation-freeze.json": _bytes(freeze)})
    writer = _Writer(out, deadline)
    env, budget = None, residual._Budget(maximum)
    try:
        writer.write("settings.json", freeze)
        env = SharedExerciseTaskEnv(scene, task)
        validator = prepare_shared_exercise_snapshot_validator(scene, task)
        weights = actor.coefficients(policy["feedback_coefficients"])
        for controller in CONTROLLERS:
            rows = []
            evaluation["controllers"][controller] = {"trials": rows}
            for trial_seed in seeds:
                try:
                    row = _trial(env, controller, trial_seed, reset, weights, deadline, budget, trace=True, validator=validator)
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
                    raise ValueError("Feedback evaluation repeated a training state; do not replace its seed")
                previous = evaluation["initial_states"].setdefault(str(trial_seed), initial_hash)
                if previous != initial_hash:
                    raise ValueError("Feedback controllers received different matched evaluation starts")
            if len(set(evaluation["initial_states"].values())) != len(evaluation["initial_states"]):
                raise ValueError("Feedback evaluation starts must remain distinct")
            evaluation["controllers"][controller].update(
                successes=sum(bool(row["metrics"] and row["metrics"]["success"]) for row in rows),
                completed_intervals=sum(bool(row["metrics"] and row["metrics"]["completed_interval"]) for row in rows),
                reset_failures=sum(row["status"] == "reset_failed" for row in rows), mean_score=_score(rows))
            writer.write("evaluation.json", evaluation)
        residual._input_unchanged(source, manifest, deadline)
        if actor.runtime_hashes(scene) != hashes:
            raise ValueError("Frozen feedback policy/runtime changed during evaluation")
        read_package_file(out, "policy.json", max_bytes=64 * 1024, expected_sha256=_sha(policy_raw))
        read_package_file(out, "pre-evaluation-freeze.json", max_bytes=64 * 1024, expected_sha256=_sha(_bytes(freeze)))
        evaluation.update(status="evaluated", seconds=time.monotonic() - started,
            independent_changed_states=len(evaluation["initial_states"]), policy_unchanged=True,
            comparison_complete=all(row["initial_state_sha256"] is not None for summary in evaluation["controllers"].values() for row in summary["trials"]),
            physical_validation="unverified")
        writer.write("evaluation.json", evaluation)
        seal = writer.seal("evaluated")
        return {"directory": str(out), "manifest_sha256": seal, **evaluation}
    except Exception as error:
        evaluation.update(control_steps=budget.used, seconds=time.monotonic() - started)
        _failed(out, "evaluation.json", evaluation, error)
        raise
    finally:
        if env is not None:
            env.close()
