"""Bounded external-policy search and frozen camera-task development evaluation."""

from copy import deepcopy
import json
import math
from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np

from weft import camera_shared_policy as actor
from weft import camera_shared_task as task_api
from weft.camera_shared_contract import validate_camera_coupling
from weft.camera_shared_state import _prepare_validated_camera_shared_snapshot_validator, _validate_action_values
from weft.io import canonical_hash, digest, local_time
from weft.shared_exercise_feedback_learning import _Writer
from weft.shared_exercise_lab import SOURCE_VIEWS, _json, _verified_inputs
from weft.shared_exercise_learning import _bytes, _integer, _output_path, _score, _seconds, _seed_plan, _sha
from weft.shared_exercise_reset import human_jitter_options
from weft.shared_exercise_residual_learning import _Budget, _excluded_seeds, _failure, _input_unchanged
from weft.shared_world_evidence import _document_paths, validate_equivalence_bindings
from weft.shared_world_packaging import collect_shared_source_files, read_package_file, validate_package_layout, write_package_files


TRAIN_PROFILE = "camera-shared-training-v1"
EVALUATION_PROFILE = "camera-shared-changed-start-evaluation-v1"
DIAGNOSTIC_PROFILE = "camera-shared-control-boundary-feedback-v1"
MAX_CONTROL_STEPS = 8640
MAX_EVALUATION_STEPS = 21600
MAX_OUTPUT_BYTES = 128 << 20
MAX_OUTPUT_FILES = 512
TRAIN_RESETS = (human_jitter_options(0.), human_jitter_options(.05))
CONTROLLERS = ("coast", "random", "authored", "learned")
CAMERA_VIEWS = SOURCE_VIEWS | {
    "human.json", "camera-coupling.json", "camera.json", "ground.json", "ground-clearance.json",
    "two-view-triangulation.json", "camera-provenance.json", "source-clock.json", "candidate.json",
}


def _check(deadline):
    if time.monotonic() >= deadline:
        raise TimeoutError("Camera policy work reached its cooperative deadline")


def _source(run, task_path, deadline):
    source = Path(run).resolve()
    source = source / "run" if (source / "run/manifest.json").is_file() else source
    manifest, files = _verified_inputs(source, deadline=deadline)
    if _json(manifest).get("status") not in {"ready", "constructed"}:
        raise ValueError("Camera training requires a completed untrained source, not a failed checkpoint")
    if any(part in {"policy.json", "training.json", "evaluation.json", "pre-evaluation-freeze.json"}
           for name in files for part in Path(name).parts):
        raise ValueError("Camera training cannot consume previous policy or evaluation outcomes")
    scene_raw = files.get("scene.json")
    if scene_raw is None or len(scene_raw) > 16 << 20:
        raise ValueError("Camera learning needs a bounded sealed actual scene")
    if task_path is None:
        task_raw = files.get("exercise-task.json")
        if task_raw is None:
            raise ValueError("Camera learning requires the exact sealed task or an explicit task file")
    else:
        path = Path(task_path).absolute()
        task_raw = read_package_file(path.parent, path.name, max_bytes=64 << 10)
    scene, task = task_api.prepare_camera_shared_task(_json(task_raw), _json(scene_raw))
    if "camera-coupling.json" not in files or _json(files["camera-coupling.json"]) != scene["camera_coupling"]:
        raise ValueError("Displayed camera coupling must remain exactly bound to the source scene")
    proof = collect_shared_source_files(scene, source, expected_manifest_sha256=_sha(manifest))
    views = {name: raw for name, raw in files.items()
             if name in CAMERA_VIEWS or name.startswith(("source-run/", "inputs/"))}
    if "exercise-task.json" in files and files["exercise-task.json"] != task_raw:
        views["input-original-task.json"] = files["exercise-task.json"]
    return source, scene, task, scene_raw, task_raw, manifest, proof, views


def _initialize(out, scene_raw, task_raw, manifest, proof, views, hashes, extra=None):
    reserved = {"manifest.json", "scene.json", "exercise-task.json", "input-manifest.json", "input-task.json",
                "source-verification.json", "implementation", "settings.json", "training.json", "evaluation.json",
                "policy.json", "failure.json", "pre-evaluation-freeze.json", "diagnostics"}
    proof.validate_layout(reserved_paths=reserved)
    files = dict(proof.files)
    for name, raw in views.items():
        if name in files and files[name] != raw:
            raise ValueError("Camera source evidence and copied view bytes disagree")
        files[name] = raw
    implementation = {"implementation/" + name: read_package_file(Path(__file__).parent, name,
        max_bytes=2 << 20, expected_sha256=pin) for name, pin in hashes.items()}
    files.update({"scene.json": scene_raw, "exercise-task.json": task_raw, "input-task.json": task_raw,
        "input-manifest.json": manifest, "source-verification.json": _bytes(proof.report),
        **implementation, **(extra or {})})
    if len(files) > MAX_OUTPUT_FILES - 96 or sum(map(len, files.values())) > MAX_OUTPUT_BYTES - (32 << 20):
        raise ValueError("Camera input/runtime custody exceeds the reserved output budget")
    out.mkdir(parents=True, mode=0o700, exist_ok=False)
    try:
        write_package_files(out, files)
    except Exception as error:
        _failure(out, "training.json", {"stage": "input_packaging"}, error)
        raise


def _settings(scene, task, manifest, views, hashes, seed, generations, population, max_seconds):
    maximum = generations * population * 2 * task["interval"]["horizon_steps"]
    return {"schema_version": 1, "profile": TRAIN_PROFILE, "algorithm": actor.ALGORITHM, "created_at": local_time(),
        "scene_sha256": canonical_hash(scene), "task_sha256": canonical_hash(task),
        **{key: task[key] for key in actor.IDENTITIES}, "gates_sha256": task["gates_sha256"], "runtime_sha256": hashes,
        "source_manifest_sha256": _sha(manifest), "source_views_sha256": {name: _sha(raw) for name, raw in views.items()},
        "seed": seed, "training_seeds": _seed_plan(seed, 2), "training_reset_options": list(TRAIN_RESETS),
        "generations": generations, "population": population, "elite_count": max(2, population // 4),
        "initial_mean": 0., "initial_std": .08, "minimum_std": .02, "coefficient_shape": list(actor.SHAPE),
        "coefficient_bound": actor.COEFFICIENT_BOUND, "maximum_control_steps": maximum,
        "maximum_physics_substeps": maximum * scene["substeps"], "horizon_steps": task["interval"]["horizon_steps"],
        "max_seconds": max_seconds, "workers": 1, "external_api_usd": 0, "evaluation_used": False,
        "candidate_anchors": ["zero_residual_authored", "previous_best_after_first_generation"],
        "selection_objective": actor.OBJECTIVE, "scope": actor.SCOPE}


def _trial(env, controller, seed, reset, weights, deadline, budget, writer, name, validator, *, trace=False):
    row = {"seed": seed, "reset_options": deepcopy(reset), "status": "starting", "control_steps": 0,
           "initial_state_sha256": None, "return": 0., "metrics": None, "diagnostics": None,
           "applied_residual": None}
    attempts = results = changed = 0
    peak_delta = delta_squared = 0.
    try:
        _check(deadline)
        try:
            env.reset(seed=seed, options=reset)
        except (ValueError, RuntimeError) as error:
            return row | {"status": "reset_failed", "error": f"{type(error).__name__}: {error}"[:2000]}
        initial = env.snapshot()
        validator(initial)
        row.update(initial_snapshot=initial, initial_state_sha256=canonical_hash(initial["engine_state"]))
        rng, groups = np.random.default_rng(seed ^ 0x5A17), actor.joint_groups(env.action_names)
        for index in range(env.horizon_steps):
            _check(deadline)
            if budget.used >= budget.maximum:
                budget.reserve()
            features = authored = correction = phase = None
            if controller in {"authored", "learned"}:
                action, features, authored, correction, phase = actor.action_sample(
                    np.zeros(actor.SHAPE) if controller == "authored" else weights, groups, env)
            elif controller == "coast":
                action = np.zeros(env.action_count)
            elif controller == "random":
                action = rng.uniform(-1., 1., env.action_count)
            else:
                raise ValueError("Unknown camera comparison controller")
            delta = None if authored is None else action[:19] - authored[:19]
            writer.diagnostic(name, {"type": "attempt", "index": index + 1, "phase": phase,
                "features": None if features is None else features.tolist(), "action": action.tolist(),
                "authored": None if authored is None else authored.tolist(),
                "correction": None if correction is None else correction.tolist(),
                "applied_delta": None if delta is None else delta.tolist()})
            attempts += 1
            budget.reserve()
            row["control_steps"] += 1
            _, reward, terminated, truncated, info = env.step(action)
            row["return"] += float(reward)
            result = {"type": "result", "index": index + 1, "reward": float(reward),
                      "terminated": bool(terminated), "truncated": bool(truncated),
                      **{key: info[key] for key in ("simulated_duration_s", "observed_substeps", "clock_status")}}
            if trace:
                snapshot = env.snapshot()
                validator(snapshot)
                result["snapshot_sha256"] = canonical_hash(snapshot)
            writer.diagnostic(name, result)
            results += 1
            if delta is not None:
                changed += bool(np.any(delta != 0.))
                peak_delta = max(peak_delta, float(np.max(abs(delta))))
                delta_squared += float(np.sum(delta ** 2))
            if terminated or truncated:
                break
        final = env.snapshot()
        validator(final)
        metrics = final["task_metrics"]
        if (not (env.terminated or env.truncated) or not (metrics["completed_interval"] or metrics["failure"] is not None)
                or metrics["completed_interval"] and row["control_steps"] != env.horizon_steps):
            raise ValueError("Camera trial must attempt the full fixed interval or preserve its terminal failure")
        row.update(status="completed" if metrics["completed_interval"] else "failed", metrics=metrics,
                   final_snapshot=final, final_snapshot_sha256=canonical_hash(final),
                   terminated=env.terminated, truncated=env.truncated)
        _check(deadline)
        return row
    except Exception as error:
        row.update(status="interrupted", error=f"{type(error).__name__}: {error}"[:2000])
        if row["initial_state_sha256"] is not None:
            row["metrics"] = env.metrics()
        error.partial_trial = row
        raise
    finally:
        if attempts:
            row["diagnostics"] = {"profile": DIAGNOSTIC_PROFILE, "artifact": name,
                "sha256": digest(writer.out / name), "attempts": attempts, "results": results}
        if controller in {"authored", "learned"} and row["initial_state_sha256"] is not None:
            row["applied_residual"] = {"observed_actions": results, "changed_actions": int(changed),
                "max_absolute_delta": peak_delta,
                "mean_squared_delta": delta_squared / (19 * results) if results else None}


def _eligible(row):
    return len(row["trials"]) == 2 and all(trial["initial_state_sha256"] is not None and trial["metrics"] is not None
                                          for trial in row["trials"])


def _selected(rows):
    eligible = [row for row in rows if _eligible(row)]
    if not eligible:
        raise ValueError("No candidate observed both fixed starts; no camera policy can be published")
    return max(eligible, key=lambda row: row["score"])


def _selected_delta(row):
    values = [trial["applied_residual"] for trial in row["trials"]]
    return {"observed_actions": sum(value["observed_actions"] for value in values),
            "changed_actions": sum(value["changed_actions"] for value in values),
            "max_absolute_delta": max(value["max_absolute_delta"] for value in values)}


def train_camera_shared_policy(run, out, *, task_path=None, seed=1729, generations=2, population=8, max_seconds=60):
    _integer(seed, "training seed", 0, 2**31 - 1)
    _integer(generations, "generations", 1, 6)
    _integer(population, "population", 4, 12)
    _seconds(max_seconds, 60)
    out = _output_path(run, out)
    started = time.monotonic()
    deadline = started + max_seconds
    source, scene, task, scene_raw, task_raw, manifest, proof, views = _source(run, task_path, deadline)
    maximum = generations * population * 2 * task["interval"]["horizon_steps"]
    if maximum > MAX_CONTROL_STEPS:
        raise ValueError("Camera training exceeds 8640 controls; never shorten the original interval")
    hashes = actor.runtime_hashes(scene)
    settings = _settings(scene, task, manifest, views, hashes, seed, generations, population, max_seconds)
    settings.update(input_task_sha256=_sha(task_raw), task_authority="sealed_source" if task_path is None else "explicit_file")
    _check(deadline)
    _initialize(out, scene_raw, task_raw, manifest, proof, views, hashes)
    writer, budget, env = _Writer(out, deadline), _Budget(maximum), None
    training = {"schema_version": 1, "profile": TRAIN_PROFILE, "status": "running", "history": [],
        "active_candidates": [], "completed_generations": 0, "evaluated_episodes": 0, "control_steps": 0,
        "cache_hits": 0, "failure": None}
    try:
        writer.write("settings.json", settings)
        env = task_api.CameraSharedTaskEnv(scene, task)
        validator = _prepare_validated_camera_shared_snapshot_validator(scene, task)
        rng = np.random.default_rng(seed)
        mean, std = np.zeros(actor.SHAPE), np.full(actor.SHAPE, .08)
        cache, origins, initial_states = {}, {}, {}
        selected = None
        for generation in range(generations):
            samples = np.clip(rng.normal(mean, std, (population, *actor.SHAPE)), -.25, .25)
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
                    row = {"candidate": index, "feedback_coefficients": weights.tolist(), "trials": [], "cached": False}
                    rows.append(row)
                    for reset_index, (trial_seed, reset) in enumerate(zip(settings["training_seeds"], TRAIN_RESETS, strict=True)):
                        training["evaluated_episodes"] += 1
                        try:
                            trial = _trial(env, "learned", trial_seed, reset, weights, deadline, budget, writer,
                                f"diagnostics/g{generation:02d}-c{index:02d}-r{reset_index}.jsonl", validator)
                        except Exception as error:
                            if hasattr(error, "partial_trial"):
                                row["trials"].append(error.partial_trial)
                            raise
                        row["trials"].append(trial)
                        training["control_steps"] = budget.used
                        if trial["initial_state_sha256"] is not None:
                            previous = initial_states.setdefault(trial_seed, trial["initial_state_sha256"])
                            if previous != trial["initial_state_sha256"]:
                                raise ValueError("Camera candidates received different matched training starts")
                    row["score"] = _score(row["trials"])
                    row["complete"] = _eligible(row)
                    cache[key], origins[key] = deepcopy(row), {"generation": generation, "candidate": index}
                writer.write("training.json", training)
            selected = _selected(list(cache.values()))
            unique = {canonical_hash(row["feedback_coefficients"]): row for row in rows}
            elites = sorted((row for row in unique.values() if _eligible(row)), key=lambda row: -row["score"])[:settings["elite_count"]]
            values = np.asarray([row["feedback_coefficients"] for row in elites])
            mean, std = values.mean(axis=0), np.maximum(values.std(axis=0), .02)
            training["history"].append({"generation": generation, "candidates": rows, "next_mean": mean.tolist(), "next_std": std.tolist()})
            training.update(active_candidates=[], completed_generations=generation + 1, control_steps=budget.used)
        _input_unchanged(source, manifest, deadline, task_path=task_path, task_raw=task_raw)
        if actor.runtime_hashes(scene) != hashes:
            raise ValueError("Camera source/runtime changed during training")
        weights = actor.coefficients(selected["feedback_coefficients"])
        policy = actor.make_policy(env, weights, seed, selected["score"], hashes)
        actor.validate_prepared_policy(policy, scene, task, hashes=hashes)
        writer.write("policy.json", policy)
        authored = cache[canonical_hash(np.zeros(actor.SHAPE).tolist())]
        kind = actor.selection_kind(policy)
        training.update(status="trained", seconds=time.monotonic() - started, policy_sha256=digest(out / "policy.json"),
            selected_candidate=origins[canonical_hash(weights.tolist())], selected_feedback_kind=kind,
            selected_authored_baseline=kind == "authored_unchanged", selected_coast_baseline=False,
            selected_applied_residual=_selected_delta(selected), nonzero_coefficients=int(np.count_nonzero(weights)),
            improvement_over_authored_training=selected["score"] - authored["score"],
            training_score_improved_over_authored=selected["score"] > authored["score"],
            evaluation_used=False, learning_qualification="unqualified_research", diagnostic_bytes=writer.diagnostic_bytes)
        writer.write("training.json", training)
        seal = writer.seal("trained")
        _check(deadline)
        return {"directory": str(out), "manifest_sha256": seal, **training}
    except Exception as error:
        training.update(control_steps=budget.used, seconds=time.monotonic() - started, failure=f"{type(error).__name__}: {error}"[:2000])
        _failure(out, "training.json", training, error)
        raise
    finally:
        if env is not None:
            env.close()


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _same_number(actual, expected, message, *, tolerance=1e-9):
    _require(type(actual) in (int, float) and math.isfinite(actual)
             and math.isclose(actual, expected, rel_tol=0, abs_tol=tolerance), message)


def _saved_context(scene, task):
    """Check pinned structural contracts, not fresh native model admission or physical validity."""
    _require(isinstance(scene, dict) and scene.get("profile") == "shared-world-v6" and scene.get("objects") == []
             and scene.get("learned_policy_available") is False, "Saved camera scene changed its source qualification")
    _require(isinstance(task, dict) and task.get("profile") == task_api.PROFILE
             and task.get("control", {}).get("root_assistance") in {"disabled", "explicit_3_axis"}, "Saved camera task differs")
    validate_camera_coupling(scene["human_spec"], scene["camera_coupling"], dt=scene["dt"],
                            coordinate_frame=scene["coordinate_frame"], source_binding=scene["source_binding"])
    expected = task_api._envelope(scene, task["control"]["root_assistance"] == "explicit_3_axis")
    _require(task_api._task_bytes(task) == task_api._task_bytes(expected), "Saved camera context/scorer/task identities differ")
    count = 22 if task["control"]["root_assistance"] == "explicit_3_axis" else 19
    _require(len(scene["action_contract"]["names"]) == 22 and scene["action_contract"]["human_motor_count"] == 22
             and scene["action_contract"]["object_force_channels"] == 0 and not scene["action_contract"]["dynamic_object_ids"],
             "Saved camera action authority differs")
    return count, _prepare_validated_camera_shared_snapshot_validator(scene, task)


def _package_identity(manifest_raw, files, status, deadline):
    _require(isinstance(files, dict) and len(files) < MAX_OUTPUT_FILES
             and len(manifest_raw) + sum(map(len, files.values())) <= MAX_OUTPUT_BYTES, "Saved camera package exceeds its bounds")
    manifest = _json(manifest_raw)
    _require(manifest.get("status") == status and set(manifest.get("files", {})) == set(files), "Saved camera terminal inventory differs")
    for name, raw in files.items():
        _check(deadline)
        _require(manifest["files"][name] == _sha(raw), "Saved camera member hash differs: " + name)


def _source_custody(scene, parent_raw, files, authority, task_sha):
    """Required members come from immutable parent/binding inventories, never remaining output files."""
    parent = _json(parent_raw)
    _require(parent.get("status") in {"ready", "constructed"} and isinstance(parent.get("files"), dict),
             "Saved camera training source was not an untrained terminal package")
    pins = parent["files"]
    expected_views = {name for name in pins if name in CAMERA_VIEWS or name.startswith(("source-run/", "inputs/"))}
    original_task = authority == "explicit_file" and "exercise-task.json" in pins and pins["exercise-task.json"] != task_sha
    if original_task:
        expected_views.add("input-original-task.json")
    present_views = {name for name in files if name in CAMERA_VIEWS or name.startswith(("source-run/", "inputs/"))
                     or name == "input-original-task.json"}
    _require(expected_views == present_views, "Saved camera required source-view inventory differs from immutable parent custody")
    required = {name: pins["exercise-task.json" if name == "input-original-task.json" else name] for name in expected_views}
    binding = scene["source_binding"]
    _require(binding.get("mode") == "source_qualified", "Camera source proof requires its source-qualified binding")
    for sha, name in _document_paths(binding).items():
        _require(pins.get(name) == sha, "Parent source-proof custody differs from its observation binding")
        required[name] = sha
    for mapping in validate_equivalence_bindings(binding):
        prefix = mapping["bundle_path"] + "/"
        name = prefix + "manifest.json"
        pin = mapping["bundle_manifest_sha256"]
        _require(name in files and len(files[name]) <= 1 << 20 and pins.get(name) == pin and _sha(files[name]) == pin,
                 "Saved camera source-equivalence manifest custody is missing or changed")
        bundle = _json(files[name])
        members = bundle.get("files")
        _require(bundle.get("schema") == 1 and isinstance(members, dict) and 1 <= len(members) < 256
                 and "manifest.json" not in members, "Saved camera source-equivalence inventory is malformed")
        validate_package_layout(["manifest.json", *members])
        required[name] = pin
        for relative, digest_value in members.items():
            member = prefix + relative
            _require(pins.get(member) == digest_value, "Parent source-equivalence member custody differs")
            required[member] = digest_value
    for name, pin in required.items():
        _require(name in files and _sha(files[name]) == pin, "Copied camera member differs from immutable parent custody: " + name)
    _require(pins.get("scene.json") == _sha(files["scene.json"]), "Saved camera scene bytes differ from the parent")
    if authority == "sealed_source":
        _require(pins.get("exercise-task.json") == task_sha, "Saved camera task bytes differ from the sealed source authority")
    return {name: files[name] for name in expected_views}


def _saved_trial(row, scene, task, weights, controller, seed, reset, files, validator, deadline):
    _require(row.get("seed") == seed and _bytes(row.get("reset_options")) == _bytes(reset), "Saved camera trial start differs")
    steps = row.get("control_steps")
    horizon, count = task["interval"]["horizon_steps"], 22 if task["control"]["root_assistance"] == "explicit_3_axis" else 19
    _integer(steps, "saved trial controls", 0, horizon)
    if row.get("status") == "reset_failed":
        _require(steps == 0 and row.get("initial_state_sha256") is None and row.get("metrics") is None
                 and row.get("diagnostics") is None and row.get("applied_residual") is None
                 and isinstance(row.get("error"), str) and 0 < len(row["error"]) <= 2000,
                 "Failed camera reset cannot contain invented observations or controls")
        _same_number(row.get("return"), 0., "Failed reset return differs")
        return
    _require(row.get("status") in {"completed", "failed"} and steps > 0, "Saved camera trial is not terminal")
    initial, final = row["initial_snapshot"], row["final_snapshot"]
    validator(initial)
    validator(final)
    _require(initial["resolved_reset_options"] == task_api._reset_options(reset)
             and initial["task_metrics"]["observed_substeps"] == 0 and initial["engine_state"]["steps"] == 0
             and not initial["terminated"] and not initial["truncated"]
             and row["initial_state_sha256"] == canonical_hash(initial["engine_state"]), "Saved camera reset snapshot differs")
    metrics = final["task_metrics"]
    _require(row["final_snapshot_sha256"] == canonical_hash(final) and row["metrics"] == metrics
             and row.get("terminated") is final["terminated"] and row.get("truncated") is final["truncated"]
             and (final["terminated"] or final["truncated"])
             and row["status"] == ("completed" if metrics["completed_interval"] else "failed")
             and (metrics["completed_interval"] and steps == horizon or metrics["failure"] is not None),
             "Saved camera outcome changed its observed terminal state")
    receipt = row.get("diagnostics")
    _require(isinstance(receipt, dict) and receipt.get("profile") == DIAGNOSTIC_PROFILE
             and receipt.get("attempts") == receipt.get("results") == steps, "Saved camera diagnostic coverage differs")
    name = receipt.get("artifact")
    _require(isinstance(name, str) and name.startswith("diagnostics/") and name in files, "Saved camera diagnostic is missing")
    raw = files[name]
    _require(len(raw) <= 8 << 20 and _sha(raw) == receipt.get("sha256"), "Saved camera diagnostic identity differs")
    lines = raw.splitlines()
    _require(len(lines) == steps * 2 and all(len(line) <= 8192 for line in lines), "Saved camera diagnostic record bounds differ")
    rng, total, changed, peak, squared = np.random.default_rng(seed ^ 0x5A17), 0., 0, 0., 0.
    groups = actor.joint_groups(scene["action_contract"]["names"][:count])
    for index in range(steps):
        _check(deadline)
        attempt, result = _json(lines[2 * index]), _json(lines[2 * index + 1])
        _require(set(attempt) == {"type", "index", "phase", "features", "action", "authored", "correction", "applied_delta"}
                 and attempt["type"] == "attempt" and attempt["index"] == index + 1
                 and result.get("type") == "result" and result.get("index") == index + 1, "Saved camera diagnostic ordering differs")
        _validate_action_values(attempt["action"], count)
        if controller in {"learned", "authored"}:
            _same_number(attempt["phase"], index / horizon, "Saved camera phase retimed the original interval", tolerance=1e-6)
            proxy = SimpleNamespace(action_count=count, observation=lambda: np.r_[np.zeros(155), attempt["phase"]],
                                    feature_matrix=lambda: np.asarray(attempt["features"]))
            action, _, authored, correction, _ = actor.action_sample(
                np.zeros(actor.SHAPE) if controller == "authored" else weights, groups, proxy)
            delta = action[:19] - authored[:19]
            for key, value in (("action", action), ("authored", authored), ("correction", correction), ("applied_delta", delta)):
                _require(np.array_equal(np.asarray(attempt[key]), value), "Saved camera action differs from frozen feedback: " + key)
            changed += bool(np.any(delta != 0.))
            peak, squared = max(peak, float(np.max(abs(delta)))), squared + float(np.sum(delta ** 2))
        else:
            expected = np.zeros(count) if controller == "coast" else rng.uniform(-1., 1., count)
            _require(np.array_equal(np.asarray(attempt["action"]), expected)
                     and all(attempt[key] is None for key in ("phase", "features", "authored", "correction", "applied_delta")),
                     "Saved camera baseline action differs")
        _require(type(result.get("reward")) in (int, float) and math.isfinite(result["reward"])
                 and type(result.get("terminated")) is bool and type(result.get("truncated")) is bool,
                 "Saved camera diagnostic result is malformed")
        total += result["reward"]
        if index < steps - 1:
            _require(not result["terminated"] and not result["truncated"] and result.get("clock_status") == "observed"
                     and result.get("observed_substeps") == (index + 1) * scene["substeps"], "Saved camera early boundary differs")
            _same_number(result["simulated_duration_s"], (index + 1) * scene["dt"], "Saved camera control clock differs")
        else:
            _require(result["terminated"] is final["terminated"] and result["truncated"] is final["truncated"]
                     and result.get("clock_status") == metrics["clock_status"]
                     and result.get("observed_substeps") == metrics["observed_substeps"], "Saved camera final diagnostic differs")
            _same_number(result["simulated_duration_s"], metrics["simulated_duration_s"], "Saved final observed time differs")
            if "snapshot_sha256" in result:
                _require(result["snapshot_sha256"] == row["final_snapshot_sha256"], "Saved final snapshot pin differs")
    _same_number(row.get("return"), total, "Saved camera accumulated return differs")
    _same_number(total * scene["dt"], metrics["reward_integral"], "Saved return differs from unchanged scorer integral")
    expected_delta = None if controller not in {"authored", "learned"} else {
        "observed_actions": steps, "changed_actions": int(changed), "max_absolute_delta": peak,
        "mean_squared_delta": squared / (19 * steps)}
    _require(row["applied_residual"] == expected_delta, "Saved camera realized action deltas differ")


def verify_camera_shared_training(manifest_raw, files, scene, task, *, deadline):
    """Verify saved search/selection and structural custody; no native readmission or rollout."""
    _package_identity(manifest_raw, files, "trained", deadline)
    _require(_json(files["scene.json"]) == scene and _json(files["exercise-task.json"]) == task
             and files["input-task.json"] == files["exercise-task.json"], "Saved camera scene/task bytes differ")
    _, validator = _saved_context(scene, task)
    hashes = actor.runtime_hashes(scene)
    for name, pin in hashes.items():
        _require("implementation/" + name in files and _sha(files["implementation/" + name]) == pin,
                 "Saved camera implementation differs from its pinned runtime")
    policy = actor.validate_prepared_policy(_json(files["policy.json"]), scene, task, hashes=hashes)
    settings, training = _json(files["settings.json"]), _json(files["training.json"])
    for key, low, high in (("seed", 0, 2**31 - 1), ("generations", 1, 6), ("population", 4, 12)):
        _integer(settings.get(key), key, low, high)
    _seconds(settings.get("max_seconds"), 60)
    authority = settings.get("task_authority")
    _require(authority in {"sealed_source", "explicit_file"}, "Saved camera task authority is missing")
    views = _source_custody(scene, files["input-manifest.json"], files, authority, _sha(files["input-task.json"]))
    expected = _settings(scene, task, files["input-manifest.json"], views, hashes,
                         settings["seed"], settings["generations"], settings["population"], settings["max_seconds"])
    expected["created_at"] = settings.get("created_at")
    expected.update(input_task_sha256=_sha(files["input-task.json"]), task_authority=authority)
    _require(settings == expected and settings["maximum_control_steps"] <= MAX_CONTROL_STEPS,
             "Saved camera search settings changed its fixed bounded protocol")
    _require(training.get("profile") == TRAIN_PROFILE and training.get("status") == "trained"
             and training.get("failure") is None and training.get("active_candidates") == []
             and training.get("completed_generations") == settings["generations"]
             and len(training.get("history", [])) == settings["generations"], "Saved camera training is not complete")
    rng, cache, origins, initial_states = np.random.default_rng(settings["seed"]), {}, {}, {}
    mean, std, selected = np.zeros(actor.SHAPE), np.full(actor.SHAPE, .08), None
    controls = episodes = cache_hits = diagnostic_bytes = 0
    for generation, entry in enumerate(training["history"]):
        samples = np.clip(rng.normal(mean, std, (settings["population"], *actor.SHAPE)), -.25, .25)
        samples[0] = 0.
        if generation:
            samples[1] = selected["feedback_coefficients"]
        rows = entry.get("candidates")
        _require(entry.get("generation") == generation and isinstance(rows, list) and len(rows) == settings["population"],
                 "Saved camera generation differs")
        for index, row in enumerate(rows):
            _check(deadline)
            weights = actor.coefficients(row.get("feedback_coefficients"))
            _require(row.get("candidate") == index and np.array_equal(weights, samples[index]), "Saved camera CEM sample differs")
            key = canonical_hash(weights.tolist())
            if key in cache:
                _require(row == deepcopy(cache[key]) | {"candidate": index, "cached": True}, "Saved camera cached trial was changed")
                cache_hits += 1
                continue
            _require(row.get("cached") is False and len(row.get("trials", [])) == 2, "Saved camera candidate coverage differs")
            for trial, trial_seed, reset in zip(row["trials"], settings["training_seeds"], TRAIN_RESETS, strict=True):
                _saved_trial(trial, scene, task, weights, "learned", trial_seed, reset, files, validator, deadline)
                episodes += 1
                controls += trial["control_steps"]
                if trial["diagnostics"] is not None:
                    diagnostic_bytes += len(files[trial["diagnostics"]["artifact"]])
                if trial["initial_state_sha256"] is not None:
                    previous = initial_states.setdefault(trial_seed, trial["initial_state_sha256"])
                    _require(previous == trial["initial_state_sha256"], "Saved training starts were not matched")
            _same_number(row.get("score"), _score(row["trials"]), "Saved camera candidate score differs")
            _require(row.get("complete") is _eligible(row), "Saved camera candidate eligibility differs")
            cache[key], origins[key] = row, {"generation": generation, "candidate": index}
        selected = _selected(list(cache.values()))
        unique = {canonical_hash(row["feedback_coefficients"]): row for row in rows}
        elites = sorted((row for row in unique.values() if _eligible(row)), key=lambda row: -row["score"])[:settings["elite_count"]]
        values = np.asarray([row["feedback_coefficients"] for row in elites])
        mean, std = values.mean(axis=0), np.maximum(values.std(axis=0), .02)
        _require(np.array_equal(np.asarray(entry.get("next_mean")), mean) and np.array_equal(np.asarray(entry.get("next_std")), std),
                 "Saved camera CEM update differs")
    weights = actor.coefficients(policy["feedback_coefficients"])
    key = canonical_hash(weights.tolist())
    _require(key in origins, "Saved camera policy was not a searched candidate")
    authored = cache[canonical_hash(np.zeros(actor.SHAPE).tolist())]
    kind = actor.selection_kind(policy)
    labels = {"policy_sha256": _sha(files["policy.json"]), "selected_candidate": origins[key],
        "selected_feedback_kind": kind, "selected_authored_baseline": kind == "authored_unchanged", "selected_coast_baseline": False,
        "selected_applied_residual": _selected_delta(selected), "nonzero_coefficients": int(np.count_nonzero(weights)),
        "improvement_over_authored_training": selected["score"] - authored["score"],
        "training_score_improved_over_authored": selected["score"] > authored["score"],
        "evaluation_used": False, "learning_qualification": "unqualified_research", "control_steps": controls,
        "evaluated_episodes": episodes, "cache_hits": cache_hits, "diagnostic_bytes": diagnostic_bytes}
    _require(policy["feedback_coefficients"] == selected["feedback_coefficients"] and policy["training_seed"] == settings["seed"],
             "Saved camera policy was not the training-only selected candidate")
    _same_number(policy["training_score"], selected["score"], "Saved camera selected score differs")
    _require(all(training.get(key) == value for key, value in labels.items()) and controls <= settings["maximum_control_steps"],
             "Saved camera result labels or work accounting differ")
    return {"verified": True, "profile": TRAIN_PROFILE, "policy_sha256": labels["policy_sha256"],
            "training_seeds": settings["training_seeds"], "initial_states": initial_states,
            "selected_feedback_kind": kind, "control_steps": controls, "evaluated_episodes": episodes,
            "native_model_readmission": False, "scope": "Saved structural/custody/search consistency, not physical validation"}


TRAINING_ALIASES = {"settings.json": "training-settings.json", "training.json": "training-history.json",
                    "input-manifest.json": "training-source-manifest.json", "source-verification.json": "training-source-verification.json"}


def _evaluation_freeze(scene, task, policy_raw, manifest, hashes, seed, seeds, training_seeds, excluded, reset, max_seconds):
    maximum = len(CONTROLLERS) * len(seeds) * task["interval"]["horizon_steps"]
    return {"schema_version": 1, "profile": EVALUATION_PROFILE, "created_at": local_time(),
        "scene_sha256": canonical_hash(scene), "task_sha256": canonical_hash(task),
        **{key: task[key] for key in actor.IDENTITIES}, "gates_sha256": task["gates_sha256"],
        "policy_sha256": _sha(policy_raw), "training_manifest_sha256": _sha(manifest), "runtime_sha256": hashes,
        "seed": seed, "evaluation_seeds": seeds, "training_seeds": training_seeds, "excluded_seeds": excluded,
        "reset_options": reset, "controllers": list(CONTROLLERS), "horizon_steps": task["interval"]["horizon_steps"],
        "maximum_control_steps": maximum, "maximum_physics_substeps": maximum * scene["substeps"],
        "max_seconds": max_seconds, "workers": 1, "external_api_usd": 0, "evaluation_feedback_to_optimizer": False,
        "learning_qualification": "unqualified_research", "scope": actor.SCOPE}


def _comparison(evaluation):
    learned = evaluation["controllers"]["learned"]["trials"]
    authored = evaluation["controllers"]["authored"]["trials"]
    return [{"seed": left["seed"], "learned_status": left["status"], "authored_status": right["status"],
             "mean_reward_delta": (left["metrics"]["mean_reward"] - right["metrics"]["mean_reward"])
                 if left["metrics"] is not None and right["metrics"] is not None else None,
             "learned_success": bool(left["metrics"] and left["metrics"]["success"]),
             "authored_success": bool(right["metrics"] and right["metrics"]["success"]),
             "learned_applied_residual": left["applied_residual"]}
            for left, right in zip(learned, authored, strict=True)]


def evaluate_camera_shared_policy(training_dir, out, *, seed=924017, trials=3, jitter_scale=.10,
                                  max_seconds=60, excluded_seeds=()):
    _integer(seed, "evaluation seed", 0, 2**31 - 1)
    _integer(trials, "evaluation trials", 1, 3)
    _seconds(max_seconds, 60)
    excluded = _excluded_seeds(excluded_seeds)
    if type(jitter_scale) not in (int, float) or not 0 < jitter_scale <= .25 or not math.isfinite(jitter_scale):
        raise ValueError("Camera changed-start jitter must be explicit in (0,.25]")
    source, out = Path(training_dir).resolve(), _output_path(training_dir, out)
    started = time.monotonic()
    deadline = started + max_seconds
    manifest, files = _verified_inputs(source, deadline=deadline)
    scene_raw, task_raw, policy_raw = files["scene.json"], files["exercise-task.json"], files["policy.json"]
    scene, task = task_api.prepare_camera_shared_task(_json(task_raw), _json(scene_raw))
    verified = verify_camera_shared_training(manifest, files, scene, task, deadline=deadline)
    settings, policy = _json(files["settings.json"]), _json(policy_raw)
    if seed == settings["seed"]:
        raise ValueError("Camera evaluation requires a separate random seed stream")
    seeds = _seed_plan(seed, trials, [*verified["training_seeds"], *excluded])
    maximum = len(CONTROLLERS) * trials * task["interval"]["horizon_steps"]
    if maximum > MAX_EVALUATION_STEPS:
        raise ValueError("Camera evaluation exceeds 21600 controls; never shorten the source interval")
    hashes = actor.runtime_hashes(scene)
    proof = collect_shared_source_files(scene, source, expected_manifest_sha256=_sha(manifest))
    views = {name: raw for name, raw in files.items()
             if name in CAMERA_VIEWS or name.startswith(("source-run/", "inputs/")) or name == "input-original-task.json"}
    freeze = _evaluation_freeze(scene, task, policy_raw, manifest, hashes, seed, seeds,
        verified["training_seeds"], excluded, human_jitter_options(float(jitter_scale)), max_seconds)
    extra = {**{target: files[name] for name, target in TRAINING_ALIASES.items()},
             **{name: raw for name, raw in files.items() if name.startswith("diagnostics/")},
             "policy.json": policy_raw, "pre-evaluation-freeze.json": _bytes(freeze)}
    _initialize(out, scene_raw, task_raw, manifest, proof, views, hashes, extra)
    writer, budget, env = _Writer(out, deadline), _Budget(maximum), None
    evaluation = {**freeze, "status": "running", "controllers": {}, "initial_states": {}, "control_steps": 0}
    try:
        writer.write("settings.json", freeze)
        env = task_api.CameraSharedTaskEnv(scene, task)
        validator = _prepare_validated_camera_shared_snapshot_validator(scene, task)
        weights = actor.coefficients(policy["feedback_coefficients"])
        train_states = set(verified["initial_states"].values())
        for controller in CONTROLLERS:
            rows = []
            evaluation["controllers"][controller] = {"trials": rows}
            for index, trial_seed in enumerate(seeds):
                try:
                    row = _trial(env, controller, trial_seed, freeze["reset_options"], weights, deadline, budget, writer,
                        f"diagnostics/e-{controller}-{index}.jsonl", validator, trace=True)
                except Exception as error:
                    if hasattr(error, "partial_trial"):
                        rows.append(error.partial_trial)
                    raise
                rows.append(row)
                evaluation["control_steps"] = budget.used
                state = row["initial_state_sha256"]
                if state is not None:
                    if state in train_states:
                        raise ValueError("Camera evaluation repeated a training state; do not replace its seed")
                    previous = evaluation["initial_states"].setdefault(str(trial_seed), state)
                    if previous != state:
                        raise ValueError("Camera comparison controllers received different matched starts")
            if len(set(evaluation["initial_states"].values())) != len(evaluation["initial_states"]):
                raise ValueError("Camera changed-start states are not distinct")
            evaluation["controllers"][controller].update(successes=sum(bool(row["metrics"] and row["metrics"]["success"]) for row in rows),
                completed_intervals=sum(bool(row["metrics"] and row["metrics"]["completed_interval"]) for row in rows),
                reset_failures=sum(row["status"] == "reset_failed" for row in rows), mean_score=_score(rows))
            writer.write("evaluation.json", evaluation)
        _input_unchanged(source, manifest, deadline)
        if actor.runtime_hashes(scene) != hashes:
            raise ValueError("Frozen camera policy/runtime changed during evaluation")
        read_package_file(out, "policy.json", max_bytes=64 << 10, expected_sha256=_sha(policy_raw))
        read_package_file(out, "pre-evaluation-freeze.json", max_bytes=64 << 10, expected_sha256=_sha(_bytes(freeze)))
        evaluation.update(status="evaluated", seconds=time.monotonic() - started, policy_unchanged=True,
            independent_changed_states=len(evaluation["initial_states"]),
            comparison_complete=all(row["initial_state_sha256"] is not None for summary in evaluation["controllers"].values() for row in summary["trials"]),
            selected_feedback_kind=actor.selection_kind(policy), nonzero_coefficients=int(np.count_nonzero(weights)),
            paired_vs_authored=_comparison(evaluation), physical_validation="unverified")
        writer.write("evaluation.json", evaluation)
        seal = writer.seal("evaluated")
        _check(deadline)
        return {"directory": str(out), "manifest_sha256": seal, **evaluation}
    except Exception as error:
        evaluation.update(control_steps=budget.used, seconds=time.monotonic() - started)
        _failure(out, "evaluation.json", evaluation, error)
        raise
    finally:
        if env is not None:
            env.close()


def verify_camera_shared_evaluation(manifest_raw, files, scene, task, *, deadline):
    """Verify matched saved outcomes and unchanged policy without rerunning the engine."""
    _package_identity(manifest_raw, files, "evaluated", deadline)
    parent = files["input-manifest.json"]
    parent_files = {name: files[TRAINING_ALIASES.get(name, name)] for name in _json(parent)["files"]}
    training = verify_camera_shared_training(parent, parent_files, scene, task, deadline=deadline)
    _, validator = _saved_context(scene, task)
    policy, settings = _json(files["policy.json"]), _json(files["training-settings.json"])
    freeze, evaluation = _json(files["pre-evaluation-freeze.json"]), _json(files["evaluation.json"])
    _require(files["settings.json"] == files["pre-evaluation-freeze.json"], "Saved evaluation settings were not frozen")
    seed, seeds = freeze.get("seed"), freeze.get("evaluation_seeds")
    _integer(seed, "evaluation seed", 0, 2**31 - 1)
    _require(isinstance(seeds, list) and 1 <= len(seeds) <= 3 and seed != settings["seed"], "Saved evaluation seed stream differs")
    excluded = _excluded_seeds(freeze.get("excluded_seeds"))
    _require(seeds == _seed_plan(seed, len(seeds), [*training["training_seeds"], *excluded]), "Saved evaluation seeds were reused or replaced")
    reset = freeze.get("reset_options")
    _require(isinstance(reset, dict), "Saved evaluation reset is missing")
    jitter = reset.get("jitter_scale")
    _require(type(jitter) in (int, float) and 0 < jitter <= .25 and reset == human_jitter_options(float(jitter)),
             "Saved evaluation reset distribution differs")
    _seconds(freeze.get("max_seconds"), 60)
    expected = _evaluation_freeze(scene, task, files["policy.json"], parent, policy["runtime_sha256"], seed, seeds,
        training["training_seeds"], excluded, reset, freeze["max_seconds"])
    expected["created_at"] = freeze.get("created_at")
    _require(freeze == expected and freeze["maximum_control_steps"] <= MAX_EVALUATION_STEPS, "Saved evaluation freeze differs")
    _require(evaluation.get("status") == "evaluated" and set(evaluation.get("controllers", {})) == set(CONTROLLERS)
             and all(evaluation.get(key) == value for key, value in freeze.items() if key != "controllers"),
             "Saved evaluation identities or baselines differ")
    weights = actor.coefficients(policy["feedback_coefficients"])
    states, controls = {}, 0
    for controller in CONTROLLERS:
        summary = evaluation["controllers"][controller]
        rows = summary.get("trials")
        _require(isinstance(rows, list) and len(rows) == len(seeds), "Saved evaluation omitted a start")
        for row, trial_seed in zip(rows, seeds, strict=True):
            _saved_trial(row, scene, task, weights, controller, trial_seed, reset, files, validator, deadline)
            controls += row["control_steps"]
            pin = row["initial_state_sha256"]
            if pin is not None:
                _require(pin not in training["initial_states"].values(), "Saved evaluation reused a training state")
                previous = states.setdefault(str(trial_seed), pin)
                _require(previous == pin, "Saved evaluation baselines did not receive matched starts")
        summaries = {"successes": sum(bool(row["metrics"] and row["metrics"]["success"]) for row in rows),
            "completed_intervals": sum(bool(row["metrics"] and row["metrics"]["completed_interval"]) for row in rows),
            "reset_failures": sum(row["status"] == "reset_failed" for row in rows), "mean_score": _score(rows)}
        _require(all(summary.get(key) == value for key, value in summaries.items()), "Saved evaluation controller summary differs")
    labels = {"initial_states": states, "independent_changed_states": len(states), "control_steps": controls,
        "policy_unchanged": True, "selected_feedback_kind": actor.selection_kind(policy),
        "nonzero_coefficients": int(np.count_nonzero(weights)), "paired_vs_authored": _comparison(evaluation),
        "comparison_complete": all(row["initial_state_sha256"] is not None
            for summary in evaluation["controllers"].values() for row in summary["trials"]), "physical_validation": "unverified"}
    _require(len(set(states.values())) == len(states) and controls <= freeze["maximum_control_steps"]
             and all(evaluation.get(key) == value for key, value in labels.items()), "Saved evaluation labels or counters differ")
    return {"verified": True, "profile": EVALUATION_PROFILE, "policy_sha256": freeze["policy_sha256"],
            "control_steps": controls, "independent_changed_states": len(states), "comparison_complete": labels["comparison_complete"],
            "native_model_readmission": False, "scope": "Saved matched-start/search/custody consistency, not physical validation"}
