"""Post-training feasibility selection from sealed residual candidates; no physics or evaluation input."""

from copy import deepcopy
import math
from pathlib import Path
import re
import time

import numpy as np

from weft.io import canonical_hash, local_time
from weft.shared_exercise_lab import _json, _validate_policy_package, _verified_inputs
from weft.shared_exercise_policies import selection_kind
from weft.shared_exercise_residual_learning import (
    ALGORITHM, BASIS_PROFILE, MAX_CONTROL_STEPS, MAX_OUTPUT_BYTES, MAX_OUTPUT_FILES, RESIDUAL_BOUND,
    SCOPE, SOURCE_VIEWS, TRAIN_RESETS, _bytes, _check, _coefficients, _failure, _input_unchanged,
    _integer, _output_path, _runtime_hashes, _score, _seal, _seconds, _seed_plan, _sha,
    validate_shared_exercise_residual_policy,
)
from weft.shared_exercise_reset import describe_reset_protocol, resolve_reset_options
from weft.shared_exercise_task import GATES, GATES_PROVENANCE, GATES_SHA256, GATE_PROFILE, PROFILE
from weft.shared_world import validate_shared_world
from weft.shared_world_packaging import collect_shared_source_files, read_package_file, read_package_manifest, write_package_files


SELECTION_PROFILE = "shared-exercise-training-feasibility-selection-v1"
RULE = "Highest original mean reward among candidates passing every frozen gate on both original training starts; otherwise retain original selection"
ARCHIVE = "selection-source"
SELECTOR_PATH = "selection-implementation/shared_exercise_selection.py"
REPLACEMENTS = {name: f"{ARCHIVE}/{name}" for name in ("policy.json", "training.json", "settings.json")}
HEX = re.compile(r"[0-9a-f]{64}")
BODY_GATES = ("angular_rmse_rad", "root_rmse_model_units", "peak_angular_rmse_rad",
              "peak_root_rmse_model_units", "max_floor_penetration_model_units")
EQUIPMENT_GATES = ("displacement_model_units", "orientation_rad", "speed_model_units_s", "angular_speed_rad_s")


def _finite(value, name, low=-math.inf, high=math.inf):
    if type(value) not in (int, float) or not math.isfinite(value) or not low <= value <= high:
        raise ValueError(f"Selection requires bounded finite {name}")
    return value


def _same_number(actual, expected, name, tolerance=1e-9):
    _finite(actual, name)
    if not math.isclose(actual, expected, rel_tol=0, abs_tol=tolerance):
        raise ValueError(f"Selection {name} differs from saved accounting")


def _hash(value, name):
    if not isinstance(value, str) or HEX.fullmatch(value) is None:
        raise ValueError(f"Selection requires an exact {name} SHA256")


def _metrics(metrics, scene, task, policy, reset, *, initial=False):
    fixed = {"profile": PROFILE, "scene_sha256": policy["scene_sha256"], "task_sha256": policy["task_sha256"],
             "evaluator_sha256": policy["evaluator_sha256"], "gates_sha256": GATES_SHA256,
             "gate_profile": GATE_PROFILE, "gates_provenance": GATES_PROVENANCE, "source_scope": task["source_scope"],
             "root_assistance": task["control"]["root_assistance"], "virtual_object_forces": "forbidden",
             "reset_protocol": describe_reset_protocol(reset)}
    numeric = {"simulated_duration_s", "valid_observed_duration_s", "declared_duration_s", "joint_effort_mean_squared",
               "root_assistance_mean_squared", "reward_integral", "mean_reward", "failure_reward_penalty"}
    keys = {*fixed, *numeric, *BODY_GATES, "completed_interval", "success", "failure", "failed_gates", "observed_substeps",
            "valid_substeps", "equipment", "contact_requirements", "clock_status"}
    if not isinstance(metrics, dict) or set(metrics) != keys or any(metrics[key] != value for key, value in fixed.items()):
        raise ValueError("Selection metrics changed task, reset, evaluator or gate identities")
    duration = task["interval"]["horizon_steps"] * scene["dt"]
    maximum = task["interval"]["horizon_steps"] * scene["substeps"]
    observed, valid = metrics["observed_substeps"], metrics["valid_substeps"]
    _integer(observed, "observed substeps", 0, maximum)
    _integer(valid, "valid substeps", 0, observed)
    for name in numeric:
        _finite(metrics[name], name)
    _same_number(metrics["declared_duration_s"], duration, "declared duration")
    tick = scene["dt"] / scene["substeps"]
    _same_number(metrics["valid_observed_duration_s"], valid * tick, "valid duration", 1e-8)
    if not valid * tick - 1e-8 <= metrics["simulated_duration_s"] <= observed * tick + 1e-8:
        raise ValueError("Selection elapsed time differs from observed substeps")
    failure = metrics["failure"]
    if failure is not None and (not isinstance(failure, str) or not 1 <= len(failure) <= 240):
        raise ValueError("Selection failure must remain a bounded explicit reason")
    if ((valid < observed or metrics["clock_status"] != "observed") and failure is None
            or metrics["clock_status"] not in {"observed", "invalid_substep_clock", "unavailable_after_last_observed_time"}):
        raise ValueError("Selection cannot conceal missing substeps or failed clocks")
    failures = [failure] if failure else []
    for name in BODY_GATES:
        value = metrics[name]
        if value is None:
            if valid or name not in {"angular_rmse_rad", "root_rmse_model_units"}:
                raise ValueError("Selection tracking metrics are missing after observed substeps")
        else:
            _finite(value, name, 0.)
        if value is None or value > GATES[name] + 1e-12:
            failures.append(name)
    for name, peak in (("angular_rmse_rad", "peak_angular_rmse_rad"), ("root_rmse_model_units", "peak_root_rmse_model_units")):
        if metrics[name] is not None and metrics[name] > metrics[peak] + 1e-8:
            raise ValueError("Selection mean tracking error exceeds its recorded peak")
    equipment = metrics["equipment"]
    ids = [obj["id"] for obj in scene["objects"] if obj["role"] != "static"]
    if (not isinstance(equipment, list) or any(not isinstance(row, dict) for row in equipment)
            or [row.get("object_id") for row in equipment] != ids):
        raise ValueError("Selection metrics omit or reorder equipment")
    for row in equipment:
        if set(row) != {*EQUIPMENT_GATES, "object_id", "within_limits"}:
            raise ValueError("Selection equipment metric fields differ")
        for name in EQUIPMENT_GATES:
            _finite(row[name], name, 0.)
        passed = all(row[name] <= GATES["equipment_" + name] + 1e-12 for name in EQUIPMENT_GATES)
        if type(row["within_limits"]) is not bool or row["within_limits"] != passed:
            raise ValueError("Selection equipment success disagrees with frozen gates")
        if not passed:
            failures.append("equipment_stability:" + row["object_id"])
    contacts = metrics["contact_requirements"]
    if not isinstance(contacts, list) or len(contacts) != len(task["contact_requirements"]):
        raise ValueError("Selection metrics omit contact requirements")
    measures = {"required_duration_s", "observed_duration_s", "qualifying_duration_s", "current_gap_s", "max_gap_s",
                "load_body_weight_fraction", "qualifying_fraction"}
    for index, (row, required) in enumerate(zip(contacts, task["contact_requirements"], strict=True)):
        if (not isinstance(row, dict) or set(row) != set(required) | measures | {"qualifying_now", "active_now", "satisfied"}
                or any(row[key] != value for key, value in required.items())
                or any(type(row[key]) is not bool for key in ("qualifying_now", "active_now", "satisfied"))):
            raise ValueError("Selection contact metadata differs from the declared task")
        for name in measures:
            _finite(row[name], name, 0.)
        required_duration = (required["end_step"] - required["start_step"]) * scene["dt"]
        _same_number(row["required_duration_s"], required_duration, "contact duration")
        fraction = row["qualifying_duration_s"] / required_duration
        _same_number(row["qualifying_fraction"], fraction, "contact qualifying fraction", 1e-8)
        if (not 0 <= row["qualifying_duration_s"] <= row["observed_duration_s"] + 1e-8 <= required_duration + 2e-8
                or not row["current_gap_s"] <= row["max_gap_s"] + 1e-8 <= row["observed_duration_s"] + 2e-8):
            raise ValueError("Selection contact interval accounting differs")
        passed = (math.isclose(row["observed_duration_s"], required_duration, abs_tol=1e-8)
                  and fraction >= GATES["contact_duration_fraction"] - 1e-12
                  and row["max_gap_s"] <= GATES["max_contact_gap_s"] + 1e-12)
        if row["satisfied"] != passed:
            raise ValueError("Selection contact success ignores frozen coverage or gap gates")
        if not passed:
            failures.append("contact_requirement:" + str(index))
    completed = observed == valid == maximum and failure is None
    if (type(metrics["completed_interval"]) is not bool or metrics["completed_interval"] != completed
            or type(metrics["success"]) is not bool or metrics["success"] != (completed and not failures)
            or metrics["failed_gates"] != failures):
        raise ValueError("Selection success must follow every existing whole-interval gate")
    penalty = -1. if failure else 0.
    if metrics["failure_reward_penalty"] != penalty:
        raise ValueError("Selection failure penalty differs from the frozen task")
    for name in ("joint_effort_mean_squared", "root_assistance_mean_squared"):
        _finite(metrics[name], name, 0., 1. + 1e-9)
    if task["control"]["root_assistance"] == "disabled" and metrics["root_assistance_mean_squared"] != 0:
        raise ValueError("Selection cannot admit forbidden root effort")
    _same_number(metrics["mean_reward"], metrics["reward_integral"] / duration, "mean reward")
    if not -.03 * valid * tick + penalty * scene["dt"] - 1e-8 <= metrics["reward_integral"] <= valid * tick + penalty * scene["dt"] + 1e-8:
        raise ValueError("Selection reward exceeds the unchanged task's possible range")
    if initial and (observed != 0 or valid != 0 or failure is not None or metrics["clock_status"] != "observed"
                    or any(metrics[name] != 0 for name in ("reward_integral", "joint_effort_mean_squared", "root_assistance_mean_squared"))):
        raise ValueError("Selection initial metrics must precede all control actions")
    if initial:
        scale = scene["coordinate_frame"]["scale_bridge"]["body_height_to_world_scale"]
        objects = {obj["id"]: obj for obj in scene["objects"]}
        for row in equipment:
            obj = objects[row["object_id"]]
            for name, expected in (("displacement_model_units", 0.),
                    ("speed_model_units_s", float(np.linalg.norm(obj["velocity"])) / scale),
                    ("angular_speed_rad_s", float(np.linalg.norm(obj["angular_velocity"])))):
                _same_number(row[name], expected, "initial equipment " + name)
            # Same-quaternion acos has roundoff; the engine may also normalize the declared quaternion.
            self_angle = 2 * math.acos(float(np.clip(np.dot(obj["quaternion"], obj["quaternion"]), 0, 1)))
            if row["orientation_rad"] > self_angle + 1e-7:
                raise ValueError("Selection initial equipment orientation differs from its reset reference")
        if any(metrics[name] != 0 for name in ("peak_angular_rmse_rad", "peak_root_rmse_model_units")):
            raise ValueError("Selection initial tracking peaks must precede all control actions")


def _trial(row, scene, task, policy, seed, reset):
    base = {"seed", "reset_options", "initial_state_sha256", "status", "control_steps", "return", "metrics"}
    if (not isinstance(row, dict) or row.get("seed") != seed or type(row.get("seed")) is not int
            or row.get("reset_options") != reset
            or resolve_reset_options(row.get("reset_options")) != row.get("reset_options")):
        raise ValueError("Selection trials must retain both original matched seeds and resets")
    _finite(row.get("return"), "trial return")
    if row.get("status") == "reset_failed":
        if (set(row) != base | {"error"} or row["initial_state_sha256"] is not None or row["metrics"] is not None
                or type(row["control_steps"]) is not int or row["control_steps"] != 0 or row["return"] != 0
                or not isinstance(row["error"], str) or not 1 <= len(row["error"]) <= 2000):
            raise ValueError("Selection reset failure cannot hide issued actions or fabricated metrics")
        return
    if set(row) != base | {"initial_metrics", "terminated", "truncated", "final_snapshot_sha256"}:
        raise ValueError("Selection requires complete bounded training trial metadata")
    _hash(row["initial_state_sha256"], "initial state")
    _hash(row["final_snapshot_sha256"], "final snapshot")
    horizon = task["interval"]["horizon_steps"]
    _integer(row["control_steps"], "control steps", 1, horizon)
    _metrics(row["initial_metrics"], scene, task, policy, reset, initial=True)
    _metrics(row["metrics"], scene, task, policy, reset)
    metrics, initial = row["metrics"], row["initial_metrics"]
    if (not metrics["completed_interval"] and metrics["failure"] is None
            or row["status"] != ("completed" if metrics["completed_interval"] else "failed")
            or metrics["completed_interval"] and row["control_steps"] != horizon
            or not (row["control_steps"] - 1) * scene["substeps"] <= metrics["observed_substeps"] <= row["control_steps"] * scene["substeps"]):
        raise ValueError("Selection cannot treat a task prefix as a completed trial")
    expected_terminated = bool(metrics["failure"] or metrics["success"])
    if (type(row["terminated"]) is not bool or row["terminated"] != expected_terminated
            or type(row["truncated"]) is not bool or row["truncated"] != (metrics["completed_interval"] and not expected_terminated)):
        raise ValueError("Selection terminal flags disagree with task outcomes")
    _same_number(row["return"], metrics["reward_integral"] / scene["dt"], "trial return", 1e-8)
    for old, new in zip(initial["equipment"], metrics["equipment"], strict=True):
        if any(new[name] < old[name] - 1e-12 for name in EQUIPMENT_GATES):
            raise ValueError("Selection equipment maxima cannot decrease from reset")


def _history(files, scene, task, policy, deadline):
    settings, training = _json(files["settings.json"]), _json(files["training.json"])
    generations, population = settings.get("generations"), settings.get("population")
    _integer(generations, "generations", 1, 6)
    _integer(population, "population", 4, 12)
    _seconds(settings.get("max_seconds"), 60)
    seed = policy["training_seed"]
    seeds = _seed_plan(seed, 2)
    horizon = task["interval"]["horizon_steps"]
    maximum = generations * population * 2 * horizon
    if maximum > MAX_CONTROL_STEPS:
        raise ValueError("Selection input exceeds the original full-interval action budget")
    expected = {"schema_version": 1, "algorithm": ALGORITHM, "seed": seed,
        "scene_sha256": policy["scene_sha256"], "task_sha256": policy["task_sha256"], "gates_sha256": GATES_SHA256,
        "runtime_sha256": policy["runtime_sha256"], "source_manifest_sha256": _sha(files["input-manifest.json"]),
        "training_seeds": seeds, "training_reset_options": list(TRAIN_RESETS), "elite_count": max(2, population // 4),
        "initial_mean": 0., "initial_std": .08, "minimum_std": .02, "residual_bound": RESIDUAL_BOUND,
        "coefficient_shape": [2, 19], "basis_profile": BASIS_PROFILE, "maximum_control_steps": maximum,
        "maximum_physics_substeps": maximum * scene["substeps"], "workers": 1, "external_api_usd": 0,
        "horizon_steps": horizon, "root_residual_channels": [], "equipment_force_channels": [],
        "selection_objective": "mean(task mean_reward -1 for incomplete intervals); reset failure score=-2; loss=-score",
        "candidate_anchors": ["zero_residual_authored", "previous_best_after_first_generation"],
        "action_accounting": "All anchors and candidates are in population; identical cached trials issue zero new actions; no retries",
        "source_scope": task["source_scope"], "scope": SCOPE,
        "source_views_sha256": {name: _sha(files[name]) for name in sorted(SOURCE_VIEWS & set(files))}}
    if set(settings) != set(expected) | {"created_at", "generations", "population", "max_seconds"} or any(settings.get(k) != v for k, v in expected.items()):
        raise ValueError("Selection input search settings differ from the frozen generator contract")
    for name, value in expected.items():
        if type(value) is int:
            _integer(settings[name], name, value, value)
        elif type(value) is float:
            _finite(settings[name], name, value, value)
    for reset in settings["training_reset_options"]:
        resolve_reset_options(reset)
    _finite(training.get("seconds"), "training seconds", 0., settings["max_seconds"])
    training_keys = {"status", "history", "completed_generations", "evaluated_episodes", "control_steps", "cache_hits",
                     "active_candidates", "failure", "seconds", "policy_sha256", "selected_authored_baseline",
                     "selected_coast_baseline", "improvement_over_authored_training", "training_score_improved_over_authored",
                     "selected_feedback_kind"}
    if (set(training) != training_keys or training.get("status") != "trained" or training.get("failure") is not None
            or training.get("active_candidates") != [] or type(training.get("completed_generations")) is not int
            or training.get("completed_generations") != generations or not isinstance(training.get("history"), list)
            or len(training["history"]) != generations):
        raise ValueError("Selection requires an original completed training history, not a derived or partial checkpoint")
    rng = np.random.default_rng(seed)
    mean, std, best, best_score = np.zeros((2, 19)), np.full((2, 19), .08), np.zeros((2, 19)), -math.inf
    unique, states, initial_metrics, episodes, steps, hits = {}, {}, {}, 0, 0, 0
    for generation, record in enumerate(training["history"]):
        _check(deadline)
        if (not isinstance(record, dict) or set(record) != {"generation", "candidates", "best_training_score", "next_mean", "next_std"}
                or type(record["generation"]) is not int or record["generation"] != generation
                or not isinstance(record["candidates"], list) or len(record["candidates"]) != population):
            raise ValueError("Selection candidate generations are incomplete or reordered")
        samples = np.clip(rng.normal(mean, std, (population, 2, 19)), -RESIDUAL_BOUND, RESIDUAL_BOUND)
        samples[0] = 0.
        if generation:
            samples[1] = best
        for index, row in enumerate(record["candidates"]):
            _check(deadline)
            keys = {"candidate", "residual_coefficients", "trials", "complete", "score", "loss", "cached"}
            if (not isinstance(row, dict) or set(row) != keys or type(row["candidate"]) is not int or row["candidate"] != index
                    or row["complete"] is not True or type(row["cached"]) is not bool
                    or not isinstance(row["trials"], list) or len(row["trials"]) != 2):
                raise ValueError("Selection candidate requires exactly two complete original training trials")
            coefficients = _coefficients(row["residual_coefficients"])
            key = canonical_hash(row["residual_coefficients"])
            if key != canonical_hash(samples[index].tolist()):
                raise ValueError("Selection candidate coefficients differ from the frozen CEM generator")
            if key in unique:
                original = unique[key]["row"]
                if row["cached"] is not True or any(row[k] != original[k] for k in keys - {"candidate", "cached"}):
                    raise ValueError("Selection cached candidate differs from its original trials")
                hits += 1
            else:
                if row["cached"]:
                    raise ValueError("Selection cached candidate has no earlier evaluated identity")
                for trial_seed, reset, trial in zip(seeds, TRAIN_RESETS, row["trials"], strict=True):
                    _trial(trial, scene, task, policy, trial_seed, reset)
                    if trial["initial_state_sha256"] is not None:
                        if states.setdefault(trial_seed, trial["initial_state_sha256"]) != trial["initial_state_sha256"]:
                            raise ValueError("Selection candidates did not use matched original initial states")
                        if initial_metrics.setdefault(trial_seed, trial["initial_metrics"]) != trial["initial_metrics"]:
                            raise ValueError("Selection initial metrics differ across matched candidates")
                    episodes += 1
                    steps += trial["control_steps"]
                unique[key] = {"generation": generation, "candidate": index, "coefficients_sha256": key, "row": row}
            _same_number(row["score"], _score(row["trials"]), "candidate score", 0.)
            _same_number(row["loss"], -row["score"], "candidate loss", 0.)
            if row["score"] > best_score:
                best, best_score = coefficients.copy(), row["score"]
        _same_number(record["best_training_score"], best_score, "generation best score", 1e-12)
        distinct = {canonical_hash(row["residual_coefficients"]): row for row in record["candidates"]}
        elites = sorted(distinct.values(), key=lambda row: -row["score"])[:settings["elite_count"]]
        elite = np.asarray([row["residual_coefficients"] for row in elites])
        mean, std = elite.mean(axis=0), np.maximum(elite.std(axis=0), .02)
        if record["next_mean"] != mean.tolist() or record["next_std"] != std.tolist():
            raise ValueError("Selection recorded CEM updates differ from the saved reward history")
    if len(set(states.values())) != len(states):
        raise ValueError("Selection original training starts are not distinct")
    for name, count in (("evaluated_episodes", episodes), ("control_steps", steps), ("cache_hits", hits)):
        if type(training.get(name)) is not int or training[name] != count:
            raise ValueError("Selection training action/episode/cache accounting differs")
    if steps > maximum or policy["residual_coefficients"] != best.tolist() or policy["training_score"] != best_score:
        raise ValueError("Selection original policy is not the reward-selected candidate from its exact history")
    authored = unique[canonical_hash(np.zeros((2, 19)).tolist())]["row"]["score"]
    _same_number(training["improvement_over_authored_training"], best_score - authored, "original authored improvement", 1e-12)
    return settings, training, unique, authored


def select_feasible_shared_exercise_policy(learning_dir, out, *, max_seconds=15):
    """Select a saved training-feasible challenger; do not retrain, simulate or read evaluation results."""
    _seconds(max_seconds, 15)
    source, out = Path(learning_dir).resolve(), _output_path(learning_dir, out)
    started = time.monotonic()
    deadline = started + max_seconds
    manifest, names = read_package_manifest(source)
    reserved = {"selection.json", "selection-source", "selection-implementation"}
    if (_json(manifest).get("status") != "trained"
            or any(name.split("/")[0] in reserved or any(part in {"evaluation.json", "pre-evaluation-freeze.json"}
                   or part.startswith("policy-evaluation") for part in Path(name).parts) for name in names)):
        raise ValueError("Selection accepts only original trained input, never evaluation or a previous selection")
    raw_manifest, files = _verified_inputs(source, deadline=deadline)
    if raw_manifest != manifest:
        raise ValueError("Selection input manifest changed during verification")
    required = {"scene.json", "exercise-task.json", "policy.json", "training.json", "settings.json", "input-manifest.json"}
    if not required <= set(files) or any(len(files[name]) > 8 * 1024 * 1024 for name in ("training.json", "settings.json")):
        raise ValueError("Selection requires bounded complete original training metadata")
    scene, task = validate_shared_world(_json(files["scene.json"])), _json(files["exercise-task.json"])
    policy, _ = _validate_policy_package(manifest, files, scene, task)
    validate_shared_exercise_residual_policy(policy, scene, task)
    collect_shared_source_files(scene, source, expected_manifest_sha256=_sha(manifest))
    _, training, candidates, authored_score = _history(files, scene, task, policy, deadline)
    original = candidates[canonical_hash(policy["residual_coefficients"])]
    feasible = [row for row in candidates.values() if all(trial["metrics"] is not None and trial["metrics"]["success"]
                                                        for trial in row["row"]["trials"])]
    chosen = max(feasible, key=lambda value: value["row"]["score"]) if feasible else original
    changed = chosen["coefficients_sha256"] != original["coefficients_sha256"]
    selected = deepcopy(policy)
    selected.update(residual_coefficients=deepcopy(chosen["row"]["residual_coefficients"]), training_score=chosen["row"]["score"])
    validate_shared_exercise_residual_policy(selected, scene, task)
    policy_raw = _bytes(selected) if changed else files["policy.json"]
    status = "selected_feasible_candidate" if changed else "retained_original_feasible" if feasible else "retained_original_no_feasible_candidate"
    selector_raw = read_package_file(Path(__file__).parent, Path(__file__).name, max_bytes=8 * 1024 * 1024)
    identity = {key: chosen[key] for key in ("generation", "candidate", "coefficients_sha256")}
    selection = {"schema_version": 1, "profile": SELECTION_PROFILE, "created_at": local_time(), "rule": RULE,
        "selection_status": status, "input_manifest_sha256": _sha(manifest),
        "original_policy_sha256": _sha(files["policy.json"]), "policy_sha256": _sha(policy_raw),
        "original_candidate": {key: original[key] for key in identity}, "selected_candidate": identity,
        "candidate_indices": "zero_based_first_evaluated_occurrence", "unique_candidates": len(candidates),
        "feasible_candidates": len(feasible), "training_seeds": _json(files["settings.json"])["training_seeds"],
        "training_reset_options": list(TRAIN_RESETS), "original_training_score": policy["training_score"],
        "selected_training_score": selected["training_score"], "authored_training_score": authored_score,
        "score_change_from_original_selection": selected["training_score"] - policy["training_score"],
        "generator_algorithm": ALGORITHM, "generator_runtime_sha256": deepcopy(policy["runtime_sha256"]),
        "selector_source_path": SELECTOR_PATH, "selector_source_sha256": _sha(selector_raw),
        "original_manifest_path": f"{ARCHIVE}/manifest.json", "original_file_replacements": REPLACEMENTS,
        "history_unchanged": True, "settings_unchanged": True, "new_training_episodes": 0, "new_control_steps": 0,
        "evaluation_used": False, "independent_evaluation_status": "not_performed_for_this_selection",
        "max_seconds": max_seconds, "seconds": time.monotonic() - started, "external_api_usd": 0,
        "scope": "Post-training choice using recorded original training metrics, not new optimization or independent physical replay. "
                 "Original reward history and rejected candidates remain preserved; training feasibility is not generalization or physical success."}
    derived = deepcopy(training)
    kind, authored, coast = selection_kind(selected)
    improvement = selected["training_score"] - authored_score
    derived.update(policy_sha256=_sha(policy_raw), selected_feedback_kind=kind, selected_authored_baseline=authored,
        selected_coast_baseline=coast, improvement_over_authored_training=improvement,
        training_score_improved_over_authored=improvement > 0,
        post_training_selection={"profile": SELECTION_PROFILE, "rule": RULE, "selection_status": status,
                                 "input_manifest_sha256": _sha(manifest), "original_policy_sha256": selection["original_policy_sha256"]})
    output = {**files, "policy.json": policy_raw, "training.json": _bytes(derived), "selection.json": _bytes(selection),
              f"{ARCHIVE}/manifest.json": manifest, **{path: files[name] for name, path in REPLACEMENTS.items()},
              SELECTOR_PATH: selector_raw}
    if len(output) >= MAX_OUTPUT_FILES or sum(map(len, output.values())) > MAX_OUTPUT_BYTES - 1024 * 1024:
        raise ValueError("Selection custody exceeds the standard trained-package file/byte bounds")
    _check(deadline)
    out.mkdir(parents=True, mode=0o700, exist_ok=False)
    try:
        write_package_files(out, output)
        _input_unchanged(source, manifest, deadline)
        if _runtime_hashes() != policy["runtime_sha256"]:
            raise ValueError("Selection cannot change the original actor runtime")
        read_package_file(Path(__file__).parent, Path(__file__).name, max_bytes=8 * 1024 * 1024, expected_sha256=_sha(selector_raw))
        manifest_hash = _seal(out, "trained", deadline)
    except Exception as error:
        _failure(out, "selection.json", selection, error)
        raise
    return {"directory": str(out), "manifest_sha256": manifest_hash, "policy_sha256": _sha(policy_raw),
            "selection_sha256": _sha(output["selection.json"]), "selection_status": status, "selected_candidate": identity,
            "feasible_candidates": len(feasible), "unique_candidates": len(candidates),
            "original_policy_sha256": selection["original_policy_sha256"], "seconds": time.monotonic() - started,
            "new_training_episodes": 0, "new_control_steps": 0}
