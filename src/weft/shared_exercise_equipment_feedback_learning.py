"""Bounded full-interval CEM for explicit equipment feedback; no evaluation or legacy dispatch."""

import base64
from copy import deepcopy
import json
import math
from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np

from weft import shared_exercise_equipment_feedback as actor
from weft import shared_exercise_equipment_feedback_policy as policy_module
from weft import shared_exercise_feedback_learning as legacy
from weft import shared_exercise_feedback_residual as base_actor
from weft import shared_exercise_residual_learning as residual
from weft.io import canonical_hash, digest, local_time
from weft.shared_exercise_feasibility_learning import RULE, _rank_key, optimization_rank
from weft.shared_exercise_learning import _bytes, _integer, _output_path, _score, _seconds, _seed_plan, _sha, _source
from weft.shared_exercise_lab import _verified_inputs
from weft.shared_exercise_reset import describe_reset_protocol
from weft.shared_exercise_selection import BODY_GATES, EQUIPMENT_GATES, _metrics as validate_metrics, _trial as validate_training_trial
from weft.shared_exercise_task import GATE_PROFILE, GATES_PROVENANCE, GATES_SHA256, SharedExerciseTaskEnv, prepare_shared_exercise_snapshot_validator
from weft.shared_world_packaging import collect_shared_source_files, read_package_file, read_package_manifest


TRAIN_RESETS = residual.TRAIN_RESETS
MAX_CONTROL_STEPS = residual.MAX_CONTROL_STEPS
MAX_DIAGNOSTIC_BYTES = legacy.MAX_DIAGNOSTIC_BYTES
MAX_DIAGNOSTIC_FILE_BYTES = legacy.MAX_DIAGNOSTIC_FILE_BYTES
MAX_DIAGNOSTIC_LINE_BYTES = 32 * 1024
DIAGNOSTIC_PROFILE = "shared-exercise-equipment-control-journal-v1"
_STATIC_METRIC_KEYS = ("profile", "scene_sha256", "task_sha256", "evaluator_sha256", "gates_sha256",
                       "gate_profile", "gates_provenance", "source_scope", "root_assistance", "virtual_object_forces",
                       "reset_protocol", "declared_duration_s")
_FLOAT_METRICS = ("simulated_duration_s", "valid_observed_duration_s", *BODY_GATES,
                  "joint_effort_mean_squared", "root_assistance_mean_squared", "reward_integral", "mean_reward",
                  "failure_reward_penalty")
_FORBIDDEN_PARTS = {"policy.json", "training.json", "evaluation.json", "pre-evaluation-freeze.json", "exercise-lab.json"}
_NORMALIZERS = {"proximity_distance_model_units": actor.PROXIMITY_DISTANCE_MODEL_UNITS,
                "joint_speed_rad_s": actor.JOINT_SPEED_RAD_S, "closing_speed_model_units_s": actor.CLOSING_SPEED_MODEL_UNITS_S}


def _check(deadline):
    if time.monotonic() >= deadline:
        raise TimeoutError("Equipment feedback reached its cooperative work deadline")


def _compact(value):
    return (json.dumps(value, allow_nan=False, separators=(",", ":")) + "\n").encode()


def _pack(value):
    if (type(value) is not np.ndarray or value.dtype not in (np.dtype("float32"), np.dtype("float64"))
            or value.size > 360 or not np.isfinite(value).all()):
        raise ValueError("Journal arrays require bounded finite native float32/float64 values")
    return {"dtype": value.dtype.str, "data": base64.b64encode(value.tobytes()).decode("ascii")}


def _unpack(value, shape):
    if type(value) is not dict or set(value) != {"dtype", "data"} or value["dtype"] not in ("<f4", "<f8"):
        raise ValueError("Journal array encoding differs from the fixed schema")
    dtype = np.dtype(value["dtype"])
    size = math.prod(shape) * dtype.itemsize
    if type(value["data"]) is not str or len(value["data"]) != 4 * ((size + 2) // 3):
        raise ValueError("Journal array exceeds its exact shape/byte bound")
    try:
        raw = base64.b64decode(value["data"], validate=True)
    except (ValueError, base64.binascii.Error) as error:
        raise ValueError("Journal array is not strict base64") from error
    if len(raw) != size:
        raise ValueError("Journal array byte count differs")
    result = np.frombuffer(raw, dtype=dtype).reshape(shape)
    if not np.isfinite(result).all():
        raise ValueError("Journal array contains nonfinite values")
    return result


def diagnostic_schema(scene, task):
    return {"profile": DIAGNOSTIC_PROFILE, "arrays": {"observation": [156 + 17 * len(task["equipment_ids"])],
            "base_features": [22, 3], "base_action": [22], "action": [22], "actor_values": [96]},
            "encoding": "strict_base64_native_little_endian_float_bytes_no_quantization",
            "actor_value_columns": ["features_row_major:44", "requested_residual:22", "applied_residual:22",
                                    "center_distance_model_units:1", "away_world_axes:3", "proximity:1",
                                    "closing_speed_model_units_s:1", "normalized_closing_speed:1", "max_abs_residual:1"],
            "equipment_ids": list(task["equipment_ids"]), "static_metric_keys": list(_STATIC_METRIC_KEYS),
            "actor_contract": policy_module.feature_contract(), "source_clock_scope":
            "Boundary simulation/reference times; immutable construction retains original source timestamp bindings",
            "count_scope": "Journal action intents precede admission; trial control_steps and returned_controls count actual step calls",
            "reference_start_s": scene["human_spec"]["reference"][0]["time_s"] + task["interval"]["start_offset_s"]}


def _static_metrics(scene, task, evaluator, reset):
    return {"profile": task["profile"], "scene_sha256": canonical_hash(scene), "task_sha256": canonical_hash(task),
            "evaluator_sha256": evaluator, "gates_sha256": GATES_SHA256, "gate_profile": GATE_PROFILE,
            "gates_provenance": GATES_PROVENANCE, "source_scope": deepcopy(task["source_scope"]),
            "root_assistance": "explicit_3_axis", "virtual_object_forces": "forbidden",
            "reset_protocol": describe_reset_protocol(reset),
            "declared_duration_s": task["interval"]["horizon_steps"] * scene["dt"]}


def _actor_report(values, observation, action_dtype, weights, task, nearest, direction_known, changed):
    return {"profile": actor.PROFILE, "status": "sampled", "scope": actor.SCOPE,
            "required_base_profile": base_actor.SPLIT_POLICY_PROFILE, "pairing": "caller_responsibility_not_authenticated",
            "action_dtype": action_dtype, "groups": list(actor.GROUP_NAMES), "coefficients": weights.tolist(),
            "zero_coefficients": not bool(np.any(weights)), "observation_size": len(observation),
            "contact_requirement_count": 0, "reference_phase": float(observation[154]),
            "elapsed_fraction": float(observation[-1]), "normalizers": dict(_NORMALIZERS),
            "nearest": {"id": task["equipment_ids"][nearest], "index": nearest,
                        "center_distance_model_units": float(values[88]), "away_direction_known": direction_known,
                        "away_world_axes": values[89:92].tolist(), "proximity": float(values[92]),
                        "closing_speed_model_units_s": float(values[93]), "normalized_closing_speed": float(values[94])},
            "features": values[:44].reshape(22, 2).tolist(), "requested_residual": values[44:66].tolist(),
            "applied_residual": values[66:88].tolist(), "max_abs_residual": float(values[95]), "changed_channels": changed}


def _attempt_record(index, scene, task, weights, action, sample):
    report, near = sample["actor"], sample["actor"]["nearest"]
    values = np.asarray([*np.asarray(report["features"]).ravel(), *report["requested_residual"], *report["applied_residual"],
        near["center_distance_model_units"], *near["away_world_axes"], near["proximity"], near["closing_speed_model_units_s"],
        near["normalized_closing_speed"], report["max_abs_residual"]], dtype=float)
    if report != _actor_report(values, sample["observation"], action.dtype.str, weights, task,
                               near["index"], near["away_direction_known"], report["changed_channels"]):
        raise ValueError("Journal schema would omit or alter an actor diagnostic value")
    boundary = (index - 1) * scene["dt"]
    return {"type": "attempt", "index": index, "simulation_time_s": boundary,
            "reference_time_s": scene["human_spec"]["reference"][0]["time_s"] + task["interval"]["start_offset_s"] + boundary,
            **{name: _pack(sample[name]) for name in ("observation", "base_features", "base_action")},
            "action": _pack(action), "actor_values": _pack(values), "nearest_index": near["index"],
            "away_direction_known": near["away_direction_known"], "changed_channels": report["changed_channels"]}


def journal_reservation(scene, task, generations, population):
    """Conservative encoded-byte bound, evaluated on metadata before any native admission."""
    _integer(generations, "generations", 1, 2)
    _integer(population, "population", 4, 8)
    horizon, substeps = task["interval"]["horizon_steps"], scene["substeps"]
    _integer(horizon, "full task horizon", 1, MAX_CONTROL_STEPS)
    _integer(substeps, "physics substeps", 1, 64)
    ids = task["equipment_ids"]
    if (type(ids) is not list or not 1 <= len(ids) <= 12 or task["contact_requirements"] != []
            or any(type(name) is not str or not 1 <= len(name) <= 160 for name in ids)):
        raise ValueError("Journal reservation requires the bounded no-contact equipment layout")
    episodes, controls = generations * population * 2, generations * population * 2 * horizon
    if controls > MAX_CONTROL_STEPS:
        raise ValueError("Equipment training exceeds 8640 full-interval controls; no prefix shortening")
    # Quoted 32-character placeholders exceed every finite JSON float representation.
    number = "n" * 32
    attempt = {"type": "attempt", "index": 2**31 - 1, "simulation_time_s": number, "reference_time_s": number,
               **{name: {"dtype": "<f8", "data": "x" * (4 * ((math.prod(shape) * 8 + 2) // 3))}
                  for name, shape in diagnostic_schema(scene, task)["arrays"].items()},
               "nearest_index": 11, "away_direction_known": False, "changed_channels": list(range(22))}
    dynamic = {**dict.fromkeys(_FLOAT_METRICS, number), "observed_substeps": 2**31 - 1, "valid_substeps": 2**31 - 1,
               "completed_interval": False, "success": False, "failure": None,
               "failed_gates": [*BODY_GATES, *("equipment_stability:" + name for name in ids)],
               "clock_status": "observed", "contact_requirements": [], "equipment": [
                   {"object_id": name, **dict.fromkeys(EQUIPMENT_GATES, number), "within_limits": False} for name in ids],
               "observation_status": "current", "legacy_coupled_score_used": False}
    result = {"type": "result", "index": 2**31 - 1, "reward": number, "terminated": False, "truncated": False,
              "info": dynamic}
    pair = len(_compact(attempt)) + len(_compact(result))
    # One terminal failure per episode may add two escaped 240-character reasons and a longer clock label.
    failure_allowance = 8192
    file_bytes = horizon * pair + failure_allowance
    total = episodes * file_bytes
    if (max(len(_compact(attempt)), len(_compact(result))) + failure_allowance > MAX_DIAGNOSTIC_LINE_BYTES
            or file_bytes > MAX_DIAGNOSTIC_FILE_BYTES or total > MAX_DIAGNOSTIC_BYTES):
        raise ValueError("Projected complete-interval journal exceeds its fixed byte budget before native work")
    return {"maximum_episodes": episodes, "maximum_control_steps": controls,
            "maximum_physics_substeps": controls * substeps, "ordinary_pair_bytes": pair,
            "terminal_failure_allowance_per_episode": failure_allowance,
            "maximum_file_bytes": file_bytes, "projected_raw_journal_bytes": total,
            "aggregate_limit_bytes": MAX_DIAGNOSTIC_BYTES, "line_limit_bytes": MAX_DIAGNOSTIC_LINE_BYTES,
            "scope": "Conservative full-horizon reservation; cached candidates can reduce actual work, never extend it"}


class _Writer(legacy._Writer):
    def diagnostic(self, name, value):
        raw = _compact(value)
        if (len(raw) > MAX_DIAGNOSTIC_LINE_BYTES or self.sizes.get(name, 0) + len(raw) > MAX_DIAGNOSTIC_FILE_BYTES
                or self.diagnostic_bytes + len(raw) > MAX_DIAGNOSTIC_BYTES):
            raise ValueError("Equipment journal exceeded its fixed line/file/aggregate bound")
        self._reserve(name, len(raw), append=True)
        path = self.out / name
        if path.parent != self.out / "diagnostics":
            raise ValueError("Equipment journal escaped its owned directory")
        path.parent.mkdir(exist_ok=True)
        with path.open("ab" if name in self.sizes else "xb") as stream:
            stream.write(raw)
        self.sizes[name] = self.sizes.get(name, 0) + len(raw)
        self.diagnostic_bytes += len(raw)


class _Journal:
    def __init__(self, writer, name, scene, task, weights, static):
        self.writer, self.name, self.scene, self.task, self.weights, self.static = writer, name, scene, task, weights, static
        self.attempts = self.results = 0

    def attempt(self, action, sample):
        self.writer.diagnostic(self.name, _attempt_record(self.attempts + 1, self.scene, self.task, self.weights, action, sample))
        self.attempts += 1

    def result(self, reward, terminated, truncated, info):
        if any(info.get(key) != value for key, value in self.static.items()):
            raise ValueError("Actual result changed a journal's immutable metric identity")
        dynamic = {key: value for key, value in info.items() if key not in self.static}
        self.writer.diagnostic(self.name, {"type": "result", "index": self.attempts, "reward": float(reward),
                                          "terminated": bool(terminated), "truncated": bool(truncated), "info": dynamic})
        self.results += 1

    def receipt(self):
        path = self.writer.out / self.name
        return {"profile": DIAGNOSTIC_PROFILE, "artifact": self.name if path.is_file() else None,
                "sha256": digest(path) if path.is_file() else None, "action_intents": self.attempts, "result_records": self.results,
                "raw_bytes": self.writer.sizes.get(self.name, 0), "schema_artifact": "settings.json"}


def _trial(env, prepared, seed, reset, deadline, budget, journal, validator):
    row = {"seed": seed, "reset_options": deepcopy(reset), "initial_state_sha256": None,
           "status": "starting", "control_steps": 0, "return": 0., "metrics": None, "returned_controls": 0}
    try:
        _check(deadline)
        try:
            observation, _ = env.reset(seed=seed, options=reset)
        except (ValueError, RuntimeError) as error:
            row.update(status="reset_failed", error=f"{type(error).__name__}: {error}"[:2000])
            return row
        initial = env.snapshot()
        validator(initial)
        row.update(initial_state_sha256=canonical_hash(initial["engine_state"]), initial_metrics=env.metrics())
        if any(row["initial_metrics"].get(key) != value for key, value in journal.static.items()):
            raise ValueError("Reset metric identities differ from the pinned original training protocol")
        for _ in range(env.horizon_steps):
            _check(deadline)
            if budget.used >= budget.maximum:
                budget.reserve()
            action, sample = prepared.sample(observation)
            journal.attempt(action, sample)
            _check(deadline)
            budget.reserve()
            row["control_steps"] += 1
            observation, reward, terminated, truncated, info = env.step(action)
            budget.returned += 1
            row["returned_controls"] += 1
            row["return"] += float(reward)
            journal.result(reward, terminated, truncated, info)
            if terminated or truncated:
                break
        metrics = env.metrics()
        if (not (env.terminated or env.truncated) or not (metrics["completed_interval"] or metrics["failure"] is not None)
                or metrics["completed_interval"] and row["control_steps"] != env.horizon_steps):
            raise ValueError("Equipment training must retain the full interval or explicit terminal failure")
        final = env.snapshot()
        validator(final)
        row.update(status="completed" if metrics["completed_interval"] else "failed", metrics=metrics,
                   terminated=env.terminated, truncated=env.truncated, final_snapshot_sha256=canonical_hash(final))
        _check(deadline)
        return row
    except Exception as error:
        row.update(status="interrupted", error=f"{type(error).__name__}: {error}"[:2000])
        if row["initial_state_sha256"] is not None:
            try:
                row["metrics"] = env.metrics()
            except Exception as metrics_error:
                row["metrics_unavailable"] = f"{type(metrics_error).__name__}: {metrics_error}"[:240]
        error.partial_trial = row
        raise
    finally:
        row["diagnostics"] = journal.receipt()


def _preflight(run, task_path, base_path, base_sha, generations, population):
    root = Path(run).resolve()
    root = root / "run" if (root / "run/manifest.json").is_file() else root
    manifest, members = read_package_manifest(root)
    residual._terminal_manifest(manifest)
    if any(part in _FORBIDDEN_PARTS or part.startswith("policy-evaluation") for name in members for part in Path(name).parts):
        raise ValueError("Equipment training accepts original untrained construction, never policy/evaluation outcomes")
    if "scene.json" not in members or task_path is None and "exercise-task.json" not in members:
        raise ValueError("Equipment training requires an explicit or sealed original task")
    scene_raw = read_package_file(root, "scene.json", max_bytes=4 * 1024 * 1024, expected_sha256=members["scene.json"])
    if task_path is None:
        task_raw = read_package_file(root, "exercise-task.json", max_bytes=64 * 1024, expected_sha256=members["exercise-task.json"])
    else:
        path = Path(task_path).absolute()
        task_raw = read_package_file(path.parent, path.name, max_bytes=64 * 1024)
    path = Path(base_path).absolute()
    base_raw = read_package_file(path.parent, path.name, max_bytes=64 * 1024, expected_sha256=base_sha)
    scene, task = json.loads(scene_raw), json.loads(task_raw)
    reservation = journal_reservation(scene, task, generations, population)
    return manifest, scene_raw, task_raw, base_raw, reservation


def _settings(scene, task, base, base_raw, manifest, hashes, generations, population, max_seconds, reservation):
    return {"schema_version": 1, "algorithm": policy_module.ALGORITHM, "created_at": local_time(),
            "scene_sha256": canonical_hash(scene), "task_sha256": canonical_hash(task), "gates_sha256": GATES_SHA256,
            "source_manifest_sha256": _sha(manifest), "base_policy_file_sha256": _sha(base_raw),
            "base_policy_sha256": canonical_hash(base), "runtime_sha256": hashes,
            "runtime_versions": policy_module._runtime_versions(), "feature_contract": policy_module.feature_contract(),
            "seed": base["training_seed"], "training_seeds": _seed_plan(base["training_seed"], 2),
            "training_reset_options": list(TRAIN_RESETS), "generations": generations, "population": population,
            "elite_count": max(2, population // 4), "initial_mean": 0., "initial_std": .08, "minimum_std": .02,
            "candidate_anchors": ["zero_residual_pinned_v5", "previous_training_best_after_first_generation"],
            "selection_objective": RULE, "max_seconds": max_seconds, "cleanup_reserve_seconds": min(2., max_seconds / 10.),
            "reservation": reservation, "diagnostic_schema": diagnostic_schema(scene, task),
            "metric_static_by_reset": [_static_metrics(scene, task, hashes["shared_exercise_task.py"], reset) for reset in TRAIN_RESETS],
            "workers": 1, "external_api_usd": 0, "evaluation_used": False, "scope": policy_module.SCOPE}


def train_shared_exercise_equipment_feedback_policy(run, out, *, base_policy_path, base_policy_sha256,
                                                   task_path=None, generations=1, population=4, max_seconds=30):
    _integer(generations, "generations", 1, 2)
    _integer(population, "population", 4, 8)
    _seconds(max_seconds, 60)
    policy_module._hash(base_policy_sha256, "base policy file")
    out = _output_path(run, out)
    if out == Path(base_policy_path).resolve() or Path(base_policy_path).resolve().is_relative_to(out):
        raise ValueError("Training output cannot contain or replace the immutable base policy")
    started = time.monotonic()
    deadline = started + max_seconds
    work_deadline = deadline - min(2., max_seconds / 10.)
    manifest_pin, scene_pin, task_pin, base_raw, reservation = _preflight(
        run, task_path, base_policy_path, base_policy_sha256, generations, population)
    _check(work_deadline)
    source, scene, task, scene_raw, task_raw, manifest, proof = _source(run, task_path, False, work_deadline)
    if (manifest != manifest_pin or scene_raw != scene_pin or task_raw != task_pin
            or journal_reservation(scene, task, generations, population) != reservation):
        raise ValueError("Construction inputs changed after metadata-only resource admission")
    base = policy_module.admit_base(base_raw, base_policy_sha256, scene, task)
    hashes = policy_module.runtime_hashes(scene)
    settings = _settings(scene, task, base, base_raw, manifest, hashes, generations, population, max_seconds, reservation)
    views = residual._source_views(source, manifest, scene)
    residual._initialize(out, scene_raw, task_raw, manifest, proof, {**views, "base-policy.json": base_raw})
    writer = _Writer(out, deadline)
    budget = residual._Budget(reservation["maximum_control_steps"])
    budget.returned = 0
    training = {"status": "running", "history": [], "active_candidates": [], "completed_generations": 0,
                "evaluated_episodes": 0, "control_steps": 0, "returned_controls": 0, "cache_hits": 0, "failure": None}
    env = None
    try:
        if (sum(writer.sizes.values()) + reservation["projected_raw_journal_bytes"] + legacy.TERMINAL_RESERVE > residual.MAX_OUTPUT_BYTES
                or len(writer.sizes) + reservation["maximum_episodes"] + 6 >= residual.MAX_OUTPUT_FILES):
            raise ValueError("Source package plus complete journal reservation exceeds the unchanged output ceiling")
        writer.write("settings.json", settings)
        _check(work_deadline)
        env = SharedExerciseTaskEnv(scene, task)
        validator = prepare_shared_exercise_snapshot_validator(scene, task)
        rng = np.random.default_rng(base["training_seed"])
        mean, std = np.zeros((8, 2)), np.full((8, 2), .08)
        cache, origins, initial_states = {}, {}, {}
        selected = baseline = None
        for generation in range(generations):
            samples = np.clip(rng.normal(mean, std, (population, 8, 2)), -.25, .25)
            samples[0] = 0.
            if generation:
                samples[1] = selected["equipment_feedback_coefficients"]
            rows = []
            training["active_candidates"] = rows
            for index, weights in enumerate(samples):
                _check(work_deadline)
                key = canonical_hash(weights.tolist())
                if key in cache:
                    row = deepcopy(cache[key]) | {"candidate": index, "cached": True}
                    training["cache_hits"] += 1
                    rows.append(row)
                else:
                    row = {"candidate": index, "equipment_feedback_coefficients": weights.tolist(), "trials": [],
                           "complete": False, "cached": False}
                    rows.append(row)
                    prepared = policy_module.PreparedEquipmentFeedbackPolicy(env, base, weights)
                    for reset_index, (seed, reset) in enumerate(zip(settings["training_seeds"], TRAIN_RESETS, strict=True)):
                        journal = _Journal(writer, f"diagnostics/g{generation:02d}-c{index:02d}-r{reset_index}.jsonl",
                                           scene, task, weights, settings["metric_static_by_reset"][reset_index])
                        training["evaluated_episodes"] += 1
                        try:
                            trial = _trial(env, prepared, seed, reset, work_deadline, budget, journal, validator)
                        except Exception as error:
                            if hasattr(error, "partial_trial"):
                                row["trials"].append(error.partial_trial)
                            raise
                        row["trials"].append(trial)
                        stripped = {name: value for name, value in trial.items() if name not in ("diagnostics", "returned_controls")}
                        validate_training_trial(stripped, scene, task, base, seed, reset)
                        if trial["returned_controls"] != trial["control_steps"]:
                            raise ValueError("Completed training trial has unreturned control attempts")
                        if trial["initial_state_sha256"] is not None:
                            old = initial_states.setdefault(seed, trial["initial_state_sha256"])
                            if old != trial["initial_state_sha256"]:
                                raise ValueError("Candidates received different original matched initial states")
                    score = _score(row["trials"])
                    row.update(score=score, loss=-score, complete=True, optimization_rank=optimization_rank(row["trials"]))
                    cache[key], origins[key] = deepcopy(row), {"generation": generation, "candidate": index}
                if index == 0:
                    baseline = row
                training.update(control_steps=budget.used, returned_controls=budget.returned)
                writer.write("training.json", training)
            selected = legacy._selected(list(cache.values()))
            unique = {canonical_hash(row["equipment_feedback_coefficients"]): row for row in rows}
            elite = sorted((row for row in unique.values() if row["optimization_rank"]["complete"]),
                           key=lambda row: _rank_key(row["optimization_rank"]))[:settings["elite_count"]]
            values = np.asarray([row["equipment_feedback_coefficients"] for row in elite])
            mean, std = values.mean(axis=0), np.maximum(values.std(axis=0), .02)
            training["history"].append({"generation": generation, "candidates": rows, "next_mean": mean.tolist(),
                                        "next_std": std.tolist(), "selected_optimization_rank": selected["optimization_rank"]})
            training.update(active_candidates=[], completed_generations=generation + 1)
        residual._input_unchanged(source, manifest, work_deadline, task_path=task_path, task_raw=task_raw)
        base_path = Path(base_policy_path).absolute()
        read_package_file(base_path.parent, base_path.name, max_bytes=64 * 1024, expected_sha256=base_policy_sha256)
        if policy_module.runtime_hashes(scene) != hashes:
            raise ValueError("Equipment policy dependencies changed during training")
        weights = actor._coefficients(selected["equipment_feedback_coefficients"])
        policy = policy_module.make_policy(env, weights, base_raw, _sha(manifest), selected["score"], selected["optimization_rank"], hashes)
        policy_module.validate_equipment_feedback_policy(policy, scene, task, base_policy_bytes=base_raw,
                                                         source_manifest_sha256=_sha(manifest))
        _check(work_deadline)
        writer.write("policy.json", policy)
        training.update(status="trained", seconds=time.monotonic() - started, policy_sha256=digest(out / "policy.json"),
                        selected_candidate=origins[canonical_hash(weights.tolist())], selected_optimization_rank=selected["optimization_rank"],
                        selected_pinned_v5_unchanged=not bool(np.any(weights)), selection_objective=RULE,
                        improvement_over_v5_training=selected["score"] - baseline["score"],
                        diagnostic_bytes=writer.diagnostic_bytes, projected_raw_journal_bytes=reservation["projected_raw_journal_bytes"])
        if writer.diagnostic_bytes > reservation["projected_raw_journal_bytes"]:
            raise ValueError("Actual journal bytes exceed the admitted full-interval projection")
        writer.write("training.json", training)
        seal = writer.seal("trained")
        return {"directory": str(out), "manifest_sha256": seal, **training}
    except Exception as error:
        training.update(control_steps=budget.used, returned_controls=budget.returned, seconds=time.monotonic() - started,
                        diagnostic_bytes=writer.diagnostic_bytes, failure=f"{type(error).__name__}: {error}"[:2000])
        residual._failure(out, "training.json", training, error)
        raise
    finally:
        if env is not None:
            env.close()


def _verify_journal(raw, receipt, trial, weights, scene, task, base, static, deadline):
    if receipt["action_intents"] != trial["control_steps"] or receipt["result_records"] != trial["returned_controls"]:
        raise ValueError("Completed journal intents/results differ from issued and returned controls")
    if not trial["control_steps"]:
        if raw is not None or receipt["artifact"] is not None or receipt["sha256"] is not None or receipt["raw_bytes"] != 0:
            raise ValueError("Reset failure cannot fabricate a journal")
        return
    if raw is None or len(raw) > MAX_DIAGNOSTIC_FILE_BYTES or receipt["raw_bytes"] != len(raw) or receipt["sha256"] != _sha(raw):
        raise ValueError("Journal file identity or byte count differs")
    records = raw.splitlines(keepends=True)
    if len(records) != 2 * trial["control_steps"]:
        raise ValueError("Journal omits or duplicates control records")
    prepared = actor.prepare_equipment_feedback(policy_module.metadata_env(scene, task), weights)
    groups = base_actor.joint_groups(scene["action_contract"]["names"][:22], profile=base_actor.SPLIT_POLICY_PROFILE)
    base_weights = base_actor.coefficients(base["feedback_coefficients"], profile=base_actor.SPLIT_POLICY_PROFILE)
    shapes = diagnostic_schema(scene, task)["arrays"]
    reference_start = diagnostic_schema(scene, task)["reference_start_s"]
    return_sum = 0.
    previous = trial["initial_metrics"]
    for index in range(trial["control_steps"]):
        _check(deadline)
        pair = records[2 * index:2 * index + 2]
        if any(len(line) > MAX_DIAGNOSTIC_LINE_BYTES or not line.endswith(b"\n") for line in pair):
            raise ValueError("Journal contains a partial or oversized line")
        attempt, result = map(json.loads, pair)
        expected_keys = {"type", "index", "simulation_time_s", "reference_time_s", *shapes,
                         "nearest_index", "away_direction_known", "changed_channels"}
        if (set(attempt) != expected_keys or attempt["type"] != "attempt" or type(attempt["index"]) is not int
                or attempt["index"] != index + 1 or attempt["simulation_time_s"] != index * scene["dt"]
                or attempt["reference_time_s"] != reference_start + index * scene["dt"]):
            raise ValueError("Journal action boundary clock or columns differ")
        arrays = {name: _unpack(attempt[name], shape) for name, shape in shapes.items()}
        observed, features = arrays["observation"], arrays["base_features"]
        if (observed.dtype != np.dtype("float32") or np.any(abs(observed) > 1e6)
                or observed[88:154].tobytes() != features.astype(np.float32).ravel().tobytes()
                or not math.isclose(float(observed[-1]), index / task["interval"]["horizon_steps"], abs_tol=1e-6)):
            raise ValueError("Journal observation and exact v5 features are not the same control boundary")
        expected_base, *_ = base_actor.action_sample(base_weights, groups, SimpleNamespace(action_count=22, feature_matrix=lambda: features))
        if expected_base.dtype != arrays["base_action"].dtype or expected_base.tobytes() != arrays["base_action"].tobytes():
            raise ValueError("Journal base action differs from the pinned v5 arithmetic")
        expected_action, expected_report = prepared.sample(arrays["base_action"], observed)
        if expected_action.dtype != arrays["action"].dtype or expected_action.tobytes() != arrays["action"].tobytes():
            raise ValueError("Journal action differs from equipment coefficients, current state or original-base bounds")
        nearest = attempt["nearest_index"]
        if (type(nearest) is not int or not 0 <= nearest < len(task["equipment_ids"])
                or type(attempt["away_direction_known"]) is not bool):
            raise ValueError("Journal nearest-center identity differs")
        reported = _actor_report(arrays["actor_values"], observed, arrays["action"].dtype.str, weights, task,
                                  nearest, attempt["away_direction_known"], attempt["changed_channels"])
        if _compact(reported) != _compact(expected_report):
            raise ValueError("Journal changed or omitted the original actor diagnostic")
        if (set(result) != {"type", "index", "reward", "terminated", "truncated", "info"}
                or result["type"] != "result" or type(result["index"]) is not int or result["index"] != index + 1
                or type(result["reward"]) not in (int, float) or not math.isfinite(result["reward"])
                or any(type(result[name]) is not bool for name in ("terminated", "truncated"))
                or any(name in result["info"] for name in static)):
            raise ValueError("Journal result changed its dynamic/static contract")
        info = {**static, **result["info"]}
        extras = {"observation_status", "legacy_coupled_score_used"}
        metrics = {key: value for key, value in info.items() if key not in extras}
        validate_metrics(metrics, scene, task, base, trial["reset_options"])
        expected_extras = {"observation_status"} if info.get("observation_status") == "last_valid_control_boundary" else extras
        if (set(info) != set(metrics) | expected_extras
                or "legacy_coupled_score_used" in expected_extras and info.get("legacy_coupled_score_used") is not False
                or info["observation_status"] not in ("current", "last_valid_control_boundary")
                or index < trial["control_steps"] - 1 and (result["terminated"] or result["truncated"])
                or metrics["observed_substeps"] < previous["observed_substeps"]
                or metrics["simulated_duration_s"] < previous["simulated_duration_s"] - 1e-8):
            raise ValueError("Journal result ordering or source/task observation status differs")
        complete_boundary = metrics["failure"] is None or metrics["completed_interval"]
        if (not index * scene["substeps"] <= metrics["observed_substeps"] <= (index + 1) * scene["substeps"]
                or complete_boundary and (metrics["observed_substeps"] != (index + 1) * scene["substeps"]
                    or not math.isclose(metrics["simulated_duration_s"], (index + 1) * scene["dt"], rel_tol=0., abs_tol=1e-8))
                or result["terminated"] != bool(metrics["failure"] or metrics["success"])
                or result["truncated"] != bool(metrics["completed_interval"] and not result["terminated"])
                or info["observation_status"] == "last_valid_control_boundary" and metrics["failure"] is None
                or not math.isclose(result["reward"], (metrics["reward_integral"] - previous["reward_integral"]) / scene["dt"],
                                    rel_tol=0., abs_tol=1e-8)):
            raise ValueError("Journal returned control clock, termination or reward increment differs")
        for key in ("peak_angular_rmse_rad", "peak_root_rmse_model_units", "max_floor_penetration_model_units"):
            if metrics[key] < previous[key] - 1e-12:
                raise ValueError("Journal accumulated body peaks cannot decrease")
        for old, new in zip(previous["equipment"], metrics["equipment"], strict=True):
            if any(new[key] < old[key] - 1e-12 for key in EQUIPMENT_GATES):
                raise ValueError("Journal accumulated equipment peaks cannot decrease")
        previous = metrics
        return_sum += result["reward"]
    if (previous != trial["metrics"] or result["terminated"] != trial["terminated"]
            or result["truncated"] != trial["truncated"] or not math.isclose(return_sum, trial["return"], abs_tol=1e-8)):
        raise ValueError("Journal final metrics/return differ from the original scored trial")


def verify_equipment_feedback_training(manifest_raw, files, scene, task, *, base_policy_sha256,
                                       source_manifest_sha256, deadline):
    """Check saved CEM and lossless journals; source-custody admission is separate from arithmetic replay."""
    _check(deadline)
    manifest = json.loads(manifest_raw)
    required = {"settings.json", "training.json", "policy.json", "base-policy.json", "scene.json", "exercise-task.json",
                "input-manifest.json", "source-verification.json"}
    if (manifest.get("status") != "trained" or "failure.json" in files or not required <= set(files)
            or len(files) >= residual.MAX_OUTPUT_FILES or sum(map(len, files.values())) > residual.MAX_OUTPUT_BYTES
            or set(manifest.get("files", {})) != set(files) or any(_sha(files[name]) != value for name, value in manifest["files"].items())
            or json.loads(files["scene.json"]) != scene or json.loads(files["exercise-task.json"]) != task
            or _sha(files["input-manifest.json"]) != source_manifest_sha256 or _sha(files["base-policy.json"]) != base_policy_sha256):
        raise ValueError("Equipment training seal, original construction or pinned base differs")
    parent = json.loads(files["input-manifest.json"])
    if any(part in _FORBIDDEN_PARTS or part.startswith("policy-evaluation") for name in parent["files"] for part in Path(name).parts):
        raise ValueError("Equipment training cannot import evaluation or policy outcomes as construction")
    policy = policy_module.validate_equipment_feedback_policy(json.loads(files["policy.json"]), scene, task,
        base_policy_bytes=files["base-policy.json"], source_manifest_sha256=source_manifest_sha256)
    base = json.loads(files["base-policy.json"])
    for name, expected in policy["runtime_sha256"].items():
        if _sha(files.get("implementation/" + name, b"")) != expected:
            raise ValueError("Equipment training did not retain its exact executed dependencies")
    settings, training = json.loads(files["settings.json"]), json.loads(files["training.json"])
    generations, population = settings["generations"], settings["population"]
    _seconds(settings["max_seconds"], 60)
    reservation = journal_reservation(scene, task, generations, population)
    expected = _settings(scene, task, base, files["base-policy.json"], files["input-manifest.json"], policy["runtime_sha256"],
                         generations, population, settings["max_seconds"], reservation)
    expected["created_at"] = settings.get("created_at")
    if settings != expected or type(settings["created_at"]) is not str:
        raise ValueError("Equipment training settings changed its generator, protocols or resource reservation")
    keys = {"status", "history", "active_candidates", "completed_generations", "evaluated_episodes", "control_steps",
            "returned_controls", "cache_hits", "failure", "seconds", "policy_sha256", "selected_candidate",
            "selected_optimization_rank", "selected_pinned_v5_unchanged", "selection_objective", "improvement_over_v5_training",
            "diagnostic_bytes", "projected_raw_journal_bytes"}
    if (set(training) != keys or training["status"] != "trained" or training["failure"] is not None
            or training["active_candidates"] != [] or training["completed_generations"] != generations
            or len(training["history"]) != generations or training["policy_sha256"] != _sha(files["policy.json"])
            or type(training["seconds"]) not in (int, float) or not 0 <= training["seconds"] <= settings["max_seconds"]):
        raise ValueError("Equipment training is not a complete frozen terminal result")
    rng = np.random.default_rng(base["training_seed"])
    mean, std, cache, origins, starts, diagnostics = np.zeros((8, 2)), np.full((8, 2), .08), {}, {}, {}, set()
    selected = baseline = None
    episodes = controls = returned = hits = diagnostic_bytes = 0
    for generation, record in enumerate(training["history"]):
        _check(deadline)
        if (set(record) != {"generation", "candidates", "next_mean", "next_std", "selected_optimization_rank"}
                or record["generation"] != generation or len(record["candidates"]) != population):
            raise ValueError("Equipment generation sequence differs")
        samples = np.clip(rng.normal(mean, std, (population, 8, 2)), -.25, .25)
        samples[0] = 0.
        if generation:
            samples[1] = selected["equipment_feedback_coefficients"]
        for index, (row, weights) in enumerate(zip(record["candidates"], samples, strict=True)):
            if (set(row) != {"candidate", "equipment_feedback_coefficients", "trials", "complete", "cached", "score", "loss", "optimization_rank"}
                    or row["candidate"] != index or row["equipment_feedback_coefficients"] != weights.tolist()
                    or row["complete"] is not True or len(row["trials"]) != 2):
                raise ValueError("Equipment candidate differs from its deterministic generator")
            key = canonical_hash(weights.tolist())
            if key in cache:
                if row != cache[key] | {"candidate": index, "cached": True}:
                    raise ValueError("Cached equipment candidate changed an earlier exact trial")
                hits += 1
            else:
                if row["cached"] is not False:
                    raise ValueError("Equipment cache references an unobserved candidate")
                for reset_index, (trial, seed, reset) in enumerate(zip(row["trials"], settings["training_seeds"], TRAIN_RESETS, strict=True)):
                    stripped = {name: value for name, value in trial.items() if name not in ("diagnostics", "returned_controls")}
                    validate_training_trial(stripped, scene, task, base, seed, reset)
                    if type(trial["returned_controls"]) is not int or trial["returned_controls"] != trial["control_steps"]:
                        raise ValueError("Eligible trial has unreturned actual controls")
                    receipt = trial["diagnostics"]
                    if (set(receipt) != {"profile", "artifact", "sha256", "action_intents", "result_records", "raw_bytes", "schema_artifact"}
                            or receipt["profile"] != DIAGNOSTIC_PROFILE or receipt["schema_artifact"] != "settings.json"):
                        raise ValueError("Equipment diagnostic receipt differs")
                    name = receipt["artifact"]
                    if name is not None:
                        if (name != f"diagnostics/g{generation:02d}-c{index:02d}-r{reset_index}.jsonl"
                                or name not in files or name in diagnostics):
                            raise ValueError("Equipment trial journal is missing, reused or outside its own slot")
                        diagnostics.add(name)
                        diagnostic_bytes += len(files[name])
                    _verify_journal(files.get(name), receipt, trial, weights, scene, task, base,
                                    settings["metric_static_by_reset"][reset_index], deadline)
                    episodes += 1; controls += trial["control_steps"]; returned += trial["returned_controls"]
                    if trial["initial_state_sha256"] is not None:
                        prior = starts.setdefault(seed, trial["initial_state_sha256"])
                        if prior != trial["initial_state_sha256"]:
                            raise ValueError("Equipment candidates changed their matched initial states")
                if row["score"] != _score(row["trials"]) or row["loss"] != -row["score"] or row["optimization_rank"] != optimization_rank(row["trials"]):
                    raise ValueError("Equipment rank differs from original full-interval gates and rewards")
                cache[key], origins[key] = row, {"generation": generation, "candidate": index}
            if index == 0:
                baseline = row
        selected = legacy._selected(list(cache.values()))
        unique = {canonical_hash(row["equipment_feedback_coefficients"]): row for row in record["candidates"]}
        elites = sorted((row for row in unique.values() if row["optimization_rank"]["complete"]),
                        key=lambda row: _rank_key(row["optimization_rank"]))[:settings["elite_count"]]
        values = np.asarray([row["equipment_feedback_coefficients"] for row in elites])
        mean, std = values.mean(axis=0), np.maximum(values.std(axis=0), .02)
        if record["next_mean"] != mean.tolist() or record["next_std"] != std.tolist() or record["selected_optimization_rank"] != selected["optimization_rank"]:
            raise ValueError("Equipment elite update or selected candidate differs")
    selected_weights = selected["equipment_feedback_coefficients"]
    fixed = {"evaluated_episodes": episodes, "control_steps": controls, "returned_controls": returned, "cache_hits": hits,
             "selected_candidate": origins[canonical_hash(selected_weights)], "selected_optimization_rank": selected["optimization_rank"],
             "selected_pinned_v5_unchanged": not bool(np.any(selected_weights)), "selection_objective": RULE,
             "improvement_over_v5_training": selected["score"] - baseline["score"], "diagnostic_bytes": diagnostic_bytes,
             "projected_raw_journal_bytes": reservation["projected_raw_journal_bytes"]}
    if (any(training[name] != value for name, value in fixed.items()) or controls > reservation["maximum_control_steps"]
            or diagnostic_bytes > reservation["projected_raw_journal_bytes"]
            or diagnostics != {name for name in files if Path(name).parts[0] == "diagnostics"}
            or policy["equipment_feedback_coefficients"] != selected_weights or policy["training_score"] != selected["score"]
            or policy["optimization_rank"] != selected["optimization_rank"]):
        raise ValueError("Equipment selected artifact, complete journals or aggregate work accounting differs")
    return {"unique_candidates": len(cache), "feasible_candidates": sum(row["optimization_rank"]["feasible"] for row in cache.values()),
            "selected_training_feasible": selected["optimization_rank"]["feasible"], "selected_candidate": fixed["selected_candidate"],
            "control_steps": controls, "returned_controls": returned, "evaluated_episodes": episodes,
            "diagnostic_files": len(diagnostics), "actual_raw_journal_bytes": diagnostic_bytes,
            "projected_raw_journal_bytes": reservation["projected_raw_journal_bytes"], "scope": "Saved arithmetic/accounting verification, not physical replay"}


def verify_equipment_feedback_package(directory, *, manifest_sha256, base_policy_sha256, source_manifest_sha256, deadline):
    """Original source custody must pass before a saved training policy becomes usable."""
    manifest, files = _verified_inputs(directory, deadline=deadline)
    if _sha(manifest) != manifest_sha256:
        raise ValueError("Equipment training package changed from its externally pinned seal")
    scene, task = json.loads(files["scene.json"]), json.loads(files["exercise-task.json"])
    collect_shared_source_files(scene, directory, expected_manifest_sha256=manifest_sha256, deadline=deadline)
    return verify_equipment_feedback_training(manifest, files, scene, task, base_policy_sha256=base_policy_sha256,
                                               source_manifest_sha256=source_manifest_sha256, deadline=deadline)
