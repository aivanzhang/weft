"""Strict v5 task snapshots; original metric validation is separate from actual engine validation."""

from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import re

import numpy as np

from weft import shared_exercise_task as base_tasks
from weft.io import canonical_hash
from weft.shared_exercise_feet_projection import FOOT_SEGMENTS, scatter_task_action
from weft.shared_exercise_reset import HUMAN_JITTER_PROFILE, resolve_reset_options
from weft.shared_exercise_task import (
    GATES, GATES_SHA256, GATE_PROFILE, GATES_PROVENANCE,
    HUMAN_ONLY_TASK_PROFILE, HUMAN_ONLY_GATE_APPLICABILITY, HUMAN_ONLY_REWARD_PROFILE,
    _reset_protocol, _support_module, _support_evaluator_sha256, _support_reset_protocol,
)
from weft.shared_world import prepare_shared_snapshot_validator


MAX_SNAPSHOT_BYTES = 480 << 10
MAX_SNAPSHOT_NODES = 100_000
MAX_CONTACTS = 256
_HASH_FIELDS = ("scene_sha256", "base_scene_sha256", "base_task_sha256", "projection_profile",
                "projection_contract_sha256", "adapter_implementation_sha256", "foot_reference_sha256")
_DIAGNOSTIC_FIXED = {
    "profile": "shared-exercise-feet-diagnostics-v1", "target_mode": "authored_initial_angle_hold",
    "source": "simulated_engine_substeps",
    "scored": False, "foot_support_credit": False, "ground_load_requirements": "unsupported",
    "retention": "latest_observed_substep_plus_episode_aggregates",
}


def _bounded_json(value):
    nodes = 0

    def visit(item, depth):
        nonlocal nodes
        nodes += 1
        if nodes > MAX_SNAPSHOT_NODES or depth > 32:
            raise ValueError("Feet snapshot exceeds its bounded JSON structure")
        if type(item) is dict:
            if len(item) > (MAX_SNAPSHOT_NODES - nodes) // 2:
                raise ValueError("Feet snapshot exceeds its bounded JSON structure")
            for key, child in item.items():
                if type(key) is not str:
                    raise ValueError("Feet snapshot keys must be strings")
                visit(key, depth + 1)
                visit(child, depth + 1)
        elif type(item) is list:
            if len(item) > MAX_SNAPSHOT_NODES - nodes:
                raise ValueError("Feet snapshot exceeds its bounded JSON structure")
            for child in item:
                visit(child, depth + 1)
        elif type(item) is str:
            if len(item) > MAX_SNAPSHOT_BYTES:
                raise ValueError("Feet snapshot exceeds its JSON byte bound")
        elif type(item) in (int, float):
            try:
                finite = math.isfinite(item)
            except OverflowError:
                finite = False
            if not finite:
                raise ValueError("Feet snapshot requires finite JSON numbers")
        elif item is not None and type(item) is not bool:
            raise ValueError("Feet snapshot must contain only JSON values")

    visit(value, 0)
    size = 0
    try:
        for chunk in json.JSONEncoder(allow_nan=False, separators=(",", ":")).iterencode(value):
            size += len(chunk.encode("utf-8"))
            if size > MAX_SNAPSHOT_BYTES:
                raise ValueError("Feet snapshot exceeds its JSON byte bound")
    except (TypeError, UnicodeError, RecursionError) as error:
        raise ValueError("Feet snapshot must be finite bounded UTF-8 JSON") from error


def _number(value, *, lower=0., upper=1e30):
    return type(value) in (int, float) and math.isfinite(value) and lower <= value <= upper


def _same(value, expected):
    if type(value) is not type(expected):
        return False
    if isinstance(expected, dict):
        return value.keys() == expected.keys() and all(_same(value[key], child) for key, child in expected.items())
    if isinstance(expected, list):
        return len(value) == len(expected) and all(_same(a, b) for a, b in zip(value, expected))
    return value == expected


def validate_shared_exercise_feet_snapshot(snapshot, scene, task):
    """One-shot preparation can compile models; reuse the prepared validator for a recording."""
    return prepare_shared_exercise_feet_snapshot_validator(scene, task)(snapshot)


def prepare_shared_exercise_feet_snapshot_validator(scene, task):
    """Pin actual physics and base scoring identities once; never compile or hash per frame."""
    from weft.shared_exercise_feet_task import STATE_PROFILE, prepare_shared_exercise_feet_task

    actual, base, outer = prepare_shared_exercise_feet_task(task, scene)
    original = outer["base_task"]
    support = original["profile"] == base_tasks.SUPPORT_TASK_PROFILE
    validate_engine = prepare_shared_snapshot_validator(actual)
    evaluator_sha = (_support_evaluator_sha256() if support else
                     hashlib.sha256(Path(base_tasks.__file__).read_bytes()).hexdigest())
    validate_metrics = (_prepare_support_metrics_validator(base, original, evaluator_sha) if support else
                        _prepare_legacy_metrics_validator(base, original))
    validate_diagnostics = _prepare_diagnostics_validator(actual)
    fixed = {"schema_version": 1, "profile": STATE_PROFILE, "task_sha256": canonical_hash(outer),
             **{key: outer[key] for key in _HASH_FIELDS}}
    expected = set(fixed) | {"engine_state", "task_metrics", "ankle_diagnostics", "last_action",
                             "engine_state_status", "resolved_reset_options", "terminated", "truncated"}
    root = original["control"]["root_assistance"] == "explicit_3_axis"
    count, substeps = len(actual["action_contract"]["names"]), actual["substeps"]
    objects = {item["id"]: item for item in actual["objects"]}

    def validate(snapshot):
        _bounded_json(snapshot)
        if (not isinstance(snapshot, dict) or set(snapshot) != expected
                or any(not _same(snapshot[key], value) for key, value in fixed.items())):
            raise ValueError("Feet snapshot fields or pinned engine/base/task/projection identities differ")
        options = snapshot["resolved_reset_options"]
        if options != resolve_reset_options(options):
            raise ValueError("Feet snapshot reset options must be resolved explicitly")
        state = snapshot["engine_state"]
        validate_engine(state)
        if state["controlled_id"] != "human" or state["human"].get("reference", False) is not False:
            raise ValueError("Feet task snapshots require actual human control, not reference playback")
        if support:
            if state["target"] is not None or [item["id"] for item in state["objects"]] != list(objects):
                raise ValueError("Feet support snapshots must retain static objects and no object goal")
            for item in state["objects"]:
                nominal = objects[item["id"]]
                if (item["position"] != nominal["position"] or item["quaternion"] != nominal["quaternion"]
                        or any(any(item[key]) for key in ("velocity", "angular_velocity"))
                        or any(item.get("virtual_force", (0., 0., 0.)))):
                    raise ValueError("Feet static support pose, motion or force differs from its scene")
        elif state["steps"] == 0 and options.get("profile") == HUMAN_JITTER_PROFILE:
            for item in state["objects"]:
                nominal = objects[item["id"]]
                if (any(not np.allclose(item[key], nominal[key], rtol=0, atol=1e-12)
                        for key in ("position", "velocity", "angular_velocity"))
                        or not math.isclose(abs(float(np.dot(item["quaternion"], nominal["quaternion"]))),
                                            1., rel_tol=0, abs_tol=1e-12)):
                    raise ValueError("Human-only feet resets must retain equipment pose and velocity")
        status = snapshot["engine_state_status"]
        if not isinstance(status, str) or status not in {"current", "last_valid_control_boundary"}:
            raise ValueError("Feet geometry must distinguish the last valid failure boundary")
        action = snapshot["last_action"]
        if not isinstance(action, list):
            raise ValueError("Feet snapshot action must be a JSON list")
        full = scatter_task_action(action, engine_action_count=count, root_assistance=root)
        engine_action = state["normalized_action"]
        if (any(engine_action[28:]) or not root and any(engine_action[19:22])
                or status == "current" and list(full) != engine_action):
            raise ValueError("Feet task/engine actions disagree or contain forbidden root/object actuation")
        metrics = snapshot["task_metrics"]
        if (not isinstance(metrics, dict) or metrics.get("evaluator_sha256") != evaluator_sha
                or not _same(metrics.get("source_scope"), original["source_scope"])):
            raise ValueError("Feet base evaluator identity or original source scope differs")
        validate_metrics(snapshot)
        observed = metrics["observed_substeps"]
        if status == "current":
            partial = state.get("partial_action")
            physical = ((state["steps"] - 1) * substeps + partial["completed_substeps"]
                        if partial is not None else state["steps"] * substeps)
            if observed != physical:
                raise ValueError("Feet observed substeps differ from the actual engine boundary")
        elif not state["steps"] * substeps <= observed <= (state["steps"] + 1) * substeps:
            raise ValueError("Feet stale geometry may precede only its single failed control action")
        validate_diagnostics(snapshot)
        return deepcopy(snapshot)

    return validate


def _prepare_diagnostics_validator(scene):
    pairs = tuple((segment, item["id"]) for item in scene["objects"] for segment in FOOT_SEGMENTS)
    fields = set(_DIAGNOSTIC_FIXED) | {"observed_substeps", "valid_substeps", "valid_observed_duration_s",
        "ankle_hold_error_rmse_rad", "peak_ankle_hold_error_rmse_rad", "ankle_effort_mean_squared",
        "excluded_foot_contact_records", "contact_aggregates", "latest_substep"}
    measures = ("ankle_hold_error_rmse_rad", "peak_ankle_hold_error_rmse_rad", "ankle_effort_mean_squared")
    substeps, tick = scene["substeps"], scene["dt"] / scene["substeps"]
    horizon = scene["task"]["horizon_steps"]
    row_fields = {"human_segment", "object_id", "contact_records", "first_contact_time_s",
                  "last_contact_time_s", "peak_normal_force"}
    latest_fields = {"step_index", "substep_index", "interval_start_s", "interval_end_s", "contact_time_s",
                     "valid", "failure", "excluded_foot_contacts"}
    contact_fields = {"human_geom", "human_segment", "object_id", "position", "distance", "normal_force",
                      "force_on_human_world", "normal_on_human_world"}

    def validate(snapshot):
        metrics, info = snapshot["task_metrics"], snapshot["ankle_diagnostics"]
        if (not isinstance(info, dict) or set(info) != fields
                or any(not _same(info[key], value) for key, value in _DIAGNOSTIC_FIXED.items())):
            raise ValueError("Ankle diagnostics must retain their unscored, latest-only scope")
        observed, valid = info["observed_substeps"], info["valid_substeps"]
        if (type(observed) is not int or type(valid) is not int
                or observed != metrics["observed_substeps"] or valid != metrics["valid_substeps"]
                or not _number(info["valid_observed_duration_s"], upper=60.)
                or not math.isclose(info["valid_observed_duration_s"], metrics["valid_observed_duration_s"],
                                    rel_tol=0, abs_tol=1e-8)):
            raise ValueError("Ankle diagnostics must preserve accepted base-scorer coverage")
        if any((info[key] is not None if not valid else not _number(info[key], upper=1. + 1e-9 if key == measures[2] else 1e30))
               for key in measures):
            raise ValueError("Ankle diagnostics remain unavailable until observed, with bounded action effort")
        if valid and info[measures[0]] > info[measures[1]] + 1e-9:
            raise ValueError("Mean ankle hold error cannot exceed its observed peak")
        total = info["excluded_foot_contact_records"]
        rows = info["contact_aggregates"]
        if (type(total) is not int or not 0 <= total <= observed * MAX_CONTACTS
                or not isinstance(rows, list) or len(rows) != len(pairs)):
            raise ValueError("Foot contact count or bounded scene-pair coverage differs")
        failure = metrics["failure"]
        aggregate = {}
        for row, pair in zip(rows, pairs):
            if (not isinstance(row, dict) or set(row) != row_fields
                    or (row["human_segment"], row["object_id"]) != pair
                    or type(row["contact_records"]) is not int or not 0 <= row["contact_records"] <= total):
                raise ValueError("Foot aggregates must retain exact scene-order identities and record counts")
            count = row["contact_records"]
            first, last, peak = (row[key] for key in ("first_contact_time_s", "last_contact_time_s", "peak_normal_force"))
            if not count:
                if any(value is not None for value in (first, last, peak)):
                    raise ValueError("Unobserved foot pairs cannot invent contact times or loads")
            elif (not _number(first, lower=-1e6, upper=1e6) or not _number(last, lower=-1e6, upper=1e6)
                    or not _number(peak) or count == 1 and first != last):
                raise ValueError("Observed foot pairs require finite raw clocks and nonnegative peak forces")
            elif failure is None and not 0 <= first <= last <= (observed - 1) * tick + 1e-8:
                raise ValueError("Foot contact timestamps exceed their observed episode coverage")
            aggregate[pair] = row
        if sum(row["contact_records"] for row in rows) != total:
            raise ValueError("Foot contact aggregates do not conserve excluded records")
        latest = info["latest_substep"]
        if not observed:
            if latest is not None:
                raise ValueError("Unobserved ankle diagnostics cannot invent a latest substep")
            return
        if (not isinstance(latest, dict) or set(latest) != latest_fields or type(latest["valid"]) is not bool
                or type(latest["step_index"]) is not int or not 0 <= latest["step_index"] <= horizon + 1
                or type(latest["substep_index"]) is not int or not 0 <= latest["substep_index"] <= substeps + 1
                or any(not _number(latest[key], lower=-1e6, upper=1e6)
                       for key in ("interval_start_s", "contact_time_s"))
                or latest["interval_end_s"] is not None and not _number(latest["interval_end_s"], lower=-1e6, upper=1e6)
                or latest["failure"] is not None and (not isinstance(latest["failure"], str)
                                                      or not 1 <= len(latest["failure"]) <= 240)):
            raise ValueError("Latest ankle substep must retain bounded raw clocks, indices and failure fields")
        if (not latest["valid"] or latest["failure"] is not None or latest["interval_end_s"] is None) and failure is None:
            raise ValueError("Rejected or unavailable latest foot observations require an explicit base failure")
        if valid == observed:
            step, substep = divmod(observed - 1, substeps)
            if (latest["valid"] is not True or latest["step_index"] != step + 1 or latest["substep_index"] != substep + 1
                    or latest["interval_end_s"] is None
                    or not math.isclose(latest["interval_start_s"], (observed - 1) * tick, rel_tol=0, abs_tol=1e-8)
                    or not math.isclose(latest["contact_time_s"], latest["interval_start_s"], rel_tol=0, abs_tol=1e-8)
                    or not math.isclose(latest["interval_end_s"], observed * tick, rel_tol=0, abs_tol=1e-8)):
                raise ValueError("Accepted latest ankle samples must retain the unmodified engine clocks and order")
        contacts = latest["excluded_foot_contacts"]
        if not isinstance(contacts, list) or not 0 <= len(contacts) <= min(MAX_CONTACTS, total):
            raise ValueError("Latest foot contacts exceed the bounded observed record count")
        latest_pairs = {}
        for contact in contacts:
            if not isinstance(contact, dict) or set(contact) != contact_fields:
                raise ValueError("Latest foot contact fields differ from immutable engine telemetry")
            pair = (contact["human_segment"], contact["object_id"])
            if (not all(isinstance(value, str) for value in pair) or pair not in aggregate
                    or contact["human_geom"] != pair[0] + "_geom"
                    or not _number(contact["distance"], lower=-1e30) or not _number(contact["normal_force"])):
                raise ValueError("Latest foot contact identity or force is not a declared foot/equipment pair")
            for key in ("position", "force_on_human_world", "normal_on_human_world"):
                if (not isinstance(contact[key], list) or len(contact[key]) != 3
                        or any(not _number(value, lower=-1e30) for value in contact[key])):
                    raise ValueError("Latest foot contact vectors must contain three finite numbers")
            count, peak = latest_pairs.get(pair, (0, 0.))
            latest_pairs[pair] = count + 1, max(peak, contact["normal_force"])
        for pair, (count, peak) in latest_pairs.items():
            row = aggregate[pair]
            if (row["contact_records"] < count or row["last_contact_time_s"] != latest["contact_time_s"]
                    or row["peak_normal_force"] < peak
                    or row["contact_records"] == count and (row["first_contact_time_s"] != latest["contact_time_s"]
                                                            or row["peak_normal_force"] != peak)):
                raise ValueError("Latest foot contacts contradict aggregate counts, clocks or peaks")
        if observed == 1 and total != len(contacts):
            raise ValueError("A single observed substep must account for every foot contact record")

    return validate


# Only legacy validation algebra is copied; neither scorer nor engine state is transformed.
def _prepare_legacy_metrics_validator(scene, task):
    scene_sha256, task_sha256 = canonical_hash(scene), canonical_hash(task)
    gates = dict(GATES)
    equipment_order = [obj["id"] for obj in scene["objects"] if obj["role"] != "static"]
    contact_windows = tuple((row, (row["end_step"] - row["start_step"]) * scene["dt"])
                            for row in task["contact_requirements"])
    root_assistance_disabled = task["control"]["root_assistance"] == "disabled"
    numeric = ("simulated_duration_s", "valid_observed_duration_s", "declared_duration_s", "peak_angular_rmse_rad",
               "peak_root_rmse_model_units", "max_floor_penetration_model_units", "joint_effort_mean_squared",
               "root_assistance_mean_squared", "reward_integral", "mean_reward", "failure_reward_penalty")
    fixed = {"profile": task["profile"], "gates_sha256": GATES_SHA256, "gate_profile": GATE_PROFILE,
             "gates_provenance": GATES_PROVENANCE, "source_scope": task["source_scope"],
             "root_assistance": task["control"]["root_assistance"], "virtual_object_forces": "forbidden",
             "scene_sha256": scene_sha256, "task_sha256": task_sha256}
    if task["profile"] == HUMAN_ONLY_TASK_PROFILE:
        fixed.update(gate_applicability=dict(HUMAN_ONLY_GATE_APPLICABILITY), reward_profile=HUMAN_ONLY_REWARD_PROFILE)
    metric_keys = {*fixed, *numeric, "angular_rmse_rad", "root_rmse_model_units", "completed_interval", "success", "failure",
            "failed_gates", "observed_substeps", "valid_substeps", "equipment", "contact_requirements", "evaluator_sha256", "clock_status", "reset_protocol"}
    duration = task["interval"]["horizon_steps"] * scene["dt"]
    tick = scene["dt"] / scene["substeps"]
    maximum = task["interval"]["horizon_steps"] * scene["substeps"]

    def validate(snapshot):
        state = snapshot["engine_state"]
        status = snapshot["engine_state_status"]
        metrics = snapshot["task_metrics"]
        if (not isinstance(metrics, dict) or set(metrics) != metric_keys or any(metrics[key] != value for key, value in fixed.items())
                or metrics["reset_protocol"] != _reset_protocol(snapshot["resolved_reset_options"])
                or not isinstance(metrics["evaluator_sha256"], str) or not re.fullmatch(r"[0-9a-f]{64}", metrics["evaluator_sha256"])):
            raise ValueError("Exercise metrics fields, gates or source scope differ")
        if any(type(metrics[key]) not in (int, float) or not math.isfinite(metrics[key]) for key in numeric):
            raise ValueError("Exercise metrics require finite numeric values")
        observed, valid = metrics["observed_substeps"], metrics["valid_substeps"]
        if (type(observed) is not int or type(valid) is not int or not 0 <= valid <= observed <= maximum
                or not math.isclose(metrics["declared_duration_s"], duration, rel_tol=0, abs_tol=1e-9)
                or not math.isclose(metrics["valid_observed_duration_s"], valid * tick, rel_tol=0, abs_tol=1e-8)
                or not valid * tick - 1e-8 <= metrics["simulated_duration_s"] <= observed * tick + 1e-8):
            raise ValueError("Exercise duration/substep accounting differs")
        failure = metrics["failure"]
        if failure is not None and (not isinstance(failure, str) or not 1 <= len(failure) <= 240):
            raise ValueError("Exercise failure must be a bounded explicit reason")
        if valid < observed and failure is None:
            raise ValueError("Missing or invalid substeps cannot masquerade as an ongoing valid task")
        if (metrics["clock_status"] not in {"observed", "invalid_substep_clock", "unavailable_after_last_observed_time"}
                or metrics["clock_status"] != "observed" and failure is None):
            raise ValueError("Unavailable exercise clocks must remain explicit failures")
        if status == "last_valid_control_boundary" and (failure is None or state["time_s"] > metrics["simulated_duration_s"] + 1e-8):
            raise ValueError("Stale exercise geometry requires a later explicit failure")
        if status == "current" and not math.isclose(state["time_s"], metrics["simulated_duration_s"], rel_tol=0, abs_tol=1e-8):
            raise ValueError("Current engine geometry must match actual exercise time")
        failures = [failure] if failure else []
        for key in ("angular_rmse_rad", "root_rmse_model_units", "peak_angular_rmse_rad", "peak_root_rmse_model_units",
                    "max_floor_penetration_model_units"):
            value = metrics[key]
            if value is None:
                if valid or key not in {"angular_rmse_rad", "root_rmse_model_units"}:
                    raise ValueError("Exercise tracking values cannot be missing after valid substeps")
            elif type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                raise ValueError("Exercise tracking values must be finite and nonnegative")
            if value is None or value > gates[key] + 1e-12:
                failures.append(key)
        equipment = metrics["equipment"]
        if (not isinstance(equipment, list) or any(not isinstance(row, dict) for row in equipment)
                or [row.get("object_id") for row in equipment] !=
                equipment_order):
            raise ValueError("Exercise metrics must retain every equipment object in scene order")
        for row in equipment:
            keys = {"displacement_model_units", "orientation_rad", "speed_model_units_s", "angular_speed_rad_s"}
            if (set(row) != keys | {"object_id", "within_limits"}
                    or any(type(row[key]) not in (int, float) or row[key] < 0 or not math.isfinite(row[key]) for key in keys)):
                raise ValueError("Equipment stability metrics must be bounded nonnegative values")
            passed = all(row[key] <= gates["equipment_" + key] + 1e-12 for key in keys)
            if type(row["within_limits"]) is not bool or row["within_limits"] != passed:
                raise ValueError("Equipment stability cannot hide a whole-interval gate failure")
            if not passed:
                failures.append("equipment_stability:" + row["object_id"])
        contacts = metrics["contact_requirements"]
        if not isinstance(contacts, list) or len(contacts) != len(contact_windows):
            raise ValueError("Exercise contact requirement metrics are incomplete")
        for index, (row, (required, required_duration)) in enumerate(zip(contacts, contact_windows)):
            measures = {"required_duration_s", "observed_duration_s", "qualifying_duration_s", "current_gap_s", "max_gap_s",
                        "load_body_weight_fraction", "qualifying_fraction"}
            if (not isinstance(row, dict) or set(row) != set(required) | measures | {"qualifying_now", "active_now", "satisfied"}
                    or any(row[key] != value for key, value in required.items())
                    or any(type(row[key]) not in (int, float) or not math.isfinite(row[key]) or row[key] < 0 for key in measures)
                    or any(type(row[key]) is not bool for key in ("qualifying_now", "active_now", "satisfied"))):
                raise ValueError("Exercise contact metrics differ from the declared requirements")
            qualifying = row["qualifying_duration_s"] / required_duration
            if (not math.isclose(row["required_duration_s"], required_duration, rel_tol=0, abs_tol=1e-9)
                    or not 0 <= row["qualifying_duration_s"] <= row["observed_duration_s"] + 1e-8 <= required_duration + 2e-8
                    or not row["current_gap_s"] <= row["max_gap_s"] + 1e-8 <= row["observed_duration_s"] + 2e-8
                    or not math.isclose(row["qualifying_fraction"], qualifying, rel_tol=0, abs_tol=1e-8)):
                raise ValueError("Exercise contact duration accounting differs")
            passed = (math.isclose(row["observed_duration_s"], required_duration, abs_tol=1e-8)
                      and qualifying >= gates["contact_duration_fraction"] - 1e-12
                      and row["max_gap_s"] <= gates["max_contact_gap_s"] + 1e-12)
            if row["satisfied"] != passed:
                raise ValueError("Exercise contact success cannot ignore interval coverage or gaps")
            if not passed:
                failures.append("contact_requirement:" + str(index))
        completed = observed == valid == maximum and failure is None
        success = completed and not failures
        if (type(metrics["completed_interval"]) is not bool or metrics["completed_interval"] != completed
                or type(metrics["success"]) is not bool or metrics["success"] != success or metrics["failed_gates"] != failures):
            raise ValueError("Exercise success differs from complete independent task gates")
        if (metrics["failure_reward_penalty"] not in (0., -1.)
                or not math.isclose(metrics["mean_reward"], metrics["reward_integral"] / duration, abs_tol=1e-9)
                or any(not 0 <= metrics[key] <= 1 + 1e-9 for key in ("joint_effort_mean_squared", "root_assistance_mean_squared"))
                or root_assistance_disabled and metrics["root_assistance_mean_squared"] != 0):
            raise ValueError("Exercise effort/reward accounting differs from the action contract")
        expected_terminated = bool(failure or success)
        expected_truncated = bool(completed and not expected_terminated)
        if (type(snapshot["terminated"]) is not bool or type(snapshot["truncated"]) is not bool
                or snapshot["terminated"] != expected_terminated or snapshot["truncated"] != expected_truncated):
            raise ValueError("Exercise completion flags differ from task completion/failure")

    return validate


def _prepare_support_metrics_validator(scene, task, evaluator_sha):
    support_task = _support_module()
    scene_sha, task_sha = canonical_hash(scene), canonical_hash(task)
    action_count = 22 if task["control"]["root_assistance"] == "explicit_3_axis" else 19
    horizon, tick = task["interval"]["horizon_steps"], scene["dt"] / scene["substeps"]
    duration, maximum = horizon * scene["dt"], horizon * scene["substeps"]
    fixed = {"profile": support_task.PROFILE, "scene_sha256": scene_sha, "task_sha256": task_sha,
             "gate_profile": support_task.GATE_PROFILE, "gates_sha256": support_task.GATES_SHA256,
             "gates_provenance": support_task.GATES_PROVENANCE, "source_scope": task["source_scope"],
             "reward_profile": support_task.REWARD_PROFILE, "evaluator_sha256": evaluator_sha,
             "root_assistance": task["control"]["root_assistance"], "virtual_object_forces": "forbidden", "equipment": [],
             "gate_applicability": {"equipment_stability": "not_applicable", "declared_support": "required"}}
    tracking = ("angular_rmse_rad", "root_rmse_model_units", "peak_angular_rmse_rad", "peak_root_rmse_model_units")
    efforts = ("joint_effort_mean_squared", "root_assistance_mean_squared")
    finite = ("simulated_duration_s", "valid_observed_duration_s", "declared_duration_s",
              "max_floor_penetration_model_units", "reward_integral", "mean_reward", "failure_reward_penalty")
    metric_keys = {*fixed, *tracking, *efforts, *finite, "support_requirements", "completed_interval", "success", "failure",
                   "failed_gates", "observed_substeps", "valid_substeps", "clock_status", "reset_protocol"}
    gates = dict(support_task.GATES)
    windows = [(row, (row["end_step"] - row["start_step"]) * scene["dt"])
               for row in task["support_requirements"]]
    def numeric(value, *, lower=0., upper=1e12):
        return type(value) in (int, float) and math.isfinite(value) and lower <= value <= upper

    def validate(snapshot):
        state = snapshot["engine_state"]
        status = snapshot["engine_state_status"]
        options = snapshot["resolved_reset_options"]
        metrics = snapshot["task_metrics"]
        if (not isinstance(metrics, dict) or set(metrics) != metric_keys
                or any(metrics[key] != value for key, value in fixed.items())
                or metrics["reset_protocol"] != _support_reset_protocol(options)
                or any(not numeric(metrics[key], lower=-1e12) for key in finite)):
            raise ValueError("Support metric fields, evaluator identity, criteria or source scope differ")
        observed, valid = metrics["observed_substeps"], metrics["valid_substeps"]
        if (type(observed) is not int or type(valid) is not int or not 0 <= valid <= observed <= maximum
                or not math.isclose(metrics["declared_duration_s"], duration, rel_tol=0, abs_tol=1e-9)
                or not math.isclose(metrics["valid_observed_duration_s"], valid * tick, rel_tol=0, abs_tol=1e-8)
                or not valid * tick - 1e-8 <= metrics["simulated_duration_s"] <= observed * tick + 1e-8):
            raise ValueError("Support duration/substep accounting differs")
        failure, clock = metrics["failure"], metrics["clock_status"]
        if failure is not None and (not isinstance(failure, str) or not 1 <= len(failure) <= 240):
            raise ValueError("Support failures must be bounded explicit reasons")
        if valid < observed and failure is None:
            raise ValueError("Missing support substeps require an explicit failure")
        if (clock not in {"not_observed", "observed", "invalid_substep_clock", "unavailable_after_last_observed_time"}
                or clock == "not_observed" and observed != 0
                or clock in {"invalid_substep_clock", "unavailable_after_last_observed_time"} and failure is None
                or observed == 0 and clock != "not_observed"):
            raise ValueError("Support clock availability contradicts observed samples")
        if (status == "current" and not math.isclose(state["time_s"], metrics["simulated_duration_s"], rel_tol=0, abs_tol=1e-8)
                or status == "last_valid_control_boundary" and (failure is None or state["time_s"] > metrics["simulated_duration_s"] + 1e-8)):
            raise ValueError("Support geometry clock contradicts current/failure state")
        failures = [failure] if failure else []
        for key in (*tracking, "max_floor_penetration_model_units"):
            value = metrics[key]
            if not valid and key in tracking and value is not None:
                raise ValueError("Unobserved support tracking must remain unavailable, not a fabricated zero")
            if value is None:
                if valid or key not in tracking:
                    raise ValueError("Support tracking cannot be missing after valid observations")
            elif not numeric(value):
                raise ValueError("Support tracking and penetration must be nonnegative finite values")
            if (value is None and key in tracking[:2]) or (value is not None and value > gates[key] + 1e-12):
                failures.append(key)
        if valid and any(metrics[mean] > metrics[peak] + 1e-12 for mean, peak in zip(tracking[:2], tracking[2:])):
            raise ValueError("Support mean tracking errors cannot exceed their observed peaks")
        for key in efforts:
            if (not valid and metrics[key] is not None) or (valid and not numeric(metrics[key], upper=1. + 1e-9)):
                raise ValueError("Support effort is unavailable until observed, then bounded by public actions")
        if valid and action_count == 19 and metrics["root_assistance_mean_squared"] != 0:
            raise ValueError("Disabled support root assistance cannot have nonzero effort")
        rows = metrics["support_requirements"]
        if not isinstance(rows, list) or len(rows) != len(windows):
            raise ValueError("Support requirement metric coverage differs")
        for index, (row, (requirement, required_duration)) in enumerate(zip(rows, windows)):
            values = {"required_duration_s", "observed_duration_s", "qualifying_duration_s", "current_gap_s", "max_gap_s"}
            optional = {"load_body_weight_fraction", "peak_load_body_weight_fraction", "qualifying_fraction"}
            flags = {"active_now", "qualifying_now", "satisfied"}
            if (not isinstance(row, dict) or set(row) != set(requirement) | values | optional | flags
                    or any(row[key] != value for key, value in requirement.items())
                    or any(not numeric(row[key], upper=60.) for key in values)
                    or any(row[key] is not None and not numeric(row[key], upper=1e30) for key in optional)
                    or any(type(row[key]) is not bool for key in flags)):
                raise ValueError("Support requirement values or authored identities differ")
            available = max(0., min(observed * tick, requirement["end_step"] * scene["dt"])
                            - requirement["start_step"] * scene["dt"])
            if (not math.isclose(row["required_duration_s"], required_duration, rel_tol=0, abs_tol=1e-9)
                    or not 0 <= row["qualifying_duration_s"] <= row["observed_duration_s"] + 1e-8 <= required_duration + 2e-8
                    or not max(0., available - (observed - valid) * tick) - 1e-8 <= row["observed_duration_s"] <= available + 1e-8
                    or not row["current_gap_s"] <= row["max_gap_s"] + 1e-8 <= row["observed_duration_s"] + 2e-8
                    or row["max_gap_s"] > row["observed_duration_s"] - row["qualifying_duration_s"] + 1e-8):
                raise ValueError("Support interval duration/coverage/gap accounting differs")
            fraction = row["qualifying_duration_s"] / required_duration if row["observed_duration_s"] else None
            if ((fraction is None) != (row["qualifying_fraction"] is None)
                    or fraction is not None and not math.isclose(row["qualifying_fraction"], fraction, rel_tol=0, abs_tol=1e-8)
                    or (row["peak_load_body_weight_fraction"] is None) != (row["observed_duration_s"] == 0)
                    or row["active_now"] != (row["load_body_weight_fraction"] is not None)
                    or row["active_now"] and row["observed_duration_s"] == 0):
                raise ValueError("Missing support loads and measured coverage must remain explicit")
            if valid == observed:
                active = observed > 0 and requirement["start_step"] <= (observed - 1) // scene["substeps"] < requirement["end_step"]
                if row["active_now"] != active:
                    raise ValueError("Support active window differs from its observed substep")
            if (row["load_body_weight_fraction"] is not None and
                    row["load_body_weight_fraction"] > row["peak_load_body_weight_fraction"] + 1e-9
                    or row["qualifying_now"] and (not row["active_now"] or row["current_gap_s"] != 0
                        or row["load_body_weight_fraction"] < requirement["minimum_load_body_weight_fraction"])
                    or row["active_now"] and not row["qualifying_now"] and row["current_gap_s"] < tick - 1e-9
                    or row["qualifying_duration_s"] > 0 and row["peak_load_body_weight_fraction"] < requirement["minimum_load_body_weight_fraction"]):
                raise ValueError("Support current load/qualification contradicts its observed peak or gap")
            passed = (math.isclose(row["observed_duration_s"], required_duration, rel_tol=0, abs_tol=1e-8)
                      and fraction is not None and fraction >= gates["contact_duration_fraction"] - 1e-12
                      and row["max_gap_s"] <= gates["max_contact_gap_s"] + 1e-12)
            if row["satisfied"] != passed:
                raise ValueError("Support success cannot bypass coverage or maximum-gap gates")
            if not passed:
                failures.append("support_requirement:" + str(index))
        completed = observed == valid == maximum and failure is None
        success = completed and not failures
        if (type(metrics["completed_interval"]) is not bool or metrics["completed_interval"] != completed
                or type(metrics["success"]) is not bool or metrics["success"] != success or metrics["failed_gates"] != failures):
            raise ValueError("Support success differs from complete independent task gates")
        penalty = metrics["failure_reward_penalty"]
        reward_before_effort = metrics["reward_integral"] - penalty * scene["dt"]
        if valid:
            reward_before_effort += (.01 * metrics[efforts[0]] + .02 * metrics[efforts[1]]) * valid * tick
        if (penalty not in (0., -1.) or penalty and failure is None
                or not -1e-8 <= reward_before_effort <= valid * tick + 1e-8
                or not math.isclose(metrics["mean_reward"], metrics["reward_integral"] / duration, rel_tol=0, abs_tol=1e-9)):
            raise ValueError("Support reward must reflect observed tracking/load and explicit action effort")
        terminated = bool(failure or success)
        truncated = bool(completed and not terminated)
        if (type(snapshot["terminated"]) is not bool or snapshot["terminated"] != terminated
                or type(snapshot["truncated"]) is not bool or snapshot["truncated"] != truncated):
            raise ValueError("Support completion flags differ from interval success/failure")

    return validate
