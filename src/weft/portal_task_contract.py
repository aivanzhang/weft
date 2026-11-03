"""Pure portal task requests and display receipts; sealed loaders retain source authority."""

from hashlib import sha256
import json
import math
import re
import time

PROFILE = "portal-moving-task-action-v1"
MAX_REQUEST_BYTES = 16 << 10
MAX_REPORT_BYTES = 64 << 10
MAX_CONTROLS = 20_000
CONTROLLERS = ("zero", "authored_feedback", "random", "frozen_policy")
_SCORERS = {"moving_equipment_task.py", "shared_exercise_task.py", "whole_body.py"}
_BINDINGS = {"parent_project_id", "parent_job_sha256", "parent_status_sha256", "parent_lab_config_sha256",
    "parent_manifest_sha256", "source_sha256", "scene_sha256", "family_sha256", "selection_sha256",
    "action_contract_sha256", "task_sha256", "scorer_sha256"}


def _require(condition, message):
    if not condition:
        raise ValueError("Portal task: " + message)


def _keys(value, fields):
    _require(type(value) is dict and set(value) == set(fields), "closed fields differ")


def _pin(value, length=64):
    _require(type(value) is str and re.fullmatch(f"[a-f0-9]{{{length}}}", value), "invalid exact identity")
    return value


def _integer(value, low, high):
    _require(type(value) is int and low <= value <= high, "integer setting outside bound")
    return value


def _number(value, low, high):
    _require(type(value) in (int, float) and low <= value <= high and math.isfinite(value), "number outside bound")
    return value


def _encode(value, maximum=MAX_REPORT_BYTES, *, compact=True):
    result = bytearray(); deadline = time.monotonic()+2
    try:
        for chunk in json.JSONEncoder(sort_keys=True, separators=(",", ":") if compact else None, allow_nan=False).iterencode(value):
            raw = chunk.encode()
            _require(len(result)+len(raw) <= maximum and time.monotonic() < deadline, "JSON work/byte bound exceeded")
            result.extend(raw)
    except (TypeError, RecursionError, OverflowError) as error:
        raise ValueError("Portal task requires bounded JSON") from error
    return bytes(result)


def _parse(raw, maximum=MAX_REPORT_BYTES):
    _require(type(raw) is bytes and 0 < len(raw) <= maximum, "bounded original JSON bytes required")
    def pairs(items):
        result = {}
        for key, value in items:
            _require(key not in result, "duplicate JSON field")
            result[key] = value
        return result
    def invalid(value):
        raise ValueError("Portal task rejects nonfinite JSON")
    try:
        value = json.loads(raw, object_pairs_hook=pairs, parse_constant=invalid)
    except (UnicodeError, RecursionError) as error:
        raise ValueError("Portal task requires strict JSON") from error
    _require(type(value) is dict, "JSON object required")
    _encode(value, maximum)
    return value


def _copy(value, maximum=MAX_REPORT_BYTES):
    return _parse(_encode(value, maximum), maximum)


def canonical_sha256(value):
    return sha256(_encode(value, 2 << 20, compact=False)).hexdigest()


def validate_task_create_request(value):
    _keys(value, {"request_id", "parent_manifest_sha256", "task"})
    _pin(value["request_id"], 32); _pin(value["parent_manifest_sha256"])
    goal = value["task"]
    _keys(goal, {"target_object_id", "target_position_world", "position_tolerance_world",
                "speed_tolerance_world_s", "hold_duration_s"})
    _require(type(goal["target_object_id"]) is str and re.fullmatch(r"[a-zA-Z0-9_-]{1,64}", goal["target_object_id"]), "invalid object ID")
    _require(type(goal["target_position_world"]) is list and len(goal["target_position_world"]) == 3, "three explicit coordinates required")
    for x in goal["target_position_world"]: _number(x, 0, 1e6)
    _number(goal["position_tolerance_world"], 1e-5, .5)
    _number(goal["speed_tolerance_world_s"], 1e-5, 1.)
    _number(goal["hold_duration_s"], 1e-6, 60.)
    # The ordinary task constructor checks actual world bounds, object membership and native duration.
    return _copy(value, MAX_REQUEST_BYTES)


def validate_task_learning_request(value, *, horizon_steps=None):
    _keys(value, {"request_id", "task_sha256", "training", "evaluation"})
    _pin(value["request_id"], 32); _pin(value["task_sha256"])
    training = value["training"]
    _keys(training, {"seed", "generations", "population", "max_seconds"})
    _integer(training["seed"], 0, 2**31-1)
    _integer(training["generations"], 1, 4); _integer(training["population"], 4, 12)
    _number(training["max_seconds"], 1, 60)
    evaluation = value["evaluation"]
    trials = training["generations"]*training["population"]+4
    if evaluation is not None:
        _keys(evaluation, {"evaluation_seeds", "position_jitter", "velocity_jitter", "max_seconds"})
        seeds = evaluation["evaluation_seeds"]
        _require(type(seeds) is list and 2 <= len(seeds) <= 16, "2–16 evaluation seeds required")
        for seed in seeds: _integer(seed, 0, 2**31-1)
        _require(len(set(seeds)) == len(seeds) and training["seed"] not in seeds, "evaluation seeds must be unique and separate from training")
        _number(evaluation["position_jitter"], 0, .05); _number(evaluation["velocity_jitter"], 0, .1)
        _require(evaluation["position_jitter"]+evaluation["velocity_jitter"] > 0, "different seeds alone do not change starts")
        _number(evaluation["max_seconds"], 1, 60)
        trials += 4*len(seeds)
    if horizon_steps is not None:
        _integer(horizon_steps, 1, MAX_CONTROLS)
        _require(trials*horizon_steps <= MAX_CONTROLS, "training and matched evaluation exceed the combined 20000-control allowance")
    return _copy(value, MAX_REQUEST_BYTES)


def build_task_action_receipt(action, request, bindings):
    _require(action in {"create", "train_evaluate"}, "unknown action")
    request = (validate_task_create_request(request) if action == "create" else validate_task_learning_request(request))
    _keys(bindings, _BINDINGS)
    for key, value in bindings.items():
        if key == "scorer_sha256":
            if value is None:
                _require(action == "create", "training requires the fixed scorer")
            else:
                _keys(value, _SCORERS)
                for pin in value.values(): _pin(pin)
        elif key == "task_sha256" and value is None:
            _require(action == "create", "training requires its exact task")
        else: _pin(value, 32 if key == "parent_project_id" else 64)
    key = "parent_manifest_sha256" if action == "create" else "task_sha256"
    _require(request[key] == bindings[key], "request differs from parent/task identity")
    return _copy({"schema_version": 1, "profile": PROFILE, "action": action, "request": request,
        "request_sha256": canonical_sha256(request), "bindings": bindings,
        "api_calls_authorized": 0, "policy_installation_authorized": False,
        "scope": "Explicit authored task or bounded development search. Scene hash retains world/force limits; source qualification requires the sealed loader. No recovered intent or physical-accuracy claim."})


def validate_task_action_receipt(receipt):
    _require(type(receipt) is dict, "receipt object required")
    expected = build_task_action_receipt(receipt.get("action"), receipt.get("request"), receipt.get("bindings"))
    _require(_encode(receipt) == _encode(expected), "action receipt differs from its closed request/bindings")
    return expected


def task_options(scene, *, parent_manifest_sha256, task=None):
    """Call on already admitted cached data, not as source admission or on every poll."""
    _pin(parent_manifest_sha256)
    scene = _copy(scene, 2 << 20)
    contract = scene.get("action_contract", {}); interval = scene.get("task", {})
    _require(type(contract) is dict and type(interval) is dict, "world action/interval contract required")
    ids = contract.get("dynamic_object_ids")
    _require(type(ids) is list and 1 <= len(ids) <= 8, "bounded dynamic object inventory required")
    _require(all(type(x) is str and re.fullmatch(r"[a-zA-Z0-9_-]{1,64}", x) for x in ids), "invalid object inventory")
    _require(len(set(ids)) == len(ids), "duplicate dynamic object ID")
    objects = scene.get("objects"); bounds = scene.get("bounds")
    _require(type(objects) is list and len(objects) == len(ids) and type(bounds) is list and len(bounds) == 3, "world inventory/bounds required")
    for x in bounds: _number(x, 1e-6, 1e6)
    _require(all(type(obj) is dict and type(obj.get("id")) is str for obj in objects)
             and {obj["id"] for obj in objects} == set(ids), "object inventory differs")
    names = contract.get("names")
    _require(type(names) is list and 1 <= len(names) <= 512
             and all(type(n) is str and 1 <= len(n) <= 128 for n in names), "bounded action names required")
    dt = _number(scene.get("dt"), 1e-6, 1)
    horizon = _integer(interval.get("horizon_steps"), 1, MAX_CONTROLS)
    _require(interval.get("start_offset_s") == 0, "task requires the original complete start")
    native = interval.get("native_interval", {})
    _require(type(native) is dict, "native interval metadata differs")
    duration = _number(native.get("duration_s", dt*horizon), 1e-6, 60)
    inventory = []
    for obj in objects:
        position = obj.get("position")
        _require(type(position) is list and len(position) == 3, "initial object position required")
        for x, limit in zip(position, bounds): _number(x, 0, limit)
        inventory.append({"id": obj["id"], "initial_position_world": position})
    task_pin = None
    if task is not None:
        task = _copy(task)
        _require(type(task.get("binding")) is dict and task["binding"].get("scene_sha256") == canonical_sha256(scene), "task/world differs")
        task_pin = canonical_sha256(task)
    return _copy({"schema_version": 1, "profile": "portal-moving-task-options-v1",
        "parent_manifest_sha256": parent_manifest_sha256, "scene_sha256": canonical_sha256(scene),
        "task_sha256": task_pin, "objects": inventory, "bounds_world": bounds,
        "coordinates": "Simulation world units; not measured physical coordinates",
        "duration_s": duration, "minimum_hold_s": min(dt, duration), "horizon_steps": horizon,
        "action_names": names, "action_contract_sha256": canonical_sha256(contract),
        "training_optional": True, "policy_auto_install": False,
        "scope": "Authored virtual-force-assisted placement; task success and contact remain unproven."})


def _text(value, maximum=500, *, nullable=False):
    if value is None and nullable:
        return None
    _require(type(value) is str and len(value) <= maximum, "bounded text required")
    return value


def _metrics(value, task_sha256):
    if value is None:
        return None
    _require(type(value) is dict and value.get("task_sha256") == task_sha256, "metrics differ from fixed task")
    _require(type(value.get("success")) is bool and type(value.get("completed_interval")) is bool, "explicit task outcome required")
    criteria = value.get("failed_criteria")
    _require(type(criteria) is list and len(criteria) <= 32, "bounded failed criteria required")
    for label in criteria: _text(label, 128)
    failure = _text(value.get("failure"), nullable=True)
    _require(not value["success"] or value["completed_interval"] and not criteria and failure is None,
             "success cannot omit incomplete or failed task criteria")
    keys = ("elapsed_s", "duration_s", "object_distance_world", "object_speed_world_s",
            "terminal_consecutive_hold_s", "angular_rmse_rad", "root_rmse_model_units",
            "peak_angular_rmse_rad", "peak_root_rmse_model_units", "max_floor_penetration_model_units", "mean_reward")
    result = {key: value.get(key) for key in keys}
    for key, item in result.items():
        if item is not None: _number(item, -1e6 if key == "mean_reward" else 0, 1e6)
    result.update(success=value["success"], completed_interval=value["completed_interval"],
                  failed_criteria=list(criteria), failure=failure)
    return result


def _trial(value, task_sha256):
    _require(type(value) is dict, "trial object required")
    return {"status": _text(value.get("status"), 64), "failure": _text(value.get("failure"), nullable=True),
        "control_steps": _integer(value.get("control_steps"), 0, MAX_CONTROLS),
        "metrics": _metrics(value.get("metrics"), task_sha256)}


def _source(document, task_sha256, scene_sha256):
    source = document.get("source")
    _require(type(source) is dict and source.get("task_sha256") == task_sha256
             and source.get("scene_sha256") == scene_sha256, "report source/task differs")
    for key in ("task_manifest_sha256", "parent_manifest_sha256", "task_sha256", "scene_sha256"):
        _pin(source.get(key))


def summarize_task_learning(training_result_raw, training_evaluation_raw, *, task_sha256, scene_sha256,
                            evaluation_result_raw=None):
    """Project exact sealed output bytes; this is not a new scorer or replay verifier."""
    _pin(task_sha256); _pin(scene_sha256)
    training = _parse(training_result_raw)
    nominal = _parse(training_evaluation_raw, 2 << 20)
    _source(nominal, task_sha256, scene_sha256)
    _require(training.get("status") == "complete" and nominal.get("profile") == "moving-equipment-feedback-policy-v1",
             "completed learning artifacts required; preserve failures separately")
    policy = _pin(training.get("policy_sha256"))
    _require(nominal.get("policy_sha256") == policy and nominal.get("independent_changed_start_evaluation") == "unavailable",
             "nominal policy/source scope differs")
    comparisons = nominal.get("comparisons")
    _require(type(comparisons) is list and len(comparisons) == 4
             and [r.get("controller") if type(r) is dict else None for r in comparisons] == list(CONTROLLERS),
             "all four ordered nominal controller results required")
    rows = []
    for row in comparisons:
        _require(type(row.get("reused_training_trial")) is bool, "baseline reuse must remain explicit")
        rows.append({"controller": row["controller"], "reused_training_trial": row["reused_training_trial"],
                     "trial": _trial(row.get("trial"), task_sha256)})
    kind = nominal.get("selected_kind")
    _require(kind in {"zero_unchanged", "authored_unchanged", "searched_feedback_gains"}, "unknown selected policy kind")
    selected_success = nominal.get("selected_policy_success")
    _require(type(selected_success) is bool and selected_success is (rows[-1]["trial"]["metrics"] or {}).get("success", False),
             "selected success differs from frozen trial")
    used = _integer(training.get("control_steps"), 0, MAX_CONTROLS)
    maximum = _integer(training.get("maximum_control_steps"), 1, MAX_CONTROLS)
    _require(used <= maximum, "reported training controls exceed reservation")
    changed = None
    if evaluation_result_raw is not None:
        evaluation = _parse(evaluation_result_raw)
        _source(evaluation, task_sha256, scene_sha256)
        _require(all(evaluation["source"].get(key) == value for key, value in nominal["source"].items()),
                 "changed-start report differs from the original admitted source package")
        _require(evaluation.get("status") == "complete" and evaluation.get("profile") == "moving-equipment-object-reset-evaluation-v1"
                 and evaluation.get("policy_sha256") == policy and evaluation.get("final_holdout_independence") == "not_established"
                 and evaluation.get("physical_validation") == "unverified", "changed-start policy or scope differs")
        requested = _integer(evaluation.get("requested_starts"), 2, 16)
        admitted = _integer(evaluation.get("admitted_changed_starts"), 0, requested)
        excluded = _integer(evaluation.get("excluded_or_refused_starts"), 0, requested)
        _require(admitted+excluded == requested, "actual changed-start denominator differs")
        evaluated = evaluation.get("comparisons")
        _require(type(evaluated) is list and len(evaluated) == 4
                 and [r.get("controller") if type(r) is dict else None for r in evaluated] == list(CONTROLLERS),
                 "all matched changed-start controllers required")
        for row in evaluated:
            _keys(row, {"controller", "successes", "requested_starts", "admitted_starts"})
            _integer(row["successes"], 0, admitted)
            _require(type(row["requested_starts"]) is int and type(row["admitted_starts"]) is int
                     and row["requested_starts"] == requested and row["admitted_starts"] == admitted, "controller denominator differs")
        eval_used = _integer(evaluation.get("control_steps"), 0, MAX_CONTROLS)
        eval_max = _integer(evaluation.get("maximum_control_steps"), 1, MAX_CONTROLS)
        _require(eval_used <= eval_max and maximum+eval_max <= MAX_CONTROLS, "combined report exceeds frozen work bounds")
        changed = {"requested_starts": requested, "admitted_changed_starts": admitted,
            "excluded_or_refused_starts": excluded, "comparisons": evaluated,
            "control_steps": eval_used, "maximum_control_steps": eval_max,
            "protocol_sha256": _pin(evaluation.get("protocol_sha256")),
            "evidence": "matched_simulated_development_starts" if admitted else "unavailable_no_admitted_changed_starts"}
    return _copy({"schema_version": 1, "profile": "portal-moving-task-learning-report-v1", "status": "complete",
        "task_sha256": task_sha256, "scene_sha256": scene_sha256, "policy_sha256": policy,
        "input_sha256": {"training_result": sha256(training_result_raw).hexdigest(),
            "training_evaluation": sha256(training_evaluation_raw).hexdigest(),
            "changed_start_result": sha256(evaluation_result_raw).hexdigest() if evaluation_result_raw is not None else None},
        "training": {"control_steps": used, "maximum_control_steps": maximum, "comparisons": rows,
                     "selected_kind": kind, "selected_policy_success": selected_success,
                     "independent_changed_starts": False}, "evaluation": changed,
        "policy_installed": False, "physical_validation": "unverified", "final_holdout_independence": "not_established",
        "scope": "Frozen task/world and development comparisons only. Failed baselines and refused starts remain visible; no controller is installed."})


def summarize_task_failure(failure_raw, *, stage):
    _require(stage in {"admission", "create", "training", "evaluation", "publication"}, "unknown failure stage")
    failure = _parse(failure_raw, 2 << 20)
    reason = failure.get("failure", failure.get("reason", failure.get("error")))
    reason = _text(reason, 1000)
    return {"schema_version": 1, "profile": "portal-moving-task-failure-v1", "status": "failed", "stage": stage,
        "failure_sha256": sha256(failure_raw).hexdigest(), "failure": reason,
        "policy_installed": False, "task_success": False,
        "scope": "Partial trials and original failure artifact remain retained; failure is not task infeasibility."}
