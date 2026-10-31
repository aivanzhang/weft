"""Saved external-policy identities; never imports an actor, simulator or learner."""

from copy import deepcopy
from hashlib import sha256
import json
import math
import re

from weft.camera_shared_receipt import IDENTITIES, RUNTIME_REQUIRED


PROFILE = "camera-shared-policy-recording-v1"
PROTOCOL = "camera-shared-policy-full-state-replay-v1"
ADMISSION_PROFILE = "camera-shared-policy-admission-v1"
COMPATIBILITY_PROFILE = "camera-shared-policy-integration-compatibility-v1"
IMPLEMENTATIONS = frozenset({"camera_shared_policy_admission.py", "camera_shared_policy_recording.py",
                             "camera_shared_policy_receipt.py", "camera_shared_policy_lab.py"})
LEARNING_FILES = frozenset({
    "camera_shared_policy.py", "camera_shared_learning.py", "shared_exercise_lab.py", "shared_world_packaging.py",
    "shared_exercise_feedback_residual.py", "shared_exercise_feasibility_learning.py", "shared_exercise_selection.py",
    "shared_exercise_residual_learning.py", "shared_exercise_learning.py", "shared_exercise_policies.py",
    "shared_exercise_feedback_learning.py", "whole_body_learning.py",
})
ORIGINAL_RUNTIME_REQUIRED = RUNTIME_REQUIRED | LEARNING_FILES
EXECUTION_RUNTIME_REQUIRED = ORIGINAL_RUNTIME_REQUIRED | IMPLEMENTATIONS | {
    "camera_shared_capabilities.py", "recording_receipt.py", "lab.py", "capabilities.py",
}
ALLOWED_CHANGED_FILES = ("camera_shared_learning.py", "recording.py", "shared_exercise_lab.py")
SCOPE = ("External bounded-search controller with unchanged source/task qualification; exact actor/physics/scorer "
         "closure, explicitly versioned integration adapters; not physical validation.")
MAX_ADMISSION_BYTES = 128 << 10


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _canonical(value):
    return sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def _hash(value):
    _require(isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None,
             "Camera policy identity must be a SHA-256 digest")
    return value


def _bounded_document(value):
    # Reject oversized leaves before JSON encoding can allocate a large temporary string.
    pending, nodes = [(value, 0)], 0
    while pending:
        item, depth = pending.pop()
        nodes += 1
        _require(nodes <= 4096 and depth <= 12, "Camera policy metadata exceeds its shape bound")
        if isinstance(item, dict):
            _require(len(item) <= 128 and all(isinstance(key, str) and len(key) <= 160 for key in item),
                     "Camera policy metadata keys exceed their bound")
            pending.extend((child, depth + 1) for child in item.values())
        elif isinstance(item, (list, tuple)):
            _require(len(item) <= 128, "Camera policy metadata array exceeds its bound")
            pending.extend((child, depth + 1) for child in item)
        elif isinstance(item, str):
            _require(len(item) <= 8192, "Camera policy metadata string exceeds its bound")
        else:
            _require(item is None or type(item) in (bool, int, float), "Camera policy metadata must be finite JSON")
            _require(type(item) not in (int, float) or -1e30 <= item <= 1e30 and math.isfinite(item),
                     "Camera policy metadata must be finite and bounded")
    _require(len(json.dumps(value, allow_nan=False).encode()) <= MAX_ADMISSION_BYTES,
             "Camera policy admission exceeds 128 KiB")


def _json(raw):
    _require(type(raw) is bytes and len(raw) <= 64 << 10, "Camera policy document exceeds 64 KiB")
    def pairs(items):
        result = {}
        for key, value in items:
            _require(key not in result, "Camera policy JSON contains a duplicate key")
            result[key] = value
        return result
    try:
        value = json.loads(raw, object_pairs_hook=pairs,
            parse_constant=lambda _: (_ for _ in ()).throw(ValueError("Camera policy JSON is nonfinite")))
    except (UnicodeError, json.JSONDecodeError, RecursionError) as error:
        raise ValueError("Camera policy requires bounded strict JSON") from error
    _bounded_document(value)
    return value


def _runtime(value):
    _require(isinstance(value, dict) and 1 <= len(value) <= 96,
             "Camera policy runtime requires a bounded file map")
    _require(all(isinstance(name, str) and re.fullmatch(r"[a-z][a-z0-9_]*\.py", name) for name in value),
             "Camera policy runtime names must be flat Python source names")
    return {name: _hash(pin) for name, pin in value.items()}


def _versions(value):
    _require(isinstance(value, dict) and set(value) == {"python", "mujoco", "numpy", "gymnasium"}
             and all(isinstance(version, str) and 0 < len(version) <= 96 for version in value.values()),
             "Camera policy runtime versions are incomplete")
    return value


def _camera_identity(scene, task, native):
    _require(isinstance(scene, dict) and scene.get("profile") == "shared-world-v6"
             and scene.get("objects") == [] and scene.get("learned_policy_available") is False
             and scene.get("physical_validation") == "unverified", "External policy cannot promote its source scene")
    _require(isinstance(task, dict) and type(task.get("schema_version")) is int and task["schema_version"] == 1
             and task.get("profile") == "camera-shared-task-v1" and set(task) == {
                 "schema_version", "profile", "scene_sha256", *IDENTITIES, "metric_context", "gate_profile",
                 "gates_sha256", "interval", "control", "scope"}, "External policy requires its unchanged camera task")
    scope = task.get("scope")
    _require(isinstance(scope, dict) and scope.get("learning") == "unqualified"
             and scope.get("physical_validation") == "unverified" and scope.get("source_contact_verified") is False
             and scope.get("ground_support_verified") is False and scope.get("equipment_scoring") == "unsupported"
             and scope.get("foot_controls") == "unsupported", "External policy cannot qualify source contact, feet or learning")
    _require(not any(key in document for document in (scene, task)
                     for key in ("avatar", "avatar_asset", "avatar_sha256", "learned_policy", "exercise_policy")),
             "Camera policy cannot introduce avatars or replace its source task")
    pins = {key: _hash(task.get(key)) for key in IDENTITIES}
    body, coupling, context = scene.get("human_spec"), scene.get("camera_coupling"), task.get("metric_context")
    _require(all(isinstance(value, dict) for value in (body, coupling, context)), "Camera identity context is missing")
    _require(task["scene_sha256"] == _canonical(scene)
             and pins["body_sha256"] == _canonical(body) == coupling.get("body_sha256")
             and pins["camera_coupling_sha256"] == _canonical(coupling)
             and pins["source_sha256"] == body.get("source_sha256") == coupling.get("source_sha256")
             and pins["source_clock_sha256"] == coupling.get("source_clock_sha256")
             and pins["metric_context_sha256"] == _canonical(context), "Camera source/body/clock/context identities differ")
    _require(pins["kernel_evaluator_sha256"] == native["shared_exercise_task.py"]
             and pins["adapter_implementation_sha256"] == native["camera_shared_task.py"]
             and pins["evaluator_sha256"] == _canonical({name: native[name] for name in (
                 "camera_shared_task.py", "shared_exercise_task.py")}), "Camera scorer or task implementation changed")
    control, interval = task.get("control"), scene.get("task")
    _require(isinstance(control, dict) and set(control) == {"root_assistance", "object_forces"}
             and control.get("root_assistance") in {"disabled", "explicit_3_axis"}
             and control.get("object_forces") == "forbidden" and isinstance(interval, dict)
             and task["interval"] == {"start_offset_s": interval.get("start_offset_s"),
                                      "horizon_steps": interval.get("horizon_steps")},
             "Camera policy changed the task interval or control authority")
    arguments = context.get("kernel_arguments")
    _require(context.get("profile") == "camera-shared-metric-context-v1"
             and context.get("valid_legacy_task") is False and context.get("scene_sha256") == task["scene_sha256"]
             and context.get("kernel_branch") == "shared-exercise-task-v2"
             and context.get("projection") == "identity_22_motor_samples_without_equipment"
             and context.get("control_dt_s") == scene.get("dt") and context.get("source_body_dt_s") == body.get("dt")
             and context.get("substeps") == scene.get("substeps") and isinstance(arguments, dict)
             and context.get("kernel_arguments_sha256") == _canonical(arguments)
             and arguments == {"profile": "shared-exercise-task-v2", "interval": task["interval"],
                 "control": control, "equipment_ids": [], "contact_requirements": [], "source_scope": scope},
             "Camera policy changed the original human-only scoring context")
    return pins


def validate_camera_policy_admission(admission, scene, task, *, policy_document=None, execution_runtime_sha256=None):
    """Validate saved metadata only; callers separately admit source geometry and current runtime bytes."""
    _bounded_document(admission)
    _require(isinstance(admission, dict) and set(admission) == {
        "schema_version", "profile", "status", "scene_sha256", "task_sha256", *IDENTITIES,
        "policy_sha256", "training_manifest_sha256", "evaluation_manifest_sha256", "original_runtime_sha256",
        "execution_runtime_sha256", "runtime_versions", "compatibility", "compatibility_sha256", "summary",
        "privileged_simulator_features", "scope"}, "Camera policy admission fields differ")
    _require(type(admission["schema_version"]) is int and admission["schema_version"] == 1
             and admission["profile"] == ADMISSION_PROFILE and admission["status"] == "admitted"
             and admission["privileged_simulator_features"] is True and admission["scope"] == SCOPE,
             "Camera policy admission profile or qualification differs")
    old, current = _runtime(admission["original_runtime_sha256"]), _runtime(admission["execution_runtime_sha256"])
    _require(ORIGINAL_RUNTIME_REQUIRED <= old.keys() <= EXECUTION_RUNTIME_REQUIRED
             and current.keys() == EXECUTION_RUNTIME_REQUIRED, "Camera policy runtime dependency closure differs")
    if execution_runtime_sha256 is not None:
        _require(_runtime(execution_runtime_sha256) == current, "Camera policy execution runtime differs from the lab")
    versions = _versions(admission["runtime_versions"])
    pins = _camera_identity(scene, task, current)
    _require(admission["scene_sha256"] == task["scene_sha256"] and admission["task_sha256"] == _canonical(task)
             and all(admission[key] == value for key, value in pins.items()), "Camera policy admission is for another scene/task")
    for key in ("policy_sha256", "training_manifest_sha256", "compatibility_sha256"):
        _hash(admission[key])
    if admission["evaluation_manifest_sha256"] is not None:
        _hash(admission["evaluation_manifest_sha256"])
    compatibility = admission["compatibility"]
    changed = {name: {"original_sha256": pin, "current_sha256": current[name]}
               for name, pin in old.items() if pin != current[name]}
    _require(isinstance(compatibility, dict) and set(compatibility) == {
                 "profile", "allowed_changed_files", "changed_files", "verifier_delta"}
             and compatibility["profile"] == COMPATIBILITY_PROFILE
             and compatibility["allowed_changed_files"] == list(ALLOWED_CHANGED_FILES)
             and set(changed) <= set(ALLOWED_CHANGED_FILES) and compatibility["changed_files"] == changed
             and admission["compatibility_sha256"] == _canonical(compatibility),
             "Camera policy compatibility cannot silently repin actor, physics or scoring")
    delta = compatibility["verifier_delta"]
    if "camera_shared_learning.py" in changed:
        pair = changed["camera_shared_learning.py"]
        _require(isinstance(delta, dict) and set(delta) == {
                     "profile", "original_sha256", "current_sha256", "unchanged_ast_sha256"}
                 and delta["profile"] == "camera-shared-saved-verifier-runtime-context-v1"
                 and all(delta[key] == value for key, value in pair.items()), "Saved verifier change needs its exact AST receipt")
        _hash(delta["unchanged_ast_sha256"])
    else:
        _require(delta is None, "Unchanged saved verifier cannot claim a compatibility rewrite")
    summary = admission["summary"]
    _require(isinstance(summary, dict) and set(summary) == {
                 "selected_feedback_kind", "nonzero_coefficients", "root_assistance", "learning_qualification"}
             and type(summary["nonzero_coefficients"]) is int and 0 <= summary["nonzero_coefficients"] <= 12
             and summary["selected_feedback_kind"] == ("searched_phase_feedback_residuals"
                 if summary["nonzero_coefficients"] else "authored_unchanged")
             and summary["root_assistance"] == task["control"]["root_assistance"]
             and summary["learning_qualification"] == "unqualified_research", "Camera policy selection or root labels differ")
    if policy_document is not None:
        policy = _json(policy_document)
        _require(sha256(policy_document).hexdigest() == admission["policy_sha256"]
                 and policy.get("profile") == "camera-shared-policy-v1"
                 and policy.get("scene_sha256") == admission["scene_sha256"]
                 and policy.get("task_sha256") == admission["task_sha256"]
                 and all(policy.get(key) == value for key, value in pins.items())
                 and policy.get("runtime_sha256") == old and policy.get("runtime_versions") == versions
                 and policy.get("privileged_simulator_features") is True
                 and policy.get("root_assistance") == summary["root_assistance"]
                 and policy.get("root_residual_channels") == [] and policy.get("object_force_channels") == [],
                 "Camera admission differs from the exact original policy or privileged control authority")
        values = policy.get("feedback_coefficients")
        _require(isinstance(values, list) and len(values) == 2
                 and all(isinstance(phase, list) and len(phase) == 3
                     and all(isinstance(group, list) and len(group) == 2
                         and all(type(value) in (int, float) and -.25 <= value <= .25 for value in group)
                         for group in phase) for phase in values), "Camera policy coefficients exceed their exact bound")
        _require(sum(value != 0 for phase in values for group in phase for value in group) == summary["nonzero_coefficients"],
                 "Camera admission selection label differs from original policy coefficients")
    return deepcopy(admission)


def admission_view(admission):
    """Present a previously validated admission without exposing duplicate editable runtime maps."""
    return {key: deepcopy(admission[key]) for key in (
        "profile", "status", "policy_sha256", "scene_sha256", "task_sha256", *IDENTITIES,
        "training_manifest_sha256", "evaluation_manifest_sha256", "compatibility_sha256",
        "privileged_simulator_features", "summary")} | {
        "admission_sha256": _canonical(admission),
        "original_runtime_identity_sha256": _canonical(admission["original_runtime_sha256"]),
        "execution_runtime_identity_sha256": _canonical(admission["execution_runtime_sha256"])}


def saved_camera_policy_bindings(scene, task, envelope, contract, native, members, *, policy, admission):
    """Join sealed receipt metadata; this does not repeat source/search or native admission."""
    _require(isinstance(envelope, dict) and type(envelope.get("schema_version")) is int
             and envelope["schema_version"] == 1 and envelope.get("profile") == PROFILE and set(envelope) == {
                 "schema_version", "profile", "engine_scene", "exercise_task", "dt", "objects", "learned_policy", "policy_admission"}
             and envelope["engine_scene"] == scene and envelope["exercise_task"] == task
             and envelope["objects"] == [] and envelope["dt"] == scene["dt"], "Camera policy recording envelope differs")
    embedded = envelope["learned_policy"]
    _require(isinstance(embedded, dict) and set(embedded) == {"document_utf8", "sha256"}
             and isinstance(embedded["document_utf8"], str) and len(embedded["document_utf8"]) <= 64 << 10,
             "Camera recording needs the exact bounded policy document")
    raw = embedded["document_utf8"].encode()
    admitted = validate_camera_policy_admission(admission, scene, task,
        policy_document=raw, execution_runtime_sha256=native)
    pin = _canonical(admitted)
    _require(_json(raw) == policy and embedded["sha256"] == admission["policy_sha256"]
             and envelope["policy_admission"] == admitted, "Camera recording policy/admission differs from sealed metadata")
    _require(contract.get("learned_policy_available") is True
             and contract.get("policy_sha256") == admitted["policy_sha256"]
             and contract.get("camera_policy_admission_sha256") == pin
             and contract.get("learning_manifest_sha256") == admitted["training_manifest_sha256"]
             and contract.get("policy_runtime_sha256") == admitted["original_runtime_sha256"]
             and contract.get("runtime_versions") == admitted["runtime_versions"], "Camera lab policy/runtime identity differs")
    evidence_kind = "evaluation" if admitted["evaluation_manifest_sha256"] is not None else "training"
    evidence_pin = admitted[evidence_kind + "_manifest_sha256"]
    _require(isinstance(members, dict) and members.get("policy.json") == admitted["policy_sha256"]
             and "camera-policy-admission.json" in members
             and members.get("policy-evidence-run/manifest.json") == evidence_pin
             and contract.get("camera_policy_evidence_manifest_sha256") == evidence_pin
             and contract.get("camera_policy_evidence_kind") == evidence_kind,
             "Camera receipt lacks its exact original policy evidence seal")
    if evidence_kind == "evaluation":
        _require(contract.get("policy_evaluation_manifest_sha256") == admitted["evaluation_manifest_sha256"]
                 and members.get("policy-evaluation.json") == _hash(contract.get("policy_evaluation_sha256")),
                 "Camera evaluation display differs from its sealed policy evidence")
    else:
        _require("policy-evaluation.json" not in members and not any(key in contract for key in (
                     "policy_evaluation_manifest_sha256", "policy_evaluation_sha256", "policy_evaluation_files_verified")),
                 "Training-only camera admission cannot display an evaluation")
    count = sum(name.startswith("policy-evidence-run/") and name != "policy-evidence-run/manifest.json" for name in members)
    _require(type(contract.get("camera_policy_evidence_files_verified")) is int
             and contract["camera_policy_evidence_files_verified"] == count and 1 <= count <= 512,
             "Camera policy evidence inventory differs from its admission")
    _require(not any(key in contract for key in ("avatar", "avatar_asset", "avatar_sha256")),
             "Camera policy receipt cannot admit an avatar")
    return {key: admitted[key] for key in IDENTITIES} | {"policy_admission_sha256": pin}
