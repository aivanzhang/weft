"""Analytical saved metadata only; no actor, source admission, model or physics."""

from copy import deepcopy
from hashlib import sha256
import builtins
import importlib
import json
import os
import socket
import subprocess

import pytest

from weft import camera_shared_policy_receipt as receipt


def pin(text):
    return sha256(text.encode()).hexdigest()


def raw(value):
    return (json.dumps(value, indent=2, allow_nan=False) + "\n").encode()


@pytest.fixture(autouse=True)
def saved_only(monkeypatch):
    def deny(*args, **kwargs):
        raise AssertionError("Native, source, media, network and process work are forbidden")
    original_import = builtins.__import__
    def guarded(name, *args, **kwargs):
        if name.split('.')[0] in {"mujoco", "numpy", "gymnasium", "cv2", "torch", "mediapipe", "requests", "httpx"}:
            deny()
        return original_import(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", guarded)
    for owner, names in ((socket, ("socket", "create_connection")),
                         (subprocess, ("Popen", "run", "check_output")),
                         (os, ("system", "popen", "fork", "posix_spawn", "posix_spawnp"))):
        for name in names:
            if hasattr(owner, name):
                monkeypatch.setattr(owner, name, deny)


def package(*, roots=True, evaluation=True, zero=False, scene=None, task=None):
    current = {name: pin("current:" + name) for name in sorted(receipt.EXECUTION_RUNTIME_REQUIRED)}
    if scene is None:
        body = {"dt": 1 / 30, "source_sha256": pin("source")}
        coupling = {"body_sha256": receipt._canonical(body), "source_sha256": body["source_sha256"],
                    "source_clock_sha256": pin("clock")}
        scene = {"schema_version": 1, "profile": "shared-world-v6", "physical_validation": "unverified",
            "human_spec": body, "camera_coupling": coupling, "dt": 1 / 30, "substeps": 8,
            "objects": [], "target": None, "learned_policy_available": False,
            "task": {"start_offset_s": 0., "horizon_steps": 3, "object_goal": None},
            "action_contract": {"names": [f"joint_{index}" for index in range(19)] + ["root_x", "root_y", "root_z"]}}
        control = {"root_assistance": "explicit_3_axis" if roots else "disabled", "object_forces": "forbidden"}
        scope = {"learning": "unqualified", "physical_validation": "unverified", "source_contact_verified": False,
                 "ground_support_verified": False, "equipment_scoring": "unsupported", "foot_controls": "unsupported"}
        interval = {"start_offset_s": 0., "horizon_steps": 3}
        arguments = {"profile": "shared-exercise-task-v2", "interval": interval, "control": control,
                     "equipment_ids": [], "contact_requirements": [], "source_scope": scope}
        context = {"profile": "camera-shared-metric-context-v1", "valid_legacy_task": False,
            "scene_sha256": receipt._canonical(scene), "kernel_branch": "shared-exercise-task-v2",
            "projection": "identity_22_motor_samples_without_equipment", "control_dt_s": scene["dt"],
            "source_body_dt_s": body["dt"], "substeps": scene["substeps"], "kernel_arguments": arguments,
            "kernel_arguments_sha256": receipt._canonical(arguments)}
        task = {"schema_version": 1, "profile": "camera-shared-task-v1", "scene_sha256": receipt._canonical(scene),
            **coupling, "camera_coupling_sha256": receipt._canonical(coupling), "metric_context": context,
            "metric_context_sha256": receipt._canonical(context), "kernel_evaluator_sha256": current["shared_exercise_task.py"],
            "adapter_implementation_sha256": current["camera_shared_task.py"],
            "evaluator_sha256": receipt._canonical({name: current[name] for name in ("camera_shared_task.py", "shared_exercise_task.py")}),
            "gate_profile": "shared-exercise-gates-v1", "gates_sha256": pin("gates"),
            "interval": interval, "control": control, "scope": scope}
    else:
        scene, task = deepcopy(scene), deepcopy(task)
        current.update({"shared_exercise_task.py": task["kernel_evaluator_sha256"],
                        "camera_shared_task.py": task["adapter_implementation_sha256"]})
    original = {name: current[name] for name in sorted(receipt.ORIGINAL_RUNTIME_REQUIRED)}
    original["recording.py"] = pin("original recording")
    compatibility = {"profile": receipt.COMPATIBILITY_PROFILE,
        "allowed_changed_files": list(receipt.ALLOWED_CHANGED_FILES), "changed_files": {
            "recording.py": {"original_sha256": original["recording.py"], "current_sha256": current["recording.py"]}},
        "verifier_delta": None}
    versions = {name: "analytical-fixture" for name in ("python", "mujoco", "numpy", "gymnasium")}
    policy = {"schema_version": 1, "profile": "camera-shared-policy-v1", "scene_sha256": task["scene_sha256"],
        "task_sha256": receipt._canonical(task), **{key: task[key] for key in receipt.IDENTITIES},
        "runtime_sha256": original, "runtime_versions": versions, "privileged_simulator_features": True,
        "root_assistance": task["control"]["root_assistance"], "root_residual_channels": [], "object_force_channels": [],
        "feedback_coefficients": [[[0. if zero else .1, 0.] for _ in range(3)] for _ in range(2)]}
    policy_raw = raw(policy)
    admission = {"schema_version": 1, "profile": receipt.ADMISSION_PROFILE, "status": "admitted",
        "scene_sha256": task["scene_sha256"], "task_sha256": receipt._canonical(task),
        **{key: task[key] for key in receipt.IDENTITIES}, "policy_sha256": sha256(policy_raw).hexdigest(),
        "training_manifest_sha256": pin("training manifest"),
        "evaluation_manifest_sha256": pin("evaluation manifest") if evaluation else None,
        "original_runtime_sha256": original, "execution_runtime_sha256": current, "runtime_versions": versions,
        "compatibility": compatibility, "compatibility_sha256": receipt._canonical(compatibility),
        "summary": {"selected_feedback_kind": "authored_unchanged" if zero else "searched_phase_feedback_residuals",
            "nonzero_coefficients": 0 if zero else 6, "root_assistance": task["control"]["root_assistance"],
            "learning_qualification": "unqualified_research"}, "privileged_simulator_features": True, "scope": receipt.SCOPE}
    envelope = {"schema_version": 1, "profile": receipt.PROFILE, "engine_scene": scene, "exercise_task": task,
        "dt": scene["dt"], "objects": [], "learned_policy": {"document_utf8": policy_raw.decode(), "sha256": admission["policy_sha256"]},
        "policy_admission": admission}
    kind = "evaluation" if evaluation else "training"
    contract = {"learned_policy_available": True, "policy_sha256": admission["policy_sha256"],
        "camera_policy_admission_sha256": receipt._canonical(admission), "policy_runtime_sha256": original,
        "runtime_versions": versions, "runtime_sha256": current,
        "learning_manifest_sha256": admission["training_manifest_sha256"],
        "camera_policy_evidence_manifest_sha256": admission[kind + "_manifest_sha256"],
        "camera_policy_evidence_kind": kind, "camera_policy_evidence_files_verified": 1}
    members = {"policy.json": admission["policy_sha256"], "camera-policy-admission.json": sha256(raw(admission)).hexdigest(),
        "policy-evidence-run/manifest.json": admission[kind + "_manifest_sha256"], "policy-evidence-run/policy.json": admission["policy_sha256"]}
    if evaluation:
        contract.update(policy_evaluation_manifest_sha256=admission["evaluation_manifest_sha256"],
                        policy_evaluation_sha256=pin("evaluation result"))
        members["policy-evaluation.json"] = contract["policy_evaluation_sha256"]
    return deepcopy({"scene": scene, "task": task, "envelope": envelope, "contract": contract,
                     "native": current, "members": members, "policy": policy, "admission": admission})


@pytest.mark.parametrize("roots,evaluation,zero", [(False, False, True), (True, True, False)])
def test_saved_admission_keeps_original_and_current_runtime_separate(roots, evaluation, zero):
    data = package(roots=roots, evaluation=evaluation, zero=zero)
    before = deepcopy(data)
    result = receipt.saved_camera_policy_bindings(**data)
    assert result["policy_admission_sha256"] == receipt._canonical(data["admission"])
    view = receipt.admission_view(data["admission"])
    assert view["original_runtime_identity_sha256"] != view["execution_runtime_identity_sha256"]
    assert view["summary"]["selected_feedback_kind"] == ("authored_unchanged" if zero else "searched_phase_feedback_residuals")
    validated = receipt.validate_camera_policy_admission(data["admission"], data["scene"], data["task"])
    validated["summary"]["root_assistance"] = "edited"
    assert data == before and data["scene"]["learned_policy_available"] is False


@pytest.mark.parametrize("fault", ["repin_actor", "unknown_runtime", "missing_runtime", "unlisted_delta",
    "wrong_source_clock", "promote_source", "promote_task", "invent_selection", "policy_versions", "oversized_admission"])
def test_invalid_admission_is_not_a_boolean_permission(fault):
    data = package()
    admission = data["admission"]
    if fault == "repin_actor":
        admission["execution_runtime_sha256"]["camera_shared_policy.py"] = pin("changed actor")
        admission["compatibility"]["changed_files"]["camera_shared_policy.py"] = {
            "original_sha256": admission["original_runtime_sha256"]["camera_shared_policy.py"], "current_sha256": pin("changed actor")}
        admission["compatibility_sha256"] = receipt._canonical(admission["compatibility"])
    elif fault == "unknown_runtime":
        admission["execution_runtime_sha256"]["unknown.py"] = pin("unknown")
    elif fault == "missing_runtime":
        del admission["original_runtime_sha256"]["whole_body.py"]
    elif fault == "unlisted_delta":
        admission["compatibility"]["changed_files"] = {}
        admission["compatibility_sha256"] = receipt._canonical(admission["compatibility"])
    elif fault == "wrong_source_clock":
        admission["source_clock_sha256"] = pin("changed clock")
    elif fault == "promote_source":
        data["scene"]["learned_policy_available"] = True
    elif fault == "promote_task":
        data["task"]["scope"]["learning"] = "qualified"
    elif fault == "invent_selection":
        admission["summary"]["nonzero_coefficients"] = 0
    elif fault == "policy_versions":
        admission["runtime_versions"]["mujoco"] = "other"
    else:
        admission["scope"] = "x" * (2 << 20)
    with pytest.raises(ValueError):
        receipt.validate_camera_policy_admission(admission, data["scene"], data["task"],
            policy_document=data["envelope"]["learned_policy"]["document_utf8"].encode())


@pytest.mark.parametrize("fault", ["original_policy", "admission", "training_seal", "evidence_seal", "inventory",
                                  "evaluation_seal", "evaluation_display", "stale_evaluation"])
def test_saved_receipt_rejects_contradictory_display_custody(fault):
    data = package(evaluation=fault != "stale_evaluation")
    if fault == "original_policy":
        data["members"]["policy.json"] = pin("other policy")
    elif fault == "admission":
        data["contract"]["camera_policy_admission_sha256"] = pin("other admission")
    elif fault == "training_seal":
        data["contract"]["learning_manifest_sha256"] = pin("other training")
    elif fault == "evidence_seal":
        data["members"]["policy-evidence-run/manifest.json"] = pin("other evidence")
    elif fault == "inventory":
        data["contract"]["camera_policy_evidence_files_verified"] = 2
    elif fault == "evaluation_seal":
        data["contract"]["policy_evaluation_manifest_sha256"] = pin("other evaluation")
    elif fault == "evaluation_display":
        data["members"]["policy-evaluation.json"] = pin("other result")
    else:
        data["contract"]["policy_evaluation_manifest_sha256"] = pin("stale evaluation")
    with pytest.raises(ValueError):
        receipt.saved_camera_policy_bindings(**data)


def test_verifier_delta_needs_bound_ast_receipt_but_does_not_claim_to_recheck_source():
    data = package()
    admission = data["admission"]
    name = "camera_shared_learning.py"
    admission["execution_runtime_sha256"][name] = pin("narrow verifier change")
    pair = {"original_sha256": admission["original_runtime_sha256"][name], "current_sha256": pin("narrow verifier change")}
    admission["compatibility"]["changed_files"][name] = pair
    admission["compatibility_sha256"] = receipt._canonical(admission["compatibility"])
    with pytest.raises(ValueError, match="AST receipt"):
        receipt.validate_camera_policy_admission(admission, data["scene"], data["task"])
    admission["compatibility"]["verifier_delta"] = {"profile": "camera-shared-saved-verifier-runtime-context-v1",
        **pair, "unchanged_ast_sha256": pin("audited unchanged AST")}
    admission["compatibility_sha256"] = receipt._canonical(admission["compatibility"])
    assert receipt.validate_camera_policy_admission(admission, data["scene"], data["task"]) == admission


def test_helper_import_remains_stdlib_only():
    assert importlib.reload(receipt).PROTOCOL == "camera-shared-policy-full-state-replay-v1"
