"""Normalize selected, hash-verified lab metadata without loading a simulator."""

import math
import re

from weft.evidence_scorecard import build_evidence_scorecard
from weft.io import canonical_hash
from weft.rigid_lab_evidence import normalize_rigid_lab_evidence


_SHA = re.compile(r"[0-9a-f]{64}")
_CONTROLLERS = {"learned", "authored", "coast", "random"}
_POLICIES = {f"shared-exercise-policy-v{version}" for version in range(1, 5)}


def _require(condition, message):
    if not condition:
        raise ValueError("Invalid saved lab evidence: " + message)


def _hash(value, label):
    _require(isinstance(value, str) and _SHA.fullmatch(value), label + " identity")
    return value


def _camera_policy(scene, task, contract, compact, files, admission, policy_document):
    from weft.camera_shared_policy_receipt import validate_camera_policy_admission

    _require(type(policy_document) is bytes, "exact camera policy bytes")
    admitted = validate_camera_policy_admission(admission, scene, task, policy_document=policy_document,
        execution_runtime_sha256=contract.get("runtime_sha256"))
    pin = canonical_hash(admitted)
    _require(contract.get("runtime_sha256") == admitted["execution_runtime_sha256"]
             and contract.get("policy_runtime_sha256") == admitted["original_runtime_sha256"]
             and contract.get("runtime_versions") == admitted["runtime_versions"]
             and contract.get("learning_manifest_sha256") == admitted["training_manifest_sha256"]
             and contract.get("camera_policy_admission_sha256") == compact.get("admission_sha256") == pin
             and compact.get("admission_profile") == admitted["profile"]
             and compact.get("privileged_simulator_features") is True
             and compact.get("selected_feedback_kind") == admitted["summary"]["selected_feedback_kind"]
             and compact.get("root_assistance") == admitted["summary"]["root_assistance"]
             and compact.get("learning_qualification") == "unqualified_research", "camera policy admission binding")
    kind = "evaluation" if admitted["evaluation_manifest_sha256"] is not None else "training"
    evidence_pin = admitted[kind + "_manifest_sha256"]
    _require(contract.get("camera_policy_evidence_kind") == kind
             and contract.get("camera_policy_evidence_manifest_sha256") == evidence_pin
             and files.get("policy-evidence-run/manifest.json") == evidence_pin
             and compact.get("evaluation_status") == ("evaluated" if kind == "evaluation" else "not_attached"),
             "camera policy evidence identity")
    count = sum(name.startswith("policy-evidence-run/") and name != "policy-evidence-run/manifest.json" for name in files)
    _require(type(contract.get("camera_policy_evidence_files_verified")) is int
             and contract["camera_policy_evidence_files_verified"] == count and 1 <= count <= 512,
             "camera policy evidence inventory")
    if kind == "training":
        _require("policy-evaluation.json" not in files and not any(name in contract for name in
            ("policy_evaluation_manifest_sha256", "policy_evaluation_sha256", "policy_evaluation_files_verified")),
            "training-only camera policy cannot claim evaluation")


def _shared_exercise(summary, scene, task, contract, policy, files, *, camera_admission=None, policy_document=None):
    _require(type(contract.get("schema_version")) is int and contract["schema_version"] == 1
             and contract.get("profile") == "shared-world"
             and contract.get("control_profile") == "shared_exercise", "exercise contract")
    scene_sha = summary.get("scene_sha256")
    camera = scene.get("profile") == "shared-world-v6"
    support = scene.get("profile") == "shared-world-v4"
    _require(scene.get("profile") in ("shared-world-v1", "shared-world-v2", "shared-world-v3", "shared-world-v4", "shared-world-v6")
             and scene_sha is not None and contract.get("scene_sha256") == scene_sha, "exercise scene binding")
    task_sha = canonical_hash(task)
    expected_profile = ("camera-shared-task-v1" if camera else
                        "shared-support-task-v1" if support else
                        "shared-exercise-task-v2" if scene["profile"] == "shared-world-v3" else "shared-exercise-task-v1")
    _require(type(task.get("schema_version")) is int and task["schema_version"] == 1
             and task.get("profile") == expected_profile and task.get("scene_sha256") == scene_sha
             and contract.get("task_sha256") == task_sha, "exercise task binding")
    if support:
        # This pure contract checks authored support scope and fixed gates, not native or physical validity.
        from weft.shared_support_task import validate_shared_support_task

        validate_shared_support_task(task, scene)
    summary["task_sha256"] = task_sha
    control = task.get("control")
    _require(isinstance(control, dict) and control.get("root_assistance") in ("disabled", "explicit_3_axis"),
             "root assistance declaration")
    summary["requested_options"]["root_assistance"] = control["root_assistance"] == "explicit_3_axis"
    available = contract.get("learned_policy_available")
    _require(type(available) is bool, "learned policy availability")
    compact = contract.get("exercise_policy")
    if not available:
        _require(compact is None, "unavailable policy has evaluation metadata")
        return
    policy_sha = _hash(files.get("policy.json"), "policy")
    profiles = {"camera-shared-policy-v1"} if camera else {"shared-support-policy-v1"} if support else _POLICIES
    _require(isinstance(compact, dict) and type(policy.get("schema_version")) is int
             and policy["schema_version"] == 1 and policy.get("profile") in profiles
             and compact.get("profile") == policy["profile"]
             and contract.get("policy_sha256") == compact.get("policy_sha256") == policy_sha
             and policy.get("scene_sha256") == scene_sha and policy.get("task_sha256") == task_sha,
             "exercise policy binding")
    if support:
        _require(policy.get("gates_sha256") == task["gates_sha256"], "support policy gates")
    if camera:
        _camera_policy(scene, task, contract, compact, files, camera_admission, policy_document)
    counts = compact.get("evaluation_successes")
    if compact.get("evaluation_status") == "not_attached":
        _require(counts is None, "unattached evaluation has success counts")
        return
    _require(compact.get("evaluation_status") == "evaluated", "evaluation status")
    # The compact receipt is the evidence read here, not its large trial arrays.
    for field, name in (("policy_evaluation_sha256", "policy-evaluation.json"),
                        ("policy_evaluation_manifest_sha256", "policy-evidence-run/manifest.json" if camera else "policy-evaluation-run/manifest.json")):
        digest = _hash(contract.get(field), field)
        _require(files.get(name) == digest, "evaluation attachment identity")
    _require(isinstance(counts, dict) and 1 <= len(counts) <= 4 and set(counts) <= _CONTROLLERS,
             "controller counts")
    episodes, controllers = None, {}
    for name, row in counts.items():
        _require(isinstance(row, dict), "controller summary")
        trials, successes = row.get("episodes"), row.get("successes")
        _require(type(trials) is int and 1 <= trials <= 2000
                 and type(successes) is int and 0 <= successes <= trials, "success count or denominator")
        _require(episodes in (None, trials), "controller trial counts differ")
        episodes = trials
        controllers[name] = {"successes": successes}
    summary["learning"] = {"scene_sha256": scene_sha, "task_sha256": task_sha,
                           "evaluation": {"trials_per_controller": episodes, "controllers": controllers}}
    summary["learning_artifact_names"] = ["exercise-lab.json", "policy.json", "scene.json", "exercise-task.json"]
    if camera:
        summary["learning_artifact_names"].append("camera-policy-admission.json")


def build_lab_evidence(*, source, scene, task, contract, policy, files, verified_metadata,
                       rigid_task=None, rigid_evaluation=None, camera_admission=None, policy_document=None):
    """Keep partial saved summaries partial; linked files are not verified reads."""
    _require(isinstance(verified_metadata, list) and len(verified_metadata) <= 128, "verified metadata bound")
    pointers = {row["source_name"]: row for row in verified_metadata}
    _require(len(pointers) == len(verified_metadata) and "manifest.json" in pointers, "manifest read identity")
    summary = {"source_manifest_sha256": pointers["manifest.json"]["sha256"],
               "artifacts": verified_metadata, "source": {}, "requested_options": {},
               "evidence_partition": "construction", "build_status": "saved_lab_metadata"}
    if source.get("sha256") is not None:
        summary["source"]["sha256"] = _hash(source["sha256"], "source")
    if "scene.json" in pointers:
        summary["scene_sha256"] = canonical_hash(scene)
        human = scene.get("human_spec") or {}
        _require(isinstance(human, dict), "human metadata")
        if human.get("source_sha256") is not None:
            human_source = _hash(human["source_sha256"], "human source")
            _require(source.get("sha256") in (None, human_source), "human/source binding")
        frame = scene.get("coordinate_frame") or {}
        if isinstance(frame, dict) and isinstance(frame.get("scale_bridge"), dict):
            scale = frame["scale_bridge"].get("body_height_to_world_scale")
            if scale is not None:
                _require(type(scale) in (int, float) and 0 < scale <= 1e6 and math.isfinite(scale), "body scale prior")
                summary["requested_options"]["body_scale"] = scale
    if "exercise-lab.json" in pointers:
        _require("scene.json" in pointers and "exercise-task.json" in pointers, "exercise scene/task missing")
        if contract.get("learned_policy_available") is True:
            _require("policy.json" in pointers, "exercise policy was not verified")
        if scene.get("profile") == "shared-world-v6" and contract.get("learned_policy_available") is True:
            _require("camera-policy-admission.json" in pointers, "camera admission was not verified")
        _shared_exercise(summary, scene, task, contract, policy, files,
                         camera_admission=camera_admission, policy_document=policy_document)
    if rigid_evaluation is not None:
        _require("exercise-lab.json" not in pointers, "conflicting exercise and rigid contracts")
        summary["rigid_learning"] = normalize_rigid_lab_evidence(scene=scene, task=rigid_task,
            policy=policy, evaluation=rigid_evaluation, verified_metadata=verified_metadata)
    return build_evidence_scorecard(summary)
