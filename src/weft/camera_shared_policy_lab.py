"""Attach one immutable research evidence bundle without duplicating its training archive."""

from copy import deepcopy

from weft.io import canonical_hash
from weft.shared_exercise_lab import _bytes, _json, _sha, _verified_inputs
from weft.shared_world_packaging import read_package_file, read_package_manifest


PREFIX = "policy-evidence-run/"
ADMISSION_FILE = "camera-policy-admission.json"


def _training(manifest_raw, files):
    from weft.camera_shared_learning import TRAINING_ALIASES

    status = _json(manifest_raw).get("status")
    if status == "trained":
        return manifest_raw, files
    if status != "evaluated" or "input-manifest.json" not in files:
        raise ValueError("Camera attachment requires complete training or separate evaluation evidence")
    parent = files["input-manifest.json"]
    try:
        return parent, {name: files[TRAINING_ALIASES.get(name, name)] for name in _json(parent)["files"]}
    except (KeyError, TypeError) as error:
        raise ValueError("Camera evaluation must retain every exact training member") from error


def _attachment(manifest_raw, files, scene, task, *, deadline):
    from weft.camera_shared_policy_admission import prepare_camera_policy_admission
    from weft.shared_exercise_reset import describe_reset_protocol

    training_manifest, training_files = _training(manifest_raw, files)
    evaluated = _json(manifest_raw)["status"] == "evaluated"
    admission = prepare_camera_policy_admission(scene, task,
        training_manifest_raw=training_manifest, training_files=training_files,
        evaluation_manifest_raw=manifest_raw if evaluated else None,
        evaluation_files=files if evaluated else None, deadline=deadline)
    policy_raw = training_files["policy.json"]
    policy, training = _json(policy_raw), _json(training_files["training.json"])
    evaluation = _json(files["evaluation.json"]) if evaluated else None
    identity = {"policy_sha256": _sha(policy_raw), "learning_manifest_sha256": _sha(training_manifest),
        "learning_files_verified": len(training_files), "policy_runtime_sha256": deepcopy(policy["runtime_sha256"]),
        "camera_policy_admission_sha256": canonical_hash(admission),
        "camera_policy_evidence_manifest_sha256": _sha(manifest_raw),
        "camera_policy_evidence_kind": "evaluation" if evaluated else "training",
        "camera_policy_evidence_files_verified": len(files),
        "policy_evaluation_scope": "Saved research comparison; no reconstructed contact or physical validation"}
    if evaluated:
        identity.update(policy_evaluation_manifest_sha256=_sha(manifest_raw),
            policy_evaluation_sha256=_sha(files["evaluation.json"]), policy_evaluation_files_verified=len(files))
    summary = {"profile": policy["profile"], "policy_sha256": _sha(policy_raw),
        "admission_profile": admission["profile"], "admission_sha256": canonical_hash(admission),
        "selected_feedback_kind": training["selected_feedback_kind"], "privileged_simulator_features": True,
        "training_score": policy["training_score"],
        "training_score_improved_over_authored": training["improvement_over_authored_training"] > 0,
        "evaluation_status": "evaluated" if evaluated else "not_attached",
        "evaluation_successes": ({name: {"successes": row["successes"], "episodes": len(row["trials"])}
                                  for name, row in evaluation["controllers"].items()} if evaluated else None),
        "evaluation_reset_protocol": describe_reset_protocol(evaluation["reset_options"]) if evaluated else None,
        "learning_qualification": "unqualified_research", "root_assistance": task["control"]["root_assistance"],
        "scope": "External searched joint feedback with authored root assistance when enabled; not learned balance, recovered contact or physical validation."}
    output = {PREFIX + "manifest.json": manifest_raw, **{PREFIX + name: raw for name, raw in files.items()},
              ADMISSION_FILE: _bytes(admission), "policy.json": policy_raw}
    if evaluated:
        output["policy-evaluation.json"] = files["evaluation.json"]
    return {"policy": policy, "policy_document": policy_raw, "admission": admission,
            "identity": identity, "summary": summary, "files": output}


def prepare_camera_attachment(learning_source, evaluation_source, scene, task, *, deadline):
    evidence_source = evaluation_source if evaluation_source is not None else learning_source
    manifest_raw, files = _verified_inputs(evidence_source, deadline=deadline)
    if evaluation_source is not None:
        if _json(manifest_raw).get("status") != "evaluated":
            raise ValueError("Camera evaluation input requires its separately evaluated package")
        selected, selected_files = read_package_manifest(learning_source)
        training_manifest, _ = _training(manifest_raw, files)
        if selected != training_manifest or "policy.json" not in selected_files:
            raise ValueError("Camera evaluation does not embed the selected training package")
        policy_raw = read_package_file(learning_source, "policy.json", max_bytes=64 << 10,
                                      expected_sha256=selected_files["policy.json"])
        if policy_raw != files.get("policy.json"):
            raise ValueError("Camera evaluation changed the selected policy bytes")
    elif _json(manifest_raw).get("status") != "trained":
        raise ValueError("The camera policy input must be its original trained package")
    return _attachment(manifest_raw, files, scene, task, deadline=deadline)


def load_camera_attachment(files, scene, task, contract, *, deadline):
    required = {"policy.json", ADMISSION_FILE, PREFIX + "manifest.json"}
    if not required <= files.keys() or any(name.startswith(("learning-run/", "policy-evaluation-run/")) for name in files):
        raise ValueError("Camera policy lab requires one complete distinct evidence bundle")
    evidence = {name[len(PREFIX):]: raw for name, raw in files.items()
                if name.startswith(PREFIX) and name != PREFIX + "manifest.json"}
    attachment = _attachment(files[PREFIX + "manifest.json"], evidence, scene, task, deadline=deadline)
    if (any(contract.get(key) != value for key, value in attachment["identity"].items())
            or contract.get("exercise_policy") != attachment["summary"]
            or any(files.get(name) != raw for name, raw in attachment["files"].items())):
        raise ValueError("Camera lab policy, admission or displayed outcomes differ from the retained evidence")
    if ("policy-evaluation.json" in files) != (attachment["identity"]["camera_policy_evidence_kind"] == "evaluation"):
        raise ValueError("Camera lab cannot expose an unbound evaluation")
    return attachment


def preflight_camera_attachment(contract, hashes):
    if (contract.get("learned_policy_available") is not True
            or not {ADMISSION_FILE, "policy.json", PREFIX + "manifest.json"} <= hashes.keys()
            or any(name.startswith(("learning-run/", "policy-evaluation-run/")) for name in hashes)
            or contract.get("camera_policy_evidence_kind") not in ("training", "evaluation")):
        raise ValueError("Camera policy metadata requires its distinct admission and evidence bundle")
