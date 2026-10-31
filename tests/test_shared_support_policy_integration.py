"""Support policies keep their own task identity through the ordinary lab/recorder."""

from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from weft.io import canonical_hash
from weft import shared_exercise_lab as lab
from weft import shared_exercise_recording as recording
from weft import shared_exercise_residual_learning as residual
from weft.shared_support_fixture import static_support_fixture


def raw(value):
    return (json.dumps(value, sort_keys=True) + "\n").encode()


@pytest.fixture
def policy_package():
    scene, task = static_support_fixture()
    hashes = residual._runtime_hashes(scene)
    contract = residual._contract(scene, hashes)
    env = SimpleNamespace(scene=scene, task=task, scene_sha256=canonical_hash(scene), task_sha256=canonical_hash(task),
                          evaluator_sha256=contract["evaluator_sha256"], action_names=scene["action_contract"]["names"][:19])
    policy = residual._policy(env, np.zeros((2, 19)), 1729, .8, hashes)
    policy_raw = raw(policy)
    training = {"status": "trained", "policy_sha256": sha256(policy_raw).hexdigest(),
                "selected_feedback_kind": "authored_unchanged", "selected_authored_baseline": True,
                "selected_coast_baseline": False, "improvement_over_authored_training": 0.,
                "training_score_improved_over_authored": False}
    source_manifest = b"{}"
    settings = {name: policy[name] for name in ("scene_sha256", "task_sha256", "gates_sha256", "runtime_sha256")}
    settings["source_manifest_sha256"] = sha256(source_manifest).hexdigest()
    files = {"policy.json": policy_raw, "training.json": raw(training), "settings.json": raw(settings),
             "scene.json": raw(scene), "exercise-task.json": raw(task), "input-manifest.json": source_manifest,
             **{"implementation/" + name: Path(residual.__file__).with_name(name).read_bytes() for name in hashes}}
    manifest = raw({"status": "trained", "files": {name: sha256(value).hexdigest() for name, value in files.items()}})
    return scene, task, policy, training, manifest, files


def test_support_package_uses_generic_lab_custody_with_distinct_policy(policy_package):
    scene, task, policy, training, manifest, files = policy_package
    accepted, identity = lab._validate_policy_package(manifest, files, scene, task)
    assert accepted == policy and accepted["profile"] == "shared-support-policy-v1"
    assert identity["policy_sha256"] == sha256(files["policy.json"]).hexdigest()
    status = lab._policy_status(accepted, files["training.json"])
    assert status["selected_feedback_kind"] == "authored_unchanged"
    assert status["training_score_improved_over_authored"] is False
    assert status["evaluation_status"] == "not_attached"
    assert task["control"]["root_assistance"] == "disabled"


def test_support_lab_rejects_resealed_legacy_policy_or_changed_evaluator(policy_package):
    scene, task, policy, _, _, files = policy_package
    for change in ("profile", "evaluator"):
        modified = deepcopy(policy)
        if change == "profile":
            modified["profile"] = "shared-exercise-policy-v2"
        else:
            modified["evaluator_sha256"] = modified["runtime_sha256"]["shared_exercise_task.py"]
        changed = {**files, "policy.json": raw(modified)}
        manifest = raw({"status": "trained", "files": {name: sha256(value).hexdigest() for name, value in changed.items()}})
        with pytest.raises(ValueError):
            lab._validate_policy_package(manifest, changed, scene, task)


def test_support_recording_pins_exact_policy_without_expanding_controls(policy_package):
    scene, task, _, _, _, files = policy_package
    envelope = recording.recording_scene(scene, task, policy_document=files["policy.json"])
    assert recording.validate_recording_scene(envelope) == envelope
    assert envelope["learned_policy"]["sha256"] == sha256(files["policy.json"]).hexdigest()
    recording.validate_action([0.] * 19, envelope)
    with pytest.raises(ValueError, match="19 finite"):
        recording.validate_action([0.] * 22, envelope)
    changed = deepcopy(envelope)
    changed["learned_policy"]["document_utf8"] += " "
    with pytest.raises(ValueError, match="hash differs"):
        recording.validate_recording_scene(changed)
    with pytest.raises(ValueError, match="separate qualification"):
        recording.recording_scene(scene, task, reference_controller_document=b"{}")


def test_support_runtime_evaluator_is_not_relabelled_as_legacy(policy_package):
    scene, task, policy, _, _, files = policy_package
    from weft.recording_receipt import _evaluator_identity
    hashes = {name: policy["runtime_sha256"][name] for name in ("shared_exercise_task.py", "shared_support_task.py")}
    assert _evaluator_identity({"shared": True, "engine_profile": scene["profile"], "native_sha256": hashes}) == policy["evaluator_sha256"]
    assert policy["evaluator_sha256"] != hashes["shared_exercise_task.py"]
    assert "shared_support_task.py" in lab.runtime_files(scene)
    assert task["gates_sha256"] == policy["gates_sha256"]
