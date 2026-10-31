"""Camera attachment storage and controller routing without native simulation."""

from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from weft import camera_shared_policy_lab as package
from weft import shared_exercise_lab as lab
from weft.io import canonical_hash


def raw(value):
    return json.dumps(value, sort_keys=True).encode()


def evaluated_fixture():
    training = {"scene.json": b"scene", "policy.json": b"policy", "settings.json": b"settings",
                "training.json": b"training", "input-manifest.json": b"source",
                "source-verification.json": b"proof"}
    manifest = raw({"status": "trained", "files": {name: lab._sha(value) for name, value in training.items()}})
    from weft.camera_shared_learning import TRAINING_ALIASES
    files = {TRAINING_ALIASES.get(name, name): value for name, value in training.items()}
    files["input-manifest.json"] = manifest
    evaluation = raw({"status": "evaluated", "files": {name: lab._sha(value) for name, value in files.items()}})
    return manifest, training, evaluation, files


def test_evaluation_reuses_exact_training_aliases():
    manifest, training, evaluated, files = evaluated_fixture()
    parent, restored = package._training(evaluated, files)
    assert parent == manifest and restored == training
    assert all(restored[name] is files["training-settings.json" if name == "settings.json" else
               "training-history.json" if name == "training.json" else
               "training-source-manifest.json" if name == "input-manifest.json" else
               "training-source-verification.json" if name == "source-verification.json" else name]
               for name in training)
    del files["training-history.json"]
    with pytest.raises(ValueError, match="every exact training member"):
        package._training(evaluated, files)


def test_attachment_reads_one_archive_and_compares_selected_parent(monkeypatch):
    manifest, training, evaluated, files = evaluated_fixture()
    reads = []
    def inputs(path, *, deadline):
        reads.append(path)
        return evaluated, files
    monkeypatch.setattr(package, "_verified_inputs", inputs)
    monkeypatch.setattr(package, "read_package_manifest", lambda path: (manifest, {"policy.json": lab._sha(b"policy")}))
    monkeypatch.setattr(package, "read_package_file", lambda *args, **kwargs: b"policy")
    monkeypatch.setattr(package, "_attachment", lambda *args, **kwargs: "attached")
    assert package.prepare_camera_attachment(Path("train"), Path("eval"), {}, {}, deadline=100) == "attached"
    assert reads == [Path("eval")]
    monkeypatch.setattr(package, "read_package_manifest", lambda path: (manifest + b" ", {"policy.json": lab._sha(b"policy")}))
    with pytest.raises(ValueError, match="selected training"):
        package.prepare_camera_attachment(Path("train"), Path("eval"), {}, {}, deadline=100)


def test_evaluation_argument_cannot_relabel_training(monkeypatch):
    manifest, training, _, _ = evaluated_fixture()
    monkeypatch.setattr(package, "_verified_inputs", lambda *args, **kwargs: (manifest, training))
    with pytest.raises(ValueError, match="separately evaluated"):
        package.prepare_camera_attachment(Path("train"), Path("train"), {}, {}, deadline=100)


def test_resume_compares_admission_policy_and_summary(monkeypatch):
    output = {"policy.json": b"p", package.ADMISSION_FILE: b"a", package.PREFIX + "manifest.json": b"m",
              package.PREFIX + "policy.json": b"p"}
    attachment = {"files": output, "identity": {"camera_policy_evidence_kind": "training"},
                  "summary": {"scope": "research"}}
    monkeypatch.setattr(package, "_attachment", lambda *args, **kwargs: attachment)
    contract = {**attachment["identity"], "exercise_policy": attachment["summary"]}
    assert package.load_camera_attachment(output, {}, {}, contract, deadline=100) is attachment
    for name in ("policy.json", package.ADMISSION_FILE):
        with pytest.raises(ValueError, match="differ"):
            package.load_camera_attachment({**output, name: b"changed"}, {}, {}, contract, deadline=100)
    with pytest.raises(ValueError, match="distinct evidence"):
        package.load_camera_attachment({**output, "learning-run/policy.json": b"p"}, {}, {}, contract, deadline=100)
    with pytest.raises(ValueError, match="unbound evaluation"):
        package.load_camera_attachment({**output, "policy-evaluation.json": b"invented"}, {}, {}, contract, deadline=100)


def test_boolean_alone_cannot_admit_camera_learning():
    with pytest.raises(ValueError, match="distinct admission"):
        package.preflight_camera_attachment({"learned_policy_available": True}, {})


def test_policy_decision_metadata_tracks_actual_chosen_action_and_reset():
    session = lab.SharedExerciseSession.__new__(lab.SharedExerciseSession)
    session.policy_sha256 = "a" * 64
    session._camera_policy_admission = {"profile": "test-admission"}
    session._policy_admission_sha256 = canonical_hash(session._camera_policy_admission)
    session._policy_sample = None
    diagnostic = {"phase": .2, "authored_action": [0.] * 19, "applied_joint_delta": [.01] * 19}
    session._policy_action = lambda: (np.full(19, .01), deepcopy(diagnostic))
    action = session.policy_action()
    metadata = session.policy_recording_metadata(action)
    assert metadata == {"policy_sha256": "a" * 64,
        "policy_admission_sha256": canonical_hash(session._camera_policy_admission), "policy_sample": diagnostic}
    metadata["policy_sample"]["phase"] = .8
    assert session.policy_recording_metadata(action)["policy_sample"]["phase"] == .2
    with pytest.raises(ValueError, match="actual prepared"):
        session.policy_recording_metadata([0.] * 19)
    class ResetEnv:
        def reset(self, **kwargs):
            return [], {}
        def snapshot(self):
            return {"reset": True}
    session.env = ResetEnv()
    session.reset(17)
    with pytest.raises(ValueError, match="actual prepared"):
        session.policy_recording_metadata(action)


def test_live_controls_forward_only_the_actual_learned_decision():
    from weft.lab import Lab

    session = lab.SharedExerciseSession.__new__(lab.SharedExerciseSession)
    session.scene_profile, session.policy, session.done = "shared-world-v6", {}, False
    session.policy_sha256, session._camera_policy_admission = "b" * 64, {"profile": "test"}
    session._policy_admission_sha256 = canonical_hash(session._camera_policy_admission)
    sample = {"phase": .2, "authored_action": [0.] * 19, "applied_joint_delta": [.02] * 19}
    session._policy_action = lambda: (np.full(19, .02), sample)
    session._policy_sample = None
    session.env = SimpleNamespace(action_count=19)
    session.step = lambda action: (.25, False, False, {})
    session.snapshot = lambda: {"fixture": "no native execution"}
    session.resolve_action = list
    app, journal = Lab.__new__(Lab), []
    app.load_session = lambda: session
    app.recorder = SimpleNamespace(check_capacity=lambda: None, append=journal.append)
    learned = app.control("/api/step", {"policy": True})["transitions"][0]
    assert learned["policy_sample"] == sample and learned["action"] == [.02] * 19
    assert learned["policy_admission_sha256"] == canonical_hash(session._camera_policy_admission)
    manual = app.control("/api/step", {"action": [0.] * 19})["transitions"][0]
    assert manual["controller"] == "manual" and "policy_sample" not in manual and "policy_sha256" not in manual
    assert journal == [learned, manual]


@pytest.mark.parametrize("learned", [None, True, -1, 4])
def test_policy_receipt_rejects_missing_or_impossible_decision_counts(learned):
    from weft import recording_receipt as receipts

    binding = {"shared": True, "camera": True, "camera_policy": True,
               "identities": {"policy_sha256": "a" * 64}}
    counts = {"events": 5, "actions": 3, "resets": 1}
    replay = {"status": "passed", "protocol": "camera-shared-policy-full-state-replay-v1",
              "events_checked": 5, "actions_checked": 3, "resets_checked": 1, "learned_actions_verified": learned}
    with pytest.raises(ValueError):
        receipts._replay(replay, counts, binding)
    replay["learned_actions_verified"] = 0
    assert receipts._replay(replay, counts, binding) == replay
    replay["protocol"] = "camera-shared-full-state-replay-v1"
    with pytest.raises(ValueError, match="supported protocol"):
        receipts._replay(replay, counts, binding)
