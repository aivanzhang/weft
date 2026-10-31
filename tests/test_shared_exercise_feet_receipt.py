"""Authored receipt fixtures only; no model, replay, media or runtime qualification."""

import builtins
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
import socket
import subprocess

import pytest

from test_recording_receipt import NAME, canonical, digest, exporter, make_lab, rewrite_receipt, write
from weft import recording_receipt as receipts


@pytest.fixture(autouse=True)
def saved_metadata_only(tmp_path, monkeypatch):
    imported = builtins.__import__

    def guard(name, *args, **kwargs):
        if name.startswith(("numpy", "mujoco", "cv2", "gymnasium", "weft.shared_exercise", "weft.shared_world",
                            "weft.shared_support", "weft.foot", "weft.recording")):
            raise AssertionError("Saved feet receipts must not import native, task, replay or media code")
        return imported(name, *args, **kwargs)

    def forbidden(*_args, **_kwargs):
        pytest.fail("Receipt tests cannot use network or spawn native/replay workers")

    monkeypatch.setattr(builtins, "__import__", guard)
    monkeypatch.setattr(socket, "socket", forbidden)
    monkeypatch.setattr(subprocess, "Popen", forbidden)
    package = tmp_path / "authored-runtime"
    package.mkdir()
    names = receipts.FEET_RUNTIME_REQUIRED | {"person_inventory.py", "shared_exercise_lab.py", "recording_receipt.py"}
    for name in names:
        (package / name).write_bytes(("# Authored receipt fixture, never executed: " + name + "\n").encode())
    monkeypatch.setattr(receipts, "_PACKAGE", package)
    return {name: digest(package / name) for name in names}


def save_source(run, session, scene, task, contract, *, envelope_profile=receipts.FEET_PROFILE, extra=None):
    task["scene_sha256"] = canonical(scene)
    contract.update(scene_sha256=canonical(scene), task_sha256=canonical(task))
    for name, document in (("scene.json", scene), ("exercise-task.json", task), ("exercise-lab.json", contract)):
        write(run / name, document)
    write(run / "manifest.json", {"schema_version": 1, "status": "ready", "files": {
        name: digest(run / name) for name in ("scene.json", "exercise-task.json", "exercise-lab.json")}})
    envelope = {"schema_version": 1, "profile": envelope_profile, "engine_scene": scene,
                "exercise_task": task, "dt": scene["dt"], "objects": scene["objects"], **(extra or {})}
    write(session / "scene.json", envelope)
    closed = json.loads((session / "recording.json").read_text())
    closed.update(scene_sha256=canonical(envelope))
    write(session / "recording.json", closed)


def feet_lab(tmp_path, runtime, *, profile="shared-world-v3"):
    lab, run, session = make_lab(tmp_path, shared=True)
    body = {"schema_version": 1, "profile": "exercise-v3", "source_sha256": "a" * 64,
            "fixture": "Metadata-only receipt fixture; not a native-valid human"}
    interval = {"start_offset_s": 0., "horizon_steps": 2}
    base = {"schema_version": 1, "profile": profile, "dt": 1 / 30, "objects": [],
            "human_spec": body, "task": interval}
    reference = {"schema_version": 1, "profile": "authored-receipt-reference-fixture",
                 "body_model_sha256": canonical(body), "source_sha256": body["source_sha256"]}
    text = json.dumps(reference, indent=2) + "\n"
    extension = {"profile": "shared-world-feet-v1", "reference_json": text,
                 "reference_sha256": sha256(text.encode()).hexdigest(), "reference_canonical_sha256": canonical(reference),
                 "body_model_sha256": canonical(body), "source_sha256": body["source_sha256"]}
    scene = {**base, "profile": "shared-world-v5", "base_profile": profile, "foot_extension": extension}
    original = {"schema_version": 1, "profile": {"shared-world-v2": "shared-exercise-task-v1",
                "shared-world-v3": "shared-exercise-task-v2", "shared-world-v4": "shared-support-task-v1"}[profile],
                "scene_sha256": canonical(base), "gate_profile": "authored-receipt-fixture", "gates_sha256": "b" * 64,
                "interval": interval, "control": {"root_assistance": "disabled", "object_forces": "forbidden"},
                "source_scope": {"physical_validation": "unverified", "fixture": "Not requalified by receipt inspection"}}
    if profile == "shared-world-v4":
        original.update(support_requirements=[], reward_profile="authored-receipt-fixture", authored_addition=True)
    else:
        original.update(equipment_ids=[], contact_requirements=[])
    names = receipts.FEET_RUNTIME_REQUIRED | ({"person_inventory.py"} if profile == "shared-world-v3" else set())
    native = {name: runtime[name] for name in names}
    task = {"schema_version": 1, "profile": "shared-exercise-feet-task-v1", "scene_sha256": canonical(scene),
            "base_scene_sha256": canonical(base), "base_task_sha256": canonical(original), "base_task": original,
            "projection_profile": "shared-exercise-feet-projection-v1", "projection_contract_sha256": "c" * 64,
            "adapter_implementation_sha256": canonical({name: native[name] for name in receipts.FEET_ADAPTER_FILES}),
            "foot_reference_sha256": extension["reference_sha256"], "interval": interval,
            "control": {**original["control"], "ankle_target_mode": "authored_initial_angle_hold"},
            "scope": {"physical_validation": "unverified"}}
    contract = json.loads((run / "exercise-lab.json").read_text())
    contract["runtime_sha256"] = native
    save_source(run, session, scene, task, contract)
    return lab, run, session, scene, task, contract


def export_feet(run, session, task, *, mutate=None):
    def result(value):
        value["action_replay"].update({key: task[key] for key in receipts.FEET_IDENTITIES})
        if mutate:
            mutate(value)

    perform, calls = exporter(shared=True, transform=result)
    exported = receipts.export_recording_with_receipt(session, export=perform, run_directory=run,
        expected_run_manifest_sha256=digest(run / "manifest.json"))
    assert calls == [session]
    return exported


@pytest.mark.parametrize("profile", ["shared-world-v2", "shared-world-v3", "shared-world-v4"])
def test_feet_export_and_saved_reader_bind_all_identities_without_importing_runtime(tmp_path, saved_metadata_only, profile, monkeypatch):
    lab, run, session, _, task, contract = feet_lab(tmp_path, saved_metadata_only, profile=profile)
    result = export_feet(run, session, task)
    assert result["recording_evidence"]["status"] == "available", result
    read = receipts._Budget.read
    seen = []

    def saved_read(self, root, name, *args, **kwargs):
        seen.append(name)
        assert name not in {"video.mp4", "preview.gif", "replay.html"} and not name.endswith(".py")
        if name == "events.jsonl":
            assert kwargs["keep"] is False
        return read(self, root, name, *args, **kwargs)

    monkeypatch.setattr(receipts._Budget, "read", saved_read)
    monkeypatch.setattr(receipts, "_runtime", lambda *args: pytest.fail("Saved inspection cannot requalify live runtime"))
    view = receipts.load_recording_evidence(lab, NAME)
    assert view["status"] == "available", view
    assert view["replay"]["protocol"] == receipts.FEET_PROTOCOL and view["replay"]["status"] == "passed"
    assert view["replay"]["learned_actions_verified"] is None
    assert all(view["identities"][key] == task[key] for key in receipts.FEET_IDENTITIES)
    assert view["recorded"] == {"events": 5, "actions": 3, "resets": 1}
    assert seen.count("events.jsonl") == 1 and view["inspection"]["journal_bytes_hashed"] > 0
    assert all(view["inspection"][key] is False for key in ("replay_performed", "journal_parsed", "media_read"))
    receipt = json.loads((session / "evidence" / "receipt.json").read_text())
    native = contract["runtime_sha256"]
    expected = canonical({name: native[name] for name in ("shared_exercise_task.py", "shared_support_task.py")}) if profile == "shared-world-v4" else native["shared_exercise_task.py"]
    assert receipt["runtime"]["evaluator_sha256"] == expected
    assert receipts.FEET_IMPLEMENTATIONS <= receipt["runtime"]["implementation_sha256"].keys()


@pytest.mark.parametrize("pin", receipts.FEET_IDENTITIES)
def test_incomplete_or_changed_feet_replay_pin_never_seals_success(tmp_path, saved_metadata_only, pin):
    _, run, session, _, task, _ = feet_lab(tmp_path, saved_metadata_only)
    result = export_feet(run, session, task, mutate=lambda value: value["action_replay"].pop(pin))
    assert result["recording_evidence"]["status"] == "unavailable"
    assert (session / "manifest.json").exists() and not (session / "evidence").exists()


@pytest.mark.parametrize("change", ["original_task", "base_scene", "reference_bytes", "reference_source", "adapter", "closure"])
def test_saved_foot_relationships_cannot_be_rebound_by_resealing_outer_documents(tmp_path, saved_metadata_only, change):
    _, run, session, scene, task, contract = feet_lab(tmp_path, saved_metadata_only)
    if change == "original_task":
        task["base_task"]["source_scope"]["fixture"] = "changed original task"
    elif change == "base_scene":
        task["base_task"]["scene_sha256"] = "0" * 64
        task["base_task_sha256"] = canonical(task["base_task"])
    elif change == "reference_bytes":
        scene["foot_extension"]["reference_json"] += " "
    elif change == "reference_source":
        extension = scene["foot_extension"]
        reference = json.loads(extension["reference_json"])
        reference["source_sha256"] = "0" * 64
        extension["reference_json"] = json.dumps(reference)
        extension["reference_canonical_sha256"] = canonical(reference)
        extension["reference_sha256"] = sha256(extension["reference_json"].encode()).hexdigest()
        task["foot_reference_sha256"] = extension["reference_sha256"]
    elif change == "adapter":
        contract["runtime_sha256"]["shared_exercise_feet_task.py"] = "0" * 64
    else:
        contract["runtime_sha256"].pop("foot_reference.py")
    save_source(run, session, scene, task, contract)
    result = export_feet(run, session, task)
    assert result["recording_evidence"]["status"] == "unavailable" and not (session / "evidence").exists()


@pytest.mark.parametrize("extra", ["avatar", "policy", "legacy_profile"])
def test_feet_cannot_use_avatar_policy_or_legacy_recording_semantics(tmp_path, saved_metadata_only, extra):
    _, run, session, scene, task, contract = feet_lab(tmp_path, saved_metadata_only)
    kwargs = {}
    if extra == "avatar":
        kwargs["extra"] = {"avatar": {"name": "unsupported"}}
    elif extra == "policy":
        contract["policy_sha256"] = "0" * 64
    else:
        kwargs["envelope_profile"] = receipts.SHARED_PROFILE
    save_source(run, session, scene, task, contract, **kwargs)
    result = export_feet(run, session, task)
    assert result["recording_evidence"]["status"] == "unavailable" and not (session / "evidence").exists()


def test_saved_receipt_rejects_old_protocol_and_conflicting_adapter_implementation(tmp_path, saved_metadata_only):
    lab, run, session, _, task, _ = feet_lab(tmp_path, saved_metadata_only)
    assert export_feet(run, session, task)["recording_evidence"]["status"] == "available"
    original = json.loads((session / "evidence" / "receipt.json").read_text())
    rewrite_receipt(session, lambda value: value["replay"].update(protocol=receipts.PROTOCOL))
    assert receipts.load_recording_evidence(lab, NAME)["status"] == "unavailable"

    def change(value):
        value.clear()
        value.update(deepcopy(original))
        value["runtime"]["implementation_sha256"]["shared_exercise_feet_task.py"] = "0" * 64

    rewrite_receipt(session, change)
    assert receipts.load_recording_evidence(lab, NAME)["status"] == "unavailable"


def test_feet_scene_extension_preserves_legacy_and_total_read_limits(tmp_path, saved_metadata_only, monkeypatch):
    lab, run, session, _, task, _ = feet_lab(tmp_path / "feet", saved_metadata_only)
    monkeypatch.setattr(receipts, "LEGACY_SCENE_BYTES", 100)
    assert export_feet(run, session, task)["recording_evidence"]["status"] == "available"
    assert receipts.load_recording_evidence(lab, NAME)["status"] == "available"
    _, old_run, old_session = make_lab(tmp_path / "legacy")
    perform, calls = exporter()
    result = receipts.export_recording_with_receipt(old_session, export=perform, run_directory=old_run)
    assert calls == [old_session] and result["recording_evidence"]["status"] == "unavailable"
    assert "4 MiB" in result["recording_evidence"]["reason"]
    monkeypatch.setattr(receipts, "MAX_READ_BYTES", 64)
    assert receipts.load_recording_evidence(lab, NAME)["status"] == "unavailable"


def test_no_replay_claim_is_fabricated_when_feet_export_only_verifies_media(tmp_path, saved_metadata_only):
    lab, run, session, _, task, _ = feet_lab(tmp_path, saved_metadata_only)
    result = export_feet(run, session, task, mutate=lambda value: value.pop("action_replay"))
    assert result["recording_evidence"]["status"] == "available"
    assert receipts.load_recording_evidence(lab, NAME)["replay"]["status"] == "not_performed"
