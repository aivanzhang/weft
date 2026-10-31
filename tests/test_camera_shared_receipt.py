"""Camera receipt custody is inspectable without importing physics or parsing the journal."""

import builtins
from hashlib import sha256
import json
import socket
import subprocess

import pytest

from test_recording_receipt import NAME, canonical, digest, exporter, make_lab, write
from test_shared_exercise_feet_receipt import save_source
from weft import camera_shared_receipt as camera
from weft import recording_receipt as receipts


@pytest.fixture
def metadata_only(tmp_path, monkeypatch):
    original_import = builtins.__import__

    def imported(name, *args, **kwargs):
        if name.startswith(("numpy", "mujoco", "cv2", "gymnasium", "weft.shared_world", "weft.shared_exercise",
                            "weft.camera_shared_task", "weft.camera_shared_state", "weft.camera_shared_recording")):
            raise AssertionError("Receipt inspection cannot import native task, replay or media code")
        return original_import(name, *args, **kwargs)

    def forbidden(*args, **kwargs):
        pytest.fail("Receipt inspection cannot call network or processes")

    monkeypatch.setattr(builtins, "__import__", imported)
    monkeypatch.setattr(socket, "socket", forbidden)
    monkeypatch.setattr(subprocess, "Popen", forbidden)
    package = tmp_path / "fixture-runtime"
    package.mkdir()
    for name in camera.RUNTIME_REQUIRED | {"shared_exercise_lab.py", "recording_receipt.py"}:
        (package / name).write_text("# Metadata fixture, never executed: " + name + "\n")
    monkeypatch.setattr(receipts, "_PACKAGE", package)
    return {p.name: digest(p) for p in package.iterdir()}


def camera_lab(tmp_path, runtime):
    lab, run, session = make_lab(tmp_path, shared=True)
    body = {"source_sha256": "a" * 64, "dt": 1 / 30, "scope": "Metadata fixture, not native-valid"}
    coupling = {"source_sha256": body["source_sha256"], "body_sha256": canonical(body), "source_clock_sha256": "b" * 64}
    interval = {"start_offset_s": 0., "horizon_steps": 3}
    scene = {"profile": "shared-world-v6", "human_spec": body, "camera_coupling": coupling,
             "dt": 1001 / 30000, "substeps": 9, "objects": [], "task": interval, "learned_policy_available": False}
    control = {"root_assistance": "disabled", "object_forces": "forbidden"}
    scope = {"physical_validation": "unverified"}
    args = {"profile": "shared-exercise-task-v2", "interval": interval, "control": control,
            "equipment_ids": [], "contact_requirements": [], "source_scope": scope}
    context = {"profile": "camera-shared-metric-context-v1", "valid_legacy_task": False,
        "scene_sha256": canonical(scene), "kernel_branch": "shared-exercise-task-v2",
        "projection": "identity_22_motor_samples_without_equipment", "control_dt_s": scene["dt"],
        "source_body_dt_s": body["dt"], "substeps": 9, "kernel_arguments": args, "kernel_arguments_sha256": canonical(args)}
    native = {name: runtime[name] for name in camera.RUNTIME_REQUIRED}
    task = {"schema_version": 1, "profile": "camera-shared-task-v1", "scene_sha256": canonical(scene),
        "camera_coupling_sha256": canonical(coupling), **coupling,
        "metric_context": context, "metric_context_sha256": canonical(context),
        "kernel_evaluator_sha256": native["shared_exercise_task.py"],
        "adapter_implementation_sha256": native["camera_shared_task.py"],
        "evaluator_sha256": canonical({name: native[name] for name in ("camera_shared_task.py", "shared_exercise_task.py")}),
        "gate_profile": "fixture", "gates_sha256": "c" * 64, "interval": interval, "control": control, "scope": scope}
    contract = json.loads((run / "exercise-lab.json").read_bytes())
    contract.update(runtime_sha256=native, learned_policy_available=False)
    save_source(run, session, scene, task, contract, envelope_profile=camera.PROFILE)
    return lab, run, session, scene, task, contract


def export_camera(run, session, task, mutate=None):
    def transform(result):
        result["action_replay"].update({key: task[key] for key in camera.IDENTITIES})
        if mutate:
            mutate(result)
    perform, calls = exporter(shared=True, transform=transform)
    result = receipts.export_recording_with_receipt(session, export=perform, run_directory=run,
        expected_run_manifest_sha256=digest(run / "manifest.json"))
    return result, calls


def test_camera_receipt_preserves_exact_pins_and_reads_no_media_or_native_runtime(tmp_path, metadata_only, monkeypatch):
    lab, run, session, scene, task, contract = camera_lab(tmp_path, metadata_only)
    result, calls = export_camera(run, session, task)
    assert calls == [session] and result["recording_evidence"]["status"] == "available", result
    read = receipts._Budget.read
    seen = []
    def observed(self, root, name, *args, **kwargs):
        seen.append(name)
        assert name not in {"video.mp4", "replay.html", "preview.gif"} and not name.endswith(".py")
        if name == "events.jsonl":
            assert kwargs["keep"] is False
        return read(self, root, name, *args, **kwargs)
    monkeypatch.setattr(receipts._Budget, "read", observed)
    view = receipts.load_recording_evidence(lab, NAME)
    assert view["status"] == "available", view
    assert view["replay"]["protocol"] == camera.PROTOCOL
    assert view["replay"]["learned_actions_verified"] is None
    assert all(view["identities"][key] == task[key] for key in camera.IDENTITIES)
    assert seen.count("events.jsonl") == 1
    receipt = json.loads((session / "evidence/receipt.json").read_bytes())
    assert receipt["runtime"]["evaluator_sha256"] == task["evaluator_sha256"]


@pytest.mark.parametrize("pin", camera.IDENTITIES)
def test_camera_missing_replay_identity_never_seals_success(tmp_path, metadata_only, pin):
    _, run, session, _, task, _ = camera_lab(tmp_path, metadata_only)
    result, _ = export_camera(run, session, task, lambda value: value["action_replay"].pop(pin))
    assert result["recording_evidence"]["status"] == "unavailable" and not (session / "evidence").exists()


@pytest.mark.parametrize("change", ["clock", "body", "selector", "native", "closure", "legacy", "policy"])
def test_camera_binding_cannot_be_changed_by_resealing_metadata(tmp_path, metadata_only, change):
    _, run, session, scene, task, contract = camera_lab(tmp_path, metadata_only)
    profile = camera.PROFILE
    if change == "clock":
        task["source_clock_sha256"] = "f" * 64
    elif change == "body":
        scene["human_spec"]["dt"] = .05
    elif change == "selector":
        task["metric_context"]["kernel_arguments"]["equipment_ids"] = ["invented"]
        task["metric_context"]["kernel_arguments_sha256"] = canonical(task["metric_context"]["kernel_arguments"])
        task["metric_context_sha256"] = canonical(task["metric_context"])
    elif change == "native":
        task["adapter_implementation_sha256"] = "f" * 64
    elif change == "closure":
        contract["runtime_sha256"].pop("camera_shared_state.py")
    elif change == "policy":
        contract["learned_policy_available"] = True
    else:
        profile = receipts.SHARED_PROFILE
    save_source(run, session, scene, task, contract, envelope_profile=profile)
    result, calls = export_camera(run, session, task)
    assert calls == [session] and result["recording_evidence"]["status"] == "unavailable"
    assert not (session / "evidence").exists()
