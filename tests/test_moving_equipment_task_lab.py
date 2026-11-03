"""Real child/source packaging and task controls; original detector/body admission is fixture-stubbed."""

from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
import shutil
import time

import pytest

from weft.io import canonical_hash, write_json
from weft import moving_equipment_task_lab as module


@pytest.fixture(scope="module")
def child(tmp_path_factory):
    patch = pytest.MonkeyPatch()
    root = tmp_path_factory.mktemp("authored-task-child")
    try:
        import test_root_motion
        from test_moving_equipment_proposal import packet
        from test_moving_depth_family import settings, choice
        from weft.moving_depth_family import prepare_moving_depth_family, materialize_moving_depth_selection
        from weft.shared_world_exploration import prepare_shared_world_exploration
        video = b"Analytical bytes; full body admission is stubbed in this fixture."
        patch.setattr(test_root_motion, "HASH", sha256(video).hexdigest())
        inputs, _ = packet(patch, raw_tracks=1, image_shift=(300, 190))
        family = prepare_moving_depth_family(inputs, settings())
        bundle = materialize_moving_depth_selection(family, choice(family, ["raw-0"]))
        original = prepare_shared_world_exploration(inputs, bundle, "candidate-1", root / "exploration", depth_family=family)
        parent = root / "exploration"
        before = {p.relative_to(parent).as_posix(): sha256(p.read_bytes()).hexdigest() for p in parent.rglob("*") if p.is_file()}
        scene = json.loads((parent / "run/scene.json").read_bytes())
        request = {"target_object_id": "raw-0", "target_position_world": scene["objects"][0]["position"],
            "position_tolerance_world": .05, "speed_tolerance_world_s": .1, "hold_duration_s": .04}
        result = module.prepare_moving_equipment_task_lab(parent, request, root / "task",
            parent_manifest_sha256=original["manifest_sha256"])
        yield root / "task", parent, before, result, request
    finally:
        patch.undo()


def config(root):
    return json.loads((root / "lab-config.json").read_bytes())


def test_sealed_child_preserves_complete_parent_and_detached_admission(child):
    root, parent, before, result, request = child
    admission = module.load_moving_equipment_task_lab(root / "run", config(root))
    assert admission.manifest_sha256 == result["manifest_sha256"]
    assert admission.parent_manifest_sha256 == result["parent_manifest_sha256"]
    assert admission.task["binding"]["scene_sha256"] == canonical_hash(admission.scene)
    assert admission.task["goal"]["position_world"] == request["target_position_world"]
    assert admission.task["binding"]["family_sha256"] == canonical_hash(admission.bundle["depth_family"])
    for name, pin in before.items():
        assert sha256((parent / name).read_bytes()).hexdigest() == pin
        if name == "lab-config.json" or name.startswith("run/"):
            assert sha256((root / "run/parent" / name).read_bytes()).hexdigest() == pin
    assert result["bytes_written"] <= module.MAX_LAB_BYTES
    assert module.validate_task_lab_admission(admission) is admission
    altered = admission.task; altered["goal"]["position_world"][0] += 1
    assert altered != admission.task
    with pytest.raises(ValueError, match="immutable source-admitted"):
        module.validate_task_lab_admission({"scene": admission.scene})
    with pytest.raises((AttributeError, TypeError)):
        admission.task_bytes = b"changed"


def test_actual_task_lab_manual_record_export_and_same_process_trusted_read(child):
    root = child[0]
    lab = module.TaskLab(root)
    try:
        summary = lab.summary()
        assert summary["control_profile"] == module.CONTROL_PROFILE
        assert summary["placement_task_sha256"] == canonical_hash(summary["placement_task"])
        assert summary["snapshot"]["task"]["task_sha256"] == summary["placement_task_sha256"]
        assert summary["task_recording_available"] is True
        assert not summary["learned_policy_available"]
        assert summary["capabilities"]["task_contract"]["task_type"] == "moving-equipment-placement-task-v1"
        assert summary["capabilities"]["reset_contract"]["supports_jitter"] is False
        for path, data in (("/api/actor", {"id": "raw-0"}), ("/api/appearance", {}),
                ("/api/step", {"policy": True, "steps": 1}), ("/api/step", {"controller": "reference", "steps": 1}),
                ("/api/reset", {"seed": 0, "jitter": True})):
            with pytest.raises(ValueError):
                lab.control(path, data)
        lab.control("/api/reset", {"seed": 1729})
        recording = lab.control("/api/record/start", {})
        assert recording["preparing"] and not recording["recording"]
        lab._record_start_future.result(timeout=30)
        assert lab.summary()["recording"]
        action = [0.] * len(summary["scene"]["action_contract"]["names"])
        for i in range(6):
            action[-3] = i * .01
            result = lab.control("/api/step", {"action": list(action), "steps": 1})
            assert len(result["transitions"]) == 1
            assert result["transitions"][0]["controller"] == "manual"
            assert result["snapshot"]["success"] == result["snapshot"]["task"]["success"]
        lab.control("/api/record/stop", {})
        exported = lab.export_future.result(timeout=30)
        assert exported["exact_replay_verified"]
        assert exported["proof"]["exact_state_replay"] and exported["proof"]["exact_task_reward_replay"]
        assert exported["proof"]["steps"] == 6
        assert exported["recording_evidence"]["status"] == "available"
        name = recording["directory"]
        lab.recording_evidence(name)
        saved = lab.recording_evidence_future.result(timeout=30)
        assert saved["status"] == "available" and saved["exact_replay_verified"]
        # Reopening has no independently retained export receipt pin; it must not upgrade a saved claim.
        lab._trusted_receipts.clear(); lab.recording_evidence_future = None
        lab.recording_evidence(name)
        reopened_scope = lab.recording_evidence_future.result(timeout=30)
        assert reopened_scope["status"] == "available" and reopened_scope["exact_replay_verified"] is False
        assert "proof" not in reopened_scope and "claimed_proof" in reopened_scope
    finally:
        lab.close()


@pytest.mark.parametrize("damage", ["task", "request", "parent_scene", "parent_config", "runtime", "alias", "profile"])
def test_child_tamper_rejects_even_if_outer_file_manifest_resealed(child, tmp_path, damage):
    root = tmp_path / "child"; shutil.copytree(child[0], root)
    cfg = config(root)
    paths = {"task": "placement-task.json", "request": "task-request.json", "parent_scene": "parent/run/scene.json",
        "parent_config": "parent/lab-config.json", "runtime": "implementation/moving_equipment_task.py", "alias": "human-reference.json"}
    if damage == "profile":
        cfg["control_profile"] = "shared_world_exploration"
    else:
        relative = paths[damage]
        path = root / "run" / relative
        if damage == "runtime":
            path.write_bytes(path.read_bytes() + b"\n# changed\n")
        else:
            value = json.loads(path.read_bytes())
            value["unexpected_change"] = True
            write_json(path, value)
        manifest = json.loads((root / "run/manifest.json").read_bytes())
        manifest["files"][relative] = sha256(path.read_bytes()).hexdigest()
        write_json(root / "run/manifest.json", manifest)
        cfg["manifest_sha256"] = sha256((root / "run/manifest.json").read_bytes()).hexdigest()
    with pytest.raises((ValueError, KeyError)):
        module.load_moving_equipment_task_lab(root / "run", cfg)


def test_original_parent_pin_and_output_custody_are_mandatory(child, tmp_path):
    _, parent, _, result, request = child
    with pytest.raises(ValueError, match="caller's exact pin"):
        module.prepare_moving_equipment_task_lab(parent, request, tmp_path / "wrong", parent_manifest_sha256="f" * 64)
    with pytest.raises(ValueError, match="outside"):
        module.prepare_moving_equipment_task_lab(parent, request, parent / "new-task", parent_manifest_sha256=result["parent_manifest_sha256"])


def test_snapshot_does_not_promote_old_engine_success():
    engine = {"success": True, "position": [1, 2, 3], "terminated": False}
    metrics = {"success": False, "completed_interval": False}
    snapshot = module.compose_task_snapshot(engine, metrics)
    assert snapshot["success"] is False and snapshot["engine_diagnostics"]["success"] is True
    snapshot["position"][0] = 99
    assert engine["position"] == [1, 2, 3]


def test_task_lab_propagates_remaining_load_deadline_once(child, monkeypatch):
    calls = []
    def refuse(run, cfg, **kwargs):
        calls.append(kwargs)
        raise ValueError("bounded admission sentinel")
    monkeypatch.setattr(module, "create_session", refuse)
    start = time.monotonic()
    lab = module.TaskLab(child[0], max_seconds=.5)
    try:
        with pytest.raises(ValueError, match="bounded admission sentinel"):
            lab.load_session()
        assert len(calls) == 1 and start <= calls[0]["deadline"] <= start + .6
    finally:
        lab.close()


def test_record_preparation_fences_controls_and_refuses_nonreset_before_files(child, monkeypatch):
    import threading
    import weft.moving_equipment_task_recording as recording
    ready, release = threading.Event(), threading.Event()
    original = recording.recording_contract
    def delayed(*args, **kwargs):
        ready.set()
        assert release.wait(5)
        return original(*args, **kwargs)
    monkeypatch.setattr(recording, "recording_contract", delayed)
    lab = module.TaskLab(child[0])
    try:
        summary = lab.summary()
        before = time.monotonic()
        started = lab.control("/api/record/start", {})
        assert time.monotonic() - before < .5 and started["preparing"]
        assert ready.wait(2)
        assert lab.summary()["recording_preparation"]["status"] == "running"
        action = [0.] * len(summary["scene"]["action_contract"]["names"])
        for path, payload in (("/api/reset", {"seed": 1}), ("/api/step", {"action": action, "steps": 1}), ("/api/record/stop", {})):
            with pytest.raises(ValueError, match="preparing"):
                lab.control(path, payload)
        release.set()
        lab._record_start_future.result(timeout=10)
        assert lab.summary()["recording"]
        lab.recorder.close(); lab.recorder = None
        lab.session._state["engine_diagnostics"]["steps"] = 1
        existing = set((lab.root / "sessions").iterdir())
        with pytest.raises(ValueError, match="Reset the task before recording"):
            lab.control("/api/record/start", {})
        assert set((lab.root / "sessions").iterdir()) == existing
    finally:
        release.set()
        lab.close()


def test_loading_facade_cannot_evict_pending_or_unadopted_record_start():
    from concurrent.futures import Future
    from types import SimpleNamespace
    facade = object.__new__(module.LoadingTaskLab)
    facade._future = Future()
    assert facade.loading
    record = Future()
    lab = SimpleNamespace(_record_start_future=record)
    facade._future.set_result(lab)
    assert facade.loading
    record.set_result("prepared recorder")
    assert facade.loading
    lab._record_start_future = None
    assert not facade.loading
