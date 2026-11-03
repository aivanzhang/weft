"""Real construction/control/import with analytic image data and an explicit body-admission stub."""

from hashlib import sha256
import json
from pathlib import Path
import threading
import time

import pytest

from weft.shared_world_exploration import prepare_shared_world_exploration, LoadingExplorationLab
from weft.moving_equipment_proposal import derive_moving_proposals


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    # Other fixture layers remain real; this is not an actual-video/body admission claim.
    import test_root_motion
    from test_moving_equipment_proposal import packet, settings
    video = b'Analytical bytes; full body admission is stubbed in this fixture.'
    monkeypatch.setattr(test_root_motion, "HASH", sha256(video).hexdigest())
    inputs, calls = packet(monkeypatch)
    bundle = derive_moving_proposals(inputs, settings(free=True))
    result = prepare_shared_world_exploration(inputs, bundle, "candidate-1", tmp_path / "lab")
    return Path(result["lab_directory"]), result, calls


def test_ordinary_lab_import_manual_record_export_and_receipt(prepared, tmp_path):
    from weft.lab import Lab
    from weft.portal_import import import_lab
    from weft.recording import verify_recording

    root, result, calls = prepared
    original = {str(p.relative_to(root / "run")): sha256(p.read_bytes()).hexdigest() for p in (root / "run").rglob('*') if p.is_file()}
    imported = import_lab(root, tmp_path / "portal")
    lab = Lab(Path(imported["directory"]) / "lab")
    try:
        summary = lab.summary()
        assert summary["control_profile"] == "shared_world_exploration"
        assert summary["exploration"]["task_success_available"] is False
        assert not summary["capabilities"]["controllers"]["authored_reference_controller"]
        assert summary["learned_policy_available"] is False
        for payload in ({"controller": "reference", "steps": 1}, {"policy": True, "steps": 1}):
            with pytest.raises(ValueError, match="explicit action"):
                lab.control("/api/step", payload)
        lab.control("/api/reset", {"seed": 1729})
        recording = lab.control("/api/record/start", {})
        action = [0.] * len(summary["scene"]["action_contract"]["names"])
        for index in range(6):
            action[-3] = .01 * index
            assert len(lab.control("/api/step", {"action": list(action), "steps": 1})["transitions"]) == 1
        lab.control("/api/record/stop", {})
        exported = lab.export_future.result(timeout=30)
        assert exported["action_replay"]["exact_state_replay"]
        assert exported["action_replay"]["steps"] == 6
        assert exported["recording_evidence"]["status"] == "available", json.dumps(exported["recording_evidence"])
        session = lab.root / "sessions" / recording["directory"]
        assert verify_recording(session)["action_replay"] == exported["action_replay"]
        from weft.recording_receipt import load_recording_evidence
        assert load_recording_evidence(lab.root, session.name, expected_run_manifest_sha256=lab.config["manifest_sha256"])["status"] == "available"
        import subprocess, sys
        script = """import importlib.abc,sys,json
class NoNative(importlib.abc.MetaPathFinder):
 def find_spec(self,name,*args):
  if name.split('.')[0] in {'mujoco','numpy','gymnasium','cv2'} or name in {'weft.shared_world','weft.shared_world_lab'}:
   raise AssertionError('Receipt inspection imported '+name)
sys.meta_path.insert(0,NoNative())
from weft.recording_receipt import load_recording_evidence
value=load_recording_evidence(sys.argv[1],sys.argv[2],expected_run_manifest_sha256=sys.argv[3])
assert value['status']=='available',value
print('pure saved receipt available')
"""
        process = subprocess.run([sys.executable, "-c", script, str(lab.root), session.name, lab.config["manifest_sha256"]],
                                 capture_output=True, text=True, timeout=10)
        assert process.returncode == 0, process.stdout + process.stderr
        evidence = session / "evidence"
        receipt = json.loads((evidence / "receipt.json").read_bytes())
        receipt["runtime"]["implementation_sha256"]["shared_world_exploration.py"] = "f" * 64
        changed = json.dumps(receipt).encode(); (evidence / "receipt.json").write_bytes(changed)
        seal = json.loads((evidence / "manifest.json").read_bytes())
        seal["files"]["receipt.json"] = sha256(changed).hexdigest()
        (evidence / "manifest.json").write_text(json.dumps(seal))
        refused = load_recording_evidence(lab.root, session.name, expected_run_manifest_sha256=lab.config["manifest_sha256"])
        assert refused["status"] == "unavailable" and "implementation identities" in refused["reason"]
    finally:
        lab.close()
    assert original == {str(p.relative_to(root / "run")): sha256(p.read_bytes()).hexdigest() for p in (root / "run").rglob('*') if p.is_file()}
    assert calls


def test_loader_is_single_future_and_close_disposes_once(tmp_path, monkeypatch):
    started, release, closed = threading.Event(), threading.Event(), []
    class FakeLab:
        def __init__(self, root): pass
        def load_session(self): started.set(); release.wait(2)
        def close(self): closed.append(1)
        def summary(self): return {"status": "ready"}
    monkeypatch.setattr("weft.lab.Lab", FakeLab)
    loader = LoadingExplorationLab(tmp_path, project=tmp_path)
    assert started.wait(1)
    for _ in range(8): assert loader.summary()["status"] == "running"
    with pytest.raises(ValueError, match="loading"): loader.control("/api/reset", {})
    loader.close(); release.set()
    loader._future.result(timeout=2)
    until = time.monotonic() + 2
    while not closed and time.monotonic() < until: time.sleep(.01)
    loader.close()
    assert closed == [1]


def test_loader_failure_is_terminal_not_repeated(tmp_path, monkeypatch):
    calls = []
    class FakeLab:
        def __init__(self, root): pass
        def load_session(self): calls.append(1); raise ValueError("saved source refused")
        def close(self): pass
    monkeypatch.setattr("weft.lab.Lab", FakeLab)
    loader = LoadingExplorationLab(tmp_path, project=tmp_path)
    try:
        with pytest.raises(ValueError): loader._future.result(timeout=2)
        for _ in range(4): assert loader.summary()["status"] == "failed"
        assert calls == [1]
    finally: loader.close()


@pytest.mark.parametrize("damage", ["source", "ledger", "scene", "bundle", "policy", "runtime", "downgrade"])
def test_tampered_or_downgraded_lab_cannot_open(prepared, damage):
    from weft.lab import Lab
    from weft.io import canonical_hash
    root, _, _ = prepared
    run = root / "run"
    manifest = json.loads((run / "manifest.json").read_bytes())
    config = json.loads((root / "lab-config.json").read_bytes())
    if damage == "downgrade":
        config.pop("control_profile")
    else:
        name = {"source": "source-inputs/base/source.mp4", "ledger": "motion-evidence.json", "scene": "scene.json",
                "bundle": "moving-proposals.json", "policy": "policy.json", "runtime": "implementation/shared_world_exploration.py"}[damage]
        if damage in {"source", "runtime"}: data = (run / name).read_bytes() + b"changed"
        elif damage == "policy": data = b"{}"
        else:
            value = json.loads((run / name).read_bytes())
            if damage == "scene":
                value["objects"][0]["position"][0] += .01
                config["scene_sha256"] = canonical_hash(value)
            elif damage == "bundle": value["candidates"] = value["candidates"][:1]
            else: value["scope"] = "tampered"
            data = json.dumps(value).encode()
        (run / name).write_bytes(data)
        manifest["files"][name] = sha256(data).hexdigest()
        (run / "manifest.json").write_text(json.dumps(manifest))
        config["manifest_sha256"] = sha256((run / "manifest.json").read_bytes()).hexdigest()
    (root / "lab-config.json").write_text(json.dumps(config))
    lab = Lab(root)
    try:
        with pytest.raises(ValueError): lab.load_session()
    finally: lab.close()


def test_available_avatar_catalog_cannot_bypass_exploration_recorder(prepared, monkeypatch):
    from weft.lab import Lab
    from weft.shared_world_action_replay import METADATA_KEY
    root, _, _ = prepared
    lab = Lab(root)
    try:
        lab.load_session()
        lab.avatars.packages = {"fixture": object()}
        monkeypatch.setattr("weft.avatar_recording.AppearanceRecorder", lambda *a, **k: pytest.fail("appearance recorder must not run"))
        # Runtime was prepared during load; starting a journal performs no package/version lookups.
        with monkeypatch.context() as scoped:
            scoped.setattr("weft.shared_world_action_replay.runtime_identity", lambda *_: pytest.fail("unexpected runtime lookup"))
            recorded = lab.control("/api/record/start", {})
        metadata = json.loads((root / "sessions" / recorded["directory"] / "recording.json").read_bytes())
        assert METADATA_KEY in metadata and "action_replay_contract" not in metadata
        lab.recorder.close(); lab.recorder = None
        with pytest.raises(ValueError, match="avatar"):
            lab.control("/api/appearance", {})
    finally: lab.close()


def test_same_size_late_source_mutation_with_restored_mtime_is_rejected(prepared, monkeypatch):
    import os
    import weft.moving_equipment_world as world
    from weft.lab import Lab
    root, _, _ = prepared
    original = world.construct_moving_equipment_world
    def changed(*args, **kwargs):
        result = original(*args, **kwargs)
        path = root / "run/source-inputs/base/source.mp4"
        old = path.stat(); data = path.read_bytes()
        path.write_bytes(data[:-1] + (b"!" if data[-1:] != b"!" else b"."))
        os.utime(path, ns=(old.st_atime_ns, old.st_mtime_ns))
        return result
    monkeypatch.setattr(world, "construct_moving_equipment_world", changed)
    lab = Lab(root)
    try:
        with pytest.raises(ValueError, match="changed during admission"):
            lab.load_session()
    finally: lab.close()


def test_loading_workers_are_globally_bounded_across_closed_owners(tmp_path, monkeypatch):
    entered, release = [], threading.Event()
    class FakeLab:
        def __init__(self, root): self.root = root
        def load_session(self): entered.append(self.root); release.wait(2)
        def close(self): pass
    monkeypatch.setattr("weft.lab.Lab", FakeLab)
    first = LoadingExplorationLab(tmp_path / "one", project=tmp_path)
    until = time.monotonic() + 1
    while not entered and time.monotonic() < until: time.sleep(.01)
    first.close()
    second = LoadingExplorationLab(tmp_path / "two", project=tmp_path)
    assert second.loading and len(entered) == 1
    second.close(); release.set(); first._future.result(timeout=2)
    assert second._future.cancelled()


def test_portal_never_evicts_active_admission_to_start_another(tmp_path):
    from collections import OrderedDict
    from weft.portal import Portal
    class Active:
        loading = True
        recorder = None
        export_future = None
        def close(self): pytest.fail("active load must not be evicted")
    portal = Portal.__new__(Portal)
    portal.lock = threading.RLock()
    portal.labs = OrderedDict((str(i), Active()) for i in range(3))
    portal._status = lambda _: {"status": "ready"}
    with pytest.raises(ValueError, match="another sandbox"):
        portal.lab("fourth")
    assert len(portal.labs) == 3


def test_complete_input_bound_precedes_constructor(tmp_path, monkeypatch):
    from test_moving_equipment_proposal import packet, settings
    import weft.shared_world_exploration as module
    inputs, _ = packet(monkeypatch)
    bundle = derive_moving_proposals(inputs, settings(free=True))
    monkeypatch.setattr(module, "MAX_LAB_BYTES", 100)
    monkeypatch.setattr("weft.moving_equipment_world.construct_moving_equipment_world", lambda *a, **k: pytest.fail("bound must precede constructor"))
    with pytest.raises(ValueError, match="before construction"):
        module.prepare_shared_world_exploration(inputs, bundle, "candidate-1", tmp_path / "no-output")
    assert not (tmp_path / "no-output").exists()


def test_console_rejects_unsupported_modes_without_changing_manual_controls():
    from weft.lab_console import Controls
    capability = {"exploration": {"evaluation_status": "not_configured"},
        "action_contract": {"space_type": "continuous_box", "dimension": 1, "actuator_names": ["motor"]},
        "controllers": {"learned_policy_available": False, "authored_reference_controller": False}}
    controls = Controls(capability)
    for mode in ("learned", "reference", "coast", "fitted"):
        with pytest.raises(ValueError, match="only manual"):
            controls.select_mode(mode)
    controls.select_motor("motor", .25)
    assert controls.step(1) == {"steps": 1, "action": [.25]}
