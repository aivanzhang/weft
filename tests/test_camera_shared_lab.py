"""Camera lab custody and dispatch without native physics or video processing."""

from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
import socket
import subprocess
from types import SimpleNamespace

import mujoco
import pytest

from weft import camera_shared_task as camera_task
from weft import shared_exercise_lab as lab
from weft.io import canonical_hash


def write(path, value):
    raw = (json.dumps(value, indent=2) + "\n").encode()
    path.write_bytes(raw)
    return raw


def seal(root):
    write(root / "manifest.json", {"schema_version": 1, "status": "constructed", "files": {
        p.name: sha256(p.read_bytes()).hexdigest() for p in root.iterdir() if p.name != "manifest.json"}})


@pytest.fixture(autouse=True)
def no_execution(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Metadata plumbing cannot execute native physics, media, network or subprocesses")

    for owner, names in ((mujoco, ("MjData", "mj_step", "mj_forward", "Renderer")),
                         (mujoco.MjModel, ("from_xml_string", "from_xml_path")),
                         (socket, ("socket", "create_connection")), (subprocess, ("Popen",))):
        for name in names:
            monkeypatch.setattr(owner, name, forbidden)


@pytest.fixture
def inputs(tmp_path, monkeypatch):
    source = tmp_path / "source"
    source.mkdir()
    scene = {"profile": "shared-world-v6", "objects": [], "source_binding": {"mode": "analytic_fixture"},
             "camera_coupling": {"fixture": "saved metadata only"}}
    task = {"profile": "camera-shared-task-v1", "scene_sha256": canonical_hash(scene),
            "control": {"root_assistance": "disabled", "object_forces": "forbidden"}}
    write(source / "scene.json", scene)
    write(source / "camera-coupling.json", scene["camera_coupling"])
    write(source / "ground-clearance.json", {"contact": "unknown"})
    seal(source)
    calls = {"task": [], "reference": []}

    def validate(value, actual):
        calls["task"].append((deepcopy(value), deepcopy(actual)))
        if value.get("profile") != task["profile"] or value.get("scene_sha256") != canonical_hash(actual):
            raise ValueError("Mock camera task binding mismatch")
        return deepcopy(value)

    def build(actual, *, root_assistance=False):
        result = deepcopy(task)
        result["control"]["root_assistance"] = "explicit_3_axis" if root_assistance else "disabled"
        return validate(result, actual)

    def references(actual, _deadline):
        calls["reference"].append(deepcopy(actual))
        return {"frames": [], "scope": "No native execution"}

    package = Path(lab.__file__).parent
    monkeypatch.setattr(lab, "validate_shared_world", deepcopy)
    monkeypatch.setattr(camera_task, "build_camera_shared_task", build)
    monkeypatch.setattr(camera_task, "validate_camera_shared_task", validate)
    monkeypatch.setattr(lab, "collect_shared_source_files", lambda *args, **kwargs:
        SimpleNamespace(report={"mode": "analytic_fixture"}, files={}, validate_layout=lambda **kwargs: None))
    monkeypatch.setattr(lab, "_reference_frames", references)
    monkeypatch.setattr(lab, "_implementation_snapshot", lambda: {
        name: (package / name).read_bytes() for name in lab.runtime_files(scene)})
    return SimpleNamespace(source=source, scene=scene, task=task, calls=calls)


@pytest.mark.parametrize("mode", ["default", "roots", "sealed", "external"])
def test_camera_lab_preserves_source_and_distinct_task(inputs, tmp_path, mode):
    options = {}
    if mode == "sealed":
        write(inputs.source / "exercise-task.json", inputs.task)
        seal(inputs.source)
    if mode == "external":
        options["task_path"] = tmp_path / "task.json"
        write(options["task_path"], inputs.task)
    if mode == "roots":
        options["root_assistance"] = True
    original = {p.name: p.read_bytes() for p in inputs.source.iterdir()}
    target = tmp_path / "lab"
    result = lab.prepare_shared_exercise_lab(inputs.source, target, **options)
    loaded = lab.load_shared_exercise_lab(target / "run")
    assert result["control_profile"] == "shared_exercise" and not result["learned_policy_available"]
    assert loaded["scene"] == inputs.scene and loaded["task"]["profile"] == "camera-shared-task-v1"
    assert loaded["task"]["control"]["root_assistance"] == ("explicit_3_axis" if mode == "roots" else "disabled")
    assert loaded["contract"]["runtime_sha256"].keys() == set(lab.runtime_files(inputs.scene))
    assert "camera_shared_state.py" in loaded["contract"]["runtime_sha256"]
    assert "ground-clearance.json" in {item["name"] for item in loaded["artifacts"]}
    assert "no recovered feet" in loaded["contract"]["scope"]
    assert inputs.calls["reference"] == [inputs.scene]
    for name, raw in original.items():
        assert (inputs.source / name).read_bytes() == (target / "run/source-run" / name).read_bytes() == raw


@pytest.mark.parametrize("fault", ["policy", "evaluation", "hidden_policy", "coupling", "oversize", "stale_task", "root_conflict"])
def test_camera_admission_rejects_before_reference_or_output(inputs, tmp_path, fault):
    options = {}
    if fault == "policy":
        options["policy_run"] = tmp_path / "unread-policy"
    elif fault == "evaluation":
        options.update(policy_run=tmp_path / "unread-policy", evaluation_run=tmp_path / "unread-evaluation")
    elif fault == "hidden_policy":
        write(inputs.source / "policy.json", {})
    elif fault == "coupling":
        write(inputs.source / "camera-coupling.json", {"fixture": "changed"})
    else:
        task = deepcopy(inputs.task)
        if fault == "oversize":
            task["extra"] = "x" * (64 << 10)
        elif fault == "stale_task":
            task["scene_sha256"] = "f" * 64
        else:
            options["root_assistance"] = True
        write(inputs.source / "exercise-task.json", task)
    seal(inputs.source)
    with pytest.raises(ValueError):
        lab.prepare_shared_exercise_lab(inputs.source, tmp_path / "lab", **options)
    assert not (tmp_path / "lab").exists() and not inputs.calls["reference"]


def test_camera_resume_rejects_resealed_display_mismatch(inputs, tmp_path):
    target = tmp_path / "lab"
    lab.prepare_shared_exercise_lab(inputs.source, target)
    run = target / "run"
    write(run / "camera-coupling.json", {"fixture": "different display"})
    manifest = json.loads((run / "manifest.json").read_bytes())
    manifest["files"]["camera-coupling.json"] = sha256((run / "camera-coupling.json").read_bytes()).hexdigest()
    write(run / "manifest.json", manifest)
    with pytest.raises(ValueError, match="Displayed camera coupling"):
        lab.load_shared_exercise_lab(run)


def test_camera_avatar_options_close_before_session_exposure(monkeypatch, tmp_path):
    from weft.lab import Lab

    closed = []
    session = SimpleNamespace(scene_profile="shared-world-v6", env=SimpleNamespace(close=lambda: closed.append(True)))
    monkeypatch.setattr(lab, "SharedExerciseSession", lambda *args, **kwargs: session)
    instance = Lab.__new__(Lab)
    instance.session, instance.root = None, tmp_path
    instance.config = {"profile": "shared-world", "control_profile": "shared_exercise"}
    instance.avatars = SimpleNamespace(packages={"unsupported": object()})
    instance.status = lambda: {"status": "ready"}
    with pytest.raises(ValueError, match="without avatar options"):
        instance.load_session()
    assert closed == [True] and instance.session is None and not list(tmp_path.iterdir())
