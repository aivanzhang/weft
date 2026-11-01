"""Complete authored source/runtime unions fit without dropping their sealed members."""

from pathlib import Path
import time

import pytest

from weft import shared_exercise_lab as lab
from weft.shared_exercise_task import build_shared_exercise_task
from test_shared_exercise_run import source_fixture
from test_split_limb_feedback_learning import analytical_metadata, no_simulation


def capacity_inputs(tmp_path, monkeypatch, *, duplicated_views=False):
    source, scene = source_fixture(tmp_path)
    task = build_shared_exercise_task(scene, root_assistance=True)
    source_files = {"scene.json": (source / "scene.json").read_bytes(),
                    "exercise-task.json": lab._bytes(task)}
    prefix = "geometry-inspection" if duplicated_views else "authored-metadata"
    for index in range(510):
        source_files[f"{prefix}/{index:03}.json"] = lab._bytes({"authored_member": index})
    for name, raw in source_files.items():
        path = source / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
    source_manifest = lab._bytes({"status": "ready", "files": {
        name: lab._sha(raw) for name, raw in source_files.items()}})
    (source / "manifest.json").write_bytes(source_manifest)

    package = Path(lab.__file__).parent
    implementation = {name: (package / name).read_bytes() for name in lab.runtime_files(scene)}
    implementation["shared_exercise_lab.py"] = (package / "shared_exercise_lab.py").read_bytes()
    for index in range(512 - len(implementation)):
        implementation[f"authored_capacity_{index:03}.py"] = b"# Authored packaging fixture; never executed.\n"
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    for name, raw in implementation.items():
        (runtime / name).write_bytes(raw)

    # Real final hash rechecks read this temporary snapshot, including every actual runtime dependency.
    monkeypatch.setattr(lab, "__file__", str(runtime / "shared_exercise_lab.py"))
    monkeypatch.setattr(lab, "_implementation_snapshot", lambda: dict(implementation))
    monkeypatch.setattr(lab, "_reference_frames", lambda *_: {
        "frames": [], "scope": "Authored capacity fixture; no geometry or simulation."})
    return source, scene, task, source_files, source_manifest, implementation


def test_complete_source_and_runtime_union_prepares_and_resumes(tmp_path, monkeypatch):
    source, scene, task, inputs, source_manifest, implementation = capacity_inputs(tmp_path, monkeypatch)
    out = tmp_path / "lab"
    result = lab.prepare_shared_exercise_lab(source, out, max_seconds=10)
    loaded = lab.load_shared_exercise_lab(out / "run", deadline=time.monotonic() + 10)

    assert result["files"] > 1024
    assert loaded["scene"] == scene and loaded["task"] == task
    assert loaded["contract"]["source_files_verified"] == 512
    assert loaded["contract"]["implementation_sha256"] == {
        name: lab._sha(raw) for name, raw in implementation.items()}
    manifest = lab._json((out / "run/manifest.json").read_bytes())
    for prefix, members in (("source-run/", inputs), ("lab-implementation/", implementation)):
        for name, raw in members.items():
            assert (out / "run" / (prefix + name)).read_bytes() == raw
            assert manifest["files"][prefix + name] == lab._sha(raw)
    assert (out / "run/source-run/manifest.json").read_bytes() == source_manifest
    for name in ("scene.json", "exercise-task.json"):
        assert (out / "run" / name).read_bytes() == inputs[name]
    assert (source / "manifest.json").read_bytes() == source_manifest
    assert all((source / name).read_bytes() == raw for name, raw in inputs.items())


def test_combined_view_count_overflow_rejects_before_output(tmp_path, monkeypatch):
    source, _, _, _, _, _ = capacity_inputs(tmp_path, monkeypatch, duplicated_views=True)
    out = tmp_path / "lab"
    with pytest.raises(ValueError, match="complete source/runtime closure exceeds its output bound"):
        lab.prepare_shared_exercise_lab(source, out, max_seconds=10)
    assert not out.exists()


def test_combined_byte_overflow_rejects_before_output(tmp_path, monkeypatch):
    source, _, _, inputs, _, implementation = capacity_inputs(tmp_path, monkeypatch)
    # A smaller test-only aggregate budget exercises rejection without a 192 MiB allocation.
    budget = max(sum(map(len, inputs.values())), sum(map(len, implementation.values())))
    monkeypatch.setattr(lab, "MAX_OUTPUT_BYTES", budget)
    out = tmp_path / "lab"
    with pytest.raises(ValueError, match="complete source/runtime closure exceeds its output bound"):
        lab.prepare_shared_exercise_lab(source, out, max_seconds=10)
    assert not out.exists()


def test_final_readback_rejects_tampered_written_member(tmp_path, monkeypatch):
    source, _, _, inputs, _, _ = capacity_inputs(tmp_path, monkeypatch)
    out = tmp_path / "lab"
    original_loader, calls = lab.load_shared_exercise_lab, []
    member = "authored-metadata/000.json"

    def tamper_before_readback(root, **kwargs):
        calls.append(root)
        (root / "source-run" / member).write_bytes(b'{"changed_after_write":true}\n')
        return original_loader(root, **kwargs)

    monkeypatch.setattr(lab, "load_shared_exercise_lab", tamper_before_readback)
    with pytest.raises(ValueError, match="SHA256"):
        lab.prepare_shared_exercise_lab(source, out, max_seconds=10)
    assert calls == [out / "run"]
    assert (source / member).read_bytes() == inputs[member]
    manifest = lab._json((out / "run/manifest.json").read_bytes())
    assert manifest["files"]["source-run/" + member] == lab._sha(inputs[member])
    assert lab._json((out / "failure.json").read_bytes())["status"] == "failed"
    assert not (out / "lab-config.json").exists()
