"""Saved workflow publication and resource boundaries, with an explicit selector stub."""

from hashlib import sha256
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest

from weft import equipment_support_workflow as workflow
from test_equipment_support_snapshot import encoded


@pytest.fixture
def pipeline(tmp_path, monkeypatch):
    source = tmp_path / "source"; source.mkdir()
    (source / "manifest.json").write_bytes(b'{"saved_fixture":true}\n')
    out = tmp_path / "comparison"
    calls, checks = [], []
    source_files = {"manifest.json": (source / "manifest.json").read_bytes()}
    silhouette = SimpleNamespace(equipment_document=b'{"equipment":"exact"}', source_equivalence=None)
    snapshot = SimpleNamespace(pose={"source_sha256": "a" * 64}, silhouette=silhouette,
        equipment_ids=("weight",), source_files=source_files,
        source_manifest_sha256=sha256(source_files["manifest.json"]).hexdigest(),
        read_work={"scope": "Analytical restoration stub", "selected_bytes": len(source_files["manifest.json"])},
        recheck=lambda deadline: checks.append(deadline))
    monkeypatch.setattr(workflow, "load_equipment_support_snapshot", lambda *a, **k: snapshot)
    selector = ModuleType("weft.exercise_support_evidence")
    selector.CONTRACT = {"maximum_seconds": 10.}
    def estimate(pose, **options):
        calls.append(options)
        assert pose is snapshot.pose and options["equipment_silhouette"] is silhouette
        return {"analytical_selector_stub": True, "contact_state": "unknown"}
    selector.estimate_exercise_support = estimate
    monkeypatch.setitem(sys.modules, selector.__name__, selector)
    renderer = ModuleType("weft.equipment_support_report")
    def build(raw, *, max_seconds):
        assert 0 < max_seconds <= 5
        return {"index.html": b"<title>Analytical report stub</title>", "support-evidence.json": raw, "brand.css": b"/* exact */"}
    renderer.build_equipment_support_report = build
    monkeypatch.setitem(sys.modules, renderer.__name__, renderer)
    monkeypatch.setattr(workflow, "_runtime_sources", lambda deadline: {"assets/brand.css": b"/* exact */", "workflow.py": b"# fixture\n"})
    return source, out, calls, checks, renderer, snapshot


def test_one_selector_call_preserved_inputs_and_terminal_manifest(pipeline):
    source, out, calls, checks, _, snapshot = pipeline
    stages = []
    result = workflow.prepare_equipment_support(source, out, fixed_camera=True,
        progress=lambda stage, _: stages.append(stage))
    assert stages == ["inputs", "selection", "report", "seal"]
    assert len(calls) == len(checks) == 1
    assert calls[0]["camera_prior"]["status"] == "authored_unverified"
    assert calls[0]["equipment_ids"] == ("weight",)
    assert all(result[k] == 0 for k in ("source_decodes", "raster_decodes", "native_calls", "api_calls", "pose_model_calls"))
    assert result["body_or_floor_installed"] is False and result["physical_validation"] == "unverified"
    assert (out / "source/manifest.json").read_bytes() == snapshot.source_files["manifest.json"]
    seal = json.loads((out / "manifest.json").read_bytes())
    assert set(seal["files"]) == {p.relative_to(out).as_posix() for p in out.rglob("*") if p.is_file()} - {"manifest.json"}
    assert all(sha256((out / name).read_bytes()).hexdigest() == pin for name, pin in seal["files"].items())
    assert sha256((out / "support-evidence.json").read_bytes()).hexdigest() == result["support_evidence_sha256"]


@pytest.mark.parametrize("fixed", [False, 1, "true"])
def test_camera_assumption_must_be_explicit_boolean(pipeline, fixed):
    source, out, calls, *_ = pipeline
    with pytest.raises(ValueError, match="fixed-camera"):
        workflow.prepare_equipment_support(source, out, fixed_camera=fixed)
    assert not calls and not out.exists()


@pytest.mark.parametrize("seconds", [0, 14.9, 30.1, float("nan")])
def test_time_bound_rejects_before_read_or_output(pipeline, seconds):
    source, out, calls, *_ = pipeline
    with pytest.raises(ValueError, match="time bound"):
        workflow.prepare_equipment_support(source, out, fixed_camera=True, max_seconds=seconds)
    assert not calls and not out.exists()


@pytest.mark.parametrize("fault", ["missing_raw", "changed_raw", "changed_css", "reserved", "unsafe", "oversized"])
def test_report_contract_violation_never_publishes(pipeline, fault, monkeypatch):
    source, out, _, _, renderer, _ = pipeline
    normal = renderer.build_equipment_support_report
    def changed(raw, **kwargs):
        files = normal(raw, **kwargs)
        if fault == "missing_raw": files.pop("support-evidence.json")
        if fault == "changed_raw": files["support-evidence.json"] = b"{}"
        if fault == "changed_css": files["brand.css"] = b"changed"
        if fault == "reserved": files["manifest.json"] = b"{}"
        if fault == "unsafe": files["../outside.html"] = b"outside"
        if fault == "oversized": files["index.html"] = b"x" * 100
        return files
    renderer.build_equipment_support_report = changed
    if fault == "oversized": monkeypatch.setattr(workflow, "MAX_OUTPUT", 100)
    with pytest.raises(ValueError):
        workflow.prepare_equipment_support(source, out, fixed_camera=True)
    assert not out.exists()


def test_existing_nested_and_symlink_outputs_are_not_overwritten(pipeline, tmp_path):
    source, out, calls, *_ = pipeline
    out.mkdir()
    sentinel = out / "keep"; sentinel.write_text("preserve")
    for target in (out, source / "nested"):
        with pytest.raises(ValueError):
            workflow.prepare_equipment_support(source, target, fixed_camera=True)
    alias = tmp_path / "alias"; alias.symlink_to(source, target_is_directory=True)
    with pytest.raises(ValueError):
        workflow.prepare_equipment_support(alias, tmp_path / "new", fixed_camera=True)
    assert sentinel.read_text() == "preserve" and not calls


def test_progress_time_cannot_escape_remaining_selector_budget(pipeline, monkeypatch):
    source, out, calls, _, _, snapshot = pipeline
    clock = {"now": 0.}
    monkeypatch.setattr(workflow.time, "monotonic", lambda: clock["now"])
    def progress(stage, detail):
        if stage == "selection": clock["now"] = 6.
    with pytest.raises(ValueError, match="remaining time"):
        workflow.prepare_equipment_support(source, out, fixed_camera=True, max_seconds=15., progress=progress)
    assert not calls and not out.exists()


def test_interrupted_package_is_unsealed_and_original_is_unchanged(pipeline, monkeypatch):
    source, out, _, _, _, snapshot = pipeline
    original = workflow.write_package_files
    def interrupted(directory, files):
        assert "manifest.json" not in files
        name = next(iter(files)); original(directory, {name: files[name]})
        raise OSError("Analytical interruption before terminal seal")
    monkeypatch.setattr(workflow, "write_package_files", interrupted)
    with pytest.raises(OSError, match="interruption"):
        workflow.prepare_equipment_support(source, out, fixed_camera=True)
    assert out.exists() and not (out / "manifest.json").exists()
    assert (source / "manifest.json").read_bytes() == snapshot.source_files["manifest.json"]


def test_recheck_failure_prevents_output(pipeline):
    source, out, _, _, _, snapshot = pipeline
    def changed(deadline): raise ValueError("selected input changed")
    snapshot.recheck = changed
    with pytest.raises(ValueError, match="selected input changed"):
        workflow.prepare_equipment_support(source, out, fixed_camera=True)
    assert not out.exists()
