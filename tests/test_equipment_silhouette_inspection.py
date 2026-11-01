"""The inspector checks selected saved bytes, never the original media closure."""

from hashlib import sha256
import json
from pathlib import Path
import sys
from types import ModuleType

import pytest

from weft import equipment_silhouette_inspection as inspector


def encoded(value):
    return (json.dumps(value, sort_keys=True) + "\n").encode()


@pytest.fixture
def study(tmp_path):
    source = tmp_path / "source"
    (source / "silhouette").mkdir(parents=True)
    values = {"silhouette/metadata.json": b'{"fixture":"metadata"}\n',
              "silhouette/frames.jsonl": b'{"source_frame":0}\n'}
    for name, raw in values.items():
        (source / name).write_bytes(raw)
    manifest = {"schema_version": 1, "profile": "pose-equipment-silhouette-composition-v1", "status": "completed",
                "files": {**{name: sha256(raw).hexdigest() for name, raw in values.items()},
                          "inputs/absent-original-video.webm": "a" * 64}}
    (source / "manifest.json").write_bytes(encoded(manifest))
    return source, tmp_path / "inspection", values, manifest


@pytest.fixture
def renderer(monkeypatch):
    module = ModuleType("weft.equipment_silhouette_report")
    calls = []
    def build(metadata, frames, *, max_seconds):
        calls.append((metadata, frames, max_seconds))
        return {"index.html": b'<!doctype html><title>Analytical report stub</title>',
                "equipment-metadata.json": metadata, "equipment-frames.jsonl": frames,
                "brand.css": (Path(inspector.__file__).parent / "assets/brand.css").read_bytes(),
                "report.json": b'{"scope":"mock rendering only"}'}
    module.build_equipment_silhouette_report = build
    monkeypatch.setitem(sys.modules, module.__name__, module)
    original = inspector.read_package_file
    def read(root, name, **kwargs):
        if Path(root) == Path(inspector.__file__).parent and name == "equipment_silhouette_report.py":
            return b'# Analytical renderer stub; not a native test.\n'
        return original(root, name, **kwargs)
    monkeypatch.setattr(inspector, "read_package_file", read)
    return module, calls


def test_report_copies_selected_bytes_once_and_keeps_source_scope_explicit(study, renderer, monkeypatch):
    source, out, values, _ = study
    original = inspector.read_package_file
    reads, stages = [], []
    def read(root, name, **kwargs):
        if Path(root) == source:
            reads.append(name)
        return original(root, name, **kwargs)
    monkeypatch.setattr(inspector, "read_package_file", read)
    seal_raw = (source / "manifest.json").read_bytes()
    result = inspector.prepare_equipment_silhouette_inspection(source, out,
        expected_manifest_sha256=sha256(seal_raw).hexdigest(), progress=lambda stage, _: stages.append(stage))
    assert result["status"] == "completed" and result["sealed"] is True
    assert reads == ["manifest.json", *inspector.MEMBERS]
    assert stages == ["inputs", "report", "seal"]
    assert "not revalidated" in result["verification_scope"]
    assert result["read_work"] == {"selected_files": 3, "selected_bytes": len(seal_raw) + sum(map(len, values.values()))}
    assert (out / "source-manifest.json").read_bytes() == seal_raw
    seal = json.loads((out / "manifest.json").read_bytes())
    assert set(seal["files"]) == {p.relative_to(out).as_posix() for p in out.rglob("*") if p.is_file()} - {"manifest.json"}
    for name, pin in seal["files"].items():
        assert sha256((out / name).read_bytes()).hexdigest() == pin
    for name, raw in values.items():
        assert (source / name).read_bytes() == raw
    assert len(renderer[1]) == 1 and 0 < renderer[1][0][2] <= 5


@pytest.mark.parametrize("fault", ["expected_seal", "metadata_pin", "frame_pin", "pending", "schema_bool", "missing_pin"])
def test_bad_source_never_creates_a_report(study, renderer, fault):
    source, out, _, manifest = study
    options = {}
    if fault == "expected_seal": options["expected_manifest_sha256"] = "f" * 64
    if fault == "metadata_pin": manifest["files"]["silhouette/metadata.json"] = "f" * 64
    if fault == "frame_pin": manifest["files"]["silhouette/frames.jsonl"] = "f" * 64
    if fault == "pending": manifest["status"] = "running"
    if fault == "schema_bool": manifest["schema_version"] = True
    if fault == "missing_pin": del manifest["files"]["silhouette/frames.jsonl"]
    (source / "manifest.json").write_bytes(encoded(manifest))
    with pytest.raises(ValueError):
        inspector.prepare_equipment_silhouette_inspection(source, out, **options)
    assert not out.exists() and not renderer[1]


def test_duplicate_manifest_keys_and_oversized_selected_member_are_rejected(study, renderer):
    source, out, _, manifest = study
    (source / "manifest.json").write_bytes(b'{"schema_version":1,"schema_version":1}')
    with pytest.raises(ValueError, match="strict finite"):
        inspector.prepare_equipment_silhouette_inspection(source, out)
    (source / "manifest.json").write_bytes(encoded(manifest))
    (source / "silhouette/metadata.json").write_bytes(b" " * ((256 << 10) + 1))
    with pytest.raises(ValueError, match="bounded regular"):
        inspector.prepare_equipment_silhouette_inspection(source, out)
    assert not out.exists() and not renderer[1]


def test_unsealed_study_explains_the_next_step_without_reading_failure_payload(tmp_path, renderer):
    source, out = tmp_path / "failed-study", tmp_path / "report"
    source.mkdir()
    (source / "failure.json").write_bytes(b"Do not parse unrelated failed-stage payloads")
    with pytest.raises(ValueError, match="no completed manifest.*failure.json or process receipt"):
        inspector.prepare_equipment_silhouette_inspection(source, out)
    assert not out.exists() and not renderer[1]


@pytest.mark.parametrize("kind", ["source", "selected", "parent"])
def test_symlinks_cannot_redirect_reads_or_output(study, renderer, tmp_path, kind):
    source, out, _, _ = study
    if kind == "source":
        alias = tmp_path / "source-link"; alias.symlink_to(source, target_is_directory=True); source = alias
    elif kind == "selected":
        original = source / "silhouette/metadata.json"
        moved = source / "original.json"; original.rename(moved); original.symlink_to(moved)
    else:
        alias = tmp_path / "output-link"; alias.symlink_to(tmp_path, target_is_directory=True); out = alias / "new"
    with pytest.raises(ValueError, match="symlinks"):
        inspector.prepare_equipment_silhouette_inspection(source, out)
    assert not renderer[1]


def test_existing_or_nested_output_never_overwrites_source(study, renderer):
    source, out, _, _ = study
    out.mkdir(); keep = out / "keep.txt"; keep.write_text("user data")
    with pytest.raises(ValueError, match="new output"):
        inspector.prepare_equipment_silhouette_inspection(source, out)
    with pytest.raises(ValueError, match="outside"):
        inspector.prepare_equipment_silhouette_inspection(source, source / "nested")
    assert keep.read_text() == "user data" and not renderer[1]


def test_source_mutation_during_report_is_not_published(study, renderer):
    source, out, _, _ = study
    module, _ = renderer
    build = module.build_equipment_silhouette_report
    def mutate(*args, **kwargs):
        result = build(*args, **kwargs)
        (source / "silhouette/frames.jsonl").write_bytes(b'{"source_frame":9}\n')
        return result
    module.build_equipment_silhouette_report = mutate
    with pytest.raises(ValueError, match="changed before"):
        inspector.prepare_equipment_silhouette_inspection(source, out)
    assert not out.exists()


@pytest.mark.parametrize("name", ["../escape.html", "/absolute.html", "manifest.json", "result.json", "source-manifest.json"])
def test_renderer_cannot_override_custody_or_escape_output(study, renderer, name):
    source, out, _, _ = study
    module, _ = renderer
    build = module.build_equipment_silhouette_report
    module.build_equipment_silhouette_report = lambda *a, **k: {**build(*a, **k), name: b"replacement"}
    with pytest.raises(ValueError, match="unsupported output"):
        inspector.prepare_equipment_silhouette_inspection(source, out)
    assert not out.exists()


def test_renderer_cannot_rewrite_the_downloadable_evidence(study, renderer):
    source, out, _, _ = study
    module, _ = renderer
    build = module.build_equipment_silhouette_report
    module.build_equipment_silhouette_report = lambda *a, **k: {**build(*a, **k), "equipment-frames.jsonl": b"rewritten"}
    with pytest.raises(ValueError, match="exact saved inputs"):
        inspector.prepare_equipment_silhouette_inspection(source, out)
    assert not out.exists()


def test_rendered_stylesheet_matches_the_archived_generator_asset(study, renderer):
    source, out, _, _ = study
    module, _ = renderer
    build = module.build_equipment_silhouette_report
    module.build_equipment_silhouette_report = lambda *a, **k: {**build(*a, **k), "brand.css": b"unrecorded stylesheet"}
    with pytest.raises(ValueError, match="stylesheet differs"):
        inspector.prepare_equipment_silhouette_inspection(source, out)
    assert not out.exists()


def test_total_read_budget_stops_before_oversized_next_member(study, renderer, monkeypatch):
    source, out, _, _ = study
    monkeypatch.setattr(inspector, "MAX_INPUT", (source / "manifest.json").stat().st_size + 3)
    with pytest.raises(ValueError, match="bounded regular"):
        inspector.prepare_equipment_silhouette_inspection(source, out)
    assert not out.exists() and not renderer[1]


def test_expired_report_never_publishes_a_terminal_seal(study, renderer, monkeypatch):
    source, out, _, _ = study
    module, _ = renderer
    clock = [100.]
    monkeypatch.setattr(inspector.time, "monotonic", lambda: clock[0])
    build = module.build_equipment_silhouette_report
    def expire(*a, **k):
        result = build(*a, **k); clock[0] = 120.; return result
    module.build_equipment_silhouette_report = expire
    with pytest.raises(TimeoutError):
        inspector.prepare_equipment_silhouette_inspection(source, out, max_seconds=1.)
    assert not out.exists()


def test_partial_output_is_retained_without_a_completed_seal(study, renderer, monkeypatch):
    source, out, _, _ = study
    def fail(root, files):
        (root / "partial.txt").write_text("Preserve interrupted output")
        raise OSError("controlled write failure")
    monkeypatch.setattr(inspector, "write_package_files", fail)
    with pytest.raises(OSError, match="controlled write"):
        inspector.prepare_equipment_silhouette_inspection(source, out)
    assert (out / "partial.txt").is_file() and not (out / "manifest.json").exists()
