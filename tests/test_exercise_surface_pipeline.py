"""Saved-input orchestration with real collector joins; native/media/network blocked."""

from copy import deepcopy
from hashlib import sha256
import importlib
import json
from pathlib import Path
import re

import pytest

from test_multiview_landmark_evidence import denied, local_only


def raw(value):
    return json.dumps(value, sort_keys=True, allow_nan=False).encode()


@pytest.fixture
def api(monkeypatch):
    module = importlib.import_module("weft.exercise_surface_pipeline")
    monkeypatch.setattr(module, "_runtime", lambda reads: ({"runtime/weft/fixture.py": b"# analytical routing fixture\n"},
                                                        {"python": "fixture", "numpy": "fixture"}))
    def report(files, max_seconds):
        state = json.loads(files["workflow-state.json"])
        assert state["status"] == "completed" and 0 < max_seconds <= 5
        assert not any(name.startswith("runtime/") or name in {"plan.json", "parent-manifest.json"} for name in files)
        return {"index.html": b"<!doctype html><title>Saved analytical workflow</title>", "report.json": raw(state)}
    monkeypatch.setattr(module, "_report", report)
    return module


@pytest.fixture
def inputs(tmp_path):
    helper = importlib.import_module("weft.exercise_surface_inputs")
    fixture = importlib.import_module("test_exercise_surface_inputs").fixture
    _, request, values = fixture(helper)
    source = tmp_path / "inputs"; source.mkdir()
    observations = []
    for index, data in enumerate(values["observation_documents"].values()):
        path = source / f"observations-{index}.json"; path.write_bytes(data); observations.append(path)
    camera, body = source / "camera.json", source / "body.json"
    camera.write_bytes(values["camera_document"]); body.write_bytes(values["body_document"])
    return {"observations": observations, "camera": camera, "body": body,
            "request": request, "values": values}


def seal(out):
    manifest = json.loads((out / "manifest.json").read_bytes())
    actual = {path.relative_to(out).as_posix(): sha256(path.read_bytes()).hexdigest()
              for path in out.rglob("*") if path.is_file() and path != out / "manifest.json"}
    assert manifest["files"] == actual
    assert json.loads((out / "result.json").read_bytes())["status"] == manifest["status"]
    return manifest


def discover(api, inputs, out, **kwargs):
    return api.prepare_exercise_surface_collection(inputs["observations"], out,
        camera=inputs["camera"], body=inputs["body"], **kwargs)


def select(api, inputs, tmp_path):
    first, second = tmp_path / "discovery", tmp_path / "draft"
    result = discover(api, inputs, first)
    assert result["status"] == "completed", result
    selection = json.loads((first / "selection.json").read_bytes())
    selection.update(channels=inputs["values"]["channels"], human_entity_ref=inputs["values"]["human_entity_ref"])
    edit = tmp_path / "selected.json"; edit.write_bytes(raw(selection))
    result = api.prepare_exercise_surface_collection([], second, resume=first, selection=edit,
                                                     expected_manifest_sha256=result["manifest_sha256"])
    assert result["status"] == "completed", result
    return first, second


def test_discovery_has_no_chosen_entities_or_geometry(api, inputs, tmp_path, monkeypatch):
    monkeypatch.setattr(api.collector, "prepare_exercise_surface_inputs", denied)
    monkeypatch.setattr(api.collector, "prepare_source_surface_envelopes", denied)
    out = tmp_path / "out"; progress = []
    result = discover(api, inputs, out, progress=lambda stage, detail: progress.append((stage, detail)))
    assert result["status"] == "completed" and result["sealed"] is True
    assert result["collection_status"] == "needs_selection" and Path(result["report_path"]).is_file()
    chosen = json.loads((out / "selection.json").read_bytes())
    assert chosen["channels"] == [] and chosen["human_entity_ref"] is None
    assert not (out / "collection-request.json").exists() and not (out / "surface-config.json").exists()
    inventory = json.loads((out / "inventory.json").read_bytes())
    assert inventory["input_bindings"]["source_sha256"] == inputs["values"]["source_sha256"]
    assert inventory["entities"][0]["entity_ref"] == inputs["values"]["human_entity_ref"]
    assert inventory["entities"][0]["construction_annotation_slots"] == 4
    assert [stage for stage, _ in progress] == ["inputs", "selection", "qualification", "report"]
    assert (out / "original-human.json").read_bytes() == inputs["body"].read_bytes()
    seal(out)


def test_selected_draft_preserves_all_slots_without_geometry(api, inputs, tmp_path, monkeypatch):
    monkeypatch.setattr(api.collector, "prepare_source_surface_envelopes", denied)
    first, out = select(api, inputs, tmp_path)
    result = json.loads((out / "result.json").read_bytes())
    assert result["collection_status"] == "incomplete_request" and result["config_path"] is None
    request = json.loads((out / "collection-request.json").read_bytes())
    assert request["weight"] is None and request["body_binding"]["body_to_world"] is None
    rows = [json.loads(row) for row in (out / "collection-frames.jsonl").read_bytes().splitlines()]
    assert [row["source"]["source_frame"] for row in rows] == list(range(4))
    assert all(row["envelope"] is None for row in rows)
    assert (out / "parent-manifest.json").read_bytes() == (first / "manifest.json").read_bytes()
    seal(first); seal(out)


def test_request_resume_uses_real_collector_config_and_no_duplicate_sources(api, inputs, tmp_path):
    first, draft = select(api, inputs, tmp_path)
    old_seals = [(path / "manifest.json").read_bytes() for path in (first, draft)]
    edit = tmp_path / "request.json"; edit.write_bytes(raw(inputs["request"]))
    out = tmp_path / "qualified"
    result = api.prepare_exercise_surface_collection([], out, resume=draft, request=edit)
    assert result["status"] == "completed" and result["collection_status"] == "conditional_inputs_available", result
    assert result["fit_or_simulator_admission"] is result["body_or_floor_installed"] is False
    report = json.loads((out / "collection-report.json").read_bytes())
    assert report["conditional_envelopes"] == 4 and report["conditional_body_channel_terms"] == 2
    assert (out / "collection-request.json").read_bytes() == edit.read_bytes()
    assert (out / "selection.json").read_bytes() == (draft / "selection.json").read_bytes()
    config = json.loads((out / "surface-config.json").read_bytes())
    assert config["profile"] == "source-surface-fit-config-v1"
    assert config["observations"] == [{"path": "surface-observation-00.json",
                                      "sha256": sha256(inputs["observations"][0].read_bytes()).hexdigest()}]
    assert len(list(out.rglob("surface-observation-00.json"))) == 1
    assert old_seals == [(path / "manifest.json").read_bytes() for path in (first, draft)]
    assert len((out / "collection-frames.jsonl").read_bytes().splitlines()) == 4
    seal(out)


def test_unknown_height_remains_qualified_unavailable(api, inputs, tmp_path):
    _, draft = select(api, inputs, tmp_path)
    request = deepcopy(inputs["request"])
    request["surface_intervals"]["surfaces"][0].update(height_interval_m=None, quality="unknown")
    edit = tmp_path / "unknown.json"; edit.write_bytes(raw(request))
    out = tmp_path / "unknown"
    result = api.prepare_exercise_surface_collection([], out, resume=draft, request=edit)
    assert result["status"] == "completed" and result["collection_status"] == "qualified_but_unavailable", result
    assert result["contact"] == "unknown" and result["config_path"] is not None
    assert json.loads((out / "collection-report.json").read_bytes())["conditional_body_channel_terms"] == 0
    seal(out)


@pytest.mark.parametrize("options", [
    {"request": "unused.json"}, {"max_seconds": True}, {"max_seconds": 31},
    {"expected_manifest_sha256": "a" * 64}, {"selection": "unused", "request": "unused"},
])
def test_invalid_fresh_options_fail_before_output(api, inputs, tmp_path, options):
    out = tmp_path / "out"
    with pytest.raises(ValueError):
        discover(api, inputs, out, **options)
    assert not out.exists()


@pytest.mark.parametrize("options", [
    {}, {"source_sha256": "a" * 64, "request": "unused"}, {"camera": "unused", "request": "unused"},
    {"body": "unused", "selection": "unused"}, {"request": "unused", "selection": "unused"},
])
def test_resume_requires_one_edit_and_rejects_mixed_source_flags(api, tmp_path, options):
    out = tmp_path / "out"
    with pytest.raises(ValueError):
        api.prepare_exercise_surface_collection([], out, resume=tmp_path / "parent", **options)
    assert not out.exists()


def test_request_cannot_guess_missing_selection(api, inputs, tmp_path):
    first = tmp_path / "first"; discover(api, inputs, first)
    edit = tmp_path / "request.json"; edit.write_bytes(raw(inputs["request"]))
    out = tmp_path / "out"
    result = api.prepare_exercise_surface_collection([], out, resume=first, request=edit)
    assert result["status"] == "failed" and "previously usable explicit selection" in result["error"]["message"]
    assert result["report_path"] is None and (out / "collection-request.json").read_bytes() == edit.read_bytes()
    seal(out)


def test_wrong_selection_pin_is_preserved_as_a_failed_package(api, inputs, tmp_path):
    first = tmp_path / "first"; discover(api, inputs, first)
    selection = json.loads((first / "selection.json").read_bytes())
    selection["input_bindings"]["source_sha256"] = "f" * 64
    edit = tmp_path / "edited.json"; edit.write_bytes(raw(selection))
    out = tmp_path / "out"
    result = api.prepare_exercise_surface_collection([], out, resume=first, selection=edit)
    assert result["status"] == "failed" and result["sealed"] is True
    assert (out / "selection.json").read_bytes() == edit.read_bytes()
    seal(out)


def test_deleted_source_after_outer_reseal_fails_required_inventory(api, inputs, tmp_path):
    first = tmp_path / "first"; discover(api, inputs, first)
    edit = tmp_path / "selected.json"; edit.write_bytes((first / "selection.json").read_bytes())
    (first / "surface-camera.json").unlink()
    manifest = json.loads((first / "manifest.json").read_bytes())
    manifest["files"].pop("surface-camera.json")
    (first / "manifest.json").write_bytes(raw(manifest))
    out = tmp_path / "out"
    result = api.prepare_exercise_surface_collection([], out, resume=first, selection=edit)
    assert result["status"] == "failed" and "original input inventory" in result["error"]["message"]
    seal(out)


def test_source_mutation_during_report_cannot_publish_completed(api, inputs, tmp_path, monkeypatch):
    def report(files, max_seconds):
        inputs["camera"].write_bytes(inputs["camera"].read_bytes() + b" ")
        return {"index.html": b"<html>attempt</html>"}
    monkeypatch.setattr(api, "_report", report)
    out = tmp_path / "out"; result = discover(api, inputs, out)
    assert result["status"] == "failed" and result["error"]["stage"] == "source_recheck"
    assert result["report_path"] is None and result["sealed"] is True
    seal(out)


def test_failure_before_manifest_preserves_attempt_and_failed_authority(api, inputs, tmp_path, monkeypatch):
    original = api._store
    def interrupted(out, files, saved, *, terminal=False):
        if "manifest.json" in files and json.loads(files["manifest.json"])["status"] == "completed":
            raise TimeoutError("Injected failure before completed seal publication")
        original(out, files, saved, terminal=terminal)
    monkeypatch.setattr(api, "_store", interrupted)
    out = tmp_path / "out"; result = discover(api, inputs, out)
    assert result["status"] == "failed" and result["sealed"] is True and result["report_path"] is None
    assert not (out / "attempted-manifest.json").exists()
    assert json.loads((out / "attempted-result.json").read_bytes())["status"] == "completed"
    assert seal(out)["status"] == "failed"


def test_completed_manifest_stays_terminal_after_publication(api, inputs, tmp_path, monkeypatch):
    original = api._store
    def notification_failure(out, files, saved, *, terminal=False):
        original(out, files, saved, terminal=terminal)
        if "manifest.json" in files:
            raise TimeoutError("Injected notification failure after valid publication")
    monkeypatch.setattr(api, "_store", notification_failure)
    out = tmp_path / "out"; result = discover(api, inputs, out)
    assert result["status"] == "completed" and result["sealed"] is True and result["report_path"] is not None
    assert not list(out.glob("attempted-*.json"))
    assert seal(out)["status"] == "completed"


@pytest.mark.parametrize("name", ["../escape.html", "/escape.html", "nested/index.html"])
def test_report_paths_cannot_escape_package(api, inputs, tmp_path, monkeypatch, name):
    monkeypatch.setattr(api, "_report", lambda *a, **kw: {"index.html": b"ok", name: b"unsafe"})
    out = tmp_path / "out"; result = discover(api, inputs, out)
    assert result["status"] == "failed" and result["error"]["stage"] == "report"
    assert not (tmp_path / "escape.html").exists()
    seal(out)


def test_symlink_and_output_overlap_fail_before_output(api, inputs, tmp_path):
    link = tmp_path / "link.json"; link.symlink_to(inputs["observations"][0])
    out = tmp_path / "out"
    with pytest.raises(ValueError, match="symlink"):
        api.prepare_exercise_surface_collection([link], out)
    with pytest.raises(ValueError, match="new"):
        api.prepare_exercise_surface_collection(inputs["observations"], inputs["observations"][0].parent)
    assert not out.exists()


def test_multiple_source_declarations_and_source_pin_are_not_relabelled(api, inputs, tmp_path):
    out = tmp_path / "out"
    result = discover(api, inputs, out, source_sha256="f" * 64)
    assert result["status"] == "failed" and "explicit source identity differs" in result["error"]["message"]
    assert (out / "surface-observation-00.json").read_bytes() == inputs["observations"][0].read_bytes()
    seal(out)


def test_cumulative_source_limit_is_applied_before_next_read(api, monkeypatch):
    monkeypatch.setattr(api, "MAX_OBSERVATION_BYTES", 32)
    limits = []
    class Reads:
        def read(self, path, limit):
            limits.append(limit)
            data = b"a" * 17 if path == "first" else b"b" * 17
            if len(data) > limit:
                raise ValueError("Bounded read rejects before allocation")
            return data
    with pytest.raises(ValueError, match="before allocation"):
        api._source_inputs(["first", "second"], {}, Reads())
    assert limits == [32, 15]


def test_real_report_and_frozen_runtime_across_one_collection(inputs, tmp_path):
    module = importlib.import_module("weft.exercise_surface_pipeline")
    first, draft = select(module, inputs, tmp_path)
    request_path = tmp_path / "edited-request.json"
    request_path.write_bytes(raw(inputs["request"]))
    out = tmp_path / "qualified"
    result = module.prepare_exercise_surface_collection([], out, resume=draft, request=request_path)
    assert result["status"] == "completed" and result["collection_status"] == "conditional_inputs_available", result
    manifest = seal(out)
    plan = json.loads((out / "plan.json").read_bytes())
    assert set(plan["runtime_files"]) == {"runtime/weft/" + name for name in module.RUNTIME_FILES}
    assert plan["runtime_files"]["runtime/weft/assets/brand.css"] == manifest["files"]["inspection/brand.css"]
    assert not any("human.py" in name or "camera_body_fit.py" in name for name in plan["runtime_files"])
    for path in (first, draft, out):
        report = (path / "inspection/index.html").read_text()
        assert "<html" in report and re.search(r'(?:src|href)=["\']https?://', report) is None
        assert len((path / "selection.json").read_bytes()) < module.MAX_SELECTION_BYTES
        seal(path)
    assert len((out / "collection-frames.jsonl").read_bytes().splitlines()) == 4
