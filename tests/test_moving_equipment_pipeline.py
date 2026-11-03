"""Orchestration boundaries with stubbed constructors; not physical/source acceptance."""

from hashlib import sha256
import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType

import pytest

from weft.moving_equipment_pipeline import prepare_moving_equipment_scene


@pytest.fixture
def workflow(tmp_path, monkeypatch):
    base, equipment, out = (tmp_path / name for name in ("base", "equipment", "out"))
    for path in (base, equipment):
        path.mkdir()
        (path / "manifest.json").write_text(json.dumps({"schema_version": 1, "files": {"source.json": "a" * 64}}))
    settings = tmp_path / "priors.json"
    settings.write_text('{"explicit":true}')
    bundle = {"status": "conditional_candidates", "candidates": [
        {"id": name, "status": "conditional_free_body_hypothesis", "reasons": []} for name in ("first", "second")], "reasons": []}
    calls, packet = [], object()

    def loaded(*args, **kwargs):
        calls.append(("load", args, kwargs))
        return packet

    def derive(inputs, priors, **kwargs):
        assert inputs is packet and priors == {"explicit": True}
        calls.append(("derive", kwargs))
        return bundle

    def unchanged(inputs, **kwargs):
        assert inputs is packet
        calls.append(("readback", kwargs))

    def build(inputs, proposals, candidate_id, destination, **kwargs):
        assert inputs is packet and proposals is bundle
        calls.append(("build", candidate_id, kwargs))
        return {"status": "ready", "lab_directory": str(destination), "manifest_sha256": "b" * 64,
                "scene_sha256": "c" * 64, "control_profile": "shared_world_exploration", "task_success_available": False}

    modules = {}
    for name, members in {
        "moving_equipment_inputs": {"load_moving_equipment_inputs": loaded, "verify_moving_equipment_inputs_unchanged": unchanged},
        "moving_equipment_proposal": {"validate_moving_settings": lambda value: value, "derive_moving_proposals": derive},
        "shared_world_exploration": {"prepare_shared_world_exploration": build, "MAX_LAB_BYTES": 256 << 20},
    }.items():
        module = ModuleType("weft." + name)
        for member, value in members.items():
            setattr(module, member, value)
        monkeypatch.setitem(sys.modules, module.__name__, module)
        modules[name] = module
    return base, equipment, settings, out, bundle, calls, modules


def run(workflow, **kwargs):
    return prepare_moving_equipment_scene(*workflow[:4], **kwargs)


def test_shared_inputs_complete_candidates_and_independent_receipt_seal(workflow):
    result = run(workflow)
    calls = workflow[5]
    assert [row[0] for row in calls] == ["load", "derive", "build", "build", "readback"]
    assert result["status"] == "candidates_ready" and result["selected_candidate"] is None
    assert not result["task_success_available"] and not result["training"]
    assert [row["id"] for row in result["candidates"]] == ["first", "second"]
    out = workflow[3]
    manifest = json.loads((out / "manifest.json").read_bytes())
    assert all(sha256((out / name).read_bytes()).hexdigest() == pin for name, pin in manifest["files"].items())
    assert json.loads((out / "moving-proposals.json").read_bytes()) == workflow[4]


def test_one_refused_world_does_not_drop_alternative_or_stop_next(workflow):
    module = workflow[6]["shared_world_exploration"]
    original = module.prepare_shared_world_exploration

    def build(inputs, bundle, candidate_id, out, **kwargs):
        if candidate_id == "first":
            raise ValueError("initial overlap")
        return original(inputs, bundle, candidate_id, out, **kwargs)

    module.prepare_shared_world_exploration = build
    result = run(workflow)
    assert result["status"] == "candidates_ready"
    assert [row["status"] for row in result["candidates"]] == ["failed", "ready"]
    assert result["candidates"][0]["reasons"] == [{"code": "ValueError", "message": "initial overlap"}]


def test_unsupported_candidates_are_preserved_without_native_construction(workflow):
    for row in workflow[4]["candidates"]:
        row.update(status="unsupported_input", reasons=["missing_initial_state"])
    result = run(workflow)
    assert result["status"] == "abstained" and len(result["candidates"]) == 2
    assert not any(row[0] == "build" for row in workflow[5])


@pytest.mark.parametrize("change", ["duplicate", "traversal", "overflow"])
def test_malformed_family_never_runs_native_constructor(workflow, change):
    candidates = workflow[4]["candidates"]
    if change == "duplicate":
        candidates[1]["id"] = candidates[0]["id"]
    elif change == "traversal":
        candidates[0]["id"] = "../escape"
    else:
        candidates.extend([dict(candidates[0], id="third"), dict(candidates[0], id="fourth")])
    result = run(workflow)
    assert result["status"] == "failed"
    assert not any(row[0] == "build" for row in workflow[5])


def test_source_change_never_publishes_overall_ready(workflow):
    def changed(*args, **kwargs):
        raise ValueError("source changed")
    workflow[6]["moving_equipment_inputs"].verify_moving_equipment_inputs_unchanged = changed
    result = run(workflow)
    assert result["status"] == "failed" and result["selected_candidate"] is None
    assert result["reasons"][0]["message"] == "source changed"
    assert len(result["candidates"]) == 2


def test_aggregate_deadline_preserves_unattempted_branches(workflow, monkeypatch):
    import weft.moving_equipment_pipeline as pipeline
    real_clock = pipeline.time.monotonic
    clock = [real_clock()]
    monkeypatch.setattr(pipeline.time, "monotonic", lambda: clock[0])
    module = workflow[6]["shared_world_exploration"]
    original = module.prepare_shared_world_exploration
    def build(*args, **kwargs):
        result = original(*args, **kwargs)
        clock[0] += 125
        return result
    module.prepare_shared_world_exploration = build
    result = run(workflow)
    assert result["status"] == "failed"
    assert [row["status"] for row in result["candidates"]] == ["ready", "deferred"]
    assert result["candidates"][1]["reasons"] == ["aggregate_time_bound"]


@pytest.mark.parametrize("raw", ['{"explicit":true,"explicit":false}', '{"explicit":NaN}'])
def test_invalid_settings_fail_before_output_or_source_load(workflow, raw):
    workflow[2].write_text(raw)
    with pytest.raises(ValueError):
        run(workflow)
    assert not workflow[3].exists() and not workflow[5]


def test_existing_output_not_overwritten(workflow):
    workflow[3].mkdir()
    marker = workflow[3] / "keep"
    marker.write_text("original")
    with pytest.raises(ValueError, match="new moving-equipment output"):
        run(workflow)
    assert marker.read_text() == "original" and not workflow[5]


@pytest.mark.parametrize("status,code", [("candidates_ready", None), ("abstained", 1), ("failed", 1)])
def test_cli_preserves_literal_paths_and_reports_nonready(monkeypatch, capsys, status, code):
    calls = []
    module = ModuleType("weft.moving_equipment_pipeline")
    def prepare(*args, **kwargs):
        calls.append((args, kwargs))
        kwargs["progress"]("inputs", "saved evidence")
        return {"status": status}
    module.prepare_moving_equipment_scene = prepare
    monkeypatch.setitem(sys.modules, module.__name__, module)
    spec = importlib.util.spec_from_file_location("_moving_equipment_cli", Path(__file__).parents[1] / "src/weft/cli.py")
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    monkeypatch.setattr(sys, "argv", ["weft", "moving-equipment-build", "--run", "body $(literal)",
        "--evidence-run", "tracks", "--settings", "prior settings.json", "--out", "new output", "--max-candidates", "2"])
    if code is None:
        cli.main()
    else:
        with pytest.raises(SystemExit) as error:
            cli.main()
        assert error.value.code == code
    assert calls[0][0] == ("body $(literal)", "tracks", "prior settings.json", "new output")
    assert calls[0][1]["max_candidates"] == 2
    output = capsys.readouterr()
    assert json.loads(output.out)["status"] == status
    assert "Validate saved video evidence" in output.err
