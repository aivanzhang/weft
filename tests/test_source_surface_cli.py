"""Public command orchestration only; reconstruction, models and physics are mocked."""

import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType

import pytest


SOURCE = "a" * 64


@pytest.fixture
def command(monkeypatch):
    def forbidden(*_args, **_kwargs):
        pytest.fail("Surface-anchor CLI must not invoke another pipeline, provider or simulator")

    for name, members in {
        "compare": ("compare_runs", "save_comparison"), "env": ("RollingEnv",),
        "experiments": ("run_experiment",), "meta": ("status",),
        "pipeline": ("run", "verify_artifacts"), "source": ("fetch_sample",), "world": ("World",),
    }.items():
        module = ModuleType("weft." + name)
        for member in members:
            setattr(module, member, forbidden)
        monkeypatch.setitem(sys.modules, module.__name__, module)
    runner = ModuleType("weft.source_surface_pipeline")
    monkeypatch.setitem(sys.modules, runner.__name__, runner)
    spec = importlib.util.spec_from_file_location("_surface_anchor_cli", Path(__file__).parents[1] / "src/weft/cli.py")
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    calls = []

    def prepare(*args, **kwargs):
        calls.append((args, kwargs))
        progress = kwargs["progress"]
        for stage in ("inputs", "anchors", "report"):
            progress(stage, {"scope": "mock orchestration only"})
        return {"status": "completed", "diagnostic_status": "conditional_anchors_available", "sealed": True,
                "simulation_ready": False, "body_or_floor_installed": False, "physical_validation": False,
                "report_path": "new/anchors/index.html"}

    runner.prepare_source_surface_anchors = prepare

    def invoke(extra=(), *, omit=None):
        argv = ["weft", "surface-anchors"]
        for option, value in (("--observation", "observations.json"), ("--source-sha256", SOURCE), ("--out", "new/anchors")):
            if option != omit:
                argv += [option, value]
        monkeypatch.setattr(sys, "argv", argv + list(extra))
        return cli.main()

    return invoke, calls, runner


def test_public_defaults_forward_exact_paths_and_cooperative_deadline(command, capsys):
    invoke, calls, _ = command
    assert invoke() is None
    args, kwargs = calls[0]
    assert args == (["observations.json"], "new/anchors")
    assert {key: value for key, value in kwargs.items() if key != "progress"} == {
        "source_sha256": SOURCE, "camera": None, "surfaces": None, "annotations": None, "max_seconds": 30.,
    }
    assert callable(kwargs["progress"])
    captured = capsys.readouterr()
    result = json.loads(captured.out)
    assert result["sealed"] is True and result["simulation_ready"] is False
    assert result["body_or_floor_installed"] is result["physical_validation"] is False
    assert all(label in captured.err for label in ("Preserve source and assumptions", "Resolve conditional 3D points",
                                                  "Save evidence and limits", "Saved conditional-coordinate report: new/anchors/index.html"))


def test_repeated_observations_and_optional_inputs_preserve_order(command):
    invoke, calls, _ = command
    invoke(["--observation", "second.json", "--camera", "camera.json", "--surfaces", "surfaces.json",
            "--annotations", "annotations.json", "--max-seconds", "2.5"])
    args, kwargs = calls[0]
    assert args == (["observations.json", "second.json"], "new/anchors")
    assert {key: value for key, value in kwargs.items() if key != "progress"} == {
        "source_sha256": SOURCE, "camera": "camera.json", "surfaces": "surfaces.json", "annotations": "annotations.json", "max_seconds": 2.5,
    }


def test_unavailable_but_sealed_diagnostic_is_a_successful_command(command, capsys):
    invoke, _, runner = command
    runner.prepare_source_surface_anchors = lambda *a, **k: {
        "status": "completed", "diagnostic_status": "unavailable", "sealed": True,
        "report_path": "new/anchors/index.html", "simulation_ready": False,
    }
    assert invoke() is None
    captured = capsys.readouterr()
    assert json.loads(captured.out)["diagnostic_status"] == "unavailable"
    assert "Saved conditional-coordinate report" in captured.err


@pytest.mark.parametrize("result", [
    {"status": "failed", "sealed": True}, {"status": "failed", "sealed": False},
    {"status": "completed", "sealed": False}, {"status": "completed"},
    {"status": "completed", "sealed": 1},
])
def test_failed_or_unsealed_receipt_exits_nonzero_without_success_link(command, capsys, result):
    invoke, _, runner = command
    runner.prepare_source_surface_anchors = lambda *a, **k: result
    with pytest.raises(SystemExit) as error:
        invoke()
    assert error.value.code == 1
    captured = capsys.readouterr()
    assert json.loads(captured.out) == result
    assert "preserved output: new/anchors" in captured.err
    assert "Saved conditional-coordinate report" not in captured.err


@pytest.mark.parametrize("option", ["--observation", "--source-sha256", "--out"])
def test_required_input_is_rejected_before_dispatch(command, capsys, option):
    invoke, calls, _ = command
    with pytest.raises(SystemExit) as error:
        invoke(omit=option)
    assert error.value.code == 2 and calls == []
    assert option in capsys.readouterr().err


@pytest.mark.parametrize("kind", [ValueError, TimeoutError, OSError])
def test_pipeline_preflight_exception_has_no_success_json_or_link(command, capsys, kind):
    invoke, _, runner = command

    def fail(*_args, **_kwargs):
        raise kind("Selected input unavailable")

    runner.prepare_source_surface_anchors = fail
    with pytest.raises(SystemExit) as error:
        invoke()
    assert error.value.code == 1
    captured = capsys.readouterr()
    assert captured.out == "" and "Selected input unavailable" in captured.err
    assert "Saved conditional-coordinate report" not in captured.err


def test_document_count_refusal_from_pipeline_is_not_hidden(command, capsys):
    invoke, _, runner = command
    seen = []

    def bounded(paths, *_args, **_kwargs):
        seen.extend(paths)
        raise ValueError("supply one to eight observation JSON paths")

    runner.prepare_source_surface_anchors = bounded
    extra = [item for index in range(8) for item in ("--observation", f"extra-{index}.json")]
    with pytest.raises(SystemExit) as error:
        invoke(extra)
    assert error.value.code == 1 and len(seen) == 9
    assert "one to eight" in capsys.readouterr().err


@pytest.mark.parametrize("extra", [["--max-seconds", "not-a-number"], ["--fit-seconds", "1"]])
def test_invalid_or_unrelated_options_never_dispatch(command, capsys, extra):
    invoke, calls, _ = command
    with pytest.raises(SystemExit) as error:
        invoke(extra)
    assert error.value.code == 2 and not calls
    assert capsys.readouterr().out == ""


def test_help_discloses_cooperative_limits_without_running_pipeline(command, capsys):
    invoke, calls, _ = command
    with pytest.raises(SystemExit) as error:
        invoke(["--help"])
    assert error.value.code == 0 and calls == []
    help_text = capsys.readouterr().out
    assert "Cooperative packaging deadline" in help_text
    assert "not a hard process" in help_text and "source-pinned camera" in help_text
