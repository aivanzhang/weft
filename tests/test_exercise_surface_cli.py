"""CLI routing cannot start fitting, simulation or source acquisition."""

import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType

import pytest


@pytest.fixture
def command(monkeypatch):
    def forbidden(*_args, **_kwargs):
        pytest.fail("Surface collection must not call unrelated execution or provider paths")

    for name, members in {
        "compare": ("compare_runs", "save_comparison"), "env": ("RollingEnv",),
        "experiments": ("run_experiment",), "meta": ("status",),
        "pipeline": ("run", "verify_artifacts"), "source": ("fetch_sample",), "world": ("World",),
    }.items():
        module = ModuleType("weft." + name)
        for member in members:
            setattr(module, member, forbidden)
        monkeypatch.setitem(sys.modules, module.__name__, module)
    runner = ModuleType("weft.exercise_surface_pipeline")
    monkeypatch.setitem(sys.modules, runner.__name__, runner)
    spec = importlib.util.spec_from_file_location("_exercise_surface_cli", Path(__file__).parents[1] / "src/weft/cli.py")
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    calls = []

    def prepare(*args, **kwargs):
        calls.append((args, kwargs))
        for stage in ("inputs", "selection", "qualification", "report"):
            kwargs["progress"](stage, {"scope": "mock routing only"})
        return {"status": "completed", "sealed": True, "collection_status": "needs_selection",
                "report_path": "new/collection/inspection/index.html", "simulation_ready": False}

    runner.prepare_exercise_surface_collection = prepare

    def invoke(options):
        monkeypatch.setattr(sys, "argv", ["weft", "surface-inputs", *options])
        return cli.main()

    return invoke, calls, runner


def test_discovery_preserves_inputs_and_emits_separate_progress(command, capsys):
    invoke, calls, _ = command
    assert invoke(["--observation", "first.json", "--observation", "second.json", "--out", "new/collection"]) is None
    args, kwargs = calls[0]
    assert args == (["first.json", "second.json"], "new/collection")
    assert {key: value for key, value in kwargs.items() if key != "progress"} == {
        "source_sha256": None, "camera": None, "body": None, "selection": None,
        "request": None, "resume": None, "expected_manifest_sha256": None, "max_seconds": 30.,
    }
    output = capsys.readouterr()
    assert json.loads(output.out)["collection_status"] == "needs_selection"
    assert "Read saved source evidence" in output.err and "Save input worksheet and report" in output.err
    assert "Saved surface-input report: new/collection/inspection/index.html" in output.err


def test_fresh_explicit_selection_preserves_options(command):
    invoke, calls, _ = command
    invoke(["--observation", "obs.json", "--out", "new/collection", "--source-sha256", "a" * 64,
            "--camera", "camera.json", "--body", "human.json", "--selection", "selection.json", "--max-seconds", "12"])
    options = calls[0][1]
    assert options["source_sha256"] == "a" * 64
    assert (options["camera"], options["body"], options["selection"], options["max_seconds"]) == (
        "camera.json", "human.json", "selection.json", 12.)


@pytest.mark.parametrize("edit", ["--selection", "--request"])
def test_resume_forwards_only_exact_saved_context_and_one_edit(command, edit):
    invoke, calls, _ = command
    invoke(["--resume", "old/collection", "--out", "new/collection", edit, "edited.json", "--manifest-sha256", "b" * 64])
    args, options = calls[0]
    assert args == ([], "new/collection") and options["resume"] == "old/collection"
    assert options[edit.removeprefix("--")] == "edited.json"
    assert options["expected_manifest_sha256"] == "b" * 64


@pytest.mark.parametrize("option,value", [("--source-sha256", "a" * 64), ("--camera", "camera.json"), ("--body", "human.json")])
def test_resume_rejects_new_raw_inputs_before_pipeline(command, capsys, option, value):
    invoke, calls, _ = command
    with pytest.raises(SystemExit) as error:
        invoke(["--resume", "old", "--out", "new", option, value])
    assert error.value.code == 1 and not calls
    assert "keeps exact saved inputs" in capsys.readouterr().err


@pytest.mark.parametrize("option,value", [("--request", "edited.json"), ("--manifest-sha256", "a" * 64)])
def test_fresh_source_rejects_resume_only_flags(command, capsys, option, value):
    invoke, calls, _ = command
    with pytest.raises(SystemExit) as error:
        invoke(["--observation", "obs.json", "--out", "new", option, value])
    assert error.value.code == 1 and not calls
    assert "require --resume" in capsys.readouterr().err


@pytest.mark.parametrize("options", [
    ["--out", "new"], ["--observation", "obs.json"],
    ["--observation", "obs.json", "--resume", "old", "--out", "new"],
    ["--resume", "old", "--selection", "s.json", "--request", "r.json", "--out", "new"],
    ["--observation", "obs.json", "--fit-seconds", "1", "--out", "new"],
])
def test_invalid_modes_never_dispatch(command, options):
    invoke, calls, _ = command
    with pytest.raises(SystemExit) as error:
        invoke(options)
    assert error.value.code == 2 and not calls


@pytest.mark.parametrize("state", ["incomplete_request", "qualified_but_unavailable", "conditional_inputs_available"])
def test_completed_unknown_or_conditional_results_are_not_failures(command, capsys, state):
    invoke, _, runner = command
    runner.prepare_exercise_surface_collection = lambda *a, **k: {
        "status": "completed", "sealed": True, "collection_status": state, "report_path": "new/inspection/index.html"}
    assert invoke(["--resume", "old", "--request", "edited.json", "--out", "new"]) is None
    assert json.loads(capsys.readouterr().out)["collection_status"] == state


@pytest.mark.parametrize("result", [
    {"status": "failed", "sealed": True}, {"status": "completed", "sealed": False},
    {"status": "completed", "sealed": 1}, {"status": "completed", "sealed": True},
])
def test_failed_unsealed_or_reportless_output_has_no_success_link(command, capsys, result):
    invoke, _, runner = command
    runner.prepare_exercise_surface_collection = lambda *a, **k: result
    with pytest.raises(SystemExit) as error:
        invoke(["--observation", "obs.json", "--out", "new"])
    assert error.value.code == 1
    output = capsys.readouterr()
    assert json.loads(output.out) == result and "Saved surface-input report" not in output.err


def test_preflight_failure_has_no_success_json(command, capsys):
    invoke, _, runner = command
    def fail(*args, **kwargs):
        raise ValueError("Exact source identity differs")
    runner.prepare_exercise_surface_collection = fail
    with pytest.raises(SystemExit) as error:
        invoke(["--observation", "obs.json", "--out", "new"])
    assert error.value.code == 1
    output = capsys.readouterr()
    assert output.out == "" and "Exact source identity differs" in output.err


def test_resume_without_an_edit_cannot_repeat_saved_work(command, capsys):
    invoke, calls, _ = command
    with pytest.raises(SystemExit) as error:
        invoke(["--resume", "old", "--out", "new"])
    assert error.value.code == 1 and not calls
    assert "saved results are not rerun" in capsys.readouterr().err


def test_help_keeps_assisted_collection_separate_from_fitting(command, capsys):
    invoke, calls, _ = command
    with pytest.raises(SystemExit) as error:
        invoke(["--help"])
    assert error.value.code == 0 and not calls
    help_text = " ".join(capsys.readouterr().out.split())
    assert "never fit or install a body" in help_text
    assert "no native model, fitting or media" in help_text
