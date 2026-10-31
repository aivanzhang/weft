"""Public command routing without simulation, source reads or worker execution."""

import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType

import pytest


@pytest.fixture
def command(monkeypatch):
    imports = {
        "compare": ("compare_runs", "save_comparison"), "env": ("RollingEnv",),
        "experiments": ("run_experiment",), "meta": ("status",),
        "pipeline": ("run", "verify_artifacts"), "source": ("fetch_sample",), "world": ("World",),
    }

    def forbidden(*args, **kwargs):
        pytest.fail("Unrelated execution must not run from the trial command")

    for name, members in imports.items():
        module = ModuleType("weft." + name)
        for member in members:
            setattr(module, member, forbidden)
        monkeypatch.setitem(sys.modules, module.__name__, module)
    runner = ModuleType("weft.shared_exercise_reference_trial")
    monkeypatch.setitem(sys.modules, runner.__name__, runner)
    spec = importlib.util.spec_from_file_location("_reference_trial_cli", Path(__file__).parents[1] / "src/weft/cli.py")
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    calls = []

    def prepare(*args, **kwargs):
        calls.append((args, kwargs))
        kwargs["progress"]("record", {"controller": "reference_proposal", "control_steps": 30})
        return {"status": "completed", "task_success": False, "promoted": False}

    runner.prepare_reference_trial = prepare

    def invoke(extra=(), *, task=True):
        argv = ["weft", "body-reference-trial", "--proposal", "saved/proposal", "--run", "saved/world",
                "--out", "new/trial"]
        if task:
            argv += ["--task", "saved/original-task.json"]
        monkeypatch.setattr(sys, "argv", argv + list(extra))
        return cli.main()

    return invoke, calls, runner


def test_completed_delivery_keeps_failed_task_distinct(command, capsys):
    invoke, calls, _ = command
    assert invoke() is None
    args, options = calls[0]
    assert args == ("saved/proposal", "saved/world", "saved/original-task.json", "new/trial")
    assert options["seed"] == 1729 and options["max_seconds"] == 60
    assert all(options[key] is None for key in ("expected_proposal_manifest_sha256", "expected_scene_manifest_sha256", "expected_task_sha256"))
    captured = capsys.readouterr()
    assert json.loads(captured.out) == {"status": "completed", "task_success": False, "promoted": False}
    assert '"control_steps": 30' in captured.err and '"stage": "record"' in captured.err


def test_exact_explicit_pins_and_bounds_reach_runner(command):
    invoke, calls, _ = command
    invoke(["--seed", "43", "--max-seconds", "21.5", "--proposal-manifest-sha256", "a" * 64,
            "--scene-manifest-sha256", "b" * 64, "--task-sha256", "c" * 64])
    options = calls[0][1]
    assert (options["seed"], options["max_seconds"]) == (43, 21.5)
    assert options["expected_proposal_manifest_sha256"] == "a" * 64
    assert options["expected_scene_manifest_sha256"] == "b" * 64
    assert options["expected_task_sha256"] == "c" * 64


def test_original_task_is_required_before_execution(command, capsys):
    invoke, calls, _ = command
    with pytest.raises(SystemExit) as error:
        invoke(task=False)
    assert error.value.code == 2 and not calls
    assert "--task" in capsys.readouterr().err


@pytest.mark.parametrize("status", ["failed", "partial", "unsupported"])
def test_failed_delivery_is_nonzero_with_preserved_report(command, capsys, status):
    invoke, calls, runner = command
    runner.prepare_reference_trial = lambda *args, **kwargs: {"status": status}
    with pytest.raises(SystemExit) as error:
        invoke()
    assert error.value.code == 1 and not calls
    captured = capsys.readouterr()
    assert json.loads(captured.out) == {"status": status}
    assert "new/trial/report.json" in captured.err


@pytest.mark.parametrize("error_type", [ValueError, TimeoutError])
def test_preflight_error_has_no_success_output(command, capsys, error_type):
    invoke, calls, runner = command

    def fail(*args, **kwargs):
        raise error_type("fixed task identity differs")

    runner.prepare_reference_trial = fail
    with pytest.raises(SystemExit) as error:
        invoke()
    assert error.value.code == 1 and not calls
    captured = capsys.readouterr()
    assert captured.out == "" and "fixed task identity differs" in captured.err


def test_help_explains_fixed_original_task_without_execution(command, capsys):
    invoke, calls, _ = command
    with pytest.raises(SystemExit) as error:
        invoke(["--help"])
    assert error.value.code == 0 and not calls
    text = " ".join(capsys.readouterr().out.split())
    assert "never regenerated from the proposal" in text
    assert "record" in text and "Cooperative" in text


@pytest.mark.parametrize("status", ["completed", "failed"])
def test_prints_saved_page_without_opening_or_corrupting_json(command, capsys, status):
    invoke, calls, runner = command
    result = {"status": status, "inspection_path": "new/trial/index.html"}
    runner.prepare_reference_trial = lambda *args, **kwargs: result
    if status == "failed":
        with pytest.raises(SystemExit) as error:
            invoke()
        assert error.value.code == 1
    else:
        invoke()
    captured = capsys.readouterr()
    assert json.loads(captured.out) == result
    assert "Saved comparison page: new/trial/index.html" in captured.err
    assert not calls
