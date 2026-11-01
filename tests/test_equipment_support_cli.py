"""The saved support command requires a camera assumption and never starts physics."""

import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType

import pytest


@pytest.fixture
def command(monkeypatch):
    def denied(*args, **kwargs): pytest.fail("Saved support command dispatched acquisition or simulation")
    for name, members in {"compare": ("compare_runs", "save_comparison"), "env": ("RollingEnv",),
        "experiments": ("run_experiment",), "meta": ("status",), "pipeline": ("run", "verify_artifacts"),
        "source": ("fetch_sample",), "world": ("World",)}.items():
        module = ModuleType("weft." + name)
        for member in members: setattr(module, member, denied)
        monkeypatch.setitem(sys.modules, module.__name__, module)
    runner = ModuleType("weft.equipment_support_workflow")
    calls = []
    def prepare(*args, **kwargs):
        calls.append((args, kwargs))
        for stage in ("inputs", "selection", "report", "seal"):
            kwargs["progress"](stage, "saved diagnostics, unknown physical contact")
        return {"status": "completed", "sealed": True, "report_path": "comparison/index.html"}
    runner.prepare_equipment_support = prepare
    monkeypatch.setitem(sys.modules, runner.__name__, runner)
    spec = importlib.util.spec_from_file_location("_equipment_support_cli", Path(__file__).parents[1] / "src/weft/cli.py")
    cli = importlib.util.module_from_spec(spec); spec.loader.exec_module(cli)
    def invoke(args):
        monkeypatch.setattr(sys, "argv", ["weft", "equipment-support", *args]); return cli.main()
    return invoke, calls, runner


def test_exact_input_pins_assumption_and_separate_progress(command, capsys):
    invoke, calls, _ = command
    invoke(["--run", "saved", "--out", "comparison", "--fixed-camera", "--manifest-sha256", "a" * 64,
            "--max-seconds", "19"])
    args, options = calls[0]
    assert args == ("saved", "comparison")
    assert {key: value for key, value in options.items() if key != "progress"} == {
        "fixed_camera": True, "expected_manifest_sha256": "a" * 64, "max_seconds": 19.}
    output = capsys.readouterr()
    assert json.loads(output.out)["sealed"] is True
    assert "Check original equipment candidates" in output.err
    assert "Saved equipment-support comparison: comparison/index.html" in output.err


@pytest.mark.parametrize("args", [["--run", "saved", "--out", "new"], ["--out", "new", "--fixed-camera"],
    ["--run", "saved", "--fixed-camera"], ["--run", "saved", "--out", "new", "--fixed-camera", "--fit"],
    ["--run", "saved", "--out", "new", "--fixed-camera", "--video", "new.mp4"]])
def test_missing_assumption_or_processing_request_never_dispatches(command, args):
    invoke, calls, _ = command
    with pytest.raises(SystemExit) as error: invoke(args)
    assert error.value.code == 2 and not calls


@pytest.mark.parametrize("result", [{"status": "failed", "sealed": True}, {"status": "completed", "sealed": False},
    {"status": "completed", "sealed": 1}, {"status": "completed", "sealed": True}])
def test_incomplete_result_is_not_reported_as_success(command, capsys, result):
    invoke, _, runner = command
    runner.prepare_equipment_support = lambda *a, **k: result
    with pytest.raises(SystemExit) as error: invoke(["--run", "saved", "--out", "new", "--fixed-camera"])
    assert error.value.code == 1
    assert "Saved equipment-support comparison:" not in capsys.readouterr().err
