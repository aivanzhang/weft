"""Saved inspection dispatch never calls source acquisition or simulation."""

import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType

import pytest


@pytest.fixture
def command(monkeypatch):
    def forbidden(*_args, **_kwargs):
        pytest.fail("Equipment evidence inspection cannot dispatch acquisition or simulation")
    for name, members in {"compare": ("compare_runs", "save_comparison"), "env": ("RollingEnv",),
        "experiments": ("run_experiment",), "meta": ("status",), "pipeline": ("run", "verify_artifacts"),
        "source": ("fetch_sample",), "world": ("World",)}.items():
        module = ModuleType("weft." + name)
        for member in members: setattr(module, member, forbidden)
        monkeypatch.setitem(sys.modules, module.__name__, module)
    runner = ModuleType("weft.equipment_silhouette_inspection")
    calls = []
    def prepare(*args, **kwargs):
        calls.append((args, kwargs))
        for stage in ("inputs", "report", "seal"): kwargs["progress"](stage, "saved metadata only")
        return {"status": "completed", "sealed": True, "report_path": "new/index.html"}
    runner.prepare_equipment_silhouette_inspection = prepare
    monkeypatch.setitem(sys.modules, runner.__name__, runner)
    spec = importlib.util.spec_from_file_location("_equipment_evidence_cli", Path(__file__).parents[1] / "src/weft/cli.py")
    cli = importlib.util.module_from_spec(spec); spec.loader.exec_module(cli)
    def invoke(options):
        monkeypatch.setattr(sys, "argv", ["weft", "equipment-evidence-report", *options]); return cli.main()
    return invoke, calls, runner


def test_saved_report_command_keeps_exact_identity_and_separate_progress(command, capsys):
    invoke, calls, _ = command
    invoke(["--run", "study", "--out", "new", "--manifest-sha256", "a" * 64, "--max-seconds", "7"])
    args, options = calls[0]
    assert args == ("study", "new")
    assert {k: v for k, v in options.items() if k != "progress"} == {
        "expected_manifest_sha256": "a" * 64, "max_seconds": 7.}
    output = capsys.readouterr()
    assert json.loads(output.out)["status"] == "completed"
    assert "Read saved image evidence" in output.err and "Save report and uncertainty" in output.err
    assert "Saved equipment-evidence report: new/index.html" in output.err


@pytest.mark.parametrize("options", [["--out", "new"], ["--run", "old"],
    ["--run", "old", "--out", "new", "--video", "new.mp4"],
    ["--run", "old", "--out", "new", "--fit"]])
def test_missing_or_processing_options_never_dispatch(command, options):
    invoke, calls, _ = command
    with pytest.raises(SystemExit) as error: invoke(options)
    assert error.value.code == 2 and not calls


@pytest.mark.parametrize("result", [{"status": "failed", "sealed": True},
    {"status": "completed", "sealed": False}, {"status": "completed", "sealed": 1},
    {"status": "completed", "sealed": True}])
def test_failed_unsealed_or_reportless_result_is_not_a_success(command, capsys, result):
    invoke, _, runner = command
    runner.prepare_equipment_silhouette_inspection = lambda *a, **k: result
    with pytest.raises(SystemExit) as error: invoke(["--run", "old", "--out", "new"])
    assert error.value.code == 1
    assert "Saved equipment-evidence report:" not in capsys.readouterr().err
