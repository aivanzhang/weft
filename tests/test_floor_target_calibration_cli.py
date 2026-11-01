"""Calibration dispatch and honest CLI completion, without numerical work."""

from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType

import pytest


@pytest.fixture
def command(monkeypatch):
    def denied(*args, **kwargs): pytest.fail("CLI dispatched unrelated inference, physics or a world build")
    for name, members in {"compare": ("compare_runs", "save_comparison"), "env": ("RollingEnv",),
        "experiments": ("run_experiment",), "meta": ("status",), "pipeline": ("run", "verify_artifacts"),
        "source": ("fetch_sample",), "world": ("World",),
        "exercise_scene_pipeline": ("prepare_exercise_scene",), "camera_ground": ("apply_camera_ground",)}.items():
        module = ModuleType("weft." + name)
        for member in members: setattr(module, member, denied)
        monkeypatch.setitem(sys.modules, module.__name__, module)
    result = {"profile": "floor-target-calibration-package-v1", "status": "qualified_nominal", "sealed": True,
              "manifest_sha256": "b" * 64, "output_path": "saved calibration", "report_path": "saved calibration/inspection/index.html",
              "camera_path": "saved calibration/camera.json", "ground_path": "saved calibration/ground.json"}
    calls = []
    runner = ModuleType("weft.floor_target_calibration_pipeline")
    def prepare(*args, **kwargs):
        calls.append((args, kwargs))
        kwargs["progress"]("calibration", "Retain alternatives; no physical validation")
        return deepcopy(result)
    runner.prepare_floor_target_calibration_package = prepare
    monkeypatch.setitem(sys.modules, runner.__name__, runner)
    spec = importlib.util.spec_from_file_location("_floor_calibration_cli", Path(__file__).parents[1] / "src/weft/cli.py")
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    def invoke(args, name="floor-calibrate"):
        monkeypatch.setattr(sys, "argv", ["weft", name, *args]); return module.main()
    return invoke, calls, result


def arguments():
    return ["--observation", "source observations", "--source-sha256", "a" * 64,
            "--request", "explicit request", "--out", "new calibration"]


def test_explicit_inputs_forward_once_without_automatic_geometry_installation(command, capsys):
    invoke, calls, result = command
    invoke(arguments() + ["--observation", "more observations", "--measurement", "survey one",
        "--measurement", "survey two", "--camera", "original camera", "--ground", "original ground", "--max-seconds", "25"])
    args, options = calls[0]
    assert len(calls) == 1 and args == (["source observations", "more observations"], "new calibration")
    assert {k: v for k, v in options.items() if k != "progress"} == {"source_sha256": "a" * 64,
        "request": "explicit request", "measurements": ["survey one", "survey two"],
        "camera": "original camera", "ground": "original ground", "max_seconds": 25.}
    output = capsys.readouterr()
    assert json.loads(output.out) == result
    assert "Check camera and floor alternatives" in output.err
    assert "no body or floor has been changed" in output.err
    assert "Camera: saved calibration/camera.json" in output.err
    assert "Ground: saved calibration/ground.json" in output.err
    assert "shared-exercise-build" not in output.err


def test_missing_optional_inputs_stay_missing_for_core_abstention(command):
    invoke, calls, _ = command
    invoke(arguments())
    options = calls[0][1]
    assert options["measurements"] == [] and options["camera"] is options["ground"] is None
    assert options["max_seconds"] == 30.


@pytest.mark.parametrize("status", ["unavailable", "rejected", "ambiguous"])
def test_completed_nonqualification_keeps_report_but_returns_nonzero(command, capsys, status):
    invoke, calls, result = command
    result.update(status=status, camera_path=None, ground_path=None)
    with pytest.raises(SystemExit) as error: invoke(arguments())
    assert error.value.code == 1 and len(calls) == 1
    output = capsys.readouterr()
    assert status in output.err and result["report_path"] in output.err
    assert "Camera:" not in output.err


@pytest.mark.parametrize("patch", [{"status": "failed", "sealed": False}, {"manifest_sha256": "z" * 64},
    {"profile": "unrelated-profile"}, {"report_path": None}, {"camera_path": "another source/camera.json"},
    {"ground_path": None}])
def test_broken_completion_cannot_print_ready_output_pair(command, capsys, patch):
    invoke, _, result = command
    result.update(patch)
    with pytest.raises(SystemExit) as error: invoke(arguments())
    assert error.value.code == 1
    assert "Nominal geometry only" not in capsys.readouterr().err


@pytest.mark.parametrize("option", ["--observation", "--source-sha256", "--request", "--out"])
def test_missing_required_argument_stops_before_dispatch(command, option):
    invoke, calls, _ = command
    args = arguments(); index = args.index(option); del args[index:index+2]
    with pytest.raises(SystemExit) as error: invoke(args)
    assert error.value.code == 2 and not calls


def test_existing_surface_anchor_route_remains_distinct(command, monkeypatch, capsys):
    invoke, calls, _ = command
    surface_calls = []
    surface = ModuleType("weft.source_surface_pipeline")
    def prepare(*args, **kwargs):
        surface_calls.append((args, kwargs))
        return {"status": "completed", "sealed": True, "report_path": "existing surface report"}
    surface.prepare_source_surface_anchors = prepare
    monkeypatch.setitem(sys.modules, surface.__name__, surface)
    invoke(["--observation", "existing observations", "--source-sha256", "a" * 64,
            "--camera", "existing camera", "--surfaces", "existing surfaces", "--out", "new anchors"], "surface-anchors")
    assert not calls and len(surface_calls) == 1
    assert surface_calls[0][1]["surfaces"] == "existing surfaces"
    assert "existing surface report" in capsys.readouterr().err
