"""The construction command never implicitly simulates or starts a server."""

import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType

import pytest


@pytest.fixture
def command(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Construction must not invoke unrelated processing")

    for name, members in {
        "compare": ("compare_runs", "save_comparison"), "env": ("RollingEnv",),
        "experiments": ("run_experiment",), "meta": ("status",),
        "pipeline": ("run", "verify_artifacts"), "source": ("fetch_sample",), "world": ("World",),
    }.items():
        module = ModuleType("weft." + name)
        for member in members:
            setattr(module, member, forbidden)
        monkeypatch.setitem(sys.modules, module.__name__, module)
    runner = ModuleType("weft.spatial_scene_pipeline")
    monkeypatch.setitem(sys.modules, runner.__name__, runner)
    spec = importlib.util.spec_from_file_location("_spatial_scene_cli", Path(__file__).parents[1] / "src/weft/cli.py")
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    calls = []

    def prepare(*args, **kwargs):
        calls.append((args, kwargs))
        return {"status": "constructed_hypothesis", "simulation_ready": False, "physical_validation": "unverified"}

    runner.prepare_spatial_scene = prepare

    def invoke(extra=(), *, config=True):
        argv = ["weft", "spatial-scene", "--run", "saved/planar", "--out", "new/spatial"]
        if config:
            argv += ["--config", "authored.json"]
        monkeypatch.setattr(sys, "argv", argv + list(extra))
        return cli.main()

    return invoke, calls, runner


def test_default_route_keeps_construction_separate_from_admission(command, capsys):
    invoke, calls, _ = command
    assert invoke() is None
    assert calls == [(("saved/planar", "authored.json", "new/spatial"), {
        "scene_artifact": "scene.json", "expected_manifest_sha256": None, "max_seconds": 20})]
    result = json.loads(capsys.readouterr().out)
    assert result["simulation_ready"] is False and result["physical_validation"] == "unverified"


def test_exact_selection_and_limits_are_forwarded(command):
    invoke, calls, _ = command
    invoke(["--scene-artifact", "selected/scene.json", "--manifest-sha256", "a" * 64, "--max-seconds", "8.5"])
    assert calls[0][1] == {"scene_artifact": "selected/scene.json", "expected_manifest_sha256": "a" * 64, "max_seconds": 8.5}


def test_config_is_required_before_construction(command, capsys):
    invoke, calls, _ = command
    with pytest.raises(SystemExit) as error:
        invoke(config=False)
    assert error.value.code == 2 and not calls
    assert "--config" in capsys.readouterr().err


@pytest.mark.parametrize("status", ["failed", "unsupported"])
def test_incomplete_result_is_nonzero(command, capsys, status):
    invoke, calls, runner = command
    runner.prepare_spatial_scene = lambda *args, **kwargs: {"status": status}
    with pytest.raises(SystemExit) as error:
        invoke()
    assert error.value.code == 1 and not calls
    captured = capsys.readouterr()
    assert json.loads(captured.out)["status"] == status and "new/spatial" in captured.err


@pytest.mark.parametrize("error_type", [ValueError, TimeoutError])
def test_failed_preflight_has_no_success_output(command, capsys, error_type):
    invoke, calls, runner = command

    def fail(*args, **kwargs):
        raise error_type("Source identity differs")

    runner.prepare_spatial_scene = fail
    with pytest.raises(SystemExit) as error:
        invoke()
    assert error.value.code == 1 and not calls
    captured = capsys.readouterr()
    assert captured.out == "" and "Source identity differs" in captured.err
