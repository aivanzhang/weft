"""Guided setup only prints explicit next steps; it never starts a world build."""

from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import shlex
import sys
from types import ModuleType

import pytest


@pytest.fixture
def command(monkeypatch):
    def denied(*args, **kwargs): pytest.fail("Setup dispatched inference, a build or another workflow")
    for name, members in {"compare": ("compare_runs", "save_comparison"), "env": ("RollingEnv",),
        "experiments": ("run_experiment",), "meta": ("status",), "pipeline": ("run", "verify_artifacts"),
        "source": ("fetch_sample",), "world": ("World",),
        "exercise_scene_pipeline": ("prepare_exercise_scene",),
        "exercise_body_source": ("prepare_exercise_body_source",)}.items():
        module = ModuleType("weft." + name)
        for member in members: setattr(module, member, denied)
        monkeypatch.setitem(sys.modules, module.__name__, module)
    runner = ModuleType("weft.shared_exercise_config")
    calls = []
    result = {"profile": "shared-exercise-config-package-v1", "status": "needs_input", "sealed": True,
        "output_path": "saved setup; literal $()", "report_path": "saved setup/index.html", "manifest_sha256": "a" * 64,
        "config_path": None, "build_inputs": None, "build_options": {"body_scale": 1., "root_assistance": False},
        "full_reference": True}
    def prepare(*args, **kwargs):
        calls.append((args, kwargs))
        kwargs["progress"]("availability", "Complete original source slots; no world admission")
        return deepcopy(result)
    runner.prepare_shared_exercise_config = prepare
    monkeypatch.setitem(sys.modules, runner.__name__, runner)
    spec = importlib.util.spec_from_file_location("_shared_exercise_config_cli", Path(__file__).parents[1] / "src/weft/cli.py")
    cli = importlib.util.module_from_spec(spec); spec.loader.exec_module(cli)
    def invoke(args):
        monkeypatch.setattr(sys, "argv", ["weft", "shared-exercise-config", *args]); return cli.main()
    return invoke, calls, result, runner


def discovery():
    return ["--human-run", "original body", "--evidence-run", "original equipment", "--out", "new setup"]


def ready(result, *, assistance=False):
    result.update(status="config_ready", config_path="saved setup/config.json",
        build_inputs={"human_run": "saved setup/sources/human", "evidence_run": "saved setup/sources/equipment",
                      "source_mappings": ["saved setup/sources/mapping ' literal"], "config": "saved setup/config.json"},
        build_options={"body_scale": 1.25, "root_assistance": assistance})


def test_discovery_preserves_explicit_sources_and_prints_pinned_resume_not_build(command, capsys):
    invoke, calls, result, _ = command
    invoke(discovery() + ["--source-mapping", "first proof", "--source-mapping", "second proof", "--max-seconds", "29"])
    args, options = calls[0]
    assert args == ("original body", "original equipment", "new setup")
    assert {key: value for key, value in options.items() if key != "progress"} == {
        "source_mappings": ["first proof", "second proof"], "resume": None, "request": None,
        "expected_manifest_sha256": None, "max_seconds": 29.}
    output = capsys.readouterr()
    assert json.loads(output.out) == result
    assert "Explain usable and missing frames" in output.err and "Do not edit the sealed package" in output.err
    assert shlex.split(output.err.splitlines()[-1]) == [sys.executable, "-m", "weft", "shared-exercise-config",
        "--resume", result["output_path"], "--manifest-sha256", "a" * 64,
        "--request", "PATH_TO_DOWNLOADED_REQUEST.json", "--out", result["output_path"] + "-configured"]
    assert "shared-exercise-build" not in output.err


def test_resume_reuses_only_copied_sources_and_exact_edited_request(command):
    invoke, calls, *_ = command
    invoke(["--resume", "sealed setup", "--request", "downloads/edited request.json", "--manifest-sha256", "b" * 64,
            "--out", "configured setup"])
    args, options = calls[0]
    assert args == (None, None, "configured setup")
    assert {key: value for key, value in options.items() if key != "progress"} == {
        "source_mappings": [], "resume": "sealed setup", "request": "downloads/edited request.json",
        "expected_manifest_sha256": "b" * 64, "max_seconds": 30.}


@pytest.mark.parametrize("assistance", [False, True])
def test_ready_result_only_prints_quoted_existing_builder_with_explicit_choices(command, capsys, assistance):
    invoke, _, result, _ = command
    ready(result, assistance=assistance)
    invoke(discovery())
    output = capsys.readouterr()
    assert "Configuration ready, not a validated world" in output.err
    expected = [sys.executable, "-m", "weft", "shared-exercise-build", "--human-run", result["build_inputs"]["human_run"],
        "--evidence-run", result["build_inputs"]["evidence_run"], "--config", result["config_path"],
        "--body-scale", "1.25", "--full-reference", "--out", result["output_path"] + "-build",
        "--source-mapping", result["build_inputs"]["source_mappings"][0]]
    assert shlex.split(output.err.splitlines()[-1]) == expected + (["--root-assistance"] if assistance else [])


@pytest.mark.parametrize("args", [
    ["--human-run", "body", "--out", "new"],
    discovery() + ["--request", "edited.json"],
    discovery() + ["--manifest-sha256", "a" * 64],
    ["--resume", "old", "--out", "new"],
    ["--resume", "old", "--request", "edited.json", "--evidence-run", "replacement", "--out", "new"],
    ["--resume", "old", "--request", "edited.json", "--source-mapping", "replacement", "--out", "new"],
])
def test_incompatible_modes_cannot_read_sources_or_dispatch(command, args):
    invoke, calls, *_ = command
    with pytest.raises(SystemExit) as error: invoke(args)
    assert error.value.code == 1 and not calls


@pytest.mark.parametrize("extra", [["--video", "fresh.mp4", "--execute"], ["--resume", "different setup"]])
def test_setup_cannot_start_new_video_processing_or_replace_discovery_mode(command, extra):
    invoke, calls, *_ = command
    with pytest.raises(SystemExit) as error: invoke(discovery() + extra)
    assert error.value.code == 2 and not calls


@pytest.mark.parametrize("status", ["unavailable", "failed"])
def test_unsupported_input_keeps_report_but_never_prints_a_build(command, capsys, status):
    invoke, _, result, _ = command
    result["status"] = status
    with pytest.raises(SystemExit) as error: invoke(discovery())
    assert error.value.code == 1
    output = capsys.readouterr()
    assert json.loads(output.out)["status"] == status
    assert "open preserved report:" in output.err and "shared-exercise-build" not in output.err


@pytest.mark.parametrize("mutation", [
    lambda result: result.update(full_reference=False),
    lambda result: result["build_inputs"].update(source_mappings="not-a-list"),
    lambda result: result["build_options"].update(root_assistance=1),
    lambda result: result["build_options"].update(body_scale=True),
    lambda result: result.update(config_path="different.json"),
])
def test_incomplete_ready_contract_never_advertises_an_executable_build(command, capsys, mutation):
    invoke, _, result, _ = command
    ready(result); mutation(result)
    with pytest.raises(SystemExit) as error: invoke(discovery())
    assert error.value.code == 1
    output = capsys.readouterr()
    assert "ready setup lacks exact build inputs/options" in output.err
    assert "Configuration ready, not a validated world" not in output.err


def test_unsealed_result_or_caller_failure_never_prints_success(command, capsys):
    invoke, calls, result, runner = command
    result["sealed"] = False
    with pytest.raises(SystemExit) as error: invoke(discovery())
    assert error.value.code == 1 and "Saved setup page" not in capsys.readouterr().err
    def failed(*args, **kwargs): raise TimeoutError("Partial setup retained")
    runner.prepare_shared_exercise_config = failed
    with pytest.raises(SystemExit) as error: invoke(discovery())
    assert error.value.code == 1 and len(calls) == 1
    assert "Partial setup retained" in capsys.readouterr().err
