"""Explicit saved-stage routing cannot start acquisition, physics or support checks."""

import importlib.util
import json
from pathlib import Path
import shlex
import sys
from types import ModuleType

import pytest


@pytest.fixture
def command(monkeypatch):
    def denied(*args, **kwargs): pytest.fail("Evidence composition dispatched unrelated processing")
    for name, members in {"compare": ("compare_runs", "save_comparison"), "env": ("RollingEnv",),
        "experiments": ("run_experiment",), "meta": ("status",), "pipeline": ("run", "verify_artifacts"),
        "source": ("fetch_sample",), "world": ("World",),
        "equipment_support_workflow": ("prepare_equipment_support",)}.items():
        module = ModuleType("weft." + name)
        for member in members: setattr(module, member, denied)
        monkeypatch.setitem(sys.modules, module.__name__, module)
    runner = ModuleType("weft.exercise_equipment_evidence")
    calls = []
    result = {"profile": "pose-equipment-silhouette-composition-v1", "status": "completed", "sealed": True,
              "output_path": "saved comparison", "manifest_sha256": "a" * 64}
    def prepare(*args, **kwargs):
        calls.append((args, kwargs))
        for stage in ("inputs", "pose", "silhouette", "seal"):
            kwargs["progress"](stage, "saved inputs; contact remains unknown")
        return dict(result)
    runner.prepare_exercise_equipment_evidence = prepare
    monkeypatch.setitem(sys.modules, runner.__name__, runner)
    masks = ModuleType("weft.segmentation_masks")
    mask_calls = []
    def export(*args, **kwargs):
        mask_calls.append((args, kwargs)); return {"status": "completed", "stub": "no decoding executed"}
    masks.export_masks = export
    monkeypatch.setitem(sys.modules, masks.__name__, masks)
    spec = importlib.util.spec_from_file_location("_exercise_equipment_cli", Path(__file__).parents[1] / "src/weft/cli.py")
    cli = importlib.util.module_from_spec(spec); spec.loader.exec_module(cli)
    def invoke(args, route="exercise-equipment-evidence"):
        monkeypatch.setattr(sys, "argv", ["weft", route, *args]); return cli.main()
    return invoke, calls, result, runner, mask_calls


def required():
    return ["--pose-run", "pose source", "--equipment-run", "equipment", "--masks-run", "masks",
        "--human-id", "person-0", "--equipment-id", "sam-2", "--minimum-confidence", ".6",
        "--uncertainty-px", "2", "--uncertainty-method", "Authored test range",
        "--uncertainty-note", "Not measured accuracy", "--out", "saved comparison"]


def test_explicit_sources_assumptions_pins_and_quoted_next_command(command, capsys):
    invoke, calls, result, _, _ = command
    result["output_path"] = "saved comparison; literal $()"
    invoke(required() + ["--equipment-id", "sam-5", "--source-equivalence-run", "verified proof",
        "--pose-manifest-sha256", "b" * 64, "--equipment-manifest-sha256", "c" * 64,
        "--masks-manifest-sha256", "d" * 64, "--max-seconds", "25"])
    args, options = calls[0]
    assert args == ("pose source", "equipment", "masks", "saved comparison")
    assert {k: v for k, v in options.items() if k != "progress"} == {
        "human_id": "person-0", "equipment_ids": ["sam-2", "sam-5"], "minimum_confidence": .6,
        "uncertainty_px": 2., "uncertainty_method": "Authored test range", "uncertainty_note": "Not measured accuracy",
        "source_equivalence_run": "verified proof", "expected_pose_manifest_sha256": "b" * 64,
        "expected_equipment_manifest_sha256": "c" * 64, "expected_masks_manifest_sha256": "d" * 64,
        "max_seconds": 25.}
    output = capsys.readouterr()
    assert json.loads(output.out) == result
    assert "Qualify original pose estimates" in output.err
    assert "only if the source camera was fixed (not verified by this command)" in output.err
    followup = shlex.split(output.err.splitlines()[-1])
    assert followup == [sys.executable, "-m", "weft", "equipment-support", "--run", result["output_path"],
        "--manifest-sha256", "a" * 64, "--hand-points", "--fixed-camera", "--out", result["output_path"] + "-hand-support"]


@pytest.mark.parametrize("option", ["--pose-run", "--equipment-run", "--masks-run", "--human-id",
    "--equipment-id", "--minimum-confidence", "--uncertainty-px", "--uncertainty-method", "--uncertainty-note"])
def test_missing_identity_or_authored_setting_never_dispatches(command, option):
    invoke, calls, *_ = command
    args = required(); index = args.index(option); del args[index:index + 2]
    with pytest.raises(SystemExit) as error: invoke(args)
    assert error.value.code == 2 and not calls


@pytest.mark.parametrize("extra", [["--video", "new.mp4"], ["--execute"], ["--fixed-camera"], ["--pose-model", "model.task"]])
def test_saved_composition_does_not_accept_processing_or_camera_flags(command, extra):
    invoke, calls, *_ = command
    with pytest.raises(SystemExit) as error: invoke(required() + extra)
    assert error.value.code == 2 and not calls


@pytest.mark.parametrize("changes", [{"status": "failed"}, {"sealed": False}, {"sealed": 1},
    {"profile": "unrelated"}, {"manifest_sha256": "not-an-exact-seal"}, {"output_path": ""}])
def test_unsealed_or_incompatible_result_cannot_print_success_or_next_step(command, capsys, changes):
    invoke, _, result, _, _ = command
    result.update(changes)
    with pytest.raises(SystemExit) as error: invoke(required())
    assert error.value.code == 1
    stderr = capsys.readouterr().err
    assert "Saved pose/equipment image evidence:" not in stderr and "Next, only if" not in stderr


def test_preparation_error_preserves_failure_instead_of_starting_support(command, capsys):
    invoke, calls, _, runner, _ = command
    def failed(*args, **kwargs): raise TimeoutError("Prepared evidence remains in the failed run")
    runner.prepare_exercise_equipment_evidence = failed
    with pytest.raises(SystemExit) as error: invoke(required())
    assert error.value.code == 1 and not calls
    assert "Prepared evidence remains" in capsys.readouterr().err


@pytest.mark.parametrize("retain", [False, True])
def test_binary_retention_is_opt_in_and_old_mask_call_signature_stays_exact(command, retain):
    invoke, _, _, _, mask_calls = command
    args = ["--run", "saved tracking", "--video", "exact.mp4", "--out", "new masks"]
    invoke(args + (["--retain-binary"] if retain else []), route="tracking-masks")
    assert mask_calls == [(("saved tracking", "exact.mp4", "new masks"), {"retain_binary": True} if retain else {})]
