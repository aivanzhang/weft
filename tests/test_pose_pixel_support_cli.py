"""Saved-image CLI routes never enable inference or modify report defaults."""

import json
import sys
from types import SimpleNamespace

import pytest

from weft import cli, exercise_build_report


@pytest.mark.parametrize("deadline", [None, "2.5"])
def test_pixel_command_forwards_only_explicit_saved_inputs(monkeypatch, capsys, deadline):
    calls = []
    def prepare(*args, **kwargs):
        calls.append((args, kwargs))
        return {"status": "complete", "slot_count": 36}
    monkeypatch.setitem(sys.modules, "weft.pose_pixel_support_package", SimpleNamespace(prepare_pose_pixel_support=prepare))
    arguments = ["weft", "shared-exercise-pixel-check", "--pose-run", "poses", "--extraction-run", "pixels", "--out", "new"]
    if deadline:
        arguments.extend(["--max-seconds", deadline])
    monkeypatch.setattr(sys, "argv", arguments)
    cli.main()
    assert calls == [(("poses", "pixels", "new"), {"max_seconds": 5 if deadline is None else 2.5})]
    assert json.loads(capsys.readouterr().out)["slot_count"] == 36


@pytest.mark.parametrize("companion", [None, "sealed-pixel-check"])
def test_report_cli_preserves_default_and_forwards_optional_companion(monkeypatch, capsys, companion):
    calls = []
    def prepare(*args, **kwargs):
        calls.append((args, kwargs))
        return {"status": "report_ready"}
    monkeypatch.setattr(exercise_build_report, "prepare_exercise_build_report", prepare)
    arguments = ["weft", "shared-exercise-inspect", "--run", "saved-build", "--out", "new-report"]
    if companion:
        arguments.extend(["--pixel-support", companion])
    monkeypatch.setattr(sys, "argv", arguments)
    cli.main()
    expected = {"max_seconds": 30}
    if companion:
        expected["pixel_support"] = companion
    assert calls == [(("saved-build", "new-report"), expected)]
    assert json.loads(capsys.readouterr().out)["status"] == "report_ready"
