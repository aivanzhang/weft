import json
from pathlib import Path
import sys
from types import ModuleType

import pytest

from weft import cli


def test_saved_pose_recovery_is_explicit_and_never_claims_a_playable_world(monkeypatch, capsys):
    module = ModuleType("weft.pose_source_recovery")
    received = {}

    def prepare(run, out, **options):
        received.update(run=run, out=out, **options)
        return Path(out)

    module.prepare_pose_source_recovery = prepare
    monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setattr(sys, "argv", ["weft", "exercise-pose-recover", "--run", "saved-proposals",
        "--video", "original.webm", "--out", "recovered", "--max-seconds", "12"])
    cli.main()
    result = json.loads(capsys.readouterr().out)
    assert received == {"run": "saved-proposals", "out": "recovered", "source_video": "original.webm", "max_seconds": 12}
    assert result["status"] == "source_evidence_only"
    assert result["simulation_ready"] is False and result["new_inference"] is False
    assert "exercise-body --evidence-run" in result["next_step"]


@pytest.mark.parametrize("build_status", ["ready", "rejected_initial_geometry", "failed"])
def test_inspection_command_preserves_saved_build_status(monkeypatch, capsys, build_status):
    module = ModuleType("weft.exercise_build_report")
    received = {}
    saved = {"status": "report_generated", "build_status": build_status, "report_path": "inspection/index.html"}

    def prepare(run, out, **options):
        received.update(run=run, out=out, **options)
        return saved

    module.prepare_exercise_build_report = prepare
    monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setattr(sys, "argv", ["weft", "shared-exercise-inspect", "--run", "saved-build",
        "--out", "inspection", "--max-seconds", "18"])
    cli.main()
    assert received == {"run": "saved-build", "out": "inspection", "max_seconds": 18}
    assert json.loads(capsys.readouterr().out) == saved


def test_unverified_report_input_exits_without_claiming_an_inspection(monkeypatch, capsys):
    module = ModuleType("weft.exercise_build_report")

    def reject(*args, **kwargs):
        raise ValueError("Saved result differs from its manifest")

    module.prepare_exercise_build_report = reject
    monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setattr(sys, "argv", ["weft", "shared-exercise-inspect", "--run", "changed-build", "--out", "inspection"])
    with pytest.raises(SystemExit) as error:
        cli.main()
    output = capsys.readouterr()
    assert error.value.code == 1 and not output.out
    assert "differs from its manifest" in output.err
