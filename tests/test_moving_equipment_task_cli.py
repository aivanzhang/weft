"""CLI custody checks; actual task/source admission belongs to the child loader."""

import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType

import pytest


@pytest.fixture
def command(tmp_path, monkeypatch):
    calls = []
    module = ModuleType("weft.moving_equipment_task_lab")
    def prepare(*args, **kwargs):
        calls.append((args, kwargs))
        return {"status": "ready", "task_success": "not_evaluated"}
    module.prepare_moving_equipment_task_lab = prepare
    monkeypatch.setitem(sys.modules, module.__name__, module)
    spec = importlib.util.spec_from_file_location("_moving_task_cli", Path(__file__).parents[1] / "src/weft/cli.py")
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    task = tmp_path / "authored task.json"
    monkeypatch.setattr(sys, "argv", ["weft", "moving-equipment-task", "--lab", "parent $(literal)",
        "--parent-manifest-sha256", "a" * 64, "--task", str(task), "--out", "new child", "--max-seconds", "75"])
    return cli, task, calls


def test_explicit_task_and_parent_pin_reach_child_without_shell_expansion(command, capsys):
    cli, path, calls = command
    parameters = {"target_object_id": "raw-0", "target_position_world": [.2, .3, .4],
        "position_tolerance_world": .03, "speed_tolerance_world_s": .1, "hold_duration_s": .5}
    path.write_text(json.dumps(parameters))
    cli.main()
    assert calls == [(("parent $(literal)", parameters, "new child"),
                      {"parent_manifest_sha256": "a" * 64, "max_seconds": 75.})]
    assert json.loads(capsys.readouterr().out)["task_success"] == "not_evaluated"


@pytest.mark.parametrize("raw", ['{"x":1,"x":2}', '{"x":NaN}', '{"x":"' + 'x' * (16 << 10) + '"}'])
def test_ambiguous_or_oversized_task_refuses_before_child_creation(command, raw):
    cli, path, calls = command
    path.write_text(raw)
    with pytest.raises(SystemExit) as error:
        cli.main()
    assert error.value.code != 0 and calls == []


def test_training_passes_explicit_budget_without_installing_policy(command, monkeypatch, capsys):
    cli, _, _ = command
    calls = []
    module = ModuleType("weft.moving_equipment_learning")
    def train(*args, **kwargs):
        calls.append((args, kwargs))
        return {"status": "complete", "independent_changed_start_evaluation": "unavailable"}
    module.train_saved_moving_equipment_policy = train
    monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setattr(sys, "argv", ["weft", "moving-equipment-train", "--lab", "saved task", "--out", "new policy",
        "--seed", "42", "--generations", "1", "--population", "4", "--max-seconds", "60"])
    cli.main()
    assert calls == [(("saved task", "new policy"), {"seed": 42, "generations": 1, "population": 4, "max_seconds": 60.})]
    assert json.loads(capsys.readouterr().out)["independent_changed_start_evaluation"] == "unavailable"


def test_task_replay_keeps_trusted_contract_pin_and_time_bound(command, monkeypatch, capsys):
    cli, _, _ = command
    calls = []
    module = ModuleType("weft.moving_equipment_task_recording")
    def verify(*args, **kwargs):
        calls.append((args, kwargs))
        return {"exact_state_replay": True, "completed_interval": False}
    module.verify_task_replay = verify
    monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setattr(sys, "argv", ["weft", "moving-equipment-replay", "saved session", "--contract-sha256", "b" * 64, "--max-seconds", "45"])
    cli.main()
    assert calls == [(("saved session",), {"expected_contract_sha256": "b" * 64, "max_seconds": 45.})]
    assert json.loads(capsys.readouterr().out)["completed_interval"] is False


@pytest.mark.parametrize("reset_args,protocol,seeds", [
    (["--evaluation-seed", "4101", "--evaluation-seed", "4102"], None, [4101, 4102]),
    (["--protocol", "frozen protocol.json"], "frozen protocol.json", None),
])
def test_changed_start_evaluation_keeps_explicit_protocol_and_policy(command, monkeypatch, capsys, reset_args, protocol, seeds):
    cli, _, _ = command
    calls = []
    module = ModuleType("weft.moving_equipment_evaluation")
    def evaluate(*args, **kwargs):
        calls.append((args, kwargs))
        return {"status": "complete", "policy_successes": 0, "training_performed": False}
    module.evaluate_saved_changed_starts = evaluate
    monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setattr(sys, "argv", ["weft", "moving-equipment-evaluate", "--lab", "saved task",
        "--policy", "frozen policy.json", "--out", "new evaluation", "--max-seconds", "75", *reset_args])
    cli.main()
    assert calls == [(("saved task", protocol, "frozen policy.json", "new evaluation"),
                     {"max_seconds": 75., "evaluation_seeds": seeds, "position_jitter": .025, "velocity_jitter": .1})]
    assert json.loads(capsys.readouterr().out)["training_performed"] is False


@pytest.mark.parametrize("reset_args", [[], ["--protocol", "saved.json", "--evaluation-seed", "4101"]])
def test_changed_start_evaluation_requires_one_explicit_reset_source(command, monkeypatch, reset_args):
    cli, _, _ = command
    monkeypatch.setattr(sys, "argv", ["weft", "moving-equipment-evaluate", "--lab", "task",
        "--policy", "policy.json", "--out", "new evaluation", *reset_args])
    with pytest.raises(SystemExit) as error:
        cli.main()
    assert error.value.code == 2


def test_preflight_only_inspects_the_saved_lab_with_bounded_admission(command, monkeypatch, capsys):
    cli, _, _ = command
    calls = []
    module = ModuleType("weft.moving_equipment_compatibility_lab")
    def inspect(*args, **kwargs):
        calls.append((args, kwargs))
        return {"status": "assessed", "blocks_manual_exploration": False}
    module.inspect_saved_compatibility = inspect
    monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setattr(sys, "argv", ["weft", "moving-equipment-preflight", "--lab", "saved task", "--max-seconds", "20"])
    cli.main()
    assert calls == [(("saved task",), {"max_seconds": 20.})]
    assert json.loads(capsys.readouterr().out)["blocks_manual_exploration"] is False
