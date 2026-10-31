"""Camera learning routes preserve explicit task authority before native work."""

import json
import sys
from types import ModuleType

import pytest

from weft import cli


@pytest.fixture
def learner(monkeypatch):
    module = ModuleType("weft.camera_shared_learning")
    calls = []

    def run(*args, **kwargs):
        calls.append((args, kwargs))
        return {"status": "mocked_without_physics"}

    module.train_camera_shared_policy = run
    module.evaluate_camera_shared_policy = run
    monkeypatch.setitem(sys.modules, module.__name__, module)
    return calls


def test_training_cannot_override_camera_task_assistance(monkeypatch, learner, capsys):
    monkeypatch.setattr(sys, "argv", ["weft", "shared-exercise-train", "--run", "source", "--out", "new",
                                     "--policy-family", "camera-feedback", "--root-assistance"])
    with pytest.raises(SystemExit) as error:
        cli.main()
    assert error.value.code == 1 and not learner
    assert "preserves an existing task" in capsys.readouterr().err


def test_training_passes_exact_task_and_explicit_bounds(monkeypatch, learner):
    monkeypatch.setattr(sys, "argv", ["weft", "shared-exercise-train", "--run", "source", "--out", "new",
        "--policy-family", "camera-feedback", "--task", "exact-task.json", "--seed", "91",
        "--generations", "1", "--population", "4", "--max-seconds", "12"])
    cli.main()
    assert learner == [(("source", "new"), {"task_path": "exact-task.json", "seed": 91,
        "generations": 1, "population": 4, "max_seconds": 12.})]


def test_evaluation_routes_only_versioned_camera_policy(tmp_path, monkeypatch, learner):
    from weft import shared_exercise_policies

    monkeypatch.setattr(shared_exercise_policies, "evaluate_policy", lambda *args, **kwargs:
                        pytest.fail("Camera policy must not enter the legacy task validator"))
    (tmp_path / "policy.json").write_text(json.dumps({"profile": "camera-shared-policy-v1"}))
    monkeypatch.setattr(sys, "argv", ["weft", "shared-exercise-policy-evaluate", "--learning", str(tmp_path),
        "--out", "new", "--seed", "104729", "--trials", "2", "--jitter-scale", ".2", "--max-seconds", "18"])
    cli.main()
    assert learner == [((str(tmp_path), "new"), {"seed": 104729, "trials": 2,
                                               "jitter_scale": .2, "max_seconds": 18.})]


@pytest.mark.parametrize("raw", [b'{"profile":"camera-shared-policy-v1","profile":"other"}', b" " * (64 * 1024 + 1)])
def test_evaluation_rejects_ambiguous_or_oversize_policy_before_work(tmp_path, monkeypatch, learner, raw):
    (tmp_path / "policy.json").write_bytes(raw)
    monkeypatch.setattr(sys, "argv", ["weft", "shared-exercise-policy-evaluate", "--learning", str(tmp_path), "--out", "new"])
    with pytest.raises(SystemExit) as error:
        cli.main()
    assert error.value.code == 1 and not learner
