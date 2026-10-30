"""Acceptance must not mistake a closed recording prefix for a completed policy trial."""

import importlib.util
from pathlib import Path

import pytest


path = Path(__file__).resolve().parents[1] / "scripts/studies/check_shared_exercise_policy_lab.py"
spec = importlib.util.spec_from_file_location("exercise_policy_study", path)
study = importlib.util.module_from_spec(spec)
spec.loader.exec_module(study)


def baseline(steps, *, terminal=False, completed=False, failure=None):
    return {"scene": {"exercise_task": {"interval": {"horizon_steps": 3}}},
            "events": [{"type": "start"}] + [
                {"type": "step", "terminated": terminal and index == steps - 1, "truncated": False,
                 "snapshot": {"task_metrics": {"completed_interval": completed, "failure": failure}}}
                for index in range(steps)]}


@pytest.mark.parametrize("recording", [baseline(0), baseline(2), baseline(2, terminal=True),
                                        baseline(2, terminal=True, completed=True)])
def test_nonterminal_or_truncated_baseline_prefix_cannot_pass(recording):
    with pytest.raises(ValueError, match="baseline"):
        study.validate_baseline_episode(recording)


@pytest.mark.parametrize("recording", [baseline(3, terminal=True, completed=True),
                                        baseline(1, terminal=True, failure="human_velocity_envelope_exceeded")])
def test_complete_interval_or_explicit_early_failure_is_preserved(recording):
    study.validate_baseline_episode(recording)
