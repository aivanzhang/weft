"""Versioned feasibility learning must reach independent evaluation and exact lab replay."""

import json

import pytest

from weft.io import digest
from weft.lab import Lab
from weft.pipeline import verify_artifacts
from weft.shared_exercise_lab import load_shared_exercise_lab, prepare_shared_exercise_lab
from weft.shared_exercise_policies import evaluate_policy
from weft.shared_exercise_residual_learning import train_shared_exercise_residual_policy
from weft.shared_exercise_task import SharedExerciseTaskEnv
from test_shared_exercise_run import source_fixture


@pytest.fixture(scope="module")
def feasibility_bundle(tmp_path_factory):
    from weft.shared_exercise_feasibility_learning import train_shared_exercise_feasibility_policy

    root = tmp_path_factory.mktemp("feasibility_integration")
    source, scene = source_fixture(root)
    original, refined = root / "original", root / "refined"
    train_shared_exercise_residual_policy(source, original, root_assistance=True,
        seed=7919, generations=2, population=8, max_seconds=30)
    train_shared_exercise_feasibility_policy(original, refined, max_seconds=30)
    return source, scene, original, refined


def test_new_policy_evaluates_matched_starts_and_records_exactly(tmp_path, monkeypatch, feasibility_bundle):
    source, scene, original, refined = feasibility_bundle
    from weft.shared_exercise_residual_learning import _seed_plan

    evaluation, lab_root = tmp_path / "evaluation", tmp_path / "lab"
    old_policy = json.loads((original / "policy.json").read_bytes())
    policy = json.loads((refined / "policy.json").read_bytes())
    settings = json.loads((refined / "settings.json").read_bytes())
    before = {path: digest(path) for directory in (source, original, refined)
              for path in directory.rglob("*") if path.is_file()}
    exclusions = _seed_plan(7920, 1, settings["training_seeds"])
    original_reset, resets = SharedExerciseTaskEnv.reset, []

    def frozen_reset(self, *, seed=None, options=None):
        freeze = json.loads((evaluation / "pre-evaluation-freeze.json").read_bytes())
        assert seed in freeze["evaluation_seeds"]
        assert freeze["policy_sha256"] == digest(refined / "policy.json")
        assert freeze["previous_reward_baseline"]["policy_sha256"] == digest(original / "policy.json")
        resets.append(seed)
        return original_reset(self, seed=seed, options=options)

    with monkeypatch.context() as context:
        context.setattr(SharedExerciseTaskEnv, "reset", frozen_reset)
        result = evaluate_policy(refined, evaluation, seed=7920, trials=2, excluded_seeds=exclusions, max_seconds=30)
    assert result["status"] == "evaluated" and len(resets) == 10
    assert set(result["evaluation_seeds"]).isdisjoint([*settings["training_seeds"], *exclusions])
    assert result["policy_profile"] == "shared-exercise-policy-v3"
    assert result["evaluation_feedback_to_optimizer"] is False
    comparison = json.loads((evaluation / "comparison.json").read_bytes())
    assert comparison["status"] == "evaluated" and comparison["evaluation_feedback_to_optimizer"] is False
    assert comparison["plan"]["training_algorithm"] == old_policy["algorithm"] != policy["algorithm"]
    assert comparison["plan"]["actor_algorithm"] == policy["actor_algorithm"]
    assert (evaluation / "previous-policy.json").read_bytes() == (original / "policy.json").read_bytes()
    assert result["aggregate_control_steps"] == result["control_steps"] + comparison["control_steps"]
    assert result["aggregate_control_steps"] <= result["aggregate_maximum_control_steps"]
    assert result["previous_reward_comparison_sha256"] == digest(evaluation / "comparison.json")
    assert all(row["initial_state_sha256"] == result["initial_states"][str(row["seed"])]
               for row in comparison["trials"])
    prepare_shared_exercise_lab(source, lab_root, root_assistance=True,
        policy_run=refined, evaluation_run=evaluation)
    loaded = load_shared_exercise_lab(lab_root / "run")
    assert loaded["policy"]["profile"] == "shared-exercise-policy-v3"
    lab = Lab(lab_root)
    try:
        assert lab.summary()["exercise_policy"]["evaluation_status"] == "evaluated"
        lab.control("/api/record/start", {})
        lab.control("/api/step", {"policy": True})
        lab.control("/api/reset", {"seed": 7921})
        lab.control("/api/step", {"policy": True, "steps": 2})
        lab.control("/api/record/stop", {})
        replay = lab.export_future.result(timeout=30)
        assert replay["action_replay"]["exact_state_replay"] is True
        assert replay["action_replay"]["learned_actions_verified"] == 3
        assert replay["action_replay"]["policy_sha256"] == digest(refined / "policy.json")
    finally:
        lab.close()
    assert all(digest(path) == sha for path, sha in before.items())
    assert verify_artifacts(evaluation)["manifest_sha256"] == result["manifest_sha256"]


def test_refine_cli_preserves_existing_training_input(tmp_path, monkeypatch, capsys):
    import sys
    from weft import cli, shared_exercise_feasibility_learning

    called = {}

    def train(learning, out, **kwargs):
        called.update(learning=learning, out=out, **kwargs)
        return {"status": "trained"}

    monkeypatch.setattr(shared_exercise_feasibility_learning, "train_shared_exercise_feasibility_policy", train)
    monkeypatch.setattr(sys, "argv", ["weft", "shared-exercise-refine", "--learning", "archive",
                                     "--out", str(tmp_path / "new"), "--max-seconds", "25"])
    cli.main()
    assert called == {"learning": "archive", "out": str(tmp_path / "new"), "max_seconds": 25.}
    assert json.loads(capsys.readouterr().out)["status"] == "trained"
