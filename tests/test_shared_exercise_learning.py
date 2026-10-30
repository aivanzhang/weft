"""Learning cannot alter task gates, add object controls or tune on changed-start evaluation."""

import importlib.util
import json
import os
from pathlib import Path

import numpy as np
import pytest

from weft.io import canonical_hash, digest
from weft.pipeline import verify_artifacts
from weft.shared_exercise_task import SharedExerciseTaskEnv, build_shared_exercise_task
from weft.shared_exercise_learning import (
    _policy, _runtime_hashes, evaluate_shared_exercise_policy, prepare_shared_exercise_policy,
    train_shared_exercise_policy, validate_shared_exercise_policy,
)
from test_shared_exercise_run import source_fixture


def policy_fixture(tmp_path, root_assistance=False):
    _, scene = source_fixture(tmp_path)
    task = build_shared_exercise_task(scene, root_assistance=root_assistance)
    env = SharedExerciseTaskEnv(scene, task)
    env.reset(seed=19)
    weights = np.ones((4 if root_assistance else 3, 3))
    policy = _policy(scene, task, env, weights, 1729, .5, _runtime_hashes())
    return scene, task, env, policy


@pytest.mark.parametrize("root_assistance", [False, True])
def test_prepared_policy_is_human_only_detached_and_reuses_existing_feedback(tmp_path, monkeypatch, root_assistance):
    scene, task, env, policy = policy_fixture(tmp_path, root_assistance)
    try:
        action = prepare_shared_exercise_policy(policy, env)
        expected = env.authored_action()
        policy["weights"][0][0] = -3
        policy["action_names"].append("object_force")
        scene["task"]["horizon_steps"] = 1
        task["control"]["root_assistance"] = "invalid"

        def no_hot_validation():
            raise AssertionError("Prepared actions must not hash files per step")

        monkeypatch.setattr("weft.shared_exercise_learning._runtime_hashes", no_hot_validation)
        np.testing.assert_array_equal(action(), expected)
        first = action()
        first[:] = 0
        np.testing.assert_array_equal(action(), expected)
        env.step(action())
        np.testing.assert_array_equal(action(), env.authored_action())
        assert len(action()) == (22 if root_assistance else 19)
        assert not any(env.snapshot()["engine_state"]["normalized_action"][22:])
    finally:
        env.close()


@pytest.mark.parametrize("field,value", [
    ("scene_sha256", "0" * 64), ("task_sha256", "0" * 64), ("evaluator_sha256", "0" * 64),
    ("gates_sha256", "0" * 64), ("runtime_sha256", {}), ("root_assistance", "explicit_3_axis"),
    ("action_names", ["object_force"]), ("weights", [[True] * 3] * 3),
    ("weights", [[4.] * 3] * 3), ("weights", [[float("nan")] * 3] * 3),
])
def test_policy_rejects_identity_gate_runtime_and_action_tampering(tmp_path, field, value):
    scene, task, env, policy = policy_fixture(tmp_path)
    try:
        policy[field] = value
        with pytest.raises(ValueError):
            validate_shared_exercise_policy(policy, scene, task)
    finally:
        env.close()


@pytest.mark.parametrize("root_assistance", [False, True])
def test_training_and_independent_evaluation_preserve_full_task_and_inputs(tmp_path, root_assistance):
    source, scene = source_fixture(tmp_path)
    source_before = {path.name: digest(path) for path in source.iterdir()}
    trained = train_shared_exercise_policy(source, tmp_path / "learning", root_assistance=root_assistance,
                                          generations=2, population=4)
    learning = Path(trained["directory"])
    settings = json.loads((learning / "settings.json").read_text())
    policy = json.loads((learning / "policy.json").read_text())
    assert trained["status"] == "trained" and trained["completed_generations"] == 2
    assert trained["cache_hits"] >= 2 and trained["evaluated_episodes"] < 16
    assert trained["control_steps"] <= settings["maximum_control_steps"] == 48
    assert policy["root_assistance"] == ("explicit_3_axis" if root_assistance else "disabled")
    assert (learning / "scene.json").read_bytes() == (source / "scene.json").read_bytes()
    assert not (learning / "evaluation.json").exists()
    before = {path.relative_to(learning).as_posix(): digest(path) for path in learning.rglob("*") if path.is_file()}
    result = evaluate_shared_exercise_policy(learning, tmp_path / "evaluation", trials=2)
    assert result["status"] == "evaluated" and result["comparison_complete"]
    assert result["independent_changed_states"] == 2
    assert set(result["training_seeds"]).isdisjoint(result["evaluation_seeds"])
    assert result["horizon_steps"] == scene["task"]["horizon_steps"] == 3
    assert result["evaluation_feedback_to_optimizer"] is False
    assert result["policy_sha256"] == digest(learning / "policy.json")
    assert result["external_api_usd"] == 0
    hashes = {}
    for controller, summary in result["controllers"].items():
        assert len(summary["trials"]) == 2
        for row in summary["trials"]:
            assert row["metrics"]["completed_interval"] or row["metrics"]["failure"]
            assert row["control_steps"] == len(row["actions"])
            hashes.setdefault(row["seed"], set()).add(row["initial_state_sha256"])
            assert row["metrics"]["virtual_object_forces"] == "forbidden"
            assert all(len(event["action"]) == len(policy["action_names"]) for event in row["actions"])
            if controller == "coast":
                assert not any(any(event["action"]) for event in row["actions"])
    assert all(len(values) == 1 for values in hashes.values())
    assert {path.relative_to(learning).as_posix(): digest(path) for path in learning.rglob("*") if path.is_file()} == before
    assert {path.name: digest(path) for path in source.iterdir()} == source_before
    assert verify_artifacts(learning)["manifest_sha256"] == trained["manifest_sha256"]
    assert verify_artifacts(tmp_path / "evaluation")["manifest_sha256"] == result["manifest_sha256"]


@pytest.mark.parametrize("anchor", ["coast", "authored"])
def test_selecting_an_unchanged_anchor_is_not_called_learned_improvement(tmp_path, monkeypatch, anchor):
    source, _ = source_fixture(tmp_path)

    def trial(env, controller, seed, reset, weights, deadline, **kwargs):
        selected = np.all(weights == (0 if anchor == "coast" else 1))
        return {"seed": seed, "reset_options": reset, "status": "completed", "control_steps": 0,
                "initial_state_sha256": canonical_hash({"seed": seed}), "return": 0,
                "metrics": {"mean_reward": 1. if selected else 0., "completed_interval": True, "success": True}}

    monkeypatch.setattr("weft.shared_exercise_learning._trial", trial)
    result = train_shared_exercise_policy(source, tmp_path / "learning", generations=1, population=4)
    assert result["selected_feedback_kind"] == anchor + "_unchanged"
    assert result["selected_" + anchor + "_baseline"]
    assert "learned_improvement_claim" not in result


def test_timeout_preserves_actual_partial_training_and_rejects_checkpoint_evaluation(tmp_path, monkeypatch):
    import weft.shared_exercise_learning as module

    source, _ = source_fixture(tmp_path)
    advanced = []
    step = SharedExerciseTaskEnv.step

    def step_once(self, action):
        result = step(self, action)
        advanced.append(True)
        return result

    def deadline(_):
        if advanced:
            raise TimeoutError("injected bounded stop")

    monkeypatch.setattr(module, "_check", deadline)
    monkeypatch.setattr(SharedExerciseTaskEnv, "step", step_once)
    learning = tmp_path / "learning"
    with pytest.raises(TimeoutError, match="bounded stop"):
        train_shared_exercise_policy(source, learning, generations=1, population=4)
    training = json.loads((learning / "training.json").read_text())
    partial = training["active_candidates"][0]["trials"][0]
    assert partial["status"] == "interrupted" and partial["control_steps"] == 1
    assert partial["metrics"]["observed_substeps"] > 0
    assert json.loads((learning / "manifest.json").read_text())["status"] == "failed"
    verify_artifacts(learning)
    with pytest.raises(ValueError):
        evaluate_shared_exercise_policy(learning, tmp_path / "evaluation")
    assert not (tmp_path / "evaluation").exists()


@pytest.mark.parametrize("mutation", ["repeat_training_state", "replace_policy_symlink", "replace_policy_fifo"])
def test_evaluation_rejects_reused_states_and_unbounded_policy_substitution(tmp_path, monkeypatch, mutation):
    import weft.shared_exercise_learning as module

    source, _ = source_fixture(tmp_path)
    learning = tmp_path / "learning"
    train_shared_exercise_policy(source, learning, generations=1, population=4)
    training = json.loads((learning / "training.json").read_text())
    trained_hash = training["history"][0]["candidates"][0]["trials"][0]["initial_state_sha256"]
    original = module._trial
    calls = []

    def trial(*args, **kwargs):
        row = original(*args, **kwargs)
        calls.append(True)
        if mutation == "repeat_training_state":
            row["initial_state_sha256"] = trained_hash
        elif len(calls) == 4:
            policy_path = learning / "policy.json"
            saved = policy_path.read_bytes()
            policy_path.unlink()
            if mutation == "replace_policy_symlink":
                other = tmp_path / "same-policy.json"
                other.write_bytes(saved)
                policy_path.symlink_to(other)
            else:
                os.mkfifo(policy_path)
        return row

    monkeypatch.setattr(module, "_trial", trial)
    evaluation = tmp_path / "evaluation"
    with pytest.raises(ValueError, match="training state|unsafe|regular file"):
        evaluate_shared_exercise_policy(learning, evaluation, trials=1)
    assert json.loads((evaluation / "manifest.json").read_text())["status"] == "failed"
    assert json.loads((evaluation / "failure.json").read_text())["partial_evidence_preserved"]


@pytest.mark.parametrize("kwargs", [{"seed": True}, {"generations": 9}, {"population": 17},
                                     {"max_seconds": 61}, {"max_seconds": float("nan")}])
def test_training_bounds_reject_before_output(tmp_path, kwargs):
    source, _ = source_fixture(tmp_path)
    with pytest.raises(ValueError):
        train_shared_exercise_policy(source, tmp_path / "unused", **kwargs)
    assert not (tmp_path / "unused").exists()


def test_packaging_failure_preserves_partial_files_and_failed_manifest(tmp_path, monkeypatch):
    import weft.shared_exercise_learning as module

    source, _ = source_fixture(tmp_path)
    original = module.write_package_files

    def interrupted(directory, files):
        name = next(iter(files))
        original(directory, {name: files[name]})
        raise OSError("injected package write failure")

    monkeypatch.setattr(module, "write_package_files", interrupted)
    out = tmp_path / "partial"
    with pytest.raises(OSError, match="package write failure"):
        train_shared_exercise_policy(source, out, generations=1, population=4)
    assert (out / "scene.json").is_file()
    assert json.loads((out / "failure.json").read_text())["stage"] == "input_packaging"
    assert json.loads((out / "manifest.json").read_text())["status"] == "failed"
    verify_artifacts(out)


@pytest.mark.parametrize("fail_evaluation", [False, True])
def test_study_driver_runs_one_fixed_pair_without_retry_or_interval_expansion(tmp_path, monkeypatch, fail_evaluation):
    path = Path(__file__).resolve().parents[1] / "scripts/studies/check_shared_exercise_learning.py"
    spec = importlib.util.spec_from_file_location("exercise_learning_study", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    source, _ = source_fixture(tmp_path)
    calls = []

    def train(run, out, **kwargs):
        calls.append(("train", kwargs))
        return {"manifest_sha256": "a" * 64, "policy_sha256": "b" * 64, "seconds": 1., "control_steps": 3,
                "evaluated_episodes": 1, "cache_hits": 1, "selected_feedback_kind": "authored_unchanged",
                "improvement_over_authored_training": 0.}

    def evaluate(run, out, **kwargs):
        calls.append(("evaluate", kwargs))
        if fail_evaluation:
            raise RuntimeError("preserve this failed evaluation")
        return {"manifest_sha256": "c" * 64, "policy_sha256": "b" * 64, "seconds": 1., "control_steps": 3,
                "comparison_complete": True, "independent_changed_states": 3, "controllers": {}}

    monkeypatch.setattr(module, "train_shared_exercise_policy", train)
    monkeypatch.setattr(module, "evaluate_shared_exercise_policy", evaluate)
    out = tmp_path / "study"
    if fail_evaluation:
        with pytest.raises(RuntimeError, match="failed evaluation"):
            module.run(source, out)
    else:
        module.run(source, out)
    assert [name for name, _ in calls] == ["train", "evaluate"]
    assert calls[0][1]["generations"] == 3 and calls[0][1]["population"] == 8
    assert calls[1][1]["trials"] == 3 and calls[1][1]["seed"] == 924017
    plan = json.loads((out / "pre-training-freeze.json").read_text())
    assert plan["horizon_steps"] == 3 and plan["declared_interval_s"] == .1
    assert json.loads((out / "manifest.json").read_text())["status"] == ("failed" if fail_evaluation else "completed")
