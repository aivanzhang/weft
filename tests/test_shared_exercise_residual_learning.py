"""Residual search must preserve authored assistance, immutable tasks and independent evaluation."""

from copy import deepcopy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from weft.io import canonical_hash, digest, write_json
from weft.pipeline import verify_artifacts
import weft.shared_exercise_residual_learning as module
from weft.shared_exercise_task import SharedExerciseTaskEnv, build_shared_exercise_task
from test_shared_exercise_run import source_fixture


def policy_fixture(tmp_path, root_assistance=False):
    _, scene = source_fixture(tmp_path)
    task = build_shared_exercise_task(scene, root_assistance=root_assistance)
    env = SharedExerciseTaskEnv(scene, task)
    env.reset(seed=19)
    policy = module._policy(env, np.zeros((2, 19)), 1729, .5, module._runtime_hashes())
    return scene, task, env, policy


@pytest.mark.parametrize("root_assistance", [False, True])
def test_zero_residual_is_exact_authored_detached_and_has_no_hot_path_hashing(tmp_path, monkeypatch, root_assistance):
    scene, task, env, policy = policy_fixture(tmp_path, root_assistance)
    try:
        action = module.prepare_shared_exercise_residual_policy(policy, env)
        policy["residual_coefficients"][0][0] = .25
        policy["action_names"].append("object_force")
        scene["task"]["horizon_steps"] = 1
        task["control"]["root_assistance"] = "invalid"
        monkeypatch.setattr(module, "_runtime_hashes", lambda: pytest.fail("Rehashed immutable files per action"))
        assert action().tobytes() == env.authored_action().tobytes()
        action()[:] = 0
        assert action().tobytes() == env.authored_action().tobytes()
        env.step(action())
        assert action().tobytes() == env.authored_action().tobytes()
        assert len(action()) == (22 if root_assistance else 19)
        assert not any(env.snapshot()["engine_state"]["normalized_action"][env.action_count:])
    finally:
        env.close()


def test_phase_residual_is_joint_specific_bounded_and_leaves_root_feedback_exact():
    authored = np.linspace(-1., 1., 22)
    authored[19] = -0.
    coefficient = np.zeros((2, 19))
    coefficient[:, 7] = .25
    for phase in np.linspace(0., 1., 21):
        env = SimpleNamespace(authored_action=lambda: authored, observation=lambda: np.array([phase]))
        action = module._action(coefficient, env)
        assert action[19:].tobytes() == authored[19:].tobytes()
        np.testing.assert_array_equal(np.delete(action, 7), np.delete(authored, 7))
        assert 0 <= action[7] - authored[7] <= .25
        assert np.all(abs(action) <= 1)
        basis = module.phase_basis(float(phase))
        assert np.all(basis >= 0) and sum(basis) <= 1
    np.testing.assert_array_equal(module.phase_basis(0.), [0., 0.])
    np.testing.assert_array_equal(module.phase_basis(.5), [1., 0.])
    np.testing.assert_array_equal(module.phase_basis(1.), [0., 1.])


@pytest.mark.parametrize("phase", [-.01, 1.01, float("nan"), float("inf"), True])
def test_phase_cannot_retime_or_read_beyond_the_declared_interval(phase):
    with pytest.raises(ValueError, match="phase"):
        module.phase_basis(phase)


@pytest.mark.parametrize("field,value", [
    ("scene_sha256", "0" * 64), ("task_sha256", "0" * 64), ("evaluator_sha256", "0" * 64),
    ("gates_sha256", "0" * 64), ("runtime_sha256", {}), ("root_assistance", "explicit_3_axis"),
    ("action_names", ["object_force"]), ("basis_profile", "retimed_basis"),
    ("feature_names", ["future_state"]), ("privileged_simulator_features", False),
    ("residual_bound", .5), ("residual_coefficients", [[True] * 19] * 2),
    ("residual_coefficients", [[.26] * 19] * 2), ("residual_coefficients", [[float("nan")] * 19] * 2),
    ("residual_coefficients", [[0.] * 22] * 2),
])
def test_policy_rejects_changed_gates_identity_channels_basis_and_privilege(tmp_path, field, value):
    scene, task, env, policy = policy_fixture(tmp_path)
    try:
        policy[field] = value
        with pytest.raises(ValueError):
            module.validate_shared_exercise_residual_policy(policy, scene, task)
    finally:
        env.close()


def test_dispatcher_is_pinned_in_v2_runtime():
    hashes = module._runtime_hashes()
    assert hashes["shared_exercise_policies.py"] == digest(Path(module.__file__).with_name("shared_exercise_policies.py"))
    assert hashes["shared_exercise_residual_learning.py"] == digest(module.__file__)


@pytest.mark.parametrize("root_assistance", [False, True])
def test_fixture_training_and_fresh_evaluation_preserve_exact_sealed_task_and_source_views(tmp_path, monkeypatch, root_assistance):
    source, scene = source_fixture(tmp_path)
    task = build_shared_exercise_task(scene, root_assistance=root_assistance)
    write_json(source / "exercise-task.json", task)
    write_json(source / "source.json", {"url": "https://example.invalid/authored-fixture", "license": "fixture"})
    (source / "source.webm").write_bytes(b"Not video inference: exact retained fixture bytes")
    write_json(source / "manifest.json", {"files": {p.name: digest(p) for p in source.iterdir() if p.name != "manifest.json"}})
    source_before = {p.name: digest(p) for p in source.iterdir()}
    learning = tmp_path / "learning"
    trained = module.train_shared_exercise_residual_policy(source, learning, generations=2, population=4)
    settings = json.loads((learning / "settings.json").read_text())
    policy = json.loads((learning / "policy.json").read_text())
    assert trained["status"] == "trained" and trained["completed_generations"] == 2
    assert trained["cache_hits"] >= 1
    assert trained["control_steps"] <= settings["maximum_control_steps"] == 48
    unique_trials = [trial for generation in trained["history"] for row in generation["candidates"]
                     if not row["cached"] for trial in row["trials"]]
    assert trained["control_steps"] == sum(row["control_steps"] for row in unique_trials)
    assert trained["evaluated_episodes"] == len(unique_trials)
    assert policy["root_assistance"] == task["control"]["root_assistance"]
    for name in ("scene.json", "exercise-task.json", "source.json", "source.webm"):
        assert (learning / name).read_bytes() == (source / name).read_bytes()
    assert not (learning / "evaluation.json").exists()
    before = {p.relative_to(learning).as_posix(): digest(p) for p in learning.rglob("*") if p.is_file()}
    evaluation_path = tmp_path / "evaluation"
    excluded = module._seed_plan(924017, 1, settings["training_seeds"])
    resets, original_reset = [], SharedExerciseTaskEnv.reset

    def reset_after_freeze(self, *, seed=None, options=None):
        frozen = json.loads((evaluation_path / "pre-evaluation-freeze.json").read_text())
        assert seed in frozen["evaluation_seeds"]
        assert frozen["policy_sha256"] == digest(evaluation_path / "policy.json") == trained["policy_sha256"]
        resets.append(seed)
        return original_reset(self, seed=seed, options=options)

    monkeypatch.setattr(SharedExerciseTaskEnv, "reset", reset_after_freeze)
    evaluated = module.evaluate_shared_exercise_residual_policy(learning, evaluation_path, trials=2, excluded_seeds=excluded)
    assert evaluated["status"] == "evaluated" and evaluated["comparison_complete"]
    assert evaluated["independent_changed_states"] == 2
    assert set(evaluated["evaluation_seeds"]).isdisjoint([*settings["training_seeds"], *excluded])
    assert len(resets) == 8
    assert evaluated["horizon_steps"] == 3 and evaluated["evaluation_feedback_to_optimizer"] is False
    assert evaluated["control_steps"] <= evaluated["maximum_control_steps"] == 24
    assert evaluated["policy_sha256"] == trained["policy_sha256"]
    assert set(evaluated["controllers"]) == set(module.CONTROLLERS)
    hashes = {}
    for controller, summary in evaluated["controllers"].items():
        for row in summary["trials"]:
            assert row["metrics"]["completed_interval"] or row["metrics"]["failure"]
            assert row["control_steps"] == len(row["actions"])
            hashes.setdefault(row["seed"], set()).add(row["initial_state_sha256"])
            assert row["metrics"]["virtual_object_forces"] == "forbidden"
            assert all(len(event["action"]) == len(policy["action_names"]) for event in row["actions"])
            if controller == "coast":
                assert not any(any(event["action"]) for event in row["actions"])
    assert all(len(values) == 1 for values in hashes.values())
    assert evaluated["control_steps"] == sum(row["control_steps"] for summary in evaluated["controllers"].values() for row in summary["trials"])
    for name in ("scene.json", "exercise-task.json", "source.json", "source.webm", "policy.json"):
        assert (evaluation_path / name).read_bytes() == (learning / name).read_bytes()
    assert {p.relative_to(learning).as_posix(): digest(p) for p in learning.rglob("*") if p.is_file()} == before
    assert {p.name: digest(p) for p in source.iterdir()} == source_before
    assert verify_artifacts(learning)["manifest_sha256"] == trained["manifest_sha256"]
    assert verify_artifacts(evaluation_path)["manifest_sha256"] == evaluated["manifest_sha256"]


@pytest.mark.parametrize("prefer_residual", [False, True])
def test_selection_labels_distinguish_zero_anchor_from_actual_changed_coefficients(tmp_path, monkeypatch, prefer_residual):
    source, _ = source_fixture(tmp_path)

    def trial(env, controller, seed, reset, coefficients, deadline, budget, **kwargs):
        for _ in range(env.horizon_steps):
            budget.reserve()
        selected = bool(np.any(coefficients)) is prefer_residual
        return {"seed": seed, "reset_options": reset, "status": "completed", "control_steps": env.horizon_steps,
                "initial_state_sha256": canonical_hash({"seed": seed}), "return": 0.,
                "metrics": {"mean_reward": 1. if selected else 0., "completed_interval": True, "success": True}}

    monkeypatch.setattr(module, "_trial", trial)
    result = module.train_shared_exercise_residual_policy(source, tmp_path / "learning", generations=1, population=4)
    assert result["control_steps"] == 24
    assert result["selected_feedback_kind"] == ("searched_joint_residuals" if prefer_residual else "authored_unchanged")
    assert result["selected_authored_baseline"] is not prefer_residual
    assert result["selected_coast_baseline"] is False
    assert result["training_score_improved_over_authored"] is prefer_residual


def test_timeout_keeps_partial_action_count_and_failed_checkpoint(tmp_path, monkeypatch):
    source, _ = source_fixture(tmp_path)
    advanced, original = [], SharedExerciseTaskEnv.step

    def step_once(self, action):
        result = original(self, action)
        advanced.append(True)
        return result

    def deadline(_):
        if advanced:
            raise TimeoutError("injected bounded stop")

    monkeypatch.setattr(module, "_check", deadline)
    monkeypatch.setattr(SharedExerciseTaskEnv, "step", step_once)
    learning = tmp_path / "learning"
    with pytest.raises(TimeoutError, match="bounded stop"):
        module.train_shared_exercise_residual_policy(source, learning, generations=1, population=4)
    training = json.loads((learning / "training.json").read_text())
    partial = training["active_candidates"][0]["trials"][0]
    assert training["control_steps"] == partial["control_steps"] == len(advanced) == 1
    assert partial["status"] == "interrupted" and partial["metrics"]["observed_substeps"] > 0
    assert json.loads((learning / "manifest.json").read_text())["status"] == "failed"
    verify_artifacts(learning)
    with pytest.raises(ValueError):
        module.evaluate_shared_exercise_residual_policy(learning, tmp_path / "evaluation")


def test_action_reservation_stops_before_an_over_budget_physics_call(tmp_path):
    scene, task, env, _ = policy_fixture(tmp_path)
    try:
        budget = module._Budget(1)
        with pytest.raises(ValueError, match="budget exhausted") as error:
            module._trial(env, "learned", 1729, module.TRAIN_RESETS[0], np.zeros((2, 19)),
                          module.time.monotonic() + 10, budget)
        assert budget.used == error.value.partial_trial["control_steps"] == env.snapshot()["engine_state"]["steps"] == 1
    finally:
        env.close()


@pytest.mark.parametrize("mutation", ["repeat_training_state", "policy_change", "freeze_change"])
def test_evaluation_rejects_reused_starts_or_changed_frozen_bytes(tmp_path, monkeypatch, mutation):
    source, _ = source_fixture(tmp_path)
    learning, evaluation = tmp_path / "learning", tmp_path / "evaluation"
    training = module.train_shared_exercise_residual_policy(source, learning, generations=1, population=4)
    trained_hash = training["history"][0]["candidates"][0]["trials"][0]["initial_state_sha256"]
    original = module._trial
    calls = []

    def trial(*args, **kwargs):
        row = original(*args, **kwargs)
        calls.append(True)
        if mutation == "repeat_training_state":
            row["initial_state_sha256"] = trained_hash
        elif len(calls) == 1:
            path = learning / "policy.json" if mutation == "policy_change" else evaluation / "pre-evaluation-freeze.json"
            path.write_bytes(path.read_bytes() + b"\n")
        return row

    monkeypatch.setattr(module, "_trial", trial)
    with pytest.raises(ValueError, match="training state|SHA256"):
        module.evaluate_shared_exercise_residual_policy(learning, evaluation, trials=1)
    assert json.loads((evaluation / "manifest.json").read_text())["status"] == "failed"
    assert json.loads((evaluation / "failure.json").read_text())["partial_evidence_preserved"]


@pytest.mark.parametrize("kwargs", [{"seed": True}, {"generations": 7}, {"population": 13},
                                     {"max_seconds": 61}, {"max_seconds": float("nan")}])
def test_training_configuration_rejects_before_output(tmp_path, kwargs):
    source, _ = source_fixture(tmp_path)
    with pytest.raises(ValueError):
        module.train_shared_exercise_residual_policy(source, tmp_path / "unused", **kwargs)
    assert not (tmp_path / "unused").exists()


def test_action_envelope_includes_all_population_anchors_without_shortening(tmp_path, monkeypatch):
    source, _ = source_fixture(tmp_path)
    monkeypatch.setattr(module, "MAX_CONTROL_STEPS", 47)
    with pytest.raises(ValueError, match="8640 actions"):
        module.train_shared_exercise_residual_policy(source, tmp_path / "unused", generations=2, population=4)
    assert not (tmp_path / "unused").exists()


@pytest.mark.parametrize("status", ["running", "queued", None])
def test_nonterminal_source_cannot_start_training(tmp_path, status):
    source, _ = source_fixture(tmp_path)
    manifest = json.loads((source / "manifest.json").read_text())
    write_json(source / "manifest.json", {**manifest, "status": status})
    with pytest.raises(ValueError, match="terminal"):
        module.train_shared_exercise_residual_policy(source, tmp_path / "unused", generations=1, population=4)
    assert not (tmp_path / "unused").exists()


def test_failed_final_publication_never_retains_a_success_manifest(tmp_path, monkeypatch):
    out = tmp_path / "failed"
    out.mkdir()
    write_json(out / "manifest.json", {"status": "trained", "files": {}})
    (out / "oversized.json").write_text("x" * 2048)
    monkeypatch.setattr(module, "MAX_OUTPUT_BYTES", 1024)
    module._failure(out, "training.json", {}, ValueError("output bound"))
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["status"] == "failed" and manifest["custody"] == "failure_receipt_only"
    assert manifest["unsealed_outputs_preserved"]
    assert (out / "oversized.json").stat().st_size == 2048


@pytest.mark.parametrize("fail_evaluation", [False, True])
def test_study_freezes_exact_seed_plan_before_one_bounded_pair_without_retries(tmp_path, monkeypatch, fail_evaluation):
    path = Path(__file__).resolve().parents[1] / "scripts/studies/check_shared_exercise_residual.py"
    spec = importlib.util.spec_from_file_location("exercise_residual_study", path)
    study = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(study)
    source, _ = source_fixture(tmp_path)
    out, calls = tmp_path / "study", []

    def train(run, directory, **kwargs):
        frozen = json.loads((out / "pre-training-freeze.json").read_text())
        assert frozen["evaluation_seeds"] and frozen["maximum_training_control_steps"] == 432
        calls.append(("train", kwargs))
        return {"manifest_sha256": "a" * 64, "policy_sha256": "b" * 64, "seconds": 1., "control_steps": 3,
                "evaluated_episodes": 1, "cache_hits": 1, "selected_feedback_kind": "authored_unchanged",
                "improvement_over_authored_training": 0., "training_score_improved_over_authored": False}

    def evaluate(run, directory, **kwargs):
        calls.append(("evaluate", kwargs))
        if fail_evaluation:
            raise RuntimeError("preserve failed evaluation")
        frozen = json.loads((out / "pre-training-freeze.json").read_text())
        return {"manifest_sha256": "c" * 64, "policy_sha256": "b" * 64, "seconds": 1., "control_steps": 3,
                "comparison_complete": True, "independent_changed_states": 3, "controllers": {},
                "evaluation_seeds": frozen["evaluation_seeds"], "training_seeds": frozen["training_seeds"]}

    monkeypatch.setattr(study, "train_shared_exercise_residual_policy", train)
    monkeypatch.setattr(study, "evaluate_shared_exercise_residual_policy", evaluate)
    if fail_evaluation:
        with pytest.raises(RuntimeError, match="failed evaluation"):
            study.run(source, out, excluded_seeds=[17, 19])
    else:
        assert study.run(source, out, excluded_seeds=[17, 19])["status"] == "completed"
    assert [name for name, _ in calls] == ["train", "evaluate"]
    assert calls[0][1]["generations"] == 6 and calls[0][1]["population"] == 12
    assert calls[1][1]["trials"] == 3 and calls[1][1]["excluded_seeds"] == [17, 19]
    assert json.loads((out / "manifest.json").read_text())["status"] == ("failed" if fail_evaluation else "completed")
