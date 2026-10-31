"""Support policy contracts and existing learner control flow; native dynamics are forbidden."""

from copy import deepcopy
import json
import time

import mujoco
import numpy as np
import pytest

from weft import shared_exercise_learning as gains
from weft import shared_exercise_policies as policies
from weft import shared_exercise_residual_learning as residual
from weft import shared_support_task as support
from weft.io import canonical_hash, digest, write_json
from weft.pipeline import verify_artifacts
from weft.shared_support_fixture import static_support_fixture


@pytest.fixture(autouse=True)
def no_native_dynamics(monkeypatch):
    monkeypatch.setattr(mujoco, "mj_step", lambda *_a, **_k: pytest.fail("Support learning tests cannot advance native physics"))


def seal(directory):
    write_json(directory / "manifest.json", {"status": "ready", "files": {
        path.relative_to(directory).as_posix(): digest(path)
        for path in directory.rglob("*") if path.is_file() and path.name != "manifest.json"}})


def source_package(tmp_path, *, root=False):
    scene, task = static_support_fixture()
    if root:
        task = support.build_shared_support_task(scene, root_assistance=True,
                                                support_requirements=task["support_requirements"])
    source = tmp_path / "source"
    source.mkdir()
    for name, value in {"scene.json": scene, "exercise-task.json": task,
                        "support-config.json": {"scope": "Authored analytical source view, not reconstructed contact"},
                        "source-clock.json": {"scope": "Authored two-second source clock"}}.items():
        write_json(source / name, value)
    seal(source)
    return source, scene, task


class AnalyticalEnv:
    """Exercise the real search/trial loop without pretending to simulate support forces."""

    def __init__(self, scene, task):
        self.scene, self.task = deepcopy(scene), deepcopy(task)
        self.scene_sha256, self.task_sha256 = canonical_hash(scene), canonical_hash(task)
        self.evaluator_sha256 = residual._contract(scene, residual._runtime_hashes(scene))["evaluator_sha256"]
        self.horizon_steps = task["interval"]["horizon_steps"]
        self.action_count = 22 if task["control"]["root_assistance"] == "explicit_3_axis" else 19
        self.action_names = scene["action_contract"]["names"][:self.action_count]
        self.reset(seed=0, options={})

    def reset(self, *, seed, options):
        self.seed, self.options, self.steps, self.reward_sum = seed, deepcopy(options), 0, 0.
        self.terminated = self.truncated = False

    def authored_action(self):
        action = np.zeros(self.action_count)
        if self.action_count == 22:
            action[19:] = [-0., .1, -.1]
        return action

    def observation(self):
        return np.array([self.steps / self.horizon_steps])

    def observation_contract(self):
        return {"profile": "shared-support-observation-v1"}

    def step(self, action):
        assert len(action) == self.action_count and np.all(abs(action) <= 1.)
        self.steps += 1
        reward = 1. - float(np.mean(abs(action[:19])))
        self.reward_sum += reward
        self.terminated = self.steps == self.horizon_steps
        return self.observation(), reward, self.terminated, False, {}

    def metrics(self):
        return {"completed_interval": self.steps == self.horizon_steps, "failure": None,
                "success": self.terminated, "mean_reward": self.reward_sum / max(1, self.steps)}

    def snapshot(self):
        return {"engine_state": {"seed": self.seed, "reset": self.options, "steps": self.steps,
                                  "supports": deepcopy(self.scene["objects"])}, "task_metrics": self.metrics()}

    def close(self):
        pass


def policy_fixture(scene, task):
    env = AnalyticalEnv(scene, task)
    policy = residual._policy(env, np.zeros((2, 19)), 1729, 1., residual._runtime_hashes(scene))
    return env, policy


def test_support_contract_is_opt_in_and_pins_both_evaluators(tmp_path, monkeypatch):
    source, scene, task = source_package(tmp_path)
    env, policy = policy_fixture(scene, task)
    assert policies.validate_policy(policy, scene, task) == policy
    assert policy["profile"] == residual.SUPPORT_POLICY_PROFILE
    assert policy["algorithm"] == residual.SUPPORT_ALGORITHM
    assert policy["scope"] == residual.SUPPORT_SCOPE
    assert policy["gates_sha256"] == support.GATES_SHA256 != residual.GATES_SHA256
    hashes = residual._runtime_hashes(scene)
    assert set(hashes) - set(residual._runtime_hashes()) == {"shared_support_task.py"}
    assert policy["evaluator_sha256"] == support.evaluator_identity({
        name: hashes[name] for name in ("shared_exercise_task.py", "shared_support_task.py")})
    assert policy["evaluator_sha256"] != hashes["shared_exercise_task.py"]
    assert gains._context(scene, task, allow_static_support=True)[2] == env.action_count == 19
    assert gains._source(source, None, False, time.monotonic() + 30, allow_static_support=True)[1:3] == (scene, task)

    monkeypatch.setattr(gains, "validate_shared_world", lambda *_a: pytest.fail("Rejected profile entered scene admission"))
    for call in (lambda: gains._context(scene, task),
                 lambda: gains._source(source, None, False, time.monotonic() + 30)):
        with pytest.raises(ValueError, match="separately qualified"):
            call()
    for profile in ("shared-exercise-policy-v1", "shared-exercise-policy-v2", "shared-exercise-policy-v3", "shared-exercise-policy-v4"):
        old = {**policy, "profile": profile}
        for call in (lambda: policies.validate_policy(old, scene, task),
                     lambda: policies.prepare_policy(old, env),
                     lambda: residual.validate_shared_exercise_residual_policy(old, scene, task)):
            with pytest.raises(ValueError, match="separately qualified"):
                call()
    with pytest.raises(ValueError, match="separately qualified"):
        policies.validate_policy(policy, {"profile": "shared-world-v3"}, {})


@pytest.mark.parametrize("root", [False, True])
def test_support_actor_keeps_authored_anchor_and_root_bytes(tmp_path, monkeypatch, root):
    _, scene, task = source_package(tmp_path, root=root)
    env, policy = policy_fixture(scene, task)
    zero = policies.prepare_policy(policy, env)
    policy["residual_coefficients"][0][3] = .25
    actor = policies.prepare_policy(policy, env)
    policy["residual_coefficients"][0][3] = -.25
    assert policies.selection_kind({**policy, "residual_coefficients": [[0.] * 19] * 2}) == ("authored_unchanged", True, False)
    assert policies.selection_kind(policy) == ("searched_joint_residuals", False, False)
    monkeypatch.setattr(residual, "_runtime_hashes", lambda *_a: pytest.fail("Hot-path file hashing"))
    for step in (0, 30, 60):
        env.steps = step
        authored = env.authored_action()
        assert zero().tobytes() == authored.tobytes()
        action = actor()
        assert len(action) == (22 if root else 19)
        assert action[19:].tobytes() == authored[19:].tobytes()
        assert 0 <= action[3] - authored[3] <= .25
        np.testing.assert_array_equal(np.delete(action, 3), np.delete(authored, 3))


def test_support_policy_rejects_exercise_identity_or_observation(tmp_path):
    _, scene, task = source_package(tmp_path)
    env, policy = policy_fixture(scene, task)
    for change in ({"gates_sha256": residual.GATES_SHA256},
                   {"evaluator_sha256": policy["runtime_sha256"]["shared_exercise_task.py"]},
                   {"runtime_sha256": residual._runtime_hashes()}, {"scope": residual.SCOPE}):
        with pytest.raises(ValueError):
            policies.validate_policy({**policy, **change}, scene, task)
    env.observation_contract = lambda: {"profile": "shared-exercise-observation-v1"}
    with pytest.raises(ValueError, match="observation contract"):
        policies.prepare_policy(policy, env)


def test_support_views_preserve_exact_config_clock_and_require_declared_task(tmp_path):
    source, scene, _ = source_package(tmp_path)
    views = residual._source_views(source, (source / "manifest.json").read_bytes(), scene)
    assert set(views) == {"support-config.json", "source-clock.json"}
    assert all(raw == (source / name).read_bytes() for name, raw in views.items())
    assert residual._source_views(source, (source / "manifest.json").read_bytes()) == {}
    (source / "inputs").mkdir()
    (source / "inputs/support-config.json").write_bytes(views["support-config.json"])
    seal(source)
    assert "inputs/support-config.json" in residual._source_views(source, (source / "manifest.json").read_bytes(), scene)
    write_json(source / "inputs/support-config.json", {"changed": True})
    seal(source)
    with pytest.raises(ValueError, match="configuration views differ"):
        residual._source_views(source, (source / "manifest.json").read_bytes(), scene)
    (source / "exercise-task.json").unlink()
    seal(source)
    with pytest.raises(ValueError, match="explicit sealed support task"):
        gains._source(source, None, False, time.monotonic() + 30, allow_static_support=True)


def test_support_reuses_bounded_search_fresh_evaluation_and_exact_source_custody(tmp_path, monkeypatch):
    source, scene, task = source_package(tmp_path)
    monkeypatch.setattr(residual, "SharedExerciseTaskEnv", AnalyticalEnv)
    monkeypatch.setattr(residual, "prepare_shared_exercise_snapshot_validator", lambda *_a: deepcopy)
    before = {p.name: digest(p) for p in source.iterdir()}
    trained_path, evaluation_path = tmp_path / "training", tmp_path / "evaluation"
    trained = residual.train_shared_exercise_residual_policy(source, trained_path, generations=1, population=4)
    settings = json.loads((trained_path / "settings.json").read_bytes())
    policy = json.loads((trained_path / "policy.json").read_bytes())
    assert trained["status"] == "trained" and trained["control_steps"] == settings["maximum_control_steps"] == 480
    assert settings["algorithm"] == residual.SUPPORT_ALGORITHM and settings["gates_sha256"] == support.GATES_SHA256
    assert settings["source_views_sha256"] == {name: before[name] for name in ("support-config.json", "source-clock.json")}
    assert settings["equipment_force_channels"] == settings["root_residual_channels"] == []
    assert settings["training_reset_options"] == list(residual.TRAIN_RESETS)
    assert trained["selected_feedback_kind"] == "authored_unchanged" and trained["selected_authored_baseline"]
    assert trained["improvement_over_authored_training"] == 0 and not trained["training_score_improved_over_authored"]
    assert policies.validate_policy(policy, scene, task) == policy
    training_before = {p.relative_to(trained_path).as_posix(): digest(p) for p in trained_path.rglob("*") if p.is_file()}
    original_reset = AnalyticalEnv.reset

    def reset_after_freeze(self, *, seed, options):
        frozen = json.loads((evaluation_path / "pre-evaluation-freeze.json").read_bytes())
        assert frozen["policy_sha256"] == trained["policy_sha256"]
        if seed != 0:
            assert seed in frozen["evaluation_seeds"] and options == frozen["reset_options"]
        return original_reset(self, seed=seed, options=options)

    monkeypatch.setattr(AnalyticalEnv, "reset", reset_after_freeze)
    evaluated = policies.evaluate_policy(trained_path, evaluation_path, trials=3)
    assert evaluated["status"] == "evaluated" and evaluated["comparison_complete"]
    assert evaluated["independent_changed_states"] == 3 and evaluated["control_steps"] == evaluated["maximum_control_steps"] == 720
    assert evaluated["evaluation_feedback_to_optimizer"] is False and evaluated["horizon_steps"] == 60
    assert set(evaluated["evaluation_seeds"]).isdisjoint(settings["training_seeds"])
    frozen = json.loads((evaluation_path / "pre-evaluation-freeze.json").read_bytes())
    assert frozen["policy_sha256"] == trained["policy_sha256"] and frozen["evaluation_seeds"] == evaluated["evaluation_seeds"]
    for summary in evaluated["controllers"].values():
        assert len(summary["trials"]) == 3
        for trial in summary["trials"]:
            assert trial["control_steps"] == len(trial["actions"]) == 60
            assert trial["initial_state_sha256"] == evaluated["initial_states"][str(trial["seed"])]
            assert trial["initial_snapshot"]["engine_state"]["supports"] == trial["final_snapshot"]["engine_state"]["supports"] == scene["objects"]
    for name in ("scene.json", "exercise-task.json", "support-config.json", "source-clock.json"):
        assert (source / name).read_bytes() == (trained_path / name).read_bytes() == (evaluation_path / name).read_bytes()
    assert before == {p.name: digest(p) for p in source.iterdir()}
    assert training_before == {p.relative_to(trained_path).as_posix(): digest(p) for p in trained_path.rglob("*") if p.is_file()}
    assert verify_artifacts(trained_path)["manifest_sha256"] == trained["manifest_sha256"]
    assert verify_artifacts(evaluation_path)["manifest_sha256"] == evaluated["manifest_sha256"]
    with pytest.raises(ValueError, match="legacy equipment-feasibility selector"):
        policies.select_policy(trained_path, tmp_path / "selection")
    assert not (tmp_path / "selection").exists()


def test_support_reset_failures_are_not_retried_or_replaced(tmp_path, monkeypatch):
    source, _, _ = source_package(tmp_path)

    class ResetFailure(AnalyticalEnv):
        def reset(self, *, seed, options):
            if seed != 0:
                raise ValueError("Analytical admission failure")
            super().reset(seed=seed, options=options)

    monkeypatch.setattr(residual, "SharedExerciseTaskEnv", ResetFailure)
    monkeypatch.setattr(residual, "prepare_shared_exercise_snapshot_validator", lambda *_a: deepcopy)
    out = tmp_path / "training"
    trained = residual.train_shared_exercise_residual_policy(source, out, generations=1, population=4)
    assert trained["control_steps"] == 0 and trained["evaluated_episodes"] == 8
    assert trained["selected_feedback_kind"] == "authored_unchanged" and not trained["training_score_improved_over_authored"]
    for row in trained["history"][0]["candidates"]:
        assert row["score"] == -2 and all(trial["status"] == "reset_failed" for trial in row["trials"])
    evaluated = residual.evaluate_shared_exercise_residual_policy(out, tmp_path / "evaluation")
    assert evaluated["control_steps"] == evaluated["independent_changed_states"] == 0
    assert not evaluated["comparison_complete"]
    assert all(summary["reset_failures"] == 3 for summary in evaluated["controllers"].values())


def test_support_interrupted_trial_preserves_failure_without_inventing_policy(tmp_path, monkeypatch):
    source, _, _ = source_package(tmp_path)

    class Interrupted(AnalyticalEnv):
        def step(self, action):
            raise RuntimeError("Analytical interrupted step")

    monkeypatch.setattr(residual, "SharedExerciseTaskEnv", Interrupted)
    monkeypatch.setattr(residual, "prepare_shared_exercise_snapshot_validator", lambda *_a: deepcopy)
    out = tmp_path / "training"
    with pytest.raises(RuntimeError, match="interrupted step"):
        residual.train_shared_exercise_residual_policy(source, out, generations=1, population=4)
    training = json.loads((out / "training.json").read_bytes())
    assert training["status"] == "failed" and training["control_steps"] == 1
    assert training["active_candidates"][0]["trials"][0]["status"] == "interrupted"
    assert json.loads((out / "manifest.json").read_bytes())["status"] == "failed"
    assert not (out / "policy.json").exists()
    with pytest.raises(ValueError, match="unsuccessful exercise lab input"):
        residual.evaluate_shared_exercise_residual_policy(out, tmp_path / "evaluation")
    assert not (tmp_path / "evaluation").exists()
