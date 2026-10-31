"""Bounded state feedback reuses existing features without changing source, gates or root control."""

from copy import deepcopy
import inspect
import json
from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np
import pytest

from weft import shared_exercise_feedback_residual as actor
from weft import shared_exercise_feedback_learning as learner
from weft import shared_exercise_residual_learning as legacy
from weft import shared_exercise_learning as gains
from weft.io import canonical_hash, digest, write_json
from weft.pipeline import verify_artifacts
from weft.shared_exercise_task import SharedExerciseTaskEnv, build_shared_exercise_task
from weft.shared_exercise_feasibility_learning import optimization_rank
from test_person_only_downstream import metric_fixture
from test_shared_exercise_selection import NoPhysicsEnv
from test_shared_world_human_only import person_scene


def metadata(assistance=True):
    scene = person_scene(steps=2)
    task = build_shared_exercise_task(scene, root_assistance=assistance)
    return NoPhysicsEnv(scene, task)


def sample_env(features, *, assistance=True):
    env = metadata(assistance)
    env.calls = 0

    def read():
        env.calls += 1
        return features.copy()

    env.feature_matrix = read
    env.observation_contract = lambda: {"profile": "shared-exercise-observation-v1"}
    return env


def sample_policy(env):
    _, _, _, _, metrics = metric_fixture(complete=True)
    rank = optimization_rank([{"status": "completed", "metrics": metrics}] * 2)
    return actor.make_policy(env, np.zeros((3, 2)), 1729, rank["reward_score"], rank, actor.runtime_hashes(env.scene))


@pytest.mark.parametrize("assistance", [False, True])
def test_actor_reuses_v1_groups_and_one_feature_sample_with_exact_zero_anchor(assistance):
    count = 22 if assistance else 19
    features = np.random.default_rng(19).normal(0., .2, (count, 3))
    features[-1] = -0.
    env = sample_env(features, assistance=assistance)
    groups = actor.joint_groups(env.action_names)
    np.testing.assert_array_equal(groups, gains._groups(env.action_names[:19]))
    result, seen, authored, delta = actor.action_sample(np.zeros((3, 2)), groups, env)
    expected = np.clip(features.sum(axis=1), -1., 1.)
    assert result.tobytes() == authored.tobytes() == expected.tobytes()
    assert env.calls == 1 and not np.any(delta)
    result[:] = 1
    assert features.tobytes() == seen.tobytes()


def test_feedback_is_restricted_gain_correction_not_duplicate_v1_or_phase_actor():
    features = np.full((22, 3), .1)
    env = sample_env(features)
    weights = np.full((3, 2), .05)
    action, _, authored, _ = actor.action_sample(weights, actor.joint_groups(env.action_names), env)
    effective = np.ones((4, 3))
    effective[:3, :2] += weights
    old = np.clip(np.sum(features * effective[gains._groups(env.action_names)], axis=1), -1., 1.)
    np.testing.assert_allclose(action, old, rtol=0, atol=1e-16)
    assert action[19:].tobytes() == authored[19:].tobytes()
    features[:19] = [100., -99.9, 0.]
    changed_env = sample_env(features)
    changed, _, authored, correction = actor.action_sample(weights, actor.joint_groups(env.action_names), changed_env)
    assert np.all(abs(changed - authored) <= actor.RESIDUAL_BOUND + 1e-16)
    assert np.all(correction == 0.)
    # Bounded state features distinguish this actor from unrestricted v1 gain multiplication.
    v1 = np.clip(np.sum(features * effective[gains._groups(env.action_names)], axis=1), -1., 1.)
    assert not np.array_equal(changed, v1)
    for phase in (0., .01, .5, 1.):
        env.observation = lambda: np.asarray([phase])
        np.testing.assert_array_equal(actor.action_sample(weights, actor.joint_groups(env.action_names), env)[0], changed)


@pytest.mark.parametrize("value", [-1e6, -1., 0., 1., 1e6])
def test_extreme_feedback_keeps_delta_and_action_bounded_and_root_exact(value):
    features = np.full((22, 3), value)
    env = sample_env(features)
    action, _, authored, delta = actor.action_sample(np.full((3, 2), .25), actor.joint_groups(env.action_names), env)
    assert np.all(abs(delta) <= .25) and np.all(abs(action) <= 1.)
    assert np.all(abs(action[:19] - authored[:19]) <= .25 + 1e-16)
    assert action[19:].tobytes() == authored[19:].tobytes()


@pytest.mark.parametrize("bad", [np.full((22, 3), np.nan), np.full((22, 3), np.inf), np.zeros((23, 3)),
                               np.ones((22, 3), dtype=bool), np.full((22, 3), 1e6 + 1)])
def test_actor_rejects_invalid_feature_matrices(bad):
    env = sample_env(bad)
    with pytest.raises(ValueError, match="feature matrix"):
        actor.action_sample(np.zeros((3, 2)), actor.joint_groups(env.action_names), env)


def test_prepared_policy_detaches_and_does_not_hash_or_query_features_twice(monkeypatch):
    env = sample_env(np.full((22, 3), .1))
    policy = sample_policy(env)
    action = actor.prepare_shared_exercise_feedback_policy(policy, env)
    policy["feedback_coefficients"][0][0] = .25
    monkeypatch.setattr(actor, "runtime_hashes", lambda *_: pytest.fail("Hot-path runtime rehash"))
    first = action()
    assert env.calls == 1
    first[:] = 0
    np.testing.assert_array_equal(action(), np.full(22, .1).reshape(22) * 3)
    assert env.calls == 2


@pytest.mark.parametrize("field,value", [
    ("profile", "shared-exercise-policy-v1"), ("actor_profile", "phase"), ("gates_sha256", "0" * 64),
    ("runtime_sha256", {}), ("scene_sha256", "0" * 64), ("feature_names", ["future_evaluation"]),
    ("groups", ["root"]), ("root_assistance", "disabled"), ("residual_bound", .5), ("feature_bound", True),
    ("feedback_coefficients", [[True, 0.]] * 3), ("feedback_coefficients", [[.26, 0.]] * 3),
    ("feedback_coefficients", [[float("nan"), 0.]] * 3), ("feedback_coefficients", [[0., 0.]] * 4),
    ("privileged_simulator_features", False), ("selection_objective", "evaluation_reward"),
])
def test_new_profile_rejects_changed_bounds_channels_privilege_and_identity(field, value):
    env = metadata()
    policy = sample_policy(env)
    policy[field] = value
    with pytest.raises(ValueError):
        actor.validate_shared_exercise_feedback_policy(policy, env.scene, env.task)


@pytest.fixture(scope="module")
def tiny_composition(tmp_path_factory):
    root = tmp_path_factory.mktemp("feedback-composition")
    source, trained, selected, evaluated = [root / name for name in ("source", "trained", "selected", "evaluated")]
    source.mkdir()
    scene = person_scene(steps=2)
    task = build_shared_exercise_task(scene, root_assistance=True)
    write_json(source / "scene.json", scene)
    write_json(source / "exercise-task.json", task)
    write_json(source / "inventory.json", scene["inventory"])
    write_json(source / "source-clock.json", {"scope": "Analytic fixture, not source media"})
    write_json(source / "manifest.json", {"status": "ready", "files": {p.name: digest(p) for p in source.iterdir()}})
    before = {p.name: p.read_bytes() for p in source.iterdir()}
    ast_before = {name: inspect.getsource(getattr(legacy, name)) for name in ("_trial", "_action", "phase_basis", "_coefficients")}
    counter = {"steps": 0}
    old = SharedExerciseTaskEnv.step

    def bounded(self, action):
        assert counter["steps"] < 24
        counter["steps"] += 1
        return old(self, action)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(SharedExerciseTaskEnv, "step", bounded)
        training = learner.train_shared_exercise_feedback_policy(source, trained, seed=1931, generations=1, population=4)
        selection = learner.select_shared_exercise_feedback_policy(trained, selected)
        evaluation = learner.evaluate_shared_exercise_feedback_policy(selected, evaluated, seed=937321, trials=1)
    assert {p.name: p.read_bytes() for p in source.iterdir()} == before
    assert ast_before == {name: inspect.getsource(getattr(legacy, name)) for name in ast_before}
    return SimpleNamespace(root=root, source=source, trained=trained, selected=selected, evaluated=evaluated,
        scene=scene, task=task, training=training, selection=selection, evaluation=evaluation, actions=counter["steps"])


def test_tiny_native_training_selection_evaluation_preserves_source_and_full_interval(tiny_composition):
    c = tiny_composition
    assert c.training["status"] == "trained" and c.training["control_steps"] == 16
    assert c.training["evaluated_episodes"] == 8 and c.training["completed_generations"] == 1
    assert c.selection["new_control_steps"] == 0 and c.selection["unique_candidates"] == 4
    assert c.selection["policy_sha256"] == c.training["policy_sha256"]
    assert c.evaluation["status"] == "evaluated" and c.evaluation["comparison_complete"]
    assert c.evaluation["evaluation_feedback_to_optimizer"] is False and c.actions == 24
    assert set(c.evaluation["controllers"]) == set(learner.CONTROLLERS)
    assert not set(c.evaluation["training_seeds"]) & set(c.evaluation["evaluation_seeds"])
    for path in (c.trained, c.selected, c.evaluated):
        verify_artifacts(path)
        assert (path / "scene.json").read_bytes() == (c.source / "scene.json").read_bytes()
        assert (path / "exercise-task.json").read_bytes() == (c.source / "exercise-task.json").read_bytes()
    for name in ("policy.json", "training.json", "settings.json"):
        assert (c.selected / name).read_bytes() == (c.trained / name).read_bytes()
    assert (c.selected / "policy.json").read_bytes() == (c.evaluated / "policy.json").read_bytes()


def load_package(root):
    manifest = (root / "manifest.json").read_bytes()
    files = {name: (root / name).read_bytes() for name in json.loads(manifest)["files"]}
    return manifest, files


def reseal_memory(files):
    return json.dumps({"status": "trained", "files": {name: learner._sha(raw) for name, raw in files.items()}}).encode()


@pytest.mark.parametrize("mutation", ["sample", "rank", "count", "diagnostic_action", "diagnostic_peak", "diagnostic_missing",
                                    "selection", "original_manifest", "runtime", "failure"])
def test_saved_verifier_rejects_self_consistent_resealed_tampering_without_physics(tiny_composition, monkeypatch, mutation):
    c = tiny_composition
    _, files = load_package(c.selected)
    monkeypatch.setattr(learner, "SharedExerciseTaskEnv", lambda *_: pytest.fail("Saved verifier ran physics"))
    if mutation in {"sample", "rank", "count"}:
        doc = json.loads(files["training.json"])
        if mutation == "sample":
            doc["history"][0]["candidates"][1]["feedback_coefficients"][0][0] += .001
        elif mutation == "rank":
            doc["history"][0]["candidates"][1]["optimization_rank"]["max_normalized_excess"] += .1
        else:
            doc["control_steps"] -= 1
        files["training.json"] = learner._bytes(doc)
    elif mutation.startswith("diagnostic"):
        doc = json.loads(files["training.json"])
        receipt = doc["history"][0]["candidates"][0]["trials"][0]["diagnostics"]
        name = receipt["artifact"]
        if mutation == "diagnostic_missing":
            files.pop(name)
        else:
            lines = [json.loads(row) for row in files[name].splitlines()]
            if mutation == "diagnostic_action":
                lines[0]["action"][19] = .7
            else:
                lines[-1]["peak_angular_rmse_rad"] += .1
            files[name] = b"".join((json.dumps(row) + "\n").encode() for row in lines)
            receipt["sha256"] = learner._sha(files[name])
            files["training.json"] = learner._bytes(doc)
    elif mutation == "selection":
        doc = json.loads(files["selection.json"]);doc["feasible_candidates"] += 1
        files["selection.json"] = learner._bytes(doc)
    elif mutation == "original_manifest":
        doc = json.loads(files["selection-input-manifest.json"]);doc["status"] = "failed"
        files["selection-input-manifest.json"] = learner._bytes(doc)
    elif mutation == "runtime":
        files["implementation/shared_exercise_feedback_residual.py"] += b"\n"
    else:
        files["failure.json"] = b'{}'
    with pytest.raises(ValueError):
        learner.verify_shared_exercise_feedback_package(reseal_memory(files), files, c.scene, c.task, deadline=time.monotonic() + 10)


def test_diagnostics_cover_all_training_controls_without_recomputing_features(tiny_composition):
    c = tiny_composition
    manifest, files = load_package(c.trained)
    proof = learner.verify_shared_exercise_feedback_package(manifest, files, c.scene, c.task, deadline=time.monotonic() + 10)
    assert proof["diagnostic_files"] == 8
    lines = [json.loads(line) for name, raw in files.items() if name.startswith("diagnostics/") for line in raw.splitlines()]
    assert sum(row["type"] == "attempt" for row in lines) == sum(row["type"] == "result" for row in lines) == 16
    assert all(len(row["feedback_features"]) == 22 for row in lines if row["type"] == "attempt")


def test_output_and_diagnostic_reservations_fail_before_write(tmp_path, monkeypatch):
    out = tmp_path / "out";out.mkdir()
    writer = learner._Writer(out, time.monotonic() + 10)
    monkeypatch.setattr(learner, "MAX_OUTPUT_BYTES", learner.TERMINAL_RESERVE + 100)
    with pytest.raises(ValueError, match="reservation"):
        writer.write("settings.json", {"large": "x" * 101})
    assert not list(out.iterdir())
    monkeypatch.setattr(learner, "MAX_OUTPUT_BYTES", 128 * 1024 * 1024)
    monkeypatch.setattr(learner, "MAX_DIAGNOSTIC_BYTES", 10)
    with pytest.raises(ValueError, match="diagnostics"):
        writer.diagnostic("diagnostics/test.jsonl", {"large": "x" * 11})
    assert not list(out.iterdir())


def test_full_horizon_budget_and_prior_evaluation_reject_before_output(tmp_path, monkeypatch):
    source = tmp_path / "source";source.mkdir()
    scene = person_scene(steps=400)
    write_json(source / "scene.json", scene)
    write_json(source / "manifest.json", {"status": "ready", "files": {"scene.json": digest(source / "scene.json")}})
    monkeypatch.setattr(learner, "SharedExerciseTaskEnv", lambda *_: pytest.fail("Preflight created physics"))
    with pytest.raises(ValueError, match="8640"):
        learner.train_shared_exercise_feedback_policy(source, tmp_path / "out", generations=3, population=4)
    assert not (tmp_path / "out").exists()


def test_old_v2_validator_cannot_reauthorize_new_v4_policy():
    env = metadata()
    policy = sample_policy(env)
    with pytest.raises(ValueError, match="contract"):
        legacy.validate_shared_exercise_residual_policy(policy, env.scene, env.task)


@pytest.mark.parametrize("reset_fails", [False, True])
def test_diagnostics_reuse_only_returned_observations_and_one_feature_query(tmp_path, reset_fails):
    env = metadata()
    env.steps = env.feature_calls = 0
    env.terminated = env.truncated = False

    def observation_for(step):
        values = np.zeros(156, dtype=np.float32)
        values[0] = .1 * (step + 1)
        return values

    def metrics():
        return {"completed_interval": env.steps == 2, "failure": None, "simulated_duration_s": env.steps / 30,
                "observed_substeps": env.steps * env.scene["substeps"],
                "peak_angular_rmse_rad": env.steps * .1, "peak_root_rmse_model_units": 0.}

    def reset(*, seed, options):
        if reset_fails:
            raise ValueError("authored reset failure")
        return observation_for(0), {}

    def features():
        env.feature_calls += 1
        return np.zeros((22, 3))

    def step(_action):
        env.steps += 1
        env.truncated = env.steps == 2
        return observation_for(env.steps), .1, False, env.truncated, metrics()

    env.reset, env.step, env.metrics, env.feature_matrix = reset, step, metrics, features
    env.snapshot = lambda: {"engine_state": {"step": env.steps}}
    env.observation = lambda: pytest.fail("Diagnostics must reuse the reset/step observation")
    out = tmp_path / "diagnostics-owner";out.mkdir()
    writer = learner._Writer(out, time.monotonic() + 10)
    diagnostic = learner._Diagnostics(writer, "diagnostics/episode.jsonl")
    row = learner._trial(env, "learned", 1729, learner.TRAIN_RESETS[0], np.zeros((3, 2)),
        time.monotonic() + 10, legacy._Budget(2), diagnostics=diagnostic, validator=lambda _snapshot: None)
    if reset_fails:
        assert row["status"] == "reset_failed" and env.feature_calls == env.steps == 0
        assert row["diagnostics"]["artifact"] is None and not list(out.iterdir())
    else:
        assert row["control_steps"] == env.feature_calls == env.steps == 2
        attempts = [json.loads(line) for line in (out / diagnostic.name).read_bytes().splitlines()][::2]
        assert [entry["joint_position_error_rad"][0] for entry in attempts] == [float(np.float32(.1)), float(np.float32(.2))]


@pytest.mark.parametrize("change", ["none", "partial_nonterminal", "wrong_duration", "backward_terminal", "partial_failure"])
def test_saved_diagnostic_clocks_require_complete_nonfailed_boundaries(change, monkeypatch):
    monkeypatch.setattr(learner, "SharedExerciseTaskEnv", lambda *_: pytest.fail("Clock validation ran physics"))
    scene = {"dt": 1 / 30, "substeps": 16, "action_contract": {"names": [f"joint_{i}" for i in range(19)]}}
    rows = []
    for index in range(2):
        rows.extend([
            {"type": "attempt", "index": index + 1, "time_s": index * scene["dt"],
             "joint_position_error_rad": [0.] * 19, "feedback_features": [[0., 0., 0.] for _ in range(19)],
             "action": [0.] * 19, "authored_saturated_joints": [], "residual_at_bound_joints": []},
            {"type": "result", "index": index + 1, "reward": .1, "terminated": False, "truncated": index == 1,
             "simulated_duration_s": (index + 1) * scene["dt"], "observed_substeps": (index + 1) * 16,
             "peak_angular_rmse_rad": .1, "peak_root_rmse_model_units": 0.},
        ])
    if change == "partial_nonterminal":
        rows[1].update(observed_substeps=0, simulated_duration_s=0.)
    elif change == "wrong_duration":
        rows[1]["simulated_duration_s"] = scene["dt"] / 2
    elif change == "backward_terminal":
        rows[3].update(observed_substeps=17, simulated_duration_s=0., terminated=True, truncated=False)
    elif change == "partial_failure":
        rows[3].update(observed_substeps=17, simulated_duration_s=scene["dt"], terminated=True, truncated=False)
    final = rows[-1]
    trial = {"control_steps": 2, "return": .2, "terminated": final["terminated"], "truncated": final["truncated"],
             "metrics": {key: final[key] for key in ("simulated_duration_s", "observed_substeps", "peak_angular_rmse_rad", "peak_root_rmse_model_units")}}
    trial["metrics"]["completed_interval"] = change not in {"backward_terminal", "partial_failure"}
    name = "diagnostics/clock.jsonl"
    raw = b"".join((json.dumps(row) + "\n").encode() for row in rows)
    receipt = {"profile": learner.DIAGNOSTIC_PROFILE, "artifact": name, "sha256": learner._sha(raw),
               "attempts": 2, "results": 2, "scope": "Authored clock-validation fixture; no dynamics"}
    call = lambda: learner._verify_diagnostics({name: raw}, receipt, trial, np.zeros((3, 2)), scene, 19, time.monotonic() + 5)
    if change in {"none", "partial_failure"}:
        assert call() == name
    else:
        with pytest.raises(ValueError, match="clock|boundary"):
            call()
