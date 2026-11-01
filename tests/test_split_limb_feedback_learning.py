"""Explicit profile routing preserves full search proofs with authored, nonphysical trial data."""

from copy import deepcopy
import json
import time

import mujoco
import numpy as np
import pytest

from weft import shared_exercise_feedback_learning as learner
from weft import shared_exercise_feedback_residual as actor
from weft import shared_exercise_task as tasks
from weft import shared_world, spatial_rigid
from weft.io import digest, write_json
from weft.shared_exercise_reset import describe_reset_protocol
from test_shared_exercise_feedback_lab import no_simulation
from test_shared_exercise_selection import NoPhysicsEnv
from test_shared_world_human_only import person_scene


@pytest.fixture(autouse=True)
def analytical_metadata(monkeypatch, no_simulation):
    def identity_rotation(quaternion):
        assert list(quaternion) == [1., 0., 0., 0.]
        return np.eye(3)

    # These authored fixtures need identity corners and a mass label, not model geometry.
    monkeypatch.setattr(spatial_rigid, "rotation", identity_rotation)
    monkeypatch.setattr(shared_world, "human_subtree_mass", lambda _spec, scale=1.: 42. * scale ** 3)


class AnalyticalEnv(NoPhysicsEnv):
    """Authored scores exercise packaging, not dynamics or learned physical performance."""

    total_steps = 0
    feature_reads = 0

    def reset(self, *, seed, options):
        self.seed, self.reset_options, self.steps = seed, deepcopy(options), 0
        self.terminated = self.truncated = False
        self.measured = tasks._ExerciseMetrics(self.scene, self.task, (), 0., 1.)
        self.action = np.zeros(self.action_count)
        return np.zeros(156, dtype=np.float32), {}

    def feature_matrix(self):
        type(self).feature_reads += 1
        return np.column_stack((np.linspace(.05, .25, self.action_count),
                                np.full(self.action_count, -.1), np.zeros(self.action_count)))

    def snapshot(self):
        return {"engine_state": {"seed": self.seed, "step": self.steps, "action": self.action.tolist()}}

    def metrics(self):
        return self.measured.metrics() | {"scene_sha256": self.scene_sha256, "task_sha256": self.task_sha256,
            "evaluator_sha256": self.evaluator_sha256, "reset_protocol": describe_reset_protocol(self.reset_options)}

    def step(self, action):
        type(self).total_steps += 1
        self.steps += 1
        self.action = action.copy()
        duration = self.steps * self.scene["dt"]
        self.measured.samples = self.measured.valid_samples = self.steps * self.scene["substeps"]
        self.measured.elapsed = self.measured.valid_duration = duration
        self.measured.angular_squared = .2 ** 2 * duration
        self.measured.root_squared = .03 ** 2 * duration
        self.measured.angular_peak, self.measured.root_peak = .2, .03
        reward = .7 + float(np.sum(action[:19])) * .001
        self.measured.reward_integral += reward * self.scene["dt"]
        metrics = self.metrics()
        self.terminated = metrics["success"]
        self.truncated = metrics["completed_interval"] and not self.terminated
        return np.zeros(156, dtype=np.float32), reward, self.terminated, self.truncated, metrics


@pytest.fixture
def analytical_training(tmp_path, monkeypatch, no_simulation):
    def forbidden(*args, **kwargs):
        pytest.fail("This fixture cannot compile or advance native physics")

    monkeypatch.setattr(mujoco, "MjModel", forbidden)
    monkeypatch.setattr(mujoco, "MjData", forbidden)
    monkeypatch.setattr(learner, "SharedExerciseTaskEnv", AnalyticalEnv)
    monkeypatch.setattr(learner, "prepare_shared_exercise_snapshot_validator", lambda *args: lambda snapshot: None)
    AnalyticalEnv.total_steps = AnalyticalEnv.feature_reads = 0
    source = tmp_path / "source"
    source.mkdir()
    scene = person_scene(steps=2)
    task = tasks.build_shared_exercise_task(scene, root_assistance=True)
    for name, value in {"scene.json": scene, "exercise-task.json": task, "inventory.json": scene["inventory"]}.items():
        write_json(source / name, value)
    write_json(source / "manifest.json", {"status": "ready", "files": {p.name: digest(p) for p in source.iterdir()}})
    return source, scene, task


@pytest.mark.parametrize("profile", [actor.POLICY_PROFILE, actor.SPLIT_POLICY_PROFILE])
def test_full_train_select_evaluate_proof_uses_explicit_shape(analytical_training, tmp_path, profile):
    source, scene, task = analytical_training
    original = {p.name: p.read_bytes() for p in source.iterdir()}
    trained, selected, evaluated = [tmp_path / name for name in ("trained", "selected", "evaluated")]
    train = learner.train_shared_exercise_feedback_policy(source, trained, seed=1931,
        generations=1, population=4, policy_profile=profile)
    settings = json.loads((trained / "settings.json").read_bytes())
    policy = json.loads((trained / "policy.json").read_bytes())
    contract = actor.contract(profile)
    assert policy["profile"] == profile
    assert settings["coefficient_shape"] == list(contract["coefficient_shape"])
    assert settings["groups"] == list(contract["groups"])
    assert train["control_steps"] == AnalyticalEnv.total_steps == AnalyticalEnv.feature_reads == 16
    selection = learner.select_shared_exercise_feedback_policy(trained, selected)
    assert selection["new_control_steps"] == 0 and AnalyticalEnv.total_steps == 16
    assert (selected / "policy.json").read_bytes() == (trained / "policy.json").read_bytes()
    result = learner.evaluate_shared_exercise_feedback_policy(selected, evaluated, seed=937321, trials=1)
    assert result["control_steps"] == 8 and AnalyticalEnv.total_steps == 24
    assert AnalyticalEnv.feature_reads == 20
    assert set(result["training_seeds"]).isdisjoint(result["evaluation_seeds"])
    assert result["scope"] == contract["scope"] and result["policy_unchanged"] is True
    assert {p.name: p.read_bytes() for p in source.iterdir()} == original
    for controller in ("authored", "learned"):
        actions = result["controllers"][controller]["trials"][0]["actions"]
        assert all(len(row["action"]) == 22 for row in actions)
    files = {p.relative_to(trained).as_posix(): p.read_bytes() for p in trained.rglob("*") if p.is_file()}
    proof = learner.verify_shared_exercise_feedback_training(files, scene, task, deadline=time.monotonic() + 10)
    assert proof["control_steps"] == 16 and proof["diagnostic_files"] == 8


@pytest.mark.parametrize("mutation", ["groups", "shape", "candidate", "action"])
def test_split_saved_proof_rejects_profile_and_trace_mismatch(analytical_training, tmp_path, mutation):
    source, scene, task = analytical_training
    out = tmp_path / "trained"
    learner.train_shared_exercise_feedback_policy(source, out, seed=1931,
        generations=1, population=4, policy_profile=actor.SPLIT_POLICY_PROFILE)
    files = {p.relative_to(out).as_posix(): p.read_bytes() for p in out.rglob("*") if p.is_file()}
    if mutation in ("groups", "shape"):
        settings = json.loads(files["settings.json"])
        settings["groups" if mutation == "groups" else "coefficient_shape"] = list(
            actor.contract()["groups" if mutation == "groups" else "coefficient_shape"])
        files["settings.json"] = learner._bytes(settings)
    else:
        training = json.loads(files["training.json"])
        candidate = training["history"][0]["candidates"][1]
        if mutation == "candidate":
            candidate["feedback_coefficients"] = candidate["feedback_coefficients"][:3]
        else:
            receipt = candidate["trials"][0]["diagnostics"]
            lines = files[receipt["artifact"]].splitlines(keepends=True)
            attempt = json.loads(lines[0])
            attempt["action"][3] += .01
            lines[0] = json.dumps(attempt).encode() + b"\n"
            raw = b"".join(lines)
            files[receipt["artifact"]] = raw
            receipt["sha256"] = learner._sha(raw)
        files["training.json"] = learner._bytes(training)
    with pytest.raises(ValueError, match="settings|candidate|action"):
        learner.verify_shared_exercise_feedback_training(files, scene, task, deadline=time.monotonic() + 10)


@pytest.mark.parametrize("profile", [None, True, [], {}, "shared-exercise-policy-v6"])
def test_invalid_profile_fails_before_source_io(tmp_path, monkeypatch, profile):
    monkeypatch.setattr(learner, "_source", lambda *a, **kw: pytest.fail("Invalid profile read source"))
    with pytest.raises(ValueError):
        learner.train_shared_exercise_feedback_policy("missing", tmp_path / "out", policy_profile=profile)
    assert not (tmp_path / "out").exists()
