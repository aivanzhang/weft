"""Support wrapper/recording checks: analytical samples plus one three-action native journal."""

from copy import deepcopy
import hashlib
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from test_shared_support_task import contact, initial_supports, support_fixture
from test_shared_world_support import support_scene
from weft import shared_exercise_recording as recording
from weft import shared_exercise_task as tasks
from weft import shared_support_task as support
from weft.io import canonical_hash
from weft.shared_world_observation import SubstepSample


class AnalyticalEngine:
    def __init__(self, scene, render_mode=None, *, substep_observer):
        self.scene = deepcopy(scene)
        self.observer = substep_observer
        self.human_mass = 1.
        self.human_action_names = self.action_names = tuple(f"joint_{index}" for index in range(22))
        self.closed = False
        self.steps = 0
        self.action = [0.] * 22

    def reset(self, *, seed=None, options=None):
        self.steps = 0
        self.action = [0.] * 22
        return np.zeros(168), {"seed": seed}

    def snapshot(self):
        return {"controlled_id": "human", "target": None, "steps": self.steps, "episode_steps": self.steps,
                "time_s": self.steps * self.scene["dt"], "terminated": False,
                "truncated": self.steps == self.scene["task"]["horizon_steps"],
                "normalized_action": list(self.action),
                "objects": [{"id": row["id"], "position": list(row["position"]), "quaternion": list(row["quaternion"]),
                             "velocity": [0., 0., 0.], "angular_velocity": [0., 0., 0.], "virtual_force": [0., 0., 0.]}
                            for row in self.scene["objects"]],
                "human": {"mobility": {"max_penetration": 0.}}}

    def step(self, action):
        self.action = list(map(float, action))
        count = self.scene["substeps"]
        tick = self.scene["dt"] / count
        for index in range(count):
            time = (self.steps * count + index) * tick
            self.observer(SubstepSample(self.steps + 1, index + 1, time, time + tick, time, True, None,
                                       (0.,) * 22, (0.,) * 22, (0.,) * 22, tuple(self.action),
                                       initial_supports(self.scene), (contact(),), {}, 0.))
        self.steps += 1
        return np.zeros(168), 0., False, self.steps == self.scene["task"]["horizon_steps"], {}

    def authored_human_action(self):
        return np.zeros(22)

    def feature_matrix(self):
        return np.zeros((22, 5))

    def close(self):
        self.closed = True


@pytest.fixture
def analytical(monkeypatch):
    scene, _ = support_fixture()
    scene["action_contract"] = {"units": ["torque"] * 19 + ["force"] * 3, "physical_limits": [30.] * 22}
    task = support.build_shared_support_task(scene, support_requirements=[{
        "object_id": "support_a", "human_segment": "left_forearm", "direction_world": [0., 0., 1.],
        "minimum_load_body_weight_fraction": .05, "start_step": 0, "end_step": 2}])
    monkeypatch.setattr(tasks, "validate_shared_world", deepcopy)
    monkeypatch.setattr(tasks, "SharedWorldEnv", AnalyticalEngine)
    monkeypatch.setattr(tasks, "prepare_shared_snapshot_validator", lambda scene: deepcopy)
    monkeypatch.setattr(recording, "validate_shared_world", deepcopy)
    return scene, task


def open_analytical(case, *, root=False):
    scene, task = deepcopy(case)
    if root:
        task = support.build_shared_support_task(scene, support_requirements=task["support_requirements"], root_assistance=True)
    env = tasks.SharedExerciseTaskEnv(scene, task)
    env.reset(seed=11, options={"jitter": False, "jitter_scale": 0.})
    return scene, task, env


def test_builder_dispatch_is_explicit_and_legacy_requirements_do_not_bleed(analytical):
    scene, task = analytical
    assert tasks.build_shared_exercise_task(scene, support_requirements=task["support_requirements"]) == task
    for kwargs in ({}, {"equipment_ids": []}, {"contact_requirements": []}):
        with pytest.raises(ValueError):
            tasks.build_shared_exercise_task(scene, **kwargs)
    legacy = deepcopy(scene)
    legacy["profile"] = "shared-world-v3"
    with pytest.raises(ValueError, match="explicit static-support"):
        tasks.build_shared_exercise_task(legacy, support_requirements=task["support_requirements"])


@pytest.mark.parametrize("root", [False, True])
def test_existing_environment_uses_support_metrics_and_bounded_public_actions(analytical, root):
    scene, task, env = open_analytical(analytical, root=root)
    before = deepcopy((scene, task))
    try:
        assert env.action_count == (22 if root else 19)
        assert env.observation().shape == (162,)
        assert env.observation_contract()["profile"] == "shared-support-observation-v1"
        assert env.action_contract()["object_force_channels"] == []
        assert env.authored_action().shape == (env.action_count,)
        assert env.feature_matrix().shape == (env.action_count, 5)
        initial = env.snapshot()
        validate = tasks.prepare_shared_exercise_snapshot_validator(scene, task)
        assert validate(initial) == initial
        assert initial["task_metrics"]["clock_status"] == "not_observed"
        assert initial["task_metrics"]["support_requirements"][0]["load_body_weight_fraction"] is None
        assert initial["task_metrics"]["reset_protocol"]["object_position_jitter_world"] == 0.
        with pytest.raises(ValueError):
            env.step(np.zeros(env.action_count + 1))
        for _ in range(2):
            observation, reward, terminated, truncated, _ = env.step(np.zeros(env.action_count))
            assert validate(env.snapshot()) == env.snapshot()
        assert terminated and not truncated and reward == pytest.approx(1.)
        assert observation[155:161].tolist() == pytest.approx([1., 1., .1, 1., 0., 0.])
        assert env.metrics()["observed_substeps"] == 16
        assert env.metrics()["success"] and (scene, task) == before
        with pytest.raises(RuntimeError):
            env.step(np.zeros(env.action_count))
    finally:
        env.close()
    assert env._engine.closed


def test_support_evaluator_binds_both_files_and_rejects_incomplete_maps():
    import pathlib
    expected = {name: hashlib.sha256(pathlib.Path(module.__file__).read_bytes()).hexdigest()
                for name, module in (("shared_exercise_task.py", tasks), ("shared_support_task.py", support))}
    assert tasks._support_evaluator_sha256() == support.evaluator_identity(expected) == canonical_hash(expected)
    for changes in ({}, {"shared_exercise_task.py": "a" * 64}, {**expected, "extra.py": "a" * 64},
                    {**expected, "shared_support_task.py": "A" * 64}):
        with pytest.raises(ValueError):
            support.evaluator_identity(changes)


@pytest.mark.parametrize("change", [
    "success", "completed", "equipment", "evaluator", "gate", "scope", "direction", "threshold", "row_missing",
    "fraction", "satisfied", "active", "load_unknown", "peak", "clock", "counts", "reward", "root_effort",
    "failed_gates", "static_pose", "root_action", "action_bool", "extra_metric", "nan",
    "unqualified_without_gap", "gap_without_unqualified_time",
])
def test_support_snapshot_rejects_forged_identity_load_time_or_success(analytical, change):
    scene, task, env = open_analytical(analytical)
    try:
        env.step(np.zeros(19))
        snapshot = env.snapshot()
        validate = tasks.prepare_shared_exercise_snapshot_validator(scene, task)
        assert validate(snapshot) == snapshot
        metrics = snapshot["task_metrics"]
        row = metrics["support_requirements"][0]
        if change == "success": metrics["success"] = True
        elif change == "completed": metrics["completed_interval"] = True
        elif change == "equipment": metrics["equipment"] = [{"object_id": "fake"}]
        elif change == "evaluator": metrics["evaluator_sha256"] = "0" * 64
        elif change == "gate": metrics["gates_sha256"] = "0" * 64
        elif change == "scope": metrics["source_scope"]["source_contact_verified"] = True
        elif change == "direction": row["direction_world"] = [1., 0., 0.]
        elif change == "threshold": row["minimum_load_body_weight_fraction"] = .001
        elif change == "row_missing": metrics["support_requirements"] = []
        elif change == "fraction": row["qualifying_fraction"] = 1.
        elif change == "satisfied": row["satisfied"] = True
        elif change == "active": row["active_now"] = False
        elif change == "load_unknown": row["load_body_weight_fraction"] = None
        elif change == "peak": row["peak_load_body_weight_fraction"] = .01
        elif change == "clock": metrics["clock_status"] = "not_observed"
        elif change == "counts": metrics["valid_substeps"] += 1
        elif change == "reward": metrics["reward_integral"] = 10.
        elif change == "root_effort": metrics["root_assistance_mean_squared"] = .01
        elif change == "failed_gates": metrics["failed_gates"] = []
        elif change == "static_pose": snapshot["engine_state"]["objects"][0]["position"][2] += .01
        elif change == "root_action": snapshot["engine_state"]["normalized_action"][19] = .1
        elif change == "action_bool": snapshot["last_action"][0] = False
        elif change == "extra_metric": metrics["measured_grasp"] = True
        elif change == "unqualified_without_gap": row["qualifying_now"] = False
        elif change == "gap_without_unqualified_time": row["max_gap_s"] = .001
        else: row["load_body_weight_fraction"] = float("nan")
        with pytest.raises(ValueError):
            validate(snapshot)
    finally:
        env.close()


def test_prepared_support_validator_has_no_repeated_scene_hash_or_source_reads(analytical, monkeypatch):
    scene, task, env = open_analytical(analytical)
    try:
        validate = tasks.prepare_shared_exercise_snapshot_validator(scene, task)
        snapshot = env.snapshot()
        def unexpected(*args, **kwargs):
            raise AssertionError("Preparation repeated in per-snapshot work")
        monkeypatch.setattr(tasks, "validate_shared_world", unexpected)
        monkeypatch.setattr(tasks, "validate_shared_exercise_task", unexpected)
        monkeypatch.setattr(tasks, "canonical_hash", unexpected)
        monkeypatch.setattr(tasks, "_support_evaluator_sha256", unexpected)
        assert validate(snapshot) == snapshot
    finally:
        env.close()


def test_unobserved_tracking_cannot_be_fabricated_as_zero(analytical):
    scene, task, env = open_analytical(analytical)
    try:
        snapshot = env.snapshot()
        snapshot["task_metrics"]["peak_angular_rmse_rad"] = 0.
        with pytest.raises(ValueError, match="fabricated zero"):
            tasks.validate_shared_exercise_snapshot(snapshot, scene, task)
    finally:
        env.close()


def test_delayed_support_window_and_completed_failure_remain_explicit(analytical):
    scene, task = deepcopy(analytical)
    task["support_requirements"][0].update(start_step=1, minimum_load_body_weight_fraction=.2)
    env = tasks.SharedExerciseTaskEnv(scene, task)
    try:
        env.reset(seed=11)
        env.step(np.zeros(19))
        first = env.snapshot()
        assert first["task_metrics"]["support_requirements"][0]["load_body_weight_fraction"] is None
        assert tasks.validate_shared_exercise_snapshot(first, scene, task) == first
        _, _, terminated, truncated, _ = env.step(np.zeros(19))
        final = env.snapshot()
        assert not terminated and truncated and final["task_metrics"]["completed_interval"]
        assert not final["task_metrics"]["success"]
        assert tasks.validate_shared_exercise_snapshot(final, scene, task) == final
    finally:
        env.close()


def test_legacy_task_and_recording_imports_do_not_expand_support_dependency_closure():
    source = str(Path(__file__).parents[1] / "src")
    script = '''import builtins, sys
sys.path.insert(0, SOURCE)
original = builtins.__import__
def guarded(name, *args, **kwargs):
    names = kwargs.get('fromlist', args[2] if len(args) > 2 else ()) or ()
    if name == 'weft.shared_support_task' or name == 'weft' and 'shared_support_task' in names:
        raise AssertionError('Legacy path imported support implementation')
    return original(name, *args, **kwargs)
builtins.__import__ = guarded
from weft import shared_exercise_task, shared_exercise_recording
assert 'weft.shared_support_task' not in sys.modules
'''.replace("SOURCE", repr(source))
    result = subprocess.run([sys.executable, "-I", "-B", "-c", script], capture_output=True, text=True, timeout=5)
    assert result.returncode == 0, result.stderr


def test_failure_preserves_last_boundary_and_penalizes_once(analytical, monkeypatch):
    scene, task, env = open_analytical(analytical)
    try:
        initial = env.snapshot()
        def broken(action):
            raise RuntimeError("bounded test failure")
        monkeypatch.setattr(env._engine, "step", broken)
        _, reward, terminated, truncated, _ = env.step(np.zeros(19))
        failed = env.snapshot()
        assert failed["engine_state"] == initial["engine_state"]
        assert failed["engine_state_status"] == "last_valid_control_boundary"
        assert terminated and not truncated and reward == -1.
        assert tasks.validate_shared_exercise_snapshot(failed, scene, task) == failed
    finally:
        env.close()


def test_recording_envelope_and_events_retain_support_unknowns_and_original_scene(analytical):
    scene, task, env = open_analytical(analytical)
    try:
        envelope = recording.recording_scene(scene, task)
        assert recording.validate_recording_scene(envelope) == envelope
        initial = env.snapshot()
        recording.validate_event({"type": "start", "seed": 11, "snapshot": initial}, envelope)
        _, reward, terminated, truncated, _ = env.step(np.zeros(19))
        event = {"type": "step", "controller": "coast", "action": [0.] * 19, "reward": reward,
                 "terminated": terminated, "truncated": truncated, "snapshot": env.snapshot()}
        recording.validate_event(event, envelope)
        view = recording.replay_view(envelope, [event])
        assert view["exercise_task"] == task and view["scene"] == scene
        assert view["events"][0]["exercise_metrics"]["support_requirements"][0]["object_id"] == "support_a"
        with pytest.raises(ValueError, match="genuine initial"):
            recording.validate_event({"type": "reset", "seed": 11, "snapshot": env.snapshot()}, envelope)
    finally:
        env.close()


@pytest.mark.parametrize("keyword", ["policy_document", "reference_controller_document"])
def test_unqualified_controller_documents_cannot_enter_support_recordings(analytical, keyword):
    scene, task = analytical
    with pytest.raises(ValueError, match="separate qualification"):
        recording.recording_scene(scene, task, **{keyword: b"{}"})


def test_native_observer_three_actions_reset_and_exact_recording_replay(support_scene, tmp_path):
    from weft.recording import Recorder, load_recording
    scene = deepcopy(support_scene)
    task = tasks.build_shared_exercise_task(scene, support_requirements=[{
        "object_id": scene["objects"][0]["id"], "human_segment": "left_forearm", "direction_world": [0., 0., 1.],
        "minimum_load_body_weight_fraction": .05, "start_step": 0, "end_step": scene["task"]["horizon_steps"]}])
    before = canonical_hash(scene), canonical_hash(task)
    env = tasks.SharedExerciseTaskEnv(scene, task)
    recorder = Recorder(tmp_path / "support-session", recording.recording_scene(scene, task), profile="shared-exercise")
    options = {"jitter": False, "jitter_scale": 0.}
    try:
        env.reset(seed=31, options=options)
        recorder.start(env.snapshot(), seed=31)
        for index in range(3):
            if index == 1:
                env.reset(seed=37, options=options)
                recorder.append({"type": "reset", "seed": 37, "snapshot": env.snapshot()})
            _, reward, terminated, truncated, _ = env.step(np.zeros(19))
            recorder.append({"type": "step", "controller": "coast", "action": [0.] * 19, "reward": reward,
                             "terminated": terminated, "truncated": truncated, "snapshot": env.snapshot()})
        assert env.metrics()["valid_substeps"] == 2 * scene["substeps"]
        assert env.metrics()["support_requirements"][0]["load_body_weight_fraction"] is not None
        assert not env.metrics()["completed_interval"]
        recorder.close()
    finally:
        env.close()
        recorder.close()
    loaded = load_recording(recorder.root)
    assert len(loaded["events"]) == 5
    proof = recording.verify_action_replay(recorder.root, max_seconds=10)
    assert proof["exact_state_replay"] and proof["steps"] == 3 and proof["events"] == 5
    assert (canonical_hash(scene), canonical_hash(task)) == before
