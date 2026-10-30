"""Changed human starts must not launch passive supports or silently rewrite old recordings."""

import numpy as np
import pytest

from weft.io import canonical_hash
from weft.recording import Recorder, load_recording
from weft.shared_exercise_recording import recording_scene, verify_action_replay
from weft.shared_exercise_reset import HUMAN_JITTER_PROFILE, human_jitter_options, resolve_reset_options
from weft.shared_exercise_task import SharedExerciseTaskEnv, build_shared_exercise_task, validate_shared_exercise_snapshot
from test_shared_exercise_task import setup


def object_state(snapshot):
    return {row["id"]: {key: row[key] for key in ("position", "quaternion", "velocity", "angular_velocity")}
            for row in snapshot["engine_state"]["objects"]}


def test_changed_humans_retain_supported_equipment_without_settling_or_retries():
    scene, task, _ = setup(horizon=8)
    env = SharedExerciseTaskEnv(scene, task)
    try:
        env.reset(seed=203)
        nominal = env.snapshot()
        expected = object_state(nominal)
        states = set()
        for seed in (203, 509, 809):
            env.reset(seed=seed, options=human_jitter_options(.1))
            initial = env.snapshot()
            assert object_state(initial) == expected
            assert initial["engine_state"]["steps"] == 0
            assert initial["task_metrics"]["simulated_duration_s"] == 0
            assert initial["task_metrics"]["reset_protocol"]["profile"] == HUMAN_JITTER_PROFILE
            assert initial["task_metrics"]["reset_protocol"]["equipment_reset_state"] == "unchanged_scene_pose_and_velocity"
            assert all(row["speed_model_units_s"] == 0 for row in initial["task_metrics"]["equipment"])
            assert canonical_hash(initial["engine_state"]) != canonical_hash(nominal["engine_state"])
            states.add(canonical_hash(initial["engine_state"]))
            validate_shared_exercise_snapshot(initial, scene, task)
            for _ in range(8):
                _, _, terminated, truncated, _ = env.step(np.zeros(env.action_count))
                if terminated or truncated:
                    break
            # Real coast dynamics must not reproduce the old reset's artificial free fall.
            assert not any(failure.startswith("equipment_stability:") for failure in env.metrics()["failed_gates"])
        assert len(states) == 3
        env.reset(seed=203, options=human_jitter_options(.1))
        assert canonical_hash(env.snapshot()["engine_state"]) in states
        assert env.task == task and env.scene == scene
    finally:
        env.close()


def test_human_only_reset_preserves_rotated_equipment_with_linear_and_angular_velocity():
    scene, _, _ = setup()
    scene["objects"][0]["velocity"] = [.03, -.02, 0.]
    scene["objects"][0]["quaternion"] = [float(np.cos(.2)), 0., 0., float(np.sin(.2))]
    scene["objects"][0]["angular_velocity"] = [.03, -.02, .04]
    task = build_shared_exercise_task(scene)
    env = SharedExerciseTaskEnv(scene, task)
    try:
        env.reset(seed=301)
        nominal = object_state(env.snapshot())
        env.reset(seed=301, options=human_jitter_options(.1))
        assert object_state(env.snapshot()) == nominal
        assert any(row["speed_model_units_s"] > 0 for row in env.metrics()["equipment"])
        validate_shared_exercise_snapshot(env.snapshot(), scene, task)
    finally:
        env.close()


def test_original_reset_keeps_its_distribution_and_exact_option_shape():
    scene, task, _ = setup()
    env = SharedExerciseTaskEnv(scene, task)
    try:
        env.reset(seed=19)
        nominal = object_state(env.snapshot())
        options = {"jitter": True, "jitter_scale": .1}
        env.reset(seed=19, options=options)
        state = env.snapshot()
        assert state["resolved_reset_options"] == options
        assert state["task_metrics"]["reset_protocol"]["profile"] == "shared-exercise-reset-v1"
        assert state["task_metrics"]["reset_protocol"]["object_velocity_jitter_world_s"] == .12
        assert object_state(state) != nominal
        env.reset(seed=19, options=options)
        assert env.snapshot() == state
    finally:
        env.close()


@pytest.mark.parametrize("options", [
    {"profile": "unversioned"}, {"profile": None}, {"profile": HUMAN_JITTER_PROFILE, "velocity_jitter": .1},
    {"profile": HUMAN_JITTER_PROFILE, "jitter": True, "jitter_scale": float("nan")},
    {"profile": HUMAN_JITTER_PROFILE, "jitter": False, "jitter_scale": .1},
    {"profile": HUMAN_JITTER_PROFILE, "jitter": 1},
])
def test_unknown_or_ambiguous_reset_protocol_rejects(options):
    with pytest.raises(ValueError):
        resolve_reset_options(options)


def test_snapshot_cannot_relabel_equipment_perturbation_as_human_only():
    scene, task, _ = setup()
    env = SharedExerciseTaskEnv(scene, task)
    try:
        env.reset(seed=511, options={"jitter": True, "jitter_scale": .1})
        state = env.snapshot()
        state["resolved_reset_options"]["profile"] = HUMAN_JITTER_PROFILE
        with pytest.raises(ValueError, match="equipment reset"):
            validate_shared_exercise_snapshot(state, scene, task)
    finally:
        env.close()


def test_human_only_snapshot_cannot_switch_actor_without_applying_force():
    scene, task, _ = setup()
    env = SharedExerciseTaskEnv(scene, task)
    try:
        env.reset(seed=557, options=human_jitter_options(.1))
        state = env.snapshot()
        state["engine_state"]["controlled_id"] = scene["objects"][0]["id"]
        state["engine_state"]["objects"][0]["role"] = "active"
        with pytest.raises(ValueError, match="only controlled actor"):
            validate_shared_exercise_snapshot(state, scene, task)
    finally:
        env.close()


def test_new_and_original_resets_share_an_exact_action_recording(tmp_path):
    scene, task, _ = setup(horizon=3, root=True)
    env = SharedExerciseTaskEnv(scene, task)
    recorder = Recorder(tmp_path / "session", recording_scene(scene, task), profile="shared-exercise")
    try:
        env.reset(seed=433, options=human_jitter_options(.1))
        recorder.start(env.snapshot(), seed=433)
        for seed, options in ((439, {"jitter": True, "jitter_scale": .1}), (443, human_jitter_options(.1))):
            action = env.authored_action()
            _, reward, terminated, truncated, _ = env.step(action)
            recorder.append({"type": "step", "action": action.tolist(), "reward": reward,
                             "terminated": terminated, "truncated": truncated, "snapshot": env.snapshot()})
            env.reset(seed=seed, options=options)
            recorder.append({"type": "reset", "seed": seed, "snapshot": env.snapshot()})
        recorder.close()
        assert verify_action_replay(recorder.root)["exact_state_replay"]
        events = load_recording(recorder.root)["events"]
        assert events[0]["snapshot"]["resolved_reset_options"] == human_jitter_options(.1)
        assert "profile" not in events[2]["snapshot"]["resolved_reset_options"]
        assert events[4]["snapshot"]["resolved_reset_options"] == human_jitter_options(.1)
    finally:
        recorder.close()
        env.close()
