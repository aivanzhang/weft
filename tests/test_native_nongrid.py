"""Synthetic non-grid timing checks preserve source endpoints and protected scoring."""

from copy import deepcopy
from dataclasses import replace

import numpy as np
import pytest

from test_native_clock_runtime import root_fixture
from test_shared_exercise_person_only import motion_sample
from test_shared_world_human_only import person_scene
from weft.native_interval import elapsed_substeps, native_interval
from weft.person_inventory import make_person_inventory
from weft.recording import Recorder, load_recording
from weft.shared_exercise_lab import runtime_files
from weft.shared_exercise_recording import recording_scene, verify_action_replay
from weft.shared_exercise_task import (
    GATES_SHA256, SharedExerciseTaskEnv, _ExerciseMetrics, build_shared_exercise_task,
    validate_shared_exercise_snapshot,
)
from weft.shared_world import validate_shared_snapshot, validate_shared_world
from weft.source_clock import clock_summary


def native_scene(endpoint=.075):
    scene = person_scene()
    _, _, binding = root_fixture()
    body = scene["human_spec"]
    sha = binding["source_sha256"]
    body.update(source_sha256=sha, native_clock=clock_summary(binding))
    for name in ("reference", "root_reference"):
        for index, row in enumerate(body[name]):
            row["time_s"] = index * endpoint / 3
    scene["coordinate_frame"]["clock_binding"].update(
        native_clock=deepcopy(body["native_clock"]), alignment_method="native_pts_elapsed_time_v1",
        source_fps=binding["nominal_fps"], source_time_offset_s=0.)
    scene["inventory"] = make_person_inventory(sha)
    scene["source_binding"] = {
        "mode": "source_qualified", "source_sha256": sha, "total_source_frames": 10,
        "evidence_documents": {"d" * 64: "evidence/human.json"},
        "entity_bindings": [{"world_entity_id": "human", "entity_kind": "articulated_human",
                             "observation_entity_id": sha + ":human:0", "observation_document_sha256": "d" * 64,
                             "provenance_mode": "observed_track", "observed_slots": 4,
                             "missing_slots": 6, "pre_init_slots": 0}],
    }
    marker = native_interval(body, scene["dt"])
    if marker:
        scene["task"].update(native_interval=marker, horizon_steps=marker["full_control_steps"] + 1)
    return validate_shared_world(scene)


def test_exact_endpoint_requires_explicit_native_contract():
    scene = native_scene()
    marker = scene["task"]["native_interval"]
    assert marker["duration_s"] == .075 and marker["terminal_duration_s"] == pytest.approx(1 / 120)
    assert scene["dt"] == 1 / 30 and scene["substeps"] == 16
    assert build_shared_exercise_task(scene)["interval"]["native_interval"] == marker
    assert build_shared_exercise_task(scene)["gates_sha256"] == GATES_SHA256
    assert "native_interval.py" in runtime_files(scene)
    assert "native_interval.py" not in runtime_files(person_scene())
    assert "native_interval" not in native_scene(.1)["task"]
    scene["task"].pop("native_interval")
    with pytest.raises(ValueError, match="grid|integral|whole|30|reference"):
        validate_shared_world(scene)


@pytest.mark.parametrize("mutate", [
    lambda s: s["task"]["native_interval"].update(source_end_s=.1),
    lambda s: s["task"]["native_interval"].update(source_clock_binding_sha256="f" * 64),
    lambda s: s["task"]["native_interval"].update(full_control_steps=True),
    lambda s: s["task"]["native_interval"].update(source_start_s=False),
    lambda s: s["task"].update(horizon_steps=2),
    lambda s: s["task"].update(start_offset_s=.001),
    lambda s: s["human_spec"].pop("native_clock"),
])
def test_native_interval_cannot_hide_truncation_or_change_source(mutate):
    scene = native_scene()
    mutate(scene)
    with pytest.raises(ValueError):
        validate_shared_world(scene)


@pytest.mark.parametrize("endpoint", [.075, .015])
def test_terminal_action_integrates_exact_time_and_records_exact_replay(tmp_path, endpoint):
    scene = native_scene(endpoint)
    task = build_shared_exercise_task(scene)
    env = SharedExerciseTaskEnv(scene, task)
    recorder = Recorder(tmp_path / "session", recording_scene(scene, task), profile="shared-exercise")
    try:
        env.reset(seed=19)
        recorder.start(env.snapshot(), seed=19)
        nominal_tick = env._engine.model.opt.timestep
        source_times = [row["time_s"] for row in scene["human_spec"]["reference"]]
        for index in range(scene["task"]["horizon_steps"]):
            action = env.authored_action()
            obs, reward, terminated, truncated, metrics = env.step(action)
            state = env.snapshot()
            validate_shared_exercise_snapshot(state, scene, task)
            validate_shared_snapshot(state["engine_state"], scene)
            assert env._engine.model.opt.timestep == nominal_tick
            assert bool(terminated or truncated) is (index == scene["task"]["horizon_steps"] - 1)
            recorder.append({"type": "step", "action": action.tolist(), "reward": reward,
                             "terminated": terminated, "truncated": truncated, "snapshot": state})
        assert env._engine.data.time == pytest.approx(endpoint, abs=1e-12)
        assert metrics["declared_duration_s"] == endpoint
        assert metrics["simulated_duration_s"] == pytest.approx(endpoint)
        assert metrics["completed_interval"]
        assert metrics["observed_substeps"] == 16 * scene["task"]["horizon_steps"]
        assert [row["time_s"] for row in env._scene["human_spec"]["reference"]] == source_times
        assert obs[-1] == pytest.approx(1.)
        recorder.close()
        assert verify_action_replay(recorder.root)["exact_state_replay"]
        assert load_recording(recorder.root)["events"][-1]["snapshot"] == state
    finally:
        if recorder._started is not None:
            recorder.close()
        env.close()


def test_duration_integral_uses_terminal_duration_and_rejects_nominal_clock():
    scene = native_scene()
    task = build_shared_exercise_task(scene)
    acc = _ExerciseMetrics(scene, task, (), 0., 1.)
    for index in range(48):
        sample = motion_sample(acc, index)
        start, end = elapsed_substeps(scene, index), elapsed_substeps(scene, index + 1)
        acc.consume(replace(sample, interval_start_s=start, interval_end_s=end, contact_time_s=start))
    metrics = acc.metrics()
    assert metrics["success"] and metrics["completed_interval"]
    assert metrics["reward_integral"] == pytest.approx(.075)
    assert metrics["mean_reward"] == pytest.approx(1.)
    broken = _ExerciseMetrics(scene, task, (), 0., 1.)
    for index in range(33):
        broken.consume(motion_sample(broken, index))
    assert broken.metrics()["failure"] == "invalid_substep_duration"
    assert not broken.metrics()["completed_interval"]


def test_terminal_observer_failure_preserves_shorter_actual_clock_and_restores_timestep():
    scene = native_scene()
    task = build_shared_exercise_task(scene)
    env = SharedExerciseTaskEnv(scene, task)
    try:
        env.reset(seed=19)
        for _ in range(2):
            env.step(env.authored_action())
        before = env.snapshot()["engine_state"]
        nominal_tick = env._engine.model.opt.timestep
        original = env._engine._substep_observer
        def fail_after_observation(row):
            original(row)
            if row.substep_index == 3:
                raise RuntimeError("injected terminal observer failure")
        env._engine._substep_observer = fail_after_observation
        _, _, terminated, truncated, metrics = env.step(env.authored_action())
        assert terminated and not truncated and not metrics["completed_interval"]
        assert metrics["observed_substeps"] == 35
        assert metrics["simulated_duration_s"] == pytest.approx(elapsed_substeps(scene, 35))
        assert env._engine.model.opt.timestep == nominal_tick
        state = env.snapshot()
        assert state["engine_state_status"] == "last_valid_control_boundary" and state["engine_state"] == before
        validate_shared_exercise_snapshot(state, scene, task)
    finally:
        env.close()


def test_selector_and_residual_action_use_native_phase_and_pinned_clock():
    from weft.shared_exercise_residual_learning import _policy, _scene_runtime_hashes, prepare_shared_exercise_residual_policy
    from weft.shared_exercise_selection import _metrics
    scene = native_scene()
    task = build_shared_exercise_task(scene)
    env = SharedExerciseTaskEnv(scene, task)
    reset = {"jitter": False, "jitter_scale": 0.}
    try:
        env.reset(seed=19, options=reset)
        hashes = _scene_runtime_hashes(scene)
        assert "native_interval.py" in hashes
        policy = _policy(env, np.zeros((2, 19)), 19, 0., hashes)
        act = prepare_shared_exercise_residual_policy(policy, env)
        _metrics(env.metrics(), scene, task, policy, reset, initial=True)
        for _ in range(scene["task"]["horizon_steps"]):
            env.step(act())
        metrics = env.metrics()
        _metrics(metrics, scene, task, policy, reset)
        metrics["declared_duration_s"] = scene["dt"] * scene["task"]["horizon_steps"]
        with pytest.raises(ValueError, match="declared duration"):
            _metrics(metrics, scene, task, policy, reset)
    finally:
        env.close()


def test_non_grid_saved_body_constructs_and_packages_an_interactive_lab(tmp_path):
    import json
    from test_exercise_body_source import forbid_runtime, pose_run, seal
    from test_native_clock_runtime import native_binding
    from weft.exercise_body_source import prepare_exercise_body_source_from_evidence
    from weft.io import write_json
    from weft.lab import Lab
    from weft.shared_exercise_lab import prepare_shared_exercise_lab
    from weft.shared_world_human import construct_human_only_run

    poses = pose_run(tmp_path / "poses")
    evidence = json.loads((poses / "evidence.json").read_bytes())
    ticks = [60000 + i * 1001 for i in range(63)]
    for row in evidence["frames"]:
        if row["decoder_timestamp_s"] is not None:
            row["decoder_timestamp_s"] = ticks[row["frame"]] / 30000 - 1.997
    from weft.exercise_vision import resolve_pose_identity, finalize_evidence
    write_json(poses / "pose-landmarks.json", {"frames": evidence["frames"], "model": evidence["model"], "scope": evidence["scope"]})
    resolve_pose_identity(evidence, poses, "raw")
    finalize_evidence(evidence, poses)
    seal(poses)
    binding = native_binding(evidence, ticks, time_base="1/30000", offset=-1.997)
    clock = tmp_path / "clock.json"
    write_json(clock, binding)
    with pytest.MonkeyPatch.context() as patch:
        forbid_runtime(patch)
        prepare_exercise_body_source_from_evidence(poses, tmp_path / "body", native_clock=clock)
        construct_human_only_run(tmp_path / "body", make_person_inventory(binding["source_sha256"]), tmp_path / "world")
        prepare_shared_exercise_lab(tmp_path / "world", tmp_path / "lab")
    scene = json.loads((tmp_path / "world/scene.json").read_bytes())
    assert scene["task"]["native_interval"]["duration_s"] == pytest.approx(58 * 1001 / 30000)
    lab = Lab(tmp_path / "lab")
    try:
        assert lab.summary()["exercise_task"]["interval"]["native_interval"] == scene["task"]["native_interval"]
        lab.control("/api/reset", {"seed": 19})
        rows = lab.control("/api/step", {"controller": "reference", "steps": 3})["transitions"]
        assert len(rows) == 3
        assert rows[-1]["snapshot"]["task_metrics"]["failure"] is None
    finally:
        lab.close()


def test_finite_terminal_failure_keeps_partial_action_clock(monkeypatch):
    scene = native_scene()
    task = build_shared_exercise_task(scene)
    env = SharedExerciseTaskEnv(scene, task)
    try:
        env.reset(seed=19)
        for _ in range(2):
            env.step(env.authored_action())
        original = env._engine._instability
        def fail_on_third_terminal_substep():
            return "human_velocity_envelope_exceeded" if env._engine.data.time >= elapsed_substeps(scene, 35) - 1e-12 else original()
        monkeypatch.setattr(env._engine, "_instability", fail_on_third_terminal_substep)
        _, _, terminated, truncated, metrics = env.step(env.authored_action())
        assert terminated and not truncated and not metrics["completed_interval"]
        state = env.snapshot()
        assert state["engine_state_status"] == "current"
        assert state["engine_state"]["partial_action"]["completed_substeps"] == 3
        assert state["engine_state"]["time_s"] == pytest.approx(elapsed_substeps(scene, 35))
        validate_shared_exercise_snapshot(state, scene, task)
    finally:
        env.close()


@pytest.mark.parametrize("profile", ["shared-world-v1", "shared-world-v2", "shared-world-v4", "shared-world-v6", "shared-world-v5"])
def test_terminal_interval_never_enters_unqualified_equipment_or_extended_body(profile):
    from weft.native_interval import validate_interval
    scene = native_scene()
    scene.update(base_profile=scene["profile"], profile=profile)
    with pytest.raises(ValueError, match="person-only"):
        validate_interval(scene)
    if profile == "shared-world-v5":
        with pytest.raises(ValueError, match="not qualified for added feet"):
            validate_shared_world(scene)
