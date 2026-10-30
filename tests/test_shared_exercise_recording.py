"""Human-only recordings preserve task gates, exact resets and failure-state labels."""

import json

import mujoco
import numpy as np
import pytest

from weft.recording import Recorder, _replay_html, export_recording, load_recording, recover_incomplete_recording
from weft.shared_exercise_recording import recording_scene, validate_recording_scene, verify_action_replay
from weft.shared_exercise_task import SharedExerciseTaskEnv, build_shared_exercise_task
from weft.shared_world_lab import authored_shared_world_scene


def short_task(root_assistance=True):
    scene = authored_shared_world_scene()
    scene["task"]["horizon_steps"] = 3
    return scene, build_shared_exercise_task(scene, root_assistance=root_assistance)


def append_step(recorder, env, action):
    _, reward, terminated, truncated, _ = env.step(action)
    event = {"type": "step", "action": action.tolist(), "reward": reward, "terminated": terminated,
             "truncated": truncated, "snapshot": env.snapshot()}
    recorder.append(event)
    return event


def test_human_only_journal_replays_actions_and_seeded_reset_exactly(tmp_path):
    scene, task = short_task()
    env = SharedExerciseTaskEnv(scene, task)
    recorder = Recorder(tmp_path / "session", recording_scene(scene, task), profile="shared-exercise")
    try:
        env.reset(seed=19)
        recorder.start(env.snapshot(), seed=19)
        append_step(recorder, env, env.authored_action())
        env.reset(seed=29, options={"jitter": True, "jitter_scale": .25})
        recorder.append({"type": "reset", "seed": 29, "snapshot": env.snapshot()})
        for _ in range(3):
            append_step(recorder, env, env.authored_action())
        recorder.close()
        report = verify_action_replay(recorder.root)
        assert report["exact_state_replay"] and report["events"] == 6
        data = load_recording(recorder.root)
        assert data["events"][2]["snapshot"]["resolved_reset_options"] == {"jitter": True, "jitter_scale": .25}
        for event in data["events"]:
            assert not any(event["snapshot"]["engine_state"]["normalized_action"][22:])
        html = _replay_html(data["scene"], data["events"])
        assert '"exercise_task"' in html and '"exercise_metrics"' in html
        assert "last valid geometry" in html
        assert "__WEFT_DATA__" not in html
    finally:
        env.close()


@pytest.mark.parametrize("jitter", [False, True])
def test_real_reset_with_contact_requirements_preserves_equipment_velocity(tmp_path, jitter):
    scene, _ = short_task(False)
    task = build_shared_exercise_task(scene, contact_requirements=[{
        "object_id": "ball", "wrist": "left", "kind": "wrist_region_contact", "start_step": 0, "end_step": 3,
    }])
    env = SharedExerciseTaskEnv(scene, task)
    recorder = Recorder(tmp_path / "session", recording_scene(scene, task), profile="shared-exercise")
    options = {"jitter": jitter, "jitter_scale": .25 if jitter else 0.}
    try:
        env.reset(seed=19, options=options)
        first = env.snapshot()
        if jitter:
            assert any(any(obj["velocity"]) for obj in first["engine_state"]["objects"])
            assert any(row["speed_model_units_s"] > 0 for row in first["task_metrics"]["equipment"])
        recorder.start(first, seed=19)
        append_step(recorder, env, np.zeros(env.action_count))
        env.reset(seed=29, options=options)
        reset = env.snapshot()
        recorder.append({"type": "reset", "seed": 29, "snapshot": reset})
        recorder.close()
        events = load_recording(recorder.root)["events"]
        assert events[0]["snapshot"] == first
        assert events[-1]["snapshot"] == reset
        assert verify_action_replay(recorder.root)["exact_state_replay"]
    finally:
        recorder.close()
        env.close()


@pytest.mark.parametrize("kind", ["start", "reset"])
def test_advanced_zero_action_state_cannot_be_recorded_as_reset(tmp_path, kind):
    scene, task = short_task(False)
    env = SharedExerciseTaskEnv(scene, task)
    recorder = Recorder(tmp_path / "session", recording_scene(scene, task), profile="shared-exercise")
    try:
        env.reset(seed=4)
        if kind == "reset":
            recorder.start(env.snapshot(), seed=4)
            before = (recorder.root / "events.jsonl").read_bytes()
        env.step(np.zeros(env.action_count))
        advanced = env.snapshot()
        assert not any(advanced["last_action"])
        assert advanced["task_metrics"]["observed_substeps"] > 0
        with pytest.raises(ValueError, match="genuine initial boundary"):
            if kind == "start":
                recorder.start(advanced, seed=4)
            else:
                recorder.append({"type": "reset", "seed": 4, "snapshot": advanced})
        if kind == "start":
            assert not recorder.root.exists()
        else:
            assert (recorder.root / "events.jsonl").read_bytes() == before
    finally:
        if recorder._started is not None:
            recorder.close()
        env.close()


@pytest.mark.parametrize("kind", ["start", "reset"])
def test_loader_rejects_advanced_coast_snapshot_relabelled_as_reset(tmp_path, kind):
    scene, task = short_task(False)
    env = SharedExerciseTaskEnv(scene, task)
    recorder = Recorder(tmp_path / "session", recording_scene(scene, task), profile="shared-exercise")
    try:
        env.reset(seed=4)
        recorder.start(env.snapshot(), seed=4)
        advanced = append_step(recorder, env, np.zeros(env.action_count))["snapshot"]
        env.reset(seed=5)
        recorder.append({"type": "reset", "seed": 5, "snapshot": env.snapshot()})
        recorder.close()
        journal = recorder.root / "events.jsonl"
        events = [json.loads(line) for line in journal.read_text().splitlines()]
        events[0 if kind == "start" else 2]["snapshot"] = advanced
        journal.write_text("".join(json.dumps(event) + "\n" for event in events))
        with pytest.raises(ValueError, match="genuine initial boundary"):
            load_recording(recorder.root)
    finally:
        recorder.close()
        env.close()


@pytest.mark.parametrize("target,seed", [("metadata", 1), ("metadata", True), ("initial_event", 1)])
def test_loader_binds_metadata_seed_to_initial_event(tmp_path, target, seed):
    root = closed_session(tmp_path, seed=1 if seed is True else 0)
    if target == "metadata":
        metadata = root / "recording.json"
        document = json.loads(metadata.read_text())
        document["seed"] = seed
        metadata.write_text(json.dumps(document))
    else:
        journal = root / "events.jsonl"
        events = [json.loads(line) for line in journal.read_text().splitlines()]
        events[0]["seed"] = seed
        journal.write_text("".join(json.dumps(event) + "\n" for event in events))
    with pytest.raises(ValueError, match="seed"):
        load_recording(root)


def test_envelope_rejects_alias_changes():
    scene, task = short_task(False)
    bundle = recording_scene(scene, task)
    bundle["objects"][0]["position"][0] += .01
    with pytest.raises(ValueError, match="aliases"):
        validate_recording_scene(bundle)
    bundle = recording_scene(scene, task)
    bundle["dt"] /= 2
    with pytest.raises(ValueError, match="aliases"):
        validate_recording_scene(bundle)


def test_recorded_flags_and_action_must_match_task_state(tmp_path):
    scene, task = short_task(False)
    env = SharedExerciseTaskEnv(scene, task)
    recorder = Recorder(tmp_path / "session", recording_scene(scene, task), profile="shared-exercise")
    try:
        env.reset(seed=0)
        recorder.start(env.snapshot())
        action = env.authored_action()
        _, reward, terminated, truncated, _ = env.step(action)
        event = {"type": "step", "action": action.tolist(), "reward": reward, "terminated": not terminated,
                 "truncated": truncated, "snapshot": env.snapshot()}
        with pytest.raises(ValueError, match="termination"):
            recorder.append(event)
        event["terminated"] = terminated
        event["action"] += [0., 0., 0.]
        with pytest.raises(ValueError, match="19.*human-only"):
            recorder.append(event)
        recorder.close()
    finally:
        env.close()


def test_failed_step_records_last_valid_geometry_explicitly(tmp_path, monkeypatch):
    scene, task = short_task()
    env = SharedExerciseTaskEnv(scene, task)
    recorder = Recorder(tmp_path / "failed", recording_scene(scene, task), profile="shared-exercise")
    original = mujoco.mj_step

    def break_step(model, data):
        original(model, data)
        data.qpos[0] = np.nan

    try:
        env.reset(seed=0)
        first = env.snapshot()
        recorder.start(first)
        monkeypatch.setattr(mujoco, "mj_step", break_step)
        event = append_step(recorder, env, env.authored_action())
        recorder.close()
        assert event["terminated"]
        assert event["snapshot"]["engine_state_status"] == "last_valid_control_boundary"
        assert event["snapshot"]["engine_state"] == first["engine_state"]
        assert event["snapshot"]["task_metrics"]["simulated_duration_s"] > 0
        assert load_recording(recorder.root)["events"][-1] == {**event, "index": 1}
    finally:
        env.close()


def test_recovery_preserves_interrupted_exercise_journal(tmp_path):
    scene, task = short_task()
    env = SharedExerciseTaskEnv(scene, task)
    recorder = Recorder(tmp_path / "interrupted", recording_scene(scene, task), profile="shared-exercise")
    try:
        env.reset(seed=0)
        recorder.start(env.snapshot())
        append_step(recorder, env, env.authored_action())
        recorder._stream.write('{"unfinished":')
        recorder._stream.flush()
        recorder._stream.close()
        original = (recorder.root / "events.jsonl").read_bytes()
        recovered = recover_incomplete_recording(recorder.root, tmp_path / "recovered", export=False)
        assert recovered["events"] == 2
        assert (recorder.root / "events.jsonl").read_bytes() == original
        assert verify_action_replay(tmp_path / "recovered")["exact_state_replay"]
    finally:
        env.close()


def test_replay_rejects_a_tampered_reward(tmp_path):
    scene, task = short_task()
    env = SharedExerciseTaskEnv(scene, task)
    recorder = Recorder(tmp_path / "session", recording_scene(scene, task), profile="shared-exercise")
    try:
        env.reset(seed=0)
        recorder.start(env.snapshot())
        append_step(recorder, env, env.authored_action())
        recorder.close()
        journal = recorder.root / "events.jsonl"
        events = [json.loads(line) for line in journal.read_text().splitlines()]
        events[1]["reward"] += 1
        journal.write_text("".join(json.dumps(event) + "\n" for event in events))
        with pytest.raises(ValueError, match="outcome replay"):
            verify_action_replay(recorder.root)
    finally:
        env.close()


def closed_session(tmp_path, *, seed=0):
    scene, task = short_task()
    env = SharedExerciseTaskEnv(scene, task)
    recorder = Recorder(tmp_path / "session", recording_scene(scene, task), profile="shared-exercise")
    try:
        env.reset(seed=seed)
        recorder.start(env.snapshot(), seed=seed)
        append_step(recorder, env, env.authored_action())
        recorder.close()
        return recorder.root
    finally:
        env.close()


def test_export_deadline_preserves_partial_output_and_closed_journal(tmp_path, monkeypatch):
    import weft.recording as recording
    import weft.shared_exercise_recording as exercise_recording

    root = closed_session(tmp_path)
    journal = (root / "events.jsonl").read_bytes()
    now = [0.]
    original_prepare = exercise_recording.prepare_renderer

    def slow_prepare(*args, **kwargs):
        render = original_prepare(*args, **kwargs)

        def slow_frame(snapshot):
            frame = render(snapshot)
            now[0] = 2.
            return frame

        return slow_frame

    monkeypatch.setattr(recording.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(exercise_recording, "prepare_renderer", slow_prepare)
    with pytest.raises(TimeoutError, match="deadline"):
        export_recording(root, deadline=1.)
    assert (root / "events.jsonl").read_bytes() == journal
    assert (root / "video.partial.mp4").exists()
    assert not (root / "manifest.json").exists()
    assert json.loads((root / "recording.json").read_text())["status"] == "closed"


def test_export_byte_bound_rejects_before_any_media_write(tmp_path):
    root = closed_session(tmp_path)
    with pytest.raises(ValueError, match="byte bound"):
        export_recording(root, max_output_bytes=1)
    assert not (root / "video.partial.mp4").exists()
    assert not (root / "manifest.json").exists()


def test_action_replay_checks_deadline_after_its_last_event(tmp_path, monkeypatch):
    import weft.shared_exercise_recording as exercise_recording

    root = closed_session(tmp_path)
    now = [0.]
    original_step = SharedExerciseTaskEnv.step

    def slow_step(self, action):
        result = original_step(self, action)
        now[0] = 2.
        return result

    monkeypatch.setattr(exercise_recording.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(SharedExerciseTaskEnv, "step", slow_step)
    with pytest.raises(TimeoutError, match="wall-time"):
        verify_action_replay(root, max_seconds=1.)


def test_prepared_renderer_uses_pinned_scene_and_validates_every_state():
    from weft.shared_exercise_recording import prepare_renderer

    scene, task = short_task()
    env = SharedExerciseTaskEnv(scene, task)
    try:
        env.reset(seed=0)
        bundle = recording_scene(scene, task)
        render = prepare_renderer(bundle)
        snapshot = env.snapshot()
        before = render(snapshot)
        bundle["engine_scene"]["objects"][0]["color"] = "#000000"
        np.testing.assert_array_equal(render(snapshot), before)
        snapshot["engine_state"]["objects"][0]["mass"] += 1
        with pytest.raises(ValueError, match="mass"):
            render(snapshot)
    finally:
        env.close()
