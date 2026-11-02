"""Native simulation clocks stay exact while fixed-rate media are explicit previews."""

import json

import pytest

from test_native_nongrid import native_scene
from test_shared_world_human_only import person_scene
from weft.recording import Recorder, _native_export_timing, export_recording, load_recording, verify_recording
from weft.shared_exercise_recording import recording_scene
from weft.shared_exercise_task import SharedExerciseTaskEnv, build_shared_exercise_task


def test_native_export_labels_actual_episode_time_and_constant_rate_preview(tmp_path):
    scene = native_scene()
    task = build_shared_exercise_task(scene)
    envelope = recording_scene(scene, task)
    env = SharedExerciseTaskEnv(scene, task)
    recorder = Recorder(tmp_path / "session", envelope, profile="shared-exercise")
    try:
        env.reset(seed=19)
        recorder.start(env.snapshot(), seed=19)
        for _ in range(3):
            action = env.authored_action()
            _, reward, terminated, truncated, _ = env.step(action)
            recorder.append({"type": "step", "action": action.tolist(), "reward": reward,
                             "terminated": terminated, "truncated": truncated, "snapshot": env.snapshot()})
        env.reset(seed=20)
        recorder.append({"type": "reset", "seed": 20, "snapshot": env.snapshot()})
        action = env.authored_action()
        _, reward, terminated, truncated, _ = env.step(action)
        recorder.append({"type": "step", "action": action.tolist(), "reward": reward,
                         "terminated": terminated, "truncated": truncated, "snapshot": env.snapshot()})
        recorder.close()
        events = load_recording(recorder.root)["events"]
        result = export_recording(recorder.root)
        timing = result["export_timing"]
        assert timing == _native_export_timing(envelope, events)
        assert timing["recorded_simulated_duration_s"] == pytest.approx(.075 + 1 / 30)
        assert timing["reference_interval_duration_s"] == .075
        assert timing["terminal_control_duration_s"] == pytest.approx(1 / 120)
        assert timing["nominal_video_duration_s"] == pytest.approx(6 / 30)
        assert timing["nominal_video_fps"] == 30 and result["video_frames"] == 6
        assert '"export_timing"' in (recorder.root / "replay.html").read_text()
        manifest_path = recorder.root / "manifest.json"
        manifest = json.loads(manifest_path.read_bytes())
        for wrong in (None, {**timing, "recorded_simulated_duration_s": 6 / 30}, {**timing, "schema_version": True}):
            if wrong is None:
                manifest.pop("export_timing")
            else:
                manifest["export_timing"] = wrong
            manifest_path.write_text(json.dumps(manifest))
            with pytest.raises(ValueError, match="export timing"):
                verify_recording(recorder.root)
    finally:
        if recorder._started is not None:
            recorder.close()
        env.close()


def test_legacy_export_has_no_new_timing_metadata():
    scene = person_scene()
    assert _native_export_timing(recording_scene(scene, build_shared_exercise_task(scene)), []) is None
