"""Terminal actions retain orientation, spin and integration diagnostics on replay."""

from copy import deepcopy

import cv2
import pytest

from weft.io import digest
from weft.recording import Recorder, load_recording, render_snapshot
from weft.scene import SceneEnv
from test_oriented_scene import ramp_scene
from test_scene_console import run_commands, sealed_run


@pytest.mark.parametrize("mutate", [
    lambda s: s["objects"][0].pop("angle"), lambda s: s["objects"][0].update(angle=float("nan")),
    lambda s: s["objects"][0].pop("angular_velocity"), lambda s: s["objects"][0].update(angular_velocity=21),
    lambda s: s.pop("units"), lambda s: s.update(substeps=257),
    lambda s: s.update(substeps=True), lambda s: s.pop("angular_speed_limited"),
])
def test_recordings_cannot_silently_drop_rotation_or_limits(tmp_path, mutate):
    env = SceneEnv(ramp_scene())
    env.reset(options={"jitter": False})
    snapshot = env.snapshot()
    mutate(snapshot)
    recorder = Recorder(tmp_path / "recording", env.scene)
    with pytest.raises(ValueError):
        recorder.start(snapshot)
    assert not (tmp_path / "recording").exists()
    env.close()


def test_terminal_recording_matches_all_rotated_states_and_decodes(tmp_path):
    scene = ramp_scene()
    scene["dt"] = 1 / 60
    run = sealed_run(tmp_path / "run", scene)
    before = {p.name: digest(p) for p in run.iterdir()}
    commands = ["coast 6"] * 3 + ["right 6", "left 6", "up 6", "down 6", "brake 6", "reset", "quit"]
    result, _ = run_commands(run, tmp_path / "recording", commands)
    saved = load_recording(tmp_path / "recording")
    assert result["steps"] == 48 and result["resets"] == 1 and result["video_frames"] == 50
    env = SceneEnv(scene)
    for event in saved["events"]:
        if event["type"] in {"start", "reset"}:
            env.reset(seed=event["seed"], options={"jitter": False})
        else:
            _, reward, terminated, truncated, _ = env.step(event["action"])
            assert (reward, terminated, truncated) == (event["reward"], event["terminated"], event["truncated"])
        assert env.snapshot() == event["snapshot"]
    initial = saved["events"][0]["snapshot"]
    flat = deepcopy(initial)
    flat["objects"][1]["angle"] = 0
    assert (render_snapshot(initial, scene) != render_snapshot(flat, scene)).any()
    capture = cv2.VideoCapture(str(tmp_path / "recording/video.mp4"))
    frames = 0
    while capture.read()[0]:
        frames += 1
    capture.release()
    assert frames == len(saved["events"])
    assert before == {p.name: digest(p) for p in run.iterdir()}
    env.close()
