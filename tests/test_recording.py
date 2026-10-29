import json
from pathlib import Path

import cv2
import pytest

from weft.recording import Recorder, export_recording, load_recording, render_snapshot, verify_recording


@pytest.fixture
def scene():
    return {"schema_version": 1, "profile": "planar-rigid-v1", "width": 1.0,
            "height": 0.6, "dt": 1/30, "controlled_id": "ball", "target": [0.8, 0.3],
            "task": {"position_tolerance": 0.05},
            "objects": [{"id": "ball", "label": "ball", "shape": "circle", "radius": 0.04,
                         "color": "#ECCB25", "position": [0.2, 0.3], "velocity": [0, 0]},
                        {"id": "obstacle", "shape": "box", "size": [0.05, 0.1],
                         "position": [0.5, 0.15], "velocity": [0, 0], "color": "#214F50"}]}


def snapshot(scene, step=0):
    state = json.loads(json.dumps(scene))
    state["objects"][0]["position"][0] += step * 0.01
    state["steps"] = step
    state["time_s"] = step * scene["dt"]
    return state


def step_event(scene, step=1):
    return {"type": "step", "action": 2, "reward": 0.1, "terminated": False,
            "truncated": False, "snapshot": snapshot(scene, step)}


def test_journal_preserves_actions_reset_and_exported_frames(tmp_path, scene):
    root = tmp_path / "session"
    recorder = Recorder(root, scene)
    recorder.start(snapshot(scene), seed=4)
    for index in range(1, 5):
        recorder.append(step_event(scene, index))
    recorder.append({"type": "reset", "seed": 9, "snapshot": snapshot(scene)})
    recorder.append(step_event(scene))
    recorder.close()
    data = load_recording(root)
    assert len(data["events"]) == 7
    assert data["events"][5]["seed"] == 9
    assert data["metadata"]["seed"] == 4
    result = export_recording(root)
    assert result["steps"] == 5 and result["resets"] == 1
    assert result["video_frames"] == 7 and result["verified_files"] == 6
    assert export_recording(root) == result
    cap = cv2.VideoCapture(str(root / "video.mp4"))
    frames = 0
    while cap.read()[0]:
        frames += 1
    cap.release()
    assert frames == 7
    assert (root / "preview.gif").is_file()
    assert "Recorded actions and states" in (root / "replay.html").read_text()


def test_interrupted_session_preserves_complete_lines(tmp_path, scene):
    root = tmp_path / "session"
    recorder = Recorder(root, scene)
    recorder.start(snapshot(scene))
    recorder.append(step_event(scene))
    # Simulate a crash after a partial write; previously flushed lines remain usable.
    recorder._stream.write('{"type":"step"')
    recorder._stream.flush()
    recorder._stream.close()
    with pytest.raises(ValueError, match="incomplete"):
        load_recording(root)
    recovered = load_recording(root, allow_incomplete=True)
    assert recovered["incomplete_journal"] and recovered["interrupted"]
    assert len(recovered["events"]) == 2
    with pytest.raises(ValueError):
        export_recording(root)


@pytest.mark.parametrize("change", [
    {"action": -1}, {"action": True}, {"reward": float("nan")},
    {"reward": float("inf")}, {"terminated": 1}, {"type": "unknown"},
])
def test_invalid_event_does_not_modify_journal(tmp_path, scene, change):
    root = tmp_path / "session"
    recorder = Recorder(root, scene)
    recorder.start(snapshot(scene))
    before = (root / "events.jsonl").read_bytes()
    with pytest.raises(ValueError):
        recorder.append(step_event(scene) | change)
    assert (root / "events.jsonl").read_bytes() == before
    recorder.close()


def test_recordings_never_overwrite_existing_sessions(tmp_path, scene):
    root = tmp_path / "session"
    recorder = Recorder(root, scene)
    recorder.start(snapshot(scene))
    recorder.close()
    before = (root / "events.jsonl").read_bytes()
    with pytest.raises(FileExistsError):
        Recorder(root, scene).start(snapshot(scene))
    assert (root / "events.jsonl").read_bytes() == before
    with pytest.raises(RuntimeError):
        recorder.append(step_event(scene))


def test_manifest_detects_journal_tampering(tmp_path, scene):
    root = tmp_path / "session"
    recorder = Recorder(root, scene)
    recorder.start(snapshot(scene))
    recorder.close()
    export_recording(root)
    with (root / "events.jsonl").open("a") as stream:
        stream.write("{}\n")
    with pytest.raises(ValueError, match="integrity"):
        verify_recording(root)


def test_snapshot_identity_and_nonfinite_state_are_rejected(tmp_path, scene):
    recorder = Recorder(tmp_path / "session", scene)
    bad = snapshot(scene)
    bad["objects"][0]["position"][0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        recorder.start(bad)
    assert not recorder.root.exists()
    bad = snapshot(scene)
    bad["objects"].pop()
    with pytest.raises(ValueError, match="every"):
        recorder.start(bad)


def test_export_escapes_untrusted_scene_labels(tmp_path, scene):
    scene["objects"][0]["label"] = '</script><script>alert("test")</script>'
    root = tmp_path / "session"
    recorder = Recorder(root, scene)
    recorder.start(snapshot(scene))
    recorder.close()
    export_recording(root)
    html = (root / "replay.html").read_text()
    assert '</script><script>alert' not in html
    assert '\\u003c/script>' in html


def test_recording_bounds_preserve_preceding_events(tmp_path, scene, monkeypatch):
    monkeypatch.setattr("weft.recording.MAX_EVENTS", 2)
    root = tmp_path / "session"
    recorder = Recorder(root, scene)
    recorder.start(snapshot(scene))
    recorder.append(step_event(scene))
    before = (root / "events.jsonl").read_bytes()
    with pytest.raises(ValueError, match="limit"):
        recorder.append(step_event(scene, 2))
    recorder.close()
    assert (root / "events.jsonl").read_bytes() == before
    assert len(load_recording(root)["events"]) == 2


def test_partial_export_is_preserved(tmp_path, scene):
    root = tmp_path / "session"
    recorder = Recorder(root, scene)
    recorder.start(snapshot(scene))
    recorder.close()
    (root / "video.partial.mp4").write_bytes(b"preserved partial export")
    with pytest.raises(FileExistsError, match="Partial export"):
        export_recording(root)
    assert (root / "video.partial.mp4").read_bytes() == b"preserved partial export"


def test_render_preserves_position_and_geometry(scene):
    image = render_snapshot(snapshot(scene), scene)
    assert image.shape == (426, 640, 3)
    assert tuple(image[192, 128]) == (236, 203, 37)


@pytest.fixture
def human_recording():
    scene = {"schema_version": 1, "profile": "exercise-v1", "width": 1, "height": 1,
             "dt": 1/30, "objects": [{"id": f"part-{i}", "shape": "circle", "radius": 0.02,
                                       "color": "#C3B19F"} for i in range(32)]}
    state = {"steps": 0, "time_s": 0,
             "objects": [{"id": obj["id"], "position": [0.5, 0.5], "velocity": [0, 0]}
                         for obj in scene["objects"]],
             "human": {"height": 1.9, "origin": [0, 0, 0],
                       "segments": [{"id": "torso", "a": [0, 0, 0.9], "b": [0, 0, 1.5], "radius": 0.12},
                                    {"id": "head", "a": [0, 0, 1.55], "b": [0, 0, 1.7], "radius": 0.1},
                                    {"id": "left_arm", "a": [0, 0, 1.45], "b": [-0.4, 0.1, 1.1], "radius": 0.04},
                                    {"id": "left_leg", "a": [-0.08, 0, 0.9], "b": [-0.1, 0, 0.05], "radius": 0.06},
                                    {"id": "right_leg", "a": [0.08, 0, 0.9], "b": [0.1, 0, 0.05], "radius": 0.06}],
                       "muscles": [{"name": "left_arm_proxy", "a": [-0.04, -0.03, 1.4],
                                    "b": [-0.35, 0.07, 1.14], "activation": 0.6, "status": "illustrative"}],
                       "angles": {"left_elbow": 30}, "selected_joint": "left_elbow", "target_angle": 60}}
    return scene, state


def test_human_recording_exports_exact_3d_states_and_camera_controls(tmp_path, human_recording):
    scene, state = human_recording
    root = tmp_path / "human-session"
    recorder = Recorder(root, scene, profile="exercise")
    recorder.start(state)
    changed = json.loads(json.dumps(state))
    changed["human"]["angles"]["left_elbow"] = 42
    changed["human"]["segments"][2]["b"] = [-0.35, 0.15, 1.2]
    changed["human"]["muscles"][0]["activation"] = 0.8
    changed["steps"] = 1
    changed["time_s"] = 1/30
    recorder.append({"type": "step", "action": 2, "reward": 0.1,
                     "terminated": False, "truncated": False, "snapshot": changed})
    recorder.close()
    result = export_recording(root)
    assert result["video_frames"] == 2
    data = load_recording(root)
    assert data["events"][0]["snapshot"] == state
    assert data["events"][1]["snapshot"] == changed
    html = (root / "replay.html").read_text()
    assert 'id="yaw"' in html and 'id="muscles"' in html
    assert "ILLUSTRATIVE MUSCLES · NOT MEASURED ACTIVATION" in html
    assert "Turning the camera does not change the recorded state" in html
    image = render_snapshot(state, scene)
    assert image.shape == (642, 640, 3)
    assert (image != render_snapshot(changed, scene)).any()


@pytest.mark.parametrize("mutate", [
    lambda h: h["segments"][0].update(a=[0, float("nan"), 0]),
    lambda h: h["segments"][0].update(b=[0, 0]),
    lambda h: h["segments"][0].update(radius=float("inf")),
    lambda h: h["segments"][0].update(radius=3),
    lambda h: h["muscles"][0].update(activation=float("nan")),
    lambda h: h["muscles"][0].update(activation=1.1),
    lambda h: h["muscles"][0].update(status="measured"),
    lambda h: h.update(muscles=[h["muscles"][0]] * 129),
    lambda h: h.update(segments=[h["segments"][0]] * 129),
    lambda h: h.update(angles={"left_elbow": [30]}),
    lambda h: h.update(angles={"left_elbow": float("inf")}),
    lambda h: h.update(joints={"left_elbow": [0, 0, float("nan")]}),
    lambda h: h.update(joints={"left_elbow": [0, 0]}),
    lambda h: h.update(angular_velocities={"left_elbow": float("inf")}),
    lambda h: h.update(height=float("nan")),
    lambda h: h.update(origin=[0, 0, float("inf")]),
    lambda h: h.update(view_height=float("nan"), view_origin=[0, 0, 0]),
    lambda h: h.update(view_height=2),
    lambda h: h.update(view_origin=[0, 0, 0]),
    lambda h: h.update(selected_joint="x" * 101),
    lambda h: h.update(target_angle=float("nan")),
])
def test_human_snapshot_rejects_nonfinite_or_unbounded_geometry(tmp_path, human_recording, mutate):
    scene, state = human_recording
    mutate(state["human"])
    recorder = Recorder(tmp_path / "invalid", scene, profile="exercise")
    with pytest.raises(ValueError):
        recorder.start(state)
    assert not recorder.root.exists()


def test_human_body_parts_are_bounded_without_expanding_planar_limit(tmp_path, human_recording):
    scene, state = human_recording
    invalid = dict(scene, objects=[{"id": f"part-{i}"} for i in range(65)])
    with pytest.raises(ValueError, match="1–64"):
        Recorder(tmp_path / "too-many", invalid, profile="exercise")
    planar = dict(scene, profile="planar-rigid-v1")
    with pytest.raises(ValueError, match="1–12"):
        Recorder(tmp_path / "planar", planar)
    recorder = Recorder(tmp_path / "missing-human", scene, profile="exercise")
    state.pop("human")
    with pytest.raises(ValueError, match="complete human"):
        recorder.start(state)


def test_invalid_human_step_does_not_change_recording(tmp_path, human_recording):
    scene, state = human_recording
    recorder = Recorder(tmp_path / "human", scene, profile="exercise")
    recorder.start(state)
    before = (recorder.root / "events.jsonl").read_bytes()
    state["human"]["muscles"][0]["a"][2] = float("inf")
    with pytest.raises(ValueError):
        recorder.append({"type": "step", "action": 2, "reward": 0,
                         "terminated": False, "truncated": False, "snapshot": state})
    recorder.close()
    assert (recorder.root / "events.jsonl").read_bytes() == before


def test_human_engine_recording_roundtrip_preserves_3d_state(tmp_path):
    pytest.importorskip("mujoco")
    from weft.human import HumanEnv, JOINT_NAMES, build_human
    evidence = {"source_sha256": "synthetic-test", "lengths": {
        "torso": 0.45, "upper_arm": 0.3, "forearm": 0.27, "thigh": 0.43,
        "shin": 0.43, "shoulder_width": 0.38, "hip_width": 0.28},
        "reference": [{"time_s": 0, "joints": {name: 0.1 for name in JOINT_NAMES}},
                      {"time_s": 2, "joints": {name: 0.8 for name in JOINT_NAMES}}]}
    env = HumanEnv(build_human(evidence, tmp_path / "model"))
    env.reset(seed=7, options={"jitter": False})
    root = tmp_path / "session"
    recorder = Recorder(root, env.scene, profile="exercise")
    recorder.start(env.snapshot(), seed=7)
    try:
        for _ in range(6):
            _, reward, terminated, truncated, _ = env.step(1)
            recorder.append({"type": "step", "action": 1, "reward": float(reward),
                             "terminated": bool(terminated), "truncated": bool(truncated),
                             "snapshot": env.snapshot()})
            if terminated or truncated:
                break
        final_state = env.snapshot()
        recorder.close()
        result = export_recording(root)
        saved = load_recording(root)
        assert saved["events"][-1]["snapshot"] == final_state
        assert len(final_state["human"]["segments"]) >= 11
        assert len(final_state["human"]["muscles"]) == 16
        assert result["video_frames"] == len(saved["events"])
    finally:
        recorder.close()
        env.close()
