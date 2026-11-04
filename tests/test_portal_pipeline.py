import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from weft.io import digest, write_json
from weft.portal_pipeline import appearance_config, sample_video, spatial_config, execute_job


def test_sampling_preserves_original_and_copies_bounded_frames(tmp_path):
    video = tmp_path / "video.avi"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), 20, (160, 100))
    for i in range(40):
        frame = np.full((100,160,3), 230, np.uint8)
        cv2.circle(frame, (30+i,50), 10, (0,180,240), -1)
        writer.write(frame)
    writer.release()
    original = digest(video)
    frames = sample_video(video, tmp_path / "frames")
    assert frames["decoded_frame_count"] == 40
    assert len(frames["frames"]) == 6
    assert frames["frames"][0]["frame"] == 0 and frames["frames"][-1]["frame"] == 39
    assert digest(video) == original == frames["source_sha256"]
    assert all("decoder_time_s" in row and "nominal_time_s" in row for row in frames["frames"])
    assert max((tmp_path / "frames" / row["artifact"]).stat().st_size for row in frames["frames"]) < 300_000
    from weft.portal_pipeline import object_localization
    plan = {"objects": [{"label":"Ball", "box_xyxy":[0,0,.1,.1], "shape":"circle", "role":"movable"}]}
    profile, settings, receipt = object_localization(plan, frames, video)
    assert profile == "planar" and receipt["ai_regions_used"] is False
    assert settings["objects"][0]["shape"] == "circle"
    assert "box_xywh" not in settings["objects"][0]


def test_roi_proposal_pins_source_and_keeps_authored_physics():
    plan = {"objects": [{"label":"Ball", "box_xyxy":[.1,.2,.3,.4], "shape":"circle", "role":"movable"},
                        {"label":"Box", "box_xyxy":[.5,.1,.7,.5], "shape":"box", "role":"static"}]}
    result = appearance_config(plan, {"width":500,"height":300,"decoded_frame_count":80,"source_sha256":"a"*64})
    assert result["tracking"]["objects"][0]["box_xywh"] == [48,58,104,64]
    assert result["tracking"]["source_sha256"] == "a"*64
    assert result["world"]["controlled_id"] == "object_1"
    assert "authored" in result["tracking"]["objects"][0]["provenance"]
    from weft.generic_vision import validate_config
    validate_config(result["tracking"])


def test_spatial_embedding_does_not_invent_metric_depth_or_ground():
    from weft.scene import validate_scene
    from weft.spatial_scene_inputs import construct_spatial_scene
    scene = validate_scene({"schema_version":1,"profile":"planar-rigid-v1","width":1.,"height":.6,"dt":1/30,
        "gravity":[0,0],"damping":.98,"restitution":.5,"objects":[{"id":"ball","label":"Ball","role":"active",
        "shape":"circle","radius":.04,"position":[.3,.3],"velocity":[.1,0],"color":"#fedc00","mass":1,
        "provenance":{"source":"synthetic test"}}],"controlled_id":"ball","target":[.75,.3],
        "task":{"acceleration":2.,"max_steps":300,"position_tolerance":.06,"speed_tolerance":.15},
        "relationships":[],"assumptions":["Synthetic test"]})
    config = spatial_config(scene)
    result = construct_spatial_scene(scene, config)
    assert result["scene"]["objects"][0]["position"] == [.3,.5,.3]
    assert result["scene"]["gravity"] == [0.,0.,0.]
    assert "authored" in " ".join(result["scene"]["assumptions"])


def moving_circle_video(path, frames=40, size=(160, 100), fps=20):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, size)
    for i in range(frames):
        frame = np.full((size[1], size[0], 3), 230, np.uint8)
        cv2.circle(frame, (20 + (i % 100), size[1] // 2), 10, (0, 60, 200), -1)
        writer.write(frame)
    writer.release()
    return path


def test_undecodable_video_stops_before_ai_or_physics(tmp_path, monkeypatch):
    import weft.portal_pipeline as pipeline
    video = tmp_path / "source.webm"
    video.write_bytes(b"source fixture")
    write_json(tmp_path / "job.json", {"id": "a"*32, "source_file": video.name, "source_sha256": digest(video),
                                       "ai": True, "mode": "auto", "episodes": 300})
    monkeypatch.setattr(pipeline, "analyze_frames",
                        lambda *args, **kwargs: pytest.fail("Undecodable video must not reach AI"))
    assert execute_job(tmp_path) == 1
    assert "decodable video" in json.loads((tmp_path / "status.json").read_text())["error"]
    assert not (tmp_path / "lab").exists()


def test_unsupported_scene_degrades_to_appearance_instead_of_refusing(tmp_path, monkeypatch):
    import weft.portal_pipeline as pipeline
    from weft import lab as weft_lab
    video = moving_circle_video(tmp_path / "source.avi")
    write_json(tmp_path / "job.json", {"id": "a"*32, "source_file": video.name, "source_sha256": digest(video),
                                       "ai": True, "mode": "auto", "episodes": 300})
    monkeypatch.setattr(pipeline, "analyze_frames", lambda *args, **kwargs: {
        "route": "unsupported", "camera": "moving", "people": 2, "objects": [],
        "summary": "Two people and a moving camera", "warnings": ["camera motion"]})
    captured = []

    def capture(video_path, lab, *, profile, config, **kwargs):
        captured.append({"profile": profile,
                         "config": json.loads(Path(config).read_text()) if config else None})
        raise RuntimeError("tracking coverage failed (synthetic)")

    monkeypatch.setattr(weft_lab, "prepare_lab", capture)
    assert execute_job(tmp_path) == 1
    degraded = json.loads((tmp_path / "analysis/degraded-route.json").read_text())
    assert degraded["proposed_route"] == "unsupported"
    # Every ladder attempt is recorded, and appearance attempts carry bounded intervals.
    assert len(degraded["attempts"]) == len(captured) >= 3
    spans = [row["config"]["tracking"] for row in captured if row["profile"] == "appearance"]
    assert spans and all(t["end_frame"] - t["start_frame"] <= 240 for t in spans)
    assert "No continuously trackable interval" in json.loads((tmp_path / "status.json").read_text())["error"]


def test_objects_mode_with_disagreeing_plan_degrades_instead_of_conflicting(tmp_path, monkeypatch):
    import weft.portal_pipeline as pipeline
    from weft import lab as weft_lab
    video = moving_circle_video(tmp_path / "source.avi")
    write_json(tmp_path / "job.json", {"id": "b"*32, "source_file": video.name, "source_sha256": digest(video),
                                       "ai": True, "mode": "objects", "episodes": 300})
    monkeypatch.setattr(pipeline, "analyze_frames", lambda *args, **kwargs: {
        "route": "unsupported", "camera": "fixed", "people": 1,
        "objects": [{"label": "puck", "shape": "circle", "role": "movable", "box_xyxy": [.1, .3, .3, .6]}],
        "summary": "Hands interact with pucks", "warnings": []})
    monkeypatch.setattr(weft_lab, "prepare_lab",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("tracking failed (synthetic)")))
    assert execute_job(tmp_path) == 1
    degraded = json.loads((tmp_path / "analysis/degraded-route.json").read_text())
    assert degraded["proposed_route"] == "unsupported" and degraded["attempts"]
    assert "No continuously trackable interval" in json.loads((tmp_path / "status.json").read_text())["error"]


def test_window_derivation_bounds_long_videos_and_preserves_the_original(tmp_path):
    from weft.portal_pipeline import bounded_source_window
    video = moving_circle_video(tmp_path / "long.avi", frames=2000, fps=20)
    original = digest(video)
    window, provenance = bounded_source_window(video, tmp_path)
    assert window != video and provenance["window_frames"] <= 1740
    assert digest(video) == original == provenance["original_sha256"]
    frames = sample_video(window, tmp_path / "frames")
    assert frames["decoded_frame_count"] == provenance["window_frames"]
    short = moving_circle_video(tmp_path / "short.avi")
    same, none = bounded_source_window(short, tmp_path)
    assert same == short and none is None


def test_demo_source_normalizes_foreign_containers_and_preserves_the_original(tmp_path):
    from weft.portal_pipeline import prepare_demo_source
    # An .avi upload is accepted by the portal but refused by perception, which demands MP4.
    video = moving_circle_video(tmp_path / "phone.avi", frames=60, size=(640, 360), fps=20)
    original = digest(video)
    normalized, provenance = prepare_demo_source(video, tmp_path / "normalized")
    assert normalized.suffix == ".mp4" and normalized != video
    assert provenance["normalized"]["container"] == ".mp4"
    assert provenance["original"]["container"] == ".avi"
    assert digest(video) == original == provenance["original_sha256"]
    assert digest(normalized) == provenance["normalized_sha256"]
    assert sample_video(normalized, tmp_path / "frames")["decoded_frame_count"] >= 12


def test_demo_source_bounds_rate_and_size_but_leaves_compliant_clips_untouched(tmp_path):
    from weft.portal_pipeline import DEMO_SOURCE_POLICY, prepare_demo_source
    wide = moving_circle_video(tmp_path / "wide.avi", frames=240, size=(1920, 1080), fps=120)
    _, provenance = prepare_demo_source(wide, tmp_path / "wide-out")
    assert max(provenance["normalized"]["width"], provenance["normalized"]["height"]) \
        <= DEMO_SOURCE_POLICY["max_long_side"]
    assert provenance["normalized"]["fps"] <= DEMO_SOURCE_POLICY["target_fps"]
    assert provenance["frame_stride"] > 1

    long_clip = moving_circle_video(tmp_path / "long.avi", frames=1400, size=(320, 180), fps=25)
    _, bounded = prepare_demo_source(long_clip, tmp_path / "long-out")
    assert bounded["normalized"]["frames"] <= DEMO_SOURCE_POLICY["max_frames"]

    compliant = tmp_path / "ready.mp4"
    writer = cv2.VideoWriter(str(compliant), cv2.VideoWriter_fourcc(*"mp4v"), 25, (320, 180))
    for index in range(40):
        frame = np.full((180, 320, 3), 220, np.uint8)
        cv2.circle(frame, (20 + index, 90), 8, (0, 60, 200), -1)
        writer.write(frame)
    writer.release()
    same, none = prepare_demo_source(compliant, tmp_path / "ready-out")
    assert same == compliant and none is None


def test_motion_region_finds_movement_and_refuses_static_video(tmp_path):
    from weft.portal_pipeline import motion_region
    moving = moving_circle_video(tmp_path / "moving.avi")
    region = motion_region(moving)
    x, y, w, h = region["box_xywh"]
    assert w >= 12 and h >= 12 and "not object identity" in region["provenance"]
    static = tmp_path / "static.avi"
    writer = cv2.VideoWriter(str(static), cv2.VideoWriter_fourcc(*"MJPG"), 20, (160, 100))
    for _ in range(20):
        writer.write(np.full((100, 160, 3), 230, np.uint8))
    writer.release()
    with pytest.raises(ValueError, match="static scene|moving region"):
        motion_region(static)


def test_spatial_archives_fit_current_implementation_but_remain_bounded():
    from weft.spatial_scene_inputs import _manifest, MAX_FILES
    from weft.spatial_scene_pipeline import MAX_OUTPUT_FILES
    files = {f"implementation-source/module_{i}.py": "a" * 64 for i in range(347)}
    manifest = {"schema_version":1,"files":files}
    assert len(_manifest(json.dumps(manifest).encode())) == 347
    assert MAX_OUTPUT_FILES == MAX_FILES + 64
    manifest["files"] = {f"module_{i}.py":"a"*64 for i in range(MAX_FILES+1)}
    with pytest.raises(ValueError, match="bounded"):
        _manifest(json.dumps(manifest).encode())


def test_multi_person_exercise_proposal_reaches_the_interactive_inventory_prompt(tmp_path, monkeypatch):
    import weft.portal_pipeline as pipeline
    video = moving_circle_video(tmp_path / "source.avi")
    write_json(tmp_path / "job.json", {"id": "c"*32, "source_file": video.name, "source_sha256": digest(video),
                                       "ai": True, "mode": "auto", "episodes": 300})
    monkeypatch.setattr(pipeline, "analyze_frames", lambda *args, **kwargs: {
        "route": "exercise", "camera": "fixed", "people": 3, "objects": [],
        "exercise_inventory": "unknown", "summary": "Three dancers", "warnings": []})
    assert execute_job(tmp_path) == 0
    status = json.loads((tmp_path / "status.json").read_text())
    assert status["status"] == "needs_input" and status["route"] == "exercise"
    assert not (tmp_path / "analysis/degraded-route.json").exists()
