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


def test_moving_camera_stops_before_pose_or_physics(tmp_path, monkeypatch):
    import weft.portal_pipeline as pipeline
    video=tmp_path/"source.webm";video.write_bytes(b"source fixture")
    write_json(tmp_path/"job.json",{"id":"a"*32,"source_file":video.name,"source_sha256":digest(video),"ai":True,"mode":"auto","episodes":300})
    monkeypatch.setattr(pipeline,"sample_video",lambda *args: {})
    monkeypatch.setattr(pipeline,"analyze_frames",lambda *args,**kwargs: {"route":"exercise","camera":"moving","people":1,"summary":"Camera moved"})
    import weft.lab
    def forbidden(*args,**kwargs): raise AssertionError("Unsupported video must not construct a world")
    monkeypatch.setattr(weft.lab,"prepare_lab",forbidden)
    assert execute_job(tmp_path)==1
    assert "Camera moved" in json.loads((tmp_path/"status.json").read_text())["error"]
    assert not (tmp_path/"lab").exists()


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
