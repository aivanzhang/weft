import json

import cv2
import pytest

from weft.demo import create_demo
from weft.scene_vision import fit_scene, inspect_video


def test_demo_is_decodable_explicitly_synthetic_and_never_overwrites(tmp_path):
    output = tmp_path / "demo"
    result = create_demo(output)
    assert result["frames"] == 90
    provenance = json.loads((output / "provenance.json").read_text())
    assert provenance["physical_evidence"] is False
    assert provenance["source_kind"] == "SYNTHETIC SOFTWARE FIXTURE"
    assert any(set(event["pairs"][0]) == {"puck", "ball"} for event in provenance["contact_events"])
    capture = cv2.VideoCapture(result["video"])
    assert capture.get(cv2.CAP_PROP_FRAME_COUNT) == 90
    assert capture.get(cv2.CAP_PROP_FPS) == 30
    assert capture.read()[0]
    capture.release()
    with pytest.raises(FileExistsError):
        create_demo(output)


def test_demo_runs_through_measured_multi_object_reconstruction(tmp_path):
    source, run = tmp_path / "demo", tmp_path / "run"
    create_demo(source)
    run.mkdir()
    config = json.loads((source / "config.json").read_text())
    assert "damping" not in config and "restitution" not in config
    assert all("position" not in obj and "velocity" not in obj and "radius" not in obj for obj in config["objects"])
    evidence, scene = inspect_video(source / "video.mp4", run, config)
    selected = fit_scene(evidence, scene, run)
    report = json.loads((run / "reconstruction.json").read_text())
    assert len(selected["objects"]) == 3
    assert evidence["observation_count"] == 270
    assert "SYNTHETIC" in evidence["source_label"]
    assert report["after"]["rmse_image_width"] < 0.03
    assert report["replay_gate"]["passes"]
