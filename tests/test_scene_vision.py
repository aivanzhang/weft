from copy import deepcopy
import json

import cv2
import numpy as np
import pytest
from PIL import Image

import weft.scene_vision as vision
from weft.scene_vision import fit_scene, inspect_video, reconstruction_metrics, validate_config


def write_clip(path, *, frames=60, ambiguous=False, blank=False):
    """Synthetic color-tracking fixture, not a physical observation."""
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30, (200, 140))
    assert writer.isOpened()
    for index in range(frames):
        image = np.full((140, 200, 3), 230, np.uint8)
        if not blank:
            cv2.circle(image, (30 + index, 40), 10, (0, 230, 230), -1)
            cv2.circle(image, (145, 100), 11, (230, 40, 20) if not ambiguous else (0, 230, 230), -1)
            cv2.rectangle(image, (95, 85), (115, 110), (20, 20, 225), -1)
        writer.write(image)
    writer.release()


def config():
    return {"objects": [
        {"id": "puck", "label": "Yellow puck", "hue_min": 20, "hue_max": 40, "role": "active", "shape": "circle"},
        {"id": "ball", "label": "Blue ball", "hue_min": 100, "hue_max": 125, "role": "passive", "shape": "circle"},
        {"id": "obstacle", "label": "Red obstacle", "hue_min": 170, "hue_max": 10, "role": "static", "shape": "box"},
    ], "controlled_id": "puck", "target": [0.8, 0.2], "source_label": "Synthetic tracker test; not physical evidence"}


def test_tracking_builds_shared_world_and_keeps_development_boundary(tmp_path):
    video = tmp_path / "fixture.mp4"
    write_clip(video)
    output = tmp_path / "run"
    output.mkdir()
    updates = []
    evidence, scene = inspect_video(video, output, config(), lambda stage, detail: updates.append((stage, detail)))
    assert evidence["decoded_frames"] == 60
    assert evidence["observation_count"] == 180
    assert len(scene["objects"]) == 3
    assert scene["controlled_id"] == "puck"
    assert len({obj["id"] for obj in scene["objects"]}) == 3
    assert max(row["frame"] for row in evidence["generation"]) < min(row["frame"] for row in evidence["development_validation"])
    assert not {row["id"] for row in evidence["generation"]} & {row["id"] for row in evidence["development_validation"]}
    assert "development" in evidence["scope"]
    assert "Synthetic" in evidence["source_label"]
    assert next(obj for obj in scene["objects"] if obj["id"] == "obstacle")["velocity"] == [0, 0]
    assert all(obj["provenance"]["mass"] == "authored assumption" for obj in scene["objects"])
    assert updates[-1][1]["objects"] == 3
    for name in ("background.jpg", "tracks.gif", "tracks-contact-sheet.png", "evidence.json", "tracked-objects.json", "scene.json"):
        assert (output / name).stat().st_size > 0


def test_auto_discovery_tracks_distinct_colors_without_semantic_claims(tmp_path):
    video = tmp_path / "fixture.mp4"
    write_clip(video)
    evidence, scene = inspect_video(video, tmp_path)
    assert len(evidence["tracking_profiles"]) == 3
    assert sum(obj["role"] == "active" for obj in scene["objects"]) == 1
    assert sum(obj["role"] == "static" for obj in scene["objects"]) == 2
    assert any("not recovered" in text for text in scene["assumptions"])


def test_bounded_repair_preserves_task_and_fixed_measurements(tmp_path):
    video = tmp_path / "fixture.mp4"
    write_clip(video)
    evidence, scene = inspect_video(video, tmp_path, config())
    evidence_before, task_before = deepcopy(evidence), deepcopy(scene["task"])
    selected = fit_scene(evidence, scene, tmp_path)
    report = json.loads((tmp_path / "reconstruction.json").read_text())
    ledger = json.loads((tmp_path / "repair-ledger.json").read_text())
    assert evidence == evidence_before
    assert selected["task"] == task_before
    assert 3 <= len(ledger) <= 4
    assert all(set(row["proposal"]) == {"damping"} for row in ledger[:3])
    assert all(set(row["proposal"]) == {"initial_velocity"} for row in ledger[3:])
    assert report["after"]["rmse_px"] <= report["before"]["rmse_px"]
    assert len(report["frames"]) == 60
    assert report["frames"][0]["partition"] == "generation"
    assert report["frames"][-1]["partition"] == "development_validation"
    assert len(report["frames"][-1]["predicted"]) == 3
    assert report["replay_gate"]["passes"]
    assert reconstruction_metrics(evidence, selected)["rmse_px"] == report["after"]["rmse_px"]
    with Image.open(tmp_path / "reconstruction.gif") as image:
        assert image.n_frames > 1
        assert image.width == 400
    assert (tmp_path / "reconstruction-contact-sheet.png").stat().st_size > 0


@pytest.mark.parametrize("invalid", [
    {"typo": 1}, {"target": [float("nan"), 0.3]}, {"damping": 0}, {"gravity": [0]},
    {"objects": []}, {"controlled_id": "missing"},
    {"objects": [{"id": "one", "hue_min": 20, "hue_max": 40}, {"id": "two", "hue_min": 30, "hue_max": 50}]},
    {"objects": [{"id": "one", "hue_min": 20, "hue_max": 40}, {"id": "one", "hue_min": 90, "hue_max": 100}]},
])
def test_config_rejects_ambiguous_or_unbounded_settings(invalid):
    with pytest.raises(ValueError):
        validate_config(invalid)


def test_ambiguous_colors_fail_instead_of_switching_identity(tmp_path):
    video = tmp_path / "ambiguous.mp4"
    write_clip(video, ambiguous=True)
    with pytest.raises(ValueError, match="Ambiguous"):
        inspect_video(video, tmp_path)


def test_absent_objects_fail_instead_of_inventing_a_scene(tmp_path):
    video = tmp_path / "blank.mp4"
    write_clip(video, blank=True)
    with pytest.raises(ValueError, match="No supported"):
        inspect_video(video, tmp_path)


def test_static_role_must_agree_with_generation_motion(tmp_path):
    video = tmp_path / "fixture.mp4"
    write_clip(video)
    settings = config()
    settings["objects"][0]["role"] = "static"
    settings["controlled_id"] = "ball"
    with pytest.raises(ValueError, match="configured static but moves"):
        inspect_video(video, tmp_path, settings)


def test_static_objects_cannot_hide_a_bad_moving_object_reconstruction(tmp_path, monkeypatch):
    # Isolate the scorer: a newly available velocity repair can legitimately fix this injected error.
    monkeypatch.setattr('weft.initial_motion.propose_initial_motion',
                        lambda *args, **kwargs: (None, {'status': 'unavailable', 'accepted': False}))
    video = tmp_path / "fixture.mp4"
    write_clip(video)
    evidence, scene = inspect_video(video, tmp_path, config())
    next(obj for obj in scene["objects"] if obj["id"] == "puck")["velocity"] = [0.09, 0]
    fit_scene(evidence, scene, tmp_path)
    report = json.loads((tmp_path / "reconstruction.json").read_text())
    assert report["after"]["rmse_image_width"] < 0.08
    assert report["after"]["per_object"]["puck"]["rmse_image_width"] > 0.08
    assert not report["replay_gate"]["passes"]
    assert not report["replay_gate"]["per_object_passes"]["puck"]


def test_repair_cannot_trade_a_large_object_regression_for_an_average_win(tmp_path, monkeypatch):
    video = tmp_path / "fixture.mp4"
    write_clip(video)
    evidence, scene = inspect_video(video, tmp_path, config())

    def score(_evidence, candidate):
        errors = {"puck": 0.01, "ball": 0.10, "obstacle": 0}
        if candidate["damping"] == 1 or any('velocity_repair' in obj['provenance'] for obj in candidate['objects']):
            errors.update(puck=0.04, ball=0.02)
        average = (sum(error ** 2 for error in errors.values()) / 3) ** 0.5
        return {"rmse_image_width": average, "rmse_px": average * 200, "finite": True,
                "per_object": {key: {"rmse_image_width": value} for key, value in errors.items()}}

    monkeypatch.setattr(vision, "reconstruction_metrics", score)
    selected = fit_scene(evidence, scene, tmp_path)
    ledger = json.loads((tmp_path / "repair-ledger.json").read_text())
    assert ledger[0]["relative_improvement"] > 0
    assert not ledger[0]["accepted"]
    assert ledger[0]["regressed_objects"] == ["puck"]
    motion = [row for row in ledger if 'initial_velocity' in row['proposal']]
    assert len(motion) == 1
    assert motion[0]['relative_improvement'] > 0
    assert motion[0]['regressed_objects'] == ['puck']
    assert not motion[0]['accepted']
    assert selected["damping"] == scene["damping"]
