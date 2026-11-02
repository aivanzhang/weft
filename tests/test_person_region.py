"""Explicit attention regions preserve source identity and never select competing poses."""

from copy import deepcopy
import json
from types import SimpleNamespace

import pytest

from test_exercise_body_source import pose_run, seal, forbid_runtime
from test_exercise_vision import inference_result, stub_detector, video
from weft import exercise_body_source as body
from weft import exercise_vision as vision
from weft.io import digest, write_json
from weft.pipeline import verify_artifacts


@pytest.mark.parametrize("region", [None, [], [1, 2, 3], [0, 0, 64, 64, 1], [False, 0, 64, 64],
    [0., 0, 64, 64], [-1, 0, 64, 64], [0, 0, 63, 64], [0, 0, 64, 63],
    [50, 0, 40, 100], [0, 0, 161, 120], [0, 0, 160, 121]])
def test_invalid_region_fails_before_inference(region):
    with pytest.raises(ValueError, match="Person region"):
        vision.validate_person_region(region, 160, 120)


def test_mapping_keeps_raw_coordinates_depth_world_and_outside_region_confidence():
    result = inference_result()
    result.pose_landmarks[0][16].x = 1.2
    original = vision.summarize_result(result)
    before = deepcopy(original)
    mapped = vision.map_person_region(original, [32, 8, 128, 112], 160, 120)
    assert original == before == mapped["region_raw"]
    assert mapped["landmarks"][16]["x"] == pytest.approx(.92)
    assert mapped["landmarks"][16]["visibility"] == 0
    assert mapped["landmarks"][16]["input_region_visibility"] == "outside_region"
    assert mapped["region_raw"]["landmarks"][16]["visibility"] == 1
    assert mapped["joint_confidence"]["right_elbow"] == 0
    assert mapped["status"] == "tracked_partial"
    assert mapped["world_landmarks"] == original["world_landmarks"]
    from weft.multiview_pose import _observation
    assert _observation(mapped, 16, 160, 120)[0] is None
    for i, point in enumerate(original["landmarks"]):
        assert mapped["landmarks"][i]["y"] == pytest.approx((8 + point["y"] * 104) / 120)
        assert mapped["landmarks"][i]["z"] == pytest.approx(point["z"] * .6)


@pytest.mark.parametrize("count", [0, 2])
def test_region_does_not_choose_or_fabricate_a_person(count):
    original = vision.summarize_result(inference_result(count=count))
    mapped = vision.map_person_region(original, [32, 8, 128, 112], 160, 120)
    assert mapped["status"] == original["status"] and mapped["pose_count"] == count
    assert mapped["landmarks"] == [] and len(mapped["poses"]) == count
    for prior, current in zip(original["poses"], mapped["poses"], strict=True):
        assert current["world_landmarks"] == prior["world_landmarks"]
        assert current["landmarks"][0]["x"] == pytest.approx((32 + prior["landmarks"][0]["x"] * 96) / 160)


@pytest.mark.parametrize("identity", ["raw", "temporal"])
def test_inference_observes_region_but_retains_full_video_and_all_source_slots(tmp_path, monkeypatch, identity):
    source = tmp_path / "source.mp4"; video(source)
    detector = stub_detector(monkeypatch, lambda i: inference_result(i / 20))
    original_detector = vision._detector
    shapes = []

    def create(*args, **kwargs):
        mp, instance = original_detector(*args, **kwargs)
        def image(**values):
            shapes.append(values["data"].shape)
            return values
        return SimpleNamespace(Image=image, ImageFormat=mp.ImageFormat), instance

    monkeypatch.setattr(vision, "_detector", create)
    result = vision.inspect_exercise(source, tmp_path, "stub", person_region=[32, 8, 128, 112],
                                     pose_identity=identity)
    assert detector.closed and shapes == [(104, 96, 3)] * 15
    assert (result["width"], result["height"], result["frame_count"], result["decoded_frames"]) == (160, 120, 30, 30)
    assert result["source_sha256"] == digest(source)
    assert [row["frame"] for row in result["frames"]] == list(range(0, 30, 2))
    assert all(row["id"] == f'{digest(source)}:frame:{row["frame"]}' for row in result["frames"])
    raw = json.loads((tmp_path / "pose-landmarks.json").read_text())
    vision.validate_person_region_evidence(result, raw, [32, 8, 128, 112])


def region_pose_run(root):
    pose_run(root)
    evidence = json.loads((root / "evidence.json").read_text())
    region = [128, 64, 512, 448]
    evidence["person_region"] = vision.person_region_metadata(region)
    evidence["model"]["person_region"] = evidence["person_region"]
    keys = {"frame", "time_s", "decoder_timestamp_s", "id", "inference_seconds"}
    evidence["frames"] = [vision.map_person_region({k: v for k, v in row.items() if k not in keys},
        region, 640, 480) | {k: row[k] for k in keys} for row in evidence["frames"]]
    write_json(root / "pose-landmarks.json", {"frames": evidence["frames"], "model": evidence["model"]})
    vision.resolve_pose_identity(evidence, root, "raw")
    vision.finalize_evidence(evidence, root)
    write_json(root / "model.json", evidence["model"])
    config = json.loads((root / "run-config.json").read_text())
    config["person_region"] = region; write_json(root / "run-config.json", config)
    seal(root)
    return root, region


def test_region_evidence_survives_saved_body_construction_without_reinference(tmp_path, monkeypatch):
    original, region = region_pose_run(tmp_path / "poses")
    forbid_runtime(monkeypatch)
    out = body.prepare_exercise_body_source_from_evidence(original, tmp_path / "body")
    verify_artifacts(out)
    receipt = json.loads((out / "body-source.json").read_text())
    assert receipt["pose_options"]["person_region"] == region
    assert (out / "pose-landmarks.json").read_bytes() == (original / "pose-landmarks.json").read_bytes()


@pytest.mark.parametrize("changed", ["config", "model", "evidence", "mapping", "missing_raw", "omitted_region"])
def test_resealed_region_mismatch_is_rejected(tmp_path, monkeypatch, changed):
    original, region = region_pose_run(tmp_path / "poses")
    raw_files = {p.name: p.read_bytes() for p in original.iterdir() if p.is_file()}
    if changed == "config":
        name = "run-config.json"; value = json.loads(raw_files[name]); value["person_region"][0] += 1
    elif changed == "model":
        name = "model.json"; value = json.loads(raw_files[name]); value["person_region"]["box_xyxy"][0] += 1
    elif changed == "evidence":
        name = "evidence.json"; value = json.loads(raw_files[name]); value.pop("person_region")
    else:
        name = "pose-landmarks.json"; value = json.loads(raw_files[name])
        if changed == "mapping": value["frames"][2]["landmarks"][0]["x"] += .01
        elif changed == "missing_raw": value["frames"][2].pop("region_raw")
        else: region = None
    raw_files[name] = json.dumps(value).encode()
    # Repair the raw-file receipt so this test reaches region validation, not the generic hash gate.
    if name == "pose-landmarks.json":
        import hashlib
        checksum = hashlib.sha256(raw_files[name]).hexdigest()
        for artifact in ("evidence.json", "identity-audit.json"):
            document = json.loads(raw_files[artifact])
            target = document["pose_identity"] if artifact == "evidence.json" else document
            target["raw_artifact_sha256"] = checksum
            raw_files[artifact] = json.dumps(document).encode()
    forbid_runtime(monkeypatch)
    with pytest.raises(ValueError, match="[Pp]ose|[Pp]erson|[Rr]egion"):
        body._validated_pose(raw_files, body._options("video", "raw", person_region=region), digest(original / "source.webm"))
