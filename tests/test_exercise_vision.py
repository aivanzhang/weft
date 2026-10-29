import json
import math
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

import weft.exercise_vision as vision


def world_pose(angle=0):
    points = np.zeros((33, 3))
    points[11], points[12] = [-0.20, -0.50, 0], [0.20, -0.50, 0]
    points[23], points[24] = [-0.13, 0, 0], [0.13, 0, 0]
    for shoulder, elbow, wrist, hip, knee, ankle in ((11, 13, 15, 23, 25, 27), (12, 14, 16, 24, 26, 28)):
        points[elbow] = points[shoulder] + [0, 0.30 * math.cos(angle), 0.30 * math.sin(angle)]
        points[wrist] = points[elbow] + [0, 0.25 * math.cos(angle), 0.25 * math.sin(angle)]
        points[knee] = points[hip] + [0, 0.42, 0]
        points[ankle] = points[knee] + [0, 0.42, 0]
    return [{"x": float(x), "y": float(y), "z": float(z), "visibility": 1.0, "presence": 1.0} for x, y, z in points]


def inference_result(angle=0, *, count=1, hidden=()):
    world = world_pose(angle)
    image = [{"x": 0.5 + row["x"] * 0.5, "y": 0.3 + row["y"] * 0.5, "z": row["z"],
              "visibility": 0.2 if index in hidden else 1.0, "presence": 1.0} for index, row in enumerate(world)]
    return SimpleNamespace(pose_landmarks=[[SimpleNamespace(**row) for row in image] for _ in range(count)],
                           pose_world_landmarks=[[SimpleNamespace(**row) for row in world] for _ in range(count)])


def video(path):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30, (160, 120))
    assert writer.isOpened()
    for _ in range(30):
        writer.write(np.full((120, 160, 3), 220, np.uint8))
    writer.release()


def test_geometry_uses_forward_positive_angles_and_zero_straight_flexion():
    result = vision.pose_geometry(world_pose(math.pi / 3))
    assert result["joints"]["left_shoulder"] == pytest.approx(math.pi / 3)
    assert result["joints"]["right_shoulder"] == pytest.approx(math.pi / 3)
    assert result["joints"]["left_elbow"] == pytest.approx(0)
    assert result["joints"]["right_knee"] == pytest.approx(0)
    assert result["lengths"]["torso"] == pytest.approx(0.5)
    assert result["lengths"]["upper_arm"] == pytest.approx(0.3)
    pose = world_pose()
    pose[15].update(y=-0.2, z=0.25)
    assert vision.pose_geometry(pose)["joints"]["left_elbow"] == pytest.approx(math.pi / 2)


@pytest.mark.parametrize("bad", [[], [{}] * 33, [{"x": float("nan"), "y": 0, "z": 0}] * 33,
                               [{"x": 0, "y": 0, "z": 0}] * 33])
def test_invalid_pose_geometry_fails_explicitly(bad):
    with pytest.raises(ValueError):
        vision.pose_geometry(bad)


def test_missing_multiple_and_hidden_people_are_not_silently_selected():
    assert vision.summarize_result(inference_result(count=0))["status"] == "no_pose"
    multiple = vision.summarize_result(inference_result(count=2))
    assert multiple["status"] == "multiple_people" and len(multiple["poses"]) == 2
    assert multiple["landmarks"] == []
    partial = vision.summarize_result(inference_result(hidden=(14, 16)))
    assert partial["status"] == "tracked_partial"
    assert partial["joint_confidence"]["right_elbow"] == 0.2
    assert partial["joint_confidence"]["left_elbow"] == 1
    poor = vision.summarize_result(inference_result(hidden=(11, 12, 23, 24)))
    assert poor["status"] == "low_confidence"
    mismatched = inference_result()
    mismatched.pose_world_landmarks = []
    assert vision.summarize_result(mismatched)["status"] == "invalid_landmark_count"


def test_temporal_identity_recomputes_geometry_and_preserves_raw_source(monkeypatch, tmp_path):
    from weft.pose_identity import IDENTITY_INDEX_MAPPING
    def sample(index):
        result=inference_result(angle=index*.04)
        if index in (5,6):
            result.pose_landmarks[0]=[result.pose_landmarks[0][i] for i in IDENTITY_INDEX_MAPPING]
            result.pose_world_landmarks[0]=[result.pose_world_landmarks[0][i] for i in IDENTITY_INDEX_MAPPING]
        return result
    stub_detector(monkeypatch,sample)
    source=tmp_path/'source.mp4';video(source)
    out=tmp_path/'out';out.mkdir()
    evidence=vision.inspect_exercise(source,out,tmp_path/'model',pose_identity='temporal')
    audit=json.loads((out/'identity-audit.json').read_text())
    raw=json.loads((out/'pose-landmarks.json').read_text())['frames']
    assert len(audit['repaired_evidence_ids'])==2
    for index in (5,6):
        corrected=evidence['frames'][index]
        assert corrected['identity_raw']==raw[index]
        assert corrected['world_landmarks'][11]==raw[index]['world_landmarks'][12]
        assert corrected['joints']['left_shoulder']==pytest.approx(index*.04)
        assert corrected['identity_recompute_required'] is False
        assert evidence['reference'][index]['joints']==corrected['joints']
    assert evidence['frames'][4]==raw[4]
    assert evidence['pose_identity']['repaired_frames']==2


@pytest.mark.parametrize("offset,confidence", [(0, 1), (0.25, 1), (0.02, 0.2)])
def test_competing_poses_are_not_merged_by_overlap_or_confidence(offset, confidence):
    result = inference_result(count=2)
    for point in result.pose_landmarks[1]:
        point.x += offset
        point.visibility = confidence
    record = vision.summarize_result(result)
    assert record["status"] == "multiple_people"
    assert record["identity_status"] == "unresolved_multiple_pose_hypotheses"
    assert record["landmarks"] == [] and record["world_landmarks"] == []
    assert len(record["poses"]) == 2
    assert record["poses"][1]["landmarks"][11]["x"] == result.pose_landmarks[1][11].x
    assert record["poses"][1]["landmarks"][11]["visibility"] == confidence


def stub_detector(monkeypatch, result_for_sample):
    class Detector:
        def __init__(self):
            self.sample = 0
            self.closed = False
            self.timestamps = []
            self.image_calls = 0

        def detect(self, image):
            self.image_calls += 1
            return self._infer()

        def detect_for_video(self, image, timestamp):
            self.timestamps.append(timestamp)
            return self._infer()

        def _infer(self):
            result = result_for_sample(self.sample)
            self.sample += 1
            return result

        def close(self):
            self.closed = True

    detector = Detector()
    mp = SimpleNamespace(Image=lambda **kwargs: kwargs, ImageFormat=SimpleNamespace(SRGB=1))
    monkeypatch.setattr(vision, "_detector", lambda _, pose_mode: (mp, detector))
    monkeypatch.setattr(vision, "model_provenance", lambda _: {"name": "Synthetic detector test", "delegate": "CPU"})
    return detector


def test_inference_preserves_dropouts_confidence_and_source_order(tmp_path, monkeypatch):
    source = tmp_path / "synthetic-test.mp4"
    video(source)
    detector = stub_detector(monkeypatch, lambda sample: inference_result(count=0) if sample < 2
                             else inference_result(sample / 20, hidden=(14, 16)))
    evidence = vision.inspect_exercise(source, tmp_path, "unused-test-model")
    assert evidence["decoded_frames"] == 30
    assert len(evidence["frames"]) == 15
    assert [frame["frame"] for frame in evidence["frames"]] == list(range(0, 30, 2))
    assert [frame["status"] for frame in evidence["frames"][:2]] == ["no_pose", "no_pose"]
    assert evidence["quality"]["tracked_frames"] == 13
    assert evidence["quality"]["joint_usable_fraction"]["right_elbow"] == 0
    assert evidence["quality"]["joint_range_radians"]["right_elbow"] == 0
    assert evidence["reference"][0]["time_s"] == 4 / 30
    assert evidence["reference"][0]["pose_status"] == "tracked_partial"
    assert evidence["frames"][2]["identity_status"] == "single_model_hypothesis_not_verified_person_count"
    assert detector.closed
    assert detector.timestamps == [round(index * 1000 / 30) for index in range(0, 30, 2)]
    assert detector.image_calls == 0
    assert evidence["model"]["running_mode"] == "VIDEO"
    assert evidence["model"]["temporal_tracking"] is True
    for file in ("pose-landmarks.json", "joint-angles.json", "evidence.json", "tracks.gif", "tracks-contact-sheet.png"):
        assert (tmp_path / file).stat().st_size > 0


def test_image_mode_is_explicit_and_does_not_reuse_temporal_tracking(tmp_path, monkeypatch):
    source = tmp_path / "synthetic-image-mode.mp4"
    video(source)
    detector = stub_detector(monkeypatch, lambda sample: inference_result(sample / 20))
    evidence = vision.inspect_exercise(source, tmp_path, "unused-test-model", pose_mode="image")
    assert detector.timestamps == [] and detector.image_calls == 15
    assert evidence["model"]["running_mode"] == "IMAGE"
    assert evidence["model"]["temporal_tracking"] is False
    assert evidence["frames"][-1]["time_s"] == 28 / 30


def test_unknown_pose_mode_fails_before_model_or_output_access(tmp_path):
    with pytest.raises(ValueError, match="Pose mode"):
        vision.inspect_exercise("missing-video", tmp_path, "missing-model", pose_mode="automatic")
    assert list(tmp_path.iterdir()) == []


def test_inference_error_closes_detector_and_preserves_prior_samples(tmp_path, monkeypatch):
    source = tmp_path / "synthetic-error.mp4"
    video(source)

    def infer(sample):
        if sample == 3:
            raise RuntimeError("inference interrupted")
        return inference_result(sample / 20)

    detector = stub_detector(monkeypatch, infer)
    with pytest.raises(RuntimeError, match="inference interrupted"):
        vision.inspect_exercise(source, tmp_path, "unused-test-model")
    assert detector.closed
    saved = json.loads((tmp_path / "evidence.json").read_text())
    assert [frame["frame"] for frame in saved["frames"]] == [0, 2, 4]
    assert "quality" not in saved


def test_static_pose_is_not_claimed_as_dynamic_exercise(tmp_path, monkeypatch):
    source = tmp_path / "synthetic-still.mp4"
    video(source)
    detector = stub_detector(monkeypatch, lambda _: inference_result())
    with pytest.raises(vision.UnsupportedVideo, match="Insufficient measurable exercise motion"):
        vision.inspect_exercise(source, tmp_path, "unused-test-model")
    saved = json.loads((tmp_path / "evidence.json").read_text())
    assert not saved["quality"]["passes"]
    assert len(saved["frames"]) == 15
    assert detector.closed


def test_no_person_saves_negative_evidence_instead_of_inventing_joints(tmp_path, monkeypatch):
    source = tmp_path / "synthetic-empty.mp4"
    video(source)
    stub_detector(monkeypatch, lambda _: inference_result(count=0))
    with pytest.raises(vision.UnsupportedVideo, match="Insufficient full-body visibility"):
        vision.inspect_exercise(source, tmp_path, "unused-test-model")
    saved = json.loads((tmp_path / "evidence.json").read_text())
    assert saved["lengths"] == {} and saved["reference"] == []
    assert saved["quality"]["status_counts"] == {"no_pose": 15}


def test_one_multi_person_frame_invalidates_an_otherwise_usable_clip(tmp_path, monkeypatch):
    source = tmp_path / "synthetic-two-people.mp4"
    video(source)
    stub_detector(monkeypatch, lambda sample: inference_result(sample / 20, count=2 if sample == 4 else 1))
    with pytest.raises(vision.UnsupportedVideo, match="Multiple people"):
        vision.inspect_exercise(source, tmp_path, "unused-test-model")
    saved = json.loads((tmp_path / "evidence.json").read_text())
    assert saved["quality"]["tracked_frames"] == 14
    assert saved["quality"]["status_counts"]["multiple_people"] == 1
    assert len(saved["frames"][4]["poses"]) == 2


def test_model_hash_is_required_and_existing_files_are_not_replaced(tmp_path):
    model = tmp_path / "pose.task"
    model.write_bytes(b"invalid model")
    with pytest.raises(ValueError, match="pinned official"):
        vision.fetch_pose_model(model)
    assert model.read_bytes() == b"invalid model"


def test_existing_exercise_evidence_is_not_overwritten(tmp_path, monkeypatch):
    source = tmp_path / "synthetic.mp4"
    video(source)
    stub_detector(monkeypatch, lambda _: inference_result())
    saved = tmp_path / "evidence.json"
    saved.write_text("preserved prior run")
    with pytest.raises(FileExistsError, match="new run"):
        vision.inspect_exercise(source, tmp_path, "unused-test-model")
    assert saved.read_text() == "preserved prior run"


def test_out_of_frame_joint_is_not_reliable_despite_high_model_confidence():
    result = inference_result()
    result.pose_landmarks[0][16].x = 1.2
    record = vision.summarize_result(result)
    assert record['status'] == 'tracked_partial'
    assert record['landmarks'][16]['visibility'] == 1
    assert record['joint_confidence']['right_elbow'] == 0
    assert record['joint_confidence']['left_elbow'] == 1


def _ebml(identifier, payload):
    size = len(payload)
    width = next(n for n in range(1, 9) if size < (1 << (7 * n)) - 1)
    return bytes.fromhex(identifier) + (size | (1 << (7 * width))).to_bytes(width, 'big') + payload


def _webm_blocks(*, laced_video=False):
    track = lambda n, kind: _ebml('ae', _ebml('d7', bytes([n])) + _ebml('83', bytes([kind])))
    tracks = _ebml('1654ae6b', track(1, 1) + track(2, 2))
    video = _ebml('a3', b'\x81\x00\x00' + bytes([2 if laced_video else 0]) + b'frame')
    audio = _ebml('a3', b'\x82\x00\x00\x02audio')
    grouped = _ebml('a0', _ebml('a1', b'\x81\x00\x01\x00frame'))
    return _ebml('1a45dfa3', b'') + _ebml('18538067', tracks + _ebml('1f43b675', video + audio + grouped))


def test_encoded_frame_count_separates_audio_and_video_and_checks_bounds(tmp_path):
    path = tmp_path / 'blocks.webm'
    path.write_bytes(_webm_blocks())
    assert vision._webm_video_frame_count(path) == 2
    path.write_bytes(_webm_blocks()[:-1])
    with pytest.raises(ValueError, match='Truncated'):
        vision._webm_video_frame_count(path)
    path.write_bytes(_webm_blocks(laced_video=True))
    with pytest.raises(ValueError, match='unlaced'):
        vision._webm_video_frame_count(path)
    path.write_bytes(b'\x00')
    with pytest.raises(ValueError, match='Invalid EBML'):
        vision._webm_video_frame_count(path)


def test_model_coverage_discloses_horizontal_torso_not_captured_by_anchored_body():
    frames = []
    for index in range(8):
        result = inference_result(index / 10)
        for point in result.pose_world_landmarks[0]:
            point.x, point.y = -point.y, point.x
        frames.append(vision.summarize_result(result) | {'time_s': index / 10})
    quality = vision._quality({'frames': frames})
    assert quality['passes']
    assert quality['model_coverage']['torso_tilt_from_camera_up_p95_radians'] == pytest.approx(math.pi / 2)
    assert any('Legacy upright anchoring' in warning for warning in quality['warnings'])
    assert quality['model_coverage']['reference_model'] == 'legacy-eight-hinge-upright'
    assert quality['model_coverage']['unrepresented_motion']
