"""Clock capture observes every decoded frame without changing detector time or evidence."""

import json

import numpy as np
import pytest

from test_exercise_vision import inference_result, stub_detector
from weft import exercise_vision as vision


class Capture:
    def __init__(self, *, frame_count=30, missing_stamp=None, fail_read=None):
        self.frame_count = frame_count
        self.frame = -1
        self.missing_stamp = missing_stamp
        self.fail_read = fail_read
        self.released = False
        self.timestamp_reads = []

    def isOpened(self):
        return True

    def get(self, prop):
        if prop == vision.cv2.CAP_PROP_POS_MSEC:
            self.timestamp_reads.append(self.frame)
            return float("nan") if self.frame == self.missing_stamp else 1003 + self.frame * 41
        return {vision.cv2.CAP_PROP_FPS: 30, vision.cv2.CAP_PROP_FRAME_COUNT: 30,
                vision.cv2.CAP_PROP_FRAME_WIDTH: 160, vision.cv2.CAP_PROP_FRAME_HEIGHT: 120}[prop]

    def read(self):
        if self.frame + 1 == self.fail_read:
            raise RuntimeError("decoder read interrupted")
        if self.frame + 1 >= self.frame_count:
            return False, None
        self.frame += 1
        return True, np.full((120, 160, 3), 220, np.uint8)

    def release(self):
        self.released = True


def setup(monkeypatch, tmp_path, **kwargs):
    source = tmp_path / "source.mp4"
    source.write_bytes(b"mocked source; no media model or decode")
    capture = Capture(**kwargs)
    monkeypatch.setattr(vision.cv2, "VideoCapture", lambda _: capture)
    monkeypatch.setattr(vision.time, "perf_counter", lambda: 0.)
    detector = stub_detector(monkeypatch, lambda sample: inference_result(sample / 20))
    return source, capture, detector


@pytest.mark.parametrize("pose_mode", ["video", "image"])
def test_sink_sees_unsampled_frames_before_inference_without_retiming(monkeypatch, tmp_path, pose_mode):
    source, capture, detector = setup(monkeypatch, tmp_path)
    rows, completed_inferences = [], []

    def sink(row):
        rows.append(row)
        completed_inferences.append(detector.sample)

    evidence = vision.inspect_exercise(source, tmp_path, "mocked-model", pose_mode=pose_mode, decoder_clock_sink=sink)
    assert rows == [{"frame": frame, "decoder_timestamp_s": (1003 + frame * 41) / 1000} for frame in range(30)]
    assert completed_inferences == [(frame + 1) // 2 for frame in range(30)]
    assert [frame["frame"] for frame in evidence["frames"]] == list(range(0, 30, 2))
    assert [frame["time_s"] for frame in evidence["frames"]] == [frame / 30 for frame in range(0, 30, 2)]
    assert detector.timestamps == ([round(frame * 1000 / 30) for frame in range(0, 30, 2)] if pose_mode == "video" else [])
    assert capture.released and detector.closed
    assert evidence["decoded_frames"] == 30


def test_callback_does_not_change_any_saved_artifact_bytes(monkeypatch, tmp_path):
    snapshots = []
    reads = []
    for name in ("default", "none", "callback"):
        root = tmp_path / name
        root.mkdir()
        source, capture, _ = setup(monkeypatch, root)
        kwargs = {} if name == "default" else {"decoder_clock_sink": None if name == "none" else lambda row: None}
        vision.inspect_exercise(source, root, "mocked-model", **kwargs)
        snapshots.append({path.name: path.read_bytes() for path in root.iterdir()})
        reads.append(capture.timestamp_reads)
    assert snapshots[0] == snapshots[1] == snapshots[2]
    assert reads[0] == reads[1] == list(range(0, 30, 2))
    assert reads[2] == [frame for frame in range(30) for _ in range(2 if frame % 2 == 0 else 1)]


@pytest.mark.parametrize("failure_frame", [0, 3, 4])
def test_sink_failure_releases_resources_and_preserves_actual_decode_count(monkeypatch, tmp_path, failure_frame):
    source, capture, detector = setup(monkeypatch, tmp_path)
    rows = []

    def sink(row):
        rows.append(row)
        if row["frame"] == failure_frame:
            raise RuntimeError("clock sink interrupted")

    with pytest.raises(RuntimeError, match="clock sink interrupted"):
        vision.inspect_exercise(source, tmp_path, "mocked-model", decoder_clock_sink=sink)
    evidence = json.loads((tmp_path / "evidence.json").read_bytes())
    assert evidence["decoded_frames"] == failure_frame + 1
    assert [row["frame"] for row in evidence["frames"]] == list(range(0, failure_frame, 2))
    assert [row["frame"] for row in rows] == list(range(failure_frame + 1))
    assert (tmp_path / "pose-landmarks.json").is_file()
    assert not (tmp_path / "joint-angles.json").exists()
    assert "quality" not in evidence and "decode_metadata_discrepancy" not in evidence
    assert capture.released and detector.closed


def test_failed_read_does_not_emit_or_increment_clock_frame(monkeypatch, tmp_path):
    source, capture, detector = setup(monkeypatch, tmp_path, fail_read=5)
    rows = []
    with pytest.raises(RuntimeError, match="decoder read interrupted"):
        vision.inspect_exercise(source, tmp_path, "mocked-model", decoder_clock_sink=rows.append)
    assert len(rows) == 5 and [row["frame"] for row in rows] == list(range(5))
    assert json.loads((tmp_path / "evidence.json").read_bytes())["decoded_frames"] == 5
    assert capture.released and detector.closed


def test_unknown_decoder_stamp_remains_unknown_and_sink_cannot_mutate_pose(monkeypatch, tmp_path):
    source, _, _ = setup(monkeypatch, tmp_path, missing_stamp=2)
    rows = []

    def sink(row):
        rows.append(dict(row))
        row.update(frame=999, decoder_timestamp_s=900.)

    evidence = vision.inspect_exercise(source, tmp_path, "mocked-model", decoder_clock_sink=sink)
    assert rows[2] == {"frame": 2, "decoder_timestamp_s": None}
    assert evidence["frames"][1]["frame"] == 2
    assert evidence["frames"][1]["decoder_timestamp_s"] is None
    assert evidence["frames"][1]["time_s"] == 2 / 30


def test_invalid_callback_fails_before_model_or_capture_access(tmp_path):
    with pytest.raises(ValueError, match="Decoder clock sink must be callable"):
        vision.inspect_exercise("missing-video", tmp_path, "missing-model", decoder_clock_sink=[])
    assert not list(tmp_path.iterdir())
