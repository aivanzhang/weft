"""Real local decoding with analytical pose estimates tests timing custody, not anatomy."""

import json

import pytest

from test_exercise_vision import inference_result, stub_detector, video
from test_exercise_body_source import model_metadata
from weft import exercise_body_source as source, exercise_vision, source_clock_capture
from weft.io import digest


def inputs(tmp_path, monkeypatch):
    movie, model = tmp_path / "video.mp4", tmp_path / "pose.task"
    video(movie)
    model.write_bytes(b"analytical pose model fixture")
    detector = stub_detector(monkeypatch, lambda i: inference_result(i * .01))
    monkeypatch.setitem(exercise_vision.MODEL_VARIANTS, "lite", {
        **exercise_vision.MODEL_VARIANTS["lite"], "sha256": digest(model), "bytes": model.stat().st_size})
    monkeypatch.setattr(exercise_vision, "model_provenance", lambda _: model_metadata())
    return movie, model, detector


def test_raw_video_captures_all_slots_and_seals_original_and_derived_clocks(tmp_path, monkeypatch):
    movie, model, detector = inputs(tmp_path, monkeypatch)
    before = movie.read_bytes()
    out = tmp_path / "body"
    source.prepare_exercise_body_source(movie, out, pose_model=model, native_timing=True)
    assert detector.closed and movie.read_bytes() == before
    evidence = json.loads((out / "evidence.json").read_bytes())
    clock = json.loads((out / "decoder-clock.json").read_bytes())
    marker = json.loads((out / "body-source.json").read_bytes())
    assert len(clock["rows"]) == evidence["decoded_frames"] == 30
    assert len(evidence["frames"]) == 15
    assert [r["time_s"] for r in evidence["frames"]] == [i / 30 for i in range(0, 30, 2)]
    assert marker["work"]["video_decodes"] == 2
    assert marker["work"]["inference_calls"] == 15
    assert marker["work"]["physics_steps"] == 0
    assert source.CAPTURE_FILES <= marker["artifacts"].keys()
    files = {name: (out / name).read_bytes() for name in json.loads((out / "manifest.json").read_bytes())["files"]}
    source.validate_body_source_files(files)
    for name in source.CAPTURE_FILES:
        broken = dict(files)
        broken.pop(name)
        with pytest.raises(ValueError):
            source.validate_body_source_files(broken)
    reused = tmp_path / "reused"
    source.prepare_exercise_body_source_from_evidence(out, reused, native_clock=out / "native-clock.json")
    assert all((out / name).read_bytes() == (reused / name).read_bytes() for name in source.CAPTURE_FILES)
    assert json.loads((reused / "body-source.json").read_bytes())["work"]["video_decodes"] == 0


def test_capture_failure_keeps_original_pose_and_full_decoder_evidence(tmp_path, monkeypatch):
    movie, model, detector = inputs(tmp_path, monkeypatch)
    def fail(*args, **kwargs):
        raise ValueError("Analytical native decoder failure")
    monkeypatch.setattr(source_clock_capture, "collect_native_clock", fail)
    out = tmp_path / "body"
    with pytest.raises(ValueError, match="Analytical native decoder failure"):
        source.prepare_exercise_body_source(movie, out, pose_model=model, native_timing=True)
    assert detector.closed
    assert json.loads((out / "decoder-clock.json").read_bytes())["complete"] is True
    assert (out / "evidence.json").is_file() and (out / "pose-landmarks.json").is_file()
    assert not (out / "human.json").exists()
    assert json.loads((out / "failure.json").read_bytes())["status"] == "failed"


@pytest.mark.parametrize("value", [1, "true", None])
def test_capture_mode_requires_a_boolean_before_output(tmp_path, monkeypatch, value):
    movie, model, _ = inputs(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="explicitly enabled"):
        source.prepare_exercise_body_source(movie, tmp_path / "out", pose_model=model, native_timing=value)
    assert not (tmp_path / "out").exists()
