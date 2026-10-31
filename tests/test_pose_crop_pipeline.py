"""Native-stage custody and bounds with analytical detector outputs, never real inference."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import time

import cv2
import mujoco
import numpy as np
import pytest

from test_exercise_vision import inference_result
from weft import exercise_vision as vision, human, pose_crop_pipeline as pipeline
from weft import pose_source_recovery as recovery
from weft.io import digest


def read(path):
    return json.loads(path.read_bytes())


def files(root):
    return {p.relative_to(root).as_posix(): p.read_bytes() for p in root.rglob("*") if p.is_file()}


def native_fixture(tmp_path, monkeypatch, *, missing=(30,), competing=(), crop_competing=False,
                   second_fault=None, detector_failure=False, unknown_clock=None, real_video=False, count=63,
                   metadata_count=None):
    video, model = tmp_path / ("input.webm" if metadata_count is not None else "input.mp4"), tmp_path / "full.task"
    model.write_bytes(b"analytical model fixture, never executed")
    monkeypatch.setitem(vision.MODEL_VARIANTS, "full", {"sha256": digest(model), "bytes": model.stat().st_size})
    calls = {"decoders": 0, "detectors": 0, "baseline": 0, "crops": 0, "closed": 0, "released": 0}
    if real_video:
        writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 30, (640, 480))
        assert writer.isOpened()
        for frame in range(count):writer.write(np.full((480, 640, 3), frame % 256, dtype=np.uint8))
        writer.release()
    else:
        video.write_bytes(b"analytical sequential decoder fixture")

        class Capture:
            def __init__(self, _path):
                self.position = 0; self.pass_index = calls["decoders"]; calls["decoders"] += 1
            def isOpened(self):return True
            def get(self, key):
                values = {cv2.CAP_PROP_FRAME_WIDTH: 640, cv2.CAP_PROP_FRAME_HEIGHT: 480,
                          cv2.CAP_PROP_FPS: 30, cv2.CAP_PROP_FRAME_COUNT: count if metadata_count is None else metadata_count}
                if key in values:
                    return values[key] + (1 if self.pass_index and second_fault == "metadata" and key == cv2.CAP_PROP_FRAME_WIDTH else 0)
                if key == cv2.CAP_PROP_POS_FRAMES:
                    return self.position + (1 if self.pass_index and second_fault == "index" else 0)
                if key == cv2.CAP_PROP_POS_MSEC:
                    if self.position - 1 == unknown_clock:return float("nan")
                    return (self.position - 1) * 1000 / 30 + (1 if self.pass_index and second_fault == "clock" else 0)
                raise AssertionError(key)
            def read(self):
                limit = count - 1 if self.pass_index and second_fault == "short" else count + 1 if self.pass_index and second_fault == "extra" else count
                if self.position == limit:return False, None
                self.position += 1
                return True, np.full((480, 640, 3), (self.position - 1) % 256, dtype=np.uint8)
            def release(self):calls["released"] += 1
        monkeypatch.setattr(vision.cv2, "VideoCapture", Capture)
        if metadata_count is not None:
            monkeypatch.setattr(vision, "_webm_video_frame_count", lambda _path: count)

    def detector(_model, mode):
        assert mode == "image"
        phase = calls["detectors"]; calls["detectors"] += 1
        class Detector:
            def detect(self, _image):
                if phase == 0:
                    frame = calls["baseline"] * 2; calls["baseline"] += 1
                    return inference_result(frame * .005, count=0 if frame in missing else 2 if frame in competing else 1)
                calls["crops"] += 1
                if detector_failure:raise RuntimeError("analytical native crop failure")
                frame = missing[calls["crops"] - 1]
                result = inference_result(frame * .005, count=2 if crop_competing else 1)
                if not crop_competing:
                    plan = read(tmp_path / "proposals/crop-plan.json")
                    crop = next(x["crop"] for x in plan["plans"] if x["frame"] == frame)
                    x1, y1, x2, y2 = crop["box_xyxy"]
                    for point in result.pose_landmarks[0]:
                        point.x = (point.x * 640 - x1) / (x2 - x1)
                        point.y = (point.y * 480 - y1) / (y2 - y1)
                        point.z *= 640 / (x2 - x1)
                return result
            def close(self):calls["closed"] += 1
        return SimpleNamespace(Image=lambda **kw:kw, ImageFormat=SimpleNamespace(SRGB=1)), Detector()

    monkeypatch.setattr(vision, "_detector", detector)
    def forbidden(*args, **kwargs):pytest.fail("Crop preparation must not construct a body, run physics or make network requests")
    monkeypatch.setattr(human, "build_human", forbidden)
    monkeypatch.setattr(mujoco, "mj_step", forbidden)
    monkeypatch.setattr(mujoco, "mj_forward", forbidden)
    monkeypatch.setattr(vision.urllib.request, "urlopen", forbidden)
    return video, model, calls


@pytest.mark.parametrize("mode", ["recovery", "unchanged", "competing", "rejected_crop", "unknown_clock"])
def test_native_proposals_feed_unchanged_recovery_with_full_custody(tmp_path, monkeypatch, mode):
    missing = () if mode in {"unchanged", "competing"} else (30,)
    video, model, calls = native_fixture(tmp_path, monkeypatch, missing=missing,
        competing=(30,) if mode == "competing" else (), crop_competing=mode == "rejected_crop",
        unknown_clock=14 if mode == "unknown_clock" else None)
    before = video.read_bytes(), model.read_bytes()
    target = tmp_path / "proposals"
    assert pipeline.prepare_pose_crop_proposals(video, target, pose_model=model) == target
    package = files(target); report = read(target / "report.json"); manifest = read(target / "manifest.json")
    assert manifest["profile"] == pipeline.PROFILE and manifest["status"] == "source_evidence_only"
    assert manifest["simulation_ready"] is False
    assert set(manifest["files"]) == set(package) - {"manifest.json"}
    assert all(hashlib.sha256(package[k]).hexdigest() == h for k,h in manifest["files"].items())
    assert sum(map(len, package.values())) + len(before[0]) <= pipeline.MAX_BYTES - pipeline.RESERVE_BYTES
    assert set(recovery.INPUT_FILES) <= set(package)
    assert not any(n in package for n in ("human.json", "scene.json", "task.json", "source.mp4"))
    assert calls["baseline"] == 32 and calls["crops"] == len(missing)
    assert calls["detectors"] == calls["closed"] == 1 + bool(missing)
    assert calls["decoders"] == calls["released"] == 1 + bool(missing)
    assert report["work"]["physics_steps"] == report["work"]["api_calls"] == 0
    audit = read(target / "decode-audit.json")
    assert audit["status"] == ("verified" if missing else "not_needed")
    assert audit["sampled_clocks_checked"] == (32 if missing else 0)
    assert audit["unknown_native_timestamps"] == (1 if mode == "unknown_clock" else 0)
    recovered = recovery.prepare_pose_source_recovery(target, tmp_path / "recovered", source_video=video)
    recovered_files = files(recovered)
    verified = recovery.validate_pose_recovery_files(recovered_files, source_sha256=digest(video))
    accepted = [30] if mode in {"recovery", "unknown_clock"} else []
    assert verified["pose_reinference"]["changed_frames"] == accepted
    baseline = read(target / "whole_image/evidence.json")
    assert len(verified["frames"]) == len(baseline["frames"]) == 32
    for old,new in zip(baseline["frames"], verified["frames"], strict=True):
        assert all(new[k] == old[k] for k in ("id", "frame", "time_s", "decoder_timestamp_s"))
        if old["frame"] not in accepted:assert new == old
    assert read(target / "selected/pose-landmarks.json") == read(target / "whole_image/pose-landmarks.json")
    assert (video.read_bytes(), model.read_bytes()) == before
    assert files(target) == package
    if mode == "competing":assert verified["quality"]["passes"] is False


def test_generated_codec_fixture_verifies_second_pass_without_real_model(tmp_path, monkeypatch):
    video, model, calls = native_fixture(tmp_path, monkeypatch, real_video=True)
    result = pipeline.prepare_pose_crop_proposals(video, tmp_path / "proposals", pose_model=model)
    audit = read(result / "decode-audit.json")
    assert audit["complete"] and audit["decoded_frames"] == 63 and audit["sampled_clocks_checked"] == 32
    assert calls["baseline"] == 32 and calls["crops"] == 1
    assert read(result / "report.json")["accepted_frames"] == [30]


@pytest.mark.parametrize("fault", ["metadata", "clock", "index", "short", "extra"])
def test_changed_second_decode_preserves_failure_without_selected_seal(tmp_path, monkeypatch, fault):
    video, model, calls = native_fixture(tmp_path, monkeypatch, second_fault=fault)
    target = tmp_path / "proposals"
    with pytest.raises(ValueError, match="decoder"):
        pipeline.prepare_pose_crop_proposals(video, target, pose_model=model)
    assert read(target / "failure.json")["status"] == read(target / "manifest.json")["status"] == "failed"
    assert (target / "whole_image/evidence.json").is_file()
    assert not (target / "selected/evidence.json").exists()
    assert calls["closed"] == calls["detectors"] and calls["released"] == calls["decoders"]
    with pytest.raises(ValueError, match="Failed proposal"):
        recovery.prepare_pose_source_recovery(target, tmp_path / "recovered", source_video=video)


def test_native_crop_failure_retains_original_and_crop_pixels(tmp_path, monkeypatch):
    video, model, calls = native_fixture(tmp_path, monkeypatch, detector_failure=True)
    target = tmp_path / "proposals"
    with pytest.raises(RuntimeError, match="analytical native"):
        pipeline.prepare_pose_crop_proposals(video, target, pose_model=model)
    assert (target / "crop-inputs/frame-0030.png").is_file()
    assert read(target / "failure.json")["work"]["crop_inference_calls"] == 1
    assert (target / "crop-observations.jsonl").read_bytes() == b""
    assert calls["detectors"] == calls["closed"] == 2


def test_shared_sink_reserves_across_directories_before_encoding(tmp_path):
    maximum = pipeline.RESERVE_BYTES + 1600
    budget = pipeline._Budget(tmp_path, maximum, time.monotonic() + 5)
    first, second = budget.sink("one"), budget.sink("two")
    first.write_text("kept.txt", "x" * 700)
    second.write_text("kept.txt", "y" * 700)
    with pytest.raises(ValueError, match="reserved output capacity"):
        first.write_text("overflow.txt", "z" * 300)
    assert not (tmp_path / "one/overflow.txt").exists()
    assert budget.total == 1400
    first.write_text("kept.txt", "a" * 900)
    assert budget.total == 1600 and (tmp_path / "two/kept.txt").read_text() == "y" * 700


def test_journal_bound_retains_complete_prefix(tmp_path, monkeypatch):
    budget = pipeline._Budget(tmp_path, pipeline.RESERVE_BYTES + 5000, time.monotonic() + 5)
    sink = budget.sink();sink.write_text("crop-observations.jsonl", "")
    sink.append_candidate({"frame": 1})
    prefix = (tmp_path / "crop-observations.jsonl").read_bytes()
    monkeypatch.setattr(pipeline, "MAX_LINE_BYTES", 32)
    with pytest.raises(ValueError, match="reserved output capacity"):
        sink.append_candidate({"frame": 2, "raw": "x" * 50})
    assert (tmp_path / "crop-observations.jsonl").read_bytes() == prefix
    assert budget.total == len(prefix)


@pytest.mark.parametrize("kind", ["deadline", "model", "source", "note", "output", "time"])
def test_invalid_inputs_reject_before_decode_or_output_creation(tmp_path, monkeypatch, kind):
    video, model, calls = native_fixture(tmp_path, monkeypatch)
    target = tmp_path / "proposals";options={}
    if kind == "deadline":options["deadline"] = time.monotonic() - 1
    elif kind == "model":model.write_bytes(b"not the pinned fixture")
    elif kind == "source":video.unlink()
    elif kind == "note":options["source_note"] = "x" * 4097
    elif kind == "time":options["max_seconds"] = 181
    elif kind == "output":target.mkdir()
    with pytest.raises((ValueError, TimeoutError)):
        pipeline.prepare_pose_crop_proposals(video, target, pose_model=model, **options)
    assert calls["decoders"] == calls["detectors"] == 0
    assert not target.exists() or kind == "output" and not list(target.iterdir())


def test_late_source_mutation_cannot_publish_proposals(tmp_path, monkeypatch):
    video, model, _ = native_fixture(tmp_path, monkeypatch)
    def progress(stage, _detail):
        if stage == "selected":video.write_bytes(b"changed original")
    with pytest.raises(ValueError, match="hash|SHA|checksum"):
        pipeline.prepare_pose_crop_proposals(video, tmp_path / "proposals", pose_model=model, progress=progress)
    assert read(tmp_path / "proposals/manifest.json")["status"] == "failed"


def test_more_than_64_eligible_frames_preserves_plans_before_any_crop_call(tmp_path, monkeypatch):
    video, model, calls = native_fixture(tmp_path, monkeypatch, count=270, missing=tuple(range(2, 268, 4)))
    target = tmp_path / "proposals"
    with pytest.raises(ValueError, match="More than 64"):
        pipeline.prepare_pose_crop_proposals(video, target, pose_model=model)
    plan = read(target / "crop-plan.json")
    assert len(plan["plans"]) == 135 and len(plan["eligible_frames"]) == 67 and plan["bound_exceeded"]
    assert calls["baseline"] == 135 and calls["crops"] == 0
    assert calls["decoders"] == calls["detectors"] == 1
    assert (target / "whole_image/pose-landmarks.json").is_file()
    assert read(target / "manifest.json")["status"] == "failed"


def test_partial_baseline_failure_keeps_completed_count_without_inventing_native_call_count(tmp_path, monkeypatch):
    video, model, calls = native_fixture(tmp_path, monkeypatch)
    target = tmp_path / "proposals"
    def progress(stage, detail):
        if stage == "whole_image":raise TimeoutError("caller cancelled before the next source sample")
    with pytest.raises(TimeoutError, match="caller cancelled"):
        pipeline.prepare_pose_crop_proposals(video, target, pose_model=model, progress=progress)
    failure = read(target / "failure.json")
    assert failure["work"]["whole_completed_samples"] == 1
    assert failure["work"]["whole_inference_calls"] is None
    assert calls["detectors"] == calls["closed"] == calls["decoders"] == calls["released"] == 1
    assert read(target / "manifest.json")["status"] == "failed"


def test_source_size_bound_is_checked_before_native_calls(tmp_path, monkeypatch):
    video, model, calls = native_fixture(tmp_path, monkeypatch)
    with video.open("r+b") as stream:stream.truncate(pipeline.MAX_FILE_BYTES + 1)
    with pytest.raises(ValueError, match="64 MiB"):
        pipeline.prepare_pose_crop_proposals(video, tmp_path / "proposals", pose_model=model)
    assert calls["decoders"] == calls["detectors"] == 0 and not (tmp_path / "proposals").exists()


def test_aggregate_encoding_failure_preserves_bounded_prefix_and_failed_seal(tmp_path, monkeypatch):
    video, model, _ = native_fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(pipeline, "MAX_BYTES", 2 * pipeline.RESERVE_BYTES + 600_000)
    target = tmp_path / "proposals"
    with pytest.raises(ValueError, match="capacity|reservation"):
        pipeline.prepare_pose_crop_proposals(video, target, pose_model=model)
    package = files(target)
    assert read(target / "manifest.json")["status"] == "failed"
    assert (target / "implementation.json").is_file()
    assert sum(map(len, package.values())) + video.stat().st_size <= pipeline.MAX_BYTES - pipeline.RESERVE_BYTES
    assert not (target / "selected/evidence.json").exists()


@pytest.mark.parametrize("limit,expected", [(None, 60), (48, 48)])
def test_preview_cap_preserves_every_sample_and_legacy_default_on_container_undercount(tmp_path, monkeypatch, limit, expected):
    video, model, calls = native_fixture(tmp_path, monkeypatch, count=120, metadata_count=20, missing=())
    drawn, retained = [], []
    monkeypatch.setattr(vision, "_draw_pose", lambda _image, row: drawn.append(row["frame"]) or row["frame"])
    monkeypatch.setattr(vision, "_save_previews", lambda previews, *_args, **_kw: retained.append(list(previews)))
    target = tmp_path / "baseline"; target.mkdir()
    options = {} if limit is None else {"max_preview_frames": limit}
    evidence = vision.inspect_exercise(video, target, model, pose_mode="image", **options)
    assert calls["baseline"] == 60 and evidence["decoded_frames"] == 120 and evidence["frame_count"] == 20
    assert [row["frame"] for row in evidence["frames"]] == list(range(0, 120, 2))
    assert all(row["decoder_timestamp_s"] == row["frame"] * 1000 / 30 / 1000 for row in evidence["frames"])
    assert len(drawn) == len(retained[0]) == expected
    assert evidence["container_frame_audit"]["matches"] is True
    if limit is None:
        assert "preview_retention" not in evidence
    else:
        assert evidence["preview_retention"]["retained_source_frames"] == list(range(0, 96, 2))
        assert evidence["preview_retention"]["omitted_scheduled_previews"] == 12


def test_native_pipeline_passes_preview_cap_for_undercounted_source(tmp_path, monkeypatch):
    video, model, calls = native_fixture(tmp_path, monkeypatch, count=120, metadata_count=20, missing=())
    drawn = []
    monkeypatch.setattr(vision, "_draw_pose", lambda _image, row: drawn.append(row["frame"]) or row["frame"])
    monkeypatch.setattr(vision, "_save_previews", lambda *_args, **_kw: None)
    target = pipeline.prepare_pose_crop_proposals(video, tmp_path / "proposals", pose_model=model)
    baseline = read(target / "whole_image/evidence.json")
    assert len(drawn) == 48 and calls["baseline"] == len(baseline["frames"]) == 60
    assert baseline["preview_retention"]["omitted_scheduled_previews"] == 12
    assert read(target / "decode-audit.json")["status"] == "not_needed"


@pytest.mark.parametrize("bad", [0, 49, True, 1.5, "48"])
def test_invalid_preview_bound_rejects_before_native_work(tmp_path, monkeypatch, bad):
    video, model, calls = native_fixture(tmp_path, monkeypatch)
    target = tmp_path / "baseline"; target.mkdir()
    with pytest.raises(ValueError, match="preview retention"):
        vision.inspect_exercise(video, target, model, max_preview_frames=bad)
    assert calls["decoders"] == calls["detectors"] == 0 and not list(target.iterdir())
