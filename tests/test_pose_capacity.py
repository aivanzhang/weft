"""Explicit detector capacity must survive source reuse without weakening admission."""

import json
import sys
from types import SimpleNamespace

import pytest

from test_exercise_body_source import (
    VIDEO, forbid_runtime, model_metadata, seal, write_pose_files,
)
from test_exercise_vision import inference_result, stub_detector, video
from weft import exercise_body_source as source, exercise_vision as vision
from weft.io import digest, write_json


@pytest.mark.parametrize("capacity", [0, 3, True, False, 1.0, "1", None])
def test_invalid_capacity_rejects_before_any_input_access(tmp_path, capacity):
    with pytest.raises(ValueError, match="Maximum pose hypotheses"):
        vision.inspect_exercise("missing", tmp_path, "missing", max_pose_hypotheses=capacity)
    with pytest.raises(ValueError, match="Maximum pose hypotheses"):
        source.prepare_exercise_body_source("missing", tmp_path / "out", pose_model="missing",
                                             max_pose_hypotheses=capacity)
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("capacity,mode", [(1, "video"), (1, "image"), (2, "video")])
def test_capacity_reaches_native_options_without_changing_thresholds(monkeypatch, capacity, mode):
    captured = []

    class BaseOptions(SimpleNamespace):
        Delegate = SimpleNamespace(CPU="CPU")

    native = SimpleNamespace(
        PoseLandmarkerOptions=lambda **options: options,
        RunningMode=SimpleNamespace(VIDEO="VIDEO", IMAGE="IMAGE"),
        PoseLandmarker=SimpleNamespace(create_from_options=lambda options: captured.append(options) or "detector"))
    mp = SimpleNamespace(tasks=SimpleNamespace(vision=native, BaseOptions=BaseOptions))
    monkeypatch.setitem(sys.modules, "mediapipe", mp)
    assert vision._detector("pose.task", mode, max_pose_hypotheses=capacity) == (mp, "detector")
    options = captured[0]
    assert options["num_poses"] == capacity and options["running_mode"] == mode.upper()
    assert options["min_pose_detection_confidence"] == options["min_pose_presence_confidence"] == .5
    assert options["min_tracking_confidence"] == .5
    assert options["output_segmentation_masks"] is False and options["base_options"].delegate == "CPU"


def test_explicit_capacity_is_truthful_in_raw_and_derived_evidence(tmp_path, monkeypatch):
    original = tmp_path / "source.mp4"
    video(original)
    detector = stub_detector(monkeypatch, lambda index: inference_result(index / 20))
    old_factory = vision._detector
    calls = []

    def factory(path, mode, **options):
        calls.append(options)
        return old_factory(path, mode)

    monkeypatch.setattr(vision, "_detector", factory)
    evidence = vision.inspect_exercise(original, tmp_path, "model", max_pose_hypotheses=1)
    assert calls == [{"max_pose_hypotheses": 1}]
    assert evidence["model"]["maximum_pose_hypotheses"] == 1
    assert evidence["model"]["person_count_assumption"] == "single_person_unverified"
    assert "can miss competing people" in " ".join(evidence["assumptions"])
    assert len(evidence["frames"]) == 15 and detector.closed
    assert json.loads((tmp_path / "pose-landmarks.json").read_text())["model"] == evidence["model"]
    assert all(row["identity_status"] == "single_model_hypothesis_not_verified_person_count"
               for row in evidence["frames"])


def single_pose_run(root, *, dropped=(0, 2, 30), allow_rejection=False):
    root.mkdir()
    (root / "source.webm").write_bytes(VIDEO)
    write_json(root / "source.json", {"artifact": "source.webm", "sha256": digest(root / "source.webm")})
    write_json(root / "run-config.json", {"profile": "exercise", "pose_mode": "video",
                                          "pose_identity": "raw", "max_pose_hypotheses": 1})
    model = model_metadata() | {"maximum_pose_hypotheses": 1, "person_count_assumption": "single_person_unverified"}
    try:
        write_pose_files(root, model=model, dropped=dropped)
    except vision.UnsupportedVideo:
        if not allow_rejection:
            raise
    write_json(root / "model.json", model)
    seal(root)
    return root


def test_saved_single_person_source_keeps_option_and_complete_evidence(tmp_path, monkeypatch):
    run = single_pose_run(tmp_path / "poses")
    forbid_runtime(monkeypatch)
    original = {p.name: p.read_bytes() for p in run.iterdir()}
    out = source.prepare_exercise_body_source_from_evidence(run, tmp_path / "body")
    assert all((run / name).read_bytes() == raw for name, raw in original.items())
    for name in source.POSE_FILES:
        assert (out / name).read_bytes() == original[name]
    files = {p.relative_to(out).as_posix(): p.read_bytes() for p in out.rglob("*") if p.is_file()}
    body, observations = source.validate_body_source_files(files)
    marker = json.loads(files["body-source.json"])
    assert marker["pose_options"]["max_pose_hypotheses"] == 1
    assert body["profile"] == "exercise-v3" and len(observations["observations"]) == 63
    assert marker["work"]["inference_calls"] == marker["work"]["physics_steps"] == 0


@pytest.mark.parametrize("damage", ["config_omission", "model_omission", "false_model_count", "unverified_assumption"])
def test_single_person_cannot_be_relabelled_as_default_or_verified(tmp_path, damage):
    run = single_pose_run(tmp_path / "poses")
    files = {name: (run / name).read_bytes() for name in source.POSE_FILES}
    if damage == "config_omission":
        config = json.loads(files["run-config.json"])
        config.pop("max_pose_hypotheses")
        files["run-config.json"] = json.dumps(config).encode()
    else:
        model = json.loads(files["model.json"])
        if damage == "model_omission":
            model.pop("person_count_assumption")
        elif damage == "false_model_count":
            model["maximum_pose_hypotheses"] = True
        else:
            model["person_count_assumption"] = "verified_single_person"
        files["model.json"] = json.dumps(model).encode()
    with pytest.raises(ValueError):
        source._validated_pose(files, source._options("video", "raw", 1), digest(run / "source.webm"))


def test_single_person_does_not_bypass_quality_failure(tmp_path, monkeypatch):
    run = single_pose_run(tmp_path / "poses", dropped=tuple(range(0, 63, 2)), allow_rejection=True)
    monkeypatch.setattr(source, "build_human", lambda *args, **kwargs: pytest.fail("Quality must reject before construction"))
    with pytest.raises(ValueError, match="Rejected/unsupported"):
        source.prepare_exercise_body_source_from_evidence(run, tmp_path / "body")
    assert json.loads((tmp_path / "body" / "failure.json").read_text())["status"] == "failed"


def test_default_model_cannot_claim_a_single_person_assumption():
    model = model_metadata() | {"person_count_assumption": "single_person_unverified"}
    with pytest.raises(ValueError, match="conflicts"):
        source._model(model, source._options("video", "raw"))


def test_default_options_remain_byte_compatible():
    assert source._options("video", "raw", 2) == {"pose_mode": "video", "pose_identity": "raw"}


def test_raw_body_preparation_persists_single_person_options(tmp_path, monkeypatch):
    original, model_path = tmp_path / "source.webm", tmp_path / "model.task"
    original.write_bytes(VIDEO)
    model_path.write_bytes(b"analytical model bytes")
    entry = {"sha256": digest(model_path), "bytes": model_path.stat().st_size}
    monkeypatch.setattr(vision, "MODEL_VARIANTS", {"lite": entry})
    monkeypatch.setattr(vision, "model_provenance", lambda _: entry)
    calls = []

    def inspect(path, out, model, progress, artifact_sink, **options):
        calls.append(options)
        metadata = model_metadata() | {"maximum_pose_hypotheses": 1,
                                       "person_count_assumption": "single_person_unverified"}
        return write_pose_files(out, model=metadata)

    monkeypatch.setattr(vision, "inspect_exercise", inspect)
    out = source.prepare_exercise_body_source(original, tmp_path / "body", pose_model=model_path,
                                              max_pose_hypotheses=1)
    expected = {"pose_mode": "video", "pose_identity": "raw", "max_pose_hypotheses": 1}
    assert calls == [expected]
    assert json.loads((out / "body-source.json").read_text())["pose_options"] == expected
    assert json.loads((out / "run-config.json").read_text())["max_pose_hypotheses"] == 1
