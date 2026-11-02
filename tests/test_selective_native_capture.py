"""Native timing after crop selection preserves original source and derivation bytes."""

from copy import deepcopy
import json

import mujoco
import pytest

from test_pose_crop_pipeline import native_fixture, files, read
from test_exercise_scene_native_recovery import full_inputs, preflight, outputs, stages
from weft import exercise_body_source as body, human, pose_crop_pipeline as crop
from weft import pose_source_recovery as recovery, source_clock_capture as capture
from weft import exercise_scene_inputs as inputs


def test_generated_crop_source_captures_native_clock_without_repeating_inference(tmp_path, monkeypatch):
    original_forward, original_build = mujoco.mj_forward, human.build_human
    video, model, calls = native_fixture(tmp_path, monkeypatch, real_video=True)
    rows = []
    proposals = crop.prepare_pose_crop_proposals(video, tmp_path / "proposals", pose_model=model,
                                                decoder_clock_sink=rows.append)
    assert len(rows) == 63  # Only the baseline pass contributes, never the crop re-decode.
    assert [row["frame"] for row in rows] == list(range(63))
    recovered = recovery.prepare_pose_source_recovery(proposals, tmp_path / "recovered", source_video=video)
    original = files(recovered)
    assert read(proposals / "report.json")["accepted_frames"] == [30]
    inference_before = deepcopy(calls)
    monkeypatch.setattr(mujoco, "mj_forward", original_forward)
    monkeypatch.setattr(human, "build_human", original_build)
    monkeypatch.setattr(body.exercise_vision, "inspect_exercise", lambda *a, **k: pytest.fail("Repeated inference"))
    out = body.prepare_exercise_body_source_from_evidence(recovered, tmp_path / "body", native_timing=True,
                                                        decoder_clock_rows=rows)
    assert calls == inference_before
    assert files(recovered) == original
    for name in recovery.POSE_FILES | recovery.PROOF_FILES:
        assert (out / name).read_bytes() == original[name]
    marker = read(out / "body-source.json")
    assert marker["work"]["video_decodes"] == 1
    assert marker["work"]["inference_calls"] == marker["work"]["physics_steps"] == 0
    assert body.CAPTURE_FILES <= marker["artifacts"].keys()
    assert read(out / "decoder-clock.json")["rows"] == rows
    assert read(out / "native-clock-capture.json")["evidence_sha256"] == body._sha(original["evidence.json"])
    body.validate_body_source_files(files(out))
    reused = body.prepare_exercise_body_source_from_evidence(recovered, tmp_path / "reused",
                                                            native_clock=out / "native-clock.json")
    assert read(reused / "body-source.json")["work"]["video_decodes"] == 0
    assert (reused / "evidence.json").read_bytes() == original["evidence.json"]


@pytest.mark.parametrize("case", ["missing", "disabled", "conflicting", "bool", "oversized", "unordered", "nonfinite"])
def test_invalid_capture_request_stops_before_output(tmp_path, case):
    rows = [{"frame": i, "decoder_timestamp_s": i / 30} for i in range(2)]
    options = {"native_timing": True, "decoder_clock_rows": rows}
    if case == "missing": options.pop("decoder_clock_rows")
    if case == "disabled": options["native_timing"] = False
    if case == "conflicting": options["native_clock"] = tmp_path / "clock.json"
    if case == "bool": options["native_timing"] = 1
    if case == "oversized": options["decoder_clock_rows"] = rows * 901
    if case == "unordered": rows[0]["frame"] = 1
    if case == "nonfinite": rows[0]["decoder_timestamp_s"] = float("nan")
    with pytest.raises(ValueError):
        body.prepare_exercise_body_source_from_evidence(tmp_path / "missing", tmp_path / "out", **options)
    assert not (tmp_path / "out").exists()


def test_failed_quality_keeps_crop_proof_without_starting_native_capture(tmp_path, monkeypatch):
    video, model, _ = native_fixture(tmp_path, monkeypatch, missing=(), competing=(30,))
    rows = []
    proposals = crop.prepare_pose_crop_proposals(video, tmp_path / "proposals", pose_model=model,
                                                decoder_clock_sink=rows.append)
    recovered = recovery.prepare_pose_source_recovery(proposals, tmp_path / "recovered", source_video=video)
    original = files(recovered)
    monkeypatch.setattr(capture, "collect_native_clock", lambda *a, **k: pytest.fail("Rejected poses started capture"))
    out = tmp_path / "body"
    with pytest.raises(ValueError, match="Rejected/unsupported"):
        body.prepare_exercise_body_source_from_evidence(recovered, out, native_timing=True, decoder_clock_rows=rows)
    assert files(recovered) == original and (out / "evidence.json").read_bytes() == original["evidence.json"]
    assert not (out / "human.json").exists()
    assert read(out / "failure.json")["status"] == "failed"


def test_composition_forwards_baseline_rows_only_to_verified_body_stage(tmp_path, monkeypatch):
    paths, _ = full_inputs(tmp_path, monkeypatch)
    ready = preflight(paths, pose_recovery="selective-crop")
    calls = stages(monkeypatch, ready)
    original = crop.prepare_pose_crop_proposals
    rows = [{"frame": i, "decoder_timestamp_s": i / 30} for i in range(3)]

    def proposals(*args, **options):
        for row in rows:
            options["decoder_clock_sink"](row)
        rows[0]["frame"] = 99
        return original(*args, **options)

    monkeypatch.setattr(crop, "prepare_pose_crop_proposals", proposals)
    inputs.prepare_exercise_scene_body(ready, **outputs(tmp_path), max_seconds=240, native_timing=True)
    assert [row[0] for row in calls] == ["proposals", "recovery", "body"]
    assert "decoder_clock_sink" not in calls[1][3]
    assert calls[2][3]["native_timing"] is True
    assert [row["frame"] for row in calls[2][3]["decoder_clock_rows"]] == [0, 1, 2]
