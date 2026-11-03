from collections import Counter
import json
from pathlib import Path

import pytest

from weft import exercise_build_report as report
from test_exercise_build_report import Links, fixture as legacy_fixture, put, run, seal, sha


COMPETING = "Multiple people or competing pose hypotheses were detected; inspect the source and raw poses"
VISIBILITY = "Insufficient full-body visibility: need at least 8 usable poses and 65% usable sampled frames"
MOTION = "Insufficient measurable exercise motion; a repeated still image is not a dynamic exercise video"
DURATION = "Usable pose sequence must span at least 0.4 seconds"
REJECTION = "Rejected/unsupported pose evidence cannot become a construction body"


def save_evidence(root, evidence):
    for prefix in ("body_source", "recovered_poses", "pose_proposals/selected"):
        put(root, prefix + "/evidence.json", evidence)
    put(root, "pose_proposals/selected/pose-quality.json", evidence["quality"])


def quality_for(frames, errors, *, motion=.8):
    tracked = sum(row["status"] in {"tracked", "tracked_partial"} for row in frames)
    return {"errors": errors, "passes": not errors, "sampled_frames": len(frames),
            "tracked_frames": tracked, "tracked_fraction": tracked / len(frames),
            "status_counts": dict(Counter(row["status"] for row in frames)),
            "minimum_tracked_frames": 8, "minimum_tracked_fraction": .65,
            "motion_threshold_radians": .08, "joint_range_radians": {"left_knee": motion},
            "warnings": ["Saved model visibility is not independently verified person identity."]}


def failed_build(root):
    """Analytical rows reproduce the failed build's schema/counts, not its video or poses."""
    root.mkdir()
    video = b"\x1aE\xdf\xa3analytical diagnostic fixture; never decode"
    digest = sha(video)
    competing = {130, 142, 144, 190, 384}
    remaining = [index for index in range(0, 408, 2) if index not in competing]
    missing = dict(zip(remaining[:34], ["low_confidence"] * 25 + ["invalid_geometry"] * 2 + ["no_pose"] * 7))
    usable = [index for index in remaining if index not in missing]
    partial = set(usable[:152])
    frames = [{"frame": index, "id": f"{digest}:frame:{index}", "time_s": index / 30,
               "decoder_timestamp_s": round(index / 30, 3),
               "status": "multiple_people" if index in competing else missing.get(index, "tracked_partial" if index in partial else "tracked")}
              for index in range(0, 408, 2)]
    evidence = {"schema_version": 1, "profile": "exercise-pose-v1", "source_sha256": digest,
                "frame_count": 410, "decoded_frames": 408, "fps": 30, "frames": frames,
                "quality": quality_for(frames, [COMPETING])}
    stages = [{"stage": name, "status": status, "elapsed_seconds": elapsed} for name, status, elapsed in (
        ("source_pose_proposals", "ready", 1.), ("source_pose_recovery", "ready", 2.),
        ("source_body_source", "running", 3.))]
    put(root, "result.json", {"schema_version": 1, "status": "failed", "seal_scope": "complete_immutable_outputs",
        "lab_status": "not_ready", "lab_directory": None, "stages": stages,
        "error": {"type": "ValueError", "message": REJECTION}})
    put(root, "settings.json", {"video": "/not-followed/original.webm", "pose_recovery": "selective-crop"})
    put(root, "stages.json", {"stages": stages})
    put(root, "body_source/source.webm", video)
    put(root, "body_source/source.json", {"artifact": "source.webm", "sha256": digest, "title": "Analytical source"})
    put(root, "body_source/failure.json", {"schema_version": 1, "profile": "exercise-body-source-v1", "status": "failed",
        "simulation_ready": False, "error": {"type": "ValueError", "message": REJECTION}})
    save_evidence(root, evidence)
    seal(root)
    return root, evidence


def test_failed_schema_explains_competing_samples_not_insufficient_coverage(tmp_path):
    source, _ = failed_build(tmp_path / "source")
    before = {p.relative_to(source).as_posix(): sha(p.read_bytes()) for p in source.rglob("*") if p.is_file()}
    result, summary = run(source, tmp_path / "report")
    diagnostic = summary["diagnostics"]
    assert diagnostic["status"] == "saved_quality_rejected" and diagnostic["explains_body_stop"] is True
    checks = {row["id"]: row for row in diagnostic["checks"]}
    competing, coverage = checks["competing_poses"], checks["visibility"]
    assert competing["status"] == "failed" and competing["observed"]["samples"] == 5
    assert competing["observed"]["bounded_transient_review"]["resolved"] is False
    assert competing["required"] == {"maximum_unresolved_samples": 0}
    assert coverage["status"] == "passed" and coverage["observed"] == {
        "usable_samples": 165, "sampled_frames": 204, "usable_fraction": 165 / 204}
    assert coverage["required"] == {"minimum_samples": 8, "minimum_fraction": .65}
    affected = competing["affected_samples"]
    assert affected["original_frames"] == [130, 142, 144, 190, 384]
    assert [row["sampled_frames"] for row in affected["sample_runs"]] == [[130], [142, 144], [190], [384]]
    assert affected["sample_runs"][1]["nominal_start_s"] == 142 / 30
    assert affected["sample_runs"][1]["native_start_s"] == 4.733
    assert summary["coverage"]["unsampled_frames"] == 204
    states = {row["stage"]: row["state"] for row in summary["stages"]}
    assert states["source_body_source"] == "stopped" and states["shared_world"] == "not_run"
    assert result["build_status"] == "failed" and result["saved_lab_status"] == "not_ready"
    assert all(value == 0 for value in summary["work"].values())
    copied = {row["source_name"]: row for row in summary["artifacts"]}
    for name in ("body_source/failure.json", "pose_proposals/selected/pose-quality.json"):
        assert (tmp_path / "report" / copied[name]["path"]).read_bytes() == (source / name).read_bytes()
    assert before == {p.relative_to(source).as_posix(): sha(p.read_bytes()) for p in source.rglob("*") if p.is_file()}
    html = (tmp_path / "report/index.html").read_text()
    assert "Why body construction stopped" in html and "5 sampled frames" in html
    assert "This does not prove that multiple people are present" in html
    assert "165 of 204 samples (80.9%)" in html and "142–144" in html
    assert "Do not lower acceptance thresholds, discard failing frames" in html
    assert "No interactive lab was produced" in html
    parsed = Links()
    parsed.feed(html)
    assert "script" not in parsed.tags and "iframe" not in parsed.tags
    assert all(link.startswith("#") or (not Path(link).is_absolute() and ":" not in link
               and (tmp_path / "report" / link).is_file()) for link in parsed.links)


@pytest.mark.parametrize("native", [None, 4.9])
def test_missing_or_reversed_native_clock_is_never_filled(tmp_path, native):
    source, evidence = failed_build(tmp_path / "source")
    next(row for row in evidence["frames"] if row["frame"] == 142)["decoder_timestamp_s"] = native
    save_evidence(source, evidence)
    seal(source)
    _, summary = run(source, tmp_path / "report")
    span = summary["diagnostics"]["checks"][0]["affected_samples"]["sample_runs"][1]
    assert span["native_start_s"] is None and span["native_end_s"] is None
    assert span["native_status"] == "unavailable_or_nonmonotonic" and span["nominal_start_s"] == 142 / 30


def test_no_pose_samples_report_visibility_and_motion_without_inventing_duration(tmp_path):
    source, evidence = failed_build(tmp_path / "source")
    for row in evidence["frames"]:
        row["status"] = "no_pose"
    evidence["quality"] = quality_for(evidence["frames"], [VISIBILITY, MOTION], motion=0.)
    save_evidence(source, evidence)
    seal(source)
    _, summary = run(source, tmp_path / "report")
    diagnostic = summary["diagnostics"]
    failed = {row["id"]: row for row in diagnostic["checks"] if row["status"] == "failed"}
    assert set(failed) == {"visibility", "motion"}
    assert failed["visibility"]["observed"]["usable_samples"] == 0
    assert failed["visibility"]["affected_samples"]["sample_count"] == 204
    assert failed["motion"]["observed"]["maximum_joint_range_radians"] == 0
    assert not any(row["id"] == "duration" for row in diagnostic["checks"])


def test_duration_failure_uses_original_nominal_endpoint_span(tmp_path):
    source, evidence = failed_build(tmp_path / "source")
    evidence["frames"] = evidence["frames"][:5]
    for row in evidence["frames"]:
        row["status"] = "tracked"
        row["decoder_timestamp_s"] = None
    evidence["quality"] = quality_for(evidence["frames"], [VISIBILITY, DURATION])
    save_evidence(source, evidence)
    seal(source)
    _, summary = run(source, tmp_path / "report")
    duration = next(row for row in summary["diagnostics"]["checks"] if row["id"] == "duration")
    assert duration["observed"] == {"nominal_seconds": 8 / 30, "first_frame": 0, "last_frame": 8}
    assert duration["required"] == {"minimum_seconds": .4}


@pytest.mark.parametrize("mutation,match", [
    ("count", "counts differ"), ("count_type", "counts differ"), ("status_count", "counts differ"),
    ("pass_flag", "pass flag"), ("wrong_error", "contradicts"), ("empty_errors", "pass flag"),
    ("sidecar", "sidecar differs"), ("failure", "terminal reason disagree"),
    ("failure_shape", "failure receipt"), ("ready", "ready lab"), ("number", "diagnostic value"),
])
def test_conflicting_or_malformed_saved_diagnostics_fail_before_output(tmp_path, mutation, match):
    source, evidence = failed_build(tmp_path / "source")
    quality = evidence["quality"]
    if mutation == "count": quality["tracked_frames"] -= 1
    if mutation == "count_type": quality["sampled_frames"] = 204.
    if mutation == "status_count": quality["status_counts"]["multiple_people"] = True
    if mutation == "pass_flag": quality["passes"] = True
    if mutation == "wrong_error": quality["errors"] = [VISIBILITY]
    if mutation == "empty_errors": quality["errors"] = []
    if mutation == "number": quality["motion_threshold_radians"] = True
    save_evidence(source, evidence)
    if mutation == "sidecar":
        put(source, "pose_proposals/selected/pose-quality.json", {**quality, "tracked_frames": 1})
    if mutation in {"failure", "failure_shape"}:
        doc = json.loads((source / "body_source/failure.json").read_bytes())
        if mutation == "failure": doc["error"]["message"] = "Another error"
        else: doc["simulation_ready"] = True
        put(source, "body_source/failure.json", doc)
    if mutation == "ready":
        doc = json.loads((source / "result.json").read_bytes())
        doc["lab_status"] = "ready"
        put(source, "result.json", doc)
    seal(source)
    with pytest.raises(ValueError, match=match):
        run(source, tmp_path / "report")
    assert not (tmp_path / "report").exists()


@pytest.mark.parametrize("name", ["body_source/evidence.json", "body_source/failure.json", "pose_proposals/selected/pose-quality.json"])
def test_tampered_diagnostic_member_is_not_used(tmp_path, name):
    source, _ = failed_build(tmp_path / "source")
    with (source / name).open("ab") as stream:
        stream.write(b" ")
    with pytest.raises(ValueError, match="hash differs"):
        run(source, tmp_path / "report")
    assert not (tmp_path / "report").exists()


def test_different_pose_selection_cannot_supply_quality_numbers(tmp_path):
    source, evidence = failed_build(tmp_path / "source")
    alternate = {**evidence, "scope": "different selection"}
    put(source, "pose_proposals/selected/evidence.json", alternate)
    put(source, "pose_proposals/selected/pose-quality.json", {"passes": True, "errors": []})
    seal(source)
    _, summary = run(source, tmp_path / "report")
    assert "pose_proposals/selected/pose-quality.json" not in {r["source_name"] for r in summary["artifacts"]}
    assert summary["diagnostics"]["checks"][0]["observed"]["samples"] == 5


def test_unknown_saved_gate_stays_verbatim_without_made_up_measurements(tmp_path):
    source, evidence = failed_build(tmp_path / "source")
    evidence["quality"]["errors"].append('<unknown gate src="https://invalid.example">')
    save_evidence(source, evidence)
    seal(source)
    _, summary = run(source, tmp_path / "report")
    assert summary["diagnostics"]["unquantified_saved_errors"] == ['<unknown gate src="https://invalid.example">']
    html = (tmp_path / "report/index.html").read_text()
    assert "&lt;unknown gate" in html and "no supported numeric explanation" in html


def test_unrelated_terminal_failure_is_not_relabelled_a_pose_failure(tmp_path):
    source, _ = failed_build(tmp_path / "source")
    result = json.loads((source / "result.json").read_bytes())
    result["error"] = {"type": "TimeoutError", "message": "Copy deadline expired"}
    put(source, "result.json", result)
    failure = json.loads((source / "body_source/failure.json").read_bytes())
    failure["error"] = result["error"]
    put(source, "body_source/failure.json", failure)
    seal(source)
    _, summary = run(source, tmp_path / "report")
    assert summary["diagnostics"]["explains_body_stop"] is False
    html = (tmp_path / "report/index.html").read_text()
    assert "Copy deadline expired" in html and "Saved pose-quality failures" in html
    assert "Why body construction stopped" not in html


@pytest.mark.parametrize("ready,receipt_only", [(True, False), (False, True)])
def test_legacy_ready_and_receipt_only_remain_honest(tmp_path, ready, receipt_only):
    source = legacy_fixture(tmp_path / "source", ready=ready, receipt_only=receipt_only)
    _, summary = run(source, tmp_path / "report")
    assert summary["diagnostics"]["status"] == ("not_available" if receipt_only else "no_recorded_pose_quality_failure")
    assert summary["saved_lab_status"] == ("ready" if ready else "not_ready")
    assert summary["task_success"] == "not_assessed_by_report"
    if receipt_only:
        assert [row["source_name"] for row in summary["artifacts"]] == ["result.json"]


def test_absent_pose_evidence_does_not_invent_a_cause(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    put(source, "result.json", {"schema_version": 1, "seal_scope": "complete_immutable_outputs", "status": "failed",
                               "lab_status": "not_ready", "stages": [], "error": {"type": "OSError", "message": "Missing input"}})
    seal(source)
    _, summary = run(source, tmp_path / "report")
    assert summary["diagnostics"]["status"] == "not_available" and summary["diagnostics"]["checks"] == []


def test_legacy_evidence_without_quality_keeps_coverage_only(tmp_path):
    source = legacy_fixture(tmp_path / "source", ready=True)
    evidence = json.loads((source / "body_source/evidence.json").read_bytes())
    del evidence["quality"]
    put(source, "body_source/evidence.json", evidence)
    seal(source)
    _, summary = run(source, tmp_path / "report")
    assert summary["coverage"]["usable_samples"] == 29
    assert summary["diagnostics"]["status"] == "no_recorded_pose_quality_failure"
    assert summary["diagnostics"]["checks"] == []


def test_quality_sidecar_mutation_before_seal_cannot_publish_success(tmp_path, monkeypatch):
    source, _ = failed_build(tmp_path / "source")
    original = report._render

    def mutate(*args):
        raw = original(*args)
        with (source / "pose_proposals/selected/pose-quality.json").open("ab") as stream:
            stream.write(b" ")
        return raw

    monkeypatch.setattr(report, "_render", mutate)
    with pytest.raises(ValueError, match="hash differs"):
        run(source, tmp_path / "report")
    assert not (tmp_path / "report/manifest.json").exists()
    assert (tmp_path / "report/failure.json").is_file()


def test_diagnostic_run_and_byte_bounds_keep_exact_frame_list(tmp_path, monkeypatch):
    source, evidence = failed_build(tmp_path / "source")
    monkeypatch.setattr(report, "MAX_DIAGNOSTIC_SPANS", 2)
    _, summary = run(source, tmp_path / "report")
    affected = summary["diagnostics"]["checks"][0]["affected_samples"]
    assert len(affected["sample_runs"]) == 2 and affected["unlisted_sample_runs"] == 2
    assert affected["original_frames"] == [130, 142, 144, 190, 384]
    monkeypatch.setattr(report, "MAX_DIAGNOSTIC_BYTES", 100)
    with pytest.raises(ValueError, match="diagnostics exceed"):
        run(source, tmp_path / "too-small")
    assert not (tmp_path / "too-small").exists()
