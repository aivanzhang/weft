import hashlib
import json
import os
from pathlib import Path
import time

import pytest

from weft import exercise_build_report as report
from weft import pose_pixel_support_package as package
from test_exercise_build_report import fixture as build_fixture, seal as seal_build, Links
from test_pose_pixel_support_package import inputs, put


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def seal_attached_build(build):
    seal_build(build)
    manifest = json.loads((build / "manifest.json").read_bytes())
    manifest["files"]["body_source/manifest.json"] = sha((build / "body_source/manifest.json").read_bytes())
    put(build, "manifest.json", manifest)


@pytest.fixture
def ready_inputs(tmp_path):
    build = build_fixture(tmp_path / "build")
    pose, pixels = inputs(tmp_path / "source")
    for path in pose.iterdir():
        put(build / "body_source", path.name, path.read_bytes())
    seal_attached_build(build)
    companion = tmp_path / "companion"
    package.prepare_pose_pixel_support(pose, pixels, companion)
    return build, pose, companion


def files(root):
    return {p.relative_to(root).as_posix(): p.read_bytes() for p in root.rglob("*") if p.is_file()}


def test_optional_companion_renders_clear_conditional_checks_without_rerunning(ready_inputs, tmp_path, monkeypatch):
    build, pose, companion = ready_inputs
    before_build, before_companion = files(build), files(companion)
    monkeypatch.setattr(package.diagnostic, "diagnose_pose_pixel_support", lambda *a, **k: pytest.fail("No patch check rerun"))
    monkeypatch.setattr(package.diagnostic.Image, "open", lambda *a, **k: pytest.fail("No image decode in report"))
    out = tmp_path / "report"
    result = report.prepare_exercise_build_report(build, out, pixel_support=companion)
    summary = json.loads((out / "report.json.txt").read_bytes())
    assert result["build_status"] == "rejected_initial_geometry" and result["saved_lab_status"] == "not_ready"
    assert summary["pixel_support"]["status_counts"] == {"not_contradicted": 36}
    assert all(value == 0 for value in summary["work"].values())
    assert summary["diagnostics"]["status"] != "saved_quality_rejected"
    assert len(summary["pixel_support"]["rows"]) == 36
    assert files(build) == before_build and files(companion) == before_companion
    html = (out / "index.html").read_text()
    for expected in ("Compare estimated motion with images", "small image areas around predicted joints",
                     "No disagreement found", "does not prove that a pose is wrong", "Other frames remain unqualified",
                     'href="#pixel-frame-0"', 'id="pixel-frame-0"', "Native times (s)"):
        assert expected in html
    assert "saved_quality_rejected" not in summary["pixel_support"]
    for row in summary["artifacts"]:
        if row["source_name"].startswith("pixel_support/"):
            assert (out / row["path"]).read_bytes() == before_companion[row["source_name"][len("pixel_support/"):]]
        assert not row["source_name"].endswith(".py")
    parsed = Links();parsed.feed(html)
    assert "script" not in parsed.tags and "iframe" not in parsed.tags
    assert all(link.startswith("#") or (out / link).is_file() for link in parsed.links)
    manifest = json.loads((out / "manifest.json").read_bytes())
    assert all(sha((out / name).read_bytes()) == digest for name, digest in manifest["files"].items())


def test_absent_optional_argument_preserves_exact_default_report_output(tmp_path, monkeypatch):
    build = build_fixture(tmp_path / "build")
    monkeypatch.setattr(report, "local_time", lambda: "2026-10-30T12:00:00-07:00")
    first, second = tmp_path / "default", tmp_path / "none"
    report.prepare_exercise_build_report(build, first)
    report.prepare_exercise_build_report(build, second, pixel_support=None)
    assert files(first) == files(second)
    summary = json.loads((first / "report.json.txt").read_bytes())
    assert "pixel_support" not in summary and "copied_companion_files" not in summary
    assert "Compare estimated motion with images" not in (first / "index.html").read_text()


@pytest.mark.parametrize("mutation", ["evidence", "parent"])
def test_companion_cannot_attach_to_other_resealed_build_evidence(ready_inputs, tmp_path, mutation):
    build, pose, companion = ready_inputs
    target = build / "body_source" / ("evidence.json" if mutation == "evidence" else "manifest.json")
    target.write_bytes(target.read_bytes() + b" ")
    seal_attached_build(build)
    out = tmp_path / "report"
    with pytest.raises(ValueError, match="different original"):
        report.prepare_exercise_build_report(build, out, pixel_support=companion)
    assert not out.exists()


def test_combined_copy_reservation_is_checked_before_output(ready_inputs, tmp_path, monkeypatch):
    build, pose, companion = ready_inputs
    source = report._Source(build.absolute(), time.monotonic() + 5)
    try:
        chosen, _, _ = report._selection(source)
    finally:
        os.close(source.fd)
    monkeypatch.setattr(report, "MAX_COPIED_FILES", len(chosen))
    out = tmp_path / "report"
    with pytest.raises(ValueError, match="combined file bound"):
        report.prepare_exercise_build_report(build, out, pixel_support=companion)
    assert not out.exists()


def test_late_companion_failure_marker_stops_report_sealing(ready_inputs, tmp_path, monkeypatch):
    build, pose, companion = ready_inputs
    actual = report._render
    def mutate(*args):
        result = actual(*args)
        put(companion, "failure.json", {"status": "failed"})
        return result
    monkeypatch.setattr(report, "_render", mutate)
    out = tmp_path / "report"
    with pytest.raises(ValueError, match="failure marker"):
        report.prepare_exercise_build_report(build, out, pixel_support=companion)
    assert (out / "failure.json").exists() and not (out / "manifest.json").exists()


def test_partial_result_is_visibly_partial_without_reclassifying_build(ready_inputs, tmp_path):
    build, pose, companion = ready_inputs
    result = json.loads((companion / "report.json").read_bytes())
    result.update(status="partial_deadline", processed_slots=2,
                  status_counts={"not_contradicted": 2, "unavailable": 34})
    for row in result["rows"][2:]:
        row.update(status="unavailable", reason="deadline_unprocessed")
        for key in ("forward", "backward", "roundtrip_error_px", "model_displacement_px",
                    "patch_displacement_px", "model_patch_residual_px"):
            row.pop(key, None)
    put(companion, "report.json", result)
    manifest = json.loads((companion / "manifest.json").read_bytes())
    manifest["status"] = "partial_deadline"
    manifest["files"]["report.json"] = sha((companion / "report.json").read_bytes())
    put(companion, "manifest.json", manifest)
    out = tmp_path / "report"
    receipt = report.prepare_exercise_build_report(build, out, pixel_support=companion)
    assert receipt["build_status"] == "rejected_initial_geometry"
    html = (out / "index.html").read_text()
    assert "Partial check" in html and "2/36 joint checks" in html and "Cannot check" in html
