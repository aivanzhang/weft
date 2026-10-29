import json
from pathlib import Path

import pytest

from weft.benchmark import extract_metrics, grade_case, load_suite, probe_interaction
from weft.io import digest, write_json
from test_lab_http import make_ready_lab


def manifest(tmp_path):
    video = tmp_path / "video.mp4"
    video.write_bytes(b"pinned test bytes; not a decoded video")
    case = {"id": "ball", "profile": "planar", "video": video.name, "source_sha256": digest(video),
            "source_kind": "analytic_synthetic", "rights": "Test fixture", "conditions": ["fixed-camera"],
            "expected": {"status": "ready"}, "gates": [{"metric": "action_effect", "operator": "min", "value": .001}]}
    path = tmp_path / "suite.json"
    write_json(path, {"schema_version": 1, "name": "test", "purpose": "Test validation", "cases": [case]})
    return path, case


def test_pin_changes_rejected_before_running(tmp_path):
    path, _ = manifest(tmp_path)
    _, cases = load_suite(path)
    assert Path(cases[0]["video"]).is_absolute()
    (tmp_path / "video.mp4").write_bytes(b"changed")
    with pytest.raises(ValueError, match="identity mismatch"):
        load_suite(path)


def test_duplicate_ids_and_unknown_gates_rejected(tmp_path):
    path, case = manifest(tmp_path)
    spec = json.loads(path.read_text())
    spec["cases"].append(case)
    write_json(path, spec)
    with pytest.raises(ValueError, match="unique"):
        load_suite(path)
    spec["cases"] = [case]
    case["gates"][0]["metric"] = "imagined_accuracy"
    write_json(path, spec)
    with pytest.raises(ValueError, match="Unknown metric"):
        load_suite(path)


def test_missing_measurements_and_empty_gates_do_not_prove_accuracy(tmp_path):
    _, case = manifest(tmp_path)
    for value in (None, float("nan"), float("inf")):
        result = grade_case(case, "ready", {"action_effect": value})
        assert not result["passed"] and result["gates"][0]["reason"] == "metric unavailable"
    assert not grade_case({**case, "gates": []}, "ready", {})["passed"]
    result = grade_case(case, "ready", {"action_effect": .1})
    assert result["passed"] and not result["accuracy_claim"]


def test_expected_rejection_does_not_hide_unrelated_crashes(tmp_path):
    _, case = manifest(tmp_path)
    case.update(expected={"status": "rejected", "error_contains": "Moving camera is unsupported"}, gates=[])
    assert grade_case(case, "failed", {}, "Moving camera is unsupported", failure_phase="perception", error_type="UnsupportedVideo")["passed"]
    assert not grade_case(case, "failed", {}, "FileNotFoundError: model missing")["passed"]
    assert not grade_case(case, "ready", {})["passed"]
    assert not grade_case(case, "failed", {}, "Moving camera is unsupported", failure_phase="integrity", error_type="RuntimeError")["passed"]


def test_extract_metrics_separates_coverage_and_reference_servo(tmp_path):
    write_json(tmp_path / "pose-quality.json", {"sampled_frames": 10, "tracked_frames": 8,
                                               "status_counts": {"tracked": 2, "tracked_partial": 6}})
    write_json(tmp_path / "human-evaluation.json", {"reference_servo_rmse_radians": .2})
    write_json(tmp_path / "rl-evaluation.json", {"policies": {
        "q_learning": {"successes": 4, "episodes": 5}, "random": {"successes": 2, "episodes": 5}}})
    metrics = extract_metrics(tmp_path)
    assert metrics["usable_pose_fraction"] == .8
    assert metrics["fully_tracked_pose_fraction"] == .2
    assert metrics["reference_rmse_rad"] == .2
    assert metrics["learned_minus_random"] == .4
    assert "reconstruction_rmse_px" not in metrics


def test_real_interaction_and_recording_preserve_run(tmp_path):
    lab = make_ready_lab(tmp_path / "lab")
    original = digest(lab.root / "run/manifest.json")
    try:
        metrics, evidence = probe_interaction(lab)
        assert metrics["action_effect"] > .001
        assert metrics["recorded_transitions"] == 12
        assert evidence["export_verification"]["steps"] == 12
        assert evidence["export_verification"]["video_frames"] == 13
        assert digest(lab.root / "run/manifest.json") == original
    finally:
        lab.close()


def test_appearance_profile_requires_config_and_records_planar_actions(tmp_path):
    from weft.lab import prepare_lab
    path, case = manifest(tmp_path)
    spec = json.loads(path.read_text())
    spec["cases"][0]["profile"] = "appearance"
    write_json(path, spec)
    with pytest.raises(ValueError, match="require.*config"):
        load_suite(path)
    with pytest.raises(ValueError, match="requires --config"):
        prepare_lab(tmp_path / "video.mp4", tmp_path / "unprepared", profile="appearance")
    assert not (tmp_path / "unprepared").exists()
    write_json(tmp_path / "regions.json", {"tracking": {}, "world": {}})
    spec["cases"][0]["config"] = "regions.json"
    write_json(path, spec)
    assert load_suite(path)[1][0]["profile"] == "appearance"
    lab = make_ready_lab(tmp_path / "lab")
    lab.config["profile"] = "appearance"
    write_json(lab.root / "lab-config.json", lab.config)
    try:
        metrics, evidence = probe_interaction(lab)
        assert metrics["action_effect"] > .001
        assert evidence["export_verification"]["steps"] == 12
        assert evidence["export_verification"]["video_frames"] == 13
    finally:
        lab.close()


def test_study_seal_covers_nested_failure_and_success_evidence(tmp_path):
    from weft.benchmark import seal_suite
    from weft.pipeline import verify_artifacts
    write_json(tmp_path / "report.json", {"counts": {"total": 2}})
    (tmp_path / "case").mkdir()
    write_json(tmp_path / "case/failure.json", {"error": "preserved failure"})
    assert seal_suite(tmp_path)["verified_files"] == 2
    with pytest.raises(ValueError, match="not-yet-sealed"):
        seal_suite(tmp_path)
    (tmp_path / "case/failure.json").write_text("changed")
    with pytest.raises(ValueError, match="integrity"):
        verify_artifacts(tmp_path)
