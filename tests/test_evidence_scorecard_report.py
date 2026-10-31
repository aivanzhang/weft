import copy
import hashlib
import json
from pathlib import Path

import pytest

from weft import evidence_scorecard, exercise_build_report
from test_exercise_build_report import Links, feasibility_fixture, fixture, put, run, seal


DIMENSIONS = {
    "execution_actions", "perception_identity", "geometry_camera", "temporal_dynamics",
    "physical_relationships", "hidden_state_continuity", "parameter_validity", "causal_response",
    "long_run_stability", "robustness_coverage", "uncertainty_abstention", "task_learning_utility",
    "reproducibility_integrity", "practical_efficiency",
}


def rows(summary):
    return {row["id"]: row for row in summary["evidence_scorecard"]["rows"]}


def metrics(row):
    return {metric["id"]: metric for metric in row["metrics"]}


def learning_fixture(root):
    fixture(root, ready=True)
    identities = {"scene_sha256": "a" * 64, "task_sha256": "b" * 64}
    result = json.loads((root / "result.json").read_bytes())
    put(root, "result.json", {**result, **identities})
    put(root, "learning_workflow/result.json", {"status": "ready", **identities,
        "training": {"evaluated_episodes": 8, "control_steps": 1872},
        "selection": {"selection_status": "retained_original_no_feasible_candidate", "evaluation_used": False},
        "evaluation": {"comparison_complete": True, "control_steps": 2295,
            "physical_validation": "unverified", "trials_per_controller": 3,
            "evaluation_seeds": [12, 34, 56], "independent_changed_states": 3,
            "controllers": {name: {"successes": 3 if name == "learned" else 0,
                "completed_intervals": 0 if name == "random" else 3, "reset_failures": 0,
                "mean_score": .9 if name == "learned" else -.1}
                for name in ("learned", "authored", "coast", "random")}}})
    seal(root)
    return root


@pytest.mark.parametrize("ready", [False, True])
def test_report_exposes_all_categories_without_grading_unknowns(tmp_path, ready):
    source = fixture(tmp_path / "build", ready=ready)
    out = tmp_path / "inspection"
    _, summary = run(source, out)
    card = summary["evidence_scorecard"]
    assert card["profile"] == "weft-evidence-scorecard-v1"
    assert set(rows(summary)) == DIMENSIONS
    assert card["source_manifest_sha256"] == hashlib.sha256((source / "manifest.json").read_bytes()).hexdigest()
    assert summary["evidence_scorecard_implementation_sha256"] == hashlib.sha256(
        Path(evidence_scorecard.__file__).read_bytes()).hexdigest()
    assert summary["task_success"] == "not_assessed_by_report"
    assert summary["physical_validation"] == "unverified"
    assert rows(summary)["execution_actions"]["status"] == "unavailable"
    assert metrics(rows(summary)["geometry_camera"])["independent_3d_error"]["value"] is None
    assert metrics(rows(summary)["task_learning_utility"])["physical_transfer_trials"]["value"] is None
    html = (out / "index.html").read_text()
    for dimension in DIMENSIONS:
        assert f'id="evidence-{dimension}"' in html
    links = Links()
    links.feed(html)
    assert "script" not in links.tags
    assert all(href.startswith("#") or (out / href).is_file() for href in links.links)
    assert "not a passed category or a completion percentage" in html


def test_floor_failure_is_visible_while_prepared_lab_stays_available(tmp_path):
    out = tmp_path / "inspection"
    _, summary = run(feasibility_fixture(tmp_path / "build"), out)
    row = rows(summary)["physical_relationships"]
    assert summary["build_status"] == summary["saved_lab_status"] == "ready"
    assert row["status"] == "failure_reported"
    values = metrics(row)
    assert (values["source_pose_floor_failures"]["numerator"], values["source_pose_floor_failures"]["denominator"]) == (1, 4)
    assert (values["control_grid_floor_failures"]["numerator"], values["control_grid_floor_failures"]["denominator"]) == (1, 7)
    assert values["peak_floor_penetration"]["recorded_gate"]["value"] == .025
    pointer = row["evidence"][0]
    assert pointer["source_name"] == "candidate/reference-feasibility.json"
    assert hashlib.sha256((out / pointer["path"]).read_bytes()).hexdigest() == pointer["sha256"]
    html = (out / "index.html").read_text()
    assert "Sandbox prepared; motion geometry checks failed" in html
    assert "Estimated motion intersects the assumed ground" in html
    assert "Failures recorded" in html


def test_bound_learning_counts_remain_exposed_simulation_not_physical_validation(tmp_path):
    _, summary = run(learning_fixture(tmp_path / "build"), tmp_path / "inspection")
    row = rows(summary)["task_learning_utility"]
    metric = metrics(row)["learned_successes"]
    assert (metric["numerator"], metric["denominator"]) == (3, 3)
    assert metric["evidence_kind"] == "simulated_development"
    assert metric["partition"] == "exposed_development"
    assert {item["source_name"] for item in metric["evidence"]} == {"result.json", "learning_workflow/result.json"}
    assert metrics(row)["physical_transfer_trials"]["value"] is None
    assert summary["task_success"] == "not_assessed_by_report"
    assert summary["physical_validation"] == "unverified"
    assert any("retained_original" in note for note in row["uncertainty_and_missing_measurements"])


@pytest.mark.parametrize("identity", ["scene_sha256", "task_sha256"])
def test_mismatched_learning_identity_cannot_seal_a_report(tmp_path, identity):
    source = learning_fixture(tmp_path / "build")
    learned = json.loads((source / "learning_workflow/result.json").read_bytes())
    put(source, "learning_workflow/result.json", {**learned, identity: "c" * 64})
    seal(source)
    with pytest.raises(ValueError, match="identities conflict"):
        run(source, tmp_path / "inspection")
    assert not (tmp_path / "inspection/manifest.json").exists()


@pytest.mark.parametrize("invalid", [True, "AB" * 32, "../elsewhere", 123])
def test_invalid_terminal_identity_is_not_exported(tmp_path, invalid):
    source = fixture(tmp_path / "build")
    result = json.loads((source / "result.json").read_bytes())
    put(source, "result.json", {**result, "task_sha256": invalid})
    seal(source)
    with pytest.raises(ValueError, match="Invalid sealed build identity"):
        run(source, tmp_path / "inspection")
    assert not (tmp_path / "inspection").exists()


def test_failure_receipt_does_not_import_plausible_unsealed_learning(tmp_path):
    source = learning_fixture(tmp_path / "build")
    result = json.loads((source / "result.json").read_bytes())
    put(source, "result.json", {**result, "status": "failed", "lab_status": "not_ready",
                              "seal_scope": "failure_receipt_only"})
    seal(source, receipt_only=True)
    _, summary = run(source, tmp_path / "inspection")
    card = summary["evidence_scorecard"]
    assert card["identities"] == {"scene_sha256": None, "task_sha256": None, "source_sha256": None}
    assert rows(summary)["task_learning_utility"]["status"] == "unavailable"
    assert set(metrics(rows(summary)["task_learning_utility"])) == {"physical_transfer_trials"}
    assert summary["learning"] == {}
    assert {item["source_name"] for item in summary["artifacts"]} == {"result.json"}


def test_multiline_failure_reason_remains_inspectable_and_escaped(tmp_path):
    source = fixture(tmp_path / "build", receipt_only=True)
    result = json.loads((source / "result.json").read_bytes())
    reason = 'Pose failed:\n\t<img src="https://example.invalid/leak">\r\nKeep the original.'
    put(source, "result.json", {**result, "reasons": [{"stage": "body_source", "message": reason}]})
    seal(source, receipt_only=True)
    out = tmp_path / "inspection"
    _, summary = run(source, out)
    assert rows(summary)["uncertainty_abstention"]["failure_examples"][0]["message"] == reason
    html = (out / "index.html").read_text()
    assert '<img src="https:' not in html
    assert "&lt;img" in html


def test_scorecard_render_escapes_text_and_rejects_unbound_links(tmp_path):
    _, summary = run(feasibility_fixture(tmp_path / "build"), tmp_path / "inspection")
    card = copy.deepcopy(summary["evidence_scorecard"])
    card["rows"][0]["label"] = '<img src="https://example.invalid/leak" onerror="alert(1)">'
    html = exercise_build_report._scorecard_html(card, summary["artifacts"], summary["source_manifest_sha256"])
    assert '<img src="https:' not in html
    assert "&lt;img" in html
    pointer = next(metric["evidence"][0] for row in card["rows"] for metric in row["metrics"]
                   if metric["evidence"] and metric["evidence"][0]["path"] is not None)
    pointer["path"] = "https://example.invalid/not-a-copied-artifact"
    with pytest.raises(ValueError, match="exact copied artifact"):
        exercise_build_report._scorecard_html(card, summary["artifacts"], summary["source_manifest_sha256"])
