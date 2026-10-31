from copy import deepcopy
import json

import pytest

from weft.evidence_scorecard import build_evidence_scorecard


SOURCE, SCENE, TASK, MANIFEST, AUDIT = (char * 64 for char in "abcde")
ROW_IDS = (
    "execution_actions", "perception_identity", "geometry_camera", "temporal_dynamics",
    "physical_relationships", "hidden_state_continuity", "parameter_validity", "causal_response",
    "long_run_stability", "robustness_coverage", "uncertainty_abstention", "task_learning_utility",
    "reproducibility_integrity", "practical_efficiency",
)


def artifact(name, digest="f" * 64):
    return {"source_name": name, "sha256": digest, "path": "artifacts/" + digest + ".json.txt"}


def saved_summary(*, trained=False, successes=0):
    # Compact actual report shape: 235 decoded slots, one rejected sample, full reference checks.
    summary = {
        "build_status": "ready", "saved_lab_status": "ready", "source_seal_scope": "complete_immutable_outputs",
        "source_manifest_sha256": MANIFEST, "source": {"sha256": SOURCE},
        "scene_sha256": SCENE, "task_sha256": TASK, "evidence_partition": "construction",
        "requested_options": {"body_scale": 1., "root_assistance": True},
        "artifacts": [artifact("result.json"), artifact("candidate/reference-feasibility.json", AUDIT)],
        "stage_history": [{"elapsed_seconds": .5}, {"elapsed_seconds": 2.}, {"elapsed_seconds": 3.}],
        "coverage": {
            "status": "available", "source_sha256": SOURCE, "decoded_frames": 235,
            "sampled_frames": 118, "unsampled_frames": 117, "usable_samples": 117,
            "partial_samples": 117, "unusable_samples": 1, "competing_samples": 0,
            "issues": [{"kind": "unusable_pose", "first_frame": 44, "last_frame": 44, "seek_s": 1.481}],
            "movement": {"long_gaps": 0, "unfilled_gaps": 0, "maximum_gap_s": 0.},
        },
        "reference_feasibility": {
            "status": "infeasible", "audit_artifact": "candidate/reference-feasibility.json",
            "audit_sha256": AUDIT, "scene_sha256": SCENE, "source_sha256": SOURCE,
            "source_binding_mode": "source_qualified", "floor_limit_model_units": .025,
            "max_penetration_model_units": .07185566666204826,
            "source_pose": {"checks": 117, "floor_failures": 8, "failed_preflight": 8},
            "control_grid": {"checks": 235, "floor_failures": 14, "failed_preflight": 14},
            "static_conflict_poses": 0, "initial_cross_entity_intersections": 0, "physics_steps": 0,
            "failure_examples": [{"kind": "source_pose", "time_s": 4 / 30, "source_reference_index": 2,
                                  "max_penetration_model_units": .04, "passes_floor_limit": False,
                                  "static_conflict_count": 0}],
            "failure_examples_omitted": 21,
        },
    }
    if trained:
        summary["artifacts"].append(artifact("learning_workflow/result.json", "1" * 64))
        summary["learning"] = {
            "status": "ready", "scene_sha256": SCENE, "task_sha256": TASK,
            "training": {"evaluated_episodes": 8, "control_steps": 1872},
            "selection": {"feasible_candidates": 0, "unique_candidates": 4, "evaluation_used": False,
                          "selection_status": "retained_original_no_feasible_candidate"},
            "evaluation": {"comparison_complete": True, "control_steps": 2295,
                           "trials_per_controller": 3, "independent_changed_states": 3,
                           "evaluation_seeds": [2115623776, 511847713, 919039072],
                           "controllers": {name: {"successes": successes, "completed_intervals": 3,
                                                    "reset_failures": 0, "mean_score": .8841404539193443}
                                           for name in ("learned", "authored", "coast", "random")}},
        }
    return summary


def rows(summary):
    return {row["id"]: row for row in build_evidence_scorecard(summary)["rows"]}


def metrics(row):
    return {metric["id"]: metric for metric in row["metrics"]}


def test_missing_evidence_is_fourteen_unavailable_dimensions_not_zero_passes():
    scorecard = build_evidence_scorecard({})
    assert tuple(row["id"] for row in scorecard["rows"]) == ROW_IDS
    assert scorecard["profile"] == "weft-evidence-scorecard-v1"
    assert all(row["status"] == "unavailable" for row in scorecard["rows"])
    assert all(metric["value"] is None for row in scorecard["rows"] for metric in row["metrics"])
    assert "percentage" not in scorecard and "score" not in scorecard


def test_ready_untrained_build_retains_gaps_floor_failure_and_exact_artifact_links():
    result = rows(saved_summary())
    assert result["execution_actions"]["status"] == "unavailable"
    assert result["task_learning_utility"]["status"] == "unavailable"
    perception = metrics(result["perception_identity"])
    assert perception["usable_sample_count"]["numerator"] == 117
    assert perception["usable_sample_count"]["denominator"] == 118
    assert perception["independent_identity_errors"]["value"] is None
    assert metrics(result["hidden_state_continuity"])["unobserved_slots"]["value"] == 118
    assert result["perception_identity"]["failure_examples"][0]["first_frame"] == 44
    physical = result["physical_relationships"]
    assert physical["status"] == "failure_reported"
    measure = metrics(physical)
    assert (measure["source_pose_floor_failures"]["value"], measure["source_pose_floor_failures"]["denominator"]) == (8, 117)
    assert (measure["control_grid_floor_failures"]["value"], measure["control_grid_floor_failures"]["denominator"]) == (14, 235)
    assert measure["peak_floor_penetration"]["recorded_gate"]["value"] == .025
    assert physical["failure_examples"][0]["max_penetration_model_units"] == .04
    assert physical["failure_examples_omitted"] == 21
    assert physical["evidence"] == [artifact("candidate/reference-feasibility.json", AUDIT)]
    assert measure["measured_contact_error"]["value"] is None


@pytest.mark.parametrize("successes", [0, 3])
def test_saved_simulation_success_never_establishes_physical_accuracy_or_execution(successes):
    result = rows(saved_summary(trained=True, successes=successes))
    learning = result["task_learning_utility"]
    measured = metrics(learning)
    assert measured["learned_successes"]["value"] == successes
    assert measured["learned_successes"]["denominator"] == 3
    assert measured["learned_successes"]["partition"] == "exposed_development"
    assert measured["physical_transfer_trials"]["value"] is None
    assert measured["learned_mean_reward"]["unit"] == "reward"
    assert len(learning["metrics"]) == 16
    assert learning["status"] == ("failure_reported" if successes == 0 else "evidence_available")
    assert result["execution_actions"]["status"] == "unavailable"
    assert metrics(result["geometry_camera"])["independent_3d_error"]["value"] is None
    assert metrics(result["long_run_stability"])["long_rollout_stress_trials"]["value"] is None
    notes = " ".join(learning["uncertainty_and_missing_measurements"])
    assert "retained_original_no_feasible_candidate" in notes and "not unassisted balance" in notes
    assert "not evidence of learned improvement" in notes
    assert measured["learned_successes"]["evidence"] == [artifact("result.json"), artifact("learning_workflow/result.json", "1" * 64)]


def test_mixed_evidence_is_tagged_per_metric_and_stage_clock_is_not_summed():
    result = rows(saved_summary(trained=True))
    robustness = result["robustness_coverage"]
    assert robustness["partition"] == robustness["evidence_kind"] == "unknown"
    observed = metrics(robustness)
    assert observed["source_count"]["partition"] == "construction"
    assert observed["saved_changed_starts"]["partition"] == "exposed_development"
    efficiency = metrics(result["practical_efficiency"])
    assert efficiency["last_recorded_stage_elapsed"]["value"] == 3.
    assert efficiency["training_actions"]["value"] == 1872
    assert efficiency["evaluation_actions"]["value"] == 2295
    assert efficiency["total_run_cost"]["value"] is None


def test_actual_reset_failure_is_visible_but_cannot_become_action_validation():
    summary = saved_summary(trained=True)
    summary["learning"]["evaluation"]["controllers"]["random"].update(completed_intervals=2, reset_failures=1)
    execution = rows(summary)["execution_actions"]
    assert execution["status"] == "failure_reported"
    assert metrics(execution)["saved_reset_failures"]["value"] == 1
    assert metrics(execution)["seeded_execution_trials"]["value"] is None


def compact_lab_summary():
    names = ["exercise-lab.json", "policy.json", "scene.json", "exercise-task.json"]
    return {
        "source_manifest_sha256": MANIFEST, "scene_sha256": SCENE, "task_sha256": TASK,
        "artifacts": [artifact(name, str(index) * 64) for index, name in enumerate(names)],
        "learning_artifact_names": names,
        "learning": {"scene_sha256": SCENE, "task_sha256": TASK, "evaluation": {
            "trials_per_controller": 3, "controllers": {
                "learned": {"successes": 3}, "authored": {"successes": 3},
                "coast": {"successes": 0}, "random": {"successes": 0},
            }}},
    }


@pytest.mark.parametrize("explicit_nulls", [False, True])
def test_compact_lab_successes_do_not_infer_completion_resets_rewards_or_training(explicit_nulls):
    summary = compact_lab_summary()
    if explicit_nulls:
        for row in summary["learning"]["evaluation"]["controllers"].values():
            row.update(completed_intervals=None, reset_failures=None, mean_score=None)
    result = rows(summary)
    measures = metrics(result["task_learning_utility"])
    for controller, successes in (("learned", 3), ("authored", 3), ("coast", 0), ("random", 0)):
        success = measures[controller + "_successes"]
        assert (success["value"], success["numerator"], success["denominator"]) == (successes, successes, 3)
        assert success["partition"] == "exposed_development"
        assert success["evidence_kind"] == "simulated_development"
        assert success["evidence"] == summary["artifacts"]
        completion = measures[controller + "_completed_intervals"]
        assert completion["value"] is completion["numerator"] is completion["denominator"] is None
        assert measures[controller + "_mean_reward"]["value"] is None
    assert measures["physical_transfer_trials"]["value"] is None
    assert "training_control_steps" not in measures
    assert result["long_run_stability"]["status"] == "unavailable"
    assert result["execution_actions"]["status"] == "unavailable"
    assert "completed_task_intervals" not in metrics(result["long_run_stability"])
    assert "saved_reset_failures" not in metrics(result["execution_actions"])
    assert "saved_changed_starts" not in metrics(result["robustness_coverage"])
    assert "missing values remain unavailable" in " ".join(result["task_learning_utility"]["uncertainty_and_missing_measurements"])


def test_partial_fields_only_aggregate_when_every_contributing_controller_has_them():
    summary = compact_lab_summary()
    controllers = summary["learning"]["evaluation"]["controllers"]
    controllers["learned"].update(completed_intervals=3, reset_failures=0)
    controllers["coast"].update(completed_intervals=0, reset_failures=1)
    partial = rows(summary)
    assert metrics(partial["task_learning_utility"])["learned_completed_intervals"]["value"] == 3
    assert "completed_task_intervals" not in metrics(partial["long_run_stability"])
    assert "saved_reset_failures" not in metrics(partial["execution_actions"])
    controllers["authored"].update(completed_intervals=3, reset_failures=0)
    controllers["random"].update(completed_intervals=0, reset_failures=0)
    complete = rows(summary)
    completed = metrics(complete["long_run_stability"])["completed_task_intervals"]
    assert (completed["value"], completed["denominator"]) == (6, 12)
    reset = metrics(complete["execution_actions"])["saved_reset_failures"]
    assert (reset["value"], reset["denominator"]) == (1, 12)
    assert complete["execution_actions"]["status"] == "failure_reported"


def test_known_reset_counts_do_not_require_unknown_completion_counts():
    summary = compact_lab_summary()
    for name, row in summary["learning"]["evaluation"]["controllers"].items():
        row["reset_failures"] = 1 if name == "coast" else 0
    result = rows(summary)
    assert result["long_run_stability"]["status"] == "unavailable"
    assert metrics(result["execution_actions"])["saved_reset_failures"]["value"] == 1


@pytest.mark.parametrize("field,value", [
    ("completed_intervals", 2), ("completed_intervals", True),
    ("completed_intervals", float("nan")), ("reset_failures", 1),
    ("reset_failures", -1), ("reset_failures", "0"), ("mean_score", float("nan")),
    ("successes", None),
])
def test_partial_controller_metadata_still_rejects_known_invalid_counts(field, value):
    summary = compact_lab_summary()
    summary["learning"]["evaluation"]["controllers"]["learned"][field] = value
    with pytest.raises(ValueError):
        build_evidence_scorecard(summary)


@pytest.mark.parametrize("names", [[], None, "scene.json", ["missing.json"], ["scene.json", "scene.json"],
                                  ["../scene.json"], ["scene.json"] * 9])
def test_custom_learning_pointers_must_be_bounded_unique_exact_artifacts(names):
    summary = compact_lab_summary()
    summary["learning_artifact_names"] = names
    with pytest.raises(ValueError):
        build_evidence_scorecard(summary)


def test_custom_pointers_cannot_make_unbound_learning_available():
    summary = compact_lab_summary()
    del summary["task_sha256"]
    assert rows(summary)["task_learning_utility"]["status"] == "unavailable"
    summary["learning_artifact_names"].append("not-copied.json")
    with pytest.raises(ValueError, match="present in artifacts"):
        build_evidence_scorecard(summary)


def test_eight_custom_learning_pointers_preserve_exact_order_and_bounds():
    summary = compact_lab_summary()
    summary["learning"]["evaluation"]["controllers"] = {"learned": {"successes": 3}}
    extra = [artifact("proof-" + str(index) + ".json", str(index) * 64) for index in range(4, 8)]
    summary["artifacts"].extend(extra)
    summary["learning_artifact_names"].extend(row["source_name"] for row in extra)
    result = build_evidence_scorecard(summary)
    learning = next(row for row in result["rows"] if row["id"] == "task_learning_utility")
    assert learning["evidence"] == summary["artifacts"]
    assert len(learning["evidence"]) == 8
    assert len(json.dumps(result).encode()) < 64 * 1024


@pytest.mark.parametrize("missing", ["scene_sha256", "task_sha256", "learning_scene", "learning_task"])
def test_unbound_historical_learning_counts_stay_unavailable(missing):
    summary = saved_summary(trained=True)
    if missing.startswith("learning_"):
        del summary["learning"][missing.removeprefix("learning_") + "_sha256"]
    else:
        del summary[missing]
    result = rows(summary)
    assert result["task_learning_utility"]["status"] == "unavailable"
    assert "learned_successes" not in metrics(result["task_learning_utility"])


def test_historical_missing_and_unconsumed_metadata_are_not_rejected():
    summary = {"coverage": None, "reference_feasibility": None, "learning": {
        "status": "ready", "evaluation": {"policy_successes": 0, "trials": 3}},
        "unconsumed_historical_metadata": {"custom": float("nan")}, "source": {"title": "<script>not rendered</script>"}}
    result = rows(summary)
    assert result["task_learning_utility"]["status"] == "unavailable"
    assert all(row["status"] == "unavailable" for row in result.values())


def test_failed_learning_with_null_unreached_stages_remains_unavailable():
    summary = saved_summary()
    summary["learning"] = {"status": "failed", "scene_sha256": SCENE, "task_sha256": TASK,
                           "training": None, "selection": None, "evaluation": None}
    assert rows(summary)["task_learning_utility"]["status"] == "unavailable"


def test_multiline_saved_failure_text_is_preserved_but_not_used_as_a_link():
    summary = {"reasons": [{"message": "Failed phase\n  original detail\tvalue"}]}
    assert rows(summary)["uncertainty_abstention"]["failure_examples"][0]["message"] == summary["reasons"][0]["message"]
    summary["artifacts"] = [artifact("result.json")]
    summary["artifacts"][0]["path"] = "artifacts/line\nbreak.txt"
    with pytest.raises(ValueError, match="relative artifact path"):
        build_evidence_scorecard(summary)


def test_receipt_only_failure_never_exposes_unsealed_body_learning_or_source():
    summary = saved_summary(trained=True)
    summary.update(source_seal_scope="failure_receipt_only", build_status="failed",
                   artifacts=[artifact("result.json")], reasons=[{"stage": "seal", "message": "Disk full"}])
    # These objects would fail normal parsing; they must not be treated as sealed evidence.
    summary["coverage"] = summary["learning"] = summary["source"] = {"unexpected": float("nan")}
    scorecard = build_evidence_scorecard(summary)
    result = {row["id"]: row for row in scorecard["rows"]}
    assert scorecard["identities"] == {"source_sha256": None, "scene_sha256": None, "task_sha256": None}
    assert result["physical_relationships"]["status"] == "unavailable"
    assert result["task_learning_utility"]["status"] == "unavailable"
    assert result["uncertainty_abstention"]["status"] == "failure_reported"
    assert result["uncertainty_abstention"]["failure_examples"] == [{"stage": "seal", "message": "Disk full"}]


def test_analytic_fixture_has_no_invented_video_identity():
    summary = saved_summary()
    summary.pop("coverage")
    summary["source"] = {}
    summary["reference_feasibility"].update(source_binding_mode="analytic_fixture", source_sha256=None)
    scorecard = build_evidence_scorecard(summary)
    assert scorecard["identities"]["source_sha256"] is None
    assert rows(summary)["physical_relationships"]["status"] == "failure_reported"


@pytest.mark.parametrize("initial", [0, 1])
def test_zero_floor_failures_are_saved_checks_not_a_physical_pass(initial):
    summary = saved_summary()
    audit = summary["reference_feasibility"]
    audit.update(status="infeasible" if initial else "feasible", initial_cross_entity_intersections=initial,
                 max_penetration_model_units=0., failure_examples=[], failure_examples_omitted=0)
    for key in ("source_pose", "control_grid"):
        audit[key].update(floor_failures=0, failed_preflight=0)
    physical = rows(summary)["physical_relationships"]
    assert physical["status"] == ("failure_reported" if initial else "evidence_available")
    assert metrics(physical)["source_pose_floor_failures"]["value"] == 0
    assert metrics(physical)["measured_contact_error"]["value"] is None


def test_failure_receipt_rejects_links_outside_its_sealed_result():
    summary = saved_summary()
    summary["source_seal_scope"] = "failure_receipt_only"
    with pytest.raises(ValueError, match="unsealed artifact"):
        build_evidence_scorecard(summary)


@pytest.mark.parametrize("path,value", [
    (("coverage", "source_sha256"), "0" * 64),
    (("coverage", "decoded_frames"), 234),
    (("coverage", "usable_samples"), 118),
    (("coverage", "partial_samples"), 118),
    (("coverage", "competing_samples"), 2),
    (("coverage", "sampled_frames"), float("nan")),
    (("coverage", "movement", "maximum_gap_s"), float("inf")),
    (("reference_feasibility", "scene_sha256"), "0" * 64),
    (("reference_feasibility", "audit_sha256"), "0" * 64),
    (("reference_feasibility", "source_sha256"), "0" * 64),
    (("reference_feasibility", "status"), "feasible"),
    (("reference_feasibility", "physics_steps"), 1),
    (("reference_feasibility", "max_penetration_model_units"), .01),
    (("reference_feasibility", "failure_examples_omitted"), 0),
    (("reference_feasibility", "source_pose", "floor_failures"), 118),
    (("learning", "task_sha256"), "0" * 64),
    (("learning", "scene_sha256"), "0" * 64),
    (("learning", "training", "control_steps"), -1),
    (("learning", "training", "control_steps"), 10 ** 400),
    (("learning", "selection", "feasible_candidates"), 5),
    (("learning", "evaluation", "trials_per_controller"), True),
    (("learning", "evaluation", "evaluation_seeds"), [1, 1, 2]),
    (("learning", "evaluation", "independent_changed_states"), 4),
    (("learning", "evaluation", "controllers", "learned", "successes"), 4),
    (("learning", "evaluation", "controllers", "learned", "reset_failures"), 1),
    (("learning", "evaluation", "controllers", "learned", "mean_score"), float("nan")),
    (("evidence_partition",), "fresh_final"),
    (("evidence_partition",), "typo"),
    (("source_manifest_sha256",), "not-a-hash"),
    (("requested_options", "root_assistance"), "true"),
    (("stage_history",), [{"elapsed_seconds": 2}, {"elapsed_seconds": 1}]),
])
def test_consumed_identity_numeric_and_count_contradictions_reject(path, value):
    summary = saved_summary(trained=True)
    target = summary
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    with pytest.raises(ValueError):
        build_evidence_scorecard(summary)


@pytest.mark.parametrize("path", ["../escape", "/absolute/file", "artifacts//duplicate", "artifacts/./dot", "https://remote", "x\\y"])
def test_only_safe_exact_artifact_links_are_allowed(path):
    summary = saved_summary()
    summary["artifacts"][0]["path"] = path
    with pytest.raises(ValueError, match="relative artifact path"):
        build_evidence_scorecard(summary)


def test_duplicate_pointer_and_issue_identities_reject():
    summary = saved_summary()
    summary["artifacts"].append(deepcopy(summary["artifacts"][0]))
    with pytest.raises(ValueError, match="unique"):
        build_evidence_scorecard(summary)
    summary = saved_summary()
    summary["coverage"]["issues"] *= 2
    with pytest.raises(ValueError, match="identities conflict"):
        build_evidence_scorecard(summary)


@pytest.mark.parametrize("kind", ["artifacts", "history", "coverage_issues", "feasibility_examples", "controllers", "reasons"])
def test_nested_inputs_have_hard_bounds(kind):
    summary = saved_summary(trained=True)
    if kind == "artifacts":
        summary["artifacts"] *= 44
    elif kind == "history":
        summary["stage_history"] = [{"elapsed_seconds": 1}] * 2049
    elif kind == "coverage_issues":
        summary["coverage"]["issues"] *= 1801
    elif kind == "feasibility_examples":
        summary["reference_feasibility"]["failure_examples"] *= 9
    elif kind == "controllers":
        summary["learning"]["evaluation"]["controllers"]["extra"] = {}
    else:
        summary["reasons"] = [{}] * 35
    with pytest.raises(ValueError):
        build_evidence_scorecard(summary)


def test_large_coverage_input_retains_only_eight_examples_and_exact_omissions():
    summary = saved_summary()
    summary.pop("reference_feasibility")
    summary["coverage"].update(decoded_frames=1800, sampled_frames=1800, unsampled_frames=0,
                               usable_samples=0, partial_samples=0, unusable_samples=1800,
                               issues=[{"kind": "unusable_pose", "first_frame": index, "last_frame": index,
                                        "seek_s": index / 30} for index in range(1800)])
    row = rows(summary)["perception_identity"]
    assert len(row["failure_examples"]) == 8 and row["failure_examples_omitted"] == 1792
    assert metrics(row)["usable_sample_count"]["value"] == 0
    assert row["status"] == "failure_reported"


def test_no_io_input_mutation_or_unbounded_output(monkeypatch):
    summary = saved_summary(trained=True)
    before = deepcopy(summary)

    def forbid(*args, **kwargs):
        raise AssertionError("Pure scorecard attempted I/O")

    monkeypatch.setattr("builtins.open", forbid)
    result = build_evidence_scorecard(summary)
    assert summary == before
    assert result == build_evidence_scorecard(summary)
    assert len(json.dumps(result, allow_nan=False).encode()) < 64 * 1024
    for row in result["rows"]:
        assert len(row["metrics"]) <= 16 and len(row["failure_examples"]) <= 8 and len(row["evidence"]) <= 8
        assert len({metric["id"] for metric in row["metrics"]}) == len(row["metrics"])
    result["rows"][1]["failure_examples"][0]["first_frame"] = 99
    assert summary == before


def test_large_individually_bounded_fields_cannot_exceed_total_output_budget():
    summary = saved_summary(trained=True)
    for pointer in summary["artifacts"]:
        pointer["path"] = "artifacts/" + "x" * 500
    summary["reasons"] = [{key: "x" * 2000 for key in ("stage", "code", "error_type", "type", "message")} for _ in range(8)]
    with pytest.raises(ValueError, match="64-KiB"):
        build_evidence_scorecard(summary)
