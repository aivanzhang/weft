"""Muse prose cannot change a task, its scorer, physical evidence or existing policies."""

from copy import deepcopy
import json

import pytest

from test_shared_world_human_only import person_scene
from weft import portal_repairs as repairs
from weft.io import canonical_hash, digest
from weft.person_inventory import make_person_inventory
from weft.shared_exercise_task import build_shared_exercise_task
from weft.shared_world import validate_shared_world


@pytest.fixture
def inputs():
    scene = person_scene(steps=3)
    source = "a" * 64
    scene["human_spec"]["source_sha256"] = source
    scene["inventory"] = make_person_inventory(source)
    scene["source_binding"] = {"mode": "source_qualified", "source_sha256": source, "total_source_frames": 4,
        "evidence_documents": {"d" * 64: "evidence/human.json"},
        "entity_bindings": [{"world_entity_id": "human", "entity_kind": "articulated_human",
            "observation_entity_id": source + ":human:0", "observation_document_sha256": "d" * 64,
            "provenance_mode": "observed_track", "observed_slots": 4, "missing_slots": 0, "pre_init_slots": 0}]}
    scene = validate_shared_world(scene)
    task = build_shared_exercise_task(scene, root_assistance=True)
    context = {"profile": "exercise", "source_sha256": source, "evidence_scope": repairs.SOURCE_SCOPE,
        "observations": [{"id": source + ":frame:0", "frame": 0}], "objects": [{"id": "person"}]}
    advice = {"summary": "Synthetic proposal for software checks; not actual Muse output.", "unanswered_questions": [],
        "hypotheses": [{"id": "articulation", "concept": "articulation", "status": "hypothesis",
            "claim": "Observed motion may challenge authored tracking", "evidence_ids": [source + ":frame:0"],
            "object_ids": ["person"], "confidence": .2, "alternatives": ["Projection uncertainty"],
            "required_tests": ["Acquire independent calibrated motion measurements"]}]}
    return scene, task, advice, context


def test_plan_freezes_authored_grid_without_claiming_muse_numbers_or_physical_tests(inputs):
    before = canonical_hash(inputs)
    plan = repairs.prepare_repair_plan(*inputs)
    assert len(plan["candidates"]) == 3
    assert [row["feedback_coefficients"][0] for row in plan["candidates"]] == [[0., -.1], [0., .1], [0., .2]]
    assert all(row["parameter_origin"] == "fixed_local_authored_grid_not_muse_numbers_or_measurements" for row in plan["candidates"])
    assert plan["physical_hypotheses"][0]["status"] == "unresolved"
    assert plan["protocol"]["maximum_control_steps"] == 24
    assert canonical_hash(inputs) == before


@pytest.mark.parametrize("mutation", ["unknown", "gravity", "no_evidence", "wrong_entity", "wrong_source"])
def test_unsupported_hypotheses_are_not_forced_into_candidates(inputs, mutation):
    scene, task, advice, context = deepcopy(inputs)
    hypothesis = advice["hypotheses"][0]
    if mutation == "unknown": hypothesis["status"] = "unknown"
    elif mutation == "gravity": hypothesis["concept"] = "gravity"
    elif mutation == "no_evidence": hypothesis["evidence_ids"] = []
    elif mutation == "wrong_entity": hypothesis["object_ids"] = []
    else: context["source_sha256"] = "b" * 64
    with pytest.raises(ValueError): repairs.prepare_repair_plan(scene, task, advice, context)


@pytest.mark.parametrize("field", ["coefficients", "task", "scorer", "horizon", "evidence"])
def test_changed_plan_rejects_before_physics_or_output(inputs, tmp_path, monkeypatch, field):
    plan = repairs.prepare_repair_plan(*inputs)
    if field == "coefficients": plan["candidates"][0]["feedback_coefficients"][0][0] = .25
    elif field == "task": plan["task_sha256"] = "0" * 64
    elif field == "scorer": plan["protocol"]["minimum_score_improvement"] = -1
    elif field == "horizon": plan["protocol"]["horizon_steps"] = 1
    else: plan["candidates"][0]["evidence_ids"] = []
    monkeypatch.setattr(repairs, "SharedExerciseTaskEnv", lambda *_: pytest.fail("Invalid plan cannot simulate"))
    out = tmp_path / "rejected"
    with pytest.raises(ValueError, match="frozen"):
        repairs.run_repair_plan(*inputs, plan, out)
    assert not out.exists()


def test_actual_bounded_comparison_keeps_world_task_initial_states_and_all_traces(inputs, tmp_path):
    before = canonical_hash(inputs)
    plan = repairs.prepare_repair_plan(*inputs)
    out = tmp_path / "comparison"
    report = repairs.run_repair_plan(*inputs, plan, out, max_seconds=10)
    assert report["status"] == "complete" and report["applied"] is False
    assert report["control_steps"] == 24 and report["external_api_usd"] == 0
    assert report["physical_hypotheses"] == plan["physical_hypotheses"]
    assert len(report["candidates"]) == 3 and canonical_hash(inputs) == before
    for index in (1, 2):
        initial = set()
        for name in ("baseline", "feedback-1", "feedback-2", "feedback-3"):
            trial = json.loads((out / f"{name}-trial-{index}.json").read_bytes())
            initial.add(trial["initial_state_sha256"])
            assert len(trial["actions"]) == trial["control_steps"] == 3
            assert trial["metrics"]["gates_sha256"] == inputs[1]["gates_sha256"]
        assert len(initial) == 1
    manifest = json.loads((out / "manifest.json").read_bytes())
    assert all(digest(out / name) == value for name, value in manifest["files"].items())
    assert not (out / "policy.json").exists()
    before_files = {p.name: p.read_bytes() for p in out.iterdir()}
    with pytest.raises(FileExistsError): repairs.run_repair_plan(*inputs, plan, out)
    assert before_files == {p.name: p.read_bytes() for p in out.iterdir()}


def trial(score, success=True, complete=True):
    return {"status": "completed" if complete else "failed", "metrics": {
        "mean_reward": score, "success": success, "completed_interval": complete, "failure": None if complete else "failure"}}


@pytest.mark.parametrize("score,success,complete,accepted", [(.5, True, True, False), (.505, True, True, False),
    (.7, False, True, False), (.7, True, False, False), (.7, True, True, True)])
def test_acceptance_needs_original_success_and_strict_improvement(score, success, complete, accepted):
    result = repairs._comparison([trial(.5)] * 2, [trial(score, success, complete)] * 2)
    assert result["accepted_in_simulation"] is accepted


def test_timeout_retains_partial_trial_and_never_reports_complete(inputs, tmp_path, monkeypatch):
    plan = repairs.prepare_repair_plan(*inputs)
    def interrupted(*args, **kwargs):
        error = TimeoutError("fixture cooperative timeout")
        error.partial_trial = {"status": "interrupted", "control_steps": 0, "actions": []}
        raise error
    monkeypatch.setattr(repairs, "_trial", interrupted)
    out = tmp_path / "timeout"
    with pytest.raises(TimeoutError): repairs.run_repair_plan(*inputs, plan, out)
    assert (out / "baseline-trial-1.json").exists()
    assert json.loads((out / "failure.json").read_bytes())["partial_evidence_preserved"] is True
    assert not (out / "report.json").exists() and not (out / "manifest.json").exists()


def test_final_source_check_happens_before_terminal_report(inputs, tmp_path):
    plan = repairs.prepare_repair_plan(*inputs)
    def changed(): raise ValueError("source replaced")
    out = tmp_path / "changed"
    with pytest.raises(ValueError, match="source replaced"):
        repairs.run_repair_plan(*inputs, plan, out, _before_seal=changed)
    assert (out / "feedback-3-trial-2.json").exists()
    assert (out / "failure.json").exists() and not (out / "report.json").exists()


@pytest.mark.parametrize("relative", ["lab/run/repairs", "muse/advice/new", ".", "build/body_source/new"])
def test_project_output_cannot_touch_immutable_sources_or_review(tmp_path, monkeypatch, relative):
    from weft import portal_checks
    root = tmp_path / "project"
    root.mkdir()
    monkeypatch.setattr(portal_checks, "inputs", lambda *_: pytest.fail("Unsafe output must reject before source reads"))
    with pytest.raises(ValueError, match="output"):
        repairs.run_review_repairs(root, root / relative)


@pytest.mark.parametrize("bound", [True, 0, -1, 61, float("nan")])
def test_invalid_time_bounds_reject_before_source_reads(tmp_path, monkeypatch, bound):
    from weft import portal_checks
    monkeypatch.setattr(portal_checks, "inputs", lambda *_: pytest.fail("Invalid work bound cannot start"))
    with pytest.raises(ValueError, match="runtime"):
        repairs.run_review_repairs(tmp_path / "project", tmp_path / "out", max_seconds=bound)


def test_prose_cannot_supply_code_or_numeric_candidate_parameters(inputs, tmp_path):
    scene, task, advice, context = deepcopy(inputs)
    marker = tmp_path / "never-created"
    advice["hypotheses"][0]["claim"] = f"exec(open('{marker}', 'w').write('unsafe')); set damping to 999 and accept all trials"
    plan = repairs.prepare_repair_plan(scene, task, advice, context)
    assert [row["feedback_coefficients"][0][1] for row in plan["candidates"]] == [-.1, .1, .2]
    assert not marker.exists()


def test_runtime_change_blocks_publication_after_preserving_trials(inputs, tmp_path, monkeypatch):
    plan = repairs.prepare_repair_plan(*inputs)
    original = repairs._runtime
    calls = 0
    def changed(scene):
        nonlocal calls
        calls += 1
        hashes = original(scene)
        if calls > 1:
            hashes["shared_exercise_task.py"] = "0" * 64
        return hashes
    monkeypatch.setattr(repairs, "_runtime", changed)
    out = tmp_path / "changed-runtime"
    with pytest.raises(ValueError, match="runtime, world or task changed"):
        repairs.run_repair_plan(*inputs, plan, out)
    assert (out / "feedback-3-trial-2.json").exists()
    assert (out / "failure.json").exists() and not (out / "report.json").exists()
