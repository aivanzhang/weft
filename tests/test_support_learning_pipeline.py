"""Support workflow orchestration and source custody without native simulation or training."""

from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from test_shared_world_support_construction import block_native, config, make_body
from weft import exercise_learning_pipeline as pipeline
from weft import shared_world_support as builder
from weft.io import canonical_hash, digest, write_json
from weft.pipeline import verify_artifacts
from weft.shared_support_fixture import static_support_fixture


SETTINGS = {"training_seed": 1729, "evaluation_seed": 924017, "generations": 1, "population": 4, "trials": 3}


def document(path):
    return json.loads(path.read_bytes())


def files(path):
    return {entry.relative_to(path).as_posix(): entry.read_bytes() for entry in path.rglob("*") if entry.is_file()}


def seal(path, status="ready", **extra):
    write_json(path / "manifest.json", {"schema_version": 1, "status": status, **extra,
        "files": {name: digest(path / name) for name in files(path) if name != "manifest.json"}})
    return digest(path / "manifest.json")


@pytest.fixture(autouse=True)
def no_native(monkeypatch):
    block_native(monkeypatch)
    monkeypatch.setattr(builder, "audit_shared_world_feasibility", lambda _scene: {
        "status": "infeasible", "summary": {"physics_steps": 0}, "scope": "Analytical fixture; no geometry execution"})


def source_package(tmp_path, *, constructed=False):
    source = tmp_path / "source"
    if constructed:
        body = make_body(tmp_path / "body")
        builder.construct_static_support_run(body, config(body), source)
    else:
        source.mkdir()
        scene, task = static_support_fixture()
        write_json(source / "scene.json", scene)
        write_json(source / "exercise-task.json", task)
        seal(source)
    return source


def stub_stages(monkeypatch, *, correction=False):
    calls = []

    def package(source, out, status, documents):
        out.mkdir()
        for name in ("scene.json", "exercise-task.json", "source-clock.json", "inputs/support-config.json"):
            if (source / name).is_file():
                (out / name).parent.mkdir(parents=True, exist_ok=True)
                (out / name).write_bytes((source / name).read_bytes())
        for name, value in documents.items():
            write_json(out / name, value)
        return seal(out, status)

    def training(source, out, **options):
        calls.append(("training", source, out, options))
        plan = document(out.parent / "plan.json")
        scene, task = document(source / "scene.json"), document(Path(options["task_path"]))
        hashes = pipeline.residual._runtime_hashes(scene)
        env = SimpleNamespace(scene=scene, task=task, scene_sha256=canonical_hash(scene), task_sha256=canonical_hash(task),
            evaluator_sha256=pipeline.residual._contract(scene, hashes)["evaluator_sha256"],
            action_names=scene["action_contract"]["names"][:19])
        policy = pipeline.residual._policy(env, np.full((2, 19), .01 if correction else 0.), options["seed"], .5, hashes)
        settings = {"training_seeds": pipeline.residual._seed_plan(options["seed"], 2),
            "maximum_control_steps": 2 * options["generations"] * options["population"] * task["interval"]["horizon_steps"]}
        assert plan["training_seeds"] == settings["training_seeds"]
        assert plan["policy_profile"] == policy["profile"] and plan["evaluator_sha256"] == policy["evaluator_sha256"]
        pin = package(source, out, "trained", {"policy.json": policy, "settings.json": settings,
            "training.json": {"scope": "Stub stage boundary; no training executed", "status": "trained"}})
        return {"manifest_sha256": pin, "policy_sha256": digest(out / "policy.json"), "control_steps": 0,
                "evaluated_episodes": 0, "completed_generations": options["generations"]}

    def evaluation(source, out, **options):
        calls.append(("independent_evaluation", source, out, options))
        assert source == out.parent / "training" and not (out.parent / "selected").exists()
        plan, settings = document(out.parent / "plan.json"), document(source / "settings.json")
        seeds = pipeline.residual._seed_plan(options["seed"], options["trials"], settings["training_seeds"])
        assert seeds == plan["evaluation_seeds"] and set(seeds).isdisjoint(settings["training_seeds"])
        selection = document(out.parent / "stages.json")["stages"][-2]
        assert selection["stage"] == "training_selection" and selection["policy_sha256"] == digest(source / "policy.json")
        pin = package(source, out, "evaluated", {"policy.json": document(source / "policy.json")})
        return {"manifest_sha256": pin, "policy_sha256": digest(out / "policy.json"), "evaluation_seeds": seeds,
                "maximum_control_steps": plan["maximum_evaluation_steps"], "control_steps": 0,
                "comparison_complete": True, "independent_changed_states": len(seeds),
                "controllers": {name: {"successes": 0, "completed_intervals": 0, "reset_failures": 0, "mean_score": 0.}
                                for name in pipeline.residual.CONTROLLERS}}

    def lab(source, out, **options):
        calls.append(("interactive_lab", source, out, options))
        assert source == options["policy_run"] == out.parent / "training"
        assert options["evaluation_run"] == out.parent / "evaluation"
        out.mkdir()
        pin = package(source, out / "run", "ready", {"policy.json": document(source / "policy.json")})
        write_json(out / "status.json", {"status": "ready", "scope": "Stub lab boundary; not simulation admission"})
        return {"manifest_sha256": pin, "lab_directory": str(out),
            "scene_sha256": canonical_hash(document(source / "scene.json")),
            "task_sha256": canonical_hash(document(source / "exercise-task.json"))}

    monkeypatch.setattr(pipeline.residual, "train_shared_exercise_residual_policy", training)
    monkeypatch.setattr(pipeline.residual, "evaluate_shared_exercise_residual_policy", evaluation)
    monkeypatch.setattr(pipeline, "prepare_shared_exercise_lab", lab)
    monkeypatch.setattr(pipeline, "select_feasible_shared_exercise_policy", lambda *_a, **_k: pytest.fail("Support invoked legacy selector"))
    return calls


@pytest.mark.parametrize("correction", [False, True])
def test_support_skips_selector_and_keeps_one_frozen_task_policy_and_lab(tmp_path, monkeypatch, correction):
    source = source_package(tmp_path)
    before = files(source)
    calls, events = stub_stages(monkeypatch, correction=correction), []
    out = tmp_path / "workflow"
    result = pipeline.prepare_exercise_learning_pipeline(source, out, train_settings=SETTINGS, progress=events.append)
    plan = document(out / "plan.json")
    assert result["profile"] == plan["profile"] == document(out / "manifest.json")["profile"] == pipeline.SUPPORT_PROFILE
    assert result["status"] == result["lab_status"] == "ready"
    assert [call[0] for call in calls] == ["training", "independent_evaluation", "interactive_lab"]
    assert [path.name for path in sorted(out.iterdir()) if path.is_dir()] == ["evaluation", "lab", "training"]
    selection = result["selection"]
    assert selection["selection_status"] == "not_applicable" and selection["performed"] is False
    assert selection["evaluation_used"] is False and selection["reason"] == pipeline.SUPPORT_SELECTION_REASON
    assert selection["directory"] == str(out / "training") and "feasible_candidates" not in selection
    assert selection["manifest_sha256"] == result["stage_manifests"]["training"]
    assert result["training"]["selected_feedback_kind"] == ("searched_joint_residuals" if correction else "authored_unchanged")
    assert events[1]["selected_feedback_kind"] == result["training"]["selected_feedback_kind"]
    assert [(row["stage"], row["status"]) for row in events] == [
        ("training", "running"), ("training", "trained"), ("training_selection", "not_applicable"),
        ("independent_evaluation", "running"), ("independent_evaluation", "evaluated"),
        ("interactive_lab", "running"), ("interactive_lab", "ready")]
    assert plan["stage_output_reservations"]["training_selection"] == {"files": 0, "bytes": 0, "seconds": 0}
    assert plan["maximum_training_steps"] == 480 and plan["maximum_evaluation_steps"] == 720
    assert plan["maximum_control_steps"] == 1200 and plan["maximum_physics_substeps"] == 19200
    policy = document(out / "training/policy.json")
    assert plan["gates_sha256"] == policy["gates_sha256"] != pipeline.GATES_SHA256
    assert plan["policy_runtime_sha256"] == policy["runtime_sha256"] and "shared_support_task.py" in policy["runtime_sha256"]
    assert plan["evaluation_feedback_to_optimizer"] is False
    for directory in ("training", "evaluation", "lab/run"):
        for name in ("scene.json", "exercise-task.json"):
            assert (out / directory / name).read_bytes() == before[name]
        assert (out / directory / "policy.json").read_bytes() == (out / "training/policy.json").read_bytes()
    assert files(source) == before
    assert verify_artifacts(out)["manifest_sha256"] == result["manifest_sha256"]


def test_source_bound_support_admission_retains_complete_body_config_and_clock(tmp_path, monkeypatch):
    source = source_package(tmp_path, constructed=True)
    before = files(source)
    stub_stages(monkeypatch)
    out = tmp_path / "workflow"
    result = pipeline.prepare_exercise_learning_pipeline(source, out, train_settings=SETTINGS)
    plan = document(out / "plan.json")
    admission = plan["preserved_body_source_admission"]
    assert admission["support_config_sha256"] == digest(source / "inputs/support-config.json")
    assert admission["body_manifest_sha256"] == digest(source / "inputs/body-source-manifest.json")
    assert "inventory_sha256" not in admission and "inventory.json" not in before
    for directory in ("training", "evaluation", "lab/run"):
        for name in ("scene.json", "exercise-task.json", "source-clock.json", "inputs/support-config.json"):
            assert (out / directory / name).read_bytes() == before[name]
    assert result["selection"]["performed"] is False and files(source) == before


@pytest.mark.parametrize("change", ["config", "parent_manifest", "report"])
def test_resealed_support_construction_conflicts_cannot_enter_training(tmp_path, monkeypatch, change):
    source = source_package(tmp_path, constructed=True)
    if change == "config":
        path = source / "inputs/support-config.json"
        value = document(path)
        value["support_requirements"][0]["minimum_load_body_weight_fraction"] = .1
        write_json(path, value)
        report = document(source / "reconstruction.json")
        report["config_sha256"] = digest(path)
        write_json(source / "reconstruction.json", report)
    elif change == "parent_manifest":
        path = source / "inputs/body-source-manifest.json"
        value = document(path)
        value["files"]["human.json"] = "0" * 64
        write_json(path, value)
    else:
        path = source / "reconstruction.json"
        value = document(path)
        value["simulation_ready"] = True
        write_json(path, value)
    seal(source, profile="shared-world-v4", simulation_ready=False)
    calls = stub_stages(monkeypatch)
    with pytest.raises(ValueError, match="construction provenance"):
        pipeline.prepare_exercise_learning_pipeline(source, tmp_path / "workflow", train_settings=SETTINGS)
    assert calls == [] and not (tmp_path / "workflow").exists()


@pytest.mark.parametrize("change", ["feedback", "missing_task", "external_task"])
def test_unsupported_support_options_fail_before_admission_output_or_native_work(tmp_path, monkeypatch, change):
    source = source_package(tmp_path)
    settings, options = dict(SETTINGS), {}
    if change == "feedback":
        settings["policy_family"] = "feedback-residual"
    elif change == "missing_task":
        (source / "exercise-task.json").unlink()
        seal(source)
    else:
        path = tmp_path / "replacement-task.json"
        write_json(path, {"different": "task"})
        options["task_path"] = path
    monkeypatch.setattr(pipeline.residual, "_source", lambda *_a, **_k: pytest.fail("Invalid support route reached admission"))
    with pytest.raises(ValueError):
        pipeline.prepare_exercise_learning_pipeline(source, tmp_path / "workflow", train_settings=settings, **options)
    assert not (tmp_path / "workflow").exists()


def test_support_evaluation_failure_preserves_frozen_training_without_dummy_selection(tmp_path, monkeypatch):
    source = source_package(tmp_path)
    calls = stub_stages(monkeypatch)

    def failed(source, out, **_options):
        calls.append(("independent_evaluation", source, out, {}))
        out.mkdir()
        write_json(out / "failure.json", {"status": "failed", "scope": "Injected failure; no native trial"})
        raise RuntimeError("Retained independent comparison failure")

    monkeypatch.setattr(pipeline.residual, "evaluate_shared_exercise_residual_policy", failed)
    out = tmp_path / "workflow"
    with pytest.raises(RuntimeError, match="Retained independent"):
        pipeline.prepare_exercise_learning_pipeline(source, out, train_settings=SETTINGS)
    result = document(out / "result.json")
    assert result["profile"] == pipeline.SUPPORT_PROFILE and result["status"] == "failed" and result["lab_status"] == "not_ready"
    assert result["selection"]["performed"] is False and result["selection"]["directory"] == str(out / "training")
    assert (out / "training/manifest.json").is_file() and (out / "evaluation/failure.json").is_file()
    assert not (out / "selected").exists() and not (out / "lab").exists()
    assert [call[0] for call in calls] == ["training", "independent_evaluation"]
    verify_artifacts(out)


def test_support_policy_mutation_stops_before_evaluation(tmp_path, monkeypatch):
    source = source_package(tmp_path)
    calls = stub_stages(monkeypatch)
    out = tmp_path / "workflow"

    def tamper(event):
        if event["stage"] == "training_selection":
            value = document(out / "training/manifest.json")
            value["changed"] = True
            write_json(out / "training/manifest.json", value)

    with pytest.raises(ValueError, match="SHA256"):
        pipeline.prepare_exercise_learning_pipeline(source, out, train_settings=SETTINGS, progress=tamper)
    assert [call[0] for call in calls] == ["training"]
    assert document(out / "result.json")["status"] == "failed"


@pytest.mark.parametrize("family", ["phase-residual", "feedback-residual"])
def test_existing_person_workflow_keeps_its_selected_stage_and_profile(tmp_path, monkeypatch, family):
    from test_exercise_learning_pipeline import _stub_stages
    from weft import shared_exercise_feedback_learning as feedback
    from weft.person_inventory import make_person_inventory
    from weft.shared_exercise_task import build_shared_exercise_task
    from weft.shared_world import validate_shared_world

    scene, _ = static_support_fixture()
    scene.pop("action_contract")
    scene.update(profile="shared-world-v3", objects=[], inventory=make_person_inventory(None, analytic=True))
    scene["task"]["type"] = "human_motion_v1"
    scene = validate_shared_world(scene)
    source = tmp_path / "source"
    source.mkdir()
    write_json(source / "scene.json", scene)
    write_json(source / "exercise-task.json", build_shared_exercise_task(scene))
    write_json(source / "inventory.json", scene["inventory"])
    seal(source)
    calls = _stub_stages(monkeypatch)
    if family == "feedback-residual":
        monkeypatch.setattr(feedback, "train_shared_exercise_feedback_policy", pipeline.residual.train_shared_exercise_residual_policy)
        monkeypatch.setattr(feedback, "select_shared_exercise_feedback_policy", pipeline.select_feasible_shared_exercise_policy)
        monkeypatch.setattr(feedback, "evaluate_shared_exercise_feedback_policy", pipeline.residual.evaluate_shared_exercise_residual_policy)
    out = tmp_path / "workflow"
    result = pipeline.prepare_exercise_learning_pipeline(source, out, train_settings={**SETTINGS, "policy_family": family})
    assert calls == ["training", "training_selection", "independent_evaluation", "interactive_lab"]
    assert result["profile"] == pipeline.PROFILE and result["selection"]["directory"] == str(out / "selected")
    assert (out / "selected/manifest.json").is_file() and "performed" not in result["selection"]
    assert document(out / "plan.json")["selection_rule"] == pipeline.RULE
