"""Full-reference tasks stay continuous through learning, inspection and recorded control."""

from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import time

import pytest

from weft.io import canonical_hash, digest, write_json
from weft.lab import Lab
from weft.pipeline import verify_artifacts
from weft.shared_exercise_lab import _reference_frames, load_shared_exercise_lab, prepare_shared_exercise_lab
from weft.shared_exercise_residual_learning import (
    evaluate_shared_exercise_residual_policy, train_shared_exercise_residual_policy,
)
from weft.shared_exercise_task import build_shared_exercise_task
from weft.shared_world import validate_shared_world
from weft.shared_world_lab import authored_shared_world_scene


def fullspan_fixture(root):
    source = root / "source"
    source.mkdir()
    scene = authored_shared_world_scene()
    body = scene["human_spec"]
    for index in range(len(body["reference"]), 255):
        for key in ("reference", "root_reference"):
            frame = deepcopy(body[key][-1])
            frame["time_s"] = index / 30
            body[key].append(frame)
    scene["profile"] = "shared-world-v2"
    scene["task"]["horizon_steps"] = 254
    scene["task"]["start_offset_s"] = 0.
    scene["learned_policy_available"] = False
    scene = validate_shared_world(scene)
    task = build_shared_exercise_task(scene, root_assistance=True)
    write_json(source / "scene.json", scene)
    write_json(source / "exercise-task.json", task)
    write_json(source / "manifest.json", {"schema_version": 1, "status": "ready", "files": {
        path.name: digest(path) for path in source.iterdir()}})
    return source, scene, task


@pytest.fixture(scope="module")
def fullspan_learning(tmp_path_factory):
    root = tmp_path_factory.mktemp("fullspan-integration")
    source, scene, task = fullspan_fixture(root)
    learning, evaluation = root / "learning", root / "evaluation"
    training = train_shared_exercise_residual_policy(
        source, learning, generations=1, population=4, seed=29481)
    result = evaluate_shared_exercise_residual_policy(
        learning, evaluation, seed=29482, trials=3, max_seconds=60)
    return root, source, scene, task, learning, evaluation, training, result


def test_fullspan_evaluation_retains_whole_interval_and_fresh_matched_starts(fullspan_learning):
    _, source, scene, task, learning, evaluation, training, result = fullspan_learning
    assert training["control_steps"] <= 2 * 4 * 254
    assert result["status"] == "evaluated"
    assert result["evaluation_protocol"] == "shared-exercise-full-reference-evaluation-v1"
    assert result["maximum_control_steps"] == 4 * 3 * 254
    assert result["horizon_steps"] == 254
    assert set(result["evaluation_seeds"]).isdisjoint(result["training_seeds"])
    assert result["independent_changed_states"] == 3
    assert result["comparison_complete"] and result["policy_unchanged"]
    for controller in result["controllers"].values():
        assert len(controller["trials"]) == 3
        for trial in controller["trials"]:
            assert trial["metrics"]["declared_duration_s"] == pytest.approx(254 / 30)
            assert trial["metrics"]["completed_interval"] or trial["status"] == "failed"
            assert trial["initial_state_sha256"] == result["initial_states"][str(trial["seed"]) ]
    for name in ("scene.json", "exercise-task.json"):
        assert (learning / name).read_bytes() == (evaluation / name).read_bytes() == (source / name).read_bytes()
    assert result["scene_sha256"] == canonical_hash(scene)
    assert result["task_sha256"] == canonical_hash(task)
    assert verify_artifacts(evaluation)["manifest_sha256"] == result["manifest_sha256"]


def test_fullspan_lab_exposes_all_references_and_records_exact_current_runtime(fullspan_learning):
    root, source, scene, task, learning, evaluation, _, result = fullspan_learning
    target = root / "lab"
    prepare_shared_exercise_lab(source, target, policy_run=learning, evaluation_run=evaluation)
    loaded = load_shared_exercise_lab(target / "run")
    assert loaded["scene"] == scene and loaded["task"] == task
    reference = json.loads((target / "run/human-reference.json").read_text())
    assert len(reference["frames"]) == reference["reference_poses_total"] == 255
    assert reference["frames"][-1]["source_time_s"] == pytest.approx(254 / 30)
    lab = Lab(target)
    try:
        summary = lab.summary()
        assert summary["scene"]["profile"] == "shared-world-v2"
        assert summary["capabilities"]["entities"]["mixed_rigid_articulated"] is True
        assert summary["capabilities"]["task_contract"]["truncation_steps"] == 254
        lab.control("/api/record/start", {})
        lab.control("/api/step", {"policy": True, "steps": 6})
        lab.control("/api/reset", {"seed": 91833})
        lab.control("/api/step", {"controller": "reference", "steps": 6})
        lab.control("/api/record/stop", {})
        exported = lab.export_future.result(timeout=30)
        assert exported["action_replay"]["exact_state_replay"]
        assert exported["action_replay"]["task_sha256"] == canonical_hash(task)
        assert exported["events"] == exported["video_frames"] == 14
    finally:
        lab.close()
    resumed = Lab(target)
    try:
        assert len(resumed.summary()["sessions"]) == 1
    finally:
        resumed.close()


def test_reference_inspection_keeps_v1_late_poses_instead_of_silently_clipping(tmp_path):
    _, scene, _ = fullspan_fixture(tmp_path)
    scene["profile"] = "shared-world-v1"
    scene["task"]["horizon_steps"] = 60
    result = _reference_frames(scene, time.monotonic() + 10)
    assert result["reference_poses_included"] == result["reference_poses_total"] == 255
    assert len(result["frames"]) == 255


def test_v1_evaluation_keeps_its_original_time_bound(tmp_path):
    source, scene, _ = fullspan_fixture(tmp_path)
    scene["profile"] = "shared-world-v1"
    scene["task"]["horizon_steps"] = 3
    write_json(source / "scene.json", scene)
    write_json(source / "exercise-task.json", build_shared_exercise_task(scene, root_assistance=True))
    write_json(source / "manifest.json", {"files": {path.name: digest(path)
        for path in source.iterdir() if path.name != "manifest.json"}})
    learning = tmp_path / "learning"
    train_shared_exercise_residual_policy(source, learning, generations=1, population=4)
    with pytest.raises(ValueError, match="30"):
        evaluate_shared_exercise_residual_policy(learning, tmp_path / "unused", max_seconds=31)
    assert not (tmp_path / "unused").exists()


def test_fullspan_study_freezes_budget_and_selects_only_before_evaluation(tmp_path, monkeypatch):
    path = Path(__file__).resolve().parents[1] / "scripts/studies/check_shared_exercise_residual.py"
    spec = importlib.util.spec_from_file_location("fullspan_study", path)
    study = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(study)
    source, _, _ = fullspan_fixture(tmp_path)
    out, calls = tmp_path / "study", []

    def train(run, target, **options):
        plan = json.loads((out / "pre-training-freeze.json").read_text())
        assert plan["maximum_training_control_steps"] == 8128
        assert plan["maximum_evaluation_control_steps"] == 3048
        assert plan["training_feasible_selection"] is True
        assert options["generations"] == 2 and options["population"] == 8
        calls.append("train")
        return dict(manifest_sha256="a" * 64, policy_sha256="b" * 64, seconds=0., control_steps=1,
                    evaluated_episodes=1, cache_hits=0, selected_feedback_kind="authored_unchanged",
                    improvement_over_authored_training=0., training_score_improved_over_authored=False)

    def select(run, target, **options):
        assert calls == ["train"] and run == out / "training"
        assert target == out / "selected"
        assert not (out / "evaluation").exists()
        calls.append("select")
        return {"status": "trained", "policy_sha256": "c" * 64}

    def evaluate(run, target, **options):
        assert calls == ["train", "select"] and run == out / "selected"
        assert options["max_seconds"] <= 60 and options["trials"] == 3
        plan = json.loads((out / "pre-training-freeze.json").read_text())
        calls.append("evaluate")
        return dict(manifest_sha256="d" * 64, policy_sha256="c" * 64, seconds=0., control_steps=1,
                    comparison_complete=True, independent_changed_states=3, controllers={},
                    evaluation_seeds=plan["evaluation_seeds"], training_seeds=plan["training_seeds"])

    monkeypatch.setattr(study, "train_shared_exercise_residual_policy", train)
    monkeypatch.setattr(study, "select_feasible_shared_exercise_policy", select)
    monkeypatch.setattr(study, "evaluate_shared_exercise_residual_policy", evaluate)
    assert study.run(source, out, select_feasible=True)["status"] == "completed"
    assert calls == ["train", "select", "evaluate"]
