import json
from types import SimpleNamespace

import pytest

from test_exercise_scene_pipeline import _fake_stages
from weft import exercise_learning_pipeline as learning
from weft import exercise_scene_inputs as raw_input
from weft import exercise_scene_pipeline as pipeline
from weft.io import digest, write_json


def fake_learning(monkeypatch, calls, *, late_failure=False):
    def prepare(source, out, **options):
        calls.append(("learning_workflow", {"source": source, **options}))
        (out / "lab/run").mkdir(parents=True)
        (out / "training").mkdir()
        (out / "training/policy.json").write_text('{"frozen":true}')
        (out / "lab/run/scene.json").write_text('{}')
        write_json(out / "lab/status.json", {"status": "ready"})
        result = {"status": "ready", "lab_status": "ready", "lab_directory": str(out / "lab"),
                  "scene_sha256": "4" * 64, "task_sha256": "5" * 64, "manifest_sha256": "6" * 64,
                  "evaluation": {"policy_successes": 0, "trials": 3}}
        options["progress"]({"stage": "interactive_lab", **result, "status": "ready"})
        if late_failure:
            raise TimeoutError("Learning finalization failed after a completed lab")
        return result

    monkeypatch.setattr(learning, "prepare_exercise_learning_pipeline", prepare)


def test_training_uses_only_its_final_lab_and_excludes_mutable_nested_files(tmp_path, monkeypatch):
    calls = _fake_stages(monkeypatch)
    fake_learning(monkeypatch, calls)
    out = tmp_path / "build"
    result = pipeline.prepare_exercise_scene(tmp_path / "base", "equipment", "config", out,
        train_settings={"evaluation_seed": 51021}, full_reference=True)
    assert result["status"] == "ready" and result["lab_status"] == "ready"
    assert result["lab_directory"] == str(out / "learning_workflow/lab")
    assert result["launch_arguments"] == ["lab", "--resume", result["lab_directory"]]
    assert "interactive_lab" not in dict(calls)
    assert dict(calls)["learning_workflow"]["source"] == out / "candidate"
    assert dict(calls)["shared_world"]["full_reference"] is True
    assert result["learning_workflow"]["evaluation"]["policy_successes"] == 0
    settings = json.loads((out / "settings.json").read_text())
    assert settings["maximum_bytes"] == 1024 * 1024 * 1024
    assert "learning_workflow" in settings["stage_output_reservations"]
    assert "interactive_lab" not in settings["stage_output_reservations"]
    manifest = json.loads((out / "manifest.json").read_text())
    assert "learning_workflow/lab/run/scene.json" in manifest["files"]
    assert "learning_workflow/lab/status.json" not in manifest["files"]
    write_json(out / "learning_workflow/lab/status.json", {"status": "closed"})
    (out / "learning_workflow/lab/sessions").mkdir()
    (out / "learning_workflow/lab/sessions/new.json").write_text('{}')
    assert all(digest(out / name) == expected for name, expected in manifest["files"].items())


def test_child_finalization_failure_retains_completed_lab_identity(tmp_path, monkeypatch):
    calls = _fake_stages(monkeypatch)
    fake_learning(monkeypatch, calls, late_failure=True)
    out = tmp_path / "build"
    with pytest.raises(TimeoutError, match="finalization"):
        pipeline.prepare_exercise_scene(tmp_path / "base", "equipment", "config", out,
                                       train_settings={"evaluation_seed": 51021})
    result = json.loads((out / "result.json").read_text())
    assert result["status"] == "failed" and result["lab_status"] == "ready"
    assert result["lab_directory"] == str(out / "learning_workflow/lab")
    assert result["scene_sha256"] == "4" * 64 and result["task_sha256"] == "5" * 64
    assert "interactive_lab" not in dict(calls)
    assert json.loads((out / "learning_workflow/lab/status.json").read_text())["status"] == "ready"


def test_complete_learning_allowance_is_reserved_before_training(tmp_path, monkeypatch):
    calls = _fake_stages(monkeypatch)
    fake_learning(monkeypatch, calls)
    monkeypatch.setattr(learning, "MAX_BYTES", 1024 * 1024 * 1024)
    out = tmp_path / "build"
    with pytest.raises(ValueError, match="reserve.*learning_workflow"):
        pipeline.prepare_exercise_scene(tmp_path / "base", "equipment", "config", out,
                                       train_settings={"evaluation_seed": 51021})
    assert "learning_workflow" not in dict(calls)
    assert (out / "candidate/scene.json").is_file()
    assert not (out / "learning_workflow").exists()


def test_raw_preflight_rejects_before_any_output_or_inference(tmp_path, monkeypatch):
    def reject(*args, **options):
        assert options["pose_model"] == "pose.task"
        raise ValueError("Video does not match equipment source")

    monkeypatch.setattr(raw_input, "preflight_exercise_scene_inputs", reject)
    monkeypatch.setattr(raw_input, "prepare_exercise_scene_body", lambda *a, **k: pytest.fail("Inference began"))
    with pytest.raises(ValueError, match="does not match"):
        pipeline.prepare_exercise_scene(None, "equipment", "config", tmp_path / "build",
                                       video="video.mp4", pose_model="pose.task")
    assert not (tmp_path / "build").exists()


def test_raw_body_flows_into_same_seed_world_and_single_trained_lab(tmp_path, monkeypatch):
    calls = _fake_stages(monkeypatch)
    fake_learning(monkeypatch, calls)
    token = SimpleNamespace(receipt=lambda: {"profile": "checked-input", "source_sha256": "a" * 64})
    out = tmp_path / "build"

    def preflight(*args, **options):
        assert not out.exists()
        calls.append(("raw_preflight", options))
        return token

    def body(inputs, destination, **options):
        assert inputs is token
        assert (out / "input-preflight.json").is_file()
        destination.mkdir()
        (destination / "body-source.json").write_text('{"simulation_ready":false}')
        options["progress"]("tracking", {"sampled_frames": 1})
        calls.append(("raw_body", {"destination": destination, **options}))
        return destination

    monkeypatch.setattr(raw_input, "preflight_exercise_scene_inputs", preflight)
    monkeypatch.setattr(raw_input, "prepare_exercise_scene_body", body)
    monkeypatch.setattr(raw_input, "recheck_exercise_scene_inputs", lambda inputs, **options: None)
    result = pipeline.prepare_exercise_scene(None, "equipment", "config", out, video="exercise.mp4",
        pose_model="pose.task", source_note="Explicit local source", train_settings={"evaluation_seed": 51021})
    assert [name for name, _ in calls] == ["raw_preflight", "raw_body", "body_seed", "equipment_priors",
                                          "shared_world", "learning_workflow"]
    assert dict(calls)["raw_preflight"]["source_note"] == "Explicit local source"
    assert result["lab_directory"] == str(out / "learning_workflow/lab")
    assert not (out / "lab").exists()
    settings = json.loads((out / "settings.json").read_text())
    assert settings["maximum_seconds"] == 450
    assert settings["maximum_bytes"] == 1152 * 1024 * 1024
    assert "body_source" in settings["stage_output_reservations"]


def test_invalid_training_settings_reject_before_raw_preflight(tmp_path, monkeypatch):
    monkeypatch.setattr(raw_input, "preflight_exercise_scene_inputs", lambda *a, **k: pytest.fail("Source work began"))
    with pytest.raises(ValueError, match="evaluation_seed"):
        pipeline.prepare_exercise_scene(None, "equipment", "config", tmp_path / "build",
                                       video="exercise.mp4", pose_model="pose.task", train_settings={})
    assert not (tmp_path / "build").exists()


@pytest.mark.parametrize("change_at", ["body_source", "learning_workflow"])
def test_raw_input_change_after_preparation_cannot_publish_ready(tmp_path, monkeypatch, change_at):
    calls = _fake_stages(monkeypatch)
    fake_learning(monkeypatch, calls)
    token = SimpleNamespace(receipt=lambda: {"source_sha256": "a" * 64})
    changed = [False]
    out = tmp_path / "build"

    def body(inputs, destination, **options):
        destination.mkdir()
        (destination / "body-source.json").write_text('{"simulation_ready":false}')
        return destination

    def recheck(inputs, **options):
        assert inputs is token
        if changed[0]:
            raise ValueError("Original preflight input changed")

    def progress(row):
        if row["stage"] == change_at and row["status"] == "ready":
            changed[0] = True

    monkeypatch.setattr(raw_input, "preflight_exercise_scene_inputs", lambda *a, **k: token)
    monkeypatch.setattr(raw_input, "prepare_exercise_scene_body", body)
    monkeypatch.setattr(raw_input, "recheck_exercise_scene_inputs", recheck)
    with pytest.raises(ValueError, match="preflight input changed"):
        pipeline.prepare_exercise_scene(None, "equipment", "config", out,
            video="exercise.mp4", pose_model="pose.task", train_settings={"evaluation_seed": 51021}, progress=progress)
    result = json.loads((out / "result.json").read_text())
    assert result["status"] == "failed"
    assert (out / "body_source/body-source.json").exists()
    if change_at == "body_source":
        assert "equipment_priors" not in dict(calls)
        assert result["lab_status"] == "not_ready"
    else:
        assert result["lab_status"] == "ready"
        assert result["lab_directory"] == str(out / "learning_workflow/lab")


def test_real_assembly_training_and_recording_share_one_task_and_world(tmp_path):
    from test_equipment_proposal import packages
    from weft.lab import Lab
    from weft.recording import load_recording

    body, equipment, config, original, _ = packages(tmp_path)
    original_files = {path: digest(path) for root in (body, equipment) for path in root.rglob("*") if path.is_file()}
    out = tmp_path / "trained-build"
    result = pipeline.prepare_exercise_scene(body, equipment, config, out, root_assistance=True,
        full_reference=True, train_settings={"training_seed": 51603, "evaluation_seed": 51604,
                                             "generations": 1, "population": 4, "trials": 1}, max_seconds=90)
    assert result["status"] == "ready"
    assert not (out / "lab").exists()
    workflow = out / "learning_workflow"
    lab_root = workflow / "lab"
    assert result["lab_directory"] == str(lab_root)
    task_bytes = (workflow / "exercise-task.json").read_bytes()
    for relative in ("training", "selected", "evaluation", "lab/run"):
        assert (workflow / relative / "exercise-task.json").read_bytes() == task_bytes
        scene = json.loads((workflow / relative / "scene.json").read_bytes())
        assert scene["human_spec"] == original["human_spec"]
        assert {obj["id"] for obj in scene["objects"]} == {"foam-pad", "training-weight"}
    plan = json.loads((workflow / "plan.json").read_bytes())
    assert plan["horizon_steps"] == 60
    assert plan["maximum_training_steps"] == 480 and plan["maximum_evaluation_steps"] == 240
    assert not set(plan["training_seeds"]) & set(plan["evaluation_seeds"])
    assert json.loads((workflow / "selected/selection.json").read_bytes())["evaluation_used"] is False
    manifest = json.loads((out / "manifest.json").read_bytes())
    lab = Lab(lab_root)
    try:
        assert lab.summary()["learned_policy_available"] is True
        lab.control("/api/record/start", {})
        lab.control("/api/step", {"policy": True, "steps": 2})
        lab.control("/api/reset", {"seed": 51605})
        lab.control("/api/step", {"action": [.05] + [0.] * 21})
        stopped = lab.control("/api/record/stop", {})
        exported = lab.export_future.result(timeout=30)
        assert exported["action_replay"]["exact_state_replay"] is True
        assert exported["action_replay"]["learned_actions_verified"] == 2
        saved = load_recording(lab_root / "sessions" / stopped["directory"])
        steps = [row for row in saved["events"] if row["type"] == "step"]
        assert [row["controller"] for row in steps] == ["learned", "learned", "manual"]
        assert all(not any(row["snapshot"]["engine_state"]["normalized_action"][22:]) for row in steps)
    finally:
        lab.close()
    assert all(digest(path) == expected for path, expected in original_files.items())
    assert all(digest(out / name) == expected for name, expected in manifest["files"].items())
