import json
import sys

import pytest

from weft import cli, exercise_scene_pipeline


def arguments(tmp_path, *options):
    return ["weft", "shared-exercise-build", "--evidence-run", "equipment", "--config", "config.json",
            "--out", str(tmp_path / "new"), *options]


def test_raw_video_and_training_share_the_existing_build_command(tmp_path, monkeypatch, capsys):
    received = {}

    def build(base, evidence, config, out, **options):
        received.update(base=base, evidence=evidence, config=config, out=out, **options)
        options["progress"]({"stage": "learning_independent_evaluation", "status": "running"})
        return {"status": "ready", "lab_directory": "new/learning_workflow/lab"}

    monkeypatch.setattr(exercise_scene_pipeline, "prepare_exercise_scene", build)
    monkeypatch.setattr(sys, "argv", arguments(tmp_path, "--video", "exercise.mp4", "--pose-model", "pose.task",
        "--pose-mode", "image", "--pose-identity", "temporal", "--source-note", "Attributed local example",
        "--full-reference", "--body-scale", "1.2", "--train", "--training-seed", "51",
        "--evaluation-seed", "61", "--training-generations", "1", "--training-population", "4",
        "--evaluation-trials", "2", "--evaluation-jitter", "0.08", "--max-seconds", "320"))
    cli.main()
    output = capsys.readouterr()
    assert json.loads(output.out)["status"] == "ready"
    assert "Compare frozen controller on separate starts: running" in output.err
    assert received["base"] is None and received["human_run"] is None
    assert (received["video"], received["pose_model"]) == ("exercise.mp4", "pose.task")
    assert received["pose_mode"] == "image" and received["pose_identity"] == "temporal"
    assert received["source_note"] == "Attributed local example"
    assert received["body_scale"] == 1.2 and received["full_reference"] is True
    assert received["train_settings"] == {"training_seed": 51, "evaluation_seed": 61, "generations": 1,
                                           "population": 4, "trials": 2, "jitter_scale": .08}
    assert received["max_seconds"] == 320


@pytest.mark.parametrize("options", [
    ("--train",), ("--evaluation-seed", "61"), ("--training-population", "4"),
    ("--training-seed", "51"), ("--evaluation-trials", "2"), ("--evaluation-jitter", "0.1"),
])
def test_training_requires_an_explicit_choice_and_evaluation_seed(tmp_path, monkeypatch, capsys, options):
    def forbidden(*args, **kwargs):
        pytest.fail("Invalid training options reached construction")

    monkeypatch.setattr(exercise_scene_pipeline, "prepare_exercise_scene", forbidden)
    monkeypatch.setattr(sys, "argv", arguments(tmp_path, "--human-run", "body", *options))
    with pytest.raises(SystemExit):
        cli.main()
    output = capsys.readouterr()
    assert "require" in output.err
    assert not (tmp_path / "new").exists()


def test_saved_build_does_not_enable_inference_or_training(tmp_path, monkeypatch, capsys):
    received = {}

    def build(*args, **options):
        received.update(options)
        return {"status": "ready"}

    monkeypatch.setattr(exercise_scene_pipeline, "prepare_exercise_scene", build)
    monkeypatch.setattr(sys, "argv", arguments(tmp_path, "--human-run", "body"))
    cli.main()
    assert json.loads(capsys.readouterr().out)["status"] == "ready"
    assert all(received[key] is None for key in ("video", "pose_model", "pose_mode", "pose_identity",
                                                "source_note", "train_settings"))


@pytest.mark.parametrize("options", [{"pose_model": "pose.task"}, {"pose_mode": "video"},
                                     {"pose_identity": "raw"}, {"source_note": "Note"}])
def test_saved_body_cannot_be_reinferred_by_raw_options(tmp_path, options):
    with pytest.raises(ValueError, match="only to raw"):
        exercise_scene_pipeline.prepare_exercise_scene(None, "missing", "missing", tmp_path / "new",
                                                       human_run="body", **options)
    assert not (tmp_path / "new").exists()


def test_raw_video_requires_a_model_before_any_output(tmp_path):
    with pytest.raises(ValueError, match="explicit local pose model"):
        exercise_scene_pipeline.prepare_exercise_scene(None, "missing", "missing", tmp_path / "new", video="video")
    assert not (tmp_path / "new").exists()


@pytest.mark.parametrize("status", ["rejected_initial_geometry", "unsupported", "failed"])
def test_nonready_build_exits_nonzero_after_printing_preserved_result(tmp_path, monkeypatch, capsys, status):
    result = {"status": status, "lab_status": "not_ready", "lab_directory": None}
    monkeypatch.setattr(exercise_scene_pipeline, "prepare_exercise_scene", lambda *a, **k: result)
    monkeypatch.setattr(sys, "argv", arguments(tmp_path, "--human-run", "body"))
    with pytest.raises(SystemExit) as error:
        cli.main()
    output = capsys.readouterr()
    assert error.value.code == 1
    assert json.loads(output.out) == result
    assert status in output.err and str(tmp_path / "new" / "result.json") in output.err


def test_ready_lab_with_failed_task_is_still_a_completed_build(tmp_path, monkeypatch, capsys):
    result = {"status": "ready", "lab_status": "ready", "task_successes": 0, "evaluation_trials": 3}
    monkeypatch.setattr(exercise_scene_pipeline, "prepare_exercise_scene", lambda *a, **k: result)
    monkeypatch.setattr(sys, "argv", arguments(tmp_path, "--human-run", "body"))
    cli.main()
    assert json.loads(capsys.readouterr().out) == result
