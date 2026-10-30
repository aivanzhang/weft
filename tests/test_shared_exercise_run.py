"""A sealed CLI run preserves sources, task versions, baseline failures and exact replays."""

import json
import sys

import pytest

from weft.io import digest, write_json
from weft.pipeline import verify_artifacts
from weft.shared_exercise_run import evaluate_shared_exercise_run
from weft.shared_exercise_task import build_shared_exercise_task
from weft.shared_world_lab import authored_shared_world_scene


def source_fixture(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    scene = authored_shared_world_scene()
    scene["task"]["horizon_steps"] = 3
    write_json(source / "scene.json", scene)
    write_json(source / "manifest.json", {"schema_version": 1, "files": {"scene.json": digest(source / "scene.json")}})
    return source, scene


def test_sealed_evaluation_records_both_baselines_and_replays_exactly(tmp_path):
    source, scene = source_fixture(tmp_path)
    before = {name: digest(source / name) for name in ("scene.json", "manifest.json")}
    report = evaluate_shared_exercise_run(source, tmp_path / "evaluation", root_assistance=True)
    assert report["status"] == "evaluated" and report["trained_policy"] is False
    assert [row["baseline"] for row in report["results"]] == ["coast", "authored_human_reference"]
    for row in report["results"]:
        assert row["metrics"]["observed_substeps"] == 3 * scene["substeps"]
        assert row["recording"]["events"] == row["recording"]["video_frames"] == 4
        assert row["action_replay"]["exact_state_replay"]
        assert row["metrics"]["virtual_object_forces"] == "forbidden"
    out = tmp_path / "evaluation"
    settings = json.loads((out / "settings.json").read_text())
    assert settings["maximum_recorded_events"] == 8
    assert settings["runtime"]["mujoco"]
    assert (out / "implementation/shared_world_observation.py").is_file()
    assert (out / "implementation/assets/shared-world-replay.html").is_file()
    assert verify_artifacts(out)["manifest_sha256"] == report["manifest_sha256"]
    assert before == {name: digest(source / name) for name in before}


@pytest.mark.parametrize("kwargs", [{"seed": True}, {"root_assistance": 1}, {"jitter": 1},
                                     {"max_seconds": float("nan")}, {"max_seconds": 181}])
def test_invalid_bounds_do_not_create_output(tmp_path, kwargs):
    source, _ = source_fixture(tmp_path)
    with pytest.raises(ValueError):
        evaluate_shared_exercise_run(source, tmp_path / "unused", **kwargs)
    assert not (tmp_path / "unused").exists()


def test_sealed_input_and_frozen_task_cannot_be_overridden(tmp_path):
    source, scene = source_fixture(tmp_path)
    task = build_shared_exercise_task(scene)
    task["gates_sha256"] = "0" * 64
    task_path = tmp_path / "task.json"
    write_json(task_path, task)
    with pytest.raises(ValueError, match="gate"):
        evaluate_shared_exercise_run(source, tmp_path / "bad_task", task_path=task_path)
    assert not (tmp_path / "bad_task").exists()
    with pytest.raises(ValueError, match="outside"):
        evaluate_shared_exercise_run(source, source / "nested")
    with pytest.raises(ValueError, match="override"):
        evaluate_shared_exercise_run(source, tmp_path / "override", task_path=task_path, root_assistance=True)
    (source / "scene.json").write_text('{}')
    with pytest.raises(ValueError, match="SHA256"):
        evaluate_shared_exercise_run(source, tmp_path / "tampered")
    assert not (tmp_path / "tampered").exists()


def test_export_failure_preserves_closed_journal_and_seals_failure(tmp_path, monkeypatch):
    source, _ = source_fixture(tmp_path)

    def reject(_, **kwargs):
        raise RuntimeError("injected encoder failure")

    monkeypatch.setattr("weft.shared_exercise_run.export_recording", reject)
    out = tmp_path / "failed"
    with pytest.raises(RuntimeError, match="injected encoder failure"):
        evaluate_shared_exercise_run(source, out, root_assistance=True)
    failure = json.loads((out / "failure.json").read_text())
    assert failure["preserved"]
    assert json.loads((out / "recordings/coast/recording.json").read_text())["status"] == "closed"
    assert (out / "recordings/coast/events.jsonl").read_text().count('\n') == 4
    assert json.loads((out / "manifest.json").read_text())["status"] == "failed"
    verify_artifacts(out)
    with pytest.raises(FileExistsError):
        evaluate_shared_exercise_run(source, out)


def test_cli_exposes_explicit_assistance_choice(tmp_path, monkeypatch, capsys):
    from weft.cli import main
    calls = []

    def evaluate(run, out, **kwargs):
        calls.append((run, out, kwargs))
        return {"status": "evaluated"}

    monkeypatch.setattr("weft.shared_exercise_run.evaluate_shared_exercise_run", evaluate)
    monkeypatch.setattr(sys, "argv", ["weft", "shared-exercise-evaluate", "--run", "source", "--out", "output",
                                    "--root-assistance", "--jitter", "--seed", "19"])
    main()
    assert json.loads(capsys.readouterr().out)["status"] == "evaluated"
    assert calls == [("source", "output", {"root_assistance": True, "task_path": None, "seed": 19,
                                           "jitter": True, "max_seconds": 60})]
