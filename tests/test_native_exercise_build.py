import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest

from test_exercise_build_pipeline import fake_learning
from test_exercise_scene_pipeline import _fake_stages
from weft import cli, exercise_scene_inputs as inputs, exercise_scene_pipeline as pipeline
from weft.io import digest, write_json


def native_stages(monkeypatch, out, calls, *, fail_body=False):
    token = SimpleNamespace(receipt=lambda: {"pose_recovery": "selective-crop"})

    def preflight(*args, **options):
        assert not out.exists()
        calls.append(("preflight", options))
        return token

    def body(source, target, **options):
        assert source is token
        assert options["pose_proposals_out"] == out / "pose_proposals"
        assert options["recovered_pose_out"] == out / "recovered_poses"
        assert 239 < options["max_seconds"] <= 240
        calls.append(("body", options))
        def complete(path, profile, status):
            files = {p.name: digest(p) for p in path.iterdir() if p.is_file()}
            write_json(path / "manifest.json", {"schema_version": 1, "profile": profile,
                "status": status, "simulation_ready": False, "files": files})
            return digest(path / "manifest.json")

        for name, key, profile in (("pose_proposals", "pose_proposals_out", "exercise-pose-crop-proposals-v1"),
                                   ("pose_recovery", "recovered_pose_out", "exercise-pose-recovery-v1")):
            options["progress"](name, {"status": "running", "message": name})
            options[key].mkdir()
            write_json(options[key] / "evidence.json", {"source_only": True})
            seal = complete(options[key], profile, "source_evidence_only")
            options["progress"](name, {"status": "ready", "manifest_sha256": seal})
        if fail_body:
            raise ValueError("Competing body hypotheses remain unsupported")
        target.mkdir()
        write_json(target / "body-source.json", {"simulation_ready": False})
        seal = complete(target, "exercise-body-source-v1", "construction_only")
        options["progress"]("body_source", {"status": "ready", "manifest_sha256": seal})
        return target

    monkeypatch.setattr(inputs, "preflight_exercise_scene_inputs", preflight)
    monkeypatch.setattr(inputs, "prepare_exercise_scene_body", body)
    monkeypatch.setattr(inputs, "recheck_exercise_scene_inputs", lambda *a, **k: None)


@pytest.mark.parametrize("train", [False, True])
def test_native_recovery_uses_one_body_world_and_final_lab_with_explicit_budgets(tmp_path, monkeypatch, train):
    out = tmp_path / "build"
    calls = _fake_stages(monkeypatch)
    native_stages(monkeypatch, out, calls)
    if train:
        fake_learning(monkeypatch, calls)
    result = pipeline.prepare_exercise_scene(None, "equipment", "config", out, video="input.webm",
        pose_model="full.task", pose_recovery="selective-crop", full_reference=True,
        train_settings={"evaluation_seed": 51021} if train else None)
    assert result["status"] == "ready"
    by_name = dict(calls)
    assert by_name["preflight"]["pose_recovery"] == "selective-crop"
    assert by_name["shared_world"]["full_reference"] is True
    assert [name for name, _ in calls].count("body_seed") == 1
    assert result["lab_directory"] == str(out / ("learning_workflow/lab" if train else "lab"))
    assert ("interactive_lab" in by_name) is not train
    settings = json.loads((out / "settings.json").read_text())
    assert settings["maximum_seconds"] == (600 if train else 450)
    assert settings["maximum_bytes"] == (1408 if train else 768) * 1024 * 1024
    assert settings["maximum_files"] == (5632 if train else 3136)
    assert settings["stage_output_reservations"]["body_source"] == {"files": 1536, "bytes": 384 * 1024 * 1024}
    assert settings["pose_recovery"] == "selective-crop"
    stages = [(row["stage"], row["status"]) for row in result["stages"]]
    assert ("source_pose_proposals", "ready") in stages and ("source_pose_recovery", "ready") in stages
    manifest = json.loads((out / "manifest.json").read_text())["files"]
    assert "pose_proposals/manifest.json" in manifest and "recovered_poses/manifest.json" in manifest
    assert all(digest(out / name) == expected for name, expected in manifest.items())


def test_recovery_keeps_completed_pose_packages_when_body_quality_stops_build(tmp_path, monkeypatch):
    out = tmp_path / "build"
    calls = _fake_stages(monkeypatch)
    native_stages(monkeypatch, out, calls, fail_body=True)
    with pytest.raises(ValueError, match="Competing body"):
        pipeline.prepare_exercise_scene(None, "equipment", "config", out, video="input.webm",
            pose_model="full.task", pose_recovery="selective-crop")
    result = json.loads((out / "result.json").read_text())
    assert result["status"] == "failed" and result["lab_status"] == "not_ready"
    assert "equipment_priors" not in dict(calls)
    manifest = json.loads((out / "manifest.json").read_text())["files"]
    assert "pose_proposals/manifest.json" in manifest and "recovered_poses/manifest.json" in manifest
    assert not (out / "body_seed").exists() and not (out / "lab").exists()


def test_native_recovery_reserves_its_whole_compound_stage_before_inference(tmp_path, monkeypatch):
    out = tmp_path / "build"
    calls = _fake_stages(monkeypatch)
    native_stages(monkeypatch, out, calls)
    original = pipeline._inventory

    def inventory(root, deadline, *, immutable_only, limits=None):
        if not immutable_only:
            return {}, 768 * 1024 * 1024
        return original(root, deadline, immutable_only=immutable_only, limits=limits)

    monkeypatch.setattr(pipeline, "_inventory", inventory)
    with pytest.raises(ValueError, match="reserve.*body_source"):
        pipeline.prepare_exercise_scene(None, "equipment", "config", out, video="input.webm",
            pose_model="full.task", pose_recovery="selective-crop")
    assert "body" not in dict(calls) and not (out / "pose_proposals").exists()


@pytest.mark.parametrize("mutation", ["reseal", "member_only", "failure_marker", "extra_file"])
def test_recovery_custody_survives_handoff_to_world_stages(tmp_path, monkeypatch, mutation):
    out = tmp_path / "build"
    calls = _fake_stages(monkeypatch)
    native_stages(monkeypatch, out, calls)

    def progress(row):
        if row["stage"] == "body_source" and row["status"] == "ready":
            root = out / "pose_proposals"
            if mutation in {"reseal", "member_only"}:
                write_json(root / "evidence.json", {"source_only": False})
                if mutation == "reseal":
                    manifest = json.loads((root / "manifest.json").read_text())
                    manifest["files"]["evidence.json"] = digest(root / "evidence.json")
                    write_json(root / "manifest.json", manifest)
            else:
                write_json(root / ("failure.json" if mutation == "failure_marker" else "extra.json"), {"status": "failed"})

    with pytest.raises(ValueError):
        pipeline.prepare_exercise_scene(None, "equipment", "config", out, video="input.webm",
            pose_model="full.task", pose_recovery="selective-crop", progress=progress)
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["seal_scope"] == "failure_receipt_only"
    assert set(manifest["files"]) == {"result.json"}
    if mutation in {"reseal", "failure_marker"}:
        assert "body_seed" not in dict(calls)


def test_progress_cannot_change_retained_child_identity(tmp_path, monkeypatch):
    out = tmp_path / "build"
    calls = _fake_stages(monkeypatch)
    native_stages(monkeypatch, out, calls)

    def progress(row):
        if row["stage"] == "source_pose_proposals" and row["status"] == "ready":
            row["detail"].update(status="failed", manifest_sha256="a" * 64)

    result = pipeline.prepare_exercise_scene(None, "equipment", "config", out, video="input.webm",
        pose_model="full.task", pose_recovery="selective-crop", progress=progress)
    row = next(row for row in result["stages"] if row["stage"] == "source_pose_proposals" and row["status"] == "ready")
    assert row["detail"]["status"] == "ready"
    assert row["detail"]["manifest_sha256"] == digest(out / "pose_proposals/manifest.json")


@pytest.mark.parametrize("options", [
    {"human_run": "body", "pose_recovery": "selective-crop"},
    {"video": "video", "pose_model": "full", "pose_recovery": "automatic"},
    {"video": "video", "pose_model": "full", "pose_recovery": "selective-crop", "max_seconds": 661},
    {"video": "video", "pose_model": "full", "max_seconds": 481},
])
def test_recovery_never_enlarges_old_modes_or_accepts_implicit_recipe(tmp_path, monkeypatch, options):
    monkeypatch.setattr(inputs, "preflight_exercise_scene_inputs", lambda *a, **k: pytest.fail("Preflight should not run"))
    with pytest.raises(ValueError):
        pipeline.prepare_exercise_scene(None, "equipment", "config", tmp_path / "build", **options)
    assert not (tmp_path / "build").exists()


def test_native_proposal_cli_remains_source_only_and_forwards_progress(monkeypatch, capsys):
    module = ModuleType("weft.pose_crop_pipeline")
    received = {}

    def prepare(video, out, **options):
        received.update(video=video, out=out, **{k: v for k, v in options.items() if k != "progress"})
        options["progress"]("crops", {"message": "Observe selected image crops"})
        return Path(out)

    module.prepare_pose_crop_proposals = prepare
    monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setattr(sys, "argv", ["weft", "exercise-pose-prepare", "--video", "new.webm", "--pose-model", "full.task",
        "--source-note", "Attributed new clip", "--max-seconds", "120", "--out", "proposals"])
    cli.main()
    output = capsys.readouterr()
    result = json.loads(output.out)
    assert received == {"video": "new.webm", "out": "proposals", "pose_model": "full.task",
                        "source_note": "Attributed new clip", "max_seconds": 120}
    assert result["profile"] == "exercise-pose-crop-proposals-v1" and result["simulation_ready"] is False
    assert "Observe selected image crops" in output.err


def test_shared_build_cli_carries_explicit_recipe_without_starting_another_lab(monkeypatch, capsys):
    received = {}

    def build(*args, **options):
        received.update(options)
        return {"status": "ready", "lab_status": "ready"}

    monkeypatch.setattr(pipeline, "prepare_exercise_scene", build)
    monkeypatch.setattr(sys, "argv", ["weft", "shared-exercise-build", "--video", "new.webm", "--pose-model", "full.task",
        "--pose-recovery", "selective-crop", "--evidence-run", "equipment", "--config", "config", "--out", "build"])
    cli.main()
    assert received["pose_recovery"] == "selective-crop"
    assert received["pose_mode"] is None and received["pose_identity"] is None
    assert json.loads(capsys.readouterr().out)["status"] == "ready"
