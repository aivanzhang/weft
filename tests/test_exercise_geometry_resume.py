"""Public routing with the existing native admission boundary replaced by a recorder."""

import json
from pathlib import Path
import sys
import time

import pytest

from test_exercise_geometry_inputs import Packet, digest, encoded
from weft import exercise_scene_pipeline as pipeline
from weft.exercise_build_report import prepare_exercise_build_report
from weft.io import canonical_hash


@pytest.fixture
def packet(tmp_path):
    return Packet(tmp_path / "study")


def fake_lab(monkeypatch, packet, *, failure=None, altered=None):
    calls = []

    def unexpected(*args, **kwargs):
        raise AssertionError("Geometry reuse must not propose, reconstruct or fit")

    def lab(source, out, **kwargs):
        assert source == packet.root / "geometry"
        assert set(kwargs) == {"max_seconds"} and 1 <= kwargs["max_seconds"] <= 60
        calls.append((source, out, kwargs))
        if failure:
            raise ValueError(failure)
        members = {name: (source / name).read_bytes() for name in ("scene.json", "exercise-task.json")}
        members["source-run/manifest.json"] = (source / "manifest.json").read_bytes()
        if altered == "raw_task":
            members["exercise-task.json"] += b" "
        if altered == "source_manifest":
            members["source-run/manifest.json"] = encoded({"files": {}})
        for name, raw in members.items():
            path = out / "run" / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(raw)
        manifest = encoded({"schema_version": 1, "status": "ready", "files": {name: digest(raw) for name, raw in members.items()}})
        (out / "run/manifest.json").write_bytes(manifest)
        (out / "status.json").write_bytes(encoded({"status": "ready"}))
        return {"lab_directory": str(out), "manifest_sha256": digest(manifest),
            "scene_sha256": canonical_hash(packet.docs["geometry/scene.json"]),
            "task_sha256": "f" * 64 if altered == "returned_task" else canonical_hash(packet.docs["geometry/exercise-task.json"])}

    monkeypatch.setattr(pipeline, "prepare_equipment_proposal", unexpected)
    monkeypatch.setattr(pipeline, "construct_source_equipment_run", unexpected)
    monkeypatch.setattr(pipeline, "prepare_shared_exercise_lab", lab)
    return calls


def build(packet, out, **kwargs):
    return pipeline.prepare_exercise_scene(None, None, None, out, geometry_run=packet.root, **kwargs)


def report(path, out):
    result = prepare_exercise_build_report(path, out)
    return result, json.loads((out / "report.json.txt").read_bytes()), (out / "index.html").read_text()


def test_public_build_routes_exact_selected_candidate_to_normal_lab_and_preserves_alternatives(packet, tmp_path, monkeypatch):
    calls = fake_lab(monkeypatch, packet)
    out = tmp_path / "build"
    manifest_before = (packet.root / "manifest.json").read_bytes()
    result = build(packet, out, geometry_manifest_sha256=digest(manifest_before))
    assert len(calls) == 1 and result["status"] == result["lab_status"] == "ready"
    assert [(row["stage"], row["status"]) for row in result["stages"]] == [
        ("geometry_source", "running"), ("geometry_source", "ready"), ("interactive_lab", "running"), ("interactive_lab", "ready")]
    assert result["launch_arguments"] == ["lab", "--resume", str(out / "lab")]
    assert not (out / "candidate").exists() and not (out / "proposal").exists()
    for name in ("geometry-input/scene.json", "geometry/scene.json", "geometry/exercise-task.json", "geometry-config.json"):
        assert (out / "geometry_source" / name).read_bytes() == (packet.root / name).read_bytes()
    assert (packet.root / "manifest.json").read_bytes() == manifest_before
    settings = json.loads((out / "settings.json").read_bytes())
    assert settings["training"] is False and settings["maximum_files"] == 1600 and settings["maximum_bytes"] == 384 << 20
    generated, summary, html = report(out, tmp_path / "report")
    assert generated["status"] == "report_generated" and summary["runtime_admission"] == "not_checked"
    assert summary["equipment"]["status"] == "saved_conditional_alternative_reused"
    assert summary["equipment"]["source_fit"] == "retained_conditional_study_claims"
    assert [row["stage"] for row in summary["stages"]] == ["geometry_source", "interactive_lab"]
    assert all(row["state"] == "completed" for row in summary["stages"])
    names = {row["source_name"] for row in summary["artifacts"]}
    assert {"geometry_source/geometry-input/scene.json", "geometry_source/geometry/scene.json",
        "geometry_source/geometry/exercise-task.json", "geometry_source/objects/weight/result.json"} <= names
    assert "No equipment candidate is available" not in html
    assert "not newly measured shape or contact" in html
    for artifact in summary["artifacts"]:
        assert (tmp_path / "report" / artifact["path"]).is_file()
    assert not any(summary["work"].values())


def test_lab_rejection_is_preserved_and_reported_without_refit(packet, tmp_path, monkeypatch):
    calls = fake_lab(monkeypatch, packet, failure="ordinary source proof rejected")
    out = tmp_path / "rejected"
    with pytest.raises(ValueError, match="ordinary source proof rejected"):
        build(packet, out)
    saved = json.loads((out / "result.json").read_bytes())
    assert len(calls) == 1 and saved["status"] == "failed" and saved["lab_status"] == "not_ready"
    _, summary, _ = report(out, tmp_path / "report")
    assert summary["build_status"] == "failed" and summary["equipment"]["status"] == "saved_conditional_alternative_reused"
    assert summary["stages"][1]["state"] == "stopped"


@pytest.mark.parametrize("altered", ["returned_task", "raw_task", "source_manifest"])
def test_lab_must_preserve_selected_raw_bytes_and_canonical_identity(packet, tmp_path, monkeypatch, altered):
    calls = fake_lab(monkeypatch, packet, altered=altered)
    out = tmp_path / "altered"
    with pytest.raises(ValueError):
        build(packet, out)
    saved = json.loads((out / "result.json").read_bytes())
    assert len(calls) == 1
    assert saved["status"] == "failed" and saved["lab_status"] == "not_ready"


def test_progress_failure_before_snapshot_is_reportable_but_ready_claim_is_rejected(packet, tmp_path, monkeypatch):
    calls = fake_lab(monkeypatch, packet)
    out = tmp_path / "early"

    def stop(row):
        raise RuntimeError("progress output unavailable")

    with pytest.raises(RuntimeError, match="progress output"):
        build(packet, out, progress=stop)
    assert not calls and not (out / "geometry_source").exists()
    _, summary, html = report(out, tmp_path / "early-report")
    assert summary["equipment"]["status"] == "saved_geometry_snapshot_incomplete"
    assert summary["saved_lab_status"] == "not_ready" and "No lab is ready" in html
    result = json.loads((out / "result.json").read_bytes())
    result["status"] = "ready"
    pipeline._seal(out, result, deadline=time.monotonic() + 5., started=time.monotonic())
    with pytest.raises(ValueError, match="lacks its selected metadata snapshot"):
        report(out, tmp_path / "invalid-report")


def test_post_lab_progress_failure_retains_ready_inner_lab(packet, tmp_path, monkeypatch):
    fake_lab(monkeypatch, packet)
    out = tmp_path / "late"

    def stop(row):
        if row["stage"] == "interactive_lab" and row["status"] == "ready":
            raise RuntimeError("progress output unavailable")

    with pytest.raises(RuntimeError, match="progress output"):
        build(packet, out, progress=stop)
    result = json.loads((out / "result.json").read_bytes())
    assert result["status"] == "failed" and result["lab_status"] == "ready"
    _, summary, _ = report(out, tmp_path / "late-report")
    assert summary["saved_lab_status"] == "ready" and summary["runtime_admission"] == "not_checked"


def test_source_mutation_at_callback_stops_before_lab(packet, tmp_path, monkeypatch):
    calls = fake_lab(monkeypatch, packet)
    out = tmp_path / "changed"

    def mutate(row):
        if row["stage"] == "geometry_source" and row["status"] == "ready":
            path = packet.root / "geometry/scene.json"
            path.write_bytes(path.read_bytes() + b" ")

    with pytest.raises(ValueError, match="changed after qualification"):
        build(packet, out, progress=mutate)
    assert not calls
    result = json.loads((out / "result.json").read_bytes())
    assert result["seal_scope"] == "failure_receipt_only" and result["lab_status"] == "not_ready"


@pytest.mark.parametrize("kwargs", [{"body_scale": 1.}, {"train_settings": {}}, {"full_reference": True},
    {"root_assistance": True}, {"source_mappings": ["mapping"]}, {"scene_scope": "human_only"}])
def test_api_rejects_overrides_before_io(tmp_path, kwargs):
    out = tmp_path / "none"
    with pytest.raises(ValueError, match="overrides are not allowed"):
        pipeline.prepare_exercise_scene(None, None, None, out, geometry_run="missing", **kwargs)
    assert not out.exists()


@pytest.mark.parametrize("flag,value", [("--body-scale", "1"), ("--pose-mode", "video"), ("--pose-identity", "raw"),
    ("--training-seed", "1729"), ("--evaluation-seed", "1729"), ("--training-generations", "2"),
    ("--training-population", "8"), ("--training-policy-family", "phase-residual"), ("--evaluation-trials", "3"),
    ("--evaluation-jitter", "0.1"), ("--evidence-run", "evidence"), ("--config", "config"),
    ("--source-mapping", "mapping"), ("--train", None), ("--root-assistance", None), ("--full-reference", None),
    ("--person-only", None)])
def test_cli_rejects_explicit_overrides_even_default_values(monkeypatch, capsys, flag, value):
    from weft import cli

    def forbidden(*args, **kwargs):
        raise AssertionError("Invalid CLI must stop before build")

    monkeypatch.setattr(pipeline, "prepare_exercise_scene", forbidden)
    monkeypatch.setattr(sys, "argv", ["weft", "shared-exercise-build", "--geometry-run", "study", "--out", "unused",
        flag, *([] if value is None else [value])])
    with pytest.raises(SystemExit) as error:
        cli.main()
    assert error.value.code != 0 and flag in capsys.readouterr().err


@pytest.mark.parametrize("mode", ["geometry", "original", "pin_without_geometry"])
def test_cli_new_route_and_unchanged_original_kwargs(monkeypatch, capsys, mode):
    from weft import cli

    called = []

    def prepare(*args, **kwargs):
        called.append((args, kwargs))
        return {"status": "ready"}

    argv = ["weft", "shared-exercise-build", "--out", "unused"]
    if mode == "geometry":
        argv += ["--geometry-run", "study", "--geometry-manifest-sha256", "a" * 64]
    else:
        argv += ["--run", "base", "--evidence-run", "evidence", "--config", "config"]
        if mode == "pin_without_geometry":
            argv += ["--geometry-manifest-sha256", "a" * 64]
    monkeypatch.setattr(sys, "argv", argv)
    monkeypatch.setattr(pipeline, "prepare_exercise_scene", prepare)
    if mode == "pin_without_geometry":
        with pytest.raises(SystemExit): cli.main()
        assert not called
        return
    cli.main()
    args, kwargs = called[0]
    assert len(called) == 1 and json.loads(capsys.readouterr().out)["status"] == "ready"
    if mode == "geometry":
        assert args[:3] == (None, None, None) and kwargs["geometry_run"] == "study"
        assert kwargs["geometry_manifest_sha256"] == "a" * 64
    else:
        assert args[:3] == ("base", "evidence", "config")
        assert "geometry_run" not in kwargs and "geometry_manifest_sha256" not in kwargs
