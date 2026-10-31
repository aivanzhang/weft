"""Preserved raw-body receipts do not substitute for completed world admission."""

import json
from pathlib import Path
import shutil
import time

import pytest

from test_exercise_body_source import forbid_runtime
from test_exercise_learning_pipeline import _stub_stages
from test_person_only_downstream import source_package
from test_shared_exercise_run import source_fixture
from test_shared_world_human import analytical_body, inventory
from weft import exercise_learning_pipeline as workflow
from weft import shared_world_human as builder
from weft.io import canonical_hash, digest, write_json


SETTINGS = {"generations": 1, "population": 4, "trials": 1, "evaluation_seed": 924017}


@pytest.fixture(autouse=True)
def no_source_or_dynamics(monkeypatch):
    forbid_runtime(monkeypatch)


@pytest.fixture(scope="module")
def completed_candidate(tmp_path_factory, analytical_body):
    out = tmp_path_factory.mktemp("person-learning-admission") / "candidate"
    with pytest.MonkeyPatch.context() as patch:
        forbid_runtime(patch)
        result = builder.construct_human_only_run(analytical_body, inventory(analytical_body), out)
    assert result["status"] == "ready" and result["simulation_ready"] is False
    return out


@pytest.fixture
def candidate(tmp_path, completed_candidate):
    return Path(shutil.copytree(completed_candidate, tmp_path / "candidate"))


def document(root, name):
    return json.loads((root / name).read_bytes())


def reseal(root):
    manifest = document(root, "manifest.json")
    manifest["files"] = {path.relative_to(root).as_posix(): digest(path)
                         for path in root.rglob("*") if path.is_file() and path != root / "manifest.json"}
    write_json(root / "manifest.json", manifest)


def test_source_bound_constructor_candidate_reaches_one_mocked_learning_workflow(candidate, tmp_path, monkeypatch):
    calls = _stub_stages(monkeypatch)
    before = {path.relative_to(candidate).as_posix(): digest(path)
              for path in candidate.rglob("*") if path.is_file()}
    result = workflow.prepare_exercise_learning_pipeline(candidate, tmp_path / "workflow", train_settings=SETTINGS)
    assert result["status"] == "ready" and calls == list(workflow.STAGE_LIMITS)
    plan = document(tmp_path / "workflow", "plan.json")
    admitted = plan["preserved_body_source_admission"]
    scene = document(candidate, "scene.json")
    assert admitted["source_sha256"] == scene["source_binding"]["source_sha256"]
    assert admitted["body_source_sha256"] == before["body-source.json"]
    assert admitted["body_manifest_sha256"] == before["inputs/body-source-manifest.json"]
    assert admitted["inventory_sha256"] == before["inventory.json"]
    assert admitted["human_file_sha256"] == before["human.json"]
    assert plan["scene_sha256"] == canonical_hash(scene)
    assert plan["horizon_steps"] == 58 and plan["maximum_control_steps"] == 696
    assert admitted["identity_documents"].items() <= plan["source_documents"].items()
    for child in ("training", "selected", "evaluation", "lab/run"):
        assert (tmp_path / "workflow" / child / "scene.json").read_bytes() == (candidate / "scene.json").read_bytes()
    assert result["training"]["control_steps"] == result["evaluation"]["control_steps"] == 0
    assert before == {path.relative_to(candidate).as_posix(): digest(path)
                      for path in candidate.rglob("*") if path.is_file()}


def test_raw_body_receipt_still_rejects_before_learning_or_output(analytical_body, tmp_path, monkeypatch):
    calls = _stub_stages(monkeypatch)
    out = tmp_path / "workflow"
    with pytest.raises(ValueError, match="completed untrained"):
        workflow.prepare_exercise_learning_pipeline(analytical_body, out, train_settings=SETTINGS)
    assert calls == [] and not out.exists()


@pytest.mark.parametrize("profile", ["legacy_mixed", "analytic_person"])
def test_receipt_filename_cannot_exempt_legacy_or_unbound_scene(tmp_path, monkeypatch, profile):
    source = source_fixture(tmp_path)[0] if profile == "legacy_mixed" else source_package(tmp_path)[0]
    write_json(source / "body-source.json", {"profile": "exercise-body-source-v1", "status": "construction_only"})
    reseal(source)
    calls = _stub_stages(monkeypatch)
    out = tmp_path / "workflow"
    with pytest.raises(ValueError, match="completed untrained"):
        workflow.prepare_exercise_learning_pipeline(source, out, train_settings=SETTINGS)
    assert calls == [] and not out.exists()


@pytest.mark.parametrize("name,path,value", [
    ("manifest.json", ("profile",), "shared-world-v2"),
    ("manifest.json", ("status",), "trained"),
    ("manifest.json", ("simulation_ready",), True),
    ("reconstruction.json", ("mode",), "other_construction"),
    ("reconstruction.json", ("readiness",), "raw_body"),
    ("reconstruction.json", ("scene_sha256",), "f" * 64),
    ("reconstruction.json", ("source_sha256",), "f" * 64),
    ("reconstruction.json", ("human_file_sha256",), "f" * 64),
    ("reconstruction.json", ("source_manifest_sha256",), "f" * 64),
    ("reconstruction.json", ("source_verification",), None),
    ("reconstruction.json", ("source_verification", "preserved_files", "absent.json"), None),
    ("body-source.json", ("profile",), "other_body"),
    ("body-source.json", ("schema_version",), True),
    ("body-source.json", ("source_sha256",), "f" * 64),
    ("body-source.json", ("source_artifact",), "unrelated.txt"),
    ("source.json", ("artifact",), "source.mov"),
    ("inventory.json", ("absence_verified",), True),
    ("human.json", ("scope",), "Not the preserved scene body"),
    ("inputs/body-source-manifest.json", ("status",), "failed"),
])
def test_resealed_conflicting_receipt_identity_rejects_before_learning(candidate, tmp_path, monkeypatch, name, path, value):
    changed = document(candidate, name)
    nested = changed
    for key in path[:-1]:
        nested = nested[key]
    nested[path[-1]] = value
    write_json(candidate / name, changed)
    if name != "manifest.json":
        reseal(candidate)
    calls = _stub_stages(monkeypatch)
    out = tmp_path / "workflow"
    with pytest.raises(ValueError, match="completed untrained"):
        workflow.prepare_exercise_learning_pipeline(candidate, out, train_settings=SETTINGS)
    assert calls == [] and not out.exists()


@pytest.mark.parametrize("name", ["body-source.json", "inputs/body-source-manifest.json", "inventory.json"])
def test_consumed_receipt_metadata_is_rechecked_after_progress_callback(candidate, tmp_path, monkeypatch, name):
    calls = _stub_stages(monkeypatch)
    def changed(row):
        if row["stage"] == "training" and row["status"] == "running":
            write_json(candidate / name, {"changed_after_admission": True})
    out = tmp_path / "workflow"
    with pytest.raises(ValueError, match="SHA256"):
        workflow.prepare_exercise_learning_pipeline(candidate, out, train_settings=SETTINGS, progress=changed)
    assert calls == []
    assert document(out, "result.json")["status"] == document(out, "manifest.json")["status"] == "failed"


def test_admission_does_not_reparse_already_verified_clock_or_observation_bytes(candidate, monkeypatch):
    manifest = (candidate / "manifest.json").read_bytes()
    names = json.loads(manifest)["files"]
    scene = document(candidate, "scene.json")
    original = workflow._document
    seen = []
    def tracked(root, name, hashes):
        assert name not in {"observations.json", "source-clock.json"}
        seen.append(name)
        return original(root, name, hashes)
    monkeypatch.setattr(workflow, "_document", tracked)
    result = workflow._admit_preserved_body_source(candidate, manifest, names, scene, time.monotonic() + 5)
    assert result["identity_documents"] == {name: names[name] for name in seen}
