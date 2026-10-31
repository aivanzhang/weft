"""Analytical source fixtures exercise construction custody, not reconstruction accuracy."""

import json
from pathlib import Path
import shutil
import time

import pytest

from test_exercise_body_source import forbid_runtime, pose_run, seal
from weft import exercise_body_source, shared_world_human as builder
from weft.io import canonical_hash, digest, write_json
from weft.person_inventory import make_person_inventory
from weft.shared_world_evidence import verify_shared_world_sources
from weft.shared_world_packaging import collect_shared_source_files


@pytest.fixture(scope="module")
def analytical_body(tmp_path_factory):
    root = tmp_path_factory.mktemp("person-only-body")
    original = pose_run(root / "poses")
    with pytest.MonkeyPatch.context() as patch:
        forbid_runtime(patch)
        exercise_body_source.prepare_exercise_body_source_from_evidence(original, root / "body")
    return root / "body"


@pytest.fixture
def body(tmp_path, analytical_body, monkeypatch):
    forbid_runtime(monkeypatch)
    return Path(shutil.copytree(analytical_body, tmp_path / "body"))


def inventory(body):
    return make_person_inventory(json.loads((body / "body-source.json").read_bytes())["source_sha256"])


def reseal(body):
    seal(body, profile=exercise_body_source.PROFILE, status="construction_only", simulation_ready=False)


def test_direct_body_candidate_has_no_props_and_preserves_exact_source_clocks(body, tmp_path):
    before = {path.relative_to(body).as_posix(): path.read_bytes() for path in body.rglob("*") if path.is_file()}
    out = tmp_path / "candidate"
    result = builder.construct_human_only_run(body, inventory(body), out)
    assert result["status"] == "ready" and result["profile"] == "shared-world-v3"
    assert result["readiness"] == "candidate_constructed_not_lab_admitted"
    assert result["lab_status"] == "not_created" and result["simulation_ready"] is False
    scene = json.loads((out / "scene.json").read_bytes())
    assert scene["objects"] == [] and scene["target"] is None and scene["controlled_id"] == "human"
    assert scene["task"]["object_goal"] is None and scene["task"]["type"] == "human_motion_v1"
    assert scene["task"]["start_offset_s"] == 0 and scene["task"]["horizon_steps"] == 58
    assert scene["human_spec"] == json.loads(before["human.json"])
    assert scene["inventory"] == inventory(body)
    assert scene["action_contract"]["object_force_channels"] == 0
    assert len(scene["action_contract"]["names"]) == 22
    assert result["scene_sha256"] == canonical_hash(scene)
    assert result["manifest_sha256"] == digest(out / "manifest.json")
    assert scene["source_binding"]["total_source_frames"] == 63
    assert [row["world_entity_id"] for row in scene["source_binding"]["entity_bindings"]] == ["human"]
    for name in exercise_body_source.POSE_FILES | {"human.json", "observations.json", "source-clock.json", "source.webm"}:
        assert (out / name).read_bytes() == before[name]
    observations = json.loads(before["observations.json"])
    assert len(observations["observations"]) == 63
    assert sum(row["visibility"] == "not_sampled" for row in observations["observations"]) == 31
    assert observations["observations"][30]["visibility"] == "not_detected"
    clock = json.loads((out / "source-clock.json").read_bytes())
    assert clock["samples"][7]["decoder_timestamp_s"] is None
    assert (out / "inputs/body-source-manifest.json").read_bytes() == before["manifest.json"]
    assert verify_shared_world_sources(scene, out)["source_evidence_verified"] is True
    pinned = collect_shared_source_files(scene, out, expected_manifest_sha256=result["manifest_sha256"])
    assert pinned.files
    report = json.loads((out / "reconstruction.json").read_bytes())
    assert report["work"]["physics_steps"] == report["work"]["inference_calls"] == report["work"]["video_decodes"] == 0
    assert report["source_verification"]["verified_files"] == len(before) - 1
    assert all((body / name).read_bytes() == raw for name, raw in before.items())
    assert not any((out / name).exists() for name in ("construction-seed.json", "lab", "lab-config.json", "policy.json"))


def test_identity_reader_is_metadata_only_then_constructor_verifies_original_media(body, tmp_path, monkeypatch):
    original_read, reads = builder.read_package_file, []

    def tracked(root, name, **options):
        reads.append(name)
        return original_read(root, name, **options)

    monkeypatch.setattr(builder, "read_package_file", tracked)
    result = builder.read_person_body_source_identity(body)
    assert result["source_sha256"] == inventory(body)["source_sha256"]
    assert result["manifest_sha256"] == digest(body / "manifest.json")
    assert set(reads) <= {"manifest.json", "source.json", "body-source.json"}
    (body / "source.webm").write_bytes(b"changed unsealed source")
    out = tmp_path / "candidate"
    with pytest.raises(ValueError, match="SHA256|bounded regular"):
        builder.construct_human_only_run(body, make_person_inventory(result["source_sha256"]), out)
    assert not out.exists()


@pytest.mark.parametrize("change", ["source_hash", "support", "equipment", "missing_proof", "extra_scene", "extra_equipment", "failure"])
def test_unsupported_or_contradictory_inputs_stop_before_output(body, tmp_path, change):
    declared = inventory(body)
    if change == "source_hash":
        declared["source_sha256"] = "f" * 64
    elif change == "support":
        declared["support"] = "unknown"
    elif change == "equipment":
        declared["task_relevant_equipment"] = "weight"
    elif change == "missing_proof":
        (body / "pose-recovery.json").write_text("{}")
        reseal(body)
    elif change in {"extra_scene", "extra_equipment"}:
        name = "scene.json" if change == "extra_scene" else "scene-observations.json"
        (body / name).write_text("{}")
        reseal(body)
    else:
        (body / "failure.json").write_text("{}")
    out = tmp_path / "candidate"
    with pytest.raises(ValueError):
        builder.construct_human_only_run(body, declared, out)
    assert not out.exists()


@pytest.mark.parametrize("mutation", ["source_bytes", "resealed_source", "failure_marker"])
def test_source_mutation_during_forward_check_cannot_publish_ready(body, tmp_path, monkeypatch, mutation):
    def changed(scene):
        if mutation == "failure_marker":
            (body / "failure.json").write_text("{}")
        else:
            (body / "source.webm").write_bytes(b"changed after source admission")
            if mutation == "resealed_source":
                reseal(body)
        return {"status": "feasible", "summary": {"physics_steps": 0}}

    monkeypatch.setattr(builder, "audit_shared_world_feasibility", changed)
    out = tmp_path / "candidate"
    with pytest.raises(ValueError):
        builder.construct_human_only_run(body, inventory(body), out)
    assert json.loads((out / "failure.json").read_bytes())["status"] == "failed"
    assert json.loads((out / "manifest.json").read_bytes())["status"] == "failed"
    assert (out / "scene.json").is_file()


def test_reference_feasibility_remains_informational_without_shifting_or_dropping_pose(body, tmp_path, monkeypatch):
    monkeypatch.setattr(builder, "audit_shared_world_feasibility", lambda scene: {
        "status": "infeasible", "scope": "Injected analytical gate result; not source reconstruction evidence",
        "summary": {"physics_steps": 0, "failed_source_poses": 1}})
    out = tmp_path / "candidate"
    result = builder.construct_human_only_run(body, inventory(body), out)
    assert result["status"] == "ready" and result["reference_feasibility"]["status"] == "infeasible"
    assert result["simulation_ready"] is False and result["lab_status"] == "not_created"
    assert (out / "human.json").read_bytes() == (body / "human.json").read_bytes()
    assert json.loads((out / "scene.json").read_bytes())["task"]["horizon_steps"] == 58


@pytest.mark.parametrize("which", ["parent_alias", "inside_input", "symlink", "occupied", "deadline", "scale", "seconds"])
def test_output_paths_and_bounds_fail_before_construction(body, tmp_path, which):
    out, options = tmp_path / "candidate", {}
    if which == "parent_alias":
        out = body / ".." / "body" / "new"
    elif which == "inside_input":
        out = body / "new"
    elif which == "symlink":
        link = tmp_path / "alias"
        link.symlink_to(body, target_is_directory=True)
        out = link / "new"
    elif which == "occupied":
        out.mkdir()
    elif which == "deadline":
        options["deadline"] = time.monotonic() - 1
    elif which == "scale":
        options["scale"] = float("nan")
    else:
        options["max_seconds"] = 31
    with pytest.raises((ValueError, TimeoutError)):
        builder.construct_human_only_run(body, inventory(body), out, **options)
    assert not (out / "manifest.json").exists()


def test_insufficient_output_reservation_creates_nothing(body, tmp_path, monkeypatch):
    monkeypatch.setattr(builder, "MAX_OUTPUT_BYTES", 1024)
    out = tmp_path / "candidate"
    with pytest.raises(ValueError, match="capacity"):
        builder.construct_human_only_run(body, inventory(body), out)
    assert not out.exists()


@pytest.mark.parametrize("which", ["marker", "media_map", "failure", "symlink"])
def test_identity_reader_rejects_unsealed_or_conflicting_identity(body, tmp_path, which):
    if which == "marker":
        (body / "body-source.json").write_text("{}")
    elif which == "media_map":
        data = json.loads((body / "manifest.json").read_bytes())
        data["files"]["source.webm"] = "f" * 64
        write_json(body / "manifest.json", data)
    elif which == "failure":
        (body / "failure.json").write_text("{}")
    else:
        link = tmp_path / "alias"
        link.symlink_to(body, target_is_directory=True)
        body = link
    with pytest.raises(ValueError):
        builder.read_person_body_source_identity(body)


@pytest.mark.parametrize("artifact", [[], {}, 1, None, "../source.webm"])
def test_identity_reader_rejects_malformed_source_artifact_as_input_error(body, artifact):
    marker = json.loads((body / "body-source.json").read_bytes())
    marker["source_artifact"] = artifact
    write_json(body / "body-source.json", marker)
    reseal(body)
    with pytest.raises(ValueError, match="source identities"):
        builder.read_person_body_source_identity(body)
