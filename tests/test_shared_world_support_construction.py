"""Analytical saved poses test authored-support custody without native geometry."""

from copy import deepcopy
import json
from pathlib import Path
import shutil
import socket
import subprocess

import mujoco
import pytest

from test_exercise_body_source import forbid_runtime, pose_run, seal
from test_whole_body_reference_refinement import fixture as analytical_body
from weft import exercise_body_source, shared_world, shared_world_human, shared_world_support as builder
from weft.io import canonical_hash, digest, write_json
from weft.root_motion import build_root_reference
from weft.shared_support_task import validate_shared_support_task
from weft.shared_world_evidence import verify_shared_world_sources


def forbidden(*_args, **_kwargs):
    pytest.fail("Static-support constructor unit checks must not run native geometry, media, network or subprocesses")


def block_native(monkeypatch):
    forbid_runtime(monkeypatch)
    for name in ("MjModel", "MjData", "mj_forward", "mj_resetData", "mj_step"):
        monkeypatch.setattr(mujoco, name, forbidden)
    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(subprocess, "Popen", forbidden)
    # Native mass extraction is unrelated to these source-custody checks.
    monkeypatch.setattr(shared_world, "human_subtree_mass", lambda _human, scale=1.: 70. * scale ** 3)


def reseal(body):
    seal(body, profile=exercise_body_source.PROFILE, status="construction_only", simulation_ready=False)


def make_body(root):
    """Retain the ordinary pose fixture; its hand-authored body never calls a floor fitter."""
    original = pose_run(root.parent / "poses")
    raw = {name: (original / name).read_bytes() for name in json.loads((original / "manifest.json").read_bytes())["files"]}
    source_sha = digest(original / "source.webm")
    evidence, observations, clocks = exercise_body_source._validated_pose(raw, {"pose_mode": "video", "pose_identity": "raw"}, source_sha)
    human, _ = analytical_body(5)
    human["source_sha256"] = source_sha
    usable = [row for row in evidence["frames"] if row["status"] in {"tracked", "tracked_partial"}]
    angles = human["reference"][0]["joints"]
    human["reference"] = [{"time_s": row["time_s"], "joints": dict(angles)} for row in usable]
    human["reference_evidence"] = [{"time_s": row["time_s"], "evidence_id": row["id"]} for row in usable]
    roots = build_root_reference(evidence, torso_length=human["lengths"]["torso"])
    human["root_reference"] = [{"time_s": row["time_s"], "translation": row["translation"]} for row in roots["reference"]]
    raw.update({"human.json": exercise_body_source._bytes(human), "observations.json": exercise_body_source._bytes(observations),
        "source-clock.json": exercise_body_source._bytes(clocks),
        "implementation.json": b'{"scope":"Analytical saved body fixture; no native producer executed"}\n',
        "inputs/source-manifest.json": (original / "manifest.json").read_bytes()})
    required = exercise_body_source.POSE_FILES | {"human.json", "observations.json", "source-clock.json", "implementation.json"}
    marker = {"schema_version": 1, "profile": exercise_body_source.PROFILE, "status": "construction_only", "simulation_ready": False,
        "source_sha256": source_sha, "source_artifact": "source.webm", "pose_options": {"pose_mode": "video", "pose_identity": "raw"},
        "construction": {"body_model": "mobile", "body_profile": "exercise-v3", "original_pose_config": "run-config.json",
            "original_pose_config_sha256": builder._sha(raw["run-config.json"])},
        "artifacts": {name: builder._sha(raw[name]) for name in required}}
    raw["body-source.json"] = exercise_body_source._bytes(marker)
    root.mkdir()
    for name, data in raw.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    reseal(root)
    return root


@pytest.fixture(scope="module")
def sealed_body(tmp_path_factory):
    with pytest.MonkeyPatch.context() as patch:
        block_native(patch)
        return make_body(tmp_path_factory.mktemp("static-support-body") / "body")


@pytest.fixture
def body(tmp_path, sealed_body, monkeypatch):
    block_native(monkeypatch)
    monkeypatch.setattr(builder, "audit_shared_world_feasibility", lambda scene: {
        "status": "infeasible", "summary": {"physics_steps": 0, "failed_source_poses": 1},
        "scope": "Injected analytical result, not an actual geometry check"})
    monkeypatch.setattr(shared_world_human, "construct_human_only_run", forbidden)
    monkeypatch.setattr(shared_world_human, "validate_person_inventory", forbidden)
    return Path(shutil.copytree(sealed_body, tmp_path / "body"))


def support(shape="box", name="bench"):
    obj = {"id": name, "label": "Authored static support", "role": "static", "shape": shape,
           "position": [1., 1., .3], "quaternion": [1., 0., 0., 0.], "velocity": [0., 0., 0.],
           "angular_velocity": [0., 0., 0.], "mass": 10., "friction": [.8, .01, .001], "color": "#707070",
           "provenance": {"mode": "authored_addition", "calibration_status": "estimated_prior",
               "physical_validation": "unverified", "source_contact_verified": False,
               "assumptions": ["Analytical geometry and material prior; no observed support identity."]}}
    if shape == "box":
        obj["size"] = [.6, .4, .2]
    elif shape == "sphere":
        obj["radius"] = .1
    else:
        obj["geometry"] = {"profile": "rigid-compound-v1", "origin": "authored_hypothesis",
            "assumptions": ["Authored multipart support, not recovered geometry."],
            "inertia": {"mode": "authored_centered_diagonal", "diagonal": [.1, .1, .1]},
            "parts": [{"id": "top", "shape": "box", "position": [0., 0., 0.], "quaternion": [1., 0., 0., 0.],
                       "size": [.4, .4, .1]}]}
    return obj


def config(body, *, shape="box", root=False):
    return {"schema_version": 1, "profile": builder.CONFIG_PROFILE, "source_sha256": digest(body / "source.webm"),
        "body_source_manifest_sha256": digest(body / "manifest.json"), "scale": 1., "objects": [support(shape)],
        "support_requirements": [{"object_id": "bench", "human_segment": "left_forearm", "direction_world": [0., 0., 1.],
                                  "minimum_load_body_weight_fraction": .05, "start_step": 0, "end_step": 58}],
        "root_assistance": root}


@pytest.mark.parametrize("shape", ["box", "sphere", "compound"])
def test_authored_static_shapes_preserve_body_and_every_source_slot(body, tmp_path, shape):
    before = {name: (body / name).read_bytes() for name in json.loads((body / "manifest.json").read_bytes())["files"]}
    settings, out = config(body, shape=shape), tmp_path / "candidate"
    original = deepcopy(settings)
    result = builder.construct_static_support_run(body, settings, out)
    scene = json.loads((out / "scene.json").read_bytes())
    task = json.loads((out / "exercise-task.json").read_bytes())
    assert settings == original
    assert scene["profile"] == result["profile"] == "shared-world-v4"
    assert scene["task"]["type"] == "human_support_v1" and scene["task"]["object_goal"] is None
    assert "inventory" not in scene and not (out / "inventory.json").exists()
    assert not any(item in scene["assumptions"] for item in builder._GROUND_ONLY_ASSUMPTIONS)
    assert all(item in scene["assumptions"] for item in builder._SUPPORT_ASSUMPTIONS)
    assert scene["human_spec"] == json.loads(before["human.json"])
    assert scene["human_spec"]["reference"][0]["time_s"] == 4 / 30
    assert scene["coordinate_frame"]["clock_binding"]["source_time_offset_s"] == 4 / 30
    assert scene["task"]["start_offset_s"] == 0 and scene["task"]["horizon_steps"] == 58
    assert scene["action_contract"]["dynamic_object_ids"] == [] and scene["action_contract"]["object_force_channels"] == 0
    assert all((out / name).read_bytes() == raw and (body / name).read_bytes() == raw for name, raw in before.items())
    assert (out / "inputs/body-source-manifest.json").read_bytes() == (body / "manifest.json").read_bytes()
    assert (out / "unrelated.txt").read_bytes() == before["unrelated.txt"]
    rows = scene["source_binding"]["entity_bindings"]
    assert [row["world_entity_id"] for row in rows] == ["human", "bench"]
    assert rows[0]["observation_document_sha256"] == digest(body / "observations.json")
    assert rows[1]["provenance_mode"] == "authored_addition"
    assert rows[1]["observation_document_sha256"] is rows[1]["observation_entity_id"] is None
    assert rows[1]["observed_slots"] == rows[1]["pre_init_slots"] == 0 and rows[1]["missing_slots"] == 63
    observations = json.loads(before["observations.json"])
    assert len(observations["observations"]) == 63
    assert observations["observations"][30]["visibility"] == "not_detected"
    assert sum(row["visibility"] == "not_sampled" for row in observations["observations"]) == 31
    assert json.loads(before["source-clock.json"])["samples"][7]["decoder_timestamp_s"] is None
    assert validate_shared_support_task(task, scene) == task
    assert task["control"] == {"root_assistance": "disabled", "object_forces": "forbidden"}
    assert task["source_scope"]["source_contact_verified"] is False
    assert result["task_sha256"] == canonical_hash(task) and result["task_file_sha256"] == digest(out / "exercise-task.json")
    assert result["source_sha256"] == settings["source_sha256"] and result["source_manifest_sha256"] == settings["body_source_manifest_sha256"]
    assert result["status"] == "ready" and result["reference_feasibility"]["status"] == "infeasible"
    assert result["simulation_ready"] is False and result["lab_status"] == "not_created"
    assert result["readiness"] == "candidate_constructed_not_lab_admitted"
    assert verify_shared_world_sources(scene, out)["source_evidence_verified"] is True
    manifest = json.loads((out / "manifest.json").read_bytes())
    assert all(digest(out / name) == sha for name, sha in manifest["files"].items())
    assert result["manifest_sha256"] == digest(out / "manifest.json")
    report = json.loads((out / "reconstruction.json").read_bytes())
    assert report["source_verification"]["preserved_files"] == {name: builder._sha(raw) for name, raw in before.items()}
    assert report["work"]["physics_steps"] == report["work"]["video_decodes"] == report["work"]["inference_calls"] == 0


def test_file_config_exact_whitespace_and_explicit_root_assistance_are_preserved(body, tmp_path):
    path = tmp_path / "support.json"
    raw = (json.dumps(config(body, root=True), indent=1) + "  \n").encode()
    path.write_bytes(raw)
    result = builder.construct_static_support_run(body, path, tmp_path / "candidate")
    out = Path(result["out"])
    assert (out / "inputs/support-config.json").read_bytes() == path.read_bytes() == raw
    assert result["config_sha256"] == builder._sha(raw)
    assert json.loads((out / "exercise-task.json").read_bytes())["control"]["root_assistance"] == "explicit_3_axis"
    assert json.loads((out / "reconstruction.json").read_bytes())["config_input_kind"] == "exact_file_bytes"


@pytest.mark.parametrize("change", ["source_pin", "manifest_pin", "unknown", "missing_root", "root_type", "scale_nan", "scale_bounds",
                                    "profile", "schema_bool", "empty", "too_many", "dynamic", "duplicate", "human_id", "malformed_shape",
                                    "observed", "calibrated", "validated", "contact", "extra_provenance", "empty_assumptions",
                                    "motion", "nonunit_orientation", "outside", "unknown_object_field", "ground_requirement", "empty_requirements"])
def test_invalid_config_and_false_provenance_reject_before_any_output(body, tmp_path, change):
    data = config(body)
    obj = data["objects"][0]
    if change == "source_pin": data["source_sha256"] = "f" * 64
    elif change == "manifest_pin": data["body_source_manifest_sha256"] = "f" * 64
    elif change == "unknown": data["camera_calibrated"] = True
    elif change == "missing_root": data.pop("root_assistance")
    elif change == "root_type": data["root_assistance"] = 1
    elif change == "scale_nan": data["scale"] = float("nan")
    elif change == "scale_bounds": data["scale"] = 2.1
    elif change == "profile": data["profile"] = "shared-world-v4"
    elif change == "schema_bool": data["schema_version"] = True
    elif change == "empty": data["objects"] = []
    elif change == "too_many": data["objects"] = [support(name=f"support_{i}") for i in range(13)]
    elif change == "dynamic": obj["role"] = "passive"
    elif change == "duplicate": data["objects"].append(deepcopy(obj))
    elif change == "human_id": obj["id"] = "human"
    elif change == "malformed_shape": obj["shape"] = []
    elif change == "observed": obj["provenance"]["mode"] = "observed_track"
    elif change == "calibrated": obj["provenance"]["calibration_status"] = "calibrated"
    elif change == "validated": obj["provenance"]["physical_validation"] = "verified"
    elif change == "contact": obj["provenance"]["source_contact_verified"] = True
    elif change == "extra_provenance": obj["provenance"]["observation_entity_id"] = "weight-0"
    elif change == "empty_assumptions": obj["provenance"]["assumptions"] = []
    elif change == "motion": obj["velocity"][0] = .1
    elif change == "nonunit_orientation": obj["quaternion"][0] = .5
    elif change == "outside": obj["position"][2] = 6.
    elif change == "unknown_object_field": obj["recovered"] = True
    elif change == "ground_requirement": data["support_requirements"][0]["object_id"] = "mobile_floor"
    else: data["support_requirements"] = []
    out = tmp_path / "candidate"
    with pytest.raises(ValueError):
        builder.construct_static_support_run(body, data, out)
    assert not out.exists()


@pytest.mark.parametrize("change", ["media", "pose", "missing", "body", "clock", "failed_marker", "failed_status", "path_collision"])
def test_unsealed_or_contradictory_body_proof_is_not_admitted(body, tmp_path, change):
    data = config(body)
    if change in {"media", "pose"}:
        (body / ("source.webm" if change == "media" else "pose-landmarks.json")).write_bytes(b"changed")
    elif change == "missing":
        (body / "human.json").unlink()
        reseal(body)
    elif change == "body":
        marker = json.loads((body / "body-source.json").read_bytes())
        marker["artifacts"]["human.json"] = "f" * 64
        write_json(body / "body-source.json", marker)
        reseal(body)
    elif change == "clock":
        clock = json.loads((body / "source-clock.json").read_bytes())
        clock["samples"][7]["decoder_timestamp_s"] = .1
        write_json(body / "source-clock.json", clock)
        marker = json.loads((body / "body-source.json").read_bytes())
        marker["artifacts"]["source-clock.json"] = digest(body / "source-clock.json")
        write_json(body / "body-source.json", marker)
        reseal(body)
    elif change == "failed_marker": (body / "failure.json").write_text("{}")
    elif change == "failed_status":
        write_json(body / "status.json", {"status": "failed"})
        reseal(body)
    else:
        write_json(body / "inputs/support-config.json", {"previous": "not replaceable"})
        reseal(body)
    data["body_source_manifest_sha256"] = digest(body / "manifest.json")
    out = tmp_path / "candidate"
    with pytest.raises(ValueError):
        builder.construct_static_support_run(body, data, out)
    assert not out.exists()


@pytest.mark.parametrize("mutation", ["source", "resealed_source", "failure_marker", "config", "output_extra", "forward_failure", "deadline"])
def test_late_failure_preserves_original_attempt_and_prevents_ready_seal(body, tmp_path, monkeypatch, mutation):
    data = config(body)
    path, out = tmp_path / "config.json", tmp_path / "candidate"
    write_json(path, data)

    def changed(_scene):
        if mutation in {"source", "resealed_source"}:
            (body / "source.webm").write_bytes(b"changed during construction")
            if mutation == "resealed_source": reseal(body)
        elif mutation == "failure_marker": (body / "failure.json").write_text("{}")
        elif mutation == "config": path.write_text("{}")
        elif mutation == "output_extra": (out / "unexpected.json").write_text("{}")
        elif mutation == "deadline": raise TimeoutError("Analytical native-call deadline boundary")
        else: raise RuntimeError("Analytical forward failure")
        return {"status": "feasible", "summary": {"physics_steps": 0}}

    monkeypatch.setattr(builder, "audit_shared_world_feasibility", changed)
    with pytest.raises((ValueError, RuntimeError, TimeoutError)):
        builder.construct_static_support_run(body, path, out)
    assert json.loads((out / "manifest.json").read_bytes())["status"] == "failed"
    assert json.loads((out / "failure.json").read_bytes())["status"] == "failed"
    assert (out / "scene.json").is_file() and (out / "exercise-task.json").is_file()
    assert (out / "inputs/support-config.json").read_bytes() != b"{}"
    if mutation == "output_extra": assert (out / "unexpected.json").read_bytes() == b"{}"
    with pytest.raises(ValueError):
        builder.construct_static_support_run(body, data, out)


@pytest.mark.parametrize("which", ["inside_input", "ancestor", "occupied", "symlink", "traversal", "seconds", "input_bytes", "output_bytes", "input_files", "output_files"])
def test_bounds_and_existing_paths_cannot_start_an_output(body, tmp_path, monkeypatch, which):
    data, out, options = config(body), tmp_path / "candidate", {}
    if which == "inside_input": out = body / "new"
    elif which == "ancestor": out = tmp_path
    elif which == "occupied": out.mkdir()
    elif which == "symlink":
        link = tmp_path / "alias"
        link.symlink_to(body, target_is_directory=True)
        out = link / "new"
    elif which == "traversal": out = body / ".." / "candidate"
    elif which == "seconds": options["max_seconds"] = 31
    elif which == "input_bytes": monkeypatch.setattr(builder, "MAX_INPUT_BYTES", 1024)
    elif which == "output_bytes": monkeypatch.setattr(builder, "MAX_OUTPUT_BYTES", 1024)
    elif which == "input_files": monkeypatch.setattr(builder, "MAX_INPUT_FILES", 2)
    else: monkeypatch.setattr(builder, "MAX_OUTPUT_FILES", 3)
    with pytest.raises(ValueError):
        builder.construct_static_support_run(body, data, out, **options)
    assert not (out / "manifest.json").exists()


def test_duplicate_config_keys_and_oversized_config_are_rejected(body, tmp_path):
    path = tmp_path / "config.json"
    path.write_text('{"schema_version":1,"schema_version":1}')
    with pytest.raises(ValueError, match="[Dd]uplicate"):
        builder.construct_static_support_run(body, path, tmp_path / "first")
    path.write_bytes(b" " * (builder.MAX_CONFIG_BYTES + 1))
    with pytest.raises(ValueError, match="bounded"):
        builder.construct_static_support_run(body, path, tmp_path / "second")
    assert not (tmp_path / "first").exists() and not (tmp_path / "second").exists()
