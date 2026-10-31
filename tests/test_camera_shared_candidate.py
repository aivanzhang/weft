"""Analytical packages and mocked shared validator; no source media or native work."""

import builtins
from copy import deepcopy
from hashlib import sha256
import importlib
import json
import math
import os
from pathlib import Path
import socket
import subprocess

import pytest


def _deny(*_args, **_kwargs):
    raise AssertionError("Native/model/media/network/subprocess work is forbidden")


@pytest.fixture(autouse=True)
def local_only(monkeypatch):
    original = builtins.__import__

    def guarded(name, *args, **kwargs):
        if (name.split(".")[0] in {"mujoco", "cv2", "PIL", "mediapipe", "torch", "tensorflow", "requests", "httpx"}
                or name in {"weft.human", "weft.mobile_human", "weft.shared_world", "weft.foot_body"}):
            _deny()
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)
    for owner, names in ((socket, ("socket", "create_connection")),
                         (subprocess, ("Popen", "run", "call", "check_call", "check_output")),
                         (os, ("system", "popen", "fork", "posix_spawn", "posix_spawnp"))):
        for name in names:
            if hasattr(owner, name):
                monkeypatch.setattr(owner, name, _deny)


@pytest.fixture
def api(monkeypatch):
    module = importlib.import_module("weft.camera_shared_candidate")
    for path, names in (("weft.multiview_pose", ("triangulate_pair", "_triangulate_rays")),
                        ("weft.camera_body", ("make_camera_body", "build_camera_reference")),
                        ("weft.multiview_root", ("make_two_view_root", "triangulate_pair")),
                        ("weft.camera_ground", ("apply_camera_ground", "audit_ground_clearance"))):
        for name in names:
            monkeypatch.setattr(importlib.import_module(path), name, _deny)
    calls = []

    def horizon(rows, dt):
        duration = rows[-1]["time_s"] - rows[0]["time_s"]
        steps = round(duration / dt)
        if not 1 <= steps <= 1800 or not math.isclose(steps * dt, duration, abs_tol=1e-9, rel_tol=0):
            raise ValueError("Original complete horizon is not grid aligned")
        return steps

    def validate(scene):
        calls.append(deepcopy(scene))
        if scene["dt"] / scene["substeps"] > 1 / 240 + 1e-12:
            raise ValueError("Camera control timing requires enough substeps to retain the physics timestep envelope")
        assert scene["objects"] == [] and "inventory" not in scene and scene["target"] is None
        assert scene["learned_policy_available"] is False
        return deepcopy(scene)

    limits = {"angular_rmse_rad": .35, "root_rmse_model_units": .15, "max_floor_penetration_model_units": .025}
    monkeypatch.setattr(module, "_shared_api", lambda: (limits, horizon, validate))
    # One mock runtime file keeps orchestration tests independent of other agents' source edits.
    module._test_runtime = module._runtime
    monkeypatch.setattr(module, "_runtime", lambda _deadline: ({}, {"python": "analytical-test"}))
    module._test_calls = calls
    return module


def _raw(value):
    return (json.dumps(value, sort_keys=True, allow_nan=False) + "\n").encode()


def _load(path):
    return json.loads(path.read_bytes())


def _replace(value, old, new):
    if type(value) is dict:
        return {key: _replace(item, old, new) for key, item in value.items()}
    if type(value) is list:
        return [_replace(item, old, new) for item in value]
    return value.replace(old, new) if type(value) is str else value


def _files(api, *, fps=30000 / 1001, first=0):
    producer = importlib.import_module("weft.multiview_foot_reference")
    fixture_tools = importlib.import_module("test_multiview_foot_reference")
    sampled = tuple(range(first, 12, 2))
    value = fixture_tools._fixture(producer, fps=fps, sampled=sampled)
    media = b"NOT A VIDEO: authored analytical custody fixture"
    source_sha = sha256(media).hexdigest()
    value = _replace(value, "a" * 64, source_sha)
    value["body"]["schema_version"] = 3
    for evidence in value["evidence"]:
        for row in evidence["frames"]:
            row["world_landmarks"] = []
    # Preserve one explicit failed detector slot, never fill it or use a later start.
    missing = sampled[-2]
    row = next(row for row in value["evidence"][0]["frames"] if row["frame"] == missing)
    row.update(status="no_pose", landmarks=[], world_landmarks=[], pose_count=0)
    for key in ("reference", "reference_evidence", "root_reference"):
        value["body"][key] = [row for row in value["body"][key] if row["time_s"] != missing / fps]
    binding = value["body"]["camera_reference"]
    binding["source_frame_ids"] = [row["evidence_id"] for row in value["body"]["reference_evidence"]]
    binding["root_refinement"]["roots"] = [row for row in binding["root_refinement"]["roots"] if row["frame"] != missing]
    fixture_tools._pin(value, producer)
    body, prediction = value["body"], value["predictions"]
    root = body["camera_reference"]["root_refinement"]
    retained = {row["frame"] for row in root["roots"]}
    report = {"method": body["camera_reference"]["method"], "source_frames": 12, "binding": deepcopy(root),
              "source_slots": [{"frame": row["frame"], "time_s": row["time_s"],
                  "status": "retained_body_pose" if row["frame"] in retained else
                            "primary_pose_not_sampled" if row["source_evidence_ids"][0] is None else "primary_body_pose_unavailable",
                  "hip_statuses": [row["joints"][i]["status"] for i in (23, 24)]} for row in prediction["frames"]]}
    config = {"schema_version": 1, "first_name": root["views"][0]["name"], "second_name": root["views"][1]["name"],
              "clock_provenance": root["views"][0]["clock_provenance"], "world_provenance": root["world_provenance"],
              "second_evidence": "../unavailable-original/evidence.json", "second_camera": "../unavailable-original/camera.json"}
    files = {name: _raw({"analytical_fixture": True}) for name in api.REQUIRED}
    files.update({"human.json": _raw(body), "evidence.json": _raw(value["evidence"][0]),
                  "camera.json": _raw(value["cameras"][0]), "ground.json": _raw(value["ground"]),
                  "two-view-triangulation.json": _raw(prediction), "two-view-root.json": _raw(report),
                  "inputs/two-view-root/evidence.json": _raw(value["evidence"][1]),
                  "inputs/two-view-root/camera.json": _raw(value["cameras"][1]),
                  "inputs/two-view-root/config.json": _raw(config),
                  "source.json": _raw({"artifact": "source.mp4", "sha256": source_sha, "original_name": "fixture.mp4",
                                       "rights": "Authored test bytes; not a recorded video or rights verification"}),
                  "source.mp4": media, "background.jpg": b"NOT AN IMAGE", "unrelated/forbidden.bin": b"never read"})
    return files


def _package(tmp_path, api, *, files=None, **kwargs):
    root = tmp_path / "source"
    root.mkdir()
    files = _files(api, **kwargs) if files is None else files
    for name, raw in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
    (root / "manifest.json").write_bytes(_raw({"schema": 1, "files": {name: sha256(raw).hexdigest() for name, raw in files.items()}}))
    return root, files


def _seal_valid(out, status):
    manifest = _load(out / "manifest.json")
    assert manifest["status"] == status
    for name, expected in manifest["files"].items():
        assert sha256((out / name).read_bytes()).hexdigest() == expected
    return manifest


@pytest.mark.parametrize("fps,first", [(30, 0), (30000 / 1001, 0), (30000 / 1001, 1)])
def test_preserves_exact_body_both_views_all_slots_clocks_and_limits(api, tmp_path, monkeypatch, fps, first):
    source, files = _package(tmp_path, api, fps=fps, first=first)
    source_manifest = (source / "manifest.json").read_bytes()
    read = api.read_package_file
    reads = []

    def bounded_read(directory, name, **kwargs):
        reads.append((Path(directory), name))
        assert name != "unrelated/forbidden.bin"
        return read(directory, name, **kwargs)

    monkeypatch.setattr(api, "read_package_file", bounded_read)
    out = tmp_path / "candidate"
    result = api.prepare_camera_shared_candidate(source, out)
    assert result["status"] == "constructed", result
    assert result["sealed"] and result["scope"]["lab_created"] is False
    manifest = _seal_valid(out, "constructed")
    assert result["manifest_sha256"] == sha256((out / "manifest.json").read_bytes()).hexdigest()
    assert (out / "source-run/manifest.json").read_bytes() == source_manifest
    assert (source / "manifest.json").read_bytes() == source_manifest
    for name in api.REQUIRED + ("source.mp4", "background.jpg"):
        target = "source-run/" + name if name in ("scene.json", "task.json") else name
        assert (out / target).read_bytes() == files[name]
    assert "unrelated/forbidden.bin" not in manifest["files"]
    scene = _load(out / "scene.json")
    assert scene["human_spec"] == json.loads(files["human.json"])
    assert scene["dt"] == 1 / fps and scene["human_spec"]["dt"] == 1 / 30
    assert scene["substeps"] == max(8, math.ceil(240 / fps))
    assert scene["coordinate_frame"]["clock_binding"]["physics_substeps"] == scene["substeps"]
    assert scene["task"]["horizon_steps"] == 10 and scene["task"]["start_offset_s"] == 0
    assert scene["coordinate_frame"]["clock_binding"]["source_time_offset_s"] == first / fps
    assert scene["task"]["human_limits"]["max_floor_penetration_model_units"] == .025
    assert scene["objects"] == [] and "inventory" not in scene and "exercise-task.json" not in manifest["files"]
    assert scene["camera_coupling"]["world_to_local"]["translation"] == [-1., -2., -0.]
    proof = _load(out / "source-verification.json")
    assert len(proof["source_slots"]) == 12 and len(proof["views"]) == 2
    assert [row["frame"] for row in proof["source_slots"]] == list(range(12))
    assert sum(row["body_status"] == "primary_body_pose_unavailable" for row in proof["source_slots"]) == 1
    assert proof["secondary_custody"]["rights"] == proof["secondary_custody"]["media"] == "unavailable"
    assert proof["primary_rights"]["status"] == "retained_source_declaration_not_new_verification"
    binding = scene["source_binding"]
    observation_path = next(iter(binding["evidence_documents"].values()))
    observations = _load(out / observation_path)
    assert observations["camera"]["motion"] == {"value": None, "status": "unknown", "evidence_ids": []}
    assert len(observations["observations"]) == 12
    assert sum(row["visibility"] == "not_sampled" for row in observations["observations"]) == 6
    assert sum(row["visibility"] == "not_detected" for row in observations["observations"]) == 1
    assert binding["entity_bindings"][0]["observed_slots"] == 5
    assert binding["entity_bindings"][0]["missing_slots"] == 7
    assert len(api._test_calls) == 1 and result["simulation_steps"] == result["paid_requests"] == 0


@pytest.mark.parametrize("change", ["camera", "secondary_source", "triangulation", "ground", "source_rights", "slot_count",
    "slot_identity", "root_report", "root_hips", "config", "raw_hash", "missing_member", "source_failed"])
def test_source_changes_fail_closed_and_seal_without_native_validation(api, tmp_path, change):
    files = _files(api)
    name = {"camera": "inputs/two-view-root/camera.json", "secondary_source": "inputs/two-view-root/evidence.json",
            "triangulation": "two-view-triangulation.json", "ground": "ground.json", "source_rights": "source.json",
            "slot_count": "two-view-triangulation.json", "slot_identity": "two-view-triangulation.json",
            "root_report": "two-view-root.json", "root_hips": "two-view-triangulation.json", "config": "inputs/two-view-root/config.json"}.get(change)
    if name:
        value = json.loads(files[name])
        if change == "camera": value["quality"] = "measured"
        elif change == "secondary_source": value["source_sha256"] = "c" * 64
        elif change == "triangulation": value["physical_validation"] = "verified"
        elif change == "ground": value["height_m"] += .1
        elif change == "source_rights": value.pop("rights")
        elif change == "slot_count": value["frames"].pop()
        elif change == "slot_identity": value["frames"][1]["source_evidence_ids"][0] = "invented"
        elif change == "root_report": value["source_slots"][0]["status"] = "repaired"
        elif change == "root_hips": value["frames"][0]["joints"][23]["world_point_m"][0] += .1
        elif change == "config": value["world_provenance"] = "fabricated replacement"
        files[name] = _raw(value)
    if change == "missing_member": files.pop("ground.json")
    if change == "source_failed": files["failure.json"] = b"{}"
    source, files = _package(tmp_path, api, files=files)
    if change == "raw_hash": (source / "human.json").write_bytes(files["human.json"] + b" ")
    before = {name: (source / name).read_bytes() for name in files}
    result = api.prepare_camera_shared_candidate(source, tmp_path / "candidate")
    assert result["status"] == "failed" and result["sealed"], result
    _seal_valid(tmp_path / "candidate", "failed")
    assert api._test_calls == []
    assert before == {name: (source / name).read_bytes() for name in files}


@pytest.mark.parametrize("change", ["unsafe_manifest", "too_many_manifest_members", "duplicate_json", "nonfinite_json", "oversized_manifest"])
def test_manifest_and_json_bounds_precede_construction(api, tmp_path, change):
    source, files = _package(tmp_path, api)
    manifest = _load(source / "manifest.json")
    if change == "unsafe_manifest": manifest["files"]["../outside"] = "a" * 64
    elif change == "too_many_manifest_members": manifest["files"].update({f"extra/{i}": "a" * 64 for i in range(513)})
    elif change == "duplicate_json":
        raw = b'{"schema_version":3,"schema_version":3}'
        (source / "human.json").write_bytes(raw)
        manifest["files"]["human.json"] = sha256(raw).hexdigest()
    elif change == "nonfinite_json":
        raw = b'{"unsafe": NaN}'
        (source / "human.json").write_bytes(raw)
        manifest["files"]["human.json"] = sha256(raw).hexdigest()
    (source / "manifest.json").write_bytes(_raw(manifest) if change != "oversized_manifest" else b" " * (api.MAX_MANIFEST_BYTES + 1))
    result = api.prepare_camera_shared_candidate(source, tmp_path / "candidate")
    assert result["status"] == "failed" and result["sealed"]
    _seal_valid(tmp_path / "candidate", "failed")
    assert not api._test_calls


@pytest.mark.parametrize("change", ["input_symlink", "output_symlink", "inside_source", "existing_output", "parent_traversal"])
def test_unsafe_paths_raise_without_overwrite(api, tmp_path, change):
    source, _ = _package(tmp_path, api)
    out = tmp_path / "out"
    if change == "input_symlink":
        link = tmp_path / "source-link"
        link.symlink_to(source, target_is_directory=True)
        source = link
    elif change == "output_symlink": out.symlink_to(source, target_is_directory=True)
    elif change == "inside_source": out = source / "new"
    elif change == "existing_output": out.mkdir(); (out / "keep").write_bytes(b"preserve")
    elif change == "parent_traversal": out = tmp_path / "nested" / ".." / "new"
    with pytest.raises(ValueError):
        api.prepare_camera_shared_candidate(source, out)
    if change == "existing_output": assert (out / "keep").read_bytes() == b"preserve"


def test_source_member_symlink_is_rejected_and_failure_is_sealed(api, tmp_path):
    source, files = _package(tmp_path, api)
    external = tmp_path / "outside-camera.json"
    external.write_bytes(files["camera.json"])
    (source / "camera.json").unlink()
    (source / "camera.json").symlink_to(external)
    result = api.prepare_camera_shared_candidate(source, tmp_path / "candidate")
    assert result["status"] == "failed" and result["sealed"]
    assert external.read_bytes() == files["camera.json"]


def test_shared_validator_rejection_is_not_repaired(api, tmp_path, monkeypatch):
    source, _ = _package(tmp_path, api)
    limits, horizon, _ = api._shared_api()

    def rejected(_scene):
        raise ValueError("Original geometry intersects floor")

    monkeypatch.setattr(api, "_shared_api", lambda: (limits, horizon, rejected))
    result = api.prepare_camera_shared_candidate(source, tmp_path / "candidate")
    assert result["status"] == "failed" and result["stage"] == "shared_schema"
    assert "intersects floor" in result["message"] and result["sealed"]
    _seal_valid(tmp_path / "candidate", "failed")


def test_validator_body_mutation_is_rejected(api, tmp_path, monkeypatch):
    source, _ = _package(tmp_path, api)
    limits, horizon, validate = api._shared_api()

    def altered(scene):
        result = validate(scene)
        result["human_spec"]["dt"] = .01
        return result

    monkeypatch.setattr(api, "_shared_api", lambda: (limits, horizon, altered))
    result = api.prepare_camera_shared_candidate(source, tmp_path / "candidate")
    assert result["status"] == "failed" and "must not alter" in result["message"]


def test_source_change_before_terminal_seal_is_preserved_as_failure(api, tmp_path, monkeypatch):
    source, _ = _package(tmp_path, api)
    original = api._scene

    def changed(*args):
        result = original(*args)
        (source / "ground.json").write_bytes(b'{"changed":true}')
        return result

    monkeypatch.setattr(api, "_scene", changed)
    result = api.prepare_camera_shared_candidate(source, tmp_path / "candidate")
    assert result["status"] == "failed" and result["stage"] == "terminal_verification"
    _seal_valid(tmp_path / "candidate", "failed")


def test_late_final_inventory_read_cannot_publish_constructed_seal(api, tmp_path, monkeypatch):
    source, _ = _package(tmp_path, api)
    clock = [0.0]
    monkeypatch.setattr(api.time, "monotonic", lambda: clock[0])
    original = api.read_package_file
    out = tmp_path / "candidate"

    def late(directory, name, **kwargs):
        raw = original(directory, name, **kwargs)
        if Path(directory) == out and name == "candidate.json":
            clock[0] = 31.0
        return raw

    monkeypatch.setattr(api, "read_package_file", late)
    result = api.prepare_camera_shared_candidate(source, out)
    assert result["status"] == "failed" and result["error"] == "TimeoutError" and result["sealed"]
    _seal_valid(out, "failed")
    assert _load(out / "candidate.json")["status"] == "prepared_unsealed"


def test_late_staged_terminal_write_cannot_publish_constructed_seal(api, tmp_path, monkeypatch):
    source, _ = _package(tmp_path, api)
    clock = [0.0]
    monkeypatch.setattr(api.time, "monotonic", lambda: clock[0])
    original = api.write_package_files

    def late(directory, files):
        result = original(directory, files)
        if "manifest.json" in files and json.loads(files["manifest.json"]).get("status") == "constructed":
            clock[0] = 31.0
        return result

    monkeypatch.setattr(api, "write_package_files", late)
    out = tmp_path / "candidate"
    result = api.prepare_camera_shared_candidate(source, out)
    assert result["status"] == "failed" and result["sealed"]
    _seal_valid(out, "failed")


def test_post_publication_cleanup_does_not_relabel_terminal_success(api, tmp_path, monkeypatch):
    source, _ = _package(tmp_path, api)
    clock = [0.0]
    monkeypatch.setattr(api.time, "monotonic", lambda: clock[0])
    link = api.os.link

    def late(*args, **kwargs):
        result = link(*args, **kwargs)
        clock[0] = 31.0
        return result

    monkeypatch.setattr(api.os, "link", late)
    result = api.prepare_camera_shared_candidate(source, tmp_path / "candidate")
    assert result["status"] == "constructed" and result["sealed"] and result["seconds"] == 31.0
    _seal_valid(tmp_path / "candidate", "constructed")


def test_output_tampering_is_not_certified_by_failure_seal(api, tmp_path, monkeypatch):
    source, _ = _package(tmp_path, api)
    out = tmp_path / "candidate"
    original = api._inventory

    def changed(*args):
        (out / "ground.json").write_bytes(b'{"tampered":true}')
        return original(*args)

    monkeypatch.setattr(api, "_inventory", changed)
    result = api.prepare_camera_shared_candidate(source, out)
    assert result["status"] == "failed" and result["sealed"]
    manifest = _seal_valid(out, "failed")
    assert set(manifest["files"]) == {"failure.json"}
    assert "ground.json" in _load(out / "failure.json")["unadmitted_attempted_files"]
    assert (out / "ground.json").read_bytes() == b'{"tampered":true}'
    assert _load(out / "candidate.json")["status"] == "prepared_unsealed"


def test_runtime_snapshot_is_explicit_bounded_and_has_import_dependencies(api, tmp_path, monkeypatch):
    root = tmp_path / "runtime" / "src" / "weft"
    root.mkdir(parents=True)
    for name in api.RUNTIME_FILES:
        (root / name).write_bytes(b"# analytical runtime pin\n")
    for name in ("pyproject.toml", "requirements-lab.lock"):
        (root.parent.parent / name).write_bytes(b"# analytical lock\n")
    monkeypatch.setattr(api, "RUNTIME_ROOT", root)
    monkeypatch.setattr(api, "version", lambda name: "analytical-" + name)
    runtime, versions = api._test_runtime(api.time.monotonic() + 10)
    assert len(runtime) == len(api.RUNTIME_FILES) + 2
    assert {"multiview_segments.py", "root_motion.py", "errors.py"} <= set(api.RUNTIME_FILES)
    assert versions["mujoco"] == "analytical-mujoco"
    assert sum(map(len, runtime.values())) < api.MAX_RUNTIME_BYTES
    monkeypatch.setattr(api, "MAX_RUNTIME_BYTES", 4)
    with pytest.raises(ValueError):
        api._test_runtime(api.time.monotonic() + 10)


@pytest.mark.parametrize("seconds", [True, 0, -1, 31, float("inf"), float("nan")])
def test_invalid_work_limits_reject_before_output_creation(api, tmp_path, seconds):
    with pytest.raises(ValueError, match="deadline"):
        api.prepare_camera_shared_candidate(tmp_path / "source", tmp_path / "out", max_seconds=seconds)
    assert not (tmp_path / "out").exists()
