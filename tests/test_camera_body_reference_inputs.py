"""Authored two-view metadata only; no native model, media or source reconstruction."""

from copy import deepcopy
from hashlib import sha256
import importlib
import json
from pathlib import Path
import time

import pytest

from test_camera_shared_candidate import _deny, _files, _raw, local_only


@pytest.fixture
def api(monkeypatch):
    module = importlib.import_module("weft.camera_body_reference_inputs")
    for path, names in (("weft.camera_body", ("make_camera_body", "build_camera_reference")),
                        ("weft.multiview_root", ("make_two_view_root", "triangulate_pair")),
                        ("weft.multiview_pose", ("triangulate_pair", "_triangulate_rays")),
                        ("weft.camera_ground", ("apply_camera_ground", "audit_ground_clearance"))):
        for name in names:
            monkeypatch.setattr(importlib.import_module(path), name, _deny)
    return module


def inputs(api, *, fps=30000 / 1001, first=0):
    producer = importlib.import_module("weft.camera_shared_candidate")
    files = _files(producer, fps=fps, first=first)
    upstream = {name: sha256(files[name]).hexdigest() for name in ("source.mp4", "source.json", "evidence.json")}
    files["source-input-manifest.json"] = _raw({"schema": 1, "files": upstream})
    files["camera-provenance.json"] = _raw({
        "source_verification": {"manifest_sha256": sha256(files["source-input-manifest.json"]).hexdigest()},
        "camera_file_sha256": sha256(files["camera.json"]).hexdigest(),
        "ground_file_sha256": sha256(files["ground.json"]).hexdigest(),
        "copied_source_files": {name: name for name in upstream},
        "new_training": False, "new_inference": False, "reference_joints_used_for_fitting": False,
        "monocular_camera_reference_unchanged": True,
        "two_view_inputs": {name: {"path": "../unavailable-original/" + name,
            "sha256": sha256(files["inputs/two-view-root/" + name]).hexdigest()}
            for name in ("camera.json", "evidence.json", "config.json")}})
    files["camera-reference.json"] = _raw({"authored_fixture": True})
    files["human.xml"] = b"Analytical fixture only; never read or compiled"
    return files


def package(tmp_path, api, *, candidate=False, files=None, **kwargs):
    files = inputs(api, **kwargs) if files is None else deepcopy(files)
    original = _raw({"schema": 1, "files": {name: sha256(raw).hexdigest() for name, raw in files.items()}})
    if candidate:
        source_files = files
        files = {name: source_files[name] for name in (*api.ALIASES, "source.mp4")}
        files["source-run/manifest.json"] = original
        files["source-verification.json"] = _raw({
            "source_manifest_sha256": sha256(original).hexdigest(), "source_manifest_members": len(source_files),
            "selected_members": {name: {"sha256": sha256(source_files[name]).hexdigest(),
                                        "bytes": len(source_files[name]), "candidate_member": name} for name in api.ALIASES}})
        manifest = {"schema_version": 1, "profile": "camera-shared-candidate-v1", "status": "constructed"}
    else:
        manifest = {"schema": 1}
    root = tmp_path / "source"
    root.mkdir()
    for name, raw in files.items():
        target = root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(raw)
    manifest["files"] = {name: sha256(raw).hexdigest() for name, raw in files.items()}
    (root / "manifest.json").write_bytes(_raw(manifest))
    return root, files


def prepare(api, root, **kwargs):
    return api.prepare_camera_reference_inputs(root, deadline=time.monotonic() + 10, **kwargs)


@pytest.mark.parametrize("candidate,fps,first", [(False, 30, 0), (True, 30000 / 1001, 0), (True, 30000 / 1001, 1)])
def test_exact_raw_aliases_source_slots_clocks_and_selected_reads(api, tmp_path, monkeypatch, candidate, fps, first):
    root, original = package(tmp_path, api, candidate=candidate, fps=fps, first=first)
    read = api._Reader.read
    seen = []
    def counted(self, name, *args, **kwargs):
        assert name not in {"source.mp4", "background.jpg", "human.xml", "scene.json", "task.json", "unrelated/forbidden.bin"}
        seen.append(name)
        return read(self, name, *args, **kwargs)
    monkeypatch.setattr(api._Reader, "read", counted)
    pin = sha256((root / "manifest.json").read_bytes()).hexdigest()
    result = prepare(api, root, expected_manifest_sha256=pin)
    files, identity = result["files"], result["identity"]
    assert files["inputs/human.json"] == original["human.json"]
    assert files["inputs/evidence-derived.json"] == files["inputs/evidence-original.json"] == original["evidence.json"]
    assert files["inputs/source-camera.json"] == original["camera.json"]
    assert "image_dimension_provenance" not in json.loads(files["inputs/evidence-derived.json"])
    assert identity["human_sha256"] != identity["human_canonical_sha256"]
    assert identity["source_camera_sha256"] != identity["source_camera_canonical_sha256"]
    assert identity["body_manifest_sha256"] == pin
    assert identity["dimension_derivation"] == "none; exact original evidence bytes"
    assert identity["clock"]["total_source_frames"] == 12 and identity["clock"]["retained_poses"] == 5
    assert identity["clock"]["first_source_frame"] == first and identity["clock"]["source_fps"] == fps
    assert identity["body_dt_s"] == 1 / 30
    proof = json.loads(files["inputs/camera-inputs-proof.json"])
    assert proof["source_slots"]["body_status_counts"] == {
        "retained_body_pose": 5, "primary_pose_not_sampled": 6, "primary_body_pose_unavailable": 1}
    assert proof["camera_coupling"]["body_sha256"] == identity["human_canonical_sha256"]
    assert set(seen) == set(identity["selected_members"]) and len(seen) == len(set(seen))
    assert len(files) == (17 if candidate else 15) and identity["copied_input_bytes"] == sum(map(len, files.values()))
    seen.clear()
    api.recheck_camera_reference_inputs(result, deadline=time.monotonic() + 10)
    assert set(seen) == set(identity["selected_members"])


@pytest.mark.parametrize("change", ["camera", "secondary", "ground_pin", "source_manifest", "refined", "camera_method", "malformed_provenance"])
def test_resealed_inconsistent_proofs_or_unsupported_body_reject(api, tmp_path, change):
    files = inputs(api)
    if change == "camera":
        name = "camera.json"; value = json.loads(files[name]); value["radial_k1"] = .1
    elif change == "secondary":
        name = "camera-provenance.json"; value = json.loads(files[name]); value["two_view_inputs"]["evidence.json"]["sha256"] = "f" * 64
    elif change == "ground_pin":
        name = "camera-provenance.json"; value = json.loads(files[name]); value["ground_file_sha256"] = "f" * 64
    elif change == "source_manifest":
        name = "source-input-manifest.json"; value = json.loads(files[name]); value["files"]["evidence.json"] = "f" * 64
    elif change == "refined":
        name = "human.json"; value = json.loads(files[name]); value["whole_body_reference_refinement"] = {"status": "prior_candidate"}
    elif change == "camera_method":
        name = "human.json"; value = json.loads(files[name]); value["camera_reference"]["method"] = "camera-root-body-v1"
    else:
        name = "camera-provenance.json"; value = json.loads(files[name]); value["source_verification"] = None
    files[name] = _raw(value)
    root, _ = package(tmp_path, api, files=files)
    with pytest.raises(ValueError):
        prepare(api, root)


@pytest.mark.parametrize("change", ["omitted", "changed", "lab", "failed"])
def test_candidate_custody_is_not_derived_from_remaining_files(api, tmp_path, change):
    root, _ = package(tmp_path, api, candidate=True)
    manifest = json.loads((root / "manifest.json").read_bytes())
    if change == "omitted":
        manifest["files"].pop("source-verification.json")
    elif change == "changed":
        name = "source-verification.json"
        value = json.loads((root / name).read_bytes()); value["selected_members"].pop("camera.json")
        (root / name).write_bytes(_raw(value)); manifest["files"][name] = sha256((root / name).read_bytes()).hexdigest()
    elif change == "lab":
        manifest["files"]["exercise-task.json"] = "f" * 64
    else:
        manifest["status"] = "failed"
    (root / "manifest.json").write_bytes(_raw(manifest))
    with pytest.raises(ValueError):
        prepare(api, root)


@pytest.mark.parametrize("change", ["source_bytes", "failure_marker", "omitted_pin", "changed_alias"])
def test_recheck_rejects_changed_source_or_prepared_copy(api, tmp_path, change):
    root, _ = package(tmp_path, api)
    snapshot = prepare(api, root)
    if change == "source_bytes":
        (root / "human.json").write_bytes(b"{}")
    elif change == "failure_marker":
        (root / "failure.json").write_bytes(b"{}")
    elif change == "omitted_pin":
        snapshot["pins"].pop()
    else:
        snapshot["files"]["inputs/evidence-derived.json"] = b"{}"
    with pytest.raises(ValueError):
        api.recheck_camera_reference_inputs(snapshot, deadline=time.monotonic() + 10)


def test_hash_deadline_and_copy_budget_fail_without_native_work(api, tmp_path, monkeypatch):
    root, _ = package(tmp_path, api)
    with pytest.raises(ValueError, match="SHA256"):
        prepare(api, root, expected_manifest_sha256="f" * 64)
    with pytest.raises(TimeoutError):
        api.prepare_camera_reference_inputs(root, deadline=time.monotonic() - 1)
    monkeypatch.setattr(api, "MAX_COPY_BYTES", 1024)
    with pytest.raises(ValueError, match="copies exceed"):
        prepare(api, root)


@pytest.mark.parametrize("kind", ["member", "root"])
def test_symlink_inputs_are_not_followed(api, tmp_path, kind):
    root, _ = package(tmp_path, api)
    if kind == "member":
        original = root / "human.json"
        moved = root / "retained-human.json"
        original.rename(moved)
        original.symlink_to(moved)
        source = root
    else:
        source = tmp_path / "linked-source"
        source.symlink_to(root, target_is_directory=True)
    with pytest.raises(OSError):
        prepare(api, source)
