"""Candidate custody and failure handling, with native and external work forbidden."""

from copy import deepcopy
import json
import os
from pathlib import Path
import socket
import subprocess
from types import SimpleNamespace

import mujoco
import pytest

from test_shared_exercise_feet_projection import scene_fixture
from test_world_evidence import document
from weft import shared_exercise_feet_candidate as candidate
from weft import shared_exercise_feet_task, shared_exercise_task, shared_world
from weft.io import canonical_hash
from weft.shared_world_evidence import build_shared_world_binding
from weft.shared_world_packaging import collect_shared_source_files


@pytest.fixture(autouse=True)
def no_native_or_external_work(monkeypatch):
    def forbidden(*_args, **_kwargs):
        pytest.fail("Feet candidate checks cannot run models, physics, network or subprocesses")

    monkeypatch.setattr(mujoco, "MjModel", SimpleNamespace(
        from_xml_string=forbidden, from_xml_path=forbidden, from_binary_path=forbidden))
    for name in ("MjData", "Renderer", "mj_forward", "mj_step", "mj_resetData", "mj_contactForce", "mj_kinematics"):
        monkeypatch.setattr(mujoco, name, forbidden)
    for module in (shared_world, shared_exercise_task, shared_exercise_feet_task, candidate.lab):
        monkeypatch.setattr(module, "SharedWorldEnv", forbidden)
    monkeypatch.setattr(shared_world, "human_subtree_mass", lambda _spec, scale=1.: 70. * scale**3)
    monkeypatch.setattr(socket, "socket", forbidden)
    monkeypatch.setattr(subprocess, "Popen", forbidden)
    monkeypatch.setattr(os, "system", forbidden)


def seal(root, files, *, status="ready"):
    for name, raw in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
    raw = candidate.lab._bytes({"schema_version": 1, "status": status,
                               "files": {name: candidate.lab._sha(raw) for name, raw in files.items()}})
    (root / "manifest.json").write_bytes(raw)
    return raw


def inputs(tmp_path, profile="shared-world-v3", *, lab_run=False, evidence_path=None):
    base, feet = scene_fixture(profile)
    base["learned_policy_available"] = True
    doc = document(partitions=("construction",) * 3)
    doc["objects"] = [item for item in doc["objects"] if item["id"] == "person"]
    for field in ("evidence", "observations"):
        doc[field] = [item for item in doc[field] if item["object_id"] == "person"]
    observed = candidate.lab._bytes(doc)
    sha = candidate.lab._sha(observed)
    refs = {"human": (sha, "person"), **{obj["id"]: None for obj in base["objects"]}}
    base["source_binding"], _ = build_shared_world_binding(
        {sha: observed}, source_sha256=base["human_spec"]["source_sha256"], world_ids=list(refs), entity_refs=refs)
    name = evidence_path or base["source_binding"]["evidence_documents"][sha]
    base["source_binding"]["evidence_documents"][sha] = name
    base = shared_world.validate_shared_world(base)
    options = {}
    if profile == "shared-world-v4":
        options["support_requirements"] = [{"object_id": "support_a", "human_segment": "left_forearm",
            "direction_world": [0., 0., 1.], "minimum_load_body_weight_fraction": .05,
            "start_step": 0, "end_step": 2}]
    elif profile == "shared-world-v2":
        options["contact_requirements"] = [{"object_id": "ball", "wrist": "left",
            "kind": "wrist_region_contact", "start_step": 0, "end_step": 2}]
    task = shared_exercise_task.build_shared_exercise_task(base, **options)
    root = tmp_path / "source" / "run" if lab_run else tmp_path / "source"
    files = {
        "scene.json": candidate.lab._bytes(base),
        "exercise-task.json": (json.dumps(task, separators=(",", ":")) + "\n\n").encode(),
        "source.json": b'{"scope":"Authored synthetic descriptor, not source media"}\n',
        "source-clock.json": b'{"kind":"authored fixture clock"}\n',
        "policy.json": b'{"inactive":"original sealed policy retained only under source-run"}\n',
        "avatar.json": b'{"inactive":"original sealed avatar retained only under source-run"}\n',
        "reconstruction.json": b'{"status":"constructed","physical_validation":"unverified"}\n',
        name: observed,
    }
    if profile == "shared-world-v3":
        files["inventory.json"] = candidate.lab._bytes(base["inventory"])
    elif profile == "shared-world-v4":
        files["inputs/support-config.json"] = candidate.lab._bytes({"scope": "Authored fixture priors", **options})
    manifest = seal(root, files)
    reference = tmp_path / "reference.json"
    reference.write_bytes(feet["foot_extension"]["reference_json"].encode())
    return root, reference, base, task, files, manifest


def checked_package(root, status):
    manifest_raw = (root / "manifest.json").read_bytes()
    manifest = json.loads(manifest_raw)
    assert manifest["status"] == status
    for name, sha in manifest["files"].items():
        assert candidate.lab._sha((root / name).read_bytes()) == sha
    return manifest, manifest_raw


@pytest.mark.parametrize("profile", ["shared-world-v2", "shared-world-v3", "shared-world-v4"])
def test_candidate_preserves_complete_source_task_and_runtime_with_explicit_rebinding(tmp_path, profile):
    source, reference, original, task, files, manifest_raw = inputs(tmp_path, profile)
    out = tmp_path / "candidate"
    result = candidate.prepare_shared_feet_candidate(source, reference, out)
    manifest, raw = checked_package(out, "constructed")
    assert result["manifest_sha256"] == candidate.lab._sha(raw)
    assert result["status"] == "constructed" and result["scope"]["simulation_verified"] is False
    assert result["scope"]["lab_created"] is False and result["scope"]["learning"] == "unsupported"
    assert (out / "source-run/manifest.json").read_bytes() == manifest_raw == (source / "manifest.json").read_bytes()
    for name, value in files.items():
        assert (out / "source-run" / name).read_bytes() == value == (source / name).read_bytes()
    assert (out / "input-task.json").read_bytes() == files["exercise-task.json"]
    assert (out / "foot-reference.json").read_bytes() == reference.read_bytes()
    base = json.loads((out / "base-scene.json").read_bytes())
    expected = deepcopy(original)
    expected["learned_policy_available"] = False
    assert base == expected
    rebound = json.loads((out / "base-exercise-task.json").read_bytes())
    expected_task = deepcopy(task)
    expected_task["scene_sha256"] = canonical_hash(base)
    assert rebound == expected_task and result["original_task_sha256"] == canonical_hash(task)
    actual = json.loads((out / "scene.json").read_bytes())
    outer = json.loads((out / "exercise-task.json").read_bytes())
    assert outer["base_task"] == rebound and actual["profile"] == "shared-world-v5"
    assert actual["base_profile"] == profile and actual["learned_policy_available"] is False
    assert actual["human_spec"] == original["human_spec"] and actual["task"] == original["task"]
    assert outer["interval"] == task["interval"] and outer["control"]["root_assistance"] == "disabled"
    assert outer["scope"]["ankle_metrics"] == "unscored_diagnostics"
    migration = json.loads((out / "feet-migration.json").read_bytes())
    assert migration["task_rebinding"] == {"field": "scene_sha256", "from": canonical_hash(original),
        "to": canonical_hash(base), "other_fields": "unchanged"}
    for name, sha in migration["implementation_sha256"].items():
        assert candidate.lab._sha((out / "candidate-implementation" / name).read_bytes()) == sha
    assert set(migration["runtime_sha256"]) == set(candidate.lab.runtime_files(actual))
    assert migration["adapter_implementation_sha256"] == outer["adapter_implementation_sha256"]
    evaluator = migration["evaluator_implementation_sha256"]
    assert migration["evaluator_sha256"] == (canonical_hash(evaluator) if profile == "shared-world-v4"
                                              else evaluator["shared_exercise_task.py"])
    assert migration["gates_sha256"] == task["gates_sha256"]
    for name in ("source.json", "source-clock.json", "reconstruction.json"):
        assert (out / name).read_bytes() == files[name]
    if profile == "shared-world-v3":
        assert (out / "inventory.json").read_bytes() == files["inventory.json"]
    if profile == "shared-world-v4":
        assert (out / "support-config.json").read_bytes() == files["inputs/support-config.json"]
        assert not (out / "inventory.json").exists()
    assert not any((out / name).exists() for name in candidate.INACTIVE)
    assert collect_shared_source_files(actual, out).report["source_evidence_verified"] is True
    assert result["files"] == len(manifest["files"]) + 1
    assert result["bytes"] == sum(path.stat().st_size for path in out.rglob("*") if path.is_file())
    assert set(manifest["files"]) == {path.relative_to(out).as_posix() for path in out.rglob("*")
                                      if path.is_file() and path.name != "manifest.json"} | {"source-run/manifest.json"}


def test_explicit_task_from_lab_run_keeps_both_original_bytes_and_root_choice(tmp_path):
    source, reference, scene, original_task, files, _ = inputs(tmp_path, lab_run=True)
    explicit = deepcopy(original_task)
    explicit["control"]["root_assistance"] = "explicit_3_axis"
    path = tmp_path / "explicit-task.json"
    raw = candidate.lab._bytes(explicit)
    path.write_bytes(raw)
    out = tmp_path / "candidate"
    candidate.prepare_shared_feet_candidate(source.parent, reference, out, task_path=path)
    assert (out / "input-task.json").read_bytes() == raw
    assert (out / "source-run/exercise-task.json").read_bytes() == files["exercise-task.json"]
    outer = json.loads((out / "exercise-task.json").read_bytes())
    assert outer["control"]["root_assistance"] == "explicit_3_axis"
    assert outer["base_task"]["control"] == explicit["control"]
    assert json.loads((out / "feet-migration.json").read_bytes())["task_input"] == "explicit_path"


@pytest.mark.parametrize("change", ["missing_task", "legacy_scene", "already_feet", "failed_source", "seed",
                                   "changed_gate", "missing_start_foot", "task_oversize", "reference_oversize"])
def test_invalid_inputs_fail_before_output_creation(tmp_path, change):
    source, reference, base, task, files, _ = inputs(tmp_path)
    if change == "missing_task":
        del files["exercise-task.json"]
    elif change == "legacy_scene":
        base["profile"] = "shared-world-v1"
        files["scene.json"] = candidate.lab._bytes(base)
    elif change == "already_feet":
        files["scene.json"] = candidate.lab._bytes(shared_world.add_feet_to_shared_world(base, reference.read_bytes()))
    elif change == "failed_source":
        files["failure.json"] = b'{"status":"failed"}'
    elif change == "seed":
        files["construction-seed.json"] = b"{}"
    elif change == "changed_gate":
        task["gates_sha256"] = "0" * 64
        files["exercise-task.json"] = candidate.lab._bytes(task)
    elif change == "missing_start_foot":
        value = json.loads(reference.read_bytes())
        value["frames"][0]["feet"]["left"].update(status="unavailable", reference=None, endpoint_geometry=None)
        reference.write_bytes(candidate.lab._bytes(value))
    elif change == "task_oversize":
        files["exercise-task.json"] += b" " * candidate.MAX_TASK_BYTES
    else:
        reference.write_bytes(b" " * (candidate.MAX_REFERENCE_BYTES + 1))
    seal(source, files)
    out = tmp_path / "candidate"
    with pytest.raises(ValueError):
        candidate.prepare_shared_feet_candidate(source, reference, out)
    assert not out.exists()


@pytest.mark.parametrize("name", ["candidate-implementation/observed.json", "policy.json"])
def test_source_proof_cannot_promote_reserved_output_paths(tmp_path, name):
    source, reference, *_ = inputs(tmp_path, evidence_path=name)
    with pytest.raises(ValueError, match="collides"):
        candidate.prepare_shared_feet_candidate(source, reference, tmp_path / "candidate")
    assert not (tmp_path / "candidate").exists()


def test_source_member_tampering_and_uncovered_proof_are_rejected(tmp_path):
    source, reference, scene, _, files, _ = inputs(tmp_path)
    evidence = next(iter(scene["source_binding"]["evidence_documents"].values()))
    (source / evidence).write_bytes(files[evidence] + b" ")
    with pytest.raises(ValueError, match="SHA256"):
        candidate.prepare_shared_feet_candidate(source, reference, tmp_path / "tampered")
    (source / evidence).write_bytes(files[evidence])
    files.pop(evidence)
    seal(source, files)
    with pytest.raises(ValueError, match="Outer manifest"):
        candidate.prepare_shared_feet_candidate(source, reference, tmp_path / "uncovered")
    assert not (tmp_path / "tampered").exists() and not (tmp_path / "uncovered").exists()


def test_output_safety_preserves_existing_files_and_rejects_overlap_or_symlink(tmp_path):
    source, reference, *_ = inputs(tmp_path)
    existing = tmp_path / "existing"
    existing.mkdir()
    (existing / "keep").write_bytes(b"untouched")
    with pytest.raises(FileExistsError):
        candidate.prepare_shared_feet_candidate(source, reference, existing)
    with pytest.raises(ValueError, match="outside"):
        candidate.prepare_shared_feet_candidate(source, reference, source / "nested")
    alias = tmp_path / "alias"
    alias.symlink_to(source, target_is_directory=True)
    with pytest.raises(ValueError, match="symlinks"):
        candidate.prepare_shared_feet_candidate(alias, reference, tmp_path / "candidate")
    assert (existing / "keep").read_bytes() == b"untouched"


def test_scoring_projection_may_change_only_policy_availability(tmp_path, monkeypatch):
    source, reference, *_ = inputs(tmp_path)
    derive = candidate.derive_base_scoring_scene
    def changed(scene):
        value = derive(scene)
        value["gravity"][2] += .01
        return value
    monkeypatch.setattr(candidate, "derive_base_scoring_scene", changed)
    with pytest.raises(ValueError, match="changed more than"):
        candidate.prepare_shared_feet_candidate(source, reference, tmp_path / "candidate")
    assert not (tmp_path / "candidate").exists()


@pytest.mark.parametrize("budget", ["files", "bytes"])
def test_complete_output_budget_rejects_without_partial_directory(tmp_path, monkeypatch, budget):
    source, reference, _, _, files, manifest = inputs(tmp_path)
    if budget == "files":
        monkeypatch.setattr(candidate, "MAX_FILES", len(files) + 3)
    else:
        monkeypatch.setattr(candidate, "MAX_BYTES", len(manifest) + sum(map(len, files.values())) + 1024)
    with pytest.raises(ValueError, match="complete source/runtime package"):
        candidate.prepare_shared_feet_candidate(source, reference, tmp_path / "candidate")
    assert not (tmp_path / "candidate").exists()


@pytest.mark.parametrize("timing", ["payload", "terminal_stage"])
def test_late_writes_preserve_failed_seal_without_constructed_terminal(tmp_path, monkeypatch, timing):
    source, reference, *_ = inputs(tmp_path)
    clock = [0.]
    monkeypatch.setattr(candidate.time, "monotonic", lambda: clock[0])
    write = candidate.write_package_files
    out = tmp_path / "candidate"
    def late(directory, files):
        write(directory, files)
        if ((timing == "payload" and Path(directory) == out and "source-run/manifest.json" in files)
                or timing == "terminal_stage" and Path(directory).name.startswith(".weft-feet-seal-")):
            clock[0] = 61.
    monkeypatch.setattr(candidate, "write_package_files", late)
    with pytest.raises(TimeoutError):
        candidate.prepare_shared_feet_candidate(source, reference, out, max_seconds=60)
    manifest, _ = checked_package(out, "failed")
    assert "failure.json" in manifest["files"]
    assert json.loads((out / "failure.json").read_bytes())["status"] == "failed"
    assert not list(tmp_path.glob(".weft-feet-seal-*"))


@pytest.mark.parametrize("changed", ["source", "reference", "output"])
def test_changed_bytes_before_seal_leave_verified_failure_members(tmp_path, monkeypatch, changed):
    source, reference, *_ = inputs(tmp_path)
    out = tmp_path / "candidate"
    write = candidate.write_package_files
    fired = [False]
    def tamper(directory, files):
        write(directory, files)
        if not fired[0] and Path(directory) == out and "source-verification.json" in files:
            fired[0] = True
            path = source / "source.json" if changed == "source" else reference if changed == "reference" else out / "scene.json"
            raw = path.read_bytes()
            path.write_bytes(raw[:-1] + b" ")
    monkeypatch.setattr(candidate, "write_package_files", tamper)
    with pytest.raises(ValueError, match="SHA256"):
        candidate.prepare_shared_feet_candidate(source, reference, out)
    manifest, _ = checked_package(out, "failed")
    if changed == "output":
        assert "scene.json" not in manifest["files"]
        assert json.loads((out / "failure.json").read_bytes())["unadmitted_changed_files"] == ["scene.json"]


@pytest.mark.parametrize("maximum", [True, 0, 181, float("nan")])
def test_invalid_time_bounds_require_no_inputs(tmp_path, maximum):
    with pytest.raises(ValueError, match="1–180"):
        candidate.prepare_shared_feet_candidate(tmp_path / "missing", tmp_path / "reference", tmp_path / "out",
                                                max_seconds=maximum)
    assert not (tmp_path / "out").exists()
