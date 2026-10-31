"""Saved crop proofs use analytical metadata only; no decode, perception or physics."""

from copy import deepcopy
import json
from pathlib import Path
import socket
import subprocess
import time
import urllib.request

import mujoco
import pytest

import test_pose_source_recovery as recovery_fixtures
from test_pose_source_recovery import proposal_case, seal as seal_proposals
from test_whole_body_reference_inputs import encode, sha
from test_whole_body_reference_refinement import fixture as analytical_body
from weft import exercise_vision, human, pose_source_recovery as recovery
from weft import whole_body_reference_inputs as inputs
from weft.root_motion import build_root_reference


@pytest.fixture(autouse=True)
def forbid_new_source_or_native_work(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Recovery input tests must not decode, infer, render, simulate, network or spawn")
    for name in ("VideoCapture", "VideoWriter", "imread", "imwrite", "imdecode", "imencode"):
        monkeypatch.setattr(exercise_vision.cv2, name, forbidden)
    for name in ("_detector", "inspect_exercise", "_save_previews"):
        monkeypatch.setattr(exercise_vision, name, forbidden)
    for name in ("HumanEnv", "build_human"):
        monkeypatch.setattr(human, name, forbidden)
    for name in ("mj_forward", "mj_step", "mj_resetData", "mj_resetDataKeyframe"):
        monkeypatch.setattr(mujoco, name, forbidden)
    monkeypatch.setattr(urllib.request, "urlopen", forbidden)
    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(subprocess, "Popen", forbidden)


def make_recovery_body(root, *, outcome="recover"):
    """Assemble an analytical body around the unchanged saved-proof fixture, without a body constructor."""
    proposals, video, baseline, selected = proposal_case(root.parent / (root.name + "-proposals"), outcome=outcome)
    producer = Path(inputs.__file__).parent / "exercise_vision.py"
    (proposals / "implementation-source").mkdir()
    (proposals / inputs._PRODUCER).write_bytes(producer.read_bytes())
    seal_proposals(proposals, profile="exercise-pose-crop-proposals-v1", status="source_evidence_only", simulation_ready=False)
    recovered = root.parent / (root.name + "-recovered")
    recovery.prepare_pose_source_recovery(proposals, recovered, source_video=video, max_seconds=10)
    seal_raw = (recovered / "manifest.json").read_bytes()
    recovered_manifest = json.loads(seal_raw)
    files = {name: (recovered / name).read_bytes() for name in recovered_manifest["files"]}
    spec, _ = analytical_body(5)
    usable = [row for row in selected["frames"] if row["status"] in {"tracked", "tracked_partial"}]
    spec["source_sha256"] = selected["source_sha256"]
    angles = dict.fromkeys(spec["reference"][0]["joints"], 0.)
    spec["reference"] = [{"time_s": row["time_s"], "joints": angles.copy()} for row in usable]
    spec["reference_evidence"] = [{"time_s": row["time_s"], "evidence_id": row["id"]} for row in usable]
    root_reference = build_root_reference(selected, torso_length=spec["lengths"]["torso"])
    spec["root_reference"] = [{"time_s": row["time_s"], "translation": row["translation"]} for row in root_reference["reference"]]
    files.update({"human.json": encode(spec), "observations.json": encode({"scope": "Analytical metadata; no body geometry admission"}),
        "source-clock.json": encode({"schema_version": 1, "source_sha256": selected["source_sha256"],
            "fps": selected["fps"], "metadata_frames": selected["frame_count"], "decoded_frames": selected["decoded_frames"],
            "nominal_basis": "frame_index_divided_by_fps", "samples": [
                {key: row[key] for key in ("frame", "time_s", "decoder_timestamp_s")} for row in selected["frames"]]}),
        "implementation.json": encode({"files": {"exercise_vision.py": sha(producer.read_bytes())}}),
        inputs._PRODUCER: producer.read_bytes(), "inputs/source-manifest.json": seal_raw})
    files["body-source.json"] = encode({"schema_version": 1, "profile": inputs.BODY_PROFILE,
        "status": "construction_only", "simulation_ready": False, "input_kind": "sealed_pose_evidence",
        "input_manifest_sha256": sha(seal_raw), "source_sha256": selected["source_sha256"], "source_artifact": "source.webm",
        "pose_options": {"pose_mode": "image", "pose_identity": "raw"},
        "construction": {"body_model": "mobile", "body_profile": "exercise-v3",
            "original_pose_config": "run-config.json", "original_pose_config_sha256": sha(files["run-config.json"])},
        "artifacts": {name: sha(files[name]) for name in inputs._REQUIRED | recovery.PROOF_FILES}})
    root.mkdir()
    for name, raw in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
    seal_body(root)
    return root


def seal_body(root, *, synchronize_proof=False):
    if synchronize_proof:
        path = root / "recovery/input-manifest.json"
        original = json.loads(path.read_bytes())
        for old, new in recovery.INPUT_FILES.items():
            original["files"][old] = sha((root / new).read_bytes())
        path.write_bytes(encode(original))
        path = root / "pose-recovery.json"
        marker = json.loads(path.read_bytes())
        marker["input_manifest_sha256"] = sha((root / "recovery/input-manifest.json").read_bytes())
        marker["artifacts"] = {name: sha((root / name).read_bytes()) for name in marker["artifacts"]}
        path.write_bytes(encode(marker))
    parent_path = root / "inputs/source-manifest.json"
    parent = json.loads(parent_path.read_bytes())
    parent["files"] = {name: sha((root / name).read_bytes()) for name in parent["files"]}
    parent_path.write_bytes(encode(parent))
    path = root / "body-source.json"
    marker = json.loads(path.read_bytes())
    marker["input_manifest_sha256"] = sha(parent_path.read_bytes())
    marker["artifacts"] = {name: sha((root / name).read_bytes()) for name in inputs._REQUIRED | recovery.PROOF_FILES}
    path.write_bytes(encode(marker))
    files = {path.relative_to(root).as_posix(): sha(path.read_bytes()) for path in root.rglob("*")
             if path.is_file() and path != root / "manifest.json"}
    (root / "manifest.json").write_bytes(encode({"schema_version": 1, "profile": inputs.BODY_PROFILE,
        "status": "construction_only", "simulation_ready": False, "files": files}))


def mutate(root, name, change, *, synchronize_proof=True):
    path = root / name
    document = json.loads(path.read_bytes())
    change(document)
    path.write_bytes(encode(document))
    seal_body(root, synchronize_proof=synchronize_proof)


def prepare(root, **kwargs):
    return inputs.prepare_reference_inputs(root, deadline=time.monotonic() + 10, **kwargs)


@pytest.mark.parametrize("outcome,accepted", [("recover", [30]), ("unchanged", []), ("reject_crop", [])])
def test_recovery_validates_once_and_preserves_selected_targets_raw_unknowns_and_proof(tmp_path, monkeypatch, outcome, accepted):
    root = make_recovery_body(tmp_path / "body", outcome=outcome)
    original = {name: (root / name).read_bytes() for name in json.loads((root / "manifest.json").read_bytes())["files"]}
    counts = {"validate": 0, "recompute": 0}
    validate, recompute = recovery.validate_pose_recovery_files, recovery._recompute
    def counted_validate(*args, **kwargs):
        counts["validate"] += 1
        return validate(*args, **kwargs)
    def counted_recompute(*args, **kwargs):
        counts["recompute"] += 1
        return recompute(*args, **kwargs)
    monkeypatch.setattr(recovery, "validate_pose_recovery_files", counted_validate)
    monkeypatch.setattr(recovery, "_recompute", counted_recompute)
    snapshot = prepare(root)
    inputs.recheck_reference_inputs(snapshot, deadline=time.monotonic() + 10)
    assert counts == {"validate": 1, "recompute": 1}
    descriptor = snapshot["identity"]["recovery"]
    assert descriptor["summary"]["accepted_frames"] == accepted
    assert descriptor["summary"] == json.loads(original["pose-recovery.json"])["result"]
    assert descriptor["method"] == "single-view-selective-crop-v1"
    assert descriptor["media"]["bytes"] == len(original["source.webm"])
    assert descriptor["media"]["sha256"] == sha(original["source.webm"])
    assert "no video decode or new inference" in snapshot["identity"]["media_verification"]
    assert set(descriptor["verifier_runtime"]["files"]) == set(recovery.IMPLEMENTATION_FILES)
    for name in recovery.PROOF_FILES | recovery.POSE_FILES | {"source.webm"}:
        target = "inputs/" + ("evidence-original.json" if name == "evidence.json" else name)
        assert snapshot["files"][target] == original[name]
        assert (root / name).read_bytes() == original[name]
    derived = json.loads(snapshot["files"]["inputs/evidence-derived.json"])
    derived.pop("image_dimension_provenance")
    selected = json.loads(original["evidence.json"])
    assert derived == selected
    frame = next(row for row in derived["frames"] if row["frame"] == 30)
    if accepted:
        assert frame["landmarks"] != frame["image_refinement_raw"]["landmarks"]
        assert frame["image_refinement"]["anatomical_identity_verified"] is False
        assert frame["image_refinement_raw"] == next(row for row in json.loads(original["pose-landmarks.json"])["frames"] if row["frame"] == 30)
    else:
        assert "image_refinement" not in frame
    assert next(row for row in derived["frames"] if row["frame"] == 14)["decoder_timestamp_s"] is None
    assert snapshot["files"]["inputs/human.json"] == original["human.json"]


@pytest.mark.parametrize("name", ["pose-recovery.json", "recovery/candidates.jsonl", "recovery/decisions.json",
                                   "recovery/baseline-evidence.json", "recovery/implementation-source/pose_crop.py"])
def test_missing_recovery_member_fails_before_validation(tmp_path, monkeypatch, name):
    root = make_recovery_body(tmp_path / "body")
    (root / name).unlink()
    monkeypatch.setattr(recovery, "validate_pose_recovery_files", lambda *a, **k: pytest.fail("Incomplete proof reached verifier"))
    with pytest.raises((ValueError, OSError)):
        prepare(root)


@pytest.mark.parametrize("name,change", [
    ("recovery/plans.json", lambda d: d["plans"][15].update(eligible=False)),
    ("recovery/decisions.json", lambda d: d[0].update(accepted=False)),
    ("recovery/baseline-evidence.json", lambda d: d["frames"][15].update(decoder_timestamp_s=1.0)),
    ("pose-recovery.json", lambda d: d["result"].update(accepted_frames=[])),
    ("evidence.json", lambda d: d["frames"][15]["landmarks"][11].update(x=.1)),
    ("evidence.json", lambda d: d["frames"][15]["image_refinement_raw"].update(status="tracked")),
    ("recovery/input-manifest.json", lambda d: d.update(status="failed")),
    ("recovery/input-manifest.json", lambda d: d["files"].update({inputs._PRODUCER: "c" * 64})),
    ("recovery/implementation.json", lambda d: d["files"].update({"pose_crop.py": "c" * 64})),
])
def test_resealed_conflicting_derivations_are_not_trusted(tmp_path, name, change):
    root = make_recovery_body(tmp_path / "body")
    mutate(root, name, change)
    with pytest.raises(ValueError):
        prepare(root)


@pytest.mark.parametrize("kind", ["mapping", "duplicate", "missing", "decision", "oversized_line"])
def test_resealed_crop_journal_cannot_skip_exact_recomputation(tmp_path, kind):
    root = make_recovery_body(tmp_path / "body")
    path = root / "recovery/candidates.jsonl"
    lines = path.read_bytes().splitlines()
    if kind == "mapping":
        row = json.loads(lines[0]);row["proposed"]["landmarks"][11]["x"] += .1
        lines[0] = json.dumps(row).encode()
    elif kind == "duplicate":
        lines *= 2
    elif kind == "missing":
        lines = []
    elif kind == "decision":
        row = json.loads(lines[0]);row["decision"]["accepted"] = False
        lines[0] = json.dumps(row).encode()
    else:
        lines = [b" " * (recovery.MAX_LINE_BYTES + 1)]
    path.write_bytes(b"\n".join(lines) + (b"\n" if lines else b""))
    seal_body(root, synchronize_proof=True)
    with pytest.raises(ValueError):
        prepare(root)


def test_recovery_media_bytes_are_required_even_when_metadata_hashes_claim_success(tmp_path):
    root = make_recovery_body(tmp_path / "body")
    (root / "source.webm").write_bytes(b"changed source, not another decode")
    with pytest.raises(ValueError, match="SHA256"):
        prepare(root)


def test_media_member_cap_rejects_oversize_before_read(tmp_path, monkeypatch):
    root = make_recovery_body(tmp_path / "body")
    assert inputs.MAX_MEDIA_BYTES == 64 * 1024 * 1024
    monkeypatch.setattr(inputs, "MAX_MEDIA_BYTES", (root / "source.webm").stat().st_size - 1)
    with pytest.raises(ValueError, match="bounded regular file"):
        prepare(root)


def test_nine_mib_original_media_uses_only_the_recovery_allowance_and_rechecks(tmp_path, monkeypatch):
    monkeypatch.setattr(recovery_fixtures, "VIDEO", b"Analytical source, not decoded.\n" + b"." * (9 * 1024 * 1024))
    root = make_recovery_body(tmp_path / "body")
    snapshot = prepare(root)
    media = snapshot["files"]["inputs/source.webm"]
    assert inputs.MAX_JSON_BYTES < len(media) < inputs.MAX_MEDIA_BYTES
    assert snapshot["identity"]["recovery"]["media"]["sha256"] == sha(media)
    assert snapshot["identity"]["recovery"]["bounds"] == {"metadata_member_bytes": 8 * 1024 * 1024,
        "metadata_total_bytes": 32 * 1024 * 1024, "original_media_bytes": 64 * 1024 * 1024,
        "recovery_total_bytes": 96 * 1024 * 1024}
    monkeypatch.setattr(recovery, "validate_pose_recovery_files", lambda *a, **k: pytest.fail("Recheck reran proof"))
    inputs.recheck_reference_inputs(snapshot, deadline=time.monotonic() + 10)


def test_recovery_media_allowance_cannot_enlarge_metadata_members(tmp_path):
    root = make_recovery_body(tmp_path / "body")
    with (root / "observations.json").open("wb") as stream:
        stream.truncate(inputs.MAX_JSON_BYTES + 1)
    with pytest.raises(ValueError, match="bounded regular file"):
        prepare(root)


def test_recovery_total_read_cap_is_enforced_in_prepare_and_recheck(tmp_path, monkeypatch):
    root = make_recovery_body(tmp_path / "body")
    snapshot = prepare(root)
    monkeypatch.setattr(inputs, "MAX_RECOVERY_BYTES", snapshot["identity"]["selected_source_bytes"] - 1)
    with pytest.raises(ValueError, match="bounded regular file"):
        prepare(root)
    with pytest.raises(ValueError, match="bounded regular file"):
        inputs.recheck_reference_inputs(snapshot, deadline=time.monotonic() + 10)


def test_source_and_verifier_reads_share_the_existing_aggregate_cap(tmp_path, monkeypatch):
    root = make_recovery_body(tmp_path / "body")
    manifest = json.loads((root / "manifest.json").read_bytes())
    source_size = (root / "manifest.json").stat().st_size + sum((root / name).stat().st_size for name in manifest["files"])
    monkeypatch.setattr(inputs, "MAX_INPUT_BYTES", source_size + 1)
    with pytest.raises(ValueError, match="bounded regular file"):
        prepare(root)


def test_recovery_deadline_failure_is_propagated_without_a_snapshot(tmp_path, monkeypatch):
    root = make_recovery_body(tmp_path / "body")
    def expired(*args, **kwargs):
        assert kwargs["deadline"] > time.monotonic()
        raise TimeoutError("Saved derivation deadline exceeded")
    monkeypatch.setattr(recovery, "validate_pose_recovery_files", expired)
    with pytest.raises(TimeoutError, match="Saved derivation"):
        prepare(root)


def test_proof_cannot_use_a_different_loaded_verifier_root(tmp_path, monkeypatch):
    root = make_recovery_body(tmp_path / "body")
    monkeypatch.setattr(recovery, "__file__", str(tmp_path / "unrelated.py"))
    with pytest.raises(ValueError, match="loaded outside"):
        prepare(root)


def test_runtime_code_pins_detect_change_without_replaying_proof(tmp_path, monkeypatch):
    root = make_recovery_body(tmp_path / "body")
    snapshot = prepare(root)
    monkeypatch.setattr(recovery, "validate_pose_recovery_files", lambda *a, **k: pytest.fail("Recheck reran proof"))
    monkeypatch.setattr(recovery, "_recompute", lambda *a, **k: pytest.fail("Recheck reran decisions"))
    original_read = inputs._Reader.read
    def changed_code(self, name, expected=None, maximum=inputs.MAX_JSON_BYTES, *, media=False):
        if self.root == Path(inputs.__file__).parent and name == "pose_crop.py":
            raise ValueError("Selected input differs from its sealed SHA256")
        return original_read(self, name, expected, maximum, media=media)
    monkeypatch.setattr(inputs._Reader, "read", changed_code)
    with pytest.raises(ValueError, match="SHA256"):
        inputs.recheck_reference_inputs(snapshot, deadline=time.monotonic() + 10)


@pytest.mark.parametrize("name", ["source.webm", "recovery/candidates.jsonl", "recovery/input-manifest.json"])
def test_final_recheck_covers_media_and_complete_recovery_proof(tmp_path, name):
    root = make_recovery_body(tmp_path / "body")
    snapshot = prepare(root)
    with (root / name).open("ab") as stream:
        stream.write(b" ")
    with pytest.raises(ValueError, match="SHA256"):
        inputs.recheck_reference_inputs(snapshot, deadline=time.monotonic() + 10)


def test_partial_or_expired_validation_never_returns_a_verified_snapshot(tmp_path, monkeypatch):
    root = make_recovery_body(tmp_path / "body")
    with pytest.raises(TimeoutError):
        inputs.prepare_reference_inputs(root, deadline=time.monotonic() - 1)
    monkeypatch.setattr(recovery, "validate_pose_recovery_files", lambda *a, **k: {})
    with pytest.raises(ValueError, match="Validated crop evidence differs"):
        prepare(root)
