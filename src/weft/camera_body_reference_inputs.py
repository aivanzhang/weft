"""Selected saved camera-body custody, without reconstruction or native admission."""

from collections import Counter
import os
from pathlib import Path
import re

from weft.camera_shared_candidate import _qualified_inputs
from weft.whole_body_reference_inputs import (
    _Reader, _bytes, _canonical, _check, _json, _manifest, _require, _root_fd, _root_path, _sha,
)


PROFILE = "camera-body-reference-inputs-v1"
MAX_MANIFEST_BYTES, MAX_MEMBER_BYTES, MAX_COPY_BYTES, MAX_COPY_FILES = 256 << 10, 8 << 20, 32 << 20, 18
ALIASES = {
    "human.json": "inputs/human.json",
    "evidence.json": "inputs/evidence-original.json",
    "camera.json": "inputs/source-camera.json",
    "ground.json": "inputs/ground.json",
    "two-view-triangulation.json": "inputs/two-view-triangulation.json",
    "two-view-root.json": "inputs/two-view-root.json",
    "inputs/two-view-root/evidence.json": "inputs/secondary-evidence.json",
    "inputs/two-view-root/camera.json": "inputs/secondary-camera.json",
    "inputs/two-view-root/config.json": "inputs/two-view-config.json",
    "source.json": "inputs/source.json",
    "source-input-manifest.json": "inputs/source-manifest.json",
    "camera-provenance.json": "inputs/camera-provenance.json",
}
_CANDIDATE = {
    "source-run/manifest.json": "inputs/camera-source-manifest.json",
    "source-verification.json": "inputs/camera-source-verification.json",
}
_OUTCOMES = {"policy.json", "exercise-task.json", "lab-config.json", "lab-contract.json",
             "training.json", "evaluation.json", "session.json", "failure.json"}
_SHA = re.compile(r"[0-9a-f]{64}\Z")
_MEDIA = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,159}\Z")


def _producer_manifest(value, members):
    _require(type(value.get("schema")) is int and value["schema"] == 1
             and "profile" not in value and value.get("status") is None
             and set(ALIASES) <= members.keys() and {"scene.json", "task.json", "human.xml", "camera-reference.json"} <= members.keys()
             and not (_OUTCOMES & members.keys()), "An original sealed camera-body producer package is required")


def _source_proof(selected, body, declared, body_manifest_sha):
    source = _json(selected["source.json"])
    _, upstream_members = _manifest(selected["source-input-manifest.json"])
    provenance = _json(selected["camera-provenance.json"])
    media = source.get("artifact")
    _require(type(media) is str and _MEDIA.fullmatch(media) and media not in ALIASES
             and media not in {"manifest.json", "failure.json", "candidate.json"}
             and declared.get(media) == upstream_members.get(media) == source.get("sha256") == body["source_sha256"],
             "Primary media identity must remain sealed; media bytes are not read")
    for name in ("evidence.json", "source.json"):
        _require(upstream_members.get(name) == _sha(selected[name]), "Upstream source bytes differ: " + name)
    verification = provenance.get("source_verification")
    _require(type(verification) is dict and verification.get("manifest_sha256") == _sha(selected["source-input-manifest.json"])
             and provenance.get("camera_file_sha256") == _sha(selected["camera.json"])
             and provenance.get("ground_file_sha256") == _sha(selected["ground.json"])
             and provenance.get("new_training") is False and provenance.get("new_inference") is False
             and provenance.get("reference_joints_used_for_fitting") is False
             and provenance.get("monocular_camera_reference_unchanged") is True,
             "Camera producer provenance differs from its exact retained inputs")
    copied = provenance.get("copied_source_files")
    _require(type(copied) is dict and all(copied.get(name) == name for name in (media, "source.json", "evidence.json")),
             "Camera producer source-copy attribution differs")
    second = provenance.get("two_view_inputs")
    _require(type(second) is dict and set(second) == {"evidence.json", "camera.json", "config.json"}
             and all(type(second[name]) is dict and second[name].get("sha256") ==
                     _sha(selected["inputs/two-view-root/" + name]) for name in second),
             "Secondary camera/evidence/config raw-byte pins differ")
    return {"camera_body_manifest_sha256": body_manifest_sha,
            "upstream_source_manifest_sha256": _sha(selected["source-input-manifest.json"]),
            "source_artifact": media, "source_sha256": body["source_sha256"],
            "primary_rights": {"status": "retained_declaration_not_new_verification", "value": source["rights"],
                               "source_member": "source.json", "sha256": _sha(selected["source.json"])},
            "secondary_media_and_rights": "unavailable; only supplied evidence, camera and config are verified",
            "historical_root_input_body": "Canonical historical body/root pins retained in the validated binding; historical body bytes are not read",
            "media_verification": "Declared sealed identity only; no media bytes read or decoded"}


def prepare_camera_reference_inputs(source, *, expected_manifest_sha256=None, deadline):
    """Read only bounded selected proof members; retain raw evidence without overlays."""
    _check(deadline)
    _require(expected_manifest_sha256 is None or type(expected_manifest_sha256) is str
             and _SHA.fullmatch(expected_manifest_sha256), "Expected camera-source manifest must be an exact SHA256")
    root = _root_path(source)
    _require(len(root.parts) <= 64, "Camera source path exceeds its depth bound")
    reader = _Reader(root, deadline)
    manifest_raw = reader.read("manifest.json", expected_manifest_sha256, MAX_MANIFEST_BYTES)
    manifest, declared = _manifest(manifest_raw)
    _require(not (_OUTCOMES & declared.keys()), "Active lab, policy or outcome bundles are not camera-body inputs")
    candidate = manifest.get("profile") == "camera-shared-candidate-v1"
    if candidate:
        _require(type(manifest.get("schema_version")) is int and manifest["schema_version"] == 1
                 and manifest.get("status") == "constructed" and set(ALIASES) | set(_CANDIDATE) <= declared.keys(),
                 "A sealed constructed camera-shared candidate is required")
    else:
        _producer_manifest(manifest, declared)
    names = {**ALIASES, **(_CANDIDATE if candidate else {})}
    selected = {name: reader.read(name, declared[name], MAX_MANIFEST_BYTES if name.endswith("manifest.json") else MAX_MEMBER_BYTES)
                for name in names}
    original_body_manifest_sha = _sha(manifest_raw)
    body_members = declared
    if candidate:
        body_manifest_raw = selected["source-run/manifest.json"]
        body_manifest, body_members = _manifest(body_manifest_raw)
        _producer_manifest(body_manifest, body_members)
        original_body_manifest_sha = _sha(body_manifest_raw)
        custody = _json(selected["source-verification.json"])
        _require(custody.get("source_manifest_sha256") == original_body_manifest_sha
                 and custody.get("source_manifest_members") == len(body_members), "Candidate source seal attribution differs")
        records = custody.get("selected_members")
        _require(type(records) is dict and all(records.get(name) == {
            "candidate_member": name, "sha256": _sha(selected[name]), "bytes": len(selected[name])}
            and body_members.get(name) == _sha(selected[name]) for name in ALIASES),
            "Candidate selected proof inventory differs from original camera-body bytes")
    body = _json(selected["human.json"])
    _require(not any("refinement" in key or "alignment" in key for key in body),
             "Already-refined or realigned camera bodies cannot be stacked")
    _check(deadline)
    body, coupling, evidence, views, slots = _qualified_inputs({name: selected[name] for name in ALIASES
        if name not in {"source-input-manifest.json", "camera-provenance.json"}}, deadline)
    _require(len(body["reference"]) >= 3, "Camera reference fitting requires at least three original poses")
    attribution = _source_proof(selected, body, body_members, original_body_manifest_sha)
    _require(declared.get(attribution["source_artifact"]) == body["source_sha256"], "Candidate primary media seal differs")
    _check(deadline)
    camera = _json(selected["camera.json"])
    raw_pins = {name: {"sha256": _sha(selected[name]), "bytes": len(selected[name]), "copied_member": alias}
                for name, alias in names.items()}
    proof = {"schema_version": 1, "profile": PROFILE, "status": "metadata_verified",
        "input_manifest_sha256": _sha(manifest_raw), "attribution": attribution,
        "selected_members": raw_pins, "camera_coupling": coupling, "views": views,
        "source_slots": {"count": len(slots), "body_status_counts": dict(Counter(row["body_status"] for row in slots)),
                         "custody": "All original slots retained in exact triangulation/root reports; no filling or retiming"},
        "dimension_derivation": "none; exact original evidence bytes",
        "scope": "Selected metadata and internal camera/root/ground bindings only; no authenticated decoder, physical validity or simulator admission"}
    proof_raw = _bytes(proof)
    _require(len(proof_raw) <= MAX_MEMBER_BYTES, "Camera selected-input proof exceeds its byte bound")
    files = {alias: selected[name] for name, alias in names.items()}
    files.update({"inputs/body-manifest.json": manifest_raw,
                  "inputs/evidence-derived.json": selected["evidence.json"],
                  "inputs/camera-inputs-proof.json": proof_raw})
    _require(len(files) <= MAX_COPY_FILES and sum(map(len, files.values())) <= MAX_COPY_BYTES,
             "Camera selected-input copies exceed their file or byte bound")
    identity = {"schema_version": 1, "profile": PROFILE, "status": "metadata_verified",
        "input_kind": "sealed_camera_shared_candidate" if candidate else "sealed_camera_body",
        "source_root": {"path": str(reader.root), "device": reader.identity[0], "inode": reader.identity[1]},
        "body_manifest_sha256": _sha(manifest_raw), "expected_manifest_supplied": expected_manifest_sha256 is not None,
        "source_manifest_sha256": attribution["upstream_source_manifest_sha256"],
        "camera_body_manifest_sha256": original_body_manifest_sha,
        "source_sha256": body["source_sha256"], "source_artifact": attribution["source_artifact"],
        "human_sha256": _sha(selected["human.json"]), "human_canonical_sha256": _canonical(body),
        "evidence_original_sha256": _sha(selected["evidence.json"]),
        "evidence_derived_sha256": _sha(selected["evidence.json"]), "evidence_canonical_sha256": _canonical(evidence),
        "source_camera_sha256": _sha(selected["camera.json"]), "source_camera_canonical_sha256": _canonical(camera),
        "camera_coupling_sha256": _canonical(coupling), "source_clock_sha256": coupling["source_clock_sha256"],
        "ground_canonical_sha256": coupling["ground_sha256"], "root_reference_canonical_sha256": coupling["root_reference_sha256"],
        "shape_canonical_sha256": coupling["shape_sha256"], "clock": coupling["clock"], "body_dt_s": body["dt"],
        "selected_source_files": len(reader.pins), "selected_source_bytes": reader.total,
        "copied_input_bytes": sum(map(len, files.values())), "copied_files": {name: _sha(raw) for name, raw in files.items()},
        "selected_members": dict(reader.pins), "proof_member": "inputs/camera-inputs-proof.json",
        "media_verification": attribution["media_verification"], "dimension_derivation": proof["dimension_derivation"],
        "scope": proof["scope"]}
    snapshot = {"files": files, "identity": identity,
                "pins": [(reader.root / name, digest) for name, digest in reader.pins.items()]}
    _check(deadline)
    fd, _ = _root_fd(reader.root, reader.identity)
    os.close(fd)
    return snapshot


def recheck_camera_reference_inputs(snapshot, *, deadline):
    """Recheck only consumed source members and exact copies before publication."""
    _check(deadline)
    _require(type(snapshot) is dict and set(snapshot) == {"files", "identity", "pins"}, "Prepared camera inputs required")
    identity = snapshot["identity"]
    _require(type(identity) is dict and identity.get("profile") == PROFILE, "Camera input profile differs")
    root_identity = identity["source_root"]
    root = _root_path(root_identity["path"])
    reader = _Reader(root, deadline, (root_identity["device"], root_identity["inode"]))
    selected, pins, files = identity["selected_members"], snapshot["pins"], snapshot["files"]
    _require(type(selected) is dict and 1 <= len(selected) <= len(ALIASES) + len(_CANDIDATE) + 1
             and type(pins) is list and len(pins) == len(selected)
             and all(isinstance(path, Path) and path.is_absolute() and type(digest) is str and _SHA.fullmatch(digest)
                     for path, digest in pins), "Invalid selected camera input pins")
    actual = {path.relative_to(root).as_posix(): digest for path, digest in pins}
    _require(actual == selected and len(actual) == len(pins) and actual.get("manifest.json") == identity["body_manifest_sha256"],
             "Selected camera source pin inventory changed")
    _require(type(files) is dict and set(files) == set(identity["copied_files"]) and len(files) <= MAX_COPY_FILES
             and all(type(raw) is bytes and len(raw) <= MAX_MEMBER_BYTES and _sha(raw) == identity["copied_files"][name]
                     for name, raw in files.items()) and sum(map(len, files.values())) <= MAX_COPY_BYTES,
             "Prepared camera input bytes changed")
    for name, digest in actual.items():
        reader.read(name, digest, MAX_MANIFEST_BYTES if name.endswith("manifest.json") else MAX_MEMBER_BYTES)
    fd, _ = _root_fd(root, reader.identity)
    os.close(fd)
    _check(deadline)
