"""Read-only admission of source-linked spatial labs for the ordinary portal importer."""

from hashlib import sha256
import json
import math
from pathlib import Path
import re
import stat
import time

from weft.io import canonical_hash
from weft.shared_world_packaging import read_package_file, read_package_manifest, validate_package_layout
from weft.spatial_scene_inputs import _json, prepare_spatial_scene_inputs


MAX_FILES = 4096
MAX_BYTES = 400 << 20
MAX_FILE_BYTES = 64 << 20
METADATA_RESERVATION_BYTES = 1 << 20
MAX_SOURCE_SCOPE_BYTES = 32 << 10
CORE_MODULES = ("spatial_rigid.py", "spatial_learning.py", "spatial_lab.py")
RUNTIME_ROOT = Path(__file__).parent
_MEDIA = re.compile(r"source\.(mp4|webm|mov|m4v|avi|mkv)\Z")


def _require(condition, message):
    if not condition:
        raise ValueError("Spatial portal import: " + message)


def _signature(path):
    _require(not any(p.is_symlink() for p in (path, *path.parents)), "linked package path")
    info = path.stat()
    _require(stat.S_ISREG(info.st_mode), "package members must be regular files")
    return info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns


def inspect_spatial_import(source, config_raw, manifest_raw, files, *, deadline):
    """Validate saved source custody without decoding, training or constructing a model."""
    _require(type(deadline) in (int, float) and math.isfinite(deadline), "finite deadline required")

    def check():
        if time.monotonic() >= deadline:
            raise TimeoutError("Spatial portal import exceeded its deadline")

    check()
    source = Path(source).absolute()
    _require(".." not in source.parts, "source traversal is unsupported")
    run = source / "run"
    failure_paths = (source / "failure.json", run / "failure.json", run / "inputs/source-run/failure.json")
    _require(not any(p.exists() or p.is_symlink() for p in failure_paths), "preserved failure marker")
    config = _json(config_raw, 64 << 10)
    _require(config.get("profile") == "spatial-rigid" and config.get("control_profile") == "spatial_rigid"
             and config.get("physical_validation") == "unverified"
             and config.get("source_video_present") is True, "source-linked spatial lab required")
    actual_raw, actual_files = read_package_manifest(run)
    _require(actual_raw == manifest_raw and dict(actual_files) == dict(files), "manifest changed")
    _require(1 <= len(files) <= MAX_FILES and len(manifest_raw) <= 1 << 20, "package inventory bound exceeded")
    validate_package_layout([*files, "manifest.json"])
    signatures, total = {}, 0
    for name in [*files, "manifest.json"]:
        check()
        signature = _signature(run / name)
        _require(signature[2] <= MAX_FILE_BYTES, "package member exceeds 64 MiB")
        total += signature[2]
        _require(total <= MAX_BYTES, "package exceeds 400 MiB")
        signatures[name] = signature
    config_signature = _signature(source / "lab-config.json")
    _require(read_package_file(source, "lab-config.json", max_bytes=64 << 10) == config_raw, "config changed")

    def read(name, limit=1 << 20):
        check()
        _require(name in files, "required member is not sealed: " + name)
        return read_package_file(run, name, max_bytes=limit, expected_sha256=files[name])

    def document(name, limit=1 << 20):
        return _json(read(name, limit), limit)

    # Reuse ordinary spatial admission; snapshots are data, never executable code.
    from weft.spatial_lab import _read_spatial_source, _validate_policy_bundle
    from weft.spatial_rigid import validate_scene

    verified, checked_raw, checked_files, _, scene, report = _read_spatial_source(run)
    check()
    _require(checked_raw == manifest_raw and dict(checked_files) == dict(files), "verified manifest changed")
    validate_scene(scene)
    _require(report is not None and report.get("status") == "ready_spatial_lab", "completed spatial lab required")
    provenance = document("lab-provenance.json")
    scene_hash = canonical_hash(scene)
    _require(provenance.get("scene_sha256") == scene_hash, "lab provenance scene differs")
    core = document("implementation.json")
    for name in CORE_MODULES:
        current = read_package_file(RUNTIME_ROOT, name, max_bytes=2 << 20)
        _require(core.get(name) == sha256(current).hexdigest()
                 and files.get("implementation-source/" + name) == core[name], "numeric runtime changed: " + name)

    original_raw = read("source-input-manifest.json")
    original = _json(original_raw, 1 << 20).get("files")
    _require(type(original) is dict and 1 <= len(original) <= MAX_FILES, "source archive manifest required")
    validate_package_layout([*original, "manifest.json"])
    original_pin = sha256(original_raw).hexdigest()
    _require(provenance.get("source_verification", {}).get("manifest_sha256") == original_pin,
             "source archive seal differs from lab provenance")
    prefix = "inputs/source-run/"
    _require(provenance.get("copied_source_files") == {name: prefix + name for name in original}
             and all(files.get(prefix + name) == pin for name, pin in original.items()), "incomplete source archive")
    _require(document(prefix + "scene.json") == scene, "source scene differs from lab scene")
    candidate = document(prefix + "candidate.json")
    construction = document(prefix + "report.json")
    _require(candidate.get("profile") == "spatial-scene-construction-v1"
             and candidate.get("status") == construction.get("status") == "constructed_hypothesis"
             and candidate.get("scene") == scene and construction.get("candidate_sha256") == canonical_hash(candidate)
             and construction.get("scene_sha256") == scene_hash, "source construction join differs")
    receipt = document(prefix + "inputs/input-receipt.json")
    planar_prefix = prefix + "inputs/planar/"
    planar_raw = read(planar_prefix + "manifest.json")
    planar = _json(planar_raw, 1 << 20).get("files")
    _require(type(planar) is dict and all(files.get(planar_prefix + n) == p for n, p in planar.items()),
             "planar archive differs from outer seal")
    planar_pin = sha256(planar_raw).hexdigest()
    _require(receipt.get("source_manifest_sha256") == construction.get("source_manifest_sha256") == planar_pin,
             "planar source identity differs")
    authored_raw = read(prefix + "inputs/authored-config.json", 64 << 10)
    prepared = prepare_spatial_scene_inputs(run / planar_prefix, authored_raw,
        scene_artifact=receipt.get("scene_artifact"), expected_manifest_sha256=planar_pin,
        max_seconds=min(10, max(.001, deadline - time.monotonic())))
    check()
    _require(_json(prepared.scene_bytes, 250_000) == scene
             and read(prefix + "inputs/original-scene.json") == prepared.original_scene_bytes
             and receipt.get("config_file_sha256") == sha256(authored_raw).hexdigest()
             and receipt.get("original_scene_file_sha256") == sha256(prepared.original_scene_bytes).hexdigest(),
             "authored construction does not reproduce the saved scene")

    metadata = document("source.json")
    media = metadata.get("artifact")
    source_hash = metadata.get("sha256")
    _require(type(media) is str and _MEDIA.fullmatch(media) and files.get(media) == source_hash
             and original.get(media) == source_hash
             and document(prefix + "source.json") == metadata, "sealed source video binding differs")
    _require(sum(bool(_MEDIA.fullmatch(n)) for n in files) == 1, "one original source video required")
    _require(provenance.get("source_video_artifact") == media, "source presentation differs")
    from weft.spatial_scene_pipeline import _media_aliases

    aliases, _ = _media_aliases(prepared, deadline)
    _require(aliases.get("source.json") == read("source.json")
             and sha256(aliases.get(media, b"")).hexdigest() == source_hash,
             "original planar source presentation differs")
    evidence = document("evidence.json", 8 << 20)
    _require(evidence.get("source_sha256") == source_hash and files.get("evidence.json") == original.get("evidence.json")
             and aliases.get("evidence.json") == read("evidence.json", 8 << 20), "original evidence differs")

    learned = config.get("learned_policy_available")
    _require(type(learned) is bool and report.get("learned_policy_available") is learned
             and provenance.get("learned_policy_available") is learned, "policy availability differs")
    if learned:
        _validate_policy_bundle(files, document, scene)
        policy_raw = read("learning-input-manifest.json")
        policy_files = _json(policy_raw, 1 << 20).get("files", {})
        _require(provenance.get("policy_verification", {}).get("manifest_sha256") == sha256(policy_raw).hexdigest(),
                 "policy source seal differs")
        _require(all(policy_files.get(n) == files.get(n) for n in
                     ("scene.json", "policy.json", "training.json", "rl-evaluation.json", "policy-rollout.json"))
                 and all(policy_files.get("implementation/" + n) == core[n] for n in CORE_MODULES),
                 "policy source or numeric runtime differs")
    else:
        _require("policy.json" not in files, "unadvertised policy is unsupported")

    for name, signature in signatures.items():
        check()
        _require(_signature(run / name) == signature, "package changed during admission")
    _require(_signature(source / "lab-config.json") == config_signature, "config changed during admission")
    _require(not any(p.exists() or p.is_symlink() for p in failure_paths), "source failed during admission")
    check()
    coverage = evidence.get("reconstruction_coverage", {})
    coverage_keys = ("method", "status", "requested_interval", "model_interval", "requested_frames",
                     "simulation_frames", "observed_frames", "missing_observation_slots", "scope")
    scope = {"profile": "source-linked-spatial-hypothesis-v1", "physical_validation": "unverified",
        "source_evidence_sha256": files["evidence.json"], "source_frame_count": evidence.get("frame_count"),
        "decoded_frames": evidence.get("decoded_frames"),
        "reconstruction_coverage": {k: coverage[k] for k in coverage_keys if k in coverage},
        "evidence_scope": evidence.get("scope"),
        "assumptions": scene.get("assumptions", []), "gravity": scene["gravity"],
        "timing": "Original saved evidence clock retained; no native-clock qualification added by import.",
        "qualification": "Saved source archive and authored planar-to-3D construction verified; tracking, physical accuracy and historical recordings not rerun."}
    _require(len(json.dumps(scope, sort_keys=True, indent=2, allow_nan=False).encode()) <= MAX_SOURCE_SCOPE_BYTES,
             "source summary exceeds 32 KiB")
    return {"scene": scene, "source_file": media, "source_sha256": source_hash,
        "learned_policy_available": learned, "source_scope": scope,
        "source_manifest_sha256": verified["manifest_sha256"],
        "runtime_compatibility": {"numeric_core_sha256": {n: core[n] for n in CORE_MODULES},
            "historical_facade_revalidated": False, "historical_recordings_revalidated": False,
            "scope": "Current numeric source matches saved code; no archived code executed or historical replay upgraded."},
        "sealed_bytes": total - len(manifest_raw),
        "import_reservation_bytes": total + len(config_raw) + signatures[media][2] + METADATA_RESERVATION_BYTES}
