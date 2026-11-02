"""Seal source-only mode recovery while preserving both original detector runs."""

from copy import deepcopy
import math
from pathlib import Path
import time

from weft import exercise_body_source as custody, exercise_vision
from weft.errors import UnsupportedVideo
from weft.exercise_source import describe_source
from weft.io import canonical_hash, local_time
from weft.pose_mode_recovery import CONTRACT, METHOD, _validate_inputs, propose_pose_mode_recovery
from weft.shared_world_packaging import read_package_file, read_package_manifest, write_package_files


PROFILE = "exercise-pose-mode-recovery-v1"
MAX_SECONDS = 30
ORIGINAL_FILES = ("evidence.json", "pose-landmarks.json", "identity-audit.json")
IMPLEMENTATION_FILES = ("pose_mode_source.py", "pose_mode_recovery.py", "exercise_body_source.py",
    "exercise_vision.py", "exercise_source.py", "io.py", "errors.py", "shared_world_packaging.py", "world_evidence.py")
PROOF_FILES = frozenset({"pose-mode-recovery.json", "mode-recovery/decisions.json", "mode-recovery/implementation.json",
    *(f"mode-recovery/{label}-{name}" for label in ("primary", "alternative") for name in (*ORIGINAL_FILES, "manifest.json")),
    *("mode-recovery/implementation-source/" + name for name in IMPLEMENTATION_FILES)})


class _NoFiles:
    def write_json(self, name, _value):
        if name not in {"joint-angles.json", "evidence.json"}:
            raise ValueError("Unexpected mode recovery finalization output")


def _finalize(evidence):
    try:
        exercise_vision.finalize_evidence(evidence, Path("."), artifact_sink=_NoFiles())
    except UnsupportedVideo:
        if evidence.get("quality", {}).get("passes") is not False:
            raise
    return evidence


def _original(files, label, source_sha, evidence):
    raw, audit = (custody._json(files[f"mode-recovery/{label}-{name}"]) for name in ORIGINAL_FILES[1:])
    mode, capacity = ("video", 1) if label == "primary" else ("image", 2)
    custody._model(evidence.get("model", {}), custody._options(mode, "raw", capacity))
    raw_sha = custody._sha(files[f"mode-recovery/{label}-pose-landmarks.json"])
    identity = evidence.get("pose_identity", {})
    if (evidence.get("source_sha256") != source_sha or raw.get("model") != evidence["model"]
            or raw.get("frames") != evidence.get("frames") or identity.get("mode") != "raw"
            or identity.get("raw_artifact_sha256") != raw_sha or identity.get("repaired_frames") != 0
            or identity.get("audit_artifact") != "identity-audit.json" or audit.get("mode") != "raw"
            or audit.get("source_sha256") != source_sha or audit.get("raw_artifact") != "pose-landmarks.json"
            or audit.get("raw_artifact_sha256") != raw_sha or audit.get("repaired_evidence_ids") != []):
        raise ValueError("Mode recovery requires exact original raw pose and identity bytes")
    if _finalize(deepcopy(evidence)) != evidence:
        raise ValueError("Original pose quality/reference fields differ from retained observations")
    return evidence


def _original_pair(files, source_sha):
    labels = ("primary", "alternative")
    documents = [custody._json(files[f"mode-recovery/{label}-evidence.json"]) for label in labels]
    _validate_inputs(*documents)
    return [_original(files, label, source_sha, document) for label, document in zip(labels, documents, strict=True)]


def _derive(primary, alternative, decisions):
    result = deepcopy(primary)
    for index, decision in enumerate(decisions["rows"]):
        if not decision["accepted"]:
            continue
        row = deepcopy(alternative["frames"][index])
        row["mode_refinement"] = {"method": METHOD,
            "observation_artifact": "mode-recovery/alternative-evidence.json",
            "observation_sha256": decision["candidate_sha256"], "decision_sha256": canonical_hash(decision),
            "original_sha256": decision["original_sha256"], "model": deepcopy(alternative["model"]),
            "scope": CONTRACT["scope"]}
        row["mode_refinement_raw"] = deepcopy(primary["frames"][index])
        result["frames"][index] = row
    result["pose_mode_recovery"] = {"method": METHOD, "primary_model": deepcopy(primary["model"]),
        "alternative_model": deepcopy(alternative["model"]), "accepted_frames": decisions["accepted_frames"],
        "decisions_artifact": "mode-recovery/decisions.json", "decisions_sha256": canonical_hash(decisions),
        "scope": CONTRACT["scope"]}
    result["assumptions"].append("Explicit source-only mode recovery may select retained IMAGE estimates for isolated VIDEO inconsistencies. Each selected frame identifies its actual model mode; original observations remain preserved. This is not verified anatomy or contact.")
    return _finalize(result)


def _config():
    return {"schema_version": 1, "profile": "exercise", "pose_mode": "video", "pose_identity": "raw",
        "max_pose_hypotheses": 1, "pose_refinement": METHOD, "preparation_profile": PROFILE,
        "scope": "Primary single-person VIDEO with explicitly identified retained IMAGE alternatives; no new inference or simulation"}


def _result(selected, decisions):
    return {"source_slots": selected["decoded_frames"], "sampled_slots": len(selected["frames"]),
        "segment_slots": sum(len(row["segments"]) for row in decisions["rows"]),
        "eligible_frames": [row["frame"] for row in decisions["rows"] if row["eligible"]],
        "accepted_frames": decisions["accepted_frames"], "bound_exceeded": decisions["bound_exceeded"],
        "unchanged_original_rows": len(selected["frames"]) - len(decisions["accepted_frames"]),
        "pose_quality_passes": selected["quality"]["passes"], "pose_quality_errors": selected["quality"]["errors"]}


def validate_pose_mode_files(files, *, source_sha256, deadline=None):
    """Recompute every decision; a resealed altered outcome cannot authorize replacement."""
    deadline = time.monotonic() + MAX_SECONDS if deadline is None else deadline
    if type(deadline) not in (int, float) or not math.isfinite(deadline):
        raise ValueError("Mode recovery requires a finite deadline")
    custody._deadline(deadline)
    if (not isinstance(files, dict) or not PROOF_FILES | custody.POSE_FILES <= files.keys()
            or len(files) > custody.MAX_FILES or any(not isinstance(raw, bytes) or len(raw) > custody.MAX_FILE_BYTES for raw in files.values())
            or sum(map(len, files.values())) > custody.MAX_BYTES):
        raise ValueError("Mode recovery requires its complete bounded byte-level proof")
    marker = custody._json(files["pose-mode-recovery.json"])
    if (marker.get("schema_version") != 1 or marker.get("profile") != PROFILE
            or marker.get("status") != "source_evidence_only" or marker.get("simulation_ready") is not False
            or marker.get("method") != METHOD or marker.get("contract") != dict(CONTRACT)
            or marker.get("source_sha256") != source_sha256):
        raise ValueError("Unsupported source-only mode recovery contract")
    source_name = marker.get("source_artifact")
    if source_name not in {"source" + suffix for suffix in custody.SUFFIXES}:
        raise ValueError("Mode recovery must retain the original source container")
    expected = (PROOF_FILES | custody.POSE_FILES | {source_name}) - {"pose-mode-recovery.json"}
    declared = marker.get("artifacts")
    if (not isinstance(declared, dict) or set(declared) != expected
            or any(name not in files or custody._sha(files[name]) != checksum for name, checksum in declared.items())
            or declared[source_name] != source_sha256):
        raise ValueError("Mode recovery artifact identities differ from retained bytes")
    for label in ("primary", "alternative"):
        manifest = custody._json(files[f"mode-recovery/{label}-manifest.json"])
        sealed = manifest.get("files", {})
        if (manifest.get("status") in custody.REJECTED or not isinstance(sealed, dict)
                or "failure.json" in sealed or not 1 <= len(sealed) < custody.MAX_FILES
                or any(sealed.get(name) != custody._sha(files[f"mode-recovery/{label}-{name}"]) for name in ORIGINAL_FILES)):
            raise ValueError("Original pose documents lack their unrejected input seals")
    implementation = custody._json(files["mode-recovery/implementation.json"])
    if set(implementation.get("files", {})) != set(IMPLEMENTATION_FILES):
        raise ValueError("Mode recovery implementation proof is incomplete")
    for name, checksum in implementation["files"].items():
        raw = files["mode-recovery/implementation-source/" + name]
        if custody._sha(raw) != checksum or custody._read(Path(__file__).with_name(name), deadline, checksum) != raw:
            raise ValueError("Mode recovery implementation differs from its frozen recipe")
    primary, alternative = _original_pair(files, source_sha256)
    decisions = propose_pose_mode_recovery(primary, alternative)
    if decisions != custody._json(files["mode-recovery/decisions.json"]):
        raise ValueError("Mode recovery decisions differ from complete original-source recomputation")
    selected = _derive(primary, alternative, decisions)
    if selected != custody._json(files["evidence.json"]) or marker.get("result") != _result(selected, decisions):
        raise ValueError("Selected mode evidence differs from its verified derivation")
    metadata = custody._json(files["source.json"])
    if (files["pose-landmarks.json"] != files["mode-recovery/primary-pose-landmarks.json"]
            or files["identity-audit.json"] != files["mode-recovery/primary-identity-audit.json"]
            or custody._json(files["model.json"]) != primary["model"]
            or custody._json(files["run-config.json"]) != _config()
            or metadata.get("sha256") != source_sha256 or metadata.get("artifact") != source_name):
        raise ValueError("Mode recovery metadata or original detector bytes were replaced")
    custody._deadline(deadline)
    return selected


def prepare_pose_mode_recovery(primary_run, alternative_run, out, *, source_video, max_seconds=30):
    """Package a saved proposal, never decode media, invoke a detector or fit a body."""
    if type(max_seconds) not in (int, float) or not math.isfinite(max_seconds) or not 1 <= max_seconds <= MAX_SECONDS:
        raise ValueError("Mode recovery requires a cooperative 1–30 second bound")
    primary, alternative, video = map(custody._path, (primary_run, alternative_run, source_video))
    if primary == alternative or video.suffix.lower() not in custody.SUFFIXES:
        raise ValueError("Mode recovery requires two original runs and a supported source container")
    target, started, deadline = custody._start(out, (primary, alternative, video), max_seconds)
    try:
        kept, pins, total = {}, [], 0
        for label, root in (("primary", primary), ("alternative", alternative)):
            if (root / "failure.json").exists() or (root / "failure.json").is_symlink():
                raise ValueError("A failed pose package cannot authorize mode recovery")
            manifest_raw, declared = read_package_manifest(root)
            manifest = custody._json(manifest_raw)
            if (manifest.get("status") in custody.REJECTED or "failure.json" in declared
                    or not set(ORIGINAL_FILES) <= declared.keys() or len(pins) + len(declared) + 1 > custody.MAX_FILES):
                raise ValueError("Mode recovery inputs must be complete bounded unrejected pose packages")
            pins.append((root / "manifest.json", custody._sha(manifest_raw)))
            kept[f"mode-recovery/{label}-manifest.json"] = manifest_raw
            total += len(manifest_raw)
            for name, checksum in sorted(declared.items()):
                custody._deadline(deadline)
                remaining = custody.MAX_BYTES - custody.RESERVE_BYTES - total
                if remaining <= 0:
                    raise ValueError("Mode recovery input byte bound exceeded")
                raw = read_package_file(root, name, expected_sha256=checksum, max_bytes=min(custody.MAX_FILE_BYTES, remaining))
                total += len(raw)
                pins.append((root / name, checksum))
                if name in {"status.json", "report.json", "body-source.json"} and custody._json(raw).get("status") in custody.REJECTED:
                    raise ValueError("Rejected pose outcomes cannot become mode recovery inputs")
                if name in ORIGINAL_FILES:
                    kept[f"mode-recovery/{label}-{name}"] = raw
        video_raw = custody._read(video, deadline)
        if not video_raw or total + len(video_raw) > custody.MAX_BYTES - custody.RESERVE_BYTES:
            raise ValueError("Source plus original pose packages exceeds the input bound")
        source_sha, source_name = custody._sha(video_raw), "source" + video.suffix.lower()
        baseline, alternate = _original_pair(kept, source_sha)
        decisions = propose_pose_mode_recovery(baseline, alternate)
        selected = _derive(baseline, alternate, decisions)
        snapshots = {name: custody._read(Path(__file__).with_name(name), deadline) for name in IMPLEMENTATION_FILES}
        implementation = {"files": {name: custody._sha(raw) for name, raw in snapshots.items()},
            "scope": "Mode selection, finalization and custody modules; not a complete runtime closure"}
        files = {**kept, source_name: video_raw, "evidence.json": custody._bytes(selected),
            "pose-landmarks.json": kept["mode-recovery/primary-pose-landmarks.json"],
            "identity-audit.json": kept["mode-recovery/primary-identity-audit.json"],
            "model.json": custody._bytes(baseline["model"]), "run-config.json": custody._bytes(_config()),
            "source.json": custody._bytes({**(describe_source(source_sha) or {"original_name": video.name,
                "rights": "Rights not independently verified; user-provided local video"}),
                "sha256": source_sha, "artifact": source_name}),
            "mode-recovery/decisions.json": custody._bytes(decisions),
            "mode-recovery/implementation.json": custody._bytes(implementation),
            **{"mode-recovery/implementation-source/" + name: raw for name, raw in snapshots.items()}}
        marker = {"schema_version": 1, "profile": PROFILE, "status": "source_evidence_only", "simulation_ready": False,
            "method": METHOD, "contract": dict(CONTRACT), "created_at": local_time(),
            "source_sha256": source_sha, "source_artifact": source_name,
            "artifacts": {name: custody._sha(raw) for name, raw in files.items()}, "result": _result(selected, decisions),
            "work": {"video_decodes": 0, "inference_calls": 0, "physics_steps": 0, "body_models": 0, "api_calls": 0},
            "bounds": {"maximum_seconds": max_seconds, "maximum_bytes": custody.MAX_BYTES,
                "maximum_files": custody.MAX_FILES, "maximum_file_bytes": custody.MAX_FILE_BYTES},
            "limits": ["Explicit saved-mode proposal, not default model selection or validated anatomy/contact",
                "Every original sample and rejected decision survives; selected frames identify actual IMAGE observations",
                "Original video bytes and sample clocks are preserved, not calibrated",
                "Body construction and physical evaluation remain separate unchanged gates",
                "Cooperative deadline only; caller subprocess must enforce a hard wall limit"]}
        files["pose-mode-recovery.json"] = custody._bytes(marker)
        custody._write(target, files, deadline)
        validate_pose_mode_files(files, source_sha256=source_sha, deadline=deadline)
        for path, checksum in [*pins, (video, source_sha)]:
            custody._read(path, deadline, checksum)
        for root in (primary, alternative):
            if (root / "failure.json").exists() or (root / "failure.json").is_symlink():
                raise ValueError("Input gained a failure marker during mode recovery")
        hashes, size = custody._inventory(target, deadline)
        if hashes != {name: custody._sha(raw) for name, raw in files.items()}:
            raise ValueError("Written mode recovery differs from verified bytes")
        seal = custody._bytes({"schema_version": 1, "profile": PROFILE, "status": "source_evidence_only",
            "simulation_ready": False, "files": hashes})
        if size + len(seal) > custody.MAX_BYTES:
            raise ValueError("Mode recovery seal exceeds the output bound")
        custody._deadline(deadline)
        write_package_files(target, {"manifest.json": seal})
        return target
    except Exception as error:
        try:
            write_package_files(target, {"failure.json": custody._bytes({"schema_version": 1, "profile": PROFILE,
                "status": "failed", "simulation_ready": False, "created_at": local_time(),
                "elapsed_seconds": time.monotonic() - started,
                "error": {"type": type(error).__name__, "message": str(error)[:2000]}})})
        except (OSError, ValueError):
            pass
        raise
