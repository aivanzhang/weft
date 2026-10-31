"""Verify saved selective-crop derivations without rerunning perception or choosing models."""

from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import time

from weft import exercise_vision
from weft.errors import UnsupportedVideo
from weft.exercise_source import describe_source
from weft.io import canonical_hash, local_time
from weft.pose_crop import map_crop_pose
from weft.pose_crop_selection import CONTRACT, METHOD, apply_selective_crop, assess_selective_crop, plan_selective_crop
from weft.shared_world_packaging import read_package_file, read_package_manifest, write_package_files
from weft.world_evidence import _check_tree


PROFILE = "exercise-pose-recovery-v1"
MAX_BYTES = 128 * 1024 * 1024
MAX_FILES = 512
MAX_FILE_BYTES = 64 * 1024 * 1024
MAX_JOURNAL_BYTES = 16 * 1024 * 1024
MAX_LINE_BYTES = 1024 * 1024
MAX_SECONDS = 30
RESERVE_BYTES = 256 * 1024
SUFFIXES = {".mp4", ".webm", ".mov", ".m4v", ".avi", ".mkv"}
REJECTED = {"failed", "error", "rejected", "cancelled", "canceled", "unsupported"}
IMPLEMENTATION_FILES = ("pose_source_recovery.py", "pose_crop_selection.py", "pose_crop.py", "exercise_vision.py",
    "io.py", "errors.py", "exercise_source.py", "shared_world_packaging.py", "world_evidence.py", "observations.py")
INPUT_FILES = {
    "whole_image/evidence.json": "recovery/baseline-evidence.json",
    "whole_image/pose-landmarks.json": "pose-landmarks.json",
    "whole_image/identity-audit.json": "recovery/baseline-identity-audit.json",
    "selected/evidence.json": "evidence.json",
    "selected/pose-landmarks.json": "pose-landmarks.json",
    "selected/identity-audit.json": "identity-audit.json",
    "crop-plan.json": "recovery/plans.json",
    "crop-observations.jsonl": "recovery/candidates.jsonl",
    "decisions.json": "recovery/decisions.json",
}
PROOF_FILES = frozenset({"pose-recovery.json", "recovery/input-manifest.json", "recovery/implementation.json",
    "recovery/baseline-evidence.json", "recovery/baseline-identity-audit.json", "recovery/plans.json",
    "recovery/candidates.jsonl", "recovery/decisions.json",
    *("recovery/implementation-source/" + name for name in IMPLEMENTATION_FILES)})
POSE_FILES = frozenset({"evidence.json", "pose-landmarks.json", "identity-audit.json", "model.json",
                        "run-config.json", "source.json"})


def _bytes(value):
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _json(raw, *, object_only=True):
    if not isinstance(raw, bytes) or len(raw) > MAX_FILE_BYTES:
        raise ValueError("Recovery JSON exceeds its byte bound")

    def unique(pairs):
        value = {}
        for name, item in pairs:
            if name in value:
                raise ValueError("Recovery JSON contains duplicate keys")
            value[name] = item
        return value

    def invalid(_value):
        raise ValueError("Recovery JSON contains non-finite numbers")

    try:
        value = json.loads(raw, object_pairs_hook=unique, parse_constant=invalid)
    except (UnicodeError, json.JSONDecodeError, RecursionError) as error:
        raise ValueError("Recovery requires strict bounded JSON") from error
    if object_only and not isinstance(value, dict):
        raise ValueError("Recovery JSON must be an object")
    _check_tree(value)
    return value


def _path(value):
    path = Path(value).absolute()
    if ".." in path.parts or any(part.is_symlink() for part in (path, *path.parents)):
        raise ValueError("Recovery paths cannot pass through symlinks")
    return path


def _check(deadline):
    if time.monotonic() >= deadline:
        raise TimeoutError("Saved pose recovery exceeded its deadline")


def _read(path, deadline, expected=None, maximum=MAX_FILE_BYTES):
    _check(deadline)
    path = _path(path)
    if maximum < 1:
        raise ValueError("Saved recovery input has no remaining byte reservation")
    raw = read_package_file(path.parent, path.name, max_bytes=min(MAX_FILE_BYTES, maximum), expected_sha256=expected)
    _check(deadline)
    return raw


def _not_failed(root):
    path = root / "failure.json"
    if path.exists() or path.is_symlink():
        raise ValueError("Failed proposal packages cannot become verified pose evidence")


class _NoFiles:
    def write_json(self, name, _value):
        if name not in {"joint-angles.json", "evidence.json"}:
            raise ValueError("Unexpected pure evidence finalization artifact")


def _finalize(evidence):
    try:
        exercise_vision.finalize_evidence(evidence, Path("."), artifact_sink=_NoFiles())
    except UnsupportedVideo:
        if evidence.get("quality", {}).get("passes") is not False:
            raise
    return evidence


def _validate_baseline(baseline, raw, audit, source_sha):
    model = baseline.get("model", {})
    full = exercise_vision.MODEL_VARIANTS["full"]
    if (baseline.get("schema_version") != 1 or baseline.get("profile") != "exercise-pose-v1"
            or baseline.get("source_sha256") != source_sha or raw.get("model") != model
            or any(model.get(key) != value for key, value in full.items())
            or model.get("variant") != "full" or model.get("model_version") != "float16/1"
            or model.get("running_mode") != "IMAGE" or model.get("temporal_tracking") is not False
            or model.get("maximum_pose_hypotheses") != 2 or model.get("delegate") != "CPU"
            or model.get("network_inference") is not False or not isinstance(model.get("mediapipe_version"), str)):
        raise ValueError("Recovery requires exact Full float16/1 IMAGE-mode source observations")
    identity = baseline.get("pose_identity", {})
    if (identity.get("mode") != "raw" or audit.get("mode") != "raw"
            or identity.get("repaired_frames") != 0 or audit.get("repaired_evidence_ids") != []
            or baseline.get("pose_reinference") or audit.get("source_sha256") != source_sha):
        raise ValueError("Recovery requires original raw identity; chained refinement is unsupported")
    fps, count, decoded, stride = (baseline.get(key) for key in ("fps", "frame_count", "decoded_frames", "inference_stride"))
    if (type(fps) not in (int, float) or not math.isfinite(fps) or not 5 <= fps <= 240
            or type(count) is not int or type(decoded) is not int or not 10 <= count <= 1800
            or not 10 <= decoded <= 1800 or max(count, decoded) / fps > 60
            or stride != max(1, math.ceil(fps / 15), math.ceil(count / 900)) or type(stride) is not int):
        raise ValueError("Recovery source exceeds the unchanged sampling/frame/duration bounds")
    if abs(count - decoded) > 1:
        container = baseline.get("container_frame_audit", {})
        if (container.get("encoded_video_frames") != decoded or container.get("decoded_frames") != decoded
                or container.get("matches") is not True):
            raise ValueError("Source frame discrepancy lacks the existing encoded-frame audit")
    for key, maximum in (("width", 4096), ("height", 2160)):
        if type(baseline.get(key)) is not int or not 64 <= baseline[key] <= maximum:
            raise ValueError("Recovery source dimensions are invalid")
    frames = baseline.get("frames")
    if (not isinstance(frames, list) or not 1 <= len(frames) <= 900 or any(not isinstance(row, dict) for row in frames)
            or raw.get("frames") != frames
            or [row.get("frame") for row in frames] != list(range(0, decoded, stride))):
        raise ValueError("Original raw poses must retain every sampled source slot")
    for row in frames:
        frame, native = row["frame"], row.get("decoder_timestamp_s")
        if (row.get("id") != f"{source_sha}:frame:{frame}" or type(row.get("time_s")) not in (int, float)
                or abs(row["time_s"] - frame / fps) > 1e-7 or "decoder_timestamp_s" not in row
                or (native is not None and (type(native) not in (int, float) or not math.isfinite(native)
                                            or not 0 <= native <= 60))
                or any(key in row for key in ("identity_raw", "image_refinement", "image_refinement_raw"))):
            raise ValueError("Original source identities, clocks or unmodified context are invalid")
    if _finalize(deepcopy(baseline)) != baseline:
        raise ValueError("Baseline quality/reference fields differ from the preserved raw observations")


def _recompute(files, source_sha, deadline):
    baseline = _json(files["recovery/baseline-evidence.json"])
    raw = _json(files["pose-landmarks.json"])
    baseline_audit = _json(files["recovery/baseline-identity-audit.json"])
    _validate_baseline(baseline, raw, baseline_audit, source_sha)
    raw_sha = _sha(files["pose-landmarks.json"])
    for audit in (baseline_audit, _json(files["identity-audit.json"])):
        if (audit != baseline_audit or audit.get("raw_artifact") != "pose-landmarks.json"
                or audit.get("raw_artifact_sha256") != raw_sha):
            raise ValueError("Recovery identity audit must retain the exact original detector bytes")
    if baseline["pose_identity"].get("raw_artifact_sha256") != raw_sha:
        raise ValueError("Baseline identity metadata differs from the original pose bytes")
    plans = {}
    for row in baseline["frames"]:
        _check(deadline)
        plans[row["frame"]] = plan_selective_crop(baseline, row)
    eligible = [frame for frame, plan in plans.items() if plan["eligible"]]
    recorded_plan = _json(files["recovery/plans.json"])
    if (recorded_plan.get("plans") != list(plans.values()) or recorded_plan.get("eligible_frames") != eligible
            or recorded_plan.get("bound_exceeded") is not (len(eligible) > CONTRACT["maximum_target_frames"])):
        raise ValueError("Saved crop plans differ from the unchanged original-context selector")
    if len(eligible) > CONTRACT["maximum_target_frames"]:
        raise ValueError("Recovery candidate count exceeds the fixed 64-frame bound; no partial selection")
    journal = files["recovery/candidates.jsonl"]
    if len(journal) > MAX_JOURNAL_BYTES:
        raise ValueError("Recovery candidate journal exceeds its byte bound")
    lines = journal.splitlines()
    if len(lines) > CONTRACT["maximum_target_frames"] or any(not line or len(line) > MAX_LINE_BYTES for line in lines):
        raise ValueError("Recovery candidate journal exceeds its row/count bound")
    proposals, decisions, original = {}, {}, {row["frame"]: row for row in baseline["frames"]}
    for line in lines:
        _check(deadline)
        candidate = _json(line)
        if set(candidate) != {"frame", "raw_crop_result", "proposed", "decision"}:
            raise ValueError("Crop journal fields differ from the saved-evidence contract")
        frame = candidate["frame"]
        if type(frame) is not int or frame not in eligible or frame in proposals:
            raise ValueError("Saved crop candidate is duplicate, unplanned or outside the source")
        saved = candidate["proposed"]
        if (not isinstance(saved, dict) or type(saved.get("inference_seconds")) not in (int, float)
                or not math.isfinite(saved["inference_seconds"]) or saved["inference_seconds"] < 0):
            raise ValueError("Saved crop proposal must be an object")
        mapped = map_crop_pose(candidate["raw_crop_result"], plans[frame]["crop"], baseline,
                               original[frame], saved.get("inference_seconds"))
        mapped["crop_proposal"] = plans[frame]["crop"]
        if mapped != saved:
            raise ValueError("Saved crop coordinates or derived geometry differ from the raw model result")
        decision = assess_selective_crop(baseline, original[frame], mapped, plans[frame])
        if decision != candidate["decision"]:
            raise ValueError("Saved crop acceptance differs from the unchanged source-only decision")
        proposals[frame], decisions[frame] = mapped, decision
    if list(proposals) != eligible or _json(files["recovery/decisions.json"], object_only=False) != list(decisions.values()):
        raise ValueError("Every eligible crop, including rejected candidates, must be retained exactly once")
    selected = _finalize(apply_selective_crop(baseline, proposals, plans, decisions))
    if selected != _json(files["evidence.json"]):
        raise ValueError("Selected pose evidence differs from the complete verified derivation")
    _check(deadline)
    return selected, {"source_slots": baseline["decoded_frames"], "sampled_slots": len(baseline["frames"]),
        "eligible_frames": eligible, "accepted_frames": [frame for frame, row in decisions.items() if row["accepted"]],
        "rejected_frames": [frame for frame, row in decisions.items() if not row["accepted"]],
        "pose_quality_passes": selected["quality"]["passes"], "pose_quality_errors": selected["quality"]["errors"],
        "unchanged_original_rows": len(baseline["frames"]) - sum(row["accepted"] for row in decisions.values())}


def validate_pose_recovery_files(files, *, source_sha256, deadline=None):
    """Verify the explicit refinement proof before allowing derived body observations."""
    deadline = time.monotonic() + MAX_SECONDS if deadline is None else deadline
    if type(deadline) not in (int, float) or not math.isfinite(deadline):
        raise ValueError("Recovery proof deadline must be finite")
    _check(deadline)
    if (not isinstance(files, dict) or not PROOF_FILES | POSE_FILES <= files.keys()
            or any(not isinstance(raw, bytes) or len(raw) > MAX_FILE_BYTES for raw in files.values())
            or len(files) > MAX_FILES or sum(map(len, files.values())) > MAX_BYTES):
        raise ValueError("Pose recovery requires its complete bounded byte-level proof")
    marker = _json(files["pose-recovery.json"])
    if (marker.get("schema_version") != 1 or marker.get("profile") != PROFILE
            or marker.get("status") != "source_evidence_only" or marker.get("simulation_ready") is not False
            or marker.get("method") != METHOD or marker.get("contract") != dict(CONTRACT)
            or marker.get("source_sha256") != source_sha256):
        raise ValueError("Unsupported source-only pose recovery profile or recipe")
    source_name = marker.get("source_artifact")
    if source_name not in {"source" + suffix for suffix in SUFFIXES}:
        raise ValueError("Recovery must identify the exact original source container")
    expected = (PROOF_FILES | POSE_FILES | {source_name}) - {"pose-recovery.json"}
    artifacts = marker.get("artifacts")
    if (not isinstance(artifacts, dict) or set(artifacts) != expected
            or any(name not in files or _sha(files[name]) != digest for name, digest in artifacts.items())
            or artifacts[source_name] != source_sha256):
        raise ValueError("Pose recovery artifacts differ from their exact byte bindings")
    input_manifest = _json(files["recovery/input-manifest.json"])
    declared = input_manifest.get("files", {})
    if (input_manifest.get("status") in REJECTED or not isinstance(declared, dict)
            or "failure.json" in declared or not 1 <= len(declared) <= MAX_FILES - 2):
        raise ValueError("Recovery proof cannot relabel a failed or oversized source manifest")
    if (marker.get("input_manifest_sha256") != _sha(files["recovery/input-manifest.json"])
            or any(declared.get(original) != _sha(files[target]) for original, target in INPUT_FILES.items())):
        raise ValueError("Recovery input bytes are not covered by their preserved source seal")
    implementation = _json(files["recovery/implementation.json"])
    if set(implementation.get("files", {})) != set(IMPLEMENTATION_FILES):
        raise ValueError("Recovery implementation proof is incomplete")
    for name, expected_sha in implementation["files"].items():
        _check(deadline)
        raw = files["recovery/implementation-source/" + name]
        if _sha(raw) != expected_sha or _read(Path(__file__).with_name(name), deadline, expected_sha) != raw:
            raise ValueError("Recovery verifier/selector implementation differs from the pinned recipe")
    selected, result = _recompute(files, source_sha256, deadline)
    if marker.get("result") != result:
        raise ValueError("Recovery summary differs from verified source decisions")
    config, model, source = (_json(files[name]) for name in ("run-config.json", "model.json", "source.json"))
    if (config != _config() or model != selected["model"] or source.get("sha256") != source_sha256
            or source.get("artifact") != source_name):
        raise ValueError("Recovery configuration/model/source metadata conflicts with its derivation")
    return selected


def _config():
    return {"schema_version": 1, "profile": "exercise", "pose_mode": "image", "pose_identity": "raw",
            "pose_refinement": METHOD, "preparation_profile": PROFILE,
            "scope": "Verified saved Full IMAGE/raw crop derivation; no new inference, body, task or training"}


def _failure(out, error, started):
    raw = _bytes({"schema_version": 1, "profile": PROFILE, "status": "failed", "simulation_ready": False,
        "created_at": local_time(), "elapsed_seconds": time.monotonic() - started,
        "error": {"type": type(error).__name__, "message": str(error)[:2000]}})
    try:
        write_package_files(out, {"failure.json": raw})
    except (OSError, ValueError):
        pass


def prepare_pose_source_recovery(proposals_run, out, *, source_video, max_seconds=30):
    """Seal one verified saved crop case; never decode, infer, fit a body or run physics."""
    if type(max_seconds) not in (int, float) or not math.isfinite(max_seconds) or not 1 <= max_seconds <= MAX_SECONDS:
        raise ValueError("Saved pose recovery requires a cooperative 1–30 second bound")
    source, video, target = map(_path, (proposals_run, source_video, out))
    if (target.exists() or video.suffix.lower() not in SUFFIXES
            or any(target.is_relative_to(path) or path.is_relative_to(target) for path in (source, video))):
        raise ValueError("Recovery output must be new and outside the source package and video")
    started = time.monotonic(); deadline = started + max_seconds
    target.mkdir(parents=True, exist_ok=False)
    try:
        snapshots = {name: _read(Path(__file__).with_name(name), deadline) for name in IMPLEMENTATION_FILES}
        _not_failed(source)
        manifest_raw, declared = read_package_manifest(source)
        manifest = _json(manifest_raw)
        if (manifest.get("status") in REJECTED or "failure.json" in declared
                or not set(INPUT_FILES) <= declared.keys() or len(declared) + 2 > MAX_FILES):
            raise ValueError("Saved recovery requires a complete, bounded and unrejected proposal case")
        kept, pins, total = {}, [(source / "manifest.json", _sha(manifest_raw))], len(manifest_raw)
        for name, expected in sorted(declared.items()):
            raw = _read(source / name, deadline, expected, MAX_BYTES - RESERVE_BYTES - total)
            total += len(raw)
            if total > MAX_BYTES - RESERVE_BYTES:
                raise ValueError("Saved recovery input exceeds its 128-MiB bound")
            pins.append((source / name, expected))
            if name in {"status.json", "report.json"} and _json(raw).get("status") in REJECTED:
                raise ValueError("Failed proposal outcomes cannot be promoted into recovery evidence")
            if name in INPUT_FILES:
                mapped = INPUT_FILES[name]
                if mapped in kept and kept[mapped] != raw:
                    raise ValueError("Selected raw poses differ from the original detector bytes")
                kept[mapped] = raw
        video_raw = _read(video, deadline, maximum=MAX_BYTES - RESERVE_BYTES - total)
        total += len(video_raw)
        if not video_raw or total > MAX_BYTES - RESERVE_BYTES:
            raise ValueError("Source plus saved crop case exceeds the bounded input envelope")
        source_sha, source_name = _sha(video_raw), "source" + video.suffix.lower()
        selected, result = _recompute(kept, source_sha, deadline)
        implementation = {"files": {name: _sha(raw) for name, raw in snapshots.items()},
                          "scope": "Selected recovery generation/verification modules, not a full Python dependency closure"}
        source_metadata = {**(describe_source(source_sha) or {"original_name": video.name,
            "rights": "Rights not independently verified; user-provided local video"}),
            "sha256": source_sha, "artifact": source_name,
            "derived_overlays": "Original video and inferred pose records preserved; no new image rendering"}
        files = {**kept, source_name: video_raw, "source.json": _bytes(source_metadata),
            "model.json": _bytes(selected["model"]), "run-config.json": _bytes(_config()),
            "recovery/input-manifest.json": manifest_raw, "recovery/implementation.json": _bytes(implementation),
            **{"recovery/implementation-source/" + name: raw for name, raw in snapshots.items()}}
        marker = {"schema_version": 1, "profile": PROFILE, "status": "source_evidence_only", "simulation_ready": False,
            "method": METHOD, "contract": dict(CONTRACT), "created_at": local_time(),
            "source_sha256": source_sha, "source_artifact": source_name, "input_manifest_sha256": _sha(manifest_raw),
            "artifacts": {name: _sha(raw) for name, raw in files.items()}, "result": result,
            "work": {"video_decodes": 0, "inference_calls": 0, "physics_steps": 0, "body_models": 0, "api_calls": 0},
            "bounds": {"maximum_seconds": max_seconds, "maximum_bytes": MAX_BYTES, "maximum_files": MAX_FILES,
                       "maximum_file_bytes": MAX_FILE_BYTES, "maximum_crop_candidates": CONTRACT["maximum_target_frames"]},
            "limits": ["Saved model results are re-derived, not re-inferred or independently verified physical observations",
                "No model/mode selection, chained refinement, new labels, task or geometry feedback enters crop selection",
                "Body construction retains its separate pose, movement and geometry gates; source evidence is not a ready body or lab",
                "All input package members are hash-checked; only required derivation bytes and original video are copied",
                "Model weights and unused previews are not duplicated; original supplied native sample times remain uncalibrated"]}
        files["pose-recovery.json"] = _bytes(marker)
        if (any(len(raw) > MAX_FILE_BYTES for raw in files.values()) or len(files) + 2 > MAX_FILES
                or sum(map(len, files.values())) > MAX_BYTES - RESERVE_BYTES):
            raise ValueError("Recovery output exceeds downstream body-source bounds")
        write_package_files(target, files)
        for name, raw in files.items():
            _read(target / name, deadline, _sha(raw))
        for path, expected in pins:
            _read(path, deadline, expected)
        _read(video, deadline, source_sha)
        _not_failed(source)
        for name, raw in snapshots.items():
            _read(Path(__file__).with_name(name), deadline, _sha(raw))
        _check(deadline)
        seal = _bytes({"schema_version": 1, "profile": PROFILE, "status": "source_evidence_only",
                       "simulation_ready": False, "files": {name: _sha(raw) for name, raw in files.items()}})
        if sum(map(len, files.values())) + len(seal) > MAX_BYTES:
            raise ValueError("Recovery seal exceeds the retained-output bound")
        write_package_files(target, {"manifest.json": seal})
        return target
    except Exception as error:
        _failure(target, error, started)
        raise
