"""Screen source windows without merging identities across competing pose samples."""

from hashlib import sha256
import json
import math

PROFILE = "pose-identity-barrier-windows-v1"
MAX_INPUT_BYTES = 16 * 1024 * 1024
MAX_SAMPLES = 1800
_USABLE = {"tracked", "tracked_partial"}
_STATUSES = _USABLE | {"multiple_people", "no_pose", "low_confidence", "invalid_geometry", "invalid_landmark_count"}
_SCREEN = {"minimum_usable_samples": 8, "minimum_usable_fraction": .65, "minimum_usable_span_s": .4}


def _pairs(items):
    result = {}
    for key, value in items:
        if key in result:
            raise ValueError("Duplicate source JSON field")
        result[key] = value
    return result


def _document(raw):
    if not isinstance(raw, bytes) or not 1 <= len(raw) <= MAX_INPUT_BYTES:
        raise ValueError("Identity windows require bounded original JSON bytes")
    try:
        result = json.loads(raw, object_pairs_hook=_pairs)
        json.dumps(result, allow_nan=False)
    except (ValueError, TypeError, RecursionError) as error:
        raise ValueError("Invalid source JSON") from error
    if not isinstance(result, dict):
        raise ValueError("Source document must be an object")
    return result


def _finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def screen_identity_windows(evidence_bytes, raw_pose_bytes):
    """Retain the full source and screen count/coverage/span only; never emit a body."""
    if (not isinstance(evidence_bytes, bytes) or not isinstance(raw_pose_bytes, bytes)
            or len(evidence_bytes) + len(raw_pose_bytes) > MAX_INPUT_BYTES):
        raise ValueError("Identity-window inputs exceed the aggregate byte bound")
    evidence, raw = _document(evidence_bytes), _document(raw_pose_bytes)
    source = evidence.get("source_sha256")
    identity = evidence.get("pose_identity", {})
    if (not isinstance(source, str) or len(source) != 64 or any(c not in "0123456789abcdef" for c in source)
            or evidence.get("profile") != "exercise-pose-v1" or evidence.get("schema_version") != 1
            or "source_clock" in evidence or not isinstance(identity, dict)
            or identity.get("mode") != "raw" or identity.get("repaired_frames") != 0
            or identity.get("raw_artifact_sha256") != sha256(raw_pose_bytes).hexdigest()):
        raise ValueError("Identity windows require original raw-mode pose evidence and its exact raw artifact")
    frames = evidence.get("frames")
    if (not isinstance(frames, list) or not 1 <= len(frames) <= MAX_SAMPLES
            or raw.get("frames") != frames or raw.get("model") != evidence.get("model")):
        raise ValueError("Every original pose row and model must match; no selected or repaired views")
    fps, stride, decoded = (evidence.get(k) for k in ("fps", "inference_stride", "decoded_frames"))
    if (not _finite(fps) or not 1 <= fps <= 120 or type(stride) is not int or not 1 <= stride <= 120
            or type(decoded) is not int or not 1 <= decoded <= MAX_SAMPLES * stride
            or len(frames) != (decoded + stride - 1) // stride):
        raise ValueError("Identity windows require the complete declared sampled-frame inventory")
    barriers, slots, windows, pending = [], [], [], []

    def finish_window():
        if not pending:
            return
        usable = [i for i in pending if frames[i]["status"] in _USABLE]
        span = frames[usable[-1]]["time_s"] - frames[usable[0]]["time_s"] if usable else 0.
        fraction = len(usable) / len(pending)
        failures = []
        if len(usable) < _SCREEN["minimum_usable_samples"]:
            failures.append("insufficient_usable_samples")
        if fraction < _SCREEN["minimum_usable_fraction"]:
            failures.append("insufficient_usable_fraction")
        if span < _SCREEN["minimum_usable_span_s"]:
            failures.append("insufficient_usable_span")
        window_id = len(windows)
        for i in pending:
            slots[i]["window_id"] = window_id
        windows.append({"id": window_id, "sample_indices": list(pending), "usable_sample_indices": usable,
                        "frame_interval": [frames[pending[0]]["frame"], frames[pending[-1]]["frame"]],
                        "source_time_interval_s": [frames[pending[0]]["time_s"], frames[pending[-1]]["time_s"]],
                        "usable_samples": len(usable), "sampled_frames": len(pending), "usable_fraction": fraction,
                        "usable_span_s": span, "screen_passed": not failures, "screen_failures": failures,
                        "body_qualified": False, "identity_verified": False,
                        "interpolation_performed": False})
        pending.clear()

    for i, frame in enumerate(frames):
        if (not isinstance(frame, dict) or type(frame.get("frame")) is not int or frame["frame"] != i * stride
                or frame.get("id") != f"{source}:frame:{frame['frame']}"
                or not _finite(frame.get("time_s")) or abs(frame["time_s"] - frame["frame"] / fps) > 1e-9
                or not isinstance(frame.get("status"), str) or frame["status"] not in _STATUSES
                or type(frame.get("pose_count")) is not int or not 0 <= frame["pose_count"] <= 2
                or frame.get("decoder_timestamp_s") is not None and not _finite(frame["decoder_timestamp_s"])
                or not isinstance(frame.get("poses", []), list)
                or "identity_raw" in frame or frame.get("identity_recompute_required")):
            raise ValueError("Malformed, retimed or incomplete original pose slot")
        count, status = frame["pose_count"], frame["status"]
        competing = count > 1 or status == "multiple_people" or len(frame.get("poses", [])) > 1
        if competing:
            if (count != 2 or status != "multiple_people" or not isinstance(frame.get("poses"), list)
                    or len(frame["poses"]) != count
                    or frame.get("identity_status") != "unresolved_multiple_pose_hypotheses"
                    or frame.get("landmarks") != [] or frame.get("world_landmarks") != []):
                raise ValueError("Competing hypotheses cannot be selected, discarded or relabeled")
        elif (status in _USABLE and count != 1) or (status == "no_pose" and count != 0):
            raise ValueError("Pose count disagrees with original status")
        slots.append({"sample_index": i, "frame": frame["frame"], "evidence_id": frame["id"],
                      "source_time_s": frame["time_s"], "decoder_timestamp_s": frame.get("decoder_timestamp_s"),
                      "raw_status": status, "hypothesis_count": count, "window_id": None,
                      "screen_usable": status in _USABLE, "identity_barrier": competing})
        if competing:
            finish_window()
            barriers.append({"sample_index": i, "frame": frame["frame"], "hypothesis_indices": list(range(count)),
                             "selected_hypothesis": None, "continuity_across_barrier": "unresolved",
                             "interpolation_across_barrier_allowed": False})
        else:
            pending.append(i)
    finish_window()
    return {"profile": PROFILE, "status": "diagnostic_only", "source_sha256": source,
            "evidence_sha256": sha256(evidence_bytes).hexdigest(), "raw_pose_sha256": sha256(raw_pose_bytes).hexdigest(),
            "retained_inputs": {"evidence": evidence, "raw_poses": raw}, "screen_policy": dict(_SCREEN),
            "sample_slots": slots, "identity_barriers": barriers, "windows": windows,
            "source_time_basis": "unchanged_nominal_frame_index_divided_by_fps",
            "complete_native_clock_verified": False, "source_video_verified": False,
            "full_clip_body_qualified": False, "full_source_interval_retained": True,
            "simulation_ready": False, "selected_window": None,
            "not_evaluated": ["motion", "gap_limits", "stable_subject_identity", "camera", "root", "floor", "physics"],
            "scope": "Count/coverage/span screen of original sampled source windows, not pose or body validation. "
                     "All raw hypotheses and the full source interval remain visible. Window identity is conditional; "
                     "no interpolation, person selection, native-clock completion or task truncation is performed."}
