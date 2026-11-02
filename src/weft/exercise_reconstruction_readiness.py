"""Locate source uncertainty and bound existing recovery before another inference pass."""

from collections import Counter
from hashlib import sha256
import json
import math
import time

from weft.exercise_support_evidence import FOOT_POINTS, HANDS, _point as support_point
from weft.exercise_vision import USABLE_STATUSES, _quality
from weft.io import canonical_hash
from weft.pose_crop_selection import CONTRACT as CROP_CONTRACT, plan_selective_crop
from weft.pose_envelope import plan_envelope
from weft.pose_identity import repair_pose_identities
from weft.source_clock import clock_summary, frame_times, validate_clock
from weft.world_evidence import _check_tree, _nonfinite, _unique_object


PROFILE = "exercise-reconstruction-readiness-v1"
MAX_INPUT_BYTES = 32 * 1024 * 1024
MAX_OUTPUT_BYTES = 8 * 1024 * 1024
MAX_SECONDS = 5.


def _require(condition, message):
    if not condition:
        raise ValueError("Exercise reconstruction readiness: " + message)


def _check(deadline):
    if time.monotonic() >= deadline:
        raise TimeoutError("Exercise reconstruction readiness exceeded its bounded deadline")


def _inputs(raw):
    _require(type(raw) is bytes and 0 < len(raw) <= MAX_INPUT_BYTES, "bounded original evidence bytes required")
    try:
        evidence = json.loads(raw, object_pairs_hook=_unique_object, parse_constant=_nonfinite)
        _check_tree(evidence)
    except (ValueError, UnicodeError, RecursionError) as error:
        raise ValueError("Exercise reconstruction readiness requires bounded finite JSON") from error
    _require(isinstance(evidence, dict) and evidence.get("profile") == "exercise-pose-v1"
             and type(evidence.get("schema_version")) is int and evidence["schema_version"] == 1,
             "original exercise pose evidence required")
    source = evidence.get("source_sha256")
    _require(isinstance(source, str) and len(source) == 64 and set(source) <= set("0123456789abcdef"), "source SHA256 required")
    _require(not any(key in evidence for key in ("source_clock", "pose_reinference", "pose_recovery")),
             "use original detector evidence, not a chained derivation")
    decoded, metadata, fps = (evidence.get(key) for key in ("decoded_frames", "frame_count", "fps"))
    _require(type(decoded) is int and 1 <= decoded <= 1800 and type(metadata) is int and 1 <= metadata <= 1808
             and type(fps) in (int, float) and math.isfinite(fps) and 1 <= fps <= 240,
             "bounded decoded counts and nominal fps required")
    _require(all(type(evidence.get(key)) is int and 16 <= evidence[key] <= 8192 for key in ("width", "height")),
             "bounded original image dimensions required")
    rows = evidence.get("frames")
    _require(isinstance(rows, list) and 1 <= len(rows) <= 900, "retain all bounded sampled frames")
    previous = -1
    for row in rows:
        _require(isinstance(row, dict), "pose sample must be an object")
        frame, stamp = row.get("frame"), row.get("time_s")
        _require(type(frame) is int and previous < frame < decoded and row.get("id") == f"{source}:frame:{frame}"
                 and type(stamp) in (int, float) and math.isfinite(stamp) and 0 <= stamp <= 60
                 and math.isclose(stamp, frame / fps, rel_tol=0, abs_tol=1e-7), "source frame identity/order/nominal clock differs")
        _require(not any(key in row for key in ("identity_raw", "image_refinement", "image_refinement_raw")),
                 "retain original unmodified hypotheses")
        _require(isinstance(row.get("status"), str), "original pose status required")
        _require(type(row.get("pose_count")) is int and 0 <= row["pose_count"] <= 2,
                 "retain the original bounded pose-hypothesis count")
        if row["pose_count"] > 1:
            _require(isinstance(row.get("poses"), list) and len(row["poses"]) == row["pose_count"],
                     "all competing hypotheses must remain available")
        previous = frame
    return evidence


def _gaps(rows, times):
    gaps, current, previous_usable = [], [], None

    def finish(next_usable):
        if current:
            gaps.append({"first_frame": current[0]["frame"], "last_frame": current[-1]["frame"],
                "first_time_s": times[current[0]["frame"]], "last_time_s": times[current[-1]["frame"]],
                "observed_span_s": times[current[-1]["frame"]] - times[current[0]["frame"]],
                "sample_count": len(current), "evidence_ids": [row["id"] for row in current],
                "status_counts": dict(Counter(row["status"] for row in current)),
                "preceding_usable_evidence_id": previous_usable, "following_usable_evidence_id": next_usable,
                "interpolated_samples": 0})
            current.clear()

    for row in rows:
        if row["status"] in USABLE_STATUSES:
            finish(row["id"])
            previous_usable = row["id"]
        else:
            current.append(row)
    finish(None)
    return gaps


def diagnose_reconstruction(evidence_bytes, *, native_clock=None, max_seconds=MAX_SECONDS):
    """Return a source-only diagnostic; no evidence, quality gate or body is changed."""
    _require(type(max_seconds) in (int, float) and math.isfinite(max_seconds) and 0 < max_seconds <= MAX_SECONDS,
             "maximum_seconds must be in (0, 5]")
    started = time.monotonic()
    deadline = started + max_seconds
    evidence = _inputs(evidence_bytes)
    _check(deadline)
    evidence_sha = sha256(evidence_bytes).hexdigest()
    clock = validate_clock(native_clock, evidence, evidence_sha256=evidence_sha) if native_clock is not None else None
    times = frame_times(clock) if clock else [frame / evidence["fps"] for frame in range(evidence["decoded_frames"])]
    rows = evidence["frames"]
    try:
        quality = _quality(evidence)
    except (KeyError, IndexError, TypeError, ZeroDivisionError) as error:
        raise ValueError("Exercise reconstruction readiness requires complete retained pose geometry/confidence") from error
    _require("quality" not in evidence or canonical_hash(evidence["quality"]) == canonical_hash(quality),
             "saved quality differs from unchanged source gates")
    plans = []
    for row in rows:
        _check(deadline)
        plans.append(plan_selective_crop(evidence, row))
    eligible = [plan for plan in plans if plan["eligible"]]
    indexed = {row["frame"]: row for row in rows}
    recoverable = [plan["frame"] for plan in eligible if indexed[plan["frame"]]["status"] not in USABLE_STATUSES]
    required = max(quality["minimum_tracked_frames"], math.ceil(quality["minimum_tracked_fraction"] * len(rows)))
    usable = quality["tracked_frames"]
    ambiguous = [row for row in rows if row["status"] == "multiple_people" or row.get("pose_count", 0) > 1]
    ambiguous_frames = {row["frame"] for row in ambiguous}
    _, identity = repair_pose_identities(rows)
    try:
        envelope = {"available": True, "plan": plan_envelope(evidence, evidence)}
    except ValueError as error:
        envelope = {"available": False, "reason": str(error)}
    _check(deadline)
    points = {**FOOT_POINTS, **HANDS}
    point_counts = {name: Counter() for name in points.values()}
    slots = []
    for frame, stamp in enumerate(times):
        row = indexed.get(frame)
        if row is None:
            slots.append({"frame": frame, "time_s": stamp, "state": "not_sampled", "evidence_id": None,
                          "support_points": None, "contact_state": "unknown"})
            continue
        available = {}
        for point, name in points.items():
            value = support_point(row, point, evidence["width"], evidence["height"])
            reason = value["reason"] or "visible_image_estimate"
            point_counts[name][reason] += 1
            available[name] = {"available": value["pixel"] is not None, "confidence": value["confidence"], "reason": value["reason"]}
        slots.append({"frame": frame, "time_s": stamp, "nominal_time_s": row["time_s"],
            "decoder_timestamp_s": row.get("decoder_timestamp_s"), "evidence_id": row["id"], "status": row["status"],
            "state": "ambiguous_identity" if frame in ambiguous_frames else "usable_pose" if row["status"] in USABLE_STATUSES else "unusable_pose",
            "support_points": available, "contact_state": "unknown"})
    within_budget = len(eligible) <= CROP_CONTRACT["maximum_target_frames"]
    ceiling = usable + len(recoverable)
    next_actions = []
    if ambiguous:
        next_actions.append({"kind": "resolve_source_identity", "frames": [row["frame"] for row in ambiguous],
            "reason": "Existing conservative recovery preserves competing hypotheses; none are silently selected."})
    if ceiling < required:
        next_actions.append({"kind": "obtain_new_supported_pose_context", "minimum_additional_usable_samples": required - ceiling,
            "reason": "Even perfect existing selective crops cannot meet the unchanged visibility gate."})
    next_actions.append({"kind": "establish_camera_and_support_evidence",
        "reason": "Visible feet/wrists alone cannot establish ground, equipment contact or camera motion; use source-bound support tools only with their declared camera assumptions."})
    result = {"schema_version": 1, "profile": PROFILE, "source_sha256": evidence["source_sha256"],
        "evidence_sha256": evidence_sha, "evidence_canonical_sha256": canonical_hash(evidence),
        "status": "source_quality_passes_only" if quality["passes"] else "source_quality_rejected",
        "physical_validation": "unverified", "body_admitted": False,
        "clock": clock_summary(clock) if clock else {"basis": "frame_index_divided_by_fps", "status": "assumed"},
        "source_quality": quality, "coverage": {"decoded_frames": evidence["decoded_frames"], "metadata_frames": evidence["frame_count"],
            "sampled_frames": len(rows), "unsampled_decoded_frames": evidence["decoded_frames"] - len(rows),
            "metadata_only_slots": list(range(evidence["decoded_frames"], evidence["frame_count"])),
            "decoded_beyond_metadata": list(range(evidence["frame_count"], evidence["decoded_frames"]))},
        "slots": slots, "unusable_spans": _gaps(rows, times),
        "identity": {"competing_samples": [{"frame": row["frame"], "evidence_id": row["id"],
            "retained_hypotheses": len(row.get("poses", [])), "selected_hypothesis": None} for row in ambiguous],
            "temporal_repair_candidate_ids": identity["repaired_evidence_ids"], "changes_applied": 0},
        "selective_crop": {"method": CROP_CONTRACT["method"], "plans": plans, "eligible_frames": [plan["frame"] for plan in eligible],
            "recoverable_unusable_frames": recoverable, "within_existing_candidate_bound": within_budget,
            "minimum_required_usable_samples": required, "current_usable_samples": usable,
            "optimistic_usable_ceiling": ceiling, "optimistic_fraction": ceiling / len(rows),
            "could_clear_visibility_gate": within_budget and ceiling >= required,
            "retry_can_address_all_source_rejections": not quality["passes"] and bool(recoverable)
                and within_budget and ceiling >= required and not ambiguous
                and all("visibility" in error for error in quality["errors"]),
            "scope": "Upper bound only: assumes every eligible unusable crop becomes usable; no success prediction or retry executed."},
        "envelope": envelope, "support_observability": {"point_counts": {key: dict(value) for key, value in point_counts.items()},
            "camera_motion": "unknown_not_inferred_from_pose", "equipment_observations": "not_supplied",
            "contact_state": "unknown", "ground_height": None, "metric_scale": None, "verified_contact": False},
        "next_actions": next_actions, "work": {"inference_calls": 0, "source_decodes": 0, "physics_steps": 0, "paid_calls": 0},
        "scope": "Exposed-source reconstruction diagnostic, not held-out evaluation or permission to weaken quality/contact gates."}
    _require(len(json.dumps(result, allow_nan=False).encode()) <= MAX_OUTPUT_BYTES, "diagnostic output exceeds byte bound")
    _check(deadline)
    return result
