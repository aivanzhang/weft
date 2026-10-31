"""Fixed-camera source consistency; no geometry repair or physical-ground admission."""

from collections import Counter
from hashlib import sha256
import json
import math
from statistics import median
import time

import numpy as np

from weft.exercise_support_evidence import _json_size


PROFILE = "exercise-support-projection-v1"
CORE_POINTS = (11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28)
FOOT_AXES = ((29, 31), (30, 32))
CORE_AXES = ((11, 23), (11, 13), (13, 15), (23, 25), (25, 27),
             (12, 24), (12, 14), (14, 16), (24, 26), (26, 28))
CONTRACT = {
    "generation_fraction": .6, "maximum_pose_rows": 900, "maximum_decoded_frames": 1800,
    "maximum_input_bytes": 32 << 20, "maximum_output_bytes": 8 << 20,
    "maximum_seconds": 10., "minimum_confidence": .5, "minimum_core_endpoints": 6,
    "maximum_condition_number": 1e8, "minimum_foot_axis_coverage": .9,
    "maximum_median_direction_error_deg": 5., "maximum_absolute_world_coordinate": 10.,
    "axis_zero_tolerance": 1e-12,
    "camera": {"fx": 640., "fy": 640., "cx": 320., "cy": 240.,
               "width": 640, "height": 480, "radial_distortion": 0.,
               "rotation": "identity_in_original_model_camera_axes", "quality": "authored_unverified"},
    "scope": "Construction-only consistency with one fixed camera hypothesis; not recovered ground, contact or physical validation",
}


def _require(value, message):
    if not value:
        raise ValueError("Exercise support projection: " + message)


def _number(value):
    return type(value) is int or type(value) is float and math.isfinite(value)


def _check(deadline):
    if time.monotonic() >= deadline:
        raise TimeoutError("Exercise support projection exceeded its ten-second bound")


def _unique(pairs):
    result = {}
    for key, value in pairs:
        _require(key not in result, "duplicate JSON key")
        result[key] = value
    return result


def _invalid_constant(value):
    raise ValueError("Exercise support projection requires finite JSON numbers")


def _document(raw, deadline):
    _check(deadline)
    try:
        value = json.loads(raw, object_pairs_hook=_unique, parse_constant=_invalid_constant)
    except (UnicodeError, json.JSONDecodeError, RecursionError) as error:
        raise ValueError("Exercise support projection requires bounded valid JSON") from error
    _require(type(value) is dict, "JSON object required")
    _check(deadline)
    return value


def _prepare(pose_bytes, selection_bytes, deadline):
    _require(type(pose_bytes) is bytes and type(selection_bytes) is bytes, "exact input bytes required")
    _require(0 < len(pose_bytes) and 0 < len(selection_bytes)
             and len(pose_bytes) + len(selection_bytes) <= CONTRACT["maximum_input_bytes"], "input byte bound exceeded")
    pose, selection = _document(pose_bytes, deadline), _document(selection_bytes, deadline)
    source = pose.get("source_sha256")
    _require(type(source) is str and len(source) == 64 and set(source) <= set("0123456789abcdef"), "source SHA256 required")
    _require(pose.get("width") == 640 and pose.get("height") == 480, "fixed camera requires original 640x480 source")
    fps, decoded, declared = (pose.get(key) for key in ("fps", "decoded_frames", "frame_count"))
    _require(_number(fps) and 1 <= fps <= 120 and type(decoded) is int and type(declared) is int
             and 1 <= decoded <= CONTRACT["maximum_decoded_frames"]
             and 1 <= declared <= CONTRACT["maximum_decoded_frames"] + 8, "explicit bounded source clock/coverage required")
    _require(type(pose.get("pose_identity")) is dict and pose["pose_identity"].get("mode") == "raw", "original raw pose mode required")
    frames = pose.get("frames")
    _require(type(frames) is list and 1 <= len(frames) <= CONTRACT["maximum_pose_rows"], "1 to 900 pose rows required")
    previous, previous_time = -1, -1.
    for row in frames:
        _check(deadline)
        _require(type(row) is dict and type(row.get("frame")) is int and previous < row["frame"] < decoded
                 and row.get("id") == f"{source}:frame:{row['frame']}", "source row identity/order differs")
        stamp, native = row.get("time_s"), row.get("decoder_timestamp_s")
        _require(_number(stamp) and previous_time < stamp <= 60 and math.isclose(stamp, row["frame"] / fps, abs_tol=1e-9)
                 and (native is None or _number(native) and 0 <= native <= 61), "source row clock differs")
        previous, previous_time = row["frame"], stamp
    cutoff = math.floor(CONTRACT["generation_fraction"] * len(frames))
    _require(selection.get("source_sha256") == source and selection.get("method") == "stationary_candidate_foot_covariance_v1"
             and selection.get("supplied_pose_rows") == len(frames) and selection.get("construction_rows") == cutoff,
             "saved selection source/partition differs")
    _require(selection.get("selected_evidence_ids") == [row["id"] for row in frames[:cutoff]]
             and selection.get("construction_source_frames") == [row["frame"] for row in frames[:cutoff]]
             and selection.get("excluded_later_source_frames") == [row["frame"] for row in frames[cutoff:]],
             "saved selection ordered partition differs")
    audit = selection.get("audit")
    _require(type(audit) is list and len(audit) == cutoff, "saved selection audit coverage differs")
    for row, saved in zip(frames, audit):
        _check(deadline)
        _require(type(saved) is dict and saved.get("frame") == row["frame"] and saved.get("evidence_id") == row["id"]
                 and saved.get("status") in ("usable", "rejected") and type(saved.get("reasons")) is list
                 and all(type(reason) is str and len(reason) <= 256 for reason in saved["reasons"]), "saved selection audit differs")
    selected = [row["evidence_id"] for row in audit if row["status"] == "usable"]
    _require(selection.get("construction_evidence_ids") == selected and selection.get("usable_frames") == len(selected)
             and selection.get("rejected_frames") == cutoff - len(selected), "saved selection counts differ")
    support = selection.get("support_selection")
    _require(type(support) is dict and support.get("profile") == "exercise-support-evidence-v1"
             and support.get("source_sha256") == source, "source-bound support selection required")
    prior = support.get("camera_prior")
    _require(type(prior) is dict and prior.get("source_sha256") == source
             and prior.get("mode") == "fixed_source_camera" and prior.get("status") == "authored_unverified",
             "source-bound authored fixed-camera prior required")
    # Integrity traverses every byte; only selected rows enter geometric calculations.
    expected_size = _json_size(pose, deadline)
    _json_size(selection, deadline)
    canonical = json.dumps(pose, sort_keys=True, allow_nan=False).encode()
    _require(len(canonical) == expected_size, "canonical pose size differs")
    canonical_hash = sha256(canonical).hexdigest()
    _require(support.get("pose_evidence_canonical_sha256") == canonical_hash, "saved selection belongs to different pose bytes")
    _check(deadline)
    return pose, frames, audit, cutoff, {
        "source_sha256": source, "pose_sha256": sha256(pose_bytes).hexdigest(),
        "selection_sha256": sha256(selection_bytes).hexdigest(), "pose_canonical_sha256": canonical_hash,
    }


def _point(row, index):
    result = {"index": index, "status": "unavailable", "reason": None, "confidence": None,
              "observed_pixel": None, "original_world": None, "projected_pixel": None,
              "camera_depth_model_units": None, "pixel_residual": None}
    if row.get("status") not in ("tracked", "tracked_partial") or row.get("pose_count", 1) != 1:
        result["reason"] = "single_raw_pose_unavailable"
        return result
    if "image_refinement" in row or "identity_raw" in row:
        result["reason"] = "nonoriginal_pose_payload"
        return result
    image, world = row.get("landmarks"), row.get("world_landmarks")
    if not all(type(points) is list and len(points) == 33 and type(points[index]) is dict for points in (image, world)):
        result["reason"] = "landmark_record_unavailable"
        return result
    a, b = image[index], world[index]
    scores = [point.get(key) for point in (a, b) for key in ("visibility", "presence")]
    values = [a.get(key) for key in ("x", "y")] + [b.get(key) for key in ("x", "y", "z")]
    if not all(_number(v) for v in scores + values) or not all(0 <= v <= 1 for v in scores):
        result["reason"] = "nonfinite_or_invalid_landmark"
        return result
    result["confidence"] = min(scores)
    if not all(0 <= v <= 1 for v in values[:2]):
        result["reason"] = "outside_source_image"
    elif max(abs(v) for v in values[2:]) > CONTRACT["maximum_absolute_world_coordinate"]:
        result["reason"] = "world_coordinate_bound_exceeded"
    elif result["confidence"] < CONTRACT["minimum_confidence"]:
        result["reason"] = "low_endpoint_confidence"
    else:
        result.update(status="observed", observed_pixel=[values[0] * 640, values[1] * 480], original_world=values[2:])
    return result


def _translation(points):
    design, target = [], []
    for point in points:
        x, y, z = point["original_world"]
        u, v = point["observed_pixel"][0] - 320, point["observed_pixel"][1] - 240
        design.extend(((640., 0., -u), (0., 640., -v)))
        target.extend((u * z - 640 * x, v * z - 640 * y))
    translation, _, rank, singular = np.linalg.lstsq(np.asarray(design), np.asarray(target), rcond=None)
    condition = float(singular[0] / singular[-1]) if singular[-1] > 0 else None
    if condition is not None and not math.isfinite(condition):
        condition = None
    details = {"rank": int(rank), "condition_number": condition,
               "singular_values": singular.tolist(), "translation_model_units": None}
    if rank != 3:
        return None, details, "core_design_rank_unresolved"
    if condition is None or not math.isfinite(condition) or condition > CONTRACT["maximum_condition_number"]:
        return None, details, "core_design_condition_excessive"
    if not np.isfinite(translation).all():
        return None, details, "nonfinite_translation"
    details["translation_model_units"] = translation.tolist()
    return translation, details, None


def _project(point, translation):
    if point["status"] != "observed":
        return
    x, y, z = [value + float(offset) for value, offset in zip(point["original_world"], translation)]
    point["camera_depth_model_units"] = z if math.isfinite(z) else None
    if not all(math.isfinite(value) for value in (x, y, z)) or z <= 0:
        point.update(status="unavailable", reason="nonpositive_or_nonfinite_camera_depth")
        return
    projected = [320 + 640 * x / z, 240 + 640 * y / z]
    if not all(math.isfinite(value) for value in projected):
        point.update(status="unavailable", reason="nonfinite_projection")
        return
    point.update(status="projected", projected_pixel=projected, camera_depth_model_units=z,
                 pixel_residual=math.dist(projected, point["observed_pixel"]))


def _angle(a, b):
    lengths = math.hypot(*a) * math.hypot(*b)
    if min(math.hypot(*a), math.hypot(*b)) <= CONTRACT["axis_zero_tolerance"]:
        return None
    return math.degrees(math.acos(max(-1., min(1., sum(x * y for x, y in zip(a, b)) / lengths))))


def _axis(indices, points, row_reason):
    result = {"indices": list(indices), "status": "unavailable", "reason": row_reason,
              "direction_error_deg": None, "raw_world_xy_direction_error_deg": None}
    if row_reason is not None:
        return result
    a, b = (points[index] for index in indices)
    if a["status"] != "projected" or b["status"] != "projected":
        result["reason"] = "endpoint_projection_unavailable"
        return result
    observed = [y - x for x, y in zip(a["observed_pixel"], b["observed_pixel"])]
    projected = [y - x for x, y in zip(a["projected_pixel"], b["projected_pixel"])]
    world_xy = [y - x for x, y in zip(a["original_world"][:2], b["original_world"][:2])]
    angle = _angle(observed, projected)
    result["raw_world_xy_direction_error_deg"] = _angle(observed, world_xy)
    if angle is None:
        result["reason"] = "zero_length_image_or_projected_axis"
    else:
        result.update(status="evaluable", reason=None, direction_error_deg=angle)
    return result


def _selected_row(row, deadline):
    _check(deadline)
    points = {index: _point(row, index) for index in CORE_POINTS + (29, 30, 31, 32)}
    core = [points[index] for index in CORE_POINTS if points[index]["status"] == "observed"]
    solve = {"rank": None, "condition_number": None, "singular_values": [], "translation_model_units": None}
    reason, attempted = "insufficient_core_endpoints", False
    if len(core) >= CONTRACT["minimum_core_endpoints"]:
        attempted = True
        try:
            translation, solve, reason = _translation(core)
        except np.linalg.LinAlgError:
            translation, reason = None, "linear_solve_failed"
        if translation is not None:
            for point in points.values():
                _project(point, translation)
            if any(point["reason"] in ("nonpositive_or_nonfinite_camera_depth", "nonfinite_projection") for point in points.values()):
                reason = "row_has_invalid_camera_projection"
    _check(deadline)
    return {"status": "evaluated" if reason is None else "unavailable", "reason": reason,
            "core_endpoints_for_solve": [p["index"] for p in core], "linear_solve_attempted": attempted,
            "solve": solve, "core_endpoints": [points[index] for index in CORE_POINTS],
            "foot_endpoints": [points[index] for index in (29, 30, 31, 32)],
            "core_axes": [_axis(pair, points, reason) for pair in CORE_AXES],
            "foot_axes": [_axis(pair, points, reason) for pair in FOOT_AXES]}


def audit_exercise_support_projection(pose_document: bytes, selection_document: bytes):
    """Audit exact saved inputs once under the frozen camera; never change geometry."""
    deadline = time.monotonic() + CONTRACT["maximum_seconds"]
    pose, frames, audit, cutoff, bindings = _prepare(pose_document, selection_document, deadline)
    rows, foot_errors, core_errors, raw_foot_errors, solves = [], [], [], [], 0
    for index, row in enumerate(frames):
        _check(deadline)
        entry = {key: row.get(key) for key in ("id", "frame", "time_s", "decoder_timestamp_s")}
        if index >= cutoff:
            entry.update(status="excluded_later", selection_reasons=["excluded_later_source_row"], foot_axis_slots=2)
        else:
            saved = audit[index]
            entry.update(selection_status=saved["status"], selection_reasons=list(saved["reasons"]), foot_axis_slots=2)
            if saved["status"] != "usable":
                entry["status"] = "not_selected_construction"
            else:
                entry.update(_selected_row(row, deadline))
                solves += int(entry["linear_solve_attempted"])
                foot_errors.extend(a["direction_error_deg"] for a in entry["foot_axes"] if a["status"] == "evaluable")
                core_errors.extend(a["direction_error_deg"] for a in entry["core_axes"] if a["status"] == "evaluable")
                raw_foot_errors.extend(a["raw_world_xy_direction_error_deg"] for a in entry["foot_axes"]
                                       if a["status"] == "evaluable" and a["raw_world_xy_direction_error_deg"] is not None)
        rows.append(entry)
    selected = sum(row.get("selection_status") == "usable" for row in rows)
    expected_axes = 2 * selected
    coverage = len(foot_errors) / expected_axes if expected_axes else 0.
    foot_median = median(foot_errors) if foot_errors else None
    core_median = median(core_errors) if core_errors else None
    if coverage < CONTRACT["minimum_foot_axis_coverage"] or foot_median is None or core_median is None:
        status, reason = "inconclusive", "insufficient_evaluable_foot_coverage_or_core_control"
    elif max(foot_median, core_median) > CONTRACT["maximum_median_direction_error_deg"]:
        status, reason = "fixed_camera_hypothesis_rejected", "direction_error_exceeds_authored_diagnostic_limit"
    else:
        status, reason = "consistent_with_fixed_camera_hypothesis", "authored_direction_and_coverage_checks_pass"
    by_frame = {row["frame"]: row for row in rows}
    source_slots = []
    for frame in range(max(pose["decoded_frames"], pose["frame_count"])):
        _check(deadline)
        row = by_frame.get(frame)
        slot_status = row["status"] if row else "unsampled" if frame < pose["decoded_frames"] else "metadata_only_not_decoded"
        source_slots.append({"source_frame": frame, "time_s": frame / pose["fps"],
                             "decoder_timestamp_s": row.get("decoder_timestamp_s") if row else None,
                             "pose_evidence_id": row["id"] if row else None, "status": slot_status, "foot_axis_slots": 2})
    selected_rows = [row for row in rows if row.get("selection_status") == "usable"]
    points = [point for row in selected_rows for point in row["core_endpoints"] + row["foot_endpoints"]]
    result = {"schema_version": 1, "profile": PROFILE, "status": status, "reason": reason,
              "bindings": bindings, "contract": json.loads(json.dumps(CONTRACT)),
              "camera": {**CONTRACT["camera"], "source_sha256": bindings["source_sha256"]},
              "counts": {"supplied_pose_rows": len(frames), "construction_rows": cutoff, "selected_rows": selected,
                         "not_selected_construction_rows": cutoff - selected, "excluded_later_rows": len(frames) - cutoff,
                         "linear_solves": solves, "selected_foot_axis_slots": expected_axes,
                         "evaluable_foot_axes": len(foot_errors), "unavailable_foot_axes": expected_axes - len(foot_errors),
                         "selected_core_axis_slots": selected * len(CORE_AXES), "evaluable_core_axes": len(core_errors),
                         "selected_endpoint_slots": len(points), "positive_depth_endpoints": sum(p["status"] == "projected" for p in points),
                         "endpoint_status_counts": dict(Counter(p["status"] for p in points)),
                         "endpoint_reason_counts": dict(Counter(p["reason"] for p in points if p["reason"])),
                         "decoded_frames": pose["decoded_frames"], "declared_frames": pose["frame_count"],
                         "source_slot_status_counts": dict(Counter(slot["status"] for slot in source_slots)),
                         "retained_source_foot_axis_slots": 2 * len(source_slots)},
              "metrics": {"evaluable_foot_axis_fraction": coverage, "median_foot_direction_error_deg": foot_median,
                          "median_core_direction_error_deg": core_median,
                          "median_raw_world_xy_foot_direction_error_deg": median(raw_foot_errors) if raw_foot_errors else None},
              "rows": rows, "source_slots": source_slots,
              "ground": "unknown", "contact": "unknown", "gravity": "unknown", "physical_validation": False,
              "body_admission_changed": False, "geometry_modified": False,
              "limitations": ["Camera intrinsics and axes are authored, not measured calibration",
                              "Translation minimizes algebraic core-point residuals, not a floor or nonlinear reprojection objective",
                              "Core segments are fitted controls; held-out foot landmarks are construction evidence, not final evaluation",
                              "Rejection does not distinguish wrong intrinsics, local model error and real foot articulation",
                              "No source/model inference, media decode or native body construction is performed"]}
    _check(deadline)
    encoded = json.dumps(result, allow_nan=False, sort_keys=True).encode()
    _require(len(encoded) <= CONTRACT["maximum_output_bytes"], "output byte bound exceeded")
    _check(deadline)
    return result
