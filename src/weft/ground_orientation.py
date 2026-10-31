"""Authored screening for a camera-relative foot plane, not measured gravity."""

from copy import deepcopy
import math
import re

import numpy as np


CAMERA_TO_SIM = np.array([[1., 0., 0.], [0., 0., 1.], [0., -1., 0.]])
FOOT_INDICES = (29, 30, 31, 32)
SETTINGS = {
    "generation_fraction": .6,
    "maximum_pose_rows": 900,
    "minimum_usable_frames": 3,
    "minimum_endpoint_confidence": .5,
    "minimum_second_to_first_eigenvalue_ratio": .05,
    "maximum_third_to_second_eigenvalue_ratio": .25,
    "minimum_normal_z": .25,
    "maximum_median_normal_angle_deg": 15.,
    "maximum_p90_normal_angle_deg": 30.,
    "numerical_rank_tolerance": 1e-12,
    "maximum_source_time_s": 60.,
    "maximum_absolute_world_coordinate": 10.,
    "maximum_absolute_image_coordinate": 10.,
}
GROUND_ORIENTATION_CONTRACT = SETTINGS


def _number(value):
    return type(value) in (int, float) and math.isfinite(value)


def _positive_up(normal):
    return -normal if normal[2] < 0 else normal


def _foot_frame(row, indices=FOOT_INDICES):
    reasons = []
    if len(indices) < 3:
        return None, {"reasons": ["insufficient_stationary_ground_candidates"]}
    if row.get("status") not in ("tracked", "tracked_partial"):
        return None, {"reasons": ["observation_not_tracked"]}
    image, world = row.get("landmarks"), row.get("world_landmarks")
    if not isinstance(image, list) or not isinstance(world, list) or len(image) != 33 or len(world) != 33:
        return None, {"reasons": ["missing_33_landmark_records"]}
    points, confidence, image_points = [], [], []
    for index in indices:
        a, b = image[index], world[index]
        if not isinstance(a, dict) or not isinstance(b, dict):
            return None, {"reasons": ["malformed_foot_landmark"]}
        values = [a.get(k) for k in ("x", "y")] + [b.get(k) for k in ("x", "y", "z")]
        scores = [record.get(k) for record in (a, b) for k in ("presence", "visibility")]
        if not all(_number(v) for v in values + scores):
            return None, {"reasons": ["nonfinite_or_missing_foot_values"]}
        if any(abs(v) > SETTINGS["maximum_absolute_image_coordinate"] for v in values[:2]):
            reasons.append("image_coordinate_magnitude_excessive")
        if any(abs(v) > SETTINGS["maximum_absolute_world_coordinate"] for v in values[2:]):
            reasons.append("world_coordinate_magnitude_excessive")
        if any(not 0 <= score <= 1 for score in scores):
            reasons.append("confidence_out_of_range")
        confidence.append(min(scores))
        image_points.append(values[:2])
        points.append(values[2:])
    if min(confidence) < SETTINGS["minimum_endpoint_confidence"]:
        reasons.append("low_endpoint_confidence")
    if any(not 0 <= coordinate <= 1 for point in image_points for coordinate in point):
        reasons.append("foot_image_out_of_bounds")
    audit = {"endpoint_confidence": confidence, "image_points": image_points, "reasons": sorted(set(reasons))}
    if reasons:
        return None, audit
    points = np.asarray(points, dtype=float) @ CAMERA_TO_SIM.T
    # Each model-world frame is hip-relative; pooling untranslated points invents plane thickness.
    with np.errstate(over="ignore", invalid="ignore"):
        centered = points-points.mean(axis=0)
        scale = float(np.max(np.abs(centered)))
    if not math.isfinite(scale) or scale <= 0:
        audit["reasons"].append("degenerate_or_nonfinite_foot_spread")
        return None, audit
    centered /= scale
    covariance = centered.T @ centered
    trace = float(np.trace(covariance))
    if not math.isfinite(trace) or trace <= 0:
        audit["reasons"].append("degenerate_or_nonfinite_covariance")
        return None, audit
    # Trace normalization gives each accepted frame equal weight, independent of inferred scale.
    covariance /= trace
    eigenvalues, axes = np.linalg.eigh(covariance)
    eigenvalues = np.maximum(eigenvalues, 0.)
    audit["normalized_eigenvalues_ascending"] = eigenvalues.tolist()
    if eigenvalues[1] <= SETTINGS["numerical_rank_tolerance"]:
        audit["reasons"].append("per_frame_rank2_unresolved")
        return None, audit
    normal = _positive_up(axes[:, 0])
    audit["normal_sim"] = normal.tolist()
    return (covariance, normal), audit


def estimate_ground_orientation(evidence, *, support_prior=None):
    """Screen only the first floor(0.6*N) ordered pose rows; never read floor scores."""
    if not isinstance(evidence, dict) or not isinstance(evidence.get("frames"), list):
        raise ValueError("Ground orientation requires a pose evidence object with frames")
    frames = evidence["frames"]
    source_hash = evidence.get("source_sha256")
    if not isinstance(source_hash, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", source_hash):
        raise ValueError("Ground orientation requires a 64-hex source_sha256")
    if len(frames) > SETTINGS["maximum_pose_rows"]:
        raise ValueError("Ground orientation exceeds the 900-row bound")
    indices = [row.get("frame") if isinstance(row, dict) else None for row in frames]
    if any(type(index) is not int or index < 0 for index in indices) or any(a >= b for a, b in zip(indices, indices[1:])):
        raise ValueError("Pose rows must have unique increasing source frame indices")
    if any(row.get("id") != f"{source_hash}:frame:{row['frame']}" for row in frames):
        raise ValueError("Pose evidence IDs must match source_sha256 and frame index")
    times = [row.get("time_s") for row in frames]
    if any(not _number(t) or not 0 <= t <= SETTINGS["maximum_source_time_s"] for t in times) or any(a >= b for a, b in zip(times, times[1:])):
        raise ValueError("Pose times must be finite, increasing and between 0 and 60 seconds")
    cutoff = math.floor(SETTINGS["generation_fraction"] * len(frames))
    selected = frames[:cutoff]
    support = None
    if support_prior is not None:
        from weft.exercise_support_evidence import estimate_exercise_support, _pose_row
        support = estimate_exercise_support(evidence, camera_prior=support_prior)
        if support["counts"]["construction_rows"] != cutoff:
            raise ValueError("Ground/support construction partitions differ")
    audit, usable = [], []
    for index, row in enumerate(selected):
        foot_indices, geometry_row = FOOT_INDICES, row
        if support is not None:
            foot_indices = tuple(point["index"] for point in support["rows"][index]["ground_points"]
                                 if point["status"] == "stationary_image_candidate")
            geometry_row, _ = _pose_row(row)
        geometry, details = _foot_frame(geometry_row, foot_indices)
        if support is not None:
            details["stationary_candidate_indices"] = list(foot_indices)
        entry = {"frame": row["frame"], "evidence_id": row.get("id"), **details,
                 "status": "usable" if geometry is not None else "rejected"}
        audit.append(entry)
        if geometry is not None:
            usable.append((row, *geometry))
    reasons = []
    if len(usable) < SETTINGS["minimum_usable_frames"]:
        reasons.append("too_few_usable_generation_frames")
    normal, metrics = None, {}
    if usable:
        covariance = np.mean([item[1] for item in usable], axis=0)
        values, axes = np.linalg.eigh(covariance)
        values = np.maximum(values, 0.)
        normal = _positive_up(axes[:, 0])
        rank_ratio = float(values[1]/values[2]) if values[2] > 0 else 0.
        thickness_ratio = float(values[0]/values[1]) if values[1] > SETTINGS["numerical_rank_tolerance"] else None
        angles = [math.degrees(math.acos(float(np.clip(np.dot(normal, item[2]), -1, 1)))) for item in usable]
        median, p90 = float(np.median(angles)), float(np.percentile(angles, 90))
        metrics = {"normalized_covariance": covariance.tolist(), "eigenvalues_ascending": values.tolist(),
                   "second_to_first_eigenvalue_ratio": rank_ratio,
                   "third_to_second_eigenvalue_ratio": thickness_ratio,
                   "median_normal_angle_deg": median, "p90_normal_angle_deg": p90,
                   "maximum_normal_angle_deg": max(angles), "normal_z": float(normal[2]),
                   "camera_up_tilt_deg": math.degrees(math.acos(float(np.clip(normal[2], -1, 1))))}
        for degrees, entry in zip(angles, [entry for entry in audit if entry["status"] == "usable"]):
            entry["angle_to_aggregate_deg"] = degrees
        if rank_ratio < SETTINGS["minimum_second_to_first_eigenvalue_ratio"]:
            reasons.append("aggregate_rank2_spread_insufficient")
        if thickness_ratio is None or thickness_ratio > SETTINGS["maximum_third_to_second_eigenvalue_ratio"]:
            reasons.append("aggregate_plane_thickness_excessive")
        if normal[2] < SETTINGS["minimum_normal_z"]:
            reasons.append("normal_too_far_from_camera_up")
        if median > SETTINGS["maximum_median_normal_angle_deg"]:
            reasons.append("median_normal_dispersion_excessive")
        if p90 > SETTINGS["maximum_p90_normal_angle_deg"]:
            reasons.append("p90_normal_dispersion_excessive")
    diagnostic = normal.tolist() if normal is not None else None
    result = {"schema_version": 1, "method": "centered_equal_frame_foot_covariance_v1",
            "status": "unavailable" if reasons else "available", "normal": diagnostic if not reasons else None,
            "diagnostic_normal": diagnostic, "reasons": reasons, "metrics": metrics, "audit": audit,
            "settings": dict(SETTINGS), "supplied_pose_rows": len(frames), "construction_rows": cutoff,
            "usable_frames": len(usable), "rejected_frames": cutoff-len(usable),
            "selected_evidence_ids": [row.get("id") for row in selected],
            "construction_evidence_ids": [row.get("id") for row, _, _ in usable],
            "construction_source_frames": [row["frame"] for row in selected],
            "excluded_later_source_frames": indices[cutoff:], "model": deepcopy(evidence.get("model")),
            "source_sha256": evidence.get("source_sha256"),
            "coordinate_system": "Camera-relative simulator axes: x=camera x, y=camera z, z=-camera y",
            "camera_to_sim": CAMERA_TO_SIM.tolist(), "normal_sign": "Positive camera-up, not measured gravity",
            "scope": "Authored screening limits, not physical truth; model-world heel/toe geometry only",
            "floor_offset": None, "floor_offset_scope": "Separate first-pose-only calibration; not estimated here"}
    if support is not None:
        result.update(method="stationary_candidate_foot_covariance_v1", support_selection=support,
                      scope="Model-world heel/toe plane conditioned on stationary image candidates; not measured ground, contact or gravity")
    return result
