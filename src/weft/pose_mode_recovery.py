"""Compare retained pose hypotheses using source-only temporal rigidity, never contact scores."""

import math
from types import MappingProxyType

from weft.io import canonical_hash


METHOD = "source-rigidity-mode-recovery-v1"
CONTRACT = MappingProxyType({
    "method": METHOD, "maximum_samples": 900, "maximum_candidates": 32,
    "minimum_visibility": .5, "maximum_anchor_span_seconds": .2,
    "maximum_anchor_length_ratio": 1.25, "collapse_ratio": .5, "expansion_ratio": 2.,
    "minimum_image_error_torso_ratio": .25, "maximum_candidate_image_error_torso_ratio": .25,
    "maximum_candidate_length_ratio": 1.25, "minimum_image_error_reduction": .5,
    "maximum_other_joint_worsening_torso_ratio": .15, "maximum_root_error_torso_ratio": .25,
    "scope": "Alternative model estimates selected by adjacent original source consistency; not verified anatomy, identity or contact",
})
SEGMENTS = MappingProxyType({
    "left_upper_arm": (11, 13), "left_forearm": (13, 15),
    "right_upper_arm": (12, 14), "right_forearm": (14, 16),
    "left_thigh": (23, 25), "left_shin": (25, 27),
    "right_thigh": (24, 26), "right_shin": (26, 28),
})
JOINTS = (11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28)
TORSO = (11, 12, 23, 24)
USABLE = frozenset({"tracked", "tracked_partial"})


def _finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def _point(frame, index, dimensions):
    landmarks = frame.get("landmarks")
    if frame.get("pose_count") != 1 or not isinstance(landmarks, list) or len(landmarks) != 33:
        return None
    point = landmarks[index]
    if not isinstance(point, dict):
        raise ValueError("Pose recovery requires landmark records")
    values = [point.get(key) for key in ("x", "y", "visibility", "presence")]
    if not all(_finite(value) for value in values):
        raise ValueError("Pose recovery requires finite image coordinates and confidence")
    x, y, visibility, presence = values
    if not (0 <= visibility <= 1 and 0 <= presence <= 1):
        raise ValueError("Pose recovery confidence must be within [0, 1]")
    if not (0 <= x <= 1 and 0 <= y <= 1 and min(visibility, presence) >= CONTRACT["minimum_visibility"]):
        return None
    return x * dimensions[0], y * dimensions[1]


def _midpoint(first, second):
    return tuple((x + y) / 2 for x, y in zip(first, second, strict=True))


def _context(frame, dimensions):
    if type(frame.get("pose_count")) is not int or frame["pose_count"] != 1 or frame.get("status") not in USABLE:
        return None
    points = {index: _point(frame, index, dimensions) for index in JOINTS}
    if any(points[index] is None for index in TORSO):
        return None
    landmarks = frame.get("world_landmarks")
    if not isinstance(landmarks, list) or len(landmarks) != 33:
        return None
    world = {}
    for index in JOINTS:
        if not isinstance(landmarks[index], dict):
            raise ValueError("Pose recovery requires world landmark records")
        values = tuple(landmarks[index].get(key) for key in ("x", "y", "z"))
        if not all(_finite(value) for value in values):
            raise ValueError("Pose recovery requires finite world estimates")
        world[index] = values
    root = _midpoint(points[23], points[24])
    image_scale = math.dist(_midpoint(points[11], points[12]), root)
    world_scale = math.dist(_midpoint(world[11], world[12]), _midpoint(world[23], world[24]))
    if not all(math.isfinite(value) and value > 1e-6 for value in (image_scale, world_scale)):
        return None
    lengths = {name: math.dist(world[a], world[b]) / world_scale for name, (a, b) in SEGMENTS.items()}
    if not all(math.isfinite(value) and value > 1e-6 for value in lengths.values()):
        return None
    return {"points": points, "lengths": lengths, "root": root, "image_scale": image_scale}


def _validate_inputs(primary, alternative):
    if not isinstance(primary, dict) or not isinstance(alternative, dict):
        raise ValueError("Pose recovery requires two original evidence documents")
    source = primary.get("source_sha256")
    if type(source) is not str or len(source) != 64 or any(c not in "0123456789abcdef" for c in source):
        raise ValueError("Pose recovery requires an exact source hash")
    fields = ("source_sha256", "width", "height", "fps", "frame_count", "decoded_frames", "inference_stride")
    if any(primary.get(key) != alternative.get(key) for key in fields):
        raise ValueError("Alternative pose source, image domain or sampling differs")
    width, height, fps, decoded, stride = (primary.get(key) for key in fields[1:4] + fields[5:])
    count = primary.get("frame_count")
    if (type(width) is not int or type(height) is not int or not 64 <= width <= 4096 or not 64 <= height <= 2160
            or not _finite(fps) or not 1 <= fps <= 240 or type(decoded) is not int or not 10 <= decoded <= 1800
            or type(count) is not int or not 10 <= count <= 1800 or max(count, decoded) / fps > 60
            or type(stride) is not int or not 1 <= stride <= 120):
        raise ValueError("Pose recovery input exceeds the complete source bounds")
    left, right = primary.get("model", {}), alternative.get("model", {})
    if not isinstance(left, dict) or not isinstance(right, dict):
        raise ValueError("Pose recovery requires model identities")
    model_fields = ("sha256", "bytes", "model_version", "mediapipe_version", "delegate", "network_inference")
    if (any(key not in left or left[key] != right.get(key) for key in model_fields)
            or left.get("network_inference") is not False or left.get("delegate") != "CPU"
            or left.get("running_mode") != "VIDEO" or left.get("temporal_tracking") is not True
            or type(left.get("maximum_pose_hypotheses")) is not int or left["maximum_pose_hypotheses"] != 1
            or left.get("person_count_assumption") != "single_person_unverified"
            or right.get("running_mode") != "IMAGE" or right.get("temporal_tracking") is not False
            or type(right.get("maximum_pose_hypotheses")) is not int or right["maximum_pose_hypotheses"] != 2):
        raise ValueError("Recovery requires the same model/runtime: explicit one-person VIDEO and two-hypothesis IMAGE")
    expected = list(range(0, decoded, stride))
    if len(expected) > CONTRACT["maximum_samples"]:
        raise ValueError("Pose recovery sample bound exceeded")
    for document in (primary, alternative):
        frames = document.get("frames")
        if abs(count - decoded) > 1:
            audit = document.get("container_frame_audit")
            if (not isinstance(audit, dict) or audit.get("encoded_video_frames") != decoded
                    or audit.get("decoded_frames") != decoded or audit.get("matches") is not True):
                raise ValueError("Source frame discrepancy requires its original encoded-frame audit")
        if (document.get("profile") != "exercise-pose-v1" or document.get("schema_version") != 1
                or not isinstance(document.get("pose_identity"), dict) or document["pose_identity"].get("mode") != "raw"
                or "pose_reinference" in document or "pose_mode_recovery" in document
                or not isinstance(frames, list) or len(frames) != len(expected)):
            raise ValueError("Recovery requires complete original raw pose evidence")
        for index, row in zip(expected, frames, strict=True):
            if not isinstance(row, dict):
                raise ValueError("Pose recovery requires complete frame records")
            timestamp, native = row.get("time_s"), row.get("decoder_timestamp_s")
            if (type(row.get("frame")) is not int or row["frame"] != index or row.get("id") != f"{source}:frame:{index}"
                    or not _finite(timestamp) or abs(timestamp - index / fps) > 1e-7
                    or "decoder_timestamp_s" not in row or (native is not None and (not _finite(native) or not 0 <= native <= 60))
                    or any(key in row for key in ("identity_raw", "image_refinement", "image_refinement_raw",
                                                 "mode_refinement", "mode_refinement_raw"))):
                raise ValueError("Recovery cannot reindex, retime or reuse repaired pose observations")
    if any(a["decoder_timestamp_s"] != b["decoder_timestamp_s"] for a, b in zip(primary["frames"], alternative["frames"], strict=True)):
        raise ValueError("Alternative native source clocks differ")
    return width, height


def propose_pose_mode_recovery(primary, alternative):
    """Return complete decisions; only original neighbors define construction context."""
    dimensions = _validate_inputs(primary, alternative)
    frames, candidates = primary["frames"], alternative["frames"]
    contexts = [_context(row, dimensions) for row in frames]
    rows = []
    for index, (frame, candidate) in enumerate(zip(frames, candidates, strict=True)):
        row = {"frame": frame["frame"], "id": frame["id"], "time_s": frame["time_s"],
               "original_sha256": canonical_hash(frame), "candidate_sha256": canonical_hash(candidate),
               "eligible": False, "accepted": False, "reasons": [],
               "segments": {name: {"triggered": False, "reason": "context_unavailable"} for name in SEGMENTS},
               "anchor_frames": []}
        rows.append(row)
        if index == 0 or index == len(frames)-1:
            row["reasons"] = ["two_original_neighbors_unavailable"]
            continue
        before, original, after = contexts[index-1:index+2]
        span = frames[index+1]["time_s"] - frames[index-1]["time_s"]
        if any(value is None for value in (before, original, after)) or span > CONTRACT["maximum_anchor_span_seconds"]:
            row["reasons"] = ["usable_short_original_context_unavailable"]
            continue
        row["anchor_frames"] = [frames[index-1]["frame"], frames[index+1]["frame"]]
        fraction = (frame["time_s"] - frames[index-1]["time_s"]) / span
        mix = lambda a, b: (1-fraction)*a + fraction*b
        scale = mix(before["image_scale"], after["image_scale"])
        expected_points = {joint: tuple(mix(a, b) for a, b in zip(before["points"][joint], after["points"][joint], strict=True))
                           for joint in JOINTS if before["points"][joint] is not None and after["points"][joint] is not None}
        triggered = []
        for name, endpoints in SEGMENTS.items():
            detail = {"triggered": False, "reason": "visible_original_context_unavailable"}
            row["segments"][name] = detail
            if any(joint not in expected_points or original["points"][joint] is None for joint in endpoints):
                continue
            a, b = before["lengths"][name], after["lengths"][name]
            expected = mix(a, b)
            ratio = original["lengths"][name] / expected
            image_error = max(math.dist(original["points"][joint], expected_points[joint]) for joint in endpoints) / scale
            stable = max(a, b) / min(a, b) <= CONTRACT["maximum_anchor_length_ratio"]
            abnormal = ratio <= CONTRACT["collapse_ratio"] or ratio >= CONTRACT["expansion_ratio"]
            eligible = stable and abnormal and image_error >= CONTRACT["minimum_image_error_torso_ratio"]
            detail.update(triggered=eligible, reason="source_rigidity_and_image_outlier" if eligible else "no_joint_source_outlier",
                          anchor_length_ratio=max(a, b)/min(a, b), expected_length_ratio=expected,
                          original_length_to_expected=ratio, original_image_error_torso_ratio=image_error)
            if eligible:
                triggered.append(name)
        if not triggered:
            row["reasons"] = ["no_joint_source_outlier"]
            continue
        row["eligible"] = True
        proposed = _context(candidate, dimensions)
        if proposed is None:
            row["reasons"] = ["alternative_unusable_or_competing"]
            continue
        reasons = []
        lost = [joint for joint in JOINTS if original["points"][joint] is not None and proposed["points"][joint] is None]
        row["lost_visible_joints"] = lost
        if lost:
            reasons.append("previous_visible_joint_support_lost")
        for name in triggered:
            detail, endpoints = row["segments"][name], SEGMENTS[name]
            if any(proposed["points"][joint] is None for joint in endpoints):
                reasons.append("alternative_triggered_segment_unavailable")
                continue
            ratio = proposed["lengths"][name] / detail["expected_length_ratio"]
            error = max(math.dist(proposed["points"][joint], expected_points[joint]) for joint in endpoints) / scale
            detail.update(candidate_length_to_expected=ratio, candidate_image_error_torso_ratio=error)
            if not 1/CONTRACT["maximum_candidate_length_ratio"] <= ratio <= CONTRACT["maximum_candidate_length_ratio"]:
                reasons.append("alternative_rigidity_inconsistent")
            if (error > CONTRACT["maximum_candidate_image_error_torso_ratio"]
                    or error > detail["original_image_error_torso_ratio"] * (1-CONTRACT["minimum_image_error_reduction"])):
                reasons.append("alternative_image_outlier_not_resolved")
        worsened = []
        for joint, expected in expected_points.items():
            old, new = original["points"][joint], proposed["points"][joint]
            if old is not None and new is not None and (math.dist(new, expected)-math.dist(old, expected))/scale > CONTRACT["maximum_other_joint_worsening_torso_ratio"]:
                worsened.append(joint)
        row["worsened_visible_joints"] = worsened
        if worsened:
            reasons.append("other_joint_context_worsens")
        root_error = math.dist(proposed["root"], _midpoint(expected_points[23], expected_points[24])) / scale
        row["candidate_root_error_torso_ratio"] = root_error
        if root_error > CONTRACT["maximum_root_error_torso_ratio"]:
            reasons.append("alternative_root_inconsistent")
        row.update(accepted=not reasons, reasons=sorted(set(reasons)))
    exceeded = sum(row["eligible"] for row in rows) > CONTRACT["maximum_candidates"]
    if exceeded:
        for row in rows:
            row["accepted"] = False
            if row["eligible"]:
                row["reasons"] = sorted(set(row["reasons"] + ["candidate_bound_exceeded"]))
    return {"schema_version": 1, "method": METHOD, "contract": dict(CONTRACT), "bound_exceeded": exceeded,
            "source_sha256": primary["source_sha256"], "primary_sha256": canonical_hash(primary),
            "alternative_sha256": canonical_hash(alternative), "rows": rows,
            "accepted_frames": [row["frame"] for row in rows if row["accepted"]],
            "scope": CONTRACT["scope"], "simulation_ready": False}
