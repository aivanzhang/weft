"""Conservative temporal bilateral identity repair; raw detections remain immutable."""

from copy import deepcopy
import math

import numpy as np


BILATERAL_PAIRS = ((1, 4), (2, 5), (3, 6), (7, 8), (9, 10)) + tuple((i, i + 1) for i in range(11, 33, 2))
IDENTITY_INDEX_MAPPING = tuple(next((b if i == a else a for a, b in BILATERAL_PAIRS if i in (a, b)), i)
                               for i in range(33))
BODY_INDICES = (11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28)
TORSO_INDICES = (11, 12, 23, 24)
DEFAULT_IDENTITY_CONFIG = {
    "max_span_frames": 3, "max_span_seconds": .20, "max_adjacent_gap_seconds": .12,
    "minimum_confidence": .8, "minimum_torso_length": .06, "minimum_pair_separation_ratio": .15,
    "minimum_reliable_body_points": 8, "maximum_anchor_torso_motion": .30,
    "minimum_torso_improvement": .20, "maximum_corrected_torso_error": .12,
    "maximum_corrected_body_error": .35, "maximum_error_ratio": .35,
}
_DERIVED_FIELDS = ("joints", "lengths", "joint_confidence", "out_of_plane_fraction", "usable_required_landmarks")


def _configuration(overrides):
    config = dict(DEFAULT_IDENTITY_CONFIG)
    if overrides is not None:
        if not isinstance(overrides, dict) or set(overrides) - set(config):
            raise ValueError("Unknown pose identity configuration")
        config.update(overrides)
    for key, value in config.items():
        if type(value) not in (float, int) or not math.isfinite(value) or value <= 0:
            raise ValueError("Pose identity bounds must be finite positive numbers")
    if (type(config["max_span_frames"]) is not int or not 1 <= config["max_span_frames"] <= 3
            or type(config["minimum_reliable_body_points"]) is not int
            or not 8 <= config["minimum_reliable_body_points"] <= 12
            or config["max_span_seconds"] > .20 or config["max_adjacent_gap_seconds"] > .12
            or config["minimum_confidence"] > 1 or config["maximum_error_ratio"] >= 1):
        raise ValueError("Pose identity configuration exceeds the conservative repair envelope")
    return config


def _sample(frame, config):
    if (frame.get("pose_count") != 1 or frame.get("status") not in {"tracked", "tracked_partial"}
            or isinstance(frame.get("poses"), list) and len(frame["poses"]) > 1):
        return None, "not_a_usable_single_pose"
    image, world = frame.get("landmarks"), frame.get("world_landmarks")
    if not isinstance(image, list) or not isinstance(world, list) or len(image) != 33 or len(world) != 33:
        return None, "missing_landmarks"
    try:
        points = np.array([[row["x"], row["y"]] for row in image], dtype=float)
        world_points = np.array([[row[key] for key in ("x", "y", "z")] for row in world], dtype=float)
        confidence = np.array([min(row.get("visibility", 0.), row.get("presence", 0.)) for row in image], dtype=float)
    except (KeyError, TypeError, ValueError):
        return None, "invalid_landmarks"
    if (not np.isfinite(points).all() or not np.isfinite(world_points).all() or not np.isfinite(confidence).all()
            or np.any(confidence < 0) or np.any(confidence > 1)):
        return None, "invalid_landmarks"
    confidence[np.any((points < -.05) | (points > 1.05), axis=1)] = 0.
    torso_length = float(np.linalg.norm((points[11] + points[12] - points[23] - points[24]) / 2))
    if torso_length < config["minimum_torso_length"]:
        return None, "insufficient_torso_extent"
    if np.min(confidence[list(TORSO_INDICES)]) < config["minimum_confidence"]:
        return None, "insufficient_torso_confidence"
    if min(np.linalg.norm(points[a] - points[b]) for a, b in ((11, 12), (23, 24))) / torso_length < config["minimum_pair_separation_ratio"]:
        return None, "insufficient_torso_pair_separation"
    return {"points": points, "confidence": confidence, "torso_length": torso_length}, None


def _rms(delta, weights):
    return float(np.sqrt(np.sum(np.sum(delta * delta, axis=1) * weights) / np.sum(weights)))


def _compare(sample, left, right, fraction, scale, config):
    expected = (1 - fraction) * left["points"] + fraction * right["points"]
    # Both hypotheses use the same confident endpoints; swapping cannot hide poor visibility.
    confidence = np.minimum.reduce((sample["confidence"], sample["confidence"][list(IDENTITY_INDEX_MAPPING)],
                                    left["confidence"], right["confidence"]))
    indices = [i for i in BODY_INDICES if confidence[i] >= config["minimum_confidence"]]
    if len(indices) < config["minimum_reliable_body_points"]:
        return None
    weights = np.array([3. if i in TORSO_INDICES else 1. for i in indices]) * confidence[indices]
    torso = list(TORSO_INDICES)
    raw = sample["points"] - expected
    swapped = sample["points"][list(IDENTITY_INDEX_MAPPING)] - expected
    return {"raw_body_error": _rms(raw[indices], weights) / scale,
            "swapped_body_error": _rms(swapped[indices], weights) / scale,
            "raw_torso_error": _rms(raw[torso], confidence[torso]) / scale,
            "swapped_torso_error": _rms(swapped[torso], confidence[torso]) / scale,
            "reliable_body_points": len(indices)}


def repair_pose_identities(frames, *, config=None):
    """Return separate corrected frames and an audit; caller recomputes derived geometry."""
    config = _configuration(config)
    if not isinstance(frames, list) or not 1 <= len(frames) <= 1800:
        raise ValueError("Identity repair requires 1–1800 retained frames")
    times, seen = [], set()
    for frame in frames:
        if not isinstance(frame, dict) or "identity_raw" in frame or frame.get("identity_recompute_required"):
            raise ValueError("Identity repair requires raw, not previously corrected frames")
        identifier, timestamp = frame.get("id"), frame.get("time_s")
        if (not isinstance(identifier, str) or not identifier or identifier in seen
                or type(timestamp) not in (int, float) or not math.isfinite(timestamp)
                or not 0 <= timestamp <= 60 or times and timestamp <= times[-1]):
            raise ValueError("Identity repair requires unique evidence IDs and ordered finite times")
        seen.add(identifier)
        times.append(timestamp)
    prepared = [_sample(frame, config) for frame in frames]
    samples = [item[0] for item in prepared]
    unavailable = [{"evidence_id": frame["id"], "reason": item[1]} for frame, item in zip(frames, prepared) if item[1]]
    candidates, decisions = [], []
    for first in range(2, len(frames) - 2):
        for size in range(1, config["max_span_frames"] + 1):
            last = first + size - 1
            if last + 2 >= len(frames):
                break
            support = (first - 2, first - 1, last + 1, last + 2)
            window = list(range(first - 2, last + 3))
            if any(samples[index] is None for index in window):
                continue
            left, right = samples[first - 1], samples[last + 1]
            scale = (left["torso_length"] + right["torso_length"]) / 2
            comparisons = []
            for index in range(first, last + 1):
                fraction = (times[index] - times[first - 1]) / (times[last + 1] - times[first - 1])
                comparison = _compare(samples[index], left, right, fraction, scale, config)
                if comparison is None:
                    break
                comparisons.append({"evidence_id": frames[index]["id"], **comparison})
            if len(comparisons) != size:
                continue
            # Ordinary motion does not generate a noisy audit entry for every scanned span.
            if not all(row["raw_torso_error"] - row["swapped_torso_error"] > config["minimum_torso_improvement"] / 2
                       for row in comparisons):
                continue
            reasons = []
            if times[last] - times[first] > config["max_span_seconds"]:
                reasons.append("span_too_long")
            if any(times[b] - times[a] > config["max_adjacent_gap_seconds"] for a, b in zip(window, window[1:])):
                reasons.append("temporal_gap")
            torso = list(TORSO_INDICES)
            anchor_motion = max(_rms(samples[a]["points"][torso] - samples[b]["points"][torso], np.ones(4)) / scale
                                for a, b in ((support[0], support[1]), (support[1], support[2]), (support[2], support[3])))
            if anchor_motion > config["maximum_anchor_torso_motion"]:
                reasons.append("anchors_do_not_support_local_continuity")
            for row in comparisons:
                if (row["raw_torso_error"] - row["swapped_torso_error"] < config["minimum_torso_improvement"]
                        or row["swapped_torso_error"] > config["maximum_corrected_torso_error"]
                        or row["swapped_body_error"] > config["maximum_corrected_body_error"]
                        or row["swapped_torso_error"] > config["maximum_error_ratio"] * row["raw_torso_error"]
                        or row["swapped_body_error"] > config["maximum_error_ratio"] * row["raw_body_error"]):
                    reasons.append("insufficient_continuity_margin")
                    break
            decision = {"evidence_ids": [frames[i]["id"] for i in range(first, last + 1)],
                        "support_evidence_ids": [frames[i]["id"] for i in support],
                        "duration_seconds": times[last] - times[first], "anchor_torso_motion": anchor_motion,
                        "comparisons": comparisons, "status": "abstained" if reasons else "supported_candidate",
                        "reasons": reasons}
            decisions.append(decision)
            if not reasons:
                candidates.append((first, last, support, decision))
    blocked, active = set(), []
    for a, (first, last, support, _) in enumerate(candidates):
        active = [b for b in active if candidates[b][1] + 2 >= first]
        indices = set(range(first, last + 1))
        for b in active:
            other_first, other_last, other_support, _ = candidates[b]
            other = set(range(other_first, other_last + 1))
            if indices & (other | set(other_support)) or other & set(support):
                blocked.update((a, b))
        active.append(a)
    corrected = deepcopy(frames)
    repaired_ids = []
    for candidate, (first, last, _, decision) in enumerate(candidates):
        if candidate in blocked:
            decision.update(status="abstained", reasons=["conflicting_candidate_or_anchor_identity"])
            continue
        decision.update(status="repaired", reasons=["short_isolated_bilateral_flip_with_two_anchors_on_each_side"])
        for index in range(first, last + 1):
            original, result = frames[index], corrected[index]
            result["identity_raw"] = deepcopy(original)
            for key in ("landmarks", "world_landmarks"):
                result[key] = [deepcopy(original[key][source]) for source in IDENTITY_INDEX_MAPPING]
            for key in _DERIVED_FIELDS:
                result.pop(key, None)
            result["identity_recompute_required"] = True
            result["identity_status"] = "temporally_repaired_bilateral_assignment_pending_geometry_recompute"
            repaired_ids.append(original["id"])
    return corrected, {"schema_version": 1, "method": "short-island-bilateral-continuity-v1", "config": config,
                       "index_mapping": list(IDENTITY_INDEX_MAPPING), "repaired_evidence_ids": repaired_ids,
                       "decisions": decisions, "unavailable_frames": unavailable,
                       "scope": "Temporal anatomical identity inference; initial global left/right assignment remains unverified",
                       "assumptions": ["Normalized image x/y distances, torso-normalized; no screen-left anatomical rule",
                                       "Only high-confidence torso and limb landmarks support correction; face geometry does not",
                                       "Two original usable anchors on each side and a strong per-frame continuity margin are required",
                                       "Multiple poses, uncertain torso geometry and long gaps are never merged or filled",
                                       "A rapid real reversal indistinguishable from a bilateral model flip remains an observational ambiguity",
                                       "Original status is retained until caller recomputes all geometry and confidence from corrected arrays",
                                       "All original detection and derived fields remain intact under identity_raw for corrected samples"]}
