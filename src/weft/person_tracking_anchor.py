"""A partial pose can seed tracking without promoting its occluded landmarks."""

from copy import deepcopy
import math

from weft.io import canonical_hash

PROFILE = "sealed-pose-visible-chain-tracking-anchor-v2"
POLICY = {"profile": "confidence-qualified-connected-pose-envelope-v1", "minimum_confidence": .5,
          "core_indices": [0, 11, 12, 23, 24], "minimum_torso_span_px": 8.,
          "arm_chains": {"left": [11, 13, 15], "right": [12, 14, 16]},
          "leg_chains": {"left": [23, 25, 27], "right": [24, 26, 28]},
          "minimum_complete_arm_chains": 1, "minimum_complete_leg_chains": 1,
          "padding_fraction": .05, "minimum_padding_px": 2, "minimum_roi_side_px": 8}


def derive_visible_chain_anchor(observation, *, width, height):
    """Derive one source-frame-zero hint; all 33 source estimates remain inspectable."""
    if (type(width) is not int or type(height) is not int or not 16 <= width <= 4096
            or not 16 <= height <= 4096 or width*height > 8_847_360):
        raise ValueError("Person anchor requires bounded original image dimensions")
    if (not isinstance(observation, dict) or type(observation.get("frame")) is not int or observation["frame"] != 0
            or observation.get("partition") != "construction" or observation.get("visibility") not in {"detected", "partial"}
            or not isinstance(observation.get("id"), str) or not 1 <= len(observation["id"]) <= 512
            or not isinstance(observation.get("evidence_ids"), list) or not 1 <= len(observation["evidence_ids"]) <= 4
            or any(not isinstance(item, str) or not 1 <= len(item) <= 512 for item in observation["evidence_ids"])):
        raise ValueError("Person anchor requires exact construction evidence at original frame zero")
    measure = observation.get("measurements", {}).get("landmarks", {})
    if tuple(measure.get(k) for k in ("coordinate_system", "unit", "status")) != ("image", "pixels", "inferred"):
        raise ValueError("Person anchor requires inferred source-pixel landmarks")
    points = measure.get("value")
    if not isinstance(points, list) or len(points) != 33:
        raise ValueError("Person anchor retains exactly all 33 pose slots")
    indexed = {}
    for point in points:
        if (not isinstance(point, dict) or set(point) != {"index", "position", "visibility", "presence"}
                or type(point["index"]) is not int or not 0 <= point["index"] < 33 or point["index"] in indexed
                or not isinstance(point["position"], list) or len(point["position"]) != 2
                or any(type(v) not in (int, float) or not math.isfinite(v) for v in point["position"])
                or any(v is not None and (type(v) not in (int, float) or not math.isfinite(v) or not 0 <= v <= 1)
                       for v in (point["visibility"], point["presence"]))):
            raise ValueError("Person anchor cannot repair malformed or duplicated pose slots")
        indexed[point["index"]] = point
    slots, usable = [], set()
    for index in range(33):
        point = indexed[index]; x, y = point["position"]
        reasons = []
        if point["visibility"] is None or point["presence"] is None:
            reasons.append("confidence_unavailable")
        elif min(point["visibility"], point["presence"]) < POLICY["minimum_confidence"]:
            reasons.append("below_original_confidence_threshold")
        if not 0 <= x < width or not 0 <= y < height:
            reasons.append("outside_original_image")
        if not reasons:
            usable.add(index)
        slots.append({"index": index, "raw": deepcopy(point), "used_for_roi": not reasons, "exclusion_reasons": reasons})
    supported = {kind: [side for side, indices in POLICY[kind+"_chains"].items() if set(indices) <= usable]
                 for kind in ("arm", "leg")}
    missing_core = [i for i in POLICY["core_indices"] if i not in usable]
    reasons = []
    if missing_core:
        reasons.append("head_or_torso_not_source_supported")
    if not supported["arm"]:
        reasons.append("no_complete_confidence_qualified_arm_chain")
    if not supported["leg"]:
        reasons.append("no_complete_confidence_qualified_leg_chain")
    span = None
    if {11, 12, 23, 24} <= usable:
        span = math.hypot(*[(indexed[11]["position"][a]+indexed[12]["position"][a]
                            -indexed[23]["position"][a]-indexed[24]["position"][a])/2 for a in (0, 1)])
        if span < POLICY["minimum_torso_span_px"]:
            reasons.append("degenerate_source_torso")
    roi = padding = None
    if not reasons:
        xs, ys = ([indexed[i]["position"][axis] for i in sorted(usable)] for axis in (0, 1))
        padding = max(POLICY["minimum_padding_px"], math.ceil(POLICY["padding_fraction"]*max(max(xs)-min(xs), max(ys)-min(ys))))
        left, top = max(0, math.floor(min(xs))-padding), max(0, math.floor(min(ys))-padding)
        right, bottom = min(width, math.ceil(max(xs))+padding), min(height, math.ceil(max(ys))+padding)
        if min(right-left, bottom-top) < POLICY["minimum_roi_side_px"]:
            reasons.append("degenerate_source_roi")
        else:
            roi = [left, top, right-left, bottom-top]
    return {"profile": PROFILE, "status": "proposed" if roi is not None else "abstained", "policy": deepcopy(POLICY),
            "observation_id": observation["id"], "evidence_ids": deepcopy(observation["evidence_ids"]), "frame": 0,
            "landmark_payload_sha256": canonical_hash(points), "landmark_slots": slots,
            "landmark_indices": sorted(usable), "excluded_indices": sorted(set(range(33))-usable),
            "missing_core_indices": missing_core, "supported_chains": supported, "torso_span_px": span,
            "roi_xywh": roi, "padding_pixels": padding, "reasons": reasons,
            "coverage_status": "unknown_full_person_extent", "physical_identity_verified": False,
            "scope": "Tracking ROI hypothesis from confidence-qualified pose estimates; no complete silhouette, stable identity or physical visibility verified. Excluded points are not imputed."}
