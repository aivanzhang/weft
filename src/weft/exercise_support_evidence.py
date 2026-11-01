"""Source-only stationary-foot and hand/equipment candidates, never contact truth."""

from collections import Counter, deque
from copy import deepcopy
from fractions import Fraction
from hashlib import sha256
import json
from json.encoder import encode_basestring_ascii
import math
from statistics import median
import time

PROFILE = "exercise-support-evidence-v1"
FOOT_POINTS = {29: "left_heel", 30: "right_heel", 31: "left_toe", 32: "right_toe"}
HANDS = {15: "left_wrist", 16: "right_wrist"}
CONTRACT = {
    "generation_fraction": .6, "maximum_pose_rows": 900, "maximum_decoded_frames": 1800,
    "maximum_seconds": 10., "maximum_input_bytes": 32 << 20, "maximum_equipment_bytes": 16 << 20,
    "maximum_json_nodes": 2_000_000, "maximum_json_depth": 64,
    "maximum_output_bytes": 8 << 20, "maximum_equipment": 12,
    "minimum_confidence": .5, "minimum_scale_frames": 3, "minimum_torso_span_px": 8.,
    "window_seconds": .4, "minimum_window_span_seconds": .2, "minimum_window_samples": 3,
    "maximum_adjacent_gap_seconds": .12, "maximum_range_torso_fraction": .04,
    "maximum_hand_box_gap_torso_fraction": .03,
    "clock": "Declared frame-index/fps pose clock; native decoder timestamps retained separately",
    "selection": "First floor(0.6*N) supplied pose rows only; no later pixels, geometry, task or floor scores",
    "scope": "Stationary image points and projected co-motion under an explicit fixed-camera prior; not ground, contact, load or grasp",
}


def _require(value, message):
    if not value:
        raise ValueError("Exercise support evidence: " + message)


def _number(value):
    return type(value) is int or type(value) is float and math.isfinite(value)


def _check(deadline):
    if time.monotonic() >= deadline:
        raise TimeoutError("Exercise support evidence exceeded its ten-second bound")


def _pose_row(row):
    selected = row.get("identity_raw", row)
    _require(isinstance(selected, dict) and all(selected.get(key) == row.get(key)
             for key in ("id", "frame", "time_s", "decoder_timestamp_s")), "raw pose identity/clock differs")
    return selected, "identity_raw" if selected is not row else "selected_crop" if "image_refinement" in row else "original"


def _point(row, index, width, height):
    if row.get("status") not in {"tracked", "tracked_partial"} or row.get("pose_count", 1) != 1:
        return {"pixel": None, "confidence": None, "reason": "single_person_estimate_unavailable"}
    points = row.get("landmarks")
    if not isinstance(points, list) or len(points) != 33 or not isinstance(points[index], dict):
        return {"pixel": None, "confidence": None, "reason": "image_landmark_unavailable"}
    point = points[index]
    x, y, visibility, presence = (point.get(key) for key in ("x", "y", "visibility", "presence"))
    if not all(_number(value) for value in (x, y, visibility, presence)) or not 0 <= min(visibility, presence) <= max(visibility, presence) <= 1:
        return {"pixel": None, "confidence": None, "reason": "image_landmark_values_invalid"}
    confidence = min(visibility, presence)
    if not 0 <= x <= 1 or not 0 <= y <= 1:
        return {"pixel": None, "confidence": confidence, "reason": "outside_source_image"}
    if confidence < CONTRACT["minimum_confidence"]:
        return {"pixel": None, "confidence": confidence, "reason": "low_image_confidence"}
    return {"pixel": [x * width, y * height], "confidence": confidence, "reason": None}


def _json_size(value, deadline):
    """Bound the canonical allocation without copying whole strings or child lists."""
    total, nodes, active, strings = 0, 0, set(), {}

    def charge(size):
        nonlocal total
        total += size
        _require(total <= CONTRACT["maximum_input_bytes"], "pose input exceeds byte bound")

    def string(text):
        _require(type(text) is str, "JSON object keys must be strings")
        _require(len(text) + 2 <= CONTRACT["maximum_input_bytes"], "pose input exceeds byte bound")
        if len(text) <= 256 and text in strings:
            charge(strings[text])
            return
        before = total
        charge(2)
        for start in range(0, len(text), 4096):
            _check(deadline)
            charge(len(encode_basestring_ascii(text[start:start + 4096])) - 2)
        if len(text) <= 256 and len(strings) < 1024:
            strings[text] = total - before

    stack = [(iter((value,)), None)]
    while stack:
        _check(deadline)
        try:
            item = next(stack[-1][0])
        except StopIteration:
            _, owner = stack.pop()
            if owner is not None:
                active.remove(owner)
            continue
        nodes += 1
        _require(nodes <= CONTRACT["maximum_json_nodes"] and len(stack) - 1 <= CONTRACT["maximum_json_depth"],
                 "pose JSON exceeds node/depth bound")
        kind = type(item)
        if kind in (dict, list):
            _require(id(item) not in active, "pose JSON contains a cycle")
            _require(nodes + len(item) <= CONTRACT["maximum_json_nodes"], "pose JSON exceeds node bound")
            charge(2 + 2 * max(0, len(item) - 1))
            if kind is dict:
                nodes += len(item)
                for key in item:
                    _check(deadline)
                    string(key)
                    charge(2)
            active.add(id(item))
            stack.append((iter(item.values() if kind is dict else item), id(item)))
        elif kind is str:
            string(item)
        elif item is None:
            charge(4)
        elif kind is bool:
            charge(4 if item else 5)
        elif kind in (int, float):
            _require(kind is int or math.isfinite(item), "pose input requires finite JSON numbers")
            try:
                size = len(repr(item))
            except ValueError as error:
                raise ValueError("Exercise support evidence requires bounded JSON numbers") from error
            charge(size)
        else:
            raise ValueError("Exercise support evidence requires JSON objects, arrays and scalars")
    return total


def _prepare(evidence, camera_prior, deadline):
    _require(type(evidence) is dict, "pose evidence object required")
    _check(deadline)
    source = evidence.get("source_sha256")
    _require(isinstance(source, str) and len(source) == 64 and set(source) <= set("0123456789abcdef"), "exact source SHA256 required")
    _require(type(camera_prior) is dict and set(camera_prior) == {"mode", "status", "source_sha256", "rationale"}
             and camera_prior["mode"] == "fixed_source_camera" and camera_prior["status"] == "authored_unverified"
             and camera_prior["source_sha256"] == source and isinstance(camera_prior["rationale"], str)
             and 1 <= len(camera_prior["rationale"]) <= 1000, "explicit source-bound authored fixed-camera prior required")
    width, height, fps = (evidence.get(key) for key in ("width", "height", "fps"))
    _require(all(type(v) is int and 16 <= v <= 8192 for v in (width, height))
             and _number(fps) and 1 <= fps <= 120, "bounded image dimensions/fps required")
    count, decoded = evidence.get("frame_count"), evidence.get("decoded_frames")
    _require(type(count) is int and type(decoded) is int and 1 <= decoded <= CONTRACT["maximum_decoded_frames"]
             and 1 <= count <= CONTRACT["maximum_decoded_frames"] + 8, "explicit decoded/container frame coverage required")
    frames = evidence.get("frames")
    _require(isinstance(frames, list) and 1 <= len(frames) <= CONTRACT["maximum_pose_rows"], "bounded nonempty pose rows required")
    previous, previous_time = -1, -1.
    for row in frames:
        _check(deadline)
        _require(isinstance(row, dict) and type(row.get("frame")) is int
                 and previous < row["frame"] < decoded and row.get("id") == f"{source}:frame:{row['frame']}", "pose identity/order differs")
        stamp, native = row.get("time_s"), row.get("decoder_timestamp_s")
        _require(_number(stamp) and previous_time < stamp <= 60 and math.isclose(stamp, row["frame"] / fps, abs_tol=1e-9)
                 and (native is None or _number(native) and 0 <= native <= 61), "explicit nominal/native pose clock differs")
        previous, previous_time = row["frame"], stamp
    expected_bytes = _json_size(evidence, deadline)
    _check(deadline)
    encoded = json.dumps(evidence, sort_keys=True, allow_nan=False).encode()
    _require(len(encoded) == expected_bytes, "canonical JSON size differs from bounded preflight")
    _check(deadline)
    return frames, width, height, sha256(encoded).hexdigest()


def _windows(times):
    half, left, right, windows = CONTRACT["window_seconds"] / 2, 0, 0, []
    gaps = [0]
    for index, stamp in enumerate(times):
        gaps.append(gaps[-1] + int(index > 0 and stamp - times[index - 1] > CONTRACT["maximum_adjacent_gap_seconds"] + 1e-12))
    for stamp in times:
        while left < len(times) and times[left] < stamp - half - 1e-12:
            left += 1
        while right < len(times) and times[right] <= stamp + half + 1e-12:
            right += 1
        reasons = []
        span = times[right - 1] - times[left]
        if right - left < CONTRACT["minimum_window_samples"] or span < CONTRACT["minimum_window_span_seconds"] - 1e-12:
            reasons.append("insufficient_temporal_context")
        if gaps[right] - gaps[left + 1]:
            reasons.append("temporal_gap")
        windows.append((left, right, span, reasons))
    return windows


def _stationarity(points, windows, scale, deadline):
    missing = [0]
    for point in points:
        missing.append(missing[-1] + int(point is None))
    extrema, added, result = [deque() for _ in range(4)], 0, []
    for index, (left, right, _, context_reasons) in enumerate(windows):
        _check(deadline)
        while added < right:
            if points[added] is not None:
                for axis in range(2):
                    for maximum in (False, True):
                        queue = extrema[axis * 2 + maximum]
                        value = points[added][axis]
                        while queue and ((queue[-1][1] <= value) if maximum else (queue[-1][1] >= value)):
                            queue.pop()
                        queue.append((added, value))
            added += 1
        for queue in extrema:
            while queue and queue[0][0] < left:
                queue.popleft()
        reasons = list(context_reasons)
        unavailable = missing[right] - missing[left]
        if unavailable:
            reasons.append("missing_or_uncertain_window_point")
        if scale is None:
            reasons.append("image_scale_unavailable")
        bounds, extent = None, None
        if not reasons:
            bounds = [extrema[0][0][1], extrema[2][0][1], extrema[1][0][1], extrema[3][0][1]]
            extent = math.hypot(bounds[2] - bounds[0], bounds[3] - bounds[1]) / scale
        result.append({"status": "unavailable" if reasons else "stationary_image_candidate" if extent <= CONTRACT["maximum_range_torso_fraction"] else "moving_image_point",
                       "reasons": reasons, "window_missing_points": unavailable,
                       "window_image_bounds_px": bounds, "range_torso_fraction": extent})
    return result


def _equipment_mapping(document, evidence, equivalence, deadline):
    if document["source"]["sha256"] == evidence["source_sha256"]:
        _require(equivalence is None, "same-source equipment must not attach an unused equivalence")
        return None, None
    from weft.source_equivalence import SourceEquivalence

    _require(type(equivalence) is SourceEquivalence, "different equipment source requires a verified SourceEquivalence")
    _require(equivalence.original_sha256 == evidence["source_sha256"]
             and equivalence.derivative_sha256 == document["source"]["sha256"], "equivalence source identities differ")
    _require(equivalence.dimensions == (evidence["width"], evidence["height"])
             and len(equivalence.frames) == evidence["decoded_frames"] == document["source"]["decoded_frames"],
             "equivalence dimensions or complete coverage differ")
    _require(document["schema_version"] == 3, "mapped equipment requires explicit decoder clocks, not nominal retiming")
    scale = Fraction(equivalence.time_base)
    clocks, mapping = document["source"]["frame_times_s"], {}
    for pair in equivalence.frames:
        _check(deadline)
        original_time = float(pair.source_decoder_pts * scale)
        derivative_time = float(pair.derivative_decoder_pts * scale)
        _require(math.isclose(clocks[pair.derivative_frame], derivative_time, rel_tol=0, abs_tol=1e-9),
                 "equipment decoder clock conflicts with verified equivalence")
        mapping[pair.source_frame] = {"original_frame": pair.source_frame, "derivative_frame": pair.derivative_frame,
            "original_decoder_time_s": original_time, "derivative_decoder_time_s": derivative_time,
            "original_native_pts": pair.source_native_pts, "derivative_native_pts": pair.derivative_native_pts,
            "original_duration_pts": pair.source_duration_pts, "derivative_duration_pts": pair.derivative_duration_pts,
            "pixel_sha256": pair.pixel_sha256}
    return mapping, equivalence.inspection()


def _equipment(raw, ids, evidence, deadline, equivalence):
    from weft.observations import validate_observations

    _require(type(raw) is bytes and 0 < len(raw) <= CONTRACT["maximum_equipment_bytes"], "bounded exact equipment observation bytes required")
    def unique(pairs):
        value = {}
        for key, item in pairs:
            _require(key not in value, "duplicate equipment JSON key")
            value[key] = item
        return value
    document = validate_observations(json.loads(raw, object_pairs_hook=unique))
    _check(deadline)
    _require(document["image"] == {"width_px": evidence["width"], "height_px": evidence["height"]}, "equipment source image differs")
    mapping, proof = _equipment_mapping(document, evidence, equivalence, deadline)
    _require(isinstance(ids, (list, tuple)) and 1 <= len(ids) <= CONTRACT["maximum_equipment"]
             and all(isinstance(i, str) for i in ids) and len(set(ids)) == len(ids)
             and set(ids) <= {obj["id"] for obj in document["objects"]}, "select bounded unique equipment IDs")
    if document["schema_version"] != 3:
        _require(document["source"]["fps"] == evidence["fps"], "equipment nominal clock differs")
    index = {(row["frame"], row["object_id"]): row for row in document["observations"] if row["object_id"] in ids}
    return document, index, sha256(raw).hexdigest(), mapping, proof


def _equipment_point(row, pose, native_clock, mapping=None):
    if row is None:
        return None, "equipment_slot_unavailable"
    if row["partition"] not in {"construction", "generation"}:
        return None, "equipment_partition_excluded"
    if mapping is not None:
        stamp = pose.get("decoder_timestamp_s")
        if stamp is None or not math.isclose(stamp, mapping["original_decoder_time_s"], rel_tol=0, abs_tol=1e-9):
            return None, "pose_decoder_clock_unavailable_or_different_from_equivalence"
        expected = mapping["derivative_decoder_time_s"]
    else:
        expected = pose.get("decoder_timestamp_s") if native_clock else pose["time_s"]
    if expected is None or not math.isclose(expected, row["time_s"], abs_tol=1e-6):
        return None, "equipment_clock_unavailable_or_different"
    if row["visibility"] != "detected":
        return None, "equipment_occluded_or_unavailable"
    box = row["measurements"].get("box")
    if box is None:
        return None, "equipment_box_unavailable"
    x, y, width, height = box["value"]
    return [x, y, x + width, y + height], None


def estimate_exercise_support(evidence, *, camera_prior, equipment_observations=None, equipment_ids=(),
                              equipment_source_equivalence=None, equipment_silhouette=None, equipment_hand_points=None):
    """Produce construction-only image candidates; original evidence and physics stay untouched."""
    started = time.monotonic()
    deadline = started + CONTRACT["maximum_seconds"]
    _require(equipment_silhouette is None or equipment_hand_points is None, "select exactly one optional equipment image constraint")
    if equipment_silhouette is not None or equipment_hand_points is not None:
        _require(equipment_observations is not None, "image constraints require their exact equipment observation document")
    frames, width, height, pose_hash = _prepare(evidence, camera_prior, deadline)
    cutoff = math.floor(CONTRACT["generation_fraction"] * len(frames))
    selected, inputs, torso_spans = frames[:cutoff], [], []
    for frame in selected:
        _check(deadline)
        row, origin = _pose_row(frame)
        points = {index: _point(row, index, width, height) for index in (*FOOT_POINTS, *HANDS, 11, 12, 23, 24)}
        torso = [points[index]["pixel"] for index in (11, 12, 23, 24)]
        if all(point is not None for point in torso):
            span = math.hypot((torso[0][0] + torso[1][0] - torso[2][0] - torso[3][0]) / 2,
                              (torso[0][1] + torso[1][1] - torso[2][1] - torso[3][1]) / 2)
            if span >= CONTRACT["minimum_torso_span_px"]:
                torso_spans.append(span)
        inputs.append((points, origin))
    scale = median(torso_spans) if len(torso_spans) >= CONTRACT["minimum_scale_frames"] else None
    windows = _windows([row["time_s"] for row in selected])
    feet = {index: _stationarity([row[0][index]["pixel"] for row in inputs], windows, scale, deadline) for index in FOOT_POINTS}
    pairs, equipment_hash, equipment_source, mapping, proof = {}, None, None, None, None
    if equipment_observations is not None:
        equipment, indexed, equipment_hash, mapping, proof = _equipment(
            equipment_observations, equipment_ids, evidence, deadline, equipment_source_equivalence)
        equipment_source = equipment["source"]["sha256"]
        source_frames = [mapping[row["frame"]]["derivative_frame"] if mapping is not None else row["frame"] for row in selected]
        for object_id in equipment_ids:
            boxes = [_equipment_point(indexed.get((frame, object_id)), row, equipment["schema_version"] == 3,
                                      mapping[row["frame"]] if mapping is not None else None)
                     for frame, row in zip(source_frames, selected, strict=True)]
            for hand in HANDS:
                offsets, gaps = [], []
                for (points, _), (box, _) in zip(inputs, boxes, strict=True):
                    point = points[hand]["pixel"]
                    valid = point is not None and box is not None
                    offsets.append([point[0] - (box[0] + box[2]) / 2, point[1] - (box[1] + box[3]) / 2] if valid else None)
                    gaps.append(math.hypot(max(box[0] - point[0], 0., point[0] - box[2]), max(box[1] - point[1], 0., point[1] - box[3])) if valid else None)
                values = _stationarity(offsets, windows, scale, deadline)
                for i, value in enumerate(values):
                    left, right, _, _ = windows[i]
                    source_row = indexed.get((source_frames[i], object_id))
                    value.update(object_id=object_id, hand=HANDS[hand], object_observation_id=source_row["id"] if source_row else None,
                                 object_evidence_ids=list(source_row["evidence_ids"]) if source_row else [],
                                 object_source_frame=source_frames[i], object_time_s=source_row["time_s"] if source_row else None,
                                 equipment_slot_reason=boxes[i][1], hand_image_confidence=inputs[i][0][hand]["confidence"],
                                 image_box_px=boxes[i][0], hand_pixel=inputs[i][0][hand]["pixel"], image_gap_px=gaps[i],
                                 verified_contact=False, grasp="unknown")
                    if value["status"] != "unavailable":
                        maximum_gap = max(gaps[left:right]) / scale
                        value["maximum_window_gap_torso_fraction"] = maximum_gap
                        value["status"] = ("projected_co_motion_candidate" if value["status"] == "stationary_image_candidate"
                                           and maximum_gap <= CONTRACT["maximum_hand_box_gap_torso_fraction"]
                                           else "image_separation" if maximum_gap > CONTRACT["maximum_hand_box_gap_torso_fraction"] else "relative_image_motion")
                    else:
                        value["maximum_window_gap_torso_fraction"] = None
                pairs[(object_id, hand)] = values
    else:
        _require(not equipment_ids, "equipment IDs require their exact observation document")
        _require(equipment_source_equivalence is None, "equivalence requires its exact equipment observation document")
    rows, foot_counts, pair_counts = [], Counter(), Counter()
    for i, frame in enumerate(frames):
        _check(deadline)
        item = {key: frame.get(key) for key in ("id", "frame", "time_s", "decoder_timestamp_s")}
        item.update(partition="construction" if i < cutoff else "excluded_later", ground_points=[], equipment_pairs=[])
        if mapping is not None:
            item["equipment_source_mapping"] = mapping[frame["frame"]]
        if i < cutoff:
            left, right, span, reasons = windows[i]
            item.update(landmark_origin=inputs[i][1], window={"first_evidence_id": selected[left]["id"],
                "last_evidence_id": selected[right - 1]["id"], "samples": right - left, "span_seconds": span, "reasons": reasons})
            for index, name in FOOT_POINTS.items():
                value = {"index": index, "name": name, **inputs[i][0][index], **feet[index][i], "verified_contact": False}
                item["ground_points"].append(value); foot_counts[value["status"]] += 1
            for values in pairs.values():
                item["equipment_pairs"].append(values[i]); pair_counts[values[i]["status"]] += 1
        else:
            item["ground_points"] = [{"index": index, "name": name, "status": "excluded_later_partition",
                                      "pixel": None, "confidence": None, "verified_contact": False} for index, name in FOOT_POINTS.items()]
            item["equipment_pairs"] = [{"object_id": object_id, "hand": HANDS[hand], "status": "excluded_later_partition",
                                        "verified_contact": False, "grasp": "unknown"} for object_id, hand in pairs]
        rows.append(item)
    retained = {row["frame"] for row in frames}
    result = {"schema_version": 1, "profile": PROFILE, "status": "candidates_only" if scale is not None else "unavailable_image_scale",
        "source_sha256": evidence["source_sha256"], "pose_evidence_canonical_sha256": pose_hash,
        "equipment_observations_sha256": equipment_hash, "equipment_object_ids": list(equipment_ids),
        "equipment_source_sha256": equipment_source, "equipment_source_equivalence": proof,
        "camera_prior": deepcopy(camera_prior), "contract": deepcopy(CONTRACT), "rows": rows,
        "image_scale": {"median_torso_span_px": scale, "usable_construction_rows": len(torso_spans), "total_construction_rows": cutoff,
                        "scope": "Image-only normalization, not metric scale, camera calibration or anatomy"},
        "counts": {"supplied_pose_rows": len(frames), "construction_rows": cutoff,
                   "excluded_later_rows": len(frames) - cutoff, "ground_point_slots": cutoff * 4,
                   "retained_ground_point_slots": len(frames) * 4,
                   "ground_status_counts": dict(foot_counts), "equipment_pair_slots": cutoff * len(pairs),
                   "retained_equipment_pair_slots": len(frames) * len(pairs),
                   "equipment_status_counts": dict(pair_counts)},
        "coverage": {"declared_frame_count": evidence["frame_count"], "decoded_frames": evidence["decoded_frames"],
                     "unsampled_decoded_frame_indices": [i for i in range(evidence["decoded_frames"]) if i not in retained],
                     "container_only_frame_indices": list(range(evidence["decoded_frames"], evidence["frame_count"])),
                     "decoded_beyond_container_metadata_indices": list(range(evidence["frame_count"], evidence["decoded_frames"]))},
        "contact_state": "unknown", "ground_height": None, "support_force": None, "promoted": False,
        "physical_validation": "unverified", "limits": [CONTRACT["scope"],
            "Moving camera, stationary lifted feet and image overlap can produce incorrect physical explanations.",
            "Equipment boxes are inferred image envelopes, not exact surfaces; occluded or withheld slots cannot establish coupling.",
            "No interpolation, native geometry, body modification, task scoring, inference or video decoding occurs."]}
    _require(len(json.dumps(result, allow_nan=False).encode()) <= CONTRACT["maximum_output_bytes"], "output exceeds byte bound")
    if equipment_silhouette is not None:
        from weft.equipment_silhouette_constraint import constrain_equipment_support
        result = constrain_equipment_support(result, equipment_silhouette, deadline=deadline)
    elif equipment_hand_points is not None:
        from weft.equipment_hand_constraint import constrain_equipment_hand_support
        result = constrain_equipment_hand_support(result, equipment_hand_points, deadline=deadline)
    _check(deadline)
    return result
