"""Restrict original wrist co-motion candidates with four exact source hand points."""

from array import array
from collections import Counter
from hashlib import sha256
import json
import time

from weft.equipment_hand_evidence import EquipmentHandEvidence, HAND_POINTS, PARTS, PROFILE as EVIDENCE_PROFILE
from weft.equipment_silhouette_constraint import (
    BASELINE_PROFILE, MAX_OUTPUT_BYTES, MAX_SECONDS, REPRESENTATIONS,
    _baseline, _bind_inputs, _check, _decode, _encode, _number, _restrict_windows, _text,
)
from weft.exercise_support_evidence import _point as source_point, _pose_row


PROFILE = "equipment-hand-window-constraint-v1"
# Four retained points per hand require their own bound; the full output remains eight MiB.
MAX_CACHE_BYTES = 8 << 20
_KNOWN = {"conditional_image_intersection": "intersection", "conditional_image_exclusion": "exclusion"}
_NULL_POINT_FIELDS = ("point_channel_id", "human_entity_ref", "human_observation_id", "human_partition",
    "equipment_observation_id", "equipment_time_s", "equipment_partition", "mask_evidence_id", "mask_phase",
    "mask_status", "mask_time_s", "model_identity", "pixel", "source_confidence", "source_visibility_score",
    "source_presence_score", "representation", "authored_uncertainty_px", "intersects",
    "distance_infimum_squared_px2", "raster_sha256", "raster_dimensions", "image_box_px")


def _require(value, message):
    if not value:
        raise ValueError("Equipment hand constraint: " + message)


def _groups(metadata):
    groups = metadata.get("hand_groups")
    _require(type(groups) is list and 1 <= len(groups) <= 2, "one or two explicitly selected hands required")
    expected, roles = {}, []
    for group in groups:
        _require(type(group) is dict and set(group) == {"hand_id", "wrist_name", "points"}, "exact named hand group required")
        hand = group["hand_id"]
        _require(hand in HAND_POINTS and hand not in expected, "unique original hand identities required")
        side = hand.removesuffix("_hand")
        _require(group["wrist_name"] == side + "_wrist" and type(group["points"]) is list
                 and len(group["points"]) == 4, "four ordered original points required")
        for part, point in zip(PARTS, group["points"], strict=True):
            _require(type(point) is dict and set(point) == {"part", "point_name", "landmark_index", "channel_id"}
                     and point["part"] == part and point["point_name"] == side + "_" + part
                     and type(point["landmark_index"]) is int and point["landmark_index"] == HAND_POINTS[hand][part]
                     and _text(point["channel_id"], 80), "original named point/channel identity differs")
            roles.append(point["point_name"])
        expected[hand] = group
    _require(list(expected) == [hand for hand in HAND_POINTS if hand in expected], "original left/right order required")
    return expected, roles


def _point_status(point, descriptor, channel, row, image, original, source_pose, width, height, origin):
    if (image.get("source_frame") != row["frame"] or type(image.get("source_frame")) is not int
            or image.get("pose_evidence_id") != row["id"] or image.get("pose_time_s") != row["time_s"]
            or image.get("pose_decoder_timestamp_s") != row.get("decoder_timestamp_s")
            or image.get("source_status") != "decoded" or image.get("source_mapping") != row.get("equipment_source_mapping")
            or image.get("equipment_source_frame") != original.get("object_source_frame")):
        return "unavailable", "source_frame_or_clock_differs"
    if (point.get("point_channel_id") != channel["id"] or point.get("point_name") != descriptor["point_name"]
            or type(point.get("landmark_index")) is not int or point["landmark_index"] != descriptor["landmark_index"]
            or point.get("human_entity_ref") != channel["entity_ref"]
            or point.get("human_observation_id") != f"{channel['source_sha256']}:observation:{row['frame']}:{channel['entity_ref']['object_id']}"
            or point.get("equipment_id") != original["object_id"]
            or point.get("equipment_observation_id") != original.get("object_observation_id")
            or point.get("equipment_time_s") != original.get("object_time_s") or point.get("contact") != "unknown"):
        return "unavailable", "qualified_point_identity_or_clock_differs"
    if point.get("human_partition") != "construction" or point.get("equipment_partition") != "construction":
        return "unavailable", "image_partition_unavailable"
    status = point.get("status")
    if status not in _KNOWN:
        return "unavailable", point["reason"] if _text(point.get("reason"), 160) else "hand_point_unavailable"
    if origin != "original" or row.get("landmark_origin") != "original" or point.get("representation") not in REPRESENTATIONS:
        return "unavailable", "source_point_origin_not_qualified"
    if (not _text(point.get("mask_evidence_id")) or point["mask_evidence_id"] not in original.get("object_evidence_ids", [])
            or not _number(point.get("mask_time_s")) or point.get("mask_time_s") != original.get("object_time_s") or point.get("reason") is not None
            or type(point.get("intersects")) is not bool or point["intersects"] != (status == "conditional_image_intersection")
            or not _number(point.get("authored_uncertainty_px")) or point["authored_uncertainty_px"] < 0):
        return "unavailable", "point_relation_or_evidence_differs"
    # Unavailable cached evidence must not reopen its raw landmark payload.
    source = source_point(source_pose, descriptor["landmark_index"], width, height)
    landmarks = source_pose.get("landmarks")
    source_landmark = landmarks[descriptor["landmark_index"]] if isinstance(landmarks, list) and len(landmarks) == 33 else {}
    source_landmark = source_landmark if type(source_landmark) is dict else {}
    pixel = point.get("pixel")
    if (source["pixel"] is None or type(pixel) is not list or len(pixel) != 2 or not all(_number(v) for v in pixel)
            or pixel != source["pixel"] or not _number(point.get("source_confidence"))
            or point.get("source_confidence") != source["confidence"]
            or point.get("source_visibility_score") != source_landmark.get("visibility")
            or point.get("source_presence_score") != source_landmark.get("presence")
            or point.get("image_box_px") != original.get("image_box_px")):
        return "unavailable", "source_point_confidence_or_box_differs"
    if descriptor["part"] == "wrist" and (point["pixel"] != original.get("hand_pixel")
                                         or point["source_confidence"] != original.get("hand_image_confidence")):
        return "unavailable", "original_wrist_baseline_differs"
    return _KNOWN[status], None


def _missing(object_id, hand):
    points = []
    for part in PARTS:
        points.append({**dict.fromkeys(_NULL_POINT_FIELDS), "equipment_id": object_id,
            "point_name": hand.removesuffix("_hand") + "_" + part, "landmark_index": HAND_POINTS[hand][part],
            "status": "unavailable", "reason": "hand_group_not_supplied", "contact": "unknown",
            "producer_record_present": False, "consumer_status": "unavailable", "consumer_reason": "hand_group_not_supplied"})
    return points


def constrain_equipment_hand_support(baseline, prepared, *, deadline):
    """Use saved point relations only; no annotation query, raster, geometry or source I/O."""
    deadline = min(deadline, time.monotonic() + MAX_SECONDS)
    baseline_raw, objects, rows, cutoff, selected = _baseline(baseline, deadline)
    _require(type(prepared) is EquipmentHandEvidence, "explicit prepared hand evidence required")
    metadata = _decode(prepared._metadata, 256 << 10, deadline)
    groups, roles = _groups(metadata)
    channels, bindings, source_count, source_bytes, pose = _bind_inputs(baseline, prepared, selected, deadline,
        evidence_type=EquipmentHandEvidence, evidence_profile=EVIDENCE_PROFILE, point_roles=roles,
        slot_count_key="point_slots", binding_prefix="hand", allow_extra_channels=True)
    _require(metadata.get("equipment_ids") == objects and metadata.get("cached_row_bytes") == source_bytes
             and type(metadata["counts"].get("hand_pair_slots")) is int
             and metadata["counts"]["hand_pair_slots"] == source_count * len(objects) * len(groups),
             "complete exact hand/object/cache denominator differs")
    supplied_channels = metadata.get("point_channels")
    _require(type(supplied_channels) is list and len(supplied_channels) == len(roles)
             and [item.get("point_role") for item in supplied_channels] == roles
             and all(item.get("id") == channels[role]["id"] and item.get("entity_ref") == channels[role]["entity_ref"]
                     for item, role in zip(supplied_channels, roles, strict=True)), "exact prepared point channel declarations differ")
    _require(all(value["entity_ref"] == channels[roles[0]]["entity_ref"] for value in channels.values()),
             "one explicitly bound original human required")
    for group in groups.values():
        _require(all(point["channel_id"] == channels[point["point_name"]]["id"] for point in group["points"]),
                 "hand group and point channel declarations differ")
    bindings["hand_groups"] = list(groups.values())
    keys = [(object_id, hand) for object_id in objects for hand in HAND_POINTS]
    prefixes = {(object_id, hand.removesuffix("_hand") + "_wrist"):
                {name: array('I', [0]) for name in ("intersection", "exclusion", "unavailable")} for object_id, hand in keys}
    pose_rows = {row["frame"]: row for row in pose["frames"]}
    encoded_rows, cache_bytes, hand_counts, point_counts = [], 0, Counter(), Counter()
    for row in selected:
        _check(deadline)
        image = _decode(prepared._rows[row["frame"]], 128 << 10, deadline)
        supplied = image.get("hands")
        _require(type(supplied) is list and len(supplied) == len(objects) * len(groups), "complete selected hand slots required")
        _require(all(type(item) is dict for item in supplied)
                 and [(item.get("equipment_id"), item.get("hand_id")) for item in supplied]
                 == [(object_id, hand) for object_id in objects for hand in groups], "ordered equipment/hand identities differ")
        by_hand = {(item["equipment_id"], item["hand_id"]): item for item in supplied}
        original = {(pair["object_id"], pair["hand"]): pair for pair in row["equipment_pairs"]}
        _require(len(original) == len(row["equipment_pairs"]) == len(keys)
                 and set(original) == set(prefixes), "complete original equipment/wrist identities required")
        supplied_pose = pose_rows[row["frame"]]
        _require(all(supplied_pose.get(key) == row.get(key) for key in ("id", "frame", "time_s", "decoder_timestamp_s")),
                 "original baseline pose identity or clock differs")
        source_pose, origin = _pose_row(supplied_pose)
        hand_row = {key: image.get(key) for key in ("source_frame", "source_status", "pose_evidence_id", "pose_time_s",
                    "pose_decoder_timestamp_s", "equipment_source_frame", "source_mapping")}
        hand_row["hands"] = []
        for object_id, hand in keys:
            wrist = hand.removesuffix("_hand") + "_wrist"
            supplied_hand = by_hand.get((object_id, hand))
            points = []
            if supplied_hand is None:
                points = _missing(object_id, hand)
            else:
                group = groups[hand]
                _require(supplied_hand.get("wrist_name") == wrist
                         and supplied_hand.get("human_entity_ref") == channels[group["points"][0]["point_name"]]["entity_ref"]
                         and all(supplied_hand.get(key) == "unknown" for key in ("contact", "physical_noncontact", "grasp", "load")),
                         "hand identity or physical claims differ")
                raw_points = supplied_hand.get("points")
                _require(type(raw_points) is list and len(raw_points) == 4 and all(type(p) is dict for p in raw_points),
                         "all four exact point records required")
                raw_counts = Counter(_KNOWN.get(p.get("status"), "unavailable") for p in raw_points)
                expected_counts = {key: raw_counts[key] for key in ("intersection", "exclusion", "unavailable")}
                expected_missing = [{"point_name": p.get("point_name"), "reason": p.get("reason")}
                                    for p in raw_points if p.get("status") not in _KNOWN]
                expected_status = ("unavailable" if expected_missing else "conditional_pointset_image_intersection"
                                   if raw_counts["intersection"] else "conditional_pointset_image_exclusion")
                _require(type(supplied_hand.get("point_counts")) is dict
                         and all(type(value) is int for value in supplied_hand["point_counts"].values())
                         and supplied_hand["point_counts"] == expected_counts and supplied_hand.get("unavailable_points") == expected_missing
                         and supplied_hand.get("status") == expected_status
                         and supplied_hand.get("reason") == ("required_hand_point_unavailable" if expected_missing else None)
                         and supplied_hand.get("intersects") is (None if expected_missing else bool(raw_counts["intersection"])),
                         "producer group must preserve the complete four-point decision")
                for raw_point, descriptor in zip(raw_points, group["points"], strict=True):
                    status, reason = _point_status(raw_point, descriptor, channels[descriptor["point_name"]], row,
                        image, original[(object_id, wrist)], source_pose, pose["width"], pose["height"], origin)
                    points.append({**raw_point, "producer_record_present": True, "consumer_status": status, "consumer_reason": reason})
            counts = {name: sum(point["consumer_status"] == name for point in points)
                      for name in ("intersection", "exclusion", "unavailable")}
            status = "unavailable" if counts["unavailable"] else "intersection" if counts["intersection"] else "exclusion"
            reason = ("hand_group_not_supplied" if supplied_hand is None else "required_hand_point_unavailable") if counts["unavailable"] else None
            for name, prefix in prefixes[(object_id, wrist)].items(): prefix.append(prefix[-1] + int(status == name))
            hand_counts[status] += 1; point_counts.update(point["consumer_status"] for point in points)
            hand_row["hands"].append({"object_id": object_id, "hand_id": hand, "wrist_name": wrist,
                "status": status, "reason": reason, "point_counts": counts,
                "unavailable_points": [{"point_name": p.get("point_name"), "reason": p["consumer_reason"]}
                                       for p in points if p["consumer_status"] == "unavailable"], "points": points})
        encoded = _encode(hand_row, 128 << 10, deadline); encoded_rows.append(encoded); cache_bytes += len(encoded)
        prefix_bytes = sum(len(prefix) * prefix.itemsize for values in prefixes.values() for prefix in values.values())
        _require(cache_bytes + prefix_bytes <= MAX_CACHE_BYTES, "complete point/prefix cache exceeds bound; no slots dropped")
    result, decisions = _restrict_windows(baseline, prefixes, deadline, decision_field="hand_constraint",
        counts_field="window_hand_counts", excluded_status="image_hand_pointset_exclusion",
        missing_reason="required_window_hand_point_unavailable", exclusion_reason="conditional_hand_pointset_exclusion_in_original_window")
    result["equipment_hand_constraint"] = {"schema_version": 1, "profile": PROFILE, "status": "applied_restrict_only",
        "baseline_profile": BASELINE_PROFILE, "baseline_canonical_sha256": sha256(baseline_raw).hexdigest(),
        "baseline_equipment_status_counts": baseline["counts"]["equipment_status_counts"], "input_bindings": bindings,
        "construction_frames": [row["frame"] for row in selected], "hand_rows": [json.loads(raw) for raw in encoded_rows],
        "counts": {"source_frames": source_count, "construction_hand_slots": cutoff * len(keys),
            "construction_point_slots": cutoff * len(keys) * 4, "retained_pose_pair_slots": len(rows) * len(keys),
            "baseline_candidates": decisions["kept"] + decisions["excluded"] + decisions["unavailable"],
            "kept_candidates": decisions["kept"], "excluded_candidates": decisions["excluded"], "unavailable_candidates": decisions["unavailable"],
            "unchanged_construction_noncandidates": decisions["not_candidate"], "excluded_later_pose_pairs": decisions["excluded_later"],
            "hand_status_counts": dict(hand_counts), "point_status_counts": dict(point_counts)},
        "work": {"selected_frame_queries": cutoff, "hand_slots": cutoff * len(keys), "point_slots": cutoff * len(keys) * 4,
            "window_queries": decisions["kept"] + decisions["excluded"] + decisions["unavailable"],
            "serialized_hand_cache_bytes": cache_bytes,
            "prefix_counter_entries": sum(len(prefix) for values in prefixes.values() for prefix in values.values()),
            "prefix_counter_buffer_bytes": sum(len(prefix) * prefix.itemsize for values in prefixes.values() for prefix in values.values()),
            "prepared_row_bytes": source_bytes, "raster_calls": 0, "pose_annotation_queries": 0},
        "bounds": {"cooperative_seconds": MAX_SECONDS, "within_original_deadline": True, "additional_cache_bytes": MAX_CACHE_BYTES,
            "complete_output_bytes": MAX_OUTPUT_BYTES, "construction_poses": 540, "equipment_ids": 8, "hands": 2, "points_per_hand": 4},
        "contact": "unknown", "physical_noncontact": "unknown", "grasp": "unknown", "load": "unknown",
        "physical_validation": "unverified", "body_or_floor_installed": False,
        "scope": "Restrict original wrist co-motion windows using all four original named hand points; point-set image consistency is not a hand surface, contact, noncontact, grasp, identity or support",
        "baseline_recovery": "Remove this top-level member and each pair's co_motion_baseline_status/hand_constraint, restore its status and original equipment_status_counts; every other field remains exact"}
    _encode(result, MAX_OUTPUT_BYTES, deadline); _check(deadline)
    return result
