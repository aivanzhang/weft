"""Restrict existing image co-motion candidates; never infer physical contact."""

from array import array
from collections import Counter
from hashlib import sha256
import json
import math
import time

from weft.equipment_silhouette_evidence import EquipmentSilhouetteEvidence, PROFILE as SILHOUETTE_PROFILE
from weft.exercise_support_evidence import CONTRACT, HANDS, PROFILE as BASELINE_PROFILE, _json_size
from weft.pose_surface_annotations import PoseSurfaceAnnotations
from weft.source_equivalence import SourceEquivalence


PROFILE = "equipment-silhouette-window-constraint-v1"
MAX_SECONDS, MAX_CACHE_BYTES, MAX_OUTPUT_BYTES = 5., 2 << 20, 8 << 20
REPRESENTATIONS = frozenset({"supplied_unmarked_image_estimate_not_authenticated_raw",
    "declared_raw_model_estimate_not_authenticated_identity", "matched_saved_precursor_model_estimate_identity_unverified"})


def _require(condition, message):
    if not condition:
        raise ValueError("Equipment silhouette constraint: " + message)


def _check(deadline):
    if time.monotonic() >= deadline:
        raise TimeoutError("Equipment silhouette constraint cooperative deadline exceeded")


def _encode(value, maximum, deadline):
    _check(deadline)
    _require(_json_size(value, deadline) <= maximum, "serialized byte bound exceeded")
    raw = json.dumps(value, sort_keys=True, allow_nan=False).encode()
    _require(len(raw) <= maximum, "serialized byte bound exceeded")
    _check(deadline)
    return raw


def _decode(raw, maximum, deadline):
    _require(type(raw) is bytes and 0 < len(raw) <= maximum, "bounded immutable cached bytes required")
    _check(deadline)
    def unique(pairs):
        result = {}
        for key, value in pairs:
            _require(key not in result, "duplicate cached JSON key")
            result[key] = value
        return result
    def invalid(_):
        raise ValueError("Nonfinite cached JSON")
    try:
        value = json.loads(raw, object_pairs_hook=unique, parse_constant=invalid)
    except (UnicodeError, json.JSONDecodeError, RecursionError) as error:
        raise ValueError("Malformed cached silhouette JSON") from error
    _require(type(value) is dict, "cached JSON object required")
    _check(deadline)
    return value


def _text(value, maximum=256):
    return type(value) is str and 0 < len(value) <= maximum


def _number(value):
    return type(value) in (int, float) and math.isfinite(value)


def _pin(raw, maximum):
    _require(type(raw) is bytes and 0 < len(raw) <= maximum, "bounded exact input bytes required")
    return sha256(raw).hexdigest()


def _bindings(baseline, prepared, selected, deadline):
    _require(type(prepared) is EquipmentSilhouetteEvidence and type(prepared.pose_annotations) is PoseSurfaceAnnotations,
             "qualified prepared silhouette and pose annotations required")
    meta = _decode(prepared._metadata, 256 << 10, deadline)
    annotations = prepared.pose_annotations
    pose_meta = _decode(annotations._metadata, 256 << 10, deadline)
    annotation = _decode(annotations.annotation_document(), 4 << 20, deadline)
    _require(meta.get("profile") == SILHOUETTE_PROFILE and meta.get("status") == "conditional_image_evidence_only"
             and meta.get("schema_version") == 1, "silhouette profile/status differs")
    _require(all(meta.get(key) == "unknown" for key in ("contact", "physical_noncontact", "grasp", "load"))
             and meta.get("body_or_floor_installed") is False, "silhouette must not assert physical admission")
    source = baseline["source_sha256"]
    original_frames = [row["frame"] for row in selected]
    _require(meta.get("source_sha256") == source and meta.get("equipment_source_sha256") == baseline["equipment_source_sha256"]
             and meta.get("construction_frames") == original_frames, "source identity or exact construction selection differs")
    provenance = meta.get("pose_provenance")
    _require(type(provenance) is dict and provenance == pose_meta.get("provenance")
             and provenance.get("source_sha256") == source
             and provenance.get("pose_canonical_sha256") == baseline["pose_evidence_canonical_sha256"], "canonical pose identity differs")
    pose_pin = _pin(annotations.pose_document, 8 << 20)
    # Raw custody and the estimator's canonical identity must join independently.
    pose_canonical = sha256(_encode(_decode(annotations.pose_document, 8 << 20, deadline),
                                    CONTRACT["maximum_input_bytes"], deadline)).hexdigest()
    _require(pose_canonical == provenance["pose_canonical_sha256"], "raw bytes derive a different canonical pose identity")
    equipment_pin = _pin(prepared.equipment_document, 16 << 20)
    mask_pin = _pin(prepared.mask_document, 8 << 20)
    annotation_pin = _pin(annotations.annotation_document(), 4 << 20)
    _require(pose_pin == provenance.get("pose_raw_sha256") and equipment_pin == baseline["equipment_observations_sha256"]
             and equipment_pin == meta.get("equipment_document_sha256") and mask_pin == meta.get("mask_document_sha256")
             and annotation_pin == pose_meta.get("annotation_sha256"), "exact prepared input pins differ")
    _require(type(annotations.input_documents) is tuple and 1 <= len(annotations.input_documents) <= 8
             and sum(len(raw) for _, raw in annotations.input_documents) <= 16 << 20, "canonical input envelope differs")
    documents = {pin for pin, raw in annotations.input_documents if _pin(raw, 8 << 20) == pin}
    _require(len(documents) == len(annotations.input_documents), "canonical input raw pins differ")
    _require(annotation.get("source_sha256") == source and annotation.get("profile") == "source-surface-annotations-v1"
             and annotation.get("quality") == "authored", "annotation source/quality differs")
    channels = annotation.get("channels")
    _require(type(channels) is list and len(channels) == 2, "both explicitly named wrist channels required")
    channel_map = {}
    for channel in channels:
        role, ref = channel.get("point_role"), channel.get("entity_ref")
        _require(role in HANDS.values() and role not in channel_map and _text(channel.get("id"), 80)
                 and type(ref) is dict and set(ref) == {"document_sha256", "object_id"}
                 and ref["document_sha256"] in documents and _text(ref["object_id"], 80), "qualified named wrist channel differs")
        channel_map[role] = {"id": channel["id"], "entity_ref": ref, "source_sha256": source}
    _require(len({channel["id"] for channel in channel_map.values()}) == 2, "duplicate wrist channel identity")
    proof = prepared.source_equivalence
    if baseline["equipment_source_sha256"] == source:
        _require(proof is None and meta.get("source_equivalence") is None and baseline["equipment_source_equivalence"] is None,
                 "same-source input cannot attach an unused proof")
        aliases = {}
    else:
        _require(type(proof) is SourceEquivalence and proof.original_sha256 == source
                 and proof.derivative_sha256 == baseline["equipment_source_sha256"], "verified exact source equivalence required")
        _require(meta.get("source_equivalence") == proof.inspection() == baseline["equipment_source_equivalence"],
                 "complete source equivalence clocks/pins differ")
        aliases = {baseline["equipment_source_sha256"]: source}
    disclosure = meta.get("canonical_frame_selection")
    _require(disclosure == {"profile": "restrict-only-original-frame-selection-v1",
        "frames_by_canonical_source_sha256": {source: original_frames}, "partition_consistency_scope": "selected_frames_only",
        "excluded_partition_labels": "preserved_not_reconciled", "source_aliases": aliases, "maximum_frames_per_source": 1800},
        "exact canonical construction admission disclosure required")
    coverage = baseline["coverage"]
    count = max(coverage["decoded_frames"], coverage["declared_frame_count"])
    _require(type(prepared._rows) is tuple and len(prepared._rows) == count <= 1800
             and all(type(raw) is bytes and 0 < len(raw) <= 128 << 10 for raw in prepared._rows), "complete cached source rows required")
    cached_bytes = sum(map(len, prepared._rows))
    _require(cached_bytes + len(prepared._metadata) <= 8 << 20, "prepared silhouette cache bound exceeded")
    counts = meta.get("counts", {})
    objects = baseline["equipment_object_ids"]
    _require(all(type(counts.get(key)) is int for key in ("source_frames", "decoded_frames", "metadata_frames", "pose_slots", "pair_slots"))
             and counts.get("source_frames") == count and counts.get("decoded_frames") == coverage["decoded_frames"]
             and counts.get("metadata_frames") == coverage["declared_frame_count"]
             and counts.get("pose_slots") == len(baseline["rows"]) and counts.get("pair_slots") == count * len(objects) * 2,
             "complete silhouette denominator differs")
    rows_digest = sha256()
    for raw in prepared._rows:
        _check(deadline)
        rows_digest.update(len(raw).to_bytes(8, "big")); rows_digest.update(raw)
    return channel_map, {
        "source_sha256": source, "equipment_source_sha256": baseline["equipment_source_sha256"],
        "pose_evidence_canonical_sha256": baseline["pose_evidence_canonical_sha256"], "pose_raw_sha256": pose_pin,
        "equipment_document_sha256": equipment_pin, "mask_document_sha256": mask_pin,
        "annotation_document_sha256": annotation_pin, "observation_document_sha256s": sorted(documents),
        "silhouette_metadata_sha256": sha256(prepared._metadata).hexdigest(),
        "silhouette_rows_length_prefixed_sha256": rows_digest.hexdigest(),
        "rows_digest_encoding": "Each original cached row prefixed by its unsigned 8-byte big-endian byte length; not a source-file hash",
        "source_equivalence_manifest_sha256": None if proof is None else proof.manifest_sha256,
    }, count, cached_bytes


def _point(row, image, original, pair, channel):
    if (type(image.get("source_frame")) is not int or image.get("source_frame") != row["frame"] or image.get("pose_evidence_id") != row["id"]
            or image.get("pose_time_s") != row["time_s"] or image.get("pose_decoder_timestamp_s") != row.get("decoder_timestamp_s")
            or image.get("source_status") != "decoded"
            or image.get("source_mapping") != row.get("equipment_source_mapping")
            or image.get("equipment_source_frame") != original.get("object_source_frame")):
        return "unavailable", "source_frame_or_clock_differs"
    if pair is None:
        return "unavailable", "silhouette_pair_missing"
    if (pair.get("wrist_channel_id") != channel["id"] or pair.get("human_entity_ref") != channel["entity_ref"]
            or pair.get("human_observation_id") != f"{channel['source_sha256']}:observation:{row['frame']}:{channel['entity_ref']['object_id']}"
            or pair.get("equipment_observation_id") != original.get("object_observation_id")
            or pair.get("equipment_time_s") != original.get("object_time_s") or pair.get("contact") != "unknown"):
        return "unavailable", "qualified_pair_identity_or_clock_differs"
    if pair.get("human_partition") != "construction" or pair.get("equipment_partition") != "construction":
        return "unavailable", "image_partition_unavailable"
    status = pair.get("status")
    if status not in ("conditional_image_intersection", "conditional_image_exclusion"):
        reason = pair.get("reason")
        return "unavailable", reason if _text(reason, 160) else "silhouette_point_unavailable"
    if row.get("landmark_origin") != "original" or pair.get("representation") not in REPRESENTATIONS:
        return "unavailable", "source_point_origin_not_qualified"
    pixel, confidence = pair.get("pixel"), pair.get("source_confidence")
    if (type(pixel) is not list or len(pixel) != 2 or not all(_number(value) for value in pixel)
            or pixel != original.get("hand_pixel") or not _number(confidence) or not 0 <= confidence <= 1
            or confidence != original.get("hand_image_confidence")
            or pair.get("image_box_px") != original.get("image_box_px")):
        return "unavailable", "source_point_confidence_or_box_differs"
    if (not _text(pair.get("mask_evidence_id")) or pair["mask_evidence_id"] not in original.get("object_evidence_ids", [])
            or pair.get("reason") is not None or type(pair.get("intersects")) is not bool
            or pair["intersects"] != (status == "conditional_image_intersection")):
        return "unavailable", "silhouette_relation_or_evidence_differs"
    return ("intersection" if pair["intersects"] else "exclusion"), None


def constrain_equipment_support(baseline, prepared, *, deadline):
    """Return a restricted copy of an existing estimator result; no source I/O."""
    deadline = min(deadline, time.monotonic() + MAX_SECONDS)
    baseline_raw = _encode(baseline, MAX_OUTPUT_BYTES, deadline)
    _require(baseline.get("profile") == BASELINE_PROFILE and baseline.get("schema_version") == 1
             and "equipment_silhouette_constraint" not in baseline, "original unconstrained support result required")
    _require(baseline.get("contact_state") == "unknown" and baseline.get("ground_height") is None
             and baseline.get("support_force") is None and baseline.get("promoted") is False, "baseline physical claims differ")
    objects, rows = baseline.get("equipment_object_ids"), baseline.get("rows")
    _require(type(objects) is list and 1 <= len(objects) <= 8 and len(set(objects)) == len(objects)
             and all(_text(value, 80) for value in objects), "one to eight explicit equipment identities required")
    _require(type(rows) is list and 1 <= len(rows) <= 900, "bounded original pose rows required")
    cutoff = math.floor(CONTRACT["generation_fraction"] * len(rows))
    _require(baseline["counts"]["construction_rows"] == cutoff and cutoff <= 540
             and [row.get("partition") for row in rows] == ["construction"] * cutoff + ["excluded_later"] * (len(rows) - cutoff),
             "exact original first-60-percent partition required")
    selected = rows[:cutoff]
    _require(all(type(row.get("frame")) is int for row in rows)
             and [row["frame"] for row in rows] == sorted({row["frame"] for row in rows}), "original frame order differs")
    channels, bindings, source_count, source_cache_bytes = _bindings(baseline, prepared, selected, deadline)
    keys = [(object_id, hand) for object_id in objects for hand in HANDS.values()]
    prefixes = {key: {name: array('I', [0]) for name in ("intersection", "exclusion", "unavailable")} for key in keys}
    point_rows, point_bytes, point_counts = [], 0, Counter()
    for row in selected:
        _check(deadline)
        image = _decode(prepared._rows[row["frame"]], 128 << 10, deadline)
        supplied = image.get("pairs")
        _require(type(supplied) is list and len(supplied) == len(keys), "complete equipment/wrist image slots required")
        image_pairs = {(pair.get("equipment_id"), pair.get("wrist_name")): pair for pair in supplied}
        original_pairs = {(pair.get("object_id"), pair.get("hand")): pair for pair in row["equipment_pairs"]}
        _require(set(image_pairs) == set(original_pairs) == set(keys) and len(original_pairs) == len(row["equipment_pairs"]),
                 "exact equipment/wrist identities or multiplicity differ")
        point_row = {"source_frame": row["frame"], "pose_evidence_id": row["id"], "pairs": []}
        for key in keys:
            pair = image_pairs[key]
            status, reason = _point(row, image, original_pairs[key], pair, channels[key[1]])
            for name, prefix in prefixes[key].items():
                prefix.append(prefix[-1] + int(name == status))
            point_counts[status] += 1
            point_row["pairs"].append({"object_id": key[0], "hand": key[1], "status": status, "reason": reason,
                "mask_evidence_id": pair.get("mask_evidence_id") if _text(pair.get("mask_evidence_id")) else None})
        encoded = _encode(point_row, 64 << 10, deadline)
        point_rows.append(encoded); point_bytes += len(encoded)
        prefix_bytes = sum(len(prefix) * prefix.itemsize for values in prefixes.values() for prefix in values.values())
        _require(point_bytes + prefix_bytes <= MAX_CACHE_BYTES, "point/prefix cache bound exceeded; no slots dropped")
    index = {row["id"]: i for i, row in enumerate(selected)}
    result, output_rows, decisions, active_counts = dict(baseline), [], Counter(), Counter()
    for i, row in enumerate(rows):
        _check(deadline)
        updated = dict(row); updated["equipment_pairs"] = []
        for original in row["equipment_pairs"]:
            pair = dict(original); old = original["status"]
            decision = {"status": "excluded_later" if i >= cutoff else "not_candidate", "reason": old,
                        "window_point_range": None, "window_image_counts": None}
            if i < cutoff and old == "projected_co_motion_candidate":
                window = row["window"]
                left, last = index.get(window["first_evidence_id"]), index.get(window["last_evidence_id"])
                _require(type(left) is int and type(last) is int and 0 <= left <= i <= last < cutoff
                         and window["samples"] == last - left + 1, "original candidate window identities differ")
                right = last + 1
                counts = {name: prefix[right] - prefix[left] for name, prefix in prefixes[(original["object_id"], original["hand"])].items()}
                _require(sum(counts.values()) == window["samples"], "complete original window point count differs")
                status = "unavailable" if counts["unavailable"] else "excluded" if counts["exclusion"] else "kept"
                decision.update(status=status, reason={"unavailable": "required_window_image_point_unavailable",
                    "excluded": "conditional_image_exclusion_in_original_window", "kept": None}[status],
                    window_point_range=[left, right], window_image_counts=counts)
                pair["status"] = {"unavailable": "unavailable", "excluded": "image_silhouette_exclusion", "kept": old}[status]
            pair["co_motion_baseline_status"] = old; pair["silhouette_constraint"] = decision
            updated["equipment_pairs"].append(pair); decisions[decision["status"]] += 1
            if i < cutoff:
                active_counts[pair["status"]] += 1
        output_rows.append(updated)
    result["rows"] = output_rows
    result["counts"] = dict(baseline["counts"], equipment_status_counts=dict(active_counts))
    prefix_entries = sum(len(prefix) for values in prefixes.values() for prefix in values.values())
    result["equipment_silhouette_constraint"] = {"schema_version": 1, "profile": PROFILE, "status": "applied_restrict_only",
        "baseline_profile": BASELINE_PROFILE, "baseline_canonical_sha256": sha256(baseline_raw).hexdigest(),
        "baseline_equipment_status_counts": baseline["counts"]["equipment_status_counts"], "input_bindings": bindings,
        "construction_frames": [row["frame"] for row in selected], "point_rows": [json.loads(raw) for raw in point_rows],
        "counts": {"source_frames": source_count, "construction_pair_slots": cutoff * len(keys),
            "retained_pose_pair_slots": len(rows) * len(keys), "baseline_candidates": decisions["kept"] + decisions["excluded"] + decisions["unavailable"],
            "kept_candidates": decisions["kept"], "excluded_candidates": decisions["excluded"], "unavailable_candidates": decisions["unavailable"],
            "unchanged_construction_noncandidates": decisions["not_candidate"], "excluded_later_pose_pairs": decisions["excluded_later"],
            "point_status_counts": dict(point_counts)},
        "work": {"selected_frame_queries": cutoff, "point_slots": cutoff * len(keys), "window_queries": decisions["kept"] + decisions["excluded"] + decisions["unavailable"],
            "serialized_point_cache_bytes": point_bytes, "prefix_counter_entries": prefix_entries,
            "prefix_counter_buffer_bytes": sum(len(prefix) * prefix.itemsize for values in prefixes.values() for prefix in values.values()),
            "prepared_row_bytes": source_cache_bytes, "raster_calls": 0, "pose_annotation_queries": 0},
        "bounds": {"cooperative_seconds": MAX_SECONDS, "within_original_deadline": True, "additional_cache_bytes": MAX_CACHE_BYTES,
            "complete_output_bytes": MAX_OUTPUT_BYTES, "construction_poses": 540, "equipment_ids": 8, "wrists": 2},
        "contact": "unknown", "physical_noncontact": "unknown", "grasp": "unknown", "load": "unknown",
        "physical_validation": "unverified", "body_or_floor_installed": False,
        "scope": "Restrict original image co-motion candidates using every original window point; masks and authored wrist ranges do not establish physical contact, noncontact, identity or support",
        "baseline_recovery": "Remove this top-level member and each pair's co_motion_baseline_status/silhouette_constraint, restore its status and original equipment_status_counts; every other field remains exact"}
    _encode(result, MAX_OUTPUT_BYTES, deadline)
    _check(deadline)
    return result
