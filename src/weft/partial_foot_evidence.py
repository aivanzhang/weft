"""Inspect saved ankle/toe estimates without completing a heel or admitting feet."""

from collections import Counter
from dataclasses import dataclass
import json
import math
import time

from weft.calibrated_foot_reference import load_calibrated_foot_reference
from weft.foot_reference import LANDMARKS
from weft.io import canonical_hash
from weft.multiview_body import _points
from weft.multiview_foot_reference import _documents
from weft.multiview_pose import CONTRACT as RAY_CONTRACT
from weft.world_alignment import load_world_alignment


PROFILE = "partial-calibrated-foot-evidence-v1"
MAX_INPUT_BYTES, MAX_DOCUMENT_BYTES = 24 << 20, 8 << 20
MAX_OUTPUT_BYTES, MAX_ROW_BYTES, MAX_METADATA_BYTES = 8 << 20, 16 << 10, 64 << 10
MAX_SECONDS, MAX_FRAMES = 5., 1800
_NO_CONFIDENCE = "not_stored_in_saved_triangulation"


def _require(value, message):
    if not value:
        raise ValueError("Partial foot evidence: " + message)


def _check(deadline):
    if time.monotonic() > deadline:
        raise TimeoutError("Partial foot evidence cooperative deadline exceeded")


def _number(value, low, high):
    return type(value) in (int, float) and math.isfinite(value) and low <= value <= high


def _encoded(value, limit):
    raw = json.dumps(value, sort_keys=True, allow_nan=False, separators=(",", ":")).encode()
    _require(len(raw) <= limit, "prepared cache exceeds its byte bound")
    return raw


def _prediction(reference, prediction, pin):
    _require(pin == reference["input_pins"]["predictions"], "triangulation bytes/canonical identity differ from original reference")
    _require(type(prediction.get("schema_version")) is int and prediction["schema_version"] == 1
             and canonical_hash(prediction.get("contract")) == canonical_hash(dict(RAY_CONTRACT))
             and prediction.get("views") == reference["views"]
             and prediction.get("physical_validation") == "unverified"
             and prediction.get("monocular_world_landmarks_used") is False
             and prediction.get("reference_body_or_contacts_used") is False,
             "saved two-view contract, view pins or unverified scope changed")
    count, rows = prediction.get("source_frames"), prediction.get("frames")
    _require(type(count) is int and 2 <= count <= MAX_FRAMES and type(rows) is list and len(rows) == count
             and type(prediction.get("joint_slots")) is int and prediction["joint_slots"] == count * 33
             and count == reference["coverage"]["original_source_frames"], "complete triangulation/source denominator required")
    return rows


def _row_admission(row, binding_valid):
    if row["partition"] != "construction_prefix":
        return "excluded_later_source"
    if "metadata_only_not_decoded" in row["view_slot_statuses"]:
        return "one_or_more_views_not_decoded"
    if None in row["source_evidence_ids"]:
        return "pose_not_sampled_in_both_views"
    if not binding_valid:
        return "body_world_binding_unavailable"
    if not row["body_pose_available"]:
        return "body_pose_unavailable"
    return None


def _endpoint(source_row, index, alignment, mapping, ground):
    """Only admitted ankle/toe records reach coordinate or diagnostic interpretation."""
    item = source_row["joints"][index]
    values = item.get("world_point_m")
    _require(type(values) is list and len(values) == 3
             and all(_number(value, -2000, 2000) for value in values), "stored estimated point must be bounded XYZ")
    # Reuse the original saved-point validator; never repeat ray reconstruction.
    point = _points(source_row, (index,))[0].tolist()
    observations = item.get("observation_statuses")
    _require(observations == ["available", "available"], "stored estimate requires both original image-observation statuses")
    errors = item.get("reprojection_px")
    _require(_number(item.get("ray_angle_degrees"), RAY_CONTRACT["minimum_ray_angle_degrees"], 90.)
             and _number(item.get("ray_separation_m"), 0., 4000.)
             and type(errors) is list and len(errors) == 2
             and all(_number(value, 0., 4096.) for value in errors), "stored ray diagnostics must remain finite and bounded")
    mapped = alignment.map_state(from_frame=mapping["from_frame"], to_frame=mapping["to_frame"],
                                 time_s=source_row["time_s"], position=point)
    _require(mapped["time_s"] == source_row["time_s"], "body mapping must not change the original clock")
    return {"source_world_point_m": point, "body_point": mapped["position"],
            "partial_endpoint_signed_ground_gap_m": point[2] - ground["height_m"],
            "stored_ray_diagnostics": {"observation_statuses": observations,
                "ray_angle_degrees": item["ray_angle_degrees"], "ray_separation_m": item["ray_separation_m"],
                "reprojection_px": errors}}


@dataclass(frozen=True)
class PartialFootEvidence:
    """Retain exact raw evidence once and parse only the requested bounded row."""

    body_bytes: bytes
    reference_bytes: bytes
    triangulation_bytes: bytes
    _metadata: bytes
    _frames: tuple

    @property
    def frame_count(self):
        return len(self._frames)

    def describe(self):
        return json.loads(self._metadata)

    def frame(self, index):
        _require(type(index) is int and 0 <= index < len(self._frames), "frame index outside original source slots")
        return json.loads(self._frames[index])

    def iter_frame_bytes(self):
        """Stream immutable cached rows without materializing another full evidence tree."""
        return iter(self._frames)


def load_partial_foot_evidence(body_bytes, reference_bytes, triangulation_bytes):
    """Prepare source-pinned partial inspection; no heel geometry, I/O, fitting or admission.

    Up to 8 MiB per exact document / 24 MiB total, 1,800 source rows, five-second
    cooperative preparation, and an 8 MiB cache. Numeric endpoint confidence is
    unknown because the stored triangulation does not contain that measurement.
    """
    started = time.monotonic()
    deadline = started + MAX_SECONDS
    raws = {"body": body_bytes, "reference": reference_bytes, "predictions": triangulation_bytes}
    _require(all(type(raw) is bytes and 0 < len(raw) <= MAX_DOCUMENT_BYTES for raw in raws.values())
             and sum(map(len, raws.values())) <= MAX_INPUT_BYTES, "three bounded exact input byte strings required")
    strict = load_calibrated_foot_reference(body_bytes, reference_bytes)
    strict_description = strict.describe()
    _check(deadline)
    data, pins = _documents(raws, deadline)
    reference, prediction = data["reference"], data["predictions"]
    original_rows = _prediction(reference, prediction, pins["predictions"])
    binding_valid = strict_description["body_clock_binding_validated"]
    mapping = reference["world_to_body"]
    alignment = load_world_alignment(mapping) if binding_valid else None
    _require(not binding_valid or alignment.ready, "validated body mapping became unavailable")
    frames, source_counts, geometry_counts, partial_gaps = [], Counter(), Counter(), []
    cache_bytes = 0
    for index, (row, original) in enumerate(zip(reference["frames"], original_rows, strict=True)):
        _check(deadline)
        _require(type(original) is dict and type(original.get("frame")) is int and original["frame"] == index
                 and _number(original.get("time_s"), 0., 60.) and original["time_s"] == row["time_s"]
                 and original.get("source_evidence_ids") == row["source_evidence_ids"]
                 and type(original.get("joints")) is list and len(original["joints"]) == 33,
                 "saved triangulation source identities or clocks differ")
        admission = _row_admission(row, binding_valid)
        feet = {}
        for side, indices in LANDMARKS.items():
            endpoints = {}
            source_foot = row["feet"][side]
            for endpoint, landmark in indices.items():
                item = original["joints"][landmark]
                original_status = source_foot["original_endpoint_statuses"][endpoint]
                _require(type(item) is dict and item.get("status") == original_status,
                         "original endpoint status differs from the strict reference")
                source_counts[original_status] += 1
                reason = ("unsupported_by_partial_profile" if endpoint == "heel" else admission
                          or ("original_endpoint_not_estimated" if original_status != "estimated" else None))
                record = {"landmark_index": landmark, "original_status": original_status,
                          "geometry_status": reason or "conditional_saved_estimate", "confidence": None,
                          "confidence_status": _NO_CONFIDENCE, "source_world_point_m": None, "body_point": None,
                          "partial_endpoint_signed_ground_gap_m": None, "stored_ray_diagnostics": None}
                if reason is None:
                    record.update(_endpoint(original, landmark, alignment, mapping, reference["ground"]))
                    partial_gaps.append(record["partial_endpoint_signed_ground_gap_m"])
                geometry_counts[record["geometry_status"]] += 1
                endpoints[endpoint] = record
            feet[side] = {"strict_reference_status": source_foot["status"], "strict_reference_reason": source_foot["reason"],
                          "endpoints": endpoints, "heel_orientation": None, "physical_admission": False}
        prepared = {"frame": index, "time_s": row["time_s"], "partition": row["partition"],
                    "view_slot_statuses": row["view_slot_statuses"], "source_evidence_ids": row["source_evidence_ids"],
                    "source_clocks": row["source_clocks"], "body_pose_available": row["body_pose_available"],
                    "row_geometry_exclusion": admission, "feet": feet}
        encoded = _encoded(prepared, MAX_ROW_BYTES)
        cache_bytes += len(encoded)
        _require(cache_bytes <= MAX_OUTPUT_BYTES - MAX_METADATA_BYTES, "complete partial-evidence cache exceeds eight MiB")
        frames.append(encoded)
    metadata = {
        "schema_version": 1, "profile": PROFILE,
        "status": "partial_endpoint_estimates_available" if partial_gaps else "unavailable",
        "input_pins": pins, "source_sha256": reference["source_sha256"], "views": reference["views"],
        "body_coordinate_frame": strict_description["body_coordinate_frame"], "world_to_body": mapping,
        "ground": reference["ground"], "original_start": strict_description["original_start"],
        "body_clock_binding_validated": binding_valid,
        "strict_consumer": {"profile": strict_description["profile"], "status": strict_description["status"],
                            "ready": strict.ready, "unavailable_reasons": strict_description["unavailable_reasons"]},
        "original_reference_coverage": reference["coverage"],
        "coverage": {"source_slots": len(frames), "foot_slots": len(frames) * 2, "endpoint_slots": len(frames) * 6,
                     "original_endpoint_status_counts": dict(source_counts), "geometry_status_counts": dict(geometry_counts)},
        "partial_endpoint_gap_summary": {"count": len(partial_gaps), "negative": sum(value < 0 for value in partial_gaps),
                                         "minimum_m": min(partial_gaps, default=None), "maximum_m": max(partial_gaps, default=None)},
        "original_gap_series": "Retained unchanged in exact reference_bytes; partial endpoint series is a distinct denominator",
        "qualification": "Exact pinned stored estimates only; no independent raw-image/camera validation",
        "confidence": {"status": _NO_CONFIDENCE, "original_minimum_confidence_rule": RAY_CONTRACT["minimum_confidence"],
                       "rule_is_not_a_measurement": True},
        "physical_validation": "unverified", "physical_admission": False, "contact": "unknown", "load": "unknown", "grasp": "unknown",
        "heel_geometry": "unsupported_by_partial_profile", "anatomical_orientation": "unknown",
        "limits": ["Two endpoints leave one free rotation; fixed triangle priors need not match their separation",
                   "A heel needs additional original-frame evidence; no authored roll, interpolation, floor movement or later-start fallback",
                   "Camera scale, ground and body coordinates remain supplied priors, not measured anatomy or physical contact"],
    }
    encoded_metadata = _encoded(metadata, MAX_METADATA_BYTES)
    _require(cache_bytes + len(encoded_metadata) <= MAX_OUTPUT_BYTES, "complete output cache exceeds eight MiB")
    _check(deadline)
    return PartialFootEvidence(body_bytes, reference_bytes, triangulation_bytes, encoded_metadata, tuple(frames))
