"""Copy qualified saved image estimates into explicit authored surface ranges."""

from collections import Counter
from dataclasses import dataclass
from hashlib import sha256
import json
import math
import time

from weft.io import canonical_hash
from weft.source_surface_anchors import _annotations, _document, _sources
from weft.source_surface_interval_construction import _encode


PROFILE = "source-pose-surface-annotations-v1"
MAX_SECONDS, MAX_POSE_BYTES, MAX_OBSERVATION_BYTES = 10., 8 << 20, 16 << 20
MAX_CACHE_BYTES, MAX_ANNOTATION_BYTES = 8 << 20, 4 << 20
LANDMARKS = {"left_shoulder": 11, "right_shoulder": 12, "left_elbow": 13, "right_elbow": 14,
    "left_wrist": 15, "right_wrist": 16, "left_hip": 23, "right_hip": 24,
    "left_knee": 25, "right_knee": 26, "left_ankle": 27, "right_ankle": 28,
    "left_heel": 29, "right_heel": 30, "left_foot_index": 31, "right_foot_index": 32}
RAW_MODEL_STATUS = "single_model_hypothesis_not_verified_person_count"


def _require(value, message):
    if not value:
        raise ValueError("Pose surface annotations: " + message)


def _check(deadline):
    if time.monotonic() >= deadline:
        raise TimeoutError("Pose surface annotations cooperative deadline exceeded")


def _number(value, low, high):
    return type(value) in (int, float) and math.isfinite(value) and low <= value <= high


def _text(value, limit):
    return type(value) is str and 0 < len(value.strip()) <= limit


def _representation(frame, declaration_present, raw_declared):
    # A completed repair can retain a false recompute flag or null derivation ID.
    if any(key in frame for key in ("identity_raw", "identity_derivation_id", "identity_recompute_required")):
        return "identity_derived", "identity_derived_requires_separate_original_proof"
    if "image_refinement" in frame or "image_refinement_raw" in frame:
        return "crop_derived", "crop_derived_requires_separate_original_proof"
    if declaration_present and not raw_declared:
        return "unsupported_identity_declaration", "unsupported_or_inconsistent_identity_declaration"
    if "identity_status" not in frame:
        return "supplied_unmarked_image_estimate_not_authenticated_raw", None
    if frame["identity_status"] != RAW_MODEL_STATUS:
        return "unsupported_identity_status", "unsupported_identity_status"
    if not raw_declared:
        return "unqualified_raw_model_declaration", "raw_identity_declaration_required"
    return "declared_raw_model_estimate_not_authenticated_identity", None


@dataclass(frozen=True)
class PoseSurfaceAnnotations:
    """Exact source inputs, ordinary annotation bytes and bounded cached diagnostics."""

    input_documents: tuple[tuple[str, bytes], ...]
    pose_document: bytes
    _annotation: bytes
    _metadata: bytes
    _rows: tuple[bytes, ...]
    precursor_custody: object | None = None

    def annotation_document(self):
        return self._annotation

    def describe(self):
        return json.loads(self._metadata)

    def frame(self, index):
        _require(type(index) is int and 0 <= index < len(self._rows), "original frame outside range")
        return json.loads(self._rows[index])


def prepare_pose_surface_annotations(observation_documents, *, pose_document, source_sha256,
        channels, landmark_bindings, minimum_confidence, uncertainty_method, uncertainty_note, precursor_custody=None):
    """Use supplied image estimates, not recovered identity, anatomy or contact.

    Existing channel point_role must equal its explicit landmark_name. Per-channel
    uncertainty_px and the confidence threshold are authored, never calibrated.
    """
    deadline = time.monotonic() + MAX_SECONDS
    _require(type(observation_documents) is dict and 1 <= len(observation_documents) <= 8
             and all(type(raw) is bytes and 0 < len(raw) <= MAX_POSE_BYTES for raw in observation_documents.values())
             and sum(map(len, observation_documents.values())) <= MAX_OBSERVATION_BYTES,
             "one to eight exact observation documents must fit sixteen MiB")
    _require(type(pose_document) is bytes and 0 < len(pose_document) <= MAX_POSE_BYTES,
             "exact original pose document must fit eight MiB")
    _require(_number(minimum_confidence, 0, 1), "explicit authored confidence threshold in [0,1] required")
    _require(_text(uncertainty_method, 192) and _text(uncertainty_note, 192),
             "explicit bounded uncertainty method/note required; confidence is not an error bound")
    source, documents, observations, admitted_records, _, _ = _sources(observation_documents, source_sha256, deadline)
    count = max(source["decoded_frames"], source["metadata_frames"])
    _require(count <= 1800, "complete source exceeds 1800 frames")
    pose = _document(pose_document, deadline)
    _require(type(pose.get("schema_version")) is int and pose["schema_version"] == 1
             and pose.get("profile") == "exercise-pose-v1" and pose.get("source_sha256") == source_sha256,
             "exact source-bound exercise-pose-v1 input required")
    _require(type(pose.get("frame_count")) is int and type(pose.get("decoded_frames")) is int
             and pose["frame_count"] == source["metadata_frames"] and pose["decoded_frames"] == source["decoded_frames"],
             "original decoded/container denominators differ")
    frames = pose.get("frames")
    _require(type(frames) is list and len(frames) <= 900, "at most 900 saved pose slots")
    declaration_present = "pose_identity" in pose
    declaration = pose.get("pose_identity")
    raw_declared = (type(declaration) is dict and declaration.get("mode") == "raw"
                    and type(declaration.get("repaired_frames")) is int and declaration["repaired_frames"] == 0)
    indexed = {}
    for frame in frames:
        _check(deadline)
        _require(type(frame) is dict and type(frame.get("frame")) is int and 0 <= frame["frame"] < count
                 and frame["frame"] not in indexed and frame.get("id") == f"{source_sha256}:frame:{frame['frame']}"
                 and _number(frame.get("time_s"), 0, 60)
                 and (frame.get("decoder_timestamp_s") is None or _number(frame["decoder_timestamp_s"], 0, 60)),
                 "duplicate/foreign pose identity or nonfinite original clock")
        indexed[frame["frame"]] = frame
    base_annotation = {"schema_version": 1, "profile": "source-surface-annotations-v1", "source_sha256": source_sha256,
        "quality": "authored", "provenance": "Pending explicit source-pose annotation provenance", "channels": channels, "annotations": []}
    checked_channels, _ = _annotations(base_annotation, source, documents, observations, deadline)
    _require(type(landmark_bindings) is list and len(landmark_bindings) == len(checked_channels),
             "one explicit landmark binding per channel required")
    bindings = {}
    for binding, channel in zip(landmark_bindings, checked_channels, strict=True):
        _require(type(binding) is dict and set(binding) == {"channel_id", "landmark_name", "uncertainty_px"}
                 and binding["channel_id"] == channel["id"] and binding["landmark_name"] in LANDMARKS
                 and channel["point_role"] == binding["landmark_name"]
                 and _number(binding["uncertainty_px"], 0, 64),
                 "exact named landmark/point_role and explicit 0–64 pixel range required; no endpoint substitution")
        bindings[channel["id"]] = binding
    pose_pin = canonical_hash(pose)
    dimension_reason = None if all(pose.get(key) == source[target] for key, target in
        (("width", "width_px"), ("height", "height_px"))) else "pose_dimensions_not_original_source"
    records, source_reasons = {}, {}
    for _, raw in sorted(observation_documents.items()):
        _check(deadline)
        digest = sha256(raw).hexdigest(); document = _document(raw, deadline)
        provenance = document.get("provenance", {})
        source_reasons[digest] = (None if provenance.get("adapter") == "exercise"
            and provenance.get("input_evidence_sha256") == pose_pin else "canonical_pose_document_binding_differs")
        for record in document["evidence"]:
            # Index identity metadata only; withheld payloads never qualify a point.
            key = (digest, record["id"])
            if key in admitted_records:
                records[key] = record
        del document
    provenance = {"profile": PROFILE, "pose_raw_sha256": sha256(pose_document).hexdigest(),
        "pose_canonical_sha256": pose_pin, "source_sha256": source_sha256,
        "coordinate_policy": "supplied_unmarked_image_estimates; raw_identity_and_anatomy_not_authenticated",
        "landmark_schema": "mediapipe_33_image", "confidence_threshold_authored": minimum_confidence,
        "uncertainty_status": "authored_pixel_range_not_calibrated_from_confidence",
        "uncertainty_method": uncertainty_method, "uncertainty_note": uncertainty_note,
        "channel_landmark_index_uncertainty_px": [[b["channel_id"], LANDMARKS[b["landmark_name"]], b["uncertainty_px"]]
                                                  for b in landmark_bindings],
        "contact_load_grasp": "unknown", "ground_or_equipment_inferred": False}
    if declaration_present:
        provenance["identity_declaration"] = {
            "status": "consistent_raw_declaration_not_authenticated" if raw_declared else "unsupported_or_inconsistent",
            "declaration_canonical_sha256": canonical_hash(declaration)}
        if raw_declared:
            provenance["coordinate_policy"] = "declared_raw_or_unmarked_image_estimates; raw_identity_and_anatomy_not_authenticated"
            provenance["identity_declaration"].update(mode="raw", repaired_frames=0)
    precursor_meta = None
    if precursor_custody is not None:
        from weft.pose_precursor_custody import PosePrecursorCustody

        _require(type(precursor_custody) is PosePrecursorCustody, "explicit prepared precursor custody required")
        precursor_meta = precursor_custody.describe()
        _require(precursor_meta["input_raw_sha256s"]["evidence.json"] == provenance["pose_raw_sha256"],
                 "precursor custody belongs to different exact context bytes")
        pins = precursor_meta["input_raw_sha256s"]
        provenance["coordinate_policy"] = "exact_saved_precursor_image_matches_only; model_identity_not_authenticated"
        provenance["precursor_custody"] = {"profile": precursor_meta["profile"],
            "manifest_sha256": pins["manifest.json"], "raw_sha256": pins["pose-landmarks.json"],
            "audit_sha256": pins["identity-audit.json"], "mode": precursor_meta["mode"],
            "audit_unavailable_frames": precursor_meta["audit_unavailable_frames"], "identity_authenticated": False}
        provenance["identity_declaration"].update(status="original_mode_retained_precursor_custody_only",
            mode=precursor_meta["mode"], repaired_frames=precursor_meta["recorded_repaired_frames"])
    provenance_raw = _encode(provenance, 2000, deadline)
    base_annotation["provenance"] = provenance_raw.decode()
    rows, row_bytes, status_counts, selected_count, payload_reads = [], 0, Counter(), 0, 0
    pose_hashes, payload_hashes = {}, {}
    qualifier = None if precursor_custody is None else precursor_custody._begin(pose_document)
    for frame in range(count):
        _check(deadline); supplied = indexed.get(frame)
        row = {"source_frame": frame, "source_status": "decoded" if frame < source["decoded_frames"] else "metadata_only_not_decoded",
            "pose_evidence_id": None if supplied is None else supplied["id"],
            "pose_time_s": None if supplied is None else supplied["time_s"],
            "decoder_timestamp_s": None if supplied is None else supplied.get("decoder_timestamp_s"), "channels": []}
        for channel in checked_channels:
            _check(deadline)
            binding, entity = bindings[channel["id"]], channel["entity_ref"]
            digest, object_id = entity["document_sha256"], entity["object_id"]
            observed = observations.get((digest, object_id, frame))
            identifier = None if supplied is None else supplied["id"]
            key = (digest, identifier); evidence = admitted_records.get(key)
            reason = None
            if frame >= source["decoded_frames"]: reason = "metadata_only_not_decoded"
            elif observed is None: reason = "missing_source_observation"
            elif observed["partition"] == "unsampled": reason = "source_not_sampled"
            elif observed["partition"] != "construction" or observed["admission"] != "included": reason = "excluded_evidence_partition"
            elif observed["visibility"] not in {"detected", "partial"}: reason = "source_entity_unavailable"
            elif supplied is None: reason = "original_pose_slot_unavailable"
            elif source_reasons[digest]: reason = source_reasons[digest]
            elif dimension_reason: reason = dimension_reason
            elif (evidence is None or evidence["object_id"] != object_id or evidence["frame"] != frame
                  or evidence["kind"] != "pose_model_output" or evidence["status"] != "inferred"
                  or {"document_sha256": digest, "evidence_id": identifier} not in observed["evidence_refs"]):
                reason = "source_pose_entity_evidence_differs"
            elif supplied["time_s"] != evidence["time_s"] or supplied["time_s"] != observed["time_s"]:
                reason = "source_pose_clock_differs"
            admitted = reason is None
            annotation, pixel, confidence, visibility, presence, representation = None, None, None, None, None, "not_read_unqualified_source"
            if admitted:
                if frame not in pose_hashes:
                    pose_hashes[frame] = canonical_hash(supplied)
                if key not in payload_hashes:
                    payload_hashes[key] = canonical_hash(records[key]["payload"]); payload_reads += 1
                if payload_hashes[key] != pose_hashes[frame]:
                    reason = "source_pose_payload_differs"
                else:
                    if qualifier is None:
                        representation, reason = _representation(supplied, declaration_present, raw_declared)
                    else:
                        qualified = qualifier.qualify(frame)
                        representation, reason = qualified["representation"], qualified["reason"]
                if reason is None:
                    landmarks = supplied.get("landmarks")
                    if (supplied.get("status") not in {"tracked", "tracked_partial"}
                            or type(supplied.get("pose_count")) is not int or supplied["pose_count"] != 1):
                        reason = "unavailable_or_ambiguous_pose"
                    elif type(landmarks) is not list or len(landmarks) != 33:
                        reason = "missing_33_image_landmarks"
                    else:
                        point = landmarks[LANDMARKS[binding["landmark_name"]]]
                        if type(point) is not dict or not all(_number(point.get(k), 0, 1) for k in ("visibility", "presence")):
                            reason = "unknown_landmark_confidence"
                        else:
                            visibility, presence = point["visibility"], point["presence"]
                            confidence = min(visibility, presence)
                            if not all(_number(point.get(k), 0, 1) for k in ("x", "y")):
                                reason = "outside_or_unavailable_source_image"
                            elif confidence < minimum_confidence:
                                reason = "below_authored_confidence_threshold"
                            else:
                                pixel = [point["x"] * source["width_px"], point["y"] * source["height_px"]]
                annotation = {"id": f"{source_sha256}:frame:{frame}:surface-anchor:{channel['id']}", "channel_id": channel["id"],
                    "frame": frame, "time_s": observed["time_s"], "observation_id": observed["id"],
                    "status": "annotated" if pixel is not None else "unknown", "pixel": pixel,
                    "uncertainty_px": binding["uncertainty_px"] if pixel is not None else None}
                base_annotation["annotations"].append(annotation)
            selected_count += pixel is not None
            status = "copied_source_image_estimate" if pixel is not None else "unavailable"
            status_counts[reason or status] += 1
            row["channels"].append({"channel_id": channel["id"], "entity_ref": entity, "landmark_name": binding["landmark_name"],
                "landmark_index": LANDMARKS[binding["landmark_name"]], "status": status, "reason": reason,
                "observation_id": None if observed is None else observed["id"], "source_time_s": None if observed is None else observed["time_s"],
                "source_partition": None if observed is None else observed["partition"], "source_visibility": None if observed is None else observed["visibility"],
                "pose_record_ref": None if evidence is None else {"document_sha256": digest, "evidence_id": identifier},
                "pixel": pixel, "source_confidence": confidence, "source_visibility_score": visibility, "source_presence_score": presence,
                "authored_uncertainty_px": binding["uncertainty_px"], "representation": representation,
                "annotation_id": None if annotation is None else annotation["id"], "contact": "unknown"})
        if qualifier is not None:
            row["precursor_custody"] = qualifier.diagnostic(frame)
        encoded = _encode(row, 128 << 10, deadline); row_bytes += len(encoded)
        _require(row_bytes <= MAX_CACHE_BYTES, "complete source cache exceeds eight MiB; no rows dropped")
        rows.append(encoded)
    annotation_raw = _encode(base_annotation, MAX_ANNOTATION_BYTES, deadline)
    _annotations(base_annotation, source, documents, observations, deadline)
    metadata = {"schema_version": 1, "profile": PROFILE, "status": "source_estimate_annotations_available" if selected_count else "unavailable",
        "provenance": provenance, "observation_document_sha256s": sorted(documents), "annotation_sha256": sha256(annotation_raw).hexdigest(),
        "source_clocks": {digest: document["time"] for digest, document in documents.items()},
        "counts": {"decoded_frames": source["decoded_frames"], "metadata_frames": source["metadata_frames"],
            "retained_source_frames": count, "retained_pose_slots": len(frames), "channel_slots": count*len(checked_channels),
            "copied_estimates": selected_count, "annotation_rows": len(base_annotation["annotations"]),
            "status_counts": dict(status_counts), "qualified_record_payload_checks": payload_reads},
        "cached_row_bytes": row_bytes, "annotation_bytes": len(annotation_raw),
        "confidence_is_pixel_error_bound": False, "raw_identity_authenticated": False, "contact": "unknown", "load": "unknown", "grasp": "unknown",
        "body_or_floor_installed": False, "physical_validation": "unverified", "geometry_evaluations": 0,
        "scope": "Exact supplied source-estimate custody plus authored image ranges; no camera, ground, equipment, anatomical identity or contact recovery",
        "bounds": {"cooperative_seconds": MAX_SECONDS, "source_frames": 1800, "pose_slots": 900, "channels": 12,
            "observation_bytes": MAX_OBSERVATION_BYTES, "pose_bytes": MAX_POSE_BYTES,
            "annotation_bytes": MAX_ANNOTATION_BYTES, "cache_bytes": MAX_CACHE_BYTES}}
    precursor_cache_bytes = 0
    if qualifier is not None:
        qualification, _ = qualifier.finish()
        metadata["precursor_custody"] = json.loads(qualification)
        precursor_cache_bytes = len(precursor_custody._metadata) + sum(map(len, precursor_custody._source_rows))
        metadata["retained_precursor_cache_bytes"] = precursor_cache_bytes
    encoded_meta = _encode(metadata, 128 << 10, deadline)
    _require(row_bytes + len(encoded_meta) + len(annotation_raw) + precursor_cache_bytes <= MAX_CACHE_BYTES,
             "complete annotation/metadata/source cache exceeds eight MiB")
    _check(deadline)
    return PoseSurfaceAnnotations(tuple(sorted(observation_documents.items())), pose_document, annotation_raw, encoded_meta, tuple(rows), precursor_custody)
