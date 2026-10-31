"""Conditional image-ray/surface intersections with source custody, never contact truth."""

from collections import Counter
from hashlib import sha256
import json
import math
import time

import numpy as np

from weft.camera_ground import validate_ground
from weft.camera_pose import _undistort, validate_camera
from weft.world_alignment import load_world_alignment
from weft.world_evidence import _check_tree, _nonfinite, _unique_object, load_world_evidence


PROFILE = "source-surface-anchors-v1"
CONTRACT = {
    "maximum_seconds": 10., "maximum_annotated_frames": 900, "maximum_channels": 12,
    "maximum_decoded_frames": 1800, "maximum_additional_input_bytes": 4 << 20,
    "maximum_output_bytes": 8 << 20, "maximum_pixel_uncertainty": 64.,
    "parallel_ray_tolerance": 1e-10, "maximum_ray_depth_m": 1000.,
    "scope": "Conditional coordinates on supplied surfaces, not measured contact, force, grasp or body admission",
}


def _require(value, message):
    if not value:
        raise ValueError("Source surface anchors: " + message)


def _check(deadline):
    if time.monotonic() >= deadline:
        raise TimeoutError("Source surface anchors exceeded the ten-second bound")


def _number(value, low, high):
    return type(value) in (int, float) and math.isfinite(value) and low <= value <= high


def _fields(value, names, label):
    _require(type(value) is dict and set(value) == set(names), label + " fields differ")


def _text(value, limit=240):
    return type(value) is str and 1 <= len(value.strip()) <= limit


def _document(raw, deadline):
    if raw is None:
        return None
    _check(deadline)
    try:
        value = json.loads(raw, object_pairs_hook=_unique_object, parse_constant=_nonfinite)
        _check_tree(value)
    except (UnicodeError, json.JSONDecodeError, RecursionError) as error:
        raise ValueError("Source surface anchors require bounded valid JSON") from error
    _require(type(value) is dict, "input JSON object required")
    _check(deadline)
    return value


def _sources(documents, source_sha256, deadline):
    _check(deadline)
    registry = load_world_evidence(documents)
    exported = json.loads(registry.construction_bytes())
    sources = exported["sources"]
    _require(all(s["source"]["sha256"] == source_sha256 for s in sources), "one exact original source is required; derivative hashes cannot be relabeled")
    first = sources[0]["source"]
    signature = (first["width_px"], first["height_px"], first["decoded_frames"], first["metadata_frames"])
    _require(first["decoded_frames"] <= CONTRACT["maximum_decoded_frames"], "decoded frame bound exceeded")
    observations, evidence, by_document, receipts = {}, {}, {}, {}
    for source in sources:
        _check(deadline)
        metadata = source["source"]
        _require((metadata["width_px"], metadata["height_px"], metadata["decoded_frames"], metadata["metadata_frames"]) == signature,
                 "original source dimension/coverage declarations differ")
        for key in ("fps", "frame_times_s"):
            if key in first and key in metadata:
                _require(first[key] == metadata[key], "same-basis original source clocks differ")
        _require(metadata.get("decode_interval", [0, metadata["decoded_frames"] - 1]) == [0, metadata["decoded_frames"] - 1]
                 and metadata.get("source_complete") is not False, "complete original source interval required")
        by_document[source["document_sha256"]] = source
        for row in source["observations"]:
            observations[(source["document_sha256"], row["object_id"], row["frame"])] = row
        for record in source["evidence"]:
            evidence[(source["document_sha256"], record["id"])] = record
        receipt_ids = {r["id"] for r in source["evidence"]
                       if r["kind"] in ("camera_calibration_measurement", "surface_height_measurement") and r["status"] == "observed"}
        if receipt_ids:
            original = _document(registry.source_bytes(source["document_sha256"]), deadline)
            receipts.update({(source["document_sha256"], r["id"]): r["payload"] for r in original["evidence"] if r["id"] in receipt_ids})
    allowed_ids = {digest: [r["id"] for r in source["evidence"]] for digest, source in by_document.items()}
    _check(deadline)
    return first, by_document, observations, evidence, allowed_ids, receipts


def _references(refs, evidence):
    _require(type(refs) is list and len(refs) <= 32, "bounded qualified evidence references required")
    keys = []
    for ref in refs:
        _fields(ref, ("document_sha256", "evidence_id"), "evidence reference")
        key = (ref["document_sha256"], ref["evidence_id"])
        _require(all(_text(item) for item in key) and key in evidence and key not in keys,
                 "evidence reference is unavailable, excluded, duplicate or foreign")
        keys.append(key)
    return [evidence[key] for key in keys]


def _receipt_bound(record, receipts, source, calibration, surface=None, world_frame=None):
    payload = receipts.get((record["document_sha256"], record["id"]))
    if (type(payload) is not dict or payload.get("source_sha256") != source
            or payload.get("calibration_sha256") != calibration or payload.get("units") != "m"
            or not _text(payload.get("method"), 1000) or not _text(payload.get("uncertainty"), 2000)):
        return False
    if surface is not None:
        return (payload.get("surface_id") == surface["id"] and payload.get("world_frame_id") == world_frame["id"]
                and _number(payload.get("height_m"), -1000, 1000) and payload["height_m"] == surface["height_m"])
    return True


def _surfaces(document, source, camera_hash, evidence, receipts):
    if document is None:
        return {}, [], None
    _fields(document, ("schema_version", "profile", "source_sha256", "calibration_sha256", "world_frame",
                       "calibration_evidence_refs", "surfaces"), "surface document")
    _require(type(document["schema_version"]) is int and document["schema_version"] == 1 and document["profile"] == "source-horizontal-surfaces-v1"
             and document["source_sha256"] == source["sha256"], "surface profile/source differs")
    _require(type(document["calibration_sha256"]) is str and len(document["calibration_sha256"]) == 64
             and set(document["calibration_sha256"]) <= set("0123456789abcdef"), "exact calibration identity required")
    if camera_hash is not None:
        _require(document["calibration_sha256"] == camera_hash, "surface belongs to different calibration bytes")
    frame = document["world_frame"]
    _fields(frame, ("id", "length_unit", "axes"), "world frame")
    _require(_text(frame["id"], 160) and frame["length_unit"] == "m" and frame["axes"] == "x_right_y_away_z_up",
             "surface world must explicitly use right-handed z-up metres")
    calibration_records = _references(document["calibration_evidence_refs"], evidence)
    rows = document["surfaces"]
    _require(type(rows) is list and len(rows) <= 12, "at most twelve declared surfaces required")
    result = {}
    for surface in rows:
        _fields(surface, ("id", "kind", "height_m", "bounds_xy_m", "quality", "provenance", "evidence_refs"), "surface")
        _require(_text(surface["id"], 80) and surface["id"] not in result
                 and surface["kind"] in ("ground", "static_support"), "unique ground/static-support identity required")
        records = _references(surface["evidence_refs"], evidence)
        _require(_text(surface["provenance"], 2000), "surface provenance required")
        if surface["height_m"] is None:
            _require(surface["quality"] == "unknown", "missing height must remain unknown")
        else:
            validate_ground({"schema_version": 1, "source_sha256": source["sha256"],
                             "calibration_sha256": document["calibration_sha256"], "world_axes": "right_handed_z_up",
                             "units": "m", "height_m": surface["height_m"], "quality": surface["quality"],
                             "provenance": surface["provenance"]},
                            {"source_sha256": source["sha256"], "calibration_sha256": document["calibration_sha256"]})
        bounds = surface["bounds_xy_m"]
        _require(bounds is None and surface["kind"] == "ground" or type(bounds) is list and len(bounds) == 4
                 and all(_number(v, -1000, 1000) for v in bounds) and bounds[0] < bounds[2] and bounds[1] < bounds[3],
                 "static support needs finite nonempty XY extents; only ground may be unbounded")
        measured = any(r["kind"] == "surface_height_measurement" and r["status"] == "observed"
                       and _receipt_bound(r, receipts, source["sha256"], document["calibration_sha256"], surface, frame) for r in records)
        result[surface["id"]] = {**surface, "measurement_receipt_supplied": measured}
    return result, calibration_records, frame


def _annotations(document, source, by_document, observations, deadline):
    if document is None:
        return [], {}
    _fields(document, ("schema_version", "profile", "source_sha256", "quality", "provenance", "channels", "annotations"), "annotation document")
    _require(type(document["schema_version"]) is int and document["schema_version"] == 1 and document["profile"] == "source-surface-annotations-v1"
             and document["source_sha256"] == source["sha256"] and document["quality"] == "authored"
             and _text(document["provenance"], 2000), "annotations require source-bound authored provenance")
    channels = document["channels"]
    _require(type(channels) is list and 1 <= len(channels) <= CONTRACT["maximum_channels"], "one to twelve anchor channels required")
    channel_index = {}
    for channel in channels:
        _fields(channel, ("id", "entity_ref", "point_role", "surface_id"), "channel")
        _fields(channel["entity_ref"], ("document_sha256", "object_id"), "entity reference")
        ref = channel["entity_ref"]
        _require(_text(channel["id"], 80) and channel["id"] not in channel_index and _text(channel["point_role"], 120)
                 and _text(channel["surface_id"], 80), "unique named channel/point/surface required")
        _require(ref["document_sha256"] in by_document and (ref["document_sha256"], ref["object_id"], 0) in observations,
                 "channel entity must resolve in exact original-source observations")
        channel_index[channel["id"]] = channel
    rows = document["annotations"]
    _require(type(rows) is list and len(rows) <= CONTRACT["maximum_annotated_frames"] * len(channels), "annotation slot bound exceeded")
    lookup, frames = {}, set()
    for row in rows:
        _check(deadline)
        _fields(row, ("id", "channel_id", "frame", "time_s", "observation_id", "status", "pixel", "uncertainty_px"), "annotation")
        _require(row["channel_id"] in channel_index and type(row["frame"]) is int and 0 <= row["frame"] < source["decoded_frames"],
                 "annotation channel/source frame differs")
        frame, channel = row["frame"], channel_index[row["channel_id"]]
        ref = channel["entity_ref"]
        observation = observations[(ref["document_sha256"], ref["object_id"], frame)]
        _require(row["observation_id"] == observation["id"] and row["id"] == f"{source['sha256']}:frame:{frame}:surface-anchor:{channel['id']}"
                 and _number(row["time_s"], 0, 60) and math.isclose(row["time_s"], observation["time_s"], rel_tol=0, abs_tol=1e-9),
                 "annotation identity/clock differs from qualified source observation")
        _require(row["status"] in ("annotated", "occluded", "unknown"), "annotation status differs")
        key = (frame, channel["id"])
        _require(key not in lookup, "duplicate annotation slot")
        lookup[key] = row
        frames.add(frame)
    _require(len(frames) <= CONTRACT["maximum_annotated_frames"], "more than 900 annotated source frames")
    return channels, lookup


def _quaternion(rotation):
    # The supplied matrix is already validated; this only changes representation.
    r = rotation
    matrix = np.array([[r[0, 0] - r[1, 1] - r[2, 2], r[0, 1] + r[1, 0], r[0, 2] + r[2, 0], r[2, 1] - r[1, 2]],
                       [r[0, 1] + r[1, 0], r[1, 1] - r[0, 0] - r[2, 2], r[1, 2] + r[2, 1], r[0, 2] - r[2, 0]],
                       [r[0, 2] + r[2, 0], r[1, 2] + r[2, 1], r[2, 2] - r[0, 0] - r[1, 1], r[1, 0] - r[0, 1]],
                       [r[2, 1] - r[1, 2], r[0, 2] - r[2, 0], r[1, 0] - r[0, 1], r.trace()]]) / 3
    _, vectors = np.linalg.eigh(matrix)
    xyzw = vectors[:, -1]
    result = np.array([xyzw[3], *xyzw[:3]])
    return (result if result[0] >= 0 else -result).tolist()


def _alignment(rotation, translation, frame, source, target, refs, allowed_ids):
    origin = -rotation.T @ translation
    source_frame = {"id": f"camera:{source}:frame:{frame}", "length_unit": "m", "axes": "x_right_y_down_z_forward"}
    claim = {"status": "assumed", "method": "Explicit supplied camera transform; no fitting",
             "uncertainty": "Pinned input quality/provenance retained; physical calibration not independently verified", "evidence_refs": refs}
    document = {"schema_version": 1, "profile": "world-alignment-v1", "from_frame": source_frame,
                "to_frame": target, "spatial": {"scale": 1., "rotation_wxyz": _quaternion(rotation.T), "translation": origin.tolist()},
                "clock": {"rate": 1., "source_origin_s": 0., "target_origin_s": 0.},
                "claims": {"spatial": claim, "clock": {"status": "assumed", "method": "Exact same source-frame nominal camera clock",
                           "uncertainty": "Native observation clocks must match explicitly; no synchronization fitted", "evidence_refs": []}}}
    return load_world_alignment(document, evidence_ids_by_document=allowed_ids), source_frame, origin


def _intersect(pixel, uncertainty, surface, camera, transform, alignment, from_frame, origin, stamp):
    width, height = camera["width"], camera["height"]
    _require(type(pixel) is list and len(pixel) == 2 and all(_number(v, -1e6, 1e6) for v in pixel)
             and _number(uncertainty, 0, CONTRACT["maximum_pixel_uncertainty"]), "finite authored pixel and bounded uncertainty required")
    x, y = pixel
    pixels = np.array([[x, y]] if uncertainty == 0 else
                      [[x, y], [x - uncertainty, y - uncertainty], [x - uncertainty, y + uncertainty],
                       [x + uncertainty, y - uncertainty], [x + uncertainty, y + uncertainty]])
    if np.any(pixels < 0) or np.any(pixels[:, 0] > width) or np.any(pixels[:, 1] > height):
        return None, "pixel_uncertainty_box_outside_source_image"
    matrix, radial, inverse = transform
    rays = _undistort(pixels, matrix, radial, inverse=inverse)
    if rays is None:
        return None, "undistortion_unavailable"
    target_frame = alignment.document()["to_frame"]
    world_points = []
    for ray in rays:
        camera_ray = np.array([*ray, 1.])
        ray_point = alignment.map_state(from_frame=from_frame, to_frame=target_frame, time_s=stamp, position=camera_ray.tolist())
        direction = np.asarray(ray_point["position"]) - origin
        if abs(direction[2]) <= CONTRACT["parallel_ray_tolerance"]:
            return None, "ray_parallel_to_surface"
        depth = (surface["height_m"] - origin[2]) / direction[2]
        if not math.isfinite(depth) or not 0 < depth <= CONTRACT["maximum_ray_depth_m"]:
            return None, "intersection_behind_camera_or_beyond_bound"
        mapped = alignment.map_state(from_frame=from_frame, to_frame=target_frame,
                                     time_s=stamp, position=(depth * camera_ray).tolist())
        position = mapped["position"]
        bounds = surface["bounds_xy_m"]
        if bounds is not None and not (bounds[0] <= position[0] <= bounds[2] and bounds[1] <= position[1] <= bounds[3]):
            return None, "intersection_uncertainty_outside_declared_surface"
        world_points.append(position)
    points = np.asarray(world_points)
    return {"position_world_m": world_points[0], "sampled_pixel_box_world_range_m": {"minimum": points.min(axis=0).tolist(),
            "maximum": points.max(axis=0).tolist()}, "pixel_uncertainty_px": uncertainty,
            "uncertainty_scope": "Sampled pixel-box corners only, not a certified bound; excludes calibration and surface-height uncertainty"}, None


def reconstruct_source_surface_anchors(observation_documents, *, source_sha256, camera_document=None,
                                       surface_document=None, annotation_document=None):
    """Return source-complete conditional anchors from exact input bytes; perform no I/O."""
    deadline = time.monotonic() + CONTRACT["maximum_seconds"]
    supplied = (camera_document, surface_document, annotation_document)
    _require(all(raw is None or type(raw) is bytes for raw in supplied)
             and sum(len(raw) for raw in supplied if raw is not None) <= CONTRACT["maximum_additional_input_bytes"],
             "additional exact-byte inputs exceed 4 MiB")
    source, by_document, observations, evidence, allowed_ids, receipts = _sources(observation_documents, source_sha256, deadline)
    camera, surface_input, annotation_input = [_document(raw, deadline) for raw in supplied]
    camera_hash = sha256(camera_document).hexdigest() if camera_document is not None else None
    surfaces, calibration_records, world_frame = _surfaces(surface_input, source, camera_hash, evidence, receipts)
    channels, annotations = _annotations(annotation_input, source, by_document, observations, deadline)
    transform, inverse, calibration_reason = None, None, "missing_camera_calibration"
    if camera is not None:
        metadata = {"source_sha256": source_sha256, "width": source["width_px"], "height": source["height_px"],
                    "frame_count": source["metadata_frames"], "fps": source.get("fps", camera.get("fps"))}
        transform = validate_camera(camera, metadata)
        inverse = np.linalg.inv(transform[0])
        calibration_reason = None
        if camera["quality"] == "measured" and not any(r["kind"] == "camera_calibration_measurement" and r["status"] == "observed"
                and _receipt_bound(r, receipts, source_sha256, camera_hash) for r in calibration_records):
            calibration_reason = "missing_supplied_camera_measurement_receipt"
    missing = [name for name, value in (("camera_calibration", camera), ("surface_geometry", surface_input), ("annotations", annotation_input)) if value is None]
    rows, counts, used_alignments = [], Counter(), {}
    frame_total = max(source["decoded_frames"], source["metadata_frames"])
    for frame in range(frame_total):
        _check(deadline)
        anchors = []
        for channel in channels:
            ref = channel["entity_ref"]
            observation = observations.get((ref["document_sha256"], ref["object_id"], frame))
            annotation = annotations.get((frame, channel["id"]))
            anchor = {"channel_id": channel["id"], "annotation_id": annotation["id"] if annotation else None,
                      "observation_id": observation["id"] if observation else None,
                      "source_time_s": observation["time_s"] if observation else None,
                      "partition": observation["partition"] if observation else None,
                      "status": "unavailable", "reason": None, "position_world_m": None}
            reason = None
            if frame >= source["decoded_frames"]:
                reason = "metadata_only_not_decoded"
            elif observation["partition"] == "unsampled":
                reason = "source_not_sampled"
            elif observation["admission"] != "included":
                reason = "excluded_evidence_partition"
            elif observation["visibility"] not in ("detected", "partial"):
                reason = "source_entity_unavailable"
            elif annotation is None:
                reason = "missing_annotation"
            elif annotation["status"] != "annotated":
                reason = "annotation_" + annotation["status"]
            elif calibration_reason:
                reason = calibration_reason
            elif surface_input is None or channel["surface_id"] not in surfaces:
                reason = "missing_surface_geometry"
            elif surfaces[channel["surface_id"]]["height_m"] is None:
                reason = "unknown_surface_height"
            else:
                surface = surfaces[channel["surface_id"]]
                if surface["quality"] == "measured" and not surface["measurement_receipt_supplied"]:
                    reason = "missing_supplied_surface_measurement_receipt"
                elif frame >= len(camera["frames"]) or not math.isclose(observation["time_s"], camera["frames"][frame]["time_s"], rel_tol=0, abs_tol=1e-9):
                    reason = "camera_and_source_clocks_differ"
                else:
                    if frame not in used_alignments:
                        rotation, translation = transform[2][frame]
                        used_alignments[frame] = _alignment(rotation, translation, frame, source_sha256, world_frame,
                                                           surface_input["calibration_evidence_refs"], allowed_ids)
                    alignment, from_frame, origin = used_alignments[frame]
                    geometry, reason = _intersect(annotation["pixel"], annotation["uncertainty_px"], surface, camera,
                                                  (transform[0], transform[1], inverse), alignment, from_frame, origin, observation["time_s"])
                    if geometry is not None:
                        anchor.update(geometry, status="conditional_surface_anchor", camera_quality_claim=camera["quality"],
                                      surface_quality_claim=surface["quality"], annotation_quality="authored",
                                      alignment_id=f"camera-frame:{frame}")
            anchor["reason"] = reason
            counts[reason or anchor["status"]] += 1
            anchors.append(anchor)
        rows.append({"source_frame": frame, "status": "decoded" if frame < source["decoded_frames"] else "metadata_only_not_decoded", "anchors": anchors})
    result = {"schema_version": 1, "profile": PROFILE,
              "status": "conditional_anchors_available" if counts["conditional_surface_anchor"] else "unavailable",
              "missing_inputs": missing, "source_sha256": source_sha256, "world_frame": world_frame,
              "bindings": {"observation_document_sha256s": sorted(by_document), "camera_sha256": camera_hash,
                           "surfaces_sha256": sha256(surface_document).hexdigest() if surface_document is not None else None,
                           "annotations_sha256": sha256(annotation_document).hexdigest() if annotation_document is not None else None},
              "source_clocks": {digest: value["time"] for digest, value in by_document.items()},
              "channels": channels, "surfaces": list(surfaces.values()), "rows": rows,
              "alignments": {f"camera-frame:{frame}": value[0].document() for frame, value in used_alignments.items()},
              "counts": {"decoded_frames": source["decoded_frames"], "metadata_frames": source["metadata_frames"],
                         "retained_source_frames": frame_total, "declared_channels": len(channels),
                         "retained_anchor_slots": frame_total * len(channels), "supplied_annotation_slots": len(annotations),
                         "conditional_anchors": counts["conditional_surface_anchor"], "slot_status_counts": dict(counts)},
              "contract": dict(CONTRACT), "geometry_modified": False, "body_or_floor_installed": False,
              "contact": "unknown", "load": "unknown", "grasp": "unknown", "physical_validation": False,
              "limits": ["Pixel/surface membership is an authored hypothesis, not a recovered contact event",
                         "Metre units do not verify supplied calibration/height claims or calibrate a monocular body's scale",
                         "Measurement receipts are pinned construction evidence, not independently remeasured here",
                         "Static support geometry is explicit; moving or handheld supports are unsupported",
                         "Only construction evidence is used; excluded, occluded and unsampled slots remain unavailable",
                         "Anchors require an explicit world/body alignment before any future constraint installation"]}
    _check(deadline)
    encoded = json.dumps(result, sort_keys=True, allow_nan=False).encode()
    _require(len(encoded) <= CONTRACT["maximum_output_bytes"], "output byte bound exceeded")
    _check(deadline)
    return result
