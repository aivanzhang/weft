"""Source-bound nominal floor calibration; measurement claims are not physical validation."""

from collections import Counter
from dataclasses import dataclass
from hashlib import sha256
import json
import math
import re
import time

import numpy as np

from weft.camera_ground import validate_ground
from weft.camera_pose import _undistort, validate_camera
from weft.io import canonical_hash
from weft.source_surface_anchors import _document, _sources
from weft.world_alignment import _matrix, load_world_alignment


PROFILE = "floor-target-calibration-result-v1"
REQUEST_PROFILE = "floor-target-calibration-request-v1"
MEASUREMENT_PROFILE = "floor-target-measurements-v1"
LIMITS = {"maximum_seconds": 5., "maximum_input_bytes": 4 << 20,
    "maximum_request_bytes": 64 << 10, "maximum_observation_documents": 8,
    "maximum_measurement_documents": 8, "maximum_source_frames": 1800,
    "maximum_corners": 64, "maximum_construction_frames": 8,
    "maximum_evaluation_frames": 8, "maximum_solver_calls": 1,
    "maximum_alternatives": 2, "maximum_point_reprojections": 1024,
    "maximum_condition": 1e6, "maximum_rms_px": 1., "maximum_error_px": 2.,
    "minimum_depth_m": .2, "maximum_depth_m": 100.,
    "duplicate_rotation_max_abs": 1e-9, "duplicate_translation_max_abs_m": 1e-9,
    "maximum_result_bytes": 1 << 20, "maximum_output_bytes": 4 << 20,
    "maximum_output_files": 8}
_UNCERTAINTY = {
    "camera_calibration": {"intrinsics_px", "radial_k1", "translation_m", "rotation_rad"},
    "planar_target": {"point_position", "scale_relative"},
    "target_to_world": {"translation_m", "rotation_rad"},
    "floor_plane": {"height_m", "normal_rad"}, "target_corners": {"pixel_error_note"}}
_KINDS = {"camera": "camera_calibration", "target": "planar_target",
          "target_to_world": "target_to_world", "floor": "floor_plane"}


class _Unavailable(Exception):
    pass


def _require(condition, message):
    if not condition:
        raise ValueError("Floor target calibration: " + message)


def _check(deadline):
    if time.monotonic() >= deadline:
        raise TimeoutError("Floor target calibration exceeded its cooperative deadline")


def _fields(value, names, label):
    _require(type(value) is dict and set(value) == set(names), label + " fields differ")


def _text(value, limit=240):
    return type(value) is str and 0 < len(value.strip()) <= limit


def _number(value, low, high):
    return type(value) in (int, float) and math.isfinite(value) and low <= value <= high


def _sha(value):
    return type(value) is str and re.fullmatch("[0-9a-f]{64}", value) is not None


def _encode(value, bound, deadline):
    _check(deadline)
    raw = (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()
    _require(len(raw) <= bound, "encoded output exceeds its byte bound; no rows dropped")
    _check(deadline)
    return raw


def _document_map(documents, maximum, label):
    from collections.abc import Mapping
    _require(isinstance(documents, Mapping) and len(documents) <= maximum, label + " document count exceeds bound")
    values = dict(documents)
    _require(all(_sha(key) and type(raw) is bytes for key, raw in values.items()), label + " requires raw SHA256-to-bytes mapping")
    return values


def _reference(value):
    _fields(value, ("document_sha256", "record_id"), "measurement reference")
    _require(_sha(value["document_sha256"]) and _text(value["record_id"], 160), "invalid measurement reference")
    return value["document_sha256"], value["record_id"]


def _evidence_refs(refs):
    _require(type(refs) is list and len(refs) <= 32, "bounded evidence references required")
    keys = []
    for ref in refs:
        _fields(ref, ("document_sha256", "evidence_id"), "evidence reference")
        key = ref["document_sha256"], ref["evidence_id"]
        _require(_sha(key[0]) and _text(key[1]) and key not in keys, "invalid or duplicate evidence reference")
        keys.append(key)
    return keys


def _measured(record, evidence, *, nullable=()):
    if record["quality"] != "measured":
        raise _Unavailable("measurement_quality_not_declared_measured")
    refs = _evidence_refs(record["evidence_refs"])
    if not refs or any(key not in evidence or evidence[key]["status"] != "observed" for key in refs):
        raise _Unavailable("missing_qualified_observed_measurement_support")
    uncertainty = record["uncertainty"]
    if uncertainty["interpretation"] == "unknown" or any(
            uncertainty[key] is None for key in _UNCERTAINTY[record["kind"]] - set(nullable)):
        raise _Unavailable("unknown_required_measurement_uncertainty")


def _records(documents, source_sha256, deadline):
    result, headers = {}, []
    for digest, raw in sorted(documents.items()):
        _check(deadline)
        _require(sha256(raw).hexdigest() == digest, "measurement raw hash differs")
        doc = _document(raw, deadline)
        _fields(doc, ("schema_version", "profile", "source_sha256", "records"), "measurement document")
        _require(type(doc["schema_version"]) is int and doc["schema_version"] == 1
                 and doc["profile"] == MEASUREMENT_PROFILE and doc["source_sha256"] == source_sha256,
                 "measurement source/profile differs")
        _require(type(doc["records"]) is list and len(result) + len(doc["records"]) <= 1804, "too many measurement records")
        for record in doc["records"]:
            _check(deadline)
            _fields(record, ("id", "kind", "quality", "method", "provenance", "uncertainty", "evidence_refs", "value"), "measurement record")
            key = digest, record["id"] if type(record["id"]) is str else None
            _require(_text(record["id"], 160) and key not in result and type(record["kind"]) is str
                     and record["kind"] in _UNCERTAINTY and record["quality"] in ("measured", "authored", "unknown")
                     and _text(record["method"], 1000) and _text(record["provenance"], 2000), "invalid measurement header")
            uncertainty = record["uncertainty"]
            _fields(uncertainty, _UNCERTAINTY[record["kind"]] | {"interpretation"}, "uncertainty")
            _require(uncertainty["interpretation"] in ("declared_absolute_bound", "standard_deviation", "unknown"), "unknown uncertainty interpretation")
            for name in _UNCERTAINTY[record["kind"]]:
                value = uncertainty[name]
                _require(value is None or (_text(value, 2000) if name == "pixel_error_note" else _number(value, 0, 1e6)), "invalid uncertainty value")
            _evidence_refs(record["evidence_refs"])
            _require(type(record["value"]) is dict, "measurement value object required")
            result[key] = record
            units = record["value"].get("units") if record["kind"] == "planar_target" else None
            headers.append({"document_sha256": digest, "target_units": units if units in ("m", "mm") else None, **{name: record[name] for name in
                ("id", "kind", "quality", "method", "provenance", "uncertainty", "evidence_refs")}})
    return result, headers


def _request(raw, source_sha256, observations, measurements, supplied, source, deadline):
    request = _document(raw, deadline)
    _fields(request, ("schema_version", "profile", "source_sha256", "mode", "observation_sha256s", "measurement_sha256s",
        "supplied_camera_sha256", "supplied_ground_sha256", "world_frame", "anchor_frame", "check_frames",
        "evaluation_frames", "fixed_interval", "measurement_refs"), "request")
    _require(type(request["schema_version"]) is int and request["schema_version"] == 1
             and request["profile"] == REQUEST_PROFILE and request["source_sha256"] == source_sha256,
             "request profile/source differs")
    _require(request["mode"] in ("supplied", "planar_target"), "unknown calibration mode")
    for name, documents in (("observation_sha256s", observations), ("measurement_sha256s", measurements)):
        pins = request[name]
        _require(type(pins) is list and all(_sha(pin) for pin in pins) and len(set(pins)) == len(pins)
                 and set(pins) == set(documents), "request raw document inventory differs")
    for name, value in zip(("supplied_camera_sha256", "supplied_ground_sha256"), supplied):
        _require(request[name] == (None if value is None else sha256(value).hexdigest()), "supplied raw input pin differs")
    world = request["world_frame"]
    _fields(world, ("id", "length_unit", "axes"), "world frame")
    _require(_text(world["id"], 160) and world["length_unit"] == "m" and world["axes"] == "x_right_y_away_z_up", "explicit metre z-up world required")
    _fields(request["measurement_refs"], _KINDS, "selected measurements")
    for ref in request["measurement_refs"].values():
        if ref is not None:
            _reference(ref)
    for name, bound in (("check_frames", 7), ("evaluation_frames", 8)):
        frames = request[name]
        _require(type(frames) is list and len(frames) <= bound and all(type(f) is int and 0 <= f < source["retained_frames"] for f in frames)
                 and len(set(frames)) == len(frames), "invalid frame selection")
    anchor = request["anchor_frame"]
    _require(anchor is None or type(anchor) is int and 0 <= anchor < source["retained_frames"], "invalid anchor frame")
    selected = ([] if anchor is None else [anchor]) + request["check_frames"] + request["evaluation_frames"]
    _require(len(set(selected)) == len(selected), "construction/evaluation frames overlap")
    if request["mode"] == "supplied":
        _require(anchor is None and not selected and request["fixed_interval"] is None
                 and request["measurement_refs"]["target"] is None and request["measurement_refs"]["target_to_world"] is None,
                 "supplied mode cannot select target fitting")
    else:
        _require(all(value is None for value in supplied), "planar mode cannot override supplied sidecars")
    return request


def _selected(request, records, name):
    ref = request["measurement_refs"][name]
    if ref is None or _reference(ref) not in records:
        raise _Unavailable("missing_" + name + "_measurement")
    record = records[_reference(ref)]
    _require(record["kind"] == _KINDS[name], "selected measurement kind differs")
    return record


def _frame_headers(records, request, source, observations, deadline):
    by_frame = {}
    for key, record in records.items():
        if record["kind"] != "target_corners":
            continue
        value = record["value"]
        _fields(value, ("frame", "nominal_time_s", "source_time_s", "observation_ref", "partition", "status", "points"), "corner frame")
        frame = value["frame"]
        _require(type(frame) is int and 0 <= frame < source["retained_frames"] and frame not in by_frame, "duplicate/out-of-range corner frame")
        _require(value["status"] in ("observed", "missing", "occluded"), "invalid corner-frame status")
        ref = value["observation_ref"]
        _fields(ref, ("document_sha256", "object_id", "observation_id"), "corner observation")
        _require(_sha(ref["document_sha256"]) and _text(ref["object_id"], 160) and _text(ref["observation_id"]), "invalid corner observation identity")
        _require(_number(value["nominal_time_s"], 0, 60) and _number(value["source_time_s"], 0, 60)
                 and type(value["partition"]) is str and len(value["partition"]) <= 80, "invalid corner clock/partition")
        by_frame[frame] = key, record
    frames = []
    for frame in range(source["retained_frames"]):
        _check(deadline)
        role = ("anchor" if frame == request["anchor_frame"] else "construction_check" if frame in request["check_frames"]
                else "evaluation_not_used" if frame in request["evaluation_frames"] else "not_selected")
        row = {"source_frame": frame, "source_status": "decoded" if frame < source["decoded_frames"] else "metadata_only_not_decoded",
            "nominal_time_s": None if source["fps"] is None else frame / source["fps"], "role": role, "partition": None, "observation_id": None,
            "source_time_s": source["native_frame_times_s"][frame] if source["native_frame_times_s"] is not None else None,
            "measurement_ref": None, "status": "unavailable", "reason": "missing_corner_measurement", "points": []}
        if frame in by_frame:
            key, record = by_frame[frame]; value = record["value"]; ref = value["observation_ref"]
            observation = observations.get((ref["document_sha256"], ref["object_id"], frame))
            row.update(measurement_ref={"document_sha256": key[0], "record_id": key[1]}, partition=value["partition"],
                       observation_id=ref["observation_id"], source_time_s=value["source_time_s"], status=record["value"]["status"])
            if row["source_status"] != "decoded":
                reason = "metadata_only_not_decoded"
            elif observation is None or observation["id"] != ref["observation_id"]:
                reason = "missing_exact_source_observation"
            elif source["fps"] is None:
                reason = "missing_source_nominal_clock"
            elif value["partition"] != observation["partition"] or not math.isclose(value["source_time_s"], observation["time_s"], abs_tol=1e-9, rel_tol=0) or not math.isclose(value["nominal_time_s"], row["nominal_time_s"], abs_tol=1e-9, rel_tol=0):
                reason = "source_clock_or_partition_mismatch"
            elif observation["admission"] != "included":
                reason = "excluded_source_partition"
            elif role not in ("anchor", "construction_check"):
                reason = "evaluation_not_used" if role == "evaluation_not_used" else "not_selected"
            elif observation["visibility"] not in ("detected", "partial"):
                reason = "source_target_unavailable"
            elif value["status"] != "observed":
                reason = "corner_frame_" + value["status"]
            else:
                reason = None
            row["reason"] = reason
        frames.append(row)
    return frames, by_frame


def _camera(source, K, k1, rotation, translation, provenance):
    return {"schema_version": 1, "source_sha256": source["sha256"], "width": source["width"], "height": source["height"],
        "fps": source["fps"], "frame_count": source["metadata_frames"], "world_axes": "right_handed_z_up", "units": "m",
        "provenance": provenance, "quality": "reconstructed", "K": K, "radial_k1": k1,
        "frames": [{"frame": f, "time_s": f/source["fps"], "R": rotation, "t": translation} for f in range(source["metadata_frames"])]}


def _camera_support(record, request, source, evidence, supplied_raw):
    planar = request["mode"] == "planar_target"
    _measured(record, evidence, nullable=("translation_m", "rotation_rad") if planar else ())
    value = record["value"]
    _fields(value, ("source_sha256", "units", "width", "height", "fps", "metadata_frames", "decoded_frames", "K", "distortion", "world_frame_id", "camera_sha256"), "camera measurement")
    if any(value[name] != source[name] for name in ("width", "height", "fps", "metadata_frames", "decoded_frames")) or value["source_sha256"] != source["sha256"] or value["units"] != "m" or value["world_frame_id"] != request["world_frame"]["id"]:
        raise _Unavailable("camera_measurement_source_world_clock_mismatch")
    _require(all(type(value[name]) is int for name in ("width", "height", "metadata_frames", "decoded_frames"))
             and _number(value["fps"], 1, 240), "camera counts must be exact integers and FPS numeric")
    expected = None if supplied_raw is None else sha256(supplied_raw).hexdigest()
    if value["camera_sha256"] != expected:
        raise _Unavailable("camera_measurement_raw_pin_mismatch")
    distortion = value["distortion"]
    if type(distortion) is not dict or set(distortion) != {"model", "k1"} or distortion["model"] != "pinhole_k1":
        raise _Unavailable("unsupported_distortion_model")
    provisional = _camera(source, value["K"], distortion["k1"], np.eye(3).tolist(), [0., 0., 1.], "Validation-only supplied intrinsic envelope")
    matrix, radial, _ = validate_camera(provisional, {"source_sha256": source["sha256"], "width": source["width"], "height": source["height"], "fps": source["fps"], "frame_count": source["metadata_frames"]})
    return value["K"], matrix, radial


def _floor_support(record, request, evidence):
    _measured(record, evidence)
    value = record["value"]
    _fields(value, ("world_frame_id", "normal_world", "height_m", "origin_convention", "target_to_world_ref"), "floor measurement")
    if value["world_frame_id"] != request["world_frame"]["id"] or value["target_to_world_ref"] != request["measurement_refs"]["target_to_world"]:
        raise _Unavailable("floor_world_relation_mismatch")
    if (type(value["normal_world"]) is not list or len(value["normal_world"]) != 3
            or not all(_number(v, -1, 1) for v in value["normal_world"]) or value["normal_world"] != [0, 0, 1]):
        raise _Unavailable("nonhorizontal_floor_unsupported")
    if value["height_m"] is None or not _text(value["origin_convention"], 1000):
        raise _Unavailable("missing_floor_height_or_origin_convention")
    _require(_number(value["height_m"], -1000, 1000), "invalid floor height")
    return value["height_m"]


def _rank(points):
    centered = points - points.mean(axis=0)
    scale = float(np.linalg.norm(centered))
    if scale <= 1e-12:
        return None
    singular = np.linalg.svd(centered / scale, compute_uv=False)
    if len(singular) != 2 or singular[1] <= 0:
        return None
    condition = float(singular[0]/singular[1])
    return condition if math.isfinite(condition) and condition <= LIMITS["maximum_condition"] else None


def _target_support(record, relation, request, evidence, allowed_ids):
    _measured(record, evidence); _measured(relation, evidence)
    value = record["value"]
    _fields(value, ("id", "frame_id", "units", "entity_ref", "corners"), "target measurement")
    _require(_text(value["id"], 160) and _text(value["frame_id"], 160), "target identities required")
    if value["units"] not in ("m", "mm"):
        raise _Unavailable("unknown_target_units")
    _fields(value["entity_ref"], ("document_sha256", "object_id"), "target entity")
    _require(_sha(value["entity_ref"]["document_sha256"]) and _text(value["entity_ref"]["object_id"], 160), "invalid target entity")
    corners = value["corners"]
    if type(corners) is not list or not 4 <= len(corners) <= 64:
        raise _Unavailable("insufficient_or_excessive_target_corners")
    points = {}; factor = 1 if value["units"] == "m" else .001
    for corner in corners:
        _fields(corner, ("id", "xy"), "target corner")
        _require(_text(corner["id"], 160) and corner["id"] not in points and type(corner["xy"]) is list and len(corner["xy"]) == 2
                 and all(_number(x, -1e6, 1e6) for x in corner["xy"]), "invalid/duplicate target corner")
        points[corner["id"]] = np.asarray(corner["xy"], dtype=float)*factor
    values = np.asarray(list(points.values()))
    if np.max(np.abs(values)) > 1000 or len({tuple(point) for point in values}) != len(values) or _rank(values) is None:
        raise _Unavailable("degenerate_or_ill_conditioned_target")
    mapping = load_world_alignment(relation["value"], evidence_ids_by_document=allowed_ids)
    doc = mapping.document()
    if (not mapping.ready or doc["from_frame"] != {"id": value["frame_id"], "length_unit": "m", "axes": "right_handed_xyz"}
            or doc["to_frame"] != request["world_frame"] or doc["spatial"]["scale"] != 1
            or doc["clock"] != {"rate": 1, "source_origin_s": 0, "target_origin_s": 0}
            or doc["claims"]["spatial"]["status"] != "observed"):
        raise _Unavailable("unknown_or_incompatible_target_world_mapping")
    translation = np.asarray(doc["spatial"]["translation"])
    if np.max(np.abs(translation)) > 1000:
        raise _Unavailable("target_translation_outside_envelope")
    return points, value["entity_ref"], _matrix(doc["spatial"]["rotation_wxyz"]), translation


def _fixed(request, source, evidence):
    value = request["fixed_interval"]
    if value is None:
        raise _Unavailable("missing_fixed_camera_target_declaration")
    _fields(value, ("first_frame", "last_frame", "camera_static", "target_static", "method", "uncertainty", "evidence_refs"), "fixed interval")
    if (type(value["first_frame"]) is not int or value["first_frame"] != 0 or type(value["last_frame"]) is not int
            or value["last_frame"] != source["retained_frames"]-1 or value["camera_static"] is not True or value["target_static"] is not True
            or not _text(value["method"], 1000) or not _text(value["uncertainty"], 2000)):
        raise _Unavailable("incomplete_fixed_camera_target_declaration")
    refs = _evidence_refs(value["evidence_refs"])
    if not refs or any(ref not in evidence for ref in refs):
        raise _Unavailable("fixed_interval_evidence_unavailable")


def _corner_payload(record, points, entity, observation, matrix, radial, source, evidence, frame_row):
    _measured(record, evidence)
    value = record["value"]; ref = value["observation_ref"]
    if {name: ref[name] for name in ("document_sha256", "object_id")} != entity:
        raise _Unavailable("target_entity_mismatch")
    if not set(_evidence_refs(record["evidence_refs"])).intersection(_evidence_refs(observation["evidence_refs"])):
        raise _Unavailable("corner_measurement_observation_evidence_mismatch")
    supplied = value["points"]
    _require(type(supplied) is list and len(supplied) <= len(points), "bounded named corner points required")
    known, headers, seen = [], [], set()
    for point in supplied:
        _fields(point, ("corner_id", "status", "pixel", "uncertainty_px", "reason"), "image corner")
        name = point["corner_id"]
        _require(type(name) is str and name in points and name not in seen and point["status"] in ("observed", "missing", "occluded"), "invalid/duplicate image corner")
        seen.add(name)
        _require(point["reason"] is None or _text(point["reason"], 1000), "invalid corner reason")
        headers.append({key: point[key] for key in ("corner_id", "status", "reason")})
    headers.extend({"corner_id": name, "status": "missing", "reason": "not_supplied"} for name in points if name not in seen)
    frame_row["points"] = headers
    for point in supplied:
        if point["status"] != "observed":
            _require(point["pixel"] is None, "missing/occluded corner must not supply a coordinate")
            continue
        pixel = point["pixel"]
        _require(type(pixel) is list and len(pixel) == 2 and _number(pixel[0], 0, source["width"])
                 and _number(pixel[1], 0, source["height"]), "image corner outside original pixel domain")
        if point["uncertainty_px"] is None:
            raise _Unavailable("unknown_pixel_uncertainty")
        _require(_number(point["uncertainty_px"], 0, 64), "invalid pixel uncertainty")
        known.append(point)
    if len(known) < 4 or _rank(np.asarray([points[p["corner_id"]] for p in known])) is None:
        raise _Unavailable("insufficient_or_degenerate_observed_corners")
    pixels = np.asarray([p["pixel"] for p in known], dtype=float)
    rays = _undistort(pixels, matrix, radial)
    if rays is None:
        raise _Unavailable("distortion_inverse_unavailable")
    if _rank(rays) is None:
        raise _Unavailable("degenerate_or_ill_conditioned_image_rays")
    xyz = np.asarray([[*points[p["corner_id"]], 0.] for p in known], dtype=float)
    return known, headers, xyz, rays


def _solve_ippe(points, normalized):
    import cv2
    solved = cv2.solvePnPGeneric(points, normalized, np.eye(3), None, flags=cv2.SOLVEPNP_IPPE)
    _require(len(solved[1]) == len(solved[2]) <= 2, "IPPE alternative count exceeds bound")
    result = []
    for rotation, translation in zip(solved[1], solved[2]):
        if not np.isfinite(rotation).all() or not np.isfinite(translation).all():
            result.append({"R_ct": None, "t_ct": None, "solver_error": "nonfinite_solver_pose"})
        else:
            result.append({"R_ct": cv2.Rodrigues(rotation)[0].tolist(), "t_ct": translation.reshape(-1).tolist(), "solver_error": None})
    return result


def _alternative(index, supplied, prepared, matrix, radial, rotation_wt, translation_wt, work, deadline):
    row = {"index": index, "R_ct": None, "t_ct": None, "R_cw": None, "t_cw": None, "duplicate_of": None,
           "status": "rejected", "reasons": [], "checks": {"proper_rotation": False, "depth": False, "projection_domain": False,
               "construction_residual": False, "rms_px": None, "maximum_px": None}, "projections": []}
    if supplied.get("solver_error") is not None:
        row["reasons"].append("invalid_solver_pose"); return row
    try:
        rotation = np.asarray(supplied["R_ct"], dtype=float); translation = np.asarray(supplied["t_ct"], dtype=float)
    except (ValueError, TypeError, KeyError):
        row["reasons"].append("invalid_solver_pose"); return row
    if rotation.shape != (3, 3) or translation.shape != (3,) or not np.isfinite(rotation).all() or not np.isfinite(translation).all():
        row["reasons"].append("nonfinite_or_malformed_solver_pose"); return row
    row.update(R_ct=rotation.tolist(), t_ct=translation.tolist())
    if (np.max(np.abs(rotation)) > 1.00001 or not np.allclose(rotation @ rotation.T, np.eye(3), atol=1e-5, rtol=0)
            or abs(float(np.linalg.det(rotation))-1) > 1e-5 or np.max(np.abs(translation)) > 1000):
        row["reasons"].append("improper_rotation_or_translation"); return row
    world_rotation = rotation @ rotation_wt.T; world_translation = translation - world_rotation @ translation_wt
    row.update(R_cw=world_rotation.tolist(), t_cw=world_translation.tolist())
    row["checks"]["proper_rotation"] = True
    depth_ok, domain_ok, residual_ok, errors = True, True, True, []
    for frame, known, xyz in prepared:
        _check(deadline)
        work["point_reprojections"] += len(known)
        _require(work["point_reprojections"] <= 1024, "point reprojection bound exceeded")
        pc = xyz @ rotation.T + translation
        projection = {"source_frame": frame["source_frame"], "measurement_ref": frame["measurement_ref"], "points": [], "rms_px": None, "maximum_px": None}
        frame_errors = []
        for point, position in zip(known, pc):
            z = float(position[2]); valid_depth = .2 <= z <= 100
            predicted, residual, error, valid_domain = None, None, None, False
            if z > 0 and math.isfinite(z):
                normalized = position[:2]/z; radius2 = float(normalized @ normalized)
                distorted = normalized*(1 + radial*radius2)
                if np.isfinite(distorted).all() and np.linalg.norm(distorted) <= 4 and 1+3*radial*radius2 > 1e-8:
                    uv = matrix @ np.array([*distorted, 1.])
                    if np.isfinite(uv).all():
                        predicted = uv[:2].tolist(); delta = uv[:2]-point["pixel"]
                        residual = delta.tolist(); error = float(np.linalg.norm(delta)); valid_domain = True
                        frame_errors.append(error); errors.append(error)
            depth_ok &= valid_depth; domain_ok &= valid_domain
            projection["points"].append({"corner_id": point["corner_id"], "pixel": point["pixel"], "uncertainty_px": point["uncertainty_px"],
                "predicted_pixel": predicted, "residual_px": residual, "error_px": error, "depth_m": z,
                "status": "projected" if valid_domain and valid_depth else "invalid_depth_or_domain"})
        if len(frame_errors) == len(known):
            projection["rms_px"] = math.sqrt(sum(e*e for e in frame_errors)/len(frame_errors))
            projection["maximum_px"] = max(frame_errors)
            residual_ok &= projection["rms_px"] <= 1 and projection["maximum_px"] <= 2
        else:
            residual_ok = False
        row["projections"].append(projection)
    row["checks"].update(depth=bool(depth_ok), projection_domain=bool(domain_ok), construction_residual=bool(residual_ok),
        rms_px=math.sqrt(sum(e*e for e in errors)/len(errors)) if errors else None, maximum_px=max(errors) if errors else None)
    row["reasons"] = [name for name in ("depth", "projection_domain", "construction_residual") if not row["checks"][name]]
    if np.max(np.abs(world_translation)) > 1000:
        row["reasons"].append("world_translation_outside_envelope")
    if not row["reasons"]:
        row["status"] = "nominal_checks_passed"
    return row


@dataclass(frozen=True)
class FloorTargetCalibration:
    _files: tuple

    def describe(self):
        return json.loads(dict(self._files)["calibration.json"])

    def files(self):
        return dict(self._files)


def prepare_floor_target_calibration(observation_documents, *, source_sha256, request_document,
        measurement_documents, supplied_camera_document=None, supplied_ground_document=None,
        max_seconds=5) -> FloorTargetCalibration:
    """Prepare a nominal camera/floor pair from explicit source-bound calibration evidence."""
    _require(_number(max_seconds, .001, 5), "cooperative time bound must be in [.001,5]")
    deadline = time.monotonic() + max_seconds
    observations_raw = _document_map(observation_documents, 8, "observation")
    _require(bool(observations_raw), "at least one observation document required")
    measurements_raw = _document_map(measurement_documents, 8, "measurement")
    supplied = supplied_camera_document, supplied_ground_document
    _require(_sha(source_sha256) and type(request_document) is bytes and len(request_document) <= 64 << 10
             and all(raw is None or type(raw) is bytes for raw in supplied), "bounded raw inputs required")
    input_bytes = len(request_document) + sum(map(len, observations_raw.values())) + sum(map(len, measurements_raw.values())) + sum(len(r) for r in supplied if r is not None)
    _require(input_bytes <= 4 << 20, "combined input exceeds 4 MiB")
    original, documents, observations, evidence, allowed_ids, _ = _sources(observations_raw, source_sha256, deadline)
    source = {"sha256": source_sha256, "width": original["width_px"], "height": original["height_px"], "fps": original.get("fps"),
        "native_frame_times_s": original.get("frame_times_s"),
        "decoded_frames": original["decoded_frames"], "metadata_frames": original["metadata_frames"],
        "retained_frames": max(original["decoded_frames"], original["metadata_frames"]), "time": {key: value["time"] for key, value in documents.items()}}
    _require(1 <= source["retained_frames"] <= 1800 and (source["fps"] is None or source["retained_frames"]/source["fps"] <= 60), "complete source exceeds frame/duration bound")
    request = _request(request_document, source_sha256, observations_raw, measurements_raw, supplied, source, deadline)
    records, headers = _records(measurements_raw, source_sha256, deadline)
    frames, corner_records = _frame_headers(records, request, source, observations, deadline)
    work = {"input_bytes": input_bytes, "solver_calls": 0, "undistorted_points": 0, "point_reprojections": 0, "prepared_construction_frames": 0}
    result = {"schema_version": 1, "profile": PROFILE, "status": "unavailable", "mode": request["mode"], "reason": None,
        "input_bindings": {"source_sha256": source_sha256, "request_sha256": sha256(request_document).hexdigest(),
            "observation_document_sha256s": sorted(observations_raw), "measurement_document_sha256s": sorted(measurements_raw),
            "supplied_camera_sha256": request["supplied_camera_sha256"], "supplied_ground_sha256": request["supplied_ground_sha256"]},
        "source": source, "request": request, "measurement_records": headers, "frames": frames, "alternatives": [], "selected_index": None,
        "artifact_bindings": None, "counts": {}, "work": work, "limits": LIMITS,
        "uncertainty": {"propagation": "not_propagated", "interpretation": "Nominal consistency only; declared bounds or standard deviations are not propagated or certified",
                        "measurement_authenticity": "unverified"},
        "claims": {"physical_validation": False, "contact": "unknown", "grasp": "unknown", "load": "unknown",
                   "geometry_modified": False, "body_or_floor_installed": False, "evaluation_used": False,
                   "source_measurement_authentication": False}}
    artifacts = {}; camera = None; ground = None
    try:
        if not _number(source["fps"], 1, 240):
            raise _Unavailable("missing_source_nominal_clock")
        if source["decoded_frames"] > source["metadata_frames"]:
            raise _Unavailable("decoded_slots_exceed_camera_metadata_clock")
        camera_record = _selected(request, records, "camera"); floor_record = _selected(request, records, "floor")
        K, matrix, radial = _camera_support(camera_record, request, source, evidence, supplied_camera_document)
        height = _floor_support(floor_record, request, evidence)
        if request["mode"] == "supplied":
            if any(raw is None for raw in supplied):
                raise _Unavailable("missing_supplied_camera_or_ground")
            camera, ground = [_document(raw, deadline) for raw in supplied]
            validate_camera(camera, {"source_sha256": source_sha256, "width": source["width"], "height": source["height"], "fps": source["fps"], "frame_count": source["metadata_frames"]})
            validate_ground(ground, {"source_sha256": source_sha256, "calibration_sha256": canonical_hash(camera)})
            if camera["K"] != K or camera["radial_k1"] != radial or ground["height_m"] != height:
                raise _Unavailable("supplied_geometry_measurement_mismatch")
            artifacts.update({"camera.json": supplied_camera_document, "ground.json": supplied_ground_document})
            result.update(status="qualified_nominal", reason="supplied_exact_nominal_pair_not_authenticated")
        else:
            if request["anchor_frame"] is None:
                raise _Unavailable("missing_anchor_frame")
            _fixed(request, source, evidence)
            target = _selected(request, records, "target"); relation = _selected(request, records, "target_to_world")
            points, entity, rotation_wt, translation_wt = _target_support(target, relation, request, evidence, allowed_ids)
            prepared, anchor_xyz, anchor_rays = [], None, None
            for frame_id in [request["anchor_frame"], *request["check_frames"]]:
                _check(deadline)
                frame = frames[frame_id]
                if frame["reason"] is not None:
                    raise _Unavailable("required_construction_frame_" + frame["reason"])
                _, record = corner_records[frame_id]; ref = record["value"]["observation_ref"]
                observation = observations[(ref["document_sha256"], ref["object_id"], frame_id)]
                known, point_headers, xyz, rays = _corner_payload(record, points, entity, observation, matrix, radial, source, evidence, frame)
                frame.update(status="prepared", reason=None, points=point_headers)
                prepared.append((frame, known, xyz)); work["undistorted_points"] += len(known); work["prepared_construction_frames"] += 1
                if frame_id == request["anchor_frame"]:
                    anchor_xyz, anchor_rays = xyz, rays
            _check(deadline); work["solver_calls"] = 1
            alternatives = _solve_ippe(anchor_xyz, anchor_rays)
            _check(deadline)
            _require(type(alternatives) is list and len(alternatives) <= 2, "solver alternative bound exceeded")
            result["alternatives"] = [_alternative(index, alternative, prepared, matrix, radial, rotation_wt, translation_wt, work, deadline)
                                      for index, alternative in enumerate(alternatives)]
            distinct = []
            for row in result["alternatives"]:
                if row["status"] != "nominal_checks_passed":
                    continue
                for earlier in distinct:
                    if (np.max(np.abs(np.asarray(row["R_ct"])-earlier["R_ct"])) <= 1e-9
                            and np.max(np.abs(np.asarray(row["t_ct"])-earlier["t_ct"])) <= 1e-9):
                        row["duplicate_of"] = earlier["index"]; break
                if row["duplicate_of"] is None:
                    distinct.append(row)
            if len(distinct) == 1:
                selected = distinct[0]
                camera = _camera(source, K, radial, selected["R_cw"], selected["t_cw"], "Nominal IPPE target calibration under supplied measurement claims; fixed-camera repetition is declared, not per-frame measured")
                validate_camera(camera, {"source_sha256": source_sha256, "width": source["width"], "height": source["height"], "fps": source["fps"], "frame_count": source["metadata_frames"]})
                ground = {"schema_version": 1, "source_sha256": source_sha256, "calibration_sha256": canonical_hash(camera),
                    "world_axes": "right_handed_z_up", "units": "m", "height_m": height, "quality": "reconstructed",
                    "provenance": "Supplied measured-floor claim in target-aligned world; nominal calibration uncertainty is not propagated"}
                validate_ground(ground, {"source_sha256": source_sha256, "calibration_sha256": canonical_hash(camera)})
                artifacts = {"camera.json": _encode(camera, 4 << 20, deadline), "ground.json": _encode(ground, 4 << 20, deadline)}
                result.update(status="qualified_nominal", selected_index=selected["index"], reason="one_distinct_nominal_solution_under_fixed_checks")
            else:
                result.update(status="ambiguous" if distinct else "rejected", reason="multiple_distinct_nominal_solutions" if distinct else "no_admissible_nominal_solution")
    except _Unavailable as error:
        result["reason"] = str(error)
    if artifacts:
        result["artifact_bindings"] = {"camera_raw_sha256": sha256(artifacts["camera.json"]).hexdigest(), "camera_canonical_sha256": canonical_hash(camera),
            "ground_raw_sha256": sha256(artifacts["ground.json"]).hexdigest(), "qualities": {"camera": camera["quality"], "ground": ground["quality"]},
            "generation": "supplied_exact_bytes" if request["mode"] == "supplied" else "planar_nominal"}
    result["counts"] = {"source_slots": len(frames), "measurement_records": len(headers), "construction_frames_requested":
        (1 if request["anchor_frame"] is not None else 0) + len(request["check_frames"]), "evaluation_frames_not_used": len(request["evaluation_frames"]),
        "prepared_construction_frames": work["prepared_construction_frames"], "used_corner_observations": work["undistorted_points"],
        "alternatives": len(result["alternatives"]), "passing_alternatives": sum(r["status"] == "nominal_checks_passed" for r in result["alternatives"]),
        "distinct_passing_alternatives": sum(r["status"] == "nominal_checks_passed" and r["duplicate_of"] is None for r in result["alternatives"]),
        "frame_status_counts": dict(Counter(r["reason"] or r["status"] for r in frames))}
    artifacts["calibration.json"] = _encode(result, 1 << 20, deadline)
    _require(len(artifacts) <= 8 and sum(map(len, artifacts.values())) <= 4 << 20, "complete output exceeds bounds")
    _check(deadline)
    return FloorTargetCalibration(tuple(sorted(artifacts.items())))
