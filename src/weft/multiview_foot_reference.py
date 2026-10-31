"""Conditional foot triangles from saved calibrated points; no inference or installation."""

from collections import Counter
from copy import deepcopy
from hashlib import sha256
import json
import math
import time

import numpy as np

from weft.camera_body import validate_camera_binding
from weft.camera_ground import validate_ground
from weft.exercise_support_evidence import _json_size
from weft.foot_reference import FOOT_CONTRACT, LANDMARKS, _median
from weft.io import canonical_hash
from weft.multiview_body import _confidence, _points
from weft.multiview_pose import CONTRACT as RAY_CONTRACT, _observation, _view
from weft.multiview_root import METHOD as ROOT_METHOD
from weft.world_alignment import load_world_alignment


PROFILE = "calibrated-two-view-foot-reference-v1"
CONTRACT = {
    "maximum_input_bytes": 32 << 20, "maximum_output_bytes": 8 << 20,
    "maximum_seconds": 10., "maximum_source_frames": 1800, "maximum_pose_samples": 900,
    "construction_fraction": FOOT_CONTRACT["training_fraction"],
    "minimum_prefix_samples": FOOT_CONTRACT["minimum_prefix_samples"],
    "shape_bounds_source": "unchanged prefix-fixed-kinematic-foot-v1",
    "later_source": "excluded_no_geometry_evaluation", "interpolation": "none",
    "triangulation": "saved estimates only; no ray reconstruction repeated",
    "ground_use": "unclipped endpoint clearance only; never fitting or selection",
    "supported_body_binding": ROOT_METHOD,
}


def _require(value, message):
    if not value:
        raise ValueError(message)


def _check(deadline):
    if time.monotonic() >= deadline:
        raise TimeoutError("Multiview foot reference exceeded its cooperative deadline")


def _number(value, low, high):
    return type(value) in (int, float) and math.isfinite(value) and low <= value <= high


def _unique(pairs):
    result = {}
    for key, value in pairs:
        _require(key not in result, "Duplicate JSON field in multiview foot input")
        result[key] = value
    return result


def _constant(_value):
    raise ValueError("Multiview foot input must contain finite JSON")


def _documents(raws, deadline):
    _check(deadline)
    _require(all(raw is None or type(raw) is bytes for raw in raws.values())
             and sum(len(raw) for raw in raws.values() if raw is not None) <= CONTRACT["maximum_input_bytes"],
             "Multiview foot inputs require at most 32 MiB of exact JSON bytes")
    documents, pins, canonical_size = {}, {}, 0
    for name, raw in raws.items():
        _check(deadline)
        if raw is None:
            documents[name], pins[name] = None, None
            continue
        try:
            value = json.loads(raw.decode("utf-8"), object_pairs_hook=_unique, parse_constant=_constant)
        except (UnicodeError, ValueError, RecursionError) as error:
            raise ValueError("Invalid bounded UTF-8 JSON input: " + name) from error
        _require(type(value) is dict, "Multiview foot inputs must be JSON objects")
        if name in ("predictions", "evidence_1", "evidence_2", "camera_1", "camera_2"):
            maximum = CONTRACT["maximum_pose_samples"] if name.startswith("evidence_") else CONTRACT["maximum_source_frames"]
            _require(type(value.get("frames")) is list and len(value["frames"]) <= maximum,
                     "Source rows exceed their envelope before canonical hashing")
        canonical_size += _json_size(value, deadline)
        _require(canonical_size <= CONTRACT["maximum_input_bytes"], "Canonical inputs exceed 32 MiB")
        documents[name] = value
        pins[name] = {"sha256": sha256(raw).hexdigest(), "canonical_sha256": canonical_hash(value), "bytes": len(raw)}
        _check(deadline)
    return documents, pins


def _headers(evidence):
    _require(type(evidence) is dict and evidence.get("profile") == "exercise-pose-v1"
             and type(evidence.get("schema_version")) is int and evidence["schema_version"] == 1,
             "Original exercise-pose-v1 evidence required")
    _require(type(evidence.get("source_sha256")) is str and len(evidence["source_sha256"]) == 64
             and set(evidence["source_sha256"]) <= set("0123456789abcdef"), "Original source SHA256 required")
    _require(_number(evidence.get("fps"), 1, 240) and type(evidence.get("frame_count")) is int
             and 2 <= evidence["frame_count"] <= CONTRACT["maximum_source_frames"], "Bounded original source clock required")
    decoded = evidence.get("decoded_frames", evidence["frame_count"])
    _require(type(decoded) is int and 2 <= decoded <= CONTRACT["maximum_source_frames"]
             and (decoded - 1) / evidence["fps"] <= 60, "Decoded source slots exceed bounds")
    rows = evidence.get("frames")
    _require(type(rows) is list and 1 <= len(rows) <= CONTRACT["maximum_pose_samples"], "Each view needs 1–900 pose slots")
    poses, previous = {}, -1
    for row in rows:
        _require(type(row) is dict and type(row.get("frame")) is int and previous < row["frame"] < decoded,
                 "Pose source frames must be unique, ordered and decoded")
        frame = row["frame"]
        _require(row.get("id") == f"{evidence['source_sha256']}:frame:{frame}"
                 and all(_number(row.get(key), 0, 60) for key in ("time_s", "decoder_timestamp_s"))
                 and abs(row["time_s"] - frame / evidence["fps"]) <= 1e-7
                 and abs(row["time_s"] - row["decoder_timestamp_s"]) <= 1e-7,
                 "Original pose identities and nominal/decoder clocks differ")
        poses[frame], previous = row, frame
    return poses, decoded


def _source_envelope(data, pins, deadline):
    prediction = data["predictions"]
    _require(type(prediction) is dict and type(prediction.get("schema_version")) is int and prediction["schema_version"] == 1,
             "Saved two-view prediction schema required")
    _require(canonical_hash(prediction.get("contract")) == canonical_hash(dict(RAY_CONTRACT))
             and prediction.get("monocular_world_landmarks_used") is False
             and prediction.get("reference_body_or_contacts_used") is False
             and prediction.get("physical_validation") == "unverified", "Original two-ray contract and scope must be unchanged")
    count, rows, views = prediction.get("source_frames"), prediction.get("frames"), prediction.get("views")
    _require(type(count) is int and 2 <= count <= CONTRACT["maximum_source_frames"]
             and prediction.get("joint_slots") == 33 * count and type(prediction.get("joint_slots")) is int
             and type(rows) is list and len(rows) == count, "Complete source prediction denominator required")
    fields = {"name", "source_sha256", "camera_sha256", "evidence_sha256", "calibration_quality", "clock_provenance"}
    _require(type(views) is list and len(views) == 2 and all(type(view) is dict and set(view) == fields for view in views),
             "Exactly two original view bindings required")
    for view in views:
        _require(all(type(view[key]) is str and len(view[key]) == 64 and set(view[key]) <= set("0123456789abcdef")
                     for key in ("source_sha256", "camera_sha256", "evidence_sha256")), "View identities require SHA256 values")
        _require(type(view["name"]) is str and 1 <= len(view["name"]) <= 64
                 and type(view["clock_provenance"]) is str and 1 <= len(view["clock_provenance"].strip()) <= 2000
                 and view["calibration_quality"] in ("authored", "reconstructed", "measured"), "Explicit camera/clock quality required")
    _require(views[0]["name"] != views[1]["name"] and views[0]["source_sha256"] != views[1]["source_sha256"],
             "Two distinct sources and view names required")
    sources, decoded, reasons = [], [], []
    for index, view in enumerate(views, 1):
        evidence = data[f"evidence_{index}"]
        poses, total = _headers(evidence)
        _require(evidence.get("source_sha256") == view["source_sha256"] and evidence["frame_count"] == count,
                 "Source metadata differs from the retained prediction views")
        if pins[f"evidence_{index}"]["canonical_sha256"] != view["evidence_sha256"]:
            reasons.append(f"view_{index}_evidence_pin_mismatch")
        camera = data[f"camera_{index}"]
        if camera is None:
            reasons.append(f"missing_camera_{index}")
        elif pins[f"camera_{index}"]["canonical_sha256"] != view["camera_sha256"]:
            reasons.append(f"view_{index}_camera_pin_mismatch")
        else:
            try:
                _view({"name": view["name"], "evidence": evidence, "camera": camera,
                       "clock_provenance": view["clock_provenance"]})
                _require(camera["quality"] == view["calibration_quality"], "Camera quality differs")
            except (ValueError, KeyError, TypeError, IndexError):
                reasons.append(f"view_{index}_camera_source_binding_unavailable")
        sources.append(poses)
        decoded.append(total)
        _check(deadline)
    fps = data["evidence_1"]["fps"]
    _require(data["evidence_2"]["fps"] == fps, "Different source clocks cannot be silently aligned")
    endpoint_indices = tuple(index for side in LANDMARKS.values() for index in side.values())
    for index, row in enumerate(rows):
        _check(deadline)
        _require(type(row) is dict and row.get("frame") == index and type(row.get("frame")) is int
                 and _number(row.get("time_s"), 0, 60) and abs(row["time_s"] - index / fps) <= 1e-7
                 and row.get("source_evidence_ids") == [source[index]["id"] if index in source else None for source in sources]
                 and type(row.get("joints")) is list and len(row["joints"]) == 33,
                 "Predictions must retain exact source slots, identities and original times")
        for joint in endpoint_indices:
            point = row["joints"][joint]
            _require(type(point) is dict and type(point.get("status")) is str and 1 <= len(point["status"]) <= 100,
                     "Every foot endpoint needs its original triangulation status")
    return prediction, sources, decoded, fps, reasons


def _body_binding(data, pins, prediction, reasons):
    body, ground, mapping = data["body"], data["ground"], data["world_to_body"]
    for name, value in (("body", body), ("ground", ground), ("world_to_body", mapping)):
        if value is None:
            reasons.append("missing_" + name)
    if body is None or ground is None or mapping is None:
        return None, {}
    try:
        _require(all(type(body.get(name)) is dict for name in ("camera_reference", "body_frame", "mobility")),
                 "Optional body binding requires structured coordinate and placement fields")
        _require(body.get("profile") == "exercise-v3" and body.get("camera_reference", {}).get("method") == ROOT_METHOD,
                 "Only an original root-only camera-world body is supported")
        binding = validate_camera_binding(body)
        validate_ground(ground, binding)
        root = binding["root_refinement"]
        _require(canonical_hash(ground) == canonical_hash(binding["ground"])
                 and root["triangulation_sha256"] == pins["predictions"]["canonical_sha256"]
                 and root["views"] == prediction["views"] and root["source_frames"] == prediction["source_frames"]
                 and root["fps"] == data["evidence_1"]["fps"], "Body/ground differs from the original triangulation")
        alignment = load_world_alignment(mapping)
        expected_from = {"id": "camera-world:" + binding["calibration_sha256"], "length_unit": "m", "axes": "x_right_y_away_z_up"}
        expected_to = {"id": "body:" + pins["body"]["canonical_sha256"], "length_unit": "human_model_unit", "axes": "x_right_y_away_z_up"}
        origin = binding["origin_world_m"]
        _require(alignment.ready and mapping["from_frame"] == expected_from and mapping["to_frame"] == expected_to
                 and mapping["spatial"] == {"scale": 1., "rotation_wxyz": [1., 0., 0., 0.],
                                           "translation": [-origin[0], -origin[1], -ground["height_m"]]}
                 and mapping["clock"] == {"rate": 1., "source_origin_s": 0., "target_origin_s": 0.},
                 "Explicit world/body units, axes, origin and unchanged clock must match the bound body")
        return alignment, {row["frame"]: row for row in root["roots"]}
    except (ValueError, KeyError, TypeError, IndexError):
        reasons.append("incompatible_body_world_clock_binding")
        return None, {}


def _triangle(points):
    ankle, heel, toe = points
    vector = toe - heel
    length = float(np.linalg.norm(vector))
    _require(FOOT_CONTRACT["foot_length_bounds"][0] <= length <= FOOT_CONTRACT["foot_length_bounds"][1],
             "foot_length_out_of_bounds")
    axis = vector / length
    offset = ankle - heel
    along = float(offset @ axis)
    radial = offset - along * axis
    height = float(np.linalg.norm(radial))
    _require(FOOT_CONTRACT["ankle_axis_height_bounds"][0] <= height <= FOOT_CONTRACT["ankle_axis_height_bounds"][1],
             "ankle_axis_plane_degenerate_or_out_of_bounds")
    _require(FOOT_CONTRACT["ankle_along_axis_fraction_bounds"][0] <= along / length <= FOOT_CONTRACT["ankle_along_axis_fraction_bounds"][1],
             "ankle_attachment_out_of_bounds")
    return {"length": length, "along": along, "height": height, "axis": axis, "radial_direction": radial / height}


def _foot(row, side, sources, cameras, alignment, ground, deadline):
    _check(deadline)
    indices = tuple(LANDMARKS[side].values())
    points = _points(row, indices)
    if points is None:
        return None, "triangulated_endpoint_unavailable"
    for index in indices:
        point = row["joints"][index]
        _require(_number(point.get("ray_angle_degrees"), RAY_CONTRACT["minimum_ray_angle_degrees"], 90.)
                 and _number(point.get("ray_separation_m"), 0, 4000)
                 and type(point.get("reprojection_px")) is list and len(point["reprojection_px"]) == 2
                 and all(_number(error, 0, RAY_CONTRACT["maximum_reprojection_fraction"] * max(camera["width"], camera["height"]))
                         for error, camera in zip(point["reprojection_px"], cameras, strict=True)),
                 "Saved estimated endpoint violates its original ray quality limits")
    image_points = []
    for source, camera in zip(sources, cameras, strict=True):
        observations = [_observation(source.get(row["frame"]), index, camera["width"], camera["height"]) for index in indices]
        if any(value is None for value, _ in observations):
            return None, "source_endpoint_unavailable"
        if np.linalg.norm(observations[2][0] - observations[1][0]) < FOOT_CONTRACT["minimum_image_axis_pixels"]:
            return None, "projected_axis_degenerate"
        image_points.append([value.tolist() for value, _ in observations])
    confidence = _confidence(row["frame"], indices, sources)
    document = alignment.document()
    mapped = [alignment.map_state(from_frame=document["from_frame"], to_frame=document["to_frame"],
                                   time_s=row["time_s"], position=point.tolist())["position"] for point in points]
    result = {"source_world_points_m": dict(zip(LANDMARKS[side], points.tolist(), strict=True)),
              "source_body_points": dict(zip(LANDMARKS[side], mapped, strict=True)), "confidence": confidence,
              "image_points_px_by_view": image_points,
              "signed_ground_clearance_m": dict(zip(LANDMARKS[side], (points[:, 2] - ground["height_m"]).tolist(), strict=True)),
              "inferred_measurements": None}
    try:
        geometry = _triangle(np.asarray(mapped))
    except ValueError as error:
        return result, str(error)
    result["inferred_measurements"] = {key: geometry[key] for key in ("length", "along", "height")}
    result["geometry"] = geometry
    return result, None


def build_multiview_foot_reference(predictions_document, first_evidence_document, second_evidence_document, *,
                                  camera_documents=None, body_document=None, ground_document=None,
                                  world_to_body_document=None):
    """Return complete conditional foot slots from exact bytes; perform no filesystem I/O."""
    deadline = time.monotonic() + CONTRACT["maximum_seconds"]
    _require(camera_documents is None or type(camera_documents) in (tuple, list) and len(camera_documents) == 2,
             "Supply exactly two camera documents or none")
    cameras = (None, None) if camera_documents is None else camera_documents
    data, pins = _documents({"predictions": predictions_document, "evidence_1": first_evidence_document,
                             "evidence_2": second_evidence_document, "camera_1": cameras[0], "camera_2": cameras[1],
                             "body": body_document, "ground": ground_document, "world_to_body": world_to_body_document}, deadline)
    prediction, sources, decoded, fps, reasons = _source_envelope(data, pins, deadline)
    alignment, body_rows = _body_binding(data, pins, prediction, reasons)
    cutoff = CONTRACT["construction_fraction"] * (decoded[0] - 1) / fps
    rows, geometries = [], {}
    total = max(prediction["source_frames"], *decoded)
    for frame in range(total):
        _check(deadline)
        original = prediction["frames"][frame] if frame < prediction["source_frames"] else None
        stamp = original["time_s"] if original else frame / fps
        current = {"frame": frame, "time_s": stamp, "partition": "construction_prefix" if stamp <= cutoff else "later_excluded",
                   "source_evidence_ids": [source[frame]["id"] if frame in source else None for source in sources],
                   "view_slot_statuses": ["decoded" if frame < count else "metadata_only_not_decoded" for count in decoded],
                   "source_clocks": [{key: source[frame][key] for key in ("time_s", "decoder_timestamp_s")} if frame in source else None
                                     for source in sources], "body_pose_available": frame in body_rows, "feet": {}}
        for side, endpoints in LANDMARKS.items():
            foot = {"status": "unavailable", "reason": None, "reference": None, "endpoint_geometry": None,
                    "source_geometry": None, "original_endpoint_statuses": {
                        name: original["joints"][index]["status"] if original else "triangulation_slot_unavailable"
                        for name, index in endpoints.items()}}
            if frame >= min(decoded):
                reason = "one_or_more_views_not_decoded"
            elif stamp > cutoff:
                reason = "excluded_later_source"
            elif original is None:
                reason = "triangulation_slot_unavailable"
            elif any(frame not in source for source in sources):
                reason = "pose_not_sampled_in_both_views"
            elif reasons:
                reason = "source_or_body_binding_unavailable"
            elif frame not in body_rows:
                reason = "body_pose_unavailable"
            else:
                source_geometry, reason = _foot(original, side, sources, [data["camera_1"], data["camera_2"]],
                                                alignment, data["ground"], deadline)
                if source_geometry is not None:
                    geometry = source_geometry.pop("geometry", None)
                    foot["source_geometry"] = source_geometry
                    if geometry is not None:
                        geometries[(frame, side)] = geometry
            foot["reason"] = reason
            current["feet"][side] = foot
        rows.append(current)
    shapes = {}
    for side in LANDMARKS:
        selected = [row for row in rows if (row["frame"], side) in geometries]
        shape = {"status": "unavailable", "fixed_parameters": None, "eligible_prefix_samples": len(selected),
                 "source_frames": [row["frame"] for row in selected], "reason": "insufficient_eligible_prefix_samples",
                 "length_unit": "human_model_unit", "scale_status": "declared_camera_body_mapping_not_measured_anatomy"}
        if len(selected) >= CONTRACT["minimum_prefix_samples"]:
            weights = [row["feet"][side]["source_geometry"]["confidence"] for row in selected]
            parameters = {key: _median([geometries[(row["frame"], side)][key] for row in selected], weights)
                          for key in ("length", "along", "height")}
            low, high = FOOT_CONTRACT["ankle_along_axis_fraction_bounds"]
            if low <= parameters["along"] / parameters["length"] <= high:
                shape.update(status="conditional_fixed_triangle", fixed_parameters=parameters, reason=None)
            else:
                shape["reason"] = "median_attachment_out_of_bounds"
        shapes[side] = shape
    counts = Counter()
    for row in rows:
        _check(deadline)
        for side, foot in row["feet"].items():
            geometry, shape = geometries.get((row["frame"], side)), shapes[side]
            if geometry is not None and shape["fixed_parameters"] is not None:
                fixed = shape["fixed_parameters"]
                heel = -fixed["along"] * geometry["axis"] - fixed["height"] * geometry["radial_direction"]
                toe = heel + fixed["length"] * geometry["axis"]
                foot.update(status="conditional_kinematic_reference", reason=None,
                            reference={"heel_to_index_axis": geometry["axis"].tolist(),
                                       "axis_to_ankle_direction": geometry["radial_direction"].tolist(),
                                       "anatomical_axial_roll_rad": None, "sole_normal": None},
                            endpoint_geometry={"ankle_relative": [0., 0., 0.], "heel_relative": heel.tolist(),
                                               "foot_index_relative": toe.tolist(), "length_unit": "human_model_unit",
                                               "coordinate_frame": alignment.document()["to_frame"],
                                               "physical_model_attached": False, "collisions": False})
            elif geometry is not None:
                foot["reason"] = shape["reason"]
            counts[foot["reason"] or foot["status"]] += 1
    result = {"schema_version": 1, "profile": PROFILE,
              "status": "conditional_references_available" if counts["conditional_kinematic_reference"] else "unavailable",
              "input_pins": pins, "views": deepcopy(prediction["views"]), "binding_unavailable_reasons": reasons,
              "source_sha256": prediction["views"][0]["source_sha256"],
              "body_model_sha256": pins["body"]["canonical_sha256"] if pins["body"] else None,
              "world_to_body": alignment.document() if alignment and not reasons else None,
              "ground": deepcopy(data["ground"]), "construction_cutoff_time_s": cutoff,
              "fixed_shapes": shapes, "frames": rows, "contract": deepcopy(CONTRACT),
              "coverage": {"original_source_frames": total, "view_decoded_frames": decoded,
                           "view_metadata_frames": [data[f"evidence_{i}"]["frame_count"] for i in (1, 2)],
                           "pose_samples_by_view": [len(source) for source in sources],
                           "retained_foot_slots": total * 2, "retained_endpoint_slots": total * 6,
                           "foot_status_counts": dict(counts)},
              "physical_validation": "unverified", "body_geometry_validated": False,
              "body_or_floor_installed": False, "monocular_world_landmarks_used": False,
              "contact": "unknown", "load": "unknown", "grasp": "unknown",
              "limits": ["Saved triangulation is not repeated; exact pins and stored quality are checked",
                         "Cross-view synchronization and reconstructed camera scale remain supplied assumptions",
                         "Fixed landmark triangles are not measured shoes, soles or anatomical axial roll",
                         "World-to-body scale is the existing declared body placement, not calibration of monocular anatomy",
                         "Ground clearance is descriptive and unclipped, never a contact label or fitting objective",
                         "This distinct reference profile is not installed by existing body consumers"]}
    _check(deadline)
    size = 0
    for chunk in json.JSONEncoder(allow_nan=False, separators=(",", ":")).iterencode(result):
        _check(deadline)
        size += len(chunk.encode("utf-8"))
        _require(size <= CONTRACT["maximum_output_bytes"], "Complete multiview foot output exceeds 8 MiB")
    _check(deadline)
    return result
