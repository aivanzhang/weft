"""Prepared conditional foot geometry; never relabel calibrated evidence or admit a world."""

from collections import Counter
from dataclasses import dataclass
from hashlib import sha256
import json
import math
import time

import numpy as np

from weft.camera_body import validate_camera_binding
from weft.camera_ground import validate_ground
from weft.exercise_support_evidence import _json_size
from weft.foot_reference import FOOT_CONTRACT
from weft.io import canonical_hash
from weft.multiview_foot_reference import CONTRACT as SOURCE_CONTRACT, PROFILE as SOURCE_PROFILE
from weft.multiview_root import METHOD as ROOT_METHOD
from weft.spatial_pose import rotation_to_zyx
from weft.world_alignment import load_world_alignment


PROFILE = "calibrated-foot-consumer-v1"
MAX_DOCUMENT_BYTES = 8 << 20
MAX_SECONDS = 5.
SIDES = ("left", "right")
ENDPOINTS = ("ankle", "heel", "foot_index")
PIN_NAMES = ("predictions", "evidence_1", "evidence_2", "camera_1", "camera_2", "body", "ground", "world_to_body")
UNAVAILABLE_REASONS = frozenset({
    "one_or_more_views_not_decoded", "excluded_later_source", "triangulation_slot_unavailable",
    "pose_not_sampled_in_both_views", "source_or_body_binding_unavailable", "body_pose_unavailable",
    "triangulated_endpoint_unavailable", "source_endpoint_unavailable", "projected_axis_degenerate",
    "foot_length_out_of_bounds", "ankle_axis_plane_degenerate_or_out_of_bounds", "ankle_attachment_out_of_bounds",
    "insufficient_eligible_prefix_samples", "median_attachment_out_of_bounds",
})


def _require(value, message):
    if not value:
        raise ValueError("Calibrated foot consumer: " + message)


def _check(deadline):
    if time.monotonic() >= deadline:
        raise TimeoutError("Calibrated foot consumer exceeded its cooperative deadline")


def _number(value, low, high):
    return type(value) in (int, float) and math.isfinite(value) and low <= value <= high


def _hash(value):
    return type(value) is str and len(value) == 64 and set(value) <= set("0123456789abcdef")


def _unique(pairs):
    result = {}
    for key, value in pairs:
        _require(key not in result, "duplicate JSON field")
        result[key] = value
    return result


def _constant(_value):
    raise ValueError("Calibrated foot consumer requires finite JSON")


def _parse(raw, name, deadline):
    _check(deadline)
    _require(type(raw) is bytes and 0 < len(raw) <= MAX_DOCUMENT_BYTES, name + " requires 1 byte–8 MiB of exact JSON")
    try:
        result = json.loads(raw.decode("utf-8"), object_pairs_hook=_unique, parse_constant=_constant)
    except (ValueError, UnicodeError, RecursionError) as exc:
        raise ValueError("Invalid bounded calibrated-foot JSON: " + name) from exc
    _require(type(result) is dict, name + " must be a JSON object")
    key, maximum = ("frames", 1800) if name == "reference" else ("reference", 900)
    _require(type(result.get(key)) is list and 1 <= len(result[key]) <= maximum, name + " row envelope exceeded")
    _require(_json_size(result, deadline) <= MAX_DOCUMENT_BYTES, name + " canonical byte envelope exceeded")
    _check(deadline)
    return result


def _vector(value, bound):
    _require(type(value) is list and len(value) == 3 and all(_number(v, -bound, bound) for v in value),
             "finite bounded three-coordinate vector required")
    return np.asarray(value, dtype=float)


def _rotation(x, z):
    x, z = _vector(x, 1), _vector(z, 1)
    rotation = np.column_stack((x, np.cross(z, x), z))
    _require(np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-8, rtol=0)
             and math.isclose(float(np.linalg.det(rotation)), 1., abs_tol=1e-8),
             "foot axes must already be proper orthonormal directions; no normalization")
    return rotation


def _envelope(reference):
    _require({"world_to_body", "ground", "body_model_sha256"} <= set(reference),
             "explicit body/alignment/ground slots are required, including nulls")
    _require(type(reference.get("schema_version")) is int and reference["schema_version"] == 1
             and reference.get("profile") == SOURCE_PROFILE and reference.get("contract") == SOURCE_CONTRACT,
             "the original calibrated profile and contract must remain distinct")
    _require(reference.get("physical_validation") == "unverified"
             and all(reference.get(key) is False for key in ("body_geometry_validated", "body_or_floor_installed", "monocular_world_landmarks_used"))
             and all(reference.get(key) == "unknown" for key in ("contact", "load", "grasp")),
             "conditional geometry cannot assert physical admission or measured contact")
    pins, views = reference.get("input_pins"), reference.get("views")
    _require(type(pins) is dict and set(pins) == set(PIN_NAMES), "all original input pins are required")
    for pin in pins.values():
        _require(pin is None or type(pin) is dict and set(pin) == {"sha256", "canonical_sha256", "bytes"}
                 and _hash(pin["sha256"]) and _hash(pin["canonical_sha256"])
                 and type(pin["bytes"]) is int and 0 < pin["bytes"] <= 32 << 20, "invalid original input pin")
    fields = {"name", "source_sha256", "camera_sha256", "evidence_sha256", "calibration_quality", "clock_provenance"}
    _require(type(views) is list and len(views) == 2 and all(type(view) is dict and set(view) == fields for view in views),
             "both original view identities are required")
    for view in views:
        _require(all(_hash(view[key]) for key in ("source_sha256", "camera_sha256", "evidence_sha256"))
                 and type(view["name"]) is str and 1 <= len(view["name"]) <= 64
                 and type(view["clock_provenance"]) is str and 1 <= len(view["clock_provenance"].strip()) <= 2000
                 and view["calibration_quality"] in ("authored", "reconstructed", "measured"), "invalid source/calibration/clock claim")
    _require(views[0]["source_sha256"] != views[1]["source_sha256"] and views[0]["name"] != views[1]["name"]
             and reference.get("source_sha256") == views[0]["source_sha256"], "distinct source views and primary identity required")
    reasons = reference.get("binding_unavailable_reasons")
    _require(type(reasons) is list and len(reasons) <= 16
             and all(type(value) is str and 1 <= len(value) <= 100 for value in reasons), "bounded original unavailable reasons required")
    coverage = reference.get("coverage")
    _require(type(coverage) is dict and type(coverage.get("original_source_frames")) is int
             and 2 <= coverage["original_source_frames"] <= 1800
             and len(reference["frames"]) == coverage["original_source_frames"]
             and type(coverage.get("retained_foot_slots")) is int and coverage["retained_foot_slots"] == 2 * len(reference["frames"])
             and type(coverage.get("retained_endpoint_slots")) is int and coverage["retained_endpoint_slots"] == 6 * len(reference["frames"]),
             "complete original source/foot/endpoint denominators required")
    for key, maximum in (("view_decoded_frames", 1800), ("view_metadata_frames", 1800), ("pose_samples_by_view", 900)):
        values = coverage.get(key)
        _require(type(values) is list and len(values) == 2
                 and all(type(value) is int and 1 <= value <= maximum for value in values), "invalid per-view denominator")
    counts = coverage.get("foot_status_counts")
    _require(type(counts) is dict and set(counts) <= UNAVAILABLE_REASONS | {"conditional_kinematic_reference"}
             and all(type(value) is int and 0 <= value <= 3600 for value in counts.values()), "integer foot-status counts required")
    _require(max(*coverage["view_decoded_frames"], *coverage["view_metadata_frames"]) == coverage["original_source_frames"]
             and _number(reference.get("construction_cutoff_time_s"), 0, 60), "source interval/partition differs")


def _binding(body, body_raw, reference):
    reasons = list(reference["binding_unavailable_reasons"])
    pin = reference["input_pins"]["body"]
    body_hash = canonical_hash(body)
    if pin is None or pin != {"sha256": sha256(body_raw).hexdigest(), "canonical_sha256": body_hash, "bytes": len(body_raw)}:
        reasons.append("exact_original_body_bytes_unavailable")
    if reference.get("body_model_sha256") != body_hash:
        reasons.append("original_body_identity_mismatch")
    if reasons:
        return None, reasons
    try:
        _require(body.get("profile") == "exercise-v3" and type(body.get("schema_version")) is int
                 and body["schema_version"] == 3 and body.get("dt") == 1 / 30
                 and all(type(body.get(key)) is dict for key in ("camera_reference", "body_frame", "mobility"))
                 and body["camera_reference"].get("method") == ROOT_METHOD, "original 30-Hz root-only camera body required")
        binding = validate_camera_binding(body)
        ground, mapping = reference.get("ground"), reference.get("world_to_body")
        validate_ground(ground, binding)
        _require(ground == binding["ground"], "supplied ground differs from bound body")
        root = binding["root_refinement"]
        _require(reference["views"] == root["views"] and reference["source_sha256"] == binding["source_sha256"]
                 and reference["input_pins"]["predictions"]["canonical_sha256"] == root["triangulation_sha256"],
                 "triangulation/body/view binding differs")
        for index, view in enumerate(reference["views"], 1):
            for name, target in (("camera", "camera_sha256"), ("evidence", "evidence_sha256")):
                _require(reference["input_pins"][f"{name}_{index}"]["canonical_sha256"] == view[target], "original camera/pose pin differs")
        for name, document in (("ground", ground), ("world_to_body", mapping)):
            _require(reference["input_pins"][name]["canonical_sha256"] == canonical_hash(document), "ground/alignment canonical pin differs")
        alignment = load_world_alignment(mapping)
        target = {"id": "body:" + body_hash, "length_unit": "human_model_unit", "axes": "x_right_y_away_z_up"}
        source = {"id": "camera-world:" + binding["calibration_sha256"], "length_unit": "m", "axes": "x_right_y_away_z_up"}
        origin = binding["origin_world_m"]
        _require(alignment.ready and mapping["from_frame"] == source and mapping["to_frame"] == target
                 and mapping["spatial"] == {"scale": 1., "rotation_wxyz": [1., 0., 0., 0.],
                                           "translation": [-origin[0], -origin[1], -ground["height_m"]]}
                 and mapping["clock"] == {"rate": 1., "source_origin_s": 0., "target_origin_s": 0.},
                 "explicit original world/body placement, units and clock required")
        _require(len(body["reference"]) == len(root["roots"])
                 and all(type(row) is dict and row.get("time_s") == pose["time_s"]
                         for row, pose in zip(body["reference"], root["roots"], strict=True)), "angular/root source clocks differ")
        _require(reference["coverage"]["view_metadata_frames"] == [root["source_frames"]] * 2,
                 "source metadata denominator differs from body")
        cutoff = .6 * (reference["coverage"]["view_decoded_frames"][0] - 1) / root["fps"]
        _require(reference["construction_cutoff_time_s"] == cutoff, "construction clock partition changed")
        return {"body_frame": target, "fps": root["fps"], "source_frames": root["source_frames"],
                "world_to_body_translation": np.asarray(mapping["spatial"]["translation"], dtype=float),
                "ground_height_m": ground["height_m"],
                "body_rows": {row["frame"]: row for row in root["roots"]}, "start": root["roots"][0]}, []
    except (ValueError, TypeError, KeyError, IndexError):
        return None, ["incompatible_body_camera_world_clock_binding"]


def _shapes(reference):
    shapes = reference.get("fixed_shapes")
    _require(type(shapes) is dict and set(shapes) == set(SIDES), "both original fixed-shape slots required")
    for shape in shapes.values():
        _require(type(shape) is dict and shape.get("status") in ("unavailable", "conditional_fixed_triangle")
                 and shape.get("length_unit") == "human_model_unit"
                 and shape.get("scale_status") == "declared_camera_body_mapping_not_measured_anatomy", "conditional shape scale/provenance differs")
        count, frames = shape.get("eligible_prefix_samples"), shape.get("source_frames")
        _require(type(count) is int and 0 <= count <= 900 and type(frames) is list and len(frames) == count
                 and all(type(frame) is int and 0 <= frame < len(reference["frames"]) for frame in frames)
                 and frames == sorted(set(frames)), "fixed-shape source slots differ")
        values = shape.get("fixed_parameters")
        if shape["status"] == "unavailable":
            _require(values is None, "unavailable shape must remain null")
        else:
            _require(type(values) is dict and set(values) == {"length", "along", "height"}
                     and _number(values["length"], *FOOT_CONTRACT["foot_length_bounds"])
                     and _number(values["height"], *FOOT_CONTRACT["ankle_axis_height_bounds"])
                     and _number(values["along"], -.5 * values["length"], 1.5 * values["length"])
                     and count >= FOOT_CONTRACT["minimum_prefix_samples"] and shape.get("reason") is None,
                     "unchanged fixed triangle bounds and prefix evidence required")
    return shapes


def _validate_source_triplet(source, rotation, binding):
    world, body = source.get("source_world_points_m"), source.get("source_body_points")
    _require(all(type(points) is dict and set(points) == set(ENDPOINTS) for points in (world, body)),
             "source triplet requires all three retained world/body points")
    points = {}
    for name in ENDPOINTS:
        world_point, body_point = _vector(world[name], 2000), _vector(body[name], 4000)
        _require(np.allclose(world_point + binding["world_to_body_translation"], body_point, atol=1e-10, rtol=0),
                 "source mapping differs from the bound original world/body placement")
        _require(math.isclose(source["signed_ground_clearance_m"][name],
                              world_point[2] - binding["ground_height_m"], abs_tol=1e-10, rel_tol=0),
                 "source gap differs from the retained world point and bound ground")
        points[name] = body_point
    values = source.get("inferred_measurements")
    _require(type(values) is dict and set(values) == {"length", "along", "height"}
             and _number(values["length"], *FOOT_CONTRACT["foot_length_bounds"])
             and _number(values["height"], *FOOT_CONTRACT["ankle_axis_height_bounds"])
             and _number(values["along"], -.5 * values["length"], 1.5 * values["length"]),
             "source triplet requires unchanged bounded inferred dimensions")
    # Check saved algebra without refitting the source triangle or replacing its gaps.
    _require(np.allclose(points["foot_index"] - points["heel"], values["length"] * rotation[:, 0], atol=1e-10, rtol=0)
             and np.allclose(points["ankle"] - points["heel"],
                             values["along"] * rotation[:, 0] + values["height"] * rotation[:, 2], atol=1e-10, rtol=0),
             "source triplet differs from the saved axes or inferred dimensions")


def _available_foot(foot, shape, binding):
    reference, geometry = foot.get("reference"), foot.get("endpoint_geometry")
    _require(shape["status"] == "conditional_fixed_triangle" and foot.get("reason") is None
             and type(reference) is dict and reference.get("sole_normal") is None
             and reference.get("anatomical_axial_roll_rad") is None and type(geometry) is dict,
             "conditional foot cannot add anatomical/contact assumptions")
    rotation = _rotation(reference.get("heel_to_index_axis"), reference.get("axis_to_ankle_direction"))
    values = shape["fixed_parameters"]
    local = {"heel": [-values["along"], 0., -values["height"]],
             "foot_index": [values["length"] - values["along"], 0., -values["height"]]}
    _require(geometry.get("physical_model_attached") is False and geometry.get("collisions") is False
             and geometry.get("length_unit") == "human_model_unit" and geometry.get("coordinate_frame") == binding["body_frame"]
             and np.array_equal(_vector(geometry.get("ankle_relative"), 1), np.zeros(3)), "bound body-relative geometry required")
    for name, point in local.items():
        _require(np.allclose(rotation @ point, _vector(geometry.get(name + "_relative"), 2), atol=1e-10, rtol=0),
                 "saved endpoint differs from the unchanged triangle/rotation")
    source = foot.get("source_geometry")
    _require(type(source) is dict and _number(source.get("confidence"), .5, 1)
             and type(source.get("signed_ground_clearance_m")) is dict
             and set(source["signed_ground_clearance_m"]) == set(ENDPOINTS)
             and all(_number(value, -3000, 3000) for value in source["signed_ground_clearance_m"].values()),
             "retain original confidence and signed conditional source gaps")
    _validate_source_triplet(source, rotation, binding)
    return {"fixed_parameters": values, "local_centerline": local, "rotation_in_body_frame": rotation.tolist(),
            "endpoint_geometry": geometry, "source_confidence": source["confidence"],
            "source_signed_ground_clearance_m": source["signed_ground_clearance_m"],
            "conditional": True, "physical_admission": False}


def _rows(reference, binding, shapes, deadline):
    counts, sampled, available = Counter(), [0, 0], {side: [] for side in SIDES}
    prepared_start = None
    previous = -1.
    decoded = reference["coverage"]["view_decoded_frames"]
    cutoff = reference["construction_cutoff_time_s"]
    for index, row in enumerate(reference["frames"]):
        _check(deadline)
        _require(type(row) is dict and type(row.get("frame")) is int and row["frame"] == index
                 and _number(row.get("time_s"), 0, 60) and row["time_s"] > previous
                 and row.get("partition") == ("construction_prefix" if row["time_s"] <= cutoff else "later_excluded")
                 and type(row.get("body_pose_available")) is bool and type(row.get("feet")) is dict
                 and set(row["feet"]) == set(SIDES), "complete ordered original source slots required")
        previous = row["time_s"]
        ids, clocks = row.get("source_evidence_ids"), row.get("source_clocks")
        _require(type(ids) is list and len(ids) == 2 and type(clocks) is list and len(clocks) == 2
                 and row.get("view_slot_statuses") == ["decoded" if index < value else "metadata_only_not_decoded" for value in decoded],
                 "original view sampling/decoding slots required")
        for view, (identifier, clock) in enumerate(zip(ids, clocks, strict=True)):
            if identifier is None:
                _require(clock is None, "unsampled source cannot acquire a clock")
            else:
                _require(identifier == f"{reference['views'][view]['source_sha256']}:frame:{index}"
                         and index < decoded[view] and type(clock) is dict and set(clock) == {"time_s", "decoder_timestamp_s"}
                         and _number(clock["time_s"], 0, 60) and clock["time_s"] == row["time_s"]
                         and _number(clock["decoder_timestamp_s"], 0, 60)
                         and abs(clock["decoder_timestamp_s"] - row["time_s"]) <= 1e-7, "original pose identity or clock changed")
                sampled[view] += 1
        if binding:
            _require(abs(row["time_s"] - index / binding["fps"]) <= 1e-7
                     and row["body_pose_available"] == (index in binding["body_rows"]), "body/source clock or original pose availability differs")
        expected = ("one_or_more_views_not_decoded" if index >= min(decoded) else
                    "excluded_later_source" if row["time_s"] > cutoff else
                    "triangulation_slot_unavailable" if binding and index >= binding["source_frames"] else
                    "pose_not_sampled_in_both_views" if None in ids else
                    "source_or_body_binding_unavailable" if reference["binding_unavailable_reasons"] else
                    "body_pose_unavailable" if binding and not row["body_pose_available"] else None)
        start = binding and index == binding["start"]["frame"]
        prepared = {}
        for side, foot in row["feet"].items():
            _require(type(foot) is dict and foot.get("status") in ("unavailable", "conditional_kinematic_reference")
                     and type(foot.get("original_endpoint_statuses")) is dict and set(foot["original_endpoint_statuses"]) == set(ENDPOINTS)
                     and all(type(value) is str and 1 <= len(value) <= 100 for value in foot["original_endpoint_statuses"].values()),
                     "every original foot/endpoint status must remain")
            if expected:
                _require(foot["status"] == "unavailable" and foot.get("reason") == expected
                         and foot.get("source_geometry") is None, "excluded/unsampled/unbound slot acquired geometry")
            if foot["status"] == "unavailable":
                _require(foot.get("reason") in UNAVAILABLE_REASONS
                         and foot.get("reference") is None and foot.get("endpoint_geometry") is None,
                         "unavailable foot cannot supply an inferred reference")
            else:
                _require(expected is None and not reference["binding_unavailable_reasons"]
                         and set(foot["original_endpoint_statuses"].values()) == {"estimated"}, "available foot contradicts original source status")
                available[side].append(index)
                if binding:
                    geometry = _available_foot(foot, shapes[side], binding)
                    if start:
                        prepared[side] = geometry
            counts[foot.get("reason") or foot["status"]] += 1
        if start and len(prepared) == 2:
            _require(row["time_s"] == binding["start"]["time_s"] and ids == binding["start"]["source_evidence_ids"],
                     "exact original body starting pose differs")
            prepared_start = {"frame": index, "time_s": row["time_s"], "source_evidence_ids": ids, "feet": prepared}
    _require(reference["coverage"].get("foot_status_counts") == dict(counts)
             and reference["coverage"]["pose_samples_by_view"] == sampled
             and reference.get("status") == ("conditional_references_available" if sum(map(len, available.values())) else "unavailable"),
             "retained source denominators or availability counts differ")
    for side, shape in shapes.items():
        _require(all(reference["frames"][index]["partition"] == "construction_prefix" for index in shape["source_frames"]),
                 "fixed shape includes later evidence")
        if shape["status"] == "conditional_fixed_triangle":
            _require(shape["source_frames"] == available[side], "fixed shape provenance differs from available prefix slots")
    return prepared_start


@dataclass(frozen=True)
class CalibratedFootReference:
    """Keep raw evidence once; repeated control queries read only bounded start caches."""

    body_bytes: bytes
    reference_bytes: bytes
    _description: bytes
    _initial: bytes | None
    _body_frame: tuple
    _rotations: tuple

    @property
    def ready(self):
        return self._initial is not None

    def describe(self):
        """Return small identity/uncertainty metadata, never the full source history."""
        return json.loads(self._description)

    def initial_geometry(self):
        """Return conditional local centerlines at the original start, or None."""
        return json.loads(self._initial) if self._initial is not None else None

    def initial_ankle_angles(self, knee_rotations, *, coordinate_frame):
        """Convert existing knee orientations in the exact bound body frame to ZYX angles."""
        _require(self.ready, "both feet are required at the original body start; no later fallback")
        _require(coordinate_frame == dict(self._body_frame), "knee rotations must name the exact body frame/units/axes")
        _require(type(knee_rotations) is dict and set(knee_rotations) == set(SIDES), "both original-start knee rotations required")
        result = []
        for side, values in zip(SIDES, self._rotations, strict=True):
            supplied = knee_rotations[side]
            if type(supplied) is np.ndarray:
                bounded = supplied.shape == (3, 3) and supplied.dtype.kind in "iuf"
            else:
                bounded = (type(supplied) in (list, tuple) and len(supplied) == 3
                           and all(type(row) in (list, tuple) and len(row) == 3
                                   and all(_number(value, -1, 1) for value in row) for row in supplied))
            _require(bounded, "knee orientation must be a bounded 3×3 proper rotation")
            try:
                knee = np.asarray(supplied)
                valid = (knee.shape == (3, 3) and knee.dtype.kind in "iuf" and np.isfinite(knee).all()
                         and np.allclose(knee.T @ knee, np.eye(3), atol=1e-8, rtol=0)
                         and math.isclose(float(np.linalg.det(knee)), 1., abs_tol=1e-8))
            except (ValueError, TypeError):
                valid = False
            _require(valid, "knee orientation must already be a proper rotation; no normalization")
            angles, _ = rotation_to_zyx(knee.T @ np.asarray(values))
            result.extend(map(float, angles))
        return result


def load_calibrated_foot_reference(body_bytes, reference_bytes):
    """Validate exact saved bytes; prepare conditional geometry without engine imports or I/O."""
    deadline = time.monotonic() + MAX_SECONDS
    body = _parse(body_bytes, "body", deadline)
    reference = _parse(reference_bytes, "reference", deadline)
    _envelope(reference)
    binding, reasons = _binding(body, body_bytes, reference)
    _check(deadline)
    shapes = _shapes(reference)
    initial = _rows(reference, binding, shapes, deadline)
    if binding and initial is None:
        reasons.append("both_feet_unavailable_at_original_body_start")
    if reasons:
        initial = None
    description = {
        "schema_version": 1, "profile": PROFILE, "source_profile": reference["profile"],
        "status": "conditional_original_start_ready" if initial else "unavailable", "unavailable_reasons": reasons,
        "reference_sha256": sha256(reference_bytes).hexdigest(), "reference_canonical_sha256": canonical_hash(reference),
        "body_sha256": sha256(body_bytes).hexdigest(), "body_model_sha256": canonical_hash(body),
        "source_sha256": reference["source_sha256"], "source_input_pins": reference["input_pins"],
        "views": reference["views"], "coverage": reference["coverage"],
        "body_coordinate_frame": binding["body_frame"] if binding else None,
        "world_to_body": reference["world_to_body"], "ground": reference["ground"],
        "original_start": {key: binding["start"][key] for key in ("frame", "time_s", "source_evidence_ids")} if binding else None,
        "body_clock_binding_validated": binding is not None, "body_geometry_validated": False,
        "coupled_world_admitted": False, "physical_validation": "unverified", "contact": "unknown", "load": "unknown", "grasp": "unknown",
        "scope": "Conditional exact-body-frame centerlines and initial angles only; camera-body shared-world admission, placement and physics remain separate",
    }
    encoded = json.dumps(description, allow_nan=False, separators=(",", ":")).encode()
    start = json.dumps(initial, allow_nan=False, separators=(",", ":")).encode() if initial else None
    _require(len(encoded) <= 32 << 10 and (start is None or len(start) <= 8 << 10), "bounded metadata/start cache exceeded")
    rotations = tuple(tuple(tuple(row) for row in initial["feet"][side]["rotation_in_body_frame"]) for side in SIDES) if initial else ()
    frame = tuple(binding["body_frame"].items()) if binding else ()
    _check(deadline)
    return CalibratedFootReference(body_bytes, reference_bytes, encoded, start, frame, rotations)
