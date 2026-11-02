"""Admission bindings for unchanged grounded camera bodies in one shared world.

This validates custody and coordinates, not anatomy, physics or task success. The
caller still validates the full human model and ordinary shared-world schema.
"""

import json
import math
import time

from weft.camera_body import ROOT_LIMIT, validate_camera_binding
from weft.camera_shared_receipt import BODY_RUNTIME_FILES
from weft.io import canonical_hash
from weft.multiview_root import METHOD, METHODS as ROOT_METHODS, shape_hash
from weft.multiview_body import METHODS as BODY_METHODS


PROFILE = "camera-shared-coupling-v1"
RECONSTRUCTED_PROFILE = "camera-shared-coupling-v2"
SOURCE_CLOCK_PROFILE = "camera-shared-source-clock-v1"
LENGTH_UNIT = "supplied_camera_metre_prior"
MASS_UNIT = "authored_body_mass_prior"
CALIBRATION_STATUS = "estimated_prior"
MAX_BODY_BYTES = 8 * 1024 * 1024
MAX_POSES = 900
MAX_SOURCE_FRAMES = 1800
MAX_SECONDS = 5.0
ROOT_ABS_MAX = 50.1
SEGMENT_CONTACT_ABS_MAX = 64.0

_MAX_NODES = 300_000
_MAX_DEPTH = 64
_SMALL_DOCUMENT_BYTES = 64 * 1024
_IDENTITY = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))


def _check_time(deadline):
    if time.monotonic() > deadline:
        raise ValueError("Camera coupling exceeded its cooperative 5-second bound")


def _bounded_json(value, limit, deadline, label):
    # Bound malformed trees before validators hash, copy or convert their arrays.
    active, nodes, characters = set(), 0, 0

    def visit(item, depth):
        nonlocal nodes, characters
        nodes += 1
        if nodes > _MAX_NODES or depth > _MAX_DEPTH:
            raise ValueError(f"{label} exceeds its JSON structure bound")
        if nodes % 256 == 0:
            _check_time(deadline)
        kind = type(item)
        if kind is str:
            characters += len(item)
            if characters > limit:
                raise ValueError(f"{label} exceeds its byte bound")
        elif kind is float:
            if not math.isfinite(item):
                raise ValueError(f"{label} contains a nonfinite number")
        elif kind is int:
            if item.bit_length() > 64:
                raise ValueError(f"{label} contains an oversized integer")
        elif item is None or kind is bool:
            pass
        elif kind in (dict, list):
            if id(item) in active:
                raise ValueError(f"{label} contains a JSON cycle")
            if len(item) > _MAX_NODES - nodes:
                raise ValueError(f"{label} exceeds its JSON structure bound")
            active.add(id(item))
            if kind is dict:
                for key, child in item.items():
                    if type(key) is not str:
                        raise ValueError(f"{label} requires string JSON keys")
                    visit(key, depth + 1)
                    visit(child, depth + 1)
            else:
                for child in item:
                    visit(child, depth + 1)
            active.remove(id(item))
        else:
            raise ValueError(f"{label} requires plain JSON values")

    _check_time(deadline)
    visit(value, 0)
    size = 0
    encoder = json.JSONEncoder(allow_nan=False)
    try:
        for index, part in enumerate(encoder.iterencode(value)):
            size += len(part.encode("utf-8"))
            if size > limit:
                raise ValueError(f"{label} exceeds its byte bound")
            if index % 256 == 0:
                _check_time(deadline)
    except (UnicodeError, RecursionError) as exc:
        raise ValueError(f"{label} is not bounded JSON") from exc
    _check_time(deadline)


def _number(value):
    return type(value) in (int, float) and math.isfinite(value)


def _text(value, limit):
    return type(value) is str and 1 <= len(value.strip()) <= limit


def _proper_rotation(value):
    if (type(value) is not list or len(value) != 3
            or any(type(row) is not list or len(row) != 3 for row in value)
            or any(not _number(v) or abs(v) > 1.000001 for row in value for v in row)):
        return False
    for i in range(3):
        for j in range(3):
            if abs(sum(value[k][i] * value[k][j] for k in range(3)) - (i == j)) > 1e-6:
                return False
    a, b, c = value
    determinant = (a[0] * (b[1] * c[2] - b[2] * c[1])
                   - a[1] * (b[0] * c[2] - b[2] * c[0])
                   + a[2] * (b[0] * c[1] - b[1] * c[0]))
    return abs(determinant - 1.0) <= 1e-6


def _prepare_body(body, deadline):
    if (type(body) is not dict or body.get("profile") != "exercise-v3"
            or type(body.get("schema_version")) is not int or body["schema_version"] != 3):
        raise ValueError("Camera coupling requires an original exercise-v3/schema3 body")
    binding = body.get("camera_reference")
    if (type(binding) is not dict or binding.get("method") not in (*ROOT_METHODS, *BODY_METHODS)
            or type(binding.get("ground")) is not dict
            or binding.get("method") in BODY_METHODS and type(binding.get("body_refinement")) is not dict):
        raise ValueError("Camera coupling requires grounded two-view/multi-view root or reconstructed bodies")
    refinement = binding.get("root_refinement")
    if (type(refinement) is not dict or type(refinement.get("source_frames")) is not int
            or not 2 <= refinement["source_frames"] <= MAX_SOURCE_FRAMES):
        raise ValueError("Camera coupling requires 2–1800 original source frames")
    references = [body.get(key) for key in ("reference", "reference_evidence", "root_reference")]
    references.append(refinement.get("roots"))
    if (any(type(rows) is not list or not 2 <= len(rows) <= MAX_POSES for rows in references)
            or len({len(rows) for rows in references}) != 1
            or any(type(row) is not dict for rows in references for row in rows)):
        raise ValueError("Camera coupling must retain the same 2–900 original pose rows")
    _bounded_json(body, MAX_BODY_BYTES, deadline, "Camera body")
    fps, dt = refinement.get("fps"), body.get("dt")
    if not _number(fps) or not 1 <= fps <= 240 or not _number(dt) or not 1 / 120 <= dt <= 1 / 20:
        raise ValueError("Camera source fps and original 20–120Hz body period must remain finite and bounded")
    frame = body.get("body_frame")
    if type(frame) is not dict or not _proper_rotation(frame.get("rotation")):
        raise ValueError("Camera body orientation must remain a proper rotation")
    mobility = body.get("mobility")
    if (type(mobility) is not dict or not _number(mobility.get("base_height"))
            or not .01 <= mobility["base_height"] <= 5.0):
        raise ValueError("Camera body root must remain 0.01–5 units above supplied ground")
    # Validate the complete constructor chain; never rerun a fit or replace it with its root-only parent.
    try:
        validate_camera_binding(body)
    except (KeyError, TypeError, IndexError, OverflowError) as exc:
        raise ValueError("Malformed camera, root or ground binding") from exc
    _check_time(deadline)
    angular, evidence, roots, source_roots = references
    clock_rows = []
    for pose, provenance, root, source in zip(angular, evidence, roots, source_roots, strict=True):
        stamp = source["time_s"]
        if (not _number(pose.get("time_s")) or pose["time_s"] != stamp
                or abs(stamp - source["frame"] / fps) > 1e-10
                or any(abs(v) > ROOT_LIMIT for v in root["translation"])):
            raise ValueError("Camera coupling requires complete aligned original clocks and bounded roots")
        clock_rows.append({"frame": source["frame"], "time_s": stamp, "evidence_id": provenance["evidence_id"]})
    if roots[0]["translation"] != [0.0, 0.0, 0.0]:
        raise ValueError("Camera root must retain its original zero translation")
    _check_time(deadline)
    return binding, refinement, clock_rows


def _coupling(body, binding, refinement, rows, deadline):
    # Source-clock control does not rewrite the independently authored human timestep.
    control_dt = 1.0 / refinement["fps"]
    clock_source = {"schema_version": 1, "profile": SOURCE_CLOCK_PROFILE,
                    "source_sha256": body["source_sha256"], "source_fps": refinement["fps"],
                    "total_source_frames": refinement["source_frames"], "control_dt_s": control_dt, "rows": rows}
    origin, ground = binding["origin_world_m"], binding["ground"]
    value = {
        "schema_version": 1, "profile": PROFILE, "source_sha256": body["source_sha256"],
        "body_sha256": canonical_hash(body), "camera_binding_sha256": canonical_hash(binding),
        "ground_sha256": canonical_hash(ground), "root_reference_sha256": canonical_hash(body["root_reference"]),
        "shape_sha256": shape_hash(body), "source_clock_sha256": canonical_hash(clock_source),
        "world_to_local": {"scale": 1.0, "rotation": [list(row) for row in _IDENTITY],
                           "translation": [-origin[0], -origin[1], -ground["height_m"]]},
        "shared_placement": {"scale": 1.0, "origin": [0.0, 0.0, 0.0]},
        "clock": {"control_dt_s": control_dt, "source_fps": refinement["fps"],
                  "source_time_offset_s": rows[0]["time_s"], "first_source_frame": rows[0]["frame"],
                  "last_source_frame": rows[-1]["frame"], "total_source_frames": refinement["source_frames"],
                  "retained_poses": len(rows), "horizon_steps": rows[-1]["frame"] - rows[0]["frame"]},
        "calibration_status": CALIBRATION_STATUS,
        "units": {"camera_length": "m", "model_length": LENGTH_UNIT, "model_mass": MASS_UNIT},
        "assumptions": {"root_scale": "supplied_camera_scale", "body_shape": "unchanged_monocular_estimate",
                        "body_mass": "unchanged_authored_model", "anatomy": "not_independently_measured", "support": "unscored"},
        "runtime_bounds": {"root_abs_max": ROOT_ABS_MAX, "segment_contact_abs_max": SEGMENT_CONTACT_ABS_MAX},
    }
    if binding["method"] != METHOD:
        value["profile"] = RECONSTRUCTED_PROFILE
        value["body_reconstruction"] = {"method": binding["method"], "view_count": len(refinement["views"]),
            "body_refinement_sha256": canonical_hash(binding["body_refinement"]) if "body_refinement" in binding else None}
        if "body_refinement" in binding:
            value["assumptions"]["body_shape"] = "unchanged_reconstructed_estimate_with_explicit_monocular_priors"
    _check_time(deadline)
    return value


def _validate_scene_binding(body, coupling, dt, frame, source):
    if not _number(dt) or dt != coupling["clock"]["control_dt_s"]:
        raise ValueError("Shared control dt must equal the exact original source frame period")
    if (type(source) is not dict or source.get("mode") != "source_qualified"
            or source.get("source_sha256") != body["source_sha256"]
            or type(source.get("total_source_frames")) is not int
            or source["total_source_frames"] != coupling["clock"]["total_source_frames"]):
        raise ValueError("Shared source binding must preserve source identity and original frame count")
    if (type(frame) is not dict or set(frame) != {"axes", "units", "calibration_status", "scale_bridge", "clock_binding"}
            or frame["axes"] != "x_right_y_depth_z_up" or frame["calibration_status"] != CALIBRATION_STATUS):
        raise ValueError("Camera shared frame must use z-up axes and estimated_prior calibration")
    units = {"length": LENGTH_UNIT, "mass": MASS_UNIT, "time": "second", "orientation": "quaternion_wxyz",
             "angular_velocity": "world_radians_per_second"}
    if frame["units"] != units:
        raise ValueError("Camera shared units must distinguish supplied metres from authored model mass")
    bridge = frame["scale_bridge"]
    if (type(bridge) is not dict or set(bridge) != {"body_height_to_world_scale", "origin_world", "method", "uncertainty_note"}
            or not _number(bridge["body_height_to_world_scale"]) or bridge["body_height_to_world_scale"] != 1.0
            or type(bridge["origin_world"]) is not list or len(bridge["origin_world"]) != 3
            or any(not _number(v) or v != 0.0 for v in bridge["origin_world"])
            or not _text(bridge["method"], 160) or not _text(bridge["uncertainty_note"], 500)):
        raise ValueError("Camera shared placement requires unit scale, zero origin and explicit uncertainty")
    clock = frame["clock_binding"]
    if (type(clock) is not dict or set(clock) != {"control_dt_s", "physics_substeps", "source_fps", "source_time_offset_s", "alignment_method"}
            or not _number(clock["control_dt_s"]) or clock["control_dt_s"] != dt
            or not _number(clock["source_fps"]) or clock["source_fps"] != coupling["clock"]["source_fps"]
            or not _number(clock["source_time_offset_s"]) or clock["source_time_offset_s"] != coupling["clock"]["source_time_offset_s"]
            or type(clock["physics_substeps"]) is not int or not 8 <= clock["physics_substeps"] <= 64
            or not _text(clock["alignment_method"], 200)):
        raise ValueError("Camera shared clock must preserve original fps, offset and control period")


def build_camera_coupling(body):
    """Bind an unchanged body without constructing, copying or validating a native model."""
    deadline = time.monotonic() + MAX_SECONDS
    binding, refinement, rows = _prepare_body(body, deadline)
    value = _coupling(body, binding, refinement, rows, deadline)
    result = json.loads(json.dumps(value, sort_keys=True, allow_nan=False))
    _check_time(deadline)
    return result


def validate_camera_coupling(body, coupling, *, dt, coordinate_frame, source_binding):
    """Validate v6 camera custody; full scene geometry and source-document proof stay caller-owned."""
    deadline = time.monotonic() + MAX_SECONDS
    binding, refinement, rows = _prepare_body(body, deadline)
    for value, label in ((coupling, "Camera coupling"), (coordinate_frame, "Shared frame"), (source_binding, "Source binding")):
        _bounded_json(value, _SMALL_DOCUMENT_BYTES, deadline, label)
    expected = _coupling(body, binding, refinement, rows, deadline)
    if canonical_hash(coupling) != canonical_hash(expected):
        raise ValueError("Camera coupling differs from its exact source, transform, assumptions or clock pins")
    _validate_scene_binding(body, expected, dt, coordinate_frame, source_binding)
    result = json.loads(json.dumps(expected, sort_keys=True, allow_nan=False))
    _check_time(deadline)
    return result
