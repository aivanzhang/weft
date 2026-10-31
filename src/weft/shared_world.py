"""Coupled 3D articulated-human and rigid-object simulation in one MuJoCo world."""

from copy import deepcopy
import json
import math
from pathlib import PurePosixPath
import re
from types import MappingProxyType
import xml.etree.ElementTree as ET

import gymnasium as gym
from gymnasium import spaces
import mujoco
import numpy as np
from PIL import Image, ImageDraw

from weft.human import joint_contract, joint_segment, validate_human
from weft.io import canonical_hash
from weft.mobile_human import ROOT_AXES, ground_normal
from weft.shared_world_evidence import validate_equivalence_bindings
from weft.shared_world_observation import EquipmentState, HumanObjectContact, SubstepSample
from weft.spatial_rigid import DIRECTIONS, FACES, corners, rotation
from weft.whole_body import GROUP_NAMES, SUCCESS_LIMITS


PROFILE = "shared-world-v1"
FULLSPAN_PROFILE = "shared-world-v2"
HUMAN_ONLY_PROFILE = "shared-world-v3"
FULLSPAN_PROFILES = (FULLSPAN_PROFILE, HUMAN_ONLY_PROFILE)
MAX_FULLSPAN_STEPS = 1800
MAX_FULLSPAN_SECONDS = 60.0
OBJECT_ACTIONS = ("coast", "left", "right", "forward", "back", "up", "down", "brake")
MAX_SPEED = 6.0
MAX_SPIN = 200.0
MAX_CONTACTS = 256
STATE_EPS = 1e-8
COMPOUND_PROFILE = "rigid-compound-v1"
MAX_OBJECT_PARTS = 8
MAX_RIGID_PRIMITIVES = 32

CALIBRATION_STATUSES = frozenset({"calibrated", "estimated_prior", "unverified"})
SOURCE_MODES = frozenset({"analytic_fixture", "source_qualified"})
PROVENANCE_MODES = frozenset({"observed_track", "inferred_prior", "authored_addition"})
CONTACT_KINDS = frozenset(
    {
        "human_floor",
        "human_static",
        "human_object",
        "object_floor",
        "object_static",
        "object_object",
    }
)

HUMAN_SEGMENT_DEFS = (
    ("torso", "pelvis", "neck", 0.09),
    ("head", "neck", "head", 0.10),
    ("pelvis", "left_hip", "right_hip", 0.06),
    ("shoulders", "left_shoulder", "right_shoulder", 0.05),
    ("left_upper_arm", "left_shoulder", "left_elbow", 0.033),
    ("left_forearm", "left_elbow", "left_wrist", 0.027),
    ("left_thigh", "left_hip", "left_knee", 0.045),
    ("left_shin", "left_knee", "left_ankle", 0.038),
    ("right_upper_arm", "right_shoulder", "right_elbow", 0.033),
    ("right_forearm", "right_elbow", "right_wrist", 0.027),
    ("right_thigh", "right_hip", "right_knee", 0.045),
    ("right_shin", "right_knee", "right_ankle", 0.038),
)

_HUMAN_SEGMENT_BY_GEOM = {"torso_geom": "torso", "head_geom": "head"} | {
    f"{side}_{joint}_geom": f"{side}_{segment}"
    for side in ("left", "right")
    for joint, segment in (("shoulder", "upper_arm"), ("elbow", "forearm"), ("hip", "thigh"), ("knee", "shin"))
}


def _reserved_human_names(human_spec):
    names, _ = joint_contract(human_spec)
    reserved = {
        "human",
        "world",
        "mobile_floor",
        "mobile_root",
        "torso",
        "torso_geom",
        "shoulder_frame",
        "pelvis_frame",
        "head_geom",
        "pelvis_site",
        "neck_site",
        "head_site",
        *ROOT_AXES,
        *(f"{axis}_motor" for axis in ROOT_AXES),
        *names,
        *(f"{name}_motor" for name in names),
        *(seg[0] for seg in HUMAN_SEGMENT_DEFS),
    }
    for side in ("left", "right"):
        for part in ("shoulder", "elbow", "hip", "knee"):
            reserved.update(
                {
                    f"{side}_{part}_body",
                    f"{side}_{part}_site",
                    f"{side}_{part}_geom",
                }
            )
        for tip in ("wrist", "ankle"):
            reserved.add(f"{side}_{tip}_site")
    return reserved


def _number(value, name, low, high):
    if type(value) not in (int, float) or not math.isfinite(value) or not low <= value <= high:
        raise ValueError(f"{name} must be finite and between {low} and {high}")
    return float(value)


def _vector(value, count, name, low, high):
    if not isinstance(value, (list, tuple)) or len(value) != count:
        raise ValueError(f"{name} requires {count} coordinates")
    return [_number(v, name, low, high) for v in value]


def _text(value, name, maximum=500):
    if not isinstance(value, str) or not 1 <= len(value.strip()) <= maximum or len(value) > maximum:
        raise ValueError(f"{name} must be non-empty text of at most {maximum} characters")
    return value


def _same_immutable(value, expected):
    if isinstance(expected, dict):
        return isinstance(value, dict) and value.keys() == expected.keys() and all(
            _same_immutable(value[key], item) for key, item in expected.items())
    if isinstance(expected, list):
        return isinstance(value, list) and len(value) == len(expected) and all(
            _same_immutable(one, two) for one, two in zip(value, expected))
    if type(expected) in (int, float):
        return type(value) in (int, float) and value == expected
    return type(value) is type(expected) and value == expected


def _dynamic_objects(scene):
    return [obj for obj in scene["objects"] if obj["role"] != "static"]


def _quaternion_product(first, second):
    a, b, c, d = first
    e, f, g, h = second
    return [a * e - b * f - c * g - d * h, a * f + b * e + c * h - d * g,
            a * g - b * h + c * e + d * f, a * h + b * g - c * f + d * e]


def _capsule_endpoints(primitive):
    offset = rotation(primitive["quaternion"])[:, 2] * (primitive["length"] / 2.0)
    center = np.asarray(primitive["position"], dtype=float)
    return center - offset, center + offset


def _primitive_bounds(primitive):
    if primitive["shape"] == "box":
        points = corners(primitive)
        return points.min(axis=0), points.max(axis=0)
    if primitive["shape"] == "capsule":
        first, second = _capsule_endpoints(primitive)
        return np.minimum(first, second) - primitive["radius"], np.maximum(first, second) + primitive["radius"]
    center = np.asarray(primitive["position"], dtype=float)
    return center - primitive["radius"], center + primitive["radius"]


def _world_primitives(obj):
    if obj["shape"] != "compound":
        return [obj]
    parent_rotation, center = rotation(obj["quaternion"]), np.asarray(obj["position"], dtype=float)
    return [{**part, "object_id": obj["id"], "role": obj["role"], "color": obj["color"],
             "position": (center + parent_rotation @ np.asarray(part["position"], dtype=float)).tolist(),
             "quaternion": _quaternion_product(obj["quaternion"], part["quaternion"])}
            for part in obj["geometry"]["parts"]]


def _geometry_names(scene):
    for index, obj in enumerate(scene["objects"]):
        if obj["shape"] == "compound":
            for part_index, part in enumerate(obj["geometry"]["parts"]):
                # Object IDs cannot begin with an underscore; generated geom names cannot alias them.
                yield obj, part["id"], f"__weft_part_{index}_{part_index}"
        else:
            yield obj, None, obj["id"]


def _validate_compound(obj):
    geometry = obj["geometry"]
    if (not isinstance(geometry, dict) or set(geometry) != {"profile", "origin", "assumptions", "inertia", "parts"}
            or geometry["profile"] != COMPOUND_PROFILE or geometry["origin"] != "authored_hypothesis"):
        raise ValueError("Compound geometry requires the explicit rigid-compound-v1 authored hypothesis")
    assumptions = geometry["assumptions"]
    if not isinstance(assumptions, list) or not 1 <= len(assumptions) <= 8:
        raise ValueError("Compound geometry requires 1–8 explicit authored assumptions")
    for assumption in assumptions:
        _text(assumption, "compound assumption", 500)
    inertia = geometry["inertia"]
    if (not isinstance(inertia, dict) or set(inertia) != {"mode", "diagonal"}
            or inertia["mode"] != "authored_centered_diagonal"):
        raise ValueError("Compound inertia must be an explicit centered diagonal prior")
    inertia["diagonal"] = _vector(inertia["diagonal"], 3, "compound inertia diagonal", 1e-8, 1e4)
    if 2.0 * max(inertia["diagonal"]) > sum(inertia["diagonal"]) + 1e-12:
        raise ValueError("Compound inertia must satisfy rigid-body triangle inequalities")
    parts = geometry["parts"]
    if not isinstance(parts, list) or not 1 <= len(parts) <= MAX_OBJECT_PARTS:
        raise ValueError("Compound objects require 1–8 rigid primitive parts")
    identities, signatures, local_bounds = set(), set(), []
    maximum = 4.0 if obj["role"] == "static" else 1.2
    for part in parts:
        shape = part.get("shape") if isinstance(part, dict) else None
        dimensions = {"box": {"size"}, "sphere": {"radius"}, "capsule": {"radius", "length"}}
        if (not isinstance(shape, str) or shape not in dimensions
                or set(part) != {"id", "shape", "position", "quaternion"} | dimensions[shape]):
            raise ValueError("Compound parts require exact box, sphere or capsule fields")
        if (not isinstance(part["id"], str) or not re.fullmatch(r"[a-z][a-z0-9_-]{0,31}", part["id"])
                or part["id"] in identities):
            raise ValueError("Compound part IDs must be bounded and unique within their object")
        identities.add(part["id"])
        part["position"] = _vector(part["position"], 3, "compound local position", -4.0, 4.0)
        part["quaternion"] = _vector(part["quaternion"], 4, "compound local quaternion", -1.0, 1.0)
        if not math.isclose(float(np.linalg.norm(part["quaternion"])), 1.0, rel_tol=0, abs_tol=1e-8):
            raise ValueError("Compound part orientation must be a unit quaternion")
        if shape == "box":
            part["size"] = _vector(part["size"], 3, "compound box size", 0.01, maximum)
        else:
            part["radius"] = _number(part["radius"], "compound radius", 0.01, 0.5)
            if shape == "capsule":
                part["length"] = _number(part["length"], "compound capsule centerline length", 0.01, maximum)
        signature = canonical_hash({key: value for key, value in part.items() if key != "id"})
        if signature in signatures:
            raise ValueError("Compound geometry cannot duplicate an identical collision primitive")
        signatures.add(signature)
        local_bounds.append(_primitive_bounds(part))
    low = np.min([item[0] for item in local_bounds], axis=0)
    high = np.max([item[1] for item in local_bounds], axis=0)
    if np.max(high - low) > maximum + 1e-12:
        raise ValueError("Compound geometry exceeds the rigid-object extent envelope")


def _human_only_xml(human_spec, scale):
    """Build a standalone scaled human XML solely to measure exact human-subtree mass."""
    minimal = {
        "human_spec": human_spec,
        "dt": human_spec["dt"],
        "substeps": 16,
        "gravity": [0.0, 0.0, -9.81],
        "contact": {"time_constant": 0.01, "damping_ratio": 1.0},
        "coordinate_frame": {
            "scale_bridge": {
                "body_height_to_world_scale": scale,
                "origin_world": [0.0, 0.0, 0.0],
            }
        },
        "objects": [],
    }
    return _build_world_xml(minimal)


def human_subtree_mass(human_spec, scale=1.0):
    model = mujoco.MjModel.from_xml_string(_human_only_xml(human_spec, float(scale)))
    return float(np.sum(model.body_mass))


def action_contract_for_scene(scene):
    human_spec = scene["human_spec"]
    joint_names, _ = joint_contract(human_spec)
    scale = float(scene["coordinate_frame"]["scale_bridge"]["body_height_to_world_scale"])
    mobility = human_spec["mobility"]
    h_mass = human_subtree_mass(human_spec, scale=scale)
    root_limit = float(min(mobility["maximum_force"], mobility["force_limit_weight_multiple"] * h_mass * 9.81))
    joint_limits = [float(human_spec.get("motor_limits", {}).get(name, 30.0)) for name in joint_names]
    dyn_objs = _dynamic_objects(scene)
    accel = float(scene["task"]["object_goal"]["virtual_acceleration"]) if dyn_objs else 0.

    names = list(joint_names) + list(ROOT_AXES)
    limits = joint_limits + [root_limit] * 3
    mass_unit = scene["coordinate_frame"]["units"]["mass"]
    length_unit = scene["coordinate_frame"]["units"]["length"]
    force_unit = f"{mass_unit} * {length_unit} / second^2"
    torque_unit = f"{mass_unit} * {length_unit}^2 / second^2"
    units = [torque_unit] * len(joint_names) + [force_unit] * 3
    for obj in dyn_objs:
        for axis in ("fx", "fy", "fz"):
            names.append(f"{obj['id']}_{axis}")
            limits.append(float(obj["mass"]) * accel)
            units.append(force_unit)
    return {
        "names": names,
        "human_motor_count": 22,
        "object_force_channels": 3 * len(dyn_objs),
        "dynamic_object_ids": [obj["id"] for obj in dyn_objs],
        "normalized_bounds": [-1.0, 1.0],
        "physical_limits": limits,
        "units": units,
        "substeps_per_action": int(scene["substeps"]),
        "hidden_assistance": False,
    }


def full_reference_horizon(reference, dt):
    """Require one complete 30-Hz reference span; never round away an unsupported endpoint."""
    duration = reference[-1]["time_s"] - reference[0]["time_s"]
    steps = round(duration / dt)
    if (not 1 <= steps <= MAX_FULLSPAN_STEPS or not 0 < duration <= MAX_FULLSPAN_SECONDS
            or not math.isclose(steps * dt, duration, rel_tol=0, abs_tol=1e-9)):
        raise ValueError("shared-world-v2 requires a complete grid-aligned reference of 1–1800 actions within 60 seconds")
    return steps


def validate_shared_world(scene):
    """Validate legacy coupled worlds or an explicit complete-reference person-only world."""
    required = {
        "schema_version",
        "profile",
        "physical_validation",
        "dt",
        "substeps",
        "bounds",
        "gravity",
        "contact",
        "coordinate_frame",
        "source_binding",
        "human_spec",
        "objects",
        "controlled_id",
        "target",
        "task",
        "assumptions",
    }
    optional = {"action_contract", "learned_policy_available"}
    human_only = isinstance(scene, dict) and scene.get("profile") == HUMAN_ONLY_PROFILE
    if human_only:
        required.add("inventory")
    if not isinstance(scene, dict) or not required <= set(scene) or set(scene) - (required | optional):
        raise ValueError("Unsupported shared-world-v1 scene fields")
    if (type(scene["schema_version"]) is not int or scene["schema_version"] != 1
            or scene["profile"] not in (PROFILE, *FULLSPAN_PROFILES)):
        raise ValueError("Unsupported shared-world schema_version or profile")
    if scene["physical_validation"] != "unverified":
        raise ValueError("shared-world-v1 must preserve physical_validation='unverified'")
    if "learned_policy_available" in scene and type(scene["learned_policy_available"]) is not bool:
        raise ValueError("learned_policy_available must be a boolean when supplied")

    result = deepcopy(scene)
    result["dt"] = _number(result["dt"], "dt", 1.0 / 30.0 - 1e-12, 1.0 / 30.0 + 1e-12)
    result["dt"] = 1.0 / 30.0
    if type(result["substeps"]) is not int or not 8 <= result["substeps"] <= 64:
        raise ValueError("shared-world-v1 substeps must be an integer between 8 and 64")
    result["bounds"] = _vector(result["bounds"], 3, "bounds", 0.8, 6.0)
    bx, by, bz = result["bounds"]
    result["gravity"] = _vector(result["gravity"], 3, "gravity", -20.0, 20.0)
    if float(np.linalg.norm(result["gravity"])) > 20.0:
        raise ValueError("Gravity exceeds the integration envelope")

    contact = result["contact"]
    if not isinstance(contact, dict) or set(contact) != {"time_constant", "damping_ratio"}:
        raise ValueError("Contact solver assumptions must specify time_constant and damping_ratio")
    contact["time_constant"] = _number(contact["time_constant"], "contact time_constant", 0.005, 0.05)
    contact["damping_ratio"] = _number(contact["damping_ratio"], "contact damping_ratio", 0.5, 2.0)

    human_spec = validate_human(result["human_spec"])
    if human_spec.get("profile") != "exercise-v3":
        raise ValueError("shared-world-v1 requires a mobile exercise-v3 human_spec")
    if "camera_reference" in human_spec:
        raise ValueError("Camera-world human candidates require trajectory validation before shared-world coupling")
    if not math.isclose(human_spec["dt"], 1.0 / 30.0, rel_tol=0, abs_tol=1e-12):
        raise ValueError("shared-world-v1 requires a 30-Hz human_spec")
    if "ground_plane" in human_spec["mobility"]:
        expected_gravity = -9.81 * ground_normal(human_spec)
        if not np.allclose(result["gravity"], expected_gravity, atol=1e-5, rtol=0):
            raise ValueError("Shared world gravity must align with human_spec ground_plane normal")
    result["human_spec"] = human_spec

    frame = result["coordinate_frame"]
    if not isinstance(frame, dict) or set(frame) != {
        "axes",
        "units",
        "calibration_status",
        "scale_bridge",
        "clock_binding",
    }:
        raise ValueError("coordinate_frame must declare axes, units, calibration_status, scale_bridge, and clock_binding")
    if frame["axes"] != "x_right_y_depth_z_up":
        raise ValueError("shared-world-v1 requires right-handed axes 'x_right_y_depth_z_up'")
    if frame["calibration_status"] not in CALIBRATION_STATUSES:
        raise ValueError(f"Unsupported calibration_status: {frame['calibration_status']!r}")
    if frame["calibration_status"] == "calibrated":
        raise ValueError("Calibrated shared-world frames require a verified calibration binding; currently unsupported")
    units = frame["units"]
    if not isinstance(units, dict) or set(units) != {
        "length",
        "mass",
        "time",
        "orientation",
        "angular_velocity",
    }:
        raise ValueError("coordinate_frame.units fields differ from contract")
    _text(units["length"], "length unit", 80)
    _text(units["mass"], "mass unit", 80)
    if units["time"] != "second" or units["orientation"] != "quaternion_wxyz" or units["angular_velocity"] != "world_radians_per_second":
        raise ValueError("coordinate_frame.units must use second, quaternion_wxyz, and world_radians_per_second")

    bridge = frame["scale_bridge"]
    if not isinstance(bridge, dict) or set(bridge) != {
        "body_height_to_world_scale",
        "origin_world",
        "method",
        "uncertainty_note",
    }:
        raise ValueError("scale_bridge must specify body_height_to_world_scale, origin_world, method, and uncertainty_note")
    bridge["body_height_to_world_scale"] = _number(
        bridge["body_height_to_world_scale"], "body_height_to_world_scale", 0.5, 2.0
    )
    bridge["origin_world"] = _vector(bridge["origin_world"], 3, "origin_world", 0.0, 6.0)
    ox, oy, oz = bridge["origin_world"]
    if not (0.0 <= ox <= bx and 0.0 <= oy <= by and 0.0 <= oz <= 0.5 * bz):
        raise ValueError("scale_bridge.origin_world must lie inside the shared world bounds")
    _text(bridge["method"], "scale_bridge.method", 160)
    _text(bridge["uncertainty_note"], "scale_bridge.uncertainty_note", 500)

    clock = frame["clock_binding"]
    if not isinstance(clock, dict) or set(clock) != {
        "control_dt_s",
        "physics_substeps",
        "source_fps",
        "source_time_offset_s",
        "alignment_method",
    }:
        raise ValueError("clock_binding fields differ from contract")
    clock["control_dt_s"] = _number(clock["control_dt_s"], "control_dt_s", 1.0 / 30.0 - 1e-12, 1.0 / 30.0 + 1e-12)
    clock["control_dt_s"] = 1.0 / 30.0
    if type(clock["physics_substeps"]) is not int or clock["physics_substeps"] != result["substeps"]:
        raise ValueError("clock_binding.physics_substeps must match scene substeps")
    clock["source_fps"] = _number(clock["source_fps"], "source_fps", 5.0, 240.0)
    clock["source_time_offset_s"] = _number(clock["source_time_offset_s"], "source_time_offset_s", 0.0, 3600.0)
    _text(clock["alignment_method"], "clock_binding.alignment_method", 200)

    if human_only:
        if result["objects"] != [] or not isinstance(result["objects"], list):
            raise ValueError("shared-world-v3 requires an empty objects list")
    elif not isinstance(result["objects"], list) or not 1 <= len(result["objects"]) <= 12:
        raise ValueError("shared-world-v1 requires 1–12 rigid objects alongside the articulated human")
    reserved = _reserved_human_names(human_spec)
    identifiers = set()
    primitive_count = 0
    for obj in result["objects"]:
        obj_required = {
            "id",
            "label",
            "role",
            "shape",
            "position",
            "quaternion",
            "velocity",
            "angular_velocity",
            "mass",
            "friction",
            "color",
            "provenance",
        }
        shape = obj.get("shape") if isinstance(obj, dict) else None
        geometry_fields = {"box": {"size"}, "sphere": {"radius"}, "compound": {"geometry"}}
        if not isinstance(shape, str) or shape not in geometry_fields or set(obj) != obj_required | geometry_fields[shape]:
            raise ValueError("Shared-world rigid object fields differ from contract")
        if (
            not isinstance(obj["id"], str)
            or not re.fullmatch(r"[a-z][a-z0-9_-]{0,63}", obj["id"])
            or obj["id"] in identifiers
            or obj["id"] in reserved
        ):
            raise ValueError(f"Rigid object ID {obj.get('id')!r} must be unique and disjoint from human identifiers")
        identifiers.add(obj["id"])
        if obj["role"] not in {"active", "passive", "static"}:
            raise ValueError("Invalid rigid object role")
        _text(obj["label"], "object label", 120)
        if not isinstance(obj["color"], str) or not re.fullmatch(r"#[0-9A-Fa-f]{6}", obj["color"]):
            raise ValueError("Object color requires hexadecimal RGB")
        obj["position"] = _vector(obj["position"], 3, "object position", 0.0, 6.0)
        obj["quaternion"] = _vector(obj["quaternion"], 4, "object quaternion", -1.0, 1.0)
        if not math.isclose(float(np.linalg.norm(obj["quaternion"])), 1.0, rel_tol=0, abs_tol=1e-8):
            raise ValueError("Object orientation must be a unit quaternion")
        obj["velocity"] = _vector(obj["velocity"], 3, "object velocity", -MAX_SPEED, MAX_SPEED)
        obj["angular_velocity"] = _vector(obj["angular_velocity"], 3, "object angular_velocity", -MAX_SPIN, MAX_SPIN)
        if float(np.linalg.norm(obj["velocity"])) > MAX_SPEED or float(np.linalg.norm(obj["angular_velocity"])) > MAX_SPIN:
            raise ValueError("Initial object speed or spin exceeds its envelope")
        if obj["role"] == "static" and any(obj["velocity"] + obj["angular_velocity"]):
            raise ValueError("Static geometry cannot have initial motion")
        obj["mass"] = _number(obj["mass"], "object mass", 0.05, 200.0)
        obj["friction"] = _vector(obj["friction"], 3, "object friction", 0.0, 2.0)
        if shape == "box":
            max_dim = 4.0 if obj["role"] == "static" else 1.2
            obj["size"] = _vector(obj["size"], 3, "box size", 0.01, max_dim)
            pts = corners(obj)
            low, high = pts.min(axis=0), pts.max(axis=0)
        elif shape == "sphere":
            obj["radius"] = _number(obj["radius"], "sphere radius", 0.01, 0.5)
            low = np.asarray(obj["position"]) - obj["radius"]
            high = np.asarray(obj["position"]) + obj["radius"]
        else:
            _validate_compound(obj)
            part_bounds = [_primitive_bounds(part) for part in _world_primitives(obj)]
            low = np.min([item[0] for item in part_bounds], axis=0)
            high = np.max([item[1] for item in part_bounds], axis=0)
        primitive_count += len(obj["geometry"]["parts"]) if shape == "compound" else 1
        if primitive_count > MAX_RIGID_PRIMITIVES:
            raise ValueError("Shared-world scenes support at most 32 rigid collision primitives")
        if (low < 0.0).any() or (high > np.asarray(result["bounds"])).any():
            raise ValueError(f"Initial rigid object {obj['id']!r} exceeds shared-world bounds")
        if not isinstance(obj["provenance"], dict) or not obj["provenance"]:
            raise ValueError("Rigid object provenance must be a non-empty object")

    dynamic_ids = [obj["id"] for obj in result["objects"] if obj["role"] != "static"]
    if not dynamic_ids and not human_only:
        raise ValueError("shared-world-v1 requires at least one non-static rigid object")

    controlled_id = result["controlled_id"]
    if controlled_id != "human" and controlled_id not in dynamic_ids:
        raise ValueError("controlled_id must be 'human' or a non-static rigid object ID")
    for obj in result["objects"]:
        if obj["role"] == "static":
            continue
        expected_role = "active" if obj["id"] == controlled_id else "passive"
        if obj["role"] != expected_role:
            raise ValueError(
                f"Dynamic object {obj['id']!r} role must be {expected_role!r} when controlled_id={controlled_id!r}"
            )

    source_binding = result["source_binding"]
    binding_fields = {
        "mode",
        "source_sha256",
        "total_source_frames",
        "entity_bindings",
        "evidence_documents",
    }
    if (not isinstance(source_binding, dict) or not binding_fields <= set(source_binding)
            or not set(source_binding) <= binding_fields | {"source_equivalences"}):
        raise ValueError("source_binding fields differ from contract")
    if source_binding["mode"] not in SOURCE_MODES:
        raise ValueError(f"Unsupported source_binding.mode: {source_binding['mode']!r}")
    validate_equivalence_bindings(source_binding)
    documents = source_binding["evidence_documents"]
    if not isinstance(documents, dict) or len(documents) > 8:
        raise ValueError("source_binding.evidence_documents must map at most eight hashes to JSON paths")
    seen_paths = set()
    for document_sha, path in documents.items():
        if not isinstance(document_sha, str) or not re.fullmatch(r"[0-9a-f]{64}", document_sha):
            raise ValueError("Evidence document identity must be an exact SHA-256")
        if (
            not isinstance(path, str)
            or not 1 <= len(path) <= 240
            or any(ord(char) < 32 for char in path)
            or "\\" in path
            or PurePosixPath(path).is_absolute()
            or any(part in ("", ".", "..") for part in path.split("/"))
            or PurePosixPath(path).suffix != ".json"
            or path in seen_paths
        ):
            raise ValueError("Evidence documents require distinct safe relative JSON paths")
        seen_paths.add(path)
    if source_binding["mode"] == "analytic_fixture":
        if (
            source_binding["source_sha256"] is not None
            or source_binding["total_source_frames"] is not None
            or source_binding["entity_bindings"] != []
            or documents != {}
        ):
            raise ValueError("analytic_fixture source_binding must leave source_sha256/frames None and bindings/documents empty")
    else:
        if not documents:
            raise ValueError("source_qualified source_binding requires hash-pinned evidence documents")
        sha = source_binding["source_sha256"]
        if not isinstance(sha, str) or not re.fullmatch(r"[0-9a-f]{64}", sha):
            raise ValueError("source_qualified source_binding requires a 64-hex source_sha256")
        if human_spec.get("source_sha256") is not None and human_spec["source_sha256"] != sha:
            raise ValueError("human_spec.source_sha256 differs from source_binding.source_sha256")
        total_frames = source_binding["total_source_frames"]
        if type(total_frames) is not int or not 2 <= total_frames <= 18000:
            raise ValueError("total_source_frames must be an integer between 2 and 18000")
        bindings = source_binding["entity_bindings"]
        if not isinstance(bindings, list) or len(bindings) != 1 + len(result["objects"]):
            raise ValueError("source_qualified entity_bindings must cover 'human' and every rigid object")
        expected_world_ids = {"human", *identifiers}
        seen_world_ids = set()
        seen_obs_ids = set()
        for row in bindings:
            if not isinstance(row, dict) or set(row) != {
                "world_entity_id",
                "entity_kind",
                "observation_entity_id",
                "observation_document_sha256",
                "provenance_mode",
                "observed_slots",
                "missing_slots",
                "pre_init_slots",
            }:
                raise ValueError("entity_bindings entry fields differ from contract")
            wid = row["world_entity_id"]
            if wid not in expected_world_ids or wid in seen_world_ids:
                raise ValueError(f"Duplicate or unknown world_entity_id in source_binding: {wid!r}")
            seen_world_ids.add(wid)
            expected_kind = "articulated_human" if wid == "human" else "rigid_object"
            if row["entity_kind"] != expected_kind:
                raise ValueError(f"entity_kind for {wid!r} must be {expected_kind!r}")
            if row["provenance_mode"] not in PROVENANCE_MODES:
                raise ValueError(f"Unsupported provenance_mode: {row['provenance_mode']!r}")
            document_sha = row["observation_document_sha256"]
            oid = row["observation_entity_id"]
            if row["provenance_mode"] == "authored_addition":
                if document_sha is not None or oid is not None:
                    raise ValueError("Authored additions must not invent observation identities")
                if row["observed_slots"] != 0 or row["pre_init_slots"] != 0:
                    raise ValueError("Authored additions cannot claim observed or pre-initialization slots")
            else:
                if not isinstance(document_sha, str) or document_sha not in documents:
                    raise ValueError("Entity observation_document_sha256 must name a supplied evidence document")
                oid = _text(oid, "observation_entity_id", 160)
                identity = (document_sha, oid)
                if identity in seen_obs_ids:
                    raise ValueError(f"Duplicate observation identity across coupled entities: {identity!r}")
                seen_obs_ids.add(identity)
            slots = []
            for slot_key in ("observed_slots", "missing_slots", "pre_init_slots"):
                val = row[slot_key]
                if type(val) is not int or val < 0:
                    raise ValueError(f"{slot_key} must be a nonnegative integer")
                slots.append(val)
            if sum(slots) != total_frames:
                raise ValueError(
                    f"Slot conservation failed for {wid!r}: {sum(slots)} != total_source_frames ({total_frames})"
                )
            if row["provenance_mode"] == "observed_track" and row["observed_slots"] < 1:
                raise ValueError(f"observed_track entity {wid!r} must have at least one observed slot")
        if seen_world_ids != expected_world_ids:
            raise ValueError("entity_bindings must cover every coupled entity")

    if human_only:
        from weft.person_inventory import validate_person_inventory

        result["inventory"] = validate_person_inventory(
            result["inventory"], source_sha256=source_binding["source_sha256"],
            analytic=source_binding["mode"] == "analytic_fixture",
        )
        if human_spec.get("source_sha256") != source_binding["source_sha256"]:
            raise ValueError("Person-only inventory, human and source identities must match")
        if result["target"] is not None:
            raise ValueError("shared-world-v3 has no object target")
    else:
        result["target"] = _vector(result["target"], 3, "target", 0.0, 6.0)
        if (np.asarray(result["target"]) > np.asarray(result["bounds"])).any():
            raise ValueError("Target lies outside shared-world bounds")

    task = result["task"]
    if not isinstance(task, dict) or set(task) != {
        "type",
        "authored_addition",
        "start_offset_s",
        "horizon_steps",
        "human_limits",
        "object_goal",
    }:
        raise ValueError("shared-world-v1 task fields differ from contract")
    task_type = "human_motion_v1" if human_only else "coupled_human_object_v1"
    if task["type"] != task_type or task["authored_addition"] is not True:
        raise ValueError(f"{result['profile']} task must declare type={task_type!r} and authored_addition=True")
    task["start_offset_s"] = _number(task["start_offset_s"], "start_offset_s", 0.0, 60.0)
    maximum = MAX_FULLSPAN_STEPS if result["profile"] in FULLSPAN_PROFILES else 180
    if type(task["horizon_steps"]) is not int or not 1 <= task["horizon_steps"] <= maximum:
        raise ValueError(f"{result['profile']} horizon_steps must be an integer in 1–{maximum}")
    ref_times = [row["time_s"] for row in human_spec["reference"]]
    if result["profile"] in FULLSPAN_PROFILES:
        horizon = full_reference_horizon(human_spec["reference"], result["dt"])
        if task["start_offset_s"] != 0 or task["horizon_steps"] != horizon:
            raise ValueError(f"{result['profile']} must preserve the complete original reference interval without offsets or clipping")
    if ref_times[0] + task["start_offset_s"] + task["horizon_steps"] * result["dt"] > ref_times[-1] + 1e-9:
        raise ValueError("shared-world-v1 task window exceeds the available human reference trajectory")
    if task["human_limits"] != SUCCESS_LIMITS:
        raise ValueError("shared-world-v1 task.human_limits must match unmodified whole-body SUCCESS_LIMITS")

    goal = task["object_goal"]
    if human_only:
        if goal is not None:
            raise ValueError("shared-world-v3 has no object goal")
    elif not isinstance(goal, dict) or set(goal) != {
        "target_object_id",
        "position_tolerance",
        "speed_tolerance",
        "virtual_acceleration",
        "require_human_contact",
    }:
        raise ValueError("task.object_goal fields differ from contract")
    if not human_only:
        if goal["target_object_id"] not in dynamic_ids:
            raise ValueError("task.object_goal.target_object_id must name a non-static rigid object")
        goal["position_tolerance"] = _number(goal["position_tolerance"], "position_tolerance", 0.01, 0.3)
        goal["speed_tolerance"] = _number(goal["speed_tolerance"], "speed_tolerance", 0.01, 1.0)
        goal["virtual_acceleration"] = _number(goal["virtual_acceleration"], "virtual_acceleration", 0.1, 15.0)
        if type(goal["require_human_contact"]) is not bool:
            raise ValueError("task.object_goal.require_human_contact must be a boolean")

    if (
        not isinstance(result["assumptions"], list)
        or not 1 <= len(result["assumptions"]) <= 100
        or any(not isinstance(item, str) or not 1 <= len(item) <= 2000 for item in result["assumptions"])
    ):
        raise ValueError("shared-world-v1 assumptions must be 1–100 bounded strings")

    expected_contract = action_contract_for_scene(result)
    if "action_contract" in result:
        supplied = result["action_contract"]
        if not isinstance(supplied, dict) or set(supplied) != set(expected_contract):
            raise ValueError("Supplied action_contract fields differ from shared-world-v1 contract")
        for key in (
            "names",
            "human_motor_count",
            "object_force_channels",
            "dynamic_object_ids",
            "normalized_bounds",
            "units",
            "substeps_per_action",
            "hidden_assistance",
        ):
            if supplied[key] != expected_contract[key]:
                raise ValueError(f"Supplied action_contract[{key!r}] does not match scene definition")
        if len(supplied["physical_limits"]) != len(expected_contract["physical_limits"]) or not np.allclose(
            supplied["physical_limits"], expected_contract["physical_limits"], atol=1e-6, rtol=0
        ):
            raise ValueError("Supplied action_contract physical_limits differ from scene definition")
    result["action_contract"] = expected_contract

    # Formatting whitespace is not scene data; bound the compact UTF-8 representation.
    if len(json.dumps(result, allow_nan=False, separators=(",", ":")).encode("utf-8")) > 500_000:
        raise ValueError("shared-world-v1 scene metadata exceeds the 500 KB bound")
    return result


def _build_world_xml(scene):
    human_spec = scene["human_spec"]
    lengths = human_spec["lengths"]
    joint_names, joint_limits = joint_contract(human_spec)
    mobility = human_spec["mobility"]
    bridge = scene["coordinate_frame"]["scale_bridge"]
    scale = float(bridge["body_height_to_world_scale"])
    ox, oy, oz = map(float, bridge["origin_world"])

    root = ET.Element("mujoco", model="weft_shared_world")
    ET.SubElement(root, "compiler", angle="radian")
    ET.SubElement(
        root,
        "option",
        timestep=str(float(scene["dt"]) / int(scene["substeps"])),
        gravity=" ".join(map(str, scene["gravity"])),
        integrator="implicitfast",
        cone="elliptic",
        solver="Newton",
        iterations="40",
        tolerance="1e-9",
    )
    ET.SubElement(root, "size", memory="32M")

    default = ET.SubElement(root, "default")
    ET.SubElement(default, "joint", type="hinge", damping="1.5", armature="0.08", limited="true")
    solref = f"{scene['contact']['time_constant']} {scene['contact']['damping_ratio']}"
    ET.SubElement(
        default,
        "geom",
        type="capsule",
        size=str(0.04 * scale),
        density="700",
        contype="2",
        conaffinity="5",
        condim="4",
        solref=solref,
        solimp="0.95 0.99 0.001",
        rgba="0.34 0.52 0.54 1",
    )

    world = ET.SubElement(root, "worldbody")
    floor_attrs = {}
    if "ground_plane" in mobility:
        normal = ground_normal(human_spec)
        quat = np.array([1.0 + normal[2], -normal[1], normal[0], 0.0])
        quat /= np.linalg.norm(quat)
        floor_attrs["quat"] = " ".join(map(str, quat))
    ET.SubElement(
        world,
        "geom",
        name="mobile_floor",
        type="plane",
        pos=f"{ox} {oy} {oz}",
        size="8 8 0.1",
        contype="1",
        conaffinity="6",
        friction="0.8 0.01 0.001",
        rgba="0.85 0.80 0.72 1",
        **floor_attrs,
    )

    mobile_root = ET.SubElement(
        world,
        "body",
        name="mobile_root",
        pos=f"{ox} {oy} {oz + scale * float(mobility['base_height'])}",
    )
    ET.SubElement(
        mobile_root,
        "inertial",
        pos="0 0 0",
        mass=str(float(mobility["root_mass"]) * (scale**3)),
        diaginertia="0.001 0.001 0.001",
    )
    slide_low, slide_high = mobility["slide_limits"]
    for axis_name, axis_vec in zip(ROOT_AXES, ("1 0 0", "0 1 0", "0 0 1")):
        ET.SubElement(
            mobile_root,
            "joint",
            name=axis_name,
            type="slide",
            axis=axis_vec,
            limited="true",
            range=f"{slide_low * scale} {slide_high * scale}",
            damping="0",
            armature="0",
        )

    orientation = np.asarray(human_spec.get("body_frame", {}).get("rotation", np.eye(3)), dtype=float)
    axes = " ".join(str(val) for val in np.concatenate((orientation[:, 0], orientation[:, 1])))
    torso = ET.SubElement(mobile_root, "body", name="torso", pos="0 0 0", xyaxes=axes)
    for jname, jaxis in (("torso_twist", "0 0 1"), ("torso_lateral", "0 1 0"), ("torso_flex", "1 0 0")):
        ET.SubElement(torso, "joint", name=jname, axis=jaxis, range=" ".join(map(str, joint_limits[jname])))

    torso_len = lengths["torso"] * scale
    sh_half = (lengths["shoulder_width"] / 2.0) * scale
    hip_half = (lengths["hip_width"] / 2.0) * scale
    ET.SubElement(torso, "geom", name="torso_geom", fromto=f"0 0 0 0 0 {torso_len}", size=str(0.09 * scale))
    ET.SubElement(
        torso,
        "geom",
        name="shoulder_frame",
        fromto=f"{-sh_half} 0 {torso_len} {sh_half} 0 {torso_len}",
        size=str(0.05 * scale),
    )
    ET.SubElement(
        torso,
        "geom",
        name="pelvis_frame",
        fromto=f"{-hip_half} 0 0 {hip_half} 0 0",
        size=str(0.06 * scale),
    )
    ET.SubElement(torso, "site", name="pelvis_site", pos="0 0 0", size=str(0.01 * scale))
    ET.SubElement(torso, "site", name="neck_site", pos=f"0 0 {torso_len}", size=str(0.01 * scale))
    ET.SubElement(torso, "site", name="head_site", pos=f"0 0 {torso_len + 0.18 * scale}", size=str(0.01 * scale))
    ET.SubElement(
        torso,
        "geom",
        name="head_geom",
        type="sphere",
        pos=f"0 0 {torso_len + 0.13 * scale}",
        size=str(0.10 * scale),
    )

    for side, sign in (("left", -1), ("right", 1)):
        for limb, proximal, distal, len1_key, len2_key, half_w, elev in (
            ("arm", "shoulder", "elbow", "upper_arm", "forearm", sh_half, torso_len),
            ("leg", "hip", "knee", "thigh", "shin", hip_half, 0.0),
        ):
            l1 = lengths[len1_key] * scale
            l2 = lengths[len2_key] * scale
            base = ET.SubElement(torso, "body", name=f"{side}_{proximal}_body", pos=f"{sign * half_w} 0 {elev}")
            joint = f"{side}_{proximal}"
            for suffix, axis_vec in (("twist", "0 0 1"), ("lateral", "0 1 0")):
                extra = f"{joint}_{suffix}"
                ET.SubElement(base, "joint", name=extra, axis=axis_vec, range=" ".join(map(str, joint_limits[extra])))
            ET.SubElement(base, "joint", name=joint, axis="1 0 0", range=" ".join(map(str, joint_limits[joint])))
            ET.SubElement(base, "site", name=f"{joint}_site", pos="0 0 0", size=str(0.01 * scale))
            ET.SubElement(
                base,
                "geom",
                name=f"{joint}_geom",
                fromto=f"0 0 0 0 0 {-l1}",
                size=str((0.045 if limb == "leg" else 0.033) * scale),
            )
            child = ET.SubElement(base, "body", name=f"{side}_{distal}_body", pos=f"0 0 {-l1}")
            child_joint = f"{side}_{distal}"
            ET.SubElement(
                child,
                "joint",
                name=child_joint,
                axis="-1 0 0" if distal == "knee" else "1 0 0",
                range=" ".join(map(str, joint_limits[child_joint])),
            )
            ET.SubElement(child, "site", name=f"{child_joint}_site", pos="0 0 0", size=str(0.01 * scale))
            ET.SubElement(
                child,
                "geom",
                name=f"{child_joint}_geom",
                fromto=f"0 0 0 0 0 {-l2}",
                size=str((0.038 if limb == "leg" else 0.027) * scale),
            )
            tip = "ankle" if limb == "leg" else "wrist"
            ET.SubElement(child, "site", name=f"{side}_{tip}_site", pos=f"0 0 {-l2}", size=str(0.01 * scale))

    for object_index, obj in enumerate(scene["objects"]):
        body = ET.SubElement(
            world,
            "body",
            name=obj["id"],
            pos=" ".join(map(str, obj["position"])),
            quat=" ".join(map(str, obj["quaternion"])),
        )
        is_static = obj["role"] == "static"
        if not is_static:
            ET.SubElement(body, "freejoint", name=f"{obj['id']}_joint")
        if obj["shape"] == "compound":
            # Collision parts never add mass or silently move the authored body-frame origin.
            ET.SubElement(body, "inertial", pos="0 0 0", mass=str(obj["mass"]),
                          diaginertia=" ".join(map(str, obj["geometry"]["inertia"]["diagonal"])))
            for part_index, part in enumerate(obj["geometry"]["parts"]):
                size = (np.asarray(part["size"], dtype=float) / 2.0 if part["shape"] == "box" else
                        [part["radius"], part["length"] / 2.0] if part["shape"] == "capsule" else [part["radius"]])
                ET.SubElement(body, "geom", name=f"__weft_part_{object_index}_{part_index}", type=part["shape"],
                              pos=" ".join(map(str, part["position"])), quat=" ".join(map(str, part["quaternion"])),
                              size=" ".join(map(str, size)), mass="0", friction=" ".join(map(str, obj["friction"])),
                              contype="1" if is_static else "4", conaffinity="6" if is_static else "7", condim="6",
                              solref=solref, solimp="0.99 0.99 0.001", margin="0")
            continue
        size = np.asarray(obj["size"], dtype=float) / 2.0 if obj["shape"] == "box" else [float(obj["radius"])]
        ET.SubElement(
            body,
            "geom",
            name=obj["id"],
            type=obj["shape"],
            size=" ".join(map(str, size)),
            mass=str(obj["mass"]),
            friction=" ".join(map(str, obj["friction"])),
            contype="1" if is_static else "4",
            conaffinity="6" if is_static else "7",
            condim="6",
            solref=solref,
            solimp="0.99 0.99 0.001",
            margin="0",
        )

    actuator = ET.SubElement(root, "actuator")
    for jname in joint_names:
        limit = float(human_spec.get("motor_limits", {}).get(jname, 30.0))
        ET.SubElement(
            actuator,
            "motor",
            name=f"{jname}_motor",
            joint=jname,
            ctrllimited="true",
            ctrlrange=f"{-limit} {limit}",
            gear="1",
        )
    max_root_force = float(mobility["maximum_force"])
    for axis_name in ROOT_AXES:
        ET.SubElement(
            actuator,
            "motor",
            name=f"{axis_name}_motor",
            joint=axis_name,
            gear="1",
            ctrllimited="true",
            ctrlrange=f"{-max_root_force} {max_root_force}",
            forcelimited="true",
            forcerange=f"{-max_root_force} {max_root_force}",
        )
    return ET.tostring(root, encoding="unicode")


def shared_world_xml(scene):
    validated = validate_shared_world(scene)
    return _build_world_xml(validated)


def _initial_intersections(scene, model, data, human_geoms):
    floor = model.geom("mobile_floor").id
    objects = [(model.geom(name).id, obj["id"], obj["role"], part) for obj, part, name in _geometry_names(scene)]
    compound = any(obj["shape"] == "compound" for obj in scene["objects"])
    intersections = []
    scale = scene["coordinate_frame"]["scale_bridge"]["body_height_to_world_scale"]
    floor_limit = SUCCESS_LIMITS["max_floor_penetration_model_units"] * scale
    for hgid, hname in human_geoms.items():
        distance = float(mujoco.mj_geomDistance(model, data, hgid, floor, 0.05, None))
        if distance < -floor_limit:
            intersections.append({"pair": [hname, "mobile_floor"], "kind": "human_floor", "distance": distance})
    for index, (gid, name, role, part) in enumerate(objects):
        if role != "static":
            distance = float(mujoco.mj_geomDistance(model, data, gid, floor, 0.05, None))
            if distance < -1e-5:
                intersections.append({"pair": [name, "mobile_floor"], "kind": "object_floor", "distance": distance,
                                      **({"parts": [part, None]} if compound else {})})
        for hgid, hname in human_geoms.items():
            distance = float(mujoco.mj_geomDistance(model, data, gid, hgid, 0.05, None))
            if distance < -1e-5:
                kind = "human_static" if role == "static" else "human_object"
                intersections.append({"pair": [hname, name], "kind": kind, "distance": distance,
                                      **({"parts": [None, part]} if compound else {})})
        for other_gid, other_name, other_role, other_part in objects[index + 1 :]:
            if name == other_name or role == other_role == "static":
                continue
            distance = float(mujoco.mj_geomDistance(model, data, gid, other_gid, 0.05, None))
            if distance < -1e-5:
                kind = "object_static" if "static" in (role, other_role) else "object_object"
                intersections.append({"pair": [name, other_name], "kind": kind, "distance": distance,
                                      **({"parts": [part, other_part]} if compound else {})})
    return intersections


def audit_shared_world_feasibility(scene):
    """Zero-physics kinematic preflight auditing human floor/static consistency and initial object clearances."""
    validated = validate_shared_world(scene)
    human_spec = validated["human_spec"]
    scale = float(validated["coordinate_frame"]["scale_bridge"]["body_height_to_world_scale"])
    model = mujoco.MjModel.from_xml_string(_build_world_xml(validated))
    data = mujoco.MjData(model)

    joint_names, _ = joint_contract(human_spec)
    q_indices = np.array([int(model.jnt_qposadr[model.joint(name).id]) for name in (*joint_names, *ROOT_AXES)])
    ref_times = np.array([row["time_s"] for row in human_spec["reference"]], dtype=float)
    duration = float(ref_times[-1] - ref_times[0])
    if not 0.0 < duration <= 60.0:
        raise ValueError("Shared-world reference feasibility duration must be within 60 seconds")
    grid_count = math.floor(duration * 30.0 + 1e-9) + 1
    if len(ref_times) > 900 or grid_count > 1801:
        raise ValueError("Shared-world reference feasibility exceeds pose/grid bounds")

    ref_angles = np.array([[row["joints"][name] for row in human_spec["reference"]] for name in joint_names], dtype=float)
    ref_roots = np.array([row["translation"] for row in human_spec["root_reference"]], dtype=float).T * scale
    ref_matrix = np.vstack((ref_angles, ref_roots))

    floor_geom = model.geom("mobile_floor").id
    static_geoms = {model.geom(name).id: obj["id"] for obj, _, name in _geometry_names(validated) if obj["role"] == "static"}
    dynamic_geoms = {model.geom(name).id: obj["id"] for obj, _, name in _geometry_names(validated) if obj["role"] != "static"}
    part_by_geom = {model.geom(name).id: part for _, part, name in _geometry_names(validated)}
    compound = any(obj["shape"] == "compound" for obj in validated["objects"])
    human_geoms = {
        gid: mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid)
        for gid in range(model.ngeom)
        if int(model.geom_contype[gid]) & 2
    }

    def _set_pose_at(timestamp):
        mujoco.mj_resetData(model, data)
        for idx in range(22):
            data.qpos[q_indices[idx]] = float(np.interp(timestamp, ref_times, ref_matrix[idx]))
        data.time = float(timestamp)
        mujoco.mj_forward(model, data)

    # Check initial reset pose at start_offset_s for cross-entity interpenetration.
    start_ts = float(ref_times[0] + validated["task"]["start_offset_s"])
    _set_pose_at(start_ts)
    initial_intersections = _initial_intersections(validated, model, data, human_geoms)

    schedule = [("source_pose", float(t), i) for i, t in enumerate(ref_times)]
    schedule += [
        ("control_grid", min(float(ref_times[-1]), float(ref_times[0] + i / 30.0)), None)
        for i in range(grid_count)
    ]
    rows = []
    limit_model_units = SUCCESS_LIMITS["max_floor_penetration_model_units"]
    for kind, timestamp, source_idx in schedule:
        _set_pose_at(timestamp)
        floor_contacts = []
        static_conflicts = []
        for c_idx in range(data.ncon):
            con = data.contact[c_idx]
            g1, g2 = int(con.geom1), int(con.geom2)
            if floor_geom in (g1, g2):
                other = g2 if g1 == floor_geom else g1
                if other in human_geoms:
                    floor_contacts.append(
                        {
                            "geom": human_geoms[other],
                            "distance_world": float(con.dist),
                            "penetration_model_units": max(0.0, -float(con.dist)) / scale,
                            "position": con.pos.tolist(),
                        }
                    )
            elif (g1 in human_geoms and g2 in static_geoms) or (g2 in human_geoms and g1 in static_geoms):
                hg = g1 if g1 in human_geoms else g2
                sg = g2 if hg == g1 else g1
                if float(con.dist) < -1e-3:
                    static_conflicts.append(
                        {
                            "human_geom": human_geoms[hg],
                            "static_object": static_geoms[sg],
                            **({"part_id": part_by_geom[sg]} if compound else {}),
                            "distance_world": float(con.dist),
                            "position": con.pos.tolist(),
                        }
                    )
        max_pen = max((c["penetration_model_units"] for c in floor_contacts), default=0.0)
        passes = max_pen <= limit_model_units and not static_conflicts
        rows.append(
            {
                "kind": kind,
                "time_s": timestamp,
                "source_reference_index": source_idx,
                "max_penetration_model_units": max_pen,
                "passes_floor_limit": max_pen <= limit_model_units,
                "static_conflict_count": len(static_conflicts),
                "passes_coupled_preflight": passes,
                "floor_contact_count": len(floor_contacts),
                "floor_contacts": floor_contacts,
                "static_conflicts": static_conflicts,
            }
        )

    failed_rows = [r for r in rows if not r["passes_coupled_preflight"]]
    status = "infeasible" if (failed_rows or initial_intersections) else "feasible"
    return {
        "schema_version": 1,
        "profile": "shared-world-feasibility-v1",
        "status": status,
        "physical_validation": "unverified",
        "scene_sha256": canonical_hash(validated),
        "scope": (
            "Zero-physics kinematic preflight of the coupled human reference and initial rigid-object placement; "
            "not dynamic stability, balance, or verified physical accuracy."
        ),
        "summary": {
            "source_pose_checks": len(ref_times),
            "control_grid_checks": grid_count,
            "checks": len(rows),
            "failed_source_poses": sum(r["kind"] == "source_pose" for r in failed_rows),
            "failed_control_grid_poses": sum(r["kind"] == "control_grid" for r in failed_rows),
            "max_penetration_model_units": max(r["max_penetration_model_units"] for r in rows),
            "static_conflict_poses": sum(r["static_conflict_count"] > 0 for r in rows),
            "initial_cross_entity_intersections": len(initial_intersections),
            "dynamic_object_count": len(set(dynamic_geoms.values())),
            "physics_steps": 0,
        },
        "initial_intersections": initial_intersections,
        "checks": rows,
    }


class SharedWorldEnv(gym.Env):
    """Single-model, single-clock coupled articulated-human and 3D rigid-object environment."""

    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}

    def __init__(self, scene, render_mode=None, *, substep_observer=None):
        if substep_observer is not None and not callable(substep_observer):
            raise TypeError("substep_observer must be callable or None")
        self._substep_observer = substep_observer
        self.scene = validate_shared_world(scene)
        self.render_mode = render_mode
        self.human_spec = self.scene["human_spec"]
        self.joint_names, self.joint_limits = joint_contract(self.human_spec)
        self.scale = float(self.scene["coordinate_frame"]["scale_bridge"]["body_height_to_world_scale"])
        self.origin_world = np.asarray(self.scene["coordinate_frame"]["scale_bridge"]["origin_world"], dtype=float)
        self.dt = float(self.scene["dt"])
        self.substeps = int(self.scene["substeps"])
        self.horizon_steps = int(self.scene["task"]["horizon_steps"])
        self.start_offset_s = float(self.scene["task"]["start_offset_s"])

        self.model = mujoco.MjModel.from_xml_string(_build_world_xml(self.scene))
        self.data = mujoco.MjData(self.model)
        self._reference_data = None

        self.human_action_names = tuple(self.joint_names) + ROOT_AXES
        self.group_names = GROUP_NAMES
        self.group_indices = np.array(
            [
                0 if name.startswith("torso_") else 1 if ("shoulder" in name or "hip" in name) else 2
                for name in self.joint_names
            ]
            + [3, 3, 3],
            dtype=int,
        )

        self._human_joint_ids = [self.model.joint(name).id for name in self.human_action_names]
        self._q = np.array([int(self.model.jnt_qposadr[jid]) for jid in self._human_joint_ids], dtype=int)
        self._v = np.array([int(self.model.jnt_dofadr[jid]) for jid in self._human_joint_ids], dtype=int)
        self._actuators = np.array(
            [self.model.actuator(f"{name}_motor").id for name in self.human_action_names], dtype=int
        )
        if len(self._actuators) != 22 or self.model.nu != 22:
            raise ValueError("SharedWorldEnv requires exact 22 human motor actuators in MjModel")

        mobility = self.human_spec["mobility"]
        self.human_mass = human_subtree_mass(self.human_spec, scale=self.scale)
        root_limit = min(mobility["maximum_force"], mobility["force_limit_weight_multiple"] * self.human_mass * 9.81)
        joint_motor_limits = np.array(
            [float(self.human_spec.get("motor_limits", {}).get(name, 30.0)) for name in self.joint_names], dtype=float
        )
        self.human_limits = np.concatenate((joint_motor_limits, np.repeat(root_limit, 3)))
        self._kp = np.concatenate((np.full(19, 40.0), np.repeat(mobility["kp_per_mass"] * self.human_mass, 3)))
        self._kd = np.concatenate((np.full(19, 5.0), np.repeat(mobility["kd_per_mass"] * self.human_mass, 3)))

        self._times = np.array([row["time_s"] for row in self.human_spec["reference"]], dtype=float)
        self.source_start = float(self._times[0] + self.start_offset_s)
        ref_angles = np.array(
            [[row["joints"][name] for row in self.human_spec["reference"]] for name in self.joint_names], dtype=float
        )
        ref_roots = np.array([row["translation"] for row in self.human_spec["root_reference"]], dtype=float).T * self.scale
        self._reference = np.vstack((ref_angles, ref_roots))
        self._slopes = np.diff(self._reference, axis=1) / np.diff(self._times)
        if not np.isfinite(self._slopes).all() or np.max(np.abs(self._slopes)) > 1e4:
            raise ValueError("Human reference velocities exceed the finite feature envelope")

        self._dynamic_objects = _dynamic_objects(self.scene)
        self._dynamic_ids = [obj["id"] for obj in self._dynamic_objects]
        self._object_body_ids = {obj["id"]: self.model.body(obj["id"]).id for obj in self.scene["objects"]}
        self._compound_geometry = any(obj["shape"] == "compound" for obj in self.scene["objects"])
        self._object_geometry_ids = {obj["id"]: [] for obj in self.scene["objects"]}
        self._geom_part_ids = {}
        for obj, part, name in _geometry_names(self.scene):
            gid = self.model.geom(name).id
            self._object_geometry_ids[obj["id"]].append(gid)
            self._geom_part_ids[gid] = part
        self._object_geometry_ids = {name: tuple(ids) for name, ids in self._object_geometry_ids.items()}
        self._object_free_addresses = {}
        if substep_observer is not None:
            for obj in self._dynamic_objects:
                joint = self.model.joint(f"{obj['id']}_joint").id
                self._object_free_addresses[obj["id"]] = (
                    int(self.model.jnt_qposadr[joint]), int(self.model.jnt_dofadr[joint])
                )
        self._object_radii = {
            obj["id"]: (float(np.linalg.norm(obj["size"]) / 2.0) if obj["shape"] == "box" else
                        float(obj["radius"]) if obj["shape"] == "sphere" else
                        max(float(np.linalg.norm(np.maximum(abs(low), abs(high))))
                            for low, high in map(_primitive_bounds, obj["geometry"]["parts"])))
            for obj in self._dynamic_objects
        }

        self._floor_geom = self.model.geom("mobile_floor").id
        self._static_geom_to_id = {
            gid: obj["id"]
            for obj in self.scene["objects"]
            if obj["role"] == "static"
            for gid in self._object_geometry_ids[obj["id"]]
        }
        self._dynamic_geom_to_id = {
            gid: obj["id"] for obj in self._dynamic_objects for gid in self._object_geometry_ids[obj["id"]]
        }
        self._human_geom_to_name = {
            gid: mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, gid)
            for gid in range(self.model.ngeom)
            if int(self.model.geom_contype[gid]) & 2
        }
        self._site_ids = {
            endpoint: self.model.site(f"{endpoint}_site").id
            for _, a, b, _ in HUMAN_SEGMENT_DEFS
            for endpoint in (a, b)
        }
        lengths = self.human_spec["lengths"]
        self.human_height = (lengths["thigh"] + lengths["shin"] + lengths["torso"] + 0.34) * self.scale

        contract = self.scene["action_contract"]
        self.action_names = tuple(contract["names"])
        self.physical_limits = np.asarray(contract["physical_limits"], dtype=float)
        self.physical_limits.setflags(write=False)

        act_dim = len(self.action_names)
        obs_dim = 155 + 13 * len(self.scene["objects"]) + (0 if self.scene["profile"] == HUMAN_ONLY_PROFILE else 3)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(act_dim,), dtype=np.float32)
        self.observation_space = spaces.Box(
            np.full(obs_dim, -1e6, dtype=np.float32),
            np.full(obs_dim, 1e6, dtype=np.float32),
            dtype=np.float32,
        )

        self._initialized = False
        self._done = True
        self._failure = None
        self.steps = 0
        self.terminated = False
        self.truncated = False
        self.success = False
        self._last_action = np.zeros(act_dim, dtype=float)
        self._last_action_substeps = 0
        self._object_forces = {obj["id"]: [0.0, 0.0, 0.0] for obj in self._dynamic_objects}

    def _target_at(self, elapsed):
        timestamp = float(np.clip(self.source_start + elapsed, self._times[0], self._times[-1]))
        index = min(len(self._times) - 2, max(0, int(np.searchsorted(self._times, timestamp, side="right")) - 1))
        target = self._reference[:, index] + self._slopes[:, index] * (timestamp - self._times[index])
        velocity = self._slopes[:, index].copy() if timestamp < self._times[-1] else np.zeros(22)
        return target, velocity

    def target_state(self):
        return self._target_at(self.data.time)

    def _human_state(self):
        target, velocity = self.target_state()
        q, v = self.data.qpos[self._q], self.data.qvel[self._v]
        features = (
            np.column_stack((self._kp * (target - q), self._kd * (velocity - v), self.data.qfrc_bias[self._v]))
            / self.human_limits[:, None]
        )
        return q, v, target, velocity, features

    def feature_matrix(self):
        if not self._initialized:
            raise RuntimeError("Reset before reading shared-world human features")
        return self._human_state()[-1].copy()

    def _objects_state(self, data=None, *, reference=False):
        data = self.data if data is None else data
        items = []
        for obj in self.scene["objects"]:
            body = self._object_body_ids[obj["id"]]
            vel6 = np.zeros(6)
            mujoco.mj_objectVelocity(self.model, data, mujoco.mjtObj.mjOBJ_BODY, body, vel6, 0)
            entry = {k: deepcopy(v) for k, v in obj.items() if k != "provenance"} | {
                "position": data.xpos[body].tolist(),
                "quaternion": data.xquat[body].tolist(),
                "velocity": vel6[3:].tolist(),
                "angular_velocity": vel6[:3].tolist(),
            }
            if obj["role"] != "static":
                entry["virtual_force"] = [0.0, 0.0, 0.0] if reference else list(self._object_forces[obj["id"]])
            items.append(entry)
        return items

    def observation(self):
        if not self._initialized:
            raise RuntimeError("Reset before reading a shared-world observation")
        q, v, target, velocity, features = self._human_state()
        phase = (self.source_start + self.data.time - self._times[0]) / max(1e-9, self._times[-1] - self._times[0])
        human_obs = np.concatenate((q, v, target, velocity, features.ravel(), [np.clip(phase, 0.0, 1.0)]))
        objects = self._objects_state()
        obj_flat = [
            val
            for obj in objects
            for field in ("position", "quaternion", "velocity", "angular_velocity")
            for val in obj[field]
        ]
        goal_offset = []
        if self.scene["profile"] != HUMAN_ONLY_PROFILE:
            target_id = self.scene["task"]["object_goal"]["target_object_id"]
            goal_obj = next(obj for obj in objects if obj["id"] == target_id)
            goal_offset = (np.asarray(self.scene["target"], dtype=float) - np.asarray(goal_obj["position"], dtype=float)).tolist()
        obs = np.concatenate((human_obs, np.asarray(obj_flat, dtype=float), np.asarray(goal_offset, dtype=float))).astype(
            np.float32
        )
        if not self.observation_space.contains(obs):
            raise RuntimeError("Shared-world observation exceeds finite declared bounds")
        return obs

    def _classify_geom(self, gid):
        if gid == self._floor_geom:
            return "floor", "mobile_floor"
        if gid in self._human_geom_to_name:
            return "human", self._human_geom_to_name[gid]
        if gid in self._static_geom_to_id:
            return "static", self._static_geom_to_id[gid]
        if gid in self._dynamic_geom_to_id:
            return "object", self._dynamic_geom_to_id[gid]
        return "other", mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, gid) or f"geom_{gid}"

    @property
    def _object_geom_ids(self):
        # Old study helpers must fail instead of silently inspecting only one compound part.
        if self._compound_geometry:
            raise ValueError("Compound objects require all object_geometry_ids(), not a single geom per object")
        return {name: ids[0] for name, ids in self._object_geometry_ids.items()}

    def object_geometry_ids(self):
        """Return every collision geom per semantic object; fixed parts are not actors."""
        return dict(self._object_geometry_ids)

    def _current_contacts(self, data=None):
        data = self.data if data is None else data
        if data.ncon > MAX_CONTACTS:
            raise RuntimeError("Shared-world contact count exceeds recording bounds")
        details = []
        pairs = set()
        counts = {kind: 0 for kind in CONTACT_KINDS}
        human_floor_list = []
        human_floor_normal_force = 0.0
        max_human_floor_pen_world = 0.0
        max_human_obj_force = 0.0
        goal_contacts = 0
        goal = self.scene["task"]["object_goal"]
        goal_id = goal["target_object_id"] if goal is not None else None

        for idx in range(data.ncon):
            con = data.contact[idx]
            cat1, name1 = self._classify_geom(int(con.geom1))
            cat2, name2 = self._classify_geom(int(con.geom2))
            pair_cats = frozenset({cat1, cat2})
            if pair_cats == frozenset({"human", "floor"}):
                kind = "human_floor"
            elif pair_cats == frozenset({"human", "static"}):
                kind = "human_static"
            elif pair_cats == frozenset({"human", "object"}):
                kind = "human_object"
            elif pair_cats == frozenset({"object", "floor"}):
                kind = "object_floor"
            elif pair_cats == frozenset({"object", "static"}):
                kind = "object_static"
            else:
                kind = "object_object"

            force6 = np.zeros(6)
            mujoco.mj_contactForce(self.model, data, idx, force6)
            normal_force = max(0.0, float(force6[0]))
            dist = float(con.dist)
            pos = con.pos.tolist()
            ordered = sorted([name1, name2])
            pairs.add(tuple(ordered))
            counts[kind] += 1
            details.append(
                {
                    "objects": ordered,
                    "kind": kind,
                    "position": pos,
                    "distance": dist,
                    "normal_force": normal_force,
                    **({"parts": [self._geom_part_ids.get(gid) for _, gid in sorted(
                        ((name1, int(con.geom1)), (name2, int(con.geom2))))]} if self._compound_geometry else {}),
                }
            )
            if kind == "human_floor":
                h_geom = name1 if cat1 == "human" else name2
                pen = max(0.0, -dist)
                max_human_floor_pen_world = max(max_human_floor_pen_world, pen)
                human_floor_normal_force += normal_force
                if len(human_floor_list) < 32:
                    human_floor_list.append(
                        {
                            "geom": h_geom,
                            "position": pos,
                            "distance": dist,
                            "normal_force": normal_force,
                        }
                    )
            elif kind == "human_object":
                max_human_obj_force = max(max_human_obj_force, normal_force)
                if (name1 if cat1 == "object" else name2) == goal_id:
                    goal_contacts += 1

        return {
            "details": details,
            "pairs": [list(p) for p in sorted(pairs)],
            "counts": counts,
            "human_floor_contacts": human_floor_list,
            "human_floor_normal_force": human_floor_normal_force,
            "max_human_floor_penetration_model_units": max_human_floor_pen_world / self.scale,
            "max_human_object_force": max_human_obj_force,
            "human_goal_object_contacts": goal_contacts,
        }

    def _substep_sample(self, start_s, contact_state, action):
        end_s = float(self.data.time)
        common = {
            "step_index": self.steps + 1,
            "substep_index": self._last_action_substeps,
            "interval_start_s": start_s,
            "interval_end_s": end_s if math.isfinite(end_s) else None,
            "contact_time_s": start_s,
            "human_action": tuple(map(float, action[:22])),
        }

        def invalid(reason):
            return SubstepSample(
                **common, valid=False, failure=reason, human_qpos=(), human_qvel=(),
                target_qpos=(), equipment=(), contacts=(), wrist_positions=MappingProxyType({}),
                max_human_floor_penetration_model_units=None,
            )

        if self._failure == "nonfinite_dynamics" or not math.isfinite(end_s) or contact_state is None:
            return invalid(self._failure or "nonfinite_observation")
        if self.model.opt.integrator != mujoco.mjtIntegrator.mjINT_IMPLICITFAST:
            return invalid("unsupported_observation_integrator")

        # implicitfast leaves contact/site caches at the solve state, before integration.
        wrists = MappingProxyType({
            side: tuple(map(float, self.data.site_xpos[self._site_ids[f"{side}_wrist"]]))
            for side in ("left", "right")
        })
        contacts = []
        for index in range(self.data.ncon):
            con = self.data.contact[index]
            first, second = self._classify_geom(int(con.geom1)), self._classify_geom(int(con.geom2))
            if first[0] == "human" and second[0] in ("static", "object"):
                human, obj, sign = first[1], second[1], -1.0
            elif second[0] == "human" and first[0] in ("static", "object"):
                human, obj, sign = second[1], first[1], 1.0
            else:
                continue
            force = np.zeros(6)
            mujoco.mj_contactForce(self.model, self.data, index, force)
            contacts.append(HumanObjectContact(
                human_geom=human, human_segment=_HUMAN_SEGMENT_BY_GEOM.get(human, human),
                object_id=obj, position=tuple(map(float, con.pos)), distance=float(con.dist),
                normal_force=max(0.0, float(force[0])),
                force_on_human_world=tuple(map(float, sign * con.frame.reshape(3, 3).T @ force[:3])),
                normal_on_human_world=tuple(map(float, sign * con.frame[:3])),
            ))
        equipment = []
        for obj in self.scene["objects"]:
            if obj["role"] == "static":
                position, quaternion = obj["position"], obj["quaternion"]
                velocity, angular_velocity, virtual_force = (0.0,) * 3, (0.0,) * 3, (0.0,) * 3
            else:
                qadr, vadr = self._object_free_addresses[obj["id"]]
                # Free joints expose integrated world translation and local angular velocity.
                position = self.data.qpos[qadr:qadr + 3]
                quaternion = self.data.qpos[qadr + 3:qadr + 7]
                velocity = self.data.qvel[vadr:vadr + 3]
                angular_velocity = rotation(quaternion) @ self.data.qvel[vadr + 3:vadr + 6]
                virtual_force = self._object_forces[obj["id"]]
            equipment.append(EquipmentState(
                object_id=obj["id"], position=tuple(map(float, position)),
                quaternion=tuple(map(float, quaternion)), velocity=tuple(map(float, velocity)),
                angular_velocity=tuple(map(float, angular_velocity)),
                virtual_force=tuple(map(float, virtual_force)),
            ))
        qpos = tuple(map(float, self.data.qpos[self._q]))
        qvel = tuple(map(float, self.data.qvel[self._v]))
        target = tuple(map(float, self._target_at(end_s)[0]))
        penetration = float(contact_state["max_human_floor_penetration_model_units"])
        numbers = [*qpos, *qvel, *target, penetration, *(value for point in wrists.values() for value in point)]
        for obj in equipment:
            numbers.extend((*obj.position, *obj.quaternion, *obj.velocity, *obj.angular_velocity, *obj.virtual_force))
        for con in contacts:
            numbers.extend((*con.position, con.distance, con.normal_force, *con.force_on_human_world,
                            *con.normal_on_human_world))
        if not all(math.isfinite(value) for value in numbers):
            return invalid("nonfinite_observation")
        return SubstepSample(
            **common, valid=True, failure=self._failure, human_qpos=qpos, human_qvel=qvel,
            target_qpos=target, equipment=tuple(equipment), contacts=tuple(contacts),
            wrist_positions=wrists, max_human_floor_penetration_model_units=penetration,
        )

    def _observe_substep(self, start_s, contact_state, action):
        try:
            self._substep_observer(self._substep_sample(start_s, contact_state, action))
        except Exception as exc:
            # Observation failure cannot leave a partially advanced episode retryable.
            self.steps += 1
            self._last_action = action.copy()
            self._failure, self._done, self.terminated = "substep_observer_failed", True, True
            raise RuntimeError("Shared-world substep observer failed after physics advanced") from exc

    def _check_initial_clearances(self):
        intersections = _initial_intersections(self.scene, self.model, self.data, self._human_geom_to_name)
        if intersections:
            pair = intersections[0]["pair"]
            raise ValueError(f"Initial shared-world geometries {pair[0]!r} and {pair[1]!r} intersect")

    def select_target(self, target_id):
        """Switch the active manual control target between 'human' and any dynamic rigid object."""
        if not isinstance(target_id, str):
            raise ValueError("Control target ID must be a string")
        if target_id != "human" and target_id not in self._dynamic_ids:
            if any(obj["id"] == target_id and obj["role"] == "static" for obj in self.scene["objects"]):
                raise ValueError(f"Static object {target_id!r} cannot be controlled")
            raise ValueError(f"Unknown shared-world control target: {target_id!r}")
        proposed = deepcopy(self.scene)
        proposed["controlled_id"] = target_id
        for obj in proposed["objects"]:
            if obj["role"] != "static":
                obj["role"] = "active" if obj["id"] == target_id else "passive"
        self.scene = validate_shared_world(proposed)
        if self._initialized:
            return self.observation()
        return None

    def authored_human_action(self):
        return np.clip(self.feature_matrix().sum(axis=1), -1.0, 1.0)

    def discrete_object_action_3d(self, object_id, code):
        if object_id not in self._dynamic_ids:
            raise ValueError(f"Unknown dynamic rigid object: {object_id!r}")
        if type(code) is not int or not 0 <= code < len(OBJECT_ACTIONS):
            raise ValueError("Discrete 3D object action must be an integer 0–7")
        if code == 0:
            return np.zeros(3, dtype=float)
        if code == 7:
            body = self._object_body_ids[object_id]
            vel6 = np.zeros(6)
            mujoco.mj_objectVelocity(self.model, self.data, mujoco.mjtObj.mjOBJ_BODY, body, vel6, 0)
            vel = vel6[3:]
            accel = float(self.scene["task"]["object_goal"]["virtual_acceleration"])
            requested = -8.0 * vel
            scale = min(1.0, accel / max(1e-12, float(np.linalg.norm(requested))))
            return np.clip((requested * scale) / accel, -1.0, 1.0)
        return DIRECTIONS[code].astype(float)

    def authored_action(self):
        """Programmed baseline: human tracks reference via explicit PD+bias motors; rigid objects unforced."""
        vec = np.zeros(len(self.action_names), dtype=float)
        vec[:22] = self.authored_human_action()
        return vec

    def manual_action_vector(self, *, selected_motor=None, motor_nudge=0.0, object_id=None, object_action_code=0):
        """Construct an explicit unified action vector for manual human-motor or 3D-object control."""
        vec = np.zeros(len(self.action_names), dtype=float)
        if object_id is not None and object_id != "human":
            idx = self._dynamic_ids.index(object_id)
            # Only requested channels act; reference feedback is a separate controller.
            vec[22 + 3 * idx : 22 + 3 * (idx + 1)] = self.discrete_object_action_3d(object_id, object_action_code)
            return vec
        if selected_motor is not None:
            if selected_motor not in self.human_action_names:
                raise ValueError(f"Unknown human motor: {selected_motor!r}")
            m_idx = self.human_action_names.index(selected_motor)
            vec[m_idx] = float(np.clip(motor_nudge, -1.0, 1.0))
        return vec

    def reset(self, *, seed=None, options=None):
        if seed is not None and (type(seed) is not int or not 0 <= seed < 2**32):
            raise ValueError("Shared-world seed must be a bounded nonnegative integer")
        if options is not None and not isinstance(options, dict):
            raise ValueError("Shared-world reset options must be a mapping")
        options = options or {}
        allowed = {"jitter", "jitter_scale", "position_jitter", "velocity_jitter"}
        if set(options) - allowed or type(options.get("jitter", False)) is not bool:
            raise ValueError("Shared-world reset accepts boolean jitter, jitter_scale, position_jitter, velocity_jitter")
        use_jitter = options.get("jitter", False)
        raw_scale = options.get("jitter_scale", 1.0 if use_jitter else 0.0)
        jitter_scale = _number(raw_scale, "jitter_scale", 0.0, 5.0)
        pos_jitter = _number(options.get("position_jitter", 0.025 if use_jitter else 0.0), "position_jitter", 0.0, 0.15)
        vel_jitter = _number(options.get("velocity_jitter", 0.12 if use_jitter else 0.0), "velocity_jitter", 0.0, 1.0)
        if not use_jitter and (jitter_scale > 0.0 or pos_jitter > 0.0 or vel_jitter > 0.0):
            raise ValueError("Nonzero jitter parameters require jitter=True")

        self._initialized = False
        self._done = True
        super().reset(seed=seed)
        self.model.opt.timestep = self.dt / self.substeps
        mujoco.mj_resetData(self.model, self.data)

        target_q, _ = self._target_at(0.0)
        self.data.qpos[self._q] = target_q
        self.data.qvel[:] = 0.0
        for obj in self._dynamic_objects:
            joint = self.model.joint(f"{obj['id']}_joint").id
            qadr = int(self.model.jnt_qposadr[joint])
            vadr = int(self.model.jnt_dofadr[joint])
            self.data.qpos[qadr : qadr + 3] = obj["position"]
            self.data.qpos[qadr + 3 : qadr + 7] = obj["quaternion"]
            self.data.qvel[vadr : vadr + 3] = obj["velocity"]
            self.data.qvel[vadr + 3 : vadr + 6] = rotation(obj["quaternion"]).T @ obj["angular_velocity"]
        mujoco.mj_forward(self.model, self.data)
        self._check_initial_clearances()

        object_perturbations = {}
        if use_jitter:
            if jitter_scale > 0.0:
                self.data.qpos[self._q[:19]] += self.np_random.uniform(-0.01 * jitter_scale, 0.01 * jitter_scale, 19)
                for idx, jname in enumerate(self.joint_names):
                    low, high = self.joint_limits[jname]
                    self.data.qpos[self._q[idx]] = float(np.clip(self.data.qpos[self._q[idx]], low + 1e-4, high - 1e-4))
                self.data.qvel[self._v[:19]] = self.np_random.uniform(-0.02 * jitter_scale, 0.02 * jitter_scale, 19)
                self.data.qpos[self._q[19:21]] += (
                    self.np_random.uniform(-0.002 * jitter_scale, 0.002 * jitter_scale, 2) * self.scale
                )
                self.data.qpos[self._q[21]] += float(self.np_random.uniform(0.0, 0.002 * jitter_scale)) * self.scale
                self.data.qvel[self._v[19:22]] = (
                    self.np_random.uniform(-0.01 * jitter_scale, 0.01 * jitter_scale, 3) * self.scale
                )
                mujoco.mj_forward(self.model, self.data)

            bounds = np.asarray(self.scene["bounds"], dtype=float)
            for obj in self._dynamic_objects:
                joint = self.model.joint(f"{obj['id']}_joint").id
                qadr = int(self.model.jnt_qposadr[joint])
                vadr = int(self.model.jnt_dofadr[joint])
                gids = self._object_geometry_ids[obj["id"]]
                nominal_pos = np.asarray(obj["position"], dtype=float)
                nominal_vel = np.asarray(obj["velocity"], dtype=float)
                chosen_pos = nominal_pos.copy()
                fallback = False
                if pos_jitter > 0.0:
                    margin = min(self._object_radii[obj["id"]] + 0.01, float(np.min(bounds)) * 0.4)
                    low = np.minimum(nominal_pos, margin)
                    high = np.maximum(nominal_pos, bounds - margin)
                    placed = False
                    other_geoms = [
                        self._floor_geom,
                        *self._human_geom_to_name.keys(),
                        *(gid for o in self.scene["objects"] if o["id"] != obj["id"]
                          for gid in self._object_geometry_ids[o["id"]]),
                    ]
                    for _ in range(32):
                        delta = np.array(
                            [
                                float(self.np_random.uniform(-pos_jitter, pos_jitter)),
                                float(self.np_random.uniform(-pos_jitter, pos_jitter)),
                                float(self.np_random.uniform(0.0, 0.5 * pos_jitter)),
                            ]
                        )
                        candidate = np.clip(nominal_pos + delta, low, high)
                        self.data.qpos[qadr : qadr + 3] = candidate
                        mujoco.mj_forward(self.model, self.data)
                        if all(
                            mujoco.mj_geomDistance(self.model, self.data, gid, ogid, 0.02, None) >= 1e-4
                            for gid in gids for ogid in other_geoms
                        ):
                            chosen_pos = candidate
                            placed = True
                            break
                    if not placed:
                        self.data.qpos[qadr : qadr + 3] = nominal_pos
                        mujoco.mj_forward(self.model, self.data)
                        fallback = True
                chosen_vel = nominal_vel.copy()
                if vel_jitter > 0.0:
                    chosen_vel = nominal_vel + self.np_random.uniform(-vel_jitter, vel_jitter, 3)
                    spd = float(np.linalg.norm(chosen_vel))
                    if spd > 0.8 * MAX_SPEED:
                        chosen_vel *= (0.8 * MAX_SPEED) / spd
                    self.data.qvel[vadr : vadr + 3] = chosen_vel
                object_perturbations[obj["id"]] = {
                    "position_delta": (chosen_pos - nominal_pos).tolist(),
                    "velocity_delta": (chosen_vel - nominal_vel).tolist(),
                    "fallback_unperturbed": fallback,
                }
            mujoco.mj_forward(self.model, self.data)
            self._check_initial_clearances()

        self.data.ctrl[:] = 0.0
        self.data.qfrc_applied[:] = 0.0
        self.data.xfrc_applied[:] = 0.0
        mujoco.mj_forward(self.model, self.data)

        self._initialized = True
        self._done = False
        self._failure = None
        self.steps = 0
        self.terminated = False
        self.truncated = False
        self.success = False
        self._episode_return = 0.0
        self._angular_squared = 0.0
        self._root_squared = 0.0
        self._metric_samples = 0
        self._last_action = np.zeros(len(self.action_names), dtype=float)
        self._last_action_substeps = 0
        self._object_forces = {obj["id"]: [0.0, 0.0, 0.0] for obj in self._dynamic_objects}

        contact_state = self._current_contacts()
        self._max_penetration = float(contact_state["max_human_floor_penetration_model_units"])
        self._initial_penetration = float(self._max_penetration)
        self._cumulative_contacts = dict(contact_state["counts"])
        self._max_human_object_force = float(contact_state["max_human_object_force"])
        self._human_goal_object_contacts = contact_state["human_goal_object_contacts"]

        q_delta = self.data.qpos[self._q] - target_q
        info = {
            "source_time_s": self.source_start,
            "reset_jitter": use_jitter,
            "jitter_scale": jitter_scale if use_jitter else 0.0,
            "initial_angular_offset_rmse_rad": float(np.sqrt(np.mean(q_delta[:19] ** 2))),
            "initial_root_offset_rmse_model_units": float(np.sqrt(np.mean((q_delta[19:22] / self.scale) ** 2))),
            "initial_floor_penetration_model_units": self._initial_penetration,
            "object_perturbations": object_perturbations,
            "hidden_assistance": False,
            "physical_validation": "unverified",
        }
        return self.observation(), info

    def _instability(self):
        if not np.isfinite(self.data.qpos).all() or not np.isfinite(self.data.qvel).all():
            return "nonfinite_dynamics"
        q, v = self.data.qpos[self._q], self.data.qvel[self._v]
        if np.max(np.abs(q[:19])) > 8.0 or np.max(np.abs(q[19:22] / self.scale)) > 5.1:
            return "human_position_envelope_exceeded"
        if np.max(np.abs(v[:19])) > 30.0 or np.max(np.abs(v[19:22] / self.scale)) > 10.0:
            return "human_velocity_envelope_exceeded"
        bounds = np.asarray(self.scene["bounds"], dtype=float)
        for obj in self._dynamic_objects:
            body = self._object_body_ids[obj["id"]]
            vel6 = np.zeros(6)
            mujoco.mj_objectVelocity(self.model, self.data, mujoco.mjtObj.mjOBJ_BODY, body, vel6, 0)
            if (
                float(np.linalg.norm(vel6[3:])) > MAX_SPEED + STATE_EPS
                or float(np.linalg.norm(vel6[:3])) > MAX_SPIN + STATE_EPS
            ):
                return "rigid_velocity_envelope_exceeded"
            pos = self.data.xpos[body]
            if (pos < -0.25).any() or (pos > bounds + 0.25).any():
                return "object_left_bounds"
        return None

    def step(self, action):
        if not self._initialized or self._done:
            raise RuntimeError("Reset before stepping shared-world-v1 and after episode completion")
        if isinstance(action, (list, tuple)) and any(isinstance(val, (bool, np.bool_)) for val in action):
            raise ValueError("Shared-world actions cannot contain booleans")
        values = np.asarray(action)
        expected_dim = len(self.action_names)
        if (
            values.shape != (expected_dim,)
            or values.dtype.kind not in "fiu"
            or not np.isfinite(values).all()
            or np.any(values < -1.0)
            or np.any(values > 1.0)
        ):
            raise ValueError(f"Shared-world action requires {expected_dim} finite numeric values within [-1, 1]")
        values = values.astype(float, copy=True)
        if self._instability() == "nonfinite_dynamics":
            self._failure, self._done = "nonfinite_dynamics", True
            raise RuntimeError("Nonfinite shared-world state; physics was not advanced")

        human_forces = values[:22] * self.human_limits
        self.data.ctrl[:] = 0.0
        self.data.ctrl[self._actuators] = human_forces
        self.data.qfrc_applied[:] = 0.0
        self.data.xfrc_applied[:] = 0.0
        accel = float(self.scene["task"]["object_goal"]["virtual_acceleration"]) if self._dynamic_objects else 0.
        for idx, obj in enumerate(self._dynamic_objects):
            f_vec = values[22 + 3 * idx : 22 + 3 * (idx + 1)] * float(obj["mass"]) * accel
            self._object_forces[obj["id"]] = f_vec.tolist()
            self.data.xfrc_applied[self._object_body_ids[obj["id"]], :3] = f_vec

        self._last_action_substeps = 0
        for _ in range(self.substeps):
            start_s = float(self.data.time) if self._substep_observer is not None else None
            mujoco.mj_step(self.model, self.data)
            self._last_action_substeps += 1
            self._failure = self._instability()
            if self._failure == "nonfinite_dynamics":
                if self._substep_observer is not None:
                    self._observe_substep(start_s, None, values)
                break
            try:
                c_now = self._current_contacts()
            except Exception as exc:
                if self._substep_observer is None:
                    raise
                self._failure = "contact_observation_failed"
                self._observe_substep(start_s, None, values)
                self.steps += 1
                self._last_action = values
                self._done, self.terminated = True, True
                raise RuntimeError("Shared-world contact observation failed after physics advanced") from exc
            self._max_penetration = max(self._max_penetration, c_now["max_human_floor_penetration_model_units"])
            self._max_human_object_force = max(self._max_human_object_force, c_now["max_human_object_force"])
            self._human_goal_object_contacts += c_now["human_goal_object_contacts"]
            for kind, cnt in c_now["counts"].items():
                self._cumulative_contacts[kind] += cnt
            if self._substep_observer is not None:
                self._observe_substep(start_s, c_now, values)
            if self._failure is not None:
                break

        self.steps += 1
        self._last_action = values
        if self._failure in ("nonfinite_dynamics", "rigid_velocity_envelope_exceeded"):
            self._done = True
            self.terminated = True
            raise RuntimeError(f"Shared-world dynamics exceeded finite/speed envelope ({self._failure}); no silent clamping")

        mujoco.mj_forward(self.model, self.data)
        c_end = self._current_contacts()
        self._max_penetration = max(self._max_penetration, c_end["max_human_floor_penetration_model_units"])
        self._max_human_object_force = max(self._max_human_object_force, c_end["max_human_object_force"])

        target_q, _ = self.target_state()
        error = self.data.qpos[self._q] - target_q
        angular_mse = float(np.mean(error[:19] ** 2))
        root_mse = float(np.mean((error[19:22] / self.scale) ** 2))
        self._angular_squared += angular_mse
        self._root_squared += root_mse
        self._metric_samples += 1

        human_score = math.exp(
            -angular_mse / (SUCCESS_LIMITS["angular_rmse_rad"] ** 2)
            - root_mse / (SUCCESS_LIMITS["root_rmse_model_units"] ** 2)
        )
        goal = self.scene["task"]["object_goal"]
        effort = float(np.mean(values**2))
        if goal is None:
            object_score = None
            reward = max(0.0, human_score - 0.01 * effort)
        else:
            objects = self._objects_state()
            goal_obj = next(obj for obj in objects if obj["id"] == goal["target_object_id"])
            dist = math.dist(goal_obj["position"], self.scene["target"])
            speed = float(np.linalg.norm(goal_obj["velocity"]))
            dist_scale = max(float(goal["position_tolerance"]), 0.22)
            pos_term = math.exp(-((dist / dist_scale) ** 2))
            park_bonus = 0.35 * math.exp(
                -((dist / float(goal["position_tolerance"])) ** 2)
                - 0.5 * ((speed / float(goal["speed_tolerance"])) ** 2)
            )
            object_score = (pos_term + park_bonus) / 1.35
            contact_gate = (
                1.0
                if (not goal["require_human_contact"] or self._human_goal_object_contacts > 0)
                else 0.65
            )
            reward = max(0.0, 0.5 * human_score + 0.5 * object_score * contact_gate - 0.01 * effort)

        self.terminated = self._failure is not None
        self.truncated = not self.terminated and self.steps >= self.horizon_steps
        self._done = self.terminated or self.truncated

        metrics = self.metrics()
        self.success = bool(metrics["success"])
        if self._done and self.success:
            reward += 0.5
        self._episode_return += reward
        metrics = self.metrics()

        info = {
            **metrics,
            "human_tracking_score": human_score,
            "object_goal_score": object_score,
            "normalized_effort": effort,
            "physical_validation": "unverified",
            "hidden_assistance": False,
        }
        return self.observation(), float(reward), self.terminated, self.truncated, info

    def metrics(self):
        if not self._initialized:
            raise RuntimeError("Reset before requesting shared-world metrics")
        valid = self._failure not in ("nonfinite_dynamics", "rigid_velocity_envelope_exceeded")
        angular = (math.sqrt(self._angular_squared / self._metric_samples) if self._metric_samples else 0.0) if valid else None
        root = (math.sqrt(self._root_squared / self._metric_samples) if self._metric_samples else 0.0) if valid else None
        goal = self.scene["task"]["object_goal"]
        obj_dist = obj_speed = obj_parked = contact_ok = None
        if goal is not None:
            objects = self._objects_state()
            goal_obj = next(obj for obj in objects if obj["id"] == goal["target_object_id"])
            obj_dist = math.dist(goal_obj["position"], self.scene["target"])
            obj_speed = float(np.linalg.norm(goal_obj["velocity"]))
            obj_parked = bool(obj_dist <= goal["position_tolerance"] and obj_speed <= goal["speed_tolerance"])
            contact_ok = bool(not goal["require_human_contact"] or self._human_goal_object_contacts > 0)

        completed = self.steps == self.horizon_steps and self._failure is None
        failures = [
            name
            for name, bad in (
                ("angular_tracking", angular is None or angular > SUCCESS_LIMITS["angular_rmse_rad"]),
                ("root_tracking", root is None or root > SUCCESS_LIMITS["root_rmse_model_units"]),
                ("floor_penetration", self._max_penetration > SUCCESS_LIMITS["max_floor_penetration_model_units"]),
                ("object_goal", goal is not None and not obj_parked),
                ("human_object_contact", goal is not None and not contact_ok),
            )
            if bad
        ]
        failure = self._failure or (",".join(failures) if completed and failures else None)
        human_ok = bool(
            completed
            and angular is not None
            and angular <= SUCCESS_LIMITS["angular_rmse_rad"]
            and root is not None
            and root <= SUCCESS_LIMITS["root_rmse_model_units"]
            and self._max_penetration <= SUCCESS_LIMITS["max_floor_penetration_model_units"]
        )
        return {
            "angular_rmse_rad": angular,
            "root_rmse_model_units": root,
            "max_floor_penetration_model_units": float(self._max_penetration),
            "initial_floor_penetration_model_units": float(self._initial_penetration),
            "target_object_id": goal["target_object_id"] if goal is not None else None,
            "object_distance_world": float(obj_dist) if obj_dist is not None else None,
            "object_speed_world_s": float(obj_speed) if obj_speed is not None else None,
            "object_parked": obj_parked,
            "human_tracking_passed": human_ok,
            "human_object_contact_satisfied": contact_ok,
            "human_object_contacts_total": int(self._cumulative_contacts["human_object"]),
            "human_goal_object_contacts_total": int(self._human_goal_object_contacts),
            "max_human_object_normal_force": float(self._max_human_object_force),
            "contact_force_units": self.scene["action_contract"]["units"][19],
            "cumulative_contacts": dict(self._cumulative_contacts),
            "completed_horizon": completed,
            "success": bool(completed and not failures),
            "failure": failure,
            "steps": self.steps,
            "return": self._episode_return / self.horizon_steps,
            "episode_return": float(self._episode_return),
            **({"object_goal_status": "not_applicable", "human_object_contact_status": "not_applicable"}
               if goal is None else {}),
        }

    def _human_snapshot_dict(self, contact_state, *, data=None, reference=False):
        data = self.data if data is None else data
        points = {name: data.site_xpos[sid].copy() for name, sid in self._site_ids.items()}
        segments = [
            {
                "id": seg_id,
                "a": points[a].tolist(),
                "b": points[b].tolist(),
                "radius": float(radius * self.scale),
            }
            for seg_id, a, b, radius in HUMAN_SEGMENT_DEFS
        ]
        muscles = []
        for idx, jname in enumerate(self.joint_names):
            _, start, end = joint_segment(jname)
            a, b = points[start], points[end]
            command = float(data.ctrl[self._actuators[idx]]) / float(self.human_limits[idx])
            for antagonist, sign in (("flexor", 1), ("extensor", -1)):
                offset = np.array([sign * 0.025 * self.scale, sign * 0.025 * self.scale, 0.0])
                muscles.append(
                    {
                        "name": f"{jname}_{antagonist}",
                        "a": (a + offset).tolist(),
                        "b": (b + offset).tolist(),
                        "activation": float(np.clip(max(0.0, sign * command), 0.0, 1.0)),
                        "status": "illustrative",
                    }
                )
        root_forces = [float(data.ctrl[self._actuators[19 + i]]) for i in range(3)]
        mobility = {
            "support_enabled": False,
            "translation": (data.qpos[self._q[19:22]] / self.scale).tolist(),
            "linear_velocity": (data.qvel[self._v[19:22]] / self.scale).tolist(),
            "support_force": root_forces,
            "floor_height": float(self.origin_world[2]),
            "contacts": contact_state["human_floor_contacts"],
            "contact_count": int(contact_state["counts"]["human_floor"]),
            "total_normal_force": float(contact_state["human_floor_normal_force"]),
            "max_penetration": float(contact_state["max_human_floor_penetration_model_units"] if reference else self._max_penetration),
            "scope": "Explicit virtual root motors in shared world; not recovered balance or measured ground reaction.",
        }
        if "ground_plane" in self.human_spec["mobility"]:
            mobility["floor_normal"] = ground_normal(self.human_spec).tolist()
            mobility["floor_scope"] = "Camera-relative inclined plane hypothesis; first-pose clearance only"
        return {
            "segments": segments,
            "muscles": muscles,
            "height": float(self.human_height),
            "origin": self.origin_world.tolist(),
            "joints": {name: val.tolist() for name, val in points.items()},
            "angles": {name: float(data.qpos[self._q[i]]) for i, name in enumerate(self.joint_names)},
            "angular_velocities": {name: float(data.qvel[self._v[i]]) for i, name in enumerate(self.joint_names)},
            "control_mode": "direct_motor_actions",
            "action_names": list(self.human_action_names),
            "normalized_action": [0.0] * 22 if reference else self._last_action[:22].tolist(),
            "motor_forces": [float(data.ctrl[aid]) for aid in self._actuators],
            "motor_units": list(self.scene["action_contract"]["units"][:22]),
            "muscle_status": "Illustrative motor-command proxies; not physiological muscle activation",
            "mobility": mobility,
        }

    def snapshot(self):
        if not self._initialized:
            raise RuntimeError("Reset before requesting a shared-world snapshot")
        contact_state = self._current_contacts()
        metrics = self.metrics()
        active_id = self.scene["controlled_id"]
        active_force = (
            list(self._object_forces[active_id])
            if active_id in self._object_forces
            else [0.0, 0.0, 0.0]
        )
        state = {
            "profile": self.scene["profile"],
            "bounds": list(self.scene["bounds"]),
            "coordinate_frame": deepcopy(self.scene["coordinate_frame"]),
            "controlled_id": active_id,
            "target": list(self.scene["target"]) if self.scene["target"] is not None else None,
            "steps": self.steps,
            "episode_steps": self.steps,
            "time_s": float(self.steps * self.dt),
            "source_time_s": float(self.source_start + self.steps * self.dt),
            "substeps": self.substeps,
            "success": bool(metrics["success"]),
            "terminated": bool(self.terminated),
            "truncated": bool(self.truncated),
            "normalized_action": self._last_action.tolist(),
            "virtual_force": active_force,
            "objects": self._objects_state(),
            "human": self._human_snapshot_dict(contact_state),
            "contacts": contact_state["pairs"],
            "contact_details": contact_state["details"],
            "contact_count": len(contact_state["details"]),
            "contact_summary": contact_state["counts"],
            "coupled_metrics": metrics,
            "physical_validation": "unverified",
        }
        if self._failure is not None and 0 < self._last_action_substeps < self.substeps:
            # A failed action consumes only the physics substeps that actually ran.
            state["time_s"] = float(self.data.time)
            state["source_time_s"] = float(self.source_start + self.data.time)
            state["partial_action"] = {
                "completed_substeps": self._last_action_substeps,
                "requested_substeps": self.substeps,
            }
        return state

    def reference_snapshot(self, index):
        """Zero-physics kinematic projection of human reference frame index alongside nominal rigid objects."""
        if type(index) is not int or not 0 <= index < len(self.human_spec["reference"]):
            raise ValueError("Reference frame index is outside the observed samples")
        # Inspection must not change live forces, solver warm starts or future dynamics.
        if self._reference_data is None:
            self._reference_data = mujoco.MjData(self.model)
        data = self._reference_data
        mujoco.mj_resetData(self.model, data)
        frame = self.human_spec["reference"][index]
        root_frame = self.human_spec["root_reference"][index]
        for idx, jname in enumerate(self.joint_names):
            data.qpos[self._q[idx]] = frame["joints"][jname]
        data.qpos[self._q[19:22]] = np.asarray(root_frame["translation"], dtype=float) * self.scale
        data.time = float(frame["time_s"] - self.human_spec["reference"][0]["time_s"])
        mujoco.mj_forward(self.model, data)
        contact_state = self._current_contacts(data)
        human = self._human_snapshot_dict(contact_state, data=data, reference=True)
        human["reference"] = True
        return {
            "profile": self.scene["profile"],
            "bounds": list(self.scene["bounds"]),
            "coordinate_frame": deepcopy(self.scene["coordinate_frame"]),
            "controlled_id": self.scene["controlled_id"],
            "target": list(self.scene["target"]) if self.scene["target"] is not None else None,
            "steps": index,
            "episode_steps": 0,
            "time_s": float(data.time),
            "source_time_s": float(frame["time_s"]),
            "substeps": 0,
            "success": False,
            "terminated": False,
            "truncated": False,
            "normalized_action": [0.0] * len(self.action_names),
            "virtual_force": [0.0, 0.0, 0.0],
            "objects": self._objects_state(data, reference=True),
            "human": human,
            "contacts": contact_state["pairs"],
            "contact_details": contact_state["details"],
            "contact_count": len(contact_state["details"]),
            "contact_summary": contact_state["counts"],
            "scope": ("Kinematic human pose projection with declared ground only; not dynamic simulation"
                      if self.scene["profile"] == HUMAN_ONLY_PROFILE else
                      "Kinematic human pose projection alongside initial 3D rigid geometry; not dynamic simulation"),
            "physical_validation": "unverified",
        }

    def render(self):
        return render_snapshot(self.snapshot(), self.scene)

    def close(self):
        self._initialized = False
        self._done = True
        self._reference_data = None


def validate_shared_snapshot(state, scene):
    """Validate a recorded or rendered shared-world-v1 state snapshot."""
    prepare_shared_snapshot_validator(scene)(state)


def prepare_shared_snapshot_validator(scene):
    """Pin a detached scene once; every subsequent state still receives all checks."""
    validated_scene = validate_shared_world(scene)
    return lambda state: _validate_shared_snapshot(state, validated_scene)


def _validate_shared_snapshot(state, validated_scene):
    if (
        not isinstance(state, dict)
        or state.get("profile") != validated_scene["profile"]
        or state.get("bounds") != validated_scene["bounds"]
        or state.get("physical_validation") != "unverified"
    ):
        raise ValueError("Shared-world snapshot must preserve profile, bounds, and physical_validation='unverified'")

    human_only = validated_scene["profile"] == HUMAN_ONLY_PROFILE
    if human_only and ("target" not in state or state["target"] is not None
                       or state.get("virtual_force") != [0., 0., 0.]):
        raise ValueError("Person-only snapshots cannot introduce an object target or virtual object force")

    expected_objects = {obj["id"]: obj for obj in validated_scene["objects"]}
    selected = state.get("controlled_id")
    if selected != "human" and (not isinstance(selected, str) or selected not in expected_objects
                                 or expected_objects[selected]["role"] == "static"):
        raise ValueError("Shared-world snapshot controlled_id must name the human or a dynamic object")
    if not isinstance(state.get("objects"), list) or len(state["objects"]) != len(expected_objects):
        raise ValueError("Shared-world snapshot must retain all rigid objects")
    seen_objs = set()
    for obj in state["objects"]:
        if not isinstance(obj, dict) or obj.get("id") not in expected_objects or obj["id"] in seen_objs:
            raise ValueError("Shared-world snapshot rigid object identity mismatch")
        seen_objs.add(obj["id"])
        orig = expected_objects[obj["id"]]
        geometry_keys = ("geometry",) if orig["shape"] == "compound" else ("size",) if orig["shape"] == "box" else ("radius",)
        for key in ("shape", "label", "color", "mass", "friction", *geometry_keys):
            if not _same_immutable(obj.get(key), orig[key]):
                raise ValueError(f"Shared-world snapshot rigid object {key} changed")
        expected_role = "static" if orig["role"] == "static" else "active" if selected == obj["id"] else "passive"
        if obj.get("role") != expected_role:
            raise ValueError("Shared-world snapshot role must preserve static geometry and the selected dynamic actor")
        if any(key in obj for key in {"size", "radius", "geometry"} - set(geometry_keys)):
            raise ValueError("Shared-world snapshot cannot introduce different rigid geometry fields")
        _vector(obj.get("position"), 3, "recorded position", -8.0, 8.0)
        _vector(obj.get("velocity"), 3, "recorded velocity", -MAX_SPEED - STATE_EPS, MAX_SPEED + STATE_EPS)
        _vector(obj.get("angular_velocity"), 3, "recorded spin", -MAX_SPIN - STATE_EPS, MAX_SPIN + STATE_EPS)
        quat = _vector(obj.get("quaternion"), 4, "recorded quaternion", -1.0 - 1e-7, 1.0 + 1e-7)
        if not math.isclose(float(np.linalg.norm(quat)), 1.0, rel_tol=0, abs_tol=1e-7):
            raise ValueError("Recorded shared-world quaternion must be unit length")
        if orig["role"] == "static":
            if (
                not np.allclose(obj["position"], orig["position"], atol=1e-9, rtol=0)
                or abs(float(np.dot(quat, orig["quaternion"]))) < 1.0 - 1e-9
                or any(obj["velocity"] + obj["angular_velocity"])
            ):
                raise ValueError("Recorded static rigid geometry must remain fixed")
        else:
            _vector(obj.get("virtual_force"), 3, "recorded object virtual_force", -3000.0, 3000.0)

    human = state.get("human")
    if not isinstance(human, dict) or human.get("control_mode") != "direct_motor_actions":
        raise ValueError("Shared-world snapshot requires human direct_motor_actions state")
    if not isinstance(human.get("segments"), list) or len(human["segments"]) != len(HUMAN_SEGMENT_DEFS):
        raise ValueError("Shared-world snapshot must retain all 12 human collision segments")
    for seg in human["segments"]:
        if not isinstance(seg, dict):
            raise ValueError("Human segment must be an object")
        _vector(seg.get("a"), 3, "segment start", -15.0, 15.0)
        _vector(seg.get("b"), 3, "segment end", -15.0, 15.0)
        _number(seg.get("radius"), "segment radius", 0.005, 1.0)
    if not isinstance(human.get("muscles"), list) or len(human["muscles"]) != 38:
        raise ValueError("Shared-world snapshot must retain 38 illustrative muscle bands")
    for mus in human["muscles"]:
        if not isinstance(mus, dict) or mus.get("status") != "illustrative":
            raise ValueError("Shared-world muscle bands must preserve status='illustrative'")
        _number(mus.get("activation"), "muscle activation", 0.0, 1.0)

    contract = validated_scene["action_contract"]
    act_dim = len(contract["names"])
    norm_act = _vector(state.get("normalized_action"), act_dim, "recorded normalized_action", -1.0, 1.0)
    if human.get("normalized_action") != norm_act[:22]:
        raise ValueError("Human normalized_action must match first 22 elements of shared-world normalized_action")
    motor_forces = _vector(human.get("motor_forces"), 22, "human motor_forces", -2000.0, 2000.0)
    if not human.get("reference"):
        for force_val, act_val, lim_val in zip(motor_forces, norm_act[:22], contract["physical_limits"][:22]):
            if not math.isclose(force_val, act_val * lim_val, rel_tol=0, abs_tol=1e-6):
                raise ValueError("Human motor_forces must equal normalized_action * physical_limits")
        dyn_objs = _dynamic_objects(validated_scene)
        for idx, dobj in enumerate(dyn_objs):
            state_obj = next(o for o in state["objects"] if o["id"] == dobj["id"])
            expected_f = [
                norm_act[22 + 3 * idx + axis] * contract["physical_limits"][22 + 3 * idx + axis]
                for axis in range(3)
            ]
            if not np.allclose(state_obj["virtual_force"], expected_f, atol=1e-6, rtol=0):
                raise ValueError(f"Recorded virtual_force on {dobj['id']!r} differs from normalized_action * limit")

    maximum_steps = max(validated_scene["task"]["horizon_steps"], len(validated_scene["human_spec"]["reference"]))
    if validated_scene["profile"] in FULLSPAN_PROFILES:
        if type(human.get("reference", False)) is not bool:
            raise ValueError("Full-reference snapshots require an explicit boolean reference marker")
        maximum_steps = (len(validated_scene["human_spec"]["reference"]) - 1 if human.get("reference")
                         else validated_scene["task"]["horizon_steps"])
    if type(state.get("steps")) is not int or not 0 <= state["steps"] <= maximum_steps:
        raise ValueError("Recorded shared-world step index exceeds bounds")
    expected_time = state["steps"] * validated_scene["dt"]
    if "partial_action" in state:
        partial = state["partial_action"]
        metrics = state.get("coupled_metrics")
        if (
            human.get("reference")
            or not isinstance(partial, dict)
            or set(partial) != {"completed_substeps", "requested_substeps"}
            or type(partial.get("completed_substeps")) is not int
            or type(partial.get("requested_substeps")) is not int
            or partial["requested_substeps"] != validated_scene["substeps"]
            or not 1 <= partial["completed_substeps"] < partial["requested_substeps"]
            or not 1 <= state["steps"] <= validated_scene["task"]["horizon_steps"]
            or state.get("substeps") != partial["requested_substeps"]
            or state.get("terminated") is not True
            or state.get("truncated") is not False
            or state.get("success") is not False
            or not isinstance(metrics, dict)
            or metrics.get("completed_horizon") is not False
            or metrics.get("success") is not False
            or metrics.get("steps") != state["steps"]
            or metrics.get("failure") not in {
                "human_position_envelope_exceeded", "human_velocity_envelope_exceeded",
                "rigid_velocity_envelope_exceeded", "object_left_bounds", "nonfinite_dynamics",
            }
        ):
            raise ValueError("Partial actions require a terminated runtime failure and a bounded substep count")
        expected_time = ((state["steps"] - 1) + partial["completed_substeps"] / partial["requested_substeps"]) * validated_scene["dt"]
        expected_source = validated_scene["human_spec"]["reference"][0]["time_s"] + validated_scene["task"]["start_offset_s"] + expected_time
        if not math.isclose(
            _number(state.get("source_time_s"), "partial source_time_s", 0.0, 3600.0),
            expected_source, rel_tol=0, abs_tol=1e-8,
        ):
            raise ValueError("Partial action source_time_s differs from the actual physics clock")
    if not human.get("reference") and not math.isclose(
        _number(state.get("time_s"), "recorded time_s", 0.0, 120.0),
        expected_time, rel_tol=0, abs_tol=1e-8,
    ):
        raise ValueError("Recorded shared-world time_s differs from completed physics substeps")
    if validated_scene["profile"] in FULLSPAN_PROFILES:
        first = validated_scene["human_spec"]["reference"][0]["time_s"]
        source_time = (validated_scene["human_spec"]["reference"][state["steps"]]["time_s"]
                       if human.get("reference") else first + expected_time)
        if (not math.isclose(_number(state.get("source_time_s"), "full-reference source time", 0., 3600.),
                             source_time, rel_tol=0, abs_tol=1e-8)
                or human.get("reference") and not math.isclose(
                    _number(state.get("time_s"), "full-reference pose time", 0., 60.), source_time - first,
                    rel_tol=0, abs_tol=1e-8)):
            raise ValueError("Full-reference snapshot clocks must retain the complete original source timeline")
    for flag in ("success", "terminated", "truncated"):
        if type(state.get(flag)) is not bool:
            raise ValueError(f"Shared-world {flag} flag must be boolean")

    details = state.get("contact_details")
    if (
        not isinstance(details, list)
        or type(state.get("contact_count")) is not int
        or not len(details) == state["contact_count"] <= MAX_CONTACTS
    ):
        raise ValueError("Shared-world contact_details must match contact_count within bounds")
    known_contact_names = _reserved_human_names(validated_scene["human_spec"]) | set(expected_objects)
    part_names = {obj["id"]: {part["id"] for part in obj["geometry"]["parts"]}
                  for obj in validated_scene["objects"] if obj["shape"] == "compound"}
    pairs = set()
    for con in details:
        keys = {"objects", "kind", "position", "distance", "normal_force"} | ({"parts"} if part_names else set())
        if not isinstance(con, dict) or set(con) != keys:
            raise ValueError("Shared-world contact detail fields differ from contract")
        if con["kind"] not in CONTACT_KINDS:
            raise ValueError(f"Unknown shared-world contact kind: {con['kind']!r}")
        if human_only and con["kind"] != "human_floor":
            raise ValueError("Person-only snapshots support only human-floor contacts")
        ids = con["objects"]
        if not isinstance(ids, list) or len(ids) != 2 or len(set(ids)) != 2 or not set(ids) <= known_contact_names:
            raise ValueError("Shared-world contact must name two distinct known world geometries")
        if part_names:
            parts = con["parts"]
            if (not isinstance(parts, list) or len(parts) != 2 or any(
                    (not isinstance(part, str) or part not in part_names[name]) if name in part_names else part is not None
                    for name, part in zip(ids, parts))):
                raise ValueError("Shared-world contact part identity must belong to its exact semantic object")
        pairs.add(tuple(sorted(ids)))
        _vector(con["position"], 3, "contact position", -15.0, 15.0)
        _number(con["distance"], "contact distance", -2.0, 2.0)
        _number(con["normal_force"], "contact normal_force", 0.0, 1e9)
    if state.get("contacts") != [list(p) for p in sorted(pairs)]:
        raise ValueError("Shared-world contact pair summary differs from contact_details")
    if human_only:
        counts = {kind: 0 for kind in CONTACT_KINDS}
        counts["human_floor"] = len(details)
        if state.get("contact_summary") != counts:
            raise ValueError("Person-only contact summary must exclude equipment contacts")
        if not human.get("reference"):
            metrics = state.get("coupled_metrics")
            null_fields = ("target_object_id", "object_distance_world", "object_speed_world_s",
                           "object_parked", "human_object_contact_satisfied")
            if (not isinstance(metrics, dict) or any(key not in metrics or metrics[key] is not None for key in null_fields)
                    or any(metrics.get(key) != "not_applicable" for key in
                           ("object_goal_status", "human_object_contact_status"))):
                raise ValueError("Person-only object goal/contact metrics must remain not applicable")


def render_snapshot(state, scene, width=960, *, show_target=True):
    """Render synchronized Front (x/z), Top (x/y), and 3D views of the coupled human + rigid-object world."""
    return prepare_shared_renderer(scene, width=width, show_target=show_target)(state)


def prepare_shared_renderer(scene, width=960, *, show_target=True):
    """A bounded export reuses immutable model metadata, never cached state validation."""
    scene = validate_shared_world(scene)

    def render(state):
        _validate_shared_snapshot(state, scene)
        return _render_snapshot(state, scene, width=width, show_target=show_target)

    return render


def _render_snapshot(state, scene, width=960, *, show_target=True):
    width = max(480, min(1280, int(width)))
    width += width % 2
    panel_width = width / 3.0
    height = round(width * 0.38)
    height += height % 2
    image = Image.new("RGB", (width, height + 72), "#F5EADC")
    draw = ImageDraw.Draw(image)

    bounds = np.asarray(scene["bounds"], dtype=float)
    center = bounds / 2.0
    scale = min(panel_width * 0.84 / max(bounds[0], bounds[1]), height * 0.76 / max(bounds))
    views = (
        ("Front: x / height", np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0]]), np.array([0.0, 1.0, 0.0])),
        ("Top: x / depth", np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0]]), np.array([0.0, 0.0, 1.0])),
        (
            "3D: human and ground" if scene["profile"] == HUMAN_ONLY_PROFILE else "3D: coupled world",
            np.array([[0.866, -0.5, 0.0], [0.25, 0.433, -0.866]]),
            np.array([0.433, 0.75, 0.5]),
        ),
    )

    human = state["human"]
    floor_z = float(human["mobility"]["floor_height"])
    goal = scene["task"]["object_goal"]
    goal_tol = float(goal["position_tolerance"]) if goal is not None else None
    rigid_primitives = [part for obj in state["objects"] for part in _world_primitives(obj)]

    for index, (label, matrix, depth) in enumerate(views):
        offset = np.array([(index + 0.5) * panel_width, height * 0.54])

        def project(points):
            return (np.asarray(points, dtype=float) - center) @ matrix.T * scale + offset

        draw.text((round(index * panel_width) + 8, 8), label, fill="#214F50")
        if index > 0:
            draw.line((round(index * panel_width), 26, round(index * panel_width), height), fill="#D8C9BA")

        # Floor grid across world x/y bounds at floor_z.
        for frac in (0.2, 0.5, 0.8):
            gx = bounds[0] * frac
            gy = bounds[1] * frac
            p1, p2 = project([gx, 0.0, floor_z]), project([gx, bounds[1], floor_z])
            q1, q2 = project([0.0, gy, floor_z]), project([bounds[0], gy, floor_z])
            draw.line((*p1, *p2), fill="#E6D8C8", width=1)
            draw.line((*q1, *q2), fill="#E6D8C8", width=1)

        # Depth-sorted rigid primitives and human capsules in one shared coordinate frame.
        draw_items = []
        for obj in rigid_primitives:
            moving = obj["role"] != "static"
            if obj["shape"] == "sphere":
                draw_items.append((0 if not moving else 1, float(np.asarray(obj["position"]) @ depth), "sphere", obj))
            elif obj["shape"] == "capsule":
                draw_items.append((0 if not moving else 1, float(np.asarray(obj["position"]) @ depth), "rigid_capsule", obj))
            else:
                verts = corners(obj)
                for face in FACES:
                    pts = verts[list(face)]
                    draw_items.append(
                        (0 if not moving else 1, float(pts.mean(axis=0) @ depth), "poly", (moving, pts, obj["color"]))
                    )
        for seg in human["segments"]:
            mid = 0.5 * (np.asarray(seg["a"]) + np.asarray(seg["b"]))
            draw_items.append((1, float(mid @ depth), "capsule", seg))

        for _, _, kind, payload in sorted(draw_items, key=lambda item: (item[0], item[1])):
            if kind == "poly":
                moving, pts, color = payload
                draw.polygon(
                    [tuple(p) for p in project(pts)],
                    fill=color if moving else None,
                    outline="#214F50" if moving else "#AAA399",
                    width=1,
                )
            elif kind == "sphere":
                x, y = project(payload["position"])
                r = payload["radius"] * scale if "object_id" in payload else max(2.0, payload["radius"] * scale)
                draw.ellipse(
                    (x - r, y - r, x + r, y + r),
                    fill=payload["color"] if payload["role"] != "static" else None,
                    outline="#214F50" if payload["role"] != "static" else "#AAA399",
                    width=1,
                )
            elif kind == "rigid_capsule":
                pa, pb = (project(point) for point in _capsule_endpoints(payload))
                px_w = max(1, round(2.0 * payload["radius"] * scale))
                outline = "#214F50" if payload["role"] != "static" else "#AAA399"
                fill = payload["color"] if payload["role"] != "static" else "#F5EADC"
                for stroke, color in ((px_w + 2, outline), (px_w, fill)):
                    draw.line((*pa, *pb), fill=color, width=stroke)
                    for x, y in (pa, pb):
                        radius = stroke / 2.0
                        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color)
            else:
                pa, pb = project(payload["a"]), project(payload["b"])
                px_w = max(2, round(2.0 * payload["radius"] * scale))
                draw.line((*pa, *pb), fill="#214F50", width=px_w + 2)
                draw.line((*pa, *pb), fill="#9BB3B0", width=px_w)
                hr = px_w / 2.0
                for hx, hy in (pa, pb):
                    draw.ellipse((hx - hr, hy - hr, hx + hr, hy + hr), fill="#9BB3B0")

        for mus in human["muscles"]:
            act = mus["activation"]
            pa, pb = project(mus["a"]), project(mus["b"])
            m_color = (round(150 + 55 * act), round(92 - 30 * act), round(72 - 20 * act))
            draw.line((*pa, *pb), fill=m_color, width=max(1, round(1.5 + 2.5 * act)))

        if show_target and goal is not None:
            tx, ty = project(state["target"])
            tr = max(3.0, goal_tol * scale)
            draw.ellipse((tx - tr, ty - tr, tx + tr, ty + tr), outline="#965644", width=2)

        for con in state["contact_details"]:
            cx, cy = project(con["position"])
            is_coupled = con["kind"] == "human_object"
            cr = 4 if is_coupled else 2
            draw.ellipse(
                (cx - cr, cy - cr, cx + cr, cy + cr),
                fill="#C7795D" if is_coupled else "#965644",
                outline="#214F50" if is_coupled else None,
            )

    ho_count = state.get("contact_summary", {}).get("human_object", 0)
    mode_tag = "KINEMATIC POSE + INITIAL OBJECTS" if human.get("reference") else "COUPLED 3D SIMULATION"
    draw.text(
        (10, height + 10),
        f"{mode_tag} | step {state['steps']} | {state['time_s']:.2f}s | {state['contact_count']} contacts ({ho_count} human-object)",
        fill="#214F50",
    )
    draw.text(
        (10, height + 34),
        "One MuJoCo model & clock · illustrative muscles (not EMG) · physical validation unverified.",
        fill="#965644",
    )
    return np.asarray(image)
