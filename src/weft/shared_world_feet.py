"""Source-bound ankle capsules for one shared world; no ankle-motion imitation."""

from copy import deepcopy
from hashlib import sha256
import json
import math
import xml.etree.ElementTree as ET

import numpy as np

from weft.foot_body import ANKLE_NAMES, FOOT_BODY_CONTRACT, SIDES, append_foot_capsules, foot_rotation, validate_reference
from weft.io import canonical_hash
from weft.mobile_human import ROOT_AXES
from weft.spatial_pose import SPATIAL_JOINT_NAMES, rotation_to_zyx


PROFILE = "shared-world-feet-v1"
ANKLE_LIMITS = (12.0,) * 6
MAX_REFERENCE_BYTES = 8 << 20
TARGET_MODE = "authored_initial_angle_hold"
SHARED_FEET_CONTRACT = {
    "profile": PROFILE,
    "engine_channel_order": "existing_19_body_then_3_root_then_6_ankles",
    "ankle_names": list(ANKLE_NAMES),
    "ankle_engine_indices": list(range(22, 28)),
    "joint_range_rad": deepcopy(FOOT_BODY_CONTRACT["joint_range_rad"]),
    "motor_limits": list(ANKLE_LIMITS),
    "motor_units": "kilogram_prior * body_scaled_world_units^2 / second^2",
    "motor_limit_scaling": "unchanged_authored_limit_like_existing_body_torques",
    "capsule_radius_model_units": FOOT_BODY_CONTRACT["radius_model_units"],
    "mass_per_foot_kg_at_unit_scale": FOOT_BODY_CONTRACT["mass_per_foot_kg"],
    "geometry_scaling": "same_linear_scale_as_human",
    "mass_scaling": "unit_scale_mass_times_human_scale_cubed",
    "reference_usage": "exact_original_start_orientation_only",
    "reference_interpolation": "none",
    "target_mode": TARGET_MODE,
    "body_or_floor_refit": False,
    "ground_force_qualification": "not_qualified",
    "learning": "unsupported",
    "physical_validation": "unverified",
}
_FIELDS = {
    "schema_version", "profile", "contract", "body_model_sha256", "source_sha256",
    "reference_json", "reference_sha256", "reference_canonical_sha256", "initial_reference", "target_mode",
}


def _scale(value):
    if type(value) not in (int, float) or not math.isfinite(value) or not .5 <= value <= 2.:
        raise ValueError("Shared feet require the human scale within [0.5, 2.0]")
    return float(value)


def added_foot_mass(scale):
    """Added subtree mass, including both authored capsules after geometric scaling."""
    return len(SIDES) * FOOT_BODY_CONTRACT["mass_per_foot_kg"] * _scale(scale)**3


def _hash(value):
    return (isinstance(value, str) and len(value) == 64
            and all(character in "0123456789abcdef" for character in value))


def _unique(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Foot reference JSON cannot contain duplicate keys")
        result[key] = value
    return result


def _constant(value):
    raise ValueError("Foot reference JSON requires finite numbers: " + value)


def _reference(spec, raw):
    if type(raw) is not bytes or not 0 < len(raw) <= MAX_REFERENCE_BYTES:
        raise ValueError("Foot reference must be exact nonempty JSON bytes within 8 MiB")
    try:
        text = raw.decode("utf-8")
        reference = json.loads(text, object_pairs_hook=_unique, parse_constant=_constant)
    except (UnicodeError, json.JSONDecodeError, RecursionError) as exc:
        raise ValueError("Foot reference must be bounded UTF-8 JSON") from exc
    validate_reference(spec, reference)
    if not _hash(spec.get("source_sha256")):
        raise ValueError("Shared feet require an explicit original source SHA-256")
    if any(shape["status"] != "inferred_fixed_triangle" for shape in reference["fixed_shapes"].values()):
        raise ValueError("Shared feet require both fixed source-inferred shapes")
    start_time = spec["reference"][0]["time_s"]
    start = next((row for row in reference["frames"] if row["time_s"] == start_time), None)
    if start is None or any(start["feet"][side]["status"] != "kinematic_reference" for side in SIDES):
        raise ValueError("Shared feet require both feet at the exact original starting pose; no later start or interpolation")
    return reference, start, text


def build_foot_extension(spec, reference_bytes):
    """Embed exact reference bytes so a saved scene can verify its source binding offline."""
    reference, start, text = _reference(spec, reference_bytes)
    return {
        "schema_version": 1, "profile": PROFILE, "contract": deepcopy(SHARED_FEET_CONTRACT),
        "body_model_sha256": canonical_hash(spec), "source_sha256": spec["source_sha256"],
        "reference_json": text, "reference_sha256": sha256(reference_bytes).hexdigest(),
        "reference_canonical_sha256": canonical_hash(reference),
        "initial_reference": {key: start[key] for key in ("time_s", "frame", "evidence_id")},
        "target_mode": TARGET_MODE,
    }


def _resolve(spec, extension):
    if (not isinstance(extension, dict) or set(extension) != _FIELDS
            or type(extension.get("schema_version")) is not int or extension["schema_version"] != 1
            or extension.get("profile") != PROFILE or extension.get("contract") != SHARED_FEET_CONTRACT
            or extension.get("target_mode") != TARGET_MODE):
        raise ValueError("Unsupported shared-world foot extension contract")
    text = extension["reference_json"]
    if not isinstance(text, str) or not 0 < len(text) <= MAX_REFERENCE_BYTES:
        raise ValueError("Foot extension requires its bounded original reference JSON")
    try:
        raw = text.encode("utf-8")
    except UnicodeError as exc:
        raise ValueError("Foot extension reference is not valid UTF-8") from exc
    reference, start, _ = _reference(spec, raw)
    expected = {
        "body_model_sha256": canonical_hash(spec), "source_sha256": spec["source_sha256"],
        "reference_sha256": sha256(raw).hexdigest(), "reference_canonical_sha256": canonical_hash(reference),
        "initial_reference": {key: start[key] for key in ("time_s", "frame", "evidence_id")},
    }
    if any(extension[key] != value for key, value in expected.items()):
        raise ValueError("Foot extension must preserve the exact source/body/reference/start bindings")
    return reference, start


def validate_foot_extension(spec, extension):
    _resolve(spec, extension)
    return deepcopy(extension)


def append_shared_feet(root, spec, extension, *, scale):
    """Append after the existing 22 motors; retain the caller's body/floor/collision setup."""
    reference, _ = _resolve(spec, extension)
    actuator = root.find("actuator") if isinstance(root, ET.Element) else None
    if actuator is None or [motor.get("joint") for motor in actuator] != list(SPATIAL_JOINT_NAMES + ROOT_AXES):
        raise ValueError("Shared feet require the unchanged ordered 19 body plus 3 root motors before attachment")
    return append_foot_capsules(root, spec, reference, scale=_scale(scale))


def initial_ankle_angles(spec, extension, knee_rotations):
    """Use nominal-start knee world rotations, already including the source body frame."""
    _, start = _resolve(spec, extension)
    if not isinstance(knee_rotations, dict) or set(knee_rotations) != set(SIDES):
        raise ValueError("Initial ankles require both nominal-start knee world rotations")
    angles = []
    for side in SIDES:
        try:
            knee = np.asarray(knee_rotations[side], dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError("Knee world rotations must be finite proper rotation matrices") from exc
        if (knee.shape != (3, 3) or not np.isfinite(knee).all()
                or not np.allclose(knee.T @ knee, np.eye(3), atol=1e-8, rtol=0)
                or not math.isclose(float(np.linalg.det(knee)), 1., abs_tol=1e-8)):
            raise ValueError("Knee world rotations must be finite proper rotation matrices; no repair")
        local = knee.T @ foot_rotation(start["feet"][side])
        local_angles, _ = rotation_to_zyx(local)
        angles.extend(map(float, local_angles))
    return angles
