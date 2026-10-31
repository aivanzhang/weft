"""Explicit ankle exclusions for the existing exercise scorer; no new task or physics."""

from dataclasses import replace
import math
from numbers import Real
from types import MappingProxyType

import numpy as np

from weft.io import canonical_hash
from weft.shared_support_task import MAX_CONTACTS
from weft.shared_world import _HUMAN_SEGMENT_BY_GEOM
from weft.shared_world_observation import EquipmentState, HumanObjectContact, SubstepSample


PROFILE = "shared-exercise-feet-projection-v1"
ENGINE_HUMAN_MOTOR_COUNT = 28
SCORER_MOTOR_INDICES = tuple(range(22))
ROTATION_MOTOR_INDICES = tuple(range(19)) + tuple(range(22, 28))
ROOT_MOTOR_INDICES = (19, 20, 21)
FOOT_SEGMENTS = ("left_foot", "right_foot")
_CONTACT_SEGMENT_BY_GEOM = MappingProxyType(dict(_HUMAN_SEGMENT_BY_GEOM))
HUMAN_OBSERVATION_VALUES = 197
MAX_ENGINE_ACTIONS = 28 + 3 * 12

PROJECTION_CONTRACT = MappingProxyType({
    "schema_version": 1,
    "profile": PROFILE,
    "engine_scene_profile": "shared-world-v5",
    "base_scene_profiles": ("shared-world-v2", "shared-world-v3", "shared-world-v4"),
    "identity_binding": "pin_engine_scene_base_scoring_scene_task_and_projection_contract",
    "scene_projection": "restore_base_profile_remove_foot_extension_regenerate_action_contract",
    "engine_human_motor_count": ENGINE_HUMAN_MOTOR_COUNT,
    "task_rotation_motor_indices": ROTATION_MOTOR_INDICES,
    "optional_root_motor_indices": ROOT_MOTOR_INDICES,
    "scorer_motor_indices": SCORER_MOTOR_INDICES,
    "angular_scoring_motor_indices": tuple(range(19)),
    "object_forces": "forbidden_zero",
    "contact_geometry_segment_pairs": tuple(sorted(_CONTACT_SEGMENT_BY_GEOM.items())),
    "excluded_contact_segments": FOOT_SEGMENTS,
    "excluded_contacts": "separate_telemetry_no_support_credit",
    "ankle_effort": "separate_telemetry_not_scored",
    "ground_load_requirements": "unsupported",
    "floor_penetration": "preserved_including_feet",
    "clock_mapping": "unchanged_preintegration_contact_and_postintegration_state",
    "human_observation_values": HUMAN_OBSERVATION_VALUES,
    "learning": "unsupported",
    "physical_validation": "unverified",
})
PROJECTION_CONTRACT_SHA256 = canonical_hash(dict(PROJECTION_CONTRACT))


def derive_base_scoring_scene(scene):
    """Prepare once off hot paths: existing scene validation can compile mass-check models.

    The result is a derived scoring identity, not a reconstruction of original file bytes.
    Source, body, floor, interval and task stay unchanged; the old action contract is rebuilt.
    """
    from weft.shared_world import FEET_PROFILE, validate_shared_world

    if not isinstance(scene, dict) or scene.get("profile") != FEET_PROFILE:
        raise ValueError("Exercise foot projection requires an explicit shared-world-v5 scene")
    base = validate_shared_world(scene)
    base["profile"] = base.pop("base_profile")
    base.pop("foot_extension")
    base.pop("action_contract")
    return validate_shared_world(base)


def task_motor_indices(*, root_assistance=False):
    if type(root_assistance) is not bool:
        raise ValueError("Foot task root_assistance must be an explicit boolean")
    return ROTATION_MOTOR_INDICES + (ROOT_MOTOR_INDICES if root_assistance else ())


def _number(value, label):
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{label} requires finite real numbers, not booleans or coerced strings")
    try:
        result = float(value)
    except (OverflowError, ValueError) as exc:
        raise ValueError(f"{label} requires finite real numbers") from exc
    if not math.isfinite(result):
        raise ValueError(f"{label} requires finite real numbers")
    return result


def _vector(values, size, label, *, normalized=False):
    if (not isinstance(values, (tuple, list, np.ndarray))
            or isinstance(values, np.ndarray) and values.ndim != 1 or len(values) != size):
        raise ValueError(f"{label} requires exactly {size} one-dimensional numeric values")
    result = tuple(_number(value, label) for value in values)
    if normalized and any(abs(value) > 1. for value in result):
        raise ValueError(f"{label} must remain within [-1, 1]; projection never clips")
    return result


def scatter_task_action(action, *, engine_action_count, root_assistance=False):
    """Map task rotations before optional roots; object-force channels always remain zero."""
    indices = task_motor_indices(root_assistance=root_assistance)
    if (type(engine_action_count) is not int or not 28 <= engine_action_count <= MAX_ENGINE_ACTIONS
            or (engine_action_count - 28) % 3):
        raise ValueError("Engine action count requires 28 human motors and up to 12 object-force triples")
    values = _vector(action, len(indices), "Foot task action", normalized=True)
    engine = [0.] * engine_action_count
    for index, value in zip(indices, values):
        engine[index] = value
    return tuple(engine)


def _project_contacts(sample):
    if (not isinstance(sample.equipment, tuple) or len(sample.equipment) > 12
            or any(not isinstance(item, EquipmentState) or not isinstance(item.object_id, str)
                   or not item.object_id for item in sample.equipment)):
        raise ValueError("Substep equipment must retain its bounded immutable engine records")
    identifiers = {item.object_id for item in sample.equipment}
    if len(identifiers) != len(sample.equipment):
        raise ValueError("Substep equipment identifiers must be unique")
    if not isinstance(sample.contacts, tuple) or len(sample.contacts) > MAX_CONTACTS:
        raise ValueError("Substep contacts must retain the bounded immutable engine tuple")
    retained, excluded = [], []
    for contact in sample.contacts:
        if (not isinstance(contact, HumanObjectContact) or not isinstance(contact.human_segment, str)
                or not isinstance(contact.object_id, str) or contact.object_id not in identifiers
                or not isinstance(contact.human_geom, str)):
            raise ValueError("Contact must name a canonical engine body geometry and observed equipment")
        if _CONTACT_SEGMENT_BY_GEOM.get(contact.human_geom) != contact.human_segment:
            raise ValueError("Every contact requires the canonical engine geometry/segment pair; aliases are forbidden")
        foot = contact.human_segment in FOOT_SEGMENTS
        if foot:
            # Exclusion must not conceal malformed telemetry from the unchanged scorer.
            for name in ("position", "force_on_human_world", "normal_on_human_world"):
                values = getattr(contact, name)
                if not isinstance(values, tuple):
                    raise ValueError("Excluded foot contact vectors must be immutable tuples")
                _vector(values, 3, "Excluded foot contact " + name)
            _number(contact.distance, "Excluded foot contact distance")
            if _number(contact.normal_force, "Excluded foot contact normal force") < 0:
                raise ValueError("Excluded foot contact normal force must be nonnegative")
            excluded.append(contact)
        else:
            retained.append(contact)
    return tuple(retained), tuple(excluded)


def project_substep(sample):
    """Return a detached 22-motor sample and unscored foot contacts, preserving both clocks."""
    if not isinstance(sample, SubstepSample) or type(sample.valid) is not bool:
        raise ValueError("Foot projection requires the immutable engine SubstepSample")
    fields = ("human_qpos", "human_qvel", "target_qpos", "human_action")
    if any(not isinstance(getattr(sample, name), tuple) for name in fields):
        raise ValueError("Substep motor arrays must be immutable tuples")
    states = [getattr(sample, name) for name in fields[:3]]
    empty_failure = not sample.valid and all(not values for values in states)
    updates = {}
    for name in fields:
        values = getattr(sample, name)
        if empty_failure and name != "human_action":
            updates[name] = ()
            continue
        _vector(values, ENGINE_HUMAN_MOTOR_COUNT, "Substep " + name, normalized=name == "human_action")
        updates[name] = values[:22]
    retained, excluded = _project_contacts(sample)
    return replace(sample, **updates, contacts=retained), excluded
