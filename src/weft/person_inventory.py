"""Explicit person-only scope declarations; a pose detector does not verify absence."""

from copy import deepcopy
from hashlib import sha256
import json
import re


PROFILE = "human-only-inventory-v1"
CONDITIONAL_PROFILE = "human-conditional-scope-v1"
MAX_BYTES = 4096


def make_conditional_person_inventory(source_sha256, original_intent):
    """Choose modeled entities without changing the original scene declaration."""
    if type(source_sha256) is not str or not re.fullmatch(r"[0-9a-f]{64}", source_sha256):
        raise ValueError("Conditional scope requires an exact recorded source SHA256")
    fields = {"schema_version", "profile", "inventory", "output", "basis", "absence_verified"}
    if (type(original_intent) is not dict or set(original_intent) != fields
            or type(original_intent["schema_version"]) is not int or original_intent["schema_version"] != 1
            or original_intent["profile"] != "portal-exercise-intent-v1"
            or type(original_intent["inventory"]) is not str or original_intent["inventory"] not in {"ground_only", "equipment_or_obstacles", "unknown"}
            or original_intent["output"] not in ("simulation", "body_preview")
            or original_intent["basis"] not in ("user_declared", "not_declared")
            or original_intent["basis"] == "not_declared" and original_intent["inventory"] != "unknown"
            or original_intent["absence_verified"] is not False):
        raise ValueError("Conditional scope must preserve the original bounded uploader intent")
    return {"schema_version": 1, "profile": CONDITIONAL_PROFILE, "source_sha256": source_sha256,
            "basis": "explicit_authored_modeling_scope", "selected_entity_id": "human", "modeled_support": "authored_ground",
            "original_exercise_intent": deepcopy(original_intent),
            "original_exercise_intent_sha256": sha256(json.dumps(original_intent, sort_keys=True, allow_nan=False).encode()).hexdigest(),
            "omitted_contacts": "unmodeled_equipment_obstacles_people_and_supports",
            "absence_verified": False, "physical_validation": "unverified"}


def make_person_inventory(source_sha256, *, analytic=False):
    """Bind a user declaration to one source; analytic fixtures have no source identity."""
    value = {
        "schema_version": 1,
        "profile": PROFILE,
        "source_sha256": source_sha256,
        "basis": "analytic_fixture" if analytic else "user_declared",
        "selected_entity_id": "human",
        "support": "ground_only",
        "task_relevant_equipment": "none_declared",
        "task_relevant_obstacles": "none_declared",
        "absence_verified": False,
    }
    return validate_person_inventory(value, source_sha256=source_sha256, analytic=analytic)


def validate_person_inventory(value, *, source_sha256, analytic=False):
    """Reject unknown or contradictory scope instead of silently excluding scene entities."""
    if type(analytic) is not bool:
        raise ValueError("Inventory analytic mode must be an explicit boolean")
    if isinstance(value, dict) and value.get("profile") == CONDITIONAL_PROFILE:
        if analytic:
            raise ValueError("Conditional source scope cannot claim analytic fixture admission")
        expected = make_conditional_person_inventory(source_sha256, value.get("original_exercise_intent"))
        if (set(value) != set(expected) or value != expected or type(value["schema_version"]) is not int
                or value["absence_verified"] is not False):
            raise ValueError("Conditional modeling scope differs from its closed original-intent contract")
        return deepcopy(expected)
    if analytic:
        if source_sha256 is not None:
            raise ValueError("Analytic person inventory must not claim a recorded source identity")
    elif not isinstance(source_sha256, str) or not re.fullmatch(r"[0-9a-f]{64}", source_sha256):
        raise ValueError("Person inventory requires an exact original source SHA256")
    expected = {
        "schema_version": 1,
        "profile": PROFILE,
        "source_sha256": source_sha256,
        "basis": "analytic_fixture" if analytic else "user_declared",
        "selected_entity_id": "human",
        "support": "ground_only",
        "task_relevant_equipment": "none_declared",
        "task_relevant_obstacles": "none_declared",
        "absence_verified": False,
    }
    if not isinstance(value, dict) or set(value) != set(expected):
        raise ValueError("Person inventory fields differ from its bounded declaration contract")
    try:
        raw = json.dumps(value, allow_nan=False, separators=(",", ":")).encode()
    except (TypeError, ValueError, RecursionError, UnicodeError) as error:
        raise ValueError("Person inventory must be finite bounded JSON") from error
    if (len(raw) > MAX_BYTES or value != expected
            or type(value["schema_version"]) is not int or type(value["absence_verified"]) is not bool):
        raise ValueError("Person-only scope requires one human, ground-only support and no declared equipment or obstacles; absence is unverified")
    return deepcopy(value)
