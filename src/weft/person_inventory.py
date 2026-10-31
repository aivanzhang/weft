"""Explicit person-only scope declarations; a pose detector does not verify absence."""

from copy import deepcopy
import json
import re


PROFILE = "human-only-inventory-v1"
MAX_BYTES = 4096


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
