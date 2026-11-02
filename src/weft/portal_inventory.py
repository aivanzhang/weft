"""Keep uploader declarations separate from inferred scenes and available construction routes."""

from copy import deepcopy
import re

PROFILE = "portal-exercise-intent-v1"
INVENTORIES = {"ground_only", "equipment_or_obstacles", "unknown"}
OUTPUTS = {"simulation", "body_preview"}


def exercise_intent(inventory=None, output=None):
    if inventory is not None and (type(inventory) is not str or inventory not in INVENTORIES):
        raise ValueError("Choose ground-only exercise, equipment/obstacles, or unsure")
    if output is not None and (type(output) is not str or output not in OUTPUTS):
        raise ValueError("Choose an interactive simulation or a body evidence preview")
    return {"schema_version": 1, "profile": PROFILE, "inventory": inventory or "unknown",
            "output": output or "simulation", "basis": "user_declared" if inventory is not None else "not_declared",
            "absence_verified": False}


def validate_intent(value):
    if value is None:
        return exercise_intent()
    if type(value) is not dict:
        raise ValueError("Exercise intent must be a bounded declaration")
    inventory = value.get("inventory") if value.get("basis") == "user_declared" else None
    expected = exercise_intent(inventory, value.get("output"))
    if (value != expected or type(value.get("schema_version")) is not int
            or type(value.get("absence_verified")) is not bool):
        raise ValueError("Exercise inventory does not match the explicit declaration contract")
    return deepcopy(expected)


def inventory_report(intent, source_sha256, *, ai_inventory=None):
    intent = validate_intent(intent)
    if type(source_sha256) is not str or not re.fullmatch(r"[0-9a-f]{64}", source_sha256):
        raise ValueError("Scene inventory requires the exact uploaded source identity")
    if ai_inventory not in (None, "ground_only", "equipment_or_obstacles", "unknown", "not_applicable"):
        raise ValueError("Unknown AI exercise inventory")
    declared_ground = intent["basis"] == "user_declared" and intent["inventory"] == "ground_only"
    conflict = declared_ground and ai_inventory == "equipment_or_obstacles"
    route = ("body_preview" if intent["output"] == "body_preview" else
             "human_only" if declared_ground and not conflict else "needs_input")
    message = {
        "human_only": "Person-only simulation under your declaration: one person, ground support only, no task-relevant equipment or obstacles. Absence is not independently verified.",
        "body_preview": "Body evidence preview only. Equipment, obstacles and their contacts are not reconstructed. No physics, learned controller or full-scene sandbox is created.",
        "needs_input": "A full exercise scene needs an explicit inventory. Equipment/obstacle reconstruction additionally needs source-bound equipment evidence, a scene configuration and source mappings. This upload cannot silently become a person-only simulation.",
    }[route]
    if conflict:
        message += " AI proposed equipment or obstacles, which conflicts with the ground-only declaration; inspect the source before choosing a new scope."
    return {"schema_version": 1, "profile": "portal-scene-inventory-v1", "source_sha256": source_sha256,
        "intent": intent, "route": route, "ai_inventory": ai_inventory, "declaration_conflict": conflict,
        "absence_verified": False, "equipment_reconstructed": False, "physical_validation": "unverified",
        "message": message,
        "next_steps": (["Choose Body evidence preview to inspect poses without excluding equipment from a claimed full scene.",
                        "Use a source-qualified mixed-scene package for equipment simulation; automatic mixed reconstruction is not available from this upload."]
                       if route == "needs_input" else [])}
