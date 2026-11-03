"""Explicit native terminal timing for coupled free-body worlds, not motion recovery."""

from weft.io import canonical_hash
from weft.native_interval import native_interval


PROFILE = "native-dynamic-terminal-control-interval-v1"


def dynamic_native_interval(human, dt, *, dynamic_object_ids):
    """Bind the unchanged native horizon to the complete dynamic-body inventory."""
    if (not isinstance(dynamic_object_ids, (list, tuple)) or not 1 <= len(dynamic_object_ids) <= 12
            or any(not isinstance(name, str) or not 1 <= len(name) <= 128 for name in dynamic_object_ids)
            or len(set(dynamic_object_ids)) != len(dynamic_object_ids)):
        raise ValueError("Dynamic native intervals require a unique bounded free-body inventory")
    marker = native_interval(human, dt)
    if marker is None:
        return None
    return {**marker, "profile": PROFILE, "dynamic_object_ids": sorted(dynamic_object_ids)}


def validate_dynamic_interval(scene):
    """The scene validator still owns all physical, source, object and task gates."""
    human = scene.get("human_spec", {})
    source = scene.get("source_binding", {})
    task = scene.get("task", {})
    objects = scene.get("objects", [])
    marker = task.get("native_interval")
    if (scene.get("profile") != "shared-world-v2" or source.get("mode") != "source_qualified"
            or source.get("source_sha256") != human.get("source_sha256")
            or not isinstance(marker, dict) or marker.get("profile") != PROFILE
            or not isinstance(objects, list) or not all(isinstance(obj, dict) for obj in objects)):
        raise ValueError("Dynamic native intervals require the explicit source-qualified coupled-world profile")
    ids = [obj.get("id") for obj in objects if obj.get("role") in ("active", "passive")]
    expected = dynamic_native_interval(human, scene.get("dt"), dynamic_object_ids=ids)
    if (expected is None or canonical_hash(marker) != canonical_hash(expected)
            or type(task.get("horizon_steps")) is not int
            or task["horizon_steps"] != expected["full_control_steps"] + 1
            or type(task.get("start_offset_s")) not in (int, float) or task["start_offset_s"] != 0):
        raise ValueError("Dynamic native interval must preserve every free body and the exact complete source span")
    return marker
