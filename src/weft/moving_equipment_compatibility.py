"""Pure advisory compatibility checks; no source admission, physics or task rejection."""

from hashlib import sha256
import json
import math
import time

from weft.io import canonical_hash

PROFILE = "moving-equipment-compatibility-v1"
FEEDBACK_PROFILE = "moving-equipment-feedback-policy-v1"
MAX_SCENE_BYTES = 2 << 20
MAX_TASK_BYTES = 64 << 10
SCOPE = ("Advisory arithmetic under the supplied authored simulation assumptions, not source admission, "
         "measured forces, recovered grasp, task impossibility or physical validation.")


class _Unassessed(ValueError):
    pass


def _require(value, reason):
    if not value:
        raise _Unassessed(reason)


def _number(value, reason, *, positive=False):
    _require(type(value) in (int, float) and abs(value) <= 1e9 and math.isfinite(value)
             and (not positive or value > 0), reason)
    return float(value)


def _vector(value, reason):
    _require(isinstance(value, list) and len(value) == 3, reason)
    return [_number(x, reason) for x in value]


def _parse(raw, maximum):
    _require(type(raw) is bytes and 0 < len(raw) <= maximum, "input_byte_bound_or_type")
    def pairs(items):
        result = {}
        for key, value in items:
            _require(key not in result, "duplicate_json_fields")
            result[key] = value
        return result
    def constant(_):
        raise _Unassessed("nonfinite_json")
    def floating(value):
        number = float(value)
        _require(math.isfinite(number), "nonfinite_json")
        return number
    try:
        value = json.loads(raw, object_pairs_hook=pairs, parse_constant=constant, parse_float=floating)
    except _Unassessed:
        raise
    except (UnicodeError, RecursionError, ValueError) as error:
        raise _Unassessed("invalid_json") from error
    _require(isinstance(value, dict), "input_not_object")
    return value


def _assess(scene, task):
    _require(scene.get("profile") == "shared-world-v2" and task.get("profile") == "moving-equipment-placement-task-v1",
             "unsupported_scene_or_task_profile")
    _require(task["binding"]["scene_sha256"] == canonical_hash(scene), "task_scene_binding_mismatch")
    _require(task["control"] == "explicit_human_root_and_virtual_object_forces", "unknown_force_control")
    coordinate = scene["coordinate_frame"]
    units = coordinate["units"]
    _require(coordinate["axes"] == "x_right_y_depth_z_up" and units["length"] == "body_scaled_world_units"
             and units["mass"] == "kilogram_prior" and units["time"] == "second", "unsupported_units_or_axes")
    mobility = scene["human_spec"]["mobility"]
    _require(isinstance(mobility, dict) and "ground_plane" not in mobility, "unsupported_floor_orientation")
    floor = _vector(coordinate["scale_bridge"]["origin_world"], "unknown_floor_origin")[2]
    gravity = _vector(scene["gravity"], "invalid_gravity")
    _require(gravity[0] == gravity[1] == 0 and gravity[2] < 0, "nonvertical_or_nonnegative_gravity")
    goal = task["goal"]
    target = _vector(goal["position_world"], "invalid_target")
    tolerance = _number(goal["position_tolerance_world"], "invalid_goal_tolerance", positive=True)
    speed = _number(goal["speed_tolerance_world_s"], "invalid_speed_tolerance", positive=True)
    hold = _number(goal["terminal_consecutive_hold_s"], "invalid_hold_duration", positive=True)
    objects = scene["objects"]
    _require(isinstance(objects, list) and 1 <= len(objects) <= 8 and all(isinstance(x, dict) for x in objects), "unsupported_object_inventory")
    selected = [obj for obj in objects if obj.get("id") == goal["object_id"]]
    _require(len(selected) == 1, "target_object_missing_or_ambiguous")
    obj = selected[0]
    _require(obj.get("shape") == "sphere" and obj.get("role") in ("passive", "controlled"), "unsupported_shape_or_static_target")
    provenance = obj["provenance"]
    _require(provenance.get("profile") == "moving-equipment-free-body-v1"
             and provenance.get("hidden_assistance") is False and provenance.get("trajectory_driving") is False,
             "unknown_object_force_assumptions")
    radius = _number(obj["radius"], "invalid_sphere_radius", positive=True)
    mass = _number(obj["mass"], "invalid_mass_prior", positive=True)
    acceleration = _number(scene["task"]["object_goal"]["virtual_acceleration"], "invalid_virtual_acceleration", positive=True)
    action = scene["action_contract"]
    _require(action == task["action_contract"], "task_action_contract_mismatch")
    ids, names, limits = action["dynamic_object_ids"], action["names"], action["physical_limits"]
    count = action["human_motor_count"]
    _require(type(count) is int and 1 <= count <= 128 and isinstance(ids, list) and 1 <= len(ids) <= 8
             and all(isinstance(x, str) and 1 <= len(x) <= 80 for x in ids) and len(set(ids)) == len(ids) and obj["id"] in ids,
             "unsupported_dynamic_action_inventory")
    _require(isinstance(names, list) and isinstance(limits, list) and len(names) == len(limits) == count+3*len(ids)
             and action["normalized_bounds"] == [-1, 1]
             and all(type(x) in (int, float) for x in action["normalized_bounds"]) and action["hidden_assistance"] is False,
             "unknown_force_channel_bounds")
    start = count+3*ids.index(obj["id"])
    _require(names[start:start+3] == [obj["id"]+suffix for suffix in ("_fx", "_fy", "_fz")], "unknown_force_channel_order")
    expected_force = mass*acceleration
    _require(all(_number(x, "invalid_object_force_limit", positive=True) == expected_force for x in limits[start:start+3]),
             "object_force_limit_inconsistent")
    deficit = max(0., -gravity[2]-acceleration)
    bottom_min = target[2]-tolerance-radius-floor
    bottom_max = target[2]+tolerance-radius-floor
    _require(bottom_max >= 0, "goal_region_below_floor")
    elevated = bottom_min > 0
    return {"object_id": obj["id"], "units": {"length": units["length"], "time": units["time"],
                "mass": units["mass"], "force": "kilogram_prior * body_scaled_world_units / second^2"},
            "floor_world_z": floor, "minimum_goal_sphere_bottom_above_floor_world": bottom_min,
            "goal_cannot_rest_on_floor": elevated, "maximum_upward_virtual_acceleration_world_s2": acceleration,
            "gravity_down_world_s2": -gravity[2], "upward_acceleration_deficit_world_s2": deficit,
            "steady_support_force_prior_units": mass*deficit,
            "continuous_unsupported_speed_window_s": 2*speed/deficit if deficit else None,
            "requested_hold_s": hold,
            "hold_exceeds_continuous_unsupported_speed_window": bool(deficit and hold > 2*speed/deficit),
            "support_dependency": bool(elevated and deficit > 0),
            "continuous_window_scope": "Necessary continuous free-flight bound only, not an exact discrete scorer certificate.",
            "other_collision_objects_present": len(objects)>1}


def assess_moving_equipment_compatibility(scene_bytes, task_bytes, *, controller_profile="manual"):
    """Return bounded warnings from exact documents; unsupported inputs never block manual exploration."""
    started = time.monotonic()
    result = {"schema_version": 1, "profile": PROFILE, "status": "unassessed", "input_sha256": {},
        "source_admission": "not_performed", "blocks_manual_exploration": False,
        "task_or_world_changed": False, "warnings": [], "next_steps": [], "scope": SCOPE}
    try:
        scene = _parse(scene_bytes, MAX_SCENE_BYTES); task = _parse(task_bytes, MAX_TASK_BYTES)
        result["input_sha256"] = {"scene_bytes": sha256(scene_bytes).hexdigest(), "task_bytes": sha256(task_bytes).hexdigest(),
                                  "scene_canonical": canonical_hash(scene), "task_canonical": canonical_hash(task)}
        facts = _assess(scene, task)
        result.update(status="assessed", facts=facts)
        if facts["support_dependency"]:
            result["warnings"].append({"code": "elevated_goal_needs_support_for_sustained_balance",
                "message": "The goal is above the floor, and the direct upward force is weaker than gravity. Sustained balance needs contact support; this does not prove the task impossible."})
            result["next_steps"].append("Inspect body or other-object support and a contact-aware controller under the unchanged task, world and force limits.")
        else:
            result["next_steps"].append("No elevated-support warning was established. This does not establish reachability, stable contact or task success.")
        if controller_profile == FEEDBACK_PROFILE:
            result["controller_assessment"] = "known_nominal_feedback_family_not_a_specific_policy_validation"
            if facts["support_dependency"]:
                result["warnings"].append({"code": "nominal_feedback_has_no_explicit_support_coordination",
                    "message": "The nominal feedback family has no explicit object-aware human support control. Incidental contact is possible, but another gain sweep does not add that capability."})
        elif controller_profile == "manual":
            result["controller_assessment"] = "manual_exploration_not_rejected"
        else:
            result["controller_assessment"] = "unassessed_unknown_controller"
        if time.monotonic()-started > 1:
            raise _Unassessed("preflight_time_bound")
    except (KeyError, TypeError, IndexError, RecursionError, _Unassessed) as error:
        reason = str(error) if isinstance(error, _Unassessed) else "missing_or_malformed_required_fields"
        result.update(status="unassessed", reason=reason, warnings=[], facts=None,
                      controller_assessment="unassessed", next_steps=["Inspect the supported shape, floor, units and explicit force assumptions; no task feasibility decision was made."])
    return result
