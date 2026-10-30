"""Versioned reset distributions are separate from exercise rewards and success gates."""

import math


HUMAN_JITTER_PROFILE = "shared-exercise-human-jitter-v1"


def human_jitter_options(scale):
    if type(scale) not in (int, float):
        raise ValueError("Exercise human jitter scale must be numeric")
    return resolve_reset_options({"profile": HUMAN_JITTER_PROFILE, "jitter": scale > 0,
                                  "jitter_scale": scale})


def resolve_reset_options(options):
    options = {} if options is None else options
    if (not isinstance(options, dict) or set(options) - {"profile", "jitter", "jitter_scale"}
            or type(options.get("jitter", False)) is not bool):
        raise ValueError("Exercise reset accepts only explicit jitter, jitter_scale and a supported profile")
    if "profile" in options and options["profile"] != HUMAN_JITTER_PROFILE:
        raise ValueError("Unsupported exercise reset profile; omit it for the original distribution")
    jitter = options.get("jitter", False)
    scale = options.get("jitter_scale", 1. if jitter else 0.)
    if (type(scale) not in (int, float) or not math.isfinite(scale) or not 0 <= scale <= 5
            or not jitter and scale != 0):
        raise ValueError("Exercise jitter_scale must be finite in [0,5]; nonzero requires jitter")
    resolved = {"jitter": jitter, "jitter_scale": float(scale)}
    if "profile" in options:
        resolved["profile"] = HUMAN_JITTER_PROFILE
    return resolved


def engine_reset_options(options):
    resolved = resolve_reset_options(options)
    if resolved.pop("profile", None) == HUMAN_JITTER_PROFILE:
        # Retain scene equipment state exactly; never lift it, settle it or replace its velocity.
        resolved.update(position_jitter=0., velocity_jitter=0.)
    return resolved


def describe_reset_protocol(options):
    options = resolve_reset_options(options)
    human_only = options.get("profile") == HUMAN_JITTER_PROFILE
    result = {"profile": HUMAN_JITTER_PROFILE if human_only else "shared-exercise-reset-v1",
              "human_jitter_scale": options["jitter_scale"],
              "object_position_jitter_world": .025 if options["jitter"] and not human_only else 0.,
              "object_velocity_jitter_world_s": .12 if options["jitter"] and not human_only else 0.,
              "equipment_stability_reference": "actual_reset_state"}
    if human_only:
        result["equipment_reset_state"] = "unchanged_scene_pose_and_velocity"
    return result
