"""Exact native endpoints use a declared shorter final action, never retimed evidence."""

from contextlib import contextmanager
import math

PROFILE = "native-terminal-control-interval-v1"


def native_interval(human, dt):
    from weft.source_clock import validate_summary

    clock = validate_summary(human.get("native_clock"), source_sha256=human.get("source_sha256"))
    reference = human["reference"]
    first, last = reference[0]["time_s"], reference[-1]["time_s"]
    duration = last - first
    if not 0 < duration <= 60 or not math.isfinite(duration) or dt != 1 / 30:
        raise ValueError("Native terminal intervals require the bounded original reference and nominal 30-Hz control")
    nearest = round(duration / dt)
    if math.isclose(nearest * dt, duration, rel_tol=0, abs_tol=1e-9):
        return None
    full = math.floor(duration / dt)
    terminal = duration - full * dt
    if not 0 <= full < 1800 or not 1e-9 < terminal < dt:
        raise ValueError("Native terminal action is outside the supported duration envelope")
    return {"schema_version": 1, "profile": PROFILE, "source_start_s": first, "source_end_s": last,
            "duration_s": duration, "full_control_steps": full, "terminal_duration_s": terminal,
            "source_clock_binding_sha256": clock["binding_sha256"]}


def validate_interval(scene):
    marker = scene["task"].get("native_interval")
    if isinstance(marker, dict) and marker.get("profile") == "native-dynamic-terminal-control-interval-v1":
        from weft.dynamic_native_interval import validate_dynamic_interval
        return validate_dynamic_interval(scene)
    if (scene["profile"] != "shared-world-v3" or "native_clock" not in scene["human_spec"]
            or not isinstance(marker, dict) or marker != native_interval(scene["human_spec"], scene["dt"])
            or type(marker.get("schema_version")) is not int or type(marker.get("full_control_steps")) is not int
            or any(type(marker[key]) not in (int, float) for key in
                   ("source_start_s", "source_end_s", "duration_s", "terminal_duration_s"))
            or scene["task"]["horizon_steps"] != marker["full_control_steps"] + 1
            or scene["task"]["start_offset_s"] != 0):
        raise ValueError("Native terminal interval must bind the exact person-only source span and clock")
    return marker


def action_duration(scene, index):
    marker = scene["task"].get("native_interval")
    return marker["terminal_duration_s"] if marker is not None and index == marker["full_control_steps"] else scene["dt"]


def elapsed_substeps(scene, count):
    marker = scene["task"].get("native_interval")
    if marker is None:
        return count * (scene["dt"] / scene["substeps"])
    boundary = marker["full_control_steps"] * scene["substeps"]
    if count <= boundary:
        return count * (scene["dt"] / scene["substeps"])
    if count == boundary + scene["substeps"]:
        return marker["duration_s"]
    return marker["full_control_steps"] * scene["dt"] + (count - boundary) * marker["terminal_duration_s"] / scene["substeps"]


@contextmanager
def action_timestep(model, scene, index):
    original = model.opt.timestep
    model.opt.timestep = action_duration(scene, index) / scene["substeps"]
    try:
        yield
    finally:
        model.opt.timestep = original
