"""Bounded planar ramp hypotheses; image alignment does not establish metric scale."""

import math

import numpy as np


def _array(value, shape, name):
    array = np.asarray(value, dtype=float)
    if array.shape != shape or not np.isfinite(array).all():
        raise ValueError(f"Invalid {name}")
    return array


def affine_ramp_camera(width, height, ramp, table, degrees):
    """Use two source lines and a declared ramp angle under an affine-plane assumption."""
    if type(width) is not int or type(height) is not int or not 32 <= min(width, height) <= max(width, height) <= 10000:
        raise ValueError("Invalid source dimensions")
    if not math.isfinite(degrees) or not 2 <= degrees <= 45:
        raise ValueError("Ramp angle must be between 2 and 45 degrees")
    lines = [_array(value, (2, 2), name) for value, name in ((ramp, "ramp"), (table, "table"))]
    for line in lines:
        if (line < 0).any() or (line > [width, height]).any() or line[1, 0] - line[0, 0] < 16:
            raise ValueError("Source support lines must point right and remain inside the image")
    phi, beta = [math.atan2(*(line[1] - line[0])[::-1]) for line in lines]
    ratio = math.tan(phi - beta) / math.tan(math.radians(degrees))
    if not 0.2 <= ratio <= 2:
        raise ValueError("Support lines do not admit the bounded affine camera hypothesis")
    rotation = np.array([[math.cos(beta), -math.sin(beta)], [math.sin(beta), math.cos(beta)]])
    transform = rotation @ np.diag([1., 1. / ratio])
    corners = np.array([[0., 0.], [width, 0.], [width, height], [0., height]]) @ transform
    low, high = corners.min(axis=0), corners.max(axis=0)
    margin = (high[0] - low[0]) * .04
    scale = high[0] - low[0] + 2 * margin
    scene_height = (high[1] - low[1] + 2 * margin) / scale
    if not .2 <= scene_height <= 2:
        raise ValueError("Affine camera exceeds supported scene dimensions")
    return {"profile": "affine-ramp-camera-v1", "image_size": [width, height],
            "image_to_plane": transform.tolist(), "plane_origin": (low - margin).tolist(),
            "pixels_per_scene_width": float(scale), "scene_height": float(scene_height),
            "vertical_scale_ratio": float(ratio), "table_angle_radians": beta,
            "ramp_degrees": degrees, "metric_scale_verified": False,
            "assumption": "Both supports and tracked centers share one affine camera plane"}


def image_to_scene(camera, points):
    return (np.asarray(points, dtype=float) @ np.asarray(camera["image_to_plane"])
            - camera["plane_origin"]) / camera["pixels_per_scene_width"]


def scene_to_image(camera, points):
    return ((np.asarray(points, dtype=float) * camera["pixels_per_scene_width"] + camera["plane_origin"])
            @ np.linalg.inv(camera["image_to_plane"]))


def fit_motion_prefix(times, positions):
    """Fit a constant-acceleration prefix without reading later observations."""
    times = np.asarray(times, dtype=float)
    if times.ndim != 1 or not 5 <= len(times) <= 120 or not np.isfinite(times).all():
        raise ValueError("Motion fit requires 5–120 finite timestamps")
    if (np.diff(times) < 1e-5).any() or not .01 <= times[-1] - times[0] <= 5:
        raise ValueError("Motion timestamps must increase over a bounded interval")
    positions = _array(positions, (len(times), 2), "positions")
    if (np.abs(positions) > 2).any():
        raise ValueError("Motion positions exceed normalized scene bounds")
    elapsed = times - times[0]
    design = np.column_stack([np.ones(len(times)), elapsed, .5 * elapsed**2])
    coefficients, _, rank, _ = np.linalg.lstsq(design, positions, rcond=None)
    if rank != 3:
        raise ValueError("Motion prefix cannot identify a quadratic trajectory")
    residuals = design @ coefficients - positions
    return {"start_time_s": float(times[0]), "end_time_s": float(times[-1]), "samples": len(times),
            "position": coefficients[0].tolist(), "velocity": coefficients[1].tolist(),
            "acceleration": coefficients[2].tolist(),
            "rmse_scene_units": float(np.sqrt(np.mean(np.sum(residuals**2, axis=1)))),
            "scope": "Construction-only kinematic fit, not independent physical validation"}


def gravity_hypotheses(acceleration, degrees, coefficients):
    """Retain the friction/scale family instead of claiming a uniquely recovered gravity."""
    acceleration = _array(acceleration, (2,), "acceleration")
    if not math.isfinite(degrees) or not 2 <= degrees <= 45:
        raise ValueError("Invalid ramp angle")
    coefficients = list(coefficients)
    if not 1 <= len(coefficients) <= 12:
        raise ValueError("Use 1–12 friction hypotheses")
    angle = math.radians(degrees)
    tangent = np.array([math.cos(angle), math.sin(angle)])
    normal = np.array([-tangent[1], tangent[0]])
    along = float(acceleration @ tangent)
    if along <= 0:
        raise ValueError("Ramp fit must accelerate downhill")
    result = []
    for coefficient in coefficients:
        if not math.isfinite(coefficient) or not 0 <= coefficient < math.tan(angle):
            raise ValueError("Friction hypothesis does not permit downhill acceleration")
        gravity = along / (math.sin(angle) - coefficient * math.cos(angle))
        if not 0 < gravity <= 20:
            raise ValueError("Inferred normalized gravity exceeds simulator bounds")
        result.append({"pair_friction": coefficient, "gravity_scene_widths_per_s2": gravity,
                       "along_acceleration": along, "normal_acceleration_residual": float(acceleration @ normal),
                       "conditional_meters_per_scene_width_at_9_81": 9.81 / gravity,
                       "metric_scale_verified": False})
    return result
