"""Spatial hinge references from model landmarks, not measured anatomy or dynamics."""

from copy import deepcopy
import math

import numpy as np

from weft.body_frame import CAMERA_TO_SIM, frame_rotation, validate_body_frame


SPATIAL_JOINT_NAMES = ("torso_twist", "torso_lateral", "torso_flex") + tuple(
    name for side in ("left", "right") for name in (
        f"{side}_shoulder_twist", f"{side}_shoulder_lateral", f"{side}_shoulder", f"{side}_elbow",
        f"{side}_hip_twist", f"{side}_hip_lateral", f"{side}_hip", f"{side}_knee"))
SPATIAL_JOINT_LIMITS = {name: (0., 2.6) if name.endswith(("_elbow", "_knee")) else (-math.tau, math.tau)
                        for name in SPATIAL_JOINT_NAMES}
LIMB_ENDPOINTS = {"left_shoulder": (11, 13, 15), "right_shoulder": (12, 14, 16),
                  "left_hip": (23, 25, 27), "right_hip": (24, 26, 28)}
BEND_PLANE_MIN_SINE = .03


def rotation_zyx(angles):
    """Return Rz(z) @ Ry(y) @ Rx(x) for authored scalar hinge charts."""
    z, y, x = angles
    cz, sz, cy, sy, cx, sx = math.cos(z), math.sin(z), math.cos(y), math.sin(y), math.cos(x), math.sin(x)
    return np.array([[cz * cy, cz * sy * sx - sz * cx, cz * sy * cx + sz * sx],
                     [sz * cy, sz * sy * sx + cz * cx, sz * sy * cx - cz * sx],
                     [-sy, cy * sx, cy * cx]])


def rotation_to_zyx(rotation, previous=None):
    """Return bounded continuous-branch angles and whether yaw/roll are singular."""
    r = np.asarray(rotation, dtype=float)
    if (r.shape != (3, 3) or not np.isfinite(r).all()
            or not np.allclose(r.T @ r, np.eye(3), atol=1e-6)
            or not math.isclose(float(np.linalg.det(r)), 1., abs_tol=1e-6)):
        raise ValueError("Spatial pose requires a proper rotation")
    prior = np.zeros(3) if previous is None else np.asarray(previous, dtype=float)
    if prior.shape != (3,) or not np.isfinite(prior).all() or np.abs(prior).max() > math.tau:
        raise ValueError("Previous spatial chart must contain three bounded finite angles")
    cy = math.hypot(r[0, 0], r[1, 0])
    y = math.atan2(-r[2, 0], cy)
    singular = cy < 1e-7
    if singular:
        # At gimbal lock only the coupled yaw/roll difference or sum is observable.
        z = float(prior[0])
        x = math.atan2(-r[1, 2], r[1, 1]) + (z if y >= 0 else -z)
    else:
        z, x = math.atan2(r[1, 0], r[0, 0]), math.atan2(r[2, 1], r[2, 2])
    candidates = []
    for branch in ((z, y, x), (z + math.pi, math.pi - y, x + math.pi)):
        wrapped = []
        for value, target in zip(branch, prior):
            options = [value + k * math.tau for k in range(-2, 3)
                       if -math.tau <= value + k * math.tau <= math.tau]
            wrapped.append(min(options, key=lambda candidate: abs(candidate - target)))
        candidates.append(np.asarray(wrapped))
    chosen = min(candidates, key=lambda candidate: float(np.linalg.norm(candidate - prior)))
    return chosen, singular


def _unit(vector):
    length = float(np.linalg.norm(vector))
    if not math.isfinite(length) or length < 1e-5:
        raise ValueError("Spatial limb has degenerate segment geometry")
    return vector / length


def _minimal_rotation(down):
    initial = np.array([0., 0., -1.])
    cosine = float(np.clip(initial @ down, -1., 1.))
    axis = np.cross(initial, down)
    if cosine < -1 + 1e-8:
        return np.diag([1., -1., -1.])
    cross = np.array([[0., -axis[2], axis[1]], [axis[2], 0., -axis[0]], [-axis[1], axis[0], 0.]])
    return np.eye(3) + cross + cross @ cross / (1 + cosine)


def _limb_rotation(proximal, distal, bend_sign):
    down, tip = _unit(proximal), _unit(distal)
    cosine = float(np.clip(down @ tip, -1., 1.))
    bend = tip - cosine * down
    assumed = float(np.linalg.norm(bend)) < BEND_PLANE_MIN_SINE
    if assumed:
        rotation = _minimal_rotation(down)
    else:
        z, y = -down, bend_sign * _unit(bend)
        x = _unit(np.cross(y, z))
        rotation = np.column_stack((x, np.cross(z, x), z))
    return rotation, math.acos(cosine), assumed


def _confidence(landmarks, indices):
    if not isinstance(landmarks, list) or len(landmarks) != 33:
        return 0.
    values = [float(landmarks[index].get(key, 0.)) for index in indices for key in ("visibility", "presence")]
    if not all(math.isfinite(value) and 0 <= value <= 1 for value in values):
        raise ValueError("Spatial landmark confidence must be finite within [0, 1]")
    coordinates = [float(landmarks[index].get(key, float("nan"))) for index in indices for key in ("x", "y")]
    if not all(math.isfinite(value) and -.05 <= value <= 1.05 for value in coordinates):
        return 0.
    return min(values)


def build_spatial_reference(evidence, body_frame):
    """Return (time/joint references, versioned provenance and uncertainty audit)."""
    validate_body_frame(body_frame)
    frames = evidence.get("frames")
    if not isinstance(frames, list) or not 1 <= len(frames) <= 1800:
        raise ValueError("Spatial reference requires 1–1800 retained pose samples")
    anchor = np.asarray(body_frame["rotation"], dtype=float)
    references, audits, excluded = [], [], []
    previous, previous_time = {}, -1.
    for frame in frames:
        if frame.get("status") not in {"tracked", "tracked_partial"}:
            excluded.append({"evidence_id": frame.get("id"), "status": frame.get("status")})
            continue
        identifier, timestamp = frame.get("id"), frame.get("time_s")
        if not isinstance(identifier, str) or not identifier or type(timestamp) not in (int, float) or not math.isfinite(timestamp) or not previous_time < timestamp <= 60:
            raise ValueError("Spatial references require evidence IDs and increasing bounded times")
        landmarks = frame.get("world_landmarks")
        torso = frame_rotation(landmarks)
        points = np.array([[row[key] for key in ("x", "y", "z")] for row in landmarks]) @ CAMERA_TO_SIM.T
        joints, confidence, singularities, assumed_twist, clamped, wraps = {}, {}, [], [], [], []

        def chart(key, rotation, names, quality):
            angles, singular = rotation_to_zyx(rotation, previous.get(key))
            if key in previous and np.max(np.abs(angles - previous[key])) > math.pi:
                wraps.append(key)
            previous[key] = angles
            for name, value, certainty in zip(names, angles, quality):
                joints[name], confidence[name] = float(value), float(certainty)
            if singular:
                singularities.append(key)
                confidence[names[0]] = confidence[names[2]] = 0.

        # World coordinates cannot establish whether an endpoint lies inside the recorded image.
        visibility = frame.get("landmarks")
        torso_confidence = _confidence(visibility, (11, 12, 23, 24))
        chart("torso", anchor.T @ torso, SPATIAL_JOINT_NAMES[:3], [torso_confidence] * 3)
        for proximal, (base, middle, end) in LIMB_ENDPOINTS.items():
            knee = proximal.endswith("hip")
            rotation, flexion, assumed = _limb_rotation(torso.T @ (points[middle] - points[base]),
                                                       torso.T @ (points[end] - points[middle]), -1 if knee else 1)
            base_confidence = frame.get("joint_confidence", {}).get(proximal, 0.)
            if type(base_confidence) not in (int, float) or not math.isfinite(base_confidence) or not 0 <= base_confidence <= 1:
                raise ValueError("Spatial proximal confidence must be finite within [0, 1]")
            bend_confidence = _confidence(visibility, (base, middle, end))
            chart(proximal, rotation, (proximal + "_twist", proximal + "_lateral", proximal),
                  [0. if assumed else bend_confidence, base_confidence, base_confidence])
            if assumed:
                assumed_twist.append(proximal)
            distal = proximal.rsplit("_", 1)[0] + ("_knee" if knee else "_elbow")
            distal_confidence = frame.get("joint_confidence", {}).get(distal, 0.)
            if type(distal_confidence) not in (int, float) or not math.isfinite(distal_confidence) or not 0 <= distal_confidence <= 1:
                raise ValueError("Spatial distal confidence must be finite within [0, 1]")
            joints[distal], confidence[distal] = min(flexion, 2.6), min(bend_confidence, distal_confidence)
            if flexion > 2.6:
                clamped.append({"joint": distal, "inferred_radians": flexion, "model_radians": 2.6})
        references.append({"time_s": timestamp, "joints": {name: joints[name] for name in SPATIAL_JOINT_NAMES}})
        audits.append({"evidence_id": identifier, "time_s": timestamp, "pose_status": frame["status"],
                       "identity_status": frame.get("identity_status"),
                       "identity_derivation_id": frame.get("identity_derivation_id"),
                       **({"image_refinement": frame["image_refinement"]} if "image_refinement" in frame else {}),
                       "joint_confidence": confidence, "assumed_twist": assumed_twist,
                       "euler_singularities": singularities, "clamped_flexion": clamped,
                       "bounded_chart_wraps": wraps})
        previous_time = timestamp
    if not references:
        raise ValueError("Spatial reference has no usable retained poses")
    return references, {"schema_version": 1, "method": "model-world-landmarks-spatial-zyx-v1",
                        "source_sha256": evidence.get("source_sha256"), "model": deepcopy(evidence.get("model", {})),
                        "body_frame": deepcopy(body_frame), "frames": audits, "excluded_frames": excluded,
                        "bend_plane_min_sine": BEND_PLANE_MIN_SINE,
                        "scope": "Inferred model-landmark geometry; not anatomical, muscle, force or physical ground truth",
                        "assumptions": ["Rotation bounds ±2π are authored Euler-chart bounds, not anatomical joint limits",
                                        "Torso is relative to the mean anchor; limbs are relative to each inferred torso frame",
                                        "Near-collinear limbs use an assumed minimal-twist frame with zero twist confidence",
                                        "ZYX gimbal lock couples twist/flex; both chart confidences become zero",
                                        "Equivalent Euler branches minimize adjacent angle changes within bounds; unavoidable large chart wraps are audited",
                                        "Confidence inherits pose-model visibility, not independent rotation accuracy",
                                        "Cropped or missing image endpoints have zero confidence; world coordinates cannot verify image visibility",
                                        "Distal flexion above 2.6 radians is explicitly clamped and audited",
                                        "Root translation, metric calibration, contacts and physical dynamics are not recovered"]}
