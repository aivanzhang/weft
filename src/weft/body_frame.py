"""Fit an anchored camera-relative body frame without inventing world calibration."""

import math

import numpy as np


# Camera x/right, y/down, z/depth become simulator x/right, y/depth, z/up.
CAMERA_TO_SIM = np.array([[1., 0., 0.], [0., 0., 1.], [0., -1., 0.]])


def _unit(vector):
    length = float(np.linalg.norm(vector))
    if not math.isfinite(length) or length < .03:
        raise ValueError("Torso frame has degenerate or nonfinite geometry")
    return vector / length


def frame_rotation(landmarks):
    if not isinstance(landmarks, list) or len(landmarks) != 33:
        raise ValueError("Torso orientation requires 33 model-world landmarks")
    points = np.array([[row[key] for key in ("x", "y", "z")] for row in landmarks], dtype=float)
    if not np.isfinite(points).all() or np.abs(points).max() > 10:
        raise ValueError("Torso landmarks exceed the inferred coordinate envelope")
    up = _unit((points[11] + points[12] - points[23] - points[24]) / 2)
    lateral = points[12] - points[11]
    right = _unit(lateral - np.dot(lateral, up) * up)
    forward = np.cross(up, right)
    return CAMERA_TO_SIM @ np.column_stack((right, forward, up))


def fit_body_frame(evidence):
    frames = evidence.get("frames", [])
    rotations, references, rejected = [], [], []
    for frame in frames:
        if frame.get("status") not in {"tracked", "tracked_partial"}:
            continue
        try:
            rotation = frame_rotation(frame["world_landmarks"])
        except (ValueError, KeyError, TypeError):
            rejected.append(frame.get("id"))
            continue
        rotations.append(rotation)
        references.append({"id": frame["id"], "time_s": frame["time_s"], "rotation": rotation.tolist(),
                           "identity_derivation_id": frame.get("identity_derivation_id")})
    if not rotations:
        return {"schema_version": 1, "status": "assumed", "rotation": np.eye(3).tolist(),
                "evidence_ids": [], "reference": [], "rejected_evidence_ids": rejected,
                "scope": "Legacy upright anchor: no usable torso orientation evidence was supplied"}
    # The polar factor is a proper mean rotation; it does not fit the scoring labels.
    u, _, vt = np.linalg.svd(np.mean(rotations, axis=0))
    orientation = u @ np.diag([1., 1., np.linalg.det(u @ vt)]) @ vt
    residuals = [math.acos(float(np.clip(np.dot(orientation[:, 2], value[:, 2]), -1, 1))) for value in rotations]
    return {"schema_version": 1, "status": "inferred", "rotation": orientation.tolist(),
            "evidence_ids": [row["id"] for row in references], "reference": references,
            "rejected_evidence_ids": rejected,
            "fit": {"method": "Proper polar mean of reliable model-world torso frames",
                    "samples": len(rotations), "torso_axis_residual_p95_radians": float(np.percentile(residuals, 95)),
                    "torso_tilt_from_camera_up_radians": math.acos(float(np.clip(orientation[2, 2], -1, 1)))},
            "scope": "Inferred camera-relative orientation; root remains fixed, not recovered balance or ground support",
            "gravity_assumption": "Camera up is assumed aligned with gravity; camera roll/tilt and metric scale are uncalibrated"}


def validate_body_frame(frame):
    if not isinstance(frame, dict) or type(frame.get("schema_version")) is not int or frame["schema_version"] != 1:
        raise ValueError("Unsupported anchored body-frame schema")
    if frame.get("status") not in {"inferred", "assumed"}:
        raise ValueError("Body-frame status must remain inferred or assumed")
    if not isinstance(frame.get("evidence_ids"), list) or any(not isinstance(value, str) for value in frame["evidence_ids"]):
        raise ValueError("Body-frame evidence IDs are required")
    if frame["status"] == "inferred" and not frame["evidence_ids"]:
        raise ValueError("Inferred body frame needs evidence references")
    reference = frame.get("reference", [])
    if not isinstance(reference, list) or not all(isinstance(row, dict) for row in reference):
        raise ValueError("Body-frame reference must be a list of orientation samples")
    rows = [{"rotation": frame.get("rotation")}] + reference
    if len(rows) > 1801:
        raise ValueError("Body-frame samples exceed the video bound")
    previous = -1.
    for index, row in enumerate(rows):
        matrix = np.asarray(row.get("rotation"), dtype=float)
        if (matrix.shape != (3, 3) or not np.isfinite(matrix).all()
                or not np.allclose(matrix.T @ matrix, np.eye(3), atol=1e-6)
                or not math.isclose(float(np.linalg.det(matrix)), 1., abs_tol=1e-6)):
            raise ValueError("Body orientation must be a proper orthonormal rotation")
        if index:
            time = row.get("time_s")
            if type(time) not in (int, float) or not math.isfinite(time) or not previous < time <= 60:
                raise ValueError("Body-frame times must increase within the video bound")
            if row.get("id") not in frame["evidence_ids"]:
                raise ValueError("Body-frame sample has a dangling evidence reference")
            previous = time
    return frame
