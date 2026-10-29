"""Image-direction foot proposal; unchanged depth, shape, body and floor."""

from copy import deepcopy
import math
import time

import numpy as np

from weft.foot_body import foot_rotation, validate_reference
from weft.foot_reference import FOOT_IMAGE_CONTRACT
from weft.io import canonical_hash


def _image_axis(foot, width, height):
    points = []
    for name in ("heel", "foot_index"):
        point = foot.get("raw", {}).get(name, {}).get("landmarks")
        if not isinstance(point, dict):
            return None
        values = [point.get(key) for key in ("x", "y", "visibility", "presence")]
        if (any(type(v) not in (int, float) or not math.isfinite(v) or not 0 <= v <= 1 for v in values)
                or min(values[2:]) < FOOT_IMAGE_CONTRACT["minimum_endpoint_confidence"]):
            return None
        points.append(np.array(values[:2]) * [width, height])
    vector = points[1] - points[0]
    length = float(np.linalg.norm(vector))
    return vector / length if length >= FOOT_IMAGE_CONTRACT["minimum_image_axis_pixels"] else None


def refine_foot_image(spec, reference):
    """Propose a versioned reference; callers must independently evaluate it."""
    started = time.monotonic()
    validate_reference(spec, reference)
    if reference["profile"] != "kinematic-foot-reference-v1":
        raise ValueError("Foot image correction accepts only an original reference; repeated corrections are forbidden")
    metadata = reference.get("source_metadata", {})
    width, height = metadata.get("width"), metadata.get("height")
    if any(type(v) is not int or not 16 <= v <= 8192 for v in (width, height)):
        raise ValueError("Foot image correction requires bounded source pixel dimensions")
    candidate = deepcopy(reference)
    candidate.update(profile="image-directed-foot-reference-v1", contract=deepcopy(FOOT_IMAGE_CONTRACT),
        orientation_refinement={"method": FOOT_IMAGE_CONTRACT["method"], "source_profile": reference["profile"],
                                "source_reference_sha256": canonical_hash(reference), "floor_used": False,
                                "status": "proposal_not_promoted", "physical_validation": "unverified"})
    counts = {"refined": 0, "unavailable_source_pose": 0, "unavailable_image_axis": 0, "edge_on_model": 0,
              "bound_hit": 0}
    errors = {"before": [], "after": []}
    for row in candidate["frames"]:
        for side, foot in row["feet"].items():
            if foot["status"] != "kinematic_reference":
                counts["unavailable_source_pose"] += 1
                continue
            original = deepcopy(foot["reference"])
            rotation = foot_rotation(foot)
            projected = rotation[[0, 2], 0] * [1, -1]
            magnitude = float(np.linalg.norm(projected))
            length = candidate["fixed_shapes"][side]["fixed_parameters"]["length"]
            target = _image_axis(foot, width, height)
            status, angle = "refined", 0.
            if target is None:
                status = "unavailable_image_axis"
            elif length * magnitude < FOOT_IMAGE_CONTRACT["minimum_model_projection_length"]:
                status = "edge_on_model"
            else:
                current = projected / magnitude
                difference = math.atan2(current[0] * target[1] - current[1] * target[0], float(current @ target))
                bound = FOOT_IMAGE_CONTRACT["maximum_orientation_change_rad"]
                angle = float(np.clip(difference, -bound, bound))
                counts["bound_hit"] += abs(difference) > bound
                errors["before"].append(abs(difference))
                errors["after"].append(abs(difference - angle))
            counts[status] += 1
            c, s = math.cos(angle), math.sin(angle)
            corrected = np.array([[c, 0., s], [0., 1., 0.], [-s, 0., c]]) @ rotation
            foot["image_refinement"] = {"status": status, "angle_change_rad": angle,
                                         "source_world_reference": original,
                                         "image_axis_y_down": target.tolist() if target is not None else None}
            foot["reference"].update(heel_to_index_axis=corrected[:, 0].tolist(),
                                      axis_to_ankle_direction=corrected[:, 2].tolist(),
                                      scope="Bounded image-direction proposal; depth and anatomical roll remain unverified")
            shape = candidate["fixed_shapes"][side]["fixed_parameters"]
            heel = -shape["along"] * corrected[:, 0] - shape["height"] * corrected[:, 2]
            toe = heel + shape["length"] * corrected[:, 0]
            foot["endpoint_geometry"].update(heel_relative=heel.tolist(), foot_index_relative=toe.tolist())
    candidate["orientation_refinement"].update(counts=counts, direction_agreement={
        "scope": "Agreement with pose-estimated image directions, not independent source accuracy",
        "samples": len(errors["before"]),
        "mean_before_rad": float(np.mean(errors["before"])) if errors["before"] else None,
        "mean_after_rad": float(np.mean(errors["after"])) if errors["after"] else None},
        runtime_seconds=time.monotonic() - started)
    validate_reference(spec, candidate)
    return candidate
