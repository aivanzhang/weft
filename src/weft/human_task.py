"""Choose an inspectable joint-control task without claiming recovered intent."""

import math

import numpy as np

from weft.errors import UnsupportedVideo
from weft.human import JOINT_LIMITS, JOINT_NAMES, joint_contract


def propose_joint_task(evidence, spec=None):
    if spec is not None and "provenance_projection" in spec:
        raise UnsupportedVideo("Joint-task selection requires the full body evidence, not a compact runtime projection")
    frames = evidence.get("reference", [])
    names, limits = JOINT_NAMES, JOINT_LIMITS
    if spec is not None and spec["profile"] in {"exercise-v2", "exercise-v3"}:
        names, limits = joint_contract(spec)
        source_rows = spec.get("reference_evidence")
        if (not isinstance(source_rows, list) or not source_rows
                or any(not isinstance(row, dict) or not isinstance(row.get("joint_confidence"), dict)
                       for row in source_rows)):
            raise UnsupportedVideo("Joint-task selection requires full per-joint confidence evidence")
        confidence = {row["time_s"]: row["joint_confidence"] for row in source_rows}
        frames = [{**row, "joint_confidence": confidence.get(row["time_s"], {})} for row in spec["reference"]]
    candidates = []
    for name in names:
        values = [row["joints"][name] for row in frames
                  if row.get("joint_confidence", {}).get(name, 0) >= .5
                  and math.isfinite(row["joints"][name])]
        coverage = len(values) / max(1, len(frames))
        clipped = np.clip(values, *limits[name])
        motion = float(np.percentile(clipped, 90) - np.percentile(clipped, 10)) if len(values) >= 8 else 0
        candidates.append({"joint": name, "reliable_samples": len(values), "coverage": coverage,
                           "robust_motion_radians": motion, "score": motion * coverage,
                           "eligible": len(values) >= 8 and coverage >= .65 and motion >= .1})
    eligible = [row for row in candidates if row["eligible"]]
    if not eligible:
        raise UnsupportedVideo("No joint has enough reliable, represented motion for a joint-control learning task")
    selected = max(eligible, key=lambda row: row["score"])
    return {"schema_version": 1, "selected_joint": selected["joint"], "candidates": candidates,
            "selection_rule": "Largest confidence-covered 10th–90th percentile motion within model joint limits",
            "provenance": "Authored single-joint task proposal using estimated motion, not recovered exercise intent",
            "scope": "Train a joint toward one reference-derived target; not whole-exercise imitation",
            "source_sha256": evidence.get("source_sha256")}
