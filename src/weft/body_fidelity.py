"""Candidate-independent image-plane torso-axis development checks."""

import json
import math
from pathlib import Path

from weft.io import canonical_hash, digest


def _xy(value):
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError("Expected a finite two-component image-plane vector")
    point = tuple(float(component) for component in value)
    if not all(math.isfinite(component) for component in point):
        raise ValueError("Expected finite image-plane coordinates")
    return point


def _axis(hip, shoulder):
    hip, shoulder = _xy(hip), _xy(shoulder)
    return shoulder[0] - hip[0], shoulder[1] - hip[1]


def angular_error_degrees(expected, observed):
    expected, observed = _xy(expected), _xy(observed)
    if math.hypot(*expected) < 1e-10 or math.hypot(*observed) < 1e-10:
        raise ValueError("Torso direction requires a nonzero axis")
    # Direction matters: an upside-down body must not match the upright torso.
    cross = expected[0] * observed[1] - expected[1] * observed[0]
    dot = expected[0] * observed[0] + expected[1] * observed[1]
    return math.degrees(math.atan2(abs(cross), dot))


def load_annotations(path, *, project_root=None):
    """Load frozen labels, optionally checking source videos and decoded images."""
    document = json.loads(Path(path).read_text())
    _validate_annotations(document)
    if project_root is not None:
        root = Path(project_root)
        for source in document["sources"]:
            if digest(root / source["path"]) != source["sha256"]:
                raise ValueError("Source video checksum mismatch")
        for frame in document["frames"]:
            if digest(root / frame["image_path"]) != frame["image_sha256"]:
                raise ValueError("Source frame checksum mismatch")
    return document


def _validate_annotations(document):
    if document.get("schema_version") != 1 or not document.get("frames"):
        raise ValueError("Expected nonempty version-1 body annotations")
    ids = set()
    sources = {row["clip_id"]: row for row in document["sources"]}
    for row in document["frames"]:
        if row["id"] in ids:
            raise ValueError("Duplicate annotation ID")
        ids.add(row["id"])
        if row["source_sha256"] != sources[row["clip_id"]]["sha256"]:
            raise ValueError("Annotation source checksum mismatch")
        for field in ("hip_xy", "shoulder_xy"):
            x, y = _xy(row[field])
            if not 0 <= x < row["width"] or not 0 <= y < row["height"]:
                raise ValueError("Annotation must use original image pixels")
        axis = _axis(row["hip_xy"], row["shoulder_xy"])
        angular_error_degrees(axis, axis)
        if not 0 < row["angular_tolerance_degrees"] < 90:
            raise ValueError("Invalid angular tolerance")


def grade_torso_projection(annotations, predictions):
    """Grade original-image axes; do not fit a camera or transform against labels.

    Predictions contain id/source_sha256 and either axis_xy or hip_xy/shoulder_xy.
    Pixel endpoints additionally report position error, normalized by label length.
    """
    _validate_annotations(annotations)
    expected_ids = {row["id"] for row in annotations["frames"]}
    indexed = {}
    for row in predictions:
        if row["id"] in indexed or row["id"] not in expected_ids:
            raise ValueError("Duplicate or unknown prediction ID")
        indexed[row["id"]] = row
    results = []
    for label in annotations["frames"]:
        result = {"id": label["id"], "clip_id": label["clip_id"],
                  "angular_tolerance_degrees": label["angular_tolerance_degrees"],
                  "angular_error_degrees": None, "within_tolerance": False,
                  "normalized_endpoint_error": None, "status": "missing"}
        prediction = indexed.get(label["id"])
        if prediction is not None:
            if prediction["source_sha256"] != label["source_sha256"]:
                raise ValueError("Prediction source checksum mismatch")
            expected = _axis(label["hip_xy"], label["shoulder_xy"])
            if "axis_xy" in prediction:
                if "hip_xy" in prediction or "shoulder_xy" in prediction:
                    raise ValueError("Provide either an axis or endpoints, not both")
                observed = _xy(prediction["axis_xy"])
            else:
                observed = _axis(prediction["hip_xy"], prediction["shoulder_xy"])
                errors = [math.dist(_xy(prediction[key]), _xy(label[key]))
                          for key in ("hip_xy", "shoulder_xy")]
                result["normalized_endpoint_error"] = sum(errors) / (2 * math.hypot(*expected))
            error = angular_error_degrees(expected, observed)
            result.update(status="scored", angular_error_degrees=error,
                          within_tolerance=error <= label["angular_tolerance_degrees"])
        results.append(result)

    def summarize(rows):
        errors = [row["angular_error_degrees"] for row in rows if row["status"] == "scored"]
        return {"expected_frames": len(rows), "scored_frames": len(errors),
                "missing_frames": len(rows) - len(errors),
                "within_tolerance_frames": sum(row["within_tolerance"] for row in rows),
                "all_within_tolerance": all(row["within_tolerance"] for row in rows),
                "mean_angular_error_degrees": sum(errors) / len(errors) if errors else None,
                "max_angular_error_degrees": max(errors) if errors else None}

    return {"schema_version": 1, "benchmark_id": annotations["benchmark_id"],
            "annotation_canonical_hash": canonical_hash(annotations),
            "scope": "Assistant visual development estimates; not validated biomechanics",
            "summary": summarize(results),
            "clips": {clip: summarize([row for row in results if row["clip_id"] == clip])
                      for clip in sorted({row["clip_id"] for row in results})},
            "frames": results}
