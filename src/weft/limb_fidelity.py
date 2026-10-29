"""Frozen source-only limb directions; no candidate-fitted camera or side mapping."""

import json
import math
from pathlib import Path

from weft.body_fidelity import angular_error_degrees
from weft.io import canonical_hash, digest


def _xy(value):
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError("Expected two finite original-image coordinates")
    result = tuple(float(item) for item in value)
    if not all(math.isfinite(item) for item in result):
        raise ValueError("Expected finite original-image coordinates")
    return result


def _axis(start, end):
    start, end = _xy(start), _xy(end)
    return end[0] - start[0], end[1] - start[1]


def _validate(annotations):
    if annotations.get("schema_version") != 1 or not annotations.get("frames"):
        raise ValueError("Expected nonempty version-1 limb annotations")
    sources = {row["clip_id"]: row["sha256"] for row in annotations["sources"]}
    definitions = annotations["segment_definitions"]
    if len(definitions) != 8:
        raise ValueError("All eight limb segment definitions are required")
    ids = set()
    for frame in annotations["frames"]:
        if frame["id"] in ids:
            raise ValueError("Duplicate annotation ID")
        ids.add(frame["id"])
        if frame["source_sha256"] != sources[frame["clip_id"]]:
            raise ValueError("Annotation source checksum mismatch")
        if set(frame["segments"]) != set(definitions):
            raise ValueError("Every source frame must retain all segment slots")
        for name, definition in definitions.items():
            segment = frame["segments"][name]
            if [segment["start"], segment["end"]] != definition:
                raise ValueError("Segment endpoint definition mismatch")
            endpoints = [frame["points"][key]["xy"] for key in definition]
            present = all(point is not None for point in endpoints)
            if segment["scorable"] != present:
                raise ValueError("Scorable flag must match source visibility")
            for point in endpoints:
                if point is not None:
                    x, y = _xy(point)
                    if not 0 <= x < frame["width"] or not 0 <= y < frame["height"]:
                        raise ValueError("Source endpoint outside original image")
            if present:
                axis = _axis(*endpoints)
                angular_error_degrees(axis, axis)
                if not 0 < segment["angular_tolerance_degrees"] < 90:
                    raise ValueError("Invalid angular tolerance")
            elif not segment.get("unavailable_reason"):
                raise ValueError("Unavailable source label requires a reason")


def load_annotations(path, *, project_root=None):
    annotations = json.loads(Path(path).read_text())
    _validate(annotations)
    if project_root is not None:
        root = Path(project_root)
        for source in annotations["sources"]:
            if digest(root / source["path"]) != source["sha256"]:
                raise ValueError("Source video checksum mismatch")
        for frame in annotations["frames"]:
            if digest(root / frame["image_path"]) != frame["image_sha256"]:
                raise ValueError("Source frame checksum mismatch")
    return annotations


def grade_limb_projection(annotations, predictions):
    """Score frame predictions with id/source_sha256/segments[name].axis_xy.

    Segment start_xy/end_xy may replace axis_xy, in original image pixels.
    Source occlusion, absent predictions, and degenerate projections remain explicit.
    """
    _validate(annotations)
    frame_ids = {frame["id"] for frame in annotations["frames"]}
    names = set(annotations["segment_definitions"])
    indexed = {}
    for prediction in predictions:
        if prediction["id"] in indexed or prediction["id"] not in frame_ids:
            raise ValueError("Duplicate or unknown prediction ID")
        if set(prediction["segments"]) - names:
            raise ValueError("Unknown predicted segment")
        indexed[prediction["id"]] = prediction
    results = []
    for frame in annotations["frames"]:
        prediction = indexed.get(frame["id"])
        if prediction is not None and prediction["source_sha256"] != frame["source_sha256"]:
            raise ValueError("Prediction source checksum mismatch")
        for name, label in frame["segments"].items():
            row = {"id": frame["id"], "clip_id": frame["clip_id"], "segment": name,
                   "source_scorable": label["scorable"], "status": "source_unavailable",
                   "unavailable_reason": label["unavailable_reason"],
                   "angular_tolerance_degrees": label["angular_tolerance_degrees"],
                   "angular_error_degrees": None, "within_tolerance": False}
            observed = prediction["segments"].get(name) if prediction is not None else None
            # Retain source-occluded slots without pretending a prediction validates them.
            if label["scorable"]:
                row["status"] = "missing_prediction"
                if observed is not None:
                    if "axis_xy" in observed:
                        if "start_xy" in observed or "end_xy" in observed:
                            raise ValueError("Provide either an axis or endpoints, not both")
                        axis = _xy(observed["axis_xy"])
                    else:
                        axis = _axis(observed["start_xy"], observed["end_xy"])
                    if math.hypot(*axis) < 1e-10:
                        row["status"] = "degenerate_prediction"
                    else:
                        expected = _axis(frame["points"][label["start"]]["xy"],
                                         frame["points"][label["end"]]["xy"])
                        error = angular_error_degrees(expected, axis)
                        row.update(status="scored", angular_error_degrees=error,
                                   within_tolerance=error <= label["angular_tolerance_degrees"])
            results.append(row)

    def summary(rows):
        eligible = sum(row["source_scorable"] for row in rows)
        errors = [row["angular_error_degrees"] for row in rows if row["status"] == "scored"]
        passed = sum(row["within_tolerance"] for row in rows)
        return {"expected_segment_slots": len(rows), "source_scorable_segments": eligible,
                "source_unavailable_segments": len(rows) - eligible, "scored_segments": len(errors),
                "missing_predictions": sum(row["status"] == "missing_prediction" for row in rows),
                "degenerate_predictions": sum(row["status"] == "degenerate_prediction" for row in rows),
                "within_tolerance_segments": passed,
                "within_tolerance_fraction_of_source_scorable": passed / eligible if eligible else None,
                "all_source_scorable_within_tolerance": bool(eligible) and passed == eligible,
                "mean_angular_error_degrees": sum(errors) / len(errors) if errors else None,
                "max_angular_error_degrees": max(errors) if errors else None}

    return {"schema_version": 1, "benchmark_id": annotations["benchmark_id"],
            "annotation_canonical_hash": canonical_hash(annotations),
            "scope": "Assistant source visual development estimates; not expert ground truth",
            "summary": summary(results),
            "clips": {clip: summary([row for row in results if row["clip_id"] == clip])
                      for clip in sorted({row["clip_id"] for row in results})},
            "segments": {name: summary([row for row in results if row["segment"] == name])
                         for name in sorted(names)},
            "observations": results}
