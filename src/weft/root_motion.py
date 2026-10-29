"""Image-pelvis references under authored scale assumptions, not physical root recovery."""

from copy import deepcopy
import math
import re
from types import MappingProxyType

import numpy as np


ROOT_MOTION_CONTRACT = MappingProxyType({
    "method": "image-pelvis-fixed-torso-scale-v1",
    "maximum_samples": 900,
    "minimum_scale_samples": 3,
    "minimum_confidence": .5,
    "minimum_torso_pixels": 8.,
    "minimum_observed_reference_fraction": .8,
    "maximum_interpolated_samples": 2,
    "maximum_reference_gap_seconds": .25,
    "maximum_translation_model_units": 5.,
    "scale_model_units_per_pixel_bounds": (.00005, .1),
})
TORSO_POINTS = (11, 12, 23, 24)


def _number(value, name, lower, upper):
    if type(value) not in (int, float) or not math.isfinite(value) or not lower <= value <= upper:
        raise ValueError(f"Invalid {name}")
    return float(value)


def _integer(value, name, lower, upper):
    if type(value) is not int or not lower <= value <= upper:
        raise ValueError(f"Invalid {name}")
    return value


def _observation(frame, width, height):
    original = frame.get("identity_raw", frame)
    if not isinstance(original, dict):
        raise ValueError("Raw identity record must remain a source record")
    if any(original.get(key) != frame[key] for key in ("id", "frame", "time_s")):
        raise ValueError("Raw identity provenance must match its source frame")
    result = {"evidence_id": frame["id"], "frame": frame["frame"], "time_s": frame["time_s"],
              "decoder_timestamp_s": frame.get("decoder_timestamp_s"),
              "source_pose_status": original.get("status"), "status": "unavailable_pose",
              "landmark_source": "identity_raw.landmarks" if "identity_raw" in frame else "landmarks",
              "source_landmarks": {}, "confidence": None, "pelvis_image_px": None,
              "torso_image_length_px": None}
    landmarks = original.get("landmarks")
    if not isinstance(landmarks, list) or len(landmarks) != 33:
        result["status"] = "unavailable_landmarks"
        return result
    points = []
    confidences = []
    for index in TORSO_POINTS:
        point = landmarks[index]
        if not isinstance(point, dict):
            raise ValueError("Image landmark must be a coordinate/confidence record")
        try:
            x = _number(point.get("x"), "image x", -10, 10)
            y = _number(point.get("y"), "image y", -10, 10)
            visibility = _number(point.get("visibility"), "visibility", 0, 1)
            presence = _number(point.get("presence"), "presence", 0, 1)
        except ValueError:
            result["status"] = "invalid_landmark"
            return result
        result["source_landmarks"][str(index)] = {"x": x, "y": y, "visibility": visibility, "presence": presence}
        points.append((x * width, y * height))
        confidences.append(min(visibility, presence))
    result["confidence"] = min(confidences)
    if original.get("status") not in {"tracked", "tracked_partial"}:
        return result
    if any(not 0 < point["x"] < 1 or not 0 < point["y"] < 1 for point in result["source_landmarks"].values()):
        result["status"] = "cropped_landmark"
        return result
    if result["confidence"] < ROOT_MOTION_CONTRACT["minimum_confidence"]:
        result["status"] = "low_confidence"
        return result
    points = np.asarray(points)
    pelvis = (points[2] + points[3]) / 2
    shoulder = (points[0] + points[1]) / 2
    torso = float(np.linalg.norm(shoulder - pelvis))
    if not ROOT_MOTION_CONTRACT["minimum_torso_pixels"] <= torso <= .75 * math.hypot(width, height):
        result["status"] = "degenerate_torso_projection"
        return result
    result.update(status="observed_image_estimate", pelvis_image_px=pelvis.tolist(), torso_image_length_px=torso)
    return result


def build_root_reference(evidence, *, torso_length=None):
    """Return aligned references and readiness; unknown samples retain null positions."""
    if (not isinstance(evidence, dict) or type(evidence.get("schema_version")) is not int
            or evidence["schema_version"] != 1 or evidence.get("profile") != "exercise-pose-v1"):
        raise ValueError("Root references require exercise-pose-v1 evidence")
    source_hash = evidence.get("source_sha256")
    if not isinstance(source_hash, str) or not re.fullmatch(r"[0-9a-f]{64}", source_hash):
        raise ValueError("Root references require a pinned source hash")
    width = _integer(evidence.get("width"), "source width", 64, 4096)
    height = _integer(evidence.get("height"), "source height", 64, 2160)
    fps = _number(evidence.get("fps"), "source fps", 1, 240)
    frame_count = _integer(evidence.get("frame_count"), "source frame count", 10, 1800)
    if frame_count / fps > 60:
        raise ValueError("Root references exceed the 60-second source bound")
    if torso_length is None:
        lengths = evidence.get("lengths")
        if not isinstance(lengths, dict):
            raise ValueError("Root references require model torso length")
        torso_length = lengths.get("torso")
    torso_length = _number(torso_length, "model torso length", .18, .9)
    frames, reference = evidence.get("frames"), evidence.get("reference")
    if (not isinstance(frames, list) or not isinstance(reference, list)
            or not 1 <= len(frames) <= ROOT_MOTION_CONTRACT["maximum_samples"]
            or not 1 <= len(reference) <= len(frames)):
        raise ValueError("Root references require 1–900 bounded source samples and aligned references")
    observations = []
    by_id = {}
    previous_frame, previous_time = -1, -1.
    for frame in frames:
        if not isinstance(frame, dict):
            raise ValueError("Source frame must be a record")
        index = _integer(frame.get("frame"), "source frame index", 0, frame_count - 1)
        timestamp = _number(frame.get("time_s"), "source time", 0, 60)
        if (index <= previous_frame or timestamp <= previous_time
                or not math.isclose(timestamp, index / fps, abs_tol=1e-7)):
            raise ValueError("Source frame indices/times must increase and agree with fps")
        identifier = frame.get("id")
        if identifier != f"{source_hash}:frame:{index}":
            raise ValueError("Source frame identity does not match its hash/index")
        decoder_time = frame.get("decoder_timestamp_s")
        if decoder_time is not None:
            _number(decoder_time, "decoder timestamp", 0, 60)
        observation = _observation(frame, width, height)
        observations.append(observation)
        by_id[identifier] = observation
        previous_frame, previous_time = index, timestamp
    aligned = []
    previous_time = -1.
    for row in reference:
        if not isinstance(row, dict):
            raise ValueError("Reference must be a record")
        identifier = row.get("evidence_id")
        if not isinstance(identifier, str):
            raise ValueError("Reference identity must be a source-frame ID")
        observation = by_id.get(identifier)
        timestamp = _number(row.get("time_s"), "reference time", 0, 60)
        if observation is None or timestamp != observation["time_s"] or timestamp <= previous_time:
            raise ValueError("Reference times/IDs must map exactly to ordered source frames")
        aligned.append(observation)
        previous_time = timestamp
    if evidence.get("origin_time_s", reference[0]["time_s"]) != reference[0]["time_s"]:
        raise ValueError("Root origin must match the first authored exercise reference")
    scale_samples = [row for row in aligned if row["status"] == "observed_image_estimate"]
    errors = []
    scale = None
    if len(scale_samples) < ROOT_MOTION_CONTRACT["minimum_scale_samples"]:
        errors.append("Insufficient confident uncropped torso samples for fixed scale")
    else:
        lengths = [row["torso_image_length_px"] for row in scale_samples]
        median = float(np.median(lengths))
        value = torso_length / median
        lo, hi = ROOT_MOTION_CONTRACT["scale_model_units_per_pixel_bounds"]
        if not lo <= value <= hi:
            errors.append("Fixed image/model scale exceeds the supported envelope")
        else:
            scale = {"status": "authored_uncalibrated_model_scale", "method": "model_torso_divided_by_median_image_torso",
                     "model_torso_length": torso_length, "torso_image_median_px": median,
                     "model_units_per_pixel": value, "samples": len(lengths),
                     "torso_image_p10_px": float(np.percentile(lengths, 10)),
                     "torso_image_p90_px": float(np.percentile(lengths, 90)),
                     "evidence_ids": [row["evidence_id"] for row in scale_samples]}
    origin = aligned[0]
    origin_px = origin["pelvis_image_px"]
    if origin_px is None:
        errors.append("The first reference lacks an observed pelvis origin; later samples cannot replace it")
    output = []
    for observation in aligned:
        row = {"evidence_id": observation["evidence_id"], "frame": observation["frame"],
               "time_s": observation["time_s"], "decoder_timestamp_s": observation["decoder_timestamp_s"],
               "status": observation["status"], "confidence": observation["confidence"],
               "pelvis_image_px": deepcopy(observation["pelvis_image_px"]),
               "image_displacement_px": None, "translation": None,
               "depth": {"status": "unobservable", "observed_value": None, "authored_model_offset": 0.},
               "derivation_evidence_ids": [observation["evidence_id"]]}
        if observation["pelvis_image_px"] is not None and origin_px is not None:
            displacement = np.asarray(observation["pelvis_image_px"]) - origin_px
            row["image_displacement_px"] = displacement.tolist()
            if scale is not None:
                translation = [float(displacement[0] * scale["model_units_per_pixel"]), 0.,
                               float(-displacement[1] * scale["model_units_per_pixel"])]
                if max(abs(value) for value in translation) <= ROOT_MOTION_CONTRACT["maximum_translation_model_units"]:
                    row["translation"] = translation
                else:
                    row["status"] = "translation_outside_envelope"
        output.append(row)
    observed_count = sum(row["translation"] is not None for row in output)
    gaps = []
    index = 0
    while index < len(output):
        if output[index]["translation"] is not None:
            index += 1
            continue
        start = index
        while index < len(output) and output[index]["translation"] is None:
            index += 1
        end = index
        left = output[start-1] if start else None
        right = output[end] if end < len(output) else None
        span = right["time_s"] - left["time_s"] if left and right else None
        interpolate = (left is not None and right is not None and end-start <= ROOT_MOTION_CONTRACT["maximum_interpolated_samples"]
                       and span <= ROOT_MOTION_CONTRACT["maximum_reference_gap_seconds"] + 1e-9
                       and all(row["status"] != "translation_outside_envelope" for row in output[start:end]))
        gaps.append({"evidence_ids": [row["evidence_id"] for row in output[start:end]],
                     "bracket_seconds": span, "status": "interpolated_reference_only" if interpolate else "unfilled"})
        if interpolate:
            for row in output[start:end]:
                fraction = (row["time_s"] - left["time_s"]) / span
                row.update(status="interpolated_reference_only", observed_status=row["status"],
                           translation=((1-fraction)*np.asarray(left["translation"]) + fraction*np.asarray(right["translation"])).tolist(),
                           derivation_evidence_ids=[left["evidence_id"], right["evidence_id"]])
    source_positions = {row["evidence_id"]: index for index, row in enumerate(observations)}
    reference_intervals = []
    for left, right in zip(output, output[1:]):
        omitted = observations[source_positions[left["evidence_id"]]+1:source_positions[right["evidence_id"]]]
        span = right["time_s"]-left["time_s"]
        reference_intervals.append({"left_evidence_id": left["evidence_id"], "right_evidence_id": right["evidence_id"],
                                    "seconds": span, "omitted_source_evidence_ids": [row["evidence_id"] for row in omitted],
                                    "status": "unsupported_gap" if span > ROOT_MOTION_CONTRACT["maximum_reference_gap_seconds"] + 1e-9
                                    else "bounded_reference_interval_not_continuous_observation"})
    intervals = [row for row in reference_intervals if row["status"] == "unsupported_gap"]
    coverage = observed_count / len(output)
    if coverage < ROOT_MOTION_CONTRACT["minimum_observed_reference_fraction"]:
        errors.append("Observed root-reference coverage is below the fixed minimum")
    if any(row["translation"] is None for row in output):
        errors.append("Unfilled root references cannot drive a mobile model")
    if intervals:
        errors.append("Reference time gaps exceed the bounded interpolation horizon")
    return {"schema_version": 1, "profile": "image-pelvis-root-reference-v1",
            "status": "insufficient_reference" if errors else "ready", "source_sha256": source_hash,
            "contract": dict(ROOT_MOTION_CONTRACT), "reference": output, "observations": observations,
            "origin": {"evidence_id": origin["evidence_id"], "time_s": origin["time_s"], "pelvis_image_px": deepcopy(origin_px)},
            "scale": scale, "quality": {"errors": errors, "observed_reference_fraction": coverage,
                                         "gaps": gaps, "reference_intervals": reference_intervals, "long_reference_intervals": intervals},
            "coordinate_mapping": {"image": "source pixels: x right, y down",
                                   "model": "x right, y depth unchanged, z up",
                                   "translation": "[image_dx * fixed_scale, authored_zero_depth, -image_dy * fixed_scale]"},
            "physical_translation_verified": False, "contact_inference": "none",
            "assumptions": ["Fixed camera is assumed, not verified or motion-corrected; camera movement also moves image landmarks.",
                            "Weak perspective and a fixed image/model torso scale are authored assumptions, not metric calibration.",
                            "Camera up is assumed gravity-up; roll, tilt, perspective, and body foreshortening remain unresolved.",
                            "Image landmarks are model estimates; source depth is ignored and model depth displacement stays zero.",
                            "Short interpolations are derived control references, never observed pelvis positions or contact evidence."]}
