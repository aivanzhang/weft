"""Local image-patch consistency at original pose seeds; never corrected joints."""

from collections import Counter
import hashlib
from io import BytesIO
import json
import math
import re
import time
from types import MappingProxyType

import cv2
import numpy as np
from PIL import Image


PROFILE = "source-pixel-pose-support-v1"
BINDING_PROFILE = "source-frame-pixel-bindings-v1"
JOINTS = ((11, "left_shoulder"), (12, "right_shoulder"), (13, "left_elbow"),
          (14, "right_elbow"), (15, "left_wrist"), (16, "right_wrist"),
          (23, "left_hip"), (24, "right_hip"), (25, "left_knee"),
          (26, "right_knee"), (27, "left_ankle"), (28, "right_ankle"))
CONTRACT = MappingProxyType({
    "maximum_images": 4, "maximum_slots": 36, "maximum_image_pixels": 4_194_304,
    "maximum_png_bytes": 16 * 1024 * 1024, "maximum_evidence_bytes": 8 * 1024 * 1024,
    "maximum_binding_bytes": 16 * 1024, "maximum_report_bytes": 128 * 1024,
    "maximum_seconds": 5., "maximum_pair_seconds": .2, "minimum_confidence": .5,
    "patch_size": 11, "search_radius_px": 16, "minimum_patch_std": 5.,
    "minimum_correlation": .90, "minimum_unique_margin": .10,
    "competitor_exclusion_px": 2, "maximum_roundtrip_px": 1.,
    "not_contradicted_residual_px": 2., "contradicted_residual_px": 5.,
    "coordinate_convention": "Original pixel-cell edges; first pixel center is (0.5, 0.5)",
    "search": "Same fractional seed offset, integer displacement grid; no subpixel refinement",
})
_SHA = re.compile(r"[0-9a-f]{64}")


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _bytes(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _json(raw, maximum):
    if not isinstance(raw, bytes) or not 0 < len(raw) <= maximum:
        raise ValueError("JSON input exceeds its fixed byte bound")
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("Duplicate JSON key")
            result[key] = value
        return result
    def nonfinite(_):
        raise ValueError("Nonfinite JSON input")
    def finite_float(raw_number):
        number = float(raw_number)
        if not math.isfinite(number):
            raise ValueError("Nonfinite JSON input")
        return number
    try:
        value = json.loads(raw, object_pairs_hook=pairs, parse_constant=nonfinite, parse_float=finite_float)
    except (RecursionError, UnicodeError) as error:
        raise ValueError("Invalid bounded JSON input") from error
    if not isinstance(value, dict):
        raise ValueError("JSON input must be an object")
    return value


def _finite(value, low=0., high=1.):
    if type(value) not in (int, float) or not low <= value <= high or not math.isfinite(value):
        raise ValueError("Invalid finite input")
    return value


def _digest(value):
    if not isinstance(value, str) or _SHA.fullmatch(value) is None:
        raise ValueError("Expected an exact lowercase SHA-256 identity")
    return value


def _check(deadline):
    if time.monotonic() >= deadline:
        raise TimeoutError("Pixel-support diagnostic deadline")


def _inputs(evidence_raw, images, bindings_raw, deadline):
    _check(deadline)
    evidence = _json(evidence_raw, CONTRACT["maximum_evidence_bytes"])
    bindings = _json(bindings_raw, CONTRACT["maximum_binding_bytes"])
    expected = {"schema_version", "profile", "source_sha256", "evidence_sha256",
                "source_manifest_sha256", "frames"}
    if (set(bindings) != expected or type(bindings["schema_version"]) is not int
            or bindings["schema_version"] != 1 or bindings["profile"] != BINDING_PROFILE):
        raise ValueError("Unsupported source-pixel binding document")
    source = _digest(evidence.get("source_sha256"))
    if bindings["source_sha256"] != source or bindings["evidence_sha256"] != _sha(evidence_raw):
        raise ValueError("Source or original pose-document binding changed")
    _digest(bindings["source_manifest_sha256"])
    identity = evidence.get("pose_identity", {})
    if (not isinstance(identity, dict) or "pose_reinference" in evidence or identity.get("mode", "raw") != "raw"):
        raise ValueError("Pixel support requires original unrepaired raw observations")
    width, height = evidence.get("width"), evidence.get("height")
    if (type(width) is not int or type(height) is not int or not 16 <= width <= 4096
            or not 16 <= height <= 4096 or width * height > CONTRACT["maximum_image_pixels"]):
        raise ValueError("Source dimensions exceed the fixed image bound")
    count = evidence.get("decoded_frames")
    if type(count) is not int or not 1 <= count <= 1800:
        raise ValueError("Source frame count exceeds the fixed bound")
    rows = evidence.get("frames")
    if not isinstance(rows, list) or not 1 <= len(rows) <= 900:
        raise ValueError("Original pose slots exceed the fixed bound")
    indexed, previous = {}, -1
    for row in rows:
        _check(deadline)
        if not isinstance(row, dict):
            raise ValueError("Original pose slots must be records")
        if "decoder_timestamp_s" not in row:
            raise ValueError("Native timestamp field is required; use null for an unknown timestamp")
        frame = row.get("frame")
        if (type(frame) is not int or not previous < frame < count
                or row.get("id") != f"{source}:frame:{frame}"
                or "image_refinement" in row or "identity_raw" in row):
            raise ValueError("Original source slot identity/order/refinement changed")
        _finite(row.get("time_s"), 0., 60.)
        if row.get("decoder_timestamp_s") is not None:
            _finite(row["decoder_timestamp_s"], 0., 3600.)
        indexed[frame], previous = row, frame
    entries = bindings["frames"]
    if (not isinstance(images, dict) or not isinstance(entries, list)
            or not 2 <= len(entries) <= CONTRACT["maximum_images"] or len(images) != len(entries)):
        raise ValueError("Supply exactly two to four bound original PNG images")
    names, previous_frame, previous_stamp, previous_time = [], -1, -1., -1.
    for entry in entries:
        _check(deadline)
        if not isinstance(entry, dict) or set(entry) != {"frame", "id", "time_s", "decoder_timestamp_s", "image_sha256"}:
            raise ValueError("Invalid image/source binding record")
        frame = entry["frame"]
        if type(frame) is not int or frame <= previous_frame or frame not in indexed or frame not in images:
            raise ValueError("Image frames must be ordered unique original pose slots")
        row = indexed[frame]
        if any(entry[key] != row[key] for key in ("id", "time_s", "decoder_timestamp_s")):
            raise ValueError("Image binding changed the exact source slot or clocks")
        stamp = entry["decoder_timestamp_s"]
        if stamp is not None:
            _finite(stamp, 0., 3600.)
        if (stamp is not None and stamp <= previous_stamp) or entry["time_s"] <= previous_time:
            raise ValueError("Both retained source clocks must strictly increase")
        raw = images[frame]
        if not isinstance(raw, bytes) or not 0 < len(raw) <= CONTRACT["maximum_png_bytes"]:
            raise ValueError("Original PNG exceeds its fixed byte bound")
        if _digest(entry["image_sha256"]) != _sha(raw):
            raise ValueError("Original image bytes changed")
        with Image.open(BytesIO(raw)) as image:
            if image.format != "PNG" or image.size != (width, height) or getattr(image, "n_frames", 1) != 1:
                raise ValueError("Source image dimensions/format/frame count changed")
        names.append(frame)
        previous_frame, previous_time = frame, entry["time_s"]
        if stamp is not None:
            previous_stamp = stamp
    if set(images) != set(names):
        raise ValueError("Every input image must have exactly one source binding")
    _check(deadline)
    return evidence, bindings, indexed, names


def _point(row, joint, width, height):
    if type(row.get("pose_count")) is not int or row["pose_count"] != 1:
        return None, "missing_or_competing_pose", None
    if row.get("status") not in {"tracked", "tracked_partial"}:
        return None, "unusable_original_pose", None
    points = row.get("landmarks")
    if not isinstance(points, list) or len(points) != 33 or not isinstance(points[joint], dict):
        return None, "missing_original_landmark", None
    original = {key: points[joint].get(key) for key in ("x", "y", "visibility", "presence")}
    # Invalid numeric fields cannot inflate the bounded report through arbitrary JSON values.
    if any(type(value) not in (int, float) or abs(value) > 10_000 or not math.isfinite(value)
           for value in original.values()):
        return None, "malformed_original_landmark", None
    try:
        x, y, visibility, presence = (_finite(original[key]) for key in original)
    except ValueError:
        return None, "invalid_or_outside_original_landmark", original
    if min(visibility, presence) < CONTRACT["minimum_confidence"]:
        return None, "weak_original_landmark", original
    return np.array([x * width, y * height]), None, original


def _patch(image, center, size):
    radius = (size - 1) / 2
    low, high = center - .5 - radius, center - .5 + radius
    if (low < 0).any() or (high > [image.shape[1] - 1, image.shape[0] - 1]).any():
        return None
    return cv2.getRectSubPix(image, (size, size), tuple((center - .5).astype(float))).astype(np.float32)


def _match(first, second, seed, deadline):
    _check(deadline)
    size, radius = CONTRACT["patch_size"], CONTRACT["search_radius_px"]
    template = _patch(first, seed, size)
    search = _patch(second, seed, size + 2 * radius)
    if template is None or search is None:
        return {"reason": "image_border_support"}
    std = float(template.std())
    if std < CONTRACT["minimum_patch_std"]:
        return {"reason": "weak_texture", "template_std": std}
    scores = cv2.matchTemplate(search, template, cv2.TM_CCOEFF_NORMED)
    windows = np.lib.stride_tricks.sliding_window_view(search, (size, size))
    valid = np.isfinite(scores) & (windows.std(axis=(-1, -2)) >= CONTRACT["minimum_patch_std"])
    _check(deadline)
    if not valid.any():
        return {"reason": "occluded_or_texture_unavailable", "template_std": std}
    scores = np.where(valid, np.clip(scores, -1., 1.), -np.inf)
    y, x = np.unravel_index(int(np.argmax(scores)), scores.shape)
    score = float(scores[y, x])
    yy, xx = np.indices(scores.shape)
    others = valid & (np.maximum(abs(yy - y), abs(xx - x)) > CONTRACT["competitor_exclusion_px"])
    competitor = float(scores[others].max()) if others.any() else None
    margin = score - competitor if competitor is not None else None
    result = {"template_std": std, "correlation": score, "competitor_correlation": competitor,
              "uniqueness_margin": margin, "offset_px": [int(x - radius), int(y - radius)]}
    if score < CONTRACT["minimum_correlation"]:
        result["reason"] = "occluded_or_appearance_changed"
    elif margin is None or margin < CONTRACT["minimum_unique_margin"]:
        result["reason"] = "ambiguous_patch_match"
    elif x in (0, 2 * radius) or y in (0, 2 * radius):
        result["reason"] = "search_boundary"
    else:
        result["reason"] = None
    return result


def _compare(first, second, start, end, deadline):
    forward = _match(first, second, start, deadline)
    result = {"status": "unavailable", "reason": forward["reason"], "forward": forward}
    if forward["reason"] is not None:
        return result
    correspondence = start + forward["offset_px"]
    backward = _match(second, first, correspondence, deadline)
    result["backward"] = backward
    if backward["reason"] is not None:
        result["reason"] = "reverse_" + backward["reason"]
        return result
    returned = correspondence + backward["offset_px"]
    error = float(np.linalg.norm(returned - start))
    result["roundtrip_error_px"] = error
    if error > CONTRACT["maximum_roundtrip_px"]:
        result["reason"] = "roundtrip_failed"
        return result
    residual = float(np.linalg.norm(correspondence - end))
    result.update(model_displacement_px=(end - start).tolist(), patch_displacement_px=forward["offset_px"],
                  model_patch_residual_px=residual, reason="residual_uncertain")
    if residual <= CONTRACT["not_contradicted_residual_px"]:
        result.update(status="not_contradicted", reason="local_patch_motion_agrees")
    elif residual >= CONTRACT["contradicted_residual_px"]:
        result.update(status="contradicted", reason="local_patch_motion_disagrees")
    return result


def diagnose_pose_pixel_support(evidence_document, image_documents, bindings_document, *, max_seconds=5.):
    """Check at most 36 source-bound slots; the caller verifies frame extraction provenance."""
    _finite(max_seconds, .01, CONTRACT["maximum_seconds"])
    started = time.monotonic()
    # Final serialization has a separate reservation inside the same cooperative deadline.
    deadline = started + max_seconds
    compute_deadline = deadline - min(.1, max_seconds / 5)
    evidence, bindings, indexed, names = _inputs(evidence_document, image_documents, bindings_document, compute_deadline)
    observation_hashes = {n: _sha(_bytes(indexed[n])) for n in names}
    rows = []
    for first, second in zip(names, names[1:]):
        for joint, name in JOINTS:
            rows.append({"frames": [first, second], "joint_index": joint, "joint": name,
                "observation_ids": [indexed[n]["id"] for n in (first, second)],
                "original_observation_sha256": [observation_hashes[n] for n in (first, second)],
                "time_s": [indexed[n]["time_s"] for n in (first, second)],
                "native_time_s": [indexed[n]["decoder_timestamp_s"] for n in (first, second)],
                "status": "unavailable", "reason": "not_processed"})
    width, height = evidence["width"], evidence["height"]
    decoded, hashes, timed_out, processed = {}, {}, False, 0
    try:
        for row in rows:
            _check(compute_deadline)
            first, second = row["frames"]
            if any(stamp is None for stamp in row["native_time_s"]):
                row["reason"] = "unknown_native_clock"
            elif row["native_time_s"][1] - row["native_time_s"][0] > CONTRACT["maximum_pair_seconds"]:
                row["reason"] = "native_clock_gap"
            else:
                a, reason_a, raw_a = _point(indexed[first], row["joint_index"], width, height)
                b, reason_b, raw_b = _point(indexed[second], row["joint_index"], width, height)
                row["original_landmarks"] = [raw_a, raw_b]
                if reason_a or reason_b:
                    row["reason"] = reason_a or reason_b
                else:
                    decoded = {n: gray for n, gray in decoded.items() if n in (first, second)}
                    for n in (first, second):
                        if n not in decoded:
                            _check(compute_deadline)
                            with Image.open(BytesIO(image_documents[n])) as image:
                                rgb = np.array(image.convert("RGB"), dtype=np.uint8)
                            hashes[str(n)] = _sha(rgb.tobytes())
                            decoded[n] = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
                            _check(compute_deadline)
                    row.update(_compare(decoded[first], decoded[second], a, b, compute_deadline))
            processed += 1
    except TimeoutError:
        timed_out = True
        for row in rows[processed:]:
            row.update(status="unavailable", reason="deadline_unprocessed")
    result = {"schema_version": 1, "profile": PROFILE, "status": "partial_deadline" if timed_out else "complete",
        "contract": dict(CONTRACT), "source_sha256": evidence["source_sha256"],
        "source_manifest_sha256": bindings["source_manifest_sha256"], "evidence_sha256": _sha(evidence_document),
        "source_provenance_status": "caller_bound_not_independently_verified",
        "bindings_sha256": _sha(bindings_document), "image_bindings": bindings["frames"],
        "decoded_rgb_sha256": hashes, "source_pose_slots": len(evidence["frames"]),
        "source_decoded_frames": evidence["decoded_frames"], "sampled_image_frames": names,
        "pair_count": len(names) - 1, "slot_count": len(rows), "processed_slots": processed,
        "status_counts": dict(Counter(row["status"] for row in rows)), "rows": rows,
        "scope": "Conditional patch correspondence at original model seeds; not joint identity, anatomy or 3D accuracy.",
        "limitations": ["A matched patch may be background or another body part; agreement cannot validate the original pose.",
            "Local uniqueness excludes nearby competitors only; occlusion, repetition and edits can remain unresolved.",
            "Only supplied image pairs are checked; intervening or omitted source frames are not qualified.",
            "Frame extraction provenance is caller-sealed; this helper never decodes the source video.",
            "The source-manifest hash is caller-supplied; this helper does not read or verify that manifest.",
            "All original pose bytes, confidence, source clocks, task and quality gates remain unchanged."],
        "work": {"model_calls": 0, "video_decodes": 0, "physics_steps": 0,
                 "png_decodes": len(hashes), "max_patch_matches": 2 * len(rows)},
        "deadline_scope": "Cooperative around bounded PNG and NCC calls, not native-call preemption",
        "seconds": time.monotonic() - started}
    if len(_bytes(result)) > CONTRACT["maximum_report_bytes"]:
        raise ValueError("Pixel-support report exceeds its fixed byte bound")
    _check(deadline)
    return result
