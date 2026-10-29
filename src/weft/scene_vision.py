"""Bounded color tracking for an explicitly planar, fixed-camera scene."""

from copy import deepcopy
import math
from pathlib import Path
import re

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageSequence

from weft.errors import UnsupportedVideo
from weft.video_io import webm_video_frame_count
from weft.io import canonical_hash, digest, write_json
from weft.reconstruction_coverage import coverage_summary, plan_reconstruction


MAX_FRAMES = 1800
MAX_SECONDS = 60
MAX_BYTES = 100_000_000
MAX_OBJECTS = 8
SCOPE = "Same-video development reconstruction; not an independent physical validation or final holdout"


def _number(value, name, low, high):
    if type(value) not in (int, float) or not math.isfinite(value) or not low <= value <= high:
        raise ValueError(f"{name} must be finite and between {low} and {high}")
    return float(value)


def _point(value, name, low, high):
    if not isinstance(value, list) or len(value) != 2:
        raise ValueError(f"{name} must be a two-number list")
    return [_number(v, name, low, high) for v in value]


def validate_config(config):
    config = {} if config is None else deepcopy(config)
    if not isinstance(config, dict):
        raise ValueError("Scene config must be an object")
    allowed = {"objects", "controlled_id", "target", "gravity", "damping", "restitution", "source_label"}
    if set(config) - allowed:
        raise ValueError("Unknown scene settings: " + ", ".join(sorted(set(config) - allowed)))
    for key in ("controlled_id", "source_label"):
        if key in config and (not isinstance(config[key], str) or not 1 <= len(config[key]) <= 200):
            raise ValueError(f"{key} must be a nonempty string of at most 200 characters")
    if "target" in config:
        config["target"] = _point(config["target"], "target", 0, 2)
    if "gravity" in config:
        config["gravity"] = _point(config["gravity"], "gravity", -10, 10)
    for key, low, high in (("damping", 0.01, 1), ("restitution", 0, 1)):
        if key in config:
            config[key] = _number(config[key], key, low, high)
    if "objects" not in config:
        if "controlled_id" in config:
            raise ValueError("controlled_id requires an explicit objects list")
        return config
    objects = config["objects"]
    if not isinstance(objects, list) or not 1 <= len(objects) <= MAX_OBJECTS:
        raise ValueError("objects must contain between one and eight color profiles")
    ids = set()
    fields = {"id", "label", "hue_min", "hue_max", "saturation_min", "value_min", "role", "shape", "mass"}
    hue_sets = []
    for obj in objects:
        if not isinstance(obj, dict) or set(obj) - fields:
            raise ValueError("Unknown or invalid object color profile")
        if not isinstance(obj.get("id"), str) or not re.fullmatch(r"[a-zA-Z0-9_-]{1,48}", obj["id"]):
            raise ValueError("Object id must contain 1–48 letters, numbers, underscores, or hyphens")
        if obj["id"] in ids:
            raise ValueError("Object IDs must be unique")
        ids.add(obj["id"])
        obj.setdefault("label", obj["id"])
        if not isinstance(obj["label"], str) or not 1 <= len(obj["label"]) <= 80:
            raise ValueError("Object labels must contain 1–80 characters")
        for key, default, high in (("hue_min", None, 179), ("hue_max", None, 179),
                                   ("saturation_min", 90, 255), ("value_min", 65, 255)):
            obj.setdefault(key, default)
            if type(obj[key]) is not int or not 0 <= obj[key] <= high:
                raise ValueError(f"Object {key} must be an integer between 0 and {high}")
        hues = set(range(obj["hue_min"], obj["hue_max"] + 1)) if obj["hue_min"] <= obj["hue_max"] else (
            set(range(obj["hue_min"], 180)) | set(range(0, obj["hue_max"] + 1)))
        if any(hues & other for other in hue_sets):
            raise ValueError("Object hue bands overlap; use visibly distinct object colors")
        hue_sets.append(hues)
        if obj.get("role", "passive") not in {"active", "passive", "static"}:
            raise ValueError("Object role must be active, passive, or static")
        if obj.get("shape", "circle") not in {"circle", "box"}:
            raise ValueError("Object shape must be circle or box")
        obj["mass"] = _number(obj.get("mass", 1), "mass", 0.1, 100)
    if sum(obj.get("role") == "active" for obj in objects) > 1:
        raise ValueError("Only one controlled active object is supported")
    if "controlled_id" in config and config["controlled_id"] not in ids:
        raise ValueError("controlled_id does not reference an object")
    return config


def _mask(hsv, profile):
    low, high = profile["hue_min"], profile["hue_max"]
    sat, value = profile.get("saturation_min", 90), profile.get("value_min", 65)
    if low <= high:
        return cv2.inRange(hsv, (low, sat, value), (high, 255, 255))
    return cv2.bitwise_or(cv2.inRange(hsv, (low, sat, value), (179, 255, 255)),
                         cv2.inRange(hsv, (0, sat, value), (high, 255, 255)))


def _candidates(mask, shape=None):
    height, width = mask.shape
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    result = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if not width * height * 0.0006 <= area <= width * height * 0.20:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        if x <= 0 or y <= 0 or x + w >= width or y + h >= height:
            continue
        perimeter = cv2.arcLength(contour, True)
        circularity = 4 * math.pi * area / max(perimeter ** 2, 1)
        extent = area / (w * h)
        detected_shape = "box" if extent >= 0.86 else "circle"
        requested = shape or detected_shape
        if requested == "circle" and not (circularity >= 0.62 and 0.65 <= w / h <= 1.55):
            continue
        if requested == "box" and not (extent >= 0.75 and 0.15 <= w / h <= 6.7):
            continue
        moments = cv2.moments(contour)
        result.append({"x_px": moments["m10"] / moments["m00"], "y_px": moments["m01"] / moments["m00"],
                       "radius_px": math.sqrt(area / math.pi), "width_px": w, "height_px": h,
                       "circularity": circularity, "shape": requested, "contour": contour})
    return result


def _discover(frame, hsv):
    candidates = _candidates(cv2.inRange(hsv, (0, 90, 65), (179, 255, 255)))
    if not candidates:
        return []
    if len(candidates) > MAX_OBJECTS:
        raise ValueError("More than eight colored objects; simplify the scene or supply explicit color profiles")
    profiles, hue_sets = [], []
    for index, obj in enumerate(sorted(candidates, key=lambda obj: obj["x_px"])):
        mask = np.zeros(hsv.shape[:2], np.uint8)
        cv2.drawContours(mask, [obj["contour"]], -1, 255, -1)
        pixels = hsv[:, :, 0][mask > 0]
        # A circular mean keeps red hues near 0 and 179 in one identity.
        angle = pixels.astype(float) * (2 * math.pi / 180)
        hue = int(round(math.atan2(np.sin(angle).mean(), np.cos(angle).mean()) * 180 / (2 * math.pi))) % 180
        hues = {(hue + delta) % 180 for delta in range(-9, 10)}
        if any(hues & prior for prior in hue_sets):
            raise UnsupportedVideo("Ambiguous object colors: each object needs a distinct hue; use a simpler scene")
        hue_sets.append(hues)
        rgb = np.median(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)[mask > 0], axis=0).astype(int)
        profiles.append({"id": f"object-{index + 1}", "label": f"Object {index + 1}",
                         "hue_min": (hue - 9) % 180, "hue_max": (hue + 9) % 180,
                         "saturation_min": 90, "value_min": 65, "shape": obj["shape"], "mass": 1.0,
                         "color": "#" + "".join(f"{v:02x}" for v in rgb)})
    return profiles


def _color(frame, contour):
    mask = np.zeros(frame.shape[:2], np.uint8)
    cv2.drawContours(mask, [contour], -1, 255, -1)
    rgb = np.median(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)[mask > 0], axis=0).astype(int)
    return "#" + "".join(f"{v:02x}" for v in rgb)


def _notify(progress, stage, **details):
    if progress:
        progress(stage, details)


def _camera_spread(points, width, height):
    coverage = float(np.prod(np.ptp(points, axis=0)) / (width * height))
    cells = np.clip((points / [width, height] * 3).astype(int), 0, 2)
    return coverage, len(set(map(tuple, cells)))


def _camera_alignment_result(transform, source, target, width, height, method):
    center = np.array([width / 2, height / 2], dtype=np.float32)
    probes = np.array([center, center + [1, 0], center + [0, 1]], dtype=np.float32)
    warped = cv2.perspectiveTransform(probes[None], transform)[0]
    if not np.isfinite(warped).all():
        return {"status": "unknown_nonfinite_alignment", "method": method}
    local = np.column_stack((warped[1] - warped[0], warped[2] - warped[0]))
    shift = float(np.linalg.norm(warped[0] - center) / width)
    scale = float(math.sqrt(abs(np.linalg.det(local))))
    rotation = float(math.degrees(math.atan2(local[1, 0], local[0, 0])))
    coverage, cells = _camera_spread(source, width, height)
    target_coverage, _ = _camera_spread(target, width, height)
    moving = shift > 0.01 or abs(scale - 1) > 0.02 or abs(rotation) > 1.0
    return {"status": "moving_background" if moving else "no_large_motion_detected", "method": method,
            "center_shift_image_width": shift, "scale": scale, "rotation_degrees": rotation,
            "inliers": len(source), "coverage": coverage, "target_coverage": target_coverage,
            "source_grid_cells": cells}


def _camera_descriptor_sample(reference, gray, points):
    # Direct matching survives larger jumps than LK, but missing matches never imply motion.
    if points is None or len(points) < 12:
        return {"status": "unknown_insufficient_background_features"}
    scale = min(1., 640 / max(reference.shape))
    width, height = round(reference.shape[1] * scale), round(reference.shape[0] * scale)
    if scale < 1:
        reference = cv2.resize(reference, (width, height), interpolation=cv2.INTER_AREA)
        gray = cv2.resize(gray, (width, height), interpolation=cv2.INTER_AREA)
    background = points.reshape(-1, 2) * scale
    mask = np.zeros(reference.shape, np.uint8)
    for x, y in background:
        cv2.circle(mask, (round(float(x)), round(float(y))), 18, 255, -1)
    sift = cv2.SIFT_create(nfeatures=600)
    source_keys, source_desc = sift.detectAndCompute(reference, mask)
    target_keys, target_desc = sift.detectAndCompute(gray, None)
    if source_desc is None or target_desc is None or min(len(source_desc), len(target_desc)) < 12:
        return {"status": "unknown_insufficient_descriptors"}
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    forward = matcher.knnMatch(source_desc, target_desc, k=2)
    backward = matcher.knnMatch(target_desc, source_desc, k=2)
    reverse = {pair[0].queryIdx: pair[0].trainIdx for pair in backward
               if len(pair) == 2 and pair[0].distance < .75 * pair[1].distance}
    matches = [pair[0] for pair in forward if len(pair) == 2 and pair[0].distance < .75 * pair[1].distance
               and reverse.get(pair[0].trainIdx) == pair[0].queryIdx]
    if len(matches) < 12:
        return {"status": "unknown_insufficient_mutual_matches", "matches": len(matches)}
    source = np.array([source_keys[m.queryIdx].pt for m in matches], dtype=np.float32)
    target = np.array([target_keys[m.trainIdx].pt for m in matches], dtype=np.float32)
    transform, inliers = cv2.findHomography(source, target, cv2.RANSAC, 2., maxIters=500, confidence=.995)
    if transform is None or inliers is None or int(inliers.sum()) < 12 or float(inliers.mean()) < .6:
        return {"status": "unknown_inconsistent_projective_alignment", "matches": len(matches)}
    source, target = source[inliers.ravel() == 1], target[inliers.ravel() == 1]
    coverage, cells = _camera_spread(source, width, height)
    target_coverage, _ = _camera_spread(target, width, height)
    if coverage < .2 or target_coverage < .08 or cells < 4:
        return {"status": "unknown_localized_descriptor_matches", "coverage": coverage,
                "target_coverage": target_coverage, "source_grid_cells": cells}
    try:
        inverse = np.linalg.inv(transform)
    except np.linalg.LinAlgError:
        return {"status": "unknown_singular_alignment"}
    projected = cv2.perspectiveTransform(source[None], transform)[0]
    restored = cv2.perspectiveTransform(target[None], inverse)[0]
    residual = np.maximum(np.linalg.norm(projected - target, axis=1), np.linalg.norm(restored - source, axis=1))
    if not np.isfinite(residual).all() or float(np.median(residual)) > 1.5:
        return {"status": "unknown_large_reprojection_error"}
    result = _camera_alignment_result(transform, source, target, width, height, "original_reference_sift_homography")
    result.update(matches=len(matches), inlier_ratio=len(source) / len(matches),
                  median_symmetric_reprojection_error_px=float(np.median(residual)), analysis_scale=scale)
    return result


def _camera_change_trigger(previous_gray, gray):
    """Request alignment after broad frame changes; never classify exposure or occlusion as motion."""
    if previous_gray is None or previous_gray.shape != gray.shape:
        return False
    previous = cv2.resize(previous_gray, (128, 96), interpolation=cv2.INTER_AREA).astype(np.float32)
    current = cv2.resize(gray, (128, 96), interpolation=cv2.INTER_AREA).astype(np.float32)
    difference = current - previous
    difference -= np.median(difference)
    blocks = np.abs(difference).reshape(6, 16, 8, 16).mean(axis=(1, 3))
    return bool(np.mean(blocks >= 12) >= .5)


def _camera_sample(reference, frame, points, index):
    if points is None or len(points) < 12:
        return {"frame": index, "status": "unknown_insufficient_background_features"}
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    points = np.asarray(points, dtype=np.float32)[:120]
    unknown = {"frame": index, "status": "unknown_untracked_background"}
    tracked, valid, _ = cv2.calcOpticalFlowPyrLK(reference, gray, points, None)
    if tracked is not None and valid is not None:
        backward, back_valid, _ = cv2.calcOpticalFlowPyrLK(gray, reference, np.nan_to_num(tracked), None)
        if backward is not None and back_valid is not None:
            error = np.linalg.norm(backward - points, axis=2).ravel()
            good = (valid.ravel() == 1) & (back_valid.ravel() == 1) & np.isfinite(error) & (error <= 1.5)
            source, target = points[good].reshape(-1, 2), tracked[good].reshape(-1, 2)
            unknown["status"] = "unknown_insufficient_tracked_features"
            if len(source) >= 12:
                affine, inliers = cv2.estimateAffinePartial2D(source, target, method=cv2.RANSAC,
                                                            ransacReprojThreshold=1.5, maxIters=500)
                unknown["status"] = "unknown_inconsistent_background"
                if affine is not None and inliers is not None and int(inliers.sum()) >= 12:
                    source, target = source[inliers.ravel() == 1], target[inliers.ravel() == 1]
                    height, width = gray.shape
                    coverage, cells = _camera_spread(source, width, height)
                    if coverage >= .2 and cells >= 4:
                        result = _camera_alignment_result(np.vstack((affine, [0, 0, 1])), source, target,
                                                          width, height, "original_reference_lk_affine")
                        return {"frame": index, **result}
                    unknown.update(status="unknown_localized_background", coverage=coverage, source_grid_cells=cells)
    fallback = _camera_descriptor_sample(reference, gray, points)
    if fallback["status"] in {"moving_background", "no_large_motion_detected"}:
        return {"frame": index, **fallback}
    return {**unknown, "fallback": fallback}


def _save_previews(previews, out, fps, stride):
    if not previews:
        return
    frames = [Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)) for frame in previews]
    frames[0].save(out / "tracks.gif", save_all=True, append_images=frames[1:],
                   duration=max(40, round(1000 * stride / fps)), loop=0)
    thumbnails = []
    for frame in frames[::max(1, len(frames) // 12)][:12]:
        frame = frame.copy()
        frame.thumbnail((320, 240))
        thumbnails.append(frame)
    row_height = max(frame.height for frame in thumbnails) + 22
    sheet = Image.new("RGB", (960, row_height * math.ceil(len(thumbnails) / 3)), "#F5EADC")
    draw = ImageDraw.Draw(sheet)
    for i, frame in enumerate(thumbnails):
        x, y = (i % 3) * 320, (i // 3) * row_height
        sheet.paste(frame, (x, y))
        draw.text((x + 5, y + frame.height + 3), "VIDEO MEASUREMENTS (not simulation)", fill="#214F50")
    sheet.save(out / "tracks-contact-sheet.png")


def inspect_video(video, out, config=None, progress=None):
    """Decode once, preserve measured tracks, and propose a labeled planar world."""
    config = validate_config(config)
    video, out = Path(video), Path(out)
    if not video.is_file() or video.stat().st_size > MAX_BYTES:
        raise ValueError("Input must be a local video of at most 100 MB")
    if not out.is_dir():
        raise ValueError("Output directory must already exist")
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        cap.release()
        raise ValueError("Video could not be decoded")
    fps, raw_count = cap.get(cv2.CAP_PROP_FPS), cap.get(cv2.CAP_PROP_FRAME_COUNT)
    raw_width, raw_height = cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    if not (all(math.isfinite(v) for v in (fps, raw_count, raw_width, raw_height))
            and 1 <= fps <= 240 and 10 <= raw_count <= MAX_FRAMES and raw_count / fps <= MAX_SECONDS
            and 64 <= raw_width <= 4096 and 64 <= raw_height <= 2160):
        cap.release()
        raise ValueError("Video must be 10–1800 frames, <=60 seconds, <=4096×2160, and 1–240 fps")
    count, raw_width, raw_height = int(raw_count), int(raw_width), int(raw_height)
    if not 0.2 <= raw_height / raw_width <= 2:
        cap.release()
        raise ValueError("Supported video aspect ratio is 0.2–2 image heights per image width")
    scale = min(1.0, 960 / raw_width)
    width, height = round(raw_width * scale), round(raw_height * scale)
    source_hash = digest(video)
    profiles = deepcopy(config.get("objects", []))
    automatic = not profiles
    rows, visibility, previews, previous = [], [], [], {}
    stride = max(1, math.ceil(count / 48))
    decoded = 0
    first_frame = None
    camera_reference, camera_points, camera_samples = None, None, []
    _notify(progress, "tracking", frames=count, objects=len(profiles), message="Tracking colored rigid objects")
    try:
        for index in range(count):
            ok, frame = cap.read()
            if not ok:
                break
            if frame.shape[1] != raw_width or frame.shape[0] != raw_height:
                raise ValueError("Video resolution changes within the recording")
            frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA) if scale < 1 else frame
            if first_frame is None:
                first_frame = frame.copy()
                if not cv2.imwrite(str(out / "background.jpg"), first_frame):
                    raise OSError("Could not write background image")
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            if camera_reference is None:
                camera_reference = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                background_mask = cv2.inRange(hsv, (0, 0, 40), (179, 60, 245))
                camera_points = cv2.goodFeaturesToTrack(camera_reference, maxCorners=120,
                                                       qualityLevel=0.02, minDistance=12,
                                                       mask=background_mask)
            if index and (index % max(1, round(fps / 2)) == 0 or index == count - 1):
                sample = _camera_sample(camera_reference, frame, camera_points, index)
                camera_samples.append(sample)
                if sample["status"] == "moving_background":
                    write_json(out / "camera-motion.json", {"scope": "Background motion check, not calibrated camera recovery",
                               "samples": camera_samples, "result": "unsupported_moving_camera_or_background"})
                    raise UnsupportedVideo(f"Moving camera or widespread background motion at frame {index}; fixed-camera profile cannot separate it from object motion")
            if automatic and not profiles:
                profiles = _discover(frame, hsv)
            overlay = frame.copy() if index % stride == 0 else None
            for profile in profiles:
                object_id = profile["id"]
                found = _candidates(_mask(hsv, profile), profile.get("shape"))
                if len(found) > 1:
                    raise UnsupportedVideo(f"Ambiguous identity for {object_id} at frame {index}: multiple matching shapes")
                status = "measured" if found else "not_detected_or_occluded"
                visibility.append({"object_id": object_id, "frame": index, "status": status})
                if not found:
                    continue
                detection = found[0]
                prior = previous.get(object_id)
                if prior and (index - prior["frame"]) / fps <= 0.25:
                    displacement = math.hypot(detection["x_px"] - prior["x_px"], detection["y_px"] - prior["y_px"])
                    if displacement / width > 3 * (index - prior["frame"]) / fps + 0.025:
                        raise ValueError(f"Identity jump or unsupported speed for {object_id} at frame {index}")
                if "color" not in profile:
                    profile["color"] = _color(frame, detection["contour"])
                row = {key: value for key, value in detection.items() if key != "contour"}
                row.update(id=f"{source_hash}:frame:{index}:object:{object_id}", object_id=object_id,
                           frame=index, time_s=index / fps, x=row["x_px"] / width, y=row["y_px"] / width,
                           status="measured_image_projection")
                rows.append(row)
                previous[object_id] = row
                if overlay is not None:
                    x, y = round(row["x_px"]), round(row["y_px"])
                    if row["shape"] == "circle":
                        cv2.circle(overlay, (x, y), round(row["radius_px"]), (255, 255, 255), 2)
                    else:
                        hw, hh = round(row["width_px"] / 2), round(row["height_px"] / 2)
                        cv2.rectangle(overlay, (x - hw, y - hh), (x + hw, y + hh), (255, 255, 255), 2)
                    cv2.putText(overlay, object_id, (max(0, x - 24), max(15, y - 16)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3, cv2.LINE_AA)
                    cv2.putText(overlay, object_id, (max(0, x - 24), max(15, y - 16)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
            if overlay is not None:
                cv2.putText(overlay, f"MEASURED | frame {index} | {index / fps:.2f}s", (8, height - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 1, cv2.LINE_AA)
                if overlay.shape[1] > 640:
                    overlay = cv2.resize(overlay, (640, round(height * 640 / width)))
                previews.append(overlay)
            decoded += 1
            if decoded % 120 == 0:
                _notify(progress, "tracking", decoded_frames=decoded, frames=count, objects=len(profiles))
        if cap.read()[0]:
            raise ValueError("Video frame count disagrees with metadata; no partial successful result")
    finally:
        cap.release()
    decode_audit = {"metadata_frames": count, "decoded_frames": decoded, "basis": "metadata_matches"}
    if decoded != count:
        if video.suffix.lower() not in {".webm", ".mkv"}:
            raise ValueError(f"Incomplete decode: expected {count} frames, decoded {decoded}")
        encoded = webm_video_frame_count(video)
        decode_audit.update(encoded_video_frames=encoded, basis="independent_encoded_video_blocks")
        write_json(out / "decode-audit.json", decode_audit)
        if encoded != decoded:
            raise ValueError(f"Incomplete decode: encoded {encoded} frames, decoded {decoded}")
        count = decoded
    _save_previews(previews, out, fps, stride)
    if not profiles or not rows:
        raise UnsupportedVideo("No supported colored rigid objects found; use a plain background and distinct saturated colors")
    by_frame = {}
    for row in rows:
        by_frame.setdefault(row["frame"], []).append(row)
    coverage, generation, development = plan_reconstruction(
        rows, [profile['id'] for profile in profiles], [0, count - 1], fps)
    write_json(out / 'reconstruction-coverage.json', coverage)
    write_json(out / 'coverage-summary.json', coverage_summary(coverage))
    write_json(out / 'tracked-objects.json', {'objects': profiles, 'observations': rows, 'scope': SCOPE})
    if coverage['reasons']:
        raise UnsupportedVideo('; '.join(coverage['reasons']))
    selected_frames = list(range(coverage['model_interval'][0], count))
    origin = selected_frames[0] / fps
    objects, movement = [], {}
    for profile in profiles:
        observations = [row for row in generation if row["object_id"] == profile["id"]]
        positions = np.array([[row["x"], row["y"]] for row in observations])
        displacement = float(np.max(np.linalg.norm(positions - positions[0], axis=1)))
        movement[profile["id"]] = displacement
        role = profile.get("role", "static" if displacement < 0.008 else "passive")
        if role == "static" and displacement > 0.025:
            raise ValueError(f"{profile['id']} is configured static but moves in the generation evidence")
        first = observations[:min(5, len(observations))]
        times = np.array([row["time_s"] - origin for row in first])
        first_positions = np.array([[row["x"], row["y"]] for row in first])
        # A straight-line slope estimates midpoint velocity, not initial velocity under gravity.
        gravity = np.array(config.get("gravity", [0, 0]))
        ballistic_positions = first_positions - 0.5 * times[:, None] ** 2 * gravity
        velocity = np.linalg.lstsq(np.column_stack((times, np.ones(len(times)))),
                                   ballistic_positions, rcond=None)[0][0]
        if np.linalg.norm(velocity) > 3:
            raise ValueError("Initial motion exceeds the supported three image-widths per second")
        obj = {"id": profile["id"], "label": profile.get("label", profile["id"]), "role": role,
               "shape": profile.get("shape", observations[0]["shape"]), "position": positions[0].tolist(),
               "velocity": [0.0, 0.0] if role == "static" else velocity.tolist(),
               "color": profile["color"], "mass": profile.get("mass", 1.0),
               "provenance": {"position": "observed image projection", "geometry": "inferred 2D collision approximation",
                              "velocity": "inferred from first five generation observations with configured gravity removed", "mass": "authored assumption",
                              "role": "authored in config" if "role" in profile else "inferred from generation motion",
                              "evidence_ids": [row["id"] for row in first]}}
        if obj["shape"] == "circle":
            obj["radius"] = float(np.median([row["radius_px"] for row in observations])) / width
        else:
            obj["size"] = [float(np.median([row[key] for row in observations])) / width
                           for key in ("width_px", "height_px")]
        objects.append(obj)
    active = next((obj["id"] for obj in objects if obj["role"] == "active"), None)
    controlled = config.get("controlled_id", active or max(movement, key=movement.get))
    actor = next(obj for obj in objects if obj["id"] == controlled)
    if active and active != controlled:
        raise ValueError("controlled_id conflicts with the object marked active")
    if actor["role"] == "static" and any(obj.get("role") == "static" and obj["id"] == controlled for obj in config.get("objects", [])):
        raise ValueError("The controlled object cannot be configured static")
    actor["role"] = "active"
    actor["provenance"]["control"] = "authored acceleration/braking actuator; not recovered from video"
    target = config.get("target", [0.75 if actor["position"][0] < 0.5 else 0.25, actor["position"][1]])
    assumptions = [
        "Fixed camera and a common 2D plane are assumed, not recovered or independently verified.",
        "Coordinates use image widths; x points right, y points down. No metric scale, depth, or 3D geometry is recovered.",
        "Object masses, boundary walls, gravity, restitution, and force controls are authored hypotheses.",
        "The parking objective and target are authored, not inferred task intent.",
        "Distinct saturated colors identify objects; arbitrary semantics, similar-colored objects, and long occlusion are unsupported.",
        "Dynamic circles and axis-aligned box approximations are used; hidden geometry and angular state are unknown.",
        "All source-video inspection is development evidence, not a final holdout or physical intervention trial.",
    ]
    if automatic:
        assumptions.append("Automatic color discovery starts at the first frame containing eligible shapes; objects entering later are not reconstructed.")
    scene = {"schema_version": 1, "profile": "planar-rigid-v1", "width": 1.0, "height": height / width,
             "dt": 1 / 30, "gravity": config.get("gravity", [0, 0]), "damping": config.get("damping", 0.9),
             "restitution": config.get("restitution", 0.65), "objects": objects, "controlled_id": controlled,
             "target": target, "task": {"position_tolerance": 0.06, "speed_tolerance": 0.15,
                                        "max_steps": 300, "acceleration": 2.0},
             "relationships": [], "assumptions": assumptions}
    contact_candidates = []
    # Color geometry cannot prove contact; projected proximity remains a hypothesis.
    for i, left in enumerate(objects):
        for right in objects[i + 1:]:
            left_radius = left.get("radius", math.hypot(*left.get("size", [0, 0])) / 2)
            right_radius = right.get("radius", math.hypot(*right.get("size", [0, 0])) / 2)
            for frame in selected_frames:
                if frame > coverage['generation_end_frame']:
                    break
                group = {row["object_id"]: row for row in by_frame.get(frame, [])}
                if left['id'] not in group or right['id'] not in group:
                    continue
                a, b = group[left["id"]], group[right["id"]]
                if math.hypot(a["x"] - b["x"], a["y"] - b["y"]) <= left_radius + right_radius + 0.015:
                    scene["relationships"].append({"type": "projected_contact_candidate", "objects": [left["id"], right["id"]],
                                                    "status": "inferred"})
                    contact_candidates.append({"objects": [left["id"], right["id"]], "time_s": frame / fps,
                                               "status": "inferred_not_verified", "evidence_ids": [a["id"], b["id"]]})
                    break
    from weft.scene import validate_scene
    scene = validate_scene(scene)
    evidence = {"schema_version": 1, "profile": "planar-rigid-v1", "source_sha256": source_hash,
                "source_label": config.get("source_label", "User-provided local video; physical provenance not independently verified"),
                "width": width, "height": height, "source_width": raw_width, "source_height": raw_height,
                "fps": fps, "frame_count": count, "decoded_frames": decoded, "decode_audit": decode_audit, "origin_time_s": origin,
                "preview_stride": stride,
                "initialization_frame": selected_frames[0], "selected_frames": selected_frames,
                "reconstruction_coverage": coverage,
                "generation": generation, "development_validation": development,
                "objects": [{"id": obj["id"], "label": obj["label"], "color": obj["color"], "role": obj["role"]} for obj in objects],
                "tracking_profiles": profiles, "visibility": visibility, "config": config,
                "contact_candidates": contact_candidates,
                "camera_check": {"scope": "Background motion screen; absence of detected motion does not prove a fixed camera",
                                 "samples": camera_samples,
                                 "result": "no_large_motion_detected" if any(row["status"] == "no_large_motion_detected" for row in camera_samples)
                                           else "unknown_insufficient_background_evidence"},
                "tracker": "OpenCV HSV contours; one distinct color per object; no model inference",
                "scope": SCOPE, "limits": assumptions,
                "excluded_frames": count - len(selected_frames),
                "excluded_frames_scope": "Before common model initialization; missing observations inside the model interval are retained",
                "observation_count": len(rows)}
    write_json(out / "tracked-objects.json", {"objects": profiles, "observations": rows, "scope": SCOPE})
    write_json(out / "evidence.json", evidence)
    write_json(out / "scene-initial.json", scene)
    write_json(out / "scene.json", scene)
    _notify(progress, "tracking", decoded_frames=decoded, objects=len(objects), observations=len(rows),
            message="Measurements and an initial planar world are ready")
    return evidence, scene


def fit_scene(evidence, scene, out, progress=None):
    """Keep bounded dynamics proposals separate from the fixed reconstruction scorer."""
    from weft.initial_motion import propose_initial_motion
    from weft.scene import validate_scene
    out = Path(out)
    initial = validate_scene(scene)
    baseline = reconstruction_metrics(evidence, initial)
    selected, score = initial, baseline
    ledger = []
    for damping in (1.0, 0.6, 0.3):
        if abs(damping - initial["damping"]) < 1e-9:
            continue
        candidate = deepcopy(initial)
        candidate["damping"] = damping
        candidate = validate_scene(candidate)
        measured = reconstruction_metrics(evidence, candidate)
        relative = (score["rmse_image_width"] - measured["rmse_image_width"]) / max(score["rmse_image_width"], 1e-12)
        regressions = [object_id for object_id, metric in measured["per_object"].items()
                       if metric["rmse_image_width"] > score["per_object"][object_id]["rmse_image_width"] * 1.02 + 0.001]
        accepted = measured["finite"] and relative >= 0.02 and not regressions
        record = {"proposal": {"damping": damping}, "base_scene_hash": canonical_hash(initial),
                  "candidate_scene_hash": canonical_hash(candidate), "accepted": accepted,
                  "before": score, "after": measured, "relative_improvement": relative,
                  "regressed_objects": regressions,
                  "reason": ("Fixed development error improved by at least 2% without object-level regression" if accepted
                             else "Object-level regression" if regressions else "Insufficient development improvement"),
                  "scope": SCOPE, "provider": "local_bounded_numeric_search", "cost_usd": 0.0}
        ledger.append(record)
        if accepted:
            selected, score = candidate, measured
        _notify(progress, "repair", attempt=len(ledger), accepted=accepted, rmse_px=measured["rmse_px"])
    candidate, motion = propose_initial_motion(selected, evidence['generation'],
                                               width=evidence['width'], origin_time_s=evidence['origin_time_s'])
    motion['base_scene_hash'] = canonical_hash(selected)
    if candidate is not None:
        candidate = validate_scene(candidate)
        measured = reconstruction_metrics(evidence, candidate)
        relative = (score['rmse_image_width'] - measured['rmse_image_width']) / max(score['rmse_image_width'], 1e-12)
        regressions = [object_id for object_id, metric in measured['per_object'].items()
                       if metric['rmse_image_width'] > score['per_object'][object_id]['rmse_image_width'] * 1.02 + .001]
        motion['accepted'] = measured['finite'] and relative >= .02 and not regressions
        motion['selection_reason'] = ('Fixed development error improved by at least 2% without object-level regression' if motion['accepted']
                                      else 'Object-level regression' if regressions else 'Insufficient development improvement')
        ledger.append({'proposal': {'initial_velocity': motion['changes']}, 'base_scene_hash': motion['base_scene_hash'],
                       'candidate_scene_hash': canonical_hash(candidate), 'accepted': motion['accepted'],
                       'before': score, 'after': measured, 'relative_improvement': relative, 'regressed_objects': regressions,
                       'reason': motion['selection_reason'], 'scope': SCOPE, 'provider': motion['contract']['method'], 'cost_usd': 0.})
        if motion['accepted']:
            selected, score = candidate, measured
        _notify(progress, 'repair', attempt=len(ledger), accepted=motion['accepted'], rmse_px=measured['rmse_px'])
    write_json(out / 'initial-motion-proposal.json', motion)
    per_object_passes = {key: metric["rmse_image_width"] <= 0.08 for key, metric in score["per_object"].items()}
    summary = {"scope": SCOPE, "before": baseline, "after": score,
               "accepted_repairs": sum(row["accepted"] for row in ledger),
               "physical_validity": "Engine execution and finite state only; no real-world action validation",
               "replay_gate": {"tolerance_image_width": 0.08, "passes": all(per_object_passes.values()),
                               "per_object_passes": per_object_passes},
               "frames": _reconstruction_frames(evidence, selected),
               "limits": ["Damping/initial-velocity fitting cannot recover unobserved forces, camera geometry, mass, or task intent.",
                          "A runnable environment may fail reconstruction; the failure remains visible."]}
    if 'reconstruction_coverage' in evidence:
        summary['reconstruction_coverage'] = coverage_summary(evidence['reconstruction_coverage'])
    write_json(out / "scene.json", selected)
    write_json(out / "repair-ledger.json", ledger)
    write_json(out / "reconstruction.json", summary)
    _save_reconstruction_preview(evidence, selected, summary["frames"], out)
    return selected


def _reconstruction_frames(evidence, scene):
    from weft.scene import SceneEnv
    rows = evidence["generation"] + evidence["development_validation"]
    if not rows:
        raise ValueError("Reconstruction requires development observations")
    by_time = {}
    for row in rows:
        by_time.setdefault(row["time_s"], []).append(row)
    development_ids = {row['id'] for row in evidence['development_validation']}
    timeline = evidence.get('reconstruction_coverage', {}).get('frames')
    if timeline is not None:
        first, last = evidence['reconstruction_coverage']['requested_interval']
        if (not 1 <= len(timeline) <= 1800 or len(timeline) != last - first + 1
                or (last - first + 1) / evidence['fps'] > 60
                or any(row['frame'] != first + i or abs(row['time_s'] - row['frame'] / evidence['fps']) > 1e-6
                       for i, row in enumerate(timeline))):
            raise ValueError('Reconstruction coverage must retain the complete bounded frame interval')
        if set(by_time) - {row['time_s'] for row in timeline}:
            raise ValueError('Reconstruction timeline excludes observations')
        origin = evidence['initialization_frame']
        cutoff = origin + max(6, int((last - origin + 1) * .6)) - 1
        if not first <= origin <= last or abs(evidence['origin_time_s'] - origin / evidence['fps']) > 1e-6:
            raise ValueError('Reconstruction initialization disagrees with source time')
        for entry in timeline:
            expected = ('before_model_initialization' if entry['frame'] < origin else
                        'generation' if entry['frame'] <= cutoff else 'development_validation')
            group = by_time.get(entry['time_s'], [])
            if (entry['partition'] != expected
                    or any((row['id'] in development_ids) != (expected == 'development_validation') for row in group)):
                raise ValueError('Reconstruction coverage cannot change the chronological scoring partition')
    else:
        # Old artifacts keep their original observation-only replay semantics.
        timeline = [{'frame': group[0]['frame'], 'time_s': source_time,
                     'partition': 'development_validation' if group[0]['id'] in development_ids else 'generation'}
                    for source_time, group in sorted(by_time.items())]
    env = SceneEnv(scene)
    env.reset(seed=0, options={"jitter": False})
    elapsed, result = 0.0, []
    before = after = {obj["id"]: obj for obj in env.snapshot()["objects"]}
    try:
        for entry in timeline:
            source_time = entry['time_s']
            group = by_time.get(source_time, [])
            if entry['partition'] == 'before_model_initialization':
                result.append({'frame': entry['frame'], 'time_s': source_time, 'simulation_time_s': None,
                               'partition': entry['partition'], 'observed': [], 'predicted': [],
                               'measurement_status': 'outside_model_interval'})
                continue
            until = source_time - evidence["origin_time_s"]
            if not 0 <= until <= 60 or math.ceil(until / scene['dt'] - 1e-9) > 1800:
                raise ValueError('Reconstruction exceeds the bounded simulation duration or step count')
            while elapsed < until - 1e-10:
                before = after
                state = env.advance_physics(0)
                after = {obj["id"]: obj for obj in state["objects"]}
                elapsed += scene["dt"]
            fraction = min(1.0, max(0.0, (until - elapsed + scene["dt"]) / scene["dt"]))
            # Render between fixed engine ticks; do not alter the fitted world's timestep.
            positions = {key: [a + fraction * (b - a) for a, b in zip(before[key]["position"], after[key]["position"])]
                         for key in after}
            result.append({"frame": entry['frame'], "time_s": source_time, "simulation_time_s": until,
                           "partition": entry['partition'],
                           "measurement_status": ('complete' if len(group) == len(scene['objects']) else
                                                  'partial' if group else 'unavailable'),
                           "observed": [{"id": row["object_id"], "position": [row["x"], row["y"]],
                                         "evidence_id": row["id"]} for row in group],
                           "predicted": [{"id": key, "position": point,
                                          "angle": before[key]["angle"] + fraction * (after[key]["angle"] - before[key]["angle"])}
                                         for key, point in positions.items()]})
    finally:
        env.close()
    return result


def reconstruction_metrics(evidence, scene):
    errors, per_object = [], {}
    for frame in _reconstruction_frames(evidence, scene):
        if frame["partition"] != "development_validation":
            continue
        positions = {obj["id"]: obj["position"] for obj in frame["predicted"]}
        for row in frame["observed"]:
            predicted = positions[row["id"]]
            error = math.dist(predicted, row["position"])
            if not math.isfinite(error):
                raise ValueError("Non-finite reconstruction state")
            errors.append(error)
            per_object.setdefault(row["id"], []).append(error)
    rmse = math.sqrt(sum(error * error for error in errors) / len(errors))
    return {"rmse_image_width": rmse, "rmse_px": rmse * evidence["width"], "observations": len(errors),
            "max_error_image_width": max(errors), "finite": True, "scope": SCOPE,
            "per_object": {key: {"observations": len(values),
                                 "rmse_image_width": math.sqrt(sum(v * v for v in values) / len(values))}
                           for key, values in per_object.items()}}


def _save_reconstruction_preview(evidence, scene, frames, out):
    preview_path = out / "tracks.gif"
    if not preview_path.is_file():
        return
    by_frame = {frame["frame"]: frame for frame in frames}
    objects = {obj["id"]: obj for obj in scene["objects"]}
    comparisons, source_times = [], []
    with Image.open(preview_path) as preview:
        for index, image in enumerate(ImageSequence.Iterator(preview)):
            source_frame = (evidence["preview_frame_indices"][index] if "preview_frame_indices" in evidence
                            else index * evidence["preview_stride"])
            if source_frame not in by_frame:
                continue
            frame = by_frame[source_frame]
            image = image.convert("RGB")
            width, height = image.size
            comparison = Image.new("RGB", (width * 2, height + 66), "#F5EADC")
            comparison.paste(image, (0, 66))
            draw = ImageDraw.Draw(comparison)
            draw.text((8, 8), "SOURCE VIDEO + TRACKS", fill="#214F50")
            draw.text((width + 8, 8), "SIMULATION | NO CONTROL", fill="#214F50")
            draw.text((8, 26), f"Source frame {source_frame} | {frame['time_s']:.3f}s", fill="#214F50")
            partition = {'generation': 'Generation evidence', 'development_validation': 'Development check',
                         'before_model_initialization': 'Before model initialization'}[frame['partition']]
            draw.text((8, 44), partition, fill="#596562")
            label = ('No simulation before initialization' if not frame['predicted'] else
                     'No measurements; unscored prediction' if not frame['observed'] else
                     f"Measured {len(frame['observed'])}/{len(objects)} | crosses: position error")
            draw.text((width + 8, 26), label, fill="#214F50")
            draw.text((width + 8, 44), "2D hypothesis, not physical proof", fill="#596562")
            draw.rectangle((width + 1, 67, width * 2 - 2, height + 64), outline="#747474", width=2)
            observed = {obj["id"]: obj["position"] for obj in frame["observed"]}
            for prediction in frame["predicted"]:
                obj = objects[prediction["id"]]
                x, y = prediction["position"]
                x, y = width + x * width, 66 + y * width
                if obj["shape"] == "circle":
                    radius = obj["radius"] * width
                    draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=obj["color"], outline="#214F50")
                else:
                    hw, hh = [side * width / 2 for side in obj["size"]]
                    c, s = math.cos(prediction["angle"]), math.sin(prediction["angle"])
                    points = [(x + px * c - py * s, y + px * s + py * c) for px, py in
                              [(-hw, -hh), (hw, -hh), (hw, hh), (-hw, hh)]]
                    draw.polygon(points, fill=obj["color"], outline="#214F50")
                if prediction["id"] in observed:
                    ox, oy = observed[prediction["id"]]
                    ox, oy = width + ox * width, 66 + oy * width
                    draw.line((x, y, ox, oy), fill="#B3261E", width=2)
                    draw.line((ox - 5, oy, ox + 5, oy), fill="#B3261E", width=2)
                    draw.line((ox, oy - 5, ox, oy + 5), fill="#B3261E", width=2)
            comparisons.append(comparison)
            source_times.append(frame["time_s"])
    if not comparisons:
        return
    duration = [max(40, round((right - left) * 1000)) for left, right in zip(source_times, source_times[1:])]
    duration.append(duration[-1] if duration else 100)
    comparisons[0].save(out / "reconstruction.gif", save_all=True, append_images=comparisons[1:], duration=duration, loop=0)
    chosen = sorted({0, len(comparisons) // 2, len(comparisons) - 1})
    sample = []
    for index in chosen:
        thumbnail = comparisons[index].copy()
        thumbnail.thumbnail((1280, 640))
        sample.append(thumbnail)
    sheet = Image.new("RGB", (max(image.width for image in sample), sum(image.height for image in sample)), "#F5EADC")
    y = 0
    for image in sample:
        sheet.paste(image, (0, y))
        y += image.height
    sheet.save(out / "reconstruction-contact-sheet.png")
