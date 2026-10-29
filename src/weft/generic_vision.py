"""ROI-initialized appearance tracking; image motion is not recovered physical dynamics."""

from copy import deepcopy
from contextlib import ExitStack
import math
from pathlib import Path
import re
import sys
import time

import cv2
import numpy as np
from PIL import Image

from weft.appearance_consistency import COLOR_CONTRACT, color_signature, compare_color
from weft.io import canonical_hash, digest, write_json
from weft.joint_tracking import JOINT_CONTRACT, JointAppearance
from weft.reconstruction_coverage import coverage_summary, plan_reconstruction
from weft.scene_vision import _camera_sample
from weft.video_io import webm_video_frame_count


LIMITS = {"bytes": 100_000_000, "frames": 1800, "seconds": 60, "objects": 8, "analysis_dimension": 960}


def validate_config(config):
    if not isinstance(config, dict) or set(config) - {"schema_version", "source_sha256", "start_frame", "end_frame", "initialization_frame", "decode_interval", "association_mode", "scale_mode", "objects", "search_radius_px", "minimum_score", "minimum_margin", "max_gap_frames"}:
        raise ValueError("Unknown generic tracking settings")
    result = deepcopy(config)
    result.setdefault("association_mode", "independent")
    if result["association_mode"] not in ("independent", "joint"):
        raise ValueError("Association mode must be independent or joint")
    if result.get('scale_mode', 'fixed') not in ('fixed', 'pyramid'):
        raise ValueError('Scale mode must be fixed or pyramid')
    if result.get('scale_mode') == 'pyramid' and result['association_mode'] != 'independent':
        raise ValueError('Scale hypotheses currently require independent association')
    if result.get("schema_version") != 1 or type(result.get("schema_version")) is not int:
        raise ValueError("Tracking config requires schema_version=1")
    if not isinstance(result.get("source_sha256"), str) or not re.fullmatch(r"[a-f0-9]{64}", result["source_sha256"]):
        raise ValueError("Tracking initialization must pin its source video")
    window = result.get("decode_interval")
    if "decode_interval" in result and (not isinstance(window, list) or len(window) != 2
            or any(type(value) is not int for value in window) or not 0 <= window[0] <= 20000
            or not window[0] <= window[1] <= 21799 or window[1] - window[0] >= 1800):
        raise ValueError("Decode interval requires inclusive source indices, at most 1800 frames and a bounded prefix")
    for key, default, lo, hi in (("start_frame", window[0] if window else 0, 0, 20000 if window else 1799),
                                ("end_frame", window[1] if window else 1799, 0, 21799 if window else 1799),
                                ("search_radius_px", 60, 8, 200), ("max_gap_frames", 8, 0, 30)):
        result.setdefault(key, default)
        if type(result[key]) is not int or not lo <= result[key] <= hi:
            raise ValueError(f"Invalid {key}")
    if result["end_frame"] < result["start_frame"]:
        raise ValueError("Tracking interval is reversed")
    if window and window != [result["start_frame"], result["end_frame"]]:
        raise ValueError("Decode interval must match the complete authored tracking interval")
    result.setdefault("initialization_frame", result["start_frame"])
    if (type(result["initialization_frame"]) is not int
            or not result["start_frame"] <= result["initialization_frame"] <= result["end_frame"]):
        raise ValueError("Initialization frame must lie inside the selected interval")
    for key, default, lo, hi in (("minimum_score", .7, .5, .99), ("minimum_margin", .06, .02, .5)):
        result.setdefault(key, default)
        if type(result[key]) not in (float, int) or not math.isfinite(result[key]) or not lo <= result[key] <= hi:
            raise ValueError(f"Invalid {key}")
    objects = result.get("objects")
    if not isinstance(objects, list) or not 1 <= len(objects) <= LIMITS["objects"]:
        raise ValueError("Initialize 1–8 object regions")
    ids = set()
    for obj in objects:
        if not isinstance(obj, dict) or set(obj) != {"id", "label", "box_xywh", "role", "shape", "provenance"}:
            raise ValueError("Object initialization needs id, label, box_xywh, role, shape and provenance")
        if not isinstance(obj["id"], str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,48}", obj["id"]) or obj["id"] in ids:
            raise ValueError("Object identities must be unique bounded identifiers")
        ids.add(obj["id"])
        if obj["role"] not in {"active", "passive", "static", "unknown"} or obj["shape"] not in {"circle", "box", "unknown"}:
            raise ValueError("Object roles and shape assumptions must be explicit")
        for name in ("label", "provenance"):
            if not isinstance(obj[name], str) or not 1 <= len(obj[name]) <= 500:
                raise ValueError("Initialization labels/provenance must be bounded text")
        box = obj["box_xywh"]
        if not isinstance(box, list) or len(box) != 4 or any(type(v) is not int for v in box) or min(box[:2]) < 0 or min(box[2:]) < 8 or max(box) > 4096:
            raise ValueError("Use integer source-pixel boxes at least 8×8")
    return result


def _overlap(a, b):
    x = max(0, min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0]))
    y = max(0, min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1]))
    return x * y / min(a[2] * a[3], b[2] * b[3])


class AppearanceTrack:
    def __init__(self, gray, box, settings, *, color=None):
        self.box = tuple(box)
        x, y, w, h = self.box
        if min(w, h) < 6 or x < 0 or y < 0 or x + w > gray.shape[1] or y + h > gray.shape[0]:
            raise ValueError("Initialized region exceeds the image or analysis resolution")
        self.template = gray[y:y+h, x:x+w].copy()
        if self.template.std() < 8:
            raise ValueError("Initialized object lacks enough contrast/texture for appearance tracking")
        self.settings = settings
        self.source_color = color_signature(color[y:y+h, x:x+w]) if color is not None else None
        self.gap = 0
        self.velocity = np.zeros(2)
        self.lost = False

    def locate(self, gray, *, color=None):
        if self.lost:
            return {"status": "lost", "score": None, "margin": None, "box": None}
        x, y, w, h = self.box
        radius = self.settings["search_radius_px"]
        predicted = np.array([x, y]) + self.velocity * min(self.gap + 1, 3)
        left, top = max(0, math.floor(predicted[0] - radius)), max(0, math.floor(predicted[1] - radius))
        right, bottom = min(gray.shape[1], math.ceil(predicted[0] + w + radius)), min(gray.shape[0], math.ceil(predicted[1] + h + radius))
        if right - left < w or bottom - top < h:
            return {"status": "out_of_frame", "score": None, "margin": None, "box": None}
        scores = cv2.matchTemplate(gray[top:bottom, left:right], self.template, cv2.TM_CCOEFF_NORMED)
        _, best, _, location = cv2.minMaxLoc(scores)
        secondary = scores.copy()
        # Nearby maxima represent the same object; distant similar peaks are identity ambiguity.
        sx, sy = location
        dx, dy = max(3, w // 3), max(3, h // 3)
        secondary[max(0, sy-dy):sy+dy+1, max(0, sx-dx):sx+dx+1] = -1
        second = float(secondary.max()) if secondary.size else -1
        margin = float(best - second)
        status = "tracked" if best >= self.settings["minimum_score"] and margin >= self.settings["minimum_margin"] else (
            "ambiguous" if best >= self.settings["minimum_score"] else "not_detected")
        appearance = {"status": "not_checked_no_candidate", "passes": None}
        if status == "tracked":
            appearance = {"status": "unavailable_color_evidence", "passes": None}
            if self.source_color is not None and color is not None:
                region = color[top+sy:top+sy+h, left+sx:left+sx+w]
                appearance = compare_color(self.source_color, color_signature(region))
                if appearance["passes"] is False:
                    status = "appearance_changed"
        return {"status": status, "score": float(best), "margin": margin,
                "appearance_evidence": appearance,
                "box": [left + sx, top + sy, w, h] if status == "tracked" else None}

    def commit(self, candidate):
        if candidate["status"] == "tracked":
            new = candidate["box"]
            displacement = (np.array(new[:2]) - self.box[:2]) / (self.gap + 1)
            self.velocity = .5 * self.velocity + .5 * displacement
            self.box = tuple(new)
            self.gap = 0
        else:
            self.gap += 1
            if self.gap > self.settings["max_gap_frames"]:
                self.lost = True


def _preview(images, out):
    if not images:
        return
    images[0].save(out / "tracks.gif", save_all=True, append_images=images[1:], duration=120, loop=0)
    thumbnail_width = 320
    height = max(1, round(images[0].height * thumbnail_width / images[0].width))
    selected = images[::max(1, math.ceil(len(images) / 12))][:12]
    sheet = Image.new("RGB", (3 * thumbnail_width, math.ceil(len(selected) / 3) * height), "white")
    for i, im in enumerate(selected):
        sheet.paste(im.resize((thumbnail_width, height)), ((i % 3) * thumbnail_width, (i // 3) * height))
    sheet.save(out / "tracks-contact-sheet.png")


def track_video(video, out, config, progress=None):
    """Track declared image regions; never silently infer object identity or a physical model."""
    video, out = Path(video), Path(out)
    config = validate_config(config)
    if not video.is_file() or video.stat().st_size > LIMITS["bytes"]:
        raise ValueError("Provide a local video no larger than 100 MB")
    source_hash = digest(video)
    if source_hash != config["source_sha256"]:
        raise ValueError("Source hash differs from ROI initialization")
    out.mkdir(parents=True, exist_ok=False)
    write_json(out / "tracking-config.json", config)
    resources = ExitStack()
    started = time.monotonic()
    frames, observations, previews, camera_samples = [], [], [], []
    trackers, joint_tracker = [], None
    initialized = False
    source_color_signatures = {}
    try:
        interval_decoder = None
        if config.get("decode_interval"):
            from weft.video_interval import IntervalVideo
            first, last = config["decode_interval"]
            interval_decoder = IntervalVideo(video, first, last, source_hash)
            resources.enter_context(interval_decoder)
            metadata = interval_decoder.metadata
            fps, count, source_w, source_h = [metadata[key] for key in ("fps", "frame_count", "width", "height")]
            source_frames = iter(interval_decoder)
        else:
            cap = cv2.VideoCapture(str(video))
            resources.callback(cap.release)
            if not cap.isOpened():
                raise ValueError("Source video could not be decoded")
            fps, count, source_w, source_h = [cap.get(key) for key in (cv2.CAP_PROP_FPS, cv2.CAP_PROP_FRAME_COUNT, cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT)]
            if not all(math.isfinite(v) for v in (fps, count, source_w, source_h)) or not 1 <= fps <= 240 or not 1 <= count <= 1800 or count / fps > 60 or not 64 <= source_w <= 4096 or not 64 <= source_h <= 2160:
                raise ValueError("Video exceeds frame, duration, dimension or FPS bounds; long sources require an explicit decode_interval")
            def whole_source():
                index = 0
                while True:
                    ok, frame = cap.read()
                    if not ok:
                        return
                    yield index, frame
                    index += 1
            source_frames = whole_source()
        count, source_w, source_h = int(count), int(source_w), int(source_h)
        scale = min(1, LIMITS["analysis_dimension"] / max(source_w, source_h))
        width, height = round(source_w * scale), round(source_h * scale)
        settings = dict(config, search_radius_px=max(8, round(config["search_radius_px"] * scale)))
        boxes = [[round(v * scale) for v in obj["box_xywh"]] for obj in config["objects"]]
        if any(_overlap(a, b) > .5 for i, a in enumerate(boxes) for b in boxes[i+1:]):
            raise ValueError("Initialized objects overlap ambiguously")
        interval = min(config["end_frame"], count - 1) - config["start_frame"] + 1
        if interval < 1:
            raise ValueError("Tracking initialization is outside the video")
        preview_stride = max(1, math.ceil(interval / 40))
        decoded, camera_reference, camera_points = 0, None, None
        for index, source_frame in source_frames:
            decoded += 1
            if decoded > 1800 or decoded / fps > 60 or time.monotonic() - started > 120:
                raise ValueError("Video processing exceeded its bounded envelope")
            if source_frame.shape[:2] != (source_h, source_w):
                raise ValueError("Source dimensions changed within the video")
            if not config["start_frame"] <= index <= config["end_frame"]:
                frames.append({"frame": index, "time_s": index / fps, "status": "outside_authored_interval"})
                continue
            frame = cv2.resize(source_frame, (width, height), interpolation=cv2.INTER_AREA) if scale != 1 else source_frame
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if index < config["initialization_frame"]:
                candidates = [{"status": "pre_initialization", "score": None, "margin": None, "box": None} for _ in boxes]
            elif not initialized:
                if config["association_mode"] == "joint":
                    joint_tracker = JointAppearance(gray, boxes, settings, color=frame,
                                                    shapes=[obj["shape"] for obj in config["objects"]])
                else:
                    tracker_class = AppearanceTrack
                    if config.get('scale_mode') == 'pyramid':
                        from weft.scale_tracking import ScaleAppearanceTrack
                        tracker_class = ScaleAppearanceTrack
                    trackers = [tracker_class(gray, box, settings, color=frame) for box in boxes]
                source_color_signatures = {obj["id"]: color_signature(frame[y:y+h, x:x+w])
                                           for obj, (x, y, w, h) in zip(config["objects"], boxes)}
                initialized = True
                camera_reference = gray
                mask = np.full(gray.shape, 255, np.uint8)
                for x, y, w, h in boxes:
                    mask[max(0,y-15):y+h+15,max(0,x-15):x+w+15] = 0
                camera_points = cv2.goodFeaturesToTrack(gray, maxCorners=120, qualityLevel=.01, minDistance=12, mask=mask)
                candidates = [{"status": "initialized", "score": None, "margin": None, "box": box} for box in boxes]
            elif joint_tracker is not None:
                candidates = joint_tracker.locate(gray, color=frame)
                joint_tracker.commit(candidates)
            else:
                candidates = [tracker.locate(gray, color=frame) for tracker in trackers]
                conflicts = {i for i, candidate in enumerate(candidates)
                             if candidate["box"] is not None and any(other["box"] is not None and _overlap(candidate["box"], other["box"]) > .5
                             for j, other in enumerate(candidates) if j != i)}
                for i in conflicts:
                    candidates[i].update(status="ambiguous_overlap", box=None)
                for tracker, candidate in zip(trackers, candidates):
                    tracker.commit(candidate)
            if initialized and (index - config["initialization_frame"]) % max(1, round(fps / 2)) == 0:
                camera_samples.append(_camera_sample(camera_reference, frame, camera_points, index))
            frame_result = {"frame": index, "time_s": index / fps, "status": "processed", "objects": []}
            overlay = frame.copy()
            for obj, candidate in zip(config["objects"], candidates):
                record = {"id": f"{source_hash}:frame:{index}:object:{obj['id']}", "frame": index,
                          "time_s": index / fps, "object_id": obj["id"], "status": candidate["status"],
                          "match_score": candidate["score"], "peak_margin": candidate["margin"],
                          "appearance_evidence": candidate.get("appearance_evidence", {"status": "not_checked_no_candidate", "passes": None}),
                          "box_xywh": candidate["box"], "coordinate_system": "analysis_image_pixels",
                          "geometry_status": "authored_initial_ROI_translated_by_appearance_match"}
                if config.get('scale_mode') == 'pyramid':
                    record['geometry_status'] = 'source_ROI_scaled_translated_by_appearance_match_not_physical_shape'
                    record['scale_evidence'] = candidate.get('scale_evidence')
                if "association_evidence" in candidate:
                    record["association_evidence"] = candidate["association_evidence"]
                    # Joint assignment confidence is not the raw visual peak separation.
                    record["peak_margin"] = candidate["association_evidence"]["peak_margin"]
                if candidate["box"] is not None:
                    x, y, w, h = candidate["box"]
                    record.update(x_px=x+w/2, y_px=y+h/2, x=(x+w/2)/width, y=(y+h/2)/width,
                                  width_px=w, height_px=h, radius_px=(w+h)/4, shape=obj["shape"])
                    cv2.rectangle(overlay, (x,y), (x+w,y+h), (0,180,255), 2)
                    cv2.putText(overlay, obj["label"], (x,max(15,y-5)), cv2.FONT_HERSHEY_SIMPLEX,.4,(0,0,255),1)
                frame_result["objects"].append(record)
                observations.append(record)
            frames.append(frame_result)
            if (index - config["start_frame"]) % preview_stride == 0:
                cv2.putText(overlay, f"frame {index} | ROI appearance tracking, not physics",(10,height-12),cv2.FONT_HERSHEY_SIMPLEX,.42,(0,0,255),1)
                preview = Image.fromarray(cv2.cvtColor(overlay,cv2.COLOR_BGR2RGB))
                preview.thumbnail((640, 640))
                previews.append(preview)
            if progress and index % 120 == 0:
                progress("tracking", {"frame": index, "objects": len(boxes), "initialized": initialized})
        if not initialized:
            raise ValueError("No initialized tracking frame was decoded")
        resources.close()
        audit = interval_decoder.audit if interval_decoder else {"metadata_frames": count, "decoded_frames": decoded}
        if interval_decoder is not None:
            write_json(out / 'decode-audit.json', audit)
            if not audit["selected_frames_complete"]:
                raise ValueError("Selected interval was not completely decoded")
            selected_interval = config["decode_interval"][:]
        else:
            selected_interval = [config["start_frame"], min(config["end_frame"], decoded-1)]
        if interval_decoder is None and count != decoded:
            if video.suffix.lower() not in {".webm", ".mkv"}:
                raise ValueError("Decoded frame count disagrees with source metadata")
            audit["encoded_video_frames"] = webm_video_frame_count(video)
            if audit["encoded_video_frames"] != decoded:
                raise ValueError("Not every encoded video frame decoded")
        motion = any(item["status"] == "moving_background" for item in camera_samples)
        result = {"schema_version": 1, "profile": "appearance-tracks-v1", "source_sha256": source_hash,
                  "width": width, "height": height, "source_width": source_w, "source_height": source_h,
                  "fps": fps, "frame_count": count, "decoded_frames": decoded, "decode_audit": audit,
                  "selected_interval": selected_interval,
                  "initialization": {"status": "authored", "frame": config["initialization_frame"], "config_sha256": canonical_hash(config), "objects": config["objects"],
                                     "source_color_signatures": source_color_signatures},
                  "appearance_contract": COLOR_CONTRACT,
                  "objects": config["objects"], "frames": frames, "observations": observations,
                  "camera_check": {"samples": camera_samples, "reference_frame": config["initialization_frame"],
                                   "before_reference": "not_sampled", "result": "moving_background" if motion else "not_verified_fixed"},
                  "physics_ready": False, "seconds": time.monotonic()-started,
                  "limits": ["Manual object regions and labels, not automatic semantic detection", "Fixed-size translated appearance templates, not segmentation or 3D geometry",
                             "Source color can reject incompatible matches, not prove visibility or distinguish same-color objects; achromatic sources have no color evidence",
                             "Missing/ambiguous detections have no invented positions; long gaps require new initialization",
                             "Image motion mixes object and camera movement; no gravity, material or force inference",
                             "All inspected frames are development evidence; independent physical accuracy not established"]}
        result["association_mode"] = config["association_mode"]
        if config.get('scale_mode') == 'pyramid':
            from weft.scale_tracking import SCALE_CONTRACT
            result['scale_contract'] = deepcopy(SCALE_CONTRACT)
            result['limits'][1] = 'Bounded immutable scaled source templates; projected extent is not segmentation, depth, rotation or physical body size'
        if joint_tracker is not None:
            result["association_contract"] = dict(JOINT_CONTRACT)
            result["initialization"]["source_template_signatures"] = dict(zip(
                (obj["id"] for obj in config["objects"]), joint_tracker.source_signatures))
            result["limits"].append("Experimental joint assignment uses bounded local-continuity preferences; matching appearance does not establish physical identity through contact.")
        if interval_decoder is not None:
            result["limits"].append("Only the explicit source interval is decoded for perception; prefix advancement and unseen suffix do not certify full-source completeness. Absolute source frame IDs and media times are retained.")
        write_json(out / "appearance-tracks.json", result)
        _preview(previews, out)
        return result
    except Exception as error:
        if interval_decoder is not None:
            write_json(out / 'decode-audit.json', interval_decoder.audit)
        write_json(out / "failure.json", {"type": type(error).__name__, "message": str(error), "seconds": time.monotonic()-started})
        write_json(out / "partial-observations.json", observations)
        raise
    finally:
        # Pass the original failure through interval cleanup instead of replacing it with an incomplete-read error.
        resources.__exit__(*sys.exc_info())


def scene_measurements(tracks):
    """Return compatible measurements without inventing an executable dynamics model."""
    rows = [{key: row[key] for key in ("id", "frame", "time_s", "object_id", "x", "y", "x_px", "y_px", "width_px", "height_px", "radius_px", "shape")}
            for row in tracks["observations"] if row["status"] == "tracked"]
    return {"source_sha256": tracks["source_sha256"], "width": tracks["width"], "height": tracks["height"],
            "fps": tracks["fps"], "observations": rows, "visibility": [{"frame": row["frame"], "object_id": row["object_id"], "status": row["status"]} for row in tracks["observations"]],
            "provenance": "ROI-initialized appearance matching; collision shape/size remain authored approximation",
            "camera_check": tracks["camera_check"], "physics_ready": False}


def inspect_generic(video, out, config=None, progress=None):
    """Build a bounded planar hypothesis only after explicit ROI and world configuration."""
    import shutil
    from weft.scene import validate_scene
    from weft.scene_vision import UnsupportedVideo

    if not isinstance(config, dict) or set(config) - {'tracking', 'world', 'source_label'} or not {'tracking', 'world'} <= set(config):
        raise ValueError('Appearance profile requires tracking and world configuration')
    tracking = validate_config(config['tracking'])
    world = config['world']
    if not isinstance(world, dict) or set(world) != {'controlled_id', 'target', 'gravity', 'damping', 'restitution'}:
        raise ValueError('Explicit world assumptions require controlled_id, target, gravity, damping and restitution')
    if any(obj['role'] == 'unknown' or obj['shape'] == 'unknown' for obj in tracking['objects']):
        raise ValueError('Simulation requires authored object roles and collision shapes')
    active = [obj['id'] for obj in tracking['objects'] if obj['role'] == 'active']
    if active != [world['controlled_id']]:
        raise ValueError('Exactly one authored active object must match controlled_id')
    out = Path(out)
    if not out.is_dir():
        raise ValueError('Output directory must already exist')
    tracks = track_video(video, out / 'appearance', tracking, progress)
    for name in ('tracks.gif', 'tracks-contact-sheet.png'):
        shutil.copyfile(out / 'appearance' / name, out / name)
    write_json(out / 'camera-motion.json', tracks['camera_check'])
    if tracking.get('scale_mode') == 'pyramid':
        raise UnsupportedVideo('Scale-aware observations require a separate projection/rigid-geometry hypothesis; image extent is not a changing physical body')
    if tracks['camera_check']['result'] == 'moving_background':
        raise UnsupportedVideo('Moving camera or widespread background motion; appearance tracks do not establish fixed-camera physics')
    measurements = scene_measurements(tracks)
    rows = measurements['observations']
    interval = tracks['selected_interval'][1] - tracks['selected_interval'][0]
    coverage, generation, development = plan_reconstruction(
        rows, [obj['id'] for obj in tracking['objects']], tracks['selected_interval'], tracks['fps'],
        initialization_frame=tracking['initialization_frame'], minimum_joint_coverage=.8)
    write_json(out / 'reconstruction-coverage.json', coverage)
    write_json(out / 'coverage-summary.json', coverage_summary(coverage))
    write_json(out / 'appearance-coverage.json', {**coverage_summary(coverage),
                                                'selected_interval': tracks['selected_interval']})
    if coverage['reasons']:
        raise UnsupportedVideo('; '.join(coverage['reasons']))
    selected = list(range(coverage['model_interval'][0], coverage['model_interval'][1] + 1))
    origin, width, height = selected[0] / tracks['fps'], tracks['width'], tracks['height']
    objects = []
    for profile in tracking['objects']:
        observed = [row for row in generation if row['object_id'] == profile['id']]
        first = observed[:5]
        times = np.array([row['time_s'] - origin for row in first])
        positions = np.array([[row['x'], row['y']] for row in first])
        gravity = np.array(world['gravity'], dtype=float)
        if gravity.shape != (2,) or not np.isfinite(gravity).all():
            raise ValueError('World gravity requires two finite coordinates')
        ballistic = positions - .5 * times[:, None] ** 2 * gravity
        velocity = np.linalg.lstsq(np.column_stack((times, np.ones(len(times)))), ballistic, rcond=None)[0][0]
        if profile['role'] == 'static' and max(math.dist([row['x'],row['y']], positions[0]) for row in observed) > .025:
            raise UnsupportedVideo('An authored static object moves in the generation evidence')
        obj = {'id': profile['id'], 'label': profile['label'], 'role': profile['role'], 'shape': profile['shape'],
               'position': positions[0].tolist(), 'velocity': [0.,0.] if profile['role'] == 'static' else velocity.tolist(),
               'color': '#747474', 'mass': 1., 'provenance': {
                   'position': 'inferred center from grayscale appearance match of authored ROI',
                   'geometry': 'authored initial ROI size and collision shape; not segmented object extent',
                   'velocity': 'inferred from first five generation positions with authored gravity removed',
                   'mass': 'authored unit-mass assumption', 'role': 'authored configuration',
                   'color': 'neutral visualization, not recovered material',
                   'control': 'authored acceleration/braking actuator, not recovered from video',
                   'evidence_ids': [row['id'] for row in first]}}
        if profile['shape'] == 'circle':
            obj['radius'] = observed[0]['radius_px'] / width
        else:
            obj['size'] = [observed[0]['width_px'] / width, observed[0]['height_px'] / width]
        objects.append(obj)
    assumptions = tracks['limits'] + [
        'Fixed camera and common 2D plane are hypotheses; a negative motion screen does not verify them.',
        'Image-width units are not meters; depth, angular state and hidden shape remain unknown.',
        'Mass=1, boundary walls, target, force controls and world physics are authored simulation assumptions.',
        'This is an appearance-initialized planar hypothesis, not general video-to-physics recovery.']
    scene = validate_scene({'schema_version':1, 'profile':'planar-rigid-v1', 'width':1., 'height':height/width,
                            'dt':1/30, **world, 'objects':objects, 'task':{'position_tolerance':.06,'speed_tolerance':.15,'max_steps':300,'acceleration':2.},
                            'relationships':[], 'assumptions':assumptions})
    stride = max(1, math.ceil((interval+1)/40))
    evidence = {key:tracks[key] for key in ('source_sha256','width','height','source_width','source_height','fps','frame_count','decoded_frames','decode_audit','camera_check')}
    evidence.update(schema_version=1, profile='planar-rigid-v1', source_label=config.get('source_label','Local video; source physics not independently verified'),
                    origin_time_s=origin, preview_stride=stride,
                    preview_frame_indices=list(range(tracks['selected_interval'][0],tracks['selected_interval'][1]+1,stride)),
                    initialization_frame=selected[0], selected_frames=selected, generation=generation, development_validation=development,
                    tracking_initialization_frame=tracking['initialization_frame'],
                    reconstruction_coverage=coverage,
                    objects=[{key:obj[key] for key in ('id','label','color','role')} for obj in objects],
                    tracking_profiles=tracking['objects'], visibility=measurements['visibility'], config=config, contact_candidates=[],
                    tracker='OpenCV grayscale templates with source-color compatibility checks where available; authored ROI identity; no semantic detector',
                    scope='Development-only image-space reconstruction; no final holdout or independent physical trial', limits=assumptions,
                    excluded_frames=tracks['decoded_frames']-len(selected), excluded_frames_scope='Decoded frames outside the modeled interval, including pre-model frames; missing measurements inside it are retained', observation_count=len(rows))
    write_json(out / 'tracked-objects.json', {'objects':tracking['objects'], 'observations':rows, 'scope':evidence['scope']})
    write_json(out / 'evidence.json', evidence)
    write_json(out / 'scene-initial.json', scene)
    write_json(out / 'scene.json', scene)
    if tracking.get('decode_interval'):
        from weft.video_interval import IntervalVideo
        with IntervalVideo(video, tracking['decode_interval'][0], selected[0], tracks['source_sha256']) as source:
            for index, background in source:
                if index == selected[0] and not cv2.imwrite(str(out/'background.jpg'), cv2.resize(background, (width, height))):
                    raise OSError('Could not write initialized scene background')
    else:
        cap = cv2.VideoCapture(str(video))
        try:
            cap.set(cv2.CAP_PROP_POS_FRAMES, selected[0])
            ok, background = cap.read()
            if not ok or not cv2.imwrite(str(out/'background.jpg'),cv2.resize(background,(width,height))):
                raise OSError('Could not write initialized scene background')
        finally:
            cap.release()
    return evidence, scene


def evaluate_manual(tracks, annotations):
    """Score separately authored labels without exposing them to tracking or scene construction."""
    if annotations.get('source_sha256') != tracks['source_sha256'] or annotations.get('coordinate_system') != 'original_source_pixels':
        raise ValueError('Annotation source hash/coordinates differ from the tracked source')
    predictions = {(row['frame'],row['object_id']):row for row in tracks['observations']}
    scale_x, scale_y = tracks['source_width']/tracks['width'], tracks['source_height']/tracks['height']
    rows, errors = [], []
    visible = accepted = false_visible = 0
    for label in annotations['annotations']:
        prediction = predictions.get((label['frame'],label['object_id']))
        status = prediction['status'] if prediction else 'outside_tracking_interval'
        row = dict(label,status=status)
        visible += bool(label['visible'])
        if status == 'tracked':
            if label['visible']:
                error = math.dist([prediction['x_px']*scale_x,prediction['y_px']*scale_y],label['center_xy'])
                row['error_source_px'] = error
                errors.append(error)
                accepted += 1
            else:
                false_visible += 1
        rows.append(row)
    return {'scope':'Small annotated development sample; missing tracks are counted, never zero error',
            'annotation_sha256':canonical_hash(annotations),'rows':rows,'visible_labels':visible,'accepted_visible_labels':accepted,
            'visible_label_coverage':accepted/max(1,visible),'false_visible_labels':false_visible,
            'accepted_rmse_source_px':math.sqrt(sum(error**2 for error in errors)/len(errors)) if errors else None,
            'accepted_max_error_source_px':max(errors) if errors else None,
            'total_tracked':sum(row['status']=='tracked' for row in tracks['observations']),
            'total_rows':len(tracks['observations']),'seconds':tracks['seconds'],'camera_check':tracks['camera_check']['result']}


def main():
    import argparse
    import json
    parser=argparse.ArgumentParser(description='Bounded ROI appearance tracking; no automatic physics recovery')
    parser.add_argument('--video',required=True)
    parser.add_argument('--config',required=True)
    parser.add_argument('--out',required=True)
    parser.add_argument('--annotations',help='Optional development labels used only after tracking')
    args=parser.parse_args()
    tracks=track_video(args.video,args.out,json.loads(Path(args.config).read_text()))
    if args.annotations:
        write_json(Path(args.out)/'manual-evaluation.json',evaluate_manual(tracks,json.loads(Path(args.annotations).read_text())))
    print(f'Tracking artifacts: {args.out}; physics_ready=False')


if __name__ == '__main__':
    main()
