"""Local pose inference with explicit visibility, projection, and anatomy limits."""

from collections import Counter
import importlib.metadata
import math
import os
from pathlib import Path
import tempfile
import time
import urllib.request

import cv2
import numpy as np
from PIL import Image, ImageDraw

from weft.video_io import webm_video_frame_count as _webm_video_frame_count
from weft.io import digest, local_time, write_json
from weft.errors import UnsupportedVideo


MODEL_URL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
MODEL_SHA256 = "59929e1d1ee95287735ddd833b19cf4ac46d29bc7afddbbf6753c459690d574a"
MODEL_BYTES = 5_777_746
MODEL_DOCS = "https://developers.google.com/edge/mediapipe/solutions/vision/pose_landmarker/python"
MODEL_VARIANTS = {
    "lite": {"sha256": MODEL_SHA256, "bytes": MODEL_BYTES},
    "full": {"sha256": "5134a3aad27a58b93da0088d431f366da362b44e3ccfbe3462b3827a839011b1", "bytes": 9_398_198},
    "heavy": {"sha256": "64437af838a65d18e5ba7a0d39b465540069bc8aae8308de3e318aad31fcbc7b", "bytes": 30_664_242},
}
REQUIRED_LANDMARKS = (11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28)
CONNECTIONS = ((11, 12), (11, 13), (13, 15), (12, 14), (14, 16), (11, 23), (12, 24),
               (23, 24), (23, 25), (25, 27), (24, 26), (26, 28), (27, 29), (29, 31), (28, 30), (30, 32))
JOINT_NAMES = ("left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
               "left_hip", "right_hip", "left_knee", "right_knee")
USABLE_STATUSES = {"tracked", "tracked_partial"}
ASSUMPTIONS = [
    "A single person and fixed camera are required; camera motion is not recovered.",
    "Detected pose count is not verified person count: a person can be missed or produce competing pose hypotheses.",
    "MediaPipe image landmarks are model estimates, not manually verified measurements.",
    "World landmarks and body lengths are monocular 3D estimates, not calibrated anatomy or metric ground truth.",
    "The legacy joint-angle projection uses eight hinges; raw landmarks also support an explicit spatial body model, neither is validated biomechanics.",
    "Shoulder and hip angles project motion onto an inferred sagittal plane; lateral motion and twist are not reproduced.",
    "Elbow and knee flexion is zero when straight; signed shoulder and hip flexion is positive forward from downward rest.",
    "No muscle activation, muscle force, EMG, tissue properties, pain, injury risk, or medical assessment is measured from video.",
    "Missing, ambiguous, and low-confidence poses remain explicit; gaps are not fabricated as measured poses.",
    "All supplied video is reconstruction/development evidence, not an independent physical validation or final holdout.",
]


def model_provenance(path, *, variant=None):
    path = Path(path)
    if variant is not None and variant not in MODEL_VARIANTS:
        raise ValueError("Pose model variant must be lite, full or heavy")
    candidates = {name: entry for name, entry in MODEL_VARIANTS.items() if variant is None or name == variant}
    if path.is_file():
        size = path.stat().st_size
        candidates = {name: entry for name, entry in candidates.items() if entry['bytes'] == size}
        checksum = digest(path) if candidates else None
        for name, entry in candidates.items():
            if checksum == entry['sha256']:
                return {"name": f"MediaPipe Pose Landmarker {name.title()}", "variant": name, "model_version": "float16/1",
                        "source_url": f"https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_{name}/float16/1/pose_landmarker_{name}.task",
                        "documentation_url": MODEL_DOCS, **entry, "delegate": "CPU", "network_inference": False}
    raise ValueError("Pose model is missing or differs from the pinned official model variant; run weft fetch-pose-model")


def fetch_pose_model(out=None, *, variant="lite"):
    if variant not in MODEL_VARIANTS:
        raise ValueError("Pose model variant must be lite, full or heavy")
    expected = MODEL_VARIANTS[variant]
    url = f"https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_{variant}/float16/1/pose_landmarker_{variant}.task"
    path = Path(out or f".local/models/pose_landmarker_{variant}.task").resolve()
    if path.exists():
        provenance = model_provenance(path, variant=variant)
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(prefix=".pose-download-", dir=path.parent, delete=False) as stream:
                temporary = Path(stream.name)
                request = urllib.request.Request(url, headers={"User-Agent": "Weft-local-pose-model/1"})
                with urllib.request.urlopen(request, timeout=30) as response:
                    received = 0
                    while block := response.read(256 * 1024):
                        received += len(block)
                        if received > expected['bytes']:
                            raise ValueError("Model download exceeds the pinned size")
                        stream.write(block)
            provenance = model_provenance(temporary, variant=variant)
            # A concurrent fetch must not replace another model or evidence file.
            os.link(temporary, path)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
    sidecar = path.with_suffix(".provenance.json")
    if not sidecar.exists():
        write_json(sidecar, provenance | {"verified_at": local_time()})
    return path


def _unit(vector, name):
    length = float(np.linalg.norm(vector))
    if not math.isfinite(length) or length < 1e-5:
        raise ValueError(f"Degenerate inferred {name}")
    return vector / length


def _flexion(a, b, c):
    u, v = _unit(a - b, "joint segment"), _unit(c - b, "joint segment")
    return float(math.pi - math.acos(float(np.clip(np.dot(u, v), -1, 1))))


def pose_geometry(world_landmarks):
    if not isinstance(world_landmarks, list) or len(world_landmarks) != 33:
        raise ValueError("A pose requires all 33 world landmarks")
    try:
        points = np.array([[landmark[key] for key in ("x", "y", "z")] for landmark in world_landmarks], dtype=float)
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("World landmarks need finite x/y/z coordinates") from error
    if not np.isfinite(points).all() or np.abs(points).max() > 10:
        raise ValueError("World landmarks exceed the supported inferred coordinate envelope")
    shoulder = (points[11] + points[12]) / 2
    pelvis = (points[23] + points[24]) / 2
    up = _unit(shoulder - pelvis, "torso")
    right = _unit(points[12] - points[11], "shoulder width")
    forward = _unit(np.cross(up, right), "body sagittal plane")
    down = -up
    joints, residuals = {}, {}
    for side, shoulder_id, elbow, wrist, hip, knee, ankle in (
        ("left", 11, 13, 15, 23, 25, 27), ("right", 12, 14, 16, 24, 26, 28)
    ):
        joints[f"{side}_elbow"] = _flexion(points[shoulder_id], points[elbow], points[wrist])
        joints[f"{side}_knee"] = _flexion(points[hip], points[knee], points[ankle])
        for name, vector in (("shoulder", points[elbow] - points[shoulder_id]), ("hip", points[knee] - points[hip])):
            vector = _unit(vector, f"{side} {name}")
            forward_component, down_component = float(np.dot(vector, forward)), float(np.dot(vector, down))
            joints[f"{side}_{name}"] = float(math.atan2(forward_component, down_component))
            residuals[f"{side}_{name}"] = float(abs(np.dot(vector, right)))
    distance = lambda a, b: float(np.linalg.norm(points[a] - points[b]))
    lengths = {"torso": float(np.linalg.norm(shoulder - pelvis)),
               "upper_arm": (distance(11, 13) + distance(12, 14)) / 2,
               "forearm": (distance(13, 15) + distance(14, 16)) / 2,
               "thigh": (distance(23, 25) + distance(24, 26)) / 2,
               "shin": (distance(25, 27) + distance(26, 28)) / 2,
               "shoulder_width": distance(11, 12), "hip_width": distance(23, 24)}
    if any(not 0.06 <= value <= 0.95 for value in lengths.values()):
        raise ValueError("Inferred segment lengths are outside the supported adult-sized skeleton envelope")
    return {"joints": joints, "lengths": lengths, "out_of_plane_fraction": residuals}


def _landmark_values(landmarks):
    result = []
    for landmark in landmarks:
        record = {key: float(getattr(landmark, key, 0.0) or 0.0) for key in ("x", "y", "z", "visibility", "presence")}
        if not all(math.isfinite(value) for value in record.values()):
            raise ValueError("Pose inference returned nonfinite landmarks")
        result.append(record)
    return result


def summarize_result(result):
    count = len(result.pose_landmarks)
    if len(result.pose_world_landmarks) != count:
        return {"status": "invalid_landmark_count", "landmarks": [], "world_landmarks": [], "pose_count": count,
                "reason": "Image and world pose counts disagree"}
    if count != 1:
        return {"status": "no_pose" if count == 0 else "multiple_people", "landmarks": [], "world_landmarks": [],
                "pose_count": count,
                "identity_status": "no_model_hypothesis" if count == 0 else "unresolved_multiple_pose_hypotheses",
                "poses": [{"landmarks": _landmark_values(image), "world_landmarks": _landmark_values(world)}
                          for image, world in zip(result.pose_landmarks, result.pose_world_landmarks)]}
    landmarks = _landmark_values(result.pose_landmarks[0])
    world = _landmark_values(result.pose_world_landmarks[0])
    return summarize_landmarks(landmarks, world)


def summarize_landmarks(landmarks, world):
    """Recompute derived geometry after an explicitly audited identity permutation."""
    record = {"landmarks": landmarks, "world_landmarks": world, "pose_count": 1,
              "identity_status": "single_model_hypothesis_not_verified_person_count"}
    if len(landmarks) != 33 or len(world) != 33:
        return record | {"status": "invalid_landmark_count"}
    visible = {index: landmarks[index]["visibility"] >= 0.5 and landmarks[index]["presence"] >= 0.5
               and -0.05 <= landmarks[index]["x"] <= 1.05 and -0.05 <= landmarks[index]["y"] <= 1.05
               for index in REQUIRED_LANDMARKS}
    joint_indices = {"left_shoulder": (11, 13, 23), "right_shoulder": (12, 14, 24),
                     "left_elbow": (11, 13, 15), "right_elbow": (12, 14, 16),
                     "left_hip": (11, 23, 25), "right_hip": (12, 24, 26),
                     "left_knee": (23, 25, 27), "right_knee": (24, 26, 28)}
    record["joint_confidence"] = {key: min(min(landmarks[index]["visibility"], landmarks[index]["presence"])
                                          if -0.05 <= landmarks[index]["x"] <= 1.05 and -0.05 <= landmarks[index]["y"] <= 1.05 else 0.0
                                          for index in indices) for key, indices in joint_indices.items()}
    record["usable_required_landmarks"] = sum(visible.values())
    enough = (sum(visible.values()) >= 8 and all(visible[index] for index in (11, 12, 23, 24))
              and (all(visible[index] for index in (11, 13, 15)) or all(visible[index] for index in (12, 14, 16)))
              and (all(visible[index] for index in (23, 25, 27)) or all(visible[index] for index in (24, 26, 28))))
    if not enough:
        return record | {"status": "low_confidence"}
    try:
        geometry = pose_geometry(world)
    except ValueError as error:
        return record | {"status": "invalid_geometry", "reason": str(error)}
    return record | {"status": "tracked" if all(visible.values()) else "tracked_partial", **geometry}


def _detector(model_path, pose_mode="video"):
    try:
        import mediapipe as mp
    except ImportError as error:
        raise RuntimeError("Exercise videos require the .venv-lab environment with MediaPipe installed") from error
    options = mp.tasks.vision.PoseLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(model_path), delegate=mp.tasks.BaseOptions.Delegate.CPU),
        running_mode=getattr(mp.tasks.vision.RunningMode, pose_mode.upper()), num_poses=2,
        min_pose_detection_confidence=0.5, min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5, output_segmentation_masks=False)
    return mp, mp.tasks.vision.PoseLandmarker.create_from_options(options)


def _draw_pose(frame, record):
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    image.thumbnail((640, 640))
    draw = ImageDraw.Draw(image)
    landmarks = record["landmarks"]
    if len(landmarks) == 33:
        for a, b in CONNECTIONS:
            if min(landmarks[a]["visibility"], landmarks[b]["visibility"]) < 0.5:
                continue
            ax, ay = landmarks[a]["x"] * image.width, landmarks[a]["y"] * image.height
            bx, by = landmarks[b]["x"] * image.width, landmarks[b]["y"] * image.height
            draw.line((ax, ay, bx, by), fill="#EABF2D", width=3)
        for index in REQUIRED_LANDMARKS:
            row = landmarks[index]
            if row["visibility"] >= 0.5:
                x, y = row["x"] * image.width, row["y"] * image.height
                draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill="#214F50", outline="white")
    banner = Image.new("RGB", (image.width, image.height + 40), "#F5EADC")
    banner.paste(image, (0, 40))
    draw = ImageDraw.Draw(banner)
    draw.text((8, 5), f"POSE ESTIMATE | frame {record['frame']} | {record['time_s']:.2f}s | {record['status']}", fill="#214F50")
    draw.text((8, 22), "Inferred joints; muscle activation is not measured", fill="#596562")
    return banner


def _save_previews(previews, timestamps, out, *, artifact_sink=None):
    if not previews:
        return
    durations = [max(40, round((right - left) * 1000)) for left, right in zip(timestamps, timestamps[1:])]
    durations.append(durations[-1] if durations else 100)
    options = {"save_all": True, "append_images": previews[1:], "duration": durations, "loop": 0}
    if artifact_sink is None:
        previews[0].save(out / "tracks.gif", **options)
    else:
        artifact_sink.write_image("tracks.gif", previews[0], **options)
    indices = sorted({0, len(previews) // 3, 2 * len(previews) // 3, len(previews) - 1})
    selected = []
    for index in indices:
        image = previews[index].copy()
        image.thumbnail((480, 520))
        selected.append(image)
    row_height = max(image.height for image in selected)
    sheet = Image.new("RGB", (960, row_height * math.ceil(len(selected) / 2)), "#F5EADC")
    for index, image in enumerate(selected):
        sheet.paste(image, ((index % 2) * 480, (index // 2) * row_height))
    if artifact_sink is None:
        sheet.save(out / "tracks-contact-sheet.png")
    else:
        artifact_sink.write_image("tracks-contact-sheet.png", sheet)


def _pose_json(out, name, value, artifact_sink):
    if artifact_sink is None:
        write_json(out / name, value)
    else:
        artifact_sink.write_json(name, value)


def _quality(evidence):
    frames = evidence["frames"]
    tracked = [frame for frame in frames if frame["status"] in USABLE_STATUSES]
    status_counts = dict(Counter(frame["status"] for frame in frames))
    reliable = {name: [frame["joints"][name] for frame in tracked if frame["joint_confidence"][name] >= 0.5]
                for name in JOINT_NAMES}
    ranges = {name: float(np.ptp(values)) if len(values) >= 4 else 0.0 for name, values in reliable.items()}
    status = {"sampled_frames": len(frames), "tracked_frames": len(tracked), "status_counts": status_counts,
              "tracked_fraction": len(tracked) / max(1, len(frames)), "joint_range_radians": ranges,
              "joint_range_observations": {name: len(values) for name, values in reliable.items()},
              "motion_threshold_radians": 0.08, "minimum_tracked_fraction": 0.65,
              "minimum_tracked_frames": 8,
              "frame_visibility_rule": "At least 8/12 required landmarks, torso, one complete arm, and one complete leg",
              "joint_usable_fraction": {name: sum(frame["joint_confidence"][name] >= 0.5 for frame in tracked) / max(1, len(tracked))
                                        for name in JOINT_NAMES},
              "out_of_plane_max": max((value for frame in tracked for value in frame["out_of_plane_fraction"].values()), default=0.0)}
    errors = []
    if status_counts.get("multiple_people", 0):
        errors.append("Multiple people or competing pose hypotheses were detected; inspect the source and raw poses")
    if len(tracked) < 8 or status["tracked_fraction"] < 0.65:
        errors.append("Insufficient full-body visibility: need at least 8 usable poses and 65% usable sampled frames")
    if max(ranges.values()) < 0.08:
        errors.append("Insufficient measurable exercise motion; a repeated still image is not a dynamic exercise video")
    if tracked and tracked[-1]["time_s"] - tracked[0]["time_s"] < 0.4:
        errors.append("Usable pose sequence must span at least 0.4 seconds")
    status["warnings"] = ["Low-confidence joint angles are retained as uncertain model estimates, not verified motion"] if status_counts.get("tracked_partial") else []
    lateral = [max(frame["out_of_plane_fraction"].values()) for frame in tracked]
    torso_tilts = []
    for frame in tracked:
        points = frame["world_landmarks"]
        torso = np.array([sum(points[i][key] for i in (11, 12)) - sum(points[i][key] for i in (23, 24))
                          for key in ("x", "y", "z")])
        torso_tilts.append(math.acos(float(np.clip(-torso[1] / np.linalg.norm(torso), -1, 1))))
    status["model_coverage"] = {
        "reference_model": "legacy-eight-hinge-upright",
        "scope": "Mismatch diagnostics against pose estimates, not independent reconstruction accuracy",
        "out_of_plane_threshold": 0.35,
        "out_of_plane_frame_fraction": sum(value > 0.35 for value in lateral) / max(1, len(lateral)),
        "torso_tilt_from_camera_up_p95_radians": float(np.percentile(torso_tilts, 95)) if torso_tilts else None,
        "torso_tilt_note": "Camera-relative model estimate; camera roll and true gravity orientation are not calibrated",
        "unrepresented_motion": ["Root translation and torso orientation", "Lateral limbs and twist", "Ground and object contacts"],
    }
    if any(value > 0.35 for value in lateral):
        status["warnings"].append("Legacy eight-hinge projection would discard lateral limb motion; inspect model_coverage and the selected body model")
    if torso_tilts and max(torso_tilts) > math.pi / 6:
        status["warnings"].append("Legacy upright anchoring would not reproduce torso tilt or root movement; inspect the selected body model")
    status["passes"] = not errors
    status["errors"] = errors
    return status



def inspect_exercise(video, out, model_path, progress=None, *, pose_mode="video", pose_identity="raw", artifact_sink=None):
    if pose_mode not in {"video", "image"}:
        raise ValueError("Pose mode must be video or image")
    if pose_identity not in {"raw", "temporal"}:
        raise ValueError("Pose identity must be raw or temporal")
    video, out, model_path = Path(video), Path(out), Path(model_path)
    provenance = model_provenance(model_path) | {"running_mode": pose_mode.upper(),
                                               "temporal_tracking": pose_mode == "video",
                                               "maximum_pose_hypotheses": 2}
    if not video.is_file() or video.stat().st_size > 100_000_000:
        raise ValueError("Exercise input must be a local video of at most 100 MB")
    if not out.is_dir():
        raise ValueError("Output directory must already exist")
    if any((out / name).exists() for name in ("evidence.json", "pose-landmarks.json", "joint-angles.json")):
        raise FileExistsError("Exercise evidence already exists; use a new run directory")
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        cap.release()
        raise ValueError("Exercise video could not be decoded")
    fps, count, width, height = [cap.get(key) for key in (cv2.CAP_PROP_FPS, cv2.CAP_PROP_FRAME_COUNT,
                                                       cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT)]
    if not (all(math.isfinite(value) for value in (fps, count, width, height)) and 1 <= fps <= 240
            and 10 <= count <= 1800 and count / fps <= 60 and 64 <= width <= 4096 and 64 <= height <= 2160):
        cap.release()
        raise ValueError("Exercise video must be 10–1800 frames, <=60seconds, <=4096×2160, and 1–240fps")
    count, width, height = int(count), int(width), int(height)
    stride = max(1, math.ceil(fps / 15), math.ceil(count / 900))
    preview_stride = max(1, math.ceil(math.ceil(count / stride) / 48))
    resized_width = min(960, width)
    resized_height = round(height * resized_width / width)
    started = time.perf_counter()
    evidence = {"schema_version": 1, "profile": "exercise-pose-v1", "source_sha256": digest(video),
                "width": width, "height": height, "fps": fps, "frame_count": count, "frames": [],
                "inference_stride": stride, "sample_rate_fps": fps / stride,
                "model": provenance, "assumptions": ASSUMPTIONS[:],
                "timestamp_basis": "Frame index divided by declared constant fps; per-frame decoder timestamps also preserved",
                "scope": "Monocular pose reconstruction/development only; muscles and biomechanics are not measured"}
    evidence["assumptions"].append(
        "VIDEO mode reuses tracked regions between sampled frames; competing hypotheses remain unresolved."
        if pose_mode == "video" else
        "IMAGE mode infers each sampled frame independently; temporal tracking regions are not reused.")
    previews, preview_times = [], []
    decoded, sampled = 0, 0
    try:
        mp, detector = _detector(model_path, pose_mode)
        try:
            for frame_index in range(1801):
                ok, frame = cap.read()
                if not ok:
                    break
                decoded += 1
                if decoded > 1800 or decoded / fps > 60:
                    raise ValueError("Decoded exercise video exceeds the frame/duration envelope")
                if frame.shape[:2] != (height, width):
                    raise ValueError("Video resolution changes within the exercise recording")
                if frame_index == 0:
                    if artifact_sink is not None:
                        artifact_sink.write_source_preview("background.jpg", frame)
                    elif not cv2.imwrite(str(out / "background.jpg"), frame):
                        raise OSError("Could not write source preview")
                if frame_index % stride:
                    continue
                if sampled >= 900:
                    raise ValueError("Pose inference exceeds the 900-frame bound")
                resized = cv2.resize(frame, (resized_width, resized_height), interpolation=cv2.INTER_AREA)
                rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
                inference_started = time.perf_counter()
                image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                result = (detector.detect_for_video(image, round(frame_index * 1000 / fps))
                          if pose_mode == "video" else detector.detect(image))
                decoder_timestamp = float(cap.get(cv2.CAP_PROP_POS_MSEC)) / 1000
                record = summarize_result(result) | {"frame": frame_index, "time_s": frame_index / fps,
                                                     "decoder_timestamp_s": decoder_timestamp if math.isfinite(decoder_timestamp) else None,
                                                     "id": f"{evidence['source_sha256']}:frame:{frame_index}",
                                                     "inference_seconds": time.perf_counter() - inference_started}
                evidence["frames"].append(record)
                if sampled % preview_stride == 0:
                    previews.append(_draw_pose(resized, record))
                    preview_times.append(record["time_s"])
                sampled += 1
                if progress and (sampled == 1 or sampled % 15 == 0):
                    progress("tracking", {"message": "Local CPU pose estimation", "decoded_frames": decoded,
                                           "frames": count, "sampled_frames": sampled, "pose_status": record["status"]})
                if time.perf_counter() - started > 180:
                    raise TimeoutError("Pose inference exceeded its 180-second local budget")
            if abs(decoded - count) > 1:
                if video.suffix.lower() not in {".webm", ".mkv"}:
                    raise ValueError(f"Incomplete exercise decode: metadata {count}, decoded {decoded}")
                encoded_count = _webm_video_frame_count(video)
                evidence["container_frame_audit"] = {"method": "Matroska unlaced video-block count",
                                                       "encoded_video_frames": encoded_count,
                                                       "decoded_frames": decoded, "matches": encoded_count == decoded}
                if encoded_count != decoded:
                    raise ValueError(f"Incomplete exercise decode: encoded {encoded_count}, decoded {decoded}")
        finally:
            detector.close()
    finally:
        cap.release()
        evidence["decoded_frames"] = decoded
        evidence["elapsed_seconds"] = time.perf_counter() - started
        try:
            evidence["model"]["mediapipe_version"] = importlib.metadata.version("mediapipe")
        except importlib.metadata.PackageNotFoundError:
            evidence["model"]["mediapipe_version"] = "unavailable"
        # Partial failures remain inspectable; they are never silently replaced with synthetic poses.
        _pose_json(out, "pose-landmarks.json", {"frames": evidence["frames"], "model": evidence["model"], "scope": evidence["scope"]}, artifact_sink)
        _pose_json(out, "evidence.json", evidence, artifact_sink)
        _save_previews(previews, preview_times, out, artifact_sink=artifact_sink)
    evidence["decode_metadata_discrepancy"] = {"metadata_frames": count, "decoded_frames": decoded,
                                              "difference": decoded - count,
                                              "note": "One-frame rounding is retained; larger WebM/Matroska differences require an independent encoded video-block count matching every decoded frame"}
    resolve_pose_identity(evidence, out, pose_identity, artifact_sink=artifact_sink)
    return finalize_evidence(evidence, out, artifact_sink=artifact_sink)


def finalize_evidence(evidence, out, *, artifact_sink=None):
    out = Path(out)
    tracked = [frame for frame in evidence["frames"] if frame["status"] in USABLE_STATUSES]
    evidence["lengths"] = {name: float(np.median([frame["lengths"][name] for frame in tracked])) for name in tracked[0]["lengths"]} if tracked else {}
    evidence["reference"] = [{"time_s": frame["time_s"], "joints": frame["joints"], "evidence_id": frame["id"],
                              "joint_confidence": frame["joint_confidence"], "pose_status": frame["status"],
                              "identity_status": frame.get("identity_status"),
                              "identity_derivation_id": frame.get("identity_derivation_id"),
                              **({"image_refinement": frame["image_refinement"]} if "image_refinement" in frame else {})} for frame in tracked]
    evidence["origin_time_s"] = tracked[0]["time_s"] if tracked else None
    evidence["quality"] = _quality(evidence)
    _pose_json(out, "joint-angles.json", {"reference": evidence["reference"], "quality": evidence["quality"],
                                          "lengths": evidence["lengths"], "assumptions": evidence["assumptions"]}, artifact_sink)
    _pose_json(out, "evidence.json", evidence, artifact_sink)
    if not evidence["quality"]["passes"]:
        raise UnsupportedVideo("; ".join(evidence["quality"]["errors"]))
    return evidence


def resolve_pose_identity(evidence, out, mode, *, artifact_sink=None):
    """Keep raw detector evidence immutable; recompute only explicitly reassigned poses."""
    if mode not in {"raw", "temporal"}:
        raise ValueError("Pose identity must be raw or temporal")
    raw_path = Path(out) / "pose-landmarks.json"
    if mode == "temporal":
        from weft.pose_identity import repair_pose_identities
        from weft import pose_identity
        frames, audit = repair_pose_identities(evidence["frames"])
        repaired = []
        for row in frames:
            if "identity_raw" not in row:
                continue
            row.update(summarize_landmarks(row["landmarks"], row["world_landmarks"]))
            row["identity_status"] = "temporally_reassigned_not_verified_anatomy"
            row["identity_derivation_id"] = row["id"] + ":temporal-identity-v1"
            row["identity_recompute_required"] = False
            repaired.append(row["id"])
        evidence["frames"] = frames
        audit.update(implementation_sha256=digest(Path(pose_identity.__file__)), repaired_evidence_ids=repaired)
        evidence["assumptions"].append("Short isolated anatomical identity swaps may be reassigned using neighboring poses; raw estimates and abstentions remain preserved. Initial/global identity is not verified.")
    else:
        audit = {"schema_version": 1, "repaired_evidence_ids": [],
                 "scope": "Raw detector identities retained; anatomical left/right is not independently verified"}
    audit.update(mode=mode, source_sha256=evidence["source_sha256"],
                 raw_artifact="pose-landmarks.json", raw_artifact_sha256=digest(raw_path))
    evidence["pose_identity"] = {"mode": mode, "audit_artifact": "identity-audit.json",
                                 "repaired_frames": len(audit["repaired_evidence_ids"]),
                                 "raw_artifact_sha256": audit["raw_artifact_sha256"]}
    _pose_json(Path(out), "identity-audit.json", audit, artifact_sink)
