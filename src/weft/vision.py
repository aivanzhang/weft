from dataclasses import asdict, dataclass
from pathlib import Path
import math

import cv2
import numpy as np

from weft.io import digest, write_json


@dataclass(frozen=True)
class TrackingConfig:
    hue_min: int = 20
    hue_max: int = 45
    saturation_min: int = 90
    value_min: int = 100
    max_seconds: float = 60
    max_frames: int = 1800
    max_width: int = 960

    def validate(self):
        for name, maximum in (("hue_min", 179), ("hue_max", 179), ("saturation_min", 255), ("value_min", 255)):
            value = getattr(self, name)
            if type(value) is not int or not 0 <= value <= maximum:
                raise ValueError(f"Invalid tracking {name}")
        if self.hue_min > self.hue_max:
            raise ValueError("Tracking hue_min cannot exceed hue_max")
        if (type(self.max_frames) is not int or not 10 <= self.max_frames <= 1800
                or type(self.max_width) is not int or not 64 <= self.max_width <= 960
                or type(self.max_seconds) not in (int, float) or not 0 < self.max_seconds <= 60):
            raise ValueError("Tracking settings exceed the supported resource bounds")
        return self


def analyze(path, out, config=TrackingConfig()):
    config.validate()
    path, out = Path(path), Path(out)
    if not path.is_file() or path.stat().st_size > 100_000_000:
        raise ValueError("Input must be a local video of at most 100 MB")
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError("Video could not be decoded")
    fps, count = cap.get(cv2.CAP_PROP_FPS), int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width, height = int(cap.get(3)), int(cap.get(4))
    if not (math.isfinite(fps) and 1 <= fps <= 240 and 0 < count <= config.max_frames
            and count / fps <= config.max_seconds and 0 < width <= 4096 and 0 < height <= 2160):
        cap.release()
        raise ValueError("Video exceeds the documented frame, duration, or resolution envelope")
    scale = min(1.0, config.max_width / width)
    source_hash = digest(path)
    observations, visibility = [], []
    first_frame = None
    index = 0
    try:
        while index < config.max_frames:
            ok, frame = cap.read()
            if not ok:
                break
            if first_frame is None:
                first_frame = frame.copy()
            if scale < 1:
                frame = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, (config.hue_min, config.saturation_min, config.value_min),
                               (config.hue_max, 255, 255))
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            candidates = []
            for contour in contours:
                area = cv2.contourArea(contour)
                x, y, w, h = cv2.boundingRect(contour)
                perimeter = cv2.arcLength(contour, True)
                circularity = 4 * math.pi * area / max(perimeter ** 2, 1)
                if (0.001 * frame.shape[0] * frame.shape[1] < area < 0.15 * frame.shape[0] * frame.shape[1]
                        and 0.55 < w / h < 1.8 and circularity > 0.45
                        and x > 0 and x + w < frame.shape[1] and y > 0 and y + h < frame.shape[0]):
                    m = cv2.moments(contour)
                    candidates.append((area, m["m10"] / m["m00"] / scale,
                                       m["m01"] / m["m00"] / scale,
                                       math.sqrt(area / math.pi) / scale, circularity))
            if candidates:
                area, x, y, radius, circularity = max(candidates)
                observations.append({"id": f"{source_hash}:frame:{index}", "frame": index,
                                     "time_s": index / fps, "object_id": "ball-0",
                                     "x_px": x, "y_px": y, "radius_px": radius,
                                     "mask_circularity": circularity,
                                     "status": "measured_image_projection"})
                visibility.append({"frame": index, "status": "tracked"})
            else:
                visibility.append({"frame": index, "status": "not_detected_or_clipped"})
            index += 1
    finally:
        cap.release()
    if index != count:
        raise ValueError(f"Incomplete decode: metadata {count} frames, decoded {index}")
    episodes = []
    for row in observations:
        if not episodes or row["frame"] - episodes[-1][-1]["frame"] > max(2, round(fps * 0.2)):
            episodes.append([])
        episodes[-1].append(row)
    episodes = [ep for ep in episodes if len(ep) >= 10]
    if not episodes:
        raise ValueError("No supported yellow rigid-object episode with at least 10 complete detections")
    for ep in episodes:
        if np.ptp([r["x_px"] for r in ep]) < width * 0.15:
            raise ValueError("Detected object does not exhibit sufficient horizontal movement")
    # Split whole passes before fitting; the audit pass never enters the repair payload.
    primary = episodes[0]
    split = max(6, int(len(primary) * 0.65))
    evidence = {"source_sha256": source_hash, "width": width, "height": height,
                "fps": fps, "frame_count": count, "tracking_config": asdict(config),
                "tracker": "local HSV/shape tracker; not a foundation model",
                "generation": primary[:split], "development_validation": primary[split:],
                "audit_episodes": episodes[1:], "visibility": visibility,
                "limits": ["Yellow isolated ball profile, fixed-camera horizontal motion only",
                           "No calibrated metric scale or recovered mass",
                           "Audit is another pass in the same video, not an external physical-control trial",
                           "Tracking profile and source were visually inspected during development"]}
    cv2.imwrite(str(out / "background.jpg"), first_frame)
    write_json(out / "evidence.json", evidence)
    return evidence
