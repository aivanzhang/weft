"""Bounded eight-object CPU tracking profile; not browser or physical accuracy evidence."""

import argparse
import importlib.util
from pathlib import Path
import shutil
import time

import cv2
import numpy as np

from weft.benchmark import seal_suite
from weft.generic_vision import AppearanceTrack
from weft.io import digest, local_time, write_json


def run(out):
    out.mkdir(parents=True, exist_ok=False)
    baseline_path = Path("artifacts/explicit-modes-e2e-03/rolling-grayscale/run/implementation-source/generic_vision.py")
    module_spec = importlib.util.spec_from_file_location("saved_appearance_baseline", baseline_path)
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    rng = np.random.default_rng(72)
    textures = [rng.integers(70, 240, (32, 32), dtype=np.uint8) for _ in range(8)]
    boxes = [(50 + (i % 4) * 220, 100 + (i // 4) * 220, 32, 32) for i in range(8)]
    settings = {"search_radius_px": 60, "minimum_score": .7, "minimum_margin": .06, "max_gap_frames": 8}
    def frame(index):
        image = np.full((540, 960, 3), 120, np.uint8)
        delta = round(10 * np.sin(index / 15))
        for i, (x, y, w, h) in enumerate(boxes):
            hsv = np.empty((h, w, 3), np.uint8)
            hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2] = i * 20, 200, textures[i]
            image[y:y+h, x+delta:x+delta+w] = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        return image
    reports = {}
    for label, cls in (("baseline", module.AppearanceTrack), ("candidate", AppearanceTrack)):
        image = frame(0)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        extra = {"color": image} if label == "candidate" else {}
        trackers = [cls(gray, box, settings, **extra) for box in boxes]
        durations, accepted = [], 0
        for index in range(1, 301):
            image = frame(index)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            extra = {"color": image} if label == "candidate" else {}
            start = time.perf_counter()
            for tracker in trackers:
                result = tracker.locate(gray, **extra)
                accepted += result["status"] == "tracked"
                tracker.commit(result)
            durations.append((time.perf_counter() - start) * 1000)
        reports[label] = {"frames": 300, "objects": 8, "accepted": accepted, "expected": 2400,
                          "frame_p50_ms": float(np.percentile(durations, 50)),
                          "frame_p95_ms": float(np.percentile(durations, 95)),
                          "frame_max_ms": max(durations), "total_tracking_seconds": sum(durations)/1000}
        if accepted != 2400:
            raise ValueError("Tracking coverage changed in the performance control")
    paths = [baseline_path, Path("src/weft/generic_vision.py"), Path("src/weft/appearance_consistency.py"), Path(__file__)]
    for i, path in enumerate(paths):
        shutil.copyfile(path, out / f"{i}-{path.name}")
    write_json(out / "report.json", {"created_at": local_time(), **reports, "inputs": {str(p): digest(p) for p in paths},
               "scope": "One local CPU process, 960x540 analytic frames and 8 simultaneous textured targets. Tracking and verification only; excludes decoding, frame creation, previews, HTTP and browser rendering. Single sequential baseline/candidate comparison; not sustained browser performance."})
    print(reports)
    print(seal_suite(out))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True, type=Path)
    run(parser.parse_args().out)
