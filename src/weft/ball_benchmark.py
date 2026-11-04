"""Independent analytic videos for development checks, never physical evidence."""

import json
import math
from pathlib import Path
import shutil
import time

import cv2
import numpy as np

from weft.browser_video import open_writer
from weft.io import digest, local_time, write_json
from weft.reconstruction_coverage import coverage_summary


CASES = ("constant_motion", "fixed_camera_texture", "gravity_flight", "gravity_bounce", "elastic_contact",
         "brief_occlusion", "same_color_ambiguity", "camera_translation")


def _state(name, t):
    yellow = {"id": "yellow", "position": [0.18 + 0.20 * t, 0.24], "radius": 0.035, "bgr": (20, 220, 230)}
    objects = [yellow]
    if name in {"gravity_flight", "gravity_bounce"}:
        gravity, floor, initial = 0.35, 0.562, 0.14
        y = initial + 0.5 * gravity * t * t
        if name == "gravity_bounce":
            contact = math.sqrt(2 * (floor - initial) / gravity)
            if t > contact:
                elapsed = t - contact
                y = floor - 0.8 * gravity * contact * elapsed + 0.5 * gravity * elapsed * elapsed
        yellow["position"] = [0.25 + 0.10 * t, y]
    elif name == "elastic_contact":
        # Equal masses exchange velocities in a one-dimensional elastic collision.
        contact = (0.58 - 0.18 - 0.07) / 0.30
        yellow["position"] = [0.18 + 0.30 * min(t, contact), 0.24]
        objects.append({"id": "blue", "position": [0.58 + 0.30 * max(0, t - contact), 0.24],
                        "radius": 0.035, "bgr": (220, 70, 20)})
    elif name == "same_color_ambiguity":
        objects.append({"id": "duplicate", "position": [0.70, 0.42], "radius": 0.035, "bgr": yellow["bgr"]})
    return objects


def create_analytic_probe(name, out):
    """Create a bounded source/config plus withheld analytic scoring coordinates."""
    if name not in CASES:
        raise ValueError("Unknown analytic ball probe")
    out = Path(out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    width, height, fps = 480, 288, 30
    count = 66 if name == "gravity_bounce" else 54 if name == "elastic_contact" else 45
    config = {"objects": [{"id": "yellow", "label": "Yellow ball", "hue_min": 20, "hue_max": 40,
                           "role": "active", "shape": "circle"}], "controlled_id": "yellow",
              "target": [0.82, 0.24], "source_label": "SYNTHETIC analytic probe; no physical recording"}
    if name in {"gravity_flight", "gravity_bounce"}:
        config.update(gravity=[0, 0.35], damping=1, restitution=0.8)
    if name == "elastic_contact":
        config["objects"].append({"id": "blue", "label": "Blue ball", "hue_min": 100, "hue_max": 125,
                                  "role": "passive", "shape": "circle"})
        config.update(damping=1, restitution=1)
    writer = open_writer(cv2, out / "video.mp4", fps, (width, height))
    if not writer.isOpened():
        writer.release()
        raise RuntimeError("MP4 encoder unavailable")
    truth = []
    try:
        for index in range(count):
            t = index / fps
            frame = np.full((height, width, 3), 230, np.uint8)
            if name in {"camera_translation", "fixed_camera_texture"}:
                for y in range(24, height, 36):
                    for x in range(24, width, 36):
                        cv2.rectangle(frame, (x - 3, y - 3), (x + 3, y + 3), (100, 100, 100), -1)
            for obj in _state(name, t):
                x, y = [round(value * width) for value in obj["position"]]
                visible = not (name == "brief_occlusion" and 20 <= index <= 23)
                if visible:
                    cv2.circle(frame, (x, y), round(obj["radius"] * width), obj["bgr"], -1, cv2.LINE_AA)
                truth.append({"frame": index, "time_s": t, "object_id": obj["id"],
                              "position": obj["position"], "visible": visible})
            if name == "camera_translation":
                frame = cv2.warpAffine(frame, np.array([[1., 0., index * 0.5], [0., 1., 0.]]),
                                       (width, height), borderValue=(230, 230, 230))
            writer.write(frame)
    finally:
        writer.release()
    cap = cv2.VideoCapture(str(out / "video.mp4"))
    decoded = 0
    try:
        while cap.read()[0]:
            decoded += 1
    finally:
        cap.release()
    if decoded != count:
        raise RuntimeError(f"Probe decode mismatch: {decoded}/{count}")
    write_json(out / "config.json", config)
    write_json(out / "truth.json", {"source_kind": "analytic_synthetic", "rows": truth,
                                   "used_by_reconstruction": False, "physical_evidence": False})
    source = {"schema_version": 1, "created_at": local_time(), "case": name,
              "generator": "weft.ball_benchmark.create_analytic_probe", "source_kind": "analytic_synthetic",
              "physical_evidence": False, "source_sha256": digest(out / "video.mp4"),
              "frames": count, "fps": fps, "width": width, "height": height,
              "independence": "Closed-form trajectories do not import or run the Pymunk environment",
              "authored_inputs": ["Color identities, roles, target", "Gravity/restitution for flight, bounce, and elastic contact"],
              "expected": "unsupported" if name in {"same_color_ambiguity", "camera_translation"} else "supported",
              "limits": ["Development software probes, not real-world or held-out physical validation",
                         "Camera-shift truth remains in world coordinates; tracking comparison is not scored for this negative case"]}
    write_json(out / "provenance.json", source)
    return {"name": name, "video": str(out / "video.mp4"), "config": str(out / "config.json"),
            "truth": str(out / "truth.json"), "expected": source["expected"], "source_sha256": source["source_sha256"]}


def run_analytic_probes(out):
    """Measure every probe and retain failures instead of turning them into passes."""
    from weft.scene_vision import fit_scene, inspect_video
    out = Path(out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    results = []
    started = time.monotonic()
    implementation = out / "implementation-source"
    implementation.mkdir()
    hashes = {}
    for name in ("ball_benchmark.py", "scene_vision.py", "scene.py", "reconstruction_coverage.py", "initial_motion.py"):
        source = Path(__file__).with_name(name)
        hashes[name] = digest(source)
        shutil.copyfile(source, implementation / name)
    write_json(out / "implementation.json", hashes)
    for name in CASES:
        source = create_analytic_probe(name, out / name / "source")
        run = out / name / "run"
        run.mkdir()
        row = {"name": name, "expected": source["expected"], "source_sha256": source["source_sha256"]}
        try:
            evidence, scene = inspect_video(source["video"], run, json.loads(Path(source["config"]).read_text()))
            fit_scene(evidence, scene, run)
            measured = json.loads((run / "tracked-objects.json").read_text())["observations"]
            truth = {(r["frame"], r["object_id"]): r for r in json.loads(Path(source["truth"]).read_text())["rows"]}
            errors = [math.dist([r["x"], r["y"]], truth[(r["frame"], r["object_id"])]["position"]) * evidence["width"]
                      for r in measured if truth[(r["frame"], r["object_id"])]["visible"]]
            report = json.loads((run / "reconstruction.json").read_text())
            replay_errors = {}
            for frame in report["frames"]:
                for obj in frame["predicted"]:
                    reference = truth[(frame["frame"], obj["id"])]
                    if reference["visible"]:
                        replay_errors.setdefault(obj["id"], []).append(
                            math.dist(obj["position"], reference["position"]) * evidence["width"])
            row.update(status="reconstructed", detections=len(measured),
                       tracking_rmse_px=math.sqrt(sum(v * v for v in errors) / len(errors)),
                       analytic_replay_rmse_px={key: math.sqrt(sum(v * v for v in values) / len(values))
                                                for key, values in replay_errors.items()},
                       development_replay_rmse_px=report["after"]["rmse_px"],
                       replay_gate=report["replay_gate"], camera_check=evidence["camera_check"],
                       reconstruction_coverage=coverage_summary(evidence['reconstruction_coverage']),
                       excluded_frames=evidence["excluded_frames"])
        except (ValueError, RuntimeError) as error:
            row.update(status="rejected", error=str(error))
            write_json(run / "failure.json", row)
        row["expectation_met"] = (row["status"] == "rejected" if row["expected"] == "unsupported"
                                  else row["status"] == "reconstructed" and row["replay_gate"]["passes"])
        results.append(row)
        write_json(out / "results.json", {"scope": "Synthetic development probes, not physical validation",
                                          "results": results, "seconds": time.monotonic() - started})
    if any(digest(Path(__file__).with_name(name)) != expected for name, expected in hashes.items()):
        raise RuntimeError("Probe implementation changed during the run; preserve results and rerun in a new directory")
    write_json(out / "manifest.json", {"schema": 1, "created_at": local_time(),
               "files": {str(path.relative_to(out)): digest(path) for path in sorted(out.rglob("*")) if path.is_file()}})
    return results
