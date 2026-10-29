"""Generate an analytic long-source interval fixture, not a physical recording."""

import argparse
from pathlib import Path
import shutil

import cv2
import numpy as np

from weft.benchmark import seal_suite
from weft.io import digest, write_json


def run(out):
    out.mkdir(parents=True, exist_ok=False)
    video = out / "video.avi"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"FFV1"), 2, (240, 160))
    if not writer.isOpened():
        raise RuntimeError("Lossless fixture codec unavailable")
    pattern = np.random.default_rng(8).integers(30, 230, (28, 24), dtype=np.uint8)
    truth = []
    try:
        for index in range(130):
            frame = np.full((160, 240), 120, np.uint8)
            x, y = 40 + index, 45
            frame[y:y+28, x:x+24] = pattern
            writer.write(cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR))
            truth.append({"frame": index, "time_s": index/2, "center_xy": [x+12, y+14]})
    finally:
        writer.release()
    source_hash = digest(video)
    config = {"source_label": "Analytic grayscale rectangle, 130 frames/65 seconds; not a physical video. Explicit interval80–119 retains four pre-initialization frames.",
              "tracking": {"schema_version": 1, "source_sha256": source_hash,
                           "start_frame": 80, "end_frame": 119, "initialization_frame": 84,
                           "decode_interval": [80, 119],
                           "objects": [{"id": "rectangle", "label": "Analytic textured rectangle",
                                        "box_xywh": [124, 45, 24, 28], "role": "active", "shape": "box",
                                        "provenance": "Authored rectangle at frame84; known fixture identity, not a detector finding"}]},
              "world": {"controlled_id": "rectangle", "target": [.75, .4], "gravity": [0, 0], "damping": 1., "restitution": .5}}
    write_json(out / "config.json", config)
    write_json(out / "truth.json", {"source_sha256": source_hash, "frames": truth,
                                   "scope": "Closed-form pixel translation, kept out of reconstruction inputs"})
    write_json(out / "report.json", {"source_sha256": source_hash, "frames_written": 130, "fps": 2,
                                    "scope": "Completed deterministic source generation; decode completeness is checked independently by the pipeline"})
    shutil.copyfile(__file__, out / "generator.py")
    print(seal_suite(out))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    run(parser.parse_args().out)
