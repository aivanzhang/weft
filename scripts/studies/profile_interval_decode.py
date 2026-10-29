"""Compare selected pixels with a full sequential decode; encoded-frame completeness remains unknown."""

import argparse
from hashlib import sha256
from pathlib import Path
import shutil
import time

import cv2

from weft.benchmark import seal_suite
from weft.io import digest, local_time, write_json
from weft.video_interval import IntervalVideo


def run(out):
    source = Path("data/balls/triple-collision-240p.webm")
    expected = "f2d9f8125a197ec135dcc4d95e5b1b615c875f097c12ee272d83f8b97baa0887"
    first, last = 2580, 2760
    out.mkdir(parents=True, exist_ok=False)
    before = time.perf_counter()
    if digest(source) != expected:
        raise ValueError("Pinned source changed")
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        capture.release()
        raise ValueError("Source could not be opened")
    full_pixels, decoded = {}, 0
    try:
        metadata = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        while True:
            if decoded > 20_000 or time.perf_counter() - before > 30:
                raise ValueError("Independent full-read comparison exceeds its bounds")
            ok, frame = capture.read()
            if not ok:
                break
            if first <= decoded <= last:
                full_pixels[decoded] = sha256(frame.tobytes()).hexdigest()
            decoded += 1
    finally:
        capture.release()
    full_seconds = time.perf_counter() - before
    before = time.perf_counter()
    selected_pixels = {}
    with IntervalVideo(source, first, last, expected) as reader:
        for index, frame in reader:
            selected_pixels[index] = sha256(frame.tobytes()).hexdigest()
    selected_seconds = time.perf_counter() - before
    if full_pixels != selected_pixels or len(selected_pixels) != last - first + 1:
        raise ValueError("Selected source pixels differ from the full sequential read")
    write_json(out / "report.json", {"created_at": local_time(), "source_sha256": expected,
        "full_read": {"metadata_frames": metadata, "retrieved_frames": decoded, "seconds": full_seconds},
        "selected_read": {"retrieved_frames": len(selected_pixels), "seconds": selected_seconds, "audit": reader.audit},
        "selected_pixels_match_full_read": True, "pixel_sha256_by_absolute_frame": selected_pixels,
        "scope": "One same-backend local comparison, including source hashing and selected-pixel hashing. Full read reaches decoder EOF, not an independent encoded-frame certification. No tracking/model/render/network work; timings are not sustained performance."})
    shutil.copyfile(__file__, out / "study.py")
    shutil.copyfile("src/weft/video_interval.py", out / "video_interval.py")
    print({"full_decoded": decoded, "selected_decoded": len(selected_pixels), "prefix_advanced": reader.audit["prefix_advanced_frames"],
           "full_seconds": full_seconds, "selected_seconds": selected_seconds, "pixels_match": True})
    print(seal_suite(out))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    run(parser.parse_args().out)
