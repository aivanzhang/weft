"""Preserve a complete exercise clip in a lossless local video-only derivative."""

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

import cv2
import imageio_ffmpeg

from weft.benchmark import seal_suite
from weft.io import digest, local_time, write_json
from weft.pipeline import verify_artifacts
from weft.segmentation_evidence import _Bundle, _mapping


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "data/exercise/kettlebell-pushup.webm"
SOURCE_SHA256 = "4664b16bab5fbefd66f3dc9116ced8dfeb6f3b979635403b715bbc5848b0afbc"
FRAME_COUNT = 257
DIMENSIONS = (640, 480)
TIME_BASE = "1/1000"
MAX_DERIVATIVE_BYTES = 37_000_000
MAX_ESTIMATED_INLINE_BYTES = 50_000_000
INLINE_METADATA_ALLOWANCE_BYTES = 1_000_000
TIMEOUT_SECONDS = 60
THREADS = 2


def read_framehash(path):
    if path.stat().st_size > 128_000:
        raise ValueError("Framehash table exceeds the complete-clip envelope")
    lines = path.read_text().splitlines()
    if "#tb 0: 1/1000" not in lines or "#dimensions 0: 640x480" not in lines:
        raise ValueError("Decoded dimensions or native time base changed")
    rows = []
    for line in lines:
        if not line or line.startswith("#"):
            continue
        fields = [part.strip() for part in line.split(",")]
        if len(fields) != 6 or fields[0] != "0" or not re.fullmatch(r"[0-9a-f]{64}", fields[5]):
            raise ValueError("Malformed SHA256 framehash row")
        dts, pts, duration, size = map(int, fields[1:5])
        if dts != pts or duration < 0 or size != DIMENSIONS[0] * DIMENSIONS[1] * 3 // 2:
            raise ValueError("Unexpected native YUV420P frame structure")
        if pts < 0 or (rows and pts <= rows[-1]["pts"]):
            raise ValueError("Native presentation timestamps must increase")
        rows.append({"pts": pts, "duration": duration, "bytes": size, "sha256": fields[5]})
    if len(rows) != FRAME_COUNT or rows[-1]["pts"] - rows[0]["pts"] != 8533:
        raise ValueError("Full source coverage changed; expected 257 frames spanning 8533 native ticks")
    return rows


def decoder_clock(path, output):
    if output.exists():
        raise FileExistsError("Decoder clock output already exists")
    cv2.setNumThreads(THREADS)
    capture = cv2.VideoCapture(str(path), cv2.CAP_FFMPEG, [cv2.CAP_PROP_N_THREADS, THREADS])
    rows = []
    try:
        if not capture.isOpened() or capture.get(cv2.CAP_PROP_N_THREADS) != THREADS:
            raise ValueError("Bounded FFmpeg decoder could not be opened")
        metadata = {"reported_frames": capture.get(cv2.CAP_PROP_FRAME_COUNT),
                    "reported_fps": capture.get(cv2.CAP_PROP_FPS), "backend": capture.getBackendName()}
        for index in range(FRAME_COUNT + 1):
            ok, frame = capture.read()
            if not ok:
                break
            if index == FRAME_COUNT or frame.shape != (DIMENSIONS[1], DIMENSIONS[0], 3):
                raise ValueError("Decoder returned a trailing frame or changed dimensions")
            rows.append({"frame": index, "decoder_time_s": capture.get(cv2.CAP_PROP_POS_MSEC) / 1000})
        if len(rows) != FRAME_COUNT:
            raise ValueError("Decoder ended before all 257 source slots")
    finally:
        capture.release()
    write_json(output, {"opencv_version": cv2.__version__, "threads": THREADS, "video_sha256": digest(path),
                       "metadata": metadata, "decoded_to_eof": True, "frames": rows})


def clock_relationship(native, decoded):
    rows = decoded.get("frames")
    if not isinstance(rows, list) or len(rows) != len(native) or not rows:
        raise ValueError("Native and decoder clocks must cover the same nonempty frame sequence")
    relative = []
    for index, row in enumerate(rows):
        time_s = row.get("decoder_time_s")
        if (row.get("frame") != index or type(time_s) not in (int, float)
                or not math.isfinite(time_s) or not 0 <= time_s <= 3600):
            raise ValueError("Malformed decoder clock or source frame identity")
        ticks = round(time_s * 1000)
        if abs(time_s - ticks / 1000) > 1e-9 or (relative and ticks <= relative[-1]):
            raise ValueError("Decoder clock is not a strictly increasing native-millisecond sequence")
        relative.append(ticks)
    offset = native[0]["pts"] - relative[0]
    if offset < 0 or any(row["pts"] - ticks != offset for row, ticks in zip(native, relative, strict=True)):
        raise ValueError("Native/decoder clocks need one explicit nonnegative origin offset, not per-frame repairs")
    return {"native_minus_decoder_pts": offset, "native_first_pts": native[0]["pts"],
            "native_last_pts": native[-1]["pts"], "decoder_first_pts": relative[0],
            "decoder_last_pts": relative[-1], "decoder_pts": relative,
            "time_base": TIME_BASE, "all_frames_same_offset": True,
            "equation": "native_pts = decoder_relative_pts + native_minus_decoder_pts"}


def prepare(out, *, reuse_source_hashes=None):
    out = Path(out).resolve()
    attribution = SOURCE.with_suffix(".source.json")
    if out.exists():
        raise FileExistsError("Use a new preparation directory; never overwrite saved evidence")
    if digest(SOURCE) != SOURCE_SHA256:
        raise ValueError("Original source SHA256 changed")
    attribution_bytes = attribution.read_bytes()
    if json.loads(attribution_bytes).get("sha256") != SOURCE_SHA256:
        raise ValueError("Source attribution does not identify the original video")
    ffmpeg = Path(imageio_ffmpeg.get_ffmpeg_exe()).resolve()
    started = time.monotonic()
    out.mkdir(parents=True, exist_ok=False)
    source_copy, derivative = out / "source.webm", out / "kettlebell-lossless.mp4"
    shutil.copyfile(SOURCE, source_copy)
    shutil.copyfile(attribution, out / "source-attribution.json")
    shutil.copyfile(__file__, out / "prepare_kettlebell_tracking.py")
    if digest(source_copy) != SOURCE_SHA256:
        raise ValueError("Copied source differs from the original")
    write_json(out / "pre-run.json", {
        "created_at": local_time(), "source_path": str(SOURCE), "source_sha256": SOURCE_SHA256,
        "source_attribution_sha256": digest(attribution), "ffmpeg_path": str(ffmpeg), "ffmpeg_sha256": digest(ffmpeg),
        "script_sha256": digest(Path(__file__)), "expected_source_frames": FRAME_COUNT,
        "known_container_metadata_frames": 259, "dimensions": list(DIMENSIONS), "native_time_base": TIME_BASE,
        "selection": "entire_original_video", "source_interval": [0, FRAME_COUNT - 1],
        "maximum_threads_per_command": THREADS, "commands_concurrent": 1, "command_timeout_seconds": TIMEOUT_SECONDS,
        "maximum_derivative_bytes": MAX_DERIVATIVE_BYTES, "maximum_estimated_inline_bytes": MAX_ESTIMATED_INLINE_BYTES,
        "encoding": "libx264 CRF0, native 8-bit YUV420P, no crop/trim/resize/frame-rate change; video only",
        "api_calls": 0, "requests_prepared": 0, "uploads": 0, "external_cost_usd": 0,
        "reused_source_hash_directory": str(reuse_source_hashes.resolve()) if reuse_source_hashes else None,
        "clock_rule": "Preserve native PTS; independently measure decoder origins and require a constant offset over every frame",
        "scope": "Local media preparation only; not object tracking, calibration or physical validation",
    })

    def execute(name, arguments):
        command = [str(ffmpeg), "-hide_banner", "-nostdin", "-n", *arguments]
        write_json(out / f"{name}-command.json", command)
        command_started = time.monotonic()
        try:
            with (out / f"{name}.log").open("x") as log:
                process = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT,
                                         timeout=TIMEOUT_SECONDS, check=False)
            returncode = process.returncode
        except subprocess.TimeoutExpired:
            write_json(out / f"{name}-result.json", {"status": "timeout", "seconds": time.monotonic() - command_started})
            raise
        write_json(out / f"{name}-result.json", {"returncode": returncode, "seconds": time.monotonic() - command_started})
        if returncode:
            raise RuntimeError(f"{name} failed with exit code {returncode}; preserve its output")

    def hash_video(name, path):
        execute(name, ["-threads", str(THREADS), "-copyts", "-i", str(path), "-map", "0:v:0",
                      "-an", "-sn", "-dn", "-c:v", "rawvideo", "-pix_fmt", "yuv420p", "-threads:v", str(THREADS),
                      "-fps_mode", "passthrough", "-enc_time_base:v", "demux", "-f", "framehash",
                      "-hash", "sha256", str(out / f"{name}.sha256")])
        return read_framehash(out / f"{name}.sha256")

    try:
        if reuse_source_hashes is None:
            before = hash_video("source-frames", source_copy)
        else:
            prior = reuse_source_hashes.resolve()
            old_run = json.loads((prior / "pre-run.json").read_text())
            old_result = json.loads((prior / "source-frames-result.json").read_text())
            old_command = json.loads((prior / "source-frames-command.json").read_text())
            expected_command = [str(ffmpeg), "-hide_banner", "-nostdin", "-n", "-threads", str(THREADS),
                "-copyts", "-i", str(prior / "source.webm"), "-map", "0:v:0", "-an", "-sn", "-dn",
                "-c:v", "rawvideo", "-pix_fmt", "yuv420p", "-threads:v", str(THREADS), "-fps_mode", "passthrough",
                "-enc_time_base:v", "demux", "-f", "framehash", "-hash", "sha256", str(prior / "source-frames.sha256")]
            if (old_run.get("source_sha256") != SOURCE_SHA256 or old_run.get("ffmpeg_sha256") != digest(ffmpeg)
                    or old_result.get("returncode") != 0 or old_command != expected_command
                    or digest(prior / "source.webm") != SOURCE_SHA256):
                raise ValueError("Prior successful source hash command does not match the exact original input/settings")
            reuse_files = ["source-frames.sha256", "source-frames.log", "source-frames-command.json", "source-frames-result.json"]
            for name in reuse_files:
                shutil.copyfile(prior / name, out / name)
            write_json(out / "source-hash-reuse.json", {"directory": str(prior), "source_sha256": SOURCE_SHA256,
                "pre_run_sha256": digest(prior / "pre-run.json"),
                "files": {name: digest(prior / name) for name in reuse_files},
                "scope": "Reuse only completed source hashing; failed zero-origin assumption is replaced by measured decoder offsets"})
            before = read_framehash(out / "source-frames.sha256")
        source_log = (out / "source-frames.log").read_text()
        if not any("Video: vp8" in line and re.search(r"\byuv420p(?:\(|,|\s)", line) and "640x480" in line
                   for line in source_log.splitlines()):
            raise ValueError("Original video does not declare the required native VP8 YUV420P format")
        execute("encode", ["-threads", str(THREADS), "-copyts", "-i", str(source_copy),
                           "-map", "0:v:0", "-an", "-sn", "-dn", "-map_metadata", "-1", "-map_chapters", "-1",
                           "-c:v", "libx264", "-crf", "0", "-preset", "medium", "-pix_fmt", "yuv420p",
                           "-threads:v", str(THREADS), "-bf", "0", "-fps_mode", "passthrough",
                           "-enc_time_base:v", "demux", "-video_track_timescale", "1000", "-movflags", "+faststart",
                           str(derivative)])
        derivative_bytes = derivative.stat().st_size
        estimated_inline_bytes = 4 * ((derivative_bytes + 2) // 3) + INLINE_METADATA_ALLOWANCE_BYTES
        if derivative_bytes > MAX_DERIVATIVE_BYTES or estimated_inline_bytes > MAX_ESTIMATED_INLINE_BYTES:
            raise ValueError("Lossless derivative exceeds its media or estimated inline-size bound")
        after = hash_video("derivative-frames", derivative)
        for name, path in (("source", source_copy), ("derivative", derivative)):
            command = [sys.executable, str(Path(__file__).resolve()),
                       "--decode-clock", str(path), "--clock-output", str(out / f"{name}-decoder-clock.json")]
            write_json(out / f"{name}-decoder-command.json", command)
            with (out / f"{name}-decoder.log").open("x") as log:
                subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=TIMEOUT_SECONDS, check=True)
        decoded_source = json.loads((out / "source-decoder-clock.json").read_text())
        decoded_derivative = json.loads((out / "derivative-decoder-clock.json").read_text())
        clocks = {"source": clock_relationship(before, decoded_source),
                  "derivative": clock_relationship(after, decoded_derivative)}
        write_json(out / "clock-origins.json", clocks)
        mapping = []
        for index, (source, output) in enumerate(zip(before, after, strict=True)):
            if source["pts"] != output["pts"] or source["sha256"] != output["sha256"]:
                raise ValueError(f"Native PTS or decoded YUV pixels changed at source frame {index}")
            mapping.append({"source_frame": index, "relative_frame": index,
                            "source_pts": source["pts"], "relative_pts": clocks["derivative"]["decoder_pts"][index],
                            "source_native_pts": source["pts"], "derivative_native_pts": output["pts"],
                            "source_decoder_pts": clocks["source"]["decoder_pts"][index],
                            "derivative_decoder_pts": clocks["derivative"]["decoder_pts"][index],
                            "source_yuv_sha256": source["sha256"], "derivative_yuv_sha256": output["sha256"],
                            "source_duration_pts": source["duration"], "derivative_duration_pts": output["duration"]})
        derivative_sha256 = digest(derivative)
        write_json(out / "frame-mapping-and-pixels.json", {
            "time_base": TIME_BASE, "source_sha256": SOURCE_SHA256, "derivative_sha256": derivative_sha256,
            "source_interval": [0, FRAME_COUNT - 1], "frames": mapping,
            "source_pts_offset": clocks["derivative"]["native_minus_decoder_pts"],
            "relative_pts_basis": "OpenCV derivative decoder clock; not rewritten media PTS",
            "source_pts_basis": "Original native FFmpeg frame PTS, preserved in the derivative media",
            "media_native_pts_shifted": False,
            "duration_scope": "Native frame durations are retained separately; identical PTS/pixels do not imply identical packet durations",
        })
        if digest(SOURCE) != SOURCE_SHA256 or digest(source_copy) != SOURCE_SHA256 or attribution.read_bytes() != attribution_bytes:
            raise ValueError("Source or attribution changed during preparation")
        report = {
            "status": "ready_local_preparation_only", "created_at": local_time(),
            "source_sha256": SOURCE_SHA256, "derivative_sha256": derivative_sha256,
            "derivative_file": derivative.name, "derivative_bytes": derivative_bytes,
            "estimated_inline_bytes": estimated_inline_bytes, "inline_metadata_allowance_bytes": INLINE_METADATA_ALLOWANCE_BYTES,
            "frame_count": FRAME_COUNT, "dimensions": list(DIMENSIONS), "time_base": TIME_BASE,
            "first_pts": before[0]["pts"], "last_pts": before[-1]["pts"],
            "all_frame_mapping_verified": True, "all_native_yuv_pixels_identical": True,
            "all_native_pts_identical": True, "media_native_pts_shifted": False,
            "decoder_clock_origin": {name: {key: value for key, value in clock.items() if key != "decoder_pts"}
                                     for name, clock in clocks.items()},
            "source_derivative_decoder_times_identical": clocks["source"]["decoder_pts"] == clocks["derivative"]["decoder_pts"],
            "mapping_file": "frame-mapping-and-pixels.json",
            "mapping_clock_conversion": "source native PTS minus explicit source_pts_offset equals derivative decoder-relative PTS",
            "source_and_derivative_decoded_to_eof": True, "trailing_frames": 0,
            "source_duration_counts_pts": dict(Counter(row["duration"] for row in before)),
            "derivative_duration_counts_pts": dict(Counter(row["duration"] for row in after)),
            "duration_mismatch_frames": sum(a["duration"] != b["duration"] for a, b in zip(before, after, strict=True)),
            "source_frame_metadata_discrepancy": {"metadata_frames": 259, "actual_decoded_frames": FRAME_COUNT,
                                                  "basis": "complete source ffmpeg framehash table"},
            "encoding": {"codec": "libx264", "crf": 0, "preset": "medium", "pixel_format": "yuv420p",
                         "bit_depth": 8, "threads": THREADS, "b_frames": 0, "video_only": True,
                         "crop": False, "trim": False, "resize": False, "fps_change": False},
            "limitations": ["Container/codec changed; original complete WebM and attribution are preserved.",
                            "Audio/subtitle/data streams are intentionally omitted from this video-only derivative.",
                            "No request, upload, model inference, API call, tracking result or physical qualification is produced.",
                            "Native media timestamps and decoder-relative clocks have separate recorded origins; no media timestamp is shifted.",
                            "Inline size is an arithmetic estimate with a 1 MB metadata allowance, not an encoded API request."],
            "api_calls": 0, "requests_prepared": 0, "uploads": 0, "external_cost_usd": 0,
            "failures": [], "seconds_before_sealing": time.monotonic() - started,
        }
        write_json(out / "report.json", report)
        sealed = seal_suite(out)
        verified = verify_artifacts(out)
        if verified != sealed:
            raise ValueError("Preparation manifest changed after sealing")
        consumer = _mapping(_Bundle(out, {"report.json", "frame-mapping-and-pixels.json"}), derivative_sha256,
                            [row["decoder_time_s"] for row in decoded_derivative["frames"]], *DIMENSIONS)
        print(json.dumps({"report": report, "derivative_path": str(derivative), "verification": verified,
                          "mapping_consumer_status": consumer["status"], "total_seconds": time.monotonic() - started}, indent=2))
    except Exception as error:
        if not (out / "manifest.json").exists():
            write_json(out / "failure.json", {"created_at": local_time(), "type": type(error).__name__,
                       "reason": str(error), "api_calls": 0, "requests_prepared": 0,
                       "preserved_directory": str(out), "seconds": time.monotonic() - started})
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path)
    parser.add_argument("--reuse-source-hashes", type=Path)
    parser.add_argument("--decode-clock", type=Path)
    parser.add_argument("--clock-output", type=Path)
    arguments = parser.parse_args()
    if arguments.decode_clock:
        if not arguments.clock_output or arguments.out or arguments.reuse_source_hashes:
            parser.error("Decoder mode needs only --decode-clock and --clock-output")
        decoder_clock(arguments.decode_clock, arguments.clock_output)
    else:
        if not arguments.out or arguments.clock_output:
            parser.error("Preparation mode needs --out")
        prepare(arguments.out, reuse_source_hashes=arguments.reuse_source_hashes)
