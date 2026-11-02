"""Collect native presentation ticks without repeating pose inference or its OpenCV decode."""

from copy import deepcopy
from fractions import Fraction
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import re
import selectors
import stat
import subprocess
import tempfile
import time

from weft.io import canonical_hash
from weft.source_clock import DETECTOR_CLOCK, MAX_EVIDENCE_BYTES, MAX_FRAMES, PROFILE, validate_clock, validate_clock_binding

CAPTURE_PROFILE = "source-native-clock-capture-v1"
MAX_SECONDS = 20.
MAX_LOG_BYTES = 2 * 1024 * 1024
MAX_VIDEO_BYTES = 64 * 1024 * 1024
MAX_BINARY_BYTES = 128 * 1024 * 1024
CLEANUP_SECONDS = 2.
_TIME_BASE = re.compile(r"\[Parsed_showinfo_\d+ @ [^\]]+\] config in time_base: ([1-9][0-9]*/[1-9][0-9]*),")
_FRAME = re.compile(r"\[Parsed_showinfo_\d+ @ [^\]]+\] n:\s*(\d+)\s+pts:\s*(-?\d+)\s+pts_time:\S+.*?\ss:(\d+)x(\d+)\s")


def _require(condition, message):
    if not condition:
        raise ValueError("Native clock capture: " + message)


def _check(deadline):
    if time.monotonic() >= deadline:
        raise TimeoutError("Native clock capture exceeded its bounded deadline")


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _json(raw):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            _require(key not in result, "duplicate evidence fields")
            result[key] = value
        return result
    try:
        return json.loads(raw, object_pairs_hook=unique, parse_constant=lambda _: (_ for _ in ()).throw(ValueError("nonfinite JSON")))
    except (ValueError, UnicodeError, RecursionError) as error:
        raise ValueError("Native clock capture requires finite bounded original evidence JSON") from error


def _file_hash(path, maximum, deadline):
    _check(deadline)
    before = path.stat()
    _require(not path.is_symlink() and stat.S_ISREG(before.st_mode) and 0 < before.st_size <= maximum,
             "input must be a bounded regular file")
    value, size = hashlib.sha256(), 0
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            _check(deadline)
            size += len(chunk)
            _require(size <= maximum, "file grew beyond its byte bound")
            value.update(chunk)
    after = path.stat()
    _require((before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) ==
             (after.st_dev, after.st_ino, size, after.st_mtime_ns), "file changed while hashing")
    return value.hexdigest()


def _bundled_ffmpeg():
    # Do not honor IMAGEIO_FFMPEG_EXE, PATH fallbacks or subprocess environment credentials.
    from imageio_ffmpeg import _utils
    filename = _utils.FNAME_PER_PLATFORM.get(_utils.get_platform())
    _require(filename is not None, "no bundled FFmpeg for this platform")
    path = Path(_utils._get_bin_dir()) / filename
    _require(path.is_file() and not path.is_symlink(), "bundled FFmpeg is unavailable")
    return path.absolute()


def _command(binary, video):
    return [str(binary), "-nostdin", "-nostats", "-loglevel", "info", "-xerror", "-copyts",
            "-protocol_whitelist", "file", "-format_whitelist", "mov,matroska,webm,avi",
            "-threads", "1", "-max_pixels", str(4 * 1024 * 1024), "-err_detect", "explode", "-i", str(video), "-map", "0:v:0",
            "-filter_threads", "1", "-vf", "showinfo", "-an", "-sn", "-dn",
            "-frames:v", str(MAX_FRAMES + 1), "-fps_mode", "passthrough", "-f", "null", "-"]


def _run(command, deadline):
    with tempfile.TemporaryDirectory(prefix="weft-native-clock-") as workspace:
        environment = {"PATH": "/usr/bin:/bin", "LANG": "C", "LC_ALL": "C", "TMPDIR": workspace}
        process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                                   stderr=subprocess.PIPE, env=environment, cwd=workspace, close_fds=True)
        output = bytearray()
        try:
            with selectors.DefaultSelector() as selector:
                selector.register(process.stderr, selectors.EVENT_READ)
                while selector.get_map():
                    _check(deadline)
                    for key, _ in selector.select(min(.1, max(0., deadline - time.monotonic()))):
                        chunk = os.read(key.fileobj.fileno(), 65536)
                        if not chunk:
                            selector.unregister(key.fileobj)
                            continue
                        _require(len(output) + len(chunk) <= MAX_LOG_BYTES, "native log exceeds its byte bound")
                        output.extend(chunk)
            _check(deadline)
            result = process.wait(timeout=max(.001, deadline - time.monotonic()))
            _require(result == 0, "FFmpeg rejected the source; exit status " + str(result))
            return bytes(output)
        finally:
            if process.poll() is None:
                process.kill()
            process.wait(timeout=CLEANUP_SECONDS)
            process.stderr.close()


def _parse_native_log(raw):
    _require(type(raw) is bytes and 0 < len(raw) <= MAX_LOG_BYTES, "bounded native log required")
    text = raw.decode("utf-8", errors="replace")
    bases, rows = [], []
    for line in text.splitlines():
        base = _TIME_BASE.search(line)
        if base:
            fraction = Fraction(base[1])
            bases.append(f"{fraction.numerator}/{fraction.denominator}")
        frame = _FRAME.search(line)
        if frame:
            index, pts, width, height = map(int, frame.groups())
            _require(index == len(rows) and len(rows) < MAX_FRAMES, "native frames are incomplete, reordered or capped")
            _require(rows == [] or pts > rows[-1]["pts"], "native PTS must be strictly increasing")
            _require(16 <= width <= 4096 and 16 <= height <= 4096 and width * height <= 4 * 1024 * 1024,
                     "native dimensions exceed the pixel bound")
            rows.append({"frame": index, "pts": pts, "width": width, "height": height})
        elif "Parsed_showinfo_" in line and re.search(r"\]\s+n:", line):
            raise ValueError("Native clock capture: malformed native frame row")
    _require(len(bases) == 1 and rows, "one complete native stream time base and frame table required")
    version = next((line for line in text.splitlines() if line.startswith("ffmpeg version ")), None)
    _require(version is not None and len(version) <= 1024, "native executable version missing")
    return bases[0], rows, version


def collect_native_clock(video, evidence_bytes, decoder_rows, *, max_seconds=20):
    """Return (validated binding, acquisition receipt, raw FFmpeg log); never modify evidence."""
    _require(type(max_seconds) in (int, float) and math.isfinite(max_seconds) and 0 < max_seconds <= MAX_SECONDS,
             "maximum time must be positive and at most 20 seconds")
    started, deadline = time.monotonic(), time.monotonic() + max_seconds
    _require(type(evidence_bytes) is bytes and 0 < len(evidence_bytes) <= MAX_EVIDENCE_BYTES,
             "original evidence bytes exceed the input bound")
    evidence = _json(evidence_bytes)
    _require(isinstance(evidence, dict), "evidence must be an object")
    _require(type(decoder_rows) is list and 1 <= len(decoder_rows) <= MAX_FRAMES,
             "complete bounded original decoder rows required")
    rows = []
    for index, row in enumerate(decoder_rows):
        _require(isinstance(row, dict) and set(row) in ({"frame", "decoder_timestamp_s"},
                 {"frame", "decoder_timestamp_s", "width", "height"}), "decoder row fields differ")
        _require(type(row["frame"]) is int and row["frame"] == index, "decoder rows must be complete and ordered")
        stamp = row["decoder_timestamp_s"]
        _require(type(stamp) in (int, float) and math.isfinite(stamp) and abs(stamp) <= 3600,
                 "every original decoder timestamp must be finite")
        if "width" in row:
            _require(all(type(row[key]) is int and row[key] == evidence.get(key) for key in ("width", "height")),
                     "original decoder dimensions differ from evidence")
        rows.append(dict(row))
    _require(type(evidence.get("decoded_frames")) is int and len(rows) == evidence["decoded_frames"],
             "original decoder count differs from evidence")
    _require(all(type(evidence.get(key)) is int and 16 <= evidence[key] <= 4096 for key in ("width", "height"))
             and evidence["width"] * evidence["height"] <= 4 * 1024 * 1024, "original dimensions exceed the pixel bound")
    source = Path(video).absolute()
    _require(not source.is_symlink(), "source symlinks are unsupported")
    source = source.resolve(strict=True)
    source_sha = _file_hash(source, MAX_VIDEO_BYTES, deadline)
    _require(source_sha == evidence.get("source_sha256"), "source hash differs from original evidence")
    binary = _bundled_ffmpeg()
    binary_sha = _file_hash(binary, MAX_BINARY_BYTES, deadline)
    command = _command(binary, source)
    raw = _run(command, deadline)
    time_base, native, version = _parse_native_log(raw)
    _require(len(native) == len(rows), "native/original decoded frame counts differ")
    for row in native:
        _require(all(type(evidence.get(key)) is int and row[key] == evidence[key] for key in ("width", "height")),
                 "native dimensions differ from original evidence")
    tick, origin = Fraction(time_base), native[0]["pts"]
    binding = {"schema_version": 1, "profile": PROFILE, "source_sha256": source_sha,
        "evidence_sha256": _sha(evidence_bytes), "evidence_canonical_sha256": canonical_hash(evidence),
        "metadata_frames": evidence.get("frame_count"), "decoded_frames": len(rows), "nominal_fps": evidence.get("fps"),
        "time_base": time_base, "native_origin_pts": origin,
        "native_to_reconstruction_offset_s": -float(origin * tick),
        "decoder_minus_native_s": rows[0]["decoder_timestamp_s"] - float(origin * tick),
        "capture_clock_status": "unknown", "detector_clock_status": DETECTOR_CLOCK,
        "rows": [{"frame": index, "pts": row["pts"], "decoder_timestamp_s": rows[index]["decoder_timestamp_s"]}
                 for index, row in enumerate(native)]}
    binding = validate_clock(binding, evidence, evidence_sha256=_sha(evidence_bytes))
    _require(_file_hash(source, MAX_VIDEO_BYTES, deadline) == source_sha, "source changed during capture")
    _require(_file_hash(binary, MAX_BINARY_BYTES, deadline) == binary_sha, "native executable changed during capture")
    _check(deadline)
    receipt = {"schema_version": 1, "profile": CAPTURE_PROFILE, "command": command,
        "source_sha256": source_sha, "evidence_sha256": _sha(evidence_bytes),
        "binding_sha256": canonical_hash(binding), "native_log_sha256": _sha(raw),
        "decoder_rows_sha256": canonical_hash(rows), "ffmpeg_sha256": binary_sha, "ffmpeg_version": version,
        "imageio_ffmpeg_version": importlib.metadata.version("imageio-ffmpeg"),
        "metadata_frames": binding["metadata_frames"], "decoded_frames": len(rows),
        "decoder_minus_native_s": binding["decoder_minus_native_s"], "native_reached_eof": True, "returncode": 0,
        "decoder_rows_origin": "original_pose_decode_callback", "capture_clock_status": "unknown",
        "bounds": {"maximum_seconds": float(max_seconds), "native_frame_cap": MAX_FRAMES + 1,
                   "accepted_frames": MAX_FRAMES, "native_log_bytes": MAX_LOG_BYTES, "source_bytes": MAX_VIDEO_BYTES,
                   "cleanup_seconds": CLEANUP_SECONDS},
        "work": {"native_decodes": 1, "additional_opencv_decodes": 0, "inference_calls": 0, "paid_calls": 0},
        "seconds": time.monotonic() - started}
    validate_capture(binding, receipt, raw, rows, source_sha256=source_sha, evidence_sha256=_sha(evidence_bytes))
    _check(deadline)
    receipt["seconds"] = time.monotonic() - started
    return binding, receipt, raw


def validate_capture(binding, receipt, raw_log, decoder_rows, *, source_sha256, evidence_sha256):
    """Verify retained acquisition custody without a decoder, executable lookup or inference."""
    binding = validate_clock_binding(binding)
    _require(isinstance(receipt, dict), "receipt must be an object")
    expected_keys = {"schema_version", "profile", "command", "source_sha256", "evidence_sha256", "binding_sha256",
        "native_log_sha256", "decoder_rows_sha256", "ffmpeg_sha256", "ffmpeg_version", "imageio_ffmpeg_version",
        "metadata_frames", "decoded_frames", "decoder_minus_native_s", "native_reached_eof", "returncode",
        "decoder_rows_origin", "capture_clock_status", "bounds", "work", "seconds"}
    _require(set(receipt) == expected_keys and type(receipt["schema_version"]) is int and receipt["schema_version"] == 1
             and receipt["profile"] == CAPTURE_PROFILE, "receipt fields differ from its versioned contract")
    try:
        _require(len(json.dumps(receipt, allow_nan=False).encode()) <= 32 * 1024, "receipt exceeds its byte bound")
    except (ValueError, TypeError, RecursionError) as error:
        raise ValueError("Native clock capture: bounded finite receipt required") from error
    _require(type(decoder_rows) is list and len(decoder_rows) == binding["decoded_frames"], "decoder receipt count differs")
    for index, row in enumerate(decoder_rows):
        _require(isinstance(row, dict) and set(row) in ({"frame", "decoder_timestamp_s"},
                 {"frame", "decoder_timestamp_s", "width", "height"}), "decoder receipt fields differ")
        stamp = row["decoder_timestamp_s"]
        _require(type(row["frame"]) is int and row["frame"] == index and type(stamp) in (int, float)
                 and math.isfinite(stamp) and stamp == binding["rows"][index]["decoder_timestamp_s"],
                 "decoder receipt differs from the pinned complete rows")
    time_base, native, version = _parse_native_log(raw_log)
    _require(time_base == binding["time_base"] and len(native) == binding["decoded_frames"]
             and all(native[index]["pts"] == row["pts"] for index, row in enumerate(binding["rows"])),
             "native log ticks differ from the binding")
    _require(all((row["width"], row["height"]) == (native[0]["width"], native[0]["height"]) for row in native),
             "native dimensions changed during capture")
    for index, row in enumerate(decoder_rows):
        if "width" in row:
            _require(all(type(row[key]) is int and row[key] == native[index][key] for key in ("width", "height")),
                     "decoder/native receipt dimensions differ")
    _require(receipt["source_sha256"] == binding["source_sha256"] == source_sha256
             and receipt["evidence_sha256"] == binding["evidence_sha256"] == evidence_sha256
             and receipt["binding_sha256"] == canonical_hash(binding) and receipt["native_log_sha256"] == _sha(raw_log)
             and receipt["decoder_rows_sha256"] == canonical_hash(decoder_rows), "receipt artifact identities differ")
    _require(type(receipt["ffmpeg_sha256"]) is str and re.fullmatch(r"[a-f0-9]{64}", receipt["ffmpeg_sha256"])
             and receipt["ffmpeg_version"] == version and isinstance(receipt["imageio_ffmpeg_version"], str)
             and re.fullmatch(r"[0-9][0-9A-Za-z.+-]{0,63}", receipt["imageio_ffmpeg_version"]), "executable provenance differs")
    command = receipt["command"]
    _require(isinstance(command, list) and len(command) == len(_command(Path("/binary"), Path("/video")))
             and all(isinstance(part, str) and len(part) <= 4096 for part in command), "bounded command receipt required")
    input_index = command.index("-i") + 1 if "-i" in command else 0
    _require(Path(command[0]).is_absolute() and Path(command[input_index]).is_absolute()
             and command == _command(Path(command[0]), Path(command[input_index])), "command differs from guarded capture")
    bounds = receipt["bounds"]
    _require(isinstance(bounds, dict) and type(bounds.get("maximum_seconds")) in (int, float)
             and math.isfinite(bounds["maximum_seconds"]) and 0 < bounds["maximum_seconds"] <= MAX_SECONDS,
             "receipt time bound differs")
    expected_bounds = {"maximum_seconds": bounds["maximum_seconds"], "native_frame_cap": MAX_FRAMES + 1,
        "accepted_frames": MAX_FRAMES, "native_log_bytes": MAX_LOG_BYTES, "source_bytes": MAX_VIDEO_BYTES,
        "cleanup_seconds": CLEANUP_SECONDS}
    _require(canonical_hash(bounds) == canonical_hash(expected_bounds), "receipt work bounds differ")
    _require(type(receipt["seconds"]) in (int, float) and math.isfinite(receipt["seconds"])
             and 0 <= receipt["seconds"] <= bounds["maximum_seconds"], "receipt elapsed time differs")
    _require(all(type(receipt[key]) is int and receipt[key] == binding[key] for key in ("metadata_frames", "decoded_frames"))
             and type(receipt["decoder_minus_native_s"]) in (int, float)
             and receipt["decoder_minus_native_s"] == binding["decoder_minus_native_s"]
             and receipt["native_reached_eof"] is True and type(receipt["returncode"]) is int and receipt["returncode"] == 0
             and receipt["decoder_rows_origin"] == "original_pose_decode_callback" and receipt["capture_clock_status"] == "unknown",
             "receipt count, clock or completion claims differ")
    _require(canonical_hash(receipt["work"]) == canonical_hash({"native_decodes": 1, "additional_opencv_decodes": 0,
             "inference_calls": 0, "paid_calls": 0}), "receipt repeats inference or changes its declared work")
    return deepcopy(receipt)
