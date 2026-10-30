"""Verify a sealed, whole-video YUV420P preparation without relabeling sources.

``load_source_equivalence(bundle)`` returns a frozen ``SourceEquivalence`` with
``original_sha256``, ``derivative_sha256``, ``dimensions``, ``time_base``,
``frames`` (ordered ``FramePair`` tuples), ``source_clock``, ``derivative_clock``,
``manifest_sha256``, ``file_names``, ``total_input_bytes`` and ``limitations``.
``source_clock`` means original media; file_names includes manifest.json.
``raw_bytes(relative_path)`` preserves every sealed file, including both videos;
``inspection()`` returns a fresh summary, not a model-facing evidence export.

The supported producer contract is the existing video-only libx264 CRF0 bundle:
report.json, pre-run.json, frame-mapping-and-pixels.json, clock-origins.json,
source/derivative-decoder-clock.json, source/derivative-frames.sha256, their
command/result/log files, encode-command.json and encode-result.json. Source
media is located by its exact hash; derivative_file names the derivative.
Commands are inspected, never executed; paths recorded in commands are never
opened. This verifies recorded decode evidence, not an independent re-decode.
With ``dir_fd=`` supplied, bundle must be a safe relative path; every component
is opened without following links from a duplicate of the caller-owned directory.
"""

from collections import Counter
from dataclasses import asdict, dataclass, field
from decimal import Decimal, InvalidOperation
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import stat
import time
from types import MappingProxyType


MAX_BUNDLE_BYTES = 64 * 1024 * 1024
MAX_FILE_BYTES = 40 * 1024 * 1024
MAX_JSON_BYTES = 4 * 1024 * 1024
MAX_FILES = 128
MAX_FRAMES = 10_000
MAX_PIXELS = 16_777_216
MAX_CLOCK_SECONDS = 86_400
MAX_JSON_NODES = 400_000
MAX_JSON_DEPTH = 32
_DIGEST = re.compile(r"[0-9a-f]{64}")
_INTEGER = re.compile(r"0|[1-9][0-9]{0,15}")
_TIME_BASE = re.compile(r"1/([1-9][0-9]{0,8})")
_TICK_ROUNDOFF = Decimal("0.0000001")
_LIMITATIONS = (
    "Sealed recorded decode evidence is checked; media is not independently decoded by this loader.",
    "Equivalence covers all recorded native YUV420P video frames and PTS, not audio or rendered colorimetry.",
    "Native and decoder clocks retain separate explicit origins; frame durations may differ and remain separate.",
    "Original and derivative hashes, observation IDs and observation clocks remain distinct; no implicit retiming is authorized.",
    "Pixel equivalence does not verify object identity, geometry, contact, anatomy, scale or camera calibration.",
)


@dataclass(frozen=True, slots=True)
class ClockOrigin:
    native_minus_decoder_pts: int
    native_first_pts: int
    native_last_pts: int
    decoder_first_pts: int
    decoder_last_pts: int


@dataclass(frozen=True, slots=True)
class FramePair:
    source_frame: int
    derivative_frame: int
    source_native_pts: int
    derivative_native_pts: int
    source_decoder_pts: int
    derivative_decoder_pts: int
    source_duration_pts: int
    derivative_duration_pts: int
    pixel_sha256: str
    decoded_bytes: int


@dataclass(frozen=True, slots=True, init=False)
class SourceEquivalence:
    original_sha256: str
    derivative_sha256: str
    dimensions: tuple[int, int]
    time_base: str
    frames: tuple[FramePair, ...]
    source_clock: ClockOrigin
    derivative_clock: ClockOrigin
    manifest_sha256: str
    file_names: tuple[str, ...]
    total_input_bytes: int
    limitations: tuple[str, ...]
    _raw: object = field(repr=False, compare=False)

    def __init__(self):
        raise TypeError("Use load_source_equivalence() to verify the recorded evidence")

    def raw_bytes(self, relative_path):
        """Return exact immutable bytes from the verified snapshot, never a live read."""
        return self._raw[_safe_name(relative_path)]

    def inspection(self):
        return {"profile": "whole-video-source-equivalence-v1",
                "original_sha256": self.original_sha256, "derivative_sha256": self.derivative_sha256,
                "manifest_sha256": self.manifest_sha256, "dimensions": list(self.dimensions),
                "time_base": self.time_base, "frame_count": len(self.frames),
                "source_clock": asdict(self.source_clock), "derivative_clock": asdict(self.derivative_clock),
                "duration_mismatch_frames": sum(row.source_duration_pts != row.derivative_duration_pts
                                                for row in self.frames),
                "verified_files": len(self._raw) - 1, "snapshot_bytes": self.total_input_bytes,
                "limitations": list(self.limitations)}


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _int(value, name, maximum=2**53 - 1):
    _require(type(value) is int and 0 <= value <= maximum, f"Invalid integer {name}")
    return value


def _sha(value):
    _require(type(value) is str and _DIGEST.fullmatch(value), "Invalid lowercase SHA256")
    return value


def _safe_name(name):
    _require(type(name) is str and 0 < len(name) <= 512, "Invalid bundle path")
    path = PurePosixPath(name)
    _require(not any(ord(char) < 32 for char in name) and "\\" not in name and ":" not in name
             and not path.is_absolute() and path.as_posix() == name
             and all(part not in (".", "..") for part in path.parts)
             and 1 <= len(path.parts) <= 8, "Unsafe bundle path")
    return name


def _unique(pairs):
    result = {}
    for key, value in pairs:
        _require(key not in result, "Duplicate JSON key")
        result[key] = value
    return result


def _nonfinite(_):
    raise ValueError("Nonfinite JSON value")


def _json(raw):
    _require(len(raw) <= MAX_JSON_BYTES, "JSON byte bound exceeded")
    try:
        value = json.loads(raw.decode("utf-8"), object_pairs_hook=_unique,
                           parse_float=Decimal, parse_constant=_nonfinite)
        pending, nodes = [(value, 0)], 0
        while pending:
            item, depth = pending.pop()
            nodes += 1
            _require(nodes <= MAX_JSON_NODES and depth <= MAX_JSON_DEPTH, "JSON structural bound exceeded")
            if isinstance(item, dict):
                pending.extend((child, depth + 1) for child in item.values())
            elif isinstance(item, list):
                pending.extend((child, depth + 1) for child in item)
            elif isinstance(item, Decimal):
                _require(item.is_finite(), "Nonfinite JSON value")
        return value
    except (UnicodeError, json.JSONDecodeError, RecursionError, InvalidOperation) as error:
        raise ValueError("Malformed bundle JSON") from error


def _check_deadline(deadline):
    if deadline is not None and time.monotonic() >= deadline:
        raise ValueError("Source-equivalence preflight time bound exceeded")


def _inventory(directory_fd, max_bytes, max_files, deadline):
    names, nodes, total = set(), 0, 0

    def visit(fd, prefix, depth):
        nonlocal nodes, total
        _require(depth <= 8, "Bundle directory depth bound exceeded")
        with os.scandir(fd) as entries:
            for entry in entries:
                _check_deadline(deadline)
                nodes += 1
                _require(nodes <= MAX_FILES * 2 + 1, "Bundle entry bound exceeded")
                name = _safe_name(prefix + entry.name)
                info = entry.stat(follow_symlinks=False)
                if stat.S_ISDIR(info.st_mode):
                    child = os.open(entry.name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
                    try:
                        visit(child, name + "/", depth + 1)
                    finally:
                        os.close(child)
                else:
                    _require(stat.S_ISREG(info.st_mode), "Bundle requires regular files, not links or devices")
                    names.add(name)
                    total += info.st_size
                    _require(len(names) <= max_files and info.st_size <= MAX_FILE_BYTES
                             and total <= max_bytes, "Bundle file or byte bound exceeded")
    visit(directory_fd, "", 0)
    return names


def _read(directory_fd, name, limit):
    parts, fd = PurePosixPath(_safe_name(name)).parts, os.dup(directory_fd)
    try:
        for part in parts[:-1]:
            child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
            os.close(fd)
            fd = child
        file_fd = os.open(parts[-1], os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=fd)
        with os.fdopen(file_fd, "rb") as stream:
            before = os.fstat(stream.fileno())
            _require(stat.S_ISREG(before.st_mode) and before.st_size <= limit, "Bundle file bound or type invalid")
            raw = stream.read(limit + 1)
            after = os.fstat(stream.fileno())
            _require(len(raw) <= limit and (before.st_size, before.st_mtime_ns, before.st_ctime_ns)
                     == (after.st_size, after.st_mtime_ns, after.st_ctime_ns), "Bundle file changed during read")
            return raw
    finally:
        os.close(fd)


def _open_bundle(bundle, dir_fd, deadline):
    if dir_fd is None:
        return os.open(Path(bundle), os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    _require(type(dir_fd) is int and dir_fd >= 0, "Invalid bundle directory descriptor")
    parts = PurePosixPath(_safe_name(os.fspath(bundle))).parts
    fd = os.dup(dir_fd)
    try:
        _require(stat.S_ISDIR(os.fstat(fd).st_mode), "Bundle descriptor must refer to a directory")
        for part in parts:
            _check_deadline(deadline)
            child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
            os.close(fd)
            fd = child
        return fd
    except BaseException:
        os.close(fd)
        raise


def _snapshot(bundle, max_bytes, max_files, deadline, dir_fd):
    _check_deadline(deadline)
    fd = _open_bundle(bundle, dir_fd, deadline)
    try:
        names = _inventory(fd, max_bytes, max_files, deadline)
        raw = {"manifest.json": _read(fd, "manifest.json", min(MAX_JSON_BYTES, max_bytes))}
        manifest = _json(raw["manifest.json"])
        _require(isinstance(manifest, dict) and type(manifest.get("schema")) is int
                 and manifest["schema"] == 1, "Unsupported bundle manifest")
        files = manifest.get("files")
        _require(isinstance(files, dict) and 1 <= len(files) < max_files, "Invalid manifest file count")
        for name, digest in files.items():
            _safe_name(name)
            _sha(digest)
        _require("manifest.json" not in files and names == set(files) | {"manifest.json"},
                 "Bundle files do not match the complete seal")
        total = len(raw["manifest.json"])
        for name, expected in files.items():
            _check_deadline(deadline)
            data = _read(fd, name, min(MAX_FILE_BYTES, max_bytes - total))
            _require(hashlib.sha256(data).hexdigest() == expected, f"Sealed file SHA256 mismatch: {name}")
            total += len(data)
            raw[name] = data
        return raw, files
    finally:
        os.close(fd)


def _framehash(raw):
    _require(len(raw) <= MAX_JSON_BYTES, "Framehash byte bound exceeded")
    try:
        lines = raw.decode("ascii").splitlines()
    except UnicodeError as error:
        raise ValueError("Framehash is not ASCII") from error
    headers, rows = {}, []
    for line in lines:
        if not line:
            continue
        if line.startswith("#"):
            if line == "#stream#, dts,        pts, duration,     size, hash":
                key, value = "columns", True
            else:
                key, separator, value = line[1:].partition(": ")
                _require(separator and key in {"format", "version", "hash", "software", "tb 0",
                                              "media_type 0", "codec_id 0", "dimensions 0", "sar 0"},
                         "Unsupported framehash header or stream")
            _require(not rows and key not in headers, "Duplicate or late framehash header")
            headers[key] = value
            continue
        fields = [value.strip() for value in line.split(",")]
        _require(len(fields) == 6 and fields[0] == "0"
                 and all(_INTEGER.fullmatch(value) for value in fields[1:5]), "Malformed framehash row")
        dts, pts, duration, size = map(int, fields[1:5])
        _require(dts == pts and (not rows or pts > rows[-1][0]), "Framehash PTS/order mismatch")
        rows.append((pts, duration, size, _sha(fields[5])))
        _require(len(rows) <= MAX_FRAMES, "Frame count bound exceeded")
    for key, expected in {"format": "frame checksums", "version": "2", "hash": "SHA256",
                          "media_type 0": "video", "codec_id 0": "rawvideo", "columns": True}.items():
        _require(headers.get(key) == expected, "Unsupported framehash format")
    time_base = headers.get("tb 0", "")
    match = _TIME_BASE.fullmatch(time_base)
    dimensions = re.fullmatch(r"([1-9][0-9]{0,3})x([1-9][0-9]{0,3})", headers.get("dimensions 0", ""))
    _require(match and dimensions and rows, "Missing framehash clock, dimensions or frames")
    width, height = map(int, dimensions.groups())
    denominator = int(match[1])
    _require(width % 2 == height % 2 == 0 and width * height <= MAX_PIXELS, "Unsupported YUV420P dimensions")
    _require(re.fullmatch(r"[1-9][0-9]{0,5}/[1-9][0-9]{0,5}", headers.get("sar 0", "")),
             "Missing or invalid sample aspect ratio")
    for pts, duration, size, _ in rows:
        _require(pts <= denominator * MAX_CLOCK_SECONDS and duration <= denominator * MAX_CLOCK_SECONDS
                 and size == width * height * 3 // 2, "Framehash time or decoded byte count invalid")
    return time_base, (width, height), headers["sar 0"], tuple(rows)


def _decoder(doc, digest, native, denominator):
    _require(isinstance(doc, dict) and doc.get("video_sha256") == digest and doc.get("decoded_to_eof") is True,
             "Decoder table lacks matching source hash or complete EOF evidence")
    rows = doc.get("frames")
    _require(isinstance(rows, list) and len(rows) == len(native), "Decoder frame coverage mismatch")
    ticks = []
    for index, row in enumerate(rows):
        _require(isinstance(row, dict) and _int(row.get("frame"), "decoder frame") == index,
                 "Decoder frames are missing, duplicated or reordered")
        value = row.get("decoder_time_s")
        _require(type(value) in (int, Decimal) and 0 <= value <= MAX_CLOCK_SECONDS, "Invalid decoder time")
        scaled = Decimal(value) * denominator
        rounded = scaled.to_integral_value()
        _require(abs(scaled - rounded) <= _TICK_ROUNDOFF, "Fractional decoder tick is not serialization roundoff")
        tick = int(rounded)
        _require(not ticks or tick > ticks[-1], "Decoder times must strictly increase")
        ticks.append(tick)
    offset = native[0][0] - ticks[0]
    _require(offset >= 0 and all(row[0] - tick == offset for row, tick in zip(native, ticks)),
             "Native/decoder origins drift or time has been scaled")
    return tuple(ticks), ClockOrigin(offset, native[0][0], native[-1][0], ticks[0], ticks[-1])


def _clock_claim(doc, clock, ticks, time_base, *, full):
    _require(isinstance(doc, dict) and doc.get("time_base") == time_base, "Clock origin time base mismatch")
    for key, value in asdict(clock).items():
        _require(_int(doc.get(key), key) == value, "Recorded clock origin mismatch")
    _require(doc.get("all_frames_same_offset") is True, "Clock origin is not explicitly constant")
    if full:
        values = doc.get("decoder_pts")
        _require(isinstance(values, list) and len(values) == len(ticks)
                 and all(_int(value, "recorded decoder tick") == tick for value, tick in zip(values, ticks)),
                 "Clock origin frame table mismatch")


def _command(doc):
    _require(isinstance(doc, list) and 1 <= len(doc) <= 96
             and all(type(value) is str and 0 < len(value) <= 4096 and "\0" not in value for value in doc),
             "Invalid recorded command")
    return doc


def _completed_frames(log, count, deadline):
    _check_deadline(deadline)
    completed = None
    # Match each nonempty CR/LF record once; cross-line whitespace permits quadratic backtracking.
    for index, record in enumerate(re.finditer(r"[^\r\n]+", log)):
        if index % 128 == 0:
            _check_deadline(deadline)
        line = record.group()
        match = re.match(r"[ \t]*frame=[ \t]*([0-9]{1,16})(?![0-9])", line)
        if match and line.find("Lsize=", match.end()) >= 0:
            _require(completed is None, "Completed framehash process count differs from complete frame tables")
            completed = int(match[1])
    _check_deadline(deadline)
    _require(completed == count, "Completed framehash process count differs from complete frame tables")


def _recorded_processes(get, files, original_name, derivative_name, dimensions, time_base, count, pre_run, deadline):
    executable = pre_run.get("ffmpeg_path")
    _sha(pre_run.get("ffmpeg_sha256"))
    for side, media in (("source", original_name), ("derivative", derivative_name)):
        _check_deadline(deadline)
        command = _command(get(f"{side}-frames-command.json"))
        _require(len(command) == 29, "Unsupported framehash command")
        threads, input_path, output_path = command[5], command[8], command[-1]
        _require(threads in {str(value) for value in range(1, 9)}, "Unbounded framehash threads")
        expected = [executable, "-hide_banner", "-nostdin", "-n", "-threads", threads, "-copyts", "-i", input_path,
                    "-map", "0:v:0", "-an", "-sn", "-dn", "-c:v", "rawvideo", "-pix_fmt", "yuv420p",
                    "-threads:v", threads, "-fps_mode", "passthrough", "-enc_time_base:v", "demux",
                    "-f", "framehash", "-hash", "sha256", output_path]
        _require(command == expected and Path(input_path).name == PurePosixPath(media).name
                 and Path(output_path).name == f"{side}-frames.sha256", "Framehash command changes frames or source")
        result = get(f"{side}-frames-result.json")
        _require(isinstance(result, dict) and type(result.get("returncode")) is int and result["returncode"] == 0,
                 "Framehash command did not complete successfully")
        log_bytes = files[f"{side}-frames.log"]
        _require(len(log_bytes) <= MAX_JSON_BYTES, "Framehash log byte bound exceeded")
        log = log_bytes.decode("utf-8")
        native_input = log.split("Output #", 1)[0]
        width, height = dimensions
        _require(any("Video:" in line and re.search(r"\byuv420p(?:\(|,|\s)", line)
                     and re.search(rf"\b{width}x{height}\b", line) for line in native_input.splitlines()),
                 "Native decoder format does not establish full-resolution 8-bit YUV420P")
        _completed_frames(log, count, deadline)
    command = _command(get("encode-command.json"))
    _require(len(command) == 39, "Unsupported encode command")
    threads, input_path, preset, output_path = command[5], command[8], command[23], command[-1]
    _require(threads in {str(value) for value in range(1, 9)}, "Unbounded encoder threads")
    expected = [executable, "-hide_banner", "-nostdin", "-n", "-threads", threads, "-copyts", "-i", input_path,
                "-map", "0:v:0", "-an", "-sn", "-dn", "-map_metadata", "-1", "-map_chapters", "-1",
                "-c:v", "libx264", "-crf", "0", "-preset", preset, "-pix_fmt", "yuv420p", "-threads:v", threads,
                "-bf", "0", "-fps_mode", "passthrough", "-enc_time_base:v", "demux", "-video_track_timescale",
                time_base.split("/")[1], "-movflags", "+faststart", output_path]
    _require(command == expected and Path(input_path).name == PurePosixPath(original_name).name
             and Path(output_path).name == PurePosixPath(derivative_name).name, "Encoding crops, trims, rescales or changes media")
    result = get("encode-result.json")
    _require(isinstance(result, dict) and type(result.get("returncode")) is int and result["returncode"] == 0,
             "Encode did not complete successfully")


def _load(raw, sealed, deadline):
    def get(name):
        _require(name in raw, f"Missing required preparation evidence: {name}")
        return _json(raw[name])

    report, pre_run = get("report.json"), get("pre-run.json")
    mapping, origins = get("frame-mapping-and-pixels.json"), get("clock-origins.json")
    _require(all(isinstance(doc, dict) for doc in (report, pre_run, mapping, origins)), "Invalid preparation document")
    original_sha, derivative_sha = _sha(report.get("source_sha256")), _sha(report.get("derivative_sha256"))
    _require(original_sha != derivative_sha, "This contract requires distinct original and derivative bytes")
    original_names = [name for name, digest in sealed.items() if digest == original_sha]
    derivative_name = _safe_name(report.get("derivative_file"))
    _require(len(original_names) == 1 and sealed.get(derivative_name) == derivative_sha,
             "Actual sealed source/derivative media hashes do not match")
    original_name = original_names[0]
    before = _framehash(raw["source-frames.sha256"])
    after = _framehash(raw["derivative-frames.sha256"])
    _require(before[:3] == after[:3] and len(before[3]) == len(after[3]), "Native frame format or coverage differs")
    time_base, dimensions, _, source_rows = before
    derivative_rows = after[3]
    count, denominator = len(source_rows), int(time_base.split("/")[1])
    source_ticks, source_clock = _decoder(get("source-decoder-clock.json"), original_sha, source_rows, denominator)
    derivative_ticks, derivative_clock = _decoder(get("derivative-decoder-clock.json"), derivative_sha, derivative_rows, denominator)
    _require(mapping.get("source_sha256") == original_sha and mapping.get("derivative_sha256") == derivative_sha
             and mapping.get("time_base") == time_base, "Mapping source identities or native time base differ")
    interval = mapping.get("source_interval")
    _require(type(interval) is list and len(interval) == 2 and all(type(value) is int for value in interval)
             and interval == [0, count - 1], "Only complete original frame intervals are supported")
    _require(mapping.get("media_native_pts_shifted") is False
             and _int(mapping.get("source_pts_offset"), "source_pts_offset") == derivative_clock.native_minus_decoder_pts,
             "Mapping native/decoder offset mismatch")
    rows = mapping.get("frames")
    _require(isinstance(rows, list) and len(rows) == count, "Mapping frame coverage mismatch")
    pairs = []
    for index, (source, derivative, row) in enumerate(zip(source_rows, derivative_rows, rows)):
        _require(source[0] == derivative[0] and source[2:] == derivative[2:], "Native PTS or decoded pixels changed")
        _require(isinstance(row, dict), "Invalid mapping row")
        expected = {"source_frame": index, "relative_frame": index,
                    "source_pts": source[0], "relative_pts": derivative_ticks[index],
                    "source_native_pts": source[0], "derivative_native_pts": derivative[0],
                    "source_decoder_pts": source_ticks[index], "derivative_decoder_pts": derivative_ticks[index],
                    "source_duration_pts": source[1], "derivative_duration_pts": derivative[1]}
        for key, value in expected.items():
            _require(_int(row.get(key), key) == value, f"Mapping {key} differs at frame {index}")
        _require(row.get("source_yuv_sha256") == source[3] and row.get("derivative_yuv_sha256") == derivative[3],
                 "Mapping pixel hash differs from native framehash evidence")
        pairs.append(FramePair(index, index, source[0], derivative[0], source_ticks[index], derivative_ticks[index],
                               source[1], derivative[1], source[3], source[2]))
    for name, clock, ticks in (("source", source_clock, source_ticks), ("derivative", derivative_clock, derivative_ticks)):
        _clock_claim(origins.get(name), clock, ticks, time_base, full=True)
        _clock_claim(report.get("decoder_clock_origin", {}).get(name), clock, ticks, time_base, full=False)
    _require(type(report.get("dimensions")) is list and all(type(value) is int for value in report["dimensions"])
             and report["dimensions"] == list(dimensions) and report.get("time_base") == time_base
             and _int(report.get("frame_count"), "frame_count") == count
             and _int(report.get("first_pts"), "first_pts") == source_rows[0][0]
             and _int(report.get("last_pts"), "last_pts") == source_rows[-1][0]
             and _int(report.get("derivative_bytes"), "derivative_bytes") == len(raw[derivative_name]),
             "Report disagrees with full frame evidence or actual media")
    _require(report.get("mapping_file") == "frame-mapping-and-pixels.json" and report.get("failures") == []
             and report.get("source_and_derivative_decoded_to_eof") is True
             and type(report.get("trailing_frames")) is int and report["trailing_frames"] == 0
             and report.get("media_native_pts_shifted") is False, "Preparation is incomplete or shifted")
    _require(all(report.get(key) is True for key in ("all_frame_mapping_verified", "all_native_pts_identical",
                                                   "all_native_yuv_pixels_identical"))
             and report.get("source_derivative_decoder_times_identical") is (source_ticks == derivative_ticks),
             "Report contradicts verified frame equivalence")
    for side, values in (("source", source_rows), ("derivative", derivative_rows)):
        counts = report.get(f"{side}_duration_counts_pts")
        expected = {str(key): value for key, value in Counter(row[1] for row in values).items()}
        _require(isinstance(counts, dict) and counts == expected and all(type(value) is int for value in counts.values()),
                 "Reported frame durations differ from native evidence")
    _require(_int(report.get("duration_mismatch_frames"), "duration_mismatch_frames")
             == sum(source[1] != derivative[1] for source, derivative in zip(source_rows, derivative_rows)),
             "Reported duration mismatch count differs")
    encoding = report.get("encoding")
    _require(isinstance(encoding, dict) and encoding.get("pixel_format") == "yuv420p"
             and type(encoding.get("bit_depth")) is int and encoding["bit_depth"] == 8
             and encoding.get("codec") == "libx264" and type(encoding.get("crf")) is int and encoding["crf"] == 0
             and all(encoding.get(key) is False for key in ("crop", "trim", "resize", "fps_change"))
             and encoding.get("video_only") is True, "Unsupported encoding or spatial/temporal transformation")
    _require(pre_run.get("source_sha256") == original_sha and pre_run.get("selection") == "entire_original_video"
             and type(pre_run.get("source_interval")) is list
             and all(type(value) is int for value in pre_run["source_interval"])
             and pre_run["source_interval"] == interval and type(pre_run.get("dimensions")) is list
             and all(type(value) is int for value in pre_run["dimensions"])
             and pre_run["dimensions"] == list(dimensions)
             and pre_run.get("native_time_base") == time_base
             and _int(pre_run.get("expected_source_frames"), "expected_source_frames") == count,
             "Producer settings do not bind the complete original sequence")
    _recorded_processes(get, raw, original_name, derivative_name, dimensions, time_base, count, pre_run, deadline)
    result = object.__new__(SourceEquivalence)
    values = {"original_sha256": original_sha, "derivative_sha256": derivative_sha,
              "dimensions": dimensions, "time_base": time_base, "frames": tuple(pairs),
              "source_clock": source_clock, "derivative_clock": derivative_clock,
              "manifest_sha256": hashlib.sha256(raw["manifest.json"]).hexdigest(),
              "file_names": tuple(sorted(raw)), "total_input_bytes": sum(map(len, raw.values())),
              "limitations": _LIMITATIONS, "_raw": MappingProxyType(raw)}
    for name, value in values.items():
        object.__setattr__(result, name, value)
    return result


def load_source_equivalence(bundle, *, max_input_bytes=None, max_files=None, deadline=None, dir_fd=None):
    """Verify saved evidence; dir_fd anchors a relative bundle without consuming the caller's handle."""
    max_bytes = MAX_BUNDLE_BYTES if max_input_bytes is None else max_input_bytes
    file_limit = MAX_FILES + 1 if max_files is None else max_files
    _require(type(max_bytes) is int and 1 <= max_bytes <= MAX_BUNDLE_BYTES, "Invalid bundle byte budget")
    _require(type(file_limit) is int and 2 <= file_limit <= MAX_FILES + 1, "Invalid bundle file budget")
    _require(deadline is None or type(deadline) in (int, float) and math.isfinite(deadline),
             "Invalid preflight deadline")
    try:
        result = _load(*_snapshot(bundle, max_bytes, file_limit, deadline, dir_fd), deadline)
        _check_deadline(deadline)
        return result
    except (OSError, KeyError, TypeError, AttributeError, UnicodeError, OverflowError, InvalidOperation) as error:
        raise ValueError("Malformed, incomplete or unsafe source-equivalence bundle") from error
