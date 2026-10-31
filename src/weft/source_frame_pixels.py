"""Extract exact original frames under one bounded, disposable native decoder."""

import hashlib
import importlib.metadata
import json
import math
import multiprocessing
import os
from pathlib import Path
import platform
import tempfile
import time


PROFILE = "source-frame-pixels-v1"
MAX_SECONDS = 30.
MAX_BYTES = 4 * 1024 * 1024
MAX_FILES = 12
MAX_VIDEO_BYTES = 64 * 1024 * 1024
MAX_READS = 1800
MAX_PIXELS = 4 * 1024 * 1024
MAX_REPORT_BYTES = 128 * 1024
RESERVE_BYTES = 256 * 1024
CLEANUP_SECONDS = 2.
WORKER_CLEANUP_SECONDS = 1.
CAPTURE_OPTIONS = "protocol_whitelist;file|format_whitelist;matroska,webm,mov,avi|enable_drefs;0|use_absolute_path;0"
IMPLEMENTATION_FILES = ("source_frame_pixels.py", "pose_pixel_support_package.py",
                        "pose_pixel_support.py", "shared_world_packaging.py")


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _bytes(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def _check(deadline):
    if time.monotonic() >= deadline:
        raise TimeoutError("Original-frame extraction exceeded its deadline")


def _validate_source(pose, frames):
    from weft import pose_pixel_support_package as package

    docs = {name: package._json(pose.read(name)) for name in package.POSE_FILES}
    evidence, marker = docs["evidence.json"], docs["body-source.json"]
    identity, clock = docs["identity-audit.json"], docs["source-clock.json"]
    source = package.diagnostic._digest(evidence.get("source_sha256"))
    source_name = marker.get("source_artifact")
    if (pose.manifest.get("profile") != "exercise-body-source-v1"
            or pose.manifest.get("status") != "construction_only"
            or evidence.get("profile") != "exercise-pose-v1"
            or marker.get("profile") != "exercise-body-source-v1"
            or marker.get("status") != "construction_only" or marker.get("simulation_ready") is not False
            or marker.get("source_sha256") != source
            or not isinstance(marker.get("pose_options"), dict)
            or marker["pose_options"].get("pose_identity") != "raw"
            or not isinstance(marker.get("artifacts"), dict)
            or source_name not in {"source" + suffix for suffix in (".webm", ".mp4", ".m4v", ".mov", ".mkv", ".avi")}
            or pose.files.get(source_name) != source
            or identity.get("mode") != "raw" or identity.get("source_sha256") != source
            or identity.get("repaired_evidence_ids") != [] or identity.get("raw_artifact") != "pose-landmarks.json"
            or identity.get("raw_artifact_sha256") != pose.files.get("pose-landmarks.json")
            or docs["model.json"] != evidence.get("model") or clock.get("source_sha256") != source
            or any(marker["artifacts"].get(name) != pose.files[name] for name in package.POSE_FILES
                   if name != "body-source.json")):
        raise ValueError("Expected matching sealed original raw-pose/source/model/clock receipts")
    package.diagnostic._digest(identity.get("raw_artifact_sha256"))
    pose_identity = evidence.get("pose_identity", {})
    if (not isinstance(pose_identity, dict) or pose_identity.get("mode", "raw") != "raw"
            or "pose_reinference" in evidence):
        raise ValueError("Only unrepaired raw pose evidence supports original-frame extraction")
    width, height = evidence.get("width"), evidence.get("height")
    if (type(width) is not int or type(height) is not int or not 16 <= width <= 4096
            or not 16 <= height <= 4096 or width * height > MAX_PIXELS):
        raise ValueError("Original dimensions exceed the pixel bound; resizing is forbidden")
    count, metadata, fps = (evidence.get(key) for key in ("decoded_frames", "frame_count", "fps"))
    if (type(count) is not int or not 1 <= count <= MAX_READS
            or type(metadata) is not int or not 1 <= metadata <= MAX_READS
            or type(fps) not in (int, float) or not math.isfinite(fps) or not 1 <= fps <= 240
            or count / fps > 60):
        raise ValueError("Original frame/duration metadata exceeds the supported bound")
    rows = evidence.get("frames")
    if not isinstance(rows, list) or not 1 <= len(rows) <= 900:
        raise ValueError("Original pose-slot inventory exceeds its bound")
    indexed, previous, previous_time = {}, -1, -1.
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("Original pose slots must be records")
        index, stamp, nominal = (row.get(key) for key in ("frame", "decoder_timestamp_s", "time_s"))
        if (type(index) is not int or not previous < index < count
                or row.get("id") != f"{source}:frame:{index}"
                or "decoder_timestamp_s" not in row or "image_refinement" in row or "identity_raw" in row):
            raise ValueError("Original pose slot identity, order or raw scope changed")
        package.diagnostic._finite(nominal, 0., 60.)
        if nominal <= previous_time or abs(nominal - index / fps) > 1e-7:
            raise ValueError("Original nominal frame clock is inconsistent")
        if stamp is not None:
            package.diagnostic._finite(stamp, 0., 3600.)
        indexed[index], previous, previous_time = row, index, nominal
    samples = [{key: row[key] for key in ("frame", "time_s", "decoder_timestamp_s")} for row in rows]
    if clock.get("samples") != samples or clock.get("decoded_frames") != count:
        raise ValueError("Saved native and nominal clock inventories disagree")
    previous_stamp = -1.
    for frame in frames:
        if frame not in indexed:
            raise ValueError("Choose retained original pose frame IDs; unsampled slots are unsupported")
        stamp = indexed[frame]["decoder_timestamp_s"]
        if stamp is None or stamp <= previous_stamp:
            raise ValueError("Selected frames require known strictly increasing native timestamps")
        previous_stamp = stamp
    return evidence, source_name, source


def _implementation(deadline):
    from weft import pose_pixel_support_package as package

    paths = {name: Path(__file__).with_name(name) for name in IMPLEMENTATION_FILES}
    paths.update({name: Path(__file__).parents[2] / name for name in ("pyproject.toml", "requirements-lab.lock")})
    raw = {}
    for name, path in paths.items():
        fd = package._directory(path.parent)
        try:
            raw[name] = package._read(fd, path.name, 256 * 1024, deadline)
        finally:
            os.close(fd)
    versions = {"python": platform.python_version(), **{name: importlib.metadata.version(name)
                for name in ("numpy", "Pillow", "opencv-contrib-python")}}
    return raw, {"files": {name: _sha(value) for name, value in raw.items()}, "versions": versions}


def _write(fd, name, raw, written, deadline, *, terminal=False):
    from weft import pose_pixel_support_package as package

    limit = MAX_BYTES if terminal else MAX_BYTES - RESERVE_BYTES
    if len(written) + 1 > MAX_FILES or sum(map(len, written.values())) + len(raw) > limit:
        raise ValueError("Original-frame output exceeds its byte/file reservation")
    package._write(fd, name, raw, written, deadline)


def _decode(video, output, plan, deadline, work):
    import cv2
    import numpy as np
    from weft import pose_pixel_support_package as package

    cv2.setNumThreads(1)
    fd, cap, written = package._directory(Path(output)), None, {}
    rows, clocks, previous = [], [], -1.
    try:
        _check(deadline)
        work["decoder_opens"] += 1
        cap = cv2.VideoCapture(video, cv2.CAP_FFMPEG, [cv2.CAP_PROP_N_THREADS, 1])
        _check(deadline)
        fields = (("width", cv2.CAP_PROP_FRAME_WIDTH), ("height", cv2.CAP_PROP_FRAME_HEIGHT),
                  ("fps", cv2.CAP_PROP_FPS), ("frame_count", cv2.CAP_PROP_FRAME_COUNT))
        if not cap.isOpened() or any(cap.get(prop) != plan[key] for key, prop in fields):
            raise ValueError("Native decoder metadata differs from the original source receipt")
        backend = cap.getBackendName()
        original = {row["frame"]: row for row in plan["samples"]}
        for index in range(plan["frames"][-1] + 1):
            _check(deadline)
            work["read_calls"] += 1
            ok, bgr = cap.read()
            if ok:
                work["decoded_frames"] += 1
            _check(deadline)
            if not ok or bgr is None or bgr.shape != (plan["height"], plan["width"], 3) or bgr.dtype != np.uint8:
                raise ValueError("Source prefix ended early or changed image shape/type")
            position = cap.get(cv2.CAP_PROP_POS_FRAMES)
            stamp = float(cap.get(cv2.CAP_PROP_POS_MSEC)) / 1000
            if position != index + 1 or not math.isfinite(stamp) or not previous < stamp <= 3600:
                raise ValueError("Native frame position/timestamp is unknown or nonmonotone")
            if index in original and stamp != original[index]["decoder_timestamp_s"]:
                raise ValueError("Native decoder timestamp differs from the exact saved source clock")
            clocks.append([index, stamp, int(position)])
            previous = stamp
            if index not in plan["frames"]:
                continue
            ok, png = cv2.imencode(".png", bgr, [cv2.IMWRITE_PNG_COMPRESSION, 3])
            _check(deadline)
            if not ok or png.nbytes > plan["maximum_png_bytes"] - sum(map(len, written.values())):
                raise ValueError("Full-resolution PNGs exceed the existing pixel-check budget; no resizing allowed")
            raw, name = png.tobytes(), f"frame-{index:04d}.png"
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            _write(fd, name, raw, written, deadline)
            rows.append({"frame": index, "id": original[index]["id"], "file": name,
                "source_sha256": plan["source_sha256"], "time_s": original[index]["time_s"],
                "decoder_timestamp_s": stamp, "width": plan["width"], "height": plan["height"],
                "sha256": _sha(raw), "decoded_rgb_sha256": _sha(rgb.tobytes())})
        return {"rows": rows, "prefix_clocks": clocks, "backend": backend, "work": dict(work)}
    finally:
        if cap is not None:
            cap.release()
        os.close(fd)


def _worker(pipe, video, output, plan, deadline):
    # Native decoding needs no credentials or inherited log streams.
    environment = {key: os.environ[key] for key in ("PATH", "TMPDIR", "TEMP", "TMP") if key in os.environ}
    os.environ.clear()
    os.environ.update(environment)
    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = CAPTURE_OPTIONS
    with open(os.devnull, "wb") as sink:
        os.dup2(sink.fileno(), 1)
        os.dup2(sink.fileno(), 2)
    work = {"decoder_opens": 0, "read_calls": 0, "decoded_frames": 0}
    try:
        result = {"status": "complete", **_decode(video, output, plan, deadline, work)}
        raw = _bytes(result)
        if len(raw) > MAX_REPORT_BYTES:
            raise ValueError("Decoder status exceeds its bounded message size")
        pipe.send_bytes(raw)
    except BaseException as error:
        pipe.send_bytes(_bytes({"status": "failed", "work": work,
            "error": {"type": type(error).__name__, "message": str(error)[:1000]}}))
    finally:
        pipe.close()


def _run_bounded(video, output, plan, deadline):
    from weft import pose_pixel_support_package as package

    context = multiprocessing.get_context("spawn")
    receive, send = context.Pipe(duplex=False)
    process = context.Process(target=_worker, args=(send, str(video), str(output), plan, deadline), daemon=True)
    try:
        _check(deadline)
        process.start()
        send.close()
        if not receive.poll(max(0., deadline - time.monotonic())):
            raise TimeoutError("Native decoder exceeded its wall timeout; partial outputs retained, in-flight counts unknown")
        result = package._json(receive.recv_bytes(MAX_REPORT_BYTES))
        if result.get("status") not in {"complete", "failed"}:
            raise ValueError("Invalid bounded decoder result")
        return result
    except EOFError as error:
        raise ValueError("Native decoder exited without a complete status; in-flight counts unknown") from error
    finally:
        receive.close()
        send.close()
        if process.pid is not None:
            if process.is_alive():
                process.terminate()
            process.join(timeout=WORKER_CLEANUP_SECONDS / 2)
            if process.is_alive():
                process.kill()
                process.join(timeout=WORKER_CLEANUP_SECONDS / 2)
            process.close()


def _output_files(fd, deadline):
    from weft import pose_pixel_support_package as package

    files = {}
    for name in package._inventory(fd, deadline):
        files[name] = package._read(fd, name, MAX_BYTES - sum(map(len, files.values())), deadline)
        if len(files) > MAX_FILES:
            raise ValueError("Extraction output exceeds its file bound")
    return files


def prepare_source_frame_pixels(pose_run, out, *, frames, max_seconds=MAX_SECONDS):
    """Extract 2–4 original pose slots; wall-bound native work, never infer or repair poses."""
    from weft import pose_pixel_support_package as package

    package.diagnostic._finite(max_seconds, 1., MAX_SECONDS)
    if (not isinstance(frames, (list, tuple)) or not 2 <= len(frames) <= 4
            or any(type(frame) is not int or not 0 <= frame < MAX_READS for frame in frames)
            or any(a >= b for a, b in zip(frames, frames[1:]))):
        raise ValueError("Supply two to four strictly increasing original frame IDs within 0–1799")
    frames = list(frames)
    source, target = package._path(pose_run), package._path(out)
    if target.is_relative_to(source) or source.is_relative_to(target):
        raise ValueError("Extraction output must be fresh and outside the immutable pose input")
    started, pose, output_fd, written, result = time.monotonic(), None, None, {}, None
    deadline = started + max_seconds
    try:
        pose = package._Package(source, deadline)
        evidence, source_name, source_sha = _validate_source(pose, frames)
        code, implementation = _implementation(deadline)
        companion_code = {"implementation/" + name: code[name]
                          for name in (*package.IMPLEMENTATION_FILES, "requirements-lab.lock", "pyproject.toml")}
        companion_implementation = _bytes({"files": {name: _sha(raw) for name, raw in companion_code.items()},
                                          "versions": implementation["versions"]})
        fixed = len(pose.manifest_raw) + sum(map(len, pose.raw.values())) + sum(map(len, companion_code.values()))
        # Reserve the consumer's exact fixed inputs plus bounded reports/bindings before any decode.
        png_budget = min(MAX_BYTES - RESERVE_BYTES, package.MAX_BYTES - fixed - len(companion_implementation)
                         - package.RESERVE - MAX_REPORT_BYTES - 32 * 1024)
        if png_budget <= 0:
            raise ValueError("Pose metadata leaves no existing pixel-check PNG reservation")
        video_raw = package._read(pose.fd, source_name, MAX_VIDEO_BYTES, deadline)
        if not video_raw or _sha(video_raw) != source_sha:
            raise ValueError("Original video bytes differ from the sealed source identity")
        video_size = len(video_raw)
        plan = {"schema_version": 1, "profile": PROFILE, "source_sha256": source_sha,
            "pose_manifest_sha256": _sha(pose.manifest_raw), "frames": frames,
            **{key: evidence[key] for key in ("width", "height", "fps", "frame_count", "decoded_frames")},
            "samples": [{key: row[key] for key in ("id", "frame", "time_s", "decoder_timestamp_s")}
                        for row in evidence["frames"] if row["frame"] <= frames[-1]],
            "maximum_png_bytes": png_budget, "maximum_read_calls": frames[-1] + 1,
            "maximum_seconds": max_seconds, "cleanup_seconds": CLEANUP_SECONDS,
            "maximum_output_bytes": MAX_BYTES, "maximum_output_files": MAX_FILES,
            "maximum_temporary_video_bytes": MAX_VIDEO_BYTES, "memory_hard_limit": False,
            "decoder": {"backend": "FFMPEG", "threads": 1, "capture_options": CAPTURE_OPTIONS},
            "input_sha256": {name: _sha(raw) for name, raw in pose.raw.items()} | {source_name: source_sha},
            "implementation": implementation}
        pose.recheck()
        parent = package._directory(target.parent)
        try:
            os.mkdir(target.name, 0o700, dir_fd=parent)
            output_fd = os.open(target.name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=parent)
        finally:
            os.close(parent)
        _write(output_fd, "plan.json", _bytes(plan), written, deadline)
        with tempfile.TemporaryDirectory(prefix="weft-original-pixels-") as scratch:
            snapshot = Path(scratch) / source_name
            with snapshot.open("xb") as stream:
                os.chmod(snapshot, 0o600)
                if stream.write(video_raw) != len(video_raw):
                    raise OSError("Incomplete verified-video snapshot write")
            del video_raw
            result = _run_bounded(snapshot, target, plan, deadline - .25)
        written = _output_files(output_fd, deadline)
        if result["status"] != "complete":
            raise ValueError("Original-frame decoder failed: " + result["error"]["message"])
        if (result["work"] != {"decoder_opens": 1, "read_calls": frames[-1] + 1, "decoded_frames": frames[-1] + 1}
                or [row["frame"] for row in result["rows"]] != frames
                or set(written) != {"plan.json", *(row["file"] for row in result["rows"])}):
            raise ValueError("Decoder output coverage differs from the frozen prefix plan")
        report = {"schema_version": 1, "profile": PROFILE, "status": "pixels_extracted",
            "source_sha256": source_sha, "source_bytes": video_size, "source_artifact": source_name,
            "pose_manifest_sha256": _sha(pose.manifest_raw), "plan_sha256": _sha(written["plan.json"]),
            "output_resampling": False, "automatic_repair": False, "api_calls": 0, "inference_calls": 0, "physics_steps": 0,
            **{key: result[key] for key in ("rows", "work", "backend", "prefix_clocks")},
            "prefix_clock_columns": ["frame", "decoder_timestamp_s", "reported_next_frame"],
            "full_video_decode_verified": False, "saved_source_decoded_frames": evidence["decoded_frames"],
            "scope": "Exact original decoded RGB pixels and saved source clocks, not anatomy or physical truth. "
                     "Only the requested prefix was decoded; later source slots were not inspected.",
            "verification_scope": "Original video bytes and consumed sealed pose/model/raw-identity/clock receipts checked; "
                     "other body files and the separately receipt-bound raw pose artifact were not reopened.",
            "elapsed_seconds": time.monotonic() - started}
        report_raw = _bytes(report)
        if len(report_raw) > MAX_REPORT_BYTES:
            raise ValueError("Extraction report exceeds its metadata bound")
        _write(output_fd, "report.json", report_raw, written, deadline)
        manifest = {"schema_version": 1, "profile": PROFILE, "status": "pixels_extracted",
            "files": {name: _sha(raw) for name, raw in written.items()}}
        manifest_raw = _bytes(manifest)
        consumer = {"inputs/pose-manifest.json": pose.manifest_raw, "inputs/extraction-manifest.json": manifest_raw,
                    "inputs/extraction-report.json": report_raw}
        consumer.update({"inputs/" + name: pose.raw[name] for name in package.POSE_FILES})
        consumer.update({"images/" + row["file"]: written[row["file"]] for row in result["rows"]})
        _, _, _, binding = package._derive(consumer)
        if sum(map(len, consumer.values())) + len(_bytes(binding)) + sum(map(len, companion_code.values())) + len(companion_implementation) + package.RESERVE > package.MAX_BYTES:
            raise ValueError("Exact extraction would exceed the existing pixel-check package bound")
        pose.recheck()
        if _sha(package._read(pose.fd, source_name, MAX_VIDEO_BYTES, deadline)) != source_sha:
            raise ValueError("Original video changed during extraction")
        if _implementation(deadline) != (code, implementation):
            raise ValueError("Extraction implementation/runtime changed")
        if _output_files(output_fd, deadline) != written:
            raise ValueError("Extraction output changed before sealing")
        current = package._directory(target)
        try:
            expected, actual = os.fstat(output_fd), os.fstat(current)
            if (expected.st_dev, expected.st_ino) != (actual.st_dev, actual.st_ino):
                raise ValueError("Extraction destination identity changed before sealing")
        finally:
            os.close(current)
        _write(output_fd, "manifest.json", manifest_raw, written, deadline, terminal=True)
        return {"profile": PROFILE, "status": "pixels_extracted", "directory": str(target),
            "manifest_sha256": _sha(manifest_raw), "source_sha256": source_sha, "frames": frames,
            **result["work"], "bytes": sum(map(len, written.values())), "files": len(written)}
    except BaseException as error:
        if output_fd is not None:
            terminal_deadline = min(time.monotonic() + CLEANUP_SECONDS - WORKER_CLEANUP_SECONDS,
                                    deadline + CLEANUP_SECONDS)
            try:
                written = _output_files(output_fd, terminal_deadline)
                failure = {"schema_version": 1, "profile": PROFILE, "status": "failed",
                    "error": {"type": type(error).__name__, "message": str(error)[:1000]},
                    "work": result.get("work") if result else None,
                    "count_scope": "Completed worker counters" if result else "Unknown: native worker did not return complete counters",
                    "elapsed_seconds": time.monotonic() - started, "api_calls": 0, "inference_calls": 0, "physics_steps": 0}
                _write(output_fd, "failure.json", _bytes(failure), written, terminal_deadline, terminal=True)
                seal = {"schema_version": 1, "profile": PROFILE, "status": "failed",
                        "files": {name: _sha(raw) for name, raw in written.items()}}
                _write(output_fd, "manifest.json", _bytes(seal), written, terminal_deadline, terminal=True)
            except (OSError, ValueError, TimeoutError):
                pass
        raise
    finally:
        if output_fd is not None:
            os.close(output_fd)
        if pose is not None:
            pose.close()
