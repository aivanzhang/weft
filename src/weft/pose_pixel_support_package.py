"""Saved-frame companions: selected-byte custody, never new video or pose inference."""

from collections import Counter
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path, PurePosixPath
import platform
import stat
import time

import numpy as np

from weft import pose_pixel_support as diagnostic
from weft.shared_world_packaging import _parent_fd, validate_package_layout


PROFILE = "source-pixel-pose-support-package-v1"
EXTRACTION_PROFILE = "source-frame-pixels-v1"
LEGACY_EXTRACTION_PROFILE = "single-leg-source-pixels-v1"
MAX_BYTES = 4 * 1024 * 1024
MAX_FILES = 24
MAX_SECONDS = 5.
RESERVE = 256 * 1024
POSE_FILES = ("evidence.json", "body-source.json", "identity-audit.json", "source-clock.json", "model.json")
IMPLEMENTATION_FILES = ("pose_pixel_support.py", "pose_pixel_support_package.py", "shared_world_packaging.py")
IMPLEMENTATION_NAMES = {"implementation/" + name for name in (*IMPLEMENTATION_FILES, "requirements-lab.lock", "pyproject.toml")}


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _bytes(value):
    return diagnostic._bytes(value) + b"\n"


def _json(raw):
    return diagnostic._json(raw, MAX_BYTES)


def _check(deadline):
    if time.monotonic() >= deadline:
        raise TimeoutError("Saved pixel-support package exceeded its cooperative deadline")


def _path(value):
    path = Path(value)
    if ".." in path.parts:
        raise ValueError("Pixel-support paths must not contain '..'")
    return path.absolute()


def _directory(path):
    fd = os.open(path.anchor, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        for part in path.parts[1:]:
            child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
            os.close(fd)
            fd = child
        return fd
    except BaseException:
        os.close(fd)
        raise


def _read(fd, name, limit, deadline):
    _check(deadline)
    validate_package_layout([name])
    parent = _parent_fd(fd, name)
    try:
        handle = os.open(PurePosixPath(name).name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=parent)
    finally:
        os.close(parent)
    with os.fdopen(handle, "rb") as stream:
        before = os.fstat(stream.fileno())
        if not stat.S_ISREG(before.st_mode) or not 0 <= before.st_size <= limit:
            raise ValueError("Pixel-support member is not a bounded regular file")
        raw = stream.read(limit + 1)
        after = os.fstat(stream.fileno())
    if (len(raw) != before.st_size or len(raw) > limit or
            (before.st_size, before.st_mtime_ns, before.st_ctime_ns) !=
            (after.st_size, after.st_mtime_ns, after.st_ctime_ns)):
        raise ValueError("Pixel-support input changed during reading")
    _check(deadline)
    return raw


def _manifest(raw):
    doc = _json(raw)
    files = doc.get("files")
    if (type(doc.get("schema_version")) is not int or doc["schema_version"] != 1
            or not isinstance(files, dict) or not 1 <= len(files) <= 512 or "manifest.json" in files):
        raise ValueError("Invalid bounded source manifest")
    validate_package_layout(files)
    for value in files.values():
        diagnostic._digest(value)
    if "failure.json" in files or doc.get("status") == "failed":
        raise ValueError("A failed package cannot supply pixel-support evidence")
    return doc


class _Package:
    def __init__(self, path, deadline):
        self.path, self.deadline = _path(path), deadline
        self.fd = _directory(self.path)
        self.identity = os.fstat(self.fd)
        self.raw = {}
        self.complete_inventory = False
        try:
            self.manifest_raw = _read(self.fd, "manifest.json", 128 * 1024, deadline)
            self.manifest = _manifest(self.manifest_raw)
            self.files = self.manifest["files"]
            self._no_failure()
        except BaseException:
            self.close()
            raise

    def _no_failure(self):
        try:
            os.stat("failure.json", dir_fd=self.fd, follow_symlinks=False)
        except FileNotFoundError:
            return
        raise ValueError("Terminal failure marker blocks pixel-support evidence")

    def read(self, name):
        if name not in self.files:
            raise ValueError("Required pixel-support input is absent from its source seal: " + name)
        if name not in self.raw:
            capacity = MAX_BYTES - len(self.manifest_raw) - sum(map(len, self.raw.values()))
            raw = _read(self.fd, name, capacity, self.deadline)
            if _sha(raw) != self.files[name]:
                raise ValueError("Pixel-support source member differs from its sealed hash")
            self.raw[name] = raw
        return self.raw[name]

    def recheck(self):
        fd = _directory(self.path)
        try:
            current = os.fstat(fd)
            if (current.st_dev, current.st_ino) != (self.identity.st_dev, self.identity.st_ino):
                raise ValueError("Pixel-support source directory changed")
        finally:
            os.close(fd)
        if _read(self.fd, "manifest.json", 128 * 1024, self.deadline) != self.manifest_raw:
            raise ValueError("Pixel-support source manifest changed")
        for name, raw in self.raw.items():
            if _read(self.fd, name, len(raw), self.deadline) != raw:
                raise ValueError("Pixel-support consumed source member changed")
        self._no_failure()
        if self.complete_inventory and _inventory(self.fd, self.deadline) != set(self.files) | {"manifest.json"}:
            raise ValueError("Pixel-support companion inventory changed after loading")

    def close(self):
        os.close(self.fd)


def _derive(files):
    parent = _manifest(files["inputs/pose-manifest.json"])
    extraction_parent = _manifest(files["inputs/extraction-manifest.json"])
    if parent.get("profile") != "exercise-body-source-v1" or parent.get("status") != "construction_only":
        raise ValueError("Expected a sealed original construction-body source")
    for name in POSE_FILES:
        if parent["files"].get(name) != _sha(files["inputs/" + name]):
            raise ValueError("Archived pose member differs from its original parent seal")
    if extraction_parent["files"].get("report.json") != _sha(files["inputs/extraction-report.json"]):
        raise ValueError("Archived extraction report differs from its original seal")
    evidence = _json(files["inputs/evidence.json"])
    marker, identity, clock, model = (_json(files["inputs/" + name]) for name in POSE_FILES[1:])
    source = diagnostic._digest(evidence.get("source_sha256"))
    if (not isinstance(marker.get("pose_options"), dict) or not isinstance(marker.get("artifacts"), dict)
            or evidence.get("profile") != "exercise-pose-v1" or marker.get("profile") != "exercise-body-source-v1"
            or marker.get("status") != "construction_only" or marker.get("simulation_ready") is not False
            or marker.get("source_sha256") != source or marker.get("pose_options", {}).get("pose_identity") != "raw"
            or identity.get("mode") != "raw" or identity.get("source_sha256") != source
            or identity.get("repaired_evidence_ids") != [] or identity.get("raw_artifact") != "pose-landmarks.json"
            or identity.get("raw_artifact_sha256") != parent["files"].get("pose-landmarks.json")
            or model != evidence.get("model") or clock.get("source_sha256") != source):
        raise ValueError("Original pose/model/raw-identity source receipts disagree")
    diagnostic._digest(identity.get("raw_artifact_sha256"))
    source_artifact = marker.get("source_artifact")
    if (source_artifact not in {"source.webm", "source.mp4", "source.m4v", "source.mov", "source.mkv", "source.avi"}
            or parent["files"].get(source_artifact) != source):
        raise ValueError("Body marker's original video identity differs from its parent manifest")
    if any(marker.get("artifacts", {}).get(name) != parent["files"][name]
           for name in POSE_FILES if name != "body-source.json"):
        raise ValueError("Body marker no longer binds its consumed original source receipts")
    observations = evidence.get("frames")
    if (not isinstance(observations, list) or not 1 <= len(observations) <= 900
            or any(not isinstance(row, dict) for row in observations)):
        raise ValueError("Original source pose inventory exceeds its bound")
    samples = [{key: row.get(key) for key in ("frame", "time_s", "decoder_timestamp_s")} for row in observations]
    if clock.get("samples") != samples or clock.get("decoded_frames") != evidence.get("decoded_frames"):
        raise ValueError("Native and nominal clock inventory differs from original pose evidence")
    original = {row.get("frame"): row for row in observations}
    if len(original) != len(observations):
        raise ValueError("Original source pose slots are duplicated")
    extraction = _json(files["inputs/extraction-report.json"])
    profile = extraction.get("profile")
    if (type(extraction.get("schema_version")) is not int or extraction["schema_version"] != 1
            or profile not in {EXTRACTION_PROFILE, LEGACY_EXTRACTION_PROFILE}
            or extraction_parent.get("profile") != profile or extraction_parent.get("status") != "pixels_extracted"
            or extraction.get("status") != "pixels_extracted" or extraction.get("source_sha256") != source
            or extraction.get("output_resampling") is not False or extraction.get("automatic_repair") is not False
            or any(type(extraction.get(key)) is not int or extraction[key] != 0
                   for key in ("api_calls", "inference_calls", "physics_steps"))):
        raise ValueError("Unsupported original-pixel extraction attestation")
    rows = extraction.get("rows")
    if not isinstance(rows, list) or not 2 <= len(rows) <= 4:
        raise ValueError("Extraction must declare two to four exact original images")
    entries, image_names, previous = [], {}, -1
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("Extraction image declarations must be records")
        frame, name = row.get("frame"), row.get("file")
        if type(frame) is not int or frame <= previous or frame not in original or not isinstance(name, str):
            raise ValueError("Extraction frames must be ordered unique original pose slots")
        validate_package_layout([name])
        if PurePosixPath(name).suffix != ".png":
            raise ValueError("Extraction images must be exact PNG artifacts")
        saved = original[frame]
        time_key = "nominal_frame_over_fps_s" if profile == LEGACY_EXTRACTION_PROFILE else "time_s"
        diagnostic._finite(row.get(time_key), 0., 60.)
        if row.get("decoder_timestamp_s") is not None:
            diagnostic._finite(row["decoder_timestamp_s"], 0., 3600.)
        if (type(row.get("width")) is not int or type(row.get("height")) is not int
                or row.get("source_sha256") != source or row.get(time_key) != saved.get("time_s")
                or row.get("decoder_timestamp_s") != saved.get("decoder_timestamp_s")
                or row.get("width") != evidence.get("width") or row.get("height") != evidence.get("height")):
            raise ValueError("Extraction source, clocks or image dimensions differ from original evidence")
        destination = f"images/frame-{frame:04d}.png"
        digest = diagnostic._digest(row.get("sha256"))
        diagnostic._digest(row.get("decoded_rgb_sha256"))
        if extraction_parent["files"].get(name) != digest or _sha(files[destination]) != digest:
            raise ValueError("Original PNG differs from its extraction receipt and manifest")
        image_names[frame] = destination
        entries.append({key: saved.get(key) for key in ("frame", "id", "time_s", "decoder_timestamp_s")}
                       | {"image_sha256": digest})
        previous = frame
    binding = {"schema_version": 1, "profile": diagnostic.BINDING_PROFILE, "source_sha256": source,
        "source_manifest_sha256": _sha(files["inputs/pose-manifest.json"]),
        "evidence_sha256": _sha(files["inputs/evidence.json"]), "frames": entries}
    return evidence, extraction, image_names, binding


def _saved_match(match):
    if not isinstance(match, dict) or match.get("reason") is not None:
        raise ValueError("Qualified pixel support lacks bidirectional patch support")
    diagnostic._finite(match.get("template_std"), diagnostic.CONTRACT["minimum_patch_std"], 256.)
    score = diagnostic._finite(match.get("correlation"), diagnostic.CONTRACT["minimum_correlation"], 1.)
    competitor = diagnostic._finite(match.get("competitor_correlation"), -1., 1.)
    margin = diagnostic._finite(match.get("uniqueness_margin"), diagnostic.CONTRACT["minimum_unique_margin"], 2.)
    if not math.isclose(margin, score - competitor, rel_tol=0., abs_tol=1e-12):
        raise ValueError("Saved patch uniqueness arithmetic changed")
    offset = match.get("offset_px")
    if (not isinstance(offset, list) or len(offset) != 2
            or any(type(v) is not int or abs(v) >= diagnostic.CONTRACT["search_radius_px"] for v in offset)):
        raise ValueError("Qualified patch offsets exceed the original interior search grid")
    return np.array(offset)


def _classification_check(row, first, second, evidence):
    stamps = row["native_time_s"]
    expected = None
    if any(stamp is None for stamp in stamps):
        expected = "unknown_native_clock"
    elif stamps[1] <= stamps[0] or stamps[1] - stamps[0] > diagnostic.CONTRACT["maximum_pair_seconds"]:
        expected = "native_clock_gap"
    a, reason_a, raw_a = diagnostic._point(first, row["joint_index"], evidence["width"], evidence["height"])
    b, reason_b, raw_b = diagnostic._point(second, row["joint_index"], evidence["width"], evidence["height"])
    expected = expected or reason_a or reason_b
    if expected is not None:
        if row["status"] != "unavailable" or row["reason"] != expected:
            raise ValueError("Saved pixel classification bypasses original confidence or native-clock availability")
        return
    if row.get("original_landmarks") != [raw_a, raw_b]:
        raise ValueError("Saved pixel-support landmark fields changed")
    forward, backward = row.get("forward"), row.get("backward")
    if not isinstance(forward, dict) or "reason" not in forward:
        raise ValueError("Processed pixel check lacks its recorded forward attempt")
    if forward["reason"] is not None:
        if row["status"] != "unavailable" or row["reason"] != forward["reason"]:
            raise ValueError("Failed forward patch support cannot classify motion")
        return
    offset = _saved_match(forward)
    if not isinstance(backward, dict) or "reason" not in backward:
        raise ValueError("Forward-supported pixel check lacks its recorded reverse attempt")
    if backward["reason"] is not None:
        if row["status"] != "unavailable" or row["reason"] != "reverse_" + backward["reason"]:
            raise ValueError("Failed reverse patch support cannot classify motion")
        return
    reverse = _saved_match(backward)
    correspondence = a + offset
    returned = correspondence + reverse
    roundtrip = float(np.linalg.norm(returned - a))
    if row.get("roundtrip_error_px") != roundtrip:
        raise ValueError("Saved round-trip arithmetic differs from exact offsets and original seed")
    if roundtrip > diagnostic.CONTRACT["maximum_roundtrip_px"]:
        if row["status"] != "unavailable" or row["reason"] != "roundtrip_failed":
            raise ValueError("Nonreturning patch correspondence cannot classify motion")
        return
    residual = float(np.linalg.norm(correspondence - b))
    if (row.get("model_displacement_px") != (b - a).tolist() or row.get("patch_displacement_px") != offset.tolist()
            or row.get("model_patch_residual_px") != residual):
        raise ValueError("Saved residual arithmetic differs from original source seeds and patch offsets")
    status, reason = "unavailable", "residual_uncertain"
    if residual <= diagnostic.CONTRACT["not_contradicted_residual_px"]:
        status, reason = "not_contradicted", "local_patch_motion_agrees"
    elif residual >= diagnostic.CONTRACT["contradicted_residual_px"]:
        status, reason = "contradicted", "local_patch_motion_disagrees"
    if (row["status"], row["reason"]) != (status, reason):
        raise ValueError("Saved status/reason differs from the original fixed residual rules")


def _result_check(result, evidence, extraction, binding, binding_raw):
    names = [row["frame"] for row in binding["frames"]]
    original = {row["frame"]: row for row in evidence["frames"]}
    rows = result.get("rows")
    slots = [(a, b, j, label) for a, b in zip(names, names[1:]) for j, label in diagnostic.JOINTS]
    if (result.get("schema_version") != 1 or result.get("profile") != diagnostic.PROFILE
            or result.get("status") not in {"complete", "partial_deadline"} or result.get("contract") != dict(diagnostic.CONTRACT)
            or any(result.get(key) != binding[key] for key in ("source_sha256", "source_manifest_sha256", "evidence_sha256"))
            or result.get("bindings_sha256") != _sha(binding_raw) or result.get("image_bindings") != binding["frames"]
            or result.get("sampled_image_frames") != names or result.get("pair_count") != len(names) - 1
            or result.get("slot_count") != len(slots) or not isinstance(rows, list) or len(rows) != len(slots)
            or result.get("source_pose_slots") != len(original) or result.get("source_decoded_frames") != evidence["decoded_frames"]
            or result.get("source_provenance_status") != "caller_bound_not_independently_verified"):
        raise ValueError("Saved pixel-support result differs from its exact input/slot contract")
    processed = result.get("processed_slots")
    if type(processed) is not int or not 0 <= processed <= len(slots) or (result["status"] == "complete" and processed != len(slots)):
        raise ValueError("Invalid saved pixel-support completion count")
    hashes = {n: _sha(diagnostic._bytes(original[n])) for n in names}
    for number, (row, (a, b, joint, name)) in enumerate(zip(rows, slots, strict=True)):
        if (row.get("frames") != [a, b] or row.get("joint_index") != joint or row.get("joint") != name
                or row.get("observation_ids") != [original[n]["id"] for n in (a, b)]
                or row.get("original_observation_sha256") != [hashes[n] for n in (a, b)]
                or row.get("time_s") != [original[n]["time_s"] for n in (a, b)]
                or row.get("native_time_s") != [original[n]["decoder_timestamp_s"] for n in (a, b)]
                or row.get("status") not in {"contradicted", "not_contradicted", "unavailable"}
                or not isinstance(row.get("reason"), str) or not 1 <= len(row["reason"]) <= 128):
            raise ValueError("Saved pixel-support pair/joint identity or classification changed")
        if number >= processed and (row["status"] != "unavailable" or row["reason"] != "deadline_unprocessed"):
            raise ValueError("Unprocessed pixel-support slots cannot be classified")
        if number >= processed and any(key in row for key in ("forward", "backward", "roundtrip_error_px",
                "model_displacement_px", "patch_displacement_px", "model_patch_residual_px")):
            raise ValueError("Unprocessed pixel-support slots cannot retain completed-match claims")
        if number < processed and row["reason"] == "deadline_unprocessed":
            raise ValueError("Processed pixel-support prefix is inconsistent")
        if number < processed:
            _classification_check(row, original[a], original[b], evidence)
    if result.get("status_counts") != dict(Counter(row["status"] for row in rows)):
        raise ValueError("Saved pixel-support totals omit or duplicate original slots")
    decoded = result.get("decoded_rgb_sha256")
    expected_rgb = {str(row["frame"]): row["decoded_rgb_sha256"] for row in extraction["rows"]}
    if not isinstance(decoded, dict) or any(expected_rgb.get(name) != digest for name, digest in decoded.items()):
        raise ValueError("Decoded PNG bytes differ from the extraction's saved RGB attestation")
    if any(str(frame) not in decoded for row in rows if row["status"] != "unavailable" for frame in row["frames"]):
        raise ValueError("Qualified patch checks require both decoded-frame RGB attestations")
    if result.get("work") != {"model_calls": 0, "video_decodes": 0, "physics_steps": 0,
                              "png_decodes": len(decoded), "max_patch_matches": 2 * len(rows)}:
        raise ValueError("Saved diagnostic work counts differ from the bounded protocol")


def _inventory(fd, deadline, *, prefix="", depth=0, counter=None):
    if depth > 3:
        raise ValueError("Pixel-support package nesting exceeds its fixed layout")
    found = set()
    counter = [0] if counter is None else counter
    with os.scandir(fd) as entries:
        for entry in entries:
            _check(deadline)
            counter[0] += 1
            if counter[0] > 2 * MAX_FILES:
                raise ValueError("Pixel-support directory entries exceed their bound")
            name = prefix + entry.name
            info = entry.stat(follow_symlinks=False)
            if stat.S_ISDIR(info.st_mode):
                child = os.open(entry.name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
                try:
                    contents = _inventory(child, deadline, prefix=name + "/", depth=depth + 1, counter=counter)
                    if not contents:
                        raise ValueError("Pixel-support output contains an unlisted empty directory")
                    found.update(contents)
                finally:
                    os.close(child)
            elif stat.S_ISREG(info.st_mode):
                found.add(name)
            else:
                raise ValueError("Pixel-support output contains an unsafe member")
            if len(found) > MAX_FILES:
                raise ValueError("Pixel-support output exceeds its fixed file bound")
    return found


def _write(fd, name, raw, written, deadline):
    _check(deadline)
    if len(written) + 1 > MAX_FILES or sum(len(value) for value in written.values()) + len(raw) > MAX_BYTES:
        raise ValueError("Pixel-support output exceeds its predeclared bound")
    parent = _parent_fd(fd, name, create=True)
    try:
        handle = os.open(PurePosixPath(name).name, os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600, dir_fd=parent)
        with os.fdopen(handle, "w+b") as stream:
            stream.write(raw)
            stream.flush()
            stream.seek(0)
            if stream.read(len(raw) + 1) != raw:
                raise ValueError("Pixel-support write readback changed")
        written[name] = raw
    finally:
        os.close(parent)


def prepare_pose_pixel_support(pose_run, extraction_run, out, *, max_seconds=5.):
    """Seal original saved inputs and one bounded patch check; no source video is opened."""
    diagnostic._finite(max_seconds, .1, MAX_SECONDS)
    started = time.monotonic()
    deadline = started + max_seconds
    paths = list(map(_path, (pose_run, extraction_run, out)))
    if any(paths[2].is_relative_to(p) or p.is_relative_to(paths[2]) for p in paths[:2]):
        raise ValueError("Pixel-support output must be new and outside both input packages")
    pose, pixels, output_fd, written = None, None, None, {}
    try:
        pose = _Package(paths[0], deadline)
        pixels = _Package(paths[1], deadline)
        files = {"inputs/pose-manifest.json": pose.manifest_raw, "inputs/extraction-manifest.json": pixels.manifest_raw,
                 "inputs/extraction-report.json": pixels.read("report.json")}
        files.update({"inputs/" + name: pose.read(name) for name in POSE_FILES})
        extraction = _json(files["inputs/extraction-report.json"])
        declarations = extraction.get("rows")
        if not isinstance(declarations, list) or not 2 <= len(declarations) <= 4:
            raise ValueError("Extraction requires two to four source-bound PNGs")
        for row in declarations:
            if not isinstance(row, dict) or type(row.get("frame")) is not int or not 0 <= row["frame"] < 1800:
                raise ValueError("Invalid extraction frame declaration")
            files[f"images/frame-{row['frame']:04d}.png"] = pixels.read(row.get("file"))
        evidence, extraction, image_names, binding = _derive(files)
        files["bindings.json"] = _bytes(binding)
        code_paths = {"implementation/" + name: Path(__file__).with_name(name) for name in IMPLEMENTATION_FILES}
        root = Path(__file__).parents[2]
        code_paths.update({"implementation/" + name: root / name for name in ("requirements-lab.lock", "pyproject.toml")})
        for name, path in code_paths.items():
            fd = _directory(path.parent)
            try:
                files[name] = _read(fd, path.name, 256 * 1024, deadline)
            finally:
                os.close(fd)
        implementation = {"files": {name: _sha(files[name]) for name in code_paths},
            "versions": {"python": platform.python_version(), **{name: importlib.metadata.version(name)
                         for name in ("numpy", "Pillow", "opencv-contrib-python")}}}
        files["implementation.json"] = _bytes(implementation)
        if len(files) + 3 > MAX_FILES or sum(map(len, files.values())) + RESERVE > MAX_BYTES:
            raise ValueError("Exact inputs leave insufficient pixel-support output/failure reservation")
        pose.recheck(); pixels.recheck()
        parent = _directory(paths[2].parent)
        try:
            os.mkdir(paths[2].name, 0o700, dir_fd=parent)
            output_fd = os.open(paths[2].name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=parent)
        finally:
            os.close(parent)
        for name, value in files.items():
            _write(output_fd, name, value, written, deadline)
        remaining = min(diagnostic.CONTRACT["maximum_seconds"], deadline - time.monotonic() - .25)
        if remaining < .01:
            raise TimeoutError("Insufficient reserved time for the saved pixel check")
        result = diagnostic.diagnose_pose_pixel_support(files["inputs/evidence.json"],
            {frame: files[name] for frame, name in image_names.items()}, files["bindings.json"], max_seconds=remaining)
        _result_check(result, evidence, extraction, binding, files["bindings.json"])
        result["package_provenance"] = {"profile": PROFILE, "extraction_profile": extraction["profile"],
            "extraction_adapter": "legacy_receipt_compatibility" if extraction["profile"] == LEGACY_EXTRACTION_PROFILE else "native_schema",
            "extraction_manifest_sha256": _sha(pixels.manifest_raw), "implementation_sha256": _sha(files["implementation.json"]),
            "scope": "Exact consumed manifest members, source/model/raw-identity/clock receipts and PNG hashes checked. Other source members and extraction truth are not independently verified; no video re-decode.",
            "raw_pose_scope": "Original raw-pose hash is receipt-bound; the separate raw-pose artifact is not reopened or copied."}
        result_raw = _bytes(result)
        if len(result_raw) > diagnostic.CONTRACT["maximum_report_bytes"]:
            raise ValueError("Saved pixel-support report exceeds its fixed metadata bound")
        _write(output_fd, "report.json", result_raw, written, deadline)
        pose.recheck(); pixels.recheck()
        for name, path in code_paths.items():
            fd = _directory(path.parent)
            try:
                if _read(fd, path.name, len(files[name]), deadline) != files[name]:
                    raise ValueError("Pixel-support implementation changed during the check")
            finally:
                os.close(fd)
        current_fd = _directory(paths[2])
        try:
            before, after = os.fstat(output_fd), os.fstat(current_fd)
            if (before.st_dev, before.st_ino) != (after.st_dev, after.st_ino):
                raise ValueError("Pixel-support output directory changed")
        finally:
            os.close(current_fd)
        if _inventory(output_fd, deadline) != set(written):
            raise ValueError("Pixel-support output inventory changed")
        for name, value in written.items():
            if _read(output_fd, name, len(value), deadline) != value:
                raise ValueError("Pixel-support output changed before sealing")
        manifest = _bytes({"schema_version": 1, "profile": PROFILE, "status": result["status"],
                           "terminal_rule": "No failure.json may exist", "files": {name: _sha(value) for name, value in written.items()}})
        total = sum(map(len, written.values())) + len(manifest)
        receipt = {"profile": PROFILE, "status": result["status"], "directory": str(paths[2]),
            "manifest_sha256": _sha(manifest), "source_sha256": result["source_sha256"],
            "evidence_sha256": result["evidence_sha256"], "pair_count": result["pair_count"],
            "slot_count": result["slot_count"], "status_counts": result["status_counts"],
            "files": len(written) + 1, "bytes": total, "seconds": time.monotonic() - started}
        _write(output_fd, "manifest.json", manifest, written, deadline)
        return receipt
    except Exception as error:
        if output_fd is not None:
            try:
                failure = _bytes({"profile": PROFILE, "status": "failed", "error": type(error).__name__,
                    "message": str(error)[:1000], "terminal_rule": "This failure marker overrides any manifest", "seconds": time.monotonic() - started})
                _write(output_fd, "failure.json", failure, written, time.monotonic() + .05)
            except (OSError, ValueError, TimeoutError):
                pass
        raise
    finally:
        for package in (pose, pixels):
            if package is not None:
                package.close()
        if output_fd is not None:
            os.close(output_fd)


def load_pose_pixel_support(directory, *, evidence_document, source_manifest_raw, deadline):
    """Read and bind a saved companion without repeating image matching or PNG decoding."""
    package = _Package(directory, deadline)
    try:
        if package.manifest.get("profile") != PROFILE or package.manifest.get("status") not in {"complete", "partial_deadline"}:
            raise ValueError("Expected a complete or explicit partial pixel-support companion")
        if len(package.files) + 1 > MAX_FILES or _inventory(package.fd, deadline) != set(package.files) | {"manifest.json"}:
            raise ValueError("Pixel-support companion inventory differs from its seal")
        package.complete_inventory = True
        for name in package.files:
            package.read(name)
        if (package.raw.get("inputs/evidence.json") != evidence_document
                or package.raw.get("inputs/pose-manifest.json") != source_manifest_raw):
            raise ValueError("Pixel-support companion belongs to different original pose evidence/parent")
        evidence, extraction, image_names, binding = _derive(package.raw)
        if _json(package.raw["bindings.json"]) != binding:
            raise ValueError("Saved pixel bindings differ from preserved extraction receipts")
        if len(package.raw["report.json"]) > diagnostic.CONTRACT["maximum_report_bytes"]:
            raise ValueError("Saved pixel-support report exceeds its fixed metadata bound")
        result = _json(package.raw["report.json"])
        _result_check(result, evidence, extraction, binding, package.raw["bindings.json"])
        if result["status"] != package.manifest["status"]:
            raise ValueError("Pixel-support companion terminal statuses disagree")
        provenance = result.get("package_provenance", {})
        if (provenance.get("profile") != PROFILE or provenance.get("extraction_profile") != extraction["profile"]
                or provenance.get("extraction_manifest_sha256") != _sha(package.raw["inputs/extraction-manifest.json"])
                or provenance.get("implementation_sha256") != _sha(package.raw["implementation.json"])):
            raise ValueError("Saved companion provenance differs from its preserved receipts")
        implementation = _json(package.raw["implementation.json"])
        entries = implementation.get("files")
        if (not isinstance(entries, dict) or set(entries) != IMPLEMENTATION_NAMES
                or any(package.files.get(name) != diagnostic._digest(digest) for name, digest in entries.items())):
            raise ValueError("Saved implementation snapshot differs from its companion seal")
        package.summary = {key: result[key] for key in ("profile", "status", "source_sha256", "evidence_sha256",
            "source_manifest_sha256", "sampled_image_frames", "pair_count", "slot_count", "processed_slots", "status_counts", "scope", "limitations", "rows", "package_provenance")}
        package.summary.update(companion_manifest_sha256=_sha(package.manifest_raw),
            verification_scope="Saved-only consumed-input bindings, slot counts and classification checks; patch matching is not rerun.",
            verifier_sha256=_sha(Path(__file__).read_bytes()))
        package.copy_names = ["report.json", "bindings.json", "implementation.json",
            "inputs/pose-manifest.json", "inputs/extraction-manifest.json", "inputs/extraction-report.json", *image_names.values()]
        package.recheck()
        return package
    except BaseException:
        package.close()
        raise
