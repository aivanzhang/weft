"""Bounded saved-body metadata custody; no decode, inference or physics admission."""

from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import stat
import time


PROFILE = "whole-body-reference-inputs-v1"
BODY_PROFILE = "exercise-body-source-v1"
MAX_JSON_BYTES = 8 * 1024 * 1024
MAX_MANIFEST_BYTES = 128 * 1024
MAX_INPUT_BYTES = 32 * 1024 * 1024
MAX_MEMBERS = 512
_SHA = re.compile(r"[0-9a-f]{64}")
_POSE = {"evidence.json", "pose-landmarks.json", "identity-audit.json", "model.json",
         "source.json", "run-config.json"}
_REQUIRED = _POSE | {"human.json", "observations.json", "source-clock.json", "implementation.json"}
_CONFLICTS = {"failure.json", "scene.json", "task.json", "exercise-task.json", "construction-seed.json",
              "geometry-proposal.json", "scene-observations.json", "lab-config.json", "policy.json"}
_REJECTED = {"failed", "error", "rejected", "cancelled", "canceled", "unsupported"}
# Reviewed decoder version checks every frame's shape before retaining normalized image landmarks.
_DECODERS = {"7a40f269189246ea07dddb416610d6419b78092824960c1ece6e4255810de3d3":
             "opencv-fixed-resolution-complete-exercise-decode-v1"}
_PRODUCER = "implementation-source/exercise_vision.py"
_SOURCE_PRODUCERS = (_PRODUCER, "implementation-src--weft--exercise_vision.py")


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _bytes(value):
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def _canonical(value):
    return _sha(json.dumps(value, sort_keys=True, allow_nan=False).encode())


def _check(deadline):
    _require(type(deadline) in (int, float) and math.isfinite(deadline), "A finite monotonic deadline is required")
    if time.monotonic() >= deadline:
        raise TimeoutError("Saved body input deadline exceeded")


def _json(raw):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            _require(key not in result, "Duplicate JSON key")
            result[key] = value
        return result

    def invalid(value):
        raise ValueError("Non-finite JSON value")

    try:
        result = json.loads(raw, object_pairs_hook=unique, parse_constant=invalid)
    except (UnicodeError, json.JSONDecodeError, RecursionError) as error:
        raise ValueError("Saved inputs require strict bounded JSON") from error
    _require(isinstance(result, dict), "Saved input must be a JSON object")
    pending, count = [(result, 0)], 0
    while pending:
        value, depth = pending.pop()
        count += 1
        _require(depth <= 64 and count <= 1_000_000, "Saved JSON exceeds structure bounds")
        if isinstance(value, dict):
            pending.extend((child, depth + 1) for child in value.values())
        elif isinstance(value, list):
            pending.extend((child, depth + 1) for child in value)
        elif isinstance(value, float):
            _require(math.isfinite(value), "Non-finite JSON value")
    return result


def _name(value):
    _require(isinstance(value, str) and 1 <= len(value) <= 1024 and "\\" not in value and ":" not in value
             and not any(ord(char) < 32 for char in value) and not PurePosixPath(value).is_absolute()
             and not any(part in ("", ".", "..") for part in value.split("/")), "Unsafe package member path")
    return value


def _manifest(raw):
    value = _json(raw)
    files = value.get("files")
    _require(isinstance(files, dict) and 1 <= len(files) <= MAX_MEMBERS, "Manifest member bound exceeded")
    for name, digest in files.items():
        _name(name)
        _require(isinstance(digest, str) and _SHA.fullmatch(digest), "Manifest requires exact SHA256 identities")
        _require(name != "manifest.json" and not name.startswith("manifest.json/")
                 and not any(parent.as_posix() in files for parent in PurePosixPath(name).parents),
                 "Manifest file/directory collision")
    _require("failure.json" not in files and value.get("status") not in _REJECTED, "Failed source manifest")
    return value, files


def _root_path(source):
    path = Path(source)
    _require(".." not in path.parts, "Saved input paths must not contain parent traversal")
    return path.absolute()


def _root_fd(root, expected=None):
    fd = os.open("/", os.O_RDONLY | os.O_DIRECTORY)
    try:
        for part in root.parts[1:]:
            child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
            os.close(fd)
            fd = child
        info = os.fstat(fd)
        identity = (info.st_dev, info.st_ino)
        _require(expected is None or identity == expected, "Saved body root directory changed")
        try:
            os.stat("failure.json", dir_fd=fd, follow_symlinks=False)
        except FileNotFoundError:
            pass
        else:
            raise ValueError("Saved body contains a failure marker")
        return fd, identity
    except BaseException:
        os.close(fd)
        raise


class _Reader:
    def __init__(self, root, deadline, identity=None):
        self.root, self.deadline = root, deadline
        self.pins, self.total = {}, 0
        _check(deadline)
        fd, self.identity = _root_fd(root, identity)
        os.close(fd)

    def read(self, name, expected=None, maximum=MAX_JSON_BYTES):
        _check(self.deadline)
        _name(name)
        _require(name not in self.pins, "A selected source member must be read only once")
        limit = min(maximum, MAX_INPUT_BYTES - self.total)
        _require(limit > 0, "Selected input aggregate byte bound exceeded")
        root_fd, _ = _root_fd(self.root, self.identity)
        parent = os.dup(root_fd)
        try:
            for part in PurePosixPath(name).parts[:-1]:
                child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=parent)
                os.close(parent)
                parent = child
            fd = os.open(PurePosixPath(name).name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=parent)
            with os.fdopen(fd, "rb") as stream:
                before = os.fstat(stream.fileno())
                _require(stat.S_ISREG(before.st_mode) and 0 <= before.st_size <= limit,
                         "Selected input is not a bounded regular file")
                parts, size = [], 0
                while block := stream.read(min(1 << 20, limit - size + 1)):
                    _check(self.deadline)
                    parts.append(block)
                    size += len(block)
                    _require(size <= limit, "Selected input grew beyond its byte bound")
                after = os.fstat(stream.fileno())
                _require((before.st_size, before.st_mtime_ns, before.st_ctime_ns) ==
                         (after.st_size, after.st_mtime_ns, after.st_ctime_ns), "Selected input changed during read")
                raw = b"".join(parts)
        finally:
            os.close(parent)
            os.close(root_fd)
        digest = _sha(raw)
        _require(expected is None or digest == expected, "Selected input differs from its sealed SHA256")
        self.pins[name] = digest
        self.total += len(raw)
        _check(self.deadline)
        return raw


def _pose_bindings(documents, raw_files, marker):
    evidence, human = documents["evidence.json"], documents["human.json"]
    raw, audit, clock = (documents[name] for name in ("pose-landmarks.json", "identity-audit.json", "source-clock.json"))
    model, config = documents["model.json"], documents["run-config.json"]
    source = marker["source_sha256"]
    options = marker.get("pose_options")
    _require(isinstance(options, dict) and set(options) == {"pose_mode", "pose_identity"}
             and options["pose_mode"] in {"video", "image"} and options["pose_identity"] in {"raw", "temporal"},
             "Unsupported pose options")
    _require(config.get("profile") == "exercise"
             and {key: config.get(key, "video" if key == "pose_mode" else "raw") for key in options} == options,
             "Original pose configuration differs from saved construction options")
    _require(evidence.get("schema_version") == 1 and evidence.get("profile") == "exercise-pose-v1"
             and evidence.get("source_sha256") == source and human.get("source_sha256") == source
             and human.get("schema_version") == 3 and human.get("profile") == "exercise-v3"
             and human.get("dt") == 1 / 30 and "camera_reference" not in human
             and not any("refinement" in key or "alignment" in key for key in human), "Unsupported original body/source identity")
    _require(evidence.get("model") == model == raw.get("model")
             and isinstance(model.get("sha256"), str) and _SHA.fullmatch(model["sha256"]), "Pose model identities differ")
    identity = evidence.get("pose_identity", {})
    raw_hash = _sha(raw_files["pose-landmarks.json"])
    _require(identity.get("mode") == options["pose_identity"] == audit.get("mode")
             and audit.get("source_sha256") == source and audit.get("raw_artifact") == "pose-landmarks.json"
             and identity.get("audit_artifact") == "identity-audit.json"
             and identity.get("raw_artifact_sha256") == raw_hash == audit.get("raw_artifact_sha256"),
             "Raw pose identity bindings differ")
    width, height, fps, count, decoded, stride = (evidence.get(key) for key in
        ("width", "height", "fps", "frame_count", "decoded_frames", "inference_stride"))
    _require(type(width) is int and 64 <= width <= 4096 and type(height) is int and 64 <= height <= 2160
             and type(fps) in (int, float) and 5 <= fps <= 240
             and type(count) is int and 10 <= count <= 1800 and count / fps <= 60
             and type(decoded) is int and 10 <= decoded <= 1800 and decoded / fps <= 60
             and type(stride) is int and stride == max(1, math.ceil(fps / 15), math.ceil(count / 900)),
             "Saved decoder dimensions/counts exceed the reviewed contract")
    if abs(decoded - count) > 1:
        container = evidence.get("container_frame_audit", {})
        _require(container.get("encoded_video_frames") == decoded == container.get("decoded_frames")
                 and container.get("matches") is True, "Incomplete decoder coverage")
    frames, originals = evidence.get("frames"), raw.get("frames")
    _require(isinstance(frames, list) and isinstance(originals, list) and 1 <= len(frames) <= 900
             and len(frames) == len(originals) and all(isinstance(row, dict) for row in frames + originals)
             and [row.get("frame") for row in frames] == list(range(0, decoded, stride)), "Incomplete sampled pose slots")
    clocks, repaired = [], []
    for frame, original in zip(frames, originals, strict=True):
        index, stamp, native = frame.get("frame"), frame.get("time_s"), frame.get("decoder_timestamp_s")
        _require(type(index) is int and type(stamp) in (int, float) and abs(stamp - index / fps) <= 1e-7
                 and frame.get("id") == f"{source}:frame:{index}"
                 and all(key in frame and frame[key] == original.get(key) for key in
                         ("id", "frame", "time_s", "decoder_timestamp_s"))
                 and (native is None or type(native) in (int, float) and 0 <= native <= 60), "Raw pose clocks/identities differ")
        _require("image_refinement" not in frame and not evidence.get("pose_reinference"),
                 "Recovered crop evidence needs a separate supported provenance adapter")
        if "identity_raw" in frame:
            _require(options["pose_identity"] == "temporal" and frame["identity_raw"] == original,
                     "Temporal identity must preserve the exact raw record")
            repaired.append(frame["id"])
        else:
            _require(frame == original, "Uncorrected pose differs from its raw record")
        clocks.append({key: frame[key] for key in ("frame", "time_s", "decoder_timestamp_s")})
    _require(audit.get("repaired_evidence_ids") == repaired and identity.get("repaired_frames") == len(repaired)
             and evidence.get("quality", {}).get("passes") is True and not evidence.get("quality", {}).get("errors"),
             "Saved pose quality or identity receipt conflicts")
    _require(clock.get("source_sha256") == source and clock.get("fps") == fps
             and clock.get("metadata_frames") == count and clock.get("decoded_frames") == decoded
             and clock.get("samples") == clocks and clock.get("nominal_basis") == "frame_index_divided_by_fps",
             "Saved decoder clock differs from retained pose slots")
    usable = [frame for frame in frames if frame.get("status") in {"tracked", "tracked_partial"}]
    references, metadata = human.get("reference"), human.get("reference_evidence")
    _require(isinstance(references, list) and isinstance(metadata, list)
             and all(isinstance(row, dict) for row in references + metadata)
             and 3 <= len(references) == len(metadata) == len(usable) <= 900
             and [row.get("time_s") for row in references] == [row["time_s"] for row in usable]
             and [row.get("time_s") for row in metadata] == [row["time_s"] for row in usable]
             and [row.get("evidence_id") for row in metadata] == [row["id"] for row in usable],
             "Body reference does not cover the original usable pose slots")


def prepare_reference_inputs(source, *, expected_manifest_sha256=None, deadline):
    """Snapshot selected sealed metadata; simulator admission and media custody remain separate."""
    _check(deadline)
    _require(expected_manifest_sha256 is None or isinstance(expected_manifest_sha256, str)
             and _SHA.fullmatch(expected_manifest_sha256), "Expected manifest identity must be a SHA256")
    reader = _Reader(_root_path(source), deadline)
    manifest_raw = reader.read("manifest.json", expected_manifest_sha256, MAX_MANIFEST_BYTES)
    manifest, declared = _manifest(manifest_raw)
    _require(type(manifest.get("schema_version")) is int and manifest["schema_version"] == 1
             and manifest.get("profile") == BODY_PROFILE and manifest.get("status") == "construction_only"
             and manifest.get("simulation_ready") is False and not (_CONFLICTS & declared.keys())
             and _REQUIRED | {"body-source.json", _PRODUCER} <= declared.keys(),
             "A sealed construction-only exercise body is required")
    _require("pose-recovery.json" not in declared, "Recovered crop packages are not supported by this input adapter")
    selected = {name: reader.read(name, declared[name]) for name in sorted(_REQUIRED | {"body-source.json", _PRODUCER})}
    documents = {name: _json(raw) for name, raw in selected.items() if name.endswith(".json")}
    marker, source_meta, implementation = (documents[name] for name in ("body-source.json", "source.json", "implementation.json"))
    _require(type(marker.get("schema_version")) is int and marker["schema_version"] == 1
             and marker.get("profile") == BODY_PROFILE and marker.get("status") == "construction_only"
             and marker.get("simulation_ready") is False, "Body construction marker differs from its seal")
    source_sha, source_name = marker.get("source_sha256"), marker.get("source_artifact")
    _require(isinstance(source_sha, str) and _SHA.fullmatch(source_sha)
             and source_name in {"source" + suffix for suffix in (".mp4", ".webm", ".mov", ".m4v", ".avi", ".mkv")}
             and declared.get(source_name) == source_sha == source_meta.get("sha256")
             and source_meta.get("artifact") == source_name, "Sealed media identity metadata conflicts")
    artifacts = marker.get("artifacts")
    _require(isinstance(artifacts, dict) and set(artifacts) == _REQUIRED
             and all(artifacts[name] == declared[name] for name in _REQUIRED), "Body marker member identities conflict")
    construction = marker.get("construction", {})
    _require(construction.get("body_model") == "mobile" and construction.get("body_profile") == "exercise-v3"
             and construction.get("original_pose_config") == "run-config.json"
             and construction.get("original_pose_config_sha256") == declared["run-config.json"], "Construction configuration binding conflicts")
    producer_sha = declared[_PRODUCER]
    _require(producer_sha in _DECODERS and implementation.get("files", {}).get("exercise_vision.py") == producer_sha,
             "Unsupported saved decoder producer version")
    _pose_bindings(documents, selected, marker)
    original = documents["evidence.json"]
    _require("image_dimension_provenance" not in original,
             "Existing dimension claims require a separately supported authenticated provenance adapter")
    input_kind, source_manifest_sha = marker.get("input_kind"), _sha(manifest_raw)
    source_producer = _PRODUCER
    if input_kind == "sealed_pose_evidence":
        name = "inputs/source-manifest.json"
        _require(name in declared and marker.get("input_manifest_sha256") == declared[name], "Original source manifest is missing or unbound")
        source_manifest_raw = reader.read(name, declared[name], MAX_MANIFEST_BYTES)
        _, source_files = _manifest(source_manifest_raw)
        _require(all(source_files.get(name) == declared[name] for name in _POSE | {source_name}),
                 "Copied original source manifest differs from retained source metadata")
        producers = [name for name in _SOURCE_PRODUCERS if name in source_files]
        _require(producers and all(source_files[name] == producer_sha for name in producers),
                 "Original source manifest lacks the reviewed decoder producer binding")
        source_producer = producers[0]
        source_manifest_sha = _sha(source_manifest_raw)
    else:
        _require(input_kind == "local_video" and marker.get("input_manifest_sha256") is None,
                 "Unsupported original body input route")
        source_manifest_raw = None
    derived = deepcopy(original)
    derived["image_dimension_provenance"] = {
        "method": "sealed_decoder_metadata", "source_sha256": source_sha,
        "width": original["width"], "height": original["height"],
        "source_evidence_sha256": declared["evidence.json"],
        "source_evidence_canonical_sha256": _canonical(original), "source_manifest_sha256": source_manifest_sha,
    }
    derived_raw = _bytes(derived)
    _require(len(derived_raw) <= MAX_JSON_BYTES, "Derived evidence exceeds its byte bound")
    files = {"inputs/" + ("evidence-original.json" if name == "evidence.json" else name): raw
             for name, raw in selected.items()}
    files.update({"inputs/body-manifest.json": manifest_raw, "inputs/evidence-derived.json": derived_raw})
    if source_manifest_raw is not None:
        files["inputs/source-manifest.json"] = source_manifest_raw
    _require(sum(map(len, files.values())) <= MAX_INPUT_BYTES, "Selected input copies exceed their aggregate byte bound")
    identity = {"schema_version": 1, "profile": PROFILE, "status": "metadata_verified",
        "source_root": {"path": str(reader.root), "device": reader.identity[0], "inode": reader.identity[1]},
        "body_manifest_sha256": _sha(manifest_raw), "expected_manifest_supplied": expected_manifest_sha256 is not None,
        "source_manifest_sha256": source_manifest_sha, "source_sha256": source_sha, "source_artifact": source_name,
        "human_sha256": declared["human.json"], "evidence_original_sha256": declared["evidence.json"],
        "evidence_derived_sha256": _sha(derived_raw), "input_kind": input_kind,
        "decoder_provenance": {"profile": _DECODERS[producer_sha], "sha256": producer_sha,
            "copied_artifact": "inputs/" + _PRODUCER, "source_manifest_member": source_producer},
        "selected_source_files": len(reader.pins), "selected_source_bytes": reader.total,
        "copied_input_bytes": sum(map(len, files.values())),
        "media_verification": "Declared sealed media identity only; media bytes were not read or decoded",
        "scope": "Selected metadata hash custody and reviewed saved decoder contract; not external authenticity, calibrated imagery, physical accuracy or simulator admission",
        "dimension_derivation": "Only image_dimension_provenance is added; raw evidence, identities and native/nominal clocks are preserved"}
    snapshot = {"files": files, "identity": identity,
                "pins": [(reader.root / name, digest) for name, digest in reader.pins.items()]}
    _check(deadline)
    fd, _ = _root_fd(reader.root, reader.identity)
    os.close(fd)
    return snapshot


def recheck_reference_inputs(snapshot, *, deadline):
    """Recheck exactly consumed bytes, source root identity and failure marker before publication."""
    _check(deadline)
    _require(isinstance(snapshot, dict) and isinstance(snapshot.get("identity"), dict)
             and snapshot["identity"].get("profile") == PROFILE, "A prepared reference-input snapshot is required")
    identity = snapshot["identity"]["source_root"]
    root = _root_path(identity["path"])
    reader = _Reader(root, deadline, (identity["device"], identity["inode"]))
    pins = snapshot.get("pins")
    _require(isinstance(pins, list) and 1 <= len(pins) <= MAX_MEMBERS, "Invalid selected input pins")
    for path, digest in pins:
        _require(isinstance(path, Path) and path.is_absolute() and isinstance(digest, str) and _SHA.fullmatch(digest),
                 "Invalid selected input pin")
        name = path.relative_to(root).as_posix()
        maximum = MAX_MANIFEST_BYTES if name in {"manifest.json", "inputs/source-manifest.json"} else MAX_JSON_BYTES
        reader.read(name, digest, maximum)
    fd, _ = _root_fd(root, reader.identity)
    os.close(fd)
    _check(deadline)
