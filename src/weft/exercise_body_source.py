"""Prepare source-bound mobile bodies without a disposable simulator or learned task."""

from copy import deepcopy
import hashlib
import importlib.metadata
import io
import json
import math
import os
from pathlib import Path
import platform
import stat
import time

from weft import exercise_vision
from weft.exercise_source import describe_source
from weft.human import build_human, validate_human
from weft.io import local_time
from weft.observations import adapt_exercise, validate_observations
from weft.shared_world_packaging import read_package_file, read_package_manifest, write_package_files
from weft.world_evidence import _check_tree


PROFILE = "exercise-body-source-v1"
MAX_BYTES = 128 * 1024 * 1024
MAX_FILES = 512
MAX_FILE_BYTES = 64 * 1024 * 1024
MAX_SECONDS = 180
RESERVE_BYTES = 256 * 1024
REJECTED = {"failed", "error", "rejected", "cancelled", "canceled", "unsupported"}
SUFFIXES = {".mp4", ".webm", ".mov", ".m4v", ".avi", ".mkv"}
POSE_FILES = {"evidence.json", "pose-landmarks.json", "identity-audit.json", "model.json",
              "source.json", "run-config.json"}
CAPTURE_FILES = {"decoder-clock.json", "native-clock-capture.json", "native-clock-decode.log"}
OPTIONAL_FILES = {"joint-angles.json", "pose-quality.json", "background.jpg", "tracks.gif",
                  "tracks-contact-sheet.png"} | CAPTURE_FILES
IMPLEMENTATION_FILES = ("exercise_body_source.py", "exercise_vision.py", "exercise_source.py",
                        "human.py", "mobile_human.py", "root_motion.py", "body_frame.py",
                        "spatial_pose.py", "pose_identity.py", "observations.py", "video_io.py",
                        "shared_world_packaging.py", "world_evidence.py", "io.py", "errors.py")


def _bytes(value):
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _json(raw):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("Source JSON contains duplicate keys")
            result[key] = value
        return result

    def invalid(value):
        raise ValueError(f"Source JSON contains a non-finite value: {value}")

    try:
        result = json.loads(raw, object_pairs_hook=unique, parse_constant=invalid)
    except (UnicodeError, json.JSONDecodeError, RecursionError) as error:
        raise ValueError("Source artifacts require bounded strict JSON") from error
    if not isinstance(result, dict):
        raise ValueError("Source artifact must be a JSON object")
    _check_tree(result)
    return result


def _path(value):
    path = Path(value).absolute()
    if ".." in path.parts or any(part.is_symlink() for part in (path, *path.parents)):
        raise ValueError("Body source paths must not pass through symlinks")
    return path


def _deadline(deadline):
    if time.monotonic() >= deadline:
        raise TimeoutError("Body source preparation exceeded its cooperative deadline")


def _read(path, deadline, expected=None):
    _deadline(deadline)
    path = _path(path)
    raw = read_package_file(path.parent, path.name, max_bytes=MAX_FILE_BYTES,
                            expected_sha256=expected)
    _deadline(deadline)
    return raw


def _options(mode, identity, max_pose_hypotheses=2, person_region=None, pose_sampling="bounded"):
    exercise_vision.pose_sampling_metadata(pose_sampling)
    if mode not in {"video", "image"} or identity not in {"raw", "temporal"}:
        raise ValueError("Pose options require video/image mode and explicit raw/temporal identity")
    if type(max_pose_hypotheses) is not int or max_pose_hypotheses not in (1, 2):
        raise ValueError("Maximum pose hypotheses must be 1 (explicit single-person assumption) or 2")
    options = {"pose_mode": mode, "pose_identity": identity}
    if pose_sampling != "bounded":
        options["pose_sampling"] = pose_sampling
    # Omission preserves the existing two-hypothesis package contract.
    if max_pose_hypotheses == 1:
        options["max_pose_hypotheses"] = 1
    if person_region is not None:
        options["person_region"] = exercise_vision.validate_person_region(person_region)
    return options


def _inventory(root, deadline, *, reserve=False, hashes=True):
    files, total, directories, visited = {}, 0, [root], 0
    while directories:
        with os.scandir(directories.pop()) as entries:
            for entry in entries:
                _deadline(deadline)
                path = Path(entry.path)
                info = entry.stat(follow_symlinks=False)
                if stat.S_ISDIR(info.st_mode):
                    directories.append(path)
                    visited += 1
                    if visited > MAX_FILES:
                        raise ValueError("Body source exceeds its directory bound")
                elif stat.S_ISREG(info.st_mode):
                    if info.st_size > MAX_FILE_BYTES:
                        raise ValueError("Body source artifact exceeds the downstream 64-MiB file bound")
                    name = path.relative_to(root).as_posix()
                    files[name] = _sha(_read(path, deadline)) if hashes else None
                    total += info.st_size
                    if len(files) > MAX_FILES - (3 if reserve else 0) or total > MAX_BYTES - (RESERVE_BYTES if reserve else 0):
                        raise ValueError("Body source exceeds its downstream 512-file/128-MiB package bound")
                else:
                    raise ValueError("Body source artifacts must be regular files without links")
    return files, total


def _write(root, files, deadline):
    current, total = _inventory(root, deadline, reserve=True, hashes=False)
    if (any(len(raw) > MAX_FILE_BYTES for raw in files.values())
            or len(current) + len(files) > MAX_FILES - 3
            or total + sum(map(len, files.values())) > MAX_BYTES - RESERVE_BYTES):
        raise ValueError("Body source output reservation exceeds downstream bounds")
    write_package_files(root, files)
    _deadline(deadline)


class _CappedBuffer(io.BytesIO):
    def __init__(self, maximum, deadline):
        super().__init__()
        self.maximum, self.deadline = maximum, deadline

    def write(self, raw):
        _deadline(self.deadline)
        if self.tell() + len(raw) > self.maximum:
            raise ValueError("Encoded artifact exceeds its reserved output capacity")
        return super().write(raw)

    def truncate(self, size=None):
        if (self.tell() if size is None else size) > self.maximum:
            raise ValueError("Encoded artifact exceeds its reserved output capacity")
        return super().truncate(size)


class BodyArtifactSink:
    """Bound producer encoding and persistence; only files created by this sink may be replaced."""

    def __init__(self, root, *, deadline):
        self.root, self.deadline = _path(root), deadline
        self.owned = {}

    def _buffer(self, name):
        if not isinstance(name, str) or Path(name).name != name or name in {"", ".", ".."}:
            raise ValueError("Producer artifacts require a single safe filename")
        files, total = _inventory(self.root, self.deadline, reserve=True, hashes=False)
        if name in files and name not in self.owned:
            raise ValueError("Producer cannot replace a preserved input artifact")
        if name not in files and len(files) + 1 > MAX_FILES - 3:
            raise ValueError("Producer output exceeds its file reservation")
        old = self.owned.get(name)
        if old is not None:
            info = (self.root / name).stat(follow_symlinks=False)
            if (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns) != old:
                raise ValueError("Producer artifact changed outside its owner")
        capacity = min(MAX_FILE_BYTES, MAX_BYTES - RESERVE_BYTES - total + (old[2] if old else 0))
        if capacity <= 0:
            raise ValueError("Producer output has no remaining byte reservation")
        return _CappedBuffer(capacity, self.deadline)

    def _commit(self, name, buffer):
        # Encode first so failed encoding leaves the previous artifact intact.
        _deadline(self.deadline)
        view = buffer.getbuffer()
        root_fd = None
        try:
            with self._buffer(name) as reservation:
                if len(view) > reservation.maximum:
                    raise ValueError("Producer capacity changed during encoding")
            root_fd = os.open(self.root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
            old = self.owned.get(name)
            flags = os.O_RDWR | os.O_NOFOLLOW | os.O_NONBLOCK
            if old is None:
                flags |= os.O_CREAT | os.O_EXCL
            fd = os.open(name, flags, 0o600, dir_fd=root_fd)
            with os.fdopen(fd, "r+b") as stream:
                info = os.fstat(stream.fileno())
                if not stat.S_ISREG(info.st_mode) or (old is not None and
                        (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns) != old):
                    raise ValueError("Producer destination changed before its bounded write")
                stream.truncate(0)
                stream.write(view)
                stream.flush()
                info = os.fstat(stream.fileno())
                self.owned[name] = (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns)
        finally:
            if root_fd is not None:
                os.close(root_fd)
            view.release()
        _deadline(self.deadline)

    def write_json(self, name, value):
        with self._buffer(name) as buffer:
            for part in json.JSONEncoder(indent=2, sort_keys=True, allow_nan=False).iterencode(value):
                buffer.write(part.encode())
            buffer.write(b"\n")
            self._commit(name, buffer)

    def write_text(self, name, value):
        with self._buffer(name) as buffer:
            buffer.write(value.encode())
            self._commit(name, buffer)

    def write_image(self, name, image, **options):
        formats = {".gif": "GIF", ".png": "PNG", ".jpg": "JPEG"}
        if Path(name).suffix not in formats:
            raise ValueError("Unsupported bounded observation-image format")
        with self._buffer(name) as buffer:
            image.save(buffer, format=formats[Path(name).suffix], **options)
            self._commit(name, buffer)

    def write_source_preview(self, name, frame):
        from PIL import Image

        self.write_image(name, Image.fromarray(frame[:, :, ::-1]), quality=95)


def _start(out, inputs, maximum):
    if type(maximum) not in (int, float) or not math.isfinite(maximum) or not 1 <= maximum <= MAX_SECONDS:
        raise ValueError("Body source preparation requires a 1–180 second bound")
    target = _path(out)
    if target.exists() or any(target == path or target.is_relative_to(path) or path.is_relative_to(target)
                              for path in inputs):
        raise ValueError("Body source output must be new and outside its inputs")
    started = time.monotonic()
    target.mkdir(parents=True, exist_ok=False)
    return target, started, started + maximum


def _failure(target, error, started):
    # Partial evidence survives a failed stage, but it cannot receive a successful construction seal.
    receipt = {"schema_version": 1, "profile": PROFILE, "status": "failed", "simulation_ready": False,
               "elapsed_seconds": time.monotonic() - started, "created_at": local_time(),
               "error": {"type": type(error).__name__, "message": str(error)[:2000]}}
    try:
        write_package_files(target, {"failure.json": _bytes(receipt)})
        files, total = _inventory(target, time.monotonic() + 2)
        manifest = _bytes({"schema_version": 1, "profile": PROFILE, "status": "failed",
                           "simulation_ready": False, "files": files})
        if len(files) + 1 <= MAX_FILES and total + len(manifest) <= MAX_BYTES:
            write_package_files(target, {"manifest.json": manifest})
    except (ValueError, OSError, TimeoutError):
        pass


def _implementation(deadline, extra_files=()):
    package = Path(__file__).parent
    files = {name: _read(package / name, deadline) for name in dict.fromkeys((*IMPLEMENTATION_FILES, *extra_files))}
    versions = {"python": platform.python_version()}
    for name in ("numpy", "mujoco", "mediapipe", "opencv-contrib-python", "Pillow"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "unavailable"
    return files, {"scope": "Selected generation and custody modules, not a complete transitive Python runtime closure",
                   "files": {name: _sha(raw) for name, raw in files.items()}, "runtime_versions": versions}


def _model(model, options):
    variants = exercise_vision.MODEL_VARIANTS
    matches = [name for name, entry in variants.items()
               if model.get("sha256") == entry["sha256"] and model.get("bytes") == entry["bytes"]]
    if (len(matches) != 1 or model.get("model_version") != "float16/1"
            or model.get("delegate") != "CPU" or model.get("network_inference") is not False
            or type(model.get("maximum_pose_hypotheses")) is not int
            or model.get("maximum_pose_hypotheses") != options.get("max_pose_hypotheses", 2)
            or model.get("running_mode") != options["pose_mode"].upper()
            or model.get("temporal_tracking") is not (options["pose_mode"] == "video")
            or not isinstance(model.get("mediapipe_version"), str) or not model["mediapipe_version"]):
        raise ValueError("Pose model identity/version/options do not match the supported source contract")
    if options.get("max_pose_hypotheses") == 1 and model.get("person_count_assumption") != "single_person_unverified":
        raise ValueError("One-hypothesis evidence must retain its explicit unverified single-person assumption")
    if options.get("max_pose_hypotheses", 2) == 2 and "person_count_assumption" in model:
        raise ValueError("Single-person model metadata conflicts with the two-hypothesis contract")
    if "variant" in model and model["variant"] != matches[0]:
        raise ValueError("Pose model variant conflicts with its pinned bytes")
    if model.get("pose_sampling") != exercise_vision.pose_sampling_metadata(options.get("pose_sampling", "bounded")):
        raise ValueError("Pose model sampling differs from the declared source options")
    region = options.get("person_region")
    if model.get("person_region") != (None if region is None else exercise_vision.person_region_metadata(region)):
        raise ValueError("Pose model person region differs from the declared source options")


def _validated_pose(raw_files, options, source_sha, *, deadline=None):
    evidence, raw, audit, model, config, source = (_json(raw_files[name]) for name in
        ("evidence.json", "pose-landmarks.json", "identity-audit.json", "model.json", "run-config.json", "source.json"))
    if (evidence.get("profile") != "exercise-pose-v1" or evidence.get("schema_version") != 1
            or evidence.get("source_sha256") != source_sha or source.get("sha256") != source_sha
            or config.get("profile") != "exercise"
            or _options(config.get("pose_mode", "video"), config.get("pose_identity", "raw"),
                        config.get("max_pose_hypotheses", 2), config.get("person_region"), config.get("pose_sampling", "bounded")) != options):
        raise ValueError("Pose/source/config identities conflict")
    _model(model, options)
    if evidence.get("pose_sampling") != exercise_vision.pose_sampling_metadata(options.get("pose_sampling", "bounded")):
        raise ValueError("Pose evidence sampling differs from the declared source options")
    if evidence.get("model") != model or raw.get("model") != model:
        raise ValueError("Raw and derived pose model metadata must remain identical")
    recovery = "pose-recovery.json" in raw_files
    mode_recovery = "pose-mode-recovery.json" in raw_files
    capture_recovery = "pose-capture-recovery.json" in raw_files
    if capture_recovery:
        from weft.pose_capture_recovery import validate_capture_recovery_files
        if recovery or mode_recovery:
            raise ValueError("Interrupted capture recovery cannot be chained with pose refinements")
        validate_capture_recovery_files(raw_files, source_sha256=source_sha, deadline=deadline)
    if recovery and mode_recovery:
        raise ValueError("Chained crop/mode recovery is unsupported")
    if recovery:
        from weft.pose_source_recovery import validate_pose_recovery_files
        validate_pose_recovery_files(raw_files, source_sha256=source_sha, deadline=deadline)
    elif evidence.get("pose_reinference") or any("image_refinement" in row for row in evidence.get("frames", [])):
        raise ValueError("Crop-derived observations require their explicit complete recovery proof")
    if mode_recovery:
        from weft.pose_mode_source import validate_pose_mode_files
        validate_pose_mode_files(raw_files, source_sha256=source_sha, deadline=deadline)
    elif evidence.get("pose_mode_recovery") or any("mode_refinement" in row for row in evidence.get("frames", [])):
        raise ValueError("Mode-derived observations require their explicit complete recovery proof")
    identity = evidence.get("pose_identity", {})
    if (identity.get("mode") != options["pose_identity"] or audit.get("mode") != options["pose_identity"]
            or audit.get("source_sha256") != source_sha or audit.get("raw_artifact") != "pose-landmarks.json"
            or identity.get("audit_artifact") != "identity-audit.json"
            or identity.get("raw_artifact_sha256") != _sha(raw_files["pose-landmarks.json"])
            or audit.get("raw_artifact_sha256") != identity.get("raw_artifact_sha256")):
        raise ValueError("Pose identity derivation is not bound to its exact original estimates")
    fps, count, decoded, stride = (evidence.get(name) for name in
                                  ("fps", "frame_count", "decoded_frames", "inference_stride"))
    if (type(fps) not in (int, float) or not math.isfinite(fps) or not 1 <= fps <= 240
            or type(count) is not int or not 10 <= count <= 1800 or count / fps > 60
            or type(decoded) is not int or not 10 <= decoded <= 1800 or decoded / fps > 60
            or type(stride) is not int
            or stride != exercise_vision.pose_sampling_stride(options.get("pose_sampling", "bounded"), fps, count, decoded)):
        raise ValueError("Pose source exceeds the existing frame/duration/sampling contract")
    if options.get("pose_sampling") == "all_frames" and evidence.get("sample_rate_fps") != fps:
        raise ValueError("All-frame pose sampling must retain the original nominal source rate")
    if abs(decoded - count) > 1:
        container = evidence.get("container_frame_audit", {})
        if (container.get("encoded_video_frames") != decoded or container.get("decoded_frames") != decoded
                or container.get("matches") is not True):
            raise ValueError("Metadata frame discrepancy lacks the existing encoded-frame audit")
    frames, original = evidence.get("frames"), raw.get("frames")
    expected = list(range(0, decoded, stride))
    if (not isinstance(frames, list) or not isinstance(original, list) or not 1 <= len(frames) <= 900
            or any(not isinstance(row, dict) for row in frames + original)
            or len(original) != len(frames) or [row.get("frame") for row in frames] != expected):
        raise ValueError("Every sampled source slot must retain its original pose or explicit failure")
    exercise_vision.validate_person_region_evidence(evidence, raw, options.get("person_region"))
    clocks, repaired = [], []
    for row, prior in zip(frames, original):
        index = row["frame"]
        if (type(index) is not int or row.get("id") != f"{source_sha}:frame:{index}"
                or type(row.get("time_s")) not in (int, float)
                or abs(row["time_s"] - index / fps) > 1e-7
                or any(key not in row or row[key] != prior.get(key) for key in
                       ("id", "frame", "time_s", "decoder_timestamp_s"))):
            raise ValueError("Raw/derived pose source identities and native clocks must match exactly")
        native = row["decoder_timestamp_s"]
        if native is not None and (type(native) not in (int, float) or not math.isfinite(native) or not 0 <= native <= 60):
            raise ValueError("Invalid native decoder timestamp; it cannot be replaced with nominal time")
        if "mode_refinement" in row:
            if not mode_recovery or row.get("mode_refinement_raw") != prior:
                raise ValueError("Mode recovery must retain its exact original detector observation")
        elif "image_refinement" in row:
            if not recovery or row.get("image_refinement_raw") != prior:
                raise ValueError("Crop recovery must retain its exact original detector observation")
        elif "identity_raw" in row:
            if options["pose_identity"] != "temporal" or row["identity_raw"] != prior:
                raise ValueError("Temporal pose correction must retain its exact original record")
            repaired.append(row["id"])
        elif row != prior:
            raise ValueError("Uncorrected pose samples differ from the preserved detector output")
        clocks.append({key: row[key] for key in ("frame", "time_s", "decoder_timestamp_s")})
    if (audit.get("repaired_evidence_ids") != repaired or identity.get("repaired_frames") != len(repaired)
            or evidence.get("quality", {}).get("passes") is not True
            or evidence.get("quality", {}).get("errors") or not exercise_vision._quality(evidence)["passes"]):
        raise ValueError("Rejected/unsupported pose evidence cannot become a construction body")
    clock = {"schema_version": 1, "source_sha256": source_sha, "fps": fps,
             "metadata_frames": count, "decoded_frames": decoded, "samples": clocks,
             "nominal_basis": "frame_index_divided_by_fps", "nominal_status": "assumed",
             "native_scope": "Exact retained decoder timestamps; no calibrated/native-clock or cadence verification claim",
             "unsampled_native_times": "unknown_not_interpolated"}
    if "native-clock.json" in raw_files:
        from weft.source_clock import derive_evidence, validate_clock

        binding = validate_clock(_json(raw_files["native-clock.json"]), evidence,
                                 evidence_sha256=_sha(raw_files["evidence.json"]))
        evidence = derive_evidence(evidence, binding)
    observations = validate_observations(adapt_exercise(evidence))
    observations["provenance"]["input_artifacts"] = {"evidence.json": _sha(raw_files["evidence.json"])}
    if "native-clock.json" in raw_files:
        observations["provenance"]["input_artifacts"]["native-clock.json"] = _sha(raw_files["native-clock.json"])
    for name in CAPTURE_FILES & raw_files.keys():
        observations["provenance"]["input_artifacts"][name] = _sha(raw_files[name])
    if capture_recovery:
        from weft.pose_capture_recovery import PROOF_FILES as CAPTURE_RECOVERY_FILES
        for name in CAPTURE_RECOVERY_FILES:
            observations["provenance"]["input_artifacts"][name] = _sha(raw_files[name])
    return evidence, observations, clock


def _validate_capture_files(files, source_sha):
    if not CAPTURE_FILES & files.keys():
        return False
    if not CAPTURE_FILES <= files.keys() or "native-clock.json" not in files:
        raise ValueError("Native timestamp capture requires its complete decoder and presentation receipts")
    from weft.source_clock_capture import validate_capture

    decoder = _json(files["decoder-clock.json"])
    if (set(decoder) != {"schema_version", "source_sha256", "complete", "rows"}
            or type(decoder["schema_version"]) is not int or decoder["schema_version"] != 1
            or decoder["source_sha256"] != source_sha or decoder["complete"] is not True):
        raise ValueError("Native decoder clock capture is incomplete or belongs to another source")
    validate_capture(_json(files["native-clock.json"]), _json(files["native-clock-capture.json"]),
                     files["native-clock-decode.log"], decoder["rows"], source_sha256=source_sha,
                     evidence_sha256=_sha(files["evidence.json"]))
    return True


def validate_body_source_files(files, *, deadline=None):
    """Validate manifest-covered body-source bytes before the existing seed constructor consumes them."""
    required = POSE_FILES | {"body-source.json", "human.json", "observations.json", "source-clock.json",
                            "implementation.json"}
    native = "native-clock.json" in files
    if CAPTURE_FILES & files.keys():
        required |= CAPTURE_FILES
    if native:
        required |= {"native-clock.json", "reconstruction-evidence.json"}
    elif "reconstruction-evidence.json" in files:
        raise ValueError("Derived reconstruction evidence requires its native clock binding")
    if "pose-recovery.json" in files:
        from weft.pose_source_recovery import PROOF_FILES
        required |= PROOF_FILES
    if "pose-mode-recovery.json" in files:
        from weft.pose_mode_source import PROOF_FILES as MODE_PROOF_FILES
        required |= MODE_PROOF_FILES
    if "pose-capture-recovery.json" in files:
        from weft.pose_capture_recovery import PROOF_FILES as CAPTURE_RECOVERY_FILES
        required |= CAPTURE_RECOVERY_FILES
    if not required <= files.keys():
        raise ValueError("Construction body source lacks required identity-bound artifacts")
    marker = _json(files["body-source.json"])
    if (marker.get("schema_version") != 1 or marker.get("profile") != PROFILE
            or marker.get("status") != "construction_only" or marker.get("simulation_ready") is not False):
        raise ValueError("Body source requires its explicit construction-only profile")
    options = marker.get("pose_options", {})
    if not isinstance(options, dict) or options != _options(options.get("pose_mode"), options.get("pose_identity"),
                                                          options.get("max_pose_hypotheses", 2), options.get("person_region"), options.get("pose_sampling", "bounded")):
        raise ValueError("Body source options are invalid")
    construction = marker.get("construction", {})
    if (not isinstance(construction, dict) or construction.get("body_model") != "mobile"
            or construction.get("body_profile") != "exercise-v3"
            or construction.get("original_pose_config") != "run-config.json"
            or construction.get("original_pose_config_sha256") != _sha(files["run-config.json"])):
        raise ValueError("Executed mobile construction settings must be distinct from preserved original run settings")
    declared = marker.get("artifacts")
    expected = required - {"body-source.json"}
    if (not isinstance(declared, dict) or set(declared) != expected
            or any(declared[name] != _sha(files[name]) for name in expected)):
        raise ValueError("Body source artifact bytes differ from their declared identities")
    source_name, source_sha = marker.get("source_artifact"), marker.get("source_sha256")
    if (source_name not in {"source" + suffix for suffix in SUFFIXES} or source_name not in files
            or _sha(files[source_name]) != source_sha
            or _json(files["source.json"]).get("artifact") != source_name):
        raise ValueError("Body source original video identity is missing or conflicting")
    captured = _validate_capture_files(files, source_sha)
    if captured != (construction.get("native_clock_capture") == "native-clock-capture.json"):
        raise ValueError("Body source native timestamp capture differs from its declared receipts")
    evidence, observations, clock = _validated_pose(files, options, source_sha, deadline=deadline)
    if native and (files["reconstruction-evidence.json"] != _bytes(evidence)
                   or construction.get("clock_profile") != "source-native-clock-v1"):
        raise ValueError("Native reconstruction view differs from exact source-bound timing")
    if files["observations.json"] != _bytes(observations) or files["source-clock.json"] != _bytes(clock):
        raise ValueError("Body source observations/native clocks differ from the original pose evidence")
    human = validate_human(_json(files["human.json"]))
    if native:
        from weft.source_clock import clock_summary, validate_derived_evidence

        if human.get("native_clock") != clock_summary(validate_derived_evidence(evidence)):
            raise ValueError("Body timing differs from the native reconstruction binding")
    elif "native_clock" in human or "clock_profile" in construction:
        raise ValueError("Native body timing requires its complete sealed derivation")
    if (human.get("profile") != "exercise-v3" or "camera_reference" in human
            or human.get("source_sha256") != source_sha or not 5 <= evidence["fps"] <= 240):
        raise ValueError("Body source is incompatible with the mobile shared-world constructor")
    usable = [row for row in evidence["frames"] if row["status"] in exercise_vision.USABLE_STATUSES]
    if ([row["time_s"] for row in human["reference"]] != [row["time_s"] for row in usable]
            or [row["evidence_id"] for row in human["reference_evidence"]] != [row["id"] for row in usable]):
        raise ValueError("Body reference identities must cover the original usable pose samples exactly")
    return human, observations


def _finish(target, options, source_name, input_kind, input_manifest, pins, implementation, started, deadline, progress,
            *, native_capture_performed=False):
    source_sha = _sha(_read(target / source_name, deadline))
    pose_names = set(POSE_FILES)
    native = (target / "native-clock.json").exists()
    captured = (target / "native-clock-capture.json").exists()
    if native:
        pose_names.add("native-clock.json")
    if captured:
        pose_names |= CAPTURE_FILES
    if (target / "pose-recovery.json").exists():
        from weft.pose_source_recovery import PROOF_FILES
        pose_names |= PROOF_FILES | {source_name}
    if (target / "pose-mode-recovery.json").exists():
        from weft.pose_mode_source import PROOF_FILES as MODE_PROOF_FILES
        pose_names |= MODE_PROOF_FILES | {source_name}
    if (target / "pose-capture-recovery.json").exists():
        from weft.pose_capture_recovery import PROOF_FILES as CAPTURE_RECOVERY_FILES
        pose_names |= CAPTURE_RECOVERY_FILES | {source_name}
    evidence, observations, clock = _validated_pose(
        {name: _read(target / name, deadline) for name in pose_names}, options, source_sha, deadline=deadline)
    if native:
        _write(target, {"reconstruction-evidence.json": _bytes(evidence)}, deadline)
        pose_names.add("reconstruction-evidence.json")
    before, _ = _inventory(target, deadline, reserve=True)
    if progress:
        progress("body", {"message": "Preparing a construction-only mobile body; no training or simulated trial"})
    _deadline(deadline)
    human = validate_human(build_human(deepcopy(evidence), target, body_model="mobile",
                                       artifact_sink=BodyArtifactSink(target, deadline=deadline)))
    _deadline(deadline)
    if (human.get("profile") != "exercise-v3" or "camera_reference" in human
            or human.get("source_sha256") != source_sha or not 5 <= evidence["fps"] <= 240):
        raise ValueError("Body constructor returned an incompatible profile/source")
    for name, expected in before.items():
        _read(target / name, deadline, expected)
    _write(target, {"observations.json": _bytes(observations), "source-clock.json": _bytes(clock)}, deadline)
    for path, expected in pins:
        _read(path, deadline, expected)
    if input_kind == "sealed_pose_evidence":
        failure_marker = pins[0][0].parent / "failure.json"
        if failure_marker.exists() or failure_marker.is_symlink():
            raise ValueError("Saved pose source gained a failure marker during body preparation")
    for name, expected in implementation["files"].items():
        _read(Path(__file__).parent / name, deadline, expected)
    files, _ = _inventory(target, deadline, reserve=True)
    receipt = {"schema_version": 1, "profile": PROFILE, "status": "construction_only", "simulation_ready": False,
        "created_at": local_time(), "input_kind": input_kind, "input_manifest_sha256": input_manifest,
        "source_sha256": source_sha, "source_artifact": source_name, "pose_options": options,
        "construction": {"body_model": "mobile", "body_profile": "exercise-v3",
            "original_pose_config": "run-config.json", "original_pose_config_sha256": files["run-config.json"],
            "original_pose_config_scope": "Preserved original pose preparation. Its historical body_model, task and training fields were not executed by this construction stage."},
        "artifacts": {name: files[name] for name in (*sorted(pose_names - {source_name}), "human.json", "observations.json",
                                                  "source-clock.json", "implementation.json")},
        "source_clock": {"basis": clock["nominal_basis"], "status": "assumed",
                         "native_scope": clock["native_scope"], "sampled_slots": len(clock["samples"]),
                         "total_decoded_slots": clock["decoded_frames"]},
        "work": {"video_decodes": int(input_kind == "local_video") + int(native_capture_performed),
                 "inference_calls": len(evidence["frames"]) if input_kind == "local_video" else 0,
                 "floor_forward_calls": 1, "physics_steps": 0, "training_episodes": 0,
                 "simulation_renders": 0, "api_calls": 0,
                 "observation_media": "existing source annotations generated" if input_kind == "local_video" else "saved bytes reused"},
        "bounds": {"maximum_seconds": deadline - started, "maximum_bytes": MAX_BYTES,
                   "maximum_files": MAX_FILES, "maximum_file_bytes": MAX_FILE_BYTES,
                   "deadline_scope": "Cooperative checks around bounded native calls; caller subprocess must enforce a hard wall limit"},
        "elapsed_seconds": time.monotonic() - started,
        "limits": ["Construction input only; no executable environment, task, training or simulation success is claimed",
                   "Body, root, floor and scale are authored/model-derived assumptions, not recovered anatomy or calibrated physical measurements",
                   "Existing first-pose floor placement uses one MuJoCo forward-geometry calculation without a dynamics step",
                   "Observed-frame pose annotations are not simulated-body renders; model weights are identified, not copied"]}
    if native:
        from weft.source_clock import clock_summary, validate_derived_evidence

        receipt["construction"]["clock_profile"] = "source-native-clock-v1"
        receipt["source_clock"] = clock_summary(validate_derived_evidence(evidence))
        receipt["limits"].append("Native elapsed presentation times drive derived reconstruction; original detector timestamps and nominal pose records remain unchanged. Capture timing is unverified.")
    if captured:
        receipt["construction"]["native_clock_capture"] = "native-clock-capture.json"
    _write(target, {"body-source.json": _bytes(receipt)}, deadline)
    validate_body_source_files({name: _read(target / name, deadline) for name in
                               {*receipt["artifacts"], "body-source.json", source_name}}, deadline=deadline)
    files, total = _inventory(target, deadline)
    manifest = _bytes({"schema_version": 1, "profile": PROFILE, "status": "construction_only",
                       "simulation_ready": False, "files": files})
    if len(files) + 1 > MAX_FILES or total + len(manifest) > MAX_BYTES:
        raise ValueError("Body source seal exceeds downstream bounds")
    _deadline(deadline)
    write_package_files(target, {"manifest.json": manifest})
    return target


def prepare_exercise_body_source(video, out, *, pose_model, pose_mode="video", pose_identity="raw",
                                 source_note="Rights not independently verified; user-provided local video",
                                 max_seconds=180, progress=None, max_pose_hypotheses=2, person_region=None,
                                 native_timing=False, pose_sampling="bounded"):
    """Decode/infer once and seal mobile construction inputs; no learned task or dynamics."""
    options = _options(pose_mode, pose_identity, max_pose_hypotheses, person_region, pose_sampling)
    if type(native_timing) is not bool:
        raise ValueError("Native timing must be explicitly enabled or disabled")
    if not isinstance(source_note, str) or len(source_note) > 4000:
        raise ValueError("Source provenance note must be at most 4000 characters")
    video, model = _path(video), _path(pose_model)
    if video.suffix.lower() not in SUFFIXES:
        raise ValueError("Provide a supported local video container")
    for path in (video, model):
        if not path.is_file() or path.stat().st_size > MAX_FILE_BYTES:
            raise ValueError("Body-source inputs must fit the downstream 64-MiB per-file bound")
    target, started, deadline = _start(out, (video, model), max_seconds)
    decoder_rows, source_sha = [], None
    try:
        video_raw, model_raw = _read(video, deadline), _read(model, deadline)
        source_sha, model_sha = _sha(video_raw), _sha(model_raw)
        provenance = exercise_vision.model_provenance(model)
        if provenance.get("sha256") != model_sha or provenance.get("bytes") != len(model_raw):
            raise ValueError("Pose model changed during source preparation")
        del model_raw
        source_name = "source" + video.suffix.lower()
        source = describe_source(source_sha) or {"original_name": video.name, "rights": source_note}
        source = {**source, "sha256": source_sha, "artifact": source_name,
                  "derived_overlays": "Pose annotations are adaptations; retain source attribution/license"}
        snapshots, implementation = (_implementation(deadline, ("source_clock.py", "source_clock_capture.py"))
                                     if native_timing else _implementation(deadline))
        files = {source_name: video_raw, "source.json": _bytes(source),
            "run-config.json": _bytes({"schema_version": 1, "profile": "exercise", "body_model": "mobile",
                "video": str(video), "pose_model": str(model), "source_note": source_note, **options,
                **({"native_timing": True} if native_timing else {})}),
            "implementation.json": _bytes(implementation),
            **{"implementation-source/" + name: raw for name, raw in snapshots.items()}}
        _write(target, files, deadline)
        # Construction reloads these durable bytes; retaining them duplicates dense-source memory.
        del files, video_raw, snapshots

        def report(stage, detail):
            _deadline(deadline)
            _inventory(target, deadline, reserve=True, hashes=False)
            if progress:
                progress(stage, detail)

        def capture_decoder(row):
            if len(decoder_rows) >= 1800:
                raise ValueError("Native decoder capture exceeds the 1800-frame bound")
            decoder_rows.append(dict(row))

        evidence = exercise_vision.inspect_exercise(target / source_name, target, model, progress=report,
            artifact_sink=BodyArtifactSink(target, deadline=deadline), **options,
            **({"decoder_clock_sink": capture_decoder} if native_timing else {}))
        _deadline(deadline)
        if evidence.get("model", {}).get("sha256") != model_sha:
            raise ValueError("Returned pose evidence used a different model than the pinned input")
        _write(target, {"model.json": _bytes(evidence["model"]), "pose-quality.json": _bytes(evidence["quality"])}, deadline)
        if evidence != _json(_read(target / "evidence.json", deadline)):
            raise ValueError("Returned pose evidence differs from its preserved artifact")
        if native_timing:
            from weft.source_clock_capture import collect_native_clock

            _write(target, {"decoder-clock.json": _bytes({"schema_version": 1, "source_sha256": source_sha,
                "complete": True, "rows": decoder_rows})}, deadline)
            raw_pose = {name: _read(target / name, deadline) for name in POSE_FILES}
            _validated_pose(raw_pose, options, source_sha, deadline=deadline)
            report("native_clock", {"message": "Verifying every frame's presentation time before body reconstruction"})
            binding, capture, log = collect_native_clock(target / source_name, raw_pose["evidence.json"],
                decoder_rows, max_seconds=min(20, max(.01, deadline - time.monotonic())))
            _write(target, {"native-clock.json": _bytes(binding), "native-clock-capture.json": _bytes(capture),
                            "native-clock-decode.log": log}, deadline)
            del raw_pose, binding, capture, log
        del evidence
        return _finish(target, options, source_name, "local_video", None,
                       [(video, source_sha), (model, model_sha)], implementation, started, deadline, progress,
                       native_capture_performed=native_timing)
    except Exception as error:
        if native_timing and decoder_rows and not (target / "decoder-clock.json").exists():
            try:
                # Reserved failure metadata retains partial timing even after the work deadline.
                write_package_files(target, {"decoder-clock.json": _bytes({"schema_version": 1,
                    "source_sha256": source_sha, "complete": False, "rows": decoder_rows})})
            except (ValueError, OSError):
                pass
        _failure(target, error, started)
        raise


def prepare_exercise_body_source_from_evidence(pose_source, out, *, native_clock=None, max_seconds=30, progress=None,
                                               native_timing=False, decoder_clock_rows=None):
    """Reuse sealed poses without inference; optionally capture native timing for verified crop recovery."""
    if type(native_timing) is not bool:
        raise ValueError("Native timing must be explicitly enabled or disabled")
    if native_timing and native_clock is not None:
        raise ValueError("Choose native capture or an existing native_clock binding, not both")
    if not native_timing and decoder_clock_rows is not None:
        raise ValueError("Original decoder rows require explicit native timing capture")
    if native_timing:
        if (not isinstance(decoder_clock_rows, list) or not 2 <= len(decoder_clock_rows) <= 1800
                or any(not isinstance(row, dict) or set(row) != {"frame", "decoder_timestamp_s"}
                       or type(row["frame"]) is not int or row["frame"] != i
                       or type(row["decoder_timestamp_s"]) not in (int, float)
                       for i, row in enumerate(decoder_clock_rows))):
            raise ValueError("Native crop capture requires complete original decoder callback rows")
        decoder_clock_rows = _json(_bytes({"rows": decoder_clock_rows}))["rows"]
    source = _path(pose_source)
    clock_path = _path(native_clock) if native_clock is not None else None
    run = _path(source / "run") if (source / "run" / "manifest.json").is_file() else source
    target, started, deadline = _start(out, (source, *((clock_path,) if clock_path is not None else ())), max_seconds)
    try:
        manifest_raw, declared = read_package_manifest(run)
        manifest = _json(manifest_raw)
        if native_timing and manifest.get("profile") != "exercise-pose-recovery-v1":
            raise ValueError("Saved automatic native capture requires verified selective-crop recovery")
        if "native-clock.json" in declared and clock_path is None:
            raise ValueError("Native-timed sources require an explicit native_clock input; timing cannot silently revert")
        if (manifest.get("status") in REJECTED or "failure.json" in declared
                or (run / "failure.json").exists() or (run / "failure.json").is_symlink()
                or not POSE_FILES <= declared.keys() or len(declared) + 1 > MAX_FILES):
            raise ValueError("Saved body preparation requires bounded, unrejected original pose artifacts")
        source_meta = _json(_read(run / "source.json", deadline, declared["source.json"]))
        source_name = source_meta.get("artifact")
        if (not isinstance(source_name, str) or source_name not in {"source" + suffix for suffix in SUFFIXES}
                or source_name not in declared or declared[source_name] != source_meta.get("sha256")):
            raise ValueError("Saved poses must retain their exact original video bytes")
        pins = [(run / "manifest.json", _sha(manifest_raw))]
        if run != source and (source / "status.json").exists():
            status_raw = _read(source / "status.json", deadline)
            if _json(status_raw).get("status") in REJECTED:
                raise ValueError("A failed lab cannot be relabeled as a successful body source")
            pins.append((source / "status.json", _sha(status_raw)))
        kept, total = {}, len(manifest_raw)
        recovery_files, recovery_implementation = set(), ()
        if manifest.get("profile") == "exercise-pose-recovery-v1":
            from weft.pose_source_recovery import PROOF_FILES, IMPLEMENTATION_FILES as RECOVERY_IMPLEMENTATION
            recovery_files, recovery_implementation = set(PROOF_FILES), RECOVERY_IMPLEMENTATION
            if not recovery_files <= declared.keys():
                raise ValueError("Saved pose recovery lacks its complete derivation proof")
        elif manifest.get("profile") == "exercise-pose-mode-recovery-v1":
            from weft.pose_mode_source import PROOF_FILES, IMPLEMENTATION_FILES as RECOVERY_IMPLEMENTATION
            recovery_files, recovery_implementation = set(PROOF_FILES), RECOVERY_IMPLEMENTATION
            if not recovery_files <= declared.keys():
                raise ValueError("Saved mode recovery lacks its complete derivation proof")
        elif manifest.get("profile") == "interrupted-pose-capture-recovery-v1":
            from weft.pose_capture_recovery import PROOF_FILES as CAPTURE_RECOVERY_FILES
            recovery_files, recovery_implementation = set(CAPTURE_RECOVERY_FILES), ("pose_capture_recovery.py",)
            if not recovery_files <= declared.keys():
                raise ValueError("Saved capture recovery lacks its complete failed-study ancestry")
        elif "pose-capture-recovery.json" in declared:
            raise ValueError("Interrupted capture proof requires its explicit sealed package profile")
        elif "pose-recovery.json" in declared or "pose-mode-recovery.json" in declared:
            raise ValueError("Pose recovery proof requires its explicit sealed package profile")
        if {"pose-recovery.json", "pose-mode-recovery.json"} <= declared.keys():
            raise ValueError("Chained crop/mode recovery is unsupported")
        for name, expected in sorted(declared.items()):
            raw = _read(run / name, deadline, expected)
            total += len(raw)
            if total > MAX_BYTES:
                raise ValueError("Saved pose input exceeds the downstream 128-MiB bound")
            if name in {"status.json", "report.json", "reconstruction.json", "body-source.json"} and _json(raw).get("status") in REJECTED:
                raise ValueError("Rejected saved artifacts cannot become construction inputs")
            pins.append((run / name, expected))
            if name in POSE_FILES | OPTIONAL_FILES | {source_name} | recovery_files:
                kept[name] = raw
        config = _json(kept["run-config.json"])
        options = _options(config.get("pose_mode", "video"), config.get("pose_identity", "raw"),
                           config.get("max_pose_hypotheses", 2), config.get("person_region"), config.get("pose_sampling", "bounded"))
        if clock_path is not None:
            clock_raw = _read(clock_path, deadline)
            kept["native-clock.json"] = clock_raw
            pins.append((clock_path, _sha(clock_raw)))
            recovery_implementation = (*recovery_implementation, "source_clock.py")
            if CAPTURE_FILES & kept.keys():
                recovery_implementation = (*recovery_implementation, "source_clock_capture.py")
        if native_timing:
            if CAPTURE_FILES & kept.keys():
                raise ValueError("Native crop capture cannot replace inherited capture evidence")
            recovery_implementation = (*recovery_implementation, "source_clock.py", "source_clock_capture.py")
        snapshots, implementation = _implementation(deadline, recovery_implementation)
        _write(target, {**kept, "inputs/source-manifest.json": manifest_raw,
                        "implementation.json": _bytes(implementation),
                        **{"implementation-source/" + name: raw for name, raw in snapshots.items()}}, deadline)
        if native_timing:
            from weft.source_clock_capture import collect_native_clock

            source_sha = source_meta["sha256"]
            _write(target, {"decoder-clock.json": _bytes({"schema_version": 1, "source_sha256": source_sha,
                "complete": True, "rows": decoder_clock_rows})}, deadline)
            # Native capture cannot admit poses that fail the unchanged crop derivation and quality gates.
            _validated_pose(kept, options, source_sha, deadline=deadline)
            if progress:
                progress("native_clock", {"message": "Verifying original presentation times for selected crop poses"})
            binding, capture, log = collect_native_clock(target / source_name, kept["evidence.json"],
                decoder_clock_rows, max_seconds=min(20, max(.01, deadline - time.monotonic())))
            _write(target, {"native-clock.json": _bytes(binding), "native-clock-capture.json": _bytes(capture),
                            "native-clock-decode.log": log}, deadline)
            del binding, capture, log
        del kept, snapshots, raw
        return _finish(target, options, source_name, "sealed_pose_evidence", _sha(manifest_raw), pins,
                       implementation, started, deadline, progress, native_capture_performed=native_timing)
    except Exception as error:
        _failure(target, error, started)
        raise
