"""Reject unusable shared-scene inputs before local pose inference starts."""

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import time

from weft import exercise_body_source, exercise_vision
from weft.equipment_proposal import validate_proposal_config
from weft.shared_world_evidence import _verify_document_clock
from weft.shared_world_packaging import read_package_file, validate_package_layout
from weft.source_equivalence import load_source_equivalence
from weft.world_evidence import _check_tree, load_world_evidence


PROFILE = "exercise-scene-inputs-v1"
MAX_INPUT_BYTES = 192 * 1024 * 1024
MAX_FILE_BYTES = 64 * 1024 * 1024
MAX_EVIDENCE_BYTES = 64 * 1024 * 1024
MAX_CONFIG_BYTES = 64 * 1024
MAX_MANIFEST_BYTES = 8 * 1024 * 1024
MAX_MANIFEST_FILES = 100_000
MAX_RECEIPT_BYTES = 256 * 1024
MAX_FILES = 320
MAX_SOURCE_FILES = 256
MAX_MAPPINGS = 7
MAX_PREFLIGHT_SECONDS = 30
MAX_BODY_SECONDS = 180
SOURCE_NOTE = "Rights not independently verified; user-provided local video"
REJECTED = {"failed", "error", "rejected", "cancelled", "canceled", "unsupported"}
IMPLEMENTATION_FILES = tuple(dict.fromkeys((*exercise_body_source.IMPLEMENTATION_FILES,
    "exercise_scene_inputs.py", "equipment_proposal.py", "shared_world_pipeline.py",
    "shared_world_reconstruction.py", "shared_world_evidence.py", "source_equivalence.py")))


def _bytes(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _json(raw):
    def unique(pairs):
        result = {}
        for name, value in pairs:
            if name in result:
                raise ValueError("Scene input JSON contains duplicate keys")
            result[name] = value
        return result

    def invalid(value):
        raise ValueError("Scene input JSON contains a non-finite number")

    try:
        result = json.loads(raw, object_pairs_hook=unique, parse_constant=invalid)
    except (UnicodeError, json.JSONDecodeError, RecursionError) as error:
        raise ValueError("Scene inputs require strict bounded JSON") from error
    if not isinstance(result, dict):
        raise ValueError("Scene input JSON must be an object")
    _check_tree(result)
    return result


def _path(value):
    path = Path(value).absolute()
    if ".." in path.parts or any(part.is_symlink() for part in (path, *path.parents)):
        raise ValueError("Scene input paths must not pass through symlinks")
    return path


def _deadline(max_seconds, maximum, deadline):
    if (type(max_seconds) not in (int, float) or not math.isfinite(max_seconds)
            or not 1 <= max_seconds <= maximum):
        raise ValueError(f"Scene input stage requires a 1–{maximum} second bound")
    if deadline is not None and (type(deadline) not in (int, float) or not math.isfinite(deadline)):
        raise ValueError("Scene input deadline must be a finite monotonic time")
    result = time.monotonic() + max_seconds
    return min(result, deadline) if deadline is not None else result


def _check(deadline):
    if time.monotonic() >= deadline:
        raise TimeoutError("Scene input preparation exceeded its shared deadline")


def _not_failed(root):
    marker = root / "failure.json"
    if marker.exists() or marker.is_symlink():
        raise ValueError("Failed equipment or mapping evidence cannot seed a shared scene")


@dataclass(frozen=True)
class _Pin:
    path: Path
    sha256: str
    size: int


@dataclass(frozen=True, init=False)
class RawExerciseInputs:
    video: Path
    evidence_run: Path
    config_path: Path
    pose_model: Path
    pose_mode: str
    pose_identity: str
    source_note: str
    source_mappings: tuple[Path, ...]
    source_sha256: str
    _pins: tuple[_Pin, ...]
    _receipt: bytes

    def __init__(self):
        raise TypeError("Use preflight_exercise_scene_inputs to validate raw inputs")

    def receipt(self):
        """Return a defensive JSON receipt; no raw observation values cross this boundary."""
        return json.loads(self._receipt)


class _Reader:
    def __init__(self, deadline):
        self.deadline, self.pins, self.total = deadline, {}, 0

    def pin(self, path, raw):
        path = _path(path)
        pin = _Pin(path, _sha(raw), len(raw))
        old = self.pins.get(path)
        if old is not None:
            if old != pin:
                raise ValueError("Scene input changed during preflight")
            return
        if (len(raw) > MAX_FILE_BYTES or self.total + len(raw) > MAX_INPUT_BYTES
                or len(self.pins) + 1 > MAX_FILES):
            raise ValueError("Scene inputs exceed their aggregate byte/file bound")
        self.pins[path] = pin
        self.total += len(raw)

    def read(self, path, maximum=MAX_FILE_BYTES, expected=None):
        _check(self.deadline)
        path = _path(path)
        remaining = MAX_INPUT_BYTES - self.total
        if path in self.pins:
            remaining += self.pins[path].size
        limit = min(maximum, remaining)
        if limit < 1 or (path not in self.pins and len(self.pins) >= MAX_FILES):
            raise ValueError("Scene inputs exceed their aggregate byte/file bound")
        raw = read_package_file(path.parent, path.name, max_bytes=limit, expected_sha256=expected)
        self.pin(path, raw)
        _check(self.deadline)
        return raw

    def manifest(self, root):
        _check(self.deadline)
        _not_failed(root)
        raw = self.read(root / "manifest.json", MAX_MANIFEST_BYTES)
        document = _json(raw)
        files = document.get("files")
        if (not isinstance(files, dict) or not 1 <= len(files) <= MAX_MANIFEST_FILES
                or document.get("status") in REJECTED):
            raise ValueError("Equipment or mapping manifest is failed or exceeds its file bound")
        validate_package_layout(files)
        if ("manifest.json" in files or any(name.startswith("manifest.json/") for name in files)
                or any(not isinstance(value, str) or len(value) != 64
                       or any(char not in "0123456789abcdef" for char in value) for value in files.values())):
            raise ValueError("Equipment manifest requires exact safe file identities")
        return files


def _recheck(inputs, deadline):
    for root in (inputs.evidence_run, *inputs.source_mappings):
        _not_failed(_path(root))
    for pin in inputs._pins:
        _check(deadline)
        path = _path(pin.path)
        raw = read_package_file(path.parent, path.name, max_bytes=max(pin.size, 1), expected_sha256=pin.sha256)
        if len(raw) != pin.size:
            raise ValueError("Scene input size changed after preflight")
    for root in (inputs.evidence_run, *inputs.source_mappings):
        _not_failed(_path(root))
    _check(deadline)


def recheck_exercise_scene_inputs(inputs, *, deadline):
    """Recheck the consumed inputs before later stages or final assembly publication."""
    if not isinstance(inputs, RawExerciseInputs):
        raise ValueError("Scene input recheck requires verified raw inputs")
    bounded = _deadline(MAX_PREFLIGHT_SECONDS, MAX_PREFLIGHT_SECONDS, deadline)
    _recheck(inputs, bounded)


def preflight_exercise_scene_inputs(video, evidence_run, config_path, *, pose_model,
                                   pose_mode="video", pose_identity="raw", source_note=SOURCE_NOTE,
                                   source_mappings=(), max_seconds=30, deadline=None):
    """Check source, model and exact construction frames without decoding or inference."""
    started = time.monotonic()
    deadline = _deadline(max_seconds, MAX_PREFLIGHT_SECONDS, deadline)
    _check(deadline)
    if (not isinstance(pose_mode, str) or not isinstance(pose_identity, str)
            or pose_mode not in {"video", "image"} or pose_identity not in {"raw", "temporal"}):
        raise ValueError("Pose options require video/image and explicit raw/temporal identity")
    if not isinstance(source_note, str) or len(source_note) > 4000:
        raise ValueError("Source provenance note must be at most 4000 characters")
    if not isinstance(source_mappings, (tuple, list)) or len(source_mappings) > MAX_MAPPINGS:
        raise ValueError("Use at most seven verified source mappings")
    video, evidence, config_path, model = map(_path, (video, evidence_run, config_path, pose_model))
    mappings = tuple(map(_path, source_mappings))
    if len(set(mappings)) != len(mappings):
        raise ValueError("Source mapping directories must be distinct")
    if video.suffix.lower() not in exercise_body_source.SUFFIXES:
        raise ValueError("Provide a supported local video container")
    reader = _Reader(deadline)
    config_raw = reader.read(config_path, MAX_CONFIG_BYTES)
    config = validate_proposal_config(_json(config_raw))
    video_raw = reader.read(video)
    if not video_raw:
        raise ValueError("Raw video input cannot be empty")
    source_sha = _sha(video_raw)
    del video_raw
    if config["source_sha256"] != source_sha:
        raise ValueError("Equipment configuration refers to a different original video")

    declared = reader.manifest(evidence)
    names = [name for name in ("scene-observations.json", "evidence/scene-observations.json") if name in declared]
    if not names:
        raise ValueError("Equipment requires sealed scene-observations.json")
    raw = reader.read(evidence / names[0], MAX_EVIDENCE_BYTES, declared[names[0]])
    for name in names[1:]:
        if reader.read(evidence / name, MAX_EVIDENCE_BYTES, declared[name]) != raw:
            raise ValueError("Equipment package contains conflicting observation documents")
    if "report.json" in declared:
        report = _json(reader.read(evidence / "report.json", 1024 * 1024, declared["report.json"]))
        if report.get("status") in REJECTED:
            raise ValueError("Equipment report is failed or unsupported")
    document_sha = _sha(raw)
    registry = load_world_evidence({document_sha: raw})
    document = _json(raw)
    source = document["source"]
    count = source["decoded_frames"]
    if (count > 1800 or source.get("decode_interval", [0, count - 1]) != [0, count - 1]
            or source.get("source_complete") is False):
        raise ValueError("Equipment evidence must preserve the complete bounded source interval")
    if document["schema_version"] == 3:
        times = source["frame_times_s"]
        if times[-1] - times[0] > 60:
            raise ValueError("Equipment source exceeds the 60-second video bound")
    elif not 5 <= source["fps"] <= 240 or count / source["fps"] > 60:
        raise ValueError("Equipment source clock exceeds the raw body profile bounds")

    equivalents, mapping_receipts = {}, []
    proof_bytes, proof_files = len(raw), 1
    for path in mappings:
        _check(deadline)
        _not_failed(path)
        remaining = min(MAX_EVIDENCE_BYTES - proof_bytes, MAX_INPUT_BYTES - reader.total)
        slots = min(MAX_SOURCE_FILES - proof_files, MAX_FILES - len(reader.pins), 129)
        if remaining < 1 or slots < 2:
            raise ValueError("Combined equipment and mapping evidence exceeds its bound")
        item = load_source_equivalence(path, max_input_bytes=remaining, max_files=slots, deadline=deadline)
        if (item.original_sha256 != source_sha or item.derivative_sha256 == source_sha
                or item.derivative_sha256 in equivalents):
            raise ValueError("Source mappings must bind distinct derivatives to the exact original video")
        equivalents[item.derivative_sha256] = item
        proof_bytes += item.total_input_bytes
        proof_files += len(item.file_names)
        for name in item.file_names:
            reader.pin(path / name, item.raw_bytes(name))
        mapping_receipts.append({"path": str(path), **item.inspection()})
    media_sha = source["sha256"]
    if media_sha != source_sha and media_sha not in equivalents:
        raise ValueError("Equipment video differs from the original without a verified source mapping")
    if set(equivalents) != ({media_sha} if media_sha != source_sha else set()):
        raise ValueError("Do not attach unused source mapping bundles")
    construction = json.loads(registry.construction_bytes(
        source_aliases={sha: source_sha for sha in equivalents}))
    filtered = construction["sources"][0]
    if media_sha in equivalents:
        _verify_document_clock(filtered, equivalents[media_sha], derivative=True)
    indexed = {(row["object_id"], row["frame"]): row for row in filtered["observations"]}
    selected, evidence_rows = [], {row["id"]: row for row in document["evidence"]}
    for choice in config["equipment"]:
        row = indexed.get((choice["object_id"], choice["source_frame"]))
        if (row is None or row["admission"] != "included" or row["visibility"] not in {"detected", "partial"}
                or len(row["evidence_refs"]) != 1):
            raise ValueError("Selected equipment ID/frame is missing, withheld or lacks one construction reference")
        box = row["measurements"].get("box", {})
        if (box.get("coordinate_system"), box.get("unit"), box.get("status")) != ("image", "pixels", "inferred"):
            raise ValueError("Selected equipment requires an inferred image box, not authored coordinates")
        if box["value"][2] <= 0 or box["value"][3] <= 0:
            raise ValueError("Selected equipment box must have positive extents")
        identifier = row["evidence_refs"][0]["evidence_id"]
        payload = evidence_rows[identifier]["payload"]
        if str(payload.get("geometry_status", "")).startswith("authored_initial_ROI"):
            raise ValueError("Appearance-translated authored ROIs are not inferred equipment boundaries")
        selected.append({**choice, "evidence_id": identifier, "time_s": row["time_s"],
                         "observation_document_sha256": document_sha})

    model_raw = reader.read(model)
    model_sha, model_size = _sha(model_raw), len(model_raw)
    del model_raw
    provenance = exercise_vision.model_provenance(model)
    if provenance.get("sha256") != model_sha or provenance.get("bytes") != model_size:
        raise ValueError("Pose model changed during preflight")
    implementation = {}
    for name in IMPLEMENTATION_FILES:
        path = Path(__file__).with_name(name)
        implementation[name] = _sha(reader.read(path))
    receipt = {"schema_version": 1, "profile": PROFILE, "status": "input_preflight_passed",
        "source_sha256": source_sha, "source_bytes": reader.pins[video].size,
        "equipment_manifest_sha256": reader.pins[evidence / "manifest.json"].sha256,
        "equipment_document_sha256": document_sha, "equipment_source_sha256": media_sha,
        "equipment_source_slots": count, "selected_equipment": selected,
        "config_sha256": _sha(config_raw), "source_mappings": mapping_receipts,
        "pose_options": {"pose_mode": pose_mode, "pose_identity": pose_identity},
        "pose_model": provenance, "source_note": source_note,
        "implementation_sha256": implementation,
        "input_pins": [{"path": str(pin.path), "sha256": pin.sha256, "bytes": pin.size} for pin in reader.pins.values()],
        "input_bytes": reader.total, "input_files": len(reader.pins),
        "equipment_mapping_bytes": proof_bytes, "equipment_mapping_files": proof_files,
        "limits": {"maximum_input_bytes": MAX_INPUT_BYTES, "maximum_files": MAX_FILES,
                   "maximum_file_bytes": MAX_FILE_BYTES, "maximum_equipment_mapping_bytes": MAX_EVIDENCE_BYTES,
                   "maximum_equipment_mapping_files": MAX_SOURCE_FILES, "maximum_seconds": max_seconds,
                   "maximum_manifest_bytes": MAX_MANIFEST_BYTES, "maximum_manifest_declarations": MAX_MANIFEST_FILES},
        "work": {"video_decodes": 0, "inference_calls": 0, "physics_steps": 0, "api_calls": 0},
        "scope": "Only consumed artifacts are verified: required sealed equipment observations/report and complete mapping proofs; not every declared member or geometry/body/task admission.",
        "remaining_checks": ["Actual video metadata and decode", "Requested frame survives pose inference",
                             "Body/equipment clock and dimension agreement", "Unchanged strict geometry/task admission"],
        "deadline_scope": "Cooperative around bounded local reads; the body stage retains its existing native-call limits.",
        "elapsed_seconds": time.monotonic() - started}
    encoded = _bytes(receipt)
    if len(encoded) > MAX_RECEIPT_BYTES:
        raise ValueError("Scene input receipt exceeds its bounded output size")
    result = object.__new__(RawExerciseInputs)
    for name, value in {"video": video, "evidence_run": evidence, "config_path": config_path, "pose_model": model,
                        "pose_mode": pose_mode, "pose_identity": pose_identity, "source_note": source_note,
                        "source_mappings": mappings, "source_sha256": source_sha,
                        "_pins": tuple(reader.pins.values()), "_receipt": encoded}.items():
        object.__setattr__(result, name, value)
    _recheck(result, deadline)
    return result


def prepare_exercise_scene_body(inputs, out, *, max_seconds=180, deadline=None, progress=None):
    """Prepare one existing body source; changed inputs stop the enclosing scene assembly."""
    if not isinstance(inputs, RawExerciseInputs):
        raise ValueError("Raw body preparation requires verified scene inputs")
    deadline = _deadline(max_seconds, MAX_BODY_SECONDS, deadline)
    target = _path(out)
    sources = (inputs.video, inputs.pose_model, inputs.evidence_run, inputs.config_path, *inputs.source_mappings)
    if target.exists() or any(target.is_relative_to(path) or path.is_relative_to(target) for path in sources):
        raise ValueError("Body output must be new and outside every input")
    _recheck(inputs, deadline)
    remaining = deadline - time.monotonic()
    if remaining < 1:
        raise TimeoutError("Raw body preparation has no remaining shared time budget")
    body = exercise_body_source.prepare_exercise_body_source(inputs.video, target,
        pose_model=inputs.pose_model, pose_mode=inputs.pose_mode, pose_identity=inputs.pose_identity,
        source_note=inputs.source_note, max_seconds=min(MAX_BODY_SECONDS, remaining), progress=progress)
    _recheck(inputs, deadline)
    return body
