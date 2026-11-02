"""Reject unusable shared-scene inputs before local pose inference starts."""

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import time

from weft import exercise_body_source, exercise_vision
from weft.equipment_proposal_config import validate_proposal_config
from weft.shared_world_evidence import _verify_document_clock
from weft.shared_world_packaging import read_package_file, read_package_manifest, validate_package_layout
from weft.source_equivalence import load_source_equivalence
from weft.world_evidence import _check_tree, load_world_evidence


PROFILE = "exercise-scene-inputs-v1"
HUMAN_PROFILE = "exercise-person-inputs-v1"
SUPPORT_PROFILE = "exercise-static-support-inputs-v1"
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
MAX_RECOVERY_BODY_SECONDS = 240
RECOVERY_STAGE_SECONDS = 30
RECOVERY_METHOD = "selective-crop"
SOURCE_NOTE = "Rights not independently verified; user-provided local video"
REJECTED = {"failed", "error", "rejected", "cancelled", "canceled", "unsupported"}
IMPLEMENTATION_FILES = tuple(dict.fromkeys((*exercise_body_source.IMPLEMENTATION_FILES,
    "exercise_scene_inputs.py", "equipment_proposal.py", "equipment_proposal_config.py", "shared_world_pipeline.py",
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
    evidence_run: Path | None
    config_path: Path | None
    pose_model: Path
    pose_mode: str
    pose_identity: str
    pose_recovery: str | None
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
        if root is not None:
            _not_failed(_path(root))
    for pin in inputs._pins:
        _check(deadline)
        path = _path(pin.path)
        raw = read_package_file(path.parent, path.name, max_bytes=max(pin.size, 1), expected_sha256=pin.sha256)
        if len(raw) != pin.size:
            raise ValueError("Scene input size changed after preflight")
    for root in (inputs.evidence_run, *inputs.source_mappings):
        if root is not None:
            _not_failed(_path(root))
    _check(deadline)


def recheck_exercise_scene_inputs(inputs, *, deadline):
    """Recheck the consumed inputs before later stages or final assembly publication."""
    if not isinstance(inputs, RawExerciseInputs):
        raise ValueError("Scene input recheck requires verified raw inputs")
    bounded = _deadline(MAX_PREFLIGHT_SECONDS, MAX_PREFLIGHT_SECONDS, deadline)
    _recheck(inputs, bounded)


def _pose_options(pose_mode, pose_identity, pose_recovery, source_note):
    if pose_recovery is not None and pose_recovery != RECOVERY_METHOD:
        raise ValueError("Pose recovery must be omitted or explicitly selective-crop")
    if pose_recovery is not None and (pose_mode not in (None, "image") or pose_identity not in (None, "raw")):
        raise ValueError("Selective-crop recovery requires IMAGE mode and raw identity; conflicting options are not replaced")
    pose_mode = ("image" if pose_recovery is not None else "video") if pose_mode is None else pose_mode
    pose_identity = "raw" if pose_identity is None else pose_identity
    if (not isinstance(pose_mode, str) or not isinstance(pose_identity, str)
            or pose_mode not in {"video", "image"} or pose_identity not in {"raw", "temporal"}):
        raise ValueError("Pose options require video/image and explicit raw/temporal identity")
    if not isinstance(source_note, str) or len(source_note) > 4000:
        raise ValueError("Source provenance note must be at most 4000 characters")
    return pose_mode, pose_identity


def _model_pins(reader, model, pose_recovery, dependencies):
    model_raw = reader.read(model)
    model_sha, model_size = _sha(model_raw), len(model_raw)
    del model_raw
    provenance = exercise_vision.model_provenance(model)
    if provenance.get("sha256") != model_sha or provenance.get("bytes") != model_size:
        raise ValueError("Pose model changed during preflight")
    if pose_recovery is not None:
        full = exercise_vision.MODEL_VARIANTS["full"]
        if (model_sha != full["sha256"] or model_size != full["bytes"]
                or provenance.get("variant") != "full" or provenance.get("model_version") != "float16/1"
                or provenance.get("delegate") != "CPU" or provenance.get("network_inference") is not False):
            raise ValueError("Selective-crop recovery requires the exact pinned official Full CPU model")
        from weft import pose_crop_pipeline, pose_source_recovery

        dependencies = tuple(dict.fromkeys((*dependencies, *pose_crop_pipeline.IMPLEMENTATION_FILES,
                                            *pose_source_recovery.IMPLEMENTATION_FILES)))
    implementation = {name: _sha(reader.read(Path(__file__).with_name(name))) for name in dependencies}
    return provenance, implementation


def _ready_inputs(reader, receipt, *, video, evidence, config_path, model, pose_mode, pose_identity,
                  pose_recovery, source_note, mappings, source_sha, deadline):
    if pose_recovery is not None:
        receipt["pose_recovery"] = pose_recovery
        receipt["recovery_contract"] = {
            "recipe": "Pinned Full / IMAGE / raw, then one source-only selective-crop pass",
            "maximum_combined_seconds": MAX_RECOVERY_BODY_SECONDS,
            "stage_seconds": {"pose_proposals": MAX_BODY_SECONDS, "pose_recovery": RECOVERY_STAGE_SECONDS,
                              "body_source": RECOVERY_STAGE_SECONDS},
            "body_gate": "Existing body-source admission is unchanged; recovery does not guarantee a body or world",
            "retries": 0, "model_selection": False, "task_or_geometry_feedback": False}
    encoded = _bytes(receipt)
    if len(encoded) > MAX_RECEIPT_BYTES:
        raise ValueError("Scene input receipt exceeds its bounded output size")
    result = object.__new__(RawExerciseInputs)
    for name, value in {"video": video, "evidence_run": evidence, "config_path": config_path, "pose_model": model,
                        "pose_mode": pose_mode, "pose_identity": pose_identity, "source_note": source_note,
                        "pose_recovery": pose_recovery, "source_mappings": mappings, "source_sha256": source_sha,
                        "_pins": tuple(reader.pins.values()), "_receipt": encoded}.items():
        object.__setattr__(result, name, value)
    _recheck(result, deadline)
    return result


def preflight_human_scene_inputs(video, *, pose_model, pose_mode=None, pose_identity=None, pose_recovery=None,
                                source_note=SOURCE_NOTE, max_seconds=30, deadline=None):
    """Declare a person/ground scope against exact source bytes, not detected absence."""
    return _preflight_declared_scene_inputs(video, pose_model=pose_model, pose_mode=pose_mode,
        pose_identity=pose_identity, pose_recovery=pose_recovery, source_note=source_note,
        max_seconds=max_seconds, deadline=deadline)


def preflight_support_scene_inputs(video, support_priors, *, pose_model, pose_mode=None, pose_identity=None,
                                  pose_recovery=None, source_note=SOURCE_NOTE, max_seconds=30, deadline=None):
    """Pin authored supports to the video without declaring equipment absent or constructing a body."""
    return _preflight_declared_scene_inputs(video, pose_model=pose_model, pose_mode=pose_mode,
        pose_identity=pose_identity, pose_recovery=pose_recovery, source_note=source_note,
        max_seconds=max_seconds, deadline=deadline, support_priors=support_priors)


def _preflight_declared_scene_inputs(video, *, pose_model, pose_mode, pose_identity, pose_recovery,
                                     source_note, max_seconds, deadline, support_priors=None):
    from weft.person_inventory import make_person_inventory

    started = time.monotonic()
    deadline = _deadline(max_seconds, MAX_PREFLIGHT_SECONDS, deadline)
    _check(deadline)
    pose_mode, pose_identity = _pose_options(pose_mode, pose_identity, pose_recovery, source_note)
    video, model = map(_path, (video, pose_model))
    if video.suffix.lower() not in exercise_body_source.SUFFIXES:
        raise ValueError("Provide a supported local video container")
    reader = _Reader(deadline)
    support, support_path, support_sha = None, None, None
    if support_priors is not None:
        from weft.exercise_support_inputs import MAX_CONFIG_BYTES as SUPPORT_BYTES, validate_support_priors

        support_path = _path(support_priors)
        support_raw = reader.read(support_path, SUPPORT_BYTES)
        support = validate_support_priors(support_raw)
        support_sha = _sha(support_raw)
    raw = reader.read(video)
    if not raw:
        raise ValueError("Raw video input cannot be empty")
    source_sha = _sha(raw)
    del raw
    if support is not None and support["source_sha256"] != source_sha:
        raise ValueError("Static-support priors name a different original video")
    dependencies = tuple(dict.fromkeys((*exercise_body_source.IMPLEMENTATION_FILES,
        "exercise_scene_inputs.py", "person_inventory.py", "shared_world_human.py",
        *(("exercise_support_inputs.py", "shared_world_support.py", "shared_support_task.py") if support is not None else ()))))
    provenance, implementation = _model_pins(reader, model, pose_recovery, dependencies)
    receipt = {"schema_version": 1, "profile": SUPPORT_PROFILE if support is not None else HUMAN_PROFILE,
        "status": "input_preflight_passed",
        "scene_scope": "human_static_supports" if support is not None else "human_only",
        **({"support_priors_sha256": support_sha, "support_priors": support} if support is not None
           else {"inventory": make_person_inventory(source_sha)}),
        "source_sha256": source_sha, "source_bytes": reader.pins[video].size,
        "pose_options": {"pose_mode": pose_mode, "pose_identity": pose_identity},
        "pose_model": provenance, "source_note": source_note, "implementation_sha256": implementation,
        "input_pins": [{"path": str(pin.path), "sha256": pin.sha256, "bytes": pin.size} for pin in reader.pins.values()],
        "input_bytes": reader.total, "input_files": len(reader.pins),
        "limits": {"maximum_input_bytes": MAX_INPUT_BYTES, "maximum_files": MAX_FILES,
                   "maximum_file_bytes": MAX_FILE_BYTES, "maximum_seconds": max_seconds},
        "work": {"video_decodes": 0, "inference_calls": 0, "physics_steps": 0, "api_calls": 0},
        "scope": ("Source-bound body with explicitly authored static supports/load windows; no equipment-absence or recovered-contact claim. Geometry, scale and ground remain unverified."
                  if support is not None else "User declares one selected person, ground-only support and no task-relevant equipment or obstacles. Absence is not detector-verified; scale, ground and depth remain priors."),
        "remaining_checks": ["Actual video metadata and decode", "Existing body/pose quality admission",
                             "Complete source clock and reference binding", "Unchanged initial geometry and human task gates"],
        "deadline_scope": "Cooperative around bounded local reads; the body stage retains its existing native-call limits.",
        "elapsed_seconds": time.monotonic() - started}
    return _ready_inputs(reader, receipt, video=video, evidence=None, config_path=support_path, model=model,
        pose_mode=pose_mode, pose_identity=pose_identity, pose_recovery=pose_recovery, source_note=source_note,
        mappings=(), source_sha=source_sha, deadline=deadline)


def preflight_exercise_scene_inputs(video, evidence_run, config_path, *, pose_model,
                                   pose_mode=None, pose_identity=None, pose_recovery=None, source_note=SOURCE_NOTE,
                                   source_mappings=(), max_seconds=30, deadline=None):
    """Check source, model and exact construction frames without decoding or inference."""
    started = time.monotonic()
    deadline = _deadline(max_seconds, MAX_PREFLIGHT_SECONDS, deadline)
    _check(deadline)
    pose_mode, pose_identity = _pose_options(pose_mode, pose_identity, pose_recovery, source_note)
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
        if 'frame_selection' in choice:
            candidates = [row for row in filtered['observations'] if row['object_id'] == choice['object_id']
                          and row['admission'] == 'included' and row['visibility'] in {'detected', 'partial'}]
            if not candidates:
                raise ValueError('Automatic equipment selection requires visible construction candidates; body-reference eligibility remains deferred')
            for row in candidates:
                _check(deadline)
                box = row['measurements'].get('box', {})
                if (len(row['evidence_refs']) != 1 or (box.get('coordinate_system'), box.get('unit'), box.get('status'))
                        != ('image', 'pixels', 'inferred') or box['value'][2] <= 0 or box['value'][3] <= 0):
                    raise ValueError('Automatic equipment candidates require one inferred construction box each')
                payload = evidence_rows[row['evidence_refs'][0]['evidence_id']]['payload']
                if str(payload.get('geometry_status', '')).startswith('authored_initial_ROI'):
                    raise ValueError('Appearance-translated authored ROIs are not inferred equipment boundaries')
            selected.append({**choice, 'observation_document_sha256': document_sha,
                             'frame_selection_status': 'deferred_until_body_references'})
            continue
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

    dependencies = (*IMPLEMENTATION_FILES, 'equipment_box_consensus.py') if any('frame_selection' in row for row in config['equipment']) else IMPLEMENTATION_FILES
    provenance, implementation = _model_pins(reader, model, pose_recovery, dependencies)
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
    return _ready_inputs(reader, receipt, video=video, evidence=evidence, config_path=config_path, model=model,
        pose_mode=pose_mode, pose_identity=pose_identity, pose_recovery=pose_recovery, source_note=source_note,
        mappings=mappings, source_sha=source_sha, deadline=deadline)


def _generated_pin(root, profile, status, deadline):
    _check(deadline)
    root = _path(root)
    _not_failed(root)
    raw, files = read_package_manifest(root)
    manifest = _json(raw)
    if (manifest.get("profile") != profile or manifest.get("status") != status or manifest.get("simulation_ready") is not False
            or "failure.json" in files or len(files) + 1 > exercise_body_source.MAX_FILES):
        raise ValueError("Recovery composition requires an unrejected bounded sealed stage")
    _check(deadline)
    return root, raw, files


def _recheck_generated(pins, deadline, *, complete=False):
    for root, manifest, files in pins:
        _check(deadline)
        root = _path(root)
        _not_failed(root)
        read_package_file(root, "manifest.json", max_bytes=MAX_MANIFEST_BYTES, expected_sha256=_sha(manifest))
        if complete:
            total = len(manifest)
            for name, expected in files.items():
                _check(deadline)
                remaining = exercise_body_source.MAX_BYTES - total
                if remaining < 1:
                    raise ValueError("Generated recovery stage exceeds its retained-byte bound")
                raw = read_package_file(root, name, max_bytes=min(MAX_FILE_BYTES, remaining), expected_sha256=expected)
                total += len(raw)
        _not_failed(root)
    _check(deadline)


def prepare_exercise_scene_body(inputs, out, *, pose_proposals_out=None, recovered_pose_out=None,
                                max_seconds=180, deadline=None, progress=None, native_timing=False):
    """Prepare one existing body source; changed inputs stop the enclosing scene assembly."""
    if not isinstance(inputs, RawExerciseInputs):
        raise ValueError("Raw body preparation requires verified scene inputs")
    if type(native_timing) is not bool:
        raise ValueError("Native timing must be explicitly enabled or disabled")
    recovering = inputs.pose_recovery == RECOVERY_METHOD
    if recovering and native_timing:
        raise ValueError("Native timing capture with selective-crop recovery is not yet supported")
    deadline = _deadline(max_seconds, MAX_RECOVERY_BODY_SECONDS if recovering else MAX_BODY_SECONDS, deadline)
    target = _path(out)
    sources = tuple(path for path in (inputs.video, inputs.pose_model, inputs.evidence_run, inputs.config_path,
                                     *inputs.source_mappings) if path is not None)
    outputs = [target]
    if recovering:
        if pose_proposals_out is None or recovered_pose_out is None:
            raise ValueError("Selective-crop requires explicit proposal and recovered-pose output paths")
        outputs += [_path(pose_proposals_out), _path(recovered_pose_out)]
        if any(path.parent != target.parent for path in outputs):
            raise ValueError("Recovery proposal, recovered-pose and body outputs must be sibling packages")
    elif pose_proposals_out is not None or recovered_pose_out is not None:
        raise ValueError("Recovery output paths require explicit selective-crop recovery")
    if (any(path.exists() or path.is_symlink() for path in outputs)
            or any(path.is_relative_to(source) or source.is_relative_to(path) for path in outputs for source in sources)
            or any(a.is_relative_to(b) or b.is_relative_to(a) for i, a in enumerate(outputs) for b in outputs[i + 1:])):
        raise ValueError("Body output must be new and outside every input")
    _recheck(inputs, deadline)
    if recovering:
        return _prepare_recovered_body(inputs, target, outputs[1], outputs[2], deadline, progress)
    remaining = deadline - time.monotonic()
    if remaining < 1:
        raise TimeoutError("Raw body preparation has no remaining shared time budget")
    body = exercise_body_source.prepare_exercise_body_source(inputs.video, target,
        pose_model=inputs.pose_model, pose_mode=inputs.pose_mode, pose_identity=inputs.pose_identity,
        source_note=inputs.source_note, max_seconds=min(MAX_BODY_SECONDS, remaining), progress=progress,
        **({"native_timing": True} if native_timing else {}))
    _recheck(inputs, deadline)
    return body


def _prepare_recovered_body(inputs, target, proposals, recovered, deadline, progress):
    from weft import pose_crop_pipeline, pose_source_recovery

    pins = []

    def guard(*, complete=False):
        _recheck(inputs, deadline)
        _recheck_generated(pins, deadline, complete=complete)

    def emit(stage, status, message, **details):
        if progress is not None:
            progress(stage, {"status": status, "message": message, **details})
        guard()

    def remaining(maximum, reserve=0):
        left = deadline - time.monotonic() - reserve
        if left < 1:
            raise TimeoutError("Recovery composition has no remaining shared time budget")
        return min(maximum, left)

    def updates(prefix):
        if progress is None:
            return None

        def update(stage, detail):
            raw = _bytes(detail)
            if len(raw) > MAX_RECEIPT_BYTES:
                raise ValueError("Recovery progress exceeds its metadata bound")
            progress(prefix + "_" + stage, json.loads(raw))
        return update

    def keep(root, expected, profile, status):
        if _path(root) != expected:
            raise ValueError("Recovery producer returned a different output directory")
        pin = _generated_pin(expected, profile, status, deadline)
        pins.append(pin)
        return _sha(pin[1])

    # Reserve the existing source-verification and body stages before new inference.
    proposal_seconds = remaining(MAX_BODY_SECONDS, 2 * RECOVERY_STAGE_SECONDS)
    emit("pose_proposals", "running", "Estimate original Full IMAGE poses and one bounded selective-crop pass")
    result = pose_crop_pipeline.prepare_pose_crop_proposals(inputs.video, proposals,
        pose_model=inputs.pose_model, source_note=inputs.source_note,
        max_seconds=proposal_seconds, deadline=deadline, progress=updates("pose_proposals"))
    seal = keep(result, proposals, pose_crop_pipeline.PROFILE, "source_evidence_only")
    emit("pose_proposals", "ready", "Original poses, crop candidates and unchanged selection decisions are sealed",
         manifest_sha256=seal)

    emit("pose_recovery", "running", "Verify the saved crop derivation without new inference")
    result = pose_source_recovery.prepare_pose_source_recovery(proposals, recovered, source_video=inputs.video,
        max_seconds=remaining(RECOVERY_STAGE_SECONDS, RECOVERY_STAGE_SECONDS))
    seal = keep(result, recovered, pose_source_recovery.PROFILE, "source_evidence_only")
    emit("pose_recovery", "ready", "Verified source pose evidence is sealed; body admission is still required",
         manifest_sha256=seal)

    emit("body_source", "running", "Apply the existing body and movement gates to the verified source poses")
    result = exercise_body_source.prepare_exercise_body_source_from_evidence(recovered, target,
        max_seconds=remaining(RECOVERY_STAGE_SECONDS), progress=updates("body_source"))
    seal = keep(result, target, exercise_body_source.PROFILE, "construction_only")
    emit("body_source", "ready", "One construction-only body is sealed; no world, task or intermediate lab was created",
         manifest_sha256=seal)
    guard(complete=True)
    return result
