"""Assemble one inspectable exercise lab from verified body and equipment evidence."""

import hashlib
import json
import math
import os
from pathlib import Path
import stat
import time

from weft.equipment_proposal import (
    MAX_OUTPUT_BYTES as PROPOSAL_BYTES, MAX_OUTPUT_FILES as PROPOSAL_FILES, prepare_equipment_proposal,
)
from weft.io import local_time
from weft.shared_exercise_lab import (
    MAX_OUTPUT_BYTES as LAB_BYTES, MAX_OUTPUT_FILES as LAB_FILES, prepare_shared_exercise_lab,
)
from weft.shared_world_packaging import read_package_file
from weft.shared_world_pipeline import (
    MAX_OUTPUT_BYTES as WORLD_BYTES, MAX_OUTPUT_FILES as WORLD_FILES, construct_source_equipment_run,
)


MAX_FILES = 1600
MAX_BYTES = 384 * 1024 * 1024
METADATA_RESERVE_BYTES = 2 * 1024 * 1024
METADATA_RESERVE_FILES = 8
STAGE_LIMITS = {"body_seed": (512, 128 * 1024 * 1024),
                "equipment_priors": (PROPOSAL_FILES, PROPOSAL_BYTES),
                "shared_world": (WORLD_FILES, WORLD_BYTES), "interactive_lab": (LAB_FILES, LAB_BYTES)}


def _bytes(value):
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def _check_deadline(deadline):
    if time.monotonic() >= deadline:
        raise TimeoutError("Exercise assembly reached its cooperative wall-time bound")


def _write_metadata(root, name, raw):
    root_fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        fd = os.open(name, os.O_WRONLY | os.O_CREAT | os.O_NOFOLLOW | os.O_NONBLOCK, 0o600, dir_fd=root_fd)
        with os.fdopen(fd, "wb") as stream:
            if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
                raise ValueError("Exercise assembly metadata must be a regular file")
            stream.truncate(0)
            stream.write(raw)
    finally:
        os.close(root_fd)


def _inventory(root, deadline, *, immutable_only, limits=None):
    maximum_files, maximum_bytes = limits or (MAX_FILES, MAX_BYTES)
    files, total, directories = {}, 0, [root]
    visited = 0
    while directories:
        _check_deadline(deadline)
        with os.scandir(directories.pop()) as entries:
            for entry in entries:
                _check_deadline(deadline)
                path = Path(entry.path)
                name = path.relative_to(root).as_posix()
                if name in {"result.json", "manifest.json"}:
                    continue
                # Mutable lifecycle files and future sessions never enter the immutable assembly seal.
                if immutable_only and any(name.startswith(lab + "/") and name != lab + "/run"
                        and not name.startswith(lab + "/run/") for lab in ("lab", "learning_workflow/lab")):
                    continue
                info = entry.stat(follow_symlinks=False)
                if stat.S_ISDIR(info.st_mode):
                    visited += 1
                    if visited > maximum_files * 16:
                        raise ValueError("Exercise assembly exceeded its directory bound")
                    directories.append(path)
                elif stat.S_ISREG(info.st_mode):
                    files[name] = info.st_size
                    total += info.st_size
                    if len(files) > maximum_files or total > maximum_bytes:
                        raise ValueError("Exercise assembly exceeded its output file/byte bound")
                else:
                    raise ValueError("Exercise assembly outputs must be regular files without symlinks")
    return files, total


def _reserve_stage(root, name, deadline, *, limits=None, allowance=None):
    maximum_files, maximum_bytes = limits or (MAX_FILES, MAX_BYTES)
    files, total = _inventory(root, deadline, immutable_only=False, limits=limits)
    count, size = allowance or STAGE_LIMITS[name]
    if (len(files) + count + METADATA_RESERVE_FILES > maximum_files or
            total + size + METADATA_RESERVE_BYTES > maximum_bytes):
        raise ValueError(f"Exercise assembly cannot reserve the complete {name} output within its aggregate bound")


def _seal(root, result, *, deadline, started, limits=None):
    maximum_files, maximum_bytes = limits or (MAX_FILES, MAX_BYTES)
    seal_started = time.monotonic()
    inventory, _ = _inventory(root, deadline, immutable_only=True, limits=limits)
    files, total = {}, 0
    for name, size in sorted(inventory.items()):
        _check_deadline(deadline)
        raw = read_package_file(root, name, max_bytes=min(64 * 1024 * 1024, max(1, size)))
        _check_deadline(deadline)
        total += len(raw)
        files[name] = hashlib.sha256(raw).hexdigest()
    result = {**result, "seconds": time.monotonic() - started,
              "sealing_seconds": time.monotonic() - seal_started, "seal_scope": "complete_immutable_outputs"}
    result_raw = _bytes(result)
    files["result.json"] = hashlib.sha256(result_raw).hexdigest()
    manifest = _bytes({"schema_version": 1, "status": result["status"], "seal_scope": result["seal_scope"],
                       "files": files})
    total += len(result_raw) + len(manifest)
    if len(files) + 1 > maximum_files or total > maximum_bytes:
        raise ValueError("Exercise assembly terminal metadata exceeds its aggregate bound")
    _check_deadline(deadline)
    _write_metadata(root, "result.json", result_raw)
    _write_metadata(root, "manifest.json", manifest)
    _check_deadline(deadline)
    return {**result, "manifest_sha256": hashlib.sha256(manifest).hexdigest(),
            "files": len(files) + 1, "bytes": total, "out": str(root)}


def _failure_receipt(root, result, sealing_error, started):
    # Oversized or expired partial output must not prevent a small, independently sealed failure receipt.
    result = {**result, "seconds": time.monotonic() - started, "seal_scope": "failure_receipt_only",
              "unsealed_stage_outputs_preserved": True,
              "sealing_error": {"type": type(sealing_error).__name__, "message": str(sealing_error)[:2000]}}
    raw = _bytes(result)
    if len(raw) > 64 * 1024:
        raise ValueError("Exercise assembly failure receipt exceeds its metadata bound")
    _write_metadata(root, "result.json", raw)
    _write_metadata(root, "manifest.json", _bytes({"schema_version": 1, "status": "failed",
        "seal_scope": "failure_receipt_only", "unsealed_stage_outputs_preserved": True,
        "files": {"result.json": hashlib.sha256(raw).hexdigest()}}))


def prepare_exercise_scene(base_run, evidence_run, config_path, out, *, source_mappings=(),
                           human_run=None, body_scale=None, root_assistance=False, full_reference=False,
                           video=None, pose_model=None, pose_mode=None, pose_identity=None, source_note=None,
                           train_settings=None, max_seconds=None, progress=None):
    """Reuse the standard stages once; unsupported evidence stops before simulation is exposed."""
    if sum(value is not None for value in (base_run, human_run, video)) != 1:
        raise ValueError("Provide exactly one shared-world base, mobile human run or raw video")
    if human_run is None and video is None and body_scale is not None:
        raise ValueError("Body scale applies only to a mobile human run; existing worlds keep their scale")
    if video is None and any(value is not None for value in (pose_model, pose_mode, pose_identity, source_note)):
        raise ValueError("Pose model/options apply only to raw --video input; saved evidence is never reinferred")
    if video is not None and pose_model is None:
        raise ValueError("Raw exercise video requires an explicit local pose model")
    learning, learning_allowance = None, None
    if train_settings is not None:
        from weft.exercise_learning_pipeline import MAX_BYTES as LEARNING_BYTES, MAX_FILES as LEARNING_FILES, validate_train_settings

        learning = validate_train_settings(train_settings)
        learning_allowance = (LEARNING_FILES, LEARNING_BYTES)
    resolved_scale = 1.0 if body_scale is None else body_scale
    if (type(resolved_scale) not in (int, float) or not math.isfinite(resolved_scale)
            or not .5 <= resolved_scale <= 2.):
        raise ValueError("Body scale must be an explicit finite prior in [0.5, 2.0]")
    if type(root_assistance) is not bool:
        raise ValueError("Root assistance must be explicitly enabled or disabled")
    if type(full_reference) is not bool:
        raise ValueError("Full-reference assembly must be explicitly enabled or disabled")
    extended = video is not None or learning is not None
    if max_seconds is None:
        max_seconds = 450 if video is not None and learning is not None else 270 if extended else 90
    maximum_seconds = 480 if extended else 180
    if type(max_seconds) not in (int, float) or not math.isfinite(max_seconds) or not 1 <= max_seconds <= maximum_seconds:
        raise ValueError(f"Exercise assembly requires a cooperative 1–{maximum_seconds} second bound")
    if not isinstance(source_mappings, (tuple, list)) or len(source_mappings) > 7:
        raise ValueError("Use at most seven verified source mapping bundles")
    source, evidence, config, target = map(lambda value: Path(value).absolute(),
                                         (video if video is not None else human_run if human_run is not None else base_run,
                                          evidence_run, config_path, out))
    inputs = [source, evidence, config, *(Path(value).absolute() for value in source_mappings)]
    if pose_model is not None:
        inputs.append(Path(pose_model).absolute())
    if target.exists() or any(target.resolve().is_relative_to(path.resolve())
                             or path.resolve().is_relative_to(target.resolve()) for path in inputs):
        raise ValueError("Exercise assembly output must be new and outside every input")
    started = time.monotonic()
    deadline = started + max_seconds
    limits = None
    if learning is not None:
        limits = (4608, 1152 * 1024 * 1024) if video is not None else (4096, 1024 * 1024 * 1024)
    elif video is not None:
        limits = (2112, 512 * 1024 * 1024)
    reservations = dict(STAGE_LIMITS)
    if video is not None:
        reservations["body_source"] = (512, 128 * 1024 * 1024)
    if learning is not None:
        reservations.pop("interactive_lab")
        reservations["learning_workflow"] = learning_allowance
    raw_inputs = None
    if video is not None:
        from weft.exercise_scene_inputs import preflight_exercise_scene_inputs

        raw_options = {key: value for key, value in (("pose_mode", pose_mode), ("pose_identity", pose_identity),
                                                    ("source_note", source_note)) if value is not None}
        raw_inputs = preflight_exercise_scene_inputs(source, evidence, config, pose_model=pose_model,
            source_mappings=source_mappings, deadline=deadline, **raw_options)
    target.mkdir(parents=True, exist_ok=False)
    stages, completed_lab, completed_learning = [], None, None
    finalization_started = False
    _write_metadata(target, "settings.json", _bytes({"schema_version": 1, "created_at": local_time(),
        "base_run": str(source) if human_run is None and video is None else None,
        "human_run": str(source) if human_run is not None else None,
        "video": str(source) if video is not None else None,
        "body_scale": float(resolved_scale) if human_run is not None or video is not None else None,
        "body_scale_scope": "Explicit body-height-to-world prior; not measured scale" if human_run is not None or video is not None else None,
        "evidence_run": str(evidence), "config": str(config),
        "source_mappings": [str(Path(value).absolute()) for value in source_mappings],
        "root_assistance": root_assistance, "full_reference": full_reference, "maximum_seconds": max_seconds,
        "maximum_files": (limits or (MAX_FILES, MAX_BYTES))[0], "maximum_bytes": (limits or (MAX_FILES, MAX_BYTES))[1],
        "stage_output_reservations": {name: {"files": bounds[0], "bytes": bounds[1]}
                                      for name, bounds in reservations.items()},
        "deadline_scope": "Cooperative checks around bounded stages, verification and sealing; in-flight native/helper calls are not preemptible.",
        "inference_calls": "bounded local pose inference requested" if video is not None else 0,
        "api_calls": 0, "training": learning if learning is not None else False,
        "scope": "Explicit initial camera/shape priors, shared-world construction and lab packaging; not automatic video reconstruction."}))

    def remaining():
        left = deadline - time.monotonic()
        if left < 1:
            raise TimeoutError("Exercise assembly reached its cooperative wall-time bound")
        return left

    def stage(name, status, **detail):
        row = {"stage": name, "status": status, "elapsed_seconds": time.monotonic() - started, **detail}
        stages.append(row)
        _write_metadata(target, "stages.json", _bytes({"stages": stages}))
        if progress is not None:
            progress(dict(row))

    def lab_detail():
        if completed_lab is None:
            return {"lab_status": "not_ready", "lab_directory": None}
        return {"lab_status": "ready", "lab_directory": completed_lab["lab_directory"],
                "scene_sha256": completed_lab["scene_sha256"], "task_sha256": completed_lab["task_sha256"],
                "launch_arguments": ["lab", "--resume", completed_lab["lab_directory"]]}

    def finish(status, **detail):
        nonlocal finalization_started
        finalization_started = True
        if raw_inputs is not None:
            from weft.exercise_scene_inputs import recheck_exercise_scene_inputs

            recheck_exercise_scene_inputs(raw_inputs, deadline=deadline)
        result = {"schema_version": 1, "status": status, "stages": stages,
                  "physical_validation": "unverified", **lab_detail(), **detail}
        if completed_learning is not None:
            result["learning_workflow"] = completed_learning
        return _seal(target, result, deadline=deadline, started=started, limits=limits)

    try:
        if raw_inputs is not None:
            from weft.exercise_scene_inputs import prepare_exercise_scene_body

            _write_metadata(target, "input-preflight.json", _bytes(raw_inputs.receipt()))
            _reserve_stage(target, "body_source", deadline, limits=limits, allowance=(512, 128 * 1024 * 1024))
            stage("body_source", "running")
            source = prepare_exercise_scene_body(raw_inputs, target / "body_source", deadline=deadline,
                max_seconds=min(180, remaining()),
                progress=lambda name, detail: stage("body_source_progress", name, detail=detail))
            stage("body_source", "ready")
        if human_run is not None or raw_inputs is not None:
            from weft.shared_world_lab import prepare_shared_world_seed

            _reserve_stage(target, "body_seed", deadline, limits=limits)
            stage("body_seed", "running")
            source = prepare_shared_world_seed(source, target / "body_seed", scale=resolved_scale, deadline=deadline)
            seed_manifest = read_package_file(source, "manifest.json", max_bytes=8 * 1024 * 1024)
            stage("body_seed", "ready", manifest_sha256=hashlib.sha256(seed_manifest).hexdigest(),
                  scope="Unexposed construction seed; no dummy-prop simulation or intermediate lab")
            remaining()
        _reserve_stage(target, "equipment_priors", deadline, limits=limits)
        if raw_inputs is not None:
            from weft.exercise_scene_inputs import recheck_exercise_scene_inputs

            recheck_exercise_scene_inputs(raw_inputs, deadline=deadline)
        stage("equipment_priors", "running")
        proposal = prepare_equipment_proposal(source, evidence, config, target / "proposal",
            source_mappings=source_mappings, max_seconds=min(30, remaining()))
        stage("equipment_priors", proposal["status"], manifest_sha256=proposal["manifest_sha256"])
        if proposal["status"] != "proposal_ready":
            return finish(proposal["status"], reasons=proposal["reasons"])
        remaining()
        resolved = proposal["construction_ready_arguments"]
        _reserve_stage(target, "shared_world", deadline, limits=limits)
        stage("shared_world", "running")
        world = construct_source_equipment_run(source, target / "proposal", evidence, target / "candidate",
            target_object_id=resolved["target_object_id"], source_mappings=source_mappings,
            frame_translation=resolved["frame_translation"], full_reference=full_reference, deadline=deadline)
        stage("shared_world", "ready", manifest_sha256=world["manifest_sha256"],
              **{key: world[key] for key in ("profile", "reference_interval") if key in world})
        remaining()
        if learning is not None:
            from weft.exercise_learning_pipeline import prepare_exercise_learning_pipeline

            def learning_progress(row):
                nonlocal completed_lab
                if row["stage"] == "interactive_lab" and row["status"] == "ready":
                    completed_lab = dict(row)
                stage("learning_" + row["stage"], row["status"],
                      **{key: value for key, value in row.items() if key not in {"stage", "status", "elapsed_seconds"}})

            _reserve_stage(target, "learning_workflow", deadline, limits=limits,
                           allowance=learning_allowance)
            stage("learning_workflow", "running")
            completed_learning = prepare_exercise_learning_pipeline(target / "candidate", target / "learning_workflow",
                train_settings=learning, root_assistance=root_assistance, max_seconds=min(180, remaining()),
                progress=learning_progress)
            completed_lab = completed_learning
            stage("learning_workflow", "ready", manifest_sha256=completed_learning["manifest_sha256"])
        else:
            _reserve_stage(target, "interactive_lab", deadline, limits=limits)
            stage("interactive_lab", "running")
            completed_lab = prepare_shared_exercise_lab(target / "candidate", target / "lab",
                root_assistance=root_assistance, max_seconds=min(60, remaining()))
            stage("interactive_lab", "ready", manifest_sha256=completed_lab["manifest_sha256"])
        remaining()
        return finish("ready")
    except Exception as error:
        result = {"schema_version": 1, "status": "failed", "stages": stages,
                  "physical_validation": "unverified", **lab_detail(),
                  "error": {"type": type(error).__name__, "message": str(error)[:2000]}}
        try:
            if finalization_started:
                _failure_receipt(target, result, error, started)
            else:
                try:
                    _seal(target, result, deadline=deadline, started=started, limits=limits)
                except Exception as sealing_error:
                    _failure_receipt(target, result, sealing_error, started)
        except Exception as receipt_error:
            error.add_note(f"Failure receipt could not be sealed: {type(receipt_error).__name__}: {str(receipt_error)[:500]}")
        raise
