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


def _reserve_stage(root, name, deadline, *, limits=None, allowance=None, cap_to_remaining=False):
    maximum_files, maximum_bytes = limits or (MAX_FILES, MAX_BYTES)
    files, total = _inventory(root, deadline, immutable_only=False, limits=limits)
    count, size = allowance or STAGE_LIMITS[name]
    if cap_to_remaining:
        # The final lab enforces this smaller allowance before writing its complete closure.
        count = min(count, maximum_files - len(files) - METADATA_RESERVE_FILES)
        size = min(size, maximum_bytes - total - METADATA_RESERVE_BYTES)
    if (count < 1 or size < 1 or len(files) + count + METADATA_RESERVE_FILES > maximum_files or
            total + size + METADATA_RESERVE_BYTES > maximum_bytes):
        raise ValueError(f"Exercise assembly cannot reserve the complete {name} output within its aggregate bound")
    return count, size


def _seal(root, result, *, deadline, started, limits=None, retained_stages=None):
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
    for prefix, expected in (retained_stages or {}).items():
        actual = {name[len(prefix) + 1:]: digest for name, digest in files.items() if name.startswith(prefix + "/")}
        if actual != expected:
            raise ValueError("Completed recovery stage changed before assembly sealing")
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
                           pose_recovery=None, train_settings=None, max_seconds=None, progress=None,
                           scene_scope="person_and_equipment", static_supports=None,
                           geometry_run=None, geometry_manifest_sha256=None, native_timing=False, auto_context_path=None):
    """Reuse the standard stages once; unsupported evidence stops before simulation is exposed."""
    if type(native_timing) is not bool:
        raise ValueError("Native timing must be explicitly enabled or disabled")
    if native_timing and video is None:
        raise ValueError("Native timing capture requires raw video; saved bodies and worlds retain their own clocks")
    if geometry_run is not None:
        if (any(value is not None for value in (base_run, evidence_run, config_path, human_run, body_scale, video,
                pose_model, pose_mode, pose_identity, source_note, pose_recovery, train_settings, static_supports, auto_context_path))
                or source_mappings or root_assistance is not False or full_reference is not False
                or scene_scope != "person_and_equipment"):
            raise ValueError("Saved geometry owns its body, source, camera, equipment and task; construction, control and training overrides are not allowed")
        return _prepare_geometry_scene(geometry_run, out, expected_manifest_sha256=geometry_manifest_sha256,
                                       max_seconds=max_seconds, progress=progress)
    if geometry_manifest_sha256 is not None:
        raise ValueError("A geometry manifest pin requires --geometry-run")
    if sum(value is not None for value in (base_run, human_run, video)) != 1:
        raise ValueError("Provide exactly one shared-world base, mobile human run or raw video")
    if not isinstance(scene_scope, str) or scene_scope not in {"person_and_equipment", "human_only", "human_static_supports"}:
        raise ValueError("Scene scope requires equipment, person-only or explicit static-support assumptions")
    person_only = scene_scope == "human_only"
    static_support = scene_scope == "human_static_supports"
    if auto_context_path is not None and (person_only or static_support):
        raise ValueError("Automatic equipment role context requires the equipment construction route")
    support, support_raw, support_path = None, None, None
    if static_support:
        if (base_run is not None or evidence_run is not None or config_path is not None or source_mappings
                or static_supports is None or body_scale is not None or root_assistance):
            raise ValueError("Static-support scope needs video/body input and its prior file, without equipment mappings or scale/root overrides")
        from weft.exercise_support_inputs import read_support_priors

        support, support_raw, support_path = read_support_priors(static_supports)
        body_scale, root_assistance = support["scale"], support["root_assistance"]
    elif static_supports is not None:
        raise ValueError("Static-support priors require the explicit human_static_supports scope")
    elif person_only:
        if base_run is not None or evidence_run is not None or config_path is not None or source_mappings:
            raise ValueError("Person-only scope accepts a video or sealed exercise-body source, without equipment, config or mappings")
    elif evidence_run is None or config_path is None:
        raise ValueError("Equipment scenes require both sealed equipment evidence and an explicit configuration")
    if human_run is None and video is None and body_scale is not None:
        raise ValueError("Body scale applies only to a mobile human run; existing worlds keep their scale")
    if video is None and any(value is not None for value in (pose_model, pose_mode, pose_identity, source_note, pose_recovery)):
        raise ValueError("Pose model/options apply only to raw --video input; saved evidence is never reinferred")
    if pose_recovery is not None and pose_recovery != "selective-crop":
        raise ValueError("Pose recovery requires the explicit selective-crop recipe")
    if video is not None and pose_model is None:
        raise ValueError("Raw exercise video requires an explicit local pose model")
    learning, learning_allowance = None, None
    if train_settings is not None:
        from weft.exercise_learning_pipeline import MAX_BYTES as LEARNING_BYTES, MAX_FILES as LEARNING_FILES, validate_train_settings

        learning = validate_train_settings(train_settings)
        if static_support and learning.get("policy_family", "phase-residual") != "phase-residual":
            raise ValueError("Static-support learning requires the phase-residual policy family")
        learning_allowance = (LEARNING_FILES, LEARNING_BYTES)
    resolved_scale = 1.0 if body_scale is None else body_scale
    if (type(resolved_scale) not in (int, float) or not math.isfinite(resolved_scale)
            or not .5 <= resolved_scale <= 2.):
        raise ValueError("Body scale must be an explicit finite prior in [0.5, 2.0]")
    if type(root_assistance) is not bool:
        raise ValueError("Root assistance must be explicitly enabled or disabled")
    if type(full_reference) is not bool:
        raise ValueError("Full-reference assembly must be explicitly enabled or disabled")
    full_reference = full_reference or person_only or static_support
    extended = video is not None or learning is not None
    if max_seconds is None:
        if pose_recovery is not None:
            max_seconds = 600 if learning is not None else 450
        else:
            max_seconds = 450 if video is not None and learning is not None else 270 if extended else 90
    maximum_seconds = 660 if pose_recovery is not None else 480 if extended else 180
    if type(max_seconds) not in (int, float) or not math.isfinite(max_seconds) or not 1 <= max_seconds <= maximum_seconds:
        raise ValueError(f"Exercise assembly requires a cooperative 1–{maximum_seconds} second bound")
    if not isinstance(source_mappings, (tuple, list)) or len(source_mappings) > 7:
        raise ValueError("Use at most seven verified source mapping bundles")
    source = Path(video if video is not None else human_run if human_run is not None else base_run).absolute()
    evidence = Path(evidence_run).absolute() if evidence_run is not None else None
    config = Path(config_path).absolute() if config_path is not None else None
    target = Path(out).absolute()
    inputs = [path for path in (source, evidence, config, support_path, *(Path(value).absolute() for value in source_mappings))
              if path is not None]
    if auto_context_path is not None:
        inputs.append(Path(auto_context_path).absolute())
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
    if pose_recovery is not None:
        limits = (limits[0] + 1024, limits[1] + 256 * 1024 * 1024)
    reservations = dict(STAGE_LIMITS)
    if person_only or static_support:
        from weft.shared_world_human import MAX_OUTPUT_BYTES as PERSON_BYTES, MAX_OUTPUT_FILES as PERSON_FILES

        reservations.pop("body_seed")
        reservations.pop("equipment_priors")
        reservations["shared_world"] = (PERSON_FILES, PERSON_BYTES)
    body_allowance = (1536, 384 * 1024 * 1024) if pose_recovery is not None else (512, 128 * 1024 * 1024)
    if video is not None:
        reservations["body_source"] = body_allowance
    if learning is not None:
        reservations.pop("interactive_lab")
        reservations["learning_workflow"] = learning_allowance
    if static_support:
        reservations["support_priors"] = (2, 256 << 10)
    raw_inputs, inventory, saved_body_identity = None, None, None
    if video is not None:
        from weft.exercise_scene_inputs import preflight_exercise_scene_inputs, preflight_human_scene_inputs, preflight_support_scene_inputs

        raw_options = {key: value for key, value in (("pose_mode", pose_mode), ("pose_identity", pose_identity),
                                                    ("source_note", source_note), ("pose_recovery", pose_recovery)) if value is not None}
        if static_support:
            raw_inputs = preflight_support_scene_inputs(source, support_path, pose_model=pose_model, deadline=deadline, **raw_options)
            if raw_inputs.receipt()["support_priors_sha256"] != hashlib.sha256(support_raw).hexdigest():
                raise ValueError("Static-support priors changed during input preparation")
        elif person_only:
            raw_inputs = preflight_human_scene_inputs(source, pose_model=pose_model, deadline=deadline, **raw_options)
            inventory = raw_inputs.receipt()["inventory"]
        else:
            raw_inputs = preflight_exercise_scene_inputs(source, evidence, config, pose_model=pose_model,
                source_mappings=source_mappings, deadline=deadline, **raw_options)
    elif person_only or static_support:
        from weft.person_inventory import make_person_inventory
        from weft.shared_world_human import read_person_body_source_identity

        identity = read_person_body_source_identity(source, deadline=deadline)
        if static_support:
            if identity["source_sha256"] != support["source_sha256"]:
                raise ValueError("Static-support priors name a different saved video source")
            saved_body_identity = identity
        else:
            inventory = make_person_inventory(identity["source_sha256"])
    target.mkdir(parents=True, exist_ok=False)
    stages, completed_lab, completed_learning = [], None, None
    finalization_started = False
    _write_metadata(target, "settings.json", _bytes({"schema_version": 1, "created_at": local_time(),
        "base_run": str(source) if human_run is None and video is None else None,
        "human_run": str(source) if human_run is not None else None,
        "video": str(source) if video is not None else None,
        "pose_recovery": pose_recovery,
        **({"native_timing": True, "reconstruction_clock": "source-native-clock-v1"} if native_timing else {}),
        "body_scale": float(resolved_scale) if human_run is not None or video is not None else None,
        "body_scale_scope": "Explicit body-height-to-world prior; not measured scale" if human_run is not None or video is not None else None,
        "evidence_run": str(evidence) if evidence is not None else None,
        "config": str(config) if config is not None else None,
        **({"static_supports": str(support_path), "support_priors_sha256": hashlib.sha256(support_raw).hexdigest(),
            "scene_scope": scene_scope} if static_support else {}),
        "source_mappings": [str(Path(value).absolute()) for value in source_mappings],
        "root_assistance": root_assistance, "full_reference": full_reference, "maximum_seconds": max_seconds,
        "maximum_files": (limits or (MAX_FILES, MAX_BYTES))[0], "maximum_bytes": (limits or (MAX_FILES, MAX_BYTES))[1],
        "stage_output_reservations": {name: {"files": bounds[0], "bytes": bounds[1]}
                                      for name, bounds in reservations.items()},
        "deadline_scope": "Cooperative checks around bounded stages, verification and sealing; in-flight native/helper calls are not preemptible.",
        "inference_calls": "bounded local pose inference requested" if video is not None else 0,
        "api_calls": 0, "training": learning if learning is not None else False,
        "scope": ("Source-bound body plus explicitly authored static supports/load windows; not recovered contact, equipment absence or physical accuracy."
                  if static_support else "User-declared person/ground scene; equipment absence is not verified. Scale, depth and ground remain priors."
                  if person_only else "Explicit initial camera/shape priors, shared-world construction and lab packaging; not automatic video reconstruction."),
        **({"scene_scope": scene_scope, "inventory": inventory} if person_only else {})}))
    if person_only:
        _write_metadata(target, "person-inventory.json", _bytes(inventory))
    elif static_support:
        _write_metadata(target, "support-priors.json", support_raw)

    recovery_pins, support_config_sha = [], None

    def guard_recovery():
        if static_support:
            from weft.exercise_support_inputs import recheck_support_priors

            recheck_support_priors(support_path, support_raw)
            read_package_file(target, "support-priors.json", max_bytes=128 << 10,
                              expected_sha256=hashlib.sha256(support_raw).hexdigest())
            if support_config_sha is not None:
                read_package_file(target, "support-config.json", max_bytes=128 << 10,
                                  expected_sha256=support_config_sha)
            if saved_body_identity is not None:
                read_package_file(source, "manifest.json", max_bytes=8 << 20,
                                  expected_sha256=saved_body_identity["manifest_sha256"])
        if person_only:
            read_package_file(target, "person-inventory.json", max_bytes=4096,
                              expected_sha256=hashlib.sha256(_bytes(inventory)).hexdigest())
        if recovery_pins:
            from weft.exercise_scene_inputs import _recheck_generated

            _recheck_generated(recovery_pins, deadline)

    def retained_recovery():
        return {path.relative_to(target).as_posix(): {**files, "manifest.json": hashlib.sha256(raw).hexdigest()}
                for path, raw, files in recovery_pins}

    def remaining():
        left = deadline - time.monotonic()
        if left < 1:
            raise TimeoutError("Exercise assembly reached its cooperative wall-time bound")
        return left

    def stage(name, status, **detail):
        guard_recovery()
        row = {"stage": name, "status": status, "elapsed_seconds": time.monotonic() - started, **detail}
        stages.append(row)
        _write_metadata(target, "stages.json", _bytes({"stages": stages}))
        if progress is not None:
            progress(json.loads(_bytes(row)))
        guard_recovery()

    def lab_detail():
        if completed_lab is None:
            return {"lab_status": "not_ready", "lab_directory": None}
        return {"lab_status": "ready", "lab_directory": completed_lab["lab_directory"],
                "scene_sha256": completed_lab["scene_sha256"], "task_sha256": completed_lab["task_sha256"],
                "launch_arguments": ["lab", "--resume", completed_lab["lab_directory"]]}

    def finish(status, **detail):
        nonlocal finalization_started
        finalization_started = True
        guard_recovery()
        if raw_inputs is not None:
            from weft.exercise_scene_inputs import recheck_exercise_scene_inputs

            recheck_exercise_scene_inputs(raw_inputs, deadline=deadline)
        result = {"schema_version": 1, "status": status, "stages": stages,
                  "physical_validation": "unverified", **lab_detail(), **detail}
        if completed_learning is not None:
            result["learning_workflow"] = completed_learning
        return _seal(target, result, deadline=deadline, started=started, limits=limits,
                     retained_stages=retained_recovery())

    try:
        if raw_inputs is not None:
            from weft.exercise_scene_inputs import prepare_exercise_scene_body

            def body_progress(name, detail):
                if pose_recovery is not None and name in {"pose_proposals", "pose_recovery", "body_source"}:
                    if detail.get("status") == "ready":
                        from weft.exercise_scene_inputs import _generated_pin

                        package, profile, status = {
                            "pose_proposals": ("pose_proposals", "exercise-pose-crop-proposals-v1", "source_evidence_only"),
                            "pose_recovery": ("recovered_poses", "exercise-pose-recovery-v1", "source_evidence_only"),
                            "body_source": ("body_source", "exercise-body-source-v1", "construction_only"),
                        }[name]
                        # Pin before exposing ready; observers cannot replace a completed child's evidence.
                        pin = _generated_pin(target / package, profile, status, deadline)
                        if detail.get("manifest_sha256") != hashlib.sha256(pin[1]).hexdigest():
                            raise ValueError("Recovery stage identity differs from its ready event")
                        recovery_pins.append(pin)
                    stage("source_" + name, detail.get("status", "running"), detail=detail)
                else:
                    stage("body_source_progress", name, detail=detail)

            _write_metadata(target, "input-preflight.json", _bytes(raw_inputs.receipt()))
            _reserve_stage(target, "body_source", deadline, limits=limits, allowance=body_allowance)
            stage("body_source", "running")
            extra = {"pose_proposals_out": target / "pose_proposals", "recovered_pose_out": target / "recovered_poses"} if pose_recovery is not None else {}
            if native_timing:
                extra["native_timing"] = True
            source = prepare_exercise_scene_body(raw_inputs, target / "body_source", deadline=deadline,
                max_seconds=min(240 if pose_recovery is not None else 180, remaining()), progress=body_progress, **extra)
            if (person_only or static_support) and pose_recovery is None:
                from weft.exercise_scene_inputs import _generated_pin

                recovery_pins.append(_generated_pin(target / "body_source", "exercise-body-source-v1", "construction_only", deadline))
            stage("body_source", "ready")
        if not (person_only or static_support) and (human_run is not None or raw_inputs is not None):
            from weft.shared_world_lab import prepare_shared_world_seed

            _reserve_stage(target, "body_seed", deadline, limits=limits)
            stage("body_seed", "running")
            source = prepare_shared_world_seed(source, target / "body_seed", scale=resolved_scale, deadline=deadline)
            seed_manifest = read_package_file(source, "manifest.json", max_bytes=8 * 1024 * 1024)
            stage("body_seed", "ready", manifest_sha256=hashlib.sha256(seed_manifest).hexdigest(),
                  scope="Unexposed construction seed; no dummy-prop simulation or intermediate lab")
            remaining()
        if static_support:
            from weft.exercise_support_inputs import resolve_support_config
            from weft.exercise_scene_inputs import _generated_pin
            from weft.shared_world_human import read_person_body_source_identity
            from weft.shared_world_support import _config, construct_static_support_run

            _reserve_stage(target, "support_priors", deadline, limits=limits, allowance=reservations["support_priors"])
            stage("support_priors", "running")
            body_identity = read_person_body_source_identity(source, deadline=deadline)
            resolved = resolve_support_config(support_raw, source_sha256=body_identity["source_sha256"],
                                             body_manifest_sha256=body_identity["manifest_sha256"])
            _, resolved_raw, _ = _config(resolved)
            _write_metadata(target, "support-config.json", resolved_raw)
            support_config_sha = hashlib.sha256(resolved_raw).hexdigest()
            stage("support_priors", "ready", support_priors_sha256=hashlib.sha256(support_raw).hexdigest(),
                  config_sha256=support_config_sha,
                  body_source_manifest_sha256=body_identity["manifest_sha256"],
                  scope="Authored static geometry/load windows bound to this exact body; not recovered contact")
            _reserve_stage(target, "shared_world", deadline, limits=limits, allowance=reservations["shared_world"])
            stage("shared_world", "running", scope="One source-bound body and authored static supports")
            world = construct_static_support_run(source, target / "support-config.json", target / "candidate",
                                                 max_seconds=min(30, remaining()))
            pin = _generated_pin(target / "candidate", "shared-world-v4", "ready", deadline)
            if world["manifest_sha256"] != hashlib.sha256(pin[1]).hexdigest():
                raise ValueError("Static-support candidate identity changed after construction")
            recovery_pins.append(pin)
            stage("shared_world", world["status"], manifest_sha256=world["manifest_sha256"],
                  **{key: world[key] for key in ("profile", "reference_interval", "readiness", "scope") if key in world})
            remaining()
        elif person_only:
            from weft.shared_world_human import construct_human_only_run

            if raw_inputs is not None:
                from weft.exercise_scene_inputs import recheck_exercise_scene_inputs

                recheck_exercise_scene_inputs(raw_inputs, deadline=deadline)
            _reserve_stage(target, "shared_world", deadline, limits=limits, allowance=reservations["shared_world"])
            stage("shared_world", "running", scope="Declared person and ground; no equipment or dummy objects")
            world = construct_human_only_run(source, inventory, target / "candidate", scale=resolved_scale,
                deadline=deadline, max_seconds=min(30, remaining()))
            if world["status"] == "ready":
                from weft.exercise_scene_inputs import _generated_pin

                pin = _generated_pin(target / "candidate", "shared-world-v3", "ready", deadline)
                if world["manifest_sha256"] != hashlib.sha256(pin[1]).hexdigest():
                    raise ValueError("Person-only candidate identity differs from its construction result")
                recovery_pins.append(pin)
            stage("shared_world", world["status"], manifest_sha256=world["manifest_sha256"],
                  **{key: world[key] for key in ("profile", "reference_interval", "readiness", "scope") if key in world})
            if world["status"] != "ready":
                return finish(world["status"], reasons=world.get("reasons", []))
            remaining()
        else:
            _reserve_stage(target, "equipment_priors", deadline, limits=limits)
            if raw_inputs is not None:
                from weft.exercise_scene_inputs import recheck_exercise_scene_inputs

                recheck_exercise_scene_inputs(raw_inputs, deadline=deadline)
            stage("equipment_priors", "running")
            proposal = prepare_equipment_proposal(source, evidence, config, target / "proposal",
                source_mappings=source_mappings, max_seconds=min(30, remaining()),
                **({"auto_context_path": auto_context_path} if auto_context_path is not None else {}))
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
            lab_limits = _reserve_stage(target, "interactive_lab", deadline, limits=limits, cap_to_remaining=True)
            stage("interactive_lab", "running", output_limits={"files": lab_limits[0], "bytes": lab_limits[1]})
            completed_lab = prepare_shared_exercise_lab(target / "candidate", target / "lab",
                root_assistance=root_assistance, max_seconds=min(60, remaining()), output_limits=lab_limits)
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
                    guard_recovery()
                    _seal(target, result, deadline=deadline, started=started, limits=limits,
                          retained_stages=retained_recovery())
                except Exception as sealing_error:
                    _failure_receipt(target, result, sealing_error, started)
        except Exception as receipt_error:
            error.add_note(f"Failure receipt could not be sealed: {type(receipt_error).__name__}: {str(receipt_error)[:500]}")
        raise


def _prepare_geometry_scene(run, out, *, expected_manifest_sha256, max_seconds, progress):
    from weft.exercise_geometry_inputs import (
        MAX_BYTES as INPUT_BYTES, MAX_FILES as INPUT_FILES, MAX_RECEIPT_BYTES, SCOPE,
        read_geometry_resume_inputs, recheck_geometry_resume_inputs,
    )
    from weft.shared_world_packaging import write_package_files

    max_seconds = 90 if max_seconds is None else max_seconds
    if type(max_seconds) not in (int, float) or not math.isfinite(max_seconds) or not 1 <= max_seconds <= 180:
        raise ValueError("Saved geometry assembly requires a cooperative 1–180 second bound")
    source, target = Path(run).absolute(), Path(out).absolute()
    if target.exists() or target.resolve().is_relative_to(source.resolve()) or source.resolve().is_relative_to(target.resolve()):
        raise ValueError("Saved geometry assembly output must be new and outside its input")
    started = time.monotonic()
    deadline = started + max_seconds
    prepared = read_geometry_resume_inputs(source, expected_manifest_sha256=expected_manifest_sha256, deadline=deadline)
    receipt, snapshot = prepared.receipt(), prepared.snapshot_files()
    target.mkdir(parents=True, exist_ok=False)
    stages, completed_lab, finalization_started = [], None, False
    _write_metadata(target, "settings.json", _bytes({"schema_version": 1, "created_at": local_time(),
        "geometry_run": str(source), "geometry_manifest_sha256": receipt["study_manifest_sha256"],
        "scene_scope": "saved_conditional_geometry", "training": False, "api_calls": 0, "inference_calls": 0,
        "maximum_seconds": max_seconds, "maximum_files": MAX_FILES, "maximum_bytes": MAX_BYTES,
        "stage_output_reservations": {"geometry_source": {"files": INPUT_FILES + 1, "bytes": INPUT_BYTES + MAX_RECEIPT_BYTES},
                                      "interactive_lab": {"files": LAB_FILES, "bytes": LAB_BYTES}},
        "scope": SCOPE, "task_scope": "Exact saved task; no control, interval or gate override"}))

    def remaining():
        left = deadline - time.monotonic()
        if left < 1:
            raise TimeoutError("Saved geometry assembly reached its cooperative wall-time bound")
        return left

    def stage(name, status, **detail):
        recheck_geometry_resume_inputs(prepared, deadline=deadline)
        row = {"stage": name, "status": status, "elapsed_seconds": time.monotonic() - started, **detail}
        stages.append(row)
        _write_metadata(target, "stages.json", _bytes({"stages": stages}))
        if progress is not None:
            progress(json.loads(_bytes(row)))
        recheck_geometry_resume_inputs(prepared, deadline=deadline)

    def result(status, **details):
        lab = {"lab_status": "not_ready", "lab_directory": None} if completed_lab is None else {
            "lab_status": "ready", "lab_directory": completed_lab["lab_directory"],
            "scene_sha256": receipt["candidate_scene_sha256"], "task_sha256": receipt["task_sha256"],
            "launch_arguments": ["lab", "--resume", completed_lab["lab_directory"]]}
        return {"schema_version": 1, "status": status, "stages": stages, "physical_validation": "unverified",
                "geometry_source": receipt, **lab, **details}

    try:
        _reserve_stage(target, "geometry_source", deadline, allowance=(INPUT_FILES + 1, INPUT_BYTES + MAX_RECEIPT_BYTES))
        stage("geometry_source", "running")
        (target / "geometry_source").mkdir()
        write_package_files(target / "geometry_source", snapshot)
        stage("geometry_source", "ready", study_manifest_sha256=receipt["study_manifest_sha256"],
              candidate_manifest_sha256=receipt["candidate_manifest_sha256"], source_fit="retained_conditional_study_claims")
        _reserve_stage(target, "interactive_lab", deadline)
        stage("interactive_lab", "running")
        # The ordinary lab loader owns full source proof and native admission; no metadata-only bypass.
        lab = prepare_shared_exercise_lab(prepared.candidate_run, target / "lab", max_seconds=min(60, remaining()))
        recheck_geometry_resume_inputs(prepared, deadline=deadline)
        if lab.get("scene_sha256") != receipt["candidate_scene_sha256"] or lab.get("task_sha256") != receipt["task_sha256"]:
            raise ValueError("Prepared lab changed the selected saved scene or task identity")
        members = dict(prepared.members)
        for name in ("scene.json", "exercise-task.json"):
            read_package_file(target / "lab" / "run", name, max_bytes=16 << 20,
                              expected_sha256=hashlib.sha256(members["geometry/" + name]).hexdigest())
        read_package_file(target / "lab" / "run", "source-run/manifest.json", max_bytes=1 << 20,
                          expected_sha256=receipt["candidate_manifest_sha256"])
        completed_lab = lab
        stage("interactive_lab", "ready", manifest_sha256=lab["manifest_sha256"])
        for name, raw in snapshot.items():
            _check_deadline(deadline)
            read_package_file(target / "geometry_source", name, max_bytes=max(1, len(raw)), expected_sha256=hashlib.sha256(raw).hexdigest())
        recheck_geometry_resume_inputs(prepared, deadline=deadline)
        finalization_started = True
        return _seal(target, result("ready"), deadline=deadline, started=started)
    except Exception as error:
        failed = result("failed", error={"type": type(error).__name__, "message": str(error)[:2000]})
        try:
            if finalization_started:
                _failure_receipt(target, failed, error, started)
            else:
                try:
                    recheck_geometry_resume_inputs(prepared, deadline=deadline)
                    _seal(target, failed, deadline=deadline, started=started)
                except Exception as sealing_error:
                    _failure_receipt(target, failed, sealing_error, started)
        except Exception as receipt_error:
            error.add_note(f"Failure receipt could not be sealed: {type(receipt_error).__name__}: {str(receipt_error)[:500]}")
        raise
