"""Matched authored/reference-proposal trials on one immutable simulator task."""

from copy import deepcopy
from hashlib import sha256
from importlib.metadata import version
import json
import math
import os
from pathlib import Path
import stat
import sys
import time
from types import SimpleNamespace

from weft.io import canonical_hash, local_time
from weft.shared_exercise_reference_controller import (
    RUNTIME_SHA256, prepare_reference_controller, reference_action_contract,
)
from weft.shared_exercise_reference_inputs import MAX_CAMERA_INPUT_BYTES, MAX_MULTIVIEW_INPUT_BYTES, load_reference_inputs
from weft.shared_world_packaging import collect_shared_source_files, read_package_file, write_package_files


PROFILE = "shared-exercise-reference-trial-v1"
MAX_INPUT_BYTES = 32 << 20
MAX_SOURCE_BYTES = 128 << 20
MAX_RUNTIME_BYTES = 16 << 20
MAX_EXPORT_BYTES = 128 << 20
MAX_OUTPUT_BYTES = 512 << 20
MAX_OUTPUT_FILES = 2048
MAX_HORIZON = 900
MAX_CONTROL_STEPS = 1800
RESET_OPTIONS = {"jitter": False, "jitter_scale": 0.}
MODES = ("authored", "reference_proposal")
RUNTIME_FILES = (
    "__init__.py", "shared_exercise_reference_trial.py", "shared_exercise_reference_report.py",
    "shared_exercise_reference_inputs.py", "whole_body_reference_report.py",
    "shared_exercise_reference_controller.py", "shared_exercise_reference_recording.py",
    "shared_exercise_recording.py", "shared_exercise_task.py", "shared_exercise_reset.py",
    "shared_world.py", "shared_world_observation.py", "shared_world_evidence.py", "shared_world_packaging.py",
    "world_evidence.py", "source_equivalence.py", "observations.py", "person_inventory.py",
    "human.py", "mobile_human.py", "whole_body.py", "spatial_rigid.py", "spatial_pose.py",
    "body_frame.py", "root_motion.py", "errors.py", "io.py", "recording.py",
    "assets/shared-world-replay.html",
)
_SCOPE = ("Two authored controller alternatives on the same original scene, task, reset and scored targets. "
          "Delivery completion is not task success, repaired geometry, learned control or physical validation.")


def _require(condition, message):
    if not condition:
        raise ValueError("Reference trial: " + message)


def _check(deadline):
    if time.monotonic() >= deadline:
        raise TimeoutError("Reference trial exceeded its cooperative deadline; partial artifacts are preserved")


def _bytes(value):
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def _safe_path(value):
    path = Path(value).absolute()
    _require(".." not in path.parts and len(path.parts) <= 64
             and not any(p.is_symlink() for p in (path, *path.parents)), "unsafe path")
    return path


def _runtime_snapshot(deadline, *, camera=False, multiview=False):
    _require(not multiview or camera, "multiview runtime requires authenticated camera inputs")
    if camera:
        from weft.camera_reference_controller import RUNTIME_SHA256 as reviewed
        from weft.camera_reference_recording import REQUIRED_RUNTIME_FILES
        names = tuple(sorted(set(RUNTIME_FILES) | REQUIRED_RUNTIME_FILES | {
            "camera_reference_inputs.py", "source_surface_fit_inputs.py"}))
        if multiview:
            from weft.camera_multiview_inputs import RUNTIME_MODULES
            names = tuple(sorted(set(names) | {name + ".py" for name in RUNTIME_MODULES}
                                 | {"camera_multiview_report.py"}))
    else:
        reviewed = RUNTIME_SHA256
        names = RUNTIME_FILES
    package, files, total = Path(__file__).parent, {}, 0
    for name in names:
        _check(deadline)
        raw = read_package_file(package, name, max_bytes=min(4 << 20, MAX_RUNTIME_BYTES // 3 - total))
        total += len(raw)
        _require(total <= MAX_RUNTIME_BYTES // 3, "runtime initial/pre-model/final read bound exceeded")
        files[name] = raw
    _require(all(sha256(files[k]).hexdigest() == v for k, v in reviewed.items()),
             "unreviewed feedback or reference-velocity runtime")
    return files


def _runtime_versions():
    return {"python": sys.version.split()[0], "executable": sys.executable,
            "packages": {name: version(name) for name in ("numpy", "mujoco", "gymnasium", "opencv-contrib-python", "Pillow")}}


def _engine_api(*, camera=False):
    # Source admission and exact runtime pins precede validators that compile mass models.
    from weft.human import reference_velocity_index
    from weft.recording import Recorder, export_recording
    from weft.shared_world import validate_shared_world
    if camera:
        from weft.camera_reference_recording import recording_scene, verify_action_replay
        from weft.camera_shared_task import CameraSharedTaskEnv as Env, validate_camera_shared_task as validate_task
    else:
        from weft.shared_exercise_recording import recording_scene, verify_action_replay
        from weft.shared_exercise_task import SharedExerciseTaskEnv as Env, validate_shared_exercise_task as validate_task

    return SimpleNamespace(velocity_index=reference_velocity_index, Recorder=Recorder, export=export_recording,
        recording_scene=recording_scene, replay=verify_action_replay, Env=Env,
        validate_scene=validate_shared_world, validate_task=validate_task)


def _write_page(out, report):
    from weft.shared_exercise_reference_report import write_reference_trial_report
    return write_reference_trial_report(out, report)


def _read_identity(root, name, expected, maximum, deadline):
    _check(deadline)
    raw = read_package_file(_safe_path(root), name, max_bytes=max(1, maximum), expected_sha256=expected)
    _check(deadline)
    return len(raw)


def _recheck(inputs, proof, runtime, runtime_versions, task_path, deadline):
    receipt = inputs.receipt()
    selected = receipt["selected_files"]
    metadata_read = receipt["input_bytes_read"]
    metadata_limit = (MAX_MULTIVIEW_INPUT_BYTES if receipt.get("profile") == "camera-multiview-reference-inputs-v1"
                      else MAX_CAMERA_INPUT_BYTES if receipt.get("profile") == "camera-reference-inputs-v1"
                      else MAX_INPUT_BYTES)
    _require(metadata_read <= metadata_limit, "aggregate selected metadata read bound exhausted")
    for item in selected:
        _require(metadata_read + item["bytes"] <= metadata_limit, "aggregate selected metadata read bound exhausted")
        metadata_read += _read_identity(item["directory"], item["name"], item["sha256"], item["bytes"], deadline)
        _require(metadata_read <= metadata_limit, "aggregate selected metadata read bound exhausted")
    _require(any(_safe_path(item["directory"]) / item["name"] == task_path for item in selected),
             "explicit task was not included in final source readback")
    source_read = len(proof.manifest_bytes or b"") + sum(map(len, proof.files.values()))
    for name, raw in proof.files.items():
        source_read += _read_identity(inputs.source_directory, name, sha256(raw).hexdigest(), len(raw), deadline)
        _require(source_read <= MAX_SOURCE_BYTES, "aggregate source proof reads exceed 128 MiB")
    runtime_read = 2 * sum(map(len, runtime.values()))
    for name, raw in runtime.items():
        runtime_read += _read_identity(Path(__file__).parent, name, sha256(raw).hexdigest(), len(raw), deadline)
        _require(runtime_read <= MAX_RUNTIME_BYTES, "aggregate runtime reads exceed 16 MiB")
    _require(_runtime_versions() == runtime_versions, "runtime versions changed during trials")
    return {"selected_metadata_bytes": metadata_read, "source_proof_bytes": source_read,
            "runtime_bytes": runtime_read, "selected_metadata_files": len(selected),
            "source_files": len(proof.files), "runtime_files": len(runtime),
            "scope": "Selected snapshot/readback accounting, not total filesystem I/O; nested validator implementation reads and library imports are excluded."}


def _inventory(root, deadline):
    files, total, pending = {}, 0, [root]
    while pending:
        _check(deadline)
        directory = pending.pop()
        with os.scandir(directory) as entries:
            for entry in entries:
                _check(deadline)
                _require(not entry.is_symlink(), "output contains a symlink")
                if entry.is_dir(follow_symlinks=False):
                    pending.append(Path(entry.path))
                    _require(len(pending) <= MAX_OUTPUT_FILES, "too many output directories")
                    continue
                _require(entry.is_file(follow_symlinks=False), "output is not a regular file")
                relative = Path(entry.path).relative_to(root).as_posix()
                _require(relative != "manifest.json", "terminal manifest already exists")
                _require(len(files) + 1 < MAX_OUTPUT_FILES, "output file bound exceeded")
                fd = os.open(entry.path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
                with os.fdopen(fd, "rb") as stream:
                    before, digest, size = os.fstat(stream.fileno()), sha256(), 0
                    _require(stat.S_ISREG(before.st_mode) and before.st_size <= MAX_EXPORT_BYTES, "output file bound exceeded")
                    while block := stream.read(1 << 20):
                        _check(deadline)
                        size += len(block)
                        total += len(block)
                        _require(total <= MAX_OUTPUT_BYTES, "output byte bound exceeded")
                        digest.update(block)
                    after = os.fstat(stream.fileno())
                _require(size == before.st_size and (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
                         == (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns), "output changed during sealing")
                files[relative] = digest.hexdigest()
    return files, total


def _seal(out, report, deadline, protected):
    files, total = _inventory(out, deadline)
    _require(all(files.get(name) == expected for name, expected in protected.items()),
             "preserved input/runtime/plan copy changed before sealing")
    for row in report["trials"]:
        if "recording" not in row:
            continue
        prefix = "recordings/" + row["mode"] + "/"
        expected = row["recording"]["manifest_sha256"]
        raw = read_package_file(out, prefix + "manifest.json", max_bytes=64 << 10, expected_sha256=expected)
        manifest = json.loads(raw)
        _require(set(manifest["files"]) == {"scene.json", "recording.json", "events.jsonl", "video.mp4", "preview.gif", "replay.html"}
                 and all(files.get(prefix + name) == value for name, value in manifest["files"].items()),
                 "completed recording changed after export")
    manifest = {"schema_version": 1, "profile": PROFILE, "status": report["status"], "created_at": local_time(),
                "scene_sha256": report.get("scene_sha256"), "task_sha256": report.get("task_sha256"),
                "physical_validation": "unverified", "files": files}
    raw = _bytes(manifest)
    _require(total + len(raw) <= MAX_OUTPUT_BYTES, "terminal seal exceeds output bound")
    write_package_files(out, {"manifest.json": raw})
    return {"manifest_sha256": sha256(raw).hexdigest(), "output_files": len(files) + 1,
            "output_bytes": total + len(raw)}


def prepare_reference_trial(proposal, scene_run, task_path, out, *, seed=1729, max_seconds=60,
                            expected_proposal_manifest_sha256=None, expected_scene_manifest_sha256=None,
                            expected_task_sha256=None, progress=None):
    """Run two bounded matched controls, then export and replay each recording once.

    This trusted fixed-code workflow has cooperative deadlines, not native-call
    preemption or a hard RSS cap. Call it off request/UI threads; failures preserve
    closed or partial journals and never relabel completed recordings cancelled.
    """
    _require(type(seed) is int and 0 <= seed <= 2**31 - 1, "seed must be a bounded integer")
    _require(type(max_seconds) in (int, float) and math.isfinite(max_seconds) and 1 <= max_seconds <= 120,
             "cooperative bound must be 1–120 seconds")
    _require(progress is None or callable(progress), "progress callback must be callable")
    proposal, source, task_path, out = map(_safe_path, (proposal, scene_run, task_path, out))
    _require(all(out != item and not out.is_relative_to(item) and not item.is_relative_to(out)
                 for item in (proposal, source)), "output must be outside immutable source directories")
    _require(task_path != out and not task_path.is_relative_to(out), "output overlaps the original task")
    if out.exists():
        raise FileExistsError("Reference trial requires a new output directory")
    started = time.monotonic()
    deadline = started + max_seconds
    work_deadline = deadline - min(5., .1 * max_seconds)
    out.mkdir(parents=True, exist_ok=False)
    report = {"schema_version": 1, "profile": PROFILE, "status": "failed", "stage": "preparing",
        "created_at": local_time(), "seed": seed, "reset_options": dict(RESET_OPTIONS), "same_initial_state": None,
        "trials": [], "error": None, "physical_validation": "unverified", "scope": _SCOPE,
        "counts": {"live_control_steps": 0, "live_trial_envs": 0, "exports_attempted": 0, "exports_completed": 0,
                   "replays_attempted": 0, "replays_completed": 0},
        "limits": {"maximum_seconds": max_seconds, "maximum_horizon_steps": MAX_HORIZON,
                   "maximum_live_control_steps": MAX_CONTROL_STEPS, "maximum_output_bytes": MAX_OUTPUT_BYTES,
                   "maximum_output_files": MAX_OUTPUT_FILES, "maximum_export_bytes_each": MAX_EXPORT_BYTES,
                   "maximum_selected_metadata_read_bytes": MAX_CAMERA_INPUT_BYTES,
                   "maximum_legacy_selected_metadata_read_bytes": MAX_INPUT_BYTES,
                   "maximum_camera_selected_metadata_read_bytes": MAX_CAMERA_INPUT_BYTES,
                   "maximum_source_read_bytes": MAX_SOURCE_BYTES,
                   "maximum_runtime_read_bytes": MAX_RUNTIME_BYTES, "hard_rss_cap": False}}
    env, active_recorder, cleanup_errors, protected = None, None, [], {}

    def emit(stage, **detail):
        report["stage"] = stage
        if progress is not None:
            progress(stage, deepcopy(detail))

    try:
        _check(work_deadline)
        inputs = load_reference_inputs(proposal, source, task_path,
            expected_proposal_manifest_sha256=expected_proposal_manifest_sha256,
            expected_scene_manifest_sha256=expected_scene_manifest_sha256, expected_task_sha256=expected_task_sha256,
            max_seconds=min(10., work_deadline - time.monotonic()), max_input_bytes=2 * (MAX_INPUT_BYTES // 3),
            allow_camera=True, camera_max_input_bytes=2 * (MAX_CAMERA_INPUT_BYTES // 3),
            multiview_max_input_bytes=2 * (MAX_MULTIVIEW_INPUT_BYTES // 3))
        input_profile = inputs.receipt().get("profile")
        multiview = input_profile == "camera-multiview-reference-inputs-v1"
        camera = input_profile in {"camera-reference-inputs-v1", "camera-multiview-reference-inputs-v1"}
        if camera:
            from weft.camera_reference_inputs import CameraReferenceInputs
            _require(isinstance(inputs, CameraReferenceInputs), "camera route requires authenticated camera inputs")
        report["limits"]["maximum_selected_metadata_read_bytes"] = (
            MAX_MULTIVIEW_INPUT_BYTES if multiview else MAX_CAMERA_INPUT_BYTES if camera else MAX_INPUT_BYTES)
        if multiview:
            report["limits"]["maximum_multiview_selected_metadata_read_bytes"] = MAX_MULTIVIEW_INPUT_BYTES
        raw_scene, raw_task = json.loads(inputs.scene_bytes), json.loads(inputs.task_bytes)
        _require((raw_scene.get("profile") == "shared-world-v6") is camera,
                 "authenticated input family differs from scene profile")
        horizon = raw_task.get("interval", {}).get("horizon_steps")
        _require(type(horizon) is int and 1 <= horizon <= MAX_HORIZON
                 and raw_scene.get("task", {}).get("horizon_steps") == horizon,
                 "raw task horizon must be at most 900 before any models")
        report.update(scene_sha256=canonical_hash(raw_scene), task_sha256=canonical_hash(raw_task),
                      source_identities=inputs.identities, horizon_steps=horizon,
                      controller_family=("camera_multiview_angle_overlay" if multiview else
                                         "camera_angle_overlay" if camera else "legacy_body_reference"))
        emit("inputs_verified", horizon_steps=horizon)
        runtime = (_runtime_snapshot(work_deadline, camera=True, multiview=True) if multiview else
                   _runtime_snapshot(work_deadline, camera=True) if camera else _runtime_snapshot(work_deadline))
        versions = _runtime_versions()
        emit("runtime_pinned", files=len(runtime))
        _check(work_deadline)
        proof = collect_shared_source_files(raw_scene, inputs.source_directory,
            expected_manifest_sha256=inputs.identities["scene_manifest_sha256"], deadline=work_deadline)
        _check(work_deadline)
        _require(proof.manifest_bytes == inputs.scene_manifest_bytes, "source manifest changed during full admission")
        _require(len(proof.manifest_bytes or b"") + 2 * sum(map(len, proof.files.values())) <= MAX_SOURCE_BYTES,
                 "source proof/readback bound exceeded")
        reserved = {"scene.json", "exercise-task.json", "reference-controller.json", "source-verification.json", "manifest.json"}
        proof.validate_layout(reserved_paths=reserved)
        emit("source_verified", files=len(proof.files), scope=proof.report.get("scope"))
        _check(work_deadline)
        api = _engine_api(camera=True) if camera else _engine_api()
        if camera:
            from weft.camera_reference_controller import prepare_camera_reference_controller, camera_reference_action_contract
            provider = prepare_camera_reference_controller(inputs.original_human_bytes, inputs.angle_proposal_bytes, inputs.report_bytes,
                scene=raw_scene, task=raw_task, action_contract=camera_reference_action_contract(raw_scene, raw_task),
                runtime_sha256={k: sha256(v).hexdigest() for k, v in runtime.items() if "/" not in k and k.endswith(".py")},
                reference_velocity_index=api.velocity_index)
        else:
            provider = prepare_reference_controller(inputs.original_human_bytes, inputs.candidate_human_bytes, inputs.report_bytes,
                scene=raw_scene, task=raw_task, action_contract=reference_action_contract(raw_scene, raw_task),
                runtime_sha256={k: sha256(runtime[k]).hexdigest() for k in RUNTIME_SHA256},
                reference_velocity_index=api.velocity_index)
        recipe_bytes = (json.dumps(provider.recipe(), sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()
        recipe_sha = sha256(recipe_bytes).hexdigest()
        report["reference_controller"] = {"document": "run/reference-controller.json", "sha256": recipe_sha,
                                           "description": provider.describe()}
        preserved = {"inputs/proposal-manifest.json": inputs.proposal_manifest_bytes,
            "inputs/scene-manifest.json": inputs.scene_manifest_bytes, "inputs/human.json": inputs.original_human_bytes,
            "inputs/proposal-report.json": inputs.report_bytes,
            "inputs/evidence-derived.json": inputs.evidence_bytes, "inputs/scene.json": inputs.scene_bytes,
            "inputs/exercise-task.json": inputs.task_bytes, "inputs/input-receipt.json": _bytes(inputs.receipt())}
        if camera:
            aliases = {"angle-proposal.json": "inputs/angle-proposal.json", "inputs/human.json": "inputs/human.json",
                       "report.json": "inputs/proposal-report.json", "inputs/evidence-derived.json": "inputs/evidence-derived.json"}
            for name, raw in inputs.proposal_members.items():
                aliases.setdefault(name, "inputs/proposal/" + name)
                preserved[aliases[name]] = raw
            report["proposal_input_aliases"] = aliases
        else:
            preserved["inputs/candidate-human.json"] = inputs.candidate_human_bytes
        _require(sum(map(len, preserved.values())) <= MAX_INPUT_BYTES, "input copy bound exceeded")
        run_files = {"scene.json": inputs.scene_bytes, "exercise-task.json": inputs.task_bytes,
                     "reference-controller.json": recipe_bytes, "source-verification.json": _bytes(proof.report),
                     **dict(proof.files)}
        run_files["manifest.json"] = _bytes({"schema_version": 1, "profile": "shared-exercise-reference-base-v1",
            "status": "source_verified_pending_original_scene_admission", "files": {k: sha256(v).hexdigest() for k, v in run_files.items()}})
        files = {**preserved, **{"run/" + k: v for k, v in run_files.items()},
                 **{"runtime/weft/" + k: v for k, v in runtime.items()}}
        _require(len(files) + 32 <= MAX_OUTPUT_FILES and sum(map(len, files.values())) + 2 * MAX_EXPORT_BYTES + (4 << 20)
                 <= MAX_OUTPUT_BYTES, "planned package exceeds output bounds")
        plan = {"schema_version": 1, "profile": PROFILE + "-plan", "status": "frozen",
            "source_identities": inputs.identities, "seed": seed, "reset_options": dict(RESET_OPTIONS),
            "controller_family": report["controller_family"],
            "modes": list(MODES), "horizon_steps_each": horizon, "reference_controller_sha256": recipe_sha,
            "runtime_sha256": {k: sha256(v).hexdigest() for k, v in runtime.items()}, "runtime_versions": versions,
            "preserved_files": {k: sha256(v).hexdigest() for k, v in files.items()}, "limits": report["limits"],
            "scope": _SCOPE}
        files["plan.json"] = _bytes(plan)
        _check(work_deadline)
        write_package_files(out, files)
        protected = {name: sha256(raw).hexdigest() for name, raw in files.items()}
        # A changed implementation cannot reach even the validators' zero-step native models.
        for name, raw in runtime.items():
            _read_identity(Path(__file__).parent, name, sha256(raw).hexdigest(), len(raw), work_deadline)
        _check(work_deadline)
        scene = api.validate_scene(raw_scene)
        _check(work_deadline)
        task = api.validate_task(raw_task, scene)
        _require(canonical_hash(scene) == report["scene_sha256"] and canonical_hash(task) == report["task_sha256"],
                 "admission rewrote the original scene or task")
        _check(work_deadline)
        emit("base_admitted", scene_sha256=report["scene_sha256"], task_sha256=report["task_sha256"])
        envelopes = {"authored": api.recording_scene(scene, task),
                     "reference_proposal": api.recording_scene(scene, task, reference_controller_document=recipe_bytes)}
        _check(work_deadline)
        env = api.Env(scene, task)
        report["counts"]["live_trial_envs"] = 1
        first_initial = None
        for mode in MODES:
            _check(work_deadline)
            env.reset(seed=seed, options=dict(RESET_OPTIONS))
            initial = env.snapshot()
            initial_sha = canonical_hash(initial)
            if first_initial is None:
                first_initial = initial_sha
            else:
                report["same_initial_state"] = initial_sha == first_initial
                _require(report["same_initial_state"], "matched trials have different complete initial states")
            active_recorder = api.Recorder(out / "recordings" / mode, envelopes[mode], profile="shared_exercise")
            active_recorder.start(initial, seed=seed)
            row = {"mode": mode, "status": "recording", "steps": 0, "initial_state_sha256": initial_sha}
            report["trials"].append(row)
            emit("trial_started", mode=mode, horizon_steps=horizon)
            terminated = truncated = False
            for _ in range(horizon):
                _check(work_deadline)
                _require(report["counts"]["live_control_steps"] < MAX_CONTROL_STEPS, "live action bound exceeded")
                active_recorder.check_capacity()
                action = (env.authored_action() if mode == "authored" else provider.action(env)).tolist()
                _, reward, terminated, truncated, _ = env.step(action)
                report["counts"]["live_control_steps"] += 1
                row["steps"] += 1
                state = env.snapshot()
                event = {"type": "step", "controller": mode, "action": action, "reward": float(reward),
                         "terminated": bool(terminated), "truncated": bool(truncated), "snapshot": state}
                if mode == "reference_proposal":
                    event["reference_controller_sha256"] = recipe_sha
                active_recorder.append(event)
                if row["steps"] % 30 == 0:
                    emit("trial_progress", mode=mode, steps=row["steps"], total_live_steps=report["counts"]["live_control_steps"])
                if terminated or truncated:
                    break
            _require(terminated or truncated, "environment did not terminate within its declared task horizon")
            active_recorder.close()
            active_recorder = None
            row.update(status="recorded", final_state_sha256=canonical_hash(state), terminated=bool(terminated),
                       truncated=bool(truncated), metrics=deepcopy(env.metrics()))
            emit("trial_recorded", mode=mode, steps=row["steps"], task_success=row["metrics"].get("success"))
        env.close()
        env = None
        for row in report["trials"]:
            mode = row["mode"]
            _check(work_deadline)
            report["counts"]["exports_attempted"] += 1
            exported = api.export(out / "recordings" / mode, deadline=work_deadline, max_output_bytes=MAX_EXPORT_BYTES)
            _require(exported.get("steps") == row["steps"] and exported.get("events") == row["steps"] + 1
                     and exported.get("video_frames") == exported["events"] and exported.get("resets") == 0,
                     "export counts differ from the recorded trial")
            report["counts"]["exports_completed"] += 1
            row.update(status="exported", recording={**{k: v for k, v in exported.items() if k != "directory"},
                                                       "directory": "recordings/" + mode})
            emit("recording_exported", mode=mode, events=exported.get("events"))
            _check(work_deadline)
            report["counts"]["replays_attempted"] += 1
            replay = api.replay(out / "recordings" / mode, max_seconds=min(180., work_deadline - time.monotonic()))
            _require(replay.get("exact_state_replay") is True and replay.get("steps") == row["steps"]
                     and replay.get("events") == row["steps"] + 1
                     and replay.get("scene_sha256") == report["scene_sha256"]
                     and replay.get("task_sha256") == report["task_sha256"], "replay proof differs from the original task/trial")
            if mode == "reference_proposal":
                _require(replay.get("reference_controller_sha256") == recipe_sha
                         and replay.get("reference_actions_verified") == row["steps"], "reference action replay is incomplete")
            proof_path = "proofs/" + mode + "-action-replay.json"
            write_package_files(out, {proof_path: _bytes(replay)})
            report["counts"]["replays_completed"] += 1
            row.update(status="verified", action_replay={**replay, "path": proof_path, "status": "passed"})
            emit("recording_replay_verified", mode=mode, steps=row["steps"])
        emit("source_readback")
        report["readback"] = _recheck(inputs, proof, runtime, versions, task_path, work_deadline)
        report["status"] = "completed"
        emit("trials_verified", task_success={row["mode"]: row["metrics"].get("success") for row in report["trials"]})
    except Exception as error:
        report["error"] = {"type": type(error).__name__, "message": str(error)[:2000]}
        report["status"] = "failed"
    finally:
        if active_recorder is not None:
            try:
                active_recorder.close()
            except Exception as error:
                cleanup_errors.append({"resource": "recorder", "message": str(error)[:500]})
        if env is not None:
            try:
                env.close()
            except Exception as error:
                cleanup_errors.append({"resource": "environment", "message": str(error)[:500]})
    if cleanup_errors:
        report["cleanup_errors"] = cleanup_errors
    report["seconds"] = time.monotonic() - started
    try:
        write_package_files(out, {"report.json": _bytes(report)})
        _write_page(out, report)
        sealed = _seal(out, report, deadline, protected)
        return {**report, **sealed, "directory": str(out), "report": str(out / "report.json"),
                "inspection_path": str(out / "index.html")}
    except Exception as error:
        finalization = {"type": type(error).__name__, "message": str(error)[:2000], "report_status": report["status"]}
        try:
            write_package_files(out, {"finalization-error.json": _bytes(finalization)})
        except Exception:
            pass
        return {**report, "status": "failed", "finalization_error": finalization, "directory": str(out)}
