"""Bounded, source-pinned human-only task evaluation and exact recorded replay."""

import hashlib
import json
import math
from pathlib import Path
import platform
import time

import cv2
import gymnasium
import mujoco
import numpy as np

from weft.io import canonical_hash, digest, local_time, write_json
from weft.recording import Recorder, export_recording
from weft.shared_exercise_recording import recording_scene, verify_action_replay
from weft.shared_exercise_task import GATES, SharedExerciseTaskEnv, build_shared_exercise_task, validate_shared_exercise_task
from weft.shared_world import validate_shared_world
from weft.shared_world_evidence import verify_shared_world_sources
from weft.shared_world_packaging import collect_shared_source_files, read_package_file, read_package_manifest, write_package_files


IMPLEMENTATION_FILES = (
    "shared_exercise_run.py", "shared_exercise_task.py", "shared_exercise_recording.py",
    "shared_world.py", "shared_world_observation.py", "shared_world_evidence.py",
    "shared_world_packaging.py", "source_equivalence.py", "world_evidence.py", "observations.py",
    "human.py", "mobile_human.py", "whole_body.py", "spatial_rigid.py", "recording.py", "io.py",
    "body_frame.py", "spatial_pose.py", "root_motion.py", "errors.py",
    "assets/shared-world-replay.html",
)


def _json(raw):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("Duplicate exercise input JSON field")
            result[key] = value
        return result

    def nonfinite(_):
        raise ValueError("Exercise input JSON must be finite")

    try:
        return json.loads(raw, object_pairs_hook=unique, parse_constant=nonfinite)
    except (UnicodeError, json.JSONDecodeError, RecursionError) as error:
        raise ValueError("Exercise input must be bounded strict JSON") from error


def _seal(out, status):
    files = {str(path.relative_to(out)): digest(path) for path in sorted(out.rglob("*"))
             if path.is_file() and path != out / "manifest.json"}
    write_json(out / "manifest.json", {"schema_version": 1, "created_at": local_time(), "status": status, "files": files})
    return digest(out / "manifest.json")


def evaluate_shared_exercise_run(run, out, *, root_assistance=False, task_path=None, seed=938401,
                                 jitter=False, max_seconds=60):
    """Evaluate matched coast/authored controls; this command does not train a policy."""
    if type(seed) is not int or not 0 <= seed < 2**31:
        raise ValueError("Exercise evaluation seed must be an integer in [0,2^31)")
    if type(root_assistance) is not bool or type(jitter) is not bool:
        raise ValueError("Exercise assistance and jitter must be explicit booleans")
    if type(max_seconds) not in (int, float) or not math.isfinite(max_seconds) or not 1 <= max_seconds <= 180:
        raise ValueError("Exercise evaluation duration must be bounded to 1–180 seconds")
    if task_path is not None and root_assistance:
        raise ValueError("Use the provided task's assistance choice; do not override it")
    source, out = Path(run).resolve(), Path(out).resolve()
    if out.exists():
        raise FileExistsError("Exercise evaluation requires a new output directory")
    if out == source or out.is_relative_to(source) or source.is_relative_to(out):
        raise ValueError("Exercise outputs must be outside the immutable input")
    source = source / "run" if (source / "run" / "manifest.json").is_file() else source
    started = time.monotonic()
    manifest_raw, source_files = read_package_manifest(source)
    if "scene.json" not in source_files:
        raise ValueError("Exercise input requires a sealed shared-world scene")
    scene_raw = read_package_file(source, "scene.json", max_bytes=8 * 1024 * 1024,
                                  expected_sha256=source_files["scene.json"])
    scene = validate_shared_world(_json(scene_raw))
    proof = collect_shared_source_files(scene, source, expected_manifest_sha256=hashlib.sha256(manifest_raw).hexdigest())
    task_raw = None
    if task_path is None:
        task = build_shared_exercise_task(scene, root_assistance=root_assistance)
    else:
        path = Path(task_path).resolve()
        task_raw = read_package_file(path.parent, path.name, max_bytes=64 * 1024)
        task = validate_shared_exercise_task(_json(task_raw), scene)
    reserved = ("scene.json", "exercise-task.json", "settings.json", "evaluation.json", "failure.json",
                "source-verification.json", "input-manifest.json", "input-scene.json", "input-task.json",
                "implementation", "recordings", "manifest.json")
    proof.validate_layout(reserved_paths=reserved)
    implementation_paths = {name: Path(__file__).parent / name for name in IMPLEMENTATION_FILES}
    project = Path(__file__).resolve().parents[2]
    for name in ("requirements-lab.lock", "pyproject.toml"):
        if (project / name).is_file():
            implementation_paths["project/" + name] = project / name
    implementation = {name: path.read_bytes() for name, path in implementation_paths.items()}
    implementation_hashes = {name: hashlib.sha256(raw).hexdigest() for name, raw in implementation.items()}
    if time.monotonic() - started > max_seconds:
        raise TimeoutError("Exercise input preparation exceeded its declared bound; no output created")
    out.mkdir(parents=True, exist_ok=False)
    try:
        proof.copy_to(out, reserved_paths=reserved)
        write_package_files(out, {"input-manifest.json": manifest_raw, "input-scene.json": scene_raw,
                                 **({"input-task.json": task_raw} if task_raw is not None else {}),
                                 **{f"implementation/{name}": raw for name, raw in implementation.items()}})
        write_json(out / "scene.json", scene)
        write_json(out / "exercise-task.json", task)
        reset_options = {"jitter": jitter, "jitter_scale": 1.0 if jitter else 0.0}
        settings = {"schema_version": 1, "created_at": local_time(), "scene_sha256": canonical_hash(scene),
                    "task_sha256": canonical_hash(task), "gates": dict(GATES), "seed": seed,
                    "reset_options": reset_options, "baselines": ["coast", "authored_human_reference"],
                    "max_seconds": max_seconds, "maximum_control_steps": 2 * scene["task"]["horizon_steps"],
                    "deadline_scope": "cooperative checks between stages, physics actions, export and replay frames",
                    "max_recording_export_bytes": 64 * 1024 * 1024,
                    "maximum_recorded_events": 2 * (scene["task"]["horizon_steps"] + 1),
                    "replay_control_steps_extra": 2 * scene["task"]["horizon_steps"],
                    "implementation_sha256": implementation_hashes, "paid_calls": 0,
                    "runtime": {"python": platform.python_version(), "mujoco": mujoco.__version__,
                                "gymnasium": gymnasium.__version__, "numpy": np.__version__, "opencv": cv2.__version__},
                    "training": False, "scope": "Exposed simulator task evaluation; not a final holdout or physical trial"}
        write_json(out / "settings.json", settings)
        write_json(out / "source-verification.json", verify_shared_world_sources(scene, out))
        results = []
        deadline = started + max_seconds
        for baseline in settings["baselines"]:
            env = SharedExerciseTaskEnv(scene, task)
            recorder = Recorder(out / "recordings" / baseline, recording_scene(scene, task), profile="shared-exercise")
            latencies = []
            try:
                env.reset(seed=seed, options=reset_options)
                recorder.start(env.snapshot(), seed=seed)
                for _ in range(scene["task"]["horizon_steps"]):
                    if time.monotonic() > deadline:
                        raise TimeoutError("Exercise evaluation exceeded its declared wall-time bound")
                    action = np.zeros(env.action_count) if baseline == "coast" else env.authored_action()
                    tick = time.perf_counter()
                    _, reward, terminated, truncated, info = env.step(action)
                    state = env.snapshot()
                    recorder.append({"type": "step", "action": action.tolist(), "reward": reward,
                                     "terminated": terminated, "truncated": truncated, "snapshot": state})
                    latencies.append(time.perf_counter() - tick)
                    if terminated or truncated:
                        break
                recorder.close()
                if time.monotonic() > deadline:
                    raise TimeoutError("Exercise recording reached the run bound before export")
                exported = export_recording(recorder.root, deadline=deadline, max_output_bytes=settings["max_recording_export_bytes"])
                if time.monotonic() > deadline:
                    raise TimeoutError("Exercise export reached the run bound before action replay")
                replay = verify_action_replay(recorder.root, max_seconds=deadline - time.monotonic())
                results.append({"baseline": baseline, "metrics": env.metrics(), "recording": exported,
                                "action_replay": replay, "control_record_latency_ms": {
                                    "samples": len(latencies), "p50": float(np.percentile(latencies, 50) * 1000),
                                    "p95": float(np.percentile(latencies, 95) * 1000),
                                    "maximum": max(latencies) * 1000},
                                "observation_status": info["observation_status"]})
            finally:
                if recorder._started is not None and not recorder._closed:
                    recorder.close()
                env.close()
        for name, expected in implementation_hashes.items():
            if digest(implementation_paths[name]) != expected:
                raise RuntimeError("Exercise implementation changed during evaluation; preserve this unfinished run")
        if time.monotonic() > deadline:
            raise TimeoutError("Exercise evaluation exceeded its declared total bound")
        report = {"schema_version": 1, "status": "evaluated", "created_at": local_time(),
                  "scene_sha256": canonical_hash(scene), "task_sha256": canonical_hash(task),
                  "results": results, "wall_seconds": time.monotonic() - started,
                  "physical_validation": "unverified", "trained_policy": False,
                  "source_scope": task["source_scope"], "source_files_verified": proof.report}
        write_json(out / "evaluation.json", report)
        return {"directory": str(out), "manifest_sha256": _seal(out, "evaluated"), **report}
    except Exception as error:
        write_json(out / "failure.json", {"status": "failed", "created_at": local_time(),
                   "error_type": type(error).__name__, "message": str(error)[:2000],
                   "wall_seconds": time.monotonic() - started, "preserved": True})
        _seal(out, "failed")
        raise
