"""Immutable human-only labs preserve the original shared scene and independent task."""

from copy import deepcopy
import hashlib
from importlib.metadata import version
import json
import math
import platform
import re
from pathlib import Path
import time

from weft.io import canonical_hash, local_time
from weft.shared_exercise_recording import recording_scene, validate_action, verify_action_replay
from weft.shared_exercise_task import SharedExerciseTaskEnv, build_shared_exercise_task, validate_shared_exercise_task
from weft.shared_world import SharedWorldEnv, validate_shared_world
from weft.shared_world_packaging import (
    collect_shared_source_files, read_package_file, read_package_manifest, validate_package_layout, write_package_files,
)


CONTROL_PROFILE = "shared_exercise"
MAX_INPUT_FILES = 512
MAX_OUTPUT_FILES = 1024
MAX_INPUT_BYTES = 128 * 1024 * 1024
MAX_OUTPUT_BYTES = 192 * 1024 * 1024
MAX_FILE_BYTES = 64 * 1024 * 1024
MAX_RUNTIME_BYTES = 16 * 1024 * 1024
RESET_OPTIONS = {"jitter": False, "jitter_scale": 0.0}
RUNTIME_FILES = (
    "shared_exercise_task.py", "shared_world.py", "shared_world_observation.py", "shared_world_evidence.py",
    "source_equivalence.py", "world_evidence.py", "observations.py", "human.py", "mobile_human.py",
    "whole_body.py", "spatial_rigid.py", "body_frame.py", "spatial_pose.py", "root_motion.py", "errors.py", "io.py",
)
REJECTED_STATUSES = frozenset({"failed", "error", "unsupported", "rejected", "cancelled"})
SOURCE_VIEWS = frozenset({
    "source.json", "source.mp4", "source.webm", "source.mov", "source.m4v", "source.avi", "source.mkv",
    "pose-quality.json", "camera-reference.json", "two-view-root.json", "evidence.json", "tracks.csv",
    "reconstruction.json", "reference-feasibility.json", "coverage-summary.json", "reconstruction-coverage.json",
    "geometry-proposal.json", "geometry-study.json",
})


def _json(raw):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("Exercise lab JSON contains duplicate fields")
            result[key] = value
        return result

    def nonfinite(_):
        raise ValueError("Exercise lab JSON must be finite")

    try:
        return json.loads(raw, object_pairs_hook=unique, parse_constant=nonfinite)
    except (UnicodeError, json.JSONDecodeError, RecursionError) as error:
        raise ValueError("Exercise lab requires strict bounded JSON") from error


def _bytes(value):
    return (json.dumps(value, indent=2, allow_nan=False) + "\n").encode()


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _deadline(deadline):
    if time.monotonic() > deadline:
        raise TimeoutError("Exercise lab operation exceeded its cooperative wall-time bound")


def _verified_inputs(directory, *, deadline, maximum_files=MAX_INPUT_FILES, maximum_bytes=MAX_INPUT_BYTES):
    manifest_raw, hashes = read_package_manifest(directory)
    manifest = _json(manifest_raw)
    if manifest.get("status") in REJECTED_STATUSES:
        raise ValueError("Cannot prepare or resume an unsuccessful exercise lab input")
    if len(hashes) > maximum_files:
        raise ValueError("Exercise lab input exceeds its file-count bound")
    files, total = {}, len(manifest_raw)
    for name, expected in hashes.items():
        _deadline(deadline)
        required_log = re.fullmatch(r"(?:source-run/)*source-mappings/[0-9a-f]{64}/[^/]+\.log", name)
        if (any(part.startswith(".") or part == "__pycache__" for part in Path(name).parts)
                or name.endswith(".pyc") or name.endswith(".log") and not required_log):
            raise ValueError("Exercise lab inputs must exclude hidden files, caches and logs")
        raw = read_package_file(directory, name, max_bytes=min(MAX_FILE_BYTES, max(1, maximum_bytes - total)),
                                expected_sha256=expected)
        total += len(raw)
        if total > maximum_bytes:
            raise ValueError("Exercise lab input exceeds its byte bound")
        files[name] = raw
    _deadline(deadline)
    return manifest_raw, files


def _implementation_snapshot():
    from weft.lab import PUBLIC_ASSETS

    package = Path(__file__).parent
    names = {path.name for path in package.glob("*.py") if not path.name.startswith(".")}
    names.update("assets/" + name for name in PUBLIC_ASSETS.values())
    names.add("assets/shared-world-replay.html")
    if len(names) > MAX_INPUT_FILES:
        raise ValueError("Exercise lab implementation exceeds its file-count bound")
    result, total = {}, 0
    for name in sorted(names):
        raw = read_package_file(package, name, max_bytes=min(8 * 1024 * 1024, max(1, MAX_RUNTIME_BYTES - total)))
        total += len(raw)
        if total > MAX_RUNTIME_BYTES:
            raise ValueError("Exercise lab implementation exceeds its byte bound")
        result[name] = raw
    project = package.parent.parent
    for name in ("pyproject.toml", "requirements.lock", "requirements-lab.lock"):
        result["project/" + name] = read_package_file(project, name, max_bytes=1024 * 1024)
    return result


def _runtime_versions():
    return {"python": platform.python_version(), **{name: version(name) for name in ("mujoco", "numpy", "gymnasium")}}


def _reference_frames(scene, deadline):
    env = SharedWorldEnv(scene)
    try:
        env.reset(seed=0, options=RESET_OPTIONS)
        frames = []
        for index in range(min(180, len(scene["human_spec"]["reference"]))):
            _deadline(deadline)
            frames.append(env.reference_snapshot(index))
        return {"schema_version": 1, "profile": "shared-world-kinematic-reference-v1", "frames": frames,
                "scope": "Kinematic body projection with initial passive equipment; no physics actions or grasp inference."}
    finally:
        env.close()


def prepare_shared_exercise_lab(run, out, *, root_assistance=False, task_path=None, max_seconds=60):
    """Prepare a new sealed lab without changing the source scene, task gates or legacy labs."""
    if type(root_assistance) is not bool or task_path is not None and root_assistance:
        raise ValueError("Choose an explicit task or default-task root assistance, not both")
    if type(max_seconds) not in (int, float) or not math.isfinite(max_seconds) or not 1 <= max_seconds <= 180:
        raise ValueError("Exercise lab preparation requires a 1–180 second bound")
    source, out = Path(run).resolve(), Path(out).resolve()
    if out.exists():
        raise FileExistsError("Exercise lab requires a new output directory")
    if out == source or out.is_relative_to(source) or source.is_relative_to(out):
        raise ValueError("Exercise lab output must be outside the immutable input")
    source = source / "run" if (source / "run" / "manifest.json").is_file() else source
    started = time.monotonic()
    deadline = started + max_seconds
    input_manifest, inputs = _verified_inputs(source, deadline=deadline)
    if "scene.json" not in inputs:
        raise ValueError("Exercise lab input requires a sealed shared-world scene")
    for name in ("status.json", "report.json", "evaluation.json", "reconstruction.json"):
        if name in inputs:
            document = _json(inputs[name])
            if isinstance(document, dict) and document.get("status") in REJECTED_STATUSES:
                raise ValueError(f"Exercise lab source artifact {name} has unsuccessful status")
    scene = validate_shared_world(_json(inputs["scene.json"]))
    proof = collect_shared_source_files(scene, source, expected_manifest_sha256=_sha(input_manifest))
    task_raw = None
    if task_path is None:
        task = build_shared_exercise_task(scene, root_assistance=root_assistance)
    else:
        path = Path(task_path).absolute()
        task_raw = read_package_file(path.parent, path.name, max_bytes=64 * 1024)
        task = validate_shared_exercise_task(_json(task_raw), scene)
    generated = {"manifest.json", "source-run", "lab-implementation", "scene.json", "exercise-task.json",
                 "exercise-lab.json", "source-verification.json", "human-reference.json", "input-task.json"}
    proof.validate_layout(reserved_paths=generated)
    implementation = _implementation_snapshot()
    contract = {"schema_version": 1, "profile": "shared-world", "control_profile": CONTROL_PROFILE,
                "scene_sha256": canonical_hash(scene), "task_sha256": canonical_hash(task),
                "reset_options": dict(RESET_OPTIONS), "learned_policy_available": False,
                "source_manifest_sha256": _sha(input_manifest), "source_files_verified": len(inputs),
                "implementation_sha256": {name: _sha(raw) for name, raw in implementation.items()},
                "runtime_sha256": {name: _sha(implementation[name]) for name in RUNTIME_FILES},
                "runtime_versions": _runtime_versions(),
                "scope": "Independent authored human-only simulator task; no recovered grasp or physical validation."}
    views = {name: raw for name, raw in inputs.items()
             if name in SOURCE_VIEWS or name.startswith("geometry-inspection/")}
    videos = [name for name in views if re.fullmatch(r"source\.(mp4|webm|mov|m4v|avi|mkv)", name)]
    if len(videos) > 1:
        raise ValueError("Exercise lab source contains ambiguous original video files")
    if videos and scene["source_binding"]["mode"] == "source_qualified":
        if _sha(views[videos[0]]) != scene["source_binding"]["source_sha256"]:
            raise ValueError("Exercise lab original video differs from its canonical source binding")
    if not videos and scene["source_binding"]["mode"] == "source_qualified":
        for name, raw in proof.files.items():
            basename = Path(name).name
            if (name.startswith("source-mappings/") and re.fullmatch(r"source\.(mp4|webm|mov|m4v|avi|mkv)", basename)
                    and _sha(raw) == scene["source_binding"]["source_sha256"]):
                views[basename] = raw
                break
    files = {"source-run/manifest.json": input_manifest, **{"source-run/" + name: raw for name, raw in inputs.items()},
             **dict(proof.files), **views,
             "scene.json": inputs["scene.json"], "exercise-task.json": task_raw or _bytes(task),
             "exercise-lab.json": _bytes(contract), "source-verification.json": _bytes(proof.report),
             "human-reference.json": _bytes(_reference_frames(scene, deadline)),
             **{"lab-implementation/" + name: raw for name, raw in implementation.items()}}
    if task_raw is not None:
        files["input-task.json"] = task_raw
    validate_package_layout(files)
    _deadline(deadline)
    manifest = _bytes({"schema_version": 1, "created_at": local_time(), "status": "ready",
                       "control_profile": CONTROL_PROFILE, "files": {name: _sha(raw) for name, raw in files.items()}})
    event = {"stage": "shared_exercise_import", "message": "Verified immutable human-only task and passive equipment",
             "time": local_time()}
    config = {key: contract[key] for key in ("schema_version", "profile", "control_profile", "scene_sha256", "task_sha256",
                                           "reset_options", "learned_policy_available")}
    config.update(created_at=local_time(), source_run=str(source), manifest_sha256=_sha(manifest))
    outer = {"lab-config.json": _bytes(config), "status.json": _bytes({"status": "ready", "events": [event]}),
             "events.jsonl": (json.dumps(event) + "\n").encode()}
    total = sum(len(raw) for raw in files.values()) + len(manifest) + sum(len(raw) for raw in outer.values())
    if len(files) + len(outer) + 1 > MAX_OUTPUT_FILES or total > MAX_OUTPUT_BYTES:
        raise ValueError("Exercise lab complete source/runtime closure exceeds its output bound")
    out.mkdir(parents=True, exist_ok=False)
    (out / "run").mkdir()
    try:
        write_package_files(out / "run", files)
        _deadline(deadline)
        for name, expected in contract["implementation_sha256"].items():
            package = Path(__file__).parent
            directory, relative = (package.parent.parent, name[8:]) if name.startswith("project/") else (package, name)
            read_package_file(directory, relative, max_bytes=8 * 1024 * 1024, expected_sha256=expected)
        write_package_files(out / "run", {"manifest.json": manifest})
        load_shared_exercise_lab(out / "run", config=config, deadline=deadline)
        _deadline(deadline)
        write_package_files(out, outer)
    except Exception as error:
        write_package_files(out, {"failure.json": _bytes({"status": "failed", "message": str(error)[:2000],
                                  "time": local_time(), "scope": "Incomplete preparation preserved; not a resumable ready lab."})})
        raise
    return {"lab_directory": str(out), "run_directory": str(out / "run"), "manifest_sha256": _sha(manifest),
            "control_profile": CONTROL_PROFILE, "task_sha256": canonical_hash(task), "scene_sha256": canonical_hash(scene),
            "learned_policy_available": False, "files": len(files), "bytes": total,
            "seconds": time.monotonic() - started}


def load_shared_exercise_lab(root, *, config=None, deadline=None):
    """Verify the complete sealed package before exposing its immutable task and scene."""
    root = Path(root)
    deadline = time.monotonic() + 60 if deadline is None else deadline
    manifest_raw, files = _verified_inputs(root, deadline=deadline, maximum_files=MAX_OUTPUT_FILES,
                                           maximum_bytes=MAX_OUTPUT_BYTES)
    if _json(manifest_raw).get("control_profile") != CONTROL_PROFILE:
        raise ValueError("Exercise lab manifest must explicitly select human-only control")
    required = {"scene.json", "exercise-task.json", "exercise-lab.json", "source-run/manifest.json"}
    if not required <= set(files):
        raise ValueError("Exercise lab is missing its pinned scene, task or source identity")
    scene = validate_shared_world(_json(files["scene.json"]))
    task = validate_shared_exercise_task(_json(files["exercise-task.json"]), scene)
    contract = _json(files["exercise-lab.json"])
    fixed = {"schema_version": 1, "profile": "shared-world", "control_profile": CONTROL_PROFILE,
             "scene_sha256": canonical_hash(scene), "task_sha256": canonical_hash(task),
             "reset_options": RESET_OPTIONS, "learned_policy_available": False,
             "source_manifest_sha256": _sha(files["source-run/manifest.json"])}
    if (not isinstance(contract, dict) or type(contract.get("schema_version")) is not int
            or contract.get("learned_policy_available") is not False
            or any(contract.get(key) != value for key, value in fixed.items())):
        raise ValueError("Exercise lab contract differs from its pinned task, scene or reset protocol")
    implementation = contract.get("implementation_sha256")
    if (not isinstance(implementation, dict) or not implementation or any(not isinstance(name, str) for name in implementation)
            or any(
            "lab-implementation/" + name not in files or _sha(files["lab-implementation/" + name]) != expected
            for name, expected in implementation.items())):
        raise ValueError("Exercise lab runtime closure differs from its pinned implementation")
    expected_runtime = {name: implementation.get(name) for name in RUNTIME_FILES}
    if (any(value is None for value in expected_runtime.values()) or contract.get("runtime_sha256") != expected_runtime
            or contract.get("runtime_versions") != _runtime_versions()):
        raise ValueError("Exercise physics/evaluator runtime differs; use the pinned implementation or prepare a new lab")
    for name, expected in expected_runtime.items():
        _deadline(deadline)
        try:
            read_package_file(Path(__file__).parent, name, max_bytes=8 * 1024 * 1024, expected_sha256=expected)
        except ValueError as error:
            raise ValueError("Exercise physics/evaluator runtime differs; use the pinned implementation or prepare a new lab") from error
    original = _json(files["source-run/manifest.json"])
    if (not isinstance(original, dict) or not isinstance(original.get("files"), dict)
            or any(not isinstance(name, str) for name in original["files"])
            or type(contract.get("source_files_verified")) is not int or contract["source_files_verified"] != len(original["files"])
            or any("source-run/" + name not in files or _sha(files["source-run/" + name]) != expected
                   for name, expected in original["files"].items())):
        raise ValueError("Exercise lab must retain the complete sealed source candidate")
    if files["scene.json"] != files.get("source-run/scene.json"):
        raise ValueError("Exercise lab cannot rewrite the original engine scene")
    videos = [name for name in files if re.fullmatch(r"source\.(mp4|webm|mov|m4v|avi|mkv)", name)]
    if len(videos) > 1 or (videos and scene["source_binding"]["mode"] == "source_qualified"
                          and _sha(files[videos[0]]) != scene["source_binding"]["source_sha256"]):
        raise ValueError("Exercise lab original video differs from its canonical source binding")
    if config is not None:
        config_keys = ("schema_version", "profile", "control_profile", "scene_sha256", "task_sha256", "reset_options",
                       "learned_policy_available")
        if (type(config.get("schema_version")) is not int or config.get("learned_policy_available") is not False
                or any(config.get(key) != contract[key] for key in config_keys)
                or config.get("manifest_sha256") != _sha(manifest_raw)):
            raise ValueError("Exercise lab configuration differs from its sealed scene/task identity")
    proof = collect_shared_source_files(scene, root, expected_manifest_sha256=_sha(manifest_raw))
    _deadline(deadline)
    return {"scene": scene, "task": task, "contract": contract, "source_verification": proof.report,
            "artifacts": [{"name": name, "bytes": len(raw)} for name, raw in sorted(files.items()) if "/" not in name]
                         + [{"name": "manifest.json", "bytes": len(manifest_raw)}]}


class SharedExerciseSession:
    control_profile = CONTROL_PROFILE
    scene_profile = "shared-world-v1"
    policy = None

    def __init__(self, root, *, config=None):
        loaded = load_shared_exercise_lab(root, config=config)
        self._scene, self._task = loaded["scene"], loaded["task"]
        self.artifacts = loaded["artifacts"]
        self.source_binding_verification = loaded["source_verification"]
        self._recording_scene = recording_scene(self._scene, self._task)
        self.env = SharedExerciseTaskEnv(self._scene, self._task)
        try:
            self.reset(0)
            from weft.capabilities import describe_capabilities
            self._capabilities = describe_capabilities(self._scene, lab_profile=CONTROL_PROFILE,
                exercise_task=self._task, reference_available=True,
                source_video_present=any(re.fullmatch(r"source\.(mp4|webm|mov|m4v|avi|mkv)", item["name"])
                                         for item in self.artifacts))
        except Exception:
            self.env.close()
            raise

    @property
    def scene(self):
        return deepcopy(self._scene)

    @property
    def exercise_task(self):
        return deepcopy(self._task)

    def recording_scene(self):
        return deepcopy(self._recording_scene)

    def action_contract(self):
        return self.env.action_contract()

    def capabilities(self):
        return deepcopy(self._capabilities)

    def reset(self, seed):
        self.observation, _ = self.env.reset(seed=seed, options=RESET_OPTIONS)
        self.done = False
        return self.snapshot()

    def policy_action(self):
        raise ValueError("This human-only exercise task has no learned policy; use manual, coast or authored reference")

    def reference_action(self):
        return self.env.authored_action().tolist()

    def resolve_action(self, action):
        validate_action(action, self._recording_scene)
        return [float(value) for value in action]

    def step(self, action):
        self.observation, reward, terminated, truncated, info = self.env.step(self.resolve_action(action))
        self.done = terminated or truncated
        return reward, terminated, truncated, info

    def snapshot(self):
        return self.env.snapshot()


def export_shared_exercise_session(root):
    from weft.recording import export_recording

    deadline = time.monotonic() + 180
    exported = export_recording(root, deadline=deadline, max_output_bytes=128 * 1024 * 1024)
    _deadline(deadline)
    replay = verify_action_replay(root, max_seconds=deadline - time.monotonic())
    return {**exported, "action_replay": replay}
