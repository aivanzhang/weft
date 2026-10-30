"""Promote a sealed short-window scene to one complete reference episode without running physics."""

from copy import deepcopy
import hashlib
from importlib.metadata import version
import json
import math
import os
from pathlib import Path
import platform
import re
import stat
import time

from weft.io import canonical_hash, local_time
from weft.shared_exercise_task import GATES_SHA256, validate_shared_exercise_task
from weft.shared_world import FULLSPAN_PROFILE, PROFILE as WINDOW_PROFILE, full_reference_horizon, validate_shared_world
from weft.shared_world_packaging import (
    collect_shared_source_files, read_package_file, read_package_manifest, validate_package_layout, write_package_files,
)
from weft.world_evidence import _check_tree, _nonfinite, _unique_object


PROFILE = "shared-exercise-fullspan-preparation-v1"
MAX_INPUT_FILES = 512
MAX_INPUT_BYTES = 128 * 1024 * 1024
MAX_OUTPUT_FILES = 1024
MAX_OUTPUT_BYTES = 192 * 1024 * 1024
MAX_FILE_BYTES = 64 * 1024 * 1024
MAX_IMPLEMENTATION_BYTES = 16 * 1024 * 1024
MAX_FAILURE_BYTES = 64 * 1024
FINALIZATION_GRACE_SECONDS = 5
IMPLEMENTATION_FILES = (
    "shared_exercise_fullspan.py", "shared_world.py", "shared_world_observation.py", "shared_world_packaging.py",
    "shared_world_evidence.py", "source_equivalence.py", "world_evidence.py", "observations.py", "human.py",
    "mobile_human.py", "whole_body.py", "body_frame.py", "spatial_pose.py", "root_motion.py", "spatial_rigid.py",
    "errors.py", "io.py", "shared_exercise_task.py", "shared_exercise_reset.py",
)
SCOPE = ("One continuous complete original reference interval at unchanged dt and physics. Missing source prefixes, "
         "gaps and assumptions remain preserved. Preparation performs no reset, physics, inference or training; ready "
         "means sealed schema/source admission, not full-video observation, clearance, task success or physical validation.")


def _bytes(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _json(raw):
    try:
        value = json.loads(raw, object_pairs_hook=_unique_object, parse_constant=_nonfinite)
        _check_tree(value)
    except (UnicodeError, json.JSONDecodeError, RecursionError) as error:
        raise ValueError("Full-reference preparation requires bounded strict JSON") from error
    if not isinstance(value, dict):
        raise ValueError("Full-reference documents must be JSON objects")
    return value


def _deadline(deadline):
    if time.monotonic() >= deadline:
        raise TimeoutError("Full-reference preparation exceeded its cooperative time bound")


def _path(value):
    path = Path(value).absolute()
    if ".." in path.parts:
        raise ValueError("Full-reference paths cannot traverse parent directories")
    for index, item in enumerate((path, *path.parents)):
        try:
            info = item.lstat()
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(info.st_mode) or index and not stat.S_ISDIR(info.st_mode):
            raise ValueError("Full-reference paths cannot contain symlinks or non-directory parents")
    return path


def _terminal(raw, name):
    document = _json(raw)
    if "status" in document and document["status"] not in ("ready", "trained", "evaluated"):
        raise ValueError("Full-reference input requires a supported terminal status: " + name)


def _inputs(source, deadline):
    manifest, hashes = read_package_manifest(source)
    _terminal(manifest, "manifest.json")
    if len(hashes) > MAX_INPUT_FILES:
        raise ValueError("Full-reference input exceeds its file bound")
    files, total = {}, len(manifest)
    for name, expected in hashes.items():
        _deadline(deadline)
        mapping_log = re.fullmatch(r"(?:(?:source-run|learning-run|policy-evaluation-run)/)*source-mappings/[0-9a-f]{64}/[^/]+\.log", name)
        if (any(part.startswith(".") or part == "__pycache__" for part in Path(name).parts)
                or name.endswith(".pyc") or name.endswith(".log") and not mapping_log):
            raise ValueError("Full-reference inputs cannot contain hidden files, caches or unrelated logs")
        raw = read_package_file(source, name, max_bytes=min(MAX_FILE_BYTES, max(1, MAX_INPUT_BYTES - total)),
                                expected_sha256=expected)
        total += len(raw)
        if total > MAX_INPUT_BYTES:
            raise ValueError("Full-reference input exceeds its byte bound")
        files[name] = raw
    for name in ("status.json", "report.json", "reconstruction.json", "evaluation.json", "training.json"):
        if name in files:
            _terminal(files[name], name)
    _deadline(deadline)
    return manifest, files


def _implementation(deadline):
    package = Path(__file__).parent
    files, size = {}, 0
    for name in IMPLEMENTATION_FILES:
        _deadline(deadline)
        raw = read_package_file(package, name, max_bytes=min(8 * 1024 * 1024, max(1, MAX_IMPLEMENTATION_BYTES - size)))
        files[name], size = raw, size + len(raw)
        if size > MAX_IMPLEMENTATION_BYTES:
            raise ValueError("Full-reference implementation exceeds its byte bound")
    for name in ("pyproject.toml", "requirements.lock", "requirements-lab.lock"):
        files["project/" + name] = read_package_file(package.parent.parent, name, max_bytes=1024 * 1024)
    if sum(map(len, files.values())) > MAX_IMPLEMENTATION_BYTES:
        raise ValueError("Full-reference implementation closure exceeds its byte bound")
    return files


def _source_views(scene, files, proof):
    views = {name: files[name] for name in ("source.json", "pose-quality.json") if name in files}
    names = [name for name in files if re.fullmatch(r"source\.(mp4|webm|mov|m4v|avi|mkv)", name)]
    source_hash = scene["source_binding"]["source_sha256"]
    if len(names) > 1 or names and _sha(files[names[0]]) != source_hash:
        raise ValueError("Full-reference source view must match one exact canonical original video")
    if names:
        views[names[0]] = files[names[0]]
    elif source_hash is not None:
        videos = {Path(name).name: raw for name, raw in proof.files.items()
                  if re.fullmatch(r"source\.(mp4|webm|mov|m4v|avi|mkv)", Path(name).name) and _sha(raw) == source_hash}
        if len(videos) > 1:
            raise ValueError("Full-reference source mappings contain ambiguous original video views")
        views.update(videos)
    return views


def _unchanged(source, manifest, implementation, deadline):
    read_package_file(source, "manifest.json", max_bytes=8 * 1024 * 1024, expected_sha256=_sha(manifest))
    for name, expected in _json(manifest)["files"].items():
        _deadline(deadline)
        read_package_file(source, name, max_bytes=MAX_FILE_BYTES, expected_sha256=expected)
    package = Path(__file__).parent
    for name, raw in implementation.items():
        _deadline(deadline)
        root = package.parent.parent if name.startswith("project/") else package
        read_package_file(root, name.removeprefix("project/"), max_bytes=8 * 1024 * 1024, expected_sha256=_sha(raw))


def _plan(files):
    validate_package_layout(files)
    if (len(files) + 2 > MAX_OUTPUT_FILES or sum(map(len, files.values())) + 1024 * 1024 > MAX_OUTPUT_BYTES
            or any(len(raw) > MAX_FILE_BYTES for raw in files.values())):
        raise ValueError("Full-reference complete custody exceeds its output bounds")


def _manifest(files, status):
    return _bytes({"schema_version": 1, "profile": PROFILE, "status": status,
                   "files": {name: _sha(raw) for name, raw in sorted(files.items())}})


def _preserve_failed(out, name):
    # Preserve this attempt's bytes while removing runnable/ready root names.
    fd = os.open(out, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        try:
            info = os.stat(name, dir_fd=fd, follow_symlinks=False)
        except FileNotFoundError:
            return
        if not stat.S_ISREG(info.st_mode):
            raise ValueError("Full-reference failure preservation requires regular files")
        saved = name.removesuffix(".json") + ".publication-failed.json"
        os.link(name, saved, src_dir_fd=fd, dst_dir_fd=fd, follow_symlinks=False)
        os.unlink(name, dir_fd=fd)
    finally:
        os.close(fd)


def _failed(out, report, error):
    report = deepcopy(report)
    report.update(status="failed", error={"type": type(error).__name__, "message": str(error)[:2000]},
                  runnable_scene_published=False)
    out.mkdir(parents=True, mode=0o700, exist_ok=True)
    for name in ("scene.json", "exercise-task.json", "fullspan.json", "manifest.json"):
        _preserve_failed(out, name)
    report["custody"] = {"mode": "preserved_attempt" if any(out.iterdir()) else "receipt_only",
                         "complete_original_archive_verified": False,
                         "scope": "Verified written bytes remain here; original inputs remain at the recorded source path."}
    raw = _bytes(report)
    if len(raw) > MAX_FAILURE_BYTES // 2:
        report = {key: report[key] for key in ("schema_version", "profile", "status", "source", "scope", "error", "custody")}
        raw = _bytes(report)
    write_package_files(out, {"fullspan.json": raw})
    files, total, deadline = {}, 0, time.monotonic() + FINALIZATION_GRACE_SECONDS
    try:
        for path in sorted(out.rglob("*")):
            _deadline(deadline)
            if path.is_symlink():
                raise ValueError("Full-reference output cannot seal symlinks")
            if not path.is_file():
                continue
            data = read_package_file(out, path.relative_to(out).as_posix(), max_bytes=MAX_FILE_BYTES)
            files[path.relative_to(out).as_posix()] = data
            total += len(data)
            if len(files) + 1 > MAX_OUTPUT_FILES or total > MAX_OUTPUT_BYTES:
                raise ValueError("Failed output exceeds complete-custody bounds")
        sealed = _manifest(files, "failed")
        if total + len(sealed) > MAX_OUTPUT_BYTES:
            raise ValueError("Failed output manifest exceeds its bound")
    except (ValueError, OSError, TimeoutError):
        sealed = _bytes({"schema_version": 1, "profile": PROFILE, "status": "failed", "custody": "receipt_only",
                         "unsealed_outputs_preserved": True, "files": {"fullspan.json": _sha(raw)}})
    write_package_files(out, {"manifest.json": sealed})
    return report | {"directory": str(out), "manifest_sha256": _sha(sealed)}


def prepare_fullspan_shared_exercise(run, out, *, max_seconds=30):
    """Seal a v2 complete-reference scene/task and archive the exact original package, without physics."""
    if type(max_seconds) not in (int, float) or not math.isfinite(max_seconds) or not 1 <= max_seconds <= 30:
        raise ValueError("Full-reference preparation requires a 1–30 second bound")
    source, out = _path(run), _path(out)
    if out.exists():
        raise FileExistsError("Full-reference preparation requires a new output directory")
    if source == out or source.is_relative_to(out) or out.is_relative_to(source):
        raise ValueError("Full-reference output must be outside its immutable source")
    started = time.monotonic()
    deadline = started + max_seconds
    report = {"schema_version": 1, "profile": PROFILE, "status": "preparing", "created_at": local_time(),
              "source": str(source), "scope": SCOPE, "max_seconds": max_seconds,
              "physics_steps": 0, "resets": 0, "training_episodes": 0, "inference_calls": 0, "api_calls": 0,
              "video_decodes": 0, "old_policy_or_evaluation_attached": False, "runnable_scene_published": False}
    try:
        manifest, originals = _inputs(source, deadline)
        report.update(input_manifest_sha256=_sha(manifest), input_files_verified=len(originals))
        if not {"scene.json", "exercise-task.json"} <= set(originals):
            raise ValueError("Full-reference preparation requires both a sealed scene and sealed exercise task")
        base = validate_shared_world(_json(originals["scene.json"]))
        if base["profile"] != WINDOW_PROFILE:
            raise ValueError("Full-reference promotion accepts only original shared-world-v1 scenes")
        if base.get("learned_policy_available") is True:
            raise ValueError("Full-reference promotion cannot preserve a learned-policy-ready flag for a different task")
        task = validate_shared_exercise_task(_json(originals["exercise-task.json"]), base)
        if task["contact_requirements"]:
            raise ValueError("Full-reference promotion cannot reinterpret or extend existing contact requirements")
        proof = collect_shared_source_files(base, source, expected_manifest_sha256=_sha(manifest))
        proof.validate_layout(reserved_paths={"scene.json", "exercise-task.json", "source-run", "implementation",
                                             "fullspan.json", "source-verification.json", "manifest.json"})
        candidate = deepcopy(base)
        candidate["profile"] = FULLSPAN_PROFILE
        candidate["task"].update(start_offset_s=0., horizon_steps=full_reference_horizon(base["human_spec"]["reference"], base["dt"]))
        candidate = validate_shared_world(candidate)
        updated = deepcopy(task)
        updated.update(scene_sha256=canonical_hash(candidate),
                       interval={key: candidate["task"][key] for key in ("start_offset_s", "horizon_steps")})
        updated = validate_shared_exercise_task(updated, candidate)
        implementation = _implementation(deadline)
        archive = {"source-run/manifest.json": manifest, **{"source-run/" + name: raw for name, raw in originals.items()}}
        files = {**archive, **dict(proof.files), **_source_views(base, originals, proof),
                 **{"implementation/" + name: raw for name, raw in implementation.items()},
                 "scene.json": _bytes(candidate), "exercise-task.json": _bytes(updated),
                 "source-verification.json": _bytes(proof.report)}
        report.update(status="ready", source_scene_sha256=canonical_hash(base), scene_sha256=canonical_hash(candidate),
            source_task_sha256=canonical_hash(task), task_sha256=canonical_hash(updated), gates_sha256=GATES_SHA256,
            scene_changed_fields=["profile", "task.start_offset_s", "task.horizon_steps"],
            task_changed_fields=["scene_sha256", "interval"], original_manifest_path="source-run/manifest.json",
            reference_interval={"first_time_s": base["human_spec"]["reference"][0]["time_s"],
                                "last_time_s": base["human_spec"]["reference"][-1]["time_s"],
                                "horizon_steps": updated["interval"]["horizon_steps"], "dt": base["dt"],
                                "substeps": base["substeps"], "reference_poses": len(base["human_spec"]["reference"])},
            preserved={"physical_scene": True, "complete_reference_arrays": True, "source_clocks_and_prefixes": True,
                       "gates_reward_root_assistance": True, "source_scope": True, "all_original_files": True},
            implementation_sha256={name: _sha(raw) for name, raw in implementation.items()},
            runtime_versions={"python": platform.python_version(), **{name: version(name) for name in ("numpy", "mujoco", "gymnasium")}},
            bounds={"max_input_files": MAX_INPUT_FILES, "max_input_bytes": MAX_INPUT_BYTES,
                    "max_output_files": MAX_OUTPUT_FILES, "max_output_bytes": MAX_OUTPUT_BYTES,
                    "max_reference_steps": 1800, "max_reference_seconds": 60, "max_failure_receipt_bytes": MAX_FAILURE_BYTES},
            runtime_compatibility="Old policy/runtime identities are not migrated or silently accepted; use their pinned implementation.",
            seconds_before_publication=time.monotonic() - started, runnable_scene_published=True)
        files["fullspan.json"] = _bytes(report)
        _plan(files)
        _unchanged(source, manifest, implementation, deadline)
        _deadline(deadline)
        out.mkdir(parents=True, mode=0o700, exist_ok=False)
        write_package_files(out, files)
        _unchanged(source, manifest, implementation, deadline)
        sealed = _manifest(files, "ready")
        if sum(map(len, files.values())) + len(sealed) > MAX_OUTPUT_BYTES:
            raise ValueError("Full-reference manifest exceeds its total output bound")
        _deadline(deadline)
        write_package_files(out, {"manifest.json": sealed})
        _deadline(deadline)
        return report | {"directory": str(out), "manifest_sha256": _sha(sealed), "seconds": time.monotonic() - started}
    except Exception as error:
        return _failed(out, report, error)
