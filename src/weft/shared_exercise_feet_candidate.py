"""Package a source-bound feet candidate without running or qualifying a simulation."""

from copy import deepcopy
import math
import os
from pathlib import Path
import re
import tempfile
import time

from weft import shared_exercise_lab as lab
from weft.io import canonical_hash, local_time
from weft.shared_exercise_feet_projection import derive_base_scoring_scene
from weft.shared_exercise_feet_task import ADAPTER_FILES, build_shared_exercise_feet_task
from weft.shared_exercise_task import validate_shared_exercise_task
from weft.shared_world import add_feet_to_shared_world, validate_shared_world
from weft.shared_world_packaging import (
    collect_shared_source_files, read_package_file, validate_package_layout, write_package_files,
)


PROFILE = "shared-exercise-feet-candidate-v1"
MIGRATION_PROFILE = "shared-exercise-feet-migration-v1"
BASE_PROFILES = frozenset({"shared-world-v2", "shared-world-v3", "shared-world-v4"})
MAX_FILES = 512
MAX_BYTES = 128 << 20
MAX_REFERENCE_BYTES = 8 << 20
MAX_TASK_BYTES = 64 << 10
RESERVE_BYTES = 256 << 10
CONSTRUCTOR_FILES = ("shared_exercise_feet_candidate.py", "shared_exercise_lab.py", "shared_world_packaging.py")
GENERATED = frozenset({
    "manifest.json", "failure.json", "source-run", "candidate-implementation", "scene.json",
    "exercise-task.json", "base-scene.json", "base-exercise-task.json", "input-task.json",
    "foot-reference.json", "feet-migration.json", "source-verification.json",
})
INACTIVE = frozenset({"policy.json", "learning-run", "policy-evaluation-run", "policy-evaluation.json",
                      "avatar.json", "avatar-manifest.json", "avatar", "avatars"})
SCOPE = {
    "simulation_verified": False, "lab_created": False, "physical_validation": "unverified",
    "ankle_target_mode": "authored_initial_angle_hold", "video_ankle_imitation": False,
    "foot_support_credit": False, "ground_load_qualification": "unsupported",
    "learning": "unsupported", "avatars": "unsupported",
}


def _path(value):
    path = Path(value).absolute()
    if ".." in path.parts or any(part.is_symlink() for part in (path, *path.parents)):
        raise ValueError("Feet candidate paths must not contain traversal or symlinks")
    return path


def _identity(path):
    stat = path.stat()
    return stat.st_dev, stat.st_ino


def _publish_manifest(target, raw, deadline, identity):
    # Stage outside the package so a late/failed write cannot look terminal or add an unsealed member.
    stage = Path(tempfile.mkdtemp(prefix=".weft-feet-seal-", dir=target.parent))
    try:
        write_package_files(stage, {"manifest.json": raw})
        lab._deadline(deadline)
        if _path(target) != target or _identity(target) != identity:
            raise ValueError("Feet candidate output directory changed before publication")
        fd = os.open(target, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        try:
            stat = os.fstat(fd)
            if (stat.st_dev, stat.st_ino) != identity:
                raise ValueError("Feet candidate output directory changed before publication")
            lab._deadline(deadline)
            os.link(stage / "manifest.json", "manifest.json", dst_dir_fd=fd, follow_symlinks=False)
        finally:
            os.close(fd)
    finally:
        # Cleanup cannot relabel an already published terminal result.
        try:
            (stage / "manifest.json").unlink(missing_ok=True)
            stage.rmdir()
        except OSError:
            pass


def _not_failed(source, inputs=None):
    if (source / "failure.json").exists() or (source / "failure.json").is_symlink():
        raise ValueError("A failed source cannot become a feet candidate")
    for name in ("status.json", "report.json", "evaluation.json", "reconstruction.json"):
        if inputs is not None and name in inputs:
            value = lab._json(inputs[name])
            if isinstance(value, dict) and value.get("status") in lab.REJECTED_STATUSES:
                raise ValueError(f"Feet candidate source artifact {name} has unsuccessful status")


def _implementation(scene, deadline):
    package = Path(__file__).parent
    names = sorted(set(lab.runtime_files(scene)) | set(CONSTRUCTOR_FILES))
    result, total = {}, 0
    for name in names:
        lab._deadline(deadline)
        raw = read_package_file(package, name, max_bytes=min(8 << 20, max(1, lab.MAX_RUNTIME_BYTES - total)))
        result[name], total = raw, total + len(raw)
    for name in ("pyproject.toml", "requirements.lock", "requirements-lab.lock"):
        lab._deadline(deadline)
        raw = read_package_file(package.parent.parent, name, max_bytes=1 << 20)
        result["project/" + name], total = raw, total + len(raw)
    if total > lab.MAX_RUNTIME_BYTES:
        raise ValueError("Feet candidate constructor/runtime exceeds its 16 MiB bound")
    return result


def _views(scene, inputs, proof):
    selected = {name: raw for name, raw in inputs.items()
                if name in lab.SOURCE_VIEWS or name.startswith("geometry-inspection/")
                or name == "source-clock.json"
                or scene["profile"] == "shared-world-v3" and name == "inventory.json"
                or scene["profile"] == "shared-world-v4" and name == "support-config.json"}
    if "geometry-hypothesis.json" in selected and "input-config.json" in inputs:
        selected["input-config.json"] = inputs["input-config.json"]
    if scene["profile"] == "shared-world-v4" and "inputs/support-config.json" in inputs:
        raw = inputs["inputs/support-config.json"]
        if "support-config.json" in selected and selected["support-config.json"] != raw:
            raise ValueError("Feet candidate has conflicting static-support configuration artifacts")
        selected["support-config.json"] = raw
    videos = [name for name in selected if re.fullmatch(r"source\.(mp4|webm|mov|m4v|avi|mkv)", name)]
    if len(videos) > 1:
        raise ValueError("Feet candidate source contains ambiguous original video files")
    binding = scene["source_binding"]
    if binding["mode"] == "source_qualified":
        if videos and lab._sha(selected[videos[0]]) != binding["source_sha256"]:
            raise ValueError("Feet candidate original video differs from its source binding")
        if not videos:
            for name, raw in proof.files.items():
                basename = Path(name).name
                if (name.startswith("source-mappings/")
                        and re.fullmatch(r"source\.(mp4|webm|mov|m4v|avi|mkv)", basename)
                        and lab._sha(raw) == binding["source_sha256"]):
                    selected[basename] = raw
                    break
    return selected


def prepare_shared_feet_candidate(run, reference_path, out, *, task_path=None, max_seconds=60):
    """Create a new bounded candidate; native model validation is not a simulation trial."""
    if type(max_seconds) not in (int, float) or not math.isfinite(max_seconds) or not 1 <= max_seconds <= 180:
        raise ValueError("Feet candidate requires a cooperative 1–180 second bound")
    started = time.monotonic()
    deadline = started + max_seconds
    source, reference, target = _path(run), _path(reference_path), _path(out)
    explicit_task = _path(task_path) if task_path is not None else None
    if target.exists():
        raise FileExistsError("Feet candidate requires a new output directory")
    for path in (source, reference, explicit_task):
        if path is not None and (target == path or target.is_relative_to(path) or path.is_relative_to(target)):
            raise ValueError("Feet candidate output must be outside every immutable input")
    source = _path(source / "run") if (source / "run" / "manifest.json").is_file() else source
    source_identity = _identity(source)
    _not_failed(source)
    manifest_raw, inputs = lab._verified_inputs(source, deadline=deadline,
                                               maximum_files=MAX_FILES, maximum_bytes=MAX_BYTES)
    _not_failed(source, inputs)
    if "construction-seed.json" in inputs or "scene.json" not in inputs:
        raise ValueError("Feet candidate requires a sealed completed shared-world source")
    original_document = lab._json(inputs["scene.json"])
    if not isinstance(original_document, dict) or original_document.get("profile") not in BASE_PROFILES:
        raise ValueError("Feet candidate requires an unextended shared-world-v2/v3/v4 scene")
    if explicit_task is None:
        if "exercise-task.json" not in inputs:
            raise ValueError("Feet candidate requires an exact existing exercise task; no defaults are inferred")
        task_raw = inputs["exercise-task.json"]
    else:
        task_raw = read_package_file(explicit_task.parent, explicit_task.name, max_bytes=MAX_TASK_BYTES)
    if len(task_raw) > MAX_TASK_BYTES:
        raise ValueError("Feet candidate input task exceeds its 64 KiB bound")
    reference_raw = read_package_file(reference.parent, reference.name, max_bytes=MAX_REFERENCE_BYTES)
    lab._deadline(deadline)
    original = validate_shared_world(original_document)
    original_task = validate_shared_exercise_task(lab._json(task_raw), original)
    if canonical_hash(original_task) != canonical_hash(lab._json(task_raw)):
        raise ValueError("Feet candidate cannot normalize the original task requirements")
    proof = collect_shared_source_files(original, source, expected_manifest_sha256=lab._sha(manifest_raw))
    proof.validate_layout(reserved_paths=GENERATED | INACTIVE)
    lab._deadline(deadline)
    actual = add_feet_to_shared_world(original, reference_raw)
    base = derive_base_scoring_scene(actual)
    expected_base = deepcopy(original)
    expected_base["learned_policy_available"] = False
    if base != expected_base:
        raise ValueError("Feet candidate scoring scene changed more than learned_policy_available")
    base_task = deepcopy(original_task)
    base_task["scene_sha256"] = canonical_hash(base)
    if validate_shared_exercise_task(base_task, base) != base_task:
        raise ValueError("Feet candidate may rebind only the original task scene identity")
    outer_task = build_shared_exercise_feet_task(actual, base_task)
    implementation = _implementation(actual, deadline)
    implementation_hashes = {name: lab._sha(raw) for name, raw in implementation.items()}
    adapter_hash = canonical_hash({name: implementation_hashes[name] for name in ADAPTER_FILES})
    if outer_task["adapter_implementation_sha256"] != adapter_hash:
        raise ValueError("Feet adapter changed during candidate preparation")
    runtime_hashes = {name: implementation_hashes[name] for name in lab.runtime_files(actual)}
    evaluator_hashes = {name: runtime_hashes[name] for name in (
        ("shared_exercise_task.py", "shared_support_task.py") if original["profile"] == "shared-world-v4"
        else ("shared_exercise_task.py",))}
    evaluator_hash = (canonical_hash(evaluator_hashes) if original["profile"] == "shared-world-v4"
                      else evaluator_hashes["shared_exercise_task.py"])
    actual_raw, base_raw = lab._bytes(actual), lab._bytes(base)
    outer_raw, base_task_raw = lab._bytes(outer_task), lab._bytes(base_task)
    pins = {
        "source_manifest_sha256": lab._sha(manifest_raw), "source_sha256": original["human_spec"]["source_sha256"],
        "original_scene_sha256": canonical_hash(original), "original_scene_bytes_sha256": lab._sha(inputs["scene.json"]),
        "base_scene_sha256": canonical_hash(base), "base_scene_bytes_sha256": lab._sha(base_raw),
        "scene_sha256": canonical_hash(actual), "scene_bytes_sha256": lab._sha(actual_raw),
        "original_task_sha256": canonical_hash(original_task), "original_task_bytes_sha256": lab._sha(task_raw),
        "base_task_sha256": canonical_hash(base_task), "base_task_bytes_sha256": lab._sha(base_task_raw),
        "task_sha256": canonical_hash(outer_task), "task_bytes_sha256": lab._sha(outer_raw),
        "foot_reference_sha256": lab._sha(reference_raw),
        "foot_reference_canonical_sha256": actual["foot_extension"]["reference_canonical_sha256"],
        "body_model_sha256": canonical_hash(original["human_spec"]),
        "projection_contract_sha256": outer_task["projection_contract_sha256"],
        "adapter_implementation_sha256": adapter_hash, "evaluator_sha256": evaluator_hash,
        "gates_sha256": original_task["gates_sha256"],
    }
    migration = {
        "schema_version": 1, "profile": MIGRATION_PROFILE, "status": "constructed", **pins,
        "base_profile": original["profile"], "task_input": "explicit_path" if explicit_task else "sealed_source",
        "scene_change": {"field": "learned_policy_available", "original_present": "learned_policy_available" in original,
                         "from": original.get("learned_policy_available"), "to": False, "other_fields": "unchanged"},
        "task_rebinding": {"field": "scene_sha256", "from": original_task["scene_sha256"],
                           "to": base_task["scene_sha256"], "other_fields": "unchanged"},
        "interval": deepcopy(original_task["interval"]), "control": deepcopy(original_task["control"]),
        "implementation_sha256": implementation_hashes, "runtime_sha256": runtime_hashes,
        "evaluator_implementation_sha256": evaluator_hashes, "runtime_versions": lab._runtime_versions(),
        "scope": dict(SCOPE), "deadline_semantics": "Cooperative checks before terminal seal; not a hard atomic wall-time cap.",
    }
    views = _views(original, inputs, proof)
    for name in set(views) & set(proof.files):
        if views[name] != proof.files[name]:
            raise ValueError("Feet candidate source views conflict with pinned evidence")
    files = {
        "source-run/manifest.json": manifest_raw, **{"source-run/" + name: raw for name, raw in inputs.items()},
        **dict(proof.files), **views, "scene.json": actual_raw, "exercise-task.json": outer_raw,
        "base-scene.json": base_raw, "base-exercise-task.json": base_task_raw, "input-task.json": task_raw,
        "foot-reference.json": reference_raw, "feet-migration.json": lab._bytes(migration),
        "source-verification.json": lab._bytes(proof.report),
        **{"candidate-implementation/" + name: raw for name, raw in implementation.items()},
    }
    validate_package_layout([*files, "manifest.json", "failure.json"])
    payload_bytes = sum(len(raw) for raw in files.values())
    if len(files) + 2 > MAX_FILES or payload_bytes + RESERVE_BYTES > MAX_BYTES:
        raise ValueError("Feet candidate complete source/runtime package exceeds its 512-file/128 MiB bound")
    hashes = {name: lab._sha(raw) for name, raw in files.items()}
    terminal = lab._bytes({"schema_version": 1, "profile": PROFILE, "status": "constructed",
                          "created_at": local_time(), "scope": dict(SCOPE), "files": hashes})
    if len(terminal) > RESERVE_BYTES // 2:
        raise ValueError("Feet candidate manifest exceeds its reserved bound")

    def guard():
        lab._deadline(deadline)
        if _path(source) != source or _identity(source) != source_identity:
            raise ValueError("Feet candidate source directory changed")
        _not_failed(source)
        for name, raw in {"manifest.json": manifest_raw, **inputs}.items():
            lab._deadline(deadline)
            read_package_file(source, name, max_bytes=max(1, len(raw)), expected_sha256=lab._sha(raw))
        for path, raw in ((reference, reference_raw), (explicit_task, task_raw)):
            if path is not None:
                _path(path)
                read_package_file(path.parent, path.name, max_bytes=max(1, len(raw)), expected_sha256=lab._sha(raw))
        for name, raw in implementation.items():
            lab._deadline(deadline)
            package = Path(__file__).parent
            directory, relative = (package.parent.parent, name[8:]) if name.startswith("project/") else (package, name)
            read_package_file(directory, relative, max_bytes=max(1, len(raw)), expected_sha256=implementation_hashes[name])
        lab._deadline(deadline)

    lab._deadline(deadline)
    _path(target)
    target.mkdir(parents=True, exist_ok=False)
    target_identity, owned = _identity(target), {}
    try:
        for name, raw in files.items():
            lab._deadline(deadline)
            write_package_files(target, {name: raw})
            owned[name] = hashes[name]
        guard()
        if _path(target) != target or _identity(target) != target_identity:
            raise ValueError("Feet candidate output directory changed")
        for name, raw in files.items():
            lab._deadline(deadline)
            read_package_file(target, name, max_bytes=max(1, len(raw)), expected_sha256=owned[name])
        lab._deadline(deadline)
        # No fallible post-seal check may turn a published constructed package into a returned failure.
        _publish_manifest(target, terminal, deadline, target_identity)
    except Exception as error:
        try:
            if _path(target) != target or _identity(target) != target_identity:
                raise ValueError("Changed output directory cannot receive a failure seal")
            unadmitted = []
            for name in tuple(owned):
                try:
                    read_package_file(target, name, max_bytes=max(1, len(files[name])), expected_sha256=owned[name])
                except (OSError, ValueError):
                    unadmitted.append(name)
                    del owned[name]
            failure = lab._bytes({"schema_version": 1, "profile": PROFILE, "status": "failed",
                                  "error": type(error).__name__, "message": str(error)[:2000],
                                  "unadmitted_changed_files": unadmitted,
                                  "created_at": local_time(), "scope": "Incomplete candidate; no lab admission or simulation qualification."})
            write_package_files(target, {"failure.json": failure})
            owned["failure.json"] = lab._sha(failure)
            write_package_files(target, {"manifest.json": lab._bytes({"schema_version": 1, "profile": PROFILE,
                "status": "failed", "files": owned, "unadmitted_partial_files_possible": True})})
        except Exception as seal_error:
            error.add_note("Feet candidate failure evidence could not be sealed: " + str(seal_error)[:500])
        raise
    return {"status": "constructed", "profile": PROFILE, "candidate_directory": str(target),
            "manifest_sha256": lab._sha(terminal), **pins, "scope": dict(SCOPE),
            "files": len(files) + 1, "bytes": payload_bytes + len(terminal), "seconds": time.monotonic() - started}
