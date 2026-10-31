"""Continue one fixed exercise world through learning, comparison and a single playable lab."""

from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import stat
import time

from weft import shared_exercise_residual_learning as residual
from weft.io import canonical_hash, local_time
from weft.shared_exercise_lab import (
    _implementation_snapshot, _json, _runtime_versions, prepare_shared_exercise_lab,
)
from weft.shared_exercise_selection import RULE, select_feasible_shared_exercise_policy
from weft.shared_exercise_task import GATES_SHA256
from weft.shared_world import FULLSPAN_PROFILES, HUMAN_ONLY_PROFILE
from weft.shared_world_packaging import read_package_file, read_package_manifest, write_package_files


PROFILE = "shared-exercise-learning-workflow-v1"
MAX_FILES = 2700
MAX_BYTES = 640 * 1024 * 1024
MAX_SECONDS = 180
FINALIZATION_RESERVE_SECONDS = 5
METADATA_RESERVE_BYTES = 4 * 1024 * 1024
METADATA_RESERVE_FILES = 16
MAX_METADATA_BYTES = 1024 * 1024
STAGE_LIMITS = {
    "training": (512, 128 * 1024 * 1024, 60),
    "training_selection": (512, 128 * 1024 * 1024, 15),
    "independent_evaluation": (512, 128 * 1024 * 1024, 60),
    "interactive_lab": (1024, 192 * 1024 * 1024, 60),
}
DEFAULT_TRAIN_SETTINGS = {"generations": 2, "population": 8, "training_seed": 1729,
                          "trials": 3, "jitter_scale": .10}
FAILED_STATUSES = {"failed", "error", "unsupported", "unsupported_input", "rejected",
                   "cancelled", "running", "incomplete", "partial"}


def validate_train_settings(settings):
    """Resolve bounded search settings; evaluation needs an explicitly chosen seed stream."""
    allowed = {*DEFAULT_TRAIN_SETTINGS, "evaluation_seed", "policy_family"}
    if (not isinstance(settings, dict) or set(settings) - allowed or "evaluation_seed" not in settings):
        raise ValueError("Training settings require explicit evaluation_seed and only supported learning fields")
    result = {**DEFAULT_TRAIN_SETTINGS, **settings}
    if result.get("policy_family", "phase-residual") not in ("phase-residual", "feedback-residual"):
        raise ValueError("Training policy_family must be phase-residual or feedback-residual")
    for name, low, high in (("generations", 1, 6), ("population", 4, 12), ("trials", 1, 3),
                            ("training_seed", 0, 2**31 - 1), ("evaluation_seed", 0, 2**31 - 1)):
        residual._integer(result[name], name, low, high)
    if result["training_seed"] == result["evaluation_seed"]:
        raise ValueError("Training and evaluation must use separate seed streams")
    jitter = result["jitter_scale"]
    if type(jitter) not in (int, float) or not math.isfinite(jitter) or not 0 < jitter <= .25:
        raise ValueError("Evaluation jitter_scale must be finite and in (0,.25]")
    result["jitter_scale"] = float(jitter)
    return result


def _learning_backend(settings, scene):
    if settings.get("policy_family", "phase-residual") == "feedback-residual":
        from weft import shared_exercise_feedback_learning as feedback
        from weft import shared_exercise_feedback_residual as actor

        return (actor.POLICY_PROFILE, actor.ALGORITHM, lambda: actor.runtime_hashes(scene),
                feedback.train_shared_exercise_feedback_policy, feedback.select_shared_exercise_feedback_policy,
                feedback.evaluate_shared_exercise_feedback_policy)
    return (residual.POLICY_PROFILE, residual.ALGORITHM,
            lambda: residual._runtime_hashes(scene) if scene["profile"] == HUMAN_ONLY_PROFILE else residual._runtime_hashes(),
            residual.train_shared_exercise_residual_policy, select_feasible_shared_exercise_policy,
            residual.evaluate_shared_exercise_residual_policy)


def _bytes(value):
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _check(deadline):
    if time.monotonic() >= deadline:
        raise TimeoutError("Exercise learning workflow exceeded its cooperative wall-time bound")


def _plain_path(value):
    path = Path(value)
    if ".." in path.parts:
        raise ValueError("Workflow paths cannot contain parent-directory traversal")
    path = path.absolute()
    for parent in (path, *path.parents):
        if parent.is_symlink():
            raise ValueError("Workflow paths must not contain symbolic links")
    return path


def _reject_failure_markers(root):
    # Failure markers can be written after a seal; absence from its file map does not admit them.
    with os.scandir(root) as entries:
        for index, entry in enumerate(entries):
            if index >= MAX_FILES:
                raise ValueError("Workflow input has too many top-level entries")
            if entry.name == "failure.json" or entry.name.endswith(("-failure.json", "_failure.json")):
                raise ValueError("Workflow input retains a failure marker")


def _runtime():
    return {name: _sha(raw) for name, raw in _implementation_snapshot().items()}, _runtime_versions()


def _inventory(root, deadline, *, immutable_only=False):
    files, total, directories, visited = {}, 0, [root], 0
    while directories:
        _check(deadline)
        with os.scandir(directories.pop()) as entries:
            for entry in entries:
                _check(deadline)
                name = Path(entry.path).relative_to(root).as_posix()
                # New sessions/lifecycle metadata never change the completed workflow seal.
                if immutable_only and name.startswith("lab/") and name != "lab/run" and not name.startswith("lab/run/"):
                    continue
                info = entry.stat(follow_symlinks=False)
                if stat.S_ISDIR(info.st_mode):
                    visited += 1
                    if visited > MAX_FILES * 4:
                        raise ValueError("Workflow output exceeds its directory bound")
                    directories.append(Path(entry.path))
                elif stat.S_ISREG(info.st_mode):
                    if info.st_size > 64 * 1024 * 1024:
                        raise ValueError("Workflow output exceeds its per-file bound")
                    files[name] = info.st_size
                    total += info.st_size
                    if len(files) > MAX_FILES or total > MAX_BYTES:
                        raise ValueError("Workflow output exceeds its aggregate file/byte bound")
                else:
                    raise ValueError("Workflow outputs must be regular files without symbolic links")
    return files, total


def _metadata(root, name, value):
    raw = _bytes(value)
    if len(raw) > MAX_METADATA_BYTES:
        raise ValueError("Workflow metadata exceeds its bounded reserve")
    root_fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        fd = os.open(name, os.O_WRONLY | os.O_CREAT | os.O_NOFOLLOW | os.O_NONBLOCK, 0o600, dir_fd=root_fd)
        with os.fdopen(fd, "wb") as stream:
            if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
                raise ValueError("Workflow metadata must be a regular file")
            stream.truncate(0)
            stream.write(raw)
    finally:
        os.close(root_fd)
    return raw


def _reserve(root, stage, deadline):
    files, size = _inventory(root, deadline)
    count, maximum, _ = STAGE_LIMITS[stage]
    if len(files) + count + METADATA_RESERVE_FILES > MAX_FILES or size + maximum + METADATA_RESERVE_BYTES > MAX_BYTES:
        raise ValueError(f"Workflow cannot reserve the complete {stage} output")


def _check_child_hashes(files, child_manifests):
    for prefix, pin in child_manifests.items():
        expected = {f"{prefix}/{name}": sha for name, sha in pin["files"].items()}
        expected[f"{prefix}/manifest.json"] = pin["manifest_sha256"]
        actual = {name: sha for name, sha in files.items() if name.startswith(prefix + "/")}
        if actual != expected:
            raise ValueError("A completed workflow stage changed after its original seal")


def _seal(root, result, deadline, started, child_manifests=None):
    paths, _ = _inventory(root, deadline, immutable_only=True)
    files, size = {}, 0
    for name, maximum in sorted(paths.items()):
        if name in {"manifest.json", "result.json"}:
            continue
        _check(deadline)
        raw = read_package_file(root, name, max_bytes=max(1, maximum))
        files[name] = _sha(raw)
        size += len(raw)
    _check_child_hashes(files, child_manifests or {})
    result = {**result, "seconds": time.monotonic() - started, "seal_scope": "complete_immutable_outputs"}
    result_raw = _bytes(result)
    files["result.json"] = _sha(result_raw)
    manifest = {"schema_version": 1, "profile": PROFILE, "status": result["status"],
                "seal_scope": result["seal_scope"], "files": files}
    manifest_raw = _bytes(manifest)
    size += len(result_raw) + len(manifest_raw)
    if len(files) + 1 > MAX_FILES or size > MAX_BYTES:
        raise ValueError("Workflow terminal seal exceeds its aggregate bound")
    _check(deadline)
    _metadata(root, "result.json", result)
    _metadata(root, "manifest.json", manifest)
    _check(deadline)
    return {**result, "manifest_sha256": _sha(manifest_raw), "files": len(files) + 1,
            "bytes": size, "out": str(root)}


def _failure_receipt(root, result, error, started):
    result = {**result, "seconds": time.monotonic() - started, "seal_scope": "failure_receipt_only",
              "unsealed_stage_outputs_preserved": True,
              "sealing_error": {"type": type(error).__name__, "message": str(error)[:2000]}}
    raw = _metadata(root, "result.json", result)
    _metadata(root, "manifest.json", {"schema_version": 1, "profile": PROFILE, "status": "failed",
        "seal_scope": "failure_receipt_only", "unsealed_stage_outputs_preserved": True,
        "files": {"result.json": _sha(raw)}})


def _identity(directory, manifest_sha256, status, scene_raw, task_raw):
    _reject_failure_markers(directory)
    raw, hashes = read_package_manifest(directory)
    if _sha(raw) != manifest_sha256 or _json(raw).get("status") != status:
        raise ValueError("Workflow stage did not return its sealed terminal output")
    for name, expected in (("scene.json", scene_raw), ("exercise-task.json", task_raw)):
        if hashes.get(name) != _sha(expected):
            raise ValueError("Workflow stage changed the frozen scene or task")
        read_package_file(directory, name, max_bytes=64 * 1024 * 1024, expected_sha256=_sha(expected))
    return hashes


def _document(root, name, hashes):
    return _json(read_package_file(root, name, max_bytes=8 * 1024 * 1024, expected_sha256=hashes[name]))


def _admit_preserved_body_source(source, manifest_raw, names, scene, deadline):
    """A completed v3 candidate retains its raw body receipt; the receipt alone is not admission."""
    if "body-source.json" not in names:
        return None
    failure = "Workflow requires a completed untrained shared world with exact preserved person-only construction identity"
    manifest = _json(manifest_raw)
    required = {"reconstruction.json", "inventory.json", "inputs/body-source-manifest.json",
                "body-source.json", "human.json", "observations.json", "source-clock.json", "source.json"}
    if (scene["profile"] != HUMAN_ONLY_PROFILE or scene["source_binding"]["mode"] != "source_qualified"
            or manifest.get("profile") != HUMAN_ONLY_PROFILE or manifest.get("status") != "ready"
            or manifest.get("simulation_ready") is not False or not required <= names.keys()):
        raise ValueError(failure)
    documents = {}
    # The source verifier already checked the full observations and clock bytes.
    metadata = required - {"observations.json", "source-clock.json"}
    for name in sorted(metadata):
        _check(deadline)
        documents[name] = _document(source, name, names)
        if not isinstance(documents[name], dict):
            raise ValueError(failure)
    marker, parent = documents["body-source.json"], documents["inputs/body-source-manifest.json"]
    report, inventory = documents["reconstruction.json"], documents["inventory.json"]
    source_sha = scene["source_binding"]["source_sha256"]
    report_fixed = {"profile": HUMAN_ONLY_PROFILE, "status": "ready", "mode": "source_person_only",
                    "readiness": "candidate_constructed_not_lab_admitted", "simulation_ready": False,
                    "lab_status": "not_created", "physical_validation": "unverified",
                    "scene_sha256": canonical_hash(scene), "source_sha256": source_sha,
                    "human_file_sha256": names["human.json"], "inventory": scene["inventory"],
                    "source_manifest_sha256": names["inputs/body-source-manifest.json"]}
    construction = {"schema_version": 1, "profile": "exercise-body-source-v1",
                    "status": "construction_only", "simulation_ready": False}
    if (any(report.get(key) != value for key, value in report_fixed.items())
            or report.get("simulation_ready") is not False or inventory != scene["inventory"]
            or any(doc.get(key) != value for doc in (marker, parent) for key, value in construction.items())
            or any(type(doc.get("schema_version")) is not int for doc in (marker, parent))
            or any(doc.get("simulation_ready") is not False for doc in (marker, parent))
            or marker.get("source_sha256") != source_sha or documents["source.json"].get("sha256") != source_sha
            or canonical_hash(documents["human.json"]) != canonical_hash(scene["human_spec"])):
        raise ValueError(failure)
    artifact = marker.get("source_artifact")
    parent_files, artifacts = parent.get("files"), marker.get("artifacts")
    verification = report.get("source_verification")
    preserved = verification.get("preserved_files") if isinstance(verification, dict) else None
    if (not isinstance(artifact, str) or artifact not in residual.SOURCE_VIEWS - {"source.json"}
            or artifact not in names or names[artifact] != source_sha
            or documents["source.json"].get("artifact") != artifact
            or not isinstance(parent_files, dict) or not isinstance(artifacts, dict) or not isinstance(preserved, dict)
            or not {"human.json", "observations.json", "source-clock.json", "source.json"} <= artifacts.keys()
            or not {"body-source.json", artifact, *artifacts} <= preserved.keys()
            or any(name not in parent_files or name not in names or parent_files[name] != expected
                   or names[name] != expected for name, expected in preserved.items())
            or any(preserved.get(name) != expected for name, expected in artifacts.items())
            or any(name in parent_files for name in ("failure.json", "scene.json", "exercise-task.json", "policy.json", "training.json"))
            or names["observations.json"] not in scene["source_binding"]["evidence_documents"]):
        raise ValueError(failure)
    _check(deadline)
    return {"scope": "Preserved construction receipt inside an exact source-bound completed person-only candidate; not raw-body or physics admission",
            "body_source_sha256": names["body-source.json"], "body_manifest_sha256": names["inputs/body-source-manifest.json"],
            "reconstruction_sha256": names["reconstruction.json"], "inventory_sha256": names["inventory.json"],
            "source_sha256": source_sha, "human_file_sha256": names["human.json"],
            "identity_documents": {name: names[name] for name in sorted(metadata)}}


def prepare_exercise_learning_pipeline(run, out, *, train_settings=None, task_path=None,
                                       root_assistance=False, max_seconds=180, progress=None):
    """Train and compare the same task once; completed failed trials remain visible in the final lab."""
    settings = validate_train_settings(train_settings)
    if type(root_assistance) is not bool or task_path is not None and root_assistance:
        raise ValueError("Choose an explicit task or default-task root assistance, not both")
    if type(max_seconds) not in (int, float) or not math.isfinite(max_seconds) or not 6 <= max_seconds <= MAX_SECONDS:
        raise ValueError("Workflow requires a cooperative 6–180 second bound including finalization")
    original, target = _plain_path(run), _plain_path(out)
    supplied_task = _plain_path(task_path) if task_path is not None else None
    if target.exists() or any(target == path or target.is_relative_to(path) or path.is_relative_to(target)
                             for path in (original, *((supplied_task,) if supplied_task else ()))):
        raise ValueError("Workflow output must be new and outside every immutable input")
    started = time.monotonic()
    deadline, work_deadline = started + max_seconds, started + max_seconds - FINALIZATION_RESERVE_SECONDS
    source = original / "run" if (original / "run/manifest.json").is_file() else original
    outer = {}
    _reject_failure_markers(original)
    if source != original:
        outer = {name: read_package_file(original, name, max_bytes=MAX_METADATA_BYTES)
                 for name in ("lab-config.json", "status.json")}
        config, status = _json(outer["lab-config.json"]), _json(outer["status.json"])
        if (status.get("status") != "ready" or config.get("control_profile") != "shared_exercise"
                or config.get("learned_policy_available") is not False):
            raise ValueError("Workflow lab input must be ready, human-only and untrained")
    _reject_failure_markers(source)
    source_manifest, source_names = read_package_manifest(source)
    if outer and "exercise-lab.json" not in source_names:
        raise ValueError("Workflow lab input requires its sealed human-only lab contract")
    if (any(name in source_names for name in ("construction-seed.json", "policy.json",
            "training.json", "selection.json", "evaluation.json", "pre-evaluation-freeze.json"))
            or "body-source.json" in source_names and "scene.json" not in source_names):
        raise ValueError("Workflow requires a completed untrained shared world, not construction or previous learning")
    if any(name.startswith(("learning-run/", "policy-evaluation-run/")) for name in source_names):
        raise ValueError("Workflow cannot consume previous policy/evaluation inputs")
    for name in ("report.json", "reconstruction.json", "result.json", "status.json"):
        if name in source_names:
            document = _document(source, name, source_names)
            if not isinstance(document, dict) or document.get("status") in FAILED_STATUSES:
                raise ValueError("Workflow source retains an unsuccessful or malformed stage")
    if supplied_task is not None and "exercise-task.json" in source_names:
        read_package_file(supplied_task.parent, supplied_task.name, max_bytes=64 * 1024,
                          expected_sha256=source_names["exercise-task.json"])
    admitted, scene, task, scene_raw, task_raw, manifest, _ = residual._source(
        source, supplied_task, root_assistance, work_deadline)
    if admitted != source or manifest != source_manifest:
        raise ValueError("Workflow source manifest changed during admission")
    residual._terminal_manifest(manifest)
    body_admission = _admit_preserved_body_source(source, manifest, source_names, scene, work_deadline)
    if outer and any(config.get(name) != expected for name, expected in {
            "manifest_sha256": _sha(manifest), "scene_sha256": canonical_hash(scene),
            "task_sha256": canonical_hash(task)}.items()):
        raise ValueError("Workflow outer lab configuration differs from the admitted source/task")
    horizon = task["interval"]["horizon_steps"]
    training_steps = 2 * settings["generations"] * settings["population"] * horizon
    evaluation_steps = len(residual.CONTROLLERS) * settings["trials"] * horizon
    evaluation_limit = residual.MAX_FULLSPAN_EVALUATION_STEPS if scene["profile"] in FULLSPAN_PROFILES else residual.MAX_EVALUATION_STEPS
    if training_steps > residual.MAX_CONTROL_STEPS or evaluation_steps > evaluation_limit:
        raise ValueError("Workflow exceeds the full-interval action budget; change explicit search settings, never shorten the task")
    training_seeds = residual._seed_plan(settings["training_seed"], 2)
    evaluation_seeds = residual._seed_plan(settings["evaluation_seed"], settings["trials"], training_seeds)
    implementation, versions = _runtime()
    policy_profile, algorithm, actor_hashes, train_policy, select_policy, evaluate_policy = _learning_backend(settings, scene)
    actor_runtime = actor_hashes()
    source_documents = {name: source_names[name] for name in source_names
                        if name in {"scene.json", "exercise-task.json", "report.json", "reconstruction.json", "status.json", "result.json"}}
    if body_admission is not None:
        source_documents.update(body_admission["identity_documents"])
    plan = {"schema_version": 1, "profile": PROFILE, "created_at": local_time(), "input": str(original),
        "source_run": str(source), "source_manifest_sha256": _sha(manifest),
        "source_documents": source_documents, "lab_metadata": {name: _sha(raw) for name, raw in outer.items()},
        "supplied_task": str(supplied_task) if supplied_task else None,
        "scene_sha256": canonical_hash(scene), "task_sha256": canonical_hash(task),
        "scene_file_sha256": _sha(scene_raw), "task_file_sha256": _sha(task_raw), "gates_sha256": GATES_SHA256,
        "implementation_sha256": implementation, "runtime_versions": versions, "policy_runtime_sha256": actor_runtime,
        "policy_profile": policy_profile, "training_algorithm": algorithm,
        "training_settings": settings, "training_seeds": training_seeds, "evaluation_seeds": evaluation_seeds,
        "training_reset_options": list(residual.TRAIN_RESETS),
        "evaluation_reset_options": residual.human_jitter_options(settings["jitter_scale"]),
        "selection_rule": RULE, "evaluation_feedback_to_optimizer": False,
        "evaluation_scope": "Development comparison independent of this training; not globally fresh evidence or a final holdout",
        "horizon_steps": horizon, "maximum_training_steps": training_steps, "maximum_evaluation_steps": evaluation_steps,
        "maximum_control_steps": training_steps + evaluation_steps,
        "maximum_physics_substeps": (training_steps + evaluation_steps) * scene["substeps"],
        "maximum_seconds": max_seconds, "finalization_reserve_seconds": FINALIZATION_RESERVE_SECONDS,
        "maximum_files": MAX_FILES, "maximum_bytes": MAX_BYTES, "workers": 1, "external_api_usd": 0,
        "stage_output_reservations": {name: {"files": values[0], "bytes": values[1], "seconds": values[2]}
                                      for name, values in STAGE_LIMITS.items()},
        "deadline_scope": "Cooperative checks; in-flight native/helper calls require a caller process deadline",
        "lab_preparation_scope": "Model construction and reference forward geometry, not additional dynamics",
        "physical_validation": "unverified"}
    if body_admission is not None:
        plan["preserved_body_source_admission"] = body_admission
    if len(_bytes(plan)) > MAX_METADATA_BYTES:
        raise ValueError("Workflow frozen plan exceeds its metadata bound")
    _check(work_deadline)
    target.mkdir(parents=True, mode=0o700, exist_ok=False)
    stages, child_manifests, completed_lab = [], {}, None
    training_summary = selection_summary = evaluation_summary = None

    def guard(*, complete=False):
        _check(work_deadline)
        _reject_failure_markers(original)
        _reject_failure_markers(source)
        read_package_file(source, "manifest.json", max_bytes=8 * 1024 * 1024, expected_sha256=_sha(manifest))
        for name, expected in source_documents.items():
            read_package_file(source, name, max_bytes=64 * 1024 * 1024, expected_sha256=expected)
        for name, raw in outer.items():
            read_package_file(original, name, max_bytes=MAX_METADATA_BYTES, expected_sha256=_sha(raw))
        if supplied_task is not None:
            read_package_file(supplied_task.parent, supplied_task.name, max_bytes=64 * 1024, expected_sha256=_sha(task_raw))
        current_actor_runtime = actor_hashes()
        if _runtime() != (implementation, versions) or current_actor_runtime != actor_runtime:
            raise RuntimeError("Workflow frozen implementation or dependency runtime changed")
        read_package_file(target, "plan.json", max_bytes=MAX_METADATA_BYTES, expected_sha256=_sha(_bytes(plan)))
        read_package_file(target, "exercise-task.json", max_bytes=64 * 1024, expected_sha256=_sha(task_raw))
        for prefix, pin in child_manifests.items():
            _reject_failure_markers(target / prefix)
            if prefix == "lab/run":
                _reject_failure_markers(target / "lab")
            read_package_file(target / prefix, "manifest.json", max_bytes=8 * 1024 * 1024,
                              expected_sha256=pin["manifest_sha256"])
        if complete:
            residual._input_unchanged(source, manifest, work_deadline, task_path=supplied_task, task_raw=task_raw)
        _check(work_deadline)

    def stage(name, status, **detail):
        row = {"stage": name, "status": status, "elapsed_seconds": time.monotonic() - started, **detail}
        stages.append(row)
        _metadata(target, "stages.json", {"stages": stages})
        if progress is not None:
            progress(deepcopy(row))

    def remember(directory, manifest_sha256, status):
        hashes = _identity(directory, manifest_sha256, status, scene_raw, task_raw)
        child_manifests[directory.relative_to(target).as_posix()] = {
            "manifest_sha256": manifest_sha256, "files": dict(hashes)}
        return hashes

    def allowance(name, **detail):
        guard()
        _reserve(target, name, work_deadline)
        stage(name, "running", **detail)
        guard()
        remaining = work_deadline - time.monotonic()
        if remaining < 1:
            raise TimeoutError("Workflow has no bounded time remaining for the next stage")
        return min(STAGE_LIMITS[name][2], remaining)

    def result(status, **detail):
        return {"schema_version": 1, "profile": PROFILE, "status": status, "stages": stages,
            "lab_status": "ready" if completed_lab else "not_ready",
            "lab_directory": completed_lab["lab_directory"] if completed_lab else None,
            "launch_arguments": ["lab", "--resume", completed_lab["lab_directory"]] if completed_lab else None,
            "scene_sha256": canonical_hash(scene), "task_sha256": canonical_hash(task),
            "training": training_summary, "selection": selection_summary, "evaluation": evaluation_summary,
            "stage_manifests": {name: pin["manifest_sha256"] for name, pin in child_manifests.items()},
            "physical_validation": "unverified", **detail}

    try:
        write_package_files(target, {"scene.json": scene_raw, "exercise-task.json": task_raw,
            "input-manifest.json": manifest, "workflow-implementation.py": read_package_file(Path(__file__).parent,
                Path(__file__).name, max_bytes=8 * 1024 * 1024, expected_sha256=implementation[Path(__file__).name]),
            **{"input-" + name: raw for name, raw in outer.items()}})
        _metadata(target, "plan.json", plan)
        maximum = allowance("training", maximum_control_steps=training_steps)
        trained = train_policy(source, target / "training",
            task_path=target / "exercise-task.json", seed=settings["training_seed"],
            generations=settings["generations"], population=settings["population"], max_seconds=maximum)
        hashes = remember(target / "training", trained["manifest_sha256"], "trained")
        trained_settings = _document(target / "training", "settings.json", hashes)
        if trained_settings["training_seeds"] != training_seeds or trained_settings["maximum_control_steps"] != training_steps:
            raise ValueError("Trainer changed the frozen workflow seed or work plan")
        training_summary = {name: trained[name] for name in ("policy_sha256", "control_steps", "evaluated_episodes", "completed_generations")}
        training_summary.update(directory=str(target / "training"), manifest_sha256=trained["manifest_sha256"])
        stage("training", "trained", **training_summary)

        maximum = allowance("training_selection")
        selected = select_policy(target / "training", target / "selected", max_seconds=maximum)
        remember(target / "selected", selected["manifest_sha256"], "trained")
        selection_summary = {name: selected[name] for name in ("manifest_sha256", "policy_sha256", "selection_status",
                            "feasible_candidates", "unique_candidates", "selected_candidate")}
        selection_summary.update(directory=str(target / "selected"), evaluation_used=False)
        stage("training_selection", "selected", **selection_summary)

        maximum = allowance("independent_evaluation", maximum_control_steps=evaluation_steps)
        evaluated = evaluate_policy(target / "selected", target / "evaluation",
            seed=settings["evaluation_seed"], trials=settings["trials"], jitter_scale=settings["jitter_scale"],
            max_seconds=min(maximum, 60 if scene["profile"] in FULLSPAN_PROFILES else 30))
        remember(target / "evaluation", evaluated["manifest_sha256"], "evaluated")
        if (evaluated["evaluation_seeds"] != evaluation_seeds or evaluated["maximum_control_steps"] != evaluation_steps
                or evaluated["policy_sha256"] != selected["policy_sha256"]):
            raise ValueError("Evaluation changed the frozen policy, reset seed or action plan")
        evaluation_summary = {name: evaluated[name] for name in ("manifest_sha256", "policy_sha256", "control_steps",
                              "comparison_complete", "independent_changed_states")}
        evaluation_summary.update(directory=str(target / "evaluation"), evaluation_seeds=evaluation_seeds,
            trials_per_controller=settings["trials"], controllers={name: {key: row[key] for key in
                ("successes", "completed_intervals", "reset_failures", "mean_score")}
                for name, row in evaluated["controllers"].items()}, physical_validation="unverified")
        stage("independent_evaluation", "evaluated", **evaluation_summary)

        maximum = allowance("interactive_lab")
        prepared_lab = prepare_shared_exercise_lab(target / "selected", target / "lab",
            task_path=target / "exercise-task.json", policy_run=target / "selected",
            evaluation_run=target / "evaluation", max_seconds=maximum)
        remember(target / "lab/run", prepared_lab["manifest_sha256"], "ready")
        completed_lab = prepared_lab
        stage("interactive_lab", "ready", manifest_sha256=completed_lab["manifest_sha256"], lab_status="ready",
              lab_directory=completed_lab["lab_directory"], scene_sha256=completed_lab["scene_sha256"],
              task_sha256=completed_lab["task_sha256"])
        guard(complete=True)
        return _seal(target, result("ready"), deadline, started, child_manifests)
    except Exception as error:
        failure = result("failed", error={"type": type(error).__name__, "message": str(error)[:2000]})
        try:
            _metadata(target, "failure.json", failure["error"])
            try:
                _seal(target, failure, deadline, started, child_manifests)
            except Exception as sealing_error:
                _failure_receipt(target, failure, sealing_error, started)
        except Exception as receipt_error:
            error.add_note(f"Workflow failure receipt failed: {type(receipt_error).__name__}: {str(receipt_error)[:500]}")
        raise
