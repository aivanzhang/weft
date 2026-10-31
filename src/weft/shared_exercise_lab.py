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
from weft.shared_exercise_reset import describe_reset_protocol, resolve_reset_options
from weft.shared_exercise_task import SharedExerciseTaskEnv, build_shared_exercise_task, validate_shared_exercise_task
from weft.shared_world import CAMERA_PROFILE, FEET_PROFILE, HUMAN_ONLY_PROFILE, STATIC_SUPPORT_PROFILE, SharedWorldEnv, validate_shared_world
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
    "shared_exercise_task.py", "shared_exercise_reset.py", "shared_world.py", "shared_world_observation.py", "shared_world_evidence.py",
    "source_equivalence.py", "world_evidence.py", "observations.py", "human.py", "mobile_human.py",
    "whole_body.py", "spatial_rigid.py", "body_frame.py", "spatial_pose.py", "root_motion.py", "errors.py", "io.py",
)
FEET_RUNTIME_FILES = (
    "shared_world_feet.py", "foot_body.py", "foot_reference.py", "whole_body_feasibility.py",
    "shared_exercise_feet_projection.py", "shared_exercise_feet_task.py", "shared_exercise_feet_state.py",
    "shared_exercise_feet_recording.py", "shared_exercise_recording.py", "recording.py", "shared_support_task.py",
)
CAMERA_RUNTIME_FILES = (
    "camera_shared_task.py", "camera_shared_state.py", "camera_shared_recording.py", "camera_shared_contract.py",
    "camera_body.py", "camera_ground.py", "camera_pose.py", "multiview_root.py", "multiview_body.py", "multiview_pose.py",
    "multiview_segments.py", "shared_exercise_recording.py", "recording.py", "camera_shared_receipt.py",
    "shared_exercise_feet_state.py", "shared_exercise_feet_projection.py", "shared_support_task.py",
)


def _base_profile(scene):
    return scene["base_profile"] if scene["profile"] == FEET_PROFILE else scene["profile"]


def _validate_task(task, scene):
    if scene["profile"] == CAMERA_PROFILE:
        from weft.camera_shared_task import validate_camera_shared_task
        return validate_camera_shared_task(task, scene)
    if scene["profile"] == FEET_PROFILE:
        from weft.shared_exercise_feet_task import validate_shared_exercise_feet_task
        return validate_shared_exercise_feet_task(task, scene)
    return validate_shared_exercise_task(task, scene)


def runtime_files(scene):
    """Extra evaluator dependencies are pinned only for the profiles that execute them."""
    if scene["profile"] == CAMERA_PROFILE:
        return (*RUNTIME_FILES, *CAMERA_RUNTIME_FILES)
    if scene["profile"] == FEET_PROFILE:
        inventory = ("person_inventory.py",) if _base_profile(scene) == HUMAN_ONLY_PROFILE else ()
        return (*RUNTIME_FILES, *FEET_RUNTIME_FILES, *inventory)
    if scene["profile"] == HUMAN_ONLY_PROFILE:
        return (*RUNTIME_FILES, "person_inventory.py")
    if scene["profile"] == STATIC_SUPPORT_PROFILE:
        return (*RUNTIME_FILES, "shared_support_task.py")
    return RUNTIME_FILES


REJECTED_STATUSES = frozenset({"failed", "error", "unsupported", "rejected", "cancelled"})
SOURCE_VIEWS = frozenset({
    "source.json", "source.mp4", "source.webm", "source.mov", "source.m4v", "source.avi", "source.mkv",
    "pose-quality.json", "camera-reference.json", "two-view-root.json", "evidence.json", "tracks.csv",
    "reconstruction.json", "reference-feasibility.json", "coverage-summary.json", "reconstruction-coverage.json",
    "geometry-proposal.json", "geometry-study.json",
    "geometry-hypothesis.json", "previous-geometry.json", "initial-geometry-check.json",
    "fullspan.json",
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


def _verified_inputs(directory, *, deadline, maximum_files=MAX_INPUT_FILES, maximum_bytes=MAX_INPUT_BYTES, preflight=None):
    manifest_raw, hashes = read_package_manifest(directory)
    manifest = _json(manifest_raw)
    if manifest.get("status") in REJECTED_STATUSES:
        raise ValueError("Cannot prepare or resume an unsuccessful exercise lab input")
    if len(hashes) > maximum_files:
        raise ValueError("Exercise lab input exceeds its file-count bound")
    prefetched = preflight(directory, hashes) if preflight is not None else {}
    files, total = {}, len(manifest_raw)
    for name, expected in hashes.items():
        _deadline(deadline)
        required_log = re.fullmatch(r"(?:(?:source-run|learning-run|policy-evaluation-run|policy-evidence-run|imported-training)/)*source-mappings/[0-9a-f]{64}/[^/]+\.log", name)
        if (any(part.startswith(".") or part == "__pycache__" for part in Path(name).parts)
                or name.endswith(".pyc") or name.endswith(".log") and not required_log):
            raise ValueError("Exercise lab inputs must exclude hidden files, caches and logs")
        raw = (prefetched[name] if name in prefetched else
               read_package_file(directory, name, max_bytes=min(MAX_FILE_BYTES, max(1, maximum_bytes - total)),
                                 expected_sha256=expected))
        total += len(raw)
        if total > maximum_bytes:
            raise ValueError("Exercise lab input exceeds its byte bound")
        files[name] = raw
    _deadline(deadline)
    return manifest_raw, files


def _feet_metadata_preflight(*, task_path=None, policy_run=None, evaluation_run=None, resume=False):
    def read(directory, hashes):
        if "scene.json" not in hashes:
            return {}
        scene_raw = read_package_file(directory, "scene.json", max_bytes=MAX_FILE_BYTES,
                                      expected_sha256=hashes["scene.json"])
        scene = _json(scene_raw)
        fetched = {"scene.json": scene_raw}
        if isinstance(scene, dict) and scene.get("profile") == CAMERA_PROFILE:
            for name, maximum in (("exercise-task.json", 64 << 10), ("exercise-lab.json", 128 << 10)):
                if name in hashes:
                    fetched[name] = read_package_file(directory, name, max_bytes=maximum, expected_sha256=hashes[name])
            if "exercise-lab.json" in fetched:
                contract = _json(fetched["exercise-lab.json"])
                if "camera_policy_admission_sha256" in contract:
                    from weft.camera_shared_policy_lab import preflight_camera_attachment
                    preflight_camera_attachment(contract, hashes)
                elif (contract.get("learned_policy_available") is not False
                        or any(key in contract for key in ("policy_sha256", "exercise_policy", "policy_evaluation_sha256"))):
                    raise ValueError("Camera-based tasks cannot expose unqualified learning")
            else:
                contract = {}
            if "camera_policy_admission_sha256" not in contract and any(
                    name in {"policy.json", "policy-evaluation.json", "camera-policy-admission.json"}
                    or name.startswith(("learning-run/", "policy-evaluation-run/", "policy-evidence-run/")) for name in hashes):
                raise ValueError("Camera-based tasks cannot import unqualified learning")
            if "camera-coupling.json" not in hashes:
                raise ValueError("Camera lab requires its exact displayed coupling")
            raw = read_package_file(directory, "camera-coupling.json", max_bytes=64 << 10,
                                    expected_sha256=hashes["camera-coupling.json"])
            if _json(raw) != scene.get("camera_coupling"):
                raise ValueError("Displayed camera coupling differs from the simulated source binding")
            fetched["camera-coupling.json"] = raw
            return fetched
        if not isinstance(scene, dict) or scene.get("profile") != FEET_PROFILE:
            return fetched
        if policy_run is not None or evaluation_run is not None:
            raise ValueError("Coupled-feet learned policies and evaluation attachments are not yet qualified")
        if "exercise-task.json" in hashes:
            fetched["exercise-task.json"] = read_package_file(directory, "exercise-task.json", max_bytes=128 << 10,
                                                               expected_sha256=hashes["exercise-task.json"])
        elif task_path is None:
            raise ValueError("Coupled feet require an explicit source-bound task; base requirements cannot be replaced by defaults")
        if task_path is not None:
            path = Path(task_path).absolute()
            read_package_file(path.parent, path.name, max_bytes=128 << 10)
        if "exercise-lab.json" in hashes:
            fetched["exercise-lab.json"] = read_package_file(directory, "exercise-lab.json", max_bytes=128 << 10,
                                                              expected_sha256=hashes["exercise-lab.json"])
            contract = _json(fetched["exercise-lab.json"])
            if (contract.get("learned_policy_available") is not False
                    or any(key in contract for key in ("policy_sha256", "exercise_policy", "policy_evaluation_sha256"))):
                raise ValueError("Coupled-feet labs cannot expose unqualified policies or policy evaluation")
        if resume and any(name in {"policy.json", "policy-evaluation.json"}
                          or name.startswith(("learning-run/", "policy-evaluation-run/")) for name in hashes):
            raise ValueError("Coupled-feet labs cannot expose unqualified policies or policy evaluation")
        if "foot-reference.json" in hashes:
            raw = read_package_file(directory, "foot-reference.json", max_bytes=8 << 20,
                                    expected_sha256=hashes["foot-reference.json"])
            extension = scene.get("foot_extension", {})
            text = extension.get("reference_json")
            if (not isinstance(text, str) or raw != text.encode("utf-8")
                    or _sha(raw) != extension.get("reference_sha256")):
                raise ValueError("Displayed foot reference differs from the exact simulated reference")
            fetched["foot-reference.json"] = raw
        elif resume:
            raise ValueError("Coupled-feet lab is missing its exact displayed foot reference")
        return fetched

    return read


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
        count = len(scene["human_spec"]["reference"])
        if count > 900:
            raise ValueError("Reference inspection exceeds its 900-pose bound; no silent clipping")
        for index in range(count):
            _deadline(deadline)
            frames.append(env.reference_snapshot(index))
        return {"schema_version": 1, "profile": "shared-world-kinematic-reference-v1", "frames": frames,
                "reference_poses_total": count, "reference_poses_included": len(frames),
                "scope": ("Kinematic body projection with fixed original-start ankle hold; no video ankle tracking, physics actions or contact inference."
                          if scene["profile"] == FEET_PROFILE else
                          "Kinematic camera-based body at original source times; no recovered feet, equipment or ground contact."
                          if scene["profile"] == CAMERA_PROFILE else
                          "Kinematic body projection with ground only; no physics actions or equipment inference."
                          if scene["profile"] == HUMAN_ONLY_PROFILE else
                          "Kinematic body projection with authored static supports; no physics actions or contact inference."
                          if scene["profile"] == STATIC_SUPPORT_PROFILE else
                          "Kinematic body projection with initial passive equipment; no physics actions or grasp inference.")}
    finally:
        env.close()


def _validate_policy_package(manifest_raw, files, scene, task, *, include_baseline=False, deadline=None):
    # Lazy imports keep the learner's existing lab-helper dependency acyclic.
    from weft.shared_exercise_policies import selection_kind, validate_policy

    manifest = _json(manifest_raw)
    required = {"policy.json", "training.json", "settings.json", "scene.json", "exercise-task.json", "input-manifest.json"}
    if (not isinstance(manifest, dict) or manifest.get("status") != "trained" or not required <= set(files)
            or not isinstance(manifest.get("files"), dict) or set(manifest["files"]) != set(files)
            or any(_sha(files[name]) != expected for name, expected in manifest["files"].items())):
        raise ValueError("Exercise policy requires the complete sealed trained package")
    if (validate_shared_world(_json(files["scene.json"])) != scene
            or validate_shared_exercise_task(_json(files["exercise-task.json"]), scene) != task):
        raise ValueError("Exercise learning scene/task must exactly match the selected lab")
    if len(files["policy.json"]) > 64 * 1024:
        raise ValueError("Exercise policy exceeds its document bound")
    policy_document = _json(files["policy.json"])
    if scene["profile"] == STATIC_SUPPORT_PROFILE and (not isinstance(policy_document, dict)
            or policy_document.get("profile") != "shared-support-policy-v1"):
        raise ValueError("Static-support labs require a newly pinned support policy")
    if scene["profile"] == HUMAN_ONLY_PROFILE and (not isinstance(policy_document, dict)
            or policy_document.get("profile") not in {"shared-exercise-policy-v2", "shared-exercise-policy-v4"}):
        raise ValueError("Person-only labs require a newly pinned v2 or v4 human residual policy")
    policy = validate_policy(policy_document, scene, task)
    training, settings = _json(files["training.json"]), _json(files["settings.json"])
    if (not isinstance(training, dict) or training.get("status") != "trained"
            or training.get("policy_sha256") != _sha(files["policy.json"])):
        raise ValueError("Exercise training must pin the exact selected policy bytes")
    kind, authored, coast = selection_kind(policy)
    improvement = training.get("improvement_over_authored_training")
    if (training.get("selected_feedback_kind") != kind
            or training.get("selected_authored_baseline") is not authored
            or training.get("selected_coast_baseline") is not coast
            or type(improvement) not in (int, float) or not math.isfinite(improvement)
            or training.get("training_score_improved_over_authored") is not (improvement > 0)):
        raise ValueError("Exercise training selection metadata differs from its exact policy")
    if (not isinstance(settings, dict) or settings.get("source_manifest_sha256") != _sha(files["input-manifest.json"])
            or any(settings.get(name) != policy[name] for name in ("scene_sha256", "task_sha256", "gates_sha256", "runtime_sha256"))
            or any("implementation/" + name not in files or _sha(files["implementation/" + name]) != expected
                   for name, expected in policy["runtime_sha256"].items())):
        raise ValueError("Exercise training source/runtime proof differs from the selected policy")
    if policy["profile"] == "shared-exercise-policy-v4":
        from weft.shared_exercise_feedback_learning import verify_shared_exercise_feedback_package

        # New feedback packages require the saved search/action proof, not just matching labels.
        proof_deadline = time.monotonic() + 30 if deadline is None else deadline
        _deadline(proof_deadline)
        verify_shared_exercise_feedback_package(manifest_raw, files, scene, task, deadline=proof_deadline)
        _deadline(proof_deadline)
    identity = {"policy_sha256": _sha(files["policy.json"]), "learning_manifest_sha256": _sha(manifest_raw),
                "learning_files_verified": len(files), "policy_runtime_sha256": deepcopy(policy["runtime_sha256"]),
                "policy_evaluation_scope": "Training completed; independent evaluation success is not attested by this package"}
    baseline = None
    if policy["profile"] == "shared-exercise-policy-v3":
        from weft.shared_exercise_feasibility_learning import verify_imported_policy_baseline

        baseline = verify_imported_policy_baseline(files, scene, task)
        identity.update(import_proof_sha256=baseline["import_proof_sha256"],
                        previous_reward_policy_sha256=baseline["policy_sha256"])
    return (policy, identity, baseline) if include_baseline else (policy, identity)


def _validate_policy_evaluation(manifest_raw, files, learning_manifest, learning_files, policy):
    from weft.shared_exercise_learning import CONTROLLERS
    from weft.shared_exercise_task import prepare_shared_exercise_snapshot_validator

    manifest = _json(manifest_raw)
    required = {"scene.json", "exercise-task.json", "policy.json", "evaluation.json", "settings.json", "input-manifest.json"}
    if (not isinstance(manifest, dict) or manifest.get("status") != "evaluated" or not required <= set(files)
            or not isinstance(manifest.get("files"), dict) or set(manifest["files"]) != set(files)
            or any(_sha(files[name]) != expected for name, expected in manifest["files"].items())):
        raise ValueError("Exercise policy evaluation requires a complete sealed evaluated package")
    if (files["input-manifest.json"] != learning_manifest
            or any(files[name] != learning_files[name] for name in ("scene.json", "exercise-task.json", "policy.json"))
            or any("implementation/" + name not in files or _sha(files["implementation/" + name]) != expected
                   for name, expected in policy["runtime_sha256"].items())):
        raise ValueError("Exercise policy evaluation source, policy or runtime differs from training")
    evaluation, settings = _json(files["evaluation.json"]), _json(files["settings.json"])
    training_settings = _json(learning_files["settings.json"])
    settings_required = {"policy_sha256", "scene_sha256", "task_sha256", "runtime_sha256", "evaluation_seeds",
                         "training_seeds", "reset_options", "evaluation_feedback_to_optimizer", "horizon_steps"}
    if (not isinstance(evaluation, dict) or evaluation.get("status") != "evaluated"
            or evaluation.get("comparison_complete") is not True or evaluation.get("policy_unchanged") is not True
            or evaluation.get("evaluation_feedback_to_optimizer") is not False
            or evaluation.get("policy_sha256") != _sha(files["policy.json"])
            or any(evaluation.get(name) != policy[name] for name in ("scene_sha256", "task_sha256", "runtime_sha256"))
            or not isinstance(settings, dict) or not settings_required <= set(settings)
            or any(evaluation.get(key) != value for key, value in settings.items())
            or evaluation.get("training_seeds") != training_settings.get("training_seeds")):
        raise ValueError("Exercise policy evaluation is incomplete or has different pinned identities")
    seeds = evaluation.get("evaluation_seeds")
    resets = evaluation.get("reset_options")
    if (not isinstance(seeds, list) or not 1 <= len(seeds) <= 3
            or any(type(seed) is not int or not 0 <= seed < 2**31 for seed in seeds) or len(set(seeds)) != len(seeds)
            or not isinstance(evaluation.get("training_seeds"), list)
            or any(type(seed) is not int for seed in evaluation["training_seeds"])
            or set(seeds) & set(evaluation["training_seeds"])
            or not isinstance(resets, dict) or resets != resolve_reset_options(resets) or resets["jitter"] is not True
            or type(resets["jitter_scale"]) not in (int, float) or not 0 < resets["jitter_scale"] <= .25):
        raise ValueError("Exercise policy evaluation requires independent bounded changed-start trials")
    initial = evaluation.get("initial_states")
    controllers = evaluation.get("controllers")
    if (not isinstance(initial, dict) or set(initial) != {str(seed) for seed in seeds}
            or any(not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value) for value in initial.values())
            or len(set(initial.values())) != len(seeds) or evaluation.get("independent_changed_states") != len(seeds)
            or not isinstance(controllers, dict) or set(controllers) != set(CONTROLLERS)):
        raise ValueError("Exercise policy evaluation does not contain the complete matched comparison")
    scene = validate_shared_world(_json(files["scene.json"]))
    task = validate_shared_exercise_task(_json(files["exercise-task.json"]), scene)
    validate_snapshot = prepare_shared_exercise_snapshot_validator(scene, task)
    if evaluation.get("horizon_steps") != task["interval"]["horizon_steps"]:
        raise ValueError("Exercise policy evaluation cannot shorten the pinned task")
    training = _json(learning_files["training.json"])
    training_states = {trial.get("initial_state_sha256") for generation in training.get("history", [])
                       for candidate in generation.get("candidates", []) for trial in candidate.get("trials", [])}
    if set(initial.values()) & training_states:
        raise ValueError("Exercise policy evaluation repeats a training state")
    successes, total_steps = {}, 0
    for controller in CONTROLLERS:
        summary = controllers[controller]
        rows = summary.get("trials") if isinstance(summary, dict) else None
        if (not isinstance(rows, list) or len(rows) != len(seeds)
                or any(not isinstance(row, dict) or row.get("seed") != seed or row.get("reset_options") != resets
                       or row.get("initial_state_sha256") != initial[str(seed)] or row.get("status") not in ("completed", "failed")
                       or not isinstance(row.get("metrics"), dict) or type(row["metrics"].get("success")) is not bool
                       or type(row["metrics"].get("completed_interval")) is not bool for row, seed in zip(rows, seeds))):
            raise ValueError("Exercise policy evaluation trial coverage differs from its declared seeds")
        count = sum(row["metrics"]["success"] for row in rows)
        if (type(summary.get("successes")) is not int or summary["successes"] != count
                or summary.get("completed_intervals") != sum(row["metrics"]["completed_interval"] for row in rows)
                or summary.get("reset_failures") != 0):
            raise ValueError("Exercise policy evaluation aggregates differ from its trials")
        for row in rows:
            first, final = row.get("initial_snapshot"), row.get("final_snapshot")
            validate_snapshot(first)
            validate_snapshot(final)
            if (row["metrics"] != final["task_metrics"] or row.get("initial_metrics") != first["task_metrics"]
                    or row["initial_state_sha256"] != canonical_hash(first["engine_state"])
                    or first["engine_state_status"] != "current" or first["engine_state"]["steps"] != 0
                    or first["task_metrics"]["observed_substeps"] != 0 or first["resolved_reset_options"] != resets
                    or final["resolved_reset_options"] != resets
                    or row.get("terminated") is not final["terminated"] or row.get("truncated") is not final["truncated"]
                    or row["status"] != ("completed" if row["metrics"]["completed_interval"] else "failed")):
                raise ValueError("Exercise policy evaluation metrics differ from their pinned boundary snapshots")
            actions = row.get("actions")
            if (not isinstance(actions, list) or not 1 <= len(actions) <= task["interval"]["horizon_steps"]
                    or type(row.get("control_steps")) is not int or row["control_steps"] != len(actions)
                    or not isinstance(actions[-1], dict) or actions[-1].get("snapshot_sha256") != canonical_hash(final)
                    or actions[-1].get("action") != final["last_action"]):
                raise ValueError("Exercise policy evaluation action trace differs from its final boundary")
            total_steps += len(actions)
        successes[controller] = {"successes": count, "episodes": len(rows)}
    if type(evaluation.get("control_steps")) is not int or evaluation["control_steps"] != total_steps:
        raise ValueError("Exercise policy evaluation control-step count differs from its trials")
    comparison_identity = {}
    if policy["profile"] == "shared-exercise-policy-v3":
        from weft.shared_exercise_feasibility_evaluation import validate_previous_reward_comparison

        comparison_identity = validate_previous_reward_comparison(manifest_raw, files, learning_files, policy,
            evaluation, scene, task, standard_control_steps=total_steps)
    reset_protocol = describe_reset_protocol(resets)
    if scene["profile"] == STATIC_SUPPORT_PROFILE:
        from weft.shared_exercise_task import _support_reset_protocol
        reset_protocol = _support_reset_protocol(resets)
    return {"policy_evaluation_manifest_sha256": _sha(manifest_raw), "policy_evaluation_sha256": _sha(files["evaluation.json"]),
            "policy_evaluation_files_verified": len(files), **comparison_identity}, successes, reset_protocol


def _policy_status(policy, training_raw, *, evaluation_successes=None, evaluation_reset_protocol=None):
    training = _json(training_raw)
    result = {"profile": policy["profile"], "policy_sha256": training["policy_sha256"],
            "selected_feedback_kind": training["selected_feedback_kind"], "privileged_simulator_features": True,
            "training_score": policy["training_score"],
            "training_score_improved_over_authored": training["training_score_improved_over_authored"],
            "evaluation_status": "evaluated" if evaluation_successes is not None else "not_attached",
            "evaluation_successes": deepcopy(evaluation_successes),
            "evaluation_reset_protocol": deepcopy(evaluation_reset_protocol),
            "scope": "Training selection and separate changed-start simulator comparison; neither verifies new-video generalization, grasp or physical accuracy."}
    if policy["profile"] in {"shared-exercise-policy-v3", "shared-exercise-policy-v4"}:
        result.update(selection_objective=policy["selection_objective"],
                      training_feasible=policy["optimization_rank"]["feasible"])
    return result


def prepare_shared_exercise_lab(run, out, *, root_assistance=False, task_path=None, policy_run=None, evaluation_run=None, max_seconds=60):
    """Prepare a new sealed lab without changing the source scene, task gates or legacy labs."""
    if type(root_assistance) is not bool or task_path is not None and root_assistance:
        raise ValueError("Choose an explicit task or default-task root assistance, not both")
    if type(max_seconds) not in (int, float) or not math.isfinite(max_seconds) or not 1 <= max_seconds <= 180:
        raise ValueError("Exercise lab preparation requires a 1–180 second bound")
    if evaluation_run is not None and policy_run is None:
        raise ValueError("Exercise policy evaluation requires its selected learning package")
    source, out = Path(run).resolve(), Path(out).resolve()
    if out.exists():
        raise FileExistsError("Exercise lab requires a new output directory")
    if out == source or out.is_relative_to(source) or source.is_relative_to(out):
        raise ValueError("Exercise lab output must be outside the immutable input")
    learning_source = Path(policy_run).resolve() if policy_run is not None else None
    if learning_source is not None and (out == learning_source or out.is_relative_to(learning_source) or learning_source.is_relative_to(out)):
        raise ValueError("Exercise lab output must be outside the immutable learning package")
    evaluation_source = Path(evaluation_run).resolve() if evaluation_run is not None else None
    if evaluation_source is not None and (out == evaluation_source or out.is_relative_to(evaluation_source) or evaluation_source.is_relative_to(out)):
        raise ValueError("Exercise lab output must be outside the immutable evaluation package")
    source = source / "run" if (source / "run" / "manifest.json").is_file() else source
    started = time.monotonic()
    deadline = started + max_seconds
    input_manifest, inputs = _verified_inputs(source, deadline=deadline,
        preflight=_feet_metadata_preflight(task_path=task_path, policy_run=policy_run, evaluation_run=evaluation_run))
    if "construction-seed.json" in inputs:
        raise ValueError("Construction seed is not a simulation; finish equipment construction before preparing a lab")
    if "scene.json" not in inputs:
        raise ValueError("Exercise lab input requires a sealed shared-world scene")
    for name in ("status.json", "report.json", "evaluation.json", "reconstruction.json"):
        if name in inputs:
            document = _json(inputs[name])
            if isinstance(document, dict) and document.get("status") in REJECTED_STATUSES:
                raise ValueError(f"Exercise lab source artifact {name} has unsuccessful status")
    scene = validate_shared_world(_json(inputs["scene.json"]))
    feet = scene["profile"] == FEET_PROFILE
    base_profile = _base_profile(scene)
    if feet and (policy_run is not None or evaluation_run is not None):
        raise ValueError("Coupled-feet learned policies and evaluation attachments are not yet qualified")
    proof = collect_shared_source_files(scene, source, expected_manifest_sha256=_sha(input_manifest))
    task_raw = None
    if task_path is None and "exercise-task.json" in inputs:
        # Geometry alternatives retain custom contact/control requirements, not just default gates.
        task_raw = inputs["exercise-task.json"]
        if feet and len(task_raw) > 128 * 1024:
            raise ValueError("Coupled-feet task exceeds its 128 KiB document bound")
        task = _validate_task(_json(task_raw), scene)
        if root_assistance and task["control"]["root_assistance"] != "explicit_3_axis":
            raise ValueError("Root assistance conflicts with the sealed input task; provide an explicit replacement task")
    elif task_path is None:
        if feet:
            raise ValueError("Coupled feet require an explicit source-bound task; base requirements cannot be replaced by defaults")
        if scene["profile"] == CAMERA_PROFILE:
            from weft.camera_shared_task import build_camera_shared_task
            task = build_camera_shared_task(scene, root_assistance=root_assistance)
        else:
            task = build_shared_exercise_task(scene, root_assistance=root_assistance)
    else:
        path = Path(task_path).absolute()
        task_raw = read_package_file(path.parent, path.name, max_bytes=(128 if feet else 64) * 1024)
        task = _validate_task(_json(task_raw), scene)
    policy, policy_identity, learning_files, learning_manifest = None, {}, {}, None
    camera_attachment = None
    if learning_source is not None and scene["profile"] == CAMERA_PROFILE:
        from weft.camera_shared_policy_lab import prepare_camera_attachment
        camera_attachment = prepare_camera_attachment(learning_source, evaluation_source, scene, task, deadline=deadline)
        policy, policy_identity = camera_attachment["policy"], camera_attachment["identity"]
    elif learning_source is not None:
        learning_manifest, learning_files = _verified_inputs(learning_source, deadline=deadline)
        policy, policy_identity = _validate_policy_package(learning_manifest, learning_files, scene, task, deadline=deadline)
        collect_shared_source_files(scene, learning_source, expected_manifest_sha256=_sha(learning_manifest))
    evaluation_files, evaluation_manifest, evaluation_successes, evaluation_reset_protocol = {}, None, None, None
    if evaluation_source is not None and camera_attachment is None:
        evaluation_manifest, evaluation_files = _verified_inputs(evaluation_source, deadline=deadline)
        evaluation_identity, evaluation_successes, evaluation_reset_protocol = _validate_policy_evaluation(
            evaluation_manifest, evaluation_files, learning_manifest, learning_files, policy)
        policy_identity.update(evaluation_identity)
        collect_shared_source_files(scene, evaluation_source, expected_manifest_sha256=_sha(evaluation_manifest))
    generated = {"manifest.json", "source-run", "lab-implementation", "scene.json", "exercise-task.json",
                 "exercise-lab.json", "source-verification.json", "human-reference.json", "input-task.json"}
    if feet:
        generated.update({"foot-reference.json", "feet-migration.json"})
    if policy is not None:
        generated.update({"policy-evidence-run", "policy.json", "camera-policy-admission.json", "policy-evaluation.json"}
                         if camera_attachment else {"learning-run", "policy.json"})
    if evaluation_manifest is not None:
        generated.update({"policy-evaluation-run", "policy-evaluation.json"})
    proof.validate_layout(reserved_paths=generated)
    implementation = _implementation_snapshot()
    contract = {"schema_version": 1, "profile": "shared-world", "control_profile": CONTROL_PROFILE,
                "scene_sha256": canonical_hash(scene), "task_sha256": canonical_hash(task),
                "reset_options": dict(RESET_OPTIONS), "learned_policy_available": policy is not None,
                "source_manifest_sha256": _sha(input_manifest), "source_files_verified": len(inputs),
                "implementation_sha256": {name: _sha(raw) for name, raw in implementation.items()},
                "runtime_sha256": {name: _sha(implementation[name]) for name in runtime_files(scene)},
                "runtime_versions": _runtime_versions(),
                "scope": ("Coupled feet with original body scoring and unscored ankle diagnostics; no video ankle tracking, foot support credit, ground-load qualification or physical validation."
                          if feet else
                          "Camera-based body with original clocks and unchanged body scoring; no recovered feet, equipment, contact, learning or physical validation."
                          if scene["profile"] == CAMERA_PROFILE else
                          "Authored human/static-support load task; no recovered contact, grasp or physical validation."
                          if scene["profile"] == STATIC_SUPPORT_PROFILE else
                          "Independent authored human-only simulator task; no recovered grasp or physical validation.")}
    contract.update(policy_identity)
    if camera_attachment is not None:
        contract["runtime_sha256"] = deepcopy(camera_attachment["admission"]["execution_runtime_sha256"])
        contract["exercise_policy"] = camera_attachment["summary"]
        contract["scope"] = camera_attachment["summary"]["scope"]
    elif policy is not None:
        contract["exercise_policy"] = _policy_status(policy, learning_files["training.json"],
            evaluation_successes=evaluation_successes, evaluation_reset_protocol=evaluation_reset_protocol)
    views = {name: raw for name, raw in inputs.items()
             if name in SOURCE_VIEWS or name.startswith("geometry-inspection/")
             or base_profile == HUMAN_ONLY_PROFILE and name in {"inventory.json", "source-clock.json"}
             or base_profile == STATIC_SUPPORT_PROFILE and name in {"support-config.json", "source-clock.json"}
             or base_profile == CAMERA_PROFILE and name in {"camera-coupling.json", "camera.json", "ground.json",
                 "ground-clearance.json", "two-view-triangulation.json", "camera-provenance.json"}
             or feet and name in {"foot-reference.json", "feet-migration.json"}}
    if feet:
        views["foot-reference.json"] = scene["foot_extension"]["reference_json"].encode("utf-8")
    if "geometry-hypothesis.json" in views and "input-config.json" in inputs:
        views["input-config.json"] = inputs["input-config.json"]
    if base_profile == STATIC_SUPPORT_PROFILE and "inputs/support-config.json" in inputs:
        if "support-config.json" in views and views["support-config.json"] != inputs["inputs/support-config.json"]:
            raise ValueError("Static-support lab has conflicting configuration artifacts")
        views["support-config.json"] = inputs["inputs/support-config.json"]
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
    if camera_attachment is not None:
        files.update(camera_attachment["files"])
    elif policy is not None:
        files.update({"learning-run/manifest.json": learning_manifest,
                      **{"learning-run/" + name: raw for name, raw in learning_files.items()},
                      "policy.json": learning_files["policy.json"]})
    if evaluation_manifest is not None:
        files.update({"policy-evaluation-run/manifest.json": evaluation_manifest,
                      **{"policy-evaluation-run/" + name: raw for name, raw in evaluation_files.items()},
                      "policy-evaluation.json": evaluation_files["evaluation.json"]})
    validate_package_layout(files)
    _deadline(deadline)
    manifest = _bytes({"schema_version": 1, "created_at": local_time(), "status": "ready",
                       "control_profile": CONTROL_PROFILE, "files": {name: _sha(raw) for name, raw in files.items()}})
    event = {"stage": "shared_exercise_import", "message": ("Verified coupled-feet task; original body scoring and separate unscored ankle diagnostics"
             if feet else "Verified camera-based body task; ground/equipment contact and learning remain unqualified"
             if scene["profile"] == CAMERA_PROFILE else "Verified immutable person-only task and ground-only inventory"
             if scene["profile"] == HUMAN_ONLY_PROFILE else "Verified immutable human task and declared static supports"
             if scene["profile"] == STATIC_SUPPORT_PROFILE else "Verified immutable human-only task and passive equipment"),
             "time": local_time()}
    config = {key: contract[key] for key in ("schema_version", "profile", "control_profile", "scene_sha256", "task_sha256",
                                           "reset_options", "learned_policy_available")}
    config.update(created_at=local_time(), source_run=str(source), manifest_sha256=_sha(manifest))
    if policy is not None:
        config.update({key: policy_identity[key] for key in ("policy_sha256", "learning_manifest_sha256")})
    if evaluation_source is not None:
        config.update({key: policy_identity[key] for key in ("policy_evaluation_manifest_sha256", "policy_evaluation_sha256")})
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
            "learned_policy_available": policy is not None, **policy_identity, "files": len(files), "bytes": total,
            "seconds": time.monotonic() - started}


def load_shared_exercise_lab(root, *, config=None, deadline=None):
    """Verify the complete sealed package before exposing its immutable task and scene."""
    root = Path(root)
    deadline = time.monotonic() + 60 if deadline is None else deadline
    manifest_raw, files = _verified_inputs(root, deadline=deadline, maximum_files=MAX_OUTPUT_FILES,
                                           maximum_bytes=MAX_OUTPUT_BYTES, preflight=_feet_metadata_preflight(resume=True))
    if _json(manifest_raw).get("control_profile") != CONTROL_PROFILE:
        raise ValueError("Exercise lab manifest must explicitly select human-only control")
    required = {"scene.json", "exercise-task.json", "exercise-lab.json", "source-run/manifest.json"}
    if not required <= set(files):
        raise ValueError("Exercise lab is missing its pinned scene, task or source identity")
    scene = validate_shared_world(_json(files["scene.json"]))
    if scene["profile"] == FEET_PROFILE and len(files["exercise-task.json"]) > 128 * 1024:
        raise ValueError("Coupled-feet task exceeds its 128 KiB document bound")
    task = _validate_task(_json(files["exercise-task.json"]), scene)
    contract = _json(files["exercise-lab.json"])
    fixed = {"schema_version": 1, "profile": "shared-world", "control_profile": CONTROL_PROFILE,
             "scene_sha256": canonical_hash(scene), "task_sha256": canonical_hash(task),
             "reset_options": RESET_OPTIONS,
             "source_manifest_sha256": _sha(files["source-run/manifest.json"])}
    if (not isinstance(contract, dict) or type(contract.get("schema_version")) is not int
            or type(contract.get("learned_policy_available")) is not bool
            or any(contract.get(key) != value for key, value in fixed.items())):
        raise ValueError("Exercise lab contract differs from its pinned task, scene or reset protocol")
    implementation = contract.get("implementation_sha256")
    if (not isinstance(implementation, dict) or not implementation or any(not isinstance(name, str) for name in implementation)
            or any(
            "lab-implementation/" + name not in files or _sha(files["lab-implementation/" + name]) != expected
            for name, expected in implementation.items())):
        raise ValueError("Exercise lab runtime closure differs from its pinned implementation")
    camera_policy = scene["profile"] == CAMERA_PROFILE and contract["learned_policy_available"]
    if camera_policy:
        from weft.camera_shared_policy_admission import execution_runtime_hashes
        expected_runtime = execution_runtime_hashes(scene)
        if any(implementation.get(name) != pin for name, pin in expected_runtime.items()):
            raise ValueError("Camera policy runtime differs from the retained lab implementation")
    else:
        expected_runtime = {name: implementation.get(name) for name in runtime_files(scene)}
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
    policy = None
    evaluation_successes, evaluation_reset_protocol = None, None
    camera_attachment = None
    if scene["profile"] == FEET_PROFILE and (contract["learned_policy_available"]
            or any(name in files for name in ("policy.json", "policy-evaluation.json"))
            or any(name.startswith(("learning-run/", "policy-evaluation-run/")) for name in files)
            or any(name in contract for name in ("policy_sha256", "exercise_policy", "policy_evaluation_sha256"))):
        raise ValueError("Coupled-feet/camera labs cannot expose unqualified policies or policy evaluation")
    if camera_policy:
        from weft.camera_shared_policy_lab import load_camera_attachment
        camera_attachment = load_camera_attachment(files, scene, task, contract, deadline=deadline)
        policy = camera_attachment["policy"]
        evaluation_successes = camera_attachment["summary"]["evaluation_successes"]
    elif contract["learned_policy_available"]:
        if not {"policy.json", "learning-run/manifest.json"} <= set(files):
            raise ValueError("Exercise lab learned policy lacks its complete training package")
        learning_files = {name[len("learning-run/"):]: raw for name, raw in files.items()
                          if name.startswith("learning-run/") and name != "learning-run/manifest.json"}
        policy, identity = _validate_policy_package(files["learning-run/manifest.json"], learning_files, scene, task, deadline=deadline)
        if (files["policy.json"] != learning_files["policy.json"]
                or any(contract.get(key) != value for key, value in identity.items())):
            raise ValueError("Exercise lab learned-policy identity differs from its sealed contract")
        collect_shared_source_files(scene, root / "learning-run", expected_manifest_sha256=identity["learning_manifest_sha256"])
        has_evaluation = ("policy-evaluation.json" in files or any(name.startswith("policy-evaluation-run/") for name in files)
                          or any(key in contract for key in ("policy_evaluation_manifest_sha256", "policy_evaluation_sha256",
                                                             "policy_evaluation_files_verified")))
        if has_evaluation:
            if not {"policy-evaluation.json", "policy-evaluation-run/manifest.json"} <= set(files):
                raise ValueError("Exercise policy evaluation lacks its complete separate package")
            evaluation_files = {name[len("policy-evaluation-run/"):]: raw for name, raw in files.items()
                                if name.startswith("policy-evaluation-run/") and name != "policy-evaluation-run/manifest.json"}
            evaluation_identity, evaluation_successes, evaluation_reset_protocol = _validate_policy_evaluation(
                files["policy-evaluation-run/manifest.json"], evaluation_files,
                files["learning-run/manifest.json"], learning_files, policy)
            if (files["policy-evaluation.json"] != evaluation_files["evaluation.json"]
                    or any(contract.get(key) != value for key, value in evaluation_identity.items())):
                raise ValueError("Exercise lab evaluation identity differs from its sealed contract")
            collect_shared_source_files(scene, root / "policy-evaluation-run",
                                        expected_manifest_sha256=evaluation_identity["policy_evaluation_manifest_sha256"])
        if contract.get("exercise_policy") != _policy_status(policy, learning_files["training.json"],
                evaluation_successes=evaluation_successes, evaluation_reset_protocol=evaluation_reset_protocol):
            raise ValueError("Exercise lab policy status differs from its sealed training/evaluation evidence")
    elif "policy.json" in files or any(name.startswith("learning-run/") for name in files):
        raise ValueError("Exercise lab cannot expose an unbound learned policy")
    elif ("policy-evaluation.json" in files or any(name.startswith("policy-evaluation-run/") for name in files)
          or "exercise_policy" in contract):
        raise ValueError("Exercise lab cannot expose evaluation without a learned policy")
    videos = [name for name in files if re.fullmatch(r"source\.(mp4|webm|mov|m4v|avi|mkv)", name)]
    if len(videos) > 1 or (videos and scene["source_binding"]["mode"] == "source_qualified"
                          and _sha(files[videos[0]]) != scene["source_binding"]["source_sha256"]):
        raise ValueError("Exercise lab original video differs from its canonical source binding")
    if config is not None:
        config_keys = ("schema_version", "profile", "control_profile", "scene_sha256", "task_sha256", "reset_options",
                       "learned_policy_available")
        if (type(config.get("schema_version")) is not int or type(config.get("learned_policy_available")) is not bool
                or any(config.get(key) != contract[key] for key in config_keys)
                or config.get("manifest_sha256") != _sha(manifest_raw)):
            raise ValueError("Exercise lab configuration differs from its sealed scene/task identity")
        if policy is not None and any(config.get(key) != contract[key] for key in ("policy_sha256", "learning_manifest_sha256")):
            raise ValueError("Exercise lab configuration differs from its pinned policy identity")
        if evaluation_successes is not None and any(config.get(key) != contract[key]
                for key in ("policy_evaluation_manifest_sha256", "policy_evaluation_sha256")):
            raise ValueError("Exercise lab configuration differs from its pinned evaluation identity")
    proof = collect_shared_source_files(scene, root, expected_manifest_sha256=_sha(manifest_raw))
    _deadline(deadline)
    return {"scene": scene, "task": task, "contract": contract, "source_verification": proof.report,
            **({"policy": policy, "policy_document": files["policy.json"]} if policy is not None else {}),
            **({"camera_policy_admission": camera_attachment["admission"]} if camera_attachment else {}),
            "artifacts": [{"name": name, "bytes": len(raw)} for name, raw in sorted(files.items()) if "/" not in name]
                         + [{"name": "manifest.json", "bytes": len(manifest_raw)}]}


class SharedExerciseSession:
    control_profile = CONTROL_PROFILE
    scene_profile = "shared-world-v1"
    policy = None

    def __init__(self, root, *, config=None):
        loaded = load_shared_exercise_lab(root, config=config)
        self._scene, self._task = loaded["scene"], loaded["task"]
        self.scene_profile = self._scene["profile"]
        self.artifacts = loaded["artifacts"]
        self.source_binding_verification = loaded["source_verification"]
        self.policy = loaded.get("policy")
        self.policy_sha256 = loaded["contract"].get("policy_sha256")
        self._exercise_policy = loaded["contract"].get("exercise_policy")
        self._camera_policy_admission = loaded.get("camera_policy_admission")
        self._policy_admission_sha256 = loaded["contract"].get("camera_policy_admission_sha256")
        self._policy_sample = None
        if self.scene_profile == CAMERA_PROFILE:
            from weft.camera_shared_recording import recording_scene as camera_recording_scene
            from weft.camera_shared_task import CameraSharedTaskEnv

            if self.policy is not None:
                from weft.camera_shared_policy_recording import recording_scene as policy_recording_scene
                self._recording_scene = policy_recording_scene(self._scene, self._task,
                    policy_document=loaded["policy_document"], policy_admission=self._camera_policy_admission)
            else:
                self._recording_scene = camera_recording_scene(self._scene, self._task)
            self.env = CameraSharedTaskEnv(self._scene, self._task)
        elif self.scene_profile == FEET_PROFILE:
            from weft.shared_exercise_feet_recording import recording_scene as feet_recording_scene
            from weft.shared_exercise_feet_task import SharedExerciseFeetTaskEnv

            self._recording_scene = feet_recording_scene(self._scene, self._task)
            self.env = SharedExerciseFeetTaskEnv(self._scene, self._task)
        else:
            self._recording_scene = recording_scene(self._scene, self._task, policy_document=loaded.get("policy_document"))
            self.env = SharedExerciseTaskEnv(self._scene, self._task)
        try:
            self.reset(0)
            self._policy_action = None
            if self._camera_policy_admission is not None:
                from weft.camera_shared_policy_admission import prepare_camera_policy_sampler
                self._policy_action = prepare_camera_policy_sampler(
                    loaded["policy_document"], self._camera_policy_admission, self.env)
            elif self.policy is not None:
                from weft.shared_exercise_policies import prepare_policy
                self._policy_action = prepare_policy(self.policy, self.env)
            from weft.capabilities import describe_capabilities
            self._capabilities = describe_capabilities(self._scene, lab_profile=CONTROL_PROFILE,
                exercise_task=self._task, reference_available=True,
                camera_policy_admission=self._camera_policy_admission,
                learned_policy_available=self.policy is not None, trained_target_id="human" if self.policy is not None else None,
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

    def exercise_policy(self):
        return deepcopy(self._exercise_policy)

    def reset(self, seed):
        self._policy_sample = None
        self.observation, _ = self.env.reset(seed=seed, options=RESET_OPTIONS)
        self.done = False
        return self.snapshot()

    def policy_action(self):
        if self._policy_action is None:
            raise ValueError("This human-only exercise task has no learned policy; use manual, coast or authored reference")
        if self._camera_policy_admission is not None:
            action, diagnostic = self._policy_action()
            self._policy_sample = (action.tolist(), diagnostic)
            return action.tolist()
        return self._policy_action().tolist()

    def policy_recording_metadata(self, action):
        if self._camera_policy_admission is None or self._policy_sample is None or action != self._policy_sample[0]:
            raise ValueError("Camera learned event requires its actual prepared policy decision")
        return {"policy_sha256": self.policy_sha256,
                "policy_admission_sha256": self._policy_admission_sha256,
                "policy_sample": deepcopy(self._policy_sample[1])}

    def reference_action(self):
        return self.env.authored_action().tolist()

    def resolve_action(self, action):
        if self.scene_profile == CAMERA_PROFILE:
            from weft.camera_shared_recording import validate_action as validate_camera_action
            validate_camera_action(action, self._recording_scene)
        elif self.scene_profile == FEET_PROFILE:
            from weft.shared_exercise_feet_recording import validate_action as validate_feet_action
            validate_feet_action(action, self._recording_scene)
        else:
            validate_action(action, self._recording_scene)
        return [float(value) for value in action]

    def step(self, action):
        self.observation, reward, terminated, truncated, info = self.env.step(self.resolve_action(action))
        self.done = terminated or truncated
        return reward, terminated, truncated, info

    def snapshot(self):
        return self.env.snapshot()


def verify_shared_exercise_action_replay(root, *, max_seconds=60):
    root = Path(root)
    envelope = _json(read_package_file(root, "scene.json", max_bytes=16 * 1024 * 1024))
    if isinstance(envelope, dict) and envelope.get("profile") == "camera-shared-policy-recording-v1":
        from weft.camera_shared_policy_recording import verify_action_replay as verify_camera_policy_replay
        return verify_camera_policy_replay(root, max_seconds=max_seconds)
    if isinstance(envelope, dict) and envelope.get("profile") == "camera-shared-recording-v1":
        from weft.camera_shared_recording import verify_action_replay as verify_camera_replay
        return verify_camera_replay(root, max_seconds=max_seconds)
    if isinstance(envelope, dict) and envelope.get("profile") == "shared-exercise-feet-recording-v1":
        from weft.shared_exercise_feet_recording import verify_action_replay as verify_feet_replay
        return verify_feet_replay(root, max_seconds=max_seconds)
    return verify_action_replay(root, max_seconds=max_seconds)


def export_shared_exercise_session(root):
    from weft.recording import export_recording

    deadline = time.monotonic() + 180
    exported = export_recording(root, deadline=deadline, max_output_bytes=128 * 1024 * 1024)
    _deadline(deadline)
    replay = verify_shared_exercise_action_replay(root, max_seconds=deadline - time.monotonic())
    return {**exported, "action_replay": replay}
