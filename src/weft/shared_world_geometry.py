"""Admit explicitly authored compound alternatives without rewriting source evidence."""

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
from weft.shared_exercise_task import validate_shared_exercise_task
from weft.shared_world import SharedWorldEnv, shared_world_xml, validate_shared_world
from weft.shared_world_packaging import (
    collect_shared_source_files, read_package_file, read_package_manifest, validate_package_layout, write_package_files,
)
from weft.world_evidence import _check_tree, _nonfinite, _unique_object


CONFIG_PROFILE = "shared-world-geometry-hypothesis-config-v1"
PROFILE = "shared-world-geometry-hypothesis-v1"
MAX_CONFIG_BYTES = 64 * 1024
MAX_SCENE_BYTES = 16 * 1024 * 1024
MAX_INPUT_FILES = 512
MAX_INPUT_BYTES = 128 * 1024 * 1024
MAX_OUTPUT_FILES = 1024
MAX_OUTPUT_BYTES = 192 * 1024 * 1024
MAX_IMPLEMENTATION_BYTES = 16 * 1024 * 1024
METADATA_RESERVE_BYTES = 2 * 1024 * 1024
FINALIZATION_GRACE_SECONDS = 5
MAX_FAILURE_RECEIPT_BYTES = 64 * 1024
IMPLEMENTATION_FILES = (
    "shared_world_geometry.py", "shared_world.py", "shared_world_observation.py", "shared_world_packaging.py",
    "shared_world_evidence.py", "source_equivalence.py", "world_evidence.py", "observations.py", "human.py",
    "mobile_human.py", "whole_body.py", "body_frame.py", "spatial_pose.py", "root_motion.py", "spatial_rigid.py",
    "errors.py", "io.py", "shared_exercise_task.py", "shared_exercise_reset.py",
)
TERMINAL_INPUT_STATUSES = frozenset({"ready", "trained", "evaluated"})
SCOPE = ("Inspectable authored compound-geometry hypothesis with an unchanged task and exact source bindings. "
         "Zero-step initial-clearance admission is not source-image fit, full-reference clearance, correct contact, "
         "recovered cavity, grasp, exercise success or physical validation.")
ASSUMPTION = ("Selected rigid geometry/inertia is an explicit authored compound hypothesis. Prior geometry-fit "
              "claims are archived only; source fit and physical accuracy of this alternative are unverified.")
BASE_CONTEXT = "Inherited base context (geometry/inertia claims superseded for selected objects): "


def _bytes(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _json(raw):
    try:
        value = json.loads(raw, object_pairs_hook=_unique_object, parse_constant=_nonfinite)
        _check_tree(value)
    except (UnicodeError, json.JSONDecodeError, RecursionError) as error:
        raise ValueError("Geometry hypotheses require bounded strict JSON") from error
    if not isinstance(value, dict):
        raise ValueError("Geometry hypothesis documents must be JSON objects")
    return value


def _deadline(deadline):
    if time.monotonic() >= deadline:
        raise TimeoutError("Geometry hypothesis exceeded its cooperative wall-time bound")


def _path(value):
    path = Path(value).absolute()
    if ".." in path.parts:
        raise ValueError("Geometry hypothesis paths cannot traverse parent directories")
    for index, item in enumerate((path, *path.parents)):
        try:
            info = item.lstat()
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(info.st_mode) or index and not stat.S_ISDIR(info.st_mode):
            raise ValueError("Geometry hypothesis paths cannot contain symlinks or non-directory parents")
    return path


def validate_geometry_hypothesis_config(config):
    keys = {"schema_version", "profile", "base_scene_sha256", "assumptions", "replacements"}
    if (not isinstance(config, dict) or set(config) != keys or type(config["schema_version"]) is not int
            or config["schema_version"] != 1 or config["profile"] != CONFIG_PROFILE
            or not isinstance(config["base_scene_sha256"], str)
            or not re.fullmatch(r"[0-9a-f]{64}", config["base_scene_sha256"])):
        raise ValueError("Use the versioned geometry-hypothesis config pinned to its canonical base scene")
    assumptions = config["assumptions"]
    if (not isinstance(assumptions, list) or not 1 <= len(assumptions) <= 8
            or any(not isinstance(value, str) or not 1 <= len(value.strip()) <= 500 or len(value) > 500 for value in assumptions)):
        raise ValueError("Geometry hypotheses require 1–8 bounded authored assumptions")
    rows = config["replacements"]
    if not isinstance(rows, list) or not 1 <= len(rows) <= 12:
        raise ValueError("Replace geometry for one to twelve existing rigid objects")
    ids = set()
    for row in rows:
        if (not isinstance(row, dict) or set(row) != {"object_id", "geometry"}
                or not isinstance(row["object_id"], str) or not re.fullmatch(r"[a-z][a-z0-9_-]{0,63}", row["object_id"])
                or row["object_id"] in ids or row["object_id"] == "human" or not isinstance(row["geometry"], dict)
                or row["geometry"].get("profile") != "rigid-compound-v1"):
            raise ValueError("Each replacement requires one unique existing object ID and a complete rigid-compound-v1 geometry")
        ids.add(row["object_id"])
    return deepcopy(config)


def _inputs(source, *, deadline, verified, archive):
    manifest, hashes = read_package_manifest(source)
    archive["source-run/manifest.json"] = manifest
    _terminal_input(manifest, "manifest.json")
    if len(hashes) > MAX_INPUT_FILES:
        raise ValueError("Geometry hypotheses require a complete successful bounded input package")
    size = len(manifest)
    for name, expected in hashes.items():
        _deadline(deadline)
        mapping_log = re.fullmatch(r"(?:(?:source-run|learning-run|policy-evaluation-run)/)*source-mappings/[0-9a-f]{64}/[^/]+\.log", name)
        if (any(part.startswith(".") or part == "__pycache__" for part in Path(name).parts)
                or name.endswith(".pyc") or name.endswith(".log") and not mapping_log):
            raise ValueError("Geometry inputs cannot contain hidden files, caches or unrelated logs")
        raw = read_package_file(source, name, max_bytes=min(64 * 1024 * 1024, max(1, MAX_INPUT_BYTES - size)),
                                expected_sha256=expected)
        size += len(raw)
        if size > MAX_INPUT_BYTES:
            raise ValueError("Geometry input package exceeds its byte bound")
        verified[name] = raw
        archive["source-run/" + name] = raw
    for name in ("status.json", "report.json", "reconstruction.json", "evaluation.json", "training.json"):
        if name in verified:
            _terminal_input(verified[name], name)
    return manifest


def _terminal_input(raw, name):
    document = _json(raw)
    if "status" in document and (not isinstance(document["status"], str)
                                or document["status"] not in TERMINAL_INPUT_STATUSES):
        raise ValueError("Geometry input requires a supported terminal status or historical absent status: " + name)


def _source_views(base, inputs, proof):
    views = {name: inputs[name] for name in ("source.json", "pose-quality.json") if name in inputs}
    names = [name for name in inputs if re.fullmatch(r"source\.(mp4|webm|mov|m4v|avi|mkv)", name)]
    source_sha = base["source_binding"]["source_sha256"]
    if len(names) > 1 or names and (source_sha is None or _sha(inputs[names[0]]) != source_sha):
        raise ValueError("The source view requires a single exact original video matching the canonical source SHA256")
    if names:
        views[names[0]] = inputs[names[0]]
    elif source_sha is not None:
        videos = {Path(name).name: raw for name, raw in proof.files.items()
                  if re.fullmatch(r"source\.(mp4|webm|mov|m4v|avi|mkv)", Path(name).name) and _sha(raw) == source_sha}
        if len(videos) > 1:
            raise ValueError("Sealed source mappings contain ambiguous original video display names")
        views.update(videos)
    return views


def _implementation():
    package = Path(__file__).parent
    files, total = {}, 0
    for name in IMPLEMENTATION_FILES:
        raw = read_package_file(package, name, max_bytes=min(8 * 1024 * 1024, max(1, MAX_IMPLEMENTATION_BYTES - total)))
        files[name], total = raw, total + len(raw)
        if total > MAX_IMPLEMENTATION_BYTES:
            raise ValueError("Geometry implementation snapshot exceeds its byte bound")
    for name in ("pyproject.toml", "requirements.lock", "requirements-lab.lock"):
        raw = read_package_file(package.parent.parent, name, max_bytes=1024 * 1024)
        files["project/" + name], total = raw, total + len(raw)
    if total > MAX_IMPLEMENTATION_BYTES:
        raise ValueError("Geometry implementation snapshot exceeds its byte bound")
    return files


def _build(base, config, config_sha):
    if config["base_scene_sha256"] != canonical_hash(base):
        raise ValueError("Geometry configuration is stale or belongs to a different canonical base scene")
    objects = {obj["id"]: obj for obj in base["objects"]}
    replacements = {row["object_id"]: row["geometry"] for row in config["replacements"]}
    if not set(replacements) <= set(objects):
        raise ValueError("Geometry replacements cannot add or rename existing rigid objects")
    scene, previous = deepcopy(base), []
    for obj in scene["objects"]:
        if obj["id"] not in replacements:
            continue
        original = objects[obj["id"]]
        old_geometry = {key: deepcopy(original[key]) for key in ("shape", "size", "radius", "geometry") if key in original}
        previous.append({"object_id": obj["id"], "previous_geometry": old_geometry,
                         "previous_provenance": deepcopy(original["provenance"]),
                         "authored_replacement": deepcopy(replacements[obj["id"]])})
        obj.pop("size", None)
        obj.pop("radius", None)
        obj["shape"], obj["geometry"] = "compound", deepcopy(replacements[obj["id"]])
        retained = {key: deepcopy(original["provenance"][key]) for key in
                    ("source_prefix", "source_reference", "source_bounds_px", "camera_hypothesis", "reference_object_force")
                    if key in original["provenance"]}
        uncertainty = original["provenance"].get("uncertainty", {})
        if isinstance(uncertainty, dict):
            retained["uncertainty"] = {key: deepcopy(uncertainty[key]) for key in ("identity", "mass", "friction", "camera")
                                       if key in uncertainty}
        retained.setdefault("uncertainty", {}).update(shape="authored_compound_not_measured",
            depth="authored_local_parts_and_retained_object_origin_not_recovered_depth",
            support="initial_clearance_is_not_measured_support", inertia="authored_centered_diagonal_prior",
            hands_and_grasp="not_reconstructed")
        obj["provenance"] = {**retained, "mode": "authored_geometry_hypothesis", "config_sha256": config_sha,
            "base_scene_sha256": config["base_scene_sha256"], "assumptions": deepcopy(config["assumptions"]),
            "previous_geometry_provenance": deepcopy(original["provenance"]),
            "previous_geometry_validation_inherited": False, "source_fit": "unverified", "physical_validation": "unverified",
            "source_binding_scope": "Exact entity observations retained; replacement geometry/inertia is authored, not measured."}
    scene["assumptions"] = [BASE_CONTEXT + value for value in scene["assumptions"]] + [ASSUMPTION]
    scene["learned_policy_available"] = False
    return scene, previous


def _invariants(base, scene, selected):
    metadata = {"objects", "assumptions", "learned_policy_available"}
    if any(scene.get(key) != value for key, value in base.items() if key not in metadata):
        raise ValueError("Geometry hypothesis changed the body, world frame, source binding or task")
    geometry_fields = {"shape", "size", "radius", "geometry", "provenance"}
    if [obj["id"] for obj in scene["objects"]] != [obj["id"] for obj in base["objects"]]:
        raise ValueError("Geometry hypothesis changed rigid-object identities or order")
    for original, changed in zip(base["objects"], scene["objects"]):
        if original["id"] not in selected:
            if original != changed:
                raise ValueError("Geometry hypothesis changed an unselected object")
        elif ({key: value for key, value in original.items() if key not in geometry_fields}
              != {key: value for key, value in changed.items() if key not in geometry_fields}):
            raise ValueError("Geometry hypothesis changed an object's pose, velocity, mass, material or role")
        for key in ("source_prefix", "source_reference", "source_bounds_px", "camera_hypothesis", "reference_object_force"):
            if original["provenance"].get(key) != changed["provenance"].get(key):
                raise ValueError("Geometry hypothesis changed object source clocks, observations or camera assumptions")
    return {"human_spec_unchanged": True, "source_bindings_unchanged": True, "coordinates_camera_clocks_unchanged": True,
            "floor_gravity_contact_solver_unchanged": True, "task_and_action_channels_unchanged": True,
            "object_ids_poses_velocities_mass_materials_unchanged": True,
            "unselected_objects_unchanged": True, "source_slots_and_reference_arrays_unchanged": True}


def _initial_check(scene):
    env = SharedWorldEnv(scene)
    try:
        env.reset(seed=0, options={"jitter": False})
        state = env.snapshot()
        return {"status": "passed_existing_initial_clearance_checks", "physics_steps": env.steps,
                "reset_options": {"jitter": False}, "source_time_s": state["source_time_s"],
                "initial_objects": state["objects"], "contact_details": state["contact_details"],
                "contact_count": state["contact_count"], "scope": SCOPE}
    finally:
        env.close()


def _rebind_task(task, raw, base, candidate):
    task = validate_shared_exercise_task(task, base)
    updated = deepcopy(task)
    updated["scene_sha256"] = canonical_hash(candidate)
    updated = validate_shared_exercise_task(updated, candidate)
    if {key: value for key, value in updated.items() if key != "scene_sha256"} != {
            key: value for key, value in task.items() if key != "scene_sha256"}:
        raise ValueError("Geometry admission changed an exercise task parameter")
    document = _bytes(updated)
    return document, {"status": "metadata_only_scene_rebind", "changed_fields": ["scene_sha256"],
        "previous_file_sha256": _sha(raw), "new_file_sha256": _sha(document),
        "previous_task_sha256": canonical_hash(task), "new_task_sha256": canonical_hash(updated),
        "previous_scene_sha256": canonical_hash(base), "new_scene_sha256": canonical_hash(candidate),
        "parameters_source_scope_and_gates_unchanged": True}


def _reconstruction(report):
    keys = ("schema_version", "profile", "status", "created_at", "source_fit", "physical_validation", "scope",
            "base_scene_sha256", "base_manifest_sha256", "input_config_sha256", "candidate_scene_sha256",
            "selected_objects", "preserved", "policy_invalidation", "exercise_task", "reasons", "full_reference_clearance_checked")
    details = {"config": "input-config.json", "geometry_changes": "previous-geometry.json",
               "full_report": "geometry-hypothesis.json", "initial_checks": "initial-geometry-check.json"}
    if report.get("custody", {}).get("mode") == "receipt_only":
        details = {"full_report": "geometry-hypothesis.json"}
    return {**{key: report[key] for key in keys}, "mode": "authored_compound_geometry_hypothesis", "details": details,
            **({"custody": report["custody"]} if "custody" in report else {})}


def _bounded_plan(files):
    validate_package_layout(files)
    if (len(files) + 6 > MAX_OUTPUT_FILES or sum(map(len, files.values())) + METADATA_RESERVE_BYTES > MAX_OUTPUT_BYTES
            or any(len(raw) > 64 * 1024 * 1024 for raw in files.values())):
        raise ValueError("Complete geometry hypothesis output exceeds its file/byte bound")


def _receipt_only(out, omitted, report, error):
    keys = ("schema_version", "profile", "status", "created_at", "source_fit", "physical_validation", "scope",
            "base_scene_sha256", "base_manifest_sha256", "input_config_sha256", "candidate_scene_sha256", "input_paths",
            "selected_objects", "preserved", "policy_invalidation", "exercise_task", "reasons", "full_reference_clearance_checked",
            "initial_geometry_admitted", "input_files_verified", "bounds", "physics_steps", "fits", "inference_calls",
            "api_calls", "video_decodes", "implementation_sha256", "runtime_versions", "deadline_scope", "failure_preservation_grace_seconds")
    compact = {key: deepcopy(report[key]) for key in keys}
    if compact["status"] == "ready":
        compact["status"] = "rejected"
    compact["reasons"].append({"stage": "receipt_only_publication", "type": type(error).__name__, "message": str(error)[:2000]})
    compact["custody"] = {"mode": "receipt_only", "complete_input_archive_preserved": False,
        "config_bytes_preserved": False, "candidate_document_preserved": False,
        "omitted_files": len(omitted), "omitted_bytes": sum(map(len, omitted.values())),
        "omitted_file_index_sha256": canonical_hash({name: _sha(raw) for name, raw in omitted.items()}),
        "scope": "Only this bounded receipt is copied. Hashes identify the verified input snapshot; original bytes remain at input paths, not in this output."}
    return _write_output(out, {}, compact, deadline=time.monotonic() + FINALIZATION_GRACE_SECONDS, receipt_only=True)


def _preserve_failed_file(out, name, preserved):
    # A no-overwrite link then unlink renames only this attempt's bytes, never an input artifact.
    fd = os.open(out, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        if not stat.S_ISREG(os.stat(name, dir_fd=fd, follow_symlinks=False).st_mode):
            raise ValueError("Publication recovery requires regular output files")
        os.link(name, preserved, src_dir_fd=fd, dst_dir_fd=fd, follow_symlinks=False)
        os.unlink(name, dir_fd=fd)
    finally:
        os.close(fd)


def _write_output(out, files, report, *, deadline, receipt_only=False):
    files = dict(files)
    files["geometry-hypothesis.json"] = _bytes(report)
    files["reconstruction.json"] = _bytes(_reconstruction(report))
    try:
        if not receipt_only:
            _bounded_plan(files)
        manifest = _bytes({"schema_version": 1, "status": report["status"], "files": {name: _sha(raw) for name, raw in files.items()}})
        if len(manifest) > METADATA_RESERVE_BYTES:
            raise ValueError("Geometry hypothesis manifest exceeds its metadata reserve")
        if receipt_only and (len(files) + 1 > MAX_OUTPUT_FILES
                or sum(map(len, files.values())) + len(manifest) > min(MAX_FAILURE_RECEIPT_BYTES, MAX_OUTPUT_BYTES)):
            raise ValueError("Failure receipt exceeds its separate bounded publication allowance")
    except ValueError as error:
        if receipt_only:
            raise
        return _receipt_only(out, files, report, error)
    _path(out)
    out.mkdir(parents=True, exist_ok=False, mode=0o700)
    written, active = {}, None
    try:
        # The runnable scene is published last; failed admission never creates it.
        order = [name for name in files if name != "scene.json"] + (["scene.json"] if "scene.json" in files else [])
        for name in order:
            _deadline(deadline)
            active = name
            write_package_files(out, {name: files[name]})
            written[name] = _sha(files[name])
        _deadline(deadline)
        active = None
        write_package_files(out, {"manifest.json": manifest})
    except Exception as error:
        # Never erase partial custody or leave a ready seal after interrupted publication.
        failure = {"schema_version": 1, "status": "failed", "scope": "Incomplete output preserved; not a ready geometry hypothesis.",
                   "error_type": type(error).__name__, "message": str(error)[:2000], "created_at": local_time()}
        failure_deadline = time.monotonic() + FINALIZATION_GRACE_SECONDS
        if active is not None and active not in written:
            try:
                actual = read_package_file(out, active, max_bytes=max(1, len(files[active])))
            except ValueError:
                failure["unsealed_partial_artifact"] = active
            else:
                written[active] = _sha(actual)
        failure["preserved_publication_artifacts"] = {}
        for name, preserved in (("manifest.json", "manifest.publication-failed.json"),
                                ("scene.json", "scene.candidate.publication-failed.json"),
                                ("geometry-hypothesis.json", "geometry-hypothesis.publication-failed.json"),
                                ("reconstruction.json", "reconstruction.publication-failed.json")):
            _deadline(failure_deadline)
            try:
                raw = read_package_file(out, name, max_bytes=max(1, len(manifest if name == "manifest.json" else files.get(name, b""))))
            except ValueError:
                continue
            _preserve_failed_file(out, name, preserved)
            written.pop(name, None)
            written[preserved] = _sha(raw)
            failure["preserved_publication_artifacts"][name] = {"path": preserved, "sha256": _sha(raw)}
        _deadline(failure_deadline)
        status_raw = _bytes(failure)
        failed_report = {**report, "status": "failed", "publication_failure": failure}
        failed_files = {"status.json": status_raw, "failure.json": status_raw,
                        "geometry-hypothesis.json": _bytes(failed_report), "reconstruction.json": _bytes(_reconstruction(failed_report))}
        write_package_files(out, failed_files)
        written.update({name: _sha(raw) for name, raw in failed_files.items()})
        _deadline(failure_deadline)
        write_package_files(out, {"manifest.json": _bytes({"schema_version": 1, "status": "failed", "files": written})})
        return {**failed_report, "out": str(out), "manifest_name": "manifest.json",
                "manifest_sha256": _sha(read_package_file(out, "manifest.json", max_bytes=METADATA_RESERVE_BYTES))}
    return {**report, "out": str(out), "manifest_sha256": _sha(manifest),
            "manifest_name": "manifest.json",
            "files": len(files) + 1, "bytes": sum(map(len, files.values())) + len(manifest)}


def prepare_shared_world_geometry(base_run, config_path, out, *, max_seconds=30):
    """Seal a ready authored alternative or a non-ready rejection; never fit or mutate inputs."""
    if type(max_seconds) not in (int, float) or not math.isfinite(max_seconds) or not 1 <= max_seconds <= 60:
        raise ValueError("Geometry hypothesis admission requires a 1–60 second cooperative bound")
    source, config_path, out = map(_path, (base_run, config_path, out))
    if out.exists() or any(out.is_relative_to(path) or path.is_relative_to(out) for path in (source, config_path)):
        raise ValueError("Geometry hypothesis output must be new and outside all immutable inputs")
    started, deadline = time.monotonic(), time.monotonic() + max_seconds
    files, root_sources, implementation, source_views = {}, {}, {}, {}
    base = candidate = config = manifest_raw = config_raw = None
    previous, reasons, preserved = [], [], {}
    inputs, proof, initial = {}, None, None
    task, task_document = None, None
    task_receipt = {"status": "not_supplied", "changed_fields": []}
    status, stage = "unsupported_input", "configuration"
    try:
        config_raw = read_package_file(config_path.parent, config_path.name, max_bytes=MAX_CONFIG_BYTES)
        files["input-config.json"] = config_raw
        config = validate_geometry_hypothesis_config(_json(config_raw))
        _deadline(deadline)
        stage = "input_verification"
        manifest_raw = _inputs(source, deadline=deadline, verified=inputs, archive=files)
        if "scene.json" not in inputs or len(inputs["scene.json"]) > MAX_SCENE_BYTES:
            raise ValueError("Geometry input requires a bounded sealed shared-world scene")
        base = validate_shared_world(_json(inputs["scene.json"]))
        proof = collect_shared_source_files(base, source, expected_manifest_sha256=_sha(manifest_raw))
        proof.validate_layout(reserved_paths={"source-run", "implementation", "scene.json", "input-config.json",
            "geometry-hypothesis.json", "previous-geometry.json", "initial-geometry-check.json", "candidate.xml",
            "scene.candidate.rejected.json", "reconstruction.json", "source-verification.json", "source.json",
            "status.json", "failure.json", "failure-manifest.json", "manifest.json", "exercise-task.json",
            "manifest.publication-failed.json", "scene.candidate.publication-failed.json",
            "geometry-hypothesis.publication-failed.json", "reconstruction.publication-failed.json"})
        files["source-verification.json"] = _bytes(proof.report)
        source_views = _source_views(base, inputs, proof)
        if "exercise-task.json" in inputs:
            if len(inputs["exercise-task.json"]) > 64 * 1024:
                raise ValueError("Exercise task exceeds its 64 KiB document bound")
            task = validate_shared_exercise_task(_json(inputs["exercise-task.json"]), base)
        implementation = _implementation()
        files.update({"implementation/" + name: raw for name, raw in implementation.items()})
        _deadline(deadline)
        stage = "candidate_schema"
        candidate, previous = _build(base, config, _sha(config_raw))
        candidate = validate_shared_world(candidate)
        preserved = _invariants(base, candidate, {row["object_id"] for row in config["replacements"]})
        if task is not None:
            task_document, task_receipt = _rebind_task(task, inputs["exercise-task.json"], base, candidate)
        if len(_bytes(candidate)) > MAX_SCENE_BYTES:
            raise ValueError("Geometry candidate exceeds its scene-document bound")
        files["candidate.xml"] = shared_world_xml(candidate).encode()
        _deadline(deadline)
        stage = "initial_geometry"
        initial = _initial_check(candidate)
        files["initial-geometry-check.json"] = _bytes(initial)
        _deadline(deadline)
        stage = "input_reverification"
        _path(source)
        _path(config_path)
        read_package_file(config_path.parent, config_path.name, max_bytes=MAX_CONFIG_BYTES, expected_sha256=_sha(config_raw))
        read_package_file(source, "manifest.json", max_bytes=8 * 1024 * 1024, expected_sha256=_sha(manifest_raw))
        for name, raw in inputs.items():
            _deadline(deadline)
            read_package_file(source, name, max_bytes=max(1, len(raw)), expected_sha256=_sha(raw))
        package = Path(__file__).parent
        for name, raw in implementation.items():
            _deadline(deadline)
            directory, relative = (package.parent.parent, name[8:]) if name.startswith("project/") else (package, name)
            read_package_file(directory, relative, max_bytes=max(1, len(raw)), expected_sha256=_sha(raw))
        root_sources = dict(proof.files) | source_views
        if task_document is not None:
            root_sources["exercise-task.json"] = task_document
        stage = "output_reservation"
        _bounded_plan({**files, **root_sources, "scene.json": _bytes(candidate),
                       "previous-geometry.json": _bytes(previous)})
        _deadline(deadline)
        status = "ready"
    except Exception as error:
        status = "rejected" if stage in {"candidate_schema", "initial_geometry", "output_reservation"} else "unsupported_input"
        if isinstance(error, TimeoutError):
            status = "failed"
        reasons.append({"stage": stage, "type": type(error).__name__, "message": str(error)[:2000]})
        if stage == "initial_geometry" and initial is None:
            files["initial-geometry-check.json"] = _bytes({"status": "rejected", "physics_steps": 0,
                "reason": reasons[-1], "scope": SCOPE})
    manifest_raw = manifest_raw or files.get("source-run/manifest.json")
    candidate_raw = _bytes(candidate) if candidate is not None else None
    if candidate_raw is not None and len(candidate_raw) <= MAX_SCENE_BYTES:
        files["scene.json" if status == "ready" else "scene.candidate.rejected.json"] = candidate_raw
    if previous:
        files["previous-geometry.json"] = _bytes({"base_assumptions": base["assumptions"],
            "hypothesis_assumptions": config["assumptions"], "objects": previous,
            "prior_validation_scope": "Previous geometry/provenance is archived evidence, not validation of this replacement."})
    if status == "ready":
        files.update(root_sources)
    report = {"schema_version": 1, "profile": PROFILE, "status": status, "created_at": local_time(),
        "source_fit": "unverified", "physical_validation": "unverified", "scope": SCOPE,
        "base_scene_sha256": canonical_hash(base) if base is not None else None,
        "base_manifest_sha256": _sha(manifest_raw) if manifest_raw is not None else None,
        "input_config_sha256": _sha(config_raw) if config_raw is not None else None,
        "input_paths": {"base_run": str(source), "config": str(config_path)},
        "candidate_scene_sha256": canonical_hash(candidate) if candidate is not None else None,
        "selected_objects": [row["object_id"] for row in config["replacements"]] if config is not None else [],
        "preserved": preserved, "reasons": reasons, "full_reference_clearance_checked": False,
        "exercise_task": task_receipt,
        "initial_geometry_admitted": initial is not None,
        "policy_invalidation": {"previous_flag": base.get("learned_policy_available") if base else None,
                                "new_flag": False, "old_policy_or_evaluation_attached": False,
                                "reason": "Changed geometry/inertia creates a new scene identity; prior policies/evaluations remain archived only."},
        "input_files_verified": len(inputs), "source_verification": proof.report if proof is not None else None,
        "source_view_files": {name: _sha(raw) for name, raw in source_views.items()},
        "implementation_sha256": {name: _sha(raw) for name, raw in implementation.items()},
        "runtime_versions": {"python": platform.python_version(), **{name: version(name) for name in ("numpy", "mujoco", "gymnasium")}},
        "bounds": {"max_seconds": max_seconds, "max_input_files": MAX_INPUT_FILES, "max_input_bytes": MAX_INPUT_BYTES,
                   "max_output_files": MAX_OUTPUT_FILES, "max_output_bytes": MAX_OUTPUT_BYTES,
                   "max_failure_receipt_bytes": MAX_FAILURE_RECEIPT_BYTES,
                   "max_parts_per_object": 8, "max_rigid_primitives": 32},
        "physics_steps": 0, "fits": 0, "inference_calls": 0, "api_calls": 0, "video_decodes": 0,
        "seconds_before_publication": time.monotonic() - started,
        "deadline_scope": "Cooperative admission/readback/publication checks; bounded native calls are not preemptible.",
        "failure_preservation_grace_seconds": FINALIZATION_GRACE_SECONDS}
    publication_deadline = deadline if status == "ready" else time.monotonic() + FINALIZATION_GRACE_SECONDS
    return _write_output(out, files, report, deadline=publication_deadline)
