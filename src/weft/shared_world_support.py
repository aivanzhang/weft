"""Construct source-bound bodies with explicitly authored static supports."""

import hashlib
from importlib.metadata import version
import math
from pathlib import Path
import platform
import re
import time

from weft import exercise_body_source, shared_world_human
from weft.io import canonical_hash, local_time
from weft.shared_support_task import build_shared_support_task
from weft.shared_world import STATIC_SUPPORT_PROFILE, audit_shared_world_feasibility, validate_shared_world
from weft.shared_world_evidence import build_shared_world_binding, verify_shared_world_sources
from weft.shared_world_packaging import (
    read_package_file, read_package_manifest, validate_package_layout, write_package_files,
)


CONFIG_PROFILE = "shared-static-support-construction-v1"
PROFILE = STATIC_SUPPORT_PROFILE
MAX_SECONDS = 30
MAX_INPUT_BYTES = MAX_OUTPUT_BYTES = 128 << 20
MAX_INPUT_FILES = MAX_OUTPUT_FILES = 512
MAX_FILE_BYTES = 64 << 20
MAX_CONFIG_BYTES = 128 << 10
RESERVE_BYTES = 256 << 10
IMPLEMENTATION_FILES = tuple(dict.fromkeys((
    "shared_world_support.py", "shared_support_task.py", *shared_world_human.IMPLEMENTATION_FILES,
)))
_GENERATED = frozenset({
    "scene.json", "task.json", "exercise-task.json", "construction-evidence.json", "reconstruction.json",
    "reference-feasibility.json", "source-binding-verification.json", "failure.json", "manifest.json",
    "inputs/body-source-manifest.json", "inputs/support-config.json", "implementation", "evidence",
})
_NO_INPUT = shared_world_human._CONFLICTS | {"inventory.json"}
_GROUND_ONLY_ASSUMPTIONS = (
    "Person-only scope is an explicit user declaration, not verified absence of equipment or obstacles.",
    "One articulated human and the authored ground plane share one MuJoCo world; no rigid prop is introduced.",
)
_SUPPORT_ASSUMPTIONS = (
    "Static support shapes, placement, orientation and materials are explicit authored hypotheses in the existing body-scaled world prior.",
    "Support additions have no observed source entity or verified contact identity; the body observations do not establish equipment absence.",
    "Named segment, direction, load and time-window requirements are authored simulator criteria, not recovered hand/foot contact or grasp truth.",
)


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _check(deadline):
    if time.monotonic() >= deadline:
        raise TimeoutError("Static-support construction exceeded its cooperative deadline")


def _path(value):
    path = Path(value).absolute()
    if ".." in path.parts or any(part.is_symlink() for part in (path, *path.parents)):
        raise ValueError("Static-support paths must not contain traversal or symlinks")
    return path


def _not_failed(root):
    if (root / "failure.json").exists() or (root / "failure.json").is_symlink():
        raise ValueError("A failed body source cannot become a static-support candidate")


def _config(value):
    path = None
    if isinstance(value, (str, Path)):
        path = _path(value)
        raw = read_package_file(path.parent, path.name, max_bytes=MAX_CONFIG_BYTES)
    else:
        try:
            raw = shared_world_human._bytes(value, MAX_CONFIG_BYTES)
        except (TypeError, RecursionError) as error:
            raise ValueError("Static-support config requires bounded JSON") from error
    config = shared_world_human._json(raw)
    if (set(config) != {"schema_version", "profile", "source_sha256", "body_source_manifest_sha256", "scale",
                       "objects", "support_requirements", "root_assistance"}
            or type(config["schema_version"]) is not int or config["schema_version"] != 1
            or config["profile"] != CONFIG_PROFILE):
        raise ValueError("Static-support config fields differ from the versioned contract")
    for field in ("source_sha256", "body_source_manifest_sha256"):
        if not isinstance(config[field], str) or not re.fullmatch(r"[0-9a-f]{64}", config[field]):
            raise ValueError("Static-support config requires exact source and body-manifest SHA256 pins")
    scale = config["scale"]
    if type(scale) not in (int, float) or not math.isfinite(scale) or not .5 <= scale <= 2.:
        raise ValueError("Static-support scale prior must be within [0.5, 2.0]")
    if type(config["root_assistance"]) is not bool:
        raise ValueError("Static-support root assistance must be an explicit boolean")
    validate_authored_supports(config["objects"])
    return config, raw, path


def validate_authored_supports(objects):
    """Check authoring provenance before inference; world geometry still needs construction admission."""
    if not isinstance(objects, list) or not 1 <= len(objects) <= 12:
        raise ValueError("Static-support config requires 1–12 authored static objects")
    identifiers = set()
    for obj in objects:
        if (not isinstance(obj, dict) or not isinstance(obj.get("id"), str)
                or not re.fullmatch(r"[a-z][a-z0-9_-]{0,63}", obj["id"])
                or obj["id"] in identifiers or obj["id"] == "human"
                or obj.get("role") != "static" or not isinstance(obj.get("shape"), str)
                or obj["shape"] not in {"box", "sphere", "compound"}):
            raise ValueError("Support objects require unique static box/sphere/compound identities")
        identifiers.add(obj["id"])
        provenance = obj.get("provenance")
        if (not isinstance(provenance, dict) or set(provenance) != {
                "mode", "calibration_status", "physical_validation", "source_contact_verified", "assumptions"}
                or provenance["mode"] != "authored_addition" or provenance["calibration_status"] != "estimated_prior"
                or provenance["physical_validation"] != "unverified" or provenance["source_contact_verified"] is not False):
            raise ValueError("Static supports require explicit authored, uncalibrated and contact-unverified provenance")
        assumptions = provenance["assumptions"]
        if (not isinstance(assumptions, list) or not 1 <= len(assumptions) <= 8
                or any(not isinstance(item, str) or not 1 <= len(item) <= 500
                       or any(ord(char) < 32 for char in item) for item in assumptions)):
            raise ValueError("Support provenance requires 1–8 bounded authored assumptions")
    return identifiers


def _scene(human, observations, binding, config):
    # Reuse placement and clocks without ever admitting a ground-only inventory.
    scene = shared_world_human._scene(human, observations, None, binding, float(config["scale"]))
    del scene["inventory"]
    scene["profile"] = PROFILE
    scene["task"]["type"] = "human_support_v1"
    scene["objects"] = config["objects"]
    for assumption in _GROUND_ONLY_ASSUMPTIONS:
        if scene["assumptions"].count(assumption) != 1:
            raise ValueError("The shared placement scaffold's scope changed")
        scene["assumptions"].remove(assumption)
    scene["assumptions"].extend(_SUPPORT_ASSUMPTIONS)
    return validate_shared_world(scene)


def construct_static_support_run(body_source, config, out, *, max_seconds=30):
    """Seal a candidate and authored support task; native feasibility does not admit a lab."""
    started = time.monotonic()
    if type(max_seconds) not in (int, float) or not math.isfinite(max_seconds) or not 1 <= max_seconds <= MAX_SECONDS:
        raise ValueError("Static-support construction requires a 1–30 second bound")
    deadline = started + max_seconds
    source, target = _path(body_source), _path(out)
    if target.exists() or target.is_relative_to(source) or source.is_relative_to(target):
        raise ValueError("Static-support output must be new and outside its immutable body source")
    _check(deadline)
    config, config_raw, config_path = _config(config)
    _not_failed(source)
    source_identity = source.stat().st_dev, source.stat().st_ino
    manifest_raw, declared = read_package_manifest(source)
    manifest = shared_world_human._json(manifest_raw)
    if (manifest.get("profile") != exercise_body_source.PROFILE or manifest.get("status") != "construction_only"
            or manifest.get("simulation_ready") is not False or len(declared) + 1 > MAX_INPUT_FILES
            or "failure.json" in declared or _NO_INPUT & declared.keys()
            or _sha(manifest_raw) != config["body_source_manifest_sha256"]):
        raise ValueError("Static-support construction requires its exact unrejected construction-only body source")
    for name in declared:
        if any(name == item or name.startswith(item + "/") or item.startswith(name + "/") for item in _GENERATED):
            raise ValueError("Body-source path collides with a generated static-support artifact")
    inputs, total = {}, len(manifest_raw) + len(config_raw)
    for name, expected in declared.items():
        _check(deadline)
        raw = read_package_file(source, name, max_bytes=min(MAX_FILE_BYTES, max(1, MAX_INPUT_BYTES - total)),
                                expected_sha256=expected)
        total += len(raw)
        if total > MAX_INPUT_BYTES:
            raise ValueError("Static-support input exceeds its 128-MiB bound")
        if name in {"status.json", "report.json"} and shared_world_human._json(raw).get("status") in exercise_body_source.REJECTED:
            raise ValueError("Rejected body-source metadata cannot become a static-support world")
        inputs[name] = raw
    marker = shared_world_human._json(inputs.get("body-source.json", b"{}"))
    if marker.get("source_sha256") != config["source_sha256"]:
        raise ValueError("Static-support config and body source video identities differ")
    implementation_names = list(IMPLEMENTATION_FILES)
    if "native-clock.json" in inputs:
        implementation_names.append("source_clock.py")
    if "pose-recovery.json" in inputs:
        from weft.pose_source_recovery import IMPLEMENTATION_FILES as RECOVERY_IMPLEMENTATION
        implementation_names.extend(RECOVERY_IMPLEMENTATION)
    implementations = {}
    for name in dict.fromkeys(implementation_names):
        _check(deadline)
        implementations[name] = read_package_file(Path(__file__).parent, name, max_bytes=2 << 20)
    try:
        human, observations = exercise_body_source.validate_body_source_files(inputs, deadline=deadline)
    except (TypeError, KeyError, AttributeError) as error:
        raise ValueError("Static-support body metadata is malformed") from error
    if (len(observations["objects"]) != 1 or observations["objects"][0]["id"] != "person-0"
            or observations["objects"][0]["properties"]["kind"]["value"] != "person" or observations["relationships"]):
        raise ValueError("Body observations must describe one person; support additions cannot replace observed entities")
    observation_raw = inputs["observations.json"]
    observation_sha = _sha(observation_raw)
    refs = {"human": (observation_sha, "person-0"), **{obj["id"]: None for obj in config["objects"]}}
    binding, construction = build_shared_world_binding({observation_sha: observation_raw},
        source_sha256=config["source_sha256"], world_ids=list(refs), entity_refs=refs, deadline=deadline)
    scene = _scene(human, observations, binding, config)
    if canonical_hash(scene["human_spec"]) != canonical_hash(shared_world_human._json(inputs["human.json"])):
        raise ValueError("Static-support construction cannot rewrite the original body")
    task = build_shared_support_task(scene, support_requirements=config["support_requirements"],
                                     root_assistance=config["root_assistance"])
    task_raw = shared_world_human._bytes(task)
    files = {**inputs, "inputs/body-source-manifest.json": manifest_raw, "inputs/support-config.json": config_raw,
        binding["evidence_documents"][observation_sha]: observation_raw, "construction-evidence.json": construction,
        "scene.json": shared_world_human._bytes(scene), "task.json": shared_world_human._bytes(scene["task"]),
        "exercise-task.json": task_raw, **{"implementation/" + name: raw for name, raw in implementations.items()}}
    validate_package_layout(files)
    owned, output_bytes = {}, 0

    def guard():
        _check(deadline)
        if _path(source) != source or (source.stat().st_dev, source.stat().st_ino) != source_identity:
            raise ValueError("Static-support body source directory changed")
        _not_failed(source)
        read_package_file(source, "manifest.json", max_bytes=8 << 20, expected_sha256=_sha(manifest_raw))
        for name, raw in inputs.items():
            _check(deadline)
            read_package_file(source, name, max_bytes=max(1, len(raw)), expected_sha256=declared[name])
        if config_path is not None:
            _path(config_path)
            read_package_file(config_path.parent, config_path.name, max_bytes=MAX_CONFIG_BYTES, expected_sha256=_sha(config_raw))
        for name, raw in implementations.items():
            _check(deadline)
            read_package_file(Path(__file__).parent, name, max_bytes=max(1, len(raw)), expected_sha256=_sha(raw))

    def write(batch, *, reserve=RESERVE_BYTES):
        nonlocal output_bytes
        if (len(owned) + len(batch) + 2 > MAX_OUTPUT_FILES
                or output_bytes + sum(map(len, batch.values())) + reserve > MAX_OUTPUT_BYTES
                or any(len(raw) > MAX_FILE_BYTES for raw in batch.values())):
            raise ValueError("Static-support output exceeds its 512-file/128-MiB reservation")
        for name, raw in batch.items():
            _check(deadline)
            write_package_files(target, {name: raw})
            owned[name] = _sha(raw)
            output_bytes += len(raw)

    if len(files) + 6 > MAX_OUTPUT_FILES or sum(map(len, files.values())) + RESERVE_BYTES > MAX_OUTPUT_BYTES:
        raise ValueError("Static-support candidate lacks bounded finalization capacity")
    guard()
    target.mkdir(parents=True, exist_ok=False)
    try:
        write(files)
        _check(deadline)
        feasibility = audit_shared_world_feasibility(scene)
        _check(deadline)
        source_check = verify_shared_world_sources(scene, target)
        _check(deadline)
        interval = {"mode": "complete_reference", "selected": task["interval"],
            "first_reference_time_s": human["reference"][0]["time_s"], "last_reference_time_s": human["reference"][-1]["time_s"],
            "dt": scene["dt"], "substeps": scene["substeps"],
            "scope": "Complete retained reference interval; missing observations and original decoder timestamps remain unchanged."}
        report = {"schema_version": 1, "profile": PROFILE, "status": "ready", "created_at": local_time(),
            "mode": "source_body_authored_static_support", "construction_profile": CONFIG_PROFILE,
            "physical_validation": "unverified", "lab_status": "not_created",
            "readiness": "candidate_constructed_not_lab_admitted", "simulation_ready": False,
            "scene_sha256": canonical_hash(scene), "task_sha256": canonical_hash(task),
            "task_file_sha256": _sha(task_raw), "source_sha256": config["source_sha256"],
            "source_manifest_sha256": _sha(manifest_raw), "config_sha256": _sha(config_raw),
            "config_input_kind": "exact_file_bytes" if config_path is not None else "serialized_json_mapping",
            "source_verification": {"scope": "Every declared body-source member was verified and copied unchanged; unlisted files were not admitted.",
                "verified_files": len(declared), "verified_bytes": total - len(config_raw), "preserved_files": dict(declared)},
            "human_file_sha256": declared["human.json"], "support_scope": task["source_scope"],
            "root_assistance": task["control"]["root_assistance"],
            "source_binding_verification": source_check, "reference_interval": interval,
            "reference_feasibility": {key: feasibility[key] for key in ("status", "summary")},
            "runtime_sha256": {name: _sha(raw) for name, raw in implementations.items()},
            "runtime_versions": {"python": platform.python_version(), **{name: version(name) for name in ("numpy", "mujoco", "gymnasium")}},
            "work": {"physics_steps": 0, "inference_calls": 0, "video_decodes": 0, "training_episodes": 0, "api_calls": 0,
                "native_work": "Existing model compilation and reference/ground/static forward checks only; no dynamics"},
            "bounds": {"maximum_seconds": max_seconds, "maximum_bytes": MAX_OUTPUT_BYTES, "maximum_files": MAX_OUTPUT_FILES,
                "maximum_file_bytes": MAX_FILE_BYTES, "deadline_scope": "Cooperative bounded native calls; caller must enforce a hard process limit"},
            "scope": "Source-bound body plus authored static supports; not recovered equipment, verified contact/grasp, physical accuracy or task success."}
        write({"reference-feasibility.json": shared_world_human._bytes(feasibility),
               "source-binding-verification.json": shared_world_human._bytes(source_check),
               "reconstruction.json": shared_world_human._bytes(report)})
        actual, _ = exercise_body_source._inventory(target, deadline)
        if actual != owned:
            raise ValueError("Static-support output differs from its complete owned artifact set")
        guard()
        seal = shared_world_human._bytes({"schema_version": 1, "profile": PROFILE, "status": "ready",
            "simulation_ready": False, "files": owned}, RESERVE_BYTES)
        write({"manifest.json": seal}, reserve=0)
        return {"out": str(target), "status": "ready", "profile": PROFILE, "construction_profile": CONFIG_PROFILE,
            "scene_sha256": report["scene_sha256"], "task_sha256": report["task_sha256"], "task_file_sha256": _sha(task_raw),
            "exercise_task_path": str(target / "exercise-task.json"), "source_sha256": config["source_sha256"],
            "source_manifest_sha256": _sha(manifest_raw), "config_sha256": _sha(config_raw), "manifest_sha256": _sha(seal),
            "readiness": report["readiness"], "lab_status": "not_created", "simulation_ready": False,
            "reference_interval": interval, "source_documents": 1, "reference_feasibility": report["reference_feasibility"],
            "scope": report["scope"]}
    except Exception as error:
        failure = shared_world_human._bytes({"schema_version": 1, "profile": PROFILE, "status": "failed", "simulation_ready": False,
            "error": {"type": type(error).__name__, "message": str(error)[:2000]}, "elapsed_seconds": time.monotonic() - started}, RESERVE_BYTES // 2)
        try:
            write_package_files(target, {"failure.json": failure})
            owned["failure.json"] = _sha(failure)
            if not (target / "manifest.json").exists():
                write_package_files(target, {"manifest.json": shared_world_human._bytes({"schema_version": 1, "profile": PROFILE,
                    "status": "failed", "simulation_ready": False, "files": owned,
                    "scope": "Completed writes and failure receipt only; any partial/unexpected files remain preserved but unadmitted."}, RESERVE_BYTES // 2)})
        except (OSError, ValueError) as receipt_error:
            error.add_note(f"Static-support failure receipt unavailable: {type(receipt_error).__name__}")
        raise
