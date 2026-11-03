"""Construct a source-bound person/floor world without temporary rigid props."""

import hashlib
from importlib.metadata import version
import json
import math
from pathlib import Path
import platform
import re
import time

import numpy as np

from weft import exercise_body_source
from weft.io import canonical_hash, local_time
from weft.mobile_human import ground_normal
from weft.person_inventory import CONDITIONAL_PROFILE, validate_person_inventory
from weft.shared_world import (
    HUMAN_ONLY_PROFILE, audit_shared_world_feasibility, full_reference_horizon, validate_shared_world,
)
from weft.shared_world_evidence import build_shared_world_binding, verify_shared_world_sources
from weft.shared_world_packaging import read_package_file, read_package_manifest, write_package_files
from weft.whole_body import SUCCESS_LIMITS
from weft.world_evidence import _check_tree, _nonfinite, _unique_object


PROFILE = HUMAN_ONLY_PROFILE
MAX_SECONDS = 30
MAX_OUTPUT_BYTES = MAX_INPUT_BYTES = 128 * 1024 * 1024
MAX_OUTPUT_FILES = MAX_INPUT_FILES = 512
MAX_FILE_BYTES = 64 * 1024 * 1024
MAX_JSON_BYTES = 8 * 1024 * 1024
RESERVE_BYTES = 256 * 1024
IMPLEMENTATION_FILES = tuple(dict.fromkeys((
    "shared_world_human.py", "person_inventory.py", *exercise_body_source.IMPLEMENTATION_FILES,
    "shared_world.py", "shared_world_observation.py", "shared_world_evidence.py",
    "source_equivalence.py", "whole_body.py", "spatial_rigid.py",
)))
_CONFLICTS = frozenset({"scene.json", "task.json", "exercise-task.json", "construction-seed.json",
    "geometry-proposal.json", "scene-observations.json", "lab-config.json", "policy.json"})
_OPTIONAL = exercise_body_source.OPTIONAL_FILES | {"root-motion.json", "floor-assumption.json"}


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _bytes(value, maximum=MAX_JSON_BYTES):
    output = bytearray()
    for part in json.JSONEncoder(indent=2, sort_keys=True, allow_nan=False).iterencode(value):
        raw = part.encode()
        if len(output) + len(raw) + 1 > maximum:
            raise ValueError("Person-only metadata exceeds its bounded output reservation")
        output.extend(raw)
    output.extend(b"\n")
    return bytes(output)


def _json(raw):
    if len(raw) > MAX_JSON_BYTES:
        raise ValueError("Person-only JSON exceeds its input bound")
    try:
        value = json.loads(raw, object_pairs_hook=_unique_object, parse_constant=_nonfinite)
        _check_tree(value)
    except (UnicodeError, json.JSONDecodeError, RecursionError) as error:
        raise ValueError("Person-only inputs require bounded strict JSON") from error
    if not isinstance(value, dict):
        raise ValueError("Person-only input must be a JSON object")
    return value


def _path(value):
    path = Path(value).absolute()
    if ".." in path.parts or any(part.is_symlink() for part in (path, *path.parents)):
        raise ValueError("Person-only paths must not contain traversal or symlinks")
    return path


def _check(deadline):
    if time.monotonic() >= deadline:
        raise TimeoutError("Person-only construction exceeded its cooperative deadline")


def _not_failed(root):
    if (root / "failure.json").exists() or (root / "failure.json").is_symlink():
        raise ValueError("A body-source failure marker prevents person-only construction")


def read_person_body_source_identity(body_source, *, max_seconds=5, deadline=None):
    """Read sealed identity metadata only; constructor admission still verifies body/media bytes."""
    started = time.monotonic()
    if type(max_seconds) not in (int, float) or not math.isfinite(max_seconds) or not 1 <= max_seconds <= 5:
        raise ValueError("Person-only identity inspection requires a 1–5 second bound")
    if deadline is not None and (type(deadline) not in (int, float) or not math.isfinite(deadline)):
        raise ValueError("Person-only identity deadline must be a finite monotonic time")
    deadline = min(started + max_seconds, deadline) if deadline is not None else started + max_seconds
    root = _path(body_source)
    _check(deadline)
    _not_failed(root)
    before = root.stat().st_dev, root.stat().st_ino
    manifest_raw, declared = read_package_manifest(root)
    manifest = _json(manifest_raw)
    if (manifest.get("profile") != exercise_body_source.PROFILE or manifest.get("status") != "construction_only"
            or manifest.get("simulation_ready") is not False or len(declared) + 1 > MAX_INPUT_FILES
            or "failure.json" in declared or _CONFLICTS & declared.keys()
            or not {"body-source.json", "source.json"} <= declared.keys()):
        raise ValueError("Person-only identity requires a sealed construction-only body source")
    pins = {"manifest.json": manifest_raw}
    for name in ("body-source.json", "source.json"):
        _check(deadline)
        pins[name] = read_package_file(root, name, max_bytes=MAX_JSON_BYTES, expected_sha256=declared[name])
    marker, source = _json(pins["body-source.json"]), _json(pins["source.json"])
    artifact, source_sha = marker.get("source_artifact"), marker.get("source_sha256")
    if (type(marker.get("schema_version")) is not int or marker["schema_version"] != 1
            or marker.get("profile") != exercise_body_source.PROFILE or marker.get("status") != "construction_only"
            or marker.get("simulation_ready") is not False
            or not isinstance(source_sha, str) or not re.fullmatch(r"[0-9a-f]{64}", source_sha)
            or not isinstance(artifact, str) or artifact not in {"source" + suffix for suffix in exercise_body_source.SUFFIXES}
            or declared.get(artifact) != source_sha or source.get("sha256") != source_sha
            or source.get("artifact") != artifact):
        raise ValueError("Person-only body source has conflicting sealed source identities")
    for name, raw in pins.items():
        _check(deadline)
        read_package_file(root, name, max_bytes=max(1, len(raw)), expected_sha256=_sha(raw))
    _not_failed(root)
    if _path(root) != root or (root.stat().st_dev, root.stat().st_ino) != before:
        raise ValueError("Body identity directory changed during inspection")
    _check(deadline)
    return {"profile": exercise_body_source.PROFILE, "status": "construction_only", "source_sha256": source_sha,
            "source_artifact": artifact, "manifest_sha256": _sha(manifest_raw),
            "scope": "Sealed source identity metadata verified; full body and original media bytes are verified by construction admission."}


def _scene(human, observations, inventory, binding, scale):
    if human.get("native_clock") != observations["source"].get("native_clock_summary"):
        raise ValueError("Body and observations must retain the same native source clock")
    interval = None
    if "native_clock" in human:
        from weft.native_interval import native_interval
        interval = native_interval(human, 1 / 30)
    # This is the existing body-scale placement prior, without the legacy ball/table scaffold.
    roots = np.asarray([row["translation"] for row in human["root_reference"]], dtype=float) * scale
    span = np.maximum(.4, np.ptp(roots[:, :2], axis=0))
    bx, by = round(max(2., 1.8 * scale + span[0]), 3), round(max(2.2, 2. * scale + span[1]), 3)
    bz = round(max(2.4, 2.3 * scale), 3)
    origin = [round(bx * .5 - float(roots[:, 0].mean()), 4),
              round(by * .34 - float(roots[:, 1].mean()), 4), 0.]
    origin[0], origin[1] = float(np.clip(origin[0], .45, bx - .45)), float(np.clip(origin[1], .4, by - .85))
    gravity = (-9.81 * ground_normal(human)).tolist()
    return {
        "schema_version": 1, "profile": PROFILE, "physical_validation": "unverified",
        "learned_policy_available": False, "dt": 1 / 30, "substeps": 16,
        "bounds": [float(bx), float(by), float(bz)], "gravity": gravity,
        "contact": {"time_constant": .012, "damping_ratio": 1.},
        "coordinate_frame": {
            "axes": "x_right_y_depth_z_up",
            "units": {"length": "body_scaled_world_units", "mass": "kilogram_prior", "time": "second",
                      "orientation": "quaternion_wxyz", "angular_velocity": "world_radians_per_second"},
            "calibration_status": "estimated_prior",
            "scale_bridge": {"body_height_to_world_scale": scale, "origin_world": origin,
                "method": "monocular_body_height_world_prior_v1",
                "uncertainty_note": "Monocular body/root estimates use an explicit scale prior; camera depth and metric scale are not calibrated."},
            "clock_binding": {"control_dt_s": 1 / 30, "physics_substeps": 16,
                "source_fps": observations["source"]["fps"],
                "source_time_offset_s": human["reference"][0]["time_s"],
                "alignment_method": "native_pts_elapsed_time_v1" if "native_clock" in human else "pose_frame_index_over_fps_prior_v1",
                **({"native_clock": human["native_clock"]} if "native_clock" in human else {})}},
        "source_binding": binding, "inventory": inventory, "human_spec": human, "objects": [],
        "controlled_id": "human", "target": None,
        "task": {"type": "human_motion_v1", "authored_addition": True, "start_offset_s": 0.,
                 "horizon_steps": interval["full_control_steps"] + 1 if interval else full_reference_horizon(human["reference"], 1 / 30),
                 **({"native_interval": interval} if interval else {}),
                 "human_limits": dict(SUCCESS_LIMITS), "object_goal": None},
        "assumptions": [
            ("Conditional body/ground approximation: original scene inventory is retained; equipment, obstacles, other people and their contacts are omitted, not absent."
             if inventory.get("profile") == CONDITIONAL_PROFILE else
             "Person-only scope is an explicit user declaration, not verified absence of equipment or obstacles."),
            "One articulated human and the authored ground plane share one MuJoCo world; no rigid prop is introduced.",
            "Ground, mass, friction, scale and monocular depth remain priors; physical accuracy is unverified.",
            ("Native presentation timing drives derived body references; original detector clocks and missing/unsampled source slots remain preserved."
             if "native_clock" in human else
             "Original body references, missing/unsampled source slots and decoder timestamps are preserved without retiming."),
            "Motor-command muscle bands are illustrative, not measured muscle activation or recovered anatomy.",
            "No learned controller, unassisted balance, grasp or physical exercise success is established by construction.",
        ],
    }


def construct_human_only_run(body_source, inventory, out, *, scale=1.0, deadline=None, max_seconds=30):
    """Seal a verified person-only candidate; model/forward checks do not advance dynamics."""
    started = time.monotonic()
    if type(max_seconds) not in (int, float) or not math.isfinite(max_seconds) or not 1 <= max_seconds <= MAX_SECONDS:
        raise ValueError("Person-only construction requires a 1–30 second bound")
    if deadline is not None and (type(deadline) not in (int, float) or not math.isfinite(deadline)):
        raise ValueError("Person-only deadline must be a finite monotonic time")
    deadline = min(started + max_seconds, deadline) if deadline is not None else started + max_seconds
    if type(scale) not in (int, float) or not math.isfinite(scale) or not .5 <= scale <= 2.:
        raise ValueError("Person-only scale prior must be within [0.5, 2.0]")
    source, target = _path(body_source), _path(out)
    if target.exists() or target.is_relative_to(source) or source.is_relative_to(target):
        raise ValueError("Person-only output must be new and outside its immutable body source")
    _check(deadline)
    _not_failed(source)
    source_identity = source.stat().st_dev, source.stat().st_ino
    manifest_raw, declared = read_package_manifest(source)
    manifest = _json(manifest_raw)
    if (manifest.get("profile") != exercise_body_source.PROFILE or manifest.get("status") != "construction_only"
            or manifest.get("simulation_ready") is not False or len(declared) + 1 > MAX_INPUT_FILES
            or "failure.json" in declared or _CONFLICTS & declared.keys()):
        raise ValueError("Person-only construction requires an unrejected construction-only body source without scene/equipment/task artifacts")
    inputs, total = {}, len(manifest_raw)
    for name, expected in declared.items():
        _check(deadline)
        if (any(part.startswith(".") or part == "__pycache__" for part in Path(name).parts)
                or name.endswith(".pyc") or name.endswith(".log") and name != "native-clock-decode.log"):
            raise ValueError("Body-source inputs must exclude hidden files, caches and logs")
        raw = read_package_file(source, name, max_bytes=min(MAX_FILE_BYTES, max(1, MAX_INPUT_BYTES - total)),
                                expected_sha256=expected)
        total += len(raw)
        if total > MAX_INPUT_BYTES:
            raise ValueError("Person-only input exceeds its 128-MiB bound")
        if name in {"status.json", "report.json", "reconstruction.json"} and _json(raw).get("status") in exercise_body_source.REJECTED:
            raise ValueError("Rejected body-source artifacts cannot become a person-only world")
        inputs[name] = raw
    marker = _json(inputs.get("body-source.json", b"{}"))
    inventory = validate_person_inventory(inventory, source_sha256=marker.get("source_sha256"))
    implementation_names = list(IMPLEMENTATION_FILES)
    if "native-clock.json" in inputs:
        implementation_names.append("source_clock.py")
        implementation_names.append("native_interval.py")
    if exercise_body_source.CAPTURE_FILES & inputs.keys():
        implementation_names.append("source_clock_capture.py")
    proof_names = set()
    if "pose-recovery.json" in declared:
        from weft.pose_source_recovery import IMPLEMENTATION_FILES as RECOVERY_IMPLEMENTATION, PROOF_FILES
        implementation_names.extend(RECOVERY_IMPLEMENTATION)
        proof_names.update(PROOF_FILES)
    if "pose-capture-recovery.json" in declared:
        from weft.pose_capture_recovery import PROOF_FILES
        implementation_names.append("pose_capture_recovery.py")
        proof_names.update(PROOF_FILES)
    implementations = {}
    for name in dict.fromkeys(implementation_names):
        _check(deadline)
        implementations[name] = read_package_file(Path(__file__).parent, name, max_bytes=2 * 1024 * 1024)
    human, observations = exercise_body_source.validate_body_source_files(inputs, deadline=deadline)
    if (len(observations["objects"]) != 1 or observations["objects"][0]["id"] != "person-0"
            or observations["objects"][0]["properties"]["kind"]["value"] != "person" or observations["relationships"]):
        raise ValueError("Body observations contradict a single person with no external equipment/support declaration")
    observation_raw = inputs["observations.json"]
    observation_sha = _sha(observation_raw)
    binding, construction = build_shared_world_binding({observation_sha: observation_raw},
        source_sha256=inventory["source_sha256"], world_ids=["human"],
        entity_refs={"human": (observation_sha, "person-0")}, deadline=deadline)
    from weft.shared_human_projection import compact_scene_if_needed, validate_projection
    projection_implementation = read_package_file(Path(__file__).parent, "shared_human_projection.py", max_bytes=1 << 20)
    scene = validate_shared_world(compact_scene_if_needed(
        _scene(human, observations, inventory, binding, float(scale)), inputs["human.json"]))
    if "human_provenance" in scene:
        validate_projection(scene["human_spec"], scene["human_provenance"], inputs["human.json"])
        implementations["shared_human_projection.py"] = projection_implementation
    elif canonical_hash(scene["human_spec"]) != canonical_hash(_json(inputs["human.json"])):
        raise ValueError("Person-only construction cannot rewrite the original body")
    required = exercise_body_source.POSE_FILES | proof_names | {
        "body-source.json", "human.json", "observations.json", "source-clock.json", "implementation.json",
        marker["source_artifact"], "inputs/source-manifest.json"}
    if "native_clock" in human:
        required |= {"native-clock.json", "reconstruction-evidence.json"}
        if marker.get("construction", {}).get("native_clock_capture"):
            required |= exercise_body_source.CAPTURE_FILES
    kept = {name: raw for name, raw in inputs.items() if name in required | _OPTIONAL or name.startswith("implementation-source/")}
    files = {**kept, "inputs/body-source-manifest.json": manifest_raw,
        binding["evidence_documents"][observation_sha]: observation_raw,
        "construction-evidence.json": construction, "inventory.json": _bytes(inventory),
        "scene.json": _bytes(scene), "task.json": _bytes(scene["task"]),
        **{"implementation/" + name: raw for name, raw in implementations.items()}}
    owned, output_bytes = {}, 0

    def guard():
        _check(deadline)
        if _path(source) != source or (source.stat().st_dev, source.stat().st_ino) != source_identity:
            raise ValueError("Person-only body source directory changed")
        _not_failed(source)
        read_package_file(source, "manifest.json", max_bytes=MAX_JSON_BYTES, expected_sha256=_sha(manifest_raw))
        for name, raw in inputs.items():
            _check(deadline)
            read_package_file(source, name, max_bytes=max(1, len(raw)), expected_sha256=declared[name])
        for name, raw in implementations.items():
            _check(deadline)
            read_package_file(Path(__file__).parent, name, max_bytes=max(1, len(raw)), expected_sha256=_sha(raw))

    def write(batch, *, reserve=RESERVE_BYTES):
        nonlocal output_bytes
        if (len(owned) + len(batch) + 2 > MAX_OUTPUT_FILES
                or output_bytes + sum(map(len, batch.values())) + reserve > MAX_OUTPUT_BYTES
                or any(len(raw) > MAX_FILE_BYTES for raw in batch.values())):
            raise ValueError("Person-only output exceeds its 512-file/128-MiB reservation")
        for name, raw in batch.items():
            _check(deadline)
            write_package_files(target, {name: raw})
            owned[name] = _sha(raw)
            output_bytes += len(raw)

    # Reserve before creating output; subsequent failures retain a terminal marker and known file seals.
    if len(files) + 6 > MAX_OUTPUT_FILES or sum(map(len, files.values())) + RESERVE_BYTES > MAX_OUTPUT_BYTES:
        raise ValueError("Person-only candidate lacks bounded finalization capacity")
    guard()
    target.mkdir(parents=True, exist_ok=False)
    try:
        write(files)
        _check(deadline)
        feasibility = audit_shared_world_feasibility(scene)
        _check(deadline)
        source_check = verify_shared_world_sources(scene, target)
        status = "ready"
        interval = {"mode": "complete_reference", "selected": {key: scene["task"][key] for key in ("start_offset_s", "horizon_steps")},
            "first_reference_time_s": human["reference"][0]["time_s"], "last_reference_time_s": human["reference"][-1]["time_s"],
            "dt": scene["dt"], "substeps": scene["substeps"],
            **({"native_interval": scene["task"]["native_interval"]} if "native_interval" in scene["task"] else {}),
            "scope": "Complete retained reference interval; missing source observations remain missing, with no native-clock accuracy claim."}
        report = {"schema_version": 1, "profile": PROFILE, "status": status, "created_at": local_time(),
            "mode": "conditional_body_ground" if inventory.get("profile") == CONDITIONAL_PROFILE else "source_person_only",
            "physical_validation": "unverified", "lab_status": "not_created",
            "readiness": "candidate_constructed_not_lab_admitted", "simulation_ready": False,
            "scene_sha256": canonical_hash(scene), "source_sha256": inventory["source_sha256"],
            "source_manifest_sha256": _sha(manifest_raw), "source_verification": {
                "scope": "Every declared input member was verified; unlisted files were not admitted.",
                "verified_files": len(declared), "verified_bytes": total,
                "preserved_files": {name: declared[name] for name in kept}},
            "human_file_sha256": declared["human.json"], "inventory": inventory,
            "source_binding_verification": source_check, "reference_interval": interval,
            "reference_feasibility": {key: feasibility[key] for key in ("status", "summary")},
            "runtime_sha256": {name: _sha(raw) for name, raw in implementations.items()},
            "runtime_versions": {"python": platform.python_version(), **{name: version(name) for name in ("numpy", "mujoco", "gymnasium")}},
            "work": {"physics_steps": 0, "inference_calls": 0, "video_decodes": 0, "training_episodes": 0,
                     "api_calls": 0, "native_work": "Existing model compilation and reference/ground forward checks only"},
            "bounds": {"maximum_seconds": max_seconds, "maximum_bytes": MAX_OUTPUT_BYTES,
                       "maximum_files": MAX_OUTPUT_FILES, "maximum_file_bytes": MAX_FILE_BYTES,
                       "deadline_scope": "Cooperative bounded native calls; no hard process interruption"},
            "scope": ("Authored body/ground approximation with original inventory retained; omitted scene contacts are unsupported, not verified absent. No recovered biomechanics or physical exercise success."
                      if inventory.get("profile") == CONDITIONAL_PROFILE else
                      "Declared person-only simulator candidate, not verified equipment absence, recovered biomechanics or physical exercise success.")}
        write({"reference-feasibility.json": _bytes(feasibility), "source-binding-verification.json": _bytes(source_check),
               "reconstruction.json": _bytes(report)})
        for name, expected in owned.items():
            _check(deadline)
            read_package_file(target, name, max_bytes=MAX_FILE_BYTES, expected_sha256=expected)
        guard()
        seal = _bytes({"schema_version": 1, "profile": PROFILE, "status": "ready" if status == "ready" else "rejected",
                       "simulation_ready": False, "files": owned}, RESERVE_BYTES)
        write({"manifest.json": seal}, reserve=0)
        return {"out": str(target), "status": status, "profile": PROFILE, "scene_sha256": report["scene_sha256"],
            "readiness": report["readiness"], "lab_status": "not_created", "simulation_ready": False,
            "manifest_sha256": _sha(seal), "reference_interval": interval, "source_documents": 1,
            "reference_feasibility": report["reference_feasibility"], "scope": report["scope"]}
    except Exception as error:
        failure = _bytes({"schema_version": 1, "profile": PROFILE, "status": "failed", "simulation_ready": False,
            "error": {"type": type(error).__name__, "message": str(error)[:2000]},
            "elapsed_seconds": time.monotonic() - started}, RESERVE_BYTES // 2)
        try:
            write_package_files(target, {"failure.json": failure})
            owned["failure.json"] = _sha(failure)
            if not (target / "manifest.json").exists():
                write_package_files(target, {"manifest.json": _bytes({"schema_version": 1, "profile": PROFILE,
                    "status": "failed", "simulation_ready": False, "files": owned}, RESERVE_BYTES // 2)})
        except (OSError, ValueError) as receipt_error:
            error.add_note(f"Person-only failure receipt unavailable: {type(receipt_error).__name__}")
        raise
