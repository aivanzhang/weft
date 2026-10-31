"""Package unchanged two-view camera bodies for shared-world-v6; no simulation run."""

from collections import Counter
from copy import deepcopy
from hashlib import sha256
from importlib.metadata import version
import json
import math
import os
from pathlib import Path
import re
import stat
import sys
import tempfile
import time

from weft.camera_ground import validate_ground
from weft.camera_shared_contract import (
    CALIBRATION_STATUS, LENGTH_UNIT, MASS_UNIT, build_camera_coupling, validate_camera_coupling,
)
from weft.io import canonical_hash, local_time
from weft.multiview_pose import CONTRACT as RAY_CONTRACT, _view
from weft.observations import adapt_exercise, validate_observations
from weft.shared_world_evidence import build_shared_world_binding
from weft.shared_world_packaging import read_package_file, validate_package_layout, write_package_files


PROFILE = "camera-shared-candidate-v1"
MAX_SECONDS = 30.0
MAX_MANIFEST_BYTES, MAX_MANIFEST_FILES = 256 << 10, 512
MAX_JSON_BYTES, MAX_MEDIA_BYTES, MAX_RUNTIME_BYTES = 32 << 20, 64 << 20, 8 << 20
MAX_MEMBER_BYTES, MAX_PACKAGE_BYTES, MAX_FILES = 8 << 20, 128 << 20, 80
RESERVE_BYTES = 256 << 10
RUNTIME_ROOT = Path(__file__).parent
RUNTIME_FILES = (
    "__init__.py", "camera_shared_candidate.py", "camera_shared_contract.py", "camera_body.py",
    "camera_ground.py", "camera_pose.py", "multiview_root.py", "multiview_body.py", "multiview_pose.py",
    "observations.py", "world_evidence.py", "shared_world_evidence.py", "shared_world_packaging.py",
    "source_equivalence.py", "shared_world.py", "shared_world_observation.py", "human.py", "mobile_human.py",
    "spatial_pose.py", "spatial_rigid.py", "body_frame.py", "whole_body.py", "io.py",
    "multiview_segments.py", "root_motion.py", "errors.py",
)
REQUIRED = (
    "human.json", "evidence.json", "camera.json", "ground.json", "two-view-triangulation.json",
    "two-view-root.json", "inputs/two-view-root/evidence.json", "inputs/two-view-root/camera.json",
    "inputs/two-view-root/config.json", "source.json", "source-input-manifest.json", "camera-provenance.json",
    "reference-feasibility.json", "ground-clearance.json", "scene.json", "task.json",
)
OPTIONAL = ("background.jpg",)
SCOPE = {
    "physical_validation": "unverified", "simulation_run": False, "lab_created": False,
    "independent_task_adapter": "unsupported", "learned_policy": "unsupported", "feet": "unsupported",
    "equipment_inventory": "unknown_not_inferred_from_empty_objects", "contact_load_grasp": "unknown",
    "secondary_media_and_rights": "unavailable_in_this_input_contract",
    "source_qualification": "saved_pose_track_and_supplied_camera_ground_priors_not_physical_accuracy",
}
_SHA = re.compile(r"[0-9a-f]{64}\Z")


def _require(value, message):
    if not value:
        raise ValueError("Camera shared candidate: " + message)


def _check(deadline):
    if deadline is not None and time.monotonic() > deadline:
        raise TimeoutError("Camera shared candidate cooperative deadline exceeded")


def _path(value):
    path = Path(value).absolute()
    _require(len(path.parts) <= 64 and ".." not in path.parts
             and not any(p.is_symlink() for p in (path, *path.parents)), "paths cannot traverse or follow symlinks")
    return path


def _identity(path):
    info = path.stat()
    _require(stat.S_ISDIR(info.st_mode), "package root must be a directory")
    return info.st_dev, info.st_ino


def _unique(pairs):
    result = {}
    for key, value in pairs:
        _require(key not in result, "duplicate JSON key")
        result[key] = value
    return result


def _constant(_value):
    raise ValueError("Camera shared candidate requires finite JSON")


def _json(raw):
    try:
        value = json.loads(raw, object_pairs_hook=_unique, parse_constant=_constant)
    except (UnicodeError, json.JSONDecodeError, RecursionError) as error:
        raise ValueError("Camera shared candidate requires bounded strict JSON") from error
    _require(type(value) is dict, "source document must be an object")
    return value


def _encoded(value, limit=MAX_MEMBER_BYTES):
    parts, size = [], 0
    for chunk in json.JSONEncoder(sort_keys=True, separators=(",", ":"), allow_nan=False).iterencode(value):
        raw = chunk.encode()
        size += len(raw)
        _require(size + 1 <= limit, "generated document exceeds its byte bound")
        parts.append(raw)
    return b"".join(parts) + b"\n"


def _manifest(source):
    raw = read_package_file(source, "manifest.json", max_bytes=MAX_MANIFEST_BYTES)
    value = _json(raw)
    files = value.get("files")
    _require(type(files) is dict and 1 <= len(files) <= MAX_MANIFEST_FILES, "source manifest needs 1–512 members")
    validate_package_layout([*files, "manifest.json"])
    _require("manifest.json" not in files and all(type(v) is str and _SHA.fullmatch(v) for v in files.values()),
             "source manifest needs exact member SHA256 values")
    _require(value.get("status") not in ("failed", "rejected", "cancelled")
             and "failure.json" not in files and not (source / "failure.json").exists()
             and not (source / "failure.json").is_symlink(), "failed source package cannot be imported")
    _require(all(name in files for name in REQUIRED), "sealed camera-body source members are missing")
    return raw, files


def _runtime(deadline):
    files, size = {}, 0
    for name in RUNTIME_FILES:
        _check(deadline)
        raw = read_package_file(_path(RUNTIME_ROOT), name, max_bytes=min(2 << 20, MAX_RUNTIME_BYTES - size))
        size += len(raw)
        files["candidate-implementation/" + name] = raw
    project = _path(RUNTIME_ROOT).parent.parent
    for name in ("pyproject.toml", "requirements-lab.lock"):
        raw = read_package_file(project, name, max_bytes=min(1 << 20, MAX_RUNTIME_BYTES - size))
        size += len(raw)
        files["candidate-implementation/project/" + name] = raw
    _check(deadline)
    return files, {"python": sys.version.split()[0], **{name: version(name) for name in ("numpy", "mujoco", "gymnasium")}}


def _qualified_inputs(inputs, deadline):
    data = {name: _json(raw) for name, raw in inputs.items() if name.endswith(".json")}
    body, source = data["human.json"], data["source.json"]
    coupling = build_camera_coupling(body)
    binding = body["camera_reference"]
    root, prediction, ground = binding["root_refinement"], data["two-view-triangulation.json"], data["ground.json"]
    validate_ground(ground, binding)
    _require(canonical_hash(ground) == canonical_hash(binding["ground"]), "ground differs from exact body ground")
    _require(prediction.get("schema_version") == 1 and type(prediction.get("schema_version")) is int
             and canonical_hash(prediction.get("contract")) == canonical_hash(dict(RAY_CONTRACT))
             and prediction.get("monocular_world_landmarks_used") is False
             and prediction.get("reference_body_or_contacts_used") is False
             and prediction.get("physical_validation") == "unverified"
             and canonical_hash(prediction) == root["triangulation_sha256"]
             and prediction.get("views") == root["views"] and prediction.get("source_frames") == root["source_frames"],
             "saved triangulation differs from the original two-view body binding")
    report = data["two-view-root.json"]
    _require(report.get("binding") == root and report.get("source_frames") == root["source_frames"]
             and report.get("method") == binding["method"], "root report differs from the exact body")
    count, rows = root["source_frames"], prediction.get("frames")
    _require(type(rows) is list and len(rows) == count and prediction.get("joint_slots") == 33 * count,
             "triangulation must retain every original source slot")
    config = data["inputs/two-view-root/config.json"]
    _require(config.get("schema_version") == 1 and type(config.get("schema_version")) is int
             and config.get("first_name") == root["views"][0]["name"]
             and config.get("second_name") == root["views"][1]["name"]
             and config.get("world_provenance") == root["world_provenance"]
             and config.get("clock_provenance") == root["views"][0]["clock_provenance"],
             "two-view config must preserve original identities and assumed alignment")
    evidence_names = ("evidence.json", "inputs/two-view-root/evidence.json")
    camera_names = ("camera.json", "inputs/two-view-root/camera.json")
    poses, views = [], []
    for view, ename, cname in zip(root["views"], evidence_names, camera_names, strict=True):
        evidence, camera = data[ename], data[cname]
        _require(canonical_hash(evidence) == view["evidence_sha256"] and canonical_hash(camera) == view["camera_sha256"]
                 and evidence.get("source_sha256") == view["source_sha256"]
                 and evidence.get("frame_count") == count and evidence.get("decoded_frames", count) == count
                 and evidence.get("fps") == root["fps"] and camera.get("quality") == view["calibration_quality"],
                 "view evidence, camera, source or original clock differs")
        indexed, _ = _view({"name": view["name"], "evidence": evidence, "camera": camera,
                            "clock_provenance": view["clock_provenance"]})
        poses.append(indexed)
        views.append({**view, "evidence_member": ename, "evidence_bytes_sha256": sha256(inputs[ename]).hexdigest(),
                      "camera_member": cname, "camera_bytes_sha256": sha256(inputs[cname]).hexdigest(),
                      "metadata_frames": evidence["frame_count"], "decoded_frames": count, "pose_slots": len(indexed),
                      "pose_status_counts": dict(Counter(row.get("status", "unknown") for row in indexed.values())),
                      "unsampled_slots": count - len(indexed)})
        _check(deadline)
    source_slots = []
    roots = {row["frame"]: row for row in root["roots"]}
    original_slots = report.get("source_slots")
    _require(type(original_slots) is list and len(original_slots) == count, "root report must preserve all source slots")
    for index, (row, slot) in enumerate(zip(rows, original_slots, strict=True)):
        _check(deadline)
        ids = [pose[index]["id"] if index in pose else None for pose in poses]
        _require(type(row) is dict and type(row.get("frame")) is int and row["frame"] == index
                 and type(row.get("time_s")) in (int, float) and abs(row["time_s"] - index / root["fps"]) <= 1e-7
                 and row.get("source_evidence_ids") == ids and type(row.get("joints")) is list
                 and len(row["joints"]) == 33 and all(type(joint) is dict and type(joint.get("status")) is str
                     and 1 <= len(joint["status"]) <= 100 for joint in row["joints"]),
                 "triangulation source slots, endpoint statuses or clocks differ")
        status = ("retained_body_pose" if index in roots else "primary_pose_not_sampled" if ids[0] is None
                  else "primary_body_pose_unavailable")
        hips = [row["joints"][i] for i in (23, 24)]
        _require(type(slot) is dict and slot.get("frame") == index and slot.get("time_s") == row["time_s"]
                 and slot.get("status") == status and slot.get("hip_statuses") == [hip["status"] for hip in hips],
                 "root report source status differs from retained triangulation")
        if index in roots:
            _require(all(hip["status"] == "estimated" for hip in hips)
                     and [hip.get("world_point_m") for hip in hips] == roots[index]["hip_world_m"],
                     "retained root hips differ from saved triangulation")
        source_slots.append({"frame": index, "time_s": row["time_s"], "body_status": status,
                             "views": [{"source_evidence_id": ident,
                                        "pose_status": pose[index].get("status", "unknown") if ident else "not_sampled",
                                        "decoder_timestamp_s": pose[index].get("decoder_timestamp_s") if ident else None}
                                       for pose, ident in zip(poses, ids, strict=True)]})
    _require(source.get("sha256") == body["source_sha256"] and type(source.get("rights")) is str
             and 1 <= len(source["rights"].strip()) <= 8000, "primary source rights and identity must remain explicit")
    return body, coupling, data["evidence.json"], views, source_slots


def _observations(evidence, evidence_raw):
    document = adapt_exercise(evidence)
    # Pose interchange's generic fixed-camera prior is not evidence for these moving-camera sources.
    document["camera"]["motion"] = {"value": None, "status": "unknown", "evidence_ids": []}
    document["provenance"]["input_artifacts"] = {"evidence.json": sha256(evidence_raw).hexdigest()}
    document["limits"].append("Camera-body importer does not infer a fixed camera, equipment absence, contact or person identity beyond the saved track.")
    raw = _encoded(validate_observations(document))
    return raw, sha256(raw).hexdigest()


def _shared_api():
    # Native-capable schema validation stays behind this single mockable boundary; no environment is built.
    from weft.shared_world import SUCCESS_LIMITS, full_reference_horizon, validate_shared_world
    return SUCCESS_LIMITS, full_reference_horizon, validate_shared_world


def _scene(body, coupling, source_binding):
    limits, horizon, validate = _shared_api()
    dt = 1.0 / coupling["clock"]["source_fps"]
    substeps = max(8, math.ceil(240 * dt))
    _require(substeps <= 64, "source clock exceeds the unchanged physics-substep envelope")
    frame = {
        "axes": "x_right_y_depth_z_up", "calibration_status": CALIBRATION_STATUS,
        "units": {"length": LENGTH_UNIT, "mass": MASS_UNIT, "time": "second",
                  "orientation": "quaternion_wxyz", "angular_velocity": "world_radians_per_second"},
        "scale_bridge": {"body_height_to_world_scale": 1.0, "origin_world": [0.0, 0.0, 0.0],
                         "method": "Exact camera world-to-local transform; no outer placement",
                         "uncertainty_note": "Supplied camera/ground priors; unchanged monocular shape and authored masses, not measured anatomy"},
        "clock_binding": {"control_dt_s": dt, "physics_substeps": substeps,
                          "source_fps": coupling["clock"]["source_fps"],
                          "source_time_offset_s": coupling["clock"]["source_time_offset_s"],
                          "alignment_method": "Unchanged original frame/fps clock; cross-view alignment remains supplied and unverified"},
    }
    validate_camera_coupling(body, coupling, dt=dt, coordinate_frame=frame, source_binding=source_binding)
    scene = {
        "schema_version": 1, "profile": "shared-world-v6", "physical_validation": "unverified",
        "dt": dt, "substeps": substeps, "bounds": [6.0, 6.0, 6.0], "gravity": [0.0, 0.0, -9.81],
        "contact": {"time_constant": 0.01, "damping_ratio": 1.0}, "coordinate_frame": frame,
        "source_binding": source_binding, "human_spec": body, "camera_coupling": coupling, "objects": [],
        "controlled_id": "human", "target": None, "learned_policy_available": False,
        "task": {"type": "camera_human_motion_v1", "authored_addition": True, "start_offset_s": 0.0,
                 "horizon_steps": horizon(body["reference"], dt), "human_limits": deepcopy(limits), "object_goal": None},
        "assumptions": ["Exact saved body, original complete reference, supplied cameras and horizontal ground are unchanged",
                        "Source-frame control period differs from the unchanged body-model dt when original fps is not 30",
                        "Camera scale, body shape, masses, friction and stabilization remain priors; physical validation is unverified",
                        "Empty objects is this importer scope, not an equipment inventory or verified absence",
                        "No learned policy, independent exercise-task adapter, feet, contact load or grasp qualification",
                        "Authored world bounds/gravity/contact solver do not establish source measurements"],
    }
    result = validate(scene)
    _require(canonical_hash(result["human_spec"]) == canonical_hash(body)
             and result["camera_coupling"] == coupling and result["source_binding"] == source_binding,
             "shared validation must not alter body, coupling or source evidence")
    return result


def _write(out, files, pins, sizes, deadline):
    for name, raw in files.items():
        _check(deadline)
        _require(name not in pins and len(pins) < MAX_FILES - 2
                 and sum(sizes.values()) + len(raw) <= MAX_PACKAGE_BYTES - RESERVE_BYTES, "output package bound exceeded")
        write_package_files(_path(out), {name: raw})
        pins[name], sizes[name] = sha256(raw).hexdigest(), len(raw)
        _check(deadline)


def _inventory(out, pins, sizes, deadline):
    pending, seen, visits = [out], set(), 0
    while pending:
        with os.scandir(pending.pop()) as entries:
            for entry in entries:
                _check(deadline)
                visits += 1
                _require(visits <= MAX_FILES * 3, "output directory entry bound exceeded")
                info = entry.stat(follow_symlinks=False)
                if stat.S_ISDIR(info.st_mode):
                    pending.append(Path(entry.path))
                    continue
                name = Path(entry.path).relative_to(out).as_posix()
                _require(stat.S_ISREG(info.st_mode) and name in pins, "unexpected or unsafe output member")
                read_package_file(_path(out), name, max_bytes=max(1, sizes[name]), expected_sha256=pins[name])
                _check(deadline)
                seen.add(name)
    _require(seen == set(pins), "output members changed before seal")
    _check(deadline)


def _publish(out, raw, identity, deadline):
    # Terminal publication is one exclusive link; staged writes cannot leave a false completed seal.
    stage = Path(tempfile.mkdtemp(prefix=".weft-camera-seal-", dir=out.parent))
    try:
        write_package_files(stage, {"manifest.json": raw})
        _check(deadline)
        _require(_identity(_path(out)) == identity, "output directory changed before publication")
        fd = os.open(out, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        try:
            info = os.fstat(fd)
            _require((info.st_dev, info.st_ino) == identity, "output directory changed before publication")
            _check(deadline)
            os.link(stage / "manifest.json", "manifest.json", dst_dir_fd=fd, follow_symlinks=False)
        finally:
            os.close(fd)
    finally:
        try:
            (stage / "manifest.json").unlink(missing_ok=True)
            stage.rmdir()
        except OSError:
            pass


def prepare_camera_shared_candidate(source_run, out, *, max_seconds=MAX_SECONDS):
    """Seal a source-bound v6 candidate or failure; never overwrite inputs, run physics or create a lab.

    Limits are cooperative trusted-code checks, not a hard wall-time/memory sandbox.
    Returns status, sealed, candidate_directory and manifest_sha256. Existing/unsafe
    output paths raise before mutation; later failures retain a bounded receipt.
    """
    _require(type(max_seconds) in (int, float) and math.isfinite(max_seconds) and 0 < max_seconds <= MAX_SECONDS,
             "deadline must be in (0, 30] seconds")
    started = time.monotonic()
    deadline = started + max_seconds
    source, target = _path(source_run), _path(out)
    _require(not target.exists() and not target.is_relative_to(source) and not source.is_relative_to(target),
             "choose a new output outside the immutable source")
    source_identity = _identity(source)
    target.mkdir(parents=True, exist_ok=False)
    identity, pins, sizes, stage = _identity(target), {}, {}, "source_manifest"
    try:
        manifest_raw, source_pins = _manifest(source)
        _write(target, {"source-run/manifest.json": manifest_raw}, pins, sizes, deadline)
        stage = "runtime"
        runtime, versions = _runtime(deadline)
        runtime_pins = {name: sha256(raw).hexdigest() for name, raw in runtime.items()}
        _write(target, runtime, pins, sizes, deadline)
        inputs, selected, json_bytes = {}, {}, 0
        stage = "source_custody"
        for name in REQUIRED:
            _check(deadline)
            raw = read_package_file(source, name, max_bytes=min(MAX_MEMBER_BYTES, MAX_JSON_BYTES - json_bytes),
                                    expected_sha256=source_pins[name])
            json_bytes += len(raw)
            inputs[name] = raw
            selected[name] = {"sha256": source_pins[name], "bytes": len(raw),
                              "candidate_member": "source-run/" + name if name in ("scene.json", "task.json") else name}
        body, coupling, evidence, views, slots = _qualified_inputs(inputs, deadline)
        source_data = _json(inputs["source.json"])
        media = source_data.get("artifact")
        _require(type(media) is str and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,159}", media)
                 and media not in REQUIRED and media not in OPTIONAL and media not in ("manifest.json", "failure.json", "candidate.json")
                 and source_pins.get(media) == body["source_sha256"], "primary media must be a sealed distinct safe leaf matching the body")
        for name, raw in inputs.items():
            _write(target, {selected[name]["candidate_member"]: raw}, pins, sizes, deadline)
        for name in (media, *(name for name in OPTIONAL if name in source_pins)):
            raw = read_package_file(source, name, max_bytes=MAX_MEDIA_BYTES if name == media else MAX_MEMBER_BYTES,
                                    expected_sha256=source_pins[name])
            selected[name] = {"sha256": source_pins[name], "bytes": len(raw), "candidate_member": name}
            _write(target, {name: raw}, pins, sizes, deadline)
            del raw
        stage = "source_binding"
        observed_raw, observed_sha = _observations(evidence, inputs["evidence.json"])
        source_binding, construction = build_shared_world_binding({observed_sha: observed_raw},
            source_sha256=body["source_sha256"], world_ids=["human"], entity_refs={"human": (observed_sha, "person-0")}, deadline=deadline)
        _require(len(construction) <= MAX_MEMBER_BYTES, "construction evidence exceeds its byte bound")
        stage = "shared_schema"
        scene = _scene(body, coupling, source_binding)
        _check(deadline)
        qualification = {"schema_version": 1, "source_manifest_sha256": sha256(manifest_raw).hexdigest(),
            "source_manifest_members": len(source_pins), "selected_members": selected,
            "verification_scope": "Only selected members verified; historical manifest retained without claiming all inherited members verified",
            "primary_rights": {"status": "retained_source_declaration_not_new_verification", "member": "source.json",
                               "sha256": source_pins["source.json"], "value": source_data["rights"]},
            "secondary_custody": {"source_sha256": views[1]["source_sha256"], "media": "unavailable", "rights": "unavailable",
                                  "available": ["inputs/two-view-root/evidence.json", "inputs/two-view-root/camera.json", "inputs/two-view-root/config.json"]},
            "views": views, "source_slots": slots,
            "time": {"nominal_basis": "frame_index_divided_by_fps", "nominal_status": "assumed",
                     "native_timestamps": "retained_exactly_for_sampled_rows; unsampled_unknown", "cross_view_alignment": "supplied_unverified"},
            "independent_physical_validation": False}
        summary = {"schema_version": 1, "profile": PROFILE, "status": "constructed", "created_at": local_time(),
            "source_manifest_sha256": sha256(manifest_raw).hexdigest(), "source_sha256": body["source_sha256"],
            "human_bytes_sha256": source_pins["human.json"], "human_sha256": canonical_hash(body),
            "scene_sha256": canonical_hash(scene), "coupling_sha256": canonical_hash(coupling),
            "observation_document_sha256": observed_sha, "source_frames": len(slots), "retained_poses": len(body["reference"]),
            "horizon_steps": scene["task"]["horizon_steps"], "control_dt_s": scene["dt"], "body_dt_s": body["dt"],
            "scope": dict(SCOPE), "runtime_sha256": runtime_pins, "runtime_versions": versions,
            "bounds": {"seconds": max_seconds, "source_json_bytes": MAX_JSON_BYTES, "media_bytes": MAX_MEDIA_BYTES,
                       "runtime_bytes": MAX_RUNTIME_BYTES, "package_bytes": MAX_PACKAGE_BYTES, "files": MAX_FILES},
            "deadline_semantics": "Cooperative checks before exclusive terminal publication, not hard process isolation; bounded failure journaling may follow expiry",
            "new_inference": False, "simulation_steps": 0, "paid_requests": 0}
        _write(target, {"scene.json": _encoded(scene), "camera-coupling.json": _encoded(coupling),
            source_binding["evidence_documents"][observed_sha]: observed_raw,
            "construction-evidence.json": construction, "source-verification.json": _encoded(qualification),
            "candidate.json": _encoded({**summary, "status": "prepared_unsealed", "terminal_status_source": "manifest.json"},
                                       RESERVE_BYTES // 2)}, pins, sizes, deadline)
        stage = "terminal_verification"
        _require(_identity(_path(source)) == source_identity and _identity(_path(target)) == identity,
                 "input/output directory identity changed")
        current_manifest, _ = _manifest(source)
        _require(current_manifest == manifest_raw, "source manifest changed")
        for name, pin in selected.items():
            read_package_file(source, name, max_bytes=max(1, pin["bytes"]), expected_sha256=pin["sha256"])
            _check(deadline)
        for name, raw in runtime.items():
            relative = name.removeprefix("candidate-implementation/")
            root_dir, relative = (RUNTIME_ROOT.parent.parent, relative[8:]) if relative.startswith("project/") else (RUNTIME_ROOT, relative)
            read_package_file(_path(root_dir), relative, max_bytes=len(raw), expected_sha256=runtime_pins[name])
            _check(deadline)
        _inventory(target, pins, sizes, deadline)
        terminal = _encoded({"schema_version": 1, "profile": PROFILE, "status": "constructed", "files": pins,
                             "scope": dict(SCOPE)}, RESERVE_BYTES // 2)
        _publish(target, terminal, identity, deadline)
        # Published terminal state wins; no later timeout check may report an already sealed success as failure.
        return {**summary, "sealed": True, "manifest_sha256": sha256(terminal).hexdigest(),
                "candidate_directory": str(target), "files": len(pins) + 1, "bytes": sum(sizes.values()) + len(terminal),
                "seconds": time.monotonic() - started}
    except Exception as error:
        failure = {"schema_version": 1, "profile": PROFILE, "status": "failed", "stage": stage,
                   "error": type(error).__name__, "message": str(error)[:2000], "created_at": local_time(),
                   "scope": dict(SCOPE), "candidate_directory": str(target), "seconds": time.monotonic() - started,
                   "unadmitted_attempted_files": dict(pins)}
        try:
            _require(_identity(_path(target)) == identity, "changed output cannot receive a failure seal")
            raw = _encoded(failure, RESERVE_BYTES // 2)
            write_package_files(target, {"failure.json": raw})
            # Failed runs preserve partial bytes but do not certify files that may have caused a custody failure.
            terminal = _encoded({"schema_version": 1, "profile": PROFILE, "status": "failed",
                                 "files": {"failure.json": sha256(raw).hexdigest()},
                                 "unadmitted_partial_files_possible": True}, RESERVE_BYTES // 2)
            _publish(target, terminal, identity, None)
            return {**failure, "sealed": True, "manifest_sha256": sha256(terminal).hexdigest()}
        except Exception as sealing_error:
            return {**failure, "sealed": False, "manifest_sha256": None,
                    "sealing_error": {"type": type(sealing_error).__name__, "message": str(sealing_error)[:1000]}}
