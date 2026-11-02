"""Bounded construction assumptions for review, excluding policies and evaluation results."""

from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path, PurePosixPath
import re

from weft.exercise_support_inputs import validate_support_priors
from weft.io import canonical_hash
from weft.shared_world_human import _json
from weft.shared_world_packaging import read_package_file, read_package_manifest

PROFILE = "portal-scene-construction-context-v1"
MAX_CONTEXT_BYTES = 12 * 1024
SCOPE = "Simulation construction assumptions, not measured physics, recovered anatomy, policy results or physical validation."
NUMBER, TEXT, BOOLEAN = "number", "text", "boolean"
VECTOR = lambda count: (NUMBER, count, count)
STRINGS = (TEXT, 0, 8)
COORDINATES = {key: TEXT for key in ("axes", "length_unit", "mass_unit", "time_unit", "orientation_unit",
    "angular_velocity_unit", "calibration_status", "scale_method", "scale_uncertainty", "clock_method")}
COORDINATES.update(scale=NUMBER, origin_world=VECTOR(3), source_fps=NUMBER, source_time_offset_s=NUMBER)
MOBILITY = {key: NUMBER for key in ("floor_height", "base_height", "root_mass", "kp_per_mass", "kd_per_mass", "maximum_force")}
MOBILITY.update(ground_normal=VECTOR(3), ground_method=TEXT)
LENGTHS = {key: NUMBER for key in ("torso", "upper_arm", "forearm", "thigh", "shin", "shoulder_width", "hip_width")}
INTERVAL = {"start_offset_s": NUMBER, "horizon_steps": NUMBER}
CONTROL = {"root_assistance": TEXT, "object_forces": TEXT}
SUPPORT = {"object_id": TEXT, "human_segment": TEXT, "direction_world": VECTOR(3),
    "minimum_load_body_weight_fraction": NUMBER, "start_step": NUMBER, "end_step": NUMBER}
CONTACT = {"object_id": TEXT, "wrist": TEXT, "kind": TEXT, "start_step": NUMBER, "end_step": NUMBER}
PROVENANCE = {"mode": TEXT, "calibration_status": TEXT, "physical_validation": TEXT,
              "source_contact_verified": "optional_boolean", "assumptions": STRINGS}


def _shape(value, spec):
    if isinstance(spec, dict):
        if not isinstance(value, dict) or set(value) != set(spec):
            raise ValueError("Scene context has unsupported nested fields")
        for key, child in spec.items():
            _shape(value[key], child)
    elif isinstance(spec, tuple):
        item, low, high = spec
        if not isinstance(value, list) or not low <= len(value) <= high:
            raise ValueError("Scene context list exceeds its bound")
        for child in value:
            _shape(child, item)
    elif spec == NUMBER:
        if type(value) not in (int, float):
            raise ValueError("Scene context requires numeric construction values")
    elif spec == TEXT:
        if not isinstance(value, str) or not 1 <= len(value) <= 500:
            raise ValueError("Scene context requires bounded text")
    elif spec == BOOLEAN:
        if type(value) is not bool:
            raise ValueError("Scene context requires a boolean")
    elif spec == "optional_boolean":
        if value is not None and type(value) is not bool:
            raise ValueError("Scene context contact status must be boolean or unknown")


def _object_schema(obj):
    shape = obj.get("shape") if isinstance(obj, dict) else None
    schema = {key: TEXT for key in ("id", "label", "role", "shape")}
    schema.update(position=VECTOR(3), quaternion=VECTOR(4), velocity=VECTOR(3), angular_velocity=VECTOR(3),
                  mass=NUMBER, friction=VECTOR(3), provenance=PROVENANCE)
    if shape == "box":
        schema["size"] = VECTOR(3)
    elif shape == "sphere":
        schema["radius"] = NUMBER
    elif shape == "compound":
        geometry = obj.get("geometry")
        if not isinstance(geometry, dict) or set(geometry) != {"profile", "origin", "assumptions", "inertia", "parts"}:
            raise ValueError("Scene context compound geometry fields differ")
        _shape({key: geometry[key] for key in geometry if key != "parts"},
               {"profile": TEXT, "origin": TEXT, "assumptions": STRINGS,
                "inertia": {"mode": TEXT, "diagonal": VECTOR(3)}})
        parts = geometry["parts"]
        if not isinstance(parts, list) or not 1 <= len(parts) <= 8:
            raise ValueError("Scene context compound part count exceeds its bound")
        for part in parts:
            dimensions = {"box": {"size": VECTOR(3)}, "sphere": {"radius": NUMBER},
                          "capsule": {"radius": NUMBER, "length": NUMBER}}.get(part.get("shape")) if isinstance(part, dict) else None
            if dimensions is None:
                raise ValueError("Unsupported compound shape")
            _shape(part, {"id": TEXT, "shape": TEXT, "position": VECTOR(3), "quaternion": VECTOR(4), **dimensions})
        schema["geometry"] = "checked"
    else:
        raise ValueError("Unsupported scene context shape")
    return schema


def validate_context(snapshot, *, source_sha256):
    """Validate the closed review projection; this does not admit a physical world."""
    from weft.spatial_pose import SPATIAL_JOINT_NAMES

    try:
        raw = json.dumps(snapshot, allow_nan=False, separators=(",", ":")).encode()
    except (ValueError, TypeError, RecursionError) as error:
        raise ValueError("Scene context must be bounded finite JSON") from error
    if len(raw) > MAX_CONTEXT_BYTES:
        raise ValueError("Scene context exceeds 12 KiB")
    schema = {"schema_version": NUMBER, "profile": TEXT, "source_sha256": TEXT, "scene_profile": TEXT,
              "physical_validation": TEXT, "scope": TEXT, "coordinates": COORDINATES,
              "dt": NUMBER, "substeps": NUMBER, "gravity": VECTOR(3), "bounds": VECTOR(3),
              "contact_solver": {"time_constant": NUMBER, "damping_ratio": NUMBER},
              "body": {"profile": TEXT, "controller": TEXT, "lengths": LENGTHS,
                       "motor_limits": {key: NUMBER for key in SPATIAL_JOINT_NAMES}, "mobility": MOBILITY},
              "objects": "checked", "task_assumptions": "checked"}
    _shape(snapshot, schema)
    if (type(snapshot["schema_version"]) is not int or snapshot["schema_version"] != 1
            or snapshot["profile"] != PROFILE or snapshot["source_sha256"] != source_sha256
            or not isinstance(source_sha256, str) or not re.fullmatch(r"[a-f0-9]{64}", source_sha256)
            or snapshot["physical_validation"] != "unverified" or snapshot["scope"] != SCOPE):
        raise ValueError("Scene context identity or physical scope differs")
    objects = snapshot["objects"]
    if not isinstance(objects, list) or len(objects) > 12:
        raise ValueError("Scene context requires at most 12 rigid objects")
    identifiers = set()
    for obj in objects:
        _shape(obj, _object_schema(obj))
        if obj["id"] in identifiers or not re.fullmatch(r"[a-z][a-z0-9_-]{0,63}", obj["id"]):
            raise ValueError("Scene context object identities must be bounded and unique")
        identifiers.add(obj["id"])
    _shape(snapshot["task_assumptions"], {"profile": TEXT, "interval": INTERVAL, "control": CONTROL,
           "support_requirements": (SUPPORT, 0, 24), "contact_requirements": (CONTACT, 0, 24)})
    for kind in ("support_requirements", "contact_requirements"):
        if any(row["object_id"] not in identifiers for row in snapshot["task_assumptions"][kind]):
            raise ValueError("Scene context requirement names an absent object")
    return deepcopy(snapshot)


def _read(root, name, maximum, expected=None):
    raw = read_package_file(root, name, max_bytes=maximum, expected_sha256=expected)
    return _json(raw), sha256(raw).hexdigest(), raw


def construction_context(root):
    """Read only pinned construction inputs; return (snapshot, binding) or (None, None)."""
    from weft.shared_exercise_task import validate_shared_exercise_task
    from weft.shared_world import validate_shared_world

    root = Path(root).absolute()
    if ".." in root.parts or any(p.is_symlink() for p in (root, *root.parents)):
        raise ValueError("Scene context root must not contain traversal or links")
    job, _, _ = _read(root, "job.json", 64 << 10)
    status, _, _ = _read(root, "status.json", 1 << 20)
    if status.get("status") != "ready":
        raise ValueError("Scene context requires a ready project")
    if status.get("route") != "exercise":
        return None, None
    name = status.get("lab")
    if (not isinstance(name, str) or PurePosixPath(name).is_absolute() or "\\" in name
            or any(part in ("", ".", "..") for part in name.split("/"))):
        raise ValueError("Scene context lab path must remain within the project")
    lab, run = root / name, root / name / "run"
    if any(p.is_symlink() for p in (run, *run.parents)):
        raise ValueError("Scene context lab path cannot contain links")
    manifest, names = read_package_manifest(run)
    if len(names) > 6000 or len(manifest) > 1 << 20:
        raise ValueError("Scene context manifest exceeds the portal bound")
    config, config_sha, _ = _read(lab, "lab-config.json", 64 << 10)
    values, pins = {}, {}
    for field, maximum in (("scene.json", 16 << 20), ("exercise-task.json", 64 << 10)):
        if field not in names:
            raise ValueError("Scene context inputs must appear in the sealed run")
        values[field], pins[field], _ = _read(run, field, maximum, names[field])
    scene, task = values["scene.json"], values["exercise-task.json"]
    scene_hash = canonical_hash(scene)
    if canonical_hash(validate_shared_world(scene)) != scene_hash:
        raise ValueError("Scene context cannot silently normalize a saved scene")
    validate_shared_exercise_task(task, scene)
    if (config.get("manifest_sha256") != sha256(manifest).hexdigest() or config.get("scene_sha256") != scene_hash
            or config.get("task_sha256") != canonical_hash(task) or config.get("control_profile") != "shared_exercise"
            or scene["source_binding"]["mode"] != "source_qualified"
            or scene["source_binding"]["source_sha256"] != job["source_sha256"]
            or scene["human_spec"]["source_sha256"] != job["source_sha256"]):
        raise ValueError("Scene context source, task or saved lab identity differs")
    binding = {"run": run.relative_to(root).as_posix(), "manifest_sha256": sha256(manifest).hexdigest(),
               "lab_config_sha256": config_sha, "input_sha256": pins, "source_sha256": job["source_sha256"]}
    if job.get("kind") == "support_experiment":
        _, prior_sha, raw = _read(root, "support-priors.json", 128 << 10, job["support_priors_sha256"])
        prior = validate_support_priors(raw)
        if (prior["source_sha256"] != job["source_sha256"] or prior["objects"] != scene["objects"]
                or prior["support_requirements"] != task.get("support_requirements")
                or prior["scale"] != scene["coordinate_frame"]["scale_bridge"]["body_height_to_world_scale"]
                or prior["root_assistance"] != (task["control"]["root_assistance"] == "explicit_3_axis")):
            raise ValueError("Authored support priors differ from the constructed world")
        binding["support_priors_sha256"] = prior_sha
    frame, body = scene["coordinate_frame"], scene["human_spec"]
    bridge, clock, mobility = frame["scale_bridge"], frame["clock_binding"], body["mobility"]
    coordinates = {"axes": frame["axes"], "calibration_status": frame["calibration_status"],
        **{key + "_unit": value for key, value in frame["units"].items()}, "scale": bridge["body_height_to_world_scale"],
        "origin_world": bridge["origin_world"], "scale_method": bridge["method"], "scale_uncertainty": bridge["uncertainty_note"],
        "source_fps": clock["source_fps"], "source_time_offset_s": clock["source_time_offset_s"], "clock_method": clock["alignment_method"]}
    objects = []
    for obj in scene["objects"]:
        item = {key: value for key, value in obj.items() if key not in {"color", "provenance"}}
        provenance = obj["provenance"]
        item["provenance"] = {key: provenance.get(key, "unspecified") for key in ("mode", "calibration_status", "physical_validation")}
        item["provenance"].update(source_contact_verified=provenance.get("source_contact_verified"),
                                  assumptions=provenance.get("assumptions", []))
        objects.append(item)
    snapshot = {"schema_version": 1, "profile": PROFILE, "source_sha256": job["source_sha256"],
        "scene_profile": scene["profile"], "physical_validation": "unverified", "scope": SCOPE,
        "coordinates": coordinates, "dt": scene["dt"], "substeps": scene["substeps"], "gravity": scene["gravity"],
        "bounds": scene["bounds"], "contact_solver": scene["contact"], "body": {
            "profile": body["profile"], "controller": body["controller"], "lengths": body["lengths"], "motor_limits": body["motor_limits"],
            "mobility": {**{key: mobility[key] for key in MOBILITY if key not in {"ground_normal", "ground_method"}},
                         "ground_normal": mobility.get("ground_plane", {}).get("normal", [0., 0., 1.]),
                         "ground_method": mobility.get("ground_plane", {}).get("method", "authored_horizontal_floor")}},
        "objects": objects, "task_assumptions": {"profile": task["profile"], "interval": task["interval"], "control": task["control"],
            "support_requirements": task.get("support_requirements", []), "contact_requirements": task.get("contact_requirements", [])}}
    return validate_context(snapshot, source_sha256=job["source_sha256"]), binding
