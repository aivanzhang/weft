"""Explicit 3D hypotheses from sealed planar scenes; no inference or simulation."""

from copy import deepcopy
from dataclasses import dataclass
from hashlib import sha256
import json
import math
from pathlib import Path, PurePosixPath
import re
import time
from types import MappingProxyType

from weft.io import canonical_hash
from weft.shared_world_packaging import read_package_file, validate_package_layout


PROFILE = "planar-to-spatial-hypothesis-v1"
MAX_INPUT_BYTES = 64 << 20
MAX_FILE_BYTES = 16 << 20
MAX_MANIFEST_BYTES = 128 << 10
MAX_SCENE_BYTES = 250_000
MAX_CONFIG_BYTES = 64 << 10
MAX_FILES = 256
_SHA = re.compile(r"[0-9a-f]{64}\Z")
_ID = re.compile(r"[a-z][a-z0-9_-]{0,63}\Z")
_TERMINAL = {"ready", "completed", "frozen_unverified_hypothesis", "executable_hypothesis",
             "completed_diagnostic_replay", "development_comparison_complete_not_promoted"}
_DISCLOSURES = [
    "Front-plane geometry and motion are inherited from an unverified planar scene, not measured 3D truth.",
    "Lateral placement, box depth, sphere interpretation and absent v1 spin are explicit authored hypotheses.",
    "Front-plane-only motion explicitly assumes zero lateral velocity and zero X/Z angular velocity.",
    "image_top_z is a coordinate translation, not measured floor height or camera calibration.",
    "Implicit planar collision walls are removed; spatial bounds only terminate escaped motion.",
    "3D mass, materials, gravity, contact and task are authored anew; planar dynamics are not equivalent.",
    "Inertia follows uniform solid 3D primitives, not inherited planar moments or measured mass distribution.",
    "Archived source relationships remain observations or hypotheses, not added 3D constraints.",
]


@dataclass(frozen=True, slots=True)
class SpatialSceneInputs:
    source_directory: Path
    scene_artifact: str
    source_manifest_bytes: bytes
    source_files: object
    original_scene_bytes: bytes
    config_bytes: bytes
    scene_bytes: bytes
    report_bytes: bytes
    _receipt_json: str

    def receipt(self):
        return json.loads(self._receipt_json)


def _require(condition, message):
    if not condition:
        raise ValueError("Spatial construction: " + message)


def _bytes(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def _json(raw, maximum):
    _require(type(raw) is bytes and 0 < len(raw) <= maximum, "bounded exact JSON bytes required")

    def unique(pairs):
        result = {}
        for key, value in pairs:
            _require(key not in result, "duplicate JSON key")
            result[key] = value
        return result

    try:
        value = json.loads(raw, object_pairs_hook=unique)
        _require(type(value) is dict, "JSON object required")
        pending, nodes = [(value, 0)], 0
        while pending:
            item, depth = pending.pop()
            nodes += 1
            _require(nodes <= 100_000 and depth <= 64, "JSON structural bound exceeded")
            if isinstance(item, dict):
                pending.extend((child, depth + 1) for child in item.values())
            elif isinstance(item, list):
                pending.extend((child, depth + 1) for child in item)
            elif type(item) is float:
                _require(math.isfinite(item), "non-finite JSON value")
        return value
    except (UnicodeError, json.JSONDecodeError, RecursionError, OverflowError) as error:
        raise ValueError("Spatial construction: invalid bounded JSON") from error


def _number(value, name, low, high):
    _require(type(value) in (int, float) and math.isfinite(value) and low <= value <= high,
             name + " must be finite and bounded")
    return float(value)


def _keys(value, keys, label):
    _require(type(value) is dict and set(value) == set(keys), label + " fields differ from the contract")


def _construct(planar, config):
    # These validators perform schema/geometry math, never construct a model or environment.
    from weft.scene import validate_scene as validate_planar
    from weft.spatial_rigid import UNITS, validate_scene as validate_spatial

    source_sha = canonical_hash(planar)
    source = validate_planar(planar)
    _keys(config, {"schema_version", "profile", "source_scene_sha256", "image_top_z", "motion_embedding",
                   "boundary_mode", "bounds", "dt", "gravity", "contact", "target", "task", "objects",
                   "assumptions"}, "Authored 3D configuration")
    _require(type(config["schema_version"]) is int and config["schema_version"] == 1
             and config["profile"] == PROFILE and config["source_scene_sha256"] == source_sha,
             "configuration must pin this exact planar scene")
    _require(config["motion_embedding"] == "front_plane_only"
             and config["boundary_mode"] == "open_3d_bounds_termination", "explicit motion and boundary modes required")
    top = _number(config["image_top_z"], "image_top_z", 0, 4)
    assumptions = config["assumptions"]
    _require(isinstance(assumptions, list) and 1 <= len(assumptions) <= 32
             and all(type(text) is str and 1 <= len(text.strip()) <= 2000 for text in assumptions),
             "authored assumptions require bounded nonempty text")
    rows = config["objects"]
    _require(isinstance(rows, list) and len(rows) == len(source["objects"]), "every source object is required exactly once")
    source_ids = {obj["id"] for obj in source["objects"]}
    mapped, output_ids = {}, set()
    for row in rows:
        _require(type(row) is dict and type(row.get("id")) is str and row["id"] in source_ids
                 and row["id"] not in mapped, "unknown or repeated source object")
        keys = {"id", "output_id", "shape", "lateral_center", "mass", "friction"}
        if row.get("shape") == "box":
            keys.add("depth")
        if source["schema_version"] == 1:
            keys.add("angular_velocity_y")
        _keys(row, keys, "Authored object")
        _require(type(row["output_id"]) is str and _ID.fullmatch(row["output_id"])
                 and row["output_id"] not in output_ids, "explicit spatial IDs must be safe and one-to-one")
        mapped[row["id"]] = row
        output_ids.add(row["output_id"])
    config_sha, objects, transformations = canonical_hash(config), [], []
    for original, old in zip(planar["objects"], source["objects"], strict=True):
        row = mapped[old["id"]]
        shape = "sphere" if old["shape"] == "circle" else "box"
        _require(row["shape"] == shape, "shape must preserve the complete original front silhouette")
        lateral = _number(row["lateral_center"], "lateral_center", 0, 2)
        angle = old["angle"] if source["schema_version"] == 2 else 0.
        spin = (old["angular_velocity"] if source["schema_version"] == 2 else
                _number(row["angular_velocity_y"], "authored v1 spin", -200, 200))
        spin_origin = "source_planar_v2" if source["schema_version"] == 2 else "explicit_authored_v1_unknown"
        obj = {"id": row["output_id"], "label": old["label"], "role": old["role"], "color": old["color"],
            "shape": shape, "position": [old["position"][0], lateral, top - old["position"][1]],
            "quaternion": [math.cos(angle / 2), 0., math.sin(angle / 2), 0.],
            "velocity": [old["velocity"][0], 0., -old["velocity"][1]], "angular_velocity": [0., spin, 0.],
            "mass": deepcopy(row["mass"]), "friction": deepcopy(row["friction"]),
            "provenance": {"method": PROFILE, "source_scene_sha256": source_sha,
                "source_object_id": old["id"], "source_object_sha256": canonical_hash(original),
                "source_planar_provenance": deepcopy(original["provenance"]), "config_sha256": config_sha,
                "angular_velocity_origin": spin_origin,
                "depth_and_materials": "Explicit authored 3D hypotheses, not recovered physical properties"}}
        if shape == "box":
            obj["size"] = [old["size"][0], _number(row["depth"], "box depth", .01, .99), old["size"][1]]
        else:
            obj["radius"] = old["radius"]
        objects.append(obj)
        transformations.append({"source_id": old["id"], "output_id": row["output_id"],
            "shape": shape, "source_angle_radians": original.get("angle"),
            "orientation_origin": "source_planar_v2" if source["schema_version"] == 2 else "planar_v1_axis_alignment",
            "source_angular_velocity": original.get("angular_velocity"), "output_angular_velocity_y": spin,
            "angular_velocity_origin": spin_origin, "source_mass": original["mass"],
            "source_friction": original.get("friction"), "new_mass": row["mass"], "new_friction": deepcopy(row["friction"])})
    scene = {"schema_version": 1, "profile": "spatial-rigid-v1", "units": deepcopy(UNITS),
        **{key: deepcopy(config[key]) for key in ("bounds", "dt", "gravity", "contact", "target", "task")},
        "objects": objects, "controlled_id": mapped[source["controlled_id"]]["output_id"],
        "assumptions": [*_DISCLOSURES, *deepcopy(assumptions)]}
    scene = validate_spatial(scene)
    report = {"schema_version": 1, "profile": PROFILE, "status": "constructed_hypothesis",
        "simulation_ready": False, "physics_ready": False, "physical_validation": "unverified", "promoted": False,
        "source_scene_sha256": source_sha, "config_sha256": config_sha, "scene_sha256": canonical_hash(scene),
        "source_profile": source["profile"], "objects": transformations,
        "projection": {"source_xy_from_spatial_xyz": "[x, image_top_z - z]", "image_top_z": top,
                       "scope": "Inherited front geometry only; no camera/depth recovery or source-time changes"},
        "motion_embedding": config["motion_embedding"], "boundary_mode": config["boundary_mode"],
        "source_dynamics": {key: deepcopy(planar.get(key)) for key in
            ("dt", "gravity", "damping", "restitution", "boundary_friction", "task", "target")},
        "source_relationships": deepcopy(planar["relationships"]), "source_assumptions": deepcopy(planar["assumptions"]),
        "dynamics_boundary": {"planar_inertia": "2D disc/rectangle moments",
            "spatial_inertia": "New uniform solid sphere/box inertia from authored 3D mass and shape",
            "friction": "Planar per-shape product is not equivalent to spatial sliding/torsional/rolling contact",
            "restitution_and_damping": "Retained as source metadata only; not converted to 3D contact parameters",
            "walls": "Implicit planar collision walls are removed; only declared 3D objects can provide support or collision",
            "relationships": "Archived source claims, not instantiated 3D constraints"},
        "counts": {"source_objects": len(objects), "output_objects": len(objects), "model_compilations": 0,
                   "environment_constructions": 0, "physics_steps": 0, "fits": 0, "inference_calls": 0},
        "required_next_check": "Normal spatial lab geometry and runtime admission before claiming playability",
        "scope": "One explicit authored 3D hypothesis; no object removal, size repair, alternative selection, policy or physical promotion"}
    return {"scene": scene, "report": report}


def construct_spatial_scene(planar_scene, config):
    """Transform bounded JSON without fitting or constructing an environment."""
    try:
        return _construct(_json(_bytes(planar_scene), MAX_SCENE_BYTES), _json(_bytes(config), MAX_CONFIG_BYTES))
    except (TypeError, KeyError, AttributeError, IndexError, RecursionError, OverflowError) as error:
        raise ValueError("Spatial construction: malformed scene/configuration") from error


def _path(value):
    path = Path(value).absolute()
    _require(".." not in path.parts and len(path.parts) <= 64
             and not any(part.is_symlink() for part in (path, *path.parents)), "unsafe source directory")
    return path


def _manifest(raw):
    doc = _json(raw, MAX_MANIFEST_BYTES)
    if "schema" in doc:
        _keys(doc, {"schema", "created_at", "files"}, "Legacy suite manifest")
        _require(type(doc["schema"]) is int and doc["schema"] == 1
                 and type(doc["created_at"]) is str and 1 <= len(doc["created_at"]) <= 100,
                 "unsupported legacy manifest")
    else:
        _require({"schema_version", "files"} <= set(doc)
                 and not set(doc) - {"schema_version", "files", "created_at", "profile", "status"}
                 and type(doc["schema_version"]) is int and doc["schema_version"] == 1,
                 "unsupported versioned manifest")
    _status(doc, "manifest")
    files = doc["files"]
    _require(type(files) is dict and 1 <= len(files) <= MAX_FILES and "manifest.json" not in files,
             "manifest requires bounded non-self-referential members")
    _require(all(type(name) is str and len(name) <= 512 and type(pin) is str and _SHA.fullmatch(pin)
                 for name, pin in files.items()), "invalid member identity")
    validate_package_layout([*files, "manifest.json"])
    return files


def _status(doc, label):
    _require("status" not in doc or type(doc["status"]) is str and doc["status"] in _TERMINAL,
             label + " is incomplete, rejected or unsupported")
    _require(doc.get("physics_ready") is not True and doc.get("physical_validation", "unverified") == "unverified",
             label + " must remain physically unverified")


def prepare_spatial_scene_inputs(planar_run, config_bytes, *, scene_artifact="scene.json",
                                 expected_manifest_sha256=None, max_seconds=10, max_input_bytes=MAX_INPUT_BYTES):
    """Freeze every declared source member; an unpinned local seal is not external authentication."""
    _require(type(max_seconds) in (int, float) and math.isfinite(max_seconds) and 0 < max_seconds <= 10,
             "time bound must be positive and at most ten seconds")
    _require(type(max_input_bytes) is int and 1024 <= max_input_bytes <= MAX_INPUT_BYTES,
             "aggregate input bound must be 1 KiB–64 MiB including rechecks")
    _require(expected_manifest_sha256 is None or type(expected_manifest_sha256) is str
             and _SHA.fullmatch(expected_manifest_sha256), "expected exact manifest SHA256")
    _require(type(scene_artifact) is str and len(scene_artifact) <= 512, "explicit bounded scene artifact required")
    validate_package_layout([scene_artifact, "manifest.json"])
    root = _path(planar_run)
    config = _json(config_bytes, MAX_CONFIG_BYTES)
    started = time.monotonic()
    deadline, total = started + max_seconds, len(config_bytes)

    def check():
        if time.monotonic() >= deadline:
            raise TimeoutError("Spatial construction exceeded its cooperative time bound")

    def read(name, pin=None, maximum=MAX_FILE_BYTES):
        nonlocal total
        check()
        remaining = (max_input_bytes - total) // 2
        _require(remaining > 0, "aggregate input byte bound exhausted")
        raw = read_package_file(root, name, expected_sha256=pin, max_bytes=min(maximum, remaining))
        total += 2 * len(raw)  # Reserve the exact final identity recheck before consuming more input.
        return raw

    manifest_raw = read("manifest.json", expected_manifest_sha256, MAX_MANIFEST_BYTES)
    files = _manifest(manifest_raw)
    _require(scene_artifact in files, "selected scene must be sealed")
    archive = {name: read(name, pin) for name, pin in files.items()}
    parent = PurePosixPath(scene_artifact).parent
    prefixes = ("",) if parent == PurePosixPath(".") else ("", parent.as_posix() + "/")
    checked, metadata = [], {}
    for prefix in prefixes:
        failure = prefix + "failure.json"
        _require(failure not in archive and not (root / failure).exists() and not (root / failure).is_symlink(),
                 "selected source has a failure marker")
        for leaf in ("report.json", "candidate.json", "manifest.json"):
            name = prefix + leaf
            if name == "manifest.json":
                continue
            path = root / name
            _require(not (path.exists() or path.is_symlink()) or name in archive, "adjacent completion metadata is unsealed")
            if name not in archive:
                continue
            doc = _json(archive[name], MAX_MANIFEST_BYTES if leaf == "manifest.json" else MAX_FILE_BYTES)
            _status(doc, name)
            metadata[name] = doc
            checked.append({"artifact": name, "status": doc.get("status")})
            if leaf == "manifest.json":
                child_files = _manifest(archive[name])
                _require(PurePosixPath(scene_artifact).name in child_files
                         and all(prefix + child in files and files[prefix + child] == pin for child, pin in child_files.items()),
                         "selected child seal differs from its parent archive")
    original = _json(archive[scene_artifact], MAX_SCENE_BYTES)
    selected_prefix = "" if parent == PurePosixPath(".") else parent.as_posix() + "/"
    candidate = metadata.get(selected_prefix + "candidate.json")
    source_report = metadata.get(selected_prefix + "report.json", {})
    if "candidate_sha256" in source_report:
        _require(candidate is not None and source_report["candidate_sha256"] == canonical_hash(candidate),
                 "selected candidate identity differs from its report")
    if candidate is not None:
        _require("scene" not in candidate or canonical_hash(candidate["scene"]) == canonical_hash(original),
                 "selected candidate differs from its sealed scene")
    check()
    result = construct_spatial_scene(original, config)
    for name, raw in {"manifest.json": manifest_raw, **archive}.items():
        check(); _path(root)
        read_package_file(root, name, expected_sha256=sha256(raw).hexdigest(), max_bytes=max(1, len(raw)))
    for prefix in prefixes:
        failure = root / (prefix + "failure.json")
        _require(not failure.exists() and not failure.is_symlink(), "source failed during construction")
    check()
    receipt = {"schema_version": 1, "profile": PROFILE, "status": "source_archive_verified",
        "source_manifest_sha256": sha256(manifest_raw).hexdigest(), "scene_artifact": scene_artifact,
        "original_scene_file_sha256": sha256(archive[scene_artifact]).hexdigest(),
        "config_file_sha256": sha256(config_bytes).hexdigest(), "external_pin_supplied": expected_manifest_sha256 is not None,
        "source_files": [{"path": name, "sha256": files[name], "bytes": len(raw)} for name, raw in archive.items()],
        "completion_metadata": checked, "input_bytes_read_including_rechecks": total,
        "maximum_input_bytes": max_input_bytes, "maximum_seconds": max_seconds,
        "scope": "Every declared source member retained exactly, including any saved clocks/missing slots. Undeclared video/evidence remains unavailable. Local seals are not external authentication; no source semantics or physical accuracy re-evaluated."}
    result["report"]["source_archive"] = deepcopy(receipt)
    return SpatialSceneInputs(root, scene_artifact, manifest_raw, MappingProxyType(archive), archive[scene_artifact],
        config_bytes, _bytes(result["scene"]), _bytes(result["report"]), json.dumps(receipt, sort_keys=True, allow_nan=False))
