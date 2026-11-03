"""Exact initial collision geometry for conditional depth domains, without dynamics."""

from hashlib import sha256
from importlib.metadata import version
import json
import math
from pathlib import Path
import platform
import re
import time

from weft.io import canonical_hash


PROFILE = "moving-depth-geometry-v1"
MAX_BYTES = 128 << 10
MAX_PRIMITIVES = 64
MAX_SECONDS = 15
GATES = {"reset_overlap_tolerance_world": 1e-5, "strict_overlap_tolerance_world": 1e-8,
         "domain_clearance_world": 0., "mask_filtering_applied": False}
FILTER = {"contype": 4, "conaffinity": 7}
SCOPE = ("Exact first-reference simulator geometry, not recovered physical shape or contact. "
         "Zero-penetration depth domains are a conservative proposal prior; every candidate still needs ordinary native admission.")
_EXTRA_RUNTIME = ("moving_depth_geometry.py", "shared_world_action_replay.py", "shared_world_reconstruction.py")


def _require(condition, message):
    if not condition:
        raise ValueError("Moving depth geometry: " + message)


def _check(deadline):
    if time.monotonic() >= deadline:
        raise TimeoutError("Moving depth geometry exceeded its aggregate deadline")


def _deadline(deadline):
    _require(deadline is None or type(deadline) in (int, float) and math.isfinite(deadline), "finite deadline required")
    return min(time.monotonic() + MAX_SECONDS, deadline) if deadline is not None else time.monotonic() + MAX_SECONDS


def _encoded(value):
    raw = bytearray()
    for part in json.JSONEncoder(sort_keys=True, separators=(",", ":"), allow_nan=False).iterencode(value):
        chunk = part.encode()
        _require(len(raw) + len(chunk) <= MAX_BYTES, "complete geometry receipt exceeds byte bound")
        raw.extend(chunk)
    return bytes(raw)


def _runtime(names, deadline):
    from weft.shared_world_packaging import read_package_file

    _require(len(names) <= 128 and all(re.fullmatch(r"[a-z][a-z0-9_]*\.py", name) for name in names), "bounded runtime closure required")
    result = {}
    for name in sorted(names):
        _check(deadline)
        result[name] = sha256(read_package_file(Path(__file__).parent, name, max_bytes=2 << 20)).hexdigest()
    return result


def _vector(value):
    return isinstance(value, list) and len(value) == 3 and all(type(v) in (int, float) and math.isfinite(v) for v in value)


def _filter(model, gid, expected):
    values = {"contype": int(model.geom_contype[gid]), "conaffinity": int(model.geom_conaffinity[gid]),
              "margin": float(model.geom_margin[gid]), "gap": float(model.geom_gap[gid])}
    _require((values["contype"], values["conaffinity"]) == expected,
             "unsupported collision masks; no geometry may be silently excluded")
    _require(all(math.isfinite(values[k]) and values[k] >= 0 for k in ("margin", "gap")), "invalid collision margin/gap")
    return values


def _geometry(env):
    import mujoco
    import numpy as np

    model, data = env.model, env.data
    _require(model.npair == 0 and model.nexclude == 0, "explicit collision pairs/exclusions are unsupported")
    _require(not (int(model.opt.disableflags) & int(mujoco.mjtDisableBit.mjDSBL_CONTACT)), "disabled contact is unsupported")
    ignored = {gid for group in env._object_geometry_ids.values() for gid in group}
    human_ids = set(env._human_geom_to_name)
    _require(1 <= len(human_ids) <= MAX_PRIMITIVES
             and human_ids == set(range(model.ngeom)) - ignored - {env._floor_geom},
             "every original human collision geometry must be retained")
    result = []
    for gid in sorted(human_ids, key=lambda value: env._human_geom_to_name[value]):
        name = env._human_geom_to_name[gid]
        _require(isinstance(name, str) and 0 < len(name) <= 128, "named human collision geometry required")
        center = np.asarray(data.geom_xpos[gid], dtype=float)
        matrix = np.asarray(data.geom_xmat[gid], dtype=float).reshape(3, 3)
        radius = float(model.geom_size[gid, 0])
        _require(np.isfinite(center).all() and np.isfinite(matrix).all() and math.isfinite(radius) and radius > 0,
                 "finite positive collision primitive required")
        record = {"name": name, "radius": radius, **_filter(model, gid, (2, 5))}
        kind = int(model.geom_type[gid])
        if kind == int(mujoco.mjtGeom.mjGEOM_SPHERE):
            record.update(type="sphere", center=center.tolist())
        elif kind == int(mujoco.mjtGeom.mjGEOM_CAPSULE):
            half = float(model.geom_size[gid, 1])
            _require(math.isfinite(half) and half >= 0, "finite capsule segment required")
            delta = matrix[:, 2] * half
            record.update(type="capsule", endpoints=[(center - delta).tolist(), (center + delta).tolist()])
        else:
            raise ValueError(f"Moving depth geometry: unsupported exact human collision shape {kind}: {name}")
        result.append(record)
    floor_id = env._floor_geom
    _require(int(model.geom_type[floor_id]) == int(mujoco.mjtGeom.mjGEOM_PLANE), "exact plane floor required")
    normal = np.asarray(data.geom_xmat[floor_id], dtype=float).reshape(3, 3)[:, 2]
    _require(np.isfinite(normal).all() and abs(float(normal @ normal) - 1.) <= 1e-12, "unit floor normal required")
    floor = {"name": "mobile_floor", "type": "plane", "normal": normal.tolist(),
             "offset": float(normal @ data.geom_xpos[floor_id]), **_filter(model, floor_id, (1, 6))}
    return result, floor


def validate_depth_geometry(value):
    """Validate the bounded receipt shape only; a saved caller must verify its source seal."""
    pending, nodes = [(value, 0)], 0
    while pending:
        item, depth = pending.pop()
        nodes += 1
        _require(nodes <= 10000 and depth <= 16, "receipt exceeds structural bounds")
        if isinstance(item, dict):
            _require(len(item) <= 128 and all(isinstance(k, str) and len(k) <= 256 for k in item), "bounded string keys required")
            pending.extend((v, depth + 1) for v in item.values())
        elif isinstance(item, list):
            _require(len(item) <= 128, "receipt list exceeds bound")
            pending.extend((v, depth + 1) for v in item)
        elif isinstance(item, str):
            _require(len(item) <= 4096, "receipt string exceeds bound")
        else:
            _require(item is None or type(item) in (bool, int, float), "JSON receipt primitives required")
    _encoded(value)
    keys = {"schema_version", "profile", "input_pins", "runtime_sha256", "runtime_versions", "source_reference",
            "coordinate_frame", "world_bounds", "floor", "primitives", "candidate_collision_filter", "geometric_gates",
            "ignored_authored_object_ids", "physics_steps", "model_compilations", "scope"}
    _require(isinstance(value, dict) and set(value) == keys and type(value["schema_version"]) is int
             and value["schema_version"] == 1 and value["profile"] == PROFILE, "closed versioned geometry receipt required")
    _require(value["candidate_collision_filter"] == FILTER and value["geometric_gates"] == GATES,
             "original collision rules and conservative domain must remain explicit")
    _require(all(type(v) is int for v in value["candidate_collision_filter"].values())
             and value["geometric_gates"]["mask_filtering_applied"] is False
             and all(type(value["geometric_gates"][key]) in (int, float) for key in GATES if key != "mask_filtering_applied"),
             "typed collision and geometry rules required")
    _require(type(value["physics_steps"]) is int and value["physics_steps"] == 0
             and type(value["model_compilations"]) is int and value["model_compilations"] == 1
             and value["scope"] == SCOPE, "zero-step geometry scope required")
    primitives = value["primitives"]
    _require(isinstance(primitives, list) and 1 <= len(primitives) <= MAX_PRIMITIVES, "bounded complete primitive inventory required")
    names = set()
    for row in primitives:
        _require(isinstance(row, dict) and row.get("type") in {"sphere", "capsule"}, "unsupported primitive shape")
        common = {"name", "type", "radius", "contype", "conaffinity", "margin", "gap"}
        _require(set(row) == common | ({"center"} if row["type"] == "sphere" else {"endpoints"}), "closed primitive required")
        name = row["name"]
        _require(isinstance(name, str) and 0 < len(name) <= 128 and name not in names, "unique bounded primitive names required")
        names.add(name)
        _require(type(row["radius"]) in (int, float) and math.isfinite(row["radius"]) and row["radius"] > 0, "positive finite radius required")
        points = [row["center"]] if row["type"] == "sphere" else row["endpoints"]
        _require(isinstance(points, list) and len(points) == (1 if row["type"] == "sphere" else 2)
                 and all(_vector(point) for point in points), "exact finite primitive coordinates required")
        _require(type(row["contype"]) is int and type(row["conaffinity"]) is int
                 and (row["contype"], row["conaffinity"]) == (2, 5), "unsupported human collision filter")
        _require(all(type(row[k]) in (int, float) and math.isfinite(row[k]) and row[k] >= 0 for k in ("margin", "gap")), "invalid contact metadata")
    bounds = value["world_bounds"]
    _require(isinstance(bounds, dict) and set(bounds) == {"low", "high", "planes"}
             and bounds["low"] == [0., 0., 0.] and _vector(bounds["high"])
             and all(v > 0 for v in bounds["high"]), "original world bounding box required")
    expected = []
    for axis, length in enumerate(bounds["high"]):
        normal = [0., 0., 0.]; normal[axis] = 1.
        expected.append({"name": f"axis-{axis}-low", "normal": normal, "offset": 0.})
        expected.append({"name": f"axis-{axis}-high", "normal": [-v for v in normal], "offset": -length})
    _require(bounds["planes"] == expected, "world containment planes differ from exact bounds")
    floor = value["floor"]
    _require(isinstance(floor, dict) and set(floor) == {"name", "type", "normal", "offset", "contype", "conaffinity", "margin", "gap"}
             and floor["name"] == "mobile_floor" and floor["type"] == "plane" and _vector(floor["normal"])
             and abs(sum(v * v for v in floor["normal"]) - 1.) <= 1e-12
             and all(type(floor[k]) in (int, float) and math.isfinite(floor[k]) for k in ("offset", "margin", "gap"))
             and floor["margin"] >= 0 and floor["gap"] >= 0
             and type(floor["contype"]) is int and type(floor["conaffinity"]) is int
             and (floor["contype"], floor["conaffinity"]) == (1, 6), "unsupported exact floor/filter")
    for field in ("input_pins", "runtime_sha256"):
        pins = value[field]
        _require(isinstance(pins, dict) and 1 <= len(pins) <= 128
                 and all(isinstance(key, str) and isinstance(pin, str) and re.fullmatch(r"[0-9a-f]{64}", pin) for key, pin in pins.items()), "bounded identity pins required")
    _require(set(value["input_pins"]) == {"admission_receipt_sha256", "base_scene_sha256", "full_human_sha256",
        "source_sha256", "motion_evidence_sha256", "body_binding_sha256"}, "closed input identities required")
    _require(all(re.fullmatch(r"[a-z][a-z0-9_]*\.py", name) for name in value["runtime_sha256"]), "safe runtime identities required")
    source = value["source_reference"]
    hash_fields = {"reference_sha256", "root_reference_sha256", "human_qpos_sha256", "native_clock_sha256"}
    _require(isinstance(source, dict) and set(source) == hash_fields | {"first", "count"}
             and type(source["count"]) is int and 3 <= source["count"] <= 900
             and all(isinstance(source[k], str) and re.fullmatch(r"[0-9a-f]{64}", source[k]) for k in hash_fields), "complete source reference identities required")
    first = source["first"]
    _require(isinstance(first, dict) and set(first) == {"index", "source_frame", "evidence_id", "time_s"}
             and type(first["index"]) is int and first["index"] == 0
             and type(first["source_frame"]) is int and 0 <= first["source_frame"] < 1800
             and isinstance(first["evidence_id"], str) and 0 < len(first["evidence_id"]) <= 256
             and type(first["time_s"]) in (int, float) and math.isfinite(first["time_s"]) and 0 <= first["time_s"] <= 60,
             "exact first original reference required")
    versions = value["runtime_versions"]
    _require(isinstance(versions, dict) and set(versions) == {"python", "system", "machine", "mujoco", "numpy", "gymnasium"}
             and all(isinstance(v, str) and 0 < len(v) <= 128 for v in versions.values()), "bounded runtime versions required")
    ignored = value["ignored_authored_object_ids"]
    _require(isinstance(ignored, list) and len(ignored) <= 12 and all(isinstance(v, str) and 0 < len(v) <= 128 for v in ignored)
             and ignored == sorted(set(ignored)), "bounded original authored prop inventory required")
    _require(isinstance(value["coordinate_frame"], dict), "original coordinate-frame contract required")
    return json.loads(_encoded(value))


def extract_depth_geometry(inputs, *, deadline=None):
    """Compile one original body, preserve every collision primitive, and never step."""
    import mujoco
    import numpy as np
    from weft.moving_equipment_inputs import _RUNTIME, validate_moving_equipment_inputs
    from weft.shared_world import SharedWorldEnv

    stop = _deadline(deadline)
    packet = validate_moving_equipment_inputs(inputs, deadline=stop)
    base = packet.base_scene
    _require(base["task"]["start_offset_s"] == 0 and base["controlled_id"] == "human", "original body start required")
    _require(all(row["provenance_mode"] == "authored_addition" for row in base["source_binding"]["entity_bindings"]
                 if row["world_entity_id"] != "human"), "only constructor-removed authored props may be ignored")
    admission = json.loads(packet.admission_receipt_bytes)
    runtime = _runtime(set(_RUNTIME) | set(_EXTRA_RUNTIME), stop)
    _require(all(runtime.get(name) == pin for name, pin in admission["runtime_sha256"].items()), "admission runtime changed")
    _check(stop)
    env = SharedWorldEnv(base)
    try:
        _check(stop)
        _require(env.scene == base, "runtime changed the admitted base")
        ref, root = base["human_spec"]["reference"][0], base["human_spec"]["root_reference"][0]
        # Match the unchanged constructor's exact original-reference projection, without prop reset checks.
        mujoco.mj_resetData(env.model, env.data)
        pose = [ref["joints"][name] for name in env.joint_names] + [v * env.scale for v in root["translation"]]
        _require(len(pose) == len(env._q) and ref["time_s"] == root["time_s"], "unaltered initial joint/root layout required")
        env.data.qpos[env._q] = pose
        mujoco.mj_forward(env.model, env.data)
        primitives, floor = _geometry(env)
        _require(env.steps == 0 and env.data.time == 0 and np.array_equal(env.data.qpos[env._q], pose)
                 and not np.any(env.data.qvel), "initial geometry extraction must not advance or shift the body")
        bounds = {"low": [0., 0., 0.], "high": list(base["bounds"]), "planes": []}
        for axis, length in enumerate(bounds["high"]):
            normal = [0., 0., 0.]; normal[axis] = 1.
            bounds["planes"] += [{"name": f"axis-{axis}-low", "normal": normal, "offset": 0.},
                                 {"name": f"axis-{axis}-high", "normal": [-v for v in normal], "offset": -length}]
        first = packet.body_references[0]
        result = {"schema_version": 1, "profile": PROFILE,
            "input_pins": {"admission_receipt_sha256": sha256(packet.admission_receipt_bytes).hexdigest(),
                "base_scene_sha256": canonical_hash(base), "full_human_sha256": sha256(packet.human_bytes).hexdigest(),
                "source_sha256": packet.source_binding["source_sha256"], "motion_evidence_sha256": sha256(packet.motion_evidence_bytes).hexdigest(),
                "body_binding_sha256": canonical_hash(packet.body_binding)},
            "runtime_sha256": runtime, "runtime_versions": {"python": platform.python_version(), "system": platform.system(),
                "machine": platform.machine(), **{name: version(name) for name in ("mujoco", "numpy", "gymnasium")}},
            "source_reference": {"first": first, "count": len(base["human_spec"]["reference"]),
                "reference_sha256": canonical_hash(base["human_spec"]["reference"]),
                "root_reference_sha256": canonical_hash(base["human_spec"]["root_reference"]),
                "human_qpos_sha256": canonical_hash(pose), "native_clock_sha256": canonical_hash(base["human_spec"]["native_clock"])},
            "coordinate_frame": base["coordinate_frame"], "world_bounds": bounds, "primitives": primitives, "floor": floor,
            "candidate_collision_filter": dict(FILTER), "geometric_gates": dict(GATES),
            "ignored_authored_object_ids": sorted(obj["id"] for obj in base["objects"]),
            "physics_steps": 0, "model_compilations": 1, "scope": SCOPE}
        result = validate_depth_geometry(result)
        _check(stop)
        _require(_runtime(set(runtime), stop) == runtime, "geometry runtime changed during extraction")
        return result
    finally:
        env.close()
