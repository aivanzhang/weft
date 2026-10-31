"""Validate editable, non-executable scene descriptions before Blender receives them."""

import math
from copy import deepcopy

from avatar_assets.safety import require, slug

SCENE_LIMITS = {"parts": 64, "bones": 32, "materials": 16, "states": 12,
                "keys_per_state": 64, "extent_m": 20, "max_seconds": 20}


def vector(value, size, label, bound=20):
    require(isinstance(value, list) and len(value) == size, f"Invalid {label}")
    require(all(type(x) in (int, float) and math.isfinite(x) and abs(x) <= bound for x in value),
            f"Out-of-range {label}")


def validate_scene(data):
    require(isinstance(data, dict) and data.get("schema_version") == 1, "Expected scene schema_version 1")
    data = deepcopy(data)
    slug(data["id"])
    require(data.get("output_profile", "general") in ("general", "weft-v1"), "Unknown output profile")
    for key in ("subject", "style", "prompt"):
        require(isinstance(data.get(key), str) and 0 < len(data[key]) <= 12000, f"Missing or oversized {key}")
    require(isinstance(data.get("authored_assumptions"), list) and data["authored_assumptions"], "Disclose authored assumptions")
    require(all(isinstance(x, str) and len(x) < 4000 for x in data["authored_assumptions"]), "Invalid authored assumptions")
    mats = data.get("materials")
    require(isinstance(mats, dict) and 0 < len(mats) <= SCENE_LIMITS["materials"], "Material limit exceeded")
    for name, material in mats.items():
        slug(name)
        vector(material["color"], 4, "material color", 1)
        require(all(0 <= x <= 1 for x in material["color"]) and material["color"][3] == 1, "Use opaque PBR materials")
        for k, default in (("roughness", .5), ("metallic", 0)):
            v = material.get(k, default)
            require(type(v) in (int, float) and 0 <= v <= 1, f"Invalid {k}")
        if data.get("output_profile") == "weft-v1":
            require(material.get("metallic", 0) == 0 and material.get("roughness", .5) == 1,
                    "Weft variant must explicitly author matte diffuse materials")
    bones = data.get("bones")
    require(isinstance(bones, list) and 0 < len(bones) <= SCENE_LIMITS["bones"], "Bone limit exceeded")
    names = set()
    roots = 0
    for b in bones:
        slug(b["name"])
        require(b["name"] not in names, "Duplicate bone")
        require(b.get("parent") is None or b["parent"] in names, "Bones must be parent-first without cycles")
        roots += b.get("parent") is None
        for k in ("head", "tail"):
            vector(b[k], 3, f"bone {k}")
        require(sum((a-bb)**2 for a, bb in zip(b["head"], b["tail"])) > .000001, "Zero-length bone")
        names.add(b["name"])
    require(roots == 1, "Use one explicit skeleton root")
    parts = data.get("parts")
    require(isinstance(parts, list) and 0 < len(parts) <= SCENE_LIMITS["parts"], "Part limit exceeded")
    part_names = set()
    for p in parts:
        slug(p["name"])
        require(p["name"] not in part_names, "Part names must be unique")
        part_names.add(p["name"])
        require(p["shape"] in ("sphere", "cube", "cone", "cylinder", "torus"), "Unsupported geometry primitive")
        require(p["material"] in mats and p["bone"] in names, "Unknown part material or bone")
        for k in ("location", "scale"):
            vector(p[k], 3, f"part {k}")
        require(min(p["scale"]) >= .002, "Part dimensions must be positive")
        vector(p.get("rotation", [0, 0, 0]), 3, "part rotation", math.tau * 2)
    states = data.get("states")
    require(isinstance(states, list) and 0 < len(states) <= SCENE_LIMITS["states"], "State limit exceeded")
    ids = set()
    for s in states:
        slug(s["id"])
        require(s["id"] not in ids, "Duplicate state")
        ids.add(s["id"])
        require(isinstance(s.get("meaning"), str) and 0 < len(s["meaning"]) < 4000, "State meaning required")
        require(type(s.get("loop")) is bool, "Explicit loop intent required")
        duration = s.get("duration")
        require(type(duration) in (int, float) and 0.5 <= duration <= 20 and (duration * 24).is_integer()
                if isinstance(duration, float) else type(duration) is int and 1 <= duration <= 20, "Invalid clip duration")
        keys = s.get("keyframes")
        require(isinstance(keys, list) and 2 <= len(keys) <= 64, "A state needs 2–64 keyframes")
        previous = -1
        for k in keys:
            require(type(k["frame"]) is int and previous < k["frame"] <= duration * 24, "Nonmonotonic/out-of-range keyframe")
            previous = k["frame"]
            require(isinstance(k["bones"], dict) and all(b in names for b in k["bones"]), "Unknown keyed bone")
            for transform in k["bones"].values():
                require(set(transform) <= {"rotation", "location"}, "Unsupported bone transform")
                for prop, v in transform.items():
                    vector(v, 3, "keyed transform", 6.3 if prop == "rotation" else 5)
        require(keys[0]["frame"] == 0 and keys[-1]["frame"] == duration * 24, "Keys must cover exact clip endpoints")
    camera = data["camera"]
    vector(camera["target"], 3, "camera target")
    vector(camera["position"], 3, "camera position")
    require(type(camera["orthographic_scale"]) in (int, float)
            and .5 <= camera["orthographic_scale"] <= 20, "Invalid camera scale")
    require(camera["target"] != camera["position"], "Camera cannot be at target")
    vector(data.get("background", [.95, .96, .98, 1]), 4, "background", 1)
    require(all(0 <= v <= 1 for v in data.get("background", [.95, .96, .98, 1])), "Negative background channel")
    attachments = data.get("attachments", [])
    require(isinstance(attachments, list) and len(attachments) <= 16, "Attachment limit exceeded")
    parents = {b["name"]: b.get("parent") for b in bones}
    attachment_names = set()
    for a in attachments:
        slug(a["name"])
        require(a["name"] not in attachment_names, "Duplicate attachment")
        attachment_names.add(a["name"])
        require(a.get("parent_bone") in names and a.get("child_bone") in names
                and parents[a["child_bone"]] == a["parent_bone"], "Invalid attachment parent/child relation")
        require(all(isinstance(a.get(k), str) and 0 < len(a[k]) < 4000
                    for k in ("placement", "authored_assumption")), "Attachment needs placement and authored assumption")
    return data


def reskin(data, changes):
    data = deepcopy(data)
    require(isinstance(changes, dict) and set(changes) <= set(data["materials"]), "Skin can change only named materials")
    data["materials"].update(changes)
    return validate_scene(data)
