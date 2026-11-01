"""Validate editable, non-executable scene descriptions before Blender receives them."""

import math
import json
from copy import deepcopy

from avatar_assets.safety import LIMITS, local_file, require, slug

SCENE_LIMITS = {"parts": 64, "bones": 32, "materials": 16, "states": 12,
                "keys_per_state": 64, "extent_m": 20, "max_seconds": 20}
QUALITY_LIMITS = {"parts": 128, "control_vertices": 30000, "faces": 60000,
                  "fur_strands": 24000, "fur_vertices": 72000, "native_fibers": 240000,
                  "texture_pixels": 32 * 1024 * 1024, "scene_bytes": 8 * 1024 * 1024}


def read_scene(path):
    """Read bounded scene data; v2 has room for explicit mesh coordinates and weights."""
    path = local_file(path, QUALITY_LIMITS["scene_bytes"])
    with path.open("rb") as handle:
        raw = handle.read(QUALITY_LIMITS["scene_bytes"]+1)
    require(len(raw) <= QUALITY_LIMITS["scene_bytes"], "Scene JSON exceeds byte limit")

    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, f"Duplicate JSON key: {key}")
            result[key] = value
        return result

    def invalid(value):
        raise ValueError(f"Nonfinite JSON constant: {value}")

    try:
        data = json.loads(raw, object_pairs_hook=pairs, parse_constant=invalid)
    except (RecursionError, UnicodeError) as error:
        raise ValueError("Invalid scene JSON") from error
    require(isinstance(data, dict), "Scene must be an object")
    require(data.get("schema_version") == 2 or len(raw) <= LIMITS["json_bytes"],
            "Version 1 scene exceeds byte limit")
    return validate_scene(data)


def vector(value, size, label, bound=20):
    require(isinstance(value, list) and len(value) == size, f"Invalid {label}")
    require(all(type(x) in (int, float) and math.isfinite(x) and abs(x) <= bound for x in value),
            f"Out-of-range {label}")


def _number(value, low, high, label):
    require(type(value) in (int, float) and math.isfinite(value) and low <= value <= high,
            f"Invalid {label}")


def _integer(value, low, high, label):
    require(type(value) is int and low <= value <= high, f"Invalid {label}")


def _surface(surface):
    require(isinstance(surface, dict) and set(surface) == {"pattern", "resolution", "strength", "seed"},
            "Surface needs pattern, resolution, strength and seed")
    require(surface["pattern"] in ("fur", "scales", "brushed", "iris"), "Unsupported surface pattern")
    require(type(surface["resolution"]) is int and surface["resolution"] in (256, 512, 1024, 2048),
            "Surface resolution must be 256, 512, 1024 or 2048")
    require(surface["pattern"] != "iris" or surface["resolution"] >= 1024, "Iris surface requires HD resolution")
    _number(surface["strength"], 0, 1, "surface strength")
    _integer(surface["seed"], 0, 2**31-1, "surface seed")


def _quality_part(part, bones, totals):
    for key, low, high in (("segments", 8, 64), ("rings", 6, 48), ("bevel_segments", 1, 4)):
        if key in part:
            _integer(part[key], low, high, key)
    _number(part.get("bevel", 0), 0, .3, "bevel width")
    if "fur" in part:
        fur = part["fur"]
        require(isinstance(fur, dict) and {"count", "length", "width", "seed"} <= set(fur)
                and set(fur) <= {"count", "length", "width", "seed", "mode", "direction"},
                "Fur needs bounded count, length, width, seed and optional mode/direction")
        require(fur.get("mode", "triangles") in ("triangles", "cards"), "Unsupported fur mode")
        cards = fur.get("mode") == "cards"
        if "direction" in fur:
            require(cards, "Comb direction requires hair cards")
            vector(fur["direction"], 3, "fur comb direction", 1)
            require(sum(value*value for value in fur["direction"]) > 1e-8, "Zero fur comb direction")
        _integer(fur["count"], 0, 10000, "fur count")
        _integer(fur["seed"], 0, 2**31-1, "fur seed")
        _number(fur["length"], .001, .04, "fur length")
        _number(fur["width"], .0003, .04 if cards else .004, "fur width")
        totals["fur_strands"] += fur["count"]
        require(totals["fur_strands"] <= QUALITY_LIMITS["fur_strands"], "Aggregate fur limit exceeded")
        totals["fur_vertices"] += fur["count"]*(6 if cards else 3)
        require(totals["fur_vertices"] <= QUALITY_LIMITS["fur_vertices"], "Aggregate fur vertex limit exceeded")
    if "native_groom" in part:
        groom = part["native_groom"]
        required = {"parents", "children", "length", "radius", "roughness", "seed"}
        require(isinstance(groom, dict) and set(groom) in (required, required | {"direction", "comb"}),
                "Native groom requires six explicit fields and optional paired direction/comb")
        _integer(groom["parents"], 16, 10000, "native groom parents")
        _integer(groom["children"], 0, 30, "native groom children")
        _integer(groom["seed"], 0, 2**31-1, "native groom seed")
        _number(groom["length"], .001, .04, "native groom length")
        _number(groom["radius"], .00001, .001, "native groom radius")
        _number(groom["roughness"], 0, .01, "native groom roughness")
        if "direction" in groom:
            vector(groom["direction"], 3, "native groom direction", 1)
            require(math.hypot(*groom["direction"]) > 1e-8, "Zero or unstable native groom direction")
            _number(groom["comb"], 0, .95, "native groom comb")
        totals["native_fibers"] += groom["parents"]*(groom["children"]+1)
        require(totals["native_fibers"] <= QUALITY_LIMITS["native_fibers"], "Aggregate native groom limit exceeded")
    if part["shape"] != "mesh":
        require(not any(key in part for key in ("vertices", "faces", "uv", "weights")),
                "Explicit topology and weights require shape mesh")
        return
    require("segments" not in part and "rings" not in part, "Mesh topology is explicit, not segmented")
    vertices, faces = part.get("vertices"), part.get("faces")
    require(isinstance(vertices, list) and 3 <= len(vertices) <= QUALITY_LIMITS["control_vertices"],
            "Invalid mesh vertex count")
    require(isinstance(faces, list) and 0 < len(faces) <= QUALITY_LIMITS["faces"], "Invalid mesh face count")
    totals["control_vertices"] += len(vertices)
    totals["faces"] += len(faces)
    require(totals["control_vertices"] <= QUALITY_LIMITS["control_vertices"]
            and totals["faces"] <= QUALITY_LIMITS["faces"], "Aggregate mesh topology limit exceeded")
    for vertex in vertices:
        vector(vertex, 3, "mesh vertex")
    for face in faces:
        require(isinstance(face, list) and len(face) in (3, 4)
                and all(type(index) is int and 0 <= index < len(vertices) for index in face)
                and len(set(face)) == len(face), "Faces must be nonrepeated in-range triangles or quads")
        triangles = [(face[0], face[1], face[2])]
        if len(face) == 4:
            triangles.append((face[0], face[2], face[3]))
        for triangle in triangles:
            a, b, c = (vertices[index] for index in triangle)
            u, v = [b[i]-a[i] for i in range(3)], [c[i]-a[i] for i in range(3)]
            cross = [u[1]*v[2]-u[2]*v[1], u[2]*v[0]-u[0]*v[2], u[0]*v[1]-u[1]*v[0]]
            require(sum(value*value for value in cross) > 1e-20, "Degenerate mesh face")
    if "uv" in part:
        require(isinstance(part["uv"], list) and len(part["uv"]) == len(vertices), "UV count must match vertices")
        for uv in part["uv"]:
            vector(uv, 2, "mesh UV", 16)
    if "weights" in part:
        require(isinstance(part["weights"], list) and len(part["weights"]) == len(vertices),
                "Weight count must match vertices")
        for weights in part["weights"]:
            require(isinstance(weights, dict) and 1 <= len(weights) <= 4 and set(weights) <= bones,
                    "Use one to four known bone weights per vertex")
            require(all(type(value) in (int, float) and math.isfinite(value) and 0 < value <= 1
                        for value in weights.values()) and abs(sum(weights.values())-1) <= 1e-5,
                    "Vertex weights must be positive and normalized")


def validate_scene(data):
    require(isinstance(data, dict) and type(data.get("schema_version")) is int
            and data["schema_version"] in (1, 2), "Expected scene schema_version 1 or 2")
    data = deepcopy(data)
    quality = data["schema_version"] == 2
    slug(data["id"])
    require(data.get("output_profile", "general") in ("general", "weft-v1"), "Unknown output profile")
    require(not quality or data.get("output_profile", "general") == "general",
            "Version 2 quality geometry/materials are general-only; Weft conversion is unsupported")
    for key in ("subject", "style", "prompt"):
        require(isinstance(data.get(key), str) and 0 < len(data[key]) <= 12000, f"Missing or oversized {key}")
    require(isinstance(data.get("authored_assumptions"), list) and data["authored_assumptions"], "Disclose authored assumptions")
    require(all(isinstance(x, str) and len(x) < 4000 for x in data["authored_assumptions"]), "Invalid authored assumptions")
    mats = data.get("materials")
    require(isinstance(mats, dict) and 0 < len(mats) <= SCENE_LIMITS["materials"], "Material limit exceeded")
    texture_pixels = 0
    for name, material in mats.items():
        slug(name)
        require(isinstance(material, dict), "Invalid material")
        vector(material["color"], 4, "material color", 1)
        require(all(0 <= x <= 1 for x in material["color"]) and material["color"][3] == 1, "Use opaque PBR materials")
        for k, default in (("roughness", .5), ("metallic", 0)):
            v = material.get(k, default)
            require(type(v) in (int, float) and 0 <= v <= 1, f"Invalid {k}")
        if "surface" in material:
            require(quality, "Surface textures require scene schema_version 2")
            _surface(material["surface"])
            texture_pixels += 2*material["surface"]["resolution"]**2
        if "fur_atlas" in material:
            require(quality and material["fur_atlas"] in ("fine-v1", "distance-v1"),
                    "Fur atlas requires a known schema v2 profile")
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
    require(isinstance(parts, list) and 0 < len(parts) <= (QUALITY_LIMITS["parts"] if quality else SCENE_LIMITS["parts"]),
            "Part limit exceeded")
    card_materials = {p.get("material") for p in parts
                      if isinstance(p, dict) and isinstance(p.get("fur"), dict) and p["fur"].get("mode") == "cards"
                      and p["fur"].get("count") and isinstance(p.get("material"), str)}
    texture_pixels += len(card_materials)*1024**2
    require(texture_pixels <= QUALITY_LIMITS["texture_pixels"], "Aggregate texture pixel limit exceeded")
    part_names = set()
    totals = {"control_vertices": 0, "faces": 0, "fur_strands": 0, "fur_vertices": 0, "native_fibers": 0}
    for p in parts:
        slug(p["name"])
        require(p["name"] not in part_names, "Part names must be unique")
        part_names.add(p["name"])
        require(p["shape"] in ("sphere", "cube", "cone", "cylinder", "torus") or quality and p["shape"] == "mesh",
                "Unsupported geometry primitive")
        require(p["material"] in mats and p["bone"] in names, "Unknown part material or bone")
        for k in ("location", "scale"):
            vector(p[k], 3, f"part {k}")
        require(min(p["scale"]) >= .002, "Part dimensions must be positive")
        vector(p.get("rotation", [0, 0, 0]), 3, "part rotation", math.tau * 2)
        if quality:
            _quality_part(p, names, totals)
        else:
            require(not any(key in p for key in ("vertices", "faces", "uv", "weights", "segments", "rings", "bevel", "bevel_segments", "fur", "native_groom")),
                    "Quality geometry fields require scene schema_version 2")
    for material in card_materials:
        coverage = {bool(part.get("native_groom")) for part in parts if part["material"] == material
                    and part.get("fur", {}).get("mode") == "cards" and part["fur"]["count"]}
        require(len(coverage) == 1,
                f"Hair-card parts sharing material {material!r} must all use native_groom or all omit it; use separate materials for mixed coverage")
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
    if "studio" in data:
        studio = data["studio"]
        require(quality and isinstance(studio, dict) and {"ground_z", "world_strength", "exposure"} <= set(studio)
                and set(studio) <= {"ground_z", "world_strength", "exposure", "preset"},
                "Studio needs ground_z, world_strength and exposure, with optional preset, in v2")
        require("preset" not in studio or studio["preset"] == "plush", "Unsupported studio preset")
        _number(studio["ground_z"], -5, 5, "studio ground")
        _number(studio["world_strength"], .05, 2, "studio world strength")
        _number(studio["exposure"], -2, 2, "studio exposure")
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
