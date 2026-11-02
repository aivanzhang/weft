"""Bounded GLB ingestion; this module neither reconstructs geometry nor simulates it."""

from __future__ import annotations

import bisect
import hashlib
import json
import math
import struct
from pathlib import Path


MAX_FILE_BYTES = 64 * 1024 * 1024
MAX_JSON_BYTES = 2 * 1024 * 1024
MAX_VERTICES = 100_000
MAX_TRIANGLE_INDICES = 300_000
MAX_NODES = 256
# Dense real fur needs position, four joints/weights and indices without widening the vertex/file limits.
MAX_ACCESSOR_VALUES = 2_000_000
MAX_ANIMATION_SECONDS = 120
MAX_MOTION_PROBES = 128
_IDENTITY = (1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1.)
_COMPONENTS = {5121: ("B", 1), 5123: ("H", 2), 5125: ("I", 4), 5126: ("f", 4)}
_WIDTHS = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT4": 16}


def validate_render_attributes(path):
    """Check HD frame validity and UV binding, not tangent derivation or Weft compatibility."""
    glb = _GLB(path)
    primitives, vertices = [], 0
    surface_value_limit = 11*MAX_VERTICES
    for mesh in glb.doc.get("meshes", []):
        for primitive in mesh.get("primitives", []):
            attrs = primitive.get("attributes", {})
            position = glb.doc["accessors"][_index(attrs.get("POSITION"), len(glb.doc["accessors"]), "position accessor")]
            count = position.get("count")
            _require(type(count) is int and 0 < count <= MAX_VERTICES, "Invalid surface vertex count")
            vertices += count
            _require(vertices <= MAX_VERTICES, "Surface vertex count exceeds bound")

            def surface_accessor(index, kind, components, label):
                index = _index(index, len(glb.doc["accessors"]), label+" accessor")
                spec = glb.doc["accessors"][index]
                _require(type(spec.get("count")) is int and spec["count"] == count,
                         "Surface "+label+" count mismatch")
                if index not in glb.cache:
                    _require(glb.value_count+count*_WIDTHS[kind] <= surface_value_limit,
                             "Decoded surface values exceed bound")
                return glb.accessor(index, kind, components)

            material = glb.doc.get("materials", [])[_index(primitive.get("material"), len(glb.doc.get("materials", [])), "material")]
            normal_mapped = "normalTexture" in material
            normal_uv = None
            if normal_mapped:
                normal_uv = material["normalTexture"].get("texCoord", 0)
                _require(type(normal_uv) is int and normal_uv in (0, 1), "Normal map requires exact integer UV index 0 or 1")
            _require("NORMAL" in attrs, "Portable surface requires normals")
            normals = surface_accessor(attrs["NORMAL"], "VEC3", (5126,), "normal")
            normal_error = max(abs(sum(value*value for value in row)-1) for row in normals)
            _require(normal_error < .001, "Surface normals are not unit vectors")
            _require(not normal_mapped or "TEXCOORD_"+str(normal_uv) in attrs and "TANGENT" in attrs,
                     f"Normal-mapped material requires UV{normal_uv} and tangents")
            verified_uv_sets = []
            for uv_index in (0, 1):
                name = "TEXCOORD_"+str(uv_index)
                if name in attrs:
                    surface_accessor(attrs[name], "VEC2", (5121, 5123, 5126), "UV"+str(uv_index))
                    verified_uv_sets.append(uv_index)
            tangent_error, dot_error = 0., 0.
            if "TANGENT" in attrs:
                tangents = surface_accessor(attrs["TANGENT"], "VEC4", (5126,), "tangent")
                tangent_error = max(abs(sum(value*value for value in row[:3])-1) for row in tangents)
                dot_error = max(abs(sum(normal[i]*tangent[i] for i in range(3))) for normal, tangent in zip(normals, tangents))
                _require(tangent_error < .001, "Surface tangents are not unit vectors")
                _require(dot_error < .002, "Surface tangent is not orthogonal to normal")
                _require(all(abs(abs(row[3])-1) < 1e-6 for row in tangents), "Invalid tangent handedness")
            primitives.append({"vertices": count, "material": material.get("name"), "normal_mapped": normal_mapped,
                               "normal_map_uv_set": normal_uv, "verified_uv_sets": verified_uv_sets,
                               "normal_squared_length_error": normal_error, "tangent_squared_length_error": tangent_error,
                               "normal_tangent_dot_error": dot_error})
    _require(primitives, "No surface primitives to verify")
    return {"schema_version": 1, "status": "validated_render_attributes", "sha256": glb.hash,
            "vertices": vertices, "primitives": primitives, "decoded_scalar_values": glb.value_count,
            "checks": ["finite normals/UV0/UV1/tangents", "matching counts", "unit normals/tangents",
                       "orthogonal normal/tangent frames", "tangent handedness", "selected normal-map UV0/UV1 binding"],
            "tangent_uv_derivation_verified": False, "weft_runtime_verified": False,
            "limitations": ["Finite/unit/orthogonal frames and selected UV binding do not prove tangents derive from that UV set.",
                            "This HD check does not change Weft's diffuse UV0-only contract; normal-map/UV1 assets remain unsupported there."],
            "limits": {"vertices": MAX_VERTICES, "file_bytes": MAX_FILE_BYTES,
                       "decoded_surface_values": surface_value_limit, "decoded_global_values": MAX_ACCESSOR_VALUES,
                       "normal_map_uv_indices": [0, 1],
                       "normal_tangent_squared_length_tolerance": .001, "orthogonality_tolerance": .002}}


def unsupported_3d(subject: str, reason: str) -> dict:
    return {
        "schema_version": 1,
        "status": "unsupported",
        "subject": str(subject),
        "reason": str(reason),
        "geometry_created": False,
        "rig_created": False,
        "animation_verified": False,
        "weft_runtime_verified": False,
        "required_inputs": [
            "Authored or licensed GLB mesh, materials, skeleton and skin weights",
            "Units, axes, rest pose and explicit bone-to-simulated-joint mapping",
            "Authoritative target joint schema and attachment specification",
            "Disclosure of inferred surfaces, anatomy and rig decisions",
        ],
        "boundary": "2D images and videos do not specify hidden surfaces or a controllable rig.",
    }


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _index(value, size, label):
    _require(type(value) is int and 0 <= value < size, f"Invalid {label} index")
    return value


def _vector(value, size, label):
    _require(isinstance(value, (list, tuple)) and len(value) == size, f"Invalid {label}")
    _require(all(type(x) in (float, int) and math.isfinite(x) for x in value),
             f"Non-finite {label}")
    return tuple(float(x) for x in value)


def _matmul(a, b):
    # glTF stores matrices column-major.
    return tuple(sum(a[k * 4 + row] * b[col * 4 + k] for k in range(4))
                 for col in range(4) for row in range(4))


def _transform(matrix, point):
    return tuple(sum(matrix[col * 4 + row] * point[col] for col in range(3))
                 + matrix[12 + row] for row in range(3))


def _determinant3(matrix):
    a, b, c = matrix[0], matrix[4], matrix[8]
    d, e, f = matrix[1], matrix[5], matrix[9]
    g, h, i = matrix[2], matrix[6], matrix[10]
    return a*(e*i-f*h)-b*(d*i-f*g)+c*(d*h-e*g)


def _local_matrix(node):
    if "matrix" in node:
        _require(not any(k in node for k in ("translation", "rotation", "scale")),
                 "Node matrix and TRS cannot coexist")
        result = _vector(node["matrix"], 16, "node matrix")
        _require(all(abs(result[i] - value) < 1e-6
                     for i, value in ((3, 0), (7, 0), (11, 0), (15, 1))),
                 "Node matrix must be affine")
        _require(abs(_determinant3(result)) > 1e-12, "Singular node matrix")
        return result
    t = _vector(node.get("translation", (0, 0, 0)), 3, "translation")
    x, y, z, w = _vector(node.get("rotation", (0, 0, 0, 1)), 4, "rotation")
    _require(abs(x*x + y*y + z*z + w*w - 1) < 1e-4, "Rotation must be a unit quaternion")
    sx, sy, sz = _vector(node.get("scale", (1, 1, 1)), 3, "scale")
    _require(min(abs(sx), abs(sy), abs(sz)) > 1e-8, "Zero node scale")
    return (
        (1-2*(y*y+z*z))*sx, 2*(x*y+z*w)*sx, 2*(x*z-y*w)*sx, 0.,
        2*(x*y-z*w)*sy, (1-2*(x*x+z*z))*sy, 2*(y*z+x*w)*sy, 0.,
        2*(x*z+y*w)*sz, 2*(y*z-x*w)*sz, (1-2*(x*x+y*y))*sz, 0.,
        t[0], t[1], t[2], 1.,
    )


def _world_matrices(nodes, parents, overrides=None):
    overrides = overrides or {}
    result = {}

    def world(i):
        if i not in result:
            node = dict(nodes[i])
            node.update(overrides.get(i, {}))
            local = _local_matrix(node)
            result[i] = _matmul(world(parents[i]), local) if i in parents else local
        return result[i]

    return [world(i) for i in range(len(nodes))]


def _slerp(a, b, fraction):
    dot = sum(x*y for x, y in zip(a, b))
    if dot < 0:
        b, dot = tuple(-x for x in b), -dot
    if dot > .9995:
        mixed = tuple(x+(y-x)*fraction for x, y in zip(a, b))
        norm = math.sqrt(sum(x*x for x in mixed))
        return tuple(x/norm for x in mixed)
    theta = math.acos(min(1., max(-1., dot)))
    d = math.sin(theta)
    return tuple((math.sin((1-fraction)*theta)*x + math.sin(fraction*theta)*y)/d
                 for x, y in zip(a, b))


class _GLB:
    def __init__(self, path):
        path = Path(path)
        _require(path.is_file() and not path.is_symlink(), "GLB must be a regular local file")
        _require(20 <= path.stat().st_size <= MAX_FILE_BYTES, "GLB file size outside bounds")
        data = path.read_bytes()
        magic, version, length = struct.unpack_from("<4sII", data)
        _require(magic == b"glTF" and version == 2 and length == len(data), "Invalid GLB header")
        chunks, offset = [], 12
        while offset < len(data):
            _require(offset + 8 <= len(data), "Truncated GLB chunk header")
            size, kind = struct.unpack_from("<II", data, offset)
            offset += 8
            _require(size % 4 == 0 and offset + size <= len(data), "Invalid GLB chunk length")
            chunks.append((kind, data[offset:offset+size]))
            _require(len(chunks) <= 2, "Extra GLB chunks are unsupported")
            offset += size
        _require(len(chunks) == 2 and chunks[0][0] == 0x4E4F534A
                 and chunks[1][0] == 0x004E4942, "Require one JSON and one BIN GLB chunk")
        _require(len(chunks[0][1]) <= MAX_JSON_BYTES, "GLB JSON exceeds bound")
        try:
            self.doc = json.loads(chunks[0][1])
        except (ValueError, UnicodeDecodeError, RecursionError) as exc:
            raise ValueError("Invalid GLB JSON") from exc
        _require(isinstance(self.doc, dict), "GLB JSON must be an object")
        self.binary = chunks[1][1]
        self.hash = hashlib.sha256(data).hexdigest()
        self.cache = {}
        self.value_count = 0
        doc = self.doc
        pending = [doc]
        while pending:
            item = pending.pop()
            if isinstance(item, dict):
                _require(not item.get("extensions"), "glTF extension objects are unsupported")
                pending.extend(item.values())
            elif isinstance(item, list):
                pending.extend(item)
        _require(doc.get("asset", {}).get("version") == "2.0", "Require glTF 2.0 asset")
        _require(not doc.get("extensionsRequired") and not doc.get("extensionsUsed"),
                 "glTF extensions are unsupported by this bounded validator")
        _require(len(doc.get("buffers", [])) == 1, "Require one embedded buffer")
        buffer = doc["buffers"][0]
        _require("uri" not in buffer, "External/data buffer URIs are unsupported")
        _require(type(buffer.get("byteLength")) is int and 0 < buffer["byteLength"] <= len(self.binary)
                 and len(self.binary)-buffer["byteLength"] <= 3, "Invalid embedded buffer length")
        self.binary = self.binary[:buffer["byteLength"]]
        for key, limit in (("nodes", MAX_NODES), ("skins", 16), ("meshes", 64),
                           ("accessors", 1024), ("bufferViews", 1024), ("animations", 16),
                           ("materials", 64), ("images", 64), ("textures", 64), ("scenes", 16)):
            value = doc.get(key, [])
            _require(isinstance(value, list) and len(value) <= limit
                     and all(isinstance(x, dict) for x in value), f"Invalid or oversized {key}")
        for image in doc.get("images", []):
            _require("uri" not in image, "External/data image URIs are unsupported")
            _index(image.get("bufferView"), len(doc.get("bufferViews", [])), "image bufferView")
            _require(image.get("mimeType") in ("image/png", "image/jpeg"), "Unsupported image MIME type")
        for view in doc.get("bufferViews", []):
            offset, length = view.get("byteOffset", 0), view.get("byteLength")
            _require(view.get("buffer") == 0 and type(offset) is int and type(length) is int
                     and offset >= 0 and length > 0 and offset+length <= len(self.binary),
                     "Invalid bufferView range")
        for texture in doc.get("textures", []):
            _index(texture.get("source"), len(doc.get("images", [])), "texture source")
        for material in doc.get("materials", []):
            pbr = material.get("pbrMetallicRoughness", {})
            factor = _vector(pbr.get("baseColorFactor", [1, 1, 1, 1]), 4, "base color")
            _require(all(0 <= x <= 1 for x in factor), "Base color outside [0,1]")
            for key in ("metallicFactor", "roughnessFactor"):
                value = pbr.get(key, 1.)
                _require(type(value) in (int, float) and 0 <= value <= 1, "Invalid PBR factor")
            for container, keys in ((pbr, ("baseColorTexture", "metallicRoughnessTexture")),
                                    (material, ("normalTexture", "occlusionTexture", "emissiveTexture"))):
                for key in keys:
                    if key in container:
                        _index(container[key].get("index"), len(doc.get("textures", [])), "material texture")

    def accessor(self, index, kind=None, components=None, *, count_limit=MAX_VERTICES):
        _index(index, len(self.doc.get("accessors", [])), "accessor")
        a = self.doc["accessors"][index]
        _require(not a.get("sparse"), "Sparse accessors are unsupported")
        _require(kind is None or a.get("type") == kind, f"Accessor must be {kind}")
        _require(components is None or a.get("componentType") in components,
                 "Unsupported accessor component type")
        _require(a.get("type") in _WIDTHS and a.get("componentType") in _COMPONENTS,
                 "Unsupported accessor shape or component type")
        width = _WIDTHS[a["type"]]
        fmt, size = _COMPONENTS[a["componentType"]]
        count = a.get("count")
        _require(type(count_limit) is int and count_limit in (MAX_VERTICES, MAX_TRIANGLE_INDICES),
                 "Unsupported accessor count limit")
        _require(count_limit == MAX_VERTICES or kind == "SCALAR" and a["componentType"] in (5121, 5123, 5125),
                 "Expanded accessor count is only for triangle indices")
        _require(type(count) is int and 0 < count <= count_limit, "Accessor count outside bounds")
        # A cached wide index accessor must not bypass a later call's stricter limit.
        if index in self.cache:
            _require(len(self.cache[index]) == count, "Cached accessor count differs from metadata")
            return self.cache[index]
        self.value_count += count * width
        _require(self.value_count <= MAX_ACCESSOR_VALUES, "Decoded accessor values exceed bound")
        view = self.doc.get("bufferViews", [])[_index(a.get("bufferView"),
                    len(self.doc.get("bufferViews", [])), "bufferView")]
        _require(view.get("buffer") == 0, "Invalid buffer index")
        begin, length = view.get("byteOffset", 0), view.get("byteLength")
        start, stride = a.get("byteOffset", 0), view.get("byteStride", width*size)
        _require(all(type(x) is int and x >= 0 for x in (begin, length, start, stride)),
                 "Invalid accessor byte range")
        _require(stride >= width*size and stride <= 252 and stride % size == 0,
                 "Invalid accessor stride")
        _require(begin+length <= len(self.binary) and start+(count-1)*stride+width*size <= length
                 and (begin+start) % size == 0, "Accessor exceeds buffer or is unaligned")
        unpack = struct.Struct("<" + fmt*width)
        values = [unpack.unpack_from(self.binary, begin+start+i*stride) for i in range(count)]
        _require(all(math.isfinite(x) for row in values for x in row), "Non-finite accessor value")
        if a.get("normalized"):
            _require(a["componentType"] in (5121, 5123), "Unsupported normalized accessor")
            maximum = 255 if a["componentType"] == 5121 else 65535
            values = [tuple(x/maximum for x in row) for row in values]
        self.cache[index] = values
        return values


def _contract(contract, bones, nodes, parents):
    _require(isinstance(contract, dict) and contract.get("schema_version") == 1,
             "Require 3D contract schema_version 1")
    _require(contract.get("units") == "meters", "Contract units must be meters")
    _require(contract.get("axes") == {"up": "+Y", "forward": "+Z", "handedness": "right"},
             "Contract axes must declare right-handed +Y-up +Z-forward")
    _require(isinstance(contract.get("rest_pose"), str) and bool(contract["rest_pose"].strip()),
             "Contract requires an explicit rest_pose description")
    assumptions = contract.get("authored_assumptions")
    _require(isinstance(assumptions, list) and all(isinstance(x, str) for x in assumptions),
             "Contract must disclose authored_assumptions, even if empty")
    mapping = contract.get("bone_to_simulated_joint")
    _require(isinstance(mapping, dict) and set(mapping) == set(bones),
             "Bone mapping must exactly cover named skin joints")
    target = contract.get("target")
    _require(isinstance(target, dict) and isinstance(target.get("name"), str)
             and bool(target["name"].strip()), "Contract requires a target name")
    names = target.get("joint_names")
    _require(isinstance(names, list) and all(isinstance(x, str) and x for x in names)
             and len(set(names)) == len(names), "Target must list unique joint_names")
    _require(all(isinstance(x, str) and x in names for x in mapping.values()),
             "Mapped simulated joint absent from supplied target joint list")
    _require(len(set(mapping.values())) == len(mapping), "Multiple bones map to one simulated joint")
    attachments = contract.get("attachments")
    _require(isinstance(attachments, list) and len(attachments) <= 16,
             "Contract requires an attachments list, even if empty")
    node_names = {n.get("name"): i for i, n in enumerate(nodes) if n.get("name")}
    for attachment in attachments:
        _require(isinstance(attachment, dict) and attachment.get("node") in node_names,
                 "Attachment must identify an existing named node")
        bone = attachment.get("parent_bone")
        _require(bone in bones and attachment.get("mode") == "rigid_child",
                 "Supported attachment mode is rigid_child of a named skin joint")
        _require(parents.get(node_names[attachment["node"]]) == node_names[bone],
                 "Rigid attachment must be a direct child of its declared bone")
        _vector(attachment.get("translation_m"), 3, "attachment translation_m")
        _vector(attachment.get("rotation_xyzw"), 4, "attachment rotation_xyzw")
        node = nodes[node_names[attachment["node"]]]
        _require("matrix" not in node and "skin" not in node, "Rigid attachment must use unskinned TRS")
        _require(all(abs(a-b) < 1e-5 for a, b in zip(node.get("translation", [0, 0, 0]), attachment["translation_m"]))
                 and all(abs(a-b) < 1e-5 for a, b in zip(node.get("rotation", [0, 0, 0, 1]), attachment["rotation_xyzw"])),
                 "Attachment transform differs from contract")
        _require(isinstance(attachment.get("placement"), str) and attachment["placement"].strip(),
                 "Attachment must describe placement and orientation assumptions")
    return mapping


def _animation_tracks(glb, nodes, joint_indices):
    result = []
    for animation in glb.doc.get("animations", []):
        channels, samplers = animation.get("channels", []), animation.get("samplers", [])
        _require(isinstance(channels, list) and isinstance(samplers, list)
                 and 0 < len(channels) <= 128 and 0 < len(samplers) <= 128,
                 "Animation channel/sampler count outside bounds")
        tracks, seen, keyed_times = [], set(), set()
        for channel in channels:
            target = channel.get("target", {})
            node = _index(target.get("node"), len(nodes), "animation node")
            path = target.get("path")
            _require(path in ("translation", "rotation", "scale"), "Only skeletal TRS animation is supported")
            _require("matrix" not in nodes[node], "Animated nodes must use TRS")
            _require((node, path) not in seen, "Duplicate animation target")
            seen.add((node, path))
            sampler = samplers[_index(channel.get("sampler"), len(samplers), "animation sampler")]
            interpolation = sampler.get("interpolation", "LINEAR")
            _require(interpolation in ("LINEAR", "STEP"), "Only LINEAR/STEP interpolation is supported")
            times = [v[0] for v in glb.accessor(sampler.get("input"), "SCALAR", (5126,))]
            _require(len(times) <= 4096 and 0 <= times[0] <= times[-1] <= MAX_ANIMATION_SECONDS
                     and all(b > a for a, b in zip(times, times[1:])),
                     "Animation times must increase within 120 seconds")
            values = glb.accessor(sampler.get("output"), "VEC4" if path == "rotation" else "VEC3", (5126,))
            _require(len(times) == len(values), "Animation input/output count mismatch")
            if path == "rotation":
                _require(all(abs(sum(x*x for x in q)-1) < 1e-4 for q in values),
                         "Animation rotations must be unit quaternions")
            if path == "scale":
                _require(all(min(abs(x) for x in row) > 1e-8 for row in values), "Zero animated scale")
            tracks.append((node, path, interpolation, times, values))
            keyed_times.update(times)
        _require(any(t[0] in joint_indices for t in tracks), "Animation has no skin-joint channels")
        ordered = sorted(keyed_times)
        # Key endpoints plus segment midpoints reveal motion without decoding every frame.
        samples = sorted(set(ordered[::max(1, len(ordered)//16)] + [ordered[-1]]
                             + [(a+b)/2 for a, b in zip(ordered, ordered[1:])][::max(1, len(ordered)//16)]))
        if len(samples) > 35:
            samples = [samples[round(i*(len(samples)-1)/34)] for i in range(35)]
        result.append((animation.get("name", f"animation_{len(result)}"), tracks, samples))
    return result


def _pose(tracks, time):
    overrides = {}
    for node, path, interpolation, times, values in tracks:
        if len(times) == 1:
            overrides.setdefault(node, {})[path] = values[0]
            continue
        index = min(len(times)-2, max(0, bisect.bisect_right(times, time)-1))
        fraction = min(1., max(0., (time-times[index])/(times[index+1]-times[index])))
        if interpolation == "STEP":
            value = values[index+1] if fraction == 1 else values[index]
        elif path == "rotation":
            value = _slerp(values[index], values[index+1], fraction)
        else:
            value = tuple(a+(b-a)*fraction for a, b in zip(values[index], values[index+1]))
        overrides.setdefault(node, {})[path] = value
    return overrides


def _skin_points(world, samples):
    result = []
    for point, weights, joints, skin, inverse in samples:
        blended = [0., 0., 0.]
        for weight, joint in zip(weights, joints):
            if weight:
                moved = _transform(_matmul(world[skin[joint]], inverse[joint]), point)
                for axis in range(3):
                    blended[axis] += weight*moved[axis]
        result.append(tuple(blended))
    return result


def validate_glb(path, contract) -> dict:
    """Load a supported GLB subset and measure skin movement; never certify Weft."""
    try:
        return _validate_glb(path, contract)
    except (KeyError, TypeError, IndexError, OSError, struct.error, RecursionError, OverflowError) as exc:
        raise ValueError(f"Malformed or unsupported GLB/contract: {type(exc).__name__}") from exc


def _validate_glb(path, contract):
    glb = _GLB(path)
    doc, nodes = glb.doc, glb.doc.get("nodes", [])
    _require(nodes and doc.get("meshes") and doc.get("materials") and doc.get("skins"),
             "Controllable 3D requires nodes, meshes, materials, skeleton and skin weights")
    parents = {}
    named_nodes = [n.get("name") for n in nodes if n.get("name")]
    _require(all(isinstance(name, str) for name in named_nodes)
             and len(set(named_nodes)) == len(named_nodes), "Named nodes must have unique names")
    for i, node in enumerate(nodes):
        _local_matrix(node)
        children = node.get("children", [])
        _require(isinstance(children, list) and len(children) <= MAX_NODES, "Invalid child list")
        for child in children:
            _index(child, len(nodes), "child")
            _require(child not in parents, "Node has multiple parents")
            parents[child] = i
    for i in range(len(nodes)):
        chain, current = set(), i
        while current in parents:
            _require(current not in chain, "Cyclic node hierarchy")
            chain.add(current)
            current = parents[current]
    scenes = doc.get("scenes", [])
    scene = scenes[_index(doc.get("scene", 0), len(scenes), "scene")]
    roots = scene.get("nodes", [])
    _require(isinstance(roots, list) and roots, "Default scene has no root nodes")
    reachable, pending = set(), list(roots)
    while pending:
        node = _index(pending.pop(), len(nodes), "scene node")
        if node not in reachable:
            reachable.add(node)
            pending.extend(nodes[node].get("children", []))
    _require(all(root not in parents for root in roots), "Scene roots cannot have parents")
    skins, bone_indices, skeleton_roots = [], set(), []
    for skin in doc["skins"]:
        joints = skin.get("joints", [])
        _require(isinstance(joints, list) and 0 < len(joints) <= 128
                 and len(set(joints)) == len(joints), "Invalid skin joints")
        for joint in joints:
            _index(joint, len(nodes), "joint")
            _require(joint in reachable, "Skin joint absent from default scene")
        chains = []
        for joint in joints:
            chain = [joint]
            while joint in parents:
                joint = parents[joint]
                chain.append(joint)
            chains.append(chain)
        if "skeleton" in skin:
            skeleton = _index(skin["skeleton"], len(nodes), "skin skeleton")
            _require(all(skeleton in chain for chain in chains),
                     "Skeleton is not an ancestor of every joint")
        else:
            # Exporters may omit this optional field; preserve the actual hierarchy.
            common = set.intersection(*(set(chain) for chain in chains))
            _require(common, "Skin joints lack a common skeleton ancestor")
            skeleton = next(index for index in chains[0] if index in common)
        skeleton_roots.append({"node": skeleton, "name": nodes[skeleton].get("name"),
                               "source": "explicit" if "skeleton" in skin else "derived_common_ancestor"})
        inverse = glb.accessor(skin.get("inverseBindMatrices"), "MAT4", (5126,))
        _require(len(inverse) == len(joints), "Inverse bind matrix count differs from joint count")
        _require(all(all(abs(m[i]-v) < 1e-5 for i, v in ((3, 0), (7, 0), (11, 0), (15, 1)))
                     for m in inverse), "Inverse bind matrices must be affine")
        _require(all(abs(_determinant3(m)) > 1e-12 for m in inverse), "Singular inverse bind matrix")
        bone_indices.update(joints)
        skins.append((joints, inverse))
    bones = [nodes[i].get("name") for i in sorted(bone_indices)]
    _require(all(isinstance(name, str) and name for name in bones) and len(set(bones)) == len(bones),
             "Skin joints need unique nonempty names")
    mapping = _contract(contract, bones, nodes, parents)
    samples, vertex_count, primitive_count, skinned_nodes = [], 0, 0, 0
    for i, node in enumerate(nodes):
        if "mesh" not in node:
            _require("skin" not in node, "Skin without mesh")
            continue
        _require(i in reachable, "Mesh absent from default scene")
        mesh = doc["meshes"][_index(node["mesh"], len(doc["meshes"]), "mesh")]
        primitives = mesh.get("primitives", [])
        _require(isinstance(primitives, list) and 0 < len(primitives) <= 32, "Invalid primitive count")
        skin = skins[_index(node["skin"], len(skins), "skin")] if "skin" in node else None
        if skin:
            skinned_nodes += 1
        for primitive in primitives:
            primitive_count += 1
            _require(primitive_count <= 128 and primitive.get("mode", 4) == 4,
                     "Only bounded triangle primitives are supported")
            _require(not primitive.get("targets"), "Morph targets are unsupported")
            _index(primitive.get("material"), len(doc["materials"]), "material")
            attrs = primitive.get("attributes", {})
            positions = glb.accessor(attrs.get("POSITION"), "VEC3", (5126,))
            vertex_count += len(positions)
            _require(vertex_count <= MAX_VERTICES, "Aggregate vertex count exceeds bound")
            if "indices" in primitive:
                indices = glb.accessor(primitive["indices"], "SCALAR", (5121, 5123, 5125),
                                       count_limit=MAX_TRIANGLE_INDICES)
                _require(len(indices) % 3 == 0 and all(type(v[0]) is int and v[0] < len(positions) for v in indices),
                         "Invalid triangle indices")
            else:
                _require(len(positions) % 3 == 0, "Nonindexed triangles require a multiple of three vertices")
            if not skin:
                _require(not any(k.startswith(("JOINTS_", "WEIGHTS_")) for k in attrs),
                         "Skin attributes without a skin")
                continue
            _require(not any(k.startswith(("JOINTS_", "WEIGHTS_"))
                             and k not in ("JOINTS_0", "WEIGHTS_0") for k in attrs),
                     "More than four skin influences is unsupported")
            joints = glb.accessor(attrs.get("JOINTS_0"), "VEC4", (5121, 5123))
            weights = glb.accessor(attrs.get("WEIGHTS_0"), "VEC4", (5121, 5123, 5126))
            weight_accessor = doc["accessors"][attrs["WEIGHTS_0"]]
            _require(weight_accessor["componentType"] == 5126 or weight_accessor.get("normalized") is True,
                     "Integer skin weights must be normalized")
            _require(len(joints) == len(weights) == len(positions), "Skin attribute count mismatch")
            _require(all(all(type(v) is int and v < len(skin[0]) for v in row) for row in joints),
                     "Skin joint index outside skin")
            _require(all(min(row) >= 0 and abs(sum(row)-1) <= .001 for row in weights),
                     "Skin weights must be nonnegative and sum to one")
            stride = max(1, math.ceil(len(positions)/MAX_MOTION_PROBES))
            samples.extend((positions[v], weights[v], joints[v], skin[0], skin[1])
                           for v in range(0, len(positions), stride))
    _require(skinned_nodes > 0 and samples, "No skinned geometry is loaded")
    samples = samples[::max(1, math.ceil(len(samples)/MAX_MOTION_PROBES))]
    rest = _skin_points(_world_matrices(nodes, parents), samples)
    _require(all(math.isfinite(v) for point in rest for v in point), "Non-finite skinned rest pose")
    animations = []
    for name, tracks, times in _animation_tracks(glb, nodes, bone_indices):
        first, displacement = None, 0.
        for time in times:
            points = _skin_points(_world_matrices(nodes, parents, _pose(tracks, time)), samples)
            _require(all(math.isfinite(v) for point in points for v in point), "Non-finite skinned motion")
            if first is None:
                first = points
            displacement = max(displacement, max(math.dist(a, b) for a, b in zip(first, points)))
        animations.append({"name": name, "sample_count": len(times),
                           "sample_times_seconds": times,
                           "max_sampled_displacement_m": displacement,
                           "motion_observed": displacement > 1e-6,
                           "loop": "not_claimed"})
    return {
        "schema_version": 1,
        "status": "validated_local_structure",
        "sha256": glb.hash,
        "loading_verified": True,
        "mesh_vertices": vertex_count,
        "primitive_count": primitive_count,
        "material_count": len(doc["materials"]),
        "skin_count": len(skins),
        "skeleton_roots": skeleton_roots,
        "bones": bones,
        "bone_to_simulated_joint": mapping,
        "units": contract["units"],
        "axes": contract["axes"],
        "rest_pose": contract["rest_pose"],
        "attachments": contract["attachments"],
        "authored_assumptions": contract["authored_assumptions"],
        "motion": {"method": "CPU linear-blend skinning at bounded key/midpoint samples",
                   "probe_vertices": len(samples), "animations": animations,
                   "animation_verified": bool(animations) and all(a["motion_observed"] for a in animations)},
        "weft_runtime_verified": False,
        "compatibility": "Mapping checked against supplied target joint names only; target runtime not exercised.",
        "visual_review": "required; materials/textures are not rendered by this validator",
        "limits": {"file_bytes": MAX_FILE_BYTES, "vertices": MAX_VERTICES,
                   "triangle_index_elements_per_accessor": MAX_TRIANGLE_INDICES,
                   "nodes": MAX_NODES, "accessor_values": MAX_ACCESSOR_VALUES,
                   "motion_probe_vertices": MAX_MOTION_PROBES},
    }
