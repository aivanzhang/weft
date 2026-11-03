"""Bounded glTF subset for visual avatars; never loads code, URLs or physics."""

from io import BytesIO
import hashlib
import json
import math
from pathlib import PurePosixPath
import re
import struct
import time

import numpy as np
from PIL import Image


RENDER_PROFILE = "weft-avatar-render-v1"
RETARGET_PROFILE = "weft-avatar-retarget-v1"
REST_RETARGET_PROFILE = "weft-avatar-retarget-v2"
MAX_NODES, MAX_JOINTS, MAX_VERTICES, MAX_TRIANGLES, MAX_PRIMITIVES = 128, 64, 20000, 40000, 16
MAX_TEXTURES, MAX_TEXTURE_SIDE, MAX_TEXTURE_PIXELS = 8, 2048, 8 * 1024 * 1024
MAX_RENDER_BYTES = 8 * 1024 * 1024
BODY_FRAMES = frozenset(("mobile_root", "torso", *(
    side + "_" + joint + "_body" for side in ("left", "right") for joint in ("shoulder", "elbow", "hip", "knee"))))
COMPONENTS = {5120: np.dtype("i1"), 5121: np.dtype("u1"), 5122: np.dtype("<i2"),
              5123: np.dtype("<u2"), 5125: np.dtype("<u4"), 5126: np.dtype("<f4")}
WIDTHS = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT4": 16}


class AvatarAssetError(ValueError):
    def __init__(self, message, *, code="invalid_avatar"):
        super().__init__(message)
        self.code = code


def fail(message, *, unsupported=False):
    raise AvatarAssetError(message, code="unsupported_avatar" if unsupported else "invalid_avatar")


def deadline_check(deadline):
    if time.monotonic() > deadline:
        raise AvatarAssetError("Avatar processing exceeded its deadline", code="avatar_deadline")


def json_document(raw, *, maximum=2 * 1024 * 1024):
    if not isinstance(raw, bytes) or len(raw) > maximum:
        fail("Avatar JSON exceeds its byte bound")

    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                fail("Avatar JSON has duplicate fields")
            result[key] = value
        return result

    try:
        value = json.loads(raw.decode("utf-8"), object_pairs_hook=unique,
                           parse_constant=lambda _: fail("Avatar JSON must contain finite numbers"))
    except (UnicodeError, json.JSONDecodeError, RecursionError) as error:
        raise AvatarAssetError("Avatar JSON is malformed or excessively nested") from error
    pending = [(value, 0)]
    count = 0
    while pending:
        item, depth = pending.pop()
        count += 1
        if depth > 24 or count > 200000:
            fail("Avatar JSON exceeds its nesting or element bound")
        if isinstance(item, dict):
            pending.extend((v, depth + 1) for v in item.values())
        elif isinstance(item, list):
            pending.extend((v, depth + 1) for v in item)
        elif type(item) is float and not math.isfinite(item):
            fail("Avatar JSON must contain finite numbers")
    if not isinstance(value, dict):
        fail("Avatar JSON must be an object")
    return value


def safe_relative(name):
    if (not isinstance(name, str) or len(name) > 240 or not re.fullmatch(r"[A-Za-z0-9_./-]+", name)
            or len(name.split("/")) > 8
            or any(part in ("", ".", "..", "__pycache__") or part.startswith(".") for part in name.split("/"))):
        fail("Avatar resource must be a confined relative path; URLs and traversal are unsupported")
    return name


def integer(value, low, high, label):
    if type(value) is not int or not low <= value <= high:
        fail(label + " is outside its integer bound")
    return value


def number(value, low, high, label):
    if type(value) not in (int, float) or not math.isfinite(value) or not low <= value <= high:
        fail(label + " must be finite and bounded")
    return float(value)


def vector(value, count, label, bound=1e6):
    if not isinstance(value, list) or len(value) != count:
        fail(label + " has the wrong dimensions")
    return [number(v, -bound, bound, label) for v in value]


def quaternion(value, label):
    result = vector(value, 4, label, 1)
    if not math.isclose(sum(v * v for v in result), 1., abs_tol=1e-6, rel_tol=0):
        fail(label + " must be a unit xyzw quaternion")
    return result


def matrix(values, label):
    result = np.asarray(vector(values, 16, label), dtype=float).reshape((4, 4), order="F")
    if not np.allclose(result[3], [0, 0, 0, 1], atol=1e-7, rtol=0) or abs(np.linalg.det(result[:3, :3])) < 1e-12:
        fail(label + " must be an invertible affine matrix")
    return result


def _trs(node):
    if "matrix" in node:
        if any(key in node for key in ("translation", "rotation", "scale")):
            fail("glTF node cannot combine a matrix with TRS")
        result = matrix(node["matrix"], "Node rest matrix")
        scales = np.linalg.norm(result[:3, :3], axis=0)
        basis = result[:3, :3] / scales
        if (np.any(scales < 1e-6) or np.any(scales > 1e4) or np.linalg.det(basis) < 0
                or not np.allclose(basis.T @ basis, np.eye(3), atol=1e-5, rtol=0)):
            fail("Sheared, reflected or degenerate node transforms are unsupported", unsupported=True)
        return result
    x, y, z, w = quaternion(node.get("rotation", [0, 0, 0, 1]), "Node rotation")
    translation = vector(node.get("translation", [0, 0, 0]), 3, "Node translation")
    scale = vector(node.get("scale", [1, 1, 1]), 3, "Node scale", 1e4)
    if any(v < 1e-6 for v in scale):
        fail("Nonpositive or degenerate node scales are unsupported", unsupported=True)
    result = np.eye(4)
    result[:3, :3] = np.array([[1 - 2 * (y*y + z*z), 2 * (x*y - z*w), 2 * (x*z + y*w)],
                              [2 * (x*y + z*w), 1 - 2 * (x*x + z*z), 2 * (y*z - x*w)],
                              [2 * (x*z - y*w), 2 * (y*z + x*w), 1 - 2 * (x*x + y*y)]]) @ np.diag(scale)
    result[:3, 3] = translation
    return result


def _array(document, name, maximum, *, required=False):
    result = document.get(name, [])
    if not isinstance(result, list) or len(result) > maximum or (required and not result):
        fail(name + " exceeds its count bound or is missing")
    if any(not isinstance(item, dict) for item in result):
        fail(name + " must contain objects")
    return result


def _load_document(entrypoint, files):
    safe_relative(entrypoint)
    if entrypoint not in files or not isinstance(files[entrypoint], bytes) or len(files[entrypoint]) > 32 * 1024 * 1024:
        fail("Model entrypoint is missing or exceeds its byte bound")
    raw = files[entrypoint]
    binary = None
    if entrypoint.endswith(".glb"):
        if len(raw) < 20 or struct.unpack_from("<III", raw) != (0x46546C67, 2, len(raw)):
            fail("GLB header/version/length is invalid")
        chunks, offset = [], 12
        while offset < len(raw):
            if offset + 8 > len(raw):
                fail("GLB chunk header is truncated")
            size, kind = struct.unpack_from("<II", raw, offset)
            offset += 8
            if size % 4 or offset + size > len(raw) or len(chunks) >= 2:
                fail("GLB chunks exceed their bounds or supported layout")
            chunks.append((kind, raw[offset:offset + size]))
            offset += size
        if not chunks or chunks[0][0] != 0x4E4F534A or (len(chunks) == 2 and chunks[1][0] != 0x004E4942):
            fail("GLB requires JSON followed by an optional BIN chunk")
        document = json_document(chunks[0][1])
        binary = chunks[1][1] if len(chunks) == 2 else None
    elif entrypoint.endswith(".gltf"):
        document = json_document(raw)
    else:
        fail("Avatar entrypoint must be glTF 2.0 .glb or .gltf", unsupported=True)
    asset = document.get("asset")
    if not isinstance(asset, dict) or asset.get("version") != "2.0" or asset.get("minVersion", "2.0") != "2.0":
        fail("Only glTF 2.0 assets are supported", unsupported=True)
    allowed = {"KHR_materials_unlit"}
    used, required = document.get("extensionsUsed", []), document.get("extensionsRequired", [])
    if (not isinstance(used, list) or not isinstance(required, list) or any(not isinstance(x, str) for x in used + required)
            or len(used) != len(set(used)) or len(required) != len(set(required))
            or not set(required) <= set(used) or not set(used) <= allowed):
        fail("Unsupported or inconsistent glTF extensions", unsupported=True)
    material_nodes = {id(item) for item in document.get("materials", [])} if isinstance(document.get("materials", []), list) else set()
    pending = [document]
    while pending:
        item = pending.pop()
        if isinstance(item, dict):
            if "extensions" in item and (not isinstance(item["extensions"], dict)
                    or not set(item["extensions"]) <= set(used)
                    or bool(item["extensions"]) and id(item) not in material_nodes):
                fail("Undeclared or unsupported glTF extension", unsupported=True)
            pending.extend(item.values())
        elif isinstance(item, list):
            pending.extend(item)
    if document.get("cameras"):
        fail("Embedded cameras are unsupported in avatar assets", unsupported=True)
    return document, binary


class _Reader:
    def __init__(self, document, binary, entrypoint, files, deadline):
        self.document, self.files, self.entrypoint, self.deadline = document, files, entrypoint, deadline
        self.buffers, self.cache, self.decoded_components = [], {}, 0
        specs = _array(document, "buffers", 16, required=True)
        for i, spec in enumerate(specs):
            size = integer(spec.get("byteLength"), 1, 32 * 1024 * 1024, "Buffer byteLength")
            if "uri" in spec:
                raw = self.resource(spec["uri"])
                if len(raw) != size:
                    fail("External buffer byteLength differs from exact resource bytes")
            elif i == 0 and binary is not None:
                if not size <= len(binary) <= size + 3 or any(binary[size:]):
                    fail("GLB binary length or zero padding is invalid")
                raw = binary[:size]
            else:
                fail("Buffer is missing its confined resource")
            self.buffers.append(raw)
        if binary is not None and "uri" in specs[0]:
            fail("GLB BIN chunk must back its first buffer")
        self.views = _array(document, "bufferViews", 256, required=True)
        for view in self.views:
            index = integer(view.get("buffer"), 0, len(self.buffers) - 1, "bufferView buffer")
            offset = integer(view.get("byteOffset", 0), 0, len(self.buffers[index]), "bufferView offset")
            size = integer(view.get("byteLength"), 1, len(self.buffers[index]), "bufferView byteLength")
            if offset + size > len(self.buffers[index]) or view.get("target") not in (None, 34962, 34963):
                fail("bufferView exceeds its buffer or has an invalid target")
            if "byteStride" in view:
                stride = integer(view["byteStride"], 4, 252, "bufferView byteStride")
                if stride % 4:
                    fail("bufferView stride must be a multiple of four")
        self.accessors = _array(document, "accessors", 256, required=True)
        for index in range(len(self.accessors)):
            self.accessor(index)

    def resource(self, uri):
        safe_relative(uri)
        name = str(PurePosixPath(self.entrypoint).parent / uri)
        if name not in self.files:
            fail("glTF resource is missing from the sealed asset files: " + name)
        return self.files[name]

    def view_bytes(self, index):
        index = integer(index, 0, len(self.views) - 1, "bufferView index")
        view = self.views[index]
        start = view.get("byteOffset", 0)
        return self.buffers[view["buffer"]][start:start + view["byteLength"]]

    def accessor(self, index):
        deadline_check(self.deadline)
        index = integer(index, 0, len(self.accessors) - 1, "Accessor index")
        if index in self.cache:
            return self.cache[index]
        spec = self.accessors[index]
        if "sparse" in spec:
            fail("Sparse accessors are not supported by this avatar profile", unsupported=True)
        kind, component = spec.get("type"), spec.get("componentType")
        if not isinstance(kind, str) or kind not in WIDTHS or type(component) is not int or component not in COMPONENTS:
            fail("Unsupported accessor type", unsupported=True)
        if kind == "MAT4" and component != 5126:
            fail("Only float matrix accessors are supported", unsupported=True)
        count = integer(spec.get("count"), 1, 120000, "Accessor count")
        dtype, width = COMPONENTS[component], WIDTHS[kind]
        self.decoded_components += count * width
        if count * width > 480000 or self.decoded_components > 2 * 1024 * 1024:
            fail("Accessor exceeds its decoded component bound")
        view_index = integer(spec.get("bufferView"), 0, len(self.views) - 1, "Accessor bufferView")
        view = self.views[view_index]
        offset = integer(spec.get("byteOffset", 0), 0, view["byteLength"], "Accessor offset")
        stride = view.get("byteStride", dtype.itemsize * width)
        if (offset % dtype.itemsize or (view.get("byteOffset", 0) + offset) % dtype.itemsize
                or stride < dtype.itemsize * width or (count - 1) * stride + dtype.itemsize * width + offset > view["byteLength"]):
            fail("Accessor alignment, stride or range is invalid")
        normalized = spec.get("normalized", False)
        if type(normalized) is not bool or normalized and component not in (5120, 5121, 5122, 5123):
            fail("Accessor normalized flag is invalid")
        value = np.ndarray((count, width), dtype=dtype, buffer=self.buffers[view["buffer"]],
                           offset=view.get("byteOffset", 0) + offset, strides=(stride, dtype.itemsize))
        if not np.isfinite(value).all():
            fail("Accessor contains nonfinite values")
        for key, operation in (("min", np.min), ("max", np.max)):
            if key in spec:
                expected = vector(spec[key], width, "Accessor " + key, 1e12)
                if not np.allclose(operation(value, axis=0), expected, rtol=1e-6, atol=1e-7):
                    fail("Accessor declared bounds differ from its values")
        if normalized:
            maximum = np.iinfo(dtype).max
            value = np.maximum(value.astype(np.float32) / maximum, -1.)
        value.setflags(write=False)
        self.cache[index] = value
        return value

    def typed(self, index, kind, components, *, normalized=None):
        value = self.accessor(index)
        spec = self.accessors[index]
        if (spec["type"] != kind or spec["componentType"] not in components
                or normalized is not None and spec.get("normalized", False) is not normalized):
            fail("Accessor format does not match its semantic")
        return value

    def vertex(self, index):
        spec = self.accessors[index]
        view = self.views[spec["bufferView"]]
        stride = view.get("byteStride", WIDTHS[spec["type"]] * COMPONENTS[spec["componentType"]].itemsize)
        if (spec.get("byteOffset", 0) % 4 or (view.get("byteOffset", 0) + spec.get("byteOffset", 0)) % 4
                or stride % 4 or view.get("target") not in (None, 34962)):
            fail("Vertex attributes require four-byte alignment and vertex buffer targets")

    def nonvertex(self, index, *, indices=False):
        view = self.views[self.accessors[index]["bufferView"]]
        if "byteStride" in view or view.get("target") not in ((None, 34963) if indices else (None,)):
            fail("Nonvertex accessors cannot use vertex strides or incompatible buffer targets")


def _nodes(document):
    nodes = _array(document, "nodes", MAX_NODES, required=True)
    parents = [None] * len(nodes)
    local = []
    for index, node in enumerate(nodes):
        children = node.get("children", [])
        if not isinstance(children, list) or len(children) > MAX_NODES:
            fail("Node children exceed their bound")
        for child in children:
            integer(child, 0, len(nodes) - 1, "Child node")
            if child == index or parents[child] is not None:
                fail("Node hierarchy contains self-parenting or multiple parents")
            parents[child] = index
        if "camera" in node or "weights" in node:
            fail("Node cameras and morph weights are unsupported", unsupported=True)
        local.append(_trs(node))
    for index in range(len(nodes)):
        current, seen = index, set()
        while current is not None:
            if current in seen:
                fail("Node hierarchy contains a cycle")
            seen.add(current)
            current = parents[current]
    scenes = _array(document, "scenes", 16, required=True)
    chosen = integer(document.get("scene", 0), 0, len(scenes) - 1, "Default scene")
    for scene in scenes:
        roots = scene.get("nodes", [])
        if not isinstance(roots, list) or not roots or len(roots) > MAX_NODES:
            fail("Scene requires bounded root nodes")
        for node in roots:
            integer(node, 0, len(nodes) - 1, "Scene root")
            if parents[node] is not None:
                fail("Scene roots cannot be children of another node")
        if len(roots) != len(set(roots)):
            fail("Scene root nodes must be unique")
    reachable, pending = set(), list(scenes[chosen]["nodes"])
    while pending:
        node = pending.pop()
        if node not in reachable:
            reachable.add(node)
            pending.extend(nodes[node].get("children", []))
    return nodes, parents, local, reachable


def _retarget(mapping, nodes, parents, rendered, skin_joints):
    keys = {"profile", "root_node", "scale", "orientation_xyzw", "bindings"}
    if not isinstance(mapping, dict) or set(mapping) != keys or mapping.get("profile") not in (RETARGET_PROFILE, REST_RETARGET_PROFILE):
        fail("Avatar requires an explicit versioned retargeting contract")
    root = integer(mapping["root_node"], 0, len(nodes) - 1, "Retarget root")
    rest_relative = mapping["profile"] == REST_RETARGET_PROFILE
    result = {"profile": mapping["profile"], "root_node": root,
              "scale": number(mapping["scale"], 1e-4, 1000, "Retarget scale"),
              "orientation_xyzw": quaternion(mapping["orientation_xyzw"], "Retarget orientation"), "bindings": []}
    bindings = mapping["bindings"]
    if not isinstance(bindings, list) or not 1 <= len(bindings) <= MAX_NODES:
        fail("Retarget bindings exceed their bound or are missing")
    seen = set()
    for binding in bindings:
        if not isinstance(binding, dict) or set(binding) != ({"node", "body"} if rest_relative else {"node", "body", "translation", "rotation_xyzw"}):
            fail("Retarget binding has unknown or missing fields")
        node = integer(binding["node"], 0, len(nodes) - 1, "Retarget bone node")
        if node in seen or not isinstance(binding["body"], str) or binding["body"] not in BODY_FRAMES:
            fail("Retarget node is duplicated or simulation body is unsupported")
        seen.add(node)
        item = {"node": node, "body": binding["body"]}
        if not rest_relative:
            item.update(translation=vector(binding["translation"], 3, "Retarget offset", 100),
                        rotation_xyzw=quaternion(binding["rotation_xyzw"], "Retarget bone rotation"))
        result["bindings"].append(item)
    if root not in seen:
        fail("Retarget root must be explicitly mapped")
    for node in set(rendered) | set(skin_joints) | seen:
        current = node
        while current is not None and current != root:
            current = parents[current]
        if current != root:
            fail("All rendered, skinned and mapped nodes must descend from the retarget root")
    if rest_relative:
        roles = {item["body"]: item["node"] for item in result["bindings"]}
        if set(roles) != BODY_FRAMES or len(roles) != len(bindings) or roles["mobile_root"] != root:
            fail("Rest-relative avatars require exactly ten distinct body roles")
        for role, node in roles.items():
            if role != "mobile_root" and node not in skin_joints:
                fail("Rest-relative controlled bones must belong to the skin")
            expected = ("mobile_root" if role == "torso" else "torso" if "shoulder" in role else
                        role.replace("elbow", "shoulder") if "elbow" in role else
                        role.replace("knee", "hip") if "knee" in role else None)
            if expected:
                ancestor = parents[node]
                while ancestor is not None and ancestor not in seen:
                    ancestor = parents[ancestor]
                if ancestor != roles[expected]:
                    fail("Rest-relative limb topology is incompatible", unsupported=True)
    return result


def _rest_relative(mapping, parents, local, joints, inverse):
    if mapping["profile"] != REST_RETARGET_PROFILE:
        return
    world = {}
    def visit(index):
        if index not in world:
            parent = parents[index]
            world[index] = local[index] if parent is None else visit(parent) @ local[index]
        return world[index]
    for index in range(len(local)):
        value = visit(index)
        if not np.isfinite(value).all() or np.max(np.abs(value)) > 1e9:
            fail("Rest-relative accumulated transform exceeds its numeric bound", unsupported=True)
        lengths = np.linalg.norm(value[:3, :3], axis=0)
        if (not np.allclose(lengths, lengths[0], atol=1e-5, rtol=1e-5)
                or not np.allclose((value[:3, :3] / lengths).T @ (value[:3, :3] / lengths),
                                   np.eye(3), atol=1e-5, rtol=0)):
            fail("Rest-relative avatars require uniform nonsheared rest transforms", unsupported=True)
    bind = [world[node] @ np.asarray(value).reshape((4, 4), order="F") for node, value in zip(joints, inverse)]
    if any(not np.allclose(value, bind[0], atol=2e-5, rtol=2e-5) for value in bind[1:]):
        fail("Rest-relative skin default pose must match a common bind pose", unsupported=True)
    basis = _trs({"rotation": mapping["orientation_xyzw"], "scale": [mapping["scale"]] * 3})
    roles = {item["body"]: basis @ world[item["node"]] for item in mapping["bindings"]}
    shoulder_height = sum(roles[f"{side}_shoulder_body"][2, 3] for side in ("left", "right"))
    hip_height = sum(roles[f"{side}_hip_body"][2, 3] for side in ("left", "right"))
    if shoulder_height - hip_height <= 2e-6:
        fail("Rest-relative orientation must place shoulders above hips", unsupported=True)
    for side, sign in (("left", -1), ("right", 1)):
        for joint in ("shoulder", "hip"):
            delta = roles[f"{side}_{joint}_body"][0, 3] - roles["torso"][0, 3]
            if sign * delta <= 1e-6:
                fail("Rest-relative left/right mapping contradicts the simulation lateral convention", unsupported=True)
        for joint, parent in (("elbow", "shoulder"), ("knee", "hip")):
            if np.linalg.norm(roles[f"{side}_{joint}_body"][:3, 3] - roles[f"{side}_{parent}_body"][:3, 3]) <= 1e-6:
                fail("Rest-relative limb has coincident pivots", unsupported=True)


def _textures(document, reader):
    images = _array(document, "images", MAX_TEXTURES)
    samplers = _array(document, "samplers", MAX_TEXTURES)
    textures = _array(document, "textures", MAX_TEXTURES)
    output, files, pixels, decoded = [], {}, 0, []
    for image in images:
        deadline_check(reader.deadline)
        if ("uri" in image) == ("bufferView" in image):
            fail("Image must have exactly one local URI or bufferView")
        if "bufferView" in image and image.get("mimeType") not in ("image/png", "image/jpeg"):
            fail("Embedded images require an explicit supported MIME type")
        raw = reader.resource(image["uri"]) if "uri" in image else reader.view_bytes(image["bufferView"])
        if "bufferView" in image:
            view = reader.views[image["bufferView"]]
            if "byteStride" in view or "target" in view:
                fail("Embedded images cannot use vertex/index buffer layout")
        if len(raw) > 8 * 1024 * 1024:
            fail("Texture compressed bytes exceed their bound")
        try:
            with Image.open(BytesIO(raw)) as opened:
                if opened.format not in ("PNG", "JPEG") or getattr(opened, "n_frames", 1) != 1:
                    fail("Only static PNG/JPEG avatar textures are supported", unsupported=True)
                width, height = opened.size
                if not 1 <= width <= MAX_TEXTURE_SIDE or not 1 <= height <= MAX_TEXTURE_SIDE:
                    fail("Texture dimensions exceed the decoded-pixel bound")
                pixels += width * height
                if pixels > MAX_TEXTURE_PIXELS:
                    fail("Textures exceed the aggregate decoded-pixel bound")
                mime = "image/png" if opened.format == "PNG" else "image/jpeg"
                if image.get("mimeType", mime) != mime:
                    fail("Texture MIME type differs from its bytes")
                opened.verify()
            with Image.open(BytesIO(raw)) as opened:
                opened.load()
        except (OSError, SyntaxError, Image.DecompressionBombError) as error:
            raise AvatarAssetError("Avatar texture is malformed") from error
        digest = hashlib.sha256(raw).hexdigest()
        name = "textures/" + digest + (".png" if mime == "image/png" else ".jpg")
        files[name] = raw
        decoded.append({"path": name, "sha256": digest, "mime_type": mime, "width": width, "height": height})
    for texture in textures:
        source = integer(texture.get("source"), 0, len(images) - 1, "Texture source")
        sampler = {} if "sampler" not in texture else samplers[integer(texture["sampler"], 0, len(samplers) - 1, "Texture sampler")]
        values = {"wrap_s": sampler.get("wrapS", 10497), "wrap_t": sampler.get("wrapT", 10497),
                  "mag_filter": sampler.get("magFilter", 9729), "min_filter": sampler.get("minFilter", 9729)}
        if (any(type(v) is not int for v in values.values()) or values["wrap_s"] not in (33071, 33648, 10497)
                or values["wrap_t"] not in (33071, 33648, 10497) or values["mag_filter"] not in (9728, 9729)
                or values["min_filter"] not in (9728, 9729, 9984, 9985, 9986, 9987)):
            fail("Texture sampler is unsupported")
        output.append({**decoded[source], **values})
    return output, files


def _materials(document, texture_count):
    materials = _array(document, "materials", 32)
    result = []
    for material in materials:
        if any(key in material for key in ("normalTexture", "occlusionTexture", "emissiveTexture")):
            fail("Normal, occlusion and emissive texture channels are unsupported", unsupported=True)
        if material.get("emissiveFactor", [0, 0, 0]) != [0, 0, 0]:
            fail("Emissive materials are unsupported", unsupported=True)
        extensions = material.get("extensions", {})
        unlit = "KHR_materials_unlit" in extensions
        if unlit and extensions["KHR_materials_unlit"] != {}:
            fail("KHR_materials_unlit must be an empty extension object")
        pbr = material.get("pbrMetallicRoughness", {})
        if not isinstance(pbr, dict) or "metallicRoughnessTexture" in pbr:
            fail("Metallic/roughness texture channels are unsupported", unsupported=True)
        number(pbr.get("metallicFactor", 1), 0, 1, "Metallic factor")
        number(pbr.get("roughnessFactor", 1), 0, 1, "Roughness factor")
        if not unlit and (pbr.get("metallicFactor", 1) != 0 or pbr.get("roughnessFactor", 1) != 1):
            fail("Use KHR_materials_unlit or explicit metallicFactor=0 and roughnessFactor=1", unsupported=True)
        color = vector(pbr.get("baseColorFactor", [1, 1, 1, 1]), 4, "Base color", 1)
        if any(v < 0 for v in color):
            fail("Base color components cannot be negative")
        texture = None
        if "baseColorTexture" in pbr:
            info = pbr["baseColorTexture"]
            if not isinstance(info, dict) or set(info) - {"index", "texCoord", "extras"} or info.get("texCoord", 0) != 0:
                fail("Base-color textures require untransformed TEXCOORD_0", unsupported=True)
            texture = integer(info.get("index"), 0, texture_count - 1, "Base color texture")
        alpha = material.get("alphaMode", "OPAQUE")
        if alpha not in ("OPAQUE", "MASK"):
            fail("Only OPAQUE and MASK materials are supported", unsupported=True)
        double_sided = material.get("doubleSided", False)
        if type(double_sided) is not bool:
            fail("Material doubleSided must be boolean")
        result.append({"base_color_factor": color, "base_color_texture": texture, "unlit": unlit,
                       "alpha_mode": alpha, "alpha_cutoff": number(material.get("alphaCutoff", .5), 0, 1, "Alpha cutoff"),
                       "double_sided": double_sided})
    return result


def _animations(document, reader, nodes):
    animations = _array(document, "animations", 16)
    channels_total = 0
    for animation in animations:
        samplers = _array(animation, "samplers", 128, required=True)
        channels = _array(animation, "channels", 128, required=True)
        channels_total += len(channels)
        if channels_total > 256:
            fail("Animation channels exceed their aggregate bound")
        used_targets, used_samplers = set(), set()
        for channel in channels:
            sampler_index = integer(channel.get("sampler"), 0, len(samplers) - 1, "Animation sampler")
            used_samplers.add(sampler_index)
            sampler = samplers[sampler_index]
            if sampler.get("interpolation", "LINEAR") not in ("LINEAR", "STEP"):
                fail("Only LINEAR/STEP animation storage is supported", unsupported=True)
            target = channel.get("target")
            if not isinstance(target, dict) or target.get("path") not in ("translation", "rotation", "scale"):
                fail("Animation target is unsupported", unsupported=True)
            node = integer(target.get("node"), 0, len(nodes) - 1, "Animation node")
            if "matrix" in nodes[node] or (node, target["path"]) in used_targets:
                fail("Animation targets require unique TRS node channels")
            used_targets.add((node, target["path"]))
            times = reader.typed(sampler.get("input"), "SCALAR", {5126}, normalized=False).ravel()
            values = reader.typed(sampler.get("output"), "VEC4" if target["path"] == "rotation" else "VEC3", {5126}, normalized=False)
            reader.nonvertex(sampler["input"])
            reader.nonvertex(sampler["output"])
            if not {"min", "max"} <= set(reader.accessors[sampler["input"]]):
                fail("Animation time accessors require declared bounds")
            if len(times) > 36000 or len(values) != len(times) or times[0] < 0 or times[-1] > 3600 or np.any(np.diff(times) <= 0):
                fail("Animation timestamps/counts are invalid or exceed their bounds")
            if target["path"] == "rotation" and not np.allclose(np.sum(values * values, axis=1), 1, atol=1e-5, rtol=0):
                fail("Animation rotations must be unit quaternions")
            if target["path"] == "scale" and np.any(values <= 0):
                fail("Animation scales must be positive")
        if len(used_samplers) != len(samplers):
            fail("Unused animation samplers are unsupported", unsupported=True)
    return len(animations)


def normalize_avatar_gltf(entrypoint, files, retargeting, *, deadline):
    """Return validated render arrays and exact texture bytes; no animation or physics execution."""
    deadline_check(deadline)
    document, binary = _load_document(entrypoint, files)
    reader = _Reader(document, binary, entrypoint, files, deadline)
    nodes, parents, local, reachable = _nodes(document)
    skins = _array(document, "skins", 1, required=True)
    skin = skins[0]
    joints = skin.get("joints")
    if not isinstance(joints, list) or not 1 <= len(joints) <= MAX_JOINTS:
        fail("A real skin with 1–64 joints is required")
    for joint in joints:
        integer(joint, 0, len(nodes) - 1, "Skin joint")
        if joint not in reachable:
            fail("Skin joint must belong to the selected scene")
    if len(joints) != len(set(joints)):
        fail("Skin joints must be unique")
    ancestors = []
    for joint in joints:
        chain = set()
        while joint is not None:
            chain.add(joint)
            joint = parents[joint]
        ancestors.append(chain)
    common = set.intersection(*ancestors)
    if not common or "skeleton" in skin and integer(skin["skeleton"], 0, len(nodes) - 1, "Skin skeleton") not in common:
        fail("Skin joints need a valid common skeleton root")
    if "inverseBindMatrices" in skin:
        inverse = reader.typed(skin["inverseBindMatrices"], "MAT4", {5126}, normalized=False)
        reader.nonvertex(skin["inverseBindMatrices"])
        if len(inverse) < len(joints):
            fail("Skin has fewer inverse-bind matrices than joints")
        inverse = inverse[:len(joints)]
    else:
        inverse = np.tile(np.eye(4).reshape(16, order="F"), (len(joints), 1))
    for values in inverse:
        matrix(values.tolist(), "Inverse-bind matrix")
    textures, texture_files = _textures(document, reader)
    materials = _materials(document, len(textures))
    meshes = _array(document, "meshes", 16, required=True)
    primitives, vertices, triangles, rendered, has_skin, used_meshes = [], 0, 0, [], False, set()
    for node_index, node in enumerate(nodes):
        if "skin" in node and "mesh" not in node:
            fail("A skinned node must reference a mesh")
        if "mesh" not in node:
            continue
        mesh = meshes[integer(node["mesh"], 0, len(meshes) - 1, "Mesh index")]
        used_meshes.add(node["mesh"])
        if node_index not in reachable:
            fail("Rendered nodes outside the selected scene are unsupported", unsupported=True)
        rendered.append(node_index)
        skin_index = integer(node["skin"], 0, 0, "Node skin") if "skin" in node else None
        for primitive in _array(mesh, "primitives", MAX_PRIMITIVES, required=True):
            deadline_check(deadline)
            if len(primitives) >= MAX_PRIMITIVES:
                fail("Avatar exceeds its primitive bound")
            if primitive.get("mode", 4) != 4 or "targets" in primitive or "weights" in mesh:
                fail("Only TRIANGLES without morph targets are supported", unsupported=True)
            attributes = primitive.get("attributes")
            if (not isinstance(attributes, dict) or not {"POSITION", "NORMAL"} <= set(attributes)
                    or set(attributes) - {"POSITION", "NORMAL", "TEXCOORD_0", "JOINTS_0", "WEIGHTS_0"}):
                fail("Avatar primitives require positions/normals and supported vertex attributes", unsupported=True)
            positions = reader.typed(attributes["POSITION"], "VEC3", {5126}, normalized=False)
            if len(positions) < 3:
                fail("Triangle primitives require at least three vertex slots")
            normals = reader.typed(attributes["NORMAL"], "VEC3", {5126}, normalized=False)
            if not {"min", "max"} <= set(reader.accessors[attributes["POSITION"]]):
                fail("POSITION accessor requires declared minimum and maximum")
            if len(normals) != len(positions) or not np.allclose(np.linalg.norm(normals, axis=1), 1, atol=1e-3, rtol=0):
                fail("Vertex normals must be unit length and match the vertex count")
            if np.max(np.abs(positions)) > 1e4:
                fail("Vertex positions exceed their numeric bound")
            vertices += len(positions)
            if vertices > MAX_VERTICES:
                fail("Avatar exceeds its total vertex bound")
            indices = (reader.typed(primitive["indices"], "SCALAR", {5121, 5123, 5125}, normalized=False).ravel()
                       if "indices" in primitive else np.arange(len(positions), dtype=np.uint32))
            if "indices" in primitive:
                reader.nonvertex(primitive["indices"], indices=True)
                if np.any(indices == np.iinfo(COMPONENTS[reader.accessors[primitive["indices"]]["componentType"]]).max):
                    fail("Triangle indices cannot contain primitive-restart values")
            if len(indices) % 3 or not len(indices) or np.max(indices) >= len(positions):
                fail("Triangle indices are incomplete or outside the vertex array")
            triangles += len(indices) // 3
            if triangles > MAX_TRIANGLES:
                fail("Avatar exceeds its triangle bound")
            material = integer(primitive.get("material"), 0, len(materials) - 1, "Explicit supported material")
            texcoords = None
            if "TEXCOORD_0" in attributes:
                texcoords = reader.typed(attributes["TEXCOORD_0"], "VEC2", {5126, 5121, 5123})
                uv_spec = reader.accessors[attributes["TEXCOORD_0"]]
                if uv_spec["componentType"] != 5126 and not uv_spec.get("normalized", False):
                    fail("Integer texture coordinates must be normalized")
                if len(texcoords) != len(positions) or np.max(np.abs(texcoords)) > 1e4:
                    fail("Texture coordinates exceed their count or numeric bound")
            if materials[material]["base_color_texture"] is not None and texcoords is None:
                fail("Textured primitive is missing TEXCOORD_0")
            vertex_joints, weights = None, None
            if skin_index is not None:
                has_skin = True
                if not {"JOINTS_0", "WEIGHTS_0"} <= set(attributes):
                    fail("Skinned meshes require JOINTS_0 and WEIGHTS_0")
                vertex_joints = reader.typed(attributes["JOINTS_0"], "VEC4", {5121, 5123}, normalized=False)
                weights = reader.typed(attributes["WEIGHTS_0"], "VEC4", {5126, 5121, 5123})
                spec = reader.accessors[attributes["WEIGHTS_0"]]
                if spec["componentType"] != 5126 and not spec.get("normalized", False):
                    fail("Integer skin weights must be normalized")
                if spec["componentType"] != 5126:
                    maximum = np.iinfo(COMPONENTS[spec["componentType"]]).max
                    if np.any(np.sum(np.rint(weights.astype(float) * maximum), axis=1) != maximum):
                        fail("Integer skin weights must sum exactly to their normalization maximum")
                if (len(vertex_joints) != len(positions) or len(weights) != len(positions)
                        or np.max(vertex_joints) >= len(joints) or np.any(weights < 0)
                        or not np.allclose(np.sum(weights, axis=1), 1, atol=1e-4, rtol=0)):
                    fail("Skin indices, weights or vertex counts are invalid")
                for a in range(4):
                    for b in range(a + 1, 4):
                        if np.any((vertex_joints[:, a] == vertex_joints[:, b]) & (weights[:, a] > 0) & (weights[:, b] > 0)):
                            fail("A vertex cannot repeat a joint with nonzero weight")
            elif "JOINTS_0" in attributes or "WEIGHTS_0" in attributes:
                fail("Skin attributes require an explicit node skin")
            attribute_views = {}
            for index in attributes.values():
                reader.vertex(index)
                view_index = reader.accessors[index]["bufferView"]
                if view_index in attribute_views and "byteStride" not in reader.views[view_index]:
                    fail("Shared vertex bufferViews require an explicit byteStride")
                attribute_views[view_index] = index
            primitives.append({"node": node_index, "skin": skin_index, "material": material,
                "positions": positions.ravel().tolist(), "normals": normals.ravel().tolist(),
                "indices": indices.tolist(), "texcoords": None if texcoords is None else texcoords.ravel().tolist(),
                "joints": None if vertex_joints is None else vertex_joints.ravel().tolist(),
                "weights": None if weights is None else weights.ravel().tolist()})
    if not has_skin:
        fail("A real skinned mesh is required; image billboards and static meshes are not avatars")
    if len(used_meshes) != len(meshes):
        fail("Unreferenced meshes are unsupported by this import profile", unsupported=True)
    mapping = _retarget(retargeting, nodes, parents, rendered, joints)
    _rest_relative(mapping, parents, local, joints, inverse)
    animations = _animations(document, reader, nodes)
    result = {"schema_version": 1, "profile": RENDER_PROFILE,
              "nodes": [{"index": i, "parent": parents[i], "local_matrix": local[i].reshape(16, order="F").tolist()}
                        for i in range(len(nodes))],
              "skins": [{"joints": joints, "inverse_bind_matrices": inverse.tolist()}],
              "primitives": primitives, "materials": materials, "textures": textures, "retargeting": mapping,
              "counts": {"vertices": vertices, "triangles": triangles, "primitives": len(primitives),
                         "nodes": len(nodes), "skin_joints": len(joints), "stored_animations": animations},
              "scope": "Visual skinned mesh only; authored clips are retained but not played. Uniform scale is an authored mapping from glTF meters to simulation world units, not metric calibration. Retargeting cannot change physics, contacts or scores."}
    deadline_check(deadline)
    raw = json.dumps(result, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    if len(raw) > MAX_RENDER_BYTES:
        fail("Normalized avatar exceeds its JSON output bound")
    return result, raw, texture_files
