"""Analytic two-bone geometry exercises glTF ingestion, not character authoring."""

from copy import deepcopy
import hashlib
from io import BytesIO
import json
import struct
import time

import numpy as np
from PIL import Image
import pytest

from weft import avatar_gltf as module


def mapping():
    return {"profile": "weft-avatar-retarget-v1", "root_node": 0, "scale": 1,
            "orientation_xyzw": [0, 0, 0, 1], "bindings": [
                {"node": 0, "body": "mobile_root", "translation": [0, 0, 0], "rotation_xyzw": [0, 0, 0, 1]},
                {"node": 1, "body": "left_elbow_body", "translation": [0, 1, 0], "rotation_xyzw": [0, 0, 0, 1]}]}


def fixture_model(*, texture=False, embedded=False, interleaved=False, integer_weights=False, animation=False):
    binary, views, accessors, extras = bytearray(), [], [], {}

    def view(raw, **kwargs):
        binary.extend(b"\0" * (-len(binary) % 4))
        result = len(views)
        views.append({"buffer": 0, "byteOffset": len(binary), "byteLength": len(raw), **kwargs})
        binary.extend(raw)
        return result

    def accessor(values, kind, component, *, normalized=False, bounds=False):
        values = np.asarray(values, dtype=module.COMPONENTS[component]).reshape((-1, module.WIDTHS[kind]))
        result = len(accessors)
        spec = {"bufferView": view(values.tobytes()), "componentType": component, "count": len(values), "type": kind}
        if normalized:
            spec["normalized"] = True
        if bounds:
            spec.update(min=values.min(axis=0).tolist(), max=values.max(axis=0).tolist())
        accessors.append(spec)
        return result

    positions = [[-.2, 0, 0], [.2, 0, 0], [-.2, 1, 0], [.2, 1, 0], [-.2, 2, 0], [.2, 2, 0]]
    normals = [[0, 0, 1]] * 6
    if interleaved:
        combined = np.hstack((positions, normals)).astype("<f4")
        v = view(combined.tobytes(), byteStride=24, target=34962)
        accessors.extend([
            {"bufferView": v, "componentType": 5126, "count": 6, "type": "VEC3", "min": [-.2, 0, 0], "max": [.2, 2, 0]},
            {"bufferView": v, "byteOffset": 12, "componentType": 5126, "count": 6, "type": "VEC3"}])
        position, normal = 0, 1
    else:
        position = accessor(positions, "VEC3", 5126, bounds=True)
        normal = accessor(normals, "VEC3", 5126)
    joints = accessor([[0, 1, 0, 0]] * 6, "VEC4", 5121)
    weights = [[1, 0, 0, 0], [1, 0, 0, 0], [.5, .5, 0, 0], [.5, .5, 0, 0], [0, 1, 0, 0], [0, 1, 0, 0]]
    if integer_weights:
        weights = [[255, 0, 0, 0], [255, 0, 0, 0], [128, 127, 0, 0], [128, 127, 0, 0], [0, 255, 0, 0], [0, 255, 0, 0]]
    weight = accessor(weights, "VEC4", 5121 if integer_weights else 5126, normalized=integer_weights)
    indices = accessor([0, 1, 2, 1, 3, 2, 2, 3, 4, 3, 5, 4], "SCALAR", 5123)
    inverse = np.eye(4)
    inverse[1, 3] = -1
    matrices = accessor([np.eye(4).reshape(16, order="F"), inverse.reshape(16, order="F")], "MAT4", 5126)
    attributes = {"POSITION": position, "NORMAL": normal, "JOINTS_0": joints, "WEIGHTS_0": weight}
    doc = {"asset": {"version": "2.0", "generator": "Analytic test fixture"}, "scene": 0,
           "scenes": [{"nodes": [0]}], "nodes": [{"children": [1, 2]}, {"translation": [0, 1, 0]},
                                                        {"mesh": 0, "skin": 0, "translation": [10, 0, 0]}],
           "skins": [{"joints": [0, 1], "skeleton": 0, "inverseBindMatrices": matrices}],
           "meshes": [{"primitives": [{"attributes": attributes, "indices": indices, "material": 0}]}],
           "materials": [{"extensions": {"KHR_materials_unlit": {}}, "pbrMetallicRoughness": {"baseColorFactor": [.8, .6, .4, 1]}}],
           "extensionsUsed": ["KHR_materials_unlit"], "bufferViews": views, "accessors": accessors}
    if texture:
        attributes["TEXCOORD_0"] = accessor([[0, 0], [1, 0], [0, .5], [1, .5], [0, 1], [1, 1]], "VEC2", 5126)
        image = BytesIO()
        Image.new("RGBA", (2, 2), (20, 80, 180, 128)).save(image, format="PNG")
        raw = image.getvalue()
        if embedded:
            doc["images"] = [{"bufferView": view(raw), "mimeType": "image/png"}]
        else:
            doc["images"] = [{"uri": "skin.png"}]
            extras["skin.png"] = raw
        doc["samplers"] = [{"wrapS": 33648, "wrapT": 33071, "magFilter": 9728, "minFilter": 9987}]
        doc["textures"] = [{"source": 0, "sampler": 0}]
        doc["materials"][0]["pbrMetallicRoughness"]["baseColorTexture"] = {"index": 0}
        doc["materials"][0].update(alphaMode="MASK", alphaCutoff=.25, doubleSided=True)
    if animation:
        times = accessor([0, 1], "SCALAR", 5126, bounds=True)
        rotations = accessor([[0, 0, 0, 1], [0, 0, 1, 0]], "VEC4", 5126)
        doc["animations"] = [{"samplers": [{"input": times, "output": rotations}],
                              "channels": [{"sampler": 0, "target": {"node": 1, "path": "rotation"}}]}]
    doc["buffers"] = [{"byteLength": len(binary)}]
    return doc, bytes(binary), extras


def packaged(doc, binary, extras=None, *, glb=False):
    doc = deepcopy(doc)
    files = dict(extras or {})
    if glb:
        doc["buffers"][0].pop("uri", None)
        encoded = json.dumps(doc, separators=(",", ":")).encode()
        encoded += b" " * (-len(encoded) % 4)
        binary += b"\0" * (-len(binary) % 4)
        chunks = struct.pack("<II", len(encoded), 0x4E4F534A) + encoded + struct.pack("<II", len(binary), 0x004E4942) + binary
        files["model.glb"] = struct.pack("<III", 0x46546C67, 2, 12 + len(chunks)) + chunks
        return "model.glb", files
    doc["buffers"][0]["uri"] = "model.bin"
    files.update({"model.gltf": json.dumps(doc, separators=(",", ":")).encode(), "model.bin": binary})
    return "model.gltf", files


def normalize(doc, binary, extras=None, *, glb=False, retarget=None):
    name, files = packaged(doc, binary, extras, glb=glb)
    return module.normalize_avatar_gltf(name, files, mapping() if retarget is None else retarget, deadline=time.monotonic() + 5)


@pytest.mark.parametrize("glb,interleaved,integer_weights", [(False, False, False), (True, False, False), (False, True, True)])
def test_real_skin_preserves_rest_geometry_and_known_elbow_motion(glb, interleaved, integer_weights):
    doc, binary, extras = fixture_model(interleaved=interleaved, integer_weights=integer_weights)
    render, raw, textures = normalize(doc, binary, extras, glb=glb)
    primitive, skin = render["primitives"][0], render["skins"][0]
    assert json.loads(raw) == render and not textures
    assert render["counts"] == {"vertices": 6, "triangles": 4, "primitives": 1, "nodes": 3, "skin_joints": 2, "stored_animations": 0}
    positions = np.array(primitive["positions"]).reshape(-1, 3)
    inverse = [np.array(x).reshape((4, 4), order="F") for x in skin["inverse_bind_matrices"]]
    frames = [np.eye(4), np.eye(4)]
    frames[1][1, 3] = 1
    vertex_joints = np.array(primitive["joints"]).reshape(-1, 4)
    weights = np.array(primitive["weights"]).reshape(-1, 4)
    rest = []
    for p, indices, values in zip(positions, vertex_joints, weights):
        rest.append(sum(weight * (frames[joint] @ inverse[joint] @ np.r_[p, 1]) for joint, weight in zip(indices, values))[:3])
    assert np.allclose(rest, positions)
    assert render["nodes"][2]["local_matrix"][12] == 10  # Skinning must ignore the mesh-node transform.
    frames[1][:3, :3] = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
    bent = frames[1] @ inverse[1] @ np.r_[positions[5], 1]
    assert bent[:3] == pytest.approx([-1, 1.2, 0])
    assert skin["joints"] == [0, 1] and vertex_joints[5, 1] == 1


@pytest.mark.parametrize("embedded", [False, True])
def test_custom_skin_texture_pixels_uv_sampler_and_alpha_are_preserved(embedded):
    doc, binary, extras = fixture_model(texture=True, embedded=embedded)
    render, _, textures = normalize(doc, binary, extras, glb=embedded)
    texture = render["textures"][0]
    raw = textures[texture["path"]]
    assert texture["sha256"] == hashlib.sha256(raw).hexdigest()
    with Image.open(BytesIO(raw)) as image:
        assert image.getpixel((0, 0)) == (20, 80, 180, 128)
    assert (texture["wrap_s"], texture["wrap_t"], texture["mag_filter"], texture["min_filter"]) == (33648, 33071, 9728, 9987)
    assert render["primitives"][0]["texcoords"][-2:] == [1, 1]
    assert render["materials"][0] == {"base_color_factor": [.8, .6, .4, 1], "base_color_texture": 0,
        "unlit": True, "alpha_mode": "MASK", "alpha_cutoff": .25, "double_sided": True}


def test_diffuse_subset_and_stored_animation_are_explicit_not_executed():
    doc, binary, extras = fixture_model(animation=True)
    doc["materials"][0].pop("extensions")
    doc["materials"][0]["pbrMetallicRoughness"].update(metallicFactor=0, roughnessFactor=1)
    render, _, _ = normalize(doc, binary, extras)
    assert render["counts"]["stored_animations"] == 1
    assert render["materials"][0]["unlit"] is False
    assert render["nodes"][1]["local_matrix"][13] == 1
    assert "retained but not played" in render["scope"]


@pytest.mark.parametrize("case,match", [
    ("cycle", "cycle|parent"), ("multiple_parents", "multiple parents"), ("bad_joint", "Skin joint"),
    ("duplicate_joint", "unique"), ("bad_ibm", "fewer inverse"), ("bad_offset", "offset|range"),
    ("bad_stride", "[sS]tride"), ("sparse", "Sparse"), ("missing_position_bounds", "POSITION accessor"),
    ("missing_normal", "positions/normals"), ("no_skin", "skins"), ("no_weights", "JOINTS_0"),
    ("unsupported_extension", "extensions"), ("extension_wrong_location", "extension"),
    ("blend", "OPAQUE"), ("normal_map", "Normal"), ("metallic", "metallicFactor"),
    ("missing_uv", "TEXCOORD_0"), ("bad_sampler", "sampler"), ("morph", "morph"),
    ("singular", "degenerate"), ("matrix_trs", "matrix with TRS"), ("wrong_target", "alignment|target"),
    ("animation_time", "declared bounds|timestamps"), ("animation_cubic", "LINEAR/STEP"),
])
def test_malformed_or_unsupported_gltf_fails_with_actionable_reason(case, match):
    doc, binary, extras = fixture_model(texture=True, animation=True)
    primitive = doc["meshes"][0]["primitives"][0]
    if case == "cycle": doc["nodes"][1]["children"] = [0]
    elif case == "multiple_parents": doc["nodes"][2]["children"] = [1]
    elif case == "bad_joint": doc["skins"][0]["joints"][1] = 999
    elif case == "duplicate_joint": doc["skins"][0]["joints"] = [0, 0]
    elif case == "bad_ibm": doc["accessors"][5]["count"] = 1
    elif case == "bad_offset": doc["accessors"][0]["byteOffset"] = 100000
    elif case == "bad_stride": doc["bufferViews"][0]["byteStride"] = 2
    elif case == "sparse": doc["accessors"][0]["sparse"] = {"count": 1}
    elif case == "missing_position_bounds": doc["accessors"][0].pop("min")
    elif case == "missing_normal": primitive["attributes"].pop("NORMAL")
    elif case == "no_skin": doc.pop("skins")
    elif case == "no_weights": primitive["attributes"].pop("WEIGHTS_0")
    elif case == "unsupported_extension": doc["extensionsUsed"].append("KHR_draco_mesh_compression")
    elif case == "extension_wrong_location": doc["nodes"][0]["extensions"] = {"KHR_materials_unlit": {}}
    elif case == "blend": doc["materials"][0]["alphaMode"] = "BLEND"
    elif case == "normal_map": doc["materials"][0]["normalTexture"] = {"index": 0}
    elif case == "metallic": doc["materials"][0].pop("extensions")
    elif case == "missing_uv": primitive["attributes"].pop("TEXCOORD_0")
    elif case == "bad_sampler": doc["samplers"][0]["minFilter"] = 1
    elif case == "morph": primitive["targets"] = []
    elif case == "singular": doc["nodes"][0]["scale"] = [0, 1, 1]
    elif case == "matrix_trs": doc["nodes"][1]["matrix"] = np.eye(4).ravel().tolist()
    elif case == "wrong_target": doc["bufferViews"][0]["target"] = 34963
    elif case == "animation_time": doc["accessors"][-2]["min"] = [-1]
    elif case == "animation_cubic": doc["animations"][0]["samplers"][0]["interpolation"] = "CUBICSPLINE"
    with pytest.raises(module.AvatarAssetError, match=match):
        normalize(doc, binary, extras)


@pytest.mark.parametrize("case,match", [("joint_range", "Skin indices"), ("zero_weights", "weights"),
    ("duplicate_influence", "repeat a joint"), ("integer_sum", "sum exactly"), ("nonfinite", "nonfinite"),
    ("bad_index", "indices"), ("ibm_affine", "affine")])
def test_binary_skinning_data_is_validated_not_only_json(case, match):
    doc, binary, extras = fixture_model(integer_weights=case == "integer_sum")
    binary = bytearray(binary)
    index = {"joint_range": 2, "duplicate_influence": 2, "zero_weights": 3, "integer_sum": 3,
             "nonfinite": 1, "bad_index": 4, "ibm_affine": 5}[case]
    spec = doc["accessors"][index]
    offset = doc["bufferViews"][spec["bufferView"]]["byteOffset"]
    if case == "joint_range": binary[offset] = 10
    elif case == "duplicate_influence": binary[offset + 2 * 4 + 1] = 0
    elif case == "zero_weights": binary[offset:offset + 16] = b"\0" * 16
    elif case == "integer_sum": binary[offset] = 254
    elif case == "nonfinite": struct.pack_into("<f", binary, offset, float("nan"))
    elif case == "bad_index": struct.pack_into("<H", binary, offset, 999)
    elif case == "ibm_affine": struct.pack_into("<f", binary, offset + 15 * 4, 0)
    with pytest.raises(module.AvatarAssetError, match=match):
        normalize(doc, bytes(binary), extras)


@pytest.mark.parametrize("uri", ["https://example.invalid/skin.png", "data:image/png;base64,AAAA", "../skin.png", "/skin.png", "%2e%2e/skin.png"])
def test_resources_never_resolve_network_data_or_parent_paths(uri):
    doc, binary, extras = fixture_model(texture=True)
    doc["images"][0]["uri"] = uri
    with pytest.raises(module.AvatarAssetError, match="confined relative"):
        normalize(doc, binary, extras)


@pytest.mark.parametrize("case", ["root_unmapped", "outside_root", "duplicate", "unknown_body", "scale", "orientation"])
def test_retargeting_requires_explicit_valid_body_mapping(case):
    doc, binary, extras = fixture_model()
    value = mapping()
    if case == "root_unmapped": value["bindings"].pop(0)
    elif case == "outside_root": value["root_node"] = 1
    elif case == "duplicate": value["bindings"].append(deepcopy(value["bindings"][0]))
    elif case == "unknown_body": value["bindings"][0]["body"] = "guess_neck"
    elif case == "scale": value["scale"] = 0
    elif case == "orientation": value["orientation_xyzw"] = [0, 0, 0, 0]
    with pytest.raises(module.AvatarAssetError):
        normalize(doc, binary, extras, retarget=value)


def test_texture_pixels_and_output_geometry_have_explicit_bounds(monkeypatch):
    doc, binary, extras = fixture_model(texture=True)
    monkeypatch.setattr(module, "MAX_TEXTURE_PIXELS", 3)
    with pytest.raises(module.AvatarAssetError, match="decoded-pixel"):
        normalize(doc, binary, extras)
    monkeypatch.setattr(module, "MAX_TEXTURE_PIXELS", 8 * 1024 * 1024)
    monkeypatch.setattr(module, "MAX_VERTICES", 5)
    with pytest.raises(module.AvatarAssetError, match="vertex bound"):
        normalize(doc, binary, extras)
    monkeypatch.setattr(module, "MAX_VERTICES", 20000)
    monkeypatch.setattr(module, "MAX_RENDER_BYTES", 100)
    with pytest.raises(module.AvatarAssetError, match="JSON output bound"):
        normalize(doc, binary, extras)


def test_expired_deadline_fails_before_parsing():
    with pytest.raises(module.AvatarAssetError, match="deadline"):
        module.normalize_avatar_gltf("missing.glb", {}, mapping(), deadline=time.monotonic() - 1)


def test_invalid_glb_length_and_json_duplicates_fail():
    doc, binary, extras = fixture_model()
    name, files = packaged(doc, binary, extras, glb=True)
    files[name] = files[name][:-1]
    with pytest.raises(module.AvatarAssetError, match="header/version/length"):
        module.normalize_avatar_gltf(name, files, mapping(), deadline=time.monotonic() + 1)
    with pytest.raises(module.AvatarAssetError, match="duplicate"):
        module.json_document(b'{"asset":{},"asset":{}}')


def test_malformed_accessor_type_is_a_structured_failure():
    doc, binary, extras = fixture_model()
    doc["accessors"][0]["type"] = {"not": "a type"}
    with pytest.raises(module.AvatarAssetError, match="accessor type"):
        normalize(doc, binary, extras)


def test_degenerate_one_slot_triangle_matches_renderer_rejection():
    doc, binary, extras = fixture_model()
    doc["accessors"][0].update(count=1, min=[-.2, 0, 0], max=[-.2, 0, 0])
    with pytest.raises(module.AvatarAssetError, match="at least three vertex slots"):
        normalize(doc, binary, extras)


def test_aliasing_accessors_cannot_multiply_decoded_work_without_bound():
    doc, binary, extras = fixture_model()
    padding = b"\0" * (-len(binary) % 4)
    view = len(doc["bufferViews"])
    doc["bufferViews"].append({"buffer": 0, "byteOffset": len(binary) + len(padding), "byteLength": 400000})
    binary += padding + b"\0" * 400000
    doc["buffers"][0]["byteLength"] = len(binary)
    doc["accessors"].extend({"bufferView": view, "componentType": 5126, "count": 100000, "type": "SCALAR"} for _ in range(22))
    with pytest.raises(module.AvatarAssetError, match="decoded component bound"):
        normalize(doc, binary, extras)


def test_representative_near_vertex_limit_normalization_is_bounded(record_property):
    doc, binary, extras = fixture_model()
    expanded, views = bytearray(), []
    repeat = 3333
    for index, spec in enumerate(doc["accessors"]):
        view = doc["bufferViews"][spec["bufferView"]]
        value = np.frombuffer(binary, dtype=module.COMPONENTS[spec["componentType"]],
                              offset=view["byteOffset"], count=spec["count"] * module.WIDTHS[spec["type"]])
        value = value.reshape(spec["count"], -1)
        if index == 4:
            value = np.concatenate([value + i * 6 for i in range(repeat)])
        elif spec["type"] != "MAT4":
            value = np.tile(value, (repeat, 1))
        expanded.extend(b"\0" * (-len(expanded) % 4))
        raw = value.astype(module.COMPONENTS[spec["componentType"]]).tobytes()
        views.append({"buffer": 0, "byteOffset": len(expanded), "byteLength": len(raw)})
        expanded.extend(raw)
        spec.update(bufferView=index, count=len(value))
    doc["bufferViews"] = views
    doc["buffers"][0]["byteLength"] = len(expanded)
    started = time.perf_counter()
    render, raw, _ = normalize(doc, bytes(expanded), extras)
    elapsed = time.perf_counter() - started
    assert render["counts"]["vertices"] == 19998 and render["counts"]["triangles"] == 13332
    assert len(raw) < module.MAX_RENDER_BYTES
    record_property("vertices", 19998)
    record_property("render_bytes", len(raw))
    record_property("normalization_seconds", round(elapsed, 6))
