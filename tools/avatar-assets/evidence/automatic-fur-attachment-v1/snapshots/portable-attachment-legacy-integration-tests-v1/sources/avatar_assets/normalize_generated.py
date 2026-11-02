"""Bounded split-normal smoothing; preserve topology, UVs, skins, clips and originals."""

from copy import deepcopy
import io
import json
from pathlib import Path
import shutil
import struct

import numpy as np
from PIL import Image

from avatar_assets.provider_model import preflight
from avatar_assets.safety import artifacts, local_file, new_directory, preserve_failure, sha256, write_json
from avatar_assets.three_d import _GLB


def _array(glb, index, width, component, maximum):
    a = glb.doc["accessors"][index]
    count = a.get("count")
    if type(count) is not int or not 0 < count <= maximum or a.get("componentType") != component or a.get("sparse") or a.get("type") != {1: "SCALAR", 2: "VEC2", 3: "VEC3", 4: "VEC4"}[width]:
        raise ValueError("Offline source accessor is outside its bounded profile")
    view = glb.doc["bufferViews"][a["bufferView"]]
    dtype = np.dtype({5126: "<f4", 5125: "<u4", 5123: "<u2", 5121: "u1"}[component])
    stride = view.get("byteStride", width*dtype.itemsize)
    offset = a.get("byteOffset", 0)
    if type(stride) is not int or stride < width*dtype.itemsize or offset < 0 or offset+(count-1)*stride+width*dtype.itemsize > view["byteLength"]:
        raise ValueError("Offline source accessor range is invalid")
    values = np.ndarray((count, width), dtype=dtype, buffer=glb.binary,
                        offset=view.get("byteOffset", 0)+offset, strides=(stride, dtype.itemsize)).copy()
    if not np.isfinite(values).all():
        raise ValueError("Nonfinite source attribute")
    return values


def normalize(source, output):
    source = local_file(source, 64*1024*1024)
    glb = _GLB(source)
    doc = deepcopy(glb.doc)
    original_images = []
    for image in doc.get("images", []):
        view = glb.doc["bufferViews"][image["bufferView"]]
        start = view.get("byteOffset", 0)
        raw = glb.binary[start:start+view["byteLength"]]
        with Image.open(io.BytesIO(raw)) as decoded:
            if decoded.width*decoded.height > 16_777_216 or getattr(decoded, "n_frames", 1) != 1:
                raise ValueError("Embedded source image exceeds the 16MP still-image profile")
            decoded.verify()
        original_images.append(raw)
    # Keep non-mesh accessors and embedded images verbatim, including skin bind matrices and motion.
    binary = bytearray(glb.binary)

    def append(raw):
        binary.extend(b"\x00"*(-len(binary)%4))
        offset = len(binary)
        binary.extend(raw)
        doc["bufferViews"].append({"buffer": 0, "byteOffset": offset, "byteLength": len(raw)})
        return len(doc["bufferViews"])-1

    def accessor(values, kind, component):
        values = np.asarray(values, dtype={5126: "<f4", 5125: "<u4", 5123: "<u2"}[component])
        view = append(values.tobytes())
        value = {"bufferView": view, "componentType": component, "count": len(values), "type": kind}
        if kind == "VEC3":
            value.update(min=values.min(axis=0).tolist(), max=values.max(axis=0).tolist())
        doc["accessors"].append(value)
        return len(doc["accessors"])-1

    source_vertices, source_indices, output_vertices = 0, 0, 0
    for mesh in doc["meshes"]:
        for primitive in mesh["primitives"]:
            attrs = primitive["attributes"]
            required = {"POSITION", "NORMAL", "TEXCOORD_0"}
            allowed = required|{"TANGENT", "JOINTS_0", "WEIGHTS_0"}
            if not required <= set(attrs) <= allowed or primitive.get("targets") or primitive.get("mode", 4) != 4:
                raise ValueError("Offline cleanup requires POSITION/NORMAL/UV0 triangles and at most one four-weight skin")
            if ("JOINTS_0" in attrs) != ("WEIGHTS_0" in attrs):
                raise ValueError("Incomplete skin attributes")
            points = _array(glb, attrs["POSITION"], 3, 5126, 200000)
            normals = _array(glb, attrs["NORMAL"], 3, 5126, 200000)
            uv = _array(glb, attrs["TEXCOORD_0"], 2, 5126, 200000)
            index_source = glb.doc["accessors"][primitive["indices"]]
            if index_source.get('componentType') not in (5121, 5123, 5125):
                raise ValueError('Triangle indices require unsigned integer components')
            indices = _array(glb, primitive["indices"], 1, index_source["componentType"], 300000).reshape(-1)
            source_vertices += len(points)
            source_indices += len(indices)
            if source_vertices > 200000 or source_indices > 300000 or len(normals) != len(points) or len(uv) != len(points):
                raise ValueError("Offline source exceeds 200000 vertices/100000 triangles")
            if len(indices)%3 or indices.max() >= len(points):
                raise ValueError("Invalid source triangles")
            _, position_inverse = np.unique(points, axis=0, return_inverse=True)
            average = np.zeros((position_inverse.max()+1, 3), dtype=np.float64)
            np.add.at(average, position_inverse, normals)
            lengths = np.linalg.norm(average, axis=1)
            if np.any(lengths < 1e-8):
                raise ValueError("Coincident opposite surfaces cannot be safely smoothed")
            averaged_normals = (average/lengths[:, None]).astype(np.float32)[position_inverse]
            # UV seams remain distinct; only authored normals change, never positions or texture bytes.
            joined = np.concatenate((points, uv), axis=1)
            joints, weights = None, None
            if "JOINTS_0" in attrs:
                joint_type = glb.doc["accessors"][attrs["JOINTS_0"]]["componentType"]
                if joint_type not in (5121, 5123):
                    raise ValueError("Invalid joint component type")
                joints = _array(glb, attrs["JOINTS_0"], 4, joint_type, 200000)
                weights = _array(glb, attrs["WEIGHTS_0"], 4, 5126, 200000)
                if len(joints) != len(points) or len(weights) != len(points) or np.any(weights < 0) or not np.allclose(weights.sum(axis=1), 1, atol=.001):
                    raise ValueError("Invalid source skin weights")
                joined = np.concatenate((joined, joints.astype(np.float32), weights), axis=1)
            _, first, remap = np.unique(joined, axis=0, return_index=True, return_inverse=True)
            output_vertices += len(first)
            if output_vertices > 100000:
                raise ValueError("Normal cleanup still exceeds the unchanged 100000-vertex runtime limit")
            primitive["attributes"] = {"POSITION": accessor(points[first], "VEC3", 5126),
                "NORMAL": accessor(averaged_normals[first], "VEC3", 5126), "TEXCOORD_0": accessor(uv[first], "VEC2", 5126)}
            if joints is not None:
                primitive["attributes"].update(JOINTS_0=accessor(joints[first], "VEC4", 5123),
                                                WEIGHTS_0=accessor(weights[first], "VEC4", 5126))
            primitive["indices"] = accessor(remap[indices], "SCALAR", 5125)
    doc["buffers"] = [{"byteLength": len(binary)}]
    doc["asset"]["generator"] = "Local authored split-normal smoothing; original Meshy source retained separately"
    encoded = json.dumps(doc, separators=(",", ":"), allow_nan=False).encode()
    encoded += b" "*(-len(encoded)%4)
    binary.extend(b"\x00"*(-len(binary)%4))
    data = struct.pack("<4sII", b"glTF", 2, 28+len(encoded)+len(binary))
    data += struct.pack("<II", len(encoded), 0x4E4F534A)+encoded
    data += struct.pack("<II", len(binary), 0x004E4942)+binary
    out = new_directory(output)
    try:
        (out/"model.glb").write_bytes(data)
        shutil.copyfile(Path(__file__), out/"normalize_generated.py")
        facts = preflight(out/"model.glb")
        report = {"schema_version": 1, "kind": "authored_normal_cleanup", "status": "needs_visual_review",
                  "source_path": str(source), "source_sha256": glb.hash, "input_vertices": source_vertices,
                  "output_vertices": output_vertices, "triangle_count": source_indices//3,
                  "position_changes": False, "texture_bytes_changed": False, "normals": "Averaged across exact coincident positions",
                  "skin_weights_preserved": True, "animation_accessors_preserved": True,
                  "tangents": "Removed from changed normals; native export must regenerate surface tangents",
                  "original_preserved": sha256(source) == glb.hash, "preflight": facts,
                  "limits": {"offline_input_vertices": 200000, "runtime_output_vertices": 100000,
                             "source_bytes": 64*1024*1024, "triangles": 100000, "texture_pixels": 16_777_216},
                  "new_provider_cost_usd": 0, "artifacts": artifacts(out)}
        write_json(out/"manifest.json", report)
        return report
    except Exception as error:
        preserve_failure(out, error)
        raise


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    result = normalize(args.source, args.out)
    print(json.dumps({k: result[k] for k in ("status", "input_vertices", "output_vertices", "triangle_count")}, indent=2))
