"""Independent, fixed-input CPU custody check; no producer/repair imports."""
import hashlib
import json
import mmap
import resource
import struct
import sys
import time
import zipfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
START = time.monotonic()


def require(value, message):
    if not value:
        raise ValueError(message)
    if time.monotonic() - START > 10:
        raise TimeoutError("Independent CPU deadline")


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


class GLB:
    def __init__(self, path, expected):
        require(digest(path) == expected, "Pinned model hash mismatch")
        self.file = path.open("rb")
        self.raw = mmap.mmap(self.file.fileno(), 0, access=mmap.ACCESS_READ)
        require(28 < len(self.raw) <= 64 * 1024**2, "GLB byte bound")
        magic, version, total, length, kind = struct.unpack_from("<4sIIII", self.raw)
        require(magic == b"glTF" and version == 2 and total == len(self.raw), "GLB header")
        require(kind == 0x4E4F534A and length <= 2 * 1024**2, "JSON chunk")
        self.doc = json.loads(self.raw[20:20 + length])
        size, kind = struct.unpack_from("<II", self.raw, 20 + length)
        self.begin = 28 + length
        require(kind == 0x004E4942 and self.begin + size == len(self.raw), "BIN chunk")
        self.binary = memoryview(self.raw)[self.begin:]
        require(not any("uri" in x for x in self.doc.get("images", []) + self.doc["buffers"]), "External data")

    def span(self, index):
        a = self.doc["accessors"][index]
        v = self.doc["bufferViews"][a["bufferView"]]
        require("sparse" not in a and not a.get("normalized"), "Nonliteral accessor")
        require(v.get("buffer", 0) == 0, "Other buffer")
        dtype = np.dtype({5126: "<f4", 5125: "<u4", 5123: "<u2", 5121: "u1"}[a["componentType"]])
        width = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT4": 16}[a["type"]]
        require(0 < a["count"] <= 300000, "Accessor count bound")
        stride = v.get("byteStride", dtype.itemsize * width)
        start = v.get("byteOffset", 0) + a.get("byteOffset", 0)
        stop = start + (a["count"] - 1) * stride + dtype.itemsize * width
        require(0 <= start < stop <= len(self.binary), "Accessor bounds")
        require(stop <= v.get("byteOffset", 0) + v["byteLength"], "Buffer view bounds")
        return start, stop, stride, dtype, width

    def array(self, index):
        start, stop, stride, dtype, width = self.span(index)
        a = self.doc["accessors"][index]
        return np.ndarray((a["count"], width), dtype=dtype, buffer=self.binary,
                          offset=start, strides=(stride, dtype.itemsize))


def differences(a, b, path=""):
    if type(a) is not type(b):
        return [path]
    if isinstance(a, dict):
        require(a.keys() == b.keys(), "JSON keys changed")
        return [p for key in a for p in differences(a[key], b[key], path + "/" + key)]
    if isinstance(a, list):
        require(len(a) == len(b), "JSON list length changed")
        return [p for i, (x, y) in enumerate(zip(a, b)) for p in differences(x, y, path + "/" + str(i))]
    return [] if a == b else [path]


def main():
    prepared = json.loads((OUT / "prepared.json").read_text())
    for row in prepared["inputs"]:
        require(digest(ROOT / row["path"]) == row["sha256"], "Input changed: " + row["path"])
    models = {row["role"]: GLB(ROOT / row["path"], row["sha256"])
              for row in prepared["inputs"] if row["role"] in ("raw_ids", "source", "candidate")}
    raw, source, candidate = (models[k] for k in ("raw_ids", "source", "candidate"))
    report = json.loads((ROOT / "outputs/panda-staged-fringe-v1/source-skin.json").read_text())
    owners = [o for o in report["owners"] if o["materials"] == ["SourceColoredFringe"]]
    require(len(owners) == 1, "Unique fringe owner")
    owner = owners[0]
    mesh_ids = [i for i, m in enumerate(source.doc["meshes"]) if m.get("name") == owner["mesh_name"]]
    require(len(mesh_ids) == 1, "Unique fringe mesh")
    mesh = mesh_ids[0]
    require(len(source.doc["meshes"][mesh]["primitives"]) == 1, "Unique fringe primitive")
    primitive = source.doc["meshes"][mesh]["primitives"][0]
    raw_primitive = raw.doc["meshes"][mesh]["primitives"][0]
    position = primitive["attributes"]["POSITION"]
    require(position == 65 and raw_primitive["attributes"]["POSITION"] == position, "Expected position identity")
    changes = differences(source.doc, candidate.doc)
    require(changes and all(p.startswith(f"/accessors/{position}/min/") or
                            p.startswith(f"/accessors/{position}/max/") for p in changes), "JSON change outside position bounds")
    require(len(source.binary) == len(candidate.binary), "BIN size changed")
    start, stop, stride, dtype, width = source.span(position)
    require(stride == 12 and width == 3 and dtype == np.dtype("<f4"), "Contiguous float32 positions")
    for i in range(len(source.doc["accessors"])):
        if i != position:
            a, b, *_ = source.span(i)
            require(b <= start or a >= stop, "Position overlaps another accessor")
    for img in source.doc.get("images", []):
        view = source.doc["bufferViews"][img["bufferView"]]
        a = view.get("byteOffset", 0)
        require(a + view["byteLength"] <= start or a >= stop, "Position overlaps image")
    require(sum(p["attributes"].get("POSITION") == position for m in source.doc["meshes"] for p in m["primitives"]) == 1,
            "Shared fringe position binding")
    for a, b in [(0, start), (stop, len(source.binary))]:
        for offset in range(a, b, 1024**2):
            end = min(b, offset + 1024**2)
            require(source.binary[offset:end] == candidate.binary[offset:end], "Non-position BIN changed")
    before, after = source.array(position), candidate.array(position)
    require(np.isfinite(after).all(), "Nonfinite candidate")
    require(np.array_equal(raw.array(position), before), "Raw/source position correspondence changed")
    require(np.array_equal(raw.array(raw_primitive["indices"]), source.array(primitive["indices"])), "Raw/source index order changed")
    ids_float = raw.array(raw_primitive["attributes"][report["attribute"]]).reshape(-1)
    require(np.isfinite(ids_float).all() and np.array_equal(ids_float, np.floor(ids_float)), "Noninteger source IDs")
    ids = ids_float.astype(np.int64) - owner["vertex_start"]
    require(np.array_equal(np.sort(ids), np.arange(owner["vertex_count"])), "Missing/duplicate fringe source IDs")
    npz = ROOT / "outputs/panda-staged-fringe-v1/fringe-correspondence.npz"
    with zipfile.ZipFile(npz) as archive:
        require(sum(item.file_size for item in archive.infolist()) <= 8 * 1024**2, "NPZ expanded byte bound")
    with np.load(npz, allow_pickle=False) as helper:
        mapping = helper["native_vertex_to_helper_vertex"]
        require(mapping.shape == (owner["vertex_count"],) and mapping.dtype == np.int32, "Corner map schema")
        require(np.array_equal(mapping, helper["triangles"].reshape(-1)), "Helper native-corner map")
        require(mapping.min() >= 0 and mapping.max() < len(helper["positions"]), "Corner map bounds")
        tips = mapping[ids] % 4 >= 2
    row_changed = np.any(before.view(np.uint8).reshape(-1, 12) != after.view(np.uint8).reshape(-1, 12), axis=1)
    require(np.array_equal(row_changed, tips), "Changed rows are not exactly mapped tips")
    require(before[~tips].tobytes() == after[~tips].tobytes(), "Root positions changed")
    require(candidate.doc["accessors"][position]["min"] == after.min(axis=0).astype(float).tolist(), "Incorrect minimum bounds")
    require(candidate.doc["accessors"][position]["max"] == after.max(axis=0).astype(float).tolist(), "Incorrect maximum bounds")
    changed_bytes = int(np.count_nonzero(before.view(np.uint8) != after.view(np.uint8)))
    for row in prepared["inputs"]:
        require(digest(ROOT / row["path"]) == row["sha256"], "Input changed during check")
    return {"schema_version": 1, "status": "passed_preservation_only", "models": prepared["inputs"][:3],
            "source_id_mapping": "Raw retained author IDs -> exact native corner map -> helper root/tip role",
            "mesh": mesh, "primitive": 0, "position_accessor": position, "position_binary_range": [start, stop],
            "roots_byte_exact": int((~tips).sum()), "tips_changed": int(tips.sum()), "changed_bytes": changed_bytes,
            "json_changed_paths": changes, "all_non_tip_binary_and_padding_exact": True,
            "positions_exclusive_to_fringe": True, "all_other_json_exact": True,
            "unchanged": ["source body", "roots", "indices", "normals", "UVs", "joint indices", "weights",
                          "skins", "bone nodes", "animation channels", "materials", "textures"],
            "root_motion_inference": "Roots, weights, joints, nodes, inverse binds and curves are byte-identical; their prior exported attachment result is unchanged. No new pose evaluation was performed.",
            "scope": "Independent byte/structure check, not a visual, collision, loop, or complete motion-quality acceptance",
            "quality_approved": False, "new_motion_audit": False,
            "elapsed_seconds": time.monotonic() - START,
            "process_high_water_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}


try:
    result = main()
except Exception as error:
    result = {"schema_version": 1, "status": "failed", "error_type": type(error).__name__, "error": str(error),
              "elapsed_seconds": time.monotonic() - START}
    data = json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    with (OUT / "audit-result.json").open("x") as handle:
        handle.write(data)
    raise
else:
    data = json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    with (OUT / "audit-result.json").open("x") as handle:
        handle.write(data)
    print(json.dumps(result, sort_keys=True))
