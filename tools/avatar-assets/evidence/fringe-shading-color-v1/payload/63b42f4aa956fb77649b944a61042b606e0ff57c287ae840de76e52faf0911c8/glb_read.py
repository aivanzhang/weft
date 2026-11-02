"""Unchanged bounded read helpers extracted from the sealed independent attribute audit."""
import json
import mmap
import struct
from pathlib import Path
import numpy as np

DTYPES = {5121: np.dtype("u1"), 5123: np.dtype("<u2"), 5125: np.dtype("<u4"), 5126: np.dtype("<f4")}

WIDTHS = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4}

def require(value, message):
    if not value:
        raise ValueError(message)

class GLB:
    def __init__(self, path):
        self.path = Path(path)
        require(self.path.is_file() and not self.path.is_symlink() and 28 <= self.path.stat().st_size <= 64 * 1024**2,
                "Invalid GLB file bound")
        self.handle = self.path.open("rb")
        self.data = mmap.mmap(self.handle.fileno(), 0, access=mmap.ACCESS_READ)
        magic, version, total, js, jt = struct.unpack_from("<4sIIII", self.data)
        require(magic == b"glTF" and version == 2 and total == len(self.data) and jt == 0x4E4F534A
                and js <= 2*1024**2 and js % 4 == 0, "Invalid GLB JSON header")
        self.doc = json.loads(self.data[20:20+js])
        size, kind = struct.unpack_from("<II", self.data, 20+js)
        self.offset = 28+js
        self.bin_size = size
        require(kind == 0x004E4942 and size % 4 == 0 and self.offset+size == len(self.data), "Invalid BIN chunk")
        require(len(self.doc["buffers"]) == 1 and "uri" not in self.doc["buffers"][0], "External buffer")
        self.declared = self.doc["buffers"][0]["byteLength"]
        require(type(self.declared) is int and 0 <= size-self.declared <= 3, "Invalid embedded buffer size")
        self.decoded = 0

    def array(self, index, kind, components, limit=100000):
        require(type(index) is int and 0 <= index < len(self.doc["accessors"]), "Invalid accessor index")
        a = self.doc["accessors"][index]
        require(a["type"] == kind and a["componentType"] in components and not a.get("sparse")
                and not a.get("normalized", False), "Unexpected accessor representation")
        count, width, dtype = a["count"], WIDTHS[kind], DTYPES[a["componentType"]]
        require(type(count) is int and 0 < count <= limit, "Accessor bound")
        view = self.doc["bufferViews"][a["bufferView"]]
        offset, length, stride = view.get("byteOffset", 0), view["byteLength"], view.get("byteStride", width*dtype.itemsize)
        start = a.get("byteOffset", 0)
        require(view["buffer"] == 0 and all(type(x) is int and x >= 0 for x in (offset, length, stride, start))
                and stride >= width*dtype.itemsize and stride <= 252 and stride % dtype.itemsize == 0
                and offset+length <= self.declared and start+(count-1)*stride+width*dtype.itemsize <= length,
                "Accessor storage exceeds bound")
        result = np.ndarray((count, width), dtype=dtype, buffer=self.data, offset=self.offset+offset+start,
                            strides=(stride, dtype.itemsize)).copy()
        self.decoded += count*width
        require(self.decoded <= 2000000 and np.isfinite(result).all(), "Nonfinite or oversized decoded values")
        return result

    def view_bytes(self, index):
        view = self.doc["bufferViews"][index]
        require(view["buffer"] == 0 and view.get("byteOffset", 0)+view["byteLength"] <= self.declared, "Invalid view")
        start = self.offset+view.get("byteOffset", 0)
        return self.data[start:start+view["byteLength"]]
