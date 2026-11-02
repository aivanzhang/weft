"""Read-only source/control audit; no texture decode, Blender, or material mutation."""
import copy
import hashlib
import json
from pathlib import Path
import resource
import struct
import sys
import time

import numpy as np

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
sys.path.insert(0, str(ROOT))


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write(name, value):
    with (OUT/name).open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


class GLB:
    def __init__(self, path):
        require(Path(path).is_file() and not Path(path).is_symlink(), 'Invalid model path')
        require(Path(path).stat().st_size <= 64*1024**2, 'Model exceeds64MiB')
        self.raw = Path(path).read_bytes()
        require(struct.unpack_from('<4sII', self.raw) == (b'glTF', 2, len(self.raw)), 'Invalid GLB header')
        n, kind = struct.unpack_from('<II', self.raw, 12)
        require(kind == 0x4e4f534a and n <= 2*1024**2, 'Invalid GLB JSON')
        self.doc = json.loads(self.raw[20:20+n])
        count, kind = struct.unpack_from('<II', self.raw, 20+n)
        require(kind == 0x004e4942 and 28+n+count == len(self.raw), 'Invalid GLB BIN')
        self.binary = memoryview(self.raw)[28+n:]

    def array(self, index):
        a = self.doc['accessors'][index]
        require('sparse' not in a and 0 < a['count'] <= 300000 and not a.get('normalized'), 'Unsupported accessor')
        view = self.doc['bufferViews'][a['bufferView']]
        require(view.get('buffer') == 0, 'External buffer')
        dtype = np.dtype({5121:'<u1', 5123:'<u2', 5125:'<u4', 5126:'<f4'}[a['componentType']])
        width = {'SCALAR':1, 'VEC2':2, 'VEC3':3, 'VEC4':4}[a['type']]
        step = view.get('byteStride', width*dtype.itemsize)
        start = view.get('byteOffset', 0)+a.get('byteOffset', 0)
        require(start+(a['count']-1)*step+width*dtype.itemsize <= len(self.binary), 'Accessor exceeds BIN')
        result = np.ndarray((a['count'], width), dtype=dtype, buffer=self.binary,
                            offset=start, strides=(step, dtype.itemsize))
        require(np.isfinite(result).all(), 'Nonfinite accessor')
        return result

