"""Repair only bounded undefined export tangents; preserve the native GLB separately."""

import struct
import numpy as np

from avatar_assets.normalize_generated import _array
from avatar_assets.safety import require, sha256
from avatar_assets.three_d import _GLB


def repair(source, target):
    require(not target.exists(), 'Tangent derivative already exists')
    glb = _GLB(source)
    raw = bytearray(source.read_bytes())
    json_size = struct.unpack_from('<I', raw, 12)[0]
    binary_start = 28+json_size
    changed, total = [], 0
    for mesh_index, mesh in enumerate(glb.doc.get('meshes', [])):
        for primitive_index, primitive in enumerate(mesh['primitives']):
            attrs = primitive['attributes']
            if 'TANGENT' not in attrs:
                continue
            tangent = _array(glb, attrs['TANGENT'], 4, 5126, 100000)
            normals = _array(glb, attrs['NORMAL'], 3, 5126, 100000)
            lengths = np.linalg.norm(tangent[:, :3], axis=1)
            undefined = np.flatnonzero((lengths < 1e-8) | (np.abs((tangent[:, :3]*normals).sum(axis=1)) >= .002))
            require(np.all((lengths < 1e-8) | (np.abs(lengths-1) < .001)), 'Nonzero invalid tangents need explicit review')
            total += len(tangent)
            a = glb.doc['accessors'][attrs['TANGENT']]
            view = glb.doc['bufferViews'][a['bufferView']]
            for i in undefined:
                n = normals[i].astype(float)
                require(abs(np.linalg.norm(n)-1) < .001, 'Cannot repair tangent without a unit normal')
                axis = tangent[i, :3].astype(float) if lengths[i] > 1e-8 else np.eye(3)[np.argmin(np.abs(n))]
                t = axis-n*np.dot(axis, n)
                if np.linalg.norm(t) < 1e-8:
                    axis = np.eye(3)[np.argmin(np.abs(n))]
                    t = axis-n*np.dot(axis, n)
                t /= np.linalg.norm(t)
                offset = binary_start+view.get('byteOffset', 0)+a.get('byteOffset', 0)+int(i)*view.get('byteStride', 16)
                struct.pack_into('<fff', raw, offset, *t)
                changed.append({'mesh': mesh_index, 'primitive': primitive_index, 'vertex': int(i)})
    require(len(changed) <= 16 and len(changed) <= max(1, total*.001), 'Undefined tangent count exceeds repair profile')
    target.write_bytes(raw)
    return {'source_sha256': glb.hash, 'result_sha256': sha256(target), 'repaired_vertices': changed,
            'method': 'Orthogonalize rare invalid export tangents; choose an axis fallback only for zero/parallel tangents',
            'geometry_uv_weights_motion_unchanged': True, 'texture_bytes_unchanged': True,
            'limitation': 'Normal-map direction at degenerate UVs is authored, not recovered'}
