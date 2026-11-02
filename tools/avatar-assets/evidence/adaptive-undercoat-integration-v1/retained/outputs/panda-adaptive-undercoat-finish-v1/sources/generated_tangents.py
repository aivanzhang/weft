"""Bounded export tangent derivatives; preserve the native GLB separately."""

import hashlib
import struct
import numpy as np

from avatar_assets.normalize_generated import _array
from avatar_assets.safety import require, sha256
from avatar_assets.three_d import _GLB


MAX_UV1_PRIMITIVES = 4
MAX_UV1_FACES = 2048


def _index(value, count, label):
    require(type(value) is int and 0 <= value < count, 'Invalid '+label)
    return value


def _exclusive_tangents(glb, accessor_index, count, owner):
    doc = glb.doc
    require(not any(value.get('sparse') for value in doc['accessors']), 'UV1 correction rejects sparse storage')
    accessor = doc['accessors'][accessor_index]
    view_index = _index(accessor.get('bufferView'), len(doc['bufferViews']), 'UV1 tangent bufferView')
    view = doc['bufferViews'][view_index]
    require(accessor.get('byteOffset', 0) == 0 and view.get('byteStride', 16) == 16
            and view['byteLength'] == count*16, 'UV1 tangent storage must be exclusive and contiguous')
    for index, other in enumerate(doc['accessors']):
        require(index == accessor_index or other.get('bufferView') != view_index,
                'UV1 tangent bufferView is shared by another accessor')
    start, end = view.get('byteOffset', 0), view.get('byteOffset', 0)+view['byteLength']
    for index, other in enumerate(doc['bufferViews']):
        require(index == view_index or max(start, other.get('byteOffset', 0)) >=
                min(end, other.get('byteOffset', 0)+other['byteLength']), 'UV1 tangent storage overlaps another bufferView')
    uses = []
    for mesh_index, mesh in enumerate(doc.get('meshes', [])):
        for primitive_index, primitive in enumerate(mesh.get('primitives', [])):
            uses.extend((mesh_index, primitive_index, key) for key, value in primitive.get('attributes', {}).items()
                        if value == accessor_index)
            require(primitive.get('indices') != accessor_index and
                    all(accessor_index not in target.values() for target in primitive.get('targets', [])),
                    'UV1 tangent accessor is shared by indices or morph targets')
    require(uses == [(*owner, 'TANGENT')], 'UV1 tangent accessor has ambiguous primitive bindings')
    require(all(image.get('bufferView') != view_index for image in doc.get('images', [])),
            'UV1 tangent storage is shared by an image')
    require(all(skin.get('inverseBindMatrices') != accessor_index for skin in doc.get('skins', [])),
            'UV1 tangent accessor is shared by a skin')
    require(all(sampler.get(key) != accessor_index for animation in doc.get('animations', [])
                for sampler in animation.get('samplers', []) for key in ('input', 'output')),
            'UV1 tangent accessor is shared by animation')
    return start


def _uv1_corrections(glb):
    doc, planned = glb.doc, []
    for mesh_index, mesh in enumerate(doc.get('meshes', [])):
        for primitive_index, primitive in enumerate(mesh.get('primitives', [])):
            material = doc.get('materials', [])[_index(primitive.get('material'),
                len(doc.get('materials', [])), 'UV1 material')]
            normal = material.get('normalTexture')
            if normal is None:
                continue
            uv_index = normal.get('texCoord', 0)
            require(type(uv_index) is int and uv_index in (0, 1), 'Invalid normal-map UV selector')
            if uv_index == 0:
                continue
            require(len(planned) < MAX_UV1_PRIMITIVES, 'UV1 correction exceeds four primitives')
            require(type(primitive.get('mode', 4)) is int and primitive.get('mode', 4) == 4
                    and not primitive.get('targets'), 'UV1 correction requires unmorphed indexed triangles')
            attrs = primitive.get('attributes', {})
            require({'POSITION', 'NORMAL', 'TEXCOORD_1', 'TANGENT'} <= set(attrs),
                    'UV1 correction needs positions, normals, UV1 and tangents')
            ids = {name: _index(attrs[name], len(doc['accessors']), 'UV1 '+name+' accessor')
                   for name in ('POSITION', 'NORMAL', 'TEXCOORD_1', 'TANGENT')}
            index_id = _index(primitive.get('indices'), len(doc['accessors']), 'UV1 index accessor')
            index_spec = doc['accessors'][index_id]
            require(type(index_spec.get('componentType')) is int and index_spec['componentType'] in (5121, 5123, 5125),
                    'UV1 indices must be unsigned integers')
            require(not any(doc['accessors'][value].get('normalized') for value in (*ids.values(), index_id)),
                    'UV1 correction requires unnormalized float frames and integer indices')
            maximum = MAX_UV1_FACES*3
            points = _array(glb, ids['POSITION'], 3, 5126, maximum).astype(np.float64)
            normals = _array(glb, ids['NORMAL'], 3, 5126, maximum).astype(np.float64)
            uv = _array(glb, ids['TEXCOORD_1'], 2, 5126, maximum).astype(np.float64)
            old = _array(glb, ids['TANGENT'], 4, 5126, maximum)
            indices = _array(glb, index_id, 1, index_spec['componentType'], maximum).ravel()
            count = len(points)
            require(len(normals) == len(uv) == len(old) == len(indices) == count and count % 3 == 0
                    and indices.max() < count and np.all(np.bincount(indices, minlength=count) == 1),
                    'UV1 atlas must reference every corner vertex exactly once')
            require(np.all(uv >= -1e-6) and np.all(uv <= 1+1e-6), 'UV1 atlas coordinates exceed bounds')
            sizes = np.linalg.norm(normals, axis=1)
            require(np.all(np.abs(sizes-1) < .001), 'UV1 correction requires unit normals')
            normals /= sizes[:, None]
            offset = _exclusive_tangents(glb, ids['TANGENT'], count, (mesh_index, primitive_index))
            faces = indices.reshape(-1, 3)
            xyz = points[faces]
            # Export flips V, not the baked PNG's green axis or Blender's bitangent sign.
            uv[:, 1] = 1-uv[:, 1]
            chart = uv[faces]
            edge1, edge2 = xyz[:, 1]-xyz[:, 0], xyz[:, 2]-xyz[:, 0]
            a, b = chart[:, 1]-chart[:, 0], chart[:, 2]-chart[:, 0]
            determinant = a[:, 0]*b[:, 1]-a[:, 1]*b[:, 0]
            require(np.all(np.abs(determinant) > 1e-12) and
                    np.all(np.linalg.norm(np.cross(edge1, edge2), axis=1) > 1e-12),
                    'UV1 correction rejects degenerate charts or geometry')
            du = (edge1*b[:, 1, None]-edge2*a[:, 1, None])/determinant[:, None]
            dv = (edge2*a[:, 0, None]-edge1*b[:, 0, None])/determinant[:, None]
            n = normals[faces]
            tangent = du[:, None, :]-n*np.sum(n*du[:, None, :], axis=2)[:, :, None]
            size = np.linalg.norm(tangent, axis=2)
            require(np.all(size > 1e-8), 'UV1 source normals cannot support a tangent frame')
            tangent /= size[:, :, None]
            sign = np.sign(np.sum(np.cross(n, tangent)*dv[:, None, :], axis=2))
            require(np.all(np.abs(sign) == 1), 'UV1 tangent handedness is undefined')
            require(np.all(sign == sign[:, :1]), 'UV1 triangle has inconsistent tangent handedness')
            corrected = np.empty((count, 4), dtype='<f4')
            corrected[faces, :3], corrected[faces, 3] = tangent, sign
            require(np.isfinite(corrected).all(), 'Nonfinite UV1 tangent correction')
            data = corrected.tobytes()
            planned.append({'owner': (mesh_index, primitive_index), 'offset': offset, 'data': data,
                'report': {'mesh': mesh_index, 'primitive': primitive_index, 'faces': count//3,
                    'vertices': count, 'tangent_accessor': ids['TANGENT'],
                    'source_tangents_sha256': hashlib.sha256(old.tobytes()).hexdigest(),
                    'authored_tangents_sha256': hashlib.sha256(data).hexdigest()}})
    return planned


def repair(source, target, *, correct_normal_uv1=False):
    require(type(correct_normal_uv1) is bool, 'UV1 correction must be explicitly boolean')
    require(not target.exists(), 'Tangent derivative already exists')
    glb = _GLB(source)
    raw = bytearray(source.read_bytes())
    require(hashlib.sha256(raw).hexdigest() == glb.hash, 'Native tangent source changed during read')
    json_size = struct.unpack_from('<I', raw, 12)[0]
    binary_start = 28+json_size
    corrections = _uv1_corrections(glb) if correct_normal_uv1 else []
    corrected_owners = {entry['owner'] for entry in corrections}
    for entry in corrections:
        start = binary_start+entry['offset']
        raw[start:start+len(entry['data'])] = entry['data']
    changed, total = [], 0
    for mesh_index, mesh in enumerate(glb.doc.get('meshes', [])):
        for primitive_index, primitive in enumerate(mesh['primitives']):
            attrs = primitive['attributes']
            if 'TANGENT' not in attrs or (mesh_index, primitive_index) in corrected_owners:
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
    report = {'source_sha256': glb.hash, 'result_sha256': sha256(target), 'repaired_vertices': changed,
            'method': 'Orthogonalize rare invalid export tangents; choose an axis fallback only for zero/parallel tangents',
            'geometry_uv_weights_motion_unchanged': True, 'texture_bytes_unchanged': True,
            'limitation': 'Normal-map direction at degenerate UVs is authored, not recovered'}
    if correct_normal_uv1:
        report['normal_uv1_correction'] = {'policy': 'authored-unshared-atlas-blender-v-up-v1',
            'primitives': [entry['report'] for entry in corrections],
            'vertices': sum(entry['report']['vertices'] for entry in corrections),
            'uv_basis': 'Blender bake basis (u, 1-v_export); preserve PNG green-axis handedness',
            'uv1_correction_changes_only_selected_tangent_float4_bytes': True, 'quality_approved': False,
            'limits': {'primitives': MAX_UV1_PRIMITIVES, 'faces_per_primitive': MAX_UV1_FACES}}
    return report
