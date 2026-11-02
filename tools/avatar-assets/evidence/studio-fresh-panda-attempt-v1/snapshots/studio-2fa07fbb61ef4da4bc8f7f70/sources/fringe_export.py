"""Omit unused fringe tangent bindings without changing any binary attribute data."""

from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import struct
import time

import numpy as np

from avatar_assets.safety import local_file, require, sha256
from avatar_assets.three_d import _GLB


MAX_BYTES = 64*1024**2
POLICY = 'omit-unused-fringe-tangent-bindings-v1'


def _index(value, count, label):
    require(type(value) is int and 0 <= value < count, 'Invalid '+label)
    return value


def _scalar(value, expected):
    # Blender exports these exact float32 round-trips, not the decimal literals.
    quantized = struct.unpack('<f', struct.pack('<f', expected))[0]
    return type(value) in (int, float) and math.isfinite(value) and value in (expected, quantized)


def _material(material):
    allowed = {'name', 'pbrMetallicRoughness', 'alphaMode', 'alphaCutoff', 'doubleSided', 'extras'}
    require(set(material) <= allowed and material.get('alphaMode') == 'MASK'
            and _scalar(material.get('alphaCutoff'), .18) and material.get('doubleSided', False) is False,
            'Fringe material must be single-sided MASK .18 with no normal map/extensions or other shader channels')
    pbr = material.get('pbrMetallicRoughness')
    require(isinstance(pbr, dict) and set(pbr) <= {'baseColorTexture', 'baseColorFactor', 'metallicFactor', 'roughnessFactor', 'extras'}
            and _scalar(pbr.get('metallicFactor'), 0) and _scalar(pbr.get('roughnessFactor'), .92),
            'Fringe material requires metallic0/roughness.92 and basecolor texture only')
    texture = pbr.get('baseColorTexture')
    require(isinstance(texture, dict) and {'index'} <= set(texture) <= {'index', 'texCoord', 'extras'}
            and type(texture.get('texCoord', 0)) is int and texture.get('texCoord', 0) == 0,
            'Fringe basecolor requires direct UV0 without extension overrides')


def _array(glb, index, kind, components, maximum=100000):
    index = _index(index, len(glb.doc['accessors']), 'fringe accessor')
    accessor = glb.doc['accessors'][index]
    count, component = accessor.get('count'), accessor.get('componentType')
    require(type(count) is int and 0 < count <= maximum and accessor.get('type') == kind
            and type(component) is int and component in components and 'sparse' not in accessor
            and accessor.get('normalized', False) is False, 'Invalid fringe accessor shape/count/encoding')
    width = {'SCALAR': 1, 'VEC2': 2, 'VEC3': 3, 'VEC4': 4}[kind]
    dtype = np.dtype({5121: 'u1', 5123: '<u2', 5125: '<u4', 5126: '<f4'}[component])
    view = glb.doc['bufferViews'][_index(accessor.get('bufferView'), len(glb.doc['bufferViews']), 'fringe bufferView')]
    begin, start, stride = view.get('byteOffset', 0), accessor.get('byteOffset', 0), view.get('byteStride', width*dtype.itemsize)
    require(all(type(v) is int and v >= 0 for v in (begin, start, stride))
            and width*dtype.itemsize <= stride <= 252 and stride % dtype.itemsize == 0
            and (begin+start) % dtype.itemsize == 0
            and start+(count-1)*stride+width*dtype.itemsize <= view['byteLength'], 'Invalid fringe accessor range')
    values = np.ndarray((count, width), dtype=dtype, buffer=glb.binary,
                        offset=begin+start, strides=(stride, dtype.itemsize)).copy()
    require(np.isfinite(values).all(), 'Nonfinite fringe attribute')
    return values


def _strict_json(data):
    def pairs(rows):
        result = {}
        for key, value in rows:
            require(key not in result, 'Duplicate GLB JSON key')
            result[key] = value
        return result
    def invalid(value):
        raise ValueError('Nonfinite GLB JSON value')
    return json.loads(data, object_pairs_hook=pairs, parse_constant=invalid)


def omit_unused_fringe_tangents(source: Path, target: Path, material_names: list[str]) -> dict:
    """Remove only selected TANGENT bindings; preserve unused accessors and padded BIN."""
    started = time.monotonic()
    require(isinstance(source, (str, Path)) and isinstance(target, (str, Path)), 'Require local source/target paths')
    require(isinstance(material_names, list) and 1 <= len(material_names) <= 4
            and all(isinstance(n, str) and n.strip() and len(n) <= 128 for n in material_names)
            and len(set(material_names)) == len(material_names), 'Require one to four exact unique material names')
    target = Path(target)
    require(not target.exists() and not target.is_symlink(), 'Fringe derivative target already exists')
    require(target.parent.is_dir() and not target.parent.is_symlink(), 'Fringe target parent must be an existing local directory')
    source = local_file(source, MAX_BYTES)
    require(source != target.resolve(), 'Fringe derivative cannot replace its source')
    try:
        glb = _GLB(source)
        raw = source.read_bytes()
        require(hashlib.sha256(raw).hexdigest() == glb.hash, 'Fringe source changed during read')
        json_size = struct.unpack_from('<I', raw, 12)[0]
        require(_strict_json(raw[20:20+json_size]) == glb.doc, 'Ambiguous GLB JSON')
        binary_header = 20+json_size
        binary = memoryview(raw)[binary_header+8:]
        doc = glb.doc
        material_ids = {}
        for name in material_names:
            matches = [i for i, material in enumerate(doc.get('materials', [])) if material.get('name') == name]
            require(len(matches) == 1, 'Missing or ambiguous fringe material: '+name)
            material_ids[matches[0]] = name
            _material(doc['materials'][matches[0]])
        selected, vertex_total, index_total = [], 0, 0
        for mi, mesh in enumerate(doc.get('meshes', [])):
            for pi, primitive in enumerate(mesh.get('primitives', [])):
                attrs = primitive.get('attributes', {})
                spec = doc['accessors'][_index(attrs.get('POSITION'), len(doc['accessors']), 'position accessor')]
                count = spec.get('count')
                require(type(count) is int and 0 < count <= 100000, 'Invalid source vertex count')
                vertex_total += count
                indices = doc['accessors'][_index(primitive['indices'], len(doc['accessors']), 'index accessor')] if 'indices' in primitive else spec
                amount = indices.get('count')
                require(type(amount) is int and 0 < amount <= 300000, 'Invalid source index count')
                index_total += amount
                require(vertex_total <= 100000 and index_total <= 300000, 'Source exceeds100k vertex/triangle bounds')
                material = primitive.get('material')
                if type(material) is not int or material not in material_ids:
                    continue
                require(len(selected) < 4, 'Fringe omission exceeds four matching primitives')
                require(type(primitive.get('mode', 4)) is int and primitive.get('mode', 4) == 4
                        and 'indices' in primitive and 'targets' not in primitive and 'weights' not in mesh
                        and 'extensions' not in primitive,
                        'Fringe omission requires unmorphed indexed TRIANGLES without primitive extensions')
                require({'POSITION', 'NORMAL', 'TEXCOORD_0', 'JOINTS_0', 'WEIGHTS_0'} <= set(attrs),
                        'Fringe omission requires actual skinned position/normal/UV0 attributes')
                instances = [node for node in doc.get('nodes', []) if node.get('mesh') == mi]
                require(instances, 'Selected fringe mesh has no actual skinned node')
                joint_counts = []
                for node in instances:
                    require('weights' not in node, 'Morphed fringe nodes are unsupported')
                    skin = doc['skins'][_index(node.get('skin'), len(doc.get('skins', [])), 'fringe skin')]
                    joints = skin.get('joints')
                    require(isinstance(joints, list) and 1 <= len(joints) <= 64
                            and len(set(joints)) == len(joints), 'Invalid fringe joint list')
                    for joint in joints:
                        _index(joint, len(doc['nodes']), 'fringe joint node')
                    joint_counts.append(len(joints))
                positions = _array(glb, attrs['POSITION'], 'VEC3', (5126,))
                normals = _array(glb, attrs['NORMAL'], 'VEC3', (5126,))
                uv = _array(glb, attrs['TEXCOORD_0'], 'VEC2', (5126,))
                joints = _array(glb, attrs['JOINTS_0'], 'VEC4', (5121, 5123))
                weights = _array(glb, attrs['WEIGHTS_0'], 'VEC4', (5126,))
                faces = _array(glb, primitive['indices'], 'SCALAR', (5121, 5123, 5125), 300000).ravel()
                require(len(positions) == len(normals) == len(uv) == len(joints) == len(weights) == count
                        and len(faces) % 3 == 0 and faces.max() < count, 'Mismatched fringe attributes or triangle indices')
                require(np.max(np.abs(np.sum(normals.astype(float)**2, axis=1)-1)) < .001, 'Fringe normals must remain unit vectors')
                require(joints.max() < min(joint_counts) and np.all(weights >= 0)
                        and np.max(np.abs(weights.astype(float).sum(axis=1)-1)) <= .001, 'Invalid actual fringe skin weights')
                tangent_index, zero_count, tangent_hash = attrs.get('TANGENT'), 0, None
                if 'TANGENT' in attrs:
                    tangent = _array(glb, tangent_index, 'VEC4', (5126,))
                    require(len(tangent) == count, 'Fringe tangent count mismatch')
                    zero_count = int(np.all(tangent[:, :3] == 0, axis=1).sum())
                    tangent_hash = hashlib.sha256(tangent.tobytes()).hexdigest()
                selected.append({'mesh': mi, 'primitive': pi, 'material': material,
                    'material_name': material_ids[material], 'vertices': count, 'triangles': len(faces)//3,
                    'tangent_accessor': tangent_index, 'zero_xyz_tangents': zero_count,
                    'tangent_values_sha256': tangent_hash, 'binding_removed': 'TANGENT' in attrs})
        require(selected and {row['material'] for row in selected} == set(material_ids), 'Requested fringe material is unused')
        result = deepcopy(doc)
        changes = []
        for row in selected:
            if row['binding_removed']:
                del result['meshes'][row['mesh']]['primitives'][row['primitive']]['attributes']['TANGENT']
                changes.append(f"/meshes/{row['mesh']}/primitives/{row['primitive']}/attributes/TANGENT")
        encoded = json.dumps(result, separators=(',', ':'), ensure_ascii=True, allow_nan=False).encode()
        encoded += b' '*(-len(encoded) % 4)
        require(len(encoded) <= 2*1024**2 and 28+len(encoded)+len(binary) <= MAX_BYTES, 'Fringe derivative exceeds GLB bounds')
        require(_strict_json(encoded) == result, 'Fringe JSON serialization changed values')
        require(time.monotonic()-started < 10 and sha256(source) == glb.hash, 'Fringe source changed or CPU deadline exceeded')
    except (KeyError, IndexError, TypeError, AttributeError, OverflowError, struct.error, UnicodeError, RecursionError) as error:
        raise ValueError('Malformed fringe export data') from error

    # Exclusive creation prevents a racing overwrite; binary bytes include original padding.
    with target.open('xb') as handle:
        if not changes:
            handle.write(raw)
        else:
            handle.write(struct.pack('<4sII', b'glTF', 2, 28+len(encoded)+len(binary)))
            handle.write(struct.pack('<II', len(encoded), 0x4E4F534A))
            handle.write(encoded)
            handle.write(memoryview(raw)[binary_header:])
    require(sha256(source) == glb.hash, 'Fringe source changed after export')
    binary_hash = hashlib.sha256(binary).hexdigest()
    with target.open('rb') as handle:
        header = handle.read(20)
        size = struct.unpack_from('<I', header, 12)[0]
        require(_strict_json(handle.read(size)) == result, 'Written fringe JSON differs from allowlist')
        length, kind = struct.unpack('<II', handle.read(8))
        require(kind == 0x004E4942 and length == len(binary), 'Written fringe BIN header changed')
        written_hash = hashlib.file_digest(handle, 'sha256').hexdigest()
        require(written_hash == binary_hash, 'Written fringe BIN bytes changed')
    return {'schema_version': 1, 'policy': POLICY, 'source_sha256': glb.hash, 'result_sha256': sha256(target),
            'source_unchanged': True, 'selected_primitives': selected, 'removed_bindings': len(changes),
            'json_changed_paths': changes, 'json_allowlist_verified': True,
            'padded_binary_bytes': len(binary), 'padded_binary_sha256': binary_hash,
            'result_padded_binary_sha256': written_hash,
            'binary_bytes_unchanged': True, 'all_accessors_buffer_views_materials_nodes_skins_animations_preserved': True,
            'normals_positions_uv_skin_weights_motion_unchanged': True, 'other_attributes_unchanged': True,
            'generator_unchanged': True, 'source_vertex_entries': vertex_total, 'source_index_entries': index_total,
            'material_admission': 'Exact .18/.92 policy scalars or their exact float32 round-trips; omitted doubleSided means false',
            'elapsed_seconds': time.monotonic()-started, 'quality_approved': False,
            'limits': {'file_bytes': MAX_BYTES, 'vertices': 100000, 'triangle_indices': 300000, 'selected_primitives': 4},
            'limitation': 'Unused attribute binding omission only; strict material/rig/render validation remains required.'}
