"""Bounded local composition of independently owned, embedded unrigged sources."""

from copy import deepcopy
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import stat
import struct
import sys
import time


POLICY = 'unrigged-local-assembly-v1'
MAX_SOURCE_BYTES = 64*1024**2
MAX_MODEL_BYTES = 64*1024**2
MAX_OUTPUT_BYTES = 128*1024**2
MAX_VERTICES = 100000
MAX_TRIANGLES = 100000
MAX_SCALARS = 2000000
MAX_IMAGE_PIXELS = 16*1024**2
MAX_TOTAL_PIXELS = 96*1024**2
MAX_CONFIG_BYTES = 65536
MAX_JSON_BYTES = 2*1024**2
SOURCE_FILES = ('avatar_assets/compose_sources.py', 'avatar_assets/three_d.py',
                'avatar_assets/fringe_export.py', 'avatar_assets/safety.py',
                'avatar_assets/runner.py', 'avatar_assets/__init__.py', 'scripts/blender_creature_rig.py')


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def _write(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')


def _file_state(info):
    return {'device': info.st_dev, 'inode': info.st_ino, 'bytes': info.st_size, 'mtime_ns': info.st_mtime_ns}


def _open_checked(path, size, expected=None):
    info = Path(path).lstat()
    _require(stat.S_ISREG(info.st_mode) and info.st_size == size
             and (expected is None or _file_state(info) == expected), 'Source size, identity or symlink changed')
    descriptor = os.open(path, os.O_RDONLY | getattr(os, 'O_NOFOLLOW', 0))
    stream = os.fdopen(descriptor, 'rb')
    try:
        _require(_file_state(os.fstat(stream.fileno())) == _file_state(info), 'Source identity changed while opening')
    except BaseException:
        stream.close(); raise
    return stream, _file_state(info)


def _copy_bounded(source, target, size, expected=None):
    incoming, state = _open_checked(source, size, expected)
    with incoming, Path(target).open('xb') as saved:
        remaining = size
        while remaining:
            block = incoming.read(min(1024**2, remaining))
            _require(block, 'Source shrank during snapshot')
            saved.write(block); remaining -= len(block)
        _require(not incoming.read(1), 'Source grew during snapshot')
        _require(_file_state(os.fstat(incoming.fileno())) == state, 'Source changed during snapshot')


def _bounded_sha(path, size, expected=None):
    stream, state = _open_checked(path, size, expected)
    digest = hashlib.sha256()
    with stream:
        remaining = size
        while remaining:
            block = stream.read(min(1024**2, remaining))
            _require(block, 'Pinned source shrank during hash')
            digest.update(block); remaining -= len(block)
        _require(not stream.read(1) and _file_state(os.fstat(stream.fileno())) == state,
                 'Pinned source changed during hash')
    return digest.hexdigest()


def _json(data):
    def pairs(rows):
        result = {}
        for key, value in rows:
            _require(key not in result, 'Duplicate JSON key')
            result[key] = value
        return result
    def bad_constant(value):
        raise ValueError('Nonfinite JSON constant')
    return json.loads(data, object_pairs_hook=pairs, parse_constant=bad_constant)


def _number(value, low, high, label):
    _require(type(value) in (int, float) and math.isfinite(value) and low <= value <= high,
             'Invalid bounded '+label)
    return float(value)


def _vector(value, count, low, high, label):
    _require(isinstance(value, list) and len(value) == count, 'Invalid '+label+' vector')
    return [_number(x, low, high, label) for x in value]


def _index(value, count, label):
    _require(type(value) is int and 0 <= value < count, 'Invalid '+label+' index')
    return value


def validate_config(config):
    required = {'schema_version', 'id', 'prompt', 'rights', 'space', 'requested_rig_height_m', 'components'}
    _require(isinstance(config, dict) and set(config) == required
             and type(config['schema_version']) is int and config['schema_version'] == 1,
             'Require exact composition schema_version1 fields')
    _require(isinstance(config['id'], str) and re.fullmatch('[a-z][a-z0-9_-]{0,47}', config['id']),
             'Invalid assembly id')
    for name in ('prompt', 'rights'):
        _require(isinstance(config[name], str) and config[name].strip()
                 and len(config[name].encode('utf-8')) <= 4096, 'Require bounded explicit '+name)
    _require(config['space'] == 'gltf_y_up_right_handed', 'Composition requires explicit glTF right-handed Y-up space')
    _number(config['requested_rig_height_m'], .1, 5, 'requested rig height')
    parts = config['components']
    _require(isinstance(parts, list) and 2 <= len(parts) <= 3, 'Require two or three components')
    ids = set()
    for part in parts:
        _require(isinstance(part, dict) and set(part) == {'id', 'source', 'source_sha256',
                 'source_meters_per_unit', 'placement'}, 'Invalid component fields')
        name = part['id']
        _require(isinstance(name, str) and re.fullmatch('[a-z][a-z0-9_-]{0,47}', name)
                 and name not in ids, 'Invalid or duplicate component id')
        ids.add(name)
        _require(isinstance(part['source'], str) and 1 <= len(part['source']) <= 4096
                 and '\0' not in part['source'], 'Invalid local source path')
        _require(isinstance(part['source_sha256'], str) and re.fullmatch('[0-9a-f]{64}', part['source_sha256']),
                 'Each component requires an exact SHA256')
        _number(part['source_meters_per_unit'], 1e-6, 1000, 'source meters per unit')
        placement = part['placement']
        _require(isinstance(placement, dict) and set(placement) == {
                 'translation_m', 'rotation_xyzw', 'uniform_scale'}, 'Require explicit placement fields')
        _vector(placement['translation_m'], 3, -100, 100, 'translation meters')
        quaternion = _vector(placement['rotation_xyzw'], 4, -1, 1, 'quaternion')
        _require(abs(math.sqrt(sum(x*x for x in quaternion))-1) <= 1e-6,
                 'Placement quaternion must be unit length')
        _number(placement['uniform_scale'], .001, 100, 'positive uniform scale')
    return deepcopy(config)


def _identity_instance(doc):
    meshes, nodes, scenes = doc.get('meshes', []), doc.get('nodes', []), doc.get('scenes', [])
    _require(len(meshes) == 1 and 1 <= len(nodes) <= 256 and len(scenes) == 1
             and type(doc.get('scene', 0)) is int and doc.get('scene', 0) == 0,
             'Require one source mesh and one unambiguous scene')
    _require(not doc.get('skins') and not doc.get('animations') and not doc.get('cameras'),
             'Skins, animations and cameras are unsupported source inputs')
    mesh_nodes = []
    identity = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]
    for index, node in enumerate(nodes):
        _require(set(node) <= {'name', 'mesh', 'children', 'matrix', 'translation', 'rotation', 'scale', 'extras'},
                 'Unsupported source node fields or skin/morph binding')
        if 'matrix' in node:
            _require(not {'translation', 'rotation', 'scale'} & set(node)
                     and _vector(node['matrix'], 16, -1, 1, 'node matrix') == identity,
                     'Source node matrix must be identity without mixed TRS')
        else:
            _require(_vector(node.get('translation', [0, 0, 0]), 3, -10000, 10000, 'node translation') == [0, 0, 0]
                     and _vector(node.get('scale', [1, 1, 1]), 3, -10000, 10000, 'node scale') == [1, 1, 1],
                     'Source node translation/scale must be identity; bake transforms separately')
            q = _vector(node.get('rotation', [0, 0, 0, 1]), 4, -1, 1, 'node rotation')
            _require(q in ([0, 0, 0, 1], [0, 0, 0, -1]), 'Source node rotation must be identity')
        if 'mesh' in node:
            _require(type(node['mesh']) is int and node['mesh'] == 0, 'Invalid or extra source mesh instance')
            mesh_nodes.append(index)
    roots = scenes[0].get('nodes')
    _require(isinstance(roots, list) and 1 <= len(roots) <= 256, 'Invalid active source scene roots')
    active, pending = set(), list(roots)
    while pending:
        index = _index(pending.pop(), len(nodes), 'scene node')
        _require(index not in active, 'Source scene repeats a node or has a cycle')
        active.add(index)
        children = nodes[index].get('children', [])
        _require(isinstance(children, list) and len(children) <= 256, 'Invalid source node children')
        pending.extend(children)
    _require(len(mesh_nodes) == 1 and mesh_nodes[0] in active,
             'Require exactly one active mesh instance; inactive/extra mesh instances are unsupported')
    _require('weights' not in meshes[0] and set(meshes[0]) <= {'name', 'primitives', 'extras'},
             'Morph targets and unknown mesh fields are unsupported')
    return mesh_nodes[0]


def _material_routes(doc):
    materials, textures, images = doc.get('materials', []), doc.get('textures', []), doc.get('images', [])
    samplers = doc.get('samplers', [])
    _require(1 <= len(materials) <= 8 and isinstance(samplers, list) and len(samplers) <= 64,
             'Source requires one to eight explicit standard materials')
    routes = []
    for sampler in samplers:
        _require(isinstance(sampler, dict) and set(sampler) <= {'name', 'wrapS', 'wrapT', 'magFilter', 'minFilter', 'extras'},
                 'Invalid sampler fields')
        for key, choices, default in (('wrapS', (33071, 33648, 10497), 10497),
                ('wrapT', (33071, 33648, 10497), 10497), ('magFilter', (9728, 9729), 9729),
                ('minFilter', (9728, 9729, 9984, 9985, 9986, 9987), 9987)):
            _require(type(sampler.get(key, default)) is int and sampler.get(key, default) in choices, 'Invalid sampler '+key)
    for texture in textures:
        _require(set(texture) <= {'name', 'sampler', 'source', 'extras'}, 'Unsupported texture fields')
        if 'sampler' in texture:
            _index(texture['sampler'], len(samplers), 'texture sampler')
    for material in materials:
        _require(set(material) <= {'name', 'pbrMetallicRoughness', 'normalTexture', 'occlusionTexture',
                 'emissiveTexture', 'emissiveFactor', 'alphaMode', 'alphaCutoff', 'doubleSided', 'extras'},
                 'Unsupported material fields')
        _require(material.get('alphaMode', 'OPAQUE') in ('OPAQUE', 'MASK', 'BLEND')
                 and type(material.get('doubleSided', False)) is bool, 'Invalid material alpha/sidedness')
        _number(material.get('alphaCutoff', .5), 0, 1, 'alpha cutoff')
        _vector(material.get('emissiveFactor', [0, 0, 0]), 3, 0, 1, 'emissive factor')
        pbr = material.get('pbrMetallicRoughness', {})
        _require(isinstance(pbr, dict) and set(pbr) <= {
                 'baseColorTexture', 'baseColorFactor', 'metallicRoughnessTexture', 'metallicFactor', 'roughnessFactor', 'extras'},
                 'Require standard constant-color or textured PBR')
        required_uvs = set()
        for container, keys in ((pbr, ('baseColorTexture', 'metallicRoughnessTexture')),
                                (material, ('normalTexture', 'occlusionTexture', 'emissiveTexture'))):
            for key in keys:
                if key not in container:
                    continue
                info = container[key]
                extras = {'scale'} if key == 'normalTexture' else {'strength'} if key == 'occlusionTexture' else set()
                _require(isinstance(info, dict) and {'index'} <= set(info) <= {'index', 'texCoord', 'extras'} | extras,
                         'Invalid material texture route')
                _index(info['index'], len(textures), 'material texture')
                uv = info.get('texCoord', 0)
                _require(type(uv) is int and uv in (0, 1), 'Only explicit UV0/UV1 routes are supported')
                required_uvs.add('TEXCOORD_'+str(uv))
                if 'scale' in info:
                    _number(info['scale'], -100, 100, 'normal texture scale')
                if 'strength' in info:
                    _number(info['strength'], 0, 1, 'occlusion strength')
        routes.append(required_uvs)
    return routes


def _image_inventory(glb, totals):
    from PIL import Image
    result = []
    for index, image in enumerate(glb.doc.get('images', [])):
        _require(set(image) <= {'name', 'bufferView', 'mimeType', 'extras'}, 'Unsupported embedded image fields')
        view = glb.doc['bufferViews'][image['bufferView']]
        offset, size = view.get('byteOffset', 0), view['byteLength']
        data = glb.binary[offset:offset+size]
        with Image.open(io.BytesIO(data)) as decoded:
            width, height = decoded.size
            _require(0 < width*height <= MAX_IMAGE_PIXELS and max(width, height) <= 16384
                     and getattr(decoded, 'n_frames', 1) == 1, 'Embedded image exceeds16Mi-pixels/single-frame bound')
            totals['image_pixels'] += width*height
            _require(totals['image_pixels'] <= MAX_TOTAL_PIXELS, 'Aggregate image metadata exceeds96Mi-pixels')
            _require((decoded.format, image['mimeType']) in (('PNG', 'image/png'), ('JPEG', 'image/jpeg')),
                     'Embedded image format differs from declared MIME')
            row = {'source_image': index, 'width': width, 'height': height, 'mode': decoded.mode,
                   'format': decoded.format, 'bytes': size, 'packed_sha256': hashlib.sha256(data).hexdigest()}
            decoded.verify()
        result.append(row)
    return result


def _add_counts(totals, *, vertices=0, triangles=0, scalars=0):
    totals['vertices'] += vertices; totals['triangles'] += triangles; totals['decoded_scalars'] += scalars
    _require(totals['vertices'] <= MAX_VERTICES, 'Aggregate geometry exceeds100k vertices')
    _require(totals['triangles'] <= MAX_TRIANGLES, 'Aggregate geometry exceeds100k triangles')
    _require(totals['decoded_scalars'] <= MAX_SCALARS, 'Aggregate decoded scalars exceed2M')


def _metadata_inventory(config):
    """Reject aggregate geometry/table growth before BIN append or attribute decoding."""
    totals = {'vertices': 0, 'triangles': 0, 'decoded_scalars': 0}
    tables = {key: 0 for key in ('bufferViews', 'accessors', 'images', 'textures', 'samplers')}
    rows = []
    for component in config['components']:
        with Path(component['source']).open('rb') as stream:
            header = stream.read(20)
            _require(len(header) == 20, 'Truncated source GLB header')
            size, kind = struct.unpack_from('<II', header, 12)
            _require(kind == 0x4E4F534A and 0 < size <= MAX_JSON_BYTES, 'Source GLB JSON exceeds2MiB or has wrong chunk type')
            encoded = stream.read(size); _require(len(encoded) == size, 'Truncated source GLB JSON')
        doc = _json(encoded)
        _require(isinstance(doc, dict), 'Source GLB JSON must be an object')
        node = _identity_instance(doc)
        primitives = doc['meshes'][0].get('primitives'); accessors = doc.get('accessors')
        _require(isinstance(primitives, list) and 1 <= len(primitives) <= 32
                 and isinstance(accessors, list) and len(accessors) <= 1024, 'Invalid bounded source primitive/accessor tables')
        seen, transforms = set(), set()
        for primitive in primitives:
            _require(isinstance(primitive, dict) and 'indices' in primitive and isinstance(primitive.get('attributes'), dict),
                     'Require indexed source geometry')
            attrs = primitive['attributes']
            _require({'POSITION', 'NORMAL'} <= set(attrs) and set(attrs) <= {
                'POSITION', 'NORMAL', 'TANGENT', 'TEXCOORD_0', 'TEXCOORD_1', 'COLOR_0'},
                'Skins, missing normals and unknown source vertex attributes are unsupported')
            pi = _index(attrs['POSITION'], len(accessors), 'position accessor')
            ni = _index(primitive['indices'], len(accessors), 'triangle accessor')
            count, indices = accessors[pi].get('count'), accessors[ni].get('count')
            _require(type(count) is int and count > 0 and type(indices) is int and indices > 0 and indices % 3 == 0,
                     'Invalid source vertex/triangle counts')
            _add_counts(totals, vertices=count, triangles=indices//3)
            for semantic, index in list(attrs.items())+[('indices', ni)]:
                index = _index(index, len(accessors), 'source accessor'); spec = accessors[index]
                width = {'SCALAR': 1, 'VEC2': 2, 'VEC3': 3, 'VEC4': 4}.get(spec.get('type'))
                _require(width is not None and type(spec.get('count')) is int and spec['count'] > 0,
                         'Invalid source accessor shape/count')
                _require(semantic == 'indices' or spec['count'] == count, 'Source attribute count mismatch')
                if index not in seen:
                    _add_counts(totals, scalars=spec['count']*width); seen.add(index)
                if semantic in ('POSITION', 'NORMAL', 'TANGENT'):
                    transforms.add((index, semantic))
        for key in tables:
            values = doc.get(key, [])
            _require(isinstance(values, list), 'Invalid source '+key+' table')
            tables[key] += len(values)+(len(transforms) if key in ('bufferViews', 'accessors') else 0)
            _require(tables[key] <= (1024 if key in ('bufferViews', 'accessors') else 64), 'Combined GLB tables exceed ingestion limits')
        rows.append({'id': component['id'], 'source_node': node, 'json_bytes': size,
                     'json_sha256': hashlib.sha256(encoded).hexdigest(), 'used_accessors': len(seen),
                     'transformed_accessor_count': len(transforms)})
    return {'totals': totals, 'output_table_counts': tables, 'components': rows}


def _assemble(config, out, receipt):
    import numpy as np
    import PIL
    from avatar_assets.three_d import _GLB
    from avatar_assets.fringe_export import _array
    start = time.monotonic()
    receipt['runtime'] = {'python': sys.version.split()[0], 'numpy': np.__version__, 'pillow': PIL.__version__}
    receipt['metadata_admission'] = _metadata_inventory(config)
    _write(out/'metadata-admission.json', receipt['metadata_admission'])
    totals = {'vertices': 0, 'triangles': 0, 'decoded_scalars': 0, 'image_pixels': 0}
    binary = bytearray()
    doc = {'asset': {'version': '2.0', 'generator': POLICY}, 'scene': 0, 'scenes': [{'nodes': [0]}],
           'nodes': [{'name': config['id'], 'mesh': 0}], 'meshes': [{'name': config['id'], 'primitives': []}],
           'buffers': [], 'bufferViews': [], 'accessors': [], 'materials': [], 'images': [], 'textures': [], 'samplers': []}
    bounds_low, bounds_high = np.full(3, np.inf), np.full(3, -np.inf)

    def append(data):
        binary.extend(b'\0'*(-len(binary) % 4))
        offset = len(binary)
        _require(offset+len(data) <= MAX_MODEL_BYTES, 'Composed binary exceeds64MiB')
        binary.extend(data)
        return offset

    for component in config['components']:
        _require(time.monotonic()-start < 50, 'Composition worker deadline exceeded')
        path = Path(component['source'])
        glb = _GLB(path)
        _require(glb.hash == component['source_sha256'], 'Worker source snapshot SHA256 differs')
        node = _identity_instance(glb.doc)
        routes = _material_routes(glb.doc)
        image_rows = _image_inventory(glb, totals)
        primitives = glb.doc['meshes'][0].get('primitives')
        _require(isinstance(primitives, list) and 1 <= len(primitives) <= 32, 'Source exceeds1..32 primitive bound')
        offsets = {key: len(doc[key]) for key in ('bufferViews', 'accessors', 'materials', 'images', 'textures', 'samplers')}
        base = append(glb.binary)
        for view in glb.doc['bufferViews']:
            value = deepcopy(view); value['byteOffset'] = base+value.get('byteOffset', 0)
            doc['bufferViews'].append(value)
        for accessor in glb.doc['accessors']:
            _require('sparse' not in accessor, 'Sparse accessors are unsupported')
            value = deepcopy(accessor)
            value['bufferView'] = offsets['bufferViews']+_index(value.get('bufferView'), len(glb.doc['bufferViews']), 'accessor bufferView')
            doc['accessors'].append(value)
        material_map = []
        for index, material in enumerate(glb.doc['materials']):
            value = deepcopy(material); value['name'] = component['id']+'__m'+str(index)
            for container, keys in ((value.get('pbrMetallicRoughness', {}), ('baseColorTexture', 'metallicRoughnessTexture')),
                    (value, ('normalTexture', 'occlusionTexture', 'emissiveTexture'))):
                for key in keys:
                    if key in container:
                        container[key]['index'] += offsets['textures']
            doc['materials'].append(value)
            material_map.append({'source_material': index, 'source_name': material.get('name'),
                                 'output_material': offsets['materials']+index, 'output_name': value['name']})
        for image in glb.doc.get('images', []):
            value = deepcopy(image); value['bufferView'] += offsets['bufferViews']; doc['images'].append(value)
        for texture in glb.doc.get('textures', []):
            value = deepcopy(texture); value['source'] += offsets['images']
            if 'sampler' in value:
                value['sampler'] += offsets['samplers']
            doc['textures'].append(value)
        doc['samplers'].extend(deepcopy(glb.doc.get('samplers', [])))
        quaternion = np.asarray(component['placement']['rotation_xyzw'], dtype=float)
        effective_q = quaternion/np.linalg.norm(quaternion)
        x, y, z, w = effective_q
        rotation = np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
            [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
            [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])
        scale = component['source_meters_per_unit']*component['placement']['uniform_scale']
        translation = np.asarray(component['placement']['translation_m'], dtype=float)
        row = {'id': component['id'], 'source_sha256': glb.hash, 'source_node': node, 'source_mesh': 0,
               'source_asset': deepcopy(glb.doc['asset']), 'source_bytes': path.stat().st_size,
               'source_transform': {'formula': 'p_out=t+(source_meters_per_unit*uniform_scale)*R(q)*p_source',
                    'source_meters_per_unit': component['source_meters_per_unit'], 'placement': deepcopy(component['placement']),
                    'effective_unit_quaternion_xyzw': effective_q.tolist(), 'quaternion_normalization_delta': float(np.max(abs(effective_q-quaternion))),
                    'scale_to_meters': scale, 'rotation_matrix_rows': rotation.tolist(), 'no_implicit_centering': True},
               'materials': material_map, 'images': image_rows, 'primitives': []}
        receipt['components'].append(row)
        cache, transformed = {}, {}
        low, high = np.full(3, np.inf), np.full(3, -np.inf)

        def read(index, kind, types, maximum=100000):
            index = _index(index, len(glb.doc['accessors']), 'source accessor')
            spec = glb.doc['accessors'][index]
            _require(spec.get('type') == kind and type(spec.get('componentType')) is int and spec['componentType'] in types,
                     'Unsupported accessor encoding')
            _require(type(spec.get('count')) is int and 0 < spec['count'] <= maximum, 'Accessor count outside bounds')
            if index not in cache:
                _add_counts(totals, scalars=spec['count']*{'SCALAR': 1, 'VEC2': 2, 'VEC3': 3, 'VEC4': 4}[kind])
                cache[index] = _array(glb, index, kind, types, maximum=maximum)
            return cache[index]

        def transformed_accessor(index, semantic, values):
            key = (index, semantic)
            if key in transformed:
                return transformed[key]
            source_values = values.astype(float)
            if semantic == 'POSITION':
                result = (source_values@rotation.T*scale+translation).astype('<f4')
            else:
                result = values.astype('<f4').copy()
                result[:, :3] = (source_values[:, :3]@rotation.T).astype('<f4')
            _require(np.isfinite(result).all() and (semantic != 'POSITION' or np.max(abs(result)) <= 1000),
                     'Transformed geometry exceeds finite1000m bounds')
            offset = append(result.tobytes())
            doc['bufferViews'].append({'buffer': 0, 'byteOffset': offset, 'byteLength': result.nbytes, 'target': 34962})
            spec = deepcopy(glb.doc['accessors'][index]); spec['bufferView'] = len(doc['bufferViews'])-1
            spec.pop('byteOffset', None); spec.pop('min', None); spec.pop('max', None)
            if semantic == 'POSITION':
                spec.update(min=result.min(axis=0).astype(float).tolist(), max=result.max(axis=0).astype(float).tolist())
            doc['accessors'].append(spec); transformed[key] = len(doc['accessors'])-1
            return transformed[key]

        for primitive_index, primitive in enumerate(primitives):
            _require(isinstance(primitive, dict) and set(primitive) <= {'attributes', 'indices', 'material', 'mode', 'extras'}
                     and type(primitive.get('mode', 4)) is int and primitive.get('mode', 4) == 4 and 'indices' in primitive,
                     'Require unmorphed indexed TRIANGLES with no unknown primitive features')
            attrs = primitive.get('attributes')
            _require(isinstance(attrs, dict) and {'POSITION', 'NORMAL'} <= set(attrs)
                     and set(attrs) <= {'POSITION', 'NORMAL', 'TEXCOORD_0', 'TEXCOORD_1', 'TANGENT', 'COLOR_0'},
                     'Require explicit normals; skins and unknown vertex attributes are unsupported')
            material = _index(primitive.get('material'), len(routes), 'primitive material')
            _require(routes[material] <= set(attrs), 'Material refers to missing source UV coordinates')
            position = read(attrs['POSITION'], 'VEC3', {5126})
            normal = read(attrs['NORMAL'], 'VEC3', {5126})
            indices = read(primitive['indices'], 'SCALAR', {5121, 5123, 5125}, 300000)
            _require(len(indices) % 3 == 0 and indices.min() >= 0 and indices.max() < len(position), 'Invalid source triangle indices')
            _add_counts(totals, vertices=len(position), triangles=len(indices)//3)
            _require(len(normal) == len(position) and np.max(abs(np.linalg.norm(normal.astype(float), axis=1)-1)) <= .002,
                     'Source normals must be explicit finite unit vectors')
            value = deepcopy(primitive); value['material'] += offsets['materials']; value['indices'] += offsets['accessors']
            value['attributes'] = {}; attribute_rows = {}
            for semantic, index in attrs.items():
                kind = 'VEC2' if semantic.startswith('TEXCOORD_') else 'VEC4' if semantic == 'TANGENT' else 'VEC3'
                if semantic == 'COLOR_0':
                    kind = glb.doc['accessors'][_index(index, len(glb.doc['accessors']), 'color accessor')].get('type')
                    _require(kind in ('VEC3', 'VEC4'), 'Invalid float color shape')
                array = read(index, kind, {5126})
                _require(len(array) == len(position), 'Source attribute count differs from positions')
                if semantic.startswith('TEXCOORD_'):
                    _require(np.max(abs(array)) <= 1e6, 'Source UV exceeds bounded repeat range')
                if semantic == 'COLOR_0':
                    _require(array.min() >= 0 and array.max() <= 1, 'Source color outside0..1')
                if semantic == 'TANGENT':
                    _require(np.max(abs(np.linalg.norm(array[:, :3].astype(float), axis=1)-1)) <= .002
                             and np.max(abs(np.sum(array[:, :3]*normal, axis=1))) <= .002
                             and np.isin(array[:, 3], [-1., 1.]).all(), 'Invalid source tangent frame')
                output_index = transformed_accessor(index, semantic, array) if semantic in ('POSITION', 'NORMAL', 'TANGENT') else index+offsets['accessors']
                value['attributes'][semantic] = output_index
                attribute_rows[semantic] = {'source_accessor': index, 'output_accessor': output_index,
                    'source_value_bytes_sha256': hashlib.sha256(array.tobytes()).hexdigest(),
                    'operation': 'explicit_position_transform' if semantic == 'POSITION' else 'rotate_xyz_only' if semantic in ('NORMAL', 'TANGENT') else 'exact_original_bytes'}
            output_primitive = len(doc['meshes'][0]['primitives']); doc['meshes'][0]['primitives'].append(value)
            output_position = doc['accessors'][value['attributes']['POSITION']]
            low = np.minimum(low, output_position['min']); high = np.maximum(high, output_position['max'])
            row['primitives'].append({'source_primitive': primitive_index, 'output_primitive': output_primitive,
                'vertices': len(position), 'triangles': len(indices)//3, 'source_material': material,
                'output_material': value['material'], 'output_material_name': doc['materials'][value['material']]['name'],
                'attributes': attribute_rows, 'index_value_bytes_sha256': hashlib.sha256(indices.tobytes()).hexdigest(),
                'tangent_status': 'rotated_xyz_original_w' if 'TANGENT' in attrs else 'absent_preserved_raw_source_not_final_shading_validation'})
        row['assembly_bounds_m'] = {'min': low.tolist(), 'max': high.tolist()}
        bounds_low = np.minimum(bounds_low, low); bounds_high = np.maximum(bounds_high, high)
        receipt['measured_totals'] = dict(totals)
        _require(len(doc['bufferViews']) <= 1024 and len(doc['accessors']) <= 1024
                 and all(len(doc[key]) <= 64 for key in ('images', 'textures', 'samplers')),
                 'Combined GLB tables exceed bounded ingestion limits')
        _write(out/('component-'+component['id']+'-report.json'), row)
        del glb
    _require(bounds_high[1]-bounds_low[1] > 1e-8, 'Zero-height assembly cannot predict requested rig calibration')
    center = np.array([(bounds_low[0]+bounds_high[0])/2, bounds_low[1], (bounds_low[2]+bounds_high[2])/2])
    factor = config['requested_rig_height_m']/(bounds_high[1]-bounds_low[1])
    blender_offset = center[[0, 2, 1]].copy(); blender_offset[1] *= -1
    receipt.update(assembly_bounds_m={'min': bounds_low.tolist(), 'max': bounds_high.tolist()},
        requested_native_canonical_transform={'applied': False, 'native_helper': 'scripts/blender_creature_rig.py:create_rig',
            'requested_height_m': config['requested_rig_height_m'], 'gltf_to_blender': '[x,-z,y]',
            'blender_offset_before_scale': blender_offset.tolist(), 'blender_translation_before_scale': (-blender_offset).tolist(),
            'uniform_scale': float(factor), 'formula': 'p_native=(gltf_to_blender(p_assembly)-blender_offset_before_scale)*uniform_scale',
            'warning': 'Future create_rig recenters and rescales every component; assembly anchors are not preserved as native world positions.'},
        measured_totals=totals, image_verification='Pillow metadata and verify only; no full pixel decode or appearance claim')
    doc['buffers'] = [{'byteLength': len(binary)}]
    for key in ('images', 'textures', 'samplers'):
        if not doc[key]:
            del doc[key]
    encoded = json.dumps(doc, separators=(',', ':'), sort_keys=True, allow_nan=False).encode()
    encoded += b' '*(-len(encoded) % 4); binary.extend(b'\0'*(-len(binary) % 4))
    _require(len(encoded) <= MAX_JSON_BYTES, 'Final assembly GLB JSON exceeds2MiB')
    size = 28+len(encoded)+len(binary)
    _require(size <= MAX_MODEL_BYTES, 'Final assembly GLB exceeds64MiB')
    existing_bytes = sum(p.stat().st_size for p in out.rglob('*') if p.is_file())
    pending_receipt_bytes = len(json.dumps(receipt, sort_keys=True, allow_nan=False).encode())
    _require(existing_bytes+size+pending_receipt_bytes+1024**2 <= MAX_OUTPUT_BYTES,
             'Assembly plus source snapshots/metadata exceeds128MiB reserved output bound')
    with (out/'model.glb').open('xb') as stream:
        stream.write(struct.pack('<4sII', b'glTF', 2, size)); stream.write(struct.pack('<II', len(encoded), 0x4E4F534A)); stream.write(encoded)
        stream.write(struct.pack('<II', len(binary), 0x004E4942)); stream.write(binary)
    checked = _GLB(out/'model.glb')
    _require(not checked.doc.get('skins') and not checked.doc.get('animations'), 'Output unexpectedly contains rig/motion')
    receipt.update(status='unrigged_local_assembly', model_sha256=checked.hash, model_bytes=size,
                   worker_elapsed_seconds=time.monotonic()-start)


def _worker(config_path, output):
    out = Path(output)
    receipt = {'schema_version': 1, 'policy': POLICY, 'status': 'unsupported', 'components': [],
               'rig_created': False, 'animation_verified': False, 'quality_approved': False,
               'paid_calls': 0, 'native_calls': 0}
    try:
        config = validate_config(_json(Path(config_path).read_bytes()))
        _assemble(config, out, receipt)
    except Exception as error:
        receipt.update(status='unsupported', error={'type': type(error).__name__, 'message': str(error)[:2048]})
    _write(out/'worker-result.json', receipt)
    print(json.dumps({'status': receipt['status'], 'error': receipt.get('error')}))
    return 0 if receipt['status'] == 'unrigged_local_assembly' else 2


def compose(config_path, output):
    """Create one new local assembly or retain an explicit unsupported attempt."""
    from avatar_assets.runner import run_bounded
    out = Path(output)
    _require(not out.exists() and not out.is_symlink(), 'Output exists; choose a new composition directory')
    out.mkdir(parents=True, mode=0o700); out = out.resolve()
    started = time.monotonic(); source_rows = []; code_rows = []; status = 'unsupported'; result = {}
    try:
        root = Path(__file__).resolve().parents[1]
        for name in SOURCE_FILES:
            source = root/name; target = out/'sources'/name; target.parent.mkdir(parents=True, exist_ok=True)
            _require(source.stat().st_size <= 1024**2, 'Source-code snapshot exceeds bound')
            _copy_bounded(source, target, source.stat().st_size)
            code_rows.append({'path': name, 'snapshot': target.relative_to(out).as_posix(), 'sha256': _sha(target), 'bytes': target.stat().st_size})
        config_path = Path(config_path)
        _require(not config_path.is_symlink() and config_path.is_file()
                 and config_path.stat().st_size <= MAX_CONFIG_BYTES, 'Require bounded regular config file')
        config_path = config_path.resolve(); original = config_path.read_bytes()
        (out/'config.original.json').write_bytes(original)
        config = validate_config(_json(original)); worker_config = deepcopy(config)
        source_bytes = 0
        for component, copied in zip(config['components'], worker_config['components']):
            source = Path(component['source'])
            if not source.is_absolute():
                source = config_path.parent/source
            _require(not source.is_symlink() and source.is_file() and source.suffix.lower() == '.glb', 'Source must be a regular local GLB')
            source = source.resolve(); original_state = _file_state(source.lstat()); size = original_state['bytes']
            source_bytes += size
            _require(28 <= size <= MAX_SOURCE_BYTES and source_bytes <= MAX_SOURCE_BYTES, 'Combined source bytes exceed64MiB')
            with source.open('rb') as stream:
                magic, version, declared = struct.unpack('<4sII', stream.read(12))
            _require(magic == b'glTF' and version == 2 and declared == size, 'Invalid source GLB header')
            target = out/'inputs'/(component['id']+'.glb'); target.parent.mkdir(mode=0o700, exist_ok=True)
            _copy_bounded(source, target, size, original_state); actual_hash = _bounded_sha(target, size)
            source_rows.append({'id': component['id'], 'original_path': str(source), 'snapshot': target.relative_to(out).as_posix(),
                                'bytes': size, 'sha256': actual_hash, 'requested_sha256': component['source_sha256'],
                                'original_file_state': original_state})
            _require(target.stat().st_size == size and actual_hash == component['source_sha256'], 'Source SHA256 does not match requested pin')
            copied['source'] = str(target)
        _write(out/'config.worker.json', worker_config)
        temp = out/'tmp'; temp.mkdir(mode=0o700)
        command = ['/usr/bin/env', '-i', 'PATH=/usr/bin:/bin', 'TMPDIR='+str(temp), 'LANG=C.UTF-8',
                   'PYTHONDONTWRITEBYTECODE=1', 'OMP_NUM_THREADS=1', 'OPENBLAS_NUM_THREADS=1', 'VECLIB_MAXIMUM_THREADS=1',
                   sys.executable, '-I', '-B', str(out/'sources/avatar_assets/compose_sources.py'),
                   '--worker', str(out/'config.worker.json'), str(out)]
        worker_config_bytes = (out/'config.worker.json').stat().st_size
        worker_config_hash = _bounded_sha(out/'config.worker.json', worker_config_bytes)
        _write(out/'execution.json', {'command': command, 'environment_policy': 'Empty environment plus listed nonsecret variables; no HOME/CODEX_HOME/auth variables',
            'python_executable_sha256': _sha(sys.executable), 'sources': code_rows, 'inputs': source_rows,
            'original_config_sha256': hashlib.sha256(original).hexdigest(), 'worker_config_sha256': worker_config_hash,
            'limits': {'wall_seconds': 60, 'rss_mib': 512, 'output_mib': 128}, 'source_header_only_parent': True})
        job = run_bounded(command, out/'worker.log', out, timeout=60, memory_mb=512, output_mb=128)
        _write(out/'job.json', job)
        if (out/'worker-result.json').is_file():
            result = _json((out/'worker-result.json').read_bytes())
            if result.get('status') == 'unsupported':
                raise ValueError(result.get('error', {}).get('message', 'Worker rejected source composition'))
        _require(job['process_exit_verified'] and not job['failure'] and job['exit_code'] == 0, 'Bounded worker failed or rejected input; see worker-result/job receipts')
        _require(result['status'] == 'unrigged_local_assembly', 'Worker did not produce a supported assembly')
        model = out/'model.glb'
        _require(model.is_file() and not model.is_symlink() and 28 <= model.stat().st_size <= MAX_MODEL_BYTES
                 and isinstance(result.get('model_sha256'), str)
                 and _bounded_sha(model, model.stat().st_size) == result['model_sha256'],
                 'Worker model is absent, oversized or differs from its receipt')
        _require(all(_bounded_sha(row['original_path'], row['bytes'], row['original_file_state']) == row['sha256']
                     and _bounded_sha(out/row['snapshot'], row['bytes']) == row['sha256'] for row in source_rows),
                 'Original or copied source changed during composition')
        _require(all(_bounded_sha(out/row['snapshot'], row['bytes']) == row['sha256'] for row in code_rows),
                 'Copied producer source changed during composition')
        _require(_bounded_sha(out/'config.worker.json', worker_config_bytes) == worker_config_hash,
                 'Prepared worker config changed during composition')
        _require(_bounded_sha(config_path, len(original)) == hashlib.sha256(original).hexdigest()
                 and _bounded_sha(out/'config.original.json', len(original)) == hashlib.sha256(original).hexdigest(),
                 'Original configuration changed during composition')
        status = 'unrigged_local_assembly'
    except Exception as error:
        result = {'status': 'unsupported', 'error': {'type': type(error).__name__, 'message': str(error)[:2048]}}
    manifest = {'schema_version': 1, 'kind': 'unrigged_source_assembly', 'policy': POLICY, 'status': status,
                'rig_created': False, 'animation_verified': False, 'quality_approved': False,
                'components': source_rows, 'source_snapshots': code_rows, 'elapsed_seconds': time.monotonic()-started,
                'rights_scope': 'Authored config declaration and exact original bytes; not independent rights verification',
                'limitations': ['Unrigged local assembly; no recovered attachments, actions or rig.',
                    'No welding, decimation, intersection repair or body smoothing.',
                    'Missing tangents remain missing; raw source composition is not final HD shading validation.',
                    'Multiple source textures are preserved; this does not enable heterogeneous texture grooming.']}
    if status == 'unsupported':
        manifest['error'] = result['error']
    else:
        manifest['model_sha256'] = result['model_sha256']
    rows = []
    for path in sorted(out.rglob('*')):
        _require(not path.is_symlink(), 'Unexpected symlink in composition output')
        if path.is_file():
            rows.append({'path': path.relative_to(out).as_posix(), 'bytes': path.stat().st_size, 'sha256': _sha(path)})
    manifest['artifacts'] = rows
    final_manifest_bytes = len(json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False).encode())+1
    if sum(row['bytes'] for row in rows)+final_manifest_bytes > MAX_OUTPUT_BYTES:
        status = 'unsupported'; manifest['status'] = status
        result = {'error': {'type': 'ValueError', 'message': 'Final output including metadata exceeds128MiB'}}
        manifest['error'] = result['error']; manifest.pop('model_sha256', None)
    _write(out/'manifest.json', manifest)
    return {'out': str(out), 'model_path': str(out/'model.glb') if status == 'unrigged_local_assembly' else None,
            'model_sha256': result.get('model_sha256') if status == 'unrigged_local_assembly' else None,
            'status': status, **({'error': result['error']} if status == 'unsupported' else {})}


if __name__ == '__main__':
    if len(sys.argv) == 4 and sys.argv[1] == '--worker':
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        raise SystemExit(_worker(sys.argv[2], sys.argv[3]))
    raise SystemExit('Use the compose-sources CLI; this private entry point only runs a prepared worker.')
