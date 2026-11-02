"""Exact semantic comparison of two immutable fox outputs; no pose reevaluation."""

import hashlib
import json
import math
import mmap
from pathlib import Path
import struct
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
from PIL import Image
from avatar_assets.fur_bindings import load_bindings
from avatar_assets.three_d import _local_matrix, _world_matrices
from avatar_assets.cli import verify_pack

OUT = Path(__file__).resolve().parent
STARTED = time.monotonic()
ID = '_AVATAR_FUR_VERTEX_ID'
WIDTHS = {'SCALAR': 1, 'VEC2': 2, 'VEC3': 3, 'VEC4': 4, 'MAT4': 16}
DTYPES = {5120: 'i1', 5121: 'u1', 5122: '<i2', 5123: '<u2', 5125: '<u4', 5126: '<f4'}


def check(value, message):
    if not value:
        raise ValueError(message)


def deadline():
    check(time.monotonic()-STARTED < 28, 'CPU comparison deadline exceeded')


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def save(name, value):
    encoded = json.dumps(value, sort_keys=True, indent=2, allow_nan=False).encode()+b'\n'
    check(len(encoded) <= 2*1024**2, 'Report exceeds bound')
    with (OUT/name).open('xb') as stream:
        stream.write(encoded)


class Model:
    def __init__(self, path):
        check(path.is_file() and not path.is_symlink() and 20 <= path.stat().st_size <= 64*1024**2,
              'Unsupported model file')
        self.stream = path.open('rb')
        self.data = mmap.mmap(self.stream.fileno(), 0, access=mmap.ACCESS_READ)
        magic, version, total, js, kind = struct.unpack_from('<4sIIII', self.data)
        check(magic == b'glTF' and version == 2 and total == len(self.data) and js <= 2*1024**2
              and kind == 0x4e4f534a, 'Invalid GLB header')
        self.doc = json.loads(self.data[20:20+js])
        length, kind = struct.unpack_from('<II', self.data, 20+js)
        self.offset = 28+js
        check(kind == 0x004e4942 and self.offset+length == total, 'Invalid embedded BIN')
        self.length = length
        check(len(self.doc['nodes']) <= 256 and len(self.doc['accessors']) <= 1024, 'GLB table bound')
        self.decoded = set()
        self.scalar_count = 0
        self.accessor_reports = {}

    def array(self, index):
        deadline()
        a = self.doc['accessors'][index]
        check(not a.get('sparse') and a['type'] in WIDTHS and a['componentType'] in DTYPES,
              'Unsupported accessor representation')
        count, width = a['count'], WIDTHS[a['type']]
        check(type(count) is int and 0 < count <= 300000, 'Accessor count bound')
        if index not in self.decoded:
            self.scalar_count += count*width
            self.decoded.add(index)
        check(self.scalar_count <= 3_000_000, 'Comparison scalar bound')
        dtype = np.dtype(DTYPES[a['componentType']])
        view = self.doc['bufferViews'][a['bufferView']]
        start = view.get('byteOffset', 0)+a.get('byteOffset', 0)
        stride = view.get('byteStride', width*dtype.itemsize)
        end = a.get('byteOffset', 0)+(count-1)*stride+width*dtype.itemsize
        check(view.get('buffer') == 0 and start >= 0 and stride >= width*dtype.itemsize
              and stride <= 252 and end <= view['byteLength']
              and view.get('byteOffset', 0)+view['byteLength'] <= self.length, 'Invalid accessor byte range')
        return np.ndarray((count, width), dtype=dtype, buffer=self.data,
                          offset=self.offset+start, strides=(stride, dtype.itemsize))

    def accessor(self, index):
        if index not in self.accessor_reports:
            values = self.array(index)
            self.accessor_reports[index] = {
                'metadata': {k: v for k, v in self.doc['accessors'][index].items()
                             if k not in ('bufferView', 'byteOffset')},
                'sha256': hashlib.sha256(values.tobytes(order='C')).hexdigest(),
                'value_bytes': int(values.size*values.dtype.itemsize)}
        return self.accessor_reports[index]

    def image(self, index):
        item = self.doc['images'][index]
        check('uri' not in item, 'External image unsupported')
        view = self.doc['bufferViews'][item['bufferView']]
        start, size = self.offset+view.get('byteOffset', 0), view['byteLength']
        check(view['buffer'] == 0 and size <= 64*1024**2 and start+size <= len(self.data), 'Image bound')
        data = memoryview(self.data)[start:start+size]
        digest = hashlib.sha256(data).hexdigest()
        del data
        return {**{k: v for k, v in item.items() if k != 'bufferView'}, 'bytes': size, 'sha256': digest}

    def semantics(self):
        d = self.doc
        node_names = [n['name'] for n in d['nodes']]
        mesh_names = [m['name'] for m in d['meshes']]
        materials = [m['name'] for m in d['materials']]
        check(len(set(node_names)) == len(node_names) and len(set(mesh_names)) == len(mesh_names)
              and len(set(materials)) == len(materials), 'Ambiguous named correspondence')
        nodes = []
        for i, n in enumerate(d['nodes']):
            row = dict(n)
            if 'mesh' in row: row['mesh'] = mesh_names[row['mesh']]
            if 'skin' in row: row['skin'] = d['skins'][row['skin']]['name']
            if 'children' in row: row['children'] = [node_names[x] for x in row['children']]
            nodes.append(row)
        meshes, added = {}, []
        for mi, mesh in enumerate(d['meshes']):
            rows = []
            for pi, primitive in enumerate(mesh['primitives']):
                check(not primitive.get('targets') and primitive.get('mode', 4) == 4, 'Unsupported primitive')
                attrs = dict(primitive['attributes'])
                if ID in attrs:
                    added.append({'mesh': mi, 'primitive': pi, 'accessor': attrs[ID],
                                  'values': self.accessor(attrs.pop(ID))})
                row = {k: v for k, v in primitive.items() if k not in ('attributes', 'indices', 'material')}
                row['material'] = materials[primitive['material']]
                row['attributes'] = {name: self.accessor(ai) for name, ai in sorted(attrs.items())}
                row['indices'] = self.accessor(primitive['indices'])
                rows.append(row)
            meshes[mesh['name']] = {**{k: v for k, v in mesh.items() if k != 'primitives'}, 'primitives': rows}
        skins = []
        for skin in d['skins']:
            row = {k: v for k, v in skin.items() if k not in ('joints', 'inverseBindMatrices', 'skeleton')}
            row['joints'] = [node_names[x] for x in skin['joints']]
            row['inverseBindMatrices'] = self.accessor(skin['inverseBindMatrices'])
            if 'skeleton' in skin: row['skeleton'] = node_names[skin['skeleton']]
            skins.append(row)
        animations, clips = [], []
        for animation in d['animations']:
            row = {k: v for k, v in animation.items() if k not in ('channels', 'samplers')}
            row['channels'] = []
            starts, ends, counts = [], [], []
            for channel in animation['channels']:
                sampler = animation['samplers'][channel['sampler']]
                times = self.array(sampler['input'])[:, 0]
                starts.append(float(times[0])); ends.append(float(times[-1])); counts.append(len(times))
                target = dict(channel['target']); target['node'] = node_names[target['node']]
                row['channels'].append({'target': target, 'interpolation': sampler.get('interpolation', 'LINEAR'),
                                        'input': self.accessor(sampler['input']), 'output': self.accessor(sampler['output'])})
            animations.append(row)
            clips.append({'name': animation['name'], 'channels': len(row['channels']), 'start_seconds': min(starts),
                          'end_seconds': max(ends), 'duration_seconds': max(ends)-min(starts),
                          'sampler_key_count_range': [min(counts), max(counts)]})
        images = [self.image(i) for i in range(len(d.get('images', [])))]
        textures = []
        for texture in d.get('textures', []):
            row = dict(texture); row['source'] = images[row['source']]
            if 'sampler' in row: row['sampler'] = d['samplers'][row['sampler']]
            textures.append(row)
        mats = []
        for material in d['materials']:
            row = json.loads(json.dumps(material))
            for container, fields in ((row.get('pbrMetallicRoughness', {}), ('baseColorTexture', 'metallicRoughnessTexture')),
                                      (row, ('normalTexture', 'occlusionTexture', 'emissiveTexture'))):
                for field in fields:
                    if field in container: container[field]['index'] = textures[container[field]['index']]
            mats.append(row)
        return {'nodes': nodes, 'meshes': meshes, 'skins': skins, 'animations': animations,
                'materials': mats, 'images': images, 'textures': textures,
                'scenes': d.get('scenes'), 'scene': d.get('scene'), 'asset': d['asset']}, added, clips


def file_pixels(path):
    with Image.open(path) as image:
        check(image.format == 'PNG' and image.width*image.height <= 16*1024**2, 'PNG dimension bound')
        pixels = image.tobytes()
        return {'mode': image.mode, 'size': list(image.size), 'pixel_bytes': len(pixels),
                'pixel_sha256': hashlib.sha256(pixels).hexdigest()}


def compare():
    preparation = json.loads((OUT/'prepared.json').read_text())
    for row in preparation['inputs']+preparation['sources']:
        path = ROOT/row['path']
        check(not path.is_symlink() and path.stat().st_size == row['bytes'] and sha(path) == row['sha256'],
              'Input/source binding changed: '+row['path'])
    packs = [ROOT/name for name in preparation['packs']]
    models = [Model(pack/'model.glb') for pack in packs]
    semantic = [model.semantics() for model in models]
    comparisons = {key: semantic[0][0][key] == semantic[1][0][key] for key in semantic[0][0]}
    save('model-semantics-old.json', semantic[0][0])
    save('model-semantics-new.json', semantic[1][0])
    save('model-comparison.json', {'comparisons': comparisons, 'old_identity_attributes': semantic[0][1],
                                 'new_identity_attributes': semantic[1][1],
                                 'scope': 'Exact typed accessor bytes in original vertex/index order, after resolving table references by names/content. Only the declared new identity attribute is excluded.'})
    reports = [json.loads((pack/'finish-report.json').read_text()) for pack in packs]
    archives = []
    old_parts, new_parts = [report['portable_fur'] for report in reports]
    check(len(old_parts) == len(new_parts) == 3, 'Expected exactly three recorded coats')
    for old, new in zip(old_parts, new_parts):
        a, b = [part['root_summary']['binding_archive'] for part in (old, new)]
        check(a['path'] == b['path'], 'Archive names/order differ')
        arrays = [load_bindings(pack, receipt) for pack, receipt in zip(packs, (a, b))]
        rows = {}
        for name in ('triangle_index', 'barycentric', 'root_inset'):
            x, y = [values[name] for values in arrays]
            rows[name] = {'dtype': x.dtype.str, 'shape': list(x.shape),
                          'old_sha256': hashlib.sha256(x.tobytes()).hexdigest(),
                          'new_sha256': hashlib.sha256(y.tobytes()).hexdigest(),
                          'exact': x.dtype == y.dtype and x.shape == y.shape and x.tobytes() == y.tobytes()}
        added = arrays[1]['source_vertex_ids']
        archives.append({'path': a['path'], 'roots': a['roots'], 'old_schema': a['schema_version'],
                         'new_schema': b['schema_version'], 'arrays': rows,
                         'new_source_ids': {'shape': list(added.shape), 'sha256': hashlib.sha256(added.tobytes()).hexdigest()}})
    save('binding-comparison.json', {'coats': archives, 'total_roots': sum(row['roots'] for row in archives)})
    pixels = []
    pngs = preparation['png_paths']
    check(len(pngs) <= 32 and len([p for p in pngs if p.startswith('renders/')]) == 20, 'PNG inventory bound')
    for name in pngs:
        deadline()
        a, b = [file_pixels(pack/name) for pack in packs]
        pixels.append({'path': name, 'old': a, 'new': b, 'pixels_exact': a == b,
                       'encoded_bytes_exact': sha(packs[0]/name) == sha(packs[1]/name)})
    save('pixel-comparison.json', {'images': pixels, 'portrait_and_draft_poses': 21, 'atlases': len(pixels)-21})
    d = models[1].doc
    parents = {child: i for i, node in enumerate(d['nodes']) for child in node.get('children', [])}
    world = _world_matrices(d['nodes'], parents)
    skin = d['skins'][0]
    bone_rows = [{'palette_index': slot, 'node_index': node, 'name': d['nodes'][node]['name'],
                  'parent_node': parents.get(node), 'parent_name': d['nodes'][parents[node]]['name'] if node in parents else None,
                  'local_transform_gltf_column_major': list(_local_matrix(d['nodes'][node])),
                  'world_rest_translation_m': list(world[node][12:15])}
                 for slot, node in enumerate(skin['joints'])]
    position_min, position_max = np.full(3, np.inf), np.full(3, -np.inf)
    counts = []
    for node_index, node in enumerate(d['nodes']):
        if 'mesh' not in node: continue
        vertex_count, triangle_count = 0, 0
        for primitive in d['meshes'][node['mesh']]['primitives']:
            positions = models[1].array(primitive['attributes']['POSITION'])
            position_min = np.minimum(position_min, positions.min(axis=0))
            position_max = np.maximum(position_max, positions.max(axis=0))
            vertex_count += len(positions); triangle_count += d['accessors'][primitive['indices']]['count']//3
        counts.append({'node_index': node_index, 'node': node['name'], 'mesh_index': node['mesh'],
                       'mesh': d['meshes'][node['mesh']]['name'], 'skin': node['skin'],
                       'rest_local_matrix': list(_local_matrix(node)), 'rest_world_matrix': list(world[node_index]),
                       'vertices': vertex_count, 'triangles': triangle_count})
    inventory = {'space': 'glTF right-handed +Y up, meters by format/producer convention; not measured physical scale',
                 'blender_conversion': '(x,y,z) Blender -> (x,z,-y) glTF',
                 'nodes': counts, 'skin_index': 0, 'skin_name': skin['name'], 'bones': bone_rows,
                 'clips': semantic[1][2], 'all_mesh_bounds_gltf': {'min': position_min.tolist(), 'max': position_max.tolist()},
                 'source_bounds_blender': reports[1]['source_bounds_blender_z_up'],
                 'rest_pose': 'Saved source rest matrices; not assumed T-pose or recovered anatomy.',
                 'loop_verified': False, 'weft_runtime_verified': False}
    save('integration-inventory.json', inventory)
    del positions
    for model in models:
        model.data.close(); model.stream.close()
    verify_started = time.monotonic()
    public_verification = verify_pack(packs[1])
    save('public-verify-pack.json', {'elapsed_seconds': time.monotonic()-verify_started,
                                   'result': public_verification, 'native_calls': 0})
    configs = [json.loads((pack/'sources/finish-design.json').read_text()) for pack in packs]
    changed = {key: {'old': configs[0].get(key), 'new': configs[1].get(key)} for key in sorted(set(configs[0])|set(configs[1]))
               if configs[0].get(key) != configs[1].get(key)}
    all_exact = all(comparisons.values()) and all(row['pixels_exact'] for row in pixels)
    all_exact = all_exact and all(value['exact'] for row in archives for value in row['arrays'].values())
    result = {'schema_version': 1, 'status': 'exact_semantics_preserved' if all_exact else 'differences_recorded',
              'passed': all_exact, 'model_comparisons': comparisons, 'config_differences': changed,
              'added_identity_attribute_occurrences': len(semantic[1][1]), 'bindings_roots': 3000,
              'binding_common_arrays_exact': all(value['exact'] for row in archives for value in row['arrays'].values()),
              'png_pixels_exact': sum(row['pixels_exact'] for row in pixels), 'png_count': len(pixels),
              'new_vertices': sum(row['vertices'] for row in counts), 'new_triangles': sum(row['triangles'] for row in counts),
              'animations_rerun': False, 'attachment_audit_rerun': False, 'quality_verified': False,
              'public_pack_verification': {'artifact_hashes_verified': public_verification['artifact_hashes_verified'],
                                           'status': public_verification['status']},
              'native_calls': 0, 'provider_calls': 0, 'role_calls': 0,
              'scope': 'Metadata addition preservation, not new fur-quality or action acceptance.',
              'elapsed_seconds': time.monotonic()-STARTED}
    for row in preparation['inputs']+preparation['sources']:
        path = ROOT/row['path']
        check(path.stat().st_size == row['bytes'] and sha(path) == row['sha256'], 'Input changed during comparison')
    save('result.json', result)
    print(json.dumps(result, sort_keys=True))


if __name__ == '__main__':
    try:
        compare()
    except Exception as error:
        save('failure.json', {'status': 'failed', 'error_type': type(error).__name__, 'error': str(error),
                              'elapsed_seconds': time.monotonic()-STARTED, 'quality_verified': False})
        raise
