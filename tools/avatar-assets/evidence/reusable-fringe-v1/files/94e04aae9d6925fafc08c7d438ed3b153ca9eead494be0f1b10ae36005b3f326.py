"""Read-only census of the existing authored fox; no geometry is generated."""

import hashlib
import io
import json
from pathlib import Path
import struct
import time

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / 'outputs/fox-hd-build-v2/model.glb'
started = time.monotonic()
assert 0 < SOURCE.stat().st_size <= 40 * 1024**2
blob = SOURCE.read_bytes()
assert blob[:4] == b'glTF' and struct.unpack_from('<I', blob, 8)[0] == len(blob)
json_size, kind = struct.unpack_from('<I4s', blob, 12)
assert kind == b'JSON' and json_size <= 2 * 1024**2
doc = json.loads(blob[20:20 + json_size])
binary_size, kind = struct.unpack_from('<I4s', blob, 20 + json_size)
assert kind == b'BIN\0'
binary_start = 28 + json_size
binary = memoryview(blob)[binary_start:binary_start + binary_size]
assert len(doc['meshes']) == 1 and len(doc['skins']) == 1


def accessor(index):
    a = doc['accessors'][index]
    assert not a.get('sparse') and a['count'] <= 300000
    v = doc['bufferViews'][a['bufferView']]
    width = {'SCALAR': 1, 'VEC2': 2, 'VEC3': 3, 'VEC4': 4, 'MAT4': 16}[a['type']]
    dtype = np.dtype({5121: '<u1', 5123: '<u2', 5125: '<u4', 5126: '<f4'}[a['componentType']])
    offset = v.get('byteOffset', 0) + a.get('byteOffset', 0)
    stride = v.get('byteStride', dtype.itemsize * width)
    end = offset + (a['count'] - 1) * stride + width * dtype.itemsize
    assert end <= len(binary)
    return np.ndarray((a['count'], width), dtype=dtype, buffer=binary, offset=offset,
                      strides=(stride, dtype.itemsize))


primitives = []
for pi, primitive in enumerate(doc['meshes'][0]['primitives']):
    material = doc['materials'][primitive['material']]
    attributes = primitive['attributes']
    p = accessor(attributes['POSITION'])
    faces = accessor(primitive['indices']).reshape(-1, 3).astype(np.int32)
    entry = {'primitive': pi, 'material': material['name'], 'vertices': len(p),
             'triangles': len(faces), 'mode': material.get('alphaMode', 'OPAQUE')}
    if material['name'] in ('coat', 'cream', 'dark'):
        n = accessor(attributes['NORMAL'])
        uv = accessor(attributes['TEXCOORD_0'])
        edges = np.sort(faces[:, [[0, 1], [1, 2], [2, 0]]].reshape(-1, 2), axis=1)
        unique, counts = np.unique(edges, axis=0, return_counts=True)
        shared = unique[counts == 2]
        length = np.linalg.norm(p[shared[:, 0]].astype(np.float64) - p[shared[:, 1]], axis=1)
        positions_unique, position_ids = np.unique(p, axis=0, return_inverse=True)
        aliases = np.sort(position_ids[unique], axis=1)
        _, alias_inverse, alias_counts = np.unique(aliases, axis=0, return_counts=True, return_inverse=True)
        alias_shared = (alias_counts[alias_inverse] > 1)[counts == 2]
        texture = material['pbrMetallicRoughness']['baseColorTexture']
        texture_def = doc['textures'][texture['index']]
        image = doc['images'][texture_def['source']]
        view = doc['bufferViews'][image['bufferView']]
        image_bytes = bytes(binary[view.get('byteOffset', 0):view.get('byteOffset', 0) + view['byteLength']])
        with Image.open(io.BytesIO(image_bytes)) as im:
            size, mode = list(im.size), im.mode
        entry.update({'shared_indexed_edges': len(shared), 'coincident_shared_aliases': int(alias_shared.sum()),
                      'span_8_5_to_20_mm': int(((length >= .0085) & (length <= .020)).sum()),
                      'span_8_5_to_10_mm': int(((length >= .0085) & (length <= .010)).sum()),
                      'span_8_5_to_20_mm_without_alias': int(((length >= .0085) & (length <= .020) & ~alias_shared).sum()),
                      'edge_length_quantiles_m': np.quantile(length, [0, .25, .5, .75, 1]).tolist(),
                      'uv_min': uv.min(axis=0).tolist(), 'uv_max': uv.max(axis=0).tolist(),
                      'image_index': texture_def['source'], 'image_size': size, 'image_mode': mode,
                      'texture_coord': texture.get('texCoord', 0), 'sampler': doc.get('samplers', [])[texture_def.get('sampler', 0)],
                      'image_sha256': hashlib.sha256(image_bytes).hexdigest(),
                      'normal_length_range': [float(np.linalg.norm(n, axis=1).min()), float(np.linalg.norm(n, axis=1).max())]})
    primitives.append(entry)

retained = [p for p in primitives if not p['material'].endswith('_cutout_fur')]
base_vertices = sum(p['vertices'] for p in retained)
base_triangles = sum(p['triangles'] for p in retained)
manifest = json.loads((SOURCE.parent / 'manifest.json').read_text())
result = {'kind': 'read_only_existing_authored_fox_census', 'source': str(SOURCE.relative_to(ROOT)),
          'source_sha256': hashlib.sha256(blob).hexdigest(), 'source_bytes': len(blob),
          'origin': manifest.get('origin'), 'generation': manifest.get('generation'),
          'source_subject': manifest.get('subject'), 'primitives': primitives,
          'prospective_card_removal': {'excluded_materials': [p['material'] for p in primitives if p not in retained],
                                      'retained_vertices': base_vertices, 'retained_triangles': base_triangles,
                                      'max_fins_by_vertex_budget': min(4800, (100000 - base_vertices - 4096) // 12),
                                      'max_fins_by_triangle_budget': min(4800, (100000 - base_triangles) // 4),
                                      'derivative_created': False},
          'skins': len(doc['skins']), 'joints': len(doc['skins'][0]['joints']),
          'animations': [a.get('name') for a in doc['animations']],
          'mesh_nodes': [{k: n[k] for k in ('name', 'mesh', 'skin', 'matrix', 'translation', 'rotation', 'scale') if k in n}
                         for n in doc['nodes'] if 'mesh' in n],
          'elapsed_seconds': time.monotonic() - started,
          'limits': ['Indexed-edge lengths only; not full allocator eligibility or selected fin count.',
                     'No pixel decode, new model, native render, visual acceptance, or provider call.',
                     'Fox is an authored existing character, not a fresh Meshy generation.']}
encoded = json.dumps(result, sort_keys=True, indent=2, allow_nan=False) + '\n'
assert len(encoded) < 64000
with (Path(__file__).parent / 'fox-census.json').open('x') as stream:
    stream.write(encoded)
print(json.dumps({k: result[k] for k in ('source_sha256', 'prospective_card_removal', 'joints', 'elapsed_seconds')}))
print(json.dumps([p for p in primitives if p['material'] in ('coat', 'cream', 'dark')]))
