"""Append-only, exact-face upstream UV derivative; no texture decoding or geometry change."""
import copy
import hashlib
import json
from pathlib import Path
import resource
import struct
import sys
import time

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / 'sources'))
from avatar_assets.three_d import validate_render_attributes

PREP = json.loads((HERE / 'prepared.json').read_text())
OUT = Path(PREP['output'])
START = time.monotonic()


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write(name, value):
    with (OUT / name).open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')


class GLB:
    def __init__(self, path):
        self.raw = Path(path).read_bytes()
        require(len(self.raw) <= 64 * 1024**2, 'GLB exceeds64MiB')
        magic, version, length = struct.unpack_from('<4sII', self.raw)
        require(magic == b'glTF' and version == 2 and length == len(self.raw), 'Invalid GLB')
        size, kind = struct.unpack_from('<II', self.raw, 12)
        require(kind == 0x4E4F534A and size <= 2 * 1024**2, 'Invalid JSON')
        self.doc = json.loads(self.raw[20:20+size])
        offset = 20 + size
        size, kind = struct.unpack_from('<II', self.raw, offset)
        require(kind == 0x004E4942 and offset+8+size == len(self.raw), 'Invalid BIN')
        self.binary = memoryview(self.raw)[offset+8:]

    def array(self, index):
        a = self.doc['accessors'][index]
        require('sparse' not in a and a['count'] <= 300000, 'Unsupported accessor')
        v = self.doc['bufferViews'][a['bufferView']]
        dtype = {5121: '<u1', 5123: '<u2', 5125: '<u4', 5126: '<f4'}[a['componentType']]
        width = {'SCALAR': 1, 'VEC2': 2, 'VEC3': 3, 'VEC4': 4}[a['type']]
        item = np.dtype(dtype).itemsize
        return np.ndarray((a['count'], width), dtype=dtype, buffer=self.binary,
            offset=v.get('byteOffset', 0)+a.get('byteOffset', 0),
            strides=(v.get('byteStride', width*item), item))


def main():
    for row in PREP['inputs'].values():
        require(sha(row['path']) == row['sha256'], 'Changed input')
    source = GLB(PREP['inputs']['source']['path'])
    selector = json.loads(Path(PREP['inputs']['selector']['path']).read_text())
    summary = json.loads(Path(PREP['inputs']['summary']['path']).read_text())
    require(selector['source_sha256'] == PREP['inputs']['source']['sha256'], 'Source selector hash mismatch')
    require(summary['actual_witness_coverage']['covered'] == 17 and summary['protected_white_intersection'] == 0,
        'Coverage/protection plan failed')
    require(not summary['geometry_repair_intersection'] and not summary['geometry_healthy_reference_intersection'], 'Geometry overlap')
    original = source.doc
    require(len(original['meshes']) == 1 and len(original['meshes'][0]['primitives']) == 4, 'Unsupported source mesh structure')
    require(not original.get('extensionsUsed') and not original.get('extensionsRequired'), 'Unsupported extensions')
    require(not original.get('skins') and not original.get('animations'), 'This exact v7 source is unrigged')
    doc = copy.deepcopy(original)
    data = bytearray(source.binary)
    new_uv = np.asarray(summary['palette']['uv_triangle_gltf_v_down'], dtype='<f4')
    require(new_uv.shape == (3, 2) and np.isfinite(new_uv).all(), 'Invalid frozen UV triangle')
    prims = doc['meshes'][0]['primitives']
    face_map, vertex_map, material_reports = [], [], []
    source_face_total = 0

    def append_array(values, old_spec=None, *, kind=None, target=34962):
        values = np.ascontiguousarray(values)
        require(np.isfinite(values).all(), 'Nonfinite appended values')
        while len(data) % 4:
            data.append(0)
        start = len(data); data.extend(values.tobytes())
        vi = len(doc['bufferViews'])
        doc['bufferViews'].append({'buffer': 0, 'byteOffset': start, 'byteLength': values.nbytes, 'target': target})
        a = copy.deepcopy(old_spec) if old_spec is not None else {}
        a.pop('byteOffset', None); a.pop('sparse', None)
        a['bufferView'] = vi; a['count'] = len(values)
        if kind is not None:
            a.update(type=kind, componentType=5126 if values.dtype.kind == 'f' else 5125)
        for key, fn in (('min', np.min), ('max', np.max)):
            if key in a or a['type'] == 'VEC3' and old_spec is not None and 'min' in old_spec:
                a[key] = fn(values, axis=0).tolist()
        ai = len(doc['accessors']); doc['accessors'].append(a)
        return ai

    selected = {(r['source_primitive'], r['source_triangle']) for r in selector['rows']}
    require(len(selected) == 249, 'Frozen selector count mismatch')
    for spi, primitive in enumerate(original['meshes'][0]['primitives']):
        require(primitive.get('mode', 4) == 4 and not primitive.get('targets') and not primitive.get('extensions'), 'Unsupported source primitive')
        attrs = primitive['attributes']
        require(set(attrs) == {'POSITION', 'NORMAL', 'TEXCOORD_0', 'TANGENT'}, 'Unexpected exact v7 attributes')
        faces = source.array(primitive['indices']).reshape(-1, 3)
        source_face_total += len(faces)
        picked = np.asarray(sorted(fid for pi, fid in selected if pi == spi), dtype=np.int32)
        count = original['accessors'][attrs['POSITION']]['count']
        require(np.array_equal(np.unique(faces), np.arange(count)), 'Source contains unreferenced vertices')
        if not len(picked):
            face_map.extend((spi, fid, spi, fid) for fid in range(len(faces)))
            vertex_map.extend((spi, i, spi, i) for i in range(count))
            continue
        require(spi in (0, 2) and picked.min() >= 0 and picked.max() < len(faces), 'Invalid selected primitive')
        for row in selector['rows']:
            if row['source_primitive'] == spi:
                fid = row['source_triangle']
                require(np.array_equal(faces[fid], row['source_vertex_rows']), 'Selected corner IDs changed')
                require(np.array_equal(source.array(attrs['TEXCOORD_0'])[faces[fid]], np.asarray(row['source_uv0'], dtype='<f4')), 'Selected UV changed')
        keep = np.ones(len(faces), dtype=bool); keep[picked] = False
        kept_ids = np.flatnonzero(keep)
        used = np.unique(faces[keep]); remap = np.full(count, -1, dtype=np.int32); remap[used] = np.arange(len(used))
        # Compact retained primitives so duplicate UV seams do not leave loose source vertices.
        retained = copy.deepcopy(primitive)
        retained['attributes'] = {name: append_array(source.array(ai)[used], original['accessors'][ai]) for name, ai in attrs.items()}
        retained['indices'] = append_array(remap[faces[keep]].astype('<u4').reshape(-1, 1), kind='SCALAR', target=34963)
        prims[spi] = retained
        face_map.extend((spi, int(fid), spi, i) for i, fid in enumerate(kept_ids))
        vertex_map.extend((spi, i, spi, int(sid)) for i, sid in enumerate(used))

        material = copy.deepcopy(original['materials'][primitive['material']])
        require(not material.get('extensions') and not material.get('extras'), 'Unsupported material fields')
        old_name = material['name']; new_name = old_name + '__contact_uv_dark_v1'; material['name'] = new_name
        require(all(m.get('name') != new_name for m in doc['materials']), 'Material collision')
        route_report = []
        texture_slots = [(material['pbrMetallicRoughness'], 'baseColorTexture', 'pbrMetallicRoughness.baseColorTexture'),
            (material['pbrMetallicRoughness'], 'metallicRoughnessTexture', 'pbrMetallicRoughness.metallicRoughnessTexture'),
            (material, 'normalTexture', 'normalTexture'), (material, 'occlusionTexture', 'occlusionTexture'),
            (material, 'emissiveTexture', 'emissiveTexture')]
        for container, key, path in texture_slots:
            if key not in container:
                continue
            route = container[key]
            require(not route.get('extensions') and type(route.get('texCoord', 0)) is int and route.get('texCoord', 0) == 0,
                'Require original untransformed UV0 texture route')
            if key != 'baseColorTexture':
                route['texCoord'] = 1
            route_report.append({'slot': path, 'texture': route['index'], 'original_texcoord': 0,
                'new_texcoord': 0 if key == 'baseColorTexture' else 1})
        material_id = len(doc['materials']); doc['materials'].append(material)
        corners = faces[picked].ravel()
        changed = copy.deepcopy(primitive); changed['material'] = material_id
        changed['attributes'] = {name: append_array(source.array(ai)[corners], original['accessors'][ai])
            for name, ai in attrs.items() if name != 'TEXCOORD_0'}
        changed['attributes']['TEXCOORD_0'] = append_array(np.tile(new_uv, (len(picked), 1)), original['accessors'][attrs['TEXCOORD_0']])
        changed['attributes']['TEXCOORD_1'] = append_array(source.array(attrs['TEXCOORD_0'])[corners], original['accessors'][attrs['TEXCOORD_0']])
        changed['indices'] = append_array(np.arange(len(corners), dtype='<u4').reshape(-1, 1), kind='SCALAR', target=34963)
        npi = len(prims); prims.append(changed)
        face_map.extend((spi, int(fid), npi, i) for i, fid in enumerate(picked))
        vertex_map.extend((npi, i, spi, int(sid)) for i, sid in enumerate(corners))
        material_reports.append({'source_primitive': spi, 'source_material': primitive['material'], 'source_name': old_name,
            'new_primitive': npi, 'new_material': material_id, 'new_name': new_name, 'faces': len(picked),
            'route_inventory': route_report, 'normal_tangents': 'Exact original corner TANGENT values preserved; UV1 is exact original UV0.',
            'ownership': 'Same explicit body part as source primitive; no companion material selected.'})

    require(len(face_map) == source_face_total <= 100000 and len(vertex_map) <= 100000, 'Output topology bounds')
    require(bytes(data[:len(source.binary)]) == source.binary, 'Original BIN prefix changed')
    doc['buffers'][0]['byteLength'] = len(data)
    encoded = json.dumps(doc, separators=(',', ':'), ensure_ascii=False, allow_nan=False).encode('utf-8')
    encoded += b' ' * (-len(encoded) % 4)
    data.extend(b'\x00' * (-len(data) % 4))
    header = struct.pack('<4sII', b'glTF', 2, 28+len(encoded)+len(data))
    target = OUT / 'model.glb'
    require(not target.exists(), 'Output exists')
    with target.open('xb') as stream:
        stream.write(header); stream.write(struct.pack('<II', len(encoded), 0x4E4F534A)); stream.write(encoded)
        stream.write(struct.pack('<II', len(data), 0x004E4942)); stream.write(data)
    require(target.stat().st_size <= 64 * 1024**2, 'Derivative file cap')
    derived = GLB(target)
    require(derived.binary[:len(source.binary)] == source.binary, 'Reparsed BIN custody failed')
    expected = copy.deepcopy(original)
    for k in ('accessors', 'bufferViews', 'materials'):
        require(derived.doc[k][:len(original[k])] == original[k], 'Original JSON prefix changed: '+k)
        expected[k] = derived.doc[k]
    expected['buffers'][0]['byteLength'] = derived.doc['buffers'][0]['byteLength']
    expected['meshes'][0]['primitives'] = copy.deepcopy(original['meshes'][0]['primitives'])
    for spi in (0, 2):
        for key in ('attributes', 'indices'):
            expected['meshes'][0]['primitives'][spi][key] = derived.doc['meshes'][0]['primitives'][spi][key]
    expected['meshes'][0]['primitives'].extend(derived.doc['meshes'][0]['primitives'][4:])
    require(expected == derived.doc, 'Unexpected JSON mutation')
    grouped = defaultdict_rows(face_map)
    changed_uv_faces = 0
    for (spi, npi), mappings in grouped.items():
        sp, np_ = original['meshes'][0]['primitives'][spi], derived.doc['meshes'][0]['primitives'][npi]
        sf = source.array(sp['indices']).reshape(-1, 3)[mappings[:, 0]]
        nf = derived.array(np_['indices']).reshape(-1, 3)[mappings[:, 1]]
        for name in ('POSITION', 'NORMAL', 'TANGENT'):
            require(np.array_equal(source.array(sp['attributes'][name])[sf], derived.array(np_['attributes'][name])[nf]), 'Corner payload changed: '+name)
        if npi < 4:
            require(np.array_equal(source.array(sp['attributes']['TEXCOORD_0'])[sf], derived.array(np_['attributes']['TEXCOORD_0'])[nf]), 'Protected UV0 changed')
        else:
            require(np.array_equal(source.array(sp['attributes']['TEXCOORD_0'])[sf], derived.array(np_['attributes']['TEXCOORD_1'])[nf]), 'Copied UV1 changed')
            require(np.array_equal(derived.array(np_['attributes']['TEXCOORD_0'])[nf], np.tile(new_uv, (len(nf), 1, 1))), 'Replacement UV0 mismatch')
            changed_uv_faces += len(nf)
    require(changed_uv_faces == 249, 'Unexpected UV change coverage')
    np.savez_compressed(OUT/'correspondence.npz', face_map=np.asarray(face_map, dtype='<u4'), vertex_map=np.asarray(vertex_map, dtype='<u4'))
    validation = validate_render_attributes(target)
    write('render-attributes.json', validation)
    face_lookup = {(spi, sf): (npi, nf) for spi, sf, npi, nf in face_map}
    payload = []
    model_sha = sha(target)
    for row in selector['rows']:
        spi, sf = row['source_primitive'], row['source_triangle']
        npi, nf = face_lookup[(spi, sf)]
        primitive = derived.doc['meshes'][0]['primitives'][npi]
        f = derived.array(primitive['indices']).reshape(-1, 3)[nf]
        replacement = derived.array(primitive['attributes']['TEXCOORD_0'])[f]
        old_uv = np.asarray(row['source_uv0'], dtype='<f4')
        payload.append({'source_primitive': spi, 'source_triangle': sf,
            'rest_positions_blender_xyz_m': row['source_triangle_blender_m'],
            'old_uv0_gltf_v_down': old_uv.tolist(), 'uv0_blender_v_up': (old_uv * [1, -1] + [0, 1]).tolist(),
            'new_uv0_gltf_v_down': replacement.tolist(), 'repaired_uv0_blender_v_up': (replacement * [1, -1] + [0, 1]).tolist(),
            'derivative_primitive': npi, 'derivative_triangle': nf,
            'source_material_name': original['materials'][original['meshes'][0]['primitives'][spi]['material']]['name'],
            'derivative_material_name': derived.doc['materials'][primitive['material']]['name'],
            'corner_order': 'Original v7 source face index order; all original corners retained without reversal.'})
    write('native-color-payload-v2.json', {'schema_version': 1, 'source_model_sha256': PREP['inputs']['source']['sha256'],
        'derivative_model_sha256': model_sha, 'image_sha256': summary['texture_sha256'], 'faces': payload,
        'scope': 'Transient existing-scene UV0-only unlit color control, original pose/camera/geometry unchanged; not regenerated fur or normal-map acceptance.'})
    for row in PREP['inputs'].values():
        require(sha(row['path']) == row['sha256'], 'Input modified')
    report = {'schema_version': 1, 'status': 'experimental_source_uv_derivative', 'model_sha256': model_sha,
        'source_sha256': PREP['inputs']['source']['sha256'], 'selector_sha256': PREP['inputs']['selector']['sha256'],
        'source_files_unchanged': True, 'original_bin_prefix_exact': True, 'original_bin_prefix_bytes': len(source.binary),
        'original_materials_images_textures_nodes_skins_animations_exact': True,
        'all_oriented_triangle_positions_normals_tangents_exact': True, 'selected_uv1_equals_original_uv0': True,
        'protected_uv0_exact': True, 'selected_faces': changed_uv_faces, 'triangles': source_face_total,
        'vertices': len(vertex_map), 'bytes': target.stat().st_size, 'texture_decode_calls': 0,
        'source_rig': 'Unrigged v7 authored source; no animation or skin completion claim.',
        'materials': material_reports, 'face_map_columns': ['source_primitive', 'source_triangle', 'output_primitive', 'output_triangle'],
        'vertex_map_columns': ['output_primitive', 'output_vertex', 'source_primitive', 'source_vertex'],
        'geometry_repair_selector_intersection': [], 'geometry_healthy_reference_intersection': [],
        'no_loose_vertices': True, 'original_json_changed_only': ['buffer byteLength', 'primitive0/2 attributes+indices with exact retained corner payload', 'appended accessors/bufferViews/materials/primitives'],
        'limits': PREP['limits'], 'elapsed_seconds': time.monotonic()-START,
        'self_peak_rss_bytes': int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * (1 if sys.platform == 'darwin' else 1024),
        'quality_approved': False,
        'limitations': ['Appearance requires new native preview; no defect-free or Muse-quality claim.',
            'Finisher must explicitly include new material clones in body ownership/groom and distinguish closure provenance.',
            'Current adaptive undercoat requires one original UV layer; these deliberate UV0 overlaps plus preserved normal UV1 need an admitted route or must fail explicitly.',
            'Native/portable fur colors and undercoat must be regenerated; existing finished asset colors are not changed.',
            'Source contact geometry/nonmanifold limits persist; this is color repair only.']}
    write('report.json', report)
    print(json.dumps({k:report[k] for k in ('status','model_sha256','selected_faces','vertices','triangles','bytes','elapsed_seconds','self_peak_rss_bytes')}))


def defaultdict_rows(rows):
    groups = {}
    for spi, sf, npi, nf in rows:
        groups.setdefault((spi, npi), []).append((sf, nf))
    return {key: np.asarray(value, dtype=np.int32) for key, value in groups.items()}


try:
    main()
except Exception as error:
    write('failure.json', {'status': 'failed', 'type': type(error).__name__, 'error': str(error), 'elapsed_seconds': time.monotonic()-START})
    raise
