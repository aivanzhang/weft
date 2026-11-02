"""Independent bounded byte audit; no renderer, repair implementation or provider calls."""

from datetime import datetime
import hashlib
from io import BytesIO
import json
from pathlib import Path
import struct
import time
from zoneinfo import ZoneInfo

import numpy as np
from PIL import Image


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
STARTED = time.monotonic()
PREPARED = json.loads((HERE/'prepared.json').read_text())
POLICY = PREPARED['expected_frame_policy']
LIMIT = PREPARED['limits']


def require(value, message):
    if not value:
        raise ValueError(message)
    if time.monotonic()-STARTED > LIMIT['deadline_seconds']:
        raise TimeoutError('Independent export audit exceeded its 30-second deadline')


def digest(data):
    return hashlib.sha256(data).hexdigest()


def file_record(path):
    require(path.is_file() and not path.is_symlink(), 'Missing or symlinked audit input')
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            value.update(block)
            require(True, 'deadline')
    return {'bytes': path.stat().st_size, 'sha256': value.hexdigest()}


def save(name, value):
    encoded = json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+'\n'
    require(len(encoded.encode()) <= LIMIT['audit_output_bytes'], 'Audit report exceeds output bound')
    with (HERE/name).open('x') as stream:
        stream.write(encoded)


def verify_inputs():
    for row in PREPARED['inputs']:
        require(file_record(ROOT/row['path']) == {key: row[key] for key in ('bytes', 'sha256')},
                'Pinned original changed: '+row['path'])
        if 'copy' in row:
            require(file_record(HERE/row['copy']) == {key: row[key] for key in ('bytes', 'sha256')},
                    'Pinned copy changed: '+row['copy'])


def read_model(path):
    require(28 <= path.stat().st_size <= LIMIT['model_bytes_each'], 'Model exceeds byte bound')
    raw = path.read_bytes()
    magic, version, length = struct.unpack_from('<4sII', raw)
    require(magic == b'glTF' and version == 2 and length == len(raw), 'Invalid GLB header')
    size, kind = struct.unpack_from('<II', raw, 12)
    require(kind == 0x4e4f534a and size % 4 == 0 and size <= LIMIT['json_bytes_each'], 'Invalid JSON chunk')
    encoded = raw[20:20+size]
    binary_size, binary_kind = struct.unpack_from('<II', raw, 20+size)
    binary_start = 28+size
    require(binary_kind == 0x004e4942 and binary_size % 4 == 0 and
            binary_start+binary_size == len(raw), 'Invalid BIN chunk')
    doc = json.loads(encoded)
    require(doc['asset']['version'] == '2.0' and len(doc['buffers']) == 1 and
            'uri' not in doc['buffers'][0], 'Require embedded glTF 2.0')
    require(not doc.get('extensionsUsed') and not doc.get('extensionsRequired'), 'Extensions not audited')
    for view in doc['bufferViews']:
        start, count = view.get('byteOffset', 0), view['byteLength']
        require(view['buffer'] == 0 and type(start) is int and type(count) is int and
                start >= 0 and count > 0 and start+count <= binary_size, 'Invalid bufferView range')
    return {'raw': raw, 'json': encoded, 'doc': doc, 'binary_start': binary_start,
            'binary': memoryview(raw)[binary_start:]}


def accessor(model, index):
    spec = model['doc']['accessors'][index]
    require(not spec.get('sparse'), 'Sparse accessor not audited')
    width = {'SCALAR': 1, 'VEC2': 2, 'VEC3': 3, 'VEC4': 4, 'MAT4': 16}[spec['type']]
    dtype = np.dtype({5121: 'u1', 5123: '<u2', 5125: '<u4', 5126: '<f4'}[spec['componentType']])
    count = spec['count']
    require(type(count) is int and 0 < count <= 300000, 'Accessor count exceeds bound')
    view = model['doc']['bufferViews'][spec['bufferView']]
    offset, stride = spec.get('byteOffset', 0), view.get('byteStride', width*dtype.itemsize)
    require(type(offset) is int and type(stride) is int and offset >= 0 and stride >= width*dtype.itemsize and
            offset+(count-1)*stride+width*dtype.itemsize <= view['byteLength'], 'Invalid accessor range')
    return np.ndarray((count, width), dtype=dtype, buffer=model['binary'],
                      offset=view.get('byteOffset', 0)+offset, strides=(stride, dtype.itemsize))


def pixels(encoded):
    with Image.open(BytesIO(encoded)) as image:
        require(image.format == 'PNG' and image.width*image.height <= 2048**2 and
                getattr(image, 'n_frames', 1) == 1, 'Normal texture exceeds bounded PNG profile')
        dimensions = list(image.size)
        result = image.convert('RGBA').tobytes()
    return {'dimensions': dimensions, 'rgba_sha256': digest(result)}


def main():
    verify_inputs()
    pack = ROOT/PREPARED['source_pack']
    manifest = json.loads((HERE/'receipts/manifest.json').read_text())
    config = json.loads((HERE/'receipts/finish-config.json').read_text())
    report = json.loads((HERE/'receipts/finish-report.json').read_text())
    receipt = json.loads((HERE/'receipts/tangent-repair.json').read_text())
    raw, corrected = [read_model(pack/name) for name in ('model-native-export.glb', 'model.glb')]
    require(digest(corrected['raw']) == PREPARED['expected_model_sha256'] == receipt['result_sha256'] and
            digest(raw['raw']) == receipt['source_sha256'], 'Model/repair receipt hash mismatch')
    require(raw['json'] == corrected['json'] and len(raw['raw']) == len(corrected['raw']) and
            raw['binary_start'] == corrected['binary_start'], 'Correction changed GLB JSON/layout/length')
    require(receipt['repaired_vertices'] == [], 'Unexpected legacy tangent repairs require a separate audit')
    undercoat, correction = report['fur_undercoat'], receipt['normal_uv1_correction']
    require(config['fur_undercoat']['uv_policy'] == 'material-adaptive-v1' and
            undercoat['frame_policy'] == undercoat['plan']['frame_policy'] == correction['policy'] == POLICY,
            'Baked/exported frame-policy handshake mismatch')
    for filename in ('undercoat_uv_atlas.py', 'generated_tangents.py'):
        require("NORMAL_UV1_FRAME_POLICY = '"+POLICY+"'" in (HERE/'sources'/filename).read_text(),
                'Pinned frame-policy source mismatch')
    source = Path(config['source'])
    require(file_record(source)['sha256'] == config['source_sha256'] == manifest['source_sha256'] ==
            report['source_sha256'], 'Authored input source binding mismatch')
    require(manifest['status'] == 'needs_visual_review', 'Audit must not promote the model')
    artifacts = {row['path']: row for row in manifest['artifacts']}
    for row in PREPARED['inputs']:
        original = ROOT/row['path']
        if original.is_relative_to(pack) and original.name != 'manifest.json':
            saved = artifacts[original.relative_to(pack).as_posix()]
            require((row['sha256'], row['bytes']) == (saved['sha256'], saved['bytes']),
                    'Published artifact binding mismatch')

    doc = raw['doc']
    changed = np.flatnonzero(np.frombuffer(raw['raw'], dtype=np.uint8) !=
                             np.frombuffer(corrected['raw'], dtype=np.uint8))
    permitted = np.zeros(len(changed), dtype=bool)
    primitive_rows, selected_ids, ranges = [], set(), []
    expected = {(row['mesh'], row['primitive']): row for row in correction['primitives']}
    source_materials = {row['material']: row for row in undercoat['materials']}
    vertices, triangles = 0, 0
    for mi, mesh in enumerate(doc['meshes']):
        for pi, primitive in enumerate(mesh['primitives']):
            attrs, material = primitive['attributes'], doc['materials'][primitive['material']]
            require(primitive.get('mode', 4) == 4, 'Nontriangular mesh not audited')
            count = doc['accessors'][attrs['POSITION']]['count']
            indices = accessor(raw, primitive['indices']).ravel()
            require(len(indices) % 3 == 0 and indices.max() < count, 'Invalid triangle indices')
            vertices += count
            triangles += len(indices)//3
            row = {'mesh': mi, 'primitive': pi, 'mesh_name': mesh.get('name'), 'material': material['name'],
                   'vertices': count, 'triangles': len(indices)//3,
                   'normal_texcoord': material.get('normalTexture', {}).get('texCoord')}
            if material.get('normalTexture', {}).get('texCoord') == 1:
                pinned = expected.pop((mi, pi))
                baked = source_materials[material['name']]
                require(baked['frame_policy'] == baked['atlas']['frame_policy'] == POLICY and
                        baked['selected_faces'] == len(indices)//3 == pinned['faces'] and
                        pinned['vertices'] == count and pinned['tangent_accessor'] == attrs['TANGENT'],
                        'UV1 primitive/bake/correction binding mismatch')
                require(len(indices) == count and np.all(np.bincount(indices, minlength=count) == 1),
                        'UV1 vertices are not unique triangle corners')
                require(count <= 2048*3 and len(selected_ids) < 4, 'UV1 correction exceeds region bounds')
                index = attrs['TANGENT']
                selected_ids.add(index)
                spec = doc['accessors'][index]
                view = doc['bufferViews'][spec['bufferView']]
                require(spec['componentType'] == 5126 and spec['type'] == 'VEC4' and
                        spec.get('byteOffset', 0) == 0 and view.get('byteStride', 16) == 16 and
                        view['byteLength'] == count*16, 'UV1 tangent storage is not exclusive contiguous float4')
                start = raw['binary_start']+view.get('byteOffset', 0)
                end = start+view['byteLength']
                permitted |= (changed >= start)&(changed < end)
                before, after = accessor(raw, index), accessor(corrected, index)
                require(digest(before.tobytes()) == pinned['source_tangents_sha256'] and
                        digest(after.tobytes()) == pinned['authored_tangents_sha256'], 'Tangent array receipt mismatch')
                norms = accessor(corrected, attrs['NORMAL']).astype(np.float64)
                require(np.isfinite(after).all() and np.all(np.abs(np.linalg.norm(after[:, :3], axis=1)-1) < .001)
                        and np.all(np.abs(np.sum(norms*after[:, :3], axis=1)) < .002), 'Invalid corrected frames')
                signs = after[indices.reshape(-1, 3), 3]
                require(np.all(np.abs(signs) == 1) and np.all(signs == signs[:, :1]), 'Mixed or invalid triangle W')
                ranges.append({'accessor': index, 'absolute_start': start, 'absolute_end': end,
                               'permitted_bytes': end-start, 'changed_bytes': int(((changed >= start)&(changed < end)).sum()),
                               'raw_sha256': pinned['source_tangents_sha256'],
                               'corrected_sha256': pinned['authored_tangents_sha256']})
                row.update(correction_scope='tangent_float4_only', mixed_w_faces=0)
            primitive_rows.append(row)
    require(not expected and len(selected_ids) > 0 and permitted.all(), 'Changed bytes escape declared UV1 tangent storage')
    require(vertices <= undercoat['plan']['projected_export_vertex_ceiling'] <= 100000 and triangles <= 100000,
            'Measured model exceeds projected/export limits')
    budget = config['undercoat_export_budget']
    require(budget == undercoat['plan']['export_vertex_budget'] and
            sum(budget.values())+undercoat['plan']['additional_export_vertices'] ==
            undercoat['plan']['projected_export_vertex_ceiling'], 'Projected budget accounting mismatch')

    scalar_count, accessor_rows = 0, []
    for index, spec in enumerate(doc['accessors']):
        before, after = accessor(raw, index), accessor(corrected, index)
        scalar_count += before.size
        require(scalar_count <= 2000000, 'Aggregate accessor scalar bound exceeded')
        require(np.isfinite(before).all() and np.isfinite(after).all(), 'Nonfinite stored attribute')
        old_hash, new_hash = digest(before.tobytes()), digest(after.tobytes())
        require(index in selected_ids or old_hash == new_hash, 'A nonselected accessor changed')
        accessor_rows.append({'accessor': index, 'type': spec['type'], 'count': spec['count'],
                              'raw_sha256': old_hash, 'corrected_sha256': new_hash,
                              'selected_tangent': index in selected_ids})
    image_rows = []
    for index, image in enumerate(doc['images']):
        require('uri' not in image, 'External image not audited')
        view = doc['bufferViews'][image['bufferView']]
        start, size = view.get('byteOffset', 0), view['byteLength']
        old, new = raw['binary'][start:start+size], corrected['binary'][start:start+size]
        require(old == new, 'Embedded texture bytes changed')
        image_rows.append({'image': index, 'mime_type': image['mimeType'], 'bytes': size, 'sha256': digest(old)})
    baked_rows = []
    for baked in undercoat['materials']:
        material = next(value for value in doc['materials'] if value['name'] == baked['material'])
        info = material['normalTexture']
        require(info.get('texCoord', 0) == baked['normal_texcoord'], 'Baked material UV binding mismatch')
        image_index = doc['textures'][info['index']]['source']
        image = doc['images'][image_index]
        view = doc['bufferViews'][image['bufferView']]
        start = view.get('byteOffset', 0)
        saved = (pack/baked['texture_path']).read_bytes()
        require(digest(saved) == baked['texture_sha256'], 'Saved normal PNG hash mismatch')
        saved_pixels = pixels(saved)
        require(saved_pixels == pixels(bytes(corrected['binary'][start:start+view['byteLength']])),
                'Exported normal pixels differ from the pinned new bake')
        baked_rows.append({'material': baked['material'], 'normal_texcoord': baked['normal_texcoord'],
                           'embedded_image': image_index, 'saved_png_sha256': baked['texture_sha256'], **saved_pixels})
    verify_inputs()
    save('accessors.json', {'decoded_scalar_values': scalar_count, 'accessors': accessor_rows})
    save('byte-preservation.json', {'model_bytes_each': len(raw['raw']), 'changed_bytes': len(changed),
         'unchanged_bytes': len(raw['raw'])-len(changed), 'json_bytes_exact': True,
         'all_nonselected_accessor_data_exact': True, 'all_embedded_texture_bytes_exact': True,
         'tangent_storage_ranges': ranges, 'images': image_rows, 'new_bake_bindings': baked_rows})
    result = {'schema_version': 1, 'status': 'passed_structure_and_byte_custody', 'frame_policy': POLICY,
        'source_sha256': config['source_sha256'], 'raw_model_sha256': receipt['source_sha256'],
        'model_sha256': receipt['result_sha256'], 'changed_bytes': len(changed),
        'corrected_uv1_vertices': correction['vertices'], 'vertices': vertices, 'triangles': triangles,
        'projected_vertex_ceiling': undercoat['plan']['projected_export_vertex_ceiling'],
        'export_budget': budget, 'primitives': primitive_rows, 'nodes': len(doc['nodes']),
        'skin_joint_counts': [len(skin['joints']) for skin in doc['skins']],
        'actions': [{'name': action.get('name'), 'channels': len(action['channels']),
                    'samplers': len(action['samplers'])} for action in doc.get('animations', [])],
        'source_json_geometry_uv_normals_weights_actions_materials_and_texture_bytes_preserved': True,
        'audit_inputs_unchanged_before_after': True, 'elapsed_seconds': time.monotonic()-STARTED,
        'sampled_peak_rss_bytes': None, 'new_native_provider_or_ai_calls': 0,
        'recorded_at': datetime.now(ZoneInfo('America/Los_Angeles')).isoformat(),
        'limitations': ['All bytes/accessors were compared, but no renderer or animation playback ran in this audit.',
            'This proves correction custody, not fidelity from the original generated model to the authored finished pack.',
            'Runtime tangent direction was checked by the separate analytic native fixture, not independently re-derived here.',
            'No visual quality, dense-fur equivalence, Weft compatibility or physical validity acceptance.'],
        'quality_approved': False, 'original_status_preserved': manifest['status']}
    save('result.json', result)
    files = [path for path in sorted(HERE.rglob('*')) if path.is_file()]
    records = [{'path': path.relative_to(HERE).as_posix(), **file_record(path)} for path in files]
    require(sum(row['bytes'] for row in records) <= LIMIT['audit_output_bytes'], 'Bundle exceeds 2MiB bound')
    save('manifest.json', {'schema_version': 1, 'kind': 'independent_cpu_export_audit',
                         'status': result['status'], 'artifacts': records,
                         'originals_preserved': True, 'quality_approved': False})
    print(json.dumps({key: result[key] for key in ('status', 'vertices', 'triangles', 'changed_bytes',
                                                  'projected_vertex_ceiling', 'elapsed_seconds')}, indent=2))


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        with (HERE/'audit-failure.json').open('x') as stream:
            json.dump({'status': 'failed_preserved', 'error': type(error).__name__+': '+str(error),
                       'quality_approved': False}, stream, indent=2)
        raise
