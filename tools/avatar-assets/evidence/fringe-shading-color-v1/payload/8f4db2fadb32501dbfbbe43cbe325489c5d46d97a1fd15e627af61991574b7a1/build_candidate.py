"""One append-only color-only derivative; preserve all geometry and skin bytes."""

from copy import deepcopy
import gc
import hashlib
import io
import json
from pathlib import Path
import shutil
import struct
import sys
import time

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE/'sources'))
from glb_read import GLB
from color_atlas import build_color_atlas, sample_alpha
from root_sampler import repeat_root_colors
from avatar_assets.provider_model import preflight
from avatar_assets.three_d import validate_glb, validate_render_attributes
from avatar_assets.safety import require, sha256

STARTED = time.monotonic()
CONFIG = json.loads((HERE/'candidate.json').read_text())
OUT = ROOT/CONFIG['output']


def save(path, value):
    payload = json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+'\n'
    with path.open('x') as stream:
        stream.write(payload)


def stats(error):
    q = np.quantile(error, [.5, .95, .99])
    return {'tip_count': int(error.size), 'p50': float(q[0]), 'p95': float(q[1]), 'p99': float(q[2]),
            'maximum_linear_component_error': float(error.max()),
            'tips_over_005': int((error > .05).sum()), 'tips_over_010': int((error > .1).sum())}


def hash_arrays(arrays):
    return {name: hashlib.sha256(value.tobytes()).hexdigest() for name, value in arrays.items()}


def close(glb):
    glb.data.close()
    glb.handle.close()


def main():
    for record in CONFIG['inputs']:
        require(sha256(ROOT/record['path']) == record['sha256'], 'Pinned source changed: '+record['path'])
    source_dir = (ROOT/CONFIG['source']).parent
    original_report = json.loads((source_dir/'build-report.json').read_text())
    contract = json.loads((source_dir/'rig-contract.json').read_text())
    glb = GLB(ROOT/CONFIG['source'])
    original = deepcopy(glb.doc)
    require(original['nodes'][19] == {'mesh': 0, 'name': 'PartitionedBodyAndPart', 'skin': 0},
            'Unexpected source mesh/skin/transform binding')
    require(len(original['meshes']) == 1 and len(original['meshes'][0]['primitives']) == 8,
            'Unexpected source primitive layout')
    source_primitive = original['meshes'][0]['primitives'][5]
    fin_primitive = original['meshes'][0]['primitives'][7]
    source_attrs, attrs = source_primitive['attributes'], fin_primitive['attributes']
    source_points = glb.array(source_attrs['POSITION'], 'VEC3', (5126,)).astype(float)
    source_uv = glb.array(source_attrs['TEXCOORD_0'], 'VEC2', (5126,))
    source_faces = glb.array(source_primitive['indices'], 'SCALAR', (5121, 5123, 5125), 300000).reshape(-1, 3)
    with np.load(source_dir/'source-correspondence.npz', allow_pickle=False) as data:
        correspondence = {name: data[name] for name in data.files}
    ids, face_ids, corners = (correspondence[k] for k in ('root_vertex_ids', 'root_face_ids', 'root_corner_ids'))
    require(ids.shape == (4800, 2) and face_ids.shape == (4800,) and corners.shape == (4800, 2)
            and ids.min() >= 0 and ids.max() < len(source_points)
            and face_ids.min() >= 0 and face_ids.max() < len(source_faces)
            and corners.min() >= 0 and corners.max() < 3, 'Invalid pinned source correspondence')
    require(np.array_equal(source_faces[face_ids[:, None], corners], ids)
            and np.array_equal(source_uv[ids], correspondence['root_source_uv']), 'Source UV/face correspondence changed')
    arrays = {
        'positions': glb.array(attrs['POSITION'], 'VEC3', (5126,)),
        'normals': glb.array(attrs['NORMAL'], 'VEC3', (5126,)),
        'uv': glb.array(attrs['TEXCOORD_0'], 'VEC2', (5126,)),
        'colors': glb.array(attrs['COLOR_0'], 'VEC3', (5126,)),
        'joints': glb.array(attrs['JOINTS_0'], 'VEC4', (5121, 5123)),
        'weights': glb.array(attrs['WEIGHTS_0'], 'VEC4', (5126,)),
        'indices': glb.array(fin_primitive['indices'], 'SCALAR', (5121, 5123, 5125), 300000)}
    require(np.array_equal(arrays['positions'].reshape(4800, 4, 3)[:, :2], source_points[ids]),
            'Existing fin endpoint positions do not match their source anchors')
    original_array_hashes = hash_arrays(arrays)
    fins = {**arrays, 'root_source_uv': correspondence['root_source_uv'], 'report': original_report['fin_report']}
    material = original['materials'][source_primitive['material']]
    pbr = material['pbrMetallicRoughness']
    binding = pbr['baseColorTexture']
    require(material.get('alphaMode', 'OPAQUE') == 'OPAQUE' and binding.get('texCoord', 0) == 0
            and 'COLOR_0' not in source_attrs, 'Require original opaque UV0 basecolor without extra vertex tint')
    texture = original['textures'][binding['index']]
    sampler = original['samplers'][texture['sampler']] if 'sampler' in texture else {}
    require(sampler.get('wrapS', 10497) == sampler.get('wrapT', 10497) == 10497
            and sampler.get('magFilter', 9729) == 9729 and sampler.get('minFilter', 9987) in (9729, 9985, 9987),
            'Unsupported source sampler')
    encoded_source = glb.view_bytes(original['images'][texture['source']]['bufferView'])
    require(hashlib.sha256(encoded_source).hexdigest() == original_report['source_selection']['exact_root_texture_sha256'],
            'Original root texture changed')
    with Image.open(io.BytesIO(encoded_source)) as image:
        require(image.mode == 'RGB' and image.width <= 4096 and image.height <= 4096
                and image.width*image.height <= 16_777_216 and getattr(image, 'n_frames', 1) == 1,
                'Source texture exceeds bounded opaque RGB decode')
        dimensions = [image.width, image.height]
        pixels = np.asarray(image, dtype=np.uint8)
    del encoded_source
    factor = np.asarray(pbr.get('baseColorFactor', [1, 1, 1, 1])[:3], dtype=float)
    with Image.open(source_dir/'two-tip-atlas.png') as image:
        require(image.size == (1024, 1024) and image.mode == 'RGBA', 'Unexpected source silhouette')
        silhouette = np.array(image.getchannel('A'), dtype=np.uint8)
    source_endpoints = source_points[ids]
    spans = np.linalg.norm(source_endpoints[:, 1]-source_endpoints[:, 0], axis=1)
    tip_centers = np.asarray(original_report['fin_report']['atlas']['tip_centers_m'])
    require(np.array_equal(tip_centers, [-.0025, .0025]), 'Unexpected physical tip centers')
    fractions = .5+tip_centers/spans[:, None]
    tip_source_uv = ((1-fractions[:, :, None])*correspondence['root_source_uv'][:, None, 0]
                     +fractions[:, :, None]*correspondence['root_source_uv'][:, None, 1])
    truth = repeat_root_colors(pixels, tip_source_uv.reshape(-1, 2), factor).reshape(4800, 2, 3)
    endpoint_colors = repeat_root_colors(pixels, correspondence['root_source_uv'].reshape(-1, 2), factor).reshape(4800, 2, 3)
    require(np.max(np.abs(endpoint_colors-arrays['colors'].reshape(4800, 4, 3)[:, :2])) < 6e-8,
            'Stored old fin colors do not match the pinned source endpoints')
    old_rgb = (1-fractions[:, :, None])*endpoint_colors[:, :1]+fractions[:, :, None]*endpoint_colors[:, 1:]
    old_error = np.max(np.abs(truth-old_rgb), axis=2)
    baseline = stats(old_error)
    require(baseline['tips_over_010'] == CONFIG['gate']['expected_old_tips_over_010'], 'Old 194-tip baseline did not reproduce')
    output = build_color_atlas(fins, np.full(4800, source_primitive['material'], dtype=np.int32),
                              {source_primitive['material']: {'pixels': pixels, 'factor': factor}},
                              silhouette, **CONFIG['atlas_settings'])
    require(hash_arrays(arrays) == original_array_hashes, 'Color baker mutated original fin arrays')
    new_uv = output['uv'].reshape(4800, 4, 2)
    tip_atlas_uv = (1-fractions[:, :, None])*new_uv[:, :1]+fractions[:, :, None]*new_uv[:, 1:2]
    tip_atlas_uv[:, :, 1] = ((new_uv[:, 0, 1]+new_uv[:, 2, 1])*.5)[:, None]
    new_rgb = repeat_root_colors(output['pixels'], tip_atlas_uv.reshape(-1, 2), np.ones(3)).reshape(4800, 2, 3)
    new_error = np.max(np.abs(truth-new_rgb), axis=2)
    alpha = sample_alpha(output['pixels'][:, :, 3], tip_atlas_uv.reshape(-1, 2)).reshape(4800, 2)
    new_stats = stats(new_error)
    gate = CONFIG['gate']
    passed = (new_stats['tip_count'] == gate['tips'] and new_stats['p95'] <= gate['p95_max_linear_component_error']
              and new_stats['p99'] <= gate['p99_max_linear_component_error']
              and new_stats['tips_over_010'] <= gate['tips_over_010_max'])
    ordered = np.argsort(new_error.reshape(-1))[-12:][::-1]
    report = {'status': 'passed_technical_color_gate' if passed else 'rejected_technical_color_gate',
        'method': 'Maximum absolute linear RGB component at all 9600 nominal ±2.5mm tips; source and atlas LOD0 samples',
        'fixed_gate': gate, 'baseline': baseline, 'new': new_stats,
        'minimum_alpha_at_nominal_tips': float(alpha.min()), 'tips_below_mask_cutoff': int((alpha < .18).sum()),
        'worst_tips': [{'fin': int(i//2), 'tip': int(i%2), 'error': float(new_error.reshape(-1)[i]),
                       'source_rgb': truth.reshape(-1, 3)[i].tolist(), 'atlas_rgb': new_rgb.reshape(-1, 3)[i].tolist()}
                       for i in ordered],
        'limits': ['LOD0 CPU samples do not establish GPU/mip, lighting, whole-surface texture or appearance equivalence.',
                   'Unique-cell alpha resampling is not byte-identical to the previous shared silhouette.',
                   'Source markings and factors are retained; no whitening/suppression or normal changes.']}
    save(OUT/'color-error-audit.json', report)
    np.savez_compressed(OUT/'nominal-tip-errors.npz', source_rgb=truth, old_rgb=old_rgb, new_rgb=new_rgb,
                        old_error=old_error, new_error=new_error, nominal_alpha=alpha,
                        tip_fraction=fractions, source_uv=tip_source_uv, atlas_uv=tip_atlas_uv)
    Image.fromarray(output['pixels']).save(OUT/'source-color-atlas.png')
    uv_bytes = output['uv'].astype('<f4', copy=False).tobytes()
    atlas_report = output['report']
    del pixels, silhouette, image, output, new_uv, tip_atlas_uv, truth, old_rgb, new_rgb, arrays, fins
    del source_points, source_uv, source_faces, source_endpoints, correspondence
    gc.collect()

    doc = deepcopy(original)
    start = glb.bin_size
    uv_offset = start
    image_offset = uv_offset+len(uv_bytes)
    require(image_offset % 4 == 0, 'UV append alignment failed')
    png = (OUT/'source-color-atlas.png').read_bytes()
    declared = image_offset+len(png)
    doc['bufferViews'].append({'buffer': 0, 'byteOffset': uv_offset, 'byteLength': len(uv_bytes), 'target': 34962})
    doc['accessors'].append({'bufferView': len(doc['bufferViews'])-1, 'componentType': 5126, 'count': 19200, 'type': 'VEC2'})
    new_uv_id = len(doc['accessors'])-1
    doc['bufferViews'].append({'buffer': 0, 'byteOffset': image_offset, 'byteLength': len(png)})
    doc['images'].append({'bufferView': len(doc['bufferViews'])-1, 'mimeType': 'image/png', 'name': 'FinSourceColorAtlas'})
    old_material = deepcopy(original['materials'][fin_primitive['material']])
    old_texture = original['textures'][old_material['pbrMetallicRoughness']['baseColorTexture']['index']]
    require(old_material.get('alphaMode') == 'MASK' and old_material.get('alphaCutoff') == .18
            and old_material.get('doubleSided') is True and 'normalTexture' not in old_material,
            'Unexpected original fin shading contract')
    doc['textures'].append({**deepcopy(old_texture), 'source': len(doc['images'])-1})
    new_material = deepcopy(old_material)
    new_material['name'] = 'ExperimentalEdgeFringeSourceColorAtlas'
    new_material['pbrMetallicRoughness']['baseColorTexture']['index'] = len(doc['textures'])-1
    require(new_material['pbrMetallicRoughness'].get('baseColorFactor') == [1, 1, 1, 1], 'Unexpected old material factor')
    doc['materials'].append(new_material)
    new_primitive = doc['meshes'][0]['primitives'][7]
    new_primitive['attributes']['TEXCOORD_0'] = new_uv_id
    new_primitive['attributes'].pop('COLOR_0')
    new_primitive['material'] = len(doc['materials'])-1
    doc['buffers'][0]['byteLength'] = declared
    encoded = json.dumps(doc, separators=(',', ':'), allow_nan=False).encode()
    encoded += b' '*((-len(encoded)) % 4)
    binary_length = declared+(-declared) % 4
    total = 28+len(encoded)+binary_length
    require(total <= 64*1024**2 and len(encoded) <= 2*1024**2, 'Derived model exceeds GLB bounds')
    current_output = sum(p.stat().st_size for p in OUT.iterdir() if p.is_file())
    require(current_output+total < 63*1024**2, 'Combined model/atlas output exceeds reserved budget')
    with (OUT/'model.glb').open('xb') as stream:
        stream.write(struct.pack('<4sII', b'glTF', 2, total))
        stream.write(struct.pack('<II', len(encoded), 0x4E4F534A));stream.write(encoded)
        stream.write(struct.pack('<II', binary_length, 0x004E4942))
        for offset in range(0, glb.bin_size, 1024*1024):
            stream.write(glb.data[glb.offset+offset:glb.offset+min(offset+1024*1024, glb.bin_size)])
        stream.write(uv_bytes);stream.write(png);stream.write(b'\0'*((-declared) % 4))
    del png, uv_bytes, encoded
    derived = GLB(OUT/'model.glb')
    expected = deepcopy(derived.doc)
    for key, additions in (('accessors', 1), ('bufferViews', 2), ('images', 1), ('textures', 1), ('materials', 1)):
        require(len(expected[key]) == len(original[key])+additions and expected[key][:-additions] == original[key],
                'Original descriptor mutation: '+key)
        expected[key] = expected[key][:-additions]
    changed_primitive = expected['meshes'][0]['primitives'][7]
    restored = deepcopy(changed_primitive)
    restored['attributes']['TEXCOORD_0'] = attrs['TEXCOORD_0']
    restored['attributes']['COLOR_0'] = attrs['COLOR_0']
    restored['material'] = fin_primitive['material']
    require(restored == fin_primitive, 'Unapproved fin primitive/geometry change')
    expected['meshes'][0]['primitives'][7] = restored
    expected['buffers'][0]['byteLength'] = original['buffers'][0]['byteLength']
    require(expected == original, 'JSON changes exceeded the exact allowlist')
    for offset in range(0, glb.bin_size, 1024*1024):
        end = min(offset+1024*1024, glb.bin_size)
        require(glb.data[glb.offset+offset:glb.offset+end] == derived.data[derived.offset+offset:derived.offset+end],
                'Original BIN prefix changed')
    preservation = {'json_allowlist_passed': True, 'original_bin_prefix_bytes': glb.bin_size,
        'original_bin_prefix_includes_original_chunk_padding': True,
        'original_base_rig_actions_and_fin_geometry_exact': True,
        'unchanged_fin_attribute_ids': {k: attrs[k] for k in ('POSITION', 'NORMAL', 'JOINTS_0', 'WEIGHTS_0')},
        'unchanged_fin_indices_id': fin_primitive['indices'], 'source_fin_array_hashes': original_array_hashes,
        'node': 19, 'mesh': 0, 'skin': 0, 'fin_primitive': 7,
        'old_COLOR_0_accessor_retained_but_unbound': attrs['COLOR_0'], 'new_uv_accessor': new_uv_id,
        'old_material_retained': fin_primitive['material'], 'new_material': new_primitive['material'],
        'shading_change': 'none; same normals, doubleSided, MASK cutoff, roughness, metalness and sampler',
        'original_material': old_material, 'new_material_definition': new_material}
    close(derived);close(glb)
    del derived, glb, expected, original, doc
    gc.collect()
    shutil.copyfile(source_dir/'rig-contract.json', OUT/'rig-contract.json')
    shutil.copyfile(source_dir/'source-correspondence.npz', OUT/'source-correspondence.npz')
    save(OUT/'preflight.json', preflight(OUT/'model.glb'))
    gc.collect()
    save(OUT/'render-attributes.json', validate_render_attributes(OUT/'model.glb'))
    gc.collect()
    rig = validate_glb(OUT/'model.glb', contract)
    require(rig['motion']['animation_verified'], 'Independent skin validator did not verify the unchanged motion')
    save(OUT/'rig-verification.json', rig)
    save(OUT/'preservation-audit.json', preservation)
    status = 'technical_color_gate_passed_pending_native_review' if passed else 'rejected_technical_color_gate'
    save(OUT/'build-report.json', {'status': status, 'quality_approved': False,
        'source_sha256': next(x['sha256'] for x in CONFIG['inputs'] if x['path'] == CONFIG['source']),
        'model_sha256': sha256(OUT/'model.glb'), 'model_bytes': total, 'source_texture_dimensions': dimensions,
        'atlas': atlas_report, 'color_gate_passed': passed, 'preservation': preservation,
        'elapsed_seconds': time.monotonic()-STARTED, 'new_native_provider_ai_role_calls': 0,
        'limitations': ['CPU structure/color result, not actual GPU or native visual acceptance.',
                       'Source fin geometric normals are unchanged; their lighting issue is a separate control.',
                       'Unique-cell alpha resampling and deep mip behavior require separate visual evaluation.']})
    print(json.dumps({'status': status, 'model_sha256': sha256(OUT/'model.glb'), 'color_error': new_stats,
                      'elapsed_seconds': time.monotonic()-STARTED}, indent=2))
    if not passed:
        raise SystemExit(2)


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        save(OUT/'failure.json', {'status': 'failed_color_only_candidate', 'type': type(error).__name__,
                                 'message': str(error), 'elapsed_seconds': time.monotonic()-STARTED})
        raise
