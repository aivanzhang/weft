"""Independent exported packing audit; no producer code or renderer is executed."""

from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE/'sources'))
from glb_read import GLB, require

CONFIG = json.loads((HERE/'config.json').read_text())
START = time.monotonic()


def save(name, value):
    payload = json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+'\n'
    with (HERE/name).open('x') as stream:
        stream.write(payload)


def atlas(glb, primitive):
    material = glb.doc['materials'][primitive['material']]
    texture = glb.doc['textures'][material['pbrMetallicRoughness']['baseColorTexture']['index']]
    image = glb.doc['images'][texture['source']]
    require(image['mimeType'] == 'image/png', 'Require embedded PNG')
    encoded = glb.view_bytes(image['bufferView'])
    with Image.open(io.BytesIO(encoded)) as picture:
        require(picture.mode == 'RGBA' and picture.size == (4096, 2400), 'Unexpected decoded cell atlas')
        info = {key: str(picture.info.get(key)) for key in ('gamma', 'srgb', 'chromaticity', 'icc_profile')}
        pixels = np.asarray(picture, dtype=np.uint8)
    del picture
    return pixels, hashlib.sha256(encoded).hexdigest(), info


def samples(pixels, coordinates):
    # Independent CLAMP sampler; atlas texels are decoded before linear interpolation.
    xy = coordinates*np.array([pixels.shape[1], pixels.shape[0]])-.5
    lower = np.floor(xy).astype(int)
    delta = xy-lower
    result = np.zeros((len(coordinates), 3), dtype=float)
    for dx, dy in ((0, 0), (1, 0), (0, 1), (1, 1)):
        indices = np.minimum(np.maximum(lower+[dx, dy], 0), np.array([pixels.shape[1], pixels.shape[0]])-1)
        encoded = pixels[indices[:, 1], indices[:, 0], :3].astype(float)/255
        decoded = np.empty_like(encoded)
        small = encoded <= .04045
        decoded[small] = encoded[small]/12.92
        decoded[~small] = ((encoded[~small]+.055)/1.055)**2.4
        weight = (delta[:, 0] if dx else 1-delta[:, 0])*(delta[:, 1] if dy else 1-delta[:, 1])
        result += decoded*weight[:, None]
    return result


def main():
    for row in CONFIG['inputs']:
        with (ROOT/row['path']).open('rb') as stream:
            require(hashlib.file_digest(stream, 'sha256').hexdigest() == row['sha256'], 'Pinned input changed')
    source, target = GLB(ROOT/CONFIG['source']), GLB(ROOT/CONFIG['candidate'])
    a, b = source.doc, target.doc
    old, new = a['meshes'][0]['primitives'][7], b['meshes'][0]['primitives'][7]
    restored = deepcopy(b)
    for key, number in (('accessors', 1), ('bufferViews', 2), ('images', 1), ('textures', 1), ('materials', 1)):
        require(len(b[key]) == len(a[key])+number and b[key][:-number] == a[key], 'Original descriptor changed: '+key)
        restored[key] = restored[key][:-number]
    original_fin = deepcopy(new)
    original_fin['attributes']['TEXCOORD_0'] = old['attributes']['TEXCOORD_0']
    original_fin['material'] = old['material']
    require(original_fin == old, 'Fin geometry, normals, indices or skin binding changed')
    restored['meshes'][0]['primitives'][7] = original_fin
    restored['buffers'][0]['byteLength'] = a['buffers'][0]['byteLength']
    require(restored == a, 'JSON mutation outside append-only packing allowlist')
    src_material, new_material = a['materials'][old['material']], b['materials'][new['material']]
    checked_material = deepcopy(new_material)
    checked_material['name'] = src_material['name']
    checked_material['pbrMetallicRoughness']['baseColorTexture']['index'] = src_material['pbrMetallicRoughness']['baseColorTexture']['index']
    require(checked_material == src_material, 'Packing changed shading properties')
    require(src_material['doubleSided'] is False and src_material['alphaMode'] == 'MASK'
            and src_material['alphaCutoff'] == .18, 'Unexpected combined input shading contract')
    old_tex = a['textures'][src_material['pbrMetallicRoughness']['baseColorTexture']['index']]
    new_tex = b['textures'][new_material['pbrMetallicRoughness']['baseColorTexture']['index']]
    require({**new_tex, 'source': old_tex['source']} == old_tex, 'Sampler changed')
    require(all(v.get('byteOffset', 0) >= source.bin_size for v in b['bufferViews'][len(a['bufferViews']):]),
            'New storage overlaps original BIN prefix')
    prefix = hashlib.sha256()
    for offset in range(0, source.bin_size, 1024*1024):
        end = min(offset+1024*1024, source.bin_size)
        first = source.data[source.offset+offset:source.offset+end]
        second = target.data[target.offset+offset:target.offset+end]
        require(first == second, 'Old BIN byte changed')
        prefix.update(first)
    source_uv = source.array(old['attributes']['TEXCOORD_0'], 'VEC2', (5126,)).reshape(4800, 4, 2)
    target_uv = target.array(new['attributes']['TEXCOORD_0'], 'VEC2', (5126,)).reshape(4800, 4, 2)
    with np.load(ROOT/CONFIG['permutation'], allow_pickle=False) as data:
        destination = data['destination_for_source_fin']
        order = data['source_fin_at_destination']
    require(destination.shape == order.shape == (4800,) and np.array_equal(np.sort(destination), np.arange(4800))
            and np.array_equal(destination[order], np.arange(4800)), 'Invalid cell permutation')
    indices = np.arange(4800)
    old_origin = np.column_stack((indices % 64*64, indices//64*32))
    new_origin = np.column_stack((destination % 64*64, destination//64*32))
    predicted = (source_uv.astype(float)*[4096, 2400]-old_origin[:, None]+new_origin[:, None])/[4096, 2400]
    require(predicted.astype('<f4').tobytes() == target_uv.tobytes(), 'Exported UV is not exact float32 mapped source UV')
    maximum_rounding = float(np.max(np.abs(predicted-target_uv)))
    original_pixels, old_png_sha, old_info = atlas(source, old)
    packed_pixels, packed_png_sha, packed_info = atlas(target, new)
    require(old_info == packed_info, 'PNG color-space metadata changed')
    with (ROOT/CONFIG['atlas']).open('rb') as stream:
        require(hashlib.file_digest(stream, 'sha256').hexdigest() == packed_png_sha, 'Published atlas differs from embedded bytes')
    for source_index, target_index in enumerate(destination):
        sy, sx = source_index//64*32, source_index % 64*64
        ty, tx = target_index//64*32, target_index % 64*64
        require(np.array_equal(original_pixels[sy:sy+32, sx:sx+64], packed_pixels[ty:ty+32, tx:tx+64]),
                'RGBA cell bytes changed at source fin '+str(source_index))
    with np.load(ROOT/CONFIG['tip_reference'], allow_pickle=False) as data:
        truth = data['source_rgb'].reshape(9600, 3)
        fractions = data['tip_fraction']
    def query(uv):
        points = (1-fractions[:, :, None])*uv[:, :1].astype(float)+fractions[:, :, None]*uv[:, 1:2].astype(float)
        points[:, :, 1] = ((uv[:, 0, 1].astype(float)+uv[:, 2, 1].astype(float))*.5)[:, None]
        return points.reshape(-1, 2)
    old_rgb = samples(original_pixels, query(source_uv))
    new_rgb = samples(packed_pixels, query(target_uv))
    error = np.max(np.abs(new_rgb-truth), axis=1)
    quantiles = np.quantile(error, [.5, .95, .99])
    gate = CONFIG['gate']
    require(len(error) == gate['expected_tips'] and quantiles[1] <= gate['p95'] and quantiles[2] <= gate['p99']
            and int((error > .1).sum()) <= gate['tips_over_010_max'], 'Independent nominal-tip color gate failed')
    np.savez_compressed(HERE/'tip-check.npz', source_rgb=truth, original_atlas_rgb=old_rgb,
                        exported_packed_rgb=new_rgb, maximum_component_error=error)
    result = {'schema_version': 1, 'status': 'passed_independent_exported_packing_audit',
        'source_sha256': CONFIG['inputs'][0]['sha256'], 'candidate_sha256': CONFIG['inputs'][1]['sha256'],
        'exact_json_allowlist_passed': True, 'exact_old_BIN_prefix_bytes': source.bin_size,
        'old_BIN_prefix_sha256': prefix.hexdigest(), 'includes_old_BIN_padding': True,
        'all_old_geometry_normals_indices_skin_inverse_binds_actions_exact': True,
        'fin_bindings': {'mesh': 0, 'primitive': 7, 'unchanged_attributes': {k: v for k, v in old['attributes'].items() if k != 'TEXCOORD_0'},
                         'unchanged_indices': old['indices'], 'source_uv': old['attributes']['TEXCOORD_0'],
                         'packed_uv': new['attributes']['TEXCOORD_0']},
        'all_4800_RGBA_cells_byte_identical': True, 'PNG_color_metadata_unchanged': True,
        'original_atlas_RGBA_sha256': hashlib.sha256(original_pixels.tobytes()).hexdigest(),
        'packed_atlas_RGBA_sha256': hashlib.sha256(packed_pixels.tobytes()).hexdigest(),
        'embedded_packed_PNG_matches_published_file': True, 'new_material_equivalent_except_name_and_texture_index': True,
        'preserved_doubleSided': False, 'preserved_MASK_cutoff': .18,
        'exported_UV_bytes_equal_recomputed_float32_mapping': True, 'maximum_float32_UV_rounding': maximum_rounding,
        'color': {'tip_count': len(error), 'p50': float(quantiles[0]), 'p95': float(quantiles[1]), 'p99': float(quantiles[2]),
                  'maximum_linear_component_error': float(error.max()), 'tips_over_010': int((error > .1).sum()),
                  'maximum_color_change_from_original_atlas_due_to_UV_rounding': float(np.max(np.abs(old_rgb-new_rgb))),
                  'fixed_gate': gate}, 'elapsed_seconds': time.monotonic()-START,
        'quality_approved': False, 'native_render_or_motion_rerun': False, 'original_4k_texture_redecoded': False,
        'limits': ['Source tip colors are reused from the pinned prior 4K-source audit, not reacquired or independently decoded here.',
                   'Byte-preserved rig/actions prove unchanged data, not a new full motion/intersection/render evaluation.',
                   'This checks actual exported UV/color at LOD0, not GPU mip behavior, filtered alpha visibility or appearance.']}
    source.data.close();source.handle.close();target.data.close();target.handle.close()
    save('result.json', result)
    print(json.dumps({'status': result['status'], 'color': result['color'], 'elapsed_seconds': result['elapsed_seconds']}, indent=2))


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        save('failure.json', {'status': 'failed_independent_packing_audit', 'type': type(error).__name__,
                             'message': str(error), 'elapsed_seconds': time.monotonic()-START})
        raise
