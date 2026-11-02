"""One modeled atlas-filtering screen; no GPU, visibility or avatar rebuild claim."""

import gc
import hashlib
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
STARTED = time.monotonic()
ENCODED = np.arange(256, dtype=np.float32)/255
LUT = np.where(ENCODED <= .04045, ENCODED/12.92, ((ENCODED+.055)/1.055)**2.4)
LUMA = np.array([.2126, .7152, .0722], dtype=np.float32)


def scalar(value):
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError('Unsupported report value: '+type(value).__name__)


def save(name, value):
    payload = json.dumps(value, indent=2, sort_keys=True, allow_nan=False, default=scalar)+'\n'
    with (HERE/name).open('x') as stream:
        stream.write(payload)


def linear_rgba(pixels):
    result = np.empty(pixels.shape, dtype=np.float32)
    result[..., :3] = LUT[pixels[..., :3]]
    result[..., 3] = pixels[..., 3].astype(np.float32)/255
    return result


def byte_rgba(linear):
    result = np.empty(linear.shape, dtype=np.uint8)
    rgb = linear[..., :3]
    result[..., :3] = np.rint(255*np.where(rgb <= .0031308, rgb*12.92,
                                         1.055*np.maximum(rgb, 0)**(1/2.4)-.055)).clip(0, 255).astype(np.uint8)
    result[..., 3] = np.rint(linear[..., 3]*255).clip(0, 255).astype(np.uint8)
    return result


def next_mip(pixels):
    height, width, channels = pixels.shape
    require(channels == 4 and height % 2 == 0 and width % 2 == 0, 'Mip input must be even RGBA')
    result = np.empty((height//2, width//2, 4), dtype=np.uint8)
    for y in range(0, height, 128):
        end = min(y+128, height)
        values = linear_rgba(pixels[y:end])
        averaged = (values[0::2, 0::2]+values[1::2, 0::2]+values[0::2, 1::2]+values[1::2, 1::2])*.25
        result[y//2:end//2] = byte_rgba(averaged)
    return result


def sample(pixels, uv, rectangles=None):
    height, width = pixels.shape[:2]
    result = np.empty((len(uv), 4), dtype=np.float32)
    for start in range(0, len(uv), 8192):
        end = min(start+8192, len(uv))
        xy = uv[start:end]*np.array([width, height])-.5
        low = np.floor(xy).astype(np.int64)
        frac = xy-low
        high = low+1
        if rectangles is None:
            low = np.clip(low, [0, 0], [width-1, height-1])
            high = np.clip(high, [0, 0], [width-1, height-1])
        else:
            box = rectangles[start:end]
            low = np.minimum(np.maximum(low, box[:, :2]), box[:, 2:])
            high = np.minimum(np.maximum(high, box[:, :2]), box[:, 2:])
        u, v = frac.T
        result[start:end] = (linear_rgba(pixels[low[:, 1], low[:, 0]])*((1-u)*(1-v))[:, None]
                            +linear_rgba(pixels[low[:, 1], high[:, 0]])*(u*(1-v))[:, None]
                            +linear_rgba(pixels[high[:, 1], low[:, 0]])*((1-u)*v)[:, None]
                            +linear_rgba(pixels[high[:, 1], high[:, 0]])*(u*v)[:, None])
    return result


def quantiles(values):
    if not len(values):
        return None
    q = np.quantile(values, [.5, .95, .99])
    return {'p50': float(q[0]), 'p95': float(q[1]), 'p99': float(q[2]), 'max': float(np.max(values))}


def level_stats(global_rgba, isolated_rgba, truth):
    rgb, alpha = global_rgba[:, :3], global_rgba[:, 3]
    isolated_rgb, isolated_alpha = isolated_rgba[:, :3], isolated_rgba[:, 3]
    source_y, actual_y, isolated_y = truth@LUMA, rgb@LUMA, isolated_rgb@LUMA
    error = np.max(np.abs(rgb-truth), axis=1)
    isolated_error = np.max(np.abs(isolated_rgb-truth), axis=1)
    visible = alpha >= .18
    dark = source_y <= .05
    dark_visible = dark&visible
    neighbor = actual_y-isolated_y
    within = isolated_y-source_y
    contaminated = dark_visible&(neighbor > .05)
    worst = np.argsort(np.where(dark_visible, neighbor, -np.inf))[-8:][::-1]
    worst = [int(index) for index in worst if dark_visible[index]]
    return {'samples': len(truth), 'modeled_visible': int(visible.sum()), 'source_dark': int(dark.sum()),
            'source_dark_modeled_visible': int(dark_visible.sum()),
            'all_samples_max_component_error': quantiles(error),
            'modeled_visible_max_component_error': quantiles(error[visible]),
            'isolated_cell_max_component_error': quantiles(isolated_error),
            'dark_visible_total_brightening': quantiles((actual_y-source_y)[dark_visible]),
            'dark_visible_neighbor_added_luminance': quantiles(neighbor[dark_visible]),
            'dark_visible_within_cell_brightening': quantiles(within[dark_visible]),
            'dark_visible_neighbor_added_over_005': int(contaminated.sum()),
            'dark_visible_neighbor_added_over_010': int((dark_visible&(neighbor > .1)).sum()),
            'dark_visible_total_brightening_over_005': int((dark_visible&((actual_y-source_y) > .05)).sum()),
            'dark_visible_within_cell_brightening_over_005': int((dark_visible&(within > .05)).sum()),
            'dark_neighbor_induced_visibility': int((dark&visible&(isolated_alpha < .18)).sum()),
            'dark_neighbor_lost_visibility': int((dark&~visible&(isolated_alpha >= .18)).sum()),
            'worst_dark_visible_neighbors': [{'fin': i//2, 'tip': i%2, 'source_rgb': truth[i].tolist(),
                'global_rgb': rgb[i].tolist(), 'isolated_rgb': isolated_rgb[i].tolist(),
                'global_alpha': float(alpha[i]), 'isolated_alpha': float(isolated_alpha[i]),
                'neighbor_added_luminance': float(neighbor[i]), 'within_cell_brightening': float(within[i])}
                for i in worst]}


def evaluate(label, atlas, uv, slots, fractions, truth):
    height, width = atlas.shape[:2]
    cols = width//64
    query = []
    for h in CONFIG['heights_from_root']:
        q = uv[:, None, 0]+fractions[:, :, None]*(uv[:, None, 1]-uv[:, None, 0])
        q[:, :, 1] = (uv[:, 0, 1]+h*(uv[:, 2, 1]-uv[:, 0, 1]))[:, None]
        query.append(q.reshape(-1, 2))
    query = np.stack(query)
    all_global, all_isolated, rows, layouts = [], [], [], []
    mip = atlas
    for lod in range(6):
        require(time.monotonic()-STARTED < 26, 'Mip screen internal deadline reached')
        if lod:
            previous = mip
            mip = next_mip(previous)
            if lod > 1:
                del previous
        ch, cw = 32 >> lod, 64 >> lod
        require(mip.shape[0] == height >> lod and mip.shape[1] == width >> lod and cw >= 1 and ch >= 1,
                'Mip/cell alignment changed')
        x, y = slots % cols*cw, slots//cols*ch
        rectangles = np.repeat(np.column_stack((x, y, x+cw-1, y+ch-1)), 2, axis=0)
        globals_at_level, isolated_at_level = [], []
        for index, h in enumerate(CONFIG['heights_from_root']):
            g = sample(mip, query[index])
            i = sample(mip, query[index], rectangles)
            globals_at_level.append(g);isolated_at_level.append(i)
            rows.append({'lod': lod, 'height_from_root': h, **level_stats(g, i, truth)})
        all_global.append(np.stack(globals_at_level));all_isolated.append(np.stack(isolated_at_level))
        layouts.append({'lod': lod, 'dimensions': [mip.shape[1], mip.shape[0]], 'cell_dimensions': [cw, ch],
                        'sha256': hashlib.sha256(mip.tobytes()).hexdigest(),
                        'box_blocks_do_not_cross_original_cell_boundaries': True})
    np.savez_compressed(HERE/(label+'-samples.npz'), global_rgba=np.stack(all_global),
                        isolated_rgba=np.stack(all_isolated), source_rgb=truth,
                        heights=np.array(CONFIG['heights_from_root']), levels=np.arange(6))
    del mip
    return {'label': label, 'levels': layouts, 'measurements': rows,
            'neighbor_trigger_count_sum_over_lods_and_heights': sum(r['dark_visible_neighbor_added_over_005'] for r in rows)}, np.stack(all_global)[0]


def adjacency(reference, order):
    color = reference.reshape(4800, 2, 3).mean(axis=1)[order]
    luma = color@LUMA
    grid = np.arange(4800).reshape(75, 64)
    left = np.r_[grid[:, :-1].reshape(-1), grid[:-1].reshape(-1)]
    right = np.r_[grid[:, 1:].reshape(-1), grid[1:].reshape(-1)]
    contrast = np.abs(luma[left]-luma[right])
    dark_bright = ((luma[left] <= .05)&(luma[right] >= .5))|((luma[right] <= .05)&(luma[left] >= .5))
    return {'neighbor_pairs': len(left), 'tip_mean_luminance_contrast': quantiles(contrast),
            'source_dark_bright_adjacent_pairs': int(dark_bright.sum()),
            'limits': 'Two nominal-tip mean colors label cells; this is not a full-cell material classification.'}


def main():
    for row in CONFIG['inputs']:
        with (ROOT/row['path']).open('rb') as stream:
            require(hashlib.file_digest(stream, 'sha256').hexdigest() == row['sha256'], 'Pinned input changed')
    oracle = np.array([[[0, 0, 0, 0], [255, 255, 255, 255]],
                       [[0, 0, 0, 0], [255, 255, 255, 255]]], dtype=np.uint8)
    require(np.array_equal(next_mip(oracle)[0, 0], [188, 188, 188, 128]), 'Linear-light BOX oracle failed')
    glb = GLB(ROOT/CONFIG['model'])
    primitive = glb.doc['meshes'][0]['primitives'][7]
    uv = glb.array(primitive['attributes']['TEXCOORD_0'], 'VEC2', (5126,)).reshape(4800, 4, 2).astype(float)
    material = glb.doc['materials'][primitive['material']]
    texture = glb.doc['textures'][material['pbrMetallicRoughness']['baseColorTexture']['index']]
    sampler = glb.doc['samplers'][texture['sampler']]
    require(sampler == {'magFilter': 9729, 'minFilter': 9987, 'wrapS': 33071, 'wrapT': 33071}, 'Unexpected actual sampler')
    image_view = glb.doc['images'][texture['source']]['bufferView']
    require(hashlib.sha256(glb.view_bytes(image_view)).hexdigest() == CONFIG['inputs'][1]['sha256'], 'Embedded atlas differs from file')
    glb.data.close();glb.handle.close();del glb
    with Image.open(ROOT/CONFIG['atlas']) as image:
        require(image.size == (4096, 2400) and image.mode == 'RGBA', 'Unexpected unique-cell atlas')
        atlas = np.asarray(image, dtype=np.uint8)
    del image
    require(atlas.nbytes <= 40*1024**2, 'Atlas memory bound')
    original_hash = hashlib.sha256(atlas.tobytes()).hexdigest()
    slots = np.arange(4800)
    expected_left = ((slots % 64)*64+2)/4096
    expected_bottom = ((slots//64)*32+30)/2400
    require(np.max(np.abs(uv[:, 0, 0]-expected_left)) < 3e-8
            and np.max(np.abs(uv[:, 0, 1]-expected_bottom)) < 3e-8, 'Actual cell order is not fin-index row-major')
    with np.load(ROOT/CONFIG['tip_reference'], allow_pickle=False) as data:
        truth = data['source_rgb'].reshape(9600, 3)
        fractions = data['tip_fraction']
        old_nominal_uv = data['atlas_uv']
        old_lod0 = data['new_rgb'].reshape(9600, 3)
    require(truth.shape == (9600, 3) and fractions.shape == (4800, 2), 'Tip reference bound')
    current, current_lod0 = evaluate('current', atlas, uv, slots, fractions, truth)
    require(np.max(np.abs(current_lod0[2, :, :3]-old_lod0)) < 2e-6, 'LOD0 failed existing actual audit replay')
    layout = {'atlas_dimensions': [4096, 2400], 'cell_grid': [64, 75], 'cell_dimensions': [64, 32],
              'padding_pixels_at_lod0': 2, 'effective_padding_pixels_by_level': [2/(2**level) for level in range(6)],
              'actual_sampler': sampler, 'current_order': 'source fin-index row-major',
              'current_adjacency': adjacency(truth, slots),
              'mip_box_boundary_alignment': 'Cell dimensions divide every BOX block through LOD5; only bilinear reads can cross cells.'}
    packed_report = None
    if current['neighbor_trigger_count_sum_over_lods_and_heights']:
        means = truth.reshape(4800, 2, 3).mean(axis=1)
        luma_bins = np.minimum((means@LUMA*32).astype(int), 31)
        order = np.lexsort((slots, means[:, 2], means[:, 1], means[:, 0], luma_bins))
        destinations = np.empty(4800, dtype=int);destinations[order] = slots
        packed = np.empty_like(atlas)
        for target, source in enumerate(order):
            sy, sx = (source//64)*32, (source % 64)*64
            ty, tx = (target//64)*32, (target % 64)*64
            packed[ty:ty+32, tx:tx+64] = atlas[sy:sy+32, sx:sx+64]
        for source, target in enumerate(destinations):
            require(np.array_equal(atlas[(source//64)*32:(source//64+1)*32, source % 64*64:(source % 64+1)*64],
                                   packed[(target//64)*32:(target//64+1)*32, target % 64*64:(target % 64+1)*64]),
                    'Packing changed source RGBA cell bytes')
        old_origins = np.column_stack((slots % 64*64, slots//64*32))
        new_origins = np.column_stack((destinations % 64*64, destinations//64*32))
        packed_uv = (uv*np.array([4096, 2400])[None, None]-old_origins[:, None]+new_origins[:, None])/np.array([4096, 2400])
        packed_report, packed_lod0 = evaluate('packed', packed, packed_uv, destinations, fractions, truth)
        require(np.max(np.abs(packed_lod0-current_lod0)) < 2e-6, 'Packing changed LOD0 samples')
        np.savez_compressed(HERE/'packing-only-permutation.npz', destination_for_source_fin=destinations,
                            source_fin_at_destination=order, simulated_uv=packed_uv)
        packed_report['adjacency'] = adjacency(truth, order)
        packed_report['all_rgba_cell_bytes_exact'] = True
        packed_report['maximum_lod0_sample_difference'] = float(np.max(np.abs(packed_lod0-current_lod0)))
        packed_report['atlas_array_sha256'] = hashlib.sha256(packed.tobytes()).hexdigest()
        del packed
    require(hashlib.sha256(atlas.tobytes()).hexdigest() == original_hash, 'Original atlas changed')
    gc.collect()
    result = {'schema_version': 1, 'status': 'completed_cpu_filtering_screen', 'layout': layout,
              'current': current, 'packing_analytic_control': packed_report,
              'packing_trigger_met': packed_report is not None, 'original_atlas_array_sha256': original_hash,
              'source_texture_redecoded': False, 'source_rgba_and_alpha_preserved': True,
              'model_geometry_or_material_edits': False, 'actual_avatar_generated_or_rendered': False,
              'new_native_provider_ai_role_calls': 0, 'quality_approved': False,
              'filter_model': CONFIG['mip_model'], 'elapsed_seconds': time.monotonic()-STARTED,
              'limits': ['No actual GPU derivatives, fractional/trilinear LOD, visibility, occlusion, lighting or anisotropic sampling were measured.',
                         'These integer levels model possible filtering, not the level used by the observed EEVEE flecks.',
                         'Independent RGB/alpha BOX filtering can mix colors within a fin; packing cannot fix that mechanism.',
                         'A byte-identical color-coherent cell permutation is only an analytic control, not a new avatar or approved repair.',
                         'Source references are nominal-tip LOD0 colors, not a pixel-perfect ground-truth render.']}
    save('result.json', result)
    print(json.dumps({'status': result['status'], 'current_neighbor_trigger_count': current['neighbor_trigger_count_sum_over_lods_and_heights'],
                      'packed_neighbor_trigger_count': packed_report['neighbor_trigger_count_sum_over_lods_and_heights'] if packed_report else None,
                      'elapsed_seconds': result['elapsed_seconds']}, indent=2))


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        save('failure.json', {'status': 'failed_filtering_screen', 'type': type(error).__name__,
                             'message': str(error), 'elapsed_seconds': time.monotonic()-STARTED})
        raise
