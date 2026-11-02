"""Analytic source-color preservation screen, not an avatar or GPU rendering test."""

import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE/'sources'))
from area_fringe import build_edge_fringe
from fixture import fixture
from root_sampler import repeat_root_colors
from color_atlas import build_color_atlas, encode_srgb, sample_alpha

STARTED = time.monotonic()


def scalar(value):
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError('Unsupported report value: '+type(value).__name__)


def save(name, value):
    text = json.dumps(value, indent=2, sort_keys=True, allow_nan=False, default=scalar)+'\n'
    with (HERE/name).open('x') as stream:
        stream.write(text)


def digest_arrays(values):
    return {k: hashlib.sha256(v.tobytes()).hexdigest() for k, v in values.items() if isinstance(v, np.ndarray)}


def decode(value):
    value = value.astype(float)/255
    return np.where(value <= .04045, value/12.92, ((value+.055)/1.055)**2.4)


def error_stats(error, active):
    values = error[active]
    assert len(values) > 0
    return {'mean_absolute_linear_rgb': float(values.mean()),
            'p95_max_channel_error': float(np.quantile(values.max(axis=1), .95)),
            'maximum_channel_error': float(values.max()), 'visible_samples': len(values)}


def atlas_checks(fins, output, source, silhouette):
    count = len(fins['root_source_uv'])
    atlas = output['pixels']
    uv = output['uv'].reshape(count, 4, 2)
    t = (np.arange(60)+.5)/60
    columns = atlas.shape[1]//64
    expected_uv = fins['root_source_uv'][:, :1]+t[None, :, None]*(fins['root_source_uv'][:, 1:]-fins['root_source_uv'][:, :1])
    truth = repeat_root_colors(source, expected_uv.reshape(-1, 2), np.ones(3)).reshape(count, 60, 3)
    ends = repeat_root_colors(source, fins['root_source_uv'].reshape(-1, 2), np.ones(3)).reshape(count, 2, 3)
    old = ends[:, :1]*(1-t[None, :, None])+ends[:, 1:]*t[None, :, None]
    actual, visible = np.empty_like(truth), np.empty((count, 60), dtype=bool)
    for index in range(count):
        x, y = (index % columns)*64, (index//columns)*32
        cell = atlas[y:y+32, x:x+64]
        assert not cell[:2, :, 3].any() and not cell[-2:, :, 3].any()
        assert not cell[:, :2, 3].any() and not cell[:, -2:, 3].any()
        actual[index] = decode(cell[16, 2:62, :3])
        visible[index] = (cell[2:30, 2:62, 3] >= 255*.18).any(axis=0)
    center_new = error_stats(np.abs(actual-truth), visible)
    center_old = error_stats(np.abs(old-truth), visible)
    assert center_new['maximum_channel_error'] < .0046
    luminance = np.array([.2126, .7152, .0722])
    dark = (truth@luminance < .05)&visible
    center_new['mean_dark_source_brightening'] = float(np.maximum((actual-truth)@luminance, 0)[dark].mean()) if dark.any() else None
    center_old['mean_dark_source_brightening'] = float(np.maximum((old-truth)@luminance, 0)[dark].mean()) if dark.any() else None

    # Fractional texel positions expose resampling loss hidden by tests at atlas texel centers.
    ids = np.linspace(0, count-1, 256, dtype=int)
    q = (np.arange(257)+.37)/257
    query_source = fins['root_source_uv'][ids, :1]+q[None, :, None]*(fins['root_source_uv'][ids, 1:]-fins['root_source_uv'][ids, :1])
    reference = repeat_root_colors(source, query_source.reshape(-1, 2), np.ones(3)).reshape(len(ids), len(q), 3)
    query_atlas = uv[ids, None, 2]+q[None, :, None]*(uv[ids, None, 3]-uv[ids, None, 2])
    query_atlas[..., 1] = ((uv[ids, 0, 1]+uv[ids, 2, 1])*.5)[:, None]
    sampled = repeat_root_colors(atlas, query_atlas.reshape(-1, 2), np.ones(3)).reshape(len(ids), len(q), 3)
    alpha = sample_alpha(atlas[:, :, 3], query_atlas.reshape(-1, 2)).reshape(len(ids), len(q))
    old_fractional = ends[ids, :1]*(1-q[None, :, None])+ends[ids, 1:]*q[None, :, None]
    active = alpha >= .18
    fractional_new = error_stats(np.abs(sampled-reference), active)
    fractional_old = error_stats(np.abs(old_fractional-reference), active)

    spans = np.linalg.norm(fins['positions'].reshape(count, 4, 3)[:, 1]-fins['positions'].reshape(count, 4, 3)[:, 0], axis=1)
    widths = []
    fine_t = (np.arange(4096)+.5)/4096
    for index in (int(np.argmin(spans)), int(np.argsort(spans)[count//2]), int(np.argmax(spans))):
        old_uv = fins['uv'].reshape(count, 4, 2)[index]
        a = old_uv[2]+fine_t[:, None]*(old_uv[3]-old_uv[2])
        a[:, 1] = (old_uv[0, 1]+old_uv[2, 1])*.5
        b = uv[index, 2]+fine_t[:, None]*(uv[index, 3]-uv[index, 2])
        b[:, 1] = (uv[index, 0, 1]+uv[index, 2, 1])*.5
        ranges = []
        for pixels, coords in ((silhouette, a), (atlas[:, :, 3], b)):
            mask = sample_alpha(pixels, coords) >= .18
            diff = np.diff(np.r_[False, mask, False].astype(int))
            begins, ends_at = np.flatnonzero(diff == 1), np.flatnonzero(diff == -1)
            assert len(begins) == 2
            ranges.append({'widths_m': ((ends_at-begins)*spans[index]/4096).tolist(),
                           'centers_m': (((begins+ends_at)*.5/4096-.5)*spans[index]).tolist()})
        maximum_width_change = float(np.max(np.abs(np.array(ranges[0]['widths_m'])-ranges[1]['widths_m'])))
        assert maximum_width_change <= 2*spans[index]/60
        widths.append({'source_fin': index, 'edge_span_m': float(spans[index]),
                       'source_alpha': ranges[0], 'resampled_alpha': ranges[1],
                       'maximum_width_change_m': maximum_width_change,
                       'two_horizontal_texel_bound_m': float(2*spans[index]/60)})
    worst = int(np.argmax(np.where(visible, np.max(np.abs(old-truth), axis=2), 0).max(axis=1)))
    return {'texel_centers': {'atlas': center_new, 'endpoint_interpolation': center_old},
            'fractional_lod0': {'atlas': fractional_new, 'endpoint_interpolation': fractional_old,
                               'fins_sampled': len(ids), 'samples_per_fin': len(q)},
            'physical_alpha_width_checks': widths,
            'limits': ['Fractional LOD0 is CPU bilinear sampling, not actual GPU validation.',
                       'Deep mipmaps, anisotropic filtering, alpha-to-coverage and perspective remain unverified.']}, (truth[worst], old[worst], actual[worst], visible[worst])


def main():
    source_fixture = fixture((24, 24, 24))
    settings = {'seed': 912, 'min_fins': 1, 'max_fins': 4800, 'length_m': .008,
                'min_spacing_m': .003, 'min_edge_m': .0085, 'max_edge_m': .020, 'comb': .15}
    fins = build_edge_fringe(**source_fixture, settings=settings,
                            source_export_vertices=len(source_fixture['points']))
    with np.load(HERE/'inputs/area_share_roots.npz', allow_pickle=False) as saved:
        assert np.array_equal(fins['root_vertex_ids'], saved['root_vertex_ids'])
        assert np.array_equal(fins['root_source_uv'], saved['root_uv'])
        assert np.array_equal(fins['root_centers'], saved['root_centers'])
    original_hashes = digest_arrays(fins)
    with Image.open(HERE/'inputs/silhouette.png') as image:
        silhouette = np.array(image.getchannel('A'), dtype=np.uint8)
    assert silhouette.shape == (1024, 1024)
    material_ids = np.zeros(4800, dtype=np.int32)
    bw = np.broadcast_to((((np.arange(512)//8) % 2)*255)[None, :, None], (64, 512, 3)).astype(np.uint8)
    x = np.linspace(0, 1, 512)[None, :]
    y = np.linspace(0, 1, 128)[:, None]
    gradient = np.empty((128, 512, 3))
    gradient[..., 0] = x
    gradient[..., 1] = .1+.8*y
    gradient[..., 2] = .1+.8*(1-x)*y
    gradient = encode_srgb(gradient)
    assert np.array_equal(encode_srgb(np.array([0., .5, 1.])), np.array([0, 188, 255], dtype=np.uint8))
    oracle = np.array([[[255, 0, 0], [0, 255, 0]], [[0, 0, 255], [255, 255, 255]]], dtype=np.uint8)
    coords = np.array([[.25, .25], [.75, .25], [.25, .75], [.75, .75], [0., 0.], [1., 1.], [0., .25]])
    expected = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1], [.5, .5, .5], [.5, .5, .5], [.5, .5, 0]])
    assert np.array_equal(repeat_root_colors(oracle, coords, np.ones(3)), expected)
    reports, example_rows = {}, []
    deterministic = None
    for name, pixels in (('black_white_transition', bw), ('colored_gradient', gradient)):
        textures = {0: {'pixels': pixels, 'factor': np.ones(3)}}
        source_hash = hashlib.sha256(pixels.tobytes()).hexdigest()
        output = build_color_atlas(fins, material_ids, textures, silhouette)
        checks, examples = atlas_checks(fins, output, pixels, silhouette)
        if name == 'black_white_transition':
            assert checks['fractional_lod0']['atlas']['mean_absolute_linear_rgb'] < .5*checks['fractional_lod0']['endpoint_interpolation']['mean_absolute_linear_rgb']
            assert checks['texel_centers']['endpoint_interpolation']['maximum_channel_error'] > .2
            duplicate = build_color_atlas(fins, material_ids, textures, silhouette)
            assert np.array_equal(output['pixels'], duplicate['pixels']) and np.array_equal(output['uv'], duplicate['uv'])
            deterministic = {'atlas_sha256': hashlib.sha256(output['pixels'].tobytes()).hexdigest(),
                             'uv_sha256': hashlib.sha256(output['uv'].tobytes()).hexdigest(),
                             'full_4800_fin_repetition_byte_identical': True}
            del duplicate
        assert original_hashes == digest_arrays(fins)
        assert source_hash == hashlib.sha256(pixels.tobytes()).hexdigest()
        Image.fromarray(pixels).save(HERE/(name+'-source.png'))
        Image.fromarray(output['pixels']).save(HERE/(name+'-atlas.png'))
        np.savez_compressed(HERE/(name+'-uv.npz'), uv=output['uv'])
        reports[name] = {'source_sha256': source_hash, 'settings_and_bounds': output['report'], 'measurements': checks}
        example_rows.append((name, examples))
        del output
    invalid = []
    base_texture = {0: {'pixels': bw, 'factor': np.ones(3)}}
    too_many = {**fins, 'root_source_uv': np.zeros((4801, 2, 2))}
    bad_uv = {**fins, 'root_source_uv': fins['root_source_uv'].copy()}
    bad_uv['root_source_uv'][0, 0, 0] = np.nan
    virtual = np.zeros((1, 1, 3), dtype=np.uint8)
    over_pixels = {i: {'pixels': np.broadcast_to(virtual, (4096, 4096, 3)), 'factor': np.ones(3)} for i in (0, 1)}
    cases = [('fin_count', {'fins': too_many}), ('cell_width', {'cell_width': 65}),
             ('batch_count', {'batch_fins': 129}), ('nonfinite_uv', {'fins': bad_uv}),
             ('source_pixel_budget', {'textures': over_pixels}),
             ('source_dimension', {'textures': {0: {'pixels': np.zeros((1, 4097, 3), dtype=np.uint8), 'factor': np.ones(3)}}})]
    for name, changes in cases:
        args = {'fins': fins, 'material_ids': material_ids, 'textures': base_texture, 'silhouette': silhouette, **changes}
        try:
            build_color_atlas(**args)
        except ValueError as error:
            invalid.append({'case': name, 'error': str(error)})
        else:
            raise AssertionError('Invalid atlas input accepted: '+name)
    image = Image.new('RGB', (900, 310), '#e9e4da')
    draw = ImageDraw.Draw(image)
    for row, (name, (truth, old, new, visible)) in enumerate(example_rows):
        draw.text((12, 12+row*150), name+' — source / old endpoint / new atlas', fill='#202020')
        for line, values in enumerate((truth, old, new)):
            strip = encode_srgb(values)[None]
            rgb = np.repeat(strip, 18, axis=0)
            rgb[:, ~visible] = [233, 228, 218]
            image.paste(Image.fromarray(rgb).resize((840, 28), Image.Resampling.NEAREST), (12, 35+row*150+line*32))
    image.save(HERE/'analytic-color-comparison.png')
    result = {'schema_version': 1, 'status': 'passed_color_atlas_analytic_screen',
              'kind': 'analytic_source_edge_color_atlas_not_avatar', 'cases': reports,
              'determinism': deterministic, 'fin_array_hashes_unchanged': original_hashes,
              'selection_matches_previous_frozen_area_allocation': True, 'invalid_inputs_rejected': invalid,
              'known_color_oracle_cases': 7, 'known_srgb_encoding_values': [0, 188, 255],
              'elapsed_seconds': time.monotonic()-STARTED,
              'quality_approved': False, 'actual_avatar_used': False, 'gpu_or_native_used': False,
              'new_provider_or_ai_role_calls': 0,
              'limits': ['Atlas samples improve source-edge color matching but are not exact texture equivalence.',
                         'No whitening, color suppression, source-mesh edits or new skin anchors are performed.',
                         'The source edge is the color assumption for the full protruding fin height.',
                         '60×28 interior samples, sRGB8 quantization and mip filtering limit retained detail.',
                         'The previous 512px phase-dependent merging and 256px unresolved-tip limits remain.']}
    save('result.json', result)
    print(json.dumps({'status': result['status'], 'elapsed_seconds': result['elapsed_seconds'],
                      'errors': {name: data['measurements']['fractional_lod0'] for name, data in reports.items()}}, indent=2))


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        save('failure.json', {'status': 'failed_color_atlas_screen', 'error_type': type(error).__name__,
                             'error': str(error), 'elapsed_seconds': time.monotonic()-STARTED})
        raise
