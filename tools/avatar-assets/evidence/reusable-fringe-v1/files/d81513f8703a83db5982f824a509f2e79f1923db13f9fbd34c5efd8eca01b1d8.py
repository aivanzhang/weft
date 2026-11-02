"""Bounded source-colored fin cells; analytic color gates are not fur-quality approval."""

import math
import time

import numpy as np

POLICY = 'edge-fringe-source-color-atlas-v1'
SOURCE_POLICY = 'edge-fringe-source-surface-v1'
SAMPLING_POLICY = 'repeat-texel-centers-linear-lod0-srgb-v1'
MAX_FINS = 4800
MAX_DIMENSION = 4096
MAX_PIXELS = 16_777_216
MAX_SAMPLES = 262_144
MAX_SECONDS = 20
SAMPLE_BATCH = 8192
LUMA = np.array([.2126, .7152, .0722])
_ENCODED = np.arange(256, dtype=np.float64)/255
_LINEAR = np.where(_ENCODED <= .04045, _ENCODED/12.92, ((_ENCODED+.055)/1.055)**2.4)


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _array(value, shape, kinds, label):
    _require(isinstance(value, np.ndarray) and value.shape == shape and value.dtype.kind in kinds,
             'Invalid '+label)
    _require(value.dtype.kind != 'f' or np.isfinite(value).all(), 'Nonfinite '+label)
    return value


def _image(pixels):
    _require(isinstance(pixels, np.ndarray) and pixels.dtype == np.uint8 and pixels.ndim == 3
             and pixels.shape[2] in (3, 4) and 1 <= pixels.shape[0] <= MAX_DIMENSION
             and 1 <= pixels.shape[1] <= MAX_DIMENSION, 'Source needs bounded RGB(A)8 pixels')


def _uv(uv):
    _require(isinstance(uv, np.ndarray) and uv.ndim == 2 and uv.shape[1] == 2
             and 1 <= len(uv) <= MAX_SAMPLES and uv.dtype.kind == 'f'
             and np.isfinite(uv).all() and np.max(np.abs(uv)) <= 1e6,
             'Sampling UVs exceed finite coordinate/count bounds')


def _factor(factor):
    _array(factor, (3,), 'f', 'linear basecolor factor')
    _require(np.all((factor >= 0)&(factor <= 1)), 'Linear basecolor factor outside 0..1')


def encode_srgb(linear):
    return np.rint(255*np.where(linear <= .0031308, linear*12.92,
                               1.055*np.maximum(linear, 0)**(1/2.4)-.055)).clip(0, 255).astype(np.uint8)


def _sample(pixels, uv, *, rgb, repeat):
    """Decode only four fetched texels; never allocate a full linear source image."""
    height, width = pixels.shape[:2]
    result = np.empty((len(uv), 3) if rgb else (len(uv),), dtype=np.float64)
    size = np.array([width, height])
    for start in range(0, len(uv), SAMPLE_BATCH):
        stop = min(start+SAMPLE_BATCH, len(uv))
        q = uv[start:stop]
        xy = (np.mod(q, 1) if repeat else q)*size-.5
        low = np.floor(xy).astype(np.int64)
        fraction = xy-low
        high = low+1
        if repeat:
            low %= size
            high %= size
        else:
            low = np.clip(low, 0, size-1)
            high = np.clip(high, 0, size-1)
        u, v = fraction.T
        result[start:stop] = 0
        for x, y, weight in ((low[:, 0], low[:, 1], (1-u)*(1-v)),
                             (high[:, 0], low[:, 1], u*(1-v)),
                             (low[:, 0], high[:, 1], (1-u)*v),
                             (high[:, 0], high[:, 1], u*v)):
            values = _LINEAR[pixels[y, x, :3]] if rgb else pixels[y, x].astype(np.float64)/255
            result[start:stop] += values*(weight[:, None] if rgb else weight)
    return result


def sample_source_rgb(pixels, uv, factor):
    """Top-down sRGB bytes + glTF V-down UVs; REPEAT and linear-light LOD0 filtering."""
    _image(pixels)
    _uv(uv)
    _factor(factor)
    return _sample(pixels, uv, rgb=True, repeat=True)*factor


def sample_alpha(pixels, uv):
    """Linear alpha with CLAMP_TO_EDGE, matching the output atlas material."""
    _require(isinstance(pixels, np.ndarray) and pixels.dtype == np.uint8 and pixels.ndim == 2
             and 1 <= pixels.shape[0] <= MAX_DIMENSION and 1 <= pixels.shape[1] <= MAX_DIMENSION,
             'Alpha needs a bounded byte image')
    _uv(uv)
    return _sample(pixels, uv, rgb=False, repeat=False)


def _source_samples(uv, material_ids, textures):
    result = np.empty((*uv.shape[:-1], 3), dtype=np.float64)
    for slot in np.unique(material_ids):
        selected = np.flatnonzero(material_ids == slot)
        texture = textures[int(slot)]
        result[selected] = (_sample(texture['pixels'], uv[selected].reshape(-1, 2), rgb=True, repeat=True)
                            *texture['factor']).reshape(len(selected), uv.shape[1], 3)
    return result


def _validate(fins, material_ids, textures, silhouette, cell_width, cell_height, padding, batch_fins, pack_by_color):
    _require(isinstance(fins, dict) and isinstance(fins.get('report'), dict)
             and fins['report'].get('profile') == SOURCE_POLICY, 'Require source-surface fin policy')
    root_uv = fins.get('root_source_uv')
    _require(isinstance(root_uv, np.ndarray) and root_uv.ndim == 3
             and 1 <= len(root_uv) <= MAX_FINS, 'Fin count outside 1..4800')
    count = len(root_uv)
    _array(root_uv, (count, 2, 2), 'f', 'source edge UVs')
    _require(np.all((root_uv >= 0)&(root_uv <= 1)), 'Source edge UVs outside 0..1')
    positions = _array(fins.get('positions'), (count*4, 3), 'f', 'fin positions').reshape(count, 4, 3)
    joints = _array(fins.get('joints'), (count*4, 4), 'iu', 'fin joints')
    weights = _array(fins.get('weights'), (count*4, 4), 'f', 'fin weights')
    _require(joints.min() >= 0 and joints.max() < 64 and np.all((weights >= 0)&(weights <= 1))
             and np.max(np.abs(weights.sum(axis=1)-1)) < 1e-6, 'Invalid inherited skin')
    crop = _array(fins.get('uv'), (count*4, 2), 'f', 'silhouette UVs').reshape(count, 4, 2)
    _require(np.all((crop >= 0)&(crop <= 1)), 'Silhouette UVs outside 0..1')
    spans = np.linalg.norm(positions[:, 1].astype(np.float64)-positions[:, 0], axis=1)
    _require(np.all((spans >= .0085-1e-6)&(spans <= .020+1e-6)), 'Edge span outside 8.5..20 mm')
    tiles = np.floor(crop*4).astype(np.int32)
    local = crop*4-tiles
    expected = np.empty((count, 4, 2))
    expected[:, [0, 2], 0] = (.5-spans/(2*.0208))[:, None]
    expected[:, [1, 3], 0] = (.5+spans/(2*.0208))[:, None]
    expected[:, [0, 1], 1] = .985
    expected[:, [2, 3], 1] = .015
    _require(np.all(tiles == tiles[:, :1]) and np.max(np.abs(local-expected)) < 2e-6,
             'Silhouette crop does not preserve the physical two-tip mapping')
    _array(material_ids, (count,), 'iu', 'material IDs')
    _require(isinstance(textures, dict) and 1 <= len(textures) <= 32, 'Invalid source texture map')
    seen, source_pixels = set(), 0
    for slot, texture in textures.items():
        _require(type(slot) is int and 0 <= slot < 32 and isinstance(texture, dict)
                 and set(texture) == {'pixels', 'factor'}, 'Invalid source texture descriptor')
        _image(texture['pixels'])
        _factor(texture['factor'])
        if id(texture['pixels']) not in seen:
            seen.add(id(texture['pixels']))
            source_pixels += math.prod(texture['pixels'].shape[:2])
    _require(source_pixels <= MAX_PIXELS and set(map(int, np.unique(material_ids))).issubset(textures),
             'Source image budget or material binding invalid')
    _require(isinstance(silhouette, np.ndarray) and silhouette.dtype == np.uint8
             and silhouette.shape == (1024, 1024), 'Require the 1024px physical alpha atlas')
    _require(type(cell_width) is int and cell_width == 64 and type(cell_height) is int and cell_height == 32
             and type(padding) is int and 2 <= padding <= 4, 'Require 64x32 cells and 2..4 px padding')
    _require(type(batch_fins) is int and 1 <= batch_fins <= 128, 'Batch count outside 1..128')
    _require(type(pack_by_color) is bool, 'pack_by_color must be boolean')
    return root_uv, crop, spans, source_pixels


def _tip_audit(atlas, uv, fractions, truth):
    query = uv[:, None, 0]+fractions[:, :, None]*(uv[:, None, 1]-uv[:, None, 0])
    query[:, :, 1] = ((uv[:, 0, 1]+uv[:, 2, 1])*.5)[:, None]
    actual = _sample(atlas, query.reshape(-1, 2), rgb=True, repeat=False).reshape(-1, 2, 3)
    alpha = _sample(atlas[..., 3], query.reshape(-1, 2), rgb=False, repeat=False).reshape(-1, 2)
    error = np.max(np.abs(actual-truth), axis=2)
    p50, p95, p99 = np.quantile(error, [.5, .95, .99])
    count = error.size
    color_passed = p95 <= .02 and p99 <= .05 and np.count_nonzero(error > .10) <= math.floor(count*.005)
    alpha_passed = bool(np.all(alpha >= .18))
    worst = np.argsort(-error.reshape(-1), kind='stable')[:12]
    report = {
        'method': 'All nominal +/-2.5 mm tips at half fin height; maximum absolute linear RGB component error',
        'tip_count': count, 'p50': float(p50), 'p95': float(p95), 'p99': float(p99),
        'maximum_linear_component_error': float(error.max()),
        'tips_over_005': int(np.count_nonzero(error > .05)), 'tips_over_010': int(np.count_nonzero(error > .10)),
        'fixed_color_gate': {'p95_max': .02, 'p99_max': .05, 'tips_over_010_max': math.floor(count*.005)},
        'minimum_alpha': float(alpha.min()), 'tips_below_mask_cutoff': int(np.count_nonzero(alpha < .18)),
        'alpha_cutoff': .18, 'color_gate_passed': bool(color_passed), 'alpha_gate_passed': alpha_passed,
        'passed': bool(color_passed and alpha_passed), 'omitted_or_hidden_samples': 0,
        'worst_tips': [{'fin': int(i//2), 'tip': int(i % 2), 'error': float(error.reshape(-1)[i]),
                        'source_rgb': truth.reshape(-1, 3)[i].tolist(), 'atlas_rgb': actual.reshape(-1, 3)[i].tolist(),
                        'alpha': float(alpha.reshape(-1)[i])} for i in worst],
    }
    return report, {'source_rgb': truth, 'atlas_rgb': actual, 'error': error, 'alpha': alpha,
                    'atlas_uv': query, 'tip_fraction': fractions}


def build_color_atlas(fins, material_ids, textures, silhouette, *, cell_width=64, cell_height=32,
                      padding=2, batch_fins=128, pack_by_color=True):
    """Return new pixels/UVs in source-fin order; permutations describe cells, not geometry."""
    started = time.monotonic()

    def deadline():
        _require(time.monotonic()-started < MAX_SECONDS, 'Atlas builder exceeded its 20-second internal deadline')

    root_uv, crop, spans, source_pixels = _validate(fins, material_ids, textures, silhouette,
                                                  cell_width, cell_height, padding, batch_fins, pack_by_color)
    count = len(root_uv)
    fractions = .5+np.array([-.0025, .0025])/spans[:, None]
    nominal_uv = root_uv[:, None, 0]+fractions[:, :, None]*(root_uv[:, None, 1]-root_uv[:, None, 0])
    truth = _source_samples(nominal_uv, material_ids, textures)
    ids = np.arange(count, dtype=np.int32)
    means = truth.mean(axis=1)
    luma_bins = np.minimum((means@LUMA*32).astype(np.int32), 31)
    order = np.lexsort((ids, means[:, 2], means[:, 1], means[:, 0], luma_bins)).astype(np.int32) if pack_by_color else ids.copy()
    destinations = np.empty(count, dtype=np.int32)
    destinations[order] = ids
    columns = min(count, MAX_DIMENSION//cell_width)
    rows = math.ceil(count/columns)
    width, height = columns*cell_width, rows*cell_height
    _require(width <= MAX_DIMENSION and height <= MAX_DIMENSION and width*height <= MAX_PIXELS,
             'Atlas exceeds dimension/pixel bound')
    deadline()
    atlas = np.zeros((height, width, 4), dtype=np.uint8)
    uv_out = np.empty((count, 4, 2), dtype=np.float32)
    inner_width, inner_height = cell_width-2*padding, cell_height-2*padding
    t = (np.arange(inner_width)+.5)/inner_width
    v = (np.arange(inner_height)+.5)/inner_height
    for start in range(0, count, batch_fins):
        deadline()
        stop = min(count, start+batch_fins)
        size = stop-start
        edge_uv = root_uv[start:stop, :1]+t[None, :, None]*(root_uv[start:stop, 1:]-root_uv[start:stop, :1])
        rgb = encode_srgb(_source_samples(edge_uv, material_ids[start:stop], textures))
        selected = crop[start:stop]
        alpha_uv = np.empty((size, inner_height, inner_width, 2))
        alpha_uv[..., 0] = selected[:, None, :1, 0]+t[None, None, :]*(selected[:, None, 1:2, 0]-selected[:, None, :1, 0])
        alpha_uv[..., 1] = selected[:, None, 2:3, 1]+v[None, :, None]*(selected[:, None, :1, 1]-selected[:, None, 2:3, 1])
        alpha = np.rint(255*_sample(silhouette, alpha_uv.reshape(-1, 2), rgb=False, repeat=False)).astype(np.uint8)
        alpha = alpha.reshape(size, inner_height, inner_width)
        # Transparent gutters retain edge RGB without inventing opaque coverage.
        padded_rgb = np.pad(rgb, ((0, 0), (padding, padding), (0, 0)), mode='edge')
        for local, source in enumerate(range(start, stop)):
            destination = int(destinations[source])
            x, y = (destination % columns)*cell_width, (destination//columns)*cell_height
            atlas[y:y+cell_height, x:x+cell_width, :3] = padded_rgb[local][None]
            atlas[y+padding:y+padding+inner_height, x+padding:x+padding+inner_width, 3] = alpha[local]
            left, right = (x+padding)/width, (x+padding+inner_width)/width
            top, bottom = (y+padding)/height, (y+padding+inner_height)/height
            uv_out[source] = ((left, bottom), (right, bottom), (left, top), (right, top))
    deadline()
    tip_audit, nominal = _tip_audit(atlas, uv_out.astype(np.float64), fractions, truth)
    nominal['source_uv'] = nominal_uv
    deadline()
    report = {
        'policy': POLICY, 'source_fin_policy': SOURCE_POLICY, 'fins': count,
        'atlas_dimensions': [width, height], 'pixels': width*height, 'cell_grid': [columns, rows],
        'cell_dimensions': [cell_width, cell_height], 'padding_px': padding,
        'interior_samples': [inner_width, inner_height], 'batch_fins': batch_fins,
        'unused_cells': columns*rows-count, 'unused_cell_rgba': [0, 0, 0, 0],
        'source_pixels': source_pixels, 'source_material_ids': sorted(map(int, np.unique(material_ids))),
        'sampling_policy': SAMPLING_POLICY, 'image_rows': 'top-down', 'uv_convention': 'glTF V-down',
        'packing': {'enabled': pack_by_color, 'policy': 'nominal-tip-mean-32-luma-bins-rgb-stable-v1' if pack_by_color else 'source-fin-order',
                    'permutation_scope': 'Cells only; UV and all diagnostic arrays retain original fin order',
                    'unused_cell_policy': 'Trailing unassigned cells stay transparent black; not populated duplicates'},
        'nominal_tip_audit': tip_audit, 'technical_gate_passed': tip_audit['passed'],
        'status': 'passed_technical_color_and_alpha_gates' if tip_audit['passed'] else 'rejected_technical_color_or_alpha_gate',
        'geometry_mutated': False, 'weights_mutated': False, 'source_colors_whitened_or_suppressed': False,
        'alpha_policy': 'Resample unchanged physical two-tip UV crop; no per-fin width stretch',
        'maximum_horizontal_sample_pitch_m': float(spans.max()/inner_width),
        'material_requirements': {'baseColorFactor': [1, 1, 1, 1], 'vertex_color': 'omit', 'alphaMode': 'MASK',
                                  'alphaCutoff': .18, 'sampler': {'magFilter': 9729, 'minFilter': 9987, 'wrapS': 33071, 'wrapT': 33071}},
        'bounds': {'max_fins': MAX_FINS, 'max_dimension': MAX_DIMENSION, 'max_pixels': MAX_PIXELS,
                   'internal_seconds': MAX_SECONDS, 'sample_batch': SAMPLE_BATCH},
        'limits': ['LOD0 CPU gates do not prove GPU filtering, lighting, visibility or avatar quality.',
                   'Color follows the source edge along fin height; this is not recovered volumetric fur.',
                   'Finite samples/sRGB8 quantization can lose subtexel markings; failing tips remain reported.',
                   'Color-coherent packing does not eliminate within-cell blending or deep-mip/anisotropic bleed.',
                   'Source alpha is ignored; the caller must admit only opaque source color materials.',
                   'Unused cells are transparent black and can affect coarse filtering near the final row.'],
        'elapsed_seconds': time.monotonic()-started, 'quality_approved': False,
    }
    return {'pixels': atlas, 'uv': uv_out.reshape(count*4, 2), 'report': report,
            'permutation': {'destination_for_source_fin': destinations, 'source_fin_at_destination': order},
            'nominal_tips': nominal}
