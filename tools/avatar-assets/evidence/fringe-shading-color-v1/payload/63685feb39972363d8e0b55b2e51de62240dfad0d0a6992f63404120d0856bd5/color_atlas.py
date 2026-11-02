"""Bounded experimental per-fin source-color atlas; geometry is read-only."""

import math
import time

import numpy as np

from root_sampler import repeat_root_colors

POLICY = 'source-edge-color-atlas-analytic-v1'
SOURCE_POLICY = 'area-share-edge-fin-analytic-v1'
MAX_FINS = 4800
MAX_DIMENSION = 4096
MAX_PIXELS = 16_777_216


def require(condition, message):
    if not condition:
        raise ValueError(message)


def checked_array(value, shape, kinds, label):
    require(isinstance(value, np.ndarray) and value.shape == shape and value.dtype.kind in kinds,
            'Invalid '+label)
    require(value.dtype.kind != 'f' or np.isfinite(value).all(), 'Nonfinite '+label)
    return value


def encode_srgb(linear):
    return np.rint(255*np.where(linear <= .0031308, linear*12.92,
                               1.055*np.maximum(linear, 0)**(1/2.4)-.055)).clip(0, 255).astype(np.uint8)


def sample_alpha(pixels, uv):
    size = np.array([pixels.shape[1], pixels.shape[0]])
    xy = uv*size-.5
    lower = np.floor(xy).astype(np.int64)
    fraction = xy-lower
    upper = np.clip(lower+1, 0, size-1)
    lower = np.clip(lower, 0, size-1)
    u, v = fraction.T
    return (pixels[lower[:, 1], lower[:, 0]]*(1-u)*(1-v)
            +pixels[lower[:, 1], upper[:, 0]]*u*(1-v)
            +pixels[upper[:, 1], lower[:, 0]]*(1-u)*v
            +pixels[upper[:, 1], upper[:, 0]]*u*v)/255


def build_color_atlas(fins, material_ids, textures, silhouette, *,
                      cell_width=64, cell_height=32, padding=2, batch_fins=128):
    started = time.monotonic()
    require(isinstance(fins, dict) and fins.get('report', {}).get('profile') == SOURCE_POLICY,
            'Require the bound area-weighted fin policy')
    uv_source = fins.get('root_source_uv')
    require(isinstance(uv_source, np.ndarray) and uv_source.ndim == 3
            and 1 <= len(uv_source) <= MAX_FINS, 'Fin count outside 1..4800')
    count = len(uv_source)
    uv_source = checked_array(uv_source, (count, 2, 2), 'f', 'source edge UVs')
    require(np.all((uv_source >= 0)&(uv_source <= 1)), 'Source edge UVs outside 0..1')
    positions = checked_array(fins.get('positions'), (count*4, 3), 'f', 'fin positions')
    joints = checked_array(fins.get('joints'), (count*4, 4), 'iu', 'fin joints')
    weights = checked_array(fins.get('weights'), (count*4, 4), 'f', 'fin weights')
    require(joints.max() < 64 and joints.min() >= 0 and np.all(weights >= 0)
            and np.max(np.abs(weights.sum(axis=1)-1)) < 1e-6, 'Invalid inherited skin')
    source_uv = checked_array(fins.get('uv'), (count*4, 2), 'f', 'silhouette UVs').reshape(count, 4, 2)
    require(np.all((source_uv >= 0)&(source_uv <= 1)), 'Silhouette UVs outside 0..1')
    require(np.all(np.abs(source_uv[:, 0, 0]-source_uv[:, 2, 0]) < 1e-7)
            and np.all(np.abs(source_uv[:, 1, 0]-source_uv[:, 3, 0]) < 1e-7)
            and np.all(np.abs(source_uv[:, 0, 1]-source_uv[:, 1, 1]) < 1e-7)
            and np.all(np.abs(source_uv[:, 2, 1]-source_uv[:, 3, 1]) < 1e-7)
            and np.all(source_uv[:, 1, 0] > source_uv[:, 0, 0])
            and np.all(source_uv[:, 0, 1] > source_uv[:, 2, 1]), 'Require axis-aligned two-tip UV crops')
    spans = np.linalg.norm(positions.reshape(count, 4, 3)[:, 1]-positions.reshape(count, 4, 3)[:, 0], axis=1)
    require(spans.min() >= .0085-1e-6 and spans.max() <= .020+1e-6, 'Edge span outside frozen 8.5..20 mm range')
    tiles = np.floor(source_uv*4).astype(int)
    require(np.all(tiles == tiles[:, :1])
            and np.max(np.abs((source_uv[:, 1, 0]-source_uv[:, 0, 0])*4-spans/.0208)) < 1e-5
            and np.max(np.abs((source_uv[:, 0, 1]-source_uv[:, 2, 1])*4-.97)) < 1e-6,
            'Silhouette crop does not preserve the physical fin mapping')
    material_ids = checked_array(material_ids, (count,), 'iu', 'material IDs')
    require(isinstance(textures, dict) and 1 <= len(textures) <= 32, 'Invalid source texture map')
    seen, total_pixels = set(), 0
    for slot, texture in textures.items():
        require(type(slot) is int and 0 <= slot < 32 and isinstance(texture, dict)
                and set(texture) == {'pixels', 'factor'}, 'Invalid source texture descriptor')
        image = texture['pixels']
        require(isinstance(image, np.ndarray) and image.dtype == np.uint8 and image.ndim == 3
                and image.shape[2] in (3, 4) and 1 <= image.shape[0] <= 4096
                and 1 <= image.shape[1] <= 4096, 'Source image exceeds RGB(A)8 bounds')
        if id(image) not in seen:
            seen.add(id(image))
            total_pixels += image.shape[0]*image.shape[1]
        factor = checked_array(texture['factor'], (3,), 'f', 'linear basecolor factor')
        require(np.all((factor >= 0)&(factor <= 1)), 'Invalid basecolor factor')
    require(total_pixels <= MAX_PIXELS and set(map(int, np.unique(material_ids))).issubset(textures),
            'Source image budget or material binding invalid')
    require(isinstance(silhouette, np.ndarray) and silhouette.dtype == np.uint8
            and silhouette.shape == (1024, 1024), 'Require the frozen 1024px alpha atlas')
    require(type(cell_width) is int and 16 <= cell_width <= 64
            and type(cell_height) is int and 8 <= cell_height <= 32
            and type(padding) is int and 2 <= padding <= 4
            and cell_width-2*padding >= 8 and cell_height-2*padding >= 4,
            'Invalid bounded atlas cell/padding')
    require(type(batch_fins) is int and 1 <= batch_fins <= 128, 'Batch count outside 1..128')
    columns = min(count, MAX_DIMENSION//cell_width)
    rows = math.ceil(count/columns)
    width, height = columns*cell_width, rows*cell_height
    require(width <= MAX_DIMENSION and height <= MAX_DIMENSION and width*height <= MAX_PIXELS,
            'Atlas exceeds dimension/pixel bound')
    atlas = np.zeros((height, width, 4), dtype=np.uint8)
    uv_out = np.empty((count, 4, 2), dtype=np.float32)
    inner_width, inner_height = cell_width-2*padding, cell_height-2*padding
    t = (np.arange(inner_width)+.5)/inner_width
    v = (np.arange(inner_height)+.5)/inner_height
    for start in range(0, count, batch_fins):
        require(time.monotonic()-started < 20, 'Atlas builder exceeded its 20-second internal deadline')
        stop = min(count, start+batch_fins)
        size = stop-start
        edge_uv = uv_source[start:stop, :1]+t[None, :, None]*(uv_source[start:stop, 1:]-uv_source[start:stop, :1])
        rgb = np.empty((size, inner_width, 3), dtype=np.uint8)
        for slot in np.unique(material_ids[start:stop]):
            selected = np.flatnonzero(material_ids[start:stop] == slot)
            texture = textures[int(slot)]
            linear = repeat_root_colors(texture['pixels'], edge_uv[selected].reshape(-1, 2), texture['factor'])
            rgb[selected] = encode_srgb(linear).reshape(len(selected), inner_width, 3)
        crop = source_uv[start:stop]
        alpha_uv = np.empty((size, inner_height, inner_width, 2))
        alpha_uv[..., 0] = crop[:, None, :1, 0]+t[None, None, :]*(crop[:, None, 1:2, 0]-crop[:, None, :1, 0])
        alpha_uv[..., 1] = crop[:, None, 2:3, 1]+v[None, :, None]*(crop[:, None, :1, 1]-crop[:, None, 2:3, 1])
        alpha = np.rint(255*sample_alpha(silhouette, alpha_uv.reshape(-1, 2))).astype(np.uint8).reshape(size, inner_height, inner_width)
        # Extruded RGB in transparent gutters avoids introducing black at base-LOD boundaries.
        padded_rgb = np.pad(rgb, ((0, 0), (padding, padding), (0, 0)), mode='edge')
        for local, index in enumerate(range(start, stop)):
            x, y = (index % columns)*cell_width, (index//columns)*cell_height
            atlas[y:y+cell_height, x:x+cell_width, :3] = padded_rgb[local][None]
            atlas[y+padding:y+padding+inner_height, x+padding:x+padding+inner_width, 3] = alpha[local]
            left, right = (x+padding)/width, (x+padding+inner_width)/width
            top, bottom = (y+padding)/height, (y+padding+inner_height)/height
            uv_out[index] = ((left, bottom), (right, bottom), (left, top), (right, top))
    return {'pixels': atlas, 'uv': uv_out.reshape(count*4, 2), 'report': {
        'policy': POLICY, 'source_fin_policy': SOURCE_POLICY, 'fins': count,
        'atlas_dimensions': [width, height], 'pixels': width*height,
        'cell_dimensions': [cell_width, cell_height], 'padding_px': padding,
        'interior_samples': [inner_width, inner_height], 'batch_fins': batch_fins,
        'source_pixels': total_pixels, 'sampling': 'REPEAT texel centers, linear-light LOD0, sRGB8 encoding',
        'geometry_mutated': False, 'weights_mutated': False, 'source_colors_whitened_or_suppressed': False,
        'alpha_policy': 'resample unchanged physical two-tip UV mapping; no per-fin width stretch',
        'maximum_horizontal_sample_pitch_m': float(spans.max()/inner_width),
        'material_requirements': 'White baseColorFactor; omit/neutralize former fin COLOR_0 to avoid double tint; MASK cutoff 0.18.',
        'limits': ['Finite samples and sRGB8 quantization do not reproduce arbitrary subtexel source markings.',
                   'Color follows each source edge and extends along fin height, not a recovered volumetric coat.',
                   'Two-pixel gutters are not proof of isolation at deep mip levels or anisotropic filtering.',
                   'Per-fin UVs follow the existing four-vertex skin; bending does not create extra attachment points.',
                   'Source texture RGB assumes the originating material is opaque; source alpha is not a shape mask.'],
        'elapsed_seconds': time.monotonic()-started, 'quality_approved': False}}
