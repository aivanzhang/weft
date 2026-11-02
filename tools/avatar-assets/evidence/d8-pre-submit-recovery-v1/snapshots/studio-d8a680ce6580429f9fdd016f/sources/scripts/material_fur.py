"""Pure material ownership and bounded color helpers for opt-in fur adapters."""

import math

import numpy as np


OWNERSHIP_POLICY = 'unique-material-and-exact-textured-uv-v1'
SELECTION_POLICY = 'legacy-nearest-repeat-encoded-selection-v1'
MAX_VERTICES = 100000
MAX_LOOPS = 300000
MAX_GROUPS = 32
MAX_PIXELS = 16777216


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _numeric_vector(value, count, label):
    _require(isinstance(value, (list, tuple, np.ndarray)), 'Invalid '+label)
    if isinstance(value, (list, tuple)):
        _require(not any(isinstance(x, (bool, np.bool_)) for x in value), 'Boolean '+label+' is unsupported')
    array = np.asarray(value)
    _require(array.shape == (count,) and array.dtype.kind in 'fiu'
             and np.isfinite(array).all(), 'Require finite numeric '+label)
    return array.astype(np.float64)


def _linear_rgb(value, label='linear RGB'):
    array = _numeric_vector(value, 3, label)
    _require(array.min() >= 0 and array.max() <= 1, label+' must be within0..1')
    return array


def _encode(linear):
    return np.where(linear <= .0031308, linear*12.92, 1.055*linear**(1/2.4)-.055)


def allocate_counts(weights, total, minimum=0, max_total=10000):
    """Reserve each minimum, then allocate largest remainders with input-order ties."""
    _require(isinstance(weights, (list, tuple, np.ndarray))
             and (not isinstance(weights, np.ndarray) or weights.ndim == 1)
             and 1 <= len(weights) <= MAX_GROUPS,
             'Require one to32 positive group weights')
    values = _numeric_vector(weights, len(weights), 'group weights')
    _require(values.min() > 0, 'Group weights must be strictly positive')
    _require(type(total) is int and type(minimum) is int and type(max_total) is int
             and 0 <= total <= max_total <= 10000 and 0 <= minimum <= max_total,
             'Invalid bounded integer count allocation')
    _require(total >= len(values)*minimum, 'Total cannot supply every group minimum')
    _require(total != 0 or minimum == 0, 'Zero total requires zero minimum')
    remaining = total-len(values)*minimum
    # Scaling before summing avoids overflow for finite but very large area weights.
    relative = values/values.max()
    quotas = relative*(remaining/math.fsum(map(float, relative)))
    base = np.floor(quotas).astype(np.int64)
    extra = remaining-int(base.sum())
    _require(0 <= extra <= len(values), 'Numerically invalid largest-remainder allocation')
    order = sorted(range(len(values)), key=lambda index: (-(quotas[index]-base[index]), index))
    for index in order[:extra]:
        base[index] += 1
    result = [int(value)+minimum for value in base]
    _require(sum(result) == total, 'Allocated count differs from requested total')
    return result


def unique_vertex_sources(vertex_count, loop_vertex_ids, loop_material_ids, loop_uvs, textured_slots):
    """Return first loops only when no material or textured UV seam is collapsed."""
    _require(type(vertex_count) is int and 1 <= vertex_count <= MAX_VERTICES,
             'Vertex count exceeds1..100000')
    _require(isinstance(loop_vertex_ids, np.ndarray) and loop_vertex_ids.ndim == 1
             and 1 <= len(loop_vertex_ids) <= MAX_LOOPS and loop_vertex_ids.dtype.kind in 'iu',
             'Require bounded integer loop vertex IDs')
    count = len(loop_vertex_ids)
    _require(isinstance(loop_material_ids, np.ndarray) and loop_material_ids.shape == (count,)
             and loop_material_ids.dtype.kind in 'iu', 'Require integer material ownership for every loop')
    _require(loop_vertex_ids.min() >= 0 and loop_vertex_ids.max() < vertex_count
             and loop_material_ids.min() >= 0 and loop_material_ids.max() < MAX_GROUPS,
             'Loop vertex or material IDs exceed bounds')
    _require(isinstance(loop_uvs, np.ndarray) and loop_uvs.shape == (count, 2)
             and loop_uvs.dtype in (np.float32, np.float64) and np.isfinite(loop_uvs).all()
             and np.max(abs(loop_uvs)) <= 1e6, 'Require bounded finite loop UVs')
    _require(isinstance(textured_slots, (list, tuple, set, frozenset, np.ndarray))
             and (not isinstance(textured_slots, np.ndarray) or textured_slots.ndim == 1)
             and len(textured_slots) <= MAX_GROUPS, 'Invalid textured material slots')
    slots = list(textured_slots)
    _require(all(isinstance(x, (int, np.integer)) and not isinstance(x, (bool, np.bool_))
                 and 0 <= x < MAX_GROUPS for x in slots) and len(set(map(int, slots))) == len(slots),
             'Textured material slots must be unique bounded integers')
    first = np.full(vertex_count, count, dtype=np.int32)
    np.minimum.at(first, loop_vertex_ids, np.arange(count, dtype=np.int32))
    missing = np.flatnonzero(first == count)
    _require(not len(missing), 'Unreferenced source vertices: '+str(missing[:8].tolist()))
    materials = loop_material_ids[first]
    disagreement = np.flatnonzero(loop_material_ids != materials[loop_vertex_ids])
    if len(disagreement):
        index = int(disagreement[0]); vertex = int(loop_vertex_ids[index])
        raise ValueError('Vertex '+str(vertex)+' is shared by differing material slots')
    textured = np.zeros(MAX_GROUPS, dtype=bool); textured[slots] = True
    selected = textured[loop_material_ids]
    if selected.any():
        # Bit comparison rejects even signed-zero seams without creating a full image copy.
        bits = np.ascontiguousarray(loop_uvs).view(np.uint32 if loop_uvs.dtype == np.float32 else np.uint64)
        different = np.any(bits != bits[first[loop_vertex_ids]], axis=1)
        mismatch = np.flatnonzero(selected & different)
        if len(mismatch):
            index = int(mismatch[0]); vertex = int(loop_vertex_ids[index])
            raise ValueError('Textured vertex '+str(vertex)+' has differing exact loop UVs')
    counts = np.bincount(materials.astype(np.int32), minlength=MAX_GROUPS)
    report = {'policy': OWNERSHIP_POLICY, 'vertex_count': vertex_count, 'loop_count': count,
              'textured_slots': sorted(map(int, slots)), 'all_vertices_referenced': True,
              'material_vertex_counts': [{'slot': slot, 'vertices': int(value), 'textured': bool(textured[slot])}
                                         for slot, value in enumerate(counts) if value],
              'first_loop_scope': 'Safe only after exact material ownership and byte-equal textured UVs; constant UV differences are ignored.'}
    return first, report


def selection_rgb(pixels, uvs, *, factor=(1, 1, 1)):
    """Legacy nearest REPEAT selection in encoded sRGB, not exact-root filtering.

    Pixel rows and UVs share the adapter's Blender bottom-up convention. Pixels
    remain borrowed; only query-sized colors are copied or converted.
    """
    _require(isinstance(pixels, np.ndarray) and pixels.ndim == 3 and pixels.shape[2] in (3, 4)
             and pixels.dtype in (np.float32, np.float64), 'Require RGB/RGBA floating-point image rows')
    height, width = pixels.shape[:2]
    _require(0 < height*width <= MAX_PIXELS, 'Selection image exceeds16Mi-pixels')
    _require(isinstance(uvs, np.ndarray) and uvs.ndim == 2 and uvs.shape[1:] == (2,)
             and len(uvs) <= MAX_VERTICES and uvs.dtype in (np.float32, np.float64)
             and np.isfinite(uvs).all() and (not len(uvs) or np.max(abs(uvs)) <= 1e6),
             'Require at most100000 finite selection UVs')
    linear_factor = _linear_rgb(factor, 'linear material factor')
    # The temporary validation masks cover64 rows, never another complete4K image.
    for start in range(0, height, 64):
        block = pixels[start:start+64, :, :3]
        _require(np.isfinite(block).all() and block.min() >= 0 and block.max() <= 1,
                 'Selection image requires finite normalized encoded RGB')
    xx = np.minimum(width-1, (np.mod(uvs[:, 0], 1)*width).astype(int))
    yy = np.minimum(height-1, (np.mod(uvs[:, 1], 1)*height).astype(int))
    colors = pixels[yy, xx, :3].copy()
    if np.array_equal(linear_factor, [1, 1, 1]):
        return colors
    encoded = colors.astype(np.float64)
    linear = np.where(encoded <= .04045, encoded/12.92, ((encoded+.055)/1.055)**2.4)
    return _encode(linear*linear_factor).astype(pixels.dtype)


def constant_selection_rgb(linear_rgb, count):
    _require(type(count) is int and 0 <= count <= MAX_VERTICES, 'Invalid bounded constant selection count')
    encoded = _encode(_linear_rgb(linear_rgb)).astype(np.float32)
    return np.tile(encoded, (count, 1))


class ConstantRootSampler:
    """Constant material roots use linear RGB, with the same root-address contract."""

    def __init__(self, linear_rgb, triangle_count):
        _require(type(triangle_count) is int and 1 <= triangle_count <= 100000,
                 'Constant root sampler requires1..100000 triangles')
        self.rgb = tuple(map(float, _linear_rgb(linear_rgb)))
        self.triangle_count = triangle_count

    def __call__(self, triangle, barycentric):
        if type(triangle) is not int or not 0 <= triangle < self.triangle_count:
            raise ValueError('Root triangle is outside the prepared sampler')
        bary = np.asarray(barycentric, dtype=np.float64)
        if (bary.shape != (3,) or not np.isfinite(bary).all() or (bary < 0).any()
                or abs(float(bary.sum())-1) > 1e-6):
            raise ValueError('Root sampling requires normalized barycentric coordinates')
        return self.rgb
