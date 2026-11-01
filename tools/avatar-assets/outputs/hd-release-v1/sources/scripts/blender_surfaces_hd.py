"""Authored HD surface textures: native detail, bounded slabs, no reference recovery.

UVs run left-to-right and bottom-to-top. Iris geometry needs disk UVs centered at
(0.5, 0.5); its pupil and outer limbal ring are color, not corneal geometry.
"""

import math
from numbers import Integral, Real
import time

import numpy as np

ROW_BATCH = 128
MAX_SECONDS = 20.0
ALGORITHM = "native_hd_surfaces_v1"
PATTERNS = frozenset(("fur", "scales", "iris", "brushed"))
RESOLUTIONS = frozenset((1024, 2048))


def _validated(color, settings):
    if not isinstance(settings, dict) or set(settings) != {"pattern", "resolution", "strength", "seed"}:
        raise ValueError("HD surfaces need exactly pattern, resolution, strength and seed")
    pattern, size, strength, seed = (settings[key] for key in ("pattern", "resolution", "strength", "seed"))
    if not isinstance(pattern, str) or pattern not in PATTERNS:
        raise ValueError("Unsupported HD surface pattern")
    if isinstance(size, bool) or not isinstance(size, Integral) or size not in RESOLUTIONS:
        raise ValueError("HD surface resolution must be 1024 or 2048")
    if isinstance(strength, bool) or not isinstance(strength, Real) or not math.isfinite(strength) or not 0 <= strength <= 1:
        raise ValueError("HD surface strength must be finite in [0, 1]")
    if isinstance(seed, bool) or not isinstance(seed, Integral) or not 0 <= seed <= 2**31-1:
        raise ValueError("HD surface seed must be an integer in [0, 2^31-1]")
    if not isinstance(color, (list, tuple, np.ndarray)) or len(color) != 4:
        raise ValueError("HD surface color must be four sRGB/alpha components")
    if any(isinstance(value, (bool, np.bool_)) or not isinstance(value, Real) or not math.isfinite(value)
           or not 0 <= value <= 1 for value in color):
        raise ValueError("HD surface color components must be finite in [0, 1]")
    return np.asarray(color, dtype=np.float32), pattern, int(size), float(strength), int(seed)


def _hash(x, y, seed):
    # Integer overflow is deliberate: lattice values do not depend on tile order.
    value = np.asarray(x, dtype=np.uint32) * np.uint32(0x8DA6B343)
    value = value + np.asarray(y, dtype=np.uint32) * np.uint32(0xD8163841)
    value ^= np.uint32(seed & 0xFFFFFFFF)
    value ^= value >> np.uint32(16)
    value *= np.uint32(0x7FEB352D)
    value ^= value >> np.uint32(15)
    value *= np.uint32(0x846CA68B)
    value ^= value >> np.uint32(16)
    return (value >> np.uint32(8)).astype(np.float32) * np.float32(1 / 16777216)


def _noise(x, y, seed, period):
    ix, iy = np.floor(x).astype(np.int32), np.floor(y).astype(np.int32)
    fx, fy = x - ix, y - iy
    fx = (fx*fx*(3-2*fx)).astype(np.float32)
    fy = (fy*fy*(3-2*fy)).astype(np.float32)
    x0, x1 = ix % period[0], (ix+1) % period[0]
    y0, y1 = iy % period[1], (iy+1) % period[1]
    bottom = _hash(x0, y0, seed)
    bottom += (_hash(x1, y0, seed)-bottom)*fx
    top = _hash(x0, y1, seed)
    top += (_hash(x1, y1, seed)-top)*fx
    bottom += (top-bottom)*fy
    return bottom


def _smoothstep(low, high, value):
    value = np.clip((value-low)/(high-low), 0, 1)
    return value*value*(3-2*value)


def _fur(u, v, size, seed):
    flow = _noise(u*9, v*9, seed+13, (9, 9)) - .5
    cross_flow = _noise(u*17, v*11, seed+27, (17, 11)) - .5
    warp_u = u + flow*.015 + cross_flow*.004
    warp_v = v + flow*.007
    bundles = _noise(warp_u*103, warp_v*29, seed+37, (103, 29)) - .5
    fibers = _noise(warp_u*431, warp_v*113, seed+47, (431, 113)) - .5
    # The last octave is sampled natively; 2048 adds finer undercoat, not enlarged 512 pixels.
    fine_x, fine_y = int(size*.43), int(size*.17)
    fine = _noise(warp_u*fine_x, warp_v*fine_y, seed+61, (fine_x, fine_y)) - .5
    height = .5 + bundles*.14 + fibers*.34 + fine*.12
    shade = 1 + bundles*.16 + fibers*.23 + fine*.08
    return height.astype(np.float32), shade.astype(np.float32), None


def _scales(u, v, size, seed):
    cells = 39
    warp = _noise(u*7, v*7, seed+7, (7, 7)) - .5
    x, y = (u + warp*.009)*cells, (v + warp*.005)*cells
    ix, iy = np.floor(x).astype(np.int32), np.floor(y).astype(np.int32)
    shape = np.broadcast_shapes(x.shape, y.shape)
    pebbles = np.zeros(shape, dtype=np.float32)
    variation = np.zeros(shape, dtype=np.float32)
    for oy in (-1, 0, 1):
        for ox in (-1, 0, 1):
            cell_x, cell_y = ix+ox, iy+oy
            sample_x, sample_y = cell_x % cells, cell_y % cells
            jitter_x = _hash(sample_x, sample_y, seed+101)
            jitter_y = _hash(sample_x, sample_y, seed+211)
            dx = x-(cell_x+.12+jitter_x*.76)
            dy = y-(cell_y+.12+jitter_y*.76)
            pigment = _hash(sample_x, sample_y, seed+307)
            radius_x = .6 + pigment*.18
            radius_y = .55 + _hash(sample_x, sample_y, seed+359)*.17
            distance = ((dx/radius_x)**2+(dy/radius_y)**2).astype(np.float32)
            # Compact rounded domes avoid Voronoi's flat polygon plates and hard straight seams.
            dome = np.maximum(1-distance, 0)**np.float32(1.5)
            dome *= .83 + jitter_x*.17
            variation = np.where(dome > pebbles, pigment, variation)
            pebbles = np.maximum(pebbles, dome)
    pores_n = int(size*.38)
    pores = _noise(u*pores_n, v*pores_n, seed+419, (pores_n, pores_n)) - .5
    height = .16 + pebbles*.68 + pores*.025
    shade = .88 + np.sqrt(pebbles)*.15 + (variation-.5)*.055 + pores*.025
    return height.astype(np.float32), shade.astype(np.float32), None


def _iris(u, v, size, seed):
    x, y = u-.5, v-.5
    radius = np.sqrt(x*x+y*y).astype(np.float32)
    angle = (np.arctan2(y, x)/np.float32(math.tau)+.5).astype(np.float32)
    fibers = _noise(angle*503, radius*31, seed+17, (503, 64)) - .5
    fine_n = int(size*.47)
    fine = _noise(angle*fine_n, radius*101, seed+31, (fine_n, 128)) - .5
    broken = _noise(angle*137, radius*53, seed+43, (137, 64)) - .5
    pupil = 1-_smoothstep(.125, .145, radius)
    limbal = _smoothstep(.407, .473, radius)
    outside = _smoothstep(.474, .49, radius)
    collarette = np.exp(-((radius-.205)/.041)**2).astype(np.float32)
    visible = (1-pupil)*(1-outside)
    shade = (1 + fibers*.66 + fine*.27 + broken*.19 + collarette*.21) * (1-limbal*.83)
    shade = shade*(1-pupil*.985)
    height = .5 + visible*(fibers*.12+fine*.025+broken*.035+collarette*.035)
    # Pigment variation is radial; no painted reflections stand in for a real cornea.
    warmth = collarette*.18 - limbal*.07 + broken*.035
    return height.astype(np.float32), shade.astype(np.float32), warmth.astype(np.float32)


def _brushed(u, v, size, seed):
    drift = (_noise(u*11, v*5, seed+11, (11, 5))-.5)*.0015
    grain = _noise((u+drift)*691, v*47, seed+23, (691, 47))-.5
    fine_x, fine_y = int(size*.46), int(size*.095)
    fine = _noise((u+drift)*fine_x, v*fine_y, seed+37, (fine_x, fine_y))-.5
    height = .5 + grain*.065 + fine*.03
    shade = 1 + grain*.045 + fine*.018
    return height.astype(np.float32), shade.astype(np.float32), None


def _encode_srgb(linear, destination):
    np.maximum(linear, 0, out=linear)
    np.minimum(linear, 1, out=linear)
    low = linear <= .0031308
    np.power(linear, np.float32(1/2.4), out=destination)
    destination *= np.float32(1.055)
    destination -= np.float32(.055)
    destination[low] = linear[low]*np.float32(12.92)


def surface_pixels_hd(color_srgb_rgba, settings):
    """Return sRGB RGBA and linear tangent-normal RGBA float32 arrays.

    Only 1024/2048 are accepted. Two 2048 outputs occupy 128MiB; temporary
    fields are restricted to 130-row slabs. Work checks a 20-second deadline
    between slabs; callers still own process-level memory/time enforcement.
    """
    color, pattern, size, strength, seed = _validated(color_srgb_rgba, settings)
    started = time.monotonic()
    base = np.empty((size, size, 4), dtype=np.float32)
    normals = np.empty_like(base)
    if strength == 0:
        base[:] = color
        normals[:] = [.5, .5, 1, 1]
        return base, normals
    linear = np.where(color[:3] <= .04045, color[:3]/12.92, ((color[:3]+.055)/1.055)**2.4).astype(np.float32)
    u = ((np.arange(size, dtype=np.float32)+.5)/size)[None, :]
    generator = {"fur": _fur, "scales": _scales, "iris": _iris, "brushed": _brushed}[pattern]
    normal_factor = {"fur": 3.2, "scales": 2.3, "iris": 1.65, "brushed": 1.4}[pattern]*strength*size/1024
    for start in range(0, size, ROW_BATCH):
        if time.monotonic()-started > MAX_SECONDS:
            raise RuntimeError("HD surface generation exceeded its 20-second deadline")
        end = min(start+ROW_BATCH, size)
        v = (((np.arange(start-1, end+1, dtype=np.int32) % size).astype(np.float32)+.5)/size)[:, None]
        height, shade, warmth = generator(u, v, size, seed)
        shade = 1 + (shade[1:-1]-1)*strength
        if warmth is not None:
            warmth = warmth[1:-1]*strength
        for channel in range(3):
            component = shade*linear[channel]
            if warmth is not None:
                component *= 1 + warmth*(.45 if channel == 0 else -.25 if channel == 2 else .05)
            _encode_srgb(component, base[start:end, :, channel])
        base[start:end, :, 3] = color[3]
        center = height[1:-1]
        dx = (np.roll(center, -1, axis=1)-np.roll(center, 1, axis=1))*normal_factor
        dy = (height[2:]-height[:-2])*normal_factor
        length = np.sqrt(1+dx*dx+dy*dy)
        normals[start:end, :, 0] = .5-dx/(2*length)
        normals[start:end, :, 1] = .5-dy/(2*length)
        normals[start:end, :, 2] = .5+.5/length
        normals[start:end, :, 3] = 1
    if time.monotonic()-started > MAX_SECONDS:
        raise RuntimeError("HD surface generation exceeded its 20-second deadline")
    return base, normals
