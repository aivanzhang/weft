"""Bounded UV-corner sampling for source-colored portable fur roots."""

import numpy as np


class RootColorSampler:
    """Borrow read-only pixel/UV arrays; the adapter owns their lifetime and coordinates."""

    def __init__(self, pixels, triangle_uvs, *, pixels_are_linear=False):
        if not isinstance(pixels_are_linear, bool):
            raise ValueError('Invalid pixel color space flag')
        if not isinstance(pixels, np.ndarray) or pixels.ndim != 3 or pixels.shape[2] not in (3, 4):
            raise ValueError('Root sampling requires RGB/RGBA pixel rows')
        height, width = pixels.shape[:2]
        if not 0 < height*width <= 16_777_216 or pixels.dtype not in (np.float32, np.float64):
            raise ValueError('Root texture exceeds bounded floating-point image format')
        if (not isinstance(triangle_uvs, np.ndarray) or triangle_uvs.ndim != 3
                or triangle_uvs.shape[1:] != (3, 2) or not 1 <= len(triangle_uvs) <= 100000
                or triangle_uvs.dtype not in (np.float32, np.float64)):
            raise ValueError('Root sampling requires bounded per-triangle UV corners')
        if not np.isfinite(triangle_uvs).all() or np.abs(triangle_uvs).max() > 1e6:
            raise ValueError('Invalid root UV coordinates')
        # Chunk validation avoids a full-size temporary mask for a4K RGBA texture.
        for first in range(0, height, 64):
            block = pixels[first:first+64, :, :3]
            if not np.isfinite(block).all() or block.min() < 0 or block.max() > 1:
                raise ValueError('Root texture must contain finite normalized RGB')
        self.pixels, self.uvs = pixels, triangle_uvs
        self.width, self.height, self.linear = width, height, pixels_are_linear

    def __call__(self, triangle, barycentric):
        if type(triangle) is not int or not 0 <= triangle < len(self.uvs):
            raise ValueError('Root triangle is outside the prepared sampler')
        bary = np.asarray(barycentric, dtype=np.float64)
        if (bary.shape != (3,) or not np.isfinite(bary).all() or (bary < 0).any()
                or abs(float(bary.sum())-1) > 1e-6):
            raise ValueError('Root sampling requires normalized barycentric coordinates')
        uv = bary@self.uvs[triangle]
        xy = np.mod(uv, 1)*[self.width, self.height]-.5
        low = np.floor(xy).astype(int)
        fraction = xy-low
        xx = [low[0] % self.width, (low[0]+1) % self.width]
        yy = [low[1] % self.height, (low[1]+1) % self.height]
        rgb = self.pixels[np.ix_(yy, xx)][:, :, :3].astype(np.float64)
        # Decode each texel before interpolation, matching linear-light texture filtering.
        if not self.linear:
            rgb = np.where(rgb <= .04045, rgb/12.92, ((rgb+.055)/1.055)**2.4)
        weights = np.outer([1-fraction[1], fraction[1]], [1-fraction[0], fraction[0]])
        return tuple(float(v) for v in (rgb*weights[:, :, None]).sum(axis=(0, 1)))
