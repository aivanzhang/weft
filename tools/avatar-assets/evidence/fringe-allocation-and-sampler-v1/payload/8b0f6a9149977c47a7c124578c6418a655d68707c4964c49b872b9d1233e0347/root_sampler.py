"""Exact LOD0 sampler snapshot; no GPU equivalence claim."""

import numpy as np

def repeat_root_colors(pixels, uv, factor):
    """glTF normalized texel centers with REPEAT, linear LOD0 and per-texel sRGB decode."""
    size = np.array([pixels.shape[1], pixels.shape[0]])
    coordinates = uv*size-.5
    lower = np.floor(coordinates).astype(np.int64)
    fraction = coordinates-lower
    lower %= size
    upper = (lower+1)%size
    texels = pixels[np.stack((lower[:, 1], lower[:, 1], upper[:, 1], upper[:, 1]), axis=1),
                    np.stack((lower[:, 0], upper[:, 0], lower[:, 0], upper[:, 0]), axis=1), :3].astype(float)/255
    linear = np.where(texels <= .04045, texels/12.92, ((texels+.055)/1.055)**2.4)
    u, v = fraction.T
    blend = np.column_stack(((1-u)*(1-v), u*(1-v), (1-u)*v, u*v))
    return np.sum(linear*blend[:, :, None], axis=1)*factor
