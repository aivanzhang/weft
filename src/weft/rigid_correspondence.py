"""Separate rigid placement error from unsupported changes in metric scale."""

import numpy as np


def fit_rigid_points(reference, observed):
    """Fit a proper rigid transform; report, but never apply, a scale hypothesis."""
    source = np.asarray(reference, dtype=float)
    target = np.asarray(observed, dtype=float)
    if (source.ndim != 2 or source.shape[1:] != (3,) or target.shape != source.shape
            or not 4 <= len(source) <= 1000 or not np.isfinite(source).all()
            or not np.isfinite(target).all() or max(np.abs(source).max(), np.abs(target).max()) > 1e4):
        raise ValueError('Rigid fitting requires 4–1000 finite paired 3D points within 10 km')
    source_center, target_center = source.mean(axis=0), target.mean(axis=0)
    a, b = source-source_center, target-target_center
    singular = [np.linalg.svd(points, compute_uv=False) for points in (a, b)]
    if any(values[1] <= max(1e-10, values[0]*1e-10) for values in singular):
        raise ValueError('Rigid orientation requires non-collinear correspondences')
    left, _, right = np.linalg.svd(a.T @ b)
    handedness = np.eye(3)
    # A reflection can lower residuals but cannot represent a physical rigid pose.
    handedness[2, 2] = np.linalg.det(right.T @ left.T)
    rotation = right.T @ handedness @ left.T
    translation = target_center-source_center @ rotation.T
    predicted = source @ rotation.T+translation
    errors = np.linalg.norm(predicted-target, axis=1)
    scale = float(np.sum((a @ rotation.T)*b)/np.sum(a*a))
    return {
        'rotation': rotation.tolist(), 'translation': translation.tolist(),
        'predicted_points': predicted.tolist(), 'residuals': errors.tolist(),
        'rmse': float(np.sqrt(np.mean(errors**2))), 'maximum_error': float(errors.max()),
        'scale_hypothesis_not_applied': scale,
        'observed_plane_rms': float(singular[1][-1]/np.sqrt(len(target))),
        'count': len(source), 'scale_applied': 1.,
    }
