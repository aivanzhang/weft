"""Estimate a shared 3D direction from image-ray planes, not gravity by itself."""

import math

import numpy as np


def estimate_parallel_axis(normals, weights, *, tolerance_degrees=.5):
    """Retain bounded plane-pair hypotheses and fit their strongest consensus."""
    planes = np.asarray(normals, dtype=float)
    importance = np.asarray(weights, dtype=float)
    if (planes.ndim != 2 or planes.shape[1:] != (3,) or not 2 <= len(planes) <= 128
            or importance.shape != (len(planes),) or not np.isfinite(planes).all()
            or not np.isfinite(importance).all() or np.any(importance <= 0)
            or np.max(importance) > 1e6 or np.max(np.abs(planes)) > 1e6):
        raise ValueError('Parallel-axis fitting needs 2–128 finite ray planes with bounded positive weights')
    if (type(tolerance_degrees) not in (int, float) or not math.isfinite(tolerance_degrees)
            or not 0 < tolerance_degrees <= 2):
        raise ValueError('Parallel-axis tolerance must be within (0, 2] degrees')
    lengths = np.linalg.norm(planes, axis=1)
    if np.any(lengths < 1e-10):
        raise ValueError('Ray-plane normals must be nonzero')
    planes = planes/lengths[:, None]
    first, last = np.triu_indices(len(planes), 1)
    directions = np.cross(planes[first], planes[last])
    separation = np.linalg.norm(directions, axis=1)
    usable = separation >= .01
    if not np.any(usable):
        raise ValueError('Distinct ray planes are required to constrain a parallel axis')
    directions = directions[usable]/separation[usable, None]
    pairs = np.column_stack([first[usable], last[usable]])
    residual = np.abs(planes @ directions.T)
    support = residual <= math.sin(math.radians(tolerance_degrees))
    supported_weight = importance @ support
    costs = (importance @ (residual**2*support))/supported_weight
    selected = int(np.lexsort((costs, -supported_weight))[0])
    initial = np.flatnonzero(support[:, selected])
    _, singular, vectors = np.linalg.svd(planes[initial]*np.sqrt(importance[initial, None]))
    if len(singular) < 2 or singular[1] <= 1e-10:
        raise ValueError('Consensus does not constrain a unique axis')
    axis = vectors[-1]
    # Parallel lines identify an unsigned axis; use one deterministic sign only.
    if axis[np.argmax(np.abs(axis))] < 0:
        axis = -axis
    errors = np.degrees(np.arcsin(np.clip(np.abs(planes @ axis), 0., 1.)))
    retained = np.flatnonzero(errors <= tolerance_degrees)
    return {
        'axis': axis.tolist(), 'angular_residuals_degrees': errors.tolist(),
        'initial_inlier_indices': initial.tolist(), 'inlier_indices': retained.tolist(),
        'supported_weight_fraction': float(importance[retained].sum()/importance.sum()),
        'fit_singular_values': singular.tolist(), 'selected_hypothesis': selected,
        'tolerance_degrees': tolerance_degrees, 'unsigned': True,
        'hypotheses': [{'source_indices': pair.tolist(), 'axis': direction.tolist(),
            'supported_weight': float(weight), 'mean_squared_plane_residual': float(cost),
            'inlier_count': int(count)} for pair, direction, weight, cost, count in
            zip(pairs, directions, supported_weight, costs, support.sum(axis=0), strict=True)],
    }
