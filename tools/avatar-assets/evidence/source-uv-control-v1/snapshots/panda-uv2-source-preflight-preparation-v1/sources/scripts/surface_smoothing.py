"""Bounded surface position refinement; authored geometry, not recovered detail."""

from time import perf_counter

import numpy as np

try:
    from surface_smoothing_config import validate_surface_smoothing
except ModuleNotFoundError as error:
    if error.name != 'surface_smoothing_config':
        raise
    from avatar_assets.surface_smoothing_config import validate_surface_smoothing


MAX_VERTICES = 100000
MAX_EDGES = 300000


def _matrix(value, columns, limit, name, kinds):
    if isinstance(value, np.ndarray):
        if value.ndim != 2 or value.shape[1] != columns or len(value) > limit:
            raise ValueError(f'Invalid or oversized smoothing {name}')
        result = value
    else:
        if not isinstance(value, (list, tuple)) or len(value) > limit:
            raise ValueError(f'Invalid or oversized smoothing {name}')
        for row in value:
            if not isinstance(row, (list, tuple, np.ndarray)) or (
                    isinstance(row, np.ndarray) and row.ndim != 1) or len(row) != columns:
                raise ValueError(f'Invalid smoothing {name} row')
            if any(not isinstance(cell, (int, float, np.integer, np.floating)) or
                   isinstance(cell, (bool, np.bool_)) for cell in row):
                raise ValueError(f'Invalid smoothing {name} scalar')
        result = np.asarray(value) if value else np.empty((0, columns), dtype=np.int64)
    if result.dtype.kind not in kinds:
        raise ValueError(f'Invalid smoothing {name} numeric type')
    return result


def _lengths(vectors):
    return np.sqrt(np.einsum('ij,ij->i', vectors, vectors))


def _clip(vectors, limit):
    lengths = _lengths(vectors)
    mask = lengths > limit
    factors = np.ones(len(lengths))
    np.divide(limit, lengths, out=factors, where=mask)
    vectors *= factors[:, None]
    return int(np.count_nonzero(mask))


def _smooth(original, edges, config):
    original = original.astype(np.float64, copy=False)
    low, high = original.min(axis=0), original.max(axis=0)
    span = high-low
    scale = float(span.max())
    if scale <= 0 or np.any(span <= max(scale*1e-12, 1e-12)):
        raise ValueError('Surface smoothing requires nondegenerate three-dimensional bounds')
    welded, inverse = np.unique(original, axis=0, return_inverse=True)
    paired = np.sort(inverse[edges.astype(np.int64, copy=False)], axis=1)
    paired = np.unique(paired[paired[:, 0] != paired[:, 1]], axis=0)
    weights = np.ones(len(welded))
    normalized = (welded-low)/span
    for region in config['exclusions']:
        distance = _lengths((normalized-region['center'])/region['radius'])
        feather = np.clip((distance-1)/config['feather'], 0, 1)
        weights = np.minimum(weights, feather*feather*(3-2*feather))
    source, target = np.concatenate((paired[:, 0], paired[:, 1])), np.concatenate((paired[:, 1], paired[:, 0]))
    degree = np.bincount(source, minlength=len(welded))
    connected = np.flatnonzero(degree)
    neighbors = target[np.argsort(source, kind='stable')]
    starts = np.concatenate(([0], np.cumsum(degree[connected])[:-1])) if len(connected) else np.array([], dtype=int)
    positions = welded.copy()
    limit = config['max_displacement_ratio']*scale
    executed = config['iterations'] if np.any(weights[connected] > 0) else 0
    method = config.get('method', 'taubin')
    taubin = method == 'taubin'
    coefficients = (config['strength'], -config['strength']*1.02) if taubin else (config['strength'],)
    maximum_step, clipped = 0., 0
    for _ in range(executed):
        for coefficient in coefficients:
            average = np.add.reduceat(positions[neighbors], starts, axis=0)/degree[connected, None]
            delta = (average-positions[connected])*(coefficient*weights[connected, None])
            clipped += _clip(delta, limit)
            candidate = positions.copy()
            candidate[connected] += delta
            displacement = candidate-welded
            clipped += _clip(displacement, limit)
            candidate = welded+displacement
            candidate[weights == 0] = welded[weights == 0]
            maximum_step = max(maximum_step, float(_lengths(candidate-positions).max()))
            positions = candidate
    result = original.copy()
    movable = weights[inverse] > 0
    # Preserve exact protected values and give every UV-seam copy the same delta.
    result[movable] += (positions-welded)[inverse[movable]]
    shifts = _lengths(result-original)
    return result, {
        'method': 'Exact-position welding, fixed spatial feather weights, '+
                  ('paired Taubin steps' if taubin else 'positive Laplacian steps'),
        'method_id': method, 'steps_per_iteration': len(coefficients),
        'authored_not_recovered': True, 'color_mask_used': False,
        'volume_preservation_intended': taubin, 'volume_preservation_verified': False,
        'shrinkage_intended': not taubin,
        'volume_scope': ('Paired steps reduce shrinkage; preserved volume is not measured or guaranteed.' if taubin else
                         'Positive Laplacian fairing intentionally permits shrinkage within the displacement cap; volume is not preserved or verified.'),
        'input_vertices': len(original), 'welded_vertices': len(welded),
        'coincident_vertices_merged': len(original)-len(welded),
        'input_edges': len(edges), 'welded_edges': len(paired),
        'excluded_vertices': int(np.count_nonzero(weights[inverse] == 0)),
        'excluded_welded_vertices': int(np.count_nonzero(weights == 0)),
        'feathered_vertices': int(np.count_nonzero((weights[inverse] > 0) & (weights[inverse] < 1))),
        'isolated_welded_vertices': int(np.count_nonzero(degree == 0)),
        'iterations_requested': config['iterations'], 'iterations_executed': executed,
        'positive_step_strength': config['strength'],
        'negative_step_strength': -config['strength']*1.02 if taubin else None,
        'displacement_limit_m': limit, 'max_displacement_m': float(shifts.max()),
        'rms_displacement_m': float(np.sqrt(np.mean(shifts*shifts))),
        'max_step_displacement_m': maximum_step, 'clipped_vertex_step_events': clipped,
        'bounds_m': {'min': low.tolist(), 'span': span.tolist()},
        'limits': {'vertices': MAX_VERTICES, 'edges': MAX_EDGES, 'iterations': 20,
                   'max_displacement_ratio': config['max_displacement_ratio']},
        'geometry_scope': 'Only positions are returned; topology, UVs and skin remain the caller\'s responsibility.',
    }


def refine_positions(vertices, edges, config):
    """Return new world-meter positions and measured changes without mutating inputs."""
    started = perf_counter()
    config = validate_surface_smoothing(config)
    vertices = _matrix(vertices, 3, MAX_VERTICES, 'vertices', 'fiu')
    edges = _matrix(edges, 2, MAX_EDGES, 'edges', 'iu')
    if not len(vertices) or not np.isfinite(vertices).all():
        raise ValueError('Surface smoothing requires nonempty finite vertices')
    if len(edges) and (edges.min() < 0 or edges.max() >= len(vertices)):
        raise ValueError('Surface smoothing edge endpoint is out of range')
    try:
        with np.errstate(over='raise', invalid='raise', divide='raise'):
            positions, report = _smooth(vertices, edges, config)
    except (FloatingPointError, OverflowError) as error:
        raise ValueError('Surface smoothing exceeds the finite numerical range') from error
    report['elapsed_seconds'] = perf_counter()-started
    return positions, report
