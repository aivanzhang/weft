"""Bounded stretched-edge evidence, not a visibility, quality or physics verdict."""

import numpy as np


MAX_VERTICES = 100000
MAX_EDGES = 300000
MAX_ROWS = 4


def _matrix(value, columns, limit, name, kinds):
    if isinstance(value, np.ndarray):
        if value.ndim != 2 or value.shape[1] != columns or len(value) > limit:
            raise ValueError(f'Invalid or oversized {name} shape')
        result = value
    else:
        if not isinstance(value, (list, tuple)) or len(value) > limit:
            raise ValueError(f'Invalid or oversized {name} rows')
        for row in value:
            if not isinstance(row, (list, tuple, np.ndarray)) or (
                    isinstance(row, np.ndarray) and row.ndim != 1) or len(row) != columns:
                raise ValueError(f'Invalid {name} row')
            if any(not isinstance(cell, (int, float, np.integer, np.floating)) or
                   isinstance(cell, (bool, np.bool_)) for cell in row):
                raise ValueError(f'Invalid {name} scalar')
        result = np.asarray(value) if value else np.empty((0, columns), dtype=int)
    if result.dtype.kind not in kinds:
        raise ValueError(f'Invalid {name} numeric type')
    return result


def _lengths(points, edges):
    return np.hypot.reduce(points[edges[:, 1]]-points[edges[:, 0]], axis=1)


def diagnose_stretched_edges(rest_points, posed_points, edges):
    """Return at most four edge outliers, ordered by extension then stretch ratio.

    Coordinates use Blender Z-up rest bounds; flat-Z meshes use their largest
    span as the height proxy. Camera occlusion and skin correctness are unknown.
    """
    rest = _matrix(rest_points, 3, MAX_VERTICES, 'rest points', 'fiu')
    posed = _matrix(posed_points, 3, MAX_VERTICES, 'posed points', 'fiu')
    edges = _matrix(edges, 2, MAX_EDGES, 'edges', 'iu')
    if not len(rest) or rest.shape != posed.shape:
        raise ValueError('Rest and posed points must have the same nonempty shape')
    if not np.isfinite(rest).all() or not np.isfinite(posed).all():
        raise ValueError('Deformation points must be finite')
    if len(edges) and (edges.min() < 0 or edges.max() >= len(rest)):
        raise ValueError('Edge endpoint outside the point array')
    if not len(edges):
        return []
    try:
        with np.errstate(over='raise', invalid='raise', divide='raise'):
            rest, posed = rest.astype(np.float64, copy=False), posed.astype(np.float64, copy=False)
            edges = edges.astype(np.int64, copy=False)
            low, high = rest.min(axis=0), rest.max(axis=0)
            span = high-low
            scale = float(span.max())
            if scale == 0:
                return []
            height = float(span[2]) if span[2] > scale*1e-6 else scale
            original = _lengths(rest, edges)
            deformed = _lengths(posed, edges)
            excess = deformed-original
            # Near-zero rest edges inflate ratios without useful surface evidence.
            valid = (original > max(scale*1e-6, 1e-12)) & (excess > height*.005)
            candidates = np.flatnonzero(valid)
            ratios = deformed[candidates]/original[candidates]
            candidates, ratios = candidates[ratios > 1.5], ratios[ratios > 1.5]
            order = np.lexsort((candidates, -ratios, -excess[candidates]))
            rows, seen = [], set()
            for offset in order:
                index = int(candidates[offset])
                endpoints = edges[index]
                identity = tuple(sorted(map(int, endpoints)))
                if identity in seen:
                    continue
                seen.add(identity)
                normalized = (rest[endpoints]-low)/np.where(span > 0, span, 1)
                rows.append({'endpoint_indices': endpoints.tolist(),
                             'rest_length': float(original[index]),
                             'posed_length': float(deformed[index]),
                             'stretch_ratio': float(ratios[offset]),
                             'excess_length': float(excess[index]),
                             'rest_normalized_bounds': normalized.tolist()})
                if len(rows) == MAX_ROWS:
                    break
            return rows
    except (FloatingPointError, OverflowError) as error:
        raise ValueError('Deformation exceeds the finite diagnostic numerical range') from error
