"""Bounded triangle-change metrics and a conservative authoring regression gate."""

import numpy as np


MAX_VERTICES = 100000
MAX_TRIANGLES = 100000


class SurfaceChangeError(ValueError):
    def __init__(self, message, report):
        super().__init__(message)
        self.report = report


def _matrix(value, limit, name, kinds):
    if isinstance(value, np.ndarray):
        if value.ndim != 2 or value.shape[1] != 3 or not 1 <= len(value) <= limit:
            raise ValueError(f'Invalid or oversized {name} shape')
        result = value
    else:
        if not isinstance(value, (list, tuple)) or not 1 <= len(value) <= limit:
            raise ValueError(f'Invalid or oversized {name} rows')
        for row in value:
            if not isinstance(row, (list, tuple, np.ndarray)) or (
                    isinstance(row, np.ndarray) and row.ndim != 1) or len(row) != 3:
                raise ValueError(f'Invalid {name} row')
            if any(not isinstance(cell, (int, float, np.integer, np.floating)) or
                   isinstance(cell, (bool, np.bool_)) for cell in row):
                raise ValueError(f'Invalid {name} scalar')
        result = np.asarray(value)
    if result.dtype.kind not in kinds:
        raise ValueError(f'Invalid {name} numeric type')
    return result


def _lengths(vectors):
    return np.sqrt(np.einsum('ij,ij->i', vectors, vectors))


def _measure_triangles(points, triangles):
    first = points[triangles[:, 0]]
    normals = np.cross(points[triangles[:, 1]]-first, points[triangles[:, 2]]-first)
    return _lengths(normals)*.5, normals


def inspect_surface_change(original_positions, refined_positions, triangle_indices):
    """Compare unchanged triangle indices in one coordinate basis; no repair verdict."""
    original = _matrix(original_positions, MAX_VERTICES, 'original positions', 'fiu')
    refined = _matrix(refined_positions, MAX_VERTICES, 'refined positions', 'fiu')
    triangles = _matrix(triangle_indices, MAX_TRIANGLES, 'triangle indices', 'iu')
    if original.shape != refined.shape:
        raise ValueError('Original and refined vertex counts must match')
    if not np.isfinite(original).all() or not np.isfinite(refined).all():
        raise ValueError('Surface positions must be finite')
    if triangles.min() < 0 or triangles.max() >= len(original):
        raise ValueError('Surface triangle index is out of range')
    try:
        with np.errstate(over='raise', invalid='raise', divide='raise'):
            original = original.astype(np.float64, copy=False)
            refined = refined.astype(np.float64, copy=False)
            triangles = triangles.astype(np.int64, copy=False)
            extent = float((original.max(axis=0)-original.min(axis=0)).max())
            epsilon = max(extent*extent*1e-14, 1e-20)
            source_area, source_normal = _measure_triangles(original, triangles)
            refined_area, refined_normal = _measure_triangles(refined, triangles)
            source_nonzero, refined_nonzero = source_area > epsilon, refined_area > epsilon
            ratios = refined_area[source_nonzero]/source_area[source_nonzero]
            comparable = source_nonzero & refined_nonzero
            source_unit = source_normal[comparable]/(2*source_area[comparable, None])
            refined_unit = refined_normal[comparable]/(2*refined_area[comparable, None])
            normal_dot = np.einsum('ij,ij->i', source_unit, refined_unit)
            shifts = _lengths(refined-original)
            return {
                'schema_version': 1, 'vertices': len(original), 'triangles': len(triangles),
                'area_epsilon_m2': epsilon,
                'area_epsilon_policy': 'max(largest original extent squared × 1e-14, 1e-20 m²), fixed for both surfaces',
                'source_degenerate_count': int(np.count_nonzero(~source_nonzero)),
                'refined_degenerate_count': int(np.count_nonzero(~refined_nonzero)),
                'new_degenerate_count': int(np.count_nonzero(source_nonzero & ~refined_nonzero)),
                'source_degenerate_remaining_count': int(np.count_nonzero(~source_nonzero & ~refined_nonzero)),
                'source_degenerate_now_nondegenerate_count': int(np.count_nonzero(~source_nonzero & refined_nonzero)),
                'nonzero_source_count': int(np.count_nonzero(source_nonzero)),
                'min_area_ratio': float(ratios.min()) if len(ratios) else None,
                'median_area_ratio': float(np.median(ratios)) if len(ratios) else None,
                'area_ratio_below_0_01_count': int(np.count_nonzero(ratios < .01)),
                'area_ratio_below_0_001_count': int(np.count_nonzero(ratios < .001)),
                'comparable_face_normal_count': int(np.count_nonzero(comparable)),
                'face_normals_changed_over_90_degrees_count': int(np.count_nonzero(normal_dot < 0)),
                'vertex_max_shift_m': float(shifts.max()),
                'vertex_rms_shift_m': float(np.sqrt(np.mean(shifts*shifts))),
                'source_degeneracy_repair_claimed': False,
                'self_intersections_checked': False, 'identity_checked': False,
                'limits': {'vertices': MAX_VERTICES, 'triangles': MAX_TRIANGLES},
                'scope': 'Numerical geometry changes only; not visual quality, identity, self-intersection or physical validity.',
            }
    except (FloatingPointError, OverflowError) as error:
        raise ValueError('Surface change exceeds the finite numerical range') from error


def enforce_surface_change(original_positions, refined_positions, triangle_indices):
    """Apply a conservative regression gate, not acceptance of the source or result."""
    report = inspect_surface_change(original_positions, refined_positions, triangle_indices)
    allowed = min(16, max(1, int(report['nonzero_source_count']*.001)))
    failures = []
    if report['new_degenerate_count']:
        failures.append(f"{report['new_degenerate_count']} newly degenerate triangles")
    if report['area_ratio_below_0_01_count'] > allowed:
        failures.append(f"{report['area_ratio_below_0_01_count']} severely collapsed triangles exceed allowance {allowed}")
    report['regression_gate'] = {
        'status': 'failed' if failures else ('passed_with_source_degeneracy' if report['source_degenerate_count'] else 'passed'),
        'allowed_new_degenerate_count': 0, 'allowed_area_ratio_below_0_01_count': allowed,
        'threshold_policy': 'min(16, max(1, int(nonzero_source_count × 0.001)))',
        'quality_proof': False, 'failures': failures,
        'scope': 'Conservative authoring regression gate; pre-existing source degeneracy is reported, not accepted as repaired.',
    }
    if failures:
        raise SurfaceChangeError('Surface refinement regression: '+'; '.join(failures), report)
    return report
