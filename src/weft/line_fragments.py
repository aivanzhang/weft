"""Join compatible vertical-edge fragments without treating duplicate pieces as evidence."""

import numpy as np


def merge_vertical_fragments(segments):
    """Fit grouped endpoints; reject non-straight groups instead of keeping easier pieces."""
    lines = np.asarray(segments, dtype=float)
    if (lines.ndim != 2 or lines.shape[1:] != (4,) or not 1 <= len(lines) <= 128
            or not np.isfinite(lines).all() or np.abs(lines).max() > 1e5):
        raise ValueError('Fragment grouping requires 1–128 bounded finite pixel segments')
    points = lines.reshape(-1, 2, 2)
    deltas = points[:, 1]-points[:, 0]
    lengths = np.linalg.norm(deltas, axis=1)
    if np.any(lengths < 1.) or np.any(np.abs(deltas[:, 0]) > .6*np.abs(deltas[:, 1])):
        raise ValueError('Fragments must be nondegenerate near-vertical image segments')
    slopes = deltas[:, 0]/deltas[:, 1]
    intercepts = points[:, 0, 0]-slopes*points[:, 0, 1]
    bottom, top = points[:, :, 1].min(axis=1), points[:, :, 1].max(axis=1)
    parent = list(range(len(lines)))

    def find(index):
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    links = []
    for first in range(len(lines)):
        for last in range(first+1, len(lines)):
            angle = abs(np.degrees(np.arctan(slopes[first])-np.arctan(slopes[last])))
            low, high = max(bottom[first], bottom[last]), min(top[first], top[last])
            gap = max(0., low-high)
            y = (low+high)/2
            separation = abs((slopes[first]-slopes[last])*y+intercepts[first]-intercepts[last])
            if angle <= 3. and gap <= 80. and separation <= 3.:
                parent[find(last)] = find(first)
                links.append({'indices': [first, last], 'gap_px': float(gap),
                    'separation_px': float(separation), 'angle_degrees': float(angle)})
    groups = {}
    for index in range(len(lines)):
        groups.setdefault(find(index), []).append(index)
    results = []
    for indices in groups.values():
        observed = points[indices].reshape(-1, 2)
        weights = np.repeat(lengths[indices]/2, 2)
        center = np.average(observed, axis=0, weights=weights)
        _, _, vectors = np.linalg.svd((observed-center)*np.sqrt(weights[:, None]), full_matrices=False)
        direction = vectors[0]
        if direction[1] < 0:
            direction = -direction
        along = (observed-center) @ direction
        residuals = np.abs((observed-center) @ np.array([-direction[1], direction[0]]))
        endpoints = center+np.array([along.min(), along.max()])[:, None]*direction
        results.append({'input_indices': indices, 'pixels': endpoints.tolist(),
            'length_px': float(along.max()-along.min()), 'maximum_endpoint_residual_px': float(residuals.max()),
            'endpoint_residuals_px': residuals.tolist(), 'status': 'straight_group' if residuals.max() <= 2. else 'nonstraight_group_rejected'})
    return {'groups': results, 'links': links, 'input_segments': len(lines),
        'contract': {'maximum_join_gap_px': 80., 'maximum_join_separation_px': 3.,
            'maximum_join_angle_degrees': 3., 'maximum_endpoint_residual_px': 2.,
            'scope': 'Endpoint-supported line hypothesis; bridged pixels are not observed edge samples.'}}
