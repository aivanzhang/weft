"""One-axis shape hypotheses under an explicit scaled-orthographic assumption."""

import numpy as np


def conditional_axis_scale(matrix, *, direction, center, scope):
    if scope != 'construction':
        raise ValueError('Shape inference requires explicit construction evidence')
    matrix = np.asarray(matrix, dtype=float)
    direction = np.asarray(direction, dtype=float)
    center = np.asarray(center, dtype=float)
    if (matrix.shape != (2, 4) or not np.isfinite(matrix).all() or np.max(np.abs(matrix)) > 1e8
            or direction.shape != (3,) or not np.isfinite(direction).all()
            or abs(np.linalg.norm(direction) - 1) > 1e-7
            or center.shape != (3,) or not np.isfinite(center).all() or np.max(np.abs(center)) > 1e4):
        raise ValueError('Use a bounded affine matrix, unit shape axis and finite center')
    linear = matrix[:, :3]
    singular = np.linalg.svd(linear, compute_uv=False)
    if singular[-1] <= singular[0] * 1e-8:
        raise ValueError('Camera must retain two independent image directions')
    component = linear @ direction
    axial = np.outer(component, component)
    remainder = linear @ linear.T - axial
    # Equal orthogonal camera rows require a zero traceless image Gram matrix.
    residual = np.array([remainder[0, 0] - remainder[1, 1], 2 * remainder[0, 1]])
    change = np.array([axial[0, 0] - axial[1, 1], 2 * axial[0, 1]])
    norm = float(change @ change)
    result = {'method': 'conditional-one-axis-orthographic-v1', 'scope': scope,
        'available': False, 'factor': None, 'matrix_px': None,
        'metric_geometry_verified': False, 'physical_validation': 'unverified',
        'assumptions': ['Scaled orthographic projection with square pixels and no skew or distortion.',
            'Other dimensions, object center, shape-axis direction and source-corner identities stay fixed.',
            'This chooses one camera/shape hypothesis; it does not independently measure a dimension.']}
    if norm <= (singular[0] ** 2 * 1e-10) ** 2:
        return dict(result, reason='shape_axis_not_observable_in_image')
    inverse_squared_factor = -float(residual @ change) / norm
    if inverse_squared_factor <= 0:
        return dict(result, reason='no_positive_axis_scale')
    factor = inverse_squared_factor ** -.5
    result['factor'] = factor
    if not .25 <= factor <= 4:
        return dict(result, reason='axis_scale_outside_quarter_to_four_bounds')
    inverse_change = np.eye(3) + (1/factor - 1) * np.outer(direction, direction)
    candidate_linear = linear @ inverse_change
    candidate = np.column_stack([candidate_linear, matrix[:, 3] + (linear-candidate_linear) @ center])
    if not np.isfinite(candidate).all() or np.max(np.abs(candidate)) > 1e8:
        return dict(result, reason='camera_after_scale_exceeds_envelope')
    gram = candidate_linear @ candidate_linear.T
    lengths = np.sqrt(np.diag(gram))
    candidate_singular = np.linalg.svd(candidate_linear, compute_uv=False)
    return dict(result, available=True, reason='conditional_axis_scale', matrix_px=candidate.tolist(),
        image_axis_scale_ratio=float(candidate_singular[0]/candidate_singular[-1]),
        row_cosine=float(gram[0, 1]/np.prod(lengths)),
        initial_traceless_gram_norm=float(np.linalg.norm(residual+change)),
        candidate_traceless_gram_norm=float(np.linalg.norm(residual+inverse_squared_factor*change)))
