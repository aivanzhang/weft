"""Conditional 3D-to-image fits; image agreement does not identify metric geometry."""

import numpy as np


def _points(value, minimum):
    points = np.asarray(value, dtype=float)
    if (points.ndim != 2 or points.shape[1:] != (3,)
            or not minimum <= len(points) <= 1024 or not np.isfinite(points).all()
            or np.max(np.abs(points)) > 1e4):
        raise ValueError(f'Use {minimum}–1024 finite 3D points within the coordinate bound')
    return points


def project_affine_points(points, matrix):
    points = _points(points, 1)
    matrix = np.asarray(matrix, dtype=float)
    if matrix.shape != (2, 4) or not np.isfinite(matrix).all() or np.max(np.abs(matrix)) > 1e8:
        raise ValueError('Use a finite bounded 2-by-4 affine camera matrix')
    return points @ matrix[:, :3].T + matrix[:, 3]


def fit_affine_camera(world_points, image_points, *, image_size, scope):
    """Keep all correspondences, including their residuals and nonmetric ambiguity."""
    if scope != 'construction':
        raise ValueError('Camera fitting accepts only explicit construction evidence')
    points = _points(world_points, 6)
    pixels = np.asarray(image_points, dtype=float)
    if (not isinstance(image_size, (list, tuple)) or len(image_size) != 2
            or any(type(value) is not int or not 1 <= value <= 16384 for value in image_size)):
        raise ValueError('Image size requires bounded integer width and height')
    if (pixels.shape != (len(points), 2) or not np.isfinite(pixels).all()
            or (pixels < 0).any() or (pixels >= image_size).any()):
        raise ValueError('Every 3D point requires one finite in-image observation')

    center = points.mean(axis=0)
    centered = points - center
    scale = float(np.sqrt(np.mean(np.sum(centered**2, axis=1))))
    if scale < 1e-6:
        raise ValueError('Camera construction points have insufficient spatial extent')
    # Uniform normalization preserves the physical-axis conditioning diagnostic.
    design = np.column_stack([centered / scale, np.ones(len(points))])
    singular = np.linalg.svd(design, compute_uv=False)
    if singular[-1] <= singular[0] / 1e4:
        raise ValueError('Camera construction requires well-conditioned noncoplanar points')
    coefficients = np.linalg.lstsq(design, pixels, rcond=None)[0]
    linear = coefficients[:3].T / scale
    matrix = np.column_stack([linear, coefficients[3] - linear @ center])
    projected = project_affine_points(points, matrix)
    projection_singular = np.linalg.svd(linear, compute_uv=False)
    if projection_singular[-1] <= projection_singular[0] * 1e-8:
        raise ValueError('Camera observations collapse the image projection to a line')
    errors = np.linalg.norm(projected - pixels, axis=1)
    lengths = np.linalg.norm(linear, axis=1)
    gram = linear @ linear.T
    isotropic = np.eye(2) * np.trace(gram) / 2
    return {
        'method': 'normalized-all-point-affine-v1',
        'scope': scope, 'image_size': list(image_size),
        'calibration': 'conditional_affine_hypothesis',
        'matrix_px': matrix.tolist(), 'fitted_points_px': projected.tolist(),
        'errors_px': errors.tolist(), 'point_count': len(points),
        'rmse_px': float(np.sqrt(np.mean(errors**2))),
        'maximum_error_px': float(errors.max()),
        'normalized_design_condition': float(singular[0] / singular[-1]),
        'orthographic_diagnostic': {
            'row_lengths_px_per_unit': lengths.tolist(),
            'row_cosine': float(gram[0, 1] / np.prod(lengths)),
            'image_axis_scale_ratio': float(projection_singular[0] / projection_singular[-1]),
            'relative_isotropy_error': float(np.linalg.norm(gram - isotropic) / np.linalg.norm(isotropic)),
            'interpretation': 'Scaled orthographic square-pixel cameras require equal orthogonal rows; passing is not calibration.',
        },
        'metric_geometry_verified': False, 'physical_validation': 'unverified',
        'limitations': [
            'Supplied 3D coordinates and point identities are assumptions unless independently measured.',
            'Compensating changes to 3D geometry and camera can preserve every fitted image point.',
            'An affine fit does not recover focal length, perspective depth, visibility or lens distortion.',
            'No point rejection, reweighting, held-out fitting or physical promotion is performed.',
        ],
    }
