"""Conditional rigid shape from complete tracks; missing observations stay missing."""

from types import MappingProxyType

import numpy as np


CONTRACT = MappingProxyType({
    'method': 'complete-track-scaled-orthographic-v1',
    'minimum_frames': 8, 'maximum_frames': 32, 'maximum_time_step_s': .2,
    'minimum_complete_tracks': 6, 'maximum_complete_tracks': 128,
    'maximum_input_tracks': 4096, 'minimum_third_singular_ratio': .01,
    'maximum_metric_condition': 1e4, 'minimum_metric_null_gap': 10.,
    'minimum_metric_eigenvalue_ratio': 1e-4,
    'maximum_fit_p95_px': 2., 'maximum_fit_px': 6.,
    'assumption': 'One rigid point set, scaled orthographic square-pixel cameras, stable point identities.',
    'missing_data': 'Use only tracks observed in every supplied frame; retain all coverage and never fill gaps.',
    'units': 'Centered relative shape with RMS radius one; not meters or a collision envelope.',
    'ambiguity': 'Global scale, reflection and camera-versus-object motion remain unresolved.',
})


def _quadratic(a, b):
    # The orthonormal symmetric basis makes the metric null-space test rotation invariant.
    return np.array([a[0]*b[0], a[1]*b[1], a[2]*b[2],
        (a[0]*b[1]+a[1]*b[0])/np.sqrt(2),
        (a[0]*b[2]+a[2]*b[0])/np.sqrt(2),
        (a[1]*b[2]+a[2]*b[1])/np.sqrt(2)])


def _errors(predicted, observed):
    errors = np.linalg.norm(predicted-observed, axis=2)
    return {'per_point_px': errors.tolist(), 'rms_px': float(np.sqrt(np.mean(errors**2))),
        'p95_px': float(np.percentile(errors, 95)), 'maximum_px': float(errors.max())}


def _fits(errors):
    return (errors['p95_px'] <= CONTRACT['maximum_fit_p95_px']
            and errors['maximum_px'] <= CONTRACT['maximum_fit_px'])


def reconstruct_track_shape(points_px, *, frame_ids, times_s, track_ids, image_size, scope):
    """Return a relative point-cloud hypothesis, or a specific unobservable/rejected result."""
    if scope != 'construction':
        raise ValueError('Shape fitting accepts only explicit construction evidence')
    if (not isinstance(frame_ids, (list, tuple)) or not 1 <= len(frame_ids) <= CONTRACT['maximum_frames']
            or any(type(f) is not int or not 0 <= f <= 1_000_000_000 for f in frame_ids)
            or any(b <= a for a, b in zip(frame_ids, frame_ids[1:]))):
        raise ValueError('Use 1–32 strictly increasing integer source frames')
    if (not isinstance(track_ids, (list, tuple)) or len(track_ids) > CONTRACT['maximum_input_tracks']
            or any(type(t) is not int or not 0 <= t < 64000 for t in track_ids)
            or len(set(track_ids)) != len(track_ids)):
        raise ValueError('Use distinct bounded track IDs')
    if (not isinstance(image_size, (list, tuple)) or len(image_size) != 2
            or any(type(v) is not int or not 1 <= v <= 16384 for v in image_size)):
        raise ValueError('Image size requires bounded integer width and height')
    times = np.asarray(times_s, dtype=float)
    if (times.shape != (len(frame_ids),) or not np.isfinite(times).all()
            or (times < 0).any() or (times > 1e7).any() or (np.diff(times) <= 0).any()):
        raise ValueError('Use one finite increasing original time per frame')
    points = np.asarray(points_px, dtype=float)
    if points.shape != (len(frame_ids), len(track_ids), 2):
        raise ValueError('Supply a frame-by-track-by-two array, with both coordinates missing together')
    available = np.isfinite(points).all(axis=2)
    missing = np.isnan(points).all(axis=2)
    if not (available | missing).all():
        raise ValueError('Missing points require two NaNs; partial missing values and infinities are invalid')
    if (points[available] < 0).any() or (points[available] >= image_size).any():
        raise ValueError('Observed pixels must remain inside the original image')
    complete = available.all(axis=0)
    included = [t for t, keep in zip(track_ids, complete, strict=True) if keep]
    if len(included) > CONTRACT['maximum_complete_tracks']:
        raise ValueError('Complete-track budget exceeded; do not silently select a smaller subset')
    result = {'method': CONTRACT['method'], 'scope': scope, 'status': 'unavailable',
        'frame_ids': list(frame_ids), 'times_s': times.tolist(), 'track_ids': list(track_ids),
        'complete_track_ids': included,
        'incomplete_track_ids': [t for t, keep in zip(track_ids, complete, strict=True) if not keep],
        'coverage': {'source_point_slots': int(available.size), 'observed_point_slots': int(available.sum()),
            'missing_point_slots': int(missing.sum()), 'available_per_frame': available.sum(axis=1).tolist(),
            'available_per_track': available.sum(axis=0).tolist()},
        'singular_values': None, 'third_singular_ratio': None, 'affine_fit': None,
        'metric': None, 'candidate': None, 'observations_modified': False,
        'physical_point_identity_verified': False, 'rigidity_verified': False,
        'metric_scale_verified': False, 'physical_validation': 'unverified'}
    if len(frame_ids) < CONTRACT['minimum_frames']:
        return result | {'status': 'insufficient_frames'}
    if (np.diff(frame_ids) != 1).any() or (np.diff(times) > CONTRACT['maximum_time_step_s']+1e-12).any():
        return result | {'status': 'source_gap'}
    if len(included) < CONTRACT['minimum_complete_tracks']:
        return result | {'status': 'insufficient_complete_tracks'}
    observed = points[:, complete, :]
    center = observed.mean(axis=1)
    matrix = (observed-center[:, None, :]).transpose(0, 2, 1).reshape(2*len(frame_ids), -1)
    amplitude = float(np.sqrt(np.mean(matrix**2)))
    if amplitude < 1e-6:
        return result | {'status': 'insufficient_image_extent'}
    left, singular, right = np.linalg.svd(matrix/amplitude, full_matrices=False)
    motion = left[:, :3]*np.sqrt(singular[:3])
    shape = np.sqrt(singular[:3])[:, None]*right[:3]
    fitted = (motion @ shape * amplitude).reshape(-1, 2, len(included)).transpose(0, 2, 1)+center[:, None, :]
    result.update(singular_values=singular.tolist(), third_singular_ratio=float(singular[2]/singular[0]),
        affine_fit={'predicted_points_px': fitted.tolist(), 'errors': _errors(fitted, observed)})
    if not _fits(result['affine_fit']['errors']):
        return result | {'status': 'inconsistent_affine_tracks'}
    if result['third_singular_ratio'] < CONTRACT['minimum_third_singular_ratio']:
        return result | {'status': 'insufficient_depth_excitation'}

    constraints = []
    for x, y in motion.reshape(-1, 2, 3):
        energy = float(x @ x+y @ y)
        if energy < 1e-12:
            return result | {'status': 'collapsed_projection'}
        constraints.extend([(_quadratic(x, x)-_quadratic(y, y))/energy,
            np.sqrt(2)*_quadratic(x, y)/energy])
    _, metric_singular, vectors = np.linalg.svd(constraints, full_matrices=False)
    identifiable = metric_singular[-2] > metric_singular[0]/CONTRACT['maximum_metric_condition']
    gap = float(metric_singular[-2]/max(metric_singular[-1], 1e-15))
    result['metric'] = {'constraint_singular_values': metric_singular.tolist(), 'null_gap': gap,
        'identifiable': bool(identifiable), 'matrix': None, 'eigenvalues': None}
    if not identifiable:
        return result | {'status': 'unobservable_metric'}
    if gap < CONTRACT['minimum_metric_null_gap']:
        return result | {'status': 'inconsistent_metric_constraints'}
    q = vectors[-1]
    if abs(q[:3].sum()) < 1e-12:
        return result | {'status': 'nonpositive_metric'}
    q = q*(3/q[:3].sum())
    metric = np.array([[q[0], q[3]/np.sqrt(2), q[4]/np.sqrt(2)],
        [q[3]/np.sqrt(2), q[1], q[5]/np.sqrt(2)],
        [q[4]/np.sqrt(2), q[5]/np.sqrt(2), q[2]]])
    eigenvalues = np.linalg.eigvalsh(metric)
    result['metric'].update(matrix=metric.tolist(), eigenvalues=eigenvalues.tolist())
    if eigenvalues[0] <= CONTRACT['minimum_metric_eigenvalue_ratio']*eigenvalues[-1]:
        return result | {'status': 'nonpositive_metric'}
    upgrade = np.linalg.cholesky(metric)
    recovered = np.linalg.solve(upgrade, shape)
    radius = float(np.sqrt(np.mean(np.sum(recovered**2, axis=0))))
    recovered /= radius
    cameras = (motion @ upgrade*radius*amplitude).reshape(-1, 2, 3)
    # Fit actual equal-scale orthogonal camera rows, then check every original residual again.
    for index, camera in enumerate(cameras):
        u, values, v = np.linalg.svd(camera, full_matrices=False)
        cameras[index] = values.mean()*(u @ v)
    first = cameras[0]/np.linalg.norm(cameras[0], axis=1)[:, None]
    rotation = np.vstack([first, np.cross(*first)])
    recovered = rotation @ recovered
    cameras = cameras @ rotation.T
    projected = np.einsum('fij,jp->fpi', cameras, recovered)+center[:, None, :]
    errors = _errors(projected, observed)
    result['candidate'] = {'track_ids': included, 'points_relative': recovered.T.tolist(),
        'camera_rows_px_per_relative_unit': cameras.tolist(), 'feature_centroids_px': center.tolist(),
        'predicted_points_px': projected.tolist(), 'errors': errors,
        'coordinate_frame': 'First image x/y axes; feature centroid origin; RMS point radius one.',
        'reflection_alternative': 'Negate every point z and every camera third column; projections are identical.',
        'unknown': ['metric scale', 'depth handedness', 'hidden shape and collision bounds',
            'camera versus object motion', 'physical identity and rigidity', 'forces and materials']}
    return result | {'status': 'conditional_shape' if _fits(errors) else 'inconsistent_orthographic_tracks'}
