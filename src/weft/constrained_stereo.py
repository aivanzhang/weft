"""Bounded two-view point proposals with explicit, fixed-length constraints."""

import math
from types import MappingProxyType

import numpy as np

from weft.camera_pose import _undistort


CONTRACT = MappingProxyType({
    'method': 'length-constrained-stereo-v2', 'maximum_joints': 12,
    'maximum_iterations': 40, 'maximum_backtracks': 8,
    'minimum_ray_angle_degrees': 2., 'depth_bounds_m': (.2, 100.),
    'huber_delta_px': 3., 'maximum_step_m': .05, 'maximum_displacement_m': .25,
    'length_tolerance_m': 1e-5, 'step_tolerance_m': 1e-6,
    'constraint_correction_iterations': 4, 'constraint_correction_trigger_m': .001,
    'maximum_constraint_correction_m': .01,
    'hessian_update': 'damped_lagrangian_bfgs',
    'maximum_reprojection_fraction': .01,
    'scope': 'Constrained proposals, not corrected observations, measured anatomy or physical validation.',
})


def _camera(value):
    if not isinstance(value, dict) or set(value) != {'matrix', 'radial', 'rotation', 'translation', 'width', 'height'}:
        raise ValueError('A camera requires explicit intrinsics, one radial term, transform and image size')
    K, R, t = [np.asarray(value[k], dtype=float) for k in ('matrix', 'rotation', 'translation')]
    k = value['radial']
    if (K.shape != (3, 3) or R.shape != (3, 3) or t.shape != (3,)
            or not all(np.isfinite(x).all() for x in (K, R, t))
            or np.max(np.abs(K)) > 1e5 or np.max(np.abs(t)) > 1e4
            or type(k) not in (int, float) or not math.isfinite(k) or not -1 <= k <= 1
            or any(type(value[n]) is not int or not 16 <= value[n] <= 16384 for n in ('width', 'height'))
            or not np.allclose(K[2], [0, 0, 1], atol=1e-12, rtol=0)
            or abs(K[1, 0]) > 1e-12 or min(K[0, 0], K[1, 1]) <= 0
            or not np.allclose(R @ R.T, np.eye(3), atol=1e-8, rtol=0)
            or not math.isclose(float(np.linalg.det(R)), 1., abs_tol=1e-8)):
        raise ValueError('Camera geometry must be finite, bounded and a proper world-to-camera transform')
    return {'matrix': K, 'radial': k, 'rotation': R, 'translation': t,
            'width': value['width'], 'height': value['height']}


def _project(points, camera):
    q = points @ camera['rotation'].T+camera['translation']
    if np.any(q[:, 2] < .2) or np.any(q[:, 2] > 100) or not np.isfinite(q).all():
        raise ValueError('Projection exceeds the forward depth envelope')
    xy = q[:, :2]/q[:, 2, None]; r2 = np.sum(xy*xy, axis=1); k = camera['radial']
    if np.any(1+3*k*r2 <= 0):
        raise ValueError('Projection leaves the monotonic radial-distortion region')
    factor = 1+k*r2
    pixel = (xy*factor[:, None]) @ camera['matrix'][:2, :2].T+camera['matrix'][:2, 2]
    perspective = np.zeros((len(points), 2, 3))
    perspective[:, 0, 0] = 1/q[:, 2]; perspective[:, 1, 1] = 1/q[:, 2]
    perspective[:, :, 2] = -xy/q[:, 2, None]
    radial = factor[:, None, None]*np.eye(2)+2*k*xy[:, :, None]*xy[:, None, :]
    jacobian = camera['matrix'][:2, :2] @ radial @ perspective @ camera['rotation']
    return pixel, jacobian


def _initial_points(pixels, cameras):
    rays, centers = [], []
    for p, c in zip(pixels, cameras, strict=True):
        xy = _undistort(p, c['matrix'], c['radial'])
        if xy is None:
            raise ValueError('Initial image rays lack invertible distortion')
        ray = np.column_stack([xy, np.ones(len(p))]) @ c['rotation']
        rays.append(ray/np.linalg.norm(ray, axis=1)[:, None])
        centers.append(-c['translation'] @ c['rotation'])
    points, angles = [], []
    for a, b in zip(*rays, strict=True):
        angle = math.degrees(math.acos(float(np.clip(abs(a @ b), 0., 1.))))
        if angle < CONTRACT['minimum_ray_angle_degrees']:
            raise ValueError('Initial ray angle is insufficient for a two-view proposal')
        distances, _, rank, _ = np.linalg.lstsq(np.column_stack([a, -b]), centers[1]-centers[0], rcond=None)
        if rank != 2 or not np.isfinite(distances).all() or np.any(distances <= 0):
            raise ValueError('Initial rays do not meet in front of both cameras')
        points.append((centers[0]+a*distances[0]+centers[1]+b*distances[1])/2)
        angles.append(angle)
    return np.array(points), angles


def _lengths(points, coefficients, targets):
    delta = coefficients @ points; lengths = np.linalg.norm(delta, axis=1)
    if np.any(lengths < 1e-6):
        raise ValueError('A constrained segment is degenerate')
    gradients = (coefficients[:, :, None]*(delta/lengths[:, None])[:, None, :]).reshape(len(targets), -1)
    return lengths-targets, gradients


def _observations(points, cameras, pixels):
    projections = [_project(points, c) for c in cameras]
    residual = np.array([p[0] for p in projections])-pixels
    errors = np.linalg.norm(residual, axis=2); delta = CONTRACT['huber_delta_px']
    cost = float(np.sum(np.where(errors <= delta, .5*errors**2, delta*(errors-.5*delta))))
    weights = np.minimum(1., delta/np.maximum(errors, 1e-12))
    jacobian = np.zeros((2*len(points)*2, len(points)*3))
    for v, (_, jac) in enumerate(projections):
        for j in range(len(points)):
            offset = (v*len(points)+j)*2
            jacobian[offset:offset+2, j*3:j*3+3] = jac[j]
    return cost, residual.ravel(), np.repeat(weights.ravel(), 2), jacobian, errors


def _correct_lengths(points, coefficients, targets):
    original = points; trial = points.copy()
    for _ in range(CONTRACT['constraint_correction_iterations']):
        residual, G = _lengths(trial, coefficients, targets)
        if np.max(np.abs(residual)) < 1e-10:
            break
        step = -G.T @ np.linalg.solve(G @ G.T, residual)
        trial += step.reshape(trial.shape)
        if np.max(np.linalg.norm(trial-original, axis=1)) > CONTRACT['maximum_constraint_correction_m']:
            raise ValueError('Length correction exceeds its local envelope')
    return trial


def fit_stereo_lengths(pixels, cameras, constraints):
    pixels = np.asarray(pixels, dtype=float)
    if (pixels.ndim != 3 or pixels.shape[0] != 2 or pixels.shape[2] != 2
            or not 2 <= pixels.shape[1] <= CONTRACT['maximum_joints'] or not np.isfinite(pixels).all()
            or not isinstance(cameras, (list, tuple)) or len(cameras) != 2):
        raise ValueError('Length fitting needs two finite image observations for each of 2–12 joints')
    cameras = [_camera(c) for c in cameras]; n = pixels.shape[1]
    for p, c in zip(pixels, cameras, strict=True):
        if np.any(p < 0) or np.any(p > [c['width'], c['height']]):
            raise ValueError('Observed joint pixels must remain inside their original images')
    if not isinstance(constraints, (list, tuple)) or not 1 <= len(constraints) < n:
        raise ValueError('Use 1–N−1 independently specified length constraints')
    names, coefficients, targets = [], [], []
    for c in constraints:
        if not isinstance(c, dict) or set(c) != {'name', 'coefficients', 'length_m'}:
            raise ValueError('Every length needs a name, joint coefficients and metric target')
        b = np.asarray(c['coefficients'], dtype=float); length = c['length_m']; name = c['name']
        if (not isinstance(name, str) or not 1 <= len(name) <= 64 or name in names
                or b.shape != (n,) or not np.isfinite(b).all() or np.max(np.abs(b)) > 1
                or np.count_nonzero(b) < 2 or abs(float(b.sum())) > 1e-10
                or type(length) not in (int, float) or not math.isfinite(length) or not .02 <= length <= 2.5):
            raise ValueError('Length constraints must be bounded, translation invariant and uniquely named')
        names.append(name); coefficients.append(b); targets.append(length)
    B, targets = np.array(coefficients), np.array(targets)
    if np.linalg.matrix_rank(B, tol=1e-9) != len(B):
        raise ValueError('Length constraints must be independent; duplicate or cyclic coefficients are unsupported')
    initial, angles = _initial_points(pixels, cameras); points = initial.copy()
    initial_cost, _, _, _, initial_errors = _observations(points, cameras, pixels)
    initial_lengths, _ = _lengths(points, B, targets)
    history = []; status = 'iteration_limit'; damping = .001; evaluations = 0; curvature = None
    for iteration in range(CONTRACT['maximum_iterations']):
        cost, residual, weights, J, errors = _observations(points, cameras, pixels)
        c, G = _lengths(points, B, targets)
        if curvature is None:
            curvature = J.T @ (weights[:, None]*J)
        H = curvature.copy()
        H += np.diag(damping*np.maximum(np.diag(H), 1.))
        gradient = J.T @ (weights*residual)
        # Scale equalities into pixel-sized units to condition the constrained linear system.
        scale = 600.; A = G*scale
        system = np.block([[H, A.T], [A, np.zeros((len(B), len(B)))]])
        try:
            solution = np.linalg.solve(system, np.r_[-gradient, -c*scale])
        except np.linalg.LinAlgError:
            status = 'singular_constraint_system'; break
        step = solution[:n*3].reshape(n, 3)
        maximum = float(np.max(np.linalg.norm(step, axis=1)))
        if not np.isfinite(solution).all():
            status = 'nonfinite_step'; break
        if maximum < CONTRACT['step_tolerance_m'] and np.max(np.abs(c)) <= CONTRACT['length_tolerance_m']:
            status = 'converged'; break
        step *= min(1., CONTRACT['maximum_step_m']/max(maximum, 1e-12))
        penalty = max(1000., 2*float(np.max(np.abs(solution[n*3:])))*scale)
        merit = cost+penalty*float(np.sum(np.abs(c))); accepted = False
        for backtrack in range(CONTRACT['maximum_backtracks']):
            trial = points+step*(.5**backtrack); evaluations += 1
            try:
                # Near feasibility, curved constraints need second-order correction before line search.
                if np.max(np.abs(c)) < CONTRACT['constraint_correction_trigger_m']:
                    trial = _correct_lengths(trial, B, targets)
            except (ValueError, np.linalg.LinAlgError):
                continue
            if np.max(np.linalg.norm(trial-points, axis=1)) > CONTRACT['maximum_step_m']:
                continue
            if np.max(np.linalg.norm(trial-initial, axis=1)) > CONTRACT['maximum_displacement_m']:
                continue
            try:
                trial_cost, trial_residual, trial_weights, trial_J, _ = _observations(trial, cameras, pixels)
                trial_c, trial_G = _lengths(trial, B, targets)
            except ValueError:
                continue
            trial_merit = trial_cost+penalty*float(np.sum(np.abs(trial_c)))
            if trial_merit < merit-1e-12:
                s = (trial-points).ravel(); Bs = curvature @ s; sBs = float(s @ Bs)
                multipliers = solution[n*3:]*scale
                y = trial_J.T @ (trial_weights*trial_residual)-gradient+(trial_G-G).T @ multipliers
                sy = float(s @ y)
                if sBs > 1e-16:
                    # Damping keeps the curvature approximation positive despite noisy residuals.
                    theta = 1. if sy >= .2*sBs else .8*sBs/(sBs-sy)
                    corrected = theta*y+(1-theta)*Bs; denominator = float(s @ corrected)
                    if denominator > 1e-16:
                        curvature += np.outer(corrected, corrected)/denominator-np.outer(Bs, Bs)/sBs
                        curvature = (curvature+curvature.T)*.5
                points = trial; accepted = True; break
        history.append({'iteration': iteration+1, 'image_huber_cost': cost,
            'maximum_length_error_m': float(np.max(np.abs(c))), 'step_m': maximum,
            'accepted': accepted, 'backtracks': backtrack+1, 'damping': damping})
        if accepted:
            damping = max(1e-6, damping*.3)
        else:
            damping *= 10
            if damping > 1e4:
                status = 'no_descent'; break
    final_cost, _, _, _, final_errors = _observations(points, cameras, pixels)
    final_lengths, _ = _lengths(points, B, targets)
    image_limits = [CONTRACT['maximum_reprojection_fraction']*max(c['width'], c['height']) for c in cameras]
    return {'status': status, 'contract': dict(CONTRACT), 'point_proposals_m': points.tolist(),
        'initial_points_m': initial.tolist(), 'initial_ray_angles_degrees': angles,
        'original_reprojection_px': initial_errors.tolist(), 'proposed_reprojection_px': final_errors.tolist(),
        'original_image_huber_cost': initial_cost, 'proposed_image_huber_cost': final_cost,
        'constraint_names': names, 'target_lengths_m': targets.tolist(),
        'initial_length_errors_m': initial_lengths.tolist(), 'proposed_length_errors_m': final_lengths.tolist(),
        'length_constraints_satisfied': bool(np.max(np.abs(final_lengths)) <= CONTRACT['length_tolerance_m']),
        'joint_image_limits_satisfied': np.all(final_errors <= np.array(image_limits)[:, None], axis=0).tolist(),
        'displacements_m': np.linalg.norm(points-initial, axis=1).tolist(), 'iterations': history,
        'trial_evaluations': evaluations, 'physical_validation': 'unverified', 'observations_modified': False}
