"""Constrain an unseen distal endpoint from pixels while preserving its upper bone."""

import math
from types import MappingProxyType

import numpy as np

from weft.constrained_stereo import _camera, _project
from weft.limb_position_fit import _array
from weft.spatial_pose import rotation_zyx


CONTRACT = MappingProxyType({
    'method': 'fixed-upper-bone-distal-pixels-v1',
    'maximum_angle_change_rad': .30, 'maximum_update_rad': .10, 'maximum_bend_rad': 2.6,
    'maximum_iterations': 12, 'maximum_backtracks': 4, 'maximum_body_fit_seconds': 20.,
    'original_angle_regularization': 1e-3, 'correction_temporal_regularization': 5e-4,
    'maximum_temporal_gap_s': .25, 'linear_system_ridge': 1e-9, 'step_tolerance_rad': 1e-9,
    'minimum_confidence': .5, 'maximum_views': 4, 'rotation_roundoff_limit': 1e-5,
    'camera_preparation': 'Nearest proper rotation, preserving original calibration and every derived rotation',
    'objective': 'Every qualified distal pixel divided by maximum image dimension, weighted by minimum visibility/presence, plus angle/correction priors',
    'fixed': 'Upper-bone origin/direction, torso, root, lengths, ground, forces, complete-limb fits and source observations',
    'scope': 'Pixel-constrained bend/twist prior; distal depth remains inferred, not measured anatomy or contact',
})
STATUSES = frozenset(('unchanged_exact_targets', 'iteration_limit', 'nonfinite_step_rejected',
                      'step_tolerance', 'no_descent', 'source_pixel_regression_rejected'))


def distal_geometry(delta, rotation, bend, length, knee, sign):
    twist, flexion = delta[0], bend+delta[1]
    st, ct, sb, cb = math.sin(twist), math.cos(twist), math.sin(flexion), math.cos(flexion)
    local = length*np.array([-sign*sb*st, sign*sb*ct, -cb])
    derivatives = length*np.array([[-sign*sb*ct, -sign*cb*st],
                                   [-sign*sb*st, sign*cb*ct], [0., sb]])
    return knee+rotation @ local, rotation @ derivatives


def fit_distal_pixels(rotation, bend, length, knee, cameras, pixels, confidence, sign, *, previous=None):
    orientation = _array(rotation, (3, 3), 1.00001, 'limb rotation')
    origin = _array(knee, (3,), 2000., 'world knee/elbow')
    if (not np.allclose(orientation.T @ orientation, np.eye(3), atol=1e-8, rtol=0)
            or not math.isclose(float(np.linalg.det(orientation)), 1., abs_tol=1e-8)
            or type(bend) not in (int, float) or not math.isfinite(bend) or not 0 <= bend <= 2.6
            or type(length) not in (int, float) or not math.isfinite(length) or not .01 <= length <= 1.
            or type(sign) is not int or sign not in (-1, 1)
            or not isinstance(cameras, (list, tuple)) or not 1 <= len(cameras) <= CONTRACT['maximum_views']):
        raise ValueError('Distal fit needs a proper fixed frame, bounded length/bend and one to four views')
    views = [_camera(c) for c in cameras]
    targets = _array(pixels, (len(views), 2), 16384., 'image observations')
    weights = _array(confidence, (len(views),), 1., 'image confidence')
    if np.min(weights) < CONTRACT['minimum_confidence'] or any(
            np.any(p < 0) or np.any(p > [c['width'], c['height']]) for p, c in zip(targets, views, strict=True)):
        raise ValueError('Distal image observations must be qualified and inside their source images')
    prior = None if previous is None else _array(previous, (2,), .30, 'previous distal correction')
    scales = np.array([math.sqrt(w)/max(c['width'], c['height']) for w, c in zip(weights, views, strict=True)])
    regularization = CONTRACT['original_angle_regularization']
    temporal = 0. if prior is None else CONTRACT['correction_temporal_regularization']

    def evaluate(delta):
        point, geometry = distal_geometry(delta, orientation, bend, length, origin, sign)
        projected = [_project(point[None, :], c) for c in views]
        raw = np.array([p[0][0] for p in projected])-targets
        residual = (raw*scales[:, None]).ravel()
        jac = np.concatenate([s*p[1][0] @ geometry for s, p in zip(scales, projected, strict=True)])
        sse = float(residual @ residual)
        value = .5*sse+.5*regularization*float(delta @ delta)
        if prior is not None: value += .5*temporal*float((delta-prior) @ (delta-prior))
        return value, sse, residual, jac, point, np.linalg.norm(raw, axis=1)

    delta = np.zeros(2); initial = evaluate(delta)
    lower = np.maximum(-.30, [-.30, -bend]); upper = np.minimum(.30, [.30, 2.6-bend])
    history = []; status = 'unchanged_exact_targets' if initial[1] <= 1e-20 else 'iteration_limit'
    if initial[1] > 1e-20:
        for iteration in range(CONTRACT['maximum_iterations']):
            objective, _, residual, jac, *_ = evaluate(delta)
            hessian = jac.T @ jac+np.eye(2)*(regularization+temporal+CONTRACT['linear_system_ridge'])
            gradient = jac.T @ residual+regularization*delta
            if prior is not None: gradient += temporal*(delta-prior)
            step = np.linalg.solve(hessian, -gradient)
            if not np.isfinite(step).all():
                status = 'nonfinite_step_rejected'; break
            step *= min(1., CONTRACT['maximum_update_rad']/max(float(np.max(abs(step))), 1e-20))
            trials = []; accepted = False
            for backtrack in range(CONTRACT['maximum_backtracks']):
                proposal = np.clip(delta+step*.5**backtrack, lower, upper)
                trial = {'backtrack': backtrack, 'correction_rad': proposal.tolist(), 'accepted': False,
                         'objective': None, 'pixel_sse': None, 'rejection': None}
                trials.append(trial)
                try:
                    evaluated = evaluate(proposal)
                except ValueError as error:
                    trial['rejection'] = str(error); continue
                value, sse = evaluated[:2]; trial.update(objective=value, pixel_sse=sse)
                if np.max(abs(proposal-delta)) <= CONTRACT['step_tolerance_rad']:
                    status = 'step_tolerance'; break
                if math.isfinite(value) and value < objective-1e-15:
                    delta = proposal; trial['accepted'] = accepted = True; break
            history.append({'iteration': iteration+1, 'objective_before': objective, 'trials': trials})
            if not accepted:
                if status != 'step_tolerance': status = 'no_descent'
                break
    final = evaluate(delta)
    if final[1] > initial[1] or not math.isfinite(final[0]):
        status = 'source_pixel_regression_rejected'; delta = np.zeros(2); final = initial
    return {'method': CONTRACT['method'], 'status': status, 'input_rotation_world': orientation.tolist(),
        'input_bend_rad': bend, 'fixed_distal_length_m': length, 'fixed_knee_world_m': origin.tolist(),
        'bend_sign': sign, 'correction_rad': delta.tolist(), 'maximum_angle_change_rad': float(np.max(abs(delta))),
        'output_rotation_world': (orientation @ rotation_zyx([delta[0], 0., 0.])).tolist(),
        'output_bend_rad': bend+float(delta[1]), 'before_world_m': initial[4].tolist(), 'after_world_m': final[4].tolist(),
        'source_pixels': targets.tolist(), 'source_confidence': weights.tolist(),
        'pixel_errors_before': initial[5].tolist(), 'pixel_errors_after': final[5].tolist(),
        'pixel_sse_before': initial[1], 'pixel_sse_after': final[1], 'objective_before': initial[0], 'objective_after': final[0],
        'initial_pixel_jacobian_rank': int(np.linalg.matrix_rank(initial[3])),
        'previous_correction_rad': None if prior is None else prior.tolist(),
        'iterations': history, 'trial_evaluations': sum(len(r['trials']) for r in history),
        'observations_modified': False, 'depth_measured': False, 'physical_validation': 'unverified'}
