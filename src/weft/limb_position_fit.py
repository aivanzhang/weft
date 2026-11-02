"""Bounded angle proposals for a fixed-length limb; no floor or dynamics input."""

import math
from types import MappingProxyType

import numpy as np


CONTRACT = MappingProxyType({
    'method': 'fixed-length-limb-position-v1', 'maximum_angle_change_rad': .30,
    'maximum_update_rad': .10, 'maximum_bend_rad': 2.6,
    'maximum_iterations': 12, 'maximum_backtracks': 4,
    'maximum_body_fit_seconds': 20.,
    'original_angle_regularization': 1e-3, 'correction_temporal_regularization': 5e-4,
    'maximum_temporal_gap_s': .25, 'linear_system_ridge': 1e-9,
    'step_tolerance_rad': 1e-9, 'maximum_target_coordinate_m': 4.,
    'objective': 'Equal squared distance to two supported joint positions plus bounded angle/correction priors',
    'fixed': 'Proximal origin, torso frame, both segment lengths, root, floor, force limits and source observations',
    'scope': 'Source-derived angle proposal, not observed anatomy, contact or independent physical accuracy',
})
STATUSES = frozenset(('unchanged_exact_targets', 'iteration_limit', 'nonfinite_step_rejected',
                      'step_tolerance', 'no_descent', 'source_position_regression_rejected'))


def limb_geometry(angles, lengths, bend_sign, *, jacobian=False):
    z, y, x, bend = angles
    cz, sz, cy, sy, cx, sx = math.cos(z), math.sin(z), math.cos(y), math.sin(y), math.cos(x), math.sin(x)
    rz = np.array([[cz, -sz, 0.], [sz, cz, 0.], [0., 0., 1.]])
    ry = np.array([[cy, 0., sy], [0., 1., 0.], [-sy, 0., cy]])
    rx = np.array([[1., 0., 0.], [0., cx, -sx], [0., sx, cx]])
    rotation = rz @ ry @ rx
    proximal, distal = lengths
    local = np.array([[0., 0., -proximal], [0., bend_sign*distal*math.sin(bend), -proximal-distal*math.cos(bend)]])
    points = local @ rotation.T
    if not jacobian:
        return points, None
    derivatives = [np.array([[-sz, -cz, 0.], [cz, -sz, 0.], [0., 0., 0.]]) @ ry @ rx,
                   rz @ np.array([[-sy, 0., cy], [0., 0., 0.], [-cy, 0., -sy]]) @ rx,
                   rz @ ry @ np.array([[0., 0., 0.], [0., -sx, -cx], [0., cx, -sx]])]
    jac = np.zeros((2, 3, 4))
    jac[:, :, :3] = np.stack([local @ derivative.T for derivative in derivatives], axis=-1)
    jac[1, :, 3] = rotation @ [0., bend_sign*distal*math.cos(bend), distal*math.sin(bend)]
    return points, jac


def _array(value, shape, limit, name):
    result = np.asarray(value, dtype=float)
    if result.shape != shape or not np.isfinite(result).all() or np.max(abs(result)) > limit:
        raise ValueError('Limb fit requires finite bounded '+name)
    return result.copy()


def fit_limb_positions(angles, lengths, targets, bend_sign, *, previous=None):
    """Targets are knee/ankle or elbow/wrist positions in the proximal parent frame."""
    original = _array(angles, (4,), math.tau, 'angles')
    size = _array(lengths, (2,), 1., 'fixed lengths')
    target = _array(targets, (2, 3), CONTRACT['maximum_target_coordinate_m'], 'target positions')
    if np.min(size) < .01 or not 0 <= original[3] <= CONTRACT['maximum_bend_rad'] or type(bend_sign) is not int or bend_sign not in (-1, 1):
        raise ValueError('Limb fit requires positive fixed lengths, an admitted bend and explicit hinge sign')
    prior = None if previous is None else _array(previous, (4,), CONTRACT['maximum_angle_change_rad'], 'previous correction')
    temporal = 0. if prior is None else CONTRACT['correction_temporal_regularization']
    regularization = CONTRACT['original_angle_regularization']

    def loss(delta, points):
        position = float(np.sum((points-target)**2))
        objective = .5*position+.5*regularization*float(delta @ delta)
        if prior is not None:
            objective += .5*temporal*float((delta-prior) @ (delta-prior))
        return objective, position

    initial, _ = limb_geometry(original, size, bend_sign)
    delta = np.zeros(4); before, before_position = loss(delta, initial)
    lower = np.maximum(-CONTRACT['maximum_angle_change_rad'], [-math.tau]*3+[0.]-original)
    upper = np.minimum(CONTRACT['maximum_angle_change_rad'], [math.tau]*3+[CONTRACT['maximum_bend_rad']]-original)
    history = []; status = 'unchanged_exact_targets' if before_position <= 1e-20 else 'iteration_limit'
    if before_position > 1e-20:
        for iteration in range(CONTRACT['maximum_iterations']):
            points, jac = limb_geometry(original+delta, size, bend_sign, jacobian=True)
            objective, _ = loss(delta, points); design = jac.reshape(6, 4)
            hessian = design.T @ design+np.eye(4)*(regularization+temporal+CONTRACT['linear_system_ridge'])
            gradient = design.T @ (points-target).ravel()+regularization*delta
            if prior is not None: gradient += temporal*(delta-prior)
            step = np.linalg.solve(hessian, -gradient)
            if not np.isfinite(step).all():
                status = 'nonfinite_step_rejected'; break
            step *= min(1., CONTRACT['maximum_update_rad']/max(float(np.max(abs(step))), 1e-20))
            attempts = []; accepted = False
            for backtrack in range(CONTRACT['maximum_backtracks']):
                trial = np.clip(delta+step*.5**backtrack, lower, upper)
                proposed, _ = limb_geometry(original+trial, size, bend_sign)
                value, position = loss(trial, proposed)
                attempts.append({'backtrack': backtrack, 'angle_correction_rad': trial.tolist(),
                                 'objective': value, 'position_sse_m2': position, 'accepted': False})
                if np.max(abs(trial-delta)) <= CONTRACT['step_tolerance_rad']:
                    status = 'step_tolerance'; break
                if math.isfinite(value) and value < objective-1e-15:
                    delta = trial; accepted = True; attempts[-1]['accepted'] = True; break
            history.append({'iteration': iteration+1, 'objective_before': objective, 'trials': attempts})
            if not accepted:
                if status != 'step_tolerance': status = 'no_descent'
                break
    final, _ = limb_geometry(original+delta, size, bend_sign)
    after, after_position = loss(delta, final)
    if after_position > before_position or not math.isfinite(after):
        status = 'source_position_regression_rejected'
        delta = np.zeros(4); final = initial.copy(); after, after_position = before, before_position
    return {'method': CONTRACT['method'], 'status': status, 'input_angles_rad': original.tolist(),
            'output_angles_rad': (original+delta).tolist(), 'angle_correction_rad': delta.tolist(),
            'maximum_angle_change_rad': float(np.max(abs(delta))), 'fixed_lengths_m': size.tolist(),
            'targets_parent_m': target.tolist(), 'before_parent_m': initial.tolist(), 'after_parent_m': final.tolist(),
            'position_sse_before_m2': before_position, 'position_sse_after_m2': after_position,
            'objective_before': before, 'objective_after': after,
            'previous_correction_rad': None if prior is None else prior.tolist(),
            'iterations': history, 'trial_evaluations': sum(len(row['trials']) for row in history),
            'observations_modified': False, 'physical_validation': 'unverified'}
