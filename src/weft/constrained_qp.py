"""Small feasible-start convex QPs; unsuccessful directions are never returned as usable."""

import math
import time

import numpy as np


CONTRACT = {
    'method': 'feasible-start-active-set-qp-v1', 'maximum_variables': 8, 'maximum_constraints': 4096,
    'maximum_iterations': 128, 'maximum_seconds': .5, 'maximum_coefficient': 1e12,
    'minimum_hessian_eigenvalue': 1e-12, 'maximum_hessian_condition': 1e12,
    'maximum_solution_abs': 1e6, 'primal_tolerance': 1e-10, 'relative_kkt_tolerance': 1e-10,
    'rank_relative_tolerance': 1e-12,
    'constraint': 'A x >= b, verified in original unscaled units',
    'conditioning': 'Cholesky whitening and positive row normalization; no regularization or relaxed constraints',
    'feasibility': 'Caller supplies a feasible initial point; no phase-I or global infeasibility claim',
    'deadline': 'Cooperative checks between bounded NumPy calls, not native-call preemption',
}


def _settings(settings):
    settings = {} if settings is None else settings
    if not isinstance(settings, dict) or set(settings) - {'maximum_iterations', 'maximum_seconds'}:
        raise ValueError('Only tighter QP iteration/time limits are supported')
    result = {name: settings.get(name, CONTRACT[name]) for name in ('maximum_iterations', 'maximum_seconds')}
    if type(result['maximum_iterations']) is not int or not 1 <= result['maximum_iterations'] <= CONTRACT['maximum_iterations']:
        raise ValueError('Invalid QP iteration limit')
    seconds = result['maximum_seconds']
    if type(seconds) not in (int, float) or not math.isfinite(seconds) or not .001 <= seconds <= CONTRACT['maximum_seconds']:
        raise ValueError('Invalid QP time limit')
    return result


def _array(value, name):
    result = np.asarray(value, dtype=float)
    if not np.all(np.isfinite(result)) or np.any(abs(result) > CONTRACT['maximum_coefficient']):
        raise ValueError(name + ' must have bounded finite coefficients')
    return result.copy()


def _residuals(hessian, gradient, matrix, lower, point, multipliers):
    slack = matrix @ point - lower
    hpoint, dual = hessian @ point, matrix.T @ multipliers
    stationarity = hpoint + gradient - dual
    scale = 1. + max(float(np.max(abs(hpoint))), float(np.max(abs(gradient))), float(np.max(abs(dual))))
    complementarity = float(np.max(abs(multipliers * slack), initial=0.))
    objective = float(.5 * point @ hpoint + gradient @ point)
    return {
        'primal_maximum_violation': max(0., float(np.max(-slack, initial=0.))),
        'dual_maximum_negative_multiplier': max(0., float(np.max(-multipliers, initial=0.))),
        'stationarity_maximum_abs': float(np.max(abs(stationarity))),
        'stationarity_relative': float(np.max(abs(stationarity))) / scale,
        'complementarity_maximum_abs': complementarity,
        'complementarity_relative': complementarity / (1. + abs(objective)),
        'objective': objective,
    }


def solve_feasible_qp(hessian, gradient, matrix, lower, *, initial=None, settings=None):
    """Minimize a strictly convex quadratic under linear lower bounds, starting feasible."""
    started = time.monotonic(); limits = _settings(settings); deadline = started + limits['maximum_seconds']
    hessian, gradient = _array(hessian, 'Hessian'), _array(gradient, 'Gradient')
    matrix, lower = _array(matrix, 'Constraint matrix'), _array(lower, 'Constraint lower bounds')
    if gradient.ndim != 1 or not 1 <= len(gradient) <= CONTRACT['maximum_variables']:
        raise ValueError('QP requires one to eight variables')
    n = len(gradient)
    if matrix.size == 0: matrix = np.empty((0, n))
    if (hessian.shape != (n, n) or matrix.ndim != 2 or matrix.shape[1] != n or
            lower.shape != (len(matrix),) or len(matrix) > CONTRACT['maximum_constraints']):
        raise ValueError('QP dimensions or constraint count differ')
    point = np.zeros(n) if initial is None else _array(initial, 'Initial point')
    if point.shape != (n,) or np.max(abs(point)) > CONTRACT['maximum_solution_abs']:
        raise ValueError('Initial QP point exceeds its bounded shape')
    if not np.allclose(hessian, hessian.T, rtol=0, atol=1e-12 * max(1., float(np.max(abs(hessian))))):
        raise ValueError('QP Hessian must be symmetric')
    eigenvalues = np.linalg.eigvalsh(hessian)
    if eigenvalues[0] < CONTRACT['minimum_hessian_eigenvalue']:
        raise ValueError('QP Hessian must be strictly positive definite without regularization')
    condition = float(eigenvalues[-1] / eigenvalues[0])
    if condition > CONTRACT['maximum_hessian_condition']:
        raise ValueError('QP Hessian exceeds its conditioning bound')
    # Whitening makes the reduced Hessian identity without changing the objective.
    factor = np.linalg.cholesky(hessian)
    transformed = np.linalg.solve(factor, matrix.T).T
    row_scale = np.linalg.norm(transformed, axis=1)
    nonzero = row_scale > 0.
    normalized = transformed[nonzero] / row_scale[nonzero, None]
    rhs = lower[nonzero] / row_scale[nonzero]
    identifiers = np.flatnonzero(nonzero)
    cost = np.linalg.solve(factor, gradient)
    value = factor.T @ point
    active, events, dependent = [], [], []
    multipliers = np.zeros(len(matrix)); iterations = 0
    residuals = _residuals(hessian, gradient, matrix, lower, point, multipliers)
    status = 'iteration_limit'

    def finish():
        report = {'method': CONTRACT['method'], 'limits': limits, 'iterations': iterations,
            'seconds': time.monotonic() - started, 'hessian_condition': condition,
            'variables': n, 'constraints': len(matrix), 'zero_rows': np.flatnonzero(~nonzero).tolist(),
            'active_constraints': [int(identifiers[i]) for i in active], 'dependent_blockers': dependent,
            'row_scale_minimum_nonzero': float(row_scale[nonzero].min()) if np.any(nonzero) else None,
            'row_scale_maximum': float(row_scale.max(initial=0.)), 'multipliers': multipliers.tolist(),
            'last_feasible_point': point.tolist() if residuals['primal_maximum_violation'] <= CONTRACT['primal_tolerance'] else None,
            'events': events, **residuals, 'global_infeasibility_claim': False}
        return {'status': status, 'step': point.tolist() if status == 'optimal' else None, 'report': report}

    if residuals['primal_maximum_violation'] > CONTRACT['primal_tolerance']:
        status = 'unsupported_infeasible_initial'; return finish()
    for iterations in range(1, limits['maximum_iterations'] + 1):
        if time.monotonic() >= deadline:
            status = 'time_limit'; break
        objective_gradient = value + cost
        working = normalized[active]
        if active:
            _, singular, vh = np.linalg.svd(working, full_matrices=True)
            rank = int(np.count_nonzero(singular > singular[0] * CONTRACT['rank_relative_tolerance']))
            if rank != len(active):
                status = 'dependent_active_set'; break
            nullspace = vh[rank:].T
            direction = -nullspace @ (nullspace.T @ objective_gradient)
        else:
            direction = -objective_gradient
        relative_direction = float(np.linalg.norm(direction)) / (1. + float(np.linalg.norm(objective_gradient)))
        if relative_direction <= CONTRACT['relative_kkt_tolerance']:
            active_dual = np.linalg.lstsq(working.T, objective_gradient, rcond=CONTRACT['rank_relative_tolerance'])[0] if active else np.empty(0)
            negative = [i for i, v in enumerate(active_dual) if v < -CONTRACT['relative_kkt_tolerance'] * (1. + np.linalg.norm(objective_gradient))]
            if negative:
                # Smallest original index provides a deterministic anti-cycling tie rule.
                position = min(negative, key=lambda i: int(identifiers[active[i]]))
                removed = active.pop(position)
                events.append({'iteration': iterations, 'event': 'remove_negative_multiplier', 'constraint': int(identifiers[removed])})
                continue
            multipliers[:] = 0.
            for i, multiplier in zip(active, active_dual): multipliers[identifiers[i]] = float(multiplier / row_scale[identifiers[i]])
            residuals = _residuals(hessian, gradient, matrix, lower, point, multipliers)
            # Original-unit primal residual prevents row scaling from relaxing feasibility.
            dual_relative = residuals['dual_maximum_negative_multiplier'] / (1. + float(np.max(abs(multipliers), initial=0.)))
            if (residuals['primal_maximum_violation'] <= CONTRACT['primal_tolerance'] and
                    residuals['stationarity_relative'] <= CONTRACT['relative_kkt_tolerance'] and
                    residuals['complementarity_relative'] <= CONTRACT['relative_kkt_tolerance'] and
                    dual_relative <= CONTRACT['relative_kkt_tolerance']):
                status = 'optimal'
            else:
                status = 'kkt_verification_failed'
            break
        slack, derivative = normalized @ value - rhs, normalized @ direction
        eligible = derivative < 0.
        if active: eligible[active] = False
        ratios = np.full(len(normalized), np.inf)
        with np.errstate(over='ignore', divide='ignore'):
            ratios[eligible] = np.maximum(0., slack[eligible]) / -derivative[eligible]
        blocker = int(np.argmin(ratios)) if len(ratios) else None
        alpha = min(1., float(ratios[blocker])) if blocker is not None else 1.
        next_value = value + alpha * direction
        next_point = np.linalg.solve(factor.T, next_value)
        if not np.all(np.isfinite(next_point)) or np.max(abs(next_point)) > CONTRACT['maximum_solution_abs']:
            status = 'solution_bound'; break
        next_residuals = _residuals(hessian, gradient, matrix, lower, next_point, np.zeros(len(matrix)))
        if next_residuals['primal_maximum_violation'] > CONTRACT['primal_tolerance']:
            events.append({'iteration': iterations, 'event': 'rejected_primal_drift', 'violation': next_residuals['primal_maximum_violation']})
            status = 'numerical_feasibility_loss'; break
        value, point, residuals = next_value, next_point, next_residuals
        if alpha < 1.:
            trial_working = normalized[active + [blocker]]
            singular = np.linalg.svd(trial_working, compute_uv=False)
            rank = int(np.count_nonzero(singular > singular[0] * CONTRACT['rank_relative_tolerance']))
            if rank != len(active) + 1:
                dependent.append(int(identifiers[blocker]))
                status = 'dependent_blocker'; break
            active.append(blocker)
            events.append({'iteration': iterations, 'event': 'add_blocker', 'constraint': int(identifiers[blocker]), 'alpha': alpha})
        else:
            events.append({'iteration': iterations, 'event': 'unblocked_step', 'alpha': alpha})
    return finish()
