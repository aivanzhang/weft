"""Bounded Phase I with an explicit slack variable; only original-row feasibility qualifies."""

import math
import time

import numpy as np


CONTRACT = {'method': 'bounded-explicit-slack-linear-phase-one-v1', 'maximum_variables': 8,
    'maximum_constraints': 4096, 'maximum_iterations': 128, 'maximum_seconds': .5,
    'maximum_coefficient': 1e12, 'maximum_bound_abs': 1e6, 'maximum_initial_slack': 8.,
    'initial_slack_buffer': 1e-6, 'primal_tolerance': 1e-10, 'rank_relative_tolerance': 1e-12,
    'stationarity_tolerance': 1e-10,
    'objective': 'Minimize explicit normalized nonnegative slack t; no Hessian or hidden regularization',
    'qualification': 'Original unscaled A*x>=b and exact variable bounds within1e-10',
    'global_infeasibility_claim': False}


def _array(value, name):
    result = np.asarray(value, dtype=float)
    if not np.all(np.isfinite(result)) or np.any(abs(result) > CONTRACT['maximum_coefficient']):
        raise ValueError(name + ' requires bounded finite coefficients')
    return result.copy()


def original_primal_check(matrix, lower, bounds_lower, bounds_upper, point):
    matrix, lower, bounds_lower, bounds_upper, point = [np.asarray(value, dtype=float)
        for value in (matrix, lower, bounds_lower, bounds_upper, point)]
    if point.ndim != 1 or point.shape != bounds_lower.shape or not np.all(np.isfinite(point)):
        raise ValueError('Original-unit primal verification requires a finite point of the exact shape')
    if (lower.ndim != 1 or bounds_upper.shape != point.shape or matrix.shape != (len(lower), len(point)) or
            any(not np.all(np.isfinite(value)) for value in (matrix, lower, bounds_lower, bounds_upper)) or
            np.any(bounds_lower > bounds_upper)):
        raise ValueError('Original-unit primal verification requires finite constraints with matching dimensions')
    with np.errstate(over='ignore', invalid='ignore'):
        row_residual = lower - matrix @ point
        residual = np.r_[row_residual, bounds_lower - point, point - bounds_upper]
    if not np.all(np.isfinite(residual)):
        raise ValueError('Original-unit primal verification produced nonfinite residuals')
    maximum = max(0., float(np.max(residual, initial=0.)))
    return {'maximum_violation': maximum, 'passes': maximum <= CONTRACT['primal_tolerance'],
            'rows_checked': len(lower), 'variable_bound_checks': 2 * len(point),
            'violating_constraint_rows': np.flatnonzero(row_residual > CONTRACT['primal_tolerance']).tolist()}


def find_bounded_feasible_point(matrix, lower, bounds_lower, bounds_upper, *, seed=None, settings=None):
    """Find a bounded feasible point or return explicit unresolved evidence, never an impossibility claim."""
    started = time.monotonic(); settings = {} if settings is None else settings
    if not isinstance(settings, dict) or set(settings) - {'maximum_iterations', 'maximum_seconds'}:
        raise ValueError('Only tighter Phase-I iteration/time limits are supported')
    iterations_limit = settings.get('maximum_iterations', CONTRACT['maximum_iterations'])
    seconds_limit = settings.get('maximum_seconds', CONTRACT['maximum_seconds'])
    if (type(iterations_limit) is not int or not 1 <= iterations_limit <= CONTRACT['maximum_iterations'] or
            type(seconds_limit) not in (int, float) or not math.isfinite(seconds_limit) or not .001 <= seconds_limit <= CONTRACT['maximum_seconds']):
        raise ValueError('Invalid bounded Phase-I settings')
    deadline = started + seconds_limit
    matrix, lower = _array(matrix, 'Matrix'), _array(lower, 'Lower rows')
    bounds_lower, bounds_upper = _array(bounds_lower, 'Lower box'), _array(bounds_upper, 'Upper box')
    if bounds_lower.ndim != 1 or not 1 <= len(bounds_lower) <= CONTRACT['maximum_variables']:
        raise ValueError('Phase I requires one to eight original variables')
    n = len(bounds_lower)
    if matrix.size == 0: matrix = np.empty((0, n))
    if (lower.ndim != 1 or matrix.shape != (len(lower), n) or bounds_upper.shape != (n,) or
            len(lower) > CONTRACT['maximum_constraints'] or np.any(bounds_lower > bounds_upper) or
            max(np.max(abs(bounds_lower)), np.max(abs(bounds_upper))) > CONTRACT['maximum_bound_abs']):
        raise ValueError('Phase-I dimensions or finite box bounds differ')
    midpoint = (bounds_lower + bounds_upper) / 2
    point = midpoint.copy() if seed is None else _array(seed, 'Seed')
    if point.shape != (n,): raise ValueError('Phase-I seed shape differs')
    point = np.clip(point, bounds_lower, bounds_upper)
    original = original_primal_check(matrix, lower, bounds_lower, bounds_upper, point)
    active, events, iterations, status = [], [], 0, 'iteration_limit'
    dependent_tangent_rows = set()
    slack_value, scales, fixed = None, [], np.flatnonzero(bounds_lower == bounds_upper).tolist()
    free = np.flatnonzero(bounds_lower < bounds_upper)
    phase_primal, phase_stationarity, phase_dual_min = None, None, None

    def finish():
        return {'status': status, 'point': point.tolist() if original['passes'] else None,
            'report': {'method': CONTRACT['method'], 'seconds': time.monotonic() - started, 'iterations': iterations,
                'maximum_iterations': iterations_limit, 'maximum_seconds': seconds_limit,
                'original_variables': n, 'internal_variables': len(free) + 1, 'constraints': len(lower),
                'fixed_coordinate_indices': fixed, 'row_normalization': scales,
                'explicit_phase_slack': slack_value, 'phase_primal_maximum_violation': phase_primal,
                'phase_stationarity_maximum_abs': phase_stationarity, 'phase_minimum_multiplier': phase_dual_min,
                'last_point': point.tolist(), 'original_primal': original, 'events': events,
                'active_phase_rows': active.copy(), 'dependent_tangent_rows': sorted(dependent_tangent_rows),
                'global_infeasibility_claim': False}}

    if original['passes']:
        status = 'feasible_point_found'; return finish()
    if not len(free):
        status = 'fixed_box_conflict'; return finish()
    halfwidth = (bounds_upper[free] - bounds_lower[free]) / 2
    transformed = matrix[:, free] * halfwidth
    row_norm = np.linalg.norm(transformed, axis=1)
    normalizer = np.where(row_norm > 0., row_norm, 1.)
    normalized = transformed / normalizer[:, None]
    rhs = (lower - matrix @ midpoint) / normalizer
    if not np.all(np.isfinite(rhs)) or np.max(abs(rhs), initial=0.) > CONTRACT['maximum_coefficient']:
        status = 'normalization_bound'; return finish()
    scales = normalizer.tolist()
    coordinates = (point[free] - midpoint[free]) / halfwidth
    slack_value = max(0., float(np.max(rhs - normalized @ coordinates, initial=0.))) + CONTRACT['initial_slack_buffer']
    if slack_value > CONTRACT['maximum_initial_slack']:
        status = 'initial_slack_bound'; return finish()
    k = len(free)
    soft = np.column_stack((normalized, np.ones(len(lower))))
    hard = np.column_stack((np.eye(k), np.zeros(k)))
    slack_row = np.r_[np.zeros(k), 1.]
    phase_matrix = np.vstack((soft, hard, -hard, slack_row, -slack_row))
    phase_lower = np.r_[rhs, -np.ones(2 * k), 0., -slack_value]
    value = np.r_[coordinates, slack_value]; gradient = slack_row
    phase_primal = max(0., float(np.max(phase_lower - phase_matrix @ value)))
    if phase_primal > CONTRACT['primal_tolerance']:
        status = 'initial_phase_numerical_failure'; return finish()
    for iterations in range(1, iterations_limit + 1):
        if time.monotonic() >= deadline:
            status = 'time_limit'; break
        working = phase_matrix[active]
        if active:
            _, singular, vh = np.linalg.svd(working, full_matrices=True)
            rank = int(np.count_nonzero(singular > singular[0] * CONTRACT['rank_relative_tolerance']))
            if rank != len(active): status = 'dependent_active_set'; break
            nullspace = vh[rank:].T
            direction = -nullspace @ (nullspace.T @ gradient)
        else:
            direction = -gradient.copy()
        if np.linalg.norm(direction) <= CONTRACT['stationarity_tolerance']:
            multipliers = np.linalg.lstsq(working.T, gradient, rcond=CONTRACT['rank_relative_tolerance'])[0] if active else np.empty(0)
            negative = [i for i, m in enumerate(multipliers) if m < -CONTRACT['stationarity_tolerance']]
            if negative:
                position = min(negative, key=lambda i: active[i]); removed = active.pop(position)
                events.append({'iteration': iterations, 'event': 'remove_negative_multiplier', 'phase_row': removed})
                continue
            phase_stationarity = float(np.max(abs(gradient - working.T @ multipliers)))
            phase_dual_min = float(np.min(multipliers, initial=0.))
            status = 'phase_stationary_without_qualified_original_point'
            break
        gaps, slopes = phase_matrix @ value - phase_lower, phase_matrix @ direction
        eligible = slopes < 0.
        if active:
            # Duplicate normals are tangent, not new blockers; every row is still checked after the step.
            tangent = np.linalg.norm(phase_matrix @ nullspace, axis=1) <= CONTRACT['rank_relative_tolerance'] * np.maximum(1., np.linalg.norm(phase_matrix, axis=1))
            dependent_tangent_rows.update(np.flatnonzero(tangent & eligible).tolist())
            eligible[tangent] = False
            eligible[active] = False
        ratios = np.full(len(phase_lower), np.inf)
        with np.errstate(over='ignore', divide='ignore'):
            ratios[eligible] = np.maximum(0., gaps[eligible]) / -slopes[eligible]
        blocker = int(np.argmin(ratios)); alpha = float(ratios[blocker])
        if not math.isfinite(alpha): status = 'no_bounded_blocking_step'; break
        trial = value + alpha * direction
        violation = max(0., float(np.max(phase_lower - phase_matrix @ trial)))
        if violation > CONTRACT['primal_tolerance'] or not np.all(np.isfinite(trial)):
            events.append({'iteration': iterations, 'event': 'rejected_phase_primal_drift', 'violation': violation})
            status = 'phase_numerical_feasibility_loss'; break
        if trial[-1] > value[-1] + CONTRACT['primal_tolerance']:
            status = 'phase_objective_regression'; break
        value = trial; slack_value = float(value[-1]); phase_primal = violation
        point = midpoint.copy(); point[free] += halfwidth * value[:k]
        original = original_primal_check(matrix, lower, bounds_lower, bounds_upper, point)
        events.append({'iteration': iterations, 'event': 'blocking_step', 'phase_row': blocker,
                       'step_length': alpha, 'explicit_slack': slack_value, 'original_violation': original['maximum_violation']})
        if original['passes']:
            status = 'feasible_point_found'; break
        singular = np.linalg.svd(phase_matrix[active + [blocker]], compute_uv=False)
        rank = int(np.count_nonzero(singular > singular[0] * CONTRACT['rank_relative_tolerance']))
        if rank != len(active) + 1:
            status = 'dependent_blocker'; break
        active.append(blocker)
    return finish()
