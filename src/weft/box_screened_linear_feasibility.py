"""Conservative box redundancy screening before the unchanged explicit-slack Phase I."""

from copy import deepcopy
import math
import time

import numpy as np

from weft.bounded_linear_feasibility import CONTRACT as PHASE_CONTRACT
from weft.bounded_linear_feasibility import find_bounded_feasible_point, original_primal_check


CONTRACT = {**deepcopy(PHASE_CONTRACT), 'method': 'outward-box-screened-explicit-phase-one-v1',
    'phase_one_method': PHASE_CONTRACT['method'],
    'screen': 'For A*x>=b, round every product and accumulation toward -infinity over the exact input box',
    'redundancy_rule': 'Omit only when the outward lower enclosure is >= original b; no tolerance',
    'qualification': 'Every original unscaled row and exact input box checked at the unchanged 1e-10 tolerance',
    'deadline_scope': 'Cooperative total including validation, screening, Phase I and final original-unit verification',
    'maximum_finalization_reserve_seconds': .01,
    'constraints_changed': False, 'global_infeasibility_claim': False}


def _finite(value, name):
    array = np.asarray(value, dtype=float)
    if not np.all(np.isfinite(array)) or np.any(abs(array) > CONTRACT['maximum_coefficient']):
        raise ValueError(name + ' requires bounded finite coefficients')
    return array.copy()


def _row_lower_enclosure(row, lower, upper):
    value = 0.
    for coefficient, lo, hi in zip(row, lower, upper):
        endpoint = lo if coefficient >= 0. else hi
        product = math.nextafter(float(coefficient) * float(endpoint), -math.inf)
        value = math.nextafter(value + product, -math.inf)
    if not math.isfinite(value): raise ValueError('Box enclosure became nonfinite')
    return value


def find_box_screened_feasible_point(matrix, lower, bounds_lower, bounds_upper, *, seed=None, settings=None):
    """Return an original-row-qualified point or bounded unresolved evidence; never claim infeasibility."""
    started = time.monotonic(); settings = {} if settings is None else settings
    if not isinstance(settings, dict) or set(settings) - {'maximum_iterations', 'maximum_seconds'}:
        raise ValueError('Only tighter screening/Phase-I iteration and total-time limits are supported')
    iterations = settings.get('maximum_iterations', CONTRACT['maximum_iterations'])
    seconds = settings.get('maximum_seconds', CONTRACT['maximum_seconds'])
    if (type(iterations) is not int or not 1 <= iterations <= CONTRACT['maximum_iterations'] or
            type(seconds) not in (int, float) or not math.isfinite(seconds) or not .001 <= seconds <= CONTRACT['maximum_seconds']):
        raise ValueError('Invalid bounded screening/Phase-I settings')
    deadline = started + seconds
    matrix, lower, lo, hi = [_finite(value, name) for value, name in
        ((matrix, 'Matrix'), (lower, 'Lower rows'), (bounds_lower, 'Lower box'), (bounds_upper, 'Upper box'))]
    if lo.ndim != 1 or not 1 <= len(lo) <= CONTRACT['maximum_variables']:
        raise ValueError('Screened Phase I requires one to eight original variables')
    n = len(lo)
    if matrix.size == 0: matrix = np.empty((0, n))
    if (lower.ndim != 1 or matrix.shape != (len(lower), n) or hi.shape != (n,) or
            len(lower) > CONTRACT['maximum_constraints'] or np.any(lo > hi) or
            max(np.max(abs(lo)), np.max(abs(hi))) > CONTRACT['maximum_bound_abs']):
        raise ValueError('Screened Phase-I dimensions or exact box bounds differ')
    point = (lo + hi) / 2 if seed is None else _finite(seed, 'Seed')
    if point.shape != (n,): raise ValueError('Screened Phase-I seed shape differs')
    point = np.clip(point, lo, hi)
    enclosures, omitted = [], []
    phase, original, original_point, status, stage = None, None, None, 'time_limit', 'validation'

    def finish():
        excluded = set(omitted)
        report = {'method': CONTRACT['method'], 'seconds': None, 'maximum_seconds': seconds,
                'maximum_iterations': iterations, 'original_constraints': len(lower), 'original_variables': n,
                'screened_rows': len(enclosures), 'screening_complete': len(enclosures) == len(lower),
                'row_lower_enclosures': enclosures, 'omitted_rows': omitted,
                'retained_rows': [i for i in range(len(lower)) if i not in excluded],
                'original_primal': original, 'original_primal_point': original_point,
                'last_point': point.tolist(), 'phase_one': phase,
                'deadline_stage': None, 'constraints_changed': False, 'global_infeasibility_claim': False}
        elapsed = time.monotonic() - started; report['seconds'] = elapsed
        final_status = status if elapsed < seconds else 'time_limit'
        if final_status == 'time_limit': report['deadline_stage'] = stage
        qualified = final_status == 'feasible_point_found' and original is not None and original['passes']
        return {'status': final_status, 'point': point.tolist() if qualified else None, 'report': report}

    if time.monotonic() >= deadline: return finish()
    stage = 'box_screening'
    for index, row in enumerate(matrix):
        if time.monotonic() >= deadline: return finish()
        enclosure = _row_lower_enclosure(row, lo, hi); enclosures.append(enclosure)
        if enclosure >= lower[index]: omitted.append(index)
    omitted_set = set(omitted)
    retained = [i for i in range(len(lower)) if i not in omitted_set]
    stage = 'initial_original_verification'
    if time.monotonic() >= deadline: return finish()
    original = original_primal_check(matrix, lower, lo, hi, point)
    original_point = point.tolist()
    if original['passes']:
        status = 'feasible_point_found'; return finish()
    stage = 'phase_one'
    remaining = deadline - time.monotonic() - min(CONTRACT['maximum_finalization_reserve_seconds'], seconds / 10)
    if remaining < .001: return finish()
    phase = find_bounded_feasible_point(matrix[retained], lower[retained], lo, hi, seed=point,
        settings={'maximum_iterations': iterations, 'maximum_seconds': min(remaining, PHASE_CONTRACT['maximum_seconds'])})
    status = phase['status']; point = np.asarray(phase['report']['last_point'], dtype=float)
    stage = 'final_original_verification'
    if time.monotonic() >= deadline:
        status = 'time_limit'; return finish()
    original = original_primal_check(matrix, lower, lo, hi, point)
    original_point = point.tolist()
    if original['passes'] and phase['point'] is not None:
        status = 'feasible_point_found'
    elif phase['point'] is not None:
        status = 'original_row_verification_failed'
    return finish()
