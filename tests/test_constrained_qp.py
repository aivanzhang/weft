"""Independent optima and KKT witnesses for bounded, feasible-start quadratic steps."""

from itertools import combinations

import numpy as np
import pytest

from weft import constrained_qp as qp


def assert_optimal(result, expected, atol=1e-8):
    assert result['status'] == 'optimal', result
    np.testing.assert_allclose(result['step'], expected, atol=atol, rtol=0)
    report = result['report']
    assert report['primal_maximum_violation'] <= 1e-10
    assert report['stationarity_relative'] <= 1e-10
    assert report['complementarity_relative'] <= 1e-10
    assert not report['global_infeasibility_claim']


def test_unconstrained_correlated_quadratic_has_exact_known_solution():
    h = np.array([[4., 1.], [1., 2.]]); g = np.array([-1., -2.])
    assert_optimal(qp.solve_feasible_qp(h, g, [], []), np.linalg.solve(h, -g))


@pytest.mark.parametrize('scales', [[1., 1., 1.], [1e-8, 1e8, 2.]])
def test_positive_row_scaling_does_not_relax_original_constraints(scales):
    a = np.array([[1., 0.], [0., 1.], [0., -1.]])
    b = np.array([0., 0., -.4]); scales = np.asarray(scales)
    assert_optimal(qp.solve_feasible_qp(np.eye(2), [2., -1.], a * scales[:, None], b * scales), [0., .4])


def test_duplicate_zero_and_opposite_rows_remain_in_final_feasibility_check():
    a = [[1., 0.], [2., 0.], [-1., 0.], [0., 1.], [0., 3.], [0., 0.]]
    result = qp.solve_feasible_qp(np.diag([2., 5.]), [-3., 4.], a, [0.] * 6)
    assert_optimal(result, [0., 0.])
    assert result['report']['constraints'] == 6 and result['report']['zero_rows'] == [5]
    assert len(result['report']['multipliers']) == 6


def test_wrong_active_constraint_is_removed_without_leaving_feasible_cone():
    h = [[4., -1., -1.], [-1., 2., 1.], [-1., 1., 2.]]
    a = np.vstack((np.eye(3), -np.eye(3), [-2., 1., 1.], [-1., 1., 0.], [-1., -1., -1.]))
    result = qp.solve_feasible_qp(h, [-2., 1., 5.], a, np.r_[np.zeros(3), np.full(3, -1.), np.zeros(3)])
    assert_optimal(result, [0., 0., 0.])
    assert any(e['event'] == 'remove_negative_multiplier' for e in result['report']['events'])


def test_numerically_dependent_blocker_is_explicit_not_silently_discarded():
    result = qp.solve_feasible_qp(np.eye(2), [1., 1.], [[1., 0.], [-1., 1e-13]], [0., 0.])
    assert result['status'] == 'dependent_blocker' and result['step'] is None
    assert result['report']['dependent_blockers'] and not result['report']['global_infeasibility_claim']
    assert result['report']['primal_maximum_violation'] <= 1e-10


def test_eight_variable_ill_conditioned_problem_matches_constructed_kkt_witness():
    rng = np.random.default_rng(17)
    rotation, _ = np.linalg.qr(rng.normal(size=(8, 8)))
    h = rotation @ np.diag(np.geomspace(1., 1e7, 8)) @ rotation.T
    expected = np.array([0., 0., -.01, -.01, .04, -.03, .02, -.02])
    a = np.vstack((np.eye(8), -np.eye(8), [1., 0., 0., 0., 0., 0., 0., 0.],
                   [0., 1., 0., 0., 0., 0., 0., 0.], [0., 0., 1., 1., 0., 0., 0., 0.]))
    b = np.r_[np.full(16, -.1), 0., 0., -.02]
    multipliers = np.r_[np.zeros(16), 2., 3., 4.]
    g = a.T @ multipliers - h @ expected
    assert_optimal(qp.solve_feasible_qp(h, g, a, b), expected, atol=1e-7)


def test_small_problem_matches_independent_active_subset_enumeration():
    h = np.array([[3., .4, .2], [.4, 2., -.1], [.2, -.1, 1.]])
    g = np.array([2., -3., 1.])
    a = np.vstack((np.eye(3), -np.eye(3), [1., 1., 0.], [0., 1., 1.]))
    b = np.r_[np.zeros(3), np.full(3, -.8), 0., 0.]
    feasible = []
    for size in range(4):
        for subset in combinations(range(len(a)), size):
            w = a[list(subset)]
            system = np.block([[h, -w.T], [w, np.zeros((size, size))]])
            try: solution = np.linalg.solve(system, np.r_[-g, b[list(subset)]])
            except np.linalg.LinAlgError: continue
            x, multipliers = solution[:3], solution[3:]
            if np.all(a @ x >= b - 1e-10) and np.all(multipliers >= -1e-10):
                feasible.append((.5 * x @ h @ x + g @ x, x))
    expected = min(feasible, key=lambda item: item[0])[1]
    assert_optimal(qp.solve_feasible_qp(h, g, a, b), expected)


def test_invalid_initial_is_not_called_global_infeasibility():
    result = qp.solve_feasible_qp([[1.]], [-2.], [[1.]], [1.])
    assert result['status'] == 'unsupported_infeasible_initial' and result['step'] is None
    assert not result['report']['global_infeasibility_claim']
    assert_optimal(qp.solve_feasible_qp([[1.]], [-2.], [[1.]], [1.], initial=[1.]), [2.])


def test_iteration_limit_keeps_last_feasible_point_but_no_usable_step():
    result = qp.solve_feasible_qp(np.eye(2), [2., -1.], [[1., 0.], [0., -1.]], [0., -.4],
                                  settings={'maximum_iterations': 1})
    assert result['status'] == 'iteration_limit' and result['step'] is None
    assert result['report']['last_feasible_point'] is not None
    assert result['report']['primal_maximum_violation'] <= 1e-10


def test_cooperative_time_limit_is_explicit(monkeypatch):
    ticks = iter([0., 1., 1.1])
    monkeypatch.setattr(qp.time, 'monotonic', lambda: next(ticks))
    result = qp.solve_feasible_qp(np.eye(2), [2., -1.], [[1., 0.]], [0.])
    assert result['status'] == 'time_limit' and result['step'] is None


@pytest.mark.parametrize('h,g,a,b', [([[-1.]], [0.], [], []), ([[1.]], [float('nan')], [], []),
    ([[1e-15]], [0.], [], []), ([[1.]], [0.], [[1e13]], [0.]),
    (np.eye(9), np.zeros(9), [], []), ([[1.]], [0.], [[1.]] * 4097, [0.] * 4097)])
def test_invalid_or_unsupported_numerics_reject_without_hidden_regularization(h, g, a, b):
    with pytest.raises(ValueError): qp.solve_feasible_qp(h, g, a, b)
