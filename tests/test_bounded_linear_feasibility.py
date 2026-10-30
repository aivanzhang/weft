"""Phase I must qualify the original constraints, not merely its auxiliary slack."""

import numpy as np
import pytest

from weft import bounded_linear_feasibility as phase


def feasible(matrix, lower, lo, hi, seed, **kwargs):
    arrays = [np.asarray(v, dtype=float) for v in (matrix, lower, lo, hi)]
    result = phase.find_bounded_feasible_point(*arrays, seed=seed, **kwargs)
    assert result['status'] == 'feasible_point_found', result
    check = phase.original_primal_check(*arrays, np.asarray(result['point']))
    assert check['passes'] and check['maximum_violation'] <= 1e-10
    assert not result['report']['global_infeasibility_claim']
    return result


def test_feasible_point_off_an_empty_return_ray():
    result = feasible([[1., 1.], [-1., 1.]], [.6, -.4], [0., -1.], [1., 1.], [1., 0.])
    assert result['point'][1] > .09
    assert result['report']['internal_variables'] == 3
    assert 'explicit_slack' in result['report']['events'][0]


def test_conflicting_rows_return_unresolved_not_global_infeasibility():
    result = phase.find_bounded_feasible_point([[1.], [-1.]], [.8, -.2], [-1.], [1.], seed=[0.])
    assert result['point'] is None and result['status'] != 'feasible_point_found'
    assert result['report']['explicit_phase_slack'] > 0.
    assert not result['report']['original_primal']['passes'] and not result['report']['global_infeasibility_claim']


def test_narrow_coupled_corridor_preserves_unscaled_tolerance():
    feasible([[1., 1.], [-1., -1.]], [1e-6, -1.01e-6], [-.1, -.1], [.1, .1], [0., 0.])


def test_nearly_dependent_rows_keep_their_independent_inequalities():
    result = feasible([[1., 0.], [1., 1e-6], [-1., 0.]], [.1, .1000001, -.100000001], [-1., -1.], [1., 1.], [0., 0.])
    assert result['report']['original_primal']['rows_checked'] == 3


def test_fixed_coordinates_duplicate_rows_and_zero_rows():
    result = feasible([[1., 1.], [2., 2.], [0., 0.]], [.75, 1.5, 0.], [.25, 0.], [.25, 1.], [.25, 0.])
    assert result['point'][0] == .25 and result['report']['fixed_coordinate_indices'] == [0]
    assert result['report']['internal_variables'] == 2


def test_eight_dimensional_bounded_problem_has_no_hidden_regularizer():
    a = np.vstack((np.eye(8), -np.eye(8), np.ones(8)))
    b = np.r_[np.full(8, -.1), np.full(8, -.1), .6]
    result = feasible(a, b, np.full(8, -.1), np.full(8, .1), np.zeros(8))
    assert result['report']['internal_variables'] == 9
    assert sum(result['point']) >= .6 - 1e-10


@pytest.mark.parametrize('scale', [1e-4, 1., 1e4])
def test_positive_row_scaling_does_not_relax_original_primal_check(scale):
    feasible([[scale], [-scale]], [.5 * scale, -.75 * scale], [0.], [1.], [0.])


def test_iteration_limit_retains_failure_without_usable_point():
    result = phase.find_bounded_feasible_point([[1., 1.], [-1., 1.]], [.6, -.4], [0., -1.], [1., 1.],
                                               seed=[1., 0.], settings={'maximum_iterations': 1})
    assert result['status'] == 'iteration_limit' and result['point'] is None
    assert not result['report']['global_infeasibility_claim']


def test_cooperative_deadline_is_explicit(monkeypatch):
    ticks = iter([0., 1., 1.1]); monkeypatch.setattr(phase.time, 'monotonic', lambda: next(ticks))
    result = phase.find_bounded_feasible_point([[1.]], [.5], [0.], [1.], seed=[0.])
    assert result['status'] == 'time_limit' and result['point'] is None


def test_nonfinite_independent_primal_point_cannot_pass():
    with pytest.raises(ValueError, match='finite point'):
        phase.original_primal_check(np.eye(2), np.zeros(2), np.zeros(2), np.ones(2), np.array([float('nan'), 0.]))


@pytest.mark.parametrize('index', [0, 1, 2, 3])
def test_nonfinite_independent_primal_inputs_cannot_pass(index):
    values = [np.eye(1), np.zeros(1), np.zeros(1), np.ones(1)]
    values[index].flat[0] = float('nan')
    with pytest.raises(ValueError, match='finite constraints'):
        phase.original_primal_check(*values, np.zeros(1))


def test_independent_primal_overflow_cannot_pass():
    with pytest.raises(ValueError, match='nonfinite residuals'):
        phase.original_primal_check([[1e308]], [0.], [0.], [1e308], [1e308])


def test_scalar_row_lower_rejects_as_validation_error():
    with pytest.raises(ValueError, match='dimensions'):
        phase.find_bounded_feasible_point([[1.]], 0., [0.], [1.])


@pytest.mark.parametrize('matrix,lower,lo,hi', [([[float('nan')]], [0.], [0.], [1.]),
    ([[1.]], [0.], [1.], [0.]), ([[1.]], [0.], [-1e7], [1.]), (np.eye(9), np.zeros(9), np.zeros(9), np.ones(9))])
def test_invalid_inputs_reject_without_solver(matrix, lower, lo, hi):
    with pytest.raises(ValueError): phase.find_bounded_feasible_point(matrix, lower, lo, hi)
