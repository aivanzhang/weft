"""Exact rational checks distinguish proved redundancy from numerical uncertainty."""

from fractions import Fraction
from types import SimpleNamespace

import numpy as np
import pytest

from weft import box_screened_linear_feasibility as screened


def exact_minimum(row, lo, hi):
    return sum((Fraction(float(a)) * Fraction(float(l if a >= 0 else h))
                for a, l, h in zip(row, lo, hi)), Fraction())


def verify_enclosures(result, matrix, lower, lo, hi):
    report = result['report']; omitted = set(report['omitted_rows'])
    assert report['screening_complete']
    assert set(report['retained_rows']) | omitted == set(range(len(lower)))
    assert not set(report['retained_rows']) & omitted
    for index, (row, bound) in enumerate(zip(matrix, report['row_lower_enclosures'])):
        exact = exact_minimum(row, lo, hi)
        assert Fraction(bound) <= exact
        if index in omitted: assert Fraction(float(lower[index])) <= Fraction(bound) <= exact


def test_tiny_remote_row_is_removed_without_changing_the_feasible_region():
    a, b, lo, hi = [[1., 1.], [-1., 1.], [1e-19, 0.]], [.6, -.4, -.6], [0., -1.], [1., 1.]
    result = screened.find_box_screened_feasible_point(a, b, lo, hi, seed=[1., 0.])
    assert result['status'] == 'feasible_point_found'
    assert result['report']['omitted_rows'] == [2]
    assert result['report']['phase_one']['report']['constraints'] == 2
    assert result['report']['original_primal']['rows_checked'] == 3
    assert result['report']['original_primal']['maximum_violation'] <= 1e-10
    assert not result['report']['constraints_changed'] and not result['report']['global_infeasibility_claim']
    verify_enclosures(result, a, b, lo, hi)


def test_equality_and_near_boundary_rows_are_retained_without_tolerance_spending():
    a, b, lo, hi = [[1.], [1.], [0.]], [1., 1. + 5e-11, 0.], [1.], [1.]
    result = screened.find_box_screened_feasible_point(a, b, lo, hi)
    assert result['report']['omitted_rows'] == []
    verify_enclosures(result, a, b, lo, hi)
    assert result['point'] == [1.]  # Qualification keeps the existing 1e-10 check, unlike the proof.


def test_cancellation_and_fixed_coordinates_do_not_create_false_proofs():
    a = [[1e12, -1e12], [1e12, -1e12], [1e-19, -1e-19]]
    b, lo, hi = [-1e-5, -1., -.6], [1., 1.], [1., 1.]
    result = screened.find_box_screened_feasible_point(a, b, lo, hi)
    assert result['report']['omitted_rows'] == [1, 2]
    verify_enclosures(result, a, b, lo, hi)


def test_fraction_verifier_covers_signed_mixed_scales_and_subnormal_products():
    rng = np.random.default_rng(1381)
    a = rng.choice([-1., 1.], (256, 8)) * np.power(10., rng.integers(-300, 12, (256, 8)))
    lo = np.array([-1e6, -1., -1e-30, 0., 1e-20, -.1, 2., -4.])
    hi = np.array([1e6, 1., 1e-30, 1e-100, 1e-20, .1, 2., 3.])
    b = np.full(256, -1e12)
    result = screened.find_box_screened_feasible_point(a, b, lo, hi)
    verify_enclosures(result, a, b, lo, hi)


def test_fixed_coordinates_and_narrow_nonredundant_corridor_are_preserved():
    a, b, lo, hi = [[1., 1.], [-1., -1.], [1e-20, 0.]], [.250001, -.25000101, -1.], [.25, -.1], [.25, .1]
    result = screened.find_box_screened_feasible_point(a, b, lo, hi, seed=[.25, 0.])
    assert result['status'] == 'feasible_point_found' and result['point'][0] == .25
    assert result['report']['retained_rows'] == [0, 1]
    verify_enclosures(result, a, b, lo, hi)


def test_unresolved_conflict_does_not_claim_global_infeasibility():
    result = screened.find_box_screened_feasible_point([[1.], [-1.]], [.8, -.2], [-1.], [1.], seed=[0.])
    assert result['point'] is None and result['status'] != 'feasible_point_found'
    assert not result['report']['original_primal']['passes']
    assert not result['report']['global_infeasibility_claim']


def test_all_original_rows_are_rechecked_after_reduced_phase_one(monkeypatch):
    def invalid(*args, **kwargs):
        return {'status': 'feasible_point_found', 'point': [0.], 'report': {'last_point': [0.]}}
    monkeypatch.setattr(screened, 'find_bounded_feasible_point', invalid)
    result = screened.find_box_screened_feasible_point([[1.], [1e-20]], [.5, -1.], [0.], [1.], seed=[0.])
    assert result['status'] == 'original_row_verification_failed' and result['point'] is None
    assert result['report']['original_primal']['rows_checked'] == 2


def test_preprocessing_deadline_preserves_partial_screen_and_never_calls_phase(monkeypatch):
    ticks = [0.]
    def clock(): ticks[0] += .02; return ticks[0]
    monkeypatch.setattr(screened, 'time', SimpleNamespace(monotonic=clock))
    monkeypatch.setattr(screened, 'find_bounded_feasible_point', lambda *a, **k: pytest.fail('Expired preprocessing reached Phase I'))
    result = screened.find_box_screened_feasible_point([[1.]] * 30, [0.] * 30, [-1.], [1.], settings={'maximum_seconds': .1})
    assert result['status'] == 'time_limit' and result['point'] is None
    assert not result['report']['screening_complete'] and result['report']['deadline_stage'] == 'box_screening'


def test_final_deadline_cannot_publish_an_unaudited_point(monkeypatch):
    now = [0.]
    monkeypatch.setattr(screened, 'time', SimpleNamespace(monotonic=lambda: now[0]))
    def late(*args, **kwargs):
        assert kwargs['settings']['maximum_seconds'] <= .49
        now[0] = .51
        return {'status': 'feasible_point_found', 'point': [1.], 'report': {'last_point': [1.]}}
    monkeypatch.setattr(screened, 'find_bounded_feasible_point', late)
    result = screened.find_box_screened_feasible_point([[1.]], [.5], [0.], [1.], seed=[0.])
    assert result['status'] == 'time_limit' and result['point'] is None
    assert result['report']['deadline_stage'] == 'final_original_verification'
    assert result['report']['original_primal_point'] == [0.]
    assert result['report']['last_point'] == [1.]


@pytest.mark.parametrize('a,b,lo,hi', [([[float('nan')]], [0.], [0.], [1.]),
    ([[1.]], float('inf'), [0.], [1.]), ([[1.]], 0., [0.], [1.]),
    ([[1.]], [0.], [1.], [0.]), ([[1e13]], [0.], [0.], [1.]),
    ([[1.]], [0.], [-1e7], [1.]), (np.eye(9), np.zeros(9), np.zeros(9), np.ones(9)),
    (np.zeros((4097, 1)), np.zeros(4097), [0.], [1.])])
def test_invalid_inputs_reject_before_phase_one(monkeypatch, a, b, lo, hi):
    monkeypatch.setattr(screened, 'find_bounded_feasible_point', lambda *a, **k: pytest.fail('Invalid input reached Phase I'))
    with pytest.raises(ValueError): screened.find_box_screened_feasible_point(a, b, lo, hi)


@pytest.mark.parametrize('settings', [{'maximum_seconds': .6}, {'maximum_iterations': 129}, {'tolerance': 1.}, {'maximum_seconds': float('nan')}])
def test_settings_cannot_expand_fixed_limits(settings):
    with pytest.raises(ValueError): screened.find_box_screened_feasible_point([[1.]], [0.], [0.], [1.], settings=settings)
