"""The opt-in initializer changes neither default outputs nor trajectory admission rules."""

from copy import deepcopy
import json

import numpy as np
import pytest

from test_arm_equipment_refinement import fixture
from test_restored_arm_alignment import curved_evaluator
from weft import box_screened_arm_alignment as screened
from weft import constrained_arm_alignment as base
from weft import restored_arm_alignment as restored
from weft.io import canonical_hash


def stable(value):
    if isinstance(value, dict): return {k: stable(v) for k, v in value.items() if k not in ('seconds', 'fit_seconds')}
    if isinstance(value, list): return [stable(v) for v in value]
    return value


@pytest.mark.parametrize('overlapping,expected', [(False, '7a453b5acc89265c930c2e3d453746039b4879c601f26e2a5520dd6168f7147f'),
    (True, '029984f8bca2f2d920db7eb0073988431196d7f491186e678e6ea0d8e6833256')])
def test_old_full_proposal_bytes_are_identical_except_timing(overlapping, expected):
    spec, evidence, options = fixture()
    if overlapping: options['equipment_boxes'] = [{'id': 'later-obstacle', 'center': [.18, .425, 1.037], 'half_extents': [.015] * 3}]
    result = restored.propose_restored_arm_alignment(json.dumps(spec).encode(), evidence, **options)
    assert canonical_hash(stable(result)) == expected


def test_old_curvature_receipt_is_identical_except_timing():
    solver = restored._SequentialSolver(float('inf'))
    solver.restore(index=3, sweep=1, sigma=.01, baseline=np.zeros(8), direction=np.r_[.1, np.zeros(7)],
        lower=np.full(8, -.1), upper=np.full(8, .1), baseline_cost=.005, evaluate=curved_evaluator(),
        affected_count=1, pair_count=2, deadline=float('inf'))
    assert canonical_hash(stable(solver.restorations)) == 'e71e1a77363c51e1c9bc04935497c87bb21e0e1c155caa11d037e38b4699194a'


@pytest.mark.parametrize('weak', [False, True])
def test_full_trajectory_keeps_latest_neighbors_all_slots_and_original_gates(monkeypatch, weak):
    spec, evidence, options = fixture(low_confidence=weak); frozen = deepcopy(spec); states = []
    actual = base._evaluate_arm_pose
    def observe(index, trial, sigma, derivatives=False, **kwargs):
        if derivatives: states.append(kwargs['changes'].copy())
        return actual(index, trial, sigma, derivatives, **kwargs)
    monkeypatch.setattr(base, '_evaluate_arm_pose', observe)
    options['equipment_boxes'] = [{'id': 'later-obstacle', 'center': [.18, .425, 1.037], 'half_extents': [.015] * 3}]
    result = screened.propose_box_screened_arm_alignment(json.dumps(spec).encode(), evidence, **options)
    report, candidate = result['report'], result['human_candidate']
    assert spec == frozen and report['profile'] == screened.CONTRACT['method']
    assert report['base_contract'] == base.CONTRACT
    assert candidate['reference'][0] == spec['reference'][0]
    assert candidate['root_reference'] == spec['root_reference']
    assert len(candidate['reference']) == len(report['references']) == 6 and len(report['source_slots']) == 11
    assert report['changed_reference_poses'] >= 2
    assert any(np.count_nonzero(np.max(abs(snapshot), axis=1) > 1e-8) >= 2 for snapshot in states)
    assert all(report['fixed_contracts'].values()) and report['geometry_summary']['worsened_pair_samples'] == 0
    assert report['gates']['pointwise_collision_nonregression']
    assert 'per_endpoint_confident_image_nonregression' in report['gates']
    assert report['maximum_angle_correction_rad'] <= .3 + 1e-8
    assert report['maximum_correction_rate_rad_s'] <= 1.5 + 1e-8
    assert report['maximum_endpoint_depth_change_model_units'] <= .08 + 1e-8
    assert report['physics_steps'] == 0 and not report['promoted']
    assert report['geometry_admission'].startswith('unadmitted')
    if weak:
        point = next(p for r in report['references'] if r['frame'] == 4 for p in r['endpoints'] if p['joint'] == 'left_wrist')
        assert point['status'] == 'low_confidence' and point['confidence'] == .2 and point['fit_weight'] == 0


def test_sequential_curved_updates_screen_only_phase_one_and_keep_all_qp_rows(monkeypatch):
    initializer = screened._ScreenedInitializer(float('inf'))
    solver = restored._SequentialSolver(float('inf'), feasibility_finder=initializer)
    trajectory = np.zeros((4, 8)); row_counts = []; neighbor_states = []
    actual = restored.solve_feasible_qp
    def solve(h, g, matrix, lower, **kwargs):
        row_counts.append(len(lower)); return actual(h, g, matrix, lower, **kwargs)
    monkeypatch.setattr(restored, 'solve_feasible_qp', solve)
    for index in range(1, len(trajectory)):
        neighbor_states.append(trajectory[index - 1].copy())
        sample = {'before_gaps': np.array([0., 0., .6]), 'current_gaps': np.array([0., 0., .6])}
        def evaluate(trial, derivatives=False):
            x, y = trial[:2]; gaps = np.array([y - x * x, x * x - y, .6 + 1e-19 * x])
            tangent = np.r_[-2 * x, 1., np.zeros(6)]
            matrix = np.vstack((tangent, -tangent, np.r_[1e-19, np.zeros(7)], np.zeros((8, 8))))
            lower = np.r_[-gaps - 1e-8, [-.08] * 8]
            return .5 * (x - .1) ** 2, np.empty(0), np.empty((0, 8)), bool(np.min(gaps) >= -1e-8), matrix, lower, [(sample, gaps)]
        lo = np.maximum(-.1, trajectory[index - 1] - .15)
        hi = np.minimum(.1, trajectory[index - 1] + .15)
        result = solver.restore(index=index, sweep=0, sigma=.01, baseline=trajectory[index].copy(),
            direction=np.r_[.1, np.zeros(7)], lower=lo, upper=hi, baseline_cost=.005, evaluate=evaluate,
            affected_count=1, pair_count=3, deadline=float('inf'))
        assert result['trial'] is not None
        trajectory[index] = result['trial']
        assert np.all(np.diff(trajectory[:index + 1], axis=0) <= .15 + 1e-10)
    assert np.all(trajectory[0] == 0) and np.count_nonzero(trajectory[:, 0]) == 3
    assert np.any(neighbor_states[1] != 0) and len(initializer.receipts) == solver.phase_calls == 3
    assert row_counts == [27] * 3  # Minimum-norm solve receives the full original rows, not the screened subset.
    for receipt in initializer.receipts:
        report = receipt['result']['report']
        assert 2 in report['omitted_rows'] and report['original_primal']['rows_checked'] == 27
        assert report['original_primal']['passes'] and report['phase_one']['report']['constraints'] < 27
    assert all(r['status'] == 'accepted_restored_step' for r in solver.restorations)


def test_receipt_budget_stops_before_unrecorded_initializer_work(monkeypatch):
    initializer = screened._ScreenedInitializer(float('inf'))
    monkeypatch.setitem(screened.CONTRACT, 'maximum_screening_report_bytes', 1)
    monkeypatch.setattr(screened, 'find_box_screened_feasible_point', lambda *a, **k: pytest.fail('Unreserved call'))
    with pytest.raises(base._BudgetExpired, match='receipt_budget'):
        initializer(np.eye(8), np.zeros(8), np.full(8, -1.), np.ones(8), seed=np.zeros(8), settings={})
    assert initializer.calls == 0 and initializer.receipts == []


def test_initializer_failure_is_retained_before_propagation(monkeypatch):
    initializer = screened._ScreenedInitializer(float('inf'))
    def fail(*a, **k): raise ValueError('Injected invalid system')
    monkeypatch.setattr(screened, 'find_box_screened_feasible_point', fail)
    with pytest.raises(ValueError, match='invalid system'):
        initializer(np.eye(8), np.zeros(8), np.full(8, -1.), np.ones(8), seed=np.zeros(8), settings={})
    assert initializer.calls == len(initializer.receipts) == 1
    assert initializer.receipts[0]['status'] == 'invalid_bounded_problem'


@pytest.mark.parametrize('stage', ['screened_output_serialization', 'screened_output_bound_check'])
def test_finalization_deadline_never_publishes_complete_bounded_success(monkeypatch, stage):
    actual = base._check_deadline
    def check(deadline, label):
        if label == stage: raise base._BudgetExpired(label)
        actual(deadline, label)
    monkeypatch.setattr(base, '_check_deadline', check)
    spec, evidence, options = fixture()
    result = screened.propose_box_screened_arm_alignment(json.dumps(spec).encode(), evidence, **options)
    assert result['status'] == 'partial_budget_exhausted'
    assert not result['report']['gates']['complete_bounded_solve']
    assert len(result['report']['references']) == 6 and len(result['report']['source_slots']) == 11
    assert result['report']['geometry_checks'] and not result['report']['promoted']
