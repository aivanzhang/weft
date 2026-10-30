"""Sequential restoration preserves full-trajectory gates and never installs a proposal."""

from copy import deepcopy
import json

import numpy as np
import pytest

from test_arm_equipment_refinement import fixture
from weft import constrained_arm_alignment as base
from weft import restored_arm_alignment as restored
from weft.arm_equipment_refinement import _Geometry
from weft.human import human_xml


def propose(spec, evidence, options):
    return restored.propose_restored_arm_alignment(json.dumps(spec).encode(), evidence, **options)


def test_whole_trajectory_uses_latest_neighbors_and_all_original_gates(monkeypatch):
    inputs = fixture(); frozen = deepcopy(inputs); snapshots = []
    evaluate = base._evaluate_arm_pose
    def observe(index, trial, sigma, derivatives=False, **kwargs):
        if derivatives: snapshots.append(kwargs['changes'].copy())
        return evaluate(index, trial, sigma, derivatives, **kwargs)
    monkeypatch.setattr(base, '_evaluate_arm_pose', observe)
    result = propose(*inputs); candidate, report = result['human_candidate'], result['report']
    assert inputs == frozen and result['status'] == 'unadmitted_construction_proposal'
    assert report['profile'] == restored.CONTRACT['method'] and report['base_contract'] == base.CONTRACT
    assert all(report['gates'].values()) and all(report['fixed_contracts'].values())
    assert len(candidate['reference']) == len(report['references']) == 6 and len(report['source_slots']) == 11
    assert candidate['reference'][0] == inputs[0]['reference'][0]
    assert report['changed_reference_poses'] >= 3
    assert any(np.count_nonzero(np.max(abs(snapshot), axis=1) > 1e-8) >= 3 for snapshot in snapshots)
    assert report['maximum_correction_rate_rad_s'] <= 1.5 + 1e-8
    assert report['maximum_endpoint_depth_change_model_units'] <= .08 + 1e-8
    assert report['geometry_summary']['worsened_pair_samples'] == 0
    assert report['numerical_summary']['direction_qp_calls'] <= 6 * 5
    assert not report['promoted'] and report['physics_steps'] == 0


def test_obstacle_constraints_survive_composed_pose_updates():
    spec, evidence, options = fixture()
    options['equipment_boxes'] = [{'id': 'later-obstacle', 'center': [.18, .425, 1.037], 'half_extents': [.015] * 3}]
    report = propose(spec, evidence, options)['report']
    before = np.array([r['before_gaps'] for r in report['geometry_checks']])
    after = np.array([r['after_gaps'] for r in report['geometry_checks']])
    assert np.any(before < -1e-8) and np.all(after >= np.minimum(0, before) - 1e-8)
    assert np.sum(np.maximum(0, -after) ** 2) < np.sum(np.maximum(0, -before) ** 2)
    assert report['changed_reference_poses'] >= 2 and report['geometry_summary']['new_equipment_penetrations'] == 0
    assert len(report['references']) == 6 and len(report['source_slots']) == 11
    assert report['gates']['pointwise_collision_nonregression']
    assert 'per_endpoint_confident_image_nonregression' in report['gates']


def test_collision_between_clear_references_cannot_be_dropped():
    spec, evidence, options = fixture()
    for i in (3, 4): spec['root_reference'][i]['translation'][0] += .6
    spec['reference'][3]['joints']['left_shoulder_lateral'] = -.6
    spec['reference'][4]['joints']['left_shoulder_lateral'] = .6
    options['human_xml_document'] = human_xml(spec).encode()
    geometry = _Geometry(spec, options['human_xml_document'], options['equipment_boxes'])
    angles = np.array([[r['joints'][name] for name in geometry.names] for r in spec['reference']])
    roots = np.array([r['translation'] for r in spec['root_reference']])
    geometry.set((angles[3] + angles[4]) / 2, (roots[3] + roots[4]) / 2)
    center = geometry.data.geom_xpos[geometry.model.geom('left_elbow_geom').id].tolist()
    options['equipment_boxes'] = [{'id': 'between-poses', 'center': center, 'half_extents': [.002] * 3}]
    report = propose(spec, evidence, options)['report']
    pair = next(i for i, p in enumerate(report['pair_columns']) if p['human_geom'] == 'left_elbow_geom' and p['object_id'] == 'between-poses')
    midpoint = next(row for row in report['geometry_checks'] if row['reference_indices'] == [3, 4])
    assert midpoint['before_gaps'][pair] < -.02
    assert all(row['before_gaps'][pair] > .1 for row in report['geometry_checks'] if row['source_reference_index'] in (3, 4))
    assert all(np.all(np.array(row['after_gaps']) >= np.minimum(0., row['before_gaps']) - 1e-8) for row in report['geometry_checks'])
    assert report['gates']['first_reference_frozen'] and report['gates']['fixed_contracts']


def test_weak_source_values_keep_labels_and_zero_image_weight():
    report = propose(*fixture(low_confidence=True))['report']
    point = next(p for row in report['references'] if row['frame'] == 4 for p in row['endpoints'] if p['joint'] == 'left_wrist')
    assert point['confidence'] == .2 and point['status'] == 'low_confidence' and point['fit_weight'] == 0
    assert point['correction_scope'] == 'inferred_geometry_depth_temporal_only_not_measured'
    assert report['per_joint_summary']['left_wrist']['all_finite']['count'] == 6
    assert report['per_joint_summary']['left_wrist']['confident']['count'] == 5


def test_withheld_measurements_never_change_the_full_proposal():
    first = propose(*fixture(withheld=True, secret_shift=0.))
    second = propose(*fixture(withheld=True, secret_shift=100.))
    assert first['human_candidate'] == second['human_candidate']
    row = next(row for row in first['report']['references'] if row['frame'] == 6)
    assert row['source_admission'] == 'withheld' and all(p['image_px'] is None and p['fit_weight'] == 0 for p in row['endpoints'])
    assert len(first['report']['source_slots']) == 11


def test_frozen_first_collision_abstains_without_solver_calls():
    spec, evidence, options = fixture()
    geometry = _Geometry(spec, options['human_xml_document'], options['equipment_boxes'])
    geometry.set([spec['reference'][0]['joints'][n] for n in geometry.names], spec['root_reference'][0]['translation'])
    options['equipment_boxes'][0]['center'] = geometry.data.geom_xpos[geometry.model.geom('left_elbow_geom').id].tolist()
    result = propose(spec, evidence, options); report = result['report']
    assert result['human_candidate'] == spec and result['status'] == 'rejected_construction_proposal'
    assert not report['gates']['no_remaining_equipment_penetration']
    assert report['numerical_summary']['direction_qp_calls'] == report['numerical_summary']['phase_one_calls'] == 0


def test_invalid_active_initial_point_is_reported_not_relabelled_infeasible():
    solver = restored._SequentialSolver(float('inf'))
    step, report = solver.project(np.eye(8), np.zeros(8), np.eye(8), [.1] * 8, cycles=48, tolerance=1e-10)
    assert not report['converged'] and report['status'] == 'unsupported_infeasible_initial'
    assert not report['global_infeasibility_claim'] and np.all(step == 0.)
    assert len(solver.diagnostics) == 1


def test_deadline_preserves_candidate_and_all_input_rows(monkeypatch):
    monkeypatch.setitem(base.CONTRACT, 'maximum_fit_seconds', 0.)
    spec, evidence, options = fixture(); result = propose(spec, evidence, options)
    assert result['status'] == 'partial_budget_exhausted' and result['human_candidate'] == spec
    assert not result['report']['gates']['complete_independent_audit']
    assert len(result['report']['references']) == 6 and len(result['report']['source_slots']) == 11
    assert result['report']['numerical_summary']['direction_qp_calls'] == 0


def test_every_reference_keeps_failed_solver_attempts(monkeypatch):
    actual = restored.solve_feasible_qp
    def exhausted(*args, **kwargs):
        result = actual(*args, **kwargs); result['status'] = 'iteration_limit'; result['step'] = None
        return result
    monkeypatch.setattr(restored, 'solve_feasible_qp', exhausted)
    spec, evidence, options = fixture(); result = propose(spec, evidence, options)
    assert result['human_candidate'] == spec
    assert len(result['report']['projections']) == 30
    assert all(not p['converged'] and p['status'] == 'iteration_limit' for p in result['report']['projections'])
    assert all(r['projection_failures'] == 6 for r in result['report']['references'][1:])


def curved_evaluator(*, decreasing_cost=True):
    # Opposing curved gaps require y=x²; returning to the baseline violates their tangent equality.
    sample = {'before_gaps': np.zeros(2), 'current_gaps': np.zeros(2)}
    def evaluate(trial, derivatives=False):
        x, y = trial[:2]; gaps = np.array([y - x * x, x * x - y])
        row = np.r_[-2 * x, 1., np.zeros(6)]
        matrix = np.vstack((row, -row, np.zeros((8, 8))))
        lower = np.r_[-gaps - base.CONTRACT['numerical_tolerance'], [-.08] * 8]
        cost = .5 * (x - .1) ** 2 if decreasing_cost else 1.
        return cost, np.empty(0), np.empty((0, 8)), bool(np.min(gaps) >= -1e-8), matrix.tolist(), lower.tolist(), [(sample, gaps)]
    return evaluate


def test_curvature_restore_uses_phase_one_when_return_witness_is_invalid():
    solver = restored._SequentialSolver(float('inf')); baseline = np.zeros(8); original = baseline.copy()
    result = solver.restore(index=3, sweep=1, sigma=.01, baseline=baseline, direction=np.r_[.1, np.zeros(7)],
        lower=np.full(8, -.1), upper=np.full(8, .1), baseline_cost=.005, evaluate=curved_evaluator(),
        affected_count=1, pair_count=2, deadline=float('inf'))
    assert result['trial'] is not None and np.array_equal(baseline, original)
    record = solver.restorations[0]
    assert record['status'] == 'accepted_restored_step' and record['rounds'][0]['phase_one']['iterations'] > 0
    assert record['rounds'][0]['linear_pair_tolerance_spending'] == 0.
    assert record['rounds'][0]['minimum_norm_qp']['status'] == 'optimal'
    x, y = result['trial'][:2]
    assert abs(y - x * x) <= 1e-8 and .5 * (x - .1) ** 2 < .005
    assert solver.phase_calls == solver.restoration_calls == 1


def test_valid_restoration_without_cost_decrease_stays_unaccepted_after_two_rounds():
    solver = restored._SequentialSolver(float('inf'))
    result = solver.restore(index=2, sweep=0, sigma=.01, baseline=np.zeros(8), direction=np.r_[.1, np.zeros(7)],
        lower=np.full(8, -.1), upper=np.full(8, .1), baseline_cost=1., evaluate=curved_evaluator(decreasing_cost=False),
        affected_count=1, pair_count=2, deadline=float('inf'))
    assert result['trial'] is None and len(solver.restorations[0]['rounds']) == 2
    assert all(not row['cost_decreased'] for row in solver.restorations[0]['rounds'])
    assert solver.phase_calls == solver.restoration_calls == 2


def test_restoration_deadline_keeps_partial_attempt_without_committing(monkeypatch):
    solver = restored._SequentialSolver(float('inf'))
    def expired(*args): raise base._BudgetExpired('restoration_fresh_nonlinear_check')
    check = base._check_deadline
    monkeypatch.setattr(base, '_check_deadline', lambda deadline, stage: expired() if stage == 'restoration_fresh_nonlinear_check' else check(deadline, stage))
    with pytest.raises(base._BudgetExpired):
        solver.restore(index=2, sweep=0, sigma=.01, baseline=np.zeros(8), direction=np.r_[.1, np.zeros(7)],
            lower=np.full(8, -.1), upper=np.full(8, .1), baseline_cost=.005, evaluate=curved_evaluator(),
            affected_count=1, pair_count=2, deadline=float('inf'))
    assert solver.restorations[0]['status'] == 'partial_budget_exhausted'
    assert len(solver.restorations[0]['rounds']) == 1 and solver.restoration_calls == 1


@pytest.mark.parametrize('phase', ['restored_output_serialization', 'restored_output_bound_check'])
def test_restored_output_deadline_keeps_completed_geometry_but_cannot_pass(monkeypatch, phase):
    original = base._check_deadline
    def check(deadline, stage):
        if stage == phase: raise base._BudgetExpired(stage)
        return original(deadline, stage)
    monkeypatch.setattr(base, '_check_deadline', check)
    result = propose(*fixture()); report = result['report']
    assert result['status'] == 'partial_budget_exhausted' and report['deadline_stage'] == phase
    assert not report['gates']['complete_bounded_solve'] and report['independent_audit_status'] == 'complete'
    assert len(report['references']) == 6 and len(report['source_slots']) == 11
    assert report['geometry_checks'] and not report['promoted']
