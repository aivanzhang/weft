"""Analytic constraints distinguish hard pair preservation from a collision penalty."""

from copy import deepcopy
import json

import numpy as np
import pytest

from test_arm_equipment_refinement import fixture
from weft import constrained_arm_alignment as alignment
from weft.arm_equipment_refinement import _Geometry
from weft.human import human_xml


def propose(spec, evidence, options):
    return alignment.propose_constrained_arm_alignment(json.dumps(spec).encode(), evidence, **options)


def test_quadratic_projection_preserves_each_constraint_instead_of_total_cost():
    # The unconstrained optimum trades the first contact for the second.
    step, report = alignment._project_step(np.eye(2), np.array([2., -1.]),
                                          [[1., 0.], [0., 1.], [0., -1.]], [0., 0., -.4])
    assert report['converged']
    np.testing.assert_allclose(step, [0., .4], atol=1e-10)
    assert report['maximum_violation'] <= 1e-10


def test_projection_exhaustion_is_explicit_and_cannot_return_an_unsafe_step():
    step, report = alignment._project_step(np.eye(2), np.array([-1., -1.]),
                                          [[-1., 0.], [1., -1.]], [0., 0.], cycles=1)
    assert not report['converged'] and report['cycles'] == 1
    np.testing.assert_array_equal(step, np.zeros(2))


def test_nonlinear_check_catches_curvature_hidden_by_a_feasible_tangent():
    before = np.array([0., -.02, .04])
    assert alignment._nonregression(before, before, np.array([0., -.01, 0.]), 1e-8)
    # A tangent at contact can permit a step whose second-order gap is negative.
    assert not alignment._nonregression(before, before, np.array([-.001, -.01, .03]), 1e-8)
    assert not alignment._nonregression(before, np.array([0., -.01, .04]), np.array([0., -.015, .04]), 1e-8)


def test_analytic_image_fit_keeps_every_reference_joint_and_source_slot():
    spec, evidence, options = fixture()
    original = deepcopy((spec, evidence, options))
    result = propose(spec, evidence, options)
    candidate, report = result['human_candidate'], result['report']
    assert (spec, evidence, options) == original
    assert result['status'] == 'unadmitted_construction_proposal'
    assert all(report['gates'].values())
    assert candidate['reference'][0] == spec['reference'][0]
    assert len(candidate['reference']) == len(report['references']) == 6
    assert len(report['source_slots']) == 11
    assert 0 < report['changed_reference_poses'] <= 5
    assert all(report['fixed_contracts'].values())
    for metric in report['image_summary'].values(): assert metric['after'] < metric['before']
    assert report['geometry_summary']['worsened_pair_samples'] == 0
    assert report['geometry_summary']['new_equipment_penetrations'] == 0
    assert report['maximum_angle_correction_rad'] <= .3 + 1e-8
    assert report['maximum_endpoint_depth_change_model_units'] <= .08 + 1e-8
    assert report['maximum_correction_rate_rad_s'] <= 1.5 + 1e-8
    assert report['physics_steps'] == 0 and not report['promoted']
    assert report['geometry_admission'].startswith('unadmitted')
    assert {name for name, row in report['per_joint_summary'].items() if 'confident' in row} == {
        'left_elbow', 'left_wrist', 'right_elbow', 'right_wrist'}
    json.dumps(result, allow_nan=False)


def test_obstacle_penetration_cannot_be_traded_for_other_pairs_or_times():
    spec, evidence, options = fixture()
    options['equipment_boxes'] = [{'id': 'later-obstacle', 'center': [.18, .425, 1.037], 'half_extents': [.015] * 3}]
    result = propose(spec, evidence, options); report = result['report']
    before = np.array([r['before_gaps'] for r in report['geometry_checks']])
    after = np.array([r['after_gaps'] for r in report['geometry_checks']])
    assert np.any(before < -1e-8)
    assert np.all(after >= np.minimum(0, before) - 1e-8)
    assert np.sum(np.maximum(0, -after) ** 2) < np.sum(np.maximum(0, -before) ** 2)
    assert any(row['kind'] == 'control_grid' and len(row['reference_indices']) == 2 for row in report['geometry_checks'])
    assert report['geometry_summary']['worsened_pair_samples'] == 0
    assert report['gates']['first_reference_frozen'] and report['gates']['fixed_contracts']
    assert report['projections'] and all(p['cycles'] <= 48 for p in report['projections'])


def test_weak_values_retain_their_status_while_inferred_motion_can_change():
    spec, evidence, options = fixture(low_confidence=True)
    result = propose(spec, evidence, options); report = result['report']
    row = next(row for row in report['references'] if row['frame'] == 4)
    point = next(point for point in row['endpoints'] if point['joint'] == 'left_wrist')
    assert point['confidence'] == .2 and point['status'] == 'low_confidence' and point['fit_weight'] == 0
    assert point['correction_scope'] == 'inferred_geometry_depth_temporal_only_not_measured'
    assert row['proposal_status'] == 'changed_inferred_proposal'
    assert report['per_joint_summary']['left_wrist']['all_finite']['count'] == 6
    assert report['per_joint_summary']['left_wrist']['confident']['count'] == 5


def test_collision_only_between_two_clear_references_is_retained_in_constraints():
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
    endpoints = [row for row in report['geometry_checks'] if row['source_reference_index'] in (3, 4)]
    assert all(row['before_gaps'][pair] > .1 for row in endpoints)
    assert all(np.all(np.array(row['after_gaps']) >= np.minimum(0., row['before_gaps']) - 1e-8) for row in report['geometry_checks'])
    assert not report['frozen_infeasibility_witnesses']


def test_withheld_measurements_never_influence_proposal_and_are_not_dropped():
    one, two = fixture(withheld=True, secret_shift=0.), fixture(withheld=True, secret_shift=100.)
    first, second = propose(*one), propose(*two)
    assert first['human_candidate'] == second['human_candidate']
    row = next(row for row in first['report']['references'] if row['frame'] == 6)
    assert row['source_admission'] == 'withheld'
    assert all(p['fit_weight'] == 0 and p['image_px'] is None for p in row['endpoints'])
    assert len(first['report']['source_slots']) == 11


def test_frozen_initial_collision_retains_all_rows_and_reports_rejection():
    spec, evidence, options = fixture()
    geometry = _Geometry(spec, options['human_xml_document'], options['equipment_boxes'])
    geometry.set([spec['reference'][0]['joints'][n] for n in geometry.names], spec['root_reference'][0]['translation'])
    options['equipment_boxes'][0]['center'] = geometry.data.geom_xpos[geometry.model.geom('left_elbow_geom').id].tolist()
    result = propose(spec, evidence, options); report = result['report']
    assert result['human_candidate'] == spec
    assert result['status'] == 'rejected_construction_proposal'
    assert report['frozen_infeasibility_witnesses'] and not report['sweeps']
    assert not report['gates']['no_remaining_equipment_penetration']
    assert len(report['references']) == 6 and len(report['source_slots']) == 11


def test_expired_fit_is_complete_unadmitted_receipt_not_a_pass(monkeypatch):
    spec, evidence, options = fixture()
    monkeypatch.setitem(alignment.CONTRACT, 'maximum_fit_seconds', 0.)
    result = propose(spec, evidence, options)
    assert result['status'] == 'partial_budget_exhausted'
    assert result['human_candidate'] == spec
    assert result['report']['independent_audit_status'] == 'incomplete'
    assert result['report']['geometry_summary']['completed_samples'] == 0
    assert not result['report']['gates']['complete_bounded_solve']
    assert len(result['report']['references']) == 6 and len(result['report']['source_slots']) == 11
    assert all(row['independent_audit_status'] == 'incomplete' for row in result['report']['references'])


def test_frozen_xml_mismatch_rejects_before_geometry():
    spec, evidence, options = fixture(); options['human_xml_document'] += b' '
    with pytest.raises(ValueError, match='exactly bind'):
        propose(spec, evidence, options)


@pytest.mark.parametrize('raw', [b'{"profile":"exercise-v3","profile":"exercise-v3"}', b'{"value":NaN}', b'{"value":1e400}'])
def test_duplicate_or_nonfinite_human_json_is_rejected_before_geometry(raw, monkeypatch):
    _, evidence, options = fixture()
    monkeypatch.setattr(alignment, '_Geometry', lambda *a, **k: pytest.fail('Malformed JSON reached kinematics'))
    with pytest.raises(ValueError):
        alignment.propose_constrained_arm_alignment(raw, evidence, **options)


@pytest.mark.parametrize('phase', ['initial_samples', 'final_audit_samples', 'output_serialization', 'output_bound_check'])
def test_deadline_in_each_expensive_phase_retains_rows_without_passing_audit(monkeypatch, phase):
    spec, evidence, options = fixture()
    original = alignment._check_deadline
    calls = 0
    def check(deadline, stage):
        nonlocal calls
        if stage == phase:
            calls += 1
            if calls == (3 if stage.endswith('_samples') else 1): raise alignment._BudgetExpired(stage)
        return original(deadline, stage)
    monkeypatch.setattr(alignment, '_check_deadline', check)
    result = propose(spec, evidence, options); report = result['report']
    assert result['status'] == 'partial_budget_exhausted' and report['deadline_stage'] == phase
    assert len(result['human_candidate']['reference']) == len(report['references']) == 6
    assert len(report['source_slots']) == 11 and not report['gates']['complete_independent_audit']
    assert report['independent_audit_status'] == 'incomplete'
    assert report['image_summary'] == {} and not report['promoted']
    assert all(row['status'] == 'unavailable_incomplete_audit' for row in report['per_joint_summary'].values())
    if phase == 'final_audit_samples': assert report['geometry_summary']['completed_samples'] == 2
