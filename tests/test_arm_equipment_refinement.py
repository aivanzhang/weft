"""Analytic image targets and signed-distance checks for one bounded proxy-aware proposal."""

from copy import deepcopy
import json

import mujoco
import numpy as np
import pytest

from test_arm_reference_alignment import inputs
from weft import arm_equipment_refinement as refinement
from weft.human import human_xml


def fixture(**kwargs):
    spec, evidence, options = inputs(**kwargs)
    options.update(equipment_boxes=[{'id': 'far-box', 'center': [2., 2., .1], 'half_extents': [.1, .1, .1]}],
                   human_xml_document=human_xml(spec).encode())
    return spec, evidence, options


@pytest.mark.parametrize('gap', [.04, -.015])
def test_signed_distance_gradient_matches_central_difference_outside_and_inside_proxy(gap):
    spec, _, options = fixture()
    geometry = refinement._Geometry(spec, options['human_xml_document'], options['equipment_boxes'])
    angles = np.array([spec['reference'][1]['joints'][name] for name in geometry.names])
    angles[geometry.names.index('left_shoulder_lateral')] += .15
    root = np.array(spec['root_reference'][1]['translation'])
    geometry.set(angles, root)
    geom = geometry.model.geom('left_elbow_geom').id
    center = geometry.data.geom_xpos[geom].copy()
    axis = geometry.data.geom_xmat[geom].reshape(3, 3)[:, 2]
    # A tilted capsule has a unique closest cap; static geoms require a rebuilt model.
    center[0] += .027 + .1 + geometry.model.geom_size[geom, 1] * abs(axis[0]) + gap
    geometry = refinement._Geometry(spec, options['human_xml_document'],
                                   [{'id': 'test-box', 'center': center.tolist(), 'half_extents': [.1, .5, .5]}])
    geom = geometry.model.geom('left_elbow_geom').id
    box = geometry.model.geom('equipment_inspection_0').id
    geometry.set(angles, root)
    distance, jac = geometry.distance(geom, box, jacobian=True)
    assert (distance > 0) == (gap > 0)
    numeric = []
    for column in geometry.columns:
        plus, minus = angles.copy(), angles.copy(); plus[column] += 1e-6; minus[column] -= 1e-6
        geometry.set(plus, root); high = geometry.distance(geom, box)[0]
        geometry.set(minus, root); low = geometry.distance(geom, box)[0]
        numeric.append((high - low) / 2e-6)
    np.testing.assert_allclose(jac, numeric, atol=2e-5, rtol=0)


def test_candidate_preserves_fixed_contracts_and_passes_full_analytic_admission():
    spec, evidence, options = fixture()
    original = deepcopy((spec, evidence, options))
    candidate, audit = refinement.refine_arm_equipment(spec, evidence, **options)
    assert (spec, evidence, options) == original
    assert audit['status'] == 'admitted_construction_candidate'
    assert all(audit['gates'].values()) and all(audit['fixed_contracts'].values())
    assert audit['maximum_angle_correction_rad'] <= .3
    assert audit['maximum_endpoint_depth_change_model_units'] <= .08
    assert audit['maximum_correction_rate_rad_s'] <= 1.5 + 1e-10
    assert audit['first_configuration_maximum_error'] == 0
    assert candidate['reference'][0] == spec['reference'][0]
    assert len(audit['source_slots']) == 11 and len(audit['references']) == 6
    assert len(audit['geometry_checks']) == 17
    assert sum(len(row['human_object_distances']) for row in audit['geometry_checks']) == 17 * 12
    for result in audit['image_summary'].values(): assert result['after'] < result['before']
    assert candidate['arm_equipment_refinement']['geometric_admission'] is True
    assert audit['physics_steps'] == 0 and audit['promoted'] is False
    json.dumps(audit, allow_nan=False)


def test_frozen_initial_intersection_is_a_concrete_infeasibility_witness_not_a_fit():
    spec, evidence, options = fixture()
    geometry = refinement._Geometry(spec, options['human_xml_document'], options['equipment_boxes'])
    geometry.set([spec['reference'][0]['joints'][name] for name in geometry.names], spec['root_reference'][0]['translation'])
    center = geometry.data.geom_xpos[geometry.model.geom('left_elbow_geom').id].copy()
    options['equipment_boxes'][0]['center'] = center.tolist()
    candidate, audit = refinement.refine_arm_equipment(spec, evidence, **options)
    assert candidate['reference'] == spec['reference']
    assert audit['status'] == 'infeasible_frozen_first_reference'
    assert audit['sweeps'] == [] and audit['accepted_updates'] == 0
    assert any(w['human_geom'] == 'left_elbow_geom' for w in audit['frozen_infeasibility_witnesses'])
    assert not audit['gates']['no_remaining_equipment_penetration']
    assert not candidate['arm_equipment_refinement']['geometric_admission']


def test_geometry_penalty_reduces_later_collision_without_moving_box_or_first_pose():
    spec, evidence, options = fixture()
    options['equipment_boxes'] = [{'id': 'later-obstacle', 'center': [.18, .425, 1.037], 'half_extents': [.015] * 3}]
    boxes = deepcopy(options['equipment_boxes'])
    candidate, audit = refinement.refine_arm_equipment(spec, evidence, **options)
    assert not audit['frozen_infeasibility_witnesses']
    pairs = [p for r in audit['geometry_checks'] for p in r['human_object_distances']]
    before = sum(max(0, -p['before_gap_model_units']) ** 2 for p in pairs)
    after = sum(max(0, -p['after_gap_model_units']) ** 2 for p in pairs)
    assert before > 0 and after < before / 2
    assert options['equipment_boxes'] == boxes and audit['equipment_boxes'] == boxes
    assert candidate['reference'][0] == spec['reference'][0]
    assert audit['gates']['endpoint_depth_change'] and audit['gates']['correction_rate']
    assert audit['gates']['no_increased_floor_penetration'] and not audit['promoted']


def test_joint_limits_hold_even_when_analytic_target_requests_negative_elbow_angle():
    spec, evidence, options = fixture(near_limit=True)
    candidate, audit = refinement.refine_arm_equipment(spec, evidence, **options)
    assert all(row['joints']['left_elbow'] >= 0 for row in candidate['reference'])
    assert audit['gates']['joint_limits'] and audit['gates']['correction_rate']


def test_withheld_values_cannot_influence_proposal_and_geometry_only_changes_are_labelled():
    one = fixture(withheld=True, secret_shift=0.)
    two = fixture(withheld=True, secret_shift=100.)
    candidate, audit = refinement.refine_arm_equipment(one[0], one[1], **one[2])
    other, _ = refinement.refine_arm_equipment(two[0], two[1], **two[2])
    assert candidate['reference'] == other['reference']
    withheld = audit['references'][3]
    assert withheld['source_admission'] == 'withheld'
    assert all(p['fit_weight'] == 0 and p['image_px'] is None and 'no_image_measurement' in p['correction_scope']
               for p in withheld['endpoints'])
    assert len(candidate['reference']) == 6 and len(audit['source_slots']) == 11


def test_weak_wrist_never_gains_image_weight_or_new_measured_status():
    spec, evidence, options = fixture(low_confidence=True)
    _, audit = refinement.refine_arm_equipment(spec, evidence, **options)
    wrist = audit['references'][2]['endpoints'][1]
    assert wrist['confidence'] == .2 and wrist['status'] == 'low_confidence'
    assert wrist['fit_weight'] == 0 and 'no_image_measurement' in wrist['correction_scope']
    assert audit['references'][2]['endpoints'][0]['fit_weight'] == 1


@pytest.mark.parametrize('change', ['xml', 'camera', 'identity', 'clock', 'partition', 'stacked', 'box', 'duplicate'])
def test_mismatched_bindings_and_unsupported_geometry_are_rejected_before_fit(change):
    spec, evidence, options = fixture()
    if change == 'xml': options['human_xml_document'] += b' '
    elif change == 'camera': options['camera_hypothesis']['world_anchor_model_units'][2] += .01
    elif change == 'identity': options['human_entity_ref']['object_id'] = 'wrong'
    elif change == 'clock': spec['reference_evidence'][1]['time_s'] += .01
    elif change == 'partition':
        doc = json.loads(evidence); doc['allowed_partitions'].append('audit'); evidence = json.dumps(doc).encode()
    elif change == 'stacked': spec['arm_reference_alignment'] = {'status': 'rejected'}
    elif change == 'box': options['equipment_boxes'][0]['half_extents'][0] = -1.
    elif change == 'duplicate': options['equipment_boxes'] *= 2
    with pytest.raises(ValueError): refinement.refine_arm_equipment(spec, evidence, **options)


def test_budget_exhaustion_preserves_all_rows_and_prevents_admission(monkeypatch):
    spec, evidence, options = fixture()
    monkeypatch.setitem(refinement.CONTRACT, 'maximum_fit_seconds', 0.)
    candidate, audit = refinement.refine_arm_equipment(spec, evidence, **options)
    assert candidate['reference'] == spec['reference']
    assert audit['status'] == 'partial_budget_exhausted'
    assert not audit['gates']['complete_bounded_solve']
    assert len(audit['geometry_checks']) == 17 and audit['promoted'] is False


def test_all_finite_rmse_gate_includes_weak_endpoints_instead_of_only_optimized_points():
    rows = [{'endpoints': [{'image_px': [0., 0.], 'fit_weight': weight,
                           'error_before_px': old, 'error_after_px': new}
                          for weight, old, new in [(1, 10, 1), (1, 10, 1), (0, 1, 40), (0, 1, 40)]]}]
    summary = refinement._image_summary(rows)
    assert summary['paired_confident']['after'] < summary['paired_confident']['before']
    assert summary['all_finite']['after'] > summary['all_finite']['before']
