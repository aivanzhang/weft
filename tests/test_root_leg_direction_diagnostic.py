from hashlib import sha256
import json
import time

import numpy as np
import pytest

from test_native_reference_inputs import views
from weft import root_leg_direction_diagnostic as diagnostic
from weft import root_leg_source_fit as source_fit


def test_simultaneous_direction_exists_when_separate_coordinate_blocks_cannot_move():
    image = np.array([[1., 0., 0., -1., 0., 0.], [0., 1., 0., 0., -1., 0.], [0., 0., 1., 0., 0., -1.]])
    floor = np.array([[0., 0., 0., 1., 0., 0.]])
    constraints = np.r_[image, -image, floor]
    result = diagnostic.analyze_linear_directions(image, floor, constraints)
    direction = np.array(result['direction_normalized_coordinates'])
    assert result['status'] == 'first_order_subspace_witness'
    assert direction[0] > .1 and direction[3] == pytest.approx(direction[0])
    assert np.max(abs(image @ direction)) < 1e-10
    assert np.min(constraints @ direction) >= -1e-10
    assert result['observability']['rank'] == 3
    assert result['observability']['nullity'] == 3
    assert not result['finite_step_validated'] and not result['global_infeasibility_claim']
    assert result['qp']['report']['variables'] <= 8


def test_data_only_rank_never_uses_floor_rows_or_qp_regularization():
    image = np.zeros((12, 6)); image[:, 0] = np.arange(12)
    result = diagnostic.analyze_linear_directions(image, np.eye(6), np.empty((0, 6)))
    assert result['observability']['rank'] == 1
    assert result['observability']['nullity'] == 5
    assert result['observability']['condition'] is None
    assert result['observability']['regularization_used'] is False


def test_conflicting_tangent_constraints_never_claim_global_infeasibility():
    floor = np.array([[0., 0., 0., 1., 0., 0.]])
    result = diagnostic.analyze_linear_directions(np.eye(6), floor, np.r_[floor, -floor])
    assert result['status'] == 'no_verified_subspace_witness'
    assert not result['global_infeasibility_claim']
    assert result['maximum_linear_violation'] <= 1e-8
    assert abs(result['summed_floor_direction_gain']) <= 1e-8


def test_matrix_work_bounds_and_deadline():
    with pytest.raises(ValueError, match='bounded'):
        diagnostic.analyze_linear_directions(np.zeros((10, 705)), np.zeros((1, 705)), np.zeros((1, 705)))
    with pytest.raises(ValueError, match='finite'):
        diagnostic.analyze_linear_directions(np.full((10, 6), np.nan), np.zeros((1, 6)), np.zeros((1, 6)))
    with pytest.raises(TimeoutError):
        diagnostic.analyze_linear_directions(np.eye(6), np.eye(6), np.empty((0, 6)), deadline=time.monotonic()-1)


def test_positive_dependency_restriction_retains_a_verified_simultaneous_direction():
    C = np.array([[1.,0.,0.], [-.5,1.,0.], [-.5,-1.,0.], [0.,0.,1.]])
    result = diagnostic.diagnose_dependent_cone(C, np.array([1.,.2,1.]))
    assert result['status'] == 'first_order_subspace_witness'
    direction = np.array(result['direction'])
    assert np.min(C @ direction) >= -1e-8
    assert np.linalg.norm(direction) == pytest.approx(1.)
    assert direction[2] > .99999
    assert result['all_original_constraints_verified']
    assert not result['finite_step_validated'] and not result['global_infeasibility_claim']


def test_small_infeasible_direction_cannot_pass_by_scaling_below_tolerance():
    C = np.array([[1.,0.],[-1.,0.],[0.,1.],[0.,-1.]])
    result = diagnostic.diagnose_dependent_cone(C, np.array([1.,2.]))
    assert result['status'] != 'first_order_subspace_witness'
    assert result['objective_direction_gain'] == 0.


def test_saved_cone_rejects_excessive_input():
    with pytest.raises(ValueError): diagnostic.diagnose_dependent_cone(np.eye(9), np.ones(9))
    with pytest.raises(ValueError): diagnostic.diagnose_dependent_cone(np.full((2,2),np.nan), np.ones(2))


def test_wide_image_matrix_retains_implicit_data_nullspace():
    image = np.eye(3, 6)
    rank, vectors = diagnostic._rank(image)
    assert rank['rank'] == 3 and rank['nullity'] == 3
    assert vectors.shape == (6, 6)
    np.testing.assert_allclose(image @ vectors[-2:].T, 0., atol=1e-15)


def test_near_null_search_rows_are_still_verified_in_original_units():
    C = np.array([[0.,0.,1.], [1e-17,0.,0.], [-1e-17,0.,0.]])
    result = diagnostic.diagnose_dependent_cone(C, np.array([1.,0.,1.]))
    assert result['status'] == 'first_order_subspace_witness'
    assert result['rounds'][0]['near_null_search_rows'] == [1, 2]
    assert len(result['directional_constraint_values']) == 3
    assert 0 < result['maximum_linear_violation'] < 1e-12
    assert np.linalg.norm(result['direction']) == pytest.approx(1.)


def test_redundant_floor_combinations_never_remove_original_witness_guards(monkeypatch):
    C = np.array([[1.,0.,0.], [0.,1.,0.], [.31748775589648115,.4861788656880651,0.],
                  [.5,.7,0.], [0.,0.,1.]])
    original = diagnostic.solve_feasible_qp
    calls = []
    def first_dependency(*args, **kwargs):
        calls.append(True)
        if len(calls) == 1:
            return {'status':'dependent_blocker', 'step':None,
                    'report':{'active_constraints':[0,1], 'dependent_blockers':[2]}}
        return original(*args, **kwargs)
    monkeypatch.setattr(diagnostic, 'solve_feasible_qp', first_dependency)
    result = diagnostic.diagnose_dependent_cone(C, np.array([-.3,-.2,1.]))
    assert result['status'] == 'first_order_subspace_witness'
    assert len(result['directional_constraint_values']) == len(C)
    assert np.min(C @ np.array(result['direction'])) >= -1e-8
    assert result['all_original_constraints_verified']
    assert result['rounds'][0]['redundancy']['excluded_search_rows'] == [2, 3]
    json.dumps(result, allow_nan=False)


def original_candidate(args):
    human, times, angles, roots, _, _, _, rotation, center, K = source_fit._prepare(*args)
    return {'profile': source_fit.PROFILE, 'contract': source_fit.CONTRACT,
        'bindings': dict(zip(('human', 'original_evidence', 'derived_evidence', 'dimensions'),
                            (sha256(raw).hexdigest() for raw in args))),
        'camera': {'rotation':rotation.tolist(), 'rotation_original':rotation.tolist(), 'center':center.tolist(), 'K':K.tolist()},
        'candidate': {'reference':human['reference'], 'root_reference':human['root_reference']}}


def test_real_kinematics_diagnostic_preserves_native_inputs_without_fitting(tmp_path):
    args = views(tmp_path); original = tuple(args)
    candidate = json.dumps(original_candidate(args)).encode()
    result = diagnostic.diagnose_root_leg_directions(*args, candidate)
    assert args == original
    human = json.loads(args[0])
    assert result['source_times'] == [r['time_s'] for r in human['reference']]
    assert result['source_times'][-1] in result['sample_times']
    assert result['geometry_calls'] == 2 * len(result['sample_times'])
    assert result['bindings']['candidate_sha256'] == sha256(candidate).hexdigest()
    assert result['rollouts'] == result['inference_calls'] == 0
    assert result['finite_step_validated'] is False
    assert not any(c['kind'] == 'root' and c['source_index'] == 0 for c in result['columns'])
    assert all('hip' in c['name'] or 'knee' in c['name'] for c in result['columns'] if c['kind'] == 'joint')


@pytest.mark.parametrize('change', ['clock','protected_angle','binding','gauge','root','joint_bound'])
def test_candidate_cannot_override_source_contract(tmp_path, change):
    args = views(tmp_path); candidate = original_candidate(args)
    if change == 'clock': candidate['candidate']['reference'][1]['time_s'] += .001
    elif change == 'protected_angle': candidate['candidate']['reference'][1]['joints']['torso_flex'] += .001
    elif change == 'binding': candidate['bindings']['human'] = '0' * 64
    elif change == 'gauge': candidate['camera']['K'][0][0] += 1
    elif change == 'root': candidate['candidate']['root_reference'][0]['translation'][2] += .0001
    else: candidate['candidate']['reference'][1]['joints']['left_knee'] = -.01
    with pytest.raises(ValueError):
        diagnostic.diagnose_root_leg_directions(*args, json.dumps(candidate).encode())
