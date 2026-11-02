from copy import deepcopy
import math

import mujoco
import numpy as np
import pytest

from test_multiview_body import setup
from weft.human import HumanEnv
from weft.limb_position_fit import CONTRACT, fit_limb_positions, limb_geometry
from weft.multiview_body import make_two_view_body
from weft.spatial_pose import rotation_zyx


@pytest.mark.parametrize('sign', [-1, 1])
def test_analytic_derivatives_match_independent_central_differences(sign):
    rng = np.random.default_rng(7231)
    for _ in range(24):
        angles = np.r_[rng.uniform(-2, 2, 3), rng.uniform(.1, 2.5)]
        lengths = rng.uniform(.15, .6, 2)
        _, jac = limb_geometry(angles, lengths, sign, jacobian=True)
        numeric = np.stack([(limb_geometry(angles+axis, lengths, sign)[0]-limb_geometry(angles-axis, lengths, sign)[0])/2e-6
                            for axis in np.eye(4)*1e-6], axis=-1)
        np.testing.assert_allclose(jac, numeric, atol=3e-10, rtol=1e-7)


def test_analytic_chain_matches_native_body_with_translated_root_and_rotated_torso(tmp_path):
    root, predictions, first, second, *_ = setup(tmp_path)
    body, _ = make_two_view_body(root, predictions, first, second)
    env = HumanEnv(body)
    try:
        frame = 6; snapshot = env.reference_snapshot(frame)
        angles = body['reference'][frame]['joints']
        torso = np.array(body['body_frame']['rotation']) @ rotation_zyx([angles[k] for k in ('torso_twist', 'torso_lateral', 'torso_flex')])
        for side in ('left', 'right'):
            for proximal, middle, end, length_names, sign in [('hip', 'knee', 'ankle', ('thigh', 'shin'), -1),
                                                            ('shoulder', 'elbow', 'wrist', ('upper_arm', 'forearm'), 1)]:
                group = side+'_'+proximal
                q = [angles[k] for k in (group+'_twist', group+'_lateral', group, side+'_'+middle)]
                local, _ = limb_geometry(q, [body['lengths'][k] for k in length_names], sign)
                joints = snapshot['human']['joints']
                world = local @ torso.T+joints[group]
                np.testing.assert_allclose(world, [joints[side+'_'+middle], joints[side+'_'+end]], atol=1e-12)
    finally:
        env.close()


@pytest.mark.parametrize('sign', [-1, 1])
def test_variable_length_observations_improve_without_shortening_model(sign):
    angles = [.15, -.2, -.6, 1.2]
    lengths = [.39, .375]
    targets = limb_geometry(angles, [.32, .28], sign)[0]
    before = deepcopy((angles, lengths, targets.tolist()))
    result = fit_limb_positions(angles, lengths, targets, sign)
    assert result['position_sse_after_m2'] < result['position_sse_before_m2']*.75
    assert 0 < result['maximum_angle_change_rad'] <= .30
    points = np.array(result['after_parent_m'])
    np.testing.assert_allclose([np.linalg.norm(points[0]), np.linalg.norm(points[1]-points[0])], lengths, atol=1e-12)
    assert (angles, lengths, targets.tolist()) == before
    assert len(result['iterations']) <= CONTRACT['maximum_iterations']
    assert result['trial_evaluations'] <= CONTRACT['maximum_iterations']*CONTRACT['maximum_backtracks']
    assert not result['observations_modified'] and result['physical_validation'] == 'unverified'


def test_exact_source_targets_do_not_follow_an_unnecessary_temporal_prior():
    angles, lengths = [.3, -.2, .4, .8], [.3, .25]
    targets = limb_geometry(angles, lengths, 1)[0]
    result = fit_limb_positions(angles, lengths, targets, 1, previous=[.2, -.1, .3, -.2])
    assert result['output_angles_rad'] == angles and result['iterations'] == []
    assert result['position_sse_after_m2'] == 0


def test_unreachable_target_retains_nonzero_error_and_respects_absolute_joint_limits():
    original = [math.tau-.01, -.2, -math.tau+.01, .01]
    result = fit_limb_positions(original, [.3, .25], [[2, 0, 0], [3, 0, 0]], -1)
    angles = np.array(result['output_angles_rad'])
    assert np.max(abs(angles[:3])) <= math.tau and 0 <= angles[3] <= 2.6
    assert result['position_sse_after_m2'] > 1
    assert result['position_sse_after_m2'] <= result['position_sse_before_m2']
    assert result['maximum_angle_change_rad'] <= .30


def test_temporal_prior_changes_proposal_without_overriding_source_nonregression():
    angles, lengths = [.15, -.2, -.6, 1.2], [.39, .375]
    targets = limb_geometry(angles, [.32, .28], 1)[0]
    independent = fit_limb_positions(angles, lengths, targets, 1)
    temporal = fit_limb_positions(angles, lengths, targets, 1, previous=[-.2, .2, -.2, .2])
    assert temporal['output_angles_rad'] != independent['output_angles_rad']
    assert temporal['position_sse_after_m2'] <= temporal['position_sse_before_m2']


@pytest.mark.parametrize('fault', ['angle', 'length', 'zero_length', 'target', 'bend', 'sign', 'previous', 'shape'])
def test_invalid_input_never_produces_a_candidate(fault):
    values = dict(angles=[.1, .2, .3, .4], lengths=[.3, .25], targets=[[0, 0, -.3], [0, .1, -.5]], bend_sign=1)
    if fault == 'angle': values['angles'][0] = math.inf
    elif fault == 'length': values['lengths'][0] = 2.
    elif fault == 'zero_length': values['lengths'][0] = 0.
    elif fault == 'target': values['targets'][0][0] = math.nan
    elif fault == 'bend': values['angles'][3] = 2.7
    elif fault == 'sign': values['bend_sign'] = True
    elif fault == 'previous': values['previous'] = [.31]*4
    elif fault == 'shape': values['angles'].pop()
    with pytest.raises(ValueError):
        fit_limb_positions(**values)
