"""Actual-motion priors must preserve source images, time, bounds and missing slots."""

from copy import deepcopy

import numpy as np
import pytest

from test_joint_pixel_trajectory import SETTINGS, inputs
from weft.joint_pixel_trajectory import (acceleration_stencil, fit_joint_trajectory,
    solve_banded_blocks, trajectory_contract, validate_uncertainty)


MOTION = {**SETTINGS, 'schema_version': 2, 'motion_acceleration_std_rad_s2': 60.}


@pytest.mark.parametrize('n', [2, 3, 9])
def test_two_band_solver_matches_independent_dense_system(n):
    rng = np.random.default_rng(723)
    data = rng.normal(size=(n, 4, 4)); diagonal = data.transpose(0, 2, 1) @ data+5*np.eye(4)
    first, second = rng.uniform(-.3, .3, n-1), rng.uniform(-.3, .3, max(0, n-2))
    rhs = rng.normal(size=(n, 4)); dense = np.zeros((4*n, 4*n))
    for i in range(n): dense[4*i:4*i+4, 4*i:4*i+4] = diagonal[i]
    for offset, values in [(1, first), (2, second)]:
        for i, value in enumerate(values):
            j = i+offset
            dense[4*i:4*i+4, 4*j:4*j+4] = value*np.eye(4)
            dense[4*j:4*j+4, 4*i:4*i+4] = value*np.eye(4)
    before = (diagonal.copy(), first.copy(), second.copy(), rhs.copy())
    result = solve_banded_blocks(diagonal, first, second, rhs)
    np.testing.assert_allclose(result, np.linalg.solve(dense, rhs.ravel()).reshape(n, 4), atol=1e-12)
    for original, value in zip(before, (diagonal, first, second, rhs)): np.testing.assert_array_equal(original, value)


def test_full_size_two_band_solver_has_correct_residual_without_dense_matrix():
    n = 900; diagonal = np.repeat((8*np.eye(4))[None], n, axis=0)
    first, second = np.full(n-1, -2.), np.full(n-2, .5)
    rhs = np.cos(np.arange(4*n).reshape(n, 4)/9)
    result = solve_banded_blocks(diagonal, first, second, rhs)
    actual = 8*result
    actual[:-1] += first[:, None]*result[1:]; actual[1:] += first[:, None]*result[:-1]
    actual[:-2] += second[:, None]*result[2:]; actual[2:] += second[:, None]*result[:-2]
    np.testing.assert_allclose(actual, rhs, atol=1e-12)


def test_nonuniform_time_uses_actual_acceleration_not_correction_speed():
    times = np.array([1., 1.04, 1.12, 1.24, 1.4])
    stencil = acceleration_stencil(np.diff(times), np.ones(4, bool), 60.)
    for values, expected in [(times*3+2, 0.), (times**2*3, .1)]:
        result = sum(stencil[:, j]*values[j:len(times)-2+j] for j in range(3))
        np.testing.assert_allclose(result, expected, atol=1e-12)


def test_missing_images_and_long_gaps_break_both_acceleration_neighbors():
    args, _ = inputs(n=7)
    args['times'] = np.array([0., .08, .16, .24, .64, .72, .8])
    args['observations'][1] = []; args['uncertainty'] = MOTION
    result = fit_joint_trajectory(**args)
    stencil = np.array(result['motion_acceleration_stencil'])
    assert np.any(stencil[-1]) and not np.any(stencil[:-1])
    assert result['correction_rad'][1] == [0.]*4
    assert result['after']['errors_px'][1] == []
    np.testing.assert_array_equal(result['source_times_s'], args['times'])


@pytest.mark.parametrize('views', [1, 2, 3, 4])
@pytest.mark.parametrize('sign', [-1, 1])
def test_motion_prior_reduces_jitter_without_discarding_image_constraints(views, sign):
    args, _ = inputs(n=8, views=views, sign=sign)
    args['angles'][::2] += [.10, -.11, .09, .08]
    original = deepcopy(args); legacy = fit_joint_trajectory(**args)
    args['uncertainty'] = MOTION; result = fit_joint_trajectory(**args)
    def acceleration(values):
        return np.diff(np.diff(values, axis=0)/.08, axis=0)/.08
    assert np.sum(acceleration(result['output_angles_rad'])**2) < np.sum(acceleration(legacy['output_angles_rad'])**2)
    assert result['after']['motion_acceleration_chi2'] < result['before']['motion_acceleration_chi2']
    assert result['after']['image_chi2'] <= result['before']['image_chi2']
    assert result['after']['objective'] < result['before']['objective']
    assert result['method'] == 'time-coupled-joint-pixels-v2'
    assert result['maximum_angle_change_rad'] <= .3
    np.testing.assert_array_equal(args['angles'], original['angles'])
    for label in ('before', 'after'):
        assert all(len(row) == 2*views for row in result[label]['errors_px'])
    for iteration in result['iterations']:
        for trial in iteration['trials']:
            if trial['accepted']: assert trial['image_chi2'] <= result['before']['image_chi2']+1e-10


def test_exact_images_are_not_changed_to_satisfy_a_smoothing_prior():
    args, _ = inputs(); args['angles'] -= [.12, -.10, .14, -.15]
    args['uncertainty'] = MOTION
    result = fit_joint_trajectory(**args)
    assert result['status'] == 'unchanged_exact_images' and result['maximum_angle_change_rad'] == 0.
    args['observations'] = [[] for _ in args['observations']]
    result = fit_joint_trajectory(**args)
    assert result['status'] == 'unchanged_no_observations' and result['maximum_angle_change_rad'] == 0.


def test_motion_descent_cannot_trade_away_the_original_image_loss():
    from weft.constrained_stereo import _project
    from weft.limb_position_fit import limb_geometry
    args, _ = inputs()
    target = args['angles'] - [.12, -.10, .14, -.15]
    target[::2, 0] += .2
    for i, rows in enumerate(args['observations']):
        points, _ = limb_geometry(target[i], args['lengths'], args['bend_sign'])
        points = points @ args['parent_rotations'][i].T + args['origins'][i]
        for row in rows: row['pixel'] = _project(points, row['camera'])[0][row['joint_index']].tolist()
    args['angles'] = target + 1e-5
    args['uncertainty'] = {**MOTION, 'motion_acceleration_std_rad_s2': 1.}
    result = fit_joint_trajectory(**args)
    assert result['maximum_angle_change_rad'] == 0.
    assert result['status'] == 'no_descent'
    assert all(t['rejection'] == 'source_image_loss_regression' for it in result['iterations'] for t in it['trials'])


@pytest.mark.parametrize('value', [True, 0, -1, 1e5, float('nan'), '60'])
def test_motion_uncertainty_must_be_explicit_finite_and_bounded(value):
    with pytest.raises(ValueError): validate_uncertainty({**MOTION, 'motion_acceleration_std_rad_s2': value})


def test_profile_versions_cannot_silently_add_or_remove_the_motion_prior():
    with pytest.raises(ValueError): validate_uncertainty({**SETTINGS, 'motion_acceleration_std_rad_s2': 60.})
    with pytest.raises(ValueError): validate_uncertainty({**SETTINGS, 'schema_version': 2})
    assert trajectory_contract(validate_uncertainty(SETTINGS))['method'].endswith('v1')
    assert trajectory_contract(validate_uncertainty(MOTION))['method'].endswith('v2')
