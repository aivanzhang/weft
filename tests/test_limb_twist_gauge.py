from copy import deepcopy
import math

import numpy as np
import pytest

from weft.constrained_stereo import _project
from weft.limb_position_fit import limb_geometry
from weft.limb_twist_gauge import (
    CONTRACT, axial_gauge, fit_limb_twist_gauge, gauge_derivative, solve_gauge_bands,
)
from weft.spatial_pose import rotation_zyx


def inputs(n=9, bend=0., error=0.):
    angles = np.zeros((n, 4)); angles[:, 0] = np.sin(np.arange(n)*2)*.7; angles[:, 3] = bend
    origins = np.tile([0., .1, 4.], (n, 1)); parents = np.tile(np.eye(3), (n, 1, 1)); observations = []
    for value, origin in zip(angles, origins, strict=True):
        points, _ = limb_geometry(value, [.4, .35], -1); points += origin; entries = []
        for view in range(2):
            camera = {'matrix': np.array([[700., 0., 640.], [0., 700., 480.], [0., 0., 1.]]),
                'rotation': np.eye(3), 'translation': np.array([view*.6, 0., 0.]),
                'radial': -.03, 'width': 1280, 'height': 960}
            pixels, _ = _project(points, camera)
            for joint in range(2):
                entries.append({'camera': deepcopy(camera), 'joint_index': joint, 'view_index': view,
                    'pixel': (pixels[joint]+[error, 0.]).tolist(), 'confidence': .9})
        observations.append(entries)
    return dict(angles=angles, lengths=[.4, .35], origins=origins, parent_rotations=parents,
        times=np.arange(n)*.08, observations=observations, bend_sign=-1,
        image_std_fraction=.0025, motion_acceleration_std_rad_s2=60., provenance='Analytic fixed assumptions')


@pytest.mark.parametrize('sign', [-1, 1])
@pytest.mark.parametrize('angles', [[.4, -.3, .2, .01], [5., 1.7, 5., .4], [-2., -1.2, 1., 2.4]])
def test_axial_gauge_preserves_upper_geometry_and_exact_distal_arc(sign, angles):
    original, _ = limb_geometry(angles, [.4, .35], sign)
    for theta in [-math.pi/2, -.17, 0., .31, math.pi/2]:
        proposed, rotation, _ = axial_gauge(angles, theta)
        points, _ = limb_geometry(proposed, [.4, .35], sign)
        np.testing.assert_allclose(rotation_zyx(proposed[:3]), rotation, atol=2e-15)
        np.testing.assert_allclose(points[0], original[0], atol=2e-15)
        assert proposed[3] == angles[3]
        assert np.linalg.norm(points[1]-original[1]) == pytest.approx(2*.35*math.sin(angles[3])*abs(math.sin(theta/2)), abs=2e-15)
        assert np.linalg.norm(points[1]-points[0]) == pytest.approx(.35, abs=2e-15)


@pytest.mark.parametrize('angles', [[.2, -.3, .7, .1], [4.2, 1.7, 4., .7], [-2., -1.1, 1., 2.]])
def test_gauge_chart_derivative_matches_independent_finite_difference(angles):
    theta = .13; value, rotation, _ = axial_gauge(angles, theta)
    before = axial_gauge(angles, theta-1e-6)[0]; after = axial_gauge(angles, theta+1e-6)[0]
    np.testing.assert_allclose(gauge_derivative(value, rotation), (after[:3]-before[:3])/2e-6, rtol=1e-7, atol=1e-8)


@pytest.mark.parametrize('n', [2, 3, 47, 900])
def test_gauge_banded_solver_matches_independent_residual(n):
    rng = np.random.default_rng(456); first = rng.uniform(-1., 1., n-1); second = rng.uniform(-1., 1., max(n-2, 0))
    diagonal = np.full(n, 5.); rhs = rng.normal(size=n)
    solution = solve_gauge_bands(diagonal, first, second, rhs)
    evaluated = diagonal*solution
    evaluated[:-1] += first*solution[1:]; evaluated[1:] += first*solution[:-1]
    evaluated[:-2] += second*solution[2:]; evaluated[2:] += second*solution[:-2]
    np.testing.assert_allclose(evaluated, rhs, atol=1e-12)


def test_exact_straight_limb_images_allow_coherent_unobserved_orientation():
    args = inputs(); original = deepcopy(args); fit = fit_limb_twist_gauge(**args)
    assert fit['after']['motion_acceleration_chi2'] < fit['before']['motion_acceleration_chi2']*1e-6
    assert max(abs(np.array(fit['correction_rad'])[:, 0])) > .3
    np.testing.assert_allclose(fit['after']['points_world_m'], fit['before']['points_world_m'], atol=1e-14)
    np.testing.assert_array_equal(args['angles'], original['angles'])
    assert fit['after']['image_chi2'] <= fit['before']['image_chi2']+1e-10
    assert fit['physical_validation'] == 'unverified' and not fit['observations_modified']


@pytest.mark.parametrize('bend', [.025, .4, 1.4])
def test_observed_endpoint_geometry_and_every_source_residual_stay_bounded(bend):
    args = inputs(bend=bend, error=4.)
    fit = fit_limb_twist_gauge(**args)
    assert fit['maximum_proximal_displacement_m'] < 1e-12
    assert fit['maximum_distal_displacement_m'] <= .02+1e-12
    assert fit['after']['image_chi2'] <= fit['before']['image_chi2']+1e-10
    assert fit['after']['objective'] <= fit['before']['objective']+1e-10
    for original, candidate in zip(fit['before']['errors_px'], fit['after']['errors_px'], strict=True):
        assert all(a <= b+1280*.0025+1e-10 for a, b in zip(candidate, original, strict=True))
    np.testing.assert_array_equal(np.array(fit['output_angles_rad'])[:, 3], args['angles'][:, 3])
    previous = np.zeros(len(args['angles']))
    for iteration in fit['iterations']:
        for trial in iteration['trials']:
            assert np.max(abs(np.array(trial['axial_rotation_rad'])-previous)) <= CONTRACT['maximum_update_rad']+1e-12
            if trial['accepted']:
                assert trial['objective'] < iteration['objective_before']; previous = np.array(trial['axial_rotation_rad'])


def test_missing_images_and_large_time_gaps_break_temporal_edges():
    args = inputs(6); args['times'] = [0., .04, .12, .20, .6, .68]; args['observations'][2] = []
    fit = fit_limb_twist_gauge(**args)
    assert fit['axial_rotation_rad'][2] == 0.
    assert np.count_nonzero(fit['motion_acceleration_stencil']) == 0
    assert fit['maximum_distal_displacement_m'] == 0.


def test_completely_missing_and_singular_observations_remain_unchanged():
    args = inputs(); args['observations'] = [[] for _ in args['observations']]
    fit = fit_limb_twist_gauge(**args)
    assert fit['status'] == 'unchanged_no_observations'
    np.testing.assert_array_equal(fit['output_angles_rad'], args['angles'])
    with pytest.raises(ValueError, match='singularity'):
        gauge_derivative([0., math.pi/2, 0., 0.], rotation_zyx([0., math.pi/2, 0.]))


def test_initially_singular_pose_keeps_its_exact_angles_and_geometry():
    args = inputs(); args['angles'][:, 1] = math.pi/2
    for angle, origin, rows in zip(args['angles'], args['origins'], args['observations'], strict=True):
        points, _ = limb_geometry(angle, args['lengths'], args['bend_sign']); points += origin
        for row in rows:
            pixels, _ = _project(points, row['camera']); row['pixel'] = pixels[row['joint_index']].tolist()
    fit = fit_limb_twist_gauge(**args)
    np.testing.assert_array_equal(fit['output_angles_rad'], args['angles'])
    assert fit['axial_rotation_rad'] == [0.]*len(args['angles'])


def test_bent_exact_images_reject_the_weak_twist_motion_hypothesis():
    args = inputs(bend=1.4)
    fit = fit_limb_twist_gauge(**args)
    np.testing.assert_array_equal(fit['output_angles_rad'], args['angles'])
    assert fit['axial_rotation_rad'] == [0.]*len(args['angles'])
    assert any(trial['rejection'] == 'complete_source_image_regression'
               for iteration in fit['iterations'] for trial in iteration['trials'])


@pytest.mark.parametrize('damage', ['time', 'gap', 'angle', 'hinge', 'parent', 'image', 'duplicate', 'confidence', 'provenance', 'uncertainty'])
def test_invalid_source_inputs_are_rejected(damage):
    args = inputs()
    if damage == 'time': args['times'][1] = 0.
    elif damage == 'gap': args['times'][1] = .001
    elif damage == 'angle': args['angles'][0, 0] = float('nan')
    elif damage == 'hinge': args['angles'][0, 3] = 3.
    elif damage == 'parent': args['parent_rotations'][0] = np.diag([-1., 1., 1.])
    elif damage == 'image': args['observations'][0][0]['pixel'][0] = -1.
    elif damage == 'duplicate': args['observations'][0].append(deepcopy(args['observations'][0][0]))
    elif damage == 'confidence': args['observations'][0][0]['confidence'] = .4
    elif damage == 'provenance': args['provenance'] = ''
    elif damage == 'uncertainty': args['image_std_fraction'] = True
    with pytest.raises(ValueError): fit_limb_twist_gauge(**args)


def test_deadline_stops_before_any_proposal(monkeypatch):
    args = inputs(); ticks = iter([0., 11.])
    monkeypatch.setattr('weft.limb_twist_gauge.time.monotonic', lambda: next(ticks))
    with pytest.raises(TimeoutError): fit_limb_twist_gauge(**args)
