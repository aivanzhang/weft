from copy import deepcopy

import cv2
import numpy as np
import pytest

from weft.constrained_stereo import _camera, _project, fit_stereo_lengths


PAIRS = ((0, 1), (6, 7), (0, 2), (2, 4), (1, 3), (3, 5), (6, 8), (8, 10), (7, 9), (9, 11))


def scene(phase=0.):
    points = np.array([[-.18, -.5, 3.], [.18, -.5, 3.], [-.43, -.28, 3.05], [.43, -.28, 3.05],
        [-.57, -.03, 3.12], [.57, -.03, 3.12], [-.12, 0., 3.], [.12, 0., 3.],
        [-.12, .42, 3.02], [.12, .42, 3.02], [-.15, .8, 3.08], [.15, .8, 3.08]])
    for root, middle, end, sign in ((0, 2, 4, 1), (1, 3, 5, -1), (6, 8, 10, -1), (7, 9, 11, 1)):
        R = cv2.Rodrigues(np.array([.35*np.sin(phase+sign), .2*np.cos(phase), 0.]))[0]
        original_middle = points[middle].copy()
        points[middle] = points[root]+R @ (points[middle]-points[root])
        lower = cv2.Rodrigues(np.array([.3*np.cos(phase), .1*np.sin(phase), 0.]))[0]
        points[end] = points[middle]+lower @ (points[end]-original_middle)
    K = np.array([[800., 0, 360.], [0, 800., 640.], [0, 0, 1.]])
    cameras = []
    for center, rvec in ((np.zeros(3), np.zeros(3)), (np.array([.5, 0., 0.]), np.array([.02, .1, -.04]))):
        R = cv2.Rodrigues(rvec)[0]
        cameras.append({'matrix': K.tolist(), 'rotation': R.tolist(), 'translation': (-R @ center).tolist(),
                        'radial': -.12, 'width': 720, 'height': 1280})
    constraints = []
    for a, b in PAIRS:
        coefficients = np.zeros(12); coefficients[a] = 1; coefficients[b] = -1
        constraints.append({'name': f'{a}-{b}', 'coefficients': coefficients.tolist(),
                            'length_m': float(np.linalg.norm(points[a]-points[b]))})
    coefficients = np.zeros(12); coefficients[[0, 1]] = .5; coefficients[[6, 7]] = -.5
    constraints.append({'name': 'torso_midpoints', 'coefficients': coefficients.tolist(),
                        'length_m': float(np.linalg.norm(coefficients @ points))})
    return points, cameras, constraints


def project_opencv(points, camera):
    return cv2.projectPoints(points, cv2.Rodrigues(np.array(camera['rotation']))[0],
        np.array(camera['translation']), np.array(camera['matrix']), np.array([camera['radial'], 0., 0., 0., 0.]))[0].reshape(-1, 2)


def observations(points, cameras):
    return np.array([project_opencv(points, c) for c in cameras])


def test_exact_geometry_is_preserved_without_a_coordinate_convention():
    points, cameras, constraints = scene()
    pixels = observations(points, cameras); original = pixels.copy(); inputs = deepcopy((cameras, constraints))
    fit = fit_stereo_lengths(pixels, cameras, constraints)
    assert fit['status'] == 'converged' and fit['length_constraints_satisfied']
    assert max(np.linalg.norm(np.array(fit['point_proposals_m'])-points, axis=1)) < 1e-9
    assert np.array_equal(pixels, original) and (cameras, constraints) == inputs
    assert fit['observations_modified'] is False and fit['physical_validation'] == 'unverified'


def test_constraints_reduce_known_3d_error_across_noisy_articulated_motion():
    rng = np.random.default_rng(37); errors = []; success = []
    for phase in np.linspace(0., 2*np.pi, 24, endpoint=False):
        points, cameras, constraints = scene(phase)
        pixels = observations(points, cameras)+rng.normal(0., 2., (2, 12, 2))
        fit = fit_stereo_lengths(pixels, cameras, constraints)
        errors.append([np.mean(np.sum((np.array(fit[k])-points)**2, axis=1))
                       for k in ('initial_points_m', 'point_proposals_m')])
        success.append(fit['length_constraints_satisfied'])
        assert fit['trial_evaluations'] <= 320
        assert max(fit['displacements_m']) <= .25
    assert all(success)
    errors = np.sqrt(np.mean(errors, axis=0))
    assert errors[1] < errors[0]*.95, errors


def test_projection_and_jacobian_match_independent_calculation():
    points, cameras, _ = scene(.7)
    camera = _camera(cameras[1]); pixel, jac = _project(points, camera)
    assert np.allclose(pixel, project_opencv(points, cameras[1]), atol=1e-10, rtol=0)
    for coordinate in range(3):
        step = np.zeros_like(points); step[:, coordinate] = 1e-6
        numeric = (project_opencv(points+step, cameras[1])-project_opencv(points-step, cameras[1]))/2e-6
        assert np.allclose(jac[:, :, coordinate], numeric, atol=1e-5, rtol=0)


def test_rigid_world_coordinate_change_preserves_solution():
    points, cameras, constraints = scene(.6)
    rng = np.random.default_rng(91)
    pixels = observations(points, cameras)+rng.normal(0., 1., (2, 12, 2))
    fit = fit_stereo_lengths(pixels, cameras, constraints)
    Q = cv2.Rodrigues(np.array([.2, -.3, .4]))[0]; shift = np.array([.5, 1., -.7])
    transformed = deepcopy(cameras)
    for c in transformed:
        R = np.array(c['rotation']) @ Q.T
        c['rotation'] = R.tolist(); c['translation'] = (np.array(c['translation'])-R @ shift).tolist()
    moved = fit_stereo_lengths(pixels, transformed, constraints)
    expected = np.array(fit['point_proposals_m']) @ Q.T+shift
    assert np.max(np.linalg.norm(expected-np.array(moved['point_proposals_m']), axis=1)) < 1e-5
    assert moved['length_constraints_satisfied'] == fit['length_constraints_satisfied']


def test_incompatible_lengths_are_not_silently_accepted():
    points, cameras, constraints = scene(); constraints[1]['length_m'] = 1.8
    fit = fit_stereo_lengths(observations(points, cameras), cameras, constraints)
    assert not fit['length_constraints_satisfied']
    assert max(fit['displacements_m']) <= .25


def test_two_joint_problem_retains_exact_length():
    points, cameras, _ = scene(); pair = points[[6, 7]]
    constraints = [{'name': 'hip_width', 'coefficients': [1., -1.], 'length_m': .24}]
    fit = fit_stereo_lengths(observations(pair, cameras), cameras, constraints)
    assert fit['length_constraints_satisfied']
    assert np.allclose(fit['point_proposals_m'], pair, atol=1e-9)


@pytest.mark.parametrize('change', ['missing_pixel', 'outside_image', 'too_many_joints', 'bad_rotation', 'bad_depth', 'tiny_baseline', 'duplicate_length', 'noninvariant_length', 'invalid_length'])
def test_invalid_or_unsupported_geometry_fails_explicitly(change):
    points, cameras, constraints = scene(); pixels = observations(points, cameras)
    if change == 'missing_pixel': pixels[0, 0, 0] = np.nan
    if change == 'outside_image': pixels[0, 0, 0] = -1
    if change == 'too_many_joints': pixels = np.concatenate([pixels, pixels[:, :1]], axis=1)
    if change == 'bad_rotation': cameras[0]['rotation'][0][0] = 2.
    if change == 'bad_depth': cameras[1]['translation'] = [0., 0., 1e20]
    if change == 'tiny_baseline': cameras[1] = deepcopy(cameras[0]); pixels[1] = pixels[0]
    if change == 'duplicate_length': constraints[1]['coefficients'] = constraints[0]['coefficients']
    if change == 'noninvariant_length': constraints[1]['coefficients'][0] = .5
    if change == 'invalid_length': constraints[0]['length_m'] = True
    with pytest.raises(ValueError):
        fit_stereo_lengths(pixels, cameras, constraints)
