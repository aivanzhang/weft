from copy import deepcopy
from itertools import permutations

import cv2
import numpy as np
import pytest

from weft.multiray_pose import CONTRACT, triangulate_pixels
from weft.multiview_pose import _triangulate_rays


def cameras(count=3):
    values = []
    for center, axis in [([-.65, 0., .1], [0., -.1, .02]),
                         ([.65, 0., .15], [0., .1, -.02]),
                         ([0., -.45, .05], [.08, 0., .01]),
                         ([0., .45, .05], [-.08, 0., -.01])][:count]:
        R = cv2.Rodrigues(np.array(axis))[0]
        values.append({'matrix': [[800., 0., 360.], [0., 820., 640.], [0., 0., 1.]],
                       'radial': -.12, 'rotation': R.tolist(), 'translation': (-R @ center).tolist(),
                       'width': 720, 'height': 1280})
    return values


def pixels(point, views):
    return np.array([cv2.projectPoints(np.array(point, dtype=float)[None, :],
        cv2.Rodrigues(np.array(c['rotation']))[0], np.array(c['translation']), np.array(c['matrix']),
        np.array([c['radial'], 0., 0., 0., 0.]))[0].reshape(2) for c in views])


def ray_camera(c):
    K = np.array(c['matrix']); R = np.array(c['rotation']); t = np.array(c['translation'])
    return K, c['radial'], R, t, c['width'], c['height'], np.linalg.inv(K), -t @ R


@pytest.mark.parametrize('count', [2, 3, 4])
def test_exact_geometry_uses_every_camera(count):
    views = cameras(count); truth = [.21, -.34, 3.2]
    fit = triangulate_pixels(pixels(truth, views), views)
    assert fit['status'] == 'estimated' and fit['view_count'] == count
    assert np.allclose(fit['world_point_m'], truth, atol=1e-9)
    assert max(fit['reprojection_px']) < 1e-8
    assert fit['unit_pixel_sensitivity_m'] > 0


def test_three_views_improve_known_geometry_under_independent_noise():
    views = cameras(); rng = np.random.default_rng(20261029); old, new = [], []
    for _ in range(96):
        truth = rng.uniform([-.35, -.65, 2.4], [.35, .65, 4.5])
        observation = pixels(truth, views)+rng.normal(0, .75, (3, 2))
        baseline = _triangulate_rays(observation[:2], [ray_camera(c) for c in views[:2]])
        fit = triangulate_pixels(observation, views)
        assert baseline['status'] == fit['status'] == 'estimated'
        old.append(np.linalg.norm(np.array(baseline['world_point_m'])-truth))
        new.append(np.linalg.norm(np.array(fit['world_point_m'])-truth))
        assert fit['final_pixel_cost'] <= fit['initial_pixel_cost']
        assert fit['iterations'] <= CONTRACT['maximum_iterations']
        assert fit['trial_evaluations'] <= CONTRACT['maximum_iterations']*CONTRACT['maximum_backtracks']
    assert np.mean(np.square(new)) < np.mean(np.square(old))


def test_conflicting_third_observation_is_not_dropped():
    views = cameras(); observation = pixels([.1, -.2, 3.2], views)
    observation[2] += [80., 100.]
    fit = triangulate_pixels(observation, views)
    assert fit['status'] != 'estimated' and fit['world_point_m'] is None
    assert fit['view_count'] == 3 and len(fit['reprojection_px']) == 3
    assert fit['point_proposal_m'] is not None
    assert max(fit['reprojection_px']) > 12.8


def test_view_order_does_not_select_a_favorable_pair():
    views = cameras(); observation = pixels([.1, -.2, 3.2], views)+[[.8, -.2], [-.4, .9], [.1, .6]]
    base = triangulate_pixels(observation, views)
    for order in permutations(range(3)):
        fit = triangulate_pixels(observation[list(order)], [views[i] for i in order])
        assert fit['status'] == base['status']
        assert np.allclose(fit['world_point_m'], base['world_point_m'], atol=1e-10)
        assert np.allclose(fit['reprojection_px'], np.array(base['reprojection_px'])[list(order)], atol=1e-9)


def test_world_coordinate_change_preserves_image_solution():
    views = cameras(); observation = pixels([.1, -.2, 3.2], views)+[[.8, -.2], [-.4, .9], [.1, .6]]
    base = triangulate_pixels(observation, views)
    Q = cv2.Rodrigues(np.array([.2, -.3, .4]))[0]; shift = np.array([.5, 1., -.7])
    transformed = deepcopy(views)
    for c in transformed:
        R = np.array(c['rotation']) @ Q.T
        c['rotation'] = R.tolist(); c['translation'] = (np.array(c['translation'])-R @ shift).tolist()
    fit = triangulate_pixels(observation, transformed)
    assert fit['status'] == 'estimated'
    assert np.allclose(fit['world_point_m'], Q @ base['world_point_m']+shift, atol=1e-9)


def test_pixels_and_cameras_are_unchanged():
    views = cameras(); old = deepcopy(views); observation = pixels([.1, -.2, 3.2], views); original = observation.copy()
    triangulate_pixels(observation, views)
    assert views == old and np.array_equal(observation, original)


def test_duplicate_camera_rays_are_unavailable_not_fake_depth():
    view = cameras(2)[0]
    result = triangulate_pixels(pixels([.1, -.2, 3.2], [view, view]), [view, view])
    assert result['status'] == 'insufficient_ray_angle' and result['world_point_m'] is None


def test_noninvertible_distortion_is_explicit():
    views = cameras(2)
    for c in views:
        c['radial'] = -.9; c['matrix'][0][0] = c['matrix'][1][1] = 100.
    result = triangulate_pixels([[0., 0.], [0., 0.]], views)
    assert result['status'] == 'distortion_not_invertible'


@pytest.mark.parametrize('fault', ['missing', 'nan', 'outside', 'too_many', 'one_view', 'bad_rotation'])
def test_invalid_inputs_fail_without_estimates(fault):
    views = cameras(); observation = pixels([.1, -.2, 3.2], views)
    if fault == 'missing': observation = observation[:2]
    if fault == 'nan': observation[0, 0] = np.nan
    if fault == 'outside': observation[0, 0] = 721.
    if fault == 'too_many': views += views[:2]; observation = np.concatenate([observation, observation[:2]])
    if fault == 'one_view': views = views[:1]; observation = observation[:1]
    if fault == 'bad_rotation': views[0]['rotation'][0][0] = 2.
    with pytest.raises(ValueError):
        triangulate_pixels(observation, views)
