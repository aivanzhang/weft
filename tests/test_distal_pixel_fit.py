from copy import deepcopy

import numpy as np
import pytest

from weft.constrained_stereo import _project
from weft.distal_pixel_fit import CONTRACT, distal_geometry, fit_distal_pixels
from weft.spatial_pose import rotation_zyx


def camera(radial=0., offset=0.):
    return {'matrix': np.array([[700., 3., 640.], [0., 710., 480.], [0., 0., 1.]]),
        'rotation': rotation_zyx([.05, -.1, .02]), 'translation': np.array([offset, .1, .2]),
        'radial': radial, 'width': 1280, 'height': 960}


@pytest.mark.parametrize('sign', [-1, 1])
@pytest.mark.parametrize('radial', [-.1, 0., .2])
def test_world_and_pixel_jacobians_match_independent_differences(sign, radial):
    rng = np.random.default_rng(1803)
    for _ in range(15):
        orientation = rotation_zyx(rng.uniform(-2, 2, 3)); knee = np.array([.1, -.2, 5.])
        delta = rng.uniform(-.3, .3, 2); bend = rng.uniform(.4, 2.2); view = camera(radial)
        point, jac = distal_geometry(delta, orientation, bend, .35, knee, sign)
        projected = _project(point[None, :], view)
        world_numeric, pixel_numeric = [], []
        for axis in np.eye(2)*1e-6:
            a, _ = distal_geometry(delta+axis, orientation, bend, .35, knee, sign)
            b, _ = distal_geometry(delta-axis, orientation, bend, .35, knee, sign)
            world_numeric.append((a-b)/2e-6)
            pixel_numeric.append((_project(a[None, :], view)[0][0]-_project(b[None, :], view)[0][0])/2e-6)
        np.testing.assert_allclose(jac, np.array(world_numeric).T, atol=5e-10, rtol=1e-6)
        np.testing.assert_allclose(projected[1][0] @ jac, np.array(pixel_numeric).T, atol=1e-7, rtol=1e-6)


@pytest.mark.parametrize('count', [1, 2, 4])
@pytest.mark.parametrize('sign', [-1, 1])
def test_fit_uses_all_views_without_changing_length_or_supported_axis(count, sign):
    rotation = rotation_zyx([.3, -.2, .7]); knee = np.array([.1, .2, 4.])
    cameras = [camera(.1, .5*i) for i in range(count)]
    target, _ = distal_geometry([.15, -.2], rotation, 1., .35, knee, sign)
    pixels = [_project(target[None, :], c)[0][0].tolist() for c in cameras]
    original = deepcopy(pixels)
    fit = fit_distal_pixels(rotation, 1., .35, knee, cameras, pixels, [.9]*count, sign)
    assert fit['pixel_sse_after'] < fit['pixel_sse_before']
    assert 0 < fit['maximum_angle_change_rad'] <= .30
    assert len(fit['pixel_errors_after']) == count and pixels == original
    assert np.linalg.norm(np.array(fit['after_world_m'])-knee) == pytest.approx(.35, abs=1e-12)
    np.testing.assert_allclose(np.array(fit['output_rotation_world'])[:, 2], rotation[:, 2], atol=1e-12)
    assert not fit['depth_measured'] and not fit['observations_modified']
    assert len(fit['iterations']) <= CONTRACT['maximum_iterations'] and fit['trial_evaluations'] <= 48
    previous = np.zeros(2)
    for iteration in fit['iterations']:
        for trial in iteration['trials']:
            proposed = np.array(trial['correction_rad'])
            assert np.max(abs(proposed-previous)) <= .1+1e-12 and np.max(abs(proposed)) <= .30
            if trial['accepted']:
                assert trial['objective'] < iteration['objective_before']
                previous = proposed


def test_exact_image_match_ignores_unnecessary_temporal_pull():
    point, _ = distal_geometry([0., 0.], np.eye(3), 1., .3, [0, 0, 5], -1)
    view = camera(); pixel = _project(point[None, :], view)[0][0]
    fit = fit_distal_pixels(np.eye(3), 1., .3, [0, 0, 5], [view], [pixel], [1.], -1, previous=[.2, -.2])
    assert fit['correction_rad'] == [0., 0.] and fit['iterations'] == []


def test_conflicting_views_keep_both_residuals_and_obey_source_nonregression():
    views = [camera(), camera(offset=1.)]
    pixels = [[500., 380.], [820., 600.]]
    fit = fit_distal_pixels(np.eye(3), .01, .3, [0., 0., 5.], views, pixels, [.8, .9], 1, previous=[.3, -.3])
    assert 0 <= fit['output_bend_rad'] <= 2.6
    assert 0 < fit['pixel_sse_after'] <= fit['pixel_sse_before']
    assert len(fit['pixel_errors_after']) == 2 and fit['source_pixels'] == pixels


@pytest.mark.parametrize('fault', ['rotation', 'bend', 'length', 'knee', 'sign', 'no_views', 'many_views',
    'pixel', 'confidence', 'previous', 'behind', 'camera_reflection'])
def test_invalid_or_unprojectable_inputs_do_not_produce_a_proposal(fault):
    args = dict(rotation=np.eye(3), bend=1., length=.3, knee=[0., 0., 5.], cameras=[camera()],
                pixels=[[640., 480.]], confidence=[1.], sign=-1)
    if fault == 'rotation': args['rotation'] = np.ones((3, 3))
    elif fault == 'bend': args['bend'] = 3.
    elif fault == 'length': args['length'] = 0.
    elif fault == 'knee': args['knee'][0] = float('nan')
    elif fault == 'sign': args['sign'] = True
    elif fault == 'no_views': args['cameras'] = []
    elif fault == 'many_views': args['cameras'] *= 5
    elif fault == 'pixel': args['pixels'][0][0] = -1
    elif fault == 'confidence': args['confidence'] = [.49]
    elif fault == 'previous': args['previous'] = [.31, 0.]
    elif fault == 'behind': args['knee'][2] = -5.
    elif fault == 'camera_reflection': args['cameras'][0]['rotation'] = np.diag([-1., 1., 1.])
    with pytest.raises(ValueError): fit_distal_pixels(**args)
