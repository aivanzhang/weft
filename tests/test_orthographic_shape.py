from copy import deepcopy

import numpy as np
import pytest

from weft.affine_camera import project_affine_points
from weft.orthographic_shape import conditional_axis_scale


TRUTH = np.array([[900., 0., -1200., 300.], [0., 1500., 0., 400.]])
CENTER = np.array([.3, .2, .1])


def wrong_depth(factor):
    matrix = TRUTH.copy(); matrix[:, 1] /= factor
    matrix[:, 3] += (TRUTH[:, :3] - matrix[:, :3]) @ CENTER
    return matrix


def infer(matrix, direction=(0., 1., 0.), center=CENTER):
    return conditional_axis_scale(matrix, direction=direction, center=center, scope='construction')


@pytest.mark.parametrize('factor', [.4, .8, 1., 2., 3.])
def test_recovers_one_wrong_dimension_under_its_explicit_camera_assumption(factor):
    result = infer(wrong_depth(factor))
    assert result['available'] and result['factor'] == pytest.approx(1/factor)
    np.testing.assert_allclose(result['matrix_px'], TRUTH, atol=1e-10)
    assert result['image_axis_scale_ratio'] == pytest.approx(1)
    assert not result['metric_geometry_verified']


def test_geometry_camera_compensation_keeps_all_opening_predictions():
    points = np.random.default_rng(45).uniform(-.4, .4, (128, 3)) + CENTER
    camera = wrong_depth(2); result = infer(camera)
    changed = points.copy(); changed[:, 1] = CENTER[1] + (points[:, 1]-CENTER[1])*result['factor']
    np.testing.assert_allclose(project_affine_points(points, camera),
                               project_affine_points(changed, result['matrix_px']), atol=1e-10)


def test_arbitrary_world_frame_and_axis_are_supported():
    rotation = np.array([[0., -1., 0.], [0., 0., -1.], [1., 0., 0.]])
    camera = wrong_depth(2); camera[:, :3] = camera[:, :3] @ rotation.T
    result = infer(camera, rotation @ [0., 1., 0.], rotation @ CENTER)
    assert result['factor'] == pytest.approx(.5)
    np.testing.assert_allclose(np.asarray(result['matrix_px'])[:, :3], TRUTH[:, :3] @ rotation.T, atol=1e-10)


def test_unobservable_axis_cannot_supply_a_dimension():
    result = infer(np.array([[1., 0., 0., 100.], [0., 0., 1., 100.]]))
    assert not result['available'] and result['reason'] == 'shape_axis_not_observable_in_image'


def test_incompatible_camera_cannot_propose_a_negative_squared_scale():
    result = infer(np.array([[2., 1., 1., 100.], [0., .1, 0., 100.]]), (1., 0., 0.))
    assert not result['available'] and result['reason'] == 'no_positive_axis_scale'


def test_large_scale_change_is_retained_as_rejected_not_clipped():
    result = infer(wrong_depth(10))
    assert not result['available'] and result['factor'] == pytest.approx(.1)
    assert result['matrix_px'] is None


def test_wrong_overall_metric_scale_remains_unidentifiable():
    camera = wrong_depth(2); camera[:, :3] /= 3
    result = infer(camera, center=CENTER*3)
    assert result['available'] and result['image_axis_scale_ratio'] == pytest.approx(1)
    assert result['metric_geometry_verified'] is False
    np.testing.assert_allclose(np.asarray(result['matrix_px'])[:, :3], TRUTH[:, :3]/3)


def test_inputs_are_unchanged():
    matrix = wrong_depth(2).tolist(); original = deepcopy(matrix)
    infer(matrix)
    assert matrix == original


def test_rescaling_cannot_emit_an_out_of_bounds_camera():
    result = infer(np.array([[1e8, 0., 0., 0.], [0., 5e7, 0., 0.]]), center=[0., 1e4, 0.])
    assert not result['available'] and result['reason'] == 'camera_after_scale_exceeds_envelope'


def test_invalid_scope_and_axis_are_rejected():
    with pytest.raises(ValueError, match='construction'):
        conditional_axis_scale(TRUTH, direction=[0., 1., 0.], center=CENTER, scope='evaluation')
    with pytest.raises(ValueError, match='unit shape axis'):
        infer(TRUTH, (0., 2., 0.))
    with pytest.raises(ValueError, match='independent image'):
        infer(np.zeros((2, 4)))
