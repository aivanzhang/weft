import numpy as np
import pytest

from weft.parallel_lines import estimate_parallel_axis


def planes(axis, count=8):
    axis = np.asarray(axis, float); axis /= np.linalg.norm(axis)
    first = np.cross(axis, [1., 0., 0.]); first /= np.linalg.norm(first)
    second = np.cross(axis, first)
    angles = np.linspace(0., 2.5, count)
    return np.cos(angles)[:, None]*first+np.sin(angles)[:, None]*second


def test_recovers_known_unsigned_axis():
    axis = np.array([.1, -.95, -.3]); axis /= np.linalg.norm(axis)
    result = estimate_parallel_axis(planes(axis), np.ones(8))
    assert abs(np.dot(result['axis'], axis)) == pytest.approx(1.)
    assert result['inlier_indices'] == list(range(8))
    assert max(result['angular_residuals_degrees']) < 1e-10


def test_outlier_planes_remain_visible():
    axis = np.array([.1, -.95, -.3]); axis /= np.linalg.norm(axis)
    normals = np.vstack([planes(axis), np.tile(axis, (3, 1))])
    result = estimate_parallel_axis(normals, np.ones(11))
    assert result['inlier_indices'] == list(range(8))
    assert min(result['angular_residuals_degrees'][8:]) > 89.99
    assert result['supported_weight_fraction'] == pytest.approx(8/11)
    assert len(result['hypotheses']) > 28


def test_rotating_the_scene_rotates_the_axis():
    axis = np.array([.1, -.95, -.3]); axis /= np.linalg.norm(axis)
    rotation = np.array([[0., -1., 0.], [0., 0., -1.], [1., 0., 0.]])
    result = estimate_parallel_axis(planes(axis) @ rotation.T, np.ones(8))
    assert abs(np.dot(result['axis'], rotation @ axis)) == pytest.approx(1.)


def test_normal_scaling_and_sign_do_not_change_axis():
    source = planes([0., 1., .3])
    scaled = source*np.array([2., -1., 7., -3., 5., -2., 9., -4.])[:, None]
    first = estimate_parallel_axis(source, np.ones(8))
    second = estimate_parallel_axis(scaled, np.ones(8))
    assert abs(np.dot(first['axis'], second['axis'])) == pytest.approx(1.)


def test_two_distinct_planes_define_only_an_unsigned_axis():
    result = estimate_parallel_axis([[1., 0., 0.], [0., 0., 1.]], [1., 1.])
    assert result['axis'] == [0., 1., 0.] and result['unsigned']


def test_duplicate_planes_do_not_claim_an_axis():
    with pytest.raises(ValueError, match='Distinct ray planes'):
        estimate_parallel_axis([[1., 0., 0.]]*5, [1.]*5)


@pytest.mark.parametrize('normals,weights', [([[0., 0., 0.], [0., 1., 0.]], [1., 1.]),
    ([[1., 0., 0.]], [1.]), (np.ones((129, 3)), np.ones(129)), (np.ones((2, 2)), [1., 1.]),
    (np.ones((2, 3)), [-1., 1.]), (np.ones((2, 3)), [1.]),
    ([[np.nan, 0., 0.], [0., 1., 0.]], [1., 1.]), (np.ones((2, 3)), [np.inf, 1.])])
def test_invalid_or_unbounded_inputs_are_rejected(normals, weights):
    with pytest.raises(ValueError):
        estimate_parallel_axis(normals, weights)


@pytest.mark.parametrize('tolerance', [0, -1, 3, float('nan'), True])
def test_tolerance_envelope_is_not_silently_relaxed(tolerance):
    with pytest.raises(ValueError, match='tolerance'):
        estimate_parallel_axis([[1., 0., 0.], [0., 0., 1.]], [1., 1.], tolerance_degrees=tolerance)
