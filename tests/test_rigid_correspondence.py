import numpy as np
import pytest

from weft.rigid_correspondence import fit_rigid_points


POINTS = np.array([[0., 0., 0.], [.3, 0., 0.], [0., .2, 0.], [0., 0., .1], [.2, .1, .3]])
ROTATION = np.array([[0., -1., 0.], [0., 0., -1.], [1., 0., 0.]])
TRANSLATION = np.array([1., -2., 3.])


def test_known_rigid_placement():
    result = fit_rigid_points(POINTS, POINTS @ ROTATION.T+TRANSLATION)
    np.testing.assert_allclose(result['rotation'], ROTATION, atol=1e-12)
    np.testing.assert_allclose(result['translation'], TRANSLATION, atol=1e-12)
    assert result['maximum_error'] < 1e-12


def test_planar_pattern_is_not_rejected_as_rank_deficient():
    points = np.array([[0., 0., 0.], [.3, 0., 0.], [0., .2, 0.], [.3, .2, 0.]])
    result = fit_rigid_points(points, points @ ROTATION.T+TRANSLATION)
    np.testing.assert_allclose(result['rotation'], ROTATION, atol=1e-12)
    assert result['rmse'] < 1e-12 and result['observed_plane_rms'] < 1e-12


def test_scale_mismatch_remains_in_metric_residuals():
    result = fit_rigid_points(POINTS, 1.12*POINTS @ ROTATION.T+TRANSLATION)
    assert result['scale_hypothesis_not_applied'] == pytest.approx(1.12)
    assert result['scale_applied'] == 1. and result['rmse'] > .01
    np.testing.assert_allclose(np.array(result['rotation']).T @ result['rotation'], np.eye(3), atol=1e-12)


def test_reverse_correspondence_recovers_inverse():
    result = fit_rigid_points(POINTS @ ROTATION.T+TRANSLATION, POINTS)
    np.testing.assert_allclose(result['rotation'], ROTATION.T, atol=1e-12)
    np.testing.assert_allclose(result['translation'], -TRANSLATION @ ROTATION, atol=1e-12)


def test_reflection_is_not_a_rigid_solution():
    result = fit_rigid_points(POINTS, POINTS*np.array([-1., 1., 1.]))
    assert np.linalg.det(result['rotation']) == pytest.approx(1.)
    assert result['rmse'] > .01


@pytest.mark.parametrize('points', [np.zeros((5, 3)), np.array([[i, 0., 0.] for i in range(5)])])
def test_collinear_correspondences_are_rejected(points):
    with pytest.raises(ValueError, match='non-collinear'):
        fit_rigid_points(points, points)


@pytest.mark.parametrize('bad', [np.zeros((3, 3)), np.zeros((1001, 3)), np.zeros((5, 2)),
                               np.full((5, 3), np.nan), np.full((5, 3), 1e5)])
def test_invalid_or_unbounded_points_are_rejected(bad):
    with pytest.raises(ValueError, match='finite paired'):
        fit_rigid_points(bad, bad)


def test_unpaired_points_are_rejected():
    with pytest.raises(ValueError, match='finite paired'):
        fit_rigid_points(POINTS, POINTS[:4])
