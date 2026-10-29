from copy import deepcopy
from itertools import product

import numpy as np
import pytest

from weft.affine_camera import fit_affine_camera, project_affine_points


POINTS = np.array(list(product((-.09, .09), (-.07, .07), (-.03, .03)))) + [.3, .2, .15]
MATRIX = np.array([[1200., 100., -350., 200.], [20., -1000., -700., 600.]])
PIXELS = POINTS @ MATRIX[:, :3].T + MATRIX[:, 3]


def fit(points=POINTS, pixels=PIXELS, **kwargs):
    return fit_affine_camera(points, pixels, image_size=[1920, 1080], scope='construction', **kwargs)


def test_recovers_known_affine_camera_and_unseen_points():
    result = fit()
    np.testing.assert_allclose(result['matrix_px'], MATRIX, atol=1e-10)
    unseen = np.random.default_rng(9).uniform(0, .5, (100, 3))
    np.testing.assert_allclose(project_affine_points(unseen, result['matrix_px']),
                               unseen @ MATRIX[:, :3].T + MATRIX[:, 3], atol=1e-10)
    assert result['maximum_error_px'] < 1e-10
    assert not result['metric_geometry_verified']


def test_planar_points_cannot_identify_an_affine_3d_camera():
    planar = POINTS.copy(); planar[:, 2] = .1
    with pytest.raises(ValueError, match='noncoplanar'):
        fit(planar)


def test_nearly_planar_points_are_rejected_before_large_camera_coefficients():
    points = POINTS.copy(); points[:, 2] = .1 + (points[:, 2] - .15) * 1e-6
    with pytest.raises(ValueError, match='noncoplanar'):
        fit(points)


def test_wrong_depth_can_fit_identical_pixels_and_stays_conditional():
    changed = POINTS.copy(); changed[:, 1] *= 3
    result = fit(changed)
    assert result['maximum_error_px'] < 1e-10
    np.testing.assert_allclose(np.asarray(result['matrix_px'])[:, 1], MATRIX[:, 1] / 3)
    assert result['physical_validation'] == 'unverified'


def test_uniform_units_and_world_rotation_preserve_projection():
    rotation = np.array([[0., -1., 0.], [0., 0., -1.], [1., 0., 0.]])
    transformed = (POINTS @ rotation.T) * 10 + [3, 4, 5]
    result = fit(transformed)
    np.testing.assert_allclose(result['fitted_points_px'], PIXELS, atol=1e-10)
    assert result['normalized_design_condition'] == pytest.approx(fit()['normalized_design_condition'])


def test_outlier_is_retained_and_cannot_disappear_through_selection():
    pixels = PIXELS.copy(); pixels[0] += [60, -30]
    result = fit(pixels=pixels)
    assert result['point_count'] == len(POINTS)
    assert len(result['errors_px']) == len(POINTS)
    assert result['errors_px'][0] > 10 and result['rmse_px'] > 10


def test_scaled_orthographic_diagnostic_distinguishes_anisotropic_camera():
    orthographic = np.array([[900., 0., -1200., 300.], [0., -1500., 0., 800.]])
    result = fit(pixels=project_affine_points(POINTS, orthographic))
    assert result['orthographic_diagnostic']['image_axis_scale_ratio'] == pytest.approx(1)
    assert result['orthographic_diagnostic']['relative_isotropy_error'] < 1e-12
    assert fit()['orthographic_diagnostic']['image_axis_scale_ratio'] > 1.1


def test_image_line_is_rejected_even_with_spatially_full_rank_points():
    pixels = PIXELS.copy(); pixels[:, 1] = pixels[:, 0] / 2
    with pytest.raises(ValueError, match='collapse'):
        fit(pixels=pixels)


def test_inputs_and_order_are_preserved():
    points, pixels = POINTS.tolist(), PIXELS.tolist()
    original = deepcopy((points, pixels))
    result = fit(points, pixels)
    assert (points, pixels) == original
    order = [7, 0, 3, 2, 1, 6, 5, 4]
    reordered = fit(POINTS[order], PIXELS[order])
    np.testing.assert_allclose(reordered['matrix_px'], result['matrix_px'], atol=1e-10)


@pytest.mark.parametrize('points', [np.zeros((5, 3)), np.zeros((1025, 3)),
                                  np.full((8, 3), np.nan), np.full((8, 3), 1e5)])
def test_invalid_or_unbounded_geometry_is_rejected(points):
    with pytest.raises(ValueError, match='finite 3D'):
        fit(points)


@pytest.mark.parametrize('pixels', [PIXELS[:7], np.full((8, 2), np.nan),
                                  np.full((8, 2), -1), np.full((8, 2), 1920)])
def test_missing_or_invalid_observations_are_rejected(pixels):
    with pytest.raises(ValueError, match='in-image'):
        fit(pixels=pixels)


def test_evaluation_evidence_cannot_be_fitted():
    with pytest.raises(ValueError, match='construction'):
        fit_affine_camera(POINTS, PIXELS, image_size=[1920, 1080], scope='evaluation')


def test_invalid_image_size_and_projection_matrix_are_rejected():
    with pytest.raises(ValueError, match='Image size'):
        fit_affine_camera(POINTS, PIXELS, image_size=[1920., 1080], scope='construction')
    with pytest.raises(ValueError, match='camera matrix'):
        project_affine_points(POINTS, np.zeros((3, 4)))
