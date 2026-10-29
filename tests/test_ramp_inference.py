import math

import numpy as np
import pytest

from weft.ramp_inference import affine_ramp_camera, fit_motion_prefix, gravity_hypotheses, image_to_scene, scene_to_image


def camera():
    return affine_ramp_camera(1920, 1080, [[10, 100], [700, 350]], [[730, 480], [1900, 492]], 20)


def test_camera_recovers_declared_support_angles_and_round_trip():
    value = camera()
    ramp = image_to_scene(value, [[10, 100], [700, 350]])
    table = image_to_scene(value, [[730, 480], [1900, 492]])
    assert math.atan2(*(ramp[1] - ramp[0])[::-1]) == pytest.approx(math.radians(20))
    assert table[0, 1] == pytest.approx(table[1, 1])
    points = [[0, 0], [1920, 1080], [232, 292.5], [1000, 487]]
    projected = image_to_scene(value, points)
    np.testing.assert_allclose(scene_to_image(value, projected), points, atol=1e-10)
    assert projected.min() > 0
    assert (projected[:, 0] < 1).all()
    assert (projected[:, 1] < value['scene_height']).all()


def test_fit_recovers_analytic_motion_with_nonzero_source_start():
    times = np.linspace(4, 4.5, 12)
    elapsed = times - times[0]
    positions = [.2, .3] + elapsed[:, None] * [.1, .2] + .5 * elapsed[:, None]**2 * [.4, .5]
    fitted = fit_motion_prefix(times, positions)
    np.testing.assert_allclose(fitted['position'], [.2, .3], atol=1e-12)
    np.testing.assert_allclose(fitted['velocity'], [.1, .2], atol=1e-12)
    np.testing.assert_allclose(fitted['acceleration'], [.4, .5], atol=1e-12)
    assert fitted['start_time_s'] == 4
    assert fitted['rmse_scene_units'] < 1e-12


def test_distinct_friction_scale_hypotheses_explain_identical_acceleration():
    angle = math.radians(20)
    acceleration = .9 * np.array([math.cos(angle), math.sin(angle)])
    family = gravity_hypotheses(acceleration, 20, [.2, .25, .3])
    assert len({row['gravity_scene_widths_per_s2'] for row in family}) == 3
    for row in family:
        assert row['gravity_scene_widths_per_s2'] * (math.sin(angle) - row['pair_friction'] * math.cos(angle)) == pytest.approx(.9)
        assert not row['metric_scale_verified']
        assert row['normal_acceleration_residual'] == pytest.approx(0, abs=1e-12)


@pytest.mark.parametrize('times', [[0, .1, .2, .3], [0, .1, .1, .3, .4], [0, .1, np.nan, .3, .4], [0, 2, 4, 6, 8]])
def test_invalid_time_support_is_rejected(times):
    with pytest.raises(ValueError):
        fit_motion_prefix(times, np.zeros((len(times), 2)))


@pytest.mark.parametrize('friction', [-.1, .5, float('nan')])
def test_nonphysical_friction_hypotheses_are_rejected(friction):
    with pytest.raises(ValueError):
        gravity_hypotheses([1, .4], 20, [friction])


def test_incompatible_camera_and_unbounded_motion_are_rejected():
    with pytest.raises(ValueError):
        affine_ramp_camera(1920, 1080, [[0, 100], [700, 100]], [[730, 480], [1900, 492]], 20)
    with pytest.raises(ValueError):
        fit_motion_prefix(np.linspace(0, 1, 5), [[math.inf, 0]] * 5)
    with pytest.raises(ValueError):
        gravity_hypotheses([-1, -.2], 20, [.2])
    with pytest.raises(ValueError):
        gravity_hypotheses([20, 5], 20, [.3])
