from copy import deepcopy

import numpy as np
import pytest

from weft.track_shape import reconstruct_track_shape


def fixture(frames=16, count=32, seed=71):
    rng = np.random.default_rng(seed)
    shape = rng.uniform(-1, 1, (count, 3))*[1, .8, .6]
    cameras = []
    for t in np.linspace(-1, 1, frames):
        a, b, c = .65*t, .5*np.sin(2*t), .3*np.cos(2*t)
        x = np.array([[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]])
        y = np.array([[np.cos(b), 0, np.sin(b)], [0, 1, 0], [-np.sin(b), 0, np.cos(b)]])
        z = np.array([[np.cos(c), -np.sin(c), 0], [np.sin(c), np.cos(c), 0], [0, 0, 1]])
        cameras.append((z @ y @ x)[:2]*(90+20*t))
    pixels = np.einsum('fij,pj->fpi', cameras, shape)+np.array([[[320+i, 240-i]] for i in range(frames)])
    return shape, pixels


def fit(pixels, **changes):
    arguments = dict(frame_ids=list(range(len(pixels))), times_s=(np.arange(len(pixels))*.04).tolist(),
        track_ids=list(range(pixels.shape[1])), image_size=[640, 480], scope='construction')
    return reconstruct_track_shape(pixels, **(arguments | changes))


def distances(shape):
    points = np.asarray(shape)-np.mean(shape, axis=0)
    points /= np.sqrt(np.mean(np.sum(points**2, axis=1)))
    return np.linalg.norm(points[:, None, :]-points[None, :, :], axis=2)


def test_recovers_known_relative_3d_shape_and_all_projections():
    truth, pixels = fixture()
    result = fit(pixels)
    assert result['status'] == 'conditional_shape'
    candidate = result['candidate']
    np.testing.assert_allclose(distances(candidate['points_relative']), distances(truth), atol=1e-11)
    np.testing.assert_allclose(candidate['predicted_points_px'], pixels, atol=1e-10)
    cameras = np.asarray(candidate['camera_rows_px_per_relative_unit'])
    for camera in cameras:
        gram = camera @ camera.T
        np.testing.assert_allclose(gram, np.eye(2)*np.trace(gram)/2, atol=1e-8)
    assert not result['rigidity_verified'] and not result['metric_scale_verified']


def test_noise_is_retained_and_shape_remains_accurate_in_control():
    truth, pixels = fixture()
    noisy = pixels+np.random.default_rng(181).normal(0, .1, pixels.shape)
    result = fit(noisy)
    assert result['status'] == 'conditional_shape'
    assert result['candidate']['errors']['maximum_px'] > 0
    assert np.sqrt(np.mean((distances(result['candidate']['points_relative'])-distances(truth))**2)) < .01


def test_depth_reflection_produces_identical_images_and_is_not_resolved():
    _, pixels = fixture()
    candidate = fit(pixels)['candidate']
    shape = np.asarray(candidate['points_relative']); shape[:, 2] *= -1
    cameras = np.asarray(candidate['camera_rows_px_per_relative_unit']); cameras[:, :, 2] *= -1
    mirrored = np.einsum('fij,pj->fpi', cameras, shape)+np.asarray(candidate['feature_centroids_px'])[:, None, :]
    np.testing.assert_allclose(mirrored, pixels, atol=1e-10)
    assert candidate['reflection_alternative']


def test_image_translation_and_scale_do_not_change_relative_shape():
    truth, pixels = fixture()
    changed = (pixels-[320, 240])*.6+[270, 180]
    result = fit(changed)
    assert result['status'] == 'conditional_shape'
    np.testing.assert_allclose(distances(result['candidate']['points_relative']), distances(truth), atol=1e-11)


def test_missing_points_are_excluded_by_track_not_filled_or_hidden():
    truth, pixels = fixture()
    pixels[5, 0] = np.nan; pixels[:, 1] = np.nan
    result = fit(pixels)
    assert result['status'] == 'conditional_shape'
    assert result['incomplete_track_ids'] == [0, 1]
    assert result['coverage']['missing_point_slots'] == 17
    assert result['coverage']['observed_point_slots'] == pixels.shape[0]*pixels.shape[1]-17
    np.testing.assert_allclose(distances(result['candidate']['points_relative']), distances(truth[2:]), atol=1e-11)


def test_no_tracks_and_short_suffix_remain_explicit():
    assert fit(np.empty((8, 0, 2)))['status'] == 'insufficient_complete_tracks'
    assert fit(fixture()[1][:3])['status'] == 'insufficient_frames'
    result = fit(np.full((8, 7, 2), np.nan))
    assert result['status'] == 'insufficient_complete_tracks'
    assert result['coverage']['missing_point_slots'] == 56


def test_pure_image_rotation_cannot_recover_depth():
    truth, _ = fixture()
    xy = truth[:, :2]*90
    pixels = np.array([xy @ np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])+[320, 240]
        for a in np.linspace(0, 1, 16)])
    result = fit(pixels)
    assert result['status'] == 'insufficient_depth_excitation'
    assert result['affine_fit']['errors']['maximum_px'] < 1e-10
    assert result['candidate'] is None


def test_coplanar_points_do_not_gain_fabricated_depth():
    _, pixels = fixture()
    # Reproject an actual plane through a varied, valid camera sequence.
    planar = np.random.default_rng(1).uniform(-60, 60, (32, 2))
    pixels = np.array([planar @ np.array([[np.cos(a), 0], [0, 1]])+[320, 240] for a in np.linspace(0, .8, 16)])
    assert fit(pixels)['status'] == 'insufficient_depth_excitation'


def test_mismatches_are_retained_and_reject_the_whole_fit():
    _, pixels = fixture()
    pixels[5, 0] += [70, -45]
    result = fit(pixels)
    assert result['status'] == 'inconsistent_affine_tracks'
    assert result['complete_track_ids'] == list(range(32))
    assert len(result['affine_fit']['errors']['per_point_px']) == 16


def test_two_distinct_view_orientations_do_not_determine_the_metric():
    _, pixels = fixture()
    pixels = np.tile(pixels[[0, -1]], (8, 1, 1))
    result = fit(pixels)
    assert result['status'] == 'unobservable_metric'
    assert result['third_singular_ratio'] > .01


def test_coherent_scale_deformation_is_indistinguishable_from_camera_scaling():
    _, pixels = fixture()
    expanded = (pixels-[320, 240])*np.linspace(.8, 1.2, len(pixels))[:, None, None]+[320, 240]
    result = fit(expanded)
    assert result['status'] == 'conditional_shape'
    assert result['physical_validation'] == 'unverified'
    assert not result['rigidity_verified']


def test_source_frame_and_clock_gaps_are_not_bridged():
    _, pixels = fixture()
    assert fit(pixels, frame_ids=list(range(0, 32, 2)))['status'] == 'source_gap'
    assert fit(pixels, times_s=(np.arange(16)*.3).tolist())['status'] == 'source_gap'


def test_inputs_order_and_identity_are_preserved():
    _, pixels = fixture(); original = deepcopy(pixels)
    result = fit(pixels)
    np.testing.assert_array_equal(pixels, original)
    order = np.random.default_rng(88).permutation(32)
    other = fit(pixels[:, order], track_ids=order.tolist())
    assert other['complete_track_ids'] == order.tolist()
    np.testing.assert_allclose(distances(other['candidate']['points_relative']),
        distances(result['candidate']['points_relative'])[order][:, order], atol=1e-11)


@pytest.mark.parametrize('changes', [dict(scope='evaluation'), dict(frame_ids=list(range(15))),
    dict(frame_ids=[True]+list(range(1, 16))), dict(times_s=[0.]*16), dict(track_ids=[0]*32),
    dict(image_size=[640., 480]), dict(track_ids=list(range(31)))])
def test_invalid_metadata_is_rejected(changes):
    with pytest.raises(ValueError):
        fit(fixture()[1], **changes)


@pytest.mark.parametrize('value', [[np.nan, 2], [np.inf, np.inf], [-1, 3], [640, 0]])
def test_invalid_points_are_not_missing_evidence(value):
    _, pixels = fixture(); pixels[0, 0] = value
    with pytest.raises(ValueError):
        fit(pixels)


def test_work_bounds_are_enforced_without_truncation():
    with pytest.raises(ValueError, match='1–32'):
        fit(fixture(frames=33)[1])
    with pytest.raises(ValueError, match='budget'):
        fit(fixture(count=129)[1])
