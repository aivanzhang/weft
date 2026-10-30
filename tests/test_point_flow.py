from copy import deepcopy

import cv2
import numpy as np
import pytest

from weft.point_flow import OpticalPointHistories


def fixture():
    rng = np.random.default_rng(94)
    image = cv2.GaussianBlur(rng.integers(20, 235, (240, 320), np.uint8), (0, 0), .7)
    mask = np.zeros_like(image); mask[35:205, 45:275] = 255
    points = np.array([[90.5, 80.5], [150.5, 80.5], [220.5, 80.5], [90.5, 160.5], [150.5, 160.5], [220.5, 160.5]])
    return image, mask, points


def translated(image, mask, x, y):
    matrix = np.float32([[1, 0, x], [0, 1, y]])
    return cv2.warpAffine(image, matrix, (320, 240)), cv2.warpAffine(mask, matrix, (320, 240), flags=cv2.INTER_NEAREST)


def test_known_subpixel_translation_keeps_individual_points():
    image, mask, points = fixture(); tracker = OpticalPointHistories(image, mask, points, frame=0, time_s=0.)
    current, region = translated(image, mask, 3.25, -1.5)
    result = tracker.advance(current, region, frame=1, time_s=.04)
    assert result['active_points'] == len(points)
    for i, row in enumerate(result['rows']):
        assert row['status'] == 'flow_pair' and row['age_frames'] == 2
        np.testing.assert_allclose(row['point_px'], points[i]+[3.25, -1.5], atol=.15)
        assert row['roundtrip_error_px'] <= 1 and row['local_patch_ncc'] >= .9


def test_known_multiframe_rotation_tracks_source_coordinates():
    image, mask, points = fixture(); tracker = OpticalPointHistories(image, mask, points, frame=0, time_s=0.)
    errors = []; active = []
    for frame in range(1, 21):
        matrix = cv2.getRotationMatrix2D((160, 120), frame*2., 1.)
        current = cv2.warpAffine(image, matrix, (320, 240))
        region = cv2.warpAffine(mask, matrix, (320, 240), flags=cv2.INTER_NEAREST)
        result = tracker.advance(current, region, frame=frame, time_s=frame*.04)
        truth = (points-.5) @ matrix[:, :2].T+matrix[:, 2]+.5
        for row in result['rows']:
            if row['point_px'] is not None:errors.append(np.linalg.norm(row['point_px']-truth[row['seed_id']]))
        active.append(result['active_points'])
    assert min(active) >= 4 and max(errors) < 1.


def test_occlusion_ends_ids_without_reconnecting_later():
    image, mask, points = fixture(); tracker = OpticalPointHistories(image, mask, points, frame=0, time_s=0.)
    hidden = tracker.advance(image, np.zeros_like(mask), frame=1, time_s=.04)
    returned = tracker.advance(image, mask, frame=2, time_s=.08)
    assert hidden['active_points'] == returned['active_points'] == 0
    assert all(r['status'] == 'mask_unavailable' for r in hidden['rows'])
    assert all(r['status'] == 'ended' and r['point_px'] is None and r['ended']['frame'] == 1 for r in returned['rows'])


@pytest.mark.parametrize('frame,time_s', [(2, .08), (1, .21)])
def test_source_gaps_end_histories(frame, time_s):
    image, mask, points = fixture(); tracker = OpticalPointHistories(image, mask, points, frame=0, time_s=0.)
    result = tracker.advance(image, mask, frame=frame, time_s=time_s)
    assert not result['active_points'] and all(r['status'] == 'source_gap' for r in result['rows'])


def test_constant_texture_cannot_initialize_motion_evidence():
    image, mask, points = fixture()
    tracker = OpticalPointHistories(np.full_like(image, 128), mask, points, frame=0, time_s=0.)
    assert tracker.initialization['active_points'] == 0
    assert all(r['status'] == 'weak_texture' for r in tracker.initialization['rows'])


def test_mask_boundary_and_empty_seeds_stay_explicit():
    image, mask, _ = fixture()
    tracker = OpticalPointHistories(image, mask, [[45.5, 60.5]], frame=0, time_s=0.)
    assert tracker.initialization['rows'][0]['status'] == 'outside_mask_support'
    empty = OpticalPointHistories(image, mask, np.empty((0, 2)), frame=0, time_s=0.)
    assert empty.advance(image, mask, frame=1, time_s=.04)['rows'] == []


def test_changed_appearance_never_publishes_a_point_from_flow_alone(monkeypatch):
    image, mask, points = fixture(); tracker = OpticalPointHistories(image, mask, points, frame=0, time_s=0.)
    def unchanged(_first, _second, values, *_args, **_kwargs):
        return values.copy(), np.ones((len(values), 1), np.uint8), np.ones((len(values), 1), np.float32)
    monkeypatch.setattr(cv2, 'calcOpticalFlowPyrLK', unchanged)
    result = tracker.advance(255-image, mask, frame=1, time_s=.04)
    assert not result['active_points']
    assert all(r['status'] == 'local_appearance_changed' and r['point_px'] is None for r in result['rows'])


def test_identical_replacement_is_not_claimed_as_verified_identity():
    image, mask, points = fixture(); tracker = OpticalPointHistories(image, mask, points, frame=0, time_s=0.)
    result = tracker.advance(image.copy(), mask, frame=1, time_s=.04)
    assert result['active_points'] == len(points)
    assert not result['physical_point_identity_verified'] and not result['source_edit_continuity_verified']


def test_backward_status_failure_is_not_accepted(monkeypatch):
    image, mask, points = fixture(); tracker = OpticalPointHistories(image, mask, points, frame=0, time_s=0.)
    calls = []
    def flow(_first, _second, values, *_args, **_kwargs):
        calls.append(1)
        return values.copy(), np.full((len(values), 1), len(calls) == 1, np.uint8), np.ones((len(values), 1), np.float32)
    monkeypatch.setattr(cv2, 'calcOpticalFlowPyrLK', flow)
    result = tracker.advance(image, mask, frame=1, time_s=.04)
    assert all(r['status'] == 'backward_flow_failed' and r['point_px'] is None for r in result['rows'])


def test_inputs_and_initial_snapshot_are_isolated():
    image, mask, points = fixture(); original = deepcopy((image, mask, points))
    tracker = OpticalPointHistories(image, mask, points, frame=0, time_s=0.)
    first = tracker.initialization; first['rows'][0]['point_px'][0] = -100
    assert tracker.initialization['rows'][0]['point_px'] == points[0].tolist()
    tracker.advance(image, mask, frame=1, time_s=.04)
    for before, after in zip(original, (image, mask, points), strict=True):np.testing.assert_array_equal(before, after)


@pytest.mark.parametrize('points', [np.zeros((129, 2)), [[np.nan, 3]], [[-1, 3]], [[320, 4]], [[90, 80], [90, 80]]])
def test_invalid_or_ambiguous_seeds_rejected(points):
    image, mask, _ = fixture()
    with pytest.raises(ValueError):OpticalPointHistories(image, mask, points, frame=0, time_s=0.)


def test_invalid_image_masks_and_clocks_are_rejected():
    image, mask, points = fixture()
    with pytest.raises(ValueError):OpticalPointHistories(image.astype(float), mask, points, frame=0, time_s=0.)
    with pytest.raises(ValueError):OpticalPointHistories(image, mask//2, points, frame=0, time_s=0.)
    with pytest.raises(ValueError):OpticalPointHistories(np.zeros((4096, 4096), np.uint8), np.zeros((4096, 4096), np.uint8), points, frame=0, time_s=0.)
    tracker = OpticalPointHistories(image, mask, points, frame=0, time_s=0.)
    with pytest.raises(ValueError):tracker.advance(image, mask, frame=0, time_s=.04)
    with pytest.raises(ValueError):tracker.advance(image[:, :-1], mask[:, :-1], frame=1, time_s=.04)
