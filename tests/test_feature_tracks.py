from copy import deepcopy

import cv2
import numpy as np
import pytest

from weft.feature_tracks import AppearanceTracklets, CONTRACT
from weft.surface_features import CONTRACT as FEATURE_CONTRACT, detect_surface_features


def features(values, points=None):
    descriptors = np.zeros((len(values), 128), np.float32)
    descriptors[:, 0] = values
    points = np.column_stack([np.arange(len(values))*5+20, np.arange(len(values))*3+20]) if points is None else np.asarray(points)
    keypoints = np.zeros((len(values), 6)); keypoints[:, :2] = points
    return {'method': FEATURE_CONTRACT['method'], 'keypoints': keypoints,
        'descriptors': descriptors, 'selected_indices': list(range(len(values))), 'aliases': list(range(len(values)))}


def test_adjacent_links_retain_ids_with_explicit_direct_anchors():
    initial = features([0, 100, 225]); tracker = AppearanceTracklets(initial, frame=4, time_s=.2)
    moved = features([0, 100, 225], initial['keypoints'][:, :2]+[7, 4])
    result = tracker.advance(moved, frame=5, time_s=.24)
    assert result['linked_tracks'] == result['anchored_tracks'] == 3
    assert [row['track_id'] for row in result['rows']] == [0, 1, 2]
    assert all(row['age_frames'] == 2 and row['anchor_status'] == 'direct_reference' for row in result['rows'])


def test_no_interpolation_or_id_reuse_across_missing_features():
    source = features([0, 100, 225]); tracker = AppearanceTracklets(source, frame=0, time_s=0.)
    absent = tracker.advance(features([]), frame=1, time_s=.04)
    assert len(absent['ended_tracks']) == 3 and absent['active_tracks'] == 0
    recovered = tracker.advance(source, frame=2, time_s=.08)
    assert recovered['linked_tracks'] == 0 and recovered['anchored_tracks'] == 3
    assert [row['track_id'] for row in recovered['rows']] == [3, 4, 5]


@pytest.mark.parametrize(('frame','time_s','reason'), [(2, .08, 'frame_gap'), (1, .3, 'time_gap')])
def test_discontinuities_break_links_even_when_descriptors_agree(frame, time_s, reason):
    source = features([0, 100, 225]); tracker = AppearanceTracklets(source, frame=0, time_s=0.)
    result = tracker.advance(source, frame=frame, time_s=time_s)
    assert result['continuity'] == reason and result['linked_tracks'] == 0
    assert all(row['reason'] == reason for row in result['ended_tracks'])


def test_original_reference_conflict_splits_instead_of_silently_relabeling():
    tracker = AppearanceTracklets(features([0, 100, 225]), frame=0, time_s=0.)
    prior = tracker.advance(features([30, 160, 225]), frame=1, time_s=.04)
    result = tracker.advance(features([70, 160, 225]), frame=2, time_s=.08)
    row = result['rows'][0]
    assert row['link_status'] == row['anchor_status'] == 'reference_conflict'
    assert row['reference_candidates'] == [0, 1] and row['reference_id'] is None
    assert row['track_id'] != prior['rows'][0]['track_id']
    assert any(item['track_id'] == prior['rows'][0]['track_id'] and item['reason'] == 'reference_conflict'
               for item in result['ended_tracks'])


def test_ambiguous_descriptors_do_not_create_links_or_merge_ids():
    source = features([10, 10, 10]); tracker = AppearanceTracklets(source, frame=0, time_s=0.)
    result = tracker.advance(source, frame=1, time_s=.04)
    assert result['linked_tracks'] == result['anchored_tracks'] == 0
    assert len({row['track_id'] for row in result['rows']}) == 3
    assert all(row['status'] == 'forward_ambiguity' for row in result['adjacent_matches']['rows'])


def test_duplicate_orientations_remain_explicit_without_extra_tracks():
    source = features([0, 0, 225]); source['selected_indices'] = [0, 2]; source['aliases'] = [0, 0, 2]
    tracker = AppearanceTracklets(source, frame=0, time_s=0.)
    result = tracker.advance(source, frame=1, time_s=.04)
    assert len(result['rows']) == 3 and result['active_tracks'] == result['linked_tracks'] == 2
    assert result['rows'][1]['duplicate_of'] == 0 and result['rows'][1]['track_id'] is None


def test_caller_mutation_cannot_replace_original_reference_or_active_ids():
    source = features([0, 100, 225]); original = deepcopy(source)
    tracker = AppearanceTracklets(source, frame=0, time_s=0.)
    source['descriptors'][:] = 0; tracker.initialization['rows'][0]['track_id'] = 999
    result = tracker.advance(original, frame=1, time_s=.04)
    result['rows'][0]['track_id'] = 999
    after = tracker.advance(original, frame=2, time_s=.08)
    assert after['linked_tracks'] == 3 and after['rows'][0]['track_id'] == 0


def test_invalid_frame_clock_and_budget_do_not_silently_mutate_state():
    source = features([0, 100, 225]); tracker = AppearanceTracklets(source, frame=0, time_s=0.)
    for frame, stamp in ((0, .04), (1, 0.), (1, float('nan')), (True, .04)):
        with pytest.raises(ValueError):tracker.advance(source, frame=frame, time_s=stamp)
    assert tracker.advance(source, frame=1, time_s=.04)['rows'][0]['age_frames'] == 2
    tracker._count = CONTRACT['maximum_frames']
    with pytest.raises(ValueError, match='budget'):tracker.advance(source, frame=2, time_s=.08)


def test_source_frame_gap_does_not_hide_original_matches():
    source = features([0, 100, 225]); tracker = AppearanceTracklets(source, frame=0, time_s=0.)
    result = tracker.advance(source, frame=3, time_s=.12)
    assert result['adjacent_matches'] is None and result['original_matches']['appearance_pairs'] == 3
    assert result['linked_tracks'] == 0 and result['anchored_tracks'] == 3


def test_known_rotating_texture_tracks_are_measured_against_true_warp():
    rng = np.random.default_rng(81)
    source = np.full((300, 360), 128, np.uint8)
    source[65:235, 80:280] = cv2.GaussianBlur(rng.integers(0, 256, (170, 200), np.uint8), (0, 0), .8)
    mask = np.zeros_like(source); mask[65:235, 80:280] = 255
    initial = detect_surface_features(source, mask); tracker = AppearanceTracklets(initial, frame=0, time_s=0.)
    prior = {row['track_id']: np.asarray(row['point_px']) for row in tracker.initialization['rows'] if row['track_id'] is not None}
    previous_transform = np.eye(3); errors = []; count = 0
    for frame in range(1, 16):
        matrix = cv2.getRotationMatrix2D((180, 150), frame*2., 1.+frame*.003)
        matrix[:, 2] += [frame*.4, frame*.15]
        image = cv2.warpAffine(source, matrix, (360, 300), borderValue=128)
        region = cv2.warpAffine(mask, matrix, (360, 300), flags=cv2.INTER_NEAREST)
        result = tracker.advance(detect_surface_features(image, region), frame=frame, time_s=frame*.04)
        transform = np.vstack([matrix, [0, 0, 1]])
        step = transform @ np.linalg.inv(previous_transform)
        for row in result['rows']:
            if row['link_status'] != 'adjacent_pair':continue
            expected = step @ np.r_[prior[row['track_id']]-.5, 1.]
            errors.append(np.linalg.norm(np.asarray(row['point_px'])-(expected[:2]/expected[2]+.5))); count += 1
        prior = {row['track_id']: np.asarray(row['point_px']) for row in result['rows'] if row['track_id'] is not None}
        previous_transform = transform
    assert count >= 200 and np.percentile(errors, 95) < 1.


def test_matching_appearance_cannot_prove_physical_identity_or_edit_continuity():
    source = features([0, 100, 225]); tracker = AppearanceTracklets(source, frame=0, time_s=0.)
    # Replacing an object with an identical-looking copy can pass all appearance checks.
    result = tracker.advance(features([0, 100, 225]), frame=1, time_s=.04)
    assert result['linked_tracks'] == 3
    assert result['physical_point_identity_verified'] is result['rigidity_verified'] is False
