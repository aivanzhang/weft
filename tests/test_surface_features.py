from copy import deepcopy

import cv2
import numpy as np
import pytest

from weft.surface_features import CONTRACT, detect_surface_features, match_surface_features


def fixture():
    image = np.full((320, 400), 128, np.uint8)
    texture = np.random.default_rng(71).integers(0, 256, (170, 200), dtype=np.uint8)
    image[65:235, 80:280] = cv2.GaussianBlur(texture, (0, 0), .8)
    mask = np.zeros_like(image); mask[65:235, 80:280] = 255
    return image, mask


def features(descriptors, points=None):
    descriptors = np.asarray(descriptors, dtype=np.float32).reshape(-1, 128)
    n = len(descriptors)
    keypoints = np.zeros((n, 6), dtype=float)
    keypoints[:, :2] = np.column_stack([np.arange(n)*3+10, np.arange(n)*2+10]) if points is None else points
    return {'method': CONTRACT['method'], 'keypoints': keypoints, 'descriptors': descriptors,
        'selected_indices': list(range(n)), 'aliases': list(range(n))}


def test_translation_matches_known_source_points_without_chaining():
    image, mask = fixture(); transform = np.array([[1., 0., 17.], [0., 1., 11.]])
    moved = cv2.warpAffine(image, transform, (400, 320), borderValue=128)
    moved_mask = cv2.warpAffine(mask, transform, (400, 320), flags=cv2.INTER_NEAREST)
    first = detect_surface_features(image, mask); last = detect_surface_features(moved, moved_mask)
    result = match_surface_features(first, last)
    paired = [row for row in result['rows'] if row['status'] == 'appearance_pair']
    assert len(paired) >= 20
    errors = [np.linalg.norm(np.asarray(row['current_point_px'])-row['reference_point_px']-[17, 11]) for row in paired]
    assert max(errors) < .2
    assert len(result['rows']) == len(first['keypoints'])


def test_rotation_scale_and_perspective_keep_accurate_matched_points():
    image, mask = fixture()
    transform = np.array([[1.03, -.13, 26.], [.11, .97, -10.], [.0002, -.0001, 1.]])
    moved = cv2.warpPerspective(image, transform, (400, 320), borderValue=128)
    moved_mask = cv2.warpPerspective(mask, transform, (400, 320), flags=cv2.INTER_NEAREST)
    first = detect_surface_features(image, mask); last = detect_surface_features(moved, moved_mask)
    result = match_surface_features(first, last); errors = []
    for row in result['rows']:
        if row['status'] != 'appearance_pair':continue
        point = np.r_[np.asarray(row['reference_point_px'])-.5, 1.]
        projected = transform @ point
        errors.append(np.linalg.norm(projected[:2]/projected[2]+.5-row['current_point_px']))
    assert len(errors) >= 15 and np.percentile(errors, 95) < 1.


def test_missing_frame_keeps_every_reference_slot_and_does_not_poison_recovery():
    image, mask = fixture(); reference = detect_surface_features(image, mask)
    absent = detect_surface_features(np.full_like(image, 128), np.zeros_like(mask))
    result = match_surface_features(reference, absent)
    assert result['appearance_pairs'] == 0 and len(result['rows']) == len(reference['keypoints'])
    recovered = match_surface_features(reference, detect_surface_features(image, mask))
    assert recovered['appearance_pairs'] == len(reference['selected_indices'])


def test_partial_occlusion_retains_missing_slots_without_inventing_points():
    image, mask = fixture(); reference = detect_surface_features(image, mask)
    current = image.copy(); current[:, 180:] = 128
    current_mask = mask.copy(); current_mask[:, 180:] = 0
    result = match_surface_features(reference, detect_surface_features(current, current_mask))
    paired = [row for row in result['rows'] if row['status'] == 'appearance_pair']
    assert len(paired) >= 10
    assert all(row['current_point_px'][0] < 180 for row in paired)
    assert len(paired) < len(reference['selected_indices'])
    assert len(result['rows']) == len(reference['keypoints'])


def test_identical_descriptors_at_distinct_locations_are_ambiguous():
    reference = features(np.full((3, 128), 30))
    result = match_surface_features(reference, deepcopy(reference))
    assert result['appearance_pairs'] == 0
    assert all(row['status'] == 'forward_ambiguity' for row in result['rows'])


def test_mutual_matching_prevents_two_references_claiming_one_target():
    a = np.zeros((3, 128)); a[1] = 1; a[2] = 50
    b = np.zeros((3, 128)); b[1] = 70; b[2] = 50
    result = match_surface_features(features(a), features(b))
    targets = [row['current_id'] for row in result['rows'] if row['status'] == 'appearance_pair']
    assert len(targets) == len(set(targets))
    assert result['rows'][1]['status'] == 'not_mutual'


def test_unique_but_unrelated_descriptors_are_rejected():
    first = features(np.zeros((2, 128)))
    last = features(np.array([np.full(128, 100), np.full(128, 200)]))
    result = match_surface_features(first, last)
    assert all(row['status'] == 'appearance_distance_rejected' for row in result['rows'])


def test_duplicate_detector_orientations_do_not_count_as_distinct_points():
    image, mask = fixture(); result = detect_surface_features(image, mask)
    points = result['keypoints'][result['selected_indices'], :2]
    if len(points) > 1:
        distances = np.linalg.norm(points[:, None, :]-points[None, :, :], axis=2)
        assert np.min(distances[np.triu_indices(len(points), 1)]) > 1
    assert len(result['aliases']) == len(result['keypoints'])


def test_featureless_surface_returns_no_fabricated_points():
    image = np.full((128, 128), 128, np.uint8)
    result = detect_surface_features(image, np.full_like(image, 255))
    assert result['keypoints'].shape == (0, 6) and result['descriptors'].shape == (0, 128)
    assert match_surface_features(result, result)['rows'] == []


def test_inputs_remain_unchanged():
    image, mask = fixture(); original, original_mask = image.copy(), mask.copy()
    reference = detect_surface_features(image, mask); saved = deepcopy(reference)
    match_surface_features(reference, reference)
    assert np.array_equal(image, original) and np.array_equal(mask, original_mask)
    assert np.array_equal(reference['keypoints'], saved['keypoints'])
    assert np.array_equal(reference['descriptors'], saved['descriptors'])


@pytest.mark.parametrize('gray,mask', [(np.zeros((16, 16), float), np.zeros((16, 16), np.uint8)),
    (np.zeros((16, 16), np.uint8), np.zeros((15, 16), np.uint8)),
    (np.zeros((8, 16), np.uint8), np.zeros((8, 16), np.uint8))])
def test_invalid_images_are_rejected(gray, mask):
    with pytest.raises(ValueError, match='bounded uint8'):
        detect_surface_features(gray, mask)


def test_oversized_region_is_not_silently_resized():
    image = np.full((600, 600), 128, np.uint8)
    with pytest.raises(ValueError, match='crop pixel budget'):
        detect_surface_features(image, np.full_like(image, 255))


def test_invalid_or_oversized_feature_sets_are_rejected():
    with pytest.raises(ValueError, match='feature arrays'):
        match_surface_features(features(np.zeros((129, 128))), features(np.zeros((2, 128))))
    bad = features(np.zeros((2, 128))); bad['descriptors'][0, 0] = np.nan
    with pytest.raises(ValueError, match='feature arrays'):
        match_surface_features(bad, features(np.zeros((2, 128))))
