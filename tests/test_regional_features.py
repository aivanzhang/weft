import cv2
import numpy as np
import pytest

from weft.regional_features import regional_sift


def textured_region():
    rng = np.random.default_rng(19)
    image = rng.integers(0, 256, (512, 512), dtype=np.uint8)
    patch = rng.integers(0, 256, (96, 96), dtype=np.uint8)
    patch = cv2.GaussianBlur(patch, (0, 0), 1.2)
    image[192:288, 192:288] = ((patch.astype(float)-128)*.3+128).astype(np.uint8)
    mask = np.zeros_like(image); mask[200:280, 200:280] = 255
    return image, mask


def test_outside_features_cannot_exhaust_regional_quota():
    image, mask = textured_region()
    before, _ = cv2.SIFT_create(nfeatures=24, contrastThreshold=.02, edgeThreshold=10).detectAndCompute(image, mask)
    points, descriptors, audit = regional_sift(image, mask, maximum_features=24)
    assert len(before) == 0
    assert 6 <= len(points) <= 24
    assert descriptors.shape == (len(points), 128)
    assert audit['regional_candidates'] >= len(points)
    assert all(mask[round(p.pt[1]), round(p.pt[0])] for p in points)


def test_mapping_preserves_translation_and_input_pixels():
    image, mask = textured_region(); original, original_mask = image.copy(), mask.copy()
    points, descriptors, audit = regional_sift(image, mask)
    padded = np.pad(image, ((17, 23), (29, 31)), constant_values=128)
    padded_mask = np.pad(mask, ((17, 23), (29, 31)))
    moved, moved_descriptors, moved_audit = regional_sift(padded, padded_mask)
    assert np.allclose(np.array([p.pt for p in moved])-np.array([29, 17]), [p.pt for p in points], atol=2e-5)
    assert np.array_equal(descriptors, moved_descriptors)
    assert np.array_equal(image, original) and np.array_equal(mask, original_mask)
    assert np.array_equal(np.array(moved_audit['crop_xyxy'])-np.array([29, 17, 29, 17]), audit['crop_xyxy'])


def test_feature_quota_is_applied_before_descriptor_creation():
    image, mask = textured_region()
    points, descriptors, audit = regional_sift(image, mask, maximum_features=3)
    assert len(points) == len(descriptors) == 3
    assert audit['regional_candidates'] > 3


def test_empty_region_is_explicit():
    points, descriptors, audit = regional_sift(np.zeros((64, 64), np.uint8), np.zeros((64, 64), np.uint8))
    assert points == [] and descriptors is None
    assert audit['crop_xyxy'] is None and audit['crop_pixels'] == 0


def test_textureless_region_is_not_fabricated():
    points, descriptors, audit = regional_sift(np.full((64, 64), 128, np.uint8), np.full((64, 64), 255, np.uint8))
    assert points == [] and descriptors is None
    assert audit['regional_candidates'] == 0


@pytest.mark.parametrize('quota', [0, 1201, True, 1.5, None])
def test_invalid_quotas_fail_closed(quota):
    with pytest.raises(ValueError, match='quota'):
        regional_sift(np.zeros((64, 64), np.uint8), np.zeros((64, 64), np.uint8), maximum_features=quota)


@pytest.mark.parametrize('image,mask', [
    (np.zeros((64, 64), np.float32), np.zeros((64, 64), np.uint8)),
    (np.zeros((64, 64), np.uint8), np.zeros((63, 64), np.uint8)),
    (np.zeros((64, 64, 3), np.uint8), np.zeros((64, 64), np.uint8)),
    (np.zeros((8, 64), np.uint8), np.zeros((8, 64), np.uint8)),
])
def test_invalid_images_fail_closed(image, mask):
    with pytest.raises(ValueError, match='bounded uint8'):
        regional_sift(image, mask)


def test_large_region_is_rejected_without_downsampling():
    with pytest.raises(ValueError, match='crop pixel budget'):
        regional_sift(np.zeros((600, 600), np.uint8), np.full((600, 600), 255, np.uint8))
