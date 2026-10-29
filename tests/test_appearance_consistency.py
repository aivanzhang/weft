"""Analytic color/occlusion controls; no real-video coordinates or identities."""

import cv2
import numpy as np
import pytest

from weft.appearance_consistency import color_signature, compare_color


def patch(hue=30, *, fraction=1., saturation=200, value=180):
    hsv = np.zeros((40, 40, 3), np.uint8)
    hsv[:, :, 2] = 120
    hsv[:round(40 * fraction), :, :] = (hue, saturation, value)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


@pytest.mark.parametrize("hue", [0, 30, 90, 150, 179])
def test_hue_wrap_and_small_lighting_changes_remain_compatible(hue):
    source = color_signature(patch(hue))
    for delta in (-3, 0, 3):
        result = compare_color(source, color_signature(patch((hue+delta)%180, saturation=150, value=120)))
        assert result["passes"]


def test_same_grayscale_shape_with_different_color_abstains():
    source = color_signature(patch(30))
    result = compare_color(source, color_signature(patch(110)))
    assert result["status"] == "appearance_changed"
    assert result["histogram_intersection"] == 0


@pytest.mark.parametrize("fraction", [0., .2, .4])
def test_mostly_covered_chromatic_region_abstains(fraction):
    result = compare_color(color_signature(patch()), color_signature(patch(fraction=fraction)))
    assert result["passes"] is False


def test_small_occlusion_keeps_color_support_without_claiming_visibility():
    result = compare_color(color_signature(patch()), color_signature(patch(fraction=.8)))
    assert result["passes"] is True
    assert "not verified visibility" in result["scope"]


def test_achromatic_source_remains_explicitly_unverified():
    result = compare_color(color_signature(patch(saturation=0)), color_signature(patch(110)))
    assert result["passes"] is None
    assert result["status"] == "unavailable_achromatic_source"


def test_same_color_different_object_cannot_be_disambiguated():
    result = compare_color(color_signature(patch()), color_signature(np.rot90(patch())))
    assert result["passes"] is True
    assert "not verified visibility, identity" in result["scope"]


def test_invalid_input_rejected():
    with pytest.raises(ValueError, match="BGR"):
        color_signature(np.zeros((20, 20), np.uint8))
