"""Immutable regional color evidence supplements, but never proves, object identity."""

import cv2
import numpy as np


COLOR_CONTRACT = {
    "method": "saturation-weighted-circular-hue-v1",
    "hue_bins": 36,
    "minimum_saturation": 40,
    "minimum_value": 40,
    "minimum_source_chromatic_fraction": .15,
    "minimum_retained_chromatic_fraction": .5,
    "minimum_histogram_intersection": .5,
}


def color_signature(bgr):
    if bgr.dtype != np.uint8 or bgr.ndim != 3 or bgr.shape[2] != 3 or not bgr.size:
        raise ValueError("Color evidence requires a nonempty uint8 BGR region")
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    mask = (hsv[:, :, 1] >= COLOR_CONTRACT["minimum_saturation"]) & (hsv[:, :, 2] >= COLOR_CONTRACT["minimum_value"])
    weights = hsv[:, :, 1][mask].astype(float) / 255
    histogram = np.bincount(hsv[:, :, 0][mask] // 5, weights=weights, minlength=36)
    # Circular smoothing tolerates small lighting/codec shifts, including red at the hue seam.
    histogram = .5 * histogram + .25 * (np.roll(histogram, 1) + np.roll(histogram, -1))
    total = histogram.sum()
    if total:
        histogram /= total
    return {"chromatic_fraction": float(mask.mean()), "histogram": histogram.tolist()}


def compare_color(source, candidate):
    if source["chromatic_fraction"] < COLOR_CONTRACT["minimum_source_chromatic_fraction"]:
        return {"status": "unavailable_achromatic_source", "passes": None,
                "scope": "No usable source color evidence; grayscale geometry remains the only match check"}
    intersection = float(np.minimum(source["histogram"], candidate["histogram"]).sum())
    retained = candidate["chromatic_fraction"] / source["chromatic_fraction"]
    passes = (intersection >= COLOR_CONTRACT["minimum_histogram_intersection"]
              and retained >= COLOR_CONTRACT["minimum_retained_chromatic_fraction"])
    return {"status": "consistent" if passes else "appearance_changed", "passes": bool(passes),
            "histogram_intersection": intersection, "retained_chromatic_fraction": retained,
            "scope": "Regional appearance compatibility, not verified visibility, identity or segmentation"}
