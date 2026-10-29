"""Bound feature detection by image area, then spend the quota inside the region."""

import cv2
import numpy as np


MAX_SOURCE_PIXELS = 4_194_304
MAX_CROP_PIXELS = 262_144
MAX_FEATURES = 1200
PADDING_PX = 32


def regional_sift(gray, mask, *, maximum_features=MAX_FEATURES):
    if (not isinstance(gray, np.ndarray) or not isinstance(mask, np.ndarray)
            or gray.dtype != np.uint8 or mask.dtype != np.uint8
            or gray.ndim != 2 or mask.shape != gray.shape
            or min(gray.shape) < 16 or gray.size > MAX_SOURCE_PIXELS):
        raise ValueError('Region detection requires matching bounded uint8 grayscale image and mask')
    if type(maximum_features) is not int or not 1 <= maximum_features <= MAX_FEATURES:
        raise ValueError('Regional feature quota must be an integer between 1 and 1200')
    audit = {'method': 'regional-sift-v1', 'source_shape': list(gray.shape),
        'mask_pixels': int(cv2.countNonZero(mask)), 'maximum_features': maximum_features,
        'maximum_crop_pixels': MAX_CROP_PIXELS, 'padding_px': PADDING_PX,
        'contrast_threshold': .02, 'edge_threshold': 10,
        'scope': 'Mask constrains feature centers, not the complete descriptor support or physical identity.'}
    x, y, width, height = cv2.boundingRect(mask)
    if not width or not height:
        return [], None, audit | {'crop_xyxy': None, 'crop_pixels': 0, 'regional_candidates': 0, 'returned_features': 0}
    x0, y0 = max(0, x-PADDING_PX), max(0, y-PADDING_PX)
    x1, y1 = min(gray.shape[1], x+width+PADDING_PX), min(gray.shape[0], y+height+PADDING_PX)
    area = (x1-x0)*(y1-y0)
    if area > MAX_CROP_PIXELS:
        raise ValueError('Regional detection exceeds the crop pixel budget; do not silently downsample')
    crop = gray[y0:y1, x0:x1]
    # OpenCV caps features before masking; bound the crop instead, then cap regional candidates.
    detector = cv2.SIFT_create(nfeatures=0, contrastThreshold=.02, edgeThreshold=10)
    candidates = detector.detect(crop, mask[y0:y1, x0:x1])
    selected = sorted(candidates, key=lambda p: (-p.response, p.pt[1], p.pt[0], p.size, p.angle, p.octave))[:maximum_features]
    points, descriptors = detector.compute(crop, selected)
    points = [cv2.KeyPoint(p.pt[0]+x0, p.pt[1]+y0, p.size, p.angle, p.response, p.octave, p.class_id)
              for p in points]
    return points, descriptors, audit | {'crop_xyxy': [x0, y0, x1, y1], 'crop_pixels': area,
        'regional_candidates': len(candidates), 'returned_features': len(points)}
