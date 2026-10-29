"""Fixed-reference appearance correspondences, not physical point identities."""

import cv2
import numpy as np

from weft.regional_features import MAX_SOURCE_PIXELS, regional_sift


CONTRACT = {
    'method': 'masked-fixed-reference-sift-v1', 'maximum_features': 128,
    'mask_inset_px': 3, 'duplicate_location_radius_px': 1.,
    'maximum_descriptor_distance': 220., 'mutual_ratio': .7,
    'coordinates': 'source pixel-cell edges; OpenCV point coordinates plus .5',
    'scope': 'Appearance hypotheses only; supplied masks, 3D identity and rigidity remain unverified',
}


def detect_surface_features(gray, mask):
    if (not isinstance(gray, np.ndarray) or not isinstance(mask, np.ndarray)
            or gray.dtype != np.uint8 or mask.dtype != np.uint8 or gray.ndim != 2
            or mask.shape != gray.shape or min(gray.shape) < 16 or gray.size > MAX_SOURCE_PIXELS):
        raise ValueError('Surface features require matching bounded uint8 images and masks')
    inset = cv2.erode(mask, np.ones((7, 7), dtype=np.uint8), borderType=cv2.BORDER_CONSTANT, borderValue=0)
    points, descriptors, audit = regional_sift(gray, inset, maximum_features=CONTRACT['maximum_features'])
    values = np.array([[p.pt[0]+.5, p.pt[1]+.5, p.size, p.angle, p.response, p.octave]
                      for p in points], dtype=np.float64).reshape(-1, 6)
    if descriptors is None:
        descriptors = np.empty((0, 128), dtype=np.float32)
    selected, aliases = [], []
    for index, point in enumerate(values):
        duplicate = next((other for other in selected
            if np.linalg.norm(values[other, :2]-point[:2]) <= CONTRACT['duplicate_location_radius_px']), None)
        if duplicate is None:
            selected.append(index); aliases.append(index)
        else:
            aliases.append(duplicate)
    return {'method': CONTRACT['method'], 'keypoints': values, 'descriptors': descriptors,
        'selected_indices': selected, 'aliases': aliases,
        'audit': {**audit, 'raw_feature_count': len(values), 'unique_location_count': len(selected),
            'mask_inset_px': CONTRACT['mask_inset_px'],
            'scope': 'No hole filling or inferred boundary. Descriptor support can extend beyond the inset mask.'}}


def _validate(features):
    if not isinstance(features, dict) or features.get('method') != CONTRACT['method']:
        raise ValueError('Use explicit fixed-reference surface features')
    points, descriptors = features.get('keypoints'), features.get('descriptors')
    selected, aliases = features.get('selected_indices'), features.get('aliases')
    if (not isinstance(points, np.ndarray) or points.ndim != 2 or points.shape[1:] != (6,)
            or len(points) > CONTRACT['maximum_features'] or not np.isfinite(points).all()
            or (points[:, :2] < 0).any() or (points[:, :2] > 16384).any()
            or not isinstance(descriptors, np.ndarray) or descriptors.dtype != np.float32
            or descriptors.shape != (len(points), 128) or not np.isfinite(descriptors).all()
            or (descriptors < 0).any() or (descriptors > 255).any()
            or not isinstance(selected, list) or not isinstance(aliases, list)
            or len(aliases) != len(points) or any(type(i) is not int or not 0 <= i < len(points) for i in selected)
            or len(set(selected)) != len(selected) or any(type(i) is not int or i not in selected for i in aliases)
            or any(aliases[i] != i for i in selected)):
        raise ValueError('Invalid or unbounded feature arrays or location identities')
    return points, descriptors, selected, aliases


def match_surface_features(reference, current):
    points, descriptors, selected, aliases = _validate(reference)
    other_points, other_descriptors, other_selected, _ = _validate(current)
    rows = [{'reference_id': i, 'reference_point_px': point[:2].tolist(), 'current_id': None,
        'current_point_px': None, 'status': 'duplicate_reference_location' if aliases[i] != i else 'insufficient_features',
        'alias_of': aliases[i] if aliases[i] != i else None} for i, point in enumerate(points)]
    if min(len(selected), len(other_selected)) >= 2:
        matcher = cv2.BFMatcher(cv2.NORM_L2)
        forward = matcher.knnMatch(descriptors[selected], other_descriptors[other_selected], k=2)
        reverse = matcher.knnMatch(other_descriptors[other_selected], descriptors[selected], k=2)
        for compact_index, pair in enumerate(forward):
            first, second = pair; ref_index = selected[compact_index]
            opposite, competitor = reverse[first.trainIdx]
            row = rows[ref_index]
            row.update(nearest_current_id=other_selected[first.trainIdx],
                descriptor_distance=float(first.distance), second_distance=float(second.distance),
                reverse_distance=float(opposite.distance), reverse_second_distance=float(competitor.distance))
            if first.distance > CONTRACT['maximum_descriptor_distance']:
                row['status'] = 'appearance_distance_rejected'
            elif not first.distance < CONTRACT['mutual_ratio']*second.distance:
                row['status'] = 'forward_ambiguity'
            elif opposite.trainIdx != compact_index:
                row['status'] = 'not_mutual'
            elif not opposite.distance < CONTRACT['mutual_ratio']*competitor.distance:
                row['status'] = 'reverse_ambiguity'
            else:
                target = other_selected[first.trainIdx]
                row.update(status='appearance_pair', current_id=target, current_point_px=other_points[target, :2].tolist())
    return {'method': CONTRACT['method'], 'rows': rows, 'raw_reference_features': len(points),
        'unique_reference_locations': len(selected), 'raw_current_features': len(other_points),
        'unique_current_locations': len(other_selected),
        'appearance_pairs': sum(row['status'] == 'appearance_pair' for row in rows),
        'physical_point_identity_verified': False, 'rigidity_verified': False}
