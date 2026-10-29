"""Independent held-out landmark grading; no registration fitting or frame selection."""

import math
import re

import numpy as np


GRADE_CONTRACT = {
    'schema_version': 1,
    'maximum_frames': 1800,
    'timestamp_tolerance_s': 1e-6,
    'maximum_absolute_matrix_value': 1e6,
    'anchor_scored': False,
    'raw_tolerance': 'hypot(current uncertainty xy) + hypot(anchor uncertainty xy)',
    'corrected_tolerance': 'max norm(A @ current uncertainty box corners) + hypot(anchor uncertainty xy)',
    'scope': 'Authored approximate source-pixel tolerances, not expert truth or metric calibration',
}


def _finite(value):
    return type(value) in (float, int) and math.isfinite(value)


def _indexed(rows, key, name):
    if not isinstance(rows, list):
        raise ValueError(f'{name} must be a list')
    result = {}
    for row in rows:
        if not isinstance(row, dict) or key not in row or type(row[key]) not in (str, int) or row[key] in result:
            raise ValueError(f'{name} has missing or duplicate identities')
        result[row[key]] = row
    return result


def _label(row, width, height):
    if row is None or row.get('visibility') != 'visible' or row.get('xy') is None:
        return None
    xy, uncertainty = row.get('xy'), row.get('uncertainty_xy_px')
    if not isinstance(xy, list) or len(xy) != 2 or not all(_finite(v) for v in xy):
        raise ValueError('Visible label xy must contain two finite numbers')
    if not 0 <= xy[0] < width or not 0 <= xy[1] < height:
        raise ValueError('Visible label is outside native image bounds')
    if not isinstance(uncertainty, list) or len(uncertainty) != 2 or not all(_finite(v) and 0 <= v <= max(width, height) for v in uncertainty):
        raise ValueError('Visible label requires finite nonnegative pixel uncertainty')
    return np.asarray(xy, dtype=float), np.asarray(uncertainty, dtype=float)


def _matrix(row):
    if row is None or row.get('status') != 'registered':
        return None, 'registration_unavailable'
    matrix = row.get('current_to_reference')
    if matrix is None:
        return None, 'transform_missing'
    if not isinstance(matrix, list) or len(matrix) != 3 or any(not isinstance(v, list) or len(v) != 3 for v in matrix):
        return None, 'transform_invalid_shape'
    if any(not _finite(v) or abs(v) > GRADE_CONTRACT['maximum_absolute_matrix_value'] for line in matrix for v in line):
        return None, 'transform_nonfinite_or_excessive'
    if matrix[2] != [0, 0, 1]:
        return None, 'projective_transform_not_supported'
    result = np.asarray(matrix, dtype=float)
    if abs(float(np.linalg.det(result[:2, :2]))) <= 1e-12:
        return None, 'singular_transform'
    return result, None


def _summary(rows):
    raw = [row for row in rows if row['raw_error_px'] is not None]
    corrected = [row for row in rows if row['corrected_error_px'] is not None]
    return {
        'slots': len(rows), 'raw_comparable_slots': len(raw), 'corrected_slots': len(corrected),
        'raw_coverage': len(raw)/len(rows) if rows else None,
        'corrected_coverage': len(corrected)/len(rows) if rows else None,
        'raw_rmse_px': math.sqrt(sum(row['raw_error_px']**2 for row in raw)/len(raw)) if raw else None,
        'corrected_rmse_px': math.sqrt(sum(row['corrected_error_px']**2 for row in corrected)/len(corrected)) if corrected else None,
        'raw_within_tolerance_count': sum(row['raw_within_tolerance'] for row in raw),
        'corrected_within_tolerance_count': sum(row['corrected_within_tolerance'] for row in corrected),
        'raw_within_tolerance_fraction': sum(row['raw_within_tolerance'] for row in raw)/len(raw) if raw else None,
        'corrected_within_tolerance_fraction': sum(row['corrected_within_tolerance'] for row in corrected)/len(corrected) if corrected else None,
        'raw_rmse_on_corrected_slots_px': math.sqrt(sum(row['raw_error_px']**2 for row in corrected)/len(corrected)) if corrected else None,
    }


def grade_camera_registration(registration, labels, source_mapping):
    """Grade every declared nonanchor label slot against the fixed anchor labels."""
    source_hash = labels.get('video_sha256')
    if not isinstance(source_hash, str) or re.fullmatch(r'[0-9a-fA-F]{64}', source_hash) is None:
        raise ValueError('Labels require a 64-hex video_sha256')
    if registration.get('source_sha256') != source_hash or source_mapping.get('video_sha256') != source_hash:
        raise ValueError('Registration, labels and mapping source hashes must match')
    if registration.get('profile') != 'fixed-reference-registration-v1':
        raise ValueError('Unsupported registration profile')
    anchor = labels.get('anchor_relative_frame')
    if type(anchor) is not int or registration.get('reference_frame') != anchor:
        raise ValueError('Reference frame must match the frozen label anchor')
    width, height = registration.get('width'), registration.get('height')
    if any(type(v) is not int or not 1 <= v <= 8192 for v in (width, height)):
        raise ValueError('Invalid native image dimensions')
    frames = registration.get('frames')
    times = registration.get('timestamps_s')
    if not isinstance(frames, list) or not 1 <= len(frames) <= GRADE_CONTRACT['maximum_frames']:
        raise ValueError('Registration frame count exceeds bounds')
    if not isinstance(times, list) or len(times) != len(frames) or any(not _finite(t) or not 0 <= t <= 60 for t in times) or any(a >= b for a, b in zip(times, times[1:])):
        raise ValueError('Registration timestamps must be finite and strictly increasing within 60 seconds')
    for index, row in enumerate(frames):
        if not isinstance(row, dict) or type(row.get('frame')) is not int or row['frame'] != index:
            raise ValueError('Registration must preserve all ordered frame slots')
        if not _finite(row.get('time_s')) or abs(row['time_s']-times[index]) > GRADE_CONTRACT['timestamp_tolerance_s']:
            raise ValueError('Registration frame timestamp mismatch')
    if type(registration.get('ready')) is not bool or registration.get('status') not in ('ready', 'unavailable') or registration['ready'] != (registration['status'] == 'ready'):
        raise ValueError('Registration ready/status metadata is inconsistent')
    label_frames = _indexed(labels.get('frames'), 'relative_frame', 'Label frames')
    mapping = _indexed(source_mapping.get('frames'), 'relative_frame', 'Source mapping')
    if source_mapping.get('decoded_frames') != len(frames) or set(mapping) != set(label_frames) or source_mapping.get('selected_relative_frames') != list(mapping):
        raise ValueError('Frozen source frame coverage mismatch')
    identities = list(_indexed(labels.get('landmarks'), 'id', 'Landmark definitions'))
    if not identities or len(identities) > 100 or anchor not in label_frames:
        raise ValueError('Labels require bounded landmark identities and an anchor')
    for index, row in label_frames.items():
        if type(index) is not int or not 0 <= index < len(frames) or index not in mapping:
            raise ValueError('Frozen label frame is missing from registration or source mapping')
        mapped = mapping[index]
        if mapped.get('dimensions') != [width, height]:
            raise ValueError('Source mapping native dimensions mismatch')
        if not _finite(mapped.get('timestamp_s')) or abs(mapped['timestamp_s']-times[index]) > GRADE_CONTRACT['timestamp_tolerance_s']:
            raise ValueError('Frozen source timestamp mismatch')
        if mapped.get('absolute_source_frame') != row.get('absolute_source_frame'):
            raise ValueError('Absolute source frame mapping mismatch')
    anchor_labels = _indexed(label_frames[anchor].get('landmarks'), 'id', 'Anchor landmarks')
    details = []
    for index, frame in label_frames.items():
        current = _indexed(frame.get('landmarks'), 'id', 'Frame landmarks')
        if set(current)-set(identities) or set(anchor_labels)-set(identities):
            raise ValueError('Undeclared landmark identity')
        matrix, transform_reason = _matrix(frames[index])
        for identity in identities:
            row = {'relative_frame': index, 'landmark_id': identity, 'anchor': index == anchor,
                   'raw_error_px': None, 'corrected_error_px': None,
                   'raw_tolerance_px': None, 'corrected_tolerance_px': None,
                   'raw_within_tolerance': None, 'corrected_within_tolerance': None,
                   'raw_delta_xy_px': None, 'corrected_delta_xy_px': None, 'reasons': []}
            reference, observation = _label(anchor_labels.get(identity), width, height), _label(current.get(identity), width, height)
            if index == anchor:
                row['status'] = 'anchor_unscored'
            else:
                if reference is None:
                    row['reasons'].append('anchor_label_missing')
                if observation is None:
                    row['reasons'].append('current_label_missing')
                if transform_reason:
                    row['reasons'].append(transform_reason)
                if reference is not None and observation is not None:
                    target, target_uncertainty = reference
                    point, uncertainty = observation
                    delta = point-target
                    row['raw_delta_xy_px'] = delta.tolist()
                    row['raw_error_px'] = float(np.linalg.norm(delta))
                    row['raw_tolerance_px'] = float(np.linalg.norm(uncertainty)+np.linalg.norm(target_uncertainty))
                    row['raw_within_tolerance'] = row['raw_error_px'] <= row['raw_tolerance_px']
                    if matrix is not None:
                        corrected = matrix[:2, :2] @ point+matrix[:2, 2]-target
                        corners = [matrix[:2, :2] @ (uncertainty*[sx, sy]) for sx in (-1, 1) for sy in (-1, 1)]
                        row['corrected_delta_xy_px'] = corrected.tolist()
                        row['corrected_error_px'] = float(np.linalg.norm(corrected))
                        row['corrected_tolerance_px'] = float(max(np.linalg.norm(corner) for corner in corners)+np.linalg.norm(target_uncertainty))
                        row['corrected_within_tolerance'] = row['corrected_error_px'] <= row['corrected_tolerance_px']
                row['status'] = 'scored' if row['corrected_error_px'] is not None else 'unavailable'
            details.append(row)
    scored = [row for row in details if not row['anchor']]
    return {'schema_version': 1, 'source_sha256': source_hash, 'reference_frame': anchor,
            'registration_ready': registration['ready'], 'registration_status': registration['status'],
            'registration_reasons': registration.get('reasons', []), 'contract': dict(GRADE_CONTRACT),
            'summary': _summary(scored), 'rows': details,
            'per_frame': [{'relative_frame': index, **_summary([r for r in scored if r['relative_frame'] == index])}
                          for index in label_frames if index != anchor],
            'scope': 'Conditional landmark alignment only; incomplete coverage and unavailable registration remain explicit'}
