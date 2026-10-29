from copy import deepcopy
import math

import pytest

from weft.camera_registration_grading import grade_camera_registration


def fixture():
    sha = 'a'*64
    points = [[10, 10], [50, 10], [10, 50], [50, 50]]
    registration = {
        'profile': 'fixed-reference-registration-v1', 'source_sha256': sha,
        'width': 100, 'height': 100, 'timestamps_s': [0., .1, .2], 'reference_frame': 0,
        'ready': True, 'status': 'ready', 'reasons': [],
        'frames': [{'frame': i, 'time_s': i/10, 'status': 'registered',
                    'current_to_reference': [[1, 0, -3 if i else 0], [0, 1, -4 if i else 0], [0, 0, 1]],
                    'method': 'reference' if i == 0 else 'orb'} for i in range(3)],
    }
    labels = {
        'video_sha256': sha, 'anchor_relative_frame': 0,
        'landmarks': [{'id': f'p{j}'} for j in range(4)],
        'frames': [{'relative_frame': i, 'absolute_source_frame': 100+i,
                    'landmarks': [{'id': f'p{j}', 'xy': [x+(3 if i else 0), y+(4 if i else 0)],
                                   'visibility': 'visible', 'uncertainty_xy_px': [1, 1]}
                                  for j, (x, y) in enumerate(points)]} for i in range(3)],
    }
    mapping = {'video_sha256': sha, 'decoded_frames': 3, 'selected_relative_frames': [0, 1, 2],
               'frames': [{'relative_frame': i, 'absolute_source_frame': 100+i,
                           'timestamp_s': i/10, 'dimensions': [100, 100]} for i in range(3)]}
    return registration, labels, mapping


def test_perfect_correction_and_raw_drift_all_slots():
    result = grade_camera_registration(*fixture())
    summary = result['summary']
    assert summary['slots'] == 8
    assert summary['raw_rmse_px'] == 5
    assert summary['corrected_rmse_px'] == 0
    assert summary['corrected_coverage'] == 1
    assert summary['raw_within_tolerance_count'] == 0
    assert summary['corrected_within_tolerance_count'] == 8
    assert all(row['corrected_error_px'] is None for row in result['rows'] if row['anchor'])


def test_wrong_transform_not_selected_away():
    registration, labels, mapping = fixture()
    registration['frames'][2]['current_to_reference'][0][2] = 7
    result = grade_camera_registration(registration, labels, mapping)
    assert result['summary']['corrected_rmse_px'] == pytest.approx(math.sqrt(50))
    assert result['summary']['corrected_within_tolerance_count'] == 4
    assert result['summary']['corrected_coverage'] == 1


def test_identity_transform_reports_unchanged_error_and_tolerance():
    registration, labels, mapping = fixture()
    for frame in registration['frames']:
        frame['current_to_reference'] = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    result = grade_camera_registration(registration, labels, mapping)
    for row in result['rows']:
        if not row['anchor']:
            assert row['raw_error_px'] == row['corrected_error_px'] == 5
            assert row['raw_tolerance_px'] == row['corrected_tolerance_px'] == pytest.approx(2*math.sqrt(2))


def test_scale_propagates_uncertainty_without_changing_raw_bound():
    registration, labels, mapping = fixture()
    registration['frames'][1]['current_to_reference'] = [[2, 0, 0], [0, 3, 0], [0, 0, 1]]
    result = grade_camera_registration(registration, labels, mapping)
    rows = [row for row in result['rows'] if row['relative_frame'] == 1]
    assert all(row['raw_tolerance_px'] == pytest.approx(2*math.sqrt(2)) for row in rows)
    assert all(row['corrected_tolerance_px'] == pytest.approx(math.sqrt(13)+math.sqrt(2)) for row in rows)


def test_missing_transform_and_unavailable_report_are_not_zero_errors():
    registration, labels, mapping = fixture()
    registration.update(ready=False, status='unavailable', reasons=['low_coverage'])
    registration['frames'][2].update(status='unavailable', current_to_reference=None)
    result = grade_camera_registration(registration, labels, mapping)
    assert result['registration_ready'] is False
    assert result['registration_status'] == 'unavailable'
    assert result['summary']['corrected_coverage'] == .5
    assert result['summary']['raw_coverage'] == 1
    assert result['summary']['corrected_rmse_px'] == 0
    assert all(row['corrected_error_px'] is None for row in result['rows'] if row['relative_frame'] == 2)


def test_no_transform_has_null_rmse():
    registration, labels, mapping = fixture()
    for row in registration['frames']:
        row['current_to_reference'] = None
    result = grade_camera_registration(registration, labels, mapping)
    assert result['summary']['corrected_rmse_px'] is None
    assert result['summary']['corrected_coverage'] == 0
    assert result['summary']['corrected_within_tolerance_fraction'] is None


def test_missing_label_and_missing_anchor_remain_slots():
    registration, labels, mapping = fixture()
    labels['frames'][0]['landmarks'][0].update(visibility='unknown', xy=None)
    labels['frames'][1]['landmarks'].pop()
    result = grade_camera_registration(registration, labels, mapping)
    assert result['summary']['slots'] == 8
    assert result['summary']['raw_comparable_slots'] == 5
    assert result['summary']['corrected_slots'] == 5
    assert sum('anchor_label_missing' in row['reasons'] for row in result['rows']) == 2
    assert sum('current_label_missing' in row['reasons'] for row in result['rows']) == 1


@pytest.mark.parametrize('which', [0, 1, 2])
def test_source_hash_mismatch_is_rejected(which):
    values = fixture()
    values[which]['source_sha256' if which == 0 else 'video_sha256'] = 'b'*64
    with pytest.raises(ValueError, match='hashes'):
        grade_camera_registration(*values)


@pytest.mark.parametrize('location', ['row', 'mapping', 'global'])
def test_timestamp_mismatch_is_rejected(location):
    registration, labels, mapping = fixture()
    if location == 'row':
        registration['frames'][1]['time_s'] += .01
    elif location == 'mapping':
        mapping['frames'][1]['timestamp_s'] += .01
    else:
        registration['timestamps_s'][1] = 0
    with pytest.raises(ValueError, match='timestamp'):
        grade_camera_registration(registration, labels, mapping)


@pytest.mark.parametrize('matrix,reason', [
    ([[1, 0, 0], [0, 1, 0], [.001, 0, 1]], 'projective_transform_not_supported'),
    ([[1, 0, 0], [0, 0, 0], [0, 0, 1]], 'singular_transform'),
    ([[1, 0, float('nan')], [0, 1, 0], [0, 0, 1]], 'transform_nonfinite_or_excessive'),
    ([[1, 0, 1e308], [0, 1, 0], [0, 0, 1]], 'transform_nonfinite_or_excessive'),
    ([[1, 0, True], [0, 1, 0], [0, 0, 1]], 'transform_nonfinite_or_excessive'),
    ([[1, 0, 0], [0, 1, 0]], 'transform_invalid_shape'),
])
def test_invalid_matrix_is_unavailable_not_a_free_pass(matrix, reason):
    registration, labels, mapping = fixture()
    registration['frames'][1]['current_to_reference'] = matrix
    result = grade_camera_registration(registration, labels, mapping)
    assert result['summary']['corrected_coverage'] == .5
    assert all(reason in row['reasons'] for row in result['rows'] if row['relative_frame'] == 1)


def test_omitted_frozen_frame_cannot_improve_coverage():
    registration, labels, mapping = fixture()
    labels['frames'].pop()
    with pytest.raises(ValueError, match='coverage'):
        grade_camera_registration(registration, labels, mapping)


def test_duplicate_identity_rejected():
    registration, labels, mapping = fixture()
    labels['frames'][1]['landmarks'][1]['id'] = 'p0'
    with pytest.raises(ValueError, match='duplicate'):
        grade_camera_registration(registration, labels, mapping)


def test_inputs_not_mutated():
    values = fixture()
    frozen = deepcopy(values)
    grade_camera_registration(*values)
    assert values == frozen
