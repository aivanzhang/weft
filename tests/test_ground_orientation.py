from copy import deepcopy
import math

import numpy as np
import pytest

from weft.ground_orientation import CAMERA_TO_SIM, estimate_ground_orientation


def evidence(normals=None, count=10, translations=None, scales=None):
    normals = normals or [[0., 0., 1.]]*count
    rows = []
    for i in range(count):
        normal = np.array(normals[i], dtype=float)
        normal /= np.linalg.norm(normal)
        u = np.cross(normal, [0., 1., 0.])
        u /= np.linalg.norm(u)
        v = np.cross(normal, u)
        corners = np.array([-.2*u-.1*v, .2*u-.1*v, -.2*u+.1*v, .2*u+.1*v])
        corners *= scales[i] if scales else 1.
        corners += translations[i] if translations else [0., 0., 0.]
        world = [{'x': 0., 'y': 0., 'z': 0., 'presence': 1., 'visibility': 1.} for _ in range(33)]
        image = [{'x': .5, 'y': .5, 'z': 0., 'presence': 1., 'visibility': 1.} for _ in range(33)]
        for j, point in zip([29, 30, 31, 32], corners @ CAMERA_TO_SIM):
            world[j].update(zip(['x', 'y', 'z'], point.tolist()))
        rows.append({'id': f"{'a'*64}:frame:{i}", 'frame': i, 'time_s': i*.05, 'status': 'tracked', 'world_landmarks': world, 'landmarks': image})
    return {'source_sha256': 'a'*64, 'frames': rows, 'model': {'name': 'independent analytic corners'}}


def test_flat_plane_with_hip_motion_and_scale_changes():
    data = evidence(translations=[[i*.1, -i*.2, i*.3] for i in range(10)], scales=[.01, 1, 3, .5, 20, .2, 1, 1, 1, 1])
    result = estimate_ground_orientation(data)
    assert result['status'] == 'available'
    assert result['normal'] == pytest.approx([0, 0, 1], abs=1e-12)
    assert result['construction_rows'] == 6
    assert result['floor_offset'] is None


def test_known_oblique_plane_and_positive_camera_up_sign():
    normal = np.array([.3, -.4, .8660254037844386])
    result = estimate_ground_orientation(evidence(normals=[(-normal).tolist()]*10))
    assert result['status'] == 'available'
    assert result['normal'] == pytest.approx(normal, abs=1e-10)


def test_later_geometry_is_not_used():
    data = evidence()
    original = estimate_ground_orientation(data)
    for row in data['frames'][6:]:
        row['world_landmarks'] = None
        row['landmarks'] = 'not read'
    result = estimate_ground_orientation(data)
    assert result == original


def test_equal_frame_weight_is_scale_invariant():
    normals = [[0, 0, 1], [.1, 0, .995], [-.1, 0, .995], [0, .1, .995], [0, -.1, .995]]*2
    baseline = estimate_ground_orientation(evidence(normals=normals))
    scaled = estimate_ground_orientation(evidence(normals=normals, scales=[.01, 20, .2, 10, 2]*2))
    assert scaled['diagnostic_normal'] == pytest.approx(baseline['diagnostic_normal'], abs=1e-10)
    assert np.array(scaled['metrics']['normalized_covariance']) == pytest.approx(np.array(baseline['metrics']['normalized_covariance']), abs=1e-12)


@pytest.mark.parametrize('space', ['landmarks', 'world_landmarks'])
@pytest.mark.parametrize('field', ['presence', 'visibility'])
def test_any_endpoint_low_confidence_is_audited(space, field):
    data = evidence()
    data['frames'][0][space][32][field] = .49
    result = estimate_ground_orientation(data)
    assert result['usable_frames'] == 5
    assert result['audit'][0]['reasons'] == ['low_endpoint_confidence']


def test_confidence_threshold_and_inclusive_image_bounds():
    data = evidence()
    for row in data['frames']:
        for j in [29, 30, 31, 32]:
            row['landmarks'][j].update(x=0., y=1., presence=.5, visibility=.5)
            row['world_landmarks'][j].update(presence=.5, visibility=.5)
    assert estimate_ground_orientation(data)['status'] == 'available'


def test_crop_and_confidence_reasons_both_retained():
    data = evidence()
    data['frames'][0]['landmarks'][29].update(x=1.001, visibility=.2)
    result = estimate_ground_orientation(data)
    assert set(result['audit'][0]['reasons']) == {'low_endpoint_confidence', 'foot_image_out_of_bounds'}


@pytest.mark.parametrize('value', [float('nan'), float('inf'), True, None])
def test_invalid_values_rejected_without_nonfinite_output(value):
    data = evidence()
    data['frames'][0]['world_landmarks'][29]['x'] = value
    result = estimate_ground_orientation(data)
    assert result['audit'][0]['reasons'] == ['nonfinite_or_missing_foot_values']


def test_too_few_frames_keeps_only_diagnostic_normal():
    result = estimate_ground_orientation(evidence(count=4))
    assert result['construction_rows'] == 2
    assert result['status'] == 'unavailable'
    assert result['normal'] is None
    assert result['diagnostic_normal'] == pytest.approx([0, 0, 1])


def test_collinear_feet_do_not_define_plane():
    data = evidence()
    for row in data['frames']:
        for i, index in enumerate([29, 30, 31, 32]):
            row['world_landmarks'][index].update(x=float(i), y=0., z=0.)
    result = estimate_ground_orientation(data)
    assert result['status'] == 'unavailable'
    assert result['usable_frames'] == 0
    assert all(a['reasons'] == ['per_frame_rank2_unresolved'] for a in result['audit'])


def test_narrow_spread_rejects_aggregate_rank_screen():
    data = evidence()
    for row in data['frames']:
        for index in [29, 30, 31, 32]:
            row['world_landmarks'][index]['z'] *= .1
    result = estimate_ground_orientation(data)
    assert 'aggregate_rank2_spread_insufficient' in result['reasons']


def test_nonplanar_tetrahedron_rejects_thickness():
    data = evidence()
    points = [[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]]
    for row in data['frames']:
        for index, point in zip([29, 30, 31, 32], points):
            row['world_landmarks'][index].update(zip(['x', 'y', 'z'], point))
    result = estimate_ground_orientation(data)
    assert result['status'] == 'unavailable'
    assert 'aggregate_plane_thickness_excessive' in result['reasons']
    assert result['normal'] is None


def test_source_and_used_frame_provenance_preserved():
    data = evidence()
    data['source_sha256'] = 'a'*64
    data['frames'][0]['landmarks'][29]['presence'] = .1
    result = estimate_ground_orientation(data)
    assert result['source_sha256'] == data['source_sha256']
    assert result['construction_source_frames'] == list(range(6))
    assert result['construction_evidence_ids'] == [f"{'a'*64}:frame:{i}" for i in range(1, 6)]
    assert result['selected_evidence_ids'] == [f"{'a'*64}:frame:{i}" for i in range(6)]
    assert result['excluded_later_source_frames'] == list(range(6, 10))


def test_steep_plane_is_unavailable():
    result = estimate_ground_orientation(evidence(normals=[[1, 0, .1]]*10))
    assert result['status'] == 'unavailable'
    assert 'normal_too_far_from_camera_up' in result['reasons']


def test_diverging_normals_reject_dispersion_without_threshold_tuning():
    radians = math.radians(40)
    normals = [[math.sin(radians)*sign, 0, math.cos(radians)] for sign in [-1, 1]*5]
    result = estimate_ground_orientation(evidence(normals=normals))
    assert result['status'] == 'unavailable'
    assert 'median_normal_dispersion_excessive' in result['reasons']
    assert 'p90_normal_dispersion_excessive' in result['reasons']


def test_empty_input_and_order_and_bound():
    assert estimate_ground_orientation({'source_sha256': 'a'*64, 'frames': []})['status'] == 'unavailable'
    data = evidence()
    data['frames'][1]['frame'] = 0
    with pytest.raises(ValueError, match='increasing'):
        estimate_ground_orientation(data)
    with pytest.raises(ValueError, match='900'):
        estimate_ground_orientation(evidence(count=901))


def test_input_and_returned_settings_do_not_mutate_subsequent_calls():
    data = evidence()
    saved = deepcopy(data)
    first = estimate_ground_orientation(data)
    first['settings']['minimum_usable_frames'] = 0
    assert data == saved
    assert estimate_ground_orientation(data)['settings']['minimum_usable_frames'] == 3


@pytest.mark.parametrize('value', [None, '', 'a'*63, 'g'*64, True])
def test_invalid_source_hash_metadata_fails(value):
    data = evidence()
    data['source_sha256'] = value
    with pytest.raises(ValueError, match='64-hex'):
        estimate_ground_orientation(data)


@pytest.mark.parametrize('value', [None, float('inf'), float('nan'), True, -1, 60.01, 0.])
def test_invalid_or_nonincreasing_time_metadata_fails(value):
    data = evidence()
    data['frames'][1]['time_s'] = value
    with pytest.raises(ValueError, match='times'):
        estimate_ground_orientation(data)


def test_bad_evidence_id_in_excluded_tail_still_fails_metadata():
    data = evidence()
    data['frames'][-1]['id'] = 'wrong:frame:9'
    with pytest.raises(ValueError, match='evidence IDs'):
        estimate_ground_orientation(data)


@pytest.mark.parametrize('status', ['lost', 'invalid', 'not_detected', None])
def test_stale_geometry_not_used(status):
    data = evidence()
    data['frames'][0]['status'] = status
    result = estimate_ground_orientation(data)
    assert result['audit'][0]['reasons'] == ['observation_not_tracked']


@pytest.mark.parametrize('space,reason', [
    ('world_landmarks', 'world_coordinate_magnitude_excessive'),
    ('landmarks', 'image_coordinate_magnitude_excessive'),
])
def test_huge_finite_coordinates_rejected_before_covariance(space, reason):
    data = evidence()
    data['frames'][0][space][29]['x'] = 1e308
    result = estimate_ground_orientation(data)
    assert reason in result['audit'][0]['reasons']
    assert result['usable_frames'] == 5


def test_tracked_partial_geometry_remains_eligible():
    data = evidence()
    data['frames'][0]['status'] = 'tracked_partial'
    assert estimate_ground_orientation(data)['usable_frames'] == 6
