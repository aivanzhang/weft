from copy import deepcopy

import numpy as np
import pytest

from test_camera_pose import fixture
from test_multiview_pose import pair
from weft.multiview_pose import triangulate_pair, triangulate_views


def views(*, frames=1, radial=0., count=3):
    roots = [[.2+.1*i, .3, 5.] for i in range(frames)]
    first, second, expected = pair(roots=roots, radial=radial)
    result = [first, second]
    for index in range(2, count):
        evidence, camera, _, _ = fixture(roots=roots, radial=radial,
            translations=[([1., .3, .1] if index == 2 else [.3, -1., -.2])]*frames)
        evidence['source_sha256'] = camera['source_sha256'] = ('c' if index == 2 else 'd')*64
        for row in evidence['frames']:
            row['id'] = f"{evidence['source_sha256']}:frame:{row['frame']}"
        result.append({'name': f'view-{index}', 'camera': camera, 'evidence': evidence,
                       'clock_provenance': 'Analytic simultaneous cameras; not physical evidence'})
    return result, expected


def test_two_view_entry_is_exactly_compatible():
    value, _ = views(count=2)
    assert triangulate_views(value) == triangulate_pair(*value)


@pytest.mark.parametrize('radial', [0., -.15, .15])
@pytest.mark.parametrize('count', [3, 4])
def test_all_qualifying_rays_recover_analytic_points_without_input_mutation(radial, count):
    value, expected = views(frames=3, radial=radial, count=count)
    before = deepcopy(value)
    result = triangulate_views(value)
    assert result['joint_status_counts'] == {'estimated': 99}
    np.testing.assert_allclose([[p['world_point_m'] for p in row['joints']] for row in result['frames']], expected, atol=1e-10)
    for row in result['frames']:
        for point in row['joints']:
            assert point['contributing_view_indices'] == list(range(count))
            assert max(point['reprojection_px']) < 1e-8
            assert point['view_count'] == count
            assert point['optimization_status'] == 'converged'
    assert value == before
    assert not result['monocular_world_landmarks_used']
    assert result['physical_validation'] == 'unverified'


def test_occluded_primary_joint_uses_other_views_and_preserves_missing_slots():
    value, expected = views(frames=4)
    value[0]['evidence']['frames'][1]['landmarks'][25]['visibility'] = .1
    value[2]['evidence']['frames'][2].update(pose_count=2, status='multiple_people', landmarks=None)
    for view in value:
        view['evidence']['frames'].pop(3)
    result = triangulate_views(value)
    rescued = result['frames'][1]['joints'][25]
    assert rescued['status'] == 'estimated' and rescued['contributing_view_indices'] == [1, 2]
    assert rescued['observation_statuses'] == ['low_confidence', 'available', 'available']
    np.testing.assert_allclose(rescued['world_point_m'], expected[1, 25], atol=1e-10)
    assert result['frames'][2]['joints'][25]['contributing_view_indices'] == [0, 1]
    assert result['frames'][3]['source_evidence_ids'] == [None, None, None]
    assert all(p['status'] == 'image_evidence_unavailable' for p in result['frames'][3]['joints'])
    assert len(result['frames']) == 4 and result['joint_slots'] == 132


def test_available_conflicting_view_is_not_discarded_to_get_a_good_pair():
    value, _ = views()
    expected = triangulate_pair(*value[:2])
    value[2]['evidence']['frames'][0]['landmarks'][25]['y'] += .1
    result = triangulate_views(value)
    point = result['frames'][0]['joints'][25]
    assert expected['frames'][0]['joints'][25]['status'] == 'estimated'
    assert point['status'] in ('inconsistent_image_rays', 'optimization_incomplete')
    assert point['world_point_m'] is None and point['contributing_view_indices'] == [0, 1, 2]
    assert len(point['reprojection_px']) == 3


def test_only_one_visible_view_does_not_invent_depth():
    value, _ = views()
    for view in value[:2]:
        view['evidence']['frames'][0]['landmarks'][25]['presence'] = .1
    point = triangulate_views(value)['frames'][0]['joints'][25]
    assert point['status'] == 'image_evidence_unavailable'
    assert point['contributing_view_indices'] == [2]
    assert point['world_point_m'] is None


def test_world_rotation_translation_and_view_order_do_not_change_geometry():
    value, expected = views()
    rotation = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
    shift = np.array([4., -3., 2.])
    for view in value:
        for row in view['camera']['frames']:
            transformed = np.asarray(row['R']) @ rotation.T
            row['R'] = transformed.tolist(); row['t'] = (np.asarray(row['t'])-transformed @ shift).tolist()
    result = triangulate_views(value[::-1])
    np.testing.assert_allclose([p['world_point_m'] for p in result['frames'][0]['joints']], expected[0] @ rotation.T+shift, atol=1e-10)


def test_parallel_views_and_corrupt_monocular_depth_do_not_supply_geometry():
    value, _ = views()
    for view in value[1:]:
        view['camera']['frames'] = deepcopy(value[0]['camera']['frames'])
        view['evidence']['frames'][0]['landmarks'] = deepcopy(value[0]['evidence']['frames'][0]['landmarks'])
    for view in value:
        view['evidence']['frames'][0]['world_landmarks'] = 'ignored untrusted depth'
    result = triangulate_views(value)
    assert result['joint_status_counts'] == {'insufficient_ray_angle': 33}


def test_decimal_camera_rotations_are_preserved_and_solver_adjustment_is_explicit():
    value, expected = views()
    value[2]['camera']['frames'][0]['R'][0][0] += 4e-7
    before = deepcopy(value)
    result = triangulate_views(value)
    adjustment = result['frames'][0]['solver_rotations'][2]
    assert 0 < adjustment['maximum_element_change'] < 1e-5
    rotation = np.array(adjustment['R'])
    np.testing.assert_allclose(rotation @ rotation.T, np.eye(3), atol=1e-12)
    assert result['joint_status_counts'] == {'estimated': 33}
    np.testing.assert_allclose([p['world_point_m'] for p in result['frames'][0]['joints']], expected[0], atol=1e-6)
    assert value == before


@pytest.mark.parametrize('damage', ['view_count', 'duplicate_name', 'duplicate_source', 'camera_clock', 'pose_clock', 'identity', 'missing_camera_frame'])
def test_unbound_or_incompatible_inputs_reject(damage):
    value, _ = views(frames=3)
    if damage == 'view_count': value *= 2
    elif damage == 'duplicate_name': value[2]['name'] = value[0]['name']
    elif damage == 'duplicate_source':
        value[2]['camera']['source_sha256'] = value[2]['evidence']['source_sha256'] = value[0]['camera']['source_sha256']
        for row in value[2]['evidence']['frames']: row['id'] = f"{value[0]['camera']['source_sha256']}:frame:{row['frame']}"
    elif damage == 'camera_clock': value[2]['camera']['fps'] = 50.
    elif damage == 'pose_clock': value[2]['evidence']['frames'][1]['time_s'] += .01
    elif damage == 'identity': value[2]['evidence']['frames'][1]['id'] = 'wrong'
    elif damage == 'missing_camera_frame': value[2]['camera']['frames'].pop()
    with pytest.raises(ValueError):
        triangulate_views(value)
