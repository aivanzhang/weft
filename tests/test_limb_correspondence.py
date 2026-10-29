"""Known two-camera motions distinguish short label islands from crossing and gaps."""

from copy import deepcopy
import json

import numpy as np
import pytest

from test_multiview_pose import pair
from weft.camera_reference import project_world
from weft.limb_correspondence import triangulate_with_limb_correspondence
from weft.multiview_pose import triangulate_pair


def sequence(count=13, baseline=1.):
    return pair(roots=[[0.2+i*.003, 0.3+i*.001, 5.+i*.002] for i in range(count)],
                translations=[[-baseline, 0., 0.]]*count)


def swap(view, frames, indices=(25, 26)):
    for frame in frames:
        points = view['evidence']['frames'][frame]['landmarks']
        a, b = indices
        points[a], points[b] = points[b], points[a]


@pytest.mark.parametrize('views', [(0,), (1,), (0, 1)])
@pytest.mark.parametrize('frames', [(6,), (5, 6, 7)])
def test_short_knee_island_recovers_known_points_without_reference_input(views, frames):
    first, second, expected = sequence()
    for index in views:
        swap((first, second)[index], frames)
    before = deepcopy((first, second))
    raw = triangulate_pair(first, second)
    result, audit = triangulate_with_limb_correspondence(first, second)
    assert {(row['frame'], row['pair']) for row in audit['changed_pairs']} == {(frame, 'knees') for frame in frames}
    assert audit['anatomical_side_verified'] is False
    assert audit['third_view_or_reference_used'] is False
    assert audit['missing_observations_filled'] is False
    for frame in range(13):
        for index, point in enumerate(result['frames'][frame]['joints']):
            if index in (25, 26) and frame in frames:
                assert np.allclose(point['world_point_m'], expected[frame, index], atol=1e-9)
                assert point['source_joint_indices'] == [index ^ 3 if v in views else index for v in (0, 1)]
            else:
                assert point == raw['frames'][frame]['joints'][index]
    assert (first, second) == before
    json.dumps(result, allow_nan=False)
    json.dumps(audit, allow_nan=False)


@pytest.mark.parametrize('indices,pair_name', [((13, 14), 'elbows'), ((15, 16), 'wrists'), ((27, 28), 'ankles')])
def test_other_bilateral_endpoints_follow_their_own_parent(indices, pair_name):
    first, second, expected = sequence(baseline=2.)
    swap(second, [6], indices)
    result, audit = triangulate_with_limb_correspondence(first, second)
    assert [(row['frame'], row['pair']) for row in audit['changed_pairs']] == [(6, pair_name)]
    assert np.allclose([result['frames'][6]['joints'][j]['world_point_m'] for j in indices], expected[6, list(indices)], atol=1e-9)


def test_unsupported_raw_ray_assignment_cannot_gain_a_fabricated_paired_score():
    first, second, _ = sequence()
    swap(second, [6], (15, 16))
    raw = triangulate_pair(first, second)
    assert any(raw['frames'][6]['joints'][j]['status'] != 'estimated' for j in (15, 16))
    result, audit = triangulate_with_limb_correspondence(first, second)
    assert not audit['changed_pairs']
    assert result['frames'] == raw['frames']


def test_ordinary_motion_and_consistent_global_side_assignment_are_not_rewritten():
    for all_swapped in (False, True):
        first, second, _ = sequence()
        if all_swapped:
            for view in (first, second):
                swap(view, range(13))
        raw = triangulate_pair(first, second)
        result, audit = triangulate_with_limb_correspondence(first, second)
        assert not audit['changed_pairs']
        assert result['frames'] == raw['frames']


def test_real_continuous_leg_crossing_is_preserved():
    first, second, expected = sequence()
    for frame in range(13):
        for index, sign in ((25, -1), (26, 1)):
            expected[frame, index, 0] = .2+frame*.003+sign*(.14-frame*.023)
        for view in (first, second):
            camera = view['camera']
            row = camera['frames'][frame]
            _, pixels, valid = project_world(expected[frame], row['R'], row['t'], camera['K'], camera['radial_k1'])
            assert valid.all()
            for index in (25, 26):
                view['evidence']['frames'][frame]['landmarks'][index].update(
                    x=float(pixels[index, 0]/camera['width']), y=float(pixels[index, 1]/camera['height']))
    raw = triangulate_pair(first, second)
    result, audit = triangulate_with_limb_correspondence(first, second)
    assert not audit['changed_pairs']
    assert result['frames'] == raw['frames']


@pytest.mark.parametrize('problem', ['child_confidence', 'parent_confidence', 'missing_anchor', 'competing',
                                    'long_gap', 'boundary', 'long_island', 'unavailable_parent'])
def test_insufficient_support_abstains_without_dropping_slots(problem):
    first, second, _ = sequence()
    swap(second, [0] if problem == 'boundary' else [4, 5, 6, 7] if problem == 'long_island' else [6])
    if problem == 'child_confidence': second['evidence']['frames'][6]['landmarks'][25]['presence'] = .79
    elif problem == 'parent_confidence': first['evidence']['frames'][6]['landmarks'][23]['presence'] = .79
    elif problem == 'missing_anchor': first['evidence']['frames'][5].update(pose_count=0, landmarks=None)
    elif problem == 'competing': second['evidence']['frames'][6]['pose_count'] = 2
    elif problem == 'long_gap':
        for view in (first, second):
            view['camera']['fps'] = view['evidence']['fps'] = 5.
            for row in view['camera']['frames']: row['time_s'] = row['frame']/5
            for row in view['evidence']['frames']:
                row['time_s'] = row['decoder_timestamp_s'] = row['frame']/5
    elif problem == 'unavailable_parent': first['evidence']['frames'][6]['landmarks'][23]['presence'] = .1
    raw = triangulate_pair(first, second)
    result, audit = triangulate_with_limb_correspondence(first, second)
    assert not audit['changed_pairs']
    assert result['frames'] == raw['frames']
    assert result['source_frames'] == 13 and result['joint_slots'] == 429
    assert len(audit['frames']) == 13


def test_unsampled_frames_stay_explicit_and_source_clocks_stay_exact():
    first, second, _ = sequence(count=26)
    swap(second, [12])
    for view in (first, second): view['evidence']['frames'] = view['evidence']['frames'][::2]
    result, audit = triangulate_with_limb_correspondence(first, second)
    assert len(result['frames']) == len(audit['frames']) == 26
    assert [(r['frame'], r['pair']) for r in audit['changed_pairs']] == [(12, 'knees')]
    assert all(point['world_point_m'] is None for row in result['frames'][1::2] for point in row['joints'])
    assert all(row['time_s'] == row['frame']/30 for row in result['frames'])


def test_monocular_depth_and_derived_body_fields_are_never_used():
    first, second, _ = sequence()
    swap(second, [6])
    expected, audit = triangulate_with_limb_correspondence(first, second)
    for view in (first, second):
        for row in view['evidence']['frames']:
            row['world_landmarks'] = [{'x': 9000, 'y': -9000, 'z': 3000}]*33
            row['joints'] = {'nonsense': 9000}
    actual, other = triangulate_with_limb_correspondence(first, second)
    assert actual['frames'] == expected['frames']
    assert other['changed_pairs'] == audit['changed_pairs']


def test_previously_repaired_sources_cannot_certify_their_own_anchors():
    first, second, _ = sequence()
    first['evidence']['frames'][0]['identity_raw'] = {}
    with pytest.raises(ValueError, match='original image labels'):
        triangulate_with_limb_correspondence(first, second)


def test_optional_raw_export_preserves_the_exact_uncorrected_baseline():
    first, second, _ = sequence()
    swap(second, [6])
    expected = triangulate_pair(first, second)
    result, audit, raw = triangulate_with_limb_correspondence(first, second, return_baseline=True)
    assert raw == expected
    assert result['frames'] != raw['frames'] and audit['changed_pairs']
    with pytest.raises(ValueError, match='boolean'):
        triangulate_with_limb_correspondence(first, second, return_baseline=1)
