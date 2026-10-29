from copy import deepcopy
import json
import math

import numpy as np
import pytest

from weft.body_frame import CAMERA_TO_SIM
from weft.foot_reference import build_foot_reference, LANDMARKS


def fixture():
    source = 'a' * 64
    spec = {'profile': 'exercise-v3', 'source_sha256': source, 'unchanged_model': {'floor': 0}}
    metadata = {'source_sha256': source, 'width': 500, 'height': 500, 'fps': 10., 'decoded_frames': 6}
    frames = []
    for i in range(6):
        image = [dict(x=.5, y=.5, z=0., visibility=.9, presence=.95) for _ in range(33)]
        world = [dict(x=0., y=0., z=0., visibility=.9, presence=.95) for _ in range(33)]
        axis, radial = np.array([math.cos(i * .1), math.sin(i * .1), 0.]), np.array([0., 0., 1.])
        for side, sign in (('left', -1), ('right', 1)):
            ankle = np.array([sign * .15, 0., .1])
            heel = ankle - .04 * axis - .08 * radial
            points = {'ankle': ankle, 'heel': heel, 'foot_index': heel + .2 * axis}
            for part, index in LANDMARKS[side].items():
                camera = CAMERA_TO_SIM.T @ points[part]
                world[index].update(dict(zip(('x', 'y', 'z'), map(float, camera))))
                image[index].update(x=float(.5 + points[part][0] / 2), y=float(.5 - points[part][2] / 2))
        frames.append({'id': source + ':frame:' + str(i), 'frame': i, 'time_s': i / 10,
                       'status': 'tracked_partial', 'pose_count': 1,
                       'identity_status': 'single_model_hypothesis_not_verified_person_count',
                       'landmarks': image, 'world_landmarks': world})
    return spec, {'source_sha256': source, 'frames': frames}, metadata


def test_fixed_triangle_exact_under_rotation_without_mutating_model_or_raw_evidence():
    args = fixture()
    before = deepcopy(args)
    result = build_foot_reference(*args)
    assert args == before
    assert result['coverage']['feet']['left'] == {'available': 6, 'unavailable': 0}
    assert result['model_modified'] is False
    assert result['contract']['collision_geometry'] is False
    for row in result['frames']:
        for side in LANDMARKS:
            foot = row['feet'][side]
            geometry = foot['endpoint_geometry']
            heel, toe = np.array(geometry['heel_relative']), np.array(geometry['foot_index_relative'])
            assert np.linalg.norm(toe - heel) == pytest.approx(.2)
            assert np.linalg.norm(heel) == pytest.approx(math.hypot(.04, .08))
            assert np.linalg.norm(toe) == pytest.approx(math.hypot(.16, .08))
            assert foot['reference']['anatomical_axial_roll_rad'] is None
            assert foot['reference']['sole_normal'] is None
            assert foot['raw']['heel']['landmarks'] == args[1]['frames'][row['frame']]['landmarks'][LANDMARKS[side]['heel']]
    json.dumps(result, allow_nan=False)


def test_later_frames_never_change_prefix_shape_parameters():
    args = fixture()
    baseline = build_foot_reference(*args)
    for frame in args[1]['frames'][4:]:
        for side in LANDMARKS:
            frame['world_landmarks'][LANDMARKS[side]['foot_index']]['x'] += .1
    changed = build_foot_reference(*args)
    assert changed['fixed_shapes'] == baseline['fixed_shapes']
    assert set(changed['fixed_shapes']['left']['evidence_ids']) == {'a' * 64 + ':frame:' + str(i) for i in range(4)}


@pytest.mark.parametrize('mutation,reason', [
    (lambda f: f['landmarks'][29].update(x=1.001), 'heel:outside_image'),
    (lambda f: f['landmarks'][29].update(visibility=.49), 'heel:low_confidence'),
    (lambda f: f['world_landmarks'][29].update(presence=.49), 'heel:low_confidence'),
    (lambda f: f.update(pose_count=2), 'unsupported_pose_status_or_multiple_hypotheses'),
    (lambda f: f.update(pose_count=True), 'unsupported_pose_status_or_multiple_hypotheses'),
    (lambda f: f.update(world_landmarks=[]), 'heel:world_landmarks:missing_or_invalid'),
    (lambda f: f['landmarks'][29].update(x=float('nan')), 'heel:landmarks:missing_or_invalid')])
def test_unavailable_rows_preserve_raw_reasons_without_interpolation(mutation, reason):
    args = fixture()
    mutation(args[1]['frames'][4])
    result = build_foot_reference(*args)
    foot = result['frames'][4]['feet']['left']
    assert foot['status'] == 'unavailable' and reason in foot['reasons']
    assert foot['endpoint_geometry'] is None and foot['reference'] is None
    assert result['frames'][5]['feet']['left']['status'] == 'kinematic_reference'
    assert len(result['frames']) == 6
    json.dumps(result, allow_nan=False)


def test_insufficient_prefix_never_uses_later_frames_to_fill_shape():
    args = fixture()
    for frame in args[1]['frames'][:2]:
        frame['landmarks'][29]['x'] = 1.01
    result = build_foot_reference(*args)
    assert result['fixed_shapes']['left']['eligible_prefix_samples'] == 2
    assert result['fixed_shapes']['left']['status'] == 'unavailable'
    assert result['coverage']['feet']['left']['available'] == 0
    assert result['coverage']['feet']['right']['available'] == 6


def test_degenerate_landmark_plane_or_image_axis_does_not_invent_roll():
    args = fixture()
    for index in (27, 29, 31):
        args[1]['frames'][4]['world_landmarks'][index]['y'] = 0.
    args[1]['frames'][5]['landmarks'][31].update(args[1]['frames'][5]['landmarks'][29])
    result = build_foot_reference(*args)
    assert 'ankle_axis_plane_degenerate_or_out_of_bounds' in result['frames'][4]['feet']['left']['reasons']
    assert 'projected_axis_degenerate' in result['frames'][5]['feet']['left']['reasons']
    assert result['frames'][4]['feet']['left']['reference'] is None


def test_missing_source_frames_stay_unsampled_not_negative_or_interpolated():
    args = fixture()
    args[1]['frames'].pop(4)
    result = build_foot_reference(*args)
    assert result['coverage']['unsampled_source_frames'] == [4]
    assert [r['frame'] for r in result['frames']] == [0, 1, 2, 3, 5]


@pytest.mark.parametrize('mutation', [lambda s,e,m: m.update(source_sha256='b' * 64),
    lambda s,e,m: e['frames'][0].update(time_s=.01), lambda s,e,m: e['frames'][0].update(frame=1),
    lambda s,e,m: e.update(frames=e['frames'] * 151), lambda s,e,m: m.update(decoded_frames=1801),
    lambda s,e,m: m.update(fps=float('nan')), lambda s,e,m: m.update(width=True),
    lambda s,e,m: e['frames'][0].update(id='a' * 64 + ':frame:9')])
def test_identity_timing_and_bounded_work_enforced(mutation):
    args = fixture()
    mutation(*args)
    with pytest.raises(ValueError):
        build_foot_reference(*args)
