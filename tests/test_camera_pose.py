from copy import deepcopy
import json
import math

import numpy as np
import pytest

from weft.camera_pose import FIT_JOINTS, build_camera_reference


SOURCE = 'a'*64


def fixture(*, rotations=None, translations=None, roots=None, radial=0.):
    relative = np.zeros((33, 3))
    for a, b, x, y, z in [(11, 12, .18, -.5, .02), (13, 14, .3, -.3, -.12),
                          (15, 16, .45, -.1, -.2), (23, 24, .12, 0, 0),
                          (25, 26, .14, .45, .03), (27, 28, .15, .9, .05)]:
        relative[a] = [-x, y, z]; relative[b] = [x, y, z]
    roots = np.array([[.2, .3, 5.]]) if roots is None else np.asarray(roots, dtype=float)
    count = len(roots)
    rotations = [np.eye(3)]*count if rotations is None else rotations
    translations = [np.zeros(3)]*count if translations is None else translations
    matrix = np.array([[700., 5., 640.], [0., 710., 480.], [0., 0., 1.]])
    camera = {'schema_version': 1, 'source_sha256': SOURCE, 'width': 1280, 'height': 960,
              'fps': 30., 'frame_count': count, 'world_axes': 'right_handed_z_up', 'units': 'm',
              'provenance': 'Analytic test cameras, not physical evidence', 'quality': 'authored',
              'K': matrix.tolist(), 'radial_k1': radial, 'frames': []}
    evidence = {'schema_version': 1, 'profile': 'exercise-pose-v1', 'source_sha256': SOURCE,
                'width': 1280, 'height': 960, 'fps': 30., 'frame_count': count, 'frames': []}
    for index, (rotation, translation, root) in enumerate(zip(rotations, translations, roots)):
        rotation, translation = np.asarray(rotation), np.asarray(translation)
        camera['frames'].append({'frame': index, 'time_s': index/30, 'R': rotation.tolist(), 't': translation.tolist()})
        absolute = (relative+root) @ rotation.T+translation
        normalized = absolute[:, :2]/absolute[:, 2:]
        normalized *= 1+radial*np.square(normalized).sum(axis=1)[:, None]
        pixels = np.column_stack((normalized, np.ones(33))) @ matrix.T
        offsets = relative @ rotation.T
        evidence['frames'].append({'frame': index, 'time_s': index/30, 'decoder_timestamp_s': index/30,
            'id': f'{SOURCE}:frame:{index}', 'pose_count': 1, 'status': 'tracked',
            'landmarks': [{'x': float(p[0]/1280), 'y': float(p[1]/960), 'visibility': 1., 'presence': 1.} for p in pixels],
            'world_landmarks': [dict(zip(('x', 'y', 'z'), p.tolist())) for p in offsets]})
    return evidence, camera, relative, roots


@pytest.mark.parametrize('radial', [0., .15, -.15])
def test_known_perspective_and_radial_geometry(radial):
    evidence, camera, relative, roots = fixture(radial=radial)
    before = deepcopy((evidence, camera))
    report = build_camera_reference(evidence, camera)
    row = report['frames'][0]
    assert row['status'] == 'estimated'
    assert np.allclose(row['root_world_m'], roots[0], atol=1e-10)
    assert np.allclose(row['world_relative_landmarks_m'], relative, atol=1e-10)
    assert row['reprojection_rmse_px'] < 1e-9
    assert (evidence, camera) == before
    json.dumps(report, allow_nan=False)


def test_camera_motion_does_not_become_body_motion():
    angles = [0., .12, -.08]
    rotations = [np.array([[math.cos(a), 0, math.sin(a)], [0, 1, 0], [-math.sin(a), 0, math.cos(a)]]) for a in angles]
    roots = [[.2, .3, 5.]]*3
    evidence, camera, relative, _ = fixture(rotations=rotations, translations=[[0, 0, 0], [.3, -.1, .2], [-.2, .1, -.3]], roots=roots)
    assert evidence['frames'][0]['landmarks'][23]['x'] != evidence['frames'][1]['landmarks'][23]['x']
    for row in build_camera_reference(evidence, camera)['frames']:
        assert row['status'] == 'estimated'
        assert np.allclose(row['root_world_m'], roots[0], atol=1e-10)
        assert np.allclose(row['world_relative_landmarks_m'], relative, atol=1e-10)


def test_true_body_motion_is_retained_in_three_axes():
    evidence, camera, _, roots = fixture(roots=[[.2, .3, 5.], [.4, .2, 5.2], [.7, .5, 4.8]])
    predicted = np.array([row['root_world_m'] for row in build_camera_reference(evidence, camera)['frames']])
    assert np.allclose(predicted, roots, atol=1e-10)


def test_pose_scale_error_is_not_observable_from_reprojection():
    evidence, camera, _, roots = fixture()
    for point in evidence['frames'][0]['world_landmarks']:
        for key in point: point[key] *= 2
    row = build_camera_reference(evidence, camera)['frames'][0]
    assert row['status'] == 'estimated'
    assert row['reprojection_rmse_px'] < 1e-9
    assert np.allclose(row['root_world_m'], roots[0]*2, atol=1e-10)
    assert not np.allclose(row['root_world_m'], roots[0])


def test_missing_competing_and_unsampled_frames_are_retained():
    evidence, camera, _, _ = fixture(roots=[[.2, .3, 5.]]*4)
    evidence['frames'][1].update(status='no_pose', pose_count=0, landmarks=None, world_landmarks=None)
    evidence['frames'][2].update(status='multiple_people', pose_count=2)
    evidence['frames'].pop()
    report = build_camera_reference(evidence, camera)
    assert [row['status'] for row in report['frames']] == ['estimated', 'pose_unavailable', 'competing_poses', 'pose_not_sampled']
    assert [row['frame'] for row in report['frames']] == list(range(4))
    assert all(row['root_world_m'] is None for row in report['frames'][1:])


@pytest.mark.parametrize('kind,status', [('hips', 'pelvis_unavailable'), ('limbs', 'insufficient_joints')])
def test_low_confidence_does_not_invent_translation(kind, status):
    evidence, camera, _, _ = fixture()
    indices = (23,) if kind == 'hips' else (11, 12, 13, 14, 15, 16, 25, 26)
    for index in indices: evidence['frames'][0]['landmarks'][index]['presence'] = .49
    assert build_camera_reference(evidence, camera)['frames'][0]['status'] == status


@pytest.mark.parametrize('spread,status', [(0., 'degenerate_projection'), (1e-6, 'ill_conditioned_projection')])
def test_insufficient_image_spread_rejects_depth(spread, status):
    evidence, camera, _, _ = fixture()
    for index in FIT_JOINTS:
        evidence['frames'][0]['landmarks'][index].update(x=.5+index*spread, y=.5)
    row = build_camera_reference(evidence, camera)['frames'][0]
    assert row['status'] == status
    assert row['root_world_m'] is None


def test_inconsistent_landmark_is_not_silently_dropped():
    evidence, camera, _, _ = fixture()
    landmarks = evidence['frames'][0]['landmarks']
    landmarks[15]['x'], landmarks[16]['x'] = landmarks[16]['x'], landmarks[15]['x']
    row = build_camera_reference(evidence, camera)['frames'][0]
    assert row['status'] == 'inconsistent_projection'
    assert {15, 16} <= set(row['fitted_joint_indices'])
    assert row['root_world_m'] is None


def test_behind_camera_depth_rejects():
    evidence, camera, _, _ = fixture(roots=[[.2, .3, -5.]])
    assert build_camera_reference(evidence, camera)['frames'][0]['status'] == 'depth_outside_envelope'


@pytest.mark.parametrize('mutation', ['reflection', 'scale', 'missing_camera', 'clock', 'wrong_source', 'size', 'intrinsics', 'extra_reference'])
def test_invalid_camera_contract_rejects(mutation):
    evidence, camera, _, _ = fixture()
    if mutation == 'reflection': camera['frames'][0]['R'][0][0] = -1
    elif mutation == 'scale': camera['frames'][0]['R'][0][0] = 1.1
    elif mutation == 'missing_camera': camera['frames'] = []
    elif mutation == 'clock': camera['frames'][0]['time_s'] = .01
    elif mutation == 'wrong_source': camera['source_sha256'] = 'b'*64
    elif mutation == 'size': camera['width'] = 640
    elif mutation == 'intrinsics': camera['K'][0][0] = 0
    elif mutation == 'extra_reference': camera['reference_joints'] = [[0, 0, 0]]
    with pytest.raises(ValueError): build_camera_reference(evidence, camera)


@pytest.mark.parametrize('field,value', [('decoder_timestamp_s', .01), ('id', 'wrong'), ('frame', True)])
def test_pose_frame_binding_is_exact(field, value):
    evidence, camera, _, _ = fixture()
    evidence['frames'][0][field] = value
    with pytest.raises(ValueError): build_camera_reference(evidence, camera)


def test_invalid_pose_count_and_world_values_are_unavailable():
    evidence, camera, _, _ = fixture(roots=[[.2, .3, 5.]]*2)
    evidence['frames'][0]['pose_count'] = True
    evidence['frames'][1]['world_landmarks'][0]['x'] = float('nan')
    report = build_camera_reference(evidence, camera)
    assert [row['status'] for row in report['frames']] == ['pose_unavailable', 'invalid_world_landmark']
    json.dumps(report, allow_nan=False)


def test_radial_inverse_ambiguity_rejects():
    evidence, camera, _, _ = fixture()
    camera['radial_k1'] = -1.
    for index in FIT_JOINTS: evidence['frames'][0]['landmarks'][index].update(x=.9, y=.9)
    assert build_camera_reference(evidence, camera)['frames'][0]['status'] == 'distortion_not_invertible'
