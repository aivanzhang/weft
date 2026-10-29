from copy import deepcopy
import json
import math

import numpy as np
import pytest

from test_camera_pose import fixture
from weft.multiview_pose import triangulate_pair


def pair(*, roots=None, radial=0., rotations=None, translations=None):
    count = 1 if roots is None else len(roots)
    options = {'roots': roots, 'radial': radial}
    first, camera1, relative, roots = fixture(**options)
    second, camera2, _, _ = fixture(**options, rotations=rotations,
        translations=[[-1., 0., 0.]]*count if translations is None else translations)
    second['source_sha256'] = camera2['source_sha256'] = 'b'*64
    for row in second['frames']: row['id'] = f"{'b'*64}:frame:{row['frame']}"
    views = [{'name': name, 'evidence': evidence, 'camera': camera,
              'clock_provenance': 'Analytic simultaneous cameras, no recorded physical evidence'}
             for name, evidence, camera in [('left', first, camera1), ('right', second, camera2)]]
    return *views, relative[None, :, :]+roots[:, None, :]


@pytest.mark.parametrize('radial', [0., .15, -.15])
def test_known_metric_geometry_without_monocular_depth(radial):
    first, second, expected = pair(radial=radial)
    before = deepcopy((first, second))
    result = triangulate_pair(first, second)
    joints = result['frames'][0]['joints']
    assert result['joint_status_counts'] == {'estimated': 33}
    assert np.allclose([row['world_point_m'] for row in joints], expected[0], atol=1e-10)
    assert max(max(row['reprojection_px']) for row in joints) < 1e-9
    assert result['monocular_world_landmarks_used'] is False
    assert result['reference_body_or_contacts_used'] is False
    assert result['physical_validation'] == 'unverified'
    assert (first, second) == before
    json.dumps(result, allow_nan=False)


def test_moving_cameras_preserve_true_body_motion():
    rotations = [np.array([[math.cos(a), 0., math.sin(a)], [0., 1., 0.],
                          [-math.sin(a), 0., math.cos(a)]]) for a in (0., .12, -.08)]
    first, second, expected = pair(roots=[[.2, .3, 5.], [.4, .2, 5.2], [.7, .5, 4.8]],
        rotations=rotations, translations=[[-1., 0., 0.], [-1.3, .2, .1], [-.8, -.2, .3]])
    result = triangulate_pair(first, second)
    assert result['joint_status_counts'] == {'estimated': 99}
    assert np.allclose([[j['world_point_m'] for j in row['joints']] for row in result['frames']], expected, atol=1e-10)


def test_world_coordinate_change_does_not_change_image_fit():
    first, second, expected = pair()
    rotation = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
    shift = np.array([10., -3., 2.])
    for view in (first, second):
        row = view['camera']['frames'][0]
        new_rotation = np.asarray(row['R']) @ rotation.T
        row['R'] = new_rotation.tolist()
        row['t'] = (np.asarray(row['t'])-new_rotation @ shift).tolist()
    result = triangulate_pair(first, second)
    assert np.allclose([j['world_point_m'] for j in result['frames'][0]['joints']], expected[0] @ rotation.T+shift, atol=1e-10)


@pytest.mark.parametrize('change', ['remove', 'contradict'])
def test_monocular_depth_never_controls_triangulation(change):
    first, second, _ = pair()
    expected = triangulate_pair(first, second)['frames']
    for view in (first, second):
        if change == 'remove': del view['evidence']['frames'][0]['world_landmarks']
        else: view['evidence']['frames'][0]['world_landmarks'] = [{'x': 1000., 'y': -1000., 'z': 0.}]*33
    assert triangulate_pair(first, second)['frames'] == expected


@pytest.mark.parametrize('status', ['tracked_partial', 'low_confidence', 'invalid_geometry'])
def test_individual_image_confidence_is_not_a_monocular_body_gate(status):
    first, second, _ = pair()
    first['evidence']['frames'][0]['status'] = status
    first['evidence']['frames'][0]['landmarks'][15]['presence'] = .49
    joints = triangulate_pair(first, second)['frames'][0]['joints']
    assert joints[15]['world_point_m'] is None
    assert joints[15]['observation_statuses'] == ['low_confidence', 'available']
    assert all(row['status'] == 'estimated' for i, row in enumerate(joints) if i != 15)


def test_missing_competing_and_unsampled_slots_stay_missing():
    first, second, _ = pair(roots=[[.2, .3, 5.]]*4)
    samples = second['evidence']['frames']
    samples[1].update(status='no_pose', pose_count=0, landmarks=None)
    samples[2].update(status='multiple_people', pose_count=2)
    samples.pop()
    result = triangulate_pair(first, second)
    assert result['source_frames'] == 4 and result['joint_slots'] == 132
    assert result['joint_status_counts'] == {'estimated': 33, 'image_evidence_unavailable': 99}
    assert result['frames'][-1]['source_evidence_ids'][1] is None
    assert all(row['world_point_m'] is None for frame in result['frames'][1:] for row in frame['joints'])


@pytest.mark.parametrize('field,value,status', [('x', -.01, 'outside_image'), ('y', 1.01, 'outside_image'),
    ('x', None, 'invalid_landmark'), ('presence', True, 'invalid_landmark'),
    ('visibility', 1.1, 'low_confidence'), ('presence', .49, 'low_confidence')])
def test_invalid_or_hidden_joint_is_not_filled(field, value, status):
    first, second, _ = pair()
    first['evidence']['frames'][0]['landmarks'][11][field] = value
    joint = triangulate_pair(first, second)['frames'][0]['joints'][11]
    assert joint['status'] == 'image_evidence_unavailable'
    assert joint['observation_statuses'][0] == status
    assert joint['world_point_m'] is None


@pytest.mark.parametrize('geometry', ['same_center', 'tiny_baseline', 'antiparallel'])
def test_ambiguous_ray_geometry_rejects(geometry):
    options = {'translations': [[0., 0., 0.]] if geometry != 'tiny_baseline' else [[-.0001, 0., 0.]]}
    if geometry == 'antiparallel': options['rotations'] = [np.diag([-1., 1., -1.])]
    first, second, _ = pair(**options)
    result = triangulate_pair(first, second)
    assert result['joint_status_counts'] == {'insufficient_ray_angle': 33}
    assert all(row['world_point_m'] is None for row in result['frames'][0]['joints'])


@pytest.mark.parametrize('root,baseline,status', [([.2, .3, -5.], 1., 'nonforward_intersection'),
    ([0., 0., .1], .01, 'depth_outside_envelope'), ([.2, .3, 101.], 10., 'depth_outside_envelope')])
def test_depth_and_forward_guards(root, baseline, status):
    first, second, _ = pair(roots=[root], translations=[[-baseline, 0., 0.]])
    joint = triangulate_pair(first, second)['frames'][0]['joints'][0]
    assert joint['status'] == status
    assert joint['world_point_m'] is None


def test_disagreeing_image_rays_reject_without_joint_dropping():
    first, second, _ = pair()
    second['evidence']['frames'][0]['landmarks'][11]['y'] += .12
    result = triangulate_pair(first, second)
    joint = result['frames'][0]['joints'][11]
    assert joint['status'] == 'inconsistent_image_rays'
    assert joint['world_point_m'] is None and max(joint['reprojection_px']) > 12.8
    assert result['joint_slots'] == 33


def test_ambiguous_radial_inverse_rejects():
    first, second, _ = pair()
    first['camera']['radial_k1'] = -1.
    first['evidence']['frames'][0]['landmarks'][11].update(x=.99, y=.99)
    joint = triangulate_pair(first, second)['frames'][0]['joints'][11]
    assert joint['status'] == 'distortion_not_invertible' and joint['world_point_m'] is None


@pytest.mark.parametrize('mutation', ['source', 'clock', 'id', 'schema', 'reflection', 'name',
    'same_source', 'fps', 'count', 'clock_provenance', 'duplicate_sample', 'empty_samples', 'provider_reference'])
def test_binding_and_scope_contract_rejects(mutation):
    first, second, _ = pair()
    if mutation == 'source': second['camera']['source_sha256'] = 'c'*64
    elif mutation == 'clock': second['evidence']['frames'][0]['decoder_timestamp_s'] = .01
    elif mutation == 'id': second['evidence']['frames'][0]['id'] = 'wrong'
    elif mutation == 'schema': second['evidence']['schema_version'] = True
    elif mutation == 'reflection': second['camera']['frames'][0]['R'][0][0] = -1.
    elif mutation == 'name': second['name'] = first['name']
    elif mutation == 'same_source': second['camera'] = deepcopy(first['camera']); second['evidence'] = deepcopy(first['evidence'])
    elif mutation == 'fps': second['camera']['fps'] = second['evidence']['fps'] = 29.
    elif mutation == 'count':
        _, extra, _ = pair(roots=[[.2, .3, 5.]]*2)
        second['evidence'] = extra['evidence']; second['camera'] = extra['camera']
    elif mutation == 'clock_provenance': second['clock_provenance'] = ' '
    elif mutation == 'duplicate_sample': second['evidence']['frames'] *= 2
    elif mutation == 'empty_samples': second['evidence']['frames'] = []
    elif mutation == 'provider_reference': second['camera']['provider_joints'] = []
    with pytest.raises(ValueError): triangulate_pair(first, second)
