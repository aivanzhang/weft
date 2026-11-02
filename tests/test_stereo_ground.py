from copy import deepcopy

import numpy as np
import pytest

from test_multiview_pose import pair
from weft.camera_ground import validate_ground
from weft.camera_pose import validate_camera
from weft.io import canonical_hash
from weft.stereo_ground import estimate_stereo_ground


def project(camera, point, frame=0):
    pose = camera['frames'][frame]
    xyz = np.asarray(pose['R']) @ point+pose['t']
    xy = xyz[:2]/xyz[2]
    xy *= 1+camera['radial_k1']*(xy @ xy)
    return (np.asarray(camera['K']) @ np.r_[xy, 1.])[:2].tolist()


def fixture(*, radial=0., assessment=False):
    first, second, _ = pair(roots=[[0., 0., 5.]]*3, radial=radial,
        translations=[[-1., 0., 0.], [-1.1, .2, .1], [-.9, .1, 0.]])
    cameras = [first['camera'], second['camera']]
    points = np.array([[-1., -1., 5.1], [1., -1., 5.3], [0., 1., 4.8]])
    labels = {'schema_version': 1, 'profile': 'stereo-ground-annotations-v1',
        'world_provenance': 'Analytic shared metric world and static sloped plane; not recorded evidence',
        'views': [{'name': name, 'source_sha256': c['source_sha256'], 'calibration_sha256': canonical_hash(c),
                   'image_sha256': 'c'*64, 'frame': 0, 'time_s': 0., 'provenance': 'Analytic original-raster image association'}
                  for name, c in zip(('left', 'right'), cameras, strict=True)],
        'points': [{'id': f'p{i}', 'partition': 'construction', 'pixels': [project(c, p) for c in cameras],
                    'uncertainty_px': [2., 2.], 'provenance': 'Authored analytic ground point with assumed pixel uncertainty'}
                   for i, p in enumerate(points)]}
    if assessment:
        p = np.array([.2, .3, 4.96])
        labels['points'].append({'id': 'check', 'partition': 'assessment', 'pixels': [project(c, p) for c in cameras],
                                 'uncertainty_px': [2., 2.], 'provenance': 'Separate analytic point, never used to construct the plane'})
    return cameras, labels, points


@pytest.mark.parametrize('radial', [0., -.15, .15])
def test_conditional_plane_reframes_all_cameras_without_changing_pixels_or_inputs(radial):
    cameras, labels, expected = fixture(radial=radial)
    before = deepcopy((cameras, labels))
    result = estimate_stereo_ground(cameras, labels)
    assert result['status'] == 'conditional_plane'
    assert result['assessment'] == 'unavailable_no_assessment_points'
    assert result['physical_validation'] == 'unavailable'
    assert result['body_pose_contact_inputs_used'] is False
    np.testing.assert_allclose([p['world_point_m'] for p in result['points']], expected, atol=1e-10)
    rotation = np.asarray(result['world_transform']['R'])
    np.testing.assert_allclose(rotation @ rotation.T, np.eye(3), atol=1e-12)
    assert np.linalg.det(rotation) == pytest.approx(1.)
    transformed = expected @ rotation.T
    np.testing.assert_allclose(transformed[:, 2], result['world_transform']['ground_height_m'], atol=1e-12)
    for original, changed, ground in zip(cameras, result['cameras'], result['grounds'], strict=True):
        validate_camera(changed, changed)
        validate_ground(ground, {'source_sha256': changed['source_sha256'], 'calibration_sha256': canonical_hash(changed)})
        for index, row in enumerate(changed['frames']):
            assert row['t'] == original['frames'][index]['t']
            assert row['time_s'] == original['frames'][index]['time_s']
            for p in (*expected, np.array([.2, .5, 7.])):
                np.testing.assert_allclose(project(changed, rotation @ p, index), project(original, p, index), atol=1e-10)
    assert result['sensitivity']['complete']
    assert len(result['sensitivity']['scenarios']) == 24
    assert result['sensitivity']['maximum_center_distance_m'] > 0
    assert (cameras, labels) == before


def test_assessment_does_not_fit_the_plane_and_failed_check_emits_no_calibration():
    cameras, labels, _ = fixture(assessment=True)
    accepted = estimate_stereo_ground(cameras, labels)
    assert accepted['assessment'] == 'consistent_source_labels'
    labels['points'][-1]['pixels'] = [project(c, np.array([.2, .3, 5.3])) for c in cameras]
    rejected = estimate_stereo_ground(cameras, labels)
    assert rejected['plane'] == accepted['plane']
    assert rejected['status'] == 'ground_unavailable'
    assert rejected['assessment'] == 'failed_source_labels'
    assert rejected['world_transform'] is None
    assert rejected['cameras'] == rejected['grounds'] == []


def test_legal_camera_origin_cannot_emit_out_of_envelope_ground():
    cameras, labels, points = fixture()
    points[:, 2] = 1005
    for index, camera in enumerate(cameras):
        for frame in camera['frames']:
            frame['t'][2] -= 1000
        labels['views'][index]['calibration_sha256'] = canonical_hash(camera)
    for label, point in zip(labels['points'], points, strict=True):
        label['pixels'] = [project(camera, point) for camera in cameras]
    result = estimate_stereo_ground(cameras, labels)
    assert result['status'] == 'ground_unavailable'
    assert not result['checks']['ground_height_envelope']
    assert result['grounds'] == []


@pytest.mark.parametrize('failure', ['reprojection', 'collinear', 'small_extent', 'nonplanar', 'steep'])
def test_bad_construction_is_retained_without_dropping_or_refitting_points(failure):
    cameras, labels, _ = fixture()
    if failure == 'reprojection':
        labels['points'][0]['pixels'][1][1] += 30
    else:
        points = {'collinear': [[-1, 0, 5], [0, 0, 5], [1, 0, 5]],
                  'small_extent': [[-.01, -.01, 5], [.01, -.01, 5], [0, .01, 5]],
                  'steep': [[-1, 0, 4], [1, 0, 4], [0, 0, 6]],
                  'nonplanar': [[-1, -1, 5], [1, -1, 5], [0, 1, 5], [0, 0, 5.5]]}[failure]
        labels['points'] = [{'id': f'p{i}', 'partition': 'construction', 'pixels': [project(c, np.asarray(p)) for c in cameras],
                              'uncertainty_px': [2., 2.], 'provenance': 'Intentionally inconsistent analytic labels'} for i, p in enumerate(points)]
    result = estimate_stereo_ground(cameras, labels)
    assert result['status'] == 'ground_unavailable'
    assert len(result['points']) == len(labels['points'])
    assert result['cameras'] == result['grounds'] == []


@pytest.mark.parametrize('damage', ['source', 'camera', 'image', 'frame', 'clock', 'duplicate_view', 'duplicate_point',
    'partition', 'too_few_construction', 'too_many', 'pixel_bool', 'pixel_nan', 'pixel_outside', 'uncertainty_zero',
    'uncertainty_bool', 'uncertainty_large', 'missing_provenance', 'extra_field', 'camera_reflection', 'units'])
def test_unbound_or_invalid_annotations_are_rejected(damage):
    cameras, labels, _ = fixture()
    view, point = labels['views'][0], labels['points'][0]
    if damage == 'source': view['source_sha256'] = 'f'*64
    elif damage == 'camera': view['calibration_sha256'] = 'f'*64
    elif damage == 'image': view['image_sha256'] = 'not a hash'
    elif damage == 'frame': view['frame'] = True
    elif damage == 'clock': view['time_s'] = .5
    elif damage == 'duplicate_view': labels['views'][1]['name'] = view['name']
    elif damage == 'duplicate_point': labels['points'][1]['id'] = point['id']
    elif damage == 'partition': point['partition'] = 'maybe'
    elif damage == 'too_few_construction': point['partition'] = 'assessment'
    elif damage == 'too_many': labels['points'] *= 6
    elif damage == 'pixel_bool': point['pixels'][0][0] = True
    elif damage == 'pixel_nan': point['pixels'][0][0] = float('nan')
    elif damage == 'pixel_outside': point['pixels'][0][0] = 1.
    elif damage == 'uncertainty_zero': point['uncertainty_px'][0] = 0.
    elif damage == 'uncertainty_bool': point['uncertainty_px'][0] = True
    elif damage == 'uncertainty_large': point['uncertainty_px'][0] = 33.
    elif damage == 'missing_provenance': point['provenance'] = ' '
    elif damage == 'extra_field': labels['pose_score'] = 0.
    elif damage == 'camera_reflection': cameras[0]['frames'][0]['R'][0][0] = -1.
    elif damage == 'units': cameras[0]['units'] = 'relative'
    with pytest.raises(ValueError):
        estimate_stereo_ground(cameras, labels)
