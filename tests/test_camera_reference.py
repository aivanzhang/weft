from copy import deepcopy
import json

import numpy as np
import pytest

from weft.camera_reference import JOINTS, compare_frame, project_world


def fixture():
    camera = {name: np.array([(i%2)*.3-.15, (i//2)*.2-.5, 3.]) for i, name in enumerate(JOINTS)}
    root = (camera['left_hip']+camera['right_hip'])/2
    pixels = {name: point[:2]*100/point[2]+50 for name, point in camera.items()}
    prediction = {'status': 'tracked', 'pose_count': 1,
                  'landmarks': [{} for _ in range(33)], 'world_landmarks': [{} for _ in range(33)]}
    for name, index in JOINTS.items():
        prediction['landmarks'][index] = dict(zip(('x', 'y'), pixels[name]/100)) | {'presence': 1., 'visibility': 1.}
        prediction['world_landmarks'][index] = dict(zip(('x', 'y', 'z'), camera[name]-root))
    # JSON input values are native Python numbers, not library scalars.
    prediction = json.loads(json.dumps(prediction))
    return prediction, camera, pixels


def compare(prediction, camera, pixels):
    return compare_frame(prediction, camera, pixels, width=100, height=100)


def test_perspective_projection_and_halfplane():
    camera, uv, valid = project_world([[2, 4, 2], [0, 0, -1], [0, 0, 0]], np.eye(3), [0, 0, 0],
                                      [[100, 0, 50], [0, 200, 40], [0, 0, 1]])
    np.testing.assert_allclose(uv[0], [150, 440]); assert valid.tolist() == [True, False, False]
    assert np.isnan(uv[1:]).all()


def test_rotation_translation_and_radial_distortion_are_applied_once():
    rotation = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    camera, uv, valid = project_world([[1, 0, 2]], rotation, [1, 0, 0], np.diag([100, 100, 1]), .1)
    np.testing.assert_allclose(camera, [[1, 1, 2]])
    np.testing.assert_allclose(uv, [[52.5, 52.5]]); assert valid[0]


@pytest.mark.parametrize('rotation', [np.diag([-1, 1, 1]), np.eye(3)*2, np.full((3, 3), np.nan)])
def test_nonrigid_or_reflected_camera_rejected(rotation):
    with pytest.raises(ValueError, match='proper rotation'):
        project_world([[0, 0, 2]], rotation, [0, 0, 0], np.eye(3))


@pytest.mark.parametrize('intrinsics', [np.diag([-1, 1, 1]), np.diag([1, 1, 2]), np.ones((2, 3))])
def test_bad_intrinsics_rejected(intrinsics):
    with pytest.raises(ValueError):
        project_world([[0, 0, 2]], np.eye(3), [0, 0, 0], intrinsics)


def test_missing_reference_point_is_not_projected():
    _, uv, valid = project_world([[float('nan'), 1, 2]], np.eye(3), [0, 0, 0], np.eye(3))
    assert not valid[0] and np.isnan(uv).all()


def test_exact_pose_comparison_preserves_inputs_and_has_no_fitted_alignment():
    prediction, camera, pixels = fixture(); before = deepcopy(prediction)
    rows = compare(prediction, camera, pixels)
    assert len(rows) == 12 and prediction == before
    assert all(r['image_distance_px'] < 1e-12 and r['relative_3d_distance_mm'] < 1e-12 for r in rows.values())
    prediction['world_landmarks'][15]['z'] += .25
    rows = compare(prediction, camera, pixels)
    assert rows['left_wrist']['relative_3d_distance_mm'] == pytest.approx(250)
    assert rows['left_wrist']['image_distance_px'] < 1e-12


def test_root_translation_is_removed_but_scale_error_is_retained():
    prediction, camera, pixels = fixture()
    for name, index in JOINTS.items():
        for key in ('x', 'y', 'z'): prediction['world_landmarks'][index][key] += 10
    assert all(r['relative_3d_distance_mm'] < 1e-11 for r in compare(prediction, camera, pixels).values())
    for index in JOINTS.values():
        for key in ('x', 'y', 'z'): prediction['world_landmarks'][index][key] *= 2
    assert compare(prediction, camera, pixels)['left_wrist']['relative_3d_distance_mm'] > 100


@pytest.mark.parametrize('prediction,status', [(None, 'pose_not_sampled'), ({'pose_count': 0}, 'pose_unavailable'),
                                              ({'pose_count': 2}, 'competing_poses')])
def test_absent_or_competing_predictions_retain_every_joint(prediction, status):
    _, camera, pixels = fixture(); rows = compare(prediction, camera, pixels)
    assert len(rows) == 12 and all(r['image_status'] == r['relative_3d_status'] == status for r in rows.values())
    assert all(r['relative_3d_distance_mm'] is None for r in rows.values())
    json.dumps(rows, allow_nan=False)


def test_low_confidence_hips_disable_3d_not_other_image_comparisons():
    prediction, camera, pixels = fixture(); prediction['landmarks'][23]['visibility'] = .49
    rows = compare(prediction, camera, pixels)
    assert rows['left_hip']['image_status'] == 'low_confidence'
    assert rows['left_wrist']['image_status'] == 'compared'
    assert rows['left_wrist']['relative_3d_status'] == 'predicted_root_unavailable'


def test_missing_reference_hip_does_not_discard_image_metric():
    prediction, camera, pixels = fixture(); camera['left_hip'][:] = np.nan
    rows = compare(prediction, camera, pixels)
    assert rows['left_hip']['relative_3d_status'] == 'reference_unavailable'
    assert rows['left_wrist']['image_status'] == 'compared'
    assert rows['left_wrist']['relative_3d_status'] == 'reference_root_unavailable'
    json.dumps(rows, allow_nan=False)


def test_out_of_image_prediction_is_not_clipped_or_hidden():
    prediction, camera, pixels = fixture(); prediction['landmarks'][15]['x'] = 2
    row = compare(prediction, camera, pixels)['left_wrist']
    assert row['predicted_image_px'][0] == 200 and row['image_distance_px'] > 100


def test_invalid_landmark_and_missing_named_slot_are_explicit():
    prediction, camera, pixels = fixture(); prediction['world_landmarks'][15]['z'] = float('nan')
    assert compare(prediction, camera, pixels)['left_wrist']['image_status'] == 'invalid_landmark'
    del pixels['left_wrist']
    with pytest.raises(ValueError, match='twelve'):
        compare(prediction, camera, pixels)
