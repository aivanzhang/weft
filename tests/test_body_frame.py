from copy import deepcopy
import math

import numpy as np
import pytest

from weft.body_frame import CAMERA_TO_SIM, fit_body_frame, frame_rotation, validate_body_frame
from weft.human import HumanEnv, build_human
from test_human import human_evidence


def landmarks(angle):
    c, s = math.cos(angle), math.sin(angle)
    rotation = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    points = np.zeros((33, 3))
    points[11], points[12] = [-.19, -.45, 0], [.19, -.45, 0]
    points[23], points[24] = [-.14, 0, 0], [.14, 0, 0]
    return [dict(zip(('x', 'y', 'z'), point)) for point in points @ rotation.T]


def evidence_with_frame(evidence, angle):
    result = deepcopy(evidence)
    result['frames'] = [{'id': f'source:frame:{i}', 'status': 'tracked', 'time_s': i/2,
                         'world_landmarks': landmarks(angle)} for i in range(3)]
    return result


def test_camera_frame_preserves_known_upright_and_horizontal_axes():
    assert np.allclose(frame_rotation(landmarks(0)), np.eye(3))
    rotation = frame_rotation(landmarks(math.pi/2))
    assert np.allclose(rotation[:, 2], [1, 0, 0])
    assert np.linalg.det(rotation) == pytest.approx(1)
    assert np.allclose(CAMERA_TO_SIM.T @ rotation[:, 2], [1, 0, 0])


def test_no_evidence_is_an_assumption_not_an_inferred_pose():
    result = fit_body_frame({})
    assert result['status'] == 'assumed' and not result['evidence_ids']
    validate_body_frame(result)


def test_reflections_scaling_and_dangling_evidence_rejected(human_evidence):
    frame = fit_body_frame(evidence_with_frame(human_evidence, .3))
    for matrix in (np.diag([-1, 1, 1]), np.diag([2, 1, 1]), np.full((3, 3), np.nan)):
        candidate = deepcopy(frame)
        candidate['rotation'] = matrix.tolist()
        with pytest.raises(ValueError, match='rotation'):
            validate_body_frame(candidate)
    frame['reference'][0]['id'] = 'not-supplied'
    with pytest.raises(ValueError, match='dangling'):
        validate_body_frame(frame)


def test_horizontal_anchor_changes_physics_not_just_rendering(human_evidence, tmp_path):
    human_evidence['reference'][0]['joints']['left_knee'] = .8
    upright = build_human(evidence_with_frame(human_evidence, 0), tmp_path/'upright')
    horizontal = build_human(evidence_with_frame(human_evidence, math.pi/2), tmp_path/'horizontal')
    environments = [HumanEnv(spec) for spec in (upright, horizontal)]
    try:
        snapshots = []
        for env in environments:
            env.reset(seed=0, options={'jitter': False})
            start = env.snapshot()
            pelvis = np.asarray(start['human']['joints']['pelvis'])
            neck = np.asarray(start['human']['joints']['neck'])
            snapshots.append((neck - pelvis) / np.linalg.norm(neck - pelvis))
            for _ in range(20):
                env.advance_physics(0)
            assert env.snapshot()['human']['joints']['pelvis'] == start['human']['joints']['pelvis']
        assert np.allclose(snapshots[0], [0, 0, 1])
        assert np.allclose(snapshots[1], [1, 0, 0])
        # The same motor-free joint responds differently under rotated gravity loading.
        angles = [env.snapshot()['human']['angles']['left_knee'] for env in environments]
        assert abs(angles[0] - angles[1]) > .01
    finally:
        for env in environments:
            env.close()


def test_old_saved_spec_retains_legacy_upright_geometry(human_evidence, tmp_path):
    spec = build_human(human_evidence, tmp_path/'legacy')
    spec.pop('body_frame')
    env = HumanEnv(spec)
    try:
        env.reset(seed=0)
        joints = env.snapshot()['human']['joints']
        assert joints['neck'][0:2] == joints['pelvis'][0:2]
        assert joints['neck'][2] > joints['pelvis'][2]
    finally:
        env.close()
