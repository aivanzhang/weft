"""Analytic mobile-body plumbing; these checks do not establish physical human accuracy."""

import json

import numpy as np
import pytest

from weft.benchmark import extract_metrics
from weft.human import HumanEnv, build_human, evaluate_reference
from weft.io import write_json
from weft.lab import HumanSession
from weft.recording import _snapshot
from test_spatial_pose import pose_frame


def evidence():
    source = 'a' * 64
    frames = []
    for index in range(10):
        frame = pose_frame(index, flexion=.5 + .02 * index)
        frame.update(id=f'{source}:frame:{index}', frame=index)
        for point in frame['landmarks']:
            point['x'] += index * .004
            point['y'] -= index * .001
        frames.append(json.loads(json.dumps(frame)))
    return {'schema_version': 1, 'profile': 'exercise-pose-v1',
            'source_sha256': source, 'width': 640, 'height': 480, 'fps': 10,
            'frame_count': 10, 'frames': frames,
            'lengths': {'torso': .5, 'shoulder_width': .4, 'hip_width': .26,
                        'upper_arm': .3, 'forearm': .25, 'thigh': .3, 'shin': .25},
            'reference': [{'time_s': frame['time_s'], 'evidence_id': frame['id']} for frame in frames]}


def test_mobile_build_keeps_translation_separate_and_reference_readonly(tmp_path):
    spec = build_human(evidence(), tmp_path, body_model='mobile')
    assert spec['profile'] == 'exercise-v3'
    assert spec['mobility']['floor_fit']['future_poses_used'] is False
    assert json.loads((tmp_path / 'root-motion.json').read_text())['status'] == 'ready'
    env = HumanEnv(spec)
    try:
        env.reset(options={'jitter': False})
        assert env.model.nq == 22 and len(env.joint_names) == 19
        before = env.snapshot()
        state = env.reference_snapshot(9)
        np.testing.assert_allclose(state['human']['mobility']['translation'], spec['root_reference'][9]['translation'])
        assert state['human']['mobility']['scope'] == 'kinematic_overlap_not_dynamic_force'
        assert state['human']['mobility']['support_force'] == [0, 0, 0]
        assert len(state['human']['muscles']) == 38
        assert env.snapshot() == before
        _snapshot(state, env.scene)
    finally:
        env.close()


def test_mobile_session_release_push_restore_are_not_short_joint_tasks(tmp_path):
    build_human(evidence(), tmp_path, body_model='mobile')
    write_json(tmp_path / 'policy.json', {'selected_joint': 'left_knee'})
    session = HumanSession(tmp_path)
    try:
        session.step(6)
        assert not session.done
        assert not session.snapshot()['human']['mobility']['support_enabled']
        assert not np.any(session.env.data.ctrl[session.env._a_array])
        initial = session.snapshot()['human']['mobility']['translation'][0]
        for _ in range(4):
            session.step(7)
        assert session.snapshot()['human']['mobility']['translation'][0] < initial
        assert not session.snapshot()['human']['mobility']['support_enabled']
        session.step(9)
        assert session.snapshot()['human']['mobility']['support_enabled']
        assert not session.done
    finally:
        session.env.close()


def test_mobile_metrics_never_mix_translation_with_radians(tmp_path):
    spec = build_human(evidence(), tmp_path, body_model='mobile')
    result = evaluate_reference(spec, tmp_path)
    assert result['root_reference_rmse_model_units'] < .15
    assert result['maximum_floor_penetration_model_units'] < .025
    metrics = extract_metrics(tmp_path)
    assert 'mobile_reference_rmse_rad' in metrics and 'root_reference_rmse_units' in metrics
    assert 'spatial_reference_rmse_rad' not in metrics and 'reference_rmse_rad' not in metrics
    assert len(result['per_joint_rmse_radians']) == 19


def test_unobserved_long_gap_preserves_audit_without_fallback(tmp_path):
    from weft.errors import UnsupportedVideo
    source = evidence()
    source['reference'] = source['reference'][:2] + source['reference'][5:]
    source['frames'] = source['frames'][:2] + source['frames'][5:]
    with pytest.raises(UnsupportedVideo, match='gaps'):
        build_human(source, tmp_path, body_model='mobile')
    assert json.loads((tmp_path / 'root-motion.json').read_text())['status'] == 'insufficient_reference'
    assert not (tmp_path / 'human.json').exists()
