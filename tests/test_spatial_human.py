from copy import deepcopy
import json
import math

import numpy as np
import pytest

from weft.benchmark import extract_metrics
from weft.body_frame import CAMERA_TO_SIM
from weft.human import HumanEnv, build_human, validate_human
from weft.human_learning import train_human
from weft.io import write_json
from weft.recording import _snapshot
from weft.spatial_pose import SPATIAL_JOINT_NAMES, rotation_zyx
from test_human import human_evidence
from test_spatial_pose import pose_frame


@pytest.fixture
def spatial_spec(human_evidence, tmp_path):
    human_evidence['lengths'].update(torso=.5, shoulder_width=.4, hip_width=.26,
                                    upper_arm=.3, forearm=.25, thigh=.3, shin=.25)
    human_evidence['frames'] = [pose_frame(i, torso=rotation_zyx([.2, -.1, i*.15]),
                                         limb=rotation_zyx([.2, -.7, .3])) for i in range(4)]
    return build_human(human_evidence, tmp_path/'spatial', body_model='spatial'), human_evidence


def test_spatial_mujoco_matches_source_segment_directions(spatial_spec):
    spec, evidence = spatial_spec
    env = HumanEnv(spec)
    try:
        assert env.model.nv == env.model.nq == 19
        assert len(json.dumps(env.scene)) < 20000
        assert 'reference' not in env.scene['human']
        env.reset(seed=0, options={'jitter': False})
        for i, frame in enumerate(evidence['frames']):
            state = env.reference_snapshot(i)
            world = np.array([[p[k] for k in ('x','y','z')] for p in frame['world_landmarks']]) @ CAMERA_TO_SIM.T
            pairs = {'left_upper_arm':(11,13),'left_forearm':(13,15),'right_thigh':(24,26),'right_shin':(26,28)}
            for segment in state['human']['segments']:
                if segment['id'] not in pairs:
                    continue
                a,b=pairs[segment['id']]
                np.testing.assert_allclose(np.subtract(segment['b'],segment['a']),world[b]-world[a],atol=1e-8)
        assert len(state['human']['muscles']) == 38
    finally:
        env.close()


def test_all_spatial_axes_have_actions_and_fixed_pelvis(spatial_spec):
    env = HumanEnv(spatial_spec[0])
    try:
        for name in SPATIAL_JOINT_NAMES:
            final=[]
            for action in (1,2):
                env.reset(seed=0,options={'jitter':False,'selected_joint':name})
                before=env.snapshot()['human']['joints']['pelvis']
                for _ in range(12):
                    env.advance_physics(action)
                state=env.snapshot()
                final.append(state['human']['angles'][name])
                np.testing.assert_allclose(state['human']['joints']['pelvis'],before,atol=1e-10)
                assert np.isfinite(list(state['human']['angles'].values())).all()
                _snapshot(state,env.scene)
            assert abs(final[0]-final[1])>.001, name
    finally:
        env.close()


def test_spatial_schema_recording_and_learning_contract(spatial_spec,tmp_path):
    spec,_=spatial_spec
    bad=deepcopy(spec);bad['schema_version']=1
    with pytest.raises(ValueError,match='schema'):
        validate_human(bad)
    bad=deepcopy(spec);bad['reference'][0]['joints'].pop('torso_flex')
    with pytest.raises(ValueError,match='all joints'):
        validate_human(bad)
    env=HumanEnv(spec)
    try:
        env.reset(seed=0)
        state=env.snapshot();state.pop('human')
        with pytest.raises(ValueError,match='complete human'):
            _snapshot(state,env.scene)
    finally:
        env.close()
    result=train_human(spec,tmp_path/'learning',episodes=2,eval_episodes=2,selected_joint='left_shoulder_lateral')
    assert result['training']['selected_joint']=='left_shoulder_lateral'


def test_supported_torso_holds_without_inventing_force_measurements(spatial_spec):
    spec,_=spatial_spec
    spec=deepcopy(spec)
    for row in spec['reference']:
        row['joints']=deepcopy(spec['reference'][0]['joints'])
    env=HumanEnv(spec)
    try:
        env.reset(seed=0,options={'jitter':False,'selected_joint':'torso_flex'})
        original=env.snapshot()['human']['angles']['torso_flex']
        for _ in range(90):
            env.advance_physics(4)
        assert abs(env.snapshot()['human']['angles']['torso_flex']-original)<.02
        assert np.all(np.abs(env.data.ctrl[env._a_array])<=env._motor_limits)
        assert env.observation_space.contains(env.observation())
    finally:
        env.close()


def test_new_rotation_model_never_overwrites_legacy_servo_metric(tmp_path):
    write_json(tmp_path/'human-evaluation.json',{'reference_servo_rmse_radians':.2,'model_profile':'exercise-v2'})
    metrics=extract_metrics(tmp_path)
    assert metrics['spatial_reference_rmse_rad']==.2
    assert 'reference_rmse_rad' not in metrics


def test_spatial_view_encloses_every_limb_for_any_yaw(spatial_spec):
    env=HumanEnv(spatial_spec[0])
    try:
        for index in range(len(spatial_spec[0]['reference'])):
            human=env.reference_snapshot(index)['human']
            origin=np.array(human['view_origin'])
            for segment in human['segments']:
                for end in ('a','b'):
                    assert np.linalg.norm(np.array(segment[end])-origin)+segment['radius'] < human['view_height']/2
        env.reset(seed=0)
        assert env.render().shape[2]==3
    finally:
        env.close()
