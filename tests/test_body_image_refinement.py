from copy import deepcopy
import json

import mujoco
import numpy as np
import pytest

from weft.body_image_refinement import (_projected_kinematics, refine_image_reference,
                                      IMAGE_REFINEMENT_CONTRACT, SEGMENT_NAMES)
from weft.body_refinement import JOINT_LANDMARKS
from weft.human import HumanEnv
from test_body_refinement import fixture


def image_fixture(target_rotation=0., width=1200, height=600, moving=False):
    spec,evidence=fixture(moving=moving)
    evidence.update(width=width,height=height,image_dimension_provenance={
        'method':'authored_analytic_fixture','source_sha256':spec['source_sha256'],'width':width,'height':height})
    env=HumanEnv(spec)
    try:
        data=mujoco.MjData(env.model)
        for reference,frame in zip(spec['reference'],evidence['frames']):
            angles=dict(reference['joints']);angles['torso_lateral']+=target_rotation
            data.qpos[env._q_array]=[angles[n] for n in env.joint_names]
            mujoco.mj_forward(env.model,data)
            center=data.site_xpos[env._site_ids['pelvis']]
            for name,index in JOINT_LANDMARKS.items():
                point=data.site_xpos[env._site_ids[name]]-center
                frame['landmarks'][index].update(x=.5+200*point[0]/width,y=.5-200*point[2]/height,
                                                  visibility=.9,presence=.9)
    finally:
        env.close()
    return spec,evidence


def mean_error(row,when):
    return np.mean([v for v in row[f'direction_errors_{when}_rad'].values() if v is not None])


def test_perfect_nonsquare_projection_is_unchanged_without_world_targets(monkeypatch):
    spec,evidence=image_fixture()
    for frame in evidence['frames']:
        frame.pop('world_landmarks')
    original_spec,original_evidence=deepcopy(spec),deepcopy(evidence)
    def forbidden(*args,**kwargs):
        raise AssertionError('No dynamics, body servo or floor fit')
    monkeypatch.setattr(mujoco,'mj_step',forbidden)
    monkeypatch.setattr('weft.mobile_human.calibrate_floor',forbidden)
    monkeypatch.setattr('weft.mobile_human.MobileDynamics.apply',forbidden)
    result,audit=refine_image_reference(spec,evidence)
    assert result['reference']==spec['reference']
    assert spec==original_spec and evidence==original_evidence
    assert all(row['status']=='unchanged_perfect_weighted_projection' for row in audit['frames'])
    assert all(mean_error(row,'before')<1e-14 for row in audit['frames'])
    assert audit['image_dimensions']['width']==1200
    assert audit['changed_frames']==0


def test_known_image_plane_rotation_improves_with_frozen_bounds():
    spec,evidence=image_fixture(target_rotation=.08)
    result,audit=refine_image_reference(spec,evidence)
    assert audit['status']=='complete' and audit['promoted'] is False
    for row in audit['frames']:
        assert mean_error(row,'after')<mean_error(row,'before')*.1
        assert row['objective_after']<row['objective_before']
        assert row['maximum_angle_change_rad']<=.15
        assert row['iterations']<=12 and row['backtrack_evaluations']<=48
        assert row['weighted_segments']==8
        assert row['raw_source_evidence']==evidence['frames'][row['reference_index']]
    for key in ['mobility','root_reference','lengths','dt','body_frame','motor_limits']:
        assert result.get(key)==spec.get(key)
    assert [r['time_s'] for r in result['reference']]==[r['time_s'] for r in spec['reference']]
    for old,new in zip(spec['reference_evidence'],result['reference_evidence']):
        assert old['joint_confidence']==new['joint_confidence']
        assert new['image_reference_derivation']['depth_identified'] is False
    json.dumps(audit,allow_nan=False)


def test_aspect_ratio_is_required_for_correct_diagonal_directions():
    spec,evidence=image_fixture()
    wrong=deepcopy(evidence)
    wrong['height']=1200;wrong['image_dimension_provenance']['height']=1200
    _,correct=refine_image_reference(spec,evidence)
    _,incorrect=refine_image_reference(spec,wrong)
    assert mean_error(correct['frames'][0],'before')<1e-14
    assert mean_error(incorrect['frames'][0],'before')>.01


def test_actual_unit_projection_jacobian_matches_numeric_derivative():
    spec,_=image_fixture()
    env=HumanEnv(spec)
    try:
        data=mujoco.MjData(env.model)
        angles=np.array([spec['reference'][0]['joints'][n] for n in env.joint_names])
        _,_,valid,analytic=_projected_kinematics(env,data,angles,True)
        assert valid.all()
        numeric=np.empty_like(analytic)
        for index in range(19):
            delta=np.zeros(19);delta[index]=1e-6
            plus=_projected_kinematics(env,data,angles+delta)[0]
            minus=_projected_kinematics(env,data,angles-delta)[0]
            numeric[:,index]=((plus-minus)/2e-6).ravel()
        np.testing.assert_allclose(analytic,numeric,atol=2e-8)
    finally:
        env.close()


def test_world_depth_is_not_a_target_or_identified_by_same_projection():
    spec,evidence=image_fixture(target_rotation=.08)
    alternative=deepcopy(evidence)
    for frame in alternative['frames']:
        for point in frame['world_landmarks']:
            point['z']=-point['z']+2.
    first,a=refine_image_reference(spec,evidence)
    second,b=refine_image_reference(spec,alternative)
    assert first['reference']==second['reference']
    assert a['input_evidence_sha256']!=b['input_evidence_sha256']
    assert 'multiple 3D' in a['uncertainty']
    assert all(row['depth_status'].startswith('unidentified') for row in a['frames'])


def test_two_different_3d_poses_with_same_image_keep_their_original_depth():
    spec,evidence=image_fixture()
    for row in spec['reference']:
        row['joints']={name:0. for name in row['joints']}
        for side in ['left','right']:
            row['joints'][side+'_shoulder']=.5
            row['joints'][side+'_hip']=.5
    other=deepcopy(spec)
    for row in other['reference']:
        for side in ['left','right']:
            row['joints'][side+'_shoulder']=-.5
            row['joints'][side+'_hip']=-.5
    # Both poses project to vertical segments, but their depth components have opposite signs.
    for frame in evidence['frames']:
        for a,b in [('left_shoulder','left_elbow'),('left_elbow','left_wrist'),('left_hip','left_knee'),('left_knee','left_ankle'),
                    ('right_shoulder','right_elbow'),('right_elbow','right_wrist'),('right_hip','right_knee'),('right_knee','right_ankle')]:
            start=frame['landmarks'][JOINT_LANDMARKS[a]]
            frame['landmarks'][JOINT_LANDMARKS[b]].update(x=start['x'],y=start['y']+.05)
    first,a=refine_image_reference(spec,evidence)
    second,b=refine_image_reference(other,evidence)
    assert first['reference']==spec['reference'] and second['reference']==other['reference']
    assert first['reference']!=second['reference']
    assert all(mean_error(row,'before')<1e-14 for row in a['frames']+b['frames'])


def test_cropped_zero_confidence_missing_and_short_endpoints_are_explicit():
    spec,evidence=image_fixture(target_rotation=.08)
    for frame in evidence['frames']:
        frame['landmarks'][15]['x']=1.2
        frame['landmarks'][16]['visibility']=0.
        frame['landmarks'][25]=None
        frame['landmarks'][28]=dict(frame['landmarks'][26])
    _,audit=refine_image_reference(spec,evidence)
    for row in audit['frames']:
        segments={s['segment']:s for s in row['segments']}
        assert segments['left_elbow__left_wrist']['source_status']=='outside_image_crop_tolerance'
        assert segments['right_elbow__right_wrist']['source_status']=='zero_source_confidence'
        assert segments['left_hip__left_knee']['source_status']=='missing_endpoint'
        assert segments['right_knee__right_ankle']['source_status']=='observed_projection_degenerate'
        for name in ['left_elbow__left_wrist','right_elbow__right_wrist','left_hip__left_knee','right_knee__right_ankle']:
            assert segments[name]['effective_fit_weight']==0


def test_model_depth_aligned_bone_is_not_assigned_an_invented_direction():
    spec,evidence=image_fixture()
    for row in spec['reference']:
        row['joints']['left_shoulder']=np.pi/2
        row['joints']['left_shoulder_twist']=0.
        row['joints']['left_shoulder_lateral']=0.
        row['joints']['left_elbow']=0.
    _,audit=refine_image_reference(spec,evidence)
    for row in audit['frames']:
        segment=next(s for s in row['segments'] if s['segment']=='left_shoulder__left_elbow')
        assert segment['model_direction_status']=='model_projection_degenerate'
        assert segment['effective_fit_weight']==0
        assert row['direction_errors_before_rad']['left_shoulder__left_elbow'] is None


def test_zero_confidence_preserves_estimates_and_does_not_move_angles():
    spec,evidence=image_fixture(target_rotation=.08)
    for frame in evidence['frames']:
        for point in frame['landmarks']:
            point['visibility']=0.
    result,audit=refine_image_reference(spec,evidence)
    assert result['reference']==spec['reference']
    assert all(r['status']=='unchanged_no_usable_weighted_directions' for r in audit['frames'])
    assert all(r['comparable_segments']==8 and r['weighted_segments']==0 for r in audit['frames'])
    assert all(mean_error(r,'before')>0 for r in audit['frames'])


def test_no_floor_or_root_position_objective():
    spec,evidence=image_fixture(target_rotation=.08)
    other=deepcopy(spec);other['mobility']['base_height']=.1
    other['root_reference'][1]['translation']=[.3,0.,-.2]
    first,_=refine_image_reference(spec,evidence)
    second,audit=refine_image_reference(other,evidence)
    for a,b in zip(first['reference'],second['reference']):
        np.testing.assert_allclose(list(a['joints'].values()),list(b['joints'].values()),atol=1e-9)
    assert 'floor_distance' in audit['loss_excludes']


@pytest.mark.parametrize('mutation',[
    lambda s,e:e.pop('image_dimension_provenance'),
    lambda s,e:e.update(width=0),
    lambda s,e:e['image_dimension_provenance'].update(height=10),
    lambda s,e:e.update(source_sha256='b'*64),
    lambda s,e:s.update(reference_refinement={}),
    lambda s,e:s.update(image_reference_refinement={}),
    lambda s,e:e.update(frames=e['frames']*451)])
def test_invalid_identity_dimensions_stacking_and_work_rejected_before_model(mutation,monkeypatch):
    spec,evidence=image_fixture()
    mutation(spec,evidence)
    def forbidden(*args):
        raise AssertionError('Invalid input must fail before construction')
    monkeypatch.setattr('weft.body_image_refinement.HumanEnv',forbidden)
    with pytest.raises(ValueError):
        refine_image_reference(spec,evidence)


def test_deadline_keeps_all_original_frames(monkeypatch):
    spec,evidence=image_fixture(target_rotation=.08)
    ticks=iter([0.,61.,62.,63.,64.])
    monkeypatch.setattr('weft.body_image_refinement.time.monotonic',lambda:next(ticks))
    result,audit=refine_image_reference(spec,evidence)
    assert result['reference']==spec['reference']
    assert audit['status']=='partial_budget_exhausted'
    assert len(audit['frames'])==2
    assert all(r['status']=='unchanged_compute_budget_exhausted' for r in audit['frames'])


def test_missing_frame_breaks_temporal_correction_prior_and_retains_timestamps():
    spec,evidence=image_fixture(target_rotation=.08)
    evidence['frames'].pop(0)
    result,audit=refine_image_reference(spec,evidence)
    assert result['reference'][0]==spec['reference'][0]
    assert audit['frames'][0]['status']=='unchanged_missing_source_frame'
    assert audit['frames'][1]['temporal_prior_source_time_s'] is None


def test_temporal_regularization_does_not_flatten_perfect_original_motion():
    spec,evidence=image_fixture(moving=True)
    result,audit=refine_image_reference(spec,evidence)
    assert result['reference']==spec['reference']
    assert result['reference'][1]['joints']['torso_flex']-result['reference'][0]['joints']['torso_flex']==pytest.approx(1.)
    assert all(r['maximum_angle_change_rad']==0 for r in audit['frames'])


def test_nonfinite_and_extreme_cropped_coordinates_are_retained_json_safely():
    spec,evidence=image_fixture()
    for frame in evidence['frames']:
        frame['landmarks'][15]['x']=1e308
        frame['landmarks'][16]['y']=float('nan')
    _,audit=refine_image_reference(spec,evidence)
    json.dumps(audit,allow_nan=False)
    for row in audit['frames']:
        segments={s['segment']:s for s in row['segments']}
        assert segments['left_elbow__left_wrist']['source_status']=='outside_image_crop_tolerance'
        assert segments['right_elbow__right_wrist']['source_status']=='nonfinite_coordinates_or_invalid_confidence'
        assert row['raw_source_evidence']['landmarks'][16]['y']=={'nonfinite_raw_value':'nan'}
