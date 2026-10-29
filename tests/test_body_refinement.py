from copy import deepcopy
import json

import mujoco
import numpy as np
import pytest

from weft.body_frame import CAMERA_TO_SIM
from weft.body_refinement import JOINT_LANDMARKS, REFINEMENT_CONTRACT, _kinematics, refine_reference
from weft.human import HumanEnv
from weft.io import canonical_hash
from weft.spatial_pose import build_spatial_reference, rotation_zyx
from test_spatial_pose import anchor, pose_frame
from test_whole_body_feasibility import model


def fixture(moving=False):
    frames = [pose_frame(index, torso=rotation_zyx([0., 0., .5 * index if moving else 0.]),
                         limb=rotation_zyx([.2, -.2, .1]), flexion=.8) for index in [0, 2]]
    for frame in frames:
        frame['time_s'] += 2.
    spec = model()
    refs, audit = build_spatial_reference({'frames': frames}, anchor())
    spec['reference'] = refs
    spec['reference_evidence'] = audit['frames']
    spec['root_reference'] = [{'time_s': row['time_s'], 'translation': [0., 0., 0.]} for row in refs]
    evidence = {'source_sha256': spec['source_sha256'], 'frames': frames}
    return spec, evidence


def inconsistent():
    spec, evidence = fixture()
    for frame in evidence['frames']:
        # A displaced endpoint conflicts with the unchanged two-bone model lengths.
        frame['world_landmarks'][15]['x'] += .08
    return spec, evidence


def test_perfect_geometry_is_unchanged_even_when_original_motion_is_large(monkeypatch):
    spec, evidence = fixture(moving=True)
    before_spec, before_evidence = deepcopy(spec), deepcopy(evidence)
    def forbidden(*args, **kwargs):
        raise AssertionError('No dynamics or floor fitting')
    monkeypatch.setattr(mujoco, 'mj_step', forbidden)
    monkeypatch.setattr('weft.mobile_human.calibrate_floor', forbidden)
    result, audit = refine_reference(spec, evidence)
    assert spec == before_spec and evidence == before_evidence
    assert result['reference'] == spec['reference']
    assert audit['changed_frames'] == 0
    assert all(row['status'] == 'unchanged_perfect_weighted_geometry' for row in audit['frames'])
    assert all(row['unweighted_endpoint_rmse_before'] < 1e-14 for row in audit['frames'])
    assert result['reference'][1]['joints']['torso_flex'] - result['reference'][0]['joints']['torso_flex'] == pytest.approx(1.)


def test_bounded_fit_improves_inconsistent_target_but_does_not_claim_perfection():
    spec, evidence = inconsistent()
    before_spec, before_evidence = deepcopy(spec), deepcopy(evidence)
    result, audit = refine_reference(spec, evidence)
    assert spec == before_spec and evidence == before_evidence
    assert audit['status'] == 'complete' and audit['physics_steps'] == 0 and audit['promoted'] is False
    for frame in audit['frames']:
        assert 1e-5 < frame['unweighted_endpoint_rmse_after'] < frame['unweighted_endpoint_rmse_before']
        assert frame['objective_after'] < frame['objective_before']
        assert 0 < frame['maximum_angle_change_rad'] <= .15 + 1e-12
        assert frame['iterations'] <= 12
        assert frame['backtrack_evaluations'] <= frame['iterations'] * 4
        assert set(frame['direction_errors_before_rad']) == set(frame['direction_errors_after_rad'])
        assert frame['raw_target_evidence'] == evidence['frames'][frame['reference_index']]
    for field in ['root_reference', 'mobility', 'lengths', 'dt', 'source_sha256', 'assumptions']:
        assert result[field] == spec[field]
    assert [r['time_s'] for r in result['reference']] == [r['time_s'] for r in spec['reference']]
    assert audit['input_model_sha256'] == canonical_hash(spec)
    assert audit['output_model_sha256'] == canonical_hash(result)
    assert result['reference_refinement']['status'] == 'candidate_not_promoted'
    for old,new in zip(spec['reference_evidence'],result['reference_evidence']):
        assert new['joint_confidence'] == old['joint_confidence'] and new['evidence_id'] == old['evidence_id']
    json.dumps(audit, allow_nan=False)


def test_real_mujoco_site_jacobian_matches_finite_differences():
    spec, _ = fixture()
    env = HumanEnv(spec)
    try:
        data = mujoco.MjData(env.model)
        angles = np.array([spec['reference'][0]['joints'][n] for n in env.joint_names])
        _, analytic = _kinematics(env,data,angles,True)
        numeric = np.empty_like(analytic)
        for index in range(19):
            delta = np.zeros(19); delta[index] = 1e-6
            positive,_ = _kinematics(env,data,angles+delta)
            negative,_ = _kinematics(env,data,angles-delta)
            numeric[:,index] = ((positive-negative)/2e-6).ravel()
        np.testing.assert_allclose(analytic,numeric,atol=2e-9)
    finally:
        env.close()


def test_zero_and_cropped_landmarks_stay_in_unweighted_audit_without_driving_fit():
    spec, evidence = inconsistent()
    for frame in evidence['frames']:
        for index in JOINT_LANDMARKS.values():
            frame['landmarks'][index]['visibility'] = 0.
        frame['landmarks'][15].update(visibility=1., presence=1., x=1.2)
    result,audit = refine_reference(spec,evidence)
    assert result['reference'] == spec['reference']
    for row in audit['frames']:
        assert row['status'] == 'unchanged_no_weighted_source_landmarks'
        assert all(weight == 0 for weight in row['landmark_weights'].values())
        assert len(row['target_joints_hip_centered']) == 12
        assert row['unweighted_endpoint_rmse_after'] == row['unweighted_endpoint_rmse_before'] > 0


@pytest.mark.parametrize('mutation, reason', [
    (lambda e: e['frames'].pop(0), 'missing_source_frame'),
    (lambda e: e['frames'][0].pop('world_landmarks'), 'missing_world_or_image_landmarks'),
    (lambda e: e['frames'][0]['world_landmarks'][15].update(x=float('nan')), 'nonfinite_or_unbounded_world_landmarks'),
    (lambda e: e['frames'][0].update(time_s=2.01), 'source_time_mismatch'),
    (lambda e: e['frames'][0].update(pose_count=2), 'unsupported_pose_status_or_multiple_hypotheses')])
def test_missing_unusable_or_ambiguous_targets_remain_unchanged_without_bridging(mutation,reason):
    spec,evidence = inconsistent()
    mutation(evidence)
    result,audit = refine_reference(spec,evidence)
    assert result['reference'][0] == spec['reference'][0]
    assert audit['frames'][0]['status'] == 'unchanged_'+reason
    assert audit['frames'][1]['temporal_prior_source_time_s'] is None
    assert len(audit['frames']) == 2
    json.dumps(audit,allow_nan=False)


def test_compute_deadline_retains_every_original_frame_and_explicit_miss(monkeypatch):
    spec,evidence = inconsistent()
    ticks=iter([0.,61.,62.,63.,64.])
    monkeypatch.setattr('weft.body_refinement.time.monotonic', lambda:next(ticks))
    result,audit=refine_reference(spec,evidence)
    assert result['reference'] == spec['reference']
    assert audit['status'] == 'partial_budget_exhausted'
    assert len(audit['frames']) == len(spec['reference'])
    assert all(row['status']=='unchanged_compute_budget_exhausted' for row in audit['frames'])


def test_joint_chart_and_delta_bounds_hold_at_distal_limit():
    spec,evidence=fixture()
    for row in spec['reference']:
        row['joints']['left_elbow']=2.6
    result,audit=refine_reference(spec,evidence)
    for old,new in zip(spec['reference'],result['reference']):
        for name,angle in new['joints'].items():
            assert abs(angle-old['joints'][name]) <= .15+1e-12
            assert np.isfinite(angle)
            if name.endswith(('_elbow','_knee')):
                assert 0 <= angle <= 2.6


def test_deterministic_results_and_cannot_stack_candidate_corrections():
    spec,evidence=inconsistent()
    first,a=refine_reference(spec,evidence)
    second,b=refine_reference(spec,evidence)
    assert first == second
    assert a['frames'] == b['frames']
    with pytest.raises(ValueError,match='stack'):
        refine_reference(first,evidence)


def test_source_mismatch_and_duplicate_ids_rejected_before_model(monkeypatch):
    spec,evidence=fixture()
    def forbidden(*args):
        raise AssertionError('Invalid identity must fail before construction')
    monkeypatch.setattr('weft.body_refinement.HumanEnv',forbidden)
    mismatch=deepcopy(evidence);mismatch['source_sha256']='b'*64
    with pytest.raises(ValueError,match='identity'):
        refine_reference(spec,mismatch)
    evidence['frames'][1]['id']=evidence['frames'][0]['id']
    with pytest.raises(ValueError,match='unique'):
        refine_reference(spec,evidence)


def test_fit_has_no_floor_or_root_translation_objective():
    spec,evidence=inconsistent()
    other=deepcopy(spec)
    other['mobility']['base_height']=.1
    other['root_reference'][1]['translation']=[.3,0.,-.2]
    first,a=refine_reference(spec,evidence)
    second,b=refine_reference(other,evidence)
    for original, shifted in zip(first['reference'],second['reference']):
        np.testing.assert_allclose(list(original['joints'].values()),list(shifted['joints'].values()),atol=1e-10)
    assert second['mobility'] == other['mobility']
    assert second['root_reference'] == other['root_reference']
    assert 'floor_distance' in b['loss_excludes']


def test_frame_bound_checked_before_model_construction(monkeypatch):
    spec,evidence=fixture()
    evidence['frames'] *= 451
    def forbidden(*args):
        raise AssertionError('Overbound evidence must fail before construction')
    monkeypatch.setattr('weft.body_refinement.HumanEnv',forbidden)
    with pytest.raises(ValueError,match='900'):
        refine_reference(spec,evidence)


def test_temporal_prior_tracks_corrections_not_the_original_motion():
    spec,evidence=fixture(moving=True)
    for frame in evidence['frames']:
        frame['world_landmarks'][15]['x'] += .08
    result,audit=refine_reference(spec,evidence)
    old_motion=spec['reference'][1]['joints']['torso_flex']-spec['reference'][0]['joints']['torso_flex']
    new_motion=result['reference'][1]['joints']['torso_flex']-result['reference'][0]['joints']['torso_flex']
    assert old_motion == pytest.approx(1.)
    assert new_motion > .7
    assert audit['frames'][0]['temporal_prior_source_time_s'] is None
    assert audit['frames'][1]['temporal_prior_source_time_s'] == 2.
    assert all(row['objective_after'] <= row['objective_before'] for row in audit['frames'])
