from copy import deepcopy

import pytest

from test_pose_crop_selection import sequence
from weft.pose_envelope import METHOD, apply_envelope, assess_envelope, plan_envelope


def inputs():
    context = sequence()
    context.update(decoded_frames=13, model={'running_mode':'IMAGE','temporal_tracking':False})
    current = deepcopy(context)
    original = current['frames'][6]
    original.update(status='no_pose',pose_count=0,landmarks=[],world_landmarks=[])
    return context,current


def test_missing_pose_uses_actual_proposal_without_mutating_other_rows_or_prior_derivation():
    context,current=inputs(); current['pose_reinference']={'method':'prior-experiment','changed_frames':[3]}
    current['frames'][3]['image_refinement']={'method':'prior-experiment'}
    frozen=deepcopy(current); plan=plan_envelope(current,context)
    assert plan['targets']==[6] and 3 not in plan['controls']
    proposal=deepcopy(context['frames'][6])
    decision=assess_envelope(current,current['frames'][6],proposal,plan)
    assert decision['accepted']
    result=apply_envelope(current,context,plan,{6:proposal},{6:decision})
    assert result['frames'][6]['image_refinement_raw']==current['frames'][6]
    assert result['pose_reinference']['prior_refinement']==current['pose_reinference']
    assert all(a==b for a,b in zip(current['frames'],result['frames']) if a['frame']!=6)
    assert current==frozen
    with pytest.raises(ValueError): plan_envelope(result,context)


@pytest.mark.parametrize('problem',['source','model','clock','sampling','context_repair','history','small_context','wide_envelope'])
def test_changed_domains_or_unsuitable_context_reject_before_inference(problem):
    context,current=inputs()
    if problem=='source': current['source_sha256']='b'*64
    elif problem=='model': current['model']['running_mode']='VIDEO'
    elif problem=='clock': current['frames'][5]['time_s']+=.1
    elif problem=='sampling': current['frames'].pop()
    elif problem=='context_repair': context['frames'][4]['image_refinement']={'method':'other'}
    elif problem=='history': current['pose_reinference']={'method':'newer','prior_refinement':{'method':METHOD}}
    elif problem=='small_context':
        for row in context['frames'][5:]: row['status']='no_pose'
    elif problem=='wide_envelope':
        context['frames'][0]['landmarks'][0].update(x=0.,y=0.)
        context['frames'][-1]['landmarks'][0].update(x=1.,y=1.)
    with pytest.raises(ValueError): plan_envelope(current,context)


def test_competing_or_previously_modified_hypotheses_are_not_targets():
    context,current=inputs(); current['frames'][6].update(status='multiple_people',pose_count=2)
    assert plan_envelope(current,context)['targets']==[]
    current['frames'][6].update(status='no_pose',pose_count=0,image_refinement={'method':'prior'})
    assert plan_envelope(current,context)['targets']==[]


@pytest.mark.parametrize('problem,reason',[('competing','candidate_unusable_or_competing'),
    ('torso','candidate_torso_not_reliable'),('short_torso','candidate_torso_projection_too_short'),
    ('lost_joint','previous_visible_joint_support_lost'),('hip_jump','reliable_original_hip_position_changed')])
def test_candidate_must_preserve_visible_support_and_reliable_hip_location(problem,reason):
    context,current=inputs(); original=current['frames'][6]; proposal=deepcopy(context['frames'][6])
    if problem in ('lost_joint','hip_jump'):
        original.update(deepcopy(context['frames'][6]));original['status']='invalid_geometry'
    if problem=='competing': proposal.update(pose_count=2,status='multiple_people')
    elif problem=='torso': proposal['landmarks'][11]['presence']=.79
    elif problem=='short_torso':
        for index in (11,12,23,24): proposal['landmarks'][index]['y']=.5
    elif problem=='lost_joint': proposal['landmarks'][15]['presence']=.49
    elif problem=='hip_jump':
        for point in proposal['landmarks']: point['x']+=.3
    plan=plan_envelope(current,context)
    decision=assess_envelope(current,original,proposal,plan)
    assert not decision['accepted'] and reason in decision['reasons']


def test_mutated_plan_decision_source_or_omitted_rejections_cannot_be_applied():
    context,current=inputs(); plan=plan_envelope(current,context); proposal=deepcopy(context['frames'][6])
    decision=assess_envelope(current,current['frames'][6],proposal,plan)
    with pytest.raises(ValueError): apply_envelope(current,context,{**plan,'box_xyxy':[0,0,640,480]},{6:proposal},{6:decision})
    with pytest.raises(ValueError): apply_envelope(current,context,plan,{},{})
    with pytest.raises(ValueError): apply_envelope(current,context,plan,{6:proposal},{6:{**decision,'accepted':False}})
    with pytest.raises(ValueError): assess_envelope(current,current['frames'][6],dict(proposal,time_s=0),plan)
    context['frames'][0]['landmarks'][0]['x']+=.01
    with pytest.raises(ValueError): apply_envelope(current,context,plan,{6:proposal},{6:decision})


def test_large_failure_sets_abstain_instead_of_truncating_the_plan():
    context,current=inputs(); template=deepcopy(context['frames'][0]);context['frames']=[]
    for frame in range(140):
        row=deepcopy(template);row.update(frame=frame,time_s=frame/30,decoder_timestamp_s=frame/30,id=f"{'a'*64}:frame:{frame}")
        context['frames'].append(row)
    context.update(frame_count=140,decoded_frames=140)
    current=deepcopy(context)
    for row in current['frames'][:129]: row.update(status='no_pose',pose_count=0,landmarks=[],world_landmarks=[])
    with pytest.raises(ValueError,match='fixed bound'): plan_envelope(current,context)
