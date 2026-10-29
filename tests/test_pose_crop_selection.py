from copy import deepcopy
import math

import pytest

from test_exercise_vision import inference_result
from weft.exercise_vision import summarize_result
from weft.pose_crop_selection import apply_selective_crop, assess_selective_crop, plan_selective_crop


def sequence():
    evidence = {'source_sha256': 'a'*64, 'width': 640, 'height': 480, 'fps': 30., 'frame_count': 13,
                'pose_identity': {'mode': 'raw'}, 'frames': []}
    for index in range(13):
        row = summarize_result(inference_result())
        row.update(frame=index, time_s=index/30, decoder_timestamp_s=index/30,
                   id=f"{'a'*64}:frame:{index}", inference_seconds=.001)
        evidence['frames'].append(row)
    return evidence


def reverse(row, pairs=((11, 12), (23, 24))):
    for a, b in pairs: row['landmarks'][a], row['landmarks'][b] = row['landmarks'][b], row['landmarks'][a]


def rotate(row, angle, pairs=((11, 12), (23, 24))):
    c, s = math.cos(math.radians(angle)), math.sin(math.radians(angle))
    for a, b in pairs:
        first, second = row['landmarks'][a], row['landmarks'][b]
        cx, cy = (first['x']+second['x'])*320, (first['y']+second['y'])*240
        for point in (first, second):
            x, y = point['x']*640-cx, point['y']*480-cy
            point.update(x=(cx+c*x-s*y)/640, y=(cy+s*x+c*y)/480)


def test_clear_torso_reversal_can_use_fresh_image_hypothesis_with_originals_retained():
    evidence = sequence(); original = evidence['frames'][6]; proposal = deepcopy(original)
    reverse(original); before = deepcopy(evidence)
    plan = plan_selective_crop(evidence, original)
    assert plan['eligible'] and plan['reason'] == 'clear_torso_reversal'
    decision = assess_selective_crop(evidence, original, proposal, plan)
    assert decision['accepted'] and decision['anatomical_identity_verified'] is False
    updated = apply_selective_crop(evidence, {6: proposal}, {6: plan}, {6: decision})
    assert updated['frames'][6]['image_refinement_raw'] == original
    assert updated['frames'][6]['landmarks'] == proposal['landmarks']
    assert all(old == new for i, (old, new) in enumerate(zip(evidence['frames'], updated['frames'])) if i != 6)
    assert evidence == before
    with pytest.raises(ValueError): apply_selective_crop(updated, {6: proposal}, {6: plan}, {6: decision})


@pytest.mark.parametrize('status,count', [('invalid_geometry', 1), ('low_confidence', 1), ('no_pose', 0)])
def test_unusable_original_needs_actual_usable_context_consistent_proposal(status, count):
    evidence = sequence(); original = evidence['frames'][6]; proposal = deepcopy(original)
    original.update(status=status, pose_count=count)
    if count == 0: original.update(landmarks=[], world_landmarks=[])
    plan = plan_selective_crop(evidence, original)
    assert plan['eligible']
    assert assess_selective_crop(evidence, original, proposal, plan)['accepted']
    proposal['status'] = 'low_confidence'
    assert not assess_selective_crop(evidence, original, proposal, plan)['accepted']


@pytest.mark.parametrize('motion', ['ordinary', 'smooth_turn', 'arm_crossing', 'rapid_context_turn', 'small_disagreement'])
def test_consistent_or_ambiguous_motion_does_not_trigger_another_inference(motion):
    evidence = sequence(); original = evidence['frames'][6]
    if motion == 'smooth_turn':
        for i, row in enumerate(evidence['frames']): rotate(row, (i-6)*5)
    elif motion == 'arm_crossing': reverse(original, ((13, 14), (15, 16)))
    elif motion == 'rapid_context_turn':
        rotate(evidence['frames'][5], -70); rotate(evidence['frames'][7], 70); reverse(original)
    elif motion == 'small_disagreement': rotate(original, 35)
    assert not plan_selective_crop(evidence, original)['eligible']


@pytest.mark.parametrize('problem', ['competing', 'already_refined', 'identity_repair', 'hidden_anchor', 'overlap', 'long_gap', 'boundary'])
def test_unreliable_or_modified_context_abstains(problem):
    evidence = sequence(); original = evidence['frames'][0 if problem == 'boundary' else 6]
    reverse(original)
    if problem == 'competing': original.update(pose_count=2, status='multiple_people')
    elif problem == 'already_refined': original['image_refinement'] = {'method': 'previous'}
    elif problem == 'identity_repair': original['identity_raw'] = {'status': 'previous'}
    elif problem in ('hidden_anchor', 'overlap'):
        for i in (5, 7):
            for j in (11, 12, 23, 24):
                if problem == 'hidden_anchor': evidence['frames'][i]['landmarks'][j]['presence'] = .79
                else: evidence['frames'][i]['landmarks'][j]['x'] = .5
    elif problem == 'long_gap':
        for row in evidence['frames']:
            if row['frame'] not in (0, 6, 12): row['status'] = 'no_pose'
    assert not plan_selective_crop(evidence, original)['eligible']


@pytest.mark.parametrize('problem,reason', [('lost_wrist', 'previous_visible_joint_support_lost'),
    ('still_reversed', 'candidate_torso_inconsistent_with_original_context'),
    ('other_pair', 'other_torso_pair_worsens'), ('competing', 'candidate_unusable_or_competing')])
def test_proposals_cannot_trade_existing_observations_for_a_torso_fix(problem, reason):
    evidence = sequence(); original = evidence['frames'][6]; proposal = deepcopy(original)
    reverse(original, ((11, 12),))
    if problem == 'lost_wrist': proposal['landmarks'][15]['presence'] = .49
    elif problem == 'still_reversed': reverse(proposal)
    elif problem == 'other_pair': rotate(proposal, 30, ((23, 24),))
    else: proposal['pose_count'] = 2
    plan = plan_selective_crop(evidence, original)
    assert plan['eligible']
    decision = assess_selective_crop(evidence, original, proposal, plan)
    assert not decision['accepted'] and reason in decision['reasons']


def test_changed_source_clock_original_or_decision_cannot_be_applied():
    evidence = sequence(); original = evidence['frames'][6]; proposal = deepcopy(original)
    reverse(original); plan = plan_selective_crop(evidence, original)
    with pytest.raises(ValueError): plan_selective_crop(evidence, dict(original, id='another-source'))
    with pytest.raises(ValueError): assess_selective_crop(evidence, dict(original, status='changed'), proposal, plan)
    with pytest.raises(ValueError): assess_selective_crop(evidence, original, dict(proposal, time_s=0), plan)
    decision = assess_selective_crop(evidence, original, proposal, plan)
    with pytest.raises(ValueError):
        apply_selective_crop(evidence, {6: proposal}, {6: plan}, {6: dict(decision, candidate_pair_errors_degrees={})})
    with pytest.raises(ValueError): apply_selective_crop(evidence, {}, {}, {i: {'accepted': False} for i in range(65)})
