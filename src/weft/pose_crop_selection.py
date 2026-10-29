"""Select single-view crop proposals only for missing geometry or clear torso reversals."""

from copy import deepcopy
import math
from types import MappingProxyType

import numpy as np

from weft.io import canonical_hash
from weft.pose_crop import BODY_JOINTS, plan_crop


METHOD = 'single-view-selective-crop-v1'
CONTRACT = MappingProxyType({
    'method': METHOD, 'minimum_anchor_confidence': .8,
    'maximum_anchor_span_seconds': .3, 'minimum_pair_to_torso_ratio': .15,
    'maximum_anchor_pair_turn_degrees': 45., 'reversal_trigger_degrees': 120.,
    'maximum_candidate_error_degrees': 45., 'minimum_reversal_improvement_degrees': 90.,
    'maximum_other_pair_worsening_degrees': 15., 'maximum_target_frames': 64,
    'scope': 'Temporal image consistency; no camera calibration, verified identity, anatomy or contact',
    'selection': 'Original unusable single/zero hypothesis or a confident short torso-pair reversal',
    'preservation': 'Competing hypotheses and unavailable context remain original; never reduce existing visible major joints',
})
PAIRS = {'shoulders': (11, 12), 'hips': (23, 24)}


def _point(row, index, evidence, minimum=CONTRACT['minimum_anchor_confidence']):
    if type(row.get('pose_count')) is not int or row['pose_count'] != 1:
        return None
    points = row.get('landmarks')
    if not isinstance(points, list) or len(points) != 33:
        return None
    point = points[index]
    values = [point.get(key) for key in ('x', 'y', 'visibility', 'presence')]
    if any(type(v) not in (float, int) or not math.isfinite(v) for v in values):
        return None
    x, y, visibility, presence = values
    if not (0 <= x <= 1 and 0 <= y <= 1 and minimum <= min(visibility, presence) <= max(visibility, presence) <= 1):
        return None
    return np.array([x*evidence['width'], y*evidence['height']])


def _axis(row, pair, evidence):
    points = [_point(row, i, evidence) for i in (11, 12, 23, 24)]
    if any(point is None for point in points):
        return None
    torso = float(np.linalg.norm((points[0]+points[1]-points[2]-points[3])/2))
    a, b = (_point(row, i, evidence) for i in pair)
    axis = b-a
    if torso < 1e-5 or np.linalg.norm(axis)/torso < CONTRACT['minimum_pair_to_torso_ratio']:
        return None
    return axis


def _angle(first, second):
    length = float(np.linalg.norm(first)*np.linalg.norm(second))
    if length < 1e-10:
        return None
    return float(np.degrees(np.arccos(np.clip(np.dot(first, second)/length, -1., 1.))))


def plan_selective_crop(evidence, original):
    plan = {'frame': original['frame'], 'time_s': original['time_s'], 'eligible': False,
            'original_observation_sha256': canonical_hash(original), 'crop': None, 'pairs': {}, 'reason': None}
    if (original.get('id') != f"{evidence['source_sha256']}:frame:{original['frame']}"
            or abs(original['time_s']-original['frame']/evidence['fps']) > 1e-7):
        raise ValueError('Selective crop must preserve source identity and frame time')
    if (type(original.get('pose_count')) is not int or original['pose_count'] not in (0, 1)
            or original.get('image_refinement') or original.get('identity_raw')):
        plan['reason'] = 'competing_or_modified_hypothesis'; return plan
    try:
        crop = plan_crop(evidence, original['frame'])
    except ValueError as error:
        plan['reason'] = str(error); return plan
    indexed = {row['frame']: row for row in evidence['frames']}
    before, after = [indexed[frame] for frame in crop['anchor_frames']]
    span = after['time_s']-before['time_s']
    if not 0 < span <= CONTRACT['maximum_anchor_span_seconds']:
        plan['reason'] = 'anchor_interval_too_long'; return plan
    fraction = (original['time_s']-before['time_s'])/span
    for name, pair in PAIRS.items():
        first, second = (_axis(row, pair, evidence) for row in (before, after))
        if first is None or second is None:
            continue
        turn = _angle(first, second)
        if turn is None or turn > CONTRACT['maximum_anchor_pair_turn_degrees']:
            continue
        expected = (1-fraction)*first+fraction*second
        raw = _axis(original, pair, evidence)
        error = _angle(expected, raw) if raw is not None else None
        plan['pairs'][name] = {'expected_axis_px': expected.tolist(), 'anchor_turn_degrees': turn,
                               'original_error_degrees': error}
    if not plan['pairs']:
        plan['reason'] = 'no_stable_visible_original_torso_pair'; return plan
    usable = original['status'] in ('tracked', 'tracked_partial')
    reversal = any(row['original_error_degrees'] is not None
                   and row['original_error_degrees'] >= CONTRACT['reversal_trigger_degrees'] for row in plan['pairs'].values())
    if usable and not reversal:
        plan['reason'] = 'original_usable_without_clear_reversal'; return plan
    plan.update(eligible=True, crop=crop, reason='clear_torso_reversal' if usable else 'unusable_original_pose')
    return plan


def assess_selective_crop(evidence, original, proposed, plan):
    if not plan['eligible'] or canonical_hash(original) != plan['original_observation_sha256']:
        raise ValueError('Candidate assessment requires the exact original eligible observation')
    if any(original[key] != proposed[key] for key in ('id', 'frame', 'time_s', 'decoder_timestamp_s')):
        raise ValueError('Crop proposal changed source identity or clocks')
    reasons = []
    if type(proposed.get('pose_count')) is not int or proposed['pose_count'] != 1 or proposed['status'] not in ('tracked', 'tracked_partial'):
        reasons.append('candidate_unusable_or_competing')
    lost = [j for j in BODY_JOINTS if _point(original, j, evidence, .5) is not None and _point(proposed, j, evidence, .5) is None]
    if lost: reasons.append('previous_visible_joint_support_lost')
    angles = {}
    for name, detail in plan['pairs'].items():
        axis = _axis(proposed, PAIRS[name], evidence)
        error = _angle(detail['expected_axis_px'], axis) if axis is not None else None
        angles[name] = error
        if error is None or error > CONTRACT['maximum_candidate_error_degrees']:
            reasons.append('candidate_torso_inconsistent_with_original_context')
        raw = detail['original_error_degrees']
        if raw is not None and error is not None:
            if raw >= CONTRACT['reversal_trigger_degrees'] and raw-error < CONTRACT['minimum_reversal_improvement_degrees']:
                reasons.append('reversal_not_resolved')
            elif error-raw > CONTRACT['maximum_other_pair_worsening_degrees']:
                reasons.append('other_torso_pair_worsens')
    return {'frame': original['frame'], 'accepted': not reasons, 'reasons': sorted(set(reasons)),
            'candidate_pair_errors_degrees': angles, 'previous_visible_joints_lost': lost,
            'anatomical_identity_verified': False}


def apply_selective_crop(evidence, proposed, plans, decisions):
    if evidence.get('pose_reinference') or evidence.get('pose_identity', {}).get('mode', 'raw') != 'raw':
        raise ValueError('Selective crops require original unmodified evidence')
    indexed = {row['frame']: row for row in evidence['frames']}
    if len(decisions) > CONTRACT['maximum_target_frames']:
        raise ValueError('Selective crop decisions exceed the frozen per-run bound')
    output = deepcopy(evidence); changed = []
    for i, row in enumerate(output['frames']):
        frame = row['frame']; decision = decisions.get(frame)
        if decision is None or not decision['accepted']:
            continue
        if assess_selective_crop(evidence, indexed[frame], proposed[frame], plans[frame]) != decision:
            raise ValueError('Accepted crop decision does not match its source-based assessment')
        new = deepcopy(proposed[frame])
        new['image_refinement'] = {'method': METHOD, 'crop': plans[frame]['crop'],
            'original_observation_sha256': canonical_hash(row), 'proposal_sha256': canonical_hash(new),
            'decision_sha256': canonical_hash(decision), 'anatomical_identity_verified': False}
        new['image_refinement_raw'] = row
        output['frames'][i] = new; changed.append(frame)
    if set(decisions)-set(indexed):
        raise ValueError('A crop decision refers to an unobserved source frame')
    output['pose_reinference'] = {'method': METHOD, 'contract': dict(CONTRACT),
        'input_evidence_sha256': canonical_hash(evidence), 'changed_frames': changed,
        'source_frame_count_unchanged': True, 'interpolated_observations': 0}
    output.setdefault('assumptions', []).append(
        'Selected image crops replace unusable or temporally reversed hypotheses; original observations remain preserved. '
        'Temporal side consistency is not independent person identity, anatomical or contact validation.')
    return output
