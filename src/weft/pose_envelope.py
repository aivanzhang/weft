"""Bounded image re-inference from an original whole-interval pose envelope."""

from copy import deepcopy
import math
from types import MappingProxyType

import numpy as np

from weft.io import canonical_hash
from weft.pose_crop import BODY_JOINTS
from weft.pose_crop_selection import _point


METHOD = 'whole-interval-pose-envelope-v1'
CONTRACT = MappingProxyType({
    'method': METHOD, 'maximum_samples': 900, 'maximum_targets': 128,
    'minimum_context_frames': 8, 'minimum_context_points': 8,
    'context_confidence': .8, 'padding_fraction': .25,
    'minimum_crop_side_pixels': 64, 'maximum_crop_area_fraction': .8,
    'minimum_torso_pixels': 8., 'maximum_original_hip_shift_torso_fraction': .5,
    'scope': 'Fixed-view image-context hypothesis; no recovered camera, physical identity, anatomy or contact',
    'selection': 'Only unusable zero/single hypotheses; preserve usable/competing/previously refined observations',
})


def _history_contains(record):
    for _ in range(8):
        if not record: return False
        if not isinstance(record, dict): raise ValueError('Refinement history must contain records')
        if record.get('method') == METHOD: return True
        record = record.get('prior_refinement')
    raise ValueError('Refinement history exceeds its bound')


def plan_envelope(evidence, context):
    if (_history_contains(evidence.get('pose_reinference')) or context.get('pose_reinference')
            or any(item.get('pose_identity', {}).get('mode') != 'raw' for item in (evidence, context))):
        raise ValueError('Use original raw context and do not repeat envelope refinement')
    for key in ('source_sha256', 'width', 'height', 'fps', 'frame_count', 'decoded_frames', 'model'):
        if evidence.get(key) != context.get(key): raise ValueError('Envelope input domains or models differ')
    if (context.get('model', {}).get('running_mode') != 'IMAGE'
            or context['model'].get('temporal_tracking') is not False):
        raise ValueError('Envelope recovery requires independent IMAGE-mode observations')
    if not 1 <= len(context['frames']) <= CONTRACT['maximum_samples'] or len(context['frames']) != len(evidence['frames']):
        raise ValueError('Preserve the complete bounded original sampling')
    previous = -1
    for raw, current in zip(context['frames'], evidence['frames'], strict=True):
        frame = raw['frame']
        if (type(frame) is not int or not previous < frame < evidence['decoded_frames']
                or raw['id'] != f"{evidence['source_sha256']}:frame:{frame}"
                or abs(raw['time_s']-frame/evidence['fps']) > 1e-7
                or any(current[key] != raw[key] for key in ('id', 'frame', 'time_s', 'decoder_timestamp_s'))
                or raw.get('image_refinement') or raw.get('identity_raw')):
            raise ValueError('Original sampling, clocks, identity or raw context changed')
        previous = frame
    points, support = [], []
    for row in context['frames']:
        if row['status'] not in ('tracked', 'tracked_partial'): continue
        visible = [_point(row, index, context, CONTRACT['context_confidence']) for index in range(33)]
        visible = [point for point in visible if point is not None]
        if len(visible) >= CONTRACT['minimum_context_points']:
            points.extend(visible); support.append(row['frame'])
    if len(support) < CONTRACT['minimum_context_frames']:
        raise ValueError('Original poses do not define a reliable source envelope')
    points = np.array(points); low, high = points.min(0), points.max(0)
    padding = (high-low)*CONTRACT['padding_fraction']
    box = [max(0, math.floor(low[0]-padding[0])), max(0, math.floor(low[1]-padding[1])),
           min(evidence['width'], math.ceil(high[0]+padding[0])), min(evidence['height'], math.ceil(high[1]+padding[1]))]
    width, height = box[2]-box[0], box[3]-box[1]
    if min(width, height) < CONTRACT['minimum_crop_side_pixels'] or width*height > evidence['width']*evidence['height']*CONTRACT['maximum_crop_area_fraction']:
        raise ValueError('Original envelope does not isolate a useful bounded image region')
    targets = [row for row in evidence['frames'] if row['status'] not in ('tracked','tracked_partial')
        and type(row.get('pose_count')) is int and row['pose_count'] in (0,1)
        and not row.get('image_refinement') and not row.get('identity_raw')]
    if len(targets) > CONTRACT['maximum_targets']: raise ValueError('Recovery targets exceed the fixed bound')
    usable = [row['frame'] for row in evidence['frames'] if row['status'] in ('tracked','tracked_partial')
        and row['pose_count'] == 1 and not row.get('image_refinement') and not row.get('identity_raw')]
    controls = sorted({usable[0], usable[len(usable)//2], usable[-1]}) if usable else []
    return {'method': METHOD, 'contract': dict(CONTRACT), 'box_xyxy': box, 'padding_fraction': CONTRACT['padding_fraction'],
        'source_sha256': evidence['source_sha256'], 'evidence_sha256': canonical_hash(evidence),
        'context_sha256': canonical_hash(context), 'support_frames': support,
        'targets': [row['frame'] for row in targets], 'controls': controls,
        'original_observation_hashes': {str(row['frame']): canonical_hash(row) for row in targets}}


def assess_envelope(evidence, original, proposed, plan):
    frame = original['frame']
    if frame not in plan['targets'] or plan['original_observation_hashes'][str(frame)] != canonical_hash(original):
        raise ValueError('Candidate must match a frozen unusable original observation')
    if any(proposed[key] != original[key] for key in ('id','frame','time_s','decoder_timestamp_s')):
        raise ValueError('Candidate changed source identity or clocks')
    reasons = []
    if type(proposed.get('pose_count')) is not int or proposed['pose_count'] != 1 or proposed['status'] not in ('tracked','tracked_partial'):
        reasons.append('candidate_unusable_or_competing')
    lost = [index for index in BODY_JOINTS if _point(original,index,evidence,.5) is not None and _point(proposed,index,evidence,.5) is None]
    if lost: reasons.append('previous_visible_joint_support_lost')
    torso = [_point(proposed,index,evidence,.8) for index in (11,12,23,24)]
    shift = None
    if any(point is None for point in torso):
        reasons.append('candidate_torso_not_reliable')
    else:
        middle = (torso[2]+torso[3])/2
        length = float(np.linalg.norm((torso[0]+torso[1])/2-middle))
        if length < CONTRACT['minimum_torso_pixels']:
            reasons.append('candidate_torso_projection_too_short')
        old_hips = [_point(original,index,evidence,.8) for index in (23,24)]
        if all(point is not None for point in old_hips) and length >= CONTRACT['minimum_torso_pixels']:
            shift = float(np.linalg.norm(middle-(old_hips[0]+old_hips[1])/2)/length)
            if shift > CONTRACT['maximum_original_hip_shift_torso_fraction']:
                reasons.append('reliable_original_hip_position_changed')
    return {'frame': frame, 'accepted': not reasons, 'reasons': sorted(set(reasons)),
        'previous_visible_joints_lost': lost, 'hip_shift_torso_fraction': shift,
        'anatomical_identity_verified': False, 'physical_contact_verified': False}


def apply_envelope(evidence, context, plan, proposed, decisions):
    if plan_envelope(evidence, context) != plan:
        raise ValueError('Envelope plan no longer matches its original source context')
    if set(proposed) != set(plan['targets']) or set(decisions) != set(plan['targets']):
        raise ValueError('Retain every planned proposal and decision, including rejections')
    output = deepcopy(evidence); changed = []
    for index, original in enumerate(evidence['frames']):
        frame = original['frame']
        if frame not in decisions: continue
        decision = assess_envelope(evidence, original, proposed[frame], plan)
        if decision != decisions[frame]: raise ValueError('Saved decision differs from the source-bound assessment')
        if not decision['accepted']: continue
        row = deepcopy(proposed[frame])
        row['image_refinement'] = {'method': METHOD, 'crop': {'frame': frame, 'box_xyxy': plan['box_xyxy']},
            'plan_sha256': canonical_hash(plan), 'original_observation_sha256': canonical_hash(original),
            'proposal_sha256': canonical_hash(row), 'decision_sha256': canonical_hash(decision),
            'anatomical_identity_verified': False}
        row['image_refinement_raw'] = deepcopy(original)
        output['frames'][index] = row; changed.append(frame)
    output['pose_reinference'] = {'method': METHOD, 'contract': dict(CONTRACT),
        'input_evidence_sha256': canonical_hash(evidence), 'plan_sha256': canonical_hash(plan),
        'changed_frames': changed, 'source_frame_count_unchanged': True, 'interpolated_observations': 0,
        'prior_refinement': deepcopy(evidence.get('pose_reinference'))}
    output.setdefault('assumptions', []).append(
        'Original whole-interval image poses define one fixed-view crop for unusable frames; every usable pose stays unchanged. '
        'New pixel inference is not verified anatomical identity, camera calibration or physical contact.')
    return output
