"""Bounded image re-inference proposals for inconsistent two-view hip observations."""

from copy import deepcopy
import math
from types import MappingProxyType

import numpy as np

from weft.exercise_vision import summarize_landmarks
from weft.io import canonical_hash


METHOD = 'two-view-temporal-crop-v1'
CONTRACT = MappingProxyType({
    'method': METHOD, 'maximum_target_frames': 32, 'maximum_anchor_gap_seconds': .2,
    'crop_padding_fraction': .25, 'minimum_anchor_visible_points': 8,
    'minimum_crop_side_px': 64, 'maximum_crop_area_fraction': .8,
    'maximum_anchor_box_ratio': 2., 'maximum_anchor_center_travel_ratio': .5,
    'minimum_estimated_body_joints': 8, 'maximum_root_deviation_torso_ratio': .25,
    'selection': 'Original same-label hip rays disagree; never infer person count or choose a side permutation',
    'acceptance': 'One crop per view; both new hypotheses must pass; no padding or candidate search',
    'scope': 'Fresh image-model observations with temporal crop context, not verified anatomy or contact',
})
BODY_JOINTS = (11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28)


def plan_crop(evidence, frame):
    width, height, fps = (evidence[key] for key in ('width', 'height', 'fps'))
    if (type(frame) is not int or not 0 <= frame < evidence['frame_count']
            or any(type(v) not in (int, float) or not math.isfinite(v) or v <= 0 for v in (width, height, fps))):
        raise ValueError('Crop target and source dimensions must be finite and within the source interval')
    anchors = [row for row in evidence['frames'] if row['frame'] != frame
               and abs(row['time_s']-frame/fps) <= CONTRACT['maximum_anchor_gap_seconds']
               and row['status'] in ('tracked', 'tracked_partial') and row['pose_count'] == 1
               and 'image_refinement' not in row and 'identity_raw' not in row]
    before = [row for row in anchors if row['frame'] < frame]
    after = [row for row in anchors if row['frame'] > frame]
    if not before or not after:
        raise ValueError('Crop requires original single-pose anchors on both sides within 0.2 seconds')
    anchors = [max(before, key=lambda row: row['frame']), min(after, key=lambda row: row['frame'])]
    bounds = []
    for row in anchors:
        if len(row['landmarks']) != 33:
            raise ValueError('Crop anchors require complete model landmark arrays')
        points = np.asarray([[p['x']*width, p['y']*height] for p in row['landmarks']
                             if min(p['visibility'], p['presence']) >= .5
                             and 0 <= p['x'] <= 1 and 0 <= p['y'] <= 1])
        if len(points) < CONTRACT['minimum_anchor_visible_points'] or not np.isfinite(points).all():
            raise ValueError('Temporal crop lacks source-supported anchor points')
        bounds.append([points.min(axis=0), points.max(axis=0)])
    bounds = np.asarray(bounds)
    sizes = bounds[:, 1]-bounds[:, 0]
    if np.min(sizes) <= 0 or np.max(np.max(sizes, axis=0)/np.min(sizes, axis=0)) > CONTRACT['maximum_anchor_box_ratio']:
        raise ValueError('Temporal anchor boxes change size too much')
    travel = np.linalg.norm(bounds[1].mean(axis=0)-bounds[0].mean(axis=0))
    if travel/np.linalg.norm(sizes.mean(axis=0)) > CONTRACT['maximum_anchor_center_travel_ratio']:
        raise ValueError('Temporal anchor boxes move too far for a supported crop')
    low, high = bounds[:, 0].min(axis=0), bounds[:, 1].max(axis=0)
    padding = (high-low)*CONTRACT['crop_padding_fraction']
    low = np.maximum([0, 0], np.floor(low-padding)).astype(int)
    high = np.minimum([width, height], np.ceil(high+padding)).astype(int)
    if np.any(high-low < CONTRACT['minimum_crop_side_px']):
        raise ValueError('Temporal crop is smaller than the supported model image')
    if np.prod(high-low)/(width*height) > CONTRACT['maximum_crop_area_fraction']:
        raise ValueError('Temporal crop does not isolate a bounded image region')
    return {'frame': frame, 'box_xyxy': [*low.tolist(), *high.tolist()],
        'padding_fraction': CONTRACT['crop_padding_fraction'],
        'anchor_frames': [row['frame'] for row in anchors],
        'anchor_hashes': [canonical_hash(row) for row in anchors],
        'maximum_anchor_gap_seconds': CONTRACT['maximum_anchor_gap_seconds']}


def map_crop_pose(result, plan, evidence, original, elapsed):
    left, top, right, bottom = plan['box_xyxy']
    width, height = evidence['width'], evidence['height']
    if (not 0 <= left < right <= width or not 0 <= top < bottom <= height
            or original['frame'] != plan['frame'] or not math.isfinite(elapsed) or elapsed < 0):
        raise ValueError('Crop transform must match its exact source frame')
    mapped = deepcopy(result)
    if result.get('pose_count') == 1 and len(result.get('landmarks', [])) == 33:
        for point in mapped['landmarks']:
            point['x'] = (left+point['x']*(right-left))/width
            point['y'] = (top+point['y']*(bottom-top))/height
            # Normalized image depth scales with width; hip-relative metric coordinates do not.
            point['z'] *= (right-left)/width
        mapped = summarize_landmarks(mapped['landmarks'], mapped['world_landmarks'])
    else:
        # Competing crop hypotheses stay crop-relative in the saved raw result, never camera inputs.
        mapped = {key: mapped[key] for key in ('status', 'pose_count', 'landmarks', 'world_landmarks')}
    mapped.update({key: original[key] for key in ('id', 'frame', 'time_s', 'decoder_timestamp_s')})
    mapped['inference_seconds'] = elapsed
    return mapped


def plan_reinference(views, baseline):
    if (len(views) != 2 or any(view['evidence'].get('pose_reinference') for view in views)
            or any(view['evidence'].get('pose_identity', {}).get('mode') != 'raw' for view in views)):
        raise ValueError('Re-inference requires two original, unrepaired pose views')
    if any(row.get('identity_raw') or row.get('image_refinement') for view in views for row in view['evidence']['frames']):
        raise ValueError('Re-inference cannot reuse repaired or previously refined observations')
    if any(canonical_hash(view['evidence']) != binding['evidence_sha256']
           or canonical_hash(view['camera']) != binding['camera_sha256']
           for view, binding in zip(views, baseline['views'], strict=True)):
        raise ValueError('Original triangulation must match both camera/evidence inputs')
    indexed = [{row['frame']: row for row in view['evidence']['frames']} for view in views]
    targets = [row for row in baseline['frames']
               if any(row['joints'][j]['status'] == 'inconsistent_image_rays' for j in (23, 24))]
    if len(targets) > CONTRACT['maximum_target_frames']:
        raise ValueError('Too many hip-ray failures for bounded local re-inference; check calibration and source alignment')
    plans = []
    supported = [row for row in baseline['frames'] if all(row['joints'][j]['status'] == 'estimated' for j in (23, 24))]
    for target in targets:
        frame = target['frame']
        row = {'frame': frame, 'time_s': target['time_s'], 'status': 'planned', 'crops': None, 'root_anchors': None}
        try:
            if any(frame not in view or view[frame].get('pose_count') != 1 for view in indexed):
                raise ValueError('Both original views must have a single hypothesis at the target frame')
            row['crops'] = [plan_crop(view['evidence'], frame) for view in views]
            before = [r for r in supported if 0 < target['time_s']-r['time_s'] <= CONTRACT['maximum_anchor_gap_seconds']]
            after = [r for r in supported if 0 < r['time_s']-target['time_s'] <= CONTRACT['maximum_anchor_gap_seconds']]
            if not before or not after:
                raise ValueError('Original two-view root support is unavailable on both sides')
            row['root_anchors'] = [max(before, key=lambda r: r['frame'])['frame'], min(after, key=lambda r: r['frame'])['frame']]
        except ValueError as error:
            row.update(status='unavailable_context', reason=str(error))
        plans.append(row)
    return plans


def assess_reinference(plan, candidates, baseline, proposal):
    if plan['status'] != 'planned':
        raise ValueError('Only a frozen supported crop plan can be assessed')
    raw, revised = baseline['frames'][plan['frame']], proposal['frames'][plan['frame']]
    reasons = []
    if any(row.get('pose_count') != 1 or row['status'] not in ('tracked', 'tracked_partial') for row in candidates):
        reasons.append('crop_pose_unusable_or_competing')
    if any(revised['joints'][j]['status'] != 'estimated' for j in (11, 12, 23, 24)):
        reasons.append('torso_or_hip_rays_unsupported')
    count = sum(revised['joints'][j]['status'] == 'estimated' for j in BODY_JOINTS)
    if count < CONTRACT['minimum_estimated_body_joints']:
        reasons.append('insufficient_supported_body_joints')
    lost = [j for j in BODY_JOINTS if raw['joints'][j]['status'] == 'estimated' and revised['joints'][j]['status'] != 'estimated']
    if lost:
        reasons.append('previous_image_support_lost')
    deviation = None
    if all(revised['joints'][j]['status'] == 'estimated' for j in (11, 12, 23, 24)):
        anchors = [baseline['frames'][f] for f in plan['root_anchors']]
        roots = [np.mean([row['joints'][j]['world_point_m'] for j in (23, 24)], axis=0) for row in anchors]
        fraction = (raw['time_s']-anchors[0]['time_s'])/(anchors[1]['time_s']-anchors[0]['time_s'])
        expected = roots[0]*(1-fraction)+roots[1]*fraction
        root = np.mean([revised['joints'][j]['world_point_m'] for j in (23, 24)], axis=0)
        shoulders = np.mean([revised['joints'][j]['world_point_m'] for j in (11, 12)], axis=0)
        torso = float(np.linalg.norm(shoulders-root))
        deviation = float(np.linalg.norm(root-expected)/max(torso, 1e-9))
        if torso < .06 or deviation > CONTRACT['maximum_root_deviation_torso_ratio']:
            reasons.append('root_outside_original_temporal_support')
    return {'frame': plan['frame'], 'accepted': not reasons, 'reasons': reasons,
            'estimated_body_joints': count, 'previously_supported_joints_lost': lost,
            'root_deviation_torso_ratio': deviation, 'anatomical_identity_verified': False}


def apply_reinference(evidence, plans, proposals, decisions, view_index):
    if view_index not in (0, 1) or evidence.get('pose_reinference'):
        raise ValueError('Choose an original view; repeated crop refinement is unsupported')
    result = deepcopy(evidence)
    indexed = {row['frame']: i for i, row in enumerate(result['frames'])}
    planned = {row['frame']: row for row in plans}
    if len({row['frame'] for row in decisions}) != len(decisions):
        raise ValueError('Every target may have only one decision')
    changed = []
    for decision in decisions:
        frame = decision['frame']
        if not decision['accepted']:
            continue
        if frame not in planned or frame not in proposals or frame not in indexed or planned[frame]['status'] != 'planned':
            raise ValueError('Accepted inference must belong to the frozen plan and original source')
        old, new = result['frames'][indexed[frame]], deepcopy(proposals[frame])
        if (any(old[key] != new[key] for key in ('id', 'frame', 'time_s', 'decoder_timestamp_s'))
                or new.get('pose_count') != 1 or new['status'] not in ('tracked', 'tracked_partial')):
            raise ValueError('Accepted crop must preserve frame identities/clocks and usable single-pose geometry')
        new['image_refinement'] = {'method': METHOD, 'original_observation_sha256': canonical_hash(old),
            'proposal_sha256': canonical_hash(new), 'decision_sha256': canonical_hash(decision),
            'crop': planned[frame]['crops'][view_index], 'anatomical_identity_verified': False}
        new['image_refinement_raw'] = old
        result['frames'][indexed[frame]] = new
        changed.append(frame)
    result['pose_reinference'] = {'method': METHOD, 'contract': dict(CONTRACT),
        'input_evidence_sha256': canonical_hash(evidence), 'changed_frames': changed,
        'source_frame_count_unchanged': True, 'missing_samples_filled': False}
    result.setdefault('assumptions', []).append(
        'Explicit temporal crops rerun the image model on original pixels; raw observations and rejected proposals are retained. '
        'No left/right permutation, interpolation, independently verified anatomy or muscle activity is inferred.')
    return result
