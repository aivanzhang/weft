"""Short bilateral label islands require original temporal and parent-joint support."""

from collections import Counter, defaultdict
from copy import deepcopy
import math
from types import MappingProxyType

import numpy as np

from weft.io import canonical_hash
from weft.multiview_pose import _observation, _triangulate_rays, _view, triangulate_pair


METHOD = 'two-view-parent-temporal-correspondence-v1'
CONTRACT = MappingProxyType({
    'method': METHOD, 'minimum_confidence': .8, 'maximum_span_samples': 3,
    'maximum_span_seconds': .2, 'maximum_adjacent_gap_seconds': .12,
    'minimum_segment_length_m': 1e-5, 'minimum_pair_separation_ratio': .15,
    'maximum_anchor_length_variation': .2, 'maximum_anchor_motion_per_second': 8.,
    'maximum_corrected_error_ratio': .15, 'minimum_error_improvement_ratio': .2,
    'maximum_corrected_to_raw_error': .35, 'minimum_alternative_margin_ratio': .1,
    'maximum_reprojection_increase_px': 1.,
    'anchors': 'Two original same-label samples on each side; never repaired anchors or interpolated observations',
    'dependency_policy': 'Proximal candidates take precedence; dependent distal candidates abstain without reusing corrected parents',
    'identity': 'Side labels remain inherited from unverified parent joints and local anchors, not independently identified anatomy',
    'scope': 'Explicit correspondence hypothesis; no reference-body, third-view, floor, muscle or monocular-depth fitting',
})
PAIRS = {'elbows': ((11, 12), (13, 14)), 'wrists': ((13, 14), (15, 16)),
         'knees': ((23, 24), (25, 26)), 'ankles': ((25, 26), (27, 28))}
PERMUTATIONS = ((0, 0), (0, 1), (1, 0), (1, 1))


def _sample(frame, parent, child, prepared, baseline):
    parents = [baseline['frames'][frame]['joints'][index] for index in parent]
    if any(row['status'] != 'estimated' for row in parents):
        return None, 'parent_geometry_unavailable'
    pixels, cameras = [], [view[1][frame] for view in prepared]
    for view, camera in zip(prepared, cameras, strict=True):
        pose = view[0].get(frame)
        row = []
        for index in (*parent, *child):
            point, status = _observation(pose, index, camera[4], camera[5])
            if point is None:
                return None, status
            confidence = min(pose['landmarks'][index][key] for key in ('visibility', 'presence'))
            if confidence < CONTRACT['minimum_confidence']:
                return None, 'insufficient_correspondence_confidence'
            row.append(point)
        pixels.append(row[2:])
    parents = np.asarray([row['world_point_m'] for row in parents])
    raw = [baseline['frames'][frame]['joints'][index] for index in child]
    cross = [_triangulate_rays([pixels[0][side], pixels[1][side ^ 1]], cameras) for side in (0, 1)]
    alternatives = []
    for estimates in (raw, cross, cross[::-1], raw[::-1]):
        if any(row['status'] != 'estimated' for row in estimates):
            alternatives.append(None)
            continue
        points = np.asarray([row['world_point_m'] for row in estimates])
        lengths = np.linalg.norm(points-parents, axis=1)
        if min(lengths) < CONTRACT['minimum_segment_length_m']:
            alternatives.append(None)
            continue
        alternatives.append({'points': points, 'vectors': points-parents, 'lengths': lengths,
            'estimates': estimates, 'reprojection': np.asarray([row['reprojection_px'] for row in estimates]),
            'separation': float(np.linalg.norm(points[0]-points[1]))})
    if all(row is None for row in alternatives):
        return None, 'no_supported_ray_assignment'
    return {'alternatives': alternatives}, 'available'


def _window(samples, times, first, last):
    support = (first-2, first-1, last+1, last+2)
    window = list(range(first-2, last+3))
    if any(samples[i] is None for i in window):
        return 'unavailable_context_sample'
    if (times[last]-times[first] > CONTRACT['maximum_span_seconds']
            or any(times[b]-times[a] > CONTRACT['maximum_adjacent_gap_seconds'] for a, b in zip(window, window[1:]))):
        return 'temporal_gap_or_span'
    anchors = [samples[i]['alternatives'][0] for i in support]
    if any(row is None for row in anchors):
        return 'original_anchor_assignment_unavailable'
    lengths = np.asarray([row['lengths'] for row in anchors])
    scale = np.median(lengths, axis=0)
    if np.any(np.ptp(lengths, axis=0)/scale > CONTRACT['maximum_anchor_length_variation']):
        return 'anchor_length_variation'
    if any(row['separation']/float(scale.mean()) < CONTRACT['minimum_pair_separation_ratio'] for row in anchors):
        return 'anchor_pair_separation'
    for a, b in ((0, 1), (2, 3)):
        speed = np.linalg.norm(anchors[b]['vectors']-anchors[a]['vectors'], axis=1)/scale/(times[support[b]]-times[support[a]])
        if max(speed) > CONTRACT['maximum_anchor_motion_per_second']:
            return 'anchor_motion'
    rows = []
    for index in range(first, last+1):
        fraction = (times[index]-times[first-1])/(times[last+1]-times[first-1])
        expected = (1-fraction)*anchors[1]['vectors']+fraction*anchors[2]['vectors']
        errors = []
        for alternative in samples[index]['alternatives']:
            if alternative is None or alternative['separation']/float(scale.mean()) < CONTRACT['minimum_pair_separation_ratio']:
                errors.append(None)
            else:
                errors.append(np.linalg.norm(alternative['vectors']-expected, axis=1)/scale)
        if errors[0] is None:
            # Without a comparable raw estimate there is no paired temporal-error test.
            return 'raw_assignment_unavailable_or_pair_overlap'
        rows.append(errors)
    scores = [None if any(row[k] is None for row in rows) else float(np.mean([np.mean(row[k]**2) for row in rows])**.5)
              for k in range(4)]
    valid = sorted((score, k) for k, score in enumerate(scores) if score is not None)
    if len(valid) < 2:
        return 'no_comparable_alternative'
    if valid[0][1] == 0:
        return 'raw_assignment_preferred'
    best_score, best = valid[0]
    raw_score, second_score = scores[0], valid[1][0]
    if raw_score-best_score < CONTRACT['minimum_error_improvement_ratio']/2:
        return 'insufficient_candidate_improvement'
    reasons = []
    if second_score-best_score < CONTRACT['minimum_alternative_margin_ratio']:
        reasons.append('multiple_temporally_plausible_assignments')
    for index, errors in zip(range(first, last+1), rows, strict=True):
        raw = float(np.sqrt(np.mean(errors[0]**2)))
        corrected = float(np.sqrt(np.mean(errors[best]**2)))
        if (max(errors[best]) > CONTRACT['maximum_corrected_error_ratio']
                or raw-corrected < CONTRACT['minimum_error_improvement_ratio']
                or corrected > CONTRACT['maximum_corrected_to_raw_error']*raw):
            reasons.append('insufficient_parent_relative_continuity_margin')
            break
        alternatives = samples[index]['alternatives']
        if np.max(alternatives[best]['reprojection']-alternatives[0]['reprojection']) > CONTRACT['maximum_reprojection_increase_px']:
            reasons.append('image_ray_agreement_worsens')
            break
    return {'first': first, 'last': last, 'support': support, 'permutation': best,
        'scores': scores, 'scale': scale.tolist(), 'reasons': reasons,
        'per_sample_errors': [[None if error is None else error.tolist() for error in row] for row in rows]}


def triangulate_with_limb_correspondence(first, second, *, return_baseline=False):
    """Return separately audited predictions; neither original view is modified."""
    if type(return_baseline) is not bool:
        raise ValueError('return_baseline must be boolean')
    baseline = triangulate_pair(first, second)
    prepared = [_view(view) for view in (first, second)]
    if any('identity_raw' in pose or pose.get('identity_recompute_required')
           for view in prepared for pose in view[0].values()):
        raise ValueError('Correspondence requires original image labels, not previously repaired evidence')
    sampled = sorted(set(prepared[0][0]) | set(prepared[1][0]))
    times = [frame/first['camera']['fps'] for frame in sampled]
    result, decisions, availability, candidates = deepcopy(baseline), [], {}, []
    cache, rejected_windows = {}, {}
    for name, (parent, child) in PAIRS.items():
        values = [_sample(frame, parent, child, prepared, baseline) for frame in sampled]
        samples = [row[0] for row in values]
        cache[name] = samples
        rejected_windows[name] = Counter()
        availability[name] = {frame: status for frame, (_, status) in zip(sampled, values, strict=True)}
        for start in range(2, len(sampled)-2):
            for size in range(1, CONTRACT['maximum_span_samples']+1):
                stop = start+size-1
                if stop+2 >= len(sampled):
                    break
                value = _window(samples, times, start, stop)
                if isinstance(value, str):
                    rejected_windows[name][value] += 1
                    continue
                decision = {'pair': name, 'parent_indices': list(parent), 'joint_indices': list(child),
                    'frames': sampled[start:stop+1], 'support_frames': [sampled[i] for i in value['support']],
                    'permutation': list(PERMUTATIONS[value['permutation']]), 'scores': value['scores'],
                    'normalization_lengths_m': value['scale'], 'per_sample_errors': value['per_sample_errors'],
                    'status': 'abstained' if value['reasons'] else 'supported_candidate', 'reasons': value['reasons']}
                decisions.append(decision)
                if not value['reasons']:
                    candidates.append((name, value, decision))
    blocked, writers, readers = set(), defaultdict(set), defaultdict(set)
    # A child's correction cannot certify its own parent or either temporal anchor.
    for a, (name, value, _) in enumerate(candidates):
        joints = set(PAIRS[name][1])
        frames = set(sampled[value['first']:value['last']+1])
        dependencies = {(sampled[i], j) for i in range(value['first']-2, value['last']+3)
                        for j in (*PAIRS[name][0], *PAIRS[name][1]) if i < value['first'] or i > value['last'] or j in PAIRS[name][0]}
        cells = {(f, j) for f in frames for j in joints}
        conflicts = set().union(*(writers[cell] | readers[cell] for cell in cells),
                                *(writers[cell] for cell in dependencies))
        if conflicts:
            # A wrong knee label can falsely implicate a correct ankle; do not reject the hip-supported correction.
            blocked.update({a} | {index for index in conflicts if candidates[index][0] == name})
        for cell in cells:
            writers[cell].add(a)
        for cell in dependencies:
            readers[cell].add(a)
    changed = []
    for index, (name, value, decision) in enumerate(candidates):
        if index in blocked:
            decision.update(status='abstained', reasons=['conflicting_candidate_or_parent_anchor_identity'])
            continue
        decision.update(status='applied_hypothesis', reasons=['short_island_with_original_parent_and_temporal_support'])
        child = PAIRS[name][1]
        permutation = PERMUTATIONS[value['permutation']]
        for i in range(value['first'], value['last']+1):
            frame = sampled[i]
            selected = cache[name][i]['alternatives'][value['permutation']]
            for side, joint in enumerate(child):
                result['frames'][frame]['joints'][joint] = deepcopy(selected['estimates'][side]) | {
                    'observation_statuses': ['available', 'available'],
                    'source_joint_indices': [child[side ^ p] for p in permutation],
                    'correspondence_status': 'parent_temporal_hypothesis'}
            changed.append({'frame': frame, 'pair': name, 'joint_indices': list(child), 'permutation': list(permutation)})
    counts = Counter(joint['status'] for frame in result['frames'] for joint in frame['joints'])
    result['joint_status_counts'] = dict(counts)
    audit = {'schema_version': 1, 'contract': dict(CONTRACT), 'raw_predictions_sha256': canonical_hash(baseline),
        'views': baseline['views'], 'decisions': decisions, 'changed_pairs': changed,
        'window_rejection_counts': {name: dict(counts) for name, counts in rejected_windows.items()},
        'source_frames': baseline['source_frames'], 'joint_slots': baseline['joint_slots'],
        'frames': [{'frame': f, 'time_s': f/first['camera']['fps'],
                    'pairs': {name: availability[name].get(f, 'pose_not_sampled') for name in PAIRS}}
                   for f in range(baseline['source_frames'])],
        'anatomical_side_verified': False, 'missing_observations_filled': False,
        'third_view_or_reference_used': False, 'monocular_world_landmarks_used': False,
        'scope': CONTRACT['scope']}
    result['correspondence'] = {'contract': dict(CONTRACT), 'audit_sha256': canonical_hash(audit),
        'raw_predictions_sha256': canonical_hash(baseline), 'changed_pair_count': len(changed),
        'anatomical_side_verified': False}
    return (result, audit, baseline) if return_baseline else (result, audit)
