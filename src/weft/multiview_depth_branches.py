"""Bind depth alternatives to unchanged image fits and source-supported temporal anchors."""

from bisect import bisect_left
from collections import Counter
from copy import deepcopy
import math
import time

import numpy as np

from weft.distal_depth_branches import CONTRACT, propose_depth_branches
from weft.io import canonical_hash
from weft.spatial_pose import LIMB_ENDPOINTS, SPATIAL_JOINT_NAMES, rotation_to_zyx, rotation_zyx


METHOD = 'camera-body-depth-branches-v1'


def _names(group):
    return (group+'_twist', group+'_lateral', group,
            group.rsplit('_', 1)[0]+('_knee' if group.endswith('hip') else '_elbow'))


def _direction(spec, index, group, angles=None):
    joints = spec['reference'][index]['joints']
    angles = [joints[k] for k in _names(group)] if angles is None else angles
    torso = np.array(spec['body_frame']['rotation']) @ rotation_zyx([joints[k] for k in SPATIAL_JOINT_NAMES[:3]])
    sign = -1 if group.endswith('hip') else 1
    return torso @ rotation_zyx(angles[:3]) @ [0., sign*math.sin(angles[3]), -math.cos(angles[3])]


def _anchors(spec):
    result = {}
    for group in LIMB_ENDPOINTS:
        rows = [{'reference_index': i, 'evidence_id': meta['evidence_id'], 'time_s': meta['time_s'],
                 'direction': _direction(spec, i, group).tolist()}
                for i, meta in enumerate(spec['reference_evidence'])
                if meta['geometry_sources'][group] in ('two_view_estimate', 'multi_view_estimate') and group in meta['geometry_fit']]
        result[group] = ([row['time_s'] for row in rows], rows)
    return result


def _temporal(anchors, group, stamp):
    times, rows = anchors[group]; cursor = bisect_left(times, stamp)
    context = {'status': 'missing_bracketing_anchors', 'anchors': [], 'direction': None}
    if cursor == 0 or cursor == len(rows): return context
    a, b = rows[cursor-1:cursor+1]; context['anchors'] = [a, b]
    if not 0 < stamp-a['time_s'] <= .25 or not 0 < b['time_s']-stamp <= .25:
        context['status'] = 'anchor_gap'; return context
    first, second = np.array(a['direction']), np.array(b['direction'])
    if float(first @ second) < 0:
        context['status'] = 'inconsistent_anchor_directions'; return context
    alpha = (stamp-a['time_s'])/(b['time_s']-a['time_s'])
    direction = (1-alpha)*first+alpha*second; direction /= np.linalg.norm(direction)
    context.update(status='smooth_motion_assumption', direction=direction.tolist())
    return context


def apply_depth_branches(candidate, audits, prepared):
    before = canonical_hash(candidate['reference']); anchors = _anchors(candidate)
    deadline = time.monotonic()+CONTRACT['maximum_body_seconds']; results = []
    for i, row in enumerate(audits):
        meta = candidate['reference_evidence'][i]; meta['geometry_method'] = METHOD
        meta['geometry_depth_branches'] = {}; joints = candidate['reference'][i]['joints']
        for group, detail in row['groups'].items():
            attempt = detail.get('distal_pixel_attempt')
            if not attempt or not attempt['applied']: continue
            if time.monotonic() >= deadline: raise TimeoutError('Depth branch construction exceeded its declared deadline')
            context = _temporal(anchors, group, meta['time_s'])
            views = []
            for item in attempt['views']:
                matrix, radial, transforms, width, height = prepared[item['view_index']]
                views.append({'matrix': matrix, 'radial': radial, 'rotation': item['solver_rotation'],
                    'translation': transforms[row['frame']][1], 'width': width, 'height': height})
            seed = attempt['fit']; names = _names(group); old = [joints[k] for k in names]
            fit = propose_depth_branches(seed['output_rotation_world'], seed['output_bend_rad'],
                seed['fixed_distal_length_m'], seed['fixed_knee_world_m'], views,
                seed['source_pixels'], seed['source_confidence'], seed['bend_sign'], anchor_direction=context['direction'])
            detail['depth_branch_attempt'] = {'temporal': context, 'source_view_indices': attempt['view_indices'],
                                             'input_angles_rad': old, 'fit': fit}
            results.append(fit)
            if fit['selected_index'] is None: continue
            selected = fit['candidates'][fit['selected_index']]; delta = selected['correction_rad']
            angles, singular = rotation_to_zyx(rotation_zyx(old[:3]) @ rotation_zyx([delta[0], 0., 0.]), old[:3])
            for name, value in zip(names, angles.tolist()+[selected['output_bend_rad']], strict=True):
                joints[name] = value; meta['joint_confidence'][name] = 0.
            detail['depth_branch_attempt']['chart_singular'] = singular
            meta['geometry_depth_branches'][group] = {'input_angles_rad': old, 'correction_rad': delta,
                'selected_candidate_index': fit['selected_index'], 'pixel_sse_before': fit['pixel_sse_before'],
                'pixel_sse_after': selected['pixel_sse'], 'temporal_distance_before': fit['temporal_distance_before'],
                'temporal_distance_after': selected['temporal_distance'], 'runner_up_gap_lengths': fit['runner_up_gap_lengths'],
                'anchor_reference_indices': [a['reference_index'] for a in context['anchors']]}
    return {'method': CONTRACT['method'], 'before_branch_reference_sha256': before,
        'attempted_group_poses': len(results), 'selected_group_poses': sum(r['selected_index'] is not None for r in results),
        'candidate_branches': sum(len(r['candidates']) for r in results), 'status_counts': dict(Counter(r['status'] for r in results)),
        'scope': CONTRACT['scope'], 'depth_measured': False}


def validate_depth_binding(spec):
    binding = spec['camera_reference']['body_refinement']
    summary = spec['reconstruction_coverage']['two_view_body'].get('depth_branches', {})
    if binding['depth_branch_contract'] != dict(CONTRACT) or summary.get('depth_measured') is not False:
        raise ValueError('Depth branches require their fixed inferred-only contract')
    before = deepcopy(spec['reference']); anchors = _anchors(spec); inputs = {}; selected = 0
    for i, (meta, reference) in enumerate(zip(spec['reference_evidence'], spec['reference'], strict=True)):
        records = meta.get('geometry_depth_branches')
        if (not isinstance(records, dict) or not isinstance(meta.get('geometry_partial_pixels'), dict)
                or not set(records) <= set(meta['geometry_partial_pixels'])):
            raise ValueError('Depth branches must identify only retained image-constrained limbs')
        inputs[i] = {}
        for group, record in records.items():
            fields = {'input_angles_rad', 'correction_rad', 'selected_candidate_index', 'pixel_sse_before',
                'pixel_sse_after', 'temporal_distance_before', 'temporal_distance_after', 'runner_up_gap_lengths',
                'anchor_reference_indices'}
            if not isinstance(record, dict) or set(record) != fields:
                raise ValueError('Depth-branch provenance differs from its compact schema')
            old, delta = record['input_angles_rad'], record['correction_rad']
            scalars = [record[k] for k in ('pixel_sse_before', 'pixel_sse_after', 'temporal_distance_before', 'temporal_distance_after')]
            gap = record['runner_up_gap_lengths']; ids = record['anchor_reference_indices']
            if (not isinstance(old, list) or len(old) != 4 or not isinstance(delta, list) or len(delta) != 2
                    or any(type(v) not in (int, float) or not math.isfinite(v) for v in old+delta+scalars)
                    or any(abs(v) > math.tau for v in old) or not 0 <= old[3] <= 2.6 or abs(delta[0]) > math.pi
                    or not 0 <= old[3]+delta[1] <= 2.6
                    or type(record['selected_candidate_index']) is not int or not 0 <= record['selected_candidate_index'] < 8
                    or not 0 <= scalars[1] < scalars[0]-CONTRACT['minimum_pixel_improvement']
                    or not 0 <= scalars[3] <= .5 or not scalars[3] < scalars[2]-CONTRACT['minimum_temporal_improvement']
                    or gap is not None and (type(gap) not in (int, float) or not math.isfinite(gap) or not .25 <= gap <= 2.)
                    or not isinstance(ids, list) or len(ids) != 2 or any(type(v) is not int for v in ids)):
                raise ValueError('Depth branches require bounded geometry, image improvement and unambiguous temporal support')
            context = _temporal(anchors, group, meta['time_s']); names = _names(group)
            if (context['direction'] is None or ids != [a['reference_index'] for a in context['anchors']]
                    or not np.allclose(rotation_zyx([reference['joints'][k] for k in names[:3]]),
                                       rotation_zyx(old[:3]) @ rotation_zyx([delta[0], 0., 0.]), atol=1e-10, rtol=0)
                    or abs(reference['joints'][names[-1]]-old[3]-delta[1]) > 1e-12
                    or any(meta['joint_confidence'].get(k) != 0. for k in names)):
                raise ValueError('Depth branch changed its upper axis, source anchors or inferred-only status')
            for angles, value in ((old, scalars[2]), (None, scalars[3])):
                distance = float(np.linalg.norm(_direction(spec, i, group, angles)-context['direction']))
                if abs(distance-value) > 1e-9: raise ValueError('Depth branch temporal distance differs from the retained source anchors')
            for name, value in zip(names, old, strict=True): before[i]['joints'][name] = value
            inputs[i][group] = old; selected += 1
    if (canonical_hash(before) != binding['depth_branch_reference_sha256']
            or summary.get('before_branch_reference_sha256') != binding['depth_branch_reference_sha256']
            or summary.get('selected_group_poses') != selected):
        raise ValueError('Depth branches changed their complete pre-branch reference or selected count')
    return inputs
