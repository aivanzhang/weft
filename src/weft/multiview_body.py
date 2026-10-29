"""Partial two-view body geometry with explicit single-view priors for missing limbs."""

from collections import Counter
from copy import deepcopy
import math
from types import MappingProxyType

import numpy as np

from weft.io import canonical_hash
from weft.multiview_root import METHOD as ROOT_METHOD
from weft.multiview_segments import METHOD as SEGMENT_METHOD, CONTRACT as SEGMENT_CONTRACT, COMPONENTS, PARTIAL_STATUS
from weft.spatial_pose import LIMB_ENDPOINTS, SPATIAL_JOINT_NAMES, _limb_rotation, rotation_to_zyx, rotation_zyx


METHOD = 'two-view-body-v1'
METHODS = (METHOD, SEGMENT_METHOD)
SCALE_ASSUMPTION = 'supplied_camera_scale_with_explicit_monocular_body_priors'
CONTRACT = MappingProxyType({'method': METHOD, 'minimum_dimension_frames': 5,
    'maximum_reference_poses': 900, 'dimensions': 'Median available segment lengths; shared left/right lengths remain an authored symmetry assumption',
    'directions': 'Torso and complete three-point limbs only; missing local angles remain explicit monocular priors',
    'floor_fitting': False, 'root_fitting': False, 'maximum_flexion_rad': 2.6})
MODES = ('directions', 'dimensions', 'both')
GROUPS = {'torso': (11, 12, 23, 24), **LIMB_ENDPOINTS}
DIMENSIONS = {'torso': (((11, 12), (23, 24)),), 'shoulder_width': (((11,), (12,)),),
              'hip_width': (((23,), (24,)),),
              'upper_arm': (((11,), (13,)), ((12,), (14,))),
              'forearm': (((13,), (15,)), ((14,), (16,))),
              'thigh': (((23,), (25,)), ((24,), (26,))),
              'shin': (((25,), (27,)), ((26,), (28,)))}


def _fixed_hash(spec):
    fields = ('schema_version', 'profile', 'source_sha256', 'dt', 'controller', 'body_frame',
              'mobility', 'root_reference', 'motor_limits')
    binding = {key: value for key, value in spec['camera_reference'].items()
               if key not in ('method', 'scale_assumption', 'angular_reference_sha256', 'body_refinement')}
    return canonical_hash({**{key: spec.get(key) for key in fields}, 'camera_reference': binding})


def validate_body_binding(spec):
    from weft.human import validate_human

    value = spec['camera_reference'].get('body_refinement')
    fields = {'schema_version', 'mode', 'input_body', 'input_body_sha256', 'triangulation_sha256',
              'output_lengths_sha256', 'output_evidence_sha256', 'audit_sha256'}
    partial = spec['camera_reference']['method'] == SEGMENT_METHOD
    if partial:
        fields.add('segment_contract')
    if (not isinstance(value, dict) or set(value) != fields or type(value['schema_version']) is not int
            or value['schema_version'] != 1 or value['mode'] not in MODES
            or not isinstance(value['input_body'], dict)
            or not isinstance(value['input_body'].get('camera_reference'), dict)
            or value['input_body'].get('camera_reference', {}).get('method') != ROOT_METHOD):
        raise ValueError('Two-view body needs one original root-only parent and an explicit geometry mode')
    # Only a root-only parent is allowed, preventing recursive candidate chains.
    parent = validate_human(value['input_body'])
    if (canonical_hash(parent) != value['input_body_sha256'] or _fixed_hash(parent) != _fixed_hash(spec)
            or value['triangulation_sha256'] != spec['camera_reference']['root_refinement']['triangulation_sha256']
            or canonical_hash(spec['lengths']) != value['output_lengths_sha256']
            or canonical_hash(spec['reference_evidence']) != value['output_evidence_sha256']
            or not isinstance(value['audit_sha256'], str) or len(value['audit_sha256']) != 64
            or any(c not in '0123456789abcdef' for c in value['audit_sha256'])):
        raise ValueError('Two-view body changed its pinned parent, geometry, root, source or forces')
    if value['mode'] == 'directions' and spec['lengths'] != parent['lengths']:
        raise ValueError('Direction-only body must preserve original dimensions')
    if value['mode'] == 'dimensions' and (spec['reference'] != parent['reference']
                                        or spec['reference_evidence'] != parent['reference_evidence']):
        raise ValueError('Dimension-only body must preserve original angles and their provenance')
    if partial:
        if value['segment_contract'] != dict(SEGMENT_CONTRACT) or value['mode'] not in ('directions', 'both'):
            raise ValueError('Partial segments require their explicit direction/uncertainty contract')
        for meta, angles, prior in zip(spec['reference_evidence'], spec['reference'], parent['reference'], strict=True):
            if meta.get('geometry_method') != SEGMENT_METHOD:
                raise ValueError('Partial-segment provenance must identify its version')
            partial_groups = {name for name, source in meta.get('geometry_sources', {}).items() if source == PARTIAL_STATUS}
            if set(meta.get('geometry_components', {})) != partial_groups or not partial_groups <= set(LIMB_ENDPOINTS):
                raise ValueError('Partial segments must explicitly identify their unobserved components')
            for group in partial_groups:
                distal = group.rsplit('_', 1)[0]+('_knee' if group.endswith('hip') else '_elbow')
                if (meta['geometry_components'][group] != COMPONENTS
                        or angles['joints'][distal] != prior['joints'][distal]
                        or any(meta['joint_confidence'].get(group+suffix) != 0. for suffix in ('_twist', '_lateral', ''))):
                    raise ValueError('Partial directions cannot promote unknown twist/bend or change the distal prior')


def _confidence(frame, indices, sources):
    values = []
    for poses in sources:
        pose = poses.get(frame)
        if pose is None or pose.get('pose_count') != 1:
            raise ValueError('Estimated two-view geometry requires both original image observations')
        for index in indices:
            point = pose['landmarks'][index]
            values.extend(point[key] for key in ('visibility', 'presence'))
    if any(type(v) not in (int, float) or not math.isfinite(v) or not .5 <= v <= 1 for v in values):
        raise ValueError('Two-view geometry confidence differs from its retained image observations')
    return min(values)


def _points(row, indices):
    if any(row['joints'][index]['status'] != 'estimated' for index in indices):
        return None
    value = np.asarray([row['joints'][index]['world_point_m'] for index in indices], dtype=float)
    if value.shape != (len(indices), 3) or not np.isfinite(value).all() or np.abs(value).max() > 2000:
        raise ValueError('Two-view body geometry exceeds the finite camera-world envelope')
    return value


def _torso_rotation(points):
    up = (points[0]+points[1]-points[2]-points[3])/2
    length = np.linalg.norm(up)
    if length < .03: raise ValueError('Degenerate torso height')
    up = up/length
    right = points[1]-points[0]
    right -= (right @ up)*up
    length = np.linalg.norm(right)
    if length < .03: raise ValueError('Degenerate torso width')
    right = right/length
    return np.column_stack((right, np.cross(up, right), up))


def make_two_view_body(spec, predictions, first_evidence, second_evidence, *, mode='both', partial_segments=False):
    from weft.human import LENGTH_BOUNDS, validate_human

    original = validate_human(spec)
    binding = original.get('camera_reference', {})
    if mode not in MODES or binding.get('method') != ROOT_METHOD:
        raise ValueError('Two-view body requires an original root-only candidate and explicit geometry mode')
    if type(partial_segments) is not bool or partial_segments and mode not in ('directions', 'both'):
        raise ValueError('Partial segments require an explicit boolean and direction-enabled mode')
    method = SEGMENT_METHOD if partial_segments else METHOD
    if canonical_hash(predictions) != binding['root_refinement']['triangulation_sha256']:
        raise ValueError('Two-view body predictions differ from the root construction')
    views = predictions['views']
    if [canonical_hash(v) for v in (first_evidence, second_evidence)] != [v['evidence_sha256'] for v in views]:
        raise ValueError('Two-view body image inputs differ from the pinned reconstruction')
    sources = [{row['frame']: row for row in evidence['frames']} for evidence in (first_evidence, second_evidence)]
    indices = [int(row['evidence_id'].rsplit(':', 1)[1]) for row in original['reference_evidence']]
    if not 2 <= len(indices) <= CONTRACT['maximum_reference_poses']:
        raise ValueError('Two-view body reference count exceeds its bound')
    candidate = deepcopy(original)
    dimensions = {}
    for name, pairs in DIMENSIONS.items():
        samples = []
        for frame in indices:
            for start, end in pairs:
                points = _points(predictions['frames'][frame], (*start, *end))
                if points is None: continue
                distance = float(np.linalg.norm(points[:len(start)].mean(axis=0)-points[len(start):].mean(axis=0)))
                samples.append({'frame': frame, 'indices': [list(start), list(end)], 'length_m': distance,
                                'image_confidence': _confidence(frame, (*start, *end), sources)})
        count = len({row['frame'] for row in samples})
        value = float(np.median([row['length_m'] for row in samples])) if samples else None
        status = ('monocular_prior_insufficient_frames' if count < CONTRACT['minimum_dimension_frames'] else
                  'monocular_prior_outside_length_bounds' if not LENGTH_BOUNDS[name][0] <= value <= LENGTH_BOUNDS[name][1]
                  else 'two_view_symmetric_estimate')
        applied = mode in ('dimensions', 'both') and status == 'two_view_symmetric_estimate'
        if applied: candidate['lengths'][name] = value
        dimensions[name] = {'status': status, 'applied': applied, 'original_length': original['lengths'][name],
                            'output_length': candidate['lengths'][name], 'median_observed_length': value,
                            'distinct_frames': count, 'samples': samples}
    audits, previous = [], {}
    anchor = np.asarray(original['body_frame']['rotation'])
    for i, frame in enumerate(indices):
        original_joints = original['reference'][i]['joints']
        joints = candidate['reference'][i]['joints']
        pose = predictions['frames'][frame]
        meta = candidate['reference_evidence'][i]
        groups = {}
        torso = anchor @ rotation_zyx([original_joints[name] for name in SPATIAL_JOINT_NAMES[:3]])
        for group, required in GROUPS.items():
            names = (SPATIAL_JOINT_NAMES[:3] if group == 'torso' else
                     (group+'_twist', group+'_lateral', group,
                      group.rsplit('_', 1)[0]+('_knee' if group.endswith('hip') else '_elbow')))
            points = _points(pose, required)
            detail = {'indices': list(required), 'status': 'monocular_prior_missing_points',
                      'point_statuses': [pose['joints'][j]['status'] for j in required],
                      'applied': False, 'image_confidence': None, 'assumed_twist': False,
                      'chart_singular': False, 'flexion_clamped': False, 'chart_wrap': False}
            if points is not None:
                detail['image_confidence'] = _confidence(frame, required, sources)
                try:
                    if group == 'torso':
                        target_torso = _torso_rotation(points)
                        rotation = anchor.T @ target_torso
                        flexion, assumed = None, False
                    else:
                        rotation, flexion, assumed = _limb_rotation(torso.T @ (points[1]-points[0]),
                            torso.T @ (points[2]-points[1]), -1 if group.endswith('hip') else 1)
                    prior = previous.get(group, [original_joints[name] for name in names[:3]])
                    angles, singular = rotation_to_zyx(rotation, prior)
                    values = angles.tolist()+([] if group == 'torso' else [min(flexion, CONTRACT['maximum_flexion_rad'])])
                    detail.update(status='two_view_estimate', assumed_twist=assumed, chart_singular=singular,
                                  flexion_clamped=flexion is not None and flexion > CONTRACT['maximum_flexion_rad'],
                                  chart_wrap=bool(np.max(np.abs(angles-prior)) > math.pi))
                    if mode in ('directions', 'both'):
                        detail['applied'] = True
                        for name, value in zip(names, values, strict=True):
                            joints[name] = value
                            meta['joint_confidence'][name] = detail['image_confidence']
                        if assumed: meta['joint_confidence'][names[0]] = 0.
                        if singular: meta['joint_confidence'][names[0]] = meta['joint_confidence'][names[2]] = 0.
                        if group == 'torso': torso = target_torso
                except ValueError as error:
                    detail.update(status='monocular_prior_degenerate_geometry', reason=str(error))
            previous[group] = np.array([joints[name] for name in names[:3]])
            groups[group] = detail
        if mode in ('directions', 'both'):
            meta['geometry_sources'] = {group: detail['status'] for group, detail in groups.items()}
            meta['geometry_method'] = METHOD
        audits.append({'frame': frame, 'evidence_id': meta['evidence_id'], 'time_s': meta['time_s'], 'groups': groups})
    segment_report = None
    if partial_segments:
        from weft.multiview_segments import apply_partial_segments
        segment_report = apply_partial_segments(candidate, original, predictions, sources, audits)
    by_frame = {row['frame']: row for row in audits}
    slots = [{'frame': row['frame'], 'time_s': row['time_s'], 'retained_body_pose': row['frame'] in by_frame,
              'groups': by_frame[row['frame']]['groups'] if row['frame'] in by_frame else None,
              'source_evidence_ids': row['source_evidence_ids']} for row in predictions['frames']]
    audit = {'schema_version': 1, 'contract': dict(CONTRACT), 'mode': mode,
             'input_body_sha256': canonical_hash(original), 'triangulation_sha256': canonical_hash(predictions),
             'dimensions': dimensions, 'frames': slots,
             'group_coverage': {group: dict(Counter(row['groups'][group]['status'] for row in audits)) for group in GROUPS},
             'scope': 'Partial two-view estimated geometry; missing local limb angles retain explicit monocular priors',
             'root_and_ground_unchanged': True, 'physical_validation': 'unverified', 'paid_requests': 0}
    if segment_report is not None:
        audit['contract'] = dict(CONTRACT) | {'method': method, 'partial_segments': dict(SEGMENT_CONTRACT)}
        audit['partial_segments'] = segment_report
        audit['scope'] = 'Partial two-view geometry; missing upper directions, distal bends and axial twist retain explicit monocular priors'
    candidate['camera_reference'].update(method=method, scale_assumption=SCALE_ASSUMPTION,
        angular_reference_sha256=canonical_hash(candidate['reference']), body_refinement={
            'schema_version': 1, 'mode': mode, 'input_body': original, 'input_body_sha256': canonical_hash(original),
            'triangulation_sha256': canonical_hash(predictions), 'audit_sha256': canonical_hash(audit),
            'output_lengths_sha256': canonical_hash(candidate['lengths']),
            'output_evidence_sha256': canonical_hash(candidate['reference_evidence'])})
    if partial_segments:
        candidate['camera_reference']['body_refinement']['segment_contract'] = dict(SEGMENT_CONTRACT)
    candidate['assumptions'] = [v for v in candidate['assumptions'] if not v.startswith(
        'Body shape, joint angles, muscle illustrations and motor forces remain')]
    candidate['assumptions'].extend([
        f'Two-view body geometry mode: {mode}; supplied camera scale and alignment remain uncertain',
        'Only available two-view geometry is applied; missing local limb angles remain original monocular priors',
        'Shared left/right dimensions are an authored symmetry assumption, including limbs hidden in the images',
        'Root, ground, motor limits and support gains are unchanged; dimension changes also alter authored mass/inertia',
        'Muscle bands remain illustrative motor-command proxies, not measured anatomy or muscle activity',
    ])
    if partial_segments:
        candidate['assumptions'].append(
            'Supported upper-segment directions transport prior limb orientation; missing distal bend and axial twist remain monocular, not measured')
    candidate['reconstruction_coverage']['two_view_body'] = {'method': method, 'mode': mode, 'group_coverage': audit['group_coverage']}
    return validate_human(candidate), audit
