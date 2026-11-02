"""Partial two-view body geometry with explicit single-view priors for missing limbs."""

from collections import Counter
from copy import deepcopy
import math
import time
from types import MappingProxyType

import numpy as np

from weft.io import canonical_hash
from weft.limb_position_fit import CONTRACT as LIMB_FIT_CONTRACT, STATUSES as LIMB_FIT_STATUSES, fit_limb_positions
from weft.multiview_root import METHOD as ROOT_METHOD, MULTI_METHOD as MULTI_ROOT_METHOD
from weft.multiview_segments import METHOD as SEGMENT_METHOD, CONTRACT as SEGMENT_CONTRACT, COMPONENTS, PARTIAL_STATUS
from weft.multiview_segments import (MULTI_METHOD as MULTI_SEGMENT_METHOD, MULTI_CONTRACT as MULTI_SEGMENT_CONTRACT,
                                    MULTI_COMPONENTS, MULTI_PARTIAL_STATUS)
from weft.multiview_distal_pixels import (METHOD as PARTIAL_PIXEL_METHOD, CONTRACT as PARTIAL_PIXEL_CONTRACT,
    pixel_status, pixel_components, prepare_pixel_views, apply_distal_pixels, validate_pixel_binding)
from weft.multiview_distal_pixels import (UNCERTAINTY_METHOD as PARTIAL_UNCERTAINTY_METHOD,
    UNCERTAINTY_CONTRACT as PARTIAL_UNCERTAINTY_CONTRACT, validate_uncertainty)
from weft.multiview_depth_branches import (METHOD as DEPTH_BRANCH_METHOD, CONTRACT as DEPTH_BRANCH_CONTRACT,
    apply_depth_branches, validate_depth_binding)
from weft.spatial_pose import LIMB_ENDPOINTS, SPATIAL_JOINT_NAMES, _limb_rotation, rotation_to_zyx, rotation_zyx


METHOD = 'two-view-body-v1'
MULTI_METHOD = 'multi-view-body-v1'
LIMB_FIT_METHOD = 'camera-body-length-fit-v1'
LIMB_FIT_SEGMENT_METHOD = 'camera-body-length-fit-v2'
METHODS = (METHOD, SEGMENT_METHOD, MULTI_METHOD, LIMB_FIT_METHOD, MULTI_SEGMENT_METHOD, LIMB_FIT_SEGMENT_METHOD,
           PARTIAL_PIXEL_METHOD, PARTIAL_UNCERTAINTY_METHOD, DEPTH_BRANCH_METHOD)
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
    method = spec['camera_reference']['method']
    branches = method == DEPTH_BRANCH_METHOD
    uncertain = method == PARTIAL_UNCERTAINTY_METHOD or branches
    pixels = method == PARTIAL_PIXEL_METHOD or uncertain
    partial = method in (SEGMENT_METHOD, MULTI_SEGMENT_METHOD, LIMB_FIT_SEGMENT_METHOD) or pixels
    fitted = method in (LIMB_FIT_METHOD, LIMB_FIT_SEGMENT_METHOD) or pixels
    multi = (method in (MULTI_METHOD, MULTI_SEGMENT_METHOD) or fitted
             and len(spec['camera_reference'].get('root_refinement', {}).get('views', [])) > 2)
    partial_status = MULTI_PARTIAL_STATUS if multi else PARTIAL_STATUS
    partial_statuses = {partial_status, pixel_status(multi)} if pixels else {partial_status}
    if partial:
        fields.add('segment_contract')
    if fitted:
        fields.add('limb_fit_contract')
    if pixels:
        fields.update(('partial_pixel_contract', 'partial_pixel_camera_sha256'))
    if uncertain:
        fields.update(('partial_pixel_uncertainty', 'partial_pixel_uncertainty_sha256'))
    if branches:
        fields.update(('depth_branch_contract', 'depth_branch_reference_sha256'))
    if (not isinstance(value, dict) or set(value) != fields or type(value['schema_version']) is not int
            or value['schema_version'] != 1 or value['mode'] not in MODES
            or not isinstance(value['input_body'], dict)
            or not isinstance(value['input_body'].get('camera_reference'), dict)
            or value['input_body'].get('camera_reference', {}).get('method') != (MULTI_ROOT_METHOD if multi else ROOT_METHOD)):
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
    if fitted or partial:
        for meta in spec['reference_evidence']:
            sources = meta.get('geometry_sources')
            allowed = {'multi_view_estimate' if multi else 'two_view_estimate',
                       'monocular_prior_missing_points', 'monocular_prior_degenerate_geometry'}
            if partial:
                allowed.update(partial_statuses)
            if (not isinstance(sources, dict) or set(sources) != set(GROUPS)
                    or any(not isinstance(v, str) or v not in allowed for v in sources.values())):
                raise ValueError('Geometry sources differ from the retained view method')
    if fitted:
        if value['mode'] not in ('directions', 'both') or value['limb_fit_contract'] != dict(LIMB_FIT_CONTRACT):
            raise ValueError('Length-fit body requires its fixed direction-fitting contract')
        for meta, reference in zip(spec['reference_evidence'], spec['reference'], strict=True):
            sources = meta['geometry_sources']
            fits = meta.get('geometry_fit')
            expected = {name for name, source in sources.items()
                        if name in LIMB_ENDPOINTS and source in ('two_view_estimate', 'multi_view_estimate')}
            if meta.get('geometry_method') != method or not isinstance(fits, dict) or set(fits) != expected:
                raise ValueError('Length-fit provenance must retain every applied supported limb')
            for group, fit in fits.items():
                names = (group+'_twist', group+'_lateral', group,
                         group.rsplit('_', 1)[0]+('_knee' if group.endswith('hip') else '_elbow'))
                required = {'input_angles_rad', 'angle_correction_rad', 'status', 'position_sse_before_m2', 'position_sse_after_m2'}
                if not isinstance(fit, dict) or set(fit) != required:
                    raise ValueError('Length-fit provenance differs from the bounded correction schema')
                old, delta = fit['input_angles_rad'], fit['angle_correction_rad']
                costs = [fit['position_sse_before_m2'], fit['position_sse_after_m2']]
                if (not isinstance(old, list) or not isinstance(delta, list) or len(old) != 4 or len(delta) != 4
                        or any(type(x) not in (int, float) or not math.isfinite(x) for x in old+delta+costs)
                        or any(abs(x) > math.tau for x in old) or not 0 <= old[3] <= 2.6
                        or any(abs(x) > LIMB_FIT_CONTRACT['maximum_angle_change_rad'] for x in delta)
                        or not isinstance(fit['status'], str) or fit['status'] not in LIMB_FIT_STATUSES
                        or not 0 <= costs[1] <= costs[0]
                        or any(abs(reference['joints'][name]-a-d) > 1e-12 for name, a, d in zip(names, old, delta, strict=True))):
                    raise ValueError('Length-fit angles or source-position nonregression changed')
    if multi and value['mode'] in ('directions', 'both'):
        view_count = len(spec['camera_reference']['root_refinement']['views'])
        for meta in spec['reference_evidence']:
            sources, contributors = meta.get('geometry_sources'), meta.get('geometry_view_indices')
            if (meta.get('geometry_method') != method
                    or not isinstance(sources, dict) or set(sources) != set(GROUPS)
                    or not isinstance(contributors, dict)
                    or set(contributors) != {name for name, source in sources.items()
                                             if source == 'multi_view_estimate' or partial and source in partial_statuses}):
                raise ValueError('Multi-view body must identify the contributors to every applied geometry group')
            for group, joints in contributors.items():
                required = GROUPS[group][:2] if partial and sources[group] in partial_statuses else GROUPS[group]
                if not isinstance(joints, dict) or set(joints) != {str(i) for i in required}:
                    raise ValueError('Multi-view group contributor joints differ from the geometry contract')
                for indices in joints.values():
                    if (not isinstance(indices, list) or not 2 <= len(indices) <= view_count
                            or any(type(i) is not int or not 0 <= i < view_count for i in indices)
                            or indices != sorted(set(indices))):
                        raise ValueError('Multi-view body joints require at least two distinct source views')
    if partial:
        contract = MULTI_SEGMENT_CONTRACT if multi else SEGMENT_CONTRACT
        components = MULTI_COMPONENTS if multi else COMPONENTS
        if value['segment_contract'] != dict(contract) or value['mode'] not in ('directions', 'both'):
            raise ValueError('Partial segments require their explicit direction/uncertainty contract')
        for meta, angles, prior in zip(spec['reference_evidence'], spec['reference'], parent['reference'], strict=True):
            if meta.get('geometry_method') != method:
                raise ValueError('Partial-segment provenance must identify its version')
            partial_groups = {name for name, source in meta['geometry_sources'].items() if source in partial_statuses}
            if set(meta.get('geometry_components', {})) != partial_groups or not partial_groups <= set(LIMB_ENDPOINTS):
                raise ValueError('Partial segments must explicitly identify their unobserved components')
            for group in partial_groups:
                distal = group.rsplit('_', 1)[0]+('_knee' if group.endswith('hip') else '_elbow')
                image_constrained = pixels and meta['geometry_sources'][group] == pixel_status(multi)
                if (meta['geometry_components'][group] != (pixel_components(multi) if image_constrained else components)
                        or not image_constrained and angles['joints'][distal] != prior['joints'][distal]
                        or any(meta['joint_confidence'].get(group+suffix) != 0. for suffix in ('_twist', '_lateral', ''))):
                    raise ValueError('Partial directions cannot promote unknown twist/bend or change the distal prior')
    if pixels:
        branch_inputs = validate_depth_binding(spec) if branches else None
        validate_pixel_binding(spec, parent, multi=multi, branch_inputs=branch_inputs)


def _confidence(frame, indices, sources, prediction=None):
    values = []
    for index in indices:
        contributors = prediction['joints'][index]['contributing_view_indices'] if prediction is not None else range(len(sources))
        if (not 2 <= len(contributors) <= len(sources) or len(set(contributors)) != len(contributors)
                or any(type(i) is not int or not 0 <= i < len(sources) for i in contributors)):
            raise ValueError('Estimated geometry needs at least two valid original image contributors')
        for contributor in contributors:
            pose = sources[contributor].get(frame)
            if pose is None or pose.get('pose_count') != 1:
                raise ValueError('Estimated geometry requires every declared original image observation')
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


def make_two_view_body(spec, predictions, first_evidence, second_evidence, *, mode='both', partial_segments=False,
                      additional_evidence=(), limb_fit=False, partial_pixel_fit=False, pixel_cameras=(), pixel_uncertainty=None,
                      partial_depth_branches=False):
    from weft.human import LENGTH_BOUNDS, validate_human

    original = validate_human(spec)
    binding = original.get('camera_reference', {})
    if not isinstance(additional_evidence, (list, tuple)) or len(additional_evidence) > 2:
        raise ValueError('At most two additional body evidence streams are supported')
    multi = bool(additional_evidence)
    label = 'Multi-view' if multi else 'Two-view'
    estimate_status = 'multi_view_estimate' if multi else 'two_view_estimate'
    dimension_status = 'multi_view_symmetric_estimate' if multi else 'two_view_symmetric_estimate'
    if mode not in MODES or binding.get('method') != (MULTI_ROOT_METHOD if multi else ROOT_METHOD):
        raise ValueError('Two-view body requires an original root-only candidate and explicit geometry mode')
    if type(partial_segments) is not bool or partial_segments and mode not in ('directions', 'both'):
        raise ValueError('Partial segments require an explicit boolean and direction-enabled mode')
    if type(limb_fit) is not bool or limb_fit and mode not in ('directions', 'both'):
        raise ValueError('Fixed-length limb fitting requires directions/both')
    if (type(partial_pixel_fit) is not bool or partial_pixel_fit and not (limb_fit and partial_segments)
            or not partial_pixel_fit and (pixel_cameras or pixel_uncertainty is not None)):
        raise ValueError('Partial pixel fitting requires explicit length fitting, partial segments and original cameras')
    uncertainty = validate_uncertainty(pixel_uncertainty) if pixel_uncertainty is not None else None
    if type(partial_depth_branches) is not bool or partial_depth_branches and not (partial_pixel_fit and uncertainty is not None):
        raise ValueError('Depth branches require the explicit uncertainty-qualified partial pixel fit')
    method = (DEPTH_BRANCH_METHOD if partial_depth_branches else PARTIAL_UNCERTAINTY_METHOD if uncertainty is not None else PARTIAL_PIXEL_METHOD if partial_pixel_fit else LIMB_FIT_SEGMENT_METHOD if limb_fit and partial_segments else LIMB_FIT_METHOD if limb_fit else
              MULTI_SEGMENT_METHOD if multi and partial_segments else MULTI_METHOD if multi else
              SEGMENT_METHOD if partial_segments else METHOD)
    segment_contract = MULTI_SEGMENT_CONTRACT if multi else SEGMENT_CONTRACT
    if canonical_hash(predictions) != binding['root_refinement']['triangulation_sha256']:
        raise ValueError('Two-view body predictions differ from the root construction')
    views = predictions['views']
    evidence_streams = [first_evidence, second_evidence, *additional_evidence]
    if [canonical_hash(v) for v in evidence_streams] != [v['evidence_sha256'] for v in views]:
        raise ValueError('Two-view body image inputs differ from the pinned reconstruction')
    pixel_views = prepare_pixel_views(pixel_cameras, evidence_streams, predictions) if partial_pixel_fit else None
    sources = [{row['frame']: row for row in evidence['frames']} for evidence in evidence_streams]
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
                                'image_confidence': _confidence(frame, (*start, *end), sources, predictions['frames'][frame] if multi else None)})
        count = len({row['frame'] for row in samples})
        value = float(np.median([row['length_m'] for row in samples])) if samples else None
        status = ('monocular_prior_insufficient_frames' if count < CONTRACT['minimum_dimension_frames'] else
                  'monocular_prior_outside_length_bounds' if not LENGTH_BOUNDS[name][0] <= value <= LENGTH_BOUNDS[name][1]
                  else dimension_status)
        applied = mode in ('dimensions', 'both') and status == dimension_status
        if applied: candidate['lengths'][name] = value
        dimensions[name] = {'status': status, 'applied': applied, 'original_length': original['lengths'][name],
                            'output_length': candidate['lengths'][name], 'median_observed_length': value,
                            'distinct_frames': count, 'samples': samples}
    audits, previous, previous_fit = [], {}, {}
    fit_deadline = time.monotonic()+LIMB_FIT_CONTRACT['maximum_body_fit_seconds'] if limb_fit else None
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
                detail['image_confidence'] = _confidence(frame, required, sources, pose if multi else None)
                if multi:
                    detail['joint_view_indices'] = {str(j): deepcopy(pose['joints'][j]['contributing_view_indices']) for j in required}
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
                    detail.update(status=estimate_status, assumed_twist=assumed, chart_singular=singular,
                                  flexion_clamped=flexion is not None and flexion > CONTRACT['maximum_flexion_rad'],
                                  chart_wrap=bool(np.max(np.abs(angles-prior)) > math.pi))
                    if limb_fit and group != 'torso':
                        if time.monotonic() >= fit_deadline:
                            raise TimeoutError('Fixed-length body fitting exceeded its declared deadline')
                        leg = group.endswith('hip'); sign = -1 if group.startswith('left_') else 1
                        lengths = candidate['lengths']
                        local_origin = [sign*lengths['hip_width' if leg else 'shoulder_width']/2, 0., 0. if leg else lengths['torso']]
                        root = np.asarray(binding['origin_world_m'])+original['root_reference'][i]['translation']
                        parent_targets = (points[1:]-root-torso @ local_origin) @ torso
                        earlier = previous_fit.get(group)
                        correction = (earlier[1] if earlier and 0 < meta['time_s']-earlier[0] <= LIMB_FIT_CONTRACT['maximum_temporal_gap_s'] else None)
                        fit = fit_limb_positions(values, [lengths[key] for key in (('thigh', 'shin') if leg else ('upper_arm', 'forearm'))],
                                                 parent_targets, -1 if leg else 1, previous=correction)
                        detail['length_fit'] = fit
                        values = fit['output_angles_rad']
                        previous_fit[group] = (meta['time_s'], fit['angle_correction_rad'])
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
            if limb_fit and group != 'torso' and 'length_fit' not in detail:
                previous_fit.pop(group, None)
            # Fitting must not select a different Euler branch for the next raw source pose.
            previous[group] = np.array(detail['length_fit']['input_angles_rad'][:3] if limb_fit and 'length_fit' in detail
                                       else [joints[name] for name in names[:3]])
            groups[group] = detail
        if mode in ('directions', 'both'):
            meta['geometry_sources'] = {group: detail['status'] for group, detail in groups.items()}
            meta['geometry_method'] = LIMB_FIT_METHOD if limb_fit else MULTI_METHOD if multi else METHOD
            if limb_fit:
                meta['geometry_fit'] = {group: {key: detail['length_fit'][key] for key in
                    ('input_angles_rad', 'angle_correction_rad', 'status', 'position_sse_before_m2', 'position_sse_after_m2')}
                    for group, detail in groups.items() if 'length_fit' in detail}
            if multi:
                meta['geometry_view_indices'] = {group: detail['joint_view_indices'] for group, detail in groups.items() if detail['applied']}
        audits.append({'frame': frame, 'evidence_id': meta['evidence_id'], 'time_s': meta['time_s'], 'groups': groups})
    segment_report = None
    if partial_segments:
        from weft.multiview_segments import apply_partial_segments
        segment_report = apply_partial_segments(candidate, original, predictions, sources, audits, multi=multi, method=method)
    pixel_report = (apply_distal_pixels(candidate, original, predictions, sources, audits, pixel_views, multi=multi, uncertainty=uncertainty)
                    if partial_pixel_fit else None)
    depth_report = apply_depth_branches(candidate, audits, pixel_views) if partial_depth_branches else None
    by_frame = {row['frame']: row for row in audits}
    slots = [{'frame': row['frame'], 'time_s': row['time_s'], 'retained_body_pose': row['frame'] in by_frame,
              'groups': by_frame[row['frame']]['groups'] if row['frame'] in by_frame else None,
              'source_evidence_ids': row['source_evidence_ids']} for row in predictions['frames']]
    audit = {'schema_version': 1, 'contract': dict(CONTRACT) | {'method': method}, 'mode': mode,
             'input_body_sha256': canonical_hash(original), 'triangulation_sha256': canonical_hash(predictions),
             'dimensions': dimensions, 'frames': slots,
             'group_coverage': {group: dict(Counter(row['groups'][group]['status'] for row in audits)) for group in GROUPS},
             'scope': f'Partial {label.lower()} estimated geometry; missing local limb angles retain explicit monocular priors',
             'root_and_ground_unchanged': True, 'physical_validation': 'unverified', 'paid_requests': 0}
    if segment_report is not None:
        audit['contract'] = dict(CONTRACT) | {'method': method, 'partial_segments': dict(segment_contract)}
        audit['partial_segments'] = segment_report
        audit['scope'] = f'Partial {label.lower()} geometry; missing upper directions, distal bends and axial twist retain explicit monocular priors'
    if limb_fit:
        audit['contract']['limb_fit'] = dict(LIMB_FIT_CONTRACT)
        audit['scope'] = 'Fixed-length source-position angle proposals; no root, floor or dynamic fit'
        fits = [detail['length_fit'] for row in audits for detail in row['groups'].values() if 'length_fit' in detail]
        audit['limb_position_fit'] = {'method': LIMB_FIT_CONTRACT['method'], 'supported_limbs': len(fits),
            'changed_limbs': sum(fit['maximum_angle_change_rad'] > 0 for fit in fits),
            'status_counts': dict(Counter(fit['status'] for fit in fits)),
            'maximum_angle_change_rad': max((fit['maximum_angle_change_rad'] for fit in fits), default=0.)}
    if pixel_report is not None:
        audit['contract']['partial_pixels'] = dict(PARTIAL_UNCERTAINTY_CONTRACT if uncertainty is not None else PARTIAL_PIXEL_CONTRACT)
        audit['partial_pixels'] = pixel_report
        audit['scope'] = 'Fixed complete-limb position fits and pixel-constrained distal priors; no measured depth or contact'
    if depth_report is not None:
        audit['contract']['depth_branches'] = dict(DEPTH_BRANCH_CONTRACT)
        audit['depth_branches'] = depth_report
        audit['scope'] = 'Temporal selection among retained image-ray depth hypotheses; not observed depth or verified contact'
    candidate['camera_reference'].update(method=method, scale_assumption=SCALE_ASSUMPTION,
        angular_reference_sha256=canonical_hash(candidate['reference']), body_refinement={
            'schema_version': 1, 'mode': mode, 'input_body': original, 'input_body_sha256': canonical_hash(original),
            'triangulation_sha256': canonical_hash(predictions), 'audit_sha256': canonical_hash(audit),
            'output_lengths_sha256': canonical_hash(candidate['lengths']),
            'output_evidence_sha256': canonical_hash(candidate['reference_evidence'])})
    if partial_segments:
        candidate['camera_reference']['body_refinement']['segment_contract'] = dict(segment_contract)
    if limb_fit:
        candidate['camera_reference']['body_refinement']['limb_fit_contract'] = dict(LIMB_FIT_CONTRACT)
    if partial_pixel_fit:
        candidate['camera_reference']['body_refinement'].update(partial_pixel_contract=dict(
            PARTIAL_UNCERTAINTY_CONTRACT if uncertainty is not None else PARTIAL_PIXEL_CONTRACT),
            partial_pixel_camera_sha256=[v['camera_sha256'] for v in views])
    if uncertainty is not None:
        candidate['camera_reference']['body_refinement'].update(partial_pixel_uncertainty=uncertainty,
            partial_pixel_uncertainty_sha256=canonical_hash(uncertainty))
    if depth_report is not None:
        candidate['camera_reference']['body_refinement'].update(depth_branch_contract=dict(DEPTH_BRANCH_CONTRACT),
            depth_branch_reference_sha256=depth_report['before_branch_reference_sha256'])
    candidate['assumptions'] = [v for v in candidate['assumptions'] if not v.startswith(
        'Body shape, joint angles, muscle illustrations and motor forces remain')]
    candidate['assumptions'].extend([
        f'{label} body geometry mode: {mode}; supplied camera scale and alignment remain uncertain',
        (f'Available {label.lower()} geometry and qualified distal images constrain explicit priors; missing depth remains inferred' if partial_pixel_fit else
         f'Only available {label.lower()} geometry is applied; missing local limb angles remain original monocular priors'),
        'Shared left/right dimensions are an authored symmetry assumption, including limbs hidden in the images',
        'Root, ground, motor limits and support gains are unchanged; dimension changes also alter authored mass/inertia',
        'Muscle bands remain illustrative motor-command proxies, not measured anatomy or muscle activity',
    ])
    if partial_segments:
        candidate['assumptions'].append(
            'Supported upper-segment directions transport prior limb orientation; distal bend and axial twist start as monocular priors, not measurements' if partial_pixel_fit else
            'Supported upper-segment directions transport prior limb orientation; missing distal bend and axial twist remain monocular, not measured')
    if limb_fit:
        candidate['assumptions'].append('Supported limb angles fit source-derived positions with fixed lengths; corrections and temporal priors are inferred, not measured anatomy or contact')
    if partial_pixel_fit:
        candidate['assumptions'].append('Available distal images constrain partial-limb bend/twist without changing the supported upper axis; missing 3D endpoints remain inferred, not observed')
    if uncertainty is not None:
        candidate['assumptions'].append('Image and angle error scales are explicit assumptions, not calibrated noise or depth confidence; pose confidence supplies heuristic weights')
    if depth_report is not None:
        candidate['assumptions'].append('Depth branches use explicit smooth-motion assumptions between complete source-supported limbs; both depths and rejected alternatives remain saved, not observed depth or contact')
    candidate['reconstruction_coverage']['two_view_body'] = {'method': method, 'mode': mode, 'group_coverage': audit['group_coverage']}
    if limb_fit:
        candidate['reconstruction_coverage']['two_view_body']['limb_position_fit'] = deepcopy(audit['limb_position_fit'])
    if partial_pixel_fit:
        candidate['reconstruction_coverage']['two_view_body']['partial_pixels'] = deepcopy(pixel_report)
    if depth_report is not None:
        candidate['reconstruction_coverage']['two_view_body']['depth_branches'] = deepcopy(depth_report)
    return validate_human(candidate), audit
