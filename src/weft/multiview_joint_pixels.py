"""Bind time-coupled joint-image fits to the unchanged camera/body construction."""

from copy import deepcopy
import math
import time

import numpy as np

from weft.io import canonical_hash
from weft.joint_pixel_trajectory import CONTRACT, STATUSES, fit_joint_trajectory, trajectory_contract, validate_uncertainty
from weft.multiview_distal_pixels import OBSERVATION_STATUSES
from weft.multiview_pose import _observation
from weft.spatial_pose import LIMB_ENDPOINTS, SPATIAL_JOINT_NAMES, rotation_zyx


METHOD = 'camera-body-joint-pixels-v1'
INPUT_METHOD = 'camera-body-length-fit-v2'


def _names(group):
    return (group+'_twist', group+'_lateral', group,
            group.rsplit('_', 1)[0]+('_knee' if group.endswith('hip') else '_elbow'))


def refine_joint_pixels(body, audit, predictions, sources, prepared, uncertainty):
    settings = validate_uncertainty(uncertainty)
    contract = trajectory_contract(settings)
    if body['camera_reference']['method'] != INPUT_METHOD:
        raise ValueError('Joint-image fitting requires the original fixed-length/partial-direction constructor')
    candidate, result = deepcopy(body), deepcopy(audit)
    binding = candidate['camera_reference']['body_refinement']
    pin = {'contract': dict(contract), 'uncertainty': settings, 'input_method': INPUT_METHOD,
           'input_reference_sha256': canonical_hash(body['reference']),
           'input_evidence_sha256': binding['output_evidence_sha256'], 'input_audit_sha256': binding['audit_sha256']}
    frames = [int(row['evidence_id'].rsplit(':', 1)[1]) for row in body['reference_evidence']]
    times = [row['time_s'] for row in body['reference_evidence']]
    parents = [np.array(body['body_frame']['rotation']) @ rotation_zyx([row['joints'][k] for k in SPATIAL_JOINT_NAMES[:3]])
               for row in body['reference']]
    roots = np.array(body['camera_reference']['origin_world_m'])+np.array([r['translation'] for r in body['root_reference']])
    cameras = []; deadline = time.monotonic()+CONTRACT['maximum_body_seconds']
    for frame in frames:
        views = []
        for matrix, radial, transforms, width, height in prepared:
            rotation, translation = transforms[frame]; u, _, vt = np.linalg.svd(rotation); proper = u @ vt
            change = float(np.max(abs(proper-rotation)))
            if np.linalg.det(proper) <= 0 or change > CONTRACT['rotation_roundoff_limit']:
                raise ValueError('Joint-image camera preparation exceeds original rounding precision')
            views.append(({'matrix': matrix, 'radial': radial, 'rotation': proper,
                           'translation': translation, 'width': width, 'height': height}, change))
        cameras.append(views)
    for meta in candidate['reference_evidence']:
        meta['geometry_method'] = METHOD; meta['geometry_joint_pixels'] = {}
    trajectories = {}; constraints = changed = weak = 0
    for group, indices in LIMB_ENDPOINTS.items():
        if time.monotonic() >= deadline: raise TimeoutError('Joint-image body fitting exceeded its declared deadline')
        leg = group.endswith('hip'); side = -1 if group.startswith('left_') else 1; sizes = body['lengths']; names = _names(group)
        origin = [root+parent @ [side*sizes['hip_width' if leg else 'shoulder_width']/2, 0., 0. if leg else sizes['torso']]
                  for root, parent in zip(roots, parents, strict=True)]
        angles = [[row['joints'][k] for k in names] for row in body['reference']]
        observations = []; attempts = []
        for i, frame in enumerate(frames):
            entries = []; detail = {'observation_statuses': {}, 'view_indices': {}, 'views': []}
            for point, source_index in enumerate(indices[1:]):
                observed = [_observation(source.get(frame), source_index, prepared[v][3], prepared[v][4]) for v, source in enumerate(sources)]
                statuses = [value[1] for value in observed]
                if statuses != predictions['frames'][frame]['joints'][source_index]['observation_statuses']:
                    raise ValueError('Joint image slots changed after original triangulation')
                detail['observation_statuses'][str(source_index)] = statuses
                detail['view_indices'][str(source_index)] = [v for v, item in enumerate(observed) if item[0] is not None]
                for view in detail['view_indices'][str(source_index)]:
                    camera, rounding = cameras[i][view]; source = sources[view][frame]
                    landmark = source['landmarks'][source_index]; confidence = min(landmark['visibility'], landmark['presence'])
                    pixel = observed[view][0].tolist()
                    entries.append({'camera': camera, 'view_index': view, 'joint_index': point, 'pixel': pixel, 'confidence': confidence})
                    detail['views'].append({'view_index': view, 'joint_index': point, 'source_landmark_index': source_index,
                        'source_evidence_id': source['id'], 'camera_sha256': predictions['views'][view]['camera_sha256'],
                        'pixel': pixel, 'confidence': confidence, 'solver_rotation': camera['rotation'].tolist(),
                        'maximum_rotation_element_change': rounding})
            observations.append(entries); attempts.append(detail); constraints += len(entries)
            result['frames'][frame]['groups'][group]['joint_pixel_attempt'] = detail
        try:
            fit = fit_joint_trajectory(angles, [sizes[k] for k in (('thigh', 'shin') if leg else ('upper_arm', 'forearm'))],
                                      origin, parents, times, observations, -1 if leg else 1, settings)
            trajectories[group] = {'status': 'fitted', 'fit': fit}
        except ValueError as error:
            fit = None; trajectories[group] = {'status': 'unavailable_projection', 'reason': str(error), 'fit': None}
        if time.monotonic() >= deadline: raise TimeoutError('Joint-image body fitting exceeded its declared deadline')
        for i, (meta, detail) in enumerate(zip(candidate['reference_evidence'], attempts, strict=True)):
            delta = fit['correction_rad'][i] if fit else [0.]*4
            old_confidence = [meta['joint_confidence'][k] for k in names]
            altered = any(value != 0. for value in delta); changed += altered
            weak += bool(fit and fit['weak_image_directions'][i] > 0 and observations[i])
            record = {'input_angles_rad': angles[i], 'correction_rad': delta, 'input_joint_confidence': old_confidence,
                      'status': fit['status'] if fit else 'unavailable_projection',
                      'view_indices': detail['view_indices'], 'observation_statuses': detail['observation_statuses']}
            meta['geometry_joint_pixels'][group] = record
            for name, old, correction in zip(names, angles[i], delta, strict=True):
                candidate['reference'][i]['joints'][name] = old+correction
                if altered: meta['joint_confidence'][name] = 0.
    fitted = [row['fit'] for row in trajectories.values() if row['fit'] is not None]
    summary = {'method': contract['method'], 'fitted_trajectories': len(fitted), 'unavailable_trajectories': 4-len(fitted),
        'changed_group_poses': changed, 'source_image_constraints': constraints, 'weak_image_group_poses': weak,
        'maximum_angle_change_rad': max((f['maximum_angle_change_rad'] for f in fitted), default=0.),
        'image_chi2_before': sum(f['before']['image_chi2'] for f in fitted),
        'image_chi2_after': sum(f['after']['image_chi2'] for f in fitted),
        'uncertainty': settings, 'uncertainty_sha256': canonical_hash(settings), 'uncertainty_calibrated': False,
        'depth_measured': False, 'scope': CONTRACT['scope']}
    if settings['schema_version'] == 2:
        summary.update({f'motion_acceleration_chi2_{key}': sum(f[key]['motion_acceleration_chi2'] for f in fitted)
                        for key in ('before', 'after')})
    result['joint_pixel_trajectories'] = trajectories; result['joint_pixels'] = summary
    result['contract']['method'] = METHOD; result['contract']['joint_pixels'] = dict(contract)
    result['scope'] = 'Joint-image trajectory hypotheses with original positions and priors retained; not physical accuracy'
    binding.update(joint_pixel_fit=pin, output_evidence_sha256=canonical_hash(candidate['reference_evidence']), audit_sha256=canonical_hash(result))
    candidate['camera_reference'].update(method=METHOD, angular_reference_sha256=canonical_hash(candidate['reference']))
    candidate['reconstruction_coverage']['two_view_body'].update(method=METHOD, joint_pixels=summary)
    candidate['assumptions'].append('Joint-image corrections allow uncertain knee/elbow directions; explicit noise/correction-speed assumptions do not measure depth, contact or physical motion')
    if settings['schema_version'] == 2:
        candidate['assumptions'].append('An authored angular-acceleration prior favors smoother source-time motion; it is not measured acceleration or a motor-force guarantee')
    return candidate, result


def restore_joint_pixel_input(spec):
    """Validate every correction before restoring the prior constructor for its original checks."""
    binding = spec['camera_reference']['body_refinement']; pin = binding.get('joint_pixel_fit')
    required = {'contract', 'uncertainty', 'input_method', 'input_reference_sha256', 'input_evidence_sha256', 'input_audit_sha256'}
    if (not isinstance(pin, dict) or set(pin) != required or pin['input_method'] != INPUT_METHOD
            or canonical_hash(spec['reference_evidence']) != binding['output_evidence_sha256']):
        raise ValueError('Joint images require their fixed contract and original constructor binding')
    settings = validate_uncertainty(pin['uncertainty']); summary = spec['reconstruction_coverage']['two_view_body'].get('joint_pixels', {})
    contract = trajectory_contract(settings)
    if (pin['contract'] != dict(contract) or summary.get('method') != contract['method'] or summary.get('uncertainty') != settings
            or summary.get('uncertainty_sha256') != canonical_hash(settings) or summary.get('uncertainty_calibrated') is not False
            or summary.get('depth_measured') is not False):
        raise ValueError('Joint-image uncertainty must remain explicit and uncalibrated')
    restored = deepcopy(spec); count = len(spec['camera_reference']['root_refinement']['views']); changed = constraints = 0
    maximum = 0.
    for reference, meta in zip(restored['reference'], restored['reference_evidence'], strict=True):
        records = meta.pop('geometry_joint_pixels', None)
        if meta.get('geometry_method') != METHOD or not isinstance(records, dict) or set(records) != set(LIMB_ENDPOINTS):
            raise ValueError('Every joint trajectory requires all four original limb records')
        meta['geometry_method'] = INPUT_METHOD
        for group, record in records.items():
            fields = {'input_angles_rad', 'correction_rad', 'input_joint_confidence', 'status', 'view_indices', 'observation_statuses'}
            if not isinstance(record, dict) or set(record) != fields:
                raise ValueError('Joint-image correction schema differs')
            old, delta, confidence = [record[k] for k in ('input_angles_rad', 'correction_rad', 'input_joint_confidence')]
            if (any(not isinstance(row, list) or len(row) != 4 for row in (old, delta, confidence))
                    or any(type(v) not in (int, float) or not math.isfinite(v) for v in old+delta+confidence)
                    or any(abs(v) > math.tau for v in old) or not 0 <= old[3] <= 2.6 or any(abs(v) > .30 for v in delta)
                    or any(not 0 <= v <= 1 for v in confidence) or not isinstance(record['status'], str)
                    or record['status'] not in STATUSES | {'unavailable_projection'}):
                raise ValueError('Joint-image corrections must preserve bounded input angles and source confidence')
            keys = {str(j) for j in LIMB_ENDPOINTS[group][1:]}; views, statuses = record['view_indices'], record['observation_statuses']
            if not isinstance(views, dict) or not isinstance(statuses, dict) or set(views) != keys or set(statuses) != keys:
                raise ValueError('Joint-image provenance must retain both endpoints and every view')
            used = 0
            for key in keys:
                if (not isinstance(statuses[key], list) or len(statuses[key]) != count
                        or any(not isinstance(s, str) or s not in OBSERVATION_STATUSES for s in statuses[key])
                        or not isinstance(views[key], list) or any(type(v) is not int for v in views[key])
                        or views[key] != [i for i, s in enumerate(statuses[key]) if s == 'available']):
                    raise ValueError('Joint-image contributors differ from original observation availability')
                used += len(views[key])
            altered = any(v != 0. for v in delta); changed += altered; constraints += used
            maximum = max(maximum, max(abs(v) for v in delta))
            if altered and (not used or record['status'] in ('unavailable_projection', 'unchanged_no_observations', 'unchanged_exact_images')):
                raise ValueError('Unobserved or rejected joint trajectories must remain unchanged')
            for name, angle, change, prior in zip(_names(group), old, delta, confidence, strict=True):
                if (abs(reference['joints'][name]-angle-change) > 1e-12
                        or meta['joint_confidence'].get(name) != (0. if altered else prior)):
                    raise ValueError('Joint-image angles or inferred-only confidence changed')
                reference['joints'][name] = angle; meta['joint_confidence'][name] = prior
    if (canonical_hash(restored['reference']) != pin['input_reference_sha256']
            or canonical_hash(restored['reference_evidence']) != pin['input_evidence_sha256']
            or summary.get('changed_group_poses') != changed or summary.get('source_image_constraints') != constraints
            or summary.get('maximum_angle_change_rad') != maximum):
        raise ValueError('Joint images changed their original reference, evidence or correction summary')
    restored['camera_reference'].update(method=INPUT_METHOD, angular_reference_sha256=pin['input_reference_sha256'])
    base = restored['camera_reference']['body_refinement']; base.pop('joint_pixel_fit')
    base.update(output_evidence_sha256=pin['input_evidence_sha256'], audit_sha256=pin['input_audit_sha256'])
    restored['reconstruction_coverage']['two_view_body'].pop('joint_pixels')
    restored['reconstruction_coverage']['two_view_body']['method'] = INPUT_METHOD
    return restored
