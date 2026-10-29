"""Optional image-direction fit; projection agreement does not identify 3D depth."""

from copy import deepcopy
import math
import time

import mujoco
import numpy as np

from weft.body_refinement import JOINT_LANDMARKS, SEGMENTS, _json_safe, _kinematics
from weft.human import HumanEnv
from weft.io import canonical_hash


IMAGE_REFINEMENT_CONTRACT = {
    'method': 'bounded-image-direction-reference-v1', 'unit_direction_huber_delta': .1,
    'original_angle_regularization': 1e-3, 'correction_temporal_regularization': 5e-4,
    'maximum_angle_change_rad': .15, 'maximum_iterations_per_frame': 12,
    'maximum_backtracks_per_iteration': 4, 'linear_system_ridge': 1e-6,
    'maximum_frames': 900, 'maximum_seconds': 60., 'maximum_temporal_gap_s': .25,
    'minimum_observed_bone_pixels': 2., 'minimum_model_projection_length': 1e-5,
    'direction_convergence': 1e-10, 'step_convergence_rad': 1e-8,
    'projection': 'Fixed orthographic simulation x,-z; no camera fit or independent depth measurement',
}
SEGMENT_NAMES = tuple(a + '__' + b for a, b in SEGMENTS)
_JOINT_INDEX = {name: index for index, name in enumerate(JOINT_LANDMARKS)}


def _dimensions(evidence, source_hash):
    width, height = evidence.get('width'), evidence.get('height')
    if any(type(value) is not int or not 16 <= value <= 8192 for value in (width, height)):
        raise ValueError('Image refinement requires bounded integer source width and height')
    provenance = evidence.get('image_dimension_provenance')
    if (not isinstance(provenance, dict) or provenance.get('method') not in
            ('verified_source_video', 'authored_analytic_fixture') or
            provenance.get('source_sha256') != source_hash or provenance.get('width') != width or
            provenance.get('height') != height):
        raise ValueError('Image dimensions require matching source provenance from the caller')
    return width, height


def _image_targets(frame, width, height):
    image = frame.get('landmarks')
    if not isinstance(image, list) or len(image) != 33:
        return None, None, None, 'missing_image_landmarks'
    target, weights, details = np.zeros((8, 2)), np.zeros(8), []
    for index, (a, b) in enumerate(SEGMENTS):
        detail = {'segment': SEGMENT_NAMES[index], 'observed_unit_direction': None,
                  'observed_length_pixels': None, 'source_confidence': 0., 'source_status': 'missing_endpoint'}
        details.append(detail)
        try:
            endpoints = [image[JOINT_LANDMARKS[name]] for name in (a, b)]
            xy = np.asarray([[point[key] for key in ('x', 'y')] for point in endpoints], dtype=float)
            conf = np.asarray([[point[key] for key in ('visibility', 'presence')] for point in endpoints], dtype=float)
        except (KeyError, TypeError, ValueError):
            continue
        if not np.isfinite(xy).all() or not np.isfinite(conf).all() or np.any(conf < 0) or np.any(conf > 1):
            detail['source_status'] = 'nonfinite_coordinates_or_invalid_confidence'
            continue
        detail['source_confidence'] = float(conf.min())
        if np.any(xy < -.05) or np.any(xy > 1.05):
            detail['source_status'] = 'outside_image_crop_tolerance'
            continue
        vector = (xy[1] - xy[0]) * [width, height]
        length = float(np.linalg.norm(vector))
        detail.update(observed_length_pixels=length, source_confidence=float(conf.min()))
        if length < IMAGE_REFINEMENT_CONTRACT['minimum_observed_bone_pixels']:
            detail['source_status'] = 'observed_projection_degenerate'
            continue
        target[index] = vector / length
        detail['observed_unit_direction'] = target[index].tolist()
        if conf.min() == 0:
            detail['source_status'] = 'zero_source_confidence'
        else:
            detail['source_status'] = 'available_image_estimate'
            weights[index] = conf.min()
    return target, weights, details, None


def _projected_kinematics(env, data, angles, jacobian=False):
    positions, jac = _kinematics(env, data, angles, jacobian)
    units, lengths = np.zeros((8, 2)), np.zeros(8)
    result = np.zeros((8, 2, 19)) if jacobian else None
    jac = jac.reshape(12, 3, 19) if jacobian else None
    for index, (a, b) in enumerate(SEGMENTS):
        ia, ib = _JOINT_INDEX[a], _JOINT_INDEX[b]
        vector3 = positions[ib] - positions[ia]
        vector = np.array([vector3[0], -vector3[2]])
        length = lengths[index] = np.linalg.norm(vector)
        if length < IMAGE_REFINEMENT_CONTRACT['minimum_model_projection_length']:
            continue
        units[index] = vector / length
        if jacobian:
            vector_jac = np.stack((jac[ib, 0] - jac[ia, 0], -jac[ib, 2] + jac[ia, 2]))
            result[index] = (np.eye(2) - np.outer(units[index], units[index])) @ vector_jac / length
    valid = lengths >= IMAGE_REFINEMENT_CONTRACT['minimum_model_projection_length']
    return units, lengths, valid, result.reshape(16, 19) if jacobian else None


def _loss(residual, weights, delta, previous):
    norm = np.linalg.norm(residual, axis=1)
    huber = IMAGE_REFINEMENT_CONTRACT['unit_direction_huber_delta']
    data = float(np.sum(weights * np.where(norm <= huber, .5 * norm ** 2, huber * (norm - .5 * huber))))
    prior = .5 * IMAGE_REFINEMENT_CONTRACT['original_angle_regularization'] * float(delta @ delta)
    temporal = (0. if previous is None else .5 * IMAGE_REFINEMENT_CONTRACT['correction_temporal_regularization']
                * float((delta - previous) @ (delta - previous)))
    return data + prior + temporal, data


def _errors(units, target, valid):
    # atan2 stays stable for perfect alignment; unavailable directions remain null.
    return {name: (float(math.atan2(abs(u[0]*t[1]-u[1]*t[0]), float(u @ t))) if ok else None)
            for name, u, t, ok in zip(SEGMENT_NAMES, units, target, valid)}


def refine_image_reference(spec, evidence):
    """Return a bounded, unpromoted candidate plus source/derivation uncertainty audit."""
    started = time.monotonic()
    contract = IMAGE_REFINEMENT_CONTRACT
    if not isinstance(spec, dict) or spec.get('profile') != 'exercise-v3':
        raise ValueError('Image refinement requires an exercise-v3 model')
    if 'reference_refinement' in spec or 'image_reference_refinement' in spec:
        raise ValueError('Image refinement must not stack on another candidate fit')
    if not isinstance(evidence, dict) or evidence.get('source_sha256') != spec.get('source_sha256') or not isinstance(spec.get('source_sha256'), str):
        raise ValueError('Image refinement source identity must match the original model')
    width, height = _dimensions(evidence, spec['source_sha256'])
    frames = evidence.get('frames')
    if not isinstance(frames, list) or len(frames) > contract['maximum_frames']:
        raise ValueError('Image refinement accepts at most 900 evidence frames')
    if not isinstance(spec.get('reference'), list) or not 2 <= len(spec['reference']) <= contract['maximum_frames']:
        raise ValueError('Image refinement accepts 2–900 reference poses')
    by_id = {}
    for frame in frames:
        if not isinstance(frame, dict) or not isinstance(frame.get('id'), str) or frame['id'] in by_id:
            raise ValueError('Image evidence requires unique source IDs')
        by_id[frame['id']] = frame
    provenance = spec.get('reference_evidence', [])
    if not isinstance(provenance, list) or len(provenance) > contract['maximum_frames']:
        raise ValueError('Reference provenance exceeds its bound')
    by_time = {}
    for row in provenance:
        if not isinstance(row, dict) or row.get('time_s') in by_time:
            raise ValueError('Reference provenance times must be unique')
        by_time[row.get('time_s')] = row
    candidate = deepcopy(spec)
    model_hash, evidence_hash = canonical_hash(spec), canonical_hash(_json_safe(evidence))
    deadline = started + contract['maximum_seconds']
    env = HumanEnv(spec)
    rows, previous, previous_time = [], None, None
    try:
        data = mujoco.MjData(env.model)
        lower = np.array([env.joint_limits[name][0] for name in env.joint_names])
        upper = np.array([env.joint_limits[name][1] for name in env.joint_names])
        for index, reference in enumerate(spec['reference']):
            timestamp = reference['time_s']
            identifier = by_time.get(timestamp, {}).get('evidence_id')
            frame = by_id.get(identifier)
            original = np.array([reference['joints'][name] for name in env.joint_names], dtype=float)
            row = {'reference_index': index, 'time_s': timestamp, 'evidence_id': identifier,
                   'raw_source_evidence': _json_safe(deepcopy(frame)), 'original_joints': deepcopy(reference['joints']),
                   'refined_joints': deepcopy(reference['joints']), 'iterations': 0, 'backtrack_evaluations': 0,
                   'degenerate_trial_rejections': 0, 'direction_errors_before_rad': None, 'direction_errors_after_rad': None,
                   'depth_status': 'unidentified_same_image_directions_can_have_multiple_3d_poses'}
            rows.append(row)
            reason = 'missing_source_frame' if frame is None else (
                'source_time_mismatch' if frame.get('time_s') != timestamp else None)
            if reason is None and (frame.get('status') not in ('tracked', 'tracked_partial') or frame.get('pose_count', 1) != 1):
                reason = 'unsupported_pose_status_or_multiple_hypotheses'
            target, weights, details = None, None, None
            if reason is None:
                target, weights, details, reason = _image_targets(frame, width, height)
                row['segments'] = details
            if time.monotonic() >= deadline:
                reason = 'compute_budget_exhausted'
            if reason is not None:
                row['status'] = 'unchanged_' + reason
                previous, previous_time = None, None
                continue
            if previous_time is None or timestamp - previous_time > contract['maximum_temporal_gap_s'] + 1e-9:
                previous = None
            row['temporal_prior_source_time_s'] = previous_time if previous is not None else None
            data.qpos[env.mobility.q] = spec['root_reference'][index]['translation']
            initial, initial_lengths, model_valid, _ = _projected_kinematics(env, data, original)
            observed_valid = np.array([d['observed_unit_direction'] is not None for d in details])
            comparable = model_valid & observed_valid
            weights *= model_valid
            for item, length, valid, weight in zip(details, initial_lengths, model_valid, weights):
                item.update(model_projection_length_before=float(length), model_direction_status='available' if valid else 'model_projection_degenerate',
                            effective_fit_weight=float(weight))
            delta = np.zeros(19)
            minimum = np.maximum(lower-original, -contract['maximum_angle_change_rad'])
            maximum = np.minimum(upper-original, contract['maximum_angle_change_rad'])
            before, before_data = _loss(initial-target, weights, delta, previous)
            status = 'unchanged_perfect_weighted_projection' if np.any(weights > 0) else 'unchanged_no_usable_weighted_directions'
            if np.any(weights > 0) and np.max(np.linalg.norm(initial-target, axis=1)[weights > 0]) > contract['direction_convergence']:
                status = 'bounded_image_fit'
                for iteration in range(contract['maximum_iterations_per_frame']):
                    if time.monotonic() >= deadline:
                        status = 'partial_compute_budget_exhausted'
                        break
                    units, _, _, jac = _projected_kinematics(env, data, original+delta, True)
                    residual = units-target
                    objective, _ = _loss(residual, weights, delta, previous)
                    robust = weights*np.minimum(1., contract['unit_direction_huber_delta']/np.maximum(np.linalg.norm(residual, axis=1),1e-15))
                    diagonal = np.repeat(robust,2)
                    lam = contract['original_angle_regularization']
                    temporal = contract['correction_temporal_regularization'] if previous is not None else 0.
                    hessian = jac.T@(diagonal[:,None]*jac)+np.eye(19)*(lam+temporal+contract['linear_system_ridge'])
                    gradient = jac.T@(diagonal*residual.ravel())+lam*delta
                    if previous is not None:
                        gradient += temporal*(delta-previous)
                    step = np.linalg.solve(hessian,-gradient)
                    row['iterations'] = iteration+1
                    if not np.isfinite(step).all():
                        status = 'bounded_image_fit_nonfinite_step_rejected'
                        break
                    accepted = False
                    for backtrack in range(contract['maximum_backtracks_per_iteration']):
                        if time.monotonic() >= deadline:
                            status = 'partial_compute_budget_exhausted'
                            break
                        trial = np.clip(delta+step*(.5**backtrack),minimum,maximum)
                        if np.max(np.abs(trial-delta)) < contract['step_convergence_rad']:
                            break
                        trial_units, _, trial_valid, _ = _projected_kinematics(env,data,original+trial)
                        row['backtrack_evaluations'] += 1
                        if np.any(comparable & ~trial_valid):
                            row['degenerate_trial_rejections'] += 1
                            continue
                        trial_objective,_ = _loss(trial_units-target,weights,trial,previous)
                        if trial_objective < objective-1e-14:
                            delta,accepted = trial,True
                            break
                    if not accepted:
                        break
            final, final_lengths, final_valid, _ = _projected_kinematics(env,data,original+delta)
            after, after_data = _loss(final-target,weights,delta,previous)
            refined = dict(zip(env.joint_names,(original+delta).tolist()))
            candidate['reference'][index]['joints'] = refined
            errors_before = _errors(initial,target,comparable)
            errors_after = _errors(final,target,comparable & final_valid)
            for item,length in zip(details,final_lengths):
                item['model_projection_length_after'] = float(length)
            row.update(status=status,refined_joints=refined,angle_correction_rad=dict(zip(env.joint_names,delta.tolist())),
                       maximum_angle_change_rad=float(np.max(np.abs(delta))),objective_before=before,objective_after=after,
                       weighted_huber_before=before_data,weighted_huber_after=after_data,
                       direction_errors_before_rad=errors_before,direction_errors_after_rad=errors_after,
                       comparable_segments=int(comparable.sum()),weighted_segments=int((weights>0).sum()))
            previous = delta.copy() if np.any(weights>0) else None
            previous_time = timestamp if previous is not None else None
    finally:
        env.close()
    changed = sum(row.get('maximum_angle_change_rad',0.)>0 for row in rows)
    partial = any('budget_exhausted' in row['status'] for row in rows)
    candidate['image_reference_refinement'] = {
        'method':contract['method'],'status':'candidate_not_promoted','input_model_sha256':model_hash,
        'input_evidence_sha256':evidence_hash,'changed_frames':changed,'compute_budget_exhausted':partial,
        'scope':'Derived angles matching estimated 2D directions; original-angle prior does not measure depth or improve source confidence'}
    for row in candidate.get('reference_evidence',[]):
        row['image_reference_derivation'] = {'method':contract['method'],'source_confidence_unchanged':True,
                                             'depth_identified':False}
    audit = {'schema_version':1,'status':'partial_budget_exhausted' if partial else 'complete','promoted':False,
             'contract':deepcopy(contract),'input_model_sha256':model_hash,'input_evidence_sha256':evidence_hash,
             'output_model_sha256':canonical_hash(candidate),'source_sha256':spec['source_sha256'],
             'image_dimensions':{'width':width,'height':height,'provenance':deepcopy(evidence['image_dimension_provenance']),
                                 'verification_scope':'Caller-attested dimensions; fitter does not decode or verify the source file'},
             'frames':rows,'changed_frames':changed,'unused_evidence_ids':sorted(set(by_id)-{r['evidence_id'] for r in rows if r['evidence_id'] is not None}),
             'seconds':time.monotonic()-started,'physics_steps':0,
             'scope':'Construction image-direction fit, not independent validation, recovered depth, anatomy, contact forces or balance',
             'uncertainty':'A camera-aligned 2D direction can correspond to multiple 3D limb poses; the original-angle prior selects a nearby solution, not measured depth',
             'fixed_fields':['root_reference','mobility','lengths','body_frame','motor_limits','dt'],
             'loss_excludes':['world_landmark_positions','floor_distance','contact_force','evaluation_labels','policy_return','root_translation'],
             'evidence_hash_encoding':'Canonical JSON; nonfinite raw values retained as tagged strings'}
    return candidate,audit
