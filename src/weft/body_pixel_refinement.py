"""Bounded image-position fit with fixed shape, root, scale and floor; depth stays inferred."""

from copy import deepcopy
import time

import mujoco
import numpy as np

from weft.body_image_refinement import _dimensions
from weft.body_refinement import JOINT_LANDMARKS, _json_safe, _kinematics
from weft.human import HumanEnv
from weft.io import canonical_hash
from weft.root_motion import build_root_reference


CONTRACT = {
    'method': 'bounded-image-position-reference-v1', 'minimum_confidence': .5,
    'huber_delta_model_units': .05, 'original_angle_regularization': 1e-3,
    'correction_temporal_regularization': 5e-4, 'maximum_angle_change_rad': .15,
    'maximum_iterations_per_frame': 12, 'maximum_backtracks_per_iteration': 4,
    'linear_system_ridge': 1e-6, 'maximum_frames': 900, 'maximum_seconds': 60.,
    'maximum_temporal_gap_s': .25, 'position_convergence_model_units': 1e-10,
    'step_convergence_rad': 1e-8,
    'projection': 'Fixed simulation x,-z and original torso scale, centered at the source pelvis estimate',
}
NAMES = tuple(JOINT_LANDMARKS)


def _projection(env, data, angles, jacobian=False):
    positions, jac = _kinematics(env, data, angles, jacobian)
    projected = positions[:, [0, 2]] * [1., -1.]
    if jacobian:
        jac = jac.reshape(12, 3, 19)[:, [0, 2], :] * np.array([1., -1.])[None, :, None]
        jac = jac.reshape(24, 19)
    return projected, jac


def _targets(frame, root, width, height, scale):
    target, weights, details = np.zeros((12, 2)), np.zeros(12), []
    if frame.get('status') not in ('tracked', 'tracked_partial') or frame.get('pose_count', 1) != 1:
        return target, weights, details, 'unsupported_pose_status_or_multiple_hypotheses'
    if root['status'] != 'observed_image_estimate' or root['pelvis_image_px'] is None:
        return target, weights, details, 'unobserved_image_origin'
    landmarks = frame.get('landmarks')
    if not isinstance(landmarks, list) or len(landmarks) != 33:
        return target, weights, details, 'missing_image_landmarks'
    for i, (name, index) in enumerate(JOINT_LANDMARKS.items()):
        item = {'joint': name, 'source_status': 'missing_endpoint', 'source_pixel': None,
                'source_confidence': None, 'effective_fit_weight': 0., 'target_model_units': None}
        details.append(item)
        point = landmarks[index]
        try:
            values = [point[k] for k in ('x', 'y', 'visibility', 'presence')]
            if any(type(v) not in (int, float) or not np.isfinite(v) for v in values):
                raise ValueError('Invalid image coordinate/confidence')
            x, y, visibility, presence = values
            if abs(x) > 10 or abs(y) > 10 or not 0 <= min(visibility, presence) <= max(visibility, presence) <= 1:
                raise ValueError('Invalid image confidence')
        except (KeyError, TypeError, ValueError):
            item['source_status'] = 'invalid_or_missing_endpoint'
            continue
        pixel = np.array([x * width, y * height])
        target[i] = (pixel - root['pelvis_image_px']) * scale
        confidence = min(visibility, presence, root['confidence'])
        item.update(source_pixel=pixel.tolist(), source_confidence=min(visibility, presence),
                    target_model_units=target[i].tolist())
        if not 0 <= x <= 1 or not 0 <= y <= 1:
            item['source_status'] = 'outside_source_image'
        elif confidence < CONTRACT['minimum_confidence']:
            item['source_status'] = 'low_confidence'
        else:
            weights[i] = confidence
            item.update(source_status='available_image_estimate', effective_fit_weight=confidence)
    return target, weights, details, None


def _loss(residual, weights, delta, previous):
    norm = np.linalg.norm(residual, axis=1)
    huber = CONTRACT['huber_delta_model_units']
    data = float(np.sum(weights * np.where(norm <= huber, .5 * norm ** 2, huber * (norm - .5 * huber))))
    prior = .5 * CONTRACT['original_angle_regularization'] * float(delta @ delta)
    temporal = (0. if previous is None else .5 * CONTRACT['correction_temporal_regularization']
                * float((delta - previous) @ (delta - previous)))
    return data + prior + temporal, data


def refine_pixel_reference(spec, evidence):
    """Return an unpromoted reference candidate; never fit floor, dimensions or root motion."""
    started = time.monotonic()
    if not isinstance(spec, dict) or spec.get('profile') != 'exercise-v3':
        raise ValueError('Pixel-position fitting requires an original mobile exercise-v3 body')
    if 'camera_reference' in spec or any('refinement' in key for key in spec):
        raise ValueError('Pixel-position fitting does not stack fits or override supplied cameras')
    if not isinstance(evidence, dict) or spec.get('source_sha256') != evidence.get('source_sha256'):
        raise ValueError('Pixel-position fitting requires matching source identity')
    width, height = _dimensions(evidence, spec['source_sha256'])
    if not isinstance(spec.get('reference'), list) or not 2 <= len(spec['reference']) <= CONTRACT['maximum_frames']:
        raise ValueError('Pixel-position fitting accepts 2–900 reference poses')
    # Recompute the existing source scale; a supplied scale cannot silently change the objective.
    root = build_root_reference(evidence, torso_length=spec['lengths']['torso'])
    expected = [{'time_s': row['time_s'], 'translation': row['translation']} for row in root['reference']]
    if root['status'] != 'ready' or expected != spec.get('root_reference'):
        raise ValueError('Pixel-position fitting must preserve the source-derived root and scale')
    provenance = spec.get('reference_evidence')
    if not isinstance(provenance, list) or len(provenance) != len(expected) or len(expected) != len(spec['reference']):
        raise ValueError('Every body reference requires aligned source provenance')
    for reference, meta, origin in zip(spec['reference'], provenance, root['reference']):
        if (not isinstance(meta, dict) or reference['time_s'] != origin['time_s']
                or meta.get('time_s') != origin['time_s'] or meta.get('evidence_id') != origin['evidence_id']):
            raise ValueError('Body/source reference IDs and clocks must match exactly')
    scale = root['scale']['model_units_per_pixel']
    by_id = {frame['id']: frame for frame in evidence['frames']}
    candidate = deepcopy(spec)
    model_hash, evidence_hash = canonical_hash(spec), canonical_hash(_json_safe(evidence))
    rows, previous, previous_time = [], None, None
    deadline = started + CONTRACT['maximum_seconds']
    env = HumanEnv(spec)
    try:
        if len(env.joint_names) != 19:
            raise ValueError('Pixel-position fitting requires the original 19-axis body')
        data = mujoco.MjData(env.model)
        lower = np.array([env.joint_limits[name][0] for name in env.joint_names])
        upper = np.array([env.joint_limits[name][1] for name in env.joint_names])
        for index, (reference, origin) in enumerate(zip(spec['reference'], root['reference'])):
            frame = by_id[origin['evidence_id']]
            stamp = reference['time_s']
            original = np.array([reference['joints'][name] for name in env.joint_names])
            row = {'reference_index': index, 'evidence_id': frame['id'], 'time_s': stamp,
                   'frame': frame['frame'], 'raw_source_frame': _json_safe(deepcopy(frame)),
                   'original_joints': deepcopy(reference['joints']), 'refined_joints': deepcopy(reference['joints']),
                   'iterations': 0, 'backtrack_evaluations': 0, 'depth_identified': False}
            rows.append(row)
            target, weights, details, reason = _targets(frame, origin, width, height, scale)
            row['points'] = details
            if time.monotonic() >= deadline:
                reason = 'compute_budget_exhausted'
            if reason is not None:
                row['status'] = 'unchanged_' + reason
                previous, previous_time = None, None
                continue
            if previous_time is None or stamp - previous_time > CONTRACT['maximum_temporal_gap_s'] + 1e-9:
                previous = None
            row['temporal_prior_source_time_s'] = previous_time if previous is not None else None
            initial, _ = _projection(env, data, original)
            delta = np.zeros(19)
            minimum = np.maximum(-CONTRACT['maximum_angle_change_rad'], lower - original)
            maximum = np.minimum(CONTRACT['maximum_angle_change_rad'], upper - original)
            before, before_data = _loss(initial - target, weights, delta, previous)
            status = 'unchanged_perfect_projection' if np.any(weights) else 'unchanged_no_weighted_points'
            if np.any(weights) and np.max(np.linalg.norm(initial - target, axis=1)[weights > 0]) > CONTRACT['position_convergence_model_units']:
                status = 'bounded_fit'
                for iteration in range(CONTRACT['maximum_iterations_per_frame']):
                    if time.monotonic() >= deadline:
                        status = 'partial_compute_budget_exhausted'
                        break
                    positions, jac = _projection(env, data, original + delta, True)
                    residual = positions - target
                    objective, _ = _loss(residual, weights, delta, previous)
                    robust = weights * np.minimum(1., CONTRACT['huber_delta_model_units'] /
                                                  np.maximum(np.linalg.norm(residual, axis=1), 1e-15))
                    diagonal = np.repeat(robust, 2)
                    lam = CONTRACT['original_angle_regularization']
                    temporal = CONTRACT['correction_temporal_regularization'] if previous is not None else 0.
                    hessian = jac.T @ (diagonal[:, None] * jac) + np.eye(19) * (lam + temporal + CONTRACT['linear_system_ridge'])
                    gradient = jac.T @ (diagonal * residual.ravel()) + lam * delta
                    if previous is not None:
                        gradient += temporal * (delta - previous)
                    step = np.linalg.solve(hessian, -gradient)
                    row['iterations'] = iteration + 1
                    if not np.isfinite(step).all():
                        status = 'nonfinite_step_rejected'
                        break
                    accepted = False
                    for backtrack in range(CONTRACT['maximum_backtracks_per_iteration']):
                        if time.monotonic() >= deadline:
                            status = 'partial_compute_budget_exhausted'
                            break
                        trial = np.clip(delta + step * .5 ** backtrack, minimum, maximum)
                        if np.max(np.abs(trial - delta)) < CONTRACT['step_convergence_rad']:
                            break
                        trial_positions, _ = _projection(env, data, original + trial)
                        trial_objective, _ = _loss(trial_positions - target, weights, trial, previous)
                        row['backtrack_evaluations'] += 1
                        if trial_objective < objective - 1e-14:
                            delta, accepted = trial, True
                            break
                    if not accepted:
                        break
            final, _ = _projection(env, data, original + delta)
            after, after_data = _loss(final - target, weights, delta, previous)
            refined = dict(zip(env.joint_names, map(float, original + delta)))
            candidate['reference'][index]['joints'] = refined
            for i, point in enumerate(details):
                point.update(model_before=initial[i].tolist(), model_after=final[i].tolist(),
                             error_before_px=(float(np.linalg.norm(initial[i] - target[i]) / scale)
                                              if point['target_model_units'] is not None else None),
                             error_after_px=(float(np.linalg.norm(final[i] - target[i]) / scale)
                                             if point['target_model_units'] is not None else None))
            row.update(status=status, refined_joints=refined, objective_before=before, objective_after=after,
                       weighted_huber_before=before_data, weighted_huber_after=after_data,
                       angle_correction_rad=dict(zip(env.joint_names, delta.tolist())),
                       maximum_angle_change_rad=float(np.max(np.abs(delta))), weighted_points=int(np.count_nonzero(weights)))
            previous = delta.copy() if np.any(weights) else None
            previous_time = stamp if previous is not None else None
    finally:
        env.close()
    changed = sum(row.get('maximum_angle_change_rad', 0.) > 0 for row in rows)
    exhausted = any('compute_budget_exhausted' in row['status'] for row in rows)
    used_ids = {row['evidence_id'] for row in root['reference']}
    metadata = {'method': CONTRACT['method'], 'status': 'candidate_not_promoted',
                'input_model_sha256': model_hash, 'input_evidence_sha256': evidence_hash,
                'source_scale': deepcopy(root['scale']), 'changed_frames': changed,
                'compute_budget_exhausted': exhausted, 'depth_identified': False,
                'floor_scope': 'Original body first-pose floor offset retained, not refitted to candidate poses'}
    candidate['pixel_reference_refinement'] = metadata
    candidate['assumptions'].append('Image-position fitted angles remain estimates; the floor offset belongs to the original body, not the changed first pose')
    for row in candidate['reference_evidence']:
        row['pixel_reference_derivation'] = {'method': CONTRACT['method'], 'depth_identified': False,
                                            'original_confidence_is_not_fit_accuracy': True}
    audit = {'schema_version': 1, 'status': 'partial_budget_exhausted' if exhausted else 'complete',
             'contract': deepcopy(CONTRACT), 'input_model_sha256': model_hash,
             'input_evidence_sha256': evidence_hash, 'output_model_sha256': canonical_hash(candidate),
             'source_sha256': spec['source_sha256'], 'root_audit_sha256': canonical_hash(root),
             'model_units_per_pixel': scale, 'image_dimensions': [width, height],
             'frames': rows, 'changed_frames': changed, 'retained_source_samples': len(evidence['frames']),
             'unused_evidence_ids': [frame['id'] for frame in evidence['frames'] if frame['id'] not in used_ids],
             'seconds': time.monotonic() - started, 'physics_steps': 0, 'promoted': False,
             'fixed_fields': ['root_reference', 'mobility', 'lengths', 'body_frame', 'motor_limits', 'dt'],
             'loss_excludes': ['world_landmarks', 'floor_distance', 'contact_force', 'evaluation_labels', 'policy_return', 'root_translation'],
             'scope': 'Image-model estimate agreement only; shared pelvis errors, depth, scale and perspective remain unresolved'}
    return candidate, audit
