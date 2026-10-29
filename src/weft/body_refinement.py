"""Optional construction-only Cartesian fit; source evidence and physical assumptions stay fixed."""

from copy import deepcopy
import math
import time

import mujoco
import numpy as np

from weft.body_frame import CAMERA_TO_SIM
from weft.human import HumanEnv
from weft.io import canonical_hash


REFINEMENT_CONTRACT = {
    'method': 'bounded-cartesian-reference-v1', 'huber_delta_model_units': .05,
    'original_angle_regularization': 1e-3, 'correction_temporal_regularization': 5e-4,
    'maximum_angle_change_rad': .15, 'maximum_iterations_per_frame': 12,
    'maximum_backtracks_per_iteration': 4, 'linear_system_ridge': 1e-6,
    'maximum_frames': 900, 'maximum_seconds': 60., 'maximum_temporal_gap_s': .25,
    'position_convergence_model_units': 1e-10, 'step_convergence_rad': 1e-8,
}
JOINT_LANDMARKS = {'left_shoulder': 11, 'right_shoulder': 12, 'left_elbow': 13,
                   'right_elbow': 14, 'left_wrist': 15, 'right_wrist': 16,
                   'left_hip': 23, 'right_hip': 24, 'left_knee': 25,
                   'right_knee': 26, 'left_ankle': 27, 'right_ankle': 28}
SEGMENTS = tuple((f'{side}_{a}', f'{side}_{b}') for side in ('left', 'right')
                 for a, b in [('shoulder', 'elbow'), ('elbow', 'wrist'), ('hip', 'knee'), ('knee', 'ankle')])


def _json_safe(value):
    # Invalid raw numeric evidence remains identifiable without emitting invalid JSON.
    if isinstance(value, float) and not math.isfinite(value):
        return {'nonfinite_raw_value': repr(value)}
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _targets(frame):
    if frame.get('status') not in ('tracked', 'tracked_partial') or frame.get('pose_count', 1) != 1:
        return None, None, 'unsupported_pose_status_or_multiple_hypotheses'
    world, image = frame.get('world_landmarks'), frame.get('landmarks')
    if not isinstance(world, list) or len(world) != 33 or not isinstance(image, list) or len(image) != 33:
        return None, None, 'missing_world_or_image_landmarks'
    try:
        points = np.asarray([[p[k] for k in ('x', 'y', 'z')] for p in world], dtype=float)
        if not np.isfinite(points).all() or np.abs(points).max() > 10:
            return None, None, 'nonfinite_or_unbounded_world_landmarks'
        points = points @ CAMERA_TO_SIM.T
        points -= (points[23] + points[24]) / 2
        weights = []
        for index in JOINT_LANDMARKS.values():
            p = image[index]
            visibility = [float(p[k]) for k in ('visibility', 'presence')]
            xy = [float(p[k]) for k in ('x', 'y')]
            if any(not math.isfinite(v) or not 0 <= v <= 1 for v in visibility):
                return None, None, 'invalid_landmark_confidence'
            weights.append(min(visibility) if all(math.isfinite(v) and -.05 <= v <= 1.05 for v in xy) else 0.)
        return points[list(JOINT_LANDMARKS.values())], np.asarray(weights), None
    except (KeyError, TypeError, ValueError):
        return None, None, 'malformed_world_or_image_landmarks'


def _kinematics(env, data, angles, jacobian=False):
    data.qpos[env._q_array] = angles
    mujoco.mj_forward(env.model, data)
    pelvis = env._site_ids['pelvis']
    sites = [env._site_ids[name] for name in JOINT_LANDMARKS]
    positions = data.site_xpos[sites] - data.site_xpos[pelvis]
    if not jacobian:
        return positions.copy(), None
    center_jac = np.zeros((3, env.model.nv))
    mujoco.mj_jacSite(env.model, data, center_jac, None, pelvis)
    jac = np.empty((12, 3, 19))
    scratch = np.zeros((3, env.model.nv))
    for index, site in enumerate(sites):
        mujoco.mj_jacSite(env.model, data, scratch, None, site)
        jac[index] = (scratch - center_jac)[:, env._v_array]
    return positions.copy(), jac.reshape(36, 19)


def _objective(residual, weights, delta, previous):
    norms = np.linalg.norm(residual, axis=1)
    huber = REFINEMENT_CONTRACT['huber_delta_model_units']
    data_loss = float(np.sum(weights * np.where(norms <= huber, .5 * norms ** 2, huber * (norms - .5 * huber))))
    prior = .5 * REFINEMENT_CONTRACT['original_angle_regularization'] * float(delta @ delta)
    temporal = (0. if previous is None else .5 * REFINEMENT_CONTRACT['correction_temporal_regularization']
                * float((delta - previous) @ (delta - previous)))
    return data_loss + prior + temporal, data_loss


def _direction_errors(positions, target):
    indices = {name: index for index, name in enumerate(JOINT_LANDMARKS)}
    result = {}
    for a, b in SEGMENTS:
        model = positions[indices[b]] - positions[indices[a]]
        source = target[indices[b]] - target[indices[a]]
        denominator = np.linalg.norm(model) * np.linalg.norm(source)
        result[a + '__' + b] = (float(np.arccos(np.clip(model @ source / denominator, -1, 1)))
                                if denominator > 1e-10 else None)
    return result


def refine_reference(spec, evidence):
    """Return a candidate spec and complete audit, never a promoted physical repair."""
    started = time.monotonic()
    contract = REFINEMENT_CONTRACT
    if not isinstance(spec, dict) or spec.get('profile') != 'exercise-v3':
        raise ValueError('Cartesian refinement requires an exercise-v3 model')
    if 'reference_refinement' in spec:
        raise ValueError('Refinement must start from the original reference, not stack candidate corrections')
    frames = evidence.get('frames') if isinstance(evidence, dict) else None
    if not isinstance(frames, list) or len(frames) > contract['maximum_frames']:
        raise ValueError('Cartesian refinement accepts at most 900 retained evidence frames')
    if (not isinstance(spec.get('source_sha256'), str) or
            evidence.get('source_sha256') != spec['source_sha256']):
        raise ValueError('Cartesian refinement source identity must match the original model')
    if not isinstance(spec.get('reference'), list) or not 2 <= len(spec['reference']) <= contract['maximum_frames']:
        raise ValueError('Cartesian refinement accepts 2–900 reference poses')
    by_id = {}
    for frame in frames:
        if not isinstance(frame, dict) or not isinstance(frame.get('id'), str) or frame['id'] in by_id:
            raise ValueError('Cartesian evidence frames require unique source IDs')
        by_id[frame['id']] = frame
    provenance = spec.get('reference_evidence', [])
    if not isinstance(provenance, list) or len(provenance) > contract['maximum_frames']:
        raise ValueError('Cartesian reference provenance exceeds its bound')
    by_time = {}
    for row in provenance:
        if not isinstance(row, dict) or row.get('time_s') in by_time:
            raise ValueError('Reference provenance times must be unique')
        by_time[row.get('time_s')] = row
    candidate = deepcopy(spec)
    model_hash, evidence_hash = canonical_hash(spec), canonical_hash(_json_safe(evidence))
    env = HumanEnv(spec)
    rows, previous, previous_time = [], None, None
    deadline = started + contract['maximum_seconds']
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
                   'original_joints': deepcopy(reference['joints']), 'refined_joints': deepcopy(reference['joints']),
                   'raw_target_evidence': _json_safe(deepcopy(frame)), 'iterations': 0, 'backtrack_evaluations': 0,
                   'unweighted_endpoint_rmse_before': None, 'unweighted_endpoint_rmse_after': None,
                   'direction_errors_before_rad': None, 'direction_errors_after_rad': None}
            rows.append(row)
            reason = 'missing_source_frame' if frame is None else (
                'source_time_mismatch' if frame.get('time_s') != timestamp else None)
            target, weights = None, None
            if reason is None:
                target, weights, reason = _targets(frame)
            if target is not None:
                row.update(target_joints_hip_centered={name: value.tolist() for name, value in zip(JOINT_LANDMARKS, target)},
                           landmark_weights=dict(zip(JOINT_LANDMARKS, weights.tolist())))
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
            initial, _ = _kinematics(env, data, original)
            row['unweighted_endpoint_rmse_before'] = float(np.sqrt(np.mean(np.sum((initial-target)**2, axis=1))))
            row['direction_errors_before_rad'] = _direction_errors(initial, target)
            delta = np.zeros(19)
            minimum = np.maximum(lower - original, -contract['maximum_angle_change_rad'])
            maximum = np.minimum(upper - original, contract['maximum_angle_change_rad'])
            before, before_data = _objective(initial-target, weights, delta, previous)
            status = 'unchanged_perfect_weighted_geometry' if np.any(weights > 0) else 'unchanged_no_weighted_source_landmarks'
            if np.any(weights > 0) and np.max(np.linalg.norm(initial-target, axis=1)[weights > 0]) > contract['position_convergence_model_units']:
                status = 'bounded_fit'
                for iteration in range(contract['maximum_iterations_per_frame']):
                    if time.monotonic() >= deadline:
                        status = 'partial_compute_budget_exhausted'
                        break
                    positions, jac = _kinematics(env, data, original+delta, jacobian=True)
                    residual = positions - target
                    objective, _ = _objective(residual, weights, delta, previous)
                    robust = weights * np.minimum(1., contract['huber_delta_model_units'] / np.maximum(np.linalg.norm(residual, axis=1), 1e-15))
                    diagonal = np.repeat(robust, 3)
                    lam = contract['original_angle_regularization']
                    temporal = contract['correction_temporal_regularization'] if previous is not None else 0.
                    hessian = jac.T @ (diagonal[:, None] * jac) + np.eye(19) * (lam + temporal + contract['linear_system_ridge'])
                    gradient = jac.T @ (diagonal * residual.ravel()) + lam * delta
                    if previous is not None:
                        gradient += temporal * (delta - previous)
                    step = np.linalg.solve(hessian, -gradient)
                    row['iterations'] = iteration + 1
                    if not np.isfinite(step).all():
                        status = 'bounded_fit_nonfinite_step_rejected'
                        break
                    accepted = False
                    for backtrack in range(contract['maximum_backtracks_per_iteration']):
                        if time.monotonic() >= deadline:
                            status = 'partial_compute_budget_exhausted'
                            break
                        trial = np.clip(delta + step * (0.5 ** backtrack), minimum, maximum)
                        if np.max(np.abs(trial-delta)) < contract['step_convergence_rad']:
                            break
                        trial_positions, _ = _kinematics(env, data, original+trial)
                        trial_objective, _ = _objective(trial_positions-target, weights, trial, previous)
                        row['backtrack_evaluations'] += 1
                        if trial_objective < objective - 1e-14:
                            delta, accepted = trial, True
                            break
                    if not accepted:
                        break
            final, _ = _kinematics(env, data, original+delta)
            after, after_data = _objective(final-target, weights, delta, previous)
            refined = {name: float(value) for name, value in zip(env.joint_names, original+delta)}
            candidate['reference'][index]['joints'] = refined
            row.update(status=status, refined_joints=refined, angle_correction_rad=dict(zip(env.joint_names, delta.tolist())),
                       maximum_angle_change_rad=float(np.max(np.abs(delta))), objective_before=before, objective_after=after,
                       weighted_huber_before=before_data, weighted_huber_after=after_data,
                       unweighted_endpoint_rmse_after=float(np.sqrt(np.mean(np.sum((final-target)**2, axis=1)))),
                       direction_errors_after_rad=_direction_errors(final, target))
            previous = delta.copy() if np.any(weights > 0) else None
            previous_time = timestamp if previous is not None else None
    finally:
        env.close()
    changed = sum(row.get('maximum_angle_change_rad', 0.) > 0 for row in rows)
    budget_exhausted = any('budget_exhausted' in row['status'] for row in rows)
    metadata = {'method': contract['method'], 'status': 'candidate_not_promoted', 'input_model_sha256': model_hash,
                'input_evidence_sha256': evidence_hash, 'angle_change_bound_rad': contract['maximum_angle_change_rad'],
                'changed_frames': changed, 'compute_budget_exhausted': budget_exhausted,
                'scope': 'Derived joint angles fitted to source model estimates; original confidence is not fit accuracy'}
    candidate['reference_refinement'] = metadata
    for row in candidate.get('reference_evidence', []):
        row['reference_derivation'] = {'method': contract['method'], 'source_confidence_unchanged': True}
    audit = {'schema_version': 1, 'status': 'partial_budget_exhausted' if budget_exhausted else 'complete',
             'promoted': False, 'contract': deepcopy(contract), 'input_model_sha256': model_hash,
             'input_evidence_sha256': evidence_hash, 'output_model_sha256': canonical_hash(candidate),
             'evidence_hash_encoding': 'Canonical JSON; nonfinite raw values retained as tagged strings',
             'source_sha256': spec['source_sha256'], 'frames': rows, 'changed_frames': changed,
             'unused_evidence_ids': sorted(set(by_id) - {row['evidence_id'] for row in rows if row['evidence_id'] is not None}),
             'seconds': time.monotonic()-started, 'physics_steps': 0,
             'scope': 'Construction evidence fit, not independent accuracy, anatomy, forces, contact repair, or balance',
             'fixed_fields': ['root_reference', 'mobility', 'lengths', 'body_frame', 'motor_limits', 'dt'],
             'loss_excludes': ['floor_distance', 'contact_force', 'evaluation_labels', 'policy_return', 'root_translation']}
    return candidate, audit
