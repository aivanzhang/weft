"""Image/geometry arm proposals with pointwise, nonlinear collision nonregression."""

from copy import deepcopy
import hashlib
import json
import time

import numpy as np

from weft.arm_equipment_refinement import _Geometry, _image_summary, _schedule, _validate_inputs
from weft.arm_reference_alignment import POINT_NAMES, perspective_points
from weft.io import canonical_hash
from weft.world_evidence import _check_tree, _nonfinite, _unique_object


CONTRACT = {
    'method': 'hard-nonregression-arm-alignment-v1', 'maximum_reference_poses': 900,
    'maximum_grid_poses': 1801, 'sample_hz': 30, 'maximum_equipment_boxes': 12,
    'maximum_fit_seconds': 30., 'maximum_finalization_seconds': 8.,
    'maximum_helper_seconds': 38., 'deadline_scope': 'Cooperative checks between bounded native calls; no native-call preemption claim',
    'sweeps': 6, 'maximum_backtracks': 6,
    'collision_sigma_schedule': [.01, .01, .003, .003, .001, .001],
    'collision_margin_model_units': .0001, 'maximum_update_rad': .10,
    'minimum_confidence': .5, 'image_sigma_px': 6., 'inferred_depth_sigma_model_units': .06,
    'weak_endpoint_depth_weight': .1, 'old_angle_sigma_rad': .35,
    'adjacent_correction_sigma_rad': .12, 'maximum_temporal_prior_gap_s': .25,
    'maximum_angle_correction_rad': .30, 'maximum_endpoint_depth_change_model_units': .08,
    'maximum_correction_rate_rad_s': 1.5, 'numerical_tolerance': 1e-8,
    'linear_system_ridge': 1e-8, 'projection_cycles': 48, 'projection_tolerance': 1e-10,
    'initialization': 'Original references only; first reference frozen; no stacked repairs',
    'constraint': 'Every affected nonlinear pair gap >= max(min(0, original gap), min(0, last accepted gap)) minus numerical roundoff',
    'weak_observations': 'Unchanged raw status/confidence, zero image objective weight; changes are inferred geometry/depth/temporal proposals',
    'geometry_scope': 'Original fixed nominal boxes only; no compound-shape, contact, grasp or physical admission',
    'objective_excludes': ['task_reward', 'policy_return', 'held_out_evidence', 'grasp', 'support_force'],
    'physics_steps': 0,
}
MAX_OUTPUT_BYTES = 8 * 1024 * 1024


class _BudgetExpired(TimeoutError):
    pass


def _check_deadline(deadline, stage):
    if time.monotonic() >= deadline:
        raise _BudgetExpired(stage)


def _project_step(hessian, gradient, matrix, lower, *, cycles=48, tolerance=1e-10):
    """Bounded dual coordinate projection; a nonconverged step is never accepted."""
    direction = np.linalg.solve(hessian, -gradient)
    matrix, lower = np.asarray(matrix, dtype=float), np.asarray(lower, dtype=float)
    if not len(lower):
        return direction, {'converged': True, 'cycles': 0, 'maximum_violation': 0.}
    inverse_rows = np.linalg.solve(hessian, matrix.T).T
    norms = np.einsum('ij,ij->i', matrix, inverse_rows)
    active = norms > 1e-20
    if np.any(lower[~active] > tolerance):
        return np.zeros_like(direction), {'converged': False, 'cycles': 0, 'maximum_violation': float(np.max(lower[~active]))}
    multipliers = np.zeros(len(lower))
    count = 0
    for count in range(1, cycles + 1):
        maximum_change = 0.
        for i in np.flatnonzero(active):
            change = max(0., multipliers[i] + (lower[i] - matrix[i] @ direction) / norms[i]) - multipliers[i]
            direction += change * inverse_rows[i]
            multipliers[i] += change
            maximum_change = max(maximum_change, abs(change) * float(np.max(abs(inverse_rows[i]))))
        violation = max(0., float(np.max(lower - matrix @ direction)))
        if violation <= tolerance and maximum_change <= tolerance:
            break
    violation = max(0., float(np.max(lower - matrix @ direction)))
    converged = bool(violation <= tolerance and maximum_change <= tolerance)
    return (direction if converged else np.zeros_like(direction)), {
        'converged': converged, 'cycles': count, 'maximum_violation': violation}


def _nonregression(before, current, trial, tolerance):
    threshold = np.maximum(np.minimum(0., before), np.minimum(0., current))
    return bool(np.all(trial >= threshold - tolerance))


def _geometry_state(engine, pairs, angles, root, derivatives=False):
    engine.set(angles, root)
    points, point_jac = engine.points(derivatives)
    values = [engine.distance(g, other, jacobian=derivatives) for g, other, _ in pairs]
    return points, point_jac, np.array([v[0] for v in values]), np.array([v[1] for v in values]) if derivatives else None


def _arm_step_bounds(index, changes, lower, upper, times):
    lo, hi = lower[index].copy(), upper[index].copy()
    for neighbor in (index - 1, index + 1):
        if 0 <= neighbor < len(times):
            radius = CONTRACT['maximum_correction_rate_rad_s'] * abs(times[index] - times[neighbor])
            lo, hi = np.maximum(lo, changes[neighbor] - radius), np.minimum(hi, changes[neighbor] + radius)
    lo = np.maximum(lo - changes[index], -CONTRACT['maximum_update_rad'])
    hi = np.minimum(hi - changes[index], CONTRACT['maximum_update_rad'])
    return lo, hi


def _evaluate_arm_pose(index, trial, sigma, derivatives=False, *, changes, rows, times, affected,
                       geometry, state, columns, camera_hypothesis, object_pair_count, tolerance, deadline):
    previous = changes[index].copy(); changes[index] = trial
    residuals, jacobians, matrix, bounds, states = [], [], [], [], []
    valid = True
    def add(residual, jac):
        residuals.extend(np.asarray(residual).ravel())
        if derivatives: jacobians.extend(np.asarray(jac).reshape(-1, 8))
    try:
        add(trial / CONTRACT['old_angle_sigma_rad'], np.eye(8) / CONTRACT['old_angle_sigma_rad'])
        for neighbor in (index - 1, index + 1):
            if 0 <= neighbor < len(rows) and abs(times[index] - times[neighbor]) <= CONTRACT['maximum_temporal_prior_gap_s']:
                add((trial - changes[neighbor]) / CONTRACT['adjacent_correction_sigma_rad'],
                    np.eye(8) / CONTRACT['adjacent_correction_sigma_rad'])
        for sample, weight in affected[index]:
            _check_deadline(deadline, 'optimization_samples')
            angles = sample['angles'].copy()
            angles[columns] += sum(changes[j] * w for j, w in sample['weights'])
            points, point_jac, gaps, gap_jac = state(geometry, angles, sample['root'], derivatives)
            states.append((sample, gaps))
            valid &= _nonregression(sample['before_gaps'], sample['current_gaps'], gaps, tolerance)
            depth = points[:, 1] - sample['points'][:, 1]
            valid &= bool(np.max(abs(depth)) <= CONTRACT['maximum_endpoint_depth_change_model_units'] + tolerance)
            if derivatives:
                threshold = np.maximum(np.minimum(0., sample['before_gaps']), np.minimum(0., sample['current_gaps']))
                matrix.extend(gap_jac * weight)
                bounds.extend(threshold - gaps - tolerance)
                depth_jac = point_jac[:, 1] * weight
                matrix.extend(depth_jac); bounds.extend(-CONTRACT['maximum_endpoint_depth_change_model_units'] - depth)
                matrix.extend(-depth_jac); bounds.extend(depth - CONTRACT['maximum_endpoint_depth_change_model_units'])
            for j, gap in enumerate(gaps[:object_pair_count]):
                if gap < CONTRACT['collision_margin_model_units']:
                    add([(CONTRACT['collision_margin_model_units'] - gap) / sigma],
                        -gap_jac[j] * weight / sigma if derivatives else None)
            if len(sample['weights']) == 1 and sample['weights'][0][0] == index:
                pixels, projection = perspective_points(points, camera_hypothesis, jacobian=derivatives)
                for j, point in enumerate(rows[index]['endpoints']):
                    if point['fit_weight']:
                        w = np.sqrt(point['fit_weight']) / CONTRACT['image_sigma_px']
                        add((pixels[j] - point['image_px']) * w, projection[j] @ point_jac[j] * w if derivatives else None)
                    w = np.sqrt(point['fit_weight'] or CONTRACT['weak_endpoint_depth_weight']) / CONTRACT['inferred_depth_sigma_model_units']
                    add([(points[j, 1] - point['depth_prior_model_units']) * w], point_jac[j, 1] * w if derivatives else None)
    except ValueError:
        valid = False
    finally:
        changes[index] = previous
    residuals = np.asarray(residuals)
    return float(residuals @ residuals / 2), residuals, np.asarray(jacobians), valid, matrix, bounds, states


def _propose(human_document, observation_document, *, human_entity_ref, camera_hypothesis,
             equipment_boxes, human_xml_document, context, projection_solver=None, curvature_restorer=None):
    started = context['started']; deadline = started + CONTRACT['maximum_fit_seconds']
    if type(human_document) is not bytes or len(human_document) > 8 * 1024 * 1024:
        raise ValueError('A bounded exact human document is required')
    try:
        spec = json.loads(human_document, object_pairs_hook=_unique_object, parse_constant=_nonfinite)
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError) as error:
        raise ValueError('Invalid constrained-arm human JSON') from error
    _check_tree(spec)
    if 'constrained_arm_alignment' in spec:
        raise ValueError('Stacked arm proposals are unsupported')
    model, source, source_rows, rows = _validate_inputs(spec, observation_document, human_entity_ref,
                                                       camera_hypothesis, equipment_boxes, human_xml_document)
    candidate = deepcopy(spec)
    context.update(candidate=candidate, original=spec, source_rows=source_rows, rows=rows)
    _check_deadline(deadline, 'input_validation')
    geometry = _Geometry(model, human_xml_document, equipment_boxes)
    _check_deadline(deadline, 'initial_geometry')
    names, columns = geometry.names, geometry.columns
    times = np.array([row['time_s'] for row in model['reference']])
    original = np.array([[row['joints'][name] for name in names] for row in model['reference']])
    roots = np.array([row['translation'] for row in model['root_reference']])
    schedule = _schedule(times)
    context['scheduled_samples'] = len(schedule)
    pairs = [(g, box, obj) for g in geometry.human_geoms for obj, box in geometry.boxes.items()]
    object_pair_count = len(pairs)
    pairs += [(g, geometry.floor, None) for g in geometry.human_geoms]
    pair_columns = [{'human_geom': geometry.model.geom(g).name, 'object_id': obj,
                     'kind': 'human_equipment' if obj is not None else 'human_floor'} for g, _, obj in pairs]

    def state(engine, angles, root, derivatives=False):
        return _geometry_state(engine, pairs, angles, root, derivatives)

    samples, by_time = [], {}
    affected = [[] for _ in rows]
    for entry in schedule:
        _check_deadline(deadline, 'initial_samples')
        key = round(entry['time_s'], 10)
        if key not in by_time:
            weights = entry['weights']
            angles = sum(original[i] * w for i, w in weights)
            root = sum(roots[i] * w for i, w in weights)
            points, _, gaps, _ = state(geometry, angles, root)
            sample = {'weights': weights, 'angles': angles, 'root': root, 'points': points,
                      'before_gaps': gaps.copy(), 'current_gaps': gaps.copy()}
            by_time[key] = len(samples); samples.append(sample)
            for index, weight in weights:
                affected[index].append((sample, weight))
    lower = np.maximum(-CONTRACT['maximum_angle_correction_rad'],
                       np.array([geometry.limits[name][0] for name in geometry.arm_names]) - original[:, columns])
    upper = np.minimum(CONTRACT['maximum_angle_correction_rad'],
                       np.array([geometry.limits[name][1] for name in geometry.arm_names]) - original[:, columns])
    lower[0] = upper[0] = 0.
    changes = np.zeros((len(rows), 8)); tolerance = CONTRACT['numerical_tolerance']
    for i, row in enumerate(rows):
        _check_deadline(deadline, 'initial_image_projection')
        base = samples[by_time[round(times[i], 10)]]['points']
        pixels, _ = perspective_points(base, camera_hypothesis)
        for j, point in enumerate(row['endpoints']):
            point.update(depth_prior_model_units=(point['raw_model_relative_depth'] + roots[i, 1]
                         if point['raw_model_relative_depth'] is not None else float(base[j, 1])),
                         depth_prior_status='inferred_model_world' if point['raw_model_relative_depth'] is not None else 'original_model_fallback',
                         model_before=base[j].tolist(), projected_before_px=pixels[j].tolist(),
                         error_before_px=float(np.linalg.norm(pixels[j] - point['image_px'])) if point['image_px'] is not None else None)
    first = samples[by_time[round(times[0], 10)]]['before_gaps']
    witnesses = [pair_columns[j] | {'gap_model_units': float(gap), 'reason': 'frozen_first_reference'}
                 for j, gap in enumerate(first[:object_pair_count]) if gap < -tolerance]

    def evaluate(index, trial, sigma, derivatives=False):
        return _evaluate_arm_pose(index, trial, sigma, derivatives, changes=changes, rows=rows, times=times,
            affected=affected, geometry=geometry, state=state, columns=columns, camera_hypothesis=camera_hypothesis,
            object_pair_count=object_pair_count, tolerance=tolerance, deadline=deadline)

    attempts = [{'accepted_updates': 0, 'nonlinear_rejections': 0, 'projection_failures': 0} for _ in rows]
    sweeps, projections, exhausted = [], [], False
    context.update(projections=projections, attempts=attempts)
    if not witnesses:
        for sweep, sigma in enumerate(CONTRACT['collision_sigma_schedule']):
            accepted = 0
            order = range(1, len(rows)) if sweep % 2 == 0 else range(len(rows) - 1, 0, -1)
            for i in order:
                if time.monotonic() >= deadline: exhausted = True; break
                lo, hi = _arm_step_bounds(i, changes, lower, upper, times)
                cost, residual, jac, valid, matrix, bounds, _ = evaluate(i, changes[i], sigma, True)
                if not valid: raise ValueError('Accepted iterate lost its nonlinear feasibility')
                matrix.extend(np.eye(8)); bounds.extend(lo)
                matrix.extend(-np.eye(8)); bounds.extend(-hi)
                project = _project_step if projection_solver is None else projection_solver
                step, projection = project(jac.T @ jac + np.eye(8) * CONTRACT['linear_system_ridge'],
                    jac.T @ residual, matrix, bounds, cycles=CONTRACT['projection_cycles'], tolerance=CONTRACT['projection_tolerance'])
                projections.append(projection | {'reference_index': i, 'sweep': sweep})
                if not projection['converged']:
                    attempts[i]['projection_failures'] += 1; continue
                step_accepted, smallest_invalid, completed_backtracks = False, False, 0
                for backtrack in range(CONTRACT['maximum_backtracks']):
                    if time.monotonic() >= deadline: exhausted = True; break
                    trial = changes[i] + step * .5 ** backtrack
                    if np.max(abs(trial - changes[i])) < 1e-10: break
                    trial_cost, _, _, valid, _, _, states = evaluate(i, trial, sigma)
                    completed_backtracks += 1; smallest_invalid = not valid
                    if not valid:
                        attempts[i]['nonlinear_rejections'] += 1
                    if valid and trial_cost < cost - 1e-12:
                        step_accepted = True
                        changes[i] = trial; accepted += 1; attempts[i]['accepted_updates'] += 1
                        for j, name in enumerate(geometry.arm_names):
                            candidate['reference'][i]['joints'][name] = float(original[i, columns[j]] + trial[j])
                        for sample, gaps in states: sample['current_gaps'] = gaps
                        break
                if (curvature_restorer is not None and not exhausted and not step_accepted and smallest_invalid and
                        completed_backtracks == CONTRACT['maximum_backtracks']):
                    restored = curvature_restorer(index=i, sweep=sweep, sigma=sigma, baseline=changes[i].copy(),
                        direction=step, lower=lo, upper=hi, baseline_cost=cost,
                        evaluate=lambda trial, derivatives=False: evaluate(i, trial, sigma, derivatives),
                        affected_count=len(affected[i]), pair_count=len(pairs), deadline=deadline)
                    attempts[i]['nonlinear_rejections'] += restored['nonlinear_rejections']
                    if restored['trial'] is not None:
                        trial, states = restored['trial'], restored['states']
                        changes[i] = trial; accepted += 1; attempts[i]['accepted_updates'] += 1
                        for j, name in enumerate(geometry.arm_names):
                            candidate['reference'][i]['joints'][name] = float(original[i, columns[j]] + trial[j])
                        for sample, gaps in states: sample['current_gaps'] = gaps
                if exhausted: break
            sweeps.append({'sweep': sweep, 'collision_sigma_model_units': sigma, 'accepted_updates': accepted})
            if exhausted: break
    fit_seconds = time.monotonic() - started
    final_deadline = min(started + CONTRACT['maximum_helper_seconds'],
                         time.monotonic() + CONTRACT['maximum_finalization_seconds'])
    for i, row in enumerate(candidate['reference']):
        for j, name in enumerate(geometry.arm_names):
            if changes[i, j]: row['joints'][name] = float(original[i, columns[j]] + changes[i, j])
    # The final audit rebuilds geometry and never treats optimization caches as evidence.
    _check_deadline(final_deadline, 'final_geometry')
    audit_geometry = _Geometry(model, human_xml_document, equipment_boxes)
    checks, maximum_depth = [], 0.
    context['checks'] = checks
    for entry in schedule:
        _check_deadline(final_deadline, 'final_audit_samples')
        weights = entry['weights']; old_angles = sum(original[i] * w for i, w in weights)
        angles = old_angles.copy(); angles[columns] += sum(changes[i] * w for i, w in weights)
        root = sum(roots[i] * w for i, w in weights)
        before, _, before_gaps, _ = state(audit_geometry, old_angles, root)
        after, _, after_gaps, _ = state(audit_geometry, angles, root)
        depth = float(np.max(abs(after[:, 1] - before[:, 1]))); maximum_depth = max(maximum_depth, depth)
        checks.append({key: entry[key] for key in ('kind', 'time_s', 'source_reference_index')} | {
            'reference_indices': [i for i, _ in weights], 'before_gaps': before_gaps.tolist(), 'after_gaps': after_gaps.tolist(),
            'maximum_endpoint_depth_change_model_units': depth,
            'joint_limits_valid': all(audit_geometry.limits[name][0] - tolerance <= angles[j] <= audit_geometry.limits[name][1] + tolerance
                                      for j, name in enumerate(names))})
        if entry['source_reference_index'] is not None:
            i = entry['source_reference_index']; pixels, _ = perspective_points(after, camera_hypothesis)
            changed = bool(np.max(abs(changes[i])) > tolerance)
            rows[i].update(angle_correction_rad=dict(zip(geometry.arm_names, changes[i].tolist())), **attempts[i],
                           proposal_status='first_pose_frozen' if i == 0 else 'changed_inferred_proposal' if changed else 'unchanged_no_accepted_step')
            for j, point in enumerate(rows[i]['endpoints']):
                point.update(model_after=after[j].tolist(), projected_after_px=pixels[j].tolist(),
                    error_after_px=float(np.linalg.norm(pixels[j] - point['image_px'])) if point['image_px'] is not None else None,
                    depth_change_model_units=float(after[j, 1] - before[j, 1]),
                    correction_scope='confident_source_image_plus_authored_priors' if point['fit_weight'] else 'inferred_geometry_depth_temporal_only_not_measured')
    image_summary = _image_summary(rows)
    per_joint = {}
    for name in names:
        _check_deadline(final_deadline, 'final_joint_metrics')
        points = [p for row in rows for p in row['endpoints'] if p['joint'] == name]
        per_joint[name] = {'changed_references': sum(a['joints'][name] != b['joints'][name] for a, b in zip(spec['reference'], candidate['reference'])),
                           'image_status': 'reported_endpoint' if points else 'joint_without_direct_endpoint_image_fit'}
        if points:
            for group, selected in (('confident', [p for p in points if p['fit_weight']]),
                                    ('all_finite', [p for p in points if p['image_px'] is not None])):
                per_joint[name][group] = {'count': len(selected), **{side: float(np.sqrt(np.mean([p['error_' + side + '_px'] ** 2 for p in selected])))
                    if selected else None for side in ('before', 'after')}}
    # Wrists are endpoints controlled by shoulder/elbow angles, not independent joints.
    for name in ('left_wrist', 'right_wrist'):
        points = [p for row in rows for p in row['endpoints'] if p['joint'] == name]
        per_joint[name] = {'image_status': 'reported_endpoint_without_independent_joint'}
        for group, selected in (('confident', [p for p in points if p['fit_weight']]),
                                ('all_finite', [p for p in points if p['image_px'] is not None])):
            per_joint[name][group] = {'count': len(selected), **{side: float(np.sqrt(np.mean([p['error_' + side + '_px'] ** 2 for p in selected])))
                if selected else None for side in ('before', 'after')}}
    before = np.asarray([row['before_gaps'] for row in checks]); after = np.asarray([row['after_gaps'] for row in checks])
    if not all(name in per_joint and 'confident' in per_joint[name] for name in POINT_NAMES):
        raise ValueError('Final image gate omitted an arm endpoint')
    rate = float(np.max(abs(np.diff(changes, axis=0)) / np.diff(times)[:, None]))
    fixed = {key: candidate.get(key) == value for key, value in spec.items() if key != 'reference'}
    fixed['non_arm_joints'] = all(a['joints'][name] == b['joints'][name] for a, b in zip(spec['reference'], candidate['reference'])
                                 for name in names if name not in geometry.arm_names)
    fixed['reference_times_and_count'] = len(candidate['reference']) == len(spec['reference']) and all(a['time_s'] == b['time_s'] for a, b in zip(spec['reference'], candidate['reference']))
    gates = {'complete_bounded_solve': not exhausted and not witnesses,
             'first_reference_frozen': candidate['reference'][0] == spec['reference'][0],
             'fixed_contracts': all(fixed.values()), 'joint_limits': all(row['joint_limits_valid'] for row in checks),
             'angle_correction': float(np.max(abs(changes))) <= CONTRACT['maximum_angle_correction_rad'] + tolerance,
             'endpoint_depth_change': maximum_depth <= CONTRACT['maximum_endpoint_depth_change_model_units'] + tolerance,
             'correction_rate': rate <= CONTRACT['maximum_correction_rate_rad_s'] + tolerance,
             'pointwise_collision_nonregression': _nonregression(before, before, after, tolerance),
             'no_remaining_equipment_penetration': bool(np.all(after[:, :object_pair_count] >= -tolerance)),
             **{'image_rmse_' + name: result['count'] > 0 and result['after'] <= result['before'] + tolerance for name, result in image_summary.items()},
             'per_endpoint_confident_image_nonregression': all(per_joint[name]['confident']['count'] == 0 or
                 per_joint[name]['confident']['after'] <= per_joint[name]['confident']['before'] + tolerance for name in POINT_NAMES)}
    before_pen = np.maximum(0., -before[:, :object_pair_count]); after_pen = np.maximum(0., -after[:, :object_pair_count])
    geometry_summary = {'samples': len(checks), 'unique_kinematic_samples': len(samples), 'object_pair_samples': int(before_pen.size),
        'before_penetrating_pair_samples': int(np.count_nonzero(before_pen > tolerance)),
        'after_penetrating_pair_samples': int(np.count_nonzero(after_pen > tolerance)),
        'worsened_pair_samples': int(np.count_nonzero(after < np.minimum(0., before) - tolerance)),
        'new_equipment_penetrations': int(np.count_nonzero((before_pen <= tolerance) & (after_pen > tolerance))),
        'before_maximum_penetration_model_units': float(before_pen.max()), 'after_maximum_penetration_model_units': float(after_pen.max())}
    failures = [name for name, passed in gates.items() if not passed]
    status = 'partial_budget_exhausted' if exhausted else 'rejected_construction_proposal' if failures else 'unadmitted_construction_proposal'
    report = {'schema_version': 1, 'profile': CONTRACT['method'], 'status': status, 'contract': deepcopy(CONTRACT),
        'human_document_sha256': hashlib.sha256(human_document).hexdigest(), 'input_model_sha256': canonical_hash(spec),
        'candidate_model_sha256': canonical_hash(candidate), 'construction_sha256': hashlib.sha256(observation_document).hexdigest(),
        'frozen_human_xml_sha256': hashlib.sha256(human_xml_document).hexdigest(), 'human_entity_ref': deepcopy(human_entity_ref),
        'camera_hypothesis': deepcopy(camera_hypothesis), 'equipment_boxes': deepcopy(equipment_boxes),
        'source_clock': deepcopy(source['time']), 'source_contract': deepcopy(source['source']),
        'source_slots': [{key: row[key] for key in ('frame', 'time_s', 'partition', 'admission', 'visibility', 'evidence_refs')}
                         for _, row in sorted(source_rows.items())], 'references': rows, 'image_summary': image_summary,
        'per_joint_summary': per_joint, 'pair_columns': pair_columns, 'geometry_checks': checks, 'geometry_summary': geometry_summary,
        'fixed_contracts': fixed, 'gates': gates, 'failures': failures, 'sweeps': sweeps, 'projections': projections,
        'maximum_angle_correction_rad': float(np.max(abs(changes))), 'maximum_endpoint_depth_change_model_units': maximum_depth,
        'maximum_correction_rate_rad_s': rate, 'changed_reference_poses': int(np.count_nonzero(np.max(abs(changes), axis=1) > tolerance)),
        'frozen_infeasibility_witnesses': witnesses, 'fit_seconds': fit_seconds, 'seconds': time.monotonic() - started,
        'independent_audit_status': 'complete', 'physics_steps': 0, 'physical_validation': 'unverified',
        'geometry_admission': 'unadmitted_pending_full_source_current_geometry_audit',
        'promoted': False, 'limit': 'Finite construction samples and authored boxes do not prove continuous clearance, measured depth, grasp, support or physical accuracy. Rejection is not global infeasibility.'}
    result = {'status': status, 'human_candidate': candidate, 'report': report}
    _check_deadline(final_deadline, 'output_serialization')
    if len(json.dumps(result, allow_nan=False, separators=(',', ':')).encode()) > MAX_OUTPUT_BYTES:
        raise ValueError('Complete proposal exceeds its output byte bound')
    _check_deadline(final_deadline, 'output_bound_check')
    return result


def _run_alignment(human_document, observation_document, *, human_entity_ref, camera_hypothesis,
                   equipment_boxes, human_xml_document, projection_solver=None, curvature_restorer=None):
    context = {'started': time.monotonic()}
    try:
        return _propose(human_document, observation_document, human_entity_ref=human_entity_ref,
                        camera_hypothesis=camera_hypothesis, equipment_boxes=equipment_boxes,
                        human_xml_document=human_xml_document, context=context,
                        projection_solver=projection_solver, curvature_restorer=curvature_restorer)
    except _BudgetExpired as error:
        candidate, original = context['candidate'], context['original']
        rows = deepcopy(context['rows'])
        for row in rows:
            row['proposal_status'] = 'partial_inferred_proposal_not_audited'
            row['independent_audit_status'] = 'incomplete'
        changed = sum(a != b for a, b in zip(candidate['reference'], original['reference']))
        report = {'schema_version': 1, 'profile': CONTRACT['method'], 'status': 'partial_budget_exhausted',
            'contract': deepcopy(CONTRACT), 'deadline_stage': str(error), 'independent_audit_status': 'incomplete',
            'geometry_admission': 'unverified_incomplete_audit', 'physical_validation': 'unverified', 'promoted': False,
            'human_document_sha256': hashlib.sha256(human_document).hexdigest(),
            'construction_sha256': hashlib.sha256(observation_document).hexdigest(),
            'human_entity_ref': deepcopy(human_entity_ref), 'camera_hypothesis': deepcopy(camera_hypothesis),
            'equipment_boxes': deepcopy(equipment_boxes), 'human_candidate_sha256': canonical_hash(candidate),
            'references': rows, 'source_slots': [{key: row[key] for key in ('frame', 'time_s', 'partition', 'admission', 'visibility', 'evidence_refs')}
                                               for _, row in sorted(context['source_rows'].items())],
            'image_summary': {}, 'per_joint_summary': {name: {'status': 'unavailable_incomplete_audit'}
                for name in dict.fromkeys([*original['reference'][0]['joints'], *POINT_NAMES])},
            'geometry_summary': {'audit_status': 'incomplete', 'completed_samples': len(context.get('checks', [])),
                                 'scheduled_samples': context.get('scheduled_samples'), 'partial_gaps_not_admission_evidence': True},
            'failures': ['cooperative_deadline:' + str(error)], 'gates': {'complete_bounded_solve': False, 'complete_independent_audit': False},
            'changed_reference_poses': changed, 'projections': context.get('projections', []),
            'maximum_angle_correction_rad': None, 'maximum_endpoint_depth_change_model_units': None,
            'maximum_correction_rate_rad_s': None, 'fit_seconds': None, 'seconds': time.monotonic() - context['started'],
            'physics_steps': 0, 'limit': 'Input rows and accepted candidate retained; incomplete image/geometry checks are unavailable, not passed.'}
        result = {'status': 'partial_budget_exhausted', 'human_candidate': candidate, 'report': report}
        if len(json.dumps(result, allow_nan=False, separators=(',', ':')).encode()) > MAX_OUTPUT_BYTES:
            raise ValueError('Partial proposal exceeds its separate output bound')
        return result


def propose_constrained_arm_alignment(human_document: bytes, observation_document: bytes, *,
                                      human_entity_ref, camera_hypothesis, equipment_boxes,
                                      human_xml_document: bytes):
    """Return an unadmitted proposal or explicit partial receipt, retaining every input row."""
    return _run_alignment(human_document, observation_document, human_entity_ref=human_entity_ref,
        camera_hypothesis=camera_hypothesis, equipment_boxes=equipment_boxes, human_xml_document=human_xml_document)
