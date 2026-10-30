"""Construction-only arm-angle repair; image agreement never establishes hand support."""

from copy import deepcopy
import hashlib
import json
import math
import time

import mujoco
import numpy as np

from weft.body_refinement import JOINT_LANDMARKS, _kinematics
from weft.human import HumanEnv, validate_human
from weft.io import canonical_hash
from weft.world_evidence import MAX_INPUT_BYTES, _check_tree, _nonfinite, _unique_object


CONTRACT = {
    'method': 'bounded-perspective-full-arm-reference-v1', 'maximum_reference_poses': 900,
    'maximum_fit_seconds': 30., 'maximum_iterations_per_frame': 12, 'maximum_backtracks': 6,
    'minimum_confidence': .5, 'image_sigma_px': 6., 'inferred_depth_sigma_model_units': .06,
    'old_angle_sigma_rad': .35, 'adjacent_correction_sigma_rad': .12,
    'maximum_angle_correction_rad': .30, 'maximum_endpoint_depth_change_model_units': .08,
    'maximum_correction_rate_rad_s': 1.5, 'maximum_temporal_prior_gap_s': .25,
    'minimum_camera_depth_model_units': .05, 'linear_system_ridge': 1e-8,
    'first_reference': 'frozen', 'unavailable_side': 'unchanged when either elbow or wrist is unavailable/low confidence',
    'depth_scope': 'Raw pose-model hip-centered depth is an uncertain prior, not measured depth; original articulated depth is the explicit fallback',
    'objective_excludes': ['equipment', 'contacts', 'floor_distance', 'grasp', 'task_reward', 'policy_return', 'held_out_evidence'],
}
POINT_NAMES = ('left_elbow', 'left_wrist', 'right_elbow', 'right_wrist')
POINT_INDICES = tuple(list(JOINT_LANDMARKS).index(name) for name in POINT_NAMES)


def _vector(value, n, name, bound=10000.):
    if (not isinstance(value, (list, tuple)) or len(value) != n
            or any(type(x) not in (float, int) or not math.isfinite(x) or abs(x) > bound for x in value)):
        raise ValueError(f'{name} requires {n} finite bounded coordinates')
    return np.asarray(value, dtype=float)


def perspective_points(points, camera, *, jacobian=False):
    """Return source-pixel positions and optional derivatives in the unchanged body frame."""
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1:] != (3,) or not np.isfinite(points).all():
        raise ValueError('Perspective points require finite 3D rows')
    depth = camera['distance_model_units'] + points[:, 1]
    if (depth <= CONTRACT['minimum_camera_depth_model_units']).any():
        raise ValueError('Arm candidate is behind or too near the fixed camera')
    focal, base = camera['focal_px'], camera['world_anchor_model_units'][2]
    pixels = np.asarray(camera['pixel_origin']) + focal * np.column_stack((points[:, 0], base - points[:, 2])) / depth[:, None]
    derivative = None
    if jacobian:
        derivative = np.zeros((len(points), 2, 3))
        derivative[:, 0, 0] = focal / depth
        derivative[:, 0, 1] = -focal * points[:, 0] / depth ** 2
        derivative[:, 1, 1] = -focal * (base - points[:, 2]) / depth ** 2
        derivative[:, 1, 2] = -focal / depth
    return pixels, derivative


def _construction_source(raw, entity, source_hash):
    if type(raw) is not bytes or not 1 <= len(raw) <= MAX_INPUT_BYTES:
        raise ValueError('Arm repair requires bounded filtered construction bytes')
    try:
        document = json.loads(raw, object_pairs_hook=_unique_object, parse_constant=_nonfinite)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError('Invalid construction JSON') from error
    _check_tree(document)
    if (document.get('profile') != 'world-construction-evidence-v1' or document.get('schema_version') != 1
            or document.get('allowed_partitions') != ['construction'] or document.get('physical_validation') != 'unverified'):
        raise ValueError('Only construction-partition evidence may influence arm repair')
    if not isinstance(entity, dict) or set(entity) != {'document_sha256', 'object_id'}:
        raise ValueError('Human source identity must be document qualified')
    sources = [row for row in document['sources'] if row['document_sha256'] == entity['document_sha256']]
    if len(sources) != 1 or sources[0]['source']['sha256'] != source_hash:
        raise ValueError('Human construction source identity differs')
    source = sources[0]
    rows = [row for row in source['observations'] if row['object_id'] == entity['object_id']]
    count = source['source']['decoded_frames']
    if (not 1 <= count <= 1800 or len(rows) != count or sorted(row['frame'] for row in rows) != list(range(count))
            or source['source'].get('decode_interval', [0, count - 1]) != [0, count - 1]):
        raise ValueError('Arm repair must retain complete original source slots')
    evidence = {row['id']: row for row in source['evidence']}
    for row in rows:
        if row['entity_ref'] != entity:
            raise ValueError('Human observation identity differs from its qualified entity')
        if row['admission'] == 'included':
            if row['partition'] != 'construction':
                raise ValueError('Non-construction observation was admitted')
            for reference in row['evidence_refs']:
                item = evidence.get(reference['evidence_id'])
                if (reference['document_sha256'] != entity['document_sha256'] or item is None
                        or item['object_id'] != row['object_id'] or item['frame'] != row['frame'] or item['time_s'] != row['time_s']):
                    raise ValueError('Construction evidence/frame/clock reference differs')
        elif row['admission'] != 'withheld' or row['measurements'] or row['evidence_refs']:
            raise ValueError('Withheld slots cannot expose construction values')
    return source, {row['frame']: row for row in rows}


def _endpoint(row, name):
    result = {'joint': name, 'image_px': None, 'confidence': None, 'status': 'unavailable', 'raw_model_relative_depth': None}
    if row['admission'] != 'included' or row['visibility'] not in ('detected', 'partial'):
        return result
    measurement = row['measurements'].get('landmarks', {})
    if (measurement.get('coordinate_system'), measurement.get('unit'), measurement.get('status')) != ('image', 'pixels', 'inferred'):
        return result
    points = {point['index']: point for point in measurement.get('value', [])}
    point = points.get(JOINT_LANDMARKS[name])
    if point is None:
        return result
    try:
        pixel = _vector(point['position'], 2, 'Image endpoint')
        confidence = _vector([point['visibility'], point['presence']], 2, 'Image endpoint confidence', 1.)
    except (KeyError, TypeError, ValueError):
        result['status'] = 'invalid_endpoint'; return result
    if np.any(confidence < 0):
        result['status'] = 'invalid_confidence'; return result
    result.update(image_px=pixel.tolist(), confidence=float(confidence.min()), status='available' if confidence.min() >= .5 else 'low_confidence')
    world = row['measurements'].get('world_landmarks', {})
    if (world.get('coordinate_system'), world.get('unit'), world.get('status')) == ('model_world', 'model_meters', 'inferred'):
        values = {point['index']: point for point in world.get('value', [])}
        try:
            endpoint = _vector(values[JOINT_LANDMARKS[name]]['position'], 3, 'Model-world endpoint', 10.)
            hips = np.mean([_vector(values[index]['position'], 3, 'Model-world hip', 10.) for index in (23, 24)], axis=0)
            result['raw_model_relative_depth'] = float(endpoint[2] - hips[2])
        except (KeyError, TypeError, ValueError):
            pass
    return result


def align_arm_reference(spec, construction_evidence: bytes, *, human_entity_ref, camera_hypothesis):
    """Return an unpromoted candidate and complete derivation audit without filesystem writes."""
    started = time.monotonic()
    model = validate_human(spec)
    if model.get('profile') != 'exercise-v3' or 'camera_reference' in model or 'arm_reference_alignment' in model:
        raise ValueError('Arm repair requires an original uncalibrated mobile body')
    references, provenance = model['reference'], model.get('reference_evidence')
    if not 3 <= len(references) <= CONTRACT['maximum_reference_poses'] or not isinstance(provenance, list) or len(provenance) != len(references):
        raise ValueError('Arm repair requires all bounded source-linked body references')
    source, source_rows = _construction_source(construction_evidence, human_entity_ref, model['source_sha256'])
    camera = deepcopy(camera_hypothesis)
    for name in ('distance_model_units', 'focal_px', 'original_scale_at_anchor_model_units_per_pixel'):
        if type(camera.get(name)) not in (int, float) or not math.isfinite(camera[name]) or camera[name] <= 0:
            raise ValueError('Camera requires fixed finite distance, focal length and scale')
    anchor = _vector(camera.get('world_anchor_model_units'), 3, 'Camera anchor')
    _vector(camera.get('pixel_origin'), 2, 'Camera pixel origin')
    if (camera.get('world_axes') != ['x_right', 'y_depth', 'z_up'] or not np.array_equal(anchor, [0., 0., model['mobility']['base_height']])
            or not .1 <= camera['distance_model_units'] <= 50 or
            not math.isclose(camera['focal_px'], camera['distance_model_units'] / camera['original_scale_at_anchor_model_units_per_pixel'], abs_tol=1e-9, rel_tol=0)):
        raise ValueError('Camera must preserve the original body anchor, axes and torso scale')
    times = np.array([row['time_s'] for row in references])
    width, height = source['image']['width_px'], source['image']['height_px']
    rows, eligibility = [], []
    for index, (reference, meta) in enumerate(zip(references, provenance)):
        identifier = meta.get('evidence_id', '')
        prefix, separator, suffix = identifier.rpartition(':frame:')
        if not separator or prefix != model['source_sha256'] or not suffix.isdigit():
            raise ValueError('Every body reference must retain its original source-frame identity')
        frame = int(suffix); observed = source_rows.get(frame)
        if (observed is None or meta['time_s'] != reference['time_s'] or observed['time_s'] != reference['time_s']
                or model['root_reference'][index]['time_s'] != reference['time_s']):
            raise ValueError('Body/source reference clocks differ')
        expected = {'document_sha256': human_entity_ref['document_sha256'], 'evidence_id': identifier}
        if observed['admission'] == 'included' and expected not in observed['evidence_refs']:
            raise ValueError('Body reference does not resolve to its exact construction observation')
        details = [_endpoint(observed, name) for name in POINT_NAMES]
        for detail in details:
            if detail['image_px'] is not None and not (0 <= detail['image_px'][0] <= width and 0 <= detail['image_px'][1] <= height):
                detail['status'] = 'outside_source_image'
        eligible = [all(point['status'] == 'available' for point in details[offset:offset + 2]) for offset in (0, 2)]
        eligibility.append(eligible)
        rows.append({'reference_index': index, 'frame': frame, 'evidence_id': identifier, 'time_s': reference['time_s'],
                     'source_admission': observed['admission'], 'endpoints': details, 'eligible_sides': eligible,
                     'status': 'frozen_first_reference' if index == 0 else 'pending', 'iterations': 0, 'backtracks': 0})
    candidate = deepcopy(spec)
    env = HumanEnv(model)
    exhausted = False
    try:
        data = mujoco.MjData(env.model)
        arm_names = tuple(name for name in env.joint_names if 'shoulder' in name or name.endswith('_elbow'))
        arm_columns = np.array([env.joint_names.index(name) for name in arm_names])
        original = np.array([[row['joints'][name] for name in env.joint_names] for row in references])
        lower = np.maximum(-.30, np.array([env.joint_limits[name][0] for name in arm_names]) - original[:, arm_columns])
        upper = np.minimum(.30, np.array([env.joint_limits[name][1] for name in arm_names]) - original[:, arm_columns])
        for index, available in enumerate(eligibility):
            for column, name in enumerate(arm_names):
                if index == 0 or not available[0 if name.startswith('left_') else 1]:
                    lower[index, column] = upper[index, column] = 0.
        # Future fixed/limited rows constrain the correction before a gap, avoiding a final snap to zero.
        for index in range(len(rows) - 2, -1, -1):
            radius = CONTRACT['maximum_correction_rate_rad_s'] * (times[index + 1] - times[index])
            lower[index] = np.maximum(lower[index], lower[index + 1] - radius)
            upper[index] = np.minimum(upper[index], upper[index + 1] + radius)
        previous = np.zeros(len(arm_names))
        corrections = []

        def evaluate(angles, jacobian=False):
            positions, jac = _kinematics(env, data, angles, jacobian)
            world = positions[list(POINT_INDICES)] + data.site_xpos[env._site_ids['pelvis']]
            pixels, projection_jac = perspective_points(world, camera, jacobian=jacobian)
            if jacobian:
                xyz_jac = jac.reshape(12, 3, 19)[list(POINT_INDICES)][:, :, arm_columns]
                pixel_jac = np.einsum('nij,njk->nik', projection_jac, xyz_jac)
                return world, pixels, xyz_jac[:, 1], pixel_jac.reshape(8, 8)
            return world, pixels, None, None

        deadline = started + CONTRACT['maximum_fit_seconds']
        for index, row in enumerate(rows):
            env.mobility.reference(data, index)
            initial_world, initial_pixels, _, _ = evaluate(original[index])
            target = np.array([point['image_px'] if point['image_px'] is not None else initial_pixels[i]
                               for i, point in enumerate(row['endpoints'])])
            weights = np.array([point['confidence'] if row['eligible_sides'][i // 2] else 0.
                                for i, point in enumerate(row['endpoints'])])
            depth_target = np.array([point['raw_model_relative_depth'] + model['root_reference'][index]['translation'][1]
                                    if point['raw_model_relative_depth'] is not None else initial_world[i, 1]
                                    for i, point in enumerate(row['endpoints'])])
            temporal = index > 0 and times[index] - times[index - 1] <= CONTRACT['maximum_temporal_prior_gap_s']
            lo, hi = lower[index].copy(), upper[index].copy()
            if index:
                rate = CONTRACT['maximum_correction_rate_rad_s'] * (times[index] - times[index - 1])
                lo, hi = np.maximum(lo, previous - rate), np.minimum(hi, previous + rate)
            delta = np.clip(np.zeros(8), lo, hi)

            def objective(world, pixels, change):
                image = .5 * np.sum(weights[:, None] * ((pixels - target) / CONTRACT['image_sigma_px']) ** 2)
                depth = .5 * np.sum(weights * ((world[:, 1] - depth_target) / CONTRACT['inferred_depth_sigma_model_units']) ** 2)
                prior = .5 * np.sum((change / CONTRACT['old_angle_sigma_rad']) ** 2)
                smooth = .5 * np.sum(((change - previous) / CONTRACT['adjacent_correction_sigma_rad']) ** 2) if temporal else 0.
                return float(image + depth + prior + smooth)

            row['objective_before'] = objective(initial_world, initial_pixels, np.zeros(8))
            if index and np.any(weights) and not exhausted:
                row['status'] = 'bounded_fit'
                for iteration in range(CONTRACT['maximum_iterations_per_frame']):
                    if time.monotonic() >= deadline:
                        exhausted = True; row['status'] = 'budget_exhausted'; break
                    angles = original[index].copy(); angles[arm_columns] += delta
                    world, pixels, depth_jac, image_jac = evaluate(angles, True)
                    cost = objective(world, pixels, delta)
                    image_jac /= CONTRACT['image_sigma_px']; depth_jac /= CONTRACT['inferred_depth_sigma_model_units']
                    image_residual = ((pixels - target) / CONTRACT['image_sigma_px']).ravel()
                    depth_residual = (world[:, 1] - depth_target) / CONTRACT['inferred_depth_sigma_model_units']
                    image_weight = np.repeat(weights, 2)
                    prior_weight = 1 / CONTRACT['old_angle_sigma_rad'] ** 2
                    temporal_weight = 1 / CONTRACT['adjacent_correction_sigma_rad'] ** 2 if temporal else 0.
                    hessian = image_jac.T @ (image_weight[:, None] * image_jac) + depth_jac.T @ (weights[:, None] * depth_jac)
                    hessian += np.eye(8) * (prior_weight + temporal_weight + CONTRACT['linear_system_ridge'])
                    gradient = image_jac.T @ (image_weight * image_residual) + depth_jac.T @ (weights * depth_residual) + prior_weight * delta
                    if temporal: gradient += temporal_weight * (delta - previous)
                    step = np.linalg.solve(hessian, -gradient)
                    row['iterations'] = iteration + 1
                    accepted = False
                    for backtrack in range(CONTRACT['maximum_backtracks']):
                        trial = np.clip(delta + step * .5 ** backtrack, lo, hi)
                        if np.max(np.abs(trial - delta)) < 1e-9: break
                        angles = original[index].copy(); angles[arm_columns] += trial
                        trial_world, trial_pixels, _, _ = evaluate(angles)
                        row['backtracks'] += 1
                        if np.max(np.abs(trial_world[:, 1] - initial_world[:, 1])) > CONTRACT['maximum_endpoint_depth_change_model_units'] + 1e-12:
                            continue
                        if objective(trial_world, trial_pixels, trial) < cost - 1e-12:
                            delta, accepted = trial, True; break
                    if not accepted: break
            elif index:
                row['status'] = 'unchanged_unavailable_sides' if not np.any(weights) else 'budget_exhausted'
            angles = original[index].copy(); angles[arm_columns] += delta
            final_world, final_pixels, _, _ = evaluate(angles)
            for name, value in zip(arm_names, angles[arm_columns]): candidate['reference'][index]['joints'][name] = float(value)
            for i, point in enumerate(row['endpoints']):
                point.update(model_before=initial_world[i].tolist(), model_after=final_world[i].tolist(),
                    projected_before_px=initial_pixels[i].tolist(), projected_after_px=final_pixels[i].tolist(),
                    error_before_px=float(np.linalg.norm(initial_pixels[i] - target[i])) if point['image_px'] is not None else None,
                    error_after_px=float(np.linalg.norm(final_pixels[i] - target[i])) if point['image_px'] is not None else None,
                    fit_weight=float(weights[i]), depth_prior_model_units=float(depth_target[i]),
                    depth_prior_status='inferred_model_world' if point['raw_model_relative_depth'] is not None else 'original_articulated_model_fallback',
                    depth_change_model_units=float(final_world[i, 1] - initial_world[i, 1]))
            row.update(angle_correction_rad=dict(zip(arm_names, delta.tolist())), objective_after=objective(final_world, final_pixels, delta),
                       maximum_angle_correction_rad=float(np.max(np.abs(delta))),
                       original_arm_joints={name: references[index]['joints'][name] for name in arm_names},
                       candidate_arm_joints={name: candidate['reference'][index]['joints'][name] for name in arm_names})
            previous = delta.copy(); corrections.append(delta)
    finally:
        env.close()
    corrections = np.asarray(corrections)
    rate = np.max(np.abs(np.diff(corrections, axis=0)) / np.diff(times)[:, None])
    # Temporal clipping can force a nonzero starting correction; audit its nonlinear depth bound too.
    depth_max = max(abs(point['depth_change_model_units']) for row in rows for point in row['endpoints'])
    bound_checks = {
        'angle_correction': float(np.max(np.abs(corrections))) <= CONTRACT['maximum_angle_correction_rad'] + 1e-10,
        'endpoint_depth_change': depth_max <= CONTRACT['maximum_endpoint_depth_change_model_units'] + 1e-10,
        'correction_rate': float(rate) <= CONTRACT['maximum_correction_rate_rad_s'] + 1e-10,
        'first_reference_frozen': candidate['reference'][0] == spec['reference'][0],
        'unavailable_sides_frozen': all(not any(corrections[i, j] for j, name in enumerate(arm_names)
                                              if name.startswith(side + '_'))
                                        for i, row in enumerate(rows) for side, available in zip(('left', 'right'), row['eligible_sides'])
                                        if not available),
    }
    metadata = {'method': CONTRACT['method'], 'status': 'candidate_not_promoted', 'input_model_sha256': canonical_hash(spec),
                'construction_sha256': hashlib.sha256(construction_evidence).hexdigest(), 'camera_sha256': canonical_hash(camera),
                'depth_identified': False, 'source_evidence_unchanged': True, 'first_reference_frozen': True}
    candidate['arm_reference_alignment'] = metadata
    candidate['assumptions'].append('Arm angles are bounded construction-image fits with an uncertain model-depth prior; raw observations, first pose, shape, root and floor are unchanged. No grasp or support is inferred.')
    validate_human(candidate)
    status = 'infeasible_bounds' if not all(bound_checks.values()) else 'partial_budget_exhausted' if exhausted else 'complete'
    audit = {'schema_version': 1, 'profile': 'arm-reference-alignment-audit-v1', 'status': status,
             'contract': deepcopy(CONTRACT), 'input_model_sha256': canonical_hash(spec), 'candidate_model_sha256': canonical_hash(candidate),
             'construction_sha256': hashlib.sha256(construction_evidence).hexdigest(), 'human_entity_ref': deepcopy(human_entity_ref),
             'camera_hypothesis': camera, 'source_clock': deepcopy(source['time']), 'source_contract': deepcopy(source['source']),
             'source_slots': [{key: row[key] for key in ('frame', 'time_s', 'partition', 'admission', 'visibility', 'evidence_refs')} for row in source_rows.values()],
             'references': rows, 'changed_reference_poses': sum(row['maximum_angle_correction_rad'] > 1e-10 for row in rows),
             'maximum_correction_rate_rad_s': float(rate), 'seconds': time.monotonic() - started, 'physics_steps': 0,
             'bound_checks': bound_checks, 'maximum_endpoint_depth_change_model_units': depth_max,
             'fixed_fields': ['lengths', 'body_frame', 'mobility', 'root_reference', 'reference_evidence', 'motor_limits', 'dt', 'source_sha256'],
             'promoted': False, 'scope': 'Construction/model agreement only; candidate requires independent geometric admission before integration'}
    return candidate, audit
