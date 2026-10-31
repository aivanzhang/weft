"""Pure, opt-in camera/root proposals from caller-bound fixed body geometry."""

from copy import deepcopy
from hashlib import sha256
import json
import math
import time

import numpy as np

from weft.camera_reference import project_world
from weft.io import canonical_hash


PROFILE = 'fixed-body-camera-root-proposal-v1'
INPUT_PROFILE = 'fixed-body-camera-root-input-v1'
CONTRACT = {
    'maximum_input_bytes': 8 * 1024 * 1024, 'maximum_output_bytes': 16 * 1024 * 1024,
    'maximum_samples': 900, 'maximum_points': 12, 'maximum_supports': 64,
    'maximum_seconds': 10., 'maximum_iterations': 8, 'maximum_backtracks': 8,
    'maximum_camera_rotation_rad': .15, 'maximum_root_translation_model_units': .08,
    'maximum_root_correction_speed_model_units_s': .3, 'maximum_gap_s': .25,
    'minimum_depth_model_units': .05, 'floor_penetration_limit_model_units': .025,
    'minimum_confidence': .5, 'rank_relative_tolerance': 1e-10,
    'maximum_local_condition': 1e8, 'numerical_tolerance': 1e-8,
    'scope': 'Local image agreement under fixed body, scale and camera-center assumptions; not unique 3D recovery',
    'floor_scope': 'Translation of caller-supplied saved gaps only; no independent geometry or contact evaluation',
    'deadline_scope': 'Cooperative checks between bounded NumPy calls, not hard native-call preemption',
}
_SETTINGS = ('maximum_seconds', 'maximum_iterations', 'maximum_backtracks',
             'maximum_camera_rotation_rad', 'maximum_root_translation_model_units',
             'maximum_root_correction_speed_model_units_s')
_BINDINGS = {'source_sha256', 'body_sha256', 'evidence_sha256', 'body_angles_sha256', 'body_shape_sha256'}


def _require(condition, message):
    if not condition:
        raise ValueError('Camera/root proposal: ' + message)


def _check(deadline):
    if deadline is not None and time.monotonic() >= deadline:
        raise TimeoutError('Camera/root proposal exceeded its cooperative time bound')


def _number(value, low, high):
    return type(value) in (int, float) and math.isfinite(value) and low <= value <= high


def _array(value, shape, bound=1e4):
    try:
        original = np.asarray(value, dtype=object)
        _require(original.shape == shape and all(isinstance(v, (int, float, np.integer, np.floating))
                 and not isinstance(v, (bool, np.bool_)) for v in original.flat), 'numeric array elements required')
        result = np.asarray(value, dtype=float)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError('Camera/root proposal: invalid numeric array') from error
    _require(result.shape == shape and np.isfinite(result).all() and np.max(abs(result), initial=0.) <= bound,
             'nonfinite, excessive or mismatched numeric array')
    return result.copy()


def _fields(value, fields):
    _require(type(value) is dict and set(value) == set(fields), 'unsupported or missing fields')


def _skew(v):
    x, y, z = v
    return np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])


def _rotation(vector):
    angle = float(np.linalg.norm(vector)); skew = _skew(vector)
    if angle < 1e-8:
        return np.eye(3) + (1 - angle * angle / 6) * skew + (.5 - angle * angle / 24) * skew @ skew
    return np.eye(3) + math.sin(angle) / angle * skew + (1 - math.cos(angle)) / angle ** 2 * skew @ skew


def project_with_jacobians(points, rotation, center, intrinsics, radial_k1=0.):
    """Project fixed world points; derivatives use left camera rotations about its fixed center."""
    points = np.asarray(points, dtype=float)
    center = _array(center, (3,))
    rotation = _array(rotation, (3, 3), 1. + 1e-10)
    intrinsics = _array(intrinsics, (3, 3), 1e6)
    camera, pixels, valid = project_world(points, rotation, -rotation @ center, intrinsics, radial_k1)
    world_jac = np.full((len(points), 2, 3), np.nan)
    rotation_jac = np.full_like(world_jac, np.nan)
    for i in np.flatnonzero(valid):
        x, y, z = camera[i]; uv = np.array([x / z, y / z])
        distortion = (1 + radial_k1 * float(uv @ uv)) * np.eye(2) + 2 * radial_k1 * np.outer(uv, uv)
        perspective = np.array([[1 / z, 0., -uv[0] / z], [0., 1 / z, -uv[1] / z]])
        jac = intrinsics[:2, :2] @ distortion @ perspective
        world_jac[i] = jac @ rotation
        rotation_jac[i] = jac @ -_skew(camera[i])
    return {'camera_points': camera, 'pixels': pixels, 'valid': valid,
            'root_jacobian': world_jac, 'rotation_jacobian': rotation_jac}


def _rank(matrix, reference_scale=0.):
    singular = np.linalg.svd(matrix, compute_uv=False)
    threshold = max(reference_scale, float(singular[0]) if len(singular) else 0.) * CONTRACT['rank_relative_tolerance']
    rank = int(np.count_nonzero(singular > threshold))
    condition = float(singular[0] / singular[-1]) if len(singular) == 3 and rank == 3 else None
    return rank, condition, singular.tolist()


def local_identifiability(rotation_jacobians, root_jacobians, weights, *, movable_roots, deadline=None):
    """Data-only block ranks after eliminating root translations; priors never supply rank."""
    weights = np.asarray(weights, dtype=float)
    _require(weights.ndim == 2 and 1 <= weights.shape[0] <= 900 and 1 <= weights.shape[1] <= 12,
             'conditioning dimensions')
    n, p = weights.shape
    _require(np.isfinite(weights).all() and ((weights >= 0) & (weights <= 1)).all(), 'conditioning weights')
    a = _array(rotation_jacobians, (n, p, 2, 3), 1e12)
    b = _array(root_jacobians, (n, p, 2, 3), 1e12)
    _require(isinstance(movable_roots, (list, tuple)) and len(movable_roots) == n
             and all(type(v) is bool for v in movable_roots), 'explicit movable-root flags required')
    remaining, roots, camera_scale_squared = [], [], 0.
    for index in range(n):
        _check(deadline)
        scale = np.repeat(np.sqrt(weights[index]), 2)[:, None]
        ai, bi = a[index].reshape(-1, 3) * scale, b[index].reshape(-1, 3) * scale
        camera_scale_squared += float(np.sum(ai * ai))
        if movable_roots[index]:
            u, singular, _ = np.linalg.svd(bi, full_matrices=False)
            rank, condition, values = _rank(bi)
            ai = ai - u[:, :rank] @ (u[:, :rank].T @ ai)
            roots.append({'sample_index': index, 'rank': rank, 'condition': condition, 'singular_values': values})
        remaining.append(ai)
    # Elimination roundoff cannot create observable directions from an exact gauge.
    rank, condition, singular = _rank(np.concatenate(remaining), math.sqrt(camera_scale_squared))
    full = rank == 3 and all(row['rank'] == 3 for row in roots)
    conditioned = full and condition <= CONTRACT['maximum_local_condition'] and all(
        row['condition'] <= CONTRACT['maximum_local_condition'] for row in roots)
    return {'status': 'locally_identifiable' if conditioned else 'unidentified_or_ill_conditioned',
            'rank': rank + sum(row['rank'] for row in roots), 'variables': 3 + 3 * len(roots),
            'camera_after_eliminating_roots': {'rank': rank, 'condition': condition, 'singular_values': singular},
            'roots': roots, 'regularization_used_for_rank': False, 'global_uniqueness_claim': False,
            'scope': 'Linearized data sensitivity in the declared radian/model-unit coordinates; not measured uncertainty'}


def _settings(values):
    values = {} if values is None else values
    _require(type(values) is dict and set(values) <= set(_SETTINGS), 'only tighter proposal bounds are supported')
    result = {key: values.get(key, CONTRACT[key]) for key in _SETTINGS}
    for key, value in result.items():
        integer = key in ('maximum_iterations', 'maximum_backtracks')
        _require(_number(value, 1 if integer else .001, CONTRACT[key]) and (not integer or type(value) is int),
                 'invalid tighter proposal bound')
    return result


def _load(raw, deadline):
    _require(type(raw) is bytes and 0 < len(raw) <= CONTRACT['maximum_input_bytes'], 'bounded exact JSON bytes required')
    def unique(pairs):
        result = {}
        for key, value in pairs:
            _require(key not in result, 'duplicate JSON key')
            result[key] = value
        return result
    try:
        document = json.loads(raw, object_pairs_hook=unique)
        json.dumps(document, allow_nan=False)
    except (ValueError, TypeError, RecursionError, OverflowError) as error:
        raise ValueError('Camera/root proposal: invalid finite JSON') from error
    _check(deadline)
    _fields(document, {'schema_version', 'profile', 'bindings', 'scale_model_units_per_unit', 'camera',
                       'floor', 'samples', 'floor_checks'})
    _require(type(document['schema_version']) is int and document['schema_version'] == 1
             and document['profile'] == INPUT_PROFILE, 'unsupported input profile')
    _fields(document['bindings'], _BINDINGS)
    _require(all(type(v) is str and len(v) == 64 and all(c in '0123456789abcdef' for c in v)
                 for v in document['bindings'].values()), 'fixed input identity required')
    _require(_number(document['scale_model_units_per_unit'], 1e-6, 1e6), 'fixed scale required')
    camera = document['camera']
    _fields(camera, {'rotation', 'center', 'K', 'radial_k1', 'quality', 'provenance'})
    _require(camera['quality'] in ('authored', 'reconstructed', 'measured') and type(camera['provenance']) is str
             and 1 <= len(camera['provenance']) <= 2000, 'camera provenance required')
    rotation, center = _array(camera['rotation'], (3, 3), 1. + 1e-10), _array(camera['center'], (3,))
    intrinsics = _array(camera['K'], (3, 3), 1e6)
    _require(np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-10, rtol=0)
             and abs(np.linalg.det(rotation) - 1.) <= 1e-10, 'proper camera rotation required')
    _require(_number(camera['radial_k1'], -1, 1) and intrinsics[0, 0] > 0 and intrinsics[1, 1] > 0
             and np.array_equal(intrinsics[2], [0., 0., 1.]) and intrinsics[1, 0] == 0,
             'fixed pinhole intrinsics required')
    _fields(document['floor'], {'normal', 'offset_model_units'})
    normal = _array(document['floor']['normal'], (3,), 1.)
    _require(abs(np.linalg.norm(normal) - 1.) <= 1e-10
             and _number(document['floor']['offset_model_units'], -1e4, 1e4), 'fixed unit-normal plane required')
    samples = document['samples']
    _require(type(samples) is list and 2 <= len(samples) <= CONTRACT['maximum_samples'], '2–900 source knots required')
    points, pixels, weights, roots, times, names = [], [], [], [], [], None
    previous_frame = -1
    for row in samples:
        _check(deadline)
        _fields(row, {'evidence_id', 'frame', 'time_s', 'decoder_timestamp_s', 'root', 'points'})
        frame, stamp = row['frame'], row['time_s']
        _require(type(frame) is int and previous_frame < frame < 1800 and row['evidence_id'] ==
                 f"{document['bindings']['source_sha256']}:frame:{frame}", 'source frame identity differs')
        _require(_number(stamp, 0, 60) and (not times or 1e-4 <= stamp - times[-1] <= .25 + 1e-9),
                 'increasing original times with bounded gaps required')
        _require(row['decoder_timestamp_s'] is None or _number(row['decoder_timestamp_s'], 0, 60), 'decoder time must stay explicit')
        previous_frame = frame; times.append(stamp); roots.append(_array(row['root'], (3,)))
        _require(type(row['points']) is list and 1 <= len(row['points']) <= 12, '1–12 fixed endpoint slots required')
        xyz, uv, confidence, identifiers = [], [], [], []
        for point in row['points']:
            _fields(point, {'name', 'world', 'pixel', 'weight'})
            _require(type(point['name']) is str and 1 <= len(point['name']) <= 80, 'bounded endpoint identity required')
            identifiers.append(point['name']); xyz.append(_array(point['world'], (3,)))
            weight = point['weight']
            _require(_number(weight, 0, 1) and (weight == 0 or weight >= .5), 'unknown/weak endpoints must have zero fit weight')
            _require(point['pixel'] is not None or weight == 0, 'missing pixels cannot receive weight')
            uv.append(np.zeros(2) if point['pixel'] is None else _array(point['pixel'], (2,), 16384))
            confidence.append(weight)
        _require(len(set(identifiers)) == len(identifiers) and (names is None or identifiers == names),
                 'every sample must retain the same distinct endpoint slots')
        names = identifiers; points.append(xyz); pixels.append(uv); weights.append(confidence)
    floor = document['floor_checks']
    if floor is not None:
        _fields(floor, {'bindings', 'rows'})
        expected = {key: document['bindings'][key] for key in ('body_sha256', 'body_angles_sha256', 'body_shape_sha256')}
        expected.update(root_reference_sha256=canonical_hash([{'time_s': row['time_s'], 'translation': row['root']} for row in samples]),
                        floor_sha256=canonical_hash(document['floor']))
        _require(floor['bindings'] == expected, 'saved floor identities differ from the fixed geometry, plane or roots')
        floor = floor['rows']
        count = math.floor((times[-1] - times[0]) * 30 + 1e-9) + 1
        schedule = [('source_pose', t) for t in times] + [
            ('control_grid', min(times[-1], times[0] + i / 30)) for i in range(count)]
        _require(type(floor) is list and len(floor) == len(schedule), 'complete source/30 Hz floor schedule required')
        support_names = None
        for check, (kind, stamp) in zip(floor, schedule, strict=True):
            _check(deadline); _fields(check, {'kind', 'time_s', 'gaps'})
            _require(check['kind'] == kind and check['time_s'] == stamp, 'saved floor schedule differs')
            gaps = check['gaps']
            _require(type(gaps) is dict and 1 <= len(gaps) <= 64 and all(
                type(k) is str and 1 <= len(k) <= 100 and _number(v, -1e4, 1e4) for k, v in gaps.items()),
                'complete finite support gaps required')
            _require(support_names is None or set(gaps) == support_names, 'floor supports differ between rows')
            support_names = set(gaps)
    return document, np.array(points), np.array(pixels), np.array(weights), np.array(roots), np.array(times), rotation, center, intrinsics, normal


def _evaluate(points, delta, rotation, center, intrinsics, radial, deadline):
    results = []
    for xyz, shift in zip(points, delta, strict=True):
        _check(deadline)
        row = project_with_jacobians(xyz + shift, rotation, center, intrinsics, radial)
        if (not row['valid'].all() or np.min(row['camera_points'][:, 2]) < CONTRACT['minimum_depth_model_units']
                or any(not np.isfinite(row[key]).all() or np.max(abs(row[key])) > 1e12
                       for key in ('pixels', 'root_jacobian', 'rotation_jacobian'))):
            return None
        results.append(row)
    return {key: np.array([row[key] for row in results]) for key in ('pixels', 'root_jacobian', 'rotation_jacobian')}


def _floor(document, times, delta, normal):
    if document['floor_checks'] is None:
        return None
    result = []
    for row in document['floor_checks']['rows']:
        translation = np.array([np.interp(row['time_s'], times, delta[:, axis]) for axis in range(3)])
        offset = float(normal @ translation)
        result.append({'kind': row['kind'], 'time_s': row['time_s'],
                       'gaps_before': row['gaps'], 'gaps_after': {name: gap + offset for name, gap in row['gaps'].items()}})
    return result


def _floor_gates(rows):
    if rows is None:
        return None, None
    no_worse = all(gap >= min(0., row['gaps_before'][name]) - 1e-8 for row in rows for name, gap in row['gaps_after'].items())
    feasible = all(gap >= -.025 for row in rows for gap in row['gaps_after'].values())
    return no_worse, feasible


def _step(current, residual, weights, movable, deadline):
    camera_h, camera_g = np.zeros((3, 3)), np.zeros(3)
    blocks = []
    for index in range(len(weights)):
        _check(deadline)
        scale = np.repeat(np.sqrt(weights[index]), 2)
        a = current['rotation_jacobian'][index].reshape(-1, 3) * scale[:, None]
        b = current['root_jacobian'][index].reshape(-1, 3) * scale[:, None]
        error = residual[index].ravel() * scale
        camera_h += a.T @ a; camera_g += a.T @ error
        blocks.append((a.T @ b, b.T @ b, b.T @ error) if movable[index] else None)
    damping = max(1e-8, float(np.max(np.diag(camera_h))) * 1e-8)
    reduced, rhs = camera_h + damping * np.eye(3), -camera_g
    for block in blocks:
        if block is not None:
            coupling, hessian, gradient = block
            hessian += damping * np.eye(3)
            reduced -= coupling @ np.linalg.solve(hessian, coupling.T)
            rhs += coupling @ np.linalg.solve(hessian, gradient)
    camera_step = np.linalg.solve(reduced, rhs)
    roots = np.zeros((len(weights), 3))
    for index, block in enumerate(blocks):
        if block is not None:
            coupling, hessian, gradient = block
            roots[index] = -np.linalg.solve(hessian, gradient + coupling.T @ camera_step)
    return camera_step, roots


def _bounded(rotation, original_rotation, delta, times, settings):
    angle = math.acos(float(np.clip((np.trace(rotation @ original_rotation.T) - 1) / 2, -1, 1)))
    displacement = float(np.linalg.norm(delta, axis=1).max())
    speed = float((np.linalg.norm(np.diff(delta, axis=0), axis=1) / np.diff(times)).max())
    okay = (angle <= settings['maximum_camera_rotation_rad'] + 1e-12
            and displacement <= settings['maximum_root_translation_model_units'] + 1e-12
            and speed <= settings['maximum_root_correction_speed_model_units_s'] + 1e-12)
    return okay, angle, displacement, speed


def propose_camera_root_refinement(input_document: bytes, *, settings=None):
    """Return a separate bounded proposal; source binding and native admission remain caller prerequisites."""
    started = time.monotonic(); settings = _settings(settings); deadline = started + settings['maximum_seconds']
    (document, points, pixels, weights, roots, times, rotation, center, intrinsics, normal) = _load(input_document, deadline)
    original_rotation = rotation.copy(); delta = np.zeros_like(roots)
    movable = [i > 0 and bool(np.any(row > 0)) for i, row in enumerate(weights)]
    radial = document['camera']['radial_k1']
    current = _evaluate(points, delta, rotation, center, intrinsics, radial, deadline)
    _require(current is not None, 'all fixed geometry must remain in front of the camera')
    before_errors = np.linalg.norm(current['pixels'] - pixels, axis=2)
    initial_rank = local_identifiability(current['rotation_jacobian'], current['root_jacobian'], weights,
                                        movable_roots=movable, deadline=deadline)
    last_rank = initial_rank
    counters = {'iterations': 0, 'evaluated_trials': 0, 'accepted_updates': 0,
                'rejected_bounds': 0, 'rejected_projection': 0, 'rejected_image': 0, 'rejected_floor': 0,
                'rejected_conditioning': 0, 'model_constructions': 0, 'physics_steps': 0, 'inference_calls': 0}
    reason = 'unidentified_or_ill_conditioned' if initial_rank['status'] != 'locally_identifiable' else 'iteration_limit'
    try:
        if initial_rank['status'] == 'locally_identifiable':
            for iteration in range(settings['maximum_iterations']):
                _check(deadline); counters['iterations'] = iteration + 1
                residual = current['pixels'] - pixels
                current_errors = np.linalg.norm(residual, axis=2)
                cost = float(np.sum(weights * current_errors ** 2))
                if cost <= 1e-20:
                    reason = 'already_matches_available_pixels'; break
                camera_step, root_step = _step(current, residual, weights, movable, deadline)
                accepted = False
                for backtrack in range(settings['maximum_backtracks']):
                    _check(deadline); fraction = 2. ** -backtrack
                    proposed_rotation = _rotation(fraction * camera_step) @ rotation
                    proposed_delta = delta + fraction * root_step
                    if not _bounded(proposed_rotation, original_rotation, proposed_delta, times, settings)[0]:
                        counters['rejected_bounds'] += 1; continue
                    floor_rows = _floor(document, times, proposed_delta, normal)
                    if _floor_gates(floor_rows)[0] is False:
                        counters['rejected_floor'] += 1; continue
                    candidate = _evaluate(points, proposed_delta, proposed_rotation, center, intrinsics, radial, deadline)
                    counters['evaluated_trials'] += 1
                    if candidate is None:
                        counters['rejected_projection'] += 1; continue
                    errors = np.linalg.norm(candidate['pixels'] - pixels, axis=2)
                    if (np.any((weights > 0) & (errors > np.minimum(current_errors, before_errors) + 1e-8))
                            or float(np.sum(weights * errors ** 2)) >= cost - 1e-12):
                        counters['rejected_image'] += 1; continue
                    rank = local_identifiability(candidate['rotation_jacobian'], candidate['root_jacobian'], weights,
                                                movable_roots=movable, deadline=deadline)
                    if rank['status'] != 'locally_identifiable':
                        counters['rejected_conditioning'] += 1; continue
                    rotation, delta, current, last_rank = proposed_rotation, proposed_delta, candidate, rank
                    counters['accepted_updates'] += 1; accepted = True; break
                if not accepted:
                    reason = 'no_accepted_direction'; break
    except TimeoutError:
        reason = 'time_limit'
    errors = np.linalg.norm(current['pixels'] - pixels, axis=2)
    floor_rows = _floor(document, times, delta, normal)
    floor_no_worse, floor_feasible = _floor_gates(floor_rows)
    bounded, angle, translation, speed = _bounded(rotation, original_rotation, delta, times, settings)
    rows = []
    for index, row in enumerate(document['samples']):
        rows.append({key: row[key] for key in ('evidence_id', 'frame', 'time_s', 'decoder_timestamp_s')})
        rows[-1].update(root_before=row['root'], root_after=(roots[index] + delta[index]).tolist(),
                        root_delta=delta[index].tolist(), fit_status='fixed_first_root' if index == 0 else
                        'unavailable_pixels_root_unchanged' if not movable[index] else
                        'conditional_root_changed' if np.any(delta[index]) else 'root_unchanged',
                        endpoints=[{'name': point['name'], 'pixel': point['pixel'], 'weight': point['weight'],
                                    'error_before_px': float(before_errors[index, j]) if point['weight'] > 0 else None,
                                    'error_after_px': float(errors[index, j]) if point['weight'] > 0 else None}
                                   for j, point in enumerate(row['points'])])
    known = weights > 0
    camera = deepcopy(document['camera']); camera['rotation'] = rotation.tolist()
    report = {'schema_version': 1, 'profile': PROFILE, 'reason': reason, 'contract': deepcopy(CONTRACT), 'settings': settings,
              'input_sha256': sha256(input_document).hexdigest(), 'bindings': document['bindings'],
              'binding_scope': 'Exact selected input bytes and caller-supplied identity claims; original sources are not read or authenticated',
              'fixed_input_sha256': canonical_hash({'bindings': document['bindings'], 'scale': document['scale_model_units_per_unit'],
                  'floor': document['floor'], 'first_root': document['samples'][0]['root'],
                  'camera_except_rotation': {k: v for k, v in document['camera'].items() if k != 'rotation'},
                  'samples': document['samples'], 'floor_checks': document['floor_checks']}),
              'initial_identifiability': initial_rank, 'final_identifiability': last_rank, 'counts': counters,
              'known_endpoints': int(known.sum()), 'unknown_endpoints': int((~known).sum()),
              'image_rmse_before_px': float(np.sqrt(np.mean(before_errors[known] ** 2))) if known.any() else None,
              'image_rmse_after_px': float(np.sqrt(np.mean(errors[known] ** 2))) if known.any() else None,
              'maximum_camera_rotation_rad': angle, 'maximum_root_translation_model_units': translation,
              'maximum_root_correction_speed_model_units_s': speed,
              'gates': {'first_root_unchanged': bool(np.array_equal(delta[0], [0., 0., 0.])), 'bounds': bounded,
                        'image_pointwise_no_worse': bool(np.all(errors[known] <= before_errors[known] + 1e-8)) if known.any() else None,
                        'saved_floor_no_worse': floor_no_worse, 'saved_floor_feasibility': floor_feasible},
              'samples': rows, 'floor_checks': floor_rows, 'elapsed_seconds': time.monotonic() - started,
              'geometry_admission': 'unverified_not_executed', 'source_integration': 'not_performed',
              'geometry_qualification': 'unavailable' if floor_rows is None else 'caller_saved_gap_checks_only',
              'source_qualified': False, 'admitted': False, 'physical_validation': 'unverified', 'promoted': False}
    result = {'status': 'partial' if reason == 'time_limit' else 'unavailable' if initial_rank['status'] != 'locally_identifiable'
              else 'proposed' if counters['accepted_updates'] else 'unchanged',
              'camera_candidate': camera, 'root_reference_candidate': [{'time_s': row['time_s'], 'translation': row['root_after']} for row in rows],
              'report': report}
    _require(len(json.dumps(result, allow_nan=False).encode()) <= CONTRACT['maximum_output_bytes'], 'output exceeds bound')
    return result
