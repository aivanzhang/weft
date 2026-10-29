"""Bounded construction-only velocity proposals; later replay decides acceptance."""

from copy import deepcopy
import math

import numpy as np


MOTION_CONTRACT = {
    'method': 'early-free-flight-velocity-v1', 'maximum_window_seconds': 1.,
    'minimum_samples': 8, 'maximum_samples_per_object': 241, 'minimum_span_seconds': .25,
    'maximum_gap_seconds': .2, 'maximum_rms_px': 1., 'maximum_error_px': 2.,
    'contact_margin_px': 2., 'maximum_speed_image_widths_per_second': 3.,
    'maximum_objects': 8, 'maximum_input_rows': 14400,
    'assumptions': 'Authored gravity and per-second damping, fixed camera, common 2D plane, no contact or external force during the fitted window. Passing image checks does not verify those assumptions.',
    'scope': 'Continuous free-flight proposal, not exact engine inversion or recovered forces; only initial velocity may change.'}


def _basis(times, damping):
    rate = math.log(damping)
    if abs(rate) < 1e-5:
        velocity = times + rate * times**2 / 2 + rate**2 * times**3 / 6
        gravity = times**2 / 2 + rate * times**3 / 6 + rate**2 * times**4 / 24
    else:
        velocity = np.expm1(rate * times) / rate
        gravity = (velocity - times) / rate
    return velocity, gravity


def _extent(obj):
    if obj['shape'] == 'circle':
        return np.array([obj['radius'], obj['radius']]), obj['radius']
    half = np.asarray(obj['size']) / 2
    return half, float(np.linalg.norm(half))


def propose_initial_motion(scene, generation, *, width, origin_time_s):
    """Receive construction rows only; never access development observations or task scores."""
    if (type(width) is not int or not 64 <= width <= 960
            or not math.isfinite(origin_time_s) or not 0 <= origin_time_s <= 3600
            or not 1 <= len(scene['objects']) <= MOTION_CONTRACT['maximum_objects']
            or not 1 <= len(generation) <= MOTION_CONTRACT['maximum_input_rows']):
        raise ValueError('Initial-motion input exceeds the bounded construction envelope')
    damping = scene['damping']
    if not math.isfinite(damping) or not .01 <= damping <= 1:
        raise ValueError('Initial motion requires bounded per-second damping')
    objects = {obj['id']: obj for obj in scene['objects']}
    window_end = origin_time_s + MOTION_CONTRACT['maximum_window_seconds']
    frames, by_object = {}, {key: [] for key in objects}
    for row in generation:
        if (row['object_id'] not in objects or type(row['frame']) is not int
                or not all(math.isfinite(row[key]) for key in ('time_s', 'x', 'y'))
                or row['time_s'] < origin_time_s - 1e-9):
            raise ValueError('Invalid construction observation for initial motion')
        if row['time_s'] > window_end + 1e-9:
            continue
        group = frames.setdefault(row['frame'], {})
        if row['object_id'] in group or any(abs(other['time_s'] - row['time_s']) > 1e-9 for other in group.values()):
            raise ValueError('Duplicate or inconsistent construction frame')
        group[row['object_id']] = row
        by_object[row['object_id']].append(row)
    candidate, details, changes = deepcopy(scene), {}, {}
    candidate_objects = {obj['id']: obj for obj in candidate['objects']}
    margin = MOTION_CONTRACT['contact_margin_px'] / width
    for object_id, obj in objects.items():
        samples = sorted(by_object[object_id], key=lambda row: row['time_s'])
        detail = {'status': 'unavailable', 'evidence_ids': [row['id'] for row in samples],
                  'source_frames': [row['frame'] for row in samples], 'samples': len(samples),
                  'reason': None, 'velocity': None}
        details[object_id] = detail
        if obj['role'] == 'static':
            detail.update(status='unchanged_static', reason='Static velocity remains zero')
            continue
        if not MOTION_CONTRACT['minimum_samples'] <= len(samples) <= MOTION_CONTRACT['maximum_samples_per_object']:
            detail['reason'] = 'insufficient_or_excessive_samples'
            continue
        times = np.array([row['time_s'] - origin_time_s for row in samples])
        if (abs(times[0]) > 1e-9 or times[-1] < MOTION_CONTRACT['minimum_span_seconds']
                or np.any(np.diff(times) <= 0) or np.max(np.diff(times)) > MOTION_CONTRACT['maximum_gap_seconds'] + 1e-9):
            detail['reason'] = 'insufficient_contiguous_initial_time_support'
            continue
        extent, radius = _extent(obj)
        positions = np.array([[row['x'], row['y']] for row in samples])
        if np.any(positions - extent <= margin) or np.any(positions + extent >= np.array([1., scene['height']]) - margin):
            detail['reason'] = 'possible_boundary_contact'
            continue
        blocked = None
        for row, position in zip(samples, positions):
            group = frames[row['frame']]
            if set(group) != set(objects):
                blocked = 'missing_other_object_contact_evidence'
                break
            for other_id, other in objects.items():
                if other_id == object_id:
                    continue
                other_row = group[other_id]
                if math.dist(position, [other_row['x'], other_row['y']]) <= radius + _extent(other)[1] + margin:
                    blocked = 'possible_object_contact'
                    break
            if blocked:
                break
        if blocked:
            detail['reason'] = blocked
            continue
        velocity_basis, gravity_basis = _basis(times, damping)
        gravity = np.asarray(scene['gravity'], dtype=float)
        coefficients = np.linalg.lstsq(np.column_stack((velocity_basis, np.ones(len(times)))),
                                      positions - gravity_basis[:, None] * gravity, rcond=None)[0]
        velocity, intercept = coefficients
        # Keep the actual initial position; a fitted intercept cannot hide a shifted body.
        predicted = np.asarray(obj['position']) + velocity_basis[:, None] * velocity + gravity_basis[:, None] * gravity
        errors = np.linalg.norm(predicted - positions, axis=1) * width
        detail.update(velocity=velocity.tolist(), nuisance_intercept=intercept.tolist(),
                      rms_px=float(np.sqrt(np.mean(errors**2))), max_error_px=float(errors.max()),
                      window_seconds=float(times[-1]), initial_position_unchanged=True)
        if not np.isfinite(velocity).all() or np.linalg.norm(velocity) > MOTION_CONTRACT['maximum_speed_image_widths_per_second']:
            detail['reason'] = 'velocity_exceeds_bound'
        elif detail['rms_px'] > MOTION_CONTRACT['maximum_rms_px'] or detail['max_error_px'] > MOTION_CONTRACT['maximum_error_px']:
            detail['reason'] = 'motion_inconsistent_with_free_flight_or_noise_allowance'
        else:
            detail.update(status='proposed', reason='Construction window passes the declared free-flight screen; not physically verified')
            changes[object_id] = velocity.tolist()
            candidate_objects[object_id]['velocity'] = velocity.tolist()
            candidate_objects[object_id]['provenance']['velocity_repair'] = {
                'method': MOTION_CONTRACT['method'], 'construction_evidence_ids': detail['evidence_ids'],
                'damping_hypothesis': damping, 'gravity_hypothesis': list(scene['gravity']),
                'initial_position_unchanged': True, 'physical_validity': 'unverified'}
    return (candidate if changes else None), {
        'contract': deepcopy(MOTION_CONTRACT), 'status': 'proposed' if changes else 'unavailable',
        'origin_time_s': origin_time_s, 'maximum_source_time_s': window_end,
        'input_partition': 'generation_only', 'objects': details, 'changes': changes,
        'accepted': False, 'scope': MOTION_CONTRACT['scope']}
