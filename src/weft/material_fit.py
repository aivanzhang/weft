"""Explicit construction-only material hypotheses; fitted image motion is not physical truth."""

from copy import deepcopy
import math
import time

import numpy as np

from weft.spatial_rigid import SpatialRigidEnv, validate_scene


def surface_hypothesis(scene, *, actor_id, target_id, support_id, actor_friction, target_friction, mass_ratio):
    result = validate_scene(scene)
    objects = {obj['id']: obj for obj in result['objects']}
    if len({actor_id, target_id, support_id}) != 3 or not {actor_id, target_id, support_id} <= objects.keys():
        raise ValueError('Material roles require three distinct known objects')
    actor, target, support = (objects[key] for key in (actor_id, target_id, support_id))
    if (actor['role'], target['role'], support['role']) != ('active', 'passive', 'static'):
        raise ValueError('Material roles must preserve active/passive/static identity')
    for value, low, high in ((actor_friction, 0, 2), (target_friction, 0, 2), (mass_ratio, .1, 4)):
        if type(value) not in (float, int) or not math.isfinite(value) or not low <= value <= high:
            raise ValueError('Material parameter lies outside its explicit envelope')
    # Equal-priority MuJoCo contacts use maximum geom friction, not a product.
    support['friction'][0] = 0.
    actor['friction'][0], target['friction'][0] = actor_friction, target_friction
    target['mass'] = actor['mass'] * mass_ratio
    parameters = {'actor_support_sliding': actor_friction, 'target_support_sliding': target_friction,
                  'target_actor_mass_ratio': mass_ratio,
                  'actor_target_sliding': max(actor_friction, target_friction)}
    for obj in (actor, target, support):
        obj['provenance']['material_fit'] = {'parameters': deepcopy(parameters),
            'status': 'construction_hypothesis', 'physical_validation': 'unverified'}
    result['assumptions'].append('Support friction is zero; moving-object coefficients set the two support pairs. Other pair coefficients follow MuJoCo maxima. Parameters are fitted hypotheses, not measurements.')
    return validate_scene(result)


def validate_inputs(scene, observations, projection, initial_frame, steps_per_frame):
    scene = validate_scene(scene)
    if type(initial_frame) is not int or initial_frame < 0 or type(steps_per_frame) is not int or not 1 <= steps_per_frame <= 16:
        raise ValueError('Source initialization and step ratio must be explicit integers')
    if not isinstance(observations, list) or not 2 <= len(observations) <= 10800:
        raise ValueError('Construction evidence must contain 2–10800 slots')
    expected = {'scope', 'source_sha256', 'image_size', 'matrix_px', 'calibration'}
    if (not isinstance(projection, dict) or set(projection) != expected
            or projection['scope'] != 'construction' or projection['calibration'] != 'unverified_affine_hypothesis'):
        raise ValueError('Only explicit construction evidence and unverified projection hypotheses are accepted')
    source = projection['source_sha256']
    if not isinstance(source, str) or len(source) != 64 or any(c not in '0123456789abcdef' for c in source):
        raise ValueError('Projection must be bound to one source hash')
    matrix = np.asarray(projection['matrix_px'], dtype=float)
    size = np.asarray(projection['image_size'], dtype=float)
    if (matrix.shape != (2, 4) or not np.isfinite(matrix).all() or (np.abs(matrix) > 100000).any()
            or np.linalg.matrix_rank(matrix[:, :3]) != 2
            or size.shape != (2,) or not np.isfinite(size).all() or (size < 32).any() or (size > 10000).any()):
        raise ValueError('Invalid bounded source-image projection')
    identifiers = {obj['id'] for obj in scene['objects'] if obj['role'] != 'static'}
    seen, source_ids, frames, objects, times = set(), set(), set(), set(), {}
    for row in observations:
        if not isinstance(row, dict) or set(row) != {'source_id', 'frame', 'time_s', 'object_id', 'center_px', 'identity', 'scope'}:
            raise ValueError('Construction slots must retain complete source identity and availability')
        frame, stamp, obj = row['frame'], row['time_s'], row['object_id']
        if (type(frame) is not int or not 0 <= frame <= 900 or obj not in identifiers
                or type(stamp) not in (float, int) or not math.isfinite(stamp) or not 0 <= stamp <= 60
                or row['scope'] != 'construction' or not isinstance(row['identity'], dict)
                or not isinstance(row['source_id'], str) or not row['source_id'].startswith(f'{source}:frame:{frame}:')
                or row['source_id'] in source_ids or (frame, obj) in seen):
            raise ValueError('Duplicate, mismatched or non-construction source slot')
        if frame in times and stamp != times[frame]:
            raise ValueError('Object slots must share their exact source-frame clock')
        if row['center_px'] is not None:
            center = np.asarray(row['center_px'], dtype=float)
            if center.shape != (2,) or not np.isfinite(center).all() or (center < 0).any() or (center > size).any():
                raise ValueError('Observed image center exceeds source bounds')
        seen.add((frame, obj));source_ids.add(row['source_id']);frames.add(frame);objects.add(obj);times[frame] = stamp
    if frames != set(range(max(frames)+1)) or seen != {(f, obj) for f in frames for obj in objects}:
        raise ValueError('Every source frame/object slot, including missing ones, must be retained')
    if initial_frame not in frames or (max(frames)-initial_frame)*steps_per_frame > scene['task']['max_steps']:
        raise ValueError('Source interval exceeds the complete episode envelope')
    if any(not math.isclose(times[f]-times[0], f*scene['dt']*steps_per_frame, abs_tol=1e-7, rel_tol=0) for f in frames):
        raise ValueError('Source clock differs from the declared simulation step ratio')
    for obj in objects:
        if next(row for row in observations if row['frame'] == initial_frame and row['object_id'] == obj)['center_px'] is None:
            raise ValueError('Initialization needs an observed center for every fitted object')
    return {'matrix': matrix, 'last_frame': max(frames), 'objects': sorted(objects), 'times': times}


def score_states(observations, states, projection):
    matrix = np.asarray(projection['matrix_px'], dtype=float)
    by_frame = {row['frame']: {obj['id']: obj for obj in row['snapshot']['objects']} for row in states}
    rows, errors = [], {}
    for original in observations:
        row = deepcopy(original)
        obj = by_frame.get(row['frame'], {}).get(row['object_id'])
        prediction = (matrix @ np.r_[obj['position'], 1.]).tolist() if obj else None
        error = math.dist(prediction, row['center_px']) if prediction is not None and row['center_px'] is not None else None
        row.update(predicted_center_px=prediction, error_px=error,
                   availability='compared' if error is not None else 'before_simulation' if obj is None else 'observation_missing')
        rows.append(row)
        if error is not None:
            errors.setdefault(row['object_id'], []).append(error)
    values = [error for group in errors.values() for error in group]
    if not values:
        raise ValueError('No comparable construction observations')
    return {'scope': 'construction_fit_only', 'rows': rows, 'slots': len(rows), 'compared': len(values),
        'rmse_px': float(np.sqrt(np.mean(np.square(values)))), 'maximum_error_px': max(values),
        'by_object': {obj: {'compared': len(group), 'rmse_px': float(np.sqrt(np.mean(np.square(group)))),
                          'maximum_error_px': max(group)} for obj, group in errors.items()}}


def coast_candidate(scene, observations, projection, *, initial_frame=0, steps_per_frame=1, max_seconds=10):
    if type(max_seconds) not in (int, float) or not math.isfinite(max_seconds) or not 0 < max_seconds <= 30:
        raise ValueError('Candidate wall-clock limit must be 0–30 seconds')
    contract = validate_inputs(scene, observations, projection, initial_frame, steps_per_frame)
    started = time.perf_counter()
    env = SpatialRigidEnv(scene)
    events, states, timings = [], [], []
    try:
        env.reset(seed=1729, options={'jitter': False})
        snapshot = env.snapshot()
        events.append({'type': 'start', 'seed': 1729, 'snapshot': snapshot})
        states.append({'frame': initial_frame, 'snapshot': snapshot})
        for step in range(1, (contract['last_frame']-initial_frame)*steps_per_frame+1):
            tick = time.perf_counter()
            _, reward, terminated, truncated, _ = env.step(0)
            snapshot = env.snapshot()
            timings.append((time.perf_counter()-tick)*1000)
            events.append({'type': 'step', 'action': 0, 'reward': reward, 'terminated': terminated,
                           'truncated': truncated, 'snapshot': snapshot})
            if step%steps_per_frame == 0:
                states.append({'frame': initial_frame+step//steps_per_frame, 'snapshot': snapshot})
            if (terminated or truncated) and step < (contract['last_frame']-initial_frame)*steps_per_frame:
                raise ValueError('Candidate ended before the complete source interval')
            if time.perf_counter()-started > max_seconds:
                raise TimeoutError('Candidate exceeds its bounded wall-clock budget')
        result = score_states(observations, states, projection)
        result.update(events=events, states=states, runtime_seconds=time.perf_counter()-started,
            step_snapshot_ms={'samples': len(timings), 'p95': float(np.percentile(timings, 95)) if timings else None},
            physical_validation='unverified')
        return result
    finally:
        env.close()
