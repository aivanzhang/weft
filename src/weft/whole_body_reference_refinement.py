"""Opt-in source-image fitting with full-body floor constraints; never a physical repair."""

from copy import deepcopy
from hashlib import sha256
import json
import math
import time

import numpy as np

from weft.constrained_qp import solve_feasible_qp
from weft.io import canonical_hash
from weft.root_motion import build_root_reference
from weft.spatial_pose import SPATIAL_JOINT_LIMITS, SPATIAL_JOINT_NAMES


CONTRACT = {
    'method': 'blocked-whole-body-floor-reference-v1', 'maximum_references': 900,
    'maximum_grid_samples': 1801, 'sample_hz': 30, 'maximum_sweeps': 2,
    'maximum_backtracks': 6, 'maximum_seconds': 30., 'finalization_seconds': 8.,
    'maximum_geometry_calls': 120000, 'maximum_floor_supports': 64,
    'maximum_angle_correction_rad': .15, 'maximum_update_rad': .10,
    'maximum_correction_rate_rad_s': 1.5, 'maximum_temporal_gap_s': .25,
    'maximum_endpoint_depth_change_model_units': .08, 'minimum_confidence': .5,
    'huber_delta_model_units': .05, 'original_angle_regularization': .001,
    'correction_temporal_regularization': .0005, 'linear_system_ridge': 1e-6,
    'numerical_tolerance': 1e-8, 'floor_penetration_limit_model_units': .025,
    'maximum_input_bytes': 8 * 1024 * 1024, 'maximum_output_bytes': 16 * 1024 * 1024,
    'geometry_scope': 'Every authored human sphere/capsule against its fixed plane; no equipment or self-collision qualification',
    'floor_constraint': 'No new/worse negative support-point gap at any affected source/grid sample',
    'image_constraint': 'Only confident source-knot endpoints; no invented image targets at grid times',
    'sampling_scope': 'Complete source and 30 Hz checks, not continuous-time collision proof',
    'deadline_scope': 'Cooperative checks between bounded calls, not native-call preemption',
}
NAMES = SPATIAL_JOINT_NAMES
POINTS = {'left_shoulder': 11, 'right_shoulder': 12, 'left_elbow': 13, 'right_elbow': 14,
          'left_wrist': 15, 'right_wrist': 16, 'left_hip': 23, 'right_hip': 24,
          'left_knee': 25, 'right_knee': 26, 'left_ankle': 27, 'right_ankle': 28}
BLOCKS = (NAMES[:3] + tuple(n for n in NAMES if n.startswith('left_') and ('hip' in n or 'knee' in n)),
          NAMES[:3] + tuple(n for n in NAMES if n.startswith('right_') and ('hip' in n or 'knee' in n)),
          tuple(n for n in NAMES if 'shoulder' in n or 'elbow' in n))


def _require(condition, message):
    if not condition:
        raise ValueError('Whole-body reference refinement: ' + message)


def _document(raw):
    _require(type(raw) is bytes and 0 < len(raw) <= CONTRACT['maximum_input_bytes'], 'bounded exact input bytes required')
    def pairs(rows):
        result = {}
        for key, value in rows:
            _require(key not in result, 'duplicate JSON key')
            result[key] = value
        return result
    try:
        result = json.loads(raw, object_pairs_hook=pairs)
        json.dumps(result, allow_nan=False)
    except (ValueError, TypeError, RecursionError) as error:
        raise ValueError('Whole-body reference refinement: invalid finite JSON') from error
    _require(type(result) is dict, 'input must be an object')
    return result


def _settings(values):
    values = {} if values is None else values
    allowed = ('maximum_sweeps', 'maximum_seconds', 'finalization_seconds', 'maximum_geometry_calls')
    _require(type(values) is dict and set(values) <= set(allowed), 'only tighter work bounds are configurable')
    result = {key: values.get(key, CONTRACT[key]) for key in allowed}
    for key, value in result.items():
        integer = key in ('maximum_sweeps', 'maximum_geometry_calls')
        _require(type(value) in ((int,) if integer else (int, float)) and math.isfinite(value)
                 and (1 if integer else .001) <= value <= CONTRACT[key], 'invalid tighter work bound')
    return result


def _targets(frame, origin, width, height, scale):
    targets, weights, details = np.zeros((12, 2)), np.zeros(12), []
    raw = frame.get('identity_raw', frame)
    _require(isinstance(raw, dict) and all(raw.get(key) == frame.get(key) for key in ('id', 'frame', 'time_s')),
             'raw source identity differs')
    source_name = 'identity_raw.landmarks' if 'identity_raw' in frame else 'landmarks'
    frame = raw
    landmarks = frame.get('landmarks')
    available = (frame.get('status') in ('tracked', 'tracked_partial') and frame.get('pose_count', 1) == 1
                 and origin['status'] == 'observed_image_estimate' and origin['pelvis_image_px'] is not None
                 and isinstance(landmarks, list) and len(landmarks) == 33)
    for i, (name, index) in enumerate(POINTS.items()):
        item = {'joint': name, 'status': 'unavailable_source_endpoint', 'weight': 0.,
                'source_confidence': None, 'landmark_source': source_name}
        details.append(item)
        if not available:
            continue
        point = landmarks[index]
        if not isinstance(point, dict):
            continue
        values = [point.get(k) for k in ('x', 'y', 'visibility', 'presence')]
        if any(type(v) not in (int, float) or not math.isfinite(v) for v in values):
            continue
        x, y, visibility, presence = values
        if not 0 <= min(visibility, presence) <= max(visibility, presence) <= 1:
            continue
        item['source_confidence'] = min(visibility, presence)
        if not 0 <= x <= 1 or not 0 <= y <= 1:
            item['status'] = 'outside_source_image'
            continue
        confidence = min(visibility, presence, origin['confidence'])
        item['status'] = 'low_confidence'
        if confidence >= CONTRACT['minimum_confidence']:
            targets[i] = (np.array([x * width, y * height]) - origin['pelvis_image_px']) * scale
            weights[i] = confidence
            item.update(status='available_image_estimate', weight=float(confidence), target_model_units=targets[i].tolist())
    return targets, weights, details


def _prepare(spec, evidence):
    _require(spec.get('profile') == 'exercise-v3' and spec.get('schema_version') == 3 and spec.get('dt') == 1 / 30,
             'only original 30 Hz mobile bodies are supported')
    _require('camera_reference' not in spec and not any('refinement' in key or 'alignment' in key for key in spec),
             'supplied-camera and stacked corrections are unsupported')
    source = spec.get('source_sha256')
    _require(isinstance(source, str) and len(source) == 64 and all(c in '0123456789abcdef' for c in source)
             and evidence.get('source_sha256') == source, 'source identity differs')
    width, height = evidence.get('width'), evidence.get('height')
    provenance = evidence.get('image_dimension_provenance', {})
    _require(all(type(v) is int and 16 <= v <= 8192 for v in (width, height))
             and isinstance(provenance, dict) and provenance.get('method') in
             ('verified_source_video', 'authored_analytic_fixture', 'sealed_decoder_metadata')
             and provenance.get('source_sha256') == source and provenance.get('width') == width
             and provenance.get('height') == height, 'source dimensions are not bound')
    if provenance['method'] == 'sealed_decoder_metadata':
        keys = ('source_evidence_sha256', 'source_evidence_canonical_sha256', 'source_manifest_sha256')
        _require(all(isinstance(provenance.get(key), str) and len(provenance[key]) == 64
                     and all(c in '0123456789abcdef' for c in provenance[key]) for key in keys),
                 'sealed decoder provenance requires exact evidence/manifest hash strings')
        original_evidence = evidence.copy()
        original_evidence.pop('image_dimension_provenance')
        _require(canonical_hash(original_evidence) == provenance['source_evidence_canonical_sha256'],
                 'derived evidence changed beyond its dimension provenance')
    references = spec.get('reference')
    _require(isinstance(references, list) and 3 <= len(references) <= CONTRACT['maximum_references'], 'reference count')
    times, angles = [], []
    for row in references:
        _require(isinstance(row, dict) and set(row) == {'time_s', 'joints'}, 'reference shape')
        stamp, joints = row['time_s'], row['joints']
        _require(type(stamp) in (int, float) and math.isfinite(stamp) and 0 <= stamp <= 60
                 and (not times or times[-1] < stamp <= times[-1] + .25 + 1e-9), 'source clock or unsupported gap')
        _require(isinstance(joints, dict) and set(joints) == set(NAMES), 'all 19 original axes are required')
        for name in NAMES:
            value = joints[name]
            _require(type(value) in (int, float) and math.isfinite(value)
                     and SPATIAL_JOINT_LIMITS[name][0] <= value <= SPATIAL_JOINT_LIMITS[name][1], 'joint bounds')
        times.append(stamp); angles.append([joints[name] for name in NAMES])
    _require(isinstance(spec.get('lengths'), dict) and 'torso' in spec['lengths'], 'fixed original torso length required')
    root = build_root_reference(evidence, torso_length=spec['lengths']['torso'])
    expected = [{'time_s': row['time_s'], 'translation': row['translation']} for row in root['reference']]
    _require(root['status'] == 'ready' and expected == spec.get('root_reference')
             and len(expected) == len(references), 'source root/scale must remain exact')
    meta = spec.get('reference_evidence')
    _require(isinstance(meta, list) and len(meta) == len(references), 'complete source provenance required')
    frames = {row['id']: row for row in evidence['frames']}
    prepared = []
    for reference, record, origin in zip(references, meta, root['reference'], strict=True):
        _require(isinstance(record, dict) and reference['time_s'] == record.get('time_s') == origin['time_s']
                 and record.get('evidence_id') == origin['evidence_id'], 'source identity/clock differs')
        frame = frames[origin['evidence_id']]
        target, weights, points = _targets(frame, origin, width, height, root['scale']['model_units_per_pixel'])
        prepared.append({'target': target, 'weights': weights, 'points': points, 'evidence_id': origin['evidence_id'],
                         'frame': frame['frame'], 'time_s': origin['time_s'], 'root_status': origin['status']})
    return np.array(times), np.array(angles), np.array([row['translation'] for row in expected]), prepared, root


class _NativeGeometry:
    def __init__(self, spec):
        import mujoco
        from weft.body_refinement import _kinematics
        from weft.human import HumanEnv

        self.mujoco, self.kinematics, self.env = mujoco, _kinematics, HumanEnv(spec)
        self.data = mujoco.MjData(self.env.model)
        self.floor = self.env.mobility.floor
        self.supports = []
        model = self.env.model
        try:
            _require(int(model.geom_type[self.floor]) == mujoco.mjtGeom.mjGEOM_PLANE, 'fixed plane required')
            for geom in range(model.ngeom):
                if model.geom_bodyid[geom] == 0:
                    continue
                # Pybind enums do not reliably compare equal to NumPy integer scalars.
                kind = int(model.geom_type[geom])
                _require(kind in (mujoco.mjtGeom.mjGEOM_SPHERE, mujoco.mjtGeom.mjGEOM_CAPSULE), 'unsupported human collision primitive')
                ends = (-1., 1.) if kind == mujoco.mjtGeom.mjGEOM_CAPSULE else (0.,)
                for end in ends:
                    self.supports.append((model.geom(geom).name, geom, end))
            _require(1 <= len(self.supports) <= CONTRACT['maximum_floor_supports'], 'floor support count')
        except Exception:
            self.close()
            raise
        self.support_names = tuple(name + ':' + str(end) for name, _, end in self.supports)
        self.primitive_names = tuple(name for name, _, _ in self.supports)

    def evaluate(self, angles, root, derivatives=False):
        self.data.qpos[self.env.mobility.q] = root
        points, point_jac = self.kinematics(self.env, self.data, angles, derivatives)
        model, data = self.env.model, self.data
        normal = data.geom_xmat[self.floor].reshape(3, 3)[:, 2]
        origin = data.geom_xpos[self.floor]
        gaps, rows = [], []
        scratch = np.zeros((3, model.nv))
        for _, geom, end in self.supports:
            endpoint = data.geom_xpos[geom] + end * model.geom_size[geom, 1] * data.geom_xmat[geom].reshape(3, 3)[:, 2]
            gaps.append(float(normal @ (endpoint - origin) - model.geom_size[geom, 0]))
            if derivatives:
                self.mujoco.mj_jac(model, data, scratch, None, endpoint, int(model.geom_bodyid[geom]))
                rows.append(normal @ scratch[:, self.env._v_array])
        return {'points': points, 'gaps': np.array(gaps),
                'point_jac': point_jac.reshape(12, 3, 19) if derivatives else None,
                'gap_jac': np.array(rows) if derivatives else None}

    def close(self):
        self.env.close()


def _make_geometry(spec):
    return _NativeGeometry(spec)


def _schedule(times):
    count = math.floor((times[-1] - times[0]) * 30 + 1e-9) + 1
    _require(count <= CONTRACT['maximum_grid_samples'], 'control-grid bound')
    rows = [('source_pose', float(t), i) for i, t in enumerate(times)]
    rows += [('control_grid', min(float(times[-1]), float(times[0] + i / 30)), None) for i in range(count)]
    samples, lookup = [], {}
    for _, stamp, _ in rows:
        if stamp in lookup:
            continue
        index = min(len(times) - 2, max(0, int(np.searchsorted(times, stamp, side='right')) - 1))
        weight = (stamp - times[index]) / (times[index + 1] - times[index])
        weights = [(index, 1 - weight), (index + 1, weight)]
        weights = [(i, w) for i, w in weights if w != 0]
        lookup[stamp] = len(samples); samples.append({'time_s': stamp, 'weights': weights})
    return rows, samples, lookup


def _project(points):
    return points[:, [0, 2]] * [1., -1.]


def _loss(row, positions, delta, neighbors):
    residual = _project(positions) - row['target']
    norms = np.linalg.norm(residual, axis=1); huber = CONTRACT['huber_delta_model_units']
    cost = np.sum(row['weights'] * np.where(norms <= huber, .5 * norms ** 2, huber * (norms - .5 * huber)))
    cost += .5 * CONTRACT['original_angle_regularization'] * (delta @ delta)
    cost += .5 * CONTRACT['correction_temporal_regularization'] * sum(float((delta - n) @ (delta - n)) for n in neighbors)
    return float(cost)


def propose_whole_body_reference_refinement(human_document: bytes, evidence_document: bytes, *, joint_mask=None, settings=None):
    """Return a separately proposed body; no source, default runtime or policy is modified."""
    started = time.monotonic(); limits = _settings(settings)
    spec, evidence = _document(human_document), _document(evidence_document)
    mask = NAMES if joint_mask is None else joint_mask
    _require(isinstance(mask, (list, tuple)) and 1 <= len(mask) <= 19 and all(isinstance(n, str) for n in mask)
             and len(set(mask)) == len(mask) and set(mask) <= set(NAMES), 'joint mask must select unique existing axes')
    blocks = [np.array([NAMES.index(name) for name in block if name in mask]) for block in BLOCKS]
    blocks = [block for block in blocks if len(block)]
    times, original, roots, rows, root_audit = _prepare(spec, evidence)
    schedule, samples, lookup = _schedule(times)
    _require(limits['maximum_geometry_calls'] >= 2 * len(samples), 'work bound cannot cover baseline and final audit')
    changes = np.zeros_like(original); affected = [[] for _ in times]
    for j, sample in enumerate(samples):
        for index, weight in sample['weights']:
            affected[index].append((j, weight))
    calls, accepted, rejected, qp_calls = 0, 0, 0, 0
    expired = False; failure = None; phase = 'fit'
    deadline = started + limits['maximum_seconds']; final_deadline = deadline + limits['finalization_seconds']
    tolerance = CONTRACT['numerical_tolerance']
    if time.monotonic() >= deadline:
        raise TimeoutError('bounded construction budget exhausted before initial geometry')
    geometry = _make_geometry(spec)
    support_names = tuple(geometry.support_names); primitive_names = tuple(geometry.primitive_names)
    def check(needs_geometry=False):
        # Reserve a complete final audit even when fitting consumes its call budget.
        call_limit = limits['maximum_geometry_calls'] - (len(samples) if phase == 'fit' else 0)
        if time.monotonic() >= (deadline if phase == 'fit' else final_deadline) or (needs_geometry and calls >= call_limit):
            raise TimeoutError('bounded construction budget exhausted')

    def state(engine, q, root, derivatives=False):
        nonlocal calls
        check(needs_geometry=True); calls += 1
        result = engine.evaluate(q.copy(), root.copy(), derivatives)
        for key, shape in (('points', (12, 3)), ('gaps', (len(support_names),)),
                           *([('point_jac', (12, 3, 19)), ('gap_jac', (len(support_names), 19))] if derivatives else [])):
            result[key] = np.asarray(result[key], dtype=float)
            _require(result[key].shape == shape and np.isfinite(result[key]).all()
                     and np.max(abs(result[key]), initial=0.) <= 1e6, 'invalid bounded kinematic output')
        check()
        return result

    def at(sample, delta, engine, derivatives=False):
        q = sum((original[i] + delta[i]) * w for i, w in sample['weights'])
        root = sum(roots[i] * w for i, w in sample['weights'])
        return state(engine, q, root, derivatives)

    try:
        _require(1 <= len(support_names) <= CONTRACT['maximum_floor_supports'] and len(set(support_names)) == len(support_names)
                 and len(primitive_names) == len(support_names), 'complete unique geometry support identities required')
        for sample in samples:
            sample['before'] = at(sample, changes, geometry)
            sample['current'] = sample['before']
        for i, row in enumerate(rows):
            row['before_errors'] = np.linalg.norm(_project(samples[lookup[float(times[i])]]['before']['points']) - row['target'], axis=1)
        lower = np.maximum(-.15, np.array([SPATIAL_JOINT_LIMITS[n][0] for n in NAMES]) - original)
        upper = np.minimum(.15, np.array([SPATIAL_JOINT_LIMITS[n][1] for n in NAMES]) - original)

        for _ in range(limits['maximum_sweeps']):
            for i in range(1, len(rows)):
                check(); row = rows[i]
                if not np.any(row['weights']):
                    continue
                adjacent = [j for j in (i - 1, i + 1) if 0 <= j < len(rows) and abs(times[j] - times[i]) <= .25 + 1e-9]
                for columns in blocks:
                    check(); n = len(columns); neighbors = [changes[j] for j in adjacent]
                    current = samples[lookup[float(times[i])]]['current']
                    current_errors = np.linalg.norm(_project(current['points']) - row['target'], axis=1)
                    image_limit = np.minimum(row['before_errors'], current_errors)
                    cost = _loss(row, current['points'], changes[i], neighbors)
                    full = state(geometry, original[i] + changes[i], roots[i], True)
                    residual = (_project(full['points']) - row['target']).ravel()
                    jac = np.take(full['point_jac'][:, [0, 2], :] * np.array([1., -1.])[None, :, None], columns, axis=2).reshape(24, n)
                    robust = row['weights'] * np.minimum(1., .05 / np.maximum(np.linalg.norm(residual.reshape(12, 2), axis=1), 1e-15))
                    diagonal = np.repeat(robust, 2)
                    prior, temporal = .001, .0005
                    hessian = jac.T @ (diagonal[:, None] * jac) + np.eye(n) * (prior + temporal * len(adjacent) + 1e-6)
                    gradient = jac.T @ (diagonal * residual) + prior * changes[i, columns]
                    for neighbor in neighbors:
                        gradient += temporal * (changes[i, columns] - neighbor[columns])
                    lo, hi = lower[i].copy(), upper[i].copy()
                    for j in adjacent:
                        radius = 1.5 * abs(times[i] - times[j])
                        lo, hi = np.maximum(lo, changes[j] - radius), np.minimum(hi, changes[j] + radius)
                    lo = np.maximum(lo[columns] - changes[i, columns], -.10)
                    hi = np.minimum(hi[columns] - changes[i, columns], .10)
                    matrix, bounds = [*np.eye(n), *-np.eye(n)], [*lo, *-hi]
                    for j, weight in affected[i]:
                        sample = samples[j]; value = at(sample, changes, geometry, True)
                        threshold = np.maximum(np.minimum(0., sample['before']['gaps']), np.minimum(0., sample['current']['gaps']))
                        matrix.extend(value['gap_jac'][:, columns] * weight)
                        bounds.extend(threshold - value['gaps'] - tolerance)
                        depth = value['points'][:, 1] - sample['before']['points'][:, 1]
                        depth_jac = value['point_jac'][:, 1, :][:, columns] * weight
                        matrix.extend(depth_jac); bounds.extend(-.08 - depth - tolerance)
                        matrix.extend(-depth_jac); bounds.extend(depth - .08 - tolerance)
                    for point in np.flatnonzero(row['weights']):
                        r = residual.reshape(12, 2)[point]; pj = jac.reshape(12, 2, n)[point]
                        norm = float(np.linalg.norm(r))
                        if norm > 1e-12:
                            matrix.append(-r @ pj / norm); bounds.append(norm - image_limit[point] - tolerance)
                        else:
                            matrix.extend(pj); bounds.extend([-tolerance / 2] * 2)
                            matrix.extend(-pj); bounds.extend([-tolerance / 2] * 2)
                    a, b = np.asarray(matrix), np.asarray(bounds)
                    check(); qp_calls += 1
                    solved = solve_feasible_qp(hessian, gradient, a, b, initial=np.zeros(n),
                        settings={'maximum_seconds': max(.001, min(.5, deadline - time.monotonic()))})
                    if solved['status'] != 'optimal':
                        rejected += 1; continue
                    direction = np.asarray(solved['step'], dtype=float)
                    if direction.shape != (n,) or not np.isfinite(direction).all() or np.max(b - a @ direction, initial=0.) > 1e-10:
                        rejected += 1; continue
                    if np.max(abs(direction), initial=0.) < 1e-10:
                        continue
                    for backtrack in range(CONTRACT['maximum_backtracks']):
                        check(); trial = changes.copy(); trial[i, columns] += direction * .5 ** backtrack
                        states = []; valid = True
                        for j, _ in affected[i]:
                            sample = samples[j]; value = at(sample, trial, geometry)
                            threshold = np.maximum(np.minimum(0., sample['before']['gaps']), np.minimum(0., sample['current']['gaps']))
                            valid &= bool(np.all(value['gaps'] >= threshold - tolerance)
                                          and np.max(abs(value['points'][:, 1] - sample['before']['points'][:, 1])) <= .08 + tolerance)
                            states.append((j, value))
                        point_state = next(value for j, value in states if j == lookup[float(times[i])])
                        errors = np.linalg.norm(_project(point_state['points']) - row['target'], axis=1)
                        valid &= bool(np.all(errors[row['weights'] > 0] <= image_limit[row['weights'] > 0] + tolerance))
                        if valid and _loss(row, point_state['points'], trial[i], neighbors) < cost - 1e-14:
                            changes = trial; accepted += 1
                            for j, value in states:
                                samples[j]['current'] = value
                            break
                        rejected += 1
    except TimeoutError as error:
        expired, failure = True, str(error)
    finally:
        geometry.close()

    # A new geometry instance recomputes every final sample, never trusting optimizer caches.
    phase = 'final'; final_deadline = min(final_deadline, time.monotonic() + limits['finalization_seconds'])
    _require(all('before' in sample for sample in samples), 'incomplete baseline; no candidate can be qualified')
    check()
    geometry = _make_geometry(spec); checks = []
    try:
        _require(tuple(geometry.support_names) == support_names and tuple(geometry.primitive_names) == primitive_names,
                 'final geometry inventory changed')
        final = {}
        for sample in samples:
            final[sample['time_s']] = at(sample, changes, geometry)
        for kind, stamp, index in schedule:
            baseline = samples[lookup[stamp]]['before']; after = final[stamp]
            checks.append({'kind': kind, 'time_s': stamp, 'source_reference_index': index,
                'before_floor_support_gaps': baseline['gaps'].tolist(), 'after_floor_support_gaps': after['gaps'].tolist(),
                'maximum_depth_change_model_units': float(np.max(abs(after['points'][:, 1] - baseline['points'][:, 1]))),
                'no_worse_floor': bool(np.all(after['gaps'] >= np.minimum(0., baseline['gaps']) - tolerance)),
                'before_max_penetration_model_units': float(np.maximum(0., -baseline['gaps']).max()),
                'after_max_penetration_model_units': float(np.maximum(0., -after['gaps']).max())})
        for i, row in enumerate(rows):
            row['after_errors'] = np.linalg.norm(_project(final[float(times[i])]['points']) - row['target'], axis=1)
    finally:
        geometry.close()
    candidate = deepcopy(spec)
    for i, row in enumerate(candidate['reference']):
        for j, name in enumerate(NAMES):
            if changes[i, j] != 0:
                row['joints'][name] = float(original[i, j] + changes[i, j])
    rate = float(np.max(abs(np.diff(changes, axis=0)) / np.diff(times)[:, None]))
    image_ok = all(np.all(row['after_errors'][row['weights'] > 0] <= row['before_errors'][row['weights'] > 0] + tolerance) for row in rows)
    gates = {'complete_bounded_solve': not expired, 'first_reference_frozen': candidate['reference'][0] == spec['reference'][0],
        'fixed_nonreference_fields': all(candidate[key] == value for key, value in spec.items() if key != 'reference'),
        'source_times_and_count': len(candidate['reference']) == len(times) and all(row['time_s'] == times[i] for i, row in enumerate(candidate['reference'])),
        'angle_correction': float(np.max(abs(changes))) <= .15 + tolerance, 'correction_rate': rate <= 1.5 + tolerance,
        'joint_limits': all(SPATIAL_JOINT_LIMITS[name][0] <= row['joints'][name] <= SPATIAL_JOINT_LIMITS[name][1]
                            for row in candidate['reference'] for name in NAMES),
        'pointwise_floor_nonregression': all(row['no_worse_floor'] for row in checks),
        'floor_feasibility': all(row['after_max_penetration_model_units'] <= .025 for row in checks),
        'endpoint_depth_change': all(row['maximum_depth_change_model_units'] <= .08 + tolerance for row in checks),
        'source_endpoint_image_nonregression': bool(image_ok)}
    source_rows = []
    for i, row in enumerate(rows):
        points = [dict(point, error_before_px=float(row['before_errors'][j] / root_audit['scale']['model_units_per_pixel']) if point['weight'] else None,
                       error_after_px=float(row['after_errors'][j] / root_audit['scale']['model_units_per_pixel']) if point['weight'] else None)
                  for j, point in enumerate(row['points'])]
        source_rows.append({key: row[key] for key in ('evidence_id', 'frame', 'time_s', 'root_status')} | {
            'status': 'frozen_first_reference' if i == 0 else 'unchanged_missing_source_targets' if not np.any(row['weights']) else
                      'changed_inferred_proposal' if np.any(changes[i]) else 'unchanged_no_accepted_step',
            'points': points, 'correction_rad': dict(zip(NAMES, changes[i].tolist()))})
    failures = [key for key, passed in gates.items() if not passed]
    status = 'partial_budget_exhausted' if expired else 'rejected_construction_proposal' if failures else 'unadmitted_construction_proposal'
    candidate['whole_body_reference_refinement'] = {
        'schema_version': 1, 'method': CONTRACT['method'], 'status': status, 'promoted': False,
        'input_human_sha256': sha256(human_document).hexdigest(), 'input_evidence_sha256': sha256(evidence_document).hexdigest()}
    before_peak = max(row['before_max_penetration_model_units'] for row in checks)
    after_peak = max(row['after_max_penetration_model_units'] for row in checks)
    image_pairs = [(row['before_errors'][j], row['after_errors'][j])
                   for row in rows for j in np.flatnonzero(row['weights'])]
    image_mean = np.mean(image_pairs, axis=0) / root_audit['scale']['model_units_per_pixel'] if image_pairs else (None, None)
    report = {'schema_version': 1, 'profile': CONTRACT['method'], 'status': status, 'contract': deepcopy(CONTRACT),
        'work_limits': limits, 'input_human_sha256': sha256(human_document).hexdigest(),
        'input_evidence_sha256': sha256(evidence_document).hexdigest(), 'candidate_sha256': canonical_hash(candidate),
        'source_sha256': spec['source_sha256'], 'root_audit_sha256': canonical_hash(root_audit),
        'joint_mask': list(mask), 'blocks': [[NAMES[j] for j in block] for block in blocks],
        'support_names': list(support_names), 'support_primitive_names': list(primitive_names),
        'retained_pose_slots': [{key: row.get(key) for key in ('id', 'frame', 'time_s', 'decoder_timestamp_s', 'status')} for row in evidence['frames']],
        'source_counts': {'retained_pose_slots': len(evidence['frames']), 'body_reference_knots': len(rows),
            'declared_frame_count': evidence['frame_count'], 'declared_decoded_frames': evidence.get('decoded_frames'),
            'scope': 'Retained inference slots only; unsampled decoded slots and missing decoder timestamps are not reconstructed'},
        'image_dimensions': {'width': evidence['width'], 'height': evidence['height'],
            'provenance': deepcopy(evidence['image_dimension_provenance']),
            'verification_scope': 'Caller-supplied provenance identifiers; this function does not authenticate an external manifest or decode video'},
        'references': source_rows, 'geometry_checks': checks, 'gates': gates, 'failures': failures,
        'accepted_updates': accepted, 'rejected_trials_or_directions': rejected, 'qp_calls': qp_calls, 'geometry_calls': calls,
        'changed_references': int(np.count_nonzero(np.max(abs(changes), axis=1))), 'maximum_correction_rate_rad_s': rate,
        'image_summary': {'confident_source_endpoint_pairs': len(image_pairs),
            'mean_error_before_px': float(image_mean[0]) if image_pairs else None,
            'mean_error_after_px': float(image_mean[1]) if image_pairs else None,
            'grid_image_targets': 0},
        'floor_summary': {'checks': len(checks), 'before_failures': sum(row['before_max_penetration_model_units'] > .025 for row in checks),
            'after_failures': sum(row['after_max_penetration_model_units'] > .025 for row in checks),
            'before_maximum_penetration_model_units': before_peak,
            'after_maximum_penetration_model_units': after_peak,
            'maximum_penetration_reduction_model_units': before_peak - after_peak,
            'samples_with_reduced_penetration': sum(row['after_max_penetration_model_units'] < row['before_max_penetration_model_units'] - tolerance for row in checks),
            'source_pose_checks': sum(row['kind'] == 'source_pose' for row in checks),
            'control_grid_checks': sum(row['kind'] == 'control_grid' for row in checks)},
        'seconds': time.monotonic() - started, 'budget_reason': failure, 'physics_steps': 0, 'promoted': False,
        'physical_validation': 'unverified', 'geometry_admission': 'unadmitted_pending_independent_current_scene_checks',
        'scope': 'Source-estimate agreement and sampled authored-floor constraints only; missing endpoints stay unknown. No equipment, anatomy, contact, balance or continuous-time qualification.'}
    result = {'status': status, 'human_candidate': candidate, 'report': report}
    check()
    _require(len(json.dumps(result, allow_nan=False).encode()) <= CONTRACT['maximum_output_bytes'], 'complete output exceeds bound')
    check()
    return result
