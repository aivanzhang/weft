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
FLOOR_FIRST_METHOD = 'blocked-whole-body-floor-first-reference-v1'
SURFACE_INTERVAL_METHOD = 'blocked-whole-body-surface-interval-reference-v1'
SURFACE_INTERVAL_PROFILE = 'whole-body-surface-interval-fit-v1'
SURFACE_INTERVAL_LIMITS = {'maximum_evaluations': 20000, 'maximum_journal_bytes': 4 << 20}
_SURFACE_RECORD_BYTES = 16 << 10
_SURFACE_AUDIT_BYTES = 4 << 20
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


def _target_origin(frame):
    if 'image_refinement' not in frame:
        _require('image_refinement_raw' not in frame, 'crop original lacks its selection marker')
        return None
    marker, original = frame['image_refinement'], frame.get('image_refinement_raw')
    _require('identity_raw' not in frame and isinstance(marker, dict)
             and marker.get('method') == 'single-view-selective-crop-v1'
             and marker.get('anatomical_identity_verified') is False, 'unsupported or mixed crop target provenance')
    fields = ('id', 'frame', 'time_s', 'decoder_timestamp_s')
    _require(isinstance(original, dict)
             and all(key in original and key in frame and original[key] == frame[key] for key in fields)
             and not any(key in original for key in ('identity_raw', 'image_refinement', 'image_refinement_raw')),
             'crop original identity, clocks or unmodified provenance differ')
    hashes = ('original_observation_sha256', 'proposal_sha256', 'decision_sha256')
    _require(all(isinstance(marker.get(key), str) and len(marker[key]) == 64
                 and all(c in '0123456789abcdef' for c in marker[key]) for key in hashes),
             'crop target provenance requires exact SHA256 identities')
    proposal = {key: value for key, value in frame.items() if key not in ('image_refinement', 'image_refinement_raw')}
    _require(marker['original_observation_sha256'] == canonical_hash(original)
             and marker['proposal_sha256'] == canonical_hash(proposal), 'crop target observation identity differs')
    # Full decision/mapping authentication belongs to the source adapter; targets stay unchanged.
    return {'kind': 'selected_crop_estimate_mapped_to_source', 'method': marker['method'],
            **{key: marker[key] for key in hashes}, 'anatomical_identity_verified': False}


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


def _prepare(spec, evidence, *, camera_fit=None):
    period = spec.get('dt')
    period_ok = (type(period) in (int, float) and math.isfinite(period) and 1 / 120 <= period <= 1 / 20
                 if camera_fit is not None else period == 1 / 30)
    _require(spec.get('profile') == 'exercise-v3' and spec.get('schema_version') == 3 and period_ok,
             'only original camera mobile bodies with the declared bounded clock are supported' if camera_fit is not None
             else 'only original 30 Hz mobile bodies are supported')
    _require((camera_fit is not None or 'camera_reference' not in spec)
             and not any('refinement' in key or 'alignment' in key for key in spec),
             'supplied-camera and stacked corrections are unsupported')
    source = spec.get('source_sha256')
    _require(isinstance(source, str) and len(source) == 64 and all(c in '0123456789abcdef' for c in source)
             and evidence.get('source_sha256') == source, 'source identity differs')
    width, height = evidence.get('width'), evidence.get('height')
    provenance = evidence.get('image_dimension_provenance', {})
    _require(all(type(v) is int and 16 <= v <= 8192 for v in (width, height))
             and isinstance(provenance, dict) and (camera_fit is not None or provenance.get('method') in
             ('verified_source_video', 'authored_analytic_fixture', 'sealed_decoder_metadata')
             and provenance.get('source_sha256') == source and provenance.get('width') == width
             and provenance.get('height') == height), 'source dimensions are not bound')
    if provenance.get('method') == 'sealed_decoder_metadata':
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
    if camera_fit is not None:
        return (np.array(times), np.array(angles), np.array([row['translation'] for row in spec['root_reference']]),
                camera_fit.rows(), camera_fit.describe())
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
        target_origin = _target_origin(frame)
        target, weights, points = _targets(frame, origin, width, height, root['scale']['model_units_per_pixel'])
        prepared.append({'target': target, 'weights': weights, 'points': points, 'evidence_id': origin['evidence_id'],
                         'frame': frame['frame'], 'time_s': origin['time_s'], 'root_status': origin['status'],
                         **({'target_origin': target_origin} if target_origin is not None else {})})
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

    def evaluate(self, angles, root, derivatives=False, *, surface_intervals=False):
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
        result = {'points': points, 'gaps': np.array(gaps),
                'point_jac': point_jac.reshape(12, 3, 19) if derivatives else None,
                'gap_jac': np.array(rows) if derivatives else None}
        if surface_intervals:
            pelvis = self.env._site_ids['pelvis']
            result['pelvis_position'] = data.site_xpos[pelvis].copy()
            if derivatives:
                self.mujoco.mj_jacSite(model, data, scratch, None, pelvis)
                result['pelvis_jac'] = scratch[:, self.env._v_array].copy()
        return result

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


def _loss(row, positions, delta, neighbors, *, projected=False):
    residual = (positions if projected else _project(positions)) - row['target']
    norms = np.linalg.norm(residual, axis=1); huber = CONTRACT['huber_delta_model_units']
    cost = np.sum(row['weights'] * np.where(norms <= huber, .5 * norms ** 2, huber * (norms - .5 * huber)))
    cost += .5 * CONTRACT['original_angle_regularization'] * (delta @ delta)
    cost += .5 * CONTRACT['correction_temporal_regularization'] * sum(float((delta - n) @ (delta - n)) for n in neighbors)
    return float(cost)


def _floor_loss(states):
    return .5 * sum(float(np.square(np.minimum(0., state['gaps'] + .025)).sum()) for state in states)


class _SurfaceIntervalFit:
    """Bound the opt-in term and retain every source slot and attempted correction."""

    def __init__(self, objective, human_raw, evidence, rows, weight, settings, *, source_frame_count=None):
        # Lazy import avoids a cycle with the released consumer's endpoint inventory.
        from weft.source_surface_interval_construction import SurfaceIntervalObjective

        _require(type(objective) is SurfaceIntervalObjective and objective.body_bytes == human_raw,
                 'surface interval objective requires the exact original body bytes')
        _require(type(weight) in (int, float) and math.isfinite(weight) and 0 < weight <= 1000,
                 'explicit authored surface weight must be finite in (0, 1000]')
        settings = {} if settings is None else settings
        _require(type(settings) is dict and set(settings) <= set(SURFACE_INTERVAL_LIMITS), 'unknown surface interval work limit')
        limits = {key: settings.get(key, value) for key, value in SURFACE_INTERVAL_LIMITS.items()}
        for key, maximum in SURFACE_INTERVAL_LIMITS.items():
            minimum = 2 * len(rows) if key == 'maximum_evaluations' else _SURFACE_RECORD_BYTES
            _require(type(limits[key]) is int and minimum <= limits[key] <= maximum, 'surface bound cannot cover complete final audit/journal')
        self.objective, self.weight, self.limits = objective, float(weight), limits
        self.inputs = objective.describe()
        _require(self.inputs['source_sha256'] == evidence['source_sha256']
                 and objective.frame_count == (evidence['frame_count'] if source_frame_count is None else source_frame_count),
                 'surface and complete original pose source differ')
        self.body_hash, self.coordinate_frame = sha256(human_raw).hexdigest(), objective.coordinate_frame
        self.keys = [{'reference_index': i, 'source_frame': row['frame'], 'evidence_id': row['evidence_id'], 'time_s': row['time_s']}
                     for i, row in enumerate(rows)]
        self.slots, source_bytes, estimates = [], 0, {}
        references = {row['frame']: i for i, row in enumerate(rows)}
        for frame in range(objective.frame_count):
            value = objective.frame(frame)
            source = value.pop('source_envelopes')
            for channel, anchor in zip(value['channels'], source['anchors'], strict=True):
                channel['world_envelope_m'] = anchor['world_envelope_m']
            value['source_status'] = source['source_status']
            encoded = json.dumps(value, allow_nan=False, separators=(',', ':')).encode()
            source_bytes += len(encoded)
            _require(source_bytes <= _SURFACE_AUDIT_BYTES, 'complete surface source audit exceeds four MiB before geometry')
            if frame in references:
                index = references[frame]
                expected = {'reference_index': index, 'evidence_id': rows[index]['evidence_id'], 'body_time_s': rows[index]['time_s']}
                _require(value['body_reference'] == expected, 'surface body reference identity or original clock differs')
                # The fixed consumer adds <=640 bytes/channel of bounded numeric output, no derivatives in audits.
                estimates[index] = len(encoded) + 1024 + 640 * len(value['channels'])
            self.slots.append(value)
        _require(len(estimates) == len(rows) and 2 * sum(estimates.values()) <= _SURFACE_AUDIT_BYTES,
                 'complete before/final surface audit cannot fit four MiB; no geometry executed')
        self.estimates = estimates
        self.before, self.current, self.after = {}, {}, {}
        self.evaluations, self.journal_bytes = 0, 0
        self.journal, self.phase = [], 'fit'

    def evaluate(self, index, state, *, derivatives=False):
        limit = self.limits['maximum_evaluations'] - (len(self.keys) if self.phase == 'fit' else 0)
        if self.evaluations >= limit:
            raise TimeoutError('surface interval evaluation bound exhausted; final audit reserved')
        key = self.keys[index]
        self.evaluations += 1
        return self.objective.evaluate_kinematics(key['source_frame'], body_sha256=self.body_hash,
            body_time_s=key['time_s'], coordinate_frame=self.coordinate_frame, point_names=tuple(POINTS), joint_names=NAMES,
            relative_positions=state['points'], pelvis_position=state['pelvis_position'],
            relative_jacobian=state['point_jac'].reshape(36, 19) if derivatives else None,
            pelvis_jacobian=state['pelvis_jac'] if derivatives else None)

    def save_audit(self, index, term, *, final=False):
        _require(len(json.dumps(term, allow_nan=False, separators=(',', ':')).encode()) <= self.estimates[index],
                 'surface audit exceeded its preflight reservation')
        (self.after if final else self.before)[index] = term
        if not final:
            self.current[index] = term

    def reserve_journal(self):
        if self.journal_bytes + _SURFACE_RECORD_BYTES > self.limits['maximum_journal_bytes']:
            raise TimeoutError('surface interval trial-journal bound exhausted; no attempted correction dropped')

    def record(self, value):
        value = {'sequence': len(self.journal), **value}
        encoded = json.dumps(value, allow_nan=False, separators=(',', ':')).encode()
        _require(len(encoded) <= _SURFACE_RECORD_BYTES, 'surface trial exceeds its reserved record bound')
        _require(self.journal_bytes + len(encoded) <= self.limits['maximum_journal_bytes'], 'surface trial journal exceeds its reservation')
        self.journal.append(value); self.journal_bytes += len(encoded)

    def finish_record(self, **changes):
        original = self.journal[-1]
        result = dict(original, **changes)
        old_bytes = len(json.dumps(original, allow_nan=False, separators=(',', ':')).encode())
        new_bytes = len(json.dumps(result, allow_nan=False, separators=(',', ':')).encode())
        _require(new_bytes <= _SURFACE_RECORD_BYTES and self.journal_bytes - old_bytes + new_bytes <= self.limits['maximum_journal_bytes'],
                 'completed surface trial exceeded its reservation')
        self.journal[-1] = result
        self.journal_bytes += new_bytes - old_bytes

    def record_qp(self, index, columns, correction, solved, outcome, violation=None):
        direction = solved.get('step')
        step = None
        if direction is not None:
            values = np.asarray(direction)
            _require(values.size <= 64, 'invalid solver direction cannot fit the bounded surface journal')
            step = [float(value) if math.isfinite(float(value)) else {'nonfinite': str(float(value))} for value in values.ravel()]
        self.record({'kind': 'qp', 'reference_index': index, 'source_frame': self.keys[index]['source_frame'],
            'block': [NAMES[j] for j in columns], 'base_correction_rad': correction.tolist(),
            'qp_status': solved['status'], 'step_rad': step, 'iterations': solved.get('iterations'),
            'outcome': outcome, 'constraint_violation': violation})

    @staticmethod
    def cost(term):
        return term['objective_m2'] if term['objective_m2'] is not None else 0.

    @staticmethod
    def nonregression(before, after):
        if before['counts'] != after['counts']:
            return False
        for old, new in zip(before['channels'], after['channels'], strict=True):
            if (old['channel_id'], old['status'], old['reason']) != (new['channel_id'], new['status'], new['reason']):
                return False
            if old['distance_to_outer_envelope_m'] is not None and new['distance_to_outer_envelope_m'] > old['distance_to_outer_envelope_m']:
                return False
        return True

    def report(self):
        before, after = [self.before[i] for i in range(len(self.keys))], [self.after[i] for i in range(len(self.keys))]
        def total(values):
            available = [row['objective_m2'] for row in values if row['objective_m2'] is not None]
            return sum(available) if available else None
        def peak(values):
            distances = [point['distance_to_outer_envelope_m'] for row in values for point in row['channels']
                         if point['distance_to_outer_envelope_m'] is not None]
            return max(distances) if distances else None
        initial, final = total(before), total(after)
        summary = {'source_frame_slots': len(self.slots), 'channel_slots': sum(len(row['channels']) for row in self.slots),
            'available_reference_terms': sum(row['objective_m2'] is not None for row in before),
            'unavailable_reference_terms': sum(row['objective_m2'] is None for row in before),
            'evaluated_channels_before': sum(row['counts']['evaluated_channels'] for row in before),
            'evaluated_channels_after': sum(row['counts']['evaluated_channels'] for row in after),
            'sum_objective_before_m2': initial, 'sum_objective_after_m2': final,
            'weighted_sum_before': None if initial is None else self.weight * initial,
            'weighted_sum_after': None if final is None else self.weight * final,
            'maximum_channel_outside_before_m': peak(before), 'maximum_channel_outside_after_m': peak(after)}
        bindings = {key: self.inputs[key] for key in ('source_sha256', 'body_sha256', 'binding_sha256', 'envelope_cache_sha256', 'envelope_bindings')}
        bindings['objective_profile'] = self.inputs['profile']
        marker = {'profile': SURFACE_INTERVAL_PROFILE, 'input_bindings': bindings, 'weight': self.weight,
                  'weight_status': 'authored', 'work_limits': dict(self.limits), 'summary': summary}
        report = {'schema_version': 1, 'profile': SURFACE_INTERVAL_PROFILE, 'weight': self.weight, 'weight_status': 'authored',
            'weight_units': 'existing_model_objective_units_per_supplied_world_metre_squared',
            'inputs': self.inputs, 'work_limits': dict(self.limits), 'source_slots': self.slots,
            'references': [dict(key, before=before[i], after=after[i]) for i, key in enumerate(self.keys)],
            'summary': summary, 'evaluations': self.evaluations, 'trial_journal': self.journal, 'journal_bytes': self.journal_bytes,
            'optimizer_installation': 'installed_in_existing_whole_body_fitter', 'contact': 'unknown', 'physical_validation': 'unverified'}
        return marker, report


def propose_whole_body_reference_refinement(human_document: bytes, evidence_document: bytes, *, joint_mask=None,
                                            settings=None, proposal_objective='image_v1', surface_interval_objective=None,
                                            surface_interval_weight=None, surface_interval_settings=None,
                                            camera_document=None, camera_image_scale=None, camera_secondary_view=None):
    """Return a separately proposed body; no source, default runtime or policy is modified."""
    started = time.monotonic(); limits = _settings(settings)
    _require(isinstance(proposal_objective, str) and proposal_objective in
             ('image_v1', 'floor_first_v1', 'surface_interval_v1', 'camera_surface_interval_v1',
              'camera_multiview_surface_interval_v1'), 'unknown proposal objective')
    floor_first = proposal_objective == 'floor_first_v1'
    multiview_mode = proposal_objective == 'camera_multiview_surface_interval_v1'
    camera_mode = multiview_mode or proposal_objective == 'camera_surface_interval_v1'
    surface_mode = camera_mode or proposal_objective == 'surface_interval_v1'
    _require(multiview_mode or camera_secondary_view is None, 'secondary view requires explicit camera_multiview_surface_interval_v1 objective')
    _require(camera_mode or camera_document is None and camera_image_scale is None,
             'camera inputs require explicit camera_surface_interval_v1 objective')
    _require(surface_mode or all(value is None for value in (surface_interval_objective, surface_interval_weight, surface_interval_settings)),
             'surface interval inputs require explicit surface_interval_v1 objective')
    method = SURFACE_INTERVAL_METHOD if surface_mode else FLOOR_FIRST_METHOD if floor_first else CONTRACT['method']
    diagnostic = {'floor_qp_calls': 0, 'image_qp_calls': 0, 'floor_descent_accepts': 0, 'image_tiebreak_accepts': 0,
                  'invalid_directions': 0, 'zero_directions': 0, 'backtrack_trials': 0,
                  'rejected_backtracks': 0, 'qp_status_counts': {name: 0 for name in (
                      'optimal', 'iteration_limit', 'time_limit', 'unsupported_infeasible_initial',
                      'dependent_active_set', 'kkt_verification_failed', 'solution_bound',
                      'numerical_feasibility_loss', 'dependent_blocker', 'other')}} if floor_first else None
    camera = None
    if camera_mode:
        from weft.camera_body_fit import METHOD, prepare_camera_body_fit
        camera = prepare_camera_body_fit(human_document, evidence_document, camera_document, image_scale=camera_image_scale,
                                         surface_objective=surface_interval_objective)
        if multiview_mode:
            from weft.camera_multiview_fit import METHOD, prepare_camera_multiview_fit
            camera = prepare_camera_multiview_fit(camera, camera_secondary_view, deadline=started + limits['maximum_seconds'])
            _require(camera.ready, 'secondary camera view unavailable; no geometry or implicit single-view fallback')
        spec, evidence, method = camera.body, camera.evidence, METHOD
    else:
        spec, evidence = _document(human_document), _document(evidence_document)
    mask = NAMES if joint_mask is None else joint_mask
    _require(isinstance(mask, (list, tuple)) and 1 <= len(mask) <= 19 and all(isinstance(n, str) for n in mask)
             and len(set(mask)) == len(mask) and set(mask) <= set(NAMES), 'joint mask must select unique existing axes')
    blocks = [np.array([NAMES.index(name) for name in block if name in mask]) for block in BLOCKS]
    blocks = [block for block in blocks if len(block)]
    times, original, roots, rows, root_audit = _prepare(spec, evidence, **({'camera_fit': camera} if camera_mode else {}))
    surface = _SurfaceIntervalFit(surface_interval_objective, human_document, evidence, rows,
                                 surface_interval_weight, surface_interval_settings,
                                 **({'source_frame_count': root_audit['complete_source_frame_slots']} if camera_mode else {})) if surface_mode else None
    if camera_mode:
        camera.validate_intervals(surface.inputs)
    image_scale = 1. if camera_mode else root_audit['scale']['model_units_per_pixel']
    schedule, samples, lookup = _schedule(times)
    _require(limits['maximum_geometry_calls'] >= 2 * len(samples), 'work bound cannot cover baseline and final audit')
    if multiview_mode:
        _require(limits['maximum_geometry_calls'] >= 2 * camera.projection_count * len(rows),
                 'projection work bound cannot cover both view baseline and final audits')
    changes = np.zeros_like(original); affected = [[] for _ in times]
    for j, sample in enumerate(samples):
        for index, weight in sample['weights']:
            affected[index].append((j, weight))
    calls, accepted, rejected, qp_calls, projection_calls = 0, 0, 0, 0, 0
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
        result = engine.evaluate(q.copy(), root.copy(), derivatives, **({'surface_intervals': True} if surface_mode else {}))
        for key, shape in (('points', (12, 3)), ('gaps', (len(support_names),)),
                           *([('point_jac', (12, 3, 19)), ('gap_jac', (len(support_names), 19))] if derivatives else [])):
            result[key] = np.asarray(result[key], dtype=float)
            _require(result[key].shape == shape and np.isfinite(result[key]).all()
                     and np.max(abs(result[key]), initial=0.) <= 1e6, 'invalid bounded kinematic output')
        if surface_mode:
            for key, shape in (('pelvis_position', (3,)), *([('pelvis_jac', (3, 19))] if derivatives else [])):
                result[key] = np.asarray(result[key], dtype=float)
                _require(result[key].shape == shape and np.isfinite(result[key]).all()
                         and np.max(abs(result[key]), initial=0.) <= 1e6, 'invalid bounded absolute-pelvis kinematics')
        check()
        return result

    def at(sample, delta, engine, derivatives=False):
        q = sum((original[i] + delta[i]) * w for i, w in sample['weights'])
        root = sum(roots[i] * w for i, w in sample['weights'])
        return state(engine, q, root, derivatives)

    def camera_projection(index, value, derivatives=False):
        nonlocal projection_calls
        cache = value.setdefault('_camera_projection', {})
        key = (index, derivatives)
        if key not in cache:
            check()
            count = camera.projection_count if multiview_mode else 1
            limit = limits['maximum_geometry_calls'] - (count * len(rows) if phase == 'fit' else 0)
            if projection_calls + count > limit:
                raise TimeoutError('camera projection bound exhausted; final source audit reserved')
            projection_calls += count
            cache[key] = camera.project(index, value, derivatives=derivatives)
            check()
        return cache[key]

    def image(index, value):
        return camera_projection(index, value)['image'] if camera_mode else _project(value['points'])

    def loss(index, value, delta, neighbors):
        if multiview_mode:
            return _loss(dict(rows[index], weights=rows[index]['objective_weights']), image(index, value), delta, neighbors, projected=True)
        return (_loss(rows[index], image(index, value), delta, neighbors, projected=True) if camera_mode
                else _loss(rows[index], value['points'], delta, neighbors))

    def image_errors(index, value):
        if camera_mode:
            residual = camera_projection(index, value)['pixels'] - rows[index]['target_pixels']
            return np.hypot(residual[:, 0], residual[:, 1])
        return np.linalg.norm(_project(value['points']) - rows[index]['target'], axis=1)

    try:
        _require(1 <= len(support_names) <= CONTRACT['maximum_floor_supports'] and len(set(support_names)) == len(support_names)
                 and len(primitive_names) == len(support_names), 'complete unique geometry support identities required')
        for sample in samples:
            sample['before'] = at(sample, changes, geometry)
            sample['current'] = sample['before']
        for i, row in enumerate(rows):
            before = samples[lookup[float(times[i])]]['before']
            row['before_errors'] = image_errors(i, before)
            if camera_mode:
                row['before_projection'] = camera_projection(i, before)
                _require(np.all(row['before_projection']['valid'][row['weights'] > 0]),
                         'known original camera endpoints are outside the fixed projection/depth domain')
            if surface_mode:
                surface.save_audit(i, surface.evaluate(i, samples[lookup[float(times[i])]]['before']))
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
                    if surface_mode:
                        surface.reserve_journal()
                    current = samples[lookup[float(times[i])]]['current']
                    current_errors = image_errors(i, current)
                    image_limit = np.minimum(row['before_errors'], current_errors)
                    cost = loss(i, current, changes[i], neighbors)
                    if surface_mode:
                        cost += surface.weight * surface.cost(surface.current[i])
                    full = state(geometry, original[i] + changes[i], roots[i], True)
                    projection = camera_projection(i, full, True) if camera_mode else None
                    point_count = len(row['weights']) if multiview_mode else 12
                    residual = ((projection['image'] if camera_mode else _project(full['points'])) - row['target']).ravel()
                    jac = (np.take(projection['image_jac'], columns, axis=2).reshape(2 * point_count, n) if camera_mode else
                           np.take(full['point_jac'][:, [0, 2], :] * np.array([1., -1.])[None, :, None], columns, axis=2).reshape(24, n))
                    robust = (row['objective_weights'] if multiview_mode else row['weights']) * np.minimum(
                        1., .05 / np.maximum(np.linalg.norm(residual.reshape(point_count, 2), axis=1), 1e-15))
                    diagonal = np.repeat(robust, 2)
                    prior, temporal = .001, .0005
                    hessian = jac.T @ (diagonal[:, None] * jac) + np.eye(n) * (prior + temporal * len(adjacent) + 1e-6)
                    gradient = jac.T @ (diagonal * residual) + prior * changes[i, columns]
                    for neighbor in neighbors:
                        gradient += temporal * (changes[i, columns] - neighbor[columns])
                    if surface_mode:
                        term = surface.evaluate(i, full, derivatives=True)
                        if term['gradient_m2_per_rad'] is not None:
                            hessian += surface.weight * np.asarray(term['gauss_newton_m2_per_rad2'])[np.ix_(columns, columns)]
                            gradient += surface.weight * np.asarray(term['gradient_m2_per_rad'])[columns]
                    lo, hi = lower[i].copy(), upper[i].copy()
                    for j in adjacent:
                        radius = 1.5 * abs(times[i] - times[j])
                        lo, hi = np.maximum(lo, changes[j] - radius), np.minimum(hi, changes[j] + radius)
                    lo = np.maximum(lo[columns] - changes[i, columns], -.10)
                    hi = np.minimum(hi[columns] - changes[i, columns], .10)
                    matrix, bounds = [*np.eye(n), *-np.eye(n)], [*lo, *-hi]
                    if camera_mode:
                        depth = projection['camera_depth'] - row['before_projection']['camera_depth']
                        depth_jac = projection['camera_depth_jac'][:, columns]
                        matrix.extend(depth_jac); bounds.extend(-.08 - depth - tolerance)
                        matrix.extend(-depth_jac); bounds.extend(depth - .08 - tolerance)
                    floor_proposal = False
                    if floor_first:
                        floor_cost = _floor_loss(samples[j]['current'] for j, _ in affected[i])
                        floor_hessian, floor_gradient = np.eye(n) * 1e-6, np.zeros(n)
                    for j, weight in affected[i]:
                        sample = samples[j]; value = at(sample, changes, geometry, True)
                        threshold = np.maximum(np.minimum(0., sample['before']['gaps']), np.minimum(0., sample['current']['gaps']))
                        matrix.extend(value['gap_jac'][:, columns] * weight)
                        bounds.extend(threshold - value['gaps'] - tolerance)
                        depth = value['points'][:, 1] - sample['before']['points'][:, 1]
                        depth_jac = value['point_jac'][:, 1, :][:, columns] * weight
                        matrix.extend(depth_jac); bounds.extend(-.08 - depth - tolerance)
                        matrix.extend(-depth_jac); bounds.extend(depth - .08 - tolerance)
                        if floor_first:
                            violating = value['gaps'] < -.025
                            if np.any(violating):
                                floor_proposal = True
                                floor_jac = value['gap_jac'][violating][:, columns] * weight
                                # Gap zero avoids asymptotically approaching the unchanged strict -.025 gate.
                                floor_hessian += floor_jac.T @ floor_jac
                                floor_gradient += floor_jac.T @ value['gaps'][violating]
                    for point in np.flatnonzero(row['weights']):
                        r = (projection['pixels'][point] - row['target_pixels'][point] if camera_mode else residual.reshape(12, 2)[point])
                        pj = projection['pixel_jac'][point][:, columns] if camera_mode else jac.reshape(12, 2, n)[point]
                        norm = float(np.hypot(r[0], r[1])) if camera_mode else float(np.linalg.norm(r))
                        if norm > 1e-12:
                            matrix.append(-r @ pj / norm); bounds.append(norm - image_limit[point] - tolerance)
                        else:
                            matrix.extend(pj); bounds.extend([-tolerance / 2] * 2)
                            matrix.extend(-pj); bounds.extend([-tolerance / 2] * 2)
                    a, b = np.asarray(matrix), np.asarray(bounds)
                    check(); qp_calls += 1
                    if floor_first:
                        diagnostic['floor_qp_calls' if floor_proposal else 'image_qp_calls'] += 1
                        if floor_proposal:
                            hessian, gradient = floor_hessian, floor_gradient
                    solved = solve_feasible_qp(hessian, gradient, a, b, initial=np.zeros(n),
                        settings={'maximum_seconds': max(.001, min(.5, deadline - time.monotonic()))})
                    if floor_first:
                        outcome = solved['status'] if solved['status'] in diagnostic['qp_status_counts'] else 'other'
                        diagnostic['qp_status_counts'][outcome] += 1
                    if solved['status'] != 'optimal':
                        if surface_mode:
                            surface.record_qp(i, columns, changes[i], solved, 'rejected_solver')
                        rejected += 1; continue
                    direction = np.asarray(solved['step'], dtype=float)
                    if direction.shape != (n,) or not np.isfinite(direction).all() or np.max(b - a @ direction, initial=0.) > 1e-10:
                        if surface_mode:
                            violation = float(np.max(b - a @ direction, initial=0.)) if direction.shape == (n,) and np.isfinite(direction).all() else None
                            surface.record_qp(i, columns, changes[i], solved, 'rejected_invalid_direction', violation)
                        if floor_first:
                            diagnostic['invalid_directions'] += 1
                        rejected += 1; continue
                    if np.max(abs(direction), initial=0.) < 1e-10:
                        if surface_mode:
                            surface.record_qp(i, columns, changes[i], solved, 'zero_direction', float(np.max(b - a @ direction, initial=0.)))
                        if floor_first:
                            diagnostic['zero_directions'] += 1
                        continue
                    if surface_mode:
                        surface.record_qp(i, columns, changes[i], solved, 'optimal_direction', float(np.max(b - a @ direction, initial=0.)))
                    for backtrack in range(CONTRACT['maximum_backtracks']):
                        check()
                        if surface_mode:
                            surface.reserve_journal()
                        trial = changes.copy(); trial[i, columns] += direction * .5 ** backtrack
                        if surface_mode:
                            surface.record({'kind': 'backtrack', 'reference_index': i, 'source_frame': row['frame'],
                                'block': [NAMES[j] for j in columns], 'backtrack_index': backtrack,
                                'trial_correction_rad': trial[i].tolist(), 'combined_cost_before': cost, 'combined_cost_after': None,
                                'interval_before_m2': surface.current[i]['objective_m2'], 'interval_after_m2': None,
                                'gate_results': {key: None for key in ('floor_nonregression', 'endpoint_depth', 'source_image_nonregression', 'surface_interval_nonregression')},
                                'accepted': False, 'outcome': 'evaluation_incomplete', 'floor_min_margin_model_units': None,
                                'maximum_depth_change_model_units': None,
                                **({'maximum_source_image_excess_px': None, 'camera_projection_valid': None,
                                    'maximum_camera_depth_change_supplied_units': None, 'camera_depth_bound_satisfied': None} if camera_mode
                                   else {'maximum_source_image_excess_model_units': None})})
                        states = []; valid = True
                        floor_margin, depth_change = math.inf, 0.
                        for j, _ in affected[i]:
                            sample = samples[j]; value = at(sample, trial, geometry)
                            threshold = np.maximum(np.minimum(0., sample['before']['gaps']), np.minimum(0., sample['current']['gaps']))
                            valid &= bool(np.all(value['gaps'] >= threshold - tolerance)
                                          and np.max(abs(value['points'][:, 1] - sample['before']['points'][:, 1])) <= .08 + tolerance)
                            if surface_mode:
                                floor_margin = min(floor_margin, float(np.min(value['gaps'] - threshold)))
                                depth_change = max(depth_change, float(np.max(abs(value['points'][:, 1] - sample['before']['points'][:, 1]))))
                            states.append((j, value))
                        point_state = next(value for j, value in states if j == lookup[float(times[i])])
                        errors = image_errors(i, point_state)
                        valid &= bool(np.all(errors[row['weights'] > 0] <= image_limit[row['weights'] > 0] + tolerance))
                        camera_valid, camera_depth_change = True, None
                        if camera_mode:
                            trial_projection = camera_projection(i, point_state)
                            camera_depth_change = float(np.max(abs(trial_projection['camera_depth'] - row['before_projection']['camera_depth'])))
                            camera_valid = bool(np.all(trial_projection['valid'][row['weights'] > 0]))
                            valid &= camera_valid and camera_depth_change <= .08 + tolerance
                        trial_surface = None
                        trial_cost = loss(i, point_state, trial[i], neighbors) if surface_mode or valid else None
                        if surface_mode:
                            trial_surface = surface.evaluate(i, point_state)
                            interval_ok = surface.nonregression(surface.current[i], trial_surface)
                            valid &= interval_ok
                            trial_cost += surface.weight * surface.cost(trial_surface)
                        if floor_first:
                            diagnostic['backtrack_trials'] += 1
                        improves = False
                        if valid:
                            image_improves = trial_cost < cost - 1e-14
                            if floor_first:
                                trial_floor_cost = _floor_loss(value for _, value in states)
                                floor_improves = trial_floor_cost < floor_cost - 1e-14
                                improves = floor_improves or (trial_floor_cost <= floor_cost and image_improves)
                            else:
                                improves = image_improves
                        if surface_mode:
                            surface.finish_record(combined_cost_after=trial_cost, interval_after_m2=trial_surface['objective_m2'],
                                gate_results={'floor_nonregression': bool(floor_margin >= -tolerance),
                                    'endpoint_depth': bool(depth_change <= .08 + tolerance),
                                    'source_image_nonregression': bool(np.all(errors[row['weights'] > 0] <= image_limit[row['weights'] > 0] + tolerance)),
                                    'surface_interval_nonregression': interval_ok}, accepted=bool(improves),
                                outcome='accepted' if improves else 'rejected_gate' if not valid else 'rejected_objective',
                                floor_min_margin_model_units=floor_margin, maximum_depth_change_model_units=depth_change,
                                **({'maximum_source_image_excess_px': float(np.max(errors[row['weights'] > 0] - image_limit[row['weights'] > 0], initial=0.)),
                                    'camera_projection_valid': camera_valid,
                                    'maximum_camera_depth_change_supplied_units': camera_depth_change,
                                    'camera_depth_bound_satisfied': camera_depth_change <= .08 + tolerance} if camera_mode else
                                   {'maximum_source_image_excess_model_units': float(np.max(errors[row['weights'] > 0] - image_limit[row['weights'] > 0], initial=0.))}))
                        if improves:
                            changes = trial; accepted += 1
                            if floor_first:
                                diagnostic['floor_descent_accepts' if floor_improves else 'image_tiebreak_accepts'] += 1
                            for j, value in states:
                                samples[j]['current'] = value
                            if surface_mode:
                                surface.current[i] = trial_surface
                            break
                        if floor_first:
                            diagnostic['rejected_backtracks'] += 1
                        rejected += 1
    except TimeoutError as error:
        expired, failure = True, str(error)
    finally:
        geometry.close()

    # A new geometry instance recomputes every final sample, never trusting optimizer caches.
    phase = 'final'; final_deadline = min(final_deadline, time.monotonic() + limits['finalization_seconds'])
    if surface_mode:
        surface.phase = 'final'
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
            row['after_errors'] = image_errors(i, final[float(times[i])])
            if camera_mode:
                row['after_projection'] = camera_projection(i, final[float(times[i])])
            if surface_mode:
                surface.save_audit(i, surface.evaluate(i, final[float(times[i])]), final=True)
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
    if surface_mode:
        gates['surface_interval_nonregression'] = all(surface.nonregression(surface.before[i], surface.after[i]) for i in range(len(rows)))
    if camera_mode:
        gates['camera_projection_domain'] = all(bool(np.all(row['after_projection']['valid'][row['weights'] > 0])) for row in rows)
        gates['camera_endpoint_depth_change'] = all(float(np.max(abs(row['after_projection']['camera_depth'] - row['before_projection']['camera_depth'])))
                                                    <= .08 + tolerance for row in rows)
        gates['source_endpoint_image_nonregression'] &= gates['camera_projection_domain']
    multiview_audit = camera.audit(rows, tolerance, deadline=final_deadline) if multiview_mode else None
    if multiview_mode:
        for name in ('source_endpoint_image_nonregression', 'camera_projection_domain', 'camera_endpoint_depth_change'):
            gates[name] = all(view['gates'][name] for view in multiview_audit['views'])
    source_rows = []
    for i, row in enumerate(rows):
        points = [dict(point, error_before_px=float(row['before_errors'][j] / image_scale) if point['weight'] else None,
                       error_after_px=float(row['after_errors'][j] / image_scale) if point['weight']
                       and (not camera_mode or row['after_projection']['valid'][j]) else None)
                  for j, point in enumerate(row['points'])]
        if camera_mode:
            for j, point in enumerate(points):
                point['projection_before'] = 'valid' if row['before_projection']['valid'][j] else 'unavailable_camera_domain'
                point['projection_after'] = 'valid' if row['after_projection']['valid'][j] else 'unavailable_camera_domain'
                for when in ('before', 'after'):
                    value = row[when + '_projection']
                    point['predicted_' + when + '_px'] = value['pixels'][j].tolist() if value['valid'][j] else None
                    point['depth_' + when + '_supplied_camera_units'] = float(value['camera_depth'][j])
        source_rows.append({key: row[key] for key in ('evidence_id', 'frame', 'time_s', 'root_status')} | {
            'status': 'frozen_first_reference' if i == 0 else 'unchanged_missing_source_targets' if not np.any(row['weights']) else
                      'changed_inferred_proposal' if np.any(changes[i]) else 'unchanged_no_accepted_step',
            'points': points, 'correction_rad': dict(zip(NAMES, changes[i].tolist())),
            **({'target_origin': deepcopy(row['target_origin'])} if 'target_origin' in row else {})})
    failures = [key for key, passed in gates.items() if not passed]
    status = 'partial_budget_exhausted' if expired else 'rejected_construction_proposal' if failures else 'unadmitted_construction_proposal'
    candidate['whole_body_reference_refinement'] = {
        'schema_version': 1, 'method': method, 'status': status, 'promoted': False,
        'input_human_sha256': sha256(human_document).hexdigest(), 'input_evidence_sha256': sha256(evidence_document).hexdigest()}
    if surface_mode:
        surface_marker, surface_report = surface.report()
        candidate['whole_body_reference_refinement']['surface_intervals'] = surface_marker
    before_peak = max(row['before_max_penetration_model_units'] for row in checks)
    after_peak = max(row['after_max_penetration_model_units'] for row in checks)
    image_pairs = [(row['before_errors'][j], row['after_errors'][j])
                   for row in rows for j in np.flatnonzero(row['weights'])]
    image_mean = np.mean(image_pairs, axis=0) / image_scale if image_pairs else (None, None)
    if camera_mode:
        # An overlay preserves the original camera/shape pins; it is deliberately not a mutated human document.
        candidate = camera.overlay(candidate['reference'], candidate['whole_body_reference_refinement'])
        if not gates['camera_projection_domain']:
            image_mean = (image_mean[0], None)
    report = {'schema_version': 1, 'profile': method, 'status': status, 'contract': dict(deepcopy(CONTRACT), method=method),
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
            'provenance': ({'method': 'supplied_camera_and_pose_metadata',
                'camera_raw_sha256': root_audit['input_bindings']['camera_raw_sha256'],
                'source_sha256': spec['source_sha256'], 'width': evidence['width'], 'height': evidence['height'],
                'decoder_verification': 'unavailable_no_decode_performed'} if camera_mode else deepcopy(evidence['image_dimension_provenance'])),
            'verification_scope': 'Caller-supplied provenance identifiers; this function does not authenticate an external manifest or decode video'},
        'references': source_rows, 'geometry_checks': checks, 'gates': gates, 'failures': failures,
        'accepted_updates': accepted, 'rejected_trials_or_directions': rejected, 'qp_calls': qp_calls, 'geometry_calls': calls,
        'changed_references': int(np.count_nonzero(np.max(abs(changes), axis=1))), 'maximum_correction_rate_rad_s': rate,
        'image_summary': {'confident_source_endpoint_pairs': len(image_pairs),
            'mean_error_before_px': float(image_mean[0]) if image_mean[0] is not None else None,
            'mean_error_after_px': float(image_mean[1]) if image_mean[1] is not None else None,
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
    if floor_first:
        report['objective'] = {'profile': proposal_objective,
            'primary': 'Half sum of squared excess penetration beyond .025 over unique sampled times/supports',
            'units': 'model_units_squared', 'proposal_target_gap_model_units': 0.,
            'primary_before': _floor_loss(sample['before'] for sample in samples),
            'primary_after': _floor_loss(final.values()), 'unique_samples': len(samples),
            'acceptance': 'Primary decreases by more than 1e-14, or primary does not increase and image/regularization loss decreases by more than 1e-14; every original nonlinear gate still applies',
            'counters': diagnostic,
            'scope': 'Gap-zero quadratic directions are local proposals, not exact nonlinear or global lexicographic optima'}
    if surface_mode:
        report['surface_intervals'] = surface_report
        report['objective'] = {'profile': proposal_objective,
            'primary': 'Existing image/regularization plus authored weighted outside-envelope loss',
            'weight': surface.weight, 'weight_status': 'authored', 'interval_units': 'supplied_world_metre_squared',
            'acceptance': 'Combined loss decreases by >1e-14; every existing nonlinear gate and per-channel interval nonregression apply',
            'grid_interval_targets': 0}
    if camera_mode:
        report['camera_fit'] = root_audit | {'projection_calls': projection_calls,
            'maximum_projection_calls': limits['maximum_geometry_calls'], 'grid_camera_projection_calls': 0}
        report['candidate_kind'] = candidate['profile'] if multiview_mode else 'camera-body-angle-proposal-v1'
        report['geometry_admission'] = 'unsupported_angle_overlay_not_a_human_spec'
        report['contract']['image_constraint'] = 'Confident source-knot pixels through unchanged supplied K/R/t/k; no grid-camera interpolation'
        report['contract']['endpoint_displacement_scope'] = 'Legacy pelvis-relative local-y plus source-knot supplied-camera depth, each <=0.08; no depth recovery'
        report['surface_intervals']['weight_units'] = 'authored_camera_loss_units_squared_per_supplied_world_metre_squared'
        report['objective']['primary'] = 'Fixed-camera pixel residual in explicit authored loss units plus angular regularization and weighted outside-envelope loss'
        report['image_summary']['invalid_final_known_endpoints'] = sum(int(np.count_nonzero((row['weights'] > 0) & ~row['after_projection']['valid'])) for row in rows)
        if multiview_mode:
            report['multiview_images'] = multiview_audit
            report['contract']['image_constraint'] = 'Every qualified source-knot endpoint through both unchanged cameras; each view must independently satisfy raw-pixel/domain/depth gates'
            report['objective']['primary'] = 'Each supplied camera uses its explicit authored pixel loss scale and view weight, plus the unchanged angular and interval terms'
        result = {'status': status, 'human_candidate': None, 'angle_proposal': candidate, 'report': report}
    else:
        result = {'status': status, 'human_candidate': candidate, 'report': report}
    check()
    _require(len(json.dumps(result, allow_nan=False).encode()) <= CONTRACT['maximum_output_bytes'], 'complete output exceeds bound')
    check()
    return result
