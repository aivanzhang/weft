"""Fixed-camera angle proposals on an unchanged body; never an admitted human spec."""

from copy import deepcopy
from hashlib import sha256
import json
import math
import time

import numpy as np

from weft.camera_pose import validate_camera
from weft.camera_reference import JOINTS, project_world
from weft.camera_shared_contract import build_camera_coupling
from weft.io import canonical_hash
from weft.spatial_pose import SPATIAL_JOINT_NAMES


PROFILE = 'fixed-camera-body-fit-inputs-v1'
METHOD = 'blocked-camera-body-surface-interval-reference-v1'
OVERLAY_PROFILE = 'camera-body-angle-proposal-v1'
MAX_BYTES, MAX_SECONDS, MAX_SOURCE_FRAMES, MAX_REFERENCES = 8 << 20, 5., 1800, 900
IMAGE_SCALE_BOUNDS = (1e-6, 1.)
DEPTH_BOUNDS = (.2, 100.)


def _require(value, message):
    if not value:
        raise ValueError('Camera body fit: ' + message)


def _check(deadline):
    if time.monotonic() >= deadline:
        raise TimeoutError('Camera body fit preparation deadline exceeded')


def _number(value, low, high):
    return type(value) in (int, float) and math.isfinite(value) and low <= value <= high


def _array(value, shape):
    value = np.asarray(value)
    _require(value.dtype.kind in 'iuf' and value.shape == shape and np.isfinite(value).all()
             and np.max(abs(value), initial=0.) <= 1e6,
             'bounded complete absolute kinematics required')
    return value.astype(float, copy=False)


def _evidence_pin(evidence, expected):
    digest = canonical_hash(evidence)
    if digest == expected:
        return {'canonical_sha256': digest, 'scope': 'complete_supplied_pose_document'}
    provenance = evidence.get('image_dimension_provenance', {})
    original = {key: value for key, value in evidence.items() if key != 'image_dimension_provenance'}
    original_hash = canonical_hash(original)
    _require(provenance.get('method') == 'sealed_decoder_metadata'
             and provenance.get('source_evidence_canonical_sha256') == expected == original_hash,
             'pose evidence differs from the original camera-body pin')
    return {'canonical_sha256': digest, 'original_canonical_sha256': original_hash,
            'scope': 'original_pose_plus_separately_checked_dimension_provenance'}


def _targets(frame, width, height, scale, *, source_qualified):
    raw = frame.get('identity_raw', frame) if source_qualified else {}
    _require(not source_qualified or type(raw) is dict and all(raw.get(key) == frame.get(key) for key in ('id', 'frame', 'time_s')),
             'raw endpoint source identity differs')
    points = raw.get('landmarks')
    usable = (raw.get('status') in ('tracked', 'tracked_partial') and raw.get('pose_count', 1) == 1
              and type(points) is list and len(points) == 33)
    targets, weights, details = np.zeros((12, 2)), np.zeros(12), []
    source = ('identity_raw.landmarks' if raw is not frame else 'landmarks') if source_qualified else 'not_read_unqualified_source'
    for i, (name, index) in enumerate(JOINTS.items()):
        detail = {'joint': name, 'status': 'unavailable_source_endpoint' if source_qualified else 'unqualified_construction_image_source', 'weight': 0.,
                  'source_confidence': None, 'landmark_source': source}
        details.append(detail)
        point = points[index] if usable else None
        if type(point) is not dict:
            continue
        values = [point.get(key) for key in ('x', 'y', 'visibility', 'presence')]
        if any(type(value) not in (int, float) or not math.isfinite(value) for value in values):
            continue
        x, y, visibility, presence = values
        if not 0 <= min(visibility, presence) <= max(visibility, presence) <= 1:
            continue
        confidence = min(visibility, presence)
        detail['source_confidence'] = confidence
        if not 0 <= x <= 1 or not 0 <= y <= 1:
            detail['status'] = 'outside_source_image'
            continue
        detail['status'] = 'low_confidence'
        if confidence >= .5:
            pixel = np.array([x * width, y * height])
            targets[i], weights[i] = pixel * scale, confidence
            detail.update(status='available_image_estimate', weight=confidence,
                          target_px=pixel.tolist(), target_authored_loss_units=targets[i].tolist())
    return targets, weights, details


class CameraBodyFit:
    """Prepare source transforms once; projection consumes named absolute kinematics only."""

    def __init__(self, body, evidence, camera, matrix, radial, transforms, coupling, pins, scale, rows, metadata):
        self.body, self.evidence = body, evidence
        self.scale, self._matrix, self._radial = float(scale), matrix, radial
        self._transforms, self._rows = transforms, rows
        self._offset = -np.asarray(coupling['world_to_local']['translation'])
        self._pins, self._metadata = pins, json.dumps(metadata, sort_keys=True, allow_nan=False).encode()
        _require(len(self._metadata) <= 2 << 20, 'complete camera metadata exceeds two MiB; no source slots dropped')

    def describe(self):
        return json.loads(self._metadata)

    def rows(self):
        return deepcopy(self._rows)

    def validate_intervals(self, inputs):
        _require(inputs['body_sha256'] == self._pins['body_raw_sha256']
                 and inputs['envelope_bindings']['camera_sha256'] == self._pins['camera_raw_sha256'],
                 'surface objective must use the exact original body and camera bytes')
        binding = inputs.get('binding')
        _require(type(binding) is dict and inputs.get('binding_unavailable_reason') is None,
                 'surface objective needs a qualified explicit body/world binding')
        mapping = binding['body_to_world']
        spatial, clock = mapping['spatial'], mapping['clock']
        _require(spatial['scale'] == 1. and spatial['rotation_wxyz'] in ([1., 0., 0., 0.], [-1., 0., 0., 0.])
                 and spatial['translation'] == self._offset.tolist()
                 and clock['rate'] == 1. and clock['source_origin_s'] == clock['target_origin_s'],
                 'surface alignment differs from the fixed camera-world translation, scale or clock')

    def project(self, reference_index, state, *, derivatives=False):
        _require(type(reference_index) is int and 0 <= reference_index < len(self._rows), 'original reference index required')
        points = _array(state['points'], (12, 3)) + _array(state['pelvis_position'], (3,)) + self._offset
        rotation, translation = self._transforms[reference_index]
        camera, pixels, valid = project_world(points, rotation, translation, self._matrix, self._radial)
        valid &= (camera[:, 2] >= DEPTH_BOUNDS[0]) & (camera[:, 2] <= DEPTH_BOUNDS[1])
        with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
            valid &= np.sum(np.square(camera[:, :2] / camera[:, 2:]), axis=1) <= 16.
        jacobian, depth_jacobian = None, None
        if derivatives:
            # The pelvis subtraction in the shared producer must be reversed in both values and derivatives.
            absolute = _array(state['point_jac'], (12, 3, 19)) + _array(state['pelvis_jac'], (3, 19))
            jacobian, depth_jacobian = np.zeros((12, 2, 19)), np.einsum('j,ijk->ik', rotation[2], absolute)
            for index in np.flatnonzero(valid):
                x, y, z = camera[index]; uv = np.array([x / z, y / z])
                radial = (1 + self._radial * float(uv @ uv)) * np.eye(2) + 2 * self._radial * np.outer(uv, uv)
                perspective = np.array([[1 / z, 0., -uv[0] / z], [0., 1 / z, -uv[1] / z]])
                jacobian[index] = self._matrix[:2, :2] @ radial @ perspective @ rotation @ absolute[index]
        # Invalid rows retain their mask; zeros only keep unused/failed terms finite, never remove their weight.
        bounded_pixels = np.zeros((12, 2)); bounded_pixels[valid] = pixels[valid]
        return {'image': bounded_pixels * self.scale, 'image_jac': None if jacobian is None else jacobian * self.scale,
                'pixels': bounded_pixels, 'pixel_jac': jacobian, 'valid': valid,
                'camera_depth': camera[:, 2].copy(), 'camera_depth_jac': depth_jacobian}

    def overlay(self, reference, marker):
        return {'schema_version': 1, 'profile': OVERLAY_PROFILE, 'input_bindings': deepcopy(self._pins),
                'reference_candidate': deepcopy(reference), 'whole_body_reference_refinement': deepcopy(marker),
                'installation': 'unsupported_angle_overlay_not_a_human_spec', 'body_installed': False,
                'raw_camera_changed': False, 'root_or_floor_changed': False, 'promoted': False,
                'contact': 'unknown', 'physical_validation': 'unverified'}


def prepare_camera_body_fit(human_document, evidence_document, camera_document, *, image_scale, surface_objective=None):
    """Bind exact files and original camera clocks without admitting a changed human model."""
    from weft.whole_body_reference_refinement import _document, _target_origin

    deadline = time.monotonic() + MAX_SECONDS
    _require(all(type(raw) is bytes and 0 < len(raw) <= MAX_BYTES for raw in
                 (human_document, evidence_document, camera_document)), 'each exact input must fit eight MiB')
    _require(_number(image_scale, *IMAGE_SCALE_BOUNDS), 'authored image loss scale must be in [1e-6, 1] per pixel')
    body, evidence, camera = (_document(raw) for raw in (human_document, evidence_document, camera_document))
    admission_info, source_cache = None, None
    if surface_objective is not None:
        from weft.source_surface_interval_construction import SurfaceIntervalObjective
        _require(type(surface_objective) is SurfaceIntervalObjective and surface_objective.body_bytes == human_document,
                 'construction source qualification needs the exact original-body interval objective')
        interval_inputs = surface_objective.describe()
        admission_info = interval_inputs.get('source_image_admission')
        _require(type(admission_info) is dict, 'camera mode needs explicitly prepared source image admission')
        source_cache = {'objective_metadata_bytes': len(surface_objective._metadata),
            'objective_join_rows_bytes': admission_info['ordinary_join_row_cache_bytes'],
            'image_admission_rows_bytes': admission_info['row_cache_bytes'],
            'shared_envelope_cache_bytes': admission_info['shared_envelope_cache_bytes']}
        source_cache['total_encoded_cache_bytes'] = sum(source_cache.values())
    _check(deadline)
    coupling = build_camera_coupling(body)
    binding = body['camera_reference']
    _require(canonical_hash(camera) == binding['calibration_sha256'], 'camera differs from the original canonical calibration pin')
    pose_pin = _evidence_pin(evidence, binding['pose_evidence_sha256'])
    matrix, radial, transforms = validate_camera(camera, evidence)
    _require(radial >= 0., 'negative radial branch is outside this fixed-camera interval fit profile')
    _require(camera['fps'] == coupling['clock']['source_fps']
             and camera['frame_count'] == coupling['clock']['total_source_frames'], 'camera and original body source clocks differ')
    decoded_frames = evidence.get('decoded_frames')
    _require(decoded_frames is None or type(decoded_frames) is int and 0 <= decoded_frames <= MAX_SOURCE_FRAMES,
             'bounded original decoded-frame denominator required')
    complete_count = max(camera['frame_count'], decoded_frames or 0)
    frames = evidence.get('frames')
    _require(type(frames) is list and len(frames) <= MAX_REFERENCES, 'at most 900 retained source pose slots')
    by_id, by_frame = {}, {}
    for frame in frames:
        _check(deadline)
        _require(type(frame) is dict, 'source pose slot must be an object')
        index, identifier, stamp = frame.get('frame'), frame.get('id'), frame.get('time_s')
        _require(type(index) is int and 0 <= index < complete_count and index not in by_frame
                 and identifier == f"{body['source_sha256']}:frame:{index}" and identifier not in by_id
                 and _number(stamp, 0., 60.)
                 and (index >= camera['frame_count'] or stamp == camera['frames'][index]['time_s']),
                 'source slot identity or exact camera clock differs')
        decoded = frame.get('decoder_timestamp_s')
        _require(decoded is None or _number(decoded, 0., 60.), 'original decoder timestamp must be finite or unavailable')
        _require(frame.get('status') is None or type(frame['status']) is str and len(frame['status']) <= 80,
                 'bounded source status required')
        by_id[identifier], by_frame[index] = frame, frame
    rows, used_transforms, indices = [], [], {}
    for index, (reference, meta, root) in enumerate(zip(body['reference'], body['reference_evidence'], body['root_reference'], strict=True)):
        _check(deadline)
        frame = by_id.get(meta['evidence_id'])
        _require(frame is not None and reference['time_s'] == meta['time_s'] == root['time_s'] == frame['time_s'],
                 'every original body reference must retain its exact source slot and clock')
        eligible = False
        image_admission = None
        if surface_objective is not None and frame['frame'] < surface_objective.frame_count:
            image_admission = surface_objective.image_admission(frame['frame'])
            expected = {'reference_index': index, 'evidence_id': meta['evidence_id'], 'body_time_s': meta['time_s']}
            eligible = image_admission['body_reference'] == expected and image_admission['status'] == 'construction_image_source_ready'
        target_origin = _target_origin(frame) if eligible else None
        target, weights, points = _targets(frame, camera['width'], camera['height'], image_scale, source_qualified=eligible)
        targets_px = np.array([point.get('target_px', [0., 0.]) for point in points])
        rows.append({'target': target, 'target_pixels': targets_px, 'weights': weights, 'points': points, 'evidence_id': meta['evidence_id'],
                     'frame': frame['frame'], 'time_s': frame['time_s'], 'root_status': 'fixed_supplied_two_view_root',
                     'image_source_eligible': eligible,
                     'image_source_admission': image_admission,
                     **({'target_origin': target_origin} if target_origin is not None else {})})
        used_transforms.append(transforms[frame['frame']]); indices[frame['frame']] = index
    source_slots = []
    for index in range(complete_count):
        _check(deadline)
        frame = by_frame.get(index)
        source_slots.append({'source_frame': index, 'camera_time_s': camera['frames'][index]['time_s'] if index < camera['frame_count'] else None,
            'reference_index': indices.get(index), 'evidence_id': None if frame is None else frame['id'],
            'source_time_s': None if frame is None else frame['time_s'],
            'decoder_timestamp_s': None if frame is None else frame.get('decoder_timestamp_s'),
            'status': 'unsampled' if frame is None else frame.get('status'),
            'camera_status': 'supplied' if index < camera['frame_count'] else 'unavailable_original_decoded_slot',
            'landmarks_used': index in indices and rows[indices[index]]['image_source_eligible'],
            'image_source_admission': rows[indices[index]]['image_source_admission'] if index in indices else None})
    pins = {'body_raw_sha256': sha256(human_document).hexdigest(), 'evidence_raw_sha256': sha256(evidence_document).hexdigest(),
        'camera_raw_sha256': sha256(camera_document).hexdigest(), 'body_canonical_sha256': canonical_hash(body),
        'camera_canonical_sha256': canonical_hash(camera), 'pose_evidence': pose_pin,
        'camera_binding_canonical_sha256': canonical_hash(binding), 'coupling_canonical_sha256': canonical_hash(coupling),
        'root_reference_canonical_sha256': canonical_hash(body['root_reference']),
        'ground_canonical_sha256': canonical_hash(binding['ground']),
        'original_angular_reference_canonical_sha256': canonical_hash(body['reference'])}
    metadata = {'schema_version': 1, 'profile': PROFILE, 'input_bindings': pins, 'coupling': coupling,
        'source_slots': source_slots, 'image_loss_scale_per_pixel': float(image_scale), 'image_loss_scale_status': 'authored',
        'point_order': list(JOINTS), 'joint_order': list(SPATIAL_JOINT_NAMES),
        'complete_source_frame_slots': complete_count,
        'image_source_policy': 'Exact bound body/entity/evidence construction admission, independent of interval availability; other endpoint payloads are not used',
        'prepared_source_cache': source_cache,
        'image_loss_units': 'authored_loss_units_not_metres', 'huber_delta_loss_units': .05,
        'raw_radial_k1': radial, 'depth_bounds_supplied_camera_units': list(DEPTH_BOUNDS),
        'maximum_normalized_radius': 4., 'image_nonregression_tolerance_px': 1e-8,
        'legacy_displacement_gate': 'Original pelvis-relative local-y change <=0.08 model units; not camera depth',
        'clock_scope': 'Exact supplied nominal source knots; original decoder timestamps retained separately, no inferred synchrony or grid camera poses',
        'body_shape_units': 'unchanged_monocular_shape_prior', 'camera_world_units': 'supplied_camera_metre_prior',
        'absolute_derivative': 'relative endpoint Jacobian plus pelvis Jacobian before fixed world/camera transforms',
        'scope': 'Conditional construction on supplied calibration and unchanged mixed-scale body; not anatomy, contact or independent accuracy'}
    result = CameraBodyFit(body, evidence, camera, matrix, radial, tuple(used_transforms), coupling, pins, image_scale, rows, metadata)
    _check(deadline)
    return result
