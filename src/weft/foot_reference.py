"""Opt-in fixed-triangle foot references, separate from the physical human model."""

from copy import deepcopy
import math
import re
import time

import numpy as np

from weft.body_frame import CAMERA_TO_SIM
from weft.io import canonical_hash


FOOT_CONTRACT = {
    'method': 'prefix-fixed-kinematic-foot-v1', 'training_fraction': .6,
    'maximum_frames': 900, 'maximum_source_frames': 1800, 'maximum_seconds': 60.,
    'minimum_prefix_samples': 3, 'minimum_endpoint_confidence': .5, 'crop_bounds': [0., 1.],
    'minimum_image_axis_pixels': 2., 'foot_length_bounds': [.03, .5],
    'ankle_axis_height_bounds': [.003, .25], 'ankle_along_axis_fraction_bounds': [-.5, 1.5],
    'interpolation': 'none', 'fit': 'positive-confidence weighted prefix medians',
    'coordinate_scope': 'Pose-inferred camera-relative model units, not calibrated metres',
    'physical_model_changed': False, 'collision_geometry': False,
    'mass': None, 'sole_thickness': None, 'contact_load': None, 'anatomical_axial_roll': None,
}
FOOT_IMAGE_CONTRACT = {**FOOT_CONTRACT, 'method': 'image-direction-fixed-kinematic-foot-v1',
                       'maximum_orientation_change_rad': math.pi / 12,
                       'minimum_model_projection_length': .01,
                       'orientation_fit': 'bounded rotation about camera-depth axis; unchanged depth components',
                       'floor_objective': False}
LANDMARKS = {'left': {'ankle': 27, 'heel': 29, 'foot_index': 31},
             'right': {'ankle': 28, 'heel': 30, 'foot_index': 32}}


def _safe(value):
    if isinstance(value, float) and not math.isfinite(value):
        return {'nonfinite_raw_value': repr(value)}
    if isinstance(value, dict):
        return {key: _safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe(item) for item in value]
    return value


def _number(value, low, high):
    return type(value) in (int, float) and math.isfinite(value) and low <= value <= high


def _median(values, weights):
    order = np.argsort(values, kind='stable')
    weights = np.asarray(weights)[order]
    return float(np.asarray(values)[order][np.searchsorted(np.cumsum(weights), weights.sum() / 2)])


def _landmarks(frame, side, width, height):
    reasons, raw, image, world, confidence, crop = [], {}, {}, {}, {}, {}
    for part, index in LANDMARKS[side].items():
        raw[part] = {}
        for field, destination in (('landmarks', image), ('world_landmarks', world)):
            array = frame.get(field)
            point = array[index] if isinstance(array, list) and len(array) == 33 else None
            raw[part][field] = _safe(deepcopy(point))
            if not isinstance(point, dict) or not all(_number(point.get(axis), -10, 10) for axis in ('x', 'y', 'z')):
                reasons.append(part + ':' + field + ':missing_or_invalid')
                continue
            destination[part] = np.array([point[axis] for axis in ('x', 'y', 'z')], dtype=float)
            scores = [point.get(name) for name in ('visibility', 'presence')]
            if not all(_number(score, 0, 1) for score in scores):
                reasons.append(part + ':' + field + ':invalid_confidence')
            else:
                confidence[part] = min(confidence.get(part, 1.), *scores)
        if part in image:
            crop[part] = bool(not ((image[part][:2] >= 0) & (image[part][:2] <= 1)).all())
            if crop[part]:
                reasons.append(part + ':outside_image')
        else:
            crop[part] = None
        if confidence.get(part, 0) < FOOT_CONTRACT['minimum_endpoint_confidence']:
            reasons.append(part + ':low_confidence')
    image_reference = None
    if all(part in image for part in ('ankle', 'heel', 'foot_index')):
        points = {part: image[part][:2] * [width, height] for part in image}
        axis = points['foot_index'] - points['heel']
        size = float(np.linalg.norm(axis))
        image_reference = {'points_px': {part: p.tolist() for part, p in points.items()},
                           'axis_length_px': size, 'axis_y_down': (axis / size).tolist() if size >= 2 else None,
                           'status': 'raw_model_projection_not_independent_annotation'}
        if size < FOOT_CONTRACT['minimum_image_axis_pixels']:
            reasons.append('projected_axis_degenerate')
    geometry = None
    if len(world) == 3:
        points = {part: CAMERA_TO_SIM @ value for part, value in world.items()}
        vector = points['foot_index'] - points['heel']
        length = float(np.linalg.norm(vector))
        low, high = FOOT_CONTRACT['foot_length_bounds']
        if not low <= length <= high:
            reasons.append('inferred_foot_length_out_of_bounds')
        else:
            axis = vector / length
            offset = points['ankle'] - points['heel']
            along = float(offset @ axis)
            radial = offset - along * axis
            height = float(np.linalg.norm(radial))
            if not FOOT_CONTRACT['ankle_axis_height_bounds'][0] <= height <= FOOT_CONTRACT['ankle_axis_height_bounds'][1]:
                reasons.append('ankle_axis_plane_degenerate_or_out_of_bounds')
            if not FOOT_CONTRACT['ankle_along_axis_fraction_bounds'][0] <= along / length <= FOOT_CONTRACT['ankle_along_axis_fraction_bounds'][1]:
                reasons.append('ankle_attachment_out_of_bounds')
            if height >= FOOT_CONTRACT['ankle_axis_height_bounds'][0]:
                geometry = {'length': length, 'along': along, 'height': height,
                            'axis': axis, 'radial_direction': radial / height, 'ankle': points['ankle']}
    return {'raw': raw, 'confidence': confidence, 'outside_image': crop, 'reasons': sorted(set(reasons)),
            'raw_image_reference': image_reference, 'geometry': geometry}


def build_foot_reference(spec, evidence, source_metadata):
    """Build a separate inspectable artifact; no model mutation, dynamics, or gap filling."""
    started = time.monotonic()
    if isinstance(spec, dict) and 'camera_reference' in spec:
        raise ValueError('Camera-world foot references require an explicit transformed-foot adapter')
    if not isinstance(spec, dict) or spec.get('profile') != 'exercise-v3':
        raise ValueError('Foot references require an existing mobile body for provenance only')
    source = spec.get('source_sha256')
    if (not isinstance(evidence, dict) or not isinstance(source_metadata, dict) or not isinstance(source, str)
            or not re.fullmatch('[0-9a-f]{64}', source) or evidence.get('source_sha256') != source or source_metadata.get('source_sha256') != source):
        raise ValueError('Foot reference source identities must match')
    width, height, fps, count = (source_metadata.get(key) for key in ('width', 'height', 'fps', 'decoded_frames'))
    if any(type(v) is not int or not 16 <= v <= 8192 for v in (width, height)):
        raise ValueError('Foot source dimensions must be verified bounded integers')
    if (type(count) is not int or not 2 <= count <= 1800 or not _number(fps, 1, 240)
            or (count - 1) / fps > 60):
        raise ValueError('Foot source timing exceeds the work envelope')
    frames = evidence.get('frames')
    if not isinstance(frames, list) or not 1 <= len(frames) <= FOOT_CONTRACT['maximum_frames']:
        raise ValueError('Foot reference requires 1–900 retained raw poses')
    cutoff = FOOT_CONTRACT['training_fraction'] * (count - 1) / fps
    rows, internals, previous = [], [], -1
    for frame in frames:
        if time.monotonic() - started > FOOT_CONTRACT['maximum_seconds']:
            raise RuntimeError('Foot reference time bound exceeded')
        if not isinstance(frame, dict):
            raise ValueError('Foot evidence rows must be objects')
        index = frame.get('frame')
        if (type(index) is not int or not previous < index < count or not _number(frame.get('time_s'), 0, 60)
                or abs(frame['time_s'] - index / fps) > 1e-6 or frame.get('id') != source + ':frame:' + str(index)):
            raise ValueError('Foot evidence IDs, absolute timestamps and unique frame order must match the source')
        previous = index
        row = {'evidence_id': frame['id'], 'frame': index, 'time_s': frame['time_s'],
               'partition': 'construction_prefix' if frame['time_s'] <= cutoff else 'later_source',
               'pose_status': frame.get('status'), 'pose_count': frame.get('pose_count'),
               'identity_status': frame.get('identity_status'), 'identity_derivation_id': frame.get('identity_derivation_id'),
               'physical_identity_verified': False, 'feet': {}}
        internal = {}
        for side in LANDMARKS:
            values = _landmarks(frame, side, width, height)
            if type(frame.get('pose_count')) is not int or frame['pose_count'] != 1 or frame.get('status') not in ('tracked', 'tracked_partial', 'low_confidence'):
                values['reasons'].append('unsupported_pose_status_or_multiple_hypotheses')
            geometry = values.pop('geometry')
            internal[side] = geometry
            values.update(status='eligible_observation' if not values['reasons'] else 'unavailable',
                          reference=None, endpoint_geometry=None,
                          inferred_measurements={key: geometry[key] for key in ('length', 'along', 'height')} if geometry else None)
            row['feet'][side] = values
        rows.append(row)
        internals.append(internal)
    shapes = {}
    for side in LANDMARKS:
        selected = [(row, internal[side]) for row, internal in zip(rows, internals)
                    if row['partition'] == 'construction_prefix' and row['feet'][side]['status'] == 'eligible_observation']
        shape = {'status': 'unavailable', 'evidence_ids': [row['evidence_id'] for row, _ in selected],
                 'eligible_prefix_samples': len(selected), 'fixed_parameters': None,
                 'scope': 'Fixed inferred landmark triangle, not a shoe, sole, measured anatomy or contact surface'}
        if len(selected) >= FOOT_CONTRACT['minimum_prefix_samples']:
            weights = [min(row['feet'][side]['confidence'].values()) for row, _ in selected]
            fixed = {key: _median([geometry[key] for _, geometry in selected], weights) for key in ('length', 'along', 'height')}
            fraction = fixed['along'] / fixed['length']
            if FOOT_CONTRACT['ankle_along_axis_fraction_bounds'][0] <= fraction <= FOOT_CONTRACT['ankle_along_axis_fraction_bounds'][1]:
                shape.update(status='inferred_fixed_triangle', fixed_parameters=fixed)
            else:
                shape['reason'] = 'median_attachment_out_of_bounds'
        else:
            shape['reason'] = 'insufficient_eligible_prefix_samples'
        shapes[side] = shape
    for row, internal in zip(rows, internals):
        for side, foot in row['feet'].items():
            shape = shapes[side]
            if shape['status'] != 'inferred_fixed_triangle':
                foot['reasons'].append('fixed_shape_unavailable')
            if foot['reasons']:
                foot['status'] = 'unavailable'
                continue
            geometry, fixed = internal[side], shape['fixed_parameters']
            heel = -fixed['along'] * geometry['axis'] - fixed['height'] * geometry['radial_direction']
            index = heel + fixed['length'] * geometry['axis']
            foot.update(status='kinematic_reference',
                reference={'heel_to_index_axis': geometry['axis'].tolist(),
                           'axis_to_ankle_direction': geometry['radial_direction'].tolist(),
                           'anatomical_axial_roll_rad': None, 'sole_normal': None,
                           'scope': 'Pose-inferred landmark-triangle directions only; anatomical roll and sole orientation unmeasured'},
                endpoint_geometry={'ankle_relative': [0., 0., 0.], 'heel_relative': heel.tolist(), 'foot_index_relative': index.tolist(),
                                   'inferred_source_ankle': geometry['ankle'].tolist(),
                                   'coordinate_scope': 'Simulator x=image-right,y=inferred-depth,z=image-up; attached to unchanged ankle',
                                   'collisions': False, 'physical_model_attached': False})
    counts = {side: {'available': sum(row['feet'][side]['status'] == 'kinematic_reference' for row in rows),
                     'unavailable': sum(row['feet'][side]['status'] == 'unavailable' for row in rows)} for side in LANDMARKS}
    sampled = {row['frame'] for row in rows}
    return {'schema_version': 1, 'profile': 'kinematic-foot-reference-v1', 'opt_in': True,
            'status': 'partial' if any(value['unavailable'] for value in counts.values()) else 'complete_retained_samples',
            'source_sha256': source, 'source_metadata': deepcopy(source_metadata), 'body_model_sha256': canonical_hash(spec),
            'contract': deepcopy(FOOT_CONTRACT), 'construction_cutoff_time_s': cutoff,
            'fixed_shapes': shapes, 'frames': rows, 'coverage': {'retained_frames': len(rows), 'source_decoded_frames': count,
                'unsampled_source_frames': sorted(set(range(count)) - sampled), 'feet': counts},
            'physical_identity_verified': False, 'physical_validation': 'unverified', 'model_modified': False,
            'runtime_seconds': time.monotonic() - started}
