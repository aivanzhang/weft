"""Explicit camera-world body candidates; never replace monocular defaults or learned policies."""

from copy import deepcopy
import math
import re

import numpy as np

from weft.camera_pose import CONTRACT, build_camera_reference
from weft.io import canonical_hash


METHOD = 'camera-root-body-v1'
ROOT_LIMIT = 50.
SCALE_ASSUMPTION = 'fixed_pose_model_metres_not_independently_measured'


def validate_camera_binding(spec):
    from weft.multiview_root import METHODS as ROOT_METHODS, SCALE_ASSUMPTION as TWO_VIEW_SCALE, validate_root_binding
    from weft.multiview_body import METHODS as BODY_METHODS, SCALE_ASSUMPTION as BODY_SCALE, validate_body_binding

    binding = spec.get('camera_reference')
    two_view = isinstance(binding, dict) and binding.get('method') in ROOT_METHODS
    body_view = isinstance(binding, dict) and binding.get('method') in BODY_METHODS
    fields = {'schema_version', 'method', 'source_sha256', 'calibration_sha256', 'pose_evidence_sha256',
              'root_reference_sha256', 'angular_reference_sha256', 'body_rotation_sha256', 'world_axes',
              'calibration_quality', 'scale_assumption', 'origin_world_m', 'source_frame_ids'}
    if two_view or body_view:
        fields.add('root_refinement')
    if body_view:
        fields.add('body_refinement')
    method = binding['method'] if body_view or two_view else METHOD
    scale = BODY_SCALE if body_view else TWO_VIEW_SCALE if two_view else SCALE_ASSUMPTION
    if (not isinstance(binding, dict) or set(binding)-{'ground'} != fields or type(binding['schema_version']) is not int
            or binding['schema_version'] != 1 or binding['method'] != method
            or binding['world_axes'] != 'right_handed_z_up' or binding['scale_assumption'] != scale
            or binding['calibration_quality'] not in ('reconstructed', 'measured', 'authored')):
        raise ValueError('Camera-body binding differs from its explicit coordinate contract')
    for name in fields:
        if name.endswith('_sha256') and (not isinstance(binding[name], str) or not re.fullmatch('[0-9a-f]{64}', binding[name])):
            raise ValueError('Camera-body binding needs pinned input and reference hashes')
    if (spec.get('source_sha256') != binding['source_sha256']
            or canonical_hash(spec.get('root_reference')) != binding['root_reference_sha256']
            or canonical_hash(spec.get('reference')) != binding['angular_reference_sha256']
            or canonical_hash(spec.get('body_frame', {}).get('rotation')) != binding['body_rotation_sha256']):
        raise ValueError('Camera-body reference or source changed after construction')
    origin = binding['origin_world_m']
    if (not isinstance(origin, list) or len(origin) != 3
            or any(type(v) not in (int, float) or not math.isfinite(v) or abs(v) > 2000 for v in origin)):
        raise ValueError('Camera-body origin exceeds its finite world envelope')
    ids = binding['source_frame_ids']
    references = spec.get('reference_evidence', [])
    if (not isinstance(ids, list) or not 2 <= len(ids) <= 900
            or any(not isinstance(v, str) for v in ids) or len(set(ids)) != len(ids)
            or len(ids) != len(spec.get('reference', []))
            or any(not isinstance(v, str) or not re.fullmatch(binding['source_sha256']+r':frame:[0-9]{1,8}', v) for v in ids)
            or not isinstance(references, list) or any(not isinstance(row, dict) for row in references)
            or ids != [row.get('evidence_id') for row in references]):
        raise ValueError('Camera-body references must retain exact source-frame identities')
    if 'ground_plane' in spec.get('mobility', {}):
        raise ValueError('Camera-world bodies cannot reuse a camera-relative inclined plane')
    if 'ground' in binding:
        from weft.camera_ground import validate_ground_binding
        validate_ground_binding(spec)
    if two_view:
        validate_root_binding(spec)
    if body_view:
        validate_body_binding(spec)
    return binding


def make_camera_body(evidence, camera, original):
    from weft.body_frame import CAMERA_TO_SIM, fit_body_frame
    from weft.human import validate_human
    from weft.mobile_human import MOBILITY_DEFAULTS, calibrate_floor
    from weft.spatial_pose import build_spatial_reference

    original = validate_human(original)
    if original['profile'] != 'exercise-v3' or 'camera_reference' in original or 'ground_plane' in original['mobility']:
        raise ValueError('Camera construction requires an original horizontal-floor mobile model')
    if original.get('source_sha256') != evidence.get('source_sha256'):
        raise ValueError('Original body and pose source differ')
    estimated = build_camera_reference(evidence, camera)
    by_id = {row['evidence_id']: row for row in estimated['frames']}
    source_refs = original.get('reference_evidence', [])
    if (not isinstance(source_refs, list) or len(source_refs) != len(original['reference']) or len(source_refs) < 2
            or any(not isinstance(row, dict) for row in source_refs)):
        raise ValueError('Original angular references need exact source provenance')
    selected = []
    for angular, row in zip(original['reference'], source_refs):
        pose = by_id.get(row.get('evidence_id'))
        if pose is None or pose['time_s'] != angular['time_s'] or row.get('time_s') != angular['time_s']:
            raise ValueError('Original angular reference identity or time differs from camera input')
        if pose['status'] != 'estimated':
            raise ValueError(f"Camera root unavailable at original source frame {pose['frame']}: {pose['status']}")
        selected.append(pose)
    origin = np.array(selected[0]['root_world_m'])
    roots = [{'time_s': row['time_s'], 'translation': (np.array(row['root_world_m'])-origin).tolist()} for row in selected]
    if any(abs(v) > ROOT_LIMIT for row in roots for v in row['translation']):
        raise ValueError('Camera root displacement exceeds the explicit 50-unit envelope')
    transformed = deepcopy(evidence)
    for frame in transformed['frames']:
        pose = by_id[frame['id']]
        if pose['status'] != 'estimated':
            if frame['status'] in ('tracked', 'tracked_partial'):
                raise ValueError('No retained body pose may be silently excluded from camera construction')
            continue
        # Existing angle extraction expects camera axes; encode the supplied world basis explicitly.
        converted = np.asarray(pose['world_relative_landmarks_m']) @ CAMERA_TO_SIM
        for point, values in zip(frame['world_landmarks'], converted):
            point.update(zip(('x', 'y', 'z'), values.tolist()))
    body_frame = fit_body_frame(transformed)
    body_frame.update(scope='Torso orientation in supplied z-up world coordinates; shape remains monocular',
                      gravity_assumption='Supplied world z-up sets authored gravity; calibration accuracy is not verified here')
    if 'fit' in body_frame:
        body_frame['fit']['torso_tilt_from_supplied_world_up_radians'] = body_frame['fit'].pop('torso_tilt_from_camera_up_radians')
    angular, audit = build_spatial_reference(transformed, body_frame)
    if [row['time_s'] for row in angular] != [row['time_s'] for row in roots]:
        raise ValueError('Camera rotation extraction changed the original reference interval or availability')
    audit['scope'] = 'Pose-model geometry rotated by supplied cameras; not independently measured anatomy'
    audit['assumptions'] = [value for value in audit['assumptions'] if 'Root translation, metric calibration' not in value]
    audit['assumptions'].append('Camera inputs determine the world basis; root depth still depends on assumed pose-model scale')
    spec = deepcopy(original)
    spec.update(body_frame=body_frame, reference=angular, root_reference=roots,
                reference_evidence=deepcopy(audit['frames']), mobility=deepcopy(MOBILITY_DEFAULTS))
    spec['mobility'].update(slide_limits=[-ROOT_LIMIT, ROOT_LIMIT],
                            base_height=min(5., max(.01, spec['lengths']['thigh']+spec['lengths']['shin']+.08)))
    spec['assumptions'] = [value for value in original['assumptions'] if not any(
        text in value.lower() for text in ('camera up', 'root x/z displacement', 'the learned task'))]
    spec['assumptions'].extend([
        'Camera-world candidate uses supplied camera transforms and monocular pose scale, not video-only metric recovery',
        'Root displacement and body orientation use the supplied z-up basis; absolute world origin is removed',
        'The native 50-unit travel envelope supports locomotion without compressing or trimming the source trajectory',
        'Floor offset uses only the first model pose, not supplied ground height; later intersections remain failures',
        'No learned policy is created or transferred for this candidate; manual control and authored reference feedback only',
    ])
    spec['reconstruction_coverage']['spatial_pose'] = audit
    spec['reconstruction_coverage']['camera_pose'] = {'method': CONTRACT['method'], 'coverage': estimated['coverage'],
                                                    'source_frames': estimated['source_frames']}
    spec['camera_reference'] = {'schema_version': 1, 'method': METHOD, 'source_sha256': evidence['source_sha256'],
        'calibration_sha256': canonical_hash(camera), 'pose_evidence_sha256': canonical_hash(evidence),
        'root_reference_sha256': canonical_hash(roots), 'angular_reference_sha256': canonical_hash(angular),
        'body_rotation_sha256': canonical_hash(body_frame['rotation']), 'world_axes': 'right_handed_z_up',
        'calibration_quality': camera['quality'], 'scale_assumption': SCALE_ASSUMPTION,
        'origin_world_m': origin.tolist(), 'source_frame_ids': [row['evidence_id'] for row in selected]}
    return validate_human(calibrate_floor(spec)), estimated
