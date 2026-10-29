"""Two-view hip positions replace root depth without changing monocular body shape."""

from collections import Counter
from copy import deepcopy
import math
import re

import numpy as np

from weft.io import canonical_hash
from weft.multiview_pose import triangulate_pair


METHOD = 'two-view-root-body-v1'
SCALE_ASSUMPTION = 'supplied_camera_scale_for_root_monocular_scale_for_shape'


def shape_hash(spec):
    fields = ('schema_version', 'profile', 'source_sha256', 'dt', 'controller', 'lengths',
              'body_frame', 'reference', 'reference_evidence', 'motor_limits')
    return canonical_hash({**{key: spec.get(key) for key in fields},
                           'mobility': {key: value for key, value in spec['mobility'].items() if key != 'base_height'}})


def _text(value, limit=2000):
    return isinstance(value, str) and 1 <= len(value.strip()) <= limit


def _hash(value):
    return isinstance(value, str) and re.fullmatch('[0-9a-f]{64}', value) is not None


def _point(value):
    return (isinstance(value, list) and len(value) == 3
            and all(type(v) in (int, float) and math.isfinite(v) and abs(v) <= 2000 for v in value))


def validate_root_binding(spec):
    binding = spec['camera_reference']
    value = binding.get('root_refinement')
    fields = {'schema_version', 'input_body_sha256', 'original_root_reference_sha256', 'unchanged_shape_sha256',
              'triangulation_sha256', 'world_provenance', 'views', 'fps', 'source_frames', 'roots'}
    if (not isinstance(value, dict) or set(value) != fields or type(value['schema_version']) is not int
            or value['schema_version'] != 1 or 'ground' not in binding
            or not _text(value['world_provenance'])
            or any(not _hash(value[key]) for key in fields if key.endswith('_sha256'))
            or value['unchanged_shape_sha256'] != shape_hash(spec)
            or type(value['source_frames']) is not int or not 2 <= value['source_frames'] <= 1800
            or type(value['fps']) not in (int, float) or not math.isfinite(value['fps']) or not 1 <= value['fps'] <= 240):
        raise ValueError('Two-view root requires pinned inputs, unchanged body shape and explicit world/clock provenance')
    views = value['views']
    view_fields = {'name', 'source_sha256', 'camera_sha256', 'evidence_sha256', 'calibration_quality', 'clock_provenance'}
    if (not isinstance(views, list) or len(views) != 2
            or any(not isinstance(view, dict) or set(view) != view_fields for view in views)):
        raise ValueError('Two-view root requires exactly two identified views')
    for view in views:
        if (not _text(view['name'], 64) or not _text(view['clock_provenance'])
                or any(not _hash(view[key]) for key in view_fields if key.endswith('_sha256'))
                or view['calibration_quality'] not in ('reconstructed', 'measured', 'authored')):
            raise ValueError('Two-view root has invalid view provenance')
    if (views[0]['name'] == views[1]['name'] or views[0]['source_sha256'] == views[1]['source_sha256']
            or views[0]['source_sha256'] != binding['source_sha256']
            or views[0]['camera_sha256'] != binding['calibration_sha256']
            or views[0]['evidence_sha256'] != binding['pose_evidence_sha256']
            or views[0]['calibration_quality'] != binding['calibration_quality']):
        raise ValueError('Two-view root inputs differ from the original body binding')
    rows, references = value['roots'], spec['reference_evidence']
    if not isinstance(rows, list) or len(rows) != len(references) or len(rows) != len(spec['root_reference']):
        raise ValueError('Two-view roots must retain every original body pose')
    row_fields = {'frame', 'time_s', 'source_evidence_ids', 'hip_world_m', 'root_world_m'}
    previous = -1
    for row, reference, root in zip(rows, references, spec['root_reference'], strict=True):
        if not isinstance(row, dict) or set(row) != row_fields:
            raise ValueError('Two-view root row differs from schema')
        frame, stamp = row['frame'], row['time_s']
        if (type(frame) is not int or not previous < frame < value['source_frames']
                or type(stamp) not in (int, float) or not math.isfinite(stamp)
                or abs(stamp-frame/value['fps']) > 1e-7 or stamp != reference.get('time_s') or stamp != root.get('time_s')
                or row['source_evidence_ids'] != [f"{view['source_sha256']}:frame:{frame}" for view in views]
                or reference.get('evidence_id') != row['source_evidence_ids'][0]
                or not _point(row['root_world_m']) or not isinstance(row['hip_world_m'], list)
                or len(row['hip_world_m']) != 2 or not all(_point(p) for p in row['hip_world_m'])
                or not _point(root.get('translation'))):
            raise ValueError('Two-view root must preserve exact source identities, times and finite hips')
        midpoint = np.mean(row['hip_world_m'], axis=0)
        expected = np.asarray(row['root_world_m'])-binding['origin_world_m']
        if (not np.allclose(midpoint, row['root_world_m'], atol=1e-10, rtol=0)
                or not np.allclose(expected, root['translation'], atol=1e-10, rtol=0)):
            raise ValueError('Two-view root differs from its retained hip midpoint')
        previous = frame
    if rows[0]['root_world_m'] != binding['origin_world_m']:
        raise ValueError('Two-view root origin must be the first retained hip midpoint')


def make_two_view_root(spec, first, second, *, world_provenance):
    from weft.camera_body import METHOD as MONOCULAR_METHOD, ROOT_LIMIT
    from weft.human import validate_human

    original = validate_human(spec)
    binding = original.get('camera_reference', {})
    if binding.get('method') != MONOCULAR_METHOD or 'ground' not in binding:
        raise ValueError('Two-view root requires an original camera body with supplied ground; no repeated refinement')
    if not _text(world_provenance):
        raise ValueError('Shared camera-world coordinates require explicit provenance')
    predictions = triangulate_pair(first, second)
    primary = predictions['views'][0]
    if (primary['source_sha256'] != binding['source_sha256']
            or primary['camera_sha256'] != binding['calibration_sha256']
            or primary['evidence_sha256'] != binding['pose_evidence_sha256']):
        raise ValueError('First view must match the original body camera and pose evidence exactly')
    rows = []
    for reference in original['reference_evidence']:
        frame = int(reference['evidence_id'].rsplit(':', 1)[1])
        if not 0 <= frame < predictions['source_frames']:
            raise ValueError('Original body reference is outside the two-view source interval')
        pose = predictions['frames'][frame]
        if (reference['evidence_id'] != pose['source_evidence_ids'][0]
                or abs(reference['time_s']-pose['time_s']) > 1e-7):
            raise ValueError('Two-view root must preserve original body identities and clocks')
        hips = [pose['joints'][joint] for joint in (23, 24)]
        if any(joint['status'] != 'estimated' for joint in hips):
            raise ValueError(f'Two-view hips unavailable at retained body frame {frame}; no fallback or dropped poses')
        points = [joint['world_point_m'] for joint in hips]
        rows.append({'frame': frame, 'time_s': reference['time_s'], 'source_evidence_ids': pose['source_evidence_ids'],
                     'hip_world_m': points, 'root_world_m': np.mean(points, axis=0).tolist()})
    origin = rows[0]['root_world_m']
    roots = [{'time_s': row['time_s'], 'translation': (np.asarray(row['root_world_m'])-origin).tolist()} for row in rows]
    if any(abs(v) > ROOT_LIMIT for row in roots for v in row['translation']):
        raise ValueError('Two-view root exceeds the unchanged 50-unit travel envelope')
    height = origin[2]-binding['ground']['height_m']
    if not .01 <= height <= 5:
        raise ValueError('Two-view root must remain 0.01–5 model units above supplied ground')
    result = deepcopy(original)
    result['root_reference'] = roots
    result['mobility']['base_height'] = height
    result['camera_reference'].update(method=METHOD, scale_assumption=SCALE_ASSUMPTION, origin_world_m=origin,
        root_reference_sha256=canonical_hash(roots), root_refinement={
            'schema_version': 1, 'input_body_sha256': canonical_hash(original),
            'original_root_reference_sha256': canonical_hash(original['root_reference']),
            'unchanged_shape_sha256': shape_hash(original), 'triangulation_sha256': canonical_hash(predictions),
            'world_provenance': world_provenance, 'views': predictions['views'], 'fps': first['camera']['fps'],
            'source_frames': predictions['source_frames'], 'roots': rows})
    result['assumptions'] = [v for v in result['assumptions'] if not v.startswith('Camera-world candidate uses supplied')]
    result['assumptions'].extend([
        'Two-view hip midpoint sets root position using supplied camera scale and declared cross-view alignment',
        'Body shape, joint angles, muscle illustrations and motor forces remain the original monocular estimates/authored model',
        'Every original body pose requires both hip estimates; no missing-pose filling, root fallback or floor fitting',
    ])
    retained = {row['frame'] for row in rows}
    slots = []
    for pose in predictions['frames']:
        status = ('retained_body_pose' if pose['frame'] in retained else
                  'primary_pose_not_sampled' if pose['source_evidence_ids'][0] is None else 'primary_body_pose_unavailable')
        slots.append({'frame': pose['frame'], 'time_s': pose['time_s'], 'status': status,
                      'hip_statuses': [pose['joints'][joint]['status'] for joint in (23, 24)]})
    report = {'method': METHOD, 'source_frames': predictions['source_frames'], 'source_slots': slots,
              'coverage': dict(Counter(row['status'] for row in slots)), 'binding': result['camera_reference']['root_refinement'],
              'scope': 'Two-view root only; original monocular shape, supplied ground and authored forces unchanged',
              'physical_validation': 'unverified', 'new_inference': False, 'paid_requests': 0}
    result['reconstruction_coverage']['two_view_root'] = {key: report[key] for key in ('method', 'source_frames', 'coverage', 'scope')}
    return validate_human(result), report, predictions
