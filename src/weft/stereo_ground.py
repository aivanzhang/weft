"""Explicit stereo landmarks define a conditional plane, never measured gravity."""

from copy import deepcopy
import math
import re
from types import MappingProxyType

import numpy as np

from weft.camera_pose import validate_camera
from weft.camera_ground import validate_ground
from weft.io import canonical_hash
from weft.multiview_pose import _triangulate_rays


CONTRACT = MappingProxyType({
    'method': 'manual-stereo-ground-v1', 'maximum_points': 16,
    'maximum_reprojection_px': 3., 'minimum_second_spread_m': .1,
    'minimum_spread_ratio': .01, 'maximum_construction_distance_m': .02,
    'maximum_assessment_distance_m': .04, 'maximum_up_angle_degrees': 45.,
    'maximum_label_uncertainty_px': 32.,
    'scope': 'Conditional plane from manually identified static ground points and supplied cameras; not measured gravity or contact',
})


def _text(value, name, maximum=2000):
    if not isinstance(value, str) or not 1 <= len(value.strip()) <= maximum:
        raise ValueError(f'{name} must be explicit bounded text')


def _hash(value):
    return isinstance(value, str) and re.fullmatch('[0-9a-f]{64}', value) is not None


def validate_stereo_ground(cameras, annotations):
    fields = {'schema_version', 'profile', 'world_provenance', 'views', 'points'}
    if (not isinstance(annotations, dict) or set(annotations) != fields
            or type(annotations['schema_version']) is not int or annotations['schema_version'] != 1
            or annotations['profile'] != 'stereo-ground-annotations-v1'):
        raise ValueError('Ground annotations must follow stereo-ground-annotations-v1')
    _text(annotations['world_provenance'], 'Shared static world and ground assumptions')
    views = annotations['views']
    if not isinstance(views, list) or len(views) != 2:
        raise ValueError('Exactly two source-frame bindings are required')
    if not isinstance(cameras, (list, tuple)) or len(cameras) != 2:
        raise ValueError('Exactly two supplied cameras are required')
    prepared = []
    names, sources = set(), set()
    for camera, view in zip(cameras, views, strict=True):
        # Camera validation needs source metadata, not any human pose observations.
        matrix, radial, transforms = validate_camera(camera, camera)
        keys = {'name', 'source_sha256', 'calibration_sha256', 'image_sha256', 'frame', 'time_s', 'provenance'}
        if not isinstance(view, dict) or set(view) != keys:
            raise ValueError('Each view must bind an image, source frame and exact calibration')
        _text(view['name'], 'View name', 64)
        _text(view['provenance'], 'Image/source association provenance')
        if (view['source_sha256'] != camera['source_sha256']
                or view['calibration_sha256'] != canonical_hash(camera)
                or not _hash(view['image_sha256'])):
            raise ValueError('Ground source/calibration/image identities do not match')
        frame, stamp = view['frame'], view['time_s']
        if (type(frame) is not int or not 0 <= frame < camera['frame_count']
                or type(stamp) not in (int, float) or not math.isfinite(stamp)
                or abs(stamp-camera['frames'][frame]['time_s']) > 1e-9):
            raise ValueError('Ground frame and clock must match the original camera interval')
        names.add(view['name']); sources.add(view['source_sha256'])
        rotation, translation = transforms[frame]
        prepared.append((matrix, radial, rotation, translation, camera['width'], camera['height'],
                         np.linalg.inv(matrix), -translation @ rotation))
    if len(names) != 2 or len(sources) != 2:
        raise ValueError('Ground needs two distinct source videos and view names')
    points = annotations['points']
    if not isinstance(points, list) or not 3 <= len(points) <= CONTRACT['maximum_points']:
        raise ValueError('Ground requires 3–16 explicitly partitioned point pairs')
    identities = set()
    for point in points:
        if not isinstance(point, dict) or set(point) != {'id', 'partition', 'pixels', 'uncertainty_px', 'provenance'}:
            raise ValueError('Ground point fields differ from the annotation contract')
        _text(point['id'], 'Point identity', 64)
        _text(point['provenance'], 'Ground point identity/footprint assumptions')
        if point['id'] in identities or point['partition'] not in ('construction', 'assessment'):
            raise ValueError('Ground points need unique identities and explicit partitions')
        identities.add(point['id'])
        pixels, errors = point['pixels'], point['uncertainty_px']
        if (not isinstance(pixels, list) or len(pixels) != 2
                or not isinstance(errors, list) or len(errors) != 2):
            raise ValueError('Every ground point needs two pixels and two uncertainty assumptions')
        for pixel, error, camera in zip(pixels, errors, cameras, strict=True):
            if (type(error) not in (int, float) or not math.isfinite(error)
                    or not 0 < error <= CONTRACT['maximum_label_uncertainty_px']
                    or not isinstance(pixel, list) or len(pixel) != 2
                    or any(type(v) not in (int, float) or not math.isfinite(v) for v in pixel)
                    or any(not error <= v <= size-1-error for v, size in zip(pixel, (camera['width'], camera['height']), strict=True))):
                raise ValueError('Finite pixel locations and positive uncertainty ranges must fit the original image')
    if sum(p['partition'] == 'construction' for p in points) < 3:
        raise ValueError('At least three construction points are required; assessment points cannot fill gaps')
    return prepared


def _plane(points):
    center = points.mean(axis=0)
    _, singular, axes = np.linalg.svd(points-center, full_matrices=False)
    spread = singular / math.sqrt(len(points))
    if spread[1] < CONTRACT['minimum_second_spread_m'] or spread[1] < CONTRACT['minimum_spread_ratio']*spread[0]:
        return None
    normal = axes[-1]
    if normal[2] < 0:
        normal = -normal
    offset = float(-normal @ center)
    return {'normal': normal.tolist(), 'offset_m': offset, 'center_m': center.tolist(),
            'spread_rms_m': spread.tolist(), 'up_angle_degrees': float(np.degrees(np.arccos(np.clip(normal[2], -1., 1.)))),
            'construction_distances_m': (points @ normal+offset).tolist()}


def _sensitivity(labels, cameras, points, nominal):
    rows = []
    normal, center = np.array(nominal['normal']), np.array(nominal['center_m'])
    # Coordinate perturbations expose local sensitivity, not simultaneous-error bounds.
    for index, label in enumerate(labels):
        for view in range(2):
            for axis in range(2):
                for direction in (-1, 1):
                    pixels = deepcopy(label['pixels'])
                    delta = direction*label['uncertainty_px'][view]
                    pixels[view][axis] += delta
                    estimate = _triangulate_rays(pixels, cameras)
                    row = {'point': label['id'], 'view': view, 'axis': axis, 'delta_px': delta,
                           'triangulation': estimate, 'angle_degrees': None, 'center_distance_m': None}
                    if estimate['status'] == 'estimated':
                        varied = points.copy(); varied[index] = estimate['world_point_m']
                        plane = _plane(varied)
                        if plane is not None:
                            row['angle_degrees'] = float(np.degrees(np.arccos(np.clip(normal @ plane['normal'], -1., 1.))))
                            row['center_distance_m'] = float(np.dot(center, plane['normal'])+plane['offset_m'])
                    rows.append(row)
    valid = [r for r in rows if r['angle_degrees'] is not None]
    return {'method': 'one-coordinate-at-a-time', 'scenarios': rows, 'complete': len(valid) == len(rows),
            'maximum_angle_degrees': max((r['angle_degrees'] for r in valid), default=None),
            'maximum_center_distance_m': max((abs(r['center_distance_m']) for r in valid), default=None),
            'scope': 'Reviewer pixel ranges only; not confidence intervals, joint-error bounds, camera uncertainty or ground accuracy. Rejected scenarios remain visible.'}


def estimate_stereo_ground(cameras, annotations):
    return _estimate_prepared(cameras, annotations, validate_stereo_ground(cameras, annotations))


def _estimate_prepared(cameras, annotations, prepared):
    rows = []
    for label in annotations['points']:
        estimate = _triangulate_rays(label['pixels'], prepared)
        passed = estimate['status'] == 'estimated' and max(estimate['reprojection_px']) <= CONTRACT['maximum_reprojection_px']
        rows.append({'id': label['id'], 'partition': label['partition'], **estimate, 'reprojection_gate_passed': passed})
    train = [row for row in rows if row['partition'] == 'construction']
    assessment = [row for row in rows if row['partition'] == 'assessment']
    complete = all(row['reprojection_gate_passed'] for row in train)
    points = np.array([row['world_point_m'] for row in train]) if complete else None
    plane = _plane(points) if complete else None
    checks = {'complete_construction_geometry': complete, 'nondegenerate_spread': plane is not None,
              'nominal_up': plane is not None and plane['up_angle_degrees'] <= CONTRACT['maximum_up_angle_degrees'],
              'ground_height_envelope': plane is not None and abs(plane['offset_m']) <= 1000,
              'construction_planarity': plane is not None and max(map(abs, plane['construction_distances_m'])) <= CONTRACT['maximum_construction_distance_m']}
    for row in assessment:
        row['signed_distance_m'] = (float(np.dot(row['world_point_m'], plane['normal'])+plane['offset_m'])
                                    if plane is not None and row['reprojection_gate_passed'] else None)
    comparison = 'unavailable_no_assessment_points'
    if assessment:
        comparison = ('consistent_source_labels' if all(row['signed_distance_m'] is not None
            and abs(row['signed_distance_m']) <= CONTRACT['maximum_assessment_distance_m'] for row in assessment)
            else 'failed_source_labels')
        checks['assessment_consistency'] = comparison == 'consistent_source_labels'
    accepted = all(checks.values())
    result = {'schema_version': 1, 'profile': 'stereo-ground-estimate-v1', 'contract': dict(CONTRACT),
              'annotations_sha256': canonical_hash(annotations), 'views': deepcopy(annotations['views']),
              'world_provenance': annotations['world_provenance'], 'points': rows, 'plane': plane, 'checks': checks,
              'status': 'conditional_plane' if accepted else 'ground_unavailable', 'assessment': comparison,
              'physical_validation': 'unavailable', 'body_pose_contact_inputs_used': False,
              'sensitivity': None, 'world_transform': None, 'cameras': [], 'grounds': []}
    if not accepted:
        return result
    labels = [p for p in annotations['points'] if p['partition'] == 'construction']
    result['sensitivity'] = _sensitivity(labels, prepared, points, plane)
    normal = np.array(plane['normal'])
    x = np.array([1., 0., 0.])-normal[0]*normal; x /= np.linalg.norm(x)
    rotation = np.array([x, np.cross(normal, x), normal])
    result['world_transform'] = {'direction': 'original_world_to_conditional_ground_world', 'R': rotation.tolist(),
                                 't_m': [0., 0., 0.], 'scale': 1., 'ground_height_m': -plane['offset_m'],
                                 'scope': 'Coordinate rotation only; treating this surface normal as gravity is a separate authored assumption'}
    for camera in cameras:
        changed = deepcopy(camera)
        changed['quality'] = 'reconstructed'
        changed['provenance'] = (f"Conditional manual ground coordinates; original calibration {canonical_hash(camera)}; "
            f"annotations {result['annotations_sha256']}. Original calibration and uncertainty retained in the package; "
            'no independently measured gravity, ground or contact. Shared metric scale and original clock unchanged.')
        for row in changed['frames']:
            row['R'] = (np.asarray(row['R']) @ rotation.T).tolist()
        validate_camera(changed, changed)
        result['cameras'].append(changed)
        ground = {'schema_version': 1, 'source_sha256': changed['source_sha256'],
            'calibration_sha256': canonical_hash(changed), 'world_axes': 'right_handed_z_up', 'units': 'm',
            'height_m': -plane['offset_m'], 'quality': 'reconstructed',
            'provenance': f"Conditional manual stereo plane; annotations {result['annotations_sha256']}. "
                          'Not measured ground/gravity; construction labels do not independently validate the plane. '
                          'Use the saved estimate for label sensitivity, source checks and unknown calibration error.'}
        validate_ground(ground, {'source_sha256': changed['source_sha256'], 'calibration_sha256': canonical_hash(changed)})
        result['grounds'].append(ground)
    return result
