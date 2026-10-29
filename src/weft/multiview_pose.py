"""Two calibrated image rays constrain 3D points without monocular world landmarks."""

from collections import Counter
import math
from types import MappingProxyType

import numpy as np

from weft.camera_pose import _undistort, validate_camera
from weft.io import canonical_hash


CONTRACT = MappingProxyType({
    'method': 'calibrated-two-ray-pose-v1', 'minimum_confidence': .5,
    'minimum_ray_angle_degrees': 2., 'maximum_reprojection_fraction': .01,
    'depth_bounds_m': (.2, 100.), 'maximum_source_frames': 1800, 'maximum_pose_samples': 900,
    'clock': 'Caller-declared aligned packaged frame indices; exact frame clocks, no interpolation or clock fitting',
    'scope': 'Calibrated two-view image estimates, not independently measured anatomy or physical validation',
})


def _triangulate_rays(pixels, cameras):
    if len(pixels) != 2 or len(cameras) != 2:
        raise ValueError('Exactly two image observations and cameras are required')
    rays, centers = [], []
    unavailable = {'status': 'unavailable', 'world_point_m': None, 'ray_angle_degrees': None,
                   'ray_separation_m': None, 'reprojection_px': None}
    for pixel, camera in zip(pixels, cameras, strict=True):
        matrix, radial, rotation, translation, _, _, inverse, center = camera
        uv = np.asarray(pixel, dtype=float)
        if uv.shape != (2,) or not np.isfinite(uv).all():
            raise ValueError('Triangulation pixels must be finite 2D points')
        normalized = _undistort(uv[None, :], matrix, radial, inverse=inverse)
        if normalized is None:
            return unavailable | {'status': 'distortion_not_invertible'}
        direction = np.r_[normalized[0], 1.] @ rotation
        rays.append(direction/np.linalg.norm(direction))
        centers.append(center)
    cosine = float(np.clip(abs(rays[0] @ rays[1]), 0., 1.))
    angle = math.degrees(math.acos(cosine))
    unavailable['ray_angle_degrees'] = angle
    if angle < CONTRACT['minimum_ray_angle_degrees']:
        return unavailable | {'status': 'insufficient_ray_angle'}
    design = np.stack((rays[0], -rays[1]), axis=1)
    distance, _, rank, _ = np.linalg.lstsq(design, np.subtract(centers[1], centers[0]), rcond=None)
    if rank != 2 or not np.isfinite(distance).all() or np.any(distance <= 0):
        return unavailable | {'status': 'nonforward_intersection'}
    endpoints = np.array([center+length*ray for center, length, ray in zip(centers, distance, rays, strict=True)])
    point = endpoints.mean(axis=0)
    separation = float(np.linalg.norm(endpoints[0]-endpoints[1]))
    errors = []
    for pixel, camera in zip(pixels, cameras, strict=True):
        matrix, radial, rotation, translation, width, height, _, _ = camera
        position = point @ rotation.T+translation
        if not CONTRACT['depth_bounds_m'][0] <= position[2] <= CONTRACT['depth_bounds_m'][1]:
            return unavailable | {'status': 'depth_outside_envelope', 'ray_separation_m': separation}
        xy = position[:2]/position[2]
        xy *= 1+radial*(xy @ xy)
        projected = (matrix @ np.r_[xy, 1.])[:2]
        errors.append(float(np.linalg.norm(projected-pixel)))
    detail = {'ray_angle_degrees': angle, 'ray_separation_m': separation, 'reprojection_px': errors}
    if any(error > CONTRACT['maximum_reprojection_fraction']*max(camera[4:6]) for error, camera in zip(errors, cameras, strict=True)):
        return unavailable | detail | {'status': 'inconsistent_image_rays'}
    return {'status': 'estimated', 'world_point_m': point.tolist(), **detail}


def _view(view):
    if not isinstance(view, dict) or set(view) != {'name', 'evidence', 'camera', 'clock_provenance'}:
        raise ValueError('View must identify its camera, source evidence and packaged clock provenance')
    if not isinstance(view['name'], str) or not 1 <= len(view['name']) <= 64:
        raise ValueError('View name must be bounded text')
    if not isinstance(view['clock_provenance'], str) or not 1 <= len(view['clock_provenance'].strip()) <= 2000:
        raise ValueError('Packaged-clock alignment must be explicitly supplied, not inferred from equal fps')
    evidence, camera = view['evidence'], view['camera']
    if (not isinstance(evidence, dict) or evidence.get('profile') != 'exercise-pose-v1'
            or type(evidence.get('schema_version')) is not int or evidence.get('schema_version') != 1):
        raise ValueError('Two-view construction requires retained exercise-pose-v1 image evidence')
    matrix, radial, transforms = validate_camera(camera, evidence)
    frames = evidence.get('frames')
    if not isinstance(frames, list) or not 1 <= len(frames) <= CONTRACT['maximum_pose_samples']:
        raise ValueError('Each view must retain 1–900 pose samples')
    poses, previous = {}, -1
    for row in frames:
        if not isinstance(row, dict): raise ValueError('Pose sample must be a record')
        frame = row.get('frame')
        if type(frame) is not int or not previous < frame < camera['frame_count']:
            raise ValueError('Source pose frame indices must be unique, ordered and within the complete video')
        stamp = row.get('time_s'); decoder = row.get('decoder_timestamp_s')
        if (any(type(v) not in (int, float) or not math.isfinite(v) for v in (stamp, decoder))
                or row.get('id') != f"{camera['source_sha256']}:frame:{frame}"
                or abs(stamp-frame/camera['fps']) > 1e-7 or abs(stamp-decoder) > 1e-7):
            raise ValueError('Image evidence identity and decoder time must match the exact camera frame')
        poses[frame] = row; previous = frame
    inverse = np.linalg.inv(matrix)
    prepared = [(matrix, radial, rotation, translation, camera['width'], camera['height'], inverse, -translation @ rotation)
                for rotation, translation in transforms]
    return poses, prepared


def _observation(pose, joint, width, height):
    if pose is None: return None, 'pose_not_sampled'
    if type(pose.get('pose_count')) is not int or pose['pose_count'] != 1:
        return None, 'pose_unavailable_or_competing'
    # Legacy body gates depend on monocular depth; this path uses image confidence only.
    if pose.get('status') not in ('tracked', 'tracked_partial', 'low_confidence', 'invalid_geometry'):
        return None, 'unsupported_pose_status'
    landmarks = pose.get('landmarks')
    if not isinstance(landmarks, list) or len(landmarks) != 33:
        return None, 'invalid_landmarks'
    point = landmarks[joint]
    if not isinstance(point, dict): return None, 'invalid_landmark'
    values = [point.get(key) for key in ('x', 'y', 'visibility', 'presence')]
    if any(type(value) not in (int, float) or not math.isfinite(value) for value in values):
        return None, 'invalid_landmark'
    x, y, visibility, presence = values
    if not 0 <= x <= 1 or not 0 <= y <= 1: return None, 'outside_image'
    if not CONTRACT['minimum_confidence'] <= min(visibility, presence) or max(visibility, presence) > 1:
        return None, 'low_confidence'
    return np.array([x*width, y*height]), 'available'


def triangulate_pair(first, second):
    views = [first, second]
    prepared = [_view(view) for view in views]
    if first['name'] == second['name'] or first['camera']['source_sha256'] == second['camera']['source_sha256']:
        raise ValueError('Two distinct source videos and camera identities are required')
    if any(first['camera'][key] != second['camera'][key] for key in ('fps', 'frame_count')):
        raise ValueError('This two-view contract requires aligned packaged frame clocks; no resampling is performed')
    frames, counts = [], Counter()
    for frame in range(first['camera']['frame_count']):
        poses = [value[0].get(frame) for value in prepared]
        cameras = [value[1][frame] for value in prepared]
        joints = []
        for joint in range(33):
            observations = [_observation(pose, joint, camera[4], camera[5]) for pose, camera in zip(poses, cameras, strict=True)]
            statuses = [value[1] for value in observations]
            if all(value[0] is not None for value in observations):
                estimate = _triangulate_rays([value[0] for value in observations], cameras)
            else:
                estimate = {'status': 'image_evidence_unavailable', 'world_point_m': None,
                            'ray_angle_degrees': None, 'ray_separation_m': None, 'reprojection_px': None}
            estimate['observation_statuses'] = statuses
            joints.append(estimate); counts[estimate['status']] += 1
        frames.append({'frame': frame, 'time_s': frame/first['camera']['fps'],
                       'source_evidence_ids': [pose['id'] if pose else None for pose in poses], 'joints': joints})
    return {'schema_version': 1, 'contract': dict(CONTRACT), 'source_frames': len(frames), 'joint_slots': len(frames)*33,
            'views': [{'name': view['name'], 'source_sha256': view['camera']['source_sha256'],
                       'camera_sha256': canonical_hash(view['camera']), 'evidence_sha256': canonical_hash(view['evidence']),
                       'calibration_quality': view['camera']['quality'], 'clock_provenance': view['clock_provenance']} for view in views],
            'frames': frames, 'joint_status_counts': dict(counts), 'physical_validation': 'unverified',
            'monocular_world_landmarks_used': False, 'reference_body_or_contacts_used': False}
