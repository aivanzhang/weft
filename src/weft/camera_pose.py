"""Camera-referenced monocular pose estimates; supplied calibration is additional input."""

from collections import Counter
import math
import re
from types import MappingProxyType

import numpy as np


CONTRACT = MappingProxyType({
    'method': 'fixed-scale-camera-root-v1', 'maximum_frames': 1800,
    'minimum_confidence': .5, 'minimum_joints': 6, 'maximum_condition': 1000.,
    'depth_bounds_m': (.2, 100.), 'maximum_reprojection_fraction': .02,
    'maximum_inverse_iterations': 20,
    'scale': 'Pose-model metre scale is assumed; no scale or body-shape fitting',
    'scope': 'Estimated world positions using supplied camera poses; not measured anatomy or contacts',
})
FIT_JOINTS = (11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28)


def _number(value, name, low, high):
    if type(value) not in (int, float) or not math.isfinite(value) or not low <= value <= high:
        raise ValueError(f'Invalid camera {name}')
    return float(value)


def _integer(value, name, low, high):
    if type(value) is not int or not low <= value <= high:
        raise ValueError(f'Invalid camera {name}')
    return value


def _array(value, shape, name, bound):
    result = np.asarray(value)
    if (result.shape != shape or result.dtype.kind not in 'iuf'
            or not np.isfinite(result).all() or np.abs(result).max() > bound):
        raise ValueError(f'Invalid camera {name}')
    return result.astype(float)


def validate_camera(camera, evidence):
    fields = {'schema_version', 'source_sha256', 'width', 'height', 'fps', 'frame_count',
              'world_axes', 'units', 'provenance', 'quality', 'K', 'radial_k1', 'frames'}
    if not isinstance(camera, dict) or set(camera) != fields or type(camera['schema_version']) is not int or camera['schema_version'] != 1:
        raise ValueError('Camera input fields differ from schema 1')
    if (camera['world_axes'] != 'right_handed_z_up' or camera['units'] != 'm'
            or camera['quality'] not in ('reconstructed', 'measured', 'authored')
            or not isinstance(camera['provenance'], str) or not 1 <= len(camera['provenance'].strip()) <= 2000):
        raise ValueError('Camera coordinates and calibration provenance must be explicit')
    source = camera['source_sha256']
    if not isinstance(source, str) or not re.fullmatch('[0-9a-f]{64}', source) or source != evidence.get('source_sha256'):
        raise ValueError('Camera and pose source identities differ')
    width = _integer(camera['width'], 'width', 64, 4096)
    height = _integer(camera['height'], 'height', 64, 2160)
    fps = _number(camera['fps'], 'fps', 1, 240)
    count = _integer(camera['frame_count'], 'frame count', 1, CONTRACT['maximum_frames'])
    if count/fps > 60 or any(camera[key] != evidence.get(key) for key in ('width', 'height', 'fps', 'frame_count')):
        raise ValueError('Camera dimensions and timebase must match the complete source')
    matrix = _array(camera['K'], (3, 3), 'intrinsics', 1e6)
    if (not np.array_equal(matrix[2], [0, 0, 1]) or matrix[1, 0] != 0
            or matrix[0, 0] < 1 or matrix[1, 1] < 1):
        raise ValueError('Camera intrinsics require a pinhole upper-triangular matrix')
    radial = _number(camera['radial_k1'], 'radial coefficient', -1, 1)
    frames = camera['frames']
    if not isinstance(frames, list) or len(frames) != count:
        raise ValueError('Camera poses must retain every source frame')
    prepared = []
    for index, frame in enumerate(frames):
        if not isinstance(frame, dict) or set(frame) != {'frame', 'time_s', 'R', 't'}:
            raise ValueError('Camera pose fields differ from schema')
        if type(frame['frame']) is not int or frame['frame'] != index:
            raise ValueError('Camera frame indices must cover the original interval in order')
        stamp = _number(frame['time_s'], 'timestamp', 0, 60)
        if abs(stamp-index/fps) > 1e-7:
            raise ValueError('Camera timestamps must match exact source-frame times')
        rotation = _array(frame['R'], (3, 3), 'rotation', 1.00001)
        translation = _array(frame['t'], (3,), 'translation', 1000)
        if (not np.allclose(rotation @ rotation.T, np.eye(3), atol=1e-5, rtol=0)
                or abs(np.linalg.det(rotation)-1) > 1e-5):
            raise ValueError('Camera rotations must be proper, not reflections or scale transforms')
        prepared.append((rotation, translation))
    return matrix, radial, prepared


def _undistort(pixels, matrix, radial, *, inverse=None):
    homogeneous = np.column_stack((pixels, np.ones(len(pixels))))
    distorted = (homogeneous @ (np.linalg.inv(matrix) if inverse is None else inverse).T)[:, :2]
    radius_d = np.linalg.norm(distorted, axis=1)
    if np.max(radius_d) > 4:
        return None
    # Negative radial coefficients have two branches; only the monotonic inner branch is supported.
    if radial < 0 and np.any(radius_d >= 2/3*math.sqrt(-1/(3*radial))):
        return None
    radius = radius_d.copy()
    for _ in range(CONTRACT['maximum_inverse_iterations']):
        derivative = 1+3*radial*radius**2
        if np.any(derivative <= 1e-8):
            return None
        delta = (radius+radial*radius**3-radius_d)/derivative
        radius -= delta
        if np.max(np.abs(delta)) <= 1e-12:
            break
    if (not np.isfinite(radius).all() or np.any(radius < 0)
            or np.max(np.abs(radius+radial*radius**3-radius_d)) > 1e-9):
        return None
    factor = np.divide(radius, radius_d, out=np.ones_like(radius), where=radius_d > 0)
    return distorted*factor[:, None]


def _point(row, keys):
    if not isinstance(row, dict):
        return None
    values = [row.get(key) for key in keys]
    if any(type(v) not in (int, float) or not math.isfinite(v) or abs(v) > 10 for v in values):
        return None
    return np.array(values, dtype=float)


def _usable(image, world):
    uv, xyz = _point(image, ('x', 'y')), _point(world, ('x', 'y', 'z'))
    confidence = _point(image, ('visibility', 'presence'))
    return (uv is not None and xyz is not None and confidence is not None
            and bool(np.all((uv >= 0) & (uv <= 1)))
            and bool(np.all((confidence >= CONTRACT['minimum_confidence']) & (confidence <= 1))))


def _estimate_frame(pose, matrix, radial, rotation, translation, *, width, height):
    result = {'status': 'pose_not_sampled', 'root_camera_m': None, 'root_world_m': None,
              'world_relative_landmarks_m': None, 'fitted_joint_indices': [], 'condition': None,
              'reprojection_rmse_px': None, 'maximum_reprojection_px': None}
    if pose is None:
        return result
    count = pose.get('pose_count', 0)
    if type(count) is not int or count != 1:
        result['status'] = 'competing_poses' if type(count) is int and count > 1 else 'pose_unavailable'
        return result
    image, world = pose.get('landmarks'), pose.get('world_landmarks')
    if (pose.get('status') not in ('tracked', 'tracked_partial') or not isinstance(image, list)
            or not isinstance(world, list) or len(image) != 33 or len(world) != 33):
        result['status'] = 'pose_unavailable'; return result
    usable = [i for i in FIT_JOINTS if _usable(image[i], world[i])]
    result['fitted_joint_indices'] = usable
    if not all(i in usable for i in (23, 24)):
        result['status'] = 'pelvis_unavailable'; return result
    if len(usable) < CONTRACT['minimum_joints']:
        result['status'] = 'insufficient_joints'; return result
    points = [_point(row, ('x', 'y', 'z')) for row in world]
    if any(point is None for point in points):
        result['status'] = 'invalid_world_landmark'; return result
    points = np.asarray(points)
    points -= (points[23]+points[24])/2
    pixels = np.array([[image[i]['x']*width, image[i]['y']*height] for i in usable])
    normalized = _undistort(pixels, matrix, radial)
    if normalized is None:
        result['status'] = 'distortion_not_invertible'; return result
    q = points[usable]
    design = np.zeros((2*len(usable), 3)); target = np.empty(2*len(usable))
    design[::2, 0] = 1; design[1::2, 1] = 1
    design[::2, 2] = -normalized[:, 0]; design[1::2, 2] = -normalized[:, 1]
    target[::2] = normalized[:, 0]*q[:, 2]-q[:, 0]
    target[1::2] = normalized[:, 1]*q[:, 2]-q[:, 1]
    root, _, rank, singular = np.linalg.lstsq(design, target, rcond=None)
    if rank < 3 or singular[-1] <= 0:
        result['status'] = 'degenerate_projection'; return result
    result['condition'] = float(singular[0]/singular[-1])
    if result['condition'] > CONTRACT['maximum_condition']:
        result['status'] = 'ill_conditioned_projection'; return result
    camera_points = q+root
    low, high = CONTRACT['depth_bounds_m']
    if not np.isfinite(root).all() or not np.all((camera_points[:, 2] >= low) & (camera_points[:, 2] <= high)):
        result['status'] = 'depth_outside_envelope'; return result
    projected = camera_points[:, :2]/camera_points[:, 2:]
    projected *= 1+radial*(projected**2).sum(axis=1)[:, None]
    projected = (np.column_stack((projected, np.ones(len(projected)))) @ matrix.T)[:, :2]
    errors = np.linalg.norm(projected-pixels, axis=1)
    result['reprojection_rmse_px'] = float(np.sqrt(np.mean(errors**2)))
    result['maximum_reprojection_px'] = float(errors.max())
    if result['reprojection_rmse_px'] > CONTRACT['maximum_reprojection_fraction']*max(width, height):
        result['status'] = 'inconsistent_projection'; return result
    result.update(status='estimated', root_camera_m=root.tolist(),
                  root_world_m=((root-translation) @ rotation).tolist(),
                  world_relative_landmarks_m=(points @ rotation).tolist())
    return result


def build_camera_reference(evidence, camera):
    if not isinstance(evidence, dict) or evidence.get('profile') != 'exercise-pose-v1' or evidence.get('schema_version') != 1:
        raise ValueError('Camera reconstruction requires exercise-pose-v1 evidence')
    matrix, radial, transforms = validate_camera(camera, evidence)
    frames = evidence.get('frames')
    if not isinstance(frames, list) or not 1 <= len(frames) <= 900:
        raise ValueError('Camera reconstruction requires 1–900 preserved pose samples')
    poses, previous = {}, -1
    for pose in frames:
        if not isinstance(pose, dict):
            raise ValueError('Pose sample must be a record')
        index = _integer(pose.get('frame'), 'pose frame', 0, camera['frame_count']-1)
        stamp = _number(pose.get('time_s'), 'pose timestamp', 0, 60)
        decoded = _number(pose.get('decoder_timestamp_s'), 'decoder timestamp', 0, 60)
        if (index <= previous or pose.get('id') != f"{camera['source_sha256']}:frame:{index}"
                or abs(stamp-camera['frames'][index]['time_s']) > 1e-7 or abs(decoded-stamp) > 1e-7):
            raise ValueError('Pose identity and decoder time must map exactly to the supplied camera frame')
        poses[index] = pose; previous = index
    rows = []
    for index, (rotation, translation) in enumerate(transforms):
        row = _estimate_frame(poses.get(index), matrix, radial, rotation, translation,
                             width=camera['width'], height=camera['height'])
        rows.append({'frame': index, 'time_s': camera['frames'][index]['time_s'],
                     'evidence_id': f"{camera['source_sha256']}:frame:{index}", **row})
    return {'schema_version': 1, 'contract': dict(CONTRACT), 'source_sha256': camera['source_sha256'],
            'calibration_quality': camera['quality'], 'calibration_provenance': camera['provenance'],
            'frames': rows, 'coverage': dict(Counter(row['status'] for row in rows)),
            'source_frames': len(rows), 'sampled_pose_frames': len(poses),
            'scope': CONTRACT['scope'], 'physical_validation': 'unverified',
            'assumptions': ['Raw pose scale and orientation remain model estimates',
                            'Camera matrices, synchronization, world axes and units are supplied additional inputs',
                            'Only root translation is fitted to same-frame image landmarks; no reference body, floor or temporal smoothing',
                            'Small reprojection error does not resolve monocular scale/depth ambiguity',
                            'Every missing, ambiguous or rejected pose remains unavailable']}
