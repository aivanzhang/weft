"""Read-only camera-aware pose comparisons; reference quality stays explicit."""

import math

import numpy as np


JOINTS = {'left_shoulder': 11, 'right_shoulder': 12, 'left_elbow': 13, 'right_elbow': 14,
          'left_wrist': 15, 'right_wrist': 16, 'left_hip': 23, 'right_hip': 24,
          'left_knee': 25, 'right_knee': 26, 'left_ankle': 27, 'right_ankle': 28}
CONTRACT = {
    'method': 'camera-relative-pose-comparison-v1', 'minimum_confidence': .5,
    'predicted_axes': 'Camera x right, y down, z away, following the existing Weft world-landmark convention',
    'reference_axes': 'World-to-camera R @ point + t; projection uses K and radial k1',
    'centering': 'Each pose subtracts its own midpoint of the two hip joints',
    'alignment': 'No fitted translation, scale, rotation, reflection, timing offset or identity reassignment',
    'scope': 'Descriptive reference agreement; does not certify reference accuracy, anatomy, absolute root motion or dynamics',
}

WORLD_CONTRACT = {
    'method': 'named-world-joint-comparison-v1',
    'units': 'Input metres; reported distances millimetres',
    'absolute': 'Original shared coordinates, without fitted alignment',
    'root_relative': 'Each pose separately subtracts its own hip midpoint; reported separately from absolute error',
    'alignment': 'No fitted scale, rotation, reflection, timing offset or identity reassignment',
    'scope': 'Descriptive published-reference agreement, not certified anatomy, ground support or contact',
}


def compare_world_frame(prediction, reference):
    """Compare twelve named joints in one declared metric frame without fitting alignment."""
    if not isinstance(prediction, dict) or not isinstance(reference, dict) or set(prediction) != set(JOINTS) or set(reference) != set(JOINTS):
        raise ValueError('World comparison requires all twelve named joint slots')

    def point(value):
        if value is None:
            return None
        if isinstance(value, (list, tuple)) and any(isinstance(v, (bool, np.bool_)) for v in value):
            raise ValueError('Joint positions cannot contain booleans')
        array = np.asarray(value)
        if array.shape != (3,) or array.dtype.kind not in 'iuf':
            raise ValueError('Joint positions require three numeric metre coordinates or null')
        if not np.isfinite(array).all():
            return None
        array = array.astype(float)
        if np.max(np.abs(array)) > 1e6:
            raise ValueError('Joint position exceeds the metric comparison envelope')
        return array

    predicted = {k: point(v) for k, v in prediction.items()}
    expected = {k: point(v) for k, v in reference.items()}

    def root(points):
        hips = [points[k] for k in ('left_hip', 'right_hip')]
        return None if any(v is None for v in hips) else (hips[0]+hips[1])/2

    predicted_root, reference_root = root(predicted), root(expected)
    rows = {}
    for name in JOINTS:
        value, target = predicted[name], expected[name]
        status = ('prediction_unavailable' if value is None else
                  'reference_unavailable' if target is None else 'compared')
        row = {'absolute_status': status, 'absolute_distance_mm': None,
               'root_relative_status': status, 'root_relative_distance_mm': None}
        if status == 'compared':
            row['absolute_distance_mm'] = float(np.linalg.norm(value-target)*1000)
            if predicted_root is None:
                row['root_relative_status'] = 'prediction_root_unavailable'
            elif reference_root is None:
                row['root_relative_status'] = 'reference_root_unavailable'
            else:
                row['root_relative_distance_mm'] = float(np.linalg.norm((value-predicted_root)-(target-reference_root))*1000)
        rows[name] = row
    root_error = (None if predicted_root is None or reference_root is None else
                  float(np.linalg.norm(predicted_root-reference_root)*1000))
    return {'joints': rows, 'hip_midpoint_distance_mm': root_error}


def project_world(points, rotation, translation, intrinsics, radial_k1=0.):
    points = np.asarray(points, dtype=float)
    rotation = np.asarray(rotation, dtype=float)
    translation = np.asarray(translation, dtype=float)
    intrinsics = np.asarray(intrinsics, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or not 1 <= len(points) <= 33:
        raise ValueError('Projection requires 1–33 world points')
    if (rotation.shape != (3, 3) or translation.shape != (3,) or intrinsics.shape != (3, 3)
            or not all(np.isfinite(v).all() for v in (rotation, translation, intrinsics))
            or not np.allclose(rotation @ rotation.T, np.eye(3), atol=1e-5, rtol=0)
            or abs(np.linalg.det(rotation)-1) > 1e-5):
        raise ValueError('Camera extrinsics require a finite proper rotation and translation')
    if (not np.allclose(intrinsics[2], [0, 0, 1], atol=1e-12, rtol=0)
            or intrinsics[0, 0] <= 0 or intrinsics[1, 1] <= 0
            or type(radial_k1) not in (int, float) or not math.isfinite(radial_k1) or abs(radial_k1) > 1):
        raise ValueError('Unsupported intrinsics or radial distortion')
    camera = points @ rotation.T + translation
    valid = np.isfinite(camera).all(axis=1) & (camera[:, 2] > 1e-6)
    pixels = np.full((len(points), 2), np.nan)
    # Behind-camera and missing points remain unavailable, never clipped into view.
    with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
        normalized = camera[valid, :2]/camera[valid, 2:]
        radius2 = (normalized*normalized).sum(axis=1)
        distorted = normalized*(1+radial_k1*radius2[:, None])
        pixels[valid] = (np.column_stack((distorted, np.ones(len(distorted)))) @ intrinsics.T)[:, :2]
    valid &= np.isfinite(pixels).all(axis=1)
    pixels[~valid] = np.nan
    return camera, pixels, valid


def _vector(landmark, keys):
    if not isinstance(landmark, dict):
        return None
    values = [landmark.get(key) for key in keys]
    if any(type(value) not in (int, float) or not math.isfinite(value) for value in values):
        return None
    return np.asarray(values, dtype=float)


def _confident(landmark):
    confidence = _vector(landmark, ('visibility', 'presence'))
    return confidence is not None and bool(np.all((confidence >= CONTRACT['minimum_confidence']) & (confidence <= 1)))


def compare_frame(prediction, reference_camera, reference_pixels, *, width, height):
    if any(type(v) is not int or not 1 <= v <= 8192 for v in (width, height)):
        raise ValueError('Comparison requires bounded source dimensions')
    if set(reference_camera) != set(JOINTS) or set(reference_pixels) != set(JOINTS):
        raise ValueError('Reference must retain all twelve named joint slots')
    reference_camera = {k: np.asarray(v, dtype=float) for k, v in reference_camera.items()}
    reference_pixels = {k: np.asarray(v, dtype=float) for k, v in reference_pixels.items()}
    if any(v.shape != (3,) for v in reference_camera.values()) or any(v.shape != (2,) for v in reference_pixels.values()):
        raise ValueError('Reference coordinates have unexpected dimensions')
    unavailable = None
    if prediction is None:
        unavailable = 'pose_not_sampled'
    elif prediction.get('pose_count') != 1:
        unavailable = 'competing_poses' if prediction.get('pose_count', 0) > 1 else 'pose_unavailable'
    elif (prediction.get('status') not in ('tracked', 'tracked_partial')
          or not isinstance(prediction.get('landmarks'), list) or len(prediction['landmarks']) != 33
          or not isinstance(prediction.get('world_landmarks'), list) or len(prediction['world_landmarks']) != 33):
        unavailable = 'pose_unavailable'
    predicted_root = None
    if unavailable is None:
        hips = [_vector(prediction['world_landmarks'][i], ('x', 'y', 'z')) for i in (23, 24)]
        if all(v is not None for v in hips) and all(_confident(prediction['landmarks'][i]) for i in (23, 24)):
            predicted_root = (hips[0]+hips[1])/2
    reference_root = (reference_camera['left_hip']+reference_camera['right_hip'])/2
    root_valid = (np.isfinite(reference_root).all()
                  and all(reference_camera[name][2] > 1e-6 for name in ('left_hip', 'right_hip')))
    result = {}
    for name, index in JOINTS.items():
        ref, uv = reference_camera[name], reference_pixels[name]
        row = {'image_status': unavailable, 'relative_3d_status': unavailable,
               'image_distance_px': None, 'relative_3d_distance_mm': None,
               'predicted_image_px': None, 'reference_image_px': uv.tolist() if np.isfinite(uv).all() else None,
               'predicted_relative_m': None, 'reference_relative_m': None}
        if unavailable is not None:
            result[name] = row; continue
        image = prediction['landmarks'][index]
        point = _vector(image, ('x', 'y'))
        world = _vector(prediction['world_landmarks'][index], ('x', 'y', 'z'))
        if point is None or world is None:
            row['image_status'] = row['relative_3d_status'] = 'invalid_landmark'
        elif not _confident(image):
            row['image_status'] = row['relative_3d_status'] = 'low_confidence'
        else:
            point = point*np.array([width, height])
            row['predicted_image_px'] = point.tolist()
            if not np.isfinite(uv).all() or not np.isfinite(ref).all() or ref[2] <= 1e-6:
                row['image_status'] = row['relative_3d_status'] = 'reference_unavailable'
            else:
                row['image_status'] = 'compared'
                row['image_distance_px'] = float(np.linalg.norm(point-uv))
                if not root_valid:
                    row['relative_3d_status'] = 'reference_root_unavailable'
                elif predicted_root is None:
                    row['relative_3d_status'] = 'predicted_root_unavailable'
                else:
                    centered, target = world-predicted_root, ref-reference_root
                    row['relative_3d_status'] = 'compared'
                    row['predicted_relative_m'], row['reference_relative_m'] = centered.tolist(), target.tolist()
                    row['relative_3d_distance_mm'] = float(np.linalg.norm(centered-target)*1000)
        result[name] = row
    return result
