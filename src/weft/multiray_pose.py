"""Bounded pixel-error triangulation; every supplied view must support the result."""

from itertools import combinations
import math
from types import MappingProxyType

import numpy as np

from weft.camera_pose import _undistort
from weft.constrained_stereo import _camera, _project


CONTRACT = MappingProxyType({
    'method': 'all-observation-pixel-triangulation-v1',
    'minimum_views': 2, 'maximum_views': 4, 'minimum_ray_angle_degrees': 2.,
    'depth_bounds_m': (.2, 100.), 'maximum_reprojection_fraction': .01,
    'maximum_iterations': 8, 'maximum_backtracks': 4,
    'maximum_step_m': .1, 'maximum_displacement_m': .5, 'step_tolerance_m': 1e-7,
    'objective': 'Unweighted squared pixel error in every available qualified image observation.',
    'missing_data': 'At least two original qualified views; no imputation, view selection or outlier removal.',
    'scope': 'Calibrated image estimates, not measured anatomy, independent accuracy or ground/contact evidence.',
})


def _residual(point, cameras, pixels):
    projected = [_project(point[None, :], c) for c in cameras]
    residual = np.array([p[0][0] for p in projected])-pixels
    jacobian = np.concatenate([p[1][0] for p in projected], axis=0)
    return residual.ravel(), jacobian


def triangulate_pixels(pixels, cameras):
    if not isinstance(cameras, (list, tuple)) or not 2 <= len(cameras) <= 4:
        raise ValueError('Triangulation requires two to four explicitly calibrated cameras')
    cameras = [_camera(c) for c in cameras]; pixels = np.asarray(pixels, dtype=float)
    if pixels.shape != (len(cameras), 2) or not np.isfinite(pixels).all():
        raise ValueError('Supply one finite original pixel pair per camera')
    result = {'status': 'unavailable', 'world_point_m': None, 'point_proposal_m': None,
              'initial_point_m': None, 'view_count': len(cameras), 'reprojection_px': None,
              'maximum_ray_angle_degrees': None, 'optimization_status': 'not_started',
              'iterations': 0, 'trial_evaluations': 0, 'initial_pixel_cost': None,
              'final_pixel_cost': None, 'unit_pixel_sensitivity_m': None,
              'physical_validation': 'unverified', 'observations_modified': False}
    rays, centers = [], []
    for pixel, camera in zip(pixels, cameras, strict=True):
        if np.any(pixel < 0) or np.any(pixel > [camera['width'], camera['height']]):
            raise ValueError('Observed pixels must remain inside their original images')
        normalized = _undistort(pixel[None, :], camera['matrix'], camera['radial'])
        if normalized is None:
            return result | {'status': 'distortion_not_invertible'}
        ray = np.r_[normalized[0], 1.] @ camera['rotation']
        rays.append(ray/np.linalg.norm(ray)); centers.append(-camera['translation'] @ camera['rotation'])
    angles = [math.degrees(math.acos(float(np.clip(abs(a @ b), 0., 1.)))) for a, b in combinations(rays, 2)]
    angle = max(angles); result['maximum_ray_angle_degrees'] = angle
    if angle < CONTRACT['minimum_ray_angle_degrees']:
        return result | {'status': 'insufficient_ray_angle'}
    projectors = [np.eye(3)-np.outer(ray, ray) for ray in rays]
    system = sum(projectors); eigenvalues = np.linalg.eigvalsh(system)
    if eigenvalues[0] <= 1e-8*eigenvalues[-1]:
        return result | {'status': 'ill_conditioned_rays'}
    initial = np.linalg.solve(system, sum(P @ center for P, center in zip(projectors, centers, strict=True)))
    result['initial_point_m'] = initial.tolist()
    if any((initial-center) @ ray <= 0 for ray, center in zip(rays, centers, strict=True)):
        return result | {'status': 'nonforward_intersection'}
    point = initial.copy()
    try:
        residual, jacobian = _residual(point, cameras, pixels)
    except ValueError:
        return result | {'status': 'projection_outside_envelope'}
    cost = float(residual @ residual); result['initial_pixel_cost'] = cost
    status = 'iteration_limit'
    for iteration in range(CONTRACT['maximum_iterations']):
        step, _, rank, _ = np.linalg.lstsq(jacobian, -residual, rcond=None)
        result['iterations'] = iteration+1
        if rank != 3 or not np.isfinite(step).all():
            status = 'singular_projection'; break
        norm = float(np.linalg.norm(step))
        if norm < CONTRACT['step_tolerance_m']:
            status = 'converged'; break
        step *= min(1., CONTRACT['maximum_step_m']/norm)
        accepted = False
        for backtrack in range(CONTRACT['maximum_backtracks']):
            trial = point+step*(.5**backtrack); result['trial_evaluations'] += 1
            if np.linalg.norm(trial-initial) > CONTRACT['maximum_displacement_m']:
                continue
            try:
                new_residual, new_jacobian = _residual(trial, cameras, pixels)
            except ValueError:
                continue
            new_cost = float(new_residual @ new_residual)
            if new_cost <= cost:
                point, residual, jacobian, cost = trial, new_residual, new_jacobian, new_cost
                accepted = True; break
        if not accepted:
            status = 'no_descent'; break
    errors = np.linalg.norm(residual.reshape(-1, 2), axis=1)
    eigenvalues = np.linalg.eigvalsh(jacobian.T @ jacobian)
    # This assumes one-pixel independent noise; it is sensitivity, not calibrated uncertainty.
    sensitivity = float(1/math.sqrt(eigenvalues[0])) if eigenvalues[0] > 0 else None
    result.update(point_proposal_m=point.tolist(), reprojection_px=errors.tolist(), optimization_status=status,
                  final_pixel_cost=cost, unit_pixel_sensitivity_m=sensitivity)
    if status != 'converged':
        return result | {'status': 'optimization_incomplete'}
    if any(error > CONTRACT['maximum_reprojection_fraction']*max(c['width'], c['height'])
           for error, c in zip(errors, cameras, strict=True)):
        return result | {'status': 'inconsistent_image_rays'}
    return result | {'status': 'estimated', 'world_point_m': point.tolist()}
