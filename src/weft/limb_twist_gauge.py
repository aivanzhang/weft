"""Source-bounded axial orientation proposals with fixed proximal limb geometry."""

import math
import time
from types import MappingProxyType

import numpy as np

from weft.constrained_stereo import _camera, _project
from weft.joint_pixel_trajectory import acceleration_stencil
from weft.limb_position_fit import _array, limb_geometry
from weft.spatial_pose import rotation_to_zyx, rotation_zyx


CONTRACT = MappingProxyType({
    'method': 'source-bounded-limb-twist-gauge-v1', 'maximum_frames': 900,
    'maximum_iterations': 16, 'maximum_backtracks': 6, 'maximum_update_rad': .20,
    'maximum_axial_rotation_rad': math.pi/2, 'maximum_distal_displacement_m': .02,
    'maximum_image_error_increase_std': 1., 'maximum_temporal_gap_s': .25,
    'minimum_temporal_gap_s': 1/240, 'maximum_seconds': 10.,
    'gauge_regularization': 1e-4, 'linear_system_ridge': 1e-9,
    'chart_cosine_minimum': 1e-6, 'step_tolerance_rad': 1e-9,
    'fixed': 'Proximal joint, upper-segment direction, hinge bend, lengths, parents, origins, cameras and source clocks',
    'image_acceptance': 'Complete original image loss cannot increase; each original residual can increase by at most one assumed standard deviation',
    'scope': 'Uncalibrated axial-orientation hypothesis, not recovered anatomy, measured motion or force feasibility',
})


def axial_gauge(angles, theta):
    """Rotate about the upper segment without moving its endpoint or hinge bend."""
    source = np.asarray(angles, dtype=float)
    rotation = rotation_zyx(source[:3]) @ rotation_zyx([theta, 0., 0.])
    result = source.copy()
    result[:3], singular = rotation_to_zyx(rotation, source[:3])
    if theta == 0.:
        result = source.copy()
    return result, rotation, singular


def gauge_derivative(angles, rotation):
    z, y, _ = angles[:3]
    if abs(math.cos(y)) < CONTRACT['chart_cosine_minimum']:
        raise ValueError('Axial gauge derivative is unavailable at an Euler chart singularity')
    basis = np.array([[0., -math.sin(z), math.cos(z)*math.cos(y)],
                      [0., math.cos(z), math.sin(z)*math.cos(y)], [1., 0., -math.sin(y)]])
    return np.linalg.solve(basis, rotation[:, 2])


def solve_gauge_bands(diagonal, first, second, rhs):
    """A scalar two-neighbor elimination keeps trajectory storage linear."""
    d, a, b, r = [np.array(value, dtype=float, copy=True) for value in (diagonal, first, second, rhs)]
    n = len(d)
    for i in range(n):
        if not math.isfinite(d[i]) or d[i] <= 0:
            raise ValueError('Axial gauge requires a positive definite local system')
        if i+1 < n:
            weight = a[i]/d[i]
            d[i+1] -= weight*a[i]; r[i+1] -= weight*r[i]
            if i+2 < n:
                a[i+1] -= weight*b[i]
        if i+2 < n:
            weight = b[i]/d[i]
            d[i+2] -= weight*b[i]; r[i+2] -= weight*r[i]
    result = np.zeros(n)
    for i in range(n-1, -1, -1):
        value = r[i]
        if i+1 < n: value -= a[i]*result[i+1]
        if i+2 < n: value -= b[i]*result[i+2]
        result[i] = value/d[i]
    return result


def fit_limb_twist_gauge(angles, lengths, origins, parent_rotations, times, observations, bend_sign,
                         *, image_std_fraction, motion_acceleration_std_rad_s2, provenance):
    if (not isinstance(observations, (list, tuple)) or not 2 <= len(observations) <= CONTRACT['maximum_frames']
            or type(bend_sign) is not int or bend_sign not in (-1, 1)
            or not isinstance(provenance, str) or not 1 <= len(provenance.strip()) <= 2000):
        raise ValueError('Axial gauge requires bounded source observations, a hinge sign and explicit provenance')
    for value, lower, upper in ((image_std_fraction, 1e-4, .05), (motion_acceleration_std_rad_s2, 1., 1e4)):
        if type(value) not in (int, float) or not math.isfinite(value) or not lower <= value <= upper:
            raise ValueError('Axial gauge requires bounded, explicit uncalibrated noise assumptions')
    n = len(observations)
    original = _array(angles, (n, 4), math.tau, 'gauge angles')
    size = _array(lengths, (2,), 1., 'gauge lengths')
    origins = _array(origins, (n, 3), 2000., 'gauge origins')
    parents = _array(parent_rotations, (n, 3, 3), 1.00001, 'gauge parent frames')
    stamps = _array(times, (n,), 60., 'gauge source times'); gaps = np.diff(stamps)
    if (np.min(size) < .01 or np.min(original[:, 3]) < 0 or np.max(original[:, 3]) > 2.6
            or stamps[0] < 0 or np.min(gaps) < CONTRACT['minimum_temporal_gap_s']-1e-9
            or not np.allclose(parents.transpose(0, 2, 1) @ parents, np.eye(3), atol=1e-8, rtol=0)
            or not np.allclose(np.linalg.det(parents), 1., atol=1e-8, rtol=0)):
        raise ValueError('Axial gauge requires proper frames, ordered clocks, positive lengths and admitted bends')
    prepared = []
    for rows in observations:
        if not isinstance(rows, (list, tuple)) or len(rows) > 8:
            raise ValueError('Axial gauge accepts at most eight original joint/view slots per pose')
        entries, seen, cameras = [], set(), {}
        for row in rows:
            if not isinstance(row, dict) or set(row) != {'camera', 'view_index', 'joint_index', 'pixel', 'confidence'}:
                raise ValueError('Axial gauge observation schema differs')
            view, joint, confidence = (row[key] for key in ('view_index', 'joint_index', 'confidence'))
            if (type(view) is not int or not 0 <= view < 4 or type(joint) is not int or joint not in (0, 1)
                    or (view, joint) in seen or type(confidence) not in (int, float)
                    or not math.isfinite(confidence) or not .5 <= confidence <= 1):
                raise ValueError('Axial gauge requires distinct qualified image slots')
            camera = _camera(row['camera']); pixel = _array(row['pixel'], (2,), 16384., 'gauge original pixel')
            if np.any(pixel < 0) or np.any(pixel > [camera['width'], camera['height']]):
                raise ValueError('Axial gauge observations must stay within their original images')
            if view in cameras and any(not np.array_equal(camera[key], cameras[view][key]) for key in camera):
                raise ValueError('Axial gauge cannot change a camera within one view')
            seen.add((view, joint)); cameras[view] = camera
            std = image_std_fraction*max(camera['width'], camera['height'])
            entries.append((joint, camera, pixel, math.sqrt(confidence)/std, std))
        prepared.append(entries)
    active = np.array([bool(rows) for rows in prepared])
    connected = active[:-1] & active[1:] & (gaps <= CONTRACT['maximum_temporal_gap_s'])
    stencil = acceleration_stencil(gaps, connected, motion_acceleration_std_rad_s2)
    radius = size[1]*np.sin(original[:, 3])
    bounds = np.minimum(CONTRACT['maximum_axial_rotation_rad'],
        2*np.arcsin(np.minimum(1., CONTRACT['maximum_distal_displacement_m']/(2*np.maximum(radius, 1e-30)))))
    bounds[~active] = 0.
    bounds[np.abs(np.cos(original[:, 1])) < CONTRACT['chart_cosine_minimum']] = 0.
    deadline = time.monotonic()+CONTRACT['maximum_seconds']

    def evaluate(theta, derivatives=False):
        if time.monotonic() >= deadline:
            raise TimeoutError('Axial gauge exceeded its declared trajectory deadline')
        values, points, errors, regressions = [], [], [], []
        image_loss = 0.; diagonal = np.zeros(n); gradient = np.zeros(n); directions = np.zeros((n, 3))
        for i, entries in enumerate(prepared):
            value, rotation, _ = axial_gauge(original[i], theta[i]); values.append(value)
            local, _ = limb_geometry(value, size, bend_sign)
            world = local @ parents[i].T+origins[i]; points.append(world)
            if derivatives and bounds[i] > 0:
                directions[i] = gauge_derivative(value, rotation)
            endpoint_derivative = parents[i] @ rotation[:, 0]*(-bend_sign*radius[i])
            frame_errors = []; frame_stds = []
            for joint, camera, target, scale, std in entries:
                pixel, jac = _project(world[joint:joint+1], camera)
                residual = pixel[0]-target; image_loss += float(residual @ residual)*scale**2
                frame_errors.append(float(np.linalg.norm(residual))); frame_stds.append(std)
                if derivatives and joint == 1 and bounds[i] > 0:
                    design = jac[0] @ endpoint_derivative*scale
                    diagonal[i] += float(design @ design); gradient[i] += float(design @ residual)*scale
            errors.append(frame_errors); regressions.append(frame_stds)
        values = np.asarray(values)
        acceleration = sum(stencil[:, j, None]*values[j:n-2+j, :3] for j in range(3))
        motion_loss = float(np.sum(acceleration**2)); prior = CONTRACT['gauge_regularization']*float(theta @ theta)
        first, second = np.zeros(n-1), np.zeros(max(0, n-2))
        if derivatives:
            for j in range(3):
                design = stencil[:, j, None]*directions[j:n-2+j]
                diagonal[j:n-2+j] += np.sum(design**2, axis=1)
                gradient[j:n-2+j] += np.sum(design*acceleration, axis=1)
                for k in range(j+1, 3):
                    cross = stencil[:, j]*stencil[:, k]*np.sum(directions[j:n-2+j]*directions[k:n-2+k], axis=1)
                    if k-j == 1: first[j:n-2+j] += cross
                    else: second += cross
            diagonal += CONTRACT['gauge_regularization']+CONTRACT['linear_system_ridge']
            gradient += CONTRACT['gauge_regularization']*theta
        result = {'objective': .5*(image_loss+motion_loss+prior), 'image_chi2': image_loss,
            'motion_acceleration_chi2': motion_loss, 'gauge_prior_chi2': prior,
            'angles_rad': values.tolist(), 'points_world_m': np.asarray(points).tolist(), 'errors_px': errors}
        return result, (diagonal, first, second, gradient), regressions

    theta = np.zeros(n); initial, _, stds = evaluate(theta); current = initial
    history = []; status = 'unchanged_no_observations' if not np.any(active) else 'iteration_limit'
    if np.any(active):
        for iteration in range(CONTRACT['maximum_iterations']):
            current, (diagonal, first, second, gradient), _ = evaluate(theta, True)
            objective_before = current['objective']
            step = solve_gauge_bands(diagonal, first, second, -gradient)
            if not np.isfinite(step).all():
                raise ValueError('Axial gauge produced a nonfinite local step')
            step *= min(1., CONTRACT['maximum_update_rad']/max(float(np.max(abs(step))), 1e-30))
            trials = []; accepted = False
            for backtrack in range(CONTRACT['maximum_backtracks']):
                trial = np.clip(theta+step*.5**backtrack, -bounds, bounds)
                record = {'backtrack': backtrack, 'axial_rotation_rad': trial.tolist(), 'accepted': False,
                          'rejection': None, 'objective': None, 'image_chi2': None}
                trials.append(record)
                if np.max(abs(trial-theta)) <= CONTRACT['step_tolerance_rad']:
                    status = 'step_tolerance'; break
                try:
                    value, _, _ = evaluate(trial)
                except ValueError:
                    record['rejection'] = 'projection_or_chart_domain'; continue
                record.update(objective=value['objective'], image_chi2=value['image_chi2'])
                if value['image_chi2'] > initial['image_chi2']+1e-10:
                    record['rejection'] = 'complete_source_image_regression'; continue
                if any(new > old+std*CONTRACT['maximum_image_error_increase_std']+1e-10
                       for newrow, oldrow, stdrow in zip(value['errors_px'], initial['errors_px'], stds, strict=True)
                       for new, old, std in zip(newrow, oldrow, stdrow, strict=True)):
                    record['rejection'] = 'individual_source_image_regression'; continue
                if value['objective'] < current['objective']-1e-12:
                    theta = trial; current = value; accepted = record['accepted'] = True; break
                record['rejection'] = 'no_objective_descent'
            history.append({'iteration': iteration+1, 'objective_before': objective_before, 'trials': trials})
            if not accepted:
                if status != 'step_tolerance': status = 'no_descent'
                break
    final, _, _ = evaluate(theta)
    displacement = np.linalg.norm(np.asarray(final['points_world_m'])-np.asarray(initial['points_world_m']), axis=2)
    if (np.max(displacement[:, 0]) > 1e-10 or np.max(displacement[:, 1]) > CONTRACT['maximum_distal_displacement_m']+1e-10
            or final['image_chi2'] > initial['image_chi2']+1e-10 or final['objective'] > initial['objective']+1e-10):
        raise ValueError('Axial gauge violated its original source/geometry objective')
    return {'method': CONTRACT['method'], 'contract': dict(CONTRACT), 'status': status,
        'input_angles_rad': original.tolist(), 'output_angles_rad': final['angles_rad'],
        'axial_rotation_rad': theta.tolist(), 'axial_bounds_rad': bounds.tolist(),
        'correction_rad': (np.asarray(final['angles_rad'])-original).tolist(),
        'fixed_lengths_m': size.tolist(), 'fixed_origins_world_m': origins.tolist(),
        'fixed_parent_rotations': parents.tolist(), 'source_times_s': stamps.tolist(),
        'active_frames': active.tolist(), 'motion_acceleration_stencil': stencil.tolist(),
        'before': initial, 'after': final, 'iterations': history,
        'maximum_proximal_displacement_m': float(np.max(displacement[:, 0])),
        'maximum_distal_displacement_m': float(np.max(displacement[:, 1])),
        'uncertainty': {'image_std_fraction': image_std_fraction,
            'motion_acceleration_std_rad_s2': motion_acceleration_std_rad_s2, 'provenance': provenance},
        'uncertainty_calibrated': False, 'observations_modified': False, 'physical_validation': 'unverified'}
