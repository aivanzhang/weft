"""Fit original joint images with bounded, time-coupled limb-angle corrections."""

import math
import time
from types import MappingProxyType

import numpy as np

from weft.constrained_stereo import _camera, _project
from weft.limb_position_fit import _array, limb_geometry


CONTRACT = MappingProxyType({
    'method': 'time-coupled-joint-pixels-v1', 'maximum_frames': 900, 'maximum_views': 4,
    'maximum_angle_change_rad': .30, 'maximum_update_rad': .10, 'maximum_bend_rad': 2.6,
    'maximum_iterations': 12, 'maximum_backtracks': 4, 'maximum_temporal_gap_s': .25,
    'minimum_temporal_gap_s': 1/240, 'maximum_trajectory_seconds': 10., 'maximum_body_seconds': 45.,
    'linear_system_ridge': 1e-9, 'step_tolerance_rad': 1e-9, 'rotation_roundoff_limit': 1e-5,
    'objective': 'All available original joint pixels, assumed angle prior and correction speed over actual source time',
    'fixed': 'Root, torso frame, proximal origin, shape, lengths, ground, forces and source observations',
    'scope': 'Noise-aware articulated image hypothesis; not observed depth, recovered anatomy or verified contact',
    'floor_selection': False, 'depth_measured': False,
})
ACCELERATION_CONTRACT = MappingProxyType({**CONTRACT,
    'method': 'time-coupled-joint-pixels-v2',
    'objective': 'Original joint pixels, angle/correction-speed priors and actual angular acceleration over source time',
    'motion_prior': 'Uncalibrated scalar Euler-coordinate acceleration; not measured motion or a force guarantee',
    'image_acceptance': 'No accepted trial exceeds the original complete image loss',
})
STATUSES = frozenset(('unchanged_exact_images', 'unchanged_no_observations', 'iteration_limit',
                      'step_tolerance', 'no_descent', 'nonfinite_step_rejected'))


def validate_uncertainty(value):
    fields = {'schema_version', 'image_std_fraction', 'angle_prior_std_rad', 'correction_speed_std_rad_s', 'provenance'}
    version = value.get('schema_version') if isinstance(value, dict) else None
    if version == 2: fields.add('motion_acceleration_std_rad_s2')
    if (not isinstance(value, dict) or set(value) != fields or type(value['schema_version']) is not int
            or version not in (1, 2) or not isinstance(value['provenance'], str)
            or not 1 <= len(value['provenance'].strip()) <= 2000):
        raise ValueError('Joint images require explicit uncertainty assumptions and provenance')
    bounds = [('image_std_fraction', 1e-4, .05), ('angle_prior_std_rad', .01, 1.),
              ('correction_speed_std_rad_s', .01, 100.)]
    if version == 2: bounds.append(('motion_acceleration_std_rad_s2', 1., 1e4))
    for key, low, high in bounds:
        if type(value[key]) not in (int, float) or not math.isfinite(value[key]) or not low <= value[key] <= high:
            raise ValueError('Invalid joint-image uncertainty '+key)
    return dict(value)


def trajectory_contract(settings):
    return ACCELERATION_CONTRACT if settings['schema_version'] == 2 else CONTRACT


def acceleration_stencil(gaps, connected, standard_deviation):
    """Difference adjacent angular velocities at their actual interval midpoints."""
    left, right = gaps[:-1], gaps[1:]
    scale = 2 / ((left + right) * standard_deviation)
    rows = np.column_stack((scale / left, -scale * (1 / left + 1 / right), scale / right))
    rows[~(connected[:-1] & connected[1:])] = 0.
    return rows


def solve_banded_blocks(diagonal, first, second, rhs):
    """Eliminate four-axis blocks with two neighbor bands; storage stays linear."""
    n = len(diagonal); blocks = np.array(diagonal, copy=True); values = np.array(rhs, copy=True)
    near = np.asarray(first)[:, None, None] * np.eye(4)
    far = np.asarray(second)[:, None, None] * np.eye(4)
    transfers = np.zeros((n, 2, 4, 4)); solved = np.zeros_like(values)
    for i in range(n):
        neighbors = min(2, n - i - 1)
        upper = [near[i]] if neighbors else []
        if neighbors == 2: upper.append(far[i])
        combined = np.linalg.solve(blocks[i], np.column_stack([*upper, values[i]]))
        solved[i] = combined[:, -1]
        for j, matrix in enumerate(upper):
            transfer = combined[:, 4*j:4*j+4]; transfers[i, j] = transfer
            blocks[i+j+1] -= matrix.T @ transfer
            values[i+j+1] -= matrix.T @ solved[i]
        if neighbors == 2: near[i+1] -= upper[0].T @ transfers[i, 1]
    for i in range(n-2, -1, -1):
        solved[i] -= transfers[i, 0] @ solved[i+1]
        if i+2 < n: solved[i] -= transfers[i, 1] @ solved[i+2]
    return solved


def solve_blocks(diagonal, edge_weights, rhs):
    """Solve symmetric four-axis blocks with -weight*I off-diagonals in linear space."""
    n = len(diagonal); blocks = np.array(diagonal, copy=True); values = np.array(rhs, copy=True)
    transfers = np.zeros((max(0, n-1), 4, 4)); solved = np.zeros_like(values)
    for i in range(n):
        if i < n-1:
            combined = np.linalg.solve(blocks[i], np.column_stack((-edge_weights[i]*np.eye(4), values[i])))
            transfers[i], solved[i] = combined[:, :4], combined[:, 4]
            blocks[i+1] += edge_weights[i]*transfers[i]
            values[i+1] += edge_weights[i]*solved[i]
        else:
            solved[i] = np.linalg.solve(blocks[i], values[i])
    for i in range(n-2, -1, -1): solved[i] -= transfers[i] @ solved[i+1]
    return solved


def fit_joint_trajectory(angles, lengths, origins, parent_rotations, times, observations, bend_sign, uncertainty):
    settings = validate_uncertainty(uncertainty)
    contract = trajectory_contract(settings); motion = settings['schema_version'] == 2
    if not isinstance(observations, (list, tuple)) or not 2 <= len(observations) <= CONTRACT['maximum_frames']:
        raise ValueError('Joint image trajectories require two to 900 reference poses')
    n = len(observations); original = _array(angles, (n, 4), math.tau, 'joint angles')
    size = _array(lengths, (2,), 1., 'limb lengths'); origin = _array(origins, (n, 3), 2000., 'proximal origins')
    parents = _array(parent_rotations, (n, 3, 3), 1.00001, 'parent rotations')
    stamps = _array(times, (n,), 60., 'source times'); gaps = np.diff(stamps)
    if (np.min(size) < .01 or np.min(original[:, 3]) < 0 or np.max(original[:, 3]) > 2.6
            or type(bend_sign) is not int or bend_sign not in (-1, 1) or stamps[0] < 0
            or np.min(gaps) < CONTRACT['minimum_temporal_gap_s']-1e-9
            or not np.allclose(parents.transpose(0, 2, 1) @ parents, np.eye(3), atol=1e-8, rtol=0)
            or not np.allclose(np.linalg.det(parents), 1., atol=1e-8, rtol=0)):
        raise ValueError('Joint images require valid hinges, positive lengths, proper frames and increasing source time')
    prepared = []
    for rows in observations:
        if not isinstance(rows, (list, tuple)) or len(rows) > 8:
            raise ValueError('At most eight joint/view observations per reference pose')
        entries = []; seen = set(); cameras = {}
        for row in rows:
            if not isinstance(row, dict) or set(row) != {'camera', 'view_index', 'joint_index', 'pixel', 'confidence'}:
                raise ValueError('Joint image observation fields differ')
            view, joint, weight = row['view_index'], row['joint_index'], row['confidence']
            if (type(view) is not int or not 0 <= view < 4 or type(joint) is not int or joint not in (0, 1)
                    or (view, joint) in seen or type(weight) not in (int, float) or not math.isfinite(weight) or not .5 <= weight <= 1):
                raise ValueError('Joint image observations require distinct qualified view/joint slots')
            camera = _camera(row['camera']); pixel = _array(row['pixel'], (2,), 16384., 'original joint pixel')
            if np.any(pixel < 0) or np.any(pixel > [camera['width'], camera['height']]):
                raise ValueError('Joint observations must remain in their original images')
            if view in cameras and any(not np.array_equal(camera[k], cameras[view][k]) for k in camera):
                raise ValueError('Joint observations from one view require the same camera')
            cameras[view] = camera; seen.add((view, joint))
            entries.append((joint, camera, pixel, math.sqrt(weight)/(settings['image_std_fraction']*max(camera['width'], camera['height']))))
        prepared.append(entries)
    active = np.array([bool(rows) for rows in prepared]); connected = active[:-1] & active[1:] & (gaps <= .25)
    edges = np.where(connected, 1/(settings['correction_speed_std_rad_s']*gaps)**2, 0.)
    stencil = acceleration_stencil(gaps, connected, settings['motion_acceleration_std_rad_s2']) if motion else None
    if motion:
        motion_diagonal, motion_first, motion_second = np.zeros(n), np.zeros(n-1), np.zeros(max(0, n-2))
        for j in range(3): motion_diagonal[j:n-2+j] += stencil[:, j]**2
        motion_first[:-1] += stencil[:, 0]*stencil[:, 1]
        motion_first[1:] += stencil[:, 1]*stencil[:, 2]
        motion_second += stencil[:, 0]*stencil[:, 2]
    regularization = 1/settings['angle_prior_std_rad']**2
    lower = np.maximum(-.30, np.array([-math.tau]*3+[0.])-original)
    upper = np.minimum(.30, np.array([math.tau]*3+[2.6])-original)
    lower[~active] = upper[~active] = 0.
    deadline = time.monotonic()+CONTRACT['maximum_trajectory_seconds']

    def evaluate(delta, derivatives=False):
        if time.monotonic() >= deadline: raise TimeoutError('Joint-image trajectory exceeded its declared deadline')
        data = 0.; points = []; errors = []; diagonals = np.zeros((n, 4, 4)); gradient = np.zeros((n, 4))
        for i, entries in enumerate(prepared):
            local, jac = limb_geometry(original[i]+delta[i], size, bend_sign, jacobian=derivatives)
            world = local @ parents[i].T+origin[i]; points.append(world.tolist()); frame_errors = []
            if derivatives: jac = np.einsum('ab,jbc->jac', parents[i], jac)
            for joint, camera, target, scale in entries:
                pixel, image_jac = _project(world[joint:joint+1], camera); difference = pixel[0]-target
                residual = scale*difference; data += float(residual @ residual)
                frame_errors.append(float(np.linalg.norm(difference)))
                if derivatives:
                    design = scale*image_jac[0] @ jac[joint]
                    diagonals[i] += design.T @ design; gradient[i] += design.T @ residual
            errors.append(frame_errors)
        prior = regularization*float(np.sum(delta*delta))
        temporal = float(np.sum(edges[:, None]*np.diff(delta, axis=0)**2))
        result = {'objective': .5*(data+prior+temporal), 'image_chi2': data, 'angle_prior_chi2': prior,
                  'correction_speed_chi2': temporal, 'points_world_m': points, 'errors_px': errors}
        if motion:
            values = original+delta
            residual = sum(stencil[:, j, None]*values[j:n-2+j] for j in range(3))
            loss = float(np.sum(residual*residual)); result['motion_acceleration_chi2'] = loss
            result['objective'] += .5*loss
            if derivatives:
                for j in range(3): gradient[j:n-2+j] += stencil[:, j, None]*residual
        return result, diagonals, gradient

    delta = np.zeros_like(original); initial, information, _ = evaluate(delta, True)
    strength = np.maximum(np.linalg.eigvalsh(information), 0.)/regularization
    history = []; status = 'unchanged_no_observations' if not np.any(active) else 'unchanged_exact_images'
    if initial['image_chi2'] > 1e-18:
        status = 'iteration_limit'
        for iteration in range(CONTRACT['maximum_iterations']):
            current, diagonals, gradient = evaluate(delta, True)
            diagonals += (regularization+CONTRACT['linear_system_ridge'])*np.eye(4)
            gradient += regularization*delta
            for i, weight in enumerate(edges):
                diagonals[i] += weight*np.eye(4); diagonals[i+1] += weight*np.eye(4)
                difference = weight*(delta[i]-delta[i+1]); gradient[i] += difference; gradient[i+1] -= difference
            if motion:
                diagonals += motion_diagonal[:, None, None]*np.eye(4)
                step = solve_banded_blocks(diagonals, motion_first-edges, motion_second, -gradient)
            else:
                step = solve_blocks(diagonals, edges, -gradient)
            if not np.isfinite(step).all(): status = 'nonfinite_step_rejected'; break
            step *= min(1., .10/max(float(np.max(abs(step))), 1e-20)); trials = []; accepted = False
            for backtrack in range(CONTRACT['maximum_backtracks']):
                trial = np.clip(delta+step*.5**backtrack, lower, upper)
                record = {'backtrack': backtrack, 'correction_rad': trial.tolist(), 'accepted': False, 'rejection': None,
                          'objective': None, 'image_chi2': None, 'angle_prior_chi2': None, 'correction_speed_chi2': None}
                if motion: record['motion_acceleration_chi2'] = None
                trials.append(record)
                try: value, _, _ = evaluate(trial)
                except ValueError:
                    record['rejection'] = 'camera_projection_domain'; continue
                record.update({k: value[k] for k in ('objective', 'image_chi2', 'angle_prior_chi2', 'correction_speed_chi2')})
                if motion:
                    record['motion_acceleration_chi2'] = value['motion_acceleration_chi2']
                    if value['image_chi2'] > initial['image_chi2']+1e-10:
                        record['rejection'] = 'source_image_loss_regression'; continue
                if np.max(abs(trial-delta)) <= CONTRACT['step_tolerance_rad']: status = 'step_tolerance'; break
                if value['objective'] < current['objective']-1e-12:
                    delta = trial; accepted = True; record['accepted'] = True; break
            history.append({'iteration': iteration+1, 'objective_before': current['objective'], 'trials': trials})
            if not accepted:
                if status != 'step_tolerance': status = 'no_descent'
                break
    final, _, _ = evaluate(delta)
    if final['image_chi2'] > initial['image_chi2']+1e-10 or final['objective'] > initial['objective']+1e-10:
        raise ValueError('Joint trajectory cannot regress its complete source objective')
    result = {'method': contract['method'], 'status': status, 'uncertainty': settings, 'uncertainty_calibrated': False,
        'input_angles_rad': original.tolist(), 'output_angles_rad': (original+delta).tolist(), 'correction_rad': delta.tolist(),
        'fixed_lengths_m': size.tolist(), 'fixed_origins_world_m': origin.tolist(), 'fixed_parent_rotations': parents.tolist(),
        'source_times_s': stamps.tolist(), 'active_frames': active.tolist(), 'temporal_edge_weights': edges.tolist(),
        'before': initial, 'after': final, 'initial_image_to_prior_eigenvalues': strength.tolist(),
        'weak_image_directions': np.count_nonzero(strength <= 1., axis=1).tolist(), 'iterations': history,
        'maximum_angle_change_rad': float(np.max(abs(delta))), 'depth_measured': False, 'observations_modified': False}
    if motion: result['motion_acceleration_stencil'] = stencil.tolist()
    return result
