from copy import deepcopy

import numpy as np
import pytest

from weft.constrained_stereo import _project
from weft.joint_pixel_trajectory import fit_joint_trajectory, solve_blocks
from weft.limb_position_fit import limb_geometry
from weft.spatial_pose import rotation_zyx


SETTINGS = {'schema_version': 1, 'image_std_fraction': .0025, 'angle_prior_std_rad': .3,
            'correction_speed_std_rad_s': 4.5, 'provenance': 'Explicit analytic development assumptions'}


def inputs(n=6, views=2, sign=-1):
    target = np.array([[.1+i*.01, -.2+i*.005, .3-i*.005, 1.1+i*.015] for i in range(n)])
    angles = target+np.array([.12, -.10, .14, -.15])
    origins = np.column_stack([np.linspace(0., .1, n), np.full(n, .1), np.full(n, 4.)])
    parents = np.repeat(rotation_zyx([.3, -.1, .2])[None, :, :], n, axis=0)
    rows = []; world = []
    for i, true in enumerate(target):
        points, _ = limb_geometry(true, [.4, .35], sign); points = points @ parents[i].T+origins[i]
        world.append(points); observed = []
        for v in range(views):
            camera = {'matrix': np.array([[700., 0., 640.], [0., 700., 480.], [0., 0., 1.]]),
                      'rotation': np.eye(3), 'translation': np.array([.6*v, 0., 0.]),
                      'radial': -.08, 'width': 1280, 'height': 960}
            pixels, _ = _project(points, camera)
            for joint in range(2):
                observed.append({'camera': deepcopy(camera), 'view_index': v, 'joint_index': joint,
                                 'pixel': pixels[joint].tolist(), 'confidence': .9})
        rows.append(observed)
    return dict(angles=angles, lengths=[.4, .35], origins=origins, parent_rotations=parents,
                times=np.arange(n)*.08, observations=rows, bend_sign=sign, uncertainty=SETTINGS), np.array(world)


@pytest.mark.parametrize('n', [2, 3, 9])
def test_block_solver_matches_an_independent_dense_system(n):
    rng = np.random.default_rng(741)
    matrix = rng.normal(size=(n, 4, 4)); diagonal = matrix.transpose(0, 2, 1) @ matrix+np.eye(4)*2
    edges = rng.uniform(0., 3., size=n-1); rhs = rng.normal(size=(n, 4))
    dense = np.zeros((4*n, 4*n))
    for i, weight in enumerate(edges):
        diagonal[i] += weight*np.eye(4); diagonal[i+1] += weight*np.eye(4)
        dense[4*i:4*i+4, 4*i+4:4*i+8] = -weight*np.eye(4)
        dense[4*i+4:4*i+8, 4*i:4*i+4] = -weight*np.eye(4)
    for i, block in enumerate(diagonal): dense[4*i:4*i+4, 4*i:4*i+4] = block
    saved = diagonal.copy()
    np.testing.assert_allclose(solve_blocks(diagonal, edges, rhs), np.linalg.solve(dense, rhs.ravel()).reshape(n, 4), atol=1e-12)
    np.testing.assert_array_equal(diagonal, saved)


def test_full_size_block_solver_retains_linear_storage_and_exact_residual():
    n = 900; edges = np.full(n-1, 3.); blocks = np.repeat((8*np.eye(4))[None, :, :], n, axis=0)
    rhs = np.sin(np.arange(4*n).reshape(n, 4)/23); solution = solve_blocks(blocks, edges, rhs)
    evaluated = np.einsum('iab,ib->ia', blocks, solution)
    evaluated[:-1] -= edges[:, None]*solution[1:]; evaluated[1:] -= edges[:, None]*solution[:-1]
    np.testing.assert_allclose(evaluated, rhs, atol=1e-12)


@pytest.mark.parametrize('views', [1, 2, 3, 4])
@pytest.mark.parametrize('sign', [-1, 1])
def test_joint_images_reconcile_a_wrong_knee_and_endpoint_with_fixed_lengths(views, sign):
    args, target = inputs(views=views, sign=sign); original = deepcopy(args)
    result = fit_joint_trajectory(**args)
    assert result['after']['image_chi2'] < .1*result['before']['image_chi2']
    assert result['after']['objective'] < result['before']['objective']
    before, after = [np.array(result[key]['points_world_m']) for key in ('before', 'after')]
    assert np.linalg.norm(after-target) < np.linalg.norm(before-target)
    assert np.linalg.norm(after[:, 0]-before[:, 0]) > .001
    np.testing.assert_allclose(np.linalg.norm(after[:, 0]-args['origins'], axis=1), .4, atol=1e-12)
    np.testing.assert_allclose(np.linalg.norm(after[:, 1]-after[:, 0], axis=1), .35, atol=1e-12)
    assert all(len(row) == 2*views for row in result['after']['errors_px'])
    assert not result['depth_measured'] and not result['observations_modified'] and not result['uncertainty_calibrated']
    assert np.max(abs(np.array(result['correction_rad']))) <= .3
    current = np.zeros_like(args['angles'])
    for iteration in result['iterations']:
        assert len(iteration['trials']) <= 4
        for trial in iteration['trials']:
            proposed = np.array(trial['correction_rad'])
            assert np.max(abs(proposed-current)) <= .1+1e-12
            if trial['accepted']:
                assert trial['objective'] < iteration['objective_before']; current = proposed
    np.testing.assert_array_equal(args['angles'], original['angles'])
    np.testing.assert_array_equal(args['origins'], original['origins'])


def test_actual_timestamps_and_missing_images_break_correction_coupling():
    args, _ = inputs(); args['times'] = [0., .04, .12, .20, .6, .68]; args['observations'][2] = []
    result = fit_joint_trajectory(**args)
    np.testing.assert_allclose(result['temporal_edge_weights'], [1/(4.5*.04)**2, 0., 0., 0., 1/(4.5*.08)**2])
    assert result['active_frames'] == [True, True, False, True, True, True]
    assert result['correction_rad'][2] == [0.]*4
    assert result['after']['errors_px'][2] == []


def test_unobserved_or_exact_trajectories_do_not_move():
    args, _ = inputs(); args['observations'] = [[] for _ in args['observations']]
    result = fit_joint_trajectory(**args)
    assert result['status'] == 'unchanged_no_observations' and result['maximum_angle_change_rad'] == 0.
    args, _ = inputs(); args['angles'] -= [.12, -.10, .14, -.15]
    result = fit_joint_trajectory(**args)
    assert result['status'] == 'unchanged_exact_images' and result['maximum_angle_change_rad'] == 0.


def test_knee_only_images_keep_missing_distal_depth_explicit():
    args, _ = inputs(views=1)
    args['observations'] = [[row for row in rows if row['joint_index'] == 0] for rows in args['observations']]
    result = fit_joint_trajectory(**args)
    assert min(result['weak_image_directions']) >= 2
    assert all(len(row) == 1 for row in result['after']['errors_px']) and not result['depth_measured']
    assert np.max(abs(np.array(result['correction_rad'])[:, 3])) == 0.


def test_resizing_images_preserves_objective_and_corrections():
    args, _ = inputs(); original = fit_joint_trajectory(**args)
    for rows in args['observations']:
        for row in rows:
            row['camera']['matrix'][:2] *= .5; row['camera']['width'] //= 2; row['camera']['height'] //= 2
            row['pixel'] = (np.array(row['pixel'])*.5).tolist()
    resized = fit_joint_trajectory(**args)
    np.testing.assert_allclose(original['correction_rad'], resized['correction_rad'], atol=1e-12)
    assert original['after']['objective'] == pytest.approx(resized['after']['objective'], abs=1e-12)


def test_conflicting_view_is_retained_in_every_joint_loss():
    args, _ = inputs(views=3); args['observations'][2][5]['pixel'][0] += 35
    result = fit_joint_trajectory(**args)
    for label in ('before', 'after'):
        errors = result[label]['errors_px']
        expected = sum(.9*(e/(1280*.0025))**2 for row in errors for e in row)
        assert result[label]['image_chi2'] == pytest.approx(expected, abs=1e-10)
    assert len(result['after']['errors_px'][2]) == 6


@pytest.mark.parametrize('damage', ['time', 'gap', 'angle', 'hinge', 'length', 'parent', 'origin', 'views',
    'duplicate', 'camera', 'pixel', 'confidence', 'settings', 'provenance'])
def test_invalid_inputs_cannot_hide_or_invent_joint_observations(damage):
    args, _ = inputs(); args['uncertainty'] = deepcopy(SETTINGS)
    if damage == 'time': args['times'][2] = args['times'][1]
    elif damage == 'gap': args['times'][2] = args['times'][1]+.001
    elif damage == 'angle': args['angles'][0, 0] = float('nan')
    elif damage == 'hinge': args['angles'][0, 3] = 3.
    elif damage == 'length': args['lengths'][0] = 0.
    elif damage == 'parent': args['parent_rotations'][0] = np.diag([-1., 1., 1.])
    elif damage == 'origin': args['origins'][0, 0] = 2001
    elif damage == 'views': args['observations'][0][0]['view_index'] = 4
    elif damage == 'duplicate': args['observations'][0].append(deepcopy(args['observations'][0][0]))
    elif damage == 'camera': args['observations'][0][1]['camera']['translation'][0] += .1
    elif damage == 'pixel': args['observations'][0][0]['pixel'][0] = -1
    elif damage == 'confidence': args['observations'][0][0]['confidence'] = .49
    elif damage == 'settings': args['uncertainty']['correction_speed_std_rad_s'] = 0.
    elif damage == 'provenance': args['uncertainty']['provenance'] = ''
    with pytest.raises(ValueError): fit_joint_trajectory(**args)


def test_declared_trajectory_deadline_stops_before_fitting(monkeypatch):
    args, _ = inputs(); ticks = iter([0., 11.])
    monkeypatch.setattr('weft.joint_pixel_trajectory.time.monotonic', lambda: next(ticks))
    with pytest.raises(TimeoutError, match='deadline'): fit_joint_trajectory(**args)
