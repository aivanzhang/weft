"""Analytic camera/root fixtures; no source fitting or native body construction."""

from copy import deepcopy
from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from weft import body_camera_root_refinement as fit
from weft.camera_reference import project_world
from weft.io import canonical_hash


def encoded(document):
    return json.dumps(document, sort_keys=True, allow_nan=False).encode()


def fixture(*, shift=True, floor=True, count=4):
    center = np.array([0., 0., -4.])
    matrix = np.array([[600., 3., 320.], [0., 620., 240.], [0., 0., 1.]])
    relative = np.array([[-.3, -.5, .1], [.3, -.5, -.1], [-.4, -.1, .2],
                         [.4, -.1, -.2], [-.15, .5, .15], [.15, .5, -.15]])
    rotation = fit._rotation(np.array([.001, -.0015, .0008])) if shift else np.eye(3)
    document = {'schema_version': 1, 'profile': fit.INPUT_PROFILE,
                'bindings': {key: 'a' * 64 for key in fit._BINDINGS},
                'scale_model_units_per_unit': 1.,
                'camera': {'rotation': np.eye(3).tolist(), 'center': center.tolist(), 'K': matrix.tolist(),
                           'radial_k1': .02, 'quality': 'authored', 'provenance': 'Analytic fixture, not physical evidence'},
                'floor': {'normal': [0., 0., 1.], 'offset_model_units': 0.}, 'samples': [], 'floor_checks': None}
    for index in range(count):
        root = np.array([.03 * index, .002 * index, 0.])
        points = relative + root
        delta = np.array([.001, -.0005, .0008]) * index if shift else np.zeros(3)
        _, pixels, _ = project_world(points + delta, rotation, -rotation @ center, matrix, .02)
        document['samples'].append({'evidence_id': 'a' * 64 + f':frame:{index * 2}', 'frame': index * 2,
            'time_s': index / 15, 'decoder_timestamp_s': index / 15 + .003, 'root': root.tolist(),
            'points': [{'name': f'endpoint_{j}', 'world': point.tolist(), 'pixel': pixels[j].tolist(), 'weight': 1.}
                       for j, point in enumerate(points)]})
    if floor:
        times = [row['time_s'] for row in document['samples']]
        schedule = [('source_pose', t) for t in times] + [('control_grid', i / 30) for i in range((count - 1) * 2 + 1)]
        document['floor_checks'] = {'bindings': {}, 'rows': [
            {'kind': kind, 'time_s': stamp, 'gaps': {'left_support': .02, 'right_support': .03}} for kind, stamp in schedule]}
        bind_floor(document)
    return document, rotation


def bind_floor(document):
    document['floor_checks']['bindings'] = {
        **{key: document['bindings'][key] for key in ('body_sha256', 'body_angles_sha256', 'body_shape_sha256')},
        'root_reference_sha256': canonical_hash([{'time_s': row['time_s'], 'translation': row['root']} for row in document['samples']]),
        'floor_sha256': canonical_hash(document['floor'])}


@pytest.mark.parametrize('radial', [0., .12, -.08])
def test_projection_and_analytic_jacobians_match_existing_semantics(radial):
    points = np.array([[-.4, .2, .3], [.3, -.2, .5], [.1, .4, -.1]])
    rotation = fit._rotation(np.array([.1, -.04, .07])); center = np.array([.2, -.3, -4.])
    matrix = np.array([[700., 8., 400.], [0., 730., 300.], [0., 0., 1.]])
    actual = fit.project_with_jacobians(points, rotation, center, matrix, radial)
    camera, pixels, valid = project_world(points, rotation, -rotation @ center, matrix, radial)
    assert np.array_equal(actual['pixels'], pixels)
    assert np.array_equal(actual['camera_points'], camera)
    assert np.array_equal(actual['valid'], valid)
    epsilon = 1e-6
    for axis in range(3):
        direction = np.eye(3)[axis] * epsilon
        plus = fit.project_with_jacobians(points + direction, rotation, center, matrix, radial)['pixels']
        minus = fit.project_with_jacobians(points - direction, rotation, center, matrix, radial)['pixels']
        assert np.allclose((plus - minus) / (2 * epsilon), actual['root_jacobian'][:, :, axis], atol=1e-6, rtol=1e-7)
        plus = fit.project_with_jacobians(points, fit._rotation(direction) @ rotation, center, matrix, radial)['pixels']
        minus = fit.project_with_jacobians(points, fit._rotation(-direction) @ rotation, center, matrix, radial)['pixels']
        assert np.allclose((plus - minus) / (2 * epsilon), actual['rotation_jacobian'][:, :, axis], atol=1e-6, rtol=1e-7)


def test_behind_camera_is_unknown_not_clipped():
    result = fit.project_with_jacobians([[0., 0., -1.], [1., 2., 3.]], np.eye(3), [0., 0., 0.], np.eye(3))
    assert result['valid'].tolist() == [False, True]
    assert np.isnan(result['pixels'][0]).all() and np.isnan(result['root_jacobian'][0]).all()


def test_proper_rotation_composition_roundoff_is_not_a_new_limit():
    rotation = np.diag([1. + 2e-16, 1., 1.])
    result = fit.project_with_jacobians([[0., 0., 1.]], rotation, [0., 0., 0.], np.eye(3))
    assert result['valid'].tolist() == [True]


def test_translation_and_scale_gauges_explain_fixed_center_and_scale():
    points = np.array([[-.3, .5, 2.], [.4, -.2, 3.], [.2, .7, 4.]])
    center = np.array([.2, .3, -1.]); rotation = fit._rotation([.03, -.1, .05]); matrix = np.diag([600., 610., 1.])
    baseline = fit.project_with_jacobians(points, rotation, center, matrix)['pixels']
    offset = np.array([.5, -.1, .3])
    assert np.allclose(fit.project_with_jacobians(points + offset, rotation, center + offset, matrix)['pixels'], baseline)
    assert np.allclose(fit.project_with_jacobians(3 * points, rotation, 3 * center, matrix)['pixels'], baseline)


def test_data_only_rank_does_not_promote_elimination_roundoff():
    rng = np.random.default_rng(7)
    b = rng.normal(size=(3, 6, 2, 3))
    a = np.einsum('npij,jk->npik', b, rng.normal(size=(3, 3)))
    report = fit.local_identifiability(a, b, np.ones((3, 6)), movable_roots=[True] * 3)
    assert report['camera_after_eliminating_roots']['rank'] == 0
    assert report['rank'] == 9 and report['variables'] == 12
    assert report['status'] == 'unidentified_or_ill_conditioned'
    assert not report['regularization_used_for_rank'] and not report['global_uniqueness_claim']


def test_arrowhead_step_matches_small_dense_system():
    rng = np.random.default_rng(12)
    a, b = rng.normal(size=(4, 6, 2, 3)), rng.normal(size=(4, 6, 2, 3))
    residual, weights = rng.normal(size=(4, 6, 2)), rng.uniform(.5, 1., size=(4, 6))
    movable = [False, True, True, True]
    camera, roots = fit._step({'rotation_jacobian': a, 'root_jacobian': b}, residual, weights, movable, None)
    design = np.zeros((48, 12))
    for index in range(4):
        design[index * 12:(index + 1) * 12, :3] = a[index].reshape(12, 3)
        if index:
            design[index * 12:(index + 1) * 12, index * 3:(index + 1) * 3] = b[index].reshape(12, 3)
    scale = np.repeat(np.sqrt(weights.ravel()), 2)
    design *= scale[:, None]
    hessian = design.T @ design
    damping = max(1e-8, np.diag(hessian[:3, :3]).max() * 1e-8)
    expected = np.linalg.solve(hessian + damping * np.eye(12), -design.T @ (residual.ravel() * scale))
    assert np.allclose(np.concatenate([camera, roots[1:].ravel()]), expected, atol=1e-12)
    assert np.array_equal(roots[0], np.zeros(3))


def test_joint_camera_root_fixture_recovers_known_changes_without_input_mutation():
    document, target_rotation = fixture(); before = deepcopy(document); raw = encoded(document)
    result = fit.propose_camera_root_refinement(raw)
    report = result['report']
    assert document == before and encoded(document) == raw
    assert result['status'] == 'proposed'
    assert report['image_rmse_after_px'] < 1e-5
    assert report['image_rmse_before_px'] > .1
    assert np.allclose(result['camera_candidate']['rotation'], target_rotation, atol=1e-6)
    for index, row in enumerate(report['samples']):
        assert np.allclose(row['root_delta'], np.array([.001, -.0005, .0008]) * index, atol=1e-6)
        assert row['decoder_timestamp_s'] == document['samples'][index]['decoder_timestamp_s']
    assert all(report['gates'].values())
    assert report['initial_identifiability']['status'] == report['final_identifiability']['status'] == 'locally_identifiable'
    assert report['input_sha256'] == sha256(raw).hexdigest()
    assert report['bindings'] == document['bindings'] and not report['source_qualified'] and not report['admitted']
    assert not report['promoted'] and report['geometry_admission'] == 'unverified_not_executed'
    assert result['root_reference_candidate'][0]['translation'] == document['samples'][0]['root']
    for key in document['camera']:
        if key != 'rotation':
            assert result['camera_candidate'][key] == document['camera'][key]
    assert set(result) == {'status', 'camera_candidate', 'root_reference_candidate', 'report'}


def test_exact_baseline_remains_unchanged_not_claimed_improvement():
    document, _ = fixture(shift=False)
    result = fit.propose_camera_root_refinement(encoded(document))
    assert result['status'] == 'unchanged'
    assert result['report']['counts']['accepted_updates'] == 0
    assert result['report']['counts']['evaluated_trials'] == 0
    assert result['report']['image_rmse_after_px'] == 0.


def test_maximum_native_free_fixture_keeps_matrix_width_and_output_bounded(monkeypatch):
    document, _ = fixture(shift=False, count=900)
    for row in document['samples']:
        row['points'] += [{**deepcopy(point), 'name': point['name'] + '_second'} for point in row['points']]
    for row in document['floor_checks']['rows']:
        row['gaps'] = {f'support_{i}': .02 for i in range(64)}
    raw = encoded(document)
    assert len(raw) <= fit.CONTRACT['maximum_input_bytes']
    original_svd = np.linalg.svd
    def narrow(matrix, *args, **kwargs):
        assert matrix.ndim == 2 and matrix.shape[1] == 3
        return original_svd(matrix, *args, **kwargs)
    monkeypatch.setattr(np.linalg, 'svd', narrow)
    monkeypatch.setattr(np.linalg, 'solve', lambda *a, **kw: pytest.fail('An exact match needs no solve'))
    result = fit.propose_camera_root_refinement(raw)
    assert result['status'] == 'unchanged'
    assert result['report']['known_endpoints'] == 10800
    assert len(result['report']['floor_checks']) == 2699
    assert result['report']['counts']['evaluated_trials'] == 0
    assert len(encoded(result)) <= fit.CONTRACT['maximum_output_bytes']


def test_ambiguous_identical_points_do_not_use_damping_as_evidence():
    document, _ = fixture()
    for row in document['samples']:
        for point in row['points']:
            point['world'] = [0., 0., 0.]
    result = fit.propose_camera_root_refinement(encoded(document))
    assert result['status'] == 'unavailable'
    assert result['report']['counts']['iterations'] == 0
    assert result['report']['initial_identifiability']['rank'] < result['report']['initial_identifiability']['variables']
    assert result['camera_candidate'] == document['camera']
    assert all(row['root_delta'] == [0., 0., 0.] for row in result['report']['samples'])


def test_missing_pixels_and_decoder_time_stay_unknown_and_missing_root_stays_fixed():
    document, _ = fixture()
    document['samples'][1]['decoder_timestamp_s'] = None
    for point in document['samples'][1]['points']:
        point.update(pixel=None, weight=0.)
    document['samples'][2]['points'][2].update(pixel=None, weight=0.)
    result = fit.propose_camera_root_refinement(encoded(document))
    report = result['report']
    assert report['unknown_endpoints'] == 7 and report['known_endpoints'] == 17
    row = report['samples'][1]
    assert row['root_delta'] == [0., 0., 0.] and row['decoder_timestamp_s'] is None
    assert row['fit_status'] == 'unavailable_pixels_root_unchanged'
    assert all(point['pixel'] is None and point['error_before_px'] is None and point['error_after_px'] is None for point in row['endpoints'])
    assert report['samples'][2]['endpoints'][2]['error_after_px'] is None


def test_no_floor_data_is_unavailable_never_qualified():
    document, _ = fixture(floor=False)
    report = fit.propose_camera_root_refinement(encoded(document))['report']
    assert report['floor_checks'] is None
    assert report['gates']['saved_floor_no_worse'] is None
    assert report['gates']['saved_floor_feasibility'] is None
    assert report['geometry_qualification'] == 'unavailable' and not report['admitted']


def test_all_pixels_missing_has_unknown_image_checks_not_a_vacuous_pass():
    document, _ = fixture()
    for row in document['samples']:
        for point in row['points']:
            point.update(pixel=None, weight=0.)
    result = fit.propose_camera_root_refinement(encoded(document))
    assert result['status'] == 'unavailable'
    assert result['report']['image_rmse_before_px'] is None
    assert result['report']['image_rmse_after_px'] is None
    assert result['report']['gates']['image_pointwise_no_worse'] is None


@pytest.mark.parametrize('value', ['0.1', True])
def test_numeric_strings_and_boolean_coordinates_are_not_coerced(value):
    document, _ = fixture()
    document['samples'][0]['points'][0]['world'][0] = value
    with pytest.raises(ValueError):
        fit.propose_camera_root_refinement(encoded(document))


def test_saved_floor_failure_stays_failure_despite_image_improvement():
    document, _ = fixture()
    for row in document['floor_checks']['rows']:
        row['gaps']['left_support'] = -.04
    report = fit.propose_camera_root_refinement(encoded(document))['report']
    assert report['image_rmse_after_px'] < report['image_rmse_before_px']
    assert report['gates']['saved_floor_no_worse']
    assert not report['gates']['saved_floor_feasibility']
    assert report['contract']['floor_penetration_limit_model_units'] == .025
    assert report['floor_checks'][0]['gaps_after']['left_support'] == -.04


def test_no_floor_loss_or_downward_repair_bypasses_fixed_gap_constraint():
    document, _ = fixture(shift=False)
    camera = document['camera']
    for index, row in enumerate(document['samples']):
        xyz = np.array([point['world'] for point in row['points']]) + [0., 0., -.001 * index]
        projected = fit.project_with_jacobians(xyz, camera['rotation'], camera['center'], camera['K'], camera['radial_k1'])['pixels']
        for point, pixel in zip(row['points'], projected, strict=True):
            point['pixel'] = pixel.tolist()
    for row in document['floor_checks']['rows']:
        row['gaps']['left_support'] = -.03
    result = fit.propose_camera_root_refinement(encoded(document))
    assert result['report']['counts']['rejected_floor'] > 0
    assert result['report']['gates']['saved_floor_no_worse']
    assert not result['report']['gates']['saved_floor_feasibility']


def test_grid_gaps_use_time_interpolation_without_invented_image_targets():
    document, _ = fixture(shift=False)
    times = np.array([row['time_s'] for row in document['samples']])
    delta = np.array([[0., 0., 0.], [0., 0., .01], [0., 0., .02], [0., 0., .03]])
    rows = fit._floor(document, times, delta, np.array([0., 0., 1.]))
    middle = next(row for row in rows if row['kind'] == 'control_grid' and row['time_s'] == 1 / 30)
    assert middle['gaps_after']['left_support'] == pytest.approx(.025)
    assert set(middle) == {'kind', 'time_s', 'gaps_before', 'gaps_after'}
    assert len(rows) == 4 + 7


@pytest.mark.parametrize('change', [
    lambda d: d['bindings'].update(body_angles_sha256='b' * 64),
    lambda d: d['bindings'].update(body_shape_sha256='b' * 64),
    lambda d: d['samples'][1]['root'].__setitem__(0, .2),
    lambda d: d['floor'].update(offset_model_units=.1),
    lambda d: d['floor_checks']['rows'].pop(),
    lambda d: d['floor_checks']['rows'][-1].update(time_s=.21),
    lambda d: d['floor_checks']['rows'][0]['gaps'].update(left_support=None),
    lambda d: d['floor_checks']['rows'][0]['gaps'].pop('left_support'),
])
def test_incomplete_or_mismatched_floor_custody_rejected_before_solver(change, monkeypatch):
    document, _ = fixture(); change(document)
    monkeypatch.setattr(fit, '_step', lambda *a, **k: pytest.fail('Solver must not run on malformed evidence'))
    with pytest.raises(ValueError):
        fit.propose_camera_root_refinement(encoded(document))


@pytest.mark.parametrize('change', [
    lambda d: d.update(schema_version=True),
    lambda d: d.update(unrecognized={}),
    lambda d: d['camera'].update(rotation=np.diag([-1., 1., 1.]).tolist()),
    lambda d: d['camera'].update(provenance='x' * 2001),
    lambda d: d['camera']['K'][2].__setitem__(0, .1),
    lambda d: d['samples'][1].update(time_s=.3),
    lambda d: d['samples'][1].update(evidence_id='b' * 64 + ':frame:2'),
    lambda d: d['samples'][0]['points'][0].update(pixel=None),
    lambda d: d['samples'][0]['points'][0].update(weight=.2),
    lambda d: d['samples'][1]['points'][0].update(name='different'),
    lambda d: d['samples'][0]['points'][0].update(world=[0., 0., -5.]),
])
def test_invalid_fixed_contract_rejected(change):
    document, _ = fixture(); change(document)
    with pytest.raises(ValueError):
        fit.propose_camera_root_refinement(encoded(document))


@pytest.mark.parametrize('raw', [b'{"a":1,"a":2}', b'{"a":NaN}', b'[' * 1100 + b']' * 1100,
                                 b' ' * (8 * 1024 * 1024 + 1)], ids=['duplicate', 'nonfinite', 'deep', 'oversized'])
def test_strict_json_and_byte_bound(raw):
    with pytest.raises(ValueError):
        fit.propose_camera_root_refinement(raw)


@pytest.mark.parametrize('settings', [{'maximum_seconds': 11}, {'maximum_iterations': 9},
                                     {'maximum_backtracks': True}, {'minimum_confidence': .1},
                                     {'maximum_root_translation_model_units': .081}])
def test_only_tighter_bounded_settings(settings):
    document, _ = fixture()
    with pytest.raises(ValueError):
        fit.propose_camera_root_refinement(encoded(document), settings=settings)


def test_tighter_iteration_limit_and_partial_timeout_preserve_last_accepted_state(monkeypatch):
    document, _ = fixture(); raw = encoded(document)
    first = fit.propose_camera_root_refinement(raw, settings={'maximum_iterations': 1})
    assert first['report']['counts']['iterations'] == first['report']['counts']['accepted_updates'] == 1
    original = fit._step; calls = 0
    def limited(*args):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise TimeoutError('Cooperative limit fixture')
        return original(*args)
    monkeypatch.setattr(fit, '_step', limited)
    partial = fit.propose_camera_root_refinement(raw)
    assert partial['status'] == 'partial' and partial['report']['reason'] == 'time_limit'
    assert partial['camera_candidate'] == first['camera_candidate']
    assert partial['root_reference_candidate'] == first['root_reference_candidate']
    assert partial['report']['counts']['accepted_updates'] == 1


def test_import_and_fixture_have_no_native_body_or_media_dependency():
    source = str(Path(__file__).resolve().parents[1] / 'src')
    code = f'''import importlib.abc, sys
sys.path.insert(0, {source!r})
class Guard(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {{'mujoco', 'gymnasium', 'cv2', 'mediapipe'}} or fullname in {{'weft.human', 'weft.mobile_human'}}:
            raise AssertionError('Forbidden native/body/media import: ' + fullname)
sys.meta_path.insert(0, Guard())
from weft.body_camera_root_refinement import project_with_jacobians
import numpy as np
assert project_with_jacobians([[0.,0.,1.]],np.eye(3),[0.,0.,0.],np.eye(3))['valid'].tolist()==[True]
print('native-free')
'''
    result = subprocess.run([sys.executable, '-I', '-B', '-c', code], capture_output=True, text=True, timeout=5,
                            env={'PATH': os.environ.get('PATH', '/usr/bin:/bin'), 'OPENBLAS_NUM_THREADS': '1', 'OMP_NUM_THREADS': '1'})
    assert result.returncode == 0, result.stderr[:2000]
    assert result.stdout == 'native-free\n'
