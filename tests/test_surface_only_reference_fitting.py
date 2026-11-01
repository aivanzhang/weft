"""Source-qualified interval scheduling with the existing QP and analytical geometry."""

from copy import deepcopy
from hashlib import sha256
import importlib
import json
import os
from pathlib import Path
import time

import numpy as np
import pytest

from test_multiview_landmark_evidence import denied, local_only


# Excludes only elapsed seconds; geometry/QP/projection calls remain in the digest.
GOLDENS = {
    'ordinary': '13d401a368d77e49339b5d686cf7c496c74ee692c6e4e2e8e4aaf0ad344d5bce',
    'camera': '76dcbc243e311ecf71184bf021f7071780c9e08363968fce34755953cc7a8297',
    'multiview': '58f0d64bf3a8937c12c494ad5f8c86d9a0c133eec971ac0dfc278b94ba491b85',
}


def raw(value):
    return json.dumps(value, allow_nan=False).encode()


def missing_images(value, indices=(1,)):
    for evidence in value['evidence']:
        for index in indices:
            for point in evidence['frames'][index]['landmarks']:
                point.update(visibility=.49, presence=.49)


def fixture(monkeypatch, *, mode='camera', annotation='known', target=.05, fps=30., missing_indices=(1,)):
    api = importlib.import_module('weft.whole_body_reference_refinement')
    camera = importlib.import_module('test_camera_body_fit')
    camera_api = importlib.import_module('weft.camera_body_fit')
    if mode == 'ordinary':
        ordinary = importlib.import_module('test_whole_body_reference_refinement')
        interval = importlib.import_module('test_whole_body_surface_interval_refinement')
        body, evidence = ordinary.fixture(count=5)
        for point in evidence['frames'][2]['landmarks']:
            point.update(visibility=.49, presence=.49)
        ordinary.bind_root(body, evidence)
        with monkeypatch.context() as nested:
            nested.setattr(ordinary, 'fixture', lambda: deepcopy((body, evidence)))
            data = interval.fixture(api)
        body, evidence, arguments = data[:3]
        geometry = lambda final, kind: interval.IntervalGeometry(api, evidence, kind, final)
        return {'body': body, 'evidence': evidence, 'arguments': arguments, 'geometry': geometry, 'missing_index': 2, 'inputs': data}
    if mode == 'multiview':
        data, secondary = importlib.import_module('test_camera_multiview_fit').fixture(
            fps=fps, change=lambda value: missing_images(value, missing_indices), annotation=annotation)
    else:
        data = camera.fixture(camera_api, fps=fps, target=target,
            source_change=lambda value: missing_images(value, missing_indices), annotation=annotation)
        secondary = None
    arguments = {'proposal_objective': 'camera_surface_interval_v1', 'camera_document': raw(data['camera']),
        'camera_image_scale': .01, 'surface_interval_objective': data['objective'],
        'surface_interval_weight': 2., 'joint_mask': ['left_knee']}
    if secondary is not None:
        arguments.update(proposal_objective='camera_multiview_surface_interval_v1', camera_secondary_view=secondary)

    class Geometry(camera.Geometry):
        def evaluate(self, angles, root, derivatives=False, *, surface_intervals=False):
            state = super().evaluate(angles, root, derivatives, surface_intervals=surface_intervals)
            if self.mode == 'zero_gradient' and derivatives:
                state['point_jac'][:] = 0.; state['pelvis_jac'][:] = 0.
            if self.mode == 'fresh_surface_failure' and self.final:
                state['pelvis_position'][1] += .2
            if self.mode == 'interval_plateau':
                column = self.names.index('left_knee'); q = angles[column]
                # The nonlinear trial lands inside the interval; no later block needs a direction.
                if q >= .04:
                    state['points'][10, 1] += .05 - q
                    if derivatives:
                        state['point_jac'][10, 1, column] -= 1.
            return state

    return {'body': data['body'], 'evidence': data['evidence'], 'arguments': arguments,
        'geometry': lambda final, kind: Geometry(camera_api, kind, final), 'missing_index': 1, 'inputs': data}


def run(monkeypatch, data, *, kind='ordinary', **options):
    api = importlib.import_module('weft.whole_body_reference_refinement')
    engines = []
    def create(body):
        assert raw(body) == raw(data['body'])
        engine = data['geometry'](bool(engines), kind); engines.append(engine); return engine
    monkeypatch.setattr(api, '_make_geometry', create)
    before = raw([data['body'], data['evidence']])
    started = time.monotonic()
    result = api.propose_whole_body_reference_refinement(raw(data['body']), raw(data['evidence']),
        **(data['arguments'] | options))
    assert time.monotonic() - started < 5.
    assert len(engines) == 2 and all(engine.closed for engine in engines)
    assert result['report']['geometry_calls'] == sum(engine.calls for engine in engines)
    assert raw([data['body'], data['evidence']]) == before
    return result


def proposed(result):
    return result['angle_proposal'] if result['human_candidate'] is None else result['human_candidate']


@pytest.mark.parametrize('mode', ['ordinary', 'camera', 'multiview'])
def test_preserve_matches_published_bytes_and_calls(monkeypatch, mode, record_property):
    data = fixture(monkeypatch, mode=mode)
    baseline = run(monkeypatch, data)
    canonical = deepcopy(baseline); canonical['report'].pop('seconds')
    digest = sha256(raw(canonical)).hexdigest()
    record_property('published_default_sha256', digest)
    capture = os.environ.get('WEFT_SURFACE_ONLY_GOLDEN_OUT')
    if capture:
        path = Path(capture); path.mkdir(exist_ok=True)
        with (path / (mode + '.json')).open('xb') as stream:
            stream.write(raw({'sha256': digest, 'fitter_sha256': sha256(Path(importlib.import_module(
                'weft.whole_body_reference_refinement').__file__).read_bytes()).hexdigest(),
                'geometry_calls': baseline['report']['geometry_calls'], 'qp_calls': baseline['report']['qp_calls']}))
        return
    assert digest == GOLDENS[mode]
    index = data['missing_index']
    term = baseline['report']['surface_intervals']['references'][index]
    assert term['before']['objective_m2'] > 0 and term['after']['objective_m2'] == term['before']['objective_m2']
    assert baseline['report']['references'][index]['status'] == 'unchanged_missing_source_targets'
    explicit = run(monkeypatch, data, missing_image_policy='preserve')
    explicit['report'].pop('seconds')
    assert raw(explicit) == raw(canonical)
    assert 'missing_image_policy' not in proposed(explicit)['whole_body_reference_refinement']
    assert 'missing_image_policy' not in explicit['report']['objective']
    assert all('surface_only_eligible' not in row and 'surface_only_attempted' not in row for row in explicit['report']['references'])


@pytest.mark.parametrize('mode', ['ordinary', 'camera', 'multiview'])
def test_qualified_interval_moves_missing_image_row_without_fabricating_targets(monkeypatch, mode, record_property):
    data = fixture(monkeypatch, mode=mode)
    result = run(monkeypatch, data, missing_image_policy='qualified_surfaces')
    report, candidate, index = result['report'], proposed(result), data['missing_index']
    row, interval = report['references'][index], report['surface_intervals']['references'][index]
    assert result['status'] == 'unadmitted_construction_proposal' and all(report['gates'].values())
    assert row['surface_only_eligible'] and row['surface_only_attempted']
    assert row['status'] == 'changed_surface_only_proposal'
    assert interval['after']['objective_m2'] < interval['before']['objective_m2']
    assert all(point['weight'] == 0 and point['error_before_px'] is point['error_after_px'] is None for point in row['points'])
    assert all(not value for value in report['references'][0]['correction_rad'].values())
    assert report['references'][0]['status'] == 'frozen_first_reference'
    assert report['references'][0]['surface_only_eligible'] is report['references'][0]['surface_only_attempted'] is False
    reference = candidate['reference_candidate'] if mode != 'ordinary' else candidate['reference']
    assert reference[0] == data['body']['reference'][0]
    assert [value['time_s'] for value in reference] == [value['time_s'] for value in data['body']['reference']]
    assert len(report['surface_intervals']['source_slots']) == 10
    assert report['objective']['missing_image_policy'] == candidate['whole_body_reference_refinement']['missing_image_policy'] == 'qualified_surfaces'
    assert report['surface_intervals']['contact'] == 'unknown' and report['physics_steps'] == 0
    assert any(record['kind'] == 'qp' and record['reference_index'] == index for record in report['surface_intervals']['trial_journal'])
    record_property('missing_row_before_m2', interval['before']['objective_m2'])
    record_property('missing_row_after_m2', interval['after']['objective_m2'])
    record_property('geometry_calls', report['geometry_calls'])
    record_property('interval_evaluations', report['surface_intervals']['evaluations'])
    destination = os.environ.get('WEFT_SURFACE_ONLY_FIXTURE_OUT')
    if destination and mode == 'camera':
        importlib.import_module('test_camera_body_fit').save_fixture(Path(destination), data['inputs'], result)


@pytest.mark.parametrize('annotation', ['missing', 'unknown_height', 'withheld', 'mismatched'])
def test_unavailable_intervals_never_schedule_the_missing_image_row(monkeypatch, annotation):
    data = fixture(monkeypatch, annotation=annotation)
    api = importlib.import_module('weft.whole_body_reference_refinement')
    calls, derivative_calls = [], []
    original = api._SurfaceIntervalFit.record_qp
    def record(self, index, *args, **kwargs):
        calls.append(index)
        return original(self, index, *args, **kwargs)
    monkeypatch.setattr(api._SurfaceIntervalFit, 'record_qp', record)
    evaluate = api._SurfaceIntervalFit.evaluate
    def evaluated(self, index, *args, **kwargs):
        if kwargs.get('derivatives'):
            derivative_calls.append(index)
        return evaluate(self, index, *args, **kwargs)
    monkeypatch.setattr(api._SurfaceIntervalFit, 'evaluate', evaluated)
    result = run(monkeypatch, data, missing_image_policy='qualified_surfaces')
    row = result['report']['references'][1]
    assert not row['surface_only_eligible'] and not row['surface_only_attempted']
    assert row['status'] == 'unchanged_missing_source_targets' and not any(row['correction_rad'].values())
    assert 1 not in calls and 1 not in derivative_calls
    assert len(result['report']['surface_intervals']['source_slots']) == 10
    assert result['report']['surface_intervals']['references'][1]['before']['objective_m2'] is None


def test_already_inside_envelope_never_schedules_surface_only_qp(monkeypatch):
    data = fixture(monkeypatch, target=0.)
    result = run(monkeypatch, data, missing_image_policy='qualified_surfaces')
    row = result['report']['references'][1]
    assert result['report']['surface_intervals']['references'][1]['before']['objective_m2'] == 0.
    assert not row['surface_only_eligible'] and not row['surface_only_attempted']
    assert not any(row['correction_rad'].values())


def test_eligible_but_zero_direction_has_truthful_unchanged_status(monkeypatch):
    result = run(monkeypatch, fixture(monkeypatch), kind='zero_gradient', missing_image_policy='qualified_surfaces')
    row = result['report']['references'][1]
    assert row['surface_only_eligible'] and row['surface_only_attempted']
    assert row['status'] == 'unchanged_no_accepted_surface_step'
    assert not any(row['correction_rad'].values())


def test_missing_first_pose_stays_frozen_despite_qualified_surface(monkeypatch):
    data = fixture(monkeypatch, missing_indices=(0, 1))
    result = run(monkeypatch, data, missing_image_policy='qualified_surfaces')
    report = result['report']; row = report['references'][0]
    interval = report['surface_intervals']['references'][0]
    assert interval['before']['objective_m2'] > 0 and interval['before'] == interval['after']
    assert row['status'] == 'frozen_first_reference'
    assert not row['surface_only_eligible'] and not row['surface_only_attempted']
    assert not any(row['correction_rad'].values())
    assert all(point['weight'] == 0 and point['error_before_px'] is point['error_after_px'] is None for point in row['points'])
    assert all(record['reference_index'] != 0 for record in report['surface_intervals']['trial_journal'])


def test_current_interval_rechecked_before_next_block_or_sweep(monkeypatch):
    data = fixture(monkeypatch)
    result = run(monkeypatch, data, kind='interval_plateau', joint_mask=['left_knee', 'right_knee'],
        missing_image_policy='qualified_surfaces')
    report = result['report']; interval = report['surface_intervals']['references'][1]
    assert interval['before']['objective_m2'] > 0 and interval['after']['objective_m2'] == 0.
    qps = [row for row in report['surface_intervals']['trial_journal'] if row['kind'] == 'qp' and row['reference_index'] == 1]
    assert len(qps) == 1 and qps[0]['block'] == ['left_knee']
    assert report['references'][1]['surface_only_attempted'] and all(report['gates'].values())


@pytest.mark.parametrize('kind', ['curved_floor', 'camera_depth', 'existing_floor_failure', 'fresh_surface_failure'])
def test_surface_only_descent_cannot_bypass_existing_gates(monkeypatch, kind):
    result = run(monkeypatch, fixture(monkeypatch, target=.15), kind=kind, missing_image_policy='qualified_surfaces')
    report = result['report']
    assert report['references'][1]['surface_only_eligible'] and report['references'][1]['surface_only_attempted']
    assert report['gates']['first_reference_frozen'] and report['gates']['source_times_and_count']
    if kind == 'fresh_surface_failure':
        assert not report['gates']['surface_interval_nonregression']
        assert result['status'] == 'rejected_construction_proposal'
    elif kind == 'existing_floor_failure':
        assert not report['gates']['floor_feasibility'] and result['status'] == 'rejected_construction_proposal'
    else:
        assert all(report['gates'].values())
        trials = [record for record in report['surface_intervals']['trial_journal']
            if record['kind'] == 'backtrack' and record['reference_index'] == 1]
        if kind == 'curved_floor':
            assert any(not row['gate_results']['floor_nonregression'] for row in trials)
        else:
            assert any(not row['camera_depth_bound_satisfied'] for row in trials)
            assert report['gates']['camera_endpoint_depth_change']


def test_original_fast_source_clock_and_rate_limit_remain_exact(monkeypatch):
    fps = 240000 / 1001
    data = fixture(monkeypatch, target=.15, fps=fps)
    result = run(monkeypatch, data, missing_image_policy='qualified_surfaces')
    report = result['report']
    assert report['gates']['correction_rate'] and report['maximum_correction_rate_rad_s'] <= 1.5 + 1e-8
    assert abs(report['references'][1]['correction_rad']['left_knee']) <= 1.5 * (2 / fps) + 1e-8
    assert [row['time_s'] for row in report['references']] == [row['time_s'] for row in data['body']['reference']]


@pytest.mark.parametrize('limit', [{'maximum_evaluations': 7}, {'maximum_journal_bytes': 16 << 10}])
def test_budget_preserves_attempt_and_complete_final_interval_audit(monkeypatch, limit):
    result = run(monkeypatch, fixture(monkeypatch), missing_image_policy='qualified_surfaces', surface_interval_settings=limit)
    report = result['report']
    assert result['status'] == 'partial_budget_exhausted' and not report['gates']['complete_bounded_solve']
    assert report['references'][1]['surface_only_eligible']
    interval = report['surface_intervals']
    assert len(interval['references']) == 3 and all(row['after'] is not None for row in interval['references'])
    assert interval['trial_journal'] and interval['journal_bytes'] <= interval['work_limits']['maximum_journal_bytes']
    assert interval['evaluations'] <= interval['work_limits']['maximum_evaluations']
    assert all(not row['accepted'] for row in interval['trial_journal'] if row['kind'] == 'backtrack')


def test_budget_before_first_qp_does_not_claim_an_attempt(monkeypatch):
    result = run(monkeypatch, fixture(monkeypatch), missing_image_policy='qualified_surfaces',
        surface_interval_settings={'maximum_evaluations': 6})
    report = result['report']; row = report['references'][1]
    assert result['status'] == 'partial_budget_exhausted' and not report['gates']['complete_bounded_solve']
    assert row['surface_only_eligible'] and not row['surface_only_attempted']
    assert row['status'] == 'unchanged_no_accepted_surface_step' and not any(row['correction_rad'].values())
    assert report['qp_calls'] == 0 and report['surface_intervals']['trial_journal'] == []
    assert report['surface_intervals']['evaluations'] == 6
    assert all(row['after'] is not None for row in report['surface_intervals']['references'])


@pytest.mark.parametrize('policy', [None, True, 1, '', 'auto', [], {}])
def test_invalid_policy_rejects_before_input_or_geometry(monkeypatch, policy):
    api = importlib.import_module('weft.whole_body_reference_refinement')
    monkeypatch.setattr(api, '_document', denied); monkeypatch.setattr(api, '_make_geometry', denied)
    with pytest.raises(ValueError, match='missing.image policy'):
        api.propose_whole_body_reference_refinement(b'{}', b'{}', missing_image_policy=policy)


@pytest.mark.parametrize('objective', ['image_v1', 'floor_first_v1'])
def test_non_surface_objective_rejects_opt_in_before_geometry(monkeypatch, objective):
    api = importlib.import_module('weft.whole_body_reference_refinement')
    monkeypatch.setattr(api, '_document', denied); monkeypatch.setattr(api, '_make_geometry', denied)
    with pytest.raises(ValueError, match='surface objective'):
        api.propose_whole_body_reference_refinement(b'{}', b'{}', proposal_objective=objective,
            missing_image_policy='qualified_surfaces')
