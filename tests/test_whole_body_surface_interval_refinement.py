"""Real bounded solver with analytical geometry; no native/source/media/model work."""

from copy import deepcopy
from hashlib import sha256
import importlib
import json
import math
from types import SimpleNamespace

import numpy as np
import pytest

from test_multiview_landmark_evidence import denied, local_only


@pytest.fixture
def api():
    return importlib.import_module('weft.whole_body_reference_refinement')


def raw(value):
    return json.dumps(value, allow_nan=False).encode()


def fixture(api, *, target_y=.05, missing=False, weight=2.):
    old = importlib.import_module('test_whole_body_reference_refinement')
    spec, evidence = old.fixture()
    envelopes_fixture = importlib.import_module('test_source_surface_envelopes')
    source, camera, surfaces, annotations = envelopes_fixture.fixture(10)
    source['source'].update(width_px=1200, height_px=600)
    source['image'].update(width_px=1200, height_px=600)
    camera.update(width=1200, height=600, K=[[600., 0., 600.], [0., 600., 300.], [0., 0., 1.]])
    for pose in camera['frames']:
        pose['t'] = [0., 0., 3.]
    surfaces['surfaces'][0].update(height_interval_m=[1.2, 1.3])
    frames = {row['frame'] for row in evidence['frames']}
    annotations['channels'][0]['point_role'] = 'Authored analytical left ankle location, not contact'
    for index, (record, observed, annotated) in enumerate(zip(source['evidence'], source['observations'], annotations['annotations'], strict=True)):
        record['id'] = f"{evidence['source_sha256']}:frame:{index}"
        observed['evidence_ids'] = [record['id']]
        if index not in frames:
            observed.update(visibility='not_sampled', partition='unsampled', evidence_ids=[])
        annotated.update(pixel=[600. + 600.*(-.25)/1.75, 300. - 600.*target_y/1.75], uncertainty_px=1.)
        if missing:
            annotated.update(status='unknown', pixel=None, uncertainty_px=None)
    prepared_inputs = envelopes_fixture.encoded((source, camera, surfaces, annotations))
    prepared = importlib.import_module('weft.source_surface_envelopes').prepare_source_surface_envelopes(**prepared_inputs)
    meta = prepared.describe()
    claim = {'status': 'assumed', 'method': 'Authored analytical world mapping fixed before the fit',
             'uncertainty': 'Model units and pixel/height ranges are authored; no physical contact measured', 'evidence_refs': []}
    binding = {'schema_version': 1, 'profile': 'source-surface-interval-body-binding-v1',
        'source_sha256': evidence['source_sha256'], 'body_sha256': sha256(raw(spec)).hexdigest(), 'envelope_bindings': meta['bindings'],
        'human_entity_ref': meta['channels'][0]['entity_ref'],
        'body_to_world': {'schema_version': 1, 'profile': 'world-alignment-v1',
            'from_frame': {'id': 'body:'+api.canonical_hash(spec), 'length_unit': 'human_model_unit', 'axes': 'x_right_y_away_z_up'},
            'to_frame': meta['world_frame'], 'spatial': {'scale': 1., 'rotation_wxyz': [1., 0., 0., 0.], 'translation': [0., 0., 0.]},
            'clock': {'rate': 1., 'source_origin_s': 0., 'target_origin_s': 0.},
            'claims': {'spatial': deepcopy(claim), 'clock': deepcopy(claim)}},
        'channels': [{'channel_id': meta['channels'][0]['id'], 'body_point': 'left_ankle', 'status': 'assumed',
                      'method': 'Authored core ankle correspondence', 'uncertainty': 'Not anatomy, a heel or contact measurement'}]}
    objective = importlib.import_module('weft.source_surface_interval_construction').prepare_surface_interval_objective(
        prepared, body_bytes=raw(spec), binding_bytes=raw(binding))
    arguments = {'proposal_objective': 'surface_interval_v1', 'surface_interval_objective': objective,
                 'surface_interval_weight': weight, 'joint_mask': ['left_knee']}
    return spec, evidence, arguments, prepared_inputs, binding


class IntervalGeometry:
    support_names = ('foot:0', 'torso:0')
    primitive_names = ('foot', 'torso')

    def __init__(self, api, evidence, mode='ordinary', final=False):
        old = importlib.import_module('test_whole_body_reference_refinement')
        self.base = old.ToyGeometry(evidence).base.copy()
        self.base[10, 0] += .08
        self.api, self.mode, self.final, self.closed = api, mode, final, False
        self.calls = 0

    def evaluate(self, q, root, derivatives=False, *, surface_intervals=False):
        self.calls += 1
        index = self.api.NAMES.index('left_knee')
        point = self.base.copy()
        pj, gj = np.zeros((12, 3, 19)), np.zeros((2, 19))
        pelvis, pelvis_jac = np.asarray(root) + [0., 0., 2.], np.zeros((3, 19))
        # Split the true absolute derivative across relative endpoint and moving pelvis.
        point[:, 1] -= .25*q[index]; pj[:, 1, index] = -.25
        point[10, 1] += q[index]; pj[10, 1, index] += 1.
        pelvis[1] += .25*q[index]; pelvis_jac[1, index] = .25
        gap = .2
        if self.mode == 'curved_floor':
            gap = .0001 - q[index]**2; gj[0, index] = -2*q[index]
        elif self.mode == 'existing_floor_failure':
            gap = -.04 + .1*q[index]; gj[0, index] = .1
        elif self.mode == 'image_guard':
            point[10, 2] += q[index]**2; pj[10, 2, index] = 2*q[index]
        elif self.mode == 'depth_guard':
            point[11, 1] += 10*q[index]; pj[11, 1, index] += 10
        if self.final and self.mode == 'fresh_failure':
            pelvis[1] += .2
        result = {'points': point, 'gaps': np.array([gap, .2]),
                  'point_jac': pj if derivatives else None, 'gap_jac': gj if derivatives else None}
        if surface_intervals:
            result['pelvis_position'] = pelvis
            if derivatives:
                result['pelvis_jac'] = pelvis_jac
        return result

    def close(self):
        self.closed = True


def run(api, monkeypatch, data, *, mode='ordinary', **changes):
    spec, evidence, arguments, _, _ = data
    engines = []
    def create(body):
        assert raw(body) == raw(spec)
        engine = IntervalGeometry(api, evidence, mode, final=bool(engines))
        engines.append(engine)
        return engine
    monkeypatch.setattr(api, '_make_geometry', create)
    result = api.propose_whole_body_reference_refinement(raw(spec), raw(evidence), **(arguments | changes))
    assert len(engines) == 2 and all(engine.closed for engine in engines)
    assert result['report']['geometry_calls'] == sum(engine.calls for engine in engines)
    return result


def test_interval_term_improves_existing_solver_without_changing_source_or_floor(api, monkeypatch, tmp_path, record_property):
    data = fixture(api)
    before = raw(data[0])
    result = run(api, monkeypatch, data)
    report, candidate = result['report'], result['human_candidate']
    interval = report['surface_intervals']; summary = interval['summary']
    assert report['profile'] == api.SURFACE_INTERVAL_METHOD and report['accepted_updates'] > 0
    assert summary['sum_objective_after_m2'] < summary['sum_objective_before_m2']
    assert all(report['gates'].values()) and result['status'] == 'unadmitted_construction_proposal'
    assert candidate['reference'][0] == data[0]['reference'][0]
    assert all(candidate[key] == value for key, value in data[0].items() if key != 'reference')
    assert raw(data[0]) == before
    assert 0. < candidate['reference'][1]['joints']['left_knee'] <= .08
    assert report['objective']['grid_interval_targets'] == report['physics_steps'] == 0
    assert summary['source_frame_slots'] == summary['channel_slots'] == 10
    assert summary['available_reference_terms'] == 3 and summary['unavailable_reference_terms'] == 0
    assert sum(channel['status'] == 'unavailable' for row in interval['source_slots'] for channel in row['channels']) == 7
    assert interval['weight'] == 2. and interval['weight_status'] == 'authored'
    assert interval['optimizer_installation'] == 'installed_in_existing_whole_body_fitter'
    assert interval['inputs']['optimizer_installation'] == 'not_installed_no_existing_fitter_callback'
    assert interval['contact'] == 'unknown' and interval['physical_validation'] == 'unverified'
    assert candidate['whole_body_reference_refinement']['surface_intervals']['summary'] == summary
    assert interval['journal_bytes'] == sum(len(json.dumps(row, allow_nan=False, separators=(',', ':')).encode()) for row in interval['trial_journal'])
    assert [row['sequence'] for row in interval['trial_journal']] == list(range(len(interval['trial_journal'])))
    assert any(row.get('accepted') for row in interval['trial_journal'])
    saved = {'result.json': result, 'human.json': data[0], 'evidence.json': data[1], 'binding.json': data[4]}
    for name, value in saved.items():
        (tmp_path / name).write_bytes(raw(value))
    prepared = data[3]
    for name, key in (('camera.json', 'camera_document'), ('intervals.json', 'surface_interval_document'), ('annotations.json', 'annotation_document')):
        (tmp_path / name).write_bytes(prepared[key])
    (tmp_path / 'observations.json').write_bytes(next(iter(prepared['observation_documents'].values())))
    record_property('analytical_report_fixture', str(tmp_path))
    record_property('result_sha256', sha256(raw(result)).hexdigest())
    record_property('geometry_calls_analytic', report['geometry_calls'])
    record_property('interval_evaluations', interval['evaluations'])
    record_property('objective_before_m2', summary['sum_objective_before_m2'])
    record_property('objective_after_m2', summary['sum_objective_after_m2'])


@pytest.mark.parametrize('mode', ['curved_floor', 'existing_floor_failure', 'image_guard', 'depth_guard'])
def test_interval_descent_cannot_override_existing_nonlinear_gates(api, monkeypatch, mode):
    data = fixture(api, target_y=.2 if mode == 'depth_guard' else .05)
    result = run(api, monkeypatch, data, mode=mode)
    report = result['report']; trials = [row for row in report['surface_intervals']['trial_journal'] if row['kind'] == 'backtrack']
    assert all(row['no_worse_floor'] for row in report['geometry_checks'])
    assert report['gates']['source_endpoint_image_nonregression'] and report['gates']['endpoint_depth_change']
    assert report['gates']['surface_interval_nonregression']
    if mode == 'existing_floor_failure':
        assert result['status'] == 'rejected_construction_proposal'
        assert report['gates']['floor_feasibility'] is False
        assert report['floor_summary']['after_maximum_penetration_model_units'] > .025
    elif mode == 'curved_floor':
        assert any(row['gate_results']['floor_nonregression'] is False for row in trials)
        assert result['human_candidate']['reference'][1]['joints']['left_knee'] <= math.sqrt(.0001+1e-8)
    elif mode == 'image_guard':
        assert any(row['gate_results']['source_image_nonregression'] is False for row in trials)
        assert report['accepted_updates'] == 0
    else:
        assert max(row['maximum_depth_change_model_units'] for row in report['geometry_checks']) <= .08 + 1e-8


def test_fresh_final_geometry_can_reject_cached_surface_success(api, monkeypatch):
    result = run(api, monkeypatch, fixture(api), mode='fresh_failure')
    report = result['report']
    assert report['accepted_updates'] > 0
    assert report['gates']['surface_interval_nonregression'] is False
    assert result['status'] == 'rejected_construction_proposal'
    first = report['surface_intervals']['references'][0]
    assert first['before']['objective_m2'] != first['after']['objective_m2']
    assert report['gates']['first_reference_frozen'] is True


def test_missing_intervals_remain_null_with_complete_denominators(api, monkeypatch):
    result = run(api, monkeypatch, fixture(api, missing=True))
    report = result['report']; interval = report['surface_intervals']
    assert interval['summary']['unavailable_reference_terms'] == 3
    assert interval['summary']['sum_objective_before_m2'] is interval['summary']['sum_objective_after_m2'] is None
    assert report['accepted_updates'] == 0
    assert len(interval['source_slots']) == 10
    assert all(row['before']['objective_m2'] is row['after']['objective_m2'] is None for row in interval['references'])


@pytest.mark.parametrize('change', [
    {'surface_interval_weight': None}, {'surface_interval_weight': 0}, {'surface_interval_weight': True},
    {'surface_interval_weight': 1001}, {'surface_interval_weight': float('inf')},
    {'surface_interval_settings': {'maximum_evaluations': 5}},
    {'surface_interval_settings': {'maximum_evaluations': 20001}},
    {'surface_interval_settings': {'maximum_journal_bytes': (4<<20)+1}},
    {'surface_interval_settings': {'maximum_journal_bytes': 1}},
    {'surface_interval_settings': {'relax_floor': True}},
    {'proposal_objective': 'image_v1'}, {'proposal_objective': 'floor_first_v1'},
])
def test_explicit_mode_weight_and_tighter_limits_fail_before_geometry(api, monkeypatch, change):
    data = fixture(api)
    monkeypatch.setattr(api, '_make_geometry', denied)
    with pytest.raises(ValueError):
        api.propose_whole_body_reference_refinement(raw(data[0]), raw(data[1]), **(data[2] | change))


def test_exact_original_body_source_and_clock_binding_is_required(api, monkeypatch):
    data = fixture(api)
    monkeypatch.setattr(api, '_make_geometry', denied)
    with pytest.raises(ValueError, match='exact original body'):
        api.propose_whole_body_reference_refinement(raw(data[0])+b' ', raw(data[1]), **data[2])
    evidence = deepcopy(data[1]); evidence['frame_count'] = 11
    with pytest.raises(ValueError, match='complete original pose source'):
        api.propose_whole_body_reference_refinement(raw(data[0]), raw(evidence), **data[2])


def test_evaluation_limit_keeps_attempted_trial_and_complete_final_audit(api, monkeypatch):
    result = run(api, monkeypatch, fixture(api), surface_interval_settings={'maximum_evaluations': 7})
    report = result['report']; interval = report['surface_intervals']
    assert result['status'] == 'partial_budget_exhausted'
    assert report['gates']['complete_bounded_solve'] is False
    assert interval['evaluations'] == 7
    assert len(interval['references']) == 3 and all('after' in row for row in interval['references'])
    incomplete = [row for row in interval['trial_journal'] if row['outcome'] == 'evaluation_incomplete']
    assert incomplete and incomplete[-1]['accepted'] is False
    assert incomplete[-1]['combined_cost_after'] is None
    assert len(incomplete[-1]['trial_correction_rad']) == 19


def test_journal_limit_stops_before_losing_an_alternative_and_still_audits(api, monkeypatch):
    result = run(api, monkeypatch, fixture(api), surface_interval_settings={'maximum_journal_bytes': 16<<10})
    report = result['report']; interval = report['surface_intervals']
    assert result['status'] == 'partial_budget_exhausted'
    assert 'journal' in report['budget_reason']
    assert len(interval['trial_journal']) == 1 and interval['trial_journal'][0]['kind'] == 'qp'
    assert len(interval['references']) == 3 and interval['journal_bytes'] <= 16<<10
    assert report['accepted_updates'] == 0


def test_nonfinite_solver_direction_is_preserved_as_tagged_failed_alternative(api, monkeypatch):
    monkeypatch.setattr(api, 'solve_feasible_qp', lambda *args, **kwargs: {'status': 'optimal', 'step': [float('nan')], 'iterations': 1})
    result = run(api, monkeypatch, fixture(api))
    ledger = result['report']['surface_intervals']['trial_journal']
    assert ledger and all(row['outcome'] == 'rejected_invalid_direction' for row in ledger)
    assert ledger[0]['step_rad'] == [{'nonfinite': 'nan'}]
    assert result['report']['accepted_updates'] == 0
    json.dumps(result, allow_nan=False)


def test_default_and_explicit_original_objective_remain_identical(api, monkeypatch):
    data = fixture(api)
    options = {'surface_interval_objective': None, 'surface_interval_weight': None,
               'surface_interval_settings': None, 'proposal_objective': 'image_v1'}
    first = run(api, monkeypatch, data, **options)
    engines = []
    def make(_):
        engine = IntervalGeometry(api, data[1]); engines.append(engine); return engine
    monkeypatch.setattr(api, '_make_geometry', make)
    second = api.propose_whole_body_reference_refinement(raw(data[0]), raw(data[1]), joint_mask=['left_knee'])
    first['report'].pop('seconds'); second['report'].pop('seconds')
    assert first == second
    assert 'surface_intervals' not in first['report'] and 'objective' not in first['report']
    assert len(engines) == 2 and all(item.closed for item in engines)


def test_opt_in_native_geometry_wrapper_captures_current_pelvis_without_a_second_forward(api):
    geometry = object.__new__(api._NativeGeometry)
    model = SimpleNamespace(nv=19, geom_size=np.array([[0., 0., 0.], [.05, .1, 0.]]), geom_bodyid=[0, 1])
    data = SimpleNamespace(qpos=np.zeros(22), geom_xpos=np.array([[0., 0., 0.], [0., 0., .5]]),
        geom_xmat=np.tile(np.eye(3).reshape(9), (2, 1)), site_xpos=np.array([[0., 0., 0.], [.2, .3, .4]]))
    geometry.env = SimpleNamespace(model=model, mobility=SimpleNamespace(q=np.array([19, 20, 21])),
        _v_array=np.arange(19), _site_ids={'pelvis': 1})
    geometry.data, geometry.floor, geometry.supports = data, 0, [('body', 1, 0.)]
    calls = {'kinematics': 0, 'support_jacobian': 0, 'pelvis_jacobian': 0}

    def kinematics(env, current, angles, derivatives):
        calls['kinematics'] += 1
        assert current is data and env is geometry.env
        return np.full((12, 3), .1), np.full((36, 19), 2.) if derivatives else None

    def support_jac(model, current, target, *rest):
        calls['support_jacobian'] += 1; target[:] = 3.

    def pelvis_jac(model, current, target, rotation, site):
        assert current is data and site == 1
        calls['pelvis_jacobian'] += 1; target[:] = 7.

    geometry.kinematics = kinematics
    geometry.mujoco = SimpleNamespace(mj_jac=support_jac, mj_jacSite=pelvis_jac)
    angles, root = np.zeros(19), np.array([.05, .1, .15])
    default = geometry.evaluate(angles, root, True)
    assert 'pelvis_position' not in default and 'pelvis_jac' not in default
    assert calls == {'kinematics': 1, 'support_jacobian': 1, 'pelvis_jacobian': 0}
    enabled = geometry.evaluate(angles, root, True, surface_intervals=True)
    np.testing.assert_array_equal(enabled['pelvis_position'], [.2, .3, .4])
    np.testing.assert_array_equal(enabled['pelvis_jac'], np.full((3, 19), 7.))
    np.testing.assert_array_equal(enabled['point_jac'], np.full((12, 3, 19), 2.))
    np.testing.assert_array_equal(enabled['gap_jac'], np.full((1, 19), 3.))
    np.testing.assert_array_equal(data.qpos[geometry.env.mobility.q], root)
    assert calls == {'kinematics': 2, 'support_jacobian': 2, 'pelvis_jacobian': 1}
