"""Fixed-source camera proposals with analytical geometry; no native or source study."""

import builtins
from copy import deepcopy
from hashlib import sha256
import importlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import time
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from test_multiview_landmark_evidence import denied, local_only


def raw(value):
    return json.dumps(value, allow_nan=False).encode()


@pytest.fixture
def api():
    return importlib.import_module('weft.camera_body_fit')


def base_points():
    return np.array([[-.2, 0., .5], [.2, 0., .5], [-.3, 0., .25], [.3, 0., .25],
                     [-.4, 0., .1], [.4, 0., .1], [-.1, 0., 0.], [.1, 0., 0.],
                     [-.1, 0., -.3], [.1, 0., -.3], [-.1, 0., -.55], [.1, 0., -.55]])


def pixels(camera, frame, points):
    extrinsic = camera['frames'][frame]
    local = points @ np.array(extrinsic['R']).T + extrinsic['t']
    normalized = local[:, :2] / local[:, 2:]
    distorted = normalized * (1 + camera['radial_k1'] * np.sum(normalized ** 2, axis=1))[:, None]
    return (np.column_stack((distorted, np.ones(len(points)))) @ np.array(camera['K']).T)[:, :2]


def fixture(api, *, fps=30., radial=0., target=.05, source_change=None, annotation='known', mapping_change=None):
    foot = importlib.import_module('weft.multiview_foot_reference')
    source_fixture = importlib.import_module('test_multiview_foot_reference')
    value = source_fixture._fixture(foot, count=10, fps=fps, sampled=(0, 2, 4))
    body, _ = importlib.import_module('test_whole_body_reference_refinement').fixture()
    original = value['body']
    body.update(body_frame=deepcopy(original['body_frame']), camera_reference=deepcopy(original['camera_reference']),
                root_reference=deepcopy(original['root_reference']), reference_evidence=deepcopy(original['reference_evidence']))
    body['mobility']['base_height'] = 1.
    for pose, source in zip(body['reference'], body['reference_evidence'], strict=True):
        pose['time_s'] = source['time_s']
    value['body'] = body
    world = base_points() + [1., 2., 1.]
    world[10, 1] += target
    for view, camera in enumerate(value['cameras']):
        camera['radial_k1'] = radial
        for frame in camera['frames']:
            frame.update(R=[[1., 0., 0.], [0., -1., 0.], [0., 0., -1.]], t=[-float(view), 0., 3.])
        for pose in value['evidence'][view]['frames']:
            uv = pixels(camera, pose['frame'], world)
            for index, name in enumerate(api.JOINTS):
                pose['landmarks'][api.JOINTS[name]].update(x=float(uv[index, 0] / camera['width']), y=float(uv[index, 1] / camera['height']))
    if source_change is not None:
        source_change(value)
    source_fixture._pin(value, foot)
    body, evidence, camera = value['body'], value['evidence'][0], value['cameras'][0]
    data = list(importlib.import_module('test_source_surface_envelopes').fixture(10))
    observation, _, surfaces, annotations = data
    data[1] = deepcopy(camera)
    observation['source'].update(width_px=camera['width'], height_px=camera['height'], fps=fps)
    observation['image'].update(width_px=camera['width'], height_px=camera['height'])
    surfaces['surfaces'][0].update(height_interval_m=[.449, .451])
    if annotation == 'unknown_height':
        surfaces['surfaces'][0].update(height_interval_m=None, quality='unknown')
    body_frames = {row['frame'] for row in evidence['frames']}
    for index, (record, observed, annotated) in enumerate(zip(observation['evidence'], observation['observations'], annotations['annotations'], strict=True)):
        record.update(id=f"{body['source_sha256']}:frame:{index}", time_s=index / fps)
        observed.update(time_s=index / fps, evidence_ids=[record['id']])
        annotated.update(time_s=index / fps, pixel=pixels(camera, index, world[[10]])[0].tolist(), uncertainty_px=.01)
        if index not in body_frames:
            observed.update(visibility='not_sampled', partition='unsampled', evidence_ids=[])
        if annotation == 'missing':
            annotated.update(status='unknown', pixel=None, uncertainty_px=None)
        if annotation == 'withheld' and index == 2:
            observed['partition'] = 'audit'
        if annotation == 'mismatched' and index == 2:
            record['id'] += ':unrelated'; observed['evidence_ids'] = [record['id']]
    encoded = importlib.import_module('test_source_surface_envelopes').encoded(tuple(data))
    envelopes = importlib.import_module('weft.source_surface_envelopes').prepare_source_surface_envelopes(**encoded)
    metadata = envelopes.describe()
    claim = {'status': 'assumed', 'method': 'Fixed authored analytical body-to-camera-world translation',
             'uncertainty': 'Supplied coordinates, not measured anatomy or contact', 'evidence_refs': []}
    binding = {'schema_version': 1, 'profile': 'source-surface-interval-body-binding-v1',
        'body_sha256': sha256(raw(body)).hexdigest(), 'source_sha256': body['source_sha256'],
        'envelope_bindings': metadata['bindings'], 'human_entity_ref': metadata['channels'][0]['entity_ref'],
        'body_to_world': {'schema_version': 1, 'profile': 'world-alignment-v1',
            'from_frame': {'id': 'body:' + api.canonical_hash(body), 'length_unit': 'human_model_unit', 'axes': 'x_right_y_away_z_up'},
            'to_frame': metadata['world_frame'], 'spatial': {'scale': 1., 'rotation_wxyz': [1., 0., 0., 0.], 'translation': [1., 2., 0.]},
            'clock': {'rate': 1., 'source_origin_s': 0., 'target_origin_s': 0.},
            'claims': {'spatial': deepcopy(claim), 'clock': deepcopy(claim)}},
        'channels': [{'channel_id': metadata['channels'][0]['id'], 'body_point': 'left_ankle', 'status': 'assumed',
                      'method': 'Authored core ankle correspondence', 'uncertainty': 'Not a heel or contact observation'}]}
    if mapping_change is not None:
        mapping_change(binding)
    objective_api = importlib.import_module('weft.source_surface_interval_construction')
    objective = objective_api.prepare_surface_interval_objective(envelopes, body_bytes=raw(body), binding_bytes=raw(binding), include_image_admission=True)
    return {'body': body, 'evidence': evidence, 'camera': camera, 'envelopes': envelopes, 'objective': objective,
            'binding': binding, 'envelope_inputs': encoded, 'views': value}


def prepare(api, data, **changes):
    return api.prepare_camera_body_fit(raw(data['body']), raw(data['evidence']), raw(data['camera']),
        **({'image_scale': .01, 'surface_objective': data['objective']} | changes))


class Geometry:
    support_names, primitive_names = ('foot:0', 'torso:0'), ('foot', 'torso')

    def __init__(self, api, mode='ordinary', final=False):
        self.names, self.mode, self.final, self.closed, self.calls = api.SPATIAL_JOINT_NAMES, mode, final, False, 0

    def evaluate(self, angles, root, derivatives=False, *, surface_intervals=False):
        self.calls += 1
        column = self.names.index('left_knee'); q = angles[column]
        point, jac = base_points(), np.zeros((12, 3, 19))
        pelvis, pelvis_jac = np.array(root) + [0., 0., 1.], np.zeros((3, 19))
        point[:, 1] -= .25 * q; jac[:, 1, column] = -.25
        pelvis[1] += .25 * q; pelvis_jac[1, column] = .25
        point[10, 1] += q; jac[10, 1, column] += 1.
        gaps, gap_jac = np.array([.2, .2]), np.zeros((2, 19))
        if self.mode == 'curved_floor':
            gaps[0] = .002 - q * q; gap_jac[0, column] = -2 * q
        elif self.mode == 'existing_floor_failure':
            gaps[0] = -.04 + .1 * q; gap_jac[0, column] = .1
        elif self.mode == 'camera_depth':
            point[10, 2] += 40 * q * q; jac[10, 2, column] = 80 * q
        if self.final and self.mode == 'fresh_invalid':
            pelvis[2] += 4.
        result = {'points': point, 'point_jac': jac if derivatives else None, 'gaps': gaps,
                  'gap_jac': gap_jac if derivatives else None}
        if surface_intervals:
            result['pelvis_position'] = pelvis
            if derivatives:
                result['pelvis_jac'] = pelvis_jac
        return result

    def close(self):
        self.closed = True


def run(api, data, monkeypatch, *, mode='ordinary', **changes):
    fitter = importlib.import_module('weft.whole_body_reference_refinement')
    engines = []
    def create(body):
        assert raw(body) == raw(data['body'])
        engine = Geometry(api, mode, bool(engines)); engines.append(engine); return engine
    monkeypatch.setattr(fitter, '_make_geometry', create)
    arguments = {'proposal_objective': 'camera_surface_interval_v1', 'camera_document': raw(data['camera']),
                 'camera_image_scale': .01, 'surface_interval_objective': data['objective'],
                 'surface_interval_weight': 2., 'joint_mask': ['left_knee']}
    result = fitter.propose_whole_body_reference_refinement(raw(data['body']), raw(data['evidence']), **(arguments | changes))
    assert len(engines) == 2 and all(engine.closed for engine in engines)
    assert result['report']['geometry_calls'] == sum(engine.calls for engine in engines)
    return result


def save_fixture(path, data, result):
    path.mkdir(exist_ok=False)
    (path / 'inputs').mkdir()
    files = {'inputs/human.json': raw(data['body']), 'inputs/evidence-derived.json': raw(data['evidence']),
        'inputs/evidence-original.json': raw(data['evidence']), 'inputs/source-camera.json': raw(data['camera']),
        'inputs/surface-camera.json': raw(data['camera']), 'inputs/surface-body-binding.json': raw(data['binding']),
        'inputs/surface-intervals.json': data['envelope_inputs']['surface_interval_document'],
        'inputs/surface-annotations.json': data['envelope_inputs']['annotation_document'],
        'angle-proposal.json': raw(result['angle_proposal']), 'report.json': raw(result['report']), 'result.json': raw(result)}
    for index, value in enumerate(data['envelope_inputs']['observation_documents'].values()):
        files[f'inputs/surface-observation-{index:02d}.json'] = value
    for name, value in files.items():
        with (path / name).open('xb') as handle:
            handle.write(value)
    manifest = {'profile': 'authored-camera-fit-analytic-fixture-v1', 'native_calls': 0,
                'members': {name: {'sha256': sha256(value).hexdigest(), 'bytes': len(value)} for name, value in files.items()}}
    (path / 'manifest.json').write_bytes(raw(manifest))


def test_existing_qp_improves_without_mutating_source_or_admitting_body(api, monkeypatch, record_property):
    data = fixture(api, radial=.00047528394168603385)
    original = raw([data['body'], data['evidence'], data['camera']])
    started = time.monotonic(); result = run(api, data, monkeypatch); elapsed = time.monotonic() - started
    report, overlay = result['report'], result['angle_proposal']
    assert result['human_candidate'] is None and overlay['profile'] == api.OVERLAY_PROFILE
    assert overlay['installation'] == 'unsupported_angle_overlay_not_a_human_spec'
    assert all(report['gates'].values()) and report['accepted_updates'] > 0
    assert report['image_summary']['mean_error_after_px'] < report['image_summary']['mean_error_before_px']
    assert report['surface_intervals']['summary']['sum_objective_after_m2'] < report['surface_intervals']['summary']['sum_objective_before_m2']
    assert overlay['reference_candidate'][0] == data['body']['reference'][0]
    assert all(0. <= row['joints']['left_knee'] <= .15 for row in overlay['reference_candidate'])
    assert raw([data['body'], data['evidence'], data['camera']]) == original
    assert report['candidate_sha256'] == api.canonical_hash(overlay)
    assert overlay['input_bindings']['body_raw_sha256'] == sha256(raw(data['body'])).hexdigest()
    assert overlay['input_bindings']['body_canonical_sha256'] == api.canonical_hash(data['body'])
    assert report['physics_steps'] == report['camera_fit']['grid_camera_projection_calls'] == 0
    assert len(report['camera_fit']['source_slots']) == 10
    assert sum(row['landmarks_used'] for row in report['camera_fit']['source_slots']) == 3
    assert report['camera_fit']['image_nonregression_tolerance_px'] == 1e-8
    assert report['surface_intervals']['journal_bytes'] == sum(len(json.dumps(row, allow_nan=False, separators=(',', ':')).encode()) for row in report['surface_intervals']['trial_journal'])
    record_property('analytic_fit_seconds', elapsed)
    record_property('geometry_calls', report['geometry_calls'])
    record_property('projection_calls', report['camera_fit']['projection_calls'])
    record_property('result_bytes', len(raw(result)))
    if os.environ.get('WEFT_CAMERA_FIT_FIXTURE_OUT'):
        save_fixture(Path(os.environ['WEFT_CAMERA_FIT_FIXTURE_OUT']), data, result)


@pytest.mark.parametrize('annotation', ['unknown_height', 'missing'])
def test_image_admission_is_independent_of_unavailable_interval_geometry(api, monkeypatch, annotation):
    data = fixture(api, annotation=annotation)
    assert data['objective'].frame(0)['channels'][0]['status'] == 'unavailable'
    assert data['objective'].image_admission(0)['status'] == 'construction_image_source_ready'
    result = run(api, data, monkeypatch)
    assert result['report']['image_summary']['confident_source_endpoint_pairs'] == 36
    assert result['report']['accepted_updates'] > 0
    assert result['report']['surface_intervals']['summary']['available_reference_terms'] == 0
    assert result['report']['surface_intervals']['summary']['sum_objective_before_m2'] is None


@pytest.mark.parametrize('annotation,reason', [('withheld', 'not_construction_source_evidence'), ('mismatched', 'body_reference_not_in_source_entity_evidence')])
def test_excluded_or_mismatched_landmark_payload_is_not_used(api, monkeypatch, annotation, reason):
    def poison(value):
        value['evidence'][0]['frames'][1]['landmarks'] = 'withheld payload must not be parsed as landmarks'
    data = fixture(api, annotation=annotation, source_change=poison)
    prepared = prepare(api, data)
    assert data['objective'].image_admission(2)['reason'] == reason
    assert not prepared.rows()[1]['weights'].any()
    assert prepared.rows()[1]['points'][0]['landmark_source'] == 'not_read_unqualified_source'
    result = run(api, data, monkeypatch)
    assert len(result['angle_proposal']['reference_candidate']) == 3
    assert result['angle_proposal']['reference_candidate'][1] == data['body']['reference'][1]
    assert result['report']['source_counts']['body_reference_knots'] == 3
    assert result['report']['image_summary']['confident_source_endpoint_pairs'] == 24


def test_opt_in_admission_preserves_every_legacy_serialized_row_and_metadata(api, monkeypatch):
    data = fixture(api)
    objective_api = importlib.import_module('weft.source_surface_interval_construction')
    before = objective_api.prepare_surface_interval_objective(data['envelopes'], body_bytes=raw(data['body']), binding_bytes=raw(data['binding']))
    explicit = objective_api.prepare_surface_interval_objective(data['envelopes'], body_bytes=raw(data['body']), binding_bytes=raw(data['binding']), include_image_admission=False)
    assert before._metadata == explicit._metadata and before._rows == explicit._rows == data['objective']._rows
    metadata = data['objective'].describe(); metadata.pop('source_image_admission')
    assert metadata == before.describe()
    with pytest.raises(ValueError, match='not explicitly prepared'):
        before.image_admission(0)
    monkeypatch.setattr(objective_api, '_sources', denied)
    for index in range(10):
        assert data['objective'].image_admission(index)['source_frame'] == index
    value = data['objective'].image_admission(0); value['human_entity_ref']['object_id'] = 'not-original'
    assert data['objective'].image_admission(0)['human_entity_ref']['object_id'] == 'person'


def test_image_admission_does_not_require_world_alignment(api):
    def unknown(binding):
        binding['body_to_world']['claims']['spatial']['status'] = 'unknown'
        binding['body_to_world']['spatial'] = None
    data = fixture(api, mapping_change=unknown)
    assert data['objective'].frame(0)['channels'][0]['status'] == 'unavailable'
    assert data['objective'].image_admission(0)['status'] == 'construction_image_source_ready'
    prepared = prepare(api, data)
    assert prepared.rows()[0]['weights'].sum() > 0
    with pytest.raises(ValueError, match='qualified explicit'):
        prepared.validate_intervals(data['objective'].describe())


def test_decoded_only_slots_remain_without_invented_camera_poses(api, monkeypatch):
    def extra(value):
        value['evidence'][0]['decoded_frames'] = 11
    data = fixture(api, source_change=extra)
    prepared = prepare(api, data); metadata = prepared.describe()
    assert metadata['complete_source_frame_slots'] == 11
    tail = metadata['source_slots'][-1]
    assert tail['source_frame'] == 10 and tail['camera_status'] == 'unavailable_original_decoded_slot'
    assert tail['camera_time_s'] is None and tail['reference_index'] is None and not tail['landmarks_used']
    fitter = importlib.import_module('weft.whole_body_reference_refinement')
    monkeypatch.setattr(fitter, '_make_geometry', denied)
    with pytest.raises(ValueError, match='complete original pose source differ'):
        fitter.propose_whole_body_reference_refinement(raw(data['body']), raw(data['evidence']),
            proposal_objective='camera_surface_interval_v1', camera_document=raw(data['camera']), camera_image_scale=.01,
            surface_interval_objective=data['objective'], surface_interval_weight=2.)


def test_known_invalid_original_camera_geometry_abstains_and_closes(api, monkeypatch):
    data = fixture(api)
    fitter = importlib.import_module('weft.whole_body_reference_refinement')
    engine = Geometry(api, 'fresh_invalid', True)
    monkeypatch.setattr(fitter, '_make_geometry', lambda _: engine)
    with pytest.raises(ValueError, match='known original camera endpoints'):
        fitter.propose_whole_body_reference_refinement(raw(data['body']), raw(data['evidence']),
            proposal_objective='camera_surface_interval_v1', camera_document=raw(data['camera']), camera_image_scale=.01,
            surface_interval_objective=data['objective'], surface_interval_weight=2.)
    assert engine.closed


def test_input_and_opt_in_query_bounds_precede_expensive_work(api, monkeypatch):
    fitter = importlib.import_module('weft.whole_body_reference_refinement')
    monkeypatch.setattr(fitter, '_document', denied)
    with pytest.raises(ValueError, match='eight MiB'):
        api.prepare_camera_body_fit(b'x' * (api.MAX_BYTES + 1), b'{}', b'{}', image_scale=.01)


def test_fractional_source_clock_decoder_slots_and_body_period_stay_distinct(api):
    def decoder(value):
        for row in value['evidence'][0]['frames']:
            row['decoder_timestamp_s'] += .003
    data = fixture(api, fps=30000 / 1001, source_change=decoder)
    prepared = prepare(api, data); meta = prepared.describe()
    assert prepared.body['dt'] == 1 / 30
    assert meta['coupling']['clock']['control_dt_s'] == 1001 / 30000
    assert [row['time_s'] for row in prepared.rows()] == [0., 2 * 1001 / 30000, 4 * 1001 / 30000]
    assert meta['source_slots'][2]['decoder_timestamp_s'] == data['evidence']['frames'][1]['decoder_timestamp_s']
    assert meta['source_slots'][1]['status'] == 'unsampled'
    assert meta['source_slots'][1]['image_source_admission'] is None
    assert meta['input_bindings']['pose_evidence']['scope'] == 'complete_supplied_pose_document'


@pytest.mark.parametrize('case', ['body_pin', 'camera_pin', 'source_clock', 'ground', 'root', 'foreign_evidence'])
def test_original_binding_mismatch_rejects_without_geometry(api, case):
    data = fixture(api)
    if case == 'body_pin':
        data['body']['reference'][1]['joints']['left_knee'] = .01
    elif case == 'camera_pin':
        data['camera']['K'][0][0] += 1.
    elif case == 'source_clock':
        data['evidence']['frames'][1]['time_s'] += .01
    elif case == 'ground':
        data['body']['mobility']['base_height'] += .1
    elif case == 'root':
        data['body']['root_reference'][1]['translation'][1] += .1
    else:
        data['evidence']['source_sha256'] = 'f' * 64
    with pytest.raises(ValueError):
        prepare(api, data, surface_objective=None)


@pytest.mark.parametrize('scale', [None, True, 0., -1., 1e-7, 1.001, float('nan'), float('inf')])
def test_authored_image_loss_scale_has_fixed_finite_bounds(api, scale):
    data = fixture(api)
    with pytest.raises(ValueError, match='loss scale'):
        prepare(api, data, image_scale=scale)


@pytest.mark.parametrize('change', [lambda mapping: mapping['spatial'].update(scale=2.),
    lambda mapping: mapping['spatial'].update(translation=[1., 2., .1]),
    lambda mapping: mapping['clock'].update(rate=2.)])
def test_no_implicit_body_scale_floor_or_clock_alignment(api, monkeypatch, change):
    data = fixture(api, mapping_change=lambda binding: change(binding['body_to_world']))
    fitter = importlib.import_module('weft.whole_body_reference_refinement')
    monkeypatch.setattr(fitter, '_make_geometry', denied)
    with pytest.raises(ValueError, match='fixed camera-world|qualified explicit'):
        fitter.propose_whole_body_reference_refinement(raw(data['body']), raw(data['evidence']),
            proposal_objective='camera_surface_interval_v1', camera_document=raw(data['camera']), camera_image_scale=.01,
            surface_interval_objective=data['objective'], surface_interval_weight=2.)


def test_perspective_radial_derivative_includes_absolute_pelvis(api):
    def camera_change(value):
        for camera in value['cameras']:
            camera['K'][0][1] = 7.
            for row in camera['frames']:
                row['R'] = [[.8, .6, 0.], [.6, -.8, 0.], [0., 0., -1.]]
    data = fixture(api, radial=.07, source_change=camera_change)
    prepared = prepare(api, data)
    engine = Geometry(api); q = np.zeros(19); q[api.SPATIAL_JOINT_NAMES.index('left_knee')] = .03
    state = engine.evaluate(q, [0., 0., 0.], True, surface_intervals=True)
    analytical = prepared.project(0, state, derivatives=True)
    numerical = np.zeros((12, 2, 19))
    for column in range(19):
        step = np.zeros(19); step[column] = 1e-6
        a = prepared.project(0, engine.evaluate(q + step, [0., 0., 0.], surface_intervals=True))['pixels']
        b = prepared.project(0, engine.evaluate(q - step, [0., 0., 0.], surface_intervals=True))['pixels']
        numerical[:, :, column] = (a - b) / 2e-6
    np.testing.assert_allclose(analytical['pixel_jac'], numerical, atol=3e-8, rtol=1e-8)
    np.testing.assert_allclose(analytical['image_jac'], numerical * .01, atol=3e-10, rtol=1e-8)
    column = api.SPATIAL_JOINT_NAMES.index('left_knee')
    assert np.max(abs(analytical['pixel_jac'][np.arange(12) != 10, :, column])) < 1e-15
    assert np.linalg.norm(analytical['pixel_jac'][10, :, column]) > 1.


@pytest.mark.parametrize('mode', ['curved_floor', 'existing_floor_failure', 'camera_depth'])
def test_new_camera_objective_cannot_override_sampled_guards(api, monkeypatch, mode):
    data = fixture(api)
    result = run(api, data, monkeypatch, mode=mode)
    report = result['report']; trials = [row for row in report['surface_intervals']['trial_journal'] if row['kind'] == 'backtrack']
    assert report['gates']['pointwise_floor_nonregression'] and report['gates']['source_endpoint_image_nonregression']
    assert report['gates']['camera_endpoint_depth_change'] and report['gates']['endpoint_depth_change']
    if mode == 'curved_floor':
        assert any(row['gate_results']['floor_nonregression'] is False for row in trials)
    elif mode == 'camera_depth':
        assert any(row['camera_depth_bound_satisfied'] is False for row in trials)
    else:
        assert report['gates']['floor_feasibility'] is False and result['status'] == 'rejected_construction_proposal'


def test_fresh_final_camera_failure_is_retained_not_hidden_as_zero_error(api, monkeypatch):
    result = run(api, fixture(api), monkeypatch, mode='fresh_invalid')
    report = result['report']
    assert result['status'] == 'rejected_construction_proposal'
    assert report['gates']['camera_projection_domain'] is False
    assert report['gates']['source_endpoint_image_nonregression'] is False
    assert report['image_summary']['invalid_final_known_endpoints'] == 36
    assert report['image_summary']['mean_error_after_px'] is None
    assert all(point['error_after_px'] is None and point['predicted_after_px'] is None
               for row in report['references'] for point in row['points'])


def test_pixel_gate_does_not_get_weaker_with_authored_objective_scale(api, monkeypatch):
    data = fixture(api)
    result = run(api, data, monkeypatch, camera_image_scale=1e-6)
    report = result['report']
    assert report['gates']['source_endpoint_image_nonregression']
    assert all(point['error_after_px'] <= point['error_before_px'] + 1e-8
               for row in report['references'] for point in row['points'] if point['weight'])
    assert report['camera_fit']['image_loss_scale_per_pixel'] == 1e-6


def test_expired_preparation_or_extra_admission_cache_fails_closed(api, monkeypatch):
    data = fixture(api)
    times = iter([0., 6.])
    with monkeypatch.context() as patch:
        patch.setattr(api.time, 'monotonic', lambda: next(times))
        with pytest.raises(TimeoutError, match='preparation deadline'):
            prepare(api, data)
    objective_api = importlib.import_module('weft.source_surface_interval_construction')
    monkeypatch.setattr(objective_api, 'MAX_IMAGE_ADMISSION_BYTES', 1)
    with pytest.raises(ValueError, match='image admission cache'):
        objective_api.prepare_surface_interval_objective(data['envelopes'], body_bytes=raw(data['body']),
            binding_bytes=raw(data['binding']), include_image_admission=True)


def test_interval_budget_keeps_attempted_trial_and_reserves_final_audit(api, monkeypatch):
    result = run(api, fixture(api), monkeypatch, surface_interval_settings={'maximum_evaluations': 8})
    report = result['report']
    assert result['status'] == 'partial_budget_exhausted'
    assert len(report['surface_intervals']['references']) == 3
    assert all(row['after'] is not None for row in report['surface_intervals']['references'])
    assert report['gates']['complete_bounded_solve'] is False


def test_camera_arguments_never_change_legacy_mode_admission(api, monkeypatch):
    fitter = importlib.import_module('weft.whole_body_reference_refinement')
    body, evidence = importlib.import_module('test_whole_body_reference_refinement').fixture()
    monkeypatch.setattr(fitter, '_make_geometry', denied)
    with pytest.raises(ValueError, match='explicit camera_surface'):
        fitter.propose_whole_body_reference_refinement(raw(body), raw(evidence), camera_document=b'{}')
    data = fixture(api)
    with pytest.raises(ValueError, match='supplied-camera'):
        fitter.propose_whole_body_reference_refinement(raw(data['body']), raw(data['evidence']))


def test_actual_shared_kinematics_producer_composes_camera_derivatives(api, monkeypatch):
    data = fixture(api); prepared = prepare(api, data)
    base = np.vstack((base_points() + [0., 0., 1.], [0., 0., 1.]))
    derivatives = np.zeros((13, 3, 19)); derivatives[12, :, 0] = [.4, -.3, .2]
    derivatives[:12, :, 0] = [1., .2, -.1]
    fake = ModuleType('mujoco'); calls = {'forward': 0, 'jacobian': 0}
    def forward(model, state):
        state.site_xpos[:] = base + np.einsum('ijk,k->ij', derivatives, state.qpos); calls['forward'] += 1
    def jac_site(model, state, destination, rotation, site):
        destination[:] = derivatives[site]; calls['jacobian'] += 1
    fake.mj_forward, fake.mj_jacSite = forward, jac_site
    human = ModuleType('weft.human'); human.HumanEnv = denied
    guarded = builtins.__import__
    def imported(name, *args, **kwargs):
        return fake if name == 'mujoco' else human if name == 'weft.human' else guarded(name, *args, **kwargs)
    monkeypatch.setattr(builtins, '__import__', imported)
    monkeypatch.setitem(sys.modules, 'mujoco', fake); monkeypatch.setitem(sys.modules, 'weft.human', human)
    spec = importlib.util.spec_from_file_location('_camera_fit_analytic_kinematics', Path(api.__file__).with_name('body_refinement.py'))
    producer = importlib.util.module_from_spec(spec); spec.loader.exec_module(producer)
    env = SimpleNamespace(model=SimpleNamespace(nv=19), _q_array=np.arange(19), _v_array=np.arange(19),
                          _site_ids=dict(zip(api.JOINTS, range(12))) | {'pelvis': 12})
    state = SimpleNamespace(qpos=np.zeros(19), site_xpos=np.zeros((13, 3)))
    relative, jacobian = producer._kinematics(env, state, np.zeros(19), True)
    result = prepared.project(0, {'points': relative, 'point_jac': jacobian.reshape(12, 3, 19),
        'pelvis_position': state.site_xpos[12], 'pelvis_jac': derivatives[12]}, derivatives=True)
    epsilon = 1e-6
    plus = pixels(data['camera'], 0, base[:12] + [1., 2., 0.] + epsilon * derivatives[:12, :, 0])
    minus = pixels(data['camera'], 0, base[:12] + [1., 2., 0.] - epsilon * derivatives[:12, :, 0])
    np.testing.assert_allclose(result['pixel_jac'][:, :, 0], (plus - minus) / (2 * epsilon), atol=3e-8)
    assert calls == {'forward': 1, 'jacobian': 13}
