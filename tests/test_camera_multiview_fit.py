"""Optional real-QP two-view composition using analytical kinematics only."""

from copy import deepcopy
import builtins
from hashlib import sha256
import importlib
import json
import os
from pathlib import Path
import time

import numpy as np
import pytest

from test_multiview_landmark_evidence import local_only
from test_camera_body_fit import Geometry, base_points, fixture as camera_fixture, pixels, raw


def fixture(*, fps=30., change=None, annotation='unknown_height'):
    camera_api = importlib.import_module('weft.camera_body_fit')
    def supplied(value):
        # The shared camera fixture's poison string is not a valid interchange field.
        for row in value['evidence'][1]['frames']:
            row['world_landmarks'] = None
        if change:
            change(value)
    data = camera_fixture(camera_api, fps=fps, source_change=supplied, annotation=annotation)
    secondary_camera, secondary_pose = data['views']['cameras'][1], data['views']['evidence'][1]
    observations = importlib.import_module('weft.observations').adapt_exercise(secondary_pose)
    obs_raw = raw(observations); obs_pin = sha256(obs_raw).hexdigest()
    claim = {'status': 'assumed', 'method': 'Authored analytical same-world/original-clock declaration',
             'uncertainty': 'Not measured calibration or synchronization', 'evidence_refs': []}
    world = lambda camera: {'id': 'camera-world:' + camera_api.canonical_hash(camera), 'length_unit': 'm', 'axes': 'x_right_y_depth_z_up'}
    alignment = {'schema_version': 1, 'profile': 'world-alignment-v1', 'from_frame': world(secondary_camera),
        'to_frame': world(data['camera']), 'spatial': {'scale': 1., 'rotation_wxyz': [1., 0., 0., 0.], 'translation': [0., 0., 0.]},
        'clock': {'rate': 1., 'source_origin_s': 0., 'target_origin_s': 0.},
        'claims': {'spatial': deepcopy(claim), 'clock': deepcopy(claim)}}
    secondary = {'camera_document': raw(secondary_camera), 'evidence_document': raw(secondary_pose),
        'observation_documents': {obs_pin: obs_raw}, 'world_alignment_document': raw(alignment),
        'correspondence': {'primary_entity_ref': deepcopy(data['binding']['human_entity_ref']),
            'secondary_entity_ref': {'document_sha256': obs_pin, 'object_id': 'person-0'}, 'status': 'assumed',
            'method': 'Authored same-person model-endpoint hypothesis', 'uncertainty': 'Not verified identity or anatomy'},
        'image_scale': .025, 'weight': 3.}
    return data, secondary


def prepare(data, secondary, *, deadline=None):
    camera_api = importlib.import_module('weft.camera_body_fit')
    primary = camera_api.prepare_camera_body_fit(raw(data['body']), raw(data['evidence']), raw(data['camera']),
        image_scale=.01, surface_objective=data['objective'])
    return importlib.import_module('weft.camera_multiview_fit').prepare_camera_multiview_fit(
        primary, secondary, deadline=time.monotonic() + 5 if deadline is None else deadline)


def test_exact_two_view_preparation_preserves_clocks_weights_and_all_source_slots():
    data, secondary = fixture(fps=30000 / 1001)
    before = raw(data['body'])
    prepared = prepare(data, secondary)
    assert prepared.ready
    meta, rows = prepared.describe(), prepared.rows()
    assert data['body']['dt'] == 1 / 30 and meta['coupling']['clock']['control_dt_s'] == 1001 / 30000
    assert len(meta['source_slots']) == len(meta['multiview']['secondary_source_slots']) == 10
    assert len(rows) == 3 and all(len(row['points']) == 24 for row in rows)
    assert rows[0]['points'][12]['weight'] == rows[0]['points'][12]['source_confidence']
    assert rows[0]['points'][12]['objective_weight'] == 3 * rows[0]['points'][12]['weight']
    np.testing.assert_array_equal(rows[0]['weights'][12:], rows[0]['objective_weights'][12:] / 3)
    assert raw(data['body']) == before
    copied = prepared.describe(); copied['multiview']['views'].clear()
    assert len(prepared.describe()['multiview']['views']) == 2


@pytest.mark.parametrize('kind', ['missing', 'unknown', 'translation', 'clock', 'foreign_frame'])
def test_unavailable_alignment_never_queries_secondary_landmarks(monkeypatch, kind):
    data, secondary = fixture()
    if kind == 'missing': secondary['world_alignment_document'] = None
    else:
        alignment = json.loads(secondary['world_alignment_document'])
        if kind == 'unknown':
            alignment['spatial'] = None; alignment['claims']['spatial']['status'] = 'unknown'
        if kind == 'translation': alignment['spatial']['translation'][0] = .1
        if kind == 'clock': alignment['clock']['target_origin_s'] = .01
        if kind == 'foreign_frame': alignment['from_frame']['id'] = 'foreign'
        secondary['world_alignment_document'] = raw(alignment)
    api = importlib.import_module('weft.camera_multiview_fit')
    original = api._targets
    def targets(*args, **kwargs):
        assert kwargs['source_qualified'] is False
        return original(*args, **kwargs)
    monkeypatch.setattr(api, '_targets', targets)
    prepared = prepare(data, secondary)
    assert not prepared.ready
    meta = prepared.describe()['multiview']
    assert len(meta['secondary_source_slots']) == 10
    assert meta['additional_view']['binding']['qualification']['known_targets_by_view'][1] == 0
    with pytest.raises(ValueError, match='unavailable'): prepared.rows()


@pytest.mark.parametrize('kind', ['camera', 'evidence', 'observation_pin', 'observation_provenance', 'primary_entity', 'secondary_entity'])
def test_exact_custody_and_entity_mismatches_reject(kind):
    data, secondary = fixture()
    if kind in ('camera', 'evidence'):
        key = kind + '_document'; value = json.loads(secondary[key]); value['unrelated_metadata'] = 'changed'
        secondary[key] = raw(value)
    if kind.startswith('observation'):
        pin, encoded = next(iter(secondary['observation_documents'].items()))
        value = json.loads(encoded)
        if kind == 'observation_pin': secondary['observation_documents'] = {'f' * 64: encoded}
        else:
            value['provenance']['input_evidence_sha256'] = 'f' * 64
            encoded = raw(value); pin = sha256(encoded).hexdigest()
            secondary['observation_documents'] = {pin: encoded}
            secondary['correspondence']['secondary_entity_ref']['document_sha256'] = pin
    if kind.endswith('_entity'): secondary['correspondence'][kind + '_ref']['object_id'] = 'other'
    with pytest.raises(ValueError): prepare(data, secondary)


def test_projection_and_jacobians_use_each_views_scale_and_absolute_pelvis():
    data, secondary = fixture()
    prepared = prepare(data, secondary)
    api = importlib.import_module('weft.camera_body_fit')
    geometry = Geometry(api); q = np.zeros(19); column = api.SPATIAL_JOINT_NAMES.index('left_knee')
    state = geometry.evaluate(q, np.zeros(3), True, surface_intervals=True)
    projected = prepared.project(1, state, derivatives=True)
    assert projected['image'].shape == (24, 2) and projected['image_jac'].shape == (24, 2, 19)
    step = 1e-6
    changed = q.copy(); changed[column] += step
    after = prepared.project(1, geometry.evaluate(changed, np.zeros(3), surface_intervals=True))
    np.testing.assert_allclose((after['pixels'] - projected['pixels']) / step,
                               projected['pixel_jac'][:, :, column], atol=2e-6)
    np.testing.assert_allclose(projected['image'][:12], projected['pixels'][:12] * .01)
    np.testing.assert_allclose(projected['image'][12:], projected['pixels'][12:] * .025)


def test_secondary_construction_is_independent_of_missing_primary_targets(monkeypatch):
    data, secondary = fixture(annotation='withheld')
    prepared = prepare(data, secondary)
    rows = prepared.rows()
    assert not np.any(rows[1]['weights'][:12]) and np.any(rows[1]['weights'][12:])
    assert rows[1]['points'][0]['landmark_source'] == 'not_read_unqualified_source'
    assert rows[1]['points'][12]['landmark_source'] == 'landmarks'


def test_secondary_excluded_payload_never_reaches_targets(monkeypatch):
    data, secondary = fixture()
    _, encoded = next(iter(secondary['observation_documents'].items()))
    doc = json.loads(encoded)
    for row in doc['observations']:
        if row['frame'] == 2: row['partition'] = 'audit'
    encoded = raw(doc); pin = sha256(encoded).hexdigest()
    secondary['observation_documents'] = {pin: encoded}
    secondary['correspondence']['secondary_entity_ref']['document_sha256'] = pin
    api = importlib.import_module('weft.camera_multiview_fit'); original = api._targets
    def targets(frame, *args, **kwargs):
        if frame.get('frame') == 2: assert not kwargs['source_qualified']
        return original(frame, *args, **kwargs)
    monkeypatch.setattr(api, '_targets', targets)
    prepared = prepare(data, secondary)
    assert not np.any(prepared.rows()[1]['weights'][12:])
    assert prepared.describe()['multiview']['secondary_source_slots'][2]['source_partition'] == 'audit'


def test_zero_secondary_targets_cannot_qualify_multiview():
    def change(value):
        for row in value['evidence'][1]['frames']:
            for point in row['landmarks']: point['visibility'] = .1
    data, secondary = fixture(change=change)
    prepared = prepare(data, secondary)
    assert not prepared.ready
    assert prepared.describe()['multiview']['additional_view']['binding']['qualification']['reason'] == 'one_or_more_views_have_no_qualified_targets'


def test_coincident_camera_centres_do_not_claim_stereo_information():
    def change(value):
        for row, original in zip(value['cameras'][1]['frames'], value['cameras'][0]['frames'], strict=True):
            row['R'], row['t'] = deepcopy(original['R']), deepcopy(original['t'])
    data, secondary = fixture(change=change)
    prepared = prepare(data, secondary)
    assert not prepared.ready
    assert prepared.describe()['multiview']['additional_view']['binding']['qualification']['reason'] == 'coincident_camera_centres_no_stereo_baseline'


def test_expired_deadline_and_byte_envelope_reject_before_parsing(monkeypatch):
    data, secondary = fixture()
    with pytest.raises(TimeoutError): prepare(data, secondary, deadline=0)
    secondary['camera_document'] = b' ' * ((8 << 20) + 1)
    with pytest.raises(ValueError, match='bounded immutable'): prepare(data, secondary)


def test_exact_secondary_nominal_clock_rejects_tolerant_but_different_stamp():
    def change(value):
        value['evidence'][1]['frames'][1]['time_s'] += 1e-9
    data, secondary = fixture(change=change)
    with pytest.raises(ValueError, match='nominal clock'): prepare(data, secondary)


def test_secondary_metadata_tail_and_unsampled_slots_remain_explicit():
    def change(value): value['evidence'][1]['decoded_frames'] = 8
    data, secondary = fixture(change=change)
    prepared = prepare(data, secondary); meta = prepared.describe()['multiview']
    assert len(meta['secondary_source_slots']) == 10
    assert [slot['source_status'] for slot in meta['secondary_source_slots']] == ['decoded'] * 8 + ['metadata_only_not_decoded'] * 2
    assert meta['secondary_source_slots'][1]['status'] == 'unsampled'
    assert meta['additional_view']['binding']['qualification']['secondary_pose_rows'] == 3


def fit(data, secondary, monkeypatch, *, geometry_type=Geometry, multiview=True, settings=None):
    api = importlib.import_module('weft.whole_body_reference_refinement')
    camera_api = importlib.import_module('weft.camera_body_fit')
    engines = []
    def geometry(body):
        assert raw(body) == raw(data['body'])
        value = geometry_type(camera_api, final=bool(engines)); engines.append(value); return value
    monkeypatch.setattr(api, '_make_geometry', geometry)
    kwargs = {'proposal_objective': 'camera_multiview_surface_interval_v1' if multiview else 'camera_surface_interval_v1',
              'camera_document': raw(data['camera']), 'camera_image_scale': .01,
              'surface_interval_objective': data['objective'], 'surface_interval_weight': 2., 'joint_mask': ['left_knee']}
    if multiview: kwargs['camera_secondary_view'] = secondary
    if settings is not None: kwargs['settings'] = settings
    result = api.propose_whole_body_reference_refinement(raw(data['body']), raw(data['evidence']), **kwargs)
    assert len(engines) == 2 and all(value.closed for value in engines)
    assert result['report']['geometry_calls'] == sum(value.calls for value in engines)
    return result


def save_fixture(path, data, secondary, result):
    path.mkdir(); (path / 'inputs').mkdir()
    files = {'inputs/human.json': raw(data['body']), 'inputs/evidence-derived.json': raw(data['evidence']),
        'inputs/evidence-original.json': raw(data['evidence']), 'inputs/source-camera.json': raw(data['camera']),
        'inputs/surface-camera.json': raw(data['camera']), 'inputs/surface-body-binding.json': raw(data['binding']),
        'inputs/surface-intervals.json': data['envelope_inputs']['surface_interval_document'],
        'inputs/surface-annotations.json': data['envelope_inputs']['annotation_document'],
        'inputs/secondary-camera.json': secondary['camera_document'], 'inputs/secondary-evidence.json': secondary['evidence_document'],
        'inputs/camera-secondary-alignment.json': secondary['world_alignment_document'],
        'secondary-settings.json': raw({key: secondary[key] for key in ('image_scale', 'weight', 'correspondence')}),
        'angle-proposal.json': raw(result['angle_proposal']), 'report.json': raw(result['report']), 'result.json': raw(result)}
    for index, value in enumerate(data['envelope_inputs']['observation_documents'].values()):
        files[f'inputs/surface-observation-{index:02d}.json'] = value
    for index, value in enumerate(secondary['observation_documents'].values()):
        files[f'inputs/camera-secondary-observation-{index:02d}.json'] = value
    for name, value in files.items():
        with (path / name).open('xb') as handle: handle.write(value)
    manifest = {'schema_version': 1, 'profile': 'authored-camera-multiview-fit-analytic-v1',
        'source_accuracy': 'unverified_authored_fixture', 'native_calls': 0,
        'members': {name: {'sha256': sha256(value).hexdigest(), 'bytes': len(value)} for name, value in files.items()}}
    (path / 'manifest.json').open('xb').write(raw(manifest))


def test_existing_qp_improves_both_views_with_exact_fourteen_gates(monkeypatch, tmp_path, record_property):
    data, secondary = fixture()
    original = raw(data['body']); started = time.perf_counter()
    result = fit(data, secondary, monkeypatch)
    report, overlay = result['report'], result['angle_proposal']
    assert result['human_candidate'] is None and overlay['profile'] == 'camera-body-multiview-angle-proposal-v1'
    assert len(report['gates']) == 14 and all(report['gates'].values()) and report['accepted_updates'] > 0
    assert report['multiview_images']['accepted'] is True
    for view in report['multiview_images']['views']:
        assert view['known_targets'] == 36 and view['missing_targets'] == 0
        assert view['mean_error_after_px'] < view['mean_error_before_px']
        for name in view['gates']:
            assert report['gates'][name] == all(item['gates'][name] for item in report['multiview_images']['views'])
    assert all(len(row['points']) == 24 for row in report['references'])
    assert overlay['reference_candidate'][0] == data['body']['reference'][0] and raw(data['body']) == original
    assert overlay['additional_view'] == report['camera_fit']['multiview']['additional_view']
    assert report['camera_fit']['projection_calls'] % 2 == 0 and report['physics_steps'] == 0
    record_property('analytical_seconds', time.perf_counter() - started)
    record_property('geometry_calls', report['geometry_calls'])
    record_property('projection_calls', report['camera_fit']['projection_calls'])
    record_property('output_bytes', len(raw(result)))
    target = os.environ.get('WEFT_CAMERA_MULTIVIEW_FIXTURE_OUT')
    path = Path(target) if target else tmp_path / 'analytic'
    if target: assert path.parts[0] == '.local' and '..' not in path.parts
    save_fixture(path, data, secondary, result)


def test_missing_alignment_rejects_before_geometry(monkeypatch):
    data, secondary = fixture(); secondary['world_alignment_document'] = None
    api = importlib.import_module('weft.whole_body_reference_refinement')
    monkeypatch.setattr(api, '_make_geometry', lambda *_: pytest.fail('Unavailable mapping reached geometry'))
    with pytest.raises(ValueError, match='secondary camera view unavailable'):
        api.propose_whole_body_reference_refinement(raw(data['body']), raw(data['evidence']),
            proposal_objective='camera_multiview_surface_interval_v1', camera_secondary_view=secondary,
            camera_document=raw(data['camera']), camera_image_scale=.01,
            surface_interval_objective=data['objective'], surface_interval_weight=2.)


def test_projection_bound_reserves_both_view_baseline_and_final_before_geometry(monkeypatch):
    data, secondary = fixture()
    api = importlib.import_module('weft.whole_body_reference_refinement')
    monkeypatch.setattr(api, '_make_geometry', lambda *_: pytest.fail('Incomplete view budget reached geometry'))
    with pytest.raises(ValueError, match='both view baseline and final'):
        api.propose_whole_body_reference_refinement(raw(data['body']), raw(data['evidence']),
            proposal_objective='camera_multiview_surface_interval_v1', camera_secondary_view=secondary,
            camera_document=raw(data['camera']), camera_image_scale=.01,
            surface_interval_objective=data['objective'], surface_interval_weight=2., settings={'maximum_geometry_calls': 10})


def test_aggregate_extra_input_bound_precedes_json_parsing(monkeypatch):
    data, secondary = fixture()
    primary = prepare(data, secondary).primary
    secondary['camera_document'] = secondary['evidence_document'] = b' ' * (8 << 20)
    fitter = importlib.import_module('weft.whole_body_reference_refinement')
    monkeypatch.setattr(fitter, '_document', lambda *_: pytest.fail('Oversized aggregate reached JSON parser'))
    api = importlib.import_module('weft.camera_multiview_fit')
    with pytest.raises(ValueError, match='sixteen MiB'):
        api.prepare_camera_multiview_fit(primary, secondary, deadline=time.monotonic() + 5)


def test_minimum_projection_budget_retains_complete_final_audit(monkeypatch):
    data, secondary = fixture()
    result = fit(data, secondary, monkeypatch, settings={'maximum_geometry_calls': 12})
    report = result['report']
    assert result['status'] == 'partial_budget_exhausted'
    assert report['camera_fit']['projection_calls'] == 12 and report['geometry_calls'] <= 12
    assert len(report['references']) == 3 and len(report['multiview_images']['views']) == 2
    assert all(view['known_targets'] == 36 for view in report['multiview_images']['views'])


def test_final_preparation_serialization_uses_the_original_shorter_deadline(monkeypatch):
    data, secondary = fixture(); primary = prepare(data, secondary).primary
    api = importlib.import_module('weft.camera_multiview_fit')
    original, clock, late = api._encode, time.monotonic, [False]
    deadline = clock() + 2
    monkeypatch.setattr(time, 'monotonic', lambda: clock() + (3 if late[0] else 0))
    def encode(value, *args):
        result = original(value, *args)
        if 'multiview' in value: late[0] = True
        return result
    monkeypatch.setattr(api, '_encode', encode)
    with pytest.raises(TimeoutError): api.prepare_camera_multiview_fit(primary, secondary, deadline=deadline)


def test_single_view_output_matches_exact_published_golden_and_never_imports_extra_mode(monkeypatch):
    monkeypatch.setattr(time, 'monotonic', lambda: 100.)
    data, secondary = fixture()
    original = builtins.__import__
    def imported(name, *args, **kwargs):
        assert name != 'weft.camera_multiview_fit', 'Single-view path imported the extra-view helper'
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins, '__import__', imported)
    result = fit(data, secondary, monkeypatch, multiview=False)
    encoded = json.dumps(result, sort_keys=True, allow_nan=False).encode()
    assert sha256(encoded).hexdigest() == 'bebe97ae1de1af55d62b08cb5244d06f55f3aab243fd81ab94a9ea782ce2c839'


def test_contradictory_targets_cannot_buy_secondary_improvement_with_primary_regression(monkeypatch):
    def change(value):
        world = base_points() + [1., 2., 1.]; world[10, 1] -= .05
        camera = value['cameras'][1]
        for row in value['evidence'][1]['frames']:
            pixel = pixels(camera, row['frame'], world)[10]
            row['landmarks'][27].update(x=float(pixel[0] / camera['width']), y=float(pixel[1] / camera['height']))
    data, secondary = fixture(change=change)
    result = fit(data, secondary, monkeypatch)
    assert result['report']['accepted_updates'] == 0
    assert result['angle_proposal']['reference_candidate'] == data['body']['reference']
    assert all(view['gates']['source_endpoint_image_nonregression'] for view in result['report']['multiview_images']['views'])


def test_fresh_secondary_domain_failure_is_not_hidden_by_pooled_error(monkeypatch):
    data, secondary = fixture()
    api = importlib.import_module('weft.camera_multiview_fit')
    original = api.CameraMultiviewFit.project
    class FinalGeometry(Geometry):
        def evaluate(self, *args, **kwargs):
            return super().evaluate(*args, **kwargs) | {'final_audit': self.final}
    def project(self, index, state, **kwargs):
        result = original(self, index, state, **kwargs)
        if state.get('final_audit'): result['valid'][12] = False
        return result
    monkeypatch.setattr(api.CameraMultiviewFit, 'project', project)
    result = fit(data, secondary, monkeypatch, geometry_type=FinalGeometry)
    report = result['report']; first, second = report['multiview_images']['views']
    assert first['gates']['camera_projection_domain'] is True and second['gates']['camera_projection_domain'] is False
    assert second['known_targets'] == 36 and second['invalid_after_known'] == 3 and second['mean_error_after_px'] is None
    assert report['gates']['camera_projection_domain'] is False and report['multiview_images']['accepted'] is False
    assert set(report['failures']) == {name for name, value in report['gates'].items() if not value}
    assert result['status'] == 'rejected_construction_proposal'


def test_second_view_resolves_an_analytical_primary_ray_ambiguity_in_existing_qp(monkeypatch):
    direction = np.array([.9, 2., -2.55])
    class RayGeometry(Geometry):
        def evaluate(self, angles, root, derivatives=False, *, surface_intervals=False):
            result = super().evaluate(np.zeros_like(angles), root, derivatives, surface_intervals=surface_intervals)
            column = self.names.index('left_knee')
            result['points'][10] += angles[column] * direction
            if derivatives:
                result['point_jac'][:] = 0.; result['pelvis_jac'][:] = 0.
                result['point_jac'][10, :, column] = direction
            return result
    def change(value):
        world = base_points() + [1., 2., 1.]; world[10] += .02 * direction
        for camera, evidence in zip(value['cameras'], value['evidence'], strict=True):
            for row in evidence['frames']:
                uv = pixels(camera, row['frame'], world)
                for index, landmark in enumerate(importlib.import_module('weft.camera_body_fit').JOINTS.values()):
                    row['landmarks'][landmark].update(x=float(uv[index, 0] / camera['width']), y=float(uv[index, 1] / camera['height']))
    data, secondary = fixture(change=change)
    single = fit(data, secondary, monkeypatch, geometry_type=RayGeometry, multiview=False)
    result = fit(data, secondary, monkeypatch, geometry_type=RayGeometry)
    assert single['report']['accepted_updates'] == 0
    assert result['report']['accepted_updates'] > 0 and all(result['report']['gates'].values())
    second = result['report']['multiview_images']['views'][1]
    assert second['mean_error_after_px'] < second['mean_error_before_px']
    # The original first reference is frozen, so its unresolved error stays in the mean.
    for row in result['report']['references'][1:]:
        point = row['points'][22]
        assert point['error_after_px'] < point['error_before_px'] * .01
    assert all(abs(row['joints']['left_knee'] - .02) < .001 for row in result['angle_proposal']['reference_candidate'][1:])
