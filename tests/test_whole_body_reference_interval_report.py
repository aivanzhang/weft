"""Saved analytical report data only; no fitter, geometry, media or native runtime."""

import builtins
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
import socket
import subprocess

import pytest

from weft import whole_body_reference_report as report
from test_whole_body_reference_report import encode, fixture_documents, generated, save_study


@pytest.fixture(autouse=True)
def saved_data_only(monkeypatch):
    original = builtins.__import__
    def guarded(name, *args, **kwargs):
        assert name.split('.')[0] not in {'numpy', 'mujoco', 'cv2', 'mediapipe', 'torch', 'pymunk'}, name
        assert name not in {'weft.whole_body_reference_refinement', 'weft.source_surface_envelopes',
                            'weft.source_surface_interval_construction'}, name
        return original(name, *args, **kwargs)
    def denied(*args, **kwargs):
        pytest.fail('Saved-data report attempted external work')
    monkeypatch.setattr(builtins, '__import__', guarded)
    monkeypatch.setattr(socket.socket, 'connect', denied)
    monkeypatch.setattr(socket, 'create_connection', denied)


def digest(value):
    return sha256(encode(value)).hexdigest()


def refresh(documents):
    saved, candidate = documents['report.json'], documents['candidate-human.json']
    surface = saved['surface_intervals']; inputs = surface['inputs']
    sides = [[row[side] for row in surface['references']] for side in ('before', 'after')]
    summary = {'source_frame_slots': len(surface['source_slots']),
        'channel_slots': sum(len(row['channels']) for row in surface['source_slots']),
        'available_reference_terms': sum(row['objective_m2'] is not None for row in sides[0]),
        'unavailable_reference_terms': sum(row['objective_m2'] is None for row in sides[0])}
    for side, values in zip(('before', 'after'), sides):
        losses = [row['objective_m2'] for row in values if row['objective_m2'] is not None]
        distances = [c['distance_to_outer_envelope_m'] for row in values for c in row['channels']
                     if c['distance_to_outer_envelope_m'] is not None]
        summary.update({f'evaluated_channels_{side}': sum(row['counts']['evaluated_channels'] for row in values),
            f'sum_objective_{side}_m2': sum(losses) if losses else None,
            f'weighted_sum_{side}': surface['weight'] * sum(losses) if losses else None,
            f'maximum_channel_outside_{side}_m': max(distances) if distances else None})
    surface['summary'] = summary
    surface['journal_bytes'] = sum(len(json.dumps(row, allow_nan=False, separators=(',', ':')).encode())
                                   for row in surface['trial_journal'])
    marker = candidate['whole_body_reference_refinement']
    marker.update(method=report.SURFACE_METHOD, status=saved['status'])
    marker['surface_intervals'] = {'profile': report.SURFACE_PROFILE, 'input_bindings': {
        'objective_profile': report.SURFACE_OBJECTIVE, **{key: inputs[key] for key in
            ('source_sha256', 'body_sha256', 'binding_sha256', 'envelope_cache_sha256', 'envelope_bindings')}},
        'weight': surface['weight'], 'weight_status': 'authored', 'work_limits': surface['work_limits'], 'summary': summary}
    saved['candidate_sha256'] = report._canonical(candidate)


def interval_documents(*, binding=True):
    documents = fixture_documents()
    human, saved = documents['inputs/human.json'], documents['report.json']
    source, body_hash = human['source_sha256'], digest(human)
    observations = {'schema_version': 1, 'profile': 'scene-observations-v1',
        'source': {'sha256': source, 'decoded_frames': 3, 'metadata_frames': 4, 'fps': 2},
        'time': {'unit': 'seconds', 'origin': 'video_start', 'basis': 'frame_index_divided_by_fps', 'status': 'assumed'},
        'observations': [{'id': f'person:{i}', 'object_id': 'person', 'frame': i, 'time_s': i / 2,
                          'partition': 'construction', 'visibility': 'detected', 'evidence_ids': [f'{source}:frame:{i}']}
                         for i in range(3)]}
    entity = {'document_sha256': digest(observations), 'object_id': 'person'}
    camera = {'source_sha256': source, 'quality': 'authored', 'units': 'm',
              'provenance': 'Analytical report fixture, no fitted camera'}
    world = {'id': 'authored-room', 'length_unit': 'm', 'axes': 'x_right_y_away_z_up'}
    intervals = {'schema_version': 1, 'profile': 'source-horizontal-surface-intervals-v1', 'source_sha256': source,
        'calibration_sha256': digest(camera), 'world_frame': world, 'calibration_evidence_refs': [],
        'surfaces': [{'id': 'platform', 'kind': 'static_support', 'height_interval_m': [.2, .6],
                      'bounds_xy_m': None, 'quality': 'authored', 'provenance': 'Supplied range, not measured', 'evidence_refs': []}]}
    annotations = {'schema_version': 1, 'profile': 'source-surface-annotations-v1', 'source_sha256': source,
        'quality': 'authored', 'provenance': 'Assumed ankle-to-pixel correspondence',
        'channels': [{'id': 'ankle', 'entity_ref': entity, 'point_role': 'assumed ankle', 'surface_id': 'platform'}],
        'annotations': [{'id': f'annotation:{i}', 'channel_id': 'ankle', 'frame': i, 'time_s': i / 2,
                         'observation_id': f'person:{i}', 'status': 'annotated', 'pixel': [60., 45.], 'uncertainty_px': 2.}
                        for i in range(3)]}
    optional = {'camera': camera, 'surface_intervals': intervals, 'annotations': annotations}
    envelope_bindings = {'observation_document_sha256s': [digest(observations)],
                         **{key + '_sha256': digest(value) for key, value in optional.items()}}
    body_binding = {'schema_version': 1, 'profile': 'source-surface-interval-body-binding-v1',
        'source_sha256': source, 'body_sha256': body_hash, 'envelope_bindings': envelope_bindings,
        'human_entity_ref': entity,
        'body_to_world': {'spatial': {'scale': 1.0, 'rotation_wxyz': [1., 0., 0., 0.], 'translation': [0., 0., 0.]},
                         'claims': {'spatial': {'status': 'assumed', 'uncertainty': 'Not a physical calibration'}}},
        'channels': [{'channel_id': 'ankle', 'body_point': 'right_ankle', 'status': 'assumed',
                      'method': 'Authored correspondence', 'uncertainty': 'Not verified anatomy'}]}
    if binding:
        optional['body_binding'] = body_binding
    config = {'schema_version': 1, 'profile': 'source-surface-fit-config-v1', 'source_sha256': source, 'weight': 2.5,
        'observations': [{'path': 'original-observation.json', 'sha256': digest(observations)}],
        **{key: {'path': f'original-{key}.json', 'sha256': digest(optional[key])} if key in optional else None
           for key in ('camera', 'surface_intervals', 'annotations', 'body_binding')}}
    documents['inputs/surface-config.json'] = config
    documents['inputs/surface-observation-00.json'] = observations
    for role, value in optional.items():
        documents['inputs/surface-' + {'surface_intervals': 'intervals', 'body_binding': 'body-binding'}.get(role, role) + '.json'] = value
    descriptor = {'profile': config['profile'], 'config_sha256': digest(config), 'source_sha256': source,
                  'weight': config['weight'], 'observation_count': 1, 'optional_inputs': list(optional)}
    documents['plan.json'] = {'schema_version': 1, 'profile': 'whole-body-reference-candidate-plan-v1', 'status': 'frozen',
        'proposal_objective': 'surface_interval_v1', 'surface_constraints': descriptor,
        'input_files': {name: digest(value) for name, value in documents.items() if name.startswith('inputs/')}}
    slots = []
    for frame in range(4):
        ref_index = {0: 0, 2: 1}.get(frame)
        body_ref = None if ref_index is None else {'reference_index': ref_index, 'evidence_id': f'{source}:frame:{frame}', 'body_time_s': frame / 2}
        reason = 'metadata_only_not_decoded' if frame == 3 else 'missing_explicit_body_binding' if not binding else 'original_body_reference_unavailable' if ref_index is None else None
        channel = {'channel_id': 'ankle', 'entity_ref': entity, 'body_point': 'right_ankle' if binding else None,
            'source_observation_id': f'person:{frame}' if frame < 3 else None, 'source_time_s': frame / 2 if frame < 3 else None,
            'source_partition': 'construction' if frame < 3 else None,
            'envelope_status': 'conditional_surface_envelope' if frame < 3 else 'unavailable',
            'envelope_reason': None if frame < 3 else 'metadata_only_not_decoded',
            'status': 'conditional_interval_ready' if reason is None else 'unavailable', 'reason': reason,
            'world_envelope_m': {'minimum': [0., 0., .2], 'maximum': [1., 1., .6], 'unit': 'm',
                                 'kind': 'outer_axis_aligned_enclosure'} if frame < 3 else None}
        slots.append({'source_frame': frame, 'source_status': 'decoded' if frame < 3 else 'metadata_only_not_decoded',
                      'body_reference': body_ref, 'channels': [channel]})
    def evaluated(frame, x):
        joined = slots[frame]['channels'][0]
        channel = {k: v for k, v in joined.items() if k != 'world_envelope_m'}
        ready = channel['status'] == 'conditional_interval_ready'
        residual = [max(0., x - 1), 0., 0.] if ready else None
        loss = .5 * sum(v*v for v in residual) if ready else None
        channel.update(status='conditional_interval_residual' if ready else 'unavailable',
            signed_outside_m=residual, world_position_m=[x, .5, .4] if ready else None,
            distance_to_outer_envelope_m=max(0., x - 1) if ready else None, objective_m2=loss,
            active_axes=[x > 1., False, False] if ready else None, jacobian_m_per_rad=None)
        return {'schema_version': 1, 'profile': report.SURFACE_OBJECTIVE, 'source_frame': frame,
            'body_sha256': body_hash, 'body_reference': slots[frame]['body_reference'], 'channels': [channel],
            'counts': {'retained_channels': 1, 'evaluated_channels': int(ready), 'unavailable_channels': int(not ready)},
            'objective_m2': loss, 'parameter_names': [], 'gradient_m2_per_rad': None, 'gauss_newton_m2_per_rad2': None,
            'derivatives_status': 'unavailable', 'geometry_modified': False, 'contact': 'unknown', 'physical_validation': 'unverified'}
    status_counts = {}
    for slot in slots:
        row = slot['channels'][0]; key = row['reason'] or row['status']
        status_counts[key] = status_counts.get(key, 0) + 1
    inputs = {'schema_version': 1, 'profile': report.SURFACE_OBJECTIVE, 'source_sha256': source,
        'body_sha256': body_hash, 'body_model_sha256': report._canonical(human), 'world_frame': world,
        'source_clocks': {digest(observations): observations['time']}, 'envelope_bindings': envelope_bindings,
        'envelope_cache_sha256': 'e' * 64, 'binding_sha256': digest(body_binding) if binding else None,
        'binding': body_binding if binding else None, 'binding_unavailable_reason': None if binding else 'missing_explicit_body_binding',
        'original_start_frame': 0, 'original_start_statuses': [{k: v for k, v in slots[0]['channels'][0].items() if k != 'world_envelope_m'}],
        'counts': {'retained_source_frames': 4, 'retained_channel_slots': 4, 'original_body_references': 2, 'status_counts': status_counts},
        'optimizer_installation': 'not_installed_no_existing_fitter_callback', 'confidence': None,
        'contact': 'unknown', 'load': 'unknown', 'grasp': 'unknown', 'physical_validation': 'unverified',
        'geometry_modified': False, 'body_or_floor_installed': False}
    saved.update(profile=report.SURFACE_METHOD, qp_calls=0, accepted_updates=0)
    saved['contract']['method'] = report.SURFACE_METHOD
    saved['gates']['surface_interval_nonregression'] = True
    saved['objective'] = {'profile': 'surface_interval_v1', 'primary': 'Existing image/prior plus interval term',
        'weight': 2.5, 'weight_status': 'authored', 'interval_units': 'supplied_world_metre_squared',
        'acceptance': 'Existing gates unchanged', 'grid_interval_targets': 0}
    saved['surface_intervals'] = {'schema_version': 1, 'profile': report.SURFACE_PROFILE, 'weight': 2.5,
        'weight_status': 'authored', 'weight_units': 'existing_model_objective_units_per_supplied_world_metre_squared',
        'inputs': inputs, 'work_limits': {'maximum_evaluations': 20000, 'maximum_journal_bytes': 4 << 20},
        'source_slots': slots, 'references': [dict(reference_index=index, source_frame=frame, evidence_id=f'{source}:frame:{frame}',
            time_s=frame/2, before=evaluated(frame, 1.2 if index == 0 else 1.4), after=evaluated(frame, 1.2 if index == 0 else .6))
            for index, frame in enumerate((0, 2))], 'evaluations': 4, 'trial_journal': [], 'journal_bytes': 0,
        'optimizer_installation': 'installed_in_existing_whole_body_fitter', 'contact': 'unknown', 'physical_validation': 'unverified'}
    refresh(documents)
    return documents


def test_interval_report_preserves_saved_ranges_unavailable_slots_and_floor_rejection(tmp_path):
    documents = interval_documents()
    source, out, _, result = generated(tmp_path, documents)
    model, page = json.loads((out / 'report.json').read_bytes()), (out / 'index.html').read_text()
    surface = model['surface_intervals']
    assert result['proposal_status'] == 'rejected_construction_proposal' and result['promoted'] is False
    assert surface['summary']['sum_objective_before_m2'] == pytest.approx(.1)
    assert surface['summary']['sum_objective_after_m2'] == pytest.approx(.02)
    assert surface['summary']['weighted_sum_after'] == pytest.approx(.05)
    assert len(surface['source_slots']) == 4 and surface['source_slots'][3]['status'] == 'metadata_only_not_decoded'
    assert surface['source_slots'][1]['reference_index'] is None
    assert surface['source_slots'][1]['channels'][0]['before_m'] is None
    assert surface['source_slots'][2]['channels'][0]['after_m'] == 0
    assert surface['source_slots'][0]['channels'][0]['pixel'] == [60., 45.]
    assert surface['source_slots'][0]['channels'][0]['height_interval_m'] == [.2, .6]
    assert model['gates']['floor_feasibility'] is False and model['floor_limit'] == .025
    assert model['objective']['grid_interval_targets'] == 0
    assert len(model['selected_members']) == 11 and result['files'] == 16
    assert 'Zero outside-box error does not verify contact or an attainable source point' in page
    assert 'not calibrated physical energy' in page and 'floor limit stays 0.025 model units' in page
    assert 'Only 11 selected members' in page and 'not recompute geometry' in page
    for name in model['selected_members']:
        target = report.MEMBERS.get(name, 'frozen-plan.json.txt' if name == 'plan.json' else Path(name).name + '.txt')
        assert (out / 'artifacts' / target).read_bytes() == (source / name).read_bytes()
    helper = model['surface_input_validator']
    assert sha256((out / helper['path']).read_bytes()).hexdigest() == helper['sha256']
    assert (out / helper['path']).read_bytes() == Path(report.__file__).with_name('source_surface_fit_inputs.py').read_bytes()
    assert not (out / 'unselected-missing.json').exists()


def test_unavailable_binding_never_becomes_zero_loss(tmp_path):
    _, out, _, _ = generated(tmp_path, interval_documents(binding=False))
    surface = json.loads((out / 'report.json').read_bytes())['surface_intervals']
    assert surface['summary']['available_reference_terms'] == 0
    assert surface['summary']['unavailable_reference_terms'] == 2
    assert surface['summary']['sum_objective_after_m2'] is None
    assert surface['summary']['weighted_sum_after'] is None
    assert surface['summary']['maximum_channel_outside_after_m'] is None
    assert surface['source_slots'][0]['channels'][0]['reason'] == 'missing_explicit_body_binding'
    assert surface['source_slots'][0]['channels'][0]['world_bounds_m'] is not None


def test_partial_attempt_and_failed_fresh_first_audit_remain_reportable(tmp_path):
    documents = interval_documents(); saved = documents['report.json']; surface = saved['surface_intervals']
    saved.update(status='partial_budget_exhausted', budget_reason='interval evaluation reservation exhausted')
    saved['gates'].update(complete_bounded_solve=False, surface_interval_nonregression=False)
    saved['failures'] = [name for name, value in saved['gates'].items() if value is False]
    surface['references'][0]['after'] = deepcopy(surface['references'][1]['before'])
    after = surface['references'][0]['after']
    after.update(source_frame=0, body_reference=surface['references'][0]['before']['body_reference'])
    for key in ('source_observation_id', 'source_time_s'):
        after['channels'][0][key] = surface['references'][0]['before']['channels'][0][key]
    surface['trial_journal'] = [{'sequence': 0, 'kind': 'backtrack', 'reference_index': 1, 'source_frame': 2,
        'block': ['right_knee'], 'backtrack_index': 0, 'trial_correction_rad': [0.] * 19,
        'combined_cost_before': 1., 'combined_cost_after': None, 'interval_before_m2': .08, 'interval_after_m2': None,
        'gate_results': {k: None for k in ('floor_nonregression', 'endpoint_depth', 'source_image_nonregression', 'surface_interval_nonregression')},
        'accepted': False, 'outcome': 'evaluation_incomplete', 'floor_min_margin_model_units': None,
        'maximum_depth_change_model_units': None, 'maximum_source_image_excess_model_units': None}]
    refresh(documents)
    _, out, _, result = generated(tmp_path, documents)
    model = json.loads((out / 'report.json').read_bytes())
    assert result['proposal_status'] == 'partial_budget_exhausted'
    assert model['surface_intervals']['journal_records'] == 1
    assert model['surface_intervals']['source_slots'][0]['channels'][0]['after_m'] > model['surface_intervals']['source_slots'][0]['channels'][0]['before_m']
    exact = json.loads((out / 'artifacts/proposal-report.json.txt').read_bytes())
    assert exact['surface_intervals']['trial_journal'][0]['outcome'] == 'evaluation_incomplete'
    assert 'Incomplete proposal — not promoted' in (out / 'index.html').read_text()


@pytest.mark.parametrize('change,message', [
    (lambda d: d['report.json']['surface_intervals']['source_slots'].pop(), 'Complete original surface source slots'),
    (lambda d: d['report.json']['surface_intervals']['source_slots'][0]['channels'][0].update(source_partition='held_out'), 'partition or clock'),
    (lambda d: d['report.json']['surface_intervals']['references'][1]['after']['channels'][0].update(signed_outside_m=[.1, 0., 0.]), 'outside-box residual'),
    (lambda d: d['report.json']['surface_intervals']['references'][1]['after']['channels'][0].update(objective_m2=1.), 'squared interval loss'),
    (lambda d: d['report.json']['surface_intervals']['summary'].update(weighted_sum_after=123.), 'summary arithmetic'),
    (lambda d: d['report.json']['surface_intervals']['summary'].update(evaluated_channels_before=True), 'summary counts'),
    (lambda d: d['report.json']['surface_intervals']['inputs'].update(binding_sha256='f' * 64), 'input bindings'),
    (lambda d: d['candidate-human.json']['reference'][0]['joints'].update(right_knee=.1), 'frozen first reference'),
    (lambda d: d['report.json']['surface_intervals'].update(journal_bytes=1), 'journal byte count'),
    (lambda d: d['plan.json']['surface_constraints'].update(weight=1.), 'frozen descriptor'),
    (lambda d: d['inputs/surface-camera.json'].update(quality='measured'), 'input pins'),
])
def test_resealed_interval_contradictions_fail_before_output(tmp_path, change, message):
    documents = interval_documents(); change(documents)
    candidate = documents['candidate-human.json']
    candidate['whole_body_reference_refinement']['surface_intervals']['summary'] = documents['report.json']['surface_intervals']['summary']
    documents['report.json']['candidate_sha256'] = report._canonical(candidate)
    source = tmp_path / 'source'; seal = save_study(source, documents)
    with pytest.raises(ValueError, match=message):
        report.prepare_whole_body_reference_report(source, tmp_path / 'out', expected_manifest_sha256=seal)
    assert not (tmp_path / 'out').exists()


def test_surface_plan_is_mandatory_and_no_external_config_paths_are_followed(tmp_path, monkeypatch):
    documents = interval_documents(); source = tmp_path / 'source'; seal = save_study(source, documents)
    names = []
    original = report.read_package_file
    def tracked(root, name, **kwargs):
        names.append(name)
        assert not name.startswith('original-')
        return original(root, name, **kwargs)
    monkeypatch.setattr(report, 'read_package_file', tracked)
    report.prepare_whole_body_reference_report(source, tmp_path / 'ready', expected_manifest_sha256=seal)
    assert names.count('inputs/surface-camera.json') == 2
    other = interval_documents(); del other['plan.json']
    missing = tmp_path / 'missing'; pin = save_study(missing, other)
    with pytest.raises(ValueError, match='required sealed surface input: plan.json'):
        report.prepare_whole_body_reference_report(missing, tmp_path / 'out', expected_manifest_sha256=pin)


def test_unknown_term_cannot_be_resealed_as_zero(tmp_path):
    documents = interval_documents(binding=False)
    documents['report.json']['surface_intervals']['references'][0]['before']['objective_m2'] = 0.
    refresh(documents)
    source = tmp_path / 'source'; pin = save_study(source, documents)
    with pytest.raises(ValueError, match='interval total differs'):
        report.prepare_whole_body_reference_report(source, tmp_path / 'out', expected_manifest_sha256=pin)


def lose_final_channel(documents):
    result = documents['report.json']['surface_intervals']['references'][1]['after']
    result['channels'][0].update(status='unavailable', reason='missing_candidate_body_point', signed_outside_m=None,
        world_position_m=None, distance_to_outer_envelope_m=None, objective_m2=None, active_axes=None)
    result.update(objective_m2=None, counts={'retained_channels': 1, 'evaluated_channels': 0, 'unavailable_channels': 1})


def increase_final_distance(documents):
    result = documents['report.json']['surface_intervals']['references'][0]['after']
    channel = result['channels'][0]
    x = channel['world_position_m'][0] + 5e-13
    channel.update(world_position_m=[x, .5, .4], signed_outside_m=[x - 1, 0., 0.],
                   distance_to_outer_envelope_m=x - 1, objective_m2=.5 * (x - 1)**2)
    result['objective_m2'] = channel['objective_m2']


@pytest.mark.parametrize('change', [lose_final_channel, increase_final_distance])
def test_true_interval_gate_cannot_erase_channels_or_tolerate_increased_distance(tmp_path, change):
    documents = interval_documents(); change(documents); refresh(documents)
    source = tmp_path / 'source'; pin = save_study(source, documents)
    with pytest.raises(ValueError, match='surface gate contradicts'):
        report.prepare_whole_body_reference_report(source, tmp_path / 'out', expected_manifest_sha256=pin)
    assert not (tmp_path / 'out').exists()


def test_false_interval_gate_preserves_newly_unavailable_final_channel(tmp_path):
    documents = interval_documents(); lose_final_channel(documents)
    documents['report.json']['gates']['surface_interval_nonregression'] = False
    documents['report.json']['failures'].append('surface_interval_nonregression')
    refresh(documents)
    _, out, _, _ = generated(tmp_path, documents)
    model = json.loads((out / 'report.json').read_bytes())
    assert model['gates']['surface_interval_nonregression'] is False
    assert model['surface_intervals']['source_slots'][2]['channels'][0]['after_m'] is None


def test_interval_inputs_and_html_keep_existing_byte_caps(tmp_path, monkeypatch):
    source = tmp_path / 'source'; pin = save_study(source, interval_documents())
    with pytest.raises(ValueError):
        report.prepare_whole_body_reference_report(source, tmp_path / 'input-bounded', expected_manifest_sha256=pin, max_input_bytes=1024)
    monkeypatch.setattr(report, 'MAX_HTML', 1024)
    with pytest.raises(ValueError, match='HTML exceeds'):
        report.prepare_whole_body_reference_report(source, tmp_path / 'html-bounded', expected_manifest_sha256=pin)
    assert not (tmp_path / 'input-bounded').exists() and not (tmp_path / 'html-bounded').exists()


def test_scripted_interval_navigation_is_lazy_bounded_and_offline(tmp_path):
    documents = interval_documents()
    _, out, _, _ = generated(tmp_path, documents)
    harness = r"""
import fs from 'node:fs';import vm from 'node:vm';import assert from 'node:assert/strict';import{performance}from'node:perf_hooks';
const html=fs.readFileSync(process.argv[2],'utf8'),scripts=[...html.matchAll(/<script(?:\s[^>]*)?>([\s\S]*?)<\/script>/g)];let created=0;
class E{constructor(){this.children=[];this.listeners=new Map();this.dataset={};this.style={};this.value='';this.checked=false;this.own='';created++;}set textContent(v){this.own=String(v);this.children=[];}get textContent(){return this.own+this.children.map(v=>v.textContent).join('');}set innerHTML(v){assert.fail('unsafe HTML');}append(...v){this.children.push(...v);}replaceChildren(...v){this.own='';this.children=[...v];}setAttribute(k,v){this[k]=String(v);}addEventListener(k,f){this.listeners.set(k,f);}fire(k){this.listeners.get(k)?.();}}
const ids=new Map([...html.matchAll(/\bid="([^"]+)"/g)].map(v=>[v[1],new E()])),el=k=>{assert(ids.has(k),k);return ids.get(k)};
el('data').textContent=scripts[0][1];el('mode').value='source';el('affected').checked=true;
vm.runInNewContext(scripts[1][1],{document:{getElementById:el,createElement:()=>new E(),createElementNS:()=>new E()},fetch:()=>assert.fail('network'),setTimeout:()=>assert.fail('timer')},{timeout:1000});
assert.equal(el('interval-sample').children.length,4);assert.match(el('interval-channels').textContent,/58.0000.*62.0000/);
const choose=i=>{el('interval-sample').value=String(i);el('interval-sample').fire('change')};choose(1);
assert.match(el('interval-context').textContent,/No body reference/);assert.match(el('interval-channels').textContent,/original_body_reference_unavailable/);
choose(3);assert.match(el('interval-context').textContent,/metadata_only_not_decoded/);assert.match(el('interval-channels').textContent,/Unknown/);
choose(2);const table=el('interval-channels'),row=table.children[0],baseline=created,start=performance.now();
for(let i=0;i<200;i++)choose(2);assert.equal(created,baseline);assert.equal(table.children[0],row);
const live=n=>1+n.children.reduce((s,c)=>s+live(c),0),count=live(table);for(let i=0;i<100;i++){choose(0);choose(2);}assert.equal(live(table),count);
process.stdout.write(JSON.stringify({same_slot_updates:200,alternating_updates:200,unchanged_slot_new_nodes:0,table_live_nodes:count,seconds:(performance.now()-start)/1000,scope:'scripted DOM only; not browser responsiveness'})+'\n');
"""
    result = subprocess.run(['node', '--input-type=module', '-', str(out / 'index.html')], input=harness,
                            text=True, capture_output=True, timeout=5)
    assert result.returncode == 0, result.stderr
    measurement = json.loads(result.stdout)
    assert measurement['unchanged_slot_new_nodes'] == 0 and measurement['table_live_nodes'] == 9
    print(result.stdout.strip())
