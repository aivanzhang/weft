"""Automatic frame choices through the actual offline setup and report, without physics."""

import builtins
from copy import deepcopy
import json
import socket
import zipfile

import pytest

from weft import equipment_box_consensus as selector
from weft import shared_exercise_config_report as report
from test_shared_exercise_config import (
    digest, encoded, fixture, prepare, request_for, reseal, resume, verify_package,
)
from test_shared_exercise_config_report import Page, check_links, package, raw, reseal as repin_report, run_dom


@pytest.fixture(autouse=True)
def no_native_or_network(monkeypatch):
    original = builtins.__import__
    def guarded(name, *args, **kwargs):
        assert name.split('.')[0] not in {'mujoco', 'gymnasium', 'cv2', 'av', 'torch', 'mediapipe'}, name
        return original(name, *args, **kwargs)
    def denied(*args, **kwargs):
        pytest.fail('Offline configuration attempted network access')
    monkeypatch.setattr(builtins, '__import__', guarded)
    monkeypatch.setattr(socket.socket, 'connect', denied)
    monkeypatch.setattr(socket, 'create_connection', denied)


def configured(tmp_path, *, mutate=None, automatic=True):
    inputs = fixture(tmp_path/'inputs', count=12)
    if mutate is not None:
        reseal(inputs['equipment'], 'scene-observations.json', mutate)
    first, output = tmp_path/'first', tmp_path/'configured'
    assert prepare(inputs, first)['status'] == 'needs_input'
    choices = [{'object_id': 'weight', **({'frame_selection': {'method': selector.CONSENSUS_METHOD}}
        if automatic else {'source_frame': 0})}]
    request = request_for(first, tmp_path/'chosen.json', equipment=choices)
    result = resume(first, request, output)
    return inputs, output, result, request


def report_inputs(output):
    return {name: (output/name).read_bytes() for name in
        ('metadata.json', 'snapshot.json', 'request.json', 'config.json', 'frame-selection.json') if (output/name).is_file()}


def repin_selection(files, receipts):
    files['frame-selection.json'] = raw(receipts)
    metadata = json.loads(files['metadata.json'])
    metadata['frame_selection_sha256'] = digest(files['frame-selection.json'])
    files['metadata.json'] = raw(metadata)


def test_setup_infers_then_seals_complete_receipt_and_unresolved_request(tmp_path):
    _, output, result, request = configured(tmp_path)
    assert result['status'] == 'config_ready', result
    manifest = verify_package(output)
    files = report_inputs(output); receipt = json.loads(files['frame-selection.json'])['weight']
    assert files['request.json'] == request.read_bytes()
    assert json.loads(files['config.json'])['equipment'] == json.loads(files['request.json'])['equipment']
    assert receipt['selected']['source_frame'] == 0 and receipt['statistics']['eligible_count'] == 6
    assert len(receipt['slots']) == 12 and receipt['slots'][1][9] == 'not_original_body_reference'
    assert result['work']['construction_frame_selection_calls'] == 1 and result['work']['model_inference_calls'] == 0
    assert result['world_admission'] == 'not_evaluated'
    with zipfile.ZipFile(output/'implementation-source.zip') as archive:
        assert 'equipment_box_consensus.py' in archive.namelist()
    assert manifest['files']['frame-selection.json'] == digest(files['frame-selection.json'])
    assert b'Selected original frame 0; 6 eligible observations' in (output/'inspection/index.html').read_bytes()
    check_links(report.build_shared_exercise_config_report(files), files)


def test_disagreement_keeps_every_observation_without_a_fallback_config(tmp_path):
    def mutate(doc):
        doc['observations'][10]['measurements']['box']['value'][0] += 4
    _, output, result, _ = configured(tmp_path, mutate=mutate)
    assert result['status'] == 'needs_input' and result['config_path'] is None, result
    verify_package(output)
    files = report_inputs(output); receipts = json.loads(files['frame-selection.json'])
    assert 'config.json' not in files and receipts['weight']['selected'] is None
    assert receipts['weight']['diagnostic_medoid'] is not None
    assert receipts['weight']['statistics']['eligible_count'] == 6
    assert receipts['weight']['slots'][10][10] is not None
    assert b'No frame selected; 6 eligible observations' in (output/'inspection/index.html').read_bytes()
    assert json.loads(files['metadata.json'])['issues'][0]['code'] == 'equipment_frame_selection_unavailable'


def test_manual_path_never_calls_selector_or_emits_selection_artifacts(tmp_path, monkeypatch):
    monkeypatch.setattr(selector, 'select_equipment_box_frames', lambda *a, **k: pytest.fail('manual selection inferred a frame'))
    _, output, result, _ = configured(tmp_path, automatic=False)
    assert result['status'] == 'config_ready', result
    assert not (output/'frame-selection.json').exists()
    assert 'frame_selection_sha256' not in json.loads((output/'metadata.json').read_bytes())
    assert json.loads((output/'config.json').read_bytes())['equipment'] == [{'object_id':'weight', 'source_frame':0}]
    assert 'construction_frame_selection_calls' not in result['work']


def test_receipt_only_report_does_not_run_selection_again(tmp_path, monkeypatch):
    _, output, result, _ = configured(tmp_path)
    assert result['status'] == 'config_ready'
    monkeypatch.setattr(selector, 'select_equipment_box_frames', lambda *a, **k: pytest.fail('report inferred a frame'))
    files = report_inputs(output); before = deepcopy(files)
    rendered = report.build_shared_exercise_config_report(files)
    assert files == before
    summary = json.loads(rendered['report.json'])
    assert summary['equipment_frame_selection']['weight']['status'] == 'selected'
    assert summary['geometry_recomputed'] is False
    assert 'no_inference' in summary['frame_selection_scope']


@pytest.mark.parametrize('change', ['source', 'entity', 'body_pin', 'body_time', 'observation_time', 'observation_id', 'tail', 'excluded', 'pin'])
def test_resealed_selection_cannot_disagree_with_snapshot(tmp_path, change):
    _, output, result, _ = configured(tmp_path)
    assert result['status'] == 'config_ready'
    files = report_inputs(output); receipts = json.loads(files['frame-selection.json']); receipt = receipts['weight']
    if change == 'source': receipt['source_sha256'] = 'f'*64
    elif change == 'entity': receipt['entity_ref']['object_id'] = 'another'
    elif change == 'body_pin': receipt['body_references_sha256'] = 'f'*64
    elif change == 'body_time': receipt['slots'][2][7] += .001
    elif change == 'observation_time': receipt['slots'][2][2] += .001
    elif change == 'observation_id': receipt['slots'][2][1] += '-changed'
    elif change == 'tail': receipt['original_source_slots'] += 1
    elif change == 'excluded':
        snapshot = json.loads(files['snapshot.json'])
        snapshot['observation_sources'][1]['slots'][2]['selection_eligible'] = False
        files['snapshot.json'] = raw(snapshot); repin_report(files, snapshot=True)
    elif change == 'pin':
        files['frame-selection.json'] += b' '
    if change != 'pin': repin_selection(files, receipts)
    with pytest.raises(ValueError): report.build_shared_exercise_config_report(files)


@pytest.mark.parametrize('mode', ['missing_receipt', 'extra_object', 'unavailable'])
def test_ready_automatic_choices_require_complete_selected_receipts(tmp_path, mode):
    _, output, result, _ = configured(tmp_path)
    assert result['status'] == 'config_ready'
    files = report_inputs(output); receipts = json.loads(files['frame-selection.json'])
    if mode == 'missing_receipt':
        files.pop('frame-selection.json'); metadata = json.loads(files['metadata.json'])
        metadata.pop('frame_selection_sha256'); files['metadata.json'] = raw(metadata)
    elif mode == 'extra_object':
        receipts['unrequested'] = receipts['weight']; repin_selection(files, receipts)
    else:
        receipts['weight'].update(status='unavailable', selected=None, reasons=['ordered_image_drift'])
        repin_selection(files, receipts)
    with pytest.raises(ValueError): report.build_shared_exercise_config_report(files)


@pytest.mark.parametrize('choice', [
    {'object_id':'box-1', 'source_frame':0, 'frame_selection':{'method':selector.CONSENSUS_METHOD}},
    {'object_id':'box-1', 'frame_selection':{'method':'unknown'}},
])
def test_invalid_or_ambiguous_requests_never_get_a_ready_report(choice):
    files = package(status='config_ready'); request = json.loads(files['request.json'])
    request['equipment'] = [choice]; files['request.json'] = raw(request); repin_report(files)
    with pytest.raises(ValueError): report.build_shared_exercise_config_report(files)


def test_editor_selects_automatic_without_frame_and_preserves_manual_shape(tmp_path, record_property):
    files = package(status='config_ready', frames=1800, decoded=1800)
    page = report.build_shared_exercise_config_report(files)['index.html']
    measured = run_dom(tmp_path, page, r'''
const D=JSON.parse(scripts[0][1]);assert.equal(el('frame-mode').value,'manual');
el('object').value='1';el('frame-mode').value=D.frame_selection.method;el('frame-mode').fire('change');
assert.equal(el('source-frame').disabled,true);assert.equal(el('add-equipment').disabled,false);
const count=created, baseline=writes, start=performance.now();for(let i=0;i<10000;i++)el('frame-mode').fire('change');
assert.equal(created,count);assert.equal(writes,baseline);const unchangedMs=performance.now()-start;
el('add-equipment').fire('click');el('download-request').fire('click');
const request=JSON.parse(blobs[0]);assert.deepEqual(request.equipment,[{object_id:'box-1',source_frame:0},{object_id:'box-2',frame_selection:{method:D.frame_selection.method}}]);
assert.match(el('selected-equipment').textContent,/choose from consistent frames/);
el('frame-mode').value='manual';el('frame-mode').fire('change');assert.equal(el('source-frame').disabled,false);
process.stdout.write(JSON.stringify({unchanged_updates:10000,unchanged_writes:0,unchanged_nodes:0,unchanged_ms:unchangedMs}));
''')
    assert measured['unchanged_updates'] == 10000
    record_property('scripted_maximum_frame_editor', json.dumps(measured, sort_keys=True))


def test_saved_auto_request_loads_without_invented_manual_frame(tmp_path):
    _, output, result, _ = configured(tmp_path)
    assert result['status'] == 'config_ready'
    run_dom(tmp_path, (output/'inspection/index.html').read_bytes(), r'''
const D=JSON.parse(scripts[0][1]);assert.equal(el('download-request').disabled,false);
el('download-request').fire('click');assert.deepEqual(JSON.parse(blobs[0]),D.request);
assert.equal('source_frame' in JSON.parse(blobs[0]).equipment[0],false);
assert.match(el('selected-equipment').textContent,/choose from consistent frames/);
''')


@pytest.mark.parametrize('reason', ['partition_withheld', 'not_original_body_reference',
    'source_visibility_unavailable', 'ambiguous_or_missing_evidence'])
def test_selected_receipt_cannot_invent_known_exclusion_reason(tmp_path, reason):
    _, output, result, _ = configured(tmp_path)
    assert result['status'] == 'config_ready'
    files = report_inputs(output); receipts = json.loads(files['frame-selection.json'])
    receipts['weight']['slots'][2][9:] = [reason, None, None, None]
    receipts['weight']['statistics']['eligible_count'] -= 1
    repin_selection(files, receipts)
    with pytest.raises(ValueError, match='exclusion contradicts'):
        report.build_shared_exercise_config_report(files)


@pytest.mark.parametrize('key,value', [
    ('source_time_span_s', .1), ('edge_range_fraction', [.03, 0., 0., 0.]),
    ('fitted_drift_fraction', [0., -.02, 0., 0.]), ('maximum_cohort_gap_fraction', [0., 0., .02, 0.]),
    ('edge_range_fraction', [-.01, 0., 0., 0.]),
])
def test_selected_receipt_must_agree_with_its_saved_policy_statistics(tmp_path, key, value):
    _, output, result, _ = configured(tmp_path)
    assert result['status'] == 'config_ready'
    files = report_inputs(output); receipts = json.loads(files['frame-selection.json'])
    receipts['weight']['statistics'][key] = value; repin_selection(files, receipts)
    with pytest.raises(ValueError, match='negative range|image-consistency statistics'):
        report.build_shared_exercise_config_report(files)


def test_derivative_metadata_tail_does_not_invent_original_receipt_slots(tmp_path):
    inputs = fixture(tmp_path/'inputs', derivative=True)
    reseal(inputs['equipment'], 'scene-observations.json', lambda doc: doc['source'].update(metadata_frames=5))
    invalid = prepare(inputs, tmp_path/'invalid')
    assert invalid['status'] == 'failed' and 'timestamp count' in invalid['error']['message']
    inputs = fixture(tmp_path/'valid-inputs', derivative=True)
    first, output = tmp_path/'first', tmp_path/'configured'
    assert prepare(inputs, first)['status'] == 'needs_input'
    request = request_for(first, tmp_path/'chosen.json', equipment=[{
        'object_id':'weight', 'frame_selection':{'method':selector.CONSENSUS_METHOD}}])
    result = resume(first, request, output)
    assert result['status'] == 'needs_input' and result['sealed'], result
    files = report_inputs(output); receipt = json.loads(files['frame-selection.json'])['weight']
    assert len(receipt['slots']) == receipt['original_source_slots'] == 3
    assert receipt['status'] == 'unavailable' and receipt['reasons'] == ['insufficient_time_span']
    # A report joins saved metadata; it must not reinterpret container metadata as original frames.
    snapshot = json.loads(files['snapshot.json']); snapshot['observation_sources'][1]['metadata_frames'] = 5
    files['snapshot.json'] = raw(snapshot); repin_report(files, snapshot=True)
    rendered = report.build_shared_exercise_config_report(files)
    assert b'No frame selected' in rendered['index.html']
    assert Page(rendered['frames-001.html']).data['views'][1]['metadata_frames'] == 5


def test_receipt_original_decoded_count_must_match_equipment_view(tmp_path):
    _, output, result, _ = configured(tmp_path)
    assert result['status'] == 'config_ready'
    files = report_inputs(output); snapshot = json.loads(files['snapshot.json'])
    snapshot['counts'].update(decoded_source_slots=11, metadata_tail_slots=1)
    snapshot['source_slots'][-1]['source_status'] = 'metadata_only_not_decoded'
    files['snapshot.json'] = raw(snapshot)
    metadata = json.loads(files['metadata.json']); metadata['counts'] = snapshot['counts']
    files['metadata.json'] = raw(metadata); repin_report(files, snapshot=True)
    receipts = json.loads(files['frame-selection.json']); receipts['weight']['decoded_frames'] = 11
    repin_selection(files, receipts)
    with pytest.raises(ValueError, match='source/body identity'):
        report.build_shared_exercise_config_report(files)
