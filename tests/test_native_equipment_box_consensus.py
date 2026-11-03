"""Native timing stays distinct from nominal FPS and independent equipment decoder time."""

from copy import deepcopy
import json
from pathlib import Path

import pytest

from test_equipment_box_consensus import inputs, raw, select
from weft import equipment_box_consensus as consensus
from weft.source_clock import PROFILE, clock_summary


def native_inputs(monkeypatch, *, metadata_frames=13):
    base, _, kwargs, _, _ = inputs(monkeypatch)
    document = json.loads(kwargs['construction_evidence'])
    source = document['sources'][0]
    ticks = [0, 80, 211, 309, 451, 552, 677, 799, 904, 1020, 1130, 1240]
    times = [tick / 1000 for tick in ticks]
    clock = {'schema_version': 1, 'profile': PROFILE, 'source_sha256': source['source']['sha256'],
             'evidence_sha256': 'b'*64, 'evidence_canonical_sha256': 'c'*64,
             'metadata_frames': metadata_frames, 'decoded_frames': 12, 'nominal_fps': 10,
             'time_base': '1/1000', 'native_origin_pts': 1000, 'decoder_minus_native_s': -.003,
             'native_to_reconstruction_offset_s': -1., 'capture_clock_status': 'unknown',
             'detector_clock_status': 'preserved_nominal_frame_index_divided_by_fps',
             'rows': [{'frame': i, 'pts': tick+1000, 'decoder_timestamp_s': (tick+1000)/1000-.003}
                      for i, tick in enumerate(ticks)]}
    source['source'].update(metadata_frames=metadata_frames, frame_times_s=times, native_clock=clock,
        native_clock_summary=clock_summary(clock), frame_count_audit={
            'encoded_video_frames': 12, 'decoded_frames': 12, 'matches': True})
    source['observation_schema_version'] = 4
    source['time'] = {'unit': 'seconds', 'origin': 'first_native_presentation_timestamp',
                      'basis': 'native_pts_elapsed_time', 'status': 'observed'}
    for collection in ('observations', 'evidence'):
        for row in source[collection]:
            row['time_s'] = times[row['frame']]
    for index, row in enumerate(base['human_spec']['reference_evidence']):
        frame = int(row['evidence_id'].rpartition(':')[2])
        row['time_s'] = times[frame]
        base['human_spec']['reference'][index]['time_s'] = times[frame]
        base['human_spec']['root_reference'][index]['time_s'] = times[frame]
    kwargs['construction_evidence'] = raw(document)
    return base, kwargs, document, times


@pytest.mark.parametrize('metadata_frames', [10, 12, 13, 14])
def test_native_times_and_metadata_only_slots_are_preserved(monkeypatch, metadata_frames):
    base, kwargs, _, times = native_inputs(monkeypatch, metadata_frames=metadata_frames)
    before = deepcopy((base, kwargs))
    receipts = select(base, kwargs)
    for receipt in receipts.values():
        assert consensus.validate_selection_receipt(receipt) == receipt
        assert receipt['status'] == 'selected'
        assert [row[2] for row in receipt['slots'][:12]] == times
        assert receipt['slots'][2][2] == receipt['slots'][2][7] == .211
        assert receipt['source_clock']['basis'] == 'native_pts_elapsed_time'
        assert receipt['decoded_frames'] == 12 and len(receipt['slots']) == max(12, metadata_frames)
        assert all(row[9] == 'metadata_only_not_decoded' and row[2] is None for row in receipt['slots'][12:])
    assert (base, kwargs) == before


def test_native_source_validation_occurs_once_not_per_slot_or_object(monkeypatch):
    base, kwargs, _, _ = native_inputs(monkeypatch)
    from weft import observations
    original = observations._validate_source
    calls = []
    def checked(source, version):
        calls.append(version)
        return original(source, version)
    monkeypatch.setattr(observations, '_validate_source', checked)
    select(base, kwargs)
    assert calls == [4]


@pytest.mark.parametrize('damage', ['times', 'ticks', 'summary', 'fps', 'audit', 'no_audit', 'clock_identity',
                                  'missing_clock_row', 'declaration', 'row_time', 'reference_time', 'schema_type'])
def test_native_clock_or_slot_tampering_never_falls_back_to_fps(monkeypatch, damage):
    base, kwargs, doc, _ = native_inputs(monkeypatch)
    source = doc['sources'][0]
    contract = source['source']
    if damage == 'times': contract['frame_times_s'] = [i/10 for i in range(12)]
    elif damage == 'ticks': contract['native_clock']['rows'][2]['pts'] += 1
    elif damage == 'summary': contract['native_clock_summary']['binding_sha256'] = 'f'*64
    elif damage == 'fps': contract['fps'] = 20
    elif damage == 'audit': contract['frame_count_audit']['matches'] = False
    elif damage == 'no_audit': contract['frame_count_audit'] = None
    elif damage == 'clock_identity': contract['native_clock']['source_sha256'] = 'f'*64
    elif damage == 'missing_clock_row': contract['native_clock']['rows'].pop()
    elif damage == 'declaration': source['time']['basis'] = 'frame_index_divided_by_fps'
    elif damage == 'row_time': source['observations'][6]['time_s'] = .2
    elif damage == 'reference_time': base['human_spec']['reference_evidence'][1]['time_s'] = .2
    else: source['observation_schema_version'] = 4.0
    kwargs['construction_evidence'] = raw(doc)
    with pytest.raises(ValueError):
        select(base, kwargs)


def test_native_body_and_equipment_decoder_clocks_remain_separate(monkeypatch):
    base, kwargs, doc, times = native_inputs(monkeypatch)
    body = doc['sources'][0]
    equipment = deepcopy(body)
    pin = 'd'*64
    equipment['document_sha256'] = pin
    equipment['observation_schema_version'] = 3
    contract = equipment['source']
    for field in ('fps', 'native_clock', 'native_clock_summary', 'frame_count_audit'):
        contract.pop(field)
    contract.update(metadata_frames=12, frame_times_s=[i/10+.075 for i in range(12)])
    equipment['time'] = {'unit': 'seconds', 'origin': 'submitted_video_start',
                         'basis': 'decoder_presentation_timestamps', 'status': 'observed'}
    for collection in ('objects', 'observations', 'evidence'):
        equipment[collection] = [r for r in equipment[collection] if r.get('object_id', r.get('id')) != 'person']
        body[collection] = [r for r in body[collection] if r.get('object_id', r.get('id')) == 'person']
    for row in equipment['objects']:
        row['entity_ref']['document_sha256'] = pin
    for row in equipment['observations']:
        row['time_s'] = contract['frame_times_s'][row['frame']]
        row['entity_ref']['document_sha256'] = pin
        row['evidence_refs'][0]['document_sha256'] = pin
    for row in equipment['evidence']:
        row['document_sha256'] = pin
        row['time_s'] = contract['frame_times_s'][row['frame']]
    doc['sources'].append(equipment)
    kwargs['source_binding']['evidence_documents'][pin] = 'evidence/'+pin+'.json'
    for entity in kwargs['source_binding']['entity_bindings'][1:]:
        entity['observation_document_sha256'] = pin
    kwargs['construction_evidence'] = raw(doc)
    receipt = select(base, kwargs)['weight']
    assert receipt['selected']['time_s'] == .075 and receipt['selected']['body_time_s'] == times[0]
    assert receipt['slots'][2][2] == .275 and receipt['slots'][2][7] == .211
    assert receipt['source_clock']['basis'] == 'decoder_presentation_timestamps'
    assert len(receipt['slots']) == 13 and receipt['slots'][-1][9] == 'metadata_only_not_decoded'


@pytest.mark.parametrize('native', [False, True])
def test_all_observed_legacy_packages_close_native_consensus_dependency(monkeypatch, tmp_path, native):
    from test_equipment_depth_prior import seal
    from weft import equipment_proposal as producer, shared_world_pipeline as pipeline

    # This fixture replaces model/native admission only; actual clock admission is tested above.
    base, config, kwargs, documents, _ = inputs(monkeypatch)
    if native:
        base['human_spec']['native_clock'] = {'scope': 'Analytical dependency closure marker; model schema mocked'}
        kwargs['human_document'] = raw(base['human_spec'])
    monkeypatch.setattr(pipeline, 'read_equipment_root_proof',
                        lambda *_a, **_k: pytest.fail('All observed legacy roots must not load v2 proof'))
    body, equipment = tmp_path/'body', tmp_path/'equipment'
    body.mkdir(); equipment.mkdir()
    (body/'scene.json').write_bytes(raw(base)); (body/'human.json').write_bytes(kwargs['human_document'])
    for pin, name in base['source_binding']['evidence_documents'].items():
        path = body/name; path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(documents[pin])
    (equipment/'scene-observations.json').write_bytes(next(iter(documents.values())))
    (equipment/'report.json').write_bytes(raw({'status': 'evidence_only', 'physics_ready': False}))
    seal(body); seal(equipment)
    config_path = tmp_path/'config.json'; config_path.write_bytes(raw(config))
    report = producer.prepare_equipment_proposal(body, equipment, config_path, tmp_path/'proposal')
    assert report['status'] == 'proposal_ready', report['reasons']
    proposal = json.loads((tmp_path/'proposal/geometry-proposal.json').read_bytes())
    assert 'profile' not in proposal['body_binding']
    pipeline.construct_source_equipment_run(body, tmp_path/'proposal', equipment, tmp_path/'candidate', target_object_id='weight')
    for directory in ('proposal', 'candidate'):
        target = tmp_path/directory/'implementation/source_clock.py'
        assert target.exists() is native
        if native:
            assert target.read_bytes() == Path(consensus.__file__).with_name('source_clock.py').read_bytes()
        assert not (tmp_path/directory/'implementation/equipment_root_binding.py').exists()
