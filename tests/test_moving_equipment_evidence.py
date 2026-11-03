"""Complete typed image custody; no physics, depth or initial-state inference."""

from copy import deepcopy
import hashlib
import json
import subprocess
import sys

import pytest

from test_equipment_root_binding import fixture as body_fixture
from weft import moving_equipment_evidence as ledger
from weft.observations import adapt_exercise
from weft.segmentation_observations import adapt_segmentation
from weft.shared_world_evidence import build_shared_world_binding
from weft.source_clock import clock_summary


def raw(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':')).encode()


def fixture(*, withheld=False, metadata_tail=False, unknown_phase=False):
    human, _, _, _, _, proof = body_fixture()
    view = json.loads(proof['reconstruction-evidence.json'])
    body = adapt_exercise(view)
    if metadata_tail:
        body['source']['metadata_frames'] = 11
        body['source']['native_clock']['metadata_frames'] = 11
        body['source']['native_clock_summary'] = clock_summary(body['source']['native_clock'])
        body['source']['frame_count_audit'] = {'encoded_video_frames': 10, 'decoded_frames': 10, 'matches': True}
        for record in body['evidence']:
            record['payload']['source_time_derivation']['binding_sha256'] = body['source']['native_clock_summary']['binding_sha256']
    identity = human['source_sha256']
    evidence = {'profile': 'segmentation-evidence-v1', 'physics_ready': False,
        'source': {'submitted_video_sha256': identity, 'width_px': 640, 'height_px': 480, 'frame_count': 10},
        'time': {'timestamps_s': [i / 20 + .017 for i in range(10)]},
        'objects': [{'id': f'sam-{i}', 'raw_model_id': str(i)} for i in (1, 2)],
        'observations': [], 'identity_groups': [], 'provenance': {}, 'limits': []}
    for frame in range(10):
        for identifier in ('1', '2'):
            missing = frame == 3
            evidence['observations'].append({'id': f'{identity}:frame:{frame}:model-object:{identifier}',
                'frame': frame, 'time_s': frame / 20 + .017, 'object_id': 'sam-' + identifier, 'raw_model_id': identifier,
                'phase': 'pre_initialization' if frame < 4 else 'initialization' if frame == 4 else 'post_initialization',
                'proposal_status': 'missing' if missing else 'returned',
                'geometry': None if missing else {'box_xywh': [100 + frame, 110, 12, 20], 'center_px': [106 + frame, 120]}})
    equipment = adapt_segmentation(evidence)
    if unknown_phase:
        for row in equipment['evidence']:
            row['kind'] = 'other_image_provider'
            row['payload'] = {}
    if withheld:
        human['reference_evidence'] = [row for row in human['reference_evidence'] if not row['evidence_id'].endswith(':5')]
        for row in body['observations']:
            if row['frame'] == 5:
                row['partition'] = 'audit'
        for row in equipment['observations']:
            if row['frame'] == 5:
                row.update(partition='audit')
        for row in equipment['evidence']:
            if row['frame'] == 5:
                row['payload']['raw_model_id'] = 'WITHHELD_SENTINEL_ID'
                row['payload']['phase'] = 'WITHHELD_SENTINEL_PHASE'
        equipment['relationships'] = [{'id': 'withheld-group', 'type': 'possible_association_ambiguity',
            'objects': ['sam-1', 'sam-2'], 'status': 'inferred',
            'evidence_ids': [row['id'] for row in equipment['evidence'] if row['frame'] == 5],
            'interval_s': [5 / 20 + .017, 5 / 20 + .017], 'attributes': {'sentinel': 'WITHHELD_GROUP_ATTRIBUTES'}}]
    return compose(body, equipment, human)


def compose(body, equipment, human):
    body_raw, equipment_raw = raw(body), raw(equipment)
    bsha, esha = hashlib.sha256(body_raw).hexdigest(), hashlib.sha256(equipment_raw).hexdigest()
    binding, construction = build_shared_world_binding({bsha: body_raw, esha: equipment_raw},
        source_sha256=human['source_sha256'], world_ids=['human', 'weight'],
        entity_refs={'human': (bsha, 'person-0'), 'weight': (esha, 'sam-1')})
    refs = [{'index': i, 'source_frame': int(row['evidence_id'].rsplit(':', 1)[1]), **row}
            for i, row in enumerate(human['reference_evidence'])]
    return {'construction_evidence': construction, 'source_binding': binding, 'body_references': refs,
            'observation_documents': {esha: equipment_raw}}, body, equipment, human


def source_tracks(value):
    return {track['entity_ref']['object_id']: track for source in value['sources'] for track in source['tracks']}


def test_complete_rows_preserve_raw_alternatives_phase_and_distinct_clocks():
    inputs, *_ = fixture()
    before = deepcopy(inputs)
    receipt = ledger.prepare_motion_evidence(**inputs)
    value = json.loads(receipt); tracks = source_tracks(value)
    assert type(receipt) is bytes and value['profile'] == ledger.PROFILE
    assert set(tracks) == {'person-0', 'sam-1', 'sam-2'}
    assert tracks['sam-2']['world_entity_ids'] == []
    assert all(len(track['slots']) == 10 for track in tracks.values())
    row = tracks['sam-1']['slots'][2]
    assert row['source_time_s'] == .117 and row['body_time_s'] == .1
    assert row['body_reference']['time_s'] == .1
    assert row['phase_records'][0] == {'evidence_id': inputs['body_references'][2]['evidence_id'] + ':model-object:1',
        'raw_model_id': '1', 'phase': 'pre_initialization', 'proposal_status': 'returned'}
    assert row['measurements']['box']['value'] == [102, 110, 12, 20]
    assert tracks['sam-1']['slots'][3]['phase_records'][0]['proposal_status'] == 'missing'
    assert tracks['sam-1']['slots'][3]['measurements'] == {}
    assert tracks['sam-1']['slots'][4]['phase_records'][0]['phase'] == 'initialization'
    assert tracks['sam-1']['slots'][5]['phase_records'][0]['phase'] == 'post_initialization'
    assert inputs == before
    assert ledger.validate_motion_evidence(receipt, **inputs) == receipt


def test_withheld_phase_identity_geometry_and_group_attributes_never_leak():
    inputs, *_ = fixture(withheld=True)
    receipt = ledger.prepare_motion_evidence(**inputs)
    assert b'WITHHELD' not in receipt and b'withheld-group' not in receipt
    row = source_tracks(json.loads(receipt))['sam-1']['slots'][5]
    assert row['phase_records'] == row['evidence_refs'] == [] and row['measurements'] == {}
    assert row['visibility'] is None and 'partition_withheld' in row['unavailable_reasons']


def test_unknown_phase_is_not_guessed_from_object_id_or_frame():
    inputs, *_ = fixture(unknown_phase=True)
    value = json.loads(ledger.prepare_motion_evidence(**inputs))
    assert all(row['phase_records'] == [] and 'provider_phase_unavailable' in row['unavailable_reasons']
               for row in source_tracks(value)['sam-1']['slots'])


def test_metadata_tail_is_retained_without_native_time_or_observation():
    inputs, *_ = fixture(metadata_tail=True)
    value = json.loads(ledger.prepare_motion_evidence(**inputs))
    row = source_tracks(value)['person-0']['slots'][10]
    assert row['source_time_s'] is row['body_time_s'] is row['body_reference'] is None
    assert row['observation_id'] is None and row['admission'] == 'not_decoded'
    assert row['unavailable_reasons'] == ['metadata_only_not_decoded']


def test_absent_body_reference_does_not_filter_equipment_slot():
    inputs, *_ = fixture()
    inputs['body_references'].pop(6)
    for i, row in enumerate(inputs['body_references']): row['index'] = i
    value = json.loads(ledger.prepare_motion_evidence(**inputs))
    row = source_tracks(value)['sam-1']['slots'][6]
    assert row['body_reference'] is None and row['body_time_s'] == .3
    assert row['measurements'] and row['phase_records']
    assert 'body_reference_unavailable' in row['unavailable_reasons']


@pytest.mark.parametrize('damage', ['raw_bytes', 'raw_pin', 'missing_raw', 'extra_raw', 'construction', 'native_time', 'reference', 'duplicate_slot'])
def test_exact_document_projection_clock_and_inventory_tampering_rejected(damage):
    inputs, *_ = fixture()
    pin = next(iter(inputs['observation_documents']))
    if damage == 'raw_bytes': inputs['observation_documents'][pin] += b' '
    elif damage == 'raw_pin': inputs['observation_documents']['f' * 64] = inputs['observation_documents'].pop(pin)
    elif damage == 'missing_raw': inputs['observation_documents'] = {}
    elif damage == 'extra_raw': inputs['observation_documents']['f' * 64] = inputs['observation_documents'][pin]
    elif damage == 'reference': inputs['body_references'][3]['time_s'] += .001
    else:
        construction = json.loads(inputs['construction_evidence'])
        if damage == 'native_time':
            source = next(row for row in construction['sources'] if row['observation_schema_version'] == 4)
            source['source']['frame_times_s'][3] += .001
        else:
            source = next(row for row in construction['sources'] if row['observation_schema_version'] == 3)
            if damage == 'construction': source['observations'][0]['measurements']['box']['value'][0] += 1
            else: source['observations'][1] = deepcopy(source['observations'][0])
        inputs['construction_evidence'] = raw(construction)
    with pytest.raises(ValueError): ledger.prepare_motion_evidence(**inputs)


@pytest.mark.parametrize('damage', ['phase', 'raw_id', 'status', 'row_identity'])
def test_even_resealed_invalid_admitted_phase_records_are_rejected(damage):
    _, body, equipment, human = fixture()
    payload = equipment['evidence'][0]['payload']
    if damage == 'phase': payload['phase'] = 'inferred_held'
    elif damage == 'raw_id': payload['raw_model_id'] = 1
    elif damage == 'status': payload['proposal_status'] = 'missing'
    else: payload['object_id'] = 'sam-other'
    inputs, *_ = compose(body, equipment, human)
    with pytest.raises(ValueError): ledger.prepare_motion_evidence(**inputs)


def test_saved_ledger_must_be_rederived_not_merely_shape_checked():
    inputs, *_ = fixture()
    value = json.loads(ledger.prepare_motion_evidence(**inputs))
    source_tracks(value)['sam-1']['slots'][0]['phase_records'][0]['phase'] = 'post_initialization'
    with pytest.raises(ValueError, match='source rederivation'):
        ledger.validate_motion_evidence(raw(value), **inputs)


def test_input_output_slot_and_time_bounds(monkeypatch):
    inputs, *_ = fixture()
    with pytest.raises(ValueError): ledger.prepare_motion_evidence(**inputs, max_seconds=11)
    with pytest.raises(ValueError): ledger.prepare_motion_evidence(**inputs, max_seconds=True)
    monkeypatch.setattr(ledger, 'MAX_SLOTS', 29)
    with pytest.raises(ValueError, match='slot bound'): ledger.prepare_motion_evidence(**inputs)
    monkeypatch.setattr(ledger, 'MAX_SLOTS', 21600)
    monkeypatch.setattr(ledger, 'MAX_OUTPUT_BYTES', 10)
    with pytest.raises(ValueError, match='output exceeds'): ledger.prepare_motion_evidence(**inputs)
    monkeypatch.setattr(ledger, 'MAX_INPUT_BYTES', 10)
    with pytest.raises(ValueError, match='input byte bound'): ledger.prepare_motion_evidence(**inputs)


def test_raw_identity_changes_remain_explicit_not_merged_or_relabelled():
    _, body, equipment, human = fixture()
    equipment['evidence'][8]['payload']['raw_model_id'] = 'unresolved-other-model-id'
    inputs, *_ = compose(body, equipment, human)
    slots = source_tracks(json.loads(ledger.prepare_motion_evidence(**inputs)))['sam-1']['slots']
    assert slots[4]['phase_records'][0]['raw_model_id'] == 'unresolved-other-model-id'
    assert slots[3]['phase_records'][0]['raw_model_id'] == slots[5]['phase_records'][0]['raw_model_id'] == '1'


def test_module_does_not_import_native_physics_or_inference_runtime():
    script = '''import builtins
original=builtins.__import__
def guarded(name,*args,**kwargs):
    if name.split('.')[0] in {'mujoco','cv2','mediapipe','torch','openai'}: raise AssertionError(name)
    return original(name,*args,**kwargs)
builtins.__import__=guarded
import weft.moving_equipment_evidence
'''
    result = subprocess.run([sys.executable, '-c', script], capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr
