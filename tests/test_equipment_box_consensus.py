"""Source-only inference and real metadata joins; native admission boundaries stay mocked."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import random

import pytest

from weft import equipment_box_consensus as consensus
from weft import equipment_proposal as producer
from weft import equipment_proposal_config as config_api
from weft import shared_world_pipeline as pipeline
from weft import shared_world_reconstruction as reconstruction
from weft.shared_world_evidence import build_shared_world_binding
from test_equipment_depth_prior import SOURCE, pure_inputs, raw, seal


def inputs(monkeypatch, *, count=12, fps=10, ratio_mode=False, shifts=None, automatic=True):
    base, config, _, documents, checks = pure_inputs(monkeypatch, ratio_mode=ratio_mode)
    original = json.loads(next(iter(documents.values())))
    document = deepcopy(original); document['source'].update(fps=fps, decoded_frames=count, metadata_frames=count + 1)
    document['observations'], document['evidence'] = [], []
    templates = {row['object_id']: row for row in original['observations'] if row['frame'] == 4}
    evidence_templates = {row['object_id']: row for row in original['evidence'] if row['frame'] == 4}
    frames = list(range(0, count, 2)); human = base['human_spec']
    human['reference'] = [{'time_s': f / fps, 'joints': {}} for f in frames]
    human['root_reference'] = [{'time_s': f / fps, 'translation': [f * .01, 0., 0.]} for f in frames]
    human['reference_evidence'] = [{'time_s': f / fps, 'evidence_id': f'{SOURCE}:frame:{f}'} for f in frames]
    for frame in range(count):
        for name in templates:
            row, evidence = deepcopy(templates[name]), deepcopy(evidence_templates[name])
            identifier = f'{SOURCE}:frame:{frame}' if name == 'person' else f'{SOURCE}:frame:{frame}:object:{name}'
            row.update(id=f'{SOURCE}:observation:{frame}:{name}', frame=frame, time_s=frame/fps, evidence_ids=[identifier])
            evidence.update(id=identifier, frame=frame, time_s=frame/fps)
            if name == 'person':
                for point in row['measurements']['landmarks']['value']:
                    point['position'][0] += frame - 4
            elif shifts is not None:
                row['measurements']['box']['value'][0] += shifts[frame]
            document['observations'].append(row); document['evidence'].append(evidence)
    return repin(base, config, document, checks, automatic=automatic)


def repin(base, config, document, checks, *, automatic=True):
    source = raw(document); pin = hashlib.sha256(source).hexdigest(); documents = {pin: source}
    base['source_binding'], _ = build_shared_world_binding(documents, source_sha256=SOURCE,
        world_ids=['human', 'dummy'], entity_refs={'human': (pin, 'person'), 'dummy': None})
    binding, construction = build_shared_world_binding(documents, source_sha256=SOURCE,
        world_ids=['human', 'weight', 'pad'], entity_refs={'human': (pin, 'person'), 'weight': (pin, 'weight'), 'pad': (pin, 'pad')})
    if automatic:
        for choice in config['equipment']:
            choice.pop('source_frame', None); choice['frame_selection'] = {'method': config_api.CONSENSUS_METHOD}
    kwargs = {'source_binding': binding, 'construction_evidence': construction, 'human_document': raw(base['human_spec'])}
    return base, config, kwargs, documents, checks


def references(base):
    return [{'index': i, 'source_frame': int(row['evidence_id'].rpartition(':')[2]), **row}
            for i, row in enumerate(base['human_spec']['reference_evidence'])]


def select(base, kwargs, *, ids=None, **options):
    return consensus.select_equipment_box_frames(kwargs['construction_evidence'], source_binding=kwargs['source_binding'],
        body_references=references(base), object_ids=ids or ['weight', 'pad'], **options)


def test_stationary_sequence_chooses_earliest_tied_observed_frame_and_keeps_tail(monkeypatch):
    base, config, kwargs, _, _ = inputs(monkeypatch)
    original = deepcopy((base, config, kwargs))
    result = select(base, kwargs)
    for name, receipt in result.items():
        assert consensus.validate_selection_receipt(receipt) == receipt
        assert receipt['status'] == 'selected' and receipt['selected']['source_frame'] == 0
        assert receipt['statistics']['eligible_count'] == 6 and len(receipt['slots']) == 13
        assert receipt['slots'][-1][9] == 'metadata_only_not_decoded'
        assert receipt['slots'][1][9] == 'not_original_body_reference'
        assert receipt['statistics']['edge_range_fraction'] == [0.] * 4
        assert receipt['physical_stationarity'] == receipt['physical_error_coverage'] == 'unknown'
        assert receipt['world_object_id'] == name
    assert (base, config, kwargs) == original


def test_noisy_sequence_exact_medoid_matches_independent_pairwise_costs(monkeypatch):
    random.seed(19)
    shifts = [random.randrange(-2, 3) for _ in range(18)]
    base, _, kwargs, _, _ = inputs(monkeypatch, count=18, shifts=shifts)
    receipt = select(base, kwargs)['weight']
    from fractions import Fraction
    rows = [r for r in receipt['slots'] if r[9] is None]
    points = [[Fraction(v)/size for v, size in zip(row[10], [640, 480, 640, 480])] for row in rows]
    costs = [sum(sum(abs(a-b) for a,b in zip(point, other)) for other in points) for point in points]
    chosen = min(range(len(rows)), key=lambda i: (costs[i], rows[i][0]))
    assert receipt['status'] == 'selected'
    assert receipt['selected']['source_frame'] == rows[chosen][0]
    assert [r[12] for r in rows] == list(map(float, costs))
    assert all(row[11] == [float(a-b) for a,b in zip(point, points[chosen])] for row, point in zip(rows, points))


@pytest.mark.parametrize('shifts,reason', [
    ([0.] * 10 + [30., 30.], 'edge_range_exceeds_image_policy'),
    ([float(i) for i in range(12)], 'ordered_image_drift'),
    ([0.] * 6 + [8.] * 6, 'competing_image_cohorts'),
])
def test_all_outliers_motion_and_competing_cohorts_cause_abstention(monkeypatch, shifts, reason):
    base, _, kwargs, _, _ = inputs(monkeypatch, shifts=shifts)
    receipt = select(base, kwargs)['weight']
    assert receipt['status'] == 'unavailable' and receipt['selected'] is None
    assert reason in receipt['reasons'] and receipt['diagnostic_medoid'] is not None
    assert receipt['statistics']['eligible_count'] == 6
    assert len([r for r in receipt['slots'] if r[10] is not None]) == 6
    assert consensus.validate_selection_receipt(receipt) == receipt


def test_scaling_changes_edges_and_cannot_hide_behind_stable_center(monkeypatch):
    base, config, _, documents, checks = inputs(monkeypatch)
    doc = json.loads(next(iter(documents.values())))
    for row in doc['observations']:
        if row['object_id'] == 'weight':
            row['measurements']['box']['value'][0] -= row['frame']
            row['measurements']['box']['value'][2] += 2 * row['frame']
    base, _, kwargs, _, _ = repin(base, config, doc, checks)
    assert 'ordered_image_drift' in select(base, kwargs)['weight']['reasons']


@pytest.mark.parametrize('mode', ['short', 'insufficient', 'none', 'partial'])
def test_insufficient_short_and_partial_evidence_remain_explicit(monkeypatch, mode):
    base, config, _, documents, checks = inputs(monkeypatch, fps=100 if mode == 'short' else 10)
    doc = json.loads(next(iter(documents.values())))
    for row in doc['observations']:
        if row['object_id'] == 'weight':
            if mode in ('none', 'insufficient') and (mode == 'none' or row['frame'] > 2):
                row.update(visibility='not_detected', measurements={}, evidence_ids=[])
            elif mode == 'partial':
                row['visibility'] = 'partial'
    base, _, kwargs, _, _ = repin(base, config, doc, checks)
    receipt = select(base, kwargs)['weight']
    assert receipt['status'] == ('selected' if mode == 'partial' else 'unavailable')
    if mode == 'short': assert 'insufficient_time_span' in receipt['reasons']
    if mode in ('none', 'insufficient'): assert 'insufficient_eligible_frames' in receipt['reasons']
    if mode == 'partial': assert all(row[5] == 'partial' for row in receipt['slots'] if row[9] is None)
    assert consensus.validate_selection_receipt(receipt) == receipt


def test_withheld_slots_never_supply_coordinates_and_all_are_retained(monkeypatch):
    base, _, kwargs, _, _ = inputs(monkeypatch)
    document = json.loads(kwargs['construction_evidence'])
    for row in document['sources'][0]['observations']:
        if row['object_id'] == 'weight' and row['frame'] == 4:
            row.update(partition='final_evaluation', admission='withheld', visibility=None, measurements={}, evidence_refs=[])
    kwargs['construction_evidence'] = raw(document)
    receipt = select(base, kwargs)['weight']; row = receipt['slots'][4]
    assert row[3:6] == ['final_evaluation', 'withheld', None]
    assert row[9:] == ['partition_withheld', None, None, None]
    assert receipt['statistics']['eligible_count'] == 5
    document['sources'][0]['observations'][4 * 3 + 1]['measurements'] = {'box': {'value': [999, 999, 999, 999]}}
    kwargs['construction_evidence'] = raw(document)
    with pytest.raises(ValueError, match='withheld payload'):
        select(base, kwargs)


@pytest.mark.parametrize('field', ['source_sha256', 'document', 'body_clock', 'row_clock', 'duplicate', 'entity', 'body_evidence'])
def test_exact_source_body_entity_and_clock_conflicts_reject(monkeypatch, field):
    base, _, kwargs, _, _ = inputs(monkeypatch)
    doc = json.loads(kwargs['construction_evidence'])
    if field == 'source_sha256': kwargs['source_binding']['source_sha256'] = 'b' * 64
    if field == 'document': kwargs['source_binding']['evidence_documents'] = {'b' * 64: 'evidence/new.json'}
    if field == 'body_clock': base['human_spec']['reference_evidence'][1]['time_s'] += .01
    if field == 'row_clock': doc['sources'][0]['observations'][1]['time_s'] += .01
    if field == 'duplicate': doc['sources'][0]['observations'][1] = deepcopy(doc['sources'][0]['observations'][0])
    if field == 'entity': kwargs['source_binding']['entity_bindings'][1]['observation_entity_id'] = 'missing'
    if field == 'body_evidence': base['human_spec']['reference_evidence'][1]['evidence_id'] += '-new'
    kwargs['construction_evidence'] = raw(doc)
    with pytest.raises(ValueError): select(base, kwargs)


@pytest.mark.parametrize('choice', [None, {}, {'method': 'other'}, {'method': config_api.CONSENSUS_METHOD, 'limit': .1}])
def test_only_explicit_fixed_frame_policy_is_accepted(choice):
    with pytest.raises(config_api.EquipmentProposalError, match='frame-selection'):
        config_api.validate_frame_selection(choice)


@pytest.mark.parametrize('ratio_mode', [False, True])
def test_config_preserves_manual_and_auto_forms_without_resolution(monkeypatch, ratio_mode):
    _, config, _, _, _ = inputs(monkeypatch, ratio_mode=ratio_mode)
    assert config_api.validate_proposal_config(config) == config
    mixed = deepcopy(config); mixed['equipment'][0].pop('frame_selection'); mixed['equipment'][0]['source_frame'] = 2
    assert config_api.validate_proposal_config(mixed) == mixed
    mixed['equipment'][0]['frame_selection'] = {'method': config_api.CONSENSUS_METHOD}
    with pytest.raises(config_api.EquipmentProposalError): config_api.validate_proposal_config(mixed)


def test_caller_deadline_input_and_output_caps_do_not_trim(monkeypatch):
    base, _, kwargs, _, _ = inputs(monkeypatch)
    with pytest.raises(TimeoutError): select(base, kwargs, deadline=0.)
    monkeypatch.setattr(consensus, 'MAX_OUTPUT_BYTES', 100)
    with pytest.raises(ValueError, match='none were trimmed'): select(base, kwargs)
    monkeypatch.setattr(consensus, 'MAX_INPUT_BYTES', 100)
    with pytest.raises(ValueError, match='bounded exact'): select(base, kwargs)


@pytest.mark.parametrize('ratio_mode', [False, True])
def test_real_producer_and_constructor_rederive_auto_receipts_without_changing_body(monkeypatch, ratio_mode):
    base, config, kwargs, _, checks = inputs(monkeypatch, ratio_mode=ratio_mode)
    original = deepcopy((base, config, kwargs))
    preview = select(base, kwargs)
    candidate = producer.build_equipment_proposal(base, config, **kwargs)
    scene = reconstruction.construct_equipment_world(base, candidate, **kwargs, target_object_id='weight')
    assert (base, config, kwargs) == original and len(checks) == 1
    for item, obj in zip(candidate['equipment'], scene['objects']):
        assert item['frame_selection_receipt'] == preview[item['object_id']]
        assert obj['provenance']['frame_selection_reference'] == consensus.selection_reference(preview[item['object_id']])
        assert 'frame_selection_receipt' not in obj['provenance']
        assert item['source_frame'] == 0
    for key in ('human_spec', 'bounds', 'gravity', 'contact', 'dt', 'substeps', 'coordinate_frame'):
        assert scene[key] == base[key]
    assert candidate['proposer']['configuration_canonical_sha256'] == producer.canonical_hash(config)


@pytest.mark.parametrize('mutation', ['choice', 'receipt_slot', 'receipt_policy', 'receipt_pin', 'receipt_absent', 'declaration_absent', 'receipt_type'])
def test_constructor_does_not_trust_authored_or_incomplete_selection_reports(monkeypatch, mutation):
    base, config, kwargs, _, checks = inputs(monkeypatch)
    candidate = producer.build_equipment_proposal(base, config, **kwargs)
    item = candidate['equipment'][0]
    if mutation == 'choice': item['source_frame'] = 2
    if mutation == 'receipt_slot': item['frame_selection_receipt']['slots'][0][10][0] += .1
    if mutation == 'receipt_policy': item['frame_selection_receipt']['policy']['maximum_edge_range_fraction'] = .5
    if mutation == 'receipt_pin': item['frame_selection_receipt']['construction_evidence_sha256'] = 'b'*64
    if mutation == 'receipt_type': item['frame_selection_receipt']['schema_version'] = True
    if mutation == 'receipt_absent': item.pop('frame_selection_receipt')
    if mutation == 'declaration_absent': candidate['proposer'].pop('frame_selection')
    with pytest.raises(ValueError, match='frame-selection'):
        reconstruction.construct_equipment_world(base, candidate, **kwargs, target_object_id='weight')
    assert not checks


def test_no_fallback_after_selection_geometry_failure(monkeypatch):
    base, config, kwargs, _, _ = inputs(monkeypatch)
    calls = []
    def reject(*args, **kw):
        calls.append(args)
        raise ValueError('existing geometric domain rejects')
    monkeypatch.setattr(producer, 'floor_box_from_bounds', reject)
    with pytest.raises(ValueError, match='geometric domain') as captured:
        producer.build_equipment_proposal(base, config, **kwargs)
    assert len(calls) == 1 and captured.value.frame_selection_receipts['weight']['selected']['source_frame'] == 0


@pytest.mark.parametrize('unavailable', [False, True])
def test_public_packages_copy_receipts_and_only_auto_dependency(monkeypatch, tmp_path, unavailable):
    shifts = [0.] * 10 + [40., 40.] if unavailable else None
    base, config, kwargs, documents, checks = inputs(monkeypatch, shifts=shifts)
    body, equipment = tmp_path/'body', tmp_path/'equipment'; body.mkdir(); equipment.mkdir()
    (body/'scene.json').write_bytes(raw(base)); (body/'human.json').write_bytes(kwargs['human_document'])
    for sha, name in base['source_binding']['evidence_documents'].items():
        path = body/name; path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(documents[sha])
    (equipment/'scene-observations.json').write_bytes(next(iter(documents.values())))
    (equipment/'report.json').write_bytes(raw({'status': 'evidence_only', 'physics_ready': False}))
    seal(body); seal(equipment); config_path = tmp_path/'config.json'; config_path.write_bytes(raw(config))
    report = producer.prepare_equipment_proposal(body, equipment, config_path, tmp_path/'proposal')
    assert (tmp_path/'proposal/input-config.json').read_bytes() == raw(config)
    receipt = json.loads((tmp_path/'proposal/frame-selection.json').read_bytes())
    assert receipt == select(base, kwargs)
    assert (tmp_path/'proposal/implementation/equipment_box_consensus.py').read_bytes() == Path(consensus.__file__).read_bytes()
    if unavailable:
        assert report['status'] == 'unsupported_input' and report['reasons'][0]['code'] == 'frame_selection_unavailable'
        assert not checks and not (tmp_path/'proposal/geometry-proposal.json').exists()
    else:
        assert report['status'] == 'proposal_ready', report['reasons']
        pipeline.construct_source_equipment_run(body, tmp_path/'proposal', equipment, tmp_path/'candidate', target_object_id='weight')
        assert json.loads((tmp_path/'candidate/frame-selection.json').read_bytes()) == receipt
        scene = json.loads((tmp_path/'candidate/scene.json').read_bytes())
        for obj in scene['objects']:
            consensus.validate_selection_reference(obj['provenance']['frame_selection_reference'], receipt[obj['id']])
        assert (tmp_path/'candidate/implementation/equipment_box_consensus.py').read_bytes() == Path(consensus.__file__).read_bytes()
        assert (tmp_path/'candidate/implementation/equipment_proposal_config.py').read_bytes() == Path(config_api.__file__).read_bytes()
        assert len(checks) == 2


@pytest.mark.parametrize('bad_metadata_count', [False, True])
def test_separate_original_body_and_equipment_clocks_are_preserved(monkeypatch, bad_metadata_count):
    base, _, kwargs, _, _ = inputs(monkeypatch)
    doc = json.loads(kwargs['construction_evidence']); source = doc['sources'][0]
    equipment = deepcopy(source); new_sha = 'd' * 64
    equipment['document_sha256'] = new_sha
    equipment['source'].update(sha256='b' * 64, metadata_frames=13 if bad_metadata_count else 12,
                              frame_times_s=[f / 10 + .075 for f in range(12)])
    equipment['source'].pop('fps'); equipment['observation_schema_version'] = 3
    equipment['time'].update(basis='decoder_presentation_timestamps', origin='submitted_video_start', status='observed')
    for collection in ('objects', 'observations', 'evidence'):
        equipment[collection] = [r for r in equipment[collection] if r.get('object_id', r.get('id')) != 'person']
        source[collection] = [r for r in source[collection] if r.get('object_id', r.get('id')) == 'person']
    for row in equipment['objects']:
        row['entity_ref']['document_sha256'] = new_sha
    for row in equipment['observations']:
        row['time_s'] = equipment['source']['frame_times_s'][row['frame']]
        row['entity_ref']['document_sha256'] = new_sha
        row['evidence_refs'][0]['document_sha256'] = new_sha
    for row in equipment['evidence']:
        row['document_sha256'] = new_sha; row['time_s'] = equipment['source']['frame_times_s'][row['frame']]
    doc['sources'].append(equipment); binding = kwargs['source_binding']
    binding['evidence_documents'][new_sha] = 'evidence/' + new_sha + '.json'
    for entity in binding['entity_bindings'][1:]: entity['observation_document_sha256'] = new_sha
    binding['source_equivalences'] = [{'original_source_sha256': SOURCE, 'derivative_source_sha256': 'b'*64,
        'bundle_manifest_sha256': 'c'*64, 'bundle_path': 'source-mappings/' + 'c'*64}]
    kwargs['construction_evidence'] = raw(doc)
    if bad_metadata_count:
        with pytest.raises(ValueError, match='timestamp table'):
            select(base, kwargs)
        return
    receipt = select(base, kwargs)['weight']
    assert receipt['selected']['time_s'] == .075 and receipt['selected']['body_time_s'] == 0.
    assert receipt['source_clock']['basis'] == 'decoder_presentation_timestamps'
    assert receipt['observation_source_sha256'] == 'b' * 64 and receipt['source_sha256'] == SOURCE


def test_maximum_saved_metadata_case_keeps_complete_cached_slots(monkeypatch, record_property):
    import time
    base, _, kwargs, _, _ = inputs(monkeypatch, count=1798, fps=30)
    started = time.monotonic(); result = select(base, kwargs); seconds = time.monotonic() - started
    assert all(len(receipt['slots']) == 1799 and receipt['statistics']['eligible_count'] == 899 for receipt in result.values())
    assert all(receipt['status'] == 'selected' for receipt in result.values())
    size = len(consensus._bytes(result)); assert size <= consensus.MAX_OUTPUT_BYTES and seconds < consensus.MAX_SECONDS
    record_property('authored_metadata_seconds', seconds)
    record_property('cached_receipt_bytes', size)
    record_property('scope', 'Two objects, 1799 declared slots, 899 body references; authored metadata, not a real source')
