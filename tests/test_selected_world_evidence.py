"""Restrict original-frame construction access without rewriting partitions."""

import importlib
from copy import deepcopy
from hashlib import sha256
import json
import time

import pytest

from test_multiview_landmark_evidence import local_only
from test_world_evidence import SOURCE, document, inputs
from test_equipment_silhouette_evidence import fixture as silhouette_fixture
from weft import world_evidence
from weft.world_evidence import load_world_evidence


def other_source(doc, identity='b' * 64):
    return json.loads(json.dumps(doc).replace(SOURCE, identity))


def test_selected_frames_restrict_real_failure_shape_without_relabeling():
    human = document(version=2, first=152, partitions=('construction', 'construction', 'unassigned', 'unassigned'))
    equipment = other_source(document(version=2, first=152, partitions=('construction',) * 4))
    registry = load_world_evidence(inputs(human)[2] | inputs(equipment)[2])
    aliases = {'b' * 64: SOURCE}
    with pytest.raises(ValueError, match='frame 154'):
        registry.construction_bytes(source_aliases=aliases)
    output = json.loads(registry.construction_bytes(source_aliases=aliases, source_frame_selection={SOURCE: [152]}))
    assert sum(len(source['observations']) for source in output['sources']) == 16
    for source in output['sources']:
        for row in source['observations']:
            assert row['admission'] == ('included' if row['frame'] == 152 else 'withheld')
            if row['frame'] != 152:
                assert row['measurements'] == {} and row['evidence_refs'] == []
    assert {row['partition'] for source in output['sources'] for row in source['observations'] if row['frame'] == 154} == {'construction', 'unassigned'}


def test_silhouette_selection_precedes_excluded_partition_join(monkeypatch):
    api = importlib.import_module('weft.equipment_silhouette_evidence')
    def exclude_later(doc):
        for row in doc['observations']:
            if row['frame'] == 4:
                row.update(partition='unassigned', visibility='unknown', measurements={}, evidence_ids=[])
    prepared, args, calls, _ = silhouette_fixture(api, observation_change=exclude_later)
    original = type(prepared).frame
    queried = []
    def frame(self, index):
        assert index in (0, 2)
        queried.append(index)
        return original(self, index)
    monkeypatch.setattr(type(prepared), 'frame', frame)
    result = api.prepare_equipment_silhouette_evidence(prepared, construction_frames=[0, 2], **args)
    assert queried == [0, 2] and len(calls) == 2
    assert result.describe()['counts']['source_frames'] == 7
    pair = result.frame(4)['pairs'][0]
    assert pair['reason'] == 'excluded_by_explicit_frame_selection'
    assert pair['human_partition'] == 'unassigned' and pair['equipment_partition'] == 'construction'
    assert pair['pixel'] is None and pair['model_identity'] is None and pair['contact'] == 'unknown'
    disclosure = result.describe()['canonical_frame_selection']
    assert disclosure == {'profile': 'restrict-only-original-frame-selection-v1',
        'frames_by_canonical_source_sha256': {json.loads(prepared.pose_document)['source_sha256']: [0, 2]},
        'partition_consistency_scope': 'selected_frames_only', 'excluded_partition_labels': 'preserved_not_reconciled',
        'source_aliases': {}, 'maximum_frames_per_source': 1800}


@pytest.mark.parametrize('version,first,expected', [
    (1, 0, '6e593d9cc795a3b71217dd1f3cc177dd28dde7adfda6c0c275dd8923d08d0159'),
    (2, 10000, 'e28511955e1aad7ca2297d8006d85f14289aedcd251bafdf082cc4786b64a313'),
    (3, 0, 'b93a971443b49f6fde6f6385694e2f6417933aeeff09d2fea17df2190fe63105'),
])
def test_default_serialized_export_remains_exact(version, first, expected):
    registry = load_world_evidence(inputs(document(version=version, first=first))[2])
    assert sha256(registry.construction_bytes()).hexdigest() == expected
    assert registry.construction_bytes() == registry.construction_bytes(source_frame_selection=None)


def test_default_alias_export_remains_exact():
    doc = document()
    registry = load_world_evidence(inputs(doc)[2] | inputs(other_source(doc))[2])
    assert sha256(registry.construction_bytes(source_aliases={'b' * 64: SOURCE})).hexdigest() == 'd437fff4438b657427884379138581225d9c4186fbe29bfa41d4de4e08443853'


def test_default_silhouette_metadata_and_frames_remain_exact():
    api = importlib.import_module('weft.equipment_silhouette_evidence')
    prepared, args, _, _ = silhouette_fixture(api)
    result = api.prepare_equipment_silhouette_evidence(prepared, **args)
    def encode(value):
        return json.dumps(value, sort_keys=True, allow_nan=False, separators=(',', ':')).encode()
    assert sha256(encode(result.describe())).hexdigest() == '8d62719c09aeb88cb44c4dab3f632420fc867ae037cd7b50f9160078f23abea8'
    assert sha256(b'\n'.join(encode(result.frame(index)) for index in range(7))).hexdigest() == '6edcfd06c59311eb049922bf5950eaa283eebf9bb74bd33be6fcd0ced8df7954'


@pytest.mark.parametrize('selection', [
    {}, [], {SOURCE: {0}}, {SOURCE: '0'}, {SOURCE: [True]}, {SOURCE: [0.]},
    {SOURCE: [-1]}, {SOURCE: [6]}, {SOURCE: [0, 0]}, {SOURCE: [2, 0]},
    {SOURCE: list(range(1801))}, {'b' * 64: [0]}, {True: [0]},
])
def test_malformed_unknown_out_of_range_or_unbounded_selection_rejected(selection):
    registry = load_world_evidence(inputs(document())[2])
    with pytest.raises(ValueError, match='Frame selection'):
        registry.construction_bytes(source_frame_selection=selection)


@pytest.mark.parametrize('partition', ['audit', 'unassigned', 'generation'])
def test_selected_aliased_conflicts_still_fail(partition):
    left = document(partitions=('construction', 'construction'))
    right = other_source(document(partitions=('construction', partition)))
    registry = load_world_evidence(inputs(left)[2] | inputs(right)[2])
    with pytest.raises(ValueError, match='frame 1'):
        registry.construction_bytes(source_aliases={'b' * 64: SOURCE}, source_frame_selection={SOURCE: [1]})
    with pytest.raises(ValueError, match='frame 1'):
        registry.construction_bytes(source_aliases={'b' * 64: SOURCE})


def test_agreed_selected_audit_remains_withheld():
    doc = document(partitions=('audit', 'construction'))
    registry = load_world_evidence(inputs(doc)[2] | inputs(other_source(doc))[2])
    result = json.loads(registry.construction_bytes(source_aliases={'b' * 64: SOURCE}, source_frame_selection={SOURCE: [0]}))
    assert all(row['admission'] == 'withheld' for source in result['sources'] for row in source['observations'])
    assert all(source['evidence'] == [] for source in result['sources'])


def test_every_mixed_source_group_requires_its_own_explicit_selection():
    first = document(version=3)
    second = other_source(document(version=3))
    registry = load_world_evidence(inputs(first)[2] | inputs(second)[2])
    with pytest.raises(ValueError, match='exactly the canonical'):
        registry.construction_bytes(source_frame_selection={SOURCE: [0]})
    result = json.loads(registry.construction_bytes(allowed_partitions=('construction', 'generation'),
        source_frame_selection={SOURCE: [0], 'b' * 64: [1]}))
    for source in result['sources']:
        expected_frame = 0 if source['source']['sha256'] == SOURCE else 1
        assert {row['frame'] for row in source['observations'] if row['admission'] == 'included'} == {expected_frame}
        assert source['source']['frame_times_s'] == first['source']['frame_times_s']
        assert source['time'] == first['time']
    with pytest.raises(ValueError, match='exactly the canonical'):
        registry.construction_bytes(source_aliases={'b' * 64: SOURCE}, source_frame_selection={SOURCE: [0], 'b' * 64: [1]})


def test_original_interval_ids_not_relative_offsets_or_uncovered_gaps():
    first = document(version=2, first=10000, partitions=('construction',) * 2)
    second = document(version=2, first=10004, partitions=('construction',) * 2)
    registry = load_world_evidence(inputs(first)[2] | inputs(second)[2])
    for wrong in (0, 10002, 10006):
        with pytest.raises(ValueError, match='original decoded frame'):
            registry.construction_bytes(source_frame_selection={SOURCE: [wrong]})
    result = json.loads(registry.construction_bytes(source_frame_selection={SOURCE: [10000, 10005]}))
    assert {row['frame'] for source in result['sources'] for row in source['observations'] if row['admission'] == 'included'} == {10000, 10005}
    assert sum(len(source['observations']) for source in result['sources']) == 8


def test_excluded_measurements_claims_and_spanning_relationships_never_export(monkeypatch):
    doc = document(partitions=('construction',) * 3)
    first, second, excluded = [doc['evidence'][index]['id'] for index in (0, 1, 2)]
    for row in doc['observations']:
        if row['frame'] > 0:
            row['measurements']['center']['value'] = [777777.125, 888888.25]
    doc['objects'][0]['properties']['unsafe'] = {'value': 'EXCLUDED_CLAIM_SENTINEL', 'status': 'inferred', 'evidence_ids': [excluded]}
    doc['objects'][0]['properties']['safe'] = {'value': 'opening', 'status': 'inferred', 'evidence_ids': [first]}
    doc['camera']['calibration'] = {'value': 'EXCLUDED_CAMERA_SENTINEL', 'status': 'inferred', 'evidence_ids': [excluded]}
    for label, interval, refs in (('opening', [0., 0.], [first, second]),
                                 ('spanning', [0., 2 / 30], [first, second]),
                                 ('unbounded', None, [first, second]),
                                 ('excluded', [1 / 30, 1 / 30], [excluded])):
        doc['relationships'].append({'id': label, 'type': 'visual_proximity_candidate', 'objects': ['person', 'ball'],
            'status': 'inferred', 'evidence_ids': refs, 'interval_s': interval, 'attributes': {'pixel': [777777.125, 888888.25]}})
    pin, raw, data = inputs(doc)
    registry = load_world_evidence(data)
    forbidden = {id(row['measurements']) for source in registry._sources for row in source.document['observations'] if row['frame'] > 0}
    original = world_evidence._thaw
    def thaw(value):
        assert id(value) not in forbidden, 'Excluded measurement reached export copying'
        return original(value)
    monkeypatch.setattr(world_evidence, '_thaw', thaw)
    output = registry.construction_bytes(source_frame_selection={SOURCE: [0]})
    result = json.loads(output)['sources'][0]
    assert '777777.125' not in output.decode() and 'SENTINEL' not in output.decode()
    assert [row['id'] for row in result['relationships']] == ['opening']
    assert result['objects'][0]['properties']['safe']['value'] == 'opening'
    assert result['objects'][0]['properties']['unsafe']['admission'] == 'withheld'
    assert result['camera']['calibration']['admission'] == 'withheld'
    assert {row['frame'] for row in result['evidence']} == {0}
    assert registry.source_bytes(pin) is raw
    assert all(row['partition'] == 'construction' for row in result['observations'])


def test_empty_selection_withholds_conflicts_without_reconciling_labels():
    left = document(partitions=('construction',))
    right = other_source(document(partitions=('audit',)))
    registry = load_world_evidence(inputs(left)[2] | inputs(right)[2])
    result = json.loads(registry.construction_bytes(source_aliases={'b' * 64: SOURCE}, source_frame_selection={SOURCE: []}))
    assert result['source_frame_selection']['partition_consistency_scope'] == 'selected_frames_only'
    assert result['source_frame_selection']['excluded_partition_labels'] == 'preserved_not_reconciled'
    assert {row['partition'] for source in result['sources'] for row in source['observations']} == {'audit', 'construction'}
    assert all(row['admission'] == 'withheld' and row['measurements'] == {} for source in result['sources'] for row in source['observations'])


def test_excluded_invalid_unsampled_row_still_fails():
    doc = document(partitions=('construction', 'construction'))
    doc['observations'][2]['partition'] = 'unsampled'
    registry = load_world_evidence(inputs(doc)[2])
    with pytest.raises(ValueError, match='Unsampled partition'):
        registry.construction_bytes(source_frame_selection={SOURCE: [0]})


def test_silhouette_verified_alias_selection_preserves_both_clocks(tmp_path):
    from test_exercise_support_evidence import _equivalent_equipment
    pose, _, proof = _equivalent_equipment(tmp_path)
    api = importlib.import_module('weft.equipment_silhouette_evidence')
    def later(doc):
        for row in doc['observations']:
            if row['frame'] > 0 and row['partition'] != 'unsampled':
                row.update(partition='unassigned', visibility='unknown', measurements={}, evidence_ids=[])
    prepared, args, calls, _ = silhouette_fixture(api, supplied_pose=pose, proof=proof, observation_change=later)
    result = api.prepare_equipment_silhouette_evidence(prepared, construction_frames=[0], **args)
    first = result.frame(0)
    assert len(calls) == 1 and first['pose_decoder_timestamp_s'] == 0
    assert first['pairs'][0]['equipment_time_s'] == .002
    assert first['source_mapping']['original_native_pts'] == first['source_mapping']['derivative_native_pts'] == 3
    assert result.describe()['canonical_frame_selection']['source_aliases'] == {proof.derivative_sha256: proof.original_sha256}
    assert result.frame(2)['pairs'][0]['human_partition'] == 'unassigned'


def test_silhouette_selected_holdout_rejects_before_coordinates_or_raster(monkeypatch):
    api = importlib.import_module('weft.equipment_silhouette_evidence')
    prepared, args, calls, _ = silhouette_fixture(api, equipment_change=lambda doc: doc['observations'][0].update(partition='audit'))
    def denied(*args, **kwargs):
        raise AssertionError('Selected partition conflict reached coordinates or raster')
    monkeypatch.setattr(type(prepared), 'frame', denied)
    args['raster_loader'] = denied
    with pytest.raises(ValueError, match='Conflicting canonical frame partitions'):
        api.prepare_equipment_silhouette_evidence(prepared, construction_frames=[0], **args)
    assert calls == []


def test_bounded_1800_frame_selection_retains_every_slot(tmp_path):
    doc = document(partitions=('construction',))
    base_evidence, base_rows = doc['evidence'], doc['observations']
    doc['source'].update(decoded_frames=1800, metadata_frames=1800)
    doc['evidence'], doc['observations'] = [], []
    for frame in range(1800):
        for evidence, observation in zip(base_evidence, base_rows):
            record, row = deepcopy(evidence), deepcopy(observation)
            record.update(id=f"{SOURCE}:frame:{frame}:object:{record['object_id']}", frame=frame, time_s=frame / 30)
            row.update(id=f"{SOURCE}:observation:{frame}:{row['object_id']}", frame=frame, time_s=frame / 30, evidence_ids=[record['id']])
            doc['evidence'].append(record)
            doc['observations'].append(row)
    registry = load_world_evidence(inputs(doc)[2])
    started = time.perf_counter()
    output = registry.construction_bytes(source_frame_selection={SOURCE: list(range(1800))})
    elapsed = time.perf_counter() - started
    result = json.loads(output)
    assert len(result['sources'][0]['observations']) == 3600
    assert len(result['source_frame_selection']['frames_by_canonical_source_sha256'][SOURCE]) == 1800
    assert elapsed < 2.
    (tmp_path / 'selection-performance.json').write_text(json.dumps({'frames': 1800, 'object_slots': 3600,
        'export_seconds': elapsed, 'output_bytes': len(output), 'scope': 'Bounded authored registry export, not source or native work'}))
