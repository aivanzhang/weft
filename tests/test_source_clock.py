"""Analytical timing contracts; no media inference or physical truth claims."""

from copy import deepcopy
from hashlib import sha256
import json

import pytest

from weft.io import canonical_hash
from weft.source_clock import (PROFILE, clock_summary, derive_evidence, frame_times,
    validate_clock, validate_clock_binding, validate_derived_evidence, validate_summary)


def clock_fixture(*, origin=3000, decoder_offset=-.7, metadata_frames=10):
    source = 'a' * 64
    ticks = [origin + n for n in [0, 33, 67, 100, 167, 200, 233, 267, 300, 333]]
    frames = [{'id': f'{source}:frame:{i}', 'frame': i, 'time_s': i / 30,
               'decoder_timestamp_s': ticks[i] / 1000 + decoder_offset,
               'status': 'missing' if i == 6 else 'tracked', 'joints': {'left_knee': i / 10},
               'landmarks': [{'x': .2 + i / 100, 'y': .4, 'z': -.1}],
               'identity_raw': {'time_s': i / 30, 'unchanged': True}}
              for i in range(0, 10, 2)]
    refs = [{'time_s': r['time_s'], 'evidence_id': r['id'], 'joints': deepcopy(r['joints'])}
            for r in frames if r['status'] == 'tracked']
    evidence = {'schema_version': 1, 'profile': 'exercise-pose-v1', 'source_sha256': source,
                'fps': 30., 'frame_count': metadata_frames, 'decoded_frames': 10,
                'frames': frames, 'reference': refs, 'origin_time_s': 0.,
                'quality': {'passes': False, 'errors': ['Analytical retained quality failure']}}
    if metadata_frames != 10:
        evidence['container_frame_audit'] = {'encoded_video_frames': 10, 'decoded_frames': 10, 'matches': True}
    raw = json.dumps(evidence, indent=2).encode()
    binding = {'schema_version': 1, 'profile': PROFILE, 'source_sha256': source,
               'evidence_sha256': sha256(raw).hexdigest(), 'evidence_canonical_sha256': canonical_hash(evidence),
               'metadata_frames': metadata_frames, 'decoded_frames': 10, 'nominal_fps': 30.,
               'time_base': '1/1000', 'native_origin_pts': origin,
               'native_to_reconstruction_offset_s': -origin / 1000,
               'decoder_minus_native_s': decoder_offset,
               'capture_clock_status': 'unknown',
               'detector_clock_status': 'preserved_nominal_frame_index_divided_by_fps',
               'rows': [{'frame': i, 'pts': pts, 'decoder_timestamp_s': pts / 1000 + decoder_offset}
                        for i, pts in enumerate(ticks)]}
    return evidence, binding, raw


def test_native_gaps_and_signed_origins_change_only_reconstruction_time():
    evidence, binding, raw = clock_fixture()
    original = deepcopy(evidence)
    original_binding = deepcopy(binding)
    value = validate_clock(binding, evidence, evidence_sha256=sha256(raw).hexdigest())
    view = derive_evidence(evidence, value)
    assert frame_times(value) == [0, .033, .067, .1, .167, .2, .233, .267, .3, .333]
    assert [r['time_s'] for r in view['frames']] == [0, .067, .167, .233, .3]
    assert [r['time_s'] for r in view['reference']] == [0, .067, .167, .3]
    assert [r['id'] for r in view['frames']] == [r['id'] for r in evidence['frames']]
    assert view['quality'] == evidence['quality'] and view['quality']['passes'] is False
    assert view['frames'][2]['identity_raw']['time_s'] == 4 / 30
    assert view['frames'][3]['status'] == 'missing'
    for left, right in zip(view['frames'], evidence['frames']):
        assert {k: v for k, v in left.items() if k != 'time_s'} == {k: v for k, v in right.items() if k != 'time_s'}
    assert validate_derived_evidence(view) == binding
    assert evidence == original and binding == original_binding
    value['rows'][0]['pts'] += 1
    assert binding == original_binding


def test_null_sampled_decoder_time_stays_unknown_while_complete_table_is_bound():
    evidence, binding, _ = clock_fixture()
    evidence['frames'][1]['decoder_timestamp_s'] = None
    binding['evidence_canonical_sha256'] = canonical_hash(evidence)
    view = derive_evidence(evidence, binding)
    assert view['frames'][1]['decoder_timestamp_s'] is None
    assert validate_derived_evidence(view) == binding


@pytest.mark.parametrize('origin,offset', [(0, .007), (3, -.003), (-1000, 1.1)])
def test_positive_negative_and_zero_native_origin_remain_explicit(origin, offset):
    evidence, binding, _ = clock_fixture(origin=origin, decoder_offset=offset)
    view = derive_evidence(evidence, binding)
    assert view['frames'][0]['time_s'] == 0
    summary = clock_summary(binding)
    assert summary['native_origin_pts'] == origin
    assert summary['native_to_reconstruction_offset_s'] == -origin / 1000
    assert summary['decoder_minus_native_s'] == offset
    assert validate_summary(summary, source_sha256=evidence['source_sha256']) == summary


@pytest.mark.parametrize('mutation', ['reorder', 'missing', 'repeated_pts', 'origin', 'offset', 'drift',
                                     'timebase', 'boolean', 'nan', 'duration', 'unknown', 'capture', 'count'])
def test_invalid_clock_cannot_become_native_time(mutation):
    evidence, binding, _ = clock_fixture()
    if mutation == 'reorder': binding['rows'][2], binding['rows'][3] = binding['rows'][3], binding['rows'][2]
    elif mutation == 'missing': binding['rows'].pop()
    elif mutation == 'repeated_pts': binding['rows'][3]['pts'] = binding['rows'][2]['pts']
    elif mutation == 'origin': binding['native_origin_pts'] += 1
    elif mutation == 'offset': binding['native_to_reconstruction_offset_s'] = 0
    elif mutation == 'drift': binding['rows'][5]['decoder_timestamp_s'] += .001
    elif mutation == 'timebase': binding['time_base'] = '2/2000'
    elif mutation == 'boolean': binding['rows'][0]['frame'] = False
    elif mutation == 'nan': binding['decoder_minus_native_s'] = float('nan')
    elif mutation == 'duration':
        binding['rows'][-1]['pts'] = binding['native_origin_pts'] + 60001
        binding['rows'][-1]['decoder_timestamp_s'] = binding['rows'][-1]['pts'] / 1000 + binding['decoder_minus_native_s']
    elif mutation == 'unknown': binding['accept_pose_failure'] = True
    elif mutation == 'capture': binding['capture_clock_status'] = 'measured'
    else: binding['decoded_frames'] = 1801
    with pytest.raises(ValueError): validate_clock(binding, evidence)


def test_source_and_exact_evidence_hashes_are_not_interchangeable():
    evidence, binding, raw = clock_fixture()
    with pytest.raises(ValueError, match='byte hash'):
        validate_clock(binding, evidence, evidence_sha256=sha256(raw + b' ').hexdigest())
    evidence['source_sha256'] = 'b' * 64
    with pytest.raises(ValueError, match='identity'):
        validate_clock(binding, evidence)


def test_metadata_count_difference_requires_encoded_audit_and_is_not_normalized():
    evidence, binding, _ = clock_fixture(metadata_frames=12)
    assert validate_clock(binding, evidence)['metadata_frames'] == 12
    original = deepcopy(evidence)
    evidence.pop('container_frame_audit')
    binding['evidence_canonical_sha256'] = canonical_hash(evidence)
    with pytest.raises(ValueError, match='encoded/decoded'):
        validate_clock(binding, evidence)
    assert original['frame_count'] == 12 and original['decoded_frames'] == 10


@pytest.mark.parametrize('metadata_frames', [9, 11])
def test_recorded_one_frame_rounding_preserves_both_counts(metadata_frames):
    evidence, binding, _ = clock_fixture(metadata_frames=metadata_frames)
    evidence.pop('container_frame_audit')
    evidence['decode_metadata_discrepancy'] = {'metadata_frames': metadata_frames, 'decoded_frames': 10,
        'difference': 10 - metadata_frames, 'note': 'One-frame container rounding retained, not a synthetic clock slot.'}
    binding['evidence_canonical_sha256'] = canonical_hash(evidence)
    before = deepcopy(evidence)
    view = derive_evidence(evidence, binding)
    assert validate_derived_evidence(view) == binding
    assert view['frame_count'] == metadata_frames and view['decoded_frames'] == 10
    assert len(frame_times(binding)) == 10 and evidence == before
    assert view['decode_metadata_discrepancy'] == evidence['decode_metadata_discrepancy']


@pytest.mark.parametrize('mutation', ['missing', 'metadata', 'decoded', 'difference', 'boolean', 'extra', 'empty_note', 'large'])
def test_rounding_receipt_cannot_excuse_missing_or_inconsistent_clock_slots(mutation):
    evidence, binding, _ = clock_fixture(metadata_frames=11)
    evidence.pop('container_frame_audit')
    receipt = {'metadata_frames': 11, 'decoded_frames': 10, 'difference': -1, 'note': 'Retained one-frame discrepancy'}
    evidence['decode_metadata_discrepancy'] = receipt
    if mutation == 'missing': evidence.pop('decode_metadata_discrepancy')
    elif mutation == 'metadata': receipt['metadata_frames'] = 10
    elif mutation == 'decoded': receipt['decoded_frames'] = 11
    elif mutation == 'difference': receipt['difference'] = 1
    elif mutation == 'boolean': receipt['difference'] = True
    elif mutation == 'extra': receipt['accepted_without_check'] = True
    elif mutation == 'empty_note': receipt['note'] = ''
    else:
        evidence['frame_count'] = binding['metadata_frames'] = receipt['metadata_frames'] = 12
        receipt['difference'] = -2
    binding['evidence_canonical_sha256'] = canonical_hash(evidence)
    with pytest.raises(ValueError, match='encoded/decoded'):
        validate_clock(binding, evidence)


@pytest.mark.parametrize('mutation', ['pose', 'timestamp', 'nominal', 'decoder', 'reference', 'marker', 'origin', 'extra'])
def test_derived_view_cannot_hide_nonclock_changes(mutation):
    evidence, binding, _ = clock_fixture()
    view = derive_evidence(evidence, binding)
    if mutation == 'pose': view['frames'][2]['landmarks'][0]['x'] += .1
    elif mutation == 'timestamp': view['frames'][2]['time_s'] = 4 / 30
    elif mutation == 'nominal': view['source_clock']['original_times']['frames'][2] += .001
    elif mutation == 'decoder': view['frames'][2]['decoder_timestamp_s'] += .001
    elif mutation == 'reference': view['reference'][1]['evidence_id'] = view['reference'][0]['evidence_id']
    elif mutation == 'marker': view['source_clock']['original_times']['hidden'] = 1
    elif mutation == 'origin': view['origin_time_s'] = .001
    else: view['quality']['passes'] = True
    with pytest.raises(ValueError): validate_derived_evidence(view)


def test_cannot_chain_native_views_or_synthesize_duration():
    evidence, binding, _ = clock_fixture()
    view = derive_evidence(evidence, binding)
    with pytest.raises(ValueError, match='chained'): derive_evidence(view, binding)
    evidence['duration_s'] = 1
    binding['evidence_canonical_sha256'] = canonical_hash(evidence)
    with pytest.raises(ValueError, match='duration-bearing'): derive_evidence(evidence, binding)


def test_summary_stays_compact_and_rejects_identity_or_field_drift():
    _, binding, _ = clock_fixture()
    summary = clock_summary(binding)
    assert len(json.dumps(summary)) < 1600 and 'rows' not in summary
    assert summary['binding_sha256'] == canonical_hash(binding)
    with pytest.raises(ValueError): validate_summary(summary, source_sha256='f' * 64)
    summary['rows'] = []
    with pytest.raises(ValueError): validate_summary(summary)


def test_absent_origin_remains_absent_after_validation():
    evidence, binding, _ = clock_fixture()
    evidence.pop('origin_time_s')
    binding['evidence_canonical_sha256'] = canonical_hash(evidence)
    view = derive_evidence(evidence, binding)
    assert 'origin_time_s' not in view
    assert validate_derived_evidence(view) == binding
