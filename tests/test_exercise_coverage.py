"""Coverage UI must retain missing evidence and never grade reconstruction accuracy."""

from copy import deepcopy
import json
import threading

import pytest

from weft.exercise_coverage import load_coverage, summarize_coverage
from weft.io import write_json
from weft.lab import Lab


HASH = 'a' * 64


def evidence():
    rows = []
    for index, status in zip(range(0, 12, 2), ('tracked', 'tracked_partial', 'no_pose', 'multiple_people', 'low_confidence', 'tracked')):
        rows.append({'id': f'{HASH}:frame:{index}', 'frame': index, 'time_s': index / 20,
            'decoder_timestamp_s': index / 20 + .003, 'status': status})
    return {'schema_version': 1, 'profile': 'exercise-pose-v1', 'source_sha256': HASH,
        'frame_count': 13, 'decoded_frames': 12, 'fps': 20, 'frames': rows,
        'quality': {'errors': ['Competing hypotheses remain unresolved']}}


def movement():
    return {'profile': 'image-pelvis-root-reference-v1', 'source_sha256': HASH, 'status': 'insufficient_reference',
        'quality': {'long_reference_intervals': [{'left_evidence_id': f'{HASH}:frame:2',
            'right_evidence_id': f'{HASH}:frame:10', 'seconds': .4}],
            'gaps': [{'status': 'unfilled', 'evidence_ids': [f'{HASH}:frame:10']}]}}


def test_counts_separate_unsampled_unusable_partial_and_competing():
    source = evidence()
    preserved = deepcopy(source)
    result = summarize_coverage(source)
    assert (result['metadata_frames'], result['decoded_frames'], result['sampled_frames']) == (13, 12, 6)
    assert (result['unsampled_frames'], result['usable_samples'], result['unusable_samples']) == (6, 3, 3)
    assert result['partial_samples'] == result['competing_samples'] == 1
    assert result['markers'][2] == [4, .203, 'unusable']
    assert result['issues'] == [{'kind': 'unusable_pose', 'first_frame': 4, 'last_frame': 8, 'seek_s': .203, 'samples': 3}]
    assert result['movement']['status'] == 'not_available'
    assert 'not anatomical accuracy' in result['scope']
    assert source == preserved


def test_movement_issues_are_source_bound_and_not_body_approval():
    result = summarize_coverage(evidence(), movement())
    assert result['movement']['long_gaps'] == result['movement']['unfilled_gaps'] == 1
    assert result['movement']['maximum_gap_s'] == .4
    assert [row['first_frame'] for row in result['issues']] == [2, 4, 10]
    assert result['issues'][0]['seek_s'] == pytest.approx(.103)
    assert 'not physical body approval' in result['movement']['scope']


@pytest.mark.parametrize('change', ['source', 'order', 'frame', 'clock', 'nan', 'oversize', 'record'])
def test_invalid_source_sequences_never_get_a_coverage_claim(change):
    source = evidence()
    if change == 'source': source['source_sha256'] = 'b' * 64
    elif change == 'order': source['frames'].reverse()
    elif change == 'frame': source['frames'][-1]['frame'] = 12
    elif change == 'clock': source['frames'][1]['time_s'] += .1
    elif change == 'nan': source['frames'][1]['decoder_timestamp_s'] = float('nan')
    elif change == 'oversize': source['frames'] *= 151
    else: source['frames'][1] = None
    with pytest.raises(ValueError): summarize_coverage(source)


@pytest.mark.parametrize('change', ['source', 'clock', 'unknown', 'empty', 'oversize'])
def test_invalid_movement_audit_never_becomes_a_green_check(change):
    audit = movement()
    if change == 'source': audit['source_sha256'] = 'b' * 64
    elif change == 'clock': audit['quality']['long_reference_intervals'][0]['seconds'] = .1
    elif change == 'unknown': audit['quality']['long_reference_intervals'][0]['right_evidence_id'] = f'{HASH}:frame:11'
    elif change == 'empty': audit['quality']['gaps'][0]['evidence_ids'] = []
    else: audit['quality']['long_reference_intervals'] *= 900
    with pytest.raises(ValueError): summarize_coverage(evidence(), audit)


def test_file_loading_is_bounded_and_keeps_originals_unchanged(tmp_path):
    path = tmp_path / 'evidence.json'
    write_json(path, evidence())
    before = path.read_bytes()
    assert load_coverage(tmp_path)['status'] == 'available'
    assert path.read_bytes() == before
    path.write_text('[]')
    assert load_coverage(tmp_path)['status'] == 'unavailable'
    with path.open('wb') as stream:
        stream.truncate(32 * 1024 * 1024 + 1)
    assert load_coverage(tmp_path)['status'] == 'unavailable'


def test_coverage_never_follows_a_symlink_outside_run(tmp_path):
    run = tmp_path / 'run'
    run.mkdir()
    outside = tmp_path / 'outside.json'
    write_json(outside, evidence())
    (run / 'evidence.json').symlink_to(outside)
    assert load_coverage(run)['status'] == 'unavailable'


def test_one_background_snapshot_for_terminal_runs_not_every_status_poll(tmp_path, monkeypatch):
    write_json(tmp_path / 'lab-config.json', {'profile': 'exercise'})
    write_json(tmp_path / 'status.json', {'status': 'running'})
    entered, release = threading.Event(), threading.Event()
    calls = []

    def load(run):
        calls.append((run, threading.get_ident()))
        entered.set()
        assert release.wait(5)
        return {'status': 'available', 'scope': 'synthetic'}

    monkeypatch.setattr('weft.lab.load_coverage', load)
    lab = Lab(tmp_path)
    try:
        assert lab.exercise_coverage()['status'] == 'pending'
        assert not calls
        write_json(tmp_path / 'status.json', {'status': 'failed'})
        assert lab.exercise_coverage()['status'] == 'loading'
        assert entered.wait(5)
        for _ in range(10): assert lab.exercise_coverage()['status'] == 'loading'
        assert len(calls) == 1 and calls[0][1] != threading.get_ident()
        release.set()
        lab.coverage_future.result(timeout=5)
        for _ in range(10): assert lab.exercise_coverage() == {'status': 'available', 'scope': 'synthetic'}
        assert len(calls) == 1
    finally:
        release.set()
        lab.close()


def test_other_profiles_do_not_read_pose_files(tmp_path):
    write_json(tmp_path / 'lab-config.json', {'profile': 'planar'})
    write_json(tmp_path / 'status.json', {'status': 'ready'})
    lab = Lab(tmp_path)
    try:
        assert lab.exercise_coverage() == {'status': 'not_applicable'}
        assert lab.coverage_future is None
    finally:
        lab.close()
