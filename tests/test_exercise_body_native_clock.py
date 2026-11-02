"""Native-time construction preserves raw detector history and quality gates."""

import json

import pytest

from weft import exercise_body_source as source
from weft.io import canonical_hash, digest, write_json
from weft.source_clock import clock_summary, frame_times
from test_exercise_body_source import pose_run, seal, forbid_runtime


def native_pose(root):
    pose_run(root)
    evidence = json.loads((root / 'evidence.json').read_bytes())
    raw = json.loads((root / 'pose-landmarks.json').read_bytes())
    ticks = [3000 + round(i * 1000 / 30) + (40 if i >= 32 else 0) for i in range(63)]
    rows = [{'frame': i, 'pts': tick, 'decoder_timestamp_s': (tick - 3000) / 1000 + .003}
            for i, tick in enumerate(ticks)]
    for collection in (evidence['frames'], raw['frames']):
        for row in collection:
            if row['decoder_timestamp_s'] is not None:
                row['decoder_timestamp_s'] = rows[row['frame']]['decoder_timestamp_s']
    write_json(root / 'pose-landmarks.json', raw)
    raw_sha = digest(root / 'pose-landmarks.json')
    audit = json.loads((root / 'identity-audit.json').read_bytes())
    audit['raw_artifact_sha256'] = raw_sha
    evidence['pose_identity']['raw_artifact_sha256'] = raw_sha
    evidence['frame_count'] = 66
    evidence['container_frame_audit'] = {'matches': True, 'encoded_video_frames': 63, 'decoded_frames': 63}
    write_json(root / 'identity-audit.json', audit)
    write_json(root / 'evidence.json', evidence)
    seal(root)
    binding = {'schema_version': 1, 'profile': 'source-native-clock-v1', 'source_sha256': evidence['source_sha256'],
        'evidence_sha256': digest(root / 'evidence.json'), 'evidence_canonical_sha256': canonical_hash(evidence),
        'metadata_frames': 66, 'decoded_frames': 63, 'nominal_fps': 30., 'time_base': '1/1000',
        'native_origin_pts': 3000, 'decoder_minus_native_s': -2.997, 'native_to_reconstruction_offset_s': -3.,
        'capture_clock_status': 'unknown', 'detector_clock_status': 'preserved_nominal_frame_index_divided_by_fps',
        'rows': rows}
    clock = root.parent / 'clock.json';write_json(clock, binding)
    return root, clock, evidence, binding


def test_native_body_uses_irregular_times_before_reconstruction_and_keeps_raw_bytes(tmp_path, monkeypatch):
    original, clock, evidence, binding = native_pose(tmp_path / 'pose')
    before = {p: p.read_bytes() for p in original.rglob('*') if p.is_file()}
    forbid_runtime(monkeypatch)
    out = tmp_path / 'body'
    source.prepare_exercise_body_source_from_evidence(original, out, native_clock=clock)
    assert all(p.read_bytes() == raw for p, raw in before.items())
    assert (out / 'evidence.json').read_bytes() == before[original / 'evidence.json']
    assert (out / 'pose-landmarks.json').read_bytes() == before[original / 'pose-landmarks.json']
    human = json.loads((out / 'human.json').read_bytes())
    usable = [row for row in evidence['frames'] if row['status'] in {'tracked', 'tracked_partial'}]
    assert [r['time_s'] for r in human['reference']] == [frame_times(binding)[r['frame']] for r in usable]
    assert human['native_clock'] == clock_summary(binding)
    assert human['reference'][-1]['time_s'] != usable[-1]['time_s']
    observations = json.loads((out / 'observations.json').read_bytes())
    assert observations['schema_version'] == 4 and len(observations['observations']) == 63
    assert observations['source']['metadata_frames'] == 66
    legacy_clock = json.loads((out / 'source-clock.json').read_bytes())
    assert legacy_clock['samples'][0]['time_s'] == evidence['frames'][0]['time_s']
    files = {name: (out / name).read_bytes() for name in json.loads((out / 'manifest.json').read_bytes())['files']}
    source.validate_body_source_files(files)
    files['reconstruction-evidence.json'] = b'{}'
    with pytest.raises(ValueError): source.validate_body_source_files(files)


def test_native_clock_never_overrides_rejected_pose_quality(tmp_path, monkeypatch):
    original, clock, evidence, binding = native_pose(tmp_path / 'pose')
    evidence['quality']['passes'] = False
    evidence['quality']['errors'] = ['Retained analytical quality rejection']
    write_json(original / 'evidence.json', evidence)
    binding.update(evidence_sha256=digest(original / 'evidence.json'), evidence_canonical_sha256=canonical_hash(evidence))
    write_json(clock, binding);seal(original)
    monkeypatch.setattr(source, 'build_human', lambda *_a, **_k: pytest.fail('No body from rejected poses'))
    with pytest.raises(ValueError, match='Rejected/unsupported'):
        source.prepare_exercise_body_source_from_evidence(original, tmp_path / 'failed', native_clock=clock)
    assert json.loads((tmp_path / 'failed/failure.json').read_bytes())['status'] == 'failed'
