"""Exercise construction boundaries on unchanged saved observations, without inference."""

import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil
import time

import numpy as np

from check_track_shape_diversity import seal
from weft.io import digest, local_time, write_json
from weft.observations import adapt_run
from weft.pipeline import verify_artifacts
from weft.world_alignment import load_world_alignment
from weft.world_evidence import load_world_evidence


ROOT = Path(__file__).resolve().parents[2]
CASES = {
    'ramp': 'artifacts/meta-phys101-ramp-evidence-01/scene-observations.json',
    'three-balls': 'artifacts/segmentation-evidence-01/three-ball/scene-observations.json',
    'toy-van': 'artifacts/segmentation-evidence-01/toy-van/scene-observations.json',
    'masked-person': 'artifacts/meta-exercise-silhouette-evidence-01/scene-observations.json',
}
BODY_RUN = ROOT/'artifacts/heavy-squat-body-01/lab/run'


def encode(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def audit_export(raws, export, allowed):
    counts = Counter()
    for item in export['sources']:
        identity = item['document_sha256']; doc = json.loads(raws[identity]); admitted = set()
        assert len(item['observations']) == len(doc['observations'])
        for source, row in zip(doc['observations'], item['observations'], strict=True):
            for name in ('id', 'frame', 'time_s', 'object_id', 'partition'):
                assert row[name] == source[name]
            assert row['entity_ref'] == {'document_sha256': identity, 'object_id': source['object_id']}
            eligible = source['partition'] in allowed
            assert row['admission'] == ('included' if eligible else 'withheld')
            assert row['visibility'] == (source['visibility'] if eligible else None)
            assert row['measurements'] == (source['measurements'] if eligible else {})
            assert [r['evidence_id'] for r in row['evidence_refs']] == (source['evidence_ids'] if eligible else [])
            if eligible:admitted.update(source['evidence_ids'])
            counts[row['admission']] += 1
        assert {e['id'] for e in item['evidence']} == admitted
        assert all('payload' not in e for e in item['evidence'])
        assert 'provenance' not in item and 'withheld_relationship_count' not in item
    return dict(counts)


def controls(raw, out):
    doc = json.loads(raw); rows = doc['observations']; midpoint = len(rows)//2
    for row in rows[midpoint:]:row['partition'] = 'audit'
    doc['provenance']['reserved_control'] = 'RAW_ONLY_CONTROL_A'
    first = next(row for row in rows[:midpoint] if row['evidence_ids'])
    doc['relationships'].append({'id': 'injected-cross-partition-relation', 'type': 'controlled_proximity',
        'objects': [first['object_id']], 'status': 'inferred', 'evidence_ids': first['evidence_ids'][:1],
        'interval_s': [rows[0]['time_s'], rows[-1]['time_s']], 'attributes': {'raw_only': 'RAW_ONLY_CONTROL_A'}})
    one = encode(doc); first_registry = load_world_evidence({sha(one): one})
    first_export = json.loads(first_registry.construction_bytes())
    counts = audit_export({sha(one): one}, first_export, {'construction'})
    changed = deepcopy(doc); changed['provenance']['reserved_control'] = 'RAW_ONLY_CONTROL_B'
    for row in changed['observations'][midpoint:]:
        row['visibility'] = 'unknown'; row['measurements'] = {}; row['evidence_ids'] = []
    changed['relationships'][-1]['attributes']['raw_only'] = 'RAW_ONLY_CONTROL_B'
    changed['relationships'].append(deepcopy(changed['relationships'][-1]) | {'id': 'another-hidden-relation'})
    two = encode(changed); second_export = load_world_evidence({sha(two): two}).construction_bytes()
    # Only the pinned document identity may differ when inaccessible evidence changes.
    first_normalized = encode(first_export).replace(sha(one).encode(), b'DOCUMENT')
    second_normalized = second_export.replace(sha(two).encode(), b'DOCUMENT')
    assert first_normalized == second_normalized and b'RAW_ONLY_CONTROL' not in first_normalized
    (out/'controlled-audit-input-a.json').write_bytes(one)
    (out/'controlled-audit-input-b.json').write_bytes(two)
    (out/'controlled-construction.json').write_bytes(first_registry.construction_bytes())
    incomplete = deepcopy(doc); incomplete['observations'].pop(); altered = encode(incomplete)
    try:load_world_evidence({sha(altered): altered})
    except ValueError as error:failure = str(error)
    else:raise AssertionError('Missing source slot was accepted')
    try:first_registry.construction_bytes(allowed_partitions=('audit',))
    except ValueError as error:audit_failure = str(error)
    else:raise AssertionError('Reserved evidence was exported')
    return {'injected_control_not_a_fresh_holdout': True, 'counts': counts,
        'withheld_changes_leave_export_unchanged_except_document_identity': True,
        'omitted_slot_rejection': failure, 'audit_access_rejection': audit_failure}


def alignment_controls(out):
    claim = {'status': 'assumed', 'method': 'Authored coordinate fixture', 'uncertainty': 'Not measured calibration.', 'evidence_refs': []}
    proposal = {'schema_version': 1, 'profile': 'world-alignment-v1',
        'from_frame': {'id': 'body-local', 'length_unit': 'human_model_unit', 'axes': 'right_handed_xyz'},
        'to_frame': {'id': 'world', 'length_unit': 'simulation_length', 'axes': 'x_right_y_away_z_up'},
        'spatial': {'scale': 2., 'rotation_wxyz': [0., 0., 0., 1.], 'translation': [3., 4., 5.]},
        'clock': {'rate': 4., 'source_origin_s': 10., 'target_origin_s': 0.},
        'claims': {'spatial': deepcopy(claim), 'clock': deepcopy(claim)}}
    alignment = load_world_alignment(proposal); durations = []; last = None
    for _ in range(1000):
        started = time.perf_counter()
        last = alignment.map_state(from_frame=proposal['from_frame'], to_frame=proposal['to_frame'],
            time_s=10.5, position=[1., 2., 3.], velocity=[2., 4., 6.],
            acceleration=[8., 16., 24.], angular_velocity=[4., 8., 12.], orientation_wxyz=[1., 0., 0., 0.])
        durations.append((time.perf_counter()-started)*1000)
    for field, expected in {'position': [1., 0., 11.], 'velocity': [-1., -2., 3.],
            'acceleration': [-1., -2., 3.], 'angular_velocity': [-1., -2., 3.],
            'orientation_wxyz': [0., 0., 0., 1.]}.items():np.testing.assert_array_equal(last[field], expected)
    assert last['time_s'] == 2. and last['source_time_s'] == 10.5
    unknown = deepcopy(proposal); unknown['spatial'] = None; unknown['claims']['spatial']['status'] = 'unknown'
    pending = load_world_alignment(unknown)
    try:pending.map_state(from_frame=unknown['from_frame'], to_frame=unknown['to_frame'], time_s=10.5, position=[1., 2., 3.])
    except ValueError as error:failure = str(error)
    else:raise AssertionError('Unknown geometry was silently supplied')
    write_json(out/'authored-alignment.json', proposal); write_json(out/'mapped-state.json', last)
    write_json(out/'unknown-alignment.json', unknown)
    return {'calls': len(durations), 'median_ms': float(np.median(durations)), 'p95_ms': float(np.percentile(durations, 95)),
        'maximum_ms': max(durations), 'unknown_alignment_rejection': failure,
        'physical_validation': 'unverified', 'scope': 'Coordinate conversion only; no simulator or UI measurement.'}


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--out', type=Path, required=True)
    out = parser.parse_args().out.resolve()
    if out.parent != ROOT/'artifacts':raise ValueError('Use a new top-level artifact directory')
    out.mkdir(exist_ok=False); started = time.perf_counter()
    used = [Path(__file__), ROOT/'src/weft/world_alignment.py', ROOT/'src/weft/world_evidence.py',
        ROOT/'src/weft/observations.py', ROOT/'tests/test_world_alignment.py', ROOT/'tests/test_world_evidence.py']
    hashes = {str(p.relative_to(ROOT)): digest(p) for p in used}
    for path in used:shutil.copyfile(path, out/path.name)
    write_json(out/'pre-run.json', {'created_at': local_time(), 'used_sources': hashes,
        'maximum_seconds': 60, 'maximum_bytes': 64*1024**2, 'workers': 1,
        'cost': 'Local CPU only; zero paid calls, inference, acquisition, video decodes or physics.',
        'scope': 'Exposed saved evidence plus explicitly injected software controls, not fresh physical validation.'})
    try:
        raw_dir = out/'observations'; raw_dir.mkdir(); inputs = {}; originals = {}
        checks = {}
        for name, relative in CASES.items():
            path = ROOT/relative; checks[relative] = verify_artifacts(path.parent)
            raw = path.read_bytes(); originals[sha(raw)] = raw
            (raw_dir/f'{name}.json').write_bytes(raw); inputs[name] = {'path': relative, 'sha256': sha(raw)}
        checks[str(BODY_RUN.relative_to(ROOT))] = verify_artifacts(BODY_RUN)
        raw = encode(adapt_run(BODY_RUN)); originals[sha(raw)] = raw
        (raw_dir/'pose-person.json').write_bytes(raw)
        inputs['pose-person'] = {'path': str(BODY_RUN.relative_to(ROOT)), 'sha256': sha(raw), 'adapter': 'adapt_run, no inference'}
        write_json(out/'input-binding.json', {'inputs': inputs, 'manifest_checks': checks})
        tick = time.perf_counter(); registry = load_world_evidence(originals); load_ms = (time.perf_counter()-tick)*1000
        tick = time.perf_counter(); summary = registry.summary(); summary_ms = (time.perf_counter()-tick)*1000
        tick = time.perf_counter(); exported = registry.construction_bytes(); export_ms = (time.perf_counter()-tick)*1000
        counts = audit_export(originals, json.loads(exported), {'construction'})
        assert all(registry.source_bytes(key) == value for key, value in originals.items())
        (out/'construction-evidence.json').write_bytes(exported); write_json(out/'registry-summary.json', summary)
        controlled = controls(originals[inputs['ramp']['sha256']], out)
        numeric = alignment_controls(out)
        for path, expected in hashes.items():assert digest(ROOT/path) == expected
        for item in inputs.values():
            if item.get('adapter') is None:assert digest(ROOT/item['path']) == item['sha256']
        size = sum(p.stat().st_size for p in out.rglob('*') if p.is_file())
        if size > 64*1024**2 or time.perf_counter()-started > 60:raise RuntimeError('Study envelope exceeded')
        report = {'status': 'verified', 'created_at': local_time(), 'input_documents': len(originals),
            'source_frames': sum(s['source']['decoded_frames'] for s in summary['sources']),
            'total_slots': summary['total_slots'], 'input_bytes': summary['total_bytes'],
            'qualified_entities': len(registry.entity_keys), 'distinct_local_ids': len({k[1] for k in registry.entity_keys}),
            'construction_counts': counts, 'controlled_partition_tests': controlled, 'alignment': numeric,
            'latency_ms': {'registry_load': load_ms, 'summary': summary_ms, 'construction_export': export_ms},
            'bytes_before_report_manifest': size, 'seconds': time.perf_counter()-started,
            'new_paid_calls': 0, 'new_world': False, 'physical_validation': 'unverified',
            'limits': 'Separate source documents remain separate scenes. No person/object alignment or mixed simulator is inferred.'}
        write_json(out/'report.json', report)
    except Exception as error:
        write_json(out/'failure.json', {'type': type(error).__name__, 'message': str(error)}); seal(out); raise
    seal(out); print(json.dumps(report | {'manifest_sha256': digest(out/'manifest.json')}))


if __name__ == '__main__':main()
