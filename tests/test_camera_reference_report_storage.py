"""Lossless saved-report storage under the unchanged offline output budget."""

import gzip
from hashlib import sha256
from html.parser import HTMLParser
import json
from pathlib import Path

import pytest

from weft import whole_body_reference_report as report
from test_camera_body_reference_report import camera_documents, metadata_only
from test_whole_body_reference_report import save_study


class Links(HTMLParser):
    def __init__(self):
        super().__init__()
        self.targets = []

    def handle_starttag(self, tag, attributes):
        if tag == 'a':
            self.targets.extend(value for name, value in attributes if name == 'href')


def inspect_output(out, result):
    manifest_raw = (out / 'manifest.json').read_bytes()
    seal = json.loads(manifest_raw)
    assert sha256(manifest_raw).hexdigest() == result['manifest_sha256']
    size = len(manifest_raw)
    for name, digest in seal['files'].items():
        raw = (out / name).read_bytes()
        assert sha256(raw).hexdigest() == digest
        size += len(raw)
    assert size == result['bytes']
    parser = Links()
    parser.feed((out / 'index.html').read_text())
    assert parser.targets
    assert all(not Path(name).is_absolute() and '..' not in Path(name).parts
               and (out / name).is_file() for name in parser.targets)
    return json.loads((out / 'report.json').read_bytes())


def test_small_camera_report_keeps_existing_raw_layout(tmp_path):
    source, out = tmp_path / 'source', tmp_path / 'inspection'
    seal = save_study(source, camera_documents())
    result = report.prepare_whole_body_reference_report(source, out, expected_manifest_sha256=seal)
    model = inspect_output(out, result)
    assert 'artifact_storage' not in model
    for original, copied in (('inputs/evidence-original.json', 'evidence-original.json.txt'),
                             ('inputs/evidence-derived.json', 'evidence-derived.json.txt'),
                             ('inputs/source-camera.json', 'source-camera.json.txt'),
                             ('inputs/surface-camera.json', 'surface-camera.json.txt'),
                             ('report.json', 'proposal-report.json.txt')):
        assert (out / 'artifacts' / copied).read_bytes() == (source / original).read_bytes()


def test_exact_aliases_can_fit_without_compression(tmp_path, monkeypatch):
    source, original_out, out = tmp_path / 'source', tmp_path / 'ordinary', tmp_path / 'aliased'
    save_study(source, camera_documents())
    result = report.prepare_whole_body_reference_report(source, original_out)
    before_seal = result['bytes'] - (original_out / 'manifest.json').stat().st_size
    savings = (source / 'inputs/evidence-original.json').stat().st_size + (source / 'inputs/source-camera.json').stat().st_size
    budget = before_seal + 65536 - savings // 3
    def unexpected_compression(*args, **kwargs):
        pytest.fail('Exact aliases already fit; compression is unnecessary')
    monkeypatch.setattr(report, '_compress_report', unexpected_compression)
    result = report.prepare_whole_body_reference_report(source, out, max_output_bytes=budget)
    model = inspect_output(out, result)
    storage = model['artifact_storage']['members']
    assert result['bytes'] <= budget
    assert all(item['encoding'] == 'identity' for item in storage.values())
    assert storage['inputs/evidence-original.json'] == storage['inputs/evidence-derived.json']
    assert storage['inputs/source-camera.json'] == storage['inputs/surface-camera.json']


def test_saved_native_size_envelope_needs_more_than_aliases(record_property):
    # Recorded fit-02 sizes exercise the failed byte envelope, not source geometry or the fitter.
    lengths = {'report.json': 7856096, 'inputs/surface-observation-00.json': 2732749,
               'inputs/evidence-derived.json': 1903436, 'inputs/evidence-original.json': 1903436,
               'inputs/human.json': 701619, 'inputs/surface-camera.json': 127993,
               'inputs/source-camera.json': 127993, 'angle-proposal.json': 110952,
               'inputs/surface-annotations.json': 87438, 'plan.json': 11523,
               'inputs/camera-inputs-proof.json': 9180, 'inputs/body-manifest.json': 5979,
               'inputs/surface-body-binding.json': 2709, 'inputs/surface-config.json': 916,
               'inputs/ground.json': 532, 'inputs/surface-intervals.json': 486}
    raw = {name: (bytes([65 + i]) * size) for i, (name, size) in enumerate(lengths.items())}
    raw['inputs/evidence-original.json'] = raw['inputs/evidence-derived.json']
    raw['inputs/source-camera.json'] = raw['inputs/surface-camera.json']
    names = {name: str(i) + '.json.txt' for i, name in enumerate(raw)}
    hashes = {name: sha256(value).hexdigest() for name, value in raw.items()}
    aliases, stored, storage = report._alias_artifacts(raw, names, hashes)
    fixed_outputs = 1778392 + 2648635 + 6248 + 9839
    original_total = sum(map(len, raw.values())) + fixed_outputs
    assert original_total == 20026151
    savings = sum(map(len, raw.values())) - sum(map(len, stored.values()))
    assert savings == 2031429
    assert original_total - savings - (report.MAX_BYTES - 65536) == 1283042
    assert aliases['inputs/evidence-original.json'] == aliases['inputs/evidence-derived.json']
    assert aliases['inputs/source-camera.json'] == aliases['inputs/surface-camera.json']
    report._compress_report(raw, aliases, stored, storage)
    first = stored[aliases['report.json']]
    assert gzip.decompress(first) == raw['report.json']
    aliases2, stored2, storage2 = report._alias_artifacts(raw, names, hashes)
    report._compress_report(raw, aliases2, stored2, storage2)
    assert stored2[aliases2['report.json']] == first
    assert sum(map(len, stored.values())) + fixed_outputs + 65536 < report.MAX_BYTES
    assert [name for name, item in storage.items() if item['encoding'] == 'gzip'] == ['report.json']
    record_property('envelope', json.dumps({'recorded_original_bytes': original_total,
        'exact_alias_savings': savings, 'scope': 'Recorded byte sizes with analytical repeated-byte payloads; not a measured source compression ratio'}))


def test_public_fallback_preserves_every_raw_pin_row_link_and_failed_gate(tmp_path):
    documents = camera_documents()
    # Uninterpreted saved details must remain exact downloadable evidence, never UI work.
    documents['report.json']['saved_diagnostic_note'] = 'retained <script>not executable</script>\n' * 60000
    source, out = tmp_path / 'source', tmp_path / 'inspection'
    seal = save_study(source, documents)
    original = {name: (source / name).read_bytes() for name in documents}
    result = report.prepare_whole_body_reference_report(source, out, expected_manifest_sha256=seal,
                                                       max_output_bytes=1 << 20)
    model = inspect_output(out, result)
    assert result['bytes'] <= 1 << 20
    assert (out / 'index.html').stat().st_size <= report.MAX_HTML
    assert model['gates']['floor_feasibility'] is False
    assert model['status'] == 'rejected_construction_proposal' and model['promoted'] is False
    assert len(model['source']) == len(documents['report.json']['retained_pose_slots'])
    assert len(model['grid']) == sum(row['kind'] == 'control_grid'
                                    for row in documents['report.json']['geometry_checks'])
    assert len(model['surface_intervals']['source_slots']) == 6
    assert model['camera_fit']['source_slots'][5]['camera_time_s'] is None
    storage = model['artifact_storage']['members']
    assert set(storage) == set(model['selected_members'])
    for name, item in storage.items():
        stored = (out / item['path']).read_bytes()
        assert sha256(stored).hexdigest() == item['sha256'] and len(stored) == item['bytes']
        restored = gzip.decompress(stored) if item['encoding'] == 'gzip' else stored
        assert restored == original[name]
        assert model['selected_members'][name] == {'sha256': sha256(restored).hexdigest(), 'bytes': len(restored)}
        assert (source / name).read_bytes() == original[name]
    for a, b in (('inputs/evidence-original.json', 'inputs/evidence-derived.json'),
                 ('inputs/source-camera.json', 'inputs/surface-camera.json')):
        assert storage[a] == storage[b]
    assert storage['report.json']['encoding'] == 'gzip'
    page = (out / 'index.html').read_text()
    assert 'gzip download; decompress to read' in page
    assert 'retained <script>not executable</script>' not in page
    assert (out / 'artifacts' / 'proposal-report.json.txt.gz').is_file()
    assert not (out / 'artifacts' / 'proposal-report.json.txt').exists()


def test_aliases_require_exact_bytes_and_too_small_budget_still_rejects(tmp_path):
    with pytest.raises(ValueError, match='alias bytes differ'):
        report._alias_artifacts({'a': b'left', 'b': b'fake'}, {'a': 'a.txt', 'b': 'b.txt'},
                                {'a': 'a' * 64, 'b': 'a' * 64})
    source, out = tmp_path / 'source', tmp_path / 'inspection'
    save_study(source, camera_documents())
    with pytest.raises(ValueError, match='output exceeds its byte bound'):
        report.prepare_whole_body_reference_report(source, out, max_output_bytes=1024)
    assert not out.exists()
