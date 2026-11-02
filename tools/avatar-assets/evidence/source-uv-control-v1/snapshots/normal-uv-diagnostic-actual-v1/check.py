"""Check exact saved native receipts inside new analytic packaging fixtures."""
import hashlib
import json
from pathlib import Path
import sys
import time

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
sys.path.insert(0, str(ROOT))
from avatar_assets.pipeline import _normal_uv_diagnostic


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    text = json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+'\n'
    with path.open('x') as handle:
        handle.write(text)


started = time.monotonic()
inputs = json.loads((OUT/'inputs.json').read_text())
for row in inputs['files']:
    assert digest(ROOT/row['path']) == row['sha256'], 'Input hash changed'
native = ROOT/'outputs/normal-uv-export-native-v1'
native_manifest = json.loads((native/'manifest.json').read_text())
checks = []
for kind, filename in (('authored', 'authored-normal-uv-export.json'),
                       ('original', 'original-normal-uv-export.json')):
    source = native/filename
    record = [row for row in native_manifest['artifacts'] if row['path'] == filename]
    assert len(record) == 1 and record[0]['sha256'] == digest(source)
    assert record[0]['bytes'] == source.stat().st_size <= 512*1024
    pack = OUT/(kind+'-fixture'); pack.mkdir()
    path = pack/'normal-uv-export.json'; path.write_bytes(source.read_bytes())
    manifest = {'schema_version': 1, 'status': 'failed',
        'provenance': 'analytic_packaging_fixture_with_exact_actual_native_report_bytes',
        'artifacts': [{'path': path.name, 'bytes': path.stat().st_size, 'sha256': digest(path)}]}
    write(pack/'manifest.json', manifest)
    result = _normal_uv_diagnostic(pack, manifest)
    if kind == 'authored':
        assert result is None, 'Supported native receipt became a rejection'
    else:
        assert result['status'] == 'unsupported'
        assert result['errors'][0]['code'] == 'nonzero_normal_uv_without_receipt'
        assert result['affected_materials'][0]['material'] == 'body_surface__contact_uv_dark_v1'
        assert result['report']['sha256'] == digest(source)
    checks.append({'kind': kind, 'native_path': str(source.relative_to(ROOT)),
                   'native_sha256': digest(source), 'copy_byte_identical': path.read_bytes() == source.read_bytes(),
                   'diagnostic': result})
for row in inputs['files']:
    assert digest(ROOT/row['path']) == row['sha256'], 'Input changed during compatibility check'
write(OUT/'result.json', {'schema_version': 1, 'status': 'actual_native_report_compatibility_passed',
    'checks': checks, 'elapsed_seconds': time.monotonic()-started, 'originals_unchanged': True,
    'fixture_manifest_scope': 'New wrapper manifests are analytic; the copied reports are exact actual native evidence.',
    'new_native_calls': 0, 'new_provider_calls': 0, 'new_ai_role_calls': 0,
    'full_pipeline_or_quality_acceptance': False})
