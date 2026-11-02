from pathlib import Path
import hashlib
import json
import re
import time

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
FULL = ROOT/'outputs/b082-coat-review-full-evidence-v1'
FULL_MANIFEST = 'e602f6e730d4cdb84c5a5d175f0198f517b27d669ace716f3f517304a61f742f'
PATTERNS = {
    'provider_key': rb'(?<![A-Za-z0-9])sk-(?:proj-)?[A-Za-z0-9_-]{20,}',
    'bearer_value': rb'Bearer\s+[A-Za-z0-9._-]{24,}',
    'private_key': rb'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----',
    'invocation_token': rb'(?i)["\x27](?:token|owner_token|lease_token|dispatch_token|studio_token|api_key|authorization)["\x27]\s*:\s*["\x27]([A-Za-z0-9._-]{24,})["\x27]',
    'token_query': rb'[?&](?:token|key|api_key)=[A-Za-z0-9._-]{24,}',
}


def digest(path):
    result = hashlib.sha256()
    with path.open('rb') as handle:
        while block := handle.read(1024**2):
            result.update(block)
    return result.hexdigest()


def write(path, value):
    with path.open('x') as handle:
        json.dump(value, handle, sort_keys=True, indent=2, allow_nan=False)
        handle.write('\n')


started = time.monotonic()
assert digest(FULL/'manifest.json') == FULL_MANIFEST
previous = json.loads((FULL/'manifest.json').read_text())
rows, payloads = [], {}
for old in previous['files']:
    source = ROOT/old['path']
    assert not source.is_symlink() and source.is_file()
    assert source.stat().st_size == old['bytes'] and digest(source) == old['sha256']
    assert '.local' not in source.relative_to(ROOT).parts
    row = {key: old[key] for key in ('path', 'bytes', 'sha256')}
    sequence = any(part in {'frames', 'renders'} for part in Path(old['path']).parts)
    if old['storage'] == 'hash_reference' or sequence:
        row.update(storage='hash_reference', local_path=str(source),
                   reason='individual video/pose sequence frame; critic sheets and isolated poses retained'
                   if sequence else old['reason'])
        if old.get('snapshot'):
            row['full_bundle_snapshot'] = str((FULL/old['snapshot']).relative_to(ROOT))
    else:
        source_copy = FULL/old['snapshot']
        assert source_copy.stat().st_size == old['bytes'] and digest(source_copy) == old['sha256']
        if old['sha256'] not in payloads:
            data = source_copy.read_bytes()
            assert not any(re.search(pattern, data) for pattern in PATTERNS.values()), 'Potential credential marker'
            relative = 'payload/'+old['sha256']+source.suffix
            payloads[old['sha256']] = {'path': relative, 'bytes': old['bytes'], 'source': source_copy}
        row.update(storage='snapshot', snapshot=payloads[old['sha256']]['path'])
    rows.append(row)
assert len(rows) == len(previous['files']) == 602
assert sum(row['bytes'] for row in payloads.values()) < 29000000
for expected, row in payloads.items():
    target = OUT/row['path']
    target.parent.mkdir(exist_ok=True)
    with row['source'].open('rb') as source, target.open('xb') as dest:
        while block := source.read(1024**2):
            dest.write(block)
    assert target.stat().st_size == row['bytes'] and digest(target) == expected

by_path = {row['path']: row for row in rows}
job_prefix = 'outputs/studio-b082dc54dbb54f9d9030f75b/'
review = json.loads((ROOT/job_prefix/'review-0.json').read_text())
for evidence in review['evidence']:
    source = Path(evidence['path']).relative_to(ROOT)
    row = by_path[str(source)]
    assert row['storage'] == 'snapshot' and row['sha256'] == evidence['sha256']
for name in ('front.png', 'back.png', 'side.png', 'three-quarter.png'):
    assert by_path[job_prefix+'exported-preview-0/'+name]['storage'] == 'snapshot'
for row in rows:
    if Path(row['path']).suffix == '.mp4' or 'contact-sheets' in Path(row['path']).parts:
        assert row['storage'] == 'snapshot'
assert digest(FULL/'manifest.json') == FULL_MANIFEST

write(OUT/'inventory.json', rows)
summary = json.loads((FULL/'summary.json').read_text())
summary['checkpoint_storage'] = 'Compact hash-deduplicated payloads; all frame sequences preserved at original paths and in moved full bundle'
write(OUT/'summary.json', summary)
write(OUT/'verification.json', {
    'schema_version': 1, 'source_records': len(rows),
    'source_bytes': sum(row['bytes'] for row in rows),
    'snapshot_records': sum(row['storage'] == 'snapshot' for row in rows),
    'hash_reference_records': sum(row['storage'] == 'hash_reference' for row in rows),
    'unique_payload_files': len(payloads),
    'unique_payload_bytes': sum(row['bytes'] for row in payloads.values()),
    'same_source_inventory': True, 'all_original_hashes_and_sizes_verified': True,
    'all_payload_hashes_and_sizes_verified': True, 'all_critic_images_retained': True,
    'four_exported_views_retained': True, 'all_mp4_and_contact_sheets_retained': True,
    'full_bundle_preserved_path': str(FULL.relative_to(ROOT)),
    'full_bundle_manifest_sha256': FULL_MANIFEST,
    'secret_marker_scan': {'unique_payloads': len(payloads), 'hits': 0, 'patterns': list(PATTERNS)},
    'elapsed_seconds': time.monotonic()-started,
})
print(json.dumps({'files': len(rows), 'payload_files': len(payloads),
                  'payload_bytes': sum(row['bytes'] for row in payloads.values())}))
