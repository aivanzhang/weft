from pathlib import Path
import hashlib
import json
import re
import time

PREP = Path(__file__).resolve().parent
ROOT = PREP.parents[1]
OUT = ROOT / 'evidence/api-streaming-status-v1'
START = time.monotonic()
PATTERNS = {
    'provider_key': rb'(?<![A-Za-z0-9])sk-(?:proj-)?[A-Za-z0-9_-]{20,}',
    'bearer': rb'Bearer\s+[A-Za-z0-9._-]{24,}',
    'private_key': rb'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----',
    'invocation': rb'(?i)["\x27](?:token|owner_token|lease_token|dispatch_token|studio_token|api_key|authorization)["\x27]\s*:\s*["\x27][A-Za-z0-9._-]{24,}["\x27]',
    'query_token': rb'[?&](?:token|key|api_key)=[A-Za-z0-9._-]{24,}',
}


def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as handle:
        while block := handle.read(1024**2):
            assert time.monotonic() - START < 25
            value.update(block)
    return value.hexdigest()


def write(path, value):
    with path.open('x') as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write('\n')


cfg = json.loads((PREP / 'prepared.json').read_text())
assert len(cfg['inputs']) <= 256 and sum(row['bytes'] for row in cfg['inputs']) <= 32 * 1024**2
rows, unique, flagged = [], {}, []
for original in cfg['inputs']:
    path = ROOT / original['path']
    assert path.is_file() and not path.is_symlink() and '.local' not in path.relative_to(ROOT).parts
    assert path.stat().st_size == original['bytes'] and digest(path) == original['sha256']
    row = dict(original)
    reason = ('cache' if any(p in {'__pycache__', '.cache'} for p in path.parts) or path.suffix == '.pyc'
              else 'file larger than 8 MiB' if row['bytes'] > 8 * 1024**2 else None)
    if reason is None:
        data = path.read_bytes()
        hits = [name for name, pattern in PATTERNS.items() if re.search(pattern, data)]
        if hits:
            reason = 'credential-shaped fixture; original exact bytes retained locally, not copied'
            flagged.append({'path': original['path'], 'marker_categories': hits})
    if reason:
        row.update(storage='hash_reference', local_path=str(path), reason=reason)
    else:
        unique.setdefault(row['sha256'], {'path': 'payload/' + row['sha256'] + path.suffix,
                                          'source': path, 'bytes': row['bytes']})
        row.update(storage='snapshot', snapshot=unique[row['sha256']]['path'])
    rows.append(row)
assert sum(row['bytes'] for row in unique.values()) <= 16 * 1024**2
for sha, row in unique.items():
    target = OUT / row['path']; target.parent.mkdir(exist_ok=True)
    with target.open('xb') as handle:
        handle.write(row['source'].read_bytes())
    assert target.stat().st_size == row['bytes'] and digest(target) == sha
for source in cfg['receipt_source_checks']:
    path = ROOT / source['snapshot_path']
    assert digest(path) == source['sha256']
for row in cfg['inputs']:
    path = ROOT / row['path']
    assert path.stat().st_size == row['bytes'] and digest(path) == row['sha256']
write(OUT / 'inventory.json', rows)
write(OUT / 'summary.json', cfg['summary'])
write(OUT / 'verification.json', {
    'schema_version': 1, 'source_records': len(rows), 'source_bytes': sum(row['bytes'] for row in rows),
    'snapshot_records': sum(row['storage'] == 'snapshot' for row in rows),
    'hash_reference_records': sum(row['storage'] == 'hash_reference' for row in rows),
    'unique_payload_files': len(unique), 'unique_payload_bytes': sum(row['bytes'] for row in unique.values()),
    'all_source_and_payload_hashes_verified': True, 'all_selected_source_bytes_unchanged': True,
    'receipt_source_checks': len(cfg['receipt_source_checks']),
    'snapshot_secret_marker_hits': 0, 'credential_shaped_references': flagged,
    'marker_categories': list(PATTERNS), 'elapsed_seconds': time.monotonic() - START,
    'scope': 'Archival custody only; no test replay, provider/browser/native call, pipeline dispatch or private-state query.',
})
with (OUT / 'README.md').open('x') as handle:
    handle.write('''# Streaming transport and stopped-AI status

This checkpoint preserves mocked streaming transport, runner integration, stopped-AI status/UI tests and a read-only input-token cost-bound review. It does **not** demonstrate a real streaming provider call, repaired avatar, resumed job or completed hands-off generation. The actual owner configuration remains unchanged.

## Verification records

- 15 streaming transport mock tests plus analytic payload/resource measurements; not provider latency.
- 13 new runner integration cases, recorded as an initial 11 plus two real-parser mock cases.
- An adjacent 112-test batch recorded 111 passes and one fixture error: missing analytic guide images. The fixture-only correction then passed its targeted test; the original failure remains preserved. A separate affected continuation case also passed after transport-selection equality was tightened.
- 53 Python status/job/release tests and three pure Node UI tests, followed by a two-case priority regression ensuring delivery failures are not replaced by an AI-status message. The latter repeats one existing case; these counts are not summed into an inflated unique total. No browser was used for the UI checks.

The receipts identify analyst-recorded commands and any fixture warnings; packaging does not invent raw logs or rerun tests. Saved source snapshots are verified against their receipts. The current named status helper/browser/tests and skill are captured where those receipts did not already include source bytes. Mutable root README/TODO files are excluded.

## Unresolved charge and cost review

The previously recorded $3.999655 settled usage and $5.618640 unknown reservation remain unchanged. Their $9.618295 combined exposure is not an invoice. Ledger hash `37be6a2c367a1d0817a90c525f2d75eba86fedc4ef3d16a585c731a8ea63c07e` and registry hash `7f3cf7d8707c27a9d367d647578ba4d3c649a55c6f036215bf84cb11d5802e83` are retained from public read-only receipts, not re-queried by packaging. No release, settlement, budget change or automatic retry is authorized by these checks.

The input-token/cost-bound review has no authenticated endpoint receipt and does not prove that token counting is free. It does not adjust the unknown charge or the reservation policy. Local/mock measurements and provider documentation remain distinct from actual billing evidence.

## Custody

`inventory.json` maps originals to deduplicated exact-byte payloads or local SHA256 references for credential-shaped fixtures, caches or large files. `summary.json` retains source-freeze and test distinctions; `verification.json` records source/payload and marker checks. `packaging/` preserves the archive command and receipt, and `manifest.json` seals this bundle. Private `.local` credentials, databases and raw lease tokens are not copied. No paid, browser, native, pipeline or Git operation occurred during packaging.
''')
print(json.dumps({'records': len(rows), 'payload_files': len(unique),
                  'payload_bytes': sum(row['bytes'] for row in unique.values()),
                  'references': sum(row['storage'] == 'hash_reference' for row in rows)}))
