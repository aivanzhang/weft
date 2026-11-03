from pathlib import Path
import hashlib
import json
import re
import time

PREP = Path(__file__).resolve().parent
ROOT = PREP.parents[1]
OUT = ROOT / 'evidence/planner-schema-recovery-v1'
STARTED = time.monotonic()
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
            assert time.monotonic() - STARTED < 25
            value.update(block)
    return value.hexdigest()


def write(path, value):
    with path.open('x') as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write('\n')


cfg = json.loads((PREP / 'prepared.json').read_text())
assert len(cfg['inputs']) <= 256 and sum(row['bytes'] for row in cfg['inputs']) <= 32 * 1024**2
rows, unique, marker_refs = [], {}, []
for original in cfg['inputs']:
    path = ROOT / original['path']
    assert path.is_file() and not path.is_symlink() and '.local' not in path.relative_to(ROOT).parts
    assert path.stat().st_size == original['bytes'] and digest(path) == original['sha256']
    row = dict(original)
    reason = ('bytecode cache' if '__pycache__' in path.parts or path.suffix == '.pyc'
              else 'file larger than 8 MiB' if row['bytes'] > 8 * 1024**2 else None)
    if reason is None:
        data = path.read_bytes()
        hits = [name for name, pattern in PATTERNS.items() if re.search(pattern, data)]
        if hits:
            reason = 'credential-shaped fixture; original local bytes retained, not copied'
            marker_refs.append({'path': original['path'], 'marker_categories': hits})
    if reason:
        row.update(storage='hash_reference', local_path=str(path), reason=reason)
    else:
        unique.setdefault(row['sha256'], {'path': 'payload/' + row['sha256'] + path.suffix,
                                          'bytes': row['bytes'], 'source': path})
        row.update(storage='snapshot', snapshot=unique[row['sha256']]['path'])
    rows.append(row)
for sha, row in unique.items():
    path = OUT / row['path']; path.parent.mkdir(exist_ok=True)
    with path.open('xb') as handle:
        handle.write(row['source'].read_bytes())
    assert path.stat().st_size == row['bytes'] and digest(path) == sha

def load(name):
    return json.loads((ROOT / 'outputs' / name).read_text())

strings = load('planner-string-contract-tests-v1/result.json')
custody = load('reference-planner-custody-tests-v1/result.json')
release = load('accepted-source-planner-release-v1/receipt.json')
grant = load('accepted-source-panda-studio-v2/grant-receipt.json')
assert strings['status'] == 'passed_frozen' and strings['unique_test_total'] == 32
assert custody['status'] == 'passed_read_only_actual_custody'
assert release['status'] == 'failed' and release['lease_released'] and release['no_cost_reset'] and release['no_dispatch']
assert release['ledger_sha256_unchanged'] == custody['ledger_sha256_unchanged']
assert grant['parent'] == 'studio-905fab75929b427ea12432a3'
assert grant['child'] == 'studio-d80fa601c945434983e496db'
assert grant['project_spend_microdollars_before_after'] == 3936227
assert grant['grant_microdollars'] == 11063773 and grant['effective_project_cap_microdollars'] == 15000000
assert grant['new_api_calls'] == grant['new_meshy_calls'] == 0
for original in cfg['inputs']:
    path = ROOT / original['path']
    assert path.stat().st_size == original['bytes'] and digest(path) == original['sha256']
write(OUT / 'inventory.json', rows)
write(OUT / 'summary.json', {
    'schema_version': 1, 'status': 'recovery_contract_and_preparation_only',
    'original_job': '905fab75929b427ea12432a3', 'original_job_released_to': 'failed',
    'old_settled_cost_microdollars': 74970, 'project_settled_at_successor_grant_microdollars': 3936227,
    'release_ledger_sha256_unchanged': release['ledger_sha256_unchanged'],
    'successor': {'id': 'd80fa601c945434983e496db', 'status_at_assignment': 'planner running; parent-reported',
                  'grant_microdollars': 11063773, 'request_sha256': grant['request_sha256'],
                  'max_calls': 7, 'max_repairs': 2, 'outcome': 'not included',
                  'live_output_accessed': False},
    'new_schema_validation': {'recorded_unique_tests': 32, 'actual_original_response_changed': False,
                              'transport_string_lengths_now_bound': True,
                              'in_memory_regression_note': strings['actual_failure']},
    'custody_tests': custody['tests'], 'harness_source_change': cfg['harness'],
    'prior_bundles': cfg['prior_bundles'],
    'source_snapshot_policy': 'Only listed frozen tests/schema/inspection/prepare-harness sources; no general runtime inventory or live job snapshot.',
    'no_provider_native_browser_database_or_git_operations': True,
})
write(OUT / 'verification.json', {
    'schema_version': 1, 'source_records': len(rows), 'source_bytes': sum(row['bytes'] for row in rows),
    'snapshot_records': sum(row['storage'] == 'snapshot' for row in rows),
    'hash_reference_records': sum(row['storage'] == 'hash_reference' for row in rows),
    'unique_payload_files': len(unique), 'unique_payload_bytes': sum(row['bytes'] for row in unique.values()),
    'all_source_and_payload_hashes_verified': True, 'source_bytes_unchanged': True,
    'snapshot_secret_marker_hits': 0, 'credential_shaped_references': marker_refs,
    'marker_categories': list(PATTERNS), 'elapsed_seconds': time.monotonic() - STARTED,
    'verification_scope': 'Archive custody only. Existing tests and release were not rerun.',
})
with (OUT / 'README.md').open('x') as handle:
    handle.write('''# Planner schema recovery contract

This checkpoint preserves the local schema correction, custody tests, guarded release of failed planner job `905fab75929b427ea12432a3`, and preparation of successor `d80fa601c945434983e496db`. The successor planner was running when assigned; its live output is not copied and no retry outcome is claimed.

The original 845-character response and original 800-character semantic limit remain unchanged in the earlier sealed failure bundle. New tests bind transport string lengths to local limits and preserve semantic rejection. An explicitly in-memory analytic shortening of that old answer revealed a separate unsupported interaction requirement; it is a regression fixture, not an edited model answer or accepted plan. The recorded schema suite has 32 unique passing tests. The custody receipt records 23 passing focused tests plus actual read-only process/API checks; existing SQLite ResourceWarnings remain disclosed. These are recorded test results, not tests newly executed by packaging.

The public release receipt records a failed terminal job, released lease, unchanged ledger hash, no dispatch and no cost reset. Its $0.074970 settled charge remains in the $3.936227 cumulative cost preserved at successor grant. The append-only successor grant is $11.063773 under the unchanged $15 project ceiling, with at most seven calls and two local repairs. Grant preparation itself made no API or Meshy calls. No private ledger, lease-token file or state database is read or copied here.

## Prepared browser harness

The frozen current harness observes Prepare for at most 270 seconds, closes the browser at 300 seconds, bounds the server at 330 seconds and caps requests at 1,500. These bound-only edits occurred after the earlier mock test. The parent reported a successful `node --check` for the current harness; the mock test was not rerun and this archive does not claim otherwise. No browser was launched during packaging.

## Custody

`inventory.json` maps exact originals to SHA256-deduplicated snapshots or local hash references. Credential-shaped fixtures are referenced, not published. Selected current source files are bound to their observed hashes; frozen source snapshots already present in the test output are retained. Prior terminal/failure bundles are referenced by manifest hash instead of copied again. `summary.json` separates preparation facts from unknown live outcomes, `verification.json` records hash/marker checks, and `manifest.json` seals the bundle. No runtime, provider, database, Git, root README/TODO or skill file was changed.
''')
print(json.dumps({'records': len(rows), 'payload_files': len(unique),
                  'payload_bytes': sum(row['bytes'] for row in unique.values()),
                  'references': sum(row['storage'] == 'hash_reference' for row in rows)}))
