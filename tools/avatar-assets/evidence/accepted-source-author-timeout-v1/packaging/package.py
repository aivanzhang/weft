from pathlib import Path
import hashlib
import json
import re
import time

PREP = Path(__file__).resolve().parent
ROOT = PREP.parents[1]
OUT = ROOT / 'evidence/accepted-source-author-timeout-v1'
JOB = ROOT / 'outputs/studio-d80fa601c945434983e496db'
START = time.monotonic()
PATTERNS = {
    'provider_key': rb'(?<![A-Za-z0-9])sk-(?:proj-)?[A-Za-z0-9_-]{20,}',
    'bearer': rb'Bearer\s+[A-Za-z0-9._-]{24,}',
    'private_key': rb'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----',
    'invocation': rb'(?i)["\x27](?:token|owner_token|lease_token|dispatch_token|studio_token|api_key|authorization)["\x27]\s*:\s*["\x27][A-Za-z0-9._-]{24,}["\x27]',
    'query_token': rb'[?&](?:token|key|api_key)=[A-Za-z0-9._-]{24,}',
}


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as handle:
        while block := handle.read(1024**2):
            assert time.monotonic() - START < 25
            h.update(block)
    return h.hexdigest()


def write(path, value):
    with path.open('x') as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write('\n')


cfg = json.loads((PREP / 'prepared.json').read_text())
assert len(cfg['inputs']) <= 512 and sum(row['bytes'] for row in cfg['inputs']) <= 256 * 1024**2
rows, unique, flagged = [], {}, []
for original in cfg['inputs']:
    path = ROOT / original['path']
    assert path.is_file() and not path.is_symlink() and '.local' not in path.relative_to(ROOT).parts
    assert path.stat().st_size == original['bytes'] and digest(path) == original['sha256']
    row = dict(original)
    reason = ('cache' if any(p in {'__pycache__', '.cache'} for p in path.parts) or path.suffix == '.pyc'
              else 'file larger than 8 MiB; original model/scene retained locally'
              if original['bytes'] > 8 * 1024**2 else None)
    if reason is None:
        data = path.read_bytes()
        hits = [name for name, pattern in PATTERNS.items() if re.search(pattern, data)]
        if hits:
            reason = 'credential-shaped artifact; original local bytes retained, not copied'
            flagged.append({'path': original['path'], 'marker_categories': hits})
    if reason:
        row.update(storage='hash_reference', local_path=str(path), reason=reason)
    else:
        unique.setdefault(row['sha256'], {'path': 'payload/' + row['sha256'] + path.suffix,
                                          'source': path, 'bytes': row['bytes']})
        row.update(storage='snapshot', snapshot=unique[row['sha256']]['path'])
    rows.append(row)
assert sum(row['bytes'] for row in unique.values()) <= 24 * 1024**2
for sha, row in unique.items():
    target = OUT / row['path']; target.parent.mkdir(exist_ok=True)
    with target.open('xb') as handle:
        handle.write(row['source'].read_bytes())
    assert target.stat().st_size == row['bytes'] and digest(target) == sha
by_path = {row['path']: row for row in rows}
for role in ['planner', 'author-0']:
    folder = JOB / 'agents' / role
    receipt = json.loads((folder / 'receipt.json').read_text())
    assert receipt['status'] == ('succeeded' if role == 'planner' else 'failed_or_uncertain')
    for artifact in receipt['artifacts']:
        row = by_path[str((folder / artifact['path']).relative_to(ROOT))]
        assert row['storage'] == 'snapshot' and row['sha256'] == artifact['sha256'] and row['bytes'] == artifact['bytes']
assert not (JOB / 'agents/author-0/response.json').exists()
assert not (JOB / 'agents/author-0/answer.json').exists()
assert not list(JOB.glob('model-*'))
pins = json.loads((JOB / 'input-receipt.json').read_text())
for artifact in pins['sources']:
    row = by_path[str((JOB / 'sources' / artifact['path']).relative_to(ROOT))]
    assert row['sha256'] == artifact['sha256'] and row['bytes'] == artifact['bytes']
preflight = json.loads((JOB / 'inspection/preflight.json').read_text())
assert preflight['skins'] == preflight['animations'] == 0 and preflight['source_unchanged']
for path in (JOB / 'inspection').glob('*.png'):
    assert by_path[str(path.relative_to(ROOT))]['storage'] == 'snapshot'
account = json.loads((ROOT / 'outputs/accepted-source-author-timeout-accounting-v1/receipt.json').read_text())
assert account['project_recorded_call_count'] == 38 and account['settled_call_count'] == 37
assert account['settled_microdollars'] == 3999655 and account['uncertain_calls'] == 1
assert account['accounting']['project_exposure_microdollars'] == 9618295
assert not account['next_full_call_fits'] and not account['actual_uncertain_charge_known']
assert account['accounting']['jobs']['studio-d80fa601c945434983e496db']['uncertain_reserved_microdollars'] == 5618640
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
    'prepared_source_pins_verified': len(pins['sources']), 'both_agent_artifact_receipts_verified': True,
    'inspection_views_retained': True, 'snapshot_secret_marker_hits': 0,
    'credential_shaped_references': flagged, 'marker_categories': list(PATTERNS),
    'elapsed_seconds': time.monotonic() - START, 'scope': 'Exact-byte archival checks, not a new job/process/ledger audit.',
})
with (OUT / 'README.md').open('x') as handle:
    handle.write('''# Accepted-source author timeout

The website successfully prepared the plan for `d80fa601c945434983e496db`. One Start then ran for 247.531631 seconds and reached an author API transport timeout after 240.009137 seconds. The planner succeeded and native inspection of the reused source completed. The source had **no skin or animation**; that inspection is not a rigging result. No author response body, response ID or answer was retained, and no rig/model attempt was built. This is an unresolved transport checkpoint, not accepted avatar quality or safe completion.

## Cost and ownership

The November 2, 2026, 4:06:14 PM PST accounting receipt records **38 calls: 37 settled at $3.999655 and one unknown call retaining a $5.618640 reservation**. The resulting $9.618295 exposure is not an actual invoice or a known timeout charge. The remaining $5.381705 does not cover another full $5.618640 reservation under the $15 ceiling. A timeout does not establish that a provider call was free.

The browser and server exits were verified, with no cancellation or signal. The separately preserved custody proof found all eight recorded local PIDs, direct children and process groups absent; ledger and registry hashes were unchanged. That local absence does not resolve remote billing. The unknown reservation and `needs_inspection` lease remain held. Without a provider response identity and usage, existing reconciliation cannot settle the charge faithfully; no release, retry eligibility or new call is claimed here.

## Preserved evidence

The bundle retains the frozen request, plan, quality policy, original source-code pins, both agent prompts/schemas/images/receipts, planner answer, source-inspection views, website Prepare v3 and Start reports, public accounting/custody receipts, prior publication receipt and the separate scratch repair-image inventory review. The scratch review remains a proposal/analytic record. A separately frozen follow-up records the narrow inventory fix and 14 passing analytic tests after local process absence was checked. Its saved source snapshots are included; no active runtime module is copied or executed. This fix does not resolve the API timeout or its charge.

Large original models/scenes and caches remain at their original paths with exact SHA256 references. `inventory.json` maps every selected original to a deduplicated snapshot or local reference. Private `.local` files, state databases and raw invocation/lease tokens are excluded. `verification.json` records source/artifact/marker checks; `summary.json` binds the retained outcomes; `manifest.json` seals this snapshot. Packaging performs no API/native/browser operation, database mutation, Git operation or retry.
''')
print(json.dumps({'records': len(rows), 'payload_files': len(unique),
                  'payload_bytes': sum(row['bytes'] for row in unique.values()),
                  'references': sum(row['storage'] == 'hash_reference' for row in rows)}))
