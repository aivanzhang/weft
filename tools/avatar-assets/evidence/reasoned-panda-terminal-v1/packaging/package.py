from pathlib import Path
from collections import Counter
import hashlib
import json
import re
import stat
import sys
import time

PREP = Path(__file__).resolve().parent
ROOT = PREP.parents[1]
OUT = ROOT / 'evidence/reasoned-panda-terminal-v1'
NAMES = [
    'studio-d8a680ce6580429f9fdd016f',
    'openai-panda-reasoned-resumed-studio-live-v1',
    'd8-coat-body-ablation-v1', 'd8-coat-body-ablation-v2',
    'd8-body-normal-ablation-v1', 'd8-posed-normal-diagnostic-v1',
    'd8-rig-weight-diagnosis-v1', 'continuation-inventory-capacity-tests-v1',
    'repair-visual-input-fix-v1', 'repair-visual-input-tests-v2',
]
PATTERNS = {
    'provider_key': rb'(?<![A-Za-z0-9])sk-(?:proj-)?[A-Za-z0-9_-]{20,}',
    'bearer': rb'Bearer\s+[A-Za-z0-9._-]{24,}',
    'private_key': rb'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----',
    'invocation': rb'(?i)["\x27](?:token|owner_token|lease_token|dispatch_token|studio_token|api_key|authorization)["\x27]\s*:\s*["\x27][A-Za-z0-9._-]{24,}["\x27]',
    'query_token': rb'[?&](?:token|key|api_key)=[A-Za-z0-9._-]{24,}',
}
STARTED = time.monotonic()


def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as handle:
        while block := handle.read(1024**2):
            assert time.monotonic() - STARTED < 25, 'Packaging deadline exceeded'
            value.update(block)
    return value.hexdigest()


def write(path, value):
    with path.open('x') as handle:
        json.dump(value, handle, sort_keys=True, indent=2, allow_nan=False)
        handle.write('\n')


def inventory():
    rows = []
    total = 0
    for name in NAMES:
        folder = ROOT / 'outputs' / name
        assert folder.is_dir() and not folder.is_symlink()
        for path in sorted(folder.rglob('*')):
            mode = path.lstat().st_mode
            assert not stat.S_ISLNK(mode), 'Symlink in source inventory'
            if stat.S_ISDIR(mode):
                continue
            assert stat.S_ISREG(mode), 'Nonregular source artifact'
            relative = path.relative_to(ROOT)
            assert '.local' not in relative.parts
            assert path.name not in {'invocation.json', 'invocation-private.json'}
            size = path.stat().st_size
            total += size
            assert len(rows) < 4096 and total <= 1024**3
            rows.append({'path': str(relative), 'bytes': size, 'sha256': digest(path)})
    return rows


parent = ROOT / 'outputs' / NAMES[0]
status = json.loads((parent / 'status.json').read_text())
assert status['status'] == 'unsupported' and status['stage'] == 'repair_budget'
account_path = ROOT / 'outputs' / NAMES[1] / 'accounting-terminal.json'
account = json.loads(account_path.read_text())
assert account['total_settled_microdollars'] == 3861257 and account['calls'] == 35
assert account['uncertain_reserved_microdollars'] == 0
assert account['current_job']['calls'] == 6
assert account['current_job']['settled_actual_microdollars'] == 1084179
assert account['current_job']['uncertain_reserved_microdollars'] == 0
browser_path = ROOT / 'outputs' / NAMES[1] / 'studio-live-test.json'
browser = json.loads(browser_path.read_text())
assert browser['agent_backend']['reasoning_effort'] == 'medium'
assert browser['cleanup']['browser_exit_verified'] and browser['cleanup']['server_exit_verified']
assert not browser['cleanup']['worker_cancel_requested']
assert len(browser['mutations']) == 1 and browser['mutations'][0]['path'].endswith('/start')
ownership = json.loads((browser_path.parent / 'ownership-after.json').read_text())
assert ownership['owner']['active_lease_job'] is None and ownership['owner']['queued_jobs'] == []
before = inventory()
rows, unique, flagged = [], {}, []
for original in before:
    path = ROOT / original['path']
    row = dict(original)
    reason = ('bytecode cache' if '__pycache__' in path.parts or path.suffix == '.pyc'
              else 'file larger than 8 MiB' if row['bytes'] > 8 * 1024**2
              else 'individual frame or pose sequence; critic sheets and isolated images retained'
              if any(part in {'frames', 'renders'} for part in path.parts) else None)
    if reason is None:
        data = path.read_bytes()
        hits = [name for name, pattern in PATTERNS.items() if re.search(pattern, data)]
        if hits:
            reason = 'credential-shaped markers; original retained locally without publishing its bytes'
            flagged.append({'path': original['path'], 'marker_categories': hits})
    if reason:
        row.update(storage='hash_reference', local_path=str(path), reason=reason)
    else:
        if row['sha256'] not in unique:
            unique[row['sha256']] = {'path': 'payload/' + row['sha256'] + path.suffix,
                                      'bytes': row['bytes'], 'source': path}
        row.update(storage='snapshot', snapshot=unique[row['sha256']]['path'])
    rows.append(row)
assert sum(row['bytes'] for row in unique.values()) <= 90 * 1024**2
for expected, row in unique.items():
    target = OUT / row['path']
    target.parent.mkdir(exist_ok=True)
    with row['source'].open('rb') as source, target.open('xb') as dest:
        while block := source.read(1024**2):
            dest.write(block)
    assert target.stat().st_size == row['bytes'] and digest(target) == expected
assert inventory() == before, 'Source inventory changed during packaging'
by_path = {row['path']: row for row in rows}
reviews = []
for path in sorted(parent.glob('review-*.json')):
    review = json.loads(path.read_text())
    assert review['review']['verdict'] == 'repair'
    for image in review['evidence']:
        item = by_path[str(Path(image['path']).relative_to(ROOT))]
        assert item['storage'] == 'snapshot' and item['sha256'] == image['sha256']
    reviews.append({'path': str(path.relative_to(ROOT)), 'sha256': digest(path),
                    'verdict': review['review']['verdict'], 'summary': review['review']['summary'],
                    'retained_evidence_images': len(review['evidence'])})
assert len(reviews) == 3
for row in rows:
    path = Path(row['path'])
    if path.suffix.lower() == '.mp4' or path.name in {'portrait.png', 'portrait.webp'}:
        assert row['storage'] == 'snapshot', 'Requested video/portrait not retained'
write(OUT / 'inventory.json', rows)
write(OUT / 'summary.json', {
    'schema_version': 1, 'status': 'unsupported', 'stage': 'repair_budget',
    'medium_api_executed': True, 'accepted_portable_result': False,
    'reviews': reviews, 'source_roots': ['outputs/' + name for name in NAMES],
    'terminal_accounting': account, 'accounting_path': str(account_path.relative_to(ROOT)),
    'accounting_sha256': digest(account_path),
    'accounting_scope': 'Historical validated usage estimate, not invoice or current successor totals.',
    'browser': {'path': str(browser_path.relative_to(ROOT)), 'sha256': digest(browser_path),
                'starts': 1, 'elapsed_seconds': browser['elapsed_seconds'],
                'cleanup': browser['cleanup'], 'active_lease_job': None, 'queued_jobs': []},
    'diagnostic_conclusion': 'Large angular body artifacts remained with the coat hidden, normal-map influence disconnected, and freshly recalculated posed normals. These controls rule out each removed factor as the sole cause; they do not identify a unique cause or prove a repair.',
    'diagnostic_limits': ['The first sandboxed coat-ablation attempt failed before rendering and remains preserved.',
                          'Subsequent diagnostic renders are transient static controls, not modified accepted assets.',
                          'The skin-weight diagnosis and numeric gates do not establish visual quality.'],
    'continuation_inventory': {'path': 'outputs/continuation-inventory-capacity-tests-v1/actual-result.json',
                               'scope': 'Previously completed read-only terminal collection; not rerun for this archive.'},
    'later_retirement': {'scope': 'Parent-reported after this terminal snapshot',
                         'statement': 'The old job ledger was retired to the fresh-source continuation without resetting history. No ledger was read or changed by this packaging job.'},
    'excluded_work': 'The new fresh-source905 job and private .local files are not inventoried or copied.',
})
write(OUT / 'verification.json', {
    'schema_version': 1, 'source_records': len(rows), 'source_bytes': sum(row['bytes'] for row in rows),
    'snapshot_records': sum(row['storage'] == 'snapshot' for row in rows),
    'hash_reference_records': sum(row['storage'] == 'hash_reference' for row in rows),
    'hash_reference_reasons': dict(Counter(row.get('reason') for row in rows if row['storage'] == 'hash_reference')),
    'unique_payload_files': len(unique), 'unique_payload_bytes': sum(row['bytes'] for row in unique.values()),
    'all_source_hashes_unchanged': True, 'all_payload_hashes_and_sizes_verified': True,
    'all_18_critic_images_retained': True, 'all_videos_and_portraits_retained': True,
    'snapshot_secret_marker_hits': 0, 'hash_referenced_marker_records': flagged,
    'secret_markers': list(PATTERNS), 'elapsed_seconds': time.monotonic() - STARTED,
    'verification_scope': 'Exact bytes and inventory custody; no media decoding, new tests, models, database calls or quality acceptance.',
})
with (OUT / 'README.md').open('x') as handle:
    handle.write('''# Medium-reasoning panda terminal evidence

The real website/backend run executed the medium-reasoning OpenAI API, but all three local attempts received a repair verdict. The terminal result is **unsupported**, not an accepted portable avatar or completed hands-off quality demonstration.

The browser sent one Start, ran for 1,127.688947 seconds, and verified browser/server shutdown. The saved terminal ownership receipt shows no active lease or queued job. At November 2, 2026, 3:29:21 PM PST, historical cumulative validated usage was **$3.861257 across 35 calls**; this run used **$1.084179 across six calls**, with no uncertain holds. These are usage estimates, not invoices or current successor totals. The parent subsequently retired this job into a fresh-source continuation without resetting history; this archive did not read or mutate that ledger or snapshot the new job.

## Diagnostic scope

The first sandboxed coat-ablation attempt failed before rendering. The preserved successful controls used the same exported model, 2-second pose, camera and lighting: coat hidden, normal-map influence disconnected, then fresh smooth normals on a temporary static posed copy. Large angular body artifacts remained. Fur cards, normal-map influence and stored/skinned normals are therefore not individually sufficient explanations. Local shading changed; these tests do not prove a unique cause, a repair or accepted animation quality. No original model or scene was overwritten.

The rig-weight diagnosis, bounded continuation inventory correction and repair-image input tests are included as separate evidence. The previous real collection admitted 1,175 files / 641,261,780 bytes under finite 4,096-file / 1 GiB / 8,192-entry inventory bounds; it was not rerun during packaging.

## Custody and compact storage

`inventory.json` maps every original artifact to a deduplicated exact-byte payload or a local-path SHA256 reference. All 18 critic evidence images, portraits, videos, contact sheets, rest views, diagnostic images, JSON, logs and eligible source snapshots are retained. Individual frame/pose sequences, files larger than 8 MiB, bytecode caches and credential-shaped fixtures remain unchanged in their original folders and are hash-referenced. Private `.local` files and invocation tokens are not copied; public receipts may retain their hash/path references.

`summary.json` binds terminal accounting, browser evidence and critic decisions. `verification.json` records source/payload hash checks and marker scanning. `packaging/` holds the exact script, preparation and bounded-job receipt. `manifest.json` seals this bundle. No provider, native renderer, browser, database mutation, Git operation, asset promotion or new acceptance test was performed for packaging.
''')
print(json.dumps({'records': len(rows), 'snapshots': sum(row['storage'] == 'snapshot' for row in rows),
                  'references': sum(row['storage'] == 'hash_reference' for row in rows),
                  'payload_files': len(unique), 'payload_bytes': sum(row['bytes'] for row in unique.values()),
                  'credential_shaped_references': len(flagged)}))
