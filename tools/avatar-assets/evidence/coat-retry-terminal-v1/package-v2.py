from pathlib import Path
import hashlib
import json
import re
import sys
import time
import traceback

PREP = Path(__file__).resolve().parent
ROOT = PREP.parents[1]
OUT = ROOT/'evidence/coat-retry-terminal-v1'
PARENT = ROOT/'outputs/studio-1f98497e18e74820bda487ce'
NAMES = ['studio-1f98497e18e74820bda487ce', 'openai-panda-coat-retry-website-v1',
         'openai-panda-coat-studio-live-v1']
sys.path.insert(0, str(ROOT))


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


def inventory():
    result = []
    for name in NAMES:
        for path in sorted((ROOT/'outputs'/name).rglob('*')):
            assert not path.is_symlink()
            if path.is_dir():
                continue
            assert path.is_file() and '.local' not in path.relative_to(ROOT).parts
            assert path.name not in {'invocation.json', 'invocation-private.json'}
            result.append({'path': str(path.relative_to(ROOT)), 'bytes': path.stat().st_size,
                           'sha256': digest(path)})
    assert len(result) <= 4096 and sum(row['bytes'] for row in result) <= 1024**3
    return result


started = time.monotonic()
status = json.loads((PARENT/'status.json').read_text())
assert status['status'] == 'unsupported' and status['stage'] == 'repair_budget'
account_path = ROOT/'outputs/openai-panda-coat-studio-live-v1/accounting-terminal.json'
assert digest(account_path) == '3a572a18d7aebd59ccd0d7481a1bf105ff3ad310154371847a396a2d413eb124'
account = json.loads(account_path.read_text())
assert account['total_settled_microdollars'] == 2777078
assert account['calls'] == 29
assert account['uncertain_reserved_microdollars'] == 0
before = inventory()

from avatar_assets.pipeline_continuation import collect
continuation_started = time.monotonic()
try:
    captured = collect(ROOT, PARENT)
    continuation = {'status': 'supported', 'files': len(captured['inventory'])}
except ValueError as error:
    continuation = {'status': 'rejected', 'error': str(error),
                    'traceback': traceback.format_exc(limit=5)}
continuation.update({'elapsed_seconds': time.monotonic()-continuation_started,
                     'source_sha256': digest(ROOT/'avatar_assets/pipeline_continuation.py'),
                     'scope': 'Read-only real terminal collect; no provider, worker or grant calls'})
write(PREP/'continuation-before.json', continuation)

patterns = {
    'provider_key': rb'(?<![A-Za-z0-9])sk-(?:proj-)?[A-Za-z0-9_-]{20,}',
    'bearer': rb'Bearer\s+[A-Za-z0-9._-]{24,}',
    'private_key': rb'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----',
    'invocation': rb'(?i)["\x27](?:token|owner_token|lease_token|dispatch_token|studio_token|api_key|authorization)["\x27]\s*:\s*["\x27][A-Za-z0-9._-]{24,}["\x27]',
    'query_token': rb'[?&](?:token|key|api_key)=[A-Za-z0-9._-]{24,}',
}
rows, unique = [], {}
for old in before:
    path = ROOT/old['path']; row = dict(old)
    reason = ('bytecode cache' if '__pycache__' in path.parts or path.suffix == '.pyc'
              else 'file larger than8MiB' if old['bytes'] > 8*1024**2
              else 'individual frame sequence; review sheets/isolated images retained'
              if any(part in {'frames', 'renders'} for part in path.parts) else None)
    if reason:
        row.update(storage='hash_reference', local_path=str(path), reason=reason)
    else:
        if old['sha256'] not in unique:
            data = path.read_bytes()
            assert not any(re.search(pattern, data) for pattern in patterns.values()), 'Potential secret marker: '+old['path']
            unique[old['sha256']] = {'path': 'payload/'+old['sha256']+path.suffix,
                                    'bytes': old['bytes'], 'source': path}
        row.update(storage='snapshot', snapshot=unique[old['sha256']]['path'])
    rows.append(row)
assert sum(row['bytes'] for row in unique.values()) < 60*1024**2
for expected, row in unique.items():
    target = OUT/row['path']; target.parent.mkdir(exist_ok=True)
    with row['source'].open('rb') as source, target.open('xb') as dest:
        while block := source.read(1024**2):
            dest.write(block)
    assert target.stat().st_size == row['bytes'] and digest(target) == expected
assert inventory() == before
reviews = []
by_path = {row['path']: row for row in rows}
for path in sorted(PARENT.glob('review-*.json')):
    review = json.loads(path.read_text())
    for image in review['evidence']:
        row = by_path[str(Path(image['path']).relative_to(ROOT))]
        assert row['storage'] == 'snapshot' and row['sha256'] == image['sha256']
    reviews.append({'path': str(path.relative_to(ROOT)), 'sha256': digest(path),
                    'verdict': review['review']['verdict'],
                    'failed_checks': [row['name'] for row in review['review']['checks'] if not row['passed']],
                    'state_reviews': [{key: row[key] for key in ('id','meaning_passed','motion_passed')}
                                      for row in review['review']['state_reviews']]})
write(OUT/'inventory.json', rows)
write(OUT/'summary.json', {'schema_version': 1, 'status': 'unsupported', 'stage': 'repair_budget',
    'accepted_portable_result': False, 'reviews': reviews,
    'terminal_cost_microdollars': 2777078, 'terminal_api_calls': 29, 'terminal_unknown_holds': 0,
    'accounting_path': str(account_path.relative_to(ROOT)), 'accounting_sha256': digest(account_path),
    'accounting_scope': 'Historical read-only validated cost estimate, not invoice or later-successor totals',
    'continuation_before': continuation})
write(OUT/'verification.json', {'schema_version': 1, 'source_records': len(rows),
    'source_bytes': sum(row['bytes'] for row in rows), 'snapshot_records': sum(row['storage']=='snapshot' for row in rows),
    'hash_reference_records': sum(row['storage']=='hash_reference' for row in rows),
    'unique_payload_files': len(unique), 'unique_payload_bytes': sum(row['bytes'] for row in unique.values()),
    'all_source_hashes_unchanged': True, 'all_payload_hashes_and_sizes_verified': True,
    'all_critic_images_retained': True, 'secret_marker_hits': 0, 'secret_markers': list(patterns),
    'elapsed_seconds': time.monotonic()-started})
print(json.dumps({'records': len(rows), 'payload_files': len(unique),
                  'payload_bytes': sum(row['bytes'] for row in unique.values()),
                  'continuation': continuation['status'], 'error': continuation.get('error')}))
