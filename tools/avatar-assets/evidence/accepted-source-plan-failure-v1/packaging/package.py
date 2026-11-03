from pathlib import Path
import hashlib
import json
import re
import time

PREP = Path(__file__).resolve().parent
ROOT = PREP.parents[1]
OUT = ROOT / 'evidence/accepted-source-plan-failure-v1'
JOB = ROOT / 'outputs/studio-905fab75929b427ea12432a3'
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
before = cfg['inputs']
assert len(before) <= 512 and sum(row['bytes'] for row in before) <= 128 * 1024**2
rows, unique, flagged = [], {}, []
for original in before:
    path = ROOT / original['path']
    assert path.is_file() and not path.is_symlink() and '.local' not in path.relative_to(ROOT).parts
    assert path.stat().st_size == original['bytes'] and digest(path) == original['sha256']
    row = dict(original)
    reason = ('bytecode cache' if '__pycache__' in path.parts or path.suffix == '.pyc'
              else 'file larger than 8 MiB; original source retained locally'
              if original['bytes'] > 8 * 1024**2 else None)
    if reason is None:
        data = path.read_bytes()
        hits = [key for key, pattern in PATTERNS.items() if re.search(pattern, data)]
        if hits:
            reason = 'credential-shaped fixture; exact original retained locally without copying bytes'
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
    path = OUT / row['path']; path.parent.mkdir(exist_ok=True)
    with path.open('xb') as handle:
        handle.write(row['source'].read_bytes())
    assert path.stat().st_size == row['bytes'] and digest(path) == sha

by_path = {row['path']: row for row in rows}
receipt = json.loads((JOB / 'agents/planner/receipt.json').read_text())
assert receipt['status'] == 'succeeded' and receipt['actual_microdollars_upper_estimate'] == 74970
for row in receipt['artifacts']:
    path = JOB / 'agents/planner' / row['path']
    entry = by_path[str(path.relative_to(ROOT))]
    assert entry['storage'] == 'snapshot' and entry['bytes'] == row['bytes'] and entry['sha256'] == row['sha256']
pins = json.loads((JOB / 'input-receipt.json').read_text())
for row in pins['sources']:
    entry = by_path[str((JOB / 'sources' / row['path']).relative_to(ROOT))]
    assert entry['sha256'] == row['sha256'] and entry['bytes'] == row['bytes']
answer = json.loads((JOB / 'agents/planner/answer.json').read_text())
assert len(answer['generation_prompt']) == 845
schema = json.loads((JOB / 'agents/planner/schema.json').read_text())
assert schema['properties']['generation_prompt'] == {'type': 'string'}
old_validator = (JOB / 'sources/pipeline_prompts.py').read_text()
assert '("generation_prompt", 800)' in old_validator
assert 'generation_prompt <=800' in (JOB / 'agents/planner/prompt.txt').read_text()
status = json.loads((JOB / 'status.json').read_text())
assert status['status'] == 'checkpointed_error' and status['error'] == 'Invalid plan generation_prompt'
assert not (JOB / 'plan.json').exists()
assert not list((JOB / 'agents').glob('author-*')) and not list(JOB.glob('model-*'))
account_path = ROOT / 'outputs/accepted-source-panda-plan-failure-accounting-v1/receipt.json'
account = json.loads(account_path.read_text())
assert account['project_call_count'] == 36 and account['project_settled_microdollars'] == 3936227
assert account['uncertain_calls'] == 0
assert account['accounting']['jobs']['studio-905fab75929b427ea12432a3']['settled_actual_microdollars'] == 74970
reference = by_path[str((JOB / 'inputs/reference.png').relative_to(ROOT))]
assert reference['storage'] == 'snapshot' and reference['sha256'] == '17d78aa5bec052232ab3978b65c95545b230df218ce54388426d47b093b2a422'
for row in before:
    path = ROOT / row['path']
    assert path.stat().st_size == row['bytes'] and digest(path) == row['sha256']
write(OUT / 'inventory.json', rows)
write(OUT / 'summary.json', {
    'schema_version': 1, 'status': 'planner_semantic_validation_failed',
    'job_id': '905fab75929b427ea12432a3', 'output_status': status,
    'canonical_status_at_assignment': 'needs_inspection with retained lease; parent-reported, not queried by this packager',
    'planner_transport': 'succeeded', 'generation_prompt_characters': 845,
    'semantic_maximum_characters': 800,
    'saved_transport_schema': schema['properties']['generation_prompt'],
    'transport_schema_note': 'The original saved schema admits string without maxLength; the saved prompt and semantic validator impose 800. All bytes remain unchanged.',
    'planner_answer_sha256': receipt['answer_sha256'], 'planner_cost_microdollars': 74970,
    'planner_receipt_path': str((JOB / 'agents/planner/receipt.json').relative_to(ROOT)),
    'accounting_path': str(account_path.relative_to(ROOT)), 'accounting_sha256': digest(account_path),
    'project_calls_observed': 36, 'project_settled_microdollars_observed': 3936227,
    'uncertain_calls_observed': 0, 'accounting_observed_local': account['observed_local'],
    'reference_sha256': reference['sha256'],
    'source_sha256': by_path[str((JOB / 'inputs/source-model.glb').relative_to(ROOT))]['sha256'],
    'author_or_native_started': False, 'accepted_result': False, 'retry_outcome': 'not included',
    'private_state_accessed': False, 'runtime_or_database_mutations': False,
})
write(OUT / 'verification.json', {
    'schema_version': 1, 'source_records': len(rows), 'source_bytes': sum(row['bytes'] for row in rows),
    'snapshot_records': sum(row['storage'] == 'snapshot' for row in rows),
    'hash_reference_records': sum(row['storage'] == 'hash_reference' for row in rows),
    'unique_payload_files': len(unique), 'unique_payload_bytes': sum(row['bytes'] for row in unique.values()),
    'frozen_input_source_pins_verified': len(pins['sources']),
    'planner_artifact_receipt_verified': True, 'selected_portrait_retained': True,
    'all_input_and_payload_hashes_verified': True, 'all_selected_source_bytes_unchanged': True,
    'snapshot_secret_marker_hits': 0, 'credential_shaped_references': flagged,
    'marker_categories': list(PATTERNS), 'elapsed_seconds': time.monotonic() - START,
    'scope': 'Byte custody and saved failure facts only; no API, native rendering, process control or current source execution.',
})
with (OUT / 'README.md').open('x') as handle:
    handle.write('''# Accepted-source planner failure

The fresh planner call for the previously generated panda/shoulder-dragon source returned successfully, then local validation rejected its **845-character generation prompt against the existing 800-character limit**. The original transport schema only required a string; the saved prompt and local semantic validator imposed the length limit. This bundle preserves all three unchanged. No author, rig build or native render started, and no accepted avatar resulted.

The selected portrait, exact user request, source-specific grant receipt, source hashes, saved planner prompt/schema/answer/API receipt and original prepared source snapshots are preserved. The large unrigged source model remains unchanged at its original local path with its exact hash. This is not a new Meshy generation or reuse of an old motion recipe.

The first website preparation check failed before server readiness. The second submitted one Prepare but its harness stopped on the intermediate launching state; the detached planner later reached the semantic failure above. Both original browser reports and the separate observation/unplanned-inspection test receipts remain intact. Those test receipts explicitly identify analyst-recorded results whose raw stdout remains in the session; this archive does not invent missing logs or replay tests.

The public accounting receipt observed **36 settled calls / $3.936227 cumulative** at November 2, 2026, 3:55:35 PM PST, including **$0.074970 for this single planner call**, with zero uncertain calls. These are recorded usage estimates, not an invoice or later retry totals. At assignment, the canonical job needed inspection and retained its lease; this packager did not query or release private state. No later release, repair or retry outcome is claimed.

`inventory.json` maps exact source bytes to deduplicated payloads or original local-path hash references. `verification.json` records source-pin, API-artifact, portrait and secret-marker checks. `summary.json` binds failure facts and historical accounting. `packaging/` preserves the bounded archive command and receipt; `manifest.json` seals the bundle. No `.local` contents, secret invocations, state databases, active source edits or future job artifacts are copied.
''')
print(json.dumps({'records': len(rows), 'payload_files': len(unique),
                  'payload_bytes': sum(row['bytes'] for row in unique.values()),
                  'references': sum(row['storage'] == 'hash_reference' for row in rows)}))
