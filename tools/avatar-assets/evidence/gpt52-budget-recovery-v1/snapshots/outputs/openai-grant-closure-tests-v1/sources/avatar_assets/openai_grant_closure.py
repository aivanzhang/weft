"""Close unused authorization without releasing, settling or retiring any call."""

import hashlib
import json
import os
from pathlib import Path
import re
import stat

from avatar_assets.safety import require

POLICY = 'close-unused-grant-preserve-charges-v1'
MAX_RECORD_BYTES = 256 * 1024
MAX_RELEASE_BYTES = 512 * 1024
_CALL_FIELDS = ('job', 'role', 'request_digest', 'reserved', 'state', 'response_id', 'actual', 'usage')


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def _hash(value):
    return hashlib.sha256(_canonical(value)).hexdigest()


def _hex(value):
    return isinstance(value, str) and re.fullmatch('[0-9a-f]{64}', value) is not None


def _rows(db, sql, parameters=()):
    cursor = db.execute(sql, parameters)
    names = [row[0] for row in cursor.description]
    return [dict(zip(names, row)) for row in cursor]


def ensure_schema(db):
    db.execute('CREATE TABLE IF NOT EXISTS grant_closures '
               '(parent TEXT PRIMARY KEY, digest TEXT NOT NULL, record TEXT NOT NULL)')


def _exists(db):
    return db.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='grant_closures'").fetchone() is not None


def _inventories(db, parents):
    from avatar_assets.openai_runner import attach_call_profiles, call_exposure
    require(db.execute('SELECT COUNT(*) FROM calls').fetchone()[0] <= 10000,
            'API call inventory exceeds10000')
    calls = _rows(db, 'SELECT '+','.join(_CALL_FIELDS)+' FROM calls ORDER BY job,role')
    bound = attach_call_profiles(db, calls)
    result = {parent: [] for parent in parents}
    for raw, call in zip(calls, bound):
        if raw['job'] not in result:
            continue
        call_exposure(call)
        if raw['state'] == 'reserved_uncertain':
            require(all(raw[name] is None for name in ('response_id', 'actual', 'usage')),
                    'Unknown call contains unvalidated settlement fields')
        result[raw['job']].append({'call': raw, 'profile_binding': _canonical(call['model_profile']).decode()
                                  if 'model_profile' in call else None})
        require(len(result[raw['job']]) <= 12, 'Closed grant call inventory exceeds12')
    return result


def _inventory(db, parent):
    return _inventories(db, {parent})[parent]


def _grant(row):
    require(isinstance(row['approval'], str) and len(row['approval'].encode()) <= 8192
            and type(row['calls']) is int and 1 <= row['calls'] <= 12
            and type(row['ceiling']) is int and 1 <= row['ceiling'] <= 1000000000
            and _hex(row['digest']), 'Invalid stored job approval')
    return {'digest': row['digest'], 'calls': row['calls'], 'ceiling': row['ceiling'],
            'approval_sha256': hashlib.sha256(row['approval'].encode()).hexdigest()}


def _approval(value, parent, parent_digest, receipt_sha256):
    from avatar_assets.openai_runner import _approval as validate_approval
    record = validate_approval(value)
    require(value.get('operation') == 'close_unused_grant'
            and value.get('parent_job_id') == parent
            and value.get('parent_request_sha256') == parent_digest
            and value.get('release_receipt_sha256') == receipt_sha256,
            'Grant closure approval must bind parent, request and local-release receipt')
    return record


def _validate_record(db, parent, parent_digest, raw, accounting, current):
    from avatar_assets.openai_runner import call_exposure
    require(isinstance(raw, str) and len(raw.encode()) <= MAX_RECORD_BYTES,
            'Invalid bounded grant closure record')
    try:
        value = json.loads(raw)
    except (ValueError, TypeError) as error:
        raise ValueError('Invalid grant closure JSON') from error
    require(_canonical(value).decode() == raw and isinstance(value, dict)
            and set(value) == {'schema_version', 'policy', 'parent', 'digest', 'grant', 'calls',
                              'calls_sha256', 'exposure_microdollars', 'release', 'approval'},
            'Invalid canonical grant closure record')
    require(value['schema_version'] == 1 and value['policy'] == POLICY
            and value['parent'] == parent and value['digest'] == parent_digest
            and _hex(parent_digest), 'Grant closure identity differs')
    parents = _rows(db, 'SELECT * FROM jobs WHERE id=?', (parent,))
    require(len(parents) == 1 and value['grant'] == _grant(parents[0])
            and parents[0]['digest'] == parent_digest, 'Closed grant history changed')
    require(parent not in accounting['retired_jobs'], 'A closed grant cannot also be retired')
    release = value['release']
    require(isinstance(release, dict) and set(release) == {'path', 'sha256'}
            and _hex(release['sha256']) and isinstance(release['path'], str)
            and len(release['path']) <= 512 and '\\' not in release['path']
            and '..' not in Path(release['path']).parts and not Path(release['path']).is_absolute(),
            'Invalid grant closure release binding')
    _approval(value['approval'], parent, parent_digest, release['sha256'])
    old = value['calls']
    require(isinstance(old, list) and 1 <= len(old) <= 12 and _hash(old) == value['calls_sha256'],
            'Closed call inventory hash differs')
    require(len(old) == len(current), 'Closed grant gained or lost a call')
    original_exposure = 0
    for before, after in zip(old, current):
        require(isinstance(before, dict) and set(before) == {'call', 'profile_binding'}
                and isinstance(before['call'], dict) and set(before['call']) == set(_CALL_FIELDS)
                and before['profile_binding'] == after['profile_binding'], 'Closed call profile changed')
        a, b = before['call'], after['call']
        require(all(a[key] == b[key] for key in ('job', 'role', 'request_digest', 'reserved')),
                'Closed call identity or reservation changed')
        bound = dict(a)
        if before['profile_binding'] is not None:
            bound['model_profile'] = json.loads(before['profile_binding'])
        original_exposure += call_exposure(bound)
        if a['state'] == 'reserved_uncertain':
            require(all(a[key] is None for key in ('response_id', 'actual', 'usage')),
                    'Closed unknown call had invalid settlement fields')
            require(b['state'] in ('reserved_uncertain', 'settled'), 'Invalid closed-call transition')
        else:
            require(a == b, 'Closed settled call history changed')
    require(type(value['exposure_microdollars']) is int
            and value['exposure_microdollars'] == original_exposure <= parents[0]['ceiling'],
            'Closed grant exposure differs from its original calls')
    current_exposure = accounting['jobs'][parent]['effective_exposure_microdollars']
    require(current_exposure <= original_exposure, 'Closed exposure increased')
    return {**value, 'current_exposure_microdollars': current_exposure}


def grant_closures(db, accounting=None):
    """Legacy read-only databases need no schema migration."""
    if not _exists(db):
        return {}
    require(db.execute('SELECT COUNT(*) FROM grant_closures').fetchone()[0] <= 10000,
            'Grant closure inventory exceeds10000')
    from avatar_assets.openai_runner import accounting_snapshot
    accounting = accounting if accounting is not None else accounting_snapshot(db)
    result = {}
    records = list(db.execute('SELECT parent,digest,record FROM grant_closures ORDER BY parent'))
    if not records:
        return result
    inventories = _inventories(db, {row[0] for row in records})
    for parent, parent_digest, raw in records:
        result[parent] = _validate_record(db, parent, parent_digest, raw, accounting, inventories[parent])
    return result


def require_open_grant(db, job_id):
    require(job_id not in grant_closures(db), 'Closed API grant cannot reserve or continue')


def allocated_grants(db):
    from avatar_assets.openai_runner import accounting_snapshot
    accounting = accounting_snapshot(db)
    closed = grant_closures(db, accounting)
    require(db.execute('SELECT COUNT(*) FROM jobs').fetchone()[0] <= 10000,
            'API grant inventory exceeds10000')
    allocated = accounting['retired_actual_microdollars']
    for job, ceiling in db.execute('SELECT id,ceiling FROM jobs'):
        require(type(ceiling) is int and 1 <= ceiling <= 1000000000, 'Invalid stored job ceiling')
        if job in accounting['retired_jobs']:
            continue
        allocated += closed[job]['current_exposure_microdollars'] if job in closed else ceiling
    return allocated


def _release(root, parent, parent_digest, path, expected_sha256):
    from avatar_assets.studio_unknown_release import verify_release
    root, path = Path(root).resolve(), Path(path)
    if not path.is_absolute():
        path = root / path
    require(_hex(expected_sha256) and not any(p.is_symlink() for p in (path, *path.parents)),
            'Invalid grant closure release path or hash')
    expected_folder = root / '.local/studio/jobs' / parent.removeprefix('studio-')
    require(path.parent == expected_folder and re.fullmatch(r'uncertain-owner-release-[0-9a-f]{32}\.json', path.name),
            'Grant closure requires the canonical local-release receipt')
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(fd, 'rb') as handle:
        info = os.fstat(handle.fileno())
        require(stat.S_ISREG(info.st_mode) and info.st_uid == os.getuid()
                and stat.S_IMODE(info.st_mode) == 0o600 and 0 < info.st_size <= MAX_RELEASE_BYTES,
                'Grant closure release receipt must be bounded and owner-only')
        raw = handle.read(MAX_RELEASE_BYTES + 1)
    require(len(raw) == info.st_size and hashlib.sha256(raw).hexdigest() == expected_sha256,
            'Grant closure release receipt changed')
    record = verify_release(root, path, expected_sha256, parent_job_id=parent, parent_digest=parent_digest)
    require(record == json.loads(raw), 'Validated local release differs from its saved bytes')
    return record, {'path': str(path.relative_to(root)), 'sha256': expected_sha256}


def close_unused_grant(ledger, *, root, parent_id, parent_digest, release_receipt,
                       release_sha256, approval):
    """One transaction preserves unknown holds and permanently closes future spending."""
    from avatar_assets.openai_runner import accounting_snapshot, budget_snapshot
    require(isinstance(parent_id, str) and re.fullmatch('studio-[0-9a-f]{24}', parent_id)
            and _hex(parent_digest), 'Invalid Studio parent identity')
    root = Path(root).resolve()
    require(Path(ledger.folder).resolve() == root / '.local/openai',
            'Grant closure must use the project canonical API ledger')
    approval_raw = _approval(approval, parent_id, parent_digest, release_sha256)
    with ledger.transaction():
        record, release = _release(root, parent_id, parent_digest, release_receipt, release_sha256)
        require(record.get('requested_transition') == 'release_local_owner_preserve_uncertain_charge'
                and record.get('no_dispatch') is True and record.get('no_cost_reset') is True
                and record.get('old_job_restart_allowed') is False,
                'Local release does not preserve unknown charge and prohibit restart')
        current = grant_closures(ledger.db)
        if parent_id in current:
            saved = current[parent_id]
            require(saved['digest'] == parent_digest and saved['release'] == release
                    and _canonical(saved['approval']).decode() == approval_raw,
                    'Existing grant closure binding or approval differs')
            return {**saved, 'idempotent': True}
        parent = _rows(ledger.db, 'SELECT * FROM jobs WHERE id=?', (parent_id,))
        require(len(parent) == 1 and parent[0]['digest'] == parent_digest,
                'Grant closure parent binding differs')
        accounting = accounting_snapshot(ledger.db)
        require(parent_id not in accounting['retired_jobs'], 'Retired grant cannot be closed')
        calls = _inventory(ledger.db, parent_id)
        require(1 <= len(calls) <= parent[0]['calls'], 'Grant closure requires recorded parent calls')
        proof = record.get('proof', {}).get('unknown_provider', {})
        witness = proof.get('uncertain_call')
        require(proof.get('policy') == 'unknown-api-author-custody-v1'
                and proof.get('job_id') == parent_id and proof.get('request_sha256') == parent_digest
                and isinstance(witness, dict), 'Local release API parent proof differs')
        require(proof.get('call_rows') == calls and proof.get('grant') == _grant(parent[0]),
                'Local release call inventory or grant changed before closure')
        matches = [row['call'] for row in calls if row['call']['role'] == witness.get('role')]
        require(len(matches) == 1 and matches[0]['state'] == witness.get('state') == 'reserved_uncertain'
                and matches[0]['request_digest'] == witness.get('request_digest')
                and matches[0]['reserved'] == witness.get('reserved_microdollars')
                and witness.get('response_id') is None and witness.get('actual_microdollars') is None,
                'Local release unknown call differs from current ledger')
        exposure = accounting['jobs'][parent_id]['effective_exposure_microdollars']
        budget = budget_snapshot(ledger.db)
        require(budget is not None and accounting['project_exposure_microdollars'] <= budget['effective_ceiling_microdollars']
                and exposure <= parent[0]['ceiling'], 'Grant closure cannot conceal excess exposure')
        value = {'schema_version': 1, 'policy': POLICY, 'parent': parent_id, 'digest': parent_digest,
                 'grant': _grant(parent[0]), 'calls': calls, 'calls_sha256': _hash(calls),
                 'exposure_microdollars': exposure, 'release': release,
                 'approval': json.loads(approval_raw)}
        raw = _canonical(value).decode()
        require(len(raw.encode()) <= MAX_RECORD_BYTES, 'Grant closure record exceeds256KiB')
        ledger.db.execute('INSERT INTO grant_closures VALUES(?,?,?)', (parent_id, parent_digest, raw))
        saved = grant_closures(ledger.db, accounting)[parent_id]
        require(allocated_grants(ledger.db) <= budget['effective_ceiling_microdollars'],
                'Other active allocations still exceed the project ceiling')
        return {**saved, 'idempotent': False}
