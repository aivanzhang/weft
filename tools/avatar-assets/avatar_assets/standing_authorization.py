"""Finite, source-bound retry grants under an existing owner-approved dollar ceiling."""
from contextlib import closing
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import stat

from avatar_assets.agent_backend import validate_settings, _private_file
from avatar_assets.openai_runner import (OpenAILedger, accounting_snapshot,
    budget_snapshot, digest, reserve_microdollars, price_profile, _decode)
from avatar_assets.safety import require, slug

POLICY_FILE = '.local/openai-standing-authorization.json'
MAX_TOTAL = 1_000_000_000
POLICY_FIELDS = {'schema_version', 'status', 'policy_id', 'approval_receipt',
    'max_total_microdollars', 'model', 'max_calls', 'max_repairs', 'max_successors',
    'allowed_source_sha256', 'lineage_root_job_id', 'lineage_root_request_sha256'}


def _hash(value):
    require(isinstance(value, str) and len(value) == 64 and all(c in '0123456789abcdef' for c in value),
            'Invalid standing authorization hash')
    return value


def _read(path, maximum, *, private=False):
    path = Path(path)
    require(not any(p.is_symlink() for p in (path, *path.parents)), 'Standing authorization path contains a symlink')
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(fd, 'rb') as stream:
        info = os.fstat(fd)
        require(stat.S_ISREG(info.st_mode) and 0 < info.st_size <= maximum,
                'Standing authorization file is missing, invalid or oversized')
        if private:
            require(info.st_uid == os.getuid() and stat.S_IMODE(info.st_mode) == 0o600,
                    'Standing authorization policy must be owner-only mode0600')
        raw = stream.read(maximum+1)
        require(len(raw) == info.st_size, 'Standing authorization file changed while reading')
    return _decode(raw), hashlib.sha256(raw).hexdigest()


def _relative(root, value):
    require(isinstance(value, str) and value and not Path(value).is_absolute()
            and '..' not in Path(value).parts, 'Invalid standing authorization relative path')
    path = root/value
    require(path.resolve().is_relative_to(root) and not any(p.is_symlink() for p in (path, *path.parents)),
            'Standing authorization path escapes its project')
    return path


def _policy(root):
    value, policy_hash = _read(root/POLICY_FILE, 16384, private=True)
    require(isinstance(value, dict) and set(value) == POLICY_FIELDS
            and type(value['schema_version']) is int and value['schema_version'] == 1,
            'Invalid standing authorization policy fields')
    require(value['status'] == 'active', 'Standing retry authorization is absent or revoked')
    slug(value['policy_id']); slug(value['lineage_root_job_id'])
    _hash(value['allowed_source_sha256']); _hash(value['lineage_root_request_sha256'])
    for field, low, high in (('max_total_microdollars', 1, MAX_TOTAL), ('max_calls', 3, 7),
                             ('max_repairs', 0, 2), ('max_successors', 1, 16)):
        require(type(value[field]) is int and low <= value[field] <= high, 'Invalid standing '+field)
    price_profile(value['model'])
    require(value['max_calls'] >= 3+2*value['max_repairs'],
            'Standing AI settings are unsupported')
    selected = value['approval_receipt']
    require(isinstance(selected, dict) and set(selected) == {'path', 'sha256'}, 'Invalid approval receipt binding')
    approval, actual = _read(_relative(root, selected['path']), 65536)
    require(actual == _hash(selected['sha256']), 'Standing approval receipt changed')
    require(isinstance(approval, dict) and approval.get('decision') == 'approved'
            and approval.get('standing_retries') is True
            and type(approval.get('max_total_microdollars')) is int
            and approval['max_total_microdollars'] == value['max_total_microdollars']
            and isinstance(approval.get('user_reply'), str) and 0 < len(approval['user_reply']) <= 4000,
            'Standing approval receipt does not authorize this policy')
    return value, policy_hash, approval


def standing_policy_status(root):
    """Small UI summary only; eligibility still checks the source, lineage and ledger."""
    try:
        value, policy_hash, _ = _policy(Path(root).resolve())
        return {'enabled': True, 'policy_sha256': policy_hash,
                'source_sha256': value['allowed_source_sha256'],
                **{k:value[k] for k in ('max_total_microdollars', 'max_calls', 'max_repairs', 'max_successors')},
                'reason': 'Standing source-bound retries are configured; each retry rechecks custody and remaining allowance.'}
    except (OSError, ValueError, TypeError):
        return {'enabled': False, 'reason': 'Standing retry authorization is absent, revoked or requires inspection.'}


def _request(root, output, policy, *, verify_source):
    output = Path(output)
    require(output.parent == root/'outputs' and output.resolve() == output, 'Retry output is not a canonical project job')
    slug(output.name)
    value, request_hash = _read(output/'request.json', 1024**2)
    require(isinstance(value, dict) and value.get('schema_version') == 1 and value.get('id') == output.name
            and value.get('max_credits') == 0 and value.get('reference') is None,
            'Standing retries require an exact saved-source request without new generation or references')
    source = value.get('reuse_model')
    require(isinstance(source, dict) and set(source) == {'path', 'original_path', 'bytes', 'sha256'}
            and source['sha256'] == policy['allowed_source_sha256']
            and type(source['bytes']) is int and 20 <= source['bytes'] <= 64*1024**2,
            'Retry source differs from the approved source')
    if verify_source:
        path = _relative(output, source['path'])
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        with os.fdopen(fd, 'rb') as stream:
            info = os.fstat(fd)
            require(stat.S_ISREG(info.st_mode) and info.st_size == source['bytes'], 'Saved retry source size changed')
            require(hashlib.file_digest(stream, 'sha256').hexdigest() == source['sha256'], 'Saved retry source bytes changed')
    settings = validate_settings(value.get('agent_backend'))
    require(settings['model'] == policy['model'] and settings['max_calls'] == policy['max_calls']
            and settings['max_repairs'] == policy['max_repairs']
            and value.get('max_agent_calls') == settings['max_calls']
            and value.get('max_repairs') == settings['max_repairs']
            and value.get('agent_model') == settings['model'] and value.get('agent_timeout') == settings['timeout'],
            'Retry limits differ from standing authorization')
    continuation = None
    if 'continuation' in value:
        from avatar_assets.pipeline_continuation import POLICY as continuation_policy
        binding = value['continuation']
        require(isinstance(binding, dict) and set(binding) == {'schema_version', 'policy', 'parent', 'receipt', 'sha256'}
                and type(binding['schema_version']) is int and binding['schema_version'] == 1
                and binding['policy'] == continuation_policy and binding['receipt'] == 'continuation/receipt.json'
                and isinstance(binding['parent'], str) and len(Path(binding['parent']).parts) == 2
                and Path(binding['parent']).parts[0] == 'outputs', 'Invalid standing continuation provenance')
        parent_path = _relative(root, binding['parent'])
        require(parent_path != output, 'Continuation cannot reference itself')
        continuation, receipt_hash = _read(output/binding['receipt'], 1024**2)
        require(receipt_hash == _hash(binding['sha256']) and isinstance(continuation, dict)
                and continuation.get('schema_version') == 1 and continuation.get('policy') == continuation_policy
                and continuation.get('parent') == binding['parent'], 'Standing continuation receipt changed')
        _hash(continuation.get('parent_request_sha256'))
    context = {k:v for k,v in value.items() if k not in
               ('id', 'created', 'agent_backend', 'max_agent_calls', 'max_repairs', 'agent_model', 'agent_timeout', 'continuation')}
    context['reuse_model'] = {k:source[k] for k in ('bytes', 'sha256')}
    return value, request_hash, digest(context), continuation


def _open_readonly(root):
    path = root/'.local/openai/ledger.sqlite3'
    _private_file(path, 64*1024**2)
    db = sqlite3.connect(path.as_uri()+'?mode=ro', uri=True, timeout=.2)
    db.row_factory = sqlite3.Row
    db.execute('PRAGMA query_only=ON')
    return db


def _grant_record(policy, policy_hash, approval, parent, child, context):
    return {'decision': 'approved', 'user_reply': approval['user_reply'], 'operation': 'retire_and_continue',
            'parent_job_id': parent, 'successor_job_id': child, 'additional_calls': policy['max_calls'],
            'standing_policy_sha256': policy_hash, 'approval_receipt_sha256': policy['approval_receipt']['sha256'],
            'lineage_root_job_id': policy['lineage_root_job_id'], 'context_sha256': context}


def _binding(root, parent, child, policy, policy_hash, approval, db):
    budget = budget_snapshot(db)
    require(budget and budget['effective_ceiling_microdollars'] <= policy['max_total_microdollars'],
            'Effective project ceiling exceeds standing authorization')
    accounting = accounting_snapshot(db)
    require(accounting['project_exposure_microdollars'] <= budget['effective_ceiling_microdollars'],
            'Known costs and unknown charges exceed the standing ceiling')
    anchor = root/'outputs'/policy['lineage_root_job_id']
    _, anchor_hash, context, _ = _request(root, anchor, policy, verify_source=False)
    require(anchor_hash == policy['lineage_root_request_sha256'], 'Standing lineage root request changed')
    _, parent_hash, parent_context, _ = _request(root, parent, policy, verify_source=True)
    require(parent_context == context, 'Retry intent differs from the approved lineage')
    chain, current = [], policy['lineage_root_job_id']
    while True:
        require(current not in chain and len(chain) <= policy['max_successors'], 'Standing successor limit or lineage cycle')
        chain.append(current)
        grant = db.execute('SELECT * FROM jobs WHERE id=?', (current,)).fetchone()
        require(grant is not None, 'Standing lineage lacks a recorded grant')
        if current == policy['lineage_root_job_id']:
            require(grant['digest'] == anchor_hash, 'Standing root grant differs from its request')
        else:
            require(isinstance(grant['approval'], str) and len(grant['approval'].encode()) <= 8192,
                    'Invalid standing lineage grant approval')
            prior = _decode(grant['approval'].encode())
            expected = _grant_record(policy, policy_hash, approval, chain[-2], current, context)
            retired = db.execute('SELECT approval FROM retirements WHERE parent=?', (chain[-2],)).fetchone()
            require(prior == expected and retired is not None and retired[0] == grant['approval'],
                    'Standing lineage grant belongs to another policy')
        next_row = db.execute('SELECT successor FROM retirements WHERE parent=?', (current,)).fetchone()
        if next_row is None: break
        current = next_row[0]
    require(parent.name in chain, 'Parent is outside the approved retry lineage')
    parent_grant = db.execute('SELECT * FROM jobs WHERE id=?', (parent.name,)).fetchone()
    require(parent_grant['digest'] == parent_hash and parent_grant['calls'] == policy['max_calls'],
            'Parent grant differs from its exact request or call limit')
    parent_cost = accounting['jobs'].get(parent.name)
    require(parent_cost and 0 < parent_cost['calls'] <= parent_grant['calls']
            and parent_cost['uncertain_reserved_microdollars'] == 0,
            'Parent calls are absent or retain unknown charges')
    child_hash = None
    if child is not None:
        child_request, child_hash, child_context, continuation = _request(root, child, policy, verify_source=True)
        require(child.name != parent.name and child_context == context, 'Child changes the approved retry intent')
        require(continuation is not None and continuation['parent'] == str(parent.relative_to(root))
                and continuation['parent_request_sha256'] == parent_hash,
                'Child continuation differs from the exact grant parent')
        next_hold = reserve_microdollars(child_request['agent_backend']['max_output_tokens'], model=child_request['agent_backend']['model'])
    else:
        next_hold = reserve_microdollars(model=policy['model'])
    existing = child is not None and len(chain) >= 2 and chain[-2:] == [parent.name, child.name]
    if existing:
        child_grant = db.execute('SELECT * FROM jobs WHERE id=?', (child.name,)).fetchone()
        require(child_grant['digest'] == child_hash and child_grant['calls'] == policy['max_calls'],
                'Existing successor grant differs from its pinned request')
        ceiling = child_grant['ceiling']
        require(type(ceiling) is int and 0 < ceiling <= policy['max_total_microdollars'],
                'Existing successor grant exceeds standing authorization')
    else:
        require(chain[-1] == parent.name, 'Parent already retired into another successor')
        require(len(chain)-1 < policy['max_successors'], 'Standing successor limit reached')
        if child is not None:
            require(not db.execute('SELECT 1 FROM jobs WHERE id=?', (child.name,)).fetchone(),
                    'Child has an unrelated existing grant')
        from avatar_assets.openai_grant_closure import allocated_grants,require_open_grant
        require_open_grant(db,parent.name)
        allocated = allocated_grants(db)
        ceiling = budget['effective_ceiling_microdollars']-allocated+parent_grant['ceiling']-parent_cost['settled_actual_microdollars']
        require(type(ceiling) is int and next_hold <= ceiling <= policy['max_total_microdollars'],
                'Standing allowance cannot fully reserve another bounded call')
    return {'policy_sha256': policy_hash, 'parent_job_id': parent.name, 'parent_request_sha256': parent_hash,
            'child_job_id': child.name if child else None, 'child_request_sha256': child_hash,
            'context_sha256': context, 'lineage_root_job_id': policy['lineage_root_job_id'],
            'successors_used': len(chain)-1, 'max_successors': policy['max_successors'],
            'grant_microdollars': ceiling, 'max_calls': policy['max_calls'],
            'effective_project_ceiling_microdollars': budget['effective_ceiling_microdollars'],
            'project_exposure_microdollars': accounting['project_exposure_microdollars'],
            'existing_grant': existing}


def retry_eligibility(root, parent_output):
    """Action-time read only; the Studio manager separately proves terminal process custody."""
    root = Path(root).resolve()
    try:
        policy, policy_hash, approval = _policy(root)
        with closing(_open_readonly(root)) as db:
            result = _binding(root, Path(parent_output), None, policy, policy_hash, approval, db)
        return {**result, 'eligible': True, 'reason': 'Exact saved-source retry fits the standing allowance.'}
    except (OSError, ValueError, TypeError, KeyError, sqlite3.Error):
        return {'eligible': False, 'reason': 'Standing retry policy, source, lineage or remaining allowance requires inspection.'}


def authorize_successor(root, parent_output, child_output, *, expected_policy_sha256):
    """Grant only an immutable child; retirement and ceilings remain the ledger's transaction."""
    root = Path(root).resolve(); parent = Path(parent_output); child = Path(child_output)
    policy, policy_hash, approval = _policy(root)
    require(policy_hash == _hash(expected_policy_sha256), 'Standing policy changed after Retry')
    with closing(_open_readonly(root)) as db:
        result = _binding(root, parent, child, policy, policy_hash, approval, db)
    if result['existing_grant']:
        return {**result, 'status': 'already_authorized'}
    record = _grant_record(policy, policy_hash, approval, parent.name, child.name, result['context_sha256'])
    require(_policy(root)[1] == policy_hash, 'Standing policy changed before grant')
    ledger = OpenAILedger(root/'.local/openai')
    try:
        ledger.continue_job(parent.name, result['parent_request_sha256'], child.name,
                            result['child_request_sha256'], policy['max_calls'], result['grant_microdollars'], record)
    finally:
        ledger.close()
    require(_policy(root)[1] == policy_hash, 'Standing policy changed during grant; stop before any call')
    return {**result, 'status': 'authorized', 'successors_used': result['successors_used']+1}
