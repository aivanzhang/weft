"""Start a new stopped-portal child from pinned body evidence, without inference."""

import fcntl
from hashlib import sha256
import math
import os
from pathlib import Path
import re
import time
from uuid import uuid4

from weft.exercise_body_source import _json, validate_body_source_files
from weft.io import canonical_hash, local_time, write_json
from weft.openai_access import PROJECT_ROOT
from weft.portal_import import _directory, _storage
from weft.shared_world_packaging import read_package_file, read_package_manifest, write_package_files

PROFILE = 'portal-sealed-body-reuse-v1'
BODY = 'body-reuse/body-source'
RECEIPT = 'body-reuse/receipt.json'
MAX_BODY_BYTES = 128 << 20
PARENT_METADATA = ('job.json', 'status.json', 'perception/status.json', 'perception/failure.json',
                   'failure.json', 'perception/preflight/failure.json', 'perception/preflight/body_source/failure.json')
FIELDS = {'schema_version', 'profile', 'child_id', 'parent_id', 'parent_files', 'source_file', 'source_sha256',
          'body_manifest_sha256', 'reuse_identity', 'api_calls', 'inference_calls'}


def _check(deadline):
    if time.monotonic() >= deadline:
        raise TimeoutError('Portal body reuse exceeded its deadline')


def _read(root, name, maximum=64 << 10, expected=None):
    return read_package_file(root, name, max_bytes=maximum, expected_sha256=expected)


def _identity(receipt):
    return canonical_hash({key: receipt[key] for key in ('parent_id', 'parent_files', 'source_file',
                                                       'source_sha256', 'body_manifest_sha256')})


def _body(source, pin, source_sha, deadline):
    _check(deadline)
    raw, names = read_package_manifest(source)
    marker = _json(raw)
    if (sha256(raw).hexdigest() != pin or len(raw) > 1 << 20 or not 1 <= len(names) <= 511
            or marker.get('profile') != 'exercise-body-source-v1' or marker.get('status') != 'construction_only'
            or (source/'failure.json').exists() or (source/'failure.json').is_symlink()):
        raise ValueError('Reuse requires the exact complete construction-only body manifest without failure')
    files, total = {}, len(raw)
    for name, expected in names.items():
        _check(deadline)
        if total >= MAX_BODY_BYTES:
            raise ValueError('Reused body exceeds its 128 MiB bound')
        value = _read(source, name, min(64 << 20, MAX_BODY_BYTES-total), expected)
        files[name] = value
        total += len(value)
    human, observations = validate_body_source_files(files, deadline=deadline)
    if human.get('source_sha256') != source_sha or observations['source']['sha256'] != source_sha:
        raise ValueError('Reused body must match the exact parent video')
    _read(source, 'manifest.json', 1 << 20, pin)
    if (source/'failure.json').exists() or (source/'failure.json').is_symlink():
        raise ValueError('Reused body gained a failure marker')
    _check(deadline)
    return raw, files, total


def _metadata(root):
    job = _json(_read(root, 'job.json'))
    binding = job.get('body_reuse')
    if (not isinstance(binding, dict) or set(binding) != {'profile', 'receipt_sha256', 'body_manifest_sha256'}
            or binding.get('profile') != PROFILE
            or any(not isinstance(binding.get(key), str) or not re.fullmatch('[a-f0-9]{64}', binding[key])
                   for key in ('receipt_sha256', 'body_manifest_sha256'))):
        raise ValueError('Missing closed source-bound body reuse request')
    receipt = _json(_read(root, RECEIPT, expected=binding['receipt_sha256']))
    _read(root, BODY+'/manifest.json', 1 << 20, binding['body_manifest_sha256'])
    if (set(receipt) != FIELDS or receipt.get('profile') != PROFILE or type(receipt.get('schema_version')) is not int
            or receipt['schema_version'] != 1 or receipt.get('child_id') != job.get('id')
            or not isinstance(receipt.get('parent_id'), str) or not re.fullmatch('[a-f0-9]{32}', receipt['parent_id'])
            or job.get('parent_id') != receipt['parent_id'] or job.get('kind') != 'video' or job.get('mode') != 'exercise'
            or receipt.get('source_file') != job.get('source_file') or receipt.get('source_file') != 'source.mp4'
            or receipt.get('source_sha256') != job.get('source_sha256')
            or not isinstance(receipt.get('source_sha256'), str) or not re.fullmatch('[a-f0-9]{64}', receipt['source_sha256'])
            or receipt.get('body_manifest_sha256') != binding['body_manifest_sha256']
            or any(type(receipt.get(k)) is not int or receipt[k] != 0 for k in ('api_calls', 'inference_calls'))
            or not isinstance(receipt.get('parent_files'), dict)
            or not {'job.json', 'status.json'} <= receipt['parent_files'].keys()
            or set(receipt['parent_files']) - set(PARENT_METADATA)
            or any(not isinstance(pin, str) or not re.fullmatch('[a-f0-9]{64}', pin) for pin in receipt['parent_files'].values())
            or receipt.get('reuse_identity') != _identity(receipt) or job.get('reuse_identity') != receipt['reuse_identity']):
        raise ValueError('Body reuse child, original source or parent receipt differs')
    for name, pin in receipt['parent_files'].items():
        _read(root, 'body-reuse/parent/' + name, expected=pin)
    parent_job = _json(_read(root, 'body-reuse/parent/job.json'))
    if (parent_job.get('id') != receipt['parent_id'] or parent_job.get('source_sha256') != receipt['source_sha256']
            or parent_job.get('kind', 'video') != 'video'):
        raise ValueError('Original parent identity differs from reused source')
    request = _json(_read(root, 'perception/preflight-request.json'))
    from weft.portal_perception_jobs import PROFILE as PREFLIGHT_PROFILE
    if (set(request) != {'profile', 'source_sha256', 'requested_at', 'api_calls_authorized', 'body_reuse'}
            or request.get('profile') != PREFLIGHT_PROFILE or request.get('source_sha256') != receipt['source_sha256']
            or request.get('body_reuse') != binding or type(request.get('api_calls_authorized')) is not int
            or request['api_calls_authorized'] != 0 or not isinstance(request.get('requested_at'), str)):
        raise ValueError('Queued preflight differs from the immutable body reuse receipt')
    return job, receipt


def validate_pending_reuse(root):
    """Only never-started explicit local work survives startup; this is not source admission."""
    try:
        root = _directory(root)
        job, _ = _metadata(root)
        if job['id'] != root.name:
            return False
        forbidden = ('preflight-start.json', 'request.json', 'preflight', 'failure.json', 'result.json',
                     'quote.json', 'preflight-complete.json', 'cancel-request.json')
        if any((root/'perception'/name).exists() or (root/'perception'/name).is_symlink() for name in forbidden):
            return False
        if any((root/name).exists() or (root/name).is_symlink() for name in ('cancel-request.json', 'equipment', 'muse', BODY+'/failure.json')):
            return False
        status = _json(_read(root, 'status.json'))
        return (_json(_read(root, 'perception/status.json')).get('status') == 'queued_preflight'
                and status.get('status') == 'needs_input' and status.get('route') == 'exercise' and not status.get('lab')
                and _read(root, 'status.json') == _read(root, 'perception/parent-status.json'))
    except (ValueError, OSError, KeyError, TypeError, AttributeError):
        return False


def validate_queued_body(root, *, deadline):
    root = _directory(root)
    _, receipt = _metadata(root)
    if receipt['child_id'] != root.name:
        raise ValueError('Body reuse request belongs to another child')
    _read(root, receipt['source_file'], 37_000_000, receipt['source_sha256'])
    body = root/BODY
    _body(body, receipt['body_manifest_sha256'], receipt['source_sha256'], deadline)
    _check(deadline)
    return body


def reuse_body(root, parent_project, body_source, body_manifest_sha256, *, project=PROJECT_ROOT, max_seconds=30):
    """Publish a new queued child atomically under the stopped-portal lease; never retry the parent."""
    from weft.portal import MAX_JOB_BYTES, MAX_JOBS, MAX_STORAGE, TERMINAL, private_directory
    from weft.portal_inventory import validate_intent
    from weft.portal_perception_jobs import ACTIVE, queue_preflight, read_status
    if (not isinstance(parent_project, str) or not re.fullmatch('[a-f0-9]{32}', parent_project)
            or not isinstance(body_manifest_sha256, str) or not re.fullmatch('[a-f0-9]{64}', body_manifest_sha256)
            or type(max_seconds) not in (int, float) or not math.isfinite(max_seconds) or not 0 < max_seconds <= 60):
        raise ValueError('Reuse needs an exact parent ID, body manifest SHA256 and bounded deadline')
    deadline = time.monotonic()+max_seconds
    root, source = private_directory(root), _directory(body_source)
    parent = _directory(root/parent_project)
    lease = os.open(root/'portal.lock', os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    stage = None
    try:
        try:
            fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            raise ValueError('Stop the portal before reusing body evidence; active work must not be interrupted') from None
        snapshots = {name: _read(parent, name) for name in PARENT_METADATA
                     if (parent/name).exists() or (parent/name).is_symlink()}
        if not {'job.json', 'status.json'} <= snapshots.keys():
            raise ValueError('Parent project metadata is incomplete')
        job, status = (_json(snapshots[name]) for name in ('job.json', 'status.json'))
        if (job.get('id') != parent_project or job.get('kind', 'video') != 'video' or job.get('mode') != 'exercise'
                or status.get('status') not in TERMINAL or status.get('lab') or status.get('equipment_candidate_id')
                or read_status(parent).get('status') in ACTIVE or (parent/'equipment').exists()
                or job.get('source_file') != 'source.mp4'):
            raise ValueError('Reuse requires an unresolved inactive exercise MP4 parent, not an active or selected world')
        source_sha = job.get('source_sha256')
        if not isinstance(source_sha, str) or not re.fullmatch('[a-f0-9]{64}', source_sha):
            raise ValueError('Parent source identity is missing')
        video = _read(parent, 'source.mp4', 37_000_000, source_sha)
        _read(source, 'manifest.json', 1 << 20, body_manifest_sha256)
        receipt = {'schema_version': 1, 'profile': PROFILE, 'child_id': uuid4().hex, 'parent_id': parent_project,
                   'parent_files': {n: sha256(v).hexdigest() for n, v in snapshots.items()}, 'source_file': 'source.mp4',
                   'source_sha256': source_sha, 'body_manifest_sha256': body_manifest_sha256,
                   'api_calls': 0, 'inference_calls': 0}
        receipt['reuse_identity'] = _identity(receipt)
        count = pending = 0
        for folder in root.iterdir():
            _check(deadline)
            if not re.fullmatch('[a-f0-9]{32}', folder.name):
                continue
            _directory(folder); count += 1
            existing = _json(_read(folder, 'job.json'))
            if existing.get('reuse_identity') == receipt['reuse_identity']:
                _, prior = _metadata(folder)
                if prior['child_id'] != folder.name:
                    raise ValueError('Existing body reuse identity belongs to another child')
                return {'id': folder.name, 'status': 'already_created', 'directory': str(folder), 'revalidated': False,
                        'api_calls': 0, 'inference_calls': 0}
            equipment = folder/'equipment/status.json'
            pending += (_json(_read(folder, 'status.json')).get('status') not in TERMINAL
                        or read_status(folder).get('status') in ACTIVE
                        or equipment.exists() and _json(_read(folder, 'equipment/status.json')).get('status') in {'queued', 'running', 'cancelling'})
        manifest, body_files, body_bytes = _body(source, body_manifest_sha256, source_sha, deadline)
        if (count >= MAX_JOBS or pending >= 4
                or _storage(root, lambda: _check(deadline)) + (pending+1)*MAX_JOB_BYTES > MAX_STORAGE
                or body_bytes*3 + len(video) + sum(map(len, snapshots.values())) + (8 << 20) > MAX_JOB_BYTES):
            raise ValueError('Portal capacity cannot reserve the complete copied body, preflight and later work')
        identity = receipt['child_id']; stage = root/('.reuse-'+identity); stage.mkdir(mode=0o700)
        (stage/BODY).mkdir(mode=0o700, parents=True)
        (stage/'body-reuse/parent').mkdir(mode=0o700)
        write_package_files(stage/BODY, {**body_files, 'manifest.json': manifest})
        write_package_files(stage/'body-reuse/parent', snapshots)
        write_package_files(stage, {'source.mp4': video})
        write_json(stage/RECEIPT, receipt)
        binding = {'profile': PROFILE, 'receipt_sha256': sha256(_read(stage, RECEIPT)).hexdigest(),
                   'body_manifest_sha256': body_manifest_sha256}
        child = {'id': identity, 'name': job.get('name', 'Reused exercise.mp4'), 'kind': 'video', 'mode': 'exercise',
                 'ai': False, 'created_at': local_time(), 'source_file': 'source.mp4', 'source_sha256': source_sha,
                 'parent_id': parent_project, 'reuse_identity': receipt['reuse_identity'], 'body_reuse': binding,
                 'exercise_intent': validate_intent(job.get('exercise_intent'))}
        write_json(stage/'job.json', child)
        write_json(stage/'status.json', {'status': 'needs_input', 'route': 'exercise', 'stage': 'body_reuse', 'events': [],
                   'message': 'New child with exact saved body; local checks and separate provider consent remain required.'})
        queue_preflight(stage)
        for name, pin in receipt['parent_files'].items():
            _read(parent, name, expected=pin)
        _read(parent, 'source.mp4', 37_000_000, source_sha)
        _read(source, 'manifest.json', 1 << 20, body_manifest_sha256)
        if (source/'failure.json').exists() or (source/'failure.json').is_symlink():
            raise ValueError('Original body gained a failure marker before publication')
        for name, value in body_files.items():
            _check(deadline); _read(source, name, len(value), sha256(value).hexdigest())
        _check(deadline)
        os.rename(stage, root/identity)
        return {'id': identity, 'status': 'queued_preflight', 'directory': str(root/identity),
                'parent_id': parent_project, 'body_manifest_sha256': body_manifest_sha256, 'api_calls': 0, 'inference_calls': 0}
    except Exception as error:
        if stage is not None and stage.exists():
            write_json(stage/'reuse-failure.json', {'error': str(error)[:1000], 'published': False,
                                                  'partial_evidence_preserved': True})
        raise
    finally:
        os.close(lease)
