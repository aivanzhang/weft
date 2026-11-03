"""Resume a sealed perception export through equipment assembly, never provider work."""

import fcntl
from hashlib import sha256
import math
import os
from pathlib import Path
import re
import time
from uuid import uuid4

from weft.exercise_body_source import _json
from weft.io import local_time, write_json
from weft.openai_access import PROJECT_ROOT
from weft.portal_import import _directory, _storage
from weft.shared_world_packaging import read_package_file, write_package_files

PROFILE = 'portal-saved-perception-recovery-v1'
RECEIPT = 'equipment/recovery.json'
SOURCE = 'perception/prepared-source.json'
BRIDGE = 'perception/bridge/manifest.json'
INPUTS = ('job.json', 'status.json', 'perception/status.json', 'perception/failure.json')
SCOPE = 'Saved source publication and ordinary equipment assembly only; original failed perception is unchanged. No provider or pose-inference retry.'
FIELDS = {'schema_version', 'profile', 'project_id', 'source_sha256', 'bridge_manifest_sha256',
          'descriptor_sha256', 'inputs', 'request_sha256', 'api_calls', 'inference_calls', 'scope'}


def _bytes(value):
    import json
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False)+'\n').encode()


def _read(root, name, maximum=64 << 10, expected=None):
    return read_package_file(root, name, max_bytes=maximum, expected_sha256=expected)


def _check(deadline):
    if time.monotonic() >= deadline:
        raise TimeoutError('Saved perception recovery exceeded its deadline')


def _original(root):
    snapshots = {name: _read(root, name) for name in INPUTS}
    job, status, failed, failure = (_json(snapshots[name]) for name in INPUTS)
    if (job.get('id') != root.name or job.get('kind', 'video') != 'video'
            or not isinstance(job.get('source_sha256'), str) or not re.fullmatch('[a-f0-9]{64}', job['source_sha256'])
            or job.get('source_file') != 'source.mp4' or status.get('status') != 'needs_input'
            or status.get('route') != 'exercise' or status.get('lab') or status.get('equipment_candidate_id')
            or failed.get('status') != 'failed' or failure != failed):
        raise ValueError('Recovery requires an unresolved exact video with its original terminal perception failure')
    return job, snapshots


def _receipt(root):
    from weft.portal_equipment_candidates import PROFILE as EQUIPMENT_PROFILE
    raw = _read(root, RECEIPT)
    value = _json(raw)
    if (set(value) != FIELDS or value.get('profile') != PROFILE or value.get('scope') != SCOPE
            or type(value.get('schema_version')) is not int or value['schema_version'] != 1
            or value.get('project_id') != root.name
            or any(type(value.get(key)) is not int or value[key] != 0 for key in ('api_calls', 'inference_calls'))
            or any(not isinstance(value.get(key), str) or not re.fullmatch('[a-f0-9]{64}', value[key])
                   for key in ('source_sha256', 'bridge_manifest_sha256', 'descriptor_sha256', 'request_sha256'))
            or not isinstance(value.get('inputs'), dict) or set(value['inputs']) != set(INPUTS)
            or any(not isinstance(pin, str) or not re.fullmatch('[a-f0-9]{64}', pin) for pin in value['inputs'].values())):
        raise ValueError('Saved perception recovery receipt differs from its closed contract')
    job, snapshots = _original(root)
    if value['source_sha256'] != job['source_sha256']:
        raise ValueError('Recovery source differs from original upload')
    for name, pin in value['inputs'].items():
        if sha256(snapshots[name]).hexdigest() != pin:
            raise ValueError('Original perception failure or project changed after recovery request')
        _read(root, 'equipment/recovery-inputs/'+name, expected=pin)
    _read(root, BRIDGE, 1 << 20, value['bridge_manifest_sha256'])
    _read(root, SOURCE, 16 << 10, value['descriptor_sha256'])
    request = _json(_read(root, 'equipment/request.json', 16 << 10, value['request_sha256']))
    expected = {'profile': EQUIPMENT_PROFILE, 'source_kind': 'perception', 'source_sha256': value['source_sha256'],
                'descriptor_sha256': value['descriptor_sha256']}
    if request != expected or _read(root, 'equipment/parent-status.json') != snapshots['status.json']:
        raise ValueError('Recovery must queue only the ordinary pinned perception-source equipment request')
    return value, sha256(raw).hexdigest()


def validate_pending_recovery(root):
    """A metadata check can resume never-started local assembly, never a failed provider job."""
    try:
        root = _directory(root)
        _receipt(root)
        forbidden = ('recovery-start.json', 'run', 'source-validation.json', 'candidates.json',
                     'cancel-request.json', 'selection.json', 'failure.json')
        if any((root/'equipment'/name).exists() or (root/'equipment'/name).is_symlink() for name in forbidden):
            return False
        if (root/'cancel-request.json').exists() or (root/'cancel-request.json').is_symlink():
            return False
        return _json(_read(root, 'equipment/status.json')).get('status') == 'queued'
    except (ValueError, OSError, KeyError, TypeError, AttributeError):
        return False


def start_recovery(root):
    """Record work before validating it, so crashes cannot turn into automatic retries."""
    root = _directory(root)
    write_package_files(root/'equipment', {'recovery-start.json': _bytes({
        'profile': PROFILE, 'started_at': local_time(), 'api_calls': 0, 'inference_calls': 0})})
    return _receipt(root)[0]


def recover_saved_perception(root, project_id, *, bridge_manifest_sha256, project=PROJECT_ROOT, max_seconds=40):
    """Publish exact saved evidence and atomically queue assembly under a stopped-portal lease."""
    from weft.portal import MAX_JOB_BYTES, MAX_STORAGE, TERMINAL, private_directory
    from weft.portal_perception_jobs import ACTIVE, read_status
    from weft.portal_equipment_candidates import PROFILE as EQUIPMENT_PROFILE, publish_perception_source
    if (not isinstance(project_id, str) or not re.fullmatch('[a-f0-9]{32}', project_id)
            or not isinstance(bridge_manifest_sha256, str) or not re.fullmatch('[a-f0-9]{64}', bridge_manifest_sha256)
            or type(max_seconds) not in (int, float) or not math.isfinite(max_seconds) or not 30 < max_seconds <= 60):
        raise ValueError('Recovery needs an exact project ID, bridge manifest SHA256 and 30–60 second deadline')
    deadline = time.monotonic()+max_seconds
    root = private_directory(root); folder = _directory(root/project_id)
    lease = os.open(root/'portal.lock', os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    stage = None
    try:
        try:
            fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            raise ValueError('Stop the portal before recovering saved perception; active work must not be interrupted') from None
        job, snapshots = _original(folder)
        _read(folder, BRIDGE, 1 << 20, bridge_manifest_sha256)
        if (folder/'equipment').exists() or (folder/'equipment').is_symlink():
            receipt, _ = _receipt(folder)
            if receipt['bridge_manifest_sha256'] != bridge_manifest_sha256:
                raise ValueError('Existing recovery binds a different saved perception bridge')
            return {'id': project_id, 'status': 'already_requested', 'directory': str(folder),
                    'revalidated': False, 'api_calls': 0, 'inference_calls': 0}
        pending = 0
        for child in root.iterdir():
            _check(deadline)
            if not re.fullmatch('[a-f0-9]{32}', child.name):
                continue
            _directory(child)
            equipment = child/'equipment/status.json'
            pending += (_json(_read(child, 'status.json')).get('status') not in TERMINAL
                        or read_status(child).get('status') in ACTIVE
                        or equipment.exists() and _json(_read(child, 'equipment/status.json')).get('status') in {'queued', 'running', 'cancelling'})
        if pending >= 4 or _storage(root, lambda: _check(deadline))+(pending+1)*MAX_JOB_BYTES > MAX_STORAGE:
            raise ValueError('Portal capacity cannot reserve saved-evidence candidate construction')
        if _storage(folder, lambda: _check(deadline))+(1 << 20) > MAX_JOB_BYTES:
            raise ValueError('Project capacity cannot reserve recovery receipts within its existing worker byte cap')
        _check(deadline)
        descriptor = publish_perception_source(folder, folder/'perception/bridge', project=project, deadline=deadline)
        if descriptor['bridge']['manifest_sha256'] != bridge_manifest_sha256:
            raise ValueError('Completed bridge changed during ordinary source publication')
        _check(deadline)
        descriptor_pin = sha256(_read(folder, SOURCE, 16 << 10)).hexdigest()
        request = _bytes({'profile': EQUIPMENT_PROFILE, 'source_kind': 'perception',
            'source_sha256': job['source_sha256'], 'descriptor_sha256': descriptor_pin})
        receipt = {'schema_version': 1, 'profile': PROFILE, 'project_id': project_id,
            'source_sha256': job['source_sha256'], 'bridge_manifest_sha256': bridge_manifest_sha256,
            'descriptor_sha256': descriptor_pin, 'inputs': {n: sha256(v).hexdigest() for n, v in snapshots.items()},
            'request_sha256': sha256(request).hexdigest(), 'api_calls': 0, 'inference_calls': 0, 'scope': SCOPE}
        stage = folder/('.equipment-recovery-'+uuid4().hex); stage.mkdir(mode=0o700)
        write_package_files(stage, {'request.json': request, 'parent-status.json': snapshots['status.json'],
            'recovery.json': _bytes(receipt), **{'recovery-inputs/'+name: value for name, value in snapshots.items()}})
        write_json(stage/'status.json', {'status': 'queued', 'message': 'Saved perception evidence recovered; waiting for ordinary candidate assembly. Original failure retained; no new API call.'})
        for name, pin in receipt['inputs'].items():
            _read(folder, name, expected=pin)
        _read(folder, BRIDGE, 1 << 20, bridge_manifest_sha256)
        _read(folder, SOURCE, 16 << 10, descriptor_pin)
        _check(deadline)
        os.rename(stage, folder/'equipment')
        return {'id': project_id, 'status': 'queued', 'directory': str(folder), 'source_descriptor_sha256': descriptor_pin,
                'api_calls': 0, 'inference_calls': 0, 'scope': SCOPE}
    except Exception as error:
        if stage is not None and stage.exists():
            write_json(stage/'recovery-failure.json', {'error': str(error)[:1000], 'published': False,
                                                     'partial_evidence_preserved': True})
        raise
    finally:
        os.close(lease)
