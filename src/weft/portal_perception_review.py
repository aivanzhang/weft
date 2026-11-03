"""Selected perception worlds reuse exact body evidence; review never qualifies repairs."""

from hashlib import sha256
from pathlib import Path
import json
import time

from weft.exercise_body_source import _json
from weft.openai_access import PROJECT_ROOT
from weft.shared_world_packaging import read_package_file, write_package_files

PROFILE = 'portal-perception-review-attachment-v1'
RECEIPT = 'muse/perception-attachment.json'
UNAVAILABLE = 'Muse needs a selected perception candidate and its sealed generated body evidence.'
CONDITIONAL_QUESTION = ('This explicitly selected world is one of retained conditional role assignments. '
    'Human-to-track correspondence and rigid-object identities remain unverified; alternatives were not resolved by simulator admission. '
    'Distinguish these assumptions from observed contact and physical task success.')


def _read(root, name, limit, pin=None):
    raw = read_package_file(root, name, max_bytes=limit, expected_sha256=pin)
    return _json(raw), sha256(raw).hexdigest()


def _selected(root):
    from weft.portal_equipment_candidates import PROFILE as EQUIPMENT_PROFILE, JOB_SOURCE_PATH, read_report, read_status

    root = Path(root).absolute()
    job, job_pin = _read(root, 'job.json', 64 << 10)
    status, _ = _read(root, 'status.json', 32 << 10)
    selection, selection_pin = _read(root, 'equipment/selection.json', 8192)
    equipment = read_status(root)
    report = read_report(root, equipment)
    request, request_pin = _read(root, 'equipment/request.json', 16 << 10)
    if (job.get('kind', 'video') != 'video' or status.get('status') != 'ready'
            or status.get('route') != 'exercise' or not status.get('equipment_candidate_id')
            or selection.get('profile') != EQUIPMENT_PROFILE or report is None
            or selection.get('candidate_id') != status['equipment_candidate_id']
            or selection.get('lab') != status.get('lab')
            or selection.get('report_sha256') != equipment.get('report_sha256')
            or request.get('source_kind') != 'perception' or request.get('profile') != EQUIPMENT_PROFILE
            or request.get('source_sha256') != job.get('source_sha256')
            or report.get('request_sha256') != request_pin or report.get('source_sha256') != job.get('source_sha256')):
        raise ValueError(UNAVAILABLE)
    rows = [c for c in report['candidates'] if c.get('id') == selection['candidate_id']]
    if len(rows) != 1 or rows[0].get('selectable') is not True or rows[0].get('status') != 'ready':
        raise ValueError('Muse cannot review an unselected or rejected equipment candidate')
    candidate = rows[0]
    if (candidate.get('lab') != selection['lab'] or candidate.get('manifest_sha256') != selection.get('manifest_sha256')
            or candidate.get('source_sha256') != job['source_sha256']):
        raise ValueError('Selected candidate differs from its admitted report')
    # Fixed producer paths, not request-provided body locations.
    descriptor, descriptor_pin = _read(root, JOB_SOURCE_PATH, 16 << 10, request.get('descriptor_sha256'))
    if (descriptor.get('job_sha256') != job_pin or descriptor.get('source_sha256') != job['source_sha256']
            or descriptor.get('base', {}).get('path') != 'perception/preflight/body_seed'
            or report.get('source_validation', {}).get('descriptor_sha256') != descriptor_pin):
        raise ValueError('Selected candidate lacks its exact perception source descriptor')
    external, external_pin = _read(root, 'perception/bridge/preflight-descriptor.json', 8192,
                                   descriptor['provenance']['preflight_descriptor_sha256'])
    preflight, preflight_pin = _read(root, 'perception/preflight/manifest.json', 1 << 20,
                                    external.get('preflight_manifest_sha256'))
    body = root / 'perception/preflight/body_source'
    body_pin = preflight.get('children', {}).get('body_source')
    if not isinstance(body_pin, str) or len(body_pin) != 64:
        raise ValueError('Perception has no sealed generated body for Muse review')
    read_package_file(body, 'manifest.json', max_bytes=1 << 20, expected_sha256=body_pin)
    lab = candidate['lab']
    read_package_file(root, lab + '/lab-config.json', max_bytes=64 << 10, expected_sha256=candidate['lab_config_sha256'])
    read_package_file(root, lab + '/run/manifest.json', max_bytes=1 << 20, expected_sha256=candidate['manifest_sha256'])
    return body, {'source_sha256': job['source_sha256'], 'descriptor_sha256': descriptor_pin,
        'selection_sha256': selection_pin, 'report_sha256': equipment['report_sha256'],
        'preflight_descriptor_sha256': external_pin, 'preflight_manifest_sha256': preflight_pin,
        'body_manifest_sha256': body_pin, 'candidate': candidate}


def attachment_summary(root):
    """Only bounded metadata here; the background review worker admits all source bytes."""
    try:
        _, pins = _selected(root)
        return {'available': True, 'status': 'pending_action_time_validation', 'repair_qualified': False,
            'candidate_id': pins['candidate']['id'],
            'reason': 'Muse revalidates the selected world and original body before any request. Conditional identities remain unresolved; local checks and repairs are unavailable.'}
    except (ValueError, OSError, KeyError, TypeError):
        return {'available': False, 'reason': UNAVAILABLE, 'repair_qualified': False}


def validate_attachment(root, *, project=PROJECT_ROOT):
    """Pure full source/body/world admission, followed by one immutable review receipt."""
    from weft.portal_equipment_candidates import validate_job_source
    from weft.portal_evidence_attachment import _body_files, _validate_body, _join, _receipt, _check

    root = Path(root).absolute()
    # Two admissions plus the existing 45-second request must fit the 90-second worker.
    deadline = time.monotonic() + 15
    body, pins = _selected(root)
    _, proof = validate_job_source(root, project=project, deadline=deadline)
    if proof['descriptor_sha256'] != pins['descriptor_sha256']:
        raise ValueError('Perception source changed before Muse review')
    _check(deadline)
    manifest, files = _body_files(body, deadline)
    if sha256(manifest).hexdigest() != pins['body_manifest_sha256']:
        raise ValueError('Perception body changed before Muse review')
    human, observations, validation = _validate_body(body, files, project=project, deadline=deadline)
    world, document = _join(root, files, human, observations, deadline)
    candidate = pins['candidate']
    if (world['run'] != candidate['lab'] + '/run' or world['manifest_sha256'] != candidate['manifest_sha256']
            or world['lab_config_sha256'] != candidate['lab_config_sha256']):
        raise ValueError('Muse body attachment differs from the explicitly selected world')
    receipt = {**_receipt(manifest, human, world, document, validation), 'profile': PROFILE,
               'perception': pins, 'physical_identity_verified': False, 'task_success': 'not established'}
    if _selected(root) != (body, pins):
        raise ValueError('Selected world changed during Muse admission')
    for name, data in files.items():
        _check(deadline)
        read_package_file(body, name, max_bytes=max(1, len(data)), expected_sha256=sha256(data).hexdigest())
    raw = (json.dumps(receipt, sort_keys=True, indent=2, allow_nan=False) + '\n').encode()
    if len(raw) > 32 << 10:
        raise ValueError('Perception review receipt exceeds its bound')
    _check(deadline)
    target = root / RECEIPT
    if target.exists():
        read_package_file(root, RECEIPT, max_bytes=32 << 10, expected_sha256=sha256(raw).hexdigest())
    else:
        target.parent.mkdir(exist_ok=True)
        write_package_files(target.parent, {target.name: raw})
    return body, {'receipt_sha256': sha256(raw).hexdigest(), **receipt}
