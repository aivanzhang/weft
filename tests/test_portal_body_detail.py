"""Child custody fixtures; no video decode, pose inference, provider or environment runs."""

from copy import deepcopy
import fcntl
from hashlib import sha256
import json
import os
from pathlib import Path
import time

import pytest

from weft import portal_body_detail as detail
from weft import portal_body_detail_recipe as recipe
from weft.portal_inventory import exercise_intent
from test_person_region_proposal import documents, competing, encoded


def raw(value):
    return (json.dumps(value, sort_keys=True, indent=2) + '\n').encode()


def put(root, name, value):
    path = root / name; path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    path.write_bytes(value if isinstance(value, bytes) else raw(value))


def all_bytes(root):
    return {p.relative_to(root).as_posix(): p.read_bytes() for p in root.rglob('*') if p.is_file()}


@pytest.fixture
def case(tmp_path, monkeypatch):
    project = tmp_path / 'project'; project.mkdir(mode=0o700)
    portal = tmp_path / 'portal'; portal.mkdir(mode=0o700)
    parent = portal / ('a' * 32); parent.mkdir(mode=0o700)
    model = b'Analytic model bytes; not an inference model'
    monkeypatch.setattr(recipe, 'MODEL_SHA256', sha256(model).hexdigest())
    monkeypatch.setattr(recipe, 'MODEL_BYTES', len(model))
    put(project, '.local/models/' + recipe.MODEL_FILENAME, model)
    video = b'Analytic source custody bytes; not decoded video'
    pin = sha256(video).hexdigest()
    evidence, poses = documents(); competing(evidence, poses)
    evidence['source_sha256'] = pin
    for frame in evidence['frames']:
        frame['id'] = f"{pin}:frame:{frame['frame']}"
    poses['frames'] = deepcopy(evidence['frames'])
    evidence_raw, poses_raw = encoded(evidence, poses)
    job = {'id': parent.name, 'name': 'Analytic exercise.mp4', 'mode': 'exercise', 'kind': 'video', 'ai': False,
        'source_file': 'source.mp4', 'source_sha256': pin, 'exercise_intent': exercise_intent('equipment_or_obstacles', 'simulation')}
    put(parent, 'job.json', job)
    put(parent, 'source.mp4', video)
    put(parent, 'status.json', {'status': 'needs_input', 'route': 'exercise', 'events': []})
    for name in ['perception/status.json', 'perception/failure.json']:
        put(parent, name, {'status': 'failed', 'error': 'Synthetic retained pose ambiguity'})
    put(parent, 'perception/preflight/failure.json', {'stage': 'body', 'api_calls': 0})
    failure = raw({'profile': 'exercise-body-source-v1', 'status': 'failed'})
    files = {'evidence.json': evidence_raw, 'pose-landmarks.json': poses_raw, 'failure.json': failure, 'source.mp4': video}
    for name, value in files.items(): put(parent, detail.BODY + name, value)
    put(parent, detail.BODY + 'manifest.json', {'profile': 'exercise-body-source-v1', 'status': 'failed',
        'files': {name: sha256(value).hexdigest() for name, value in files.items()}})
    lease = os.open(portal / 'portal.lock', os.O_CREAT | os.O_RDWR, 0o600)
    fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
    result = {'root': portal, 'parent': parent, 'project': project, 'lease': lease, 'jobs': {parent.name: job}}
    yield result
    os.close(lease)


def create(case, **changes):
    options = {'lease_fd': case['lease'], 'existing_jobs': case['jobs'], 'stored_bytes': 100000,
        'reserved_bytes': 0, 'pending_jobs': 0, 'deadline': time.monotonic() + 5}
    options.update(changes)
    return detail.create_body_detail_child(case['root'], case['parent'].name,
        {'recipe': detail.RECIPE, 'request_id': 'b' * 32}, **options)


def started_child(case):
    config = create(case); child = case['root'] / config['id']
    put(child, 'perception/preflight-start.json', {'api_calls': 0, 'started_at': 'synthetic exclusive worker start'})
    return child


def test_handler_only_copies_small_metadata_and_worker_freezes_all_hypotheses(case, monkeypatch):
    before = all_bytes(case['parent']); seen = []
    original = detail._read
    def counted(root, name, **kwargs):
        seen.append(name); return original(root, name, **kwargs)
    monkeypatch.setattr(detail, '_read', counted)
    assert detail.body_detail_availability(case['parent'])['available']
    config = create(case); child = case['root'] / config['id']
    assert not (child / 'source.mp4').exists()
    assert not (child / 'body-detail/inputs').exists()
    assert not any(n in {'source.mp4', *detail.INPUT_FILES.values(), recipe.MODEL_FILENAME} for n in seen)
    assert all_bytes(case['parent']) == before
    assert json.loads((child / 'perception/status.json').read_bytes())['status'] == 'queued_preflight'
    put(child, 'perception/preflight-start.json', {'api_calls': 0})
    options = detail.prepare_queued_body_detail(child, project=case['project'], deadline=time.monotonic() + 10)
    assert set(options) == {'pose_model', 'body_detail_recipe', 'body_detail_inputs'}
    actual = options['body_detail_recipe']
    assert actual['region_proposal']['histories'][0]['hypotheses'] == 13
    assert len(actual['region_proposal']['histories'][0]['samples'][5]['hypotheses']) == 2
    assert actual['inference']['pose_sampling'] == 'all_frames'
    assert actual['model']['sha256'] == recipe.MODEL_SHA256
    assert actual['parent']['project_id'] == case['parent'].name
    assert (child / 'source.mp4').read_bytes() == before['source.mp4']
    assert all_bytes(case['parent']) == before
    assert not (child / 'perception/request.json').exists()
    assert not actual['inference_executed'] and not actual['simulation_ready']


def test_one_attempt_deduplicates_before_full_capacity_and_never_replaces_failed_child(case):
    child = create(case); case['jobs'][child['id']] = child
    path = case['root'] / child['id']
    put(path, 'perception/status.json', {'status': 'failed', 'reason': 'Preserved failed child'})
    before = all_bytes(path)
    assert create(case, stored_bytes=2_000_000_000, reserved_bytes=650_000_000, pending_jobs=4) == child
    different_id = detail.create_body_detail_child(case['root'], case['parent'].name,
        {'recipe': detail.RECIPE, 'request_id': 'c' * 32}, lease_fd=case['lease'], existing_jobs=case['jobs'],
        stored_bytes=2_000_000_000, reserved_bytes=0, pending_jobs=4, deadline=time.monotonic() + 5)
    assert different_id == child and all_bytes(path) == before
    summary = detail.body_detail_availability(path)
    assert not summary['available'] and summary['parent_id'] == case['parent'].name
    assert 'original failure is preserved' in summary['reason']


def test_request_id_cannot_collide_with_unrelated_upload(case):
    case['jobs']['d' * 32] = {'id': 'd' * 32, 'request_id': 'b' * 32}
    with pytest.raises(ValueError, match='already belongs'):
        create(case)
    assert len([p for p in case['root'].iterdir() if p.is_dir()]) == 1


@pytest.mark.parametrize('damage', ['active', 'selected', 'paid', 'incomplete', 'unsealed', 'chain', 'too_large'])
def test_ineligible_parent_refuses_without_child(case, damage, monkeypatch):
    p = case['parent']
    if damage == 'active': put(p, 'perception/status.json', {'status': 'preflighting'})
    elif damage == 'selected': put(p, 'status.json', {'status': 'ready', 'route': 'exercise', 'lab': 'lab'})
    elif damage == 'paid': put(p, 'perception/request.json', {'preserved': True})
    elif damage == 'incomplete': (p / detail.INPUT_FILES['pose-landmarks.json']).unlink()
    elif damage == 'unsealed':
        marker = json.loads((p / (detail.BODY + 'manifest.json')).read_bytes()); del marker['files']['evidence.json']
        put(p, detail.BODY + 'manifest.json', marker)
    elif damage == 'chain':
        job = dict(case['jobs'][p.name], body_detail={}); put(p, 'job.json', job); case['jobs'][p.name] = job
    else: monkeypatch.setattr(detail, 'MAX_INPUT_BYTES', 100)
    assert not detail.body_detail_availability(p)['available']
    with pytest.raises((ValueError, OSError)):
        create(case)
    assert [x.name for x in case['root'].iterdir() if x.is_dir()] == [p.name]


@pytest.mark.parametrize('change', [{'stored_bytes': 1_350_000_001}, {'reserved_bytes': 1_350_000_001}, {'pending_jobs': 4}])
def test_existing_quota_reservation_blocks_new_child(case, change):
    with pytest.raises(ValueError, match='reserve'):
        create(case, **change)
    assert len([p for p in case['root'].iterdir() if p.is_dir()]) == 1


@pytest.mark.parametrize('damage', ['source', 'raw', 'model', 'parent', 'request', 'cancel', 'missing_start'])
def test_worker_tamper_fails_before_recipe_or_inference_and_preserves_start(case, damage):
    child = started_child(case); parent = case['parent']
    if damage == 'source': put(parent, 'source.mp4', b'Changed original')
    elif damage == 'raw': put(parent, detail.INPUT_FILES['pose-landmarks.json'], b'{}')
    elif damage == 'model': put(case['project'], '.local/models/' + recipe.MODEL_FILENAME, b'Changed model')
    elif damage == 'parent': put(parent, 'perception/failure.json', {'status': 'failed', 'changed': True})
    elif damage == 'request':
        request = json.loads((child / detail.REQUEST).read_bytes()); request['recipe'] = 'other'
        put(child, detail.REQUEST, request)
    elif damage == 'cancel': put(child, 'perception/cancel-request.json', {})
    else: (child / 'perception/preflight-start.json').unlink()
    with pytest.raises((ValueError, OSError)):
        detail.prepare_queued_body_detail(child, project=case['project'], deadline=time.monotonic() + 5)
    assert not (child / 'body-detail/recipe.json').exists()
    assert not (child / 'source.mp4').exists()


def test_partial_worker_cannot_silently_repeat_or_overwrite(case):
    child = started_child(case)
    detail.prepare_queued_body_detail(child, project=case['project'], deadline=time.monotonic() + 5)
    before = all_bytes(child)
    with pytest.raises(ValueError, match='cannot be retried'):
        detail.prepare_queued_body_detail(child, project=case['project'], deadline=time.monotonic() + 5)
    assert all_bytes(child) == before


def test_closed_payload_paths_deadline_and_lease(case, tmp_path):
    with pytest.raises(TimeoutError): create(case, deadline=time.monotonic() - 1)
    with pytest.raises(ValueError, match='finite worker deadline'): create(case, deadline=float('nan'))
    other = os.open(tmp_path / 'unrelated-lock', os.O_CREAT | os.O_RDWR, 0o600)
    try:
        with pytest.raises(ValueError, match='lease'): create(case, lease_fd=other)
    finally: os.close(other)
    with pytest.raises(ValueError, match='fixed recipe'):
        detail.create_body_detail_child(case['root'], case['parent'].name,
            {'recipe': detail.RECIPE, 'request_id': 'c' * 32, 'model': '/arbitrary/path'},
            lease_fd=case['lease'], existing_jobs=case['jobs'], stored_bytes=0, reserved_bytes=0, pending_jobs=0,
            deadline=time.monotonic() + 5)


def test_publication_failure_preserves_private_stage_and_parent(case, monkeypatch):
    before = all_bytes(case['parent'])
    def fail(*args): raise OSError('Synthetic atomic publication failure')
    monkeypatch.setattr(detail.os, 'rename', fail)
    with pytest.raises(OSError): create(case)
    stages = list(case['root'].glob('.body-detail-*'))
    assert len(stages) == 1 and (stages[0] / 'publication-failure.json').is_file()
    assert all_bytes(case['parent']) == before
