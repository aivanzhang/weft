"""Stopped-portal reuse preserves failed parents and never silently runs provider work."""

import fcntl
import json
import os
from pathlib import Path
import time

import pytest

from test_exercise_perception_pipeline import local_preflight_source
from weft import portal_body_reuse as reuse
from weft import portal_perception_jobs as jobs
from weft.io import digest, write_json
from weft.portal import Portal


def files(root):
    return {p.relative_to(root).as_posix(): p.read_bytes() for p in root.rglob('*') if p.is_file()}


@pytest.fixture
def inputs(local_preflight_source, tmp_path):
    video, body = local_preflight_source
    root = tmp_path/'portal'; root.mkdir(mode=0o700)
    parent = root/('a'*32); parent.mkdir(mode=0o700)
    (parent/'source.mp4').write_bytes(video.read_bytes())
    write_json(parent/'job.json', {'id': parent.name, 'name': 'Exercise.mp4', 'mode': 'exercise',
        'ai': False, 'source_file': 'source.mp4', 'source_sha256': digest(video)})
    write_json(parent/'status.json', {'status': 'needs_input', 'route': 'exercise', 'events': []})
    (parent/'perception').mkdir()
    failure = {'status': 'failed', 'error': 'Preserved earlier pose failure', 'api_calls': 0}
    write_json(parent/'perception/status.json', failure)
    write_json(parent/'perception/failure.json', failure)
    return root, parent, body


def create(inputs, **kwargs):
    root, parent, body = inputs
    return reuse.reuse_body(root, parent.name, body, digest(body/'manifest.json'), **kwargs)


def test_new_child_is_idempotent_and_queued_across_stopped_portal_restart(inputs, tmp_path):
    root, parent, body = inputs
    before_parent, before_body = files(parent), files(body)
    result = create(inputs, project=tmp_path)
    child = Path(result['directory'])
    assert result['status'] == 'queued_preflight' and result['api_calls'] == result['inference_calls'] == 0
    assert child != parent and files(child/reuse.BODY) == before_body
    assert reuse.validate_pending_reuse(child)
    original_request = (child/'perception/preflight-request.json').read_bytes()
    portal = Portal(root, project=tmp_path, start=False)
    try:
        assert jobs.read_status(child)['status'] == 'queued_preflight'
        assert portal.summary(child.name)['kind'] == 'video'
    finally:
        portal.close()
    assert create(inputs, project=tmp_path)['id'] == child.name
    assert create(inputs, project=tmp_path)['status'] == 'already_created'
    assert (child/'perception/preflight-request.json').read_bytes() == original_request
    assert files(parent) == before_parent and files(body) == before_body
    assert reuse.validate_queued_body(child, deadline=time.monotonic()+10) == child/reuse.BODY


def test_actual_worker_retains_body_and_waits_for_separate_consent(inputs, tmp_path, monkeypatch):
    import weft.exercise_body_source as bodies
    import weft.exercise_perception_pipeline as pipeline
    root, parent, body = inputs
    before = files(parent)
    child = Path(create(inputs, project=tmp_path)['directory'])
    monkeypatch.setattr(bodies, 'prepare_exercise_body_source', lambda *_a, **_k: pytest.fail('No pose inference'))
    monkeypatch.setattr(jobs, '_budget_guard', lambda *_a, **_k: pytest.fail('No provider reservation'))
    assert jobs.execute_perception_job(child, project=tmp_path) == 0
    assert jobs.read_status(child)['status'] == 'awaiting_consent'
    result = pipeline.validate_perception_preflight(child/'source.mp4', child/'perception/preflight',
                                                    deadline=time.monotonic()+10)
    assert result['inference_calls'] == result['api_calls'] == 0
    assert files(child/'perception/preflight/body_source') == files(body)
    assert not (child/'perception/request.json').exists()
    assert files(parent) == before


@pytest.mark.parametrize('damage', ['pin', 'source', 'member', 'failure', 'active', 'selected'])
def test_invalid_original_refuses_before_child_publication(inputs, damage):
    root, parent, body = inputs
    pin = digest(body/'manifest.json')
    if damage == 'pin': pin = '0'*64
    elif damage == 'source': (parent/'source.mp4').write_bytes(b'changed')
    elif damage == 'member': (body/'human.json').write_bytes(b'{}')
    elif damage == 'failure': write_json(body/'failure.json', {'error': 'retained'})
    elif damage == 'active': write_json(parent/'perception/status.json', {'status': 'preflighting'})
    else: write_json(parent/'status.json', {'status': 'ready', 'lab': 'lab'})
    with pytest.raises(ValueError):
        reuse.reuse_body(root, parent.name, body, pin)
    assert [p.name for p in root.iterdir() if len(p.name) == 32] == [parent.name]


def test_live_lease_and_capacity_do_not_create_children(inputs, monkeypatch):
    root, _, _ = inputs
    fd = os.open(root/'portal.lock', os.O_CREAT | os.O_RDWR, 0o600)
    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    try:
        with pytest.raises(ValueError, match='Stop the portal'):
            create(inputs)
    finally:
        os.close(fd)
    monkeypatch.setattr(reuse, '_storage', lambda *_: 2_000_000_000)
    with pytest.raises(ValueError, match='capacity'):
        create(inputs)
    assert not list(root.glob('.reuse-*'))


@pytest.mark.parametrize('started', ['preflight-start.json', 'request.json', 'preflight', 'failure.json',
                                   'result.json', 'quote.json', 'preflight-complete.json', 'cancel-request.json', 'receipt_tamper'])
def test_restart_never_retries_started_or_tampered_reuse(inputs, tmp_path, started):
    root, _, _ = inputs
    child = Path(create(inputs)['directory'])
    target = child/reuse.RECEIPT if started == 'receipt_tamper' else child/'perception'/started
    if started == 'preflight': target.mkdir()
    else: target.write_bytes(b'{}')
    assert not reuse.validate_pending_reuse(child)
    portal = Portal(root, project=tmp_path, start=False)
    try:
        assert jobs.read_status(child)['status'] == 'interrupted'
    finally:
        portal.close()
    assert target.exists()


@pytest.mark.parametrize('damage', ['body', 'receipt', 'request', 'source'])
def test_worker_revalidates_pins_before_preflight_or_provider(inputs, monkeypatch, damage):
    import weft.exercise_perception_pipeline as pipeline
    child = Path(create(inputs)['directory'])
    paths = {'body': child/reuse.BODY/'human.json', 'receipt': child/reuse.RECEIPT,
             'request': child/'perception/preflight-request.json', 'source': child/'source.mp4'}
    paths[damage].write_bytes(b'{}')
    monkeypatch.setattr(pipeline, 'prepare_exercise_perception_preflight', lambda *_a, **_k: pytest.fail('No invalid preflight'))
    monkeypatch.setattr(jobs, '_budget_guard', lambda *_a, **_k: pytest.fail('No provider work'))
    assert jobs.execute_perception_job(child) == 1
    assert jobs.read_status(child)['status'] in {'failed', 'unsupported'}
    assert (child/'perception/preflight-start.json').exists() == (damage in {'body', 'receipt'})


def test_removed_both_reuse_markers_never_falls_back_to_inference(inputs, monkeypatch):
    import weft.exercise_perception_pipeline as pipeline
    child = Path(create(inputs)['directory'])
    for name in ('job.json', 'perception/preflight-request.json'):
        data = json.loads((child/name).read_bytes()); data.pop('body_reuse')
        write_json(child/name, data)
    monkeypatch.setattr(pipeline, 'prepare_exercise_perception_preflight', lambda *_a, **_k: pytest.fail('No inference fallback'))
    assert jobs.execute_perception_job(child) == 1
    assert (child/'perception/preflight-start.json').exists()


def test_reuse_validation_start_is_durable_before_expensive_work(inputs, monkeypatch, tmp_path):
    root, _, _ = inputs
    child = Path(create(inputs)['directory'])
    def fail_during_validation(folder, **_):
        assert folder == child and (child/'perception/preflight-start.json').is_file()
        raise SystemExit('Analytic crash during full body validation')
    monkeypatch.setattr(reuse, 'validate_queued_body', fail_during_validation)
    with pytest.raises(SystemExit, match='Analytic crash'):
        jobs.execute_perception_job(child)
    assert jobs.read_status(child)['status'] == 'queued_preflight'
    assert not reuse.validate_pending_reuse(child)
    portal = Portal(root, project=tmp_path, start=False)
    try:
        assert jobs.read_status(child)['status'] == 'interrupted'
    finally:
        portal.close()


def test_duplicate_child_does_not_readmit_entire_body(inputs, monkeypatch):
    result = create(inputs)
    monkeypatch.setattr(reuse, '_body', lambda *_a, **_k: pytest.fail('Duplicate must not repeat full body admission'))
    duplicate = create(inputs)
    assert duplicate['id'] == result['id'] and duplicate['status'] == 'already_created'
    assert duplicate['revalidated'] is False


def test_cli_routes_exact_manifest_pin_without_running_work(inputs, monkeypatch, capsys):
    import sys
    from weft import cli
    root, parent, body = inputs
    pin = digest(body/'manifest.json')
    seen = []
    def record(*args):
        seen.append(args)
        return {'status': 'queued_preflight'}
    monkeypatch.setattr(reuse, 'reuse_body', record)
    monkeypatch.setattr(sys, 'argv', ['weft', 'portal-reuse-body', '--root', str(root), '--parent-project', parent.name,
        '--body-source', str(body), '--body-manifest-sha256', pin])
    cli.main()
    assert seen == [(str(root), parent.name, str(body), pin)]
    assert 'queued_preflight' in capsys.readouterr().out
