"""Completed perception recovery reuses sealed receipts without retrying providers."""

import fcntl
import json
import os
from pathlib import Path

import pytest

from test_portal_equipment_candidates import perception_source
from weft import portal_perception_recovery as recovery
from weft import portal_equipment_candidates as equipment
from weft.io import digest, write_json
from weft.portal import Portal


def files(root):
    return {p.relative_to(root).as_posix(): p.read_bytes() for p in root.rglob('*') if p.is_file()}


@pytest.fixture
def failed_bridge(perception_source, tmp_path):
    original, _, project = perception_source
    portal = tmp_path/'portal'; portal.mkdir(mode=0o700)
    root = portal/('c'*32); original.rename(root)
    bridge = root/'perception/bridge'
    result = json.loads((bridge/'result.json').read_bytes())
    for name in ('base_run', 'evidence_run'):
        result[name] = str(root/Path(result[name]).relative_to(original))
    write_json(bridge/'result.json', result)
    manifest = json.loads((bridge/'manifest.json').read_bytes())
    manifest['files']['result.json'] = digest(bridge/'result.json')
    write_json(bridge/'manifest.json', manifest)
    job = json.loads((root/'job.json').read_bytes()); job.update(id=root.name, mode='exercise', ai=False)
    write_json(root/'job.json', job)
    failure = {'status': 'failed', 'error': 'Analytical source publication interruption after complete export', 'api_calls': 2}
    write_json(root/'perception/status.json', failure)
    write_json(root/'perception/failure.json', failure)
    return portal, root, bridge, project


def recover(fixture):
    portal, root, bridge, project = fixture
    return recovery.recover_saved_perception(portal, root.name,
        bridge_manifest_sha256=digest(bridge/'manifest.json'), project=project)


def test_ordinary_saved_source_publication_queues_existing_equipment_without_changing_failure(failed_bridge, monkeypatch):
    portal, root, bridge, project = failed_bridge
    before = files(root)
    import weft.portal_pipeline as pipeline
    import weft.exercise_perception_pipeline as perception
    import weft.exercise_body_source as bodies
    def forbidden(*_a, **_k): pytest.fail('Recovery cannot infer, analyze or track')
    monkeypatch.setattr(pipeline, 'analyze_frames', forbidden)
    monkeypatch.setattr(perception, 'prepare_exercise_perception', forbidden)
    monkeypatch.setattr(bodies, 'prepare_exercise_body_source', forbidden)
    result = recover(failed_bridge)
    assert result['status'] == 'queued' and result['api_calls'] == result['inference_calls'] == 0
    assert all((root/name).read_bytes() == raw for name, raw in before.items())
    assert recovery.validate_pending_recovery(root)
    request = json.loads((root/'equipment/request.json').read_bytes())
    assert request == {'profile': equipment.PROFILE, 'source_kind': 'perception',
        'source_sha256': digest(root/'source.mp4'), 'descriptor_sha256': digest(root/equipment.JOB_SOURCE_PATH)}
    saved = files(root)
    monkeypatch.setattr(equipment, 'publish_perception_source', forbidden)
    assert recover(failed_bridge)['status'] == 'already_requested'
    assert files(root) == saved
    assert not (portal/equipment.CATALOG).exists()


@pytest.mark.parametrize('damage', ['caller_pin', 'source', 'child', 'failure', 'active', 'existing_equipment'])
def test_invalid_saved_evidence_cannot_queue_or_overwrite(failed_bridge, damage):
    portal, root, bridge, project = failed_bridge
    pin = digest(bridge/'manifest.json')
    if damage == 'caller_pin': pin = '0'*64
    elif damage == 'source': (root/'source.mp4').write_bytes(b'changed')
    elif damage == 'child': (bridge/'tracking/manifest.json').write_bytes(b'{}')
    elif damage == 'failure': write_json(root/'perception/failure.json', {'status': 'failed', 'different': True})
    elif damage == 'active': write_json(root/'perception/status.json', {'status': 'exporting_evidence'})
    else: (root/'equipment').mkdir()
    before = files(root)
    with pytest.raises(ValueError):
        recovery.recover_saved_perception(portal, root.name, bridge_manifest_sha256=pin, project=project)
    assert all((root/name).read_bytes() == raw for name, raw in before.items())
    assert not (root/'equipment/request.json').exists()


def test_lease_and_capacity_precede_source_publication(failed_bridge, monkeypatch):
    portal, root, _, _ = failed_bridge
    monkeypatch.setattr(equipment, 'publish_perception_source', lambda *_a, **_k: pytest.fail('No publication'))
    fd = os.open(portal/'portal.lock', os.O_CREAT | os.O_RDWR, 0o600)
    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    try:
        with pytest.raises(ValueError, match='Stop the portal'): recover(failed_bridge)
    finally:
        os.close(fd)
    monkeypatch.setattr(recovery, '_storage', lambda *_a: 2_000_000_000)
    with pytest.raises(ValueError, match='capacity'): recover(failed_bridge)
    assert not (root/'equipment').exists()


@pytest.mark.parametrize('name', ['recovery-start.json', 'run', 'source-validation.json', 'candidates.json',
                                'cancel-request.json', 'selection.json', 'failure.json'])
def test_started_recovery_never_resumes_automatically(failed_bridge, name):
    _, root, _, _ = failed_bridge
    recover(failed_bridge)
    target = root/'equipment'/name
    if name == 'run': target.mkdir()
    else: target.write_bytes(b'{}')
    assert not recovery.validate_pending_recovery(root)


def test_recovery_start_marker_precedes_receipt_validation(failed_bridge):
    _, root, _, _ = failed_bridge
    recover(failed_bridge)
    (root/'equipment/recovery.json').write_bytes(b'{}')
    with pytest.raises(ValueError): recovery.start_recovery(root)
    assert (root/'equipment/recovery-start.json').is_file()
    assert not recovery.validate_pending_recovery(root)


def test_restart_resumes_only_never_started_recovery_without_rewriting_failure(failed_bridge):
    portal_root, root, _, project = failed_bridge
    failure = (root/'perception/status.json').read_bytes()
    recover(failed_bridge)
    portal = Portal(portal_root, project=project, start=False)
    try:
        assert equipment.read_status(root)['status'] == 'queued'
    finally:
        portal.close()
    assert (root/'perception/status.json').read_bytes() == failure
    recovery.start_recovery(root)
    portal = Portal(portal_root, project=project, start=False)
    try:
        assert equipment.read_status(root)['status'] == 'interrupted'
    finally:
        portal.close()
    assert (root/'perception/status.json').read_bytes() == failure


@pytest.mark.parametrize('tamper', ['receipt', 'request', 'bridge', 'descriptor'])
def test_restart_rejects_changed_recovery_metadata(failed_bridge, tamper):
    portal_root, root, _, project = failed_bridge
    recover(failed_bridge)
    paths = {'receipt': recovery.RECEIPT, 'request': 'equipment/request.json', 'bridge': recovery.BRIDGE,
             'descriptor': recovery.SOURCE}
    (root/paths[tamper]).write_bytes(b'{}')
    portal = Portal(portal_root, project=project, start=False)
    try:
        assert equipment.read_status(root)['status'] == 'interrupted'
    finally:
        portal.close()


def test_cli_keeps_explicit_bridge_pin_and_only_calls_recovery(failed_bridge, monkeypatch, capsys):
    import sys
    from weft import cli
    portal_root, root, bridge, _ = failed_bridge
    pin = digest(bridge/'manifest.json'); calls = []
    def capture(*args, **kwargs):
        calls.append((args, kwargs))
        return {'status': 'queued', 'api_calls': 0}
    monkeypatch.setattr(recovery, 'recover_saved_perception', capture)
    monkeypatch.setattr(sys, 'argv', ['weft', 'portal-recover-perception', '--root', str(portal_root),
        '--project-id', root.name, '--bridge-manifest-sha256', pin])
    cli.main()
    assert calls == [((str(portal_root), root.name), {'bridge_manifest_sha256': pin})]
    assert 'queued' in capsys.readouterr().out


def test_publication_receives_remaining_recovery_deadline(failed_bridge, monkeypatch):
    import time
    original = equipment.publish_perception_source
    started = time.monotonic(); seen = []
    def publish(*args, **kwargs):
        seen.append(kwargs['deadline'])
        assert started < kwargs['deadline'] <= started+40.1
        return original(*args, **kwargs)
    monkeypatch.setattr(equipment, 'publish_perception_source', publish)
    assert recover(failed_bridge)['status'] == 'queued'
    assert len(seen) == 1


def test_worker_uses_existing_source_admission_and_assembly_dispatch(failed_bridge, monkeypatch):
    import weft.equipment_conditional_assembly as assembly
    _, root, _, project = failed_bridge
    recover(failed_bridge)
    original_failure = (root/'perception/failure.json').read_bytes()
    actual = equipment.validate_job_source
    calls = []
    def validate(*args, **kwargs):
        assert (root/'equipment/recovery-start.json').is_file()
        calls.append('source_admission')
        return actual(*args, **kwargs)
    def construct(**kwargs):
        calls.append('assembly_boundary')
        assert kwargs['out'] == root/'equipment/run'
        return {'status': 'abstained', 'candidates': [], 'reasons': ['Analytic orchestration boundary; no physics executed']}
    monkeypatch.setattr(equipment, 'validate_job_source', validate)
    monkeypatch.setattr(assembly, 'prepare_conditional_equipment_scene', construct)
    assert equipment.execute_equipment_job(root, project=project) == 0
    assert calls == ['source_admission', 'assembly_boundary']
    assert (root/'perception/failure.json').read_bytes() == original_failure
    assert (root/'perception/status.json').read_bytes() == original_failure


@pytest.mark.parametrize('damage', ['receipt', 'failure', 'removed_receipt'])
def test_worker_rejects_changed_recovery_before_source_admission(failed_bridge, monkeypatch, damage):
    _, root, _, project = failed_bridge
    recover(failed_bridge)
    if damage == 'failure':
        (root/'perception/failure.json').write_bytes(b'{}')
    elif damage == 'removed_receipt':
        (root/'equipment/recovery.json').rename(root/'equipment/retained-invalid-receipt.json')
    else:
        (root/'equipment/recovery.json').write_bytes(b'{}')
    monkeypatch.setattr(equipment, 'validate_job_source', lambda *_a, **_k: pytest.fail('Invalid receipt cannot admit source'))
    assert equipment.execute_equipment_job(root, project=project) == 1
    assert (root/'equipment/recovery-start.json').is_file()
    assert equipment.read_status(root)['status'] == 'failed'
