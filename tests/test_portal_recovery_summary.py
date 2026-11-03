"""Recovery display reads bounded metadata, preserving original failure and selection."""

import json

import pytest

from test_portal_perception_recovery import failed_bridge, perception_source, recover
from weft.portal import Portal, equipment_recovery_summary
from weft.io import digest, write_json
from weft.portal_perception_recovery import start_recovery


@pytest.fixture
def recovered(failed_bridge):
    portal, root, _, project = failed_bridge
    job = json.loads((root/'job.json').read_bytes())
    job.update(name='Analytical recovered video.mp4', created_at='2026-11-02 12:00:00 PST')
    write_json(root/'job.json', job)
    recover(failed_bridge)
    return portal, root, project, job


def test_summary_is_bounded_metadata_only_and_receipt_download_is_exact(recovered, monkeypatch):
    portal_root, root, project, job = recovered
    import weft.portal_equipment_candidates as equipment
    import weft.shared_world_packaging as packaging
    import weft.exercise_body_source as bodies
    portal = Portal(portal_root, project=project, start=False)
    try:
        forbidden = lambda *_a, **_k: pytest.fail('Request summary cannot qualify large source evidence')
        monkeypatch.setattr(equipment, '_perception_source', forbidden)
        monkeypatch.setattr(equipment, 'validate_job_source', forbidden)
        monkeypatch.setattr(bodies, 'validate_body_source_files', forbidden)
        actual = packaging.read_package_file; reads = []
        def bounded(directory, name, **kwargs):
            assert name not in {'source.mp4', 'perception/bridge/manifest.json'}
            assert '/body_source/' not in name and kwargs['max_bytes'] <= 64 << 10
            value = actual(directory, name, **kwargs); reads.append((name, len(value)))
            return value
        monkeypatch.setattr(packaging, 'read_package_file', bounded)
        summary = portal.summary(root.name)
        value = summary['equipment_candidates']['recovery']
        assert value['status'] == 'recorded' and value['started'] is False
        assert value['validation_scope'] == 'bounded_metadata_only'
        assert value['api_calls'] == value['inference_calls'] == 0
        assert value['receipt_sha256'] == digest(root/value['artifact'])
        assert summary['perception']['status'] == 'failed'
        assert 'interruption' in summary['perception']['error']
        assert sum(size for _, size in reads) < 512 << 10
        assert portal.artifact(root.name, value['artifact']).read_bytes() == (root/'equipment/recovery.json').read_bytes()
        for denied in ('equipment/recovery-start.json', 'equipment/recovery-inputs/job.json'):
            with pytest.raises(ValueError, match='public project artifact|outside this project'):
                portal.artifact(root.name, denied)
    finally:
        portal.close()


def test_selected_root_status_does_not_erase_original_recovery(recovered):
    _, root, _, job = recovered
    start_recovery(root)
    before = (root/'perception/failure.json').read_bytes()
    write_json(root/'status.json', {'status': 'ready', 'route': 'exercise', 'lab': 'equipment/run/candidate/lab',
                                 'equipment_candidate_id': 'candidate-1'})
    summary = equipment_recovery_summary(root, job)
    assert summary['status'] == 'recorded' and summary['started'] is True
    assert summary['original_perception_status'] == 'failed'
    assert (root/'perception/failure.json').read_bytes() == before


@pytest.mark.parametrize('damage', ['receipt', 'start', 'request', 'original_failure', 'snapshot', 'descriptor', 'symlink'])
def test_changed_recovery_cannot_offer_verified_receipt(recovered, damage):
    portal_root, root, project, job = recovered
    start_recovery(root)
    paths = {'receipt': 'equipment/recovery.json', 'start': 'equipment/recovery-start.json',
             'request': 'equipment/request.json', 'original_failure': 'perception/failure.json',
             'snapshot': 'equipment/recovery-inputs/status.json', 'descriptor': 'perception/prepared-source.json',
             'symlink': 'equipment/recovery.json'}
    target = root/paths[damage]
    if damage == 'symlink':
        retained = root/'retained-receipt.json'; target.rename(retained); target.symlink_to(retained)
    else:
        target.write_bytes(b'{}')
    summary = equipment_recovery_summary(root, job)
    assert summary['status'] == 'unavailable' and 'artifact' not in summary
    portal = Portal(portal_root, project=project, start=False)
    try:
        with pytest.raises(ValueError, match='unavailable'):
            portal.artifact(root.name, 'equipment/recovery.json')
    finally:
        portal.close()


def test_unrecovered_project_has_no_added_status_or_artifact(tmp_path):
    root = tmp_path/'ordinary'; root.mkdir()
    assert equipment_recovery_summary(root, {}) is None
