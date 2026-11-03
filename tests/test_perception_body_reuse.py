"""Copied bodies retain exact evidence while inference and provider accounting stay zero."""

import json
import time

import pytest

from test_exercise_perception_pipeline import local_preflight_source
from weft import exercise_perception_pipeline as pipeline
from weft.io import digest, write_json
from weft.perception_body_reuse import copy_sealed_body, validate_reuse_receipt


def test_retained_preflight_is_self_contained_without_inference(local_preflight_source, tmp_path, monkeypatch):
    import weft.exercise_body_source as bodies
    video, body = local_preflight_source
    original = {p.relative_to(body).as_posix(): p.read_bytes() for p in body.rglob('*') if p.is_file()}
    monkeypatch.setattr(bodies, 'prepare_exercise_body_source', lambda *a, **k: pytest.fail('No inference'))
    out = tmp_path/'preflight'
    result = pipeline.prepare_exercise_perception_preflight(video, out, body_source=body, retain_body_source=True)
    assert result['status'] == 'local_ready' and result['inference_calls'] == result['api_calls'] == 0
    assert {p.relative_to(out/'body_source').as_posix(): p.read_bytes()
            for p in (out/'body_source').rglob('*') if p.is_file()} == original
    body.rename(tmp_path/'source-no-longer-mounted')
    validated = pipeline.validate_perception_preflight(video, out, deadline=time.monotonic()+10)
    assert validated == result
    receipt = json.loads((out/'body-reuse.json').read_bytes())
    assert receipt['origin'] == 'sealed_body_copy'
    assert receipt['body_manifest_sha256'] == digest(out/'body_source/manifest.json')


@pytest.mark.parametrize('damage', ['source', 'body_manifest', 'copied_member', 'accounting', 'receipt_origin', 'implementation'])
def test_reuse_identity_and_accounting_cannot_be_relabelled(local_preflight_source, tmp_path, damage):
    video, body = local_preflight_source; out = tmp_path/'preflight'
    pipeline.prepare_exercise_perception_preflight(video, out, body_source=body, retain_body_source=True)
    manifest = json.loads((out/'manifest.json').read_bytes())
    if damage == 'body_manifest':
        (out/'body_source/manifest.json').write_bytes(b'{}')
    elif damage == 'copied_member':
        (out/'body_source/human.json').write_bytes(b'{}')
    elif damage == 'implementation':
        (out/'implementation/perception_body_reuse.py').write_bytes(b'# altered')
    else:
        name = 'result.json' if damage == 'accounting' else 'body-reuse.json'
        value = json.loads((out/name).read_bytes())
        if damage == 'source': value['source_sha256'] = '0'*64
        elif damage == 'accounting': value['inference_calls'] = 1
        else: value['origin'] = 'generated'
        write_json(out/name, value)
        manifest['files'][name] = digest(out/name)
        write_json(out/'manifest.json', manifest)
    with pytest.raises(ValueError):
        pipeline.validate_perception_preflight(video, out, deadline=time.monotonic()+10)


@pytest.mark.parametrize('damage', ['failed', 'source', 'member', 'deadline', 'overlap'])
def test_copy_rejects_invalid_original_before_sealing(local_preflight_source, tmp_path, damage):
    video, body = local_preflight_source; out = tmp_path/'copy'; deadline = time.monotonic()+10
    source = digest(video)
    if damage == 'failed': write_json(body/'failure.json', {'error': 'preserve'})
    elif damage == 'source': source = '0'*64
    elif damage == 'member': (body/'human.json').write_bytes(b'{}')
    elif damage == 'deadline': deadline = time.monotonic()-1
    else: out = body/'copy'
    with pytest.raises((ValueError, TimeoutError)):
        copy_sealed_body(body, out, source_sha256=source, deadline=deadline)
    assert not (out/'manifest.json').exists()


@pytest.mark.parametrize('field', ['schema_version', 'inference_calls', 'api_calls'])
def test_copy_receipt_rejects_boolean_work_accounting(local_preflight_source, tmp_path, field):
    video, body = local_preflight_source
    receipt = copy_sealed_body(body, tmp_path/'copy', source_sha256=digest(video), deadline=time.monotonic()+10)
    receipt[field] = bool(receipt[field])
    with pytest.raises(ValueError):
        validate_reuse_receipt(receipt, source_sha256=digest(video), body_manifest_sha256=digest(body/'manifest.json'))


def test_retention_requires_existing_body_before_decode(tmp_path, monkeypatch):
    import weft.segmentation_lab as tracking
    monkeypatch.setattr(tracking, 'inspect_video', lambda *a, **k: pytest.fail('Invalid request cannot decode'))
    with pytest.raises(ValueError, match='existing sealed body'):
        pipeline.prepare_exercise_perception_preflight(tmp_path/'source.mp4', tmp_path/'out', retain_body_source=True)


def test_copy_aggregate_limit_prevents_a_published_body(local_preflight_source, tmp_path, monkeypatch):
    import weft.perception_body_reuse as reuse
    video, body = local_preflight_source; out = tmp_path/'copy'
    monkeypatch.setattr(reuse, 'MAX_BYTES', (body/'manifest.json').stat().st_size+1)
    with pytest.raises(ValueError):
        copy_sealed_body(body, out, source_sha256=digest(video), deadline=time.monotonic()+10)
    assert not (out/'manifest.json').exists()


def test_copy_rechecks_original_failure_before_publication(local_preflight_source, tmp_path, monkeypatch):
    import weft.perception_body_reuse as reuse
    video, body = local_preflight_source; out = tmp_path/'copy'
    write = reuse.write_package_files
    def changed_source(target, files):
        write(target, files)
        write_json(body/'failure.json', {'error': 'became terminal during copy'})
    monkeypatch.setattr(reuse, 'write_package_files', changed_source)
    with pytest.raises(ValueError, match='changed while copying'):
        copy_sealed_body(body, out, source_sha256=digest(video), deadline=time.monotonic()+10)
    assert not (out/'manifest.json').exists()
