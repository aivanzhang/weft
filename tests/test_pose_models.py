from hashlib import sha256
import io
import json
from pathlib import Path
import sys

import pytest

import weft.exercise_vision as vision


@pytest.fixture
def models(monkeypatch):
    payloads = {'lite': b'known lite model', 'full': b'known full local model', 'heavy': b'known heavy local model'}
    monkeypatch.setattr(vision, 'MODEL_VARIANTS', {name: {'bytes': len(data), 'sha256': sha256(data).hexdigest()}
                                               for name, data in payloads.items()})
    return payloads


@pytest.mark.parametrize('variant', ['lite', 'full', 'heavy'])
def test_model_identity_depends_on_bytes_not_filename(tmp_path, models, variant):
    model = tmp_path/'arbitrary-name.task'; model.write_bytes(models[variant])
    result = vision.model_provenance(model)
    assert result['variant'] == variant and result['network_inference'] is False
    assert result['sha256'] == sha256(model.read_bytes()).hexdigest()
    assert f'pose_landmarker_{variant}/float16/1' in result['source_url']


def test_wrong_requested_variant_preserves_existing_model(tmp_path, models, monkeypatch):
    model = tmp_path/'pose_landmarker_lite.task'; model.write_bytes(models['full'])
    monkeypatch.setattr(vision.urllib.request, 'urlopen', lambda *a, **k: pytest.fail('No download for conflicting file'))
    with pytest.raises(ValueError, match='pinned official'): vision.fetch_pose_model(model)
    assert model.read_bytes() == models['full']
    assert not model.with_suffix('.provenance.json').exists()


def test_modified_model_with_expected_size_rejects(tmp_path, models):
    model = tmp_path/'model.task'; model.write_bytes(b'x'*len(models['full']))
    with pytest.raises(ValueError, match='pinned official'): vision.model_provenance(model)


@pytest.mark.parametrize('variant', ['full', 'heavy'])
def test_fetch_is_pinned_and_cached_without_another_request(tmp_path, models, monkeypatch, variant):
    model = tmp_path/'model.task'
    calls = []
    def response(request, timeout):
        calls.append((request.full_url, timeout))
        return io.BytesIO(models[variant])
    monkeypatch.setattr(vision.urllib.request, 'urlopen', response)
    assert vision.fetch_pose_model(model, variant=variant) == model
    first_sidecar = model.with_suffix('.provenance.json').read_bytes()
    assert json.loads(first_sidecar)['variant'] == variant
    assert vision.fetch_pose_model(model, variant=variant) == model
    assert model.with_suffix('.provenance.json').read_bytes() == first_sidecar
    assert calls == [(f'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_{variant}/float16/1/pose_landmarker_{variant}.task', 30)]


@pytest.mark.parametrize('kind', ['oversized', 'truncated', 'wrong_hash', 'concurrent_file'])
def test_failed_or_conflicting_download_never_installs_or_overwrites(tmp_path, models, monkeypatch, kind):
    model = tmp_path/'model.task'
    data = models['full']
    if kind == 'oversized': data += b'extra'
    elif kind == 'truncated': data = data[:-1]
    elif kind == 'wrong_hash': data = b'x'*len(data)
    def response(*args, **kwargs):
        if kind == 'concurrent_file': model.write_bytes(b'preserve concurrent owner')
        return io.BytesIO(data)
    monkeypatch.setattr(vision.urllib.request, 'urlopen', response)
    with pytest.raises(FileExistsError if kind == 'concurrent_file' else ValueError):
        vision.fetch_pose_model(model, variant='full')
    if kind == 'concurrent_file': assert model.read_bytes() == b'preserve concurrent owner'
    else: assert not model.exists()
    assert not model.with_suffix('.provenance.json').exists()
    assert not list(tmp_path.glob('.pose-download-*'))


def test_unknown_variant_rejects_before_io(tmp_path, monkeypatch):
    monkeypatch.setattr(vision.urllib.request, 'urlopen', lambda *a, **k: pytest.fail('No request for unknown variant'))
    path = tmp_path/'new'/'model.task'
    with pytest.raises(ValueError, match='lite, full or heavy'): vision.fetch_pose_model(path, variant='unknown')
    assert not path.parent.exists()


@pytest.mark.parametrize('variant', ['lite', 'full', 'heavy'])
def test_cli_preserves_lite_default_and_supports_explicit_variants(monkeypatch, capsys, variant):
    from weft.cli import main
    calls = []
    def fetch(out, *, variant):
        calls.append((out, variant)); return Path('/local')/variant
    monkeypatch.setattr(vision, 'fetch_pose_model', fetch)
    monkeypatch.setattr(sys, 'argv', ['weft', 'fetch-pose-model']+(['--variant', variant] if variant != 'lite' else []))
    main()
    assert calls == [(None, variant)]
    assert json.loads(capsys.readouterr().out)['path'] == '/local/'+variant
