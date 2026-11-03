"""Oversized seeds retain exact original observations once, never fewer source rows."""
from copy import deepcopy
import hashlib
import json

import pytest

from test_shared_human_projection import _source_scene
from test_shared_world_packaging import seal
from weft import shared_world_lab as lab
from weft.shared_world_packaging import collect_shared_source_files, write_package_files


def source_files(root):
    scene,_=_source_scene(root)
    files=dict(collect_shared_source_files(scene,root).files)
    name=next(iter(scene['source_binding']['evidence_documents'].values()))
    files['inputs/human-source/observations.json']=files[name]
    files['scene.json']=lab._seed_bytes(scene)
    return scene,files,name


def force_overflow(monkeypatch,files):
    monkeypatch.setattr(lab,'SEED_MAX_BYTES',sum(map(len,files.values()))+(1<<20)-1)


def test_exact_observation_reuse_preserves_normal_source_and_outer_manifest_custody(tmp_path,monkeypatch):
    source=tmp_path/'source';source.mkdir()
    scene,files,name=source_files(source);before=deepcopy((scene,files))
    force_overflow(monkeypatch,files)
    receipt=lab._seed_reuse_observations(scene,files,name,construction_body=True)
    retained='inputs/human-source/observations.json'
    assert receipt['duplicate_payload_bytes_avoided']==len(before[1][name])
    assert receipt['document_sha256']==hashlib.sha256(before[1][name]).hexdigest()
    assert name not in files and files[retained]==before[1][name]
    assert len(files)==len(before[1])-1
    expected=deepcopy(before[0]);expected['source_binding']['evidence_documents'][receipt['document_sha256']]=retained
    assert scene==expected and files['scene.json']==lab._seed_bytes(scene)
    assert sum(map(len,files.values()))+(1<<20)<lab.SEED_MAX_BYTES
    output=tmp_path/'out';output.mkdir();write_package_files(output,files);seal(output)
    pinned=collect_shared_source_files(scene,output)
    assert pinned.files[retained]==before[1][name]
    assert pinned.report['source_evidence_verified']
    assert json.loads(pinned.files[retained])==json.loads(before[1][name])


@pytest.mark.parametrize('case',['small','not_body','unequal','missing_original'])
def test_no_lossy_or_unnecessary_storage_rewrite(tmp_path,monkeypatch,case):
    scene,files,name=source_files(tmp_path)
    if case!='small':force_overflow(monkeypatch,files)
    retained='inputs/human-source/observations.json'
    if case=='unequal':files[retained]+=b' '
    if case=='missing_original':files.pop(retained)
    before=deepcopy((scene,files))
    assert lab._seed_reuse_observations(scene,files,name,construction_body=case!='not_body') is None
    assert (scene,files)==before


@pytest.mark.parametrize('damage',['changed','missing','unsealed'])
def test_reused_original_cannot_be_changed_omitted_or_unpinned(tmp_path,monkeypatch,damage):
    source=tmp_path/'source';source.mkdir()
    scene,files,name=source_files(source);force_overflow(monkeypatch,files)
    receipt=lab._seed_reuse_observations(scene,files,name,construction_body=True)
    retained=receipt['artifact'];output=tmp_path/'out';output.mkdir()
    if damage=='changed':files[retained]+=b' '
    elif damage=='missing':files.pop(retained)
    write_package_files(output,files);seal(output)
    if damage=='unsealed':
        manifest=json.loads((output/'manifest.json').read_bytes());manifest['files'].pop(retained)
        (output/'manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError):collect_shared_source_files(scene,output)


def test_reuse_never_increases_bound_or_removes_other_large_payloads(tmp_path,monkeypatch):
    scene,files,name=source_files(tmp_path)
    files['unrelated-proof.bin']=b'keep'*1000
    monkeypatch.setattr(lab,'SEED_MAX_BYTES',1<<20)
    assert lab._seed_reuse_observations(scene,files,name,construction_body=True)
    assert files['unrelated-proof.bin']==b'keep'*1000
    assert sum(map(len,files.values()))+(1<<20)>lab.SEED_MAX_BYTES
    assert lab.SEED_MAX_BYTES==1<<20


from test_capture_recovery_seed import analytical_recovered_body


def test_full_seed_reuse_receipt_and_preserved_body_revalidate(analytical_recovered_body,tmp_path,monkeypatch):
    from weft.exercise_body_source import validate_body_source_files
    original_reuse=lab._seed_reuse_observations
    def exercise_overflow_branch(scene,files,name,**kwargs):
        with monkeypatch.context() as bounded:
            bounded.setattr(lab,'SEED_MAX_BYTES',sum(map(len,files.values()))+(1<<20)-1)
            return original_reuse(scene,files,name,**kwargs)
    monkeypatch.setattr(lab,'_seed_reuse_observations',exercise_overflow_branch)
    output=lab.prepare_shared_world_seed(analytical_recovered_body,tmp_path/'seed')
    scene=json.loads((output/'scene.json').read_bytes())
    receipt=json.loads((output/'construction-seed.json').read_bytes())
    assert receipt['source_binding']==scene['source_binding']
    assert receipt['source_evidence_storage']['artifact']=='inputs/human-source/observations.json'
    assert not (output/receipt['source_evidence_storage']['redundant_generated_path']).exists()
    pinned=collect_shared_source_files(scene,output)
    assert pinned.files['inputs/human-source/observations.json']==(analytical_recovered_body/'observations.json').read_bytes()
    kept={p.relative_to(output/'inputs/human-source').as_posix():p.read_bytes()
          for p in (output/'inputs/human-source').rglob('*') if p.is_file()}
    source_name=json.loads(kept['body-source.json'])['source_artifact']
    kept[source_name]=(output/source_name).read_bytes()
    human,observations=validate_body_source_files(kept)
    assert observations==json.loads((analytical_recovered_body/'observations.json').read_bytes())
    assert human==json.loads((analytical_recovered_body/'human.json').read_bytes())
    assert sum(p.stat().st_size for p in output.rglob('*') if p.is_file())<=128<<20
