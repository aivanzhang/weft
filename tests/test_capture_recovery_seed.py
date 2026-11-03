"""Source-only capture recovery ancestry must survive both downstream body adapters."""

import json
from pathlib import Path
import shutil
import time

import pytest

from test_pose_capture_recovery import study_fixture, prepare
from weft import exercise_body_source as body, shared_world_human
from weft.io import digest, write_json
from weft.person_inventory import make_person_inventory
from weft.pose_capture_recovery import PROOF_FILES
from weft.shared_world_lab import prepare_shared_world_seed
from weft.shared_world_packaging import collect_shared_source_files


@pytest.fixture(scope='module')
def analytical_recovered_body(tmp_path_factory):
    root=tmp_path_factory.mktemp('capture-recovery-seed')
    with pytest.MonkeyPatch.context() as patch:
        study,original=study_fixture(root,patch)
        poses=prepare(study,original,root/'recovered')
        result=body.prepare_exercise_body_source_from_evidence(poses,root/'body',native_clock=poses/'native-clock.json')
    return result


def build(kind,original,out):
    if kind=='seed':return prepare_shared_world_seed(original,out)
    inventory=make_person_inventory(json.loads((original/'body-source.json').read_bytes())['source_sha256'])
    shared_world_human.construct_human_only_run(original,inventory,out)
    return out


@pytest.mark.parametrize('kind',['seed','human_only'])
def test_capture_recovery_proofs_and_current_validator_survive_adapter(kind,analytical_recovered_body,tmp_path):
    original=analytical_recovered_body;before={p:p.read_bytes() for p in original.rglob('*') if p.is_file()}
    out=build(kind,original,tmp_path/'output')
    prefix='inputs/human-source/' if kind=='seed' else ''
    manifest=json.loads((out/'manifest.json').read_bytes())
    for name in PROOF_FILES:
        retained=prefix+name
        assert (out/retained).read_bytes()==(original/name).read_bytes()
        assert manifest['files'][retained]==digest(original/name)
    implementation='implementation/pose_capture_recovery.py'
    assert (out/implementation).read_bytes()==Path(body.__file__).with_name('pose_capture_recovery.py').read_bytes()
    assert manifest['files'][implementation]==digest(out/implementation)
    scene=json.loads((out/'scene.json').read_bytes())
    observed=collect_shared_source_files(scene,out,expected_manifest_sha256=digest(out/'manifest.json'),deadline=time.monotonic()+10)
    assert observed.files
    # Revalidation consumes the same original body bytes, not a reconstructed substitute.
    if kind=='seed':
        copied={p.relative_to(out/'inputs/human-source').as_posix():p.read_bytes() for p in (out/'inputs/human-source').rglob('*') if p.is_file()}
        name=json.loads(copied['body-source.json'])['source_artifact'];copied[name]=(out/name).read_bytes()
    else:copied={p.relative_to(out).as_posix():p.read_bytes() for p in out.rglob('*') if p.is_file()}
    _,observations=body.validate_body_source_files(copied)
    assert PROOF_FILES<=observations['provenance']['input_artifacts'].keys()
    assert all(p.read_bytes()==raw for p,raw in before.items())


@pytest.mark.parametrize('kind',['seed','human_only'])
@pytest.mark.parametrize('damage',['missing','changed'])
def test_missing_or_changed_recovery_ancestry_cannot_be_dropped_at_adapter(kind,damage,analytical_recovered_body,tmp_path):
    original=Path(shutil.copytree(analytical_recovered_body,tmp_path/'body'))
    name='capture-recovery/resource-failure.json'
    manifest=json.loads((original/'manifest.json').read_bytes())
    if damage=='missing':
        manifest['files'].pop(name)
    else:
        (original/name).write_bytes((original/name).read_bytes()+b' ')
        manifest['files'][name]=digest(original/name)
    write_json(original/'manifest.json',manifest)
    with pytest.raises(ValueError):build(kind,original,tmp_path/'output')
    assert not (tmp_path/'output/manifest.json').exists()
