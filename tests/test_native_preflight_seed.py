"""Native construction seeds must retain the complete original clock custody chain."""
import json
import time

import pytest

from test_exercise_body_capture import inputs
from test_exercise_body_native_clock import native_pose
from weft import exercise_body_source as body
from weft.io import digest, write_json
from weft.shared_world_lab import prepare_shared_world_seed
from weft.shared_world_packaging import collect_shared_source_files


def _verify_seed(original, seed):
    marker=json.loads((original/'body-source.json').read_bytes())
    assert (seed/marker['source_artifact']).read_bytes()==(original/marker['source_artifact']).read_bytes()
    assert (seed/'inputs/human-source/source-manifest.json').read_bytes()==(original/'manifest.json').read_bytes()
    names={'native-clock.json','reconstruction-evidence.json'}
    if (original/'decoder-clock.json').exists():names |= body.CAPTURE_FILES
    manifest=json.loads((seed/'manifest.json').read_bytes())
    for name in names:
        retained='inputs/human-source/'+name
        assert (seed/retained).read_bytes()==(original/name).read_bytes()
        assert manifest['files'][retained]==digest(original/name)
    scene=json.loads((seed/'scene.json').read_bytes())
    human=json.loads((original/'human.json').read_bytes())
    assert scene['human_spec']['native_clock']==human['native_clock']
    assert scene['human_spec']['reference']==human['reference']
    assert scene['coordinate_frame']['clock_binding']['alignment_method']=='native_pts_elapsed_time_v1'
    assert scene['dt']==1/30 and scene['substeps']==16
    assert (seed/'inputs/human-source/observations.json').read_bytes()==(original/'observations.json').read_bytes()
    collect_shared_source_files(scene,seed,expected_manifest_sha256=digest(seed/'manifest.json'),deadline=time.monotonic()+10)


def test_native_capture_seed_retains_all_clock_receipts_and_rejects_partial_chain(tmp_path,monkeypatch):
    movie,model,_=inputs(tmp_path,monkeypatch)
    original=tmp_path/'body'
    body.prepare_exercise_body_source(movie,original,pose_model=model,native_timing=True)
    before={p:p.read_bytes() for p in original.rglob('*') if p.is_file()}
    seed=prepare_shared_world_seed(original,tmp_path/'seed')
    _verify_seed(original,seed)
    assert all(p.read_bytes()==raw for p,raw in before.items())
    manifest=json.loads((original/'manifest.json').read_bytes())
    manifest['files'].pop('decoder-clock.json');write_json(original/'manifest.json',manifest)
    with pytest.raises(ValueError,match='sealed human'):
        prepare_shared_world_seed(original,tmp_path/'partial')
    assert not (tmp_path/'partial/manifest.json').exists()


def test_native_binding_without_capture_remains_supported(tmp_path):
    original,clock,_,_=native_pose(tmp_path/'pose')
    native=tmp_path/'body';body.prepare_exercise_body_source_from_evidence(original,native,native_clock=clock)
    seed=prepare_shared_world_seed(native,tmp_path/'seed')
    _verify_seed(native,seed)
    from weft.shared_world_lab import _adapted_shared_world
    observations=json.loads((native/'observations.json').read_bytes())
    observations['source']['native_clock_summary']['binding_sha256']='0'*64
    with pytest.raises(ValueError,match='same native source clock'):
        _adapted_shared_world(json.loads((native/'human.json').read_bytes()),observations,native)
