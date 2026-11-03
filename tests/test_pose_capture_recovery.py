"""Interrupted measurements are recoverable; interrupted bodies are not relabelled successful."""
import json
from pathlib import Path
import time

import pytest

from test_pose_sampling import dense_run
from test_source_clock_capture import native_log
from test_exercise_vision import inference_result
from weft import exercise_body_source as body, exercise_vision as vision, source_clock_capture as capture
from weft import pose_capture_recovery as recovery
from weft.io import digest, write_json


def study_fixture(tmp_path,monkeypatch,*,gap=False):
    study=tmp_path/'study';study.mkdir();source=dense_run(study/'body_source')
    (source/'manifest.json').unlink()
    e=json.loads((source/'evidence.json').read_text())
    if gap:
        for i in range(10,19):
            old=e['frames'][i]
            e['frames'][i]=vision.summarize_result(inference_result(count=0))|{k:old[k] for k in ('frame','id','time_s','decoder_timestamp_s','inference_seconds')}
        write_json(source/'pose-landmarks.json',{'frames':e['frames'],'model':e['model'],'scope':e['scope']})
        vision.resolve_pose_identity(e,source,'raw');vision.finalize_evidence(e,source)
    binary=tmp_path/'ffmpeg';binary.write_bytes(b'Analytical binary identity, never executed')
    monkeypatch.setattr(capture,'_bundled_ffmpeg',lambda:binary)
    monkeypatch.setattr(capture,'_run',lambda *args:native_log([90+1000*i for i in range(30)],time_base='1/30000',width=640,height=480))
    rows=[{'frame':i,'decoder_timestamp_s':i/30+.003} for i in range(30)]
    clock,receipt,log=capture.collect_native_clock(source/'source.webm',(source/'evidence.json').read_bytes(),rows)
    write_json(source/'native-clock.json',clock);write_json(source/'native-clock-capture.json',receipt)
    (source/'native-clock-decode.log').write_bytes(log)
    write_json(source/'decoder-clock.json',{'schema_version':1,'source_sha256':digest(source/'source.webm'),'complete':True,'rows':rows})
    snapshots={name:Path(vision.__file__).with_name(name).read_bytes() for name in recovery.PRODUCERS}
    (source/'implementation-source').mkdir()
    for name,raw in snapshots.items():(source/'implementation-source'/name).write_bytes(raw)
    write_json(source/'implementation.json',{'files':{name:digest(source/'implementation-source'/name) for name in snapshots}})
    write_json(study/'resource-failure.json',{'reason':'Sampled RSS exceeds1GiB','stage':'body','seconds':21.,'peak_rss_bytes':1_080_000_000})
    reseal(study)
    return study,source


def reseal(study):
    write_json(study/'manifest-study.json',{'schema_version':1,'profile':'dense-pose-partial-study-seal-v1',
        'files':{p.relative_to(study).as_posix():digest(p) for p in study.rglob('*') if p.is_file() and p.name!='manifest-study.json'}})


def prepare(study,source,out):
    return recovery.prepare_pose_capture_recovery(study,out,source_video=source/'source.webm',
        expected_study_manifest_sha256=digest(study/'manifest-study.json'))


def test_recovery_retains_failed_ancestry_and_native_raw_bytes_through_ordinary_body(tmp_path,monkeypatch):
    study,original=study_fixture(tmp_path,monkeypatch);before={p.relative_to(study).as_posix():p.read_bytes() for p in study.rglob('*') if p.is_file()}
    monkeypatch.setattr(vision,'inspect_exercise',lambda *a,**k:pytest.fail('No new pose inference'))
    monkeypatch.setattr(capture,'collect_native_clock',lambda *a,**k:pytest.fail('No new native decode'))
    saved=prepare(study,original,tmp_path/'poses')
    assert all((study/name).read_bytes()==raw for name,raw in before.items())
    for name in body.POSE_FILES|body.CAPTURE_FILES|{'native-clock.json','source.webm'}:
        assert (saved/name).read_bytes()==(original/name).read_bytes()
    assert json.loads((saved/'manifest.json').read_text())['status']=='pose_evidence_only'
    built=body.prepare_exercise_body_source_from_evidence(saved,tmp_path/'body',native_clock=saved/'native-clock.json')
    files={p.relative_to(built).as_posix():p.read_bytes() for p in built.rglob('*') if p.is_file()}
    _,observations=body.validate_body_source_files(files)
    assert recovery.PROOF_FILES<=files.keys()
    assert recovery.PROOF_FILES<=observations['provenance']['input_artifacts'].keys()
    assert json.loads(files['body-source.json'])['work']['inference_calls']==0
    assert json.loads(files['capture-recovery/resource-failure.json'])['reason']=='Sampled RSS exceeds1GiB'


@pytest.mark.parametrize('damage',['source','partial_decode','native_log','pose_quality','producer','path','failed_body','missing_pose','seal_pin'])
def test_tampered_or_incomplete_capture_cannot_be_recovered(tmp_path,monkeypatch,damage):
    study,original=study_fixture(tmp_path,monkeypatch)
    expected=digest(study/'manifest-study.json')
    if damage=='source':(original/'source.webm').write_bytes(b'changed')
    elif damage=='partial_decode':
        v=json.loads((original/'decoder-clock.json').read_text());v['complete']=False;write_json(original/'decoder-clock.json',v)
    elif damage=='native_log':(original/'native-clock-decode.log').write_bytes(b'not a native log')
    elif damage=='pose_quality':
        v=json.loads((original/'evidence.json').read_text());v['quality']['passes']=False;write_json(original/'evidence.json',v)
    elif damage=='producer':(original/'implementation-source/exercise_vision.py').write_bytes(b'not executed')
    elif damage=='failed_body':write_json(original/'failure.json',{'status':'failed'})
    elif damage=='missing_pose':(original/'pose-landmarks.json').unlink()
    elif damage=='seal_pin':expected='0'*64
    if damage not in {'path','seal_pin'}:reseal(study);expected=digest(study/'manifest-study.json')
    if damage=='path':
        m=json.loads((study/'manifest-study.json').read_text());m['files']['../outside']='0'*64;write_json(study/'manifest-study.json',m);expected=digest(study/'manifest-study.json')
    with pytest.raises((ValueError,KeyError)):
        recovery.prepare_pose_capture_recovery(study,tmp_path/'out',source_video=original/'source.webm',expected_study_manifest_sha256=expected)
    assert not (tmp_path/'out').exists()


def test_pose_only_recovery_does_not_relax_reference_gap(tmp_path,monkeypatch):
    study,original=study_fixture(tmp_path,monkeypatch,gap=True)
    poses=prepare(study,original,tmp_path/'poses')
    with pytest.raises(vision.UnsupportedVideo,match='interpolation horizon'):
        body.prepare_exercise_body_source_from_evidence(poses,tmp_path/'body',native_clock=poses/'native-clock.json')
    assert json.loads((tmp_path/'body/failure.json').read_text())['status']=='failed'
    assert json.loads((poses/'pose-capture-recovery.json').read_text())['status']=='pose_evidence_only'


@pytest.mark.parametrize('field,value',[('scope','Body and physical identity verified'),('schema_version',True),
                                      ('work',{'video_decodes':False,'inference_calls':0,'physics_steps':0,'api_calls':0})])
def test_recovery_receipt_cannot_promote_scope_or_change_typed_work(tmp_path,monkeypatch,field,value):
    study,original=study_fixture(tmp_path,monkeypatch);poses=prepare(study,original,tmp_path/'poses')
    files={p.relative_to(poses).as_posix():p.read_bytes() for p in poses.rglob('*') if p.is_file()}
    receipt=json.loads(files['pose-capture-recovery.json']);receipt[field]=value
    files['pose-capture-recovery.json']=json.dumps(receipt).encode()
    with pytest.raises(ValueError):
        recovery.validate_capture_recovery_files(files,source_sha256=digest(original/'source.webm'))


def test_aggregate_bound_rejects_before_publication(tmp_path,monkeypatch):
    study,original=study_fixture(tmp_path,monkeypatch)
    monkeypatch.setattr(recovery,'MAX_BYTES',100)
    with pytest.raises(ValueError):prepare(study,original,tmp_path/'poses')
    assert not (tmp_path/'poses').exists()


def test_original_body_cannot_become_terminal_during_recovery(tmp_path,monkeypatch):
    study,original=study_fixture(tmp_path,monkeypatch)
    validate=body._validated_pose
    def mutate_after_validation(*args,**kwargs):
        value=validate(*args,**kwargs)
        write_json(original/'failure.json',{'status':'failed'})
        return value
    monkeypatch.setattr(body,'_validated_pose',mutate_after_validation)
    with pytest.raises(ValueError,match='during capture recovery'):
        prepare(study,original,tmp_path/'poses')
    assert not (tmp_path/'poses').exists()
