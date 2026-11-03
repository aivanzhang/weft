"""Failed pose diagnostics explain ambiguity without retrying or changing source status."""

from hashlib import sha256
import json
from types import SimpleNamespace
import time

import pytest

from weft import portal_perception_jobs as jobs
from weft.io import digest, write_json
from weft.portal import Portal
from test_pose_identity_windows import fixture, encoded


def source(root, *, barriers=(9, 13)):
    root.mkdir(exist_ok=True)
    video=root/'source.mp4';video.write_bytes(b'fixture-only original video')
    pin=digest(video)
    write_json(root/'job.json',{'id':'a'*32,'source_file':video.name,'source_sha256':pin})
    write_json(root/'status.json',{'status':'needs_input','route':'exercise'})
    body=root/'perception/preflight/body_source';body.mkdir(parents=True)
    e,r=fixture(barriers=barriers);e['source_sha256']=pin
    for doc in (e,r):
        for row in doc['frames']:row['id']=f"{pin}:frame:{row['frame']}"
    eb,rb=encoded(e,r);(body/'evidence.json').write_bytes(eb);(body/'pose-landmarks.json').write_bytes(rb)
    failure={'status':'failed','error':'Original competing-pose refusal','retry':False}
    write_json(root/'perception/failure.json',failure);write_json(root/'perception/status.json',failure)
    return root,pin


def attach(root):return jobs.prepare_pose_failure_diagnostic(root,deadline=time.monotonic()+5)


def test_attachment_preserves_all_original_bytes_and_never_promotes_failed_job(tmp_path):
    root,pin=source(tmp_path/'job');before={p:p.read_bytes() for p in root.rglob('*') if p.is_file()}
    result=attach(root)
    assert result['sampled_frames']==24 and result['usable_samples']==22 and result['identity_breaks']==2
    assert result['simulation_ready'] is False and result['native_clock_complete'] is False
    assert all(p.read_bytes()==value for p,value in before.items())
    assert jobs.pose_failure_summary(root,pin)==result
    complete=json.loads((root/'perception/pose-diagnostic/report.json').read_bytes())
    assert len(complete['sample_slots'])==24 and len(complete['retained_inputs']['raw_poses']['frames'][9]['poses'])==2
    assert attach(root)==result
    assert jobs.read_status(root)['status']=='failed'


def test_request_summary_reads_only_bounded_metadata_not_pose_or_report_payloads(tmp_path,monkeypatch):
    root,pin=source(tmp_path/'job');attach(root);original=jobs.read_package_file;seen=[]
    def read(root,name,**kwargs):
        seen.append((name,kwargs['max_bytes']))
        assert not name.endswith(('evidence.json','pose-landmarks.json','report.json'))
        return original(root,name,**kwargs)
    monkeypatch.setattr(jobs,'read_package_file',read)
    result=jobs.summary(root,json.loads((root/'job.json').read_bytes()),{'status':'needs_input','route':'exercise'})
    assert result['pose_diagnostic']['identity_breaks']==2
    assert result['artifacts']['raw_poses'].endswith('pose-landmarks.json')
    assert max(size for _,size in seen)<=32<<10


@pytest.mark.parametrize('part',['failure','summary','source'])
def test_changed_handshake_hides_stale_summary_without_mutating_failure(tmp_path,part):
    root,pin=source(tmp_path/'job');attach(root)
    if part=='source':pin='0'*64
    else:
        p=root/('perception/failure.json' if part=='failure' else 'perception/pose-diagnostic/summary.json')
        p.write_bytes(p.read_bytes()+b' ')
    assert jobs.pose_failure_summary(root,pin) is None


def test_malformed_resealed_summary_does_not_break_portal_rendering(tmp_path):
    root,pin=source(tmp_path/'job');attach(root);base=root/'perception/pose-diagnostic'
    s=json.loads((base/'summary.json').read_bytes());s['listed_barriers'][0]['source_time_s']='not a timestamp';write_json(base/'summary.json',s)
    m=json.loads((base/'manifest.json').read_bytes());m['summary_sha256']=digest(base/'summary.json');write_json(base/'manifest.json',m)
    assert jobs.pose_failure_summary(root,pin) is None


def test_missing_decoder_time_field_is_not_accepted_as_explicit_unknown(tmp_path):
    root,pin=source(tmp_path/'job');attach(root);base=root/'perception/pose-diagnostic'
    s=json.loads((base/'summary.json').read_bytes());del s['listed_barriers'][0]['decoder_timestamp_s']
    write_json(base/'summary.json',s)
    m=json.loads((base/'manifest.json').read_bytes());m['summary_sha256']=digest(base/'summary.json');write_json(base/'manifest.json',m)
    assert jobs.pose_failure_summary(root,pin) is None
    s['listed_barriers'][0]['decoder_timestamp_s']=None;write_json(base/'summary.json',s)
    m['summary_sha256']=digest(base/'summary.json');write_json(base/'manifest.json',m)
    assert jobs.pose_failure_summary(root,pin)['listed_barriers'][0]['decoder_timestamp_s'] is None


@pytest.mark.parametrize('kind',['raw_mismatch','source_mismatch','deadline','not_failed','symlink'])
def test_rejected_diagnostic_leaves_original_error_unchanged(tmp_path,kind):
    root,pin=source(tmp_path/'job');original=(root/'perception/failure.json').read_bytes()
    if kind=='raw_mismatch':
        p=root/'perception/preflight/body_source/pose-landmarks.json';p.write_bytes(p.read_bytes()+b' ')
    if kind=='source_mismatch':
        j=json.loads((root/'job.json').read_bytes());j['source_sha256']='0'*64;write_json(root/'job.json',j)
    if kind=='not_failed':write_json(root/'perception/status.json',{'status':'cancelled'})
    if kind=='symlink':(root/'perception/pose-diagnostic').symlink_to(tmp_path)
    with pytest.raises((ValueError,TimeoutError)):
        jobs.prepare_pose_failure_diagnostic(root,deadline=time.monotonic()+(-1 if kind=='deadline' else 5))
    assert (root/'perception/failure.json').read_bytes()==original
    assert not (root/'perception/pose-diagnostic/manifest.json').is_file()


def test_no_competing_poses_does_not_create_identity_break_report(tmp_path):
    root,_=source(tmp_path/'job',barriers=())
    assert attach(root) is None
    assert not (root/'perception/pose-diagnostic').exists()


def test_existing_artifact_route_allows_only_explicit_diagnostic_and_raw_pose_paths(tmp_path):
    root,_=source(tmp_path/'job');attach(root)
    portal=SimpleNamespace(_folder=lambda _:root)
    for name in ('pose-diagnostic/report.json','pose-diagnostic/summary.json','pose-diagnostic/manifest.json',
                 'preflight/body_source/pose-landmarks.json'):
        assert Portal.artifact(portal,'unused','perception/'+name)==root/'perception'/name
    with pytest.raises(ValueError,match='public perception artifact'):
        Portal.artifact(portal,'unused','perception/pose-diagnostic/../../job.json')
    p=root/'perception/pose-diagnostic/report.json'
    with p.open('wb') as stream:stream.truncate((16<<20)+1)
    with pytest.raises(ValueError,match='download size'):Portal.artifact(portal,'unused','perception/pose-diagnostic/report.json')


def test_background_failure_hook_runs_diagnostic_once_without_retry(tmp_path,monkeypatch):
    import weft.exercise_perception_pipeline as bridge
    root,pin=source(tmp_path/'job');work=root/'perception'
    write_json(work/'parent-status.json',json.loads((root/'status.json').read_bytes()))
    write_json(work/'preflight-request.json',{'profile':jobs.PROFILE,'source_sha256':pin})
    write_json(work/'status.json',{'status':'queued_preflight'})
    calls=[]
    def reject(*a,**k):calls.append('preflight');raise ValueError('Original competing-pose refusal')
    monkeypatch.setattr(bridge,'prepare_exercise_perception_preflight',reject)
    assert jobs.execute_perception_job(root,project=tmp_path)==1
    assert jobs.pose_failure_summary(root,pin)['identity_breaks']==2
    assert jobs.read_status(root)['error']=='Original competing-pose refusal'
    assert jobs.execute_perception_job(root,project=tmp_path)==1 and calls==['preflight']


@pytest.mark.parametrize('error_type', [ValueError, RuntimeError])
def test_bad_optional_diagnostic_and_existing_error_receipt_never_mask_original_failure(tmp_path,monkeypatch,error_type):
    import weft.exercise_perception_pipeline as bridge
    root,pin=source(tmp_path/'job');work=root/'perception'
    write_json(work/'parent-status.json',json.loads((root/'status.json').read_bytes()))
    write_json(work/'preflight-request.json',{'profile':jobs.PROFILE,'source_sha256':pin})
    write_json(work/'status.json',{'status':'queued_preflight'})
    write_json(work/'pose-diagnostic-error.json',{'preserved':'previous diagnostic failure'})
    def reject(*a,**k):raise ValueError('Original body failure')
    monkeypatch.setattr(bridge,'prepare_exercise_perception_preflight',reject)
    def diagnostic_error(*a,**k):raise error_type('Unrelated optional diagnostic failure')
    monkeypatch.setattr(jobs,'prepare_pose_failure_diagnostic',diagnostic_error)
    assert jobs.execute_perception_job(root,project=tmp_path)==1
    assert jobs.read_status(root)['error']=='Original body failure'
    assert json.loads((work/'pose-diagnostic-error.json').read_bytes())=={'preserved':'previous diagnostic failure'}
