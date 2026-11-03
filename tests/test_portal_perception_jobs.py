"""Consent/lifecycle orchestration fixtures; source/native admission is tested by its own builders."""

from copy import deepcopy
import json
from pathlib import Path

import pytest

from weft import portal_perception_jobs as jobs
from weft.io import digest, write_json


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    root = tmp_path/'job';root.mkdir()
    video = root/'source.mp4';video.write_bytes(b'Explicit orchestration fixture, not decoded video')
    write_json(root/'job.json', {'id':'a'*32,'source_file':video.name,'source_sha256':digest(video), 'mode':'exercise'})
    write_json(root/'status.json', {'status':'needs_input','route':'exercise','events':[{'message':'original upload'}]})
    calls=[]
    import weft.exercise_perception_pipeline as bridge
    def local(video,out,**kwargs):
        assert kwargs['anchor_profile'] == bridge.VISIBLE_ANCHOR_PROFILE
        calls.append('local');out.mkdir();write_json(out/'manifest.json',{'fixture':True})
        return {'metadata':{'frame_count':100},'manifest_sha256':digest(out/'manifest.json')}
    monkeypatch.setattr(bridge,'prepare_exercise_perception_preflight',local)
    def validate(video,out,**kwargs):
        calls.append('validate')
        return {'metadata':{'frame_count':100},'manifest_sha256':digest(out/'manifest.json')}
    monkeypatch.setattr(bridge,'validate_perception_preflight',validate)
    monkeypatch.setattr(jobs,'_budget_guard',lambda *a:calls.append('budget'))
    jobs.queue_preflight(root)
    assert jobs.execute_perception_job(root,project=tmp_path)==0
    quote,pin=jobs._quote(root)
    payload={'quote_sha256':pin,'consent':{'openai_sampled_images':True,'meta_full_video_and_regions':True},
             'openai_max_usd':quote['openai_max_usd'],'meta_max_usd':quote['meta_max_usd'],'request_id':'b'*32}
    return root,payload,calls


def test_local_preflight_preserves_parent_and_does_not_touch_provider_budgets(prepared):
    root,payload,calls=prepared
    assert calls==['local']
    assert (root/'status.json').read_bytes()==(root/'perception/parent-status.json').read_bytes()
    assert jobs.read_status(root)['status']=='awaiting_consent'
    assert payload['meta_max_usd']==.15 and payload['openai_max_usd']==.3135
    jobs.queue_preflight(root)
    assert calls==['local']


@pytest.mark.parametrize('change',[
    lambda p:p['consent'].update(meta_full_video_and_regions=False),
    lambda p:p['consent'].update(openai_sampled_images=False),
    lambda p:p['consent'].update(meta_full_video_and_regions=1),
    lambda p:p.update(consent={}),lambda p:p.update(consent=True),
    lambda p:p.update(quote_sha256='0'*64),lambda p:p.update(meta_max_usd=5),
    lambda p:p.update(openai_max_usd=float('nan')),lambda p:p.update(request_id='../unsafe'),
    lambda p:p.update(extra='unexpected'),
])
def test_invalid_or_cross_provider_consent_never_reserves_or_queues(prepared,change):
    root,payload,calls=prepared;change(payload)
    with pytest.raises(ValueError):jobs.queue_prepare(root,payload,project=root.parent)
    assert calls==['local'] and not (root/'perception/request.json').exists()


def test_duplicate_paid_action_is_idempotent_and_conflicting_action_is_rejected(prepared):
    root,payload,calls=prepared
    jobs.queue_prepare(root,payload,project=root.parent)
    before=(root/'perception/request.json').read_bytes()
    jobs.queue_prepare(root,deepcopy(payload),project=root.parent)
    assert (root/'perception/request.json').read_bytes()==before and calls==['local','budget']
    payload['request_id']='c'*32
    with pytest.raises(ValueError,match='different immutable'):jobs.queue_prepare(root,payload,project=root.parent)


def test_changed_preflight_or_analysis_contract_cannot_authorize_new_request(prepared,monkeypatch):
    root,payload,calls=prepared
    monkeypatch.setattr(jobs,'_contract',lambda:'changed')
    with pytest.raises(ValueError,match='stale quote'):jobs.queue_prepare(root,payload,project=root.parent)
    assert calls==['local']


def test_existing_equipment_work_stops_before_any_paid_action(prepared):
    root,payload,calls=prepared;(root/'equipment').mkdir()
    with pytest.raises(ValueError,match='Existing equipment'):jobs.queue_prepare(root,payload,project=root.parent)
    assert calls==['local'] and not (root/'perception/request.json').exists()


def test_no_credentials_or_budget_is_not_silently_bypassed(prepared,monkeypatch):
    root,payload,calls=prepared
    def denied(*args):raise ValueError('Insufficient budget')
    monkeypatch.setattr(jobs,'_budget_guard',denied)
    with pytest.raises(ValueError,match='budget'):jobs.queue_prepare(root,payload,project=root.parent)
    assert not (root/'perception/request.json').exists()


def install_stages(monkeypatch,calls,*,fail_analysis=False,cancel_after_analysis=False):
    import weft.portal_pipeline as pipeline
    import weft.exercise_perception_pipeline as bridge
    import weft.portal_equipment_candidates as equipment
    def sample(video,out):
        calls.append('sample');out.mkdir();return {'source_sha256':digest(video)}
    def analyze(frames,out,**kwargs):
        calls.append(('analysis',kwargs['identity']))
        if fail_analysis:raise ValueError('Provider transport fixture failed; unknown charge preserved')
        write_json(out/'receipt.json',{'fixture':True})
        if cancel_after_analysis:(out.parent/'cancel-request.json').write_text('{}')
    def perceive(video,analysis,out,**kwargs):
        assert kwargs['execute_tracking'] and kwargs['max_meta_cost_usd']==.15
        assert kwargs['preflight']==out.parent/'preflight'
        calls.append('tracking');kwargs['progress']({'stage':'tracking','message':'Fixture tracking'})
        out.mkdir();return {'status':'evidence_ready','assembly_inputs_ready':True}
    def publish(root,bridge,**kwargs):
        calls.append('descriptor');write_json(root/'perception/prepared-source.json',{'fixture':True})
    monkeypatch.setattr(pipeline,'sample_video',sample)
    monkeypatch.setattr(pipeline,'analyze_frames',analyze)
    monkeypatch.setattr(bridge,'prepare_exercise_perception',perceive)
    monkeypatch.setattr(equipment,'publish_perception_source',publish)


def test_one_consented_flow_queues_the_existing_constructor_without_changing_upload(prepared,monkeypatch):
    root,payload,calls=prepared;install_stages(monkeypatch,calls)
    before=(root/'status.json').read_bytes();jobs.queue_prepare(root,payload,project=root.parent)
    assert jobs.execute_perception_job(root,project=root.parent)==0
    assert jobs.read_status(root)['status']=='evidence_ready'
    assert json.loads((root/'equipment/status.json').read_bytes())['status']=='queued'
    request=json.loads((root/'equipment/request.json').read_bytes())
    assert set(request)=={'profile','source_kind','source_sha256','descriptor_sha256'}
    assert request['source_kind']=='perception' and request['descriptor_sha256']==digest(root/'perception/prepared-source.json')
    assert (root/'status.json').read_bytes()==before==(root/'equipment/parent-status.json').read_bytes()
    count=len(calls);assert jobs.execute_perception_job(root,project=root.parent)==0;assert len(calls)==count
    assert not (root.parent/'equipment-sources.json').exists()


def test_failed_analysis_is_preserved_and_never_retried_even_after_lifecycle_reset(prepared,monkeypatch):
    root,payload,calls=prepared;install_stages(monkeypatch,calls,fail_analysis=True)
    jobs.queue_prepare(root,payload,project=root.parent)
    assert jobs.execute_perception_job(root,project=root.parent)==1
    assert not (root/'equipment').exists() and 'tracking' not in calls
    assert jobs.read_status(root)['status']=='failed'
    write_json(root/'perception/status.json',{'status':'queued'})
    assert jobs.execute_perception_job(root,project=root.parent)==1
    assert sum(type(c) is tuple and c[0]=='analysis' for c in calls)==1


def test_cancel_after_analysis_cannot_send_full_video(prepared,monkeypatch):
    root,payload,calls=prepared;install_stages(monkeypatch,calls,cancel_after_analysis=True)
    jobs.queue_prepare(root,payload,project=root.parent)
    assert jobs.execute_perception_job(root,project=root.parent)==1
    assert jobs.read_status(root)['status']=='cancelled' and 'tracking' not in calls
    assert (root/'perception/analysis/receipt.json').exists()


def test_unsupported_format_stops_before_local_model_or_any_provider(prepared):
    root,_,calls=prepared
    other=root.parent/'unsupported';other.mkdir()
    (other/'source.webm').write_bytes(b'fixture')
    write_json(other/'job.json',{'id':'c'*32,'source_file':'source.webm','source_sha256':digest(other/'source.webm')})
    write_json(other/'status.json',{'status':'needs_input','route':'exercise'})
    jobs.queue_preflight(other)
    assert jobs.execute_perception_job(other,project=root.parent)==1
    assert jobs.read_status(other)['status']=='unsupported' and calls==['local']


def test_local_integrity_failure_is_not_misreported_as_unsupported_video(prepared,monkeypatch):
    root,_,calls=prepared
    import weft.exercise_perception_pipeline as bridge
    def invalid(*args,**kwargs):raise ValueError('Body source artifact bytes differ')
    monkeypatch.setattr(bridge,'prepare_exercise_perception_preflight',invalid)
    other=root.parent/'broken';other.mkdir();(other/'source.mp4').write_bytes(b'fixture')
    write_json(other/'job.json',{'id':'d'*32,'source_file':'source.mp4','source_sha256':digest(other/'source.mp4')})
    write_json(other/'status.json',{'status':'needs_input','route':'exercise'})
    jobs.queue_preflight(other)
    assert jobs.execute_perception_job(other,project=root.parent)==1
    assert jobs.read_status(other)['status']=='failed' and calls==['local']
