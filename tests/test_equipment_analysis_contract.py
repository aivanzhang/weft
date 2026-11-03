"""A clearer source image changes the prompt, never source indices or retrospective evidence."""
from copy import deepcopy
import base64
import json
from pathlib import Path
import time

import pytest

from test_exercise_perception_pipeline import inputs,local_preflight_source,_rehash_analysis
from test_portal_perception_jobs import prepared,install_stages
from weft import portal_pipeline as pipeline,exercise_perception_pipeline as bridge,portal_perception_jobs as jobs
from weft.equipment_analysis_contract import LEGACY_PROFILE,SAMPLED_PROFILE,analysis_contract
from weft.io import canonical_hash,digest,write_json


def make_sampled(analysis,frame=None):
    frames=json.loads((analysis/'frames.json').read_bytes())
    selected=frames['frames'][2]['frame'] if frame is None else frame
    plan=json.loads((analysis/'plan.json').read_bytes());plan['objects'][0]['source_frame']=selected
    write_json(analysis/'plan.json',plan)
    request=json.loads((analysis/'request.json').read_bytes());contract=analysis_contract(SAMPLED_PROFILE)
    request.update(instructions=contract['instructions'],reasoning={'effort':'none'},
        text={'format':{'type':'json_schema','name':contract['format_name'],'strict':True,'schema':contract['schema']}})
    write_json(analysis/'request.json',request)
    response=json.loads((analysis/'response.json').read_bytes());response['output'][0]['content'][0]['text']=json.dumps(plan)
    write_json(analysis/'response.json',response);_rehash_analysis(analysis)
    return frames,plan,selected


@pytest.mark.parametrize('selected',[0,6,15])
def test_source_selected_frame_and_image_hash_are_retained(inputs,selected):
    _,analysis,_=inputs;frames,plan,_=make_sampled(analysis,selected)
    proposal,files=bridge.read_equipment_analysis(analysis)
    assert proposal['boxes']==[[1,selected,19,40,23,31]]
    chosen=next(r for r in frames['frames'] if r['frame']==selected)
    assert proposal['selected_samples']==[{'object_id':1,'source_frame':selected,'artifact':chosen['artifact'],'sha256':chosen['sha256']}]
    assert proposal['analysis_profile']==SAMPLED_PROFILE
    assert len([n for n in files if n.endswith('.jpg')])==6


@pytest.mark.parametrize('frame',[1,16,-1,True,6.0,500])
def test_frame_choice_must_be_integer_member_of_exact_samples(inputs,frame):
    _,analysis,_=inputs;make_sampled(analysis,frame)
    with pytest.raises(ValueError):bridge.read_equipment_analysis(analysis)


@pytest.mark.parametrize('damage',['prompt','schema','format','extra_request','image','response','unknown_frame_artifact'])
def test_coordinated_rehashed_request_cannot_change_closed_contract(inputs,damage):
    _,analysis,_=inputs;frames,plan,_=make_sampled(analysis)
    request=json.loads((analysis/'request.json').read_bytes())
    if damage=='prompt':request['instructions']+=' Invent unobserved prefixes.'
    elif damage=='schema':request['text']['format']['schema']['properties']['objects']['maxItems']=4
    elif damage=='format':request['text']['format']['name']='unreviewed_v3'
    elif damage=='extra_request':request['tools']=[{'type':'web_search'}]
    elif damage=='image':(analysis/frames['frames'][2]['artifact']).write_bytes(b'changed')
    elif damage=='response':
        response=json.loads((analysis/'response.json').read_bytes());plan['objects'][0]['source_frame']=15
        response['output'][0]['content'][0]['text']=json.dumps(plan);write_json(analysis/'response.json',response)
    else:
        row=frames['frames'][2];new='frame-00007.jpg';(analysis/new).write_bytes((analysis/row['artifact']).read_bytes())
        row['artifact']=new;write_json(analysis/'frames.json',frames)
    write_json(analysis/'request.json',request);_rehash_analysis(analysis)
    with pytest.raises(ValueError):bridge.read_equipment_analysis(analysis)


def test_offline_full_video_request_keeps_later_initialization_and_source_bytes(inputs,monkeypatch):
    video,analysis,out=inputs;_,_,selected=make_sampled(analysis)
    import weft.meta_segmentation as meta
    monkeypatch.setattr(meta,'resolve_key',lambda *a,**k:pytest.fail('No credentials for offline fixture'))
    before=video.read_bytes();result=bridge.prepare_exercise_perception(video,analysis,out)
    assert result['api_calls']==0 and result['status']=='prepared_not_sent'
    report=json.loads((out/'tracking/report.json').read_bytes())
    assert report['objects']==[{'object_id':1,'frame':selected,'roi':[19,40,23,31]}]
    payload=json.loads((out/'tracking/request.json').read_bytes())
    encoded=payload['input'][0]['content'][1]['video_url'].split(',',1)[1]
    assert base64.b64decode(encoded)==before==video.read_bytes()
    assert report['video']['frame_count']==16
    manifest=json.loads((out/'manifest.json').read_bytes())
    assert manifest['files']['implementation/equipment_analysis_contract.py']==digest(Path(bridge.__file__).with_name('equipment_analysis_contract.py'))


def test_legacy_standalone_plan_and_reader_remain_first_frame(inputs):
    _,analysis,_=inputs;plan=json.loads((analysis/'plan.json').read_bytes())
    assert pipeline.validate_plan(plan)==plan
    proposal,_=bridge.read_equipment_analysis(analysis)
    assert proposal['boxes']==[[1,0,19,40,23,31]] and 'analysis_profile' not in proposal
    contract=analysis_contract(LEGACY_PROFILE)
    assert contract['instructions']==pipeline.INSTRUCTIONS and contract['schema']==pipeline.SCHEMA
    contract['schema']['properties']['objects']['maxItems']=1
    assert pipeline.SCHEMA['properties']['objects']['maxItems']==4


def test_returned_and_missing_prefix_proposals_keep_original_phase_and_visibility(tmp_path):
    from test_segmentation_evidence import fixture
    from weft.segmentation_evidence import adapt_tracking_run
    from weft.segmentation_observations import adapt_segmentation
    root=fixture(tmp_path/'tracking',boxes=[[[0,0,6,8]],[None],[[0,0,6,8]]],initial=2,ids=(1,))
    evidence=adapt_tracking_run(root);observations=adapt_segmentation(evidence)
    assert [r['phase'] for r in evidence['observations']]==['pre_initialization','pre_initialization','initialization']
    assert [r['proposal_status'] for r in evidence['observations']]==['returned','missing','returned']
    assert [r['visibility'] for r in observations['observations']]==['detected','not_sampled','detected']
    assert evidence['source']['frame_count']==3


def test_reviewed_old_preflight_producer_keeps_full_source_rederivation(local_preflight_source,tmp_path,monkeypatch):
    video,body=local_preflight_source;out=tmp_path/'preflight'
    bridge.prepare_exercise_perception_preflight(video,out,body_source=body)
    producer=Path(__file__).parent/'fixtures/perception/preflight-producer-v1.py.txt'
    assert digest(producer) in bridge.REVIEWED_PREFLIGHT_PRODUCERS
    destination=out/'implementation/exercise_perception_pipeline.py';destination.write_bytes(producer.read_bytes())
    manifest=json.loads((out/'manifest.json').read_bytes());manifest['files']['implementation/exercise_perception_pipeline.py']=digest(destination)
    write_json(out/'manifest.json',manifest)
    before={p:p.read_bytes() for p in out.rglob('*') if p.is_file()}
    import weft.segmentation_lab as segmentation
    monkeypatch.setattr(segmentation,'inspect_video',lambda *a,**k:pytest.fail('Validation must not decode'))
    checked=bridge.validate_perception_preflight(video,out,deadline=time.monotonic()+10)
    assert checked['status']=='local_ready' and all(p.read_bytes()==raw for p,raw in before.items())
    destination.write_bytes(destination.read_bytes()+b'\n# unreviewed change\n')
    manifest['files']['implementation/exercise_perception_pipeline.py']=digest(destination);write_json(out/'manifest.json',manifest)
    with pytest.raises(ValueError,match='explicit compatibility'):
        bridge.validate_perception_preflight(video,out,deadline=time.monotonic()+10)


def test_future_equipment_quote_and_worker_bind_v2_while_legacy_hash_remains(prepared,monkeypatch):
    root,payload,calls=prepared;quote,_=jobs._quote(root)
    assert quote['analysis_profile']==SAMPLED_PROFILE and quote['analysis_contract_sha256']==jobs._contract()
    from weft.openai_access import MODEL,MAX_OUTPUT_TOKENS
    assert jobs._contract(LEGACY_PROFILE)==canonical_hash({'model':MODEL,'instructions':pipeline.INSTRUCTIONS,'schema':pipeline.SCHEMA,'max_output_tokens':MAX_OUTPUT_TOKENS})
    install_stages(monkeypatch,calls)
    original=pipeline.analyze_frames
    def analyze(*args,**kwargs):
        assert kwargs['analysis_profile']==SAMPLED_PROFILE
        return original(*args,**kwargs)
    monkeypatch.setattr(pipeline,'analyze_frames',analyze)
    jobs.queue_prepare(root,payload,project=root.parent)
    assert jobs.execute_perception_job(root,project=root.parent)==0


def test_quote_cannot_relabel_v1_hash_as_v2_or_unknown_profile(prepared):
    root,_,_=prepared;quote,_=jobs._quote(root);job=json.loads((root/'job.json').read_bytes())
    legacy=deepcopy(quote);legacy.pop('analysis_profile');legacy['analysis_contract_sha256']=jobs._contract(LEGACY_PROFILE)
    jobs._validate_quote(root,legacy,job)
    for profile in (SAMPLED_PROFILE,'unreviewed-v3'):
        changed=deepcopy(legacy);changed['analysis_profile']=profile
        with pytest.raises(ValueError):jobs._validate_quote(root,changed,job)


@pytest.mark.parametrize('profile',[None,SAMPLED_PROFILE])
def test_producer_uses_versioned_payload_and_cached_response_without_key_access(inputs,monkeypatch,profile):
    _,analysis,_=inputs;frames=json.loads((analysis/'frames.json').read_bytes())
    plan=json.loads((analysis/'plan.json').read_bytes())
    if profile==SAMPLED_PROFILE:plan['objects'][0]['source_frame']=frames['frames'][-1]['frame']
    response={'id':'cached-fixture','model':pipeline.MODEL,'status':'completed',
        'output':[{'type':'message','content':[{'type':'output_text','text':json.dumps(plan)}]}]}
    seen=[]
    class Cache:
        def __init__(self,*args):pass
        def cached_response(self,request_hash,source_hash):
            seen.append((request_hash,source_hash));return response,{'fixture':True}
    monkeypatch.setattr(pipeline,'OpenAIBudget',Cache)
    monkeypatch.setattr(pipeline,'read_key',lambda *args:pytest.fail('Cache path must not read keys'))
    def forbidden(*args,**kwargs):pytest.fail('No provider transport in this test')
    result=pipeline.analyze_frames(frames,analysis,identity='1'*32,transport=forbidden,analysis_profile=profile)
    assert result==plan and len(seen)==1
    request=json.loads((analysis/'request.json').read_bytes())
    contract=analysis_contract(LEGACY_PROFILE if profile is None else profile)
    assert request['instructions']==contract['instructions']
    assert request['text']['format']=={'type':'json_schema','name':contract['format_name'],'strict':True,'schema':contract['schema']}
    assert request['request_sha256']==seen[0][0]
    assert json.loads((analysis/'receipt.json').read_bytes())['provider_request_sent'] is False
    proposal,_=bridge.read_equipment_analysis(analysis)
    assert proposal['boxes'][0][1]==(15 if profile else 0)


def test_sampled_producer_refuses_unordered_inventory_before_budget_or_keys(inputs,monkeypatch):
    _,analysis,_=inputs;frames=json.loads((analysis/'frames.json').read_bytes());frames['frames'].reverse()
    monkeypatch.setattr(pipeline,'OpenAIBudget',lambda *a:pytest.fail('Malformed inventory cannot access budget'))
    with pytest.raises(ValueError,match='unique ordered'):
        pipeline.analyze_frames(frames,analysis,identity='1'*32,analysis_profile=SAMPLED_PROFILE)
