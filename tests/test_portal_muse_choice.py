"""Explicit model choice is not inferred from unrelated physical uncertainty."""
from copy import deepcopy
from types import SimpleNamespace
import json
import base64
from hashlib import sha256

import pytest

from test_portal_repairs import inputs
from test_portal_repair_jobs import job, completed_run
from weft import portal_repairs as repairs, portal_repair_jobs as jobs, scene_advisor as advisor
from weft.io import canonical_hash, write_json


def choice_inputs(inputs):
    scene, task, advice, context = deepcopy(inputs)
    context['questions'] = list(repairs.CHOICE_QUESTIONS)
    hypothesis = advice['hypotheses'][0]
    hypothesis['concept'] = 'control'
    hypothesis['required_tests'] = ['weft:test:feedback-2', 'Measure physical uncertainty separately']
    return scene, task, advice, context


def test_only_explicit_model_choice_is_tested_with_original_scorer(inputs, tmp_path):
    values = choice_inputs(inputs)
    plan = repairs.prepare_repair_plan(*values)
    assert [r['id'] for r in plan['candidates']] == ['feedback-2']
    assert plan['protocol']['maximum_control_steps'] == 12
    report = repairs.run_repair_plan(*values, plan, tmp_path/'comparison', max_seconds=10)
    assert [r['id'] for r in report['candidates']] == ['feedback-2']
    assert report['model_choice']['selected_ids'] == ['feedback-2']
    assert report['applied'] is False and report['control_steps'] == 12
    assert (tmp_path/'comparison/baseline-trial-1.json').exists()
    assert (tmp_path/'comparison/feedback-2-trial-2.json').exists()
    assert not (tmp_path/'comparison/feedback-1-trial-1.json').exists()


@pytest.mark.parametrize('change', ['no_token', 'empty', 'unknown', 'articulation', 'bad_id', 'duplicate', 'foreign_evidence'])
def test_abstention_and_invalid_choices_never_run(inputs, change):
    values = choice_inputs(inputs)
    h = values[2]['hypotheses'][0]
    if change == 'no_token': h['required_tests'] = ['Obtain multiple views']
    elif change == 'empty': values[2]['hypotheses'] = []
    elif change == 'unknown': h['status'] = 'unknown'
    elif change == 'articulation': h['concept'] = 'articulation'
    elif change == 'bad_id': h['required_tests'] = ['weft:test:unbounded']
    elif change == 'duplicate': h['required_tests'] = ['weft:test:feedback-2'] * 2
    else: h['evidence_ids'] = ['foreign']
    with pytest.raises(ValueError): repairs.prepare_repair_plan(*values)


def review_fixture(tmp_path, inputs, monkeypatch):
    scene, task, advice, context = choice_inputs(inputs)
    root, review = tmp_path/'project', tmp_path/'proposal'
    root.mkdir(); (review/'advice').mkdir(parents=True)
    binding = {'run':'lab/run', 'run_manifest_sha256':'f'*64, 'source_sha256':context['source_sha256'], 'construction':{}}
    monkeypatch.setattr('weft.portal_checks.inputs', lambda _: (scene, task, advice, binding))
    monkeypatch.setattr('weft.portal_review.review_input', lambda *_a, **_k: (root, {}, {}))
    monkeypatch.setattr(advisor, 'prepare_context', lambda *_a, **_k: context)
    monkeypatch.setattr('weft.shared_world_packaging.collect_shared_source_files',
                        lambda *_a, **_k: SimpleNamespace(files={},report={}))
    write_json(review/'binding.json', {'profile':repairs.CHOICE_PROFILE,'review_binding':binding,
        'scene_sha256':canonical_hash(scene),'task_sha256':canonical_hash(task)})
    request = {'model':advisor.MODEL,'messages':[{'role':'developer','content':advisor.SYSTEM},
        {'role':'user','content':[{'type':'text','text':json.dumps(context,sort_keys=True,allow_nan=False)}]}],
        'response_format':{'type':'json_schema','json_schema':{'name':'WeftSceneAdvice','schema':advisor.PROPOSAL_SCHEMA}},
        'max_completion_tokens':advisor.OUTPUT_TOKEN_LIMIT,'reasoning_effort':'low'}
    row=context['observations'][0]
    request['messages'][1]['content'].extend([{'type':'text','text':f"Construction frame {row['frame']}; evidence {row['id']}"},
        {'type':'image_url','image_url':{'url':'data:image/jpeg;base64,'+base64.b64encode(b'fixture').decode(),'detail':'low'}}])
    documents = {'context.json':context,'request.json':request,
        'response.json':{'model':advisor.MODEL,'choices':[{'finish_reason':'stop','message':{'content':json.dumps(advice)}}]},
        'hypotheses.json':{'source':'model_proposal','accepted':False,'advice':advice},
        'advisor-call.json':{'status':'proposed','schema_validated':True,'request_count':1,'retries':0,
            'live_access_verified':True,'model':advisor.MODEL,'model_returned':advisor.MODEL,
            'request_sha256':canonical_hash(request),'source_sha256':binding['source_sha256'],
            'simulation_world_sha256':canonical_hash({}), 'frame_ids':[row['id']],
            'frame_jpeg_sha256':[sha256(b'fixture').hexdigest()]}}
    for name, value in documents.items(): write_json(review/'advice'/name,value)
    return root,review,documents


@pytest.mark.parametrize('change', ['response', 'context', 'receipt', 'request', 'binding', 'image', 'caption'])
def test_provider_and_original_world_joins_before_comparison(tmp_path, inputs, monkeypatch, change):
    root, review, docs = review_fixture(tmp_path,inputs,monkeypatch)
    if change == 'response': docs['response.json']['choices'][0]['message']['content'] = '{}'
    elif change == 'context': docs['context.json'] = {**docs['context.json'],'questions':['other']}
    elif change == 'receipt': docs['advisor-call.json']['request_count'] = 0
    elif change == 'request': docs['request.json']['max_completion_tokens'] = 90000
    elif change == 'image': docs['advisor-call.json']['frame_jpeg_sha256']=['0'*64]
    elif change == 'caption':
        docs['request.json']['messages'][1]['content'][1]['text']='foreign frame'
        docs['advisor-call.json']['request_sha256']=canonical_hash(docs['request.json'])
    else:
        d=json.loads((review/'binding.json').read_bytes());d['task_sha256']='0'*64;write_json(review/'binding.json',d)
    for name,value in docs.items(): write_json(review/'advice'/name,value)
    monkeypatch.setattr(repairs,'run_repair_plan',lambda *_a,**_k:pytest.fail('Invalid provider/source joins cannot simulate'))
    with pytest.raises(ValueError): repairs.run_controller_review_repairs(root,review,tmp_path/'comparison')


def test_verified_provider_choice_enters_existing_comparison(tmp_path, inputs, monkeypatch):
    root,review,_ = review_fixture(tmp_path,inputs,monkeypatch)
    calls=[]
    def run(scene,task,advice,context,plan,out,**kwargs):
        calls.append(plan);kwargs['_before_seal']();return {'checked':True}
    monkeypatch.setattr(repairs,'run_repair_plan',run)
    assert repairs.run_controller_review_repairs(root,review,tmp_path/'comparison') == {'checked':True}
    assert calls[0]['model_choice']['selected_ids'] == ['feedback-2']


def test_paid_choice_queue_is_explicit_single_request_and_separate_advice(tmp_path,monkeypatch):
    root=job(tmp_path/'project')
    write_json(root/'muse/repairs/request.json',{'schema_version':1,'profile':repairs.CHOICE_PROFILE})
    calls=[]
    def request(source,out,**kwargs):
        assert kwargs=={'project':tmp_path,'execute':True};calls.append('request');out.mkdir()
    def run(source,review,out,**kwargs):
        assert review==root/'muse/repairs/proposal';calls.append('test');return completed_run(out)
    monkeypatch.setattr(repairs,'request_controller_review',request)
    monkeypatch.setattr(repairs,'run_controller_review_repairs',run)
    assert jobs.execute_repair_job(root,project=tmp_path)==0
    assert jobs.execute_repair_job(root,project=tmp_path)==0
    assert calls==['request','test']


def test_valid_abstention_preserved_no_physics_no_retry(tmp_path,monkeypatch):
    root=job(tmp_path/'project')
    write_json(root/'muse/repairs/request.json',{'schema_version':1,'profile':repairs.CHOICE_PROFILE})
    calls=[]
    monkeypatch.setattr(repairs,'request_controller_review',lambda *_a,**_k:calls.append('request'))
    def abstain(*_a,**_k): raise repairs.MuseAbstained('No supported option')
    monkeypatch.setattr(repairs,'run_controller_review_repairs',abstain)
    assert jobs.execute_repair_job(root,project=tmp_path)==1
    assert json.loads((root/'muse/repairs/status.json').read_bytes())['status']=='abstained'
    assert jobs.execute_repair_job(root,project=tmp_path)==1 and calls==['request']
    assert not (root/'muse/repairs/run').exists()


def test_saved_body_choice_available_without_enabling_legacy(tmp_path):
    root=job(tmp_path/'project',queue=False)
    write_json(root/'job.json',{'kind':'saved_body'})
    assert not jobs.availability(root)['available']
    report=jobs.availability(root,model_choice=True)
    assert report['available'] and report['requires_new_ai_request'] and report['estimated_external_api_usd']>0


@pytest.mark.parametrize('marker',[{}, {'schema_version':1,'profile':'unknown'},
                                  {'schema_version':1,'profile':repairs.CHOICE_PROFILE,'extra':True}])
def test_key_forwarding_marker_closed(tmp_path,marker):
    root=job(tmp_path/'project');write_json(root/'muse/repairs/request.json',marker)
    with pytest.raises(ValueError): jobs.choice_requested(root)


def test_no_call_helper_uses_real_bounded_context_validation(tmp_path, inputs, monkeypatch):
    scene,task,advice,_ = inputs
    source=tmp_path/'source';source.mkdir()
    source_hash=scene['source_binding']['source_sha256']
    write_json(source/'evidence.json',{'source_sha256':source_hash,'reference':[
        {'evidence_id':source_hash+':frame:0','frame':0,'time_s':0}]})
    binding={'run':'lab/run','run_manifest_sha256':'f'*64,'construction':{}}
    monkeypatch.setattr('weft.portal_checks.inputs',lambda _: (scene,task,advice,binding))
    monkeypatch.setattr('weft.portal_review.review_input',lambda *_a,**_k:(source,{},None))
    monkeypatch.setattr('weft.shared_world_packaging.collect_shared_source_files',lambda *_a,**_k:SimpleNamespace(report={}))
    monkeypatch.setattr(advisor,'_frames',lambda *_:[])
    monkeypatch.setattr(advisor,'_open',lambda *_:pytest.fail('No-call preparation cannot contact provider'))
    result=repairs.request_controller_review(tmp_path/'project',tmp_path/'choice',project=tmp_path)
    assert result['status']=='prepared' and result['request_count']==0
    context=json.loads((tmp_path/'choice/advice/context.json').read_bytes())
    assert context['questions']==list(repairs.CHOICE_QUESTIONS)
    assert context==advisor.prepare_context(source,list(repairs.CHOICE_QUESTIONS))
