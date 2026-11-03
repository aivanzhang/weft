"""Perception review custody; analytic fixtures never assert measured physical accuracy."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import json

import pytest

from weft import portal_perception_review as module
from weft.io import digest, write_json


def seal_metadata(root, body, lab, *, source_sha):
    from weft.portal_equipment_candidates import PROFILE
    from weft.io import canonical_hash
    job={'id':'a'*32,'kind':'video','source_sha256':source_sha,'source_file':'source.mp4'}
    write_json(root/'job.json',job)
    preflight=root/'perception/preflight'
    write_json(preflight/'manifest.json',{'children':{'body_source':digest(body/'manifest.json')}})
    external={'preflight_manifest_sha256':digest(preflight/'manifest.json')}
    (root/'perception/bridge').mkdir(parents=True,exist_ok=True)
    write_json(root/'perception/bridge/preflight-descriptor.json',external)
    descriptor={'source_sha256':source_sha,'job_sha256':digest(root/'job.json'),
        'base':{'path':'perception/preflight/body_seed'},
        'provenance':{'preflight_descriptor_sha256':digest(root/'perception/bridge/preflight-descriptor.json')}}
    write_json(root/'perception/prepared-source.json',descriptor)
    request={'profile':PROFILE,'source_kind':'perception','source_sha256':source_sha,
             'descriptor_sha256':digest(root/'perception/prepared-source.json')}
    write_json(root/'equipment/request.json',request)
    candidate={'id':'alternative-1','status':'ready','selectable':True,'lab':lab.relative_to(root).as_posix(),
        'source_sha256':source_sha,'lab_config_sha256':digest(lab/'lab-config.json'),
        'manifest_sha256':digest(lab/'run/manifest.json'),
        'assignment':{'body_correspondence':'sam-2','rigid_proxies':['sam-1']}}
    report={'profile':PROFILE,'schema_version':1,'source_sha256':source_sha,'request_sha256':digest(root/'equipment/request.json'),
        'source_validation':{'descriptor_sha256':request['descriptor_sha256']},
        'candidates':[candidate,{'id':'alternative-2','status':'rejected','selectable':False,'reasons':['Unchanged geometry refusal']}]}
    write_json(root/'equipment/candidates.json',report)
    write_json(root/'equipment/status.json',{'status':'complete','report_sha256':digest(root/'equipment/candidates.json')})
    write_json(root/'equipment/selection.json',{'profile':PROFILE,'candidate_id':candidate['id'],'lab':candidate['lab'],
        'manifest_sha256':candidate['manifest_sha256'],'report_sha256':digest(root/'equipment/candidates.json')})
    write_json(root/'status.json',{'status':'ready','route':'exercise','equipment_candidate_id':candidate['id'],'lab':candidate['lab']})
    return candidate


@pytest.fixture
def selected(tmp_path,monkeypatch):
    # This fixture isolates metadata admission. Full body joins have independent real-fixture coverage.
    root=tmp_path/'job';body=root/'perception/preflight/body_source';body.mkdir(parents=True)
    write_json(body/'manifest.json',{'profile':'exercise-body-source-v1','files':{}})
    lab=root/'equipment/run/candidates/alternative-1/lab';(lab/'run').mkdir(parents=True)
    write_json(lab/'lab-config.json',{'fixture':True});write_json(lab/'run/manifest.json',{'files':{}})
    seal_metadata(root,body,lab,source_sha='b'*64)
    return root,body,lab


def test_summary_is_metadata_only_and_keeps_rejected_world(selected,monkeypatch):
    import weft.portal_equipment_candidates as equipment
    root,body,_=selected
    before={p:p.read_bytes() for p in root.rglob('*') if p.is_file()}
    monkeypatch.setattr(equipment,'validate_job_source',lambda *a,**k:pytest.fail('No heavy admission on request thread'))
    result=module.attachment_summary(root)
    assert result['available'] and result['status']=='pending_action_time_validation' and not result['repair_qualified']
    assert before=={p:p.read_bytes() for p in before}


@pytest.mark.parametrize('change',['selection','rejected','report','descriptor','body','lab','source','catalog','link','missing_body'])
def test_pending_eligibility_rejects_changed_or_wrong_world(selected,change,tmp_path):
    root,body,lab=selected
    paths={'selection':'equipment/selection.json','report':'equipment/candidates.json','descriptor':'perception/prepared-source.json',
           'body':'perception/preflight/body_source/manifest.json','lab':str((lab/'run/manifest.json').relative_to(root)),
           'source':'job.json','catalog':'equipment/request.json'}
    if change in paths:
        (root/paths[change]).write_bytes(b'{}')
    elif change=='rejected':
        status=json.loads((root/'status.json').read_bytes());status['equipment_candidate_id']='alternative-2';write_json(root/'status.json',status)
    elif change=='missing_body':(body/'manifest.json').unlink()
    else:
        path=body/'manifest.json';saved=path.read_bytes();path.unlink();outside=tmp_path/'outside';outside.write_bytes(saved);path.symlink_to(outside)
    assert not module.attachment_summary(root)['available']


def test_pending_guard_allows_review_not_checks_repairs(selected):
    from weft.portal import Portal
    root,_,_=selected
    server=SimpleNamespace(jobs={'a'*32:json.loads((root/'job.json').read_bytes())},_folder=lambda _:root,
        _status=lambda _:json.loads((root/'status.json').read_bytes()))
    Portal._require_review_body(server,'a'*32)
    for action in ('checks','repairs'):
        with pytest.raises(ValueError,match='review only'):Portal._require_review_body(server,'a'*32,action=action)


def test_failed_full_source_admission_cannot_send_muse(selected,monkeypatch):
    import weft.portal_equipment_candidates as equipment
    import weft.portal_review as review
    root,_,_=selected;(root/'muse').mkdir()
    def reject(*a,**k):raise ValueError('Changed original perception source')
    monkeypatch.setattr(equipment,'validate_job_source',reject)
    monkeypatch.setattr(review,'advise_run',lambda *a,**k:pytest.fail('No provider call before admission'))
    assert review.execute_review(root,project=root.parent)==1
    assert not (root/module.RECEIPT).exists()
    assert json.loads((root/'muse/status.json').read_bytes())['error_type']=='ValueError'


def prepare_analytic_perception(root, staging, monkeypatch):
    """Real sealed body/seed + synthetic no-call detector receipts; no model inference."""
    import shutil
    import cv2
    import numpy as np
    from test_exercise_perception_pipeline import _fixture_analysis
    from test_exercise_body_source import pose_run,write_pose_files,seal as seal_body
    from test_segmentation_evidence import returned,seal
    from weft import exercise_perception_pipeline as bridge_module
    from weft import exercise_body_source as bodies
    from weft import portal_equipment_candidates as equipment
    from weft.segmentation_observations import export_tracking_evidence
    staging.mkdir();video=staging/'original.mp4'
    writer=cv2.VideoWriter(str(video),cv2.VideoWriter_fourcc(*'mp4v'),30,(640,480));assert writer.isOpened()
    for _ in range(63):writer.write(np.full((480,640,3),180,np.uint8))
    writer.release()
    poses=pose_run(staging/'poses');(poses/'source.webm').write_bytes(video.read_bytes())
    write_pose_files(poses,dropped=())
    (poses/'source.webm').rename(poses/'source.mp4')
    write_json(poses/'source.json',{'artifact':'source.mp4','sha256':digest(video),'rights':'Analytical fixture, no physical person'})
    seal_body(poses)
    body=staging/'body';bodies.prepare_exercise_body_source_from_evidence(poses,body)
    root.mkdir();(root/'source.mp4').write_bytes(video.read_bytes())
    write_json(root/'job.json',{'id':'a'*32,'kind':'video','source_file':'source.mp4','source_sha256':digest(video)})
    write_json(root/'status.json',{'status':'needs_input','route':'exercise','events':[]})
    model=staging/'model.task';model.write_bytes(b'Fixture marker; no model executed')
    with monkeypatch.context() as patch:
        patch.setattr(bodies,'prepare_exercise_body_source',lambda video,out,**k:shutil.copytree(body,out))
        bridge_module.prepare_exercise_perception_preflight(root/'source.mp4',root/'perception/preflight',pose_model=model)
    analysis=_fixture_analysis(root/'source.mp4',root/'analysis')
    plan=json.loads((analysis/'plan.json').read_bytes())
    # A floor object overlaps the person's lower image region, never the torso landmarks.
    plan['objects'][0]['box_xyxy']=[.3,.4,.4,.71]
    write_json(analysis/'plan.json',plan)
    response=json.loads((analysis/'response.json').read_bytes())
    response['output'][0]['content'][0]['text']=json.dumps(plan);write_json(analysis/'response.json',response)
    bridge=root/'perception/bridge'
    bridge_module.prepare_exercise_perception(root/'source.mp4',analysis,bridge,preflight=root/'perception/preflight')
    tracking=bridge/'tracking';(tracking/'call').mkdir()
    prompts=json.loads((tracking/'report.json').read_bytes())['objects'];frames=[]
    for i in range(63):
        rows=[]
        for prompt in prompts:
            x,y,w,h=prompt['roi']
            row=returned(i,prompt['object_id'],[x,y,x+w,y+h]);row['source_dimensions']=[640,480];rows.append(row)
        frames.append({'frame':i,'objects':rows})
    write_json(tracking/'call/call.json',{'model':'sam-3.1','video_sha256':digest(video),'status':'completed',
        'initialization':prompts,'response_id':'analytic-no-call','request_sha256':digest(tracking/'request.json')})
    write_json(tracking/'call/segmentation.json',{'status':'completed','object_ids':[str(p['object_id']) for p in prompts],
        'initialization':prompts,'response_id':'analytic-no-call','frames':frames,'raw_text':'Analytic response fixture; no provider call'})
    seal(tracking);export_tracking_evidence(tracking,bridge/'equipment_evidence')
    result=json.loads((bridge/'result.json').read_bytes());result.update(status='evidence_ready',assembly_inputs_ready=True,
        base_run=str(root/'perception/preflight/body_seed'),evidence_run=str(bridge/'equipment_evidence'),source_mappings=[],
        tracking_manifest_sha256=digest(tracking/'manifest.json'));write_json(bridge/'result.json',result)
    manifest=json.loads((bridge/'manifest.json').read_bytes());manifest['status']='evidence_ready'
    manifest['files']['result.json']=digest(bridge/'result.json')
    manifest['children']={n:digest(bridge/n/'manifest.json') for n in ('tracking','equipment_evidence')};write_json(bridge/'manifest.json',manifest)
    equipment.publish_perception_source(root,bridge,project=staging.parent)
    work=root/'equipment';work.mkdir();(work/'parent-status.json').write_bytes((root/'status.json').read_bytes())
    write_json(work/'request.json',{'profile':equipment.PROFILE,'source_kind':'perception','source_sha256':digest(video),
        'descriptor_sha256':digest(root/'perception/prepared-source.json')})
    write_json(work/'status.json',{'status':'queued'})
    return root


def test_actual_generated_body_and_conditional_world_join_before_muse(tmp_path,monkeypatch):
    import time
    from weft import portal_equipment_candidates as equipment
    from weft.portal_review import review_input
    from weft import scene_advisor
    root=prepare_analytic_perception(tmp_path/'job',tmp_path/'fixtures',monkeypatch)
    assert equipment.execute_equipment_job(root,project=tmp_path)==0
    report=equipment.read_report(root,equipment.read_status(root))
    ready=[r for r in report['candidates'] if r['selectable']]
    assert ready,report
    equipment.select_candidate(root,ready[0]['id'])
    before={p:p.read_bytes() for parent in (root/'perception',root/'equipment') for p in parent.rglob('*') if p.is_file()}
    started=time.monotonic();source,binding,world=review_input(root,project=tmp_path)
    assert time.monotonic()-started<30
    attachment=binding['perception_attachment']
    assert attachment['profile']==module.PROFILE and not attachment['repair_qualified']
    assert attachment['perception']['candidate']['id']==ready[0]['id']
    assert attachment['human_sha256']==__import__('weft.io',fromlist=['canonical_hash']).canonical_hash(json.loads((source/'human.json').read_bytes()))
    assert world['physical_validation']=='unverified' and world['objects']
    # Prepare the ordinary Muse payload locally, without transport or reservation.
    context=scene_advisor.prepare_context(source,scene_context=world)
    assert context['simulation_world']==world
    assert before=={p:p.read_bytes() for p in before}
    assert (root/module.RECEIPT).is_file()
    assert review_input(root,project=tmp_path)[1]==binding
    # Source tampering must fail before any Muse transport, preserving the original receipt.
    evidence=source/'evidence.json';saved=evidence.read_bytes();evidence.write_bytes(b'{}')
    import weft.portal_review as review
    monkeypatch.setattr(review,'advise_run',lambda *a,**k:pytest.fail('Changed body cannot reach Muse'))
    assert review.execute_review(root,project=tmp_path)==1
    evidence.write_bytes(saved)


def test_attachment_shares_fifteen_second_deadline_before_body_read(selected,monkeypatch):
    import time
    import weft.portal_equipment_candidates as equipment
    import weft.portal_evidence_attachment as body
    root,_,_=selected
    def validate(*args,deadline,**kwargs):
        assert 0<deadline-time.monotonic()<=15
        raise TimeoutError('Shared bounded deadline')
    monkeypatch.setattr(equipment,'validate_job_source',validate)
    monkeypatch.setattr(body,'_body_files',lambda *a,**k:pytest.fail('No body read after expired source admission'))
    with pytest.raises(TimeoutError,match='Shared bounded deadline'):module.validate_attachment(root)
    assert not (root/module.RECEIPT).exists()


def test_conditional_scope_reaches_existing_muse_request_without_schema_change(selected,monkeypatch):
    from weft import portal_review as review
    from weft.scene_advisor import DEFAULT_QUESTIONS
    root,_,_=selected;(root/'muse').mkdir()
    questions=[*DEFAULT_QUESTIONS,module.CONDITIONAL_QUESTION]
    binding={'review_questions':questions,'review_scope':'coupled_world_assumptions'}
    monkeypatch.setattr(review,'review_input',lambda *a,**k:(root,binding,{'fixture':'construction context'}))
    calls=[]
    def advise(*args,questions,**kwargs):
        calls.append(questions);assert kwargs['scene_context']=={'fixture':'construction context'}
        return {'cost_usd':0}
    monkeypatch.setattr(review,'advise_run',advise)
    assert review.execute_review(root,project=root.parent)==0 and calls==[questions]
    assert len(questions)==4 and all(len(q)<=500 for q in questions)
