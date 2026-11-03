"""Prepared equipment queue, source custody, retained alternatives and immutable selection."""
from copy import deepcopy
from io import BytesIO
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from weft import portal_equipment_candidates as module
from weft.io import canonical_hash,digest,write_json
from weft.portal import Portal,PortalHandler


@pytest.fixture
def project(tmp_path,monkeypatch):
    import weft.portal as portal_module
    monkeypatch.setattr(portal_module,'muse_availability',lambda _: {'available':False})
    source=tmp_path/'prepared';source.mkdir()
    for folder in ('base','objects'):
        (source/folder).mkdir();write_json(source/folder/'fixture.json',{'scope':'analytic fixture'})
        write_json(source/folder/'manifest.json',{'files':{'fixture.json':digest(source/folder/'fixture.json')}})
    root=tmp_path/'portal'
    portal=Portal(root,project=tmp_path,start=False)
    job=portal.upload(BytesIO(b'fixture source'),14,name='exercise.webm',mode='exercise',ai=False,request_id='1'*32,
                      exercise_inventory='equipment_or_obstacles',exercise_output='simulation')
    folder=root/job['id']
    write_json(folder/'status.json',{'status':'needs_input','route':'exercise','events':[{'stage':'original','message':'preserve parent history'}]})
    entry={'source_sha256':job['source_sha256'],'base':{'path':'prepared/base','manifest_sha256':digest(source/'base/manifest.json')},
           'equipment':{'path':'prepared/objects','manifest_sha256':digest(source/'objects/manifest.json')},'mappings':[]}
    entry.update(id=canonical_hash(entry)[:16],label='Prepared fixture, not physical evidence')
    catalog={'schema_version':1,'profile':module.CATALOG_PROFILE,'entries':[entry]}
    write_json(root/module.CATALOG,catalog);portal.equipment_catalog=catalog
    def validate(value,**kwargs):
        assert value==entry
        return {'base_run':source/'base','evidence_run':source/'objects','source_mappings':()}, {
            'body_reference_count':3,'source_slots':8,'raw_object_ids':['person-mask','object-mask'],'unresolved_reasons':['conditional_identity'],
            'source_sha256':job['source_sha256'],'scope':module.SCOPE}
    monkeypatch.setattr(module,'validate_source_entry',validate)
    import weft.equipment_conditional_assembly as assembly
    calls=[]
    def assemble(**kwargs):
        calls.append(kwargs)
        out=kwargs['out'];lab=out/'candidate-a/lab';(lab/'run').mkdir(parents=True)
        scene={'source_binding':{'source_sha256':job['source_sha256']}}
        write_json(lab/'run/scene.json',scene)
        write_json(lab/'run/exercise-task.json',{'frozen_gate':.025})
        write_json(lab/'run/manifest.json',{'status':'ready','files':{p.name:digest(p) for p in (lab/'run').iterdir()}})
        write_json(lab/'lab-config.json',{'control_profile':'shared_exercise','learned_policy_available':False,
            'manifest_sha256':digest(lab/'run/manifest.json'),'scene_sha256':canonical_hash(scene),'task_sha256':canonical_hash({'frozen_gate':.025})})
        return {'status':'candidates_ready','candidates':[{'id':'candidate-a','status':'ready','lab_directory':str(lab),
            'reasons':[],'assignment':{'body_correspondence':'person-mask','rigid_proxies':['object-mask']}},
            {'id':'candidate-b','status':'rejected','lab_directory':None,'reasons':['initial_geometry_below_floor'],
             'assignment':{'body_correspondence':'object-mask','rigid_proxies':['person-mask']}}]}
    monkeypatch.setattr(assembly,'prepare_conditional_equipment_scene',assemble)
    import weft.portal_lab_runtime as runtime
    monkeypatch.setattr(runtime,'open_portal_lab',lambda lab,**kwargs:SimpleNamespace(
        summary=lambda:{'scene':json.loads((lab/'run/scene.json').read_bytes()),'learned_policy_available':False},close=lambda:None))
    yield portal,folder,entry,calls
    portal.close()


def test_matching_is_shortlist_worker_qualifies_and_user_selects_once(project):
    portal,folder,entry,calls=project
    before=(folder/'status.json').read_bytes()
    summary=portal.summary(folder.name)
    assert summary['equipment_availability']['available'] and not calls
    portal.prepare_equipment(folder.name,entry['id'])
    assert (folder/'status.json').read_bytes()==before
    assert module.execute_equipment_job(folder,project=portal.project)==0
    assert len(calls)==1 and calls[0]['max_candidates']==3 and calls[0]['max_seconds']==90
    summary=portal.summary(folder.name);report=summary['equipment_candidates']['report']
    assert summary['status']=='needs_input' and summary['sandbox'] is None
    assert [c['selectable'] for c in report['candidates']]==[True,False]
    assert report['candidates'][1]['reasons']==['initial_geometry_below_floor']
    with pytest.raises(ValueError,match='not admitted'):portal.choose_equipment(folder.name,'candidate-b')
    selected=portal.choose_equipment(folder.name,'candidate-a')
    assert selected['status']=='ready' and selected['equipment_candidate_id']=='candidate-a'
    assert (folder/'equipment/parent-status.json').read_bytes()==before
    assert selected['review_unavailable_reason'] and selected['sandbox']
    with pytest.raises(ValueError,match='locked'):portal.choose_equipment(folder.name,'candidate-b')
    assert module.execute_equipment_job(folder,project=portal.project)==0 and len(calls)==1


def test_cancel_restart_and_duplicate_do_not_retry_or_lose_parent(project):
    portal,folder,entry,calls=project
    original=(folder/'status.json').read_bytes()
    portal.prepare_equipment(folder.name,entry['id']);first=(folder/'equipment/request.json').read_bytes()
    portal.prepare_equipment(folder.name,entry['id'])
    assert (folder/'equipment/request.json').read_bytes()==first
    cancelled=portal.cancel(folder.name)
    assert cancelled['equipment_candidates']['status']=='cancelled'
    assert (folder/'status.json').read_bytes()==original
    assert module.execute_equipment_job(folder,project=portal.project)==1 and not calls


@pytest.mark.parametrize('change',['catalog','source','parent'])
def test_action_time_changed_input_rejects_before_assembly(project,change):
    portal,folder,entry,calls=project
    portal.prepare_equipment(folder.name,entry['id'])
    if change=='catalog':write_json(portal.root/module.CATALOG,{'schema_version':1,'profile':module.CATALOG_PROFILE,'entries':[]})
    elif change=='source':(folder/'source.webm').write_bytes(b'changed')
    else:write_json(folder/'status.json',{'status':'failed'})
    assert module.execute_equipment_job(folder,project=portal.project)==1 and not calls
    assert module.read_status(folder)['status']=='failed'


@pytest.mark.parametrize('name',['equipment/candidates.json','equipment/run/candidate-a/lab/lab-config.json'])
def test_inspectable_report_and_retained_candidate_artifacts(project,name):
    portal,folder,entry,_=project
    portal.prepare_equipment(folder.name,entry['id']);module.execute_equipment_job(folder,project=portal.project)
    assert portal.artifact(folder.name,name)==folder/name
    with pytest.raises(ValueError):portal.artifact(folder.name,'equipment/request.json')


def test_selection_detects_modified_lab_and_forbids_opened_runtime_switch(project):
    portal,folder,entry,_=project
    portal.prepare_equipment(folder.name,entry['id']);module.execute_equipment_job(folder,project=portal.project)
    lab=folder/'equipment/run/candidate-a/lab'
    (lab/'lab-config.json').write_bytes(b'{}')
    with pytest.raises(ValueError,match='pinned SHA256'):portal.choose_equipment(folder.name,'candidate-a')
    assert not (folder/'equipment/selection.json').exists()
    portal.labs[folder.name]=SimpleNamespace(close=lambda:None)
    with pytest.raises(ValueError,match='opened sandbox'):portal.choose_equipment(folder.name,'candidate-a')


def test_no_match_never_qualifies_by_filename_or_queues(project):
    portal,folder,entry,calls=project
    catalog=deepcopy(portal.equipment_catalog);catalog['entries'][0]['source_sha256']='f'*64
    assert not module.availability(portal.jobs[folder.name],portal._status(folder.name),catalog)['available']
    with pytest.raises(ValueError,match='matches'):portal.prepare_equipment(folder.name,'f'*16)
    assert not (folder/'equipment').exists() and not calls


@pytest.mark.parametrize('change',['traversal','duplicate','oversize','changed_identity'])
def test_catalog_is_closed_bounded_and_project_relative(project,change):
    portal,_,_,_=project
    catalog=deepcopy(portal.equipment_catalog)
    if change=='traversal':catalog['entries'][0]['base']['path']='../private'
    elif change=='duplicate':catalog['entries']*=2
    elif change=='oversize':catalog['entries']*=13
    else:catalog['entries'][0]['id']='f'*16
    write_json(portal.root/module.CATALOG,catalog)
    with pytest.raises(ValueError):module.read_catalog(portal.root)


def test_authenticated_choice_handler_accepts_only_bounded_identifier():
    calls=[];response=[]
    handler=SimpleNamespace(path='/api/projects/'+'a'*32+'/equipment/select',headers={'Content-Length':'20'},
        rfile=BytesIO(b'{"id":"candidate-a"}'),connection=SimpleNamespace(settimeout=lambda _:None),route_lab=lambda:None,
        authorized=lambda:True,server=SimpleNamespace(portal=SimpleNamespace(choose_equipment=lambda *a:calls.append(a) or {'status':'ready'})),
        reply=lambda value,status=200:response.append((value,status)))
    PortalHandler.do_POST(handler)
    assert calls==[('a'*32,'candidate-a')] and response[-1][1]==200
    handler.authorized=lambda:False;PortalHandler.do_POST(handler);assert len(calls)==1
    handler.authorized=lambda:True;handler.headers={'Content-Length':'300'};PortalHandler.do_POST(handler)
    assert response[-1][1]==400 and len(calls)==1


def test_full_registered_source_validation_recomputes_slots_and_rejects_changed_custody(tmp_path):
    from test_equipment_proposal import packages
    body,equipment,_,base,_=packages(tmp_path)
    entry={'source_sha256':base['source_binding']['source_sha256'],'base':{'path':'body','manifest_sha256':digest(body/'manifest.json')},
           'equipment':{'path':'equipment','manifest_sha256':digest(equipment/'manifest.json')},'mappings':[]}
    entry.update(id=canonical_hash(entry)[:16],label='Analytic source custody fixture')
    inputs,proof=module.validate_source_entry(entry,project=tmp_path)
    assert inputs['base_run']==body and proof['source_slots']==63 and proof['body_reference_count']>0
    raw=next(iter(base['source_binding']['evidence_documents'].values()))
    (body/raw).write_bytes(b'{}')
    with pytest.raises(ValueError):module.validate_source_entry(entry,project=tmp_path)


def test_registration_is_bounded_and_holds_existing_portal_lease(project):
    portal,folder,entry,_=project
    with pytest.raises(ValueError,match='Stop the portal'):
        module.register_equipment_source(portal.root,source_sha256=entry['source_sha256'],base_run=portal.project/'prepared/base',
            evidence_run=portal.project/'prepared/objects',label=entry['label'],project=portal.project)


def test_selected_candidate_disables_muse_repairs_and_supports_even_if_body_exists(project):
    portal,folder,entry,_=project
    portal.prepare_equipment(folder.name,entry['id']);module.execute_equipment_job(folder,project=portal.project)
    portal.choose_equipment(folder.name,'candidate-a')
    (folder/'build/body_source').mkdir(parents=True);(folder/'build/body_source/manifest.json').write_bytes(b'{}')
    assert not portal.summary(folder.name)['support_experiment_available']
    for action in (portal.review,portal.check,portal.repair):
        with pytest.raises(ValueError,match='sealed generated body evidence'):action(folder.name)


def test_interrupted_selection_retries_same_choice_without_overwriting_receipt(project,monkeypatch):
    portal,folder,entry,_=project
    portal.prepare_equipment(folder.name,entry['id']);module.execute_equipment_job(folder,project=portal.project)
    original=module.atomic_json
    def fail(path,value):
        if path==folder/'status.json':raise OSError('Interrupted publication')
        return original(path,value)
    monkeypatch.setattr(module,'atomic_json',fail)
    with pytest.raises(OSError,match='Interrupted'):portal.choose_equipment(folder.name,'candidate-a')
    saved=(folder/'equipment/selection.json').read_bytes()
    monkeypatch.setattr(module,'atomic_json',original)
    assert portal.choose_equipment(folder.name,'candidate-a')['status']=='ready'
    assert (folder/'equipment/selection.json').read_bytes()==saved


def test_actual_shape_of_constructor_reasons_is_rendered_without_dropping_rejection():
    raw={'status':'abstained','candidates':[{'id':'alternative-1','status':'rejected_initial_geometry',
        'assignment':{'body_correspondence':'sam-1','rigid_proxies':['sam-2','sam-3']},
        'reasons':[{'code':'initial_geometry_admission_rejected','error_type':'ValueError','stage':'initial_geometry_admission',
                    'message':"Initial rigid object 'sam-3' exceeds shared-world bounds"}]}]}
    rows=module.candidate_rows(Path('/unused'),raw)
    assert not rows[0]['selectable'] and rows[0]['reasons']==["initial_geometry_admission_rejected: Initial rigid object 'sam-3' exceeds shared-world bounds"]


@pytest.fixture
def rejected_receipts(tmp_path):
    from test_equipment_proposal import packages,seal
    from test_equipment_autoproposal import role_inputs
    from weft.equipment_conditional_assembly import derive_conditional_equipment_configs
    body,equipment,_,base,_=packages(tmp_path)
    documents,kwargs=role_inputs(ambiguous=True)
    (equipment/'scene-observations.json').write_bytes(documents[kwargs['equipment_document_sha256']]);seal(equipment)
    context,construction=derive_conditional_equipment_configs(documents,**kwargs)
    source=tmp_path/'saved';source.mkdir();(source/'configs').mkdir()
    for name,value in [('conditional-context.json',context),('settings.json',{'maximum_candidates':3})]:write_json(source/name,value)
    (source/'construction-evidence.json').write_bytes(construction)
    (source/'inputs/observations').mkdir(parents=True)
    for pin,data in documents.items():(source/f'inputs/observations/{pin}.json').write_bytes(data)
    for name,data in [('base-manifest.json',(body/'manifest.json').read_bytes()),('equipment-manifest.json',(equipment/'manifest.json').read_bytes()),('base-scene.json',(body/'scene.json').read_bytes())]:(source/'inputs'/name).write_bytes(data)
    candidates=[]
    for row in context['candidates']:
        child=source/'candidates'/row['id'];child.mkdir(parents=True)
        result={'status':'unsupported_input','reasons':[{'code':'analytic_rejection','message':'Synthetic receipt fixture, not a measured result'}],'lab_directory':None}
        write_json(child/'result.json',result);seal(child)
        candidates.append({**result,'id':row['id'],'assignment':row['assignment'],'task_comparable':False,'assembly_manifest_sha256':digest(child/'manifest.json')})
        write_json(source/'configs'/(row['id']+'.json'),row['config'])
    write_json(source/'result.json',{'status':'abstained','selected_candidate':None,'training':False,'physics_steps':0,'api_calls':0,'inference_calls':0,'candidates':candidates})
    def seal_outer():
        write_json(source/'manifest.json',{'files':{p.relative_to(source).as_posix():digest(p) for p in source.rglob('*') if p.is_file() and 'candidates' not in p.relative_to(source).parts and p.name!='manifest.json'}})
    seal_outer()
    entry={'source_sha256':base['source_binding']['source_sha256'],'base':{'path':'body','manifest_sha256':digest(body/'manifest.json')},'equipment':{'path':'equipment','manifest_sha256':digest(equipment/'manifest.json')},'mappings':[]}
    entry.update(id=canonical_hash(entry)[:16],label='Analytic sealed rejection fixture')
    return tmp_path,source,entry,seal_outer


def test_saved_rejection_inspection_rederives_all_roles_without_construction(rejected_receipts,monkeypatch):
    project,source,entry,_=rejected_receipts
    import weft.exercise_scene_pipeline as pipeline
    monkeypatch.setattr(pipeline,'prepare_exercise_scene',lambda *a,**k:pytest.fail('No constructor replay'))
    before={p:p.read_bytes() for p in source.rglob('*') if p.is_file()}
    report,files=module.inspect_rejected_assembly(source,entry,expected_manifest_sha256=digest(source/'manifest.json'),project=project)
    assert len(report['candidates'])==2 and all(not c['selectable'] for c in report['candidates'])
    assert report['reuse']['constructor_replayed'] is False and report['reuse']['alternatives_independently_rederived']
    assert 'candidates/alternative-1/result.json' in files
    assert before=={p:p.read_bytes() for p in before}


@pytest.mark.parametrize('mutation',['missing_alternative','context','config','child_result','playable_claim','source_manifest'])
def test_saved_rejection_intake_rejects_coordinated_outer_rehash(rejected_receipts,mutation):
    project,source,entry,reseal=rejected_receipts
    if mutation in {'missing_alternative','playable_claim'}:
        path=source/'result.json';data=json.loads(path.read_bytes())
        if mutation=='missing_alternative':data['candidates'].pop()
        else:data['candidates'][0].update(status='ready',lab_directory='invented')
        write_json(path,data)
    elif mutation=='context':
        path=source/'conditional-context.json';data=json.loads(path.read_bytes());data['candidates'][0]['assignment']['body_correspondence']='invented';write_json(path,data)
    elif mutation=='config':
        path=source/'configs/alternative-1.json';data=json.loads(path.read_bytes());data['camera']['distance_model_units']=9;write_json(path,data)
    elif mutation=='child_result':write_json(source/'candidates/alternative-1/result.json',{'status':'ready'})
    else:(source/'inputs/base-manifest.json').write_bytes(b'{}')
    reseal()
    with pytest.raises(ValueError):module.inspect_rejected_assembly(source,entry,expected_manifest_sha256=digest(source/'manifest.json'),project=project)


def test_separable_source_roles_use_existing_automatic_route_not_conditional_abstention(project,monkeypatch):
    portal,folder,entry,calls=project
    validate=module.validate_source_entry
    def separable(*args,**kwargs):
        inputs,proof=validate(*args,**kwargs)
        proof.update(role_status='config_proposed',proposed_assignment={'body_correspondence':'person-mask','rigid_proxies':['object-mask']})
        return inputs,proof
    monkeypatch.setattr(module,'validate_source_entry',separable)
    import weft.equipment_autoproposal as automatic
    import weft.equipment_conditional_assembly as conditional
    def build(**kwargs):
        calls.append(kwargs)
        return {'status':'rejected_initial_geometry','reasons':[{'code':'floor','message':'Frozen original floor failed'}]}
    monkeypatch.setattr(automatic,'prepare_automatic_equipment_scene',build)
    monkeypatch.setattr(conditional,'prepare_conditional_equipment_scene',lambda **kwargs:pytest.fail('Separable input must use automatic source roles'))
    portal.prepare_equipment(folder.name,entry['id'])
    assert module.execute_equipment_job(folder,project=portal.project)==0 and len(calls)==1
    rows=portal.summary(folder.name)['equipment_candidates']['report']['candidates']
    assert len(rows)==1 and rows[0]['id']=='automatic-roles-1' and rows[0]['reasons']==['floor: Frozen original floor failed']
    assert not rows[0]['selectable']


@pytest.fixture
def perception_source(tmp_path,monkeypatch):
    """Analytic sealed bridge/provider receipts; no provider or media reconstruction."""
    import base64
    import math
    import test_shared_world_reconstruction as source_fixture
    from test_equipment_proposal import packages,seal
    from test_segmentation_evidence import returned
    from weft.shared_world_evidence import build_shared_world_binding
    from weft.shared_world import validate_shared_world
    from weft.exercise_perception_pipeline import body_tracking_anchor,read_equipment_analysis,PROFILE as BRIDGE_PROFILE
    from weft.portal_pipeline import INSTRUCTIONS,SCHEMA
    from weft.openai_access import MODEL,MAX_OUTPUT_TOKENS
    from weft.segmentation_observations import export_tracking_evidence
    root=tmp_path/'job';root.mkdir();video=root/'source.mp4'
    video.write_bytes(b'Analytic source fixture; not recorded physical footage')
    monkeypatch.setattr(source_fixture,'SOURCE',digest(video))
    write_json(root/'job.json',{'kind':'video','source_file':'source.mp4','source_sha256':digest(video)})
    write_json(root/'status.json',{'status':'needs_input','route':'exercise','events':[]})
    bridge=root/'perception/bridge';bridge.mkdir(parents=True)
    staging=tmp_path/'staging';staging.mkdir()
    body,old_equipment,_,scene,_=packages(staging)
    base=bridge/'body_seed';body.rename(base)
    old_pin=scene['source_binding']['entity_bindings'][0]['observation_document_sha256']
    document=json.loads((base/scene['source_binding']['evidence_documents'][old_pin]).read_bytes())
    torso=deepcopy(next(r for r in document['observations'] if r['object_id']=='person' and r['frame']==2))
    points=torso['measurements']['landmarks']['value']
    positions={0:[320,160],13:[290,210],14:[350,210],15:[280,230],16:[360,230],25:[310,275],26:[330,275],27:[310,310],28:[330,310]}
    points.extend({'index':i,'position':pos,'visibility':1.,'presence':1.} for i,pos in positions.items())
    initial=next(r for r in document['observations'] if r['object_id']=='person' and r['frame']==0)
    evidence_id=digest(video)+':frame:0'
    initial.update(visibility='detected',measurements=torso['measurements'],evidence_ids=[evidence_id])
    document['evidence'].append({'id':evidence_id,'frame':0,'time_s':0.,'object_id':'person',
        'kind':'pose_model_output','status':'inferred','payload':{'scope':'Analytic fixture, no model call'}})
    encoded=json.dumps(document).encode();pin=__import__('hashlib').sha256(encoded).hexdigest()
    scene['source_binding'],_=build_shared_world_binding({pin:encoded},source_sha256=digest(video),
        world_ids=['human',*[o['id'] for o in scene['objects']]],entity_refs={'human':(pin,'person'),**{o['id']:None for o in scene['objects']}})
    scene=validate_shared_world(scene)
    target=base/scene['source_binding']['evidence_documents'][pin];target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(encoded)
    write_json(base/'scene.json',scene);seal(base)
    equipment_bounds=[]
    for oid in ('foam-pad','training-weight'):
        box=next(r['measurements']['box']['value'] for r in document['observations'] if r['object_id']==oid and r['frame']==8)
        x,y,w,h=box;equipment_bounds.append([math.floor(x),math.floor(y),math.ceil(x+w),math.ceil(y+h)])
    analysis=bridge/'analysis';analysis.mkdir()
    frame={'frame':0,'decoder_time_s':0.,'nominal_time_s':0.,'artifact':'frame-00000.jpg'}
    image=b'Analytically authored image receipt; no physical claim';(analysis/frame['artifact']).write_bytes(image);frame['sha256']=digest(analysis/frame['artifact'])
    frames={'source_sha256':digest(video),'width':640,'height':480,'fps':30.,'decoded_frame_count':63,'frames':[frame]}
    plan={'route':'exercise','camera':'fixed','people':1,'exercise_inventory':'equipment_or_obstacles',
        'summary':'Synthetic provider receipt fixture, no paid call','warnings':[],
        'objects':[{'label':f'Analytic equipment{i}','shape':'box','role':'static','box_xyxy':[b[0]/640,b[1]/480,b[2]/640,b[3]/480]} for i,b in enumerate(equipment_bounds)]}
    description={'video':{k:frames[k] for k in ('width','height','fps','decoded_frame_count')},
        'frames':[{k:frame[k] for k in ('frame','decoder_time_s','nominal_time_s')}]}
    content=[{'type':'input_text','text':json.dumps(description)}]
    payload={'model':MODEL,'instructions':INSTRUCTIONS,'store':False,'input':[{'role':'user','content':content}],
        'max_output_tokens':MAX_OUTPUT_TOKENS,'text':{'format':{'type':'json_schema','name':'weft_scene_plan','strict':True,'schema':SCHEMA}}}
    saved=deepcopy(payload)
    content.append({'type':'input_image','detail':'high','image_url':'data:image/jpeg;base64,'+base64.b64encode(image).decode()})
    request_pin=canonical_hash(payload);saved.update(images=[frame],request_sha256=request_pin,scope='Synthetic fixture')
    response={'id':'synthetic-response','model':MODEL,'status':'completed','output':[{'type':'message','content':[{'type':'output_text','text':json.dumps(plan)}]}]}
    receipt={'request_sha256':request_pin,'response_id':'synthetic-response','model':MODEL,'provider':'openai','authentication':'server_project_api_key','fixture':True}
    for name,value in [('frames.json',frames),('plan.json',plan),('request.json',saved),('response.json',response),('receipt.json',receipt)]:write_json(analysis/name,value)
    proposal,_=read_equipment_analysis(analysis)
    anchor=body_tracking_anchor(base,scene,proposal)
    write_json(bridge/'body-anchor.json',anchor)
    tracking=bridge/'tracking';(tracking/'call').mkdir(parents=True);(tracking/'source.mp4').write_bytes(video.read_bytes())
    prompts=[{'object_id':row[0],'frame':row[1],'roi':row[2:]} for row in [*proposal['boxes'],anchor['prompt']]]
    boxes=[]
    for index in range(63):
        row=next(r for r in document['observations'] if r['object_id']=='person' and r['frame']==max(2,index))
        pts=row['measurements']['landmarks']['value'];xs=[p['position'][0] for p in pts];ys=[p['position'][1] for p in pts]
        bounds=[math.floor(min(xs)-15),math.floor(min(ys)-35),math.ceil(max(xs)+15),math.ceil(max(ys)+75)]
        values=[]
        for identifier,box in enumerate([*equipment_bounds,bounds],1):
            value=returned(index,identifier,box);value['source_dimensions']=[640,480];values.append(value)
        boxes.append({'frame':index,'objects':values})
    write_json(tracking/'video.json',{'source_sha256':digest(video),'width':640,'height':480,'frame_count':63,'decoder_timestamps_s':[i/30 for i in range(63)]})
    write_json(tracking/'call/call.json',{'model':'sam-3.1','video_sha256':digest(video),'status':'completed','initialization':prompts,'response_id':'analytic-tracking','request_sha256':'a'*64})
    write_json(tracking/'call/segmentation.json',{'status':'completed','object_ids':['1','2','3'],'initialization':prompts,'response_id':'analytic-tracking','frames':boxes,'raw_text':'Synthetic detector fixture'})
    write_json(tracking/'report.json',{'objects':prompts});seal(tracking)
    equipment=bridge/'equipment_evidence';export_tracking_evidence(tracking,equipment)
    write_json(bridge/'plan.json',{'profile':BRIDGE_PROFILE,'source_sha256':digest(video),'scope':'Synthetic bridge receipt'})
    write_json(bridge/'result.json',{'profile':BRIDGE_PROFILE,'status':'evidence_ready','source_sha256':digest(video),
        'assembly_inputs_ready':True,'simulation_ready':False,'physics_steps':0,'source_mappings':[],
        'base_run':str(base),'evidence_run':str(equipment),'body_anchor_status':'sealed_pose_tracking_hypothesis',
        'tracking_manifest_sha256':digest(tracking/'manifest.json')})
    (bridge/'implementation').mkdir();(bridge/'implementation/exercise_perception_pipeline.py').write_bytes(b'Analytic receipt fixture; never executed')
    members={str(p.relative_to(bridge)):digest(p) for directory in (analysis,bridge/'implementation') for p in directory.rglob('*') if p.is_file()}
    members.update({n:digest(bridge/n) for n in ('plan.json','result.json','body-anchor.json')})
    write_json(bridge/'manifest.json',{'schema_version':1,'profile':BRIDGE_PROFILE,'status':'evidence_ready','files':members,
        'children':{n:digest(bridge/n/'manifest.json') for n in ('body_seed','tracking','equipment_evidence')}})
    return root,bridge,tmp_path


def test_job_local_descriptor_rederives_complete_source_without_global_catalog(perception_source):
    root,bridge,project=perception_source
    descriptor=module.publish_perception_source(root,bridge,project=project)
    saved=(root/module.JOB_SOURCE_PATH).read_bytes()
    assert descriptor['base']['path']=='perception/bridge/body_seed'
    assert descriptor['physical_validation']=='unverified' and descriptor['mappings']==[]
    assert module.publish_perception_source(root,bridge,project=project)==descriptor
    assert (root/module.JOB_SOURCE_PATH).read_bytes()==saved
    inputs,proof=module.validate_job_source(root,project=project)
    assert inputs['base_run']==bridge/'body_seed' and proof['source_kind']=='perception'
    assert proof['raw_object_ids']==['sam-1','sam-2','sam-3']
    assert not (root.parent/module.CATALOG).exists()


@pytest.mark.parametrize('change',['source','descriptor','bridge','child','anchor'])
def test_job_local_changed_custody_cannot_reach_construction(perception_source,change):
    root,bridge,project=perception_source
    module.publish_perception_source(root,bridge,project=project)
    path={'source':root/'source.mp4','descriptor':root/module.JOB_SOURCE_PATH,'bridge':bridge/'result.json',
          'child':bridge/'equipment_evidence/scene-observations.json','anchor':bridge/'body-anchor.json'}[change]
    path.write_bytes(path.read_bytes()+b' ')
    with pytest.raises(ValueError):module.validate_job_source(root,project=project)


def test_job_local_worker_uses_real_ordinary_builder_and_retains_descriptor(perception_source):
    root,bridge,project=perception_source
    module.publish_perception_source(root,bridge,project=project)
    work=root/'equipment';work.mkdir()
    write_json(work/'request.json',{'profile':module.PROFILE,'source_kind':'perception','source_sha256':digest(root/'source.mp4'),
        'descriptor_sha256':digest(root/module.JOB_SOURCE_PATH)})
    (work/'parent-status.json').write_bytes((root/'status.json').read_bytes())
    write_json(work/'status.json',{'status':'queued'})
    assert module.execute_equipment_job(root,project=project)==0,json.loads((work/'status.json').read_bytes())
    report=module.read_report(root,module.read_status(root))
    assert any(row['selectable'] for row in report['candidates']),report
    chosen=next(row for row in report['candidates'] if row['selectable'])
    assert (root/chosen['lab']/'run/manifest.json').is_file()
    assert report['source_validation']['descriptor_sha256']==digest(root/module.JOB_SOURCE_PATH)
    assert not (root.parent/module.CATALOG).exists()


def test_job_local_external_preflight_is_exact_and_cannot_escape_job(tmp_path,monkeypatch):
    """Real local preflight/anchor; synthetic tracking response completes the no-call fixture."""
    from test_exercise_perception_pipeline import local_preflight_source,_fixture_analysis
    from test_segmentation_evidence import returned,seal
    from weft import exercise_perception_pipeline as bridge_module
    from weft.segmentation_observations import export_tracking_evidence
    source=tmp_path/'source';source.mkdir()
    video,body=local_preflight_source.__wrapped__(source)
    root=tmp_path/'job';root.mkdir();(root/'source.mp4').write_bytes(video.read_bytes())
    write_json(root/'job.json',{'kind':'video','source_file':'source.mp4','source_sha256':digest(video)})
    preflight=root/'perception/preflight'
    bridge_module.prepare_exercise_perception_preflight(root/'source.mp4',preflight,body_source=body)
    analysis=_fixture_analysis(root/'source.mp4',root/'analysis')
    bridge=root/'perception/bridge'
    bridge_module.prepare_exercise_perception(root/'source.mp4',analysis,bridge,preflight=preflight)
    tracking=bridge/'tracking';(tracking/'call').mkdir()
    track_report=json.loads((tracking/'report.json').read_bytes());prompts=track_report['objects']
    frames=[]
    for i in range(63):
        rows=[]
        for prompt in prompts:
            x,y,w,h=prompt['roi'];row=returned(i,prompt['object_id'],[x,y,x+w,y+h]);row['source_dimensions']=[640,480];rows.append(row)
        frames.append({'frame':i,'objects':rows})
    write_json(tracking/'call/call.json',{'model':'sam-3.1','video_sha256':digest(video),'status':'completed',
        'initialization':prompts,'response_id':'synthetic-tracking-no-call','request_sha256':digest(tracking/'request.json')})
    write_json(tracking/'call/segmentation.json',{'status':'completed','object_ids':[str(p['object_id']) for p in prompts],
        'initialization':prompts,'response_id':'synthetic-tracking-no-call','frames':frames,'raw_text':'Analytic response fixture, no provider call'})
    seal(tracking)
    export_tracking_evidence(tracking,bridge/'equipment_evidence')
    result=json.loads((bridge/'result.json').read_bytes())
    result.update(status='evidence_ready',assembly_inputs_ready=True,base_run=str(preflight/'body_seed'),
        evidence_run=str(bridge/'equipment_evidence'),source_mappings=[],tracking_manifest_sha256=digest(tracking/'manifest.json'))
    write_json(bridge/'result.json',result)
    manifest=json.loads((bridge/'manifest.json').read_bytes());manifest['status']='evidence_ready'
    manifest['files']['result.json']=digest(bridge/'result.json')
    manifest['children']={n:digest(bridge/n/'manifest.json') for n in ('tracking','equipment_evidence')}
    write_json(bridge/'manifest.json',manifest)
    descriptor=module.publish_perception_source(root,bridge,project=tmp_path)
    assert descriptor['base']['path']=='perception/preflight/body_seed'
    assert descriptor['provenance']['preflight_descriptor_sha256']==digest(bridge/'preflight-descriptor.json')
    assert not (bridge/'body_seed').exists()
    inputs,_=module.validate_job_source(root,project=tmp_path)
    assert inputs['base_run']==preflight/'body_seed'
    # Reserve a generated child before its independent validator can allocate/read it.
    preflight_before=(preflight/'manifest.json').read_bytes();external_before=(bridge/'preflight-descriptor.json').read_bytes()
    generated=preflight/'body_source';generated.mkdir();(generated/'fixture.bin').write_bytes(b'x'*1024);seal(generated)
    changed_preflight=json.loads(preflight_before);changed_preflight['children']['body_source']=digest(generated/'manifest.json')
    write_json(preflight/'manifest.json',changed_preflight)
    external=json.loads(external_before);external['preflight_manifest_sha256']=digest(preflight/'manifest.json')
    write_json(bridge/'preflight-descriptor.json',external)
    manifest['files']['preflight-descriptor.json']=digest(bridge/'preflight-descriptor.json');write_json(bridge/'manifest.json',manifest)
    baseline=sum((bridge/name).stat().st_size for name in manifest['files'])
    for directory in (tracking,bridge/'equipment_evidence',preflight,preflight/'body_seed'):
        sealed=json.loads((directory/'manifest.json').read_bytes())
        baseline+=(directory/'manifest.json').stat().st_size+sum((directory/name).stat().st_size for name in sealed['files'])
    with monkeypatch.context() as patch:
        patch.setattr(module,'MAX_JOB_SOURCE_BYTES',baseline+512)
        patch.setattr(bridge_module,'validate_perception_preflight',lambda *a,**k:pytest.fail('Reserve generated child before independent validation'))
        with pytest.raises(ValueError,match='aggregate byte bound'):
            module.validate_job_source(root,project=tmp_path)
    (preflight/'manifest.json').write_bytes(preflight_before)
    (bridge/'preflight-descriptor.json').write_bytes(external_before)
    external=json.loads((bridge/'preflight-descriptor.json').read_bytes());external['base_run']=str(body)
    write_json(bridge/'preflight-descriptor.json',external)
    manifest['files']['preflight-descriptor.json']=digest(bridge/'preflight-descriptor.json');write_json(bridge/'manifest.json',manifest)
    with pytest.raises(ValueError,match='fixed job-local preflight'):
        module.validate_job_source(root,project=tmp_path)


def test_job_local_resealed_anchor_cannot_replace_original_landmarks(perception_source):
    root,bridge,project=perception_source
    anchor=json.loads((bridge/'body-anchor.json').read_bytes());anchor['prompt'][2]+=1
    write_json(bridge/'body-anchor.json',anchor)
    manifest=json.loads((bridge/'manifest.json').read_bytes());manifest['files']['body-anchor.json']=digest(bridge/'body-anchor.json')
    write_json(bridge/'manifest.json',manifest)
    with pytest.raises(ValueError,match='original analysis|exact source landmarks'):
        module.publish_perception_source(root,bridge,project=project)


@pytest.mark.parametrize('mutation',['extra_path','pin','source'])
def test_job_local_worker_request_is_closed_and_pinned_before_builder(perception_source,monkeypatch,mutation):
    root,bridge,project=perception_source
    module.publish_perception_source(root,bridge,project=project)
    work=root/'equipment';work.mkdir()
    request={'profile':module.PROFILE,'source_kind':'perception','source_sha256':digest(root/'source.mp4'),
             'descriptor_sha256':digest(root/module.JOB_SOURCE_PATH)}
    if mutation=='extra_path':request['base_run']=str(bridge/'body_seed')
    elif mutation=='pin':request['descriptor_sha256']='0'*64
    else:request['source_sha256']='0'*64
    write_json(work/'request.json',request);write_json(work/'status.json',{'status':'queued'})
    (work/'parent-status.json').write_bytes((root/'status.json').read_bytes())
    import weft.equipment_conditional_assembly as conditional
    monkeypatch.setattr(conditional,'prepare_conditional_equipment_scene',lambda **kwargs:pytest.fail('No construction after invalid request'))
    assert module.execute_equipment_job(root,project=project)==1
    assert module.read_status(root)['status']=='failed' and not (work/'run').exists()


def test_job_local_anchor_profile_rederives_explicit_visible_chain(perception_source,monkeypatch):
    from weft import exercise_perception_pipeline as bridge_module
    root,bridge,project=perception_source
    anchor=json.loads((bridge/'body-anchor.json').read_bytes())
    anchor['profile']=bridge_module.VISIBLE_ANCHOR_PROFILE
    write_json(bridge/'body-anchor.json',anchor)
    manifest=json.loads((bridge/'manifest.json').read_bytes())
    manifest['files']['body-anchor.json']=digest(bridge/'body-anchor.json');write_json(bridge/'manifest.json',manifest)
    # This truncated legacy fixture checks dispatch only; v2's complete-slot proof has its own real-source tests.
    def rederive(*args,anchor_profile):
        assert anchor_profile==bridge_module.VISIBLE_ANCHOR_PROFILE
        raise ValueError('Independent v2 rederivation reached')
    monkeypatch.setattr(bridge_module,'body_tracking_anchor',rederive)
    with pytest.raises(ValueError,match='Independent v2 rederivation reached'):
        module.publish_perception_source(root,bridge,project=project)
    assert not (root/'perception/prepared-source.json').exists()


@pytest.mark.parametrize('deadline',[True,float('nan'),float('inf'),'later',-1.])
def test_job_local_validation_rejects_invalid_or_expired_shared_deadline(tmp_path,deadline):
    with pytest.raises((ValueError,TimeoutError)):
        module.validate_job_source(tmp_path,deadline=deadline)
