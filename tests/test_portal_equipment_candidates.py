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
        with pytest.raises(ValueError,match='Conditional equipment'):action(folder.name)


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
