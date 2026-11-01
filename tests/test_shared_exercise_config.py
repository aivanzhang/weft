"""Authored metadata exercises config custody; no body is constructed or admitted."""

from copy import deepcopy
from hashlib import sha256
import builtins
import json
from pathlib import Path
import socket
import subprocess
import sys
from types import ModuleType
import zipfile

import pytest

from weft import shared_exercise_config as module
from weft.observations import COORDINATES, adapt_exercise
from test_source_equivalence import fixture as proof_fixture


def encoded(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False)+'\n').encode()


def digest(raw):
    return sha256(raw).hexdigest()


def write_package(root, values, **metadata):
    root.mkdir(parents=True)
    for name, raw in values.items():
        path=root/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(raw)
    seal=encoded({'schema':1,**metadata,'files':{name:digest(raw) for name,raw in values.items()}})
    (root/'manifest.json').write_bytes(seal)
    return digest(seal)


def reseal(root, name, mutate):
    value=json.loads((root/name).read_bytes());mutate(value);(root/name).write_bytes(encoded(value))
    manifest=json.loads((root/'manifest.json').read_bytes())
    manifest['files'][name]=digest((root/name).read_bytes());(root/'manifest.json').write_bytes(encoded(manifest))


def observation(source, count, *, width=100, height=80, metadata=None, times=None):
    unknown={'value':None,'status':'unknown','evidence_ids':[]}
    result={'schema_version':1 if times is None else 3,'profile':'scene-observations-v1',
        'source':{'sha256':source,'width_px':width,'height_px':height,'fps':30,
                  'metadata_frames':count if metadata is None else metadata,'decoded_frames':count},
        'time':{'unit':'seconds','origin':'video_start','basis':'frame_index_divided_by_fps','status':'assumed'},
        'image':{'width_px':width,'height_px':height},'coordinates':deepcopy(COORDINATES),
        'camera':{'motion':deepcopy(unknown),'calibration':deepcopy(unknown),'screen':None},
        'objects':[{'id':'weight','label':'Authored test object','properties':{'kind':deepcopy(unknown),'role':deepcopy(unknown)},'unknowns':[]}],
        'observations':[],'evidence':[],'relationships':[],'provenance':{},'limits':['Authored metadata, not physical evidence']}
    if times is not None:
        result['source'].pop('fps');result['source']['frame_times_s']=times
        result['time']={'unit':'seconds','origin':'submitted_video_start','basis':'decoder_presentation_timestamps','status':'observed'}
    for frame in range(count):
        stamp=frame/30 if times is None else times[frame];identifier=f'{source}:frame:{frame}:object:weight'
        result['evidence'].append({'id':identifier,'frame':frame,'time_s':stamp,'object_id':'weight',
            'kind':'segmentation_model_output','status':'inferred','payload':{'geometry_status':'decoded_mask','sentinel':'original_saved_payload'}})
        result['observations'].append({'id':f'{source}:observation:{frame}:weight','frame':frame,'time_s':stamp,
            'object_id':'weight','visibility':'detected','partition':'construction','evidence_ids':[identifier],
            'measurements':{'box':{'value':[.1,.2,.3,.4],'status':'inferred','unit':'pixels','coordinate_system':'image'}}})
    return result


def fixture(root, *, derivative=False, saved_human=False, count=6):
    root.mkdir(parents=True,exist_ok=True)
    proof=None
    if derivative:
        proof=proof_fixture(root/'mapping',derivative_offset=1)
        source=(proof/'source.webm').read_bytes();other=(proof/'lossless.mp4').read_bytes()
        count,metadata,width,height,reference_frames=3,3,2,2,[0,1,2]
    else:
        source=b'authored source bytes; not a decodable video';other=source
        metadata,width,height,reference_frames=count+2 if count==6 else count,100,80,list(range(0,count,2))
    source_pin=digest(source);other_pin=digest(other)
    point={'x':.5,'y':.5,'z':0.,'visibility':.9,'presence':.9}
    pose={'profile':'exercise-pose-v1','source_sha256':source_pin,'width':width,'height':height,'fps':30,
        'frame_count':metadata,'decoded_frames':count,'frames':[{'id':f'{source_pin}:frame:{f}','frame':f,'time_s':f/30,
        'status':'tracked','pose_count':1,'landmarks':[deepcopy(point) for _ in range(33)]} for f in reference_frames]}
    references=[{'time_s':f/30,'angles':{}} for f in reference_frames]
    body={'profile':'exercise-v3','source_sha256':source_pin,'reference':references,
        'reference_evidence':[{'evidence_id':f'{source_pin}:frame:{f}','time_s':f/30} for f in reference_frames],
        'root_reference':[{'time_s':f/30,'position':[0,0,1]} for f in reference_frames],
        'mobility':{'floor_height':0},'lengths':{'torso':.4}}
    values={'human.json':encoded(body),'scene.json':encoded({'profile':'exercise-v3','source_sha256':source_pin}),
        'evidence.json':encoded(pose),'source.webm':source,'reconstruction.json':encoded({'status':'completed'})}
    if saved_human:
        canonical=adapt_exercise(pose,scene=json.loads(values['scene.json']))
        canonical['provenance']['input_artifacts']={name:digest(values[name]) for name in ('evidence.json','scene.json')}
        values['observations.json']=encoded(canonical)
    human=root/'human';write_package(human,values)
    equipment=root/'equipment';doc=observation(other_pin,count,width=width,height=height,metadata=metadata,
        times=[.002,.035,.069] if derivative else None)
    write_package(equipment,{'scene-observations.json':encoded(doc),'report.json':encoded({'status':'evidence_only'}),
        'segmentation-evidence.json':encoded({'scope':'authored fixture only'})})
    return {'human':human,'equipment':equipment,'mapping':proof,'pose':pose,'body':body,'equipment_doc':doc}


@pytest.fixture(autouse=True)
def local_only(monkeypatch):
    def denied(*args,**kwargs):raise AssertionError('Native, process, model or network work is forbidden')
    original=builtins.__import__
    def guarded(name,*args,**kwargs):
        if name.split('.')[0] in {'mujoco','av','cv2','mediapipe','tensorflow','torch'}:denied()
        return original(name,*args,**kwargs)
    monkeypatch.setattr(builtins,'__import__',guarded)
    monkeypatch.setattr(socket,'socket',denied);monkeypatch.setattr(subprocess,'Popen',denied)
    report=ModuleType('weft.shared_exercise_config_report')
    report.build_shared_exercise_config_report=lambda files,**kwargs:{'index.html':b'<!doctype html><p>Authored report stub</p>',
        'brand.css':(Path(module.__file__).parent/'assets/brand.css').read_bytes()}
    monkeypatch.setitem(sys.modules,report.__name__,report)


def prepare(inputs,out,**kwargs):
    return module.prepare_shared_exercise_config(inputs['human'],inputs['equipment'],out,
        source_mappings=() if inputs['mapping'] is None else (inputs['mapping'],),**kwargs)


def request_for(output, path, **patch):
    snapshot=json.loads((output/'snapshot.json').read_bytes())
    value=json.loads((output/'request.json').read_bytes())
    human=snapshot['observation_sources'][0]
    value.update(human_entity_ref={'document_sha256':human['document_sha256'],'object_id':'person-0'},
        camera={'mode':'fixed_pelvis_perspective_prior','distance_model_units':3.},
        geometry_prior='floor_box_depth_equals_width',equipment=[{'object_id':'weight','source_frame':0}],target_object_id='weight')
    value.update(patch);path.write_bytes(encoded(value));return path


def resume(old,request,out,**kwargs):
    return module.prepare_shared_exercise_config(None,None,out,resume=old,request=request,
        expected_manifest_sha256=digest((old/'manifest.json').read_bytes()),**kwargs)


def verify_package(root):
    manifest=json.loads((root/'manifest.json').read_bytes())
    assert set(manifest['files'])=={p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file()}-{'manifest.json'}
    assert all(digest((root/name).read_bytes())==pin for name,pin in manifest['files'].items())
    return manifest


def test_discovery_has_no_scene_choices_and_preserves_all_slots_copies_and_runtime(tmp_path):
    inputs=fixture(tmp_path/'inputs');out=tmp_path/'setup';result=prepare(inputs,out)
    assert result['status']=='needs_input',result
    manifest=verify_package(out);snapshot=json.loads((out/'snapshot.json').read_bytes());request=json.loads((out/'request.json').read_bytes())
    assert request==module._default_request(digest((out/'snapshot.json').read_bytes()))
    assert snapshot['counts']=={'original_source_slots':8,'decoded_source_slots':6,'body_references':3,
        'human_entity_slots':6,'equipment_entity_slots':6,'metadata_tail_slots':2}
    assert [r['visibility'] for r in snapshot['observation_sources'][0]['slots']]==['detected',None,'detected',None,'detected',None]
    assert result['build_inputs'] is None and result['world_admission']=='not_evaluated' and result['full_reference'] is True
    assert not any(result['work'].values()) and result['resume_input_budget']['maximum_selected_input_bytes']<=module.MAX_INPUT_BYTES
    for role in ('human','equipment'):
        for path in inputs[role].iterdir():assert path.read_bytes()==(out/'sources'/role/path.name).read_bytes()
    with zipfile.ZipFile(out/'implementation-source.zip') as archive:
        assert archive.read('assets/brand.css')==(out/'inspection/brand.css').read_bytes()
        assert archive.read('shared_exercise_config.py')==Path(module.__file__).read_bytes()
    assert manifest['status']=='needs_input'


def test_resume_uses_only_complete_copied_sources_and_validates_ordinary_config(tmp_path,monkeypatch):
    inputs=fixture(tmp_path/'inputs');old=tmp_path/'setup';assert prepare(inputs,old)['status']=='needs_input'
    request=request_for(old,tmp_path/'edited.json',build_options={'body_scale':1.25,'root_assistance':True},frame_translation_xy=[.1,.2])
    before={p:p.read_bytes() for p in old.rglob('*') if p.is_file()}
    calls=[];original=module.read_package_file
    def selected(root,name,**kwargs):
        path=Path(root)/name;calls.append(path)
        assert not path.is_relative_to(tmp_path/'inputs')
        assert not(path.is_relative_to(old) and (name.startswith('inspection/') or name=='implementation-source.zip'))
        return original(root,name,**kwargs)
    monkeypatch.setattr(module,'read_package_file',selected)
    new=tmp_path/'configured';result=resume(old,request,new)
    assert result['status']=='config_ready',result
    config=json.loads((new/'config.json').read_bytes())
    assert config==module.validate_proposal_config(config) and config['frame_translation_xy']==[.1,.2]
    assert result['build_options']=={'body_scale':1.25,'root_assistance':True}
    assert result['build_inputs']['human_run']==str(new/'sources/human')
    assert (new/'snapshot.json').read_bytes()==before[old/'snapshot.json']
    assert all(path.read_bytes()==raw for path,raw in before.items())
    verify_package(new)
    assert not {'snapshot_sha256','build_options','human_entity_ref'}&config.keys()


@pytest.mark.parametrize('patch,code',[
    ({'human_entity_ref':{'document_sha256':'f'*64,'object_id':'person-0'}},'source_or_config_unavailable'),
    ({'equipment':[{'object_id':'missing','source_frame':0}],'target_object_id':'missing'},'source_or_config_unavailable'),
    ({'equipment':[{'object_id':'weight','source_frame':1}]},'source_or_config_unavailable'),
    ({'camera':{'mode':'fixed_pelvis_perspective_prior','distance_model_units':True}},'invalid_camera_distance'),
    ({'geometry_prior':'inferred_contact'},'unsupported_geometry_prior'),
    ({'build_options':{'body_scale':True,'root_assistance':False}},'invalid_build_options'),
    ({'build_options':{'body_scale':1.,'root_assistance':1}},'invalid_build_options'),
    ({'frame_translation_xy':[0.,0.]},'invalid_frame_translation'),
])
def test_invalid_explicit_choices_stay_inspectable_without_config(tmp_path,patch,code):
    old=tmp_path/'setup';prepare(fixture(tmp_path/'inputs'),old)
    request=request_for(old,tmp_path/'edited.json',**patch);new=tmp_path/'failed-choices';result=resume(old,request,new)
    assert result['status']=='needs_input' and result['sealed'] and result['config_path'] is None,result
    metadata=json.loads((new/'metadata.json').read_bytes())
    assert code in {r['code'] for r in metadata['issues']}
    assert (new/'request.json').read_bytes()==request.read_bytes()
    verify_package(new)


def test_withheld_human_and_equipment_slots_survive_without_payloads(tmp_path):
    inputs=fixture(tmp_path/'inputs',saved_human=True)
    for root,name in ((inputs['human'],'observations.json'),(inputs['equipment'],'scene-observations.json')):
        reseal(root,name,lambda doc:doc['observations'][0].update(partition='audit'))
    out=tmp_path/'setup';result=prepare(inputs,out);assert result['status']=='unavailable',result
    snapshot=json.loads((out/'snapshot.json').read_bytes())
    for view in snapshot['observation_sources']:
        row=view['slots'][0]
        assert row['partition']=='audit' and row['visibility'] is None and not row['evidence_ids'] and not row['selection_eligible']
    assert b'original_saved_payload' not in (out/'snapshot.json').read_bytes()
    result=resume(out,request_for(out,tmp_path/'request.json'),tmp_path/'configured')
    assert result['status']=='unavailable' and result['config_path'] is None


def test_verified_derivative_preserves_distinct_decoder_clock_and_full_proof(tmp_path):
    inputs=fixture(tmp_path/'inputs',derivative=True);out=tmp_path/'setup';result=prepare(inputs,out)
    assert result['status']=='needs_input',result
    snapshot=json.loads((out/'snapshot.json').read_bytes());equipment=snapshot['observation_sources'][1]
    assert equipment['alignment']=='verified_derivative' and [r['time_s'] for r in equipment['slots']]==[.002,.035,.069]
    assert [r['original_frame'] for r in equipment['slots']]==[0,1,2]
    mapping=snapshot['inputs']['source_mappings'][0]
    for p in inputs['mapping'].iterdir():assert p.read_bytes()==(out/mapping['directory']/p.name).read_bytes()
    result=resume(out,request_for(out,tmp_path/'request.json'),tmp_path/'configured')
    assert result['status']=='config_ready',result


def test_missing_mapping_preserves_unaligned_slots_and_unknowns(tmp_path):
    inputs=fixture(tmp_path/'inputs',derivative=True);inputs['mapping']=None
    out=tmp_path/'setup';result=prepare(inputs,out);assert result['status']=='unavailable',result
    snapshot=json.loads((out/'snapshot.json').read_bytes())
    assert all(r['original_frame'] is None and not r['selection_eligible'] for r in snapshot['observation_sources'][1]['slots'])
    assert snapshot['contact']==snapshot['grasp']==snapshot['load']=='unknown'
    verify_package(out)


@pytest.mark.parametrize('change', ['source','clock','reference_count','floor','camera_body','failed_parent'])
def test_source_body_mismatches_cannot_produce_ready_config(tmp_path,change):
    inputs=fixture(tmp_path/'inputs');human=inputs['human']
    if change=='source':reseal(human,'evidence.json',lambda d:d.update(source_sha256='f'*64))
    if change=='clock':reseal(human,'human.json',lambda d:d['reference_evidence'][0].update(time_s=.01))
    if change=='reference_count':reseal(human,'human.json',lambda d:d['reference'].pop())
    if change=='floor':reseal(human,'human.json',lambda d:d['mobility'].update(floor_height=.1))
    if change=='camera_body':reseal(human,'human.json',lambda d:d.update(camera_reference={}))
    if change=='failed_parent':reseal(human,'reconstruction.json',lambda d:d.update(status='failed'))
    out=tmp_path/'setup';result=prepare(inputs,out)
    assert result['status'] in {'failed','unavailable'} and result['build_inputs'] is None,result
    if change in {'floor','camera_body','failed_parent'}:
        assert result['status']=='unavailable' and result['sealed']
        expected={'floor':'unsupported_original_floor','camera_body':'unsupported_body_profile','failed_parent':'human_saved_status_failed'}[change]
        assert expected in {r['code'] for r in json.loads((out/'metadata.json').read_bytes())['issues']}
        verify_package(out)
    else:assert (out/'failure.json').is_file() and not (out/'manifest.json').exists()


def test_wrong_snapshot_pin_retains_exact_invalid_request_without_seal(tmp_path):
    old=tmp_path/'setup';prepare(fixture(tmp_path/'inputs'),old)
    request=request_for(old,tmp_path/'wrong.json',snapshot_sha256='f'*64);out=tmp_path/'rejected'
    result=resume(old,request,out)
    assert result['status']=='failed' and not result['sealed'] and not (out/'manifest.json').exists()
    assert (out/'failure-inputs/request.json').read_bytes()==request.read_bytes()


def test_unknown_resume_directory_cannot_escape_snapshot(tmp_path):
    old=tmp_path/'setup';prepare(fixture(tmp_path/'inputs'),old)
    reseal(old,'snapshot.json',lambda d:d['inputs'].update(source_mappings=[{'manifest_sha256':'f'*64,'directory':'../../outside'}]))
    request=tmp_path/'request.json';request.write_bytes((old/'request.json').read_bytes())
    out=tmp_path/'rejected';result=resume(old,request,out)
    assert result['status']=='failed' and 'seal-named' in result['error']['message']


def test_changed_copied_source_fails_its_outer_pin_before_admission(tmp_path):
    old=tmp_path/'setup';prepare(fixture(tmp_path/'inputs'),old);request=request_for(old,tmp_path/'request.json')
    (old/'sources/human/source.webm').write_bytes(b'tampered')
    result=resume(old,request,tmp_path/'rejected')
    assert result['status']=='failed' and not result['sealed']


def test_preexisting_output_and_symlink_inputs_are_never_changed(tmp_path):
    inputs=fixture(tmp_path/'inputs');out=tmp_path/'occupied';out.mkdir();sentinel=out/'keep';sentinel.write_bytes(b'keep')
    with pytest.raises(ValueError,match='new'):prepare(inputs,out)
    assert list(out.iterdir())==[sentinel] and sentinel.read_bytes()==b'keep'
    link=tmp_path/'linked';link.symlink_to(inputs['human'],target_is_directory=True);inputs['human']=link
    with pytest.raises(ValueError,match='symlink'):prepare(inputs,tmp_path/'new')


def test_complete_input_and_resumability_caps_fail_without_trimming(tmp_path,monkeypatch):
    inputs=fixture(tmp_path/'inputs');monkeypatch.setattr(module,'MAX_INPUT_FILES',4)
    result=prepare(inputs,tmp_path/'small-files');assert result['status']=='failed' and '256-file' in result['error']['message']
    monkeypatch.setattr(module,'MAX_INPUT_FILES',256);monkeypatch.setattr(module,'MAX_INPUT_BYTES',100_000)
    result=prepare(inputs,tmp_path/'small-resume')
    assert result['status']=='failed' and 'resume metadata' in result['error']['message']
    assert not (tmp_path/'small-resume/manifest.json').exists()


def test_parent_mutation_before_publication_leaves_unsealed_failure(tmp_path):
    inputs=fixture(tmp_path/'inputs')
    def progress(stage,_):
        if stage=='write':(inputs['human']/'source.webm').write_bytes(b'changed after snapshot')
    out=tmp_path/'setup';result=prepare(inputs,out,progress=progress)
    assert result['status']=='failed' and 'Source changed' in result['error']['message'] and not (out/'manifest.json').exists()


def test_deadline_failure_never_publishes_terminal_seal(tmp_path,monkeypatch):
    inputs=fixture(tmp_path/'inputs');clock=[100.];monkeypatch.setattr(module.time,'monotonic',lambda:clock[0])
    def progress(stage,_):
        if stage=='availability':clock[0]+=31
    out=tmp_path/'setup';result=prepare(inputs,out,progress=progress)
    assert result['status']=='failed' and result['error']['type']=='TimeoutError' and not (out/'manifest.json').exists()


def test_valid_terminal_seal_is_monotonic_after_caller_write_exception(tmp_path,monkeypatch):
    original=module.write_package_files
    def fail_late(root,files):
        original(root,files)
        if set(files)=={'manifest.json'}:raise OSError('caller failed after exact terminal publication')
    monkeypatch.setattr(module,'write_package_files',fail_late)
    out=tmp_path/'setup'
    with pytest.raises(OSError,match='caller failed'):prepare(fixture(tmp_path/'inputs'),out)
    assert verify_package(out)['status']=='needs_input' and not (out/'failure.json').exists()
    failure=json.loads((tmp_path/'setup.caller-failure.json').read_bytes())
    assert failure['status']=='caller_failed_after_terminal_publication' and failure['artifact_unchanged'] is True


def test_partial_terminal_write_never_becomes_completed(tmp_path,monkeypatch):
    original=module.write_package_files
    def fail_partial(root,files):
        if set(files)=={'manifest.json'}:
            (Path(root)/'manifest.json').write_bytes(b'{');raise OSError('partial write')
        original(root,files)
    monkeypatch.setattr(module,'write_package_files',fail_partial)
    out=tmp_path/'setup';result=prepare(fixture(tmp_path/'inputs'),out)
    assert result['status']=='failed' and not result['sealed'] and (out/'manifest.json').read_bytes()==b'{'
    assert (out/'failure.json').exists()


def test_explicit_alternate_human_is_not_silently_replaced_by_builder_person_zero(tmp_path):
    inputs=fixture(tmp_path/'inputs',saved_human=True)
    def add_person(doc):
        obj=deepcopy(doc['objects'][0]);obj['id']='person-1';doc['objects'].append(obj)
        for original in list(doc['observations']):
            row=deepcopy(original);row.update(id=original['id'].replace('person-0','person-1'),object_id='person-1',
                visibility='not_sampled',partition='unsampled',evidence_ids=[],measurements={});doc['observations'].append(row)
        doc['observations'].sort(key=lambda r:(r['frame'],r['object_id']))
    reseal(inputs['human'],'observations.json',add_person)
    human=inputs['human'];marker={'schema_version':1,'profile':'exercise-body-source-v1','status':'construction_only',
        'simulation_ready':False,'source_sha256':inputs['body']['source_sha256']}
    (human/'body-source.json').write_bytes(encoded(marker))
    manifest=json.loads((human/'manifest.json').read_bytes());manifest['files']['body-source.json']=digest(encoded(marker))
    (human/'manifest.json').write_bytes(encoded(manifest))
    old=tmp_path/'setup';assert prepare(inputs,old)['status']=='needs_input'
    snapshot=json.loads((old/'snapshot.json').read_bytes());ref={'document_sha256':snapshot['observation_sources'][0]['document_sha256'],'object_id':'person-1'}
    request=request_for(old,tmp_path/'request.json',human_entity_ref=ref);out=tmp_path/'choice';result=resume(old,request,out)
    assert result['status']=='needs_input' and result['config_path'] is None
    assert 'person-0 binding' in json.loads((out/'metadata.json').read_bytes())['issues'][0]['message']
    assert len(snapshot['observation_sources'][0]['slots'])==12


def test_construction_only_body_source_retains_its_exact_original_marker(tmp_path):
    inputs=fixture(tmp_path/'inputs',saved_human=True);human=inputs['human'];values={p.name:p.read_bytes() for p in human.iterdir() if p.name!='manifest.json'}
    marker={'schema_version':1,'profile':'exercise-body-source-v1','status':'construction_only','simulation_ready':False,
        'source_sha256':inputs['body']['source_sha256']}
    (human/'body-source.json').write_bytes(encoded(marker))
    manifest=json.loads((human/'manifest.json').read_bytes());manifest.update(profile=marker['profile'],status='construction_only',simulation_ready=False)
    manifest['files']['body-source.json']=digest(encoded(marker));(human/'manifest.json').write_bytes(encoded(manifest))
    out=tmp_path/'setup';assert prepare(inputs,out)['status']=='needs_input'
    assert (out/'sources/human/body-source.json').read_bytes()==encoded(marker)
    result=resume(out,request_for(out,tmp_path/'request.json'),tmp_path/'configured')
    assert result['status']=='config_ready',result
    assert json.loads((out/'snapshot.json').read_bytes())['body']['validation']=='deferred_to_existing_builder'


def test_original_same_source_partition_conflict_still_blocks_ready_config(tmp_path):
    inputs=fixture(tmp_path/'inputs');reseal(inputs['equipment'],'scene-observations.json',lambda d:d['observations'][2].update(partition='audit'))
    old=tmp_path/'setup';assert prepare(inputs,old)['status']=='needs_input'
    out=tmp_path/'configured';result=resume(old,request_for(old,tmp_path/'request.json'),out)
    assert result['status']=='needs_input' and result['config_path'] is None
    assert 'Conflicting canonical frame partitions' in json.loads((out/'metadata.json').read_bytes())['issues'][0]['message']


def test_late_verified_proof_mutation_fails_before_seal(tmp_path):
    inputs=fixture(tmp_path/'inputs',derivative=True)
    def progress(stage,_):
        if stage=='write':(inputs['mapping']/'source.webm').write_bytes(b'changed verified member')
    out=tmp_path/'setup';result=prepare(inputs,out,progress=progress)
    assert result['status']=='failed' and 'Source changed' in result['error']['message'] and not (out/'manifest.json').exists()


def test_maximum_original_clock_keeps_all_1800_slots_and_900_references(tmp_path):
    import time
    inputs=fixture(tmp_path/'inputs',count=1800);out=tmp_path/'setup';start=time.monotonic();result=prepare(inputs,out)
    elapsed=time.monotonic()-start
    assert result['status']=='needs_input',result.get('error')
    assert result['counts']=={'original_source_slots':1800,'decoded_source_slots':1800,'body_references':900,
        'human_entity_slots':1800,'equipment_entity_slots':1800,'metadata_tail_slots':0}
    assert result['input_bytes']<=module.MAX_INPUT_BYTES and (out/'snapshot.json').stat().st_size<=module.MAX_SNAPSHOT_BYTES
    assert elapsed<30 and result['resume_input_budget']['maximum_selected_input_bytes']<=module.MAX_INPUT_BYTES
    print(json.dumps({'authored_maximum_clock_seconds':elapsed,'input_bytes':result['input_bytes'],
        'snapshot_bytes':(out/'snapshot.json').stat().st_size,'resume_maximum_bytes':result['resume_input_budget']['maximum_selected_input_bytes']}))


def test_legacy_derivation_matches_actual_builder_scene_and_exact_input_artifacts(tmp_path):
    inputs=fixture(tmp_path/'inputs');out=tmp_path/'setup';assert prepare(inputs,out)['status']=='needs_input'
    human=inputs['human'];manifest=json.loads((human/'manifest.json').read_bytes())
    expected=adapt_exercise(inputs['pose'],scene=json.loads((human/'scene.json').read_bytes()))
    expected['provenance']['input_artifacts']={name:manifest['files'][name] for name in ('evidence.json','scene.json')}
    assert json.loads((out/'inputs/human-observations.json').read_bytes())==expected


@pytest.mark.parametrize('marker',['human_failure','equipment_failure','absent_lifecycle'])
def test_new_failure_or_lifecycle_marker_before_seal_blocks_publication(tmp_path,marker):
    inputs=fixture(tmp_path/'inputs')
    if marker=='absent_lifecycle':
        lab=tmp_path/'lab';lab.mkdir();inputs['human'].rename(lab/'run');inputs['human']=lab;path=lab/'status.json'
    else:path=inputs['human' if marker=='human_failure' else 'equipment']/'failure.json'
    def progress(stage,_):
        if stage=='write':path.write_bytes(encoded({'status':'failed','reason':'new after complete input reads'}))
    out=tmp_path/'setup';result=prepare(inputs,out,progress=progress)
    assert result['status']=='failed' and not result['sealed'] and not (out/'manifest.json').exists()
    assert 'marker' in result['error']['message'].lower()
