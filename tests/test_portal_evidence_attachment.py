"""Attachment custody and imported action boundaries; no inference, physics or network."""
from copy import deepcopy
import fcntl
import json
import os
from pathlib import Path
import time

import pytest

from weft import portal_evidence_attachment as module
from weft.io import digest, write_json
from weft.portal_import import attach_existing_import


def raw(value):
    return (json.dumps(value, sort_keys=True, indent=2) + '\n').encode()


@pytest.fixture
def fixture(tmp_path, monkeypatch):
    source = tmp_path / 'body';source.mkdir()
    root = tmp_path / 'portal';folder = root / ('a'*32);folder.mkdir(parents=True)
    human = {'source_sha256':'b'*64,'reference':[{'time_s':0}]}
    evidence = {'fps':30.,'decoded_frames':2}
    (source / 'evidence.json').write_bytes(raw(evidence))
    observations = {'observations':[{'partition':'construction'}], 'source':{'sha256':'b'*64},
        'provenance':{'adapter':'exercise','input_evidence_sha256':'c'*64,'source_metadata':{},
                      'input_artifacts':{'evidence.json':digest(source/'evidence.json')}},
        'objects':[], 'relationships':[]}
    for name,value in [('human.json',human),('observations.json',observations)]:
        (source/name).write_bytes(raw(value))
    write_json(source/'manifest.json', {'profile':'exercise-body-source-v1','files':{p.name:digest(p) for p in source.iterdir()}})
    observed=deepcopy(observations)
    observed['objects']=[{'authored_role':'active'}]
    observed['relationships']=[{'assumed_joint':'hip'}]
    observed['provenance']['scene']={'authored':True}
    run=folder/'lab/run';run.mkdir(parents=True)
    write_json(run/'observed.json',observed);obs_hash=digest(run/'observed.json')
    scene={'human_spec':human,'coordinate_frame':{'clock_binding':{'source_fps':30.}},
           'source_binding':{'total_source_frames':2,'entity_bindings':[{'world_entity_id':'human','observation_entity_id':'person-0','observation_document_sha256':obs_hash}],
                             'evidence_documents':{obs_hash:'observed.json'}}}
    write_json(run/'scene.json',scene)
    world={'run':'lab/run','source_sha256':'b'*64,'manifest_sha256':'d'*64,'input_sha256':{'scene.json':digest(run/'scene.json')}}
    write_json(folder/'job.json', {'id':'a'*32,'kind':'imported_lab','source_sha256':'b'*64})
    write_json(folder/'status.json', {'status':'ready'})
    import weft.portal_scene_context as context
    import weft.shared_world_evidence as sources
    import weft.observations as adapters
    monkeypatch.setattr(context,'construction_context',lambda root: ({'closed_context':True},deepcopy(world)))
    monkeypatch.setattr(sources,'verify_shared_world_sources',lambda *a,**k: {})
    monkeypatch.setattr(adapters,'validate_observations',lambda x:x)
    monkeypatch.setattr(module,'_validate_body',lambda body,files,**kwargs: (json.loads(files['human.json']),json.loads(files['observations.json']),{'kind':'fixture'}))
    return source,root,folder,scene,observations,world


def test_attachment_preserves_source_world_and_enables_only_metadata_summary(fixture, monkeypatch):
    source,root,folder,*_=fixture
    before={str(p):p.read_bytes() for base in (source,folder/'lab') for p in base.rglob('*') if p.is_file()}
    result=attach_existing_import(root,folder.name,source)
    assert result['status']=='attached' and result['api_calls']==0 and result['repair_qualified'] is False
    assert all(Path(p).read_bytes()==b for p,b in before.items())
    body,receipt=module.validate_attachment(folder)
    assert body==folder/'review-evidence/body' and receipt['human_sha256']
    job=json.loads((folder/'job.json').read_bytes())
    monkeypatch.setattr(module,'_validate_body',lambda *a,**k:pytest.fail('Listing cannot execute source validation'))
    assert module.attachment_summary(folder,job)['available']
    (folder/'review-evidence/receipt.json').write_bytes(b'{}')
    assert not module.attachment_summary(folder,job)['available']


@pytest.mark.parametrize('mutation',['human','source','clock','partitions','measurements','provenance','evidence_pin'])
def test_semantic_source_join_rejects_even_coordinated_metadata_changes(fixture,mutation):
    source,root,folder,scene,observations,world=fixture
    files={p.name:p.read_bytes() for p in source.iterdir() if p.name!='manifest.json'}
    human=json.loads(files['human.json'])
    if mutation=='human': human['reference'][0]['time_s']=.1
    elif mutation=='source':human['source_sha256']='e'*64
    elif mutation=='clock':files['evidence.json']=raw({'fps':29.,'decoded_frames':2})
    elif mutation=='partitions':observations['observations'][0]['partition']='audit'
    elif mutation=='measurements':observations['observations'][0]['measured']='invented'
    elif mutation=='provenance':observations['provenance']['source_metadata']['new']='invented'
    else: observations['provenance']['input_artifacts']['evidence.json']='e'*64
    with pytest.raises(ValueError):module._join(folder,files,human,observations,time.monotonic()+10)


def test_even_matching_withheld_partition_cannot_enter_request(fixture):
    source,root,folder,scene,observations,world=fixture
    observations['observations'][0]['partition']='audit'
    path=folder/'lab/run/observed.json';write_json(path,observations);pin=digest(path)
    scene['source_binding']['entity_bindings'][0]['observation_document_sha256']=pin
    scene['source_binding']['evidence_documents']={pin:'observed.json'}
    write_json(folder/'lab/run/scene.json',scene);world['input_sha256']['scene.json']=digest(folder/'lab/run/scene.json')
    files={p.name:p.read_bytes() for p in source.iterdir() if p.name!='manifest.json'}
    with pytest.raises(ValueError,match='withheld'):
        module._join(folder,files,scene['human_spec'],observations,time.monotonic()+10)


@pytest.mark.parametrize('mutation',['body','receipt','scene'])
def test_action_time_validation_rejects_changed_artifacts(fixture,mutation):
    source,root,folder,*_=fixture
    attach_existing_import(root,folder.name,source)
    path={'body':folder/'review-evidence/body/evidence.json','receipt':folder/'review-evidence/receipt.json','scene':folder/'lab/run/scene.json'}[mutation]
    path.write_bytes(b'{}')
    with pytest.raises(ValueError):module.validate_attachment(folder)


def test_portal_lease_and_existing_history_not_bypassed(fixture):
    source,root,folder,*_=fixture
    fd=os.open(root/'portal.lock',os.O_CREAT|os.O_RDWR,0o600)
    fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
    try:
        with pytest.raises(ValueError,match='Stop the portal'):attach_existing_import(root,folder.name,source)
    finally:os.close(fd)
    (folder/'muse').mkdir()
    with pytest.raises(ValueError,match='history'):attach_existing_import(root,folder.name,source)
    assert not (folder/'review-evidence').exists()


def test_imported_server_guard_only_allows_attached_review(fixture):
    from weft.portal import Portal
    from types import SimpleNamespace
    source,root,folder,*_=fixture
    server=SimpleNamespace(jobs={folder.name:json.loads((folder/'job.json').read_bytes())},_folder=lambda _:folder,_status=lambda _: {})
    with pytest.raises(ValueError,match='attachment'):Portal._require_review_body(server,folder.name)
    attach_existing_import(root,folder.name,source)
    server.jobs[folder.name]=json.loads((folder/'job.json').read_bytes())
    Portal._require_review_body(server,folder.name)
    for action in ['checks','repairs']:
        with pytest.raises(ValueError,match='review only'):Portal._require_review_body(server,folder.name,action=action)


def test_validation_revision_cannot_be_used_as_interactive_runtime(tmp_path):
    from weft.portal_lab_runtime import prepare_runtime, VALIDATION_COMMIT, TRUSTED_RELEASES
    assert VALIDATION_COMMIT not in TRUSTED_RELEASES
    with pytest.raises(ValueError,match='fixed trusted release'):
        prepare_runtime(tmp_path,trusted_commit=VALIDATION_COMMIT)


def test_package_links_and_budget_rejected_before_reading_arbitrary_bytes(fixture,monkeypatch):
    source,*_=fixture
    (source/'evidence.json').rename(source/'outside.json')
    (source/'evidence.json').symlink_to(source/'outside.json')
    with pytest.raises(ValueError):module._body_files(source,time.monotonic()+10)


def test_exercise_advisor_cannot_select_alternate_generation_payload(fixture):
    source,root,folder,scene,observations,world=fixture
    files={p.name:p.read_bytes() for p in source.iterdir() if p.name!='manifest.json'}
    files['evidence.json']=raw({'fps':30.,'decoded_frames':2,'generation':[{'withheld':'DO_NOT_SEND'}]})
    observations['provenance']['input_artifacts']['evidence.json']=module.sha256(files['evidence.json']).hexdigest()
    path=folder/'lab/run/observed.json';write_json(path,observations);pin=digest(path)
    scene['source_binding']['entity_bindings'][0]['observation_document_sha256']=pin
    scene['source_binding']['evidence_documents']={pin:'observed.json'}
    write_json(folder/'lab/run/scene.json',scene);world['input_sha256']['scene.json']=digest(folder/'lab/run/scene.json')
    with pytest.raises(ValueError,match='alternate generation'):
        module._join(folder,files,scene['human_spec'],observations,time.monotonic()+10)


def test_interrupted_unregistered_attachment_preserved_not_silently_rebound(fixture):
    source,root,folder,*_=fixture
    (folder/'review-evidence').mkdir()
    (folder/'review-evidence/partial.txt').write_bytes(b'preserve')
    with pytest.raises(ValueError,match='Unregistered attachment preserved'):
        attach_existing_import(root,folder.name,source)
    assert (folder/'review-evidence/partial.txt').read_bytes()==b'preserve'


def test_historical_validator_timeout_reaps_child_and_receives_no_keys(tmp_path,monkeypatch):
    import subprocess
    runtime=tmp_path/'runtime';(runtime/'src/weft').mkdir(parents=True)
    source=b'# trusted test fixture; never executed\n'
    (runtime/'src/weft/pose_source_recovery.py').write_bytes(source)
    files={'pose-recovery.json':b'{}','recovery/implementation.json':raw({'files':{'pose_source_recovery.py':module.sha256(source).hexdigest()}}),
           'recovery/implementation-source/pose_source_recovery.py':source}
    monkeypatch.setattr(module,'prepare_validation_runtime',lambda _:runtime)
    monkeypatch.setenv('OPENAI_API_KEY','DO_NOT_COPY');monkeypatch.setenv('META_API_KEY','DO_NOT_COPY')
    calls=[]
    class Child:
        def wait(self,timeout=None):
            calls.append(('wait',timeout))
            if timeout is not None: raise subprocess.TimeoutExpired('fixed driver',timeout)
            return -9
        def kill(self):calls.append(('kill',))
    def start(command,**kwargs):
        assert command[2]==module.DRIVER and command[2]!=source.decode()
        assert 'OPENAI_API_KEY' not in kwargs['env'] and 'META_API_KEY' not in kwargs['env']
        return Child()
    monkeypatch.setattr(module.subprocess,'Popen',start)
    with pytest.raises(subprocess.TimeoutExpired):module._validate_body(tmp_path,files,project=tmp_path,deadline=time.monotonic()+10)
    assert [c[0] for c in calls]==['wait','kill','wait']


def test_unreviewed_artifact_code_cannot_select_validation_runtime(tmp_path,monkeypatch):
    runtime=tmp_path/'runtime';(runtime/'src/weft').mkdir(parents=True)
    (runtime/'src/weft/pose_source_recovery.py').write_bytes(b'trusted')
    injected=b'raise RuntimeError("artifact code must not execute")'
    files={'pose-recovery.json':b'{}','recovery/implementation.json':raw({'files':{'pose_source_recovery.py':module.sha256(injected).hexdigest()}}),
           'recovery/implementation-source/pose_source_recovery.py':injected}
    monkeypatch.setattr(module,'prepare_validation_runtime',lambda _:runtime)
    monkeypatch.setattr(module.subprocess,'Popen',lambda *a,**k:pytest.fail('No child for unreviewed code'))
    with pytest.raises(ValueError,match='pinned SHA256'):module._validate_body(tmp_path,files,project=tmp_path,deadline=time.monotonic()+10)


def prepared_fixture(fixture):
    from weft.io import canonical_hash
    source,root,folder,*_=fixture
    attach_existing_import(root,folder.name,source)
    context={'source_sha256':'b'*64,'simulation_world':{'closed_context':True}}
    request={'model':'fixture','messages':[{}, {'content':[{'text':json.dumps(context)}]}]}
    call={'status':'prepared','request_count':0,'model':'fixture','source_sha256':'b'*64,
          'request_sha256':canonical_hash(request),'simulation_world_sha256':canonical_hash(context['simulation_world'])}
    prepared=folder/'review-evidence/prepared';prepared.mkdir()
    for name,value in [('context.json',context),('request.json',request),('advisor-call.json',call)]:write_json(prepared/name,value)
    return {name:digest(prepared/name) for name in module.PREPARED_FILES}


def test_exact_public_attachment_artifacts_and_payload_pins(fixture):
    from types import SimpleNamespace
    from weft.portal import Portal
    from weft.portal_import import register_prepared_review
    _,root,folder,*_=fixture
    pins=prepared_fixture(fixture)
    registered=register_prepared_review(root,folder.name,expected_files=pins)
    assert registered['api_calls']==registered['decodes']==0
    job=json.loads((folder/'job.json').read_bytes());server=SimpleNamespace(jobs={folder.name:job},_folder=lambda _:folder)
    for name in module.PUBLIC_ARTIFACTS:
        assert Portal.artifact(server,folder.name,name)==folder/name
    assert module.attachment_summary(folder,job)['prepared']
    for name in ['review-evidence/body/human.json','review-evidence/body/evidence.json','review-evidence/prepared/advisor-call.json',
                 'review-evidence/prepared/manifest.json','review-evidence/prepared/../body/human.json','review-evidence/../job.json']:
        with pytest.raises(ValueError):Portal.artifact(server,folder.name,name)


@pytest.mark.parametrize('change',['receipt','manifest','context','request','coordinated_request','link','wrong_kind'])
def test_attachment_download_rejects_corruption_and_coordinated_payload_rehash(fixture,change):
    from weft.portal_import import register_prepared_review
    _,root,folder,*_=fixture
    pins=prepared_fixture(fixture);register_prepared_review(root,folder.name,expected_files=pins)
    job=json.loads((folder/'job.json').read_bytes());prepared=folder/'review-evidence/prepared'
    name='review-evidence/prepared/context.json'
    if change=='receipt':(folder/'review-evidence/receipt.json').write_bytes(b'{}')
    elif change=='manifest':(prepared/'manifest.json').write_bytes(b'{}')
    elif change in {'context','request'}:
        (prepared/(change+'.json')).write_bytes(b'{}');name='review-evidence/prepared/'+change+'.json'
    elif change=='coordinated_request':
        (prepared/'request.json').write_bytes(b'{}');m=json.loads((prepared/'manifest.json').read_bytes());m['files']['request.json']=digest(prepared/'request.json');write_json(prepared/'manifest.json',m);name='review-evidence/prepared/request.json'
    elif change=='link':
        (prepared/'context.json').rename(prepared/'elsewhere.json');(prepared/'context.json').symlink_to(prepared/'elsewhere.json')
    else:job['kind']='video'
    with pytest.raises(ValueError):module.attachment_artifact(folder,job,name)


def test_unregistered_and_wrong_frozen_preparation_rejected(fixture):
    from weft.portal_import import register_prepared_review
    _,root,folder,*_=fixture
    pins=prepared_fixture(fixture);job=json.loads((folder/'job.json').read_bytes())
    with pytest.raises(ValueError,match='no registered'):module.attachment_artifact(folder,job,'review-evidence/prepared/request.json')
    assert not module.attachment_summary(folder,job)['prepared']
    pins['request.json']='f'*64
    with pytest.raises(ValueError,match='pinned SHA256'):register_prepared_review(root,folder.name,expected_files=pins)
    assert not (folder/'review-evidence/prepared/manifest.json').exists()
