"""Proof archives retain exact immutable evidence without extraction or decoding."""

from hashlib import sha256
from io import BytesIO
import json
import stat
import time
import zipfile

import pytest

from weft import association_geometry_inputs as module


def validate(raw):
    return module.validate_association_archive(raw, expected_sha256=sha256(raw).hexdigest(), deadline=time.monotonic()+2)


def make(entries, *, compression=zipfile.ZIP_STORED, mode=stat.S_IFREG):
    out=BytesIO()
    with zipfile.ZipFile(out,'w') as archive:
        for name,value in entries:
            item=zipfile.ZipInfo(name)
            item.compress_type=compression
            item.external_attr=(mode|0o600)<<16
            archive.writestr(item,value)
    return out.getvalue()


def test_deterministic_archive_roundtrip_retains_all_bytes():
    files={'source/mask.json':b'{"uncertainty":"unresolved"}', 'study/runtime/source.py':b'# source only, never executed\n'}
    first=module._archive(files,time.monotonic()+2)
    assert first==module._archive(files,time.monotonic()+2)
    result=validate(first)
    assert {k:v for k,v in result.items() if k!='inventory.json'}==files


@pytest.mark.parametrize('name',['../secret','/absolute','one//two','one/../two','one\\two'])
def test_archive_rejects_unsafe_paths(name):
    with pytest.raises(ValueError,match='unsafe'):
        validate(make([(name,b'123')]))


def test_duplicate_compressed_symlink_and_oversized_archives_reject(monkeypatch):
    with pytest.warns(UserWarning):
        duplicate=make([('same',b'a'),('same',b'b')])
    for raw in [duplicate,make([('a',b'x'*100)],compression=zipfile.ZIP_DEFLATED),make([('link',b'target')],mode=stat.S_IFLNK)]:
        with pytest.raises(ValueError):validate(raw)
    raw=module._archive({'a':b'payload'},time.monotonic()+2)
    monkeypatch.setattr(module,'MAX_ARCHIVE_BYTES',len(raw)-1)
    with pytest.raises(ValueError,match='archive bytes'):validate(raw)


def test_inventory_hash_and_manifest_pin_are_not_optional():
    raw=make([('a',b'altered'),('inventory.json',json.dumps({'profile':'association-proof-archive-v1','files':{'a':'0'*64}}).encode())])
    with pytest.raises(ValueError,match='inventory differs'):validate(raw)
    raw=module._archive({'a':b'actual'},time.monotonic()+2)
    with pytest.raises(ValueError,match='trusted receipt'):
        module.validate_association_archive(raw,expected_sha256='0'*64,deadline=time.monotonic()+2)


def test_archive_deadline_fails_before_reading_contents():
    raw=module._archive({'a':b'value'},time.monotonic()+2)
    with pytest.raises(TimeoutError):
        module.validate_association_archive(raw,expected_sha256=sha256(raw).hexdigest(),deadline=time.monotonic()-1)


@pytest.fixture
def intake(tmp_path,monkeypatch):
    # Exercise intake joins only; analytic source-fit claims are deliberately stubbed.
    from copy import deepcopy
    from test_exercise_geometry_inputs import Packet,encoded,digest
    from weft.io import canonical_hash
    from weft.shared_exercise_task import GATES_SHA256
    old=Packet(tmp_path/'old')
    base=old.docs['geometry-input/scene.json'];candidate=old.docs['geometry/scene.json'];config=old.docs['geometry-config.json']
    task=deepcopy(old.docs['geometry-input/exercise-task.json']);task['interval']={'horizon_steps':254}
    changed={**task,'scene_sha256':canonical_hash(candidate)}
    source_seal=encoded({'files':{'scene.json':digest(base),'exercise-task.json':digest(task)}})
    archived={'study/manifest.json':b'{"scope":"analytic receipt fixture"}',
        'scene/manifest.json':source_seal,'scene/scene.json':encoded(base),'scene/exercise-task.json':encoded(task)}
    archive=module._archive(archived,time.monotonic()+2)
    proposals=[{'object_id':'weight','status':'conditional_proposal','scope':'analytic source-fit stub'}]
    monkeypatch.setattr(module,'_validate_study',lambda files: ({},deepcopy(config),deepcopy(proposals)))
    receipt={'archive_sha256':digest(archive),'study_manifest_sha256':digest(archived['study/manifest.json']),
        'retained_source_fit':proposals,'selected_objects':['weight'],'base_manifest_sha256':digest(source_seal),
        'base_scene_sha256':canonical_hash(base),'original_task_sha256':digest(task)}
    admission=deepcopy(old.docs['geometry/geometry-hypothesis.json'])
    admission['exercise_task'].update(previous_file_sha256=digest(task),new_file_sha256=digest(changed))
    comparison={'profile':'association-geometry-matched-trials-v1','controller':'authored','seed':1729,
        'reset_options':{'jitter':False,'jitter_scale':0.},'horizon_steps':254,'base_scene_sha256':canonical_hash(base),
        'candidate_scene_sha256':canonical_hash(candidate),'gates_sha256':GATES_SHA256,'trials':[]}
    docs={'association-context.json':receipt,'geometry-config.json':config,'baseline/scene.json':base,
        'geometry/scene.json':candidate,'baseline/exercise-task.json':task,'geometry/exercise-task.json':changed,
        'geometry/geometry-hypothesis.json':admission,'geometry/input-config.json':deepcopy(config),
        'geometry/source-run/association-context.json':receipt,'matched-trials.json':comparison}
    for name,one_task in [('baseline',task),('candidate',changed)]:
        trial={'seed':1729,'reset_options':comparison['reset_options'],'status':'failed','control_steps':1,
               'metrics':{'success':False,'task_sha256':canonical_hash(one_task),'scene_sha256':one_task['scene_sha256'],'gates_sha256':GATES_SHA256}}
        docs['trials/'+name+'.json']=trial
        comparison['trials'].append({'mode':name,'task_sha256':canonical_hash(one_task),'trial_sha256':digest(trial),
                                    **{k:trial[k] for k in ['status','control_steps','metrics']}})
    blobs={'association-context.zip':archive,'geometry/source-run/association-context.zip':archive}
    root=tmp_path/'new'
    def seal():
        data={**{n:encoded(v) for n,v in docs.items()},**blobs}
        data['geometry/manifest.json']=encoded({'schema_version':1,'status':'ready','files':{
            n.removeprefix('geometry/'):digest(v) for n,v in data.items() if n.startswith('geometry/')}})
        manifest=encoded({'schema_version':1,'profile':module.PROFILE,'status':'completed','files':{n:digest(v) for n,v in data.items()}})
        data['manifest.json']=manifest
        for n,v in data.items():
            p=root/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(v)
        return digest(manifest)
    def read():return module.read_association_geometry_inputs(root,expected_manifest_sha256=seal(),deadline=time.monotonic()+2)
    return docs,read


def test_reader_accepts_intact_analytic_receipt_boundary(intake):
    docs,read=intake
    result=read()
    assert result.receipt()['selected_objects']==['weight']
    assert result.receipt()['work']['physics_steps']==0
    assert not any(name.endswith('.zip') for name,_ in result.members)
    assert any(name.endswith('.zip') for name,_ in result.signatures)
    assert result.receipt()['complete_proof']['candidate_member']=='source-run/association-context.zip'
    assert result.receipt()['complete_proof']['sha256']==docs['association-context.json']['archive_sha256']


def test_coordinated_resealed_task_weakening_cannot_replace_archived_task(intake):
    from test_exercise_geometry_inputs import digest
    docs,read=intake
    docs['baseline/exercise-task.json']['gates']['object_speed']=999
    docs['geometry/exercise-task.json']['gates']['object_speed']=999
    docs['association-context.json']['original_task_sha256']=digest(docs['baseline/exercise-task.json'])
    with pytest.raises(ValueError,match='original task gates'):read()


def test_selected_object_broadening_cannot_admit_unconfigured_object(intake):
    docs,read=intake
    docs['association-context.json']['selected_objects'].append('support')
    docs['geometry/geometry-hypothesis.json']['selected_objects'].append('support')
    with pytest.raises(ValueError,match='retained source study receipt'):read()


@pytest.mark.parametrize('change',['config','initial','trial_task','trial_hash','gates'])
def test_ordinary_admission_and_trial_provenance_are_not_optional(intake,change):
    docs,read=intake
    if change=='config':docs['geometry/input-config.json']['assumptions']=['changed']
    if change=='initial':docs['geometry/geometry-hypothesis.json']['initial_geometry_admitted']=False
    if change=='trial_task':docs['matched-trials.json']['trials'][0]['task_sha256']='0'*64
    if change=='trial_hash':docs['matched-trials.json']['trials'][0]['trial_sha256']='0'*64
    if change=='gates':docs['matched-trials.json']['gates_sha256']='0'*64
    with pytest.raises(ValueError):read()


def test_aggregate_source_reservation_rejects_before_third_large_read(tmp_path,monkeypatch):
    from pathlib import Path
    from test_exercise_geometry_inputs import encoded,digest
    source,scene,study=tmp_path/'source',tmp_path/'scene',tmp_path/'study'
    block=b'x'*(16<<20);pin=digest(block)
    source_manifest=encoded({'schema_version':1,'files':{name:pin for name in ['a','b','c']}})
    scene_manifest=encoded({'schema_version':1,'files':{}})
    plan={'source':{'path':str(source),'manifest_sha256':digest(source_manifest)},
          'scene':{'path':str(scene),'manifest_sha256':digest(scene_manifest)}}
    result={'source_members':{str(source/name):{'sha256':pin,'bytes':len(block)} for name in ['a','b','c']}}
    data={'plan.json':encoded(plan),'result.json':encoded(result)}
    study_manifest=encoded({'schema_version':1,'status':'complete','files':{n:digest(v) for n,v in data.items()}})
    manifests={study:study_manifest,source:source_manifest,scene:scene_manifest}
    monkeypatch.setattr(module,'read_package_manifest',lambda root:(manifests[Path(root)],json.loads(manifests[Path(root)])['files']))
    reads=[]
    def read(root,name,**kwargs):
        reads.append((Path(root),name))
        return data[name] if Path(root)==study else block
    monkeypatch.setattr(module,'read_package_file',read)
    with pytest.raises(ValueError,match='reservation'):
        module.prepare_association_study_archive(study,expected_manifest_sha256=digest(study_manifest),deadline=time.monotonic()+2)
    assert (source,'a') in reads and (source,'b') in reads and (source,'c') not in reads


@pytest.mark.parametrize('identity',['task_sha256','scene_sha256','gates_sha256'])
def test_nested_trial_metrics_cannot_be_relabelled_with_updated_trial_hash(intake,identity):
    from test_exercise_geometry_inputs import digest
    docs,read=intake
    docs['trials/baseline.json']['metrics'][identity]='0'*64
    docs['matched-trials.json']['trials'][0]['trial_sha256']=digest(docs['trials/baseline.json'])
    with pytest.raises(ValueError,match='nested trial metric identities'):read()
