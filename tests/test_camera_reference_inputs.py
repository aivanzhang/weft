"""Authored saved measurements exercise custody, not geometry or source accuracy."""

import builtins
from copy import deepcopy
from dataclasses import FrozenInstanceError
from hashlib import sha256
import json
import socket
import subprocess

import pytest

from test_camera_body_reference_report import camera_documents, repin_overlay
from test_whole_body_reference_report import encode, save_study
from test_shared_exercise_reference_inputs import reseal
from weft import camera_reference_inputs as camera
from weft import shared_exercise_reference_inputs as shared
from weft import whole_body_reference_report as report


@pytest.fixture(autouse=True)
def metadata_only(monkeypatch):
    importing = builtins.__import__
    def guarded(name, *args, **kwargs):
        assert name.split('.')[0] not in {'numpy', 'mujoco', 'gymnasium', 'cv2', 'mediapipe', 'torch', 'PIL'}
        assert name not in {'weft.camera_body_fit', 'weft.whole_body_reference_refinement',
                            'weft.source_surface_envelopes', 'weft.source_surface_interval_construction'}
        return importing(name, *args, **kwargs)
    def denied(*args, **kwargs):
        pytest.fail('Metadata loader attempted process or network work')
    monkeypatch.setattr(builtins, '__import__', guarded)
    monkeypatch.setattr(socket.socket, 'connect', denied)
    monkeypatch.setattr(socket, 'create_connection', denied)
    monkeypatch.setattr(subprocess, 'Popen', denied)


def passing_camera_documents():
    documents = camera_documents()
    saved, overlay = documents['report.json'], documents['angle-proposal.json']
    saved.update(status='unadmitted_construction_proposal', failures=[], budget_reason=None,
                 gates=dict.fromkeys(camera.PROPOSAL_GATES, True))
    overlay['whole_body_reference_refinement']['status'] = saved['status']
    saved['contract'].update(sample_hz=30, maximum_grid_samples=1801,
                            maximum_endpoint_depth_change_model_units=.08, numerical_tolerance=1e-8)
    template = deepcopy(saved['geometry_checks'][0])
    template.update(before_floor_support_gaps=[-.02, .1, .2], after_floor_support_gaps=[-.02, .1, .2],
                    before_max_penetration_model_units=.02, after_max_penetration_model_units=.02)
    saved['geometry_checks'] = [dict(template, kind='source_pose', time_s=float(i), source_reference_index=i)
                               for i in range(3)]
    saved['geometry_checks'] += [dict(template, kind='control_grid', time_s=i/30, source_reference_index=None)
                                for i in range(61)]
    saved['floor_summary'].update(checks=64, source_pose_checks=3, control_grid_checks=61,
        before_failures=0, after_failures=0, before_maximum_penetration_model_units=.02,
        after_maximum_penetration_model_units=.02)
    repin_overlay(documents)
    return documents


def packages(tmp_path, *, status='ready'):
    documents = passing_camera_documents()
    proposal, source = tmp_path/'proposal', tmp_path/'scene'
    proposal_pin = save_study(proposal, documents)
    human, coupling = documents['inputs/human.json'], documents['report.json']['camera_fit']['coupling']
    scene = {'schema_version':1, 'profile':'shared-world-v6', 'physical_validation':'unverified',
        'controlled_id':'human', 'human_spec':deepcopy(human), 'camera_coupling':deepcopy(coupling),
        'objects':[], 'target':None, 'learned_policy_available':False, 'dt':.5,
        'source_binding':{'mode':'source_qualified', 'source_sha256':human['source_sha256'], 'total_source_frames':5},
        'task':{'start_offset_s':0., 'horizon_steps':4, 'object_goal':None}}
    task = {'schema_version':1, 'profile':'camera-shared-task-v1', 'scene_sha256':report._canonical(scene),
        'interval':{'start_offset_s':0.,'horizon_steps':4}, **dict.fromkeys(camera.IDENTITIES, 'd'*64)}
    task.update(camera_coupling_sha256=report._canonical(coupling), body_sha256=report._canonical(human),
                source_sha256=human['source_sha256'], source_clock_sha256=coupling['source_clock_sha256'])
    source.mkdir()
    files = {'scene.json':encode(scene), 'exercise-task.json':encode(task), 'camera-coupling.json':encode(coupling),
             **{name:encode(documents[alias]) for name,alias in camera._ORIGINAL_ALIASES.items()}}
    for name, raw in files.items(): (source/name).write_bytes(raw)
    manifest = encode({'schema_version':1, 'profile':'camera-shared-candidate-v1', 'status':status,
        'files':{name:sha256(raw).hexdigest() for name,raw in files.items()} | {'unselected-video.mp4':'e'*64}})
    (source/'manifest.json').write_bytes(manifest)
    return proposal, source, source/'exercise-task.json', proposal_pin, sha256(manifest).hexdigest()


@pytest.mark.parametrize('status', ['ready', 'constructed'])
def test_actual_saved_model_and_exact_original_views_without_native(tmp_path, monkeypatch, status):
    proposal, source, task, ppin, spin = packages(tmp_path, status=status)
    calls=[]; model=report._model
    def inspect(documents, hashes, check):
        calls.append(set(documents))
        return model(documents,hashes,check)
    monkeypatch.setattr(report,'_model',inspect)
    value=camera.load_camera_reference_inputs(proposal,source,task,
        expected_proposal_manifest_sha256=ppin,expected_scene_manifest_sha256=spin)
    assert isinstance(value,camera.CameraReferenceInputs) and len(calls)==1
    assert set(report._CAMERA_EXTRAS)<=calls[0] and 'inputs/surface-config.json' in calls[0]
    assert value.original_human_bytes==(source/'human.json').read_bytes()
    assert value.angle_proposal_bytes==(proposal/'angle-proposal.json').read_bytes()
    assert not hasattr(value,'candidate_human_bytes')
    assert not any('candidate_human' in k or k=='candidate_canonical_sha256' for k in value.identities)
    receipt=value.receipt()
    assert len(receipt['saved_gates'])==14 and all(receipt['saved_gates'].values())
    assert receipt['input_bytes_read']==2*sum(r['bytes'] for r in receipt['selected_files'])
    assert receipt['body_installed'] is False and receipt['scene_completion']['status']==status
    assert {k:sha256(v).hexdigest() for k,v in value.proposal_members.items()}==receipt['selected_proposal_members']
    assert not any('unselected' in r['name'] for r in receipt['selected_files'])
    with pytest.raises(FrozenInstanceError): value.scene_bytes=b'changed'
    with pytest.raises(TypeError): value.proposal_members['new']=b'changed'


def test_default_legacy_loader_still_rejects_camera_overlay(tmp_path):
    args=packages(tmp_path)[:3]
    with pytest.raises(ValueError,match='proposal lacks required'): shared.load_reference_inputs(*args)
    value=shared.load_reference_inputs(*args,allow_camera=True)
    assert isinstance(value,camera.CameraReferenceInputs)


@pytest.mark.parametrize('kind',['gate','incomplete_grid','missing_original','changed_camera','task_interval','wrong_scene'])
def test_resealed_changes_fail_closed(tmp_path,kind):
    proposal,source,task,*_=packages(tmp_path)
    if kind in {'gate','incomplete_grid'}:
        saved=json.loads((proposal/'report.json').read_bytes())
        if kind=='gate': saved['gates']['camera_projection_domain']=False
        else:
            saved['geometry_checks'].pop()
            saved['floor_summary']['checks']-=1; saved['floor_summary']['control_grid_checks']-=1
        reseal(proposal,'report.json',saved)
    elif kind=='missing_original':
        seal=json.loads((proposal/'manifest.json').read_bytes())
        del seal['files']['inputs/source-camera.json']
        (proposal/'manifest.json').write_bytes(encode(seal))
    elif kind=='changed_camera':
        raw=(source/'camera.json').read_bytes()+b' '
        reseal(source,'camera.json',raw)
    elif kind=='task_interval':
        value=json.loads(task.read_bytes()); value['interval']['horizon_steps']=3
        reseal(source,'exercise-task.json',value)
    else:
        value=json.loads((source/'scene.json').read_bytes()); value['human_spec']['reference'][1]['joints']['left_elbow']=.3
        reseal(source,'scene.json',value)
    with pytest.raises(ValueError): camera.load_camera_reference_inputs(proposal,source,task)


@pytest.mark.parametrize('kind',['expected_pin','aggregate','source_change'])
def test_shared_read_limits_and_final_recheck_apply_to_camera(tmp_path,monkeypatch,kind):
    args=packages(tmp_path)[:3]
    kwargs={}
    if kind=='expected_pin': kwargs['expected_proposal_manifest_sha256']='0'*64
    elif kind=='aggregate': kwargs['max_input_bytes']=1024
    else:
        model=report._model
        def changed(documents,hashes,check):
            result=model(documents,hashes,check)
            path=args[0]/'inputs/human.json'; path.write_bytes(path.read_bytes()+b' ')
            return result
        monkeypatch.setattr(report,'_model',changed)
    with pytest.raises(ValueError): camera.load_camera_reference_inputs(*args,**kwargs)
