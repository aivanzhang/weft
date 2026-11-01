"""Pure saved two-view report checks; no fitter, projection or native admission."""

import builtins
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
import re
import socket
import subprocess
import time

import pytest

from weft import whole_body_reference_report as report
from test_whole_body_reference_report import encode, fixture_documents
from test_whole_body_reference_interval_report import interval_documents
from test_camera_body_reference_report import camera_documents
from test_whole_body_reference_report import save_study


@pytest.fixture(autouse=True)
def metadata_only(monkeypatch):
    importing = builtins.__import__
    def guard(name, *args, **kwargs):
        assert name.split('.')[0] not in {'numpy', 'mujoco', 'cv2', 'mediapipe', 'torch', 'pymunk'}, name
        assert name not in {'weft.camera_multiview_fit', 'weft.camera_body_fit', 'weft.camera_reference',
                            'weft.camera_shared_contract', 'weft.whole_body_reference_refinement',
                            'weft.source_surface_envelopes', 'weft.source_surface_interval_construction'}, name
        return importing(name, *args, **kwargs)
    monkeypatch.setattr(builtins, '__import__', guard)
    monkeypatch.setattr(socket.socket, 'connect', lambda *a, **k: pytest.fail('network'))
    popen = subprocess.Popen
    def process(args, *a, **k):
        assert type(args) is list and args[0] == 'node', 'Only a local scripted DOM process is allowed'
        return popen(args, *a, **k)
    monkeypatch.setattr(subprocess, 'Popen', process)


def default_projection(kind, implementation=report):
    documents = {'image-first': lambda: fixture_documents(), 'floor-first': lambda: fixture_documents(True),
                 'surface': interval_documents, 'single-camera': camera_documents}[kind]()
    hashes = {name: sha256(encode(value)).hexdigest() for name, value in documents.items()}
    model = implementation._model(documents, hashes, lambda: None)
    if kind == 'floor-first':
        # The fixture builds counters from a set; freeze its insertion order for a byte golden.
        model['objective']['counters'] = json.loads(json.dumps(model['objective']['counters'], sort_keys=True))
    model.update(external_manifest_pin_supplied=True, source_manifest_sha256='f'*64,
                 created_at='2026-10-31 22:10:00 PDT')
    members = ({'angle-proposal.json': 'angle-proposal.json.txt', **{k:v for k,v in report.MEMBERS.items()
                if k != 'candidate-human.json'}} if kind == 'single-camera' else dict(report.MEMBERS))
    return implementation._bytes(model), implementation._render(model, ':root{}', members)


GOLDEN = {
    'image-first': ['61b4c9d34d9d646dc21efb6bea4dfbd4395be7feb15cb52d827c62a9640b08b2',
                    '4d2e5e622550934660e92c12e8fcd708f3416f58f7556273fada0ab38f58619d'],
    'floor-first': ['cfb97b2b17e5c7a1d9b5e99c1591e66df65d17039fa648ca6e9f7f62c1ffbb8c',
                    'c94906e735d480445990599af0f00510617c30016796a1eb62bb920ac5d0cb25'],
    'surface': ['dcf37d03e5cf7a8a926ec323b43c05058439bf36c6f44f9cad86fa15ad9fde57',
                '2c12d396609036b7a8e136561fbb3b618402db0d45edb83e3d06a6c086752ec2'],
    'single-camera': ['7dc795ecc90012b548bfacc156b4222682a55a5923c76ed19832326319c2417e',
                      'e7628f2cf813da5e9f75ac1f1d324147582c32cee54d7d84e8829452b1a84eb9'],
}


@pytest.mark.parametrize('kind', ['image-first', 'floor-first', 'surface', 'single-camera'])
def test_default_model_and_render_bytes_remain_identical_without_optional_import(monkeypatch, kind):
    importing = builtins.__import__
    def guard(name, *args, **kwargs):
        assert name not in {'weft.camera_multiview_report', 'weft.camera_multiview_inputs'}, 'Default imported optional multiview code'
        return importing(name, *args, **kwargs)
    monkeypatch.setattr(builtins, '__import__', guard)
    actual = [sha256(value).hexdigest() for value in default_projection(kind)]
    assert actual == GOLDEN[kind]


def multiview_documents(*, excluded_secondary=None, missing_secondary=None, secondary_radial=.1, secondary_clock_shift=0.):
    """Portable authored saved records; source-proof joins are local fixture declarations."""
    from weft import camera_multiview_report as multi
    from weft.camera_multiview_inputs import _descriptor
    d = camera_documents(); body=d['inputs/human.json']; saved=d['report.json']; fit=saved['camera_fit']
    digest=lambda value:sha256(encode(value)).hexdigest()
    old={key:value for key,value in (('body',digest(body)),('canonical_body',report._canonical(body)),
        ('binding',report._canonical(body['camera_reference'])),('coupling',report._canonical(fit['coupling'])),
        ('surface_binding',digest(d['inputs/surface-body-binding.json'])),('surface_config',digest(d['inputs/surface-config.json'])))}
    source=body['source_sha256']; secondary='b'*64
    camera=json.loads(json.dumps(d['inputs/surface-camera.json']).replace(source,secondary))
    camera['radial_k1']=secondary_radial
    for frame in camera['frames']:frame['t']=[1.,0.,2.]
    camera['frames'][1]['time_s']+=secondary_clock_shift
    evidence=json.loads(json.dumps(d['inputs/evidence-derived.json']).replace(source,secondary))
    evidence['frames']=evidence['frames'][:5];evidence['decoded_frames']=5
    for frame in evidence['frames']:frame['decoder_timestamp_s']=frame['time_s']
    evidence['frames']=[row for row in evidence['frames'] if row['frame']!=missing_secondary]
    views=[{'name':'primary','source_sha256':source,'camera_sha256':report._canonical(d['inputs/surface-camera.json']),
            'evidence_sha256':report._canonical(d['inputs/evidence-derived.json']),'clock_provenance':'Explicit authored fixture clock'},
           {'name':'secondary','source_sha256':secondary,'camera_sha256':report._canonical(camera),
            'evidence_sha256':report._canonical(evidence),'clock_provenance':'Explicit authored fixture clock'}]
    root=body['camera_reference']['root_refinement'];root['views']=views
    for row in root['roots']:row['source_evidence_ids']=[f'{source}:frame:{row["frame"]}',f'{secondary}:frame:{row["frame"]}']
    def replace(value,mapping):
        if type(value) is str:return mapping.get(value,value)
        if type(value) is list:return [replace(item,mapping) for item in value]
        if type(value) is dict:return {mapping.get(key,key):replace(item,mapping) for key,item in value.items()}
        return value
    d=replace(d,{old['body']:digest(body),old['canonical_body']:report._canonical(body),old['binding']:report._canonical(body['camera_reference'])})
    d=replace(d,{old['coupling']:report._canonical(d['report.json']['camera_fit']['coupling']),
                 old['surface_binding']:digest(d['inputs/surface-body-binding.json'])})
    d=replace(d,{old['surface_config']:digest(d['inputs/surface-config.json'])})
    body,saved=d['inputs/human.json'],d['report.json'];fit=saved['camera_fit'];root=body['camera_reference']['root_refinement']
    observation=json.loads(json.dumps(d['inputs/surface-observation-00.json']).replace(source,secondary))
    observation['source'].update(decoded_frames=5,metadata_frames=5,width_px=100,height_px=80)
    observation['observations']=observation['observations'][:5];observation['evidence']=observation['evidence'][:5]
    for row in observation['observations']:
        if row['frame']==excluded_secondary:row['partition']='validation'
    observation['provenance']={'adapter':'exercise','input_evidence_sha256':report._canonical(evidence)}
    secondary_entity={'document_sha256':digest(observation),'object_id':'person'}
    relation={'primary_entity_ref':d['inputs/surface-body-binding.json']['human_entity_ref'],
        'secondary_entity_ref':secondary_entity,'status':'assumed','method':'Authored same-person test correspondence',
        'uncertainty':'Not independently verified source identity or physical anatomy'}
    claim={'status':'assumed','method':'Explicit fixture same-world/clock map','uncertainty':'Authored, unverified','evidence_refs':[]}
    alignment={'schema_version':1,'profile':'world-alignment-v1',
        'from_frame':{'id':'camera-world:'+report._canonical(camera),'length_unit':'m','axes':'x_right_y_depth_z_up'},
        'to_frame':{'id':'camera-world:'+fit['input_bindings']['camera_canonical_sha256'],'length_unit':'m','axes':'x_right_y_depth_z_up'},
        'spatial':{'scale':1.,'rotation_wxyz':[1.,0.,0.,0.],'translation':[0.,0.,0.]},
        'clock':{'rate':1.,'source_origin_s':0.,'target_origin_s':0.},'claims':{'spatial':deepcopy(claim),'clock':deepcopy(claim)}}
    config={'schema_version':1,'profile':'camera-secondary-view-fit-config-v1','primary_source_sha256':source,
        'secondary_source_sha256':secondary,'observations':[{'path':'secondary-observation.json','sha256':digest(observation)}],
        'world_alignment':{'path':'alignment.json','sha256':digest(alignment)},'correspondence':relation,'image_scale':.01,'weight':3.}
    d.update({'inputs/secondary-camera.json':camera,'inputs/secondary-evidence.json':evidence,
        'inputs/camera-secondary-observation-00.json':observation,'inputs/camera-secondary-alignment.json':alignment,
        'inputs/camera-secondary-config.json':config})
    metadata=[{'view_id':view['name'],'source_sha256':view['source_sha256'],
        'image_loss_scale_per_pixel':fit['image_loss_scale_per_pixel'] if index==0 else .01,
        'authored_view_weight':1. if index==0 else 3.} for index,view in enumerate(views)]
    slots=[]
    for frame in range(5):
        pose=next((row for row in evidence['frames'] if row['frame']==frame),None)
        identifier=f'{secondary}:frame:{frame}';stamp=frame/2
        index={0:0,2:1,4:2}.get(frame);reference=None if index is None else {'reference_index':index,'evidence_id':identifier,'body_time_s':stamp}
        excluded=frame==excluded_secondary or pose is None
        admission=None if index is None else {'schema_version':1,'profile':'source-surface-body-image-admission-v1',
            'source_frame':frame,'status':'unavailable' if excluded else 'construction_image_source_ready',
            'reason':'secondary_pose_not_sampled' if pose is None else 'not_construction_source_evidence' if excluded else None,'human_entity_ref':secondary_entity,
            'body_reference':reference,'evidence_ref':{'document_sha256':secondary_entity['document_sha256'],'evidence_id':identifier},
            'observation_ref':{'document_sha256':secondary_entity['document_sha256'],'observation_id':f'person:{frame}'},
            'source_time_s':stamp,'source_partition':'validation' if frame==excluded_secondary else 'construction'}
        slots.append({'source_frame':frame,'source_status':'decoded','camera_time_s':camera['frames'][frame]['time_s'],'source_time_s':pose['time_s'] if pose else None,
            'decoder_timestamp_s':pose['decoder_timestamp_s'] if pose else None,'evidence_id':pose['id'] if pose else None,'reference_index':index,'status':pose['status'] if pose else 'unsampled',
            'source_partition':'validation' if frame==excluded_secondary else 'construction','image_source_admission':admission,'target_origin':None,
            'landmarks_used':index is not None and not excluded,'camera_status':'supplied'})
    for reference in saved['references']:
        points=reference['points'];extra=deepcopy(points)
        for index,group in enumerate((points,extra)):
            for point in group:
                if index and reference['frame'] in (excluded_secondary,missing_secondary):
                    point.update(weight=0.,source_confidence=None,status='unqualified_construction_image_source',
                        landmark_source='not_read_unqualified_source',error_before_px=None,error_after_px=None)
                    point.pop('target_px',None);point.pop('target_authored_loss_units',None)
                point.update(view_id=metadata[index]['view_id'],confidence_weight=point['weight'],authored_view_weight=metadata[index]['authored_view_weight'],
                             objective_weight=point['weight']*metadata[index]['authored_view_weight'])
                if point['weight']:point['target_authored_loss_units']=[value*metadata[index]['image_loss_scale_per_pixel'] for value in point['target_px']]
        points.extend(extra)
    known=[sum(p['weight']>0 for row in saved['references'] for p in row['points'][i*12:(i+1)*12]) for i in range(2)]
    means=[[sum(p['error_'+side+'_px'] for row in saved['references'] for p in row['points'][i*12:(i+1)*12] if p['weight'])/known[i]
            for side in ('before','after')] for i in range(2)]
    qualification={'status':'conditional_two_view_ready','reason':None,'known_targets_by_view':known,'missing_targets_by_view':[36-n for n in known],
        'source_slots_by_view':[6,5],'secondary_source_slots_canonical_sha256':report._canonical(slots),
        'original_root_source_binding_canonical_sha256':report._canonical(root['roots']),
        'minimum_camera_baseline_supplied_units':1.,'maximum_camera_baseline_supplied_units':1.,
        'secondary_declared_frames':5,'secondary_decoded_frames':5,'secondary_pose_rows':len(evidence['frames'])}
    binding={'schema_version':1,'profile':multi.EXTENSION,'primary_input_bindings_canonical_sha256':report._canonical(fit['input_bindings']),
        'views':metadata,'secondary_original_view':views[1], 'secondary_input_bindings':{
            'camera_raw_sha256':digest(camera),'evidence_raw_sha256':digest(evidence),'camera_canonical_sha256':report._canonical(camera),
            'evidence_canonical_sha256':report._canonical(evidence),'observation_document_sha256s':[digest(observation)],
            'alignment_raw_sha256':digest(alignment),'alignment_canonical_sha256':report._canonical(alignment),
            'correspondence_canonical_sha256':report._canonical(relation)},'alignment':alignment,'correspondence':relation,
        'qualification':qualification,'assumptions':'Explicit authored metadata, not recovered source history',
        'contact':'unknown','physical_validation':'unverified'}
    extension={'binding':binding,'binding_canonical_sha256':report._canonical(binding)}
    fit['multiview']={'schema_version':1,'profile':multi.EXTENSION,'views':metadata,'additional_view':extension,
        'secondary_source_slots':slots,'point_order':[{'view_id':view['name'],'joint':joint} for view in views for joint in report._CAMERA_POINTS],
        'bounds':{'additional_raw_bytes':16<<20,'preparation_seconds':5.,'within_original_deadline':True,'supplied_views':2,'source_slots_per_view':1800,'retained_references':900},
        'image_units':'Per-view authored loss scales; raw-pixel nonregression'}
    saved.update(profile=report.MULTIVIEW_METHOD,candidate_kind=report.MULTIVIEW_OVERLAY)
    saved['contract']['method']=report.MULTIVIEW_METHOD;saved['contract']['numerical_tolerance']=1e-8
    saved['objective']['profile']='camera_multiview_surface_interval_v1';saved['image_summary']['confident_source_endpoint_pairs']=sum(known)
    for i,side in enumerate(('before','after')):saved['image_summary']['mean_error_'+side+'_px']=sum(means[j][i]*known[j] for j in range(2))/sum(known)
    saved['gates']={name: name!='floor_feasibility' for name in sorted(multi.GATES)};saved['failures']=['floor_feasibility']
    saved['multiview_images']={'schema_version':1,'profile':multi.AUDIT,'binding_canonical_sha256':extension['binding_canonical_sha256'],
        'views':[meta|{'point_range':[index*12,(index+1)*12],'reference_count':3,'known_targets':known[index],'missing_targets':36-known[index],
            'invalid_before_known':0,'invalid_after_known':0,'mean_error_before_px':means[index][0],'mean_error_after_px':means[index][1],
            'maximum_camera_depth_change_supplied_units':0.,'gates':{name:True for name in multi.VIEW_GATES}}
            for index,meta in enumerate(metadata)],'accepted':True,'scope':'Authored saved image checks, not physical accuracy'}
    overlay=d['angle-proposal.json'];overlay.update(profile=report.MULTIVIEW_OVERLAY,additional_view=extension)
    overlay['whole_body_reference_refinement']['method']=report.MULTIVIEW_METHOD
    selected={'human.json':'inputs/human.json','evidence.json':'inputs/evidence-original.json','camera.json':'inputs/source-camera.json',
              'ground.json':'inputs/ground.json','inputs/two-view-root/camera.json':'inputs/secondary-camera.json',
              'inputs/two-view-root/evidence.json':'inputs/secondary-evidence.json'}
    d['inputs/body-manifest.json']={'schema_version':1,'files':{name:digest(d[alias]) for name,alias in selected.items()}}
    proof=d['inputs/camera-inputs-proof.json'];proof.update(input_manifest_sha256=digest(d['inputs/body-manifest.json']),
        camera_coupling=fit['coupling'],selected_members={name:{'sha256':digest(d[alias]),'bytes':len(encode(d[alias])),'copied_member':alias}
                                                       for name,alias in selected.items()})
    plan=d['plan.json'];plan['proposal_objective']='camera_multiview_surface_interval_v1'
    plan['secondary_view']=_descriptor(config,encode(config),{name:encode(value) for name,value in d.items()})
    plan['source']['copied_files']={name:digest(value) for name,value in d.items() if name.startswith('inputs/')}
    plan['input_files']={name:digest(value) for name,value in d.items() if name.startswith('inputs/')}
    repin(d)
    return d


def repin(documents):
    saved=documents['report.json'];extension=documents['angle-proposal.json']['additional_view']
    extension['binding_canonical_sha256']=report._canonical(extension['binding'])
    saved['multiview_images']['binding_canonical_sha256']=extension['binding_canonical_sha256']
    saved['root_audit_sha256']=report._canonical({k:v for k,v in saved['camera_fit'].items()
        if k not in {'projection_calls','maximum_projection_calls','grid_camera_projection_calls'}})
    saved['candidate_sha256']=report._canonical(documents['angle-proposal.json'])


def model(documents):
    return report._model(documents,{name:sha256(encode(value)).hexdigest() for name,value in documents.items()},lambda:None)


def test_two_named_views_retain_each_error_gate_clock_and_complete_slots():
    d=multiview_documents();value=model(d)
    assert value['method']==report.MULTIVIEW_METHOD and value['candidate_kind']==report.MULTIVIEW_OVERLAY
    assert [view['known_targets'] for view in value['camera_multiview']['views']]==[36,36]
    assert [view['authored_view_weight'] for view in value['camera_multiview']['views']]==[1.,3.]
    assert value['camera_multiview']['source_slot_counts']==[6,5]
    assert value['camera_multiview']['body_clock_dt_s']==1/30 != value['camera_fit']['coupling']['clock']['control_dt_s']
    assert 'multiview' not in value['camera_fit'] and len(value['source'][0]['points'])==24
    assert value['status']=='rejected_construction_proposal' and value['camera_multiview']['image_checks_passed']
    assert value['gates']['floor_feasibility'] is False and value['physical_validation']=='unverified'


@pytest.mark.parametrize('fault',['view_identity','duplicate_joint','secondary_raw_pin','clock','missing_slot',
    'gate','known_count','source_partition','baseline','boolean_weight','point_weight'])
def test_multiview_saved_identity_scope_and_per_view_contradictions_reject(fault):
    d=multiview_documents();saved=d['report.json'];multi=saved['camera_fit']['multiview'];binding=d['angle-proposal.json']['additional_view']['binding']
    if fault=='view_identity':saved['references'][0]['points'][12]['view_id']='primary'
    elif fault=='duplicate_joint':saved['references'][0]['points'][13]['joint']=saved['references'][0]['points'][12]['joint']
    elif fault=='secondary_raw_pin':binding['secondary_input_bindings']['evidence_raw_sha256']='0'*64
    elif fault=='clock':multi['secondary_source_slots'][2]['source_time_s']+=.001
    elif fault=='missing_slot':multi['secondary_source_slots'].pop()
    elif fault=='gate':saved['multiview_images']['views'][1]['gates']['camera_projection_domain']=False
    elif fault=='known_count':saved['multiview_images']['views'][1]['known_targets']-=1
    elif fault=='source_partition':multi['secondary_source_slots'][2]['source_partition']='validation'
    elif fault=='baseline':binding['qualification']['minimum_camera_baseline_supplied_units']=.5
    elif fault=='boolean_weight':saved['references'][0]['points'][0]['authored_view_weight']=True
    elif fault=='point_weight':saved['references'][0]['points'][12]['objective_weight']=.8
    binding['qualification']['secondary_source_slots_canonical_sha256']=report._canonical(multi['secondary_source_slots'])
    repin(d)
    with pytest.raises(ValueError):model(d)


def test_per_view_failure_is_visible_even_with_better_pooled_mean():
    d=multiview_documents();saved=d['report.json'];point=saved['references'][1]['points'][12]
    point.update(predicted_after_px=[point['target_px'][0]+6.,point['target_px'][1]],error_after_px=6.)
    secondary=saved['multiview_images']['views'][1];secondary['mean_error_after_px']+=(6.-4.)/36
    secondary['gates']['source_endpoint_image_nonregression']=False;saved['multiview_images']['accepted']=False
    saved['image_summary']['mean_error_after_px']+=(6.-4.)/72
    saved['gates']['source_endpoint_image_nonregression']=False;saved['failures'].append('source_endpoint_image_nonregression');repin(d)
    value=model(d)
    assert value['image_summary']['mean_error_after_px']<value['image_summary']['mean_error_before_px']
    assert value['camera_multiview']['image_checks_passed'] is False
    assert value['camera_multiview']['views'][0]['gates']['source_endpoint_image_nonregression'] is True
    secondary['gates']['source_endpoint_image_nonregression']=True;saved['multiview_images']['accepted']=True
    saved['gates']['source_endpoint_image_nonregression']=True;saved['failures'].remove('source_endpoint_image_nonregression')
    with pytest.raises(ValueError,match='image gate'):model(d)


def test_excluded_secondary_source_remains_unknown_without_erasing_primary_terms():
    value=model(multiview_documents(excluded_secondary=2))
    assert [view['known_targets'] for view in value['camera_multiview']['views']]==[36,24]
    slot=value['camera_multiview']['secondary_source_slots'][2]
    assert not slot['landmarks_used'] and slot['source_partition']=='validation'
    assert slot['image_source_admission']['reason']=='not_construction_source_evidence'
    points=value['source'][2]['points']
    assert all(p['weight']>0 for p in points[:12])
    assert all(p['weight']==0 and p['error_after_px'] is None and 'target_px' not in p for p in points[12:])


def test_invalid_final_known_target_retains_its_denominator_and_unknown_mean():
    d=multiview_documents();saved=d['report.json'];point=saved['references'][1]['points'][12]
    point.update(projection_after='unavailable_camera_domain',predicted_after_px=None,error_after_px=None)
    secondary=saved['multiview_images']['views'][1];secondary.update(invalid_after_known=1,mean_error_after_px=None)
    for name in ('source_endpoint_image_nonregression','camera_projection_domain'):
        secondary['gates'][name]=False;saved['gates'][name]=False;saved['failures'].append(name)
    saved['multiview_images']['accepted']=False;saved['image_summary'].update(invalid_final_known_endpoints=1,mean_error_after_px=None);repin(d)
    value=model(d);view=value['camera_multiview']['views'][1]
    assert view['known_targets']==36 and view['invalid_after_known']==1 and view['mean_error_after_px'] is None


def test_unsampled_secondary_pose_stays_unavailable_without_invented_source_row():
    d=multiview_documents(missing_secondary=2);before=encode(d['inputs/secondary-evidence.json'])
    value=model(d);slot=value['camera_multiview']['secondary_source_slots'][2]
    assert slot['status']=='unsampled' and slot['evidence_id'] is None and slot['source_time_s'] is None
    assert slot['image_source_admission']['reason']=='secondary_pose_not_sampled'
    assert value['camera_multiview']['views'][1]['known_targets']==24
    assert encode(d['inputs/secondary-evidence.json'])==before


@pytest.mark.parametrize('depth',[-1.,100.001])
def test_valid_saved_projection_cannot_hide_absolute_depth_outside_domain(depth):
    d=multiview_documents();point=d['report.json']['references'][1]['points'][12]
    point.update(depth_before_supplied_camera_units=depth,depth_after_supplied_camera_units=depth)
    with pytest.raises(ValueError,match='depth domain'):model(d)


@pytest.mark.parametrize('radial',[-.001,1.001])
def test_secondary_camera_rejects_coherently_pinned_unsupported_radial_domain(radial):
    with pytest.raises(ValueError,match='radial domain'):model(multiview_documents(secondary_radial=radial))


def test_unsampled_secondary_camera_clock_cannot_move_off_original_source_grid():
    with pytest.raises(ValueError,match='camera frame/order'):
        model(multiview_documents(secondary_clock_shift=.001))


def test_zero_known_secondary_targets_cannot_claim_per_view_success():
    d=multiview_documents();saved=d['report.json'];view=saved['multiview_images']['views'][1]
    for row in saved['references']:
        for point in row['points'][12:]:
            point.update(weight=0.,confidence_weight=0.,objective_weight=0.,source_confidence=None,
                error_before_px=None,error_after_px=None,status='unqualified_construction_image_source')
    view.update(known_targets=0,missing_targets=36,mean_error_before_px=None,mean_error_after_px=None)
    saved['image_summary'].update(confident_source_endpoint_pairs=36,mean_error_after_px=13/3)
    with pytest.raises(ValueError,match='declared view'):model(d)


def test_public_saved_report_copies_exact_secondary_inputs_and_current_helper_pins(tmp_path):
    d=multiview_documents();source=tmp_path/'source';pin=save_study(source,d);out=tmp_path/'inspection'
    result=report.prepare_whole_body_reference_report(source,out,expected_manifest_sha256=pin)
    assert result['status']=='report_generated'
    saved=json.loads((out/'report.json').read_bytes())
    assert saved['camera_multiview']['source_slot_counts']==[6,5]
    assert saved['selected_input_byte_bound']==24<<20 and saved['source_read_byte_bound']==48<<20
    assert set(saved['multiview_inspection_helpers'])=={'camera_multiview_report.py','camera_multiview_inputs.py'}
    for name in ('inputs/secondary-camera.json','inputs/secondary-evidence.json','inputs/camera-secondary-config.json'):
        copied=out/'artifacts'/(Path(name).name+'.txt')
        assert copied.read_bytes()==encode(d[name]) and saved['selected_members'][name]['sha256']==sha256(copied.read_bytes()).hexdigest()
    for item in saved['multiview_inspection_helpers'].values():assert sha256((out/item['path']).read_bytes()).hexdigest()==item['sha256']
    assert len((out/'index.html').read_bytes())<=report.MAX_HTML


def test_public_preflight_rejects_false_secondary_source_byte_count(tmp_path,monkeypatch):
    d=multiview_documents();proof=d['inputs/camera-inputs-proof.json'];proof['selected_members']['inputs/two-view-root/evidence.json']['bytes']+=1
    d['plan.json']['source']['copied_files']['inputs/camera-inputs-proof.json']=sha256(encode(proof)).hexdigest()
    source=tmp_path/'source';pin=save_study(source,d)
    monkeypatch.setattr(report,'_model',lambda *a:pytest.fail('Model inspection before byte preflight'))
    with pytest.raises(ValueError,match='Secondary selected-source byte count'):
        report.prepare_whole_body_reference_report(source,tmp_path/'rejected',expected_manifest_sha256=pin)


def test_multiview_automatic_input_envelope_accounts_all_alias_bytes(tmp_path,monkeypatch):
    d=multiview_documents();raw={name:encode(value) for name,value in d.items()}
    evidence=raw['inputs/evidence-derived.json'];evidence+=b' '*((7<<20)-len(evidence))
    for name in ('inputs/evidence-original.json','inputs/evidence-derived.json'):raw[name]=evidence
    original=d['inputs/body-manifest.json'];original['files']['evidence.json']=sha256(evidence).hexdigest()
    raw['inputs/body-manifest.json']=encode(original)
    proof=d['inputs/camera-inputs-proof.json'];proof.update(input_manifest_sha256=sha256(raw['inputs/body-manifest.json']).hexdigest(),scope='x'*(3<<20))
    proof['selected_members']['evidence.json'].update(sha256=sha256(evidence).hexdigest(),bytes=len(evidence))
    raw['inputs/camera-inputs-proof.json']=encode(proof)
    d['plan.json']['source']['copied_files']={name:sha256(value).hexdigest() for name,value in raw.items() if name.startswith('inputs/')}
    d['plan.json']['input_files']=d['plan.json']['source']['copied_files'];raw['plan.json']=encode(d['plan.json'])
    source=tmp_path/'source';source.mkdir()
    for name,value in raw.items():
        path=source/name;path.parent.mkdir(exist_ok=True);path.write_bytes(value)
    seal=encode({'schema_version':1,'profile':'whole-body-reference-candidate-study-v1','status':'completed',
        'files':{name:sha256(value).hexdigest() for name,value in raw.items()}});(source/'manifest.json').write_bytes(seal)
    assert 16<<20<sum(map(len,raw.values()))<24<<20
    class ReachedModel(Exception):pass
    monkeypatch.setattr(report,'_model',lambda *a:(_ for _ in ()).throw(ReachedModel()))
    # Only budget/custody preflight is exercised; the padded report is not admitted by a mocked model.
    with pytest.raises(ReachedModel):report.prepare_whole_body_reference_report(source,tmp_path/'automatic')
    with pytest.raises(ValueError,match='byte bound|bounded regular file'):
        report.prepare_whole_body_reference_report(source,tmp_path/'explicit',max_input_bytes=16<<20)
    assert not (tmp_path/'automatic').exists() and not (tmp_path/'explicit').exists()


def test_explicit_multiview_envelope_does_not_expand_legacy_or_tighter_caller_limits(tmp_path,monkeypatch):
    source=tmp_path/'legacy';save_study(source,camera_documents())
    reader=report.read_package_file;reads=[]
    def read(*a,**k):reads.append(a[1]);return reader(*a,**k)
    monkeypatch.setattr(report,'read_package_file',read)
    with pytest.raises(ValueError,match='legacy'):
        report.prepare_whole_body_reference_report(source,tmp_path/'legacy-out',max_input_bytes=17<<20)
    assert reads==['manifest.json','report.json']
    source2=tmp_path/'multiview';save_study(source2,multiview_documents());reads.clear()
    with pytest.raises(ValueError,match='byte bound|bounded regular file'):
        report.prepare_whole_body_reference_report(source2,tmp_path/'too-small',max_input_bytes=1024)
    assert not (tmp_path/'too-small').exists()


def test_multiview_report_member_cannot_consume_expanded_aggregate_envelope(tmp_path,monkeypatch):
    d=multiview_documents();source=tmp_path/'source';save_study(source,d)
    raw=encode(d['report.json']);raw+=b' '*((8<<20)+1-len(raw));(source/'report.json').write_bytes(raw)
    seal=json.loads((source/'manifest.json').read_bytes());seal['files']['report.json']=sha256(raw).hexdigest()
    (source/'manifest.json').write_bytes(encode(seal))
    reader=report.read_package_file;reads=[]
    def read(*a,**k):reads.append(a[1]);return reader(*a,**k)
    monkeypatch.setattr(report,'read_package_file',read)
    with pytest.raises(ValueError,match='8 MiB byte bound'):
        report.prepare_whole_body_reference_report(source,tmp_path/'rejected')
    assert reads==['manifest.json','report.json'] and not (tmp_path/'rejected').exists()


DOM_HARNESS=r"""
import fs from'node:fs';import vm from'node:vm';import assert from'node:assert/strict';import{performance}from'node:perf_hooks';
const html=fs.readFileSync(process.argv[2],'utf8'),scripts=[...html.matchAll(/<script(?:\s[^>]*)?>([\s\S]*?)<\/script>/g)];let created=0,writes=0;
assert.equal(scripts.length,2);const data=JSON.parse(scripts[0][1]);
class E{constructor(){this.children=[];this.listeners=new Map();this.dataset={};this.style={};this.value='';this.checked=false;this.own='';created++;}set textContent(v){writes++;this.own=String(v);this.children=[];}get textContent(){return this.own+this.children.map(v=>v.textContent).join('');}set innerHTML(v){assert.fail('unsafe HTML');}append(...v){this.children.push(...v);}replaceChildren(...v){writes++;this.own='';this.children=[...v];}setAttribute(k,v){this[k]=String(v);}addEventListener(k,f){this.listeners.set(k,f);}fire(k){this.listeners.get(k)?.();}}
const ids=new Map([...html.matchAll(/\bid="([^"]+)"/g)].map(v=>[v[1],new E()])),el=k=>{assert(ids.has(k),k);return ids.get(k)};
el('data').textContent=scripts[0][1];el('mode').value='source';el('affected').checked=true;
const start=performance.now();vm.runInNewContext(scripts[1][1],{document:{getElementById:el,createElement:()=>new E(),createElementNS:()=>new E()},fetch:()=>assert.fail('network'),setTimeout:()=>assert.fail('timer')},{timeout:1000});
const initial_ms=performance.now()-start,live=n=>1+n.children.reduce((s,c)=>s+live(c),0),count=live(el('points')),line=el('lines').children[1];
assert.equal(el('points').children.length,12);assert.equal(count,109);
const selected=data.source.findIndex(r=>r.frame===2);assert(selected>=0);el('sample').value=String(selected);el('sample').fire('change');el('camera-view').value=data.camera_multiview.views[1].view_id;el('camera-view').fire('change');
assert.equal(el('source-id').textContent,data.camera_multiview.secondary_source_slots[2].evidence_id);assert(el('identity').textContent.includes('decoder time '+Number(data.camera_multiview.secondary_source_slots[2].decoder_timestamp_s).toPrecision(6)));
assert(el('camera-reference-clock').textContent.includes('original primary body-reference ID '+data.source[selected].evidence_id));
const missing=data.source.findIndex(r=>!r.points);if(missing>=0){el('sample').value=String(missing);el('sample').fire('change');assert.equal(el('source-id').textContent,data.camera_multiview.secondary_source_slots[data.source[missing].frame].evidence_id);assert.match(el('points').textContent,/Unavailable/);}
el('sample').value=String(selected);el('sample').fire('change');
let slots=0;const navigation=performance.now();
for(let v=0;v<2;v++){el('camera-view').value=data.camera_multiview.views[v].view_id;el('camera-view').fire('change');const rows=v?data.camera_multiview.secondary_source_slots:data.camera_fit.source_slots;for(let i=0;i<rows.length;i++){el('camera-slot').value=String(i);el('camera-slot').fire('change');const shown=JSON.parse(el('camera-slot-detail').textContent);assert.equal(shown.source_frame,i);assert.equal(shown.view_id,data.camera_multiview.views[v].view_id);assert.equal(shown.evidence_id,rows[i].evidence_id);slots++;}}
const navigation_ms=performance.now()-navigation,stableNodes=created,stableWrites=writes;
for(let i=0;i<1000;i++){el('camera-slot').fire('change');el('camera-view').fire('change');el('sample').fire('change');}
assert.equal(created,stableNodes);assert.equal(writes,stableWrites);const duplicateWrites=writes-stableWrites;assert.equal(live(el('points')),count);assert.equal(el('lines').children[1],line);
const changes=performance.now();for(let i=0;i<200;i++){el('camera-view').value=data.camera_multiview.views[i%2].view_id;el('camera-view').fire('change');}
assert.equal(live(el('points')),count);assert.equal(el('lines').children[1],line);
el('mode').value='grid';el('mode').fire('change');assert.match(el('points').textContent,/Unavailable/);assert.match(el('camera-reference-clock').textContent,/No source-camera image target/);
process.stdout.write(JSON.stringify({initial_ms,source_slots_visited:slots,navigation_ms,view_changes:200,view_changes_ms:performance.now()-changes,duplicate_events:3000,duplicate_dom_writes:duplicateWrites,point_live_nodes:count,scope:'Scripted DOM only; no browser responsiveness or source qualification claim'})+'\n');
"""


@pytest.mark.parametrize('frames',[5,1800])
def test_selected_view_and_complete_slot_navigation_are_safe_bounded_and_deduplicated(tmp_path,record_property,frames):
    start=time.monotonic();value=model(multiview_documents())
    if frames==1800:
        # Renderer stress only: expand unknown slot projections, never claim producer/source admission.
        for slots in (value['camera_fit']['source_slots'],value['camera_multiview']['secondary_source_slots']):
            slots[:]=slots[:5]
            for frame in range(5,frames):
                slots.append({'source_frame':frame,'source_status':'decoded','camera_time_s':frame/30,
                    'reference_index':None,'evidence_id':None,'source_time_s':None,'decoder_timestamp_s':None,
                    'status':'unsampled','camera_status':'supplied','landmarks_used':False,'image_source_admission':None})
        value['camera_fit']['complete_source_frame_slots']=frames;value['camera_multiview']['source_slot_counts']=[frames,frames]
    value['camera_multiview']['additional_view']['binding']['assumptions']='Untrusted </script><img src=x onerror="bad()"> text'
    value.update(external_manifest_pin_supplied=False,source_manifest_sha256='f'*64,created_at='2026-10-31 22:10:00 PDT')
    page=report._render(value,':root{}',{'angle-proposal.json':'angle-proposal.json.txt','report.json':'proposal-report.json.txt'})
    assert len(page)<=report.MAX_HTML and b'</script><img' not in page
    path=tmp_path/'index.html';path.write_bytes(page)
    run=subprocess.run(['node','--input-type=module','-',str(path)],input=DOM_HARNESS,text=True,capture_output=True,timeout=5)
    assert run.returncode==0,run.stderr
    measurement=json.loads(run.stdout);assert measurement['source_slots_visited']==(11 if frames==5 else 3600)
    measurement.update(html_bytes=len(page),total_seconds=time.monotonic()-start,renderer_stress_only=frames==1800)
    record_property('multiview_scripted_navigation',json.dumps(measurement,sort_keys=True));print(json.dumps(measurement))


def packed_payload(raw):
    return json.loads(raw)


def test_local_payload_preserves_every_point_field_and_absent_null_values_without_mutating_model():
    from weft.camera_multiview_report import build_multiview_display_payload
    value=model(multiview_documents(excluded_secondary=2));point=value['source'][0]['points'][0]
    point['inspection_note']='</script><img onerror="bad()">';point['explicit_null']=None
    before=encode(value);raw=build_multiview_display_payload(value,lambda:None);packed=packed_payload(raw)
    assert encode(value)==before and b'</script' not in raw and b'<img' not in raw
    for original,row in zip(value['source'],packed['model']['source'],strict=True):
        if original['points'] is None:assert row['points'] is None;continue
        decoded=[]
        for item in row['points']:
            fields=packed['point_layouts'][item[0]];result={}
            for column,entry in zip(fields,item[1:],strict=True):
                dictionary=packed['point_strings'].get(str(column))
                result[packed['point_columns'][column]]=dictionary[entry] if dictionary and entry is not None else entry
            decoded.append(result)
        assert decoded==original['points'] and [list(p) for p in decoded]==[list(p) for p in original['points']]
    assert packed['model']['camera_fit']['source_slots']==value['camera_fit']['source_slots']
    assert packed['model']['camera_multiview']['secondary_source_slots']==value['camera_multiview']['secondary_source_slots']
    assert packed['model']['surface_intervals']['source_slots']==value['surface_intervals']['source_slots']


def test_overflow_only_package_keeps_plain_full_model_and_inline_payload_without_fetch(tmp_path,monkeypatch,record_property):
    from weft import camera_multiview_report as multi
    d=multiview_documents();value=model(d);rows=[r for r in value['source'] if r['points']]
    value['source']=[deepcopy(row) for _ in range(60) for row in rows]
    source=tmp_path/'source';pin=save_study(source,d);out=tmp_path/'inspection';calls=[]
    monkeypatch.setattr(report,'_model',lambda *a:deepcopy(value))
    original=multi.build_multiview_display_payload
    def build(*a):calls.append(1);return original(*a)
    monkeypatch.setattr(multi,'build_multiview_display_payload',build)
    result=report.prepare_whole_body_reference_report(source,out,expected_manifest_sha256=pin)
    saved=json.loads((out/'report.json').read_bytes());page=(out/'index.html').read_bytes()
    raw=re.search(rb'<script id="data" type="application/json">(.*?)</script>',page,re.S).group(1)
    assert calls==[1] and result['bytes']<=16<<20 and len(page)<=2<<20
    assert saved['source']==value['source'] and saved['display_payload']['sha256']==sha256(raw).hexdigest()
    assert b'fetch(' not in raw and b'fetch(' not in page and b'<script src=' not in page
    assert 'report-data.js' not in json.loads((out/'manifest.json').read_bytes())['files']
    harness=DOM_HARNESS.replace("const data=JSON.parse(scripts[0][1]);", "const data=JSON.parse(scripts[0][1]).model;")
    harness=harness.replace('const start=performance.now();','const decodeEvents=[];const start=performance.now();')
    harness=harness.replace("scripts[1][1],{document:","scripts[1][1].replace('function unpackPoint(record){','function unpackPoint(record){decodeEvents.push(1);'),{decodeEvents,document:")
    harness=harness.replace('const initial_ms=', 'assert.equal(decodeEvents.length,12);const initial_ms=')
    harness=harness.replace('stableWrites=writes;', 'stableWrites=writes,stableDecodes=decodeEvents.length;')
    harness=harness.replace('assert.equal(created,stableNodes);', 'assert.equal(decodeEvents.length,stableDecodes);assert.equal(created,stableNodes);')
    run=subprocess.run(['node','--input-type=module','-',str(out/'index.html')],input=harness,text=True,capture_output=True,timeout=5)
    assert run.returncode==0,run.stderr
    measurement=json.loads(run.stdout);measurement.update(html_bytes=len(page),payload_bytes=len(raw),total_output_bytes=result['bytes'],render_only_model_stub=True)
    record_property('packed_navigation',json.dumps(measurement,sort_keys=True))


@pytest.mark.parametrize('fault',['missing','layout','string_index','dictionary','invalid_json'])
def test_inline_payload_missing_or_malformed_fails_visibly(tmp_path,fault):
    from weft.camera_multiview_report import build_multiview_display_payload
    value=model(multiview_documents());raw=build_multiview_display_payload(value,lambda:None)
    value.update(external_manifest_pin_supplied=False,source_manifest_sha256='f'*64,created_at='2026-10-31 22:10:00 PDT')
    page=report._render(value,':root{}',{'report.json':'proposal-report.json.txt'},packed_payload=raw)
    (tmp_path/'index.html').write_bytes(page)
    prefix=DOM_HARNESS.split('const start=performance.now();')[0]
    prefix=prefix.replace("assert.equal(scripts.length,2);const data=JSON.parse(scripts[0][1]);", "assert.equal(scripts.length,2);")
    harness=prefix+r"""
const context={document:{getElementById:el,createElement:()=>new E(),createElementNS:()=>new E()},fetch:()=>assert.fail('network'),setTimeout:()=>assert.fail('timer')};
const fault=process.argv[3];
if(fault==='missing')el('data').textContent='null';
else if(fault==='invalid_json')el('data').textContent='{';
else{const p=JSON.parse(el('data').textContent);
 if(fault==='layout')p.model.source[0].points[0][0]=1000;
 if(fault==='string_index'){const column=p.point_columns.indexOf('joint'),record=p.model.source[0].points[0],layout=p.point_layouts[record[0]];record[layout.indexOf(column)+1]=1000;}
 if(fault==='dictionary')delete p.point_strings[p.point_columns.indexOf('joint')];el('data').textContent=JSON.stringify(p);}
assert.throws(()=>vm.runInNewContext(scripts[1][1],context,{timeout:1000}),/Invalid local multiview display payload/);
assert.equal(el('payload-error').hidden,false);assert.match(el('payload-error').textContent,/missing or malformed/);assert.equal(el('camera-view').disabled,true);assert.equal(el('points').children.length,0);
"""
    run=subprocess.run(['node','--input-type=module','-',str(tmp_path/'index.html'),fault],input=harness,text=True,capture_output=True,timeout=5)
    assert run.returncode==0,run.stderr
