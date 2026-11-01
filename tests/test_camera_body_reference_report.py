"""Camera overlay inspection uses saved analytical data, never geometry or a fitter."""

import builtins
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
import socket
import subprocess

import pytest

from weft import whole_body_reference_report as report
from test_whole_body_reference_report import encode, generated, save_study
from test_whole_body_reference_interval_report import interval_documents


NAMES = ('torso_twist', 'torso_lateral', 'torso_flex') + tuple(name for side in ('left', 'right') for name in (
    f'{side}_shoulder_twist', f'{side}_shoulder_lateral', f'{side}_shoulder', f'{side}_elbow',
    f'{side}_hip_twist', f'{side}_hip_lateral', f'{side}_hip', f'{side}_knee'))


@pytest.fixture(autouse=True)
def metadata_only(monkeypatch):
    importing = builtins.__import__
    def guard(name, *args, **kwargs):
        assert name.split('.')[0] not in {'numpy', 'mujoco', 'cv2', 'mediapipe', 'pymunk', 'torch'}, name
        assert name not in {'weft.camera_body_fit', 'weft.camera_pose', 'weft.camera_reference',
                            'weft.camera_body_reference_inputs', 'weft.camera_shared_contract',
                            'weft.whole_body_reference_refinement', 'weft.source_surface_envelopes',
                            'weft.source_surface_interval_construction'}, name
        return importing(name, *args, **kwargs)
    def denied(*args, **kwargs):
        pytest.fail('Camera report attempted network work')
    monkeypatch.setattr(builtins, '__import__', guard)
    monkeypatch.setattr(socket.socket, 'connect', denied)
    monkeypatch.setattr(socket, 'create_connection', denied)


def digest(value):
    return sha256(encode(value)).hexdigest()


def repin_overlay(documents):
    documents['report.json']['candidate_sha256'] = report._canonical(documents['angle-proposal.json'])


def repin_camera_audit(documents):
    fit = documents['report.json']['camera_fit']
    documents['report.json']['root_audit_sha256'] = report._canonical({k: v for k, v in fit.items() if k not in {
        'projection_calls', 'maximum_projection_calls', 'grid_camera_projection_calls'}})


def camera_documents(*, excluded_frame=None, unknown_heights=False):
    d = interval_documents(); saved = d['report.json']; surface = saved['surface_intervals']
    body, evidence = d['inputs/human.json'], d['inputs/evidence-derived.json']; source = body['source_sha256']
    evidence.update(width=100, height=80, frame_count=5, decoded_frames=6)
    frames = []
    for i in range(6):
        frames.append({'id': f'{source}:frame:{i}', 'frame': i, 'time_s': i/2, 'decoder_timestamp_s': None if i == 0 else i/2+.014,
            'status': 'tracked_partial' if i in (0, 2, 4) else 'low_confidence', 'pose_count': 1,
            'landmarks': [{'x': .6, 'y': .5, 'visibility': .9, 'presence': .8} for _ in range(33)]})
    evidence['frames'] = frames
    body['reference'] = [{'time_s': f/2, 'joints': {name: 0. for name in NAMES}} for f in (0, 2, 4)]
    body['reference_evidence'] = [{'time_s': f/2, 'evidence_id': f'{source}:frame:{f}'} for f in (0, 2, 4)]
    body['root_reference'] = [{'time_s': f/2, 'translation': [0., 0., 0.]} for f in (0, 2, 4)]
    body['dt'] = 1/30
    camera = {'schema_version': 1, 'source_sha256': source, 'width': 100, 'height': 80, 'frame_count': 5,
        'fps': 2., 'radial_k1': .1, 'quality': 'authored', 'units': 'm', 'world_axes': 'right_handed_z_up',
        'K': [[100., 0., 50.], [0., 100., 40.], [0., 0., 1.]],
        'frames': [{'frame': f, 'time_s': f/2, 'R': [[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]], 't': [0., 0., 2.]} for f in range(5)]}
    ground = {'profile': 'analytical-ground-prior', 'height_m': 0., 'physical_validation': 'unverified'}
    body['camera_reference'] = {'method': 'two-view-root-body-v1', 'ground': ground,
        'calibration_sha256': report._canonical(camera), 'pose_evidence_sha256': report._canonical(evidence),
        'origin_world_m': [0., 0., 0.], 'root_refinement': {'fps': 2., 'source_frames': 5,
            'roots': [{'frame': f, 'time_s': f/2} for f in (0, 2, 4)]}}
    body_hash, evidence_hash = digest(body), digest(evidence)
    observations = d['inputs/surface-observation-00.json']
    observations['source'].update(decoded_frames=6, metadata_frames=5)
    observations['observations'] = [{'id': f'person:{f}', 'object_id': 'person', 'frame': f, 'time_s': f/2,
        'partition': 'validation' if f == excluded_frame else 'construction', 'visibility': 'detected', 'evidence_ids': [f'{source}:frame:{f}']} for f in range(6)]
    observations['evidence'] = [{'id': f'{source}:frame:{f}', 'object_id': 'person', 'frame': f, 'time_s': f/2,
                                 'kind': 'person_hypothesis', 'status': 'inferred', 'payload': {}} for f in range(6)]
    entity = {'document_sha256': digest(observations), 'object_id': 'person'}
    annotations = d['inputs/surface-annotations.json']; annotations['channels'][0]['entity_ref'] = entity
    annotations['annotations'] = [{'id': f'annotation:{f}', 'channel_id': 'ankle', 'frame': f, 'time_s': f/2,
        'observation_id': f'person:{f}', 'status': 'annotated', 'pixel': [60., 45.], 'uncertainty_px': 2.} for f in range(6)]
    d['inputs/surface-camera.json'] = camera
    d['inputs/surface-intervals.json']['calibration_sha256'] = digest(camera)
    if unknown_heights:
        d['inputs/surface-intervals.json']['surfaces'][0].update(height_interval_m=None, quality='unknown')
    envelope_bindings = {'observation_document_sha256s': [digest(observations)],
        'camera_sha256': digest(camera), 'annotations_sha256': digest(annotations),
        'surface_intervals_sha256': digest(d['inputs/surface-intervals.json'])}
    body_binding = d['inputs/surface-body-binding.json']
    body_binding.update(body_sha256=body_hash, envelope_bindings=envelope_bindings, human_entity_ref=entity)
    config = d['inputs/surface-config.json']
    config['observations'][0]['sha256'] = digest(observations)
    for key, name in (('camera', 'camera'), ('surface_intervals', 'intervals'), ('annotations', 'annotations'), ('body_binding', 'body-binding')):
        config[key]['sha256'] = digest(d['inputs/surface-'+name+'.json'])
    saved.update(profile=report.CAMERA_METHOD, candidate_kind=report.CAMERA_OVERLAY,
                 geometry_admission='unsupported_angle_overlay_not_a_human_spec',
                 input_human_sha256=body_hash, input_evidence_sha256=evidence_hash)
    saved['contract']['method'] = report.CAMERA_METHOD
    saved['gates'].update(camera_projection_domain=True, camera_endpoint_depth_change=True)
    saved['objective']['profile'] = 'camera_surface_interval_v1'
    saved['retained_pose_slots'] = [{k: f.get(k) for k in ('id', 'frame', 'time_s', 'decoder_timestamp_s', 'status')} for f in frames]
    saved['source_counts'].update(retained_pose_slots=6, body_reference_knots=3, declared_decoded_frames=6, declared_frame_count=5)
    saved['references'] = []
    for index, f in enumerate((0, 2, 4)):
        points = []
        for name in report._CAMERA_POINTS:
            error = 5. if index == 0 else 4.
            point = {'joint': name, 'status': 'available_image_estimate', 'weight': .8, 'source_confidence': .8,
                'landmark_source': 'landmarks', 'target_px': [60., 40.], 'target_authored_loss_units': [.12, .08],
                'error_before_px': 5., 'error_after_px': error, 'predicted_before_px': [65., 40.], 'predicted_after_px': [60.+error, 40.],
                'depth_before_supplied_camera_units': 2., 'depth_after_supplied_camera_units': 2.,
                'projection_before': 'valid', 'projection_after': 'valid'}
            if f == excluded_frame:
                point.update(status='unqualified_construction_image_source', weight=0., source_confidence=None,
                             landmark_source='not_read_unqualified_source', error_before_px=None,error_after_px=None)
                point.pop('target_px'); point.pop('target_authored_loss_units')
            points.append(point)
        saved['references'].append({'evidence_id': f'{source}:frame:{f}', 'frame': f, 'time_s': f/2,
            'status': 'frozen_first_reference' if index == 0 else 'changed_inferred_proposal', 'points': points})
    checks = []
    for kind, stamps in (('source_pose', (0., 1., 2.)), ('control_grid', (0., .5, 1., 1.5, 2.))):
        for index, t in enumerate(stamps):
            checks.append({'kind': kind, 'time_s': t, 'source_reference_index': index if kind == 'source_pose' else None,
                'before_floor_support_gaps': [-.04, .1, .2], 'after_floor_support_gaps': [-.04, .1, .2],
                'before_max_penetration_model_units': .04, 'after_max_penetration_model_units': .04,
                'maximum_depth_change_model_units': 0., 'no_worse_floor': True})
    saved['geometry_checks'] = checks
    saved['floor_summary'].update(checks=8, source_pose_checks=3, control_grid_checks=5, before_failures=8, after_failures=8,
                                  before_maximum_penetration_model_units=.04, after_maximum_penetration_model_units=.04)
    saved['image_summary'].update(confident_source_endpoint_pairs=36, mean_error_before_px=5., mean_error_after_px=13/3,
                                  invalid_final_known_endpoints=0)
    image_pairs = [p for ref in saved['references'] for p in ref['points'] if p['weight']]
    saved['image_summary'].update(confident_source_endpoint_pairs=len(image_pairs),
                                  mean_error_after_px=sum(p['error_after_px'] for p in image_pairs)/len(image_pairs))
    slots = []
    for f in range(6):
        ref = {0: 0, 2: 1, 4: 2}.get(f)
        envelope_reason = ('excluded_evidence_partition' if f == excluded_frame else 'unknown_surface_height_interval' if unknown_heights
                           else 'camera_and_source_clocks_differ' if f == 5 else None)
        reason = envelope_reason or ('original_body_reference_unavailable' if ref is None else None)
        channel = {'channel_id': 'ankle', 'entity_ref': entity, 'body_point': 'right_ankle', 'source_observation_id': f'person:{f}',
            'source_time_s': f/2, 'source_partition': 'validation' if f == excluded_frame else 'construction',
            'envelope_status': 'unavailable' if envelope_reason else 'conditional_surface_envelope', 'envelope_reason': envelope_reason,
            'status': 'conditional_interval_ready' if reason is None else 'unavailable', 'reason': reason,
            'world_envelope_m': None if envelope_reason else {'minimum': [0., 0., .2], 'maximum': [1., 1., .6], 'unit': 'm', 'kind': 'outer_axis_aligned_enclosure'}}
        slots.append({'source_frame': f, 'source_status': 'decoded', 'body_reference': None if ref is None else {
            'reference_index': ref, 'evidence_id': f'{source}:frame:{f}', 'body_time_s': f/2}, 'channels': [channel]})
    old_results = deepcopy(surface['references'])
    surface['source_slots'], surface['references'] = slots, []
    for index, f in enumerate((0, 2, 4)):
        row = deepcopy(old_results[min(index, 1)])
        row.update(reference_index=index, source_frame=f, evidence_id=f'{source}:frame:{f}', time_s=f/2)
        for side in ('before', 'after'):
            row[side].update(source_frame=f, body_sha256=body_hash, body_reference=slots[f]['body_reference'])
            channel = row[side]['channels'][0]
            channel.update({k: v for k, v in slots[f]['channels'][0].items() if k not in {'world_envelope_m', 'status', 'reason'}})
            if slots[f]['channels'][0]['status'] == 'unavailable':
                channel.update(status='unavailable',reason=slots[f]['channels'][0]['reason'],signed_outside_m=None,world_position_m=None,
                               distance_to_outer_envelope_m=None,objective_m2=None,active_axes=None)
                row[side].update(objective_m2=None,counts={'retained_channels':1,'evaluated_channels':0,'unavailable_channels':1})
        surface['references'].append(row)
    inputs = surface['inputs']
    inputs.update(source_sha256=source, body_sha256=body_hash, body_model_sha256=report._canonical(body), binding=body_binding,
                  binding_sha256=digest(body_binding), envelope_bindings=envelope_bindings,
                  source_clocks={digest(observations): observations['time']},
                  original_start_statuses=[{k: v for k, v in slots[0]['channels'][0].items() if k != 'world_envelope_m'}],
                  counts={'retained_source_frames': 6, 'retained_channel_slots': 6, 'original_body_references': 3,'status_counts':{}})
    for slot in slots:
        c=slot['channels'][0]; key=c['reason'] or c['status']
        inputs['counts']['status_counts'][key]=inputs['counts']['status_counts'].get(key,0)+1
    surface['weight_units'] = 'authored_camera_loss_units_squared_per_supplied_world_metre_squared'
    available=sum(r['before']['objective_m2'] is not None for r in surface['references'])
    surface['summary'] = {'source_frame_slots': 6, 'channel_slots': 6, 'available_reference_terms': available,
                         'unavailable_reference_terms': 3-available, 'evaluated_channels_before': available, 'evaluated_channels_after': available}
    for side in ('before', 'after'):
        values = [r[side]['objective_m2'] for r in surface['references'] if r[side]['objective_m2'] is not None]
        distances=[r[side]['channels'][0]['distance_to_outer_envelope_m'] for r in surface['references'] if r[side]['objective_m2'] is not None]
        surface['summary'].update({f'sum_objective_{side}_m2': sum(values) if values else None,
            f'weighted_sum_{side}': 2.5*sum(values) if values else None, f'maximum_channel_outside_{side}_m': max(distances) if distances else None})
    admissions=[]
    for f,slot in enumerate(slots):
        reference=slot['body_reference']; reason='not_construction_source_evidence' if f==excluded_frame else 'original_body_reference_unavailable' if reference is None else None
        admissions.append({'schema_version':1,'profile':'source-surface-body-image-admission-v1','source_frame':f,
            'status':'construction_image_source_ready' if reason is None else 'unavailable','reason':reason,
            'human_entity_ref':entity,'body_reference':reference,'evidence_ref':None if reference is None else {
                'document_sha256':entity['document_sha256'],'evidence_id':reference['evidence_id']},
            'observation_ref':{'document_sha256':entity['document_sha256'],'observation_id':f'person:{f}'},
            'source_time_s':f/2,'source_partition':'validation' if f==excluded_frame else 'construction'})
    compact=lambda value:len(json.dumps(value,sort_keys=True,separators=(',',':')).encode())
    admission_bytes=sum(map(compact,admissions))
    ordinary_bytes=sum(compact({'source_frame':s['source_frame'],'body_reference':s['body_reference'],
        'channels':[{k:v for k,v in c.items() if k!='world_envelope_m'} for c in s['channels']]}) for s in slots)
    inputs['source_image_admission']={'profile':'source-surface-body-image-admission-v1','slots':6,'row_cache_bytes':admission_bytes,
        'ordinary_join_row_cache_bytes':ordinary_bytes,'combined_row_cache_bytes':admission_bytes+ordinary_bytes,
        'shared_envelope_cache_bytes':2048,'maximum_additional_cache_bytes':2<<20,'scope':'Encoded source custody; independent of supplied ranges'}
    clock_rows = [{'frame': f, 'time_s': f/2, 'evidence_id': f'{source}:frame:{f}'} for f in (0, 2, 4)]
    coupling = {'schema_version': 1, 'profile': 'camera-shared-coupling-v1', 'source_sha256': source,
        'body_sha256': report._canonical(body), 'camera_binding_sha256': report._canonical(body['camera_reference']),
        'ground_sha256': report._canonical(ground), 'root_reference_sha256': report._canonical(body['root_reference']),
        'source_clock_sha256': report._canonical({'schema_version': 1, 'profile': 'camera-shared-source-clock-v1',
            'source_sha256': source, 'source_fps': 2., 'total_source_frames': 5, 'control_dt_s': .5, 'rows': clock_rows}),
        'clock': {'control_dt_s': .5, 'source_fps': 2., 'source_time_offset_s': 0., 'first_source_frame': 0, 'last_source_frame': 4,
                  'total_source_frames': 5, 'retained_poses': 3, 'horizon_steps': 4},
        'calibration_status': 'estimated_prior', 'world_to_local': {'scale': 1., 'translation': [0., 0., 0.]}}
    pins = {'body_raw_sha256': body_hash, 'evidence_raw_sha256': evidence_hash, 'camera_raw_sha256': digest(camera),
        'body_canonical_sha256': report._canonical(body), 'camera_canonical_sha256': report._canonical(camera),
        'pose_evidence': {'canonical_sha256': report._canonical(evidence), 'scope': 'complete_supplied_pose_document'},
        'camera_binding_canonical_sha256': report._canonical(body['camera_reference']), 'coupling_canonical_sha256': report._canonical(coupling),
        'root_reference_canonical_sha256': report._canonical(body['root_reference']), 'ground_canonical_sha256': report._canonical(ground),
        'original_angular_reference_canonical_sha256': report._canonical(body['reference'])}
    fit = {'schema_version': 1, 'profile': 'fixed-camera-body-fit-inputs-v1', 'input_bindings': pins, 'coupling': coupling,
        'source_slots': [{'source_frame': f, 'camera_time_s': f/2 if f < 5 else None, 'reference_index': {0:0, 2:1, 4:2}.get(f),
            'evidence_id': frames[f]['id'], 'source_time_s': f/2, 'decoder_timestamp_s': frames[f]['decoder_timestamp_s'],
            'status': frames[f]['status'], 'camera_status': 'supplied' if f < 5 else 'unavailable_original_decoded_slot',
            'landmarks_used': f in (0, 2, 4) and f!=excluded_frame,
            'image_source_admission':admissions[f] if f in (0,2,4) else None} for f in range(6)],
        'image_loss_scale_per_pixel': .002, 'image_loss_scale_status': 'authored', 'complete_source_frame_slots': 6,
        'image_loss_units': 'authored_loss_units_not_metres', 'huber_delta_loss_units': .05, 'raw_radial_k1': .1,
        'depth_bounds_supplied_camera_units': [.2,100.], 'maximum_normalized_radius': 4., 'image_nonregression_tolerance_px': 1e-8,
        'projection_calls': 8, 'maximum_projection_calls': 120000, 'grid_camera_projection_calls': 0}
    fit['prepared_source_cache']={'objective_metadata_bytes':compact(inputs),'objective_join_rows_bytes':ordinary_bytes,
        'image_admission_rows_bytes':admission_bytes,'shared_envelope_cache_bytes':2048}
    fit['prepared_source_cache']['total_encoded_cache_bytes']=sum(fit['prepared_source_cache'].values())
    saved['camera_fit'] = fit
    saved['image_dimensions'] = {'width': 100, 'height': 80, 'provenance': {'method': 'supplied_camera_and_pose_metadata',
        'camera_raw_sha256': digest(camera), 'source_sha256': source, 'width': 100, 'height': 80,
        'decoder_verification': 'unavailable_no_decode_performed'}}
    old = d.pop('candidate-human.json'); marker = old['whole_body_reference_refinement']
    marker.update(method=report.CAMERA_METHOD, input_human_sha256=body_hash, input_evidence_sha256=evidence_hash)
    marker['surface_intervals'] = {'profile': report.SURFACE_PROFILE, 'input_bindings': {'objective_profile': report.SURFACE_OBJECTIVE,
        **{key: inputs[key] for key in ('source_sha256','body_sha256','binding_sha256','envelope_cache_sha256','envelope_bindings')}},
        'weight': 2.5, 'weight_status': 'authored', 'work_limits': surface['work_limits'], 'summary': surface['summary']}
    proposed = deepcopy(body['reference'])
    for row in proposed[1:]: row['joints']['right_knee'] = .1
    d['angle-proposal.json'] = {'schema_version':1, 'profile':report.CAMERA_OVERLAY, 'input_bindings': pins,
        'reference_candidate': proposed, 'whole_body_reference_refinement': marker,
        'installation':'unsupported_angle_overlay_not_a_human_spec', 'body_installed':False, 'raw_camera_changed':False,
        'root_or_floor_changed':False, 'promoted':False, 'contact':'unknown', 'physical_validation':'unverified'}
    d.update({'inputs/evidence-original.json': deepcopy(evidence), 'inputs/source-camera.json': deepcopy(camera), 'inputs/ground.json': ground})
    selected = {'human.json': 'inputs/human.json', 'evidence.json': 'inputs/evidence-original.json',
                'camera.json': 'inputs/source-camera.json', 'ground.json': 'inputs/ground.json'}
    d['inputs/body-manifest.json'] = {'schema_version':1, 'files':{name:digest(d[alias]) for name,alias in selected.items()}}
    d['inputs/camera-inputs-proof.json'] = {'schema_version':1,'profile':'camera-body-reference-inputs-v1','status':'metadata_verified',
        'input_manifest_sha256':digest(d['inputs/body-manifest.json']), 'selected_members': {
            name:{'sha256':digest(d[alias]), 'bytes':len(encode(d[alias])), 'copied_member':alias} for name,alias in selected.items()},
        'camera_coupling': coupling, 'dimension_derivation':'none; exact original evidence bytes',
        'scope':'Selected saved metadata, not authenticated decoder or physical validity'}
    plan = d['plan.json']; plan.update(proposal_objective='camera_surface_interval_v1',camera_image_scale=.002)
    plan['surface_constraints'].update(config_sha256=digest(config))
    plan['input_files'] = {name:digest(d[name]) for name in plan['input_files']}
    plan['source'] = {'profile':'camera-body-reference-inputs-v1', 'copied_files': {
        name:digest(d[name]) for name in {*report._CAMERA_EXTRAS,'inputs/human.json','inputs/evidence-derived.json'}}}
    repin_camera_audit(d); repin_overlay(d)
    return d


def test_camera_angle_overlay_report_keeps_raw_sources_units_and_missing_camera_tail(tmp_path):
    documents = camera_documents(); source, out, _, result = generated(tmp_path, documents)
    model = json.loads((out/'report.json').read_bytes()); page = (out/'index.html').read_text()
    assert result['proposal_status'] == 'rejected_construction_proposal' and result['promoted'] is False
    assert model['candidate_kind'] == report.CAMERA_OVERLAY and model['camera_fit']['complete_source_frame_slots'] == 6
    assert model['camera_fit']['source_slots'][5]['camera_time_s'] is None
    assert model['camera_fit']['source_slots'][5]['source_time_s'] == 2.5
    assert model['camera_fit']['source_slots'][5]['decoder_timestamp_s'] == 2.514
    assert model['image_summary']['confident_source_endpoint_pairs'] == 36
    assert model['image_summary']['mean_error_after_px'] == pytest.approx(13/3)
    assert len(model['surface_intervals']['source_slots']) == 6
    assert model['gates']['floor_feasibility'] is False and model['floor_limit'] == .025
    assert 'candidate-human.json' not in model['selected_members']
    assert (out/'artifacts/angle-proposal.json.txt').read_bytes() == (source/'angle-proposal.json').read_bytes()
    assert (out/'artifacts/evidence-original.json.txt').read_bytes() == (out/'artifacts/evidence-derived.json.txt').read_bytes()
    assert (out/'artifacts/source-camera.json.txt').read_bytes() == (out/'artifacts/surface-camera.json.txt').read_bytes()
    for text in ('not a replacement human model', 'not installed', 'raw pixels', 'not metres', 'Proposed angles',
                 'authored camera loss units² per m²', 'no camera pose is interpolated', 'not a fresh verification'):
        assert text in page
    assert 'Proposed body' not in page.split('<script', 1)[0]


def invalidate_known_projection(d):
    point = d['report.json']['references'][1]['points'][0]
    point.update(projection_after='unavailable_camera_domain', predicted_after_px=None, error_after_px=None,
                 depth_after_supplied_camera_units=-1.)
    saved = d['report.json']; saved['image_summary'].update(mean_error_after_px=None,invalid_final_known_endpoints=1)
    saved['gates'].update(camera_projection_domain=False,camera_endpoint_depth_change=False,source_endpoint_image_nonregression=False)
    saved['failures'] = [name for name,value in saved['gates'].items() if value is False]


def test_known_invalid_projection_and_partial_trial_remain_visible(tmp_path):
    d=camera_documents(); invalidate_known_projection(d)
    saved=d['report.json']; saved.update(status='partial_budget_exhausted',budget_reason='bounded projection work exhausted')
    saved['gates']['complete_bounded_solve']=False
    saved['failures']=[name for name,value in saved['gates'].items() if value is False]
    overlay=d['angle-proposal.json']; overlay['whole_body_reference_refinement']['status']=saved['status']
    surface=saved['surface_intervals']; surface['trial_journal']=[{'sequence':0,'kind':'backtrack','reference_index':1,'source_frame':2,
        'block':['right_knee'],'backtrack_index':0,'trial_correction_rad':[0.]*19,'combined_cost_before':.1,
        'combined_cost_after':None,'interval_before_m2':.08,'interval_after_m2':None,'gate_results':{},'accepted':False,
        'outcome':'evaluation_incomplete','camera_projection_valid':None,'maximum_camera_depth_change_supplied_units':None,
        'camera_depth_bound_satisfied':None,'maximum_source_image_excess_px':None}]
    surface['journal_bytes']=sum(len(json.dumps(row,separators=(',',':')).encode()) for row in surface['trial_journal'])
    repin_overlay(d)
    _,out,_,_=generated(tmp_path,d); model=json.loads((out/'report.json').read_bytes())
    point=model['source'][2]['points'][0]
    assert point['weight']==.8 and point['error_after_px'] is None and point['predicted_after_px'] is None
    assert model['image_summary']['mean_error_after_px'] is None
    assert model['image_summary']['confident_source_endpoint_pairs']==36
    assert model['surface_intervals']['journal_records']==1
    assert model['status']=='partial_budget_exhausted'


def test_unknown_surface_height_does_not_remove_qualified_image_targets(tmp_path):
    _, out, _, _ = generated(tmp_path, camera_documents(unknown_heights=True))
    model = json.loads((out/'report.json').read_bytes())
    assert model['image_summary']['confident_source_endpoint_pairs'] == 36
    assert model['image_summary']['mean_error_after_px'] == pytest.approx(13/3)
    assert model['surface_intervals']['summary']['sum_objective_after_m2'] is None
    assert model['surface_intervals']['summary']['unavailable_reference_terms'] == 3
    assert model['camera_fit']['source_slots'][2]['landmarks_used'] is True


def test_excluded_source_ref_keeps_geometry_but_never_observed_image_targets(tmp_path):
    _, out, _, _ = generated(tmp_path, camera_documents(excluded_frame=2))
    model = json.loads((out/'report.json').read_bytes())
    assert model['image_summary']['confident_source_endpoint_pairs'] == 24
    assert model['camera_fit']['source_slots'][2]['landmarks_used'] is False
    assert model['camera_fit']['source_slots'][2]['image_source_admission']['reason'] == 'not_construction_source_evidence'
    assert all(point['weight'] == 0 and 'target_px' not in point and point['predicted_before_px'] is not None
               for point in model['source'][2]['points'])
    assert model['source'][2]['target_label'] == 'Unavailable — not construction-qualified source image'


def test_excluded_source_cannot_be_relabelled_image_ready(tmp_path):
    d = camera_documents(excluded_frame=2)
    slot = d['report.json']['camera_fit']['source_slots'][2]
    slot['landmarks_used'] = True
    slot['image_source_admission'].update(status='construction_image_source_ready', reason=None)
    repin_camera_audit(d)
    source = tmp_path/'source'; pin = save_study(source, d)
    with pytest.raises(ValueError, match='source slot'):
        report.prepare_whole_body_reference_report(source, tmp_path/'out', expected_manifest_sha256=pin)


@pytest.mark.parametrize('change,message',[
    (lambda d:d['angle-proposal.json'].update(body_installed=True),'must remain separate'),
    (lambda d:d['angle-proposal.json']['reference_candidate'][0]['joints'].update(right_knee=.1),'frozen first reference'),
    (lambda d:d['angle-proposal.json']['reference_candidate'][1].update(time_s=.75),'clocks differ'),
    (lambda d:d['report.json']['camera_fit']['source_slots'].pop(),'Complete camera source slots'),
    (lambda d:d['report.json']['camera_fit']['source_slots'][5].update(camera_time_s=2.5),'missing tail'),
    (lambda d:d['report.json']['references'][1]['points'][0].update(predicted_after_px=[65.,40.]),'pixel error differs'),
    (lambda d:d['report.json']['references'][1]['points'][0].update(weight=0.,error_before_px=None,error_after_px=None),'Confident endpoint counts|confidence/mask'),
    (lambda d:d['report.json']['references'][1]['points'][0].update(target_authored_loss_units=[60.,40.]),'loss conversion'),
    (lambda d:d['plan.json'].update(camera_image_scale=.02),'authored scale'),
    (lambda d:d['report.json']['image_dimensions']['provenance'].update(decoder_verification='verified'),'decoder verification'),
    (lambda d:d['report.json']['camera_fit'].update(grid_camera_projection_calls=1),'grid poses'),
    (lambda d:d['angle-proposal.json']['input_bindings'].update(root_reference_canonical_sha256='f'*64),'input bindings'),
])
def test_resealed_camera_overlay_contradictions_fail_closed(tmp_path,change,message):
    d=camera_documents(); change(d); repin_camera_audit(d); repin_overlay(d)
    source=tmp_path/'source'; pin=save_study(source,d)
    with pytest.raises(ValueError,match=message):
        report.prepare_whole_body_reference_report(source,tmp_path/'out',expected_manifest_sha256=pin)
    assert not (tmp_path/'out').exists()


@pytest.mark.parametrize('alias,change',[
    ('inputs/evidence-original.json',lambda d:d['frames'][0].update(decoder_timestamp_s=.3)),
    ('inputs/source-camera.json',lambda d:d.update(radial_k1=.2)),
])
def test_resealed_alias_substitution_cannot_replace_original_bytes(tmp_path,alias,change):
    d=camera_documents(); change(d[alias]); d['plan.json']['source']['copied_files'][alias]=digest(d[alias])
    source=tmp_path/'source'; pin=save_study(source,d)
    with pytest.raises(ValueError,match='exact original camera/evidence bytes'):
        report.prepare_whole_body_reference_report(source,tmp_path/'out',expected_manifest_sha256=pin)


def test_invalid_known_projection_cannot_be_hidden_by_resealed_average_or_gate(tmp_path):
    d=camera_documents(); invalidate_known_projection(d); d['report.json']['image_summary']['mean_error_after_px']=4.
    source=tmp_path/'source'; pin=save_study(source,d)
    with pytest.raises(ValueError,match='image mean hides'):
        report.prepare_whole_body_reference_report(source,tmp_path/'out',expected_manifest_sha256=pin)


def test_camera_report_rejects_duplicate_human_candidate_and_missing_original_proof(tmp_path):
    d=camera_documents(); d['candidate-human.json']=deepcopy(d['inputs/human.json'])
    source=tmp_path/'both'; pin=save_study(source,d)
    with pytest.raises(ValueError,match='replacement human'):
        report.prepare_whole_body_reference_report(source,tmp_path/'both-out',expected_manifest_sha256=pin)
    d=camera_documents(); del d['inputs/camera-inputs-proof.json']
    source=tmp_path/'missing'; pin=save_study(source,d)
    with pytest.raises(ValueError,match='source-copy pins'):
        report.prepare_whole_body_reference_report(source,tmp_path/'missing-out',expected_manifest_sha256=pin)


def test_scripted_camera_details_show_saved_projection_and_preserve_unknown_tail(tmp_path, record_property):
    _,out,_,_=generated(tmp_path,camera_documents())
    harness=r"""
import fs from'node:fs';import vm from'node:vm';import assert from'node:assert/strict';import{performance}from'node:perf_hooks';
const html=fs.readFileSync(process.argv[2],'utf8'),scripts=[...html.matchAll(/<script(?:\s[^>]*)?>([\s\S]*?)<\/script>/g)];let created=0;
class E{constructor(){this.children=[];this.listeners=new Map();this.dataset={};this.style={};this.value='';this.checked=false;this.own='';created++;}set textContent(v){this.own=String(v);this.children=[];}get textContent(){return this.own+this.children.map(v=>v.textContent).join('');}set innerHTML(v){assert.fail('unsafe HTML');}append(...v){this.children.push(...v);}replaceChildren(...v){this.own='';this.children=[...v];}setAttribute(k,v){this[k]=String(v);}addEventListener(k,f){this.listeners.set(k,f);}fire(k){this.listeners.get(k)?.();}}
const ids=new Map([...html.matchAll(/\bid="([^"]+)"/g)].map(v=>[v[1],new E()])),el=k=>{assert(ids.has(k),k);return ids.get(k)};
el('data').textContent=scripts[0][1];el('mode').value='source';el('affected').checked=true;
vm.runInNewContext(scripts[1][1],{document:{getElementById:el,createElement:()=>new E(),createElementNS:()=>new E()},fetch:()=>assert.fail('network'),setTimeout:()=>assert.fail('timer')},{timeout:1000});
assert.match(el('penetration').textContent,/Proposed angles/);assert.match(el('points').textContent,/Saved camera fit/);assert.match(el('points').textContent,/Source target.*px; input/);
el('interval-sample').value='5';el('interval-sample').fire('change');assert.match(el('interval-context').textContent,/camera unavailable_original_decoded_slot at Unknown/);assert.match(el('interval-context').textContent,/2.50000 s.*2.51400 s/);
const live=n=>1+n.children.reduce((s,c)=>s+live(c),0),count=live(el('points')),line=el('lines').children[1],start=performance.now();
for(let i=0;i<100;i++){el('sample').value='2';el('sample').fire('change');el('sample').value='0';el('sample').fire('change');}assert.equal(live(el('points')),count);assert.equal(el('lines').children[1],line);
el('mode').value='grid';el('mode').fire('change');assert.match(el('points').textContent,/Unavailable/);assert.doesNotMatch(el('points').textContent,/Saved camera fit/);
process.stdout.write(JSON.stringify({detail_updates:200,point_live_nodes:count,seconds:(performance.now()-start)/1000,scope:'scripted DOM only; no browser performance claim'})+'\n');
"""
    result=subprocess.run(['node','--input-type=module','-',str(out/'index.html')],input=harness,text=True,capture_output=True,timeout=5)
    assert result.returncode==0,result.stderr
    measurement=json.loads(result.stdout); assert measurement['point_live_nodes']==109
    record_property('scripted_camera_details', json.dumps(measurement, sort_keys=True))
    print(result.stdout.strip())
