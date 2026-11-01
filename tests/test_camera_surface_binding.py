"""Analytical fixed-coordinate qualification, without fitting or native evaluation."""

from copy import deepcopy
from hashlib import sha256
import importlib
import json
import time

import pytest

from test_multiview_landmark_evidence import denied, local_only


def raw(value):
    return json.dumps(value,allow_nan=False).encode()


@pytest.fixture
def api():
    return importlib.import_module('weft.camera_surface_binding')


@pytest.fixture
def data():
    fitter=importlib.import_module('weft.camera_body_fit')
    return importlib.import_module('test_camera_body_fit').fixture(fitter,fps=30000/1001,radial=.00047528394168603385)


def arguments(data):
    meta=data['envelopes'].describe()
    return {'body_document':raw(data['body']),'camera_document':raw(data['camera']),
        'calibration_world_declaration':raw(meta['calibration_world_alignment']),
        'human_entity_ref':deepcopy(data['binding']['human_entity_ref']),
        'channel_correspondences':deepcopy(data['binding']['channels'])}


def prepare(api,data,**changes):
    return api.prepare_camera_surface_binding(data['envelopes'],**(arguments(data)|changes))


def test_exact_binding_reuses_real_interval_and_camera_fit_contract_without_evaluation(api,data,monkeypatch):
    camera_fit=importlib.import_module('weft.camera_body_fit')
    interval=importlib.import_module('weft.source_surface_interval_construction')
    monkeypatch.setattr(interval.SurfaceIntervalObjective,'evaluate',denied)
    monkeypatch.setattr(interval.SurfaceIntervalObjective,'evaluate_kinematics',denied)
    monkeypatch.setattr(camera_fit.CameraBodyFit,'project',denied)
    before=raw([data['body'],data['camera'],data['binding']])
    result=prepare(api,data)
    meta=result.describe(); binding=json.loads(result.binding_document())
    assert meta['status']=='conditional_binding_available' and meta['unavailable_reasons']==[]
    assert binding==data['binding'] | {'body_to_world':binding['body_to_world']}
    assert binding['body_to_world']['spatial']=={'scale':1.,'rotation_wxyz':[1.,0.,0.,0.],'translation':[1.,2.,0.]}
    assert binding['body_to_world']['clock']=={'rate':1.,'source_origin_s':0.,'target_origin_s':0.}
    assert result.objective.envelopes is result.envelopes is data['envelopes']
    assert result.objective.body_bytes is result.body_document
    assert result.objective.binding_bytes is result.binding_document()
    fitted=camera_fit.prepare_camera_body_fit(result.body_document,raw(data['evidence']),result.camera_document,
        image_scale=.01,surface_objective=result.objective)
    fitted.validate_intervals(result.objective.describe())
    assert raw([data['body'],data['camera'],data['binding']])==before
    assert meta['original_body_dt_s']==1/30
    assert meta['camera_coupling']['clock']['source_fps']==30000/1001
    assert meta['camera_coupling']['clock']['control_dt_s']==1001/30000
    assert meta['camera_coupling']['clock']['control_dt_s']!=meta['original_body_dt_s']
    assert meta['original_start_frame']==0
    assert meta['source_counts']['retained_source_frames']==10
    assert meta['objective_counts']['original_body_references']==3
    assert meta['objective_counts']['retained_channel_slots']==10
    assert [result.frame(i)['source_frame'] for i in range(10)]==list(range(10))
    assert result.frame(1)['body_join']['channels'][0]['reason']=='source_not_sampled'
    assert result.frame(0)['body_join']['channels'][0]['status']=='conditional_interval_ready'
    assert meta['geometry_evaluations']==0 and not meta['fit_or_simulator_admission']
    assert meta['contact']==meta['load']==meta['grasp']=='unknown'


def test_raw_and_canonical_identities_are_distinct_and_claims_survive_binding(api,data):
    inputs=arguments(data); inputs['body_document']=json.dumps(data['body'],indent=2).encode()
    result=api.prepare_camera_surface_binding(data['envelopes'],**inputs)
    meta=result.describe(); pins=meta['input_bindings']; binding=json.loads(result.binding_document())
    assert pins['body_raw_sha256']==sha256(inputs['body_document']).hexdigest()
    assert pins['body_canonical_sha256']==api.canonical_hash(data['body'])
    assert pins['body_raw_sha256']!=pins['body_canonical_sha256']
    assert pins['camera_raw_sha256']==sha256(inputs['camera_document']).hexdigest()
    assert pins['camera_canonical_sha256']==data['body']['camera_reference']['calibration_sha256']
    assert binding['body_sha256']==pins['body_raw_sha256']
    claim=binding['body_to_world']['claims']['spatial']
    assert claim['status']=='assumed' and claim['evidence_refs']==[]
    assert pins['declaration_raw_sha256'] in claim['method']
    assert meta['camera_coupling_sha256'] in claim['method']
    assert 'priors' in claim['uncertainty'] and 'contact' in claim['uncertainty']


@pytest.mark.parametrize('missing',['body_document','camera_document','calibration_world_declaration',
                                    'human_entity_ref','channel_correspondences'])
def test_missing_inputs_do_not_get_inferred_defaults(api,data,missing):
    result=prepare(api,data,**{missing:None}); meta=result.describe()
    assert meta['status']=='unavailable' and meta['unavailable_reasons']
    assert meta['source_counts']['retained_source_frames']==10
    assert all(result.frame(i)['source_envelopes']==data['envelopes'].frame(i) for i in range(10))
    assert not meta['body_or_floor_installed']
    if missing in ('camera_document','calibration_world_declaration'):
        assert json.loads(result.binding_document())['body_to_world'] is None
        assert result.objective.image_admission(0)['status']=='construction_image_source_ready'
    if missing=='body_document':
        assert result.binding_document() is result.objective is None


@pytest.mark.parametrize('change,reason',[
    (lambda d:d['from_frame'].update(id='calibration-world:'+'f'*64),'calibration_or_surface_world_identity_differs'),
    (lambda d:d['to_frame'].update(id='different-surface-world'),'calibration_or_surface_world_identity_differs'),
    (lambda d:d['spatial'].update(scale=2.),'nonidentity_calibration_world_mapping_unsupported'),
    (lambda d:d['spatial'].update(translation=[0.,0.,.1]),'nonidentity_calibration_world_mapping_unsupported'),
    (lambda d:d['spatial'].update(rotation_wxyz=[0.,0.,0.,1.]),'nonidentity_calibration_world_mapping_unsupported'),
    (lambda d:d['clock'].update(rate=2.),'nonidentity_original_source_clock_unsupported'),
    (lambda d:d['clock'].update(source_origin_s=1.,target_origin_s=1.),'nonidentity_original_source_clock_unsupported'),
])
def test_no_reframing_scaling_or_retiming(api,data,change,reason):
    declaration=json.loads(arguments(data)['calibration_world_declaration']); change(declaration)
    result=prepare(api,data,calibration_world_declaration=raw(declaration))
    assert reason in result.describe()['unavailable_reasons']
    assert result.describe()['mapping_status']=='unavailable'
    assert json.loads(result.binding_document())['body_to_world'] is None
    assert result.objective.frame(0)['channels'][0]['status']=='unavailable'


@pytest.mark.parametrize('part',['spatial','clock'])
def test_explicit_unknown_world_alignment_preserves_unknown(api,data,part):
    declaration=json.loads(arguments(data)['calibration_world_declaration'])
    declaration[part]=None; declaration['claims'][part]['status']='unknown'
    result=prepare(api,data,calibration_world_declaration=raw(declaration))
    assert 'unknown_calibration_world_alignment' in result.describe()['unavailable_reasons']
    assert json.loads(result.binding_document())['body_to_world'] is None


def test_unknown_height_or_endpoint_is_not_a_zero_target_or_contact(api,monkeypatch):
    fitter=importlib.import_module('weft.camera_body_fit')
    data=importlib.import_module('test_camera_body_fit').fixture(fitter,annotation='unknown_height')
    monkeypatch.setattr(importlib.import_module('weft.source_surface_interval_construction').SurfaceIntervalObjective,'evaluate',denied)
    result=prepare(api,data)
    assert result.describe()['status']=='conditional_binding_available'
    assert result.describe()['objective_status']=='unavailable'
    assert result.frame(0)['body_join']['channels'][0]['status']=='unavailable'
    assert result.frame(0)['source_envelopes']['anchors'][0]['world_envelope_m'] is None
    assert result.objective.image_admission(0)['status']=='construction_image_source_ready'
    correspondences=arguments(data)['channel_correspondences']
    correspondences[0].update(body_point=None,status='unknown')
    unknown=prepare(api,data,channel_correspondences=correspondences)
    assert json.loads(unknown.binding_document())['channels'][0]['body_point'] is None
    assert unknown.describe()['contact']=='unknown'


def test_foreign_entity_and_unsupported_endpoint_cannot_become_ready(api,data):
    selected=arguments(data)['human_entity_ref']; selected['object_id']='person-not-in-source'
    result=prepare(api,data,human_entity_ref=selected)
    assert 'human_entity_not_in_exact_source' in result.describe()['unavailable_reasons']
    assert result.objective.image_admission(0)['status']=='unavailable'
    selected=arguments(data)['channel_correspondences']; selected[0]['body_point']='left_heel'
    with pytest.raises(ValueError,match='core-body point'):
        prepare(api,data,channel_correspondences=selected)


@pytest.mark.parametrize('part',['spatial','clock'])
def test_metadata_cannot_upgrade_a_declared_prior_to_measurement(api,data,part):
    declaration=json.loads(arguments(data)['calibration_world_declaration'])
    declaration['claims'][part]['status']='observed'
    with pytest.raises(ValueError,match='assumed/unknown'):
        prepare(api,data,calibration_world_declaration=raw(declaration))


def test_equivalent_camera_reserialization_is_not_exact_camera_custody(api,data):
    camera=json.dumps(data['camera'],indent=2).encode()
    assert api.canonical_hash(json.loads(camera))==api.canonical_hash(data['camera'])
    with pytest.raises(ValueError,match='exactly match'):
        prepare(api,data,camera_document=camera)


def test_original_body_camera_pin_mismatch_fails(api,data):
    body=deepcopy(data['body']); body['camera_reference']['calibration_sha256']='d'*64
    with pytest.raises(ValueError): prepare(api,data,body_document=raw(body))


def test_supplied_nonzero_ground_survives_exact_inverse_without_floor_change(api,data):
    value=deepcopy(data['views'])
    value['ground']['height_m']=.25
    value['body']['mobility']['base_height']=.75
    importlib.import_module('test_multiview_foot_reference')._pin(value,importlib.import_module('weft.multiview_foot_reference'))
    body_raw=raw(value['body'])
    result=prepare(api,data,body_document=body_raw)
    assert json.loads(result.binding_document())['body_to_world']['spatial']['translation']==[1.,2.,.25]
    assert result.body_document is body_raw
    assert result.describe()['camera_coupling']['ground_sha256']==api.canonical_hash(value['ground'])
    assert json.loads(result.body_document)['camera_reference']['ground']['height_m']==.25
    assert result.describe()['geometry_modified'] is False


def test_original_camera_clock_change_is_not_hidden_by_approximate_rate_check(api,data):
    inputs=deepcopy(data['envelope_inputs']); value=deepcopy(data['views']); camera=value['cameras'][0]
    camera['frames'][2]['time_s']+=1e-9
    importlib.import_module('test_multiview_foot_reference')._pin(value,importlib.import_module('weft.multiview_foot_reference'))
    inputs['camera_document']=raw(camera)
    surfaces=json.loads(inputs['surface_interval_document']); surfaces['calibration_sha256']=sha256(inputs['camera_document']).hexdigest()
    inputs['surface_interval_document']=raw(surfaces)
    envelopes=importlib.import_module('weft.source_surface_envelopes').prepare_source_surface_envelopes(**inputs)
    with pytest.raises(ValueError,match='exact body/camera original source clock differs'):
        api.prepare_camera_surface_binding(envelopes,**(arguments(data)|{'body_document':raw(value['body']),
            'camera_document':raw(camera),'calibration_world_declaration':raw(envelopes.describe()['calibration_world_alignment'])}))


@pytest.mark.parametrize('key,limit',[('body_document',8<<20),('camera_document',4<<20),('calibration_world_declaration',16<<10)])
def test_byte_limits_precede_parsing(api,data,monkeypatch,key,limit):
    monkeypatch.setattr(api,'_document',denied)
    with pytest.raises(ValueError,match='bounded exact input'):
        prepare(api,data,**{key:b' '*(limit+1)})


def test_complete_slot_bound_precedes_body_parsing(api,data,monkeypatch):
    envelopes=data['envelopes']
    oversized=type(envelopes)(envelopes.observation_inputs,envelopes.additional_inputs,envelopes._metadata,
                              (envelopes._rows[0],)*1801)
    monkeypatch.setattr(api,'_document',denied)
    with pytest.raises(ValueError,match='bounded prepared'):
        api.prepare_camera_surface_binding(oversized,**arguments(data))


def test_timeout_retains_no_partial_binding(api,data,monkeypatch):
    monkeypatch.setattr(api,'MAX_SECONDS',0.)
    with pytest.raises(TimeoutError): prepare(api,data)


def test_cached_queries_are_detached_without_requalification(api,data,monkeypatch,record_property):
    started=time.perf_counter(); result=prepare(api,data); preparation=time.perf_counter()-started
    for name in ('build_camera_coupling','prepare_surface_interval_objective','validate_camera','load_world_alignment','_document'):
        monkeypatch.setattr(api,name,denied)
    original=result.binding_document()
    started=time.perf_counter()
    for i in range(1000): result.frame(i%10)
    queries=time.perf_counter()-started
    row=result.frame(0); row['body_join']['channels'].clear()
    assert result.frame(0)['body_join']['channels']
    meta=result.describe(); meta['camera_coupling']['world_to_local']['translation'][0]=900.
    assert result.describe()['camera_coupling']['world_to_local']['translation'][0]==-1.
    assert result.binding_document() is original
    measurement={'source_frames':10,'body_references':3,'channels':1,'preparation_seconds':preparation,
                 'cached_1000_queries_seconds':queries,'cache_bytes':result.describe()['cache_bytes'],
                 'body_bytes':len(result.body_document),'camera_bytes':len(result.camera_document),
                 'native_or_objective_evaluations':0}
    record_property('camera_surface_binding_workload',json.dumps(measurement,sort_keys=True))
