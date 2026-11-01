"""Explicit construction export preserves excluded metadata, not coordinate payloads."""

from copy import deepcopy
from hashlib import sha256
import importlib
import json
import time

import pytest

from test_multiview_landmark_evidence import local_only


SOURCE='a'*64


def fixture(*,count=7,fps=30.,interval=False):
    first=2 if interval else 0
    sampled=list(range(first,count-1,2))
    rows=[{'id':f'{SOURCE}:frame:{index}','frame':index,'time_s':index/fps,
        'decoder_timestamp_s':index/fps+.001,'status':'tracked','pose_count':1,
        'identity_status':'single_model_hypothesis_not_verified_person_count',
        'landmarks':[{'x':.2+i*.01,'y':.4+i*.001,'z':.03,'visibility':.9,'presence':.8} for i in range(33)],
        'world_landmarks':[{'x':.2+i*.01,'y':.4+i*.001,'z':.03,'visibility':.9,'presence':.8} for i in range(33)],
        'joints':{'left_knee':.5},'joint_confidence':{'left_knee':.8}} for index in sampled]
    evidence={'profile':'exercise-pose-v1','schema_version':1,'source_sha256':SOURCE,'width':100,'height':80,
        'fps':fps,'frame_count':count,'decoded_frames':count-1-first,'frames':rows,
        'model':{'name':'Authored analytical model metadata','sha256':'b'*64},
        'pose_identity':{'mode':'raw','repaired_frames':0},'scope':'Analytical source estimates, not measured identity/contact',
        'limits':['Source fixture only; no native or model work']}
    if interval:
        evidence['decode_audit']={'scope':'selected_interval','selected_interval':[first,count-2],
            'prefix_advanced_frames':first,'source_complete':False,'selected_frames_complete':True}
    return evidence


def encoded(value): return json.dumps(value,sort_keys=True,allow_nan=False).encode()


def golden_cases():
    a=fixture()
    b=fixture(); b['frames'][0].update(status='no_pose',pose_count=0,landmarks=[],world_landmarks=[],joints={})
    return [(a,None),(b,{'profile':'exercise-v1'}),(fixture(fps=30000/1001),None),(fixture(interval=True),None)]


@pytest.fixture
def api(): return importlib.import_module('weft.observations')


class ExcludedFrame(dict):
    def __deepcopy__(self,memo):
        raise AssertionError('Excluded original row reached deepcopy')

    def get(self,key,default=None):
        if key in ('landmarks','world_landmarks','joints','joint_confidence'):
            raise AssertionError('Excluded coordinate payload was inspected')
        return super().get(key,default)

    def __getitem__(self,key):
        if key in ('landmarks','world_landmarks','joints','joint_confidence'):
            raise AssertionError('Excluded coordinate payload was inspected')
        return super().__getitem__(key)


def test_excluded_rows_never_copy_inspect_or_record_coordinate_payloads(api,monkeypatch):
    evidence=fixture(); evidence['frames'][2]['landmarks']={'withheld_coordinate_sentinel':[999,999]}
    evidence['frames'][2]=ExcludedFrame(evidence['frames'][2])
    original=api._record; recorded=[]
    def record(document,row,*args,**kwargs):
        assert row['frame'] in {0,2}; recorded.append(row['frame'])
        return original(document,row,*args,**kwargs)
    monkeypatch.setattr(api,'_record',record)
    result=api.adapt_exercise(evidence,construction_frames=[0,2])
    assert recorded==[0,2]
    assert 'withheld_coordinate_sentinel' not in encoded(result).decode()
    assert result['provenance']['input_evidence_sha256']==sha256(encoded(evidence)).hexdigest()
    assert result['observations'][4]['partition']=='unassigned'
    assert result['observations'][4]['visibility']=='unknown'
    assert result['observations'][4]['evidence_ids']==[] and result['observations'][4]['measurements']=={}
    slots=result['provenance']['construction_selection']['source_slots']
    assert slots[4]['pose_evidence_id']==f'{SOURCE}:frame:4'
    assert slots[4]['decoder_timestamp_s']==4/30+.001
    assert slots[4]['pose_status']=='tracked' and slots[4]['reason']=='excluded_by_explicit_construction_selection'
    assert len(slots)==7 and slots[-1]['reason']=='metadata_only_not_decoded'


def test_all_excluded_selection_retains_every_slot_without_recording(api,monkeypatch):
    evidence=fixture(); evidence['frames']=[ExcludedFrame(row) for row in evidence['frames']]
    def forbidden(*args,**kwargs): raise AssertionError('No source rows may reach _record')
    monkeypatch.setattr(api,'_record',forbidden)
    result=api.adapt_exercise(evidence,construction_frames=[])
    assert not result['evidence'] and len(result['observations'])==6
    assert all(not row['measurements'] for row in result['observations'])
    meta=result['provenance']['construction_selection']
    assert meta['counts']=={'original_source_slots':7,'decoded_slots':6,'pose_rows':3,'selected_pose_rows':0,
        'excluded_pose_rows':3,'unsampled_decoded_slots':3,'not_decoded_source_slots':1}
    assert len(meta['source_slots'])==7


def test_default_serialized_output_is_identical_to_published_golden(api):
    # Captured before editing the published 321a4505e8d5e5a2 exporter.
    expected=['41d24fd9927fdf73c45c99bb88affa6e9177c498d8418e8ae490b6e43e951a7b',
        '7f2f6d61bfe56c5e4d9b6201c1976e75f0f48ec186db215162378e10fe74e205',
        'd372be7c6cf5503ad6b118eb015cdfce61993f3f2e58843eaf936f79728bb882',
        '1abfd56e852293a8143976a19a26a172f1caee3ff6fc144629d24ec1b9cdb06d']
    for (evidence,scene),pin in zip(golden_cases(),expected,strict=True):
        result=api.adapt_exercise(evidence,scene=scene)
        assert sha256(encoded(result)).hexdigest()==pin
        assert 'construction_selection' not in result['provenance']
        assert api.adapt_exercise(evidence,scene=scene,construction_frames=None)==result


@pytest.mark.parametrize('selection',[True,(0,),{},[True],[0.0],[2,0],[0,0],[-1],list(range(901))])
def test_selection_preflight_precedes_full_source_fingerprinting(api,monkeypatch,selection):
    def forbidden(*args,**kwargs): raise AssertionError('Invalid selection reached full source fingerprint')
    monkeypatch.setattr(api,'_base',forbidden)
    with pytest.raises(ValueError,match='Construction selection'):
        api.adapt_exercise(fixture(),construction_frames=selection)


@pytest.mark.parametrize('selection',[[1],[6],[900]])
def test_selection_requires_existing_original_sampled_ids(api,selection):
    with pytest.raises(ValueError,match='sampled pose frames'):
        api.adapt_exercise(fixture(),construction_frames=selection)


@pytest.mark.parametrize('kind',['too_many','float_frame','boolean_frame','non_row'])
def test_pose_row_envelope_is_checked_before_fingerprinting(api,monkeypatch,kind):
    evidence=fixture()
    if kind=='too_many': evidence['frames']=[{}]*901
    if kind=='float_frame': evidence['frames'][0]['frame']=0.
    if kind=='boolean_frame': evidence['frames'][0]['frame']=False
    if kind=='non_row': evidence['frames'][0]=[]
    def forbidden(*args,**kwargs): raise AssertionError('Invalid source envelope reached fingerprint')
    monkeypatch.setattr(api,'_base',forbidden)
    with pytest.raises(ValueError,match='900 original integer pose rows'):
        api.adapt_exercise(evidence,construction_frames=[])


@pytest.mark.parametrize('field,value,message',[
    ('status',{'coordinate_sentinel':[1,2]},'bounded scalar'),
    ('identity_status',['coordinate_sentinel',1,2],'bounded scalar'),
    ('status','x'*81,'bounded scalar'),('pose_count',True,'model-hypothesis count'),
    ('pose_count',1.,'model-hypothesis count'),('pose_count',129,'model-hypothesis count'),
    ('decoder_timestamp_s',{'coordinate_sentinel':[1,2]},'original decoder timestamp'),
    ('decoder_timestamp_s',True,'original decoder timestamp'),
    ('id','b'*64+':frame:4','original pose identities'),
    ('time_s',1.,'Frame/time mismatch'),
])
def test_excluded_original_metadata_is_typed_and_source_bound(api,field,value,message):
    evidence=fixture(); evidence['frames'][2][field]=value
    with pytest.raises(ValueError,match=message):
        api.adapt_exercise(evidence,construction_frames=[0])


def test_original_unknowns_derivation_names_clocks_and_nonmutation_survive(api):
    evidence=fixture(fps=30000/1001)
    evidence['frames'][2].pop('pose_count'); evidence['frames'][2].pop('identity_status')
    evidence['frames'][2]['identity_raw']={'coordinate_sentinel':[500,600]}
    original=deepcopy(evidence)
    result=api.adapt_exercise(evidence,construction_frames=[0])
    assert evidence==original and result['source']['fps']==30000/1001
    slot=result['provenance']['construction_selection']['source_slots'][4]
    assert slot['pose_count'] is None and slot['identity_status'] is None
    assert slot['pose_time_s']==original['frames'][2]['time_s']
    assert slot['decoder_timestamp_s']==original['frames'][2]['decoder_timestamp_s']
    assert slot['derivation_markers']==['identity_raw']
    assert 'coordinate_sentinel' not in encoded(result).decode()
    assert result['provenance']['construction_selection']['identity_authenticated'] is False
    assert result['provenance']['construction_selection']['contact']=='unknown'


def test_interval_uses_absolute_ids_and_retains_outside_decode_metadata_slots(api):
    evidence=fixture(count=11,interval=True)
    result=api.adapt_exercise(evidence,construction_frames=[2,4])
    assert result['schema_version']==2 and result['source']['decode_interval']==[2,9]
    assert [row['frame'] for row in result['observations']]==list(range(2,10))
    meta=result['provenance']['construction_selection']
    assert len(meta['source_slots'])==11 and meta['source_slots'][2]['pose_time_s']==2/30
    assert all(meta['source_slots'][index]['reason']=='outside_declared_decode_interval' for index in (0,1,10))
    assert meta['counts']['not_decoded_source_slots']==3


def test_full_canonical_pin_still_changes_with_excluded_input_without_exporting_it(api):
    evidence=fixture(); first=api.adapt_exercise(evidence,construction_frames=[0])
    evidence['frames'][2]['landmarks'][0]['x']=.81
    second=api.adapt_exercise(evidence,construction_frames=[0])
    assert first['provenance']['input_evidence_sha256']!=second['provenance']['input_evidence_sha256']
    assert first['evidence']==second['evidence'] and first['observations']==second['observations']
    assert first['provenance']['construction_selection']==second['provenance']['construction_selection']


def test_original_start_unavailable_is_retained_not_replaced(api):
    evidence=fixture(); evidence['frames'][0].update(status='no_pose',pose_count=0,landmarks=[],world_landmarks=[],joints={})
    result=api.adapt_exercise(evidence,construction_frames=[0,2])
    assert result['observations'][0]['visibility']=='not_detected' and result['observations'][0]['measurements']=={}
    assert result['evidence'][0]['payload']['frame']==0 and result['evidence'][0]['payload']['pose_count']==0
    assert result['provenance']['construction_selection']['construction_frames']==[0,2]


def test_partitioned_export_joins_actual_custody_and_annotations_without_excluded_queries(api,monkeypatch):
    from test_pose_precursor_custody import documents
    source=documents(context_change=lambda value:value['frames'][2].update(landmarks={'withheld_coordinate_sentinel':[999,999]}))
    context=json.loads(source['context_pose_document'])
    exported=api.adapt_exercise(context,construction_frames=[0,2]); raw=encoded(exported); pin=sha256(raw).hexdigest()
    module=importlib.import_module('weft.pose_precursor_custody')
    custody=module.prepare_pose_precursor_custody(**source)
    original=module._Qualification.qualify; queried=[]
    def qualify(self,frame):
        assert frame in (0,2); queried.append(frame); return original(self,frame)
    monkeypatch.setattr(module._Qualification,'qualify',qualify)
    names=('left_wrist','right_wrist')
    channels=[{'id':name,'entity_ref':{'document_sha256':pin,'object_id':'person-0'},
        'point_role':name,'surface_id':'unknown-surface'} for name in names]
    result=importlib.import_module('weft.pose_surface_annotations').prepare_pose_surface_annotations(
        {pin:raw},pose_document=source['context_pose_document'],source_sha256=SOURCE,channels=channels,
        landmark_bindings=[{'channel_id':name,'landmark_name':name,'uncertainty_px':2.} for name in names],
        minimum_confidence=.5,uncertainty_method='Authored analytical two-pixel range',
        uncertainty_note='Not calibrated from confidence; no contact claim',precursor_custody=custody)
    assert set(queried)=={0,2} and result.describe()['precursor_custody']['qualification']['requested_frames']==2
    assert all(result.frame(index)['channels'][0]['pixel'] is not None for index in (0,2))
    assert result.frame(4)['channels'][0]['reason']=='excluded_evidence_partition'
    assert result.frame(4)['precursor_custody']['status']=='not_queried'
    assert result.describe()['counts']['retained_source_frames']==7
    assert result.describe()['counts']['channel_slots']==14 and result.describe()['contact']=='unknown'
    assert 'withheld_coordinate_sentinel' not in result.annotation_document().decode()


def test_maximum_pose_metadata_work_retains_all_slots_with_small_selected_payload(api,record_property):
    evidence=fixture(count=1800)
    tick=time.perf_counter(); result=api.adapt_exercise(evidence,construction_frames=[0,2]); seconds=time.perf_counter()-tick
    size=len(encoded(result)); meta=result['provenance']['construction_selection']
    record_property('900_pose_1800_slot_export_seconds',seconds)
    record_property('canonical_output_bytes',size)
    record_property('source_slots',len(meta['source_slots']))
    assert len(meta['source_slots'])==1800 and meta['counts']['pose_rows']==900
    assert len(result['evidence'])==2 and len(result['observations'])==1799
    assert meta['counts']['excluded_pose_rows']==898 and seconds<5 and size<2<<20
