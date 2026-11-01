"""Exact saved-raster geometry and source custody; no image decoder or model."""

from copy import deepcopy
from fractions import Fraction
from functools import lru_cache
from hashlib import sha256
import importlib
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from test_multiview_landmark_evidence import denied, local_only
from test_pose_surface_annotations import declared_raw_pose, fixture as pose_fixture, raw


@pytest.fixture
def api():
    return importlib.import_module('weft.equipment_silhouette_evidence')


@lru_cache(maxsize=1)
def segmentation_adapter():
    # The unchanged pure adapter imports an unrelated native export helper at module scope.
    spec=importlib.util.spec_from_file_location('weft._silhouette_test_segmentation_observations',
        Path(__file__).parents[1]/'src/weft/segmentation_observations.py')
    module=importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules,{'weft.benchmark':SimpleNamespace(seal_suite=denied)}):
        spec.loader.exec_module(module)
    return module.adapt_segmentation


def fixture(api, *, pose_change=None, observation_change=None, count=7,
            raster=b'\1\1\1\1\0\1\1\1\1', dimensions=(3,3), box=(32,30,38,36), uncertainty=.25,
            equipment_change=None, mask_change=None, proof=None, supplied_pose=None):
    def change(pose):
        declared_raw_pose(pose)
        for row in pose['frames']:
            row['decoder_timestamp_s']=row['frame']/30
            for index in (15,16):
                row['landmarks'][index].update(x=35/pose['width'],y=33/pose['height'])
        if pose_change: pose_change(pose)
    inputs=pose_fixture(count,names=('left_wrist','right_wrist'),pose_change=change,
                        observation_change=observation_change)
    if supplied_pose is not None:
        pose=deepcopy(supplied_pose); declared_raw_pose(pose)
        for row in pose['frames']:
            for index in (15,16): row['landmarks'][index].update(x=35/pose['width'],y=33/pose['height'])
        if pose_change: pose_change(pose)
        document=importlib.import_module('weft.observations').adapt_exercise(pose)
        if observation_change: observation_change(document)
        encoded=raw(document); digest=sha256(encoded).hexdigest()
        inputs.update(pose_document=raw(pose),source_sha256=pose['source_sha256'],observation_documents={digest:encoded})
        for channel in inputs['channels']: channel['entity_ref']['document_sha256']=digest
    pose=json.loads(inputs['pose_document'])
    for binding in inputs['landmark_bindings']: binding['uncertainty_px']=uncertainty
    prepared=importlib.import_module('weft.pose_surface_annotations').prepare_pose_surface_annotations(**inputs)
    source=pose['source_sha256'] if proof is None else proof.derivative_sha256
    times=[index/pose['fps'] for index in range(pose['decoded_frames'])] if proof is None else [float(row.derivative_decoder_pts*Fraction(proof.time_base)) for row in proof.frames]
    segmentation={'profile':'segmentation-evidence-v1','physics_ready':False,
        'source':{'submitted_video_sha256':source,'width_px':pose['width'],'height_px':pose['height'],'frame_count':pose['decoded_frames']},
        'time':{'timestamps_s':times},'objects':[{'id':'sam-2','raw_model_id':'2'}],
        'observations':[],'provenance':{'scope':'Authored analytical saved-raster fixture; no decoder/model invocation'},
        'limits':[],'identity_groups':[]}
    if proof is not None: segmentation['source']['original_source_sha256']=pose['source_sha256']
    left,top,right,bottom=box
    for index,stamp in enumerate(times):
        row={'id':f'{source}:frame:{index}:model-object:2','frame':index,'time_s':stamp,
             'object_id':'sam-2','raw_model_id':'2','phase':'post_initialization','proposal_status':'returned',
             'geometry':{'box_xyxy_halfopen':list(box),'box_xywh':[left,top,right-left,bottom-top],
                         'center_px':[(left+right)/2,(top+bottom)/2]},
             'confidence':{'box':.9,'mask':.8},
             'identity':{'status':'assumed_model_identity','unresolved_group_id':None,'gap_seen':False,'physical_identity_verified':False},
             'mask_provenance':{'encoded_sha256':'c'*64},'raw_model_output':{'analytical_fixture':True}}
        if proof is not None:
            mapped=proof.frames[index]
            row['original_source']={'sha256':pose['source_sha256'],'frame':mapped.source_frame,
                'time_s':float(mapped.source_native_pts*Fraction(proof.time_base))}
        segmentation['observations'].append(row)
    equipment=segmentation_adapter()(segmentation)
    masks={'profile':'decoded-segmentation-masks-v1','physics_ready':False,
        'contract':{'method':'saved-sam-mask-raster-v1','mapping':api.CELL_MAPPING},
        'source':deepcopy(segmentation['source']),'original_evidence_sha256':api.canonical_hash(segmentation),
        'frames':[],'observations':[]}
    rasters={}
    for index,source_row in enumerate(segmentation['observations']):
        row={key:deepcopy(source_row[key]) for key in ('id','frame','time_s','object_id','raw_model_id','phase',
            'proposal_status','identity','confidence')}
        if 'original_source' in source_row: row['original_source']=deepcopy(source_row['original_source'])
        path=f'masks/{index:05d}-0.png'; rasters[path]=raster
        row.update(raw_observation_sha256=api.canonical_hash(source_row),mask_status='decoded',box=deepcopy(source_row['geometry']),
            raster_path=path,raster_dimensions=list(dimensions),raster_sha256=sha256(raster).hexdigest(),
            silhouette={'foreground_pixels':raster.count(b'\1')},encoded_sha256='c'*64)
        masks['observations'].append(row)
        masks['frames'].append({'frame':index,'time_s':times[index],'observation_indices':[index]})
    if equipment_change: equipment_change(equipment)
    if mask_change: mask_change(masks)
    calls=[]
    def loader(request):
        assert request['maximum_bytes']==request['dimensions'][0]*request['dimensions'][1]
        assert request['maximum_bytes']<=request['remaining_total_bytes']<=api.MAX_RASTER_BYTES
        assert request['maximum_seconds']>0 and request['deadline_monotonic']>time.monotonic()
        assert request['mask_document_sha256']==sha256(raw(masks)).hexdigest()
        calls.append(deepcopy(request))
        return rasters[request['path']]
    args={'equipment_document':raw(equipment),'mask_document':raw(masks),'equipment_ids':['sam-2'],
          'wrist_channel_ids':['channel-0','channel-1'],'raster_loader':loader,'source_equivalence':proof}
    return prepared,args,calls,rasters


def run(api, **kwargs):
    prepared,args,calls,rasters=fixture(api,**kwargs)
    return api.prepare_equipment_silhouette_evidence(prepared,**args),calls


def test_holes_exact_cell_geometry_and_complete_source_slots(api):
    result,calls=run(api)
    meta=result.describe(); first=result.frame(0)['pairs'][0]
    assert meta['counts']=={'source_frames':7,'decoded_frames':6,'metadata_frames':7,'pose_slots':3,'pair_slots':14,
        'status_counts':{'conditional_image_exclusion':6,'source_pose_not_sampled':6,'metadata_only_not_decoded':2}}
    assert first['status']=='conditional_image_exclusion' and first['intersects'] is False
    assert first['distance_infimum_squared_px2']=={'numerator':9,'denominator':16,'outward_interval':[.5625,.5625]}
    assert first['pixel']==[35.,33.] and first['authored_uncertainty_px']==.25
    assert first['source_confidence']==.8 and first['source_visibility_score']==.9 and first['source_presence_score']==.8
    assert first['representation']=='declared_raw_model_estimate_not_authenticated_identity'
    assert first['equipment_time_s']==first['mask_time_s']==0
    assert len(calls)==3 and meta['work']['pair_relations']==6
    assert meta['work']['binary_raster_bytes']==27 and meta['work']['pose_frame_queries']==3
    assert all(meta[key]=='unknown' for key in ('contact','physical_noncontact','grasp','load'))
    assert meta['body_or_floor_installed'] is False
    assert result.frame(6)['pairs'][0]['mask_evidence_id'] is None


def test_filled_silhouette_is_conditional_intersection_not_contact(api):
    result,_=run(api,raster=b'\1'*9)
    assert result.frame(0)['pairs'][0]['status']=='conditional_image_intersection'
    assert result.frame(0)['pairs'][0]['contact']=='unknown'


@pytest.mark.parametrize('rectangle,intersects,touches,distance',[
    ((2,2,0,0),False,True,Fraction(0)),
    ((0,0,0,0),True,False,Fraction(0)),
    ((1,1,2,2),False,True,Fraction(0)),
    ((2,3,2,3),False,True,Fraction(0)),
    ((3,4,3,4),False,False,Fraction(2)),
    ((-1,0,-1,0),True,False,Fraction(0)),
])
def test_closed_rectangles_and_half_open_cells_are_exact(api,rectangle,intersects,touches,distance):
    result=api._relation(tuple(map(Fraction,rectangle)),((Fraction(0),Fraction(2),Fraction(0),Fraction(2)),),time.monotonic()+1)
    assert result['intersects'] is intersects and result['touches_excluded_edge'] is touches
    value=result['distance_infimum_squared_px2']; assert Fraction(value['numerator'],value['denominator'])==distance


def test_noninteger_cell_scaling_and_outward_rounding(api):
    work={'foreground_runs':0}
    runs,foreground=api._runs(b'\1\0\1',(1,3),[0,0,10,7],time.monotonic()+1,work)
    assert runs==((Fraction(0),Fraction(10,3),Fraction(0),Fraction(7)),
                  (Fraction(20,3),Fraction(10),Fraction(0),Fraction(7)))
    assert foreground==2
    result=api._relation((Fraction(5),Fraction(5),Fraction(3),Fraction(3)),runs,time.monotonic()+1)
    value=result['distance_infimum_squared_px2']; assert Fraction(value['numerator'],value['denominator'])==Fraction(25,9)
    lower,upper=map(Fraction,value['outward_interval']); assert lower<=Fraction(25,9)<=upper


def test_empty_mask_preserves_unknown(api):
    result,_=run(api,raster=b'\0'*9)
    pair=result.frame(0)['pairs'][0]
    assert pair['reason']=='empty_model_silhouette' and pair['intersects'] is None


def test_explicit_selection_not_implicit_prefix_and_cached_queries(api,monkeypatch):
    prepared,args,calls,_=fixture(api)
    original=type(prepared).frame; queried=[]
    def frame(self,index):
        assert index in (0,4)
        queried.append(index); return original(self,index)
    monkeypatch.setattr(type(prepared),'frame',frame)
    result=api.prepare_equipment_silhouette_evidence(prepared,construction_frames=[0,4],**args)
    assert queried==[0,4] and len(calls)==2
    assert result.frame(2)['pairs'][0]['reason']=='excluded_by_explicit_frame_selection'
    snapshot=result.frame(0); snapshot['pairs'].clear()
    assert len(result.frame(0)['pairs'])==2
    before=len(calls)
    for _ in range(50): result.frame(4)
    assert len(calls)==before and queried==[0,4]
    assert result.describe()['construction_frames']==[0,4]


@pytest.mark.parametrize('kind,reason',[
    ('preinit','mask_pre_initialization'),('missing','equipment_occluded_or_unavailable'),
    ('occluded','equipment_occluded_or_unavailable'),('decode','mask_not_decoded'),
    ('phase','mask_phase_unqualified'),('identity','mask_identity_status_unqualified'),
])
def test_unqualified_equipment_does_not_query_raster_or_wrist(api,monkeypatch,kind,reason):
    def equipment(doc):
        row=doc['observations'][0]
        if kind in ('missing','occluded'):
            row['visibility']='not_detected' if kind=='missing' else 'partial'; row['measurements']={}
        if kind=='identity': doc['evidence'][0]['payload']['identity']['status']='unknown'
    def mask(doc):
        row=doc['observations'][0]
        if kind=='preinit': row['phase']='pre_initialization'
        if kind=='decode': row['mask_status']='decode_failed'
        if kind=='phase': row['phase']='future_unknown_phase'
        if kind=='identity':
            row['identity']['status']='unknown'
            # Re-pin the explicitly changed analytical model proposal, not real evidence.
            source_row=deepcopy(base_equipment['evidence'][0]['payload']); source_row['identity']['status']='unknown'
            row['raw_observation_sha256']=api.canonical_hash(source_row)
    _,base_args,_,_=fixture(api); base_equipment=json.loads(base_args['equipment_document'])
    prepared,args,calls,_=fixture(api,equipment_change=equipment,mask_change=mask)
    original=type(prepared).frame
    def frame(self,index):
        assert index!=0
        return original(self,index)
    monkeypatch.setattr(type(prepared),'frame',frame)
    result=api.prepare_equipment_silhouette_evidence(prepared,**args)
    assert result.frame(0)['pairs'][0]['reason']==reason and len(calls)==2


def test_canonical_holdout_retained_without_payload_queries(api,monkeypatch):
    def heldout(doc): doc['observations'][0]['partition']='audit'
    prepared,args,calls,_=fixture(api,observation_change=heldout,equipment_change=heldout)
    original=type(prepared).frame
    def frame(self,index):
        assert index!=0
        return original(self,index)
    monkeypatch.setattr(type(prepared),'frame',frame)
    result=api.prepare_equipment_silhouette_evidence(prepared,**args)
    pair=result.frame(0)['pairs'][0]
    assert pair['reason']=='human_partition_excluded' and pair['human_partition']==pair['equipment_partition']=='audit'
    assert pair['pixel'] is None and len(calls)==2


@pytest.mark.parametrize('selection',['canonical_holdout','explicit_exclusion'])
def test_excluded_identity_payload_cannot_leak_coordinates(api,monkeypatch,selection):
    def heldout(doc): doc['observations'][0]['partition']='audit'
    def sentinel(doc): doc['observations'][0]['identity']['coordinate_sentinel']={'pixel':[35,33]}
    prepared,args,calls,_=fixture(api,count=3,mask_change=sentinel,
        observation_change=heldout if selection=='canonical_holdout' else None,
        equipment_change=heldout if selection=='canonical_holdout' else None)
    monkeypatch.setattr(type(prepared),'frame',denied)
    args['raster_loader']=denied
    if selection=='explicit_exclusion': args['construction_frames']=[]
    result=api.prepare_equipment_silhouette_evidence(prepared,**args)
    output=[result.frame(index) for index in range(3)]
    assert 'coordinate_sentinel' not in json.dumps(output)
    assert all(pair['model_identity'] is None for row in output for pair in row['pairs'])
    assert calls==[] and result.describe()['work']['pose_frame_queries']==result.describe()['work']['raster_loads']==0


def test_qualified_identity_projects_only_bounded_noncoordinate_fields(api):
    def equipment(doc): doc['evidence'][0]['payload']['identity']['coordinate_sentinel']={'pixel':[35,33]}
    _,base_args,_,_=fixture(api,count=3)
    payload=json.loads(base_args['equipment_document'])['evidence'][0]['payload']
    payload['identity']['coordinate_sentinel']={'pixel':[35,33]}
    def mask(doc):
        doc['observations'][0]['identity']['coordinate_sentinel']={'pixel':[35,33]}
        doc['observations'][0]['raw_observation_sha256']=api.canonical_hash(payload)
    result,_=run(api,count=3,equipment_change=equipment,mask_change=mask)
    pair=result.frame(0)['pairs'][0]
    assert pair['model_identity']=={'status':'assumed_model_identity','unresolved_group_id':None,
        'gap_seen':False,'physical_identity_verified':False}


@pytest.mark.parametrize('field',['phase','mask_status','time_s'])
def test_excluded_metadata_fields_reject_coordinate_dicts_before_queries(api,monkeypatch,field):
    def heldout(doc): doc['observations'][0]['partition']='audit'
    prepared,args,calls,_=fixture(api,count=3,observation_change=heldout,equipment_change=heldout,
        mask_change=lambda doc:doc['observations'][0].update({field:{'pixel':[35,33]}}))
    monkeypatch.setattr(type(prepared),'frame',denied); args['raster_loader']=denied
    with pytest.raises(ValueError,match='bounded scalar mask'):
        api.prepare_equipment_silhouette_evidence(prepared,**args)
    assert calls==[]


@pytest.mark.parametrize('key,value',[('gap_seen',0),('unresolved_group_id',{'pixel':[35,33]}),('unresolved_group_id','x'*161)])
def test_invalid_known_identity_fields_abstain_before_queries(api,monkeypatch,key,value):
    _,base_args,_,_=fixture(api,count=3)
    payload=json.loads(base_args['equipment_document'])['evidence'][0]['payload']
    payload['identity'][key]=value
    def equipment(doc): doc['evidence'][0]['payload']=deepcopy(payload)
    def mask(doc):
        doc['observations'][0]['identity']=deepcopy(payload['identity'])
        doc['observations'][0]['raw_observation_sha256']=api.canonical_hash(payload)
    prepared,args,calls,_=fixture(api,count=3,equipment_change=equipment,mask_change=mask)
    monkeypatch.setattr(type(prepared),'frame',denied); args['raster_loader']=denied
    result=api.prepare_equipment_silhouette_evidence(prepared,**args)
    pair=result.frame(0)['pairs'][0]
    assert pair['reason']=='mask_identity_status_unqualified' and pair['model_identity'] is None
    assert calls==[]


def test_conflicting_source_partitions_fail_closed_before_raster(api):
    prepared,args,calls,_=fixture(api,equipment_change=lambda d:d['observations'][0].update(partition='audit'))
    with pytest.raises(ValueError,match='Conflicting canonical frame partitions'):
        api.prepare_equipment_silhouette_evidence(prepared,**args)
    assert calls==[]


@pytest.mark.parametrize('kind,reason',[
    ('confidence','below_authored_confidence_threshold'),
    ('temporal','unsupported_or_inconsistent_identity_declaration'),
    ('outside','authored_pixel_range_outside_source_image'),
])
def test_unqualified_pose_never_loads_equipment_raster(api,kind,reason):
    def change(pose):
        if kind=='temporal': pose['pose_identity']['mode']='temporal'
        for row in pose['frames']:
            for index in (15,16):
                if kind=='confidence': row['landmarks'][index]['presence']=.1
                if kind=='outside': row['landmarks'][index]['x']=0
    result,calls=run(api,pose_change=change)
    assert calls==[] and result.frame(0)['pairs'][0]['reason']==reason
    if kind=='outside': assert result.frame(0)['pairs'][0]['pixel']==[0.,33.]


@pytest.mark.parametrize('field,value,message',[
    ('raw_observation_sha256','f'*64,'raw proposal'),('box',{},'pixel-cell box'),
    ('encoded_sha256','f'*64,'encoded provenance'),('time_s',.001,'mask frame clock'),
    ('original_source',{'sha256':'a'*64,'frame':0,'time_s':0.},'raw proposal'),
    ('raster_path','../escape.png','raster descriptor'),('raster_sha256','z'*64,'raster descriptor'),
    ('raster_dimensions',[True,3],'raster descriptor'),
])
def test_changed_custody_or_loader_descriptor_fails_before_read(api,field,value,message):
    prepared,args,calls,_=fixture(api,mask_change=lambda d:d['observations'][0].update({field:value}))
    with pytest.raises(ValueError,match=message): api.prepare_equipment_silhouette_evidence(prepared,**args)
    assert calls==[]


@pytest.mark.parametrize('kind,message',[
    ('hash','raster bytes differ'),('size','raster bytes differ'),('binary','must be binary'),('count','foreground count'),
])
def test_raster_bytes_are_exactly_requalified(api,kind,message):
    kwargs={}
    if kind=='binary': kwargs['raster']=b'\2'*9
    if kind=='count': kwargs['mask_change']=lambda d:d['observations'][0]['silhouette'].update(foreground_pixels=9)
    prepared,args,calls,rasters=fixture(api,**kwargs)
    if kind=='hash': rasters['masks/00000-0.png']=b'\0'*9
    if kind=='size': rasters['masks/00000-0.png']=b'\1'*8
    with pytest.raises(ValueError,match=message): api.prepare_equipment_silhouette_evidence(prepared,**args)
    assert len(calls)==1


@pytest.mark.parametrize('limit,value,loads',[
    ('MAX_RASTER_BYTES',17,1),('MAX_LOADS',1,1),('MAX_RUNS',2,1),('MAX_CACHE_BYTES',1,1),
])
def test_exhausted_work_fails_whole_preparation_without_truncation(api,monkeypatch,limit,value,loads):
    prepared,args,calls,_=fixture(api); monkeypatch.setattr(api,limit,value)
    with pytest.raises(ValueError,match='bound|budget|cache'):
        api.prepare_equipment_silhouette_evidence(prepared,**args)
    assert len(calls)==loads


def test_loader_deadline_crossing_never_publishes_partial_result(api,monkeypatch):
    prepared,args,calls,_=fixture(api); original=time.monotonic; late=[False]
    monkeypatch.setattr(api.time,'monotonic',lambda:original()+20 if late[0] else original())
    loader=args['raster_loader']
    def late_loader(request):
        result=loader(request); late[0]=True; return result
    args['raster_loader']=late_loader
    with pytest.raises(TimeoutError,match='cooperative deadline'): api.prepare_equipment_silhouette_evidence(prepared,**args)
    assert len(calls)==1


def test_verified_derivative_keeps_original_native_and_decoder_clocks(api,tmp_path):
    from test_exercise_support_evidence import _equivalent_equipment
    data,_,proof=_equivalent_equipment(tmp_path)
    result,calls=run(api,supplied_pose=data,proof=proof)
    assert len(calls)==3
    first=result.frame(0); pair=first['pairs'][0]
    assert first['pose_decoder_timestamp_s']==0 and pair['equipment_time_s']==.002
    assert first['source_mapping']['original_native_pts']==first['source_mapping']['derivative_native_pts']==3
    assert result.describe()['source_equivalence']['manifest_sha256']==proof.manifest_sha256
    assert result.describe()['source_sha256']!=result.describe()['equipment_source_sha256']


@pytest.mark.parametrize('kind',['missing_proof','wrong_pose_clock','changed_original_clock'])
def test_derivative_mapping_cannot_be_implied_or_retimed(api,tmp_path,kind):
    from test_exercise_support_evidence import _equivalent_equipment
    data,_,proof=_equivalent_equipment(tmp_path)
    if kind=='wrong_pose_clock': data['frames'][0]['decoder_timestamp_s']=.002
    prepared,args,calls,_=fixture(api,supplied_pose=data,proof=proof)
    if kind=='missing_proof': args['source_equivalence']=None
    if kind=='changed_original_clock':
        equipment=json.loads(args['equipment_document']); masks=json.loads(args['mask_document'])
        payload=equipment['evidence'][0]['payload']; payload['original_source']['time_s']=0.
        masks['observations'][0]['original_source']['time_s']=0.
        masks['observations'][0]['raw_observation_sha256']=api.canonical_hash(payload)
        args.update(equipment_document=raw(equipment),mask_document=raw(masks))
    if kind=='wrong_pose_clock':
        result=api.prepare_equipment_silhouette_evidence(prepared,**args)
        assert result.frame(0)['pairs'][0]['reason']=='pose_decoder_clock_unavailable_or_different_from_equivalence'
        assert len(calls)==2
    else:
        with pytest.raises(ValueError,match='Equivalence|native clock'): api.prepare_equipment_silhouette_evidence(prepared,**args)
        assert calls==[]


def test_representative_raster_preparation_and_cached_queries(api,record_property):
    dimensions=(64,64); raster=bytes((row+col)%2 for row in range(64) for col in range(64))
    prepared,args,calls,_=fixture(api,raster=raster,dimensions=dimensions)
    tick=time.perf_counter(); result=api.prepare_equipment_silhouette_evidence(prepared,**args); elapsed=time.perf_counter()-tick
    before=len(calls); tick=time.perf_counter()
    for index in range(1000): result.frame(index%7)
    query=time.perf_counter()-tick
    record_property('preparation_seconds',elapsed); record_property('cached_1000_queries_seconds',query)
    record_property('work',json.dumps(result.describe()['work'],sort_keys=True))
    assert elapsed<api.MAX_SECONDS and len(calls)==before==3
    assert result.describe()['work']['foreground_runs']==6144
    assert result.describe()['work']['binary_raster_bytes']==12288
