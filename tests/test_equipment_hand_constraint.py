"""Pure named-point restriction of the unchanged co-motion selector."""

import builtins
from copy import deepcopy
from dataclasses import replace
from hashlib import sha256
import importlib
import json
import os
from pathlib import Path
import time

import pytest

from test_equipment_hand_evidence import fixture as hand_fixture
from test_equipment_silhouette_constraint import fixture as wrist_fixture, constrained as wrist_constrained
from test_exercise_support_evidence import prior
from test_multiview_landmark_evidence import local_only
from weft.exercise_support_evidence import estimate_exercise_support


def encode(value):
    return json.dumps(value, sort_keys=True, allow_nan=False).encode()


def fixture(*, count=31, pose_change=None, hands=('left', 'right'), raster=b'\1\1\1\1\0\1\1\1\1', dense=False):
    selected = list(range(0, count-1, 1 if dense else 2))
    selected = selected[:int(.6 * len(selected))]
    def pose_adjust(pose):
        if dense:
            originals={row['frame']:row for row in pose['frames']}
            pose['frames']=[dict(deepcopy(originals[frame-frame%2]), id=f"{pose['source_sha256']}:frame:{frame}",
                frame=frame,time_s=frame/pose['fps'],decoder_timestamp_s=frame/pose['fps']) for frame in range(count-1)]
        for row in pose['frames']:
            for index, (x,y) in {11:(.3,.15),12:(.7,.15),23:(.35,.75),24:(.65,.75)}.items():
                row['landmarks'][index].update(x=x,y=y)
        if pose_change:pose_change(pose)
    def partition(document):
        for row in document['observations']:
            if row['partition']!='unsampled' and row['frame'] not in selected:
                row.update(partition='unassigned',visibility='unknown',evidence_ids=[],measurements={})
    annotations,args,calls,*_ = hand_fixture(count=count,pose_change=pose_adjust,hands=hands,
                                           observation_change=partition,raster=raster)
    prepared=importlib.import_module('weft.equipment_hand_evidence').prepare_equipment_hand_evidence(
        annotations,construction_frames=selected,**args)
    pose=json.loads(annotations.pose_document)
    selector={'camera_prior':prior(pose['source_sha256']),'equipment_observations':args['equipment_document'],
              'equipment_ids':args['equipment_ids']}
    return pose,selector,prepared,estimate_exercise_support(pose,**selector),calls


def constrained(baseline,prepared):
    return importlib.import_module('weft.equipment_hand_constraint').constrain_equipment_hand_support(
        baseline,prepared,deadline=time.monotonic()+10)


def recover(result):
    baseline=deepcopy(result); extra=baseline.pop('equipment_hand_constraint')
    baseline['counts']['equipment_status_counts']=extra['baseline_equipment_status_counts']
    for row in baseline['rows']:
        for pair in row['equipment_pairs']:
            pair['status']=pair.pop('co_motion_baseline_status');pair.pop('hand_constraint')
    return baseline


def changed_frame(prepared,frame,change):
    rows=list(prepared._rows);row=json.loads(rows[frame]);change(row);rows[frame]=encode(row)
    metadata=prepared.describe();metadata['cached_row_bytes']=sum(map(len,rows))
    return replace(prepared,_rows=tuple(rows),_metadata=encode(metadata))


def test_actual_selector_hand_opt_in_preserves_original_wrist_and_complete_windows(tmp_path,monkeypatch):
    pose,args,prepared,baseline,calls=fixture(); before=encode(baseline); count=len(calls)
    prepared=replace(prepared,pose_annotations=replace(prepared.pose_annotations,precursor_custody=None))
    monkeypatch.setattr(type(prepared.pose_annotations),'frame',lambda *_:pytest.fail('annotation requery'))
    monkeypatch.setattr(type(prepared),'frame',lambda *_:pytest.fail('payload query instead of bounded cache'))
    result=estimate_exercise_support(pose,**args,equipment_hand_points=prepared); extra=result['equipment_hand_constraint']
    assert extra['counts']['kept_candidates']==extra['counts']['baseline_candidates']>0
    assert extra['counts']['excluded_candidates']==extra['counts']['unavailable_candidates']==0
    assert extra['counts']['construction_hand_slots']==18 and extra['counts']['construction_point_slots']==72
    first=extra['hand_rows'][0]['hands'][0]
    assert first['point_counts']=={'intersection':3,'exclusion':1,'unavailable':0}
    assert first['points'][0]['consumer_status']=='exclusion' and first['status']=='intersection'
    assert [p['landmark_index'] for p in first['points']]==[15,17,19,21]
    assert encode(recover(result))==before==encode(baseline)
    assert result['coverage']==baseline['coverage'] and len(calls)==count
    assert extra['work']['raster_calls']==extra['work']['pose_annotation_queries']==0
    assert all(extra[key]=='unknown' for key in ('contact','physical_noncontact','grasp','load'))
    target=os.environ.get('WEFT_HAND_CONSTRAINT_FIXTURE_DIR')
    out=Path(target) if target else tmp_path/'analytical'
    if target:assert out.parts[0]=='.local' and '..' not in out.parts
    out.mkdir()
    saved={'support-evidence.json':encode(result),'baseline.json':before,'hand-metadata.json':prepared._metadata,
        'hand-frames.jsonl':b'\n'.join(prepared._rows)+b'\n','pose.json':prepared.pose_annotations.pose_document,
        'equipment.json':prepared.equipment_document,'mask-metadata.json':prepared.mask_document,
        'annotation.json':prepared.pose_annotations.annotation_document(),
        'annotation-metadata.json':prepared.pose_annotations._metadata}
    for name,raw in saved.items():(out/name).write_bytes(raw)
    (out/'manifest.json').write_bytes(encode({'scope':'Authored analytical restriction only; no source acquisition, decoder, native model or physical claim',
        'files':{name:sha256(raw).hexdigest() for name,raw in saved.items()}}))


def test_all_four_disjoint_excludes_only_original_candidates():
    def hole(pose):
        for row in pose['frames']:
            for index in (17,19,21,18,20,22):row['landmarks'][index].update(x=35/pose['width'],y=33/pose['height'])
    _,_,prepared,baseline,_=fixture(pose_change=hole)
    result=constrained(baseline,prepared);extra=result['equipment_hand_constraint']
    assert extra['counts']['excluded_candidates']==extra['counts']['baseline_candidates']>0
    for old,new in zip(baseline['rows'],result['rows'],strict=True):
        for a,b in zip(old['equipment_pairs'],new['equipment_pairs'],strict=True):
            assert b['status']==('image_hand_pointset_exclusion' if a['status']=='projected_co_motion_candidate' else a['status'])
    assert encode(recover(result))==encode(baseline)


def test_any_missing_finger_overrides_intersection_across_original_window():
    def low(pose):pose['frames'][4]['landmarks'][17]['presence']=.1
    _,_,prepared,baseline,_=fixture(pose_change=low,raster=b'\1'*9)
    result=constrained(baseline,prepared);extra=result['equipment_hand_constraint']
    hand=extra['hand_rows'][4]['hands'][0]
    assert hand['status']=='unavailable' and hand['point_counts']=={'intersection':3,'exclusion':0,'unavailable':1}
    assert extra['counts']['unavailable_candidates']>0
    for row in result['rows'][:9]:
        decision=row['equipment_pairs'][0]['hand_constraint']
        if decision['window_point_range'] is not None:
            left,right=decision['window_point_range']
            assert (decision['status']=='unavailable')==(left<=4<right)


def test_unavailable_selected_point_is_not_queried_again(monkeypatch):
    def low(pose):pose['frames'][4]['landmarks'][17]['presence']=.1
    _,_,prepared,baseline,_=fixture(pose_change=low,raster=b'\1'*9)
    api=importlib.import_module('weft.equipment_hand_constraint');original=api.source_point;queried=[]
    def guarded(row,index,width,height):
        assert (row['frame'],index)!=(8,17), 'Unavailable source landmark was queried'
        queried.append((row['frame'],index))
        return original(row,index,width,height)
    monkeypatch.setattr(api,'source_point',guarded)
    result=constrained(baseline,prepared)
    assert result['equipment_hand_constraint']['hand_rows'][4]['hands'][0]['status']=='unavailable'
    assert len(queried)==71 and (8,19) in queried and (8,15) in queried


def test_single_selected_hand_keeps_other_hand_explicitly_unavailable():
    _,_,prepared,baseline,_=fixture(hands=('left',))
    result=constrained(baseline,prepared);extra=result['equipment_hand_constraint']
    assert extra['counts']['unavailable_candidates']>0 and extra['counts']['kept_candidates']>0
    right=extra['hand_rows'][0]['hands'][1]
    assert right['reason']=='hand_group_not_supplied' and len(right['points'])==4
    assert all(p['producer_record_present'] is False and p['pixel'] is None and p['source_confidence'] is None for p in right['points'])
    assert [p['landmark_index'] for p in right['points']]==[16,18,20,22]


@pytest.mark.parametrize('kind',['pixel','confidence','name','clock','partition'])
def test_selected_point_custody_mismatch_cannot_keep_candidate(kind):
    _,_,prepared,baseline,_=fixture()
    def corrupt(row):
        point=row['hands'][0]['points'][1]
        if kind=='pixel':point['pixel'][0]+=.1
        elif kind=='confidence':point['source_confidence']=.7
        elif kind=='name':point['point_name']='right_pinky'
        elif kind=='clock':row['pose_decoder_timestamp_s']+=.014
        else:point['human_partition']='audit'
    changed=changed_frame(prepared,8,corrupt)
    result=constrained(baseline,changed)
    assert result['equipment_hand_constraint']['hand_rows'][4]['hands'][0]['status']=='unavailable'
    assert result['equipment_hand_constraint']['counts']['unavailable_candidates']>0
    assert encode(recover(result))==encode(baseline)


def test_group_intersection_cannot_hide_all_point_exclusion_or_missing_slots():
    _,_,prepared,baseline,_=fixture()
    changed=changed_frame(prepared,0,lambda row:row['hands'][0].update(point_counts={'intersection':4,'exclusion':0,'unavailable':0}))
    with pytest.raises(ValueError,match='complete four-point'):constrained(baseline,changed)
    changed=changed_frame(prepared,0,lambda row:row['hands'][0]['points'].pop())
    with pytest.raises(ValueError,match='four exact'):constrained(baseline,changed)


def test_excluded_payload_never_parsed_and_raw_pose_hash_still_binds_it():
    _,_,prepared,baseline,_=fixture()
    rows=list(prepared._rows);rows[20]=b'EXCLUDED: do not parse this payload'
    metadata=prepared.describe();metadata['cached_row_bytes']=sum(map(len,rows))
    opaque=replace(prepared,_rows=tuple(rows),_metadata=encode(metadata))
    assert constrained(baseline,opaque)['equipment_hand_constraint']['work']['selected_frame_queries']==9
    pose=json.loads(prepared.pose_annotations.pose_document);pose['frames'][-1]['landmarks'][17]['x']+=.01
    new_raw=encode(pose);pose_meta=json.loads(prepared.pose_annotations._metadata)
    pose_meta['provenance']['pose_raw_sha256']=sha256(new_raw).hexdigest()
    metadata=prepared.describe();metadata['pose_provenance']=pose_meta['provenance']
    changed=replace(prepared,pose_annotations=replace(prepared.pose_annotations,pose_document=new_raw,_metadata=encode(pose_meta)),_metadata=encode(metadata))
    with pytest.raises(ValueError,match='canonical pose'):constrained(baseline,changed)


@pytest.mark.parametrize('kind',['exact','missing','clock'])
def test_actual_derivative_proof_retains_original_and_equipment_clocks(tmp_path,kind):
    from test_exercise_support_evidence import _equivalent_equipment
    pose,_,proof=_equivalent_equipment(tmp_path)
    annotations,args,*_=hand_fixture(supplied_pose=pose,proof=proof,raster=b'\1'*9)
    prepared=importlib.import_module('weft.equipment_hand_evidence').prepare_equipment_hand_evidence(
        annotations,construction_frames=[0],**args)
    pose=json.loads(annotations.pose_document)
    baseline=estimate_exercise_support(pose,camera_prior=prior(pose['source_sha256']),equipment_observations=args['equipment_document'],
        equipment_ids=args['equipment_ids'],equipment_source_equivalence=proof)
    if kind=='missing':
        with pytest.raises(ValueError,match='equivalence'):constrained(baseline,replace(prepared,source_equivalence=None))
        return
    if kind=='clock':prepared=changed_frame(prepared,0,lambda row:row['hands'][0]['points'][1].update(equipment_time_s=0))
    extra=constrained(baseline,prepared)['equipment_hand_constraint']
    assert extra['input_bindings']['source_equivalence_manifest_sha256']==proof.manifest_sha256
    assert extra['hand_rows'][0]['source_mapping']==baseline['rows'][0]['equipment_source_mapping']
    hand=extra['hand_rows'][0]['hands'][0]
    assert hand['status']==('unavailable' if kind=='clock' else 'intersection')
    assert hand['points'][0]['equipment_time_s']!=extra['hand_rows'][0]['pose_decoder_timestamp_s']


def test_ambiguous_constraints_reject_before_source_preparation(monkeypatch):
    api=importlib.import_module('weft.exercise_support_evidence')
    monkeypatch.setattr(api,'_prepare',lambda *_:pytest.fail('source preparation entered'))
    with pytest.raises(ValueError,match='exactly one'):
        api.estimate_exercise_support({},camera_prior={},equipment_silhouette=object(),equipment_hand_points=object())
    with pytest.raises(ValueError,match='exact equipment'):
        api.estimate_exercise_support({},camera_prior={},equipment_hand_points=object())


def test_old_wrist_and_default_serialized_outputs_stay_exact(monkeypatch):
    pose,args,prepared,baseline,_=wrist_fixture()
    assert sha256(encode(baseline)).hexdigest()=='adabae52b761e440a7b3e9db84a6ea12e9ac4b7d170dfb489fb92bec64652417'
    assert sha256(encode(wrist_constrained(baseline,prepared))).hexdigest()=='ce3c8a7055cb8b23df8aff5e719493146cb9a146f30c60047872f03afb46ece9'
    original=builtins.__import__
    def imported(name,*values,**options):
        assert name not in ('weft.equipment_hand_constraint','weft.equipment_hand_evidence')
        return original(name,*values,**options)
    monkeypatch.setattr(builtins,'__import__',imported)
    assert encode(estimate_exercise_support(pose,**args,equipment_hand_points=None))==encode(baseline)


def test_deadline_and_cache_exhaustion_return_no_partial_output(monkeypatch):
    _,_,prepared,baseline,_=fixture();api=importlib.import_module('weft.equipment_hand_constraint');before=encode(baseline)
    with pytest.raises(TimeoutError):api.constrain_equipment_hand_support(baseline,prepared,deadline=time.monotonic()-1)
    monkeypatch.setattr(api,'MAX_CACHE_BYTES',1)
    with pytest.raises(ValueError,match='cache exceeds'):constrained(baseline,prepared)
    assert encode(baseline)==before


def test_selected_point_pass_is_linear_and_bounded(tmp_path):
    _,_,prepared,baseline,_=fixture(count=301)
    started=time.perf_counter();result=constrained(baseline,prepared);elapsed=time.perf_counter()-started
    extra=result['equipment_hand_constraint']
    assert extra['work']['selected_frame_queries']==90 and extra['work']['point_slots']==720
    assert extra['work']['prefix_counter_entries']==91*3*2
    assert extra['counts']['kept_candidates']==extra['counts']['baseline_candidates']>0 and elapsed<2
    (tmp_path/'performance.json').write_bytes(encode({'seconds':elapsed,'output_bytes':len(encode(result)),
        'selected_frames':90,'point_slots':720,'cache_bytes':extra['work']['serialized_hand_cache_bytes']+extra['work']['prefix_counter_buffer_bytes']}))


def test_complete_four_point_capacity_exceeds_wrist_cache_without_dropping_records(tmp_path):
    _,_,prepared,baseline,calls=fixture(count=259,dense=True)
    before=encode(baseline); raster_calls=len(calls)
    started=time.perf_counter();result=constrained(baseline,prepared);elapsed=time.perf_counter()-started
    api=importlib.import_module('weft.equipment_hand_constraint')
    wrist=importlib.import_module('weft.equipment_silhouette_constraint')
    extra=result['equipment_hand_constraint']; work=extra['work']
    cache_bytes=work['serialized_hand_cache_bytes']+work['prefix_counter_buffer_bytes']
    assert work['selected_frame_queries']==len(extra['hand_rows'])==154
    assert work['point_slots']==extra['counts']['construction_point_slots']==1232
    assert (2<<20)<cache_bytes<=extra['bounds']['additional_cache_bytes']==api.MAX_CACHE_BYTES==8<<20
    assert wrist.MAX_CACHE_BYTES==2<<20 and api.MAX_OUTPUT_BYTES==wrist.MAX_OUTPUT_BYTES==8<<20
    output_bytes=len(encode(result));assert output_bytes<=8<<20 and elapsed<2
    for row in extra['hand_rows']:
        original=json.loads(prepared._rows[row['source_frame']])
        for actual,expected in zip(row['hands'],original['hands'],strict=True):
            assert len(actual['points'])==4
            for point,raw in zip(actual['points'],expected['points'],strict=True):
                assert {k:v for k,v in point.items() if k not in
                        {'producer_record_present','consumer_status','consumer_reason'}}==raw
    assert extra['counts']['kept_candidates']==extra['counts']['baseline_candidates']>0
    assert encode(recover(result))==before==encode(baseline) and len(calls)==raster_calls
    assert work['raster_calls']==work['pose_annotation_queries']==0
    assert all(extra[key]=='unknown' for key in ('contact','physical_noncontact','grasp','load'))
    metrics={'scope':'Authored 154-frame/one-object cardinality regression, not the source pilot',
        'seconds':elapsed,'output_bytes':output_bytes,'selected_frames':154,'point_slots':1232,
        'cache_bytes':cache_bytes,'prefix_bytes':work['prefix_counter_buffer_bytes'],
        'original_point_records_exact':True,'baseline_recovered_exact':True}
    (tmp_path/'capacity.json').write_bytes(encode(metrics))
    target=os.environ.get('WEFT_HAND_CAPACITY_RECEIPT')
    if target:
        path=Path(target);assert path.parts[0]=='.local' and '..' not in path.parts
        with path.open('xb') as stream:stream.write(encode(metrics))
