from copy import deepcopy
from hashlib import sha256
import json
import pytest
from weft.person_region_proposal import propose_person_region


def documents():
    points=[{'x':.4+(i%3)*.1,'y':.3+(i%4)*.1,'visibility':.9,'presence':.9} for i in range(33)]
    rows=[{'frame':i,'id':f"{'a'*64}:frame:{i}",'time_s':i/30,'status':'tracked','pose_count':1,'landmarks':deepcopy(points)} for i in range(12)]
    model={'maximum_pose_hypotheses':2,'network_inference':False,'name':'analytical fixture, no inference',
           'running_mode':'VIDEO','temporal_tracking':True}
    evidence={'profile':'exercise-pose-v1','schema_version':1,'source_sha256':'a'*64,'width':640,'height':480,'fps':30,
              'frame_count':12,'decoded_frames':12,'inference_stride':1,'frames':rows,'model':model,
              'pose_identity':{'mode':'raw','repaired_frames':0}}
    return evidence,{'frames':deepcopy(rows),'model':deepcopy(model)}


def encoded(e,r):
    raw=json.dumps(r,allow_nan=False).encode();e=deepcopy(e)
    e['pose_identity']['raw_artifact_sha256']=sha256(raw).hexdigest()
    return json.dumps(e,allow_nan=False).encode(),raw


def competing(e,r):
    row=e['frames'][5];other=deepcopy(row);other['landmarks'][0]['x']=.9
    row.update(pose_count=2,poses=[deepcopy(row),other],landmarks=[],status='multiple_people',identity_status='unresolved_multiple_pose_hypotheses')
    r['frames']=deepcopy(e['frames'])


def test_union_retains_both_competing_hypotheses_and_pins_exact_inputs():
    e,r=documents();competing(e,r);pair=encoded(e,r);before=deepcopy((e,r))
    report=propose_person_region([pair])
    assert report['region_xyxy'][2]>576
    assert report['histories'][0]['hypotheses']==13
    assert len(report['histories'][0]['samples'])==12
    assert len(report['histories'][0]['samples'][5]['hypotheses'])==2
    assert report['histories'][0]['raw_pose_sha256']==sha256(pair[1]).hexdigest()
    assert report['histories'][0]['evidence_sha256']==sha256(pair[0]).hexdigest()
    assert not report['simulation_ready'] and not report['inference_executed']
    assert (e,r)==before


@pytest.mark.parametrize('mutation',['missing','reorder','raw_changed','capacity','retimed','region','selected','no_pose','nondict','count','native_view'])
def test_changed_incomplete_selected_or_repaired_views_never_define_region(mutation):
    e,r=documents()
    if mutation=='missing':e['frames'].pop();r['frames'].pop()
    elif mutation=='reorder':e['frames'].reverse();r['frames'].reverse()
    elif mutation=='raw_changed':r['frames'][0]['landmarks'][0]['x']=.1
    elif mutation=='capacity':e['model']['maximum_pose_hypotheses']=r['model']['maximum_pose_hypotheses']=1
    elif mutation=='retimed':e['frames'][1]['time_s']+=.001;r['frames']=deepcopy(e['frames'])
    elif mutation=='region':e['person_region']={'ignored':False}
    elif mutation=='selected':
        competing(e,r);e['frames'][5]['poses'].pop();r['frames']=deepcopy(e['frames'])
    elif mutation=='no_pose':e['frames'][0]['pose_count']=0;r['frames']=deepcopy(e['frames'])
    elif mutation=='nondict':e['frames'].append(None);r['frames'].append(None)
    elif mutation=='count':e['decoded_frames']-=1
    elif mutation=='native_view':e['source_clock']={'view':'changed'}
    with pytest.raises(ValueError):propose_person_region([encoded(e,r)])


def test_exact_raw_byte_hash_is_required_and_duplicate_json_is_rejected():
    pair=list(encoded(*documents()));pair[1]+=b'\n'
    with pytest.raises(ValueError):propose_person_region([pair])
    with pytest.raises(ValueError):propose_person_region([(b'{"x":1,"x":2}',b'{}')])


def test_low_confidence_point_is_not_used_but_original_hash_changes():
    e,r=documents();e['frames'][0]['landmarks'][0].update(x=100.,visibility=.49);r['frames']=deepcopy(e['frames'])
    first=propose_person_region([encoded(e,r)])
    e['frames'][0]['landmarks'][0]['x']=-100.;r['frames']=deepcopy(e['frames'])
    second=propose_person_region([encoded(e,r)])
    assert first['region_xyxy']==second['region_xyxy']
    assert first['histories'][0]['raw_pose_sha256']!=second['histories'][0]['raw_pose_sha256']
    assert 0 not in first['histories'][0]['samples'][0]['hypotheses'][0]['credible_landmark_indices']


def test_clipping_preserves_every_credible_in_image_point_and_reports_outside_points():
    e,r=documents();e['frames'][0]['landmarks'][0]['x']=-.1;e['frames'][1]['landmarks'][0]['x']=1.1;r['frames']=deepcopy(e['frames'])
    report=propose_person_region([encoded(e,r)])
    assert report['region_xyxy'][0]==0 and report['region_xyxy'][2]==640
    assert report['source_outside_frame_points']==2


def test_incompatible_histories_and_aggregate_budget_rejected_before_parsing():
    a,b=documents(),documents();b[0]['fps']=29
    with pytest.raises(ValueError):propose_person_region([encoded(*a),encoded(*b)])
    with pytest.raises(ValueError,match='aggregate'):propose_person_region([(b'x'*(8<<20),b'y'*(8<<20)+b'1')])


def test_observed_decoder_times_cannot_disagree_and_mode_provenance_stays_explicit():
    a,b=documents(),documents()
    b[0]['frames'][0]['decoder_timestamp_s']=.003;b[1]['frames']=deepcopy(b[0]['frames'])
    with pytest.raises(ValueError,match='decoder'):propose_person_region([encoded(*a),encoded(*b)])
    b[0]['frames'][0]['decoder_timestamp_s']=-1;b[1]['frames']=deepcopy(b[0]['frames'])
    with pytest.raises(ValueError,match='decoder'):propose_person_region([encoded(*b)])
    a[0]['model'].update(running_mode='IMAGE',temporal_tracking=False);a[1]['model']=deepcopy(a[0]['model'])
    result=propose_person_region([encoded(*a)])
    assert not result['native_clock_verified'] and result['histories'][0]['model']['running_mode']=='IMAGE'
    a[0]['model']['temporal_tracking']=True;a[1]['model']=deepcopy(a[0]['model'])
    with pytest.raises(ValueError):propose_person_region([encoded(*a)])


@pytest.mark.parametrize('mutation',['identity_type','boolean_frame','pixel_overflow'])
def test_malformed_identity_frame_and_numeric_overflow_fail_closed(mutation):
    e,r=documents()
    if mutation=='boolean_frame':
        e['frames'][0].update(frame=False,id=f"{'a'*64}:frame:False");r['frames']=deepcopy(e['frames'])
    elif mutation=='pixel_overflow':
        e['frames'][0]['landmarks'][0]['x']=1e308;r['frames']=deepcopy(e['frames'])
    pair=encoded(e,r)
    if mutation=='identity_type':
        value=json.loads(pair[0]);value['pose_identity']=None;pair=(json.dumps(value).encode(),pair[1])
    with pytest.raises(ValueError):propose_person_region([pair])
