"""Synthetic fixed-reference controls; no real videos or background evaluation labels."""
from copy import deepcopy
import time

import cv2
import numpy as np
import pytest

from weft.io import canonical_hash,digest
from weft import camera_stabilization as camera


def texture(seed=13,width=320,height=240):
    rng=np.random.default_rng(seed)
    image=rng.integers(0,256,(height,width),dtype=np.uint8)
    image=cv2.GaussianBlur(image,(3,3),.6)
    for _ in range(80):
        x,y=rng.integers(12,width-12),rng.integers(12,height-12)
        cv2.circle(image,(int(x),int(y)),int(rng.integers(2,8)),int(rng.integers(0,256)),-1)
    return image


def pair(reference,current,*,boxes=(),reference_boxes=(),identity=False):
    height,width=reference.shape
    size,scale=camera._analysis_shape(width,height)
    prepared=camera._prepare(reference,camera._mask(size,scale,reference_boxes))
    return camera._pair(prepared,current,camera._mask(size,scale,boxes),scale,width,height,identity=identity)


def passed(attempts):
    return next((row for row in attempts if row['status']=='registered'),None)


def synthetic_registration():
    """A full analytically supported report for root adapter integration tests."""
    reference=texture()
    affine=cv2.getRotationMatrix2D((160,120),1.5,1.005)
    affine[:,2]+=[3,-2]
    current=cv2.warpAffine(reference,affine,(320,240))
    rows=[]
    for frame,image in enumerate((reference,current)):
        attempts=pair(reference,image,identity=frame==0)
        accepted=passed(attempts)
        assert accepted is not None
        rows.append({'frame':frame,'time_s':frame*.02,'status':'registered',
            'current_to_reference':accepted['candidate_current_to_reference'],
            'method':'reference' if frame==0 else accepted['method'],'attempts':attempts,'exclusion_boxes':[],'reason':None})
    return {'schema_version':1,'profile':'fixed-reference-registration-v1','source_sha256':'a'*64,'evidence_sha256':'b'*64,
        'width':320,'height':240,'timestamps_s':[0,.02],'reference_frame':0,
        'contract':deepcopy(camera.CONTRACT),'contract_sha256':canonical_hash(camera.CONTRACT),
        'reference_pool':camera._prepare(reference,camera._mask((320,240),np.ones(2),[]))['pool'].tolist(),
        'frames':rows,'ready':True,'status':'ready','reasons':[],
        'provenance':{'worker_complete':True,'decoded_frames':2,'source_hash_after':'a'*64,
                      'fixed_reference_no_composition':True,'manual_labels_used':False}}


def test_known_translation_rotation_scale_are_current_to_reference():
    reference=texture()
    for angle,scale,translation in ((0,1,[4,-3]),(2,1.01,[2,-1]),(-3,.98,[-2,1])):
        forward=cv2.getRotationMatrix2D((160,120),angle,scale)
        forward[:,2]+=translation
        current=cv2.warpAffine(reference,forward,(320,240))
        accepted=passed(pair(reference,current))
        assert accepted is not None
        matrix=np.asarray(accepted['candidate_current_to_reference'])
        expected=np.vstack([cv2.invertAffineTransform(forward),[0,0,1]])
        corners=np.array([[40,40,1],[280,40,1],[40,200,1],[280,200,1]])
        assert np.max(np.linalg.norm((corners@matrix.T-corners@expected.T)[:,:2],axis=1))<.6
        assert accepted['withheld_count']>=12 and accepted['withheld_source_spread']['quadrants']==4


def test_moving_foreground_exclusion_does_not_become_camera_motion():
    reference=texture();current=reference.copy()
    patch=texture(seed=99,width=80,height=80)
    reference[80:160,80:160]=patch
    current[90:170,130:210]=patch
    accepted=passed(pair(reference,current,boxes=[[109,69,231,191]],reference_boxes=[[59,59,181,181]]))
    assert accepted is not None
    matrix=np.asarray(accepted['candidate_current_to_reference'])
    assert np.max(np.abs(matrix-np.eye(3)))<.2
    for row in accepted['correspondences']:
        x,y=row['reference_xy'];assert not (59<=x<181 and 59<=y<181)
        x,y=row['current_xy'];assert not (109<=x<231 and 69<=y<191)


def test_parallax_and_large_change_are_rejected_not_hidden_by_fit_inliers():
    reference=texture()
    left=cv2.warpAffine(reference,np.float32([[1,0,4],[0,1,0]]),(320,240))
    right=cv2.warpAffine(reference,np.float32([[1,0,-4],[0,1,0]]),(320,240))
    current=np.hstack([left[:,:160],right[:,160:]])
    assert passed(pair(reference,current)) is None
    rotated=cv2.warpAffine(reference,cv2.getRotationMatrix2D((160,120),9,1),(320,240))
    assert passed(pair(reference,rotated)) is None


def test_blur_blank_and_localized_texture_lack_supported_global_registration():
    reference=texture()
    for current in (np.zeros_like(reference),cv2.GaussianBlur(reference,(0,0),5)):
        assert passed(pair(reference,current)) is None
    local=np.zeros_like(reference);local[60:150,100:200]=reference[60:150,100:200]
    assert passed(pair(local,local,identity=True)) is None


def test_withheld_points_are_never_used_to_fit_or_silently_discarded():
    points=np.array([[x,y] for y in range(20,230,20) for x in range(20,310,20)],dtype=float)
    ref,cur=camera._subset(points,points,320,240)
    # The split is fixed before fitting; only independent withheld observations disagree.
    cur=cur.copy();cur[~camera._fit_partition(ref,320,240)]+=[8,0]
    audit=camera._fit(ref,cur,320,240,'orb')
    assert audit['status']=='unavailable'
    assert audit['reason'] in ('withheld_residual_failed','insufficient_fit_inliers')
    assert audit['fit_inlier_count']>=20
    assert any(row['partition']=='withheld' and row['error_px']>6 for row in audit['correspondences'])


def test_collinear_fit_rank_is_rejected():
    points=np.array([[x,100] for x in range(20,300,3)],dtype=float)
    audit=camera._fit(points,points,320,240,'lk')
    assert audit['status']=='unavailable' and audit['reason']=='rank_deficient_fit'


def test_report_validator_and_exact_reference_identity():
    report=synthetic_registration()
    validated=camera.validate_registration(report)
    assert validated['frames'][0]['current_to_reference']==np.eye(3).tolist()
    validated['frames'][0]['status']='changed'
    assert report['frames'][0]['status']=='registered'


@pytest.mark.parametrize('mutate',[
    lambda r:r['frames'][0]['current_to_reference'][0].__setitem__(2,.1),
    lambda r:r['frames'][1]['current_to_reference'][0].__setitem__(0,1.3),
    lambda r:r['frames'][1].update(time_s=.03),
    lambda r:r['frames'][1]['attempts'][0].update(heldout_rmse_px=0),
    lambda r:r['provenance'].update(worker_complete=False),
    lambda r:r['frames'][1].update(status='unavailable'),
])
def test_validator_rejects_source_clock_transform_or_support_conflicts(mutate):
    report=synthetic_registration();mutate(report)
    with pytest.raises(ValueError):
        camera.validate_registration(report)


def evidence_for(video,count=3):
    return {'profile':'segmentation-evidence-v1','physics_ready':False,
        'source':{'submitted_video_sha256':digest(video),'width_px':320,'height_px':240,'frame_count':count},
        'time':{'timestamps_s':[i*.02 for i in range(count)]},
        'objects':[{'id':'sam-1','initialization':{'frame':0,'roi':[150,110,8,8]}}],
        'observations':[{'frame':i,'time_s':i*.02,'object_id':'sam-1','proposal_status':'returned',
            'geometry':{'box_xyxy_halfopen':[150,110,158,118]}} for i in range(count)]}


def write_video(path,images):
    writer=cv2.VideoWriter(str(path),cv2.VideoWriter_fourcc(*'MJPG'),50,(320,240))
    assert writer.isOpened()
    try:
        for image in images:
            writer.write(cv2.cvtColor(image,cv2.COLOR_GRAY2BGR))
    finally:
        writer.release()


def test_spawned_registration_source_lock_full_frames_clock_and_foreground_hash(tmp_path):
    reference=texture()
    images=[cv2.warpAffine(reference,np.float32([[1,0,i],[0,1,0]]),(320,240)) for i in range(3)]
    path=tmp_path/'synthetic.avi';write_video(path,images)
    evidence=evidence_for(path)
    report=camera.register_video(path,evidence,reference_frame=0)
    assert report['ready'] is True
    assert len(report['frames'])==3 and report['timestamps_s']==[0,.02,.04]
    assert report['evidence_sha256']==camera.registration_evidence_hash(evidence)
    changed=deepcopy(evidence);changed['observations'][1]['geometry']['box_xyxy_halfopen']=[151,110,159,118]
    assert camera.registration_evidence_hash(changed)!=report['evidence_sha256']
    assert report['provenance']['manual_labels_used'] is False
    path.write_bytes(b'changed')
    with pytest.raises(ValueError,match='hash'):
        camera.register_video(path,evidence,reference_frame=0)


def _hung_worker(pipe,*args):
    time.sleep(10)


def test_hard_deadline_fills_all_missing_rows_and_leaves_no_worker(tmp_path,monkeypatch):
    path=tmp_path/'synthetic.avi';path.write_bytes(b'fixture')
    evidence=evidence_for(path)
    before={child.pid for child in camera.multiprocessing.active_children()}
    monkeypatch.setattr(camera,'_worker',_hung_worker)
    monkeypatch.setitem(camera.CONTRACT,'deadline_seconds',.2)
    started=time.monotonic()
    report=camera.register_video(path,evidence,reference_frame=0)
    assert time.monotonic()-started<2.2
    assert report['ready'] is False and len(report['frames'])==3
    assert all(row['status']=='unavailable' and row['current_to_reference'] is None for row in report['frames'])
    assert 'hard_worker_deadline_exceeded' in report['reasons']
    assert {child.pid for child in camera.multiprocessing.active_children()}<=before


def test_complete_initialized_suffix_required_but_preinit_failure_retained(tmp_path):
    reference=texture();blank=np.zeros_like(reference)
    before=tmp_path/'preinit.avi';write_video(before,[blank,reference,reference])
    evidence=evidence_for(before);evidence['objects'][0]['initialization']['frame']=1
    report=camera.register_video(before,evidence,reference_frame=1)
    assert report['ready'] is True
    assert report['frames'][0]['status']=='unavailable'
    assert report['frames'][1]['current_to_reference']==np.eye(3).tolist()
    after=tmp_path/'suffix.avi';write_video(after,[reference,reference,blank])
    evidence=evidence_for(after);evidence['objects'][0]['initialization']['frame']=1
    report=camera.register_video(after,evidence,reference_frame=1)
    assert report['ready'] is False and report['frames'][2]['current_to_reference'] is None
    assert report['provenance']['worker_complete'] is True


def _oversized_worker(pipe,video,compact,reference_frame):
    row=camera._empty(0,compact,'x'*5000)
    pipe.send(('frame',row))
    time.sleep(10)


def test_parent_output_limit_terminates_worker_without_publishing_oversized_row(tmp_path,monkeypatch):
    path=tmp_path/'fixture.avi';path.write_bytes(b'fixture')
    evidence=evidence_for(path)
    monkeypatch.setattr(camera,'_worker',_oversized_worker)
    monkeypatch.setitem(camera.CONTRACT,'maximum_ipc_event_bytes',1000)
    report=camera.register_video(path,evidence,reference_frame=0)
    assert report['ready'] is False
    assert 'bounded_registration_output_exceeded' in report['reasons']
    assert all(row['attempts']==[] for row in report['frames'])


def test_maximum_retained_audit_shape_fits_declared_output_budget():
    import json
    report=synthetic_registration()
    row=deepcopy(report['frames'][1])
    assert len(row['attempts'][-1]['correspondences'])==camera.CONTRACT['maximum_correspondences']
    row['attempts']=[deepcopy(row['attempts'][-1]),deepcopy(row['attempts'][-1])]
    # This is an output-size envelope, not a claim of 500 measured frames.
    size=len(json.dumps([row for _ in range(500)],allow_nan=False).encode())
    assert size<camera.CONTRACT['maximum_report_bytes']-10000


def test_residual_and_spread_audits_cannot_be_changed_independently():
    report=synthetic_registration()
    report['frames'][1]['attempts'][-1]['correspondences'][0]['error_px']=99
    with pytest.raises(ValueError,match='residual audit'):
        camera.validate_registration(report)
    report=synthetic_registration()
    report['frames'][1]['attempts'][-1]['withheld_source_spread']['cells']=16
    # The actual value may already be 16, so force an impossible count.
    report['frames'][1]['attempts'][-1]['withheld_source_spread']['cells']=99
    with pytest.raises(ValueError,match='spatial support'):
        camera.validate_registration(report)


def test_lk_fallback_order_is_fixed_and_audits_orb_failure():
    reference=texture();current=cv2.warpAffine(reference,np.float32([[1,0,3],[0,1,-2]]),(320,240))
    size,scale=camera._analysis_shape(320,240)
    mask=camera._mask(size,scale,[])
    prepared=camera._prepare(reference,mask)
    prepared['descriptors']=None
    attempts=camera._pair(prepared,current,mask,scale,320,240)
    assert [row['method'] for row in attempts]==['orb','descriptor_seeded_lk']
    assert all(row['status']=='unavailable' for row in attempts)
    assert attempts[1]['reason']=='descriptor_training_seed_unavailable'
    assert attempts[1]['seed'] is None


def test_bounded_analysis_resize_returns_original_source_coordinate_transform():
    reference=texture(width=1280,height=720)
    current=cv2.warpAffine(reference,np.float32([[1,0,8],[0,1,-5]]),(1280,720))
    size,scale=camera._analysis_shape(1280,720)
    mask=camera._mask(size,scale,[])
    prepared=camera._prepare(cv2.resize(reference,size,interpolation=cv2.INTER_AREA),mask)
    attempts=camera._pair(prepared,cv2.resize(current,size,interpolation=cv2.INTER_AREA),mask,scale,1280,720)
    accepted=passed(attempts)
    assert accepted is not None and max(size)==960
    matrix=np.asarray(accepted['candidate_current_to_reference'])
    assert matrix[:2,2]==pytest.approx([-8,5],abs=.5)


def _cell_regression_points(cell_count,uneven):
    cells=([(0,0),(1,0),(2,0),(3,0),(0,1),(3,1),(0,3),(1,3),(3,3)] if cell_count==9 else
           [(x,y) for x in range(4) for y in range(4) if x in (0,3) or y in (0,3)])
    points=[]
    for index,(x,y) in enumerate(cells):
        count=2+index%9 if uneven else 12 if cell_count==9 else 9
        for position in range(count):
            points.append([x*80+12+(position%4)*14,y*60+10+(position//4)*14])
    return np.asarray(points,dtype=float),set(cells)


def _occupied(points):
    return set(map(tuple,np.floor(np.asarray(points)/[80,60]).astype(int)))


@pytest.mark.parametrize('cell_count',[9,12])
@pytest.mark.parametrize('uneven',[False,True])
def test_stratified_split_covers_each_populated_cell_for_fit_and_withheld(cell_count,uneven):
    points,cells=_cell_regression_points(cell_count,uneven)
    ref,cur=camera._subset(points,points+[2,-1],320,240)
    fit=camera._fit_partition(ref,320,240)
    assert _occupied(ref[fit])==cells
    assert _occupied(ref[~fit])==cells
    if not uneven:
        old_fit=np.arange(len(ref))%3!=0
        assert len(_occupied(ref[~old_fit]))==cell_count//3
        assert _occupied(ref[~old_fit])!=cells
    audit=camera._fit(points,points+[2,-1],320,240,'orb')
    assert audit['status']=='registered',audit['reason']
    assert audit['fit_spread']['cells']==cell_count
    assert audit['withheld_reference_spread']['cells']==cell_count
    assert audit['withheld_source_spread']['cells']==cell_count
    assert audit['heldout_max_px']<1e-6


@pytest.mark.parametrize('cell_count',[9,12])
def test_retained_point_partition_is_coordinate_stable_under_input_permutations(cell_count):
    points,_=_cell_regression_points(cell_count,True)
    baseline=camera._fit_partition(points,320,240)
    expected={tuple(point):bool(value) for point,value in zip(points,baseline)}
    for seed in (0,1,2,3):
        permutation=np.random.default_rng(seed).permutation(len(points))
        shuffled=points[permutation]
        actual=camera._fit_partition(shuffled,320,240)
        assert {tuple(point):bool(value) for point,value in zip(shuffled,actual)}==expected
        audit=camera._fit(shuffled,shuffled+[2,-1],320,240,'lk')
        assert audit['status']=='registered'
        assert {tuple(row['reference_xy']):row['partition']=='fit' for row in audit['correspondences']}==expected


def test_singleton_cells_cannot_manufacture_independent_support():
    points=np.array([[x*80+30,y*60+30] for x in range(4) for y in range(4)],dtype=float)
    fit=camera._fit_partition(points,320,240)
    assert len(fit)==16 and any(fit) and any(~fit)
    assert _occupied(points[fit]).isdisjoint(_occupied(points[~fit]))
    audit=camera._fit(points,points,320,240,'orb')
    assert audit['status']=='unavailable' and audit['reason']=='insufficient_correspondences'


def test_validator_recomputes_stratified_partition_and_rejects_v1_contract():
    report=synthetic_registration()
    row=report['frames'][1]['attempts'][-1]
    for index,point in enumerate(row['correspondences']):
        old_fit=index%3!=0
        point['partition']='fit' if old_fit else 'withheld'
        point['fit_inlier']=True if old_fit else None
    with pytest.raises(ValueError,match='split changed'):
        camera.validate_registration(report)
    report=synthetic_registration()
    report['contract']['method']='fixed-reference-background-similarity-v1'
    report['contract'].pop('partition_rule')
    report['contract_sha256']=canonical_hash(report['contract'])
    with pytest.raises(ValueError,match='Unsupported'):
        camera.validate_registration(report)


def _force_seeded_branch(monkeypatch,*,seed_offset=None):
    original=camera._fit
    def fitted(*args,**kwargs):
        audit=original(*args,**kwargs)
        if args[4]=='orb' and audit['candidate_current_to_reference'] is not None:
            # Exercise the fallback without claiming this synthetic image defeated ORB.
            audit.update(status='unavailable',reason='synthetic_forced_first_method_rejection')
            if seed_offset is not None:
                audit['candidate_current_to_reference'][0][2]+=seed_offset
        return audit
    monkeypatch.setattr(camera,'_fit',fitted)


@pytest.mark.parametrize('angle,scale,shift',[(0,1,[16,0]),(4,1.02,[3,2]),(-4,.97,[1,1])])
def test_seeded_flow_known_motion_and_training_only_seed(monkeypatch,angle,scale,shift):
    reference=texture()
    forward=cv2.getRotationMatrix2D((160,120),angle,scale);forward[:,2]+=shift
    current=cv2.warpAffine(reference,forward,(320,240))
    _force_seeded_branch(monkeypatch)
    attempts=pair(reference,current);accepted=passed(attempts)
    assert accepted is not None,[(a['reason'],a['heldout_max_px']) for a in attempts]
    assert accepted['method']=='descriptor_seeded_lk'
    expected=np.vstack([cv2.invertAffineTransform(forward),[0,0,1]])
    corners=np.array([[40,40,1],[280,40,1],[40,200,1],[280,200,1]])
    actual=np.asarray(accepted['candidate_current_to_reference'])
    assert max(np.linalg.norm((corners@actual.T-corners@expected.T)[:,:2],axis=1))<.4
    trained=set(accepted['seed']['training_reference_feature_ids'])
    withheld={p['reference_feature_id'] for p in accepted['correspondences'] if p['partition']=='withheld'}
    assert trained and withheld and trained.isdisjoint(withheld)
    assert trained=={p['reference_feature_id'] for p in attempts[0]['correspondences'] if p['partition']=='fit'}


def test_partition_precedes_matching_and_never_rebalances_survivors():
    pool,_=_cell_regression_points(12,False)
    membership=camera._fit_partition(pool,320,240)
    # Drop alternating training matches: surviving IDs retain original memberships.
    ids=np.array([i for i in range(len(pool)) if not membership[i] or i%2])
    audit=camera._fit(pool[ids],pool[ids]+[2,-1],320,240,'orb',pool=pool,feature_ids=ids)
    assert audit['correspondences']
    assert all(p['partition']==('fit' if membership[p['reference_feature_id']] else 'withheld') for p in audit['correspondences'])
    survivors=np.array([p['reference_xy'] for p in audit['correspondences']])
    assert not np.array_equal(camera._fit_partition(survivors,320,240),
        [p['partition']=='fit' for p in audit['correspondences']])


def test_ransac_never_receives_withheld_pool_points(monkeypatch):
    pool,_=_cell_regression_points(12,False)
    membership=camera._fit_partition(pool,320,240)
    current=pool+[2,-1];current[~membership]+=[8,0]
    original=camera.cv2.estimateAffinePartial2D;calls=[]
    def captured(source,target,**kwargs):
        calls.append(target.copy())
        return original(source,target,**kwargs)
    monkeypatch.setattr(camera.cv2,'estimateAffinePartial2D',captured)
    audit=camera._fit(pool,current,320,240,'orb',pool=pool,feature_ids=np.arange(len(pool)))
    assert audit['status']=='unavailable' and calls
    held={tuple(p) for p in pool[~membership]}
    assert all(not (set(map(tuple,points))&held) for points in calls)


def test_seeded_flow_uneven_density_keeps_spread_and_bounded_histograms(monkeypatch):
    reference=texture();reference[:120,:160]=cv2.GaussianBlur(reference[:120,:160],(5,5),1)
    current=cv2.warpAffine(reference,np.float32([[1,0,6],[0,1,-3]]),(320,240))
    _force_seeded_branch(monkeypatch)
    attempts=pair(reference,current);accepted=passed(attempts)
    assert accepted is not None and accepted['method']=='descriptor_seeded_lk'
    for audit in attempts:
        for name,histogram in audit['stages'].items():
            assert len(histogram['cells'])==16
            assert sum(histogram['cells'])==histogram['count']<=1000
        assert audit['stages']['matched_reference']['count']==audit['matched_count']
        assert audit['stages']['retained_reference']['count']==len(audit['correspondences'])<=120


def test_repeated_texture_cannot_create_descriptor_training_seed():
    tile=np.zeros((16,16),np.uint8);tile[3:9,3:9]=255;tile[10:13,11:14]=160
    reference=np.tile(tile,(15,20));current=np.roll(reference,8,axis=1)
    attempts=pair(reference,current)
    assert passed(attempts) is None
    assert attempts[-1]['reason']=='descriptor_training_seed_unavailable'
    assert attempts[-1]['seed'] is None


def test_wrong_seed_is_rejected_or_corrected_not_published_as_prediction(monkeypatch):
    reference=texture();current=cv2.warpAffine(reference,np.float32([[1,0,6],[0,1,-3]]),(320,240))
    _force_seeded_branch(monkeypatch,seed_offset=14)
    attempts=pair(reference,current);accepted=passed(attempts)
    if accepted is not None:
        assert np.asarray(accepted['candidate_current_to_reference'])[:2,2]==pytest.approx([-6,3],abs=.3)
        assert accepted['heldout_max_px']<=6
    else:
        assert all(a['status']=='unavailable' for a in attempts)
    assert attempts[-1]['seed']['current_to_reference'][0][2]>7


def test_seeded_fallback_foreground_masks_and_parallax_rejection(monkeypatch):
    _force_seeded_branch(monkeypatch)
    test_moving_foreground_exclusion_does_not_become_camera_motion()
    test_parallax_and_large_change_are_rejected_not_hidden_by_fit_inliers()
    test_blur_blank_and_localized_texture_lack_supported_global_registration()


def test_seeded_report_validates_and_tampered_seed_or_pool_is_rejected(monkeypatch):
    _force_seeded_branch(monkeypatch)
    report=synthetic_registration()
    assert camera.validate_registration(report)['ready']
    broken=deepcopy(report)
    audit=broken['frames'][1]['attempts'][-1]
    held=next(p['reference_feature_id'] for p in audit['correspondences'] if p['partition']=='withheld')
    audit['seed']['training_reference_feature_ids'].append(held)
    with pytest.raises(ValueError,match='training fit'):
        camera.validate_registration(broken)
    broken=deepcopy(report);broken['reference_pool'][0][0]+=.2
    with pytest.raises(ValueError,match='pool|histogram|split'):
        camera.validate_registration(broken)
    broken=deepcopy(report);broken['frames'][1]['attempts'][-1]['stages']['matched_reference']['count']+=1
    with pytest.raises(ValueError,match='histogram'):
        camera.validate_registration(broken)


def test_v2_contract_is_not_reinterpreted_as_v3():
    report=synthetic_registration()
    report['contract']['method']='fixed-reference-background-similarity-v2'
    report['contract_sha256']=canonical_hash(report['contract'])
    with pytest.raises(ValueError,match='Unsupported'):
        camera.validate_registration(report)


def test_seeded_flow_resized_coordinates_and_full_output_envelope(monkeypatch):
    import json
    _force_seeded_branch(monkeypatch)
    test_bounded_analysis_resize_returns_original_source_coordinate_transform()
    report=synthetic_registration()
    assert all(row['attempts'][-1]['method']=='descriptor_seeded_lk' for row in report['frames'])
    row=report['frames'][1]
    assert len(json.dumps(row).encode())<camera.CONTRACT['maximum_ipc_event_bytes']
    # Pool is stored once; this bounds 500 two-attempt rows without a 500-frame worker.
    report['frames']=[deepcopy(row) for _ in range(500)]
    assert len(json.dumps(report).encode())<camera.CONTRACT['maximum_report_bytes']-10000


def test_raw_detection_and_deduplication_histograms_explain_retention():
    points,_=_cell_regression_points(12,False)
    points=np.vstack([points,points[:3]])
    audit=camera._fit(points,points+[2,-1],320,240,'orb')
    assert audit['matched_count']==111
    assert audit['stages']['deduplicated_reference']['count']==108
    reference=texture()
    prepared=camera._prepare(reference,camera._mask((320,240),np.ones(2),[]))
    assert len(prepared['pool'])<=len(prepared['detected_points'])<=1000
    audit=pair(reference,reference,identity=True)[0]
    assert audit['stages']['reference_detected']['count']==len(prepared['detected_points'])
    assert audit['stages']['reference_pool']['count']==len(prepared['pool'])


@pytest.mark.parametrize('pool',[[1,2,3,4],[[True,2]],[['1',2]],[[float('inf'),2]]])
def test_reference_pool_rejects_coerced_or_malformed_coordinates(pool):
    report=synthetic_registration();report['reference_pool']=pool
    with pytest.raises(ValueError):
        camera.validate_registration(report)


def patch_pool(image,points,mask=None):
    side=2*camera.CONTRACT['patch_radius_analysis_px']+1
    if mask is None:
        mask=np.full_like(image,255)
    return {'patches':[cv2.getRectSubPix(image,(side,side),tuple(map(float,p))) for p in points],
            'patch_support':[bool(np.min(cv2.getRectSubPix(mask,(side,side),tuple(map(float,p))))==255) for p in points]}


def test_patch_gate_accepts_uniform_illumination_change_without_relocating_points():
    reference=texture();current=(reference.astype(float)*.8+20).astype(np.uint8)
    points=np.array([[40,40],[100,80],[220,160],[280,200]],dtype=float)
    unchanged=points.copy()
    keep,metrics,rejected=camera._appearance_filter(patch_pool(reference,points),current,np.full_like(current,255),
                                                  list(range(len(points))),points,np.ones(2))
    assert keep.tolist()==[0,1,2,3] and sum(rejected.values())==0
    assert min(value[0] for value in metrics.values())>.99
    assert np.array_equal(points,unchanged)
    assert passed(pair(reference,current)) is not None


def test_patch_gate_rejects_disappearing_textured_foreground_and_shadow_boundary():
    background=texture();reference=background.copy();current=background.copy()
    reference[80:160,100:180]=texture(seed=99,width=80,height=80)
    # A crossing shadow edge changes local structure, unlike a uniform brightness offset.
    current[30:70,50:70]=(current[30:70,50:70]*.1).astype(np.uint8)
    points=np.array([[130,110],[150,140],[50,50],[240,180]],dtype=float)
    keep,metrics,rejected=camera._appearance_filter(patch_pool(reference,points),current,np.full_like(current,255),
                                                  list(range(len(points))),points,np.ones(2))
    assert keep.tolist()==[3]
    assert rejected['correlation']==3 and metrics[3][0]>.99


def test_patch_uniqueness_rejects_local_repetitions_even_at_perfect_correspondence():
    tile=np.random.default_rng(51).integers(0,256,(5,5),dtype=np.uint8)
    reference=np.tile(tile,(48,64));points=np.array([[80,80],[160,120]],dtype=float)
    keep,metrics,rejected=camera._appearance_filter(patch_pool(reference,points),reference,np.full_like(reference,255),
                                                  [0,1],points,np.ones(2))
    assert keep.size==0 and not metrics
    assert rejected['ambiguous']==2


def test_wrong_flow_and_untextured_or_masked_patches_are_not_validated_by_seed():
    reference=texture();points=np.array([[80,80],[160,120]],dtype=float)
    prepared=patch_pool(reference,points)
    current_points=points+[6,0]
    keep,_,rejected=camera._appearance_filter(prepared,reference,np.full_like(reference,255),[0,1],current_points,np.ones(2),np.eye(2))
    assert keep.size==0 and rejected['correlation']==2
    mask=np.full_like(reference,255);mask[70:72,70:72]=0
    keep,_,rejected=camera._appearance_filter(prepared,reference,mask,[0,1],points,np.ones(2))
    assert keep.tolist()==[1] and rejected['support']==1
    flat=np.full_like(reference,100)
    keep,_,rejected=camera._appearance_filter(patch_pool(flat,points),flat,np.full_like(flat,255),[0,1],points,np.ones(2))
    assert keep.size==0 and rejected['texture']==2


def test_true_parallax_visual_matches_survive_appearance_then_geometry_rejects():
    reference=texture()
    left=cv2.warpAffine(reference,np.float32([[1,0,4],[0,1,0]]),(320,240))
    right=cv2.warpAffine(reference,np.float32([[1,0,-4],[0,1,0]]),(320,240))
    current=np.hstack([left[:,:160],right[:,160:]])
    points=np.array([[x,y] for y in range(30,220,25) for x in range(30,300,25) if abs(x-160)>30],dtype=float)
    moved=points.copy();moved[:,0]+=np.where(points[:,0]<160,4,-4)
    keep,_,rejected=camera._appearance_filter(patch_pool(reference,points),current,np.full_like(current,255),
                                            list(range(len(points))),moved,np.ones(2))
    assert len(keep)==len(points) and not sum(rejected.values())
    audit=camera._fit(points[keep],moved[keep],320,240,'orb',pool=points,feature_ids=keep)
    assert audit['status']=='unavailable'
    assert audit['heldout_max_px']>6


@pytest.mark.parametrize('angle,scale',[(4.8,1.045),(-4.8,.96),(0,1.045)])
def test_patch_gated_near_limit_known_rotation_scale(angle,scale):
    reference=texture();forward=cv2.getRotationMatrix2D((160,120),angle,scale)
    current=cv2.warpAffine(reference,forward,(320,240))
    accepted=passed(pair(reference,current))
    assert accepted is not None
    expected=np.vstack([cv2.invertAffineTransform(forward),[0,0,1]])
    corners=np.array([[40,40,1],[280,40,1],[40,200,1],[280,200,1]])
    actual=np.asarray(accepted['candidate_current_to_reference'])
    assert max(np.linalg.norm((corners@actual.T-corners@expected.T)[:,:2],axis=1))<.65


def test_patch_metrics_counts_and_old_v3_contract_cannot_be_forged():
    report=synthetic_registration()
    assert camera.validate_registration(report)['ready']
    broken=deepcopy(report);broken['frames'][1]['attempts'][0]['correspondences'][0]['appearance'][0]=.79
    with pytest.raises(ValueError):
        camera.validate_registration(broken)
    broken=deepcopy(report);broken['frames'][1]['attempts'][0]['appearance_rejections']['support']+=1
    with pytest.raises(ValueError,match='Appearance filter counts'):
        camera.validate_registration(broken)
    broken=deepcopy(report);broken['contract']['method']='fixed-reference-background-similarity-v3'
    broken['contract_sha256']=canonical_hash(broken['contract'])
    with pytest.raises(ValueError,match='Unsupported'):
        camera.validate_registration(broken)


def test_appearance_rejections_happen_before_ransac_and_preserve_pool_partition(monkeypatch):
    reference=texture();current=reference.copy()
    current[120:,160:]=texture(seed=47,width=160,height=120)
    prepared=camera._prepare(reference,camera._mask((320,240),np.ones(2),[]))
    pool=prepared['pool'];membership=camera._fit_partition(pool,320,240)
    keep,_,_=camera._appearance_filter(prepared,current,np.full_like(current,255),np.arange(len(pool)),pool,np.ones(2))
    assert len(keep)<len(pool)
    original=camera.cv2.estimateAffinePartial2D;calls=[]
    def capture(source,target,**kwargs):
        calls.append(target.copy())
        return original(source,target,**kwargs)
    monkeypatch.setattr(camera.cv2,'estimateAffinePartial2D',capture)
    audit=camera._fit(pool[keep],pool[keep],320,240,'orb',pool=pool,feature_ids=keep)
    allowed={tuple(pool[i]) for i in keep if membership[i]}
    assert calls and all(set(map(tuple,p)).issubset(allowed) for p in calls)
    assert all(p['partition']==('fit' if membership[p['reference_feature_id']] else 'withheld') for p in audit['correspondences'])
