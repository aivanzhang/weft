"""Prototype checks use grayscale analytic scenes, not physical evidence."""
import cv2
import numpy as np
import pytest

from weft.generic_flow_prototype import FlowValidatedAppearance


@pytest.mark.parametrize('change', [
    {'association_mode':'joint'}, {'initialization_frame':1},
    {'decode_interval':[0,9], 'end_frame':9},
])
def test_unsupported_tracking_contract_rejected_before_reading_source(tmp_path, change):
    from weft.generic_flow_prototype import track_prototype
    config={'schema_version':1, 'source_sha256':'a'*64, 'objects':[
        {'id':'one', 'label':'Object', 'box_xywh':[20,20,16,16],
         'role':'unknown', 'shape':'box', 'provenance':'Analytic configuration'}], **change}
    with pytest.raises(ValueError,match='does not support'):
        track_prototype(tmp_path/'missing.mp4', tmp_path/'out', config)
    assert not (tmp_path/'out').exists()


def settings():
    return {'search_radius_px':60,'minimum_score':.7,'minimum_margin':.06,'max_gap_frames':3}


def object_patch():
    patch=np.full((48,48),110,np.uint8)
    cv2.ellipse(patch,(24,24),(19,17),20,0,360,200,-1)
    cv2.line(patch,(15,11),(30,34),35,5)
    cv2.circle(patch,(32,15),5,250,-1)
    cv2.circle(patch,(17,29),4,80,-1)
    return patch


def frame(x=70,y=65,angle=0,blur=0,visible=True,duplicate=False):
    image=np.full((200,280),110,np.uint8)
    if visible:
        patch=object_patch()
        if angle:
            patch=cv2.warpAffine(patch,cv2.getRotationMatrix2D((24,24),angle,1),(48,48),borderValue=110)
        if blur:patch=cv2.GaussianBlur(patch,(0,0),blur)
        image[y:y+48,x:x+48]=patch
    if duplicate:image[y:y+48,x+80:x+128]=object_patch()
    return image


def test_rotated_blurred_compact_gray_object_requires_immutable_validation():
    tracker=FlowValidatedAppearance(frame(),[70,65,48,48],settings())
    reference=[entry[1].copy() for entry in tracker.bank]
    candidate=tracker.locate(frame(78,71,angle=60,blur=1.5))
    assert candidate['status']=='tracked'
    assert np.linalg.norm(np.array(candidate['box'][:2])-[78,71])<3
    assert candidate['score']>=.7 and candidate['margin']>=.06
    tracker.commit(candidate,frame(78,71,angle=60,blur=1.5))
    assert all(np.array_equal(before,after[1]) for before,after in zip(reference,tracker.bank))


def test_ambiguous_appearance_rejects_both_peak_locations():
    tracker=FlowValidatedAppearance(frame(),[70,65,48,48],dict(settings(),search_radius_px=100))
    result=tracker.locate(frame(duplicate=True))
    assert result['status']=='ambiguous'
    assert result['box'] is None


def test_no_position_published_during_occlusion_or_after_long_loss():
    tracker=FlowValidatedAppearance(frame(),[70,65,48,48],settings())
    for _ in range(4):
        candidate=tracker.locate(frame(visible=False))
        assert candidate['status']=='not_detected'
        assert candidate['box'] is None
        tracker.commit(candidate,frame(visible=False))
    candidate=tracker.locate(frame())
    assert candidate['status']=='lost' and candidate['box'] is None


def test_flow_proposal_alone_cannot_create_a_detection(monkeypatch):
    tracker=FlowValidatedAppearance(frame(),[70,65,48,48],settings())
    monkeypatch.setattr(tracker,'_flow_proposal',lambda gray:(np.array([102,95]),{'status':'proposal_only'}))
    result=tracker.locate(frame(visible=False))
    assert result['status']=='not_detected' and result['box'] is None


def test_no_texture_or_outside_initialization_rejected():
    with pytest.raises(ValueError,match='contrast'):
        FlowValidatedAppearance(frame(visible=False),[70,65,48,48],settings())
    with pytest.raises(ValueError,match='bounds'):
        FlowValidatedAppearance(frame(),[270,65,48,48],settings())


def test_video_camera_motion_stays_ineligible_for_physics(tmp_path):
    from weft.generic_flow_prototype import track_prototype
    from weft.io import digest
    source=tmp_path/'camera.mp4'
    rng=np.random.default_rng(19)
    background=cv2.GaussianBlur(rng.integers(40,190,(200,280),dtype=np.uint8),(5,5),0)
    background[65:113,70:118]=object_patch()
    writer=cv2.VideoWriter(str(source),cv2.VideoWriter_fourcc(*'mp4v'),10,(280,200))
    assert writer.isOpened()
    for index in range(8):
        moving=cv2.warpAffine(background,np.float32([[1,0,index*2.],[0,1,0]]),(280,200),borderMode=cv2.BORDER_REFLECT)
        writer.write(cv2.cvtColor(moving,cv2.COLOR_GRAY2BGR))
    writer.release()
    config={'schema_version':1,'source_sha256':digest(source),'objects':[
        {'id':'gray','label':'Authored grayscale object','box_xywh':[70,65,48,48],
         'role':'unknown','shape':'unknown','provenance':'Analytic software test, not real observation'}]}
    result=track_prototype(source,tmp_path/'result',config)
    assert result['decoded_frames']==8
    assert result['camera_check']['result']=='moving_background'
    assert result['physics_ready'] is False
    assert sum(row['status']=='tracked' for row in result['observations'])==7
