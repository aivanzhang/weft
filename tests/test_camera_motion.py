"""Analytic camera-screen checks; image alignment does not recover a calibrated camera."""
import cv2
import numpy as np

from weft.scene_vision import _camera_change_trigger, _camera_descriptor_sample, _camera_sample


def background(width=480,height=360):
    rng=np.random.default_rng(47)
    gray=cv2.resize(rng.integers(35,210,(height//5,width//5),dtype=np.uint8),(width,height),interpolation=cv2.INTER_CUBIC)
    gray=cv2.GaussianBlur(gray,(3,3),.6)
    for _ in range(35):
        x,y=rng.integers(20,width-20),rng.integers(20,height-20)
        cv2.circle(gray,(int(x),int(y)),int(rng.integers(3,12)),int(rng.integers(20,230)),2)
    return gray


def points(gray,mask=None):
    return cv2.goodFeaturesToTrack(gray,maxCorners=120,qualityLevel=.02,minDistance=12,mask=mask)


def sample(reference,current,mask=None):
    return _camera_sample(reference,cv2.cvtColor(current,cv2.COLOR_GRAY2BGR),points(reference,mask),1)


def test_fixed_textured_background_is_not_claimed_to_move():
    reference=background()
    result=sample(reference,reference.copy())
    assert result['status']=='no_large_motion_detected'
    assert result['center_shift_image_width']<.001
    assert result['source_grid_cells']>=4


def test_local_foreground_motion_does_not_become_camera_motion():
    reference=background();current=reference.copy()
    reference[90:150,70:140]=200
    current[90:150,110:180]=200
    mask=np.full(reference.shape,255,np.uint8);mask[75:165,55:155]=0
    result=sample(reference,current,mask)
    assert result['status']=='no_large_motion_detected'


def test_translation_and_rotation_remain_positive_motion_evidence():
    reference=background();height,width=reference.shape
    for matrix in [np.float32([[1,0,15],[0,1,-10]]),cv2.getRotationMatrix2D((width/2,height/2),5,1)]:
        current=cv2.warpAffine(reference,matrix,(width,height),borderMode=cv2.BORDER_REFLECT)
        result=sample(reference,current)
        assert result['status']=='moving_background'
        assert result['inliers']>=12 and result['coverage']>=.2


def test_direct_descriptor_fallback_handles_large_projective_displacement():
    reference=background();height,width=reference.shape
    source=np.float32([[0,0],[width-1,0],[width-1,height-1],[0,height-1]])
    target=np.float32([[80,35],[width-40,15],[width-10,height-50],[25,height-15]])
    current=cv2.warpPerspective(reference,cv2.getPerspectiveTransform(source,target),(width,height))
    result=_camera_descriptor_sample(reference,current,points(reference))
    assert result['status']=='moving_background'
    assert result['method']=='original_reference_sift_homography'
    assert result['inlier_ratio']>=.6
    assert result['median_symmetric_reprojection_error_px']<=1.5


def test_complete_feature_loss_is_unknown_not_camera_motion():
    reference=background()
    result=sample(reference,np.zeros_like(reference))
    assert result['status'].startswith('unknown_')
    assert 'center_shift_image_width' not in result


def test_localized_texture_does_not_establish_scene_motion():
    reference=np.full((360,480),120,np.uint8)
    reference[100:180,100:190]=background(90,80)
    current=cv2.warpAffine(reference,np.float32([[1,0,20],[0,1,5]]),(480,360),borderValue=120)
    result=sample(reference,current)
    assert result['status'].startswith('unknown_')


def test_repeated_texture_descriptor_ambiguity_is_not_positive_evidence():
    tile=np.kron(np.array([[40,190],[190,40]],np.uint8),np.ones((9,9),np.uint8))
    reference=np.tile(tile,(20,26))[:360,:468]
    current=np.roll(reference,9,axis=1)
    result=_camera_descriptor_sample(reference,current,points(reference))
    assert result['status'].startswith('unknown_')


def test_change_trigger_ignores_exposure_shift_and_small_foreground():
    reference=background()
    exposed=np.clip(reference.astype(int)+20,0,255).astype(np.uint8)
    assert not _camera_change_trigger(reference,exposed)
    foreground=reference.copy();foreground[100:170,100:180]=255
    assert not _camera_change_trigger(reference,foreground)
    assert not _camera_change_trigger(None,reference)


def test_change_trigger_requests_alignment_for_scene_wide_translation():
    reference=background()
    current=cv2.warpAffine(reference,np.float32([[1,0,35],[0,1,25]]),(480,360),borderMode=cv2.BORDER_REFLECT)
    assert _camera_change_trigger(reference,current)
