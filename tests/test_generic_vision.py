"""Grayscale fixtures are analytic software tests, not physical transfer evidence."""
import json
from copy import deepcopy

import cv2
import numpy as np
import pytest

from weft.generic_vision import AppearanceTrack, scene_measurements, track_video, validate_config
from weft.io import digest


def settings():
    return {'search_radius_px': 45, 'minimum_score': .7, 'minimum_margin': .06, 'max_gap_frames': 4}


def pattern(seed=8):
    rng = np.random.default_rng(seed)
    return rng.integers(30, 230, (28, 24), dtype=np.uint8)


def frame(x=40, y=45, *, visible=True, duplicate=False):
    image = np.full((160, 240), 120, np.uint8)
    if visible:
        image[y:y+28,x:x+24] = pattern()
    if duplicate:
        image[y:y+28,x+50:x+74] = pattern()
    return image


def test_grayscale_translation_and_brightness_without_color_assumption():
    tracker = AppearanceTrack(frame(), [40,45,24,28], settings())
    for index in range(1, 16):
        image = np.clip(frame(40+index*3,45+index).astype(int)+12,0,255).astype(np.uint8)
        result = tracker.locate(image)
        assert result['status'] == 'tracked'
        assert result['box'] == [40+index*3,45+index,24,28]
        tracker.commit(result)


def test_occlusion_has_no_position_and_short_gap_can_recover():
    tracker = AppearanceTrack(frame(), [40,45,24,28], settings())
    for _ in range(3):
        result = tracker.locate(frame(visible=False))
        assert result['status'] == 'not_detected'
        assert result['box'] is None
        tracker.commit(result)
    result = tracker.locate(frame(48,46))
    assert result['box'] == [48,46,24,28]
    tracker.commit(result)
    assert tracker.gap == 0


def test_long_gap_stays_lost_instead_of_guessing_identity():
    tracker = AppearanceTrack(frame(), [40,45,24,28], settings())
    for _ in range(5):
        tracker.commit(tracker.locate(frame(visible=False)))
    result = tracker.locate(frame())
    assert result['status'] == 'lost'
    assert result['box'] is None


def test_identical_distant_regions_are_ambiguous():
    tracker = AppearanceTrack(frame(), [40,45,24,28], dict(settings(),search_radius_px=80))
    result = tracker.locate(frame(duplicate=True))
    assert result['status'] == 'ambiguous'
    assert result['box'] is None


def test_featureless_region_rejected():
    with pytest.raises(ValueError,match='contrast'):
        AppearanceTrack(np.full((160,240),128,np.uint8),[40,45,24,28],settings())


def test_color_mismatch_cannot_publish_grayscale_match_or_update_identity():
    gray = frame()
    def colored(hue):
        hsv = np.zeros((*gray.shape, 3), np.uint8)
        hsv[:, :, 2] = gray
        hsv[45:73, 40:64, 0] = hue
        hsv[45:73, 40:64, 1] = 200
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    tracker = AppearanceTrack(gray, [40,45,24,28], settings(), color=colored(30))
    signature = deepcopy(tracker.source_color)
    rejected = tracker.locate(gray, color=colored(110))
    assert rejected['score'] > .99
    assert rejected['status'] == 'appearance_changed'
    assert rejected['box'] is None
    tracker.commit(rejected)
    assert tracker.gap == 1
    assert tracker.box == (40,45,24,28)
    recovered = tracker.locate(gray, color=colored(30))
    assert recovered['status'] == 'tracked'
    tracker.commit(recovered)
    assert tracker.source_color == signature
    assert tracker.gap == 0


def write_gray_clip(path):
    rng = np.random.default_rng(91)
    background = rng.integers(50, 190, (160,240), dtype=np.uint8)
    background = cv2.GaussianBlur(background,(5,5),0)
    writer = cv2.VideoWriter(str(path),cv2.VideoWriter_fourcc(*'mp4v'),20,(240,160))
    assert writer.isOpened()
    for index in range(32):
        image = background.copy()
        image[45:73,40:64] = pattern()
        image = cv2.warpAffine(image,np.float32([[1,0,index*.6],[0,1,0]]),(240,160),borderMode=cv2.BORDER_REFLECT)
        writer.write(cv2.cvtColor(image,cv2.COLOR_GRAY2BGR))
    writer.release()


def config(video):
    return {'schema_version':1,'source_sha256':digest(video),'objects':[{'id':'gray-object','label':'Authored textured object','box_xywh':[40,45,24,28],'role':'unknown','shape':'box','provenance':'Analytic grayscale fixture; known source rectangle'}]}


def test_video_audit_camera_motion_and_scene_measurements(tmp_path):
    video=tmp_path/'fixture.mp4'
    write_gray_clip(video)
    result=track_video(video,tmp_path/'run',config(video))
    assert result['decoded_frames']==32
    assert result['camera_check']['result']=='moving_background'
    assert result['physics_ready'] is False
    assert result['observations'][0]['status']=='initialized'
    measurements=scene_measurements(result)
    assert len(measurements['observations'])==31
    assert all(r['frame']>0 for r in measurements['observations'])
    assert max(abs(r['x_px']-(52+.6*r['frame'])) for r in measurements['observations'])<1
    assert (tmp_path/'run'/'tracks.gif').stat().st_size>0
    assert json.loads((tmp_path/'run'/'appearance-tracks.json').read_text())['initialization']['status']=='authored'
    with pytest.raises(FileExistsError):
        track_video(video,tmp_path/'run',config(video))


def test_source_hash_and_scope_are_required(tmp_path):
    video=tmp_path/'fixture.mp4';write_gray_clip(video)
    bad=config(video);bad['source_sha256']='0'*64
    with pytest.raises(ValueError,match='hash'):
        track_video(video,tmp_path/'run',bad)
    assert not (tmp_path/'run').exists()
    bad=config(video);bad['secret']='not allowed'
    with pytest.raises(ValueError,match='Unknown'):
        validate_config(bad)
    bad=config(video);bad['objects'].append(deepcopy(bad['objects'][0]))
    with pytest.raises(ValueError,match='unique'):
        validate_config(bad)


def world_config(video):
    tracking=config(video)
    tracking['objects'][0]['role']='active'
    return {'tracking':tracking,'world':{'controlled_id':'gray-object','target':[.75,.4],'gravity':[0,0],'damping':1.,'restitution':.5}}


def test_scene_adapter_keeps_camera_guard(tmp_path):
    from weft.generic_vision import inspect_generic
    from weft.scene_vision import UnsupportedVideo
    video=tmp_path/'fixture.mp4';write_gray_clip(video)
    out=tmp_path/'run';out.mkdir()
    with pytest.raises(UnsupportedVideo,match='Moving camera'):
        inspect_generic(video,out,world_config(video))
    assert (out/'appearance'/'appearance-tracks.json').exists()
    assert not (out/'scene.json').exists()


@pytest.mark.parametrize("start", [0, 5])
def test_grayscale_scene_adapter_fits_and_runs_interactive_environment(tmp_path, start):
    from weft.generic_vision import inspect_generic
    from weft.scene_vision import fit_scene
    from weft.scene import SceneEnv
    video=tmp_path/'moving.mp4'
    writer=cv2.VideoWriter(str(video),cv2.VideoWriter_fourcc(*'mp4v'),20,(240,160))
    for index in range(40):
        writer.write(cv2.cvtColor(frame(40+index,45),cv2.COLOR_GRAY2BGR))
    writer.release()
    out=tmp_path/'run';out.mkdir()
    authored = world_config(video)
    authored['tracking']['start_frame'] = start
    authored['tracking']['objects'][0]['box_xywh'][0] += start
    evidence,scene=inspect_generic(video,out,authored)
    assert evidence['generation'][-1]['frame']<evidence['development_validation'][0]['frame']
    assert not any(row['frame']==0 for row in evidence['generation'])
    assert evidence['preview_frame_indices'][0]==start
    selected=fit_scene(evidence,scene,out)
    report=json.loads((out/'reconstruction.json').read_text())
    assert report['replay_gate']['passes']
    assert report['after']['rmse_px']<1
    assert (out/'reconstruction.gif').is_file()
    env=SceneEnv(selected)
    try:
        env.reset(seed=1,options={'jitter':False})
        before=env.snapshot()['objects'][0]['position']
        observation,reward,terminated,truncated,info=env.step(2)
        assert np.isfinite(observation).all()
        assert env.snapshot()['objects'][0]['position'][0]>before[0]
    finally:
        env.close()


def test_manual_evaluation_counts_misses_and_false_visibility():
    from weft.generic_vision import evaluate_manual
    tracks={'source_sha256':'a'*64,'width':100,'height':80,'source_width':200,'source_height':160,'seconds':.1,
            'camera_check':{'result':'not_verified_fixed'},'observations':[
                {'frame':1,'object_id':'one','status':'tracked','x_px':10,'y_px':10},
                {'frame':2,'object_id':'one','status':'lost'},
                {'frame':3,'object_id':'one','status':'tracked','x_px':10,'y_px':10}]}
    labels={'source_sha256':'a'*64,'coordinate_system':'original_source_pixels','annotations':[
        {'frame':1,'object_id':'one','visible':True,'center_xy':[23,24]},
        {'frame':2,'object_id':'one','visible':True,'center_xy':[23,24]},
        {'frame':3,'object_id':'one','visible':False,'center_xy':None}]}
    result=evaluate_manual(tracks,labels)
    assert result['accepted_rmse_source_px']==5
    assert result['visible_label_coverage']==.5
    assert result['false_visible_labels']==1
    tracks['observations'][0]['status']='lost'
    result=evaluate_manual(tracks,labels)
    assert result['accepted_rmse_source_px'] is None
    assert result['visible_label_coverage']==0


def write_long_source(path):
    writer=cv2.VideoWriter(str(path),cv2.VideoWriter_fourcc(*'mp4v'),2,(240,160))
    assert writer.isOpened()
    for index in range(130):
        writer.write(cv2.cvtColor(frame(40+index,45),cv2.COLOR_GRAY2BGR))
    writer.release()


def interval_config(video):
    authored=world_config(video)
    authored['tracking'].update(start_frame=80,end_frame=119,initialization_frame=84,decode_interval=[80,119])
    authored['tracking']['objects'][0]['box_xywh'][0]=124
    return authored


@pytest.mark.parametrize('association_mode', ['independent', 'joint'])
def test_long_source_interval_preserves_absolute_frames_and_preinitialization(tmp_path, association_mode):
    video=tmp_path/'long.mp4';write_long_source(video)
    with pytest.raises(ValueError,match='explicit decode_interval'):
        track_video(video,tmp_path/'whole',config(video))
    tracking=interval_config(video)['tracking']
    tracking['association_mode']=association_mode
    result=track_video(video,tmp_path/'interval',tracking)
    assert result['frame_count']==130 and result['decoded_frames']==40
    assert result['selected_interval']==[80,119]
    assert [row['frame'] for row in result['frames']]==list(range(80,120))
    assert result['frames'][0]['time_s']==40
    audit=result['decode_audit']
    assert audit['scope']=='selected_interval' and audit['selected_frames_complete']
    assert audit['source_complete'] is False and audit['prefix_advanced_frames']==80
    observations=result['observations']
    assert [row['status'] for row in observations[:5]]==['pre_initialization']*4+['initialized']
    assert all(row['box_xywh'] is None and 'x_px' not in row for row in observations[:4])
    assert observations[0]['id']==f'{digest(video)}:frame:80:object:gray-object'
    assert all(row['status']=='tracked' for row in observations[5:])
    assert result['camera_check']['reference_frame']==84
    assert all(row['frame']>=84 for row in result['camera_check']['samples'])
    assert result['association_mode']==association_mode
    if association_mode=='joint':
        assert len(result['initialization']['source_template_signatures']['gray-object'])==64
        assert result['association_contract']['maximum_assignment_nodes']==20000
        for row in observations[5:]:
            audit=row['association_evidence']
            assert audit['assignment_complete']
            assert row['peak_margin']==audit['peak_margin']
            accepted=[item for item in audit['candidate_hypotheses'] if item['status']=='accepted_visual_measurement']
            assert len(accepted)==1 and accepted[0]['box']==row['box_xywh']


@pytest.mark.parametrize('mode', [None, 'auto', True, [], {}])
def test_invalid_association_mode_rejected(tmp_path, mode):
    video=tmp_path/'gray.mp4';write_gray_clip(video)
    tracking=config(video)
    tracking['association_mode']=mode
    with pytest.raises(ValueError,match='Association mode'):
        validate_config(tracking)


def test_interval_scene_and_shared_observations_do_not_rebase_source(tmp_path):
    from weft.generic_vision import inspect_generic
    from weft.observations import adapt_tracks, validate_observations
    from weft.scene_vision import fit_scene
    video=tmp_path/'long.mp4';write_long_source(video)
    out=tmp_path/'scene';out.mkdir()
    evidence,scene=inspect_generic(video,out,interval_config(video))
    coverage=json.loads((out/'appearance-coverage.json').read_text())
    assert coverage['joint_tracking_coverage']==pytest.approx(35/39)
    assert evidence['tracking_initialization_frame']==84
    assert evidence['initialization_frame']==85
    assert evidence['origin_time_s']==42.5
    assert evidence['preview_frame_indices'][0]==80
    assert evidence['decoded_frames']==40 and evidence['frame_count']==130
    assert evidence['selected_frames']==list(range(85,120))
    assert evidence['reconstruction_coverage']['pre_model_frames']==5
    selected=fit_scene(evidence,scene,out)
    replay=json.loads((out/'reconstruction.json').read_text())
    assert replay['after']['rmse_px']<1
    assert [row['frame'] for row in replay['frames']]==list(range(80,120))
    assert replay['frames'][-1]['simulation_time_s']==17
    assert all(not row['predicted'] for row in replay['frames'][:5])
    observations=adapt_tracks(evidence,profile='planar',scene=selected,
                             tracks=json.loads((out/'tracked-objects.json').read_text()))
    validate_observations(observations)
    assert observations['schema_version']==2
    assert observations['source']['decoded_frames']==40
    assert observations['source']['decode_interval']==[80,119]
    assert observations['observations'][0]['frame']==80
    assert observations['observations'][0]['visibility']=='not_sampled'
    assert (out/'background.jpg').is_file()


@pytest.mark.parametrize('change', [
    {'decode_interval':None}, {'decode_interval':[0,1800]}, {'decode_interval':[20001,20002]},
    {'decode_interval':[3,5],'start_frame':4}, {'initialization_frame':-1},
    {'decode_interval':[3,5],'initialization_frame':6}, {'decode_interval':[True,5]},
])
def test_interval_config_rejects_unbounded_or_conflicting_selection(change):
    configured={'schema_version':1,'source_sha256':'a'*64,'objects':[]}
    configured.update(change)
    with pytest.raises(ValueError,match='interval|Initialization'):
        validate_config(configured)
