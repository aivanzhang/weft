"""Analytic controls for authored planar candidates, not physical-video scoring."""
from copy import deepcopy
import math

import pytest

from weft.io import canonical_hash, write_json
from weft.benchmark import seal_suite
from weft.pipeline import verify_artifacts
from weft.segmentation_scene import UnsupportedSegmentationScene, evaluate_candidate, export_candidate, generate_candidate


def evidence(times=None, *, start=0, count=2, gravity=0.):
    times=times or [0,.016,.033,.05,.067,.083,.1,.117]
    objects=[{'id':f'sam-{i+1}','raw_model_id':str(i+1)} for i in range(count)]
    rows=[]
    for frame,stamp in enumerate(times):
        t=stamp-times[start]
        for i,obj in enumerate(objects):
            x=40+20*t if i==0 else 120+(i-1)*12
            y=30+100*gravity*t*t if i==0 else 65
            box=[x-2,y-2,x+2,y+2]
            rows.append({'id':f"{'a'*64}:frame:{frame}:model-object:{i+1}",'frame':frame,'time_s':stamp,
                'object_id':obj['id'],'raw_model_id':obj['raw_model_id'],
                'phase':'pre_initialization' if frame<start else 'initialization' if frame==start else 'post_initialization',
                'proposal_status':'missing' if frame<start else 'returned',
                'geometry':None if frame<start else {'box_xyxy_halfopen':box,'center_px':[(box[0]+box[2])/2,(box[1]+box[3])/2],
                    'box_xywh':[box[0],box[1],4,4],'coordinate_system':'submitted_image_pixels','status':'inferred'},
                'identity':{'physical_identity_verified':False}})
    return {'schema_version':1,'profile':'segmentation-evidence-v1','physics_ready':False,
        'source':{'submitted_video_sha256':'a'*64,'width_px':200,'height_px':100,'frame_count':len(times),'original_source_sha256':None},
        'time':{'timestamps_s':times},'objects':objects,'observations':rows,'identity_groups':[],'provenance':{},'limits':[]}


def assumptions(count=2, *, end=4, gravity=0.):
    return {'schema_version':1,'camera_model':'fixed_orthographic_assumed','known_camera_motion':'unknown',
        'plane':'common_2d_plane_assumed','units':'image_widths_not_meters','boundaries':'closed_image_edges_assumed',
        'shape_friction':.5,'fit_motion':'unforced_prefix_assumed','generation_end_frame':end,
        'dt':1/60,'gravity':[0,gravity],'damping':1.,'restitution':.9,
        'objects':[{'id':f'sam-{i+1}','role':'active' if i==0 else 'static','shape':'circle',
                    'mass':1.,'radius_px':2.,'color':'#808080'} for i in range(count)],
        'controlled_id':'sam-1','target_px':[160,40],
        'task':{'position_tolerance':.03,'speed_tolerance':.1,'max_steps':100,'acceleration':1.}}


def test_exact_time_executable_scene_and_no_physics_promotion():
    data,config=evidence(),assumptions()
    candidate=generate_candidate(data,config)
    assert candidate['status']=='executable_hypothesis' and candidate['physics_ready'] is False
    assert candidate['scene']['objects'][0]['velocity']==pytest.approx([.1,0],abs=1e-12)
    assert candidate['scene']['objects'][1]['velocity']==[0,0]
    replay=evaluate_candidate(candidate,data)
    assert replay['status']=='completed_diagnostic_replay' and replay['physics_ready'] is False
    assert [row['source_time_s'] for row in replay['snapshots']]==data['time']['timestamps_s']
    assert replay['partitions']['development_frames']['fixed_raw_id_hypothesis']['conditional_rmse_px']<1e-10
    assert replay['initialization_frame_unscored']==0
    assert replay['partitions']['generation_frames']['observed_returned_slots']==8
    assert data['physics_ready'] is False and config['dt']==1/60


def test_static_body_with_nonzero_gravity_is_not_fitted_as_falling():
    data=evidence(gravity=1.)
    candidate=generate_candidate(data,assumptions(gravity=1.))
    assert candidate['generation_fit'][1]['maximum_residual_px']==0
    assert candidate['scene']['objects'][1]['velocity']==[0,0]
    assert candidate['scene']['objects'][0]['velocity']==pytest.approx([.1,0],abs=1e-10)
    replay=evaluate_candidate(candidate,data)
    assert replay['status']=='completed_diagnostic_replay'
    assert replay['partitions']['development_frames']['anonymous_location_sets']['conditional_max_error_px']<.1


def test_development_geometry_and_future_groups_do_not_change_scene_or_generation_hypotheses():
    data=evidence()
    baseline=generate_candidate(data,assumptions())
    changed=deepcopy(data)
    for row in changed['observations']:
        if row['frame']>4:
            row['geometry']['box_xyxy_halfopen']=[10,10,14,14]
            row['geometry']['center_px']=[12,12]
    changed['identity_groups']=[{'id':'future','object_ids':['sam-1','sam-2'],'onset_frame':5}]
    revised=generate_candidate(changed,assumptions())
    assert canonical_hash(revised['scene'])==canonical_hash(baseline['scene'])
    assert revised['generation_evidence_sha256']==baseline['generation_evidence_sha256']
    assert revised['identity_hypotheses']==baseline['identity_hypotheses']
    assert revised['input_evidence_sha256']!=baseline['input_evidence_sha256']


def test_anonymous_score_is_separate_when_development_raw_ids_swap():
    data=evidence()
    for frame in range(5,8):
        left,right=data['observations'][frame*2:frame*2+2]
        left['geometry'],right['geometry']=right['geometry'],left['geometry']
    candidate=generate_candidate(data,assumptions())
    replay=evaluate_candidate(candidate,data)
    scores=replay['partitions']['development_frames']
    assert scores['fixed_raw_id_hypothesis']['conditional_rmse_px']>80
    assert scores['anonymous_location_sets']['conditional_rmse_px']<1e-10
    assert candidate['identity_hypotheses']['selected_alternative'] is None
    assert data['observations'][-1]['object_id']=='sam-2'


def test_missing_development_slots_and_preinit_coverage_are_explicit():
    data=evidence(times=[0,.017,.033,.05,.067,.083,.1,.117,.133,.15],start=2)
    for row in data['observations']:
        if row['frame']==9:
            row.update(proposal_status='missing',geometry=None)
    candidate=generate_candidate(data,assumptions(end=6))
    replay=evaluate_candidate(candidate,data)
    assert candidate['coverage']['pre_initialization_slots']==4
    assert candidate['coverage']['pre_initialization_returned']==0
    assert candidate['coverage']['discarded_postinit_frames']==0
    score=replay['partitions']['development_frames']
    assert score['missing_observed_slots']==2 and score['unmatched_simulated_slots']==2
    assert score['fixed_raw_id_hypothesis']['count']==4
    assert replay['snapshots'][0]['frame']==2


def test_identity_alternatives_are_bounded_same_world_hypotheses_not_selected():
    data=evidence(count=3)
    data['identity_groups']=[{'id':'g','object_ids':['sam-1','sam-2','sam-3'],'onset_frame':2}]
    candidate=generate_candidate(data,assumptions(count=3))
    assert len(candidate['identity_hypotheses']['generation_only_alternatives'])==6
    assert candidate['identity_hypotheses']['selected_alternative'] is None
    data=evidence(count=4)
    data['identity_groups']=[{'id':'g','object_ids':['sam-1','sam-2','sam-3','sam-4'],'onset_frame':2}]
    with pytest.raises(UnsupportedSegmentationScene,match='six-correspondence'):
        generate_candidate(data,assumptions(count=4))


@pytest.mark.parametrize('change,match',[
    (lambda c:c.update(known_camera_motion='moving'),'moving camera'),
    (lambda c:c.update(units='meters'),'Unsupported camera'),
    (lambda c:c.update(boundaries='open'),'Unsupported camera'),
    (lambda c:c.update(shape_friction=.4),'friction'),
    (lambda c:c.update(generation_end_frame=7),'development suffix'),
    (lambda c:c['objects'][0].update(radius_px=50),'geometry repair'),
    (lambda c:c['objects'][0].pop('mass'),'must be authored'),
])
def test_unsupported_authored_conditions_are_not_silently_repaired(change,match):
    config=assumptions()
    change(config)
    with pytest.raises(UnsupportedSegmentationScene,match=match):
        generate_candidate(evidence(),config)


def test_missing_prefix_not_searched_around_and_no_position_invention():
    data=evidence()
    data['observations'][4].update(proposal_status='missing',geometry=None)
    with pytest.raises(UnsupportedSegmentationScene,match='no interpolation'):
        generate_candidate(data,assumptions())


def test_initial_overlap_is_rejected_without_shrinking_authored_geometry():
    data=evidence()
    for row in data['observations']:
        if row['object_id']=='sam-2':
            row['geometry']['box_xyxy_halfopen']=[40,28,44,32]
            row['geometry']['center_px']=[42,30]
    config=assumptions()
    with pytest.raises(UnsupportedSegmentationScene,match='overlap'):
        generate_candidate(data,config)
    assert config['objects'][0]['radius_px']==2


@pytest.mark.parametrize('change',[
    lambda c:c['partitions'].update(development_frames=[6,7]),
    lambda c:c['scene'].update(restitution=.2),
    lambda c:c['scene']['objects'][0].update(velocity=[1,0]),
])
def test_candidate_mutation_cannot_change_the_evaluated_world_or_partition(change):
    data=evidence(); candidate=generate_candidate(data,assumptions())
    change(candidate)
    with pytest.raises(UnsupportedSegmentationScene,match='deterministic'):
        evaluate_candidate(candidate,data)


def test_speed_clamp_failure_preserves_partial_state_and_missing_score_denominator(monkeypatch):
    from weft import segmentation_scene as module
    original=module.SceneEnv.advance_physics
    def fail_after_first(self,action=0):
        result=original(self,action)
        if self.steps>=2:
            result['speed_limited']=True
        return result
    monkeypatch.setattr(module.SceneEnv,'advance_physics',fail_after_first)
    data=evidence(); candidate=generate_candidate(data,assumptions())
    replay=evaluate_candidate(candidate,data)
    assert replay['status']=='failed_replay' and len(replay['snapshots'])==2
    assert replay['failures'][0]['frame']==2
    score=replay['partitions']['development_frames']
    assert score['observed_returned_slots']==6 and score['simulation_unavailable_frames']==3
    assert score['anonymous_location_sets']['conditional_rmse_px'] is None
    assert score['fixed_raw_id_hypothesis']['count']==0


def test_export_seals_success_and_explicit_rejection_without_api(tmp_path):
    source=tmp_path/'source';source.mkdir()
    write_json(source/'segmentation-evidence.json',evidence())
    write_json(source/'report.json',{'status':'synthetic'})
    seal_suite(source)
    result=export_candidate(source,assumptions(),tmp_path/'candidate')
    assert result['status']=='completed_diagnostic_replay' and result['new_api_calls']==0
    verify_artifacts(tmp_path/'candidate')
    rejected=assumptions();rejected['known_camera_motion']='moving'
    result=export_candidate(source,rejected,tmp_path/'rejected')
    assert result['status']=='unsupported_retained' and result['candidate_created'] is False
    assert not (tmp_path/'rejected/scene.json').exists()
    verify_artifacts(tmp_path/'rejected')


def test_shared_decoder_time_interchange_generates_same_scene():
    from weft.segmentation_observations import adapt_segmentation
    data=evidence()
    shared=adapt_segmentation(data)
    assert generate_candidate(shared,assumptions())['scene']==generate_candidate(data,assumptions())['scene']


def test_independent_group_cartesian_product_is_capped_overall():
    data=evidence(count=6)
    data['identity_groups']=[{'id':f'g{i}','object_ids':[f'sam-{i}',f'sam-{i+1}'],'onset_frame':1} for i in (1,3,5)]
    with pytest.raises(UnsupportedSegmentationScene,match='six-correspondence'):
        generate_candidate(data,assumptions(count=6))


def test_exact_time_replay_preserves_authored_collision_not_free_pass_through():
    times=[i*.02 for i in range(13)]
    data=evidence(times)
    for row in data['observations']:
        t=row['time_s']; first=row['object_id']=='sam-1'
        x=(60+100*t if t<=.16 else 76-100*(t-.16)) if first else (100-100*t if t<=.16 else 84+100*(t-.16))
        row['geometry'].update(box_xyxy_halfopen=[x-4,36,x+4,44],center_px=[x,40],box_xywh=[x-4,36,8,8])
    config=assumptions();config['objects'][1]['role']='passive';config['restitution']=1.
    for obj in config['objects']:
        obj['radius_px']=4.
    candidate=generate_candidate(data,config)
    replay=evaluate_candidate(candidate,data)
    assert replay['status']=='completed_diagnostic_replay'
    assert any(['sam-1','sam-2'] in row['simulation_contacts'] for row in replay['snapshots'])
    assert replay['snapshots'][-1]['positions_px']['sam-1'][0]<76
    assert replay['snapshots'][-1]['positions_px']['sam-2'][0]>84
    assert replay['partitions']['development_frames']['fixed_raw_id_hypothesis']['conditional_max_error_px']<1


def test_initial_speed_and_gaps_are_rejected_without_search_or_clamp():
    data=evidence()
    for row in data['observations']:
        if row['object_id']=='sam-1':
            x=20+1000*row['time_s']
            row['geometry'].update(box_xyxy_halfopen=[x-2,28,x+2,32],center_px=[x,30])
    with pytest.raises(UnsupportedSegmentationScene,match='speed'):
        generate_candidate(data,assumptions())
    data=evidence([0,.02,.04,.06,.08,.4,.42,.44])
    with pytest.raises(UnsupportedSegmentationScene,match='gaps'):
        generate_candidate(data,assumptions())


def test_nonstring_raw_id_is_rejected_as_unsupported():
    data=evidence();data['objects'][0]['raw_model_id']={'bad':'value'}
    with pytest.raises(UnsupportedSegmentationScene,match='Raw model'):
        generate_candidate(data,assumptions())


def test_export_cannot_add_outputs_inside_immutable_input(tmp_path):
    source=tmp_path/'source';source.mkdir()
    write_json(source/'segmentation-evidence.json',evidence())
    write_json(source/'report.json',{'status':'synthetic'})
    seal_suite(source)
    before=verify_artifacts(source)
    with pytest.raises(UnsupportedSegmentationScene,match='immutable evidence input'):
        export_candidate(source,assumptions(),source/'new-candidate')
    assert not (source/'new-candidate').exists()
    assert verify_artifacts(source)==before
