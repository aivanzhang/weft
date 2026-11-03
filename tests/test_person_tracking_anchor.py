"""Partial source hints never impute excluded points or qualify a complete silhouette."""
from copy import deepcopy
import json
import time

import pytest

from weft.person_tracking_anchor import derive_visible_chain_anchor, PROFILE
from weft import exercise_perception_pipeline as bridge
from weft.io import digest, write_json
from test_exercise_perception_pipeline import _anchor_fixture, local_preflight_source, _fixture_analysis
from test_exercise_vision import inference_result


def row():
    points=inference_result().pose_landmarks[0]
    return {'frame':0,'id':'source:observation:0:person','evidence_ids':['source:frame:0'],
            'partition':'construction','visibility':'partial','measurements':{'landmarks':{
            'coordinate_system':'image','unit':'pixels','status':'inferred','value':[
                {'index':i,'position':[p.x*640,p.y*480],'visibility':p.visibility,'presence':p.presence} for i,p in enumerate(points)]}}}


def derive(value):return derive_visible_chain_anchor(value,width=640,height=480)


def test_occluded_arm_remains_unknown_but_other_connected_evidence_proposes_roi():
    value=row();points=value['measurements']['landmarks']['value'];points[14]['visibility']=.262;points[16]['visibility']=.485
    before=deepcopy(value);report=derive(value)
    assert report['status']=='proposed' and report['supported_chains']=={'arm':['left'],'leg':['left','right']}
    assert report['excluded_indices']==[14,16] and report['missing_core_indices']==[]
    assert report['coverage_status']=='unknown_full_person_extent' and not report['physical_identity_verified']
    assert [slot['raw'] for slot in report['landmark_slots']]==points and value==before
    assert all(not report['landmark_slots'][i]['used_for_roi'] for i in (14,16))
    points[14]['position']=[-50000.,50000.];points[16]['position']=[50000.,-50000.]
    moved=derive(value)
    assert moved['roi_xywh']==report['roi_xywh'] and moved['padding_pixels']==report['padding_pixels']
    assert moved['landmark_payload_sha256']!=report['landmark_payload_sha256']


@pytest.mark.parametrize('indices,reason',[
    ([13,14],'arm_chain'),([25,26],'leg_chain'),([0],'head_or_torso'),([11],'head_or_torso'),([24],'head_or_torso')])
def test_core_and_connected_support_are_not_replaced_by_other_confident_points(indices,reason):
    value=row()
    for i in indices:value['measurements']['landmarks']['value'][i]['visibility']=.49
    report=derive(value)
    assert report['status']=='abstained' and report['roi_xywh'] is None
    assert any(reason in r for r in report['reasons']) and len(report['landmark_slots'])==33


def test_confidence_boundary_and_unavailable_value_preserve_raw_semantics():
    value=row();points=value['measurements']['landmarks']['value'];points[13]['visibility']=.5
    assert derive(value)['supported_chains']['arm']==['left','right']
    points[13]['visibility']=None
    report=derive(value)
    assert report['supported_chains']['arm']==['right']
    assert report['landmark_slots'][13]['raw']['visibility'] is None
    assert report['landmark_slots'][13]['exclusion_reasons']==['confidence_unavailable']


@pytest.mark.parametrize('mutation',[
    lambda r:r.update(frame=1),lambda r:r.update(partition='audit'),lambda r:r.update(evidence_ids=[]),
    lambda r:r['measurements']['landmarks']['value'].pop(),
    lambda r:r['measurements']['landmarks']['value'][1].update(index=0),
    lambda r:r['measurements']['landmarks']['value'][1].update(visibility=float('nan')),
    lambda r:r['measurements']['landmarks']['value'][1].update(position=[float('inf'),0]),
])
def test_malformed_or_withheld_inputs_never_create_a_hint(mutation):
    value=row();mutation(value)
    with pytest.raises(ValueError):derive(value)


def test_outside_core_and_degenerate_torso_abstain_without_clamping_points():
    value=row();value['measurements']['landmarks']['value'][23]['position']=[-1.,100.]
    report=derive(value)
    assert report['status']=='abstained' and report['landmark_slots'][23]['raw']['position'][0]==-1
    value=row();points=value['measurements']['landmarks']['value']
    for i in (11,12,23,24):points[i]['position']=[100.,100.]
    assert 'degenerate_source_torso' in derive(value)['reasons']


def test_v1_stays_default_while_explicit_v2_uses_same_exact_source_pin(tmp_path):
    proposal={'source_sha256':'a'*64,'width':160,'height':100,'boxes':[[1,0,5,5,10,10]]}
    def hidden(d):
        for i in (14,16):d['observations'][0]['measurements']['landmarks']['value'][i]['visibility']=.2
    seed,scene=_anchor_fixture(tmp_path,proposal,hidden)
    with pytest.raises(ValueError,match='confident head'):bridge.body_tracking_anchor(seed,scene,proposal)
    candidate=bridge.body_tracking_anchor(seed,scene,proposal,anchor_profile=PROFILE)
    assert candidate['profile']==PROFILE and candidate['prompt'][0]==2 and candidate['excluded_indices']==[14,16]
    assert candidate['body_seed_manifest_sha256']==digest(seed/'manifest.json')


def test_v2_preflight_seals_helper_and_reuses_identical_roi(local_preflight_source,tmp_path,monkeypatch):
    import weft.shared_world_lab as worlds
    import weft.exercise_body_source as bodies
    import weft.meta_segmentation as meta
    video,body=local_preflight_source;p=tmp_path/'preflight'
    result=bridge.prepare_exercise_perception_preflight(video,p,body_source=body,anchor_profile=PROFILE)
    assert bridge.validate_perception_preflight(video,p,deadline=time.monotonic()+10)==result
    manifest=json.loads((p/'manifest.json').read_bytes())
    assert manifest['files']['implementation/person_tracking_anchor.py']==digest(p/'implementation/person_tracking_anchor.py')
    def forbidden(*a,**k):pytest.fail('No new inference, seed reconstruction or provider access')
    monkeypatch.setattr(worlds,'prepare_shared_world_seed',forbidden);monkeypatch.setattr(bodies,'prepare_exercise_body_source',forbidden)
    monkeypatch.setattr(meta,'resolve_key',forbidden)
    analysis=_fixture_analysis(video,tmp_path/'analysis')
    output=tmp_path/'bridge';bridge.prepare_exercise_perception(video,analysis,output,preflight=p)
    anchor=json.loads((output/'body-anchor.json').read_bytes())
    assert anchor=={**result['body_anchor'],'prompt':[2,*result['body_anchor']['prompt'][1:]]}
    helper=p/'implementation/person_tracking_anchor.py';helper.write_bytes(helper.read_bytes()+b'\n#changed')
    manifest['files']['implementation/person_tracking_anchor.py']=digest(helper);write_json(p/'manifest.json',manifest)
    with pytest.raises(ValueError,match='anchor implementation'):
        bridge.validate_perception_preflight(video,p,deadline=time.monotonic()+10)
