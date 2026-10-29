from copy import deepcopy

import pytest

from test_exercise_vision import inference_result
from test_multiview_pose import pair
from weft.exercise_vision import summarize_result
from weft.multiview_pose import triangulate_pair
from weft.pose_crop import apply_reinference, assess_reinference, map_crop_pose, plan_crop, plan_reinference


def sequence(count=13, bad=(6,)):
    first, second, _ = pair(roots=[[.2, .3, 5.]]*count)
    for view in (first, second):
        view['evidence']['pose_identity'] = {'mode': 'raw', 'repaired_frames': 0}
    clean = deepcopy([first, second])
    for frame in bad:
        for index in (23, 24): first['evidence']['frames'][frame]['landmarks'][index]['y'] += .12
    baseline = triangulate_pair(first, second)
    return [first, second], baseline, clean


def test_bad_rays_get_original_neighbor_crops_without_modifying_any_input():
    views, baseline, clean = sequence()
    before = deepcopy((views, baseline))
    plans = plan_reinference(views, baseline)
    assert len(plans) == 1 and plans[0]['frame'] == 6 and plans[0]['status'] == 'planned'
    assert plans[0]['root_anchors'] == [5, 7]
    assert all(crop['anchor_frames'] == [5, 7] for crop in plans[0]['crops'])
    proposal = triangulate_pair(*clean)
    decision = assess_reinference(plans[0], [view['evidence']['frames'][6] for view in clean], baseline, proposal)
    assert decision['accepted'] and decision['root_deviation_torso_ratio'] < 1e-12
    assert decision['estimated_body_joints'] == 12 and decision['anatomical_identity_verified'] is False
    assert (views, baseline) == before


def test_consistent_motion_is_not_selected_even_if_fast_or_nonuniform():
    first, second, _ = pair(roots=[[.2+.05*i*i, .3, 5.] for i in range(6)])
    for view in (first, second): view['evidence']['pose_identity'] = {'mode': 'raw'}
    assert plan_reinference([first, second], triangulate_pair(first, second)) == []


@pytest.mark.parametrize('kind', ['before', 'after', 'low_confidence', 'competing', 'long_gap', 'modified'])
def test_unavailable_temporal_context_is_not_repaired(kind):
    views, _, _ = sequence()
    evidence = views[0]['evidence']
    if kind in ('before', 'after', 'competing', 'modified'):
        for row in evidence['frames']:
            if row['frame'] == 6: continue
            if kind == 'before' and row['frame'] < 6: row['status'] = 'no_pose'
            elif kind == 'after' and row['frame'] > 6: row['status'] = 'no_pose'
            elif kind == 'competing': row['pose_count'] = 2
            elif kind == 'modified': row['image_refinement'] = {'method': 'prior_repair'}
    elif kind == 'low_confidence':
        for row in evidence['frames']:
            if row['frame'] != 6:
                for point in row['landmarks']: point['visibility'] = .1
    else:
        evidence['fps'] = 2.
        for row in evidence['frames']: row['time_s'] = row['frame']/2
    with pytest.raises(ValueError): plan_crop(evidence, 6)


@pytest.mark.parametrize('kind', ['changed_evidence', 'changed_camera', 'repaired_view', 'repaired_frame', 'identity_repair'])
def test_input_provenance_and_repeat_repair_reject(kind):
    views, baseline, _ = sequence()
    if kind == 'changed_evidence': views[0]['evidence']['frames'][5]['landmarks'][11]['x'] += .001
    elif kind == 'changed_camera': views[0]['camera']['radial_k1'] += .001
    elif kind == 'repaired_view': views[0]['evidence']['pose_reinference'] = {'method': 'previous'}
    elif kind == 'repaired_frame': views[0]['evidence']['frames'][6]['image_refinement'] = {'method': 'previous'}
    else: views[0]['evidence']['pose_identity']['mode'] = 'temporal'
    with pytest.raises(ValueError): plan_reinference(views, baseline)


def test_boundaries_abstain_and_excessive_failures_reject_without_dropping_targets():
    views, baseline, _ = sequence(bad=(0, 6, 12))
    plans = plan_reinference(views, baseline)
    assert [p['frame'] for p in plans] == [0, 6, 12]
    assert [p['status'] for p in plans] == ['unavailable_context', 'planned', 'unavailable_context']
    views, baseline, _ = sequence(count=40, bad=range(40))
    with pytest.raises(ValueError, match='Too many'): plan_reinference(views, baseline)


@pytest.mark.parametrize('kind,reason', [('competing', 'crop_pose_unusable_or_competing'),
    ('lost_joint', 'previous_image_support_lost'), ('few_joints', 'insufficient_supported_body_joints'),
    ('bad_hip', 'torso_or_hip_rays_unsupported'), ('root_jump', 'root_outside_original_temporal_support')])
def test_new_hypotheses_must_keep_geometry_and_temporal_support(kind, reason):
    views, baseline, clean = sequence()
    plan = plan_reinference(views, baseline)[0]
    proposal = triangulate_pair(*clean)
    rows = [view['evidence']['frames'][6] for view in clean]
    if kind == 'competing': rows[1]['pose_count'] = 2
    elif kind in ('lost_joint', 'few_joints', 'bad_hip'):
        indices = [13] if kind == 'lost_joint' else [23] if kind == 'bad_hip' else [13, 14, 15, 16, 25]
        for j in indices: proposal['frames'][6]['joints'][j]['status'] = 'image_evidence_unavailable'
    else:
        for point in proposal['frames'][6]['joints']: point['world_point_m'][0] += 1.
    result = assess_reinference(plan, rows, baseline, proposal)
    assert not result['accepted'] and reason in result['reasons']


def test_crop_coordinate_transform_preserves_metric_world_and_original_clock():
    views, _, _ = sequence()
    evidence = views[0]['evidence']; original = evidence['frames'][6]
    plan = {'frame': 6, 'box_xyxy': [100, 70, 500, 650]}
    raw = summarize_result(inference_result())
    for point in raw['landmarks']: point['z'] = .4
    before = deepcopy(raw)
    result = map_crop_pose(raw, plan, evidence, original, .01)
    assert result['world_landmarks'] == raw['world_landmarks']
    assert result['id'] == original['id'] and result['time_s'] == original['time_s']
    for old, new in zip(raw['landmarks'], result['landmarks'], strict=True):
        assert new['x'] == pytest.approx((100+400*old['x'])/evidence['width'])
        assert new['y'] == pytest.approx((70+580*old['y'])/evidence['height'])
        assert new['z'] == pytest.approx(.4*400/evidence['width'])
    assert raw == before


@pytest.mark.parametrize('count', [0, 2])
def test_no_or_multiple_crop_hypotheses_never_become_camera_landmarks(count):
    views, _, _ = sequence()
    evidence = views[0]['evidence']; original = evidence['frames'][6]
    result = map_crop_pose(summarize_result(inference_result(count=count)),
                          {'frame': 6, 'box_xyxy': [100, 70, 500, 650]}, evidence, original, .01)
    assert result['pose_count'] == count and result['landmarks'] == []
    assert 'poses' not in result


@pytest.mark.parametrize('change', ['invalid_box', 'wrong_frame', 'negative_time', 'full_image'])
def test_crop_input_bounds_reject(change):
    views, _, _ = sequence()
    evidence = views[0]['evidence']; original = evidence['frames'][6]
    if change == 'full_image':
        for row in evidence['frames']:
            row['landmarks'][0].update(x=0., y=0.)
            row['landmarks'][1].update(x=1., y=1.)
        with pytest.raises(ValueError, match='bounded image'): plan_crop(evidence, 6)
        return
    plan = {'frame': 6, 'box_xyxy': [100, 70, 500, 650]}
    if change == 'invalid_box': plan['box_xyxy'][0] = -1
    if change == 'wrong_frame': plan['frame'] = 7
    with pytest.raises(ValueError):
        map_crop_pose(summarize_result(inference_result()), plan, evidence, original, -.01 if change == 'negative_time' else .01)


def test_apply_preserves_originals_missing_slots_clocks_and_records_exact_derivation():
    views, baseline, clean = sequence()
    plans = plan_reinference(views, baseline)
    proposals = {6: clean[0]['evidence']['frames'][6]}
    decision = assess_reinference(plans[0], [v['evidence']['frames'][6] for v in clean], baseline, triangulate_pair(*clean))
    old = deepcopy(views[0]['evidence'])
    result = apply_reinference(old, plans, proposals, [decision], 0)
    assert len(result['frames']) == len(old['frames'])
    assert result['pose_reinference']['changed_frames'] == [6]
    assert result['frames'][6]['image_refinement_raw'] == old['frames'][6]
    assert result['frames'][6]['landmarks'] == proposals[6]['landmarks']
    assert all(row == old['frames'][i] for i, row in enumerate(result['frames']) if i != 6)
    assert old == views[0]['evidence']
    rejected = apply_reinference(old, plans, proposals, [dict(decision, accepted=False)], 0)
    assert rejected['frames'] == old['frames']
    with pytest.raises(ValueError): apply_reinference(result, plans, proposals, [decision], 0)
    with pytest.raises(ValueError): apply_reinference(old, plans, proposals, [decision, decision], 0)
    proposals[6] = dict(proposals[6], time_s=0.)
    with pytest.raises(ValueError): apply_reinference(old, plans, proposals, [decision], 0)


def test_spatial_reference_retains_crop_derivation_without_changing_geometry():
    from test_spatial_pose import pose_frame, anchor
    from weft.spatial_pose import build_spatial_reference
    evidence = {'frames': [pose_frame()]}
    expected, audit = build_spatial_reference(evidence, anchor())
    assert 'image_refinement' not in audit['frames'][0]
    evidence['frames'][0]['image_refinement'] = {'method': 'test', 'proposal_sha256': 'a'*64}
    reference, derived = build_spatial_reference(evidence, anchor())
    assert reference == expected and derived['frames'][0]['image_refinement'] == evidence['frames'][0]['image_refinement']
