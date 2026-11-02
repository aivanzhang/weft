from copy import deepcopy
import math

import numpy as np
import pytest

from weft.constrained_stereo import _project
from weft.distal_depth_branches import propose_depth_branches
from weft.distal_pixel_fit import distal_geometry
from weft.spatial_pose import rotation_zyx


def camera(offset=0.):
    return {'matrix': np.array([[700., 0., 640.], [0., 700., 480.], [0., 0., 1.]]),
        'rotation': np.eye(3), 'translation': np.array([offset, 0., 0.]),
        'radial': 0., 'width': 1280, 'height': 960}


def case(count=1, sign=-1):
    rotation = rotation_zyx([.2, -.1, .3]); knee = np.array([.1, .2, 4.])
    target, _ = distal_geometry([.2, .9], rotation, 1.3, .35, knee, sign)
    views = [camera(.5*i) for i in range(count)]
    return dict(rotation=rotation, bend=1.3, length=.35, knee=knee, cameras=views,
        pixels=[_project(target[None, :], c)[0][0].tolist() for c in views], confidence=[.9]*count,
        sign=sign, anchor_direction=(target-knee)/.35)


@pytest.mark.parametrize('count', [1, 2, 4])
@pytest.mark.parametrize('sign', [-1, 1])
def test_supported_branch_retains_all_rays_and_preserves_native_geometry(count, sign):
    args = case(count, sign); before = deepcopy(args)
    result = propose_depth_branches(**args)
    assert result['status'] == 'selected_temporal_hypothesis'
    assert len(result['rays']) == count and sum(len(r['roots']) for r in result['rays']) == 2*count
    assert len(result['candidates']) <= 2*count and not result['depth_measured']
    selected = result['candidates'][result['selected_index']]
    np.testing.assert_allclose(selected['world_m'], args['knee']+.35*args['anchor_direction'], atol=1e-10)
    point, _ = distal_geometry(selected['correction_rad'], args['rotation'], args['bend'], .35, args['knee'], sign)
    np.testing.assert_allclose(point, selected['world_m'], atol=1e-10)
    updated = args['rotation'] @ rotation_zyx([selected['correction_rad'][0], 0., 0.])
    np.testing.assert_allclose(updated[:, 2], args['rotation'][:, 2], atol=1e-12)
    assert len(selected['pixel_errors']) == count and selected['pixel_sse'] < result['pixel_sse_before']
    assert selected['temporal_distance'] < result['temporal_distance_before']
    assert max(abs(selected['correction_rad'][0]), 0.) <= math.pi and 0 <= selected['output_bend_rad'] <= 2.6
    assert args['pixels'] == before['pixels']


def test_missing_anchors_still_retains_every_root_without_selecting_depth():
    args = case(2); args['anchor_direction'] = None
    report = propose_depth_branches(**args)
    assert report['status'] == 'missing_temporal_anchor' and report['selected_index'] is None
    assert sum(len(r['roots']) for r in report['rays']) == 4
    assert any(root['rejection'] == 'duplicate_endpoint' for r in report['rays'] for root in r['roots'])


def central_ray(knee_x):
    return dict(rotation=np.eye(3), bend=.4, length=.35, knee=[knee_x, 0., 4.], cameras=[camera()],
                pixels=[[640., 480.]], confidence=[1.], sign=-1, anchor_direction=[-1., 0., 0.])


def test_tangent_root_is_retained_twice_but_not_ranked_as_two_independent_choices():
    result = propose_depth_branches(**central_ray(.35))
    assert result['status'] == 'selected_temporal_hypothesis'
    assert len(result['rays'][0]['roots']) == 2 and len(result['candidates']) == 1
    assert result['rays'][0]['roots'][1]['rejection'] == 'duplicate_endpoint'
    assert result['runner_up_gap_lengths'] is None


def test_a_missing_intersection_does_not_move_the_knee_or_length():
    result = propose_depth_branches(**central_ray(.36))
    assert result['selected_index'] is None and not result['candidates']
    assert result['rays'][0]['status'] == 'no_sphere_intersection'
    assert result['fixed_knee_world_m'] == [.36, 0., 4.] and result['fixed_distal_length_m'] == .35


@pytest.mark.parametrize('near_cutoff', [False, True])
def test_similar_depth_branches_abstain_even_when_runner_is_just_outside_cutoff(near_cutoff):
    args = central_ray(.30625 if near_cutoff else .33)
    if near_cutoff: args['anchor_direction'] = [-math.cos(.01), 0., math.sin(.01)]
    result = propose_depth_branches(**args)
    assert result['status'] == 'ambiguous_branches' and result['selected_index'] is None
    assert result['runner_up_gap_lengths'] < .25
    if near_cutoff:
        assert min(c['temporal_distance'] for c in result['candidates']) < .5
        assert max(c['temporal_distance'] for c in result['candidates']) > .5


def test_exact_existing_image_match_cannot_change_depth_just_to_follow_temporal_prior():
    args = case(); point, _ = distal_geometry([0., 0.], args['rotation'], args['bend'], .35, args['knee'], -1)
    args['pixels'] = [_project(point[None, :], args['cameras'][0])[0][0].tolist()]
    result = propose_depth_branches(**args)
    assert result['selected_index'] is None
    assert all(c['rejection'] is not None for c in result['candidates'])


def test_conflicting_views_cannot_be_silently_dropped():
    args = case(2); args['pixels'][1][0] += 40
    result = propose_depth_branches(**args)
    assert len(result['rays']) == 2
    for c in result['candidates']:
        if c['pixel_errors'] is not None:
            assert len(c['pixel_errors']) == 2
            expected = sum(.9*(e/1280)**2 for e in c['pixel_errors'])
            assert c['pixel_sse'] == pytest.approx(expected)
    if result['selected_index'] is not None:
        assert result['candidates'][result['selected_index']]['pixel_sse'] < result['pixel_sse_before']


def test_resizing_images_preserves_branches_and_selection():
    args = case(2); original = propose_depth_branches(**args)
    for view in args['cameras']:
        view['matrix'][:2] *= .5; view['width'] //= 2; view['height'] //= 2
    args['pixels'] = (np.array(args['pixels'])*.5).tolist()
    resized = propose_depth_branches(**args)
    assert resized['selected_index'] == original['selected_index']
    assert resized['rays'] == original['rays']
    for a, b in zip(resized['candidates'], original['candidates'], strict=True):
        assert a['correction_rad'] == b['correction_rad'] and a['pixel_sse'] == b['pixel_sse']


def test_nonforward_roots_and_hinge_rejections_remain_visible():
    args = central_ray(0.); args.update(knee=[0., 0., .3], bend=2., anchor_direction=[0., 0., 1.])
    report = propose_depth_branches(**args)
    assert report['rays'][0]['roots'][0]['rejection'] == 'nonforward_root'
    assert report['candidates'][0]['rejection'] == 'hinge_range' and report['selected_index'] is None


@pytest.mark.parametrize('damage', ['rotation', 'bend', 'length', 'knee', 'pixel', 'confidence', 'anchor', 'sign', 'views'])
def test_invalid_geometry_cannot_create_branches(damage):
    args = case()
    if damage == 'rotation': args['rotation'] = np.diag([-1., 1., 1.])
    elif damage == 'bend': args['bend'] = 3.
    elif damage == 'length': args['length'] = 0.
    elif damage == 'knee': args['knee'][0] = float('nan')
    elif damage == 'pixel': args['pixels'][0][0] = -1
    elif damage == 'confidence': args['confidence'] = [.49]
    elif damage == 'anchor': args['anchor_direction'] = [2., 0., 0.]
    elif damage == 'sign': args['sign'] = True
    elif damage == 'views': args['cameras'] *= 5
    with pytest.raises(ValueError): propose_depth_branches(**args)
