from copy import deepcopy

import numpy as np
import pytest

from test_camera_ground import references
from test_camera_pose import fixture
from test_multiray_body import setup as multi_setup
from test_multiview_body import setup as stereo_setup
from test_multiview_root import CLOCK, WORLD
from weft.human import validate_human
from weft.io import canonical_hash
from weft.multiview_body import LIMB_FIT_SEGMENT_METHOD, make_two_view_body
from weft.multiview_root import make_two_view_root
from weft.multiview_segments import METHOD, MULTI_METHOD, PARTIAL_STATUS, MULTI_PARTIAL_STATUS
from weft.spatial_pose import rotation_zyx


def inputs(tmp_path, count, *, middle_missing=False):
    if count == 2:
        hidden = [13, 25] if middle_missing else [15, 27]
        root, predictions, first, second, expected, *_ = stereo_setup(tmp_path,
            hidden={i: hidden for i in range(10)}, monocular_flexion=1.,
            monocular_limb=rotation_zyx([.2, -.2, .15]))
        return root, predictions, first, second, [], expected
    base, first, second, third, roots = multi_setup(tmp_path)
    extra = [third]
    evidence, camera, relative, _ = fixture(roots=roots, translations=[[-.7, .3, .1]]*len(roots))
    if count == 4:
        evidence['source_sha256'] = camera['source_sha256'] = 'd'*64
        for row in evidence['frames']: row['id'] = f"{'d'*64}:frame:{row['frame']}"
        extra.append({'name': 'fourth', 'evidence': evidence, 'camera': camera, 'clock_provenance': CLOCK})
    for index, view in enumerate([first, second, *extra]):
        for row in view['evidence']['frames']:
            if index != 0:
                for joint in ([13, 25] if middle_missing else [15, 27]):
                    row['landmarks'][joint]['visibility'] = .1
            if index in (1, 3):
                for joint in (13, 25): row['landmarks'][joint]['presence'] = .1
    root, _, predictions = make_two_view_root(base, first, second, additional_views=extra, world_provenance=WORLD)
    return root, predictions, first['evidence'], second['evidence'], [v['evidence'] for v in extra], np.asarray(roots)[:, None, :]+relative


@pytest.mark.parametrize('count', [2, 3, 4])
@pytest.mark.parametrize('fit', [False, True])
@pytest.mark.parametrize('mode', ['directions', 'both'])
def test_partial_transport_matches_native_direction_and_keeps_complete_fits_exact(tmp_path, count, fit, mode):
    root, predictions, first, second, extra, expected = inputs(tmp_path, count)
    before = canonical_hash([root, predictions, first, second, extra])
    options = {'additional_evidence': extra, 'limb_fit': fit, 'mode': mode}
    old, original_audit = make_two_view_body(root, predictions, first, second, **options)
    new, audit = make_two_view_body(root, predictions, first, second, partial_segments=True, **options)
    assert canonical_hash([root, predictions, first, second, extra]) == before
    assert new['camera_reference']['method'] == (LIMB_FIT_SEGMENT_METHOD if fit else MULTI_METHOD if count > 2 else METHOD)
    for field in ('lengths', 'root_reference', 'body_frame', 'mobility', 'motor_limits'):
        assert new[field] == old[field]
    for field in ('ground', 'root_refinement'):
        assert new['camera_reference'][field] == old['camera_reference'][field]
    assert audit['dimensions'] == original_audit['dimensions']
    assert len(audit['frames']) == len(predictions['frames']) == 10
    assert audit['partial_segments']['complete_limb_reference_sha256'] == canonical_hash(old['reference'])
    assert audit['partial_segments']['applied_group_poses'] == 20
    changed_names = {g+s for g in ('left_hip', 'left_shoulder') for s in ('_twist', '_lateral', '')}
    for i, snapshot in enumerate(references(new)):
        a, b = old['reference'][i]['joints'], new['reference'][i]['joints']
        assert {k: v for k, v in a.items() if k not in changed_names} == {k: v for k, v in b.items() if k not in changed_names}
        meta = new['reference_evidence'][i]
        for group, distal, joint_ids in [('left_hip', 'left_knee', (23, 25)), ('left_shoulder', 'left_elbow', (11, 13))]:
            points = snapshot['human']['joints']
            actual = np.subtract(points[distal], points[group]); target = expected[i, joint_ids[1]]-expected[i, joint_ids[0]]
            np.testing.assert_allclose(actual/np.linalg.norm(actual), target/np.linalg.norm(target), atol=1e-10)
            assert b[distal] == root['reference'][i]['joints'][distal]
            assert meta['geometry_sources'][group] == (MULTI_PARTIAL_STATUS if count > 2 else PARTIAL_STATUS)
            assert meta['geometry_components'][group]['distal_endpoint'] == 'unavailable'
            assert all(meta['joint_confidence'][group+s] == 0. for s in ('_twist', '_lateral', ''))
            detail = audit['frames'][i]['groups'][group]
            assert detail['point_statuses'][-1] != 'estimated'
            assert 'length_fit' not in detail
            if count > 2:
                assert detail['proximal_image_confidence'] == 1.
                assert meta['geometry_view_indices'][group][str(joint_ids[1])] == [0, 2]
                assert set(meta['geometry_view_indices'][group]) == {str(j) for j in joint_ids}
        if fit:
            assert meta['geometry_fit'] == old['reference_evidence'][i]['geometry_fit']
            for group in meta['geometry_fit']:
                assert audit['frames'][i]['groups'][group]['length_fit'] == original_audit['frames'][i]['groups'][group]['length_fit']


@pytest.mark.parametrize('count', [2, 3, 4])
def test_missing_middle_keeps_the_entire_explicit_prior(tmp_path, count):
    root, predictions, first, second, extra, _ = inputs(tmp_path, count, middle_missing=True)
    old, _ = make_two_view_body(root, predictions, first, second, additional_evidence=extra, limb_fit=True)
    new, audit = make_two_view_body(root, predictions, first, second, additional_evidence=extra, limb_fit=True, partial_segments=True)
    assert new['reference'] == old['reference']
    assert audit['partial_segments']['applied_group_poses'] == 0


@pytest.mark.parametrize('damage', ['contract', 'source', 'view_method', 'distal', 'twist_confidence',
    'components', 'missing_contributors', 'invented_endpoint', 'one_camera', 'duplicate', 'camera_range', 'boolean_camera', 'method', 'invented_fit'])
def test_partial_multiview_binding_rejects_resealed_claims(tmp_path, damage):
    root, predictions, first, second, extra, _ = inputs(tmp_path, 3)
    body, _ = make_two_view_body(root, predictions, first, second, additional_evidence=extra, limb_fit=True, partial_segments=True)
    nested = body['camera_reference']['body_refinement']; meta = body['reference_evidence'][0]
    contributors = meta['geometry_view_indices']['left_hip']
    if damage == 'contract': nested['segment_contract']['ground_fitting'] = True
    elif damage == 'source': meta['geometry_sources']['left_hip'] = 'measured_anatomy'
    elif damage == 'view_method': meta['geometry_sources']['left_hip'] = PARTIAL_STATUS
    elif damage == 'distal': body['reference'][0]['joints']['left_knee'] += .01
    elif damage == 'twist_confidence': meta['joint_confidence']['left_hip_twist'] = .9
    elif damage == 'components': meta['geometry_components']['left_hip']['distal_endpoint'] = 'observed'
    elif damage == 'missing_contributors': del meta['geometry_view_indices']['left_hip']
    elif damage == 'invented_endpoint': contributors['27'] = [0, 2]
    elif damage == 'one_camera': contributors['25'] = [0]
    elif damage == 'duplicate': contributors['25'] = [0, 0]
    elif damage == 'camera_range': contributors['25'] = [0, 3]
    elif damage == 'boolean_camera': contributors['25'] = [False, 2]
    elif damage == 'method': body['camera_reference']['method'] = 'camera-body-length-fit-v1'
    elif damage == 'invented_fit': meta['geometry_fit']['left_hip'] = deepcopy(meta['geometry_fit']['right_hip'])
    nested['output_evidence_sha256'] = canonical_hash(body['reference_evidence'])
    body['camera_reference']['angular_reference_sha256'] = canonical_hash(body['reference'])
    with pytest.raises(ValueError): validate_human(body)
