from copy import deepcopy

import numpy as np
import pytest

from test_camera_ground import references
from test_camera_pose import fixture
from test_multiray_body import setup as multi_setup
from test_multiview_body import setup as stereo_setup
from test_multiview_root import CLOCK, WORLD
from weft.camera_body_lab import prepare_camera_body_lab
from weft.camera_body import make_camera_body
from weft.camera_ground import apply_camera_ground
from weft.human import HumanEnv, build_human, validate_human
from weft.io import canonical_hash
from weft.multiview_body import make_two_view_body
from weft.multiview_distal_pixels import METHOD, pixel_status
from weft.multiview_root import make_two_view_root


def inputs(tmp_path, count):
    if count == 2:
        root, predictions, first, second, _, _, camera, camera2, _ = stereo_setup(tmp_path,
            hidden={i: [15, 27] for i in range(10)}, monocular_flexion=1.)
        return root, predictions, [first, second], [camera, camera2]
    base, first, second, third, roots = multi_setup(tmp_path)
    views = [first, second, third]
    if count == 4:
        evidence, camera, *_ = fixture(roots=roots, translations=[[-.7, .3, .1]]*len(roots))
        evidence['source_sha256'] = camera['source_sha256'] = 'd'*64
        for row in evidence['frames']: row['id'] = f"{'d'*64}:frame:{row['frame']}"
        views.append({'name': 'fourth', 'evidence': evidence, 'camera': camera, 'clock_provenance': CLOCK})
    for index, view in enumerate(views):
        for row in view['evidence']['frames']:
            if index == 0:
                # Image targets stay exact while monocular distal depth supplies a biased prior.
                row['world_landmarks'][15]['z'] += .06
                row['world_landmarks'][27]['z'] += .08
            if row['frame'] != 7 and (index != 0 or row['frame'] == 5):
                for joint in (15, 27): row['landmarks'][joint]['visibility'] = .1
            if index in (1, 3):
                for joint in (13, 25): row['landmarks'][joint]['presence'] = .1
    original = build_human(first['evidence'], tmp_path/'pixel-body', body_model='mobile')
    calibrated, _ = make_camera_body(first['evidence'], first['camera'], original)
    base = apply_camera_ground(calibrated, base['camera_reference']['ground'])
    root, _, predictions = make_two_view_root(base, first, second, additional_views=views[2:], world_provenance=WORLD)
    return root, predictions, [v['evidence'] for v in views], [v['camera'] for v in views]


def build(root, predictions, evidence, cameras, *, fit=True, mode='both'):
    return make_two_view_body(root, predictions, *evidence[:2], additional_evidence=evidence[2:],
        mode=mode, limb_fit=True, partial_segments=True, partial_pixel_fit=fit, pixel_cameras=cameras if fit else ())


@pytest.mark.parametrize('count', [2, 3, 4])
@pytest.mark.parametrize('mode', ['directions', 'both'])
def test_pixel_proposals_preserve_native_upper_bones_and_every_complete_fit(tmp_path, count, mode):
    root, predictions, evidence, cameras = inputs(tmp_path, count)
    original = canonical_hash([root, predictions, evidence, cameras])
    old, before_audit = build(root, predictions, evidence, cameras, fit=False, mode=mode)
    new, audit = build(root, predictions, evidence, cameras, mode=mode)
    assert canonical_hash([root, predictions, evidence, cameras]) == original
    assert new['camera_reference']['method'] == METHOD and len(audit['frames']) == 10
    for name in ('lengths', 'root_reference', 'body_frame', 'mobility', 'motor_limits'):
        assert new[name] == old[name]
    assert new['camera_reference']['ground'] == old['camera_reference']['ground']
    assert audit['limb_position_fit'] == before_audit['limb_position_fit']
    assert audit['partial_pixels']['partial_reference_sha256'] == canonical_hash(old['reference'])
    assert audit['partial_pixels']['changed_group_poses'] > 0
    for i, (a, b) in enumerate(zip(references(old), references(new), strict=True)):
        meta = new['reference_evidence'][i]; changed = set()
        assert meta['geometry_fit'] == old['reference_evidence'][i]['geometry_fit']
        for group, fit in meta['geometry_partial_pixels'].items():
            leg = group.endswith('hip'); distal = group.rsplit('_', 1)[0]+('_knee' if leg else '_elbow')
            end = group.rsplit('_', 1)[0]+('_ankle' if leg else '_wrist')
            changed.update((group+'_twist', group+'_lateral', group, distal))
            for name in (group, distal):
                np.testing.assert_allclose(a['human']['joints'][name], b['human']['joints'][name], atol=1e-12)
            assert np.linalg.norm(np.subtract(b['human']['joints'][end], b['human']['joints'][distal])) == pytest.approx(new['lengths']['shin' if leg else 'forearm'])
            offset = np.r_[new['camera_reference']['origin_world_m'][:2], new['camera_reference']['ground']['height_m']]
            full = audit['frames'][i]['groups'][group]['distal_pixel_attempt']['fit']
            np.testing.assert_allclose(np.array(b['human']['joints'][end])+offset, full['after_world_m'], atol=1e-12)
            assert fit['pixel_sse_after'] <= fit['pixel_sse_before'] and len(fit['observation_statuses']) == count
            assert fit['view_indices'] == [0] and not full['depth_measured']
            assert meta['geometry_sources'][group] == pixel_status(count > 2)
            assert meta['geometry_components'][group]['distal_endpoint'] == 'image_constrained_depth_unresolved'
            assert meta['joint_confidence'][distal] == 0.
        assert {k: v for k, v in new['reference'][i]['joints'].items() if k not in changed} == {
            k: v for k, v in old['reference'][i]['joints'].items() if k not in changed}
        for group in meta['geometry_fit']:
            assert audit['frames'][i]['groups'][group]['length_fit'] == before_audit['frames'][i]['groups'][group]['length_fit']
    if count > 2:
        assert new['reference_evidence'][5]['geometry_partial_pixels'] == {}
        assert 'No qualified' in audit['frames'][5]['groups']['left_hip']['distal_pixel_attempt']['reason']
        assert audit['frames'][6]['groups']['left_hip']['distal_pixel_attempt']['fit']['previous_correction_rad'] is None
        assert new['reference_evidence'][7]['geometry_partial_pixels'] == {}
        assert audit['frames'][8]['groups']['left_hip']['distal_pixel_attempt']['fit']['previous_correction_rad'] is None
    env = HumanEnv(new)
    try:
        assert env.scene['human']['partial_pixels'] == audit['partial_pixels']
        assert not env.scene['learned_policy_available']
    finally: env.close()


@pytest.mark.parametrize('damage', ['contract', 'calibration', 'rotation', 'bend', 'correction', 'objective',
    'missing_fit', 'confidence', 'component', 'fake_view', 'omit_view_status', 'unknown_status', 'method'])
def test_resealed_pixel_proposals_cannot_promote_depth_or_rewrite_supported_geometry(tmp_path, damage):
    root, predictions, evidence, cameras = inputs(tmp_path, 3)
    body, _ = build(root, predictions, evidence, cameras)
    nested = body['camera_reference']['body_refinement']; meta = body['reference_evidence'][0]
    fit = meta['geometry_partial_pixels']['left_hip']
    if damage == 'contract': nested['partial_pixel_contract']['maximum_angle_change_rad'] = 1.
    elif damage == 'calibration': nested['partial_pixel_camera_sha256'][0] = 'f'*64
    elif damage == 'rotation': body['reference'][0]['joints']['left_hip_lateral'] += .01
    elif damage == 'bend': fit['input_angles_rad'][3] += .01
    elif damage == 'correction': fit['correction_rad'][0] = .31
    elif damage == 'objective': fit['pixel_sse_after'] = fit['pixel_sse_before']+1.
    elif damage == 'missing_fit': del meta['geometry_partial_pixels']['left_hip']
    elif damage == 'confidence': meta['joint_confidence']['left_knee'] = .8
    elif damage == 'component': meta['geometry_components']['left_hip']['distal_endpoint'] = 'observed'
    elif damage == 'fake_view': fit['view_indices'] = [0, 2]
    elif damage == 'omit_view_status': fit['observation_statuses'].pop()
    elif damage == 'unknown_status': fit['status'] = 'physically_validated'
    elif damage == 'method': body['camera_reference']['method'] = 'camera-body-length-fit-v2'
    nested['output_evidence_sha256'] = canonical_hash(body['reference_evidence'])
    body['camera_reference']['angular_reference_sha256'] = canonical_hash(body['reference'])
    with pytest.raises(ValueError): validate_human(body)


@pytest.mark.parametrize('damage', ['missing_camera', 'camera', 'source', 'no_partial', 'no_length_fit', 'not_boolean'])
def test_partial_pixels_reject_incompatible_or_unbound_inputs(tmp_path, damage):
    root, predictions, evidence, cameras = inputs(tmp_path, 2)
    before = canonical_hash(root)
    options = dict(partial_pixel_fit=True, partial_segments=True, limb_fit=True, pixel_cameras=deepcopy(cameras))
    if damage == 'missing_camera': options['pixel_cameras'].pop()
    elif damage == 'camera': options['pixel_cameras'][1]['frames'][0]['t'][0] += .01
    elif damage == 'source': options['pixel_cameras'][1]['source_sha256'] = 'e'*64
    elif damage == 'no_partial': options['partial_segments'] = False
    elif damage == 'no_length_fit': options['limb_fit'] = False
    elif damage == 'not_boolean': options['partial_pixel_fit'] = 1
    with pytest.raises(ValueError): make_two_view_body(root, predictions, *evidence, **options)
    assert canonical_hash(root) == before


def test_packager_rejects_incomplete_pixel_options_before_creating_output(tmp_path):
    with pytest.raises(ValueError, match='requires --fit-limbs'):
        prepare_camera_body_lab(tmp_path/'input', tmp_path/'camera.json', tmp_path/'out', partial_pixel_fit=True)
    assert not (tmp_path/'out').exists()
