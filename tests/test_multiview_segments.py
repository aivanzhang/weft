from copy import deepcopy
import math

import numpy as np
import pytest

from test_camera_ground import references
from test_multiview_body import setup
from weft.camera_body_lab import prepare_camera_body_lab
from weft.human import validate_human
from weft.io import canonical_hash
from weft.multiview_body import make_two_view_body
from weft.multiview_segments import PARTIAL_STATUS, transport_rotation
from weft.spatial_pose import rotation_zyx


@pytest.mark.parametrize('prior_angles,swing_angles', [([0, 0, 0], [0, .8, 0]),
    ([0, 0, 0], [0, 0, -.6]), ([0, math.pi/2, 0], [.7, 0, 0]), ([.3, .4, .5], [0, 0, 0])])
def test_minimum_rotation_transports_existing_frame_without_invented_axial_rotation(prior_angles, swing_angles):
    prior, swing = rotation_zyx(prior_angles), rotation_zyx(swing_angles)
    target = swing @ prior @ [0, 0, -1]
    result, angle = transport_rotation(prior, target)
    np.testing.assert_allclose(result, swing @ prior, atol=1e-10)
    np.testing.assert_allclose(result.T @ result, np.eye(3), atol=1e-10)
    assert np.linalg.det(result) == pytest.approx(1.)
    assert angle == pytest.approx(float(np.linalg.norm(swing_angles)), abs=1e-7)


@pytest.mark.parametrize('target', [[0, 0, 1], [0, 0, 0], [1e-7, 0, 0], [float('nan'), 0, 1], [0, 1]])
def test_ambiguous_degenerate_or_invalid_directions_are_rejected(target):
    with pytest.raises(ValueError): transport_rotation(np.eye(3), target)


@pytest.mark.parametrize('prior', [np.diag([-1, 1, 1]), np.ones((3, 3)), np.eye(2)])
def test_transport_rejects_improper_prior_frames(prior):
    with pytest.raises(ValueError): transport_rotation(prior, [0, 0, -1])


@pytest.mark.parametrize('mode', ['directions', 'both'])
def test_upper_directions_recover_known_geometry_without_measuring_hidden_bend(tmp_path, mode):
    base, predictions, first, second, expected, *_ = setup(tmp_path,
        hidden={i: [15, 27] for i in range(10)}, monocular_flexion=1.0,
        monocular_limb=rotation_zyx([.2, -.2, .15]))
    before = canonical_hash([base, predictions, first, second])
    old, old_audit = make_two_view_body(base, predictions, first, second, mode=mode)
    new, audit = make_two_view_body(base, predictions, first, second, mode=mode, partial_segments=True)
    assert canonical_hash([base, predictions, first, second]) == before
    for name in ('lengths', 'root_reference', 'mobility', 'body_frame', 'motor_limits'):
        assert new[name] == old[name]
    assert audit['partial_segments']['complete_limb_reference_sha256'] == canonical_hash(old['reference'])
    assert audit['partial_segments']['applied_group_poses'] == 20
    assert audit['partial_segments']['rejected_group_poses'] == 0
    offset = [*new['camera_reference']['origin_world_m'][:2], 0]
    changes = []
    for frame, (a, b, pose) in enumerate(zip(old['reference'], new['reference'], references(new), strict=True)):
        for name in a['joints']:
            if name not in ('left_hip', 'left_hip_lateral', 'left_hip_twist', 'left_shoulder', 'left_shoulder_lateral', 'left_shoulder_twist'):
                assert a['joints'][name] == b['joints'][name]
        changes.append(max(abs(a['joints'][name]-b['joints'][name]) for name in a['joints']))
        positions = {name: np.asarray(point)+offset for name, point in pose['human']['joints'].items()}
        for group, start, end, indices, distal in [('left_hip', 'left_hip', 'left_knee', (23, 25), 'left_knee'),
                                                  ('left_shoulder', 'left_shoulder', 'left_elbow', (11, 13), 'left_elbow')]:
            vector, target = positions[end]-positions[start], expected[frame, indices[1]]-expected[frame, indices[0]]
            np.testing.assert_allclose(vector/np.linalg.norm(vector), target/np.linalg.norm(target), atol=1e-10)
            assert b['joints'][distal] == pytest.approx(1., abs=1e-10)
            meta = new['reference_evidence'][frame]
            assert meta['geometry_sources'][group] == PARTIAL_STATUS
            assert meta['geometry_components'][group]['distal_endpoint'] == 'unavailable'
            assert all(meta['joint_confidence'][group+suffix] == 0. for suffix in ('_twist', '_lateral', ''))
            detail = audit['frames'][frame]['groups'][group]
            assert detail['point_statuses'][-1] == 'image_evidence_unavailable'
            assert detail['proximal_image_confidence'] == .9
    assert min(changes) > .01
    assert audit['dimensions'] == old_audit['dimensions']
    assert audit['group_coverage']['left_shoulder'] == {PARTIAL_STATUS: 10}


def test_missing_middle_or_base_does_not_invent_a_proximal_direction(tmp_path):
    base, predictions, first, second, *_ = setup(tmp_path, hidden={i: [11, 25] for i in range(10)})
    old, _ = make_two_view_body(base, predictions, first, second)
    new, audit = make_two_view_body(base, predictions, first, second, partial_segments=True)
    assert new['reference'] == old['reference']
    assert audit['partial_segments']['applied_group_poses'] == 0
    assert all(row['groups']['left_shoulder']['status'] == 'monocular_prior_missing_points' for row in audit['frames'])


def test_all_complete_limb_outputs_remain_byte_identical(tmp_path):
    base, predictions, first, second, *_ = setup(tmp_path)
    old, old_audit = make_two_view_body(base, predictions, first, second)
    new, audit = make_two_view_body(base, predictions, first, second, partial_segments=True)
    assert new['reference'] == old['reference']
    assert audit['group_coverage'] == old_audit['group_coverage']
    assert audit['partial_segments']['applied_group_poses'] == 0
    for a, b in zip(references(old), references(new), strict=True): assert a == b


@pytest.mark.parametrize('damage', ['contract', 'distal', 'twist_confidence', 'component_source', 'method', 'metadata_version'])
def test_partial_binding_does_not_promote_uncertain_components_or_relax_old_mode(tmp_path, damage):
    base, predictions, first, second, *_ = setup(tmp_path, hidden={i: [27] for i in range(10)})
    new, _ = make_two_view_body(base, predictions, first, second, partial_segments=True)
    binding = new['camera_reference']; nested = binding['body_refinement']
    if damage == 'contract': nested['segment_contract']['ground_fitting'] = True
    elif damage == 'distal':
        new['reference'][0]['joints']['left_knee'] += .01
        binding['angular_reference_sha256'] = canonical_hash(new['reference'])
    elif damage == 'twist_confidence':
        new['reference_evidence'][0]['joint_confidence']['left_hip_twist'] = .9
        nested['output_evidence_sha256'] = canonical_hash(new['reference_evidence'])
    elif damage == 'component_source':
        new['reference_evidence'][0]['geometry_components']['left_hip']['distal_endpoint'] = 'observed'
        nested['output_evidence_sha256'] = canonical_hash(new['reference_evidence'])
    elif damage == 'method': binding['method'] = 'two-view-body-v1'
    elif damage == 'metadata_version':
        new['reference_evidence'][0]['geometry_method'] = 'two-view-body-v1'
        nested['output_evidence_sha256'] = canonical_hash(new['reference_evidence'])
    with pytest.raises(ValueError): validate_human(new)


@pytest.mark.parametrize('mode,partial', [('dimensions', True), ('both', 1), ('both', 'yes')])
def test_partial_segments_require_an_explicit_direction_enabled_mode(tmp_path, mode, partial):
    base, predictions, first, second, *_ = setup(tmp_path)
    with pytest.raises(ValueError, match='explicit boolean'):
        make_two_view_body(base, predictions, first, second, mode=mode, partial_segments=partial)


@pytest.mark.parametrize('mode', [None, 'dimensions'])
def test_packager_rejects_unsupported_partial_request_before_creating_output(tmp_path, mode):
    with pytest.raises(ValueError, match='Partial segments require'):
        prepare_camera_body_lab(tmp_path/'source', tmp_path/'camera.json', tmp_path/'out',
            root_view_path=tmp_path/'view.json', body_view=mode, partial_segments=True)
    assert not (tmp_path/'out').exists()
