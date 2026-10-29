from copy import deepcopy
import json

import numpy as np
import pytest

from test_camera_ground import ground_for, references
from test_spatial_pose import pose_frame
from weft.body_frame import CAMERA_TO_SIM
from weft.camera_body import make_camera_body
from weft.camera_body_lab import prepare_camera_body_lab
from weft.camera_ground import apply_camera_ground
from weft.camera_reference import JOINTS
from weft.human import build_human, validate_human
from weft.io import canonical_hash, digest, write_json
from weft.lab import CameraHumanSession, Lab
from weft.multiview_body import make_two_view_body
from weft.multiview_root import make_two_view_root
from weft.pipeline import verify_artifacts
from weft.recording import load_recording, verify_recording
from weft.spatial_pose import LIMB_ENDPOINTS, rotation_zyx


LENGTHS = {'torso': .5, 'shoulder_width': .4, 'hip_width': .26,
           'upper_arm': .3, 'forearm': .25, 'thigh': .3, 'shin': .25}


def setup(tmp_path, *, hidden=None, flexion=.8, monocular_flexion=None, monocular_limb=None):
    source = tmp_path/'source'; source.mkdir()
    (source/'source.mp4').write_bytes(b'Analytic geometry, not recorded physical evidence')
    source_hash = digest(source/'source.mp4')
    rotation = np.array([[1., 0., 0.], [0., 0., -1.], [0., 1., 0.]])
    intrinsics = np.array([[700., 0, 640], [0, 710., 480.], [0, 0, 1.]])
    evidence = {'schema_version': 1, 'profile': 'exercise-pose-v1', 'source_sha256': source_hash,
                'width': 1280, 'height': 960, 'fps': 30., 'frame_count': 10, 'frames': [],
                'lengths': {key: value*.8 for key, value in LENGTHS.items()}, 'reference': []}
    camera = {'schema_version': 1, 'source_sha256': source_hash, 'width': 1280, 'height': 960,
              'fps': 30., 'frame_count': 10, 'world_axes': 'right_handed_z_up', 'units': 'm',
              'provenance': 'Known analytic cameras', 'quality': 'authored',
              'K': intrinsics.tolist(), 'radial_k1': 0., 'frames': []}
    second = deepcopy(evidence); second['source_sha256'] = 'b'*64
    camera2 = deepcopy(camera); camera2['source_sha256'] = 'b'*64
    expected = []
    for frame in range(10):
        raw = pose_frame(frame, torso=rotation_zyx([.2, -.1, .1]), limb=rotation_zyx([.2, -.3, .1]), flexion=flexion)
        points = np.asarray([[p[k] for k in ('x', 'y', 'z')] for p in raw['world_landmarks']]) @ CAMERA_TO_SIM.T
        root = np.array([.05*frame, 5., 1.2+.01*frame])
        expected.append(points+root)
        mono_points = points
        if monocular_flexion is not None or monocular_limb is not None:
            mono = pose_frame(frame, torso=rotation_zyx([.2, -.1, .1]),
                             limb=rotation_zyx([.2, -.3, .1]) if monocular_limb is None else monocular_limb,
                             flexion=flexion if monocular_flexion is None else monocular_flexion)
            mono_points = np.asarray([[p[k] for k in ('x', 'y', 'z')] for p in mono['world_landmarks']]) @ CAMERA_TO_SIM.T
        biased = .8*mono_points @ rotation_zyx([.04, -.04, .03]).T @ rotation.T
        for view, cam, translation in [(evidence, camera, np.zeros(3)), (second, camera2, np.array([-1., 0, 0]))]:
            positions = (points+root) @ rotation.T+translation
            projected = (positions/positions[:, 2, None]) @ intrinsics.T
            landmarks = [{'x': float(p[0]/1280), 'y': float(p[1]/960), 'visibility': .95, 'presence': .9} for p in projected]
            if hidden:
                for joint in hidden.get(frame, []):
                    if view is second: landmarks[joint]['visibility'] = .1
            view['frames'].append({'id': f"{view['source_sha256']}:frame:{frame}", 'frame': frame, 'time_s': frame/30,
                'decoder_timestamp_s': frame/30, 'pose_count': 1, 'status': 'tracked', 'landmarks': landmarks,
                'world_landmarks': [dict(zip(('x', 'y', 'z'), p.tolist())) for p in biased], 'joint_confidence': raw['joint_confidence']})
            view['reference'].append({'time_s': frame/30, 'evidence_id': f"{view['source_sha256']}:frame:{frame}"})
            cam['frames'].append({'frame': frame, 'time_s': frame/30, 'R': rotation.tolist(), 't': translation.tolist()})
    original = build_human(evidence, source, body_model='mobile')
    base, _ = make_camera_body(evidence, camera, original)
    ground = ground_for(base, height=0.)
    base = apply_camera_ground(base, ground)
    first, other = ({'name': name, 'evidence': ev, 'camera': cam, 'clock_provenance': 'Analytic simultaneous frames'}
                    for name, ev, cam in [('first', evidence, camera), ('second', second, camera2)])
    root, _, predictions = make_two_view_root(base, first, other, world_provenance='Known analytic shared world')
    return root, predictions, evidence, second, np.array(expected), source, camera, camera2, ground


@pytest.mark.parametrize('mode', ['directions', 'dimensions', 'both'])
def test_known_geometry_modes_preserve_root_ground_and_declared_unchanged_fields(tmp_path, mode):
    base, predictions, first, second, expected, *_ = setup(tmp_path)
    before = canonical_hash([base, predictions, first, second])
    new, audit = make_two_view_body(base, predictions, first, second, mode=mode)
    assert canonical_hash([base, predictions, first, second]) == before
    for key in ('root_reference', 'body_frame', 'mobility', 'motor_limits'):
        assert new[key] == base[key]
    assert new['camera_reference']['ground'] == base['camera_reference']['ground']
    assert canonical_hash(audit) == new['camera_reference']['body_refinement']['audit_sha256']
    if mode == 'directions': assert new['lengths'] == base['lengths']
    if mode == 'dimensions':
        assert new['reference'] == base['reference']
        assert new['reference_evidence'] == base['reference_evidence']
    if mode in ('dimensions', 'both'):
        for name, value in LENGTHS.items(): assert new['lengths'][name] == pytest.approx(value, abs=1e-10)
    offset = [*new['camera_reference']['origin_world_m'][:2], 0]
    for frame, pose in enumerate(references(new)):
        positions = {name: np.asarray(point)+offset for name, point in pose['human']['joints'].items()}
        if mode == 'both':
            for name, index in JOINTS.items(): np.testing.assert_allclose(positions[name], expected[frame, index], atol=1e-10)
        if mode in ('directions', 'both'):
            for group, indices in LIMB_ENDPOINTS.items():
                names = [next(name for name, value in JOINTS.items() if value == index) for index in indices]
                for a, b, ia, ib in zip(names[:2], names[1:], indices[:2], indices[1:]):
                    actual = positions[b]-positions[a]; target = expected[frame, ib]-expected[frame, ia]
                    np.testing.assert_allclose(actual/np.linalg.norm(actual), target/np.linalg.norm(target), atol=1e-10)
    assert audit['group_coverage']['right_shoulder'] == {'two_view_estimate': 10}


def test_missing_limbs_keep_local_prior_angles_and_zero_invented_coverage(tmp_path):
    base, predictions, first, second, *_ = setup(tmp_path, hidden={i: [16, *([27] if i == 5 else [])] for i in range(10)})
    new, audit = make_two_view_body(base, predictions, first, second)
    assert audit['group_coverage']['right_shoulder'] == {'monocular_prior_missing_points': 10}
    assert audit['group_coverage']['left_hip'] == {'two_view_estimate': 9, 'monocular_prior_missing_points': 1}
    for a, b in zip(base['reference'], new['reference'], strict=True):
        for name in ('right_shoulder_twist', 'right_shoulder_lateral', 'right_shoulder', 'right_elbow'):
            assert a['joints'][name] == b['joints'][name]
    assert audit['dimensions']['forearm']['distinct_frames'] == 10
    assert len(audit['dimensions']['forearm']['samples']) == 10
    for name in ('left_hip_twist', 'left_hip_lateral', 'left_hip', 'left_knee'):
        assert new['reference'][5]['joints'][name] == base['reference'][5]['joints'][name]
    assert new['reference_evidence'][5]['geometry_sources']['left_hip'] == 'monocular_prior_missing_points'


def test_dimensions_require_five_distinct_frames_without_clipping_or_filling(tmp_path):
    base, predictions, first, second, *_ = setup(tmp_path, hidden={i: [15, 16] for i in range(4, 10)})
    new, audit = make_two_view_body(base, predictions, first, second)
    detail = audit['dimensions']['forearm']
    assert detail['distinct_frames'] == 4 and len(detail['samples']) == 8
    assert detail['status'] == 'monocular_prior_insufficient_frames' and not detail['applied']
    assert new['lengths']['forearm'] == base['lengths']['forearm']


def test_flexion_limits_are_audited_not_relaxed(tmp_path):
    base, predictions, first, second, *_ = setup(tmp_path, flexion=2.8)
    new, audit = make_two_view_body(base, predictions, first, second)
    for row, angles in zip(audit['frames'], new['reference'], strict=True):
        for group in LIMB_ENDPOINTS:
            assert row['groups'][group]['flexion_clamped'] is True
        assert angles['joints']['left_knee'] == angles['joints']['right_elbow'] == 2.6


@pytest.mark.parametrize('damage', ['predictions', 'first_image', 'second_image', 'mode', 'repeat'])
def test_mismatched_inputs_and_stacked_candidates_are_rejected(tmp_path, damage):
    base, predictions, first, second, *_ = setup(tmp_path)
    mode = 'both'
    if damage == 'predictions': predictions['frames'][0]['joints'][27]['world_point_m'][0] += .1
    elif damage == 'first_image': first['frames'][0]['landmarks'][27]['x'] += .01
    elif damage == 'second_image': second['source_sha256'] = 'c'*64
    elif damage == 'mode': mode = 'automatic'
    elif damage == 'repeat': base, _ = make_two_view_body(base, predictions, first, second)
    with pytest.raises(ValueError): make_two_view_body(base, predictions, first, second, mode=mode)


@pytest.mark.parametrize('damage', ['length', 'angle', 'evidence', 'root', 'floor', 'force', 'parent', 'parent_type',
                                   'parent_chain', 'mode', 'audit', 'binding', 'directions_length', 'dimensions_angle'])
def test_nested_binding_preserves_original_root_contract_and_geometry_provenance(tmp_path, damage):
    base, predictions, first, second, *_ = setup(tmp_path)
    mode = 'directions' if damage == 'directions_length' else 'dimensions' if damage == 'dimensions_angle' else 'both'
    new, _ = make_two_view_body(base, predictions, first, second, mode=mode)
    nested = new['camera_reference']['body_refinement']
    if damage in ('length', 'directions_length'):
        new['lengths']['shin'] += .01
        if damage == 'directions_length': nested['output_lengths_sha256'] = canonical_hash(new['lengths'])
    elif damage in ('angle', 'dimensions_angle'):
        new['reference'][1]['joints']['left_knee'] += .01
        if damage == 'dimensions_angle': new['camera_reference']['angular_reference_sha256'] = canonical_hash(new['reference'])
    elif damage == 'evidence': new['reference_evidence'][0]['geometry_sources']['right_hip'] = 'measured'
    elif damage == 'root': new['root_reference'][1]['translation'][0] += .01
    elif damage == 'floor': new['mobility']['base_height'] += .01
    elif damage == 'force': new['motor_limits']['left_knee'] += 1
    elif damage == 'parent': nested['input_body']['lengths']['shin'] += .01
    elif damage == 'parent_type': nested['input_body']['camera_reference'] = None
    elif damage == 'parent_chain': nested['input_body']['camera_reference']['method'] = 'two-view-body-v1'
    elif damage == 'mode': nested['mode'] = 'arbitrary'
    elif damage == 'audit': nested['audit_sha256'] = 'z'*64
    elif damage == 'binding': del new['camera_reference']['body_refinement']
    with pytest.raises(ValueError): validate_human(new)


@pytest.mark.parametrize('partial_segments', [False, True])
def test_full_partial_body_package_interaction_and_recording(tmp_path, partial_segments):
    hidden = {i: [16, 27] for i in range(10)} if partial_segments else None
    _, _, first, second, _, source, camera, camera2, ground = setup(tmp_path, hidden=hidden)
    write_json(source/'evidence.json', first)
    write_json(source/'source.json', {'artifact': 'source.mp4', 'sha256': first['source_sha256']})
    write_json(source/'task-proposal.json', {'selected_joint': 'left_knee'})
    write_json(source/'manifest.json', {'files': {p.name: digest(p) for p in source.iterdir() if p.is_file()}})
    for name, value in [('first-camera.json', camera), ('second-camera.json', camera2),
                        ('second-evidence.json', second), ('ground.json', ground)]: write_json(tmp_path/name, value)
    write_json(tmp_path/'config.json', {'schema_version': 1, 'first_name': 'first', 'second_name': 'second',
        'second_evidence': 'second-evidence.json', 'second_camera': 'second-camera.json',
        'clock_provenance': 'Known analytic clock', 'world_provenance': 'Known analytic common world'})
    before = verify_artifacts(source)
    result = prepare_camera_body_lab(source, tmp_path/'first-camera.json', tmp_path/'lab',
        ground_path=tmp_path/'ground.json', root_view_path=tmp_path/'config.json', body_view='both', partial_segments=partial_segments)
    run = tmp_path/'lab/run'
    assert (run/'two-view-body.json').exists() and (run/'inputs/two-view-root-body.json').exists()
    assert not (run/'policy.json').exists()
    lab = Lab(tmp_path/'lab')
    try:
        session = lab.load_session()
        assert session.scene['human']['geometry_mode'] == 'both'
        assert session.scene['human']['root_method'] == ('two-view-body-v2' if partial_segments else 'two-view-body-v1')
        assert session.scene['learned_policy_available'] is False
        lab.control('/api/record/start', {})
        for action in (1, 2, 7, 9): lab.control('/api/step', {'action': action, 'steps': 2})
        lab.control('/api/reset', {'seed': 4})
        stopped = lab.control('/api/record/stop', {})
        lab.export_future.result(timeout=15)
        recording = tmp_path/'lab/sessions'/stopped['directory']
        replay = CameraHumanSession(run)
        try:
            for event in load_recording(recording)['events']:
                if event['type'] in ('start', 'reset'): replay.reset(event['seed'])
                else:
                    reward, terminated, truncated, _ = replay.step(event['action'])
                    assert (reward, terminated, truncated) == (event['reward'], event['terminated'], event['truncated'])
                assert replay.snapshot() == event['snapshot']
        finally:
            replay.env.close()
        assert verify_recording(recording)['video_frames'] == 10
    finally:
        lab.close()
    assert verify_artifacts(run) == result['verification']
    assert verify_artifacts(source) == before


def test_body_mode_needs_root_inputs_before_output_creation(tmp_path):
    with pytest.raises(ValueError, match='requires --root-view'):
        prepare_camera_body_lab(tmp_path/'source', tmp_path/'camera', tmp_path/'output', body_view='both')
    assert not (tmp_path/'output').exists()
