from copy import deepcopy
from hashlib import sha256
import json

import numpy as np
import mujoco
import pytest

from weft.camera_body import ROOT_LIMIT, make_camera_body
from weft.camera_body_lab import prepare_camera_body_lab
from weft.foot_reference import build_foot_reference
from weft.human import HumanEnv, build_human, validate_human
from weft.io import canonical_hash, digest, write_json
from weft.lab import CameraHumanSession, Lab
from weft.pipeline import verify_artifacts
from weft.recording import load_recording, render_snapshot, verify_recording
from weft.whole_body import WholeBodyEnv
from test_camera_pose import fixture


def source_run(tmp_path, *, travel=18., with_policy=True, selected_joint='left_knee'):
    root = tmp_path/'source'; root.mkdir()
    video = root/'source.mp4'; video.write_bytes(b'Analytic source identity, not encoded physical video')
    source_hash = sha256(video.read_bytes()).hexdigest()
    displacement = np.linspace(0, travel, 10)
    roots = [[.2+x, .3+x/4, 5.] for x in displacement]
    evidence, camera, _, _ = fixture(roots=roots, translations=[[-x, -x/4, 0] for x in displacement])
    camera['source_sha256'] = evidence['source_sha256'] = source_hash
    for frame in evidence['frames']:
        frame['id'] = f"{source_hash}:frame:{frame['frame']}"
    evidence['reference'] = [{'time_s': f['time_s'], 'evidence_id': f['id']} for f in evidence['frames']]
    evidence['lengths'] = {'torso': .5, 'shoulder_width': .36, 'hip_width': .24,
                           'upper_arm': .3, 'forearm': .25, 'thigh': .45, 'shin': .45}
    original = build_human(evidence, root, body_model='mobile')
    write_json(root/'evidence.json', evidence)
    write_json(root/'pose-landmarks.json', {'frames': evidence['frames']})
    write_json(root/'source.json', {'sha256': source_hash, 'artifact': 'source.mp4', 'scope': 'Analytic fixture'})
    write_json(root/('policy.json' if with_policy else 'task-proposal.json'),
               {'selected_joint': selected_joint, 'scope': 'Selector metadata, not a trained policy'})
    write_json(root/'manifest.json', {'files': {p.name: digest(p) for p in root.iterdir() if p.is_file()}})
    camera_path = tmp_path/'camera.json'; write_json(camera_path, camera)
    return root, evidence, camera, camera_path, original


def test_full_camera_travel_uses_native_joints_and_preserves_original(tmp_path):
    _, evidence, camera, _, original = source_run(tmp_path)
    before = canonical_hash([original, evidence, camera])
    candidate, estimates = make_camera_body(evidence, camera, original)
    assert canonical_hash([original, evidence, camera]) == before
    assert candidate['lengths'] == original['lengths']
    assert len(candidate['reference']) == len(original['reference']) == 10
    assert candidate['mobility']['slide_limits'] == [-ROOT_LIMIT, ROOT_LIMIT]
    assert candidate['root_reference'][0]['translation'] == [0., 0., 0.]
    np.testing.assert_allclose(candidate['root_reference'][-1]['translation'], [18, 4.5, 0], atol=1e-10)
    env = HumanEnv(candidate)
    try:
        start = env.reference_snapshot(0); end = env.reference_snapshot(9)
        joints = [mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in ('root_x', 'root_y', 'root_z')]
        np.testing.assert_allclose(env.model.jnt_range[joints], [[-50, 50]]*3)
        np.testing.assert_allclose(np.array(end['human']['joints']['pelvis'])-start['human']['joints']['pelvis'], [18, 4.5, 0], atol=1e-10)
        assert end['human']['view_mode'] == 'body_follow'
        assert end['human']['view_origin'][0] == pytest.approx(18)
        assert end['human']['view_height'] < 5
        assert env.scene['learned_policy_available'] is False
        assert 'supplied cameras' in env.scene['human']['mobility']['root_depth']
        assert render_snapshot(end, env.scene).shape[0] > 0
    finally:
        env.close()
    assert estimates['coverage'] == {'estimated': 10}


@pytest.mark.parametrize('damage', ['root', 'rotation', 'angle', 'source', 'identities', 'quality', 'limits', 'missing_binding'])
def test_camera_model_binding_rejects_mismatched_or_forged_references(tmp_path, damage):
    _, evidence, camera, _, original = source_run(tmp_path)
    candidate, _ = make_camera_body(evidence, camera, original)
    if damage == 'root': candidate['root_reference'][1]['translation'][1] += .01
    elif damage == 'rotation': candidate['body_frame']['rotation'][0][0] += .01
    elif damage == 'angle': candidate['reference'][1]['joints']['left_knee'] += .01
    elif damage == 'source': candidate['source_sha256'] = 'b'*64
    elif damage == 'identities': candidate['camera_reference']['source_frame_ids'][1] = {}
    elif damage == 'quality': candidate['camera_reference']['calibration_quality'] = 'guaranteed'
    elif damage == 'limits': candidate['mobility']['slide_limits'] = [-100, 100]
    elif damage == 'missing_binding': del candidate['camera_reference']
    with pytest.raises(ValueError): validate_human(candidate)


def test_missing_camera_root_does_not_trim_original_reference(tmp_path):
    _, evidence, camera, _, original = source_run(tmp_path)
    evidence['frames'][5]['landmarks'][23]['visibility'] = .1
    with pytest.raises(ValueError, match='source frame 5'):
        make_camera_body(evidence, camera, original)


def test_travel_envelope_is_not_increased_to_match_input(tmp_path):
    _, evidence, camera, _, original = source_run(tmp_path, travel=51.)
    with pytest.raises(ValueError, match='50-unit'):
        make_camera_body(evidence, camera, original)


def test_legacy_learning_and_raw_camera_feet_do_not_silently_accept_world_candidate(tmp_path):
    _, evidence, camera, _, original = source_run(tmp_path)
    candidate, _ = make_camera_body(evidence, camera, original)
    with pytest.raises(ValueError, match='independent trajectory validation'):
        WholeBodyEnv(candidate)
    with pytest.raises(ValueError, match='transformed-foot adapter'):
        build_foot_reference(candidate, {}, {})
    with pytest.raises(ValueError, match='original horizontal-floor'):
        make_camera_body(evidence, camera, candidate)


@pytest.mark.parametrize('with_policy', [True, False])
def test_packaged_manual_lab_records_exactly_without_a_learned_policy(tmp_path, with_policy):
    source, _, _, camera_path, _ = source_run(tmp_path, with_policy=with_policy)
    before = verify_artifacts(source)
    out = tmp_path/'lab'
    result = prepare_camera_body_lab(source, camera_path, out)
    assert result['learned_policy_available'] is False
    assert not (out/'run/policy.json').exists()
    selector_file = 'policy.json' if with_policy else 'task-proposal.json'
    assert (out/'run/inputs/original'/selector_file).exists()
    if not with_policy:
        assert not (out/'run/inputs/original/policy.json').exists()
        assert 'no learned policy exists' in json.loads((out/'run/camera-provenance.json').read_text())['selected_joint_source']
    lab = Lab(out)
    try:
        assert isinstance(lab.load_session(), CameraHumanSession)
        assert 'reference_feasibility' in lab.summary()
        with pytest.raises(ValueError, match='no learned policy'):
            lab.control('/api/step', {'policy': True})
        lab.control('/api/record/start', {})
        for action in (1, 2, 7, 9):
            if lab.session.done: lab.control('/api/reset', {'seed': 4})
            lab.control('/api/step', {'action': action, 'steps': 3})
        lab.control('/api/reset', {'seed': 4})
        stopped = lab.control('/api/record/stop', {})
        exported = lab.export_future.result(timeout=15)
        recording = out/'sessions'/stopped['directory']
        data = load_recording(recording)
        assert exported['steps'] == 12
        replay = CameraHumanSession(out/'run')
        try:
            for event in data['events']:
                if event['type'] in ('start', 'reset'): replay.reset(event['seed'])
                else:
                    reward, terminated, truncated, _ = replay.step(event['action'])
                    assert (reward, terminated, truncated) == (event['reward'], event['terminated'], event['truncated'])
                assert replay.snapshot() == event['snapshot']
        finally:
            replay.env.close()
        assert verify_recording(recording)['video_frames'] == 14
    finally:
        lab.close()
    assert verify_artifacts(source) == before
    assert verify_artifacts(out/'run') == result['verification']


@pytest.mark.parametrize('selected_joint', [None, 'nonexistent_joint'])
def test_untrained_source_requires_a_valid_explicit_control_joint(tmp_path, selected_joint):
    source, _, _, camera_path, _ = source_run(tmp_path, with_policy=False, selected_joint=selected_joint)
    with pytest.raises(ValueError): prepare_camera_body_lab(source, camera_path, tmp_path/'lab')
    assert not (tmp_path/'lab').exists()


@pytest.mark.parametrize('damage', ['camera_source', 'output_inside', 'input_mutation'])
def test_packager_preserves_source_and_rejects_bad_inputs(tmp_path, monkeypatch, damage):
    import weft.camera_body_lab as module
    source, _, camera, camera_path, _ = source_run(tmp_path)
    out = tmp_path/'lab'
    if damage == 'camera_source':
        camera['source_sha256'] = 'b'*64; write_json(camera_path, camera)
    elif damage == 'output_inside': out = source/'candidate'
    else:
        original_verify = module.verify_artifacts
        def changed(path):
            result = original_verify(path)
            (source/'evidence.json').write_text('{}')
            return result
        monkeypatch.setattr(module, 'verify_artifacts', changed)
    with pytest.raises(ValueError): prepare_camera_body_lab(source, camera_path, out)
    assert not out.exists()
