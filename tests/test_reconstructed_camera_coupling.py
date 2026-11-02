"""Saved reconstruction custody and native migration; no recorded physical truth."""

from copy import deepcopy
from hashlib import sha256
import json

import mujoco
import numpy as np
import pytest

from test_joint_pixel_trajectory import SETTINGS
from test_multiview_depth_branches import source_inputs
from test_shared_exercise_feet_native import local_native_only
from weft import camera_shared_candidate as candidate
from weft.camera_body import make_camera_body
from weft.camera_ground import apply_camera_ground
from weft.camera_shared_contract import BODY_RUNTIME_FILES, build_camera_coupling
from weft.camera_shared_task import CameraSharedTaskEnv
from weft.camera_shared_recording import recording_scene, verify_action_replay
from weft.human import HumanEnv, build_human
from weft.io import canonical_hash, digest, write_json
from weft.multiview_body import make_two_view_body
from weft.multiview_root import make_two_view_root
from weft.pipeline import verify_artifacts
from weft.recording import Recorder, export_recording
from weft.shared_exercise_lab import load_shared_exercise_lab, prepare_shared_exercise_lab, runtime_files
from weft.shared_world import SharedWorldEnv


def raw(value):
    return (json.dumps(value, sort_keys=True, allow_nan=False) + '\n').encode()


def package(path, files):
    path.mkdir()
    for name, payload in files.items():
        target = path/name; target.parent.mkdir(parents=True, exist_ok=True); target.write_bytes(payload)
    write_json(path/'manifest.json', {'files': {name: sha256(payload).hexdigest() for name, payload in files.items()}})
    return path


def source_files(args, body, audit):
    root, prediction, evidence, cameras = args
    binding = root['camera_reference']['root_refinement']
    report = {'method': root['camera_reference']['method'], 'source_frames': binding['source_frames'],
        'binding': binding, 'source_slots': [
            {'frame': row['frame'], 'time_s': row['time_s'], 'status': 'retained_body_pose',
             'hip_statuses': [row['joints'][i]['status'] for i in (23, 24)]} for row in prediction['frames']]}
    files = {name: raw({'authored_fixture': True}) for name in candidate.REQUIRED}
    files.update({'human.json': raw(body), 'ground.json': raw(body['camera_reference']['ground']),
        'two-view-triangulation.json': raw(prediction), 'two-view-root.json': raw(report),
        'source.mp4': b'Analytic geometry, not recorded physical evidence',
        'source.json': raw({'artifact': 'source.mp4', 'sha256': body['source_sha256'], 'rights': 'Authored analytical test bytes'})})
    for index, (view, ev, camera) in enumerate(zip(binding['views'], evidence, cameras, strict=True)):
        prefix = '' if index == 0 else 'inputs/two-view-root/' if index == 1 else f'inputs/two-view-root/extra-{index-2}/'
        files[prefix+'evidence.json'], files[prefix+'camera.json'] = raw(ev), raw(camera)
        if index == 1:
            config = {'schema_version': 1, 'first_name': binding['views'][0]['name'], 'second_name': view['name'],
                'clock_provenance': view['clock_provenance'], 'world_provenance': binding['world_provenance']}
        elif index > 1:
            config = {'schema_version': 1, 'name': view['name'], 'clock_provenance': view['clock_provenance']}
        else:
            continue
        files[prefix+'config.json'] = raw(config)
    if audit is not None:
        files['inputs/two-view-root-body.json'], files['two-view-body.json'] = raw(root), raw(audit)
        refinement = body['camera_reference']['body_refinement']
        if 'joint_pixel_fit' in refinement:
            files['joint-pixel-uncertainty.json'] = raw(refinement['joint_pixel_fit']['uncertainty'])
        if 'partial_pixel_uncertainty' in refinement:
            files['partial-pixel-uncertainty.json'] = raw(refinement['partial_pixel_uncertainty'])
    return files


@pytest.fixture(scope='module')
def inputs(tmp_path_factory, local_native_only):
    results = {}
    for count in (2, 3, 4):
        path = tmp_path_factory.mktemp(f'views-{count}')
        root, _, evidence, cameras = source_inputs(path, count)
        # The observation interchange also carries authored image-depth values; the ray fixture omits them.
        for stream in evidence:
            for row in stream['frames']:
                for point in row['landmarks']: point['z'] = 0.
        base = build_human(evidence[0], path/'observation-body', body_model='mobile')
        base, _ = make_camera_body(evidence[0], cameras[0], base)
        base = apply_camera_ground(base, root['camera_reference']['ground'])
        views = [{'name': view['name'], 'clock_provenance': view['clock_provenance'], 'evidence': ev, 'camera': cam}
                 for view, ev, cam in zip(root['camera_reference']['root_refinement']['views'], evidence, cameras, strict=True)]
        root, _, prediction = make_two_view_root(base, *views[:2], additional_views=views[2:],
            world_provenance='Known analytic shared world')
        results[count] = root, prediction, evidence, cameras
    return results


def construct(args, family):
    root, predictions, evidence, cameras = args
    if family == 'root':
        return root, None
    options = {'mode': family if family in ('directions', 'dimensions', 'both') else 'both'}
    if family in ('length', 'partial', 'pixels', 'uncertainty', 'branches', 'joint'):
        options['limb_fit'] = True
    if family in ('partial', 'pixels', 'uncertainty', 'branches', 'joint'):
        options['partial_segments'] = True
    if family in ('pixels', 'uncertainty', 'branches', 'joint'):
        options['pixel_cameras'] = cameras
        if family == 'joint':
            options['joint_pixel_uncertainty'] = SETTINGS
        else:
            options['partial_pixel_fit'] = True
    if family in ('uncertainty', 'branches'):
        from test_distal_pixel_uncertainty import SETTINGS as distal_settings
        options['pixel_uncertainty'] = distal_settings
    if family == 'branches':
        options['partial_depth_branches'] = True
    return make_two_view_body(root, predictions, *evidence[:2], additional_evidence=evidence[2:], **options)


@pytest.mark.parametrize('views', [2, 3, 4])
@pytest.mark.parametrize('family', ['root', 'directions', 'dimensions', 'both', 'length', 'partial', 'pixels', 'uncertainty', 'branches', 'joint'])
def test_exact_constructor_body_all_views_and_audit_survive_packaging(inputs, tmp_path, views, family):
    args = inputs[views]; body, audit = construct(args, family); before = canonical_hash(args)
    files = source_files(args, body, audit); source = package(tmp_path/'source', files)
    source_pin = verify_artifacts(source); out = tmp_path/'candidate'
    result = candidate.prepare_camera_shared_candidate(source, out)
    assert result['status'] == 'constructed', result
    assert result['files'] <= candidate.MAX_FILES and result['bytes'] <= candidate.MAX_PACKAGE_BYTES
    scene = json.loads((out/'scene.json').read_bytes())
    proof = json.loads((out/'source-verification.json').read_bytes())
    assert scene['human_spec'] == body and scene['camera_coupling'] == build_camera_coupling(body)
    assert scene['task']['human_limits']['max_floor_penetration_model_units'] == .025
    assert scene['dt'] == 1/args[2][0]['fps'] and not scene['learned_policy_available']
    assert len(proof['views']) == views and len(proof['source_slots']) == args[2][0]['frame_count']
    assert all(len(slot['views']) == views for slot in proof['source_slots'])
    assert all((out/name).read_bytes() == files[name] for name in candidate._extension_members(body))
    if family != 'root' or views > 2:
        assert scene['camera_coupling']['profile'] == 'camera-shared-coupling-v2'
        assert set(BODY_RUNTIME_FILES) <= set(runtime_files(scene))
        assert proof['body_reconstruction']['exact_body_preserved']
    else:
        assert scene['camera_coupling']['profile'] == 'camera-shared-coupling-v1'
    assert verify_artifacts(source) == source_pin and canonical_hash(args) == before
    verify_artifacts(out)


@pytest.fixture(scope='module')
def joint_input(inputs):
    body, audit = construct(inputs[3], 'joint')
    return body, source_files(inputs[3], body, audit)


@pytest.mark.parametrize('damage', ['camera', 'evidence', 'config', 'audit', 'parent', 'uncertainty', 'missing_view', 'missing_audit', 'angles', 'floor', 'forces'])
def test_resealed_changes_reject_without_compiling_or_rerunning_fit(joint_input, tmp_path, monkeypatch, damage):
    _, original = joint_input; files = dict(original)
    name = {'camera': 'inputs/two-view-root/extra-0/camera.json', 'evidence': 'inputs/two-view-root/extra-0/evidence.json',
        'config': 'inputs/two-view-root/extra-0/config.json', 'audit': 'two-view-body.json', 'parent': 'inputs/two-view-root-body.json',
        'uncertainty': 'joint-pixel-uncertainty.json', 'missing_view': 'inputs/two-view-root/extra-0/evidence.json',
        'missing_audit': 'two-view-body.json'}.get(damage, 'human.json')
    value = json.loads(files[name])
    if damage == 'camera': value['quality'] = 'measured'
    elif damage == 'evidence': value['frames'][0]['landmarks'][25]['x'] += .01
    elif damage == 'config': value['clock_provenance'] = 'Changed clock'
    elif damage == 'audit': value['physical_validation'] = 'verified'
    elif damage == 'parent': value['lengths']['shin'] += .01
    elif damage == 'uncertainty': value['image_std_fraction'] *= 2
    elif damage == 'angles': value['reference'][0]['joints']['left_knee'] += .01
    elif damage == 'floor': value['camera_reference']['ground']['height_m'] += .01
    elif damage == 'forces': value['motor_limits']['left_knee'] *= 2
    files[name] = raw(value)
    if damage.startswith('missing'): del files[name]
    def deny(*a, **kw): pytest.fail('Admission cannot run fitting or native models')
    monkeypatch.setattr(mujoco.MjModel, 'from_xml_string', deny)
    monkeypatch.setattr('weft.multiview_joint_pixels.fit_joint_trajectory', deny)
    source = package(tmp_path/'source', files); pin = digest(source/'manifest.json')
    result = candidate.prepare_camera_shared_candidate(source, tmp_path/'candidate')
    assert result['status'] == 'failed' and result['sealed'], result
    assert digest(source/'manifest.json') == pin


def test_native_joint_geometry_and_parameters_match_input_exactly(joint_input):
    body, _ = joint_input
    from test_camera_shared_world import camera_scene
    scene = camera_scene(body); shared, original = SharedWorldEnv(scene), HumanEnv(body)
    try:
        assert shared.human_spec == body and shared.human_mass == pytest.approx(float(original.model.body_mass.sum()), abs=1e-12)
        for name in shared.human_action_names:
            actual, expected = shared.model.joint(name).id, original.model.joint(name).id
            np.testing.assert_array_equal(shared.model.jnt_range[actual], original.model.jnt_range[expected])
        np.testing.assert_array_equal(shared.model.actuator_forcerange, original.model.actuator_forcerange)
        for index in range(len(body['reference'])):
            actual, expected = shared.reference_snapshot(index), original.reference_snapshot(index)
            for name, point in expected['human']['joints'].items():
                np.testing.assert_allclose(actual['human']['joints'][name], point, atol=1e-12)
    finally:
        shared.close(); original.close()


@pytest.mark.parametrize('assisted', [False, True])
def test_reconstructed_body_lab_task_record_export_and_exact_replay(joint_input, tmp_path, assisted):
    body, files = joint_input
    source = package(tmp_path/'source', files); out = tmp_path/'candidate'
    result = candidate.prepare_camera_shared_candidate(source, out)
    assert result['status'] == 'constructed', result
    lab = tmp_path/'lab'; prepare_shared_exercise_lab(out, lab, root_assistance=assisted)
    loaded = load_shared_exercise_lab(lab/'run'); scene, task = loaded['scene'], loaded['task']
    assert scene['human_spec'] == body and task['control']['root_assistance'] == ('explicit_3_axis' if assisted else 'disabled')
    assert 'reconstructed_shape' in task['scope']['body_scale']
    assert (lab/'run/two-view-body.json').read_bytes() == files['two-view-body.json']
    assert set(BODY_RUNTIME_FILES) <= set(loaded['contract']['runtime_sha256'])
    env = CameraSharedTaskEnv(scene, task); pin = verify_artifacts(lab/'run')
    try:
        env.reset(seed=11, options={'jitter': False})
        directory = lab/'sessions'/'manual-test'
        recorder = Recorder(directory, recording_scene(scene, task), profile='camera-shared')
        recorder.start(env.snapshot(), seed=11)
        for _ in range(3):
            action = np.zeros(env.action_space.shape); action[env.action_names.index('left_knee')] = .1
            _, reward, terminated, truncated, _ = env.step(action)
            recorder.append({'type': 'step', 'controller': 'manual', 'action': action.tolist(), 'snapshot': env.snapshot(),
                             'reward': reward, 'terminated': terminated, 'truncated': truncated})
        recorder.close()
        export_recording(directory)
        replay = verify_action_replay(directory)
        assert replay['steps'] == 3 and replay['exact_state_replay']
    finally: env.close()
    assert verify_artifacts(lab/'run') == pin


def test_missing_third_view_slot_is_retained_not_filled(inputs, tmp_path):
    root, _, evidence, cameras = deepcopy(inputs[3])
    evidence[2]['frames'].pop(2)
    evidence[2]['reference'].pop(2)
    # Rebuild only the controlled fixture from its original grounded monocular parent.
    base = build_human(evidence[0], tmp_path/'monocular', body_model='mobile')
    base, _ = make_camera_body(evidence[0], cameras[0], base)
    base = apply_camera_ground(base, root['camera_reference']['ground'])
    views = [{'name': view['name'], 'clock_provenance': view['clock_provenance'], 'evidence': ev, 'camera': cam}
             for view, ev, cam in zip(root['camera_reference']['root_refinement']['views'], evidence, cameras, strict=True)]
    root, _, prediction = make_two_view_root(base, *views[:2], additional_views=views[2:], world_provenance='Analytic world')
    args = root, prediction, evidence, cameras
    body, audit = construct(args, 'joint')
    source = package(tmp_path/'source', source_files(args, body, audit)); out = tmp_path/'candidate'
    result = candidate.prepare_camera_shared_candidate(source, out)
    assert result['status'] == 'constructed', result
    proof = json.loads((out/'source-verification.json').read_bytes())
    assert proof['views'][2]['unsampled_slots'] == 1
    assert proof['source_slots'][2]['views'][2] == {
        'source_evidence_id': None, 'pose_status': 'not_sampled', 'decoder_timestamp_s': None}
    assert len(proof['source_slots']) == 10 and len(body['reference']) == 10


def test_reconstructed_body_cannot_skip_initial_floor_gate(inputs):
    root, prediction, evidence, cameras = deepcopy(inputs[3])
    root['camera_reference']['ground']['height_m'] += 1.
    root['mobility']['base_height'] -= 1.
    body, _ = construct((root, prediction, evidence, cameras), 'joint')
    from test_camera_shared_world import camera_scene
    scene = camera_scene(body); pin = canonical_hash(scene); env = SharedWorldEnv(scene)
    try:
        with pytest.raises(ValueError, match='intersect'): env.reset(seed=7, options={'jitter': False})
    finally: env.close()
    assert canonical_hash(scene) == pin


@pytest.mark.parametrize('assisted', [False, True])
def test_reconstructed_learning_evaluation_lab_and_policy_replay(joint_input, tmp_path, assisted):
    from weft.camera_shared_learning import train_camera_shared_policy, evaluate_camera_shared_policy
    from weft.camera_shared_policy_recording import verify_action_replay as verify_policy_replay
    from weft.lab import Lab

    body, files = joint_input
    source = package(tmp_path/'source', files)
    candidate_path, untrained = tmp_path/'candidate', tmp_path/'untrained'
    assert candidate.prepare_camera_shared_candidate(source, candidate_path)['status'] == 'constructed'
    prepare_shared_exercise_lab(candidate_path, untrained, root_assistance=assisted)
    run = untrained/'run'
    source_pin = verify_artifacts(run)
    trained, evaluated, playable = (tmp_path/name for name in ('trained', 'evaluated', 'playable'))
    train_camera_shared_policy(run, trained, seed=26371, generations=1, population=4)
    comparison = evaluate_camera_shared_policy(trained, evaluated, seed=39127, trials=2)
    assert comparison['comparison_complete'] and comparison['independent_changed_states'] == 2
    assert set(comparison['controllers']) == {'coast', 'random', 'authored', 'learned'}
    for directory in (trained, evaluated):
        assert (directory/'scene.json').read_bytes() == (run/'scene.json').read_bytes()
        assert (directory/'exercise-task.json').read_bytes() == (run/'exercise-task.json').read_bytes()
    prepare_shared_exercise_lab(run, playable, policy_run=trained, evaluation_run=evaluated)
    loaded = load_shared_exercise_lab(playable/'run')
    admission = json.loads((playable/'run/camera-policy-admission.json').read_bytes())
    assert set(BODY_RUNTIME_FILES) <= admission['original_runtime_sha256'].keys()
    assert set(BODY_RUNTIME_FILES) <= admission['execution_runtime_sha256'].keys()
    assert loaded['scene']['human_spec'] == body and loaded['contract']['learned_policy_available']
    assert not loaded['scene']['learned_policy_available'] and loaded['task']['scope']['learning'] == 'unqualified'
    from weft.camera_shared_policy_receipt import validate_camera_policy_admission
    for name in BODY_RUNTIME_FILES:
        for key in ('original_runtime_sha256', 'execution_runtime_sha256'):
            missing = deepcopy(admission)
            del missing[key][name]
            with pytest.raises(ValueError, match='dependency closure'):
                validate_camera_policy_admission(missing, loaded['scene'], loaded['task'])
        changed = deepcopy(admission)
        changed['execution_runtime_sha256'][name] = '0'*64
        with pytest.raises(ValueError):
            validate_camera_policy_admission(changed, loaded['scene'], loaded['task'])
    lab = Lab(playable)
    try:
        session = lab.load_session()
        assert session.policy is not None
        lab.control('/api/record/start', {})
        lab.control('/api/step', {'policy': True, 'steps': 3})
        lab.control('/api/reset', {'seed': 29})
        lab.control('/api/step', {'controller': 'coast', 'steps': 1})
        stopped = lab.control('/api/record/stop', {})
        result = lab.export_future.result(timeout=30)
        assert result['recording_evidence']['status'] == 'available'
        replay = verify_policy_replay(playable/'sessions'/stopped['directory'])
        assert replay['exact_state_replay'] and replay['steps'] == 4 and replay['learned_actions_verified'] == 3
    finally:
        lab.close()
    assert verify_artifacts(run) == source_pin
