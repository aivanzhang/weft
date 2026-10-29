from copy import deepcopy
import json

import numpy as np
import pytest

from test_camera_body import source_run
from test_camera_ground import ground_for, references
from test_camera_pose import fixture
from weft.camera_body import make_camera_body
from weft.camera_body_lab import prepare_camera_body_lab
from weft.camera_ground import apply_camera_ground, audit_ground_clearance
from weft.foot_reference import build_foot_reference
from weft.human import HumanEnv, build_human, validate_human
from weft.io import canonical_hash, digest, write_json
from weft.lab import CameraHumanSession, Lab
from weft.mobile_human import calibrate_floor
from weft.multiview_root import make_two_view_root, shape_hash
from weft.pipeline import verify_artifacts
from weft.recording import load_recording, verify_recording
from weft.whole_body import WholeBodyEnv


WORLD = 'Analytic shared metre/z-up basis; not measured physical evidence'
CLOCK = 'Analytic simultaneous frames; no physical synchronization claim'


def inputs(tmp_path, *, bias=.8, missing=False):
    source, evidence, camera, camera_path, original = source_run(tmp_path, travel=.6, with_policy=False)
    for frame in evidence['frames']:
        for point in frame['world_landmarks']:
            for key in ('x', 'y', 'z'): point[key] *= bias
    if missing:
        evidence['frames'][6].update(status='no_pose', pose_count=0, landmarks=None, world_landmarks=None)
        evidence['frames'].pop(3)
        evidence['reference'] = [{'time_s': row['time_s'], 'evidence_id': row['id']}
                                 for row in evidence['frames'] if row['status'] == 'tracked']
    original = build_human(evidence, tmp_path/'new-body', body_model='mobile')
    base, _ = make_camera_body(evidence, camera, original)
    ground = ground_for(base, height=3.)
    base = apply_camera_ground(base, ground)
    xs = np.linspace(0, .6, 10)
    other, camera2, _, roots = fixture(roots=[[.2+x, .3+x/4, 5.] for x in xs],
                                     translations=[[-x-1, -x/4, 0] for x in xs])
    other['source_sha256'] = camera2['source_sha256'] = 'b'*64
    for frame in other['frames']: frame['id'] = f"{'b'*64}:frame:{frame['frame']}"
    first, second = ({'name': name, 'camera': cam, 'evidence': ev, 'clock_provenance': CLOCK}
                     for name, cam, ev in [('first', camera, evidence), ('second', camera2, other)])
    return base, first, second, roots, source, camera_path, ground


def test_root_recovers_metric_position_without_changing_shape_or_inputs(tmp_path):
    base, first, second, expected, *_ = inputs(tmp_path)
    before = canonical_hash([base, first, second])
    new, report, predictions = make_two_view_root(base, first, second, world_provenance=WORLD)
    assert canonical_hash([base, first, second]) == before
    assert shape_hash(new) == shape_hash(base)
    assert new['camera_reference']['ground'] == base['camera_reference']['ground']
    world = np.asarray([row['translation'] for row in new['root_reference']])+new['camera_reference']['origin_world_m']
    np.testing.assert_allclose(world, expected, atol=1e-10)
    assert not np.allclose(base['camera_reference']['origin_world_m'], expected[0])
    assert report['coverage'] == {'retained_body_pose': 10}
    assert predictions['monocular_world_landmarks_used'] is False
    for a, b, old_root, new_root in zip(references(base), references(new), base['root_reference'], new['root_reference'], strict=True):
        old_origin, new_origin = (np.asarray(s['camera_reference']['origin_world_m']) for s in (base, new))
        delta = np.asarray(new_root['translation'])+new_origin-old_root['translation']-old_origin
        for name in a['human']['joints']:
            actual = (np.asarray(b['human']['joints'][name])+[new_origin[0], new_origin[1], 3]
                      -np.asarray(a['human']['joints'][name])-[old_origin[0], old_origin[1], 3])
            np.testing.assert_allclose(actual, delta, atol=1e-10)
    old_gap, new_gap = (audit_ground_clearance(s) for s in (base, new))
    np.testing.assert_allclose([b['minimum_signed_clearance']-a['minimum_signed_clearance']
                               for a, b in zip(old_gap['checks'], new_gap['checks'], strict=True)], 1., atol=1e-10)
    env = HumanEnv(new)
    try:
        assert 'Two-view hip midpoint' in env.scene['human']['mobility']['root_depth']
        assert env.scene['learned_policy_available'] is False
    finally:
        env.close()


def test_missing_source_slots_are_not_filled_even_when_other_view_has_hips(tmp_path):
    base, first, second, *_ = inputs(tmp_path, missing=True)
    new, report, predictions = make_two_view_root(base, first, second, world_provenance=WORLD)
    assert len(new['reference']) == len(base['reference']) == 8
    assert report['source_frames'] == len(report['source_slots']) == len(predictions['frames']) == 10
    assert report['coverage'] == {'retained_body_pose': 8, 'primary_pose_not_sampled': 1, 'primary_body_pose_unavailable': 1}


@pytest.mark.parametrize('damage', ['hip_confidence', 'missing_frame', 'parallel', 'camera_hash', 'evidence_hash',
                                   'clock', 'world', 'same_source', 'fps'])
def test_bad_or_unavailable_views_fail_without_dropping_body_poses(tmp_path, damage):
    base, first, second, *_ = inputs(tmp_path)
    world = WORLD
    if damage == 'hip_confidence': second['evidence']['frames'][5]['landmarks'][23]['visibility'] = .1
    elif damage == 'missing_frame': second['evidence']['frames'].pop(5)
    elif damage == 'parallel':
        second['camera']['frames'] = deepcopy(first['camera']['frames'])
        for a, b in zip(first['evidence']['frames'], second['evidence']['frames'], strict=True): b['landmarks'] = a['landmarks']
    elif damage == 'camera_hash': first['camera']['provenance'] += ' changed'
    elif damage == 'evidence_hash': first['evidence']['frames'][0]['world_landmarks'][0]['x'] += .01
    elif damage == 'clock': second['clock_provenance'] = ''
    elif damage == 'world': world = ''
    elif damage == 'same_source': second = deepcopy(first)
    elif damage == 'fps': second['camera']['fps'] = 29
    with pytest.raises(ValueError): make_two_view_root(base, first, second, world_provenance=world)


@pytest.mark.parametrize('damage', ['shape', 'forces', 'origin', 'nested_root', 'hip', 'time', 'frame', 'ids',
                                   'hash', 'view', 'extra', 'missing', 'nan', 'schema', 'method'])
def test_refinement_binding_rejects_changed_geometry_and_provenance(tmp_path, damage):
    base, first, second, *_ = inputs(tmp_path)
    new, _, _ = make_two_view_root(base, first, second, world_provenance=WORLD)
    binding = new['camera_reference']; refinement = binding['root_refinement']
    if damage == 'shape': new['lengths']['thigh'] += .01
    elif damage == 'forces': new['mobility']['kp_per_mass'] += 1
    elif damage == 'origin': binding['origin_world_m'][0] += .01
    elif damage == 'nested_root': refinement['roots'][1]['root_world_m'][0] += .01
    elif damage == 'hip': refinement['roots'][1]['hip_world_m'][0][0] += .01
    elif damage == 'time': refinement['roots'][1]['time_s'] += .01
    elif damage == 'frame': refinement['roots'][1]['frame'] = True
    elif damage == 'ids': refinement['roots'][1]['source_evidence_ids'].reverse()
    elif damage == 'hash': refinement['triangulation_sha256'] = 'invalid'
    elif damage == 'view': refinement['views'][0]['camera_sha256'] = 'c'*64
    elif damage == 'extra': refinement['fitted_floor'] = True
    elif damage == 'missing': del binding['root_refinement']
    elif damage == 'nan': refinement['roots'][0]['hip_world_m'][0][0] = float('nan')
    elif damage == 'schema': refinement['schema_version'] = True
    elif damage == 'method': binding['method'] = 'camera-root-body-v1'
    with pytest.raises(ValueError): validate_human(new)


def test_existing_guards_reject_unsupported_learning_feet_and_repeated_fitting(tmp_path):
    base, first, second, *_ = inputs(tmp_path)
    new, _, _ = make_two_view_root(base, first, second, world_provenance=WORLD)
    with pytest.raises(ValueError, match='no repeated'): make_two_view_root(new, first, second, world_provenance=WORLD)
    with pytest.raises(ValueError, match='independent trajectory'): WholeBodyEnv(new)
    with pytest.raises(ValueError, match='transformed-foot'): build_foot_reference(new, {}, {})
    with pytest.raises(ValueError, match='must not be replaced'): calibrate_floor(new)


def package_inputs(tmp_path):
    base, first, second, _, source, camera_path, ground = inputs(tmp_path, bias=1.)
    # The package fixture reuses the untouched sealed evidence from source_run.
    ground_path = tmp_path/'ground.json'; write_json(ground_path, ground)
    write_json(tmp_path/'second-evidence.json', second['evidence'])
    write_json(tmp_path/'second-camera.json', second['camera'])
    config_path = tmp_path/'root-view.json'
    write_json(config_path, {'schema_version': 1, 'first_name': 'first', 'second_name': 'second',
        'second_evidence': 'second-evidence.json', 'second_camera': 'second-camera.json',
        'clock_provenance': CLOCK, 'world_provenance': WORLD})
    return source, camera_path, ground_path, config_path


def test_two_view_manual_lab_preserves_raw_estimates_and_records_exactly(tmp_path):
    source, camera, ground, config = package_inputs(tmp_path)
    before = verify_artifacts(source)
    out = tmp_path/'lab'
    result = prepare_camera_body_lab(source, camera, out, ground_path=ground, root_view_path=config)
    run = out/'run'
    body, original = (json.loads((run/name).read_text()) for name in ('human.json', 'inputs/monocular-camera-body.json'))
    assert shape_hash(body) == shape_hash(original)
    estimate = json.loads((run/'camera-reference.json').read_text())
    assert canonical_hash(json.loads((run/'two-view-triangulation.json').read_text())) == body['camera_reference']['root_refinement']['triangulation_sha256']
    assert estimate['frames'][0]['root_world_m'] == original['camera_reference']['origin_world_m']
    assert not (run/'policy.json').exists()
    lab = Lab(out)
    try:
        lab.load_session()
        with pytest.raises(ValueError, match='no learned policy'): lab.control('/api/step', {'policy': True})
        lab.control('/api/record/start', {})
        for action in (1, 2, 7, 9): lab.control('/api/step', {'action': action, 'steps': 2})
        lab.control('/api/reset', {'seed': 4})
        stopped = lab.control('/api/record/stop', {})
        lab.export_future.result(timeout=15)
        recording = out/'sessions'/stopped['directory']
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
    assert verify_artifacts(source) == before
    assert verify_artifacts(run) == result['verification']


@pytest.mark.parametrize('damage', ['no_ground', 'unknown_config', 'oversize', 'during_copy', 'sealed_output'])
def test_second_view_input_guards_prevent_ready_publication(tmp_path, monkeypatch, damage):
    import weft.camera_body_lab as module
    source, camera, ground, config = package_inputs(tmp_path)
    out = tmp_path/'lab'
    if damage == 'no_ground': ground = None
    elif damage == 'unknown_config':
        value = json.loads(config.read_text()); value['guess_clock'] = True; write_json(config, value)
    elif damage == 'oversize': config.write_bytes(b' '*(64*1024+1))
    elif damage == 'sealed_output':
        other = tmp_path/'sealed'; other.mkdir()
        config.rename(other/'config.json'); config = other/'config.json'
        write_json(other/'manifest.json', {'files': {}}); out = other/'lab'
    elif damage == 'during_copy':
        real = module.shutil.copyfile
        def mutate(src, dst):
            result = real(src, dst)
            if str(src).endswith('source.mp4'): (tmp_path/'second-evidence.json').write_text('{}')
            return result
        monkeypatch.setattr(module.shutil, 'copyfile', mutate)
    with pytest.raises((ValueError, RuntimeError)):
        prepare_camera_body_lab(source, camera, out, ground_path=ground, root_view_path=config)
    assert not (out/'status.json').exists()
