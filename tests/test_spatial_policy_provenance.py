"""Saved-state decision checks and mocked export wiring; no native replay or media work."""

from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest

from weft import lab, recording, recording_receipt as receipts, spatial_learning as learning
from weft import spatial_recording as proof, spatial_rigid as rigid
from weft.io import canonical_hash, digest, write_json
from weft.spatial_lab import authored_spatial_scene


@pytest.fixture(autouse=True)
def prohibit_native(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail('This suite must not execute native physics or real media work')

    monkeypatch.setattr(mujoco, 'MjModel', SimpleNamespace(from_xml_string=forbidden, from_xml_path=forbidden))
    for name in ('MjData', 'mj_step', 'mj_forward', 'mj_resetData', 'mju_quat2Mat'):
        monkeypatch.setattr(mujoco, name, forbidden)
    monkeypatch.setattr(rigid, 'SpatialRigidEnv', forbidden)
    # Every fixture quaternion is identity; no native rotation helper is needed.
    monkeypatch.setattr(rigid, 'rotation', lambda quaternion: np.eye(3))
    for name in ('VideoWriter', 'VideoCapture'):
        monkeypatch.setattr(recording.cv2, name, forbidden)


def encoded(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n').encode()


def policy_for(scene, version=2):
    return {'schema_version': version, 'profile': f'spatial-policy-v{version}', 'algorithm': f'spatial_cem_v{version}',
        'scene_sha256': canonical_hash(scene), 'controlled_id': scene['controlled_id'], 'controlled_index': 0,
        'object_ids': [obj['id'] for obj in scene['objects']], 'observation_dim': 13 * len(scene['objects']) + 4,
        'position_tolerance': scene['task']['position_tolerance'], 'speed_tolerance': scene['task']['speed_tolerance'],
        'weights': [1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.], 'physical_validation': 'unverified'}


def state(scene, steps=0, x=None):
    objects = [{key: deepcopy(value) for key, value in obj.items() if key != 'provenance'} for obj in scene['objects']]
    if x is not None:
        next(obj for obj in objects if obj['id'] == scene['controlled_id'])['position'][0] = x
    return {'profile': rigid.PROFILE, 'units': deepcopy(rigid.UNITS), 'bounds': scene['bounds'][:],
        'objects': objects, 'controlled_id': scene['controlled_id'], 'target': scene['target'][:],
        'steps': steps, 'time_s': steps * scene['dt'], 'success': False, 'terminated': False, 'truncated': False,
        'contacts': [], 'contact_details': [], 'contact_count': 0, 'substeps': 0, 'virtual_force': [0., 0., 0.],
        'physical_validation': 'unverified'}


def step(snapshot, action, controller, policy_raw):
    return {'type': 'step', 'action': action, 'reward': .125, 'terminated': snapshot['terminated'],
        'truncated': snapshot['truncated'], 'snapshot': snapshot,
        **({'controller': controller} if controller is not None else {}),
        **({'policy_sha256': sha256(policy_raw).hexdigest()} if controller == 'learned' else {})}


def journal(root, *, policy=True, selected=False, contract_version=2, policy_version=2):
    original = authored_spatial_scene()
    raw = encoded(policy_for(original, policy_version)) if policy else None
    scene = deepcopy(original)
    if selected:
        scene['controlled_id'] = 'companion'
        for obj in scene['objects']:
            if obj['role'] != 'static':
                obj['role'] = 'active' if obj['id'] == 'companion' else 'passive'
    recorder = recording.Recorder(root, scene, profile='spatial-rigid')
    if contract_version == 2:
        recorder.configure_spatial_controller_provenance(raw)
    mode = 'learned' if policy and not selected else 'manual'
    if contract_version == 1:
        mode = None
    recorder.start(state(scene), seed=17)
    recorder.append(step(state(scene, 1, .3), 5, 'manual' if contract_version == 2 else None, raw))
    recorder.append(step(state(scene, 2, .85), 2, mode, raw))
    recorder.append({'type': 'reset', 'seed': 31, 'snapshot': state(scene, 0, .85)})
    recorder.append(step(state(scene, 1, .78), 1, mode, raw))
    recorder.append(step(state(scene, 2, .7), 7, 'manual' if contract_version == 2 else None, raw))
    recorder.close()
    return original, scene, raw


def seal_selected(root):
    write_json(root / 'manifest.json', {'schema_version': 1,
        'files': {name: digest(root / name) for name in ('scene.json', 'recording.json', 'events.jsonl')}})
    return digest(root / 'manifest.json')


def documents(root):
    return json.loads((root / 'recording.json').read_bytes()), [json.loads(line) for line in (root / 'events.jsonl').read_bytes().splitlines()]


def rewrite(root, metadata, events):
    (root / 'events.jsonl').write_bytes(b''.join(json.dumps(event, allow_nan=False).encode() + b'\n' for event in events))
    metadata.update(events=len(events), journal_bytes=(root / 'events.jsonl').stat().st_size)
    write_json(root / 'recording.json', metadata)
    return seal_selected(root)


@pytest.mark.parametrize('policy_version', [1, 2])
def test_sealed_policy_decisions_preserve_modes_resets_exact_bytes_and_versions(tmp_path, policy_version):
    root = tmp_path / 'recording'
    _, _, raw = journal(root, policy_version=policy_version)
    pin = seal_selected(root)
    before = {path.name: path.read_bytes() for path in root.iterdir()}
    result = proof.verify_policy_decisions(root, expected_manifest_sha256=pin)
    assert result['status'] == 'passed'
    assert (result['events'], result['steps'], result['resets']) == (6, 4, 1)
    assert result['learned_actions_verified'] == result['manual_actions_recorded'] == 2
    assert result['policy_sha256'] == sha256(raw).hexdigest()
    assert result['manifest_sha256'] == pin and result['external_manifest_pin_supplied'] is True
    metadata, events = documents(root)
    contract = metadata['action_replay_contract']
    assert contract['learned_policy']['document_utf8'].encode() == raw
    assert set(contract['runtime_sha256']) == proof.POLICY_RUNTIME_FILES
    assert contract['runtime_versions']['numpy'] == np.__version__
    assert [event.get('controller') for event in events] == [None, 'manual', 'learned', None, 'learned', 'manual']
    assert events[3]['snapshot']['steps'] == 0
    assert before == {path.name: path.read_bytes() for path in root.iterdir()}


def test_observation_reconstruction_uses_pre_action_float32_not_post_action_or_float64(tmp_path):
    scene = authored_spatial_scene()
    policy = policy_for(scene); policy['weights'][-1] = 1.
    raw = encoded(policy)
    prior = state(scene, x=scene['target'][0] - .0200000001)
    observation = proof.observation_from_snapshot(scene, prior)
    assert observation.dtype == np.float32
    assert learning.policy_action(policy, observation) == 0
    wrong = observation.astype(float); wrong[-4] = .0200000001
    assert learning.policy_action(policy, wrong) == 2
    root = tmp_path / 'threshold'
    recorder = recording.Recorder(root, scene, profile='spatial-rigid')
    recorder.configure_spatial_controller_provenance(raw)
    recorder.start(prior, seed=0)
    recorder.append(step(state(scene, 1, .7), 0, 'learned', raw))
    recorder.close(); seal_selected(root)
    assert proof.verify_policy_decisions(root)['learned_actions_verified'] == 1


@pytest.mark.parametrize('policy,selected,count', [(False, False, None), (True, True, 0)])
def test_manual_only_and_alternate_actor_do_not_invent_learned_actions(tmp_path, policy, selected, count):
    root = tmp_path / 'recording'
    journal(root, policy=policy, selected=selected); seal_selected(root)
    result = proof.verify_policy_decisions(root)
    assert result['learned_actions_verified'] is count or result['learned_actions_verified'] == count
    assert result['manual_actions_recorded'] == 4


@pytest.mark.parametrize('kind', ['action', 'post_action', 'mode_missing', 'mode_unknown', 'manual_policy_hash',
    'policy_hash', 'policy_bytes', 'policy_version', 'actor', 'geometry', 'task', 'runtime', 'numpy', 'reset_state',
    'step_gap', 'terminal_without_reset', 'controller_on_reset', 'object_order', 'nonfinite', 'seal'])
def test_tampered_or_ambiguous_decision_inputs_are_rejected_offline(tmp_path, kind):
    root = tmp_path / 'recording'
    journal(root, selected=kind == 'actor')
    metadata, events = documents(root)
    contract = metadata['action_replay_contract']
    if kind == 'action': events[2]['action'] = 1
    elif kind == 'post_action': events[2]['action'] = 1
    elif kind == 'mode_missing': events[2].pop('controller')
    elif kind == 'mode_unknown': events[2]['controller'] = 'random'
    elif kind == 'manual_policy_hash': events[1]['policy_sha256'] = contract['learned_policy']['sha256']
    elif kind == 'policy_hash': events[2]['policy_sha256'] = '0' * 64
    elif kind == 'policy_bytes': contract['learned_policy']['document_utf8'] += ' '
    elif kind == 'policy_version':
        policy = json.loads(contract['learned_policy']['document_utf8']); policy['algorithm'] = 'spatial_cem_v99'
        contract['learned_policy'] = {'document_utf8': encoded(policy).decode(), 'sha256': sha256(encoded(policy)).hexdigest()}
    elif kind == 'actor':
        events[2].update(controller='learned', policy_sha256=contract['learned_policy']['sha256'])
    elif kind in {'geometry', 'task'}:
        scene = json.loads((root / 'scene.json').read_bytes())
        if kind == 'geometry': scene['objects'][0]['mass'] += .1
        else: scene['target'][0] += .01
        write_json(root / 'scene.json', scene)
        metadata['scene_sha256'] = contract['scene_sha256'] = canonical_hash(scene)
    elif kind == 'runtime': contract['runtime_sha256']['spatial_learning.py'] = '0' * 64
    elif kind == 'numpy': contract['runtime_versions']['numpy'] = 'changed'
    elif kind == 'reset_state': events[3]['snapshot'].update(steps=2, time_s=2 / 120)
    elif kind == 'step_gap': events[2]['snapshot'].update(steps=3, time_s=3 / 120)
    elif kind == 'terminal_without_reset':
        events[1]['terminated'] = events[1]['snapshot']['terminated'] = True
    elif kind == 'controller_on_reset': events[3]['controller'] = 'manual'
    elif kind == 'object_order': events[1]['snapshot']['objects'].reverse()
    elif kind == 'nonfinite':
        (root / 'events.jsonl').write_bytes((root / 'events.jsonl').read_bytes().replace(b'0.125', b'1e999', 1))
        metadata['journal_bytes'] = (root / 'events.jsonl').stat().st_size
        write_json(root / 'recording.json', metadata); seal_selected(root)
    if kind != 'nonfinite': rewrite(root, metadata, events)
    if kind == 'seal': (root / 'events.jsonl').write_bytes((root / 'events.jsonl').read_bytes() + b'\n')
    with pytest.raises((ValueError, OverflowError)):
        proof.verify_policy_decisions(root)


@pytest.mark.parametrize('kind', ['policy_bytes', 'metadata_bytes', 'event_count', 'line_bytes', 'journal_bytes', 'deadline', 'expected_seal'])
def test_decision_work_and_input_bounds(tmp_path, monkeypatch, kind):
    root = tmp_path / 'recording'
    if kind in {'policy_bytes', 'metadata_bytes'}:
        scene = authored_spatial_scene(); recorder = recording.Recorder(root, scene)
        if kind == 'policy_bytes':
            with pytest.raises(ValueError): recorder.configure_spatial_controller_provenance(b' ' * (proof.MAX_POLICY_BYTES + 1))
        else:
            policy = policy_for(scene); policy['scope'] = '\x01' * 10000
            raw = json.dumps(policy, ensure_ascii=False).encode()
            # Control escapes expand the saved metadata while the exact policy stays bounded.
            monkeypatch.setattr(proof, 'MAX_POLICY_BYTES', 65536)
            recorder.configure_spatial_controller_provenance(raw)
            with pytest.raises(ValueError, match='metadata'): recorder.start(state(scene))
        assert not root.exists()
        return
    journal(root); seal_selected(root)
    if kind == 'event_count': monkeypatch.setattr(proof, 'MAX_EVENTS', 3)
    elif kind == 'line_bytes': monkeypatch.setattr(proof, 'MAX_EVENT_BYTES', 20)
    elif kind == 'journal_bytes': monkeypatch.setattr(proof, 'MAX_JOURNAL_BYTES', 20)
    elif kind == 'deadline':
        ticks = iter([0., 10.]); monkeypatch.setattr(proof.time, 'monotonic', lambda: next(ticks, 10.))
    with pytest.raises((ValueError, TimeoutError)):
        proof.verify_policy_decisions(root, expected_manifest_sha256='0' * 64 if kind == 'expected_seal' else None)


def mock_replay_and_media(monkeypatch, events):
    calls = {'environments': 0, 'steps': 0, 'resets': 0, 'writes': 0}

    class Environment:
        def __init__(self, scene):
            calls['environments'] += 1; self.position = -1
        def reset(self, *, seed, options):
            self.position += 1; event = events[self.position]
            assert event['type'] in {'start', 'reset'} and seed == event['seed'] and options == event['reset_options']
            calls['resets'] += 1
        def step(self, action):
            self.position += 1; event = events[self.position]
            assert event['type'] == 'step' and action == event['action']; calls['steps'] += 1
            return None, event['reward'], event['terminated'], event['truncated'], {}
        def snapshot(self): return deepcopy(events[self.position]['snapshot'])
        def close(self): pass

    class Writer:
        def __init__(self, path, *args): Path(path).write_bytes(b'fake encoded media')
        def isOpened(self): return True
        def write(self, frame): calls['writes'] += 1
        def release(self): pass

    class Capture:
        def __init__(self, *args): self.remaining = len(events)
        def read(self): self.remaining -= 1; return self.remaining >= 0, None
        def release(self): pass

    class Image:
        def thumbnail(self, *args): pass
        def save(self, path, **kwargs): Path(path).write_bytes(b'fake preview')

    monkeypatch.setattr(rigid, 'SpatialRigidEnv', Environment)
    monkeypatch.setattr(recording.cv2, 'VideoWriter', Writer)
    monkeypatch.setattr(recording.cv2, 'VideoCapture', Capture)
    monkeypatch.setattr(recording.cv2, 'VideoWriter_fourcc', lambda *args: 0)
    monkeypatch.setattr(recording.cv2, 'cvtColor', lambda frame, code: frame)
    monkeypatch.setattr(recording.Image, 'fromarray', lambda frame: Image())
    monkeypatch.setattr(recording, 'render_snapshot', lambda *args: np.zeros((2, 2, 3), dtype=np.uint8))
    monkeypatch.setattr(recording, '_replay_html', lambda *args: '<html>Saved fixture states</html>')
    return calls


@pytest.mark.parametrize('mode', ['learned', 'alternate_actor', 'untrained', 'legacy_v1'])
def test_export_keeps_one_native_replay_and_six_files_with_bound_receipt(tmp_path, monkeypatch, mode):
    root = tmp_path / 'lab'; session = root / 'sessions/session-123456abcdef'
    original, scene, raw = journal(session, selected=mode == 'alternate_actor', policy=mode != 'untrained',
                                   contract_version=1 if mode == 'legacy_v1' else 2)
    run = root / 'run'; run.mkdir()
    write_json(run / 'scene.json', original)
    if raw is not None: (run / 'policy.json').write_bytes(raw)
    write_json(run / 'manifest.json', {'schema_version': 1, 'files': {p.name: digest(p) for p in run.iterdir()}})
    _, events = documents(session)
    calls = mock_replay_and_media(monkeypatch, events)
    result = receipts.export_recording_with_receipt(session, export=recording.export_recording, run_directory=run)
    assert calls == {'environments': 1, 'steps': 4, 'resets': 2, 'writes': 6}
    assert set(json.loads((session / 'manifest.json').read_bytes())['files']) == receipts.BASE_MEMBERS
    view = result['recording_evidence']; assert view['status'] == 'available', view
    expected = 2 if mode == 'learned' else 0 if mode == 'alternate_actor' else None
    assert view['replay']['learned_actions_verified'] == expected
    assert view['replay']['protocol'] == (proof.PROTOCOL if mode == 'legacy_v1' else proof.POLICY_PROTOCOL)
    assert view['identities']['policy_sha256'] == (sha256(raw).hexdigest() if mode in {'learned', 'alternate_actor'} else None)
    assert view['identities']['task_sha256'] is None
    before = {p: p.read_bytes() for p in root.rglob('*') if p.is_file()}
    monkeypatch.setattr(proof, 'verify_action_replay', lambda *a, **k: pytest.fail('Second native replay'))
    monkeypatch.setattr(proof, 'verify_policy_decisions', lambda *a, **k: pytest.fail('Receipt recomputed decisions'))
    monkeypatch.setattr(proof, 'runtime_identity', lambda *a, **k: pytest.fail('Legacy export used current runtime'))
    assert recording.export_recording(session)['action_replay'] == result['action_replay']
    saved = receipts.load_recording_evidence(root, session.name)
    assert saved['replay'] == view['replay'] and saved['identities'] == view['identities']
    assert not saved['inspection']['replay_performed'] and not saved['inspection']['journal_parsed']
    assert before == {p: p.read_bytes() for p in root.rglob('*') if p.is_file()}


@pytest.mark.parametrize('field', ['learned_count', 'policy_identity', 'decision_scope', 'decision_contract', 'decision_source'])
def test_saved_proof_and_receipt_cannot_inflate_decision_claims(tmp_path, monkeypatch, field):
    root = tmp_path / 'recording'; _, scene, _ = journal(root)
    metadata, events = documents(root); mock_replay_and_media(monkeypatch, events)
    result = proof.verify_action_replay(root)
    changed = deepcopy(result)
    if field == 'learned_count': changed['learned_actions_verified'] += 1
    elif field == 'policy_identity': changed['policy_decisions']['policy_sha256'] = '0' * 64
    elif field == 'decision_scope': changed['policy_decisions']['scope'] = 'Physical policy validation'
    elif field == 'decision_contract': changed['policy_decisions']['contract_sha256'] = '0' * 64
    else: changed['policy_decisions']['source_sha256']['events.jsonl'] = '0' * 64
    with pytest.raises(ValueError):
        proof.validate_replay_result(changed, metadata['action_replay_contract'], scene, result['source_sha256'],
                                     events=6, steps=4, resets=1)


def test_old_protocol_never_acquires_policy_claims(tmp_path):
    root = tmp_path / 'legacy'; journal(root, contract_version=1); seal_selected(root)
    result = proof.verify_policy_decisions(root)
    assert result['status'] == 'unavailable' and result['learned_actions_verified'] is None


def test_lab_records_manual_learned_reset_transitions_and_blocks_actor_changes(tmp_path, monkeypatch):
    scene = authored_spatial_scene(); raw = encoded(policy_for(scene)); current = state(scene)
    session = lab.SpatialSession.__new__(lab.SpatialSession)
    session.scene, session.policy_document, session.policy_sha256 = scene, raw, sha256(raw).hexdigest()
    session.done, session.failure = False, None
    session.snapshot = lambda: deepcopy(current)
    session.reset = lambda seed: current.update(state(scene)) or session.snapshot()
    session.policy_action = lambda: learning.policy_action(json.loads(raw), proof.observation_from_snapshot(scene, current))
    def advance(action):
        current.update(steps=current['steps'] + 1, time_s=(current['steps'] + 1) * scene['dt'])
        return .125, False, False, {}
    session.step = advance
    machine = lab.Lab.__new__(lab.Lab)
    machine.root, machine.seed, machine.config = tmp_path, 17, {'profile': 'spatial-rigid'}
    machine.recorder = machine.export_future = None
    machine.avatars = SimpleNamespace(packages={})
    machine.load_session = lambda: session
    machine.control('/api/record/start', {})
    for request in ({'steps': 1, 'action': 5}, {'steps': 1, 'policy': True}): machine.control('/api/step', request)
    with pytest.raises(ValueError, match='controlled object'): machine.control('/api/actor', {'actor': 'companion'})
    machine.control('/api/reset', {'seed': 31})
    machine.control('/api/step', {'steps': 1, 'policy': True})
    root = machine.recorder.root; machine.recorder.close(); seal_selected(root)
    result = proof.verify_policy_decisions(root)
    assert (result['learned_actions_verified'], result['manual_actions_recorded'], result['resets']) == (2, 1, 1)


def test_new_receipt_inspection_is_stdlib_only_in_a_fresh_process(tmp_path, monkeypatch):
    root = tmp_path / 'lab'; session = root / 'sessions/session-123456abcdef'
    original, _, raw = journal(session)
    run = root / 'run'; run.mkdir()
    write_json(run / 'scene.json', original); (run / 'policy.json').write_bytes(raw)
    write_json(run / 'manifest.json', {'schema_version': 1, 'files': {p.name: digest(p) for p in run.iterdir()}})
    mock_replay_and_media(monkeypatch, documents(session)[1])
    result = receipts.export_recording_with_receipt(session, export=recording.export_recording, run_directory=run)
    assert result['recording_evidence']['status'] == 'available'
    script = """
import importlib.abc, json, sys
class Guard(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'numpy','mujoco','gymnasium','cv2','PIL'} or fullname in {
            'weft.spatial_rigid','weft.spatial_learning','weft.lab','weft.recording'}:
            raise AssertionError('Receipt attempted non-stdlib runtime import: ' + fullname)
sys.meta_path.insert(0, Guard())
sys.path.insert(0, sys.argv[1])
from weft.recording_receipt import load_recording_evidence
result = load_recording_evidence(sys.argv[2], 'session-123456abcdef')
assert result['status'] == 'available', result
assert result['replay']['learned_actions_verified'] == 2
assert result['inspection']['journal_parsed'] is False
print(json.dumps({'status':result['status'],'replay':result['replay']}))
"""
    completed = subprocess.run([sys.executable, '-I', '-B', '-c', script,
        str(Path(recording.__file__).resolve().parent.parent), str(root)], capture_output=True, text=True, timeout=5)
    assert completed.returncode == 0, completed.stderr
    assert len(completed.stdout.encode()) <= 4096 and json.loads(completed.stdout)['status'] == 'available'
