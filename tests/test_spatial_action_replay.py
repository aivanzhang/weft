"""Small native journals test exact replay; receipt inspection remains metadata-only."""

from copy import deepcopy
import json
import shutil
import time
from types import SimpleNamespace

import pytest

from weft.io import digest, write_json
from weft.recording import Recorder, export_recording
from weft import recording_receipt, spatial_recording
from weft.spatial_rigid import SpatialRigidEnv
from test_spatial_rigid import scene


def record(root, *, selected=False, jitter=True, terminal=False):
    value = scene()
    value['objects'].insert(1, deepcopy(value['objects'][0]) | {
        'id': 'companion', 'label': 'Companion', 'role': 'passive', 'position': [.6, .2, .3]})
    if terminal:
        value['task']['max_steps'] = 10
    env = SpatialRigidEnv(value)
    source = deepcopy(env.scene)
    if selected:
        env.select_actor('companion')
    options = {'jitter': jitter, 'position_jitter': .01 if jitter else 0., 'velocity_jitter': .02 if jitter else 0.}
    recorder = Recorder(root, env.scene, profile='spatial-rigid')
    try:
        env.reset(seed=17, options=options)
        recorder.start(env.snapshot(), seed=17, reset_options=options)

        def step(action):
            _, reward, terminated, truncated, _ = env.step(action)
            recorder.append({'type': 'step', 'action': action, 'reward': reward, 'terminated': terminated,
                             'truncated': truncated, 'snapshot': env.snapshot()})

        for action in ([0] * 10 if terminal else [2, 3, 7]):
            step(action)
        env.reset(seed=31, options=options)
        recorder.append({'type': 'reset', 'seed': 31, 'reset_options': options, 'snapshot': env.snapshot()})
        for action in [1, 0]:
            step(action)
        recorder.close()
    finally:
        env.close()
    return root, source


def rewrite(root, events=None, metadata=None):
    if events is not None:
        (root / 'events.jsonl').write_text(''.join(json.dumps(e) + '\n' for e in events))
    meta = metadata or json.loads((root / 'recording.json').read_text())
    meta['journal_bytes'] = (root / 'events.jsonl').stat().st_size
    write_json(root / 'recording.json', meta)


@pytest.mark.parametrize('selected', [False, True])
def test_exact_actions_resets_actor_and_full_snapshots_without_input_mutation(tmp_path, selected):
    root, _ = record(tmp_path / 'recording', selected=selected)
    before = {p.name: digest(p) for p in root.iterdir()}
    result = spatial_recording.verify_action_replay(root)
    assert (result['events'], result['steps'], result['resets']) == (7, 5, 1)
    assert result['protocol'] == spatial_recording.PROTOCOL and result['exact_state_replay']
    assert result['controlled_id'] == ('companion' if selected else 'block')
    assert result['learned_actions_verified'] is None
    assert result['source_sha256'] == before
    assert set(result['runtime_sha256']) == spatial_recording.RUNTIME_FILES
    assert before == {p.name: digest(p) for p in root.iterdir()}


@pytest.mark.parametrize('change,match', [
    ('action', 'replay differs'), ('reward', 'outcome replay differs'), ('flag', 'outcome replay differs'),
    ('snapshot', 'full-state replay differs'), ('reset_seed', 'full-state replay differs'),
    ('reset_options', 'full-state replay differs'), ('index', 'sequence'), ('start_seed', 'seed differs'),
    ('nan', 'JSON compliant'), ('actor', 'task and evidence scope'), ('extra_snapshot', 'full-state replay differs'),
])
def test_changed_recorded_values_cannot_pass_exact_replay(tmp_path, change, match):
    root, _ = record(tmp_path / 'recording')
    events = [json.loads(line) for line in (root / 'events.jsonl').read_text().splitlines()]
    if change == 'action': events[1]['action'] = 3
    elif change == 'reward': events[1]['reward'] += .01
    elif change == 'flag': events[1]['terminated'] = True
    elif change == 'snapshot': events[1]['snapshot']['objects'][0]['position'][0] += .0001
    elif change == 'reset_seed': events[4]['seed'] = 32
    elif change == 'reset_options': events[4]['reset_options'] = spatial_recording.reset_options()
    elif change == 'index': events[2]['index'] = 3
    elif change == 'start_seed': events[0]['seed'] = 19
    elif change == 'nan': events[1]['reward'] = float('nan')
    elif change == 'actor': events[1]['snapshot']['controlled_id'] = 'companion'
    elif change == 'extra_snapshot': events[1]['snapshot']['unrecorded_change'] = True
    rewrite(root, events)
    with pytest.raises(ValueError, match=match): spatial_recording.verify_action_replay(root)


@pytest.mark.parametrize('field', ['runtime_sha256', 'runtime_versions', 'missing_contract', 'schema_version'])
def test_unknown_recording_time_runtime_is_rejected_before_native_construction(tmp_path, monkeypatch, field):
    root, _ = record(tmp_path / 'recording')
    metadata = json.loads((root / 'recording.json').read_text())
    if field == 'missing_contract': metadata.pop('action_replay_contract')
    elif field == 'schema_version': metadata[field] = 2
    elif field == 'runtime_sha256': metadata['action_replay_contract'][field]['spatial_rigid.py'] = '0' * 64
    else: metadata['action_replay_contract'][field]['mujoco'] = 'unknown-old-runtime'
    rewrite(root, metadata=metadata)
    monkeypatch.setattr(SpatialRigidEnv, '__init__', lambda *_: pytest.fail('Unpinned native construction'))
    with pytest.raises(ValueError, match='runtime|start-time|closed, scene-bound'): spatial_recording.verify_action_replay(root)


def test_bounds_stop_before_native_construction(tmp_path, monkeypatch):
    root, _ = record(tmp_path / 'recording')
    monkeypatch.setattr(SpatialRigidEnv, '__init__', lambda *_: pytest.fail('Construction beyond replay bound'))
    with pytest.raises(ValueError, match='step bound'): spatial_recording.verify_action_replay(root, max_steps=4)
    with pytest.raises(TimeoutError): spatial_recording.verify_action_replay(root, deadline=time.monotonic() - 1)


def test_deadline_during_replay_stops_next_action_and_closes_environment(tmp_path, monkeypatch):
    root, _ = record(tmp_path / 'recording')
    now, actions, closed = 100., 0, []
    original_step, original_close = SpatialRigidEnv.step, SpatialRigidEnv.close

    def step(env, action):
        nonlocal now, actions
        result = original_step(env, action)
        actions += 1; now = 102.
        return result

    def close(env):
        closed.append(True)
        return original_close(env)

    monkeypatch.setattr(spatial_recording, 'time', SimpleNamespace(monotonic=lambda: now))
    monkeypatch.setattr(SpatialRigidEnv, 'step', step)
    monkeypatch.setattr(SpatialRigidEnv, 'close', close)
    with pytest.raises(TimeoutError): spatial_recording.verify_action_replay(root, max_seconds=1)
    assert actions == 1 and closed == [True]


def test_last_validation_deadline_prevents_native_construction(tmp_path, monkeypatch):
    from weft import spatial_rigid
    from weft.io import canonical_hash

    root = tmp_path / 'recording'; root.mkdir()
    value = scene(); now = 100.
    event = {'index': 0, 'type': 'start', 'seed': 17, 'reset_options': spatial_recording.reset_options(), 'snapshot': {}}
    journal = (json.dumps(event) + '\n').encode()
    write_json(root / 'scene.json', value)
    (root / 'events.jsonl').write_bytes(journal)
    write_json(root / 'recording.json', {'schema_version': 1, 'status': 'closed', 'scene_sha256': canonical_hash(value),
        'seed': 17, 'events': 1, 'journal_bytes': len(journal), 'action_replay_contract': spatial_recording.recording_contract(value)})

    def expire(*_args):
        nonlocal now
        now = 102.

    monkeypatch.setattr(spatial_recording, 'time', SimpleNamespace(monotonic=lambda: now))
    monkeypatch.setattr(spatial_rigid, 'validate_scene', lambda scene: scene)
    monkeypatch.setattr(spatial_rigid, 'validate_snapshot', expire)
    monkeypatch.setattr(SpatialRigidEnv, '__init__', lambda *_a, **_k: pytest.fail('Construction after validation deadline'))
    with pytest.raises(TimeoutError): spatial_recording.verify_action_replay(root, max_seconds=1)


def test_terminal_episode_requires_reset_and_reset_replays(tmp_path):
    root, _ = record(tmp_path / 'recording', terminal=True)
    assert spatial_recording.verify_action_replay(root)['steps'] == 12
    events = [json.loads(line) for line in (root / 'events.jsonl').read_text().splitlines()]
    events.insert(11, deepcopy(events[10]))
    for i, event in enumerate(events): event['index'] = i
    metadata = json.loads((root / 'recording.json').read_text())
    metadata['events'] = len(events)
    rewrite(root, events, metadata)
    with pytest.raises(RuntimeError, match='after an episode ends'): spatial_recording.verify_action_replay(root)


def test_replay_rechecks_runtime_and_source_bytes_after_execution(tmp_path, monkeypatch):
    root, _ = record(tmp_path / 'recording')
    original, calls = spatial_recording.runtime_identity, 0

    def change_runtime():
        nonlocal calls
        calls += 1
        identity = original()
        if calls > 1: identity['runtime_sha256']['spatial_rigid.py'] = '0' * 64
        return identity

    monkeypatch.setattr(spatial_recording, 'runtime_identity', change_runtime)
    with pytest.raises(ValueError, match='runtime changed during'): spatial_recording.verify_action_replay(root)
    monkeypatch.setattr(spatial_recording, 'runtime_identity', original)
    original_step = SpatialRigidEnv.step

    def change_journal(env, action):
        result = original_step(env, action)
        with (root / 'events.jsonl').open('ab') as stream: stream.write(b' ')
        return result

    monkeypatch.setattr(SpatialRigidEnv, 'step', change_journal)
    with pytest.raises(ValueError, match='pinned SHA256'): spatial_recording.verify_action_replay(root)


@pytest.fixture(scope='module')
def exported_lab(tmp_path_factory):
    lab = tmp_path_factory.mktemp('spatial-exact-export') / 'lab'
    session = lab / 'sessions' / 'session-123456abcdef'
    root, source = record(session, selected=True, jitter=False)
    run = lab / 'run'; run.mkdir()
    write_json(run / 'scene.json', source)
    write_json(run / 'manifest.json', {'schema_version': 1, 'files': {'scene.json': digest(run / 'scene.json')}})
    result = recording_receipt.export_recording_with_receipt(root, export=export_recording, run_directory=run)
    assert result['recording_evidence']['status'] == 'available', result['recording_evidence']
    return lab, result


def test_new_export_seals_six_artifacts_and_bound_spatial_receipt(exported_lab, monkeypatch):
    lab, result = exported_lab
    root = lab / 'sessions' / 'session-123456abcdef'
    manifest = json.loads((root / 'manifest.json').read_text())
    assert len(manifest['files']) == result['verified_files'] == 6
    assert manifest['action_replay'] == result['action_replay']
    expected = {'status': 'passed', 'protocol': spatial_recording.PROTOCOL, 'events_checked': 7,
                'actions_checked': 5, 'resets_checked': 1, 'learned_actions_verified': None}
    assert result['recording_evidence']['replay'] == expected
    before = {str(p.relative_to(lab)): digest(p) for p in lab.rglob('*') if p.is_file()}
    monkeypatch.setattr(spatial_recording, 'verify_action_replay', lambda *_a, **_k: pytest.fail('Replayed existing export'))
    monkeypatch.setattr(SpatialRigidEnv, '__init__', lambda *_a, **_k: pytest.fail('Native inspection'))
    assert export_recording(root)['action_replay'] == result['action_replay']
    view = recording_receipt.load_recording_evidence(lab, root.name)
    assert view['replay'] == expected
    assert view['identities']['engine_scene_sha256'] != view['identities']['recording_scene_sha256']
    assert view['identities']['policy_sha256'] is None
    assert view['inspection']['replay_performed'] is False and view['inspection']['journal_parsed'] is False
    assert before == {str(p.relative_to(lab)): digest(p) for p in lab.rglob('*') if p.is_file()}


@pytest.mark.parametrize('change', ['geometry', 'task', 'role', 'proof', 'runtime', 'policy_claim'])
def test_spatial_receipt_rejects_resealed_identity_and_proof_tampering(exported_lab, tmp_path, change):
    original, _ = exported_lab; lab = tmp_path / 'lab'; shutil.copytree(original, lab)
    root = lab / 'sessions' / 'session-123456abcdef'
    if change in {'geometry', 'task', 'role'}:
        scene_path = lab / 'run' / 'scene.json'; value = json.loads(scene_path.read_text())
        if change == 'geometry': value['objects'][0]['position'][0] += .01
        elif change == 'task': value['task']['speed_tolerance'] += .01
        else: value['objects'][1]['role'] = 'static'
        write_json(scene_path, value)
        write_json(lab / 'run' / 'manifest.json', {'schema_version': 1, 'files': {'scene.json': digest(scene_path)}})
    else:
        path = root / 'evidence' / 'receipt.json'; receipt = json.loads(path.read_text())
        if change == 'proof': receipt['replay']['actions_checked'] = 4
        elif change == 'runtime': receipt['runtime']['native_sha256']['spatial_rigid.py'] = '0' * 64
        else: receipt['replay']['learned_actions_verified'] = 5
        write_json(path, receipt)
        seal_path = root / 'evidence' / 'manifest.json'; seal = json.loads(seal_path.read_text())
        seal['files']['receipt.json'] = digest(path); write_json(seal_path, seal)
    assert recording_receipt.load_recording_evidence(lab, root.name)['status'] == 'unavailable'


def test_legacy_export_never_gains_runtime_or_exact_replay(exported_lab, tmp_path, monkeypatch):
    original, _ = exported_lab; lab = tmp_path / 'lab'; shutil.copytree(original, lab)
    root = lab / 'sessions' / 'session-123456abcdef'
    metadata = json.loads((root / 'recording.json').read_text()); metadata.pop('action_replay_contract')
    events = [json.loads(line) for line in (root / 'events.jsonl').read_text().splitlines()]
    for event in events: event.pop('reset_options', None)
    rewrite(root, events, metadata)
    manifest = json.loads((root / 'manifest.json').read_text()); manifest.pop('action_replay')
    for name in manifest['files']: manifest['files'][name] = digest(root / name)
    write_json(root / 'manifest.json', manifest)
    before = {p.name: digest(p) for p in root.iterdir() if p.is_file()}
    monkeypatch.setattr(spatial_recording, 'verify_action_replay', lambda *_a, **_k: pytest.fail('Replayed legacy export'))
    monkeypatch.setattr(spatial_recording, 'runtime_identity', lambda: pytest.fail('Backfilled unknown runtime'))
    result = recording_receipt.export_recording_with_receipt(root, export=export_recording, run_directory=lab / 'run')
    assert 'action_replay' not in result and result['recording_evidence']['code'] == 'legacy_not_backfilled'
    assert before == {p.name: digest(p) for p in root.iterdir() if p.is_file()}


def test_unsealed_legacy_journal_remains_state_only_without_runtime_backfill(tmp_path, monkeypatch):
    from weft import recording

    root, _ = record(tmp_path / 'recording')
    metadata = json.loads((root / 'recording.json').read_text()); metadata.pop('action_replay_contract')
    events = [json.loads(line) for line in (root / 'events.jsonl').read_text().splitlines()]
    for event in events: event.pop('reset_options', None)
    rewrite(root, events, metadata)
    before = {p.name: digest(p) for p in root.iterdir()}
    monkeypatch.setattr(spatial_recording, 'verify_action_replay', lambda *_a, **_k: pytest.fail('Replayed old journal'))
    monkeypatch.setattr(spatial_recording, 'runtime_identity', lambda: pytest.fail('Invented recording-time runtime'))

    def stop_before_media(*_args, **_kwargs):
        raise RuntimeError('State-only render reached without replay')

    monkeypatch.setattr(recording, 'render_snapshot', stop_before_media)
    with pytest.raises(RuntimeError, match='State-only render reached'): export_recording(root)
    assert before == {p.name: digest(p) for p in root.iterdir()}
