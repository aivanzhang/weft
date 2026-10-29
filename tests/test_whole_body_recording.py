"""Vector-action journals preserve direct commands, not implied automatic assistance."""

from copy import deepcopy
import json
import re

import numpy as np
import pytest

from weft.recording import Recorder, export_recording, load_recording, render_snapshot
from test_mobile_recording import mobile_fixture, event


def whole_body_fixture():
    scene, state = mobile_fixture()
    names = list(state['human']['angles']) + ['root_x', 'root_y', 'root_z']
    units = ['Nm'] * 19 + ['N'] * 3
    scene.update(profile='exercise-imitation-v1', task={'type': 'whole_body_reference_tracking'},
                 action_contract={'names': names, 'normalized_bounds': [-1., 1.],
                                  'physical_limits': [30.] * 19 + [200.] * 3, 'units': units,
                                  'substeps_per_action': 8, 'hidden_assistance': False})
    state['human'].update(control_mode='direct_motor_actions', action_names=names,
                          normalized_action=[0.] * 22, motor_forces=[0.] * 22, motor_units=units,
                          view_height=30., view_origin=[0., 0., 1.])
    state['human']['mobility'].update(support_enabled=False, support_force=[0., 0., 0.])
    return scene, state


def test_vectors_states_contacts_and_replay_roundtrip_without_support_mislabel(tmp_path, monkeypatch):
    scene, state = whole_body_fixture()
    recorder = Recorder(tmp_path / 'whole', scene)
    recorder.start(state, seed=12)
    action = [-1., 1., .123456789012345] + [0.] * 16 + [.1, -.05, .75]
    state['human']['normalized_action'] = action.copy()
    state['human']['motor_forces'] = [a * limit for a, limit in zip(action, scene['action_contract']['physical_limits'])]
    state['human']['mobility']['support_force'] = state['human']['motor_forces'][-3:]
    recorder.append(event(state, action))
    recorder.append({'type': 'reset', 'seed': 3, 'snapshot': whole_body_fixture()[1]})
    recorder.close()
    loaded = load_recording(recorder.root)
    assert loaded['events'][1]['action'] == action
    assert loaded['events'][1]['snapshot'] == state
    assert len(state['human']['angles']) == 19 and len(state['human']['muscles']) == 38
    assert not state['human']['mobility']['support_enabled']
    assert state['human']['mobility']['support_force'] == [20., -10., 150.]
    result = export_recording(recorder.root)
    assert result['video_frames'] == 3 and result['resets'] == 1
    html = (recorder.root / 'replay.html').read_text()
    payload = json.loads(re.search(r'<script id="data" type="application/json">(.*?)</script>', html, re.S).group(1))
    assert payload['events'][1]['action'] == action
    assert payload['events'][1]['snapshot'] == state
    assert payload['scene']['action_contract']['names'] == scene['action_contract']['names']
    assert 'DIRECT MOTOR ACTIONS · VIRTUAL ROOT FORCES · NO AUTOMATIC ASSISTANCE' in html
    assert 'virtual root force [' in html and 'contact.position' in html and 'h.mobility?.floor_height' in html
    # Inspect labels drawn into video frames without brittle OCR or pixel-font assumptions.
    from PIL import ImageDraw
    original = ImageDraw.ImageDraw.text
    labels = []
    def capture(draw, xy, text, *args, **kwargs):
        labels.append(text)
        return original(draw, xy, text, *args, **kwargs)
    monkeypatch.setattr(ImageDraw.ImageDraw, 'text', capture)
    render_snapshot(state, scene)
    assert any('DIRECT MOTOR ACTIONS' in text and 'VIRTUAL ROOT FORCES' in text for text in labels)
    assert not any('SUPPORT OFF' in text for text in labels)


@pytest.mark.parametrize('action', [0, True, [0.] * 21, [0.] * 23, [[0.]] * 22,
    [False] + [0.] * 21, [float('nan')] + [0.] * 21, [float('inf')] + [0.] * 21,
    [1.00001] + [0.] * 21, [-1.00001] + [0.] * 21, ['0'] + [0.] * 21])
def test_invalid_vector_actions_leave_existing_journal_unchanged(tmp_path, action):
    scene, state = whole_body_fixture()
    recorder = Recorder(tmp_path / 'invalid-action', scene)
    recorder.start(state)
    before = (recorder.root / 'events.jsonl').read_bytes()
    with pytest.raises(ValueError, match='22 finite bounded'):
        recorder.append(event(state, action))
    assert (recorder.root / 'events.jsonl').read_bytes() == before
    recorder.close()


@pytest.mark.parametrize('mutate', [
    lambda s: s.pop('human'),
    lambda s: s['human'].pop('mobility'),
    lambda s: s['human'].pop('control_mode'),
    lambda s: s['human'].update(control_mode='reference_servo'),
    lambda s: s['human']['angles'].pop('joint-18'),
    lambda s: s['human']['muscles'].pop(),
    lambda s: s['human'].update(action_names=['wrong'] * 22),
    lambda s: s['human'].update(normalized_action=[True] * 22),
    lambda s: s['human'].update(motor_forces=[0.] * 21),
    lambda s: s['human'].update(motor_units=['N'] * 22),
    lambda s: s['human'].update(view_height=30.01),
    lambda s: s['human'].update(view_origin=[101., 0., 0.]),
])
def test_whole_body_requires_complete_bounded_human_state(tmp_path, mutate):
    scene, state = whole_body_fixture()
    mutate(state)
    with pytest.raises(ValueError):
        Recorder(tmp_path / 'invalid-state', scene).start(state)
    assert not (tmp_path / 'invalid-state').exists()
    with pytest.raises(ValueError):
        render_snapshot(state, scene)


@pytest.mark.parametrize('mutate', [
    lambda s: s.pop('action_contract'),
    lambda s: s['action_contract'].update(names=['same'] * 22),
    lambda s: s['action_contract'].update(names=['one']),
    lambda s: s['action_contract'].update(normalized_bounds=[0., 1.]),
    lambda s: s['action_contract'].update(normalized_bounds=[-1., True]),
    lambda s: s['action_contract'].pop('physical_limits'),
    lambda s: s['action_contract'].update(physical_limits=[30.] * 21),
    lambda s: s['action_contract'].update(physical_limits=[True] + [30.] * 21),
    lambda s: s['action_contract'].update(physical_limits=[float('nan')] + [30.] * 21),
    lambda s: s['action_contract'].update(physical_limits=[0.] + [30.] * 21),
    lambda s: s['action_contract'].update(physical_limits=[-1.] + [30.] * 21),
    lambda s: s['action_contract'].update(physical_limits=[120.01] + [30.] * 21),
    lambda s: s['action_contract'].update(physical_limits=[30.] * 21 + [2000.01]),
    lambda s: s['action_contract'].update(units=['N'] * 22),
    lambda s: s['action_contract'].update(substeps_per_action=True),
    lambda s: s['action_contract'].update(substeps_per_action=8.),
    lambda s: s['action_contract'].update(substeps_per_action=7),
    lambda s: s['action_contract'].update(hidden_assistance=True),
    lambda s: s['action_contract'].update(hidden_assistance=0),
    lambda s: s.update(task=None),
    lambda s: s['task'].update(type='selected_joint_tracking'),
])
def test_whole_body_scene_contract_must_preserve_action_meaning(tmp_path, mutate):
    scene, _ = whole_body_fixture()
    mutate(scene)
    with pytest.raises(ValueError):
        Recorder(tmp_path / 'invalid-contract', scene)


def test_load_rejects_malformed_vector_even_if_json_is_valid(tmp_path):
    scene, state = whole_body_fixture()
    recorder = Recorder(tmp_path / 'load-validation', scene)
    recorder.start(state)
    recorder.append(event(state, [0.] * 22))
    recorder.close()
    path = recorder.root / 'events.jsonl'
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[1]['action'] = [True] + [0.] * 21
    path.write_text(''.join(json.dumps(row) + '\n' for row in rows))
    with pytest.raises(ValueError, match='22 finite bounded'):
        load_recording(recorder.root)


def test_actual_whole_body_env_snapshot_and_action_export(tmp_path):
    pytest.importorskip('mujoco')
    from weft.human import build_human
    from weft.whole_body import WholeBodyEnv
    from test_mobile_integration import evidence
    spec = build_human(evidence(), tmp_path / 'model', body_model='mobile')
    env = WholeBodyEnv(spec, horizon_steps=2)
    try:
        env.reset(seed=4)
        recorder = Recorder(tmp_path / 'actual-env', env.scene)
        recorder.start(env.snapshot(), seed=4)
        action = np.zeros(22)
        action[-1] = .25
        _, reward, terminated, truncated, _ = env.step(action)
        state = env.snapshot()
        recorder.append(dict(type='step', action=action.tolist(), reward=reward,
                             terminated=terminated, truncated=truncated, snapshot=state))
        recorder.close()
        assert 'selected_joint' not in state['human']
        assert state['human']['mobility']['support_force'][2] != 0
        assert load_recording(recorder.root)['events'][1]['snapshot'] == state
        assert export_recording(recorder.root)['video_frames'] == 2
    finally:
        env.close()


@pytest.mark.parametrize(('mutate', 'message'), [
    (lambda row: row.update(action=[.5] + [0.] * 21), 'exactly match'),
    (lambda row: row['snapshot']['human']['motor_forces'].__setitem__(0, 1e-8), 'times physical limits'),
    (lambda row: row['snapshot']['human']['mobility']['support_force'].__setitem__(2, 1e-8), 'last three'),
    (lambda row: row['snapshot']['human']['mobility'].update(support_enabled=True), 'automatic support'),
])
def test_inconsistent_commands_rejected_on_append_and_load(tmp_path, mutate, message):
    scene, state = whole_body_fixture()
    recorder = Recorder(tmp_path / 'inconsistent', scene)
    recorder.start(state)
    valid = event(state, [0.] * 22)
    invalid = deepcopy(valid)
    mutate(invalid)
    before = (recorder.root / 'events.jsonl').read_bytes()
    with pytest.raises(ValueError, match=message):
        recorder.append(invalid)
    assert (recorder.root / 'events.jsonl').read_bytes() == before
    recorder.append(valid)
    recorder.close()
    journal = recorder.root / 'events.jsonl'
    rows = [json.loads(line) for line in journal.read_text().splitlines()]
    mutate(rows[1])
    journal.write_text(''.join(json.dumps(row) + '\n' for row in rows))
    with pytest.raises(ValueError, match=message):
        load_recording(recorder.root)


def coherent_nonzero_state(state):
    state['human']['normalized_action'][-1] = .5
    state['human']['motor_forces'][-1] = 100.
    state['human']['mobility']['support_force'][-1] = 100.
    return state


def test_start_and_reset_require_coherent_zero_commands_including_load(tmp_path):
    scene, state = whole_body_fixture()
    nonzero = coherent_nonzero_state(deepcopy(state))
    with pytest.raises(ValueError, match='start/reset'):
        Recorder(tmp_path / 'nonzero-start', scene).start(nonzero)
    assert not (tmp_path / 'nonzero-start').exists()
    recorder = Recorder(tmp_path / 'reset-check', scene)
    recorder.start(state)
    before = (recorder.root / 'events.jsonl').read_bytes()
    with pytest.raises(ValueError, match='start/reset'):
        recorder.append({'type': 'reset', 'seed': 4, 'snapshot': nonzero})
    assert (recorder.root / 'events.jsonl').read_bytes() == before
    recorder.append({'type': 'reset', 'seed': 4, 'snapshot': state})
    recorder.close()
    journal = recorder.root / 'events.jsonl'
    rows = [json.loads(line) for line in journal.read_text().splitlines()]
    for index in [0, 1]:
        altered = deepcopy(rows)
        coherent_nonzero_state(altered[index]['snapshot'])
        journal.write_text(''.join(json.dumps(row) + '\n' for row in altered))
        with pytest.raises(ValueError, match='start/reset'):
            load_recording(recorder.root)


def test_force_crosscheck_uses_absolute_not_relative_tolerance(tmp_path):
    scene, state = whole_body_fixture()
    recorder = Recorder(tmp_path / 'force-tolerance', scene)
    recorder.start(state)
    state['human']['normalized_action'][-1] = 1.
    state['human']['motor_forces'][-1] = 200. + 5e-10
    state['human']['mobility']['support_force'][-1] = 200.
    action = state['human']['normalized_action'].copy()
    recorder.append(event(state, action))
    state['human']['motor_forces'][-1] = 200. + 1e-8
    with pytest.raises(ValueError, match='times physical limits'):
        recorder.append(event(state, action))
    recorder.close()


@pytest.mark.parametrize(('field', 'value', 'message'), [
    ('physical_limits', [0.] * 22, 'physical limits'),
    ('units', ['N'] * 22, 'action units'),
    ('substeps_per_action', 4, '8 substeps'),
    ('hidden_assistance', True, 'hidden assistance'),
])
def test_load_revalidates_physical_action_contract(tmp_path, field, value, message):
    scene, state = whole_body_fixture()
    recorder = Recorder(tmp_path / 'contract-load', scene)
    recorder.start(state)
    recorder.close()
    path = recorder.root / 'scene.json'
    altered = json.loads(path.read_text())
    altered['action_contract'][field] = value
    path.write_text(json.dumps(altered))
    with pytest.raises(ValueError, match=message):
        load_recording(recorder.root)
