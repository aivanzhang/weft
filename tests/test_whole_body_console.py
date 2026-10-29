import json
from types import SimpleNamespace

import numpy as np
import pytest

from weft.human import build_human
from weft.recording import load_recording
from weft.whole_body_console import command_action, console
from weft.whole_body_learning import train_whole_body
from test_mobile_integration import evidence


def test_manual_command_only_drives_named_motor_and_is_bounded():
    env = SimpleNamespace(action_names=tuple(f'm{i}' for i in range(22)))
    choose, count = command_action('motor m21 -.4 6', env, {})
    assert count == 6
    np.testing.assert_array_equal(choose(), [0] * 21 + [-.4])
    for command in ('motor m0 nan', 'motor m0 1.1', 'motor absent 0', 'release 7',
                    'release 0', 'release 1 2', 'learned -1', 'motor m1 1.0 0'):
        with pytest.raises(ValueError):
            command_action(command, env, {})


def test_terminal_records_actual_actions_resets_and_visual_export(tmp_path):
    model = tmp_path / 'model'
    model.mkdir()
    spec = build_human(evidence(), model, body_model='mobile')
    run = tmp_path / 'learning'
    train_whole_body(spec, run, generations=1, population=4, horizon_steps=6)
    commands = iter(['motor root_x .1 2', 'reset', 'learned 6', 'release', 'reset', 'follow 2', 'quit'])
    output = []
    result = console(run, tmp_path / 'recording', input_fn=lambda _: next(commands), output_fn=output.append)
    saved = load_recording(tmp_path / 'recording')
    steps = [row for row in saved['events'] if row['type'] == 'step']
    horizon = json.loads((run / 'settings.json').read_text())['partition']['horizon_steps']
    assert len(steps) == horizon + 4 and result['resets'] == 2 and result['video_frames'] == horizon + 7
    np.testing.assert_array_equal(steps[0]['action'], [0] * 19 + [.1, 0, 0])
    assert any('Episode ended' in text for text in output)
    assert all(len(row['action']) == 22 for row in steps)
    scene = json.loads((tmp_path / 'recording' / 'scene.json').read_text())
    assert scene['action_contract']['hidden_assistance'] is False
    assert scene['profile'] == 'exercise-imitation-v1'
    with pytest.raises(ValueError, match='immutable'):
        console(run, run / 'recording')


def test_failed_action_is_retained_without_fabricating_terminal_pose(tmp_path, monkeypatch):
    from weft.whole_body import WholeBodyEnv
    model = tmp_path / 'model'
    model.mkdir()
    spec = build_human(evidence(), model, body_model='mobile')
    run = tmp_path / 'learning'
    train_whole_body(spec, run, generations=1, population=4, horizon_steps=3)

    def fail_step(self, action):
        raise RuntimeError('nonfinite test failure')

    monkeypatch.setattr(WholeBodyEnv, 'step', fail_step)
    result = console(run, tmp_path / 'failed', input_fn=lambda _: 'motor root_x .2', output_fn=lambda _: None)
    saved = load_recording(tmp_path / 'failed')
    assert result['steps'] == 0 and result['video_frames'] == 1
    assert saved['metadata']['failed_attempt']['reason'] == 'nonfinite test failure'
    assert saved['metadata']['failed_attempt']['attempted_action'][19] == .2
    assert len(saved['events']) == 1
