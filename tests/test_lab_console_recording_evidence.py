from copy import deepcopy
import json

import pytest

from test_lab_console import fake_lab, run_commands


NAME = 'session-111111111111'


def evidence():
    return {'status': 'available', 'session': NAME, 'recorded': {'events': 5, 'actions': 3, 'resets': 1},
            'replay': {'status': 'passed', 'exact_state_replay': True}, 'scope': 'Saved simulation only.'}


def test_recording_evidence_is_read_only_and_does_not_change_selected_control(fake_lab, monkeypatch):
    cls, root = fake_lab
    calls = []
    def read(self, name):
        calls.append(name)
        return evidence()
    monkeypatch.setattr(cls, 'recording_evidence', read, raising=False)
    _, output = run_commands(root, ['motor hip .5', 'record evidence ' + NAME, 'controls', 'quit'])
    assert calls == [NAME]
    assert cls.instances[0].calls == [('/api/reset', {'seed': 1729})]
    objects = [json.loads(line) for line in output if line.startswith('{')]
    assert next(row for row in objects if 'recorded' in row)['recorded']['actions'] == 3
    assert objects[-1]['selected_manual_action'] == [0., .5]


def test_loading_does_not_poll_or_block(fake_lab, monkeypatch):
    cls, root = fake_lab
    replies = iter([{'status': 'loading', 'session': NAME}, evidence()])
    monkeypatch.setattr(cls, 'recording_evidence', lambda self, name: next(replies), raising=False)
    _, output = run_commands(root, ['record evidence ' + NAME, 'status', 'record evidence ' + NAME, 'quit'])
    assert sum('Recording evidence is loading' in line for line in output) == 1
    assert cls.instances[0].calls == [('/api/reset', {'seed': 1729})]


@pytest.mark.parametrize('argument', ['', '../recording', 'session-invalid', 'session-111111111111/scene.json'])
def test_missing_or_unsafe_selection_does_not_start_backend_work(fake_lab, monkeypatch, argument):
    cls, root = fake_lab
    def forbidden(*_):
        raise AssertionError('Invalid selection started work')
    monkeypatch.setattr(cls, 'recording_evidence', forbidden, raising=False)
    run_commands(root, ['record evidence ' + argument, 'quit'])
    assert cls.instances[0].calls == [('/api/reset', {'seed': 1729})]


@pytest.mark.parametrize('change', ['session', 'negative', 'boolean', 'counts', 'replay', 'missing', 'large'])
def test_invalid_receipt_does_not_close_an_active_recording(fake_lab, monkeypatch, change):
    cls, root = fake_lab
    result = deepcopy(evidence())
    if change == 'session': result['session'] = 'session-222222222222'
    elif change == 'negative': result['recorded']['actions'] = -1
    elif change == 'boolean': result['recorded']['actions'] = True
    elif change == 'counts': result['recorded']['events'] = 1
    elif change == 'replay': result['replay']['status'] = 'maybe'
    elif change == 'missing': result.pop('recorded')
    elif change == 'large': result['unexpected'] = 'x' * 65536
    monkeypatch.setattr(cls, 'recording_evidence', lambda self, name: result, raising=False)
    final, output = run_commands(root, ['record start', 'record evidence ' + NAME, 'step', 'record stop', 'quit'])
    assert 'Saved recording evidence has an unsupported format' in output
    assert final['recording']['export'] == 'ready'
    assert [row['type'] for row in cls.instances[0].events] == ['start', 'step']


def test_legacy_unavailable_response_is_escaped_without_an_exact_replay_claim(fake_lab, monkeypatch):
    cls, root = fake_lab
    reply = {'status': 'unavailable', 'session': NAME, 'reason': 'Old export\n\x1b[2Jhas no saved replay receipt'}
    monkeypatch.setattr(cls, 'recording_evidence', lambda self, name: reply, raising=False)
    _, output = run_commands(root, ['record evidence ' + NAME, 'quit'])
    assert all('\x1b' not in line for line in output)
    assert any('Old export' in line for line in output)
