from concurrent.futures import Future
import json
from types import SimpleNamespace

import pytest

from weft import lab as lab_module, lab_inspection, shared_exercise_diagnostics
from weft.lab import Lab, LabHandler


NAME = 'session-123456789abc'
OTHER = 'session-abcdef012345'
SCENE, TASK = 'a' * 64, 'b' * 64


def make_lab(root):
    (root / 'lab-config.json').write_text(json.dumps({
        'profile': 'shared-world', 'control_profile': 'shared_exercise',
        'scene_sha256': SCENE, 'task_sha256': TASK}))
    (root / 'status.json').write_text('{"status":"ready"}')
    return Lab(root)


def test_selected_recording_summary_is_scene_task_bound_without_physics(tmp_path, monkeypatch):
    calls = []
    timeline = {'scene_sha256': SCENE, 'task_sha256': TASK, 'marker': 'saved recording only'}
    monkeypatch.setattr(shared_exercise_diagnostics, 'inspect_recording_failure',
                        lambda path, **kwargs: calls.append((path, kwargs)) or timeline)
    result = lab_inspection.load_recording_inspection(tmp_path, NAME, scene_sha256=SCENE, task_sha256=TASK)
    assert result['status'] == 'available' and result['timeline'] is timeline
    assert calls == [(tmp_path / 'sessions' / NAME, {'max_seconds': 10})]
    assert 'separate from the current live state' in result['scope']


@pytest.mark.parametrize('changed', ['scene_sha256', 'task_sha256'])
def test_other_scene_or_task_never_appears_as_current_trial(tmp_path, monkeypatch, changed):
    timeline = {'scene_sha256': SCENE, 'task_sha256': TASK, changed: 'c' * 64}
    monkeypatch.setattr(shared_exercise_diagnostics, 'inspect_recording_failure', lambda *a, **k: timeline)
    result = lab_inspection.load_recording_inspection(tmp_path, NAME, scene_sha256=SCENE, task_sha256=TASK)
    assert result['status'] == 'unavailable' and 'different scene or task' in result['reason']
    assert 'timeline' not in result


@pytest.mark.parametrize('name', ['', '../session-123456789abc', '/tmp/session-123456789abc',
                                  'session-123456789abc/scene.json', 'session-not-an-id', None])
def test_arbitrary_recording_paths_never_schedule_io(tmp_path, name):
    instance = make_lab(tmp_path)
    try:
        with pytest.raises(ValueError, match='Choose one completed recording'):
            instance.recording_inspection(name)
        assert instance.recording_inspection_future is None and instance.session is None
    finally:
        instance.close()


def test_recording_background_work_is_single_bounded_and_expires(tmp_path, monkeypatch):
    instance = make_lab(tmp_path)
    instance.executor.shutdown()
    calls, futures, clock = [], [], [0.0]

    def submit(*args, **kwargs):
        calls.append((args, kwargs))
        future = Future()
        future.set_running_or_notify_cancel()
        futures.append(future)
        return future

    instance.executor = SimpleNamespace(submit=submit, shutdown=lambda **kwargs: None)
    monkeypatch.setattr(lab_module.time, 'monotonic', lambda: clock[0])
    monkeypatch.setattr(instance, 'load_session', lambda: pytest.fail('No simulation for saved inspection'))
    try:
        assert instance.recording_inspection(NAME)['status'] == 'loading'
        for name in [NAME, OTHER, NAME, OTHER]:
            assert instance.recording_inspection(name)['status'] == 'loading'
        assert len(calls) == 1
        futures[0].set_result({'status': 'available', 'session': NAME})
        assert instance.recording_inspection(NAME)['session'] == NAME
        assert len(calls) == 1
        assert instance.recording_inspection(OTHER)['status'] == 'loading'
        assert len(calls) == 2
        futures[1].set_result({'status': 'available', 'session': OTHER})
        clock[0] = 601
        assert instance.recording_inspection(OTHER)['status'] == 'loading'
        assert len(calls) == 3 and instance.session is None
        futures[2].set_result({'status': 'unavailable'})
    finally:
        instance.close()


@pytest.mark.parametrize('control_profile,status', [('whole_body', 'ready'), ('shared_exercise', 'failed')])
def test_unsupported_lab_state_does_not_start_inspection(tmp_path, control_profile, status):
    instance = make_lab(tmp_path)
    instance.config['control_profile'] = control_profile
    (tmp_path / 'status.json').write_text(json.dumps({'status': status}))
    try:
        assert instance.recording_inspection(NAME)['status'] == 'unavailable'
        assert instance.recording_inspection_future is None
    finally:
        instance.close()


def route(instance, path, *, authorized=True):
    handler = object.__new__(LabHandler)
    handler.path, handler.server = path, SimpleNamespace(lab=instance)
    authorization, replies = [], []
    handler.authorized = lambda **kwargs: authorization.append(kwargs) or authorized
    handler.reply = lambda value, code=200: replies.append((code, value))
    handler.do_GET()
    return authorization, replies


def test_recording_route_uses_auth_and_rejects_ambiguous_queries_without_server(tmp_path, monkeypatch):
    instance = make_lab(tmp_path)
    calls = []
    def inspect(name):
        lab_inspection.recording_session_name(name)
        calls.append(name)
        return {'status': 'loading'}
    monkeypatch.setattr(instance, 'recording_inspection', inspect)
    try:
        path = '/api/recording-inspection?session=' + NAME
        auth, replies = route(instance, path, authorized=False)
        assert auth == [{'public': False}] and not replies and not calls
        for query in ['', '?session=', '?session=' + NAME + '&session=' + OTHER, '?session=' + NAME + '&path=/tmp']:
            _, replies = route(instance, '/api/recording-inspection' + query)
            assert replies[0][0] == 400
        assert not calls
        _, replies = route(instance, path)
        assert replies == [(200, {'status': 'loading'})] and calls == [NAME]
    finally:
        instance.close()
