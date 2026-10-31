from concurrent.futures import Future
from functools import partial
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from weft.lab import Lab
from test_lab_http import make_ready_lab, request, serving


NAME = 'session-111111111111'
OTHER = 'session-222222222222'


class Executor:
    def __init__(self):
        self.calls = []
        self.future = Future()
        self.future.set_running_or_notify_cancel()

    def submit(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return self.future


@pytest.fixture
def backend(monkeypatch):
    read = lambda *args, **kwargs: None
    write = lambda *args, **kwargs: None
    module = SimpleNamespace(load_recording_evidence=read, export_recording_with_receipt=write)
    monkeypatch.setitem(sys.modules, 'weft.recording_receipt', module)
    return module


def lab_state():
    return SimpleNamespace(root=Path('/unused/lab'), config={'profile': 'spatial-rigid'},
        status=lambda: {'status': 'ready'}, executor=Executor(), export_path=None, export_future=None,
        recording_evidence_future=None, recording_evidence_name=None, recording_evidence_started=0)


def test_selected_receipt_uses_one_off_thread_job_and_never_loads_a_session(backend):
    lab = lab_state()
    assert Lab.recording_evidence(lab, NAME) == {'status': 'loading', 'session': NAME}
    assert Lab.recording_evidence(lab, NAME)['status'] == 'loading'
    assert Lab.recording_evidence(lab, OTHER)['status'] == 'loading'
    assert len(lab.executor.calls) == 1
    args, kwargs = lab.executor.calls[0]
    assert args == (backend.load_recording_evidence, lab.root, NAME)
    assert kwargs == {'expected_run_manifest_sha256': None, 'scene_sha256': None, 'task_sha256': None}
    result = {'status': 'unavailable', 'session': NAME, 'reason': 'Legacy export has no receipt'}
    lab.executor.future.set_result(result)
    assert Lab.recording_evidence(lab, NAME) is result
    assert len(lab.executor.calls) == 1
    Lab.recording_evidence(lab, OTHER)
    assert len(lab.executor.calls) == 2


def test_in_progress_export_and_not_ready_lab_do_not_queue_readers(backend):
    lab = lab_state()
    lab.export_path, lab.export_future = NAME, Future()
    assert Lab.recording_evidence(lab, NAME)['status'] == 'loading'
    lab.status = lambda: {'status': 'building'}
    assert Lab.recording_evidence(lab, OTHER)['status'] == 'unavailable'
    assert not lab.executor.calls


@pytest.mark.parametrize('name', ['../bad', 'session-111111111111/manifest.json', 'session-example', None])
def test_unsafe_selection_never_queues_work(backend, name):
    lab = lab_state()
    with pytest.raises(ValueError):
        Lab.recording_evidence(lab, name)
    assert not lab.executor.calls


@pytest.mark.parametrize('shared,appearance', [(False, False), (True, False), (True, True)])
def test_export_wrapper_preserves_original_export_and_appearance_path(backend, shared, appearance):
    lab = lab_state()
    lab.config['manifest_sha256'] = 'a' * 64
    lab.recording_evidence_name = NAME
    recorder = SimpleNamespace(root=Path('/unused/lab/sessions') / NAME)
    if appearance:
        recorder.asset_future = object()
    session = SimpleNamespace(control_profile='shared_exercise' if shared else 'spatial_rigid')
    Lab._queue_export(lab, recorder, session)
    assert len(lab.executor.calls) == 1 and lab.recording_evidence_name is None
    args, kwargs = lab.executor.calls[0]
    assert not kwargs
    wrapped = args[2] if appearance else args[0]
    assert isinstance(wrapped, partial) and wrapped.func is backend.export_recording_with_receipt
    assert wrapped.keywords['run_directory'] == lab.root / 'run'
    assert wrapped.keywords['expected_run_manifest_sha256'] == 'a' * 64
    assert wrapped.keywords['export'].__name__ == ('export_shared_exercise_session' if shared else 'export_recording')
    if appearance:
        assert args[0].__name__ == 'export_with_appearance' and args[1] is recorder
    else:
        assert args[1] == recorder.root


def test_selected_recording_route_requires_auth_and_exactly_one_session(tmp_path, monkeypatch):
    lab = make_ready_lab(tmp_path / 'lab')
    calls = []
    def inspect(name):
        calls.append(name)
        return {'status': 'unavailable', 'session': name, 'reason': 'No saved receipt'}
    monkeypatch.setattr(lab, 'recording_evidence', inspect)
    with serving(lab) as server:
        route = '/api/recording-evidence?session=' + NAME
        assert request(server, route, token=False)[0] == 401
        assert request(server, route, headers={'Origin': 'https://attacker.example'})[0] == 403
        assert request(server, route + '&session=' + OTHER)[0] == 400
        assert request(server, '/api/recording-evidence')[0] == 400
        assert request(server, route + '&extra=1')[0] == 400
        code, _, raw = request(server, route)
        assert code == 200 and json.loads(raw)['session'] == NAME
        assert lab.session is None and calls == [NAME]
