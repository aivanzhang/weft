"""Developer reloads never interrupt jobs, requests or live worlds."""
from types import SimpleNamespace
import signal
import threading

import pytest

from weft.portal_reload import ReloadablePortalServer


def fixture():
    portal = SimpleNamespace(lock=threading.RLock(), active=None, labs={}, thread=None,
        jobs={}, stopping=threading.Event(), _pending_jobs=lambda: 0,
        _review_status=lambda _: {}, _check_status=lambda _: {}, _repair_status=lambda _: {})
    server = object.__new__(ReloadablePortalServer)
    server.portal = portal
    server.reload_pending = True
    server.next_reload_check = 0
    server.slots = threading.BoundedSemaphore(8)
    return server, portal


def test_idle_restart_uses_normal_cleanup_and_returns_restart_exit(monkeypatch):
    signals = []
    monkeypatch.setattr(signal, 'signal', lambda *args: signals.append(args))
    server, portal = fixture()
    with pytest.raises(SystemExit) as exit:
        server.service_actions()
    assert exit.value.code == 75
    assert portal.stopping.is_set()
    assert signals == [(signal.SIGINT, signal.SIG_IGN)]
    for _ in range(8):
        assert server.slots.acquire(blocking=False)


@pytest.mark.parametrize('busy', ['worker', 'world', 'request', 'queue', 'review', 'thread', 'avatar'])
def test_restart_defers_without_cancelling_work(busy, monkeypatch):
    server, portal = fixture()
    if busy == 'worker': portal.active = 'job'
    if busy == 'world': portal.labs['world'] = object()
    if busy == 'request': server.slots.acquire()
    if busy == 'queue': portal._pending_jobs = lambda: 1
    if busy == 'review':
        portal.jobs['job'] = {}
        portal._review_status = lambda _: {'status': 'queued'}
    if busy == 'thread':
        worker = threading.Thread()
        monkeypatch.setattr(worker, 'is_alive', lambda: True)
        monkeypatch.setattr(threading, 'enumerate', lambda: [worker])
    if busy == 'avatar':
        portal.avatar_manager = SimpleNamespace(_lock=threading.RLock(), _future=SimpleNamespace(done=lambda: False))
    server.service_actions()
    assert not portal.stopping.is_set()
    assert server.reload_pending
