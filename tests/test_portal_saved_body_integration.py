"""Portal boundary checks; actual source construction is verified separately."""
from copy import deepcopy
from io import BytesIO
import json
import threading
from types import SimpleNamespace

import pytest

from test_portal_saved_body import setup, queue, write
from test_portal_task_integration import project
from weft.portal import Portal, PortalHandler


def test_explicit_scope_uses_existing_queue_and_deduplicates(setup):
    root, parent, payload, options = setup
    p = SimpleNamespace(root=root, lease=options['lease_fd'], jobs=options['existing_jobs'],
        lock=threading.RLock(), active=None, wake=threading.Event(), _folder=lambda n: root/n,
        _stored_bytes=lambda: 0, _reserved_growth=lambda: 0, _pending_jobs=lambda: 0)
    p.summary = lambda n: deepcopy(p.jobs[n])
    first = Portal.prepare_saved_body(p, parent.name, payload)
    assert p.wake.is_set() and len(p.jobs) == 2
    assert Portal.prepare_saved_body(p, parent.name, payload) == first
    p.active = parent.name
    with pytest.raises(ValueError, match='worker is finishing'):
        Portal.prepare_saved_body(p, parent.name, payload)


def test_saved_body_worker_has_no_provider_keys_and_keeps_ready(project, monkeypatch):
    portal, identity, folder = project
    portal.jobs[identity]['kind'] = 'saved_body'
    seen = []
    for key in ('OPENAI_API_KEY', 'META_API_KEY', 'MODEL_API_KEY'):
        monkeypatch.setenv(key, 'fixture-not-forwarded')
    def launch(argv, **kw):
        seen.append((argv, kw['env']))
        write(folder/'status.json', {'status': 'ready', 'route': 'exercise', 'lab': 'lab'})
        return SimpleNamespace(poll=lambda: 0, wait=lambda **_: 0, returncode=0)
    monkeypatch.setattr('weft.portal.subprocess.Popen', launch)
    portal._run(identity)
    assert seen[0][0][-1] == 'saved-body'
    assert not {'OPENAI_API_KEY', 'META_API_KEY', 'MODEL_API_KEY'} & seen[0][1].keys()
    assert json.loads((folder/'status.json').read_bytes())['status'] == 'ready'


def test_dispatch_preserves_saved_body_queued_admission(project):
    portal, identity, folder = project
    portal.jobs[identity]['kind'] = 'saved_body'
    write(folder/'status.json', {'status': 'queued', 'route': 'exercise'})
    seen = []
    def run(*args, **kwargs):
        seen.append(json.loads((folder/'status.json').read_bytes())['status'])
        portal.stopping.set()
    portal._run = run
    portal._work()
    assert seen == ['queued']


def test_saved_body_http_requires_auth_and_strict_small_json():
    calls = []
    h = SimpleNamespace(path='/api/projects/'+'a'*32+'/saved-body', headers={'Content-Length':'2'},
        rfile=BytesIO(b'{}'), connection=SimpleNamespace(settimeout=lambda _: None),
        server=SimpleNamespace(portal=SimpleNamespace(prepare_saved_body=lambda *a: calls.append(a) or {})),
        route_lab=lambda: None, authorized=lambda: True, reply=lambda v,status=200: (status,v))
    assert PortalHandler.do_POST(h)[0] == 202 and calls == [('a'*32,{})]
    h.authorized = lambda: False
    assert PortalHandler.do_POST(h) is None and len(calls) == 1
    h.authorized = lambda: True
    for raw in (b'{"x":1,"x":2}', b'{"x":NaN}', b'{'*2049):
        h.rfile = BytesIO(raw); h.headers = {'Content-Length':str(len(raw))}
        assert PortalHandler.do_POST(h)[0] == 400
    assert len(calls) == 1


def test_saved_body_downloads_exclude_parent_and_code(setup):
    child, job = queue(setup)
    p = SimpleNamespace(_folder=lambda _: child, jobs={job['id']:job})
    assert Portal.artifact(p,job['id'],'saved-body/request.json') == child/'saved-body/request.json'
    for name in ('saved-body/../job.json','saved-body/parent/job.json','saved-body/implementation/portal_saved_body.py'):
        with pytest.raises(ValueError): Portal.artifact(p,job['id'],name)
    (child/'saved-body/report.json').symlink_to(child/'job.json')
    with pytest.raises(ValueError): Portal.artifact(p,job['id'],'saved-body/report.json')


def test_saved_body_review_does_not_enable_checks_or_repairs(project):
    portal, identity, folder = project
    portal.jobs[identity]['kind'] = 'saved_body'
    Portal._require_review_body(portal, identity)
    for action in ('checks', 'repairs'):
        with pytest.raises(ValueError, match='review only'):
            Portal._require_review_body(portal, identity, action=action)
