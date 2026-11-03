"""Portal ownership, consent transport and supervised perception lifecycle; no providers invoked."""
from io import BytesIO
import json
from types import SimpleNamespace
import pytest
from weft import portal as module
from weft.io import write_json
from weft.portal import Portal, PortalHandler


@pytest.fixture
def portal(tmp_path, monkeypatch):
    monkeypatch.setattr(module, 'muse_availability', lambda _: {'available':False})
    monkeypatch.setattr(module, 'credential_status', lambda _: {'configured':True})
    p = Portal(tmp_path/'portal', project=tmp_path, start=False)
    monkeypatch.setattr(p.budget, 'status', lambda: {'remaining_usd':5,'reservation_usd':.3135})
    job = p.upload(BytesIO(b'fixture'), 7, name='exercise.mp4', mode='exercise', ai=True,
                   request_id='a'*32, exercise_inventory='equipment_or_obstacles', exercise_output='simulation')
    folder = p._folder(job['id'])
    write_json(folder/'status.json', {'status':'needs_input','route':'exercise','events':[{'message':'original'}]})
    yield p, folder
    p.close()


def handler(action, raw=b'', *, authorized=True, length=None):
    calls, replies = [], []
    h = SimpleNamespace(path='/api/projects/'+'a'*32+'/perception/'+action,
        headers={'Content-Length':str(len(raw) if length is None else length)},rfile=BytesIO(raw),
        connection=SimpleNamespace(settimeout=lambda _:None),route_lab=lambda:None,authorized=lambda:authorized,
        server=SimpleNamespace(portal=SimpleNamespace(preflight_perception=lambda *a:calls.append(a) or {},
            prepare_perception=lambda *a:calls.append(a) or {},
            prepare_body_detail=lambda *a:calls.append(a) or {})),reply=lambda value,status=200:replies.append((value,status)))
    PortalHandler.do_POST(h)
    return calls,replies


def test_authenticated_transport_delivers_exact_closed_request():
    payload={'quote_sha256':'b'*64,'consent':{'openai_sampled_images':True,'meta_full_video_and_regions':True},
             'openai_max_usd':.3135,'meta_max_usd':.15,'request_id':'c'*32}
    calls,replies=handler('prepare',json.dumps(payload).encode())
    assert calls==[('a'*32,payload)] and replies[-1][1]==202
    assert handler('prepare',json.dumps(payload).encode(),authorized=False)==([],[])
    assert handler('preflight')[0]==[('a'*32,)]


def test_detailed_body_transport_requires_authenticated_explicit_request():
    payload={'recipe':'full-dense-region-v1','request_id':'d'*32}
    raw=json.dumps(payload).encode()
    calls,replies=handler('body-detail',raw)
    assert calls==[('a'*32,payload)] and replies[-1][1]==202
    assert handler('body-detail',raw,authorized=False)==([],[])


@pytest.mark.parametrize('action,raw,length',[
    ('preflight',b'{}',2),('prepare',b'',0),('prepare',b'{}',2049),('prepare',b'{}',3),
    ('prepare',b'{"x":1,"x":2}',None),('prepare',b'{',None),
    ('body-detail',b'',0),('body-detail',b'{}',2049),('body-detail',b'{}',3),
    ('body-detail',b'{"recipe":1,"recipe":2}',None)])
def test_transport_rejects_arguments_size_truncation_and_duplicate_keys(action,raw,length):
    calls,replies=handler(action,raw,length=length)
    assert not calls and replies[-1][1]==400


def test_local_queue_deduplicates_preserves_parent_and_has_no_paid_consent(portal):
    p,folder=portal;original=(folder/'status.json').read_bytes()
    result=p.preflight_perception(folder.name)
    assert result['perception']['status']=='queued_preflight' and result['ai'] is True
    before=(folder/'perception/preflight-request.json').read_bytes()
    p.preflight_perception(folder.name)
    assert (folder/'perception/preflight-request.json').read_bytes()==before
    assert (folder/'status.json').read_bytes()==original and not (folder/'perception/request.json').exists()
    assert p._pending_jobs()==1


def test_capacity_and_worker_completion_guard(portal,monkeypatch):
    p,folder=portal
    monkeypatch.setattr(p,'_pending_jobs',lambda:4)
    with pytest.raises(ValueError,match='queue is full'):p.preflight_perception(folder.name)
    assert not (folder/'perception').exists()
    monkeypatch.setattr(p,'_pending_jobs',lambda:0)
    monkeypatch.setattr(p,'_stored_bytes',lambda:module.MAX_STORAGE)
    with pytest.raises(ValueError,match='storage reservation'):p.preflight_perception(folder.name)
    p.active=folder.name
    with pytest.raises(ValueError,match='worker is finishing'):p.prepare_perception(folder.name,{})


@pytest.mark.parametrize('state',['queued_preflight','preflighting','awaiting_consent','evidence_ready'])
def test_restart_interruption_and_resting_preservation(portal,state):
    p,folder=portal
    (folder/'perception').mkdir()
    write_json(folder/'perception/status.json',{'status':state})
    # Awaiting consent must carry its bounded quote for summaries, but restart never parses it.
    p.close()
    p.close = lambda:None
    reopened=Portal(p.root,project=p.project,start=False)
    try:
        assert module.perception_jobs.read_status(folder)['status']==('interrupted' if state in module.perception_jobs.ACTIVE else state)
        assert reopened._status(folder.name)['status']=='needs_input'
    finally:reopened.close()


def test_cancel_handoff_never_starts_queued_child_or_relabels_complete_evidence(portal):
    p,folder=portal
    p.preflight_perception(folder.name);p.active=folder.name
    p.cancel(folder.name)
    assert module.perception_jobs.read_status(folder)['status']=='cancelling'
    # Simulate publication racing with cancellation: cleanup catches the later child.
    (folder/'equipment').mkdir()
    write_json(folder/'equipment/request.json',{'source_kind':'perception'})
    write_json(folder/'equipment/status.json',{'status':'queued'})
    write_json(folder/'perception/status.json',{'status':'evidence_ready'})
    p._cancel_perception_handoff(folder.name)
    assert module.equipment_status(folder)['status']=='cancelled'
    assert module.perception_jobs.read_status(folder)['status']=='evidence_ready'
    assert p._status(folder.name)['status']=='needs_input'


@pytest.mark.parametrize('paid',[False,True])
def test_supervised_worker_filters_credentials_and_keeps_distinct_phase_logs(portal,monkeypatch,paid):
    p,folder=portal;p.preflight_perception(folder.name)
    if paid:write_json(folder/'perception/request.json',{'fixture':True})
    for key in ('OPENAI_API_KEY','META_API_KEY','MODEL_API_KEY','CHATGPT_TOKEN','UNRELATED_SECRET'):
        monkeypatch.setenv(key,'not-a-real-key')
    calls=[]
    def spawn(command,**kwargs):
        calls.append((command,kwargs))
        write_json(folder/'perception/status.json',{'status':'evidence_ready' if paid else 'unsupported'})
        return SimpleNamespace(poll=lambda:0,wait=lambda **kw:0,returncode=0)
    monkeypatch.setattr(module.subprocess,'Popen',spawn)
    p._run(folder.name,perception=True)
    command,kwargs=calls[0]
    assert command[-1]=='perception'
    assert not {'CHATGPT_TOKEN','UNRELATED_SECRET'} & kwargs['env'].keys()
    assert ({'OPENAI_API_KEY','META_API_KEY','MODEL_API_KEY'} & kwargs['env'].keys())==({'OPENAI_API_KEY','META_API_KEY','MODEL_API_KEY'} if paid else set())
    assert (folder/'perception'/('preparation-worker.log' if paid else 'preflight-worker.log')).exists()
    assert p._status(folder.name)['status']=='needs_input'


def test_queue_dispatch_does_not_destroy_phase_or_parent_status(portal,monkeypatch):
    p,folder=portal;p.preflight_perception(folder.name)
    calls=[]
    def run(identity,**kwargs):
        calls.append(kwargs);assert module.perception_jobs.read_status(folder)['status']=='queued_preflight'
        p.stopping.set()
    monkeypatch.setattr(p,'_run',run)
    p._work()
    assert calls==[{'review':False,'checks':False,'perception':True}] and p.active is None
    assert p._status(folder.name)['status']=='needs_input'


def test_artifacts_are_exact_bounded_no_follow_not_whole_body(portal):
    p,folder=portal
    (folder/'perception').mkdir()
    write_json(folder/'perception/quote.json',{'fixture':True})
    assert p.artifact(folder.name,'perception/quote.json')==folder/'perception/quote.json'
    for name in ('perception/../job.json','perception/preflight/body/private.json','perception/preparation-worker.log'):
        with pytest.raises(ValueError):p.artifact(folder.name,name)
    (folder/'perception/request.json').symlink_to(folder/'job.json')
    with pytest.raises(ValueError,match='unsafe'):p.artifact(folder.name,'perception/request.json')
    (folder/'perception/failure.json').write_bytes(b'x'*(1+(1<<20)))
    with pytest.raises(ValueError,match='bounded'):p.artifact(folder.name,'perception/failure.json')


@pytest.mark.parametrize('paid',[False,True])
def test_worker_receipt_retains_original_phase_when_request_changes(portal,monkeypatch,paid):
    from weft import portal_worker as worker
    p,folder=portal;p.preflight_perception(folder.name)
    if paid:write_json(folder/'perception/request.json',{'fixture':True})
    monkeypatch.setattr(worker,'_limit',lambda *a:None)
    monkeypatch.setattr(worker.sys,'argv',['worker',str(folder),str(p.project),'perception'])
    def execute(root,**kwargs):
        if not paid:write_json(root/'perception/request.json',{'fixture':'phase must already be frozen'})
        return 0
    monkeypatch.setattr(module.perception_jobs,'execute_perception_job',execute)
    assert worker.main()==0
    expected='preparation-worker-receipt.json' if paid else 'preflight-worker-receipt.json'
    receipt=json.loads((folder/'perception'/expected).read_bytes())
    assert receipt['exit_code']==0 and receipt['peak_rss_bytes']>0


def test_completed_perception_queued_child_cancels_without_stuck_active_state(portal):
    p,folder=portal;p.preflight_perception(folder.name);p.active=folder.name
    write_json(folder/'perception/status.json',{'status':'evidence_ready'})
    (folder/'equipment').mkdir()
    write_json(folder/'equipment/status.json',{'status':'queued'})
    p.cancel(folder.name)
    assert module.equipment_status(folder)['status']=='cancelled'
    assert module.perception_jobs.read_status(folder)['status']=='evidence_ready'
    assert not (folder/'perception/cancel-request.json').exists()


def test_preflight_pose_downloads_are_explicit_not_unbounded_directory(portal):
    p,folder=portal
    body=folder/'perception/preflight/body_source';body.mkdir(parents=True)
    for name in ('human.xml','human.json','evidence.json','native-clock.json','tracks.gif','tracks-contact-sheet.png'):
        (body/name).write_bytes(b'fixture')
        assert p.artifact(folder.name,'perception/preflight/body_source/'+name)==body/name
    (body/'implementation.py').write_bytes(b'not public')
    with pytest.raises(ValueError):p.artifact(folder.name,'perception/preflight/body_source/implementation.py')
