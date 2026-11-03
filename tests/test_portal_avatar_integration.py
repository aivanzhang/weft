"""Avatar HTTP dispatch and portal ownership; worker internals are separately tested."""
from copy import deepcopy
from concurrent.futures import Future
import json
from types import SimpleNamespace
import pytest
from weft.io import write_json
from weft.portal import Portal,MAX_STORAGE
from test_portal import serving,request


class Manager:
    def __init__(self,*args):
        self.state={'status':'ready','reserved_bytes':0,'attachments':{}}
        self.wrappers={};self.calls=[];self.reserve=97<<20;self.on_close=None
    def status(self):return deepcopy(self.state)
    def reserve_for(self,raw=None):self.calls.append(('reserve',raw));return self.reserve
    def submit(self,raw=None):self.calls.append(('submit',raw));return self.status()
    def attach_async(self,identity,lab):self.calls.append(('attach',identity,lab));return self.status()
    def attached(self,identity):return self.wrappers.get(identity)
    def close(self):
        self.calls.append(('close',))
        if self.on_close:self.on_close()


@pytest.fixture
def portal(tmp_path,monkeypatch):
    monkeypatch.setattr('weft.portal_avatar_jobs.PortalAvatarManager',Manager)
    value=Portal(tmp_path/'portal',project=tmp_path,start=False)
    yield value
    if not getattr(value,'_test_closed',False):value.close()


def project(portal,letter):
    identity=letter*32;folder=portal.root/identity;folder.mkdir()
    portal.jobs[identity]={'id':identity,'kind':'video','mode':'exercise','source_sha256':'f'*64}
    write_json(folder/'status.json',{'status':'ready','route':'exercise','lab':'lab'})
    return identity


def lab(label,events,**values):
    return SimpleNamespace(recorder=None,export_future=None,loading=False,
                           close=lambda:events.append(label),**values)


@pytest.mark.parametrize('path', ['/api/avatars/builtins','/api/avatars/import','/api/projects/'+'a'*32+'/avatars/attach'])
def test_http_avatar_mutations_require_auth_origin_and_closed_length(portal,path,monkeypatch):
    dispatched=[]
    monkeypatch.setattr(portal,'prepare_avatars',lambda raw=None:dispatched.append(raw) or {'status':'running'})
    monkeypatch.setattr(portal,'attach_avatars',lambda identity:dispatched.append(identity) or {'status':'running'})
    with serving(portal) as server:
        assert request(server,path,'POST',b'{}',auth=False)[0]==401
        assert request(server,path,'POST',b'{}',headers={'Origin':'https://attacker.invalid'})[0]==403
        assert request(server,path,'POST',b'{}',headers={'Transfer-Encoding':'chunked'})[0]==400
        assert request(server,path,'POST',b'{}',headers={'Content-Length':str((32<<20)+1)})[0]==400
        if not path.endswith('/import'):
            assert request(server,path,'POST',b'[]')[0]==400
        assert not dispatched
        assert request(server,path,'POST',b'{}')[0]==202
        assert len(dispatched)==1


def test_storage_reservation_refuses_before_dispatch(portal,monkeypatch):
    manager=portal.avatar_manager
    monkeypatch.setattr(portal,'_stored_bytes',lambda:MAX_STORAGE-manager.reserve-10)
    manager.state['reserved_bytes']=11
    with pytest.raises(ValueError,match='storage capacity'):portal.prepare_avatars(b'zip')
    assert manager.calls==[('reserve',b'zip')]
    manager.state['reserved_bytes']=10
    portal.prepare_avatars(b'zip')
    assert manager.calls[-1]==('submit',b'zip')


def test_cold_attachment_refuses_without_loading_source(portal,monkeypatch):
    identity=project(portal,'a')
    monkeypatch.setattr(portal,'lab',lambda *_:pytest.fail('cold HTTP attachment cannot load a source/model'))
    with pytest.raises(ValueError,match='(?i)(open|sandbox)'):portal.attach_avatars(identity)
    assert not portal.avatar_manager.calls


def test_warm_attachment_uses_existing_lab_and_cached_wrapped_return(portal,monkeypatch):
    identity=project(portal,'a');events=[];base=lab('base',events);wrapped=lab('wrapped',events)
    portal.labs[identity]=base
    monkeypatch.setattr('weft.portal.open_portal_lab',lambda *_a,**_k:pytest.fail('must reuse admitted session'))
    portal.attach_avatars(identity)
    assert portal.avatar_manager.calls==[('attach',identity,base)]
    portal.avatar_manager.wrappers[identity]=wrapped
    assert portal.lab(identity) is wrapped and portal.labs[identity] is base


def test_inflight_attachment_is_not_evicted_and_idle_wrapper_closes_once(portal,monkeypatch):
    events=[];ids=[project(portal,c) for c in 'abcd']
    for i in ids[:3]:portal.labs[i]=lab(i,events)
    wrapped=lab('wrapped-b',events);portal.avatar_manager.wrappers[ids[1]]=wrapped
    portal.avatar_manager.state['attachments']={ids[0]:{'status':'running'},ids[1]:{'status':'ready'}}
    created=[]
    monkeypatch.setattr('weft.portal.open_portal_lab',lambda *a,**k:created.append(a[0]) or lab('new',events))
    result=portal.lab(ids[3])
    assert events==['wrapped-b'] and ids[0] in portal.labs and ids[1] not in portal.labs
    assert result is portal.labs[ids[3]] and len(created)==1


def test_no_eviction_when_all_labs_owned_by_attach_record_or_export(portal,monkeypatch):
    events=[];ids=[project(portal,c) for c in 'abcd']
    for i in ids[:3]:portal.labs[i]=lab(i,events)
    portal.avatar_manager.state['attachments']={ids[0]:{'status':'running'}}
    portal.labs[ids[1]].recorder=object();portal.labs[ids[2]].export_future=Future()
    monkeypatch.setattr('weft.portal.open_portal_lab',lambda *_a,**_k:pytest.fail('no free slot'))
    with pytest.raises(ValueError,match='fourth'):portal.lab(ids[3])
    assert events==[] and list(portal.labs)==ids[:3]


def test_close_drains_attachment_owner_before_closing_published_wrapper(portal):
    events=[];identity=project(portal,'a');base=lab('base',events);wrapped=lab('wrapped',events)
    portal.labs[identity]=base
    portal.avatar_manager.state['attachments']={identity:{'status':'running'}}
    def publish():
        events.append('attach-drained');portal.avatar_manager.wrappers[identity]=wrapped
    portal.avatar_manager.on_close=publish
    portal.close();portal._test_closed=True
    assert events==['attach-drained','wrapped']
    assert portal.avatar_manager.calls==[('close',)]
