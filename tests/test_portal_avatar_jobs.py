"""Worker ownership, bounded catalog restore and lock-fenced attachment."""
from concurrent.futures import Future
import json
from pathlib import Path
import threading
from types import SimpleNamespace

import pytest

from test_portal_avatar import archive
from test_avatar_assets import factory, metadata
from weft.portal_avatar_jobs import PortalAvatarManager, RESERVE_PER_ASSET


def wait(manager):
    manager._future.result(timeout=20)
    return manager.status()


def upload(tmp_path, identity='custom-test'):
    source, files = factory(tmp_path, glb=True)
    meta = metadata(); meta['asset_id'] = identity
    return archive({'model.glb': files['model.glb'], 'mapping.json': json.dumps(meta).encode()})


def test_worker_import_restore_and_cached_status(tmp_path, monkeypatch):
    raw = upload(tmp_path)
    manager = PortalAvatarManager(tmp_path/'library', Path.cwd())
    try:
        assert wait(manager)['status']=='not_prepared'
        assert manager.reserve_for(raw)==RESERVE_PER_ASSET
        manager.submit(raw)
        status=wait(manager)
        assert status['status']=='ready' and status['custom_count']==1 and not status['builtins_ready']
        assert status['reserved_bytes']==0 and len(status['choices'])==1
        # Polling cannot normalize, read files or instantiate another catalog.
        monkeypatch.setattr('weft.portal_avatar_jobs.AvatarCatalog', lambda *_: pytest.fail('status must stay cached'))
        assert manager.status()==status
    finally: manager.close()
    monkeypatch.undo()
    resumed=PortalAvatarManager(tmp_path/'library',Path.cwd())
    try:
        assert wait(resumed)['choices']==status['choices']
    finally: resumed.close()


def test_overlap_rejected_without_extra_worker(tmp_path):
    manager=PortalAvatarManager(tmp_path/'library',Path.cwd());wait(manager)
    prior=manager._future;manager._future=Future()
    try:
        with pytest.raises(ValueError,match='running'): manager.submit()
        with pytest.raises(ValueError,match='running'): manager.attach_async('a'*32,object())
    finally: manager._future=prior;manager.close()


def test_failed_import_preserves_previous_catalog_and_receipt(tmp_path):
    manager=PortalAvatarManager(tmp_path/'library',Path.cwd());wait(manager)
    try:
        manager.submit(upload(tmp_path)); prior=wait(manager)['choices']
        manager.submit(b'not zip'); state=wait(manager)
        assert state['status']=='failed' and state['choices']==prior and state['reserved_bytes']==0
        assert len(list((tmp_path/'library/operations').glob('*/failure.json')))==1
    finally: manager.close()


def test_restore_tamper_fails_closed(tmp_path):
    manager=PortalAvatarManager(tmp_path/'library',Path.cwd());wait(manager)
    manager.submit(upload(tmp_path));wait(manager);manager.close()
    doc=json.loads((tmp_path/'library/catalog.json').read_bytes());doc['entries'][0]['key']='0'*64
    (tmp_path/'library/catalog.json').write_text(json.dumps(doc))
    manager=PortalAvatarManager(tmp_path/'library',Path.cwd())
    try:
        assert wait(manager)['status']=='failed'
        with pytest.raises(ValueError,match='restored'):manager.submit()
    finally:manager.close()


def test_attach_uses_existing_session_lock_and_stale_close_does_not_reuse(tmp_path, monkeypatch):
    manager=PortalAvatarManager(tmp_path/'library',Path.cwd());wait(manager)
    manager.submit(upload(tmp_path));wait(manager)
    lock=threading.RLock(); events=[]
    session=SimpleNamespace(scene={'profile':'shared-world-v3'},control_profile='shared_exercise')
    def loaded():
        assert lock._is_owned();events.append('load');return session
    lab=SimpleNamespace(lock=lock,load_session=loaded,recorder=None,export_future=None,close=lambda:events.append('close'))
    try:
        manager.attach_async('a'*32,lab)
        assert wait(manager)['attachments']['a'*32]['status']=='ready'
        wrapped=manager.attached('a'*32)
        assert wrapped._lab is lab and events==['load']
        wrapped.close();assert manager.attached('a'*32) is None and events==['load','close']
    finally:manager.close()


def test_active_recorder_attachment_refused_and_original_retained(tmp_path):
    manager=PortalAvatarManager(tmp_path/'library',Path.cwd());wait(manager)
    manager.submit(upload(tmp_path));wait(manager)
    recorder=object();session=SimpleNamespace(scene={'profile':'shared-world-v3'})
    lab=SimpleNamespace(lock=threading.RLock(),load_session=lambda:session,recorder=recorder,export_future=None)
    try:
        manager.attach_async('a'*32,lab)
        assert wait(manager)['attachments']['a'*32]['status']=='failed'
        assert lab.recorder is recorder and manager.attached('a'*32) is None
    finally:manager.close()


def test_failed_reattach_keeps_original_wrapper_and_recording(tmp_path):
    manager=PortalAvatarManager(tmp_path/'library',Path.cwd());wait(manager)
    manager.submit(upload(tmp_path));wait(manager)
    session=SimpleNamespace(scene={'profile':'shared-world-v3'})
    lab=SimpleNamespace(lock=threading.RLock(),load_session=lambda:session,recorder=None,export_future=None,close=lambda:None)
    try:
        manager.attach_async('a'*32,lab);wait(manager)
        original=manager.attached('a'*32);recorder=object();lab.recorder=recorder
        manager.attach_async('a'*32,original)
        assert manager.attached('a'*32) is original
        assert wait(manager)['attachments']['a'*32]['status']=='failed'
        assert manager.attached('a'*32) is original and lab.recorder is recorder
    finally:manager.close()


def test_failed_and_interrupted_status_remain_visible_on_restore(tmp_path):
    for prior in ('failed','running'):
        root=tmp_path/prior;root.mkdir();(root/'status.json').write_text(json.dumps({'status':prior,'error':'retained failure'}))
        manager=PortalAvatarManager(root,Path.cwd())
        try:
            state=wait(manager)
            assert state['status']==('failed' if prior=='failed' else 'interrupted')
            assert state['error'] and state['choices']==[]
        finally:manager.close()


def test_builtin_manager_preparation_is_idempotent_and_bounded(tmp_path):
    manager=PortalAvatarManager(tmp_path/'library',Path.cwd());wait(manager)
    try:
        assert manager.reserve_for()==2*RESERVE_PER_ASSET
        manager.submit();state=wait(manager)
        assert state['builtins_ready'] and state['custom_count']==0 and len(state['choices'])==2
        operations=list((tmp_path/'library/operations').iterdir())
        assert manager.reserve_for()==0
        assert manager.submit()==state
        assert list((tmp_path/'library/operations').iterdir())==operations
    finally:manager.close()


def test_six_custom_limit_refuses_before_new_operation(tmp_path):
    manager=PortalAvatarManager(tmp_path/'library',Path.cwd());wait(manager)
    try:
        for i in range(6):
            source=tmp_path/str(i);source.mkdir();manager.submit(upload(source,'custom-'+str(i)));assert wait(manager)['status']=='ready'
        count=len(list((tmp_path/'library/operations').iterdir()))
        with pytest.raises(ValueError,match='six custom'):manager.reserve_for(b'new')
        with pytest.raises(ValueError,match='six custom'):manager.submit(b'new')
        assert len(list((tmp_path/'library/operations').iterdir()))==count
    finally:manager.close()
