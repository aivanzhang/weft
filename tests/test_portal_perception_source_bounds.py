"""Publication inventory grows only within its measured profile; other guards stay closed."""
from pathlib import Path
import os
import time

import pytest

from test_portal_equipment_candidates import perception_source
from test_equipment_proposal import seal
from weft import portal_equipment_candidates as module, exercise_perception_pipeline as bridge_module
from weft.io import digest,write_json


class InventoryPassed(Exception):
    pass


def padded_inventory(fixture):
    root,bridge,project=fixture
    base=bridge/'body_seed'
    padded=[]
    for index in range(5):
        path=base/f'inventory-bound-{index}.bin';path.write_bytes(b'x');padded.append(path)
    seal(base)
    manifest=module._json((bridge/'manifest.json').read_bytes())
    manifest['children']['body_seed']=digest(base/'manifest.json')
    write_json(bridge/'manifest.json',manifest)
    total=sum((bridge/name).stat().st_size for name in manifest['files'])
    for name in manifest['children']:
        directory=bridge/name;raw=(directory/'manifest.json').read_bytes();members=module._json(raw)['files']
        total+=len(raw)+sum((directory/member).stat().st_size for member in members)
    return root,bridge,project,padded,total-len(padded)


def virtual_sizes(monkeypatch,sizes):
    original=Path.stat
    def bounded_stat(path,*args,**kwargs):
        value=original(path,*args,**kwargs)
        if path in sizes:
            fields=list(value);fields[6]=sizes[path];return os.stat_result(fields)
        return value
    monkeypatch.setattr(Path,'stat',bounded_stat)


def stop_after_inventory(monkeypatch):
    def reached(*args,**kwargs):raise InventoryPassed
    monkeypatch.setattr(bridge_module,'read_equipment_analysis',reached)


@pytest.mark.parametrize('total,admitted',[(257<<20,True),(320<<20,True),((320<<20)+1,False)])
def test_profile_aggregate_boundary_before_semantic_allocation(perception_source,monkeypatch,total,admitted):
    root,bridge,project,padded,base=padded_inventory(perception_source)
    assert module.MAX_JOB_SOURCE_BYTES==320<<20
    quotient,remainder=divmod(total-base,len(padded))
    sizes={p:quotient+(index<remainder) for index,p in enumerate(padded)}
    assert max(sizes.values())<=64<<20
    virtual_sizes(monkeypatch,sizes);stop_after_inventory(monkeypatch)
    expected=InventoryPassed if admitted else ValueError
    with pytest.raises(expected,match=None if admitted else 'aggregate byte bound'):
        module._perception_source(root,bridge,project=project,deadline=time.monotonic()+10)
    assert not (root/module.JOB_SOURCE_PATH).exists()


def test_individual_member_bound_stays_64_mib(perception_source,monkeypatch):
    root,bridge,project,padded,_=padded_inventory(perception_source)
    virtual_sizes(monkeypatch,{padded[0]:(64<<20)+1});stop_after_inventory(monkeypatch)
    with pytest.raises(ValueError,match='aggregate byte bound'):
        module._perception_source(root,bridge,project=project,deadline=time.monotonic()+10)


def test_aggregate_file_bound_stays_1536_before_member_reads(perception_source,monkeypatch):
    root,bridge,project=perception_source
    base=bridge/'body_seed'
    write_json(base/'manifest.json',{'files':{f'bounded-{index}.json':'a'*64 for index in range(1536)}})
    manifest=module._json((bridge/'manifest.json').read_bytes())
    manifest['children']['body_seed']=digest(base/'manifest.json');write_json(bridge/'manifest.json',manifest)
    stop_after_inventory(monkeypatch)
    with pytest.raises(ValueError,match='aggregate file bound'):
        module._perception_source(root,bridge,project=project,deadline=time.monotonic()+10)


def test_expired_deadline_still_refuses_before_publication(perception_source):
    root,bridge,project=perception_source
    with pytest.raises(TimeoutError):
        module._perception_source(root,bridge,project=project,deadline=time.monotonic()-1)
    assert not (root/module.JOB_SOURCE_PATH).exists()


@pytest.mark.parametrize('given,expected',[(None,130.),(150.,130.),(105.,105.)])
def test_publication_uses_smaller_caller_or_default_deadline(tmp_path,monkeypatch,given,expected):
    calls=[]
    monkeypatch.setattr(module.time,'monotonic',lambda:100.)
    def qualification(*args,**kwargs):
        calls.append(kwargs['deadline']);return {'fixture':'qualified'},None,None
    monkeypatch.setattr(module,'_perception_source',qualification)
    assert module.publish_perception_source(tmp_path,tmp_path/'bridge',deadline=given)=={'fixture':'qualified'}
    assert calls==[expected]


@pytest.mark.parametrize('given',[True,'soon',float('inf'),float('nan')])
def test_publication_refuses_invalid_deadline_before_qualification(tmp_path,monkeypatch,given):
    monkeypatch.setattr(module,'_perception_source',lambda *a,**k:pytest.fail('Invalid bounds cannot qualify'))
    with pytest.raises(ValueError,match='deadline must be finite'):
        module.publish_perception_source(tmp_path,tmp_path/'bridge',deadline=given)


def test_publication_expired_caller_never_starts_new_thirty_seconds(tmp_path,monkeypatch):
    monkeypatch.setattr(module.time,'monotonic',lambda:100.)
    monkeypatch.setattr(module,'_perception_source',lambda *a,**k:pytest.fail('Expired caller cannot qualify'))
    with pytest.raises(TimeoutError):module.publish_perception_source(tmp_path,tmp_path/'bridge',deadline=99.)
    assert not (tmp_path/module.JOB_SOURCE_PATH).exists()


def test_publication_rechecks_deadline_before_immutable_write(tmp_path,monkeypatch):
    clock=[100.]
    monkeypatch.setattr(module.time,'monotonic',lambda:clock[0])
    def qualification(*args,**kwargs):
        clock[0]=106.;return {'fixture':'qualified'},None,None
    monkeypatch.setattr(module,'_perception_source',qualification)
    with pytest.raises(TimeoutError):module.publish_perception_source(tmp_path,tmp_path/'bridge',deadline=105.)
    assert not (tmp_path/module.JOB_SOURCE_PATH).exists()
