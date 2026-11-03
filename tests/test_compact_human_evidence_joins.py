"""Compact scenes join exact full body bytes; projection never substitutes new evidence."""

from copy import deepcopy
import json
import os
import time

import pytest

from test_portal_evidence_attachment import fixture, raw
from test_static_support_build_report import support_build
from test_exercise_build_report import put, seal
from weft import portal_evidence_attachment as attachment, exercise_build_report as report
from weft.io import digest, write_json


def full_source(source_sha256):
    from test_shared_human_projection import full_human
    return full_human(source_sha256=source_sha256)


def projected(full):
    from weft.shared_human_projection import create_projection
    value=raw(full)
    compact,descriptor=create_projection(value)
    return value,compact,descriptor


def install_portal(fixture):
    source,root,folder,scene,observations,world=fixture
    human=full_source(world['source_sha256']);full_bytes,compact,descriptor=projected(human)
    (source/'human.json').write_bytes(full_bytes)
    scene['human_spec']=compact;scene['human_provenance']=descriptor
    write_json(folder/'lab/run/scene.json',scene);(folder/'lab/run/human.json').write_bytes(full_bytes)
    world['input_sha256']['scene.json']=digest(folder/'lab/run/scene.json')
    files={p.name:p.read_bytes() for p in source.iterdir() if p.is_file()}
    return folder,scene,observations,world,files,human


def test_muse_compact_join_uses_original_full_body_and_preserves_scope(fixture):
    folder,scene,observations,world,files,human=install_portal(fixture)
    before=deepcopy((files,human,scene))
    joined,document=attachment._join(folder,files,human,observations,time.monotonic()+10)
    assert joined['source_sha256']==human['source_sha256'] and document
    assert (files,human,scene)==before
    assert scene['human_spec']!=human


@pytest.mark.parametrize('damage',['angle','descriptor','raw_bytes','full_body','missing_descriptor'])
def test_muse_compact_join_rejects_new_body_or_unbound_projection(fixture,damage):
    folder,scene,observations,world,files,human=install_portal(fixture)
    if damage=='angle':
        joints=scene['human_spec']['reference'][0]['joints'];joints[next(iter(joints))]+=.001
    elif damage=='descriptor':scene['human_provenance']['full_human_sha256']='0'*64
    elif damage=='raw_bytes':files['human.json']+=b' '
    elif damage=='full_body':human['dt']+=.001
    else:scene.pop('human_provenance')
    write_json(folder/'lab/run/scene.json',scene);world['input_sha256']['scene.json']=digest(folder/'lab/run/scene.json')
    with pytest.raises(ValueError):attachment._join(folder,files,human,observations,time.monotonic()+10)


def install_report(tmp_path):
    root=tmp_path/'build';support_build(root,status='rejected')
    scene=json.loads((root/'candidate/scene.json').read_bytes())
    source_sha=scene['source_binding']['source_sha256']
    full_bytes,compact,descriptor=projected(full_source(source_sha))
    put(root,'body_source/human.json',full_bytes);put(root,'candidate/human.json',full_bytes)
    scene['human_spec']=compact;scene['human_provenance']=descriptor;put(root,'candidate/scene.json',scene)
    seal(root)
    return root,scene,source_sha


def summarize(root,source_sha):
    source=report._Source(root,time.monotonic()+10)
    try:return report._static_support_summary(source,json.loads((root/'settings.json').read_bytes()),source_sha,'body_source')
    finally:os.close(source.fd)


def test_offline_support_report_rejects_unsupported_projection_without_native_imports(tmp_path,monkeypatch):
    import builtins
    root,scene,source_sha=install_report(tmp_path)
    original=builtins.__import__
    def guarded(name,*args,**kwargs):
        if name.split('.')[0] in {'mujoco','gymnasium','cv2','mediapipe','torch'}:pytest.fail('Offline join cannot import native runtimes')
        return original(name,*args,**kwargs)
    monkeypatch.setattr(builtins,'__import__',guarded)
    with pytest.raises(ValueError, match='unsupported for static-support'):
        summarize(root,source_sha)


