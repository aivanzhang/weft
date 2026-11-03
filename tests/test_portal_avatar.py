"""Avatar delivery bounds and actual body-journal parity; no physical fidelity claim."""
from copy import deepcopy
from io import BytesIO
import json
from pathlib import Path
from zipfile import ZipFile, ZipInfo

import pytest

from test_avatar_assets import factory, metadata
from test_avatar_lab import imported, look
from test_shared_exercise_lab import prepared
from test_shared_world_human import analytical_body, body, inventory
from weft.avatar_assets import load_imported_avatar
from weft.avatar_lab import AvatarCatalog
from weft.avatar_recording import load_appearance
from weft.io import digest
from weft.lab import Lab
from weft.portal_avatar import import_avatar_zip, prepare_builtin_avatars, PortalAvatarLab
from weft.recording import load_recording


def archive(payload):
    stream = BytesIO()
    with ZipFile(stream, 'w') as z:
        for key, value in payload.items(): z.writestr(key, value)
    return stream.getvalue()


def test_zip_real_import_preserves_delivery_and_rejects_overwrite(tmp_path):
    source, files = factory(tmp_path, glb=True)
    raw = archive({'model.glb': files['model.glb'], 'mapping.json': json.dumps(metadata()).encode()})
    receipt = import_avatar_zip(raw, tmp_path / 'import')
    package = load_imported_avatar(receipt['asset']['directory'])
    assert package.originals['model.glb'] == files['model.glb']
    assert package.manifest['rights'] == metadata()['rights']
    with pytest.raises(FileExistsError): import_avatar_zip(raw, tmp_path / 'import')


@pytest.mark.parametrize('payload', [
    {'../model.glb': b'a', 'mapping.json': b'{}'},
    {'model.glb': b'a'},
    {'model.glb': b'a', 'mapping.json': b'{}', 'run.py': b'pass'},
    {'model.glb': b'a', 'mapping.json': b'x' * (128 * 1024 + 1)},
])
def test_zip_closed_inventory_rejected_before_files(tmp_path, payload):
    with pytest.raises(ValueError): import_avatar_zip(archive(payload), tmp_path / 'unused')
    assert not (tmp_path / 'unused').exists()


def test_zip_symlink_and_duplicate_rejected(tmp_path):
    for link in (True, False):
        stream = BytesIO()
        with ZipFile(stream, 'w') as z:
            info = ZipInfo('model.glb'); info.external_attr = (0o120777 if link else 0o100644) << 16
            z.writestr(info, 'target'); z.writestr('mapping.json', '{}')
            if not link:
                with pytest.warns(UserWarning): z.writestr('mapping.json', '{}')
        with pytest.raises(ValueError): import_avatar_zip(stream.getvalue(), tmp_path / str(link))


def test_builtin_delivery_v2_retains_originals_and_all_ten_roles(tmp_path):
    project = Path(__file__).resolve().parents[1]
    results = prepare_builtin_avatars(tmp_path / 'builtin', project=project)
    catalog = AvatarCatalog([row['directory'] for row in results.values()])
    assert len(catalog.packages) == 2
    for package in catalog.packages.values():
        mapping = package.manifest['retargeting']
        assert mapping['profile'] == 'weft-avatar-retarget-v2'
        assert len(mapping['bindings']) == len({row['body'] for row in mapping['bindings']}) == 10
        assert all(set(row) == {'node', 'body'} for row in mapping['bindings'])
        assert package.manifest['provenance']['weft_delivery_bridge']['source_model_sha256'] == digest(
            Path(results['muse-panda' if 'panda' in package.manifest['asset_id'] else 'robot']['directory']).parent / 'input/model.glb')


def compare_journals(target, avatar):
    ordinary = Lab(target)
    physical = Lab(target)
    wrapped = PortalAvatarLab(physical, AvatarCatalog([avatar]))
    before = {p.relative_to(target / 'run').as_posix(): digest(p) for p in (target / 'run').rglob('*') if p.is_file()}
    try:
        records = []
        for lab in (ordinary, wrapped):
            lab.control('/api/reset', {'seed': 37})
            records.append(lab.control('/api/record/start', {}))
        assert records[0]['snapshot'] == records[1]['snapshot']
        key = next(iter(wrapped.avatars.packages))
        for i in range(3):
            prior = deepcopy(physical.session.snapshot())
            wrapped.control('/api/appearance', look(key if i != 1 else None))
            assert physical.session.snapshot() == prior
            count = len(physical.session.action_contract()['names'])
            action = [.1] + [0.] * (count - 1)
            assert ordinary.control('/api/step', {'action': action}) == wrapped.control('/api/step', {'action': action})
        for lab in (ordinary, wrapped): lab.recorder.close()
        wrapped.recorder.asset_future.result(timeout=10)
        roots = [target / 'sessions' / record['directory'] for record in records]
        assert (roots[0] / 'events.jsonl').read_bytes() == (roots[1] / 'events.jsonl').read_bytes()
        assert (roots[0] / 'scene.json').read_bytes() == (roots[1] / 'scene.json').read_bytes()
        assert len(load_appearance(roots[1], load_recording(roots[1]))['changes']) == 4
        assert before == {p.relative_to(target / 'run').as_posix(): digest(p) for p in (target / 'run').rglob('*') if p.is_file()}
    finally:
        ordinary.close(); wrapped.close()


def test_existing_ordinary_lab_exact_physical_journal_parity(tmp_path):
    avatar = imported(tmp_path / 'asset')
    (tmp_path / 'physics').mkdir()
    target, _, _ = prepared(tmp_path / 'physics')
    compare_journals(target, avatar)


def test_source_person_only_v3_exact_physical_journal_parity(body, tmp_path, monkeypatch):
    from weft.shared_world_human import construct_human_only_run
    from weft.shared_exercise_lab import prepare_shared_exercise_lab
    avatar = imported(tmp_path / 'asset')
    construct_human_only_run(body, inventory(body), tmp_path / 'world')
    prepare_shared_exercise_lab(tmp_path / 'world', tmp_path / 'lab')
    assert json.loads((tmp_path / 'lab/run/scene.json').read_bytes())['profile'] == 'shared-world-v3'
    monkeypatch.undo()  # Construction alone forbids physics; journal parity deliberately executes controls.
    compare_journals(tmp_path / 'lab', avatar)


def test_active_recording_and_unsupported_profile_cannot_rebind_catalog(tmp_path):
    from types import SimpleNamespace
    catalog = AvatarCatalog()
    session = SimpleNamespace(scene={'profile': 'shared-world-v3'}, control_profile='shared_exercise')
    lab = SimpleNamespace(load_session=lambda: session, recorder=object(), export_future=None)
    with pytest.raises(ValueError, match='Stop and export'): PortalAvatarLab(lab, catalog)
    lab.recorder = None
    session.control_profile = 'shared_world_exploration'
    with pytest.raises(ValueError, match='ordinary'): PortalAvatarLab(lab, catalog)
    session.control_profile = 'shared_exercise'; session.scene['profile'] = 'shared-world-v6'
    with pytest.raises(ValueError, match='ordinary'): PortalAvatarLab(lab, catalog)


def test_duplicate_record_start_and_override_do_not_reset_session(tmp_path):
    from types import SimpleNamespace
    catalog = AvatarCatalog([imported(tmp_path / 'asset')])
    session = SimpleNamespace(scene={'profile': 'shared-world-v3'}, control_profile='shared_exercise',
                              reset=lambda *args: pytest.fail('must reject before reset'))
    lab = SimpleNamespace(load_session=lambda: session, recorder=None, export_future=None)
    wrapped = PortalAvatarLab(lab, catalog)
    with pytest.raises(ValueError, match='no overrides'): wrapped.control('/api/record/start', {'seed': 3})
    lab.recorder = object()
    with pytest.raises(ValueError, match='already active'): wrapped.control('/api/record/start', {})


@pytest.mark.parametrize('seconds', [True, float('nan'), 10**400, 0, 61])
def test_invalid_time_bounds_fail_before_input_or_files(tmp_path, seconds):
    with pytest.raises(ValueError, match='finite bound'):
        import_avatar_zip(b'', tmp_path / 'unused', max_seconds=seconds)
    assert not (tmp_path / 'unused').exists()


def test_offline_avatar_replay_preserves_person_only_v3_engine_and_task():
    from test_avatar_replay import fixture, payload
    from weft.avatar_replay import build_avatar_replay
    args=list(fixture());args[0]['engine_scene']['profile']='shared-world-v3';args[0]['engine_scene']['objects']=[]
    original=deepcopy(args[:3])
    saved=payload(build_avatar_replay(*args))
    assert [saved['scene'],saved['events'],saved['changes']]==original
    assert saved['scene']['exercise_task']==args[0]['exercise_task']


def test_offline_replay_accepts_valid_large_imported_render_but_not_geometry_growth(tmp_path):
    from dataclasses import replace
    from test_avatar_replay import fixture, payload
    from weft.avatar_replay import build_avatar_replay
    results=prepare_builtin_avatars(tmp_path/'builtin',project=Path(__file__).resolve().parents[1])
    package=load_imported_avatar(results['muse-panda']['directory'])
    args=list(fixture());args[3]={package.source_manifest_sha256:package}
    saved=payload(build_avatar_replay(*args))
    assert package.source_manifest_sha256 in saved['assets']
    bad=deepcopy(package.render);bad['primitives'][0]['positions']=[0.]*60003
    raw=json.dumps(bad,sort_keys=True,separators=(',',':')).encode()
    malicious=replace(package,render=bad,render_json=raw)
    with pytest.raises(ValueError,match='geometry exceeds'):
        build_avatar_replay(args[0],args[1],args[2],{malicious.source_manifest_sha256:malicious},args[4])
    changed=deepcopy(package.render);changed['scope']='mutated detached render'
    with pytest.raises(ValueError,match='verified render bytes'):
        build_avatar_replay(args[0],args[1],args[2],{package.source_manifest_sha256:replace(package,render=changed)},args[4])
