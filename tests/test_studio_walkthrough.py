"""Presentation exports preserve evidence and never grant live/private access."""
import hashlib
import json
from pathlib import Path

import pytest

from weft.portal_demo import GALLERY_SEEDS
from weft.portal_studio import PUBLIC_FILES, public_file, recorded_model

ROOT = Path(__file__).resolve().parents[1]


def test_curated_walkthroughs_reference_only_existing_allowlisted_assets():
    for identity in [row['id'] for row in GALLERY_SEEDS] + ['waiters-squat-panda', 'waiters-squat-created']:
        data = json.loads(public_file(ROOT, identity + '.json').read_text())
        assert public_file(ROOT, data['model']).is_file()
        assert data['steps'][0]['id'] == 'intake'
        assert len({s['id'] for s in data['steps']}) == len(data['steps'])
        for step in data['steps']:
            assert step['title'] and step['description']
            for asset in step['assets']:
                assert public_file(ROOT, asset['name']).stat().st_size <= 2_000_000
    assert {p.name for p in (ROOT / 'assets/demo/studio').iterdir()} == PUBLIC_FILES
    for path in (ROOT / 'assets/demo/studio').glob('*.json'):
        text = path.read_text()
        assert '/Users/' not in text and '.local/' not in text and 'token=' not in text


def test_public_assets_refuse_unknown_names_and_symlinked_parents(tmp_path):
    with pytest.raises(ValueError):
        public_file(ROOT, '../../../.local/secret.json')
    with pytest.raises(ValueError):
        public_file(ROOT, 'secret.json')
    (tmp_path / 'assets/demo').mkdir(parents=True)
    (tmp_path / 'assets/demo/studio').symlink_to(ROOT / 'assets/demo/studio', target_is_directory=True)
    with pytest.raises(ValueError):
        public_file(tmp_path, 'body-model.json')


def test_skeleton_preview_retains_terminal_frame_and_source_hash(tmp_path):
    path = tmp_path / 'events.jsonl'
    events = [{'snapshot': {'engine_state': {'time_s': i / 30,
               'human': {'segments': [{'a': [0, 0, 0], 'b': [1, 0, i / 1000]}]}}}} for i in range(700)]
    path.write_text(''.join(json.dumps(event) + '\n' for event in events))
    preview = recorded_model(path)
    assert preview['source_frames'] == 700
    assert len(preview['frames']) <= 361
    assert preview['frames'][-1]['segments'][0][1][2] == .699
    assert preview['source_sha256'] == hashlib.sha256(path.read_bytes()).hexdigest()
    events[0]['snapshot']['engine_state']['human']['segments'][0]['a'][0] = float('nan')
    path.write_text(json.dumps(events[0]) + '\n')
    with pytest.raises(ValueError, match='Invalid recorded body'):
        recorded_model(path)


def test_exported_body_preview_and_outcome_share_the_same_evidence():
    for prefix in ['body', 'character']:
        model = json.loads(public_file(ROOT, prefix + '-model.json').read_text())
        outcome = json.loads(public_file(ROOT, prefix + '-outcome.json').read_text())
        assert model['source_sha256'] == outcome['source_sha256']
        assert model['frames'][-1]['time'] == pytest.approx(outcome['metrics']['simulated_duration_s'], abs=1e-5)
        assert model['source_frames'] == 246
        assert outcome['metrics']['success'] is True


def test_motion_studios_are_bound_to_the_matching_video_reference():
    for seed in GALLERY_SEEDS:
        if seed['id'] == 'rolling-motion':
            continue
        identity = seed['id']
        model = json.loads(public_file(ROOT, identity + '-model.json').read_text())
        receipt = json.loads((ROOT / 'assets/demo/motion-showcase' / identity / 'receipt.json').read_text())
        assert model['source_sha256'] == receipt['reference_sha256']
        assert model['duration'] == receipt['duration_s']
        assert model['frames'][0]['time'] == 0
        assert model['frames'][-1]['time'] >= model['duration'] - .00001
        assert model['playback_kind'] == 'kinematic reference'


def test_private_recorded_models_are_contained_in_the_owned_collection(tmp_path):
    from weft.portal_demo import DemoManager
    manager = DemoManager(tmp_path / 'collection', project=ROOT, token='test')
    session = manager.root / 'owned-session'
    session.mkdir()
    event = {'snapshot': {'engine_state': {'time_s': 0, 'human': {'segments': [{'a': [0, 0, 0], 'b': [1, 0, 0]}]}}}}
    (session / 'events.jsonl').write_text(json.dumps(event) + '\n')
    saved = manager.dir / 'run-abc'
    saved.mkdir()
    record = {'id': 'run-abc', 'video': str(session / 'video.mp4')}
    (saved / 'record.json').write_text(json.dumps(record))
    assert manager.studio_model('run-abc')['source_frames'] == 1
    outside = tmp_path / 'outside'
    outside.mkdir()
    (outside / 'events.jsonl').write_text(json.dumps(event) + '\n')
    record['video'] = str(outside / 'video.mp4')
    (saved / 'record.json').write_text(json.dumps(record))
    with pytest.raises(ValueError, match='not available'):
        manager.studio_model('run-abc')


def test_stage_reference_preview_is_bounded_and_does_not_modify_source(tmp_path):
    from weft.portal_studio import save_reference_preview
    project = 'a' * 32
    folder = tmp_path / project / 'lab/run'
    folder.mkdir(parents=True)
    source = folder / 'human-reference.json'
    frames = [{'time_s': i / 30, 'human': {'segments': [
        {'id': 'torso', 'a': [0, 0, 0], 'b': [1, 0, i / 1000]}]}} for i in range(700)]
    source.write_text(json.dumps({'frames': frames}))
    original = source.read_bytes()
    assets = save_reference_preview(tmp_path, project)
    preview = json.loads((folder / 'studio-reference-preview.json').read_text())
    assert len(preview['frames']) <= 361
    assert preview['frames'][-1]['time'] == frames[-1]['time_s']
    assert preview['segment_ids'] == ['torso']
    assert assets[0]['url'].endswith('/lab/run/studio-reference-preview.json')
    assert source.read_bytes() == original


def test_stage_recordings_are_distinct_and_cannot_escape_the_project(tmp_path):
    from weft.portal_studio import recording_visuals
    project = 'a' * 32
    session = tmp_path / project / 'lab/sessions/session-one'
    session.mkdir(parents=True)
    (session / 'video.mp4').write_bytes(b'physics')
    (session / 'avatar-video.mp4').write_bytes(b'appearance')
    record = {'project': project, 'video': str(session / 'avatar-video.mp4')}
    visuals = recording_visuals(tmp_path, record)
    assert visuals['rollout'][0]['url'] != visuals['avatar'][0]['url']
    assert visuals['rollout'][0]['url'].endswith('/video.mp4')
    (session / 'video.mp4').unlink()
    assert recording_visuals(tmp_path, record)['rollout'] == []
    (session / 'video.mp4').symlink_to(session / 'avatar-video.mp4')
    assert recording_visuals(tmp_path, record)['rollout'] == []
    assert recording_visuals(tmp_path, {**record, 'project': 'b' * 32}) == {}


def test_measured_character_reuse_never_requests_generation(tmp_path, monkeypatch):
    import time
    from weft.portal_demo import DemoManager, _measured_character
    assert _measured_character({'appearance': {'available': True, 'avatars_substituted': ['muse-panda', 'robot']}}) == {'id': 'muse-panda', 'prompt': 'Panda'}
    assert _measured_character({'appearance': {'available': False, 'avatars_substituted': ['muse-panda']}}) is None
    manager = DemoManager(tmp_path / 'portal', ROOT, 'test')
    manager.run = {'created': time.time(), 'steps': manager._blank_steps(),
                   'character': {'id': 'muse-panda', 'prompt': 'Panda'}, 'prompt': 'Panda'}
    monkeypatch.setattr(manager, '_request', lambda *a, **kw: pytest.fail('Measured reuse must not make generation requests'))
    manager._step_create(tmp_path, 'a' * 32, {'Panda': 'existing'})
    step = manager.run['steps'][6]
    assert step['status'] == 'done'
    assert step['metrics']['live generation'] == 'not requested'
    assert step['visuals'][0]['items'][0] == ['Character', 'Panda']


def test_public_video_posters_are_real_decodable_frames():
    from PIL import Image
    for seed in GALLERY_SEEDS:
        for kind in ('source', 'result'):
            path = public_file(ROOT, seed['id'] + '-' + kind + '-poster.jpg')
            with Image.open(path) as image:
                image.verify()
            with Image.open(path) as image:
                assert image.width == 960 and image.height > 100


def test_video_poster_is_reused_without_changing_recorded_video(tmp_path):
    from weft.portal_studio_media import poster
    source = ROOT / GALLERY_SEEDS[0]['video']
    original = hashlib.sha256(source.read_bytes()).hexdigest()
    target = tmp_path / 'poster.jpg'
    assert poster(source, target) == target
    stamp = target.stat().st_mtime_ns
    assert poster(source, target) == target
    assert target.stat().st_mtime_ns == stamp
    assert hashlib.sha256(source.read_bytes()).hexdigest() == original


def test_saved_studio_reads_presentation_without_rewriting_evidence(tmp_path):
    from weft.portal_demo import DemoManager
    from weft.portal_studio_media import read_presentation
    manager = DemoManager(tmp_path / 'portal', ROOT, 'test')
    project = 'a' * 32
    (manager.root / project / 'lab').mkdir(parents=True)
    folder = manager.dir / 'run-face'; folder.mkdir()
    record = {'id': 'run-face', 'project': project, 'steps': [{'id': 'create', 'status': 'done'}]}
    (folder / 'record.json').write_text(json.dumps(record))
    original = (folder / 'record.json').read_bytes()
    presentation = {'poster_url': '/api/projects/' + project + '/files/lab/run/poster.jpg',
                    'steps': {'create': [{'kind': 'image', 'url': '/api/projects/' + project + '/files/lab/run/character.png', 'label': 'Saved character'}],
                              'avatar': [{'kind': 'video', 'url': '/appearance.mp4', 'poster': '/appearance.jpg', 'label': 'Character swap'}]}}
    (folder / 'presentation.json').write_text(json.dumps(presentation))
    studio = manager.studio('run-face')
    assert studio['steps'][0]['visuals'] == presentation['steps']['avatar']
    assert json.loads((folder / 'presentation.json').read_text()) == presentation
    assert studio['poster_url'] == presentation['poster_url']
    assert (folder / 'record.json').read_bytes() == original
    assert read_presentation(manager.root, '../run-face') == {}


def test_character_motion_samples_preserve_clock_and_bound_work(tmp_path):
    from weft.portal_studio_media import _motion_samples
    path = tmp_path / 'events.jsonl'
    events = [{'snapshot': {'engine_state': {'time_s': i / 30}}} for i in range(600)]
    path.write_text(''.join(json.dumps(row) + '\n' for row in events))
    frames, digest = _motion_samples(path)
    assert len(frames) == 240
    assert frames[1]['engine_state']['time_s'] == pytest.approx(1 / 15)
    assert frames[-1]['engine_state']['time_s'] == pytest.approx(239 / 15)
    assert digest == hashlib.sha256(path.read_bytes()).hexdigest()
    path.write_text(json.dumps(events[2]) + '\n' + json.dumps(events[0]) + '\n')
    with pytest.raises(ValueError, match='clock'):
        _motion_samples(path)


def test_character_video_uses_selected_asset_has_poster_and_reuses_cache(tmp_path, monkeypatch):
    import cv2
    import numpy as np
    from types import SimpleNamespace
    from weft import portal_studio_media as media
    from weft import avatar_offscreen
    project = 'b' * 32
    output = tmp_path / project / 'lab/run'; output.mkdir(parents=True)
    session = tmp_path / project / 'lab/sessions/session-test'; session.mkdir(parents=True)
    (session / 'scene.json').write_text('{}')
    events = [{'snapshot': {'engine_state': {'time_s': i / 15}, 'index': i}} for i in range(3)]
    source = ''.join(json.dumps(row) + '\n' for row in events)
    (session / 'events.jsonl').write_text(source)
    package = SimpleNamespace(source_manifest_sha256='c' * 64, render_sha256='d' * 64)
    monkeypatch.setattr(media, '_character_package', lambda *args: (package, None))
    rendered, closed = [], []
    class Renderer:
        def __init__(self, scene, packages, **kwargs):
            assert packages == {'c' * 64: package}
        def render(self, snapshot, settings):
            assert settings['asset'] == 'c' * 64
            rendered.append(snapshot['index'])
            return np.full((560, 1440, 3), snapshot['index'] * 70, dtype=np.uint8)
        def close(self):
            closed.append(True)
    monkeypatch.setattr(avatar_offscreen, 'AvatarOffscreen', Renderer)
    record = {'project': project}
    visuals = media.character_preview(tmp_path, tmp_path, record, session)
    assert rendered == [0, 1, 2] and closed == [True]
    assert visuals[0]['kind'] == 'video' and visuals[0]['poster'].endswith('.jpg')
    video = output / visuals[0]['url'].rsplit('/', 1)[-1]
    cap = cv2.VideoCapture(str(video))
    assert cap.get(cv2.CAP_PROP_FRAME_COUNT) == 3
    assert cap.get(cv2.CAP_PROP_FPS) == 15
    assert cap.read()[0]
    cap.release()
    assert media.character_preview(tmp_path, tmp_path, record, session) == visuals
    assert rendered == [0, 1, 2], 'cached preview must not render again'
    assert (session / 'events.jsonl').read_text() == source
    assert not (session / 'avatar-video.mp4').exists(), 'preview does not replace the scored recording'


@pytest.mark.parametrize('identity,label', [('muse-panda', 'Panda'), ('robot', 'Robot')])
def test_character_preview_resolves_measured_presets_beyond_old_catalog_limit(tmp_path, monkeypatch, identity, label):
    from weft import avatar_assets
    from weft.portal_studio_media import _character_package
    avatars = tmp_path / 'avatars'; avatars.mkdir()
    entries = []
    for index in range(18):
        kind = 'builtin' if index == 17 else 'custom'
        directory = f'operations/{index}/{kind}/{identity}/asset'
        folder = avatars / directory
        (folder / 'originals').mkdir(parents=True)
        (folder / 'originals/avatar.json').write_text(json.dumps({'display_name': label if index == 17 else 'Other'}))
        (folder / 'manifest.json').write_text('{}')
        entries.append({'directory': directory})
    (avatars / 'catalog.json').write_text(json.dumps({'entries': entries}))
    monkeypatch.setattr(avatar_assets, 'load_imported_avatar', lambda folder: folder)
    record = {'character': {'id': identity, 'prompt': label}, 'steps': [{'id': 'create', 'metrics': {'character': label}}]}
    package, folder = _character_package(tmp_path, tmp_path / 'run-unused', record)
    assert package == avatars / entries[-1]['directory'] == folder
