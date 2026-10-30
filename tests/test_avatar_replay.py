"""Offline export custody and injection checks; no GPU, simulation or browser is invoked."""

import base64
from copy import deepcopy
import hashlib
from html.parser import HTMLParser
import json
import re
from types import SimpleNamespace

import pytest

from weft import avatar_replay as replay
from weft.avatar_lab import DEFAULT_APPEARANCE


def fixture():
    key = 'a' * 64
    texture = b'authored texture bytes; exporter checks identity, not image decoding'
    render = {'profile': 'weft-avatar-render-v1', 'textures': [{'path': 'textures/authored.png',
        'mime_type': 'image/png', 'sha256': hashlib.sha256(texture).hexdigest()}]}
    raw = json.dumps(render).encode()
    package = SimpleNamespace(source_manifest_sha256=key, manifest={'asset_id': 'authored', 'display_name': 'Authored panda',
        'rights': {'declaration': 'Synthetic test only'}, 'provenance': {'scope': 'not a real mesh'}}, render_json=raw,
        render_sha256=hashlib.sha256(raw).hexdigest(), texture_files={'textures/authored.png': texture})
    scene = {'profile': 'shared-exercise-recording-v1', 'engine_scene': {'profile': 'shared-world-v2', 'bounds': [2., 2., 2.]}, 'exercise_task': {'unchanged': True}}
    events = [{'type': kind, 'seed': 7, 'snapshot': {'time_s': index / 30, 'objects': [], 'human': {}}, 'index': index,
               'retained_extra': {'note': 'original field', 'actions': [0., .25]}} for index, kind in enumerate(['start', 'step', 'reset'])]
    changes = []
    for index, (anchor, selected) in enumerate([(0, None), (1, key), (1, None), (2, key)]):
        settings = deepcopy(DEFAULT_APPEARANCE); settings['asset'] = selected; settings['camera']['yaw'] = .1 * index
        changes.append({'index': index, 'after_event_index': anchor,
            'after_event_sha256': hashlib.sha256((replay._encoded(events[anchor]) + '\n').encode()).hexdigest(), 'settings': settings})
    files = {'avatar_retarget.js': b'globalThis.WeftAvatarRetarget={};',
        'avatar_worker.js': b'if(!globalThis.WeftAvatarRetarget)importScripts("/avatar_retarget.js");',
        'avatar_viewer.js': b'globalThis.WeftAvatarViewer={};'}
    return scene, events, changes, {key: package}, files


def payload(document):
    return json.loads(re.search(r'<script type="application/json" id="avatar-replay-data">(.*?)</script>', document, re.S).group(1))


def test_self_contained_export_retains_all_events_appearance_and_frozen_bytes():
    args = fixture(); frozen = deepcopy(args)
    document = replay.build_avatar_replay(*args); data = payload(document)
    assert args == frozen and data['scene'] == args[0] and data['events'] == args[1] and data['changes'] == args[2]
    key = next(iter(args[3])); package = args[3][key]
    assert base64.b64decode(data['assets'][key]['render_base64']) == package.render_json
    assert base64.b64decode(data['assets'][key]['textures_base64'][0]) == package.texture_files['textures/authored.png']
    assert data['assets'][key]['manifest'] == package.manifest
    assert base64.b64decode(data['worker_base64']) == args[4]['avatar_worker.js']
    assert data['integrity']['events_sha256'] == hashlib.sha256(''.join(replay._encoded(e) + '\n' for e in args[1]).encode()).hexdigest()
    assert 'not a screen recording' in document and 'no new physics runs' in document


class Elements(HTMLParser):
    def __init__(self): super().__init__(); self.tags = []; self.meta = {}
    def handle_starttag(self, tag, attrs):
        self.tags.append(tag)
        if tag == 'meta': self.meta.update(dict(attrs))


def test_untrusted_labels_and_events_cannot_end_the_json_script():
    scene, events, changes, packages, files = fixture()
    attack = '</ScRiPt><script>alert("xss")</script><img src=x onerror=alert(1)>&\u2028'
    scene['title'] = attack; events[0]['retained_extra']['note'] = attack
    for change in changes: change['after_event_sha256'] = hashlib.sha256((replay._encoded(events[change['after_event_index']]) + '\n').encode()).hexdigest()
    next(iter(packages.values())).manifest['display_name'] = attack
    document = replay.build_avatar_replay(scene, events, changes, packages, files)
    parsed = Elements(); parsed.feed(document)
    assert parsed.tags.count('script') == 4 and 'img' not in parsed.tags
    assert payload(document)['events'][0]['retained_extra']['note'] == attack
    assert '<ScRiPt>' not in document and '\\u003c/ScRiPt' in document


def test_csp_hashes_exact_scripts_and_style_without_network_or_eval_permissions():
    document = replay.build_avatar_replay(*fixture()); parsed = Elements(); parsed.feed(document)
    csp = parsed.meta['content']
    assert "default-src 'none'" in csp and "connect-src 'none'" in csp and 'worker-src blob:' in csp
    assert 'unsafe-inline' not in csp and 'unsafe-eval' not in csp and 'https:' not in csp
    for code in re.findall(r'<script id="[^"]+">(.*?)</script>', document, re.S):
        assert "'sha256-" + base64.b64encode(hashlib.sha256(code.encode()).digest()).decode() + "'" in csp
    style = re.search(r'<style>(.*?)</style>', document, re.S).group(1)
    assert "style-src 'sha256-" + base64.b64encode(hashlib.sha256(style.encode()).digest()).decode() + "'" in csp
    assert 'script src=' not in document and 'Offline replay forbids imported scripts' in document


def test_missing_recorded_asset_remains_selected_for_explicit_fallback():
    args = list(fixture()); key = next(iter(args[3])); args[3] = {}
    data = payload(replay.build_avatar_replay(*args))
    assert data['assets'] == {} and data['changes'][-1]['settings']['asset'] == key


@pytest.mark.parametrize('change', ['event', 'anchor', 'order', 'hash', 'first', 'boolean_index'])
def test_physical_or_appearance_identity_mismatch_rejects(change):
    args = list(fixture()); events, changes = args[1:3]
    if change == 'event': events[1]['retained_extra']['actions'][0] = .5
    if change == 'anchor': changes[-1]['after_event_index'] = len(events)
    if change == 'order': changes[-1]['after_event_index'] = 0
    if change == 'hash': changes[0]['after_event_sha256'] = '0' * 64
    if change == 'first': changes[0]['after_event_index'] = 1
    if change == 'boolean_index': events[1]['index'] = True
    with pytest.raises(ValueError): replay.build_avatar_replay(*args)


@pytest.mark.parametrize('target', ['render', 'texture', 'source'])
def test_asset_tampering_rejects_before_export(target):
    args = fixture(); package = next(iter(args[3].values()))
    if target == 'render': package.render_json += b' '
    if target == 'texture': package.texture_files['textures/authored.png'] += b'changed'
    if target == 'source': package.source_manifest_sha256 = 'b' * 64
    with pytest.raises(ValueError, match='identity'): replay.build_avatar_replay(*args)


@pytest.mark.parametrize('name,value', [('MAX_EVENTS', 2), ('MAX_CHANGES', 2), ('MAX_HTML_BYTES', 1024), ('MAX_EVENT_BYTES', 10), ('MAX_RENDER_BYTES', 10)])
def test_finite_bounds_never_trim_the_physical_recording(monkeypatch, name, value):
    args = fixture(); original = deepcopy(args[1:3]); monkeypatch.setattr(replay, name, value)
    with pytest.raises(ValueError): replay.build_avatar_replay(*args)
    assert args[1:3] == original


def test_nonfinite_json_and_missing_frozen_runtime_fail_explicitly():
    args = list(fixture()); args[0]['invalid'] = float('nan')
    with pytest.raises(ValueError): replay.build_avatar_replay(*args)
    args = list(fixture()); args[4].pop('avatar_worker.js')
    with pytest.raises(ValueError, match='three frozen viewer'): replay.build_avatar_replay(*args)


def test_runtime_html_delimiter_cannot_break_out_of_hash_authorized_script():
    args = fixture(); args[4]['avatar_viewer.js'] = b'// </script><script>attack()</script>'
    with pytest.raises(ValueError, match='unsafe HTML delimiter'): replay.build_avatar_replay(*args)
