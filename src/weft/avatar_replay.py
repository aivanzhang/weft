"""Self-contained appearance replay; every action and state remains recorded evidence."""

import base64
import hashlib
import html
import io
import json
from pathlib import Path
import re

from weft.avatar_gltf import json_document
from weft.avatar_lab import validate_appearance


MAX_HTML_BYTES = 256 * 1024 * 1024
MAX_EVENTS, MAX_CHANGES, MAX_PACKAGES = 18001, 1024, 4
MAX_EVENT_BYTES, MAX_JOURNAL_BYTES = 65536, 128 * 1024 * 1024
MAX_SCENE_BYTES, MAX_RENDER_BYTES, MAX_TEXTURE_BYTES = 8 * 1024 * 1024, 8 * 1024 * 1024, 32 * 1024 * 1024
MAX_SCRIPT_BYTES = 512 * 1024
VIEWER_FILES = ('avatar_retarget.js', 'avatar_worker.js', 'avatar_viewer.js')
SHA = re.compile(r'[0-9a-f]{64}\Z')


def _encoded(value):
    return json.dumps(value, allow_nan=False, separators=(',', ':'))


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _script(raw, name):
    if type(raw) is not bytes or not 0 < len(raw) <= MAX_SCRIPT_BYTES:
        raise ValueError('Replay runtime must contain bounded frozen bytes: ' + name)
    code = raw.decode('utf-8')
    if re.search(r'</script|<!--|-->', code, re.IGNORECASE):
        raise ValueError('Frozen replay script contains an unsafe HTML delimiter: ' + name)
    return code


def _asset(key, package):
    if not isinstance(key, str) or not SHA.fullmatch(key) or package.source_manifest_sha256 != key:
        raise ValueError('Replay avatar source identity differs')
    raw = package.render_json
    if type(raw) is not bytes or not 0 < len(raw) <= MAX_RENDER_BYTES or _sha(raw) != package.render_sha256:
        raise ValueError('Replay avatar render identity or bound differs')
    render = json_document(raw, maximum=MAX_RENDER_BYTES)
    textures = render.get('textures')
    if render.get('profile') != 'weft-avatar-render-v1' or not isinstance(textures, list) or len(textures) > 8:
        raise ValueError('Replay requires a normalized bounded avatar render document')
    encoded, total = [], 0
    for texture in textures:
        if not isinstance(texture, dict) or texture.get('mime_type') not in ('image/png', 'image/jpeg'):
            raise ValueError('Replay texture type is unsupported')
        content = package.texture_files.get(texture.get('path'))
        if type(content) is not bytes or not content or _sha(content) != texture.get('sha256'):
            raise ValueError('Replay avatar texture identity differs')
        total += len(content)
        if total > MAX_TEXTURE_BYTES: raise ValueError('Replay avatar texture byte bound exceeded')
        encoded.append(base64.b64encode(content).decode('ascii'))
    manifest = package.manifest
    if not isinstance(manifest, dict) or len(_encoded(manifest).encode()) > 128 * 1024:
        raise ValueError('Replay avatar source manifest exceeds its bound')
    return {'source_manifest_sha256': key, 'manifest': manifest,
        'display_name': str(manifest.get('display_name', 'Recorded avatar'))[:128],
        'render_sha256': package.render_sha256, 'render_base64': base64.b64encode(raw).decode('ascii'),
        'textures_base64': encoded}


STYLE = """html{color-scheme:light;background:#F5EADC;color:#214F50;font:16px system-ui,sans-serif}body{max-width:1200px;margin:auto;padding:20px}h1{font-size:1.7rem;margin-bottom:8px}p{line-height:1.45}button,input{font:inherit}button{border:1px solid #965644;border-radius:7px;padding:9px 13px;background:#fff9f2;color:#214F50;cursor:pointer}button:disabled{opacity:.5;cursor:default}#play{background:#965644;color:white}button:focus-visible,input:focus-visible,summary:focus-visible{outline:3px solid #214F50;outline-offset:3px}.controls,.camera{display:flex;gap:10px;align-items:center;flex-wrap:wrap;margin:14px 0}.controls label{flex:1;min-width:220px}#scrub{width:100%}.camera label{display:flex;gap:7px;align-items:center}.camera input[type=range]{width:140px}.stage{border:1px solid #cdbba8;border-radius:9px;overflow:hidden;background:#fff9f2}canvas{display:block;width:100%;height:auto}#event-label{font-weight:600}#appearance-status{min-height:1.5em}small,.muted{color:#5d6967}pre{overflow:auto;max-height:380px;padding:14px;background:#fff9f2;border:1px solid #cdbba8;white-space:pre-wrap;overflow-wrap:anywhere}details{margin:16px 0}summary{cursor:pointer}noscript{display:block;padding:16px;background:#fff9f2}[hidden]{display:none!important}@media(max-width:620px){body{padding:12px}.camera input[type=range]{width:110px}}"""


def build_avatar_replay(scene, events, changes, packages_by_hash, viewer_files) -> str:
    """Embed exact physical events, their anchored appearance changes and pinned local render code."""
    engine = scene.get('engine_scene', scene) if isinstance(scene, dict) else {}
    if not isinstance(engine, dict) or engine.get('profile') not in ('shared-world-v1', 'shared-world-v2'):
        raise ValueError('Avatar replay requires a shared-world recording')
    if len(_encoded(scene).encode()) > MAX_SCENE_BYTES:
        raise ValueError('Replay scene exceeds its byte bound')
    if not isinstance(events, list) or not 1 <= len(events) <= MAX_EVENTS:
        raise ValueError('Replay requires one to 18001 physical events')
    event_hashes, total = [], 0
    journal_hash = hashlib.sha256()
    for index, event in enumerate(events):
        if (not isinstance(event, dict) or type(event.get('index')) is not int or event['index'] != index or
                event.get('type') not in ('start', 'step', 'reset', 'simulation_failure') or
                not isinstance(event.get('snapshot'), dict) or (index == 0 and event['type'] != 'start')):
            raise ValueError('Replay physical event identity/order is invalid')
        raw = (_encoded(event) + '\n').encode()
        total += len(raw)
        if len(raw) > MAX_EVENT_BYTES or total > MAX_JOURNAL_BYTES:
            raise ValueError('Replay physical journal exceeds its byte bound')
        event_hashes.append(_sha(raw)); journal_hash.update(raw)
    if not isinstance(changes, list) or not 1 <= len(changes) <= MAX_CHANGES or len(_encoded(changes).encode()) > 1024 * 1024:
        raise ValueError('Replay appearance journal exceeds its bound')
    previous = -1
    for index, change in enumerate(changes):
        if not isinstance(change, dict) or set(change) != {'index', 'after_event_index', 'after_event_sha256', 'settings'}:
            raise ValueError('Replay appearance change fields differ')
        anchor = change['after_event_index']
        if (type(change['index']) is not int or change['index'] != index or type(anchor) is not int or
                not 0 <= anchor < len(events) or anchor < previous or (index == 0 and anchor != 0) or
                change['after_event_sha256'] != event_hashes[anchor]):
            raise ValueError('Replay appearance change is not bound to its physical event')
        asset = change['settings'].get('asset') if isinstance(change['settings'], dict) else None
        if asset is not None and (not isinstance(asset, str) or not SHA.fullmatch(asset)):
            raise ValueError('Replay appearance asset identity is invalid')
        # A missing recorded asset is an explicit fallback, never a substitute appearance.
        validate_appearance(change['settings'], {asset: True} if asset is not None else {})
        previous = anchor
    if not hasattr(packages_by_hash, 'items') or len(packages_by_hash) > MAX_PACKAGES:
        raise ValueError('Replay supports at most four avatar packages')
    if set(viewer_files) != set(VIEWER_FILES):
        raise ValueError('Replay requires all three frozen viewer files, including for null selection')
    scripts = {name: _script(viewer_files[name], name) for name in VIEWER_FILES}
    controller_raw = (Path(__file__).parent / 'assets/avatar_replay.js').read_bytes()
    controller = _script(controller_raw, 'avatar_replay.js')
    assets = {key: _asset(key, value) for key, value in packages_by_hash.items()}
    payload = {'profile': 'weft-offline-avatar-replay-v1', 'scene': scene, 'events': events, 'changes': changes, 'assets': assets,
        'worker_base64': base64.b64encode(viewer_files['avatar_worker.js']).decode('ascii'),
        'integrity': {'events_sha256': journal_hash.hexdigest(), 'appearance_changes_sha256': _sha(_encoded(changes).encode()),
            'viewer_sha256': {name: _sha(raw) for name, raw in viewer_files.items()}, 'controller_sha256': _sha(controller_raw)},
        'scope': 'Selected appearance reconstructed from saved states, not a screen recording; client GPU fallback history is not captured. Actions, outcomes and simulation states are unchanged.'}
    stream, size = io.StringIO(), 0
    for chunk in json.JSONEncoder(allow_nan=False, separators=(',', ':')).iterencode(payload):
        safe = chunk.replace('&', '\\u0026').replace('<', '\\u003c').replace('>', '\\u003e')
        size += len(safe.encode())
        if size > MAX_HTML_BYTES - 2 * 1024 * 1024: raise ValueError('Offline avatar replay exceeds its HTML byte bound')
        stream.write(safe)
    data = stream.getvalue()
    code = [('avatar-replay-math', scripts['avatar_retarget.js']), ('avatar-replay-viewer', scripts['avatar_viewer.js']),
            ('avatar-replay-controller', controller)]
    hashes = ' '.join("'sha256-" + base64.b64encode(hashlib.sha256(text.encode()).digest()).decode() + "'" for _, text in code)
    style_hash = base64.b64encode(hashlib.sha256(STYLE.encode()).digest()).decode()
    csp = "default-src 'none'; script-src " + hashes + "; style-src 'sha256-" + style_hash + "'; worker-src blob:; connect-src 'none'; img-src blob: data:; object-src 'none'; base-uri 'none'; form-action 'none'; frame-src 'none'"
    result = '''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><meta http-equiv="Content-Security-Policy" content="''' + html.escape(csp, quote=True) + '''"><title>Weft recorded avatar replay</title><style>''' + STYLE + '''</style></head><body>
<h1>Recorded avatar replay</h1><p>Selected appearance reconstructed from saved states—not a screen recording. The live browser may have shown an original-body fallback. Saved actions, world and time are unchanged; no new physics runs.</p>
<p id="load-status" role="status">Preparing the saved recording…</p>
<div class="controls"><button id="previous" disabled aria-label="Previous physical event">Previous</button><button id="play" disabled aria-pressed="false">Play</button><button id="next" disabled aria-label="Next physical event">Next</button><button id="next-reset" disabled>Next reset</button><label for="scrub">Physical event <input id="scrub" type="range" min="0" max="0" value="0" step="1" disabled></label></div>
<p id="event-label"></p><p id="outcome"></p><p id="controller-label" class="muted"></p>
<div class="stage"><canvas id="avatar-canvas" width="1080" height="420" hidden aria-label="Recorded avatar in front, top and three-dimensional views"></canvas><canvas id="fallback-canvas" width="1080" height="420" aria-label="Original recorded body segments and equipment in three views"></canvas></div>
<p id="appearance-status" role="status">Original recorded body while loading</p><p id="appearance-label" class="muted"></p>
<div class="camera"><label><input id="inspect-camera" type="checkbox">Inspect with another camera</label><label for="yaw">View angle <input id="yaw" type="range" min="-3.141592653589793" max="3.141592653589793" step="0.02" value="0.5" disabled></label><label for="zoom">Zoom <input id="zoom" type="range" min="1" max="3" step="0.05" value="1" disabled></label><label><input id="follow" type="checkbox" disabled>Follow objects</label><button id="recorded-camera" disabled>Use recorded camera</button></div>
<p id="camera-label">Recorded camera</p><small>Space: play/pause. Left/right: one event. Home/end: first/last event. Playback pauses when this page is hidden.</small>
<details id="event-details"><summary>Exact recorded event and appearance change</summary><pre id="event-json"></pre></details>
<details><summary>Replay limits</summary><p>Original-body fallback uses the recorded segment endpoints and equipment outlines, not a replacement avatar. Muscles are illustrative. Unsupported avatar rendering is shown explicitly. Camera inspection is not recorded. Browser/GPU capability and performance vary.</p></details>
<noscript>JavaScript is required to inspect this offline recording. No server or network is needed.</noscript>
<script type="application/json" id="avatar-replay-data">''' + data + '</script>\n' + '\n'.join('<script id="' + name + '">' + text + '</script>' for name, text in code) + '\n</body></html>'
    if len(result.encode()) > MAX_HTML_BYTES: raise ValueError('Offline avatar replay exceeds its HTML byte bound')
    return result
