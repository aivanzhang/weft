"""Loopback creation API; bounded jobs own execution and the gallery owns streaming."""

from copy import deepcopy
from email import policy
from email.parser import BytesParser
import hashlib
from http.server import HTTPServer
import io
import json
import os
from pathlib import Path
import re
import secrets
import stat
import threading
from urllib.parse import parse_qs, urlsplit
import uuid

from PIL import Image

from avatar_assets import gallery
from avatar_assets.safety import require

ROOT = Path(__file__).resolve().parents[1]
MAX_JSON = 32 * 1024
MAX_REFERENCE = 8 * 1024**2
MAX_UPLOADS = 32
JOB_ID = r'[a-f0-9]{24}'
REFERENCE_ID = r'[a-f0-9]{32}'
FILE_ID = r'[a-f0-9]{32}'
ASSET_SUFFIXES = {'.png', '.jpg', '.jpeg', '.webp', '.mp4', '.glb', '.blend'}
STATIC = {'/': 'browser/studio.html', '/studio.js': 'browser/studio.js',
          '/studio.css': 'browser/studio.css', '/favicon.svg': 'browser/favicon.svg',
          '/vendor/model-viewer.min.js': gallery._STATIC['/vendor/model-viewer.min.js']}


def _json(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, 'Duplicate JSON key')
            result[key] = value
        return result
    def bad_constant(_):
        raise ValueError('Nonfinite JSON value')
    try:
        value = json.loads(raw, object_pairs_hook=pairs, parse_constant=bad_constant)
    except RecursionError:
        raise ValueError('JSON nesting exceeds studio bounds') from None
    require(isinstance(value, dict), 'Expected a JSON object')
    return value


def _asset_id(job, relative):
    return hashlib.sha256((job+'\0'+relative).encode()).hexdigest()[:32]


def _descriptors(snapshot):
    result = []
    pending = [snapshot.get('previews', []), snapshot.get('result')]
    while pending:
        value = pending.pop()
        if isinstance(value, dict):
            if 'path' in value and 'kind' in value:
                result.append(value)
            pending.extend(v for k, v in value.items() if k not in ('path', 'model_path', 'portrait_path', 'poster_path'))
        elif isinstance(value, list):
            pending.extend(value)
        require(len(result)+len(pending) <= 256, 'Too many result artifacts')
    return result


def _wire(snapshot):
    value = deepcopy(snapshot)
    job_id = value['id']
    paths = {item['path']: f'/api/studio/jobs/{job_id}/files/{_asset_id(job_id,item["path"])}'
             for item in _descriptors(value)}
    def convert(node):
        if isinstance(node, list):
            return [convert(item) for item in node]
        if not isinstance(node, dict):
            return node
        result = {}
        for key, item in node.items():
            if key in ('path', 'model_path', 'portrait_path', 'poster_path'):
                if item in paths:
                    result['url' if key == 'path' else key.replace('_path', '_url')] = paths[item]
            elif key not in ('original_path', 'output_path'):
                result[key] = convert(item)
        return result
    value = convert(value)
    if isinstance(value.get('request'), dict):
        appearance = value['request'].get('appearance_policy')
        value['request'] = {k: v for k, v in value['request'].items()
                            if k in ('prompt', 'style', 'custom_actions', 'preset',
                                     'reference_selected', 'reuse_model_selected')}
        if appearance == 'preserve_current':
            value['request']['appearance_policy'] = appearance
    return value


class References:
    def __init__(self, root, state_dir):
        self.root = root
        self.folder = state_dir/'uploads'
        require(not self.folder.is_symlink(), 'Unsafe reference directory')
        self.folder.mkdir(mode=0o700, exist_ok=True)
        info = self.folder.stat()
        require(stat.S_IMODE(info.st_mode) == 0o700 and info.st_uid == os.getuid(),
                'References need private local ownership')
        self.lock = threading.Lock()

    def add(self, data, name):
        require(0 < len(data) <= MAX_REFERENCE, 'Reference exceeds 8 MiB')
        try:
            with Image.open(io.BytesIO(data)) as im:
                require(im.format in ('PNG', 'JPEG', 'WEBP') and im.width*im.height <= 16_777_216
                        and getattr(im, 'n_frames', 1) == 1, 'Use a single PNG/JPEG/WebP image no larger than 16 MP')
                suffix = {'PNG': '.png', 'JPEG': '.jpg', 'WEBP': '.webp'}[im.format]
                dimensions = [im.width, im.height]
                im.verify()
        except (Image.DecompressionBombError, Image.DecompressionBombWarning):
            raise ValueError('Reference exceeds the image decode limit') from None
        name = ''.join(c for c in str(name) if ord(c) >= 32)[:160] or 'Reference image'
        with self.lock:
            entries = []
            for index, path in enumerate(self.folder.iterdir()):
                require(index < MAX_UPLOADS*3, 'Reference inventory exceeds its bound')
                require(not path.is_symlink() and path.is_file(), 'Unsafe reference inventory')
                entries.append(path)
            require(len(entries) < MAX_UPLOADS*2 and sum(p.stat().st_size for p in entries)+len(data)
                    <= MAX_UPLOADS*MAX_REFERENCE, 'Reference storage is full; preserve existing references before adding more')
            identifier = uuid.uuid4().hex
            target = self.folder/(identifier+suffix)
            with target.open('xb') as stream:
                os.fchmod(stream.fileno(), 0o600)
                stream.write(data)
                stream.flush()
                os.fsync(stream.fileno())
            receipt = {'id': identifier, 'name': name, 'file': target.name,
                       'sha256': hashlib.sha256(data).hexdigest(), 'bytes': len(data), 'dimensions': dimensions}
            with (self.folder/(identifier+'.json')).open('x') as stream:
                os.fchmod(stream.fileno(), 0o600)
                json.dump(receipt, stream)
        return {'id': identifier, 'name': name, 'preview_url': '/api/studio/references/'+identifier}

    def resolve(self, identifier):
        require(isinstance(identifier, str) and re.fullmatch(REFERENCE_ID, identifier), 'Invalid reference ID')
        receipt_path = self.folder/(identifier+'.json')
        with gallery._open_file(self.root, receipt_path.relative_to(self.root), 2048)[0] as stream:
            receipt = _json(stream.read(2049))
        require(receipt['id'] == identifier and re.fullmatch(REFERENCE_ID+r'\.(png|jpg|webp)', receipt['file']),
                'Reference receipt changed')
        path = self.folder/receipt['file']
        with gallery._open_file(self.root, path.relative_to(self.root), MAX_REFERENCE)[0] as stream:
            raw = stream.read(MAX_REFERENCE+1)
        require(len(raw) == receipt['bytes'] and hashlib.sha256(raw).hexdigest() == receipt['sha256'],
                'Selected reference changed')
        return path


class StudioHandler(gallery._Handler):
    server_version = 'AvatarStudio/1'

    def _error(self, status, message, extra=None):
        self._reply(status, {'error': {'message': message}}, extra=extra)

    def _reply(self, status, value, extra=None):
        raw = json.dumps(value, separators=(',', ':'), allow_nan=False).encode()
        require(len(raw) <= 1024**2, 'Response exceeds studio bounds')
        self._headers(status, len(raw), 'application/json; charset=utf-8', extra)
        if self.command != 'HEAD':
            self.wfile.write(raw)

    def _path(self):
        parsed = urlsplit(self.path)
        require(self.path.startswith('/') and not parsed.scheme and not parsed.netloc and not parsed.fragment
                and '%' not in parsed.path and '\\' not in parsed.path and '..' not in parsed.path.split('/'),
                'Unsafe request path')
        return parsed

    def _body(self, maximum):
        lengths = self.headers.get_all('Content-Length', [])
        require(not self.headers.get_all('Transfer-Encoding') and len(lengths) == 1
                and re.fullmatch(r'[0-9]{1,10}', lengths[0]), 'A single Content-Length is required')
        size = int(lengths[0])
        require(0 < size <= maximum, 'Request body exceeds its bound')
        raw = self.rfile.read(size)
        require(len(raw) == size, 'Incomplete request body')
        return raw

    def _get(self):
        if not self._allowed():
            return
        try:
            parsed = self._path()
            path = parsed.path
            if path in STATIC:
                require(not parsed.query, 'Unexpected static query')
                file = STATIC[path]
                return self._serve_resource(self.server.root, gallery._Resource(file, gallery._MIME[Path(file).suffix]))
            if path == '/api/studio/session':
                return self._reply(200, self.server.session)
            if path == '/api/studio/jobs':
                query = parse_qs(parsed.query, strict_parsing=True)
                require(set(query) <= {'limit', 'offset'} and all(len(v) == 1 for v in query.values()), 'Invalid list query')
                limit, offset = int(query.get('limit', ['20'])[0]), int(query.get('offset', ['0'])[0])
                result = self.server.manager.list(limit=limit, offset=offset)
                if isinstance(result, list):
                    result = {'jobs': result}
                return self._reply(200, {**result, 'jobs': [_wire(job) for job in result['jobs']]})
            require(not parsed.query, 'Unexpected request query')
            if match := re.fullmatch('/api/studio/jobs/('+JOB_ID+')', path):
                return self._reply(200, _wire(self.server.manager.get(match[1])))
            if match := re.fullmatch('/api/studio/references/('+REFERENCE_ID+')', path):
                file = self.server.references.resolve(match[1])
                return self._serve_resource(self.server.root, gallery._Resource(str(file.relative_to(self.server.root)),
                                                                               gallery._MIME[file.suffix]))
            if match := re.fullmatch('/api/studio/jobs/('+JOB_ID+')/files/('+FILE_ID+')', path):
                snapshot = self.server.manager.get(match[1])
                item = next((item for item in _descriptors(snapshot) if _asset_id(match[1], item['path']) == match[2]), None)
                require(item is not None, 'Unknown job artifact')
                relative = gallery._relative(item['path'])
                file = self.server.root/relative
                require(file.is_relative_to(self.server.manager.resolve_job(match[1]))
                        and file.suffix in ASSET_SUFFIXES, 'Artifact is outside this job or has an unsupported type')
                return self._serve_resource(self.server.root, gallery._Resource(str(relative), gallery._MIME[file.suffix],
                                             attachment=file.suffix in ('.blend', '.glb')))
            self._error(404, 'Unknown studio resource')
        except (ValueError, TypeError, KeyError, OSError) as error:
            self._error(400 if not isinstance(error, FileNotFoundError) else 404, self._safe_error(error))

    @staticmethod
    def _safe_error(error):
        from avatar_assets.studio_jobs import _safe_text
        return _safe_text(str(error), 600) or 'This request could not be completed'

    def do_POST(self):
        if not self._allowed():
            return
        tokens = self.headers.get_all('X-Avatar-Studio-Token', [])
        if len(tokens) != 1 or not secrets.compare_digest(tokens[0], self.server.token):
            return self._error(403, 'A valid local studio session is required')
        try:
            parsed = self._path()
            require(not parsed.query, 'Mutation queries are unsupported')
            content_types = self.headers.get_all('Content-Type', [])
            require(len(content_types) == 1, 'One Content-Type is required')
            if parsed.path == '/api/studio/references':
                require(content_types[0].lower().startswith('multipart/form-data;'), 'Use a single multipart file')
                raw = self._body(MAX_REFERENCE+64*1024)
                header = BytesParser(policy=policy.default).parsebytes(
                    b'Content-Type: '+content_types[0].encode('ascii')+b'\r\n\r\n')
                boundary = header.get_boundary()
                require(isinstance(boundary, str) and re.fullmatch(r'[A-Za-z0-9_+./:=?-]{1,70}', boundary),
                        'Invalid multipart boundary')
                marker = b'--'+boundary.encode('ascii')
                ending = b'\r\n'+marker+b'--'
                body_raw = raw[:-2] if raw.endswith(b'\r\n') else raw
                require(body_raw.startswith(marker+b'\r\n') and body_raw.endswith(ending), 'Incomplete multipart upload')
                part = body_raw[len(marker)+2:-len(ending)]
                split = part.find(b'\r\n\r\n')
                require(0 < split <= 8192 and b'\r\n'+marker not in part, 'Upload exactly one bounded image file')
                headers = BytesParser(policy=policy.default).parsebytes(part[:split]+b'\r\n\r\n')
                require(not headers.defects and not any(getattr(h, 'defects', ()) for h in headers.values())
                        and len(headers.get_all('Content-Disposition', [])) == 1
                        and len(headers.get_all('Content-Type', [])) <= 1
                        and headers.get_content_disposition() == 'form-data'
                        and isinstance(headers.get_filename(), str) and bool(headers.get_filename().strip())
                        and headers.get_param('name', header='content-disposition') == 'file'
                        and not headers.get('Content-Transfer-Encoding') and headers.get_content_maintype() != 'multipart',
                        'Unsupported image upload headers')
                return self._reply(201, self.server.references.add(part[split+4:], headers.get_filename()))
            require(content_types[0].split(';')[0].strip().lower() == 'application/json', 'Use application/json')
            body = _json(self._body(MAX_JSON))
            if parsed.path == '/api/studio/jobs':
                require(set(body) <= {'prompt', 'style', 'custom_actions', 'preset', 'reference_id'} and 'prompt' in body,
                        'Unsupported creation fields')
                reference = self.server.references.resolve(body['reference_id']) if body.get('reference_id') else None
                keys = self.headers.get_all('X-Avatar-Request-Id', [])
                require(len(keys) == 1, 'A unique request ID is required')
                job = self.server.manager.create(body['prompt'], style=body.get('style', ''),
                        custom_actions=body.get('custom_actions', []), preset=body.get('preset', 'preview'),
                        reference=reference, idempotency_key=keys[0])
                return self._reply(201, _wire(job))
            match = re.fullmatch('/api/studio/jobs/('+JOB_ID+')/(plan|start|cancel|retry)', parsed.path)
            require(match is not None, 'Unknown studio operation')
            identifier, operation = match.groups()
            if operation == 'retry':
                require(not body or body == {'appearance_policy': 'preserve_current'},
                        'Retry accepts only the explicit preserve_current appearance choice, not prompt, path, key or spending fields')
                result = (self.server.manager.retry(identifier, appearance_policy='preserve_current') if body
                          else self.server.manager.retry(identifier))
            elif operation == 'start':
                require(set(body) == {'plan_id'} and isinstance(body['plan_id'], str)
                        and 0 < len(body['plan_id']) <= 128, 'Start must name the reviewed plan')
                result = self.server.manager.start(identifier, plan_id=body['plan_id'])
            else:
                require(not body, 'Unexpected operation fields')
                result = (self.server.manager.plan(identifier) if operation == 'plan'
                          else self.server.manager.cancel_cooperative(identifier))
            self._reply(202, _wire(result))
        except (ValueError, TypeError, KeyError, OSError, UnicodeError) as error:
            self._error(400, self._safe_error(error))

    def _unsupported_method(self):
        if self._allowed():
            self._error(405, 'Unsupported studio method', {'Allow': 'GET, HEAD, POST'})

    do_PUT = do_DELETE = do_PATCH = do_OPTIONS = _unsupported_method


class StudioServer(gallery._Server):
    def __init__(self, root, manager, port):
        self.root, self.manager = root, manager
        require(manager.state_dir.is_relative_to(root), 'Studio state must be within the authoring workspace')
        self.token = secrets.token_urlsafe(32)
        self.references = References(root, manager.state_dir)
        self._slots = threading.BoundedSemaphore(gallery.MAX_REQUESTS)
        from avatar_assets.agent_backend import readiness
        backend = readiness(root)
        from avatar_assets.standing_authorization import standing_policy_status
        standing = standing_policy_status(root)
        self.session = {'csrf_token': self.token, 'agent_backend': backend, 'standing_retry': standing,
            'presets': [{'id': 'preview', 'label': 'HD preview', 'description': '1600px portrait and 720px action videos'},
                        {'id': 'draft', 'label': 'Quick draft', 'description': 'Small previews for early review'},
                        {'id': 'muse', 'label': 'Long HD clips', 'description': 'Eight-second clips; not a quality guarantee'}],
            'limits': {'subject_chars': 4000, 'style_chars': 800, 'actions': 8, 'action_chars': 600,
                       'reference_bytes': MAX_REFERENCE},
            'capabilities': {'reference_upload': True},
            'planning_cost_note': 'Planning, rigging instructions and review use the server OpenAI API, not a personal AI login. '
                'Saving a prompt is free. Planning requires a saved-request dollar grant; an enabled owner standing policy can grant an exact saved-work retry. '
                'Meshy generation credits are separate from OpenAI dollars. Retry reuses the saved source and plan; it never buys credits or increases the allowance. '
                + (standing['reason'] if standing.get('enabled') else backend['reason'])}
        HTTPServer.__init__(self, ('127.0.0.1', port), StudioHandler)


def create_server(port=8766, *, manager=None, root=ROOT):
    require(type(port) is int and 0 <= port <= 65535, 'Invalid studio port')
    root = Path(root).resolve()
    if manager is None:
        from avatar_assets.studio_jobs import StudioJobs
        manager = StudioJobs(root=root)
    return StudioServer(root, manager, port)


def serve_studio(port=8766):
    server = create_server(port)
    print(f'Avatar studio: http://127.0.0.1:{server.server_address[1]}/', flush=True)
    try:
        server.serve_forever(poll_interval=.25)
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
