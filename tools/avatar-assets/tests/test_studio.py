"""Real loopback HTTP tests with analytic jobs and media; no workers or providers."""

from copy import deepcopy
import hashlib
import http.client
import io
import json
import os
from pathlib import Path
import re
import struct
import sys
import tempfile
import threading
import unittest
import zlib

from PIL import Image

from avatar_assets import studio


class AnalyticJobs:
    """Only records HTTP dispatch; lifecycle/execution validation has separate tests."""

    def __init__(self, root):
        self.root = root
        self.state_dir = root/'.local/studio'
        self.state_dir.mkdir(parents=True, mode=0o700)
        self.jobs = {}
        self.calls = []
        self.keys = {}
        self.seed_id = 'a'*24
        self.model = b'glTF-analytic-HTTP-routing-fixture'
        self.video = bytes(range(256))*4
        self.seed = self._make(self.seed_id, 'Analytic HTTP fixture')

    def _make(self, identifier, prompt, **fields):
        folder = self.root/'outputs'/('studio-'+identifier)
        folder.mkdir(parents=True)
        payloads = {'model.glb': self.model, 'source.blend': b'analytic editable bytes',
                    'portrait.png': png(), 'fallback.png': png(), 'idle.mp4': self.video,
                    'private.txt': b'ANALYTIC_PRIVATE_SENTINEL'}
        for name, data in payloads.items():
            (folder/name).write_bytes(data)
        def artifact(name, kind):
            return {'path': str((folder/name).relative_to(self.root)), 'kind': kind,
                    'label': name, 'bytes': len(payloads[name])}
        model, portrait, fallback = artifact('model.glb', 'model'), artifact('portrait.png', 'image'), artifact('fallback.png', 'image')
        video = {**artifact('idle.mp4', 'video'), 'id': 'idle', 'poster_path': fallback['path'], 'loop_verified': False}
        saved = {'id': identifier, 'title': prompt, 'status': 'created', 'elapsed_seconds': 0,
                 'request': {'prompt': prompt, 'style': fields.get('style', ''),
                    'custom_actions': list(fields.get('custom_actions', [])), 'preset': fields.get('preset', 'preview'),
                    'reference_selected': fields.get('reference') is not None, 'reuse_model_selected': False,
                    'original_path': str(self.root/'never-expose-this-path')},
                 'plan_id': None, 'plan': None, 'stages': [], 'reviews': [], 'error': None,
                 'previews': [portrait, fallback],
                 'result': {'model_path': model['path'], 'portrait_path': portrait['path'], 'videos': [video],
                    'downloads': [model, artifact('source.blend', 'download')],
                    'limitations': ['Analytic HTTP bytes, not generated assets.']},
                 'provenance': 'analytic_fixture'}
        self.jobs[identifier] = saved
        return saved

    def create(self, prompt, *, style='', custom_actions=(), preset='preview', reference=None, idempotency_key=None):
        # Match the real manager boundary before copying user data into fixture snapshots.
        if not isinstance(prompt, str) or not prompt.strip() or len(prompt) > 4000:
            raise ValueError('Prompt must be a nonempty string of at most 4000 characters')
        request = {'prompt': prompt, 'style': style, 'custom_actions': custom_actions,
                   'preset': preset, 'reference': reference, 'idempotency_key': idempotency_key}
        self.calls.append(('create', request))
        if idempotency_key in self.keys:
            return deepcopy(self.jobs[self.keys[idempotency_key]])
        identifier = f'{len(self.jobs)+1:024x}'
        self.keys[idempotency_key] = identifier
        return deepcopy(self._make(identifier, prompt, style=style, custom_actions=custom_actions,
                                   preset=preset, reference=reference))

    def get(self, identifier):
        if identifier not in self.jobs:
            raise ValueError('Unknown analytic job')
        return deepcopy(self.jobs[identifier])

    def list(self, *, limit, offset):
        if not (1 <= limit <= 50 and 0 <= offset <= 256):
            raise ValueError('Invalid studio list bounds')
        return {'jobs': deepcopy(list(self.jobs.values())[offset:offset+limit]),
                'total': len(self.jobs), 'offset': offset, 'limit': limit}

    def resolve_job(self, identifier):
        return self.root/'outputs'/('studio-'+identifier)

    def plan(self, identifier):
        self.calls.append(('plan', identifier))
        self.jobs[identifier].update(status='planned', plan_id='fixture-plan-sha256', plan={
            'id': 'fixture-plan-sha256', 'summary': 'Analytic plan, no generation.',
            'estimate': {'credits': 0, 'amount': None, 'currency': 'USD'},
            'authorization': {'can_start': True, 'reason': 'Analytic fixture only'}})
        return self.get(identifier)

    def start(self, identifier, *, plan_id):
        self.calls.append(('start', {'id': identifier, 'plan_id': plan_id}))
        if plan_id != self.jobs[identifier]['plan_id']:
            raise ValueError('Reviewed plan changed')
        self.jobs[identifier]['status'] = 'running'
        return self.get(identifier)

    def cancel_cooperative(self, identifier):
        self.calls.append(('cancel', identifier))
        self.jobs[identifier]['status'] = 'cancel_requested'
        return self.get(identifier)


def png():
    stream = io.BytesIO()
    Image.new('RGB', (3, 2), (60, 120, 170)).save(stream, format='PNG')
    return stream.getvalue()


def multipart(data, *, boundary='AvatarBoundary123', headers=None, ending=True):
    part_headers = headers or [('Content-Disposition', 'form-data; name="file"; filename="reference.png"'),
                               ('Content-Type', 'image/png')]
    raw = (('--'+boundary+'\r\n').encode() +
           b''.join(f'{key}: {value}\r\n'.encode() for key, value in part_headers) + b'\r\n' + data)
    if ending:
        raw += ('\r\n--'+boundary+'--\r\n').encode()
    return raw


class StudioHTTPTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix='avatar-studio-http-')
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()/'workspace'
        self.root.mkdir()
        for name in ('studio.html', 'studio.js', 'studio.css', 'favicon.svg'):
            path = self.root/'browser'/name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes((studio.ROOT/'browser'/name).read_bytes())
        vendor = self.root/'browser/node_modules/@google/model-viewer/dist/model-viewer.min.js'
        vendor.parent.mkdir(parents=True)
        vendor.write_text('/* Analytic HTTP vendor fixture; not a renderer. */')
        self.manager = AnalyticJobs(self.root)
        self.server = studio.create_server(0, root=self.root, manager=self.manager)
        self.server_errors = []
        self.server.handle_error = lambda *_: self.server_errors.append(type(sys.exception()).__name__)
        self.thread = threading.Thread(target=self.server.serve_forever, kwargs={'poll_interval': .01}, daemon=True)
        self.thread.start()
        self.addCleanup(self.stop)
        self.host = f'127.0.0.1:{self.server.server_address[1]}'
        self.job_path = '/api/studio/jobs/'+self.manager.seed_id

    def stop(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=3)
        self.assertFalse(self.thread.is_alive(), 'Owned test HTTP server must stop')

    def request(self, path, method='GET', body=None, headers=None):
        headers = list((headers or {}).items()) if isinstance(headers, dict) else list(headers or [])
        if not any(key.lower() == 'host' for key, _ in headers):
            headers.insert(0, ('Host', self.host))
        if body is not None and not any(key.lower() == 'content-length' for key, _ in headers):
            headers.append(('Content-Length', str(len(body))))
        connection = http.client.HTTPConnection('127.0.0.1', self.server.server_address[1], timeout=3)
        try:
            connection.putrequest(method, path, skip_host=True, skip_accept_encoding=True)
            for key, value in headers:
                connection.putheader(key, value)
            connection.endheaders(body)
            response = connection.getresponse()
            return response.status, dict(response.getheaders()), response.read()
        finally:
            connection.close()

    def post(self, path, value=None, *, body=None, headers=None):
        default = {'Content-Type': 'application/json', 'X-Avatar-Studio-Token': self.server.token,
                   'X-Avatar-Request-Id': 'analytic-request-1', 'Origin': 'http://'+self.host}
        if isinstance(headers, dict):
            default.update(headers)
            headers = {key: value for key, value in default.items() if value is not None}
        elif headers is None:
            headers = default
        else:
            headers = list(default.items())+headers
        raw = body if body is not None else json.dumps(value if value is not None else {}).encode()
        return self.request(path, 'POST', raw, headers)

    def upload(self, data=None, **options):
        raw = multipart(png() if data is None else data, **options)
        return self.post('/api/studio/references', body=raw,
                         headers={'Content-Type': 'multipart/form-data; boundary='+options.get('boundary', 'AvatarBoundary123')})

    def asset(self, suffix):
        value = json.loads(self.request(self.job_path)[2])
        for item in value['previews']+value['result']['downloads']+value['result']['videos']:
            if item['label'].endswith(suffix):
                return item['url']
        self.fail('Missing fixture artifact '+suffix)

    def assert_error(self, response, status=400):
        self.assertEqual(response[0], status)
        self.assertIsInstance(json.loads(response[2])['error']['message'], str)
        self.assertEqual(response[1]['Connection'], 'close')

    def test_static_entrypoint_and_session_are_loopback_only(self):
        self.assertEqual(self.server.server_address[0], '127.0.0.1')
        for route in studio.STATIC:
            with self.subTest(route=route):
                status, headers, raw = self.request(route)
                self.assertEqual(status, 200)
                self.assertTrue(raw)
                self.assertIn("script-src 'self'", headers['Content-Security-Policy'])
                self.assertNotIn('Access-Control-Allow-Origin', headers)
        session = json.loads(self.request('/api/studio/session')[2])
        self.assertEqual(session['csrf_token'], self.server.token)
        self.assertTrue(session['capabilities']['reference_upload'])
        self.assertEqual(session['limits']['reference_bytes'], 8*1024**2)
        self.assertIn('Dollar cost', session['planning_cost_note'])
        status, _, raw = self.request('/', 'HEAD')
        self.assertEqual((status, raw), (200, b''))
        self.assertEqual(self.manager.calls, [])

    def test_host_origin_and_fetch_site_guards_cover_reads_and_writes(self):
        invalid = ({'Host': 'example.invalid'}, {'Host': '127.0.0.1:1'},
                   {'Origin': 'https://example.invalid'}, {'Origin': 'null'},
                   {'Origin': 'http://localhost:'+str(self.server.server_address[1])},
                   {'Sec-Fetch-Site': 'cross-site'}, {'Sec-Fetch-Site': 'same-site'})
        for headers in invalid:
            with self.subTest(headers=headers):
                self.assert_error(self.request('/api/studio/session', headers=headers), 403)
                self.assert_error(self.post('/api/studio/jobs', {'prompt': 'fixture'}, headers=headers), 403)
        self.assertEqual(self.request('/api/studio/session', headers={'Origin': 'http://'+self.host})[0], 200)
        self.assert_error(self.request('/api/studio/session', headers=[('Host', self.host), ('Host', self.host)]), 403)
        self.assert_error(self.request('/api/studio/session', headers=[('Origin', 'http://'+self.host), ('Origin', 'http://'+self.host)]), 403)
        self.assertEqual(self.manager.calls, [])

    def test_missing_wrong_and_duplicate_csrf_never_dispatch(self):
        for token in (None, 'not-the-session-token'):
            self.assert_error(self.post('/api/studio/jobs', {'prompt': 'fixture'}, headers={'X-Avatar-Studio-Token': token}), 403)
        self.assert_error(self.post('/api/studio/jobs', {'prompt': 'fixture'}, headers=[('X-Avatar-Studio-Token', self.server.token)]), 403)
        self.assertEqual(self.manager.calls, [])

    def test_exact_actions_and_idempotency_key_reach_manager_without_starting(self):
        actions = ['Take off while the panda dances, then land.', 'Turn left; hold "hello" for one second. 🐉']
        body = {'prompt': 'A dragon with a copper backpack', 'style': 'Soft fabric, ivory horns',
                'custom_actions': actions, 'preset': 'draft'}
        first = self.post('/api/studio/jobs', body)
        second = self.post('/api/studio/jobs', body)
        self.assertEqual((first[0], second[0]), (201, 201))
        self.assertEqual(json.loads(first[2])['id'], json.loads(second[2])['id'])
        self.assertEqual(len(self.manager.jobs), 2)
        operation, request = self.manager.calls[0]
        self.assertEqual(operation, 'create')
        self.assertEqual(request, {**body, 'reference': None, 'idempotency_key': 'analytic-request-1'})
        self.assertEqual([call[0] for call in self.manager.calls], ['create', 'create'])
        self.assert_error(self.post('/api/studio/jobs', body, headers={'X-Avatar-Request-Id': None}))
        self.assert_error(self.post('/api/studio/jobs', body, headers=[('X-Avatar-Request-Id', 'duplicate')]))

    def test_creation_rejects_paths_and_unrequested_spending_fields(self):
        for field, value in [('reference', '/tmp/private.png'), ('reuse_model', '/tmp/private.glb'),
                             ('output', '../escape'), ('max_credits', 500), ('environment', {'SECRET': 'analytic'})]:
            with self.subTest(field=field):
                self.assert_error(self.post('/api/studio/jobs', {'prompt': 'fixture', field: value}))
        for identifier in ('../private', 'https://external.invalid/file', 'a'*31, 4):
            self.assert_error(self.post('/api/studio/jobs', {'prompt': 'fixture', 'reference_id': identifier}))
        self.assertEqual(self.manager.calls, [])

    def test_plan_start_and_stop_are_separate_explicit_operations(self):
        self.assertEqual(self.post(self.job_path+'/plan')[0], 202)
        self.assertEqual(self.manager.calls, [('plan', self.manager.seed_id)])
        for value in ({}, {'plan_id': None}, {'plan_id': 'x'*129}, {'plan_id': 'fixture-plan-sha256', 'max_credits': 500}):
            self.assert_error(self.post(self.job_path+'/start', value))
        self.assertEqual(len(self.manager.calls), 1)
        self.assert_error(self.post(self.job_path+'/start', {'plan_id': 'stale-plan'}))
        self.assertEqual(self.manager.seed['status'], 'planned')
        started = self.post(self.job_path+'/start', {'plan_id': 'fixture-plan-sha256'})
        self.assertEqual(started[0], 202)
        self.assertEqual(json.loads(started[2])['status'], 'running')
        self.assertEqual(self.manager.calls[-1], ('start', {'id': self.manager.seed_id, 'plan_id': 'fixture-plan-sha256'}))
        self.assert_error(self.post(self.job_path+'/cancel', {'force': True}))
        stopped = self.post(self.job_path+'/cancel')
        self.assertEqual((stopped[0], json.loads(stopped[2])['status']), (202, 'cancel_requested'))

    def test_json_and_mutation_request_bounds_fail_before_dispatch(self):
        for raw in (b'[]', b'{"prompt":"x","prompt":"y"}', b'{"prompt":NaN}', b'{"prompt":Infinity}', b'\xff', b'{'):
            with self.subTest(raw=raw):
                self.assert_error(self.post('/api/studio/jobs', body=raw))
        self.assert_error(self.post('/api/studio/jobs', {'prompt': 'x'}, headers={'Content-Type': 'text/plain'}))
        self.assert_error(self.post('/api/studio/jobs', {'prompt': 'x'}, headers=[('Content-Type', 'application/json')]))
        self.assert_error(self.post('/api/studio/jobs?prompt=bad', {'prompt': 'x'}))
        self.assert_error(self.post('/api/studio/jobs', body=b'', headers={'Content-Length': str(studio.MAX_JSON+1)}))
        self.assert_error(self.post('/api/studio/jobs', body=b'{}', headers=[('Content-Length', '2'), ('Content-Length', '2')]))
        self.assert_error(self.post('/api/studio/jobs', body=b'{}', headers={'Transfer-Encoding': 'chunked'}))
        self.assert_error(self.request('/api/studio/jobs', 'POST', headers={'Content-Type': 'application/json', 'X-Avatar-Studio-Token': self.server.token}))
        self.assertEqual(self.manager.calls, [])

    def test_deep_json_is_a_bounded_json_error_not_a_handler_crash(self):
        raw = b'{"prompt":'+b'['*1500+b'0'+b']'*1500+b'}'
        self.assert_error(self.post('/api/studio/jobs', body=raw))
        self.assertEqual(self.server_errors, [])
        self.assertEqual(self.request('/api/studio/session')[0], 200)

    def test_upload_preserves_bytes_and_uses_only_opaque_reference_id(self):
        data = png()
        response = self.upload(data)
        self.assertEqual(response[0], 201)
        value = json.loads(response[2])
        self.assertRegex(value['id'], r'^[a-f0-9]{32}$')
        self.assertNotIn(str(self.root), response[2].decode())
        self.assertEqual(self.request(value['preview_url'])[2], data)
        selected = self.post('/api/studio/jobs', {'prompt': 'An analytic robot', 'reference_id': value['id']})
        self.assertEqual(selected[0], 201)
        self.assertTrue(json.loads(selected[2])['request']['reference_selected'])
        path = self.manager.calls[-1][1]['reference']
        self.assertEqual(path.read_bytes(), data)
        self.assertTrue(path.is_relative_to(self.server.references.folder))
        self.assertEqual(path.stat().st_mode & 0o777, 0o600)
        receipt = json.loads(path.with_suffix('.json').read_text())
        self.assertEqual(receipt['sha256'], hashlib.sha256(data).hexdigest())
        self.assertEqual(receipt['dimensions'], [3, 2])

    def test_all_supported_image_formats_are_content_verified(self):
        for format_name, suffix in [('PNG', 'png'), ('JPEG', 'jpg'), ('WEBP', 'webp')]:
            with self.subTest(format_name=format_name):
                stream = io.BytesIO()
                Image.new('RGB', (3, 2), (40, 80, 120)).save(stream, format=format_name)
                response = self.upload(stream.getvalue(), headers=[('Content-Disposition', f'form-data; name="file"; filename="image.{suffix}"'), ('Content-Type', 'application/octet-stream')])
                self.assertEqual(response[0], 201)
                preview = json.loads(response[2])['preview_url']
                self.assertEqual(self.request(preview)[2], stream.getvalue())

    def test_upload_invalid_image_pixel_bound_and_animation_are_rejected(self):
        self.assert_error(self.upload(b'not an image'))
        self.assert_error(self.upload(b''))
        large_header = bytearray(png())
        large_header[16:24] = struct.pack('>II', 4097, 4097)
        large_header[29:33] = struct.pack('>I', zlib.crc32(large_header[12:29]))
        self.assert_error(self.upload(bytes(large_header)))
        stream = io.BytesIO()
        Image.new('RGB', (2, 2), 'red').save(stream, format='PNG', save_all=True,
                                           append_images=[Image.new('RGB', (2, 2), 'blue')], duration=10)
        self.assert_error(self.upload(stream.getvalue()))
        self.assertEqual(list(self.server.references.folder.iterdir()), [])

    def test_upload_oversize_body_is_rejected_before_reading_or_storage(self):
        response = self.post('/api/studio/references', body=b'', headers={
            'Content-Type': 'multipart/form-data; boundary=AvatarBoundary123',
            'Content-Length': str(studio.MAX_REFERENCE+64*1024+1)})
        self.assert_error(response)
        self.assertEqual(list(self.server.references.folder.iterdir()), [])
        self.assertEqual(self.request('/api/studio/session')[0], 200)

    def test_malformed_multiple_and_nested_multipart_are_rejected(self):
        self.assert_error(self.upload(ending=False))
        self.assert_error(self.post('/api/studio/references', body=png(), headers={'Content-Type': 'image/png'}))
        for value in ('multipart/form-data', 'multipart/form-data; boundary="bad boundary"', 'multipart/form-data; boundary='+('x'*71)):
            self.assert_error(self.post('/api/studio/references', body=b'abc', headers={'Content-Type': value}))
        multiple = multipart(png())[:-len(b'--AvatarBoundary123--\r\n')]+multipart(png())
        self.assert_error(self.post('/api/studio/references', body=multiple, headers={'Content-Type': 'multipart/form-data; boundary=AvatarBoundary123'}))
        for headers in ([('Content-Disposition', 'form-data; name="wrong"; filename="a.png"')],
                        [('Content-Disposition', 'form-data; name="file"; filename="a.png"'), ('Content-Type', 'multipart/mixed; boundary=inside')],
                        [('Content-Disposition', 'form-data; name="file"; filename="a.png"'), ('Content-Transfer-Encoding', 'base64')]):
            self.assert_error(self.upload(headers=headers))
        self.assertEqual(list(self.server.references.folder.iterdir()), [])

    def test_ambiguous_multipart_headers_are_rejected(self):
        disposition = ('Content-Disposition', 'form-data; name="file"; filename="a.png"')
        for headers in ([disposition, disposition],
                        [disposition, ('Content-Type', 'image/png'), ('Content-Type', 'image/jpeg')],
                        [('Content-Disposition', 'attachment; name="file"; filename="a.png"')],
                        [('Content-Disposition', 'form-data; name="file"')]):
            with self.subTest(headers=headers):
                self.assert_error(self.upload(headers=headers))
        self.assertEqual(list(self.server.references.folder.iterdir()), [])

    def test_reference_tampering_and_symlink_substitution_fail_closed(self):
        uploaded = json.loads(self.upload()[2])
        path = self.server.references.resolve(uploaded['id'])
        path.write_bytes(path.read_bytes()+b'changed')
        self.assert_error(self.request(uploaded['preview_url']))
        second = json.loads(self.upload()[2])
        second_path = self.server.references.resolve(second['id'])
        second_path.unlink()
        second_path.symlink_to(path)
        self.assertIn(self.request(second['preview_url'])[0], (400, 403))
        self.assertEqual(self.manager.calls, [])

    def test_metadata_has_opaque_urls_and_no_filesystem_paths(self):
        response = self.request(self.job_path)
        self.assertEqual(response[0], 200)
        value = json.loads(response[2])
        self.assertNotIn(str(self.root), response[2].decode())
        self.assertNotIn('outputs/studio-', response[2].decode())
        self.assertNotIn('original_path', response[2].decode())
        self.assertFalse(value['request']['reference_selected'])
        self.assertFalse(value['request']['reuse_model_selected'])
        self.assertEqual(value['provenance'], 'analytic_fixture')
        pattern = rf'^/api/studio/jobs/{self.manager.seed_id}/files/[a-f0-9]{{32}}$'
        result = value['result']
        for url in [result['model_url'], result['portrait_url'], result['videos'][0]['url'], result['videos'][0]['poster_url']]:
            self.assertRegex(url, pattern)
        self.assertEqual(self.request(result['model_url'])[2], self.manager.model)
        self.assertEqual(self.manager.calls, [])

    def test_artifact_ranges_head_etag_and_attachments(self):
        route = self.asset('.mp4')
        for value, expected in [('bytes=0-9', self.manager.video[:10]), ('bytes=1000-', self.manager.video[1000:]), ('bytes=-7', self.manager.video[-7:])]:
            with self.subTest(value=value):
                status, headers, raw = self.request(route, headers={'Range': value})
                self.assertEqual((status, raw), (206, expected))
                self.assertEqual(int(headers['Content-Length']), len(expected))
                self.assertIn('/1024', headers['Content-Range'])
        status, headers, raw = self.request(route, 'HEAD')
        self.assertEqual((status, raw), (200, b''))
        self.assertEqual(headers['Content-Length'], '1024')
        self.assertEqual(self.request(route, headers={'If-None-Match': headers['ETag']})[0], 304)
        self.assertEqual(self.request(route, headers={'Range': 'bytes=0-9', 'If-Range': headers['ETag']})[2], self.manager.video)
        for value in ('bytes=1024-', 'bytes=0-1,3-5', 'bytes=-0'):
            self.assert_error(self.request(route, headers={'Range': value}), 416)
        self.assert_error(self.request(route, headers=[('Range', 'bytes=0-1'), ('Range', 'bytes=2-3')]), 416)
        for suffix in ('.glb', '.blend'):
            self.assertIn('attachment;', self.request(self.asset(suffix))[1]['Content-Disposition'])

    def test_unlisted_outside_symlink_and_oversize_artifacts_are_not_streamed(self):
        folder = self.manager.resolve_job(self.manager.seed_id)
        relative = str((folder/'private.txt').relative_to(self.root))
        unlisted = self.job_path+'/files/'+studio._asset_id(self.manager.seed_id, relative)
        self.assert_error(self.request(unlisted))
        outside = self.root/'outside.png'
        outside.write_bytes(b'ANALYTIC_PRIVATE_SENTINEL')
        linked = folder/'linked.png'
        linked.symlink_to(outside)
        oversized = folder/'oversized.glb'
        with oversized.open('wb') as stream:
            stream.truncate(studio.gallery.MAX_FILE_BYTES+1)
        for file in (outside, linked, oversized):
            path = str(file.relative_to(self.root))
            self.manager.seed['previews'].append({'kind': 'image', 'path': path, 'label': file.name})
            route = self.job_path+'/files/'+studio._asset_id(self.manager.seed_id, path)
            response = self.request(route)
            self.assertIn(response[0], (400, 403))
            self.assertNotIn(b'ANALYTIC_PRIVATE_SENTINEL', response[2])

    def test_arbitrary_paths_queries_and_unsupported_methods_are_rejected(self):
        for path in ('/outputs/', '/.git/config', '/.local/studio/', '/browser/studio.html', '/api/studio/jobs/unknown'):
            self.assertEqual(self.request(path)[0], 404)
        for path in ('/../private', '/%2e%2e/private', '/api/studio/session#bad', 'http://external.invalid/', '/studio.js?x=1', self.job_path+'?path=/tmp/private'):
            self.assert_error(self.request(path))
        for query in ('limit=1&limit=2', 'limit=51', 'offset=-1', 'path=private', 'bad'):
            self.assert_error(self.request('/api/studio/jobs?'+query))
        status, _, raw = self.request('/api/studio/jobs?limit=1&offset=0')
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(raw)['limit'], 1)
        for method in ('PUT', 'DELETE', 'PATCH', 'OPTIONS'):
            response = self.request('/api/studio/jobs', method)
            self.assert_error(response, 405)
            self.assertEqual(response[1]['Allow'], 'GET, HEAD, POST')
        self.assertEqual(self.manager.calls, [])


if __name__ == '__main__':
    unittest.main()
