"""One bounded native Meta segmentation request; failures never trigger retries."""

import base64
import http.client
import json
import multiprocessing
from contextlib import contextmanager
from pathlib import Path
import time

from weft.io import digest, local_time, write_json
from weft.meta_access import reserve_call, resolve_key, settle_call

MAX_REQUEST_BYTES = 50_000_000
MAX_RESPONSE_BYTES = 64_000_000
MAX_EVENT_BYTES = 1_000_000
TOTAL_SECONDS = 120
SOCKET_SECONDS = 30
RESERVATION_USD = .15


def _prompts(frame_count, width, height, roi, objects):
    if (any(type(n) is not int or n <= 0 for n in (frame_count, width, height))
            or frame_count > 500 or max(width, height) > 4096 or width * height > 8_847_360):
        raise ValueError('Source exceeds the bounded frame/dimension envelope')
    if (roi is None) == (objects is None):
        raise ValueError('Supply either roi or explicit objects, not both')
    if objects is None:
        objects = [{'object_id': 1, 'frame': 0, 'roi': roi}]
    if not isinstance(objects, list) or not 1 <= len(objects) <= 8:
        raise ValueError('Expected a list of 1..8 object prompts')
    normalized, seen = [], set()
    for obj in objects:
        if not isinstance(obj, dict) or set(obj) != {'object_id', 'frame', 'roi'}:
            raise ValueError('Object prompt requires only object_id, frame, roi')
        identity, frame, box = obj['object_id'], obj['frame'], obj['roi']
        if type(identity) is not int or not 1 <= identity <= 8 or identity in seen:
            raise ValueError('Object IDs must be unique integers within 1..8')
        if type(frame) is not int or not 0 <= frame < frame_count:
            raise ValueError('Initialization frame is outside the source')
        if not isinstance(box, (list, tuple)) or len(box) != 4 or any(type(n) is not int for n in box):
            raise ValueError('ROI must contain four integer pixel coordinates')
        x, y, w, h = box
        if min(x, y) < 0 or min(w, h) <= 0 or x + w > width or y + h > height:
            raise ValueError('ROI lies outside the verified video')
        normalized.append({'object_id': identity, 'frame': frame, 'roi': list(box)})
        seen.add(identity)
    return normalized


def build_request(video, *, expected_sha256, frame_count, width, height, roi=None, objects=None):
    prompts = _prompts(frame_count, width, height, roi, objects)
    video = Path(video)
    if video.suffix.lower() != '.mp4' or video.stat().st_size > 37_000_000:
        raise ValueError('A verified MP4 within the inline media bound is required')
    if digest(video) != expected_sha256:
        raise ValueError('Video changed after verification')
    blocks = {}
    for obj in sorted(prompts, key=lambda obj: (obj['frame'], obj['object_id'])):
        x, y, w, h = obj['roi']
        blocks.setdefault(obj['frame'], []).append(
            f"{obj['object_id']}<|box;x1={x};y1={y};x2={x+w-1};y2={y+h-1};w={width};h={height}|>")
    prompt = ''.join(f'<{frame}f>' + ','.join(rows) for frame, rows in blocks.items())
    payload = {'model': 'sam-3.1', 'stream': True, 'store': False,
        'input': [{'type': 'message', 'role': 'user', 'content': [
            {'type': 'input_text', 'text': prompt},
            {'type': 'input_video', 'video_url': 'data:video/mp4;base64,' + base64.b64encode(video.read_bytes()).decode('ascii')}]}],
        'metadata': {'propagation_direction': 'forward', 'include_confidence': 'true', 'mask_encoding': 'one_bit'}}
    body = json.dumps(payload, separators=(',', ':'), allow_nan=False).encode()
    if len(body) > MAX_REQUEST_BYTES:
        raise ValueError('Complete inline JSON exceeds 50 MB')
    return body


def _events(response, raw, started):
    size, event_size, data = 0, 0, []
    while True:
        if time.monotonic() - started > TOTAL_SECONDS:
            raise TimeoutError('Segmentation total deadline exceeded')
        line = response.readline(MAX_EVENT_BYTES + 1)
        if not line:
            if data:
                raise ValueError('SSE ended within an event')
            return
        raw.write(line)
        size += len(line)
        event_size += len(line)
        if size > MAX_RESPONSE_BYTES or event_size > MAX_EVENT_BYTES:
            raise ValueError('Segmentation response exceeded its byte bound')
        line = line.rstrip(b'\r\n')
        if not line:
            if data:
                value = b'\n'.join(data)
                if value != b'[DONE]':
                    yield json.loads(value)
            data, event_size = [], 0
        elif line.startswith(b'data:'):
            data.append(line[5:].removeprefix(b' '))


def _cost(usage):
    frames = usage.get('video_frames_processed') if isinstance(usage, dict) else None
    if type(frames) is not int or frames < 0:
        return None
    # Integer-rate division avoids a floating tail being rounded up by the USD ledger.
    return max(50, frames) / 5000


def _network_worker(pipe, body, key):
    # Isolation makes DNS, TLS, uploads and slow trickles cancellable by the parent.
    connection = None
    try:
        connection = http.client.HTTPSConnection('api.meta.ai', timeout=SOCKET_SECONDS)
        connection.request('POST', '/v1/responses', body=body,
                           headers={'Authorization': f'Bearer {key}', 'Content-Type': 'application/json', 'Accept': 'text/event-stream'})
        response = connection.getresponse()
        pipe.send(('headers', response.status, response.getheaders()))
        if response.status != 200:
            pipe.send(('bytes', response.read(8192)))
        else:
            size = 0
            while True:
                line = response.readline(MAX_EVENT_BYTES + 1)
                if not line:
                    break
                size += len(line)
                pipe.send(('bytes', line))
                if len(line) > MAX_EVENT_BYTES or size > MAX_RESPONSE_BYTES:
                    break
        pipe.send(('end',))
    except Exception as error:
        # Exception messages may contain request data; retain only the type.
        pipe.send(('error', type(error).__name__))
    finally:
        if connection is not None:
            connection.close()
        pipe.close()


class _NetworkResponse:
    def __init__(self, pipe, deadline):
        self.pipe, self.deadline = pipe, deadline
        event = self._receive()
        if event[0] != 'headers':
            raise ValueError('Missing HTTP response headers')
        self.status, self.headers = event[1:]

    def _receive(self):
        remaining = self.deadline - time.monotonic()
        if remaining <= 0 or not self.pipe.poll(remaining):
            raise TimeoutError('Segmentation hard network deadline exceeded')
        event = self.pipe.recv()
        if event[0] == 'error':
            if event[1] == 'TimeoutError':
                raise TimeoutError('Segmentation network timeout')
            raise RuntimeError('Segmentation network failure: ' + event[1])
        return event

    def getheaders(self):
        return self.headers

    def getheader(self, key, default=None):
        return next((value for name, value in self.headers if name.lower() == key.lower()), default)

    def readline(self, limit):
        event = self._receive()
        if event[0] == 'end':
            return b''
        if event[0] != 'bytes' or len(event[1]) > limit:
            raise ValueError('Invalid bounded network chunk')
        return event[1]

    def read(self, limit):
        return self.readline(limit)


@contextmanager
def _network_response(body, key, started):
    context = multiprocessing.get_context('spawn')
    receive, send = context.Pipe(duplex=False)
    process = context.Process(target=_network_worker, args=(send, body, key), daemon=True)
    try:
        process.start()
        send.close()
        yield _NetworkResponse(receive, started + TOTAL_SECONDS)
    finally:
        receive.close()
        send.close()
        if process.pid is not None:
            if process.is_alive():
                process.terminate()
            process.join(timeout=1)
            if process.is_alive():
                process.kill()
                process.join(timeout=1)
            process.close()


def segment_video(video, out, *, expected_sha256, frame_count, width, height, roi=None, objects=None, project_root=None):
    from weft.meta_segmentation_parser import SamStream

    body = build_request(video, expected_sha256=expected_sha256, frame_count=frame_count,
                         width=width, height=height, roi=roi, objects=objects)
    prompts = _prompts(frame_count, width, height, roi, objects)
    estimate = max(50, frame_count) * len(prompts) * .20 / 1000
    reservation_usd = max(RESERVATION_USD, estimate)
    key = resolve_key(project_root)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    (out / 'request.json').write_bytes(body)
    record = {'status': 'prepared', 'created_at': local_time(), 'model': 'sam-3.1',
        'video_sha256': expected_sha256, 'request_sha256': digest(out / 'request.json'),
        'implementation_sha256': digest(Path(__file__)), 'request_count': 0, 'retries': 0,
        'expected_cost_usd': estimate, 'reserved_usd': reservation_usd,
        'initialization': prompts, 'estimate_assumption': 'Conservative per-object frame upper envelope; not an assertion of provider billing units',
        'pricing': {'video_per_1000_frames_usd': .20, 'minimum_frames': 50, 'checked_on': '2026-10-28'},
        'provider_hard_cap': False, 'usage': None, 'cost_usd': None, 'applied': False,
        'limits': {'between_read_deadline_seconds': TOTAL_SECONDS, 'socket_inactivity_seconds': SOCKET_SECONDS,
                   'hard_network_deadline_seconds': TOTAL_SECONDS, 'deadline_cleanup_allowance_seconds': 2,
                   'max_response_bytes': MAX_RESPONSE_BYTES, 'max_event_bytes': MAX_EVENT_BYTES}}
    write_json(out / 'call.json', record)
    collector = SamStream(frame_count=frame_count, width=width, height=height,
                          object_ids=[obj['object_id'] for obj in prompts])
    reservation = None
    started = time.monotonic()
    try:
        reservation = reserve_call(project_root, estimate_usd=reservation_usd, request_sha256=record['request_sha256'])
        record.update(status='requesting', request_count=1, reservation=reservation)
        write_json(out / 'call.json', record)
        with _network_response(body, key, started) as response:
            record['http_status'] = response.status
            record['response_headers'] = {k: v for k, v in response.getheaders()
                                          if k.lower() in ('content-type', 'x-request-id', 'request-id', 'date')}
            if response.status != 200:
                (out / 'http-error.txt').write_bytes(response.read(8192))
                raise RuntimeError(f'HTTP {response.status}; no retry')
            if 'text/event-stream' not in response.getheader('Content-Type', ''):
                raise ValueError('Expected an SSE response')
            with (out / 'response.sse').open('wb') as raw:
                for event in _events(response, raw, started):
                    if isinstance(event, dict) and event.get('type') == 'response.completed':
                        result = event.get('response', {})
                        record.update(usage=result.get('usage'), response_id=result.get('id'))
                        record['cost_usd'] = _cost(record['usage'])
                    collector.feed(event)
        parsed = collector.finish()
        parsed['initialization'] = prompts
        parsed['missing_scope'] = 'No returned observation; pre-initialization slots are not graded detection failures'
        write_json(out / 'segmentation.json', parsed)
        record.update(status='completed', live_access_verified=True)
    except Exception as error:
        record.update(status='failed', error_type=type(error).__name__)
        raise
    finally:
        record['seconds'] = time.monotonic() - started
        try:
            if reservation:
                record['aggregate_budget'] = settle_call(project_root, reservation_id=reservation['reservation_id'],
                    actual_cost_usd=record['cost_usd'], outcome='completed' if record['status'] == 'completed' else 'failed')
        finally:
            write_json(out / 'call.json', record)
    return record
