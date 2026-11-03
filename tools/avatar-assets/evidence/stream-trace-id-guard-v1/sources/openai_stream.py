"""Bounded Responses SSE transport; completion is not model validation or settlement.

Socket deadlines exclude OS DNS stalls; the owning worker supplies the hard wall limit.
"""
from contextlib import suppress
import http.client
import json
import math
import os
from pathlib import Path
import re
import stat
import tempfile
import time

MAX_WIRE_BYTES = 16 * 1024**2
MAX_EVENT_BYTES = 4 * 1024**2
MAX_RESPONSE_BYTES = 4 * 1024**2
MAX_REQUEST_BYTES = 48 * 1024**2
MAX_EVENTS = 65536
MAX_TRACE_BYTES = 4096
_SAFE_ID = re.compile(r'^[A-Za-z0-9_.:-]{1,512}$')
_RESPONSE_ID = re.compile(r'^resp_[A-Za-z0-9_-]{1,250}$')
_EVENTS = frozenset(('response.created', 'response.in_progress', 'response.completed',
    'response.output_item.added', 'response.output_item.done', 'response.content_part.added',
    'response.content_part.done', 'response.output_text.delta', 'response.output_text.done',
    'response.output_text.annotation.added', 'response.reasoning_summary_part.added',
    'response.reasoning_summary_part.done', 'response.reasoning_summary_text.delta',
    'response.reasoning_summary_text.done', 'response.reasoning_text.delta', 'response.reasoning_text.done'))


class StreamError(ValueError):
    def __init__(self, code):
        self.code = code
        super().__init__('OpenAI stream '+code+'; reservation retained, no automatic retry')


def _require(condition, code):
    if not condition:
        raise StreamError(code)


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False).encode('utf-8')


def _decode(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            _require(key not in result, 'duplicate_json_key')
            result[key] = value
        return result
    try:
        return json.loads(raw, object_pairs_hook=pairs,
                          parse_constant=lambda _: (_ for _ in ()).throw(StreamError('nonfinite_json')))
    except (UnicodeError, json.JSONDecodeError, RecursionError):
        raise StreamError('invalid_event_json') from None


class _Trace:
    def __init__(self, path, client_id, started):
        self.path, self.started, self.saved = Path(path), started, None
        _require(self.path.is_absolute() and self.path.parent.is_dir()
                 and not any(p.is_symlink() for p in (self.path, *self.path.parents)), 'unsafe_trace_path')
        parent = self.path.parent.stat()
        _require(parent.st_uid == os.getuid() and not stat.S_IMODE(parent.st_mode) & 0o022, 'unsafe_trace_directory')
        self.value = {'schema_version': 1, 'transport': 'sse-v1', 'status': 'prepared',
                      'client_request_id': client_id, 'server_request_id': None, 'response_id': None,
                      'wire_bytes': 0, 'events': 0, 'last_sequence': None, 'elapsed_seconds': 0}
        self.write()

    def write(self):
        self.value['elapsed_seconds'] = round(time.monotonic()-self.started, 6)
        raw = _canonical(self.value)
        _require(len(raw) <= MAX_TRACE_BYTES, 'trace_size')
        if self.saved is not None:
            info = self.path.lstat()
            _require(stat.S_ISREG(info.st_mode) and info.st_uid == os.getuid()
                     and stat.S_IMODE(info.st_mode) == 0o600 and info.st_size == len(self.saved), 'trace_changed')
            fd = os.open(self.path, os.O_RDONLY | os.O_NOFOLLOW)
            with os.fdopen(fd, 'rb') as stream:
                _require(stream.read(MAX_TRACE_BYTES+1) == self.saved, 'trace_changed')
        fd, temporary = tempfile.mkstemp(prefix='.'+self.path.name+'.', dir=self.path.parent)
        try:
            with os.fdopen(fd, 'wb') as stream:
                stream.write(raw); stream.flush(); os.fsync(stream.fileno())
            if self.saved is None:
                os.link(temporary, self.path)
            else:
                os.replace(temporary, self.path)
            directory = os.open(self.path.parent, os.O_RDONLY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
            self.saved = raw
        finally:
            with suppress(FileNotFoundError):
                os.unlink(temporary)


class _Events:
    def __init__(self, trace, *, secret=None):
        self.trace = trace
        self._secret = secret
        self.buffer = bytearray()
        self.scan = 0
        self.data, self.event_name, self.event_bytes = [], None, 0
        self.completed, self.done = None, False

    def feed(self, chunk, *, eof=False):
        self.buffer.extend(chunk)
        while self.buffer:
            lf, cr = self.buffer.find(b'\n', self.scan), self.buffer.find(b'\r', self.scan)
            locations = [x for x in (lf, cr) if x >= 0]
            if not locations:
                self.scan = len(self.buffer)
                break
            end = min(locations)
            if self.buffer[end] == 13 and end+1 == len(self.buffer) and not eof:
                self.scan = end
                break
            size = end+1+(self.buffer[end:end+2] == b'\r\n')
            line = bytes(self.buffer[:end]); del self.buffer[:size]
            self.scan = 0
            self._line(line)
        _require(len(self.buffer)+self.event_bytes <= MAX_EVENT_BYTES, 'event_size')
        if eof:
            _require(not self.buffer and not self.data and self.event_name is None, 'truncated_event')
            _require(self.completed is not None, 'missing_completion')

    def _line(self, raw):
        self.event_bytes += len(raw)+1
        _require(self.event_bytes <= MAX_EVENT_BYTES, 'event_size')
        try:
            line = raw.decode('utf-8')
        except UnicodeError:
            raise StreamError('invalid_utf8') from None
        if not line:
            if self.data:
                self._event('\n'.join(self.data))
            self.data, self.event_name, self.event_bytes = [], None, 0
            return
        if line.startswith(':'):
            return
        field, sep, value = line.partition(':')
        if value.startswith(' '):
            value = value[1:]
        if field == 'data':
            self.data.append(value)
        elif field == 'event':
            _require(self.event_name is None and len(value) <= 128, 'invalid_event_field')
            self.event_name = value
        else:
            _require(field in ('id', 'retry'), 'unsupported_sse_field')

    def _event(self, raw):
        self.trace.value['events'] += 1
        _require(self.trace.value['events'] <= MAX_EVENTS, 'event_count')
        if raw == '[DONE]':
            _require(self.completed is not None and not self.done, 'unexpected_done')
            self.done = True
            return
        _require(self.completed is None and not self.done, 'event_after_completion')
        value = _decode(raw)
        _require(isinstance(value, dict), 'event_type')
        kind, sequence = value.get('type'), value.get('sequence_number')
        _require(type(sequence) is int and 0 <= sequence <= 2**53-1
                 and (self.trace.value['last_sequence'] is None or sequence > self.trace.value['last_sequence']),
                 'event_sequence')
        self.trace.value['last_sequence'] = sequence
        _require(self.event_name in (None, kind), 'event_name_mismatch')
        _require(kind not in ('error', 'response.failed', 'response.incomplete',
                             'response.refusal.delta', 'response.refusal.done'), 'provider_terminal_failure')
        _require(kind in _EVENTS, 'unsupported_event')
        response = value.get('response')
        if response is not None:
            _require(isinstance(response, dict) and isinstance(response.get('id'), str)
                     and _RESPONSE_ID.fullmatch(response['id'])
                     and (self._secret is None or self._secret not in response['id']), 'response_id')
            _require(kind == 'response.created' or response['id'] == self.trace.value['response_id'], 'response_id_changed')
        if 'response_id' in value:
            _require(value['response_id'] == self.trace.value['response_id'], 'response_id_changed')
        if kind == 'response.created':
            _require(self.trace.value['response_id'] is None and response is not None
                     and response.get('status') == 'in_progress', 'invalid_created')
            self.trace.value.update(response_id=response['id'], status='response_created')
            self.trace.write()
        else:
            _require(self.trace.value['response_id'] is not None, 'missing_created')
        for field, allowed in (('item', ('message', 'reasoning')), ('part', ('output_text', 'reasoning_text', 'summary_text'))):
            if field in value:
                _require(isinstance(value[field], dict) and value[field].get('type') in allowed, 'unsupported_output_item')
        if kind == 'response.completed':
            _require(response is not None and response.get('status') == 'completed'
                     and response.get('error') is None and response.get('incomplete_details') is None,
                     'invalid_completion')
            _require(isinstance(response.get('output'), list)
                     and all(isinstance(item, dict) and item.get('type') in ('message', 'reasoning')
                             for item in response['output']), 'unsupported_output_item')
            self.completed = _canonical(response)
            _require(len(self.completed) <= MAX_RESPONSE_BYTES, 'response_size')


def post_stream_response(payload, timeout, key, *, trace_path, client_request_id):
    """One request, no retries; only a complete terminal response reaches the caller."""
    _require(type(timeout) in (int, float) and math.isfinite(timeout) and 1 <= timeout <= 600, 'invalid_timeout')
    _require(isinstance(client_request_id, str) and _SAFE_ID.fullmatch(client_request_id), 'invalid_client_id')
    _require(isinstance(key, str) and 10 <= len(key) <= 4096 and all(33 <= ord(c) <= 126 for c in key)
             and key not in client_request_id, 'invalid_key')
    _require(isinstance(payload, dict) and payload.get('stream') is True and payload.get('store') is False
             and payload.get('tools') == [] and payload.get('background', False) is False, 'invalid_stream_settings')
    try:
        body = _canonical(payload)
    except (TypeError, ValueError, UnicodeError, RecursionError):
        raise StreamError('invalid_request_json') from None
    _require(len(body) <= MAX_REQUEST_BYTES, 'request_size')
    started = time.monotonic()
    trace = _Trace(trace_path, client_request_id, started)
    connection = None
    def remaining():
        value = timeout-(time.monotonic()-started)
        _require(value > 0, 'deadline')
        return value
    try:
        connection = http.client.HTTPSConnection('api.openai.com', timeout=remaining())
        connection.connect()
        sock = connection.sock
        sock.settimeout(remaining())
        connection.request('POST', '/v1/responses', body=body, headers={'Authorization': 'Bearer '+key,
            'Content-Type': 'application/json', 'Accept': 'text/event-stream', 'X-Client-Request-Id': client_request_id})
        sock.settimeout(remaining())
        response = connection.getresponse()
        request_id = response.getheader('x-request-id')
        trace.value.update(status='headers_received', server_request_id=request_id if isinstance(request_id,str)
                           and _SAFE_ID.fullmatch(request_id) and key not in request_id else None)
        trace.write()
        _require(response.status == 200, 'http_status')
        _require(response.getheader('Content-Encoding') in (None, 'identity'), 'compressed_response')
        content_type = response.getheader('Content-Type')
        _require(isinstance(content_type,str) and content_type.split(';',1)[0].strip().lower() == 'text/event-stream',
                 'content_type')
        length = response.getheader('Content-Length')
        _require(length is None or length.isascii() and length.isdigit() and int(length) <= MAX_WIRE_BYTES, 'wire_size')
        parser = _Events(trace, secret=key)
        while True:
            sock.settimeout(remaining())
            chunk = response.read1(min(65536, MAX_WIRE_BYTES+1-trace.value['wire_bytes']))
            remaining()
            _require(isinstance(chunk,bytes), 'invalid_transport_chunk')
            trace.value['wire_bytes'] += len(chunk)
            _require(trace.value['wire_bytes'] <= MAX_WIRE_BYTES, 'wire_size')
            if not chunk:
                break
            parser.feed(chunk)
        _require(length is None or trace.value['wire_bytes'] == int(length), 'truncated_response')
        parser.feed(b'', eof=True)
        trace.value['status'] = 'completed'; trace.write()
        return parser.completed
    except Exception as error:
        code = error.code if isinstance(error, StreamError) else 'transport_failed'
        trace.value.update(status='failed_or_uncertain', error_code=code)
        with suppress(Exception):
            trace.write()
        raise StreamError(code) from None
    finally:
        if connection is not None:
            with suppress(Exception):
                connection.close()
