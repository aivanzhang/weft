"""Mocked HTTP/SSE custody tests; no keys, provider calls or charge settlement."""
from copy import deepcopy
import json
import os
from pathlib import Path
import socket
import stat
import tempfile
import unittest
from unittest.mock import patch

from avatar_assets import openai_stream as stream

KEY = 'synthetic-test-secret-not-real'
CLIENT = 'avatar-'+'a'*48
RID = 'resp_test123'


def completed():
    return {'id': RID, 'status': 'completed', 'model': 'pinned-test-model', 'error': None,
            'incomplete_details': None, 'output': [{'type': 'message', 'role': 'assistant', 'status': 'completed',
             'content': [{'type': 'output_text', 'text': '{"subject":"café owl"}'}]}],
            'usage': {'input_tokens': 100, 'output_tokens': 20, 'total_tokens': 120}}


def events():
    return [{'type': 'response.created', 'sequence_number': 0,
             'response': {'id': RID, 'status': 'in_progress'}},
            {'type': 'response.output_text.delta', 'sequence_number': 1, 'delta': 'café partial text'},
            {'type': 'response.completed', 'sequence_number': 2, 'response': completed()}]


def encode(rows, ending='\n', multiline=False):
    pieces = []
    for row in rows:
        raw = json.dumps(row, ensure_ascii=False)
        if multiline:
            # JSON whitespace can span SSE data lines without changing decoded values.
            at = raw.index(',')+1
            data = 'data: '+raw[:at]+ending+'data: '+raw[at:]
        else:
            data = 'data: '+raw
        pieces.append('event: '+row['type']+ending+data+ending+ending)
    return ''.join(pieces).encode()


class FakeSocket:
    def __init__(self): self.timeouts = []
    def settimeout(self, value): self.timeouts.append(value)


class FakeResponse:
    def __init__(self, raw, *, chunks=None, headers=None, status=200, fail_after=None, on_read=None):
        self.chunks = list(chunks) if chunks is not None else [raw]
        self.headers = {'Content-Type': 'text/event-stream', 'x-request-id': 'req_synthetic'}
        self.headers.update(headers or {})
        self.status, self.fail_after, self.on_read, self.reads = status, fail_after, on_read, 0
    def getheader(self, name): return self.headers.get(name)
    def read1(self, maximum):
        self.reads += 1
        if self.on_read: self.on_read()
        if self.fail_after is not None and self.reads > self.fail_after:
            raise socket.timeout('untrusted exception '+KEY)
        if not self.chunks: return b''
        chunk = self.chunks.pop(0)
        if len(chunk) > maximum:
            self.chunks.insert(0, chunk[maximum:]); chunk = chunk[:maximum]
        return chunk


class FakeConnection:
    def __init__(self, response, before_request=None):
        self.response, self.before_request = response, before_request
        self.sock, self.closed, self.requests = FakeSocket(), False, []
    def connect(self): pass
    def request(self, method, url, body, headers):
        if self.before_request: self.before_request()
        self.requests.append((method, url, json.loads(body), headers))
    def getresponse(self):
        self.sock = None  # Connection:close leaves ownership with HTTPResponse.
        return self.response
    def close(self): self.closed = True


class StreamTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name).resolve(); self.trace = self.folder/'transport.json'
        self.payload = {'stream': True, 'store': False, 'tools': [], 'model': 'pinned-test-model',
                        'input': 'Private synthetic prompt never stored in the trace.'}
    def call(self, response, *, payload=None, timeout=60, client=CLIENT, connection=None):
        conn = connection or FakeConnection(response)
        with patch.object(stream.http.client, 'HTTPSConnection', return_value=conn) as factory:
            try:
                return stream.post_stream_response(self.payload if payload is None else payload, timeout, KEY,
                    trace_path=self.trace, client_request_id=client)
            finally:
                self.last = conn
                self.connection_calls = factory.call_count
                if factory.called: self.assertEqual(factory.call_args.args, ('api.openai.com',))
    def trace_value(self): return json.loads(self.trace.read_text())
    def test_fragmented_utf8_crlf_multiline_returns_only_terminal_and_four_trace_writes(self):
        raw = encode(events(), ending='\r\n', multiline=True)+b'data: [DONE]\r\n\r\n'
        response = FakeResponse(raw, chunks=[raw[n:n+1] for n in range(len(raw))])
        writes = []; original = stream._Trace.write
        def capture(value):
            original(value); writes.append(json.loads(value.path.read_text()))
        conn = FakeConnection(response, before_request=lambda: self.assertEqual(self.trace_value()['status'], 'prepared'))
        with patch.object(stream._Trace, 'write', capture): result = self.call(response, connection=conn)
        self.assertEqual(json.loads(result), completed()); self.assertTrue(conn.closed)
        self.assertEqual([v['status'] for v in writes], ['prepared','headers_received','response_created','completed'])
        trace = self.trace_value()
        self.assertEqual((trace['client_request_id'], trace['server_request_id'], trace['response_id']), (CLIENT,'req_synthetic',RID))
        self.assertEqual(trace['events'],4); self.assertEqual(trace['wire_bytes'],len(raw))
        self.assertEqual(stat.S_IMODE(self.trace.stat().st_mode),0o600)
        for secret in (KEY, 'café', 'Private synthetic', 'partial text'):
            self.assertNotIn(secret,self.trace.read_text())
        self.assertEqual(len(conn.requests),1)
        method,url,payload,headers=conn.requests[0]
        self.assertEqual((method,url),('POST','/v1/responses'))
        self.assertEqual(payload,self.payload);self.assertEqual(headers['X-Client-Request-Id'],CLIENT)
    def test_failure_after_created_preserves_id_without_partial_text_or_retry(self):
        response=FakeResponse(b'',chunks=[encode(events()[:2])],fail_after=1)
        with self.assertRaisesRegex(stream.StreamError,'transport_failed') as caught:self.call(response)
        self.assertNotIn(KEY,str(caught.exception));self.assertEqual(len(self.last.requests),1);self.assertTrue(self.last.closed)
        trace=self.trace_value();self.assertEqual(trace['response_id'],RID);self.assertEqual(trace['status'],'failed_or_uncertain')
        self.assertNotIn('partial text',self.trace.read_text())
    def test_created_id_cannot_copy_secret_into_any_trace_write(self):
        for index, unsafe_id in enumerate(('resp_'+KEY, 'resp_prefix_'+KEY+'_suffix')):
            with self.subTest(index=index):
                self.trace = self.folder/f'secret-id-{index}.json'
                rows = events(); rows[0]['response']['id'] = unsafe_id
                writes = []; original = stream._Trace.write
                def capture(value):
                    original(value); writes.append(value.path.read_text())
                with patch.object(stream._Trace, 'write', capture):
                    with self.assertRaisesRegex(stream.StreamError, 'response_id') as caught:
                        self.call(FakeResponse(encode(rows)))
                self.assertTrue(writes)
                for raw in writes:
                    self.assertNotIn(KEY, raw)
                    self.assertIsNone(json.loads(raw)['response_id'])
                self.assertNotIn(KEY, str(caught.exception))
                self.assertNotIn(KEY, repr(caught.exception))
                self.assertEqual(self.trace_value()['status'], 'failed_or_uncertain')
                self.assertEqual(self.connection_calls, 1)
                self.assertEqual(len(self.last.requests), 1)
                self.assertTrue(self.last.closed)
    def test_provider_failures_and_incomplete_never_return_an_answer(self):
        for kind in ('error','response.failed','response.incomplete','response.refusal.delta'):
            with self.subTest(kind=kind):
                self.trace=self.folder/(kind+'.json')
                rows=events()[:1]+[{'type':kind,'sequence_number':1,'message':KEY,'response':{'id':RID,'status':'failed'}}]
                with self.assertRaisesRegex(stream.StreamError,'provider_terminal_failure'):self.call(FakeResponse(encode(rows)))
                self.assertEqual(self.trace_value()['response_id'],RID);self.assertNotIn(KEY,self.trace.read_text())
    def test_missing_created_missing_completion_and_truncated_frames(self):
        for name,raw,code in [('no-created',encode(events()[1:]),'missing_created'),
                              ('no-completion',encode(events()[:2]),'missing_completion'),
                              ('truncated',encode(events())[:-1],'truncated_event')]:
            with self.subTest(name=name):
                self.trace=self.folder/(name+'.json')
                with self.assertRaisesRegex(stream.StreamError,code):self.call(FakeResponse(raw))
    def test_sequence_id_and_terminal_identity_are_strict(self):
        variants=[]
        x=events();x[1]['sequence_number']=0;variants.append((x,'event_sequence'))
        x=events();x[1]['sequence_number']=True;variants.append((x,'event_sequence'))
        x=events();x[2]['response']['id']='resp_changed';variants.append((x,'response_id_changed'))
        x=events();x.insert(1,{**deepcopy(x[0]),'sequence_number':1});variants.append((x,'invalid_created'))
        x=events();x.append({**deepcopy(x[-1]),'sequence_number':3});variants.append((x,'event_after_completion'))
        x=events();x[1]['response_id']='resp_changed';variants.append((x,'response_id_changed'))
        for index,(rows,code) in enumerate(variants):
            with self.subTest(index=index):
                self.trace=self.folder/f'variant-{index}.json'
                with self.assertRaisesRegex(stream.StreamError,code):self.call(FakeResponse(encode(rows)))
    def test_tool_events_and_terminal_tool_items_are_rejected(self):
        rows=events();rows[1]={'type':'response.output_item.added','sequence_number':1,'item':{'type':'function_call'}}
        with self.assertRaisesRegex(stream.StreamError,'unsupported_output_item'):self.call(FakeResponse(encode(rows)))
        self.trace=self.folder/'terminal-tool.json';rows=events();rows[-1]['response']['output']=[{'type':'function_call'}]
        with self.assertRaisesRegex(stream.StreamError,'unsupported_output_item'):self.call(FakeResponse(encode(rows)))
    def test_redirect_encoding_content_type_and_declared_size_reject(self):
        variants=[({'status':307},'http_status'),({'headers':{'Content-Encoding':'gzip'}},'compressed_response'),
                  ({'headers':{'Content-Type':'application/json'}},'content_type'),
                  ({'headers':{'Content-Length':str(stream.MAX_WIRE_BYTES+1)}},'wire_size'),
                  ({'headers':{'Content-Length':str(len(encode(events()))+1)}},'truncated_response')]
        for index,(kwargs,code) in enumerate(variants):
            with self.subTest(index=index):
                self.trace=self.folder/f'headers-{index}.json'
                with self.assertRaisesRegex(stream.StreamError,code):self.call(FakeResponse(encode(events()),**kwargs))
                self.assertTrue(self.last.closed);self.assertEqual(len(self.last.requests),1)
    def test_invalid_provider_header_is_omitted(self):
        self.call(FakeResponse(encode(events()),headers={'x-request-id':'req_bad\n'+KEY}))
        self.assertIsNone(self.trace_value()['server_request_id']);self.assertNotIn(KEY,self.trace.read_text())
    def test_wire_event_response_and_event_count_limits(self):
        variants=[('MAX_WIRE_BYTES',100,encode(events()),'wire_size'),
                  ('MAX_EVENT_BYTES',100,b':'+b'x'*101+b'\n\n','event_size'),
                  ('MAX_RESPONSE_BYTES',40,encode(events()),'response_size'),
                  ('MAX_EVENTS',2,encode(events()),'event_count')]
        for index,(constant,bound,raw,code) in enumerate(variants):
            with self.subTest(constant=constant):
                self.trace=self.folder/f'bound-{index}.json'
                with patch.object(stream,constant,bound),self.assertRaisesRegex(stream.StreamError,code):self.call(FakeResponse(raw))
    def test_absolute_deadline_during_read_retains_created_id(self):
        now=[0.0]
        def advance():now[0]+=35
        response=FakeResponse(b'',chunks=[encode(events()[:1]),encode(events()[1:])],on_read=advance)
        with patch.object(stream.time,'monotonic',side_effect=lambda:now[0]),self.assertRaisesRegex(stream.StreamError,'deadline'):
            self.call(response,timeout=60)
        self.assertEqual(self.trace_value()['response_id'],RID);self.assertTrue(self.last.closed)
    def test_dns_return_after_deadline_cannot_submit_late(self):
        now=[0.0];conn=FakeConnection(FakeResponse(encode(events())))
        def slow_connect():now[0]=61
        conn.connect=slow_connect
        with patch.object(stream.time,'monotonic',side_effect=lambda:now[0]),self.assertRaisesRegex(stream.StreamError,'deadline'):
            self.call(conn.response,connection=conn,timeout=60)
        self.assertEqual(conn.requests,[]);self.assertTrue(conn.closed)
    def test_invalid_input_never_connects_or_overwrites_trace(self):
        for field,value in [('stream',False),('store',True),('tools',[{'type':'web_search'}]),('background',True)]:
            payload={**self.payload,field:value}
            with self.subTest(field=field),patch.object(stream.http.client,'HTTPSConnection') as conn:
                with self.assertRaises(stream.StreamError):self.call(FakeResponse(b''),payload=payload)
                conn.assert_not_called();self.assertEqual(self.connection_calls,0);self.assertFalse(self.trace.exists())
        with self.assertRaisesRegex(stream.StreamError,'invalid_client_id'):self.call(FakeResponse(b''),client='bad\r\nAuthorization')
        for timeout in (0,601,float('nan'),True):
            with self.assertRaisesRegex(stream.StreamError,'invalid_timeout'):self.call(FakeResponse(b''),timeout=timeout)
    def test_trace_is_exclusive_and_symlinks_are_rejected_before_network(self):
        self.trace.write_text('existing evidence');os.chmod(self.trace,0o600)
        with self.assertRaises(FileExistsError):self.call(FakeResponse(encode(events())))
        self.assertEqual(self.trace.read_text(),'existing evidence')
        linked=self.folder/'link.json';linked.symlink_to(self.trace);self.trace=linked
        with self.assertRaisesRegex(stream.StreamError,'unsafe_trace_path'):self.call(FakeResponse(b''))
    def test_json_utf8_name_and_done_sentinels_fail_closed(self):
        variants=[b'data: {"type":"response.created","type":"response.created"}\n\n',
                  b'data: \xff\n\n',b'data: [DONE]\n\n',
                  encode(events()).replace(b'event: response.created',b'event: response.completed',1)]
        for index,raw in enumerate(variants):
            self.trace=self.folder/f'malformed-{index}.json'
            with self.subTest(index=index),self.assertRaises(stream.StreamError):self.call(FakeResponse(raw))

    def test_long_fragmented_line_scans_only_new_bytes(self):
        class ObservedBuffer(bytearray):
            scanned = 0
            def find(self, needle, start=0, *args):
                self.scanned += len(self)-start
                return super().find(needle,start,*args)
        trace=stream._Trace(self.trace,CLIENT,stream.time.monotonic())
        parser=stream._Events(trace);parser.buffer=ObservedBuffer()
        rows=events();rows[1]['delta']='x'*(128*1024)
        raw=encode(rows)
        for i in range(0,len(raw),17):parser.feed(raw[i:i+17])
        parser.feed(b'',eof=True)
        self.assertEqual(json.loads(parser.completed),completed())
        self.assertLess(parser.buffer.scanned,5*len(raw))


if __name__=='__main__':unittest.main()
