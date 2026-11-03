"""Analytic stream integration; no real credentials, transport or native work."""
from copy import deepcopy
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import test_openai_runner as fixtures
from test_openai_runner import SCHEMA, response
from avatar_assets import openai_runner as api
from avatar_assets.agent_backend import DEFAULT, validate_settings
from avatar_assets.pipeline_continuation import _role


class StreamRunnerTests(unittest.TestCase):
    setUp=fixtures.RunnerTests.setUp
    runner=fixtures.RunnerTests.runner

    def completed(self,payload,timeout,key,*,trace_path,client_request_id):
        request=json.loads((trace_path.parent/'request.json').read_text())
        self.assertEqual(request['client_request_id'],client_request_id)
        self.assertEqual(self.ledger.call('test-job',request['role'])['state'],'reserved_uncertain')
        self.assertTrue(payload['stream']);self.assertFalse(payload['store']);self.assertEqual(payload['tools'],[])
        trace_path.write_text(json.dumps({'schema_version':1,'transport':'sse-v1','status':'completed',
            'client_request_id':client_request_id,'response_id':'resp_analytic'}))
        return response()

    def test_optional_configuration_preserves_old_shape_and_rejects_arbitrary_transports(self):
        self.assertEqual(validate_settings(DEFAULT),DEFAULT)
        self.assertEqual(validate_settings({**DEFAULT,'transport':'sse-v1'})['transport'],'sse-v1')
        for invalid in (None,True,'http','https://example.com',{},'sse-v2'):
            with self.subTest(value=invalid),self.assertRaises(ValueError):validate_settings({**DEFAULT,'transport':invalid})
        with self.assertRaises(ValueError):self.runner(transport='arbitrary')

    def test_complete_response_settles_and_cache_never_reposts(self):
        with patch('avatar_assets.openai_stream.post_stream_response',side_effect=self.completed) as stream:
            runner=self.runner(transport='sse-v1')
            self.assertEqual(runner.call('author-0','analytic',SCHEMA),{'value':3})
            self.assertEqual(runner.call('author-0','analytic',SCHEMA),{'value':3})
            self.assertEqual(stream.call_count,1);self.transport.assert_not_called()
            row=self.ledger.call('test-job','author-0');self.assertEqual(row['actual'],550)
            self.assertEqual(_role(self.root,'author-0',row,runner.settings),{'value':3})
            with self.assertRaisesRegex(ValueError,'settings cannot change'):self.runner()

    def test_partial_stream_retains_hold_and_client_id_without_an_answer_or_retry(self):
        def partial(*args,**kwargs):
            trace=kwargs['trace_path'];trace.write_text(json.dumps({'schema_version':1,'transport':'sse-v1',
                'status':'failed_or_uncertain','client_request_id':kwargs['client_request_id'],'response_id':'resp_early'}))
            raise ValueError('Stream interrupted')
        with patch('avatar_assets.openai_stream.post_stream_response',side_effect=partial) as stream:
            runner=self.runner(transport='sse-v1')
            with self.assertRaises(ValueError):runner.call('author-0','analytic',SCHEMA)
            self.assertEqual(api.call_exposure(self.ledger.call('test-job','author-0')),5618640)
            self.assertFalse((self.root/'agents/author-0/answer.json').exists())
            self.assertFalse((self.root/'agents/author-0/response.json').exists())
            with self.assertRaises(ValueError):runner.call('author-0','analytic',SCHEMA)
            self.assertEqual(stream.call_count,1)

    def test_trace_mismatch_does_not_settle_or_accept_response(self):
        def wrong(*args,**kwargs):
            raw=self.completed(*args,**kwargs);path=kwargs['trace_path']
            trace=json.loads(path.read_text());trace['response_id']='resp_wrong';path.write_text(json.dumps(trace))
            return raw
        with patch('avatar_assets.openai_stream.post_stream_response',side_effect=wrong):
            with self.assertRaisesRegex(ValueError,'identity differs'):self.runner(transport='sse-v1').call('planner','analytic',SCHEMA)
        self.assertEqual(self.ledger.call('test-job','planner')['state'],'reserved_uncertain')

    def test_completed_transport_does_not_bypass_usage_model_or_tool_validation(self):
        def bad(*args,**kwargs):
            self.completed(*args,**kwargs)
            return response(model='unpriced')
        with patch('avatar_assets.openai_stream.post_stream_response',side_effect=bad):
            with self.assertRaisesRegex(ValueError,'unexpected API response'):
                self.runner(transport='sse-v1').call('planner','analytic',SCHEMA)
        self.assertEqual(self.ledger.call('test-job','planner')['state'],'reserved_uncertain')

    def test_trace_tampering_breaks_cached_custody(self):
        with patch('avatar_assets.openai_stream.post_stream_response',side_effect=self.completed) as stream:
            runner=self.runner(transport='sse-v1');runner.call('planner','analytic',SCHEMA)
            path=self.root/'agents/planner/transport.json';trace=json.loads(path.read_text());trace['status']='response_created'
            path.write_text(json.dumps(trace))
            with self.assertRaises(ValueError):runner.call('planner','analytic',SCHEMA)
            self.assertEqual(stream.call_count,1)

    def test_real_stream_parser_and_runner_settle_only_complete_http_fixture(self):
        from test_openai_stream import FakeConnection,FakeResponse,encode
        data=json.loads(response())
        events=[{'type':'response.created','sequence_number':0,
                 'response':{'id':data['id'],'status':'in_progress'}},
                {'type':'response.output_text.delta','sequence_number':1,'delta':'not accepted separately'},
                {'type':'response.completed','sequence_number':2,'response':data}]
        raw=encode(events,ending='\r\n',multiline=True)
        connection=FakeConnection(FakeResponse(b'',chunks=[raw[n:n+7] for n in range(0,len(raw),7)]))
        with patch('avatar_assets.openai_stream.http.client.HTTPSConnection',return_value=connection):
            runner=self.runner(transport='sse-v1');self.assertEqual(runner.call('planner','analytic',SCHEMA),{'value':3})
        trace=json.loads((self.root/'agents/planner/transport.json').read_text())
        self.assertEqual(trace['status'],'completed');self.assertEqual(trace['server_request_id'],'req_synthetic')
        self.assertEqual(connection.requests[0][3]['X-Client-Request-Id'],trace['client_request_id'])
        self.assertEqual(self.ledger.call('test-job','planner')['actual'],550)
        self.assertEqual(len(connection.requests),1);self.assertTrue(connection.closed)

    def test_real_stream_timeout_preserves_early_identity_and_uncertain_reservation(self):
        from test_openai_stream import FakeConnection,FakeResponse,encode
        raw=encode([{'type':'response.created','sequence_number':0,
                     'response':{'id':'resp_early','status':'in_progress'}}])
        connection=FakeConnection(FakeResponse(b'',chunks=[raw],fail_after=1))
        with patch('avatar_assets.openai_stream.http.client.HTTPSConnection',return_value=connection):
            with self.assertRaisesRegex(ValueError,'transport_failed'):
                self.runner(transport='sse-v1').call('planner','analytic',SCHEMA)
        folder=self.root/'agents/planner';trace=json.loads((folder/'transport.json').read_text())
        self.assertEqual(trace['response_id'],'resp_early');self.assertEqual(trace['status'],'failed_or_uncertain')
        self.assertEqual(api.call_exposure(self.ledger.call('test-job','planner')),5618640)
        self.assertFalse((folder/'answer.json').exists());self.assertFalse((folder/'response.json').exists())
        self.assertEqual(len(connection.requests),1);self.assertTrue(connection.closed)


class StreamStudioSettingsTests(unittest.TestCase):
    def test_stream_selection_is_pinned_through_studio_pipeline(self):
        from test_openai_studio_integration import OpenAIStudioIntegrationTests
        from avatar_assets import pipeline
        f=OpenAIStudioIntegrationTests();f.setUp();self.addCleanup(f.doCleanups)
        f.settings={**f.settings,'transport':'sse-v1'};f.config();f.key();f.prepare();ledger=f.grant()
        original=(f.out/'request.json').read_bytes()
        with patch('avatar_assets.openai_runner.OpenAIRunner') as runner,patch('avatar_assets.openai_runner.OpenAILedger',return_value=ledger):
            pipeline.Pipeline(f.out)
        self.assertEqual(runner.call_args.kwargs['transport'],'sse-v1')
        old=deepcopy(f.settings);old.pop('transport');f.config(old)
        with self.assertRaises(ValueError):pipeline.Pipeline(f.out)
        self.assertEqual((f.out/'request.json').read_bytes(),original)


if __name__=='__main__':unittest.main()
