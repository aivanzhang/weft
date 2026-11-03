"""Unanswered streaming author fixtures; no provider or native calls."""
import json
from contextlib import ExitStack
import unittest
from unittest.mock import patch

from avatar_assets import openai_runner as api, studio_unknown_release as release
from avatar_assets.safety import sha256,read_json
from tests.test_author_input_recovery import AuthorInputRecoveryTests
from tests.test_studio_api_release import StudioApiReleaseTests,write


class UnknownStreamReleaseTests(unittest.TestCase):
    def setUp(self):
        fixture=AuthorInputRecoveryTests('runTest')
        with patch.object(StudioApiReleaseTests,'fixture_stream',True,create=True), \
                patch.object(StudioApiReleaseTests,'fixture_model','gpt-5.2-2025-12-11',create=True):
            fixture.setUp()
        self.addCleanup(fixture.doCleanups);self.f=fixture.f;self.job=fixture.job;self.jobs=self.f.jobs
        self.message='OpenAI stream unsupported_event; reservation retained, no automatic retry'
        def failed(payload,timeout,key,*,trace_path,client_request_id):
            write(trace_path,{'schema_version':1,'transport':'sse-v1','status':'failed_or_uncertain',
                'client_request_id':client_request_id,'server_request_id':'req_analytic','response_id':'resp_analytic_unsettled',
                'wire_bytes':100,'events':4,'last_sequence':3,'elapsed_seconds':31.0,'error_code':'unsupported_event'})
            raise ValueError(self.message)
        ledger=api.OpenAILedger(self.f.root/'.local/openai')
        try:
            with patch.object(api,'read_api_key',return_value='ANALYTIC NOT REAL'), \
                    patch('avatar_assets.openai_stream.post_stream_response',side_effect=failed):
                runner=api.OpenAIRunner(self.job/'agents',job_id=self.job.name,job_digest=sha256(self.job/'request.json'),
                    ledger=ledger,max_calls=7,model='gpt-5.2-2025-12-11',transport='sse-v1')
                with self.assertRaisesRegex(ValueError,'unsupported_event'):
                    runner.call('author-0','Analytic author',{'type':'object','properties':{},'required':[],'additionalProperties':False})
        finally:ledger.close()
        write(self.job/'status.json',{'status':'checkpointed_error','stage':'pipeline','automatic_retry':False,'error':self.message})
        self.stack=ExitStack();self.addCleanup(self.stack.close)
        self.stack.enter_context(patch.object(api,'read_api_key',side_effect=AssertionError('No API key')))
        self.stack.enter_context(patch.object(api,'post_response',side_effect=AssertionError('No API')))
        self.stack.enter_context(patch('avatar_assets.openai_stream.post_stream_response',side_effect=AssertionError('No stream')))
        self.stack.enter_context(patch.object(self.jobs,'_dispatch',side_effect=AssertionError('No dispatch')))

    def invoke(self):
        return self.jobs.release_uncertain_inspected(self.f.job,expected_pid=self.f.owner_pid,
            expected_identity=self.f.identity,reason='Analytic unknown streaming request; hold preserved.')

    def change_trace(self,**values):
        path=self.job/'agents/author-0/transport.json';write(path,{**read_json(path),**values})

    def test_stream_identity_does_not_settle_or_reduce_hold(self):
        ledger=sha256(self.f.ledger);result=self.invoke()
        record=release.verify_release(self.f.root,self.jobs.state_dir/result['receipt'],result['sha256'],
            parent_job_id=self.job.name,parent_digest=sha256(self.job/'request.json'))
        proof=record['proof']['unknown_provider']
        self.assertEqual(result['unknown_reserved_microdollars'],929376)
        self.assertEqual(proof['stream']['observed_response_id'],'resp_analytic_unsettled')
        self.assertIsNone(proof['uncertain_call']['response_id']);self.assertEqual(sha256(self.f.ledger),ledger)
        self.assertTrue(proof['stream']['response_id_is_not_settlement'])
        self.assertFalse(result['old_job_restart_allowed'])

    def test_wrong_client_id_rejected(self):
        self.change_trace(client_request_id='avatar_wrong')
        with self.assertRaisesRegex(ValueError,'trace differs'):self.invoke()

    def test_completed_or_other_failure_trace_rejected(self):
        self.change_trace(status='completed')
        with self.assertRaisesRegex(ValueError,'trace differs'):self.invoke()
        self.change_trace(status='failed_or_uncertain',error_code='some_new_error')
        with self.assertRaisesRegex(ValueError,'trace differs'):self.invoke()

    def test_trace_identity_and_size_bounds(self):
        self.change_trace(response_id='https://untrusted.invalid/response')
        with self.assertRaisesRegex(ValueError,'identity'):self.invoke()
        self.change_trace(response_id=None,wire_bytes=16*1024**2+1)
        with self.assertRaisesRegex(ValueError,'counters'):self.invoke()
        self.change_trace(wire_bytes=100,server_request_id='x'*5000)
        with self.assertRaisesRegex(ValueError,'oversized'):self.invoke()

    def test_pinned_stream_module_and_model_profile_required(self):
        path=self.job/'agents/runner.json';saved=read_json(path)
        write(path,{**saved,'transport_implementation_sha256':'f'*64})
        with self.assertRaisesRegex(ValueError,'source pin'):self.invoke()
        write(path,{**saved,'model_profile_implementation_sha256':'f'*64})
        with self.assertRaisesRegex(ValueError,'source pin'):self.invoke()

    def test_missing_trace_or_response_file_rejected(self):
        path=self.job/'agents/author-0/transport.json';saved=path.read_bytes();path.unlink()
        with self.assertRaisesRegex(ValueError,'extra|files'):self.invoke()
        path.write_bytes(saved);write(self.job/'agents/author-0/response.json',{'id':'resp_unvalidated'})
        with self.assertRaisesRegex(ValueError,'response|extra'):self.invoke()


if __name__=='__main__':unittest.main()
