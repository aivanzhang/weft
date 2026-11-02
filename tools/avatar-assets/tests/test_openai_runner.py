"""Analytic provider fixtures only: no real key, model call or billing."""
import copy
import json
import io
import os
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import Mock, patch

from PIL import Image

from avatar_assets import openai_runner as api

SCHEMA = {'type':'object','properties':{'value':{'type':'integer','minimum':0,'maximum':5}},
          'required':['value'],'additionalProperties':False}
APPROVAL = {'decision':'approved','user_reply':'Analytic test approval; not a real spending grant'}
SECRET = 'sk-analytic-only-never-real'


def response(**changes):
    data = {'id':'resp_analytic','model':api.MODEL,'service_tier':'default','status':'completed',
            'usage':{'input_tokens':100,'output_tokens':20,'total_tokens':120},
            'output':[{'type':'message','role':'assistant','status':'completed',
                       'content':[{'type':'output_text','text':'{"value":3}'}]}]}
    data.update(changes)
    return json.dumps(data).encode()


class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.ledger = api.OpenAILedger(self.root/'ledger')
        self.addCleanup(self.ledger.close)
        self.ledger.configure(100_000_000, APPROVAL)
        self.ledger.authorize_job('test-job','a'*64,3,30_000_000,APPROVAL)
        self.key = patch.object(api,'read_api_key',return_value=SECRET).start()
        self.transport = patch.object(api,'post_response',return_value=response()).start()
        self.addCleanup(patch.stopall)

    def runner(self, **kwargs):
        return api.OpenAIRunner(self.root/'agents',job_id='test-job',job_digest='a'*64,
                               ledger=self.ledger,max_calls=3,**kwargs)

    def test_success_exact_cache_usage_and_prompt_custody(self):
        runner = self.runner()
        self.assertEqual(runner.call('planner','Create a robot',SCHEMA),{'value':3})
        self.assertEqual(self.runner().call('planner','Create a robot',SCHEMA),{'value':3})
        self.assertEqual(self.transport.call_count,1)
        self.assertEqual(self.key.call_count,1)
        payload = self.transport.call_args.args[0]
        self.assertEqual(payload['text']['format']['schema'],SCHEMA)
        self.assertFalse(payload['store'])
        self.assertEqual(payload['tools'],[])
        self.assertEqual(payload['tool_choice'],'none')
        self.assertEqual(payload['truncation'],'disabled')
        self.assertEqual(payload['model'],api.MODEL)
        receipt = json.loads((self.root/'agents/planner/receipt.json').read_text())
        self.assertEqual(receipt['actual_microdollars_upper_estimate'],550)
        self.assertEqual(receipt['reserved_microdollars'],5_618_640)
        self.assertEqual(self.ledger.call('test-job','planner')['state'],'settled')
        for item in (self.root/'agents').rglob('*'):
            if item.is_file():
                self.assertNotIn(SECRET.encode(), item.read_bytes())

    def test_changed_inputs_and_artifacts_fail_without_resubmission(self):
        runner = self.runner(); runner.call('planner','robot',SCHEMA)
        with self.assertRaises(ValueError):
            runner.call('planner','different robot',SCHEMA)
        (self.root/'agents/planner/answer.json').write_text('{"value":4}')
        with self.assertRaises(ValueError):
            runner.call('planner','robot',SCHEMA)
        self.assertEqual(self.transport.call_count,1)

    def test_image_request_is_exact_and_locally_preserved(self):
        picture = self.root/'reference.png'
        Image.new('RGB',(16,16),(22,33,44)).save(picture)
        self.runner().call('planner','robot',SCHEMA,images=[picture])
        content = self.transport.call_args.args[0]['input'][0]['content']
        self.assertTrue(content[1]['image_url'].startswith('data:image/png;base64,'))
        self.assertEqual(content[1]['detail'],'high')
        self.assertEqual((self.root/'agents/planner/image-00.png').read_bytes(),picture.read_bytes())

    def test_timeout_retains_uncertain_charge_and_cannot_retry(self):
        self.transport.side_effect = TimeoutError('analytic timeout')
        runner = self.runner()
        with self.assertRaises(TimeoutError):
            runner.call('planner','robot',SCHEMA)
        self.assertEqual(self.ledger.call('test-job','planner')['state'],'reserved_uncertain')
        with self.assertRaises(ValueError):
            runner.call('planner','robot',SCHEMA)
        self.assertEqual(self.transport.call_count,1)

    def test_refusal_is_charged_but_not_successful(self):
        self.transport.return_value = response(output=[{'type':'message','role':'assistant','status':'completed',
                                                        'content':[{'type':'refusal','refusal':'No'}]}])
        with self.assertRaisesRegex(ValueError,'refused'):
            self.runner().call('planner','robot',SCHEMA)
        self.assertEqual(self.ledger.call('test-job','planner')['state'],'settled')
        self.assertEqual(json.loads((self.root/'agents/planner/receipt.json').read_text())['status'],'failed_or_uncertain')

    def test_incomplete_tool_output_wrong_usage_model_and_bad_json_rejected(self):
        cases = [response(status='incomplete'), response(model='unpriced'),
                 response(usage={'input_tokens':True,'output_tokens':0,'total_tokens':1}),
                 response(usage={'input_tokens':1,'output_tokens':32769,'total_tokens':32770}),
                 response(output=[{'type':'function_call','name':'danger'}]), b'{"x":1,"x":2}',
                 b'{broken',response(output=[{'type':'message','role':'assistant','status':'completed',
                                            'content':[{'type':'output_text','text':'{"value":100}'}]}])]
        for index, raw in enumerate(cases):
            with self.subTest(index=index):
                name='job-'+str(index)
                ledger=api.OpenAILedger(self.root/('ledger-'+str(index)))
                try:
                    ledger.configure(10_000_000,APPROVAL)
                    ledger.authorize_job(name,'a'*64,1,10_000_000,APPROVAL)
                    runner=api.OpenAIRunner(self.root/('agents-'+str(index)),job_id=name,job_digest='a'*64,
                                           ledger=ledger,max_calls=1)
                    self.transport.return_value=raw
                    with self.assertRaises((ValueError,KeyError)):
                        runner.call('planner','robot',SCHEMA)
                    self.assertIsNotNone(ledger.call(name,'planner'))
                finally:
                    ledger.close()

    def test_prompt_schema_and_image_limits_prevent_transport(self):
        runner=self.runner()
        cases=[('x'*(api.MAX_PROMPT_BYTES+1),SCHEMA,[]),('robot',{'type':'object'},[]),
               ('robot',SCHEMA,[self.root/'missing.png']*9)]
        for prompt,schema,images in cases:
            with self.assertRaises(ValueError):
                runner.call('planner',prompt,schema,images)
        self.transport.assert_not_called(); self.key.assert_not_called()
        self.assertIsNone(self.ledger.call('test-job','planner'))

    def test_response_limit_preserves_reservation(self):
        self.transport.return_value=b' '*(api.MAX_RESPONSE+1)
        with self.assertRaises(ValueError):
            self.runner().call('planner','robot',SCHEMA)
        self.assertEqual(self.ledger.call('test-job','planner')['state'],'reserved_uncertain')
        self.assertFalse((self.root/'agents/planner/response.json').exists())

    def test_settings_and_missing_grant_fail_before_key_or_transport(self):
        self.runner()
        with self.assertRaises(ValueError):
            self.runner(timeout=20)
        with self.assertRaises(ValueError):
            self.runner(model='gpt-unpriced')
        with self.assertRaises(ValueError):
            api.OpenAIRunner(self.root/'other',job_id='missing',job_digest='b'*64,ledger=self.ledger)
        self.key.assert_not_called(); self.transport.assert_not_called()

    def test_real_pipeline_schemas_support_strict_outputs(self):
        from avatar_assets.pipeline_prompts import PLAN_SCHEMA,AUTHOR_SCHEMA,CRITIC_SCHEMA
        for schema in (PLAN_SCHEMA,AUTHOR_SCHEMA,CRITIC_SCHEMA):
            api._strict(schema)
        self.assertEqual(AUTHOR_SCHEMA['properties']['design_json']['type'],'string')

    def test_project_grants_and_calls_cannot_be_reset(self):
        with self.assertRaises(ValueError):
            self.ledger.configure(101_000_000,APPROVAL)
        with self.assertRaises(ValueError):
            self.ledger.authorize_job('test-job','b'*64,3,30_000_000,APPROVAL)
        with self.assertRaises(ValueError):
            self.ledger.authorize_job('other','b'*64,3,80_000_000,APPROVAL)
        self.ledger.reserve('test-job','a'*64,3,'planner','c'*64,29_000_000)
        with self.assertRaises(ValueError):
            self.ledger.reserve('test-job','a'*64,3,'planner','c'*64,1)
        with self.assertRaises(ValueError):
            self.ledger.reserve('test-job','a'*64,3,'critic','d'*64,2_000_000)
        for amount in (-1,True,0):
            with self.assertRaises(ValueError):
                self.ledger.reserve('test-job','a'*64,3,'critic','d'*64,amount)
        self.assertEqual(self.ledger.call('test-job','planner')['reserved'],29_000_000)

    def test_call_limit_survives_reopening_and_exact_binding(self):
        for role in ('planner','author','critic'):
            self.ledger.reserve('test-job','a'*64,3,role,'c'*64,10)
        other=api.OpenAILedger(self.root/'ledger')
        try:
            with self.assertRaises(ValueError):
                other.reserve('test-job','a'*64,3,'repair','d'*64,10)
            with self.assertRaises(ValueError):
                other.binding('test-job','b'*64,3)
        finally:
            other.close()


class KeyAndTransportTests(unittest.TestCase):
    def test_dummy_key_permissions_and_symlink_are_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp).resolve(); key=root/'fake-key'
            key.write_text(SECRET); key.chmod(0o600)
            self.assertEqual(api.read_api_key(key),SECRET)
            key.chmod(0o644)
            with self.assertRaises(ValueError): api.read_api_key(key)
            key.chmod(0o600); link=root/'link'; link.symlink_to(key)
            with self.assertRaises(ValueError): api.read_api_key(link)

    def transport(self,status=200,declared=None,chunks=None):
        conn=Mock(); res=Mock(status=status)
        conn.getresponse.return_value=res
        res.isclosed.return_value=False
        res.getheader.side_effect=lambda name: declared if name=='Content-Length' else None
        res.read1.side_effect=chunks or [b'{}',b'']
        return conn

    def test_fixed_https_endpoint_no_redirect_proxy_or_retry(self):
        conn=self.transport(status=302)
        with patch.object(api.http.client,'HTTPSConnection',return_value=conn) as factory:
            with self.assertRaisesRegex(ValueError,'status 302'):
                api.post_response({'model':api.MODEL},2,SECRET)
        factory.assert_called_once_with('api.openai.com',timeout=2)
        self.assertEqual(conn.request.call_args.args[:2],('POST','/v1/responses'))
        conn.close.assert_called_once(); self.assertEqual(conn.request.call_count,1)

    def test_oversized_content_length_and_stream_rejected(self):
        for conn in (self.transport(declared=str(api.MAX_RESPONSE+1)),
                     self.transport(chunks=[b'x'*(api.MAX_RESPONSE+1)])):
            with patch.object(api.http.client,'HTTPSConnection',return_value=conn):
                with self.assertRaises(ValueError): api.post_response({},2,SECRET)
            conn.close.assert_called_once()

    def test_deadline_and_network_errors_are_bounded_sanitized(self):
        conn=self.transport(); conn.request.side_effect=OSError(SECRET)
        with patch.object(api.http.client,'HTTPSConnection',return_value=conn):
            with self.assertRaises(ValueError) as error: api.post_response({},2,SECRET)
        self.assertNotIn(SECRET,str(error.exception)); conn.close.assert_called_once()
        conn=self.transport()
        with patch.object(api.http.client,'HTTPSConnection',return_value=conn), \
             patch.object(api.time,'monotonic',side_effect=[0,3]):
            with self.assertRaisesRegex(ValueError,'deadline'): api.post_response({},2,SECRET)
        conn.close.assert_called_once()

    def test_partial_eof_compression_and_slow_read_are_rejected(self):
        partial=self.transport(declared='20')
        compressed=self.transport()
        compressed.getresponse.return_value.getheader.side_effect=lambda name: 'gzip' if name=='Content-Encoding' else None
        for conn in (partial,compressed):
            with patch.object(api.http.client,'HTTPSConnection',return_value=conn):
                with self.assertRaises(ValueError): api.post_response({},2,SECRET)
            conn.close.assert_called_once()
        slow=self.transport(chunks=[b'x',b''])
        with patch.object(api.http.client,'HTTPSConnection',return_value=slow), \
             patch.object(api.time,'monotonic',side_effect=[0,0,1,3]):
            with self.assertRaisesRegex(ValueError,'deadline'): api.post_response({},2,SECRET)
        self.assertEqual(slow.getresponse.return_value.read1.call_count,1)
        slow.close.assert_called_once()

    def test_close_delimited_response_keeps_socket_deadline(self):
        conn=self.transport(); sock=conn.sock
        response_object=conn.getresponse.return_value
        def close_delimited():
            conn.sock=None
            return response_object
        conn.getresponse.side_effect=close_delimited
        with patch.object(api.http.client,'HTTPSConnection',return_value=conn):
            self.assertEqual(api.post_response({},2,SECRET),b'{}')
        self.assertGreaterEqual(sock.settimeout.call_count,3)

    def test_real_http_response_last_read_closes_socket_safely(self):
        wire=io.BytesIO(b'HTTP/1.1 200 OK\r\nContent-Length: 2\r\nConnection: close\r\n\r\n{}')
        server=Mock(); server.makefile.return_value=wire
        reply=api.http.client.HTTPResponse(server); reply.begin()
        conn=Mock(); conn.getresponse.return_value=reply
        def set_timeout(_):
            if reply.isclosed(): raise OSError('closed response socket')
        conn.sock.settimeout.side_effect=set_timeout
        with patch.object(api.http.client,'HTTPSConnection',return_value=conn):
            self.assertEqual(api.post_response({},2,SECRET),b'{}')

    def test_transport_valid_stream(self):
        conn=self.transport(chunks=[b'{',b'}',b''])
        with patch.object(api.http.client,'HTTPSConnection',return_value=conn):
            self.assertEqual(api.post_response({},2,SECRET),b'{}')
        conn.close.assert_called_once()


class ContinuationTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.folder=Path(self.temp.name).resolve()/'ledger'
        self.ledger=api.OpenAILedger(self.folder); self.addCleanup(self.ledger.close)
        self.ledger.configure(17_000_000,APPROVAL)
        self.ledger.authorize_job('parent','a'*64,3,17_000_000,APPROVAL)
        self.ledger.reserve('parent','a'*64,3,'planner','c'*64,5_618_640)

    def settle(self):
        self.ledger.settle('parent','planner','resp_known',
                          {'input_tokens':100,'output_tokens':20,'total_tokens':120},550)

    def approval(self, parent='parent', successor='successor'):
        return {**APPROVAL,'operation':'retire_and_continue','parent_job_id':parent,
                'successor_job_id':successor,'additional_calls':3}

    def continue_job(self, approval=None, amount=16_855_920):
        self.ledger.continue_job('parent','a'*64,'successor','b'*64,3,amount,
                                 self.approval() if approval is None else approval)

    def test_explicit_retirement_preserves_history_and_stays_in_original_budget(self):
        self.settle()
        old_job=dict(self.ledger.db.execute('SELECT * FROM jobs WHERE id="parent"').fetchone())
        old_call=self.ledger.call('parent','planner')
        old_budget=tuple(self.ledger.db.execute('SELECT * FROM budget').fetchone())
        self.continue_job()
        self.assertEqual(self.ledger._allocated(),16_856_470)
        self.assertEqual(dict(self.ledger.db.execute('SELECT * FROM jobs WHERE id="parent"').fetchone()),old_job)
        self.assertEqual(self.ledger.call('parent','planner'),old_call)
        self.assertEqual(tuple(self.ledger.db.execute('SELECT * FROM budget').fetchone()),old_budget)
        with self.assertRaisesRegex(ValueError,'Retired'):
            self.ledger.reserve('parent','a'*64,3,'author','d'*64,5_618_640)
        for role in ('planner','author','critic'):
            self.ledger.reserve('successor','b'*64,3,role,'e'*64,5_618_640)
        with self.assertRaises(ValueError):
            self.ledger.reserve('successor','b'*64,3,'repair','e'*64,1)

    def test_uncertain_missing_or_misbound_approval_never_releases(self):
        with self.assertRaisesRegex(ValueError,'Unknown'):
            self.continue_job()
        self.settle()
        for approval in ({},APPROVAL,{**self.approval(),'additional_calls':2},
                         {**self.approval(),'successor_job_id':'other'},
                         {**self.approval(),'operation':'retry'}):
            with self.subTest(approval=approval):
                with self.assertRaises(ValueError): self.continue_job(approval)
        self.assertEqual(self.ledger._allocated(),17_000_000)
        self.assertEqual(self.ledger.db.execute('SELECT COUNT(*) FROM retirements').fetchone()[0],0)

    def test_duplicate_and_overspend_are_atomic(self):
        self.settle()
        with self.assertRaisesRegex(ValueError,'unchanged project'):
            self.continue_job(amount=17_000_000)
        self.assertEqual(self.ledger.db.execute('SELECT COUNT(*) FROM retirements').fetchone()[0],0)
        self.assertEqual(self.ledger.db.execute('SELECT COUNT(*) FROM jobs').fetchone()[0],1)
        self.continue_job()
        with self.assertRaisesRegex(ValueError,'already retired'): self.continue_job()
        self.assertEqual(self.ledger.db.execute('SELECT COUNT(*) FROM retirements').fetchone()[0],1)

    def test_inconsistent_settled_usage_is_not_released(self):
        self.ledger.settle('parent','planner','resp_known',
                          {'input_tokens':100,'output_tokens':20,'total_tokens':120},1)
        with self.assertRaisesRegex(ValueError,'validated usage'): self.continue_job()
        self.assertEqual(self.ledger._allocated(),17_000_000)

    def test_concurrent_retirements_cannot_double_allocate_global_budget(self):
        folder=self.folder.parent/'concurrent'
        ledger=api.OpenAILedger(folder)
        ledger.configure(40_000_000,APPROVAL)
        for parent in ('first','second'):
            ledger.authorize_job(parent,'a'*64,3,17_000_000,APPROVAL)
            ledger.reserve(parent,'a'*64,3,'planner','c'*64,5_618_640)
            ledger.settle(parent,'planner','resp_known',
                          {'input_tokens':100,'output_tokens':20,'total_tokens':120},550)
        barrier=threading.Barrier(2); results=[]
        def retire(parent):
            separate=api.OpenAILedger(folder)
            try:
                barrier.wait(timeout=3)
                separate.continue_job(parent,'a'*64,parent+'-next','b'*64,3,22_000_000,
                                      self.approval(parent,parent+'-next'))
                results.append('granted')
            except ValueError:
                results.append('rejected')
            finally:
                separate.close()
        workers=[threading.Thread(target=retire,args=(parent,)) for parent in ('first','second')]
        for worker in workers: worker.start()
        for worker in workers: worker.join(timeout=5)
        try:
            self.assertTrue(all(not worker.is_alive() for worker in workers))
            self.assertCountEqual(results,['granted','rejected'])
            self.assertEqual(ledger._allocated(),39_000_550)
            self.assertEqual(ledger.db.execute('SELECT COUNT(*) FROM retirements').fetchone()[0],1)
        finally:
            ledger.close()


if __name__=='__main__': unittest.main()
