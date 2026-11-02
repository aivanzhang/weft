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
        self.settle()
        self.ledger.db.execute('UPDATE calls SET actual=1')
        self.ledger.db.commit()
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


class ExposureTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.folder=Path(self.temp.name).resolve()/'ledger'
        self.ledger=api.OpenAILedger(self.folder);self.addCleanup(self.ledger.close)
        self.ledger.configure(17_000_000,APPROVAL)
        self.ledger.authorize_job('job','a'*64,5,17_000_000,APPROVAL)
        self.hold=api.reserve_microdollars()

    def reserve(self,role):
        self.ledger.reserve('job','a'*64,5,role,'c'*64,self.hold)

    def settle(self,role):
        self.ledger.settle('job',role,'resp_'+role,
                          {'input_tokens':40000,'output_tokens':0,'total_tokens':40000},100000)

    def test_five_sequential_known_charges_fit_same_cap_without_erasing_holds_or_call_counts(self):
        budget=tuple(self.ledger.db.execute('SELECT * FROM budget').fetchone())
        grant=tuple(self.ledger.db.execute('SELECT * FROM jobs').fetchone())
        for number,role in enumerate(('planner','author-0','reviewer-0','author-1','reviewer-1'),1):
            self.reserve(role);self.settle(role)
            totals=api.accounting_snapshot(self.ledger.db)
            self.assertEqual(totals['project_exposure_microdollars'],number*100000)
            self.assertEqual(totals['jobs']['job']['reserved_microdollars'],number*self.hold)
            self.assertEqual(totals['jobs']['job']['calls'],number)
            self.assertEqual(self.ledger.call('job',role)['reserved'],self.hold)
        with self.assertRaisesRegex(ValueError,'exhausted'):self.reserve('unapproved-sixth-call')
        self.assertEqual(tuple(self.ledger.db.execute('SELECT * FROM budget').fetchone()),budget)
        self.assertEqual(tuple(self.ledger.db.execute('SELECT * FROM jobs').fetchone()),grant)

    def test_unknown_failed_or_unsupported_states_never_free_the_original_hold(self):
        for role in ('planner','author-0','reviewer-0'):self.reserve(role)
        with self.assertRaisesRegex(ValueError,'exhausted'):self.reserve('author-1')
        self.ledger.db.execute("UPDATE calls SET actual=0,usage='{}' WHERE role='planner'")
        self.ledger.db.commit()
        with self.assertRaisesRegex(ValueError,'exhausted'):self.reserve('author-1')
        self.ledger.db.execute("UPDATE calls SET state='failed' WHERE role='planner'")
        self.ledger.db.commit()
        with self.assertRaisesRegex(ValueError,'accounting state'):self.reserve('author-1')
        self.assertEqual(self.ledger.call('job','planner')['reserved'],self.hold)

    def test_even_settled_usage_must_leave_enough_headroom_for_a_full_next_hold(self):
        for role in ('planner','author-0','reviewer-0'):self.reserve(role)
        self.ledger.settle('job','planner','resp_expensive',
                          {'input_tokens':1050000,'output_tokens':10000,'total_tokens':1060000},5475000)
        with self.assertRaisesRegex(ValueError,'exhausted'):self.reserve('author-1')
        self.assertIsNone(self.ledger.call('job','author-1'))

    def test_invalid_settlement_is_atomic_and_corrupt_settled_rows_block_reservation(self):
        self.reserve('planner')
        original=self.ledger.call('job','planner')
        with self.assertRaisesRegex(ValueError,'validated usage'):
            self.ledger.settle('job','planner','resp_bad',
                              {'input_tokens':40000,'output_tokens':0,'total_tokens':40000},1)
        self.assertEqual(self.ledger.call('job','planner'),original)
        self.settle('planner')
        for field,value in [('actual',1),('actual',self.hold+1),('usage','{}'),
                            ('usage','{"input_tokens":true,"output_tokens":0,"total_tokens":1}'),
                            ('usage','{"input_tokens":NaN,"output_tokens":0,"total_tokens":0}'),
                            ('response_id','unvalidated-id')]:
            with self.subTest(field=field,value=value):
                before=self.ledger.call('job','planner')[field]
                self.ledger.db.execute('UPDATE calls SET '+field+'=? WHERE role=?',(value,'planner'))
                self.ledger.db.commit()
                with self.assertRaises(ValueError):self.reserve('author-0')
                self.assertIsNone(self.ledger.call('job','author-0'))
                self.ledger.db.execute('UPDATE calls SET '+field+'=? WHERE role=?',(before,'planner'))
                self.ledger.db.commit()

    def test_concurrent_next_reservations_cannot_spend_the_same_settled_headroom(self):
        self.reserve('planner');self.settle('planner')
        self.reserve('author-0');self.reserve('reviewer-0')
        barrier=threading.Barrier(2);results=[]
        def submit(role):
            ledger=api.OpenAILedger(self.folder)
            try:
                barrier.wait(timeout=3)
                ledger.reserve('job','a'*64,5,role,'d'*64,self.hold)
                results.append('reserved')
            except ValueError:results.append('rejected')
            finally:ledger.close()
        workers=[threading.Thread(target=submit,args=(role,)) for role in ('author-1','reviewer-1')]
        for worker in workers:worker.start()
        for worker in workers:worker.join(timeout=5)
        self.assertTrue(all(not worker.is_alive() for worker in workers))
        self.assertCountEqual(results,['reserved','rejected'])
        totals=api.accounting_snapshot(self.ledger.db)
        self.assertEqual(totals['jobs']['job']['calls'],4)
        self.assertEqual(totals['project_exposure_microdollars'],100000+3*self.hold)
        self.assertLessEqual(totals['project_exposure_microdollars'],17000000)

    def test_bounded_ten_thousand_row_snapshot_refuses_oversize_before_decoding(self):
        import sqlite3,time
        db=sqlite3.connect(':memory:');self.addCleanup(db.close)
        db.execute('CREATE TABLE calls(job,reserved,state,response_id,actual,usage)')
        usage=json.dumps({'input_tokens':40,'output_tokens':0,'total_tokens':40})
        db.executemany('INSERT INTO calls VALUES(?,?,?,?,?,?)',
            [(f'fixture-{i}',self.hold,'settled','resp_fixture',100,usage) for i in range(10000)])
        started=time.monotonic();totals=api.accounting_snapshot(db);elapsed=time.monotonic()-started
        self.assertEqual(totals['project_exposure_microdollars'],1000000)
        self.assertEqual(len(totals['jobs']),10000)
        print(json.dumps({'analytic_accounting_rows':10000,'elapsed_seconds':elapsed}))
        db.execute('INSERT INTO calls VALUES(?,?,?,?,?,?)',('extra',1,'settled','invalid',0,'invalid'))
        with patch.object(api,'call_exposure',side_effect=AssertionError('No oversized decoding')):
            with self.assertRaisesRegex(ValueError,'exceeds10000'):api.accounting_snapshot(db)


class BudgetReductionTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.folder=Path(self.temp.name).resolve()/'ledger'
        self.ledger=api.OpenAILedger(self.folder);self.addCleanup(self.ledger.close)
        self.ledger.configure(17_000_000,APPROVAL)

    def approval(self, amount):
        return {**APPROVAL,'operation':'tighten_project_ceiling','max_microdollars':amount}

    def grant(self):
        self.ledger.authorize_job('parent','a'*64,5,17_000_000,APPROVAL)

    def reserve(self, role, amount=None):
        self.ledger.reserve('parent','a'*64,5,role,'c'*64,amount or api.reserve_microdollars())

    def test_lower_only_history_keeps_original_approvals_calls_and_grants_exact(self):
        self.grant();self.reserve('planner')
        self.ledger.settle('parent','planner','resp_known',
                          {'input_tokens':40000,'output_tokens':0,'total_tokens':40000},100000)
        old={table:[tuple(r) for r in self.ledger.db.execute('SELECT * FROM '+table)]
             for table in ('budget','jobs','calls','retirements')}
        result=self.ledger.tighten_budget(15_000_000,self.approval(15_000_000))
        self.assertEqual(result['original_ceiling_microdollars'],17_000_000)
        self.assertEqual(result['effective_ceiling_microdollars'],15_000_000)
        self.assertEqual(result['project_exposure_microdollars'],100000)
        first=tuple(self.ledger.db.execute('SELECT * FROM budget_reductions').fetchone())
        self.ledger.tighten_budget(14_000_000,self.approval(14_000_000))
        self.assertEqual(tuple(self.ledger.db.execute('SELECT * FROM budget_reductions WHERE id=1').fetchone()),first)
        for table,rows in old.items():
            self.assertEqual([tuple(r) for r in self.ledger.db.execute('SELECT * FROM '+table)],rows)
        self.assertEqual(api.budget_snapshot(self.ledger.db)['reductions'],2)

    def test_increase_equal_invalid_or_unbound_reduction_cannot_change_limit(self):
        self.ledger.tighten_budget(15_000_000,self.approval(15_000_000))
        for amount in (15_000_000,16_000_000,0,True,14_000_000.0):
            with self.subTest(amount=amount),self.assertRaises(ValueError):
                self.ledger.tighten_budget(amount,self.approval(amount))
        for approval in (APPROVAL,self.approval(13_000_000),{**self.approval(14_000_000),'operation':'increase'}):
            with self.assertRaises(ValueError):self.ledger.tighten_budget(14_000_000,approval)
        with self.assertRaisesRegex(ValueError,'immutable'):self.ledger.configure(15_000_000,APPROVAL)
        self.ledger.configure(17_000_000,APPROVAL)
        self.assertEqual(api.budget_snapshot(self.ledger.db)['effective_ceiling_microdollars'],15_000_000)
        self.assertEqual(self.ledger.db.execute('SELECT COUNT(*) FROM budget_reductions').fetchone()[0],1)

    def test_full_unknown_holds_above_lower_cap_block_reduction_without_releasing_them(self):
        self.grant()
        for role in ('planner','author-0','reviewer-0'):self.reserve(role)
        before=[tuple(r) for r in self.ledger.db.execute('SELECT * FROM calls')]
        with self.assertRaisesRegex(ValueError,'uncertain holds exceed'):
            self.ledger.tighten_budget(15_000_000,self.approval(15_000_000))
        self.assertEqual([tuple(r) for r in self.ledger.db.execute('SELECT * FROM calls')],before)
        self.assertEqual(api.budget_snapshot(self.ledger.db)['reductions'],0)

    def test_validated_settled_cost_above_new_cap_blocks_reduction(self):
        self.grant()
        for role in ('planner','author-0','reviewer-0'):
            self.reserve(role)
            self.ledger.settle('parent',role,'resp_'+role,
                {'input_tokens':1050000,'output_tokens':0,'total_tokens':1050000},5250000)
        with self.assertRaisesRegex(ValueError,'Known costs'):
            self.ledger.tighten_budget(15_000_000,self.approval(15_000_000))
        self.assertEqual(api.budget_snapshot(self.ledger.db)['effective_ceiling_microdollars'],17_000_000)

    def test_old_unretired_grant_cannot_reserve_above_lower_global_ceiling(self):
        self.grant();self.ledger.tighten_budget(15_000_000,self.approval(15_000_000))
        self.reserve('planner');self.reserve('author-0')
        with self.assertRaisesRegex(ValueError,'project spending allowance exhausted'):self.reserve('reviewer-0')
        self.assertEqual(self.ledger.binding('parent','a'*64,5)['ceiling'],17_000_000)
        self.assertIsNone(self.ledger.call('parent','reviewer-0'))
        self.assertEqual(api.accounting_snapshot(self.ledger.db)['project_exposure_microdollars'],2*api.reserve_microdollars())

    def test_new_grants_and_successors_use_lower_ceiling(self):
        self.ledger.tighten_budget(15_000_000,self.approval(15_000_000))
        with self.assertRaisesRegex(ValueError,'already allocated'):
            self.ledger.authorize_job('parent','a'*64,5,16_000_000,APPROVAL)
        self.ledger.authorize_job('parent','a'*64,5,15_000_000,APPROVAL)
        self.reserve('planner')
        self.ledger.settle('parent','planner','resp_known',
                          {'input_tokens':40000,'output_tokens':0,'total_tokens':40000},100000)
        approval={**APPROVAL,'operation':'retire_and_continue','parent_job_id':'parent',
                  'successor_job_id':'next','additional_calls':5}
        with self.assertRaisesRegex(ValueError,'project allowance'):
            self.ledger.continue_job('parent','a'*64,'next','b'*64,5,15_000_000,approval)
        self.ledger.continue_job('parent','a'*64,'next','b'*64,5,14_900_000,approval)
        self.assertEqual(self.ledger._allocated(),15_000_000)
        self.assertEqual(api.accounting_snapshot(self.ledger.db)['project_exposure_microdollars'],100000)
        with self.assertRaisesRegex(ValueError,'Retired'):self.reserve('author-0')

    def test_old_oversized_parent_can_retire_into_fitting_successor_without_rewriting_grant(self):
        self.grant();self.reserve('planner')
        self.ledger.settle('parent','planner','resp_known',
                          {'input_tokens':40000,'output_tokens':0,'total_tokens':40000},100000)
        self.ledger.tighten_budget(15_000_000,self.approval(15_000_000))
        approval={**APPROVAL,'operation':'retire_and_continue','parent_job_id':'parent',
                  'successor_job_id':'next','additional_calls':5}
        self.ledger.continue_job('parent','a'*64,'next','b'*64,5,14_900_000,approval)
        self.assertEqual(self.ledger.binding('parent','a'*64,5)['ceiling'],17_000_000)
        self.assertEqual(self.ledger._allocated(),15_000_000)

    def test_corrupt_reduction_chain_stops_reservations(self):
        self.grant();self.ledger.tighten_budget(15_000_000,self.approval(15_000_000))
        self.ledger.db.execute('UPDATE budget_reductions SET previous_ceiling=18000000')
        self.ledger.db.commit()
        with self.assertRaisesRegex(ValueError,'reduction chain'):self.reserve('planner')
        self.assertIsNone(self.ledger.call('parent','planner'))

    def test_concurrent_tightening_and_reservation_share_one_atomic_exposure_limit(self):
        self.grant();self.reserve('planner',8_000_000)
        barrier=threading.Barrier(2);results=[]
        def act(operation):
            ledger=api.OpenAILedger(self.folder)
            try:
                barrier.wait(timeout=3)
                if operation=='lower':ledger.tighten_budget(15_000_000,self.approval(15_000_000))
                else:ledger.reserve('parent','a'*64,5,'author-0','d'*64,8_000_000)
                results.append((operation,'passed'))
            except ValueError:results.append((operation,'rejected'))
            finally:ledger.close()
        workers=[threading.Thread(target=act,args=(operation,)) for operation in ('lower','reserve')]
        for worker in workers:worker.start()
        for worker in workers:worker.join(timeout=5)
        self.assertTrue(all(not worker.is_alive() for worker in workers))
        self.assertEqual(sum(status=='passed' for _,status in results),1)
        self.assertLessEqual(api.accounting_snapshot(self.ledger.db)['project_exposure_microdollars'],
                             api.budget_snapshot(self.ledger.db)['effective_ceiling_microdollars'])


if __name__=='__main__': unittest.main()
