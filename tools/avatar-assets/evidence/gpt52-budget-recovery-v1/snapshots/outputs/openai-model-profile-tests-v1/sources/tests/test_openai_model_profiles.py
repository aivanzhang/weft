"""Analytic mixed-model accounting; no real keys, provider requests or ledger writes."""
from copy import deepcopy
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from avatar_assets import openai_runner as api
from avatar_assets import agent_backend as backend
from avatar_assets.openai_models import GPT52

APPROVAL = {'decision': 'approved', 'user_reply': 'Synthetic local test only'}
SCHEMA = {'type':'object', 'properties':{'value':{'type':'integer'}},
          'required':['value'], 'additionalProperties':False}


def response(model=GPT52, inp=100, out=20):
    return json.dumps({'id':'resp_mock', 'model':model, 'status':'completed', 'service_tier':'default',
        'usage':{'input_tokens':inp, 'output_tokens':out, 'total_tokens':inp+out,
                 'output_tokens_details':{'reasoning_tokens':min(10,out)}},
        'output':[{'type':'message','role':'assistant','status':'completed',
                   'content':[{'type':'output_text','text':'{"value":3}'}]}]}).encode()


class ModelProfiles(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()/'repo/tools/avatar-assets'
        self.root.mkdir(parents=True)
        self.ledger = api.OpenAILedger(self.root/'.local/openai'); self.addCleanup(self.ledger.close)
        self.ledger.configure(15_000_000, APPROVAL)
        self.ledger.authorize_job('old', 'a'*64, 3, 7_000_000, APPROVAL)
        self.ledger.authorize_job('new', 'b'*64, 3, 8_000_000, APPROVAL)

    def runner(self):
        return api.OpenAIRunner(self.root/'agents', job_id='new',job_digest='b'*64,
                               ledger=self.ledger,max_calls=3,model=GPT52,reasoning_effort='medium')

    def reserve_new(self, role='author'):
        self.ledger.reserve('new','b'*64,3,role,'c'*64,929376,model=GPT52,max_output_tokens=16384)

    def test_exact_full_context_bounds_and_legacy_threshold_are_preserved(self):
        self.assertEqual(api.reserve_microdollars(),5618640)
        self.assertEqual(api.reserve_microdollars(model=GPT52),929376)
        self.assertEqual(api.usage_cost_microdollars(GPT52,1,0),2)
        self.assertEqual(api.usage_cost_microdollars(GPT52,400000,16384),929376)
        self.assertEqual(api.usage_cost_microdollars(api.MODEL,272000,1),680015)
        self.assertEqual(api.usage_cost_microdollars(api.MODEL,272001,1),1360028)
        profile=api.price_profile(GPT52);profile['context_tokens']=1
        self.assertEqual(api.price_profile(GPT52)['context_tokens'],400000)
        for invalid in ('gpt-5.2','unknown',None):
            with self.assertRaises(ValueError):api.price_profile(invalid)
        with self.assertRaises(ValueError):api.call_model({'model_profile':None})

    def test_payload_cache_receipt_and_binding_use_selected_model(self):
        with patch.object(api,'read_api_key',return_value='synthetic-only-key'), \
             patch.object(api,'post_response',return_value=response()) as transport:
            runner=self.runner();self.assertEqual(runner.call('author','robot',SCHEMA),{'value':3})
            self.assertEqual(runner.call('author','robot',SCHEMA),{'value':3})
        self.assertEqual(transport.call_count,1)
        self.assertEqual(transport.call_args.args[0]['model'],GPT52)
        row=self.ledger.call('new','author')
        self.assertEqual(api.call_model(row),GPT52)
        self.assertEqual(row['actual'],455)
        self.assertEqual(row['reserved'],929376)
        self.assertEqual(json.loads((self.root/'agents/author/receipt.json').read_text())['model'],GPT52)
        self.assertEqual(runner.settings['pricing'],api.price_profile(GPT52))
        self.assertIn('model_profile_implementation_sha256',runner.settings)

    def test_wrong_response_model_retains_new_full_hold_without_retry(self):
        with patch.object(api,'read_api_key',return_value='synthetic-only-key'), \
             patch.object(api,'post_response',return_value=response(api.MODEL)) as transport:
            runner=self.runner()
            with self.assertRaisesRegex(ValueError,'unexpected API response'):runner.call('author','robot',SCHEMA)
            with self.assertRaises(ValueError):runner.call('author','robot',SCHEMA)
        self.assertEqual(transport.call_count,1)
        self.assertEqual(api.call_exposure(self.ledger.call('new','author')),929376)
        self.assertEqual(self.ledger.call('new','author')['state'],'reserved_uncertain')

    def test_mixed_legacy_and_bound_calls_keep_old_unknown_hold(self):
        self.ledger.reserve('old','a'*64,3,'author','d'*64,5618640)
        old=deepcopy(self.ledger.call('old','author'))
        self.reserve_new()
        usage={'input_tokens':100,'output_tokens':20,'total_tokens':120,
               'output_tokens_details':{'reasoning_tokens':10}}
        self.ledger.settle('new','author','resp_mock',usage,455)
        self.assertEqual(api.accounting_snapshot(self.ledger.db)['project_exposure_microdollars'],5619095)
        self.assertEqual(self.ledger.call('old','author'),old)
        self.assertNotIn('model_profile',old)

    def test_spoofed_price_model_and_request_bindings_fail_closed(self):
        self.reserve_new();saved=self.ledger.db.execute('SELECT binding FROM call_profiles').fetchone()[0]
        for field,value in [('model',api.MODEL),('request_digest','d'*64),('pricing',api.price_profile(api.MODEL))]:
            binding=json.loads(saved);binding[field]=value
            self.ledger.db.execute('UPDATE call_profiles SET binding=?',(api._canonical(binding).decode(),))
            with self.subTest(field=field),self.assertRaises(ValueError):api.accounting_snapshot(self.ledger.db)
        self.ledger.db.execute('UPDATE call_profiles SET binding=?',(saved,));self.ledger.db.commit()
        with self.assertRaises(ValueError):
            self.ledger.reserve('new','b'*64,3,'other','c'*64,100,model=GPT52)
        self.assertIsNone(self.ledger.call('new','other'))

    def test_settlement_rejects_wrong_profile_cost_context_and_output(self):
        self.reserve_new()
        for inp,out,cost in [(100,20,550),(400001,0,700002),(100,16385,229565)]:
            usage={'input_tokens':inp,'output_tokens':out,'total_tokens':inp+out}
            with self.subTest(inp=inp,out=out),self.assertRaises(ValueError):
                self.ledger.settle('new','author','resp_mock',usage,cost)
        self.assertEqual(self.ledger.call('new','author')['state'],'reserved_uncertain')

    def test_profile_oversize_and_orphan_fail_before_usage(self):
        self.reserve_new();saved=self.ledger.db.execute('SELECT binding FROM call_profiles').fetchone()[0]
        self.ledger.db.execute('UPDATE call_profiles SET binding=?',('x'*4097,))
        with self.assertRaisesRegex(ValueError,'oversized'):api.accounting_snapshot(self.ledger.db)
        self.ledger.db.execute('UPDATE call_profiles SET binding=?,role=?',(saved,'orphan'))
        with self.assertRaisesRegex(ValueError,'no original call'):api.accounting_snapshot(self.ledger.db)

    def test_readonly_readiness_uses_new_reserve_and_preserves_database(self):
        settings={**backend.DEFAULT,'model':GPT52,'max_calls':3,'max_repairs':0,'reasoning_effort':'medium'}
        cfg=self.root/'.local/openai-backend.json';cfg.write_text(json.dumps(settings));os.chmod(cfg,0o600)
        key=self.root.parents[1]/'.local/credentials/openai-api-key';key.parent.mkdir(parents=True)
        key.write_text('synthetic-test-key');os.chmod(key,0o600)
        self.ledger.reserve('old','a'*64,3,'author','d'*64,5618640)
        path=self.root/'.local/openai/ledger.sqlite3';before=path.read_bytes()
        with patch.object(api,'read_api_key',side_effect=AssertionError('No secret read')):
            ready=backend.readiness(self.root,job_id='new',job_digest='b'*64)
        self.assertTrue(ready['can_run'],ready)
        self.assertEqual(ready['reservation_microdollars_per_call'],929376)
        self.assertEqual(ready['project_effective_exposure_microdollars'],5618640)
        self.assertEqual(path.read_bytes(),before)
        self.assertEqual(backend.DEFAULT['model'],api.MODEL)

    def test_selected_raw_rows_attach_profiles_without_reinterpreting_legacy(self):
        self.reserve_new();self.ledger.reserve('old','a'*64,3,'author','d'*64,5618640)
        rows=self.ledger.db.execute('SELECT * FROM calls ORDER BY job').fetchall()
        attached=api.attach_call_profiles(self.ledger.db,rows)
        self.assertEqual([api.call_model(row) for row in attached],[GPT52,api.MODEL])
        self.assertEqual([api.call_exposure(row) for row in attached],[929376,5618640])


if __name__=='__main__':unittest.main()
