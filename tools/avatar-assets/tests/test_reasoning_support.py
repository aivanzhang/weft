"""Proposed reasoning support: mocked transport, temporary ledgers, no real keys or calls."""
import copy
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import test_openai_runner as runner_fixtures
from test_openai_runner import SCHEMA, response
from avatar_assets import openai_runner as api
from avatar_assets.agent_backend import DEFAULT, validate_settings
from avatar_assets.pipeline_continuation import _role


def reasoned(**changes):
    data=json.loads(response())
    data['output'].insert(0,{'id':'rs_analytic','type':'reasoning','summary':[]})
    data['usage']['output_tokens_details']={'reasoning_tokens':8}
    data.update(changes)
    return data


class ReasoningSupportTests(unittest.TestCase):
    setUp=runner_fixtures.RunnerTests.setUp
    runner=runner_fixtures.RunnerTests.runner

    def test_optional_settings_preserve_legacy_bytes_and_reject_unbounded_values(self):
        original=dict(DEFAULT)
        self.assertEqual(validate_settings(original),original)
        self.assertNotIn('reasoning_effort',validate_settings(original))
        for effort in ('none','low','medium','high'):
            value={**DEFAULT,'reasoning_effort':effort}
            self.assertEqual(validate_settings(value),value)
        for effort in ('xhigh','minimal','max',None,True,{},'MEDIUM'):
            with self.subTest(effort=effort),self.assertRaises(ValueError):
                validate_settings({**DEFAULT,'reasoning_effort':effort})
        self.assertNotEqual(api.digest(DEFAULT),api.digest({**DEFAULT,'reasoning_effort':'medium'}))

    def test_medium_payload_fingerprint_usage_and_reservation_are_bound(self):
        self.transport.return_value=json.dumps(reasoned()).encode()
        runner=self.runner(reasoning_effort='medium')
        self.assertEqual(runner.call('author-0','analytic rig repair',SCHEMA),{'value':3})
        payload=self.transport.call_args.args[0]
        self.assertEqual(payload['reasoning'],{'effort':'medium'})
        self.assertEqual(payload['max_output_tokens'],16384)
        self.assertEqual(runner.reservation,5_618_640)
        receipt=json.loads((self.root/'agents/author-0/receipt.json').read_text())
        self.assertEqual(receipt['usage']['output_tokens_details']['reasoning_tokens'],8)
        self.assertEqual(receipt['actual_microdollars_upper_estimate'],550)
        self.assertEqual(api.call_exposure(self.ledger.call('test-job','author-0')),550)
        saved=json.loads((self.root/'agents/author-0/request.json').read_text())
        self.assertEqual(saved['settings_sha256'],api.digest(runner.settings))
        with self.assertRaisesRegex(ValueError,'settings cannot change'):self.runner(reasoning_effort='none')
        self.assertEqual(self.transport.call_count,1)

    def test_shared_continuation_parser_accepts_exact_reasoned_answer(self):
        self.transport.return_value=json.dumps(reasoned()).encode()
        runner=self.runner(reasoning_effort='medium');runner.call('author-0','analytic',SCHEMA)
        self.assertEqual(_role(self.root,'author-0',self.ledger.call('test-job','author-0'),runner.settings),{'value':3})

    def test_none_default_settings_keep_existing_shape(self):
        runner=self.runner();self.assertEqual(runner.settings['reasoning'],{'effort':'none'})
        runner.call('planner','analytic',SCHEMA)
        self.assertEqual(self.transport.call_args.args[0]['reasoning'],{'effort':'none'})
        self.assertEqual(self.runner().call('planner','analytic',SCHEMA),{'value':3})
        self.assertEqual(self.transport.call_count,1)

    def test_parser_accepts_bounded_summary_and_rejects_tools_extra_answers_and_partial(self):
        good=reasoned();good['output'][0]['summary']=[{'type':'summary_text','text':'Brief metadata'}]
        good['output'][-1]['phase']='final_answer'
        self.assertEqual(api.response_answer_text(good,reasoning_effort='medium'),'{'+'"value":3}')
        bad=[]
        item=copy.deepcopy(good);item['output'].append(item['output'][-1]);bad.append(item)
        item=copy.deepcopy(good);item['output'].insert(1,{'type':'function_call','name':'forbidden'});bad.append(item)
        item=copy.deepcopy(good);item['output'][-1]['phase']='commentary';bad.append(item)
        item=copy.deepcopy(good);item['status']='incomplete';bad.append(item)
        item=copy.deepcopy(good);item['output'][0]['tool_calls']=[];bad.append(item)
        item=copy.deepcopy(good);item['output'][0]['summary'][0]['text']='x'*(256*1024+1);bad.append(item)
        item=copy.deepcopy(good);item['output']=[item['output'][0]]*16+[item['output'][-1]];bad.append(item)
        for index,value in enumerate(bad):
            with self.subTest(index=index),self.assertRaises(ValueError):api.response_answer_text(value,reasoning_effort='medium')
        with self.assertRaises(ValueError):api.response_answer_text(good)

    def test_reasoning_tokens_are_a_subset_not_an_extra_charge(self):
        for count in (-1,21,True,'8'):
            with self.subTest(count=count),self.assertRaises(ValueError):
                api.validate_reasoning_usage({'output_tokens_details':{'reasoning_tokens':count}},20)
        api.validate_reasoning_usage({},20)
        api.validate_reasoning_usage({'output_tokens_details':{'reasoning_tokens':20}},20)

    def test_old_saved_none_receipt_still_authenticates_without_rewriting(self):
        candidates=[parent/'outputs/studio-b082dc54dbb54f9d9030f75b' for parent in Path(__file__).resolve().parents[:5]]
        actual=next((path for path in candidates if path.exists()),None)
        if actual is None:self.skipTest('Local immutable b082 fixture unavailable')
        folder=actual/'agents/planner';receipt=json.loads((folder/'receipt.json').read_text())
        settings=json.loads((actual/'agents/runner.json').read_text());request=json.loads((folder/'request.json').read_text())
        row={'request_digest':api.digest(request),'response_id':receipt['response_id'],
             'actual':receipt['actual_microdollars_upper_estimate'],'reserved':receipt['reserved_microdollars'],
             'usage':json.dumps(receipt['usage'])}
        self.assertEqual(_role(actual,'planner',row,settings),json.loads((folder/'answer.json').read_text()))
        self.transport.assert_not_called();self.key.assert_not_called()


class ReasoningStudioTests(unittest.TestCase):
    def test_saved_medium_selection_reaches_runner_and_cannot_change_after_prepare(self):
        from test_openai_studio_integration import OpenAIStudioIntegrationTests
        from avatar_assets import pipeline
        fixture=OpenAIStudioIntegrationTests()
        fixture.setUp();self.addCleanup(fixture.doCleanups)
        fixture.settings={**fixture.settings,'reasoning_effort':'medium'}
        fixture.config();fixture.key();fixture.prepare();ledger=fixture.grant()
        before=(fixture.out/'request.json').read_bytes()
        with patch('avatar_assets.openai_runner.OpenAIRunner') as runner, \
             patch('avatar_assets.openai_runner.OpenAILedger',return_value=ledger):
            pipeline.Pipeline(fixture.out)
        self.assertEqual(runner.call_args.kwargs['reasoning_effort'],'medium')
        self.assertEqual(json.loads(before)['agent_backend']['reasoning_effort'],'medium')
        fixture.config({**fixture.settings,'reasoning_effort':'low'})
        with self.assertRaises(ValueError):pipeline.Pipeline(fixture.out)
        self.assertEqual((fixture.out/'request.json').read_bytes(),before)


if __name__=='__main__':unittest.main()
