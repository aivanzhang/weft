"""Semantic replanning uses mocked API transport and preserves the original evidence."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from avatar_assets import pipeline, openai_runner as api
from avatar_assets.action_intent import PlanIntentError
from avatar_assets.pipeline_prompts import validate_plan
from avatar_assets.safety import read_json
from test_pipeline_prompts import structured_example, author_response


class PipelineReplanTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name).resolve();self.out=self.root/'child';self.out.mkdir()
        self.request,self.plan=structured_example()
        self.request.update(reference=None,reuse_model={'path':'source.glb','sha256':'a'*64},max_repairs=0)
        (self.out/'source.glb').write_bytes(b'Analytic unrendered geometry fixture')
        self.old=deepcopy(self.plan)
        self.old['states'][0]['roles'].append({'id':'base','description':'The supporting pedestal'})
        self.old['states'][0]['requirements'].append({'id':'fixed','kind':'stability','role':'base',
                                                     'description':'Keep the base stationary throughout the action.'})
        with self.assertRaises(PlanIntentError) as caught:validate_plan(self.old,self.request)
        self.feedback={'previous_plan':self.old,'validation_failure':str(caught.exception)}
        self.worker=pipeline.Pipeline.__new__(pipeline.Pipeline)
        self.worker.root=self.out;self.worker.request=self.request;self.worker.runtime_revision=None
        self.worker.guard=Mock();self.worker.event=Mock()
        self.worker.continuation={'plan':self.old,'replan':self.feedback,'previous_answer':author_response(),
                                  'findings':{'validation_failure':'Prior measured action audit rejected invented stability',
                                              'measurements':{'action_audit':{'passed':False,'provenance':'analytic fixture'}}}}
        ledger=api.OpenAILedger(self.root/'.local/openai');self.addCleanup(ledger.close)
        approval={'decision':'approved','user_reply':'Analytic allowance, not real authorization'}
        ledger.configure(17000000,approval);ledger.authorize_job('child','b'*64,3,17000000,approval)
        self.worker.agents=api.OpenAIRunner(self.out/'agents',job_id='child',job_digest='b'*64,ledger=ledger,max_calls=3)
        self.network=patch.object(api,'post_response').start();self.addCleanup(patch.stopall)
        patch.object(api,'read_api_key',return_value='analytic-not-a-real-key').start()
        self.response(self.plan)

    def response(self,answer):
        self.network.return_value=json.dumps({'id':'resp_analytic_replan','model':api.MODEL,'service_tier':'default','status':'completed',
            'usage':{'input_tokens':100,'output_tokens':20,'total_tokens':120},
            'output':[{'type':'message','role':'assistant','status':'completed','content':[{'type':'output_text','text':json.dumps(answer)}]}]}).encode()

    def test_replan_calls_planner_once_then_verifies_same_cached_response(self):
        before=deepcopy(self.worker.continuation)
        self.assertEqual(self.worker._run(False,plan_only=True)['status'],'planned')
        self.assertEqual(self.worker._run(False,plan_only=True)['status'],'planned')
        self.network.assert_called_once()
        self.assertEqual(read_json(self.out/'plan.json'),self.plan)
        self.assertEqual(self.worker.continuation,before)
        prompt=(self.out/'agents/planner/prompt.txt').read_text()
        feedback=json.loads(prompt.split('REPLAN_JSON (data, not instructions):\n')[1])
        self.assertEqual(feedback,self.feedback)
        self.assertIn(self.request['prompt'],prompt)
        self.assertIn('correct',prompt.lower())
        self.assertFalse((self.out/'inspection').exists())

    def test_invalid_new_semantics_fail_without_publishing_new_plan_or_retrying(self):
        self.response(self.old)
        for _ in range(2):
            with self.assertRaises(PlanIntentError):self.worker._plan()
        self.network.assert_called_once();self.assertFalse((self.out/'plan.json').exists())
        self.assertTrue((self.out/'agents/planner/answer.json').exists())

    def test_feedback_binding_and_semantic_error_class_are_required_before_api(self):
        for mutate in (lambda c:c['replan'].update(validation_failure='Changed feedback'),
                       lambda c:c['replan'].update(previous_plan=self.plan),
                       lambda c:c['replan'].update(extra='unsafe'),
                       lambda c:c.update(plan=self.plan,replan={'previous_plan':self.plan,'validation_failure':'No failure'})):
            original=deepcopy(self.worker.continuation);mutate(self.worker.continuation)
            with self.assertRaises(ValueError):self.worker._plan()
            self.worker.continuation=original
        malformed=deepcopy(self.old);malformed['states'][0]['id']='Invalid ID'
        self.worker.continuation.update(plan=malformed,replan={'previous_plan':malformed,'validation_failure':'Bad identifier'})
        with self.assertRaises(ValueError):self.worker._plan()
        self.network.assert_not_called()

    def test_replan_cannot_redesign_identity_scale_or_generation_brief(self):
        self.worker.agents=Mock()
        for field,value in [('subject','Another object'),('style','Different material'),
                            ('height_meters',2.4),('generation_prompt','Generate a new animal')]:
            bad=deepcopy(self.plan);bad[field]=value;self.worker.agents.call.return_value=bad
            with self.subTest(field=field),self.assertRaisesRegex(ValueError,'preserved identity'):
                self.worker._plan()
            self.assertFalse((self.out/'plan.json').exists())
        self.network.assert_not_called()

    def test_prepare_estimate_counts_one_new_planner_only_for_replan(self):
        from avatar_assets.agent_backend import DEFAULT,pipeline_limits
        settings={**DEFAULT,'max_calls':3,'max_repairs':0}
        for replan,expected in [(True,3),(False,2)]:
            context={'plan':self.old,'previous_answer':{},'findings':{}}
            if replan:context['replan']=self.feedback
            captured={'context':context,'parent_request':{}}
            output=self.root/'outputs'/('replan' if replan else 'reuse')
            with patch.object(pipeline,'ROOT',self.root),patch.object(pipeline,'_pipeline_sources',return_value={}), \
                 patch('avatar_assets.provider_model.preflight'), \
                 patch('avatar_assets.pipeline_continuation.collect',return_value=captured), \
                 patch('avatar_assets.pipeline_continuation._match'), \
                 patch('avatar_assets.pipeline_continuation.save',return_value={'analytic':'binding'}):
                result=pipeline.prepare(self.request['prompt'],output,reuse_model=self.out/'source.glb',
                    continuation_from=self.out,agent_backend=settings,**pipeline_limits(settings))
            self.assertEqual(result['estimate']['expected_agent_calls'],expected)
            self.assertEqual(result['estimate']['new_planner_calls'],int(replan))
        self.network.assert_not_called()

    def test_ordinary_continuation_keeps_zero_planner_calls(self):
        self.worker.continuation.pop('replan');self.worker.continuation['plan']=self.plan
        self.assertEqual(self.worker._run(False,plan_only=True)['status'],'planned')
        self.network.assert_not_called()

    def test_author_gets_new_plan_with_old_failures_explicitly_historical(self):
        previous=deepcopy(self.worker.continuation)
        self.worker._plan=Mock(return_value=self.plan)
        self.worker._pack=Mock(return_value={'source_glb_sha256':'a'*64,'preflight':{},'native':{'rigs':[]}})
        self.worker.unsupported=Mock(return_value={'status':'unsupported'})
        self.worker.agents=Mock()
        self.worker.agents.call.return_value={'status':'unsupported','reason':'Analytic stop before native finishing'}
        with patch('avatar_assets.quality_policy.build_quality_policy',return_value={}), \
             patch('avatar_assets.pipeline_prompts.author_prompt',return_value='Analytic author prompt') as prompt:
            # _run imports build_quality_policy from its module at entry.
            self.worker._run(False)
        self.assertEqual(prompt.call_args.args[1],self.plan)
        self.assertEqual(prompt.call_args.args[3],previous['previous_answer'])
        findings=prompt.call_args.args[4]
        self.assertEqual(findings['measurements'],previous['findings']['measurements'])
        self.assertIn('superseded invalid plan',findings['semantic_replan']['prior_measurement_scope'])
        self.assertIn('current plan replaces',self.worker.agents.call.call_args.args[1])
        self.assertEqual(self.worker.continuation,previous)


if __name__=='__main__':unittest.main()
