"""Analytic runtime/custody fixtures; no provider, native or semantic-quality approval."""

from contextlib import ExitStack
from pathlib import Path
import sqlite3
import unittest
from unittest.mock import Mock,patch

from avatar_assets import pipeline,pipeline_recovery as recovery,studio_inspection
from avatar_assets.safety import artifacts,read_json,sha256
from tests import test_studio_api_release as api_fixture
from tests.test_studio_api_release import ABSENT,write


class PipelineRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.fixture=api_fixture.StudioApiReleaseTests('test_settled_semantically_invalid_answer_releases_only_as_failed_without_calls_or_reset')
        self.fixture.setUp();self.addCleanup(self.fixture.doCleanups)
        f=self.fixture;self.job=f.output;self.root=f.root
        for name,content in [('action_intent.py',b'ANALYTIC OLD VALIDATOR'),('pipeline.py',b'ANALYTIC OLD RUNTIME')]:
            (self.job/'sources'/name).write_bytes(content)
        receipt=read_json(self.job/'input-receipt.json');receipt['sources']=artifacts(self.job/'sources')
        write(self.job/'input-receipt.json',receipt)
        self.live={}
        for name in ('openai_runner.py','action_intent.py','pipeline.py','pipeline_recovery.py'):
            path=self.root/'runtime'/name;path.parent.mkdir(exist_ok=True)
            path.write_bytes((self.job/'sources'/name).read_bytes() if name=='openai_runner.py' else ('ANALYTIC NEW '+name).encode())
            self.live[name]=path
        self.stack=ExitStack();self.addCleanup(self.stack.close)
        self.stack.enter_context(patch.object(pipeline,'ROOT',self.root))
        self.stack.enter_context(patch.object(pipeline,'_pipeline_sources',return_value=self.live))
        self.plan={'status':'ready','subject':'Analytic fixture','states':[{'id':'dance','meaning':'Fixture dance'}],'limitations':[]}
        def validate(answer,request):
            self.assertEqual(answer,f.answer)
            self.assertEqual(request,read_json(self.job/'request.json'))
            return dict(self.plan)
        self.stack.enter_context(patch('avatar_assets.pipeline_prompts.validate_plan',side_effect=validate))
        with patch.object(studio_inspection,'_process_snapshot',return_value=ABSENT):
            f.release()
        self.revision=self.root/'outputs/runtime-revision'
        self.changes={name:{'before':sha256(self.job/'sources'/name) if (self.job/'sources'/name).exists() else None,
                            'after':sha256(path)} for name,path in self.live.items() if name!='openai_runner.py'}

    def prepare(self):
        return recovery.prepare_revision(self.job,self.revision,
            expected_request_sha256=sha256(self.job/'request.json'),expected_changes=self.changes,
            reason='Analytic local validator correction; unchanged cached response only.')

    def adopt(self):
        prepared=self.prepare()
        return self.fixture.jobs.adopt_cached_planner_revision(self.fixture.job,self.revision,
            expected_revision_sha256=prepared['sha256'])

    def test_adoption_and_plan_use_zero_agent_calls_and_preserve_original_evidence_and_grant(self):
        originals={str(p):sha256(p) for folder in (self.job/'sources',self.job/'agents',self.job/'events')
                   for p in folder.rglob('*') if p.is_file()}
        status=(self.job/'status.json').read_bytes();ledger=sha256(self.fixture.ledger)
        result=self.adopt();self.assertEqual(result['status'],'planned')
        self.assertTrue(result['no_dispatch'] and result['no_cost_reset'])
        self.assertEqual(len(self.fixture.executor.calls),2)
        self.assertEqual((self.revision/'original-failure-status.json').read_bytes(),status)
        agents=Mock();agents.call.side_effect=AssertionError('Cached planner must never call an agent')
        worker=pipeline.Pipeline(self.job,agents=agents);worker.guard=Mock()
        self.assertEqual(worker._plan(),self.plan)
        agents.call.assert_not_called()
        self.assertEqual(sha256(self.fixture.ledger),ledger)
        self.assertTrue(all(sha256(Path(p))==digest for p,digest in originals.items()))
        self.assertEqual(read_json(self.job/'input-receipt.json')['sources'],artifacts(self.job/'sources'))

    def test_missing_cache_after_constructor_never_falls_back_to_a_paid_planner(self):
        self.adopt();agents=Mock()
        worker=pipeline.Pipeline(self.job,agents=agents);worker.guard=Mock()
        (self.job/'agents/planner/answer.json').unlink()
        with self.assertRaisesRegex(ValueError,'cache changed'):
            worker._plan()
        agents.call.assert_not_called()

    def test_original_cache_or_failure_events_cannot_be_rewritten(self):
        self.adopt()
        (self.job/'events/0000.json').write_text('{}')
        with self.assertRaisesRegex(ValueError,'failure event changed'):
            recovery.verify_runtime_revision(self.job,self.live)

    def test_unrelated_live_source_change_is_not_admitted_even_if_requested(self):
        self.live['openai_runner.py'].write_bytes(b'UNREVIEWED BILLING CHANGE')
        self.changes['openai_runner.py']={'before':sha256(self.job/'sources/openai_runner.py'),
                                          'after':sha256(self.live['openai_runner.py'])}
        with self.assertRaisesRegex(ValueError,'exactly match the reviewed'):
            self.prepare()
        self.assertFalse(self.revision.exists())

    def test_wrong_expected_change_or_request_digest_is_rejected_before_output(self):
        self.changes['action_intent.py']['before']='0'*64
        with self.assertRaisesRegex(ValueError,'reviewed'):
            self.prepare()
        self.assertFalse(self.revision.exists())

    def test_current_runtime_drift_or_changed_old_snapshot_cannot_bypass_pins(self):
        self.adopt();path=self.live['action_intent.py'];original=path.read_bytes()
        path.write_bytes(original+b'CHANGED')
        with self.assertRaisesRegex(ValueError,'Current runtime differs'):
            pipeline.Pipeline(self.job,agents=Mock())
        path.write_bytes(original)
        (self.job/'sources/action_intent.py').write_bytes(b'OVERWRITTEN ORIGINAL')
        with self.assertRaisesRegex(ValueError,'Original source snapshots changed'):
            pipeline.Pipeline(self.job,agents=Mock())

    def test_pointer_without_public_studio_adoption_cannot_enable_runtime(self):
        prepared=self.prepare()
        write(self.job/'runtime-revision.json',{'schema_version':1,'path':str(self.revision.relative_to(self.root)),
                                             'sha256':prepared['sha256']})
        with self.assertRaisesRegex(ValueError,'not adopted through Studio'):
            pipeline.Pipeline(self.job,agents=Mock())

    def test_uncertain_planner_or_changed_original_grant_blocks_recovery(self):
        self.adopt()
        with sqlite3.connect(self.fixture.ledger) as db:db.execute("UPDATE calls SET state='reserved_uncertain'")
        with self.assertRaisesRegex(ValueError,'planner accounting changed'):
            recovery.verify_runtime_revision(self.job,self.live)

    def test_adoption_is_explicit_single_use_and_changed_revision_is_rejected(self):
        prepared=self.prepare()
        with self.assertRaisesRegex(ValueError,'Reviewed runtime revision changed'):
            self.fixture.jobs.adopt_cached_planner_revision(self.fixture.job,self.revision,
                expected_revision_sha256='0'*64)
        result=self.fixture.jobs.adopt_cached_planner_revision(self.fixture.job,self.revision,
                expected_revision_sha256=prepared['sha256'])
        with self.assertRaisesRegex(ValueError,'released failed job'):
            self.fixture.jobs.adopt_cached_planner_revision(self.fixture.job,self.revision,
                expected_revision_sha256=prepared['sha256'])
        self.assertEqual(result['new_planner_calls'],0)


if __name__=='__main__':unittest.main()
