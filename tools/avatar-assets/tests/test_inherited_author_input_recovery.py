"""Zero-call custody fixtures; ancestor verification mocked, real child ledger/recovery exercised."""
from contextlib import ExitStack
import json
from pathlib import Path
import shutil
import sqlite3
import unittest
from unittest.mock import Mock,patch

from avatar_assets import pipeline,pipeline_recovery as recovery
from avatar_assets.studio_jobs import _digest
from avatar_assets.safety import read_json,sha256
from tests import test_author_input_recovery as paid_fixture
from tests.test_studio_api_release import write


class InheritedAuthorInputRecoveryTests(unittest.TestCase):
    def setUp(self):
        fixture=paid_fixture.AuthorInputRecoveryTests();fixture.setUp();self.addCleanup(fixture.doCleanups)
        self.base=fixture;self.f=fixture.f;self.job=fixture.job;self.root=fixture.root
        self.plan=fixture.plan;self.context={'plan':self.plan,'previous_answer':{'status':'ready'},'findings':{}}
        # Establish an analytic zero-call starting state; no production records are modified.
        shutil.rmtree(self.job/'agents/planner')
        request=read_json(self.job/'request.json')
        request['continuation']={'schema_version':1,'policy':'terminal-native-repair-continuation-v1',
            'parent':'outputs/studio-analytic-parent','receipt':'continuation/receipt.json','sha256':'e'*64}
        write(self.job/'request.json',request);digest=sha256(self.job/'request.json')
        input_receipt=read_json(self.job/'input-receipt.json');input_receipt['request_sha256']=digest
        write(self.job/'input-receipt.json',input_receipt)
        binding=self.f.jobs.jobs_dir/self.f.job/'pipeline-request.json';value=read_json(binding)
        self.f.request={**self.f.request,'continuation_from':'analytic-parent'}
        studio_digest=_digest(self.f.request)
        write(self.f.jobs.jobs_dir/self.f.job/'request.json',self.f.request)
        with self.f.jobs._db(True) as db:
            db.execute('UPDATE jobs SET request=?,request_digest=? WHERE id=?',
                       (json.dumps(self.f.request),studio_digest,self.f.job))
        for invocation in (self.f.prior,self.f.invocation):
            meta=read_json(invocation/'invocation.json');meta['request_digest']=studio_digest;write(invocation/'invocation.json',meta)
        value['pipeline_request_sha256']=digest;value['studio_request_digest']=studio_digest;write(binding,value)
        runner=read_json(self.job/'agents/runner.json');runner['job_digest']=digest;write(self.job/'agents/runner.json',runner)
        write(self.job/'continuation/context.json',self.context)
        write(self.job/'continuation/receipt.json',{'fixture':'Analytic ancestor; verified by controlled stand-in'})
        with sqlite3.connect(self.f.ledger) as db:
            db.execute('DELETE FROM calls WHERE job=?',(self.job.name,))
            db.execute('UPDATE jobs SET digest=? WHERE id=?',(digest,self.job.name))
        status=read_json(self.job/'status.json');status['error']='API prompt exceeds bound';write(self.job/'status.json',status)
        self.stack=ExitStack();self.addCleanup(self.stack.close)
        self.verify=self.stack.enter_context(patch('avatar_assets.pipeline_continuation.verify',return_value=self.context))

    def adopt(self):
        self.f.release();prepared=self.base.prepare()
        return self.f.jobs.adopt_cached_planner_revision(self.f.job,self.base.revision,expected_revision_sha256=prepared['sha256'])

    def test_release_adopt_keeps_all_seven_calls_and_never_requests_a_planner(self):
        old_ledger=sha256(self.f.ledger);old_plan=(self.job/'plan.json').read_bytes()
        result=self.adopt();self.assertEqual(result['status'],'planned')
        record=read_json(self.base.revision/'revision.json')
        self.assertEqual(record['remaining_role_calls'],7);self.assertEqual(record['planner_artifacts'],[])
        proof=read_json(self.base.revision/'api-custody.json')
        self.assertEqual(proof['calls'],[]);self.assertEqual(proof['policy'],'inherited-plan-author-input-bound-custody-v1')
        agents=Mock();agents.call.side_effect=AssertionError('No paid planner')
        worker=pipeline.Pipeline(self.job,agents=agents);worker.guard=Mock()
        self.assertEqual(worker._plan(),self.plan);agents.call.assert_not_called()
        self.assertEqual(sha256(self.f.ledger),old_ledger);self.assertEqual((self.job/'plan.json').read_bytes(),old_plan)
        self.assertFalse((self.job/'agents/planner').exists());self.verify.assert_called()

    def test_any_call_or_author_folder_blocks_zero_call_custody(self):
        with sqlite3.connect(self.f.ledger) as db:
            db.execute('INSERT INTO calls(job,role,request_digest,reserved,state) VALUES(?,?,?,?,?)',
                (self.job.name,'author-0','f'*64,5618640,'reserved_uncertain'))
        with self.assertRaisesRegex(ValueError,'zero API calls'):self.f.release()
        with sqlite3.connect(self.f.ledger) as db:db.execute('DELETE FROM calls')
        (self.job/'agents/author-0').mkdir()
        with self.assertRaisesRegex(ValueError,'Only one API'):self.f.release()

    def test_missing_inherited_proof_replan_or_changed_plan_cannot_be_recovered(self):
        self.verify.side_effect=ValueError('Analytic ancestor hash mismatch')
        with self.assertRaisesRegex(ValueError,'ancestor hash'):self.f.release()
        self.verify.side_effect=None;self.verify.return_value={**self.context,'replan':{}}
        with self.assertRaisesRegex(ValueError,'required paid replan'):self.f.release()
        self.verify.return_value=self.context
        write(self.job/'plan.json',{**self.plan,'subject':'Changed'})
        with self.assertRaisesRegex(ValueError,'unchanged valid'):self.f.release()

    def test_post_adoption_uncertain_charge_or_changed_continuation_blocks_resume(self):
        self.adopt()
        with sqlite3.connect(self.f.ledger) as db:
            db.execute('INSERT INTO calls(job,role,request_digest,reserved,state) VALUES(?,?,?,?,?)',
                (self.job.name,'author-0','f'*64,5618640,'reserved_uncertain'))
        with self.assertRaisesRegex(ValueError,'uncertain call'):
            recovery.verify_runtime_revision(self.job,self.base.live)
        with sqlite3.connect(self.f.ledger) as db:db.execute('DELETE FROM calls')
        write(self.job/'continuation/context.json',{'changed':True})
        with self.assertRaisesRegex(ValueError,'inherited plan custody changed'):
            recovery.verify_runtime_revision(self.job,self.base.live)


if __name__=='__main__':unittest.main()
