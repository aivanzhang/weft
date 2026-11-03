"""Analytic unknown-charge ownership recovery; no provider or native execution."""
from contextlib import ExitStack, closing
import json
from pathlib import Path
import sqlite3
import unittest
from unittest.mock import patch

from avatar_assets import openai_runner as api, studio_inspection as inspection
from avatar_assets import studio_unknown_release as release
from avatar_assets.safety import sha256,read_json
from tests.test_author_input_recovery import AuthorInputRecoveryTests
from tests.test_studio_api_release import ABSENT,write


class UnknownReleaseTests(unittest.TestCase):
    def setUp(self):
        fixture=AuthorInputRecoveryTests('runTest');fixture.setUp();self.addCleanup(fixture.doCleanups)
        self.fixture=fixture;self.f=fixture.f;self.output=fixture.job;self.jobs=self.f.jobs
        ledger=api.OpenAILedger(self.f.root/'.local/openai')
        try:
            with patch.object(api,'read_api_key',return_value='ANALYTIC NOT REAL'), \
                    patch.object(api,'post_response',side_effect=ValueError('OpenAI transport failed; reservation retained, no automatic retry')):
                runner=api.OpenAIRunner(self.output/'agents',job_id=self.output.name,
                    job_digest=sha256(self.output/'request.json'),ledger=ledger,max_calls=7)
                with self.assertRaisesRegex(ValueError,'transport failed'):
                    runner.call('author-0','Analytic first author',{'type':'object','properties':{},'required':[],
                                                                 'additionalProperties':False})
        finally:ledger.close()
        write(self.output/'status.json',{'status':'checkpointed_error','stage':'pipeline','automatic_retry':False,
              'error':'OpenAI transport failed; reservation retained, no automatic retry'})
        self.stack=ExitStack();self.addCleanup(self.stack.close)
        self.stack.enter_context(patch.object(api,'read_api_key',side_effect=AssertionError('No key read')))
        self.stack.enter_context(patch.object(api,'post_response',side_effect=AssertionError('No provider call')))
        self.stack.enter_context(patch.object(self.jobs,'_dispatch',side_effect=AssertionError('No dispatch')))

    def invoke(self):
        return self.jobs.release_uncertain_inspected(self.f.job,expected_pid=self.f.owner_pid,
            expected_identity=self.f.identity,reason='Analytic custody release, preserve the entire unknown hold.')

    def verify(self,result):
        return release.verify_release(self.f.root,self.jobs.state_dir/result['receipt'],result['sha256'],
            parent_job_id=self.output.name,parent_digest=sha256(self.output/'request.json'))

    def test_release_changes_only_lifecycle_and_cannot_restart_or_dispatch(self):
        before={str(p):sha256(p) for p in self.output.rglob('*') if p.is_file()};ledger=sha256(self.f.ledger)
        result=self.invoke();record=self.verify(result)
        self.assertEqual(result['status'],'needs_inspection');self.assertFalse(result['old_job_restart_allowed'])
        self.assertEqual(result['unknown_reserved_microdollars'],5618640)
        self.assertEqual(sha256(self.f.ledger),ledger)
        self.assertEqual(before,{str(p):sha256(p) for p in self.output.rglob('*') if p.is_file()})
        self.assertNotIn(self.f.token,json.dumps(record));self.assertEqual(len(self.f.executor.calls),2)
        self.assertEqual(record['proof']['unknown_provider']['call_rows'][0]['call']['state'],'reserved_uncertain')
        with self.assertRaisesRegex(ValueError,'terminal|inspection'):self.jobs.plan(self.f.job)
        with self.assertRaisesRegex(ValueError,'unreleased'):self.invoke()
        with self.jobs._db() as db:self.assertFalse(db.execute('SELECT 1 FROM active').fetchone())

    def test_live_process_blocks_release(self):
        with patch.object(inspection,'_process_snapshot',return_value=([(self.f.owner_pid,1,self.f.owner_pid)],'a'*64)):
            with self.assertRaisesRegex(ValueError,'live|present|exists'):self.invoke()
        self.f.retained()

    def test_unknown_row_mutation_rejected(self):
        with closing(sqlite3.connect(self.f.ledger)) as db, db:db.execute("UPDATE calls SET request_digest=? WHERE role='author-0'",('f'*64,))
        with self.assertRaisesRegex(ValueError,'held charge|price profile'):self.invoke()
        self.f.retained()

    def test_response_or_extra_author_prevents_release(self):
        write(self.output/'agents/author-0/response.json',{'id':'resp_unvalidated'})
        with self.assertRaisesRegex(ValueError,'response|extra'):self.invoke()
        self.f.retained()

    def test_wrong_checkpoint_or_remote_work_prevents_release(self):
        write(self.output/'status.json',{'status':'checkpointed_error','stage':'pipeline','automatic_retry':False,'error':'Other error'})
        with self.assertRaisesRegex(ValueError,'checkpoint'):self.invoke()
        (self.output/'generation').mkdir()
        with self.assertRaisesRegex(ValueError,'remote'):self.invoke()
        self.f.retained()

    def test_symlink_and_changed_input_prevent_release(self):
        path=self.output/'agents/author-0/schema.json';saved=path.read_bytes();path.unlink();path.symlink_to(self.output/'plan.json')
        with self.assertRaisesRegex(ValueError,'symlink|unsafe'):self.invoke()
        path.unlink();path.write_bytes(saved);self.f.source.write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError,'identity|bytes'):self.invoke()
        self.f.retained()

    def test_held_lock_prevents_release(self):
        import fcntl
        with (self.output/'pipeline.lock').open('rb') as stream:
            fcntl.flock(stream,fcntl.LOCK_EX|fcntl.LOCK_NB)
            with self.assertRaisesRegex((ValueError,BlockingIOError),'lock|temporarily'):self.invoke()
        self.f.retained()

    def test_release_verifier_rejects_marker_and_output_tamper(self):
        result=self.invoke();self.verify(result)
        with self.jobs._db(True) as db:
            row=self.jobs._row(db,self.f.job);data=json.loads(row['data']);data['uncertain_owner_release']['sha256']='f'*64
            self.jobs._update(db,row,data)
        with self.assertRaisesRegex(ValueError,'marker'):self.verify(result)
        with self.jobs._db(True) as db:
            row=self.jobs._row(db,self.f.job);data=json.loads(row['data']);data['uncertain_owner_release']['sha256']=result['sha256']
            self.jobs._update(db,row,data)
        (self.output/'extra.txt').write_text('unbound')
        with self.assertRaisesRegex(ValueError,'output changed'):self.verify(result)

    def test_release_verifier_does_not_require_unchanged_global_ledger(self):
        result=self.invoke()
        with closing(sqlite3.connect(self.f.ledger)) as db, db:db.execute('CREATE TABLE unrelated_test(value TEXT)')
        self.verify(result)
        # Accounting closure independently compares exact current parent rows under its transaction.

    def test_wrong_owner_and_queued_work_retain_lease(self):
        with self.assertRaisesRegex(ValueError,'identity changed'):
            self.jobs.release_uncertain_inspected(self.f.job,expected_pid=self.f.owner_pid+1,
                expected_identity=self.f.identity,reason='Analytic mismatch')
        with self.jobs._db(True) as db:
            db.execute("INSERT INTO jobs SELECT 'ffffffffffffffffffffffff','other',request_digest,request,'studio-other',created,'queued',1,data FROM jobs WHERE id=?",(self.f.job,))
        with self.assertRaisesRegex(ValueError,'Queued work'):self.invoke()
        self.f.retained()

    def test_oversized_or_unsealed_inspection_retain_lease(self):
        path=self.output/'agents/author-0/prompt.txt';path.write_text('x'*(160*1024+1))
        with self.assertRaisesRegex(ValueError,'bound'):self.invoke()
        path.write_text('Analytic first author')
        native=read_json(self.output/'inspection/job.json');native['process_exit_verified']=False
        write(self.output/'inspection/job.json',native);self.fixture.reseal_inspection()
        with self.assertRaisesRegex(ValueError,'Unsealed|successful exit'):self.invoke()
        self.f.retained()

    def test_release_verifier_rejects_receipt_hash_and_wrong_parent(self):
        result=self.invoke();path=self.jobs.state_dir/result['receipt']
        with self.assertRaisesRegex(ValueError,'receipt changed'):
            release.verify_release(self.f.root,path,'f'*64,parent_job_id=self.output.name,
                                   parent_digest=sha256(self.output/'request.json'))
        with self.assertRaisesRegex(ValueError,'request binding'):
            release.verify_release(self.f.root,path,result['sha256'],parent_job_id=self.output.name,parent_digest='f'*64)

    def test_ledger_change_between_collection_and_commit_retains_lease(self):
        original=release.recheck_unknown;count=0
        def changed(proof,**kwargs):
            nonlocal count
            count+=1
            if count==2:
                with closing(sqlite3.connect(self.f.ledger)) as db, db:db.execute('CREATE TABLE concurrent_change(value TEXT)')
            return original(proof,**kwargs)
        with patch.object(release,'recheck_unknown',side_effect=changed):
            with self.assertRaisesRegex(ValueError,'custody changed'):self.invoke()
        self.f.retained()


if __name__=='__main__':unittest.main()
