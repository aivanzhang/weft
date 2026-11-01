"""Analytic custody fixtures only; no real workers, OS inventory, providers or generated assets."""

import fcntl
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from avatar_assets import studio_inspection as inspection
from avatar_assets.studio_jobs import StudioJobs
from avatar_assets.studio_worker import run_child
from tests.test_studio_jobs import FixtureExecutor, FixturePipeline, write


ABSENT=([(90000,1,90000)],'f'*64)


class StudioInspectionTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name).resolve()/'project/tools/avatar-assets';self.root.mkdir(parents=True)
        self.source=self.root/'outputs/fixture.glb';self.source.parent.mkdir()
        self.source.write_bytes(b'ANALYTIC FIXTURE; NOT A GENERATED MODEL')
        self.executor=FixtureExecutor();self.jobs=StudioJobs(self.root,executor=self.executor)
        self.job=self.jobs.create('A fixture robot',custom_actions=['Wave one hand'],reuse_model=self.source)['id']
        self.jobs.plan(self.job)
        self._claim()
        result=run_child(self.jobs,self.job,self.token,pipeline_module=FixturePipeline())
        self.jobs._worker_complete(self.job,self.token,result['status'],safe_to_release=True)
        self.jobs.start(self.job,self.jobs.get(self.job)['plan_id'])
        self._claim()
        self.jobs._worker_complete(self.job,self.token,'needs_inspection',safe_to_release=False,
                                   error='Analytic critic contract failure; no accepted delivery.')
        self.output=self.jobs.resolve_job(self.job)
        self.invocation=self.jobs.state_dir/self.data['invocation_dir']
        self.owner_pid=self.data['owner_pid'];self.identity=self.data['owner_identity']
        self.child_pid=20001;self.native_pid=30001;self.agent_pid=40001
        write(self.invocation/'owner.json',{'pid':self.owner_pid,'start_identity':self.identity,'token':self.token})
        write(self.invocation/'worker-job.json',self.exit(self.child_pid,exit_code=1))
        for name in ('result.json','supervisor-result.json'):
            write(self.invocation/name,{'status':'needs_inspection','safe_to_release':False,'provenance':'analytic_fixture'})
        write(self.output/'model-1/finish-handle.json',{'pid':self.native_pid,'monitor_checkpoint':'finish-monitor.json'})
        write(self.output/'model-1/finish-monitor.json',self.exit(self.native_pid))
        write(self.output/'model-1/job.json',self.exit(self.native_pid))
        agent=self.exit(self.agent_pid);agent.pop('process_exit_verified')
        write(self.output/'agents/author-0/agent-process.json',agent)
        write(self.output/'agents/usage.json',{'calls':4,'input_tokens':1234,'output_tokens':567,'dollars':None})
        (self.output/'model-1/rejected.glb').write_bytes(b'REJECTED ANALYTIC ARTIFACT; KEEP IMMUTABLE')
        self.locks=[self.jobs.state_dir/'worker.lock',self.jobs.jobs_dir/self.job/'worker.lock',self.output/'pipeline.lock']
        for path in self.locks:path.touch()
        self.original_files=self.file_hashes()

    @staticmethod
    def exit(pid,**fields):
        return {'pid':pid,'status':'finished','exit_code':0,'process_exit_verified':True,
                'manual_stop_required':False,**fields}

    def _claim(self):
        with self.jobs._db() as db:
            row=self.jobs._row(db,self.job);self.data=json.loads(row['data'])
        self.token=self.data['token']
        self.jobs._worker_claim(self.job,self.token,self.data['owner_pid'],self.data['owner_identity'])

    def file_hashes(self):
        return {str(path.relative_to(self.root)):hashlib.sha256(path.read_bytes()).hexdigest()
                for folder in (self.output,self.invocation) for path in folder.rglob('*') if path.is_file()}

    def release(self,**fields):
        return self.jobs.release_inspected(self.job,expected_pid=self.owner_pid,
                expected_identity=self.identity,reason='Reviewed failed analytic custody; never replay.',**fields)

    def retained(self):
        with self.jobs._db() as db:
            row=self.jobs._row(db,self.job)
            self.assertEqual(row['status'],'needs_inspection')
            self.assertEqual(tuple(db.execute('SELECT job,token FROM active').fetchone()),(self.job,self.token))
        self.assertFalse(list((self.jobs.jobs_dir/self.job).glob('inspection-release-*.json')))
        self.assertEqual(len(self.executor.calls),2)

    def blocked(self,message):
        with patch.object(inspection,'_process_snapshot',return_value=ABSENT),self.assertRaisesRegex(ValueError,message):
            self.release()
        self.retained()

    def test_release_is_hash_bound_private_preserves_outputs_costs_history_and_never_dispatches(self):
        with self.jobs._db() as db:
            before=self.jobs._row(db,self.job);before_data=json.loads(before['data'])
            previous_events=list(db.execute('SELECT seq,event FROM journal WHERE job=?',(self.job,)))
        with patch.object(inspection,'_process_snapshot',return_value=ABSENT) as observed,\
                patch.object(self.jobs,'_dispatch',side_effect=AssertionError('Recovery must not dispatch')):
            result=self.release()
        self.assertEqual(observed.call_count,2)
        self.assertEqual(result['status'],'failed');self.assertTrue(result['lease_released'])
        self.assertTrue(result['no_dispatch']);self.assertTrue(result['no_cost_reset'])
        path=self.jobs.state_dir/result['receipt'];raw=path.read_bytes();proof=json.loads(raw)
        self.assertEqual(hashlib.sha256(raw).hexdigest(),result['sha256'])
        self.assertEqual(path.stat().st_mode&0o777,0o600)
        self.assertNotIn(self.token,raw.decode());self.assertNotIn('command',raw.decode())
        self.assertEqual(proof['proof']['known_pids'],sorted([self.owner_pid,self.child_pid,self.native_pid,self.agent_pid]))
        self.assertEqual(len(proof['proof']['locks']),3)
        self.assertEqual(sum(r['kind']=='terminal_result' for r in proof['proof']['receipts']),2)
        for record in proof['proof']['receipts']:
            self.assertEqual(hashlib.sha256((self.root/record['path']).read_bytes()).hexdigest(),record['sha256'])
        self.assertEqual(self.original_files,self.file_hashes())
        with self.jobs._db() as db:
            after=self.jobs._row(db,self.job);after_data=json.loads(after['data'])
            self.assertEqual(after['status'],'failed');self.assertFalse(db.execute('SELECT * FROM active').fetchall())
            self.assertEqual(after['request'],before['request']);self.assertEqual(after['request_digest'],before['request_digest'])
            for key,value in before_data.items():
                if key!='updated':self.assertEqual(after_data[key],value)
            events=list(db.execute('SELECT seq,event FROM journal WHERE job=?',(self.job,)))
            self.assertEqual([tuple(row) for row in events[:-1]],[tuple(row) for row in previous_events])
            self.assertEqual(json.loads(events[-1]['event'])['action'],'inspection_release')
            self.assertNotIn(self.token,events[-1]['event'])
        self.assertEqual(len(self.executor.calls),2)
        with self.assertRaisesRegex(ValueError,'Only a retained'):self.release()

    def test_live_owner_child_or_process_group_is_not_released(self):
        for rows in [[(self.owner_pid,1,self.owner_pid)],[(99991,self.child_pid,99991)],[(99992,1,self.native_pid)]]:
            with self.subTest(rows=rows),patch.object(inspection,'_process_snapshot',return_value=(rows,'f'*64)),\
                    self.assertRaisesRegex(ValueError,'still present'):
                self.release()
            self.retained()

    def test_second_process_check_blocks_newly_present_owner(self):
        with patch.object(inspection,'_process_snapshot',side_effect=[ABSENT,([(self.owner_pid,1,self.owner_pid)],'e'*64)]),\
                self.assertRaisesRegex(ValueError,'still present'):
            self.release()
        self.retained()

    def test_controller_signal_unverified_or_unfinished_exit_is_not_sufficient(self):
        for fields in [{'exit_code':-9},{'exit_code':True},{'status':'running'},
                       {'manual_stop_required':True},{'process_exit_verified':False}]:
            with self.subTest(fields=fields):
                write(self.invocation/'worker-job.json',self.exit(self.child_pid,**fields))
                self.blocked('Controller child')

    def test_missing_or_unsealed_terminal_results_fail_closed(self):
        for name in ('result.json','supervisor-result.json'):
            path=self.invocation/name;original=path.read_bytes()
            for fields in [None,{'status':'running','safe_to_release':False},
                           {'status':'needs_inspection','safe_to_release':True}]:
                with self.subTest(name=name,fields=fields):
                    if fields is None:path.unlink()
                    else:write(path,fields)
                    self.blocked('missing|terminal result')
                    path.write_bytes(original)

    def test_unclosed_handle_and_uncertain_agent_receipt_block_release(self):
        path=self.output/'model-1/finish-monitor.json';original=path.read_bytes();path.unlink()
        self.blocked('matching terminal monitor');path.write_bytes(original)
        write(self.output/'agents/author-0/agent-process.json',self.exit(self.agent_pid,manual_stop_required=True))
        self.blocked('Unsealed or unresolved')

    def test_reused_pid_cannot_close_a_different_handle_or_omit_native_exit_verification(self):
        path=self.output/'model-1/finish-monitor.json';value=self.exit(self.agent_pid);write(path,value)
        self.blocked('matching terminal monitor')
        value=self.exit(self.native_pid);value.pop('process_exit_verified');write(path,value)
        self.blocked('Unsealed or unresolved')
        write(path,self.exit(self.native_pid))
        write(self.output/'model-1/finish-handle.json',{'pid':self.native_pid,'monitor_checkpoint':'../other-monitor.json'})
        self.blocked('exact monitor')

    def test_identity_lease_and_invocation_binding_are_exact(self):
        with self.assertRaisesRegex(ValueError,'identity changed'):
            self.jobs.release_inspected(self.job,expected_pid=self.owner_pid,expected_identity='other',reason='Fixture')
        self.retained()
        meta_path=self.invocation/'invocation.json';meta=json.loads(meta_path.read_text());meta['request_digest']='a'*64
        write(meta_path,meta);self.blocked('Invocation request binding')
        with self.jobs._db(True) as db:db.execute('UPDATE active SET token=?',('other',))
        with self.assertRaisesRegex(ValueError,'Exact old lease'):self.release()
        with self.jobs._db() as db:
            self.assertEqual(tuple(db.execute('SELECT job,token FROM active').fetchone()),(self.job,'other'))

    def test_immutable_request_and_generation_artifacts_are_not_reinterpreted(self):
        path=self.jobs.jobs_dir/self.job/'request.json';original=path.read_bytes();write(path,{'prompt':'replacement'})
        self.blocked('Immutable Studio request');path.write_bytes(original)
        (self.output/'generation').mkdir();self.blocked('Remote-generation artifacts')

    def test_remote_request_and_queued_work_need_separate_inspection(self):
        with self.jobs._db(True) as db:
            row=self.jobs._row(db,self.job);original=row['request'];request=json.loads(original);request['reuse_model']=None
            db.execute('UPDATE jobs SET request=? WHERE id=?',(json.dumps(request),self.job))
        self.blocked('Remote-generation custody')
        with self.jobs._db(True) as db:db.execute('UPDATE jobs SET request=? WHERE id=?',(original,self.job))
        other=self.jobs.create('Another fixture',reuse_model=self.source)['id'];self.jobs.plan(other)
        self.blocked('Queued work')
        self.assertEqual(self.jobs.get(other)['status'],'queued')

    def test_existing_pipeline_and_supervisor_locks_must_be_free(self):
        for path in self.locks:
            with self.subTest(path=path.name),path.open('rb') as stream:
                fcntl.flock(stream,fcntl.LOCK_EX | fcntl.LOCK_NB)
                self.blocked('still owns its lock')
        self.locks[2].unlink();self.blocked('lock is missing')

    def test_symlinks_and_oversized_receipts_cannot_bypass_custody_bounds(self):
        link=self.output/'unrelated-link';link.symlink_to(self.source)
        self.blocked('symlink');link.unlink()
        path=self.output/'model-1/job.json';path.write_bytes(b' '* (inspection.MAX_JSON+1))
        self.blocked('oversized')

    def test_inventory_and_receipt_limits_are_enforced(self):
        with patch.object(inspection,'MAX_FILES',1):self.blocked('inventory exceeds')
        with patch.object(inspection,'MAX_RECEIPTS',1):self.blocked('Too many custody receipts')
        with patch.object(inspection,'MAX_RECEIPT_BYTES',1):self.blocked('receipt bytes')

    def test_hash_race_after_initial_inventory_preserves_lease(self):
        real=inspection.collect
        def changed(*args):
            proof=real(*args)
            path=self.output/'model-1/job.json';value=json.loads(path.read_text());value['late_field']='changed'
            write(path,value)
            return proof
        with patch.object(inspection,'collect',side_effect=changed):self.blocked('changed during inspection')

    def test_owner_race_under_database_lock_preserves_replacement_lease(self):
        real=inspection.collect
        def changed(*args):
            proof=real(*args)
            with self.jobs._db(True) as db:db.execute('UPDATE active SET token=?',('replacement',))
            return proof
        with patch.object(inspection,'collect',side_effect=changed),\
                patch.object(inspection,'_process_snapshot',return_value=ABSENT),\
                self.assertRaisesRegex(ValueError,'Exact old lease'):
            self.release()
        with self.jobs._db() as db:
            self.assertEqual(self.jobs._row(db,self.job)['status'],'needs_inspection')
            self.assertEqual(tuple(db.execute('SELECT job,token FROM active').fetchone()),(self.job,'replacement'))
        self.assertFalse(list((self.jobs.jobs_dir/self.job).glob('inspection-release-*.json')))

    def test_failed_transaction_leaves_audit_receipt_but_never_releases_or_resets(self):
        with patch.object(inspection,'_process_snapshot',return_value=ABSENT),\
                patch.object(self.jobs,'_journal',side_effect=ValueError('fixture journal bound')),\
                self.assertRaisesRegex(ValueError,'journal bound'):
            self.release()
        with self.jobs._db() as db:
            self.assertEqual(self.jobs._row(db,self.job)['status'],'needs_inspection')
            self.assertEqual(tuple(db.execute('SELECT job,token FROM active').fetchone()),(self.job,self.token))
        receipts=list((self.jobs.jobs_dir/self.job).glob('inspection-release-*.json'));self.assertEqual(len(receipts),1)
        self.assertEqual(json.loads(receipts[0].read_text())['requested_transition'],'failed_and_release_exact_lease')
        self.assertEqual(self.original_files,self.file_hashes());self.assertEqual(len(self.executor.calls),2)


if __name__=='__main__':unittest.main()
