"""Guarded saved-work retry fixtures; no models, native jobs or provider requests."""

from copy import deepcopy
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch
from types import SimpleNamespace

from avatar_assets.studio_jobs import StudioJobs, _digest, _write_private
from avatar_assets.studio_worker import _validate_prepared_request, PIPELINE_LIMITS, MEDIA_SPECS
from test_studio_jobs import FixtureExecutor, FixturePipeline


class StudioRetryTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name).resolve()/'project/tools/avatar-assets';self.root.mkdir(parents=True)
        self.executor=FixtureExecutor();self.jobs=StudioJobs(self.root,executor=self.executor)
        source=self.root/'outputs/source.glb';source.parent.mkdir();source.write_bytes(b'ANALYTIC SOURCE')
        parent=self.jobs.create('A copper telescope',style='Copper',custom_actions=['Scan the room'],reuse_model=source)
        self.parent_id=parent['id'];self.parent=self.jobs.resolve_job(self.parent_id);self.parent.mkdir()
        invocation=self.jobs.jobs_dir/self.parent_id/'run-0001';invocation.mkdir(mode=0o700)
        self.invocation=invocation;self.token='a'*32
        with self.jobs._db(True) as db:
            row=self.jobs._row(db,self.parent_id);data=json.loads(row['data'])
            data.update(operation='execute',token=self.token,owner_pid=900001,owner_identity='analytic-dead',
                        invocation_dir=str(invocation.relative_to(self.jobs.state_dir)),invocations=1)
            self.jobs._update(db,row,data,'unsupported')
            self.parent_request=json.loads(row['request'])
            receipt={'root':str(self.root),'job_id':self.parent_id,'token':self.token,'operation':'execute',
                     'invocation_dir':str(invocation),'request_digest':row['request_digest']}
        for name,value in {'invocation.json':receipt,'owner.json':{'pid':900001,'start_identity':'analytic-dead','token':self.token},
            'worker-job.json':{'pid':900002,'status':'finished','process_exit_verified':True,'exit_code':0,'manual_stop_required':False},
            'result.json':{'status':'unsupported','safe_to_release':True},
            'supervisor-result.json':{'status':'unsupported','safe_to_release':True}}.items():
            _write_private(invocation/name,value)
        for path in self.jobs._retry_locks(self.parent_id,self.parent,global_lock=True):
            path.parent.mkdir(parents=True,exist_ok=True);path.touch(mode=0o600)
        self.absent=patch('avatar_assets.studio_inspection._process_snapshot',return_value=([(os.getpid(),os.getppid(),os.getpgrp())],'analytic-os-snapshot'))
        self.absent.start();self.addCleanup(self.absent.stop)
        self.eligibility=patch('avatar_assets.standing_authorization.retry_eligibility',return_value={'eligible':True,'policy_sha256':'e'*64})
        self.eligible=self.eligibility.start();self.addCleanup(self.eligibility.stop)

    def test_save_retry_is_immutable_idempotent_and_does_not_launch_or_grant(self):
        with patch('avatar_assets.standing_authorization.authorize_successor') as grant:
            first=self.jobs.retry(self.parent_id);again=self.jobs.retry(self.parent_id)
        self.assertEqual(first['id'],again['id']);self.assertEqual(first['status'],'created')
        self.assertFalse(self.executor.calls);grant.assert_not_called()
        with self.jobs._db() as db:
            child=self.jobs._row(db,first['id']);request=json.loads(child['request'])
        self.assertEqual({k:request[k] for k in self.parent_request},self.parent_request)
        self.assertEqual(request['continuation_from'],self.parent_id)
        self.assertEqual(request['continuation_policy_sha256'],'e'*64)
        proof=json.loads((self.jobs.jobs_dir/first['id']/'retry-parent-custody.json').read_text())
        self.assertEqual(_digest(proof),request['continuation_custody_sha256'])
        self.assertEqual(self.jobs.get(self.parent_id)['status'],'unsupported')
        self.assertFalse(self.jobs.resolve_job(first['id']).exists())

    def test_active_queued_nonterminal_and_unsafe_receipts_do_not_retry(self):
        with self.jobs._db(True) as db:
            db.execute('INSERT INTO active VALUES(1,?,?)',(self.parent_id,self.token))
        with self.assertRaisesRegex(ValueError,'worker lane'):self.jobs.retry(self.parent_id)
        with self.jobs._db(True) as db:
            db.execute('DELETE FROM active');db.execute("UPDATE jobs SET status='running' WHERE id=?",(self.parent_id,))
        with self.assertRaisesRegex(ValueError,'unsupported'):self.jobs.retry(self.parent_id)
        with self.jobs._db(True) as db:db.execute("UPDATE jobs SET status='unsupported' WHERE id=?",(self.parent_id,))
        other=self.jobs.create('Other fixture')
        with self.jobs._db(True) as db:db.execute("UPDATE jobs SET status='queued' WHERE id=?",(other['id'],))
        with self.assertRaisesRegex(ValueError,'Queued'):self.jobs.retry(self.parent_id)
        with self.jobs._db(True) as db:db.execute("UPDATE jobs SET status='created' WHERE id=?",(other['id'],))
        (self.invocation/'result.json').write_text(json.dumps({'status':'unsupported','safe_to_release':False}))
        with self.assertRaisesRegex(ValueError,'terminal result'):self.jobs.retry(self.parent_id)
        self.assertFalse(self.executor.calls)

    def test_policy_source_and_present_process_fail_before_child_creation(self):
        self.eligible.return_value={'eligible':False,'reason':'No standing authorization'}
        with self.assertRaisesRegex(ValueError,'No standing'):self.jobs.retry(self.parent_id)
        self.eligible.return_value={'eligible':True,'policy_sha256':'e'*64}
        with patch('avatar_assets.studio_inspection._process_snapshot',return_value=([(900001,1,900001)],'analytic-present')):
            with self.assertRaisesRegex(ValueError,'still present'):self.jobs.retry(self.parent_id)
        source=Path(self.parent_request['reuse_model']['path']);source.write_bytes(b'CHANGED')
        with self.assertRaisesRegex(ValueError,'Selected source changed'):self.jobs.retry(self.parent_id)
        with self.jobs._db() as db:self.assertEqual(db.execute('SELECT COUNT(*) FROM jobs').fetchone()[0],1)

    def claimed_child(self):
        child=self.jobs.retry(self.parent_id);self.jobs.plan(child['id'])
        invocation=self.executor.calls[-1]
        with self.jobs._db() as db:data=json.loads(self.jobs._row(db,child['id'])['data'])
        self.jobs._worker_claim(child['id'],invocation['token'],data['owner_pid'],data['owner_identity'])
        output=self.jobs.resolve_job(child['id']);output.mkdir()
        (output/'request.json').write_text('{}')
        return child,invocation,output

    def test_worker_grant_uses_exact_child_and_rechecks_custody(self):
        child,invocation,output=self.claimed_child()
        with patch('avatar_assets.standing_authorization.authorize_successor',return_value={'granted':True}) as grant:
            self.assertEqual(self.jobs.authorize_retry(child['id'],invocation['token']),{'granted':True})
            grant.assert_called_once_with(self.root,self.parent,output,expected_policy_sha256='e'*64)
        with patch('avatar_assets.standing_authorization.authorize_successor') as grant:
            proof=self.jobs.jobs_dir/child['id']/'retry-parent-custody.json';proof.write_text('{}')
            with self.assertRaisesRegex(ValueError,'custody proof changed'):
                self.jobs.authorize_retry(child['id'],invocation['token'])
            grant.assert_not_called()

    def test_actual_worker_grants_after_binding_and_forwards_saved_parent(self):
        from avatar_assets import pipeline
        from avatar_assets.agent_backend import DEFAULT
        from avatar_assets.studio_worker import run_child
        child=self.jobs.retry(self.parent_id);self.jobs.plan(child['id'])
        invocation=self.executor.calls[-1]
        with self.jobs._db() as db:data=json.loads(self.jobs._row(db,child['id'])['data'])
        self.jobs._worker_claim(child['id'],invocation['token'],data['owner_pid'],data['owner_identity'])
        fixture=FixturePipeline();calls=[]
        settings={**DEFAULT,'max_calls':7,'max_repairs':2}
        def prepare(prompt,output,**kwargs):
            self.assertEqual(kwargs.pop('continuation_from'),self.parent)
            fixture.prepare(prompt,output,**kwargs)
            saved=json.loads((output/'request.json').read_text())
            saved['continuation']={'parent':'outputs/studio-'+self.parent_id}
            (output/'request.json').write_text(json.dumps(saved))
        def constructor(output):
            return SimpleNamespace(request=json.loads((output/'request.json').read_text()),
                run=lambda **kwargs: (calls.append(kwargs) or {'status':'planned'}))
        def grant(*args,**kwargs):
            self.assertTrue((self.jobs.jobs_dir/child['id']/'pipeline-request.json').is_file())
            self.assertTrue((self.jobs.resolve_job(child['id'])/'request.json').is_file())
            return {'granted':True}
        with patch('avatar_assets.agent_backend.server_settings',return_value=settings), \
             patch('avatar_assets.agent_backend.readiness',return_value={'can_run':True}), \
             patch('avatar_assets.pipeline_continuation.verify',return_value={}), \
             patch('avatar_assets.standing_authorization.authorize_successor',side_effect=grant) as authorization, \
             patch.object(pipeline,'prepare',side_effect=prepare),patch.object(pipeline,'Pipeline',side_effect=constructor):
            result=run_child(self.jobs,child['id'],invocation['token'])
        self.assertEqual(result['status'],'planned');authorization.assert_called_once()
        self.assertEqual(len(calls),1);self.assertTrue(calls[0]['plan_only'])
        with patch('avatar_assets.agent_backend.server_settings',return_value=settings), \
             patch('avatar_assets.pipeline_continuation.verify',return_value={}), \
             patch('avatar_assets.standing_authorization.authorize_successor',side_effect=ValueError('Policy revoked')), \
             patch.object(pipeline,'Pipeline') as runner:
            result=run_child(self.jobs,child['id'],invocation['token'])
        self.assertEqual(result['status'],'awaiting_authorization');runner.assert_not_called()

    def test_prepared_continuation_is_exact_parent_and_requires_verification(self):
        request={**self.parent_request,'continuation_from':self.parent_id}
        output=self.root/'outputs/studio-fixture'
        prepared={'schema_version':1,'id':output.name,'prompt':request['prompt'],'style':request['style'],
                  'custom_actions':request['custom_actions'],'states':[],'preset':'preview','media_spec':MEDIA_SPECS['preview'],
                  'max_credits':0,**PIPELINE_LIMITS,'reference':None,'created':'analytic',
                  'reuse_model':{'path':'inputs/source-model.glb','original_path':request['reuse_model']['path'],
                                 'bytes':request['reuse_model']['bytes'],'sha256':'a'*64},
                  'continuation':{'parent':'outputs/studio-'+self.parent_id}}
        with patch('avatar_assets.pipeline_continuation.verify',return_value={}) as verify:
            _validate_prepared_request(prepared,request,output)
            verify.assert_called_once_with(self.root,output,prepared)
            bad=deepcopy(prepared);bad['continuation']['parent']='outputs/studio-other'
            with self.assertRaisesRegex(ValueError,'parent differs'):_validate_prepared_request(bad,request,output)
            bad=deepcopy(prepared);del bad['continuation']
            with self.assertRaisesRegex(ValueError,'contract'):_validate_prepared_request(bad,request,output)


if __name__=='__main__':unittest.main()
