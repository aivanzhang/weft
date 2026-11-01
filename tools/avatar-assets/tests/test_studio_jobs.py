"""Durable lifecycle fixtures only; no agents, providers, native rendering or servers."""

import hashlib
import ctypes
from copy import deepcopy
import json
import os
from pathlib import Path
import sqlite3
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from unittest.mock import Mock

from avatar_assets.studio_jobs import (ACTIVE_RESERVATION_BYTES, STORED_BYTES_LIMIT, StudioJobs,
                                      _digest, _write_private, process_identity)
from avatar_assets.studio_worker import run_child, supervise


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value))


class FixtureExecutor:
    provenance='analytic_fixture'

    def __init__(self):self.calls=[]

    def launch(self,invocation):
        self.calls.append(invocation)
        return {'pid':10000+len(self.calls),'start_identity':'fixture-start-'+str(len(self.calls)),
                'provenance':self.provenance}


class FixturePipeline:
    """Writes clearly analytic packaging fixtures, not valid media/model content."""
    def __init__(self):self.prepared=[];self.calls=[];self.terminal='complete'

    def prepare(self,prompt,output,**settings):
        self.prepared.append((prompt,settings))
        output.mkdir(parents=True)
        spec={'draft':{'size':256,'portrait':512,'seconds':2,'samples':4},
              'preview':{'size':720,'portrait':1600,'seconds':4,'samples':8},
              'muse':{'size':720,'portrait':1600,'seconds':8,'samples':8}}[settings['preset']]
        request={'schema_version':1,'id':output.name,'prompt':prompt,'style':'','custom_actions':[],
                 'states':[],'max_repairs':1,'max_agent_calls':10,'agent_timeout':240,'agent_model':None,
                 **settings,'media_spec':spec,'created':'2026-11-01T12:00:00-08:00'}
        for field,name in (('reference','reference'),('reuse_model','source-model')):
            original=settings.get(field)
            request[field]=None
            if original:
                original=Path(original);target=output/'inputs'/(name+original.suffix.lower())
                target.parent.mkdir(exist_ok=True);target.write_bytes(original.read_bytes())
                request[field]={'path':str(target.relative_to(output)),'original_path':str(original),
                                'bytes':target.stat().st_size,'sha256':hashlib.sha256(target.read_bytes()).hexdigest()}
        write(output/'request.json',request)
        write(output/'input-receipt.json',{'request_sha256':hashlib.sha256((output/'request.json').read_bytes()).hexdigest()})

    def Pipeline(self,folder):
        request=json.loads((folder/'request.json').read_text())
        receipt=json.loads((folder/'input-receipt.json').read_text())
        if receipt['request_sha256']!=hashlib.sha256((folder/'request.json').read_bytes()).hexdigest():
            raise ValueError('Analytic pipeline self-receipt changed')
        def run(**settings):
            self.calls.append(settings)
            if (folder/'delivery.json').exists():return {'status':'complete'}
            if settings['cancel_requested']():return {'status':'cancelled'}
            request=json.loads((folder/'request.json').read_text())
            write(folder/'plan.json',{'status':'supported','subject':request['prompt'],'limitations':['Analytic fixture; no generated avatar.']})
            write(folder/'events/0000.json',{'stage':'planner','status':'supported'})
            if not request['reuse_model']:
                write(folder/'generation/plan.json',{'id':folder.name+'-mesh','max_credits':35,'estimated_credits':35})
            if settings['plan_only']:
                return {'status':'planned' if request['reuse_model'] else 'awaiting_generation_authorization'}
            if self.terminal!='complete':return {'status':self.terminal}
            for name in ['model-0/model.glb','model-0/source.blend','model-0/portrait.png',
                         'media-portable/portrait.webp','media-portable/animations/dance.mp4','media-portable/fallbacks/dance.png']:
                path=folder/name;path.parent.mkdir(parents=True,exist_ok=True)
                path.write_bytes(b'ANALYTIC FIXTURE ONLY; not generated media')
            write(folder/'media-portable/manifest.json',{'clips':[{'id':'dance','loop_verified':False}]})
            write(folder/'delivery.json',{'status':'complete','model':'model-0/model.glb',
                'editable_source':'model-0/source.blend','limitations':['Analytic fixture, not actual avatar generation.']})
            write(folder/'events/0001.json',{'stage':'delivery','status':'complete'})
            return {'status':'complete'}
        return SimpleNamespace(run=run,request=request)


class StudioJobsTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.project=Path(self.temp.name).resolve()/'project'
        self.root=self.project/'tools/avatar-assets';self.root.mkdir(parents=True)
        self.executor=FixtureExecutor();self.jobs=StudioJobs(self.root,executor=self.executor)
        self.pipeline=FixturePipeline()
        self.source=self.root/'outputs/fixture.glb';self.source.parent.mkdir()
        self.source.write_bytes(b'ANALYTIC GLB PLACEHOLDER; only the injected fixture accepts this')

    def create(self,reuse=True,**kwargs):
        return self.jobs.create('A copper fox with glasses',custom_actions=['Dance then bow'],
                                reuse_model=self.source if reuse else None,**kwargs)

    def active(self):
        invocation=self.executor.calls[-1]
        with self.jobs._db() as db:
            row=self.jobs._row(db,invocation['job_id']);data=json.loads(row['data'])
        return invocation,row,data

    def finish_fixture(self,*,dispatch=True):
        invocation,row,data=self.active()
        self.jobs._worker_claim(row['id'],invocation['token'],data['owner_pid'],data['owner_identity'])
        result=run_child(self.jobs,row['id'],invocation['token'],pipeline_module=self.pipeline)
        self.assertEqual(result['provenance'],'analytic_fixture')
        self.jobs._worker_complete(row['id'],invocation['token'],result['status'],
                                   safe_to_release=result['safe_to_release'],error=result.get('reason'))
        if dispatch:self.jobs._dispatch()
        return self.jobs.get(row['id'])

    def ledger(self,limit=35,used=0,grant=None,job=None):
        folder=self.project/'.local/meshy';folder.mkdir(parents=True,mode=0o700)
        path=folder/'ledger.sqlite3'
        db=sqlite3.connect(path)
        db.executescript('''CREATE TABLE budget(id INTEGER,credit_limit INTEGER);
            CREATE TABLE jobs(id TEXT,output TEXT,plan_hash TEXT,credit_limit INTEGER);
            CREATE TABLE calls(job TEXT,credits INTEGER,state TEXT,task_id TEXT);
            CREATE TABLE budget_grants(job TEXT,credits INTEGER,plan_hash TEXT);''')
        db.execute('INSERT INTO budget VALUES(1,?)',(limit,))
        if used:db.execute('INSERT INTO calls VALUES(?,?,?,?)',('previous',used,'SUCCEEDED','fixture-task'))
        if grant:
            folder=self.jobs.resolve_job(job)/'generation';plan=json.loads((folder/'plan.json').read_text())
            db.execute('INSERT INTO budget_grants VALUES(?,?,?)',(plan['id'],grant,_digest(plan)))
        db.commit();db.close();path.chmod(0o600)
        return path

    def test_create_is_private_immutable_idempotent_and_has_no_pipeline_work(self):
        first=self.create(idempotency_key='request-1');second=self.create(idempotency_key='request-1')
        self.assertEqual(first['id'],second['id']);self.assertFalse(self.executor.calls)
        self.assertEqual(first['request']['custom_actions'],['Dance then bow'])
        self.assertFalse(self.jobs.resolve_job(first['id']).exists())
        self.assertEqual(os.stat(self.jobs.state_dir).st_mode&0o777,0o700)
        self.assertEqual(os.stat(self.jobs.db_path).st_mode&0o777,0o600)
        self.assertNotIn(str(self.root),json.dumps(first))
        with self.assertRaisesRegex(ValueError,'different immutable request'):
            self.jobs.create('Other subject',idempotency_key='request-1')
        with self.assertRaisesRegex(ValueError,'different canonical studio registry'):
            StudioJobs(self.root,self.root/'.local/second',executor=self.executor)

    def test_invalid_inputs_bounds_paths_and_unknown_ids_fail_without_launch(self):
        for fields in [{'prompt':''},{'prompt':'x'*4001},{'style':'x'*801},{'preset':'unlimited'},
                       {'custom_actions':['x'*601]},{'custom_actions':['a']*9},{'custom_actions':[True]},
                       {'custom_actions':['same','same']},{'idempotency_key':'../escape'}]:
            with self.subTest(fields=fields),self.assertRaises(ValueError):
                self.jobs.create(**{'prompt':'Object',**fields})
        self.jobs.create('Object',custom_actions=['a'*600])
        outside=Path(self.temp.name).resolve()/'private.png';outside.write_bytes(b'not selected')
        with self.assertRaisesRegex(ValueError,'approved tool'):
            self.jobs.create('Object',reference=outside)
        with self.assertRaisesRegex(ValueError,'Invalid studio job ID'):self.jobs.get('../bad')
        with self.assertRaisesRegex(ValueError,'Unknown studio job'):self.jobs.get('a'*24)
        for limit,offset in [(0,0),(51,0),(True,0),(10,-1),(10,257)]:
            with self.assertRaises(ValueError):self.jobs.list(limit,offset)
        self.assertFalse(self.executor.calls)

    def test_detached_plan_then_execute_survives_manager_restart(self):
        job=self.create();self.jobs.plan(job['id'])
        self.assertEqual(len(self.executor.calls),1)
        self.assertEqual(self.jobs.get(job['id'])['status'],'launching')
        self.jobs=StudioJobs(self.root,executor=self.executor)
        planned=self.finish_fixture()
        self.assertEqual(planned['status'],'planned');self.assertTrue(planned['capabilities']['start'])
        self.assertEqual(self.pipeline.calls[0]['execute'],False)
        self.assertTrue(self.pipeline.calls[0]['plan_only'])
        self.jobs.plan(job['id']);self.assertEqual(len(self.executor.calls),1)
        self.jobs.start(job['id'],planned['plan_id'])
        complete=self.finish_fixture()
        self.assertEqual(complete['status'],'complete');self.assertEqual(complete['provenance'],'analytic_fixture')
        self.assertFalse(self.pipeline.calls[1]['plan_only']);self.assertTrue(self.pipeline.calls[1]['execute'])
        self.assertEqual(len(self.pipeline.prepared),1)
        self.assertEqual(complete['result']['videos'][0]['id'],'dance')
        self.assertTrue(all(d['path'].startswith('outputs/studio-') for d in complete['result']['downloads']))
        self.assertFalse(complete['result']['videos'][0]['loop_verified'])
        self.jobs.start(job['id'],planned['plan_id']);self.assertEqual(len(self.executor.calls),2)

    def test_fresh_planning_stops_and_start_never_creates_canonical_allowance(self):
        job=self.create(reuse=False);self.jobs.plan(job['id']);planned=self.finish_fixture()
        self.assertEqual(planned['status'],'awaiting_authorization')
        self.assertEqual(planned['plan']['estimate']['credits'],35)
        self.assertFalse(planned['plan']['authorization']['can_start'])
        self.assertEqual(self.pipeline.prepared[0][1]['max_credits'],35)
        self.assertFalse((self.project/'.local/meshy/ledger.sqlite3').exists())
        with self.assertRaisesRegex(ValueError,'canonical Meshy allowance'):self.jobs.start(job['id'],planned['plan_id'])
        ledger=self.ledger(limit=35)
        before=hashlib.sha256(ledger.read_bytes()).hexdigest()
        self.jobs.start(job['id'],planned['plan_id']);complete=self.finish_fixture()
        self.assertEqual(complete['status'],'complete')
        self.assertEqual(hashlib.sha256(ledger.read_bytes()).hexdigest(),before)

    def test_allowance_counts_prior_reservations_and_exact_scoped_grant(self):
        job=self.create(reuse=False);self.jobs.plan(job['id']);planned=self.finish_fixture()
        ledger=self.ledger(limit=40,used=32)
        with self.assertRaisesRegex(ValueError,'8 base credits remain'):self.jobs.start(job['id'],planned['plan_id'])
        folder=self.jobs.resolve_job(job['id'])/'generation';plan=json.loads((folder/'plan.json').read_text())
        db=sqlite3.connect(ledger)
        db.execute('INSERT INTO budget_grants VALUES(?,?,?)',(plan['id'],35,_digest(plan)));db.commit();db.close()
        self.assertTrue(self.jobs.get(job['id'])['plan']['authorization']['can_start'])
        db=sqlite3.connect(ledger);db.execute('UPDATE budget_grants SET plan_hash=?',('wrong',));db.commit();db.close()
        self.assertFalse(self.jobs.get(job['id'])['plan']['authorization']['can_start'])

    def test_uncertain_provider_stage_is_not_replayed(self):
        job=self.create(reuse=False);self.jobs.plan(job['id']);planned=self.finish_fixture();ledger=self.ledger(100)
        plan=json.loads((self.jobs.resolve_job(job['id'])/'generation/plan.json').read_text())
        db=sqlite3.connect(ledger);db.execute('INSERT INTO calls VALUES(?,?,?,?)',(plan['id'],25,'uncertain',None));db.commit();db.close()
        with self.assertRaisesRegex(ValueError,'uncertain charge'):self.jobs.start(job['id'],planned['plan_id'])
        self.assertEqual(len(self.executor.calls),1)

    def test_plan_id_and_worker_time_allowance_are_rechecked(self):
        job=self.create(reuse=False);self.jobs.plan(job['id']);planned=self.finish_fixture();ledger=self.ledger(35)
        with self.assertRaisesRegex(ValueError,'Stale or invalid'):self.jobs.start(job['id'],'a'*64)
        self.jobs.start(job['id'],planned['plan_id'])
        db=sqlite3.connect(ledger);db.execute('INSERT INTO calls VALUES(?,?,?,?)',('racing-job',35,'PENDING','known'));db.commit();db.close()
        result=self.finish_fixture()
        self.assertEqual(result['status'],'awaiting_authorization')
        self.assertEqual(len(self.pipeline.calls),1)

    def test_one_active_four_queued_and_cooperative_cancellation(self):
        ids=[self.create()['id'] for _ in range(6)]
        for job in ids[:5]:self.jobs.plan(job)
        self.assertEqual(len(self.executor.calls),1)
        with self.assertRaisesRegex(ValueError,'queue is full'):self.jobs.plan(ids[5])
        self.assertEqual(self.jobs.cancel_cooperative(ids[1])['status'],'cancelled')
        running=self.jobs.cancel_cooperative(ids[0]);self.assertEqual(running['status'],'cancel_requested')
        result=self.finish_fixture()
        self.assertEqual(result['status'],'cancelled');self.assertFalse(self.pipeline.calls)
        self.assertEqual(len(self.executor.calls),2)
        self.assertEqual(self.executor.calls[-1]['job_id'],ids[2])

    def test_owner_identity_and_lost_heartbeat_never_duplicate_work(self):
        job=self.create();self.jobs.plan(job['id']);invocation,row,data=self.active()
        with self.assertRaisesRegex(ValueError,'PID/start identity'):
            self.jobs._worker_claim(row['id'],invocation['token'],data['owner_pid']+1,data['owner_identity'])
        with self.jobs._db(True) as db:
            data['heartbeat']=time.time()-60;self.jobs._update(db,row,data)
        self.assertEqual(self.jobs.get(job['id'])['status'],'needs_inspection')
        self.jobs=StudioJobs(self.root,executor=self.executor)
        self.jobs.plan(job['id']);self.assertEqual(len(self.executor.calls),1)
        other=self.create();self.jobs.plan(other['id']);self.assertEqual(len(self.executor.calls),1)

    def test_uncertain_launch_keeps_lease_and_reports_no_fake_completion(self):
        self.executor.launch=lambda _:(_ for _ in ()).throw(OSError('fixture uncertain spawn'))
        job=self.create();snapshot=self.jobs.plan(job['id'])
        self.assertEqual(snapshot['status'],'needs_inspection');self.assertIsNone(snapshot['result'])
        with self.assertRaisesRegex(ValueError,'cannot be automatically replayed'):self.jobs.plan(job['id'])
        next_job=self.create();self.assertEqual(self.jobs.plan(next_job['id'])['status'],'queued')

    def test_input_change_and_request_tamper_fail_before_fixture_pipeline(self):
        job=self.create();self.jobs.plan(job['id']);invocation,row,data=self.active()
        self.jobs._worker_claim(row['id'],invocation['token'],data['owner_pid'],data['owner_identity'])
        self.source.write_bytes(b'changed selected input')
        with self.assertRaisesRegex(ValueError,'Selected input changed'):
            run_child(self.jobs,row['id'],invocation['token'],pipeline_module=self.pipeline)
        self.assertFalse(self.pipeline.prepared)
        (self.jobs.jobs_dir/job['id']/'request.json').write_text('{}')
        with self.assertRaisesRegex(ValueError,'request receipt changed'):
            run_child(self.jobs,row['id'],invocation['token'],pipeline_module=self.pipeline)

    def test_prepared_reuse_survives_original_removal(self):
        job=self.create();self.jobs.plan(job['id']);planned=self.finish_fixture()
        binding=self.jobs.jobs_dir/job['id']/'pipeline-request.json';original=binding.read_bytes()
        self.source.unlink();self.jobs.start(job['id'],planned['plan_id'])
        self.assertEqual(self.finish_fixture()['status'],'complete')
        self.assertEqual(binding.read_bytes(),original)

    def test_precreated_self_consistent_pipeline_cannot_replace_studio_prompt(self):
        job=self.create();folder=self.jobs.resolve_job(job['id'])
        self.pipeline.prepare('A different object',folder,style='',custom_actions=['Dance then bow'],
                              preset='preview',reference=None,reuse_model=str(self.source),max_credits=0)
        self.jobs.plan(job['id']);invocation,row,data=self.active()
        self.jobs._worker_claim(row['id'],invocation['token'],data['owner_pid'],data['owner_identity'])
        with patch('avatar_assets.meshy.run') as provider,self.assertRaisesRegex(ValueError,'immutable Studio request: prompt'):
            run_child(self.jobs,row['id'],invocation['token'],pipeline_module=self.pipeline)
        provider.assert_not_called();self.assertFalse(self.pipeline.calls)
        self.assertFalse((self.jobs.jobs_dir/job['id']/'pipeline-request.json').exists())

    def test_prepared_fields_and_selections_are_bound_before_pipeline_execution(self):
        reference=self.root/'references/fixture.png';reference.parent.mkdir()
        reference.write_bytes(b'ANALYTIC SELECTED IMAGE FIXTURE')
        job=self.create(reference=reference);self.jobs.plan(job['id']);planned=self.finish_fixture()
        self.jobs.start(job['id'],planned['plan_id']);invocation,row,data=self.active()
        self.jobs._worker_claim(row['id'],invocation['token'],data['owner_pid'],data['owner_identity'])
        folder=self.jobs.resolve_job(job['id']);original=json.loads((folder/'request.json').read_text())
        cases=[('schema_version',2),('id','another-job'),('prompt','Another prompt'),('style','Another style'),
               ('custom_actions',['Fly away']),('states',['different']),('preset','muse'),
               ('media_spec',{**original['media_spec'],'seconds':30}),('max_credits',35),
               ('max_repairs',2),('max_repairs',True),('max_agent_calls',12),('agent_timeout',600),
               ('agent_model','another-model'),('reference',None),('reuse_model',None)]
        for field in ('reference','reuse_model'):
            for key,value in [('original_path',str(self.source.parent/'another.glb')),
                              ('path','inputs/another.glb'),('bytes',1)]:
                cases.append((field,{**original[field],key:value}))
        for field,value in cases:
            with self.subTest(field=field,value=value):
                changed=deepcopy(original);changed[field]=value
                write(folder/'request.json',changed)
                write(folder/'input-receipt.json',{'request_sha256':hashlib.sha256((folder/'request.json').read_bytes()).hexdigest()})
                with self.assertRaisesRegex(ValueError,'Prepared pipeline'):
                    run_child(self.jobs,row['id'],invocation['token'],pipeline_module=self.pipeline)
                self.assertEqual(len(self.pipeline.calls),1)

    def test_prepared_request_exact_bytes_remain_pinned_after_plan(self):
        job=self.create();self.jobs.plan(job['id']);planned=self.finish_fixture()
        self.jobs.start(job['id'],planned['plan_id']);invocation,row,data=self.active()
        self.jobs._worker_claim(row['id'],invocation['token'],data['owner_pid'],data['owner_identity'])
        folder=self.jobs.resolve_job(job['id']);request=folder/'request.json'
        request.write_text(request.read_text()+'\n')
        write(folder/'input-receipt.json',{'request_sha256':hashlib.sha256(request.read_bytes()).hexdigest()})
        with self.assertRaisesRegex(ValueError,'changed after its Studio binding'):
            run_child(self.jobs,row['id'],invocation['token'],pipeline_module=self.pipeline)
        self.assertEqual(len(self.pipeline.calls),1)

    def test_storage_reservation_blocks_without_request_thread_tree_scans(self):
        first=self.create();second=self.create()
        with self.jobs._db(True) as db:
            row=self.jobs._row(db,first['id']);data=json.loads(row['data'])
            data['retained_bytes']=STORED_BYTES_LIMIT-ACTIVE_RESERVATION_BYTES+1
            self.jobs._update(db,row,data)
        snapshot=self.jobs.plan(second['id'])
        self.assertEqual(snapshot['status'],'queued');self.assertIn('capacity',snapshot['error']['message'])
        self.assertFalse(self.executor.calls)
        self.assertFalse(self.jobs.resolve_job(second['id']).exists())

    def test_snapshots_redact_secrets_and_expose_only_available_selected_artifacts(self):
        job=self.create();self.jobs.plan(job['id']);planned=self.finish_fixture()
        folder=self.jobs.resolve_job(job['id'])
        write(folder/'events/0001.json',{'stage':'build','status':'running',
            'reason':'msy_secret_fixture_key https://signed.example/model?secret=abc /Users/example/.local/credentials/key'})
        serialized=json.dumps(self.jobs.get(job['id']))
        self.assertNotIn('msy_secret',serialized);self.assertNotIn('signed.example',serialized);self.assertNotIn('/Users/example',serialized)
        self.assertNotIn('percent',serialized)
        self.jobs.start(job['id'],planned['plan_id']);complete=self.finish_fixture()
        delivery=json.loads((folder/'delivery.json').read_text());delivery['model']='../../private.glb'
        write(folder/'delivery.json',delivery)
        self.assertNotIn('model_path',self.jobs.get(job['id'])['result'])
        self.assertEqual(complete['provenance'],'analytic_fixture')

    def test_completed_delivery_takes_precedence_over_late_cancel(self):
        job=self.create();self.jobs.plan(job['id']);planned=self.finish_fixture()
        self.jobs.start(job['id'],planned['plan_id'])
        invocation,row,data=self.active()
        self.jobs._worker_claim(row['id'],invocation['token'],data['owner_pid'],data['owner_identity'])
        result=run_child(self.jobs,row['id'],invocation['token'],pipeline_module=self.pipeline)
        self.jobs.cancel_cooperative(job['id'])
        resumed=run_child(self.jobs,row['id'],invocation['token'],pipeline_module=self.pipeline)
        self.assertEqual(resumed['status'],'complete')
        self.jobs._worker_complete(job['id'],invocation['token'],result['status'],safe_to_release=True)
        self.assertEqual(self.jobs.cancel_cooperative(job['id'])['status'],'complete')

    def test_repeated_cancel_is_idempotent_and_safe_completion_releases_queue(self):
        job=self.create();self.jobs.plan(job['id'])
        queued=self.create();self.jobs.plan(queued['id'])
        self.assertEqual(self.jobs.cancel_cooperative(job['id'])['status'],'cancel_requested')
        invocation,_,_=self.active()
        receipt=Path(invocation['invocation_dir'])/'cancel.json';original=receipt.read_bytes()
        with self.jobs._db() as db:
            before=db.execute('SELECT COUNT(*) FROM journal WHERE job=?',(job['id'],)).fetchone()[0]
        for _ in range(150):
            self.assertEqual(self.jobs.cancel_cooperative(job['id'])['status'],'cancel_requested')
        self.assertEqual(receipt.read_bytes(),original)
        with self.jobs._db() as db:
            after=db.execute('SELECT COUNT(*) FROM journal WHERE job=?',(job['id'],)).fetchone()[0]
        self.assertEqual(before,after)
        self.assertEqual(self.finish_fixture()['status'],'cancelled')
        self.assertEqual(len(self.executor.calls),2)
        self.assertEqual(self.executor.calls[-1]['job_id'],queued['id'])

    def test_supervisor_seals_bounded_fixture_receipts_and_storage_accounting(self):
        job=self.create();self.jobs.plan(job['id']);invocation,row,data=self.active()
        def runner(command,log,output,**limits):
            self.assertEqual(limits,{'timeout':5400,'memory_mb':1024,'output_mb':64})
            self.assertIn('--child',command)
            result=run_child(self.jobs,job['id'],invocation['token'],pipeline_module=self.pipeline)
            _write_private(output/'result.json',result)
            return {'exit_code':0,'failure':None,'process_exit_verified':True}
        with patch('avatar_assets.studio_worker.os.getpid',return_value=data['owner_pid']),\
             patch('avatar_assets.studio_worker.process_identity',return_value=data['owner_identity']),\
             patch('avatar_assets.runner.run_bounded',side_effect=runner):
            supervise(self.jobs,job['id'],invocation['token'])
        value=self.jobs.get(job['id'])
        self.assertEqual(value['status'],'planned')
        self.assertGreater(value['storage']['retained_bytes_measured'],value['storage']['pipeline_bytes_measured'])
        path=Path(invocation['invocation_dir'])
        self.assertTrue((path/'worker-job.json').is_file())
        self.assertTrue(json.loads((path/'supervisor-result.json').read_text())['safe_to_release'])

    def test_unsealed_worker_exit_preserves_uncertain_owner_and_blocks_dispatch(self):
        job=self.create();self.jobs.plan(job['id']);invocation,row,data=self.active()
        next_job=self.create();self.jobs.plan(next_job['id'])
        with patch('avatar_assets.studio_worker.os.getpid',return_value=data['owner_pid']),\
             patch('avatar_assets.studio_worker.process_identity',return_value=data['owner_identity']),\
             patch('avatar_assets.runner.run_bounded',return_value={'exit_code':-15,'failure':'fixture stop'}):
            supervise(self.jobs,job['id'],invocation['token'])
        self.assertEqual(self.jobs.get(job['id'])['status'],'needs_inspection')
        self.assertEqual(self.jobs.get(next_job['id'])['status'],'queued')
        self.assertEqual(len(self.executor.calls),1)

    def test_source_snapshot_tamper_blocks_child_before_work(self):
        job=self.create();self.jobs.plan(job['id']);invocation,row,data=self.active()
        self.jobs._worker_claim(row['id'],invocation['token'],data['owner_pid'],data['owner_identity'])
        (Path(invocation['invocation_dir'])/'sources/studio_worker.py').write_text('changed fixture')
        with self.assertRaisesRegex(ValueError,'worker backend changed'):
            run_child(self.jobs,job['id'],invocation['token'],pipeline_module=self.pipeline)
        self.assertFalse(self.pipeline.prepared)

    def test_new_attempt_media_discovery_uses_pinned_delivery_pack(self):
        job=self.create();self.jobs.plan(job['id']);planned=self.finish_fixture()
        self.jobs.start(job['id'],planned['plan_id']);self.finish_fixture()
        folder=self.jobs.resolve_job(job['id'])
        (folder/'media-portable').rename(folder/'media-portable-1')
        delivery=json.loads((folder/'delivery.json').read_text())
        delivery['packs']=[{'path':'media-portable-1'}];delivery['selected_attempt']=1
        write(folder/'delivery.json',delivery)
        snapshot=self.jobs.get(job['id'])
        self.assertIn('/media-portable-1/',snapshot['result']['portrait_path'])
        self.assertIn('/media-portable-1/',snapshot['result']['videos'][0]['path'])
        self.assertTrue(snapshot['result']['portrait_path'].endswith('/portrait.webp'))
        (folder/'media-portable-1/portrait.png').write_bytes(b'ANALYTIC LEGACY PORTRAIT FIXTURE')
        self.assertEqual(self.jobs.get(job['id'])['result']['portrait_path'],snapshot['result']['portrait_path'])
        from avatar_assets.studio import _asset_id, _wire
        wire=_wire(snapshot)
        self.assertEqual(wire['result']['portrait_url'],
                         '/api/studio/jobs/'+job['id']+'/files/'+_asset_id(job['id'],snapshot['result']['portrait_path']))
        self.assertNotIn('portrait_path',wire['result'])

    def test_legacy_completed_portrait_png_remains_available(self):
        job=self.create();self.jobs.plan(job['id']);planned=self.finish_fixture()
        self.jobs.start(job['id'],planned['plan_id']);self.finish_fixture()
        folder=self.jobs.resolve_job(job['id'])
        (folder/'media-portable/portrait.webp').rename(folder/'media-portable/portrait.png')
        self.assertTrue(self.jobs.get(job['id'])['result']['portrait_path'].endswith('/portrait.png'))

    def test_restart_dispatches_only_previously_queued_work_without_active_owner(self):
        first=self.create();second=self.create();self.jobs.plan(first['id']);self.jobs.plan(second['id'])
        self.finish_fixture(dispatch=False)
        self.assertEqual(len(self.executor.calls),1)
        self.jobs=StudioJobs(self.root,executor=self.executor)
        self.assertEqual(len(self.executor.calls),2)
        self.assertEqual(self.executor.calls[-1]['job_id'],second['id'])

    def test_macos_start_identity_uses_precise_owned_sdk_fields_and_fails_closed(self):
        payload=bytearray(136)
        payload[12:16]=(1234).to_bytes(4,'little')
        payload[20:24]=os.getuid().to_bytes(4,'little')
        payload[120:128]=(1700000000).to_bytes(8,'little')
        payload[128:136]=(123456).to_bytes(8,'little')
        def info(pid,flavor,arg,buffer,size):
            self.assertEqual((pid,flavor,arg,size),(1234,3,0,136))
            ctypes.memmove(buffer,bytes(payload),size)
            return size
        library=SimpleNamespace(proc_pidinfo=Mock(side_effect=info))
        with patch('avatar_assets.studio_jobs.sys.platform','darwin'),\
             patch('avatar_assets.studio_jobs.ctypes.CDLL',return_value=library):
            self.assertEqual(process_identity(1234),'darwin-start:1700000000:123456')
            payload[12:16]=(4321).to_bytes(4,'little')
            with self.assertRaisesRegex(ValueError,'not owned'):process_identity(1234)
            library.proc_pidinfo.side_effect=None;library.proc_pidinfo.return_value=0
            with self.assertRaises(OSError):process_identity(1234)


if __name__=='__main__':unittest.main()
