"""Exact pre-author input-bound recovery fixtures; no provider or native execution."""
from contextlib import ExitStack
from copy import deepcopy
import json
from pathlib import Path
import sqlite3
import unittest
from unittest.mock import Mock,patch

from avatar_assets import pipeline,pipeline_recovery as recovery,studio_inspection as inspection
from avatar_assets.safety import artifacts,read_json,sha256
from tests import test_studio_api_release as api_fixture
from tests.test_studio_api_release import ABSENT,write
from tests.test_pipeline_prompts import plan
from tests.test_pipeline_continuation import schema


class AuthorInputRecoveryTests(unittest.TestCase):
    def setUp(self):
        f=api_fixture.StudioApiReleaseTests('test_settled_semantically_invalid_answer_releases_only_as_failed_without_calls_or_reset')
        valid=plan();valid['states']=[{'id':'dance','meaning':'Dance with a visible rhythm',
            'user_request':'Dance with a visible rhythm','roles':[{'id':'body','description':'Articulated character body'}],
            'requirements':[{'id':'move','kind':'articulation','role':'body','description':'Body limbs change angles'},
                            {'id':'beat','kind':'rhythm','role':'body','description':'Body moves with repeated rhythmic changes'}]}]
        valid['surface_intent']=[{'region':'body','material':'fabric','finish':'Plush cloth'}]
        f.fixture_calls=7;f.fixture_repairs=2;f.fixture_answer=valid;f.fixture_schema=schema(valid)
        f.setUp();self.addCleanup(f.doCleanups);self.f=f;self.job=f.output;self.root=f.root;self.plan=valid
        with f.jobs._db(True) as db:
            row=f.jobs._row(db,f.job);data=json.loads(row['data']);data['operation']='execute';f.jobs._update(db,row,data)
        meta=read_json(f.invocation/'invocation.json');meta['operation']='execute';write(f.invocation/'invocation.json',meta)
        write(self.job/'plan.json',valid)
        write(self.job/'quality-policy.json',{'provenance':'analytic fixture'})
        write(self.job/'status.json',{'status':'checkpointed_error','stage':'pipeline','automatic_retry':False,
                                    'error':'Agent input exceeds 160KiB'})
        native={'source_sha256':sha256(self.job/'inputs/source-model.glb'),'source_unchanged':True}
        write(self.job/'inspection/native-inspection.json',native)
        write(self.job/'inspection/job.json',{'pid':70001,'status':'finished','exit_code':0,
            'manual_stop_required':False,'process_exit_verified':True,'failure':None})
        (self.job/'inspection/original.glb').write_bytes((self.job/'inputs/source-model.glb').read_bytes())
        self.reseal_inspection()
        for name in ('pipeline_prompts.py','pipeline_recovery.py'):
            (self.job/'sources'/name).write_text('ANALYTIC OLD '+name)
        receipt=read_json(self.job/'input-receipt.json');receipt['sources']=artifacts(self.job/'sources');write(self.job/'input-receipt.json',receipt)
        self.live={}
        for name in ('openai_runner.py','pipeline_prompts.py','pipeline_recovery.py'):
            path=self.root/'runtime'/name;path.parent.mkdir(exist_ok=True)
            path.write_bytes((self.job/'sources'/name).read_bytes() if name=='openai_runner.py' else ('ANALYTIC NEW '+name).encode())
            self.live[name]=path
        self.stack=ExitStack();self.addCleanup(self.stack.close)
        self.stack.enter_context(patch.object(pipeline,'ROOT',self.root))
        self.stack.enter_context(patch.object(pipeline,'_pipeline_sources',return_value=self.live))
        self.stack.enter_context(patch.object(inspection,'_process_snapshot',return_value=ABSENT))
        self.changes={name:{'before':sha256(self.job/'sources'/name),'after':sha256(path)}
                      for name,path in self.live.items() if name!='openai_runner.py'}
        self.revision=self.root/'outputs/input-bound-revision'

    def reseal_inspection(self):
        write(self.job/'inspection/manifest.json',{'status':'needs_visual_review','kind':'generated_geometry_preview','provider_calls':0,
            'source_glb_sha256':sha256(self.job/'inputs/source-model.glb'),'artifacts':artifacts(self.job/'inspection')})

    def prepare(self):
        return recovery.prepare_revision(self.job,self.revision,expected_request_sha256=sha256(self.job/'request.json'),
            expected_changes=self.changes,reason='Compact only the bounded author prompt view.',profile='author_input_bound')

    def test_release_adopt_and_reuse_paid_plan_without_reset_or_new_calls(self):
        before={str(p):sha256(p) for p in self.job.rglob('*') if p.is_file()};ledger=sha256(self.f.ledger)
        result=self.f.release();self.assertEqual(result['status'],'failed')
        prepared=self.prepare()
        adopted=self.f.jobs.adopt_cached_planner_revision(self.f.job,self.revision,expected_revision_sha256=prepared['sha256'])
        self.assertEqual(adopted['status'],'planned');self.assertEqual(read_json(self.job/'plan.json'),self.plan)
        record=read_json(self.revision/'revision.json');self.assertEqual(record['remaining_role_calls'],6)
        self.assertEqual(record['profile'],'author_input_bound')
        worker=pipeline.Pipeline(self.job,agents=Mock());worker.guard=Mock()
        self.assertEqual(worker._plan(),self.plan);worker.agents.call.assert_not_called()
        self.assertEqual(sha256(self.f.ledger),ledger)
        self.assertTrue(all(sha256(Path(path))==checksum for path,checksum in before.items() if Path(path).name!='status.json'))
        self.assertEqual(sha256(self.revision/'original-failure-status.json'),before[str(self.job/'status.json')])
        self.assertEqual(read_json(self.job/'request.json')['max_repairs'],2)
        self.assertEqual(read_json(self.job/'request.json')['max_agent_calls'],7)

    def test_any_author_folder_or_call_blocks_this_profile(self):
        (self.job/'agents/author-0').mkdir()
        with self.assertRaisesRegex(ValueError,'inventory|Only one API'):self.f.release()
        (self.job/'agents/author-0').rmdir()
        with sqlite3.connect(self.f.ledger) as db:
            db.execute("INSERT INTO calls(job,role,request_digest,reserved,state) VALUES(?,?,?,?,?)",(self.job.name,'author-0','e'*64,5618640,'reserved_uncertain'))
        with self.assertRaisesRegex(ValueError,'API calls are uncertain'):self.f.release()

    def test_wrong_checkpoint_plan_or_unfinished_inspection_cannot_release(self):
        status=read_json(self.job/'status.json');write(self.job/'status.json',{**status,'error':'Another failure'})
        with self.assertRaisesRegex(ValueError,'exact bounded-input'):self.f.release()
        write(self.job/'status.json',status)
        write(self.job/'plan.json',{**self.plan,'subject':'Changed'})
        with self.assertRaisesRegex(ValueError,'unchanged valid paid'):self.f.release()
        write(self.job/'plan.json',self.plan)
        job=read_json(self.job/'inspection/job.json');job['process_exit_verified']=False;write(self.job/'inspection/job.json',job);self.reseal_inspection()
        with self.assertRaisesRegex(ValueError,'Unsealed|successful exit'):self.f.release()

    def test_only_reviewed_prompt_recovery_changes_and_existing_inspection_remain_bound(self):
        self.f.release();self.live['openai_runner.py'].write_text('Changed billing implementation')
        self.changes['openai_runner.py']={'before':sha256(self.job/'sources/openai_runner.py'),'after':sha256(self.live['openai_runner.py'])}
        with self.assertRaisesRegex(ValueError,'exactly match'):self.prepare()
        self.live['openai_runner.py'].write_bytes((self.job/'sources/openai_runner.py').read_bytes());self.changes.pop('openai_runner.py')
        prepared=self.prepare();self.f.jobs.adopt_cached_planner_revision(self.f.job,self.revision,expected_revision_sha256=prepared['sha256'])
        write(self.job/'inspection/native-inspection.json',{'changed':True})
        with self.assertRaisesRegex(ValueError,'completed inspection changed'):recovery.verify_runtime_revision(self.job,self.live)


if __name__=='__main__':unittest.main()
