"""Analytic continuation custody; provider and native execution are forbidden fixtures."""
from copy import deepcopy
import fcntl
import json
from pathlib import Path
import shutil
import tempfile
import time
import unittest
from unittest.mock import Mock, patch

from avatar_assets import pipeline, pipeline_continuation as cont, openai_runner as api
from avatar_assets.agent_backend import DEFAULT, pipeline_limits
from avatar_assets.pipeline_prompts import finish_config
from avatar_assets.quality_policy import build_quality_policy
from avatar_assets.safety import artifacts, read_json, sha256, write_json
from test_pipeline_prompts import request as fixture_request, plan as fixture_plan, author_response
from test_native_failure_diagnostics import native_functions

APPROVAL={'decision':'approved','user_reply':'Analytic test grant, not real authorization'}


def schema(value):
    if isinstance(value,dict):
        return {'type':'object','properties':{k:schema(v) for k,v in value.items()},'required':list(value),'additionalProperties':False}
    if isinstance(value,list):return {'type':'array','items':schema(value[0]) if value else {'type':'string'}}
    return {'type': 'null' if value is None else 'string' if isinstance(value,str) else 'number'}


class ContinuationTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name).resolve()
        sources=pipeline._pipeline_sources()
        for name,path in sources.items():
            target=self.root/('scripts' if name.startswith('scripts/') else 'avatar_assets')/Path(name).name
            target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(path,target)
        self.addCleanup(patch.stopall)
        patch.object(pipeline,'ROOT',self.root).start()
        self.geometry={'status':'available','failures':[],'fixture':'analytic geometry, not measured media'}
        patch('avatar_assets.author_geometry.author_diagnostics',return_value=self.geometry).start()
        patch('avatar_assets.provider_model.preflight',return_value={}).start()
        self.network=patch.object(api,'post_response').start()
        patch.object(api,'read_api_key',return_value='analytic-not-real-key').start()
        self.ledger=api.OpenAILedger(self.root/'.local/openai');self.addCleanup(self.ledger.close)
        self.ledger.configure(100_000_000,APPROVAL)
        self.backend={**DEFAULT,'max_calls':3,'max_repairs':0}
        self.original=self.root/'fixture.glb';self.original.write_bytes(b'Analytic non-renderable source')
        self.parent=self.root/'outputs/parent'
        self.prepare(self.parent)
        self.plan=fixture_plan();self.plan['surface_intent']=[{'region':'body','material':'metal','finish':'Copper'}]
        self.answer=author_response()
        self.make_terminal(self.parent)

    def prepare(self,out,continuation=None,**changes):
        request=fixture_request()
        values={'style':request['style'],'states':request['states'],'preset':request['preset'],
                'reuse_model':self.original,'agent_backend':self.backend,**pipeline_limits(self.backend),
                'continuation_from':continuation,**changes}
        return pipeline.prepare(request['prompt'],out,**values)

    def make_terminal(self,out,inherited=None,granted=False):
        req=read_json(out/'request.json')
        if not granted:self.ledger.authorize_job(out.name,sha256(out/'request.json'),req['max_agent_calls'],30_000_000,APPROVAL)
        runner=api.OpenAIRunner(out/'agents',job_id=out.name,job_digest=sha256(out/'request.json'),ledger=self.ledger,max_calls=req['max_agent_calls'])
        for role,answer in ([('planner',self.plan)] if inherited is None else [])+[('author-0',self.answer)]:
            response={'id':'resp_'+out.name+role,'model':api.MODEL,'service_tier':'default','status':'completed',
                      'usage':{'input_tokens':100,'output_tokens':20,'total_tokens':120},
                      'output':[{'type':'message','role':'assistant','status':'completed','content':[{'type':'output_text','text':json.dumps(answer)}]}]}
            self.network.return_value=json.dumps(response).encode()
            runner.call(role,'Analytic fixture',schema(answer))
        (out/'pipeline.lock').touch()
        if (out/'plan.json').exists():self.assertEqual(read_json(out/'plan.json'),self.plan)
        else:write_json(out/'plan.json',self.plan)
        digest=req['reuse_model']['sha256']
        native={'source_sha256':digest,'source_unchanged':True,'bounds_blender_z_up':{'min':[-1,-1,0],'max':[1,1,2]},'rigs':[]}
        context={'native':native,'preflight':{'sha256':digest,'source_unchanged':True}}
        quality=build_quality_policy(req,self.plan,context);write_json(out/'quality-policy.json',quality)
        (out/'inspection').mkdir();write_json(out/'inspection/manifest.json',{'source_glb_sha256':digest,**context})
        recipe=finish_config(self.answer,req,self.plan,{'path':str(out/req['reuse_model']['path']),'sha256':digest},0,quality_policy=quality,bone_parents={})
        write_json(out/'finish-0.json',recipe)
        pack=out/'model-0';(pack/'sources').mkdir(parents=True)
        for name in ('blender_finish_generated.py','blender_fur_undercoat.py'):
            shutil.copyfile(self.root/'scripts'/name,pack/'sources'/name)
        write_json(pack/'sources/finish-design.json',recipe);write_json(pack/'finish-config.json',recipe)
        scope=native_functions(pack/'sources/blender_finish_generated.py',Mock(side_effect=ValueError('Analytic native material rejection')))
        with self.assertRaises(ValueError):scope['finish'](pack/'finish-config.json',pack)
        write_json(pack/'job.json',{'exit_code':1,'process_exit_verified':True,'manual_stop_required':False})
        write_json(pack/'manifest.json',{'status':'failed','artifacts':artifacts(pack)})
        write_json(out/'stage-failure-model-0.json',{'manifest_sha256':sha256(pack/'manifest.json'),'error':'Finishing failed; inspect preserved attempt'})
        write_json(out/'author-geometry-0.json',self.geometry)
        findings={'validation_failure':'Finishing failed; inspect preserved attempt',
                  'measurements':{**pipeline.Pipeline.__new__(pipeline.Pipeline)._diagnostics(pack),'author_geometry':self.geometry}}
        write_json(out/'unsupported.json',{'status':'unsupported','stage':'repair_budget','reason':'Budget ended',
                                           'request_sha256':sha256(out/'request.json'),'findings':findings})
        (out/'status.json').write_text(json.dumps({'status':'unsupported','stage':'repair_budget'}))

    def author_rejection(self,out):
        req=read_json(out/'request.json')
        answer={'schema_version':2,'status':'unsupported','reason':'Analytic false rig-only scope claim','design':None}
        response={'id':'resp_'+out.name+'author-1','model':api.MODEL,'service_tier':'default','status':'completed',
                  'usage':{'input_tokens':100,'output_tokens':20,'total_tokens':120},
                  'output':[{'type':'message','role':'assistant','status':'completed','content':[{'type':'output_text','text':json.dumps(answer)}]}]}
        self.network.return_value=json.dumps(response).encode()
        runner=api.OpenAIRunner(out/'agents',job_id=out.name,job_digest=sha256(out/'request.json'),ledger=self.ledger,max_calls=req['max_agent_calls'])
        runner.call('author-1','Analytic refusal after measured native failure',schema(answer))
        terminal={'status':'unsupported','stage':'author','attempt':1,'reason':answer['reason'],
                  'request_sha256':sha256(out/'request.json')}
        (out/'unsupported.json').write_text(json.dumps(terminal));(out/'status.json').write_text(json.dumps(terminal))
        return answer

    def rejected_child(self):
        self.backend={**self.backend,'max_calls':5,'max_repairs':1}
        child=self.root/'outputs/refusal';self.prepare(child,self.parent)
        self.make_terminal(child,inherited=True)
        return child,self.author_rejection(child)

    def test_paid_author_refusal_retains_valid_seed_and_native_failure_without_planner(self):
        child,rejection=self.rejected_child();before=cont._inventory(child)
        captured=cont.collect(self.root,child)
        self.assertEqual(captured['depth'],2)
        self.assertEqual(captured['context']['previous_answer'],self.answer)
        findings=captured['context']['findings']
        self.assertEqual(findings['author_rejection']['answer'],rejection)
        self.assertEqual(findings['author_rejection']['receipt_sha256'],sha256(child/'agents/author-1/receipt.json'))
        self.assertIn('native_failure',findings['measurements'])
        grandchild=self.root/'outputs/refusal-successor';self.prepare(grandchild,child)
        agents=Mock();agents.call.side_effect=AssertionError('Planner forbidden')
        worker=pipeline.Pipeline(grandchild,agents=agents)
        self.assertEqual(worker._run(False,plan_only=True)['status'],'planned');agents.call.assert_not_called()
        self.assertEqual(worker.continuation['previous_answer'],self.answer)
        self.assertEqual(cont._inventory(child),before)

    def test_author_refusal_reason_attempt_and_receipt_are_authenticated(self):
        child,_=self.rejected_child()
        for field,value in (('reason','Changed refusal'),('attempt',0),('attempt',2)):
            originals={name:(child/name).read_bytes() for name in ('unsupported.json','status.json')}
            for name in originals:
                terminal=read_json(child/name);terminal[field]=value;(child/name).write_text(json.dumps(terminal))
            with self.subTest(field=field,value=value),self.assertRaises(ValueError):cont.collect(self.root,child)
            for name,original in originals.items():(child/name).write_bytes(original)
        path=child/'agents/author-1/answer.json';path.write_text(path.read_text()+' ')
        with self.assertRaisesRegex(ValueError,'receipt changed'):cont.collect(self.root,child)

    def retire(self,parent,child):
        grant=tuple(self.ledger.db.execute('SELECT * FROM jobs WHERE id=?',(parent.name,)).fetchone())
        calls=[tuple(row) for row in self.ledger.db.execute('SELECT * FROM calls WHERE job=? ORDER BY role',(parent.name,))]
        approval={**APPROVAL,'operation':'retire_and_continue','parent_job_id':parent.name,
                  'successor_job_id':child.name,'additional_calls':3}
        self.ledger.continue_job(parent.name,sha256(parent/'request.json'),child.name,
                                 sha256(child/'request.json'),3,30_000_000,approval)
        self.assertEqual(tuple(self.ledger.db.execute('SELECT * FROM jobs WHERE id=?',(parent.name,)).fetchone()),grant)
        self.assertEqual([tuple(row) for row in self.ledger.db.execute('SELECT * FROM calls WHERE job=? ORDER BY role',(parent.name,))],calls)

    def test_child_preserves_context_and_never_replans_or_resets_parent(self):
        before=cont._inventory(self.parent);parent_calls=self.ledger.db.execute('SELECT COUNT(*) FROM calls WHERE job=?',(self.parent.name,)).fetchone()[0]
        child=self.root/'outputs/child';result=self.prepare(child,self.parent)
        self.assertEqual(result['estimate']['new_planner_calls'],0)
        self.assertEqual(result['estimate']['expected_agent_calls'],2)
        agents=Mock();agents.call.side_effect=AssertionError('Planner call forbidden')
        worker=pipeline.Pipeline(child,agents=agents)
        self.assertEqual(worker._plan(),self.plan);agents.call.assert_not_called()
        self.assertEqual(worker.continuation['previous_answer'],self.answer)
        self.assertIn('native_failure',worker.continuation['findings']['measurements'])
        self.assertEqual(cont._inventory(self.parent),before)
        self.assertEqual(self.ledger.db.execute('SELECT COUNT(*) FROM calls WHERE job=?',(self.parent.name,)).fetchone()[0],parent_calls)
        self.assertIsNone(self.ledger.db.execute('SELECT * FROM jobs WHERE id="child"').fetchone())

    def test_first_author_receives_parent_answer_and_exact_failure(self):
        child=self.root/'outputs/author-child';self.prepare(child,self.parent)
        agents=Mock();agents.call.side_effect=RuntimeError('Stop before any provider')
        worker=pipeline.Pipeline(child,agents=agents)
        worker._pack=Mock(return_value=read_json(self.parent/'inspection/manifest.json'))
        with self.assertRaisesRegex(RuntimeError,'Stop before'):
            worker._run(False)
        self.assertEqual(agents.call.call_count,1)
        role,prompt,_=agents.call.call_args.args
        self.assertEqual(role,'author-0');self.assertIn('Analytic native material rejection',prompt)
        self.assertIn('previous_design',prompt);self.assertIn('Base',prompt)
        self.assertIn('CONTINUATION: reuse the previous authored rig and motion design',prompt)

    def test_mismatched_intent_rejected_before_child_creation(self):
        other=self.root/'other.glb';other.write_bytes(b'Different analytic source')
        for changes in ({'style':'Changed appearance'},{'preset':'preview'},{'states':['nod']},{'max_credits':35},
                        {'reuse_model':other}):
            child=self.root/'outputs/mismatch'
            with self.subTest(changes=changes),self.assertRaises(ValueError):self.prepare(child,self.parent,**changes)
            self.assertFalse(child.exists())

    def test_parent_and_child_tampering_stop_without_fallback(self):
        child=self.root/'outputs/tamper';self.prepare(child,self.parent)
        answer=self.parent/'agents/author-0/answer.json';answer.write_text(answer.read_text()+' ')
        agents=Mock()
        with self.assertRaisesRegex(ValueError,'parent artifacts changed'):pipeline.Pipeline(child,agents=agents)
        agents.call.assert_not_called()

    def test_modified_seed_receipt_or_missing_context_stops_without_replan(self):
        child=self.root/'outputs/context';self.prepare(child,self.parent)
        path=child/'continuation/context.json';path.write_text(path.read_text()+' ')
        with self.assertRaises(ValueError):pipeline.Pipeline(child,agents=Mock())
        path.unlink()
        with self.assertRaises((ValueError,OSError)):pipeline.Pipeline(child,agents=Mock())

    def test_uncertain_call_and_live_parent_lock_are_rejected(self):
        handle=(self.parent/'pipeline.lock').open('rb');fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            with self.assertRaisesRegex(ValueError,'live owner'):cont.collect(self.root,self.parent)
        finally:handle.close()
        self.ledger.reserve(self.parent.name,sha256(self.parent/'request.json'),3,'author-1','f'*64,1000)
        with self.assertRaisesRegex(ValueError,'uncertain'):cont.collect(self.root,self.parent)

    def test_inherited_plan_can_continue_again_without_planner_role(self):
        child=self.root/'outputs/chained';self.prepare(child,self.parent)
        self.retire(self.parent,child)
        agents=Mock();agents.call.side_effect=AssertionError('Never replan')
        worker=pipeline.Pipeline(child,agents=agents)
        self.assertEqual(worker._run(False,plan_only=True)['status'],'planned')
        agents.call.assert_not_called()
        self.make_terminal(child,inherited=True,granted=True)
        captured=cont.collect(self.root,child)
        self.assertEqual(captured['depth'],2)
        self.assertEqual(captured['context']['plan'],self.plan)
        self.assertFalse((child/'agents/planner').exists())
        grandchild=self.root/'outputs/grandchild';self.prepare(grandchild,child)
        self.retire(child,grandchild)
        agents=Mock();agents.call.side_effect=AssertionError('Never replan')
        self.assertEqual(pipeline.Pipeline(grandchild,agents=agents)._run(False,plan_only=True)['status'],'planned')
        agents.call.assert_not_called()

    def test_retirement_is_only_valid_for_exact_child_and_known_parent_cost(self):
        child=self.root/'outputs/retired';self.prepare(child,self.parent)
        self.retire(self.parent,child)
        saved=read_json(child/'request.json')
        self.assertEqual(cont.verify(self.root,child,saved)['plan'],self.plan)
        with self.assertRaisesRegex(ValueError,'retired'):cont.collect(self.root,self.parent)
        self.ledger.db.execute('UPDATE jobs SET digest=? WHERE id=?',('e'*64,child.name));self.ledger.db.commit()
        with self.assertRaisesRegex(ValueError,'accounting differs'):cont.verify(self.root,child,saved)
        self.ledger.db.execute('UPDATE jobs SET digest=? WHERE id=?',(sha256(child/'request.json'),child.name));self.ledger.db.commit()
        self.ledger.db.execute('UPDATE retirements SET actual=actual+1 WHERE parent=?',(self.parent.name,));self.ledger.db.commit()
        with self.assertRaisesRegex(ValueError,'accounting differs'):cont.verify(self.root,child,saved)
        self.ledger.db.execute('UPDATE retirements SET actual=actual-1,successor=? WHERE parent=?',('wrong-child',self.parent.name));self.ledger.db.commit()
        with self.assertRaisesRegex(ValueError,'different'):cont.verify(self.root,child,saved)

    def test_nonterminal_and_changed_recipe_are_not_continuable(self):
        status=self.parent/'status.json';original=status.read_bytes();status.write_text('{"status":"running","stage":"build"}')
        with self.assertRaisesRegex(ValueError,'terminal'):cont.collect(self.root,self.parent)
        status.write_bytes(original)
        recipe=self.parent/'finish-0.json';recipe.write_text(recipe.read_text()+' ')
        with self.assertRaisesRegex(ValueError,'recipe differs'):cont.collect(self.root,self.parent)

    def test_parent_inventory_is_bounded_before_reading_oversized_assets(self):
        root=self.root/'oversized';root.mkdir()
        with (root/'large.glb').open('wb') as stream:stream.truncate(64*1024**2+1)
        with self.assertRaisesRegex(ValueError,'64MiB'):cont._inventory(root)
        (root/'large.glb').unlink()
        for index in range(cont.MAX_FILES+1):(root/str(index)).touch()
        with self.assertRaisesRegex(ValueError,'512 files'):cont._inventory(root)


if __name__=='__main__':unittest.main()
