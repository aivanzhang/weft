"""Analytic continuation custody; provider and native execution are forbidden fixtures."""
from copy import deepcopy
import fcntl
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import time
import unittest
from unittest.mock import Mock, patch

from avatar_assets import pipeline, pipeline_continuation as cont, openai_runner as api
from avatar_assets.agent_backend import DEFAULT, pipeline_limits
from avatar_assets.action_intent import PlanIntentError
from avatar_assets.pipeline_prompts import finish_config
from avatar_assets.quality_policy import build_quality_policy
from avatar_assets.safety import artifacts, read_json, sha256, write_json
from test_pipeline_prompts import request as fixture_request, plan as fixture_plan, author_response, review as fixture_review
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

    def replan_terminal(self):
        self.backend={**self.backend,'max_calls':5,'max_repairs':1}
        out=self.root/'outputs/replan-parent';self.prepare(out);self.make_terminal(out)
        req=read_json(out/'request.json');pack=out/'model-0'
        (pack/'model.glb').write_bytes(b'Analytic built model, not renderable')
        (pack/'job.json').write_text(json.dumps({'exit_code':0,'process_exit_verified':True}))
        (pack/'manifest.json').write_text(json.dumps({'status':'needs_visual_review','source_sha256':req['reuse_model']['sha256'],
            'artifacts':artifacts(pack)}))
        audit=out/'action-audit-0';audit.mkdir();model_hash=sha256(pack/'model.glb')
        contract={'states':self.plan['states'],'action_bindings':[],'height_meters':self.plan['height_meters']}
        write_json(audit/'intent.json',{'schema_version':1,**contract,'source_sha256':model_hash})
        implementations={name:sha256(out/'sources'/name) for name in ('action_audit.py','action_intent.py','three_d.py')}
        report={'model_sha256':model_hash,'passed':False,'source_unchanged':True,'intent_sha256':sha256(audit/'intent.json'),
            'input_contract_sha256':hashlib.sha256(json.dumps(contract,sort_keys=True,ensure_ascii=False,allow_nan=False).encode()).hexdigest(),
            'implementation_sha256':implementations,'actual_bone_parents':{'Base':None,'Tube':'Base'},
            'states':[{'id':s['id'],'requirement_results':{'passed':False},'fixture':'Analytic proof, not measured media'} for s in self.plan['states']]}
        write_json(audit/'report.json',report);write_json(audit/'job.json',{'exit_code':0,'process_exit_verified':True})
        write_json(audit/'manifest.json',{'kind':'weighted_action_audit','status':'measured','model_sha256':model_hash,
            'implementation_sha256':implementations['action_audit.py'],'artifacts':artifacts(audit)})
        bad=json.loads(self.answer['design_json']);bad['authored_actions'][0]['keys'][-1]['second']=3
        answer=author_response(bad)
        response={'id':'resp_replan_author1','model':api.MODEL,'service_tier':'default','status':'completed',
            'usage':{'input_tokens':100,'output_tokens':20,'total_tokens':120},
            'output':[{'type':'message','role':'assistant','status':'completed','content':[{'type':'output_text','text':json.dumps(answer)}]}]}
        self.network.return_value=json.dumps(response).encode()
        api.OpenAIRunner(out/'agents',job_id=out.name,job_digest=sha256(out/'request.json'),ledger=self.ledger,max_calls=5).call('author-1','Analytic invalid repair',schema(answer))
        write_json(out/'author-geometry-1.json',self.geometry)
        failure='Authored actions must span0–4seconds'
        terminal=read_json(out/'unsupported.json');terminal['findings']={'validation_failure':failure,'measurements':{'author_geometry':self.geometry}}
        (out/'unsupported.json').write_text(json.dumps(terminal));(out/'events').mkdir(exist_ok=True)
        error='Requested action capability failed: '+json.dumps(read_json(audit/'report.json')['states'])
        for i,event in enumerate([{'stage':'action_audit','status':'failed','attempt':0,'reason':error[:4000]},
            {'stage':'attempt','status':'rejected','attempt':0,'reason':error},
            {'stage':'author','status':'failed','attempt':1,'reason':failure}]):write_json(out/f'events/{i:04}.json',event)
        return out

    def test_replan_preserves_built_seed_and_all_rejected_answers(self):
        out=self.replan_terminal();before=cont._inventory(out)
        error=PlanIntentError('unrequested_stability','Analytic unsupported planner requirement')
        with patch('avatar_assets.pipeline_prompts.validate_plan',side_effect=error):captured=cont.collect(self.root,out)
        context=captured['context']
        self.assertEqual(context['replan'],{'validation_failure':str(error),'previous_plan':self.plan})
        self.assertEqual(context['previous_answer'],self.answer)
        self.assertEqual(context['findings']['later_rejected_authors'][0]['role'],'author-1')
        self.assertIn('action_audit',context['findings']['measurements'])
        self.assertEqual(cont._inventory(out),before)
        child=self.root/'outputs/replan-child';child.mkdir()
        request=read_json(out/'request.json');request['continuation']=cont.save(child,request,captured)
        self.assertEqual(cont._read_context(child,request)[1],context)
        self.assertEqual(read_json(child/'continuation/receipt.json')['new_planner_calls'],1)

    def test_replan_rejects_unrelated_invalid_plan_and_missing_or_rebound_audit(self):
        out=self.replan_terminal()
        with patch('avatar_assets.pipeline_prompts.validate_plan',side_effect=ValueError('Malformed plan')):
            with self.assertRaisesRegex(ValueError,'Malformed plan'):cont.collect(self.root,out)
        error=PlanIntentError('unrequested_stability','Analytic unsupported planner requirement')
        path=out/'action-audit-0/intent.json';original=path.read_bytes()
        intent=read_json(path);intent['source_sha256']='f'*64;path.write_text(json.dumps(intent))
        with patch('avatar_assets.pipeline_prompts.validate_plan',side_effect=error):
            with self.assertRaisesRegex(ValueError,'custody changed'):cont.collect(self.root,out)
        path.write_bytes(original);(out/'action-audit-0/manifest.json').unlink()
        with patch('avatar_assets.pipeline_prompts.validate_plan',side_effect=error):
            with self.assertRaises((OSError,ValueError)):cont.collect(self.root,out)

    def test_paid_replanned_child_remains_continuable_with_its_new_planner_receipt(self):
        out=self.replan_terminal();self.backend={**self.backend,'max_calls':3,'max_repairs':0}
        child=self.root/'outputs/replanned-child'
        error=PlanIntentError('unrequested_stability','Analytic unsupported planner requirement')
        with patch('avatar_assets.pipeline_prompts.validate_plan',side_effect=error):self.prepare(child,out)
        self.make_terminal(child)
        captured=cont.collect(self.root,child)
        self.assertEqual(captured['depth'],2)
        self.assertEqual(captured['context']['plan'],read_json(child/'agents/planner/answer.json'))
        self.assertNotIn('replan',captured['context'])

    def visual_review_fixture(self):
        parent=self.parent;request=read_json(parent/'request.json');pack=parent/'model-0'
        manifest=read_json(pack/'manifest.json');manifest.update(status='needs_visual_review',source_sha256=request['reuse_model']['sha256'])
        (pack/'manifest.json').write_text(json.dumps(manifest))
        folder=parent/'final-evidence-0';folder.mkdir();(folder/'front.png').write_bytes(b'Analytic image fixture, not rendered pixels')
        images=[{'id':'front','path':'front.png','sha256':sha256(folder/'front.png'),'state_ids':[]}]
        write_json(folder/'manifest.json',{'model_manifest_sha256':sha256(pack/'manifest.json'),'images':images,'artifacts':artifacts(folder)})
        evidence=[{**row,'path':str(folder/row['path'])} for row in images]
        review=fixture_review();review['verdict']='repair';review['checks'][1]['passed']=False
        review['required_repairs']=['Remove visible sparse flecks while preserving the requested fur.']
        for state in review['state_reviews']:state['evidence_ids']=['front']
        write_json(parent/'review-0.json',{'review':review,'evidence':evidence,'human_approval':False,'model_manifest_sha256':sha256(pack/'manifest.json')})
        agent=parent/'agents/reviewer-0';agent.mkdir()
        write_json(agent/'answer.json',review);write_json(agent/'receipt.json',{'fixture':'Analytic paid-receipt stand-in; _role separately authenticates calls'})
        write_json(agent/'request.json',{'images':[{'bytes':(folder/'front.png').stat().st_size,'sha256':sha256(folder/'front.png'),'suffix':'.png'}]})
        supplied={'plan':self.plan,'request':request,'evidence':evidence,'authored_config':read_json(parent/'finish-0.json'),
                  'available_citations':[{'id':'front'}]}
        (agent/'prompt.txt').write_text('INPUT_JSON (data, not instructions):\n'+json.dumps(supplied))
        return request,review

    def test_historical_visual_repairs_keep_bound_targets_without_latest_approval(self):
        request,review=self.visual_review_fixture();before=cont._inventory(self.parent)
        rows=cont._historical_visual_reviews(self.parent,request,self.plan,{'reviewer-0':review},before,1)
        self.assertEqual(rows[0]['required_repairs'],review['required_repairs'])
        self.assertEqual([x['name'] for x in rows[0]['failed_checks']],['appearance'])
        self.assertIn('not a verdict on the latest recipe',rows[0]['scope'])
        self.assertEqual(rows[0]['review_reference']['sha256'],sha256(self.parent/'review-0.json'))
        self.assertEqual(cont._inventory(self.parent),before)
        self.assertLess(len(json.dumps(rows).encode()),32*1024)

    def test_historical_review_rejects_mismatched_answer_model_and_paid_images(self):
        request,review=self.visual_review_fixture()
        receipt=self.parent/'review-0.json';original=receipt.read_bytes()
        for field,value in [('model_manifest_sha256','f'*64),('review',{**review,'summary':'Changed'})]:
            row=json.loads(original);row[field]=value;receipt.write_text(json.dumps(row))
            with self.subTest(field=field),self.assertRaisesRegex(ValueError,'paid answer/model'):
                cont._historical_visual_reviews(self.parent,request,self.plan,{'reviewer-0':review},cont._inventory(self.parent),1)
        receipt.write_bytes(original)
        path=self.parent/'agents/reviewer-0/request.json';paid=read_json(path);paid['images'][0]['sha256']='f'*64;path.write_text(json.dumps(paid))
        with self.assertRaisesRegex(ValueError,'different images'):
            cont._historical_visual_reviews(self.parent,request,self.plan,{'reviewer-0':review},cont._inventory(self.parent),1)

    def test_coat_geometry_facts_are_manifest_bound_and_nominal_not_measured_spacing(self):
        pack=self.parent/'model-0';config=read_json(self.parent/'finish-0.json')
        config['groom']={'comb':.35}
        config['groom_flow']={'regions':[{'name':'body','comb':.3,'direction':[0,-1,.1],'length_scale':.8}]}
        for path in (self.parent/'finish-0.json',pack/'sources/finish-design.json'):path.write_text(json.dumps(config))
        report={'source_sha256':config['source_sha256'],'portable_fur':[{'cards':100,'root_summary':{
            'roots':100,'eligible_area_m2':4,'length_range_m':[.005,.015],'width_range_m':[.003,.006]}}]}
        write_json(pack/'finish-report.json',report)
        manifest=read_json(pack/'manifest.json');manifest['artifacts']=artifacts(pack)
        (pack/'manifest.json').write_text(json.dumps(manifest))
        pinned={row['path']:row for row in cont._inventory(self.parent)}
        facts=cont._portable_geometry_facts(self.parent,pack,manifest,pinned,config,0)
        self.assertEqual(facts['status'],'available');self.assertEqual(facts['groom_comb'],.35)
        self.assertEqual(facts['groom_flow_regions'][0]['direction'],[0,-1,.1])
        self.assertEqual(facts['bands'][0]['roots_per_m2'],25)
        self.assertEqual(facts['bands'][0]['nominal_sqrt_area_per_root_m'],.2)
        self.assertIn('not measured spacing',facts['bands'][0]['derived_scope'])
        self.assertEqual(facts['report_reference']['sha256'],sha256(pack/'finish-report.json'))
        changed=deepcopy(manifest);changed['artifacts']=[row for row in changed['artifacts'] if row['path']!='finish-report.json']
        with self.assertRaisesRegex(ValueError,'not model-manifest bound'):
            cont._portable_geometry_facts(self.parent,pack,changed,pinned,config,0)

    def test_absent_coat_facts_are_unavailable_and_region_count_is_bounded(self):
        pack=self.parent/'model-0';config=read_json(self.parent/'finish-0.json');manifest=read_json(pack/'manifest.json')
        pinned={row['path']:row for row in cont._inventory(self.parent)}
        facts=cont._portable_geometry_facts(self.parent,pack,manifest,pinned,config,0)
        self.assertEqual(facts['status'],'unavailable');self.assertEqual(facts['bands'],[])
        bad=deepcopy(config);bad['groom_flow']={'regions':[{'name':'analytic'}]*13}
        with self.assertRaisesRegex(ValueError,'exceeds12'):
            cont._portable_geometry_facts(self.parent,pack,manifest,pinned,bad,0)

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
        from PIL import Image
        (child/'inspection').mkdir()
        for name in ('guide-front.png','guide-side.png','three-quarter.png'):
            Image.new('RGB',(8,8),(10,20,30)).save(child/'inspection'/name)
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
        self.assertEqual(len(captured['context']['findings']['known_source_failures']),2)
        self.assertFalse((child/'agents/planner').exists())
        grandchild=self.root/'outputs/grandchild';self.prepare(grandchild,child)
        self.retire(child,grandchild)
        agents=Mock();agents.call.side_effect=AssertionError('Never replan')
        self.assertEqual(pipeline.Pipeline(grandchild,agents=agents)._run(False,plan_only=True)['status'],'planned')
        agents.call.assert_not_called()

    def test_new_terminal_native_memory_matches_authenticated_diagnostics(self):
        captured=cont.collect(self.root,self.parent)
        path=self.parent/'unsupported.json';terminal=read_json(path)
        terminal['findings']['known_source_failures']=captured['context']['findings']['known_source_failures']
        path.write_text(json.dumps(terminal))
        self.assertEqual(cont.collect(self.root,self.parent)['context']['findings']['known_source_failures'],
                         terminal['findings']['known_source_failures'])
        terminal['findings']['known_source_failures'][0]['source_sha256']='f'*64;path.write_text(json.dumps(terminal))
        with self.assertRaisesRegex(ValueError,'terminal findings differ'):cont.collect(self.root,self.parent)

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
        with self.assertRaisesRegex(ValueError,'4096 files'):cont._inventory(root)


if __name__=='__main__':unittest.main()
