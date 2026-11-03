"""Analytic patch contract tests; no model/native calls or generated quality claims."""
from copy import deepcopy
import json
import unittest
from unittest.mock import Mock,patch
from avatar_assets import repair_composition as comp,pipeline_prompts as prompts,pipeline
from avatar_assets.agent_runner import _schema_check,_validate
from avatar_assets.openai_runner import _strict
from avatar_assets.author_contract import decode_typed_author
from test_author_contract import typed


def fixture():
    design={'authored_rig':{'space':'normalized_bounds_blender_z_up','height_meters':.4,
        'bones':[{'name':'root','parent':None,'head':[.5,.5,0],'tail':[.5,.5,.4]},
                 {'name':'arm','parent':'root','head':[.5,.5,.4],'tail':[.8,.5,.6]}],
        'base_bones':['root'],'layers':[{'bones':['arm'],'min':[.4,.4,.4],'max':[1,1,1],'feather':.1,'min_vertices':1}]},
        'weight_constraints':{'fallback_bone':'root','regions':{'arm':{'center':[.7,.5,.5],'radius':[.2,.2,.2]}}},
        'smoothing':{'iterations':1,'strength':.1},
        'authored_actions':[{'id':state,'interpolation':'LINEAR','keys':[
            {'second':0,'rotations':{'arm':[1,2,3]},'translations_m':{'root':[0,0,0],'arm':[.001,0,0]}},
            {'second':4,'rotations':{'arm':[4,5,6]},'translations_m':{'root':[0,0,.01],'arm':[.002,0,0]}}]} for state in ['dance','idle']],
        'action_bindings':[{'state_id':state,'roles':[{'id':'body','root_bone':'root','bones':['root','arm']}]} for state in ['dance','idle']],
        'camera_fit':'motion'}
    answer=typed(design)
    plan={'states':[{'id':s,'requirements':[{'id':'travel','kind':'travel','role':'body'}]} for s in ['dance','idle']]}
    review={'verdict':'repair','checks':[{'name':n,'passed':n!='appearance'} for n in ['identity','appearance','framing','deformation']],
        'state_reviews':[{'id':s,'motion_passed':True,'meaning_passed':s!='dance','requirement_reviews':[{'id':'travel','passed':s!='dance'}]} for s in ['dance','idle']]}
    scope=comp.eligible_scope(answer,review,plan)
    return {'schema_version':1,'policy':comp.POLICY,'source_sha256':'a'*64,'baseline_answer':answer,**scope},review,plan


def ready(baseline):
    return {'schema_version':3,'status':'ready','reason':'Analytic finish/root-only repair',
        'patch':{'finish_updates':{key:None for key in comp.FINISH_FIELDS},'remove_finish_fields':[],'root_translations':None}}


class RepairCompositionTests(unittest.TestCase):
    def test_strict_schema_excludes_protected_rig_and_full_motion(self):
        baseline,_,_=fixture();schema=comp.patch_schema(baseline)
        _schema_check(schema);_strict(schema)
        fields=schema['properties']['patch']['anyOf'][0]['properties']['finish_updates']['properties']
        self.assertFalse(set(fields)&(comp.PROTECTED|{'authored_actions'}))
        self.assertNotIn('rig',schema.get('$defs',{}));self.assertLess(len(json.dumps(schema).encode()),32*1024)

    def test_finish_and_root_patch_preserve_all_protected_values_and_original_bytes(self):
        b,_,_=fixture();raw=ready(b);raw['patch']['finish_updates']['camera_view']='front'
        raw['patch']['root_translations']=[{'state_id':'dance','values':[[0,0,0],[.01,0,.035]]}]
        before=json.dumps([b,raw],sort_keys=True);effective=comp.compose_patch(raw,b)
        self.assertEqual(effective['schema_version'],2);self.assertEqual(effective['design']['camera_view'],'front')
        self.assertEqual(comp.protected_projection(effective,'root',{'dance'}),comp.protected_projection(b['baseline_answer'],'root',{'dance'}))
        self.assertEqual(effective['design']['authored_rig'],b['baseline_answer']['design']['authored_rig'])
        self.assertEqual(effective['design']['weight_constraints'],b['baseline_answer']['design']['weight_constraints'])
        self.assertEqual(json.dumps([b,raw],sort_keys=True),before)
        self.assertEqual(decode_typed_author(effective)['authored_actions'][0]['keys'][1]['translations_m']['root'],[.01,0,.035])

    def test_appearance_only_cannot_change_root_and_other_states_remain_protected(self):
        b,_,_=fixture();raw=ready(b)
        raw['patch']['root_translations']=[{'state_id':'idle','values':[[0,0,0],[0,0,.1]]}]
        with self.assertRaises(ValueError):comp.compose_patch(raw,b)
        b['failed_root_trajectory_requirements']=[]
        self.assertEqual(comp.patch_schema(b)['properties']['patch']['anyOf'][0]['properties']['root_translations'],{'type':'null'})
        raw['patch']['root_translations']=[{'state_id':'dance','values':[[0,0,0],[0,0,.1]]}]
        with self.assertRaises(ValueError):comp.compose_patch(raw,b)

    def test_attempted_skin_bone_geometry_or_motion_edits_are_not_silently_dropped(self):
        b,_,_=fixture()
        for field in sorted(comp.PROTECTED|{'authored_actions','source','action_bindings'}):
            raw=ready(b);raw['patch']['finish_updates'][field]={}
            with self.subTest(field=field),self.assertRaises(ValueError):comp.compose_patch(raw,b)
        raw=ready(b);raw['patch']['root_translations']=[{'state_id':'dance','values':[[0,0,0]]}]
        with self.assertRaises(ValueError):comp.compose_patch(raw,b)

    def test_removal_is_explicit_and_conflicting_edits_rejected(self):
        b,_,_=fixture();raw=ready(b);raw['patch']['remove_finish_fields']=['camera_fit']
        self.assertIsNone(comp.compose_patch(raw,b)['design']['camera_fit'])
        raw['patch']['finish_updates']['camera_fit']='rest'
        with self.assertRaisesRegex(ValueError,'updated and removed'):comp.compose_patch(raw,b)
        raw=ready(b);raw['patch']['remove_finish_fields']=['weight_constraints']
        with self.assertRaises(ValueError):comp.compose_patch(raw,b)

    def test_rejected_patch_does_not_replace_fixed_baseline(self):
        b,_,_=fixture();raw=ready(b);raw['patch']['root_translations']=[{'state_id':'dance','values':[[0,0,0],[0,0,2]]}]
        rejected=comp.compose_patch(raw,b)
        self.assertEqual(comp.compose_patch(ready(b),b),{**b['baseline_answer'],'reason':ready(b)['reason']})
        self.assertNotEqual(rejected['design']['authored_actions'],b['baseline_answer']['design']['authored_actions'])

    def test_arm_deformation_identity_or_unexplained_meaning_failures_do_not_qualify(self):
        b,review,plan=fixture()
        for field in ('identity','framing','deformation'):
            changed=deepcopy(review);next(r for r in changed['checks'] if r['name']==field)['passed']=False
            self.assertIsNone(comp.eligible_scope(b['baseline_answer'],changed,plan))
        changed=deepcopy(plan);changed['states'][0]['requirements'][0]['kind']='articulation'
        self.assertIsNone(comp.eligible_scope(b['baseline_answer'],review,changed))
        changed=deepcopy(review);changed['state_reviews'][0]['requirement_reviews']=[]
        self.assertIsNone(comp.eligible_scope(b['baseline_answer'],changed,plan))
        changed=deepcopy(review);changed['state_reviews'][0]['motion_passed']=False
        self.assertIsNone(comp.eligible_scope(b['baseline_answer'],changed,plan))

    def test_nonroot_role_and_multiple_hierarchy_roots_do_not_qualify(self):
        b,review,plan=fixture();changed=deepcopy(b['baseline_answer'])
        changed['design']['action_bindings'][0]['roles'][0]['root_bone']='arm'
        self.assertIsNone(comp.eligible_scope(changed,review,plan))
        changed=deepcopy(b['baseline_answer']);changed['design']['authored_rig']['bones'][1]['parent']=None
        self.assertIsNone(comp.eligible_scope(changed,review,plan))

    def test_unsupported_is_explicit_and_cannot_include_a_patch(self):
        b,_,_=fixture();raw={'schema_version':3,'status':'unsupported','reason':'Outside protected scope','patch':None}
        self.assertEqual(comp.compose_patch(raw,b)['status'],'unsupported')
        raw['patch']=ready(b)['patch']
        with self.assertRaises(ValueError):comp.compose_patch(raw,b)

    def test_duplicate_state_and_remove_entries_are_rejected(self):
        b,_,_=fixture();raw=ready(b);raw['patch']['remove_finish_fields']=['camera_fit']*2
        with self.assertRaisesRegex(ValueError,'Duplicate'):comp.compose_patch(raw,b)
        raw=ready(b);change={'state_id':'dance','values':[[0,0,0],[0,0,0]]};raw['patch']['root_translations']=[change,change]
        with self.assertRaises(ValueError):comp.compose_patch(raw,b)

    def test_new_passing_root_pose_becomes_immutable_and_failures_do_not_rebaseline(self):
        b,review,plan=fixture();raw=ready(b)
        raw['patch']['root_translations']=[{'state_id':'dance','values':[[0,0,0],[0,0,.035]]}]
        candidate=deepcopy(b);candidate['baseline_answer']=comp.compose_patch(raw,b)
        for state in review['state_reviews']:
            state['meaning_passed']=True
            for req in state['requirement_reviews']:req['passed']=True
        candidate.update(comp.eligible_scope(candidate['baseline_answer'],review,plan))
        updated=comp.advance_baseline(b,candidate)
        self.assertEqual(updated['failed_root_trajectory_requirements'],[])
        self.assertEqual(comp.compose_patch(ready(updated),updated)['design']['authored_actions'],candidate['baseline_answer']['design']['authored_actions'])
        self.assertEqual(comp.advance_baseline(updated,None),updated)
        changed=deepcopy(candidate);changed['baseline_answer']['design']['authored_rig']['layers'][0]['feather']=.2
        with self.assertRaisesRegex(ValueError,'changed protected'):comp.advance_baseline(b,changed)

    def test_fresh_coordinator_switches_after_review_and_advances_only_on_new_pass(self):
        from contextlib import ExitStack
        from pathlib import Path
        import tempfile,shutil
        from avatar_assets.safety import write_json,sha256
        from typed_review_fixtures import review_fixture,wire_review
        baseline,_,plan=fixture();plan['status']='ready'
        root_patch=ready(baseline);root_patch['patch']['root_translations']=[{'state_id':'dance','values':[[0,0,0],[0,0,.035]]}]
        newer=deepcopy(baseline);newer['baseline_answer']=comp.compose_patch(root_patch,baseline)
        newer['failed_root_trajectory_requirements']=[]
        review=review_fixture()
        with tempfile.TemporaryDirectory() as temp,ExitStack() as stack:
            folder=Path(temp);worker=pipeline.Pipeline.__new__(pipeline.Pipeline)
            worker.root=folder;worker.request={'max_repairs':2,'agent_backend':{'provider':'openai'},'reference':None};worker.continuation=None
            write_json(folder/'request.json',worker.request)
            worker.guard=Mock();worker.event=Mock();worker._fail_active_stage=Mock()
            worker._plan=Mock(return_value=plan);worker._source=Mock(return_value={'path':'analytic.glb','sha256':'a'*64})
            worker._author_preflight=Mock(return_value={'status':'not_applicable'});worker._coat_admission_feedback=Mock(return_value={'status':'not_applicable'})
            worker._diagnostics=Mock(return_value={});worker._deformation=Mock(return_value={'passed':True})
            worker._portable_attachment=Mock(return_value={'status':'not_applicable','passed':True});worker._attachment_summary=Mock(return_value={'evidence':{'manifest_sha256':'b'*64}})
            worker._action_audit=Mock(return_value={'passed':True,'states':[]});worker._full_evidence=Mock(return_value=([],{}))
            worker._bind_review=Mock();worker._contacts=Mock(return_value=(folder,[]))
            portable=folder/'portable';portable.mkdir();write_json(portable/'manifest.json',{'clips':[]})
            worker._media=Mock(return_value=(folder/'media',portable))
            def pack(path,create):
                if path.name=='inspection':return {'source_glb_sha256':'a'*64,'preflight':{},'native':{'rigs':[]}}
                index=int(path.name.rsplit('-',1)[1]);model=folder/f'model-{index}'
                if path.name.startswith('model-'):
                    (path/'sources').mkdir(parents=True);shutil.copyfile(folder/f'finish-{index}.json',path/'sources/finish-design.json')
                    write_json(path/'manifest.json',{});write_json(path/'finish-report.json',{'states':[]});write_json(path/'render-attributes.json',{})
                    (path/'model.glb').write_bytes(b'Analytic nonrenderable model')
                    return {'source_sha256':'a'*64,'rig_verification':{}}
                if path.name.startswith('reload-'):return {'status':'passed','source_manifest_sha256':sha256(model/'manifest.json')}
                return {'source_glb_sha256':sha256(model/'model.glb')}
            worker._pack=Mock(side_effect=pack);worker.agents=Mock()
            worker.agents.call.side_effect=[{'status':'ready'},wire_review(review),root_patch,wire_review(review),
                {'schema_version':3,'status':'unsupported','reason':'Analytic terminal after checking scoped transport','patch':None}]
            for name,value in [('avatar_assets.quality_policy.build_quality_policy',{}),('avatar_assets.quality_policy.evaluate_quality_evidence',{'passed':True,'missing':[]}),
                ('avatar_assets.pipeline_prompts.finish_config',{}),('avatar_assets.pipeline_prompts.author_schema',{'full':'V2'}),
                ('avatar_assets.pipeline_prompts.author_prompt','Full author'),('avatar_assets.pipeline_prompts.typed_critic_prompt','Critic'),
                ('avatar_assets.pipeline_prompts.typed_critic_schema',{}),('avatar_assets.pipeline_prompts.validate_review',None),
                ('avatar_assets.repair_visual_inputs.author_base_images',[Path('/analytic/front.png'),Path('/analytic/side.png'),Path('/analytic/three.png')]),
                ('avatar_assets.repair_visual_inputs.from_current_review',{'status':'unavailable','images':[]})]:
                stack.enter_context(patch(name,return_value=value))
            stack.enter_context(patch('avatar_assets.repair_visual_inputs.author_images',side_effect=lambda base,repair:base))
            memories=[{'receipt':{'sha256':str(i)*64},'review':review} for i in [1,2]]
            stack.enter_context(patch('avatar_assets.pipeline._current_visual_memory',side_effect=memories))
            selector=stack.enter_context(patch.object(comp,'select_baseline',side_effect=[None,baseline,newer]))
            compose=stack.enter_context(patch.object(comp,'materialize',side_effect=lambda job,attempt,raw,b:comp.compose_patch(raw,b)))
            prompt=stack.enter_context(patch.object(prompts,'protected_author_prompt',return_value='Protected patch'))
            result=worker._run(False)
            self.assertEqual(result['status'],'unsupported');self.assertEqual(selector.call_count,3,result)
            self.assertEqual(worker.agents.call.call_args_list[0].args[2],{'full':'V2'})
            self.assertEqual(worker.agents.call.call_args_list[2].args[2],comp.patch_schema(baseline))
            self.assertEqual(worker.agents.call.call_args_list[4].args[2],comp.patch_schema(newer))
            self.assertEqual(prompt.call_args_list[1].args[3]['failed_root_trajectory_requirements'],[])
            compose.assert_called_once()

    def test_protected_prompt_includes_one_baseline_and_no_stale_weight_directives(self):
        b,review,plan=fixture();b['active_review']=review
        request={'agent_backend':{'provider':'openai'}}
        findings={'historical_visual_reviews':[{'required_repairs':['OLD CHANGE WEIGHTS']}],
                  'last_visual_review':{'review':{'summary':'CURRENT APPEARANCE ISSUE'}},'review':{'summary':'OLD REVIEW'},
                  'validation_failure':'Current patch exceeds count'}
        before=deepcopy(findings)
        with patch.object(prompts,'build_quality_policy',return_value={}),patch.object(prompts,'_author_resource_envelope',return_value={}):
            text=prompts.protected_author_prompt(request,plan,{},b,findings)
        payload=json.loads(text.split('INPUT_JSON (data, not instructions):\n')[1])
        self.assertEqual(payload['protected_repair_baseline'],b)
        self.assertNotIn('previous_design',payload)
        self.assertNotIn('OLD CHANGE',text);self.assertNotIn('Return AUTHOR_SCHEMA_V2',text)
        self.assertEqual(payload['active_visual_review'],{'summary':'CURRENT APPEARANCE ISSUE'})
        self.assertIn('Current patch exceeds count',text);self.assertEqual(findings,before)
        self.assertLess(len(text.encode()),128*1024)


if __name__=='__main__':unittest.main()
