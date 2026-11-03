"""Separate-subject contract tests; analytic bytes, no generated assets or dispatch."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from avatar_assets import scene_requests as scene


def asset(identifier='fox',name='Fox',action='Trot forward, then turn its head.'):
    return {'id':identifier,'name':name,'prompt':'One '+name+' only, with complete visible limbs.',
            'style':'Soft stylized animal','custom_actions':[action],'preset':'preview'}


def request():
    return {'schema_version':1,'id':'forest-pair','title':'Two independent animals',
            'assets':[asset(),asset('owl','Owl','Fly forward while flapping both wings, then land.')],
            'relationships':[]}


def reused(identifier,path=None,checksum=None):
    return {'mode':'reuse','path':path or 'outputs/'+identifier+'/model.glb',
            'sha256':checksum or hashlib.sha256(identifier.encode()).hexdigest(),'subject_ids':[identifier]}


def delivery(asset,index):
    return {'subject_id':asset['id'],**{k:deepcopy(asset[k]) for k in ('prompt','style','custom_actions')},
            'source_sha256':hashlib.sha256(('source'+str(index)).encode()).hexdigest(),
            'model_path':'outputs/child-'+str(index)+'/model.glb','model_sha256':hashlib.sha256(('model'+str(index)).encode()).hexdigest(),
            'delivery_path':'outputs/child-'+str(index)+'/delivery.json','delivery_sha256':hashlib.sha256(('delivery'+str(index)).encode()).hexdigest()}


class SceneContractTests(unittest.TestCase):
    def test_independent_wording_actions_and_output_paths_are_preserved(self):
        value=request();value['assets'][0]['custom_actions']=['Dance  three beats.\nReturn to the start pose.','Wave the left forepaw.']
        before=deepcopy(value);plan=scene.plan_scene(value);children=plan['children']
        self.assertEqual(value,before)
        self.assertEqual([row['subject_id'] for row in children],['fox','owl'])
        self.assertEqual(children[0]['request']['custom_actions'],value['assets'][0]['custom_actions'])
        self.assertEqual(children[1]['request']['custom_actions'],value['assets'][1]['custom_actions'])
        self.assertNotEqual(children[0]['output_key'],children[1]['output_key'])
        self.assertFalse(plan['execution_ready']);self.assertFalse(plan['complete'])
        self.assertEqual(plan['aggregate_estimate']['new_meshy_credits'],70)
        self.assertEqual(plan['aggregate_estimate']['expected_ai_calls'],6)
        children[0]['request']['custom_actions'][0]='CHANGED'
        self.assertEqual(value,before)

    def test_three_different_subjects_and_maximum_safe_ids_remain_bounded(self):
        value=request();value['id']='s'*20
        value['assets'].append(asset('l'*20,'Hinged lamp','Bend the lamp head twice while the base stays fixed.'))
        plan=scene.plan_scene(value)
        self.assertEqual(len(plan['children']),3)
        self.assertTrue(all(len(child['output_key'])<=48 for child in plan['children']))
        self.assertEqual(plan['aggregate_estimate']['new_meshy_credits'],105)
        self.assertEqual(plan['aggregate_estimate']['expected_ai_calls'],9)

    def test_deterministic_normalization_does_not_guess_placement_or_merge_sources(self):
        value=request();del value['id']
        first=scene.validate_scene(value);second=scene.validate_scene(deepcopy(value))
        self.assertEqual(first,second);self.assertEqual(scene.plan_scene(first),scene.plan_scene(second))
        self.assertTrue(all(row['placement'] is None and row['source']=={'mode':'generate'} for row in first['assets']))
        self.assertTrue(all(item['kind']=='independently_rigged_glb' for item in scene.plan_scene(first)['deliverables']))

    def test_attachment_is_recorded_as_unsupported_not_a_fused_child(self):
        value=request();value['relationships']=[{'id':'perch','kind':'attachment','parent':'fox','child':'owl',
            'parent_anchor':'Shoulder','child_anchor':None,'user_request':'The owl perches on the fox and later flies away.'}]
        plan=scene.plan_scene(value);row=plan['scene_manifest']['relationships'][0]
        self.assertEqual(row['status'],'unsupported');self.assertEqual(row['user_request'],value['relationships'][0]['user_request'])
        self.assertEqual(len(plan['children']),2);self.assertEqual(plan['children'][1]['request']['custom_actions'],value['assets'][1]['custom_actions'])
        self.assertFalse(plan['complete']);self.assertFalse(plan['execution_ready'])

    def test_static_transform_requires_explicit_valid_units_and_no_shear_or_reflection(self):
        value=request();value['assets'][1]['placement']={'translation_m':[.5,0,0],'rotation_xyzw':[0,0,0,1],'uniform_scale':.5}
        self.assertEqual(scene.validate_scene(value)['assets'][1]['placement'],value['assets'][1]['placement'])
        for placement in ({'translation_m':[0,0,0],'rotation_xyzw':[0,0,0,2],'uniform_scale':1},
                          {'translation_m':[float('inf'),0,0],'rotation_xyzw':[0,0,0,1],'uniform_scale':1},
                          {'translation_m':[0,0,0],'rotation_xyzw':[0,0,0,1],'uniform_scale':-1},
                          {'translation_m':[0,0,0],'rotation_xyzw':[0,0,0,1],'uniform_scale':[1,2,1]}):
            changed=deepcopy(value);changed['assets'][1]['placement']=placement
            with self.subTest(placement=placement):
                with self.assertRaises(ValueError):scene.validate_scene(changed)

    def test_fused_foreign_and_duplicate_sources_are_refused_without_guessing_split(self):
        value=request();value['assets'][0]['source']=reused('fox');value['assets'][1]['source']=reused('owl')
        self.assertEqual(scene.plan_scene(value)['aggregate_estimate']['new_meshy_credits'],0)
        for ownership in (['fox','owl'],['owl']):
            changed=deepcopy(value);changed['assets'][0]['source']['subject_ids']=ownership
            with self.assertRaises(scene.UnsupportedSceneError) as caught:scene.validate_scene(changed)
            self.assertEqual(caught.exception.code,'fused_or_foreign_source')
        for field in ('path','sha256'):
            changed=deepcopy(value);changed['assets'][1]['source'][field]=changed['assets'][0]['source'][field]
            with self.assertRaisesRegex(scene.UnsupportedSceneError,'same source'):scene.validate_scene(changed)
        changed=deepcopy(value);changed['assets'][0]['source']={'mode':'split_fused'}
        with self.assertRaises(scene.UnsupportedSceneError):scene.validate_scene(changed)

    def test_unknown_fields_oversized_counts_duplicate_ids_and_traversal_fail(self):
        mutations=[]
        for field,value in (('automatic_start',True),('schema_version',True),('space','blender_z_up')):
            mutations.append({**request(),field:value})
        mutations.append({**request(),'assets':[asset()]})
        mutations.append({**request(),'assets':[asset()]*4})
        changed=request();changed['assets'][1]['id']='fox';mutations.append(changed)
        changed=request();changed['assets'][0]['custom_actions']=['a']*9;mutations.append(changed)
        changed=request();changed['assets'][0]['prompt']='a'*4001;mutations.append(changed)
        changed=request();changed['assets'][0]['script']='run';mutations.append(changed)
        for path in ('../model.glb','/tmp/model.glb','outputs/../model.glb','outputs/x.py','outputs\\x.glb'):
            changed=request();changed['assets'][0]['source']=reused('fox',path);mutations.append(changed)
        for changed in mutations:
            with self.subTest(changed=changed):
                with self.assertRaises(ValueError):scene.validate_scene(changed)
        wide=request();wide['assets'].append(asset('lamp','Lamp','Bend the shade.'))
        for row in wide['assets']:row.update(prompt='🦉'*4000,custom_actions=['🦉'*600]*8)
        with self.assertRaisesRegex(ValueError,'64KiB'):scene.validate_scene(wide)

    def test_relationship_unknown_subject_cycles_and_extra_timeline_are_rejected(self):
        value=request();row={'id':'one','kind':'attachment','parent':'fox','child':'owl',
                            'parent_anchor':None,'child_anchor':None,'user_request':'Stay attached.'}
        for relationships in ([{**row,'parent':'wolf'}],[{**row,'start_second':0}],
                              [row,{**row,'id':'two','parent':'owl','child':'fox'}]):
            value['relationships']=relationships
            with self.assertRaises(ValueError):scene.validate_scene(value)

    def test_delivery_metadata_refuses_duplicate_models_sources_and_substituted_actions(self):
        value=request();rows=[delivery(a,i) for i,a in enumerate(value['assets'])]
        result=scene.validate_deliveries(value,rows)
        self.assertEqual(result['status'],'references_validated_only');self.assertFalse(result['complete'])
        for field in ('source_sha256','model_sha256','model_path','subject_id'):
            changed=deepcopy(rows);changed[1][field]=changed[0][field]
            with self.subTest(field=field):
                with self.assertRaises(ValueError):scene.validate_deliveries(value,changed)
        changed=deepcopy(rows);changed[1]['custom_actions']=['Stand still and flap.']
        with self.assertRaisesRegex(ValueError,'prompt/style/actions'):scene.validate_deliveries(value,changed)
        with self.assertRaises(ValueError):scene.validate_deliveries(value,rows[:1])


class ScenePreparationTests(unittest.TestCase):
    def setUp(self):
        temporary=tempfile.TemporaryDirectory();self.addCleanup(temporary.cleanup)
        self.root=Path(temporary.name).resolve();(self.root/'outputs').mkdir();self.config=self.root/'scene.json'
        self.config.write_text(json.dumps(request()));self.out=self.root/'outputs/scene-draft'

    def test_prepare_is_append_only_and_saves_separate_drafts_without_pipeline_or_provider_calls(self):
        before=self.config.read_bytes()
        with patch('avatar_assets.pipeline.prepare',side_effect=AssertionError('No child jobs')), \
             patch('avatar_assets.openai_runner.OpenAIRunner.call',side_effect=AssertionError('No AI')), \
             patch('avatar_assets.meshy.prepare',side_effect=AssertionError('No provider')):
            result=scene.prepare(self.config,self.out,root=self.root)
        self.assertEqual(result['child_jobs_created'],0);self.assertEqual(result['paid_calls'],0)
        self.assertEqual(self.config.read_bytes(),before);self.assertEqual((self.out/'inputs/config.json').read_bytes(),before)
        self.assertTrue((self.out/'children/fox/request.json').is_file());self.assertTrue((self.out/'children/owl/request.json').is_file())
        self.assertFalse((self.root/'.local').exists())
        self.assertTrue((self.out/'sources/scene_requests.py').is_file())
        with self.assertRaisesRegex(ValueError,'already exists'):scene.prepare(self.config,self.out,root=self.root)

    def test_reuse_hash_checks_preserve_sources_and_fail_before_output_on_tamper(self):
        value=request();path=self.root/'outputs/fox-source.glb';path.write_bytes(b'Analytic bytes only, not a renderable model')
        value['assets'][0]['source']=reused('fox','outputs/fox-source.glb',hashlib.sha256(path.read_bytes()).hexdigest())
        self.config.write_text(json.dumps(value));original=path.read_bytes()
        result=scene.prepare(self.config,self.out,root=self.root)
        self.assertEqual(path.read_bytes(),original);self.assertEqual(len(result['source_checks']),1)
        plan=json.loads((self.out/'plan.json').read_text());self.assertEqual(plan['aggregate_estimate']['new_meshy_credits'],35)
        path.write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError,'hash differs'):scene.prepare(self.config,self.root/'outputs/tampered',root=self.root)
        self.assertFalse((self.root/'outputs/tampered').exists())

    def test_input_output_symlinks_and_duplicate_json_fields_are_rejected(self):
        link=self.root/'link.json';link.symlink_to(self.config)
        with self.assertRaises(ValueError):scene.prepare(link,self.out,root=self.root)
        target=self.root/'outputs/target';target.mkdir();linked=self.root/'outputs/link';linked.symlink_to(target,target_is_directory=True)
        with self.assertRaises(ValueError):scene.prepare(self.config,linked/'child',root=self.root)
        self.config.write_text('{"title":"A","title":"B","assets":[]}')
        with self.assertRaisesRegex(ValueError,'Duplicate JSON key'):scene.prepare(self.config,self.out,root=self.root)


if __name__=='__main__':unittest.main()
