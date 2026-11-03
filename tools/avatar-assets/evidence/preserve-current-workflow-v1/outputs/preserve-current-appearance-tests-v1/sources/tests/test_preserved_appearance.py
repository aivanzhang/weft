"""Analytic preservation contracts; no native generation, provider calls or asset approval."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from PIL import Image

from avatar_assets import repair_composition as comp, pipeline_prompts as prompts
from avatar_assets.pipeline import Pipeline
from avatar_assets.pipeline_evidence import _appearance_baseline
from avatar_assets.safety import sha256,write_json,artifacts
from avatar_assets.openai_runner import _strict
from avatar_assets.agent_runner import _schema_check
from test_repair_composition import fixture,ready


def policy():
    return {'schema_version':1,'mode':'preserve_current','baseline':{
        'parent_output':'outputs/original','attempt':0,**{k:'a'*64 for k in
        ('source_sha256','model_manifest_sha256','design_sha256','review_sha256','descriptor_sha256')}},
        'authorization':{'path':'.local/studio/jobs/'+'1'*24+'/appearance-preference.json','sha256':'b'*64}}


class PreservedAppearanceTests(unittest.TestCase):
    def test_strict_patch_has_no_finishing_mutation_and_keeps_motion_scope(self):
        b,_,_=fixture();schema=comp.patch_schema(b,policy());_schema_check(schema);_strict(schema)
        properties=schema['properties']['patch']['anyOf'][0]['properties']
        self.assertEqual(set(properties['finish_updates']['properties']),set(comp.FINISH_FIELDS))
        self.assertTrue(all(v=={'type':'null'} for v in properties['finish_updates']['properties'].values()))
        self.assertEqual(properties['remove_finish_fields']['maxItems'],0)
        raw=ready(b);raw['patch']['root_translations']=[{'state_id':'dance','values':[[0,0,0],[.01,0,.035]]}]
        original=deepcopy(b);effective=comp.compose_patch(raw,b,policy())
        self.assertEqual({k:v for k,v in effective['design'].items() if k!='authored_actions'},
                         {k:v for k,v in b['baseline_answer']['design'].items() if k!='authored_actions'})
        self.assertEqual(b,original)
        raw['patch']['root_translations'][0]['state_id']='idle'
        with self.assertRaises(ValueError):comp.compose_patch(raw,b,policy())

    def test_every_finish_change_and_removal_rejected_not_dropped(self):
        b,_,_=fixture()
        for field in comp.FINISH_FIELDS:
            raw=ready(b);raw['patch']['finish_updates'][field]={}
            with self.subTest(field=field),self.assertRaises(ValueError):comp.compose_patch(raw,b,policy())
        raw=ready(b);raw['patch']['remove_finish_fields']=['camera_fit']
        with self.assertRaises(ValueError):comp.compose_patch(raw,b,policy())
        b['failed_root_trajectory_requirements']=[]
        self.assertEqual(comp.patch_schema(b,policy())['properties']['patch']['anyOf'][0]['properties']['root_translations'],{'type':'null'})

    def test_advancing_motion_baseline_does_not_advance_original_look(self):
        b,_,plan=fixture();raw=ready(b);raw['patch']['root_translations']=[{'state_id':'dance','values':[[0,0,0],[0,0,.035]]}]
        newer=deepcopy(b);newer['baseline_answer']=comp.compose_patch(raw,b,policy());newer['failed_root_trajectory_requirements']=[]
        with patch.object(comp,'preserved_original',return_value=b):
            comp.assert_preserved_baseline(Path('/fixture'),{'appearance_policy':policy()},plan,newer)
            changed=deepcopy(newer);changed['baseline_answer']['design']['camera_view']='front'
            with self.assertRaisesRegex(ValueError,'original finishing'):
                comp.assert_preserved_baseline(Path('/fixture'),{'appearance_policy':policy()},plan,changed)
            changed=deepcopy(newer);changed['baseline_answer']['design']['authored_actions'][0]['keys'][0]['rotations'][0]['xyz'][0]=60
            with self.assertRaisesRegex(ValueError,'articulation'):
                comp.assert_preserved_baseline(Path('/fixture'),{'appearance_policy':policy()},plan,changed)

    def test_exact_coordinator_render_scale_and_finishing_not_only_typed_fields(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);folder=root/'outputs/original';folder.mkdir(parents=True)
            config={'id':'old','source':'old/path','source_sha256':'a'*64,'normal_strength':.3,'resolution':720,
                    'samples':16,'height':.38,'groom':{'parents':100},'authored_actions':[{'id':'dance'}]}
            path=folder/'finish-0.json';write_json(path,config);p=policy();p['baseline']['design_sha256']=sha256(path)
            new={**config,'id':'new','source':'new/path','authored_actions':[{'id':'dance','root_motion':'new'}]}
            comp.assert_preserved_config({'appearance_policy':p},new,root=root)
            for key,value in [('normal_strength',.2),('samples',48),('height',1.2),('groom',{'parents':101})]:
                with self.subTest(key=key),self.assertRaisesRegex(ValueError,'finishing, rendering'):
                    comp.assert_preserved_config({'appearance_policy':p},{**new,key:value},root=root)
            path.write_text('{}')
            with self.assertRaisesRegex(ValueError,'recipe changed'):
                comp.assert_preserved_config({'appearance_policy':p},new,root=root)

    def test_prompt_preserves_reference_and_current_motion_without_fur_tuning(self):
        b,review,plan=fixture();b['active_review']=review;request={'appearance_policy':policy()}
        with patch.object(prompts,'build_quality_policy',return_value={}),patch.object(prompts,'_author_resource_envelope',return_value={}):
            text=prompts.protected_author_prompt(request,plan,{},b,{'review':{'old':'coat changes'}})
            critic=prompts.typed_critic_prompt(request,plan,{}, {'quality_policy':{}},[])
        guide,payload=text.split('INPUT_JSON (data, not instructions):\n')
        self.assertIn('Every finish_updates value must be null',guide)
        self.assertNotIn('Allowed finishing properties',guide);self.assertNotIn('groom:{parents',guide)
        self.assertEqual(json.loads(payload)['request'],request)
        self.assertIn('not Muse parity or further fur refinement',critic)
        self.assertIn('Identity, reference fidelity, framing, deformation and motion',critic)
        self.assertNotIn('Reject bald patches, polygon/colored',critic)
        self.assertLess(len(text.encode()),128*1024)

    def test_baseline_is_not_current_identity_or_motion_evidence(self):
        evidence=[{'id':'current','kind':'multiview','state_ids':[]},
            {'id':'motion','kind':'isolated_motion_pose','state_ids':['dance']},
            {'id':'evidence-preserved-appearance-baseline','kind':'appearance_baseline','state_ids':[]}]
        review={'verdict':'accept','checks':[{'name':n,'evidence_ids':['current']} for n in
                    ('identity','appearance','framing','deformation')],
                'state_reviews':[{'id':'dance','evidence_ids':['motion'],'requirement_reviews':[]}]}
        with self.assertRaisesRegex(ValueError,'selected baseline'):Pipeline._bind_review(review,evidence)
        review['checks'][1]['evidence_ids'].append('evidence-preserved-appearance-baseline')
        Pipeline._bind_review(review,evidence)
        review['checks'][0]['evidence_ids']=['evidence-preserved-appearance-baseline']
        with self.assertRaisesRegex(ValueError,'actual output'):Pipeline._bind_review(review,evidence)
        review['checks'][0]['evidence_ids']=['current'];review['state_reviews'][0]['evidence_ids']=['evidence-preserved-appearance-baseline']
        with self.assertRaisesRegex(ValueError,'own actual motion'):Pipeline._bind_review(review,evidence)

    def test_baseline_comparison_keeps_pixels_and_rejects_changed_review_image(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);parent=root/'outputs/original';folder=parent/'final-evidence-0';folder.mkdir(parents=True)
            child=root/'outputs/child';child.mkdir();out=child/'evidence';out.mkdir();p=policy()
            images=[]
            for name,color in [('native-views','red'),('exported-views','blue')]:
                path=folder/(name+'.png');Image.new('RGB',(100,100),color).save(path)
                images.append({'id':'evidence-'+name,'path':path.name,'sha256':sha256(path)})
            write_json(folder/'manifest.json',{'model_manifest_sha256':p['baseline']['model_manifest_sha256'],
                'images':images,'artifacts':artifacts(folder)})
            write_json(parent/'review-0.json',{'evidence':images});p['baseline']['review_sha256']=sha256(parent/'review-0.json')
            write_json(child/'request.json',{'appearance_policy':p});(child/'inputs').mkdir();write_json(child/'inputs/appearance-preference.json',{});sources=[]
            def source(path):sources.append(str(path));return path
            with patch('avatar_assets.appearance_policy.verify_policy',return_value=p):
                path,report=_appearance_baseline(child/'request.json',out,source)
                with Image.open(path) as im:
                    self.assertEqual(im.size,(200,128));self.assertEqual(im.getpixel((20,60)),(255,0,0));self.assertEqual(im.getpixel((120,60)),(0,0,255))
                self.assertEqual(report['appearance_policy'],p);self.assertEqual(len(report['baseline_images']),2)
                Image.new('RGB',(100,100),'green').save(folder/'native-views.png')
                with self.assertRaisesRegex(ValueError,'differs from reviewed'):_appearance_baseline(child/'request.json',out,source)

    def test_baseline_is_sealed_as_eighth_attachment_without_changing_current_coverage(self):
        from avatar_assets.pipeline_evidence import build_evidence
        from test_pipeline_evidence import FullEvidenceTests
        helper=FullEvidenceTests();helper.setUp()
        try:
            reference=helper.root/'reference.png';Image.new('RGB',(720,720),'gray').save(reference)
            def compare(request,out,source):
                path=out/'preserved-appearance-baseline.png';Image.new('RGB',(100,100),'red').save(path)
                return path,{'appearance_policy':policy(),'scope':'analytic baseline'}
            with patch('avatar_assets.pipeline_evidence._appearance_baseline',side_effect=compare):
                report=build_evidence(helper.pack,helper.exported,helper.media,helper.root/'new-evidence',reference,
                                      appearance_request=helper.root/'request.json')
            self.assertEqual(len(report['images']),8)
            baseline=report['images'][-1]
            self.assertEqual(baseline['id'],'evidence-preserved-appearance-baseline')
            self.assertEqual(baseline['kind'],'appearance_baseline');self.assertEqual(baseline['state_ids'],[])
            self.assertTrue(any(r['path']==baseline['path'] and r['sha256']==baseline['sha256'] for r in report['artifacts']))
            self.assertEqual(report['appearance_preservation']['appearance_policy'],policy())
            self.assertFalse(any(r['id']==baseline['id'] for r in report['coverage']))
        finally:helper.doCleanups()

    def test_cached_final_evidence_cannot_omit_preservation_scope(self):
        from unittest.mock import Mock
        from avatar_assets.safety import read_json
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);worker=object.__new__(Pipeline);worker.root=root;worker.request={'appearance_policy':policy()}
            packs=[]
            for name in ('model','exported','media'):
                path=root/name;path.mkdir();write_json(path/'manifest.json',{});packs.append(path)
            proof={'model_manifest_sha256':sha256(packs[0]/'manifest.json'),
                'exported_manifest_sha256':sha256(packs[1]/'manifest.json'),
                'media_manifest_sha256':sha256(packs[2]/'manifest.json'),'images':[],'coverage':[]}
            worker._pack=Mock(return_value=proof)
            with self.assertRaisesRegex(ValueError,'selected preserved appearance'):worker._full_evidence(0,*packs)
            proof['appearance_preservation']={'appearance_policy':policy()}
            proof['images']=[{'path':'baseline.png','kind':'appearance_baseline'}]
            images,coverage=worker._full_evidence(0,*packs)
            self.assertEqual(len(images),1);self.assertEqual(coverage,[])


if __name__=='__main__':unittest.main()
