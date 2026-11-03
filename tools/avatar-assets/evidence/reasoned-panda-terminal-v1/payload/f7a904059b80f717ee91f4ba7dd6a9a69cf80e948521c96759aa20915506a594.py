"""Analytic image-custody fixtures, not model/provider/native executions."""
from pathlib import Path
import importlib.util
import tempfile
import unittest
from copy import deepcopy
from PIL import Image
from avatar_assets.safety import artifacts,read_json,write_json,sha256

_spec=importlib.util.spec_from_file_location('scratch_visual_inputs',Path(__file__).with_name('repair_visual_inputs.py'))
visual=importlib.util.module_from_spec(_spec);_spec.loader.exec_module(visual)
SOURCE='a'*64


class RepairVisualInputsTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name).resolve();self.parent=self.root/'outputs/parent';self.parent.mkdir(parents=True)
        self.child=self.root/'outputs/child';(self.child/'inspection').mkdir(parents=True)
        self.guides=[]
        for n in range(3):
            path=self.child/'inspection'/f'guide-{n}.png';Image.new('RGB',(8,8),(n,0,0)).save(path);self.guides.append(path)
        self.source={'sha256':SOURCE}
        write_json(self.parent/'request.json',{'reuse_model':self.source})
        model=self.parent/'model-0';model.mkdir()
        write_json(model/'manifest.json',{'source_sha256':SOURCE,'artifacts':[]})
        final=self.parent/'final-evidence-0';final.mkdir();images=[]
        names=['evidence-native-views','evidence-motion-00','evidence-delivered-portrait','evidence-isolated-native-wave',
               'evidence-isolated-exported-wave','evidence-exported-views']
        for n,name in enumerate(names):
            path=final/f'{name}.png';Image.new('RGB',(8,8),(30+n*20,10,0)).save(path)
            images.append({'id':name,'path':path.name,'sha256':sha256(path),'kind':'analytic still','state_ids':[]})
        model_sha=sha256(model/'manifest.json')
        write_json(final/'manifest.json',{'model_manifest_sha256':model_sha,'images':images,'artifacts':artifacts(final)})
        self.evidence=[{**row,'path':str(final/row['path'])} for row in images]
        self.review={'verdict':'repair','checks':[{'name':'appearance','passed':False,'evidence_ids':names}]}
        write_json(self.parent/'review-0.json',{'model_manifest_sha256':model_sha,'review':self.review,'evidence':self.evidence})
        agents=self.parent/'agents/reviewer-0';agents.mkdir(parents=True)
        write_json(agents/'answer.json',self.review)
        write_json(agents/'receipt.json',{'status':'succeeded','answer_sha256':sha256(agents/'answer.json')})
        write_json(agents/'request.json',{'images':[{'bytes':Path(row['path']).stat().st_size,'sha256':row['sha256'],'suffix':'.png'} for row in self.evidence]})
        self.summary={'attempt':0,'parent_output':'outputs/parent',
            'review_reference':{'path':'review-0.json','sha256':sha256(self.parent/'review-0.json')},
            'model_manifest_sha256':model_sha,'answer_sha256':sha256(agents/'answer.json'),
            'receipt_sha256':sha256(agents/'receipt.json'),
            'evidence':[{'id':row['id'],'path':str(Path(row['path']).relative_to(self.parent)),
                         'sha256':row['sha256'],'bytes':Path(row['path']).stat().st_size} for row in self.evidence]}
        self.context={'plan':{},'previous_answer':{},'findings':{'historical_visual_reviews':[self.summary],
            'repair_images':['/untrusted/freeform.png']}}
        (self.child/'continuation').mkdir();write_json(self.child/'continuation/context.json',self.context)
        from avatar_assets.pipeline_continuation import POLICY
        receipt={'policy':POLICY,'schema_version':1,'depth':1,'parent':'outputs/parent','inventory':artifacts(self.parent),
            'parent_request_sha256':sha256(self.parent/'request.json'),'context_sha256':sha256(self.child/'continuation/context.json')}
        write_json(self.child/'continuation/receipt.json',receipt)
        self.request={'reuse_model':self.source,'continuation':{'schema_version':1,'policy':POLICY,'parent':'outputs/parent',
            'receipt':'continuation/receipt.json','sha256':sha256(self.child/'continuation/receipt.json')}}

    def continuation(self):
        return visual.from_continuation(self.root,self.child,self.request,self.context,self.guides)

    def test_continuation_and_current_choose_failure_citations_and_prioritize_exported(self):
        result=self.continuation()
        current=visual.from_current_review(self.parent,SOURCE,0,self.review,self.evidence,self.guides)
        self.assertEqual(result,current);self.assertEqual(result['total_agent_images'],8)
        self.assertEqual([x['id'] for x in result['images']][:2],['evidence-exported-views','evidence-isolated-exported-wave'])
        self.assertEqual(len(result['images']),5)
        self.assertNotIn('/untrusted/freeform.png',str(result))

    def test_changed_image_hash_rejected_before_attachment(self):
        Path(self.evidence[0]['path']).write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError,'changed artifact'):self.continuation()

    def test_changed_source_or_model_rejected(self):
        with self.assertRaisesRegex(ValueError,'source/model mismatch'):
            visual.from_current_review(self.parent,'b'*64,0,self.review,self.evidence,self.guides)
        path=self.parent/'model-0/manifest.json';path.write_text('{}')
        with self.assertRaises(ValueError):self.continuation()

    def test_escaped_paths_and_unbound_context_rejected(self):
        with self.assertRaisesRegex(ValueError,'escaped path'):visual._file(self.parent,'../outside.png')
        changed=deepcopy(self.context);changed['findings']['historical_visual_reviews'][0]['parent_output']='../elsewhere'
        with self.assertRaisesRegex(ValueError,'context differs'):
            visual.from_continuation(self.root,self.child,self.request,changed,self.guides)

    def test_paid_answer_or_paid_images_mismatch_rejected(self):
        path=self.parent/'agents/reviewer-0/request.json';path.write_text('{"images":[]}')
        with self.assertRaisesRegex(ValueError,'paid images differ'):
            visual.from_current_review(self.parent,SOURCE,0,self.review,self.evidence,self.guides)

    def test_static_only_and_no_failure_images_are_explicit(self):
        records=[{**row,'bytes':Path(row['path']).stat().st_size} for row in self.evidence]
        static={'checks':[{'passed':False,'evidence_ids':['evidence-exported-views']}]}
        result=visual._selected(static,records,self.guides)
        self.assertEqual(result['motion_images'],0);self.assertIn('static-only',result['reason'])
        self.assertEqual(visual._selected({'checks':[]},records,self.guides)['status'],'unavailable')

    def test_byte_and_count_bounds_remain_authoritative(self):
        from unittest.mock import patch
        records=[{**row,'bytes':Path(row['path']).stat().st_size} for row in self.evidence]
        with patch.object(visual,'_image_inputs',return_value=[{'bytes':33*1024**2}]),self.assertRaisesRegex(ValueError,'total byte'):
            visual._selected(self.review,records,self.guides)
        with self.assertRaisesRegex(ValueError,'three source'):
            visual._selected(self.review,records,self.guides[:2])


if __name__=='__main__':unittest.main()
