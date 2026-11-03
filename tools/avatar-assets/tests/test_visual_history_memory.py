"""Analytic orchestration only: no provider, Blender, or generated asset claims."""
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from typed_review_fixtures import review_fixture, wire_review
from unittest.mock import Mock, patch

from avatar_assets import pipeline
from avatar_assets.safety import read_json, write_json


SOURCE = 'a'*64


def diagnostic(index=0, source=SOURCE):
    return {'native_failure': {'report': {'source_sha256': source,
        'finish_config_sha256': f'{index:064x}', 'failure_type': 'ValueError',
        'message': 'Analytic UV1 bound rejected the optional layer', 'message_truncated': False},
        'receipt': {'path': 'native-failure.json', 'bytes': 900, 'sha256': 'b'*64},
        'manifest_sha256': 'c'*64}}


class VisualHistoryMemoryTests(unittest.TestCase):
    def _sequence(self, native_first):
        built_index, failed_index = (1, 0) if native_first else (0, 1)
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);worker=pipeline.Pipeline.__new__(pipeline.Pipeline)
            worker.root=root;worker.request={'max_repairs':2}
            historical=[{'summary':'Analytic prior exported coat flecks', 'model_manifest_sha256':'f'*64}]
            worker.continuation={'previous_answer':{'status':'ready'},'findings':{'historical_visual_reviews':deepcopy(historical)}}
            original_continuation=deepcopy(worker.continuation)
            worker.guard=Mock();worker.event=Mock();worker._fail_active_stage=Mock()
            worker._plan=Mock(return_value={'status':'ready'})
            worker._source=Mock(return_value={'path':'analytic.glb','sha256':SOURCE})
            worker._author_preflight=Mock(return_value={'status':'not_applicable'})
            worker._diagnostics=Mock(side_effect=lambda p:diagnostic() if p.name==f'model-{failed_index}' else {})
            worker._deformation=Mock(return_value={});worker._portable_attachment=Mock(return_value={'status':'not_applicable','passed':True})
            attachment={'evidence':{'manifest_sha256':'d'*64}}
            worker._attachment_summary=Mock(return_value=attachment)
            worker._action_audit=Mock(return_value={'passed':True,'states':[]})
            worker._full_evidence=Mock(return_value=([],{}));worker._contacts=Mock(return_value=(None,[]))
            worker._bind_review=Mock();worker.unsupported=Mock(return_value={'status':'unsupported'})
            def pack(path,create):
                if path.name=='inspection':return {'source_glb_sha256':SOURCE,'preflight':{},'native':{'rigs':[]}}
                if path.name==f'model-{failed_index}':raise ValueError('Analytic native failure')
                if path.name==f'model-{built_index}':
                    (path/'sources').mkdir(parents=True)
                    (path/'sources/finish-design.json').write_bytes((root/f'finish-{built_index}.json').read_bytes())
                    write_json(path/'manifest.json',{'fixture':True})
                    write_json(path/'finish-report.json',{'states':[]})
                    write_json(path/'render-attributes.json',{})
                    (path/'model.glb').write_bytes(b'Analytic fixture, not a renderable model')
                    return {'source_sha256':SOURCE,'rig_verification':{}}
                if path.name==f'reload-{built_index}':return {'status':'passed','source_manifest_sha256':pipeline.sha256(root/f'model-{built_index}/manifest.json')}
                if path.name==f'exported-preview-{built_index}':return {'source_glb_sha256':pipeline.sha256(root/f'model-{built_index}/model.glb')}
                raise AssertionError(path)
            worker._pack=Mock(side_effect=pack)
            portable=root/'portable';portable.mkdir();write_json(portable/'manifest.json',{'clips':[]})
            worker._media=Mock(return_value=(root/'media',portable))
            review=review_fixture()
            sequence=[{'status':'ready'},{'status':'ready'},wire_review(review)] if native_first else [{'status':'ready'},wire_review(review),{'status':'ready'}]
            worker.agents=Mock();worker.agents.call.side_effect=sequence+[
                {'status':'unsupported','reason':'Analytic stop after checking repaired prompt'}]
            with patch('avatar_assets.repair_visual_inputs.author_base_images',return_value=[Path('/analytic/guide-'+str(i)+'.png') for i in range(3)]), \
                 patch('avatar_assets.repair_visual_inputs.author_images',side_effect=lambda base,repair:base+repair[:8-len(base)]), \
                 patch('avatar_assets.repair_visual_inputs.from_continuation',return_value={'status':'unavailable','images':[]}), \
                 patch('avatar_assets.repair_visual_inputs.from_current_review',return_value={'status':'unavailable','images':[]}), \
                 patch('avatar_assets.quality_policy.build_quality_policy',return_value={}), \
                 patch('avatar_assets.quality_policy.evaluate_quality_evidence',return_value={'passed':True,'missing':[]}), \
                 patch('avatar_assets.pipeline_prompts.finish_config',return_value={'analytic':True}), \
                 patch('avatar_assets.pipeline_prompts.author_schema',return_value={}), \
                 patch('avatar_assets.pipeline_prompts.author_prompt',return_value='Analytic') as author_prompt, \
                 patch('avatar_assets.pipeline_prompts.typed_critic_prompt',return_value='Analytic'), \
                 patch('avatar_assets.pipeline_prompts.typed_critic_schema',return_value={}), \
                 patch('avatar_assets.pipeline_prompts.validate_review'):
                worker._run(False)
            prompts=author_prompt.call_args_list
            self.assertEqual(len(prompts),3)
            for call in prompts:
                self.assertEqual(call.args[4]['historical_visual_reviews'],historical)
            self.assertEqual(worker.continuation,original_continuation)
            self.assertIsNot(prompts[0].args[4]['historical_visual_reviews'],prompts[1].args[4]['historical_visual_reviews'])
            after_critic=prompts[built_index+1].args[4]
            after_native=prompts[failed_index+1].args[4]
            self.assertEqual(after_critic['review'],review)
            self.assertIn('native_failure',after_native['measurements'])
            self.assertEqual(prompts[2].args[4]['known_source_failures'][0]['finish_config_sha256'],'0'*64)
            last=prompts[2].args[4]['last_visual_review']
            self.assertEqual(last['review'],review)
            self.assertEqual(last['source_sha256'],SOURCE)
            self.assertEqual(last['model_manifest_sha256'],pipeline.sha256(root/f'model-{built_index}/manifest.json'))
            self.assertEqual(last['receipt']['sha256'],pipeline.sha256(root/f'review-{built_index}.json'))
            if not native_first:
                self.assertEqual(prompts[1].args[4]['last_visual_review'],last)
                self.assertIsNot(prompts[1].args[4]['last_visual_review'],last)
            self.assertEqual([call.args[0] for call in worker.agents.call.call_args_list],
                ['author-0','author-1','reviewer-1','author-2'] if native_first else
                ['author-0','reviewer-0','author-1','author-2'])

    def test_inherited_history_survives_native_failure_then_critic_repair(self):
        self._sequence(True)

    def test_latest_bound_critic_survives_following_native_failure(self):
        self._sequence(False)

    def test_current_visual_memory_rejects_changed_receipt_and_oversized_feedback(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);pack=root/'model';pack.mkdir();write_json(pack/'manifest.json',{})
            receipt={'model_manifest_sha256':pipeline.sha256(pack/'manifest.json'),
                     'review':{'summary':'Analytic'},'evidence':[]}
            path=root/'review-0.json';write_json(path,receipt)
            original=deepcopy(receipt)
            result=pipeline._current_visual_memory(SOURCE,pack,path,receipt)
            result['review']['summary']='Changed copy'
            self.assertEqual(receipt,original)
            receipt['review']['summary']='Changed unbound review'
            with self.assertRaisesRegex(ValueError,'receipt/model changed'):
                pipeline._current_visual_memory(SOURCE,pack,path,receipt)
            receipt['review']['summary']='x'*(64*1024);path=root/'oversized-review.json';write_json(path,receipt)
            with self.assertRaisesRegex(ValueError,'exceeds bounded'):
                pipeline._current_visual_memory(SOURCE,pack,path,receipt)

    def test_visual_history_is_bounded_and_copied_without_promoting_current_review(self):
        findings={'historical_visual_reviews':[{'summary':'Prior'}], 'review':{'summary':'Current'}}
        memory=pipeline._historical_visual_memory(findings)
        self.assertEqual(memory,[{'summary':'Prior'}])
        memory[0]['summary']='Changed scratch copy'
        self.assertEqual(findings['historical_visual_reviews'],[{'summary':'Prior'}])
        self.assertEqual(pipeline._historical_visual_memory({'review':{'summary':'Not historical'}}),[])
        for reviews in ([{}]*9, [{'summary':'x'*(64*1024)}], {}):
            with self.assertRaises(ValueError):pipeline._historical_visual_memory({'historical_visual_reviews':reviews})


if __name__=='__main__':unittest.main()
