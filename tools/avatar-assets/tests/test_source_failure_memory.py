"""Analytic orchestration only: no provider, Blender, or generated asset claims."""
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
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


class SourceFailureMemoryTests(unittest.TestCase):
    def test_duplicate_bound_and_current_diagnostics_remain_exact(self):
        findings={'measurements':diagnostic(), 'review':{'summary':'Preserved current review'}}
        original=deepcopy(findings)
        memory=pipeline._remember_source_failures([],findings,SOURCE)
        for _ in range(20):memory=pipeline._remember_source_failures(memory,findings,SOURCE)
        self.assertEqual(len(memory),1);self.assertEqual(findings,original)
        for index in range(1,12):
            memory=pipeline._remember_source_failures(memory,{'measurements':diagnostic(index)},SOURCE)
        self.assertEqual(len(memory),8)
        self.assertEqual(memory[0]['finish_config_sha256'],f'{4:064x}')
        self.assertEqual(memory[-1]['finish_config_sha256'],f'{11:064x}')
        self.assertEqual(pipeline._remember_source_failures([],{'known_source_failures':memory},SOURCE),memory)

    def test_wrong_source_and_unbounded_or_unbound_entries_rejected(self):
        for findings in ({'measurements':diagnostic(source='d'*64)},
                         {'known_source_failures':[{}]*9}):
            with self.assertRaises(ValueError):pipeline._remember_source_failures([],findings,SOURCE)
        original=pipeline._remember_source_failures([],{'measurements':diagnostic()},SOURCE)
        for mutate in (lambda m:m[0].update(message='x'*1001),
                       lambda m:m[0].update(source_sha256='e'*64),
                       lambda m:m[0]['evidence']['receipt'].update(path='../unbound.json')):
            memory=deepcopy(original);mutate(memory)
            with self.assertRaises(ValueError):pipeline._remember_source_failures([],{'known_source_failures':memory},SOURCE)

    def test_native_failure_survives_critic_overwrite_before_next_author(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);worker=pipeline.Pipeline.__new__(pipeline.Pipeline)
            worker.root=root;worker.request={'max_repairs':2};worker.continuation=None
            worker.guard=Mock();worker.event=Mock();worker._fail_active_stage=Mock()
            worker._plan=Mock(return_value={'status':'ready'})
            worker._source=Mock(return_value={'path':'analytic.glb','sha256':SOURCE})
            worker._author_preflight=Mock(return_value={'status':'not_applicable'})
            worker._diagnostics=Mock(side_effect=lambda p:diagnostic() if p.name=='model-0' else {})
            worker._deformation=Mock(return_value={});worker._portable_attachment=Mock(return_value={'status':'not_applicable','passed':True})
            attachment={'evidence':{'manifest_sha256':'d'*64}}
            worker._attachment_summary=Mock(return_value=attachment)
            worker._action_audit=Mock(return_value={'passed':True,'states':[]})
            worker._full_evidence=Mock(return_value=([],{}));worker._contacts=Mock(return_value=(None,[]))
            worker._bind_review=Mock();worker.unsupported=Mock(return_value={'status':'unsupported'})
            def pack(path,create):
                if path.name=='inspection':return {'source_glb_sha256':SOURCE,'preflight':{},'native':{'rigs':[]}}
                if path.name=='model-0':raise ValueError('Analytic native failure')
                if path.name=='model-1':
                    (path/'sources').mkdir(parents=True)
                    (path/'sources/finish-design.json').write_bytes((root/'finish-1.json').read_bytes())
                    write_json(path/'manifest.json',{'fixture':True})
                    write_json(path/'finish-report.json',{'states':[]})
                    write_json(path/'render-attributes.json',{})
                    (path/'model.glb').write_bytes(b'Analytic fixture, not a renderable model')
                    return {'source_sha256':SOURCE,'rig_verification':{}}
                if path.name=='reload-1':return {'status':'passed','source_manifest_sha256':pipeline.sha256(root/'model-1/manifest.json')}
                if path.name=='exported-preview-1':return {'source_glb_sha256':pipeline.sha256(root/'model-1/model.glb')}
                raise AssertionError(path)
            worker._pack=Mock(side_effect=pack)
            portable=root/'portable';portable.mkdir();write_json(portable/'manifest.json',{'clips':[]})
            worker._media=Mock(return_value=(root/'media',portable))
            worker.agents=Mock();worker.agents.call.side_effect=[{'status':'ready'},{'status':'ready'},
                {'verdict':'repair','summary':'Analytic portable flecks need adjustment'},
                {'status':'unsupported','reason':'Analytic stop after checking repaired prompt'}]
            with patch('avatar_assets.repair_visual_inputs.author_base_images',return_value=[Path('/analytic/guide-'+str(i)+'.png') for i in range(3)]), \
                 patch('avatar_assets.repair_visual_inputs.author_images',side_effect=lambda base,repair:base+repair[:8-len(base)]), \
                 patch('avatar_assets.repair_visual_inputs.from_current_review',return_value={'status':'unavailable','images':[]}), \
                 patch('avatar_assets.quality_policy.build_quality_policy',return_value={}), \
                 patch('avatar_assets.quality_policy.evaluate_quality_evidence',return_value={'passed':True,'missing':[]}), \
                 patch('avatar_assets.pipeline_prompts.finish_config',return_value={'analytic':True}), \
                 patch('avatar_assets.pipeline_prompts.author_schema',return_value={}), \
                 patch('avatar_assets.pipeline_prompts.author_prompt',return_value='Analytic') as author_prompt, \
                 patch('avatar_assets.pipeline_prompts.critic_prompt',return_value='Analytic'), \
                 patch('avatar_assets.pipeline_prompts.critic_schema',return_value={}), \
                 patch('avatar_assets.pipeline_prompts.validate_review'):
                worker._run(False)
            prompts=author_prompt.call_args_list
            self.assertEqual(len(prompts),3)
            after_native,after_critic=prompts[1].args[4],prompts[2].args[4]
            self.assertEqual(after_native['known_source_failures'],after_critic['known_source_failures'])
            self.assertIn('native_failure',after_native['measurements'])
            self.assertIn('review',after_critic);self.assertIn('portable_attachment',after_critic['measurements'])
            self.assertEqual(after_critic['known_source_failures'][0]['finish_config_sha256'],'0'*64)
            self.assertEqual([call.args[0] for call in worker.agents.call.call_args_list],
                             ['author-0','author-1','reviewer-1','author-2'])

    def test_verified_continuation_seeds_first_author_without_mutating_context(self):
        # The caller has already verified the continuation receipt before this pure summary step.
        context={'findings':{'measurements':diagnostic(), 'historical_visual_reviews':[{'summary':'Preserved'}]}}
        original=deepcopy(context)
        memory=pipeline._remember_source_failures([],context['findings'],SOURCE)
        self.assertEqual(len(memory),1);self.assertEqual(context,original)


if __name__=='__main__':unittest.main()
