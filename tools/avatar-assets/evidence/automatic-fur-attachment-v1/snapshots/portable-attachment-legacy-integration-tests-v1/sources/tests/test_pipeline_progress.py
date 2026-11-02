"""Analytic stage receipts and real coordinator control flow; no avatar/model calls."""

from contextlib import ExitStack
import json
from pathlib import Path
import shutil
import tempfile
import time
import unittest
from unittest.mock import Mock, patch

from avatar_assets.pipeline import Pipeline, PipelineCancelled
from avatar_assets.safety import read_json, sha256, write_json
from avatar_assets.studio_jobs import _stage_progress


class PipelineProgressTests(unittest.TestCase):
    def setUp(self):
        temp=tempfile.TemporaryDirectory();self.addCleanup(temp.cleanup)
        self.root=Path(temp.name).resolve();(self.root/'events').mkdir()
        self.worker=object.__new__(Pipeline)
        self.worker.root=self.root;self.worker.started=time.monotonic()
        self.worker.cancel_requested=None;self.worker._active_progress=None
        quiet=patch('builtins.print');quiet.start();self.addCleanup(quiet.stop)

    def events(self):
        return [read_json(path) for path in sorted((self.root/'events').glob('*.json'))]

    def test_stage_must_close_before_next_starts_and_elapsed_is_measured(self):
        self.worker.event('build','running',attempt=0)
        with self.assertRaisesRegex(ValueError,'no terminal progress receipt'):
            self.worker.event('reviewer','running',attempt=0)
        self.worker.event('build','complete',attempt=0,reason='Built; quality review pending.')
        self.worker.event('reviewer','running',attempt=0)
        self.worker.event('reviewer','repair',attempt=0)
        stages=_stage_progress(self.root,'running')
        self.assertEqual([item['status'] for item in stages],['complete','repair'])
        self.assertGreaterEqual(stages[0]['elapsed_seconds'],0)
        self.assertIsNone(self.worker._active_progress)
        self.assertFalse(any(item['status']=='accept' for item in stages))

    def test_outer_failure_closes_exact_attempt_without_acceptance(self):
        def fail(*args,**kwargs):
            self.worker.event('reviewer','running',attempt=1)
            raise ValueError('Invalid numeric evidence citation')
        self.worker._run=fail
        with self.assertRaisesRegex(ValueError,'Invalid numeric evidence'):
            self.worker.run(execute=True)
        stages=_stage_progress(self.root,'needs_inspection')
        self.assertEqual(stages[0]['id'],'reviewer:1')
        self.assertEqual(stages[0]['status'],'failed')
        self.assertIn('Invalid numeric evidence',stages[0]['detail'])
        self.assertEqual(stages[-1]['status'],'checkpointed_error')
        self.assertFalse(any(item['status'] in ('complete','accept') for item in stages))

    def test_cancellation_closes_only_started_work(self):
        def stop(*args,**kwargs):
            self.worker.event('media','running',attempt=0)
            raise PipelineCancelled('Stopped at the safe boundary')
        self.worker._run=stop
        self.assertEqual(self.worker.run(execute=True)['status'],'cancelled')
        self.assertEqual([item['status'] for item in _stage_progress(self.root,'cancelled')],['cancelled','cancelled'])

    def test_unsupported_author_keeps_attempt_and_closes_active_stage(self):
        write_json(self.root/'request.json',{'provenance':'analytic_fixture'})
        self.worker.event('author','running',attempt=1)
        self.worker.unsupported('author','Requested controls cannot be authored',attempt=1)
        stage=_stage_progress(self.root,'unsupported')[0]
        self.assertEqual((stage['id'],stage['status']),('author:1','unsupported'))
        self.assertIsNone(self.worker._active_progress)

    def _coordinator(self, review_error=None):
        source=self.root/'fixture.glb';source.write_bytes(b'ANALYTIC, NOT A RENDERABLE MODEL')
        source_hash=sha256(source)
        self.worker.request={'max_repairs':1,'media_spec':{'size':128,'portrait':128,'seconds':1,'samples':1}}
        write_json(self.root/'request.json',self.worker.request)
        self.worker.guard=Mock();self.worker._plan=Mock(return_value={'status':'ready'})
        self.worker._source=Mock(return_value={'path':str(source),'sha256':source_hash})
        self.worker.agents=Mock()
        self.worker.agents.call.side_effect=[{'status':'ready'},{'status':'ready'},
            {'verdict':'accept','state_reviews':[],'limitations':[]}]
        self.worker._action_audit=Mock(side_effect=[{'passed':False,'states':['fixture torso moves']},
                                                  {'passed':True,'states':[]}])
        self.worker._deformation=Mock(return_value={'passed':True,'provenance':'analytic_fixture'})
        attachment={'status':'not_applicable','passed':True,'reason':'Analytic fixture requests no portable fur',
                    'provenance':'analytic_fixture'}
        self.worker._portable_attachment=Mock(return_value=attachment)
        self.worker._attachment_summary=Mock(return_value={**attachment,'evidence':{'manifest_sha256':'0'*64}})
        self.worker._full_evidence=Mock(return_value=([],{}))
        self.worker._deliver=Mock(return_value={'status':'complete','provenance':'analytic_fixture'})

        def pack(folder,create):
            folder.mkdir(exist_ok=True)
            if folder.name=='inspection':
                return {'source_glb_sha256':source_hash,'preflight':{},'native':{'rigs':[]}}
            attempt=int(folder.name.rsplit('-',1)[1])
            model=self.root/f'model-{attempt}'
            if folder.name.startswith('model-'):
                (folder/'sources').mkdir()
                shutil.copyfile(self.root/f'finish-{attempt}.json',folder/'sources/finish-design.json')
                (folder/'model.glb').write_bytes(b'ANALYTIC NONRENDERABLE CANDIDATE')
                write_json(folder/'manifest.json',{})
                write_json(folder/'finish-report.json',{'states':[]})
                write_json(folder/'render-attributes.json',{})
                return {'source_sha256':source_hash,'rig_verification':{}}
            if folder.name.startswith('reload-'):
                return {'status':'passed','source_manifest_sha256':sha256(model/'manifest.json')}
            if folder.name.startswith('exported-preview-'):
                return {'source_glb_sha256':sha256(model/'model.glb')}
            if folder.name.startswith('media-portable-'):write_json(folder/'manifest.json',{'clips':[]})
            return {}

        self.worker._pack=pack
        with ExitStack() as stack:
            for target,value in (
                ('avatar_assets.quality_policy.build_quality_policy',{}),
                ('avatar_assets.quality_policy.evaluate_quality_evidence',{'passed':True,'missing':[]}),
                ('avatar_assets.pipeline_prompts.author_prompt','Analytic author'),
                ('avatar_assets.pipeline_prompts.finish_config',{}),
                ('avatar_assets.pipeline_prompts.critic_prompt','Analytic critic'),
                ('avatar_assets.pipeline_prompts.critic_schema',{}),
                ('avatar_assets.pipeline_prompts.validate_review',None)):
                stack.enter_context(patch(target,return_value=value))
            if review_error:
                stack.enter_context(patch('avatar_assets.pipeline_prompts.validate_review',side_effect=ValueError(review_error)))
            result=self.worker.run(execute=True)
            self.assertEqual(result['status'],'unsupported' if review_error else 'complete',result.get('findings'))
        states={item['id']:item['status'] for item in _stage_progress(self.root,'complete')}
        self.assertEqual(states['action_audit:0'],'failed')
        self.assertEqual(states['attempt:0'],'rejected')
        for name in ('author','build','deformation_gate','portable_attachment','native_reload','action_audit','exported_views','media','review_evidence'):
            self.assertEqual(states[name+':1'],'complete')
        self.assertEqual(states['portable_attachment:0'],'complete')
        self.assertEqual(self.worker._portable_attachment.call_count,2)
        self.assertEqual(self.worker._attachment_summary.call_count,2)
        self.assertEqual([call.args[0] for call in self.worker._portable_attachment.call_args_list],[0,1])
        self.assertEqual(states['reviewer:1'],'unsupported' if review_error else 'accept')
        self.assertNotIn('running',states.values())
        self.assertEqual(self.worker.agents.call.call_count,3)
        return result

    def test_real_coordinator_repairs_audit_and_closes_each_completed_stage(self):
        self._coordinator()

    def test_exited_reviewer_invalid_data_is_unsupported_not_uncertain_or_accepted(self):
        result=self._coordinator('Reviewer invented evidence')
        self.assertEqual(result['failure_kind'],'review_contract')
        self.worker._deliver.assert_not_called()
        self.assertEqual(read_json(self.root/'unsupported.json')['status'],'unsupported')


class StageSnapshotTests(unittest.TestCase):
    def setUp(self):
        temp=tempfile.TemporaryDirectory();self.addCleanup(temp.cleanup)
        self.root=Path(temp.name).resolve();(self.root/'events').mkdir();self.sequence=0

    def add(self,stage,status,**fields):
        write_json(self.root/'events'/f'{self.sequence:04d}.json',{'stage':stage,'status':status,**fields});self.sequence+=1

    def test_all_bounded_history_keeps_failed_attempt_before_success(self):
        self.add('author','failed',attempt=0,reason='Missing requested motion')
        for i in range(28):self.add('inspection','complete',reason=f'Fixture checkpoint {i}')
        self.add('author','running',attempt=1)
        values=_stage_progress(self.root,'running')
        self.assertEqual([(v['id'],v['status']) for v in values],
                         [('author:0','failed'),('inspection','complete'),('author:1','running')])
        self.assertIn('attempt 1',values[0]['label']);self.assertIn('attempt 2',values[-1]['label'])

    def test_legacy_missing_receipts_are_unresolved_not_implicitly_complete(self):
        self.add('inspection','running');self.add('author','running',attempt=0);self.add('media','running',attempt=0)
        active=_stage_progress(self.root,'running')
        self.assertEqual([v['status'] for v in active],['unresolved','unresolved','running'])
        terminal=_stage_progress(self.root,'needs_inspection')
        self.assertTrue(all(v['status']=='unresolved' and v['recorded_status']=='running' for v in terminal))
        self.assertTrue(all('No terminal stage receipt' in v['detail'] for v in terminal))

    def test_error_detail_redacts_secrets_and_does_not_invent_elapsed(self):
        self.add('pipeline','checkpointed_error',error='msy_fixture_secret https://private.invalid/x /Users/example/key')
        item=_stage_progress(self.root,'needs_inspection')[0]
        self.assertNotIn('elapsed_seconds',item)
        self.assertNotIn('msy_fixture',item['detail']);self.assertNotIn('private.invalid',item['detail'])
        self.assertNotIn('/Users/example',item['detail'])

    def test_invalid_attempt_and_event_byte_inventory_fail_closed(self):
        self.add('build','running',attempt=True)
        with self.assertRaisesRegex(ValueError,'Invalid stage attempt'):_stage_progress(self.root,'running')
        (self.root/'events/0000.json').write_text(' '*1024**2+' ')
        with self.assertRaisesRegex(ValueError,'event bytes'):_stage_progress(self.root,'running')


if __name__=='__main__':
    unittest.main()
