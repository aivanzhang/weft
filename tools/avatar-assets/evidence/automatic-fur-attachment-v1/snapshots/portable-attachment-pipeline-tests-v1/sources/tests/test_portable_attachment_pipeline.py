"""Analytic orchestration/custody fixtures; no renderer, provider or model-role calls."""

from contextlib import ExitStack
from copy import deepcopy
import json
from pathlib import Path
import shutil
import tempfile
import time
import unittest
from unittest.mock import Mock, patch

from avatar_assets import pipeline
from avatar_assets.safety import artifacts, read_json, sha256, write_json
from avatar_assets.studio_jobs import _stage_progress


def report_for(intent, *, status='measured', passed=True):
    return {'schema_version':1,'policy':'portable-fur-attachment-v1','status':status,'passed':passed,
        'intent':deepcopy(intent),'reason':'Explicit analytic orchestration fixture',
        'roots':0 if status=='not_applicable' else 2,
        'states':[{'id':identifier,'times':[0,4]} for identifier in intent['state_ids']],
        'gates':{'animated_attachment_m':1e-5},
        'metrics':{'analytic_failures':0 if passed else 1},
        'witnesses':[] if passed else [{'root':1,'observed_m':.001,'second':2,'state':intent['state_ids'][0]}]}


class AttachmentPipelineTests(unittest.TestCase):
    def setUp(self):
        from avatar_assets import fur_attachment
        temporary=tempfile.TemporaryDirectory(prefix='portable-attachment-pipeline-')
        self.addCleanup(temporary.cleanup)
        self.root=Path(temporary.name).resolve()
        for name,source in pipeline._pipeline_sources().items():
            target=self.root/('scripts' if name.startswith('scripts/') else 'avatar_assets')/Path(name).name
            target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,target)
        self.stack=ExitStack();self.addCleanup(self.stack.close)
        self.stack.enter_context(patch.object(pipeline,'ROOT',self.root))
        self.out=self.root/'outputs/test-attachment'
        pipeline.prepare('A furry articulated creature waves.',self.out,max_repairs=1)
        self.agents=Mock();self.worker=pipeline.Pipeline(self.out,agents=self.agents)
        self.worker.guard=Mock()
        self.pack=self.out/'model-0';self.pack.mkdir()
        (self.pack/'model.glb').write_bytes(b'ANALYTIC NON-RENDERABLE SOURCE; NEVER DECODED')
        write_json(self.pack/'manifest.json',{'scope':'Analytic fixture'})
        self.config=self.out/'finish-0.json';write_json(self.config,{'scope':'Analytic design'})
        self.plan={'status':'ready','states':[{'id':'wave'}],'subject':'Fixture creature','height_meters':1,'limitations':[]}
        self.source={'path':'analytic source','sha256':'a'*64}
        self.status='measured';self.passed=True;self.worker_failure=None;self.invalid_report=False
        self.report_transform=lambda value:value
        self.calls=[]

        def validate(pack,intent,report):
            if self.invalid_report:
                raise ValueError('Analytic custody failure')
            if report.get('intent')!=intent or report.get('status') not in ('measured','not_applicable','unsupported'):
                raise ValueError('Analytic report binding changed')
            expected=report['status']=='not_applicable' or report['status']=='measured' and report['metrics']['analytic_failures']==0
            if report['passed'] is not expected:
                raise ValueError('Analytic cached boolean is inconsistent')
            return report
        self.validate=self.stack.enter_context(patch.object(fur_attachment,'validate_report',side_effect=validate))

        def run(command,log,folder,**limits):
            self.calls.append((list(command),dict(limits)))
            log.write_text('Explicit analytic controller fixture; no child process launched.\n')
            intent=read_json(folder/'intent.json')
            write_json(folder/'report.json',self.report_transform(report_for(intent,status=self.status,passed=self.passed)))
            job={'exit_code':0,'failure':None,'process_exit_verified':True,'manual_stop_required':False,
                 'command':list(command),'limits':{'wall_seconds':60,'rss_mb':512,'output_mb':16},
                 'scope':'Analytic receipt, not a real native or numeric job'}
            if self.worker_failure:job.update(self.worker_failure)
            return job
        self.runner=self.stack.enter_context(patch('avatar_assets.runner.run_bounded',side_effect=run))
        self.stack.enter_context(patch('avatar_assets.meshy.run',side_effect=AssertionError('No provider calls')))

    def audit(self):
        return self.worker._portable_attachment(0,self.pack,self.plan,self.source,self.config)

    def reseal(self):
        folder=self.out/'portable-root-audit-0'
        manifest=read_json(folder/'manifest.json');manifest['artifacts']=artifacts(folder)
        (folder/'manifest.json').write_text(json.dumps(manifest))

    def test_pass_pins_exact_intent_limits_minimal_environment_and_reuses_without_worker(self):
        result=self.audit();again=self.audit()
        self.assertEqual(result,again);self.assertEqual(self.runner.call_count,1)
        self.assertEqual(self.validate.call_count,2)
        command,limits=self.calls[0]
        self.assertEqual(limits,{'timeout':60,'memory_mb':512,'output_mb':16})
        self.assertEqual(command[:3],['/usr/bin/env','-i','PATH=/usr/bin:/bin'])
        self.assertFalse(any('KEY=' in x or x.startswith('HOME=') for x in command))
        intent=read_json(self.out/'portable-root-audit-0/intent.json')
        self.assertEqual(intent,{'schema_version':1,'source_sha256':'a'*64,
            'design_sha256':sha256(self.config),'model_sha256':sha256(self.pack/'model.glb'),'state_ids':['wave']})
        self.assertEqual(read_json(self.out/'portable-root-audit-0.json')['report_sha256'],
                         sha256(self.out/'portable-root-audit-0/report.json'))

    def test_numeric_rejection_remains_measured_cached_evidence(self):
        self.passed=False
        result=self.audit();self.assertFalse(result['passed'])
        self.assertEqual(read_json(self.out/'portable-root-audit-0/manifest.json')['status'],'measured')
        self.assertEqual(self.audit(),result);self.assertEqual(self.runner.call_count,1)
        self.assertFalse((self.out/'stage-failure-portable-root-audit-0.json').exists())

    def test_not_applicable_and_unsupported_are_structured_not_worker_errors(self):
        self.status='not_applicable';self.passed=True
        self.assertEqual(self.audit()['status'],'not_applicable')
        second=self.out/'model-1';second.mkdir()
        (second/'model.glb').write_bytes(b'ANALYTIC SECOND MODEL')
        write_json(second/'manifest.json',{'scope':'Analytic'})
        self.status='unsupported';self.passed=False
        result=self.worker._portable_attachment(1,second,self.plan,self.source,self.config)
        self.assertEqual(result['status'],'unsupported');self.assertFalse(result['passed'])

    def test_failed_or_uncertain_worker_stops_and_does_not_reexecute(self):
        self.worker_failure={'exit_code':1,'failure':'analytic child failure','process_exit_verified':False,
                             'manual_stop_required':True}
        with self.assertRaises(pipeline.EvidenceCustodyError):self.audit()
        self.assertEqual(read_json(self.out/'portable-root-audit-0/manifest.json')['status'],'failed')
        with self.assertRaises(pipeline.EvidenceCustodyError):self.audit()
        self.assertEqual(self.runner.call_count,1)
        self.assertFalse((self.out/'portable-root-audit-0.json').exists())

    def test_validator_rejection_is_not_an_author_repair(self):
        self.invalid_report=True
        with self.assertRaises(pipeline.EvidenceCustodyError):self.audit()
        self.assertFalse((self.out/'portable-root-audit-0.json').exists())

    def test_missing_unsealed_or_previous_started_pack_never_replays(self):
        folder=self.out/'portable-root-audit-0';folder.mkdir()
        with self.assertRaises(pipeline.EvidenceCustodyError):self.audit()
        self.runner.assert_not_called()
        folder.rmdir()
        self.worker.event('portable_attachment','running',attempt=0)
        self.worker.event('portable_attachment','failed',attempt=0,reason='Analytic previous work')
        with self.assertRaises(pipeline.EvidenceCustodyError):self.audit()
        self.runner.assert_not_called()

    def test_resealed_report_change_is_detected_by_controller_receipt(self):
        self.audit();path=self.out/'portable-root-audit-0/report.json'
        report=read_json(path);report['reason']='Changed after measurement'
        path.write_text(json.dumps(report));self.reseal()
        with self.assertRaisesRegex(pipeline.EvidenceCustodyError,'custody receipt'):self.audit()
        self.assertEqual(self.runner.call_count,1)

    def test_missing_controller_receipt_or_model_change_stops_cache(self):
        self.audit();receipt=self.out/'portable-root-audit-0.json';saved=receipt.read_bytes()
        receipt.unlink()
        with self.assertRaises(pipeline.EvidenceCustodyError):self.audit()
        receipt.write_bytes(saved)
        (self.pack/'model.glb').write_bytes(b'CHANGED ANALYTIC MODEL')
        with self.assertRaises(pipeline.EvidenceCustodyError):self.audit()
        self.assertEqual(self.runner.call_count,1)

    def test_cached_job_budget_change_is_rejected_even_if_resealed(self):
        self.audit();path=self.out/'portable-root-audit-0/job.json'
        job=read_json(path);job['limits']['rss_mb']=4096;path.write_text(json.dumps(job));self.reseal()
        with self.assertRaisesRegex(pipeline.EvidenceCustodyError,'worker receipt'):self.audit()
        self.assertEqual(self.runner.call_count,1)

    def test_summary_preserves_bounded_witnesses_and_report_hash(self):
        self.passed=False
        def witnesses(value):
            value['roots']=8;value['metrics']['analytic_failures']=8
            value['witnesses']=[{**value['witnesses'][0],'root':index} for index in range(8)]
            return value
        self.report_transform=witnesses;report=self.audit()
        summary=self.worker._attachment_summary(0,report)
        self.assertEqual(len(summary['witnesses']),4)
        self.assertEqual(summary['retained_witnesses'],8)
        self.assertEqual(summary['metrics'],report['metrics'])
        self.assertEqual(summary['evidence']['sha256'],sha256(self.out/'portable-root-audit-0/report.json'))
        self.assertLessEqual(len(json.dumps(summary).encode()),16*1024)

    def test_new_helper_pins_reject_unfinished_backend_changes(self):
        receipt=read_json(self.out/'input-receipt.json')
        paths={x['path'] for x in receipt['sources']}
        self.assertTrue({'fur_attachment.py','scripts/blender_fur_identity.py'}<=paths)
        for name in ('fur_attachment.py','scripts/blender_fur_identity.py'):
            self.assertTrue((self.out/'sources'/name).is_file())
        (self.root/'avatar_assets/fur_attachment.py').write_text('changed backend')
        with self.assertRaisesRegex(ValueError,'backend changed'):
            pipeline.Pipeline(self.out,agents=self.agents)

    def progression(self, mode):
        worker=self.worker
        worker._plan=Mock(return_value=self.plan);worker._source=Mock(return_value=self.source)
        worker._deformation=Mock(return_value={'passed':True})
        worker._diagnostics=Mock(return_value={});worker._contacts=Mock(return_value=(None,[]))
        worker._action_audit=Mock(return_value={'passed':True,'states':[]})
        worker._media=Mock();worker._full_evidence=Mock();worker._deliver=Mock()
        original_pack=worker._pack;order=[]
        def pack(path,create):
            name=path.name
            if name.startswith('portable-root-audit-'):
                order.append('attachment');return original_pack(path,create)
            if name=='inspection':return {'preflight':{},'native':{'rigs':[]}}
            if name.startswith('model-'):
                path.mkdir(exist_ok=True);(path/'sources').mkdir(exist_ok=True)
                config=self.out/f'finish-{name.rsplit("-",1)[1]}.json'
                shutil.copyfile(config,path/'sources/finish-design.json')
                if not (path/'model.glb').exists():(path/'model.glb').write_bytes(b'ANALYTIC MODEL')
                if not (path/'manifest.json').exists():write_json(path/'manifest.json',{'scope':'Analytic model'})
                write_json(path/'finish-report.json',{'states':[]})
                return {'source_sha256':self.source['sha256'],'rig_verification':{}}
            if name.startswith('reload-'):
                order.append('reload');raise RuntimeError('Analytic stop after gate before expensive work')
            raise AssertionError('Unexpected expensive pack '+name)
        worker._pack=pack
        self.agents.call.side_effect=[{'status':'ready'},{'status':'unsupported','reason':'Analytic repair boundary'}]
        captured=[]
        def author(*args,**kwargs):captured.append(deepcopy(args[-1]));return 'Analytic author input'
        self.stack.enter_context(patch('avatar_assets.pipeline_prompts.author_prompt',side_effect=author))
        self.stack.enter_context(patch('avatar_assets.pipeline_prompts.finish_config',return_value={'scope':'Analytic design'}))
        self.stack.enter_context(patch('avatar_assets.quality_policy.build_quality_policy',return_value={'scope':'Analytic'}))
        if mode=='numeric':self.passed=False
        if mode=='unsupported':self.status='unsupported';self.passed=False
        if mode=='custody':self.invalid_report=True
        return order,captured

    def test_numeric_failure_repairs_once_with_witnesses_before_media_or_critic(self):
        order,captured=self.progression('numeric')
        result=self.worker.run(execute=False)
        self.assertEqual([x.args[0] for x in self.agents.call.call_args_list],['author-0','author-1'])
        self.assertEqual(order,['attachment']);self.assertEqual(result['stage'],'author')
        proof=captured[1]['measurements']['portable_attachment']
        self.assertEqual(proof['witnesses'][0]['root'],1);self.assertFalse(proof['passed'])
        self.worker._media.assert_not_called();self.worker._action_audit.assert_not_called()

    def test_unsupported_mapping_stops_without_repair_call(self):
        order,_=self.progression('unsupported')
        result=self.worker.run(execute=False)
        self.assertEqual(result['status'],'unsupported');self.assertEqual(result['stage'],'portable_attachment')
        self.assertEqual(self.agents.call.call_count,1);self.assertEqual(order,['attachment'])
        self.worker._media.assert_not_called();self.worker._deliver.assert_not_called()

    def test_custody_failure_stops_without_repair_call(self):
        order,_=self.progression('custody')
        with self.assertRaises(pipeline.EvidenceCustodyError):self.worker.run(execute=False)
        self.assertEqual(self.agents.call.call_count,1);self.assertEqual(order,['attachment'])
        self.worker._media.assert_not_called();self.worker._contacts.assert_not_called()

    def test_pass_completes_gate_before_native_reload(self):
        order,_=self.progression('pass')
        with self.assertRaisesRegex(RuntimeError,'Analytic stop'):self.worker.run(execute=False)
        self.assertEqual(order,['attachment','reload'])
        events=[read_json(p) for p in sorted((self.out/'events').glob('*.json'))]
        audit=next(x for x in events if x['stage']=='portable_attachment' and x['status']=='complete')
        self.assertEqual(audit['audit_manifest_sha256'],sha256(self.out/'portable-root-audit-0/manifest.json'))
        self.worker._media.assert_not_called()

    def test_new_delivery_and_critic_bind_the_validated_audit(self):
        self.progression('pass');worker=self.worker;previous_pack=worker._pack
        def fixture_pack(path):
            path.mkdir(exist_ok=True)
            if not (path/'manifest.json').exists():
                (path/'fixture.txt').write_text('Analytic pipeline stage; not media or a model')
                write_json(path/'manifest.json',{'clips':[],'artifacts':artifacts(path)})
        def pack(path,create):
            if path.name=='reload-0':
                fixture_pack(path)
                return {'status':'passed','source_manifest_sha256':sha256(self.pack/'manifest.json')}
            if path.name=='exported-preview-0':
                fixture_pack(path);return {'source_glb_sha256':sha256(self.pack/'model.glb')}
            return previous_pack(path,create)
        worker._pack=pack
        media=self.out/'media-0';portable=self.out/'portable-0'
        for path in (media,portable,self.out/'action-audit-0',self.out/'final-evidence-0'):
            fixture_pack(path)
        worker._media.return_value=(media,portable);worker._full_evidence.return_value=([],{})
        worker._deliver=pipeline.Pipeline._deliver.__get__(worker)
        worker._bind_review=Mock()
        write_json(self.out/'plan.json',self.plan)
        write_json(self.out/'generation-cost.json',{'provider_reported_credits':0,'scope':'Analytic receipt'})
        write_json(self.out/'deformation-gate-0.json',{'passed':True,'scope':'Analytic receipt'})
        review={'verdict':'accept','summary':'Analytic orchestration acceptance only','limitations':['No media was rendered.']}
        self.agents.call.side_effect=[{'status':'ready'},review]
        self.stack.enter_context(patch('avatar_assets.quality_policy.evaluate_quality_evidence',return_value={'passed':True,'missing':[]}))
        critic=self.stack.enter_context(patch('avatar_assets.pipeline_prompts.critic_prompt',return_value='Analytic critic input'))
        self.stack.enter_context(patch('avatar_assets.pipeline_prompts.critic_schema',return_value={}))
        self.stack.enter_context(patch('avatar_assets.pipeline_prompts.review_citations',return_value=[]))
        self.stack.enter_context(patch('avatar_assets.pipeline_prompts.validate_review'))
        self.stack.enter_context(patch('avatar_assets.gallery._Catalog'))
        from avatar_assets.cli import verify_pack
        self.stack.enter_context(patch('avatar_assets.cli.verify_pack',side_effect=lambda path:
            verify_pack(path) if Path(path).name.startswith('portable-root-audit-') else {'scope':'Analytic non-media fixture'}))
        result=worker.run(execute=False)
        self.assertEqual(result['status'],'complete');self.assertEqual(len(result['packs']),8)
        self.assertIn('portable-root-audit-0',[p['path'] for p in result['packs']])
        proof=critic.call_args.args[3]['portable_attachment']
        self.assertTrue(proof['passed']);self.assertEqual(result['portable_attachment'],proof)
        receipt=read_json(self.out/'delivery-receipt.json')
        self.assertEqual(receipt['portable-root-audit-0.json'],sha256(self.out/'portable-root-audit-0.json'))
        (self.out/'portable-root-audit-0/report.json').write_text('{}')
        with self.assertRaises(ValueError):
            pipeline.Pipeline(self.out,agents=Mock()).run(execute=True)
        self.assertEqual(self.runner.call_count,1)

    def test_studio_projects_new_stage_and_marks_interrupted_work_unresolved(self):
        self.worker.event('portable_attachment','running',attempt=0)
        stage=_stage_progress(self.out,'running')[-1]
        self.assertEqual(stage['label'],'Check portable fur attachment · attempt 1')
        self.assertEqual(stage['status'],'running')
        self.assertEqual(_stage_progress(self.out,'needs_inspection')[-1]['status'],'unresolved')
        self.runner.assert_not_called()

    def test_historical_completed_delivery_never_runs_new_audit(self):
        delivery={'status':'complete','request_sha256':sha256(self.out/'request.json'),'packs':[]}
        for index in range(5):
            folder=self.out/f'old-pack-{index}';folder.mkdir()
            (folder/'fixture.txt').write_text('Analytic historical evidence, not media')
            write_json(folder/'manifest.json',{'artifacts':artifacts(folder)})
            delivery['packs'].append({'path':folder.name,'manifest_sha256':sha256(folder/'manifest.json')})
        write_json(self.out/'delivery.json',delivery)
        write_json(self.out/'delivery-receipt.json',{'delivery.json':sha256(self.out/'delivery.json')})
        (self.root/'avatar_assets/fur_attachment.py').write_text('Different current backend')
        worker=pipeline.Pipeline(self.out,agents=self.agents)
        result=worker.run(execute=True)
        self.assertTrue(result['resumed_without_generation'])
        self.agents.call.assert_not_called();self.runner.assert_not_called()


if __name__=='__main__':unittest.main()
