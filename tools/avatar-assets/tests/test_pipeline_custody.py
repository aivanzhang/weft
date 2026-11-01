"""Source/custody regressions with analytic reports; no native jobs or paid calls."""

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
from tests import test_deformation_gate as gate_fixture


class PipelineSourcePinTests(unittest.TestCase):
    def setUp(self):
        temp=tempfile.TemporaryDirectory();self.addCleanup(temp.cleanup)
        self.root=Path(temp.name).resolve()
        for name,source in pipeline._pipeline_sources().items():
            target=self.root/('scripts' if name.startswith('scripts/') else 'avatar_assets')/Path(name).name
            target.parent.mkdir(exist_ok=True);shutil.copyfile(source,target)
        override=patch.object(pipeline,'ROOT',self.root);override.start();self.addCleanup(override.stop)
        self.out=self.root/'outputs/source-pin-fixture'
        pipeline.prepare('Analytic source pin fixture',self.out)
        self.agents=Mock()
        quiet=patch('builtins.print');quiet.start();self.addCleanup(quiet.stop)

    def omit_script_pin(self):
        path=self.out/'input-receipt.json';receipt=read_json(path)
        receipt['sources']=[row for row in receipt['sources'] if row['path']!='scripts/motion_samples.py']
        path.write_text(json.dumps(receipt))

    def seal_fixture_delivery(self):
        delivery={'status':'complete','request_sha256':sha256(self.out/'request.json'),'packs':[]}
        for index in range(5):
            folder=self.out/f'pack-{index}';folder.mkdir()
            (folder/'fixture.txt').write_text('ANALYTIC PACK, NOT AN AVATAR')
            write_json(folder/'manifest.json',{'artifacts':artifacts(folder)})
            delivery['packs'].append({'path':folder.name,'manifest_sha256':sha256(folder/'manifest.json')})
        write_json(self.out/'delivery.json',delivery)
        write_json(self.out/'delivery-receipt.json',{'delivery.json':sha256(self.out/'delivery.json')})

    def test_motion_sampling_source_is_copied_hashed_and_blocks_changed_live_code_before_calls(self):
        receipt=read_json(self.out/'input-receipt.json')
        saved=self.out/'sources/scripts/motion_samples.py'
        record=next(row for row in receipt['sources'] if row['path']=='scripts/motion_samples.py')
        self.assertEqual(sha256(saved),record['sha256'])
        self.assertEqual(saved.read_bytes(),(self.root/'scripts/motion_samples.py').read_bytes())
        (self.root/'scripts/motion_samples.py').write_text('Changed analytic implementation')
        with self.assertRaisesRegex(ValueError,'backend changed'):
            pipeline.Pipeline(self.out,agents=self.agents)
        self.agents.call.assert_not_called()

    def test_unfinished_receipt_cannot_omit_new_script_pin(self):
        self.omit_script_pin()
        with self.assertRaisesRegex(ValueError,'lacks current source pins'):
            pipeline.Pipeline(self.out,agents=self.agents)
        self.agents.call.assert_not_called()

    def test_sealed_historic_delivery_without_new_script_pin_remains_read_only(self):
        self.omit_script_pin();(self.out/'sources/scripts/motion_samples.py').unlink()
        self.seal_fixture_delivery()
        (self.root/'scripts/motion_samples.py').write_text('New analytic implementation')
        (self.root/'avatar_assets/pipeline.py').write_text('New coordinator implementation')
        result=pipeline.run(self.out,execute=True,agents=self.agents)
        self.assertTrue(result['resumed_without_generation']);self.agents.call.assert_not_called()

    def test_sealed_delivery_does_not_allow_saved_source_bytes_to_change(self):
        self.seal_fixture_delivery()
        (self.out/'sources/scripts/motion_samples.py').write_text('Changed archived source')
        with self.assertRaisesRegex(ValueError,'backend changed'):
            pipeline.Pipeline(self.out,agents=self.agents)
        self.agents.call.assert_not_called()


class PipelineDeformationCustodyTests(unittest.TestCase):
    def setUp(self):
        temp=tempfile.TemporaryDirectory();self.addCleanup(temp.cleanup)
        self.root=Path(temp.name).resolve();(self.root/'events').mkdir()
        self.worker=object.__new__(pipeline.Pipeline)
        self.worker.root=self.root;self.worker.started=time.monotonic()
        self.worker.cancel_requested=None;self.worker._active_progress=None
        self.fixture=gate_fixture.DeformationGateTests();self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.fixture.native_measurement();self.fixture.rig_proof['motion']={'provenance':'analytic_fixture'}
        self.fixture.save()
        self.plan={'status':'ready','states':[{'id':'wave'}],'height_meters':1.2}
        self.source={'path':'unused-analytic.glb','sha256':gate_fixture.SOURCE}
        quiet=patch('builtins.print');quiet.start();self.addCleanup(quiet.stop)

    def audit(self):
        return self.worker._deformation(0,self.fixture.root,self.plan,self.source,
                                        self.fixture.root/'sources/finish-design.json')

    def test_valid_failed_measurement_is_sealed_and_remains_repairable(self):
        with self.assertRaisesRegex(ValueError,'Severe sampled skin deformation') as failure:self.audit()
        self.assertNotIsInstance(failure.exception,pipeline.EvidenceCustodyError)
        report=read_json(self.root/'deformation-gate-0.json')
        self.assertFalse(report['passed']);self.assertEqual(report['measurement_scope'],'bound_native_rest_and_pose_samples')
        self.assertEqual(report['bindings']['source_sha256'],gate_fixture.SOURCE)

    def test_changed_cached_gate_is_custody_failure_and_never_overwritten(self):
        path=self.root/'deformation-gate-0.json';write_json(path,{'passed':True,'invented':'fixture'})
        before=path.read_bytes()
        with self.assertRaisesRegex(pipeline.EvidenceCustodyError,'Cached deformation gate evidence changed'):self.audit()
        self.assertEqual(path.read_bytes(),before)

    def test_missing_previously_started_gate_cannot_be_synthesized_on_resume(self):
        self.worker.event('deformation_gate','running',attempt=0)
        self.worker.event('deformation_gate','complete',attempt=0)
        self.worker.event('deformation_gate','running',attempt=0)
        with self.assertRaisesRegex(pipeline.EvidenceCustodyError,'no saved evidence'):self.audit()
        self.assertFalse((self.root/'deformation-gate-0.json').exists())

    def test_missing_or_changed_native_binding_is_not_a_repairable_measurement(self):
        path=self.fixture.root/'finish-report.json';original=path.read_bytes()
        for value in (None,original+b' '):
            with self.subTest(missing=value is None):
                if value is None:path.unlink()
                else:path.write_bytes(value)
                with self.assertRaises(pipeline.EvidenceCustodyError):self.audit()
                self.assertFalse((self.root/'deformation-gate-0.json').exists())
                path.write_bytes(original)

    def coordinator(self,*,corrupt_gate=False):
        self.worker.request={'max_repairs':1,'media_spec':{'size':720,'portrait':1600,'seconds':4,'samples':8}}
        write_json(self.root/'request.json',self.worker.request)
        self.worker.guard=Mock();self.worker._plan=Mock(return_value=self.plan)
        self.worker._source=Mock(return_value=self.source)
        self.worker.agents=Mock()
        self.worker.agents.call.side_effect=[{'status':'ready'},
            {'status':'unsupported','reason':'Analytic second author cannot repair this geometry.'}]
        self.worker._contacts=Mock(return_value=(self.root,[]))
        self.worker._action_audit=Mock();self.worker._media=Mock();self.worker._deliver=Mock()
        self.built=[]
        def pack(folder,create):
            self.built.append(folder.name);folder.mkdir(exist_ok=True)
            if folder.name=='inspection':
                return {'source_glb_sha256':gate_fixture.SOURCE,'preflight':{},'native':{'rigs':[]}}
            self.assertEqual(folder.name,'model-0','Native reload/media must not run after this gate')
            (folder/'sources').mkdir()
            self.fixture.root=folder;self.fixture.save()
            shutil.copyfile(self.root/'finish-0.json',folder/'sources/finish-design.json')
            manifest=read_json(folder/'manifest.json')
            record=next(row for row in manifest['artifacts'] if row['path']=='sources/finish-design.json')
            record.update(bytes=(folder/record['path']).stat().st_size,sha256=sha256(folder/record['path']))
            (folder/'manifest.json').write_text(json.dumps(manifest))
            return manifest
        self.worker._pack=pack
        if corrupt_gate:write_json(self.root/'deformation-gate-0.json',{'passed':True,'binding':'changed'})
        with ExitStack() as stack:
            stack.enter_context(patch('avatar_assets.quality_policy.build_quality_policy',return_value={}))
            stack.enter_context(patch('avatar_assets.pipeline_prompts.author_prompt',return_value='Analytic author prompt'))
            stack.enter_context(patch('avatar_assets.pipeline_prompts.finish_config',return_value=deepcopy(self.fixture.design)))
            if corrupt_gate:
                with self.assertRaises(pipeline.EvidenceCustodyError):self.worker.run(execute=True)
            else:
                result=self.worker.run(execute=True)
                self.assertEqual(result['status'],'unsupported');self.assertEqual(result['stage'],'author')
        self.worker._action_audit.assert_not_called();self.worker._media.assert_not_called();self.worker._deliver.assert_not_called()
        self.assertEqual(self.built,['inspection','model-0'])

    def test_real_coordinator_stops_corrupt_gate_without_another_author_or_native_stage(self):
        self.coordinator(corrupt_gate=True)
        self.assertEqual([row.args[0] for row in self.worker.agents.call.call_args_list],['author-0'])
        self.worker._contacts.assert_not_called()
        events=[read_json(path) for path in sorted((self.root/'events').glob('*.json'))]
        self.assertEqual(events[-1]['status'],'checkpointed_error')
        self.assertFalse(any(e['stage']=='attempt' and e['status']=='rejected' for e in events))

    def test_real_coordinator_allows_bounded_repair_only_for_valid_failed_measurement(self):
        self.coordinator()
        self.assertEqual([row.args[0] for row in self.worker.agents.call.call_args_list],['author-0','author-1'])
        self.worker._contacts.assert_called_once()
        self.assertFalse(read_json(self.root/'deformation-gate-0.json')['passed'])

    def test_cached_native_pack_corruption_is_custody_failure_before_gate(self):
        pack=self.root/'cached-pack';pack.mkdir()
        (pack/'artifact.txt').write_text('Analytic original')
        write_json(pack/'manifest.json',{'artifacts':artifacts(pack)})
        (pack/'artifact.txt').write_text('Changed analytic evidence')
        create=Mock()
        with self.assertRaisesRegex(pipeline.EvidenceCustodyError,'Stage evidence requires inspection'):
            self.worker._pack(pack,create)
        create.assert_not_called()


if __name__=='__main__':unittest.main()
