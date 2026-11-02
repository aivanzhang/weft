"""Analytic source/receipt custody; no native processes, providers, or generated assets."""

import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import Mock, patch

from avatar_assets import finish_generated, pipeline
from avatar_assets.pipeline_prompts import author_prompt, finish_config
from avatar_assets.safety import artifacts, read_json, sha256, write_json
from tests import test_pipeline_prompts as prompts
from tests.test_finish_quality_contracts import finish_config as base_config


HELPERS = ('normal_uv_export.py', 'blender_normal_uv_export.py')
ROOT = finish_generated.ROOT


class NormalUVPipelinePinTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(); self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        for name, source in pipeline._pipeline_sources().items():
            target = self.root/('scripts' if name.startswith('scripts/') else 'avatar_assets')/Path(name).name
            target.parent.mkdir(exist_ok=True); shutil.copyfile(source, target)
        override = patch.object(pipeline, 'ROOT', self.root)
        override.start(); self.addCleanup(override.stop)
        quiet = patch('builtins.print'); quiet.start(); self.addCleanup(quiet.stop)
        self.out = self.root/'outputs/analytic-normal-uv-pins'

    def test_both_helpers_are_snapshotted_before_any_execution(self):
        with patch('avatar_assets.meshy.run') as provider, \
             patch('avatar_assets.agent_runner.AgentRunner') as agents, \
             patch('avatar_assets.finish_generated.run_bounded') as native:
            pipeline.prepare('An analytic object.', self.out)
        provider.assert_not_called(); agents.assert_not_called(); native.assert_not_called()
        rows = {row['path']: row for row in read_json(self.out/'input-receipt.json')['sources']}
        for name in HELPERS:
            key = 'scripts/'+name
            self.assertIn(key, rows)
            self.assertEqual(rows[key]['sha256'], sha256(self.root/key))
            self.assertEqual((self.out/'sources'/key).read_bytes(), (self.root/key).read_bytes())

    def test_either_changed_live_helper_stops_unfinished_job_without_agent_call(self):
        pipeline.prepare('An analytic object.', self.out)
        before = artifacts(self.out); agents = Mock()
        for name in HELPERS:
            path = self.root/'scripts'/name; original = path.read_bytes()
            try:
                path.write_bytes(original+b'\n# Analytic changed backend\n')
                with self.subTest(helper=name), self.assertRaisesRegex(ValueError, 'backend changed'):
                    pipeline.Pipeline(self.out, agents=agents)
                self.assertEqual(artifacts(self.out), before)
            finally:
                path.write_bytes(original)
        agents.call.assert_not_called()

    def test_omitting_either_new_pin_cannot_upgrade_an_unfinished_receipt(self):
        pipeline.prepare('An analytic object.', self.out)
        path = self.out/'input-receipt.json'; original = path.read_bytes(); agents = Mock()
        for name in HELPERS:
            receipt = json.loads(original)
            receipt['sources'] = [row for row in receipt['sources'] if row['path'] != 'scripts/'+name]
            path.write_text(json.dumps(receipt)); before = artifacts(self.out)
            with self.subTest(helper=name), self.assertRaisesRegex(ValueError, 'lacks current source pins'):
                pipeline.Pipeline(self.out, agents=agents)
            self.assertEqual(artifacts(self.out), before)
        agents.call.assert_not_called()


class NormalUVFinisherCustodyTests(unittest.TestCase):
    def fixture(self, root):
        source = root/'native'; source.mkdir(); (source/'sources').mkdir()
        target = root/'reused'; target.mkdir()
        for directory in ('avatar_assets', 'scripts'):
            for path in (ROOT/directory).glob('*.py'):
                shutil.copyfile(path, source/'sources'/path.name)
        config = {'export_model': True}
        receipt = {'schema_version': 1, 'policy': 'live-normal-uv-export-v1', 'status': 'supported',
                   'errors': [], 'materials': [], 'authored_targets': [],
                   'scope': 'Analytic copy-custody fixture, not measured native evidence'}
        write_json(source/'normal-uv-export.json', receipt)
        write_json(source/'finish-report.json', {'normal_uv_export': receipt})
        write_json(source/'finish-config.json', config)
        write_json(source/'job.json', {'exit_code': 0, 'failure': None})
        for name in ('source.blend', 'portrait.png', 'finish.log', 'finish-monitor.json',
                     'finish-handle.json', 'model.glb'):
            (source/name).write_bytes(b'ANALYTIC CUSTODY BYTES, NOT MEDIA')
        return source, target, config

    def seal(self, root):
        (root/'manifest.json').write_text(json.dumps({'artifacts': artifacts(root)}))

    def test_real_finisher_copies_both_helpers_before_the_native_boundary(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); cfg = base_config()
            source = root/'analytic.glb'; source.write_bytes(b'ANALYTIC PREFLIGHT FIXTURE')
            cfg.update(source=str(source), source_sha256=sha256(source))
            write_json(root/'config.json', cfg)
            out = root/'output'

            def stop_before_native(command, *args, **kwargs):
                for name in HELPERS:
                    self.assertEqual((out/'sources'/name).read_bytes(), (ROOT/'scripts'/name).read_bytes())
                self.assertIn(str(out/'sources/blender_finish_generated.py'), command)
                raise RuntimeError('Analytic stop before native execution')

            with patch.object(finish_generated, 'preflight', return_value={'vertices': 3, 'triangles': 1, 'skins': 1}), \
                 patch.object(finish_generated, 'run_bounded', side_effect=stop_before_native) as native, \
                 self.assertRaisesRegex(RuntimeError, 'Analytic stop'):
                finish_generated.finish(root/'config.json', out)
            native.assert_called_once()

    def test_successful_reuse_preserves_exact_dedicated_guard_receipt(self):
        with tempfile.TemporaryDirectory() as temporary:
            source, target, cfg = self.fixture(Path(temporary)); self.seal(source)
            before = artifacts(source)
            finish_generated.reuse_native(source, target, cfg)
            self.assertEqual((target/'normal-uv-export.json').read_bytes(), (source/'normal-uv-export.json').read_bytes())
            self.assertEqual(read_json(target/'finish-report.json')['normal_uv_export'], read_json(target/'normal-uv-export.json'))
            self.assertFalse(read_json(target/'native-reuse.json')['native_execution_repeated'])
            self.assertEqual(artifacts(source), before)

    def test_resealed_old_copy_of_either_helper_cannot_be_reused(self):
        for name in HELPERS:
            with self.subTest(helper=name), tempfile.TemporaryDirectory() as temporary:
                source, target, cfg = self.fixture(Path(temporary))
                path = source/'sources'/name
                path.write_bytes(path.read_bytes()+b'\n# Analytic stale helper\n'); self.seal(source)
                with self.assertRaisesRegex(ValueError, 'Native backend changed'):
                    finish_generated.reuse_native(source, target, cfg)
                self.assertEqual(list(target.iterdir()), [])

    def test_resealed_missing_mismatched_or_unsupported_receipt_cannot_be_reused(self):
        for fault in ('missing', 'mismatched', 'unsupported', 'nonempty_errors', 'oversized'):
            with self.subTest(fault=fault), tempfile.TemporaryDirectory() as temporary:
                source, target, cfg = self.fixture(Path(temporary))
                path = source/'normal-uv-export.json'
                if fault == 'missing':
                    path.unlink()
                elif fault == 'oversized':
                    path.write_bytes(b' '*(512*1024+1))
                else:
                    receipt = read_json(path)
                    if fault == 'mismatched': receipt['materials'] = [{'different': True}]
                    elif fault == 'unsupported': receipt['status'] = 'unsupported'
                    else: receipt['errors'] = [{'code': 'unsupported_original_uv'}]
                    path.write_text(json.dumps(receipt))
                    if fault != 'mismatched':
                        (source/'finish-report.json').write_text(json.dumps({'normal_uv_export': receipt}))
                self.seal(source)
                with self.assertRaises(ValueError):
                    finish_generated.reuse_native(source, target, cfg)
                self.assertEqual(list(target.iterdir()), [])


class NormalUVAuthorBoundaryTests(unittest.TestCase):
    def test_author_cannot_inject_tangent_targets_or_export_guard_results(self):
        for field in ('authored_normal_targets', 'normal_uv_export', 'normal_uv_export_policy'):
            design = prompts.design(); design[field] = {'status': 'supported'}
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, 'override'):
                finish_config(prompts.author_response(design), prompts.request(), prompts.plan(),
                              {'path': '/selected/model.glb', 'sha256': 'a'*64}, 0)

    def test_prompt_preserves_repair_evidence_and_distinguishes_rebake_from_preservation(self):
        finding = {'normal_uv_export_rejection': {'status': 'unsupported',
            'errors': [{'code': 'unsupported_original_uv', 'material': 'AnalyticShell', 'normal_uv_index': 1}]}}
        prompt = author_prompt(prompts.request(), prompts.plan(), {'native': {}, 'preflight': {}},
                               prompts.design(), {'measurements': finding})
        data = json.loads(prompt.split('INPUT_JSON (data, not instructions):\n', 1)[1])
        self.assertEqual(data['repair_findings']['measurements'], finding)
        for phrase in ('UV2 when source UV1 exists', 'must not reinterpret an original source UV1 normal frame',
                       'every affected face is intentionally', 'separate frame-preservation implementation',
                       'never supply private budget or tangent-target fields'):
            self.assertIn(phrase, prompt)


if __name__ == '__main__':
    unittest.main()
