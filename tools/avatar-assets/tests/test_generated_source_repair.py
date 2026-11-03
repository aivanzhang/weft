"""Analytic receipt fixtures; no provider or native execution."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from avatar_assets.generated_source_repair import generated_source, geometry_failure, GEOMETRY_ERROR
from avatar_assets.meshy import stage_request
from avatar_assets.meshy_access import digest
from avatar_assets.safety import sha256, write_json


class GeneratedSourceRepairTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()/'job'
        self.root.mkdir()
        self.request = {'id': 'test-robot', 'reuse_model': None, 'reference': None,
                        'max_credits': 35, 'max_repairs': 0}
        self.plan = {'generation_prompt': 'One articulated robot', 'subject': 'robot',
                     'style': 'painted metal', 'height_meters': 0.4}
        self.config = {'id': 'test-robot-mesh', 'prompt': self.plan['generation_prompt'],
                       'subject': self.plan['subject'], 'style': self.plan['style'],
                       'height_meters': 0.4, 'rig': 'none', 'states': [],
                       'generation': {'mode': 'text', 'geometry_resolution': '4k',
                                      'texture_resolution': '4k', 'target_polycount': 35000, 'pose_mode': ''}}
        self.put('plan.json', self.plan)
        self.put('generation-config.json', self.config)
        self.put('generation/inputs/config.json', self.config)
        sources = []
        for name in ('meshy.py', 'meshy_access.py'):
            path = self.root/'generation/sources'/name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text('# analytic source snapshot\n')
            sources.append({'path': name, 'bytes': path.stat().st_size, 'sha256': sha256(path)})
        execution = {row['path']: row['sha256'] for row in sources}
        self.generation = {'id': self.config['id'], 'config': self.config, 'model': 'meshy-7.1',
                           'automatic_paid_retries': 0, 'sources': sources}
        self.put('generation/plan.json', self.generation)
        plan_hash = sha256(self.root/'generation/plan.json')
        self.put('generation-receipt.json', {'plan_sha256': plan_hash})
        class Dependencies:
            def call(self, job, stage):
                return {'state': 'SUCCEEDED', 'task_id': 'task-mesh'}
        for stage in ('mesh', 'texture'):
            prefix = 'generation/stages/'+stage+'/'
            endpoint, body, credits = stage_request(self.root/'generation', self.generation, stage, Dependencies())
            self.put(prefix+'request-000.json', {'endpoint': endpoint, 'body': body,
                     'request_sha256': digest(body), 'estimated_credits': credits, 'execution_sources': execution})
            self.put(prefix+'submission-000.json', {'task_id': 'task-'+stage,
                     'estimated_credits': credits, 'execution_sources': execution})
            self.put(prefix+'status-000.json', {'task': {'id': 'task-'+stage, 'status': 'SUCCEEDED',
                     'task_error': None, 'consumed_credits': credits}, 'execution_sources': execution})
            for name, checksum in execution.items():
                path = self.root/prefix/'tools'/(checksum[:16]+'-'+name)
                path.parent.mkdir(exist_ok=True)
                path.write_text('# analytic source snapshot\n')
            model = self.root/prefix/'downloads/model.glb'
            model.parent.mkdir(exist_ok=True)
            model.write_bytes(b'analytic unrigged placeholder; not a generated asset')
            row = {'path': 'model.glb', 'bytes': model.stat().st_size, 'sha256': sha256(model)}
            self.put(prefix+'downloads/manifest.json', {'kind': 'meshy_original_downloads',
                     'status': 'downloaded_not_visually_accepted', 'stage': stage, 'task_id': 'task-'+stage,
                     'plan_sha256': plan_hash, 'artifacts': [row], 'downloads': {'model.glb': row}})
        self.source_hash = sha256(model)
        self.put('generation-cost.json', {'reserved_credits': 35, 'provider_reported_credits': 35,
                 'paid_submissions': 2, 'estimated_credits': 35, 'usd_cost': None})
        self.put('inspection/manifest.json', {'source_glb_sha256': self.source_hash, 'native': {'rigs': []}})

    def put(self, path, value):
        target = self.root/path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(value)+'\n')

    def modify(self, path, key, value):
        data = json.loads((self.root/path).read_text())
        data[key] = value
        self.put(path, data)

    def test_completed_independent_source_has_exact_pins_and_no_generation(self):
        before = {str(p): sha256(p) for p in self.root.rglob('*') if p.is_file()}
        with patch('avatar_assets.meshy.Client', side_effect=AssertionError('No provider client')):
            source = generated_source(self.root, self.request)
        self.assertEqual(source['sha256'], self.source_hash)
        self.assertEqual(set(source['stages']), {'mesh', 'texture'})
        self.assertEqual(before, {str(p): sha256(p) for p in self.root.rglob('*') if p.is_file()})

    def test_download_bytes_or_manifest_tampering_is_rejected(self):
        (self.root/'generation/stages/texture/downloads/model.glb').write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'artifact changed'):
            generated_source(self.root, self.request)

    def test_unfinished_provider_or_unknown_credit_is_rejected(self):
        path = 'generation/stages/texture/status-000.json'
        data = json.loads((self.root/path).read_text())
        data['task']['consumed_credits'] = None
        self.put(path, data)
        with self.assertRaisesRegex(ValueError, 'known-charge'):
            generated_source(self.root, self.request)

    def test_task_identity_and_original_plan_are_bound(self):
        self.modify('generation/stages/texture/submission-000.json', 'task_id', 'other-task')
        with self.assertRaisesRegex(ValueError, 'known-charge'):
            generated_source(self.root, self.request)

    def test_stage_request_or_source_snapshot_change_is_rejected(self):
        self.modify('generation/stages/texture/request-000.json', 'body', {})
        with self.assertRaisesRegex(ValueError, 'request differs'):
            generated_source(self.root, self.request)

    def test_paid_retry_is_not_hidden_by_successful_download(self):
        original = json.loads((self.root/'generation/stages/mesh/submission-000.json').read_text())
        self.put('generation/stages/mesh/submission-001.json', original)
        with self.assertRaisesRegex(ValueError, 'paid retries'):
            generated_source(self.root, self.request)

    def test_oversized_receipt_is_rejected_before_parse(self):
        (self.root/'generation-receipt.json').write_bytes(b' '*(4*1024**2+1))
        with self.assertRaisesRegex(ValueError, 'too large'):
            generated_source(self.root, self.request)

    def test_symlink_parent_or_source_is_rejected(self):
        alias = Path(self.temp.name)/'alias'
        alias.symlink_to(self.root, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, 'unchanged fresh-source'):
            generated_source(alias, self.request)

    def test_unsupported_image_route_is_explicit(self):
        with self.assertRaisesRegex(ValueError, 'text-generated sources only'):
            generated_source(self.root, {**self.request, 'reference': {'path': 'image.png'}})

    def geometry(self):
        source = generated_source(self.root, self.request)
        diagnostic = {'status': 'available', 'failures': [{'min_vertices': 20, 'selected_vertices': 6}]}
        findings = {'validation_failure': GEOMETRY_ERROR, 'measurements': {'author_geometry': diagnostic}}
        self.put('author-geometry-0.json', diagnostic)
        self.put('quality-policy.json', {})
        self.put('unsupported.json', {'stage': 'repair_budget', 'status': 'unsupported', 'findings': findings})
        return source, diagnostic, findings

    def test_geometry_repair_recomputes_exact_findings_without_changing_minima(self):
        source, diagnostic, findings = self.geometry()
        raw = {'status': 'ready', 'immutable_analytic_design': {'min_vertices': 20}}
        original = copy.deepcopy(raw)
        with patch('avatar_assets.pipeline_prompts._author_design', return_value=raw['immutable_analytic_design']), \
             patch('avatar_assets.pipeline_prompts.finish_config', return_value={}) as config, \
             patch('avatar_assets.author_geometry.author_diagnostics', return_value=diagnostic) as check:
            actual = geometry_failure(self.root, self.request, self.plan, raw, source)
        self.assertEqual(actual, findings)
        self.assertEqual(raw, original)
        self.assertEqual(check.call_args.args[0]['min_vertices'], 20)
        config.assert_called_once()
        self.assertFalse((self.root/'model-0').exists())

    def test_geometry_measurement_or_terminal_mismatch_is_rejected(self):
        source, diagnostic, findings = self.geometry()
        with patch('avatar_assets.pipeline_prompts._author_design', return_value={}), \
             patch('avatar_assets.pipeline_prompts.finish_config', return_value={}), \
             patch('avatar_assets.author_geometry.author_diagnostics', return_value={**diagnostic, 'failures': []}):
            with self.assertRaisesRegex(ValueError, 'exact measured diagnostics'):
                geometry_failure(self.root, self.request, self.plan, {}, source)

    def test_native_artifact_cannot_be_relabelled_geometry_only(self):
        source, _, _ = self.geometry()
        (self.root/'model-0').mkdir()
        with self.assertRaisesRegex(ValueError, 'unexpected native'):
            geometry_failure(self.root, self.request, self.plan, {}, source)
