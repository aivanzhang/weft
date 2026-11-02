"""Analytic repair routing; no generation, API, Blender or browser execution."""
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from avatar_assets.pipeline import Pipeline, EvidenceCustodyError
from avatar_assets.safety import write_json
from test_pipeline_prompts import plan, request


class PipelineAuthorGeometryTests(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.worker = object.__new__(Pipeline)
        self.worker.root = Path(folder.name)
        self.worker.request = {**request(), 'max_repairs': 1}
        write_json(self.worker.root/'request.json', self.worker.request)
        self.source = {'path': str(self.worker.root/'source.glb'), 'sha256': 'a'*64}
        self.report = {'schema_version': 1, 'status': 'available',
                       'unknown_bone_references': ['missing_hip'],
                       'layers': [{'selected_vertices': 12, 'min_vertices': 30, 'passed': False}],
                       'failures': ['Unknown bone missing_hip', 'Layer requires 30; selected 12']}

    def test_first_validation_failure_keeps_all_geometry_findings_for_one_repair(self):
        worker = self.worker
        worker.guard = Mock()
        worker.event = Mock()
        worker._fail_active_stage = Mock()
        worker._plan = Mock(return_value=plan())
        worker._source = Mock(return_value=self.source)
        worker._pack = Mock(return_value={'source_glb_sha256': 'a'*64, 'preflight': {}, 'native': {'rigs': []}})
        worker._author_preflight = Mock(return_value=deepcopy(self.report))
        worker.agents = Mock()
        worker.agents.call.return_value = {'status': 'ready', 'reason': '', 'design_json': '{}'}
        with patch('avatar_assets.quality_policy.build_quality_policy', return_value={}), \
             patch('avatar_assets.pipeline_prompts.author_prompt', return_value='analytic') as prompt, \
             patch('avatar_assets.pipeline_prompts.finish_config', side_effect=[ValueError('Unknown action target bone'), {}]), \
             patch('avatar_assets.finish_generated.finish') as native:
            result = worker._run(False)
        self.assertEqual(result['status'], 'unsupported')
        self.assertEqual(worker.agents.call.call_count, 2)
        repair = prompt.call_args_list[1].args[4]
        self.assertEqual(repair['validation_failure'], 'Unknown action target bone')
        self.assertEqual(repair['measurements']['author_geometry'], self.report)
        self.assertEqual(result['findings']['measurements']['author_geometry'], self.report)
        self.assertEqual(worker._pack.call_count, 1)  # Inspection only; failed coverage cannot start finishing.
        native.assert_not_called()

    def test_saved_diagnostics_are_reused_without_mutating_the_answer(self):
        answer = {'opaque_fixture': 'unchanged'}
        original = deepcopy(answer)
        with patch('avatar_assets.pipeline_prompts._author_design', return_value={}), \
             patch('avatar_assets.author_geometry.author_diagnostics', return_value=self.report):
            first = self.worker._author_preflight(answer, self.source, {}, 0)
            receipt = self.worker.root/'author-geometry-0.json'
            before = receipt.read_bytes()
            self.assertEqual(self.worker._author_preflight(answer, self.source, {}, 0), first)
        self.assertEqual(receipt.read_bytes(), before)
        self.assertEqual(answer, original)

    def test_changed_diagnostics_fail_custody_without_rewriting(self):
        receipt = self.worker.root/'author-geometry-0.json'
        write_json(receipt, {**self.report, 'layers': []})
        before = receipt.read_bytes()
        with patch('avatar_assets.pipeline_prompts._author_design', return_value={}), \
             patch('avatar_assets.author_geometry.author_diagnostics', return_value=self.report), \
             self.assertRaises(EvidenceCustodyError):
            self.worker._author_preflight({}, self.source, {}, 0)
        self.assertEqual(receipt.read_bytes(), before)


if __name__ == '__main__':
    unittest.main()
