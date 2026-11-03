"""Composed continuation custody with analytic files; no native or provider execution."""
import json
import unittest
from unittest.mock import patch

from avatar_assets import openai_runner as api, pipeline_continuation as cont, repair_composition as composition
from avatar_assets.action_intent import PlanIntentError
from avatar_assets.safety import read_json, sha256
from test_author_contract import typed
import test_pipeline_continuation as native_fixtures
import test_visual_repair_continuation as visual_fixtures


class ComposedContinuationTests(unittest.TestCase):
    prepare = native_fixtures.ContinuationTests.prepare
    make_terminal = native_fixtures.ContinuationTests.make_terminal
    visual = visual_fixtures.VisualContinuationTests.visual
    seal = staticmethod(visual_fixtures.VisualContinuationTests.seal)

    def setUp(self):
        native_fixtures.ContinuationTests.setUp(self)
        design = json.loads(self.answer['design_json'])
        for action in design['authored_actions']:
            action['interpolation'] = 'LINEAR'
        self.answer = typed(design)
        self.parent = self.root/'outputs/passing-baseline'
        self.prepare(self.parent)
        self.make_terminal(self.parent)
        self.visual()
        self.baseline_inventory = cont._inventory(self.parent)

    def composed_child(self, *, max_repairs=0):
        parent = self.parent
        child = self.root/'outputs/composed-child'
        self.backend = {**self.backend, 'max_calls': 3 + 2*max_repairs, 'max_repairs': max_repairs}
        self.prepare(child, parent)
        request = read_json(child/'request.json')
        _, context = cont._read_context(child, request)
        baseline = composition.select_baseline(self.root, child, request, context, None,
                                               request['reuse_model']['sha256'], self.plan)
        self.assertIsNotNone(baseline)
        raw = {'schema_version': 3, 'status': 'ready', 'reason': 'Analytic framing finish update.',
               'patch': {'finish_updates': {name: None for name in composition.FINISH_FIELDS},
                         'remove_finish_fields': [], 'root_translations': None}}
        raw['patch']['finish_updates']['camera_view'] = 'front'
        self.answer = composition.compose_patch(raw, baseline)
        (child/'plan.json').write_text(json.dumps(self.plan))
        original_call = api.OpenAIRunner.call

        def submit(runner, role, prompt, schema, images=None):
            self.assertEqual(role, 'author-0')
            self.network.return_value = json.dumps({
                'id': 'resp_composed_author', 'model': api.MODEL, 'service_tier': 'default', 'status': 'completed',
                'usage': {'input_tokens': 100, 'output_tokens': 20, 'total_tokens': 120},
                'output': [{'type': 'message', 'role': 'assistant', 'status': 'completed',
                            'content': [{'type': 'output_text', 'text': json.dumps(raw)}]}]}).encode()
            prompt = 'INPUT_JSON (data, not instructions):\n' + json.dumps({'protected_repair_baseline': baseline})
            returned = original_call(runner, role, prompt, composition.patch_schema(baseline), images=images)
            self.assertEqual(returned, raw)
            self.assertEqual(composition.materialize(child, 0, returned, baseline), self.answer)
            return returned

        self.network.side_effect = None
        with patch.object(api.OpenAIRunner, 'call', new=submit):
            self.make_terminal(child, inherited=True)
        self.network.reset_mock()
        self.network.side_effect = AssertionError('No provider calls during continuation')
        return child, raw, baseline

    def test_native_continuation_uses_effective_design_and_retains_paid_patch(self):
        child, raw, _ = self.composed_child()
        before = cont._inventory(child)
        captured = cont.collect(self.root, child)
        self.assertEqual(captured['context']['previous_answer'], self.answer)
        self.assertEqual(captured['context']['previous_answer']['schema_version'], 2)
        self.assertEqual(read_json(child/'agents/author-0/answer.json'), raw)
        pins = {row['path']: row for row in captured['inventory']}
        self.assertEqual(pins['agents/author-0/answer.json']['sha256'], sha256(child/'agents/author-0/answer.json'))
        self.assertIn('composition/author-0/receipt.json', pins)
        self.assertEqual(cont._inventory(child), before)
        self.assertEqual(cont._inventory(self.parent), self.baseline_inventory)
        self.network.assert_not_called()

    def test_missing_composition_never_falls_back_to_raw_patch(self):
        child, _, _ = self.composed_child()
        (child/'composition/author-0/receipt.json').unlink()
        with self.assertRaises((ValueError, OSError)):
            cont.collect(self.root, child)
        self.network.assert_not_called()

    def test_changed_effective_answer_is_not_a_new_paid_design(self):
        child, _, _ = self.composed_child()
        path = child/'composition/author-0/effective-answer.json'
        changed = read_json(path)
        changed['design']['authored_rig']['base_bones'] = ['Base']
        path.write_text(json.dumps(changed))
        with self.assertRaisesRegex(ValueError, 'Composition receipt/effective'):
            cont.collect(self.root, child)
        self.network.assert_not_called()

    def test_changed_passing_baseline_stops_before_reconstruction(self):
        child, _, _ = self.composed_child()
        path = self.parent/'deformation-gate-0.json'
        changed = read_json(path)
        changed['passed'] = False
        path.write_text(json.dumps(changed))
        with self.assertRaisesRegex(ValueError, 'Protected baseline evidence changed'):
            cont.collect(self.root, child)
        self.network.assert_not_called()

    def test_raw_paid_patch_tampering_is_rejected_before_composition(self):
        child, _, _ = self.composed_child()
        path = child/'agents/author-0/answer.json'
        path.write_text(path.read_text() + ' ')
        with self.assertRaisesRegex(ValueError, 'agent receipt changed'):
            cont.collect(self.root, child)
        self.network.assert_not_called()

    def test_legacy_answer_with_unexplained_composition_is_rejected(self):
        folder = self.parent/'composition/author-0'
        folder.mkdir(parents=True)
        (folder/'receipt.json').write_text('{}')
        with self.assertRaisesRegex(ValueError, 'Legacy author unexpectedly'):
            cont.collect(self.root, self.parent)
        self.network.assert_not_called()

    def test_visual_terminal_reconstructs_from_composed_design(self):
        child, raw, _ = self.composed_child()
        baseline = self.parent
        self.parent = child
        self.network.side_effect = None
        self.visual()
        captured = cont.collect(self.root, child)
        self.assertEqual(captured['context']['previous_answer'], self.answer)
        self.assertEqual(captured['context']['findings']['review'], self.review)
        self.assertEqual(read_json(child/'agents/author-0/answer.json'), raw)
        self.assertEqual(cont._inventory(baseline), self.baseline_inventory)
        self.network.assert_not_called()

    def test_paid_patch_refusal_remains_raw_while_previous_seed_is_effective(self):
        child, _, baseline = self.composed_child(max_repairs=1)
        raw = {'schema_version': 3, 'status': 'unsupported', 'reason': 'Analytic scoped patch refusal.', 'patch': None}
        self.network.side_effect = None
        self.network.return_value = json.dumps({
            'id': 'resp_patch_refusal', 'model': api.MODEL, 'service_tier': 'default', 'status': 'completed',
            'usage': {'input_tokens': 100, 'output_tokens': 20, 'total_tokens': 120},
            'output': [{'type': 'message', 'role': 'assistant', 'status': 'completed',
                        'content': [{'type': 'output_text', 'text': json.dumps(raw)}]}]}).encode()
        runner = api.OpenAIRunner(child/'agents', job_id=child.name, job_digest=sha256(child/'request.json'),
                                  ledger=self.ledger, max_calls=5)
        runner.call('author-1', 'Analytic refusal', composition.patch_schema(baseline))
        terminal = {'status': 'unsupported', 'stage': 'author', 'attempt': 1, 'reason': raw['reason'],
                    'request_sha256': sha256(child/'request.json')}
        for name in ('unsupported.json', 'status.json'):
            (child/name).write_text(json.dumps(terminal))
        self.network.reset_mock()
        self.network.side_effect = AssertionError('No provider calls during continuation')
        captured = cont.collect(self.root, child)
        self.assertEqual(captured['context']['previous_answer'], self.answer)
        refusal = captured['context']['findings']['author_rejection']
        self.assertEqual(refusal['answer'], raw)
        self.assertEqual(refusal['answer_sha256'], sha256(child/'agents/author-1/answer.json'))
        self.assertFalse((child/'composition/author-1').exists())
        self.network.assert_not_called()

    def test_semantic_replan_receives_effective_design_and_separate_paid_raw(self):
        child, raw, _ = self.composed_child()
        expected = {'plan': self.plan, 'previous_answer': self.answer, 'findings': {},
                    'replan': {'validation_failure': 'Analytic semantic failure', 'previous_plan': self.plan}}
        error = PlanIntentError('unrequested_stability', 'Analytic semantic failure')
        with patch('avatar_assets.pipeline_prompts.validate_plan', side_effect=error), \
                patch.object(cont, '_replan_context', return_value=expected) as rebuild:
            captured = cont.collect(self.root, child)
        self.assertEqual(captured['context'], expected)
        self.assertEqual(rebuild.call_args.args[3]['author-0'], self.answer)
        self.assertEqual(rebuild.call_args.kwargs['raw_answers']['author-0'], raw)
        self.network.assert_not_called()


if __name__ == '__main__':
    unittest.main()
