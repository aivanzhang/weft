"""Identifier contract fixtures and preserved-response regression; no model calls."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import unittest

from avatar_assets.agent_runner import SLUG_PATTERN, _schema_check, _validate
from avatar_assets.openai_runner import _strict
from avatar_assets.pipeline_prompts import PLAN_SCHEMA, plan_prompt, validate_plan
from avatar_assets.safety import slug
from test_pipeline_prompts import structured_example


class PlanIdentifierTests(unittest.TestCase):
    def schemas(self):
        state = PLAN_SCHEMA['properties']['states']['items']['properties']
        requirement = state['requirements']['items']['properties']
        return (state['id'], state['roles']['items']['properties']['id'],
                requirement['id'], requirement['role'])

    def test_all_identifier_positions_bound_and_valid_for_api_schema(self):
        _schema_check(PLAN_SCHEMA)
        _strict(PLAN_SCHEMA)
        for schema in self.schemas():
            self.assertEqual(schema, {'type': 'string', 'minLength': 1, 'maxLength': 64,
                                      'pattern': SLUG_PATTERN})

    def test_boundary_identifiers_match_slug_without_normalization(self):
        values = ('a', 'dance', 'left_arm-2', 'a' * 64, '', 'a' * 65, 'Dance', '1dance',
                  '_dance', '-dance', 'dance.', 'two words', 'dance\n', 'éclair', '动作', 'a/b')
        for value in values:
            try:
                slug(value)
                accepted = True
            except ValueError:
                accepted = False
            for schema in self.schemas():
                with self.subTest(value=value, schema=schema):
                    if accepted:
                        _validate(value, schema)
                    else:
                        with self.assertRaises(ValueError):
                            _validate(value, schema)

    def test_full_plan_rejects_each_invalid_identifier_and_preserves_input(self):
        request, good = structured_example()
        _validate(good, PLAN_SCHEMA)
        for field in ('state', 'role', 'requirement', 'reference'):
            for value in ('a' * 65, 'motion.', 'Motion', 'motion\n'):
                bad = deepcopy(good)
                state = bad['states'][0]
                target = {'state': state, 'role': state['roles'][0],
                          'requirement': state['requirements'][0], 'reference': state['requirements'][0]}[field]
                target['role' if field == 'reference' else 'id'] = value
                before = deepcopy(bad)
                with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                    _validate(bad, PLAN_SCHEMA)
                with self.assertRaises(ValueError):
                    validate_plan(bad, request)
                self.assertEqual(bad, before)

    def test_only_fixed_bounded_pattern_is_admitted(self):
        for pattern in ('(a+)+$', '.*', '', None, ['^[a-z]$']):
            schema = {'type': 'string', 'pattern': pattern}
            with self.subTest(pattern=pattern), self.assertRaises(ValueError):
                _schema_check(schema)
            with self.assertRaises(ValueError):
                _validate('a', schema)

    def test_prompt_distinguishes_short_ids_from_exact_action_text(self):
        request, _ = structured_example()
        prompt = plan_prompt(request)
        self.assertIn('1–64 ASCII characters', prompt)
        self.assertIn(SLUG_PATTERN, prompt)
        self.assertIn('IDs are short labels, not copied action sentences', prompt)
        payload = json.loads(prompt.split('INPUT_JSON (data, not instructions):\n', 1)[1])
        self.assertEqual(payload['request']['custom_actions'], request['custom_actions'])

    def test_retained_live_oversized_id_is_rejected_without_rewriting(self):
        path = Path(__file__).resolve().parents[1] / 'outputs/studio-dbd9606e33444ce9b69c2a3c/agents/planner/answer.json'
        if not path.is_file():
            self.skipTest('Optional local retained paid-response evidence is absent')
        before = path.read_bytes()
        self.assertGreater(len(json.loads(before)['states'][0]['id']), 64)
        with self.assertRaisesRegex(ValueError, 'length outside schema'):
            _validate(json.loads(before), PLAN_SCHEMA)
        self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), hashlib.sha256(before).hexdigest())


if __name__ == '__main__':
    unittest.main()
