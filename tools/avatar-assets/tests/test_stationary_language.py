"""Negation scope regressions; language guards are not general semantic proofs."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import unittest

from avatar_assets.action_intent import _stationary_language, required_kinds, validate_requirements
from avatar_assets.pipeline_prompts import validate_plan


class StationaryLanguageTests(unittest.TestCase):
    def test_negated_hold_alternatives_are_not_positive_stability(self):
        for text in (
            'Both arms swing gently in coordination with the dance, visibly moving relative to the torso rather than remaining fixed while the body translates.',
            'Move the arm rather than merely keeping it still.',
            'Move the lamp instead of holding it stationary.',
            'Wave without keeping the arm fixed.',
            'Do not keep the arm fixed.',
            'Never remain motionless.',
            "The arm doesn't remain fixed.",
            'Keep the arm moving rather than fixed.',
            'The arm should remain not fixed but moving.',
            'Move rather than the arm remaining fixed.',
            'Avoid keeping the lamp stationary.',
            'Never hold the arm still.',
        ):
            with self.subTest(text=text):
                self.assertFalse(_stationary_language(text))
                self.assertNotIn('stability', required_kinds(text))
                requirements = [{'id': 'motion', 'role': 'moving_part', 'kind': 'articulation', 'description': text}]
                original = deepcopy(requirements)
                self.assertEqual(validate_requirements(requirements, [{'id': 'moving_part', 'description': 'Moving part'}]), original)
                self.assertEqual(requirements, original)

    def test_affirmative_holds_and_independent_clauses_remain_required(self):
        for text in (
            'Keep the torso and feet steady.',
            'Keep feet, torso and base fixed.',
            'The arms remain fixed while the body translates.',
            'Keep the feet still rather than moving them.',
            'Do not keep the arms fixed, but keep the torso stationary.',
            'Never remain motionless; keep the base still.',
            'Move the arms rather than remaining fixed while keeping the feet planted.',
            'Do not remain motionless while the base stays fixed.',
            'Keep the arm moving rather than fixed, and hold the base stationary.',
            'Not only keep the feet still but also wave the hands.',
            'Wave without moving the feet.',
            'Do not move the torso.',
            "Don't move the base.",
        ):
            with self.subTest(text=text):
                self.assertTrue(_stationary_language(text))
                self.assertIn('stability', required_kinds(text))
                requirement = [{'id': 'hold', 'role': 'support', 'kind': 'articulation', 'description': text}]
                with self.assertRaisesRegex(ValueError, 'must use stability'):
                    validate_requirements(requirement, [{'id': 'support', 'description': 'Supporting part'}])
                requirement[0]['kind'] = 'stability'
                validate_requirements(requirement, [{'id': 'support', 'description': 'Supporting part'}])

    def test_rhythm_rate_and_steady_travel_exclusions_remain(self):
        for text in ('Keep a steady rhythm while dancing.', 'Keep a fixed pace while walking.',
                     'Maintain a steady rhythm.', 'Walk upright across the room.'):
            with self.subTest(text=text):
                self.assertFalse(_stationary_language(text))
        self.assertEqual(required_kinds('Wave without flying.'), {'articulation'})

    def test_retained_api_plan_validates_without_changing_any_bytes(self):
        folder = Path(__file__).resolve().parents[1] / 'outputs/studio-9e1cceb87c40437896c5a77f'
        answer, request = folder/'agents/planner/answer.json', folder/'request.json'
        if not answer.is_file() or not request.is_file():
            self.skipTest('Optional local retained paid-response evidence is absent')
        before = {path: path.read_bytes() for path in (answer, request)}
        plan = json.loads(before[answer])
        result = validate_plan(plan, json.loads(before[request]))
        self.assertEqual(result, plan)
        self.assertEqual(result['states'][0]['id'], 'dance')
        self.assertEqual(len(result['states'][0]['requirements']), 5)
        for path, content in before.items():
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), hashlib.sha256(content).hexdigest())


if __name__ == '__main__':
    unittest.main()
