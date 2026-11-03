"""Planner string bounds mirror local validation, not generated-asset acceptance."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import unittest

from avatar_assets.agent_runner import _validate
from avatar_assets.openai_runner import _strict
from avatar_assets.pipeline_prompts import PLAN_SCHEMA, plan_prompt, validate_plan


def fixture():
    return {'status': 'ready', 'reason': '', 'subject': 'Copper telescope', 'style': 'Copper and glass',
            'generation_prompt': 'A copper telescope with an articulated optical tube.', 'height_meters': 1,
            'states': [{'id': 'scan', 'meaning': 'The optical tube turns.', 'user_request': '',
                        'roles': [{'id': 'tube', 'description': 'Optical tube'}],
                        'requirements': [{'id': 'turn', 'kind': 'articulation', 'role': 'tube',
                                          'description': 'The optical tube visibly turns.'}]}],
            'surface_intent': [{'region': 'housing', 'material': 'metal', 'finish': 'Satin copper'}],
            'appearance_checks': ['Copper housing remains recognizable.'], 'limitations': []}


REQUEST = {'states': ['scan'], 'style': '', 'custom_actions': []}


def assign(value, path, replacement):
    for key in path[:-1]:
        value = value[key]
    value[path[-1]] = replacement


class PlanStringBoundsTests(unittest.TestCase):
    def test_schema_is_strict_and_every_string_is_finite(self):
        _strict(PLAN_SCHEMA)
        def walk(schema):
            if schema.get('type') == 'string':
                self.assertTrue('enum' in schema or type(schema.get('maxLength')) is int)
            for child in schema.get('properties', {}).values():
                walk(child)
            if 'items' in schema:
                walk(schema['items'])
        walk(PLAN_SCHEMA)

    def test_exact_local_limits_accept_and_one_over_rejects_in_both_layers(self):
        paths = [(('reason',), 4000), (('subject',), 400), (('style',), 800),
                 (('generation_prompt',), 800), (('states', 0, 'meaning'), 2000),
                 (('states', 0, 'roles', 0, 'description'), 600),
                 (('states', 0, 'requirements', 0, 'description'), 600),
                 (('surface_intent', 0, 'region'), 200), (('surface_intent', 0, 'finish'), 600),
                 (('appearance_checks', 0), 2000), (('limitations', 0), 2000)]
        for path, limit in paths:
            with self.subTest(path=path):
                candidate = fixture()
                if path[0] == 'limitations': candidate['limitations'] = ['A limitation']
                assign(candidate, path, 'é'*limit)
                _validate(candidate, PLAN_SCHEMA)
                self.assertEqual(validate_plan(candidate, REQUEST), candidate)
                assign(candidate, path, 'é'*(limit+1))
                with self.assertRaises(ValueError): _validate(candidate, PLAN_SCHEMA)
                with self.assertRaises(ValueError): validate_plan(candidate, REQUEST)

    def test_custom_request_is_bounded_without_rewriting(self):
        candidate = fixture(); action = 'x'*600
        candidate['states'][0]['user_request'] = action
        request = {**REQUEST, 'custom_actions': [action]}
        _validate(candidate, PLAN_SCHEMA)
        self.assertEqual(validate_plan(candidate, request)['states'][0]['user_request'], action)
        candidate['states'][0]['user_request'] += 'x'
        with self.assertRaises(ValueError): _validate(candidate, PLAN_SCHEMA)
        with self.assertRaises(ValueError): validate_plan(candidate, request)

    def test_unsupported_can_leave_ready_fields_empty(self):
        candidate = fixture()
        candidate.update(status='unsupported', reason='Unsupported fluid simulation', subject='', style='',
                         generation_prompt='', states=[], surface_intent=[], appearance_checks=[], limitations=[])
        _validate(candidate, PLAN_SCHEMA)
        self.assertEqual(validate_plan(candidate, REQUEST), candidate)
        candidate['reason'] = ''
        with self.assertRaises(ValueError): validate_plan(candidate, REQUEST)

    def test_repeated_845_character_failure_is_rejected_without_truncation(self):
        candidate = fixture(); candidate['generation_prompt'] = 'x'*845
        original = deepcopy(candidate)
        with self.assertRaisesRegex(ValueError, 'string length'): _validate(candidate, PLAN_SCHEMA)
        with self.assertRaisesRegex(ValueError, 'Invalid plan generation_prompt'): validate_plan(candidate, REQUEST)
        self.assertEqual(candidate, original)

    def test_retained_paid_answer_is_invalid_and_unchanged(self):
        folder = Path(__file__).resolve().parents[1]/'evidence/accepted-source-plan-failure-v1/payload'
        path = folder/'b43896be47682e48fcffa009d80f97823b21ce22f3ada652310b262678915928.json'
        request_path = folder/'2713f0d5710ac0b14a54fbf1a3423f0a9a7c03043d5e6a6faaf1b9cb3ca77a59.json'
        self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), path.stem)
        self.assertEqual(hashlib.sha256(request_path.read_bytes()).hexdigest(), request_path.stem)
        raw = path.read_bytes(); candidate = json.loads(raw); request = json.loads(request_path.read_text())
        self.assertEqual(len(candidate['generation_prompt']), 845)
        with self.assertRaisesRegex(ValueError, 'string length'): _validate(candidate, PLAN_SCHEMA)
        with self.assertRaisesRegex(ValueError, 'Invalid plan generation_prompt'): validate_plan(candidate, request)
        self.assertEqual(hashlib.sha256(path.read_bytes()).digest(), hashlib.sha256(raw).digest())

    def test_attachment_guidance_preserves_distinct_independent_motion(self):
        prompt = plan_prompt(REQUEST)
        self.assertIn('Preserving an existing attached accessory while it is carried with the body belongs in appearance_checks', prompt)
        self.assertIn('Staying perched relative to a moving body is not world-space stability', prompt)
        self.assertIn('If the companion must fly, flap or otherwise move independently', prompt)
        self.assertIn('A ready plan must never contain interaction or expression requirement kinds', prompt)

    def test_attached_object_appearance_can_share_moving_role_but_interaction_cannot(self):
        candidate = fixture()
        candidate['appearance_checks'].append('The existing attached lens cover stays carried by the optical tube.')
        candidate['states'][0]['roles'][0]['description'] = 'Optical tube and its attached lens cover'
        self.assertEqual(validate_plan(candidate, REQUEST), candidate)
        for kind in ('interaction', 'expression'):
            trial = deepcopy(candidate)
            trial['states'][0]['requirements'][0]['kind'] = kind
            with self.assertRaisesRegex(ValueError, 'unsupported interaction or expression'):
                validate_plan(trial, REQUEST)

    def test_analytic_copy_of_saved_plan_exposes_secondary_interaction_failure(self):
        folder = Path(__file__).resolve().parents[1]/'evidence/accepted-source-plan-failure-v1/payload'
        path = folder/'b43896be47682e48fcffa009d80f97823b21ce22f3ada652310b262678915928.json'
        request_path = folder/'2713f0d5710ac0b14a54fbf1a3423f0a9a7c03043d5e6a6faaf1b9cb3ca77a59.json'
        self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), path.stem)
        self.assertEqual(hashlib.sha256(request_path.read_bytes()).hexdigest(), request_path.stem)
        raw = path.read_bytes(); original = json.loads(raw)
        analytic = deepcopy(original)
        analytic['generation_prompt'] = 'Analytic fixture only: a rounded panda with an existing perched green companion.'
        request = json.loads(request_path.read_text())
        _validate(analytic, PLAN_SCHEMA)
        with self.assertRaisesRegex(ValueError, 'unsupported interaction or expression'):
            validate_plan(analytic, request)
        self.assertEqual(len(original['generation_prompt']), 845)
        self.assertEqual(path.read_bytes(), raw)
