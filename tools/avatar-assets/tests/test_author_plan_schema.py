"""Plan-bound API transport rejects extra states; local validators still own order."""

from copy import deepcopy
import json
import unittest

from avatar_assets.agent_runner import MAX_SCHEMA_BYTES, _schema_check, _validate
from avatar_assets.action_intent import validate_action_bindings
from avatar_assets.author_contract import AUTHOR_SCHEMA_V2_SKIN_CONSISTENT
from avatar_assets.openai_runner import _strict
from avatar_assets.pipeline_prompts import AUTHOR_SCHEMA, author_schema, finish_config
from test_author_contract import typed
from test_pipeline_prompts import design, plan, request, structured_example


def api_request():
    return {**request(), 'agent_backend': {'provider': 'openai'}}


class PlanBoundAuthorSchemaTests(unittest.TestCase):
    def test_single_dance_rejects_rest_extra_and_missing_actions(self):
        candidate = plan()
        candidate['states'] = [{'id': 'dance', 'meaning': 'Dance.'}]
        value = design()
        value['authored_actions'] = value['authored_actions'][:1]
        value['authored_actions'][0]['id'] = 'dance'
        wire = typed(value)
        schema = author_schema(api_request(), candidate)
        _schema_check(schema); _strict(schema); _validate(wire, schema)
        for mutate in (lambda w: w['design']['authored_actions'][0].update(id='rest'),
                       lambda w: w['design']['authored_actions'].append(deepcopy(w['design']['authored_actions'][0])),
                       lambda w: w['design'].update(authored_actions=[]),
                       lambda w: w['design'].update(authored_actions=None)):
            bad = deepcopy(wire); mutate(bad)
            with self.assertRaises(ValueError):
                _validate(bad, schema)
        _validate({'schema_version': 2, 'status': 'unsupported', 'reason': 'Cannot author this motion.', 'design': None}, schema)

    def test_binding_ids_and_role_counts_are_bound_per_state(self):
        _, candidate = structured_example()
        candidate['states'][1]['roles'][0]['id'] = 'lens'
        value = design()
        value['authored_actions'] = value['authored_actions'][1:3]
        value['action_bindings'] = [
            {'state_id': state['id'], 'roles': [{'id': state['roles'][0]['id'], 'root_bone': 'Base', 'bones': ['Tube']}]}
            for state in candidate['states']]
        wire = typed(value)
        schema = author_schema(api_request(), candidate)
        _strict(schema); _validate(wire, schema)
        _validate({'schema_version': 2, 'status': 'unsupported', 'reason': 'Unsupported.', 'design': None}, schema)
        mutations = [lambda w: w['design']['action_bindings'][0].update(state_id='rest'),
                     lambda w: w['design']['action_bindings'][0]['roles'][0].update(id='lens'),
                     lambda w: w['design']['action_bindings'][0]['roles'].append(
                         deepcopy(w['design']['action_bindings'][0]['roles'][0])),
                     lambda w: w['design'].update(action_bindings=[]),
                     lambda w: w['design'].update(action_bindings=None),
                     lambda w: w['design']['action_bindings'].append(deepcopy(w['design']['action_bindings'][0]))]
        for mutate in mutations:
            bad = deepcopy(wire); mutate(bad)
            with self.assertRaises(ValueError):
                _validate(bad, schema)
        # The supported array schema cannot reject permutations; the authoritative gate must.
        for reorder in (lambda rows: rows.reverse(), lambda rows: rows.__setitem__(1, deepcopy(rows[0]))):
            bad = deepcopy(wire); reorder(bad['design']['action_bindings'])
            _validate(bad, schema)
            with self.assertRaises(ValueError):
                validate_action_bindings(bad['design']['action_bindings'], candidate['states'], {'Base': None, 'Tube': 'Base'})

    def test_action_order_and_duplicates_remain_native_rejections(self):
        candidate, req = plan(), api_request()
        schema = author_schema(req, candidate)
        for reorder in (lambda rows: rows.reverse(), lambda rows: rows.__setitem__(1, deepcopy(rows[0]))):
            wire = typed(design()); reorder(wire['design']['authored_actions'])
            _validate(wire, schema)
            with self.assertRaisesRegex(ValueError, 'match named states'):
                finish_config(wire, req, candidate, {'path': '/tmp/source.glb', 'sha256': 'a'*64}, 0)

    def test_base_schema_legacy_plan_and_optional_call_are_unchanged(self):
        original = deepcopy(AUTHOR_SCHEMA_V2_SKIN_CONSISTENT)
        candidate = plan(); before = deepcopy(candidate)
        schema = author_schema(api_request(), candidate)
        self.assertEqual(candidate, before)
        self.assertEqual(AUTHOR_SCHEMA_V2_SKIN_CONSISTENT, original)
        self.assertIs(author_schema(api_request()), AUTHOR_SCHEMA_V2_SKIN_CONSISTENT)
        self.assertIs(author_schema(request(), candidate), AUTHOR_SCHEMA)
        self.assertEqual(schema['$defs']['design']['properties']['action_bindings']['anyOf'][0]['maxItems'], 0)
        self.assertLess(len(json.dumps(schema).encode()), MAX_SCHEMA_BYTES)

    def test_maximum_plan_is_strict_bounded_and_malformed_plans_fail(self):
        candidate = plan()
        candidate['states'] = [{'id': 'state_'+str(i), 'requirements': [{'kind': 'articulation'}],
                               'roles': [{'id': 'role_'+str(j)} for j in range(4)]} for i in range(8)]
        schema = author_schema(api_request(), candidate)
        _schema_check(schema); _strict(schema)
        self.assertLess(len(json.dumps(schema).encode()), MAX_SCHEMA_BYTES)
        for mutate in (lambda p: p['states'].append(deepcopy(p['states'][0])),
                       lambda p: p['states'][1].update(id=p['states'][0]['id']),
                       lambda p: p['states'][0]['roles'][1].update(id='role_0'),
                       lambda p: p.update(status='unsupported'),
                       lambda p: p['states'][0].update(id='Bad state')):
            bad = deepcopy(candidate); mutate(bad)
            with self.assertRaises(ValueError):
                author_schema(api_request(), bad)


if __name__ == '__main__':
    unittest.main()
