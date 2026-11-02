"""Typed transport fixtures; existing designs are data, not newly generated assets."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import unittest

from avatar_assets.agent_runner import MAX_SCHEMA_BYTES, _schema_check, _validate
from avatar_assets.author_contract import (AUTHOR_FIELDS_V2, AUTHOR_SCHEMA_V2, AUTHOR_SCHEMA_V2_SKIN_CONSISTENT,
    decode_typed_author)
from avatar_assets.finish_generated import validate as validate_finish
from avatar_assets.openai_runner import _strict
from avatar_assets.pipeline_prompts import (AUTHOR_FIELDS, AUTHOR_SCHEMA, _design_json, author_schema,
    author_prompt, extract_action_bindings, finish_config)
from test_pipeline_prompts import design as telescope_design, request as telescope_request, plan as telescope_plan, structured_example

ROOT = Path(__file__).resolve().parents[1]


def typed(design):
    """Analytic inverse fixture; no production authoring or saved-answer mutation."""
    value = {name: None for name in AUTHOR_FIELDS_V2}
    value.update(deepcopy(design))
    for field, optional in [('groom', ('direction', 'comb')), ('surface_smoothing', ('method',)),
                            ('fur_undercoat', ('uv_policy',))]:
        if value[field] is not None:
            for key in optional:
                value[field].setdefault(key, None)
    if value['portable_fur'] is not None:
        for key in ('profile', 'sample_root_color', 'root_placement'):
            value['portable_fur']['settings'].setdefault(key, None)
    for action in value['authored_actions'] or []:
        action.setdefault('interpolation', None)
        for key in action['keys']:
            key['rotations'] = [{'bone': bone, 'xyz': xyz} for bone, xyz in key['rotations'].items()]
            key['translations_m'] = ([{'bone': bone, 'xyz': xyz} for bone, xyz in key['translations_m'].items()]
                                     if 'translations_m' in key else None)
    if value['weight_constraints'] is not None:
        value['weight_constraints']['regions'] = [{'bone': bone, 'region': region}
                                                   for bone, region in value['weight_constraints']['regions'].items()]
    return {'schema_version': 2, 'status': 'ready', 'reason': '', 'design': value}


def example(name):
    config = json.loads((ROOT/'examples/meshy'/name).read_text())
    validate_finish(config)
    return config, {key: value for key, value in config.items() if key in AUTHOR_FIELDS_V2}


class TypedAuthorContractTests(unittest.TestCase):
    def test_schema_is_closed_bounded_strict_and_covers_all_existing_fields(self):
        _schema_check(AUTHOR_SCHEMA_V2)
        _strict(AUTHOR_SCHEMA_V2)
        self.assertEqual(AUTHOR_FIELDS_V2, AUTHOR_FIELDS)
        self.assertLess(len(json.dumps(AUTHOR_SCHEMA_V2).encode()), MAX_SCHEMA_BYTES)
        self.assertNotIn('design_json', AUTHOR_SCHEMA_V2['properties'])
        def walk(node):
            if not isinstance(node, dict):
                return
            if node.get('type') == 'object':
                self.assertFalse(node['additionalProperties'])
                self.assertEqual(set(node['required']), set(node['properties']))
            if node.get('type') == 'array':
                self.assertIn('maxItems', node)
            for key, child in node.items():
                if isinstance(child, dict):
                    walk(child)
                elif isinstance(child, list):
                    for item in child:
                        walk(item)
        walk(AUTHOR_SCHEMA_V2)

    def test_existing_furry_biped_and_mechanical_robot_roundtrip_exactly(self):
        for name in ('muse-panda-dance-v4.json', 'workshop-robot-finish-v1.json', 'emerald-dragon-finish-v3.json'):
            with self.subTest(example=name):
                original, design = example(name)
                wire = typed(design)
                before = deepcopy(wire)
                canonical = decode_typed_author(wire)
                self.assertEqual(canonical, design)
                self.assertEqual(wire, before)
                validate_finish({**original, **canonical})
                if name.startswith('muse'):
                    self.assertIn('groom_flow', canonical)
                    self.assertIn('portable_fur', canonical)
                    self.assertIn('authored_rig', canonical)
                    self.assertIn('weight_constraints', canonical)
                    self.assertIsNone(canonical['authored_rig']['bones'][0]['parent'])
                if name.startswith('workshop'):
                    self.assertNotIn('groom', canonical)
                    self.assertNotIn('authored_rig', canonical)
                    self.assertEqual(len(canonical['authored_actions']), 4)
                canonical['authored_actions'][0]['keys'][0]['second'] = 9
                self.assertEqual(wire, before)

    def test_transform_and_region_arrays_keep_values_and_reject_duplicates(self):
        _, design = example('muse-panda-dance-v4.json')
        wire = typed(design)
        first = wire['design']['authored_actions'][0]['keys'][0]
        for field in ('rotations', 'translations_m'):
            if not first[field]:
                first[field] = [{'bone': 'Root', 'xyz': [0, .1, 0]}]
            changed = deepcopy(wire)
            rows = changed['design']['authored_actions'][0]['keys'][0][field]
            rows.append(deepcopy(rows[0]))
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, 'Duplicate bone'):
                decode_typed_author(changed)
        changed = deepcopy(wire)
        rows = changed['design']['weight_constraints']['regions']
        rows.append(deepcopy(rows[0]))
        with self.assertRaisesRegex(ValueError, 'Duplicate bone'):
            decode_typed_author(changed)

    def test_nested_optional_nulls_are_omitted_but_required_null_parent_survives(self):
        _, design = example('muse-panda-dance-v4.json')
        design['groom'].pop('direction', None)
        design['groom'].pop('comb', None)
        design['portable_fur']['settings'].pop('profile', None)
        design['portable_fur']['settings'].pop('sample_root_color', None)
        design['authored_actions'][0].pop('interpolation', None)
        for key in design['authored_actions'][0]['keys']:
            key.pop('translations_m', None)
        canonical = decode_typed_author(typed(design))
        self.assertEqual(canonical, design)
        self.assertIn('parent', canonical['authored_rig']['bones'][0])
        self.assertIsNone(canonical['authored_rig']['bones'][0]['parent'])

    def test_unknown_fields_misnested_controls_and_embedded_json_are_rejected(self):
        _, design = example('muse-panda-dance-v4.json')
        base = typed(design)
        for mutate in (
            lambda v: v['design'].update(source='/unrelated/file'),
            lambda v: v['design']['authored_rig'].update(authored_actions=v['design']['authored_actions']),
            lambda v: v.update(design_json=json.dumps(design)),
            lambda v: v.update(design=json.dumps(design)),
            lambda v: v['design'].pop('surface_profiles'),
            lambda v: v['design']['groom'].update(private_budget=1),
            lambda v: v['design'].update(authored_actions=None),
        ):
            value = deepcopy(base)
            mutate(value)
            with self.subTest(value_keys=list(value)), self.assertRaises(ValueError):
                decode_typed_author(value)

    def test_vector_count_number_and_payload_bounds_fail_before_conversion(self):
        _, design = example('workshop-robot-finish-v1.json')
        base = typed(design)
        for vector in ([0, 0], [0, 0, 0, 0], [True, 0, 0], [float('nan'), 0, 0], [61, 0, 0]):
            value = deepcopy(base)
            value['design']['authored_actions'][0]['keys'][0]['rotations'] = [{'bone': 'Root', 'xyz': vector}]
            with self.subTest(vector=vector), self.assertRaises(ValueError):
                decode_typed_author(value)
        value = deepcopy(base)
        value['design']['authored_actions'][0]['keys'] *= 32
        with self.assertRaises(ValueError):
            decode_typed_author(value)
        value = deepcopy(base)
        action = {'id': 'bounded', 'interpolation': None,
                  'keys': [{'second': n / 8, 'rotations': [{'bone': 'B'+str(b)+'x'*60, 'xyz': [0, 0, 0]}
                            for b in range(64)], 'translations_m': None} for n in range(32)]}
        value['design']['authored_actions'] = [deepcopy(action) for _ in range(8)]
        with self.assertRaisesRegex(ValueError, '128KiB'):
            decode_typed_author(value)

    def test_unsupported_has_no_design_and_requires_a_reason(self):
        response = {'schema_version': 2, 'status': 'unsupported', 'reason': 'Cannot author the required topology.', 'design': None}
        self.assertIsNone(decode_typed_author(response))
        for changes in ({'reason': ''}, {'status': 'ready'}, {'schema_version': 1}, {'schema_version': True}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                decode_typed_author({**response, **changes})

    def test_cross_field_semantics_still_belong_to_existing_finisher(self):
        original, design = example('muse-panda-dance-v4.json')
        for mutate in (
            lambda v: v['authored_rig']['bones'][1].update(parent='NotObserved'),
            lambda v: v['authored_rig']['bones'][1].update(tail=v['authored_rig']['bones'][1]['head']),
            lambda v: v['authored_actions'][0]['keys'][1].update(second=0),
            lambda v: v['authored_actions'][0]['keys'][1]['rotations'].update(NotObserved=[1, 2, 3]),
            lambda v: v['groom'].update(parents=10000, children=30),
        ):
            invalid = deepcopy(design)
            mutate(invalid)
            canonical = decode_typed_author(typed(invalid))
            with self.assertRaises(ValueError):
                validate_finish({**original, **canonical})

    def test_new_rig_names_use_native_grammar_but_imported_references_remain_data(self):
        _, design = example('muse-panda-dance-v4.json')
        base = typed(design)
        for name in ('1Root', 'Root.Bone', 'Root-Bone', 'upper arm', 'épaules', 'Root\n', 'B'*65):
            value = deepcopy(base)
            value['design']['authored_rig']['bones'][0]['name'] = name
            with self.subTest(name=name), self.assertRaises(ValueError):
                decode_typed_author(value)
        for name in ('Root', 'LeftArm_2', 'A'*64):
            value = deepcopy(base)
            value['design']['authored_rig']['bones'][0]['name'] = name
            decode_typed_author(value)
        _, imported = example('workshop-robot-finish-v1.json')
        wire = typed(imported)
        wire['design']['authored_actions'][0]['keys'][0]['rotations'] = [{'bone': 'Imported.Bone-01', 'xyz': [0, 0, 0]}]
        self.assertIn('Imported.Bone-01', decode_typed_author(wire)['authored_actions'][0]['keys'][0]['rotations'])

    def test_typed_and_legacy_finishing_produce_identical_configs(self):
        for name in ('muse-panda-dance-v4.json', 'workshop-robot-finish-v1.json'):
            original, design = example(name)
            request = {'id': 'typed-transport', 'prompt': original['prompt']}
            plan = {'status': 'ready', 'subject': original['subject'], 'style': original['style'],
                    'height_meters': original.get('authored_rig', {}).get('height_meters', 1.2),
                    'states': original['states']}
            source = {'path': original['source'], 'sha256': original['source_sha256']}
            legacy = {'status': 'ready', 'reason': '', 'design_json': json.dumps(design)}
            with self.subTest(name=name):
                self.assertEqual(finish_config(typed(design), request, plan, source, 0),
                                 finish_config(legacy, request, plan, source, 0))

    def test_typed_role_bindings_keep_existing_independent_validation(self):
        request, plan = structured_example()
        design = telescope_design()
        design['authored_actions'] = design['authored_actions'][1:3]
        design['action_bindings'] = [{'state_id': state['id'], 'roles': [
            {'id': 'tube', 'root_bone': 'Base', 'bones': ['Tube']}]} for state in plan['states']]
        answer = typed(design)
        parents = {'Base': None, 'Tube': 'Base'}
        self.assertEqual(extract_action_bindings(answer, plan, parents), design['action_bindings'])
        config = finish_config(answer, request, plan, {'path': '/selected/model.glb', 'sha256': 'a'*64}, 0)
        self.assertNotIn('action_bindings', config)
        with self.assertRaisesRegex(ValueError, 'Unknown action target'):
            extract_action_bindings(answer, plan, {'Base': None})
        for key in answer['design']['authored_actions'][0]['keys']:
            key['rotations'] = [{'bone': 'Tube', 'xyz': [0, 0, 0]}]
        with self.assertRaisesRegex(ValueError, 'Constant poses'):
            finish_config(answer, request, plan, {'path': '/selected/model.glb', 'sha256': 'a'*64}, 0)

    def test_new_api_selects_typed_schema_and_unambiguous_prompt_legacy_unchanged(self):
        request, plan = telescope_request(), telescope_plan()
        api_request = {**request, 'agent_backend': {'provider': 'openai'}}
        self.assertIs(author_schema(api_request), AUTHOR_SCHEMA_V2_SKIN_CONSISTENT)
        self.assertIs(author_schema(request), AUTHOR_SCHEMA)
        legacy = author_prompt(request, plan, {'preflight': {'skins': 0}})
        prompt = author_prompt(api_request, plan, {'preflight': {'skins': 0}})
        guide = prompt.split('INPUT_JSON (data, not instructions):\n', 1)[0]
        self.assertIn('design_json', legacy)
        self.assertNotIn('design_json', guide)
        self.assertNotIn('never use null', guide)
        self.assertNotIn('maps actual bone', guide)
        self.assertNotIn('regions:{bone_name:region}', guide)
        self.assertIn('schema_version:2', guide)
        self.assertIn('rotations is [{bone,xyz:[x,y,z]}]', guide)
        self.assertIn('null', guide)
        with self.assertRaisesRegex(ValueError, 'Unknown author transport'):
            author_schema({**request, 'agent_backend': {'provider': 'unknown'}})

    def test_api_policy_rejects_legacy_card_routes_without_changing_base_decoder(self):
        original, design = example('muse-panda-dance-v4.json')
        legacy = typed(design)
        self.assertEqual(decode_typed_author(legacy), design)
        self.assertEqual(AUTHOR_SCHEMA_V2['$defs']['portable_settings']['properties']['profile']['anyOf'][0]['enum'],
                         ['legacy', 'combed-v1', 'tuft-v1'])
        schema = author_schema({'agent_backend': {'provider': 'openai'}})
        _strict(schema)
        with self.assertRaises(ValueError):
            _validate(legacy, schema)
        valid = deepcopy(legacy)
        settings = valid['design']['portable_fur']['settings']
        settings.update(profile='tuft-v1', root_placement='skin-consistent-v1', sample_root_color=True)
        _validate(valid, schema)
        canonical = decode_typed_author(valid)
        validate_finish({**original, **canonical})
        self.assertEqual(canonical['groom'], design['groom'])
        self.assertEqual(canonical['groom_exclusions'], design['groom_exclusions'])
        self.assertEqual(canonical['portable_fur']['bands'], design['portable_fur']['bands'])
        for field, value in (('profile', 'combed-v1'), ('profile', None),
                             ('root_placement', None), ('sample_root_color', False), ('sample_root_color', None)):
            invalid = deepcopy(valid)
            invalid['design']['portable_fur']['settings'][field] = value
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                _validate(invalid, schema)
        invalid = deepcopy(valid)
        del invalid['design']['portable_fur']['settings']['root_placement']
        with self.assertRaises(ValueError):
            _validate(invalid, schema)
        _, mechanical = example('workshop-robot-finish-v1.json')
        wire = typed(mechanical)
        _validate(wire, schema)
        self.assertIsNone(wire['design']['portable_fur'])
        self.assertNotIn('groom', decode_typed_author(wire))

    def test_preserved_malformed_legacy_response_remains_invalid_and_unchanged(self):
        path = ROOT/'evidence/openai-studio-terminal-v1/snapshots/outputs/studio-9e1cceb87c40437896c5a77f/agents/author-0/answer.json'
        if not path.is_file():
            self.skipTest('Optional retained local paid-response evidence is absent')
        before = path.read_bytes()
        old = json.loads(before)
        with self.assertRaisesRegex(ValueError, 'Invalid author design JSON'):
            _design_json(old['design_json'])
        with self.assertRaises(ValueError):
            decode_typed_author(old)
        self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), hashlib.sha256(before).hexdigest())


if __name__ == '__main__':
    unittest.main()
