"""Rig descriptions must not infer a source's anatomy, provider, or generation route."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

from avatar_assets.assemble_generated import rig_contract
from avatar_assets.three_d import validate_glb
from tests.test_three_d import analytic_rig, glb_bytes


class RigContractMetadataTests(unittest.TestCase):
    def document(self):
        return {'nodes': [{'name': 'shade', 'translation': [.1, .2, .3]},
                          {'name': 'lamp_mesh', 'mesh': 0, 'skin': 0},
                          {'name': 'hinge', 'rotation': [0, 0, 0, 1]},
                          {'name': 'turntable', 'scale': [1, 1, 1]}],
                'skins': [{'joints': [3, 0, 2], 'inverseBindMatrices': 7},
                          {'joints': [2, 0], 'inverseBindMatrices': 8}]}

    def test_existing_object_rig_does_not_acquire_provider_or_humanoid_provenance(self):
        result = rig_contract(self.document())
        descriptions = ' '.join([result['rest_pose'], result['target']['name'], *result['authored_assumptions']]).lower()
        for unsupported in ('generated', 'humanoid', 'provider', 'creature'):
            self.assertNotIn(unsupported, descriptions)
        self.assertIn('not inferred', descriptions)
        self.assertFalse(result['weft_runtime_verified'])
        self.assertIn('NOT Weft', result['target']['name'])

    def test_authored_object_rig_discloses_local_rigging_not_mesh_origin(self):
        result = rig_contract(self.document(), authored=True)
        descriptions = ' '.join([result['rest_pose'], result['target']['name'], *result['authored_assumptions']]).lower()
        self.assertIn('locally authored', descriptions)
        for unsupported in ('generated', 'humanoid', 'provider', 'creature'):
            self.assertNotIn(unsupported, descriptions)
        self.assertFalse(result['weft_runtime_verified'])

    def test_joint_order_identity_mapping_and_all_behavioral_fields_are_preserved(self):
        doc = self.document()
        before = deepcopy(doc)
        expected = {'schema_version': 1, 'units': 'meters',
                    'axes': {'up': '+Y', 'forward': '+Z', 'handedness': 'right'},
                    'bone_to_simulated_joint': {'shade': 'shade', 'hinge': 'hinge', 'turntable': 'turntable'},
                    'target': {'joint_names': ['shade', 'hinge', 'turntable']},
                    'attachments': [], 'weft_runtime_verified': False}
        for authored in (False, True):
            with self.subTest(authored=authored):
                result = rig_contract(doc, authored=authored)
                self.assertEqual(set(result), set(expected)|{'rest_pose', 'authored_assumptions'})
                del result['rest_pose'], result['authored_assumptions'], result['target']['name']
                self.assertEqual(result, expected)
                self.assertEqual(doc, before)

    def test_explicit_assumption_override_is_kept_verbatim_without_input_mutation(self):
        assumptions = ['Source is a procedural articulated object.',
                       'Its hidden inner mechanism is authored and unvalidated.']
        original = deepcopy(assumptions)
        for authored in (False, True):
            with self.subTest(authored=authored):
                result = rig_contract(self.document(), assumptions, authored=authored)
                self.assertIs(result['authored_assumptions'], assumptions)
                self.assertEqual(result['authored_assumptions'], original)
        self.assertEqual(assumptions, original)

    def test_metadata_change_preserves_actual_analytic_skin_motion_verification(self):
        doc, binary, _ = analytic_rig()
        doc['nodes'][1]['name'], doc['nodes'][2]['name'] = 'turntable', 'shade_hinge'
        before = json.dumps(doc, sort_keys=True)
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder)/'authored-object-fixture.glb'
            original = glb_bytes(doc, binary)
            source.write_bytes(original)
            for authored in (False, True):
                with self.subTest(authored=authored):
                    contract = rig_contract(doc, authored=authored)
                    historical = deepcopy(contract)
                    historical['rest_pose'] = 'Default exported node transforms and inverse bind matrices; generated, not recovered anatomy'
                    historical['target']['name'] = 'Local provider bone controls, NOT Weft'
                    historical['authored_assumptions'] = ['Hidden surfaces and humanoid skeleton are generated assumptions']
                    old_result = validate_glb(source, historical)
                    result = validate_glb(source, contract)
                    self.assertTrue(result['loading_verified'])
                    self.assertTrue(result['motion']['animation_verified'])
                    for field in ('rest_pose', 'authored_assumptions'):
                        del result[field], old_result[field]
                    self.assertEqual(result, old_result)
                    self.assertEqual(source.read_bytes(), original)
        self.assertEqual(json.dumps(doc, sort_keys=True), before)


if __name__ == '__main__':
    unittest.main()
