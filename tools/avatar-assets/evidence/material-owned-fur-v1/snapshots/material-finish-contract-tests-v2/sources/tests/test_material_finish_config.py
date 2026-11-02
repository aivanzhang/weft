"""Keep opt-in material fur explicit and bounded before any native process."""

from copy import deepcopy
import unittest

from avatar_assets.finish_generated import validate


def fixture():
    return {'schema_version': 1, 'id': 'material-fixture', 'subject': 'Analytic furry regions',
            'style': 'Analytic configuration', 'rights': 'No generated asset in this test',
            'prompt': 'Exercise bounded opt-in material-owned fur configuration.',
            'source': 'unused.glb', 'source_sha256': '0'*64, 'normal_strength': 1,
            'resolution': 720, 'samples': 8, 'states': [{'id': 'idle', 'meaning': 'Analytic configuration only'}],
            'material_fur': {'version': 1, 'materials': ['Coat', 'Cream']},
            'groom': {'parents': 32, 'children': 30, 'length': .015, 'radius': .0002,
                      'roughness': .001, 'seed': 11, 'direction': [0, 0, -1], 'comb': .2},
            'groom_flow': {'regions': [{'name': 'body', 'center': [.5]*3, 'radius': [.5]*3,
                                      'direction': [0, 0, -1], 'comb': .2, 'length_scale': 1, 'feather': .5}]}}


class MaterialFinishConfigTests(unittest.TestCase):
    def test_explicit_material_names_and_global_minimum_are_admitted(self):
        value = fixture()
        self.assertEqual(validate(deepcopy(value)), value)
        value['normal_strength'] = 'preserve'
        validate(value)
        del value['material_fur']
        with self.assertRaisesRegex(ValueError, 'explicit material-owned'): validate(value)

    def test_missing_duplicate_or_unversioned_material_ownership_rejects(self):
        for material in ({'version': True, 'materials': ['Coat']}, {'version': 2, 'materials': ['Coat']},
                         {'version': 1, 'materials': []}, {'version': 1, 'materials': ['Coat', 'Coat']},
                         {'version': 1, 'materials': ['Coat'], 'fallback': 'all'}):
            with self.subTest(material=material), self.assertRaises(ValueError):
                config = fixture(); config['material_fur'] = material; validate(config)

    def test_parent_minimum_and_render_override_are_global_not_per_region(self):
        config = fixture(); config['groom']['parents'] = 31
        with self.assertRaisesRegex(ValueError, 'at least16'): validate(config)
        config = fixture(); config['groom']['parents'] = 5000; config['native_render_children'] = 48
        with self.assertRaisesRegex(ValueError, 'aggregate240000'): validate(config)
        config['native_render_children'] = 47
        validate(config)
        del config['material_fur']; config['native_render_children'] = 90
        validate(config)  # The older dense single-source profile remains unchanged.

    def test_material_mutations_cannot_silently_invalidate_color_ownership(self):
        for name in ('surface_profiles', 'color_corrections', 'fur_undercoat', 'normal_continuity',
                     'attachment', 'portable_fringe', 'smoothing', 'surface_smoothing', 'weight_constraints'):
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, 'repair controls'):
                config = fixture(); config[name] = {'enabled_fixture': True}; validate(config)

    def test_native_flow_and_exact_portable_roots_are_required(self):
        config = fixture(); del config['groom_flow']
        with self.assertRaisesRegex(ValueError, 'shared regional flow'): validate(config)
        config = fixture()
        config['portable_fur'] = {'settings': {'seed': 11, 'length': .015, 'width': .004,
                                             'direction': [0, 0, -1], 'sample_root_color': True,
                                             'profile': 'tuft-v1'},
                                  'bands': [{'brightness': [0, 1.01], 'color': [1]*4, 'count': 64}]}
        validate(config)
        config['portable_fur']['settings']['profile'] = 'combed-v1'
        with self.assertRaisesRegex(ValueError, 'exact-root'): validate(config)


if __name__ == '__main__':
    unittest.main()
