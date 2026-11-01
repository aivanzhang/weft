"""Analytic finishing contracts; no native rendering or generated asset claims."""

import ast
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
import sys
import numpy as np

from avatar_assets.finish_generated import validate
from scripts import surface_integrity


def finish_config():
    return {'schema_version': 1, 'id': 'analytic-quality-contract',
            'subject': 'Analytic creature', 'style': 'Analytic fixture',
            'rights': 'Analytic fixture, not a generated asset', 'prompt': 'Test bounded finishing settings',
            'source': 'analytic.glb', 'source_sha256': '0'*64,
            'normal_strength': .6, 'resolution': 512, 'samples': 8,
            'states': [{'id': 'idle', 'meaning': 'Analytic idle state'}],
            'groom': {'parents': 16, 'children': 0, 'length': .01, 'radius': .0001,
                      'roughness': .001, 'seed': 3}}


def material(image, *, route=(), projection='FLAT', extension='REPEAT'):
    links = [SimpleNamespace(from_node=SimpleNamespace(type=kind, name=name, uv_map=uv_map),
                             from_socket=SimpleNamespace(name=socket))
             for kind, name, socket, uv_map in route]
    texture = SimpleNamespace(type='TEX_IMAGE', image=image, name='BaseColor', projection=projection,
                              extension=extension, inputs={'Vector': SimpleNamespace(links=links)})
    shader = SimpleNamespace(type='BSDF_PRINCIPLED', inputs={
        'Base Color': SimpleNamespace(links=[SimpleNamespace(from_node=texture)])})
    return SimpleNamespace(use_nodes=True, node_tree=SimpleNamespace(nodes=[shader, texture])), texture


class FinishQualityConfigTests(unittest.TestCase):
    def test_groom_shader_accepts_both_inclusive_roughness_limits(self):
        for roughness, radial in ((.04, 1), (1, .04)):
            settings = finish_config()
            settings['groom_shader'] = {'roughness': roughness, 'radial_roughness': radial}
            before = deepcopy(settings)
            validate(settings)
            self.assertEqual(settings, before)

    def test_groom_shader_requires_a_valid_native_groom(self):
        for groom in (None, {}, {'parents': 0, 'children': 0, 'length': .01, 'radius': .0001,
                                 'roughness': .001, 'seed': 3}):
            settings = finish_config()
            settings['groom'] = groom
            settings['groom_shader'] = {'roughness': .6, 'radial_roughness': .7}
            with self.subTest(groom=groom), self.assertRaises(ValueError):
                validate(settings)
        settings = finish_config()
        del settings['groom']
        settings['groom_shader'] = {'roughness': .6, 'radial_roughness': .7}
        with self.assertRaisesRegex(ValueError, 'requires native hair'):
            validate(settings)

    def test_groom_shader_requires_exactly_the_two_named_settings(self):
        for shader in (None, [], {}, {'roughness': .5}, {'radial_roughness': .5},
                       {'roughness': .5, 'radial_roughness': .5, 'specular': 1}):
            settings = finish_config()
            settings['groom_shader'] = shader
            with self.subTest(shader=shader), self.assertRaisesRegex(ValueError, 'two explicit roughness settings'):
                validate(settings)

    def test_both_groom_roughness_values_reject_nonfinite_boolean_and_out_of_bounds(self):
        for key in ('roughness', 'radial_roughness'):
            for value in (False, True, float('nan'), float('inf'), -float('inf'), 0, .039, 1.001, '0.5', None):
                settings = finish_config()
                settings['groom_shader'] = {'roughness': .5, 'radial_roughness': .5, key: value}
                with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                    validate(settings)

    def test_legacy_and_spatial_smoothing_are_individually_valid_but_exclusive(self):
        legacy = {'iterations': 4, 'strength': .2}
        spatial = {'iterations': 4, 'strength': .2, 'max_displacement_ratio': .01,
                   'feather': .25, 'exclusions': []}
        for key, value in (('smoothing', legacy), ('surface_smoothing', spatial),
                           ('surface_smoothing', {**spatial, 'method': 'laplacian'})):
            settings = finish_config()
            settings[key] = value
            validate(settings)
        for legacy_value in (legacy, {}, None):
            settings = finish_config()
            settings.update(smoothing=legacy_value, surface_smoothing=spatial)
            with self.subTest(legacy=legacy_value), self.assertRaisesRegex(ValueError, 'Choose one smoothing method'):
                validate(settings)


class SharedColorTextureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        source = Path(__file__).resolve().parents[1]/'scripts/blender_finish_generated.py'
        tree = ast.parse(source.read_text(), filename=str(source))
        helper = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'shared_color_texture')
        namespace = {}
        # Execute only the actual data helper, never the bpy imports or native entrypoint.
        exec(compile(ast.Module(body=[helper], type_ignores=[]), str(source), 'exec'), namespace)
        cls.shared_texture = staticmethod(namespace['shared_color_texture'])

    def test_identical_direct_active_uv_textures_are_accepted(self):
        image = object()
        first, _ = material(image)
        second, texture = material(image)
        self.assertIs(self.shared_texture([first, second]), texture)

    def test_different_images_are_rejected_even_with_identical_routes(self):
        first, _ = material(object())
        second, _ = material(object())
        with self.assertRaises(ValueError):
            self.shared_texture([first, second])

    def test_same_name_different_uv_layers_and_identical_explicit_routes_are_rejected(self):
        image = object()
        first_route = [('UVMAP', 'UV Map', 'UV', 'UV_Main')]
        for second_route in (first_route, [('UVMAP', 'UV Map', 'UV', 'UV_Alternate')]):
            first, _ = material(image, route=first_route)
            second, _ = material(image, route=second_route)
            with self.subTest(route=second_route), self.assertRaises(ValueError):
                self.shared_texture([first, second])
        mapped, _ = material(image, route=[('MAPPING', 'Mapping', 'Vector', None)])
        with self.assertRaises(ValueError):
            self.shared_texture([mapped])

    def test_nonflat_projection_and_nonrepeat_wrapping_are_rejected(self):
        for changes in ({'projection': 'BOX'}, {'projection': 'SPHERE'}, {'extension': 'CLIP'}, {'extension': 'EXTEND'}):
            source, _ = material(object(), **changes)
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                self.shared_texture([source])

    def test_missing_or_indirect_base_color_sources_are_rejected(self):
        untextured, _ = material(None)
        indirect, texture = material(object())
        texture.type = 'MIX_RGB'
        no_link, _ = material(object())
        no_link.node_tree.nodes[0].inputs['Base Color'].links.clear()
        ambiguous, _ = material(object())
        ambiguous.node_tree.nodes.append(deepcopy(ambiguous.node_tree.nodes[0]))
        for sources in ([], [None], [untextured], [indirect], [no_link], [ambiguous]):
            with self.subTest(count=len(sources)), self.assertRaises(ValueError):
                self.shared_texture(sources)


class StoredSurfaceTests(unittest.TestCase):
    def test_float32_collapse_is_rejected_before_writing_mesh(self):
        path = Path(__file__).resolve().parents[1]/'scripts/blender_finish_generated.py'
        tree = ast.parse(path.read_text())
        function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'set_surface_positions')
        write = Mock()
        scope = {'np': np, 'write_json': write}
        exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), 'exec'), scope)
        step = 2**-23
        original = np.array([[1.,0.,0.], [1.+step,0.,0.], [1.,1.,0.]])
        refined = original.copy()
        refined[:2,0] = [1.+.48*step, 1.+.499*step]
        triangles = np.array([[0,1,2]])
        self.assertEqual(surface_integrity.enforce_surface_change(original, refined, triangles)
                         ['regression_gate']['status'], 'passed')
        data = SimpleNamespace(calc_loop_triangles=Mock(), vertices=Mock(),
                               loop_triangles=[SimpleNamespace(vertices=(0,1,2))])
        with patch.dict(sys.modules, {'surface_integrity': surface_integrity}):
            with self.assertRaises(surface_integrity.SurfaceChangeError):
                scope['set_surface_positions'](SimpleNamespace(data=data), original, refined, Path('/unused'))
        data.vertices.foreach_set.assert_not_called()
        self.assertEqual(write.call_args.args[1]['new_degenerate_count'], 1)


if __name__ == '__main__':
    unittest.main()
