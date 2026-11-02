"""Exercise the native sampling boundary without importing Blender or decoding files."""
import ast
from pathlib import Path
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock

import numpy as np


SOURCE = Path(__file__).resolve().parents[1]/'scripts/blender_finish_generated.py'
CONTROLS = ('groom', 'portable_fur', 'portable_fringe', 'fur_undercoat',
            'smoothing', 'surface_profiles', 'color_corrections', 'attachment')


class Pixels:
    def __init__(self):
        self.values = np.array([.1, .2, .3, 1, .4, .5, .6, 1,
                                .1, .8, .2, 1, .2, .4, .8, 1], dtype=np.float32)
        self.reads = 0

    def foreach_get(self, target):
        self.reads += 1
        target[:] = self.values


def image(name='Source', color_space='sRGB'):
    return NS(name=name, size=(2, 2), pixels=Pixels(),
              colorspace_settings=NS(name=color_space))


def material(texture=None, name='Material', vector_linked=False):
    node = NS(type='TEX_IMAGE', image=texture, projection='FLAT', extension='REPEAT',
              inputs={'Vector': NS(links=[object()] if vector_linked else [])})
    base = NS(default_value=(.25, .4, .6, 1), links=[NS(from_node=node)] if texture else [])
    shader = NS(type='BSDF_PRINCIPLED', inputs={'Base Color': base})
    return NS(name=name, use_nodes=True, node_tree=NS(nodes=[shader, node] if texture else [shader])), node


def emitter(materials):
    uv = [NS(uv=np.array(value)) for value in ((.1, .1), (.6, .1), (.1, .6))]
    return NS(data=NS(materials=materials, vertices=[object() for _ in uv],
                      loops=[NS(index=i, vertex_index=i) for i in range(len(uv))],
                      uv_layers=NS(active=NS(data=uv))))


def graph_identity(materials):
    return [(id(m), m.name, m.use_nodes,
             [(id(node), node.type,
               [(key, id(socket), tuple(id(link) for link in socket.links),
                 getattr(socket, 'default_value', None)) for key, socket in node.inputs.items()])
              for node in m.node_tree.nodes]) for m in materials]


class LazySourceColorSamplingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        tree = ast.parse(SOURCE.read_text(), filename=str(SOURCE))
        names = {'texture_colors', 'shared_color_texture', 'source_color_controls',
                 'source_color_samples', 'frame_mask'}
        helpers = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
        if len(helpers) != len(names):
            raise AssertionError('Actual native sampling helpers are missing')
        finish = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'finish')
        # Execute the actual selection block so wiring regressions cannot hide behind helper tests.
        first = next(i for i, node in enumerate(finish.body) if isinstance(node, ast.Assign)
                     and isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name)
                     and node.value.func.id == 'source_color_samples')
        end = next(i for i in range(first+1, len(finish.body)) if isinstance(finish.body[i], ast.Assign)
                   and any(isinstance(t, ast.Name) and t.id == 'smoothing' for t in finish.body[i].targets))
        stage = ast.parse('def selection_stage(obj, cfg, points, low, high):\n    pass\n').body[0]
        stage.body = finish.body[first:end] + ast.parse('return color_node, colors, mask, color_sampling').body
        cls.compiled = compile(ast.fix_missing_locations(ast.Module(body=helpers+[stage], type_ignores=[])), str(SOURCE), 'exec')

    def setUp(self):
        self.exact = Mock()
        self.namespace = {'np': np, 'exact_root_texture_contract': self.exact}
        exec(self.compiled, self.namespace)
        self.sample = self.namespace['source_color_samples']
        self.stage = self.namespace['selection_stage']

    def test_geometry_only_controls_skip_all_object_and_image_access(self):
        class ForbiddenObject:
            def __getattribute__(self, name):
                raise AssertionError('No-color sampling touched the source object')

        cfg = {'surface_smoothing': {'iterations': 3, 'strength': .2},
               'authored_rig': {'bones': ['Root']}, 'authored_actions': [{'id': 'hinge'}],
               'normal_continuity': [{'materials': ['Body']}], 'weight_constraints': {'regions': {}},
               'normal_strength': .2, 'groom_exclusions': [{'center': [0, 0, 0], 'radius': [.1]*3}],
               'fur_max_chroma': .2, 'export_model': True}
        node, colors, report = self.sample(ForbiddenObject(), cfg)
        self.assertIsNone(node)
        self.assertIsNone(colors)
        self.assertEqual(report['status'], 'skipped')
        self.assertEqual(report['required_by'], [])
        self.assertEqual(report['selection_sampling_calls'], 0)
        self.assertEqual(report['sampled_vertices'], 0)
        self.exact.assert_not_called()

    def test_actual_selection_block_preserves_distinct_textures_and_constant_material(self):
        first, second = image('First'), image('Second')
        materials = [material(first, 'A')[0], material(second, 'B', vector_linked=True)[0], material(name='Constant')[0]]
        obj = emitter(materials)
        before = graph_identity(materials)
        points = np.array([[0., 0, 0], [1, 0, 0], [0, 1, 1]])
        node, colors, mask, report = self.stage(obj, {'surface_smoothing': {'iterations': 3}}, points, points.min(0), points.max(0))
        self.assertIsNone(node)
        self.assertIsNone(colors)
        self.assertEqual(report['status'], 'skipped')
        self.assertTrue(mask.all())
        self.assertEqual(graph_identity(materials), before)
        self.assertEqual((first.pixels.reads, second.pixels.reads), (0, 0))
        self.exact.assert_not_called()

    def test_every_color_consumer_samples_actual_shared_texture_once(self):
        for control in CONTROLS:
            with self.subTest(control=control):
                source = image()
                first, _ = material(source, 'A')
                second, selected = material(source, 'B')
                obj = emitter([first, second])
                original = source.pixels.values.copy()
                node, colors, report = self.sample(obj, {control: {'enabled_fixture': True}})
                self.assertIs(node, selected)
                np.testing.assert_array_equal(colors, np.array([[.1, .2, .3], [.4, .5, .6], [.1, .8, .2]], dtype=np.float32))
                self.assertEqual(source.pixels.reads, 1)
                self.assertEqual(report['required_by'], [control])
                self.assertEqual(report['status'], 'sampled')
                self.assertEqual(report['sampled_vertices'], 3)
                self.assertEqual(report['selection_sampling_calls'], 1)
                colors[0] = 0
                np.testing.assert_array_equal(source.pixels.values, original)

    def test_each_consumer_retains_heterogeneous_image_rejection_before_decode(self):
        for control in CONTROLS:
            with self.subTest(control=control):
                first, second = image('First'), image('Second')
                obj = emitter([material(first)[0], material(second)[0]])
                with self.assertRaisesRegex(ValueError, 'Heterogeneous'):
                    self.sample(obj, {control: {'enabled_fixture': True}})
                self.assertEqual((first.pixels.reads, second.pixels.reads), (0, 0))

    def test_each_consumer_retains_constant_color_rejection_without_fallback(self):
        for control in CONTROLS:
            with self.subTest(control=control):
                source = image()
                obj = emitter([material(source)[0], material(name='Constant')[0]])
                with self.assertRaisesRegex(ValueError, 'directly textured'):
                    self.sample(obj, {control: {'enabled_fixture': True}})
                self.assertEqual(source.pixels.reads, 0)

    def test_required_sampling_does_not_admit_custom_uv_routes(self):
        source = image()
        obj = emitter([material(source, vector_linked=True)[0]])
        with self.assertRaisesRegex(ValueError, 'Custom UV routes'):
            self.sample(obj, {'attachment': {'enabled_fixture': True}})
        self.assertEqual(source.pixels.reads, 0)

    def test_combined_consumers_share_one_sampling_call_and_retain_chroma_mask(self):
        source = image()
        obj = emitter([material(source)[0]])
        points = np.array([[0., 0, 0], [1, 0, 0], [0, 1, 1]])
        cfg = {'groom': {'enabled_fixture': True}, 'smoothing': {'iterations': 2, 'strength': .1},
               'surface_profiles': [{'enabled_fixture': True}], 'fur_max_chroma': .25,
               'groom_exclusions': [{'center': [0, 0, 0], 'radius': [.1]*3}]}
        _, colors, mask, report = self.stage(obj, cfg, points, points.min(0), points.max(0))
        self.assertEqual(report['required_by'], ['groom', 'smoothing', 'surface_profiles'])
        self.assertEqual(source.pixels.reads, 1)
        self.assertEqual(colors.shape, (3, 3))
        np.testing.assert_array_equal(mask, [False, True, False])

    def test_root_colored_tuft_retains_srgb_and_exact_root_contract(self):
        cfg = {'portable_fur': {'settings': {'profile': 'tuft-v1', 'sample_root_color': True}}}
        bad = image(color_space='Linear')
        with self.assertRaisesRegex(ValueError, 'sRGB'):
            self.sample(emitter([material(bad)[0]]), cfg)
        self.assertEqual(bad.pixels.reads, 0)
        self.exact.assert_not_called()
        source = image()
        obj = emitter([material(source)[0]])
        _, _, report = self.sample(obj, cfg)
        self.exact.assert_called_once_with(obj, source)
        self.assertTrue(report['exact_root_contract'])

    def test_fringe_retains_exact_root_contract_and_texture_size_bound(self):
        source = image()
        obj = emitter([material(source)[0]])
        _, _, report = self.sample(obj, {'portable_fringe': {'profile': 'edge-fringe-v1'}})
        self.exact.assert_called_once_with(obj, source)
        self.assertTrue(report['exact_root_contract'])
        oversized = image()
        oversized.size = (4097, 4096)
        with self.assertRaisesRegex(ValueError, 'sampling limit'):
            self.sample(emitter([material(oversized)[0]]), {'groom': {'enabled_fixture': True}})
        self.assertEqual(oversized.pixels.reads, 0)

    def test_disabled_optional_controls_do_not_require_sampling(self):
        for disabled in (None, {}, []):
            with self.subTest(disabled=disabled):
                node, colors, report = self.sample(None, dict.fromkeys(CONTROLS, disabled))
                self.assertIsNone(node)
                self.assertIsNone(colors)
                self.assertEqual(report['status'], 'skipped')


if __name__ == '__main__':
    unittest.main()
