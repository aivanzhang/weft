"""Analytic source-shader admission; no Blender process or appearance claim."""
import ast
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from scripts import fur_root_sampling, groom_flow


class UVLayers(list):
    pass


class ExactRootAdapterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = Path(__file__).resolve().parents[1]/'scripts/blender_finish_generated.py'
        tree = ast.parse(path.read_text())
        helpers = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in (
            'texture_colors', 'shared_color_texture', 'exact_root_texture_contract', 'tuft_sampling_context')]
        cls.scope = {'np': np}
        exec(compile(ast.Module(body=helpers, type_ignores=[]), str(path), 'exec'), cls.scope)

    def fixture(self):
        pixels = np.array([[[1, 0, 0, 1], [0, 1, 0, 1]],
                           [[0, 0, 1, 1], [1, 1, 1, 1]]], dtype=np.float32)
        image = SimpleNamespace(size=(2, 2), colorspace_settings=SimpleNamespace(name='sRGB'),
            pixels=SimpleNamespace(foreach_get=Mock(side_effect=lambda output: output.__setitem__(slice(None), pixels.ravel()))))
        uv = SimpleNamespace(name='BodyUV', active_render=True,
            data=[SimpleNamespace(uv=point) for point in [(.25, .25)]*3+[(.75, .75)]*3])
        layers = UVLayers([uv]);layers.active = uv
        materials, textures = [], []
        for _ in range(2):
            texture = SimpleNamespace(type='TEX_IMAGE', image=image, interpolation='Linear',
                extension='REPEAT', projection='FLAT', inputs={'Vector': SimpleNamespace(links=[])})
            shader = SimpleNamespace(type='BSDF_PRINCIPLED', inputs={
                'Base Color': SimpleNamespace(links=[SimpleNamespace(from_node=texture)])})
            materials.append(SimpleNamespace(use_nodes=True, node_tree=SimpleNamespace(nodes=[shader, texture])))
            textures.append(texture)
        triangles = [SimpleNamespace(vertices=(0, 1, 2), loops=(0, 1, 2), material_index=0),
                     SimpleNamespace(vertices=(0, 1, 2), loops=(3, 4, 5), material_index=1)]
        obj = SimpleNamespace(data=SimpleNamespace(materials=materials, uv_layers=layers,
            vertices=[SimpleNamespace(co=value) for value in ((0, 0, 0), (1, 0, 0), (0, 1, 1))],
            loops=[SimpleNamespace(vertex_index=i % 3, index=i) for i in range(6)]))
        return obj, image, triangles, textures

    def modules(self):
        return patch.dict(sys.modules, {'fur_root_sampling': fur_root_sampling, 'groom_flow': groom_flow})

    def context(self, obj, image, triangles):
        return self.scope['tuft_sampling_context'](obj, image, triangles,
            {'direction': [0, 0, -1], 'comb': .6}, {'regions': [
                {'name': 'surface', 'center': [.5]*3, 'radius': [.6]*3, 'feather': .5,
                 'direction': [0, 0, -1], 'comb': .6, 'length_scale': 1.}]})

    def test_valid_shader_preserves_numeric_samples_and_reports_all_source_slots(self):
        obj, image, triangles, _ = self.fixture()
        with self.modules():
            sampler, _, report = self.context(obj, image, triangles)
        self.assertEqual(sampler(0, (1, 0, 0)), (1., 0., 0.))
        self.assertEqual(sampler(1, (1, 0, 0)), (1., 1., 1.))
        self.assertEqual(report['sampling_policy'], fur_root_sampling.ROOT_COLOR_POLICY)
        self.assertEqual(report['source_contract']['material_slots'], [0, 1])
        self.assertEqual(report['source_contract']['sampled_uv_layer'], 'BodyUV')
        image.pixels.foreach_get.assert_called_once()

    def test_first_or_last_non_linear_material_rejects_before_any_pixel_buffer(self):
        for mode in ('Closest', 'Cubic', 'Smart'):
            for slot in (0, 1):
                obj, image, triangles, textures = self.fixture()
                textures[slot].interpolation = mode
                with self.subTest(mode=mode, slot=slot), self.modules(), patch.object(np, 'empty') as allocate:
                    with self.assertRaisesRegex(ValueError, 'Linear/REPEAT'):
                        self.context(obj, image, triangles)
                    allocate.assert_not_called()
                image.pixels.foreach_get.assert_not_called()
                self.assertEqual(textures[slot].interpolation, mode)

    def test_uv_mismatch_rejects_before_the_initial_mask_texture_buffer(self):
        obj, image, _, _ = self.fixture()
        obj.data.uv_layers.active.active_render = False
        obj.data.uv_layers.append(SimpleNamespace(name='OtherUV', active_render=True))
        with self.modules(), patch.object(np, 'empty') as allocate:
            with self.assertRaisesRegex(ValueError, 'sampled/render-active UV'):
                self.scope['texture_colors'](obj, image, exact_root=True)
            allocate.assert_not_called()
        image.pixels.foreach_get.assert_not_called()

    def test_direct_uv_wrap_and_image_requirements_reject_before_root_sampling(self):
        for defect in ('linked', 'wrap', 'image', 'no_render_uv', 'two_render_uvs'):
            obj, image, triangles, textures = self.fixture()
            if defect == 'linked': textures[0].inputs['Vector'].links = [object()]
            if defect == 'wrap': textures[0].extension = 'EXTEND'
            if defect == 'image': textures[0].image = object()
            if defect == 'no_render_uv': obj.data.uv_layers.active.active_render = False
            if defect == 'two_render_uvs': obj.data.uv_layers.append(SimpleNamespace(name='Other', active_render=True))
            with self.subTest(defect=defect), self.modules(), patch.object(np, 'empty') as allocate:
                with self.assertRaises(ValueError):
                    self.context(obj, image, triangles)
                allocate.assert_not_called()
            image.pixels.foreach_get.assert_not_called()

    def test_exact_root_recheck_ignores_materials_without_contributing_triangles(self):
        obj, image, triangles, textures = self.fixture()
        textures[1].interpolation = 'Closest'
        with self.modules():
            sampler, _, report = self.context(obj, image, triangles[:1])
        self.assertEqual(sampler(0, (1, 0, 0)), (1., 0., 0.))
        self.assertEqual(report['source_contract']['material_slots'], [0])

    def test_legacy_admission_and_mask_colors_remain_unchanged(self):
        obj, image, _, textures = self.fixture()
        textures[0].interpolation = 'Closest'
        self.assertIs(self.scope['shared_color_texture'](obj.data.materials), textures[-1])
        colors = self.scope['texture_colors'](obj, image)
        np.testing.assert_array_equal(colors, [[1., 0, 0]]*3)
        image.pixels.foreach_get.assert_called_once()
        self.assertEqual(textures[0].interpolation, 'Closest')


if __name__ == '__main__':
    unittest.main()
