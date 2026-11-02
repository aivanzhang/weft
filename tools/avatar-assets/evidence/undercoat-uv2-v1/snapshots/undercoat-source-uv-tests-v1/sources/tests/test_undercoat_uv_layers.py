"""Pure and mocked UV preservation contracts; no native render or appearance evidence."""

from copy import deepcopy
import hashlib
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from scripts import blender_fur_undercoat as undercoat
from tests.test_undercoat_material_adapter import Images, budget, config, material, native_fixture


def source_uv1(obj):
    mesh = obj.data
    layer = mesh.uv_layers.new(name='SourceNormalMRUV1')
    values = np.asarray([value.uv for value in mesh.uv_layers[0].data], dtype=np.float32)
    values = np.column_stack((.8-values[:, 1]*.5, .1+values[:, 0]*.7)).astype(np.float32)
    layer.data.foreach_set('uv', values.ravel())
    layer.active_render = False
    mesh.uv_layers.active_index = 0
    mesh.uv_layers[0].active_render = True
    return values


def fake_bake(*_args, **_kwargs):
    return np.array([[[.5, .5, 1., 1.]]], dtype=np.float32), {'analytic_mock_not_png': True}


def apply(obj, mask, flow, out):
    with patch.dict(sys.modules, {'bpy': SimpleNamespace(data=SimpleNamespace(images=Images()))}), \
            patch.object(undercoat, 'bake_short_pile', side_effect=fake_bake):
        return undercoat.apply_undercoat(obj, mask, flow, config(), Path(out), export_vertex_budget=budget())


class UndercoatUVLayerTests(unittest.TestCase):
    def test_bounded_layer_planning_does_not_confuse_source_uv1_with_authored_uv(self):
        one = undercoat.plan_undercoat_uv_layers(['UV0'], 0, [True], ['UV0'])
        two = undercoat.plan_undercoat_uv_layers(['UV0', 'source-map'], 0, [True, False], ['UV0', 'source-map'])
        self.assertEqual((one['authored_index'], one['authored_name']), (1, 'AuthoredUndercoatUV1'))
        self.assertEqual((two['authored_index'], two['authored_name']), (2, 'AuthoredUndercoatUV2'))
        self.assertEqual(two['original_indices'], [0, 1])
        bad = [([], 0, [], []), (['u', 'v', 'w'], 0, [True, False, False], []),
               (['u', 'u'], 0, [True, False], []), (['u'], True, [True], []),
               (['u', 'v'], 1, [True, False], []), (['u', 'v'], 0, [False, True], []),
               (['u', 'v'], 0, [True, True], []), (['u'], 0, [1], []),
               ([''], 0, [True], []), (['u'*64], 0, [True], []),
               (['AuthoredUndercoatUV1'], 0, [True], []),
               (['u', 'AuthoredUndercoatUV2'], 0, [True, False], []),
               (['u'], 0, [True], ['AuthoredUndercoatUV2']),
               (['u'], 0, [True], ['x']*257)]
        for args in bad:
            with self.subTest(args=args), self.assertRaises(ValueError):
                undercoat.plan_undercoat_uv_layers(*args)

    def test_uv2_preserves_both_source_layers_and_reports_exact_authored_binding(self):
        obj, mask, flow = native_fixture()
        uv1 = source_uv1(obj)
        uv0 = np.asarray([value.uv for value in obj.data.uv_layers[0].data], dtype=np.float32)
        positions = [value.co for value in obj.data.vertices]
        groups = deepcopy([value.groups for value in obj.data.vertices])
        originals = obj.data.materials[:]
        with tempfile.TemporaryDirectory() as out:
            report = apply(obj, mask, flow, out)
        self.assertEqual(len(obj.data.uv_layers), 3)
        self.assertEqual(obj.data.uv_layers.active_index, 0)
        self.assertEqual([layer.active_render for layer in obj.data.uv_layers], [True, False, False])
        for index, before in enumerate((uv0, uv1)):
            actual = np.asarray([value.uv for value in obj.data.uv_layers[index].data], dtype=np.float32)
            np.testing.assert_array_equal(actual, before)
            row = report['original_uv_layers'][index]
            self.assertEqual(row['index'], index)
            self.assertEqual(row['sha256'], hashlib.sha256(before.tobytes()).hexdigest())
        self.assertEqual(report['authored_uv']['index'], 2)
        self.assertEqual(report['authored_uv']['name'], 'AuthoredUndercoatUV2')
        self.assertIsNone(report['authored_uv1_sha256'])
        authored = np.asarray([value.uv for value in obj.data.uv_layers[2].data], dtype=np.float32)
        self.assertEqual(report['authored_uv']['sha256'], hashlib.sha256(authored.tobytes()).hexdigest())
        self.assertFalse(np.array_equal(authored, uv1))
        self.assertEqual([row['normal_texcoord'] for row in report['materials']], [0, 2])
        self.assertEqual(report['plan']['authored_normal_texcoord'], 2)
        self.assertEqual([row['normal_texcoord'] for row in report['plan']['regions']], [0, 2])
        self.assertEqual(report['materials'][1]['normal_uv_map'], 'AuthoredUndercoatUV2')
        self.assertEqual(report['materials'][1]['selected_faces'], 4)
        self.assertEqual([value.co for value in obj.data.vertices], positions)
        self.assertEqual([value.groups for value in obj.data.vertices], groups)
        self.assertTrue(all(len(m.node_tree.nodes) == 4 for m in originals))
        for stale in obj.data.uv_layers.stale_handles:
            with self.assertRaises(ReferenceError):
                _ = stale.data

    def test_source_uv1_is_not_a_correction_target_when_all_selected_uv0_is_healthy(self):
        obj, mask, flow = native_fixture()
        uv1 = source_uv1(obj); mask[4:] = False
        with tempfile.TemporaryDirectory() as out:
            report = apply(obj, mask, flow, out)
        self.assertEqual(len(obj.data.uv_layers), 2)
        self.assertIsNone(report['authored_uv'])
        self.assertIsNone(report['authored_uv1_sha256'])
        self.assertIsNone(report['plan']['authored_normal_texcoord'])
        self.assertEqual([row['normal_texcoord'] for row in report['materials']], [0])
        np.testing.assert_array_equal(np.asarray([value.uv for value in obj.data.uv_layers[1].data]), uv1)

    def test_source_uv_layer_failures_precede_tangent_bake_or_material_mutation(self):
        for failure in ('nonfinite', 'oversized', 'uv2_exists', 'reserved_name', 'reserved_attribute', 'uv1_render'):
            obj, mask, flow = native_fixture(); source_uv1(obj)
            if failure == 'nonfinite': obj.data.uv_layers[1].data[0].uv = (float('nan'), .2)
            if failure == 'oversized': obj.data.uv_layers[1].data[0].uv = (1e7, .2)
            if failure == 'uv2_exists':
                obj.data.uv_layers.new(name='SourceUV2')
                obj.data.uv_layers.active_index = 0
                for index, layer in enumerate(obj.data.uv_layers): layer.active_render = index == 0
            if failure == 'reserved_name': obj.data.uv_layers[1].name = 'AuthoredUndercoatUV1'
            if failure == 'reserved_attribute': obj.data.attributes['AuthoredUndercoatUV2'] = object()
            if failure == 'uv1_render':
                obj.data.uv_layers[0].active_render = False; obj.data.uv_layers[1].active_render = True
            before = len(obj.data.uv_layers)
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as out, \
                    patch.dict(sys.modules, {'bpy': SimpleNamespace()}), \
                    patch.object(obj.data, 'calc_tangents') as tangents, \
                    patch.object(undercoat, 'bake_short_pile') as bake, self.assertRaises(ValueError):
                undercoat.apply_undercoat(obj, mask, flow, config(), Path(out), export_vertex_budget=budget())
            tangents.assert_not_called(); bake.assert_not_called()
            self.assertEqual(len(obj.data.materials), 2)
            self.assertEqual(len(obj.data.uv_layers), before)

    def test_uv1_data_change_during_new_layer_allocation_is_detected(self):
        obj, mask, flow = native_fixture(); source_uv1(obj)
        actual_new = obj.data.uv_layers.new
        def corrupt(*, name):
            layer = actual_new(name=name)
            obj.data.uv_layers[1].data[0].uv = (.75, .75)
            return layer
        with tempfile.TemporaryDirectory() as out, patch.object(obj.data.uv_layers, 'new', side_effect=corrupt), \
                self.assertRaisesRegex(ValueError, 'UV1 changed'):
            apply(obj, mask, flow, out)

    def test_linked_response_graph_and_uv1_route_are_preserved_only_normal_is_replaced(self):
        source = material('source')
        tree = source.node_tree; shader = tree.nodes[2]
        for name in ('Roughness', 'Metallic'):
            for link in list(shader.inputs[name].links): tree.links.remove(link)
        uv = tree.nodes.new('ShaderNodeUVMap'); uv.name = 'SourceMRUV'; uv.uv_map = 'SourceNormalMRUV1'
        texture = tree.nodes.new('ShaderNodeTexImage'); texture.name = 'SourceMR'; texture.image = object()
        tree.links.new(uv.outputs['UV'], texture.inputs['Vector'])
        for name in ('Roughness', 'Metallic'): tree.links.new(texture.outputs['Color'], shader.inputs[name])
        original_links = list(tree.links)
        policy = undercoat._response_policy(source, .92)
        clone = undercoat._clone_material(source, object(), 'AuthoredUndercoatUV2', .92)
        self.assertEqual(policy['roughness']['mode'], 'preserved_link')
        self.assertEqual(policy['metallic']['mode'], 'preserved_link')
        self.assertEqual(tree.links, original_links)
        nodes = clone.node_tree.nodes
        copied_texture = next(n for n in nodes if getattr(n, 'name', None) == 'SourceMR')
        copied_uv = next(n for n in nodes if getattr(n, 'name', None) == 'SourceMRUV')
        self.assertEqual(copied_uv.uv_map, 'SourceNormalMRUV1')
        self.assertIs(copied_texture.image, texture.image)
        self.assertIs(copied_texture.inputs['Vector'].links[0].from_socket, copied_uv.outputs['UV'])
        for name in ('Roughness', 'Metallic'):
            self.assertIs(nodes[2].inputs[name].links[0].from_socket, copied_texture.outputs['Color'])
            self.assertEqual(nodes[2].inputs[name].default_value, shader.inputs[name].default_value)
        self.assertEqual(nodes[-1].uv_map, 'AuthoredUndercoatUV2')
        self.assertEqual(len(nodes[2].inputs['Base Color'].links), 1)

    def test_unlinked_response_gets_explicit_constants_and_legacy_uv1_report_stays_true(self):
        source = material('source')
        shader = source.node_tree.nodes[2]
        for name in ('Roughness', 'Metallic'):
            for link in list(shader.inputs[name].links): source.node_tree.links.remove(link)
        clone = undercoat._clone_material(source, object(), 'AuthoredUndercoatUV1', .92)
        self.assertEqual(clone.node_tree.nodes[2].inputs['Roughness'].default_value, .92)
        self.assertEqual(clone.node_tree.nodes[2].inputs['Metallic'].default_value, 0)
        self.assertEqual(undercoat._response_policy(source, .92)['roughness']['authored_value'], .92)
        obj, mask, flow = native_fixture()
        with tempfile.TemporaryDirectory() as out:
            report = apply(obj, mask, flow, out)
        self.assertEqual(report['authored_uv']['index'], 1)
        self.assertEqual(report['authored_uv1_sha256'], report['authored_uv']['sha256'])
        self.assertEqual([r['normal_texcoord'] for r in report['materials']], [0, 1])
        self.assertEqual(report['materials'][0]['response_sockets']['roughness']['mode'], 'preserved_link')


if __name__ == '__main__':
    unittest.main()
