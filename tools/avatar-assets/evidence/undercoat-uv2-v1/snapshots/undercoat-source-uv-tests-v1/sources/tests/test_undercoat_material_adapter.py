"""Analytic planning and mocked native contracts; no render or appearance claims."""

from copy import deepcopy
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from scripts import blender_fur_undercoat as undercoat
from scripts import undercoat_uv_atlas as atlas
from tests.test_fur_undercoat import fixture as bake_fixture, settings
from tests.test_surface_profiles import Material, Links, socket
from tests.test_undercoat_uv_atlas import fixture


def config():
    return {**settings(), 'uv_policy': 'material-adaptive-v1'}


def budget(**changes):
    return {'source_export_vertices': 12, 'portable_export_vertices': 24, 'reserve_vertices': 4096, **changes}


def plan(args=None, **changes):
    return undercoat.plan_material_undercoat(*(fixture() if args is None else args), config(),
                                            export_vertex_budget=budget(**changes))


class NativeLinks(Links):
    def new(self, source, target):
        link = SimpleNamespace(from_socket=source, to_socket=target)
        self.append(link)
        target.links.append(link)


class NativeNodes(list):
    def new(self, kind):
        names = {'ShaderNodeUVMap': ('UVMAP', [], ['UV']),
                 'ShaderNodeTexImage': ('TEX_IMAGE', ['Vector'], ['Color']),
                 'ShaderNodeNormalMap': ('NORMAL_MAP', ['Color', 'Strength'], ['Normal'])}
        node_type, inputs, outputs = names[kind]
        node = SimpleNamespace(type=node_type, inputs={name: socket(name) for name in inputs},
                               outputs={name: socket(name) for name in outputs})
        self.append(node)
        return node


def material(name):
    result = Material(name)
    result.node_tree.nodes = NativeNodes(result.node_tree.nodes)
    result.node_tree.links = NativeLinks(result.node_tree.links)
    result.node_tree.nodes[2].inputs['Normal'] = socket('Normal')
    return result


class UVData(list):
    def foreach_set(self, name, values):
        assert name == 'uv'
        for item, value in zip(self, np.asarray(values).reshape(-1, 2)):
            item.uv = tuple(value)


class UVLayer:
    def __init__(self, name, active_render, data):
        self._values = {'name': name, 'active_render': active_render, 'data': data}
        self.valid = True

    def __getattr__(self, name):
        if name not in self._values:
            raise AttributeError(name)
        if not self.valid:
            raise ReferenceError('stale analytic UV layer handle')
        return self._values[name]

    def __setattr__(self, name, value):
        if name in ('name', 'active_render', 'data'):
            if not self.valid:
                raise ReferenceError('stale analytic UV layer handle')
            self._values[name] = value
        else:
            object.__setattr__(self, name, value)


class UVLayers(list):
    def __init__(self, values):
        super().__init__([UVLayer(name='OriginalUV0', active_render=True,
            data=UVData(SimpleNamespace(uv=tuple(value)) for value in values))])
        self.active_index = 0
        self.stale_handles = []

    @property
    def active(self):
        return self[self.active_index]

    def refresh_handles(self):
        for index, old in enumerate(list(self)):
            self[index] = UVLayer(old.name, old.active_render, old.data)
            old.valid = False
            self.stale_handles.append(old)

    def new(self, *, name):
        # CustomData growth can invalidate prior layer RNA references.
        self.refresh_handles()
        self[0].active_render = False
        result = UVLayer(name=name, active_render=True,
            data=UVData(SimpleNamespace(uv=(0., 0.)) for _ in self[0].data))
        self.append(result)
        self.active_index = len(self)-1
        return result


def native_fixture():
    points, faces, uv, selected, regions = fixture()
    loops = np.arange(len(faces)*3).reshape(-1, 3)
    mesh = SimpleNamespace(vertices=[SimpleNamespace(co=tuple(p), groups={'root': 1}) for p in points],
        loops=[SimpleNamespace(tangent=(1., 0., 0.), bitangent_sign=1) for _ in loops.ravel()],
        corner_normals=[SimpleNamespace(vector=(0., 0., 1.)) for _ in loops.ravel()],
        polygons=[SimpleNamespace(loop_total=3, material_index=int(region)) for region in regions],
        loop_triangles=[SimpleNamespace(vertices=tuple(face), loops=tuple(loop), polygon_index=i)
                        for i, (face, loop) in enumerate(zip(faces, loops))],
        uv_layers=UVLayers(uv.reshape(-1, 2)), attributes={}, materials=[material('surface'), material('aliases')],
        calc_loop_triangles=lambda: None, calc_tangents=lambda **_: None)
    mesh.calc_tangents = lambda **_: mesh.uv_layers.refresh_handles()
    mask = np.ones(len(points), dtype=bool)
    mask[3] = False
    return SimpleNamespace(data=mesh), mask, bake_fixture()[7]


class Images:
    """Fake files are labeled bytes, never valid generated images."""

    def __init__(self):
        self.generated, self.loaded, self.peak_generated = [], [], 0

    def new(self, name, **_):
        image = SimpleNamespace(name=name, colorspace_settings=SimpleNamespace(name=None),
                                pixels=SimpleNamespace(foreach_set=lambda _: None))
        image.save = lambda: Path(image.filepath_raw).write_bytes(b'analytic image-API fixture, not a PNG')
        self.generated.append(image)
        self.peak_generated = max(self.peak_generated, len(self.generated))
        return image

    def remove(self, image):
        self.generated.remove(image)

    def load(self, path, check_existing):
        assert not check_existing and not self.generated
        image = SimpleNamespace(name=path, colorspace_settings=SimpleNamespace(name=None), packed=False, freed=False)
        image.pack = lambda: setattr(image, 'packed', True)
        image.buffers_free = lambda: setattr(image, 'freed', True)
        self.loaded.append(image)
        return image


class MaterialUndercoatTests(unittest.TestCase):
    def test_optional_settings_and_exact_whole_pack_budget(self):
        self.assertEqual(undercoat.validate_undercoat(settings()), settings())
        self.assertEqual(undercoat.validate_undercoat(config()), config())
        for value in (None, True, 'uv1', 'legacy'):
            with self.subTest(policy=value), self.assertRaises(ValueError):
                undercoat.validate_undercoat({**settings(), 'uv_policy': value})
        for value in (None, {}, budget(source_export_vertices=True), budget(source_export_vertices=11),
                      budget(portable_export_vertices=-1), budget(reserve_vertices=10001),
                      {**budget(), 'extra': 0}, budget(portable_export_vertices=100000)):
            with self.subTest(value=value), patch.object(atlas, 'analyze_uv_regions') as analyze, self.assertRaises(ValueError):
                undercoat.plan_material_undercoat(*fixture(), config(), export_vertex_budget=value)
            analyze.assert_not_called()

    def test_only_measured_material_needs_uv1_and_every_budget_term_is_retained(self):
        args = fixture()
        before = [value.copy() for value in args]
        result, payloads = plan(args)
        self.assertEqual(result['atlas_regions'], [1])
        self.assertEqual(result['source_export_vertices'], 36)
        self.assertEqual(result['export_vertex_budget'], budget())
        self.assertEqual(result['projected_export_vertex_ceiling'], 4136)
        self.assertEqual(result['additional_export_vertices'], 4)
        self.assertEqual(result['regions'][0]['method'], 'keep_uv0')
        self.assertEqual(payloads[1]['uvs'].shape, (4, 3, 2))
        self.assertFalse(result['quality_approved'])
        self.assertFalse(result['weft_appearance_compatible'])
        for actual, original in zip(args, before):
            np.testing.assert_array_equal(actual, original)

    def test_material_boundary_corner_copies_are_not_undercounted(self):
        points = np.array([[0, 0, 0], [.01, 0, 0], [0, .01, 0], [-.01, 0, 0], [0, -.01, 0]], dtype=np.float32)
        faces = np.array([[0, 1, 2], [0, 2, 3], [0, 3, 4], [0, 4, 1]])
        uv = points[faces, :2]*20+.5
        args = points, faces, uv, np.array([True, True, False, False]), np.zeros(4, dtype=np.int32)
        result, _ = plan(args, source_export_vertices=99996, portable_export_vertices=0, reserve_vertices=0)
        self.assertEqual(result['regions'][0]['material_split_extra_vertices'], 3)
        self.assertEqual(result['regions'][0]['material_split_corner_bound'], 4)
        self.assertEqual(result['projected_export_vertex_ceiling'], 100000)
        with self.assertRaisesRegex(ValueError, 'material seams'):
            plan(args, source_export_vertices=99997, portable_export_vertices=0, reserve_vertices=0)

    def test_region_face_and_invalid_input_bounds_fail_without_atlas_packing(self):
        args = list(fixture())
        args[4] = np.arange(6, dtype=np.int32)
        with patch.object(atlas, 'triangle_atlas') as pack, self.assertRaisesRegex(ValueError, 'four selected'):
            plan(args)
        pack.assert_not_called()
        args = list(fixture())
        args[0][0] = np.nan
        with patch.object(atlas, '_coverage') as raster, self.assertRaisesRegex(ValueError, 'Nonfinite'):
            plan(args)
        raster.assert_not_called()
        args = list(fixture())
        args[1] = np.tile(args[1][:1], (2049, 1))
        args[2] = np.zeros((2049, 3, 2), dtype=np.float32)
        args[3] = np.ones(2049, dtype=bool)
        args[4] = np.zeros(2049, dtype=np.int32)
        with patch.object(atlas, 'triangle_atlas') as pack, self.assertRaisesRegex(ValueError, '2048face'):
            undercoat.plan_material_undercoat(*args, {**config(), 'max_faces': 3000}, export_vertex_budget=budget())
        pack.assert_not_called()

    def test_selected_polygons_require_one_triangle_and_exclusive_loops(self):
        loops = np.arange(6).reshape(2, 3)
        selected = np.array([True, False])
        np.testing.assert_array_equal(undercoat._selected_triangular_polygons(
            loops, np.array([0, 1]), np.array([3, 4]), selected, 7), [0])
        for bad_loops, ids, counts in ((loops, np.array([0, 0]), np.array([4])),
                                      (np.array([[0, 1, 2], [2, 3, 4]]), np.array([0, 1]), np.array([3, 3])),
                                      (loops, np.array([0, 1]), np.array([4, 3]))):
            with self.subTest(counts=counts), self.assertRaisesRegex(ValueError, 'triangular selected polygons'):
                undercoat._selected_triangular_polygons(bad_loops, ids, counts, selected, 7)

    def test_shader_clone_keeps_basecolor_and_binds_both_normal_uv_inputs(self):
        original = material('source')
        links = list(original.node_tree.links)
        image = object()
        clone = undercoat._clone_material(original, image, 'AuthoredUndercoatUV1', .92)
        self.assertEqual(original.node_tree.links, links)
        self.assertEqual(len(original.node_tree.nodes), 4)
        self.assertIs(clone.node_tree.nodes[0].image, original.node_tree.nodes[0].image)
        bsdf = clone.node_tree.nodes[2]
        self.assertEqual(len(bsdf.inputs['Base Color'].links), 1)
        self.assertEqual(len(bsdf.inputs['Roughness'].links), 1)
        self.assertEqual(len(bsdf.inputs['Metallic'].links), 1)
        self.assertEqual(bsdf.inputs['Roughness'].default_value, .9)
        self.assertEqual(bsdf.inputs['Metallic'].default_value, .2)
        uv, texture, normal = clone.node_tree.nodes[-3:]
        self.assertIs(texture.image, image)
        self.assertEqual(uv.uv_map, normal.uv_map)
        self.assertEqual(normal.uv_map, 'AuthoredUndercoatUV1')
        self.assertIs(texture.inputs['Vector'].links[0].from_socket, uv.outputs['UV'])
        original.copy = lambda: original
        with self.assertRaisesRegex(ValueError, 'isolate'):
            undercoat._clone_material(original, image, 'UV0', .8)
        self.assertEqual(original.node_tree.links, links)

    def test_mocked_adapter_preserves_uv0_defaults_and_unselected_material_faces(self):
        obj, mask, flow = native_fixture()
        original_positions = [v.co for v in obj.data.vertices]
        original_weights = [deepcopy(v.groups) for v in obj.data.vertices]
        original_uv = [v.uv for v in obj.data.uv_layers[0].data]
        originals = obj.data.materials[:]
        images = Images()
        calls = []
        real_bake = undercoat.bake_short_pile
        def bake(*args, **kwargs):
            pixels, baked = real_bake(*args, **kwargs)
            calls.append((args, kwargs, baked))
            return pixels, baked
        with tempfile.TemporaryDirectory() as output, patch.dict(sys.modules, {'bpy': SimpleNamespace(data=SimpleNamespace(images=images))}), \
                patch.object(undercoat, 'bake_short_pile', side_effect=bake), \
                patch.object(undercoat, 'evaluate_prepared_flow', wraps=undercoat.evaluate_prepared_flow) as evaluate:
            report = undercoat.apply_undercoat(obj, mask, flow, config(), Path(output), export_vertex_budget=budget())
        self.assertEqual(evaluate.call_count, len(obj.data.vertices))
        self.assertEqual(len(calls), 2)
        self.assertIs(calls[0][1]['_budget'], calls[1][1]['_budget'])
        self.assertIs(calls[0][1]['_directions'], calls[1][1]['_directions'])
        shared = calls[0][1]['_directions']
        self.assertFalse(shared.flags.writeable)
        self.assertEqual(shared.nbytes, len(obj.data.vertices)*12)
        for args, _, baked in calls:
            _, legacy = real_bake(*args)
            self.assertEqual({key: value for key, value in baked.items() if key != 'elapsed_seconds'},
                             {key: value for key, value in legacy.items() if key != 'elapsed_seconds'})
        self.assertEqual(report['shared_flow']['evaluated_vertices'], len(obj.data.vertices))
        self.assertEqual(report['shared_flow']['material_bakes'], 2)
        self.assertEqual([r['normal_texcoord'] for r in report['materials']], [0, 1])
        self.assertEqual(report['selected_polygons'], 5)
        self.assertEqual([p.material_index for p in obj.data.polygons], [2, 0, 3, 3, 3, 3])
        self.assertEqual(obj.data.uv_layers.active_index, 0)
        self.assertEqual(obj.data.uv_layers[0].name, 'OriginalUV0')
        self.assertTrue(obj.data.uv_layers[0].active_render)
        self.assertFalse(obj.data.uv_layers[1].active_render)
        self.assertEqual(report['materials'][0]['normal_uv_map'], 'OriginalUV0')
        self.assertGreaterEqual(len(obj.data.uv_layers.stale_handles), 2)
        for stale in obj.data.uv_layers.stale_handles:
            with self.assertRaises(ReferenceError):
                _ = stale.data
        self.assertEqual([v.uv for v in obj.data.uv_layers[0].data], original_uv)
        self.assertEqual([v.co for v in obj.data.vertices], original_positions)
        self.assertEqual([v.groups for v in obj.data.vertices], original_weights)
        self.assertTrue(all(len(m.node_tree.nodes) == 4 for m in originals))
        self.assertEqual(images.peak_generated, 1)
        self.assertFalse(images.generated)
        self.assertTrue(all(i.packed and i.freed for i in images.loaded))
        self.assertFalse(report['quality_approved'])

    def test_invalid_uv_defaults_and_ambiguous_polygons_fail_before_native_mutation(self):
        for failure in ('active_uv1', 'attribute', 'polygon', 'budget', 'normal', 'tangent', 'sign'):
            obj, mask, flow = native_fixture()
            if failure == 'active_uv1': obj.data.uv_layers.new(name='ExistingUV1')
            if failure == 'attribute': obj.data.attributes[undercoat.NORMAL_UV1_NAME] = object()
            if failure == 'polygon': obj.data.polygons[0].loop_total = 4
            if failure == 'normal': obj.data.corner_normals[0].vector = (float('nan'), 0, 1)
            if failure == 'tangent': obj.data.loops[0].tangent = (0., 0., 1.)
            if failure == 'sign': obj.data.loops[0].bitangent_sign = .5
            before_layers = len(obj.data.uv_layers)
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as output, \
                    patch.dict(sys.modules, {'bpy': SimpleNamespace()}), \
                    patch.object(undercoat, 'bake_short_pile') as bake, self.assertRaises(ValueError):
                undercoat.apply_undercoat(obj, mask, flow, config(), output,
                    export_vertex_budget=None if failure == 'budget' else budget())
            bake.assert_not_called()
            self.assertEqual(len(obj.data.materials), 2)
            self.assertEqual(len(obj.data.uv_layers), before_layers)

    def test_new_bake_policy_handles_opposed_normals_without_changing_them(self):
        obj, mask, flow = native_fixture()
        obj.data.corner_normals[6].vector = (0., 0., -1.)
        before = [corner.vector for corner in obj.data.corner_normals]
        with tempfile.TemporaryDirectory() as output, \
                patch.dict(sys.modules, {'bpy': SimpleNamespace(data=SimpleNamespace(images=Images()))}), \
                patch.object(undercoat, 'bake_short_pile', return_value=(
                    np.array([[[.5, .5, 1., 1.]]], dtype=np.float32), {'analytic_mock': True})) as bake:
            report = undercoat.apply_undercoat(obj, mask, flow, config(), output, export_vertex_budget=budget())
        self.assertEqual(report['frame_policy'], atlas.NORMAL_UV1_FRAME_POLICY)
        self.assertEqual(report['plan']['frame_policy'], report['frame_policy'])
        self.assertEqual([row['frame_policy'] for row in report['materials']], [None, report['frame_policy']])
        self.assertEqual([corner.vector for corner in obj.data.corner_normals], before)
        np.testing.assert_array_equal(bake.call_args_list[1].args[5], np.ones((4, 3)))

    def test_healthy_uv0_reacquires_handle_after_tangent_allocation_without_uv1(self):
        obj, mask, flow = native_fixture()
        mask[4:] = False
        before = [value.uv for value in obj.data.uv_layers[0].data]
        with tempfile.TemporaryDirectory() as output, \
                patch.dict(sys.modules, {'bpy': SimpleNamespace(data=SimpleNamespace(images=Images()))}), \
                patch.object(undercoat, 'bake_short_pile', return_value=(
                    np.array([[[.5, .5, 1., 1.]]], dtype=np.float32), {'analytic_mock': True})):
            report = undercoat.apply_undercoat(obj, mask, flow, config(), output, export_vertex_budget=budget())
        self.assertEqual(len(obj.data.uv_layers), 1)
        self.assertEqual(len(obj.data.uv_layers.stale_handles), 1)
        self.assertEqual(report['plan']['atlas_regions'], [])
        self.assertEqual(report['materials'][0]['normal_uv_map'], 'OriginalUV0')
        self.assertEqual(obj.data.uv_layers.active_index, 0)
        self.assertTrue(obj.data.uv_layers[0].active_render)
        self.assertEqual([value.uv for value in obj.data.uv_layers[0].data], before)

    def test_regional_bakes_share_raster_and_runtime_limits(self):
        values = bake_fixture()
        work = {'started': undercoat.time.monotonic(), 'remaining_raster': 1}
        with self.assertRaisesRegex(ValueError, 'raster work'):
            undercoat.bake_short_pile(*values, _budget=work)
        work = {'started': undercoat.time.monotonic()-91, 'remaining_raster': 32000000}
        with patch.object(undercoat, '_raster') as raster, self.assertRaisesRegex(RuntimeError, 'deadline'):
            undercoat.bake_short_pile(*values, _budget=work)
        raster.assert_not_called()

    def test_borrowed_direction_validation_and_preparation_deadline(self):
        values = bake_fixture()
        valid = undercoat._prepare_directions(values[0], values[7], undercoat.time.monotonic())
        before = valid.copy()
        self.assertIs(undercoat._checked_directions(valid, len(values[0])), valid)
        bad_values = [np.ones((5, 3), dtype=np.float32), np.full((4, 3), np.nan, dtype=np.float32),
                      np.zeros((4, 3), dtype=np.float32), valid.astype(np.float64),
                      np.ones((4, 3), dtype=bool), valid.copy()]
        for bad in bad_values[:-1]:
            bad.flags.writeable = False
        for bad in bad_values:
            with self.subTest(dtype=bad.dtype, shape=bad.shape), patch.object(undercoat, '_raster') as raster, \
                    patch.object(undercoat, 'evaluate_prepared_flow') as evaluate, self.assertRaises(ValueError):
                undercoat.bake_short_pile(*values, _directions=bad)
            raster.assert_not_called()
            evaluate.assert_not_called()
        np.testing.assert_array_equal(valid, before)
        with patch.object(undercoat, 'evaluate_prepared_flow') as evaluate, self.assertRaisesRegex(RuntimeError, 'deadline'):
            undercoat._prepare_directions(values[0], values[7], undercoat.time.monotonic()-91)
        evaluate.assert_not_called()


if __name__ == '__main__':
    unittest.main()
