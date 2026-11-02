"""Bounded fake Blender contracts; no native scene, pixel decode or fur-quality proof."""

import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace as NS
import sys
import unittest
from unittest.mock import patch
import weakref

import numpy as np

from scripts import material_fur


SOURCE = Path(__file__).resolve().parents[1]/'scripts/blender_material_colors.py'
spec = importlib.util.spec_from_file_location('tested_material_color_adapter', SOURCE)
adapter = importlib.util.module_from_spec(spec)
with patch.dict(sys.modules, {'material_fur': material_fur}):
    spec.loader.exec_module(adapter)


class Socket:
    def __init__(self, name, value=None):
        self.name = self.identifier = name
        self.default_value = value
        self.links = []


class Node:
    def __init__(self, kind, name, inputs=(), outputs=()):
        self.type, self.name = kind, name
        self.inputs = {name: Socket(name, value) for name, value in inputs}
        self.outputs = {name: Socket(name) for name in outputs}


class Rows(list):
    def foreach_get(self, key, target):
        target[:] = np.asarray([getattr(row, key) for row in self]).ravel()


class UVLayers(list):
    def __init__(self, values, active=None):
        super().__init__(values)
        self.active = active if active is not None else values[0] if values else None


class BufferTrace:
    def __init__(self):
        self.refs = []; self.peak_live = 0; self.calls = []

    def observe(self, name, target):
        owner = target.base if target.base is not None else target
        alive = [value for value in self.refs if value() is not None]
        self.peak_live = max(self.peak_live, len(alive)+1)
        self.refs.append(weakref.ref(owner)); self.calls.append(name)


class Pixels:
    def __init__(self, values, name, trace):
        self.values = np.asarray(values, dtype=np.float32)
        self.name, self.trace, self.reads = name, trace, 0

    def foreach_get(self, target):
        self.reads += 1
        self.trace.observe(self.name, target)
        if len(target) != self.values.size:
            raise AssertionError('A rejected metadata-only image reached pixel allocation/read')
        target[:] = self.values.ravel()


class Image:
    def __init__(self, name, values, trace, *, size=None):
        self.name = name; array = np.asarray(values, dtype=np.float32)
        self.size = size if size is not None else (array.shape[1], array.shape[0])
        self.colorspace_settings = NS(name='sRGB')
        # Packed bytes bind this analytic fake; image decoding is outside this adapter's scope.
        packed = b'ANALYTIC-IMAGE-'+name.encode()+array.tobytes()
        self.packed_file = NS(data=packed, size=len(packed))
        self.pixels = Pixels(array, name, trace)

    def as_pointer(self):
        return id(self)


def material(name, *, image=None, constant=(.25, .5, 1., 1.), alpha=1.):
    shader = Node('BSDF_PRINCIPLED', name+' shader', [('Base Color', constant), ('Alpha', alpha)], ['BSDF'])
    output = Node('OUTPUT_MATERIAL', name+' output', [('Surface', None)])
    output.is_active_output = True
    links = []
    def link(source, socket, target, input_name):
        value = NS(from_node=source, from_socket=source.outputs[socket],
                   to_node=target, to_socket=target.inputs[input_name])
        target.inputs[input_name].links.append(value); links.append(value)
    link(shader, 'BSDF', output, 'Surface')
    nodes = [shader, output]
    texture = None
    if image is not None:
        texture = Node('TEX_IMAGE', name+' texture', [('Vector', None)], ['Color', 'Alpha'])
        texture.image = image; texture.interpolation = 'Linear'; texture.extension = 'REPEAT'; texture.projection = 'FLAT'
        link(texture, 'Color', shader, 'Base Color'); nodes.append(texture)
    result = NS(name=name, use_nodes=True, node_tree=NS(nodes=nodes, links=links), diffuse_color=(1., 0., 1., 1.))
    return result, shader, texture, output


def owner(materials, *, slots=None, vertex_ids=None, uv_values=None, uv=True):
    slots = list(range(len(materials))) if slots is None else slots
    vertex_ids = np.arange(len(slots)*3, dtype=np.int32) if vertex_ids is None else np.asarray(vertex_ids, dtype=np.int32)
    if uv_values is None:
        uv_values = np.tile([[.1, .1], [.9, .1], [.1, .9]], (len(slots), 1))
    layers = []
    if uv:
        layer = NS(name='SourceUV', active_render=True, data=Rows(NS(uv=tuple(value)) for value in uv_values))
        layers.append(layer)
    loops = Rows(NS(index=index, vertex_index=int(vertex)) for index, vertex in enumerate(vertex_ids))
    polygons = [NS(index=index, vertices=tuple(map(int, vertex_ids[index*3:index*3+3])),
                   material_index=slot, loop_start=index*3, loop_total=3) for index, slot in enumerate(slots)]
    return NS(data=NS(vertices=[NS(index=i) for i in range(int(vertex_ids.max())+1)], loops=loops,
                      polygons=polygons, materials=materials, uv_layers=UVLayers(layers)))


def graph_fingerprint(obj):
    mesh = obj.data
    graphs = []
    for value in mesh.materials:
        if value is None:
            graphs.append(None); continue
        if not value.use_nodes or value.node_tree is None:
            graphs.append({'id': id(value), 'name': value.name, 'use_nodes': value.use_nodes}); continue
        nodes = []
        for node in value.node_tree.nodes:
            entry = {'id': id(node), 'type': node.type, 'name': node.name,
                     'inputs': {name: {'default': socket.default_value,
                         'links': [(id(link.from_node), link.from_socket.name, id(link)) for link in socket.links]}
                                for name, socket in node.inputs.items()}}
            if node.type == 'TEX_IMAGE':
                entry.update(image_pointer=node.image.as_pointer(), extension=node.extension,
                    projection=node.projection, interpolation=node.interpolation,
                    colorspace=node.image.colorspace_settings.name,
                    packed_sha256=hashlib.sha256(node.image.packed_file.data).hexdigest(),
                    pixel_values_sha256=hashlib.sha256(node.image.pixels.values.tobytes()).hexdigest())
            nodes.append(entry)
        graphs.append({'id': id(value), 'name': value.name, 'nodes': nodes,
                       'tree_links': [id(link) for link in value.node_tree.links]})
    return json.dumps({'materials': graphs,
        'loops': [loop.vertex_index for loop in mesh.loops],
        'polygons': [(p.vertices, p.material_index, p.loop_start, p.loop_total) for p in mesh.polygons],
        'uv_layers': [(layer.name, layer.active_render, [row.uv for row in layer.data]) for layer in mesh.uv_layers],
        'active_uv': mesh.uv_layers.active.name if mesh.uv_layers.active else None}, sort_keys=True)


class MaterialColorAdapterTests(unittest.TestCase):
    def images(self):
        trace = BufferTrace()
        a = Image('RedGreenBlue', [[[1, 0, 0, 1], [0, 1, 0, 1]], [[0, 0, 1, 1], [1, 1, 1, 1]]], trace)
        b = Image('GrayCyanGold', [[[.5, .5, .5, 1], [0, 1, 1, 1]], [[1, .25, 0, 1], [0, 0, 0, 1]]], trace)
        return trace, a, b

    def test_distinct_images_and_constant_faces_keep_owned_colors_and_source_graph(self):
        trace, a, b = self.images()
        mats = [material('CoatA', image=a)[0], material('CoatB', image=b)[0], material('Constant')[0]]
        obj = owner(mats); before = graph_fingerprint(obj)
        context, colors, report = adapter.context_for_source(obj, {'material_fur': {'materials': ['CoatA', 'CoatB', 'Constant']}})
        expected = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [.5, .5, .5], [0, 1, 1], [1, .25, 0],
                             [.53709873, .73535698, 1]]*1+[[.53709873, .73535698, 1]]*2, dtype=np.float32)
        np.testing.assert_allclose(colors, expected, atol=6e-8)
        self.assertEqual(colors[:6].tobytes(), expected[:6].tobytes())
        self.assertEqual(context['policy'], 'material-owned-fur-v1')
        self.assertIs(context['sources'][0]['image'], a); self.assertIs(context['sources'][1]['image'], b)
        self.assertEqual(context['sources'][2]['linear_rgb'], [.25, .5, 1])
        self.assertEqual(report['selection_sampling_calls'], 2)
        self.assertEqual(report['source_materials']['unique_source_images'], 2)
        self.assertEqual(report['maximum_cpu_pixel_buffer_bytes'], 64)
        self.assertEqual(trace.peak_live, 1)
        self.assertTrue(all(value() is None for value in trace.refs))
        self.assertEqual(graph_fingerprint(obj), before)
        self.assertFalse(report['source_materials']['source_materials_changed'])
        self.assertFalse(report['source_materials']['source_uvs_changed'])

    def test_same_image_is_read_once_even_with_different_material_objects(self):
        trace, image, _ = self.images()
        obj = owner([material('First', image=image)[0], material('Second', image=image)[0]])
        sources, description = adapter.describe_sources(obj, ['First', 'Second'])
        self.assertEqual(image.pixels.reads, 0)
        self.assertEqual(description['aggregate_source_pixels'], 4)
        self.assertEqual(description['unique_source_images'], 1)
        colors, report = adapter.sample_selection(obj, sources)
        np.testing.assert_array_equal(colors[:3], colors[3:])
        self.assertEqual(image.pixels.reads, 1); self.assertEqual(report['selection_sampling_calls'], 1)
        self.assertEqual(trace.peak_live, 1)

    def test_constant_only_source_needs_no_uv_or_pixel_buffer(self):
        obj = owner([material('Gray', constant=(.5, .5, .5, 1))[0], material('Black', constant=(0, 0, 0, 1))[0]], uv=False)
        before = graph_fingerprint(obj)
        with patch.object(adapter, 'pixel_buffer', side_effect=AssertionError('Constant colors must not allocate images')):
            context, colors, report = adapter.context_for_source(obj, {'material_fur': {'materials': ['Gray']}})
        np.testing.assert_allclose(colors[:3], [[.73535698]*3]*3, atol=6e-8)
        np.testing.assert_array_equal(colors[3:], np.zeros((3, 3), dtype=np.float32))
        self.assertEqual(report['selection_sampling_calls'], 0)
        self.assertEqual(context['sources'][1]['linear_rgb'], [0, 0, 0])
        self.assertEqual(graph_fingerprint(obj), before)

    def test_used_nonemitter_is_colored_but_unused_invalid_slots_are_ignored(self):
        trace, image, _ = self.images()
        emitted = material('Selected', image=image)[0]
        nonemitter = material('VisibleNonFur', constant=(0, .25, 0, 1), alpha=.2)[0]
        unused = NS(name='UnusedBroken', use_nodes=False, node_tree=None)
        obj = owner([emitted, nonemitter, unused, None], slots=[0, 1])
        context, colors, report = adapter.context_for_source(obj, {'material_fur': {'materials': ['Selected']}})
        self.assertEqual(set(context['sources']), {0, 1})
        np.testing.assert_allclose(colors[3:], [[0, .53709873, 0]]*3, atol=6e-8)
        self.assertEqual([r['emits_fur'] for r in report['source_materials']['materials']], [True, False])
        with self.assertRaisesRegex(ValueError, 'no source faces'):
            adapter.describe_sources(obj, ['UnusedBroken'])
        self.assertEqual(image.pixels.reads, 1)

    def test_differing_material_and_textured_uv_vertex_ownership_reject_before_pixels(self):
        for mode in ('material', 'uv'):
            trace, image, _ = self.images()
            mats = [material('A', image=image)[0], material('B', image=image)[0]]
            obj = owner(mats, slots=[0, 1] if mode == 'material' else [0, 0],
                        vertex_ids=[0, 1, 2, 0, 3, 4],
                        uv_values=[[.1, .1], [.9, .1], [.1, .9], [.2, .1], [.9, .1], [.1, .9]])
            with self.subTest(mode=mode), self.assertRaisesRegex(ValueError, 'material|UV'):
                adapter.context_for_source(obj, {'material_fur': {'materials': ['A']}})
            self.assertEqual(image.pixels.reads, 0)
            self.assertEqual(trace.calls, [])

    def test_wrong_graph_filter_wrap_color_space_and_uv_are_rejected_before_pixel_reads(self):
        def mismatch_uv(obj, shader, texture, output):
            obj.data.uv_layers.append(NS(name='OtherUV', active_render=True, data=obj.data.uv_layers.active.data))
            obj.data.uv_layers.active.active_render = False
        changes = [
            lambda o, s, t, out: setattr(t, 'interpolation', 'Closest'),
            lambda o, s, t, out: setattr(t, 'interpolation', 'Cubic'),
            lambda o, s, t, out: setattr(t, 'extension', 'EXTEND'),
            lambda o, s, t, out: setattr(t, 'projection', 'BOX'),
            lambda o, s, t, out: setattr(t.image.colorspace_settings, 'name', 'Linear'),
            lambda o, s, t, out: setattr(t.image, 'packed_file', None),
            lambda o, s, t, out: t.inputs['Vector'].links.append(NS(from_node=Node('UVMAP', 'ExplicitUV'), from_socket=Socket('UV'))),
            lambda o, s, t, out: setattr(s.inputs['Base Color'].links[0], 'from_socket', t.outputs['Alpha']),
            lambda o, s, t, out: setattr(t, 'type', 'MIX_RGB'),
            lambda o, s, t, out: s.inputs['Base Color'].links.append(s.inputs['Base Color'].links[0]),
            lambda o, s, t, out: setattr(out, 'is_active_output', False),
            lambda o, s, t, out: setattr(out.inputs['Surface'].links[0], 'from_node', t),
            lambda o, s, t, out: setattr(o.data.uv_layers, 'active', None), mismatch_uv]
        for index, change in enumerate(changes):
            trace, image, _ = self.images(); mat, shader, texture, output = material('Selected', image=image)
            obj = owner([mat]); change(obj, shader, texture, output)
            with self.subTest(change=index), self.assertRaises(ValueError):
                adapter.context_for_source(obj, {'material_fur': {'materials': ['Selected']}})
            self.assertEqual(image.pixels.reads, 0)

    def test_transparent_emitted_surfaces_and_missing_used_materials_reject_before_pixels(self):
        for mode in ('alpha_value', 'alpha_link', 'used_none', 'duplicate_names'):
            trace, image, _ = self.images()
            mat, shader, texture, output = material('Selected', image=image)
            materials = [mat]
            if mode == 'alpha_value': shader.inputs['Alpha'].default_value = .9
            elif mode == 'alpha_link': shader.inputs['Alpha'].links.append(NS(from_node=texture, from_socket=texture.outputs['Alpha']))
            elif mode == 'used_none': materials = [None]
            else: materials.append(material('Selected', image=image)[0])
            obj = owner(materials)
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                adapter.context_for_source(obj, {'material_fur': {'materials': ['Selected']}})
            self.assertEqual(image.pixels.reads, 0)

    def test_aggregate_16mp_is_rejected_in_description_before_any_pixel_allocation(self):
        trace, a, b = self.images(); a.size = (4096, 4096); b.size = (1, 1)
        obj = owner([material('A', image=a)[0], material('B', image=b)[0]])
        with patch.object(adapter, 'pixel_buffer', side_effect=AssertionError('No oversized buffer should be allocated')):
            with self.assertRaisesRegex(ValueError, 'aggregate16MP'):
                adapter.context_for_source(obj, {'material_fur': {'materials': ['A', 'B']}})
        self.assertEqual((a.pixels.reads, b.pixels.reads), (0, 0))
        obj = owner([material('First', image=a)[0], material('Second', image=a)[0]])
        _, report = adapter.describe_sources(obj, ['First', 'Second'])
        self.assertEqual(report['aggregate_source_pixels'], 16777216)
        self.assertEqual(report['unique_source_images'], 1)


if __name__ == '__main__':
    unittest.main()
