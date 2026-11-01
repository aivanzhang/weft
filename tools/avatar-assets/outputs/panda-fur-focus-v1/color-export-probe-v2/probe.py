"""Analytic real-export proof of the production portable_coat material graph."""
import json
from pathlib import Path
import sys

import bpy
import numpy as np

OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(OUT/'sources'))
from blender_finish_generated import portable_coat
from blender_verify import hash_file, write_json
from three_d import _GLB

bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.context.scene.render.threads_mode = 'FIXED'
bpy.context.scene.render.threads = 1
points = [(x+u, v, 0.) for x in (0., 2., 4.) for u, v in ((0., 0.), (1., 0.), (0., 1.))]
triangles = [(0, 1, 2), (3, 4, 5), (6, 7, 8)]
srgb = np.asarray([[.1, .08, .06], [.8, .2, .1], [.9, .85, .8]])
expected = np.where(srgb <= .04045, srgb/12.92, ((srgb+.055)/1.055)**2.4)
mesh = bpy.data.meshes.new('AnalyticEmitter')
mesh.from_pydata(points, [], triangles)
mesh.update()
emitter = bpy.data.objects.new('AnalyticEmitter', mesh)
bpy.context.scene.collection.objects.link(emitter)
data = bpy.data.armatures.new('AnalyticRig')
rig = bpy.data.objects.new('AnalyticRig', data)
bpy.context.scene.collection.objects.link(rig)
bpy.context.view_layer.objects.active = rig
rig.select_set(True)
bpy.ops.object.mode_set(mode='EDIT')
bone = data.edit_bones.new('Root')
bone.head, bone.tail = (0, 0, 0), (0, 0, 1)
bpy.ops.object.mode_set(mode='OBJECT')
emitter.vertex_groups.new(name='Root').add(list(range(9)), 1., 'REPLACE')
config = {'settings': {'seed': 27, 'length': .012, 'width': .002,
                      'direction': [1., 0., 0.], 'profile': 'combed-v1', 'sample_root_color': True},
          'bands': [{'brightness': [0, 1.01], 'color': [1, 1, 1, 1], 'count': 30}]}
write_json(OUT/'fixture-config.json', {'kind': 'analytic_fixture', 'emitter_points': points,
           'emitter_triangles': triangles, 'srgb_per_triangle': srgb.tolist(),
           'expected_linear_per_triangle': expected.tolist(), 'portable_coat': config})
cards, construction = portable_coat(emitter, rig, np.ones(9, dtype=bool), np.repeat(srgb, 3, axis=0), config, OUT)
assert len(cards) == 1
card = cards[0]
card.parent = rig
bsdf = next(n for n in card.data.materials[0].node_tree.nodes if n.type == 'BSDF_PRINCIPLED')
mix = bsdf.inputs['Base Color'].links[0].from_node
assert mix.type == 'MIX' and mix.data_type == 'RGBA' and mix.blend_type == 'MULTIPLY'
assert mix.inputs['Factor'].default_value == 1.
bpy.ops.object.select_all(action='DESELECT')
card.select_set(True)
rig.select_set(True)
bpy.context.view_layer.objects.active = rig
output = OUT/'colored-cards.glb'
bpy.ops.export_scene.gltf(filepath=str(output), export_format='GLB', use_selection=True,
    export_animations=False, export_yup=True, export_skins=True, export_normals=True,
    export_tangents=True, export_morph=False, export_extras=False, export_image_format='AUTO')
glb = _GLB(output)
primitive = glb.doc['meshes'][0]['primitives'][0]
attrs = primitive['attributes']
assert 'COLOR_0' in attrs and 'COLOR_1' not in attrs, attrs
positions = np.asarray(glb.accessor(attrs['POSITION']))
colors = np.asarray(glb.accessor(attrs['COLOR_0']))
bucket = np.floor(positions[:, 0]/2).astype(int)
assert set(bucket) == {0, 1, 2}
actual_expected = expected[bucket]
accessor = glb.doc['accessors'][attrs['COLOR_0']]
tolerance = {5121: 1/255, 5123: 1/65535, 5126: 0}[accessor['componentType']]+1e-6
error = float(np.max(np.abs(colors[:, :3]-actual_expected)))
assert error <= tolerance, (error, tolerance)
assert not np.all(colors[:, :3] == 1.)
assert np.max(colors[bucket == 0, :3]) < .011
if colors.shape[1] == 4:
    assert np.all(colors[:, 3] == 1.)
material = glb.doc['materials'][primitive['material']]
assert material['alphaMode'] == 'MASK' and 'baseColorTexture' in material['pbrMetallicRoughness']
assert len(glb.doc['skins']) == 1
export_hash = hash_file(output)
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.gltf(filepath=str(output))
imported = [o for o in bpy.context.scene.objects if o.type == 'MESH' and o.name == 'PortableCoat_0']
assert len(imported) == 1
loaded = imported[0]
loaded_bsdf = next(n for n in loaded.data.materials[0].node_tree.nodes if n.type == 'BSDF_PRINCIPLED')
loaded_mix = loaded_bsdf.inputs['Base Color'].links[0].from_node
assert loaded_mix.type == 'MIX' and loaded_mix.blend_type == 'MULTIPLY'
vertex_nodes = [n for n in loaded.data.materials[0].node_tree.nodes if n.type == 'VERTEX_COLOR']
assert len(vertex_nodes) == 1
attribute = loaded.data.color_attributes[vertex_nodes[0].layer_name]
assert attribute.domain == 'CORNER'
reload_error = max(float(np.max(np.abs(np.asarray(attribute.data[loop.index].color[:3])-
                   expected[int(loaded.data.vertices[loop.vertex_index].co.x//2)]))) for loop in loaded.data.loops)
assert reload_error <= tolerance
assert hash_file(output) == export_hash
write_json(OUT/'result.json', {'schema_version': 1, 'kind': 'analytic_real_glb_export_probe', 'status': 'passed',
    'blender_version': bpy.app.version_string, 'actual_production_function': 'blender_finish_generated.portable_coat',
    'construction': construction, 'export_attributes': attrs, 'color_accessor': accessor,
    'expected_linear_per_region': expected.tolist(), 'max_export_color_error': error,
    'max_reload_color_error': reload_error, 'quantization_tolerance': tolerance,
    'exported_vertex_count': len(colors), 'regions_sampled': sorted(set(map(int, bucket))),
    'has_dummy_white_COLOR_0': False, 'has_unused_COLOR_1': False,
    'reloaded_material_uses_COLOR_0': True, 'glb_sha256': export_hash,
    'limitations': ['Analytic colored triangle emitters, not a generated character or visual-quality acceptance.',
                    'No render, native groom, animation or browser-runtime behavior tested.'],
    'provider_calls': 0, 'new_provider_cost_usd': 0})
print(json.dumps({'status': 'passed', 'COLOR_0_error': error, 'reload_error': reload_error, 'sha256': export_hash}))
