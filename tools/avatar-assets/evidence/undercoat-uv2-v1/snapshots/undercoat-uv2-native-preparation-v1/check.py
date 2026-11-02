"""Actual Blender UV2 bake/export on an analytic mesh, not avatar acceptance."""
import hashlib
import json
from pathlib import Path
import sys

import bpy
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE/'sources'))
from scripts.blender_fur_undercoat import apply_undercoat
from scripts.groom_flow import prepare_flow

out = Path(sys.argv[sys.argv.index('--')+1])
bpy.ops.wm.read_factory_settings(use_empty=True)


def write(name, value):
    (out/name).write_text(json.dumps(value, indent=2, sort_keys=True)+'\n')


def image(name, color, colorspace):
    path = out/(name+'.png')
    generated = bpy.data.images.new(name, width=4, height=4, alpha=False)
    generated.colorspace_settings.name = colorspace
    generated.pixels.foreach_set(np.tile([*color, 1.], 16))
    generated.filepath_raw, generated.file_format = str(path), 'PNG'
    generated.save()
    bpy.data.images.remove(generated)
    result = bpy.data.images.load(str(path), check_existing=False)
    result.colorspace_settings.name = colorspace
    result.pack()
    return result


color = image('analytic-color', [.14, .3, .42], 'sRGB')
normal = image('analytic-source-normal', [.5, .5, 1.], 'Non-Color')
response = image('analytic-source-response', [.1, .73, .18], 'Non-Color')


def material(name):
    result = bpy.data.materials.new(name)
    result.use_nodes = True
    nodes, links = result.node_tree.nodes, result.node_tree.links
    shader = next(node for node in nodes if node.type == 'BSDF_PRINCIPLED')
    uv = nodes.new('ShaderNodeUVMap'); uv.uv_map = 'SourceUV1'
    textures = []
    for source in (color, normal, response):
        texture = nodes.new('ShaderNodeTexImage'); texture.image = source
        textures.append(texture)
    links.new(textures[0].outputs['Color'], shader.inputs['Base Color'])
    for texture in textures[1:]:
        links.new(uv.outputs['UV'], texture.inputs['Vector'])
    mapping = nodes.new('ShaderNodeNormalMap'); mapping.uv_map = 'SourceUV1'
    links.new(textures[1].outputs['Color'], mapping.inputs['Color'])
    links.new(mapping.outputs['Normal'], shader.inputs['Normal'])
    channels = nodes.new('ShaderNodeSeparateColor')
    links.new(textures[2].outputs['Color'], channels.inputs['Color'])
    links.new(channels.outputs['Green'], shader.inputs['Roughness'])
    links.new(channels.outputs['Blue'], shader.inputs['Metallic'])
    return result


square = np.array([[0, 0, 0], [.02, 0, 0], [.02, .02, 0], [0, .02, 0]], dtype=np.float32)
points = np.concatenate((square, square+[.03, 0, 0], square+[.03, 0, .02])).astype(np.float32)
faces = np.array([[0, 1, 2], [0, 2, 3], [4, 5, 6], [4, 6, 7], [8, 9, 10], [8, 10, 11]])
uv0 = np.tile(np.array([[[0., 0.], [1., 0.], [1., 1.]], [[0., 0.], [1., 1.], [0., 1.]]]), (3, 1, 1))
mesh = bpy.data.meshes.new('AnalyticUVPreservation')
mesh.from_pydata(points.tolist(), [], faces.tolist()); mesh.update()
obj = bpy.data.objects.new('AnalyticUVPreservation', mesh)
bpy.context.collection.objects.link(obj)
bpy.context.view_layer.objects.active = obj; obj.select_set(True)
mesh.materials.append(material('HealthyUV0'))
mesh.materials.append(material('AliasedUV0'))
for polygon, slot in zip(mesh.polygons, [0, 0, 1, 1, 1, 1]):
    polygon.material_index = slot
for name, data in (('OriginalUV0', uv0), ('SourceUV1', uv0*.6+.2)):
    layer = mesh.uv_layers.new(name=name)
    layer.data.foreach_set('uv', data.astype(np.float32).ravel())
mesh.uv_layers.active_index = 0
mesh.uv_layers[0].active_render = True
mesh.uv_layers[1].active_render = False
group = obj.vertex_groups.new(name='analytic-root')
group.add(list(range(len(points))), 1., 'REPLACE')


def fingerprint():
    return {'positions': [list(vertex.co) for vertex in mesh.vertices],
        'faces': [list(polygon.vertices) for polygon in mesh.polygons],
        'weights': [[(item.group, item.weight) for item in vertex.groups] for vertex in mesh.vertices],
        'source_uvs': [[list(value.uv) for value in mesh.uv_layers[index].data] for index in (0, 1)],
        'source_graphs': [[(link.from_node.name, link.from_socket.name, link.to_node.name, link.to_socket.name)
                           for link in material.node_tree.links] for material in mesh.materials[:2]]}


before = fingerprint()
mask = np.ones(len(points), dtype=bool); mask[3] = False
flow = prepare_flow([0, 0, -.01], [.05, .02, .04], [0, 1, 0], .6, {'regions': [
    {'name': 'analytic-flow', 'center': [.5, .5, .5], 'radius': [1., 1., 1.], 'feather': .5,
     'direction': [0, 1, 0], 'comb': .6, 'length_scale': 1.}]})
settings = {'profile': 'short-pile-v1', 'resolution': 1024, 'seed': 41, 'pitch_m': .004,
            'strength': .22, 'roughness': .92, 'min_faces': 1, 'max_faces': 10,
            'uv_policy': 'material-adaptive-v1'}
report = apply_undercoat(obj, mask, flow, settings, out,
    export_vertex_budget={'source_export_vertices': 18, 'portable_export_vertices': 0, 'reserve_vertices': 4096})
assert before == fingerprint(), 'Original geometry, weights, UV or material graph changed'
assert report['authored_uv']['index'] == 2
assert [row['normal_texcoord'] for row in report['materials']] == [0, 2]
assert mesh.uv_layers.active_index == 0 and mesh.uv_layers[0].active_render
for row in report['materials']:
    clone = bpy.data.materials[row['material']]
    shader = next(node for node in clone.node_tree.nodes if node.type == 'BSDF_PRINCIPLED')
    assert len(shader.inputs['Roughness'].links) == len(shader.inputs['Metallic'].links) == 1
write('finish-report.json', {'fur_undercoat': report})
write('preservation.json', {'analytic_fixture': True, 'original_state_unchanged': before == fingerprint(),
    'source_state_sha256': hashlib.sha256(json.dumps(before, sort_keys=True).encode()).hexdigest(),
    'uv_layers': [layer.name for layer in mesh.uv_layers], 'source_response_links_preserved': True,
    'generated_avatar': False, 'appearance_accepted': False})
bpy.ops.wm.save_as_mainfile(filepath=str(out/'source.blend'), check_existing=False)
bpy.ops.export_scene.gltf(filepath=str(out/'model-native.glb'), export_format='GLB', use_selection=True,
                          export_texcoords=True, export_normals=True, export_tangents=True,
                          export_materials='EXPORT', export_animations=False)
