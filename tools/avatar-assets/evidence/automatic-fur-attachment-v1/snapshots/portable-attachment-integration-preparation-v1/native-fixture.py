"""Three analytic exports; no character generation, image rendering or provider call."""
from pathlib import Path
import json
import sys

import bpy
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE/'sources/scripts'), str(HERE/'sources/avatar_assets')]
from blender_finish_generated import portable_coat
from blender_fur_identity import stamp_fur_identities
from blender_verify import write_json


def build(folder, profile):
    folder.mkdir()
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.render.fps = 24
    points = [(-.25, 0., .1), (-.05, .01, .1), (-.25, 0., .3),
              (.05, 0., .1), (.25, .01, .1), (.25, 0., .3)]
    faces = [(0, 1, 2), (3, 4, 5)]
    mesh = bpy.data.meshes.new('two_band_fixture_mesh')
    mesh.from_pydata(points, [], faces); mesh.update()
    body = bpy.data.objects.new('analytic_surface_'+profile.replace('-', '_'), mesh)
    scene.collection.objects.link(body)
    material = bpy.data.materials.new('fixture_diffuse'); material.use_nodes = True
    shader = material.node_tree.nodes.get('Principled BSDF')
    shader.inputs['Base Color'].default_value = (.35, .55, .25, 1.)
    shader.inputs['Metallic'].default_value = 0.
    shader.inputs['Roughness'].default_value = 1.
    mesh.materials.append(material)
    uv = mesh.uv_layers.new(name='UVMap')
    for loop in mesh.loops:
        uv.data[loop.index].uv = ((loop.vertex_index % 3)/2, loop.vertex_index//3)
    data = bpy.data.armatures.new('analytic_links')
    rig = bpy.data.objects.new('FixtureRig', data); scene.collection.objects.link(rig)
    bpy.context.view_layer.objects.active = rig; rig.select_set(True)
    bpy.ops.object.mode_set(mode='EDIT')
    anchor = data.edit_bones.new('anchor'); anchor.head = (0, 0, 0); anchor.tail = (0, 0, .1)
    link = data.edit_bones.new('flap_link'); link.head = (0, 0, .1); link.tail = (0, 0, .3); link.parent = anchor
    bpy.ops.object.mode_set(mode='OBJECT')
    groups = {name: body.vertex_groups.new(name=name) for name in ('anchor', 'flap_link')}
    weights = [{'anchor': 1.} for _ in range(4)]+[{'flap_link': 1.}, {'anchor': .5, 'flap_link': .5}]
    for index, row in enumerate(weights):
        for name, weight in row.items():
            groups[name].add([index], weight, 'REPLACE')
    body.modifiers.new('FixtureSkin', 'ARMATURE').object = rig
    rig.data.pose_position = 'REST'
    colors = np.asarray([[.2]*3]*3+[[.8]*3]*3)
    settings = {'seed': 211, 'profile': profile, 'length': .012, 'width': .003,
                'direction': [0, 0, 1], 'sample_root_color': True}
    if profile == 'tuft-v1':
        settings['root_placement'] = 'skin-consistent-v1'
    cfg = {'settings': settings, 'bands': [
        {'count': 2, 'brightness': [0, .5], 'color': [1, 1, 1, 1]},
        {'count': 2, 'brightness': [.5, 1.01], 'color': [1, 1, 1, 1]}]}
    native_groom = {'direction': [0, 0, 1], 'comb': .3}
    flow = {'regions': [{'name': 'fixture-flow', 'center': [.5, .5, .5], 'radius': [1, 1, 1],
                         'direction': [0, 0, 1], 'comb': .3, 'length_scale': 1, 'feather': .5}]}
    options = {'tuft_source': {'kind': 'constant', 'linear_rgb': [.2, .4, .3], 'material_name': 'fixture_diffuse'},
               'native_groom': native_groom, 'groom_flow': flow} if profile == 'tuft-v1' else {}
    coats, reports = portable_coat(body, rig, np.ones(6, dtype=bool), colors, cfg, folder, **options)
    identity = stamp_fur_identities(body, coats, rig, reports)
    for second, angle in ((0., 0.), (.5, .6), (1., 0.)):
        bone = rig.pose.bones['flap_link']; bone.rotation_mode = 'XYZ'; bone.rotation_euler = (0, angle, 0)
        bone.keyframe_insert('rotation_euler', frame=second*24)
    action = rig.animation_data.action; action.name = 'flex_probe'; action.use_fake_user = True
    rig.animation_data.action = None
    rig.data.pose_position = 'POSE'
    bpy.ops.object.select_all(action='DESELECT')
    for obj in [rig, body, *coats]:
        obj.select_set(True)
    bpy.context.view_layer.objects.active = rig
    bpy.ops.export_scene.gltf(filepath=str(folder/'model.glb'), export_format='GLB', use_selection=True,
        export_animations=True, export_animation_mode='ACTIONS', export_anim_single_armature=True,
        export_force_sampling=True, export_frame_range=False, export_yup=True, export_skins=True,
        export_normals=True, export_tangents=True, export_image_format='AUTO', export_morph=False,
        export_extras=False, export_materials='EXPORT', export_current_frame=False, export_attributes=True)
    rig.data.pose_position = 'REST'
    bpy.ops.wm.save_as_mainfile(filepath=str(folder/'source.blend'))
    write_json(folder/'binding-report.json', {'schema_version': 1, 'analytic_fixture_not_avatar': True,
        'portable_fur': reports, 'portable_fur_identity': identity, 'config': cfg,
        'state_ids': ['flex_probe'], 'native_body_points': points, 'native_body_faces': faces,
        'native_body_weights': weights, 'expected_attachment': 'pass' if profile == 'tuft-v1' else 'numeric_failure',
        'custom_attribute_export_is_not_weft_compatibility': True})


argv = sys.argv[sys.argv.index('--')+1:]
assert len(argv) == 1
output = Path(argv[0])
assert output.is_dir() and {p.name for p in output.iterdir()} <= {'blender-config', 'blender-scripts', 'blender-extensions'}
for profile in ('legacy', 'combed-v1', 'tuft-v1'):
    build(output/profile, profile)
write_json(output/'fixture-result.json', {'schema_version': 1, 'status': 'exported_needs_cpu_audit',
    'profiles': ['legacy', 'combed-v1', 'tuft-v1'], 'native_exports': 3, 'new_provider_calls': 0,
    'new_generation_pipeline_role_calls': 0, 'appearance_accepted': False})
