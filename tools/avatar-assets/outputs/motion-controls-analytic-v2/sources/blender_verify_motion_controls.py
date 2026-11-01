"""Analytic native/export proof for rest-axis metre offsets, parenting and interpolation."""

import json
import math
from pathlib import Path
import sys

import bpy
import numpy as np
from mathutils import Euler, Vector

sys.path.insert(0, str(Path(__file__).parent))
from blender_finish_generated import author_actions, motion_bounds
from blender_verify import bind_action, clear_actions, imported_bindings, set_frame, write_json


def points(objects):
    result = []
    for obj in objects:
        evaluated = obj.evaluated_get(bpy.context.evaluated_depsgraph_get())
        mesh = evaluated.to_mesh()
        try:
            result.extend(tuple(evaluated.matrix_world@v.co) for v in mesh.vertices)
        finally:
            evaluated.to_mesh_clear()
    return np.asarray(result)


def verify(out):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    armature = bpy.data.armatures.new('AnalyticSkeleton')
    rig = bpy.data.objects.new('AnalyticRig', armature)
    bpy.context.collection.objects.link(rig)
    bpy.context.view_layer.objects.active = rig
    rig.select_set(True)
    bpy.ops.object.mode_set(mode='EDIT')
    root = armature.edit_bones.new('Root')
    root.head, root.tail = (0,0,0), (0,0,.3)
    child = armature.edit_bones.new('Companion')
    child.head, child.tail, child.parent = (.2,0,.7), (.2,0,1), root
    bpy.ops.object.mode_set(mode='OBJECT')
    rig.rotation_euler.z, rig.scale = .4, (1.7,1.7,1.7)
    bpy.context.view_layer.update()
    vertices = [(-.1,0,0), (.1,0,0), (0,.15,0), (0,0,.25),
                (.1,0,.7), (.3,0,.7), (.2,.15,.7), (.2,0,1)]
    tetra = [(0,2,1),(0,1,3),(1,2,3),(2,0,3)]
    mesh = bpy.data.meshes.new('AnalyticTwoParts')
    mesh.from_pydata(vertices, [], tetra+[tuple(i+4 for i in tri) for tri in tetra])
    obj = bpy.data.objects.new('AnalyticSkin', mesh)
    bpy.context.collection.objects.link(obj)
    obj.matrix_world = rig.matrix_world.copy()
    for name, indices in [('Root', list(range(4))), ('Companion', list(range(4,8)))]:
        obj.vertex_groups.new(name=name).add(indices, 1, 'REPLACE')
    obj.modifiers.new('Skin', 'ARMATURE').object = rig
    material = bpy.data.materials.new('AnalyticMaterial')
    material.use_nodes = True
    obj.data.materials.append(material)
    designs = [{'id':'analytic-flight', 'interpolation':'LINEAR', 'keys':[
        {'second':0, 'rotations':{}},
        {'second':1, 'rotations':{'Root':[0,0,30]},
         'translations_m':{'Root':[.2,-.1,.15], 'Companion':[.3,0,.4]}},
        {'second':2, 'rotations':{}}]}]
    author_actions(rig, designs)
    action = bpy.data.actions['analytic-flight']
    assert all(p.interpolation == 'LINEAR' for c in action.fcurves for p in c.keyframe_points)
    bindings, owners = {action.name:[(rig, action.slots[0])]}, [rig, obj]
    source_samples, errors = {}, []
    for second in [0,.5,1,1.5,2]:
        bind_action(action, bindings, owners, [rig]); set_frame(second*24)
        fraction = 1-abs(second-1)
        actual_root = rig.matrix_world@rig.pose.bones['Root'].head
        errors.append((actual_root-Vector((.2,-.1,.15))*fraction).length)
        source_samples[second] = points([obj])
        actual_child = rig.matrix_world@rig.pose.bones['Companion'].head
        rig.pose.bones['Companion'].location = (0,0,0)
        bpy.context.view_layer.update()
        zero_child = rig.matrix_world@rig.pose.bones['Companion'].head
        rotation = Euler((0,0,math.radians(30*fraction))).to_matrix()
        expected = rotation@Vector((.3,0,.4))*fraction
        errors.append((actual_child-zero_child-expected).length)
    assert max(errors) < 2e-6, errors
    low, high, envelope = motion_bounds(obj, rig, [action], bindings, owners, designs)
    all_points = np.concatenate(list(source_samples.values()))
    assert np.max(low-all_points.min(axis=0)) < 2e-6
    assert np.max(all_points.max(axis=0)-high) < 2e-6
    clear_actions(owners, [rig])
    rig.data.pose_position = 'POSE'
    bpy.ops.object.select_all(action='DESELECT')
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    # Match the finisher: glTF skins require baked emitter transforms, not a scaled mesh node.
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    bpy.ops.object.select_all(action='DESELECT')
    rig.select_set(True); obj.select_set(True)
    bpy.context.view_layer.objects.active = rig
    bpy.ops.export_scene.gltf(filepath=str(out/'analytic.glb'), export_format='GLB', use_selection=True,
        export_animations=True, export_animation_mode='ACTIONS', export_anim_single_armature=True,
        export_force_sampling=True, export_frame_range=False, export_yup=True, export_skins=True,
        export_normals=True, export_morph=False, export_extras=False, export_current_frame=False)
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.gltf(filepath=str(out/'analytic.glb'))
    rigs = [o for o in bpy.context.scene.objects if o.type == 'ARMATURE']
    meshes = [o for o in bpy.context.scene.objects if o.type == 'MESH']
    imported = list(bpy.data.actions)
    imported_bind, imported_owners = imported_bindings(list(bpy.context.scene.objects), imported)
    assert len(rigs) == 1 and len(imported) == 1
    reload_errors = []
    for second, reference in source_samples.items():
        bind_action(imported[0], imported_bind, imported_owners, rigs); set_frame(second*24)
        actual = points(meshes)
        distances = np.linalg.norm(actual[:,None,:]-reference[None,:,:], axis=2)
        reload_errors.append(float(max(distances.min(axis=0).max(), distances.min(axis=1).max())))
    write_json(out/'verification.json', {'kind':'analytic_fixture_not_generated_avatar',
        'status':'passed' if max(reload_errors) < 3e-5 else 'failed',
        'blender_version':bpy.app.version_string, 'native_max_error_m':max(errors),
        'export_reload_max_error_m':max(reload_errors), 'seconds':[0,.5,1,1.5,2],
        'object_scale':1.7, 'object_z_rotation_radians':.4, 'parent_rotation_degrees':30,
        'missing_translation_returns_to_zero':True, 'interpolation':'LINEAR', 'motion_envelope':envelope})
    assert max(reload_errors) < 3e-5, reload_errors


if __name__ == '__main__':
    verify(Path(sys.argv[sys.argv.index('--')+1]).resolve())
