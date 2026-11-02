"""Inspect untrusted embedded GLB data and render neutral local review views."""

import json
from pathlib import Path
import sys
from collections import Counter

import bpy
from mathutils import Vector

sys.path.insert(0, str(Path(__file__).parent))
from blender_verify import hash_file, source_header, write_json, imported_bindings, bind_action, clear_actions, set_frame


def configure_exported_preview_engine(scene, samples):
    """The pinned Cycles control ignored glTF culling; never fall back to it."""
    if type(samples) is not int or not 1 <= samples <= 64:
        raise ValueError('Exported preview samples must be an integer in 1..64')
    try:
        scene.render.engine = 'BLENDER_EEVEE_NEXT'
        if scene.render.engine != 'BLENDER_EEVEE_NEXT':
            raise ValueError('EEVEE Next engine selection did not take effect')
        controls = getattr(scene, 'eevee', None)
        if controls is None or not hasattr(controls, 'taa_render_samples'):
            raise ValueError('EEVEE render-sample control is unavailable')
        controls.taa_render_samples = samples
        actual = controls.taa_render_samples
        if type(actual) is not int or actual != samples:
            raise ValueError('EEVEE render-sample control did not preserve the requested bound')
    except (AttributeError, TypeError, ValueError, RuntimeError) as error:
        raise ValueError('Exported glTF preview requires bounded EEVEE Next; no engine fallback') from error
    return {'policy': 'gltf-single-sided-culling-eevee-next-v1',
            'engine': scene.render.engine, 'samples_requested': samples, 'samples_actual': actual,
            'sample_control': 'scene.eevee.taa_render_samples', 'engine_fallback': False,
            'reason': 'Respect imported glTF single-sided culling and authored-normal semantics',
            'scope': 'Exported GLB review only; native hair/portrait renderers are separate'}


def main(model, output, resolution, samples, rig_guides=False, motion_path=None):
    model, out = Path(model), Path(output)
    doc = source_header(model)
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.gltf(filepath=str(model))
    scene = bpy.context.scene
    engine_policy = configure_exported_preview_engine(scene, samples)
    rigs = [obj for obj in bpy.context.scene.objects if obj.type == "ARMATURE"]
    helpers = {b.custom_shape for rig in rigs for b in rig.pose.bones if b.custom_shape}
    meshes = [obj for obj in bpy.context.scene.objects if obj.type == "MESH" and obj not in helpers]
    for rig in rigs:
        rig.data.pose_position = "REST"
    bpy.context.view_layer.update()
    points = ([obj.matrix_world @ vertex.co for obj in meshes for vertex in obj.data.vertices] if rig_guides else
              [obj.matrix_world @ Vector(corner) for obj in meshes for corner in obj.bound_box])
    low = Vector(tuple(min(p[i] for p in points) for i in range(3)))
    high = Vector(tuple(max(p[i] for p in points) for i in range(3)))
    center, extent = (low+high)*.5, max(high-low)
    material_counts = {obj.name: Counter(polygon.material_index for polygon in obj.data.polygons) for obj in meshes}
    report = {"schema_version": 1, "source_sha256": hash_file(model), "blender_version": bpy.app.version_string,
              "render_engine": engine_policy['engine'], "render_policy": engine_policy,
              "bounds_blender_z_up": {"min": list(low), "max": list(high)},
              "meshes": [{"name": obj.name, "vertices": len(obj.data.vertices), "faces": len(obj.data.polygons),
                          "uv_layers": list(obj.data.uv_layers.keys()),
                          "materials": [{"slot": index, "name": material.name if material else None,
                                         "faces": material_counts[obj.name][index]}
                                        for index, material in enumerate(obj.data.materials)]} for obj in meshes],
              "rigs": [{"name": rig.name, "bones": [{"name": b.name, "parent": b.parent.name if b.parent else None,
                        "head": list(b.head_local), "tail": list(b.tail_local)} for b in rig.data.bones]} for rig in rigs],
              "actions": [a.name for a in bpy.data.actions], "animation_verified": False,
              "material_count": len(doc.get("materials", [])), "source_is_untextured": not doc.get("materials"),
              "assumptions": ["Blender converts glTF +Y up/+Z forward to +Z up/-Y forward",
                              "Raw generated units are not a calibrated physical measurement",
                              "Hidden geometry is generated, not recovered"], "renders": []}
    if not doc.get("materials"):
        neutral = bpy.data.materials.new("UntexturedPreviewNeutral")
        neutral.use_nodes = True
        neutral.node_tree.nodes["Principled BSDF"].inputs["Base Color"].default_value = (.35, .38, .42, 1)
        neutral.node_tree.nodes["Principled BSDF"].inputs["Roughness"].default_value = .6
        for obj in meshes:
            obj.data.materials.append(neutral)
    for image in bpy.data.images:
        if image.size[0] and image.size[1]:
            image.pack()
    scene.render.threads_mode, scene.render.threads = "FIXED", 4
    scene.render.resolution_x = scene.render.resolution_y = resolution
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGB"
    scene.render.film_transparent = False
    scene.view_settings.view_transform = "AgX"
    scene.world = bpy.data.worlds.new("PreviewWorld")
    scene.world.use_nodes = True
    scene.world.node_tree.nodes["Background"].inputs[0].default_value = (.8, .8, .8, 1)
    scene.world.node_tree.nodes["Background"].inputs[1].default_value = .35
    camera_data = bpy.data.cameras.new("PreviewCamera")
    camera_data.type, camera_data.ortho_scale = "ORTHO", extent*1.18
    camera_data.clip_start, camera_data.clip_end = .001, max(100, extent*20)
    camera = bpy.data.objects.new("PreviewCamera", camera_data)
    scene.collection.objects.link(camera)
    scene.camera = camera
    for name, offset, power, size in (("Key", (-2, -3, 4), 600, 2.5), ("Fill", (3, -1, 2), 250, 3),
                                      ("Rim", (0, 3, 3), 350, 2)):
        data = bpy.data.lights.new(name, "AREA")
        data.energy, data.size = power*extent*extent, size*extent
        obj = bpy.data.objects.new(name, data)
        scene.collection.objects.link(obj)
        obj.location = center+Vector(offset)*extent
        obj.rotation_euler = (center-obj.location).to_track_quat("-Z", "Y").to_euler()
    for name, offset in (("front", (0, -3, .05)), ("three-quarter", (-1.6, -3, .1)),
                         ("back", (0, 3, .05)), ("side", (3, 0, .05))):
        camera.location = center+Vector(offset)*extent
        camera.rotation_euler = (center-camera.location).to_track_quat("-Z", "Y").to_euler()
        basis = camera.rotation_euler.to_matrix()
        projected = [basis.transposed()@(p-center) for p in points]
        spans = [max(p[i] for p in projected)-min(p[i] for p in projected) for i in (0, 1)]
        camera_data.ortho_scale = max(spans)*1.18
        for i in (0, 1):
            camera.location += basis.col[i]*(max(p[i] for p in projected)+min(p[i] for p in projected))*.5
        path = out/(name+".png")
        scene.render.filepath = str(path)
        bpy.ops.render.render(write_still=True)
        report["renders"].append({"view": name, "path": path.name, "sha256": hash_file(path)})
    if motion_path:
        states = json.loads(Path(motion_path).read_text())
        actions = {a.name: a for a in bpy.data.actions}
        bindings, owners = imported_bindings(list(scene.objects), list(actions.values()))
        if set(actions) != {s['id'] for s in states} or len(rigs) != 1:
            raise ValueError('Exported review needs exactly the requested actions and one actual armature')
        scene.render.fps, scene.render.fps_base = 24, 1
        for rig in rigs:
            rig.data.pose_position = 'POSE'
        bounds = []
        for state in states:
            action = actions[state['id']]
            bind_action(action, bindings, owners, rigs)
            if max(state['seconds'])*24 > action.frame_range[1]+.01:
                raise ValueError('Requested exported pose is outside the actual action')
            for second in state['seconds']:
                set_frame(second*24)
                depsgraph = bpy.context.evaluated_depsgraph_get()
                for obj in meshes:
                    evaluated = obj.evaluated_get(depsgraph)
                    bounds.extend(evaluated.matrix_world @ Vector(v) for v in evaluated.bound_box)
        motion_low = Vector(tuple(min(v[i] for v in bounds) for i in range(3)))
        motion_high = Vector(tuple(max(v[i] for v in bounds) for i in range(3)))
        motion_center = (motion_low+motion_high)*.5
        motion_extent = max(motion_high-motion_low)
        camera.location = motion_center+Vector((-1.6, -3, .1))*motion_extent
        camera.rotation_euler = (motion_center-camera.location).to_track_quat('-Z', 'Y').to_euler()
        basis = camera.rotation_euler.to_matrix().transposed()
        camera_points = [basis@(v-motion_center) for v in bounds]
        camera_data.ortho_scale = max(max(v[i] for v in camera_points)-min(v[i] for v in camera_points)
                                      for i in (0, 1))*1.15
        camera_data.clip_end = max(camera_data.clip_end, motion_extent*20)
        report['motion_states'] = []
        for state in states:
            bind_action(actions[state['id']], bindings, owners, rigs)
            folder = out/'renders'/state['id']
            folder.mkdir(parents=True)
            poses = []
            for index, second in enumerate(state['seconds']):
                set_frame(second*24)
                path = folder/f'{index:04d}.png'
                scene.render.filepath = str(path)
                bpy.ops.render.render(write_still=True)
                poses.append({'second': second, 'path': str(path.relative_to(out)), 'sha256': hash_file(path)})
            report['motion_states'].append({'id': state['id'], 'poses': poses, 'resolution': resolution})
        clear_actions(owners, rigs)
        for rig in rigs:
            rig.data.pose_position = 'REST'
        set_frame(0)
    if rig_guides:
        from bpy_extras.object_utils import world_to_camera_view
        from rig_guide_annotations import guide_projection, normalized_to_pixel

        bounds = {'min': list(low), 'max': list(high)}
        guides = {'schema_version': 1, 'source_sha256': report['source_sha256'],
                  'space': 'normalized_bounds_blender_z_up', 'bounds_blender_z_up': bounds,
                  'bounds_method': 'Exact raw mesh vertices transformed by object world matrices, not transformed bounding-box corners',
                  'rest_pose': 'Imported mesh coordinates with armatures set to REST; no authored bones inferred',
                  'scope': 'Union of inspected imported meshes; matches single-mesh authored_rig normalization after uniform calibration',
                  'views': []}
        scene.render.resolution_x = scene.render.resolution_y = 640
        for name, offset in (('front', (0, -3, 0)), ('side', (3, 0, 0))):
            projection = guide_projection(bounds, name)
            camera_data.ortho_scale = projection['ortho_scale']
            camera.location = center+Vector(offset)*extent
            camera.rotation_euler = (center-camera.location).to_track_quat('-Z', 'Y').to_euler()
            bpy.context.view_layer.update()
            anchors, error = [], 0
            for a, z in ((0, 0), (1, 1), (.5, .5)):
                normalized = [.5, .5, z]
                normalized[projection['horizontal_index']] = a
                point = low+Vector(tuple(v*(high[i]-low[i]) for i, v in enumerate(normalized)))
                projected = world_to_camera_view(scene, camera, point)
                actual = [float(projected.x*640), float((1-projected.y)*640)]
                expected = normalized_to_pixel(normalized, projection)
                error = max(error, *(abs(x-y) for x, y in zip(actual, expected)))
                anchors.append({'normalized': normalized, 'raw_pixel': actual})
            if error > .05:
                raise ValueError('Native camera and rig-guide coordinates disagree')
            raw = out/f'guide-{name}-raw.png'
            scene.render.filepath = str(raw)
            bpy.ops.render.render(write_still=True)
            guides['views'].append({'view': name, 'path': f'guide-{name}.png', 'raw_path': raw.name,
                                   'raw_sha256': hash_file(raw), 'projection': projection,
                                   'camera_world_matrix_row_major': [list(row) for row in camera.matrix_world],
                                   'world_to_camera_matrix_row_major': [list(row) for row in camera.matrix_world.inverted()],
                                   'native_projection_anchors': anchors, 'projection_error_max_pixels': error})
        report['rig_guides'] = guides
    bpy.ops.wm.save_as_mainfile(filepath=str(out/"source.blend"), check_existing=False)
    report["source_unchanged"] = hash_file(model) == report["source_sha256"]
    write_json(out/"native-inspection.json", report)


if __name__ == "__main__":
    args = sys.argv[sys.argv.index("--")+1:]
    main(args[0], args[1], int(args[2]), int(args[3]), len(args) > 4 and args[4] == 'true',
         args[5] if len(args) > 5 else None)
