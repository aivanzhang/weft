"""Inspect untrusted embedded GLB data and render neutral local review views."""

import json
from pathlib import Path
import sys
from collections import Counter

import bpy
from mathutils import Vector

sys.path.insert(0, str(Path(__file__).parent))
from blender_verify import hash_file, source_header, write_json


def main(model, output, resolution, samples, rig_guides=False):
    model, out = Path(model), Path(output)
    doc = source_header(model)
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.gltf(filepath=str(model))
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
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.device = "CPU"
    scene.cycles.samples = samples
    scene.cycles.use_denoising = False
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
    for name, offset in (("front", (0, -3, .05)), ("three-quarter", (-1.6, -3, .1)), ("back", (0, 3, .05))):
        camera.location = center+Vector(offset)*extent
        camera.rotation_euler = (center-camera.location).to_track_quat("-Z", "Y").to_euler()
        path = out/(name+".png")
        scene.render.filepath = str(path)
        bpy.ops.render.render(write_still=True)
        report["renders"].append({"view": name, "path": path.name, "sha256": hash_file(path)})
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
    main(args[0], args[1], int(args[2]), int(args[3]), len(args) > 4 and args[4] == 'true')
