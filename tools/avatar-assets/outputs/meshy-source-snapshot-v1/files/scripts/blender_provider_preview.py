"""Inspect untrusted embedded GLB data and render neutral local review views."""

import json
from pathlib import Path
import sys

import bpy
from mathutils import Vector

sys.path.insert(0, str(Path(__file__).parent))
from blender_verify import hash_file, source_header, write_json


def main(model, output, resolution, samples):
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
    points = [obj.matrix_world @ Vector(corner) for obj in meshes for corner in obj.bound_box]
    low = Vector(tuple(min(p[i] for p in points) for i in range(3)))
    high = Vector(tuple(max(p[i] for p in points) for i in range(3)))
    center, extent = (low+high)*.5, max(high-low)
    report = {"schema_version": 1, "source_sha256": hash_file(model), "blender_version": bpy.app.version_string,
              "bounds_blender_z_up": {"min": list(low), "max": list(high)},
              "meshes": [{"name": obj.name, "vertices": len(obj.data.vertices), "faces": len(obj.data.polygons),
                          "uv_layers": list(obj.data.uv_layers.keys())} for obj in meshes],
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
    bpy.ops.wm.save_as_mainfile(filepath=str(out/"source.blend"), check_existing=False)
    report["source_unchanged"] = hash_file(model) == report["source_sha256"]
    write_json(out/"native-inspection.json", report)


if __name__ == "__main__":
    args = sys.argv[sys.argv.index("--")+1:]
    main(args[0], args[1], int(args[2]), int(args[3]))
