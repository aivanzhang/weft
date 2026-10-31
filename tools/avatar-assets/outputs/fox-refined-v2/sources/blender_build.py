"""Trusted Blender backend. Input JSON is data; no user expressions are evaluated."""

import json
import math
import sys
from pathlib import Path

import bpy
from mathutils import Euler, Vector


def build(config, output, resolution, samples):
    out = Path(output)
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.unit_settings.system = "METRIC"
    scene.unit_settings.scale_length = 1
    scene.render.engine = "CYCLES"
    scene.cycles.device = "CPU"
    scene.cycles.samples = samples
    scene.cycles.use_denoising = True
    scene.render.threads_mode = "FIXED"
    scene.render.threads = 4
    scene.render.resolution_x = scene.render.resolution_y = resolution
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    scene.render.film_transparent = False
    scene.render.fps = 24
    scene.view_settings.view_transform = "Standard"
    scene.world = bpy.data.worlds.new("StudioWorld")
    scene.world.use_nodes = True
    scene.world.node_tree.nodes["Background"].inputs[0].default_value = config.get("background", [.95, .96, .98, 1])
    scene.world.node_tree.nodes["Background"].inputs[1].default_value = .65

    materials = {}
    for name, m in config["materials"].items():
        mat = bpy.data.materials.new(name)
        color = [v / 12.92 if v < .04045 else ((v + .055) / 1.055) ** 2.4 for v in m["color"][:3]] + [1]
        mat.diffuse_color = color
        mat.use_nodes = True
        bsdf = mat.node_tree.nodes.get("Principled BSDF")
        bsdf.inputs["Base Color"].default_value = color
        bsdf.inputs["Roughness"].default_value = m.get("roughness", .5)
        bsdf.inputs["Metallic"].default_value = m.get("metallic", 0)
        materials[name] = mat

    bpy.ops.object.armature_add(enter_editmode=True)
    rig = bpy.context.object
    rig.name = "AvatarRig"
    rig.data.name = config["id"] + "_skeleton"
    rig.data.edit_bones.remove(rig.data.edit_bones[0])
    for b in config["bones"]:
        bone = rig.data.edit_bones.new(b["name"])
        bone.head, bone.tail = b["head"], b["tail"]
        if b.get("parent"):
            bone.parent = rig.data.edit_bones[b["parent"]]
        bone.use_deform = True
    bpy.ops.object.mode_set(mode="OBJECT")
    meshes = []
    for p in config["parts"]:
        common = {"location": p["location"], "rotation": p.get("rotation", [0, 0, 0])}
        shape = p["shape"]
        if shape == "sphere":
            bpy.ops.mesh.primitive_uv_sphere_add(segments=20, ring_count=12, radius=1, **common)
        elif shape == "cube":
            bpy.ops.mesh.primitive_cube_add(size=2, **common)
        elif shape == "cone":
            bpy.ops.mesh.primitive_cone_add(vertices=24, radius1=1, radius2=0, depth=2, **common)
        elif shape == "cylinder":
            bpy.ops.mesh.primitive_cylinder_add(vertices=24, radius=1, depth=2, **common)
        elif shape == "torus":
            bpy.ops.mesh.primitive_torus_add(major_segments=24, minor_segments=10,
                                           major_radius=1, minor_radius=.25, **common)
        obj = bpy.context.object
        obj.name, obj.scale = p["name"], p["scale"]
        bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
        if shape != "cube":
            for polygon in obj.data.polygons:
                polygon.use_smooth = True
        obj.data.materials.append(materials[p["material"]])
        group = obj.vertex_groups.new(name=p["bone"])
        group.add(list(range(len(obj.data.vertices))), 1, "REPLACE")
        meshes.append(obj)
    bpy.ops.object.select_all(action="DESELECT")
    for obj in meshes:
        obj.select_set(True)
    bpy.context.view_layer.objects.active = meshes[0]
    bpy.ops.object.join()
    mesh = bpy.context.object
    mesh.name = config["id"] + "_skinned_mesh"
    modifier = mesh.modifiers.new("SkeletalDeformation", "ARMATURE")
    modifier.object = rig
    mesh.parent = rig
    mesh["construction"] = "Authored smooth primitive surfaces; rigid 1-bone skin weights per part"
    rig["units"] = "meters"
    rig["source_prompt"] = config["prompt"]
    actions = {}
    rig.animation_data_create()
    for s in config["states"]:
        action = bpy.data.actions.new(s["id"])
        rig.animation_data.action = action
        for key in s["keyframes"]:
            for bone in rig.pose.bones:
                transform = key["bones"].get(bone.name, {})
                # Scene authors use model-rest axes; Blender pose axes follow each bone's rest basis.
                basis = bone.bone.matrix_local.to_3x3()
                authored = Euler(transform.get("rotation", [0, 0, 0]), "XYZ").to_matrix()
                bone.rotation_mode = "QUATERNION"
                bone.rotation_quaternion = (basis.inverted() @ authored @ basis).to_quaternion()
                bone.location = basis.inverted() @ Vector(transform.get("location", [0, 0, 0]))
                bone.keyframe_insert(data_path="rotation_quaternion", frame=key["frame"], group=bone.name)
                bone.keyframe_insert(data_path="location", frame=key["frame"], group=bone.name)
        for fc in action.fcurves:
            for point in fc.keyframe_points:
                point.interpolation = "LINEAR"
        action.use_fake_user = True
        action["meaning"] = s["meaning"]
        action["loop_intent"] = s["loop"]
        actions[s["id"]] = action
    rig.animation_data.action = None
    for bone in rig.pose.bones:
        bone.rotation_quaternion = (1, 0, 0, 0)
        bone.location = (0, 0, 0)
    scene.frame_set(0)
    bpy.context.view_layer.update()
    bpy.ops.object.select_all(action="DESELECT")
    rig.select_set(True)
    mesh.select_set(True)
    bpy.context.view_layer.objects.active = rig
    bpy.ops.export_scene.gltf(filepath=str(out / "model.glb"), export_format="GLB", use_selection=True,
                              export_animations=True, export_animation_mode="ACTIONS", export_anim_single_armature=True, export_force_sampling=True,
                              export_frame_range=False, export_yup=True, export_skins=True,
                              export_morph=False, export_extras=False, export_materials="EXPORT",
                              export_current_frame=False)

    camera_data = bpy.data.cameras.new("PortraitCamera")
    camera = bpy.data.objects.new("PortraitCamera", camera_data)
    scene.collection.objects.link(camera)
    camera.location = config["camera"]["position"]
    direction = Vector(config["camera"]["target"]) - camera.location
    camera.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()
    camera_data.type = "ORTHO"
    camera_data.ortho_scale = config["camera"]["orthographic_scale"]
    scene.camera = camera
    for name, location, power, size in (("Key", (3, -4, 6), 200, 4), ("Fill", (-3, -2, 3), 100, 3)):
        lamp_data = bpy.data.lights.new(name, "AREA")
        lamp_data.energy, lamp_data.shape, lamp_data.size = power, "DISK", size
        lamp = bpy.data.objects.new(name, lamp_data)
        scene.collection.objects.link(lamp)
        lamp.location = location
        lamp.rotation_euler = (Vector(config["camera"]["target"]) - lamp.location).to_track_quat("-Z", "Y").to_euler()

    scene.frame_start, scene.frame_end = 0, int(config["states"][0]["duration"] * 24) - 1
    rig.animation_data.action = actions[config["states"][0]["id"]]
    scene.frame_set(0)
    bpy.ops.wm.save_as_mainfile(filepath=str(out / "source.blend"))
    # Renders are downstream evidence; the GLB remains independent of cameras and lights.
    scene.render.filepath = str(out / "portrait.png")
    bpy.ops.render.render(write_still=True)
    for s in config["states"]:
        rig.animation_data.action = actions[s["id"]]
        folder = out / "renders" / s["id"]
        folder.mkdir(parents=True)
        frames = int(s["duration"] * 24)
        scene.frame_start, scene.frame_end = 0, frames - 1
        # Six unique samples plus the exact endpoint prove poses before full renders.
        for frame in sorted({0, frames // 5, 2 * frames // 5, 3 * frames // 5, 4 * frames // 5, frames - 1, frames}):
            scene.frame_set(frame)
            scene.render.filepath = str(folder / f"{frame:04d}.png")
            bpy.ops.render.render(write_still=True)
    (out / "blender-build.json").write_text(json.dumps({"blender": bpy.app.version_string,
        "vertices": len(mesh.data.vertices), "polygons": len(mesh.data.polygons),
        "bones": [b.name for b in rig.data.bones], "actions": list(actions),
        "render_engine": "CYCLES", "device": "CPU", "samples": samples,
        "render_resolution": resolution, "preview_only": True}, indent=2) + "\n")


if __name__ == "__main__":
    args = sys.argv[sys.argv.index("--") + 1:]
    build(json.loads(Path(args[0]).read_text()), args[1], int(args[2]), int(args[3]))
