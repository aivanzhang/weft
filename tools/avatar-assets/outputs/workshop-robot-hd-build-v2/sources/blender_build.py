"""Trusted Blender backend. Input JSON is data; no user expressions are evaluated."""

import json
import math
import sys
import struct
from pathlib import Path

import bpy
from mathutils import Euler, Vector

sys.path.insert(0, str(Path(__file__).resolve().parent))
from blender_geometry import attach_surface, build_part, repair_surface_uvs, validate_card_material_export
from blender_fur_cards import attach_fur_material


def build(config, output, resolution, samples, render_previews=True, portrait_only=False):
    out = Path(output)
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    quality = config.get("schema_version", 1) == 2
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
    scene.view_settings.view_transform = "AgX" if quality else "Standard"
    scene.world = bpy.data.worlds.new("StudioWorld")
    scene.world.use_nodes = True
    scene.world.node_tree.nodes["Background"].inputs[0].default_value = config.get("background", [.95, .96, .98, 1])
    studio_settings = config.get("studio")
    plush = bool(studio_settings and studio_settings.get("preset") == "plush")
    scene.world.node_tree.nodes["Background"].inputs[1].default_value = studio_settings["world_strength"] if studio_settings else .65
    if studio_settings:
        scene.view_settings.exposure = studio_settings["exposure"]
    if plush:
        scene.view_settings.look = "AgX - Medium High Contrast"
    scene.cycles.transparent_max_bounces = 16

    materials, texture_report = {}, []
    for name, m in config["materials"].items():
        mat = bpy.data.materials.new(name)
        color = [v / 12.92 if v < .04045 else ((v + .055) / 1.055) ** 2.4 for v in m["color"][:3]] + [1]
        mat.diffuse_color = color
        mat.use_nodes = True
        bsdf = mat.node_tree.nodes.get("Principled BSDF")
        bsdf.inputs["Base Color"].default_value = color
        bsdf.inputs["Roughness"].default_value = m.get("roughness", .5)
        bsdf.inputs["Metallic"].default_value = m.get("metallic", 0)
        if "surface" in m:
            texture_report.extend(attach_surface(mat, name, m, out))
        materials[name] = mat
    card_materials = {}
    for name in sorted({part["material"] for part in config["parts"]
                        if part.get("fur", {}).get("mode") == "cards" and part["fur"]["count"]}):
        card_materials[name], report = attach_fur_material(name, config["materials"][name], out)
        report["source_material"] = name
        texture_report.append(report)

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
    meshes, part_report, groom_report = [], [], []
    vertex_budget = 0
    for p in config["parts"]:
        objects, report = build_part(p, materials, quality, card_materials=card_materials)
        if p.get("native_groom"):
            from blender_native_groom import add_native_groom
            _, native_report = add_native_groom(objects[0], rig, p["native_groom"], materials[p["material"]])
            groom_report.append(native_report)
        vertex_budget += sum(len(obj.data.vertices) for obj in objects)
        if vertex_budget > 100000:
            raise ValueError("Constructed geometry exceeds 100000 vertices before export")
        meshes.extend(objects)
        part_report.append(report)
    bpy.ops.object.select_all(action="DESELECT")
    for obj in meshes:
        obj.select_set(True)
    bpy.context.view_layer.objects.active = meshes[0]
    bpy.ops.object.join()
    mesh = bpy.context.object
    mesh.name = config["id"] + "_skinned_mesh"
    # Fixed splits preserve triangle topology across rebuilds; byte order can still vary.
    triangles = mesh.modifiers.new("StableTriangulation", "TRIANGULATE")
    triangles.quad_method = "FIXED"
    triangles.ngon_method = "CLIP"
    bpy.ops.object.modifier_apply(modifier=triangles.name)
    uv_repair = repair_surface_uvs(mesh) if quality else None
    modifier = mesh.modifiers.new("SkeletalDeformation", "ARMATURE")
    modifier.object = rig
    mesh.parent = rig
    mesh["construction"] = ("Authored v2 mesh surfaces, blended weights, geometric/cutout fur and embedded textures"
                            if quality else "Authored smooth primitive surfaces; rigid 1-bone skin weights per part")
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
                              export_optimize_animation_keep_anim_armature=config.get("output_profile") != "weft-v1",
                              export_frame_range=False, export_yup=True, export_skins=True,
                              export_normals=True, export_tangents=quality, export_image_format="AUTO",
                              export_morph=False, export_extras=False, export_materials="EXPORT",
                              export_current_frame=False)
    if config.get("output_profile") == "weft-v1":
        # Emit explicit diffuse defaults for the frozen consumer contract.
        path = out / "model.glb"
        raw = path.read_bytes()
        json_size = struct.unpack_from("<I", raw, 12)[0]
        doc = json.loads(raw[20:20+json_size])
        for material in doc["materials"]:
            material["pbrMetallicRoughness"]["metallicFactor"] = 0
            material["pbrMetallicRoughness"]["roughnessFactor"] = 1
            material["alphaMode"] = "OPAQUE"
        encoded = json.dumps(doc, separators=(",", ":")).encode()
        encoded += b" " * (-len(encoded) % 4)
        remainder = raw[20+json_size:]
        path.write_bytes(struct.pack("<4sII", b"glTF", 2, 20+len(encoded)+len(remainder)) +
                         struct.pack("<II", len(encoded), 0x4E4F534A) + encoded + remainder)

    raw = (out / "model.glb").read_bytes()
    json_size = struct.unpack_from("<I", raw, 12)[0]
    exported = json.loads(raw[20:20+json_size])
    exported_vertices = sum(exported["accessors"][primitive["attributes"]["POSITION"]]["count"]
                            for item in exported.get("meshes", []) for primitive in item["primitives"])
    if exported_vertices > 100000:
        raise ValueError("Exported geometry exceeds 100000 vertices including UV/normal splits")
    if texture_report and (len(exported.get("images", [])) != len(texture_report)
                           or any(image.get("mimeType") != "image/png" or "uri" in image
                                  for image in exported.get("images", []))):
        raise ValueError("Surface images were not exported as embedded PNG textures")
    validate_card_material_export(exported, {material.name for material in card_materials.values()})
    hidden_card_materials = []
    if groom_report:
        # Native-only fiber renders replace cutout cards; the already exported GLB retains the cards.
        native_materials = {part["material"] for part in config["parts"] if part.get("native_groom")}
        for name in card_materials:
            covered = [part for part in config["parts"] if part["material"] == name
                       and part.get("fur", {}).get("mode") == "cards" and part["fur"]["count"]]
            if name in native_materials and all(part.get("native_groom") for part in covered):
                material = card_materials[name]
                alpha = material.node_tree.nodes.get("Principled BSDF").inputs["Alpha"]
                for link in list(alpha.links):
                    material.node_tree.links.remove(link)
                alpha.default_value = 0
                material["native_render_hidden"] = True
                hidden_card_materials.append(material.name)

    camera_data = bpy.data.cameras.new("PortraitCamera")
    camera = bpy.data.objects.new("PortraitCamera", camera_data)
    scene.collection.objects.link(camera)
    camera.location = config["camera"]["position"]
    direction = Vector(config["camera"]["target"]) - camera.location
    camera.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()
    camera_data.type = "ORTHO"
    camera_data.ortho_scale = config["camera"]["orthographic_scale"]
    scene.camera = camera
    if studio_settings:
        bpy.ops.mesh.primitive_plane_add(size=200, location=(0, 0, studio_settings["ground_z"]))
        floor = bpy.context.object
        floor.name = "StudioFloor_not_exported"
        mat = bpy.data.materials.new("StudioMatteWhite")
        mat.use_nodes = True
        bsdf = mat.node_tree.nodes.get("Principled BSDF")
        bsdf.inputs["Base Color"].default_value = (.94, .94, .94, 1)
        bsdf.inputs["Roughness"].default_value = 1
        floor.data.materials.append(mat)
    if plush:
        studio = (("Key", (3, -4, 6), 900, 3), ("Fill", (-3, -2, 3), 140, 4), ("Rim", (-2, 3, 5), 600, 2))
    else:
        studio = (("Key", (3, -4, 6), 450, 5), ("Fill", (-3, -2, 3), 180, 4),
                  ("Rim", (-2, 3, 5), 320, 3)) if quality else (("Key", (3, -4, 6), 200, 4), ("Fill", (-3, -2, 3), 100, 3))
    for name, location, power, size in studio:
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
    if render_previews:
        scene.render.filepath = str(out / "portrait.png")
        bpy.ops.render.render(write_still=True)
        for s in ([] if portrait_only else config["states"]):
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
        "scene_schema_version": config.get("schema_version", 1),
        "exported_vertices": exported_vertices, "parts": part_report,
        "mixed_weight_vertices": sum(item["mixed_weight_vertices"] for item in part_report),
        "fur_strands": sum(item["fur_strands"] for item in part_report),
        "fur_cards": sum(item["fur_cards"] for item in part_report),
        "native_grooms": groom_report,
        "fur_representation": {"glb": "skinned alpha-masked cards" if card_materials else "opaque triangle strands or none",
                               "native": "editable native groom with per-material cutout visibility" if groom_report else "same geometry as GLB",
                               "native_hidden_card_materials": hidden_card_materials,
                               "native_visible_card_materials": [material.name for material in card_materials.values()
                                                                 if material.name not in hidden_card_materials],
                               "native_hair_exported_to_glb": False},
        "textures": texture_report, "embedded_images": len(exported.get("images", [])),
        "embedded_normal_maps": sum("normalTexture" in material for material in exported.get("materials", [])),
        "control_vertices": sum(item["control_vertices"] for item in part_report),
        "uv_repair": uv_repair,
        "color_management": scene.view_settings.view_transform,
        "color_management_look": scene.view_settings.look,
        "studio_preset": "plush" if plush else "existing_default",
        "bones": [b.name for b in rig.data.bones], "actions": list(actions),
        "render_engine": "CYCLES", "device": "CPU", "samples": samples,
        "render_resolution": resolution, "preview_only": True, "previews_rendered": render_previews,
        "portrait_only": portrait_only}, indent=2) + "\n")


if __name__ == "__main__":
    args = sys.argv[sys.argv.index("--") + 1:]
    build(json.loads(Path(args[0]).read_text()), args[1], int(args[2]), int(args[3]),
          portrait_only=len(args) > 4 and args[4] == "portrait-only")
