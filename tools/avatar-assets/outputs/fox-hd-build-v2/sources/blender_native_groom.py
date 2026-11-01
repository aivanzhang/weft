"""Optional editable Blender hair; native-only, never a glTF groom conversion."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import bpy
from mathutils import Vector
from mathutils.bvhtree import BVHTree


MAX_FIBERS = 240000
GROOM_PREFIX = "NativeGroom_"


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def validate_groom_settings(settings):
    """Validate one data-only groom; the caller also caps total fibers across parts."""
    required = {"parents", "children", "length", "radius", "roughness", "seed"}
    _require(isinstance(settings, dict) and set(settings) in (required, required | {"direction", "comb"}),
             "Native groom requires six explicit fields and optional paired direction/comb")
    for name, low, high in (("parents", 16, 10000), ("children", 0, 30), ("seed", 0, 2**31 - 1)):
        _require(type(settings[name]) is int and low <= settings[name] <= high, "Invalid native groom " + name)
    for name, low, high in (("length", .001, .04), ("radius", .00001, .001), ("roughness", 0, .01)):
        value = settings[name]
        _require(type(value) in (int, float) and math.isfinite(value) and low <= value <= high,
                 "Invalid native groom " + name)
    _require(settings["parents"] * (settings["children"] + 1) <= MAX_FIBERS, "Native groom exceeds 240000 fibers")
    if "direction" in settings:
        direction = settings["direction"]
        _require(isinstance(direction, list) and len(direction) == 3
                 and all(type(value) in (int, float) and math.isfinite(value) and abs(value) <= 1
                         for value in direction), "Invalid native groom direction")
        _require(math.hypot(*direction) > 1e-8, "Zero or unstable native groom direction")
        _require(type(settings["comb"]) in (int, float) and math.isfinite(settings["comb"])
                 and 0 <= settings["comb"] <= .95, "Invalid native groom comb")
    return dict(settings)


def _skin_signature(obj, rig):
    _require(obj.type == "MESH" and 0 < len(obj.data.vertices) <= 100000 and obj.data.polygons,
             "Native groom requires a bounded base mesh, not existing fur triangles")
    groups = {group.index: group.name for group in obj.vertex_groups}
    bones = set(rig.data.bones.keys())
    rows, mixed = [], 0
    for vertex in obj.data.vertices:
        weights = [(groups[item.group], item.weight) for item in vertex.groups if item.weight > 0]
        _require(1 <= len(weights) <= 4 and all(name in bones for name, _ in weights)
                 and abs(sum(weight for _, weight in weights) - 1) <= .001,
                 "Every emitter vertex requires 1–4 normalized weights on the supplied armature")
        mixed += len(weights) > 1
        rows.append([*vertex.co, sorted(weights)])
    return {"vertices": len(rows), "mixed_weight_vertices": mixed,
            "geometry_weights_sha256": hashlib.sha256(json.dumps(rows, separators=(",", ":")).encode()).hexdigest()}


def _hair_material(material, settings):
    color = list(material.diffuse_color)
    if material.use_nodes:
        bsdf = material.node_tree.nodes.get("Principled BSDF")
        if bsdf and "Base Color" in bsdf.inputs:
            color = list(bsdf.inputs["Base Color"].default_value)
    result = bpy.data.materials.new(GROOM_PREFIX + material.name + "_hair")
    result.diffuse_color = color
    result.use_nodes = True
    nodes = result.node_tree.nodes
    nodes.clear()
    output = nodes.new("ShaderNodeOutputMaterial")
    hair = nodes.new("ShaderNodeBsdfHairPrincipled")
    hair.parametrization = "COLOR"
    hair.inputs["Color"].default_value = color
    hair.inputs["Roughness"].default_value = .38
    if "Radial Roughness" in hair.inputs:
        hair.inputs["Radial Roughness"].default_value = .45
    result.node_tree.links.new(hair.outputs[0], output.inputs["Surface"])
    result["native_only"] = True
    return result, {"material": result.name, "shader": hair.bl_idname, "model": hair.model,
                    "parametrization": hair.parametrization, "base_color_linear": color,
                    "shader_roughness": .38, "texture_color_sampling": False}


def add_native_groom(emitter, rig, settings, material):
    """Return (weighted emitter copy, report); preserve source and exclude the copy from glTF export."""
    settings = validate_groom_settings(settings)
    _require(rig.type == "ARMATURE" and material is not None, "Native groom needs an armature and material")
    _require(not emitter.particle_systems, "Emitter must be base geometry without an existing particle groom")
    _require(all(abs(value - 1) < 1e-6 for value in emitter.matrix_world.to_scale()),
             "Apply source scale before native grooming so length and thickness remain consistent")
    original = _skin_signature(emitter, rig)
    old_active = bpy.context.view_layer.objects.active
    old_selection = list(bpy.context.selected_objects)
    groom = emitter.copy()
    groom.data = emitter.data.copy()
    groom.name = GROOM_PREFIX + emitter.name
    groom.data.name = groom.name + "_editable_emitter"
    bpy.context.collection.objects.link(groom)
    world_matrix = emitter.matrix_world.copy()
    groom.parent = rig
    groom.matrix_world = world_matrix
    armatures = [modifier for modifier in groom.modifiers if modifier.type == "ARMATURE"]
    _require(len(armatures) <= 1 and all(modifier.object == rig for modifier in armatures),
             "Emitter armature differs from the requested rig")
    modifier = armatures[0] if armatures else groom.modifiers.new("NativeHairSkin", "ARMATURE")
    modifier.object = rig
    hair_material, material_report = _hair_material(material, settings)
    groom.data.materials.append(hair_material)
    try:
        bpy.ops.object.select_all(action="DESELECT")
        groom.select_set(True)
        bpy.context.view_layer.objects.active = groom
        bpy.ops.object.particle_system_add()
        system = groom.particle_systems[-1]
        system.name = groom.name + "_editable_hair"
        particle = system.settings
        particle.type = "HAIR"
        particle.count = settings["parents"]
        particle.hair_length = settings["length"]
        particle.hair_step = 3
        particle.render_step = 3
        particle.display_step = 3
        particle.use_modifier_stack = True
        particle.use_hair_bspline = True
        particle.use_regrow_hair = False
        particle.child_type = "INTERPOLATED" if settings["children"] else "NONE"
        particle.rendered_child_count = settings["children"]
        particle.child_percent = min(4, settings["children"])
        particle.use_parent_particles = True
        particle.clump_factor = .06
        particle.clump_shape = 0
        particle.roughness_1 = settings["roughness"]
        particle.roughness_1_size = .5
        particle.roughness_2 = settings["roughness"] * .35
        particle.roughness_2_size = .1
        # Blender's radius-named properties are diameters; keep the input's physical radius meaning.
        particle.root_radius = 2
        particle.tip_radius = .12
        particle.radius_scale = settings["radius"]
        particle.material = len(groom.data.materials)
        system.seed = settings["seed"]
        system.child_seed = settings["seed"]
        system.use_hair_dynamics = False
        groom.show_instancer_for_render = False
        groom.show_instancer_for_viewport = False
        groom["native_groom"] = True
        groom["gltf_export_supported"] = False
        groom["native_groom_source"] = emitter.name
        groom["native_groom_settings"] = json.dumps(settings, sort_keys=True)
        bpy.context.view_layer.update()
    finally:
        bpy.ops.object.select_all(action="DESELECT")
        for obj in old_selection:
            obj.select_set(True)
        bpy.context.view_layer.objects.active = old_active
    copied = _skin_signature(groom, rig)
    _require(copied == original and _skin_signature(emitter, rig) == original,
             "Native emitter copy changed original geometry or skin weights")
    report = {"schema_version": 1, "kind": "editable_native_blender_particle_hair", "native_only": True,
              "gltf_export_supported": False, "object": groom.name, "source_object": emitter.name,
              "armature": rig.name, "reserved_prefix": GROOM_PREFIX, "settings": settings,
              "render_fiber_upper_bound": settings["parents"] * (settings["children"] + 1),
              "render_count_scope": "Configured parents plus interpolated children; not an independently counted image feature",
              "emitter_surface_visible": False, "dynamics_enabled": False, "copied_skin": copied,
              "material": material_report,
              "native_parameters": {"hair_step": particle.hair_step, "render_step": particle.render_step,
                  "child_type": particle.child_type, "viewport_children": particle.child_percent,
                  "render_children": particle.rendered_child_count, "use_parent_particles": particle.use_parent_particles,
                  "use_modifier_stack": particle.use_modifier_stack, "root_radius": particle.root_radius,
                  "tip_radius": particle.tip_radius, "radius_scale": particle.radius_scale,
                  "root_diameter_m": 2 * settings["radius"], "tip_diameter_m": .12 * settings["radius"],
                  "radius_semantics": "Input is authoring-space radius at unit meter scale; Blender multiplies diameter properties by radius_scale",
                  "radius_parameter_description": particle.bl_rna.properties["radius_scale"].description,
                  "root_radius_description": particle.bl_rna.properties["root_radius"].description},
              "limitations": ["Native Blender hair is not exported as glTF geometry or browser hair.",
                  "Material uses the source base-color value, not sampled texture colors; use separate material regions.",
                  "Hair follows the copied emitter's same armature; keep emitter topology unchanged after grooming.",
                  "The caller caps aggregate fibers at 240000 and excludes native groom objects from GLB selection."]}
    if "direction" in settings:
        # Persistent rest-space edits must follow final particle settings and precede action poses.
        from blender_comb_groom import comb_native_groom
        report["comb"] = comb_native_groom(groom, settings["direction"], settings["comb"])
    return groom, report


def _write(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def inspect_api(output):
    started = time.monotonic()
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.mesh.primitive_uv_sphere_add(segments=16, ring_count=8)
    emitter = bpy.context.object
    bpy.ops.object.particle_system_add()
    system = emitter.particle_systems[-1]
    settings = system.settings
    settings.type = "HAIR"
    settings.count = 32
    settings.hair_length = .04
    bpy.context.view_layer.update()
    groups = {}
    for label, value in (("settings", settings), ("system", system), ("object", emitter)):
        props = value.bl_rna.properties
        groups[label] = {prop.identifier: {"type": prop.type,
                          "readonly": prop.is_readonly,
                          "enum": [item.identifier for item in prop.enum_items] if prop.type == "ENUM" else None}
                         for prop in props if any(word in prop.identifier for word in
                         ("hair", "child", "rough", "clump", "radius", "modifier", "render", "parent", "shape", "seed", "material"))}
    report = {"schema_version": 1, "status": "installed_api_inspection", "blender_version": bpy.app.version_string,
              "particle_hair_supported": settings.type == "HAIR", "properties": groups,
              "particle_methods": [name for name in dir(bpy.types.Particle) if "hair" in name or "co" in name],
              "particle_system_methods": [name for name in dir(bpy.types.ParticleSystem) if "hair" in name or "co" in name],
              "hair_shader_properties": [prop.identifier for prop in bpy.types.ShaderNodeBsdfHairPrincipled.bl_rna.properties],
              "elapsed_seconds": time.monotonic() - started,
              "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    _write(Path(output) / "api-inspection.json", report)
    print(json.dumps({"status": report["status"], "particle_hair_supported": report["particle_hair_supported"]}))


def _surface(obj):
    graph = bpy.context.evaluated_depsgraph_get()
    evaluated = obj.evaluated_get(graph)
    mesh = evaluated.to_mesh()
    try:
        mesh.calc_loop_triangles()
        vertices = [obj.matrix_world @ vertex.co for vertex in mesh.vertices]
        triangles = [tuple(triangle.vertices) for triangle in mesh.loop_triangles]
        return BVHTree.FromPolygons(vertices, triangles, all_triangles=True)
    finally:
        evaluated.to_mesh_clear()


def _hair_snapshot(groom):
    graph = bpy.context.evaluated_depsgraph_get()
    evaluated = groom.evaluated_get(graph)
    system = evaluated.particle_systems[0]
    _require(len(system.particles) > 0, "Native hair has no evaluated parent particles")
    _require(hasattr(system, "co_hair"), "Installed Blender does not expose evaluated hair coordinates")
    count = len(system.particles)
    indices = sorted({round(i * (count - 1) / 95) for i in range(min(96, count))})
    roots, tips = [], []
    for index in indices:
        roots.append(groom.matrix_world @ Vector(system.co_hair(evaluated, particle_no=index, step=0)))
        tips.append(groom.matrix_world @ Vector(system.co_hair(evaluated, particle_no=index, step=8)))
    _require(max((point - roots[0]).length for point in roots) > .1, "Native hair roots are unavailable or collapsed")
    tree = _surface(groom)
    distances = [tree.find_nearest(point)[3] for point in roots]
    _require(all(value is not None and math.isfinite(value) for value in distances), "Cannot locate native hair roots on emitter")
    return {"parent_count": count, "viewport_children": len(system.child_particles), "indices": indices,
            "roots": [list(point) for point in roots], "tips": [list(point) for point in tips],
            "max_root_surface_distance": max(distances), "coordinate_api": "ParticleSystem.co_hair on evaluated object"}


def _material(name, color):
    result = bpy.data.materials.new(name)
    result.diffuse_color = (*color, 1)
    result.use_nodes = True
    shader = result.node_tree.nodes.get("Principled BSDF")
    shader.inputs["Base Color"].default_value = (*color, 1)
    shader.inputs["Roughness"].default_value = .6
    return result


def _aim(obj, target):
    obj.rotation_euler = (Vector(target) - obj.location).to_track_quat("-Z", "Y").to_euler()


def probe(output):
    started = time.monotonic()
    out = Path(output)
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.device = "CPU"
    scene.cycles.samples = 16
    scene.cycles.use_denoising = True
    scene.render.threads_mode = "FIXED"
    scene.render.threads = 4
    scene.render.resolution_x = scene.render.resolution_y = 512
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    scene.render.film_transparent = False
    scene.view_settings.view_transform = "AgX"
    scene.world = bpy.data.worlds.new("NativeGroomStudio")
    scene.world.use_nodes = True
    scene.world.node_tree.nodes["Background"].inputs["Color"].default_value = (.2, .22, .27, 1)
    scene.world.node_tree.nodes["Background"].inputs["Strength"].default_value = .25

    bpy.ops.object.armature_add(enter_editmode=True)
    rig = bpy.context.object
    rig.name = "AnalyticTwoBoneRig"
    root = rig.data.edit_bones[0]
    root.name = "root"
    root.head, root.tail = (.5, 0, .12), (.5, 0, .73)
    tip = rig.data.edit_bones.new("tip")
    tip.head, tip.tail, tip.parent = (.5, 0, .73), (.5, 0, 1.36), root
    tip.use_connect = True
    bpy.ops.object.mode_set(mode="OBJECT")
    materials = [_material("IvoryFiber", (.62, .55, .42)), _material("BlueFiber", (.045, .17, .24))]
    bases = []
    for name, center, scale, material in (("analytic_sphere", (-.65, 0, .64), (.46, .46, .46), materials[0]),
                                           ("analytic_limb", (.5, 0, .75), (.19, .19, .62), materials[1])):
        bpy.ops.mesh.primitive_uv_sphere_add(segments=48, ring_count=32, location=center)
        obj = bpy.context.object
        obj.name, obj.scale = name, scale
        bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
        obj.data.materials.append(material)
        for polygon in obj.data.polygons:
            polygon.use_smooth = True
        groups = {name: obj.vertex_groups.new(name=name) for name in ("root", "tip")}
        for vertex in obj.data.vertices:
            weight = min(1., max(0., (vertex.co.z - .52) / .38)) if obj.name == "analytic_limb" else 0
            if weight < 1:
                groups["root"].add([vertex.index], 1 - weight, "REPLACE")
            if weight > 0:
                groups["tip"].add([vertex.index], weight, "REPLACE")
        modifier = obj.modifiers.new("AnalyticSkin", "ARMATURE")
        modifier.object = rig
        obj.parent = rig
        bases.append(obj)

    grooms, reports = [], []
    for obj, material, parents in zip(bases, materials, (2400, 1600)):
        groom, report = add_native_groom(obj, rig, {"parents": parents, "children": 11,
            "length": .035, "radius": .0004, "roughness": .003, "seed": 211}, material)
        grooms.append(groom)
        reports.append(report)
    _require(sum(row["render_fiber_upper_bound"] for row in reports) <= MAX_FIBERS, "Probe fiber cap exceeded")
    rig.pose.bones["tip"].rotation_mode = "XYZ"
    rig.pose.bones["tip"].rotation_euler = (0, 0, 0)
    rig.pose.bones["tip"].keyframe_insert(data_path="rotation_euler", frame=1)
    rig.pose.bones["tip"].rotation_euler.z = math.radians(-50)
    rig.pose.bones["tip"].keyframe_insert(data_path="rotation_euler", frame=24)
    scene.frame_start, scene.frame_end = 1, 24
    bpy.ops.mesh.primitive_plane_add(size=200)
    bpy.context.object.data.materials.append(_material("StudioFloor", (.035, .055, .075)))
    for name, position, power, size in (("Key", (-3, -4, 5), 900, 3), ("Fill", (3, -2, 2), 450, 3),
                                       ("Rim", (0, 3, 4), 1200, 2)):
        lamp = bpy.data.lights.new(name, "AREA")
        lamp.energy, lamp.shape, lamp.size = power, "DISK", size
        obj = bpy.data.objects.new(name, lamp)
        bpy.context.collection.objects.link(obj)
        obj.location = position
        _aim(obj, (0, 0, .7))
    camera = bpy.data.objects.new("ProofCamera", bpy.data.cameras.new("ProofCamera"))
    bpy.context.collection.objects.link(camera)
    camera.location = (2.4, -6, 2.3)
    camera.data.type, camera.data.ortho_scale = "ORTHO", 2.65
    _aim(camera, (-.06, 0, .74))
    scene.camera = camera
    scene.frame_set(1)
    bpy.context.view_layer.update()
    rest = {groom.name: _hair_snapshot(groom) for groom in grooms}
    _write(out / "rest-hair-samples.json", rest)
    for groom in grooms:
        groom.hide_render = True
    scene.render.filepath = str(out / "baseline-no-hair.png")
    bpy.ops.render.render(write_still=True)
    for groom in grooms:
        groom.hide_render = False
    scene.render.filepath = str(out / "fur-rest.png")
    bpy.ops.render.render(write_still=True)
    scene.frame_set(24)
    bpy.context.view_layer.update()
    articulated = {groom.name: _hair_snapshot(groom) for groom in grooms}
    _write(out / "articulated-hair-samples.json", articulated)
    scene.render.filepath = str(out / "fur-articulated.png")
    bpy.ops.render.render(write_still=True)
    checks = []
    for groom, base in zip(grooms, bases):
        before, after = rest[groom.name], articulated[groom.name]
        _require(before["indices"] == after["indices"], "Hair parent identities changed after articulation")
        displacement = max(math.dist(a, b) for a, b in zip(before["roots"], after["roots"], strict=True))
        surface_error = max(before["max_root_surface_distance"], after["max_root_surface_distance"])
        _require(surface_error < .003, "Native hair roots detached from deformed skin")
        _require(displacement > .1 if "limb" in groom.name else displacement < 1e-6,
                 "Native hair motion differs from the analytic two-bone pose")
        checks.append({"object": groom.name, "sampled_parents": len(before["indices"]),
                       "max_parent_root_displacement": displacement, "max_root_surface_distance": surface_error,
                       "copied_weights_preserved": _skin_signature(groom, rig) == _skin_signature(base, rig),
                       "status": "passed"})
    scene.frame_set(1)
    bpy.context.view_layer.update()
    bpy.context.preferences.filepaths.save_version = 0
    bpy.ops.wm.save_as_mainfile(filepath=str(out / "source.blend"), check_existing=False)
    names = [groom.name for groom in grooms]
    bpy.ops.wm.open_mainfile(filepath=str(out / "source.blend"), load_ui=False, use_scripts=False)
    reload_checks = []
    for name in names:
        obj = bpy.data.objects[name]
        system = obj.particle_systems[0]
        armatures = [modifier.object.name for modifier in obj.modifiers if modifier.type == "ARMATURE"]
        _require(system.settings.type == "HAIR" and armatures == ["AnalyticTwoBoneRig"]
                 and not obj.show_instancer_for_render and not system.use_hair_dynamics,
                 "Reloaded editable groom lost its native skin binding")
        reload_checks.append({"object": name, "hair_system": system.name, "parents": system.settings.count,
                              "armature": armatures[0], "editable_native_source": True})
    report = {"schema_version": 1, "status": "native_groom_probe_passed", "analytic_fixture": True,
              "blender_version": bpy.app.version_string, "gltf_export_supported": False,
              "configured_render_fiber_upper_bound": sum(row["render_fiber_upper_bound"] for row in reports),
              "grooms": reports, "deformation_checks": checks, "reload_checks": reload_checks,
              "render_settings": {"engine": "CYCLES", "device": "CPU", "threads": 4, "samples": 16,
                                  "dimensions": [512, 512], "view_transform": "AgX"},
              "elapsed_seconds": time.monotonic() - started,
              "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "scope": "Editable dense native hair on an analytic weighted sphere and limb; not a finished character or glTF deliverable",
              "files": {path.name: {"bytes": path.stat().st_size,
                                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                        for path in sorted(out.iterdir()) if path.suffix in (".png", ".blend")}}
    _write(out / "native-groom-proof.json", report)
    print(json.dumps({"status": report["status"], "fiber_upper_bound": report["configured_render_fiber_upper_bound"],
                      "elapsed_seconds": report["elapsed_seconds"]}))


def main():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inspect", action="store_true")
    parser.add_argument("--probe", action="store_true")
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)
    if not args.out.is_dir():
        raise ValueError("Output directory must already exist")
    if args.inspect:
        inspect_api(args.out)
    elif args.probe:
        try:
            probe(args.out)
        except Exception as error:
            _write(args.out / "unsupported.json", {"status": "unsupported_native_groom_probe", "error": str(error),
                                                   "blender_version": bpy.app.version_string})
            raise
    else:
        raise ValueError("Select --inspect or --probe")


if __name__ == "__main__":
    main()
