"""Data-configured texture-driven groom and accessory repair on a generated skin."""

import json
import math
from pathlib import Path
import sys

import bpy
import numpy as np
from mathutils import Vector, Euler
from mathutils.kdtree import KDTree

sys.path.insert(0, str(Path(__file__).parent))
from blender_verify import source_header, hash_file, write_json, imported_bindings, bind_action, clear_actions, set_frame
from blender_native_groom import add_native_groom
from motion_samples import sample_action


def texture_colors(obj, image):
    width, height = image.size
    if width*height > 16_777_216:
        raise ValueError("Base texture exceeds groom sampling limit")
    pixels = np.empty(width*height*4, dtype=np.float32)
    image.pixels.foreach_get(pixels)
    pixels = pixels.reshape(height, width, 4)
    uv = np.zeros((len(obj.data.vertices), 2), dtype=np.float32)
    seen = np.zeros(len(obj.data.vertices), dtype=bool)
    for loop in obj.data.loops:
        if not seen[loop.vertex_index]:
            uv[loop.vertex_index] = obj.data.uv_layers.active.data[loop.index].uv
            seen[loop.vertex_index] = True
    if not seen.all():
        raise ValueError("Unreferenced source vertices have no texture coordinates")
    xx = np.minimum(width-1, (np.mod(uv[:, 0], 1)*width).astype(int))
    yy = np.minimum(height-1, (np.mod(uv[:, 1], 1)*height).astype(int))
    return pixels[yy, xx, :3].copy()


def frame_mask(points, low, span, boxes):
    normalized = (points-low)/np.maximum(span, 1e-8)
    excluded = np.zeros(len(points), dtype=bool)
    for box in boxes:
        center, radius = np.asarray(box["center"]), np.asarray(box["radius"])
        excluded |= np.sum(((normalized-center)/radius)**2, axis=1) < 1
    return excluded


def smooth_surface(obj, mask, iterations, strength):
    original = np.array([tuple(v.co) for v in obj.data.vertices])
    points, inverse = np.unique(original, axis=0, return_inverse=True)
    edges = np.array([tuple(e.vertices) for e in obj.data.edges])
    edges = np.unique(np.sort(inverse[edges], axis=1), axis=0)
    left, right = edges[:, 0], edges[:, 1]
    degree = np.bincount(edges.ravel(), minlength=len(points)).clip(1)
    movable = np.ones(len(points), dtype=bool)
    np.logical_and.at(movable, inverse, mask)
    for _ in range(iterations):
        total = np.zeros_like(points)
        np.add.at(total, left, points[right])
        np.add.at(total, right, points[left])
        points[movable] += strength*(total/degree[:, None]-points)[movable]
    result = points[inverse]
    obj.data.vertices.foreach_set('co', result.astype(np.float32).ravel())
    obj.data.update()
    obj.data.calc_loop_triangles()
    normals = np.zeros_like(points)
    for triangle in obj.data.loop_triangles:
        a, b, c = inverse[list(triangle.vertices)]
        cross = np.cross(points[b]-points[a], points[c]-points[a])
        normals[[a, b, c]] += cross
    normals /= np.maximum(np.linalg.norm(normals, axis=1)[:, None], 1e-12)
    obj.data.normals_split_custom_set_from_vertices(normals[inverse].tolist())
    for polygon in obj.data.polygons:
        polygon.use_smooth = True
    return {'iterations': iterations, 'strength': strength,
            'max_vertex_shift_m': float(np.linalg.norm(result-original, axis=1).max()),
            'topology_uv_and_weights_preserved': True, 'authored_not_recovered': True}


def constrain_weights(obj, rig, points, low, span, config):
    names = {g.index: g.name for g in obj.vertex_groups}
    factors = {}
    normalized = (points-low)/span
    for name, region in config['regions'].items():
        if name not in rig.data.bones:
            raise ValueError('Weight constraint names an absent bone')
        distance = np.linalg.norm((normalized-region['center'])/region['radius'], axis=1)
        factors[name] = np.clip((1-distance)/.25, 0, 1)
    fallback = config['fallback_bone']
    if fallback not in rig.data.bones:
        raise ValueError('Weight fallback bone is absent')
    grouped, changed = {}, 0
    for index, vertex in enumerate(obj.data.vertices):
        before = {names[g.group]: g.weight for g in vertex.groups if g.weight > 0}
        after = {name: weight*(factors[name][index] if name in factors else 1)
                 for name, weight in before.items()}
        displaced = max(0., 1-sum(after.values()))
        after[fallback] = after.get(fallback, 0)+displaced
        after = dict(sorted(after.items(), key=lambda pair: -pair[1])[:4])
        total = sum(after.values())
        changed += displaced > 1e-5
        for name, weight in after.items():
            if weight > 0:
                grouped.setdefault(name, []).append((index, float(weight/total)))
    obj.vertex_groups.clear()
    for name, values in grouped.items():
        group = obj.vertex_groups.new(name=name)
        for index, weight in values:
            group.add([index], weight, 'REPLACE')
    return {'changed_vertices': int(changed), 'config': config, 'method': 'Bound provider bone weights to authored ellipsoids; transfer removed weight to torso',
            'authored_not_recovered': True}


def author_actions(rig, designs):
    for design in designs:
        for key in design['keys']:
            if not set(key['rotations']) <= set(rig.pose.bones.keys()):
                raise ValueError('Action names an absent bone; refusing silently ignored controls')
    if rig.animation_data:
        rig.animation_data_clear()
    for action in list(bpy.data.actions):
        bpy.data.actions.remove(action)
    rig.animation_data_create()
    bpy.context.scene.render.fps = 24
    for design in designs:
        action = bpy.data.actions.new(design['id'])
        rig.animation_data.action = action
        for key in design['keys']:
            for bone in rig.pose.bones:
                world_basis = (rig.matrix_world @ bone.bone.matrix_local).to_3x3().normalized()
                rotation = Euler([math.radians(x) for x in key['rotations'].get(bone.name, [0, 0, 0])]).to_matrix()
                bone.rotation_mode = 'QUATERNION'
                bone.rotation_quaternion = (world_basis.inverted() @ rotation @ world_basis).to_quaternion()
                bone.location = (0, 0, 0)
                bone.scale = (1, 1, 1)
                bone.keyframe_insert(data_path='rotation_quaternion', frame=key['second']*24, group=bone.name)
        action.use_fake_user = True
    rig.animation_data.action = None


def portable_coat(obj, rig, mask, colors, config, out):
    from blender_fur_cards import card_geometry, attach_fur_material
    obj.data.calc_loop_triangles()
    triangles = [tuple(t.vertices) for t in obj.data.loop_triangles if all(mask[i] for i in t.vertices)]
    vertices = [tuple(v.co) for v in obj.data.vertices]
    normals = [tuple(v.normal) for v in obj.data.vertices]
    names = {g.index: g.name for g in obj.vertex_groups}
    weights = [{names[g.group]: g.weight for g in v.groups if g.weight > 0} for v in obj.data.vertices]
    meshes, reports = [], []
    for index, band in enumerate(config['bands']):
        selected = [t for t in triangles if band['brightness'][0] <= float(colors[list(t)].mean()) < band['brightness'][1]]
        settings = {**config['settings'], 'count': band['count'], 'seed': config['settings']['seed']+index}
        data = card_geometry(vertices, selected, weights, settings, normals)
        mesh = bpy.data.meshes.new('PortableCoat_'+str(index))
        mesh.from_pydata(data['vertices'], [], data['faces'])
        mesh.update()
        uv = mesh.uv_layers.new(name='UVMap')
        for loop in mesh.loops:
            uv.data[loop.index].uv = data['uv'][loop.vertex_index]
        mesh.normals_split_custom_set_from_vertices(data['normals'])
        material, report = attach_fur_material(mesh.name, {'color': band['color'], 'fur_atlas': 'distance-v1'}, out)
        mesh.materials.append(material)
        cards = bpy.data.objects.new(mesh.name, mesh)
        bpy.context.scene.collection.objects.link(cards)
        cards.matrix_world = obj.matrix_world.copy()
        groups = {name: cards.vertex_groups.new(name=name) for name in rig.data.bones.keys()}
        for vertex, row in enumerate(data['weights']):
            for name, weight in row.items():
                groups[name].add([vertex], weight, 'REPLACE')
        cards.modifiers.new('PortableSkin', 'ARMATURE').object = rig
        meshes.append(cards)
        reports.append({'cards': band['count'], 'vertices': len(data['vertices']), 'material': report, 'config': band})
    return meshes, reports


def finish(config, output):
    cfg, out = json.loads(Path(config).read_text()), Path(output)
    model = Path(cfg["source"])
    source_header(model)
    if hash_file(model) != cfg["source_sha256"]:
        raise ValueError("Source rig changed")
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.gltf(filepath=str(model))
    rigs = [o for o in bpy.context.scene.objects if o.type == "ARMATURE"]
    helpers = {b.custom_shape for rig in rigs for b in rig.pose.bones if b.custom_shape}
    meshes = [o for o in bpy.context.scene.objects if o.type == "MESH" and o not in helpers]
    if len(meshes) != 1 or len(meshes[0].data.vertices) > 100000:
        raise ValueError("Current texture-groom profile requires one mesh under 100000 vertices")
    obj = meshes[0]
    authored_rig = None
    if cfg.get('authored_rig'):
        if rigs:
            raise ValueError('Cannot replace an existing armature with an authored creature rig')
        from blender_creature_rig import create_rig
        rig, authored_rig = create_rig(obj, cfg['authored_rig'])
    else:
        if len(rigs) != 1:
            raise ValueError('Finishing requires one source armature or an explicit authored creature rig')
        rig = rigs[0]
    rig.data.pose_position = 'REST'
    points = np.array([tuple(obj.matrix_world @ v.co) for v in obj.data.vertices])
    # Meshy puts centimeter scale on the parent; applying only local scale is a no-op.
    world = obj.matrix_world.copy()
    obj.parent = None
    obj.matrix_world = world
    bpy.context.view_layer.update()
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    bpy.context.view_layer.update()
    after_apply = np.array([tuple(obj.matrix_world @ v.co) for v in obj.data.vertices])
    if np.max(np.abs(after_apply-points)) > 1e-6:
        raise ValueError("Applying emitter scale changed the world rest geometry")
    low, high = points.min(axis=0), points.max(axis=0)
    base = obj.data.materials[0]
    bsdf = next(n for n in base.node_tree.nodes if n.type == "BSDF_PRINCIPLED")
    color_node = bsdf.inputs["Base Color"].links[0].from_node
    if color_node.type != "TEX_IMAGE":
        raise ValueError("Texture-driven finish requires an explicit base-color image")
    colors = texture_colors(obj, color_node.image)
    mask = ~frame_mask(points, low, high-low, cfg.get('groom_exclusions', []))
    mask &= colors.max(axis=1)-colors.min(axis=1) < cfg.get('fur_max_chroma', 1)
    smoothing = None
    if cfg.get('smoothing'):
        smoothing = smooth_surface(obj, mask, **cfg['smoothing'])
        points = np.array([tuple(obj.matrix_world @ v.co) for v in obj.data.vertices])
    for material in obj.data.materials:
        for node in material.node_tree.nodes:
            if node.type == "NORMAL_MAP":
                node.inputs["Strength"].default_value = cfg["normal_strength"]
    report = {"schema_version": 1, "source_sha256": cfg["source_sha256"], "blender_version": bpy.app.version_string,
              "normal_strength": cfg["normal_strength"], "groom": None, "attachment": None,
              "smoothing": smoothing, 'authored_rig': authored_rig,
              "emitter_scale_applied": True, "world_rest_position_max_error": float(np.max(np.abs(after_apply-points))),
              "source_bounds_blender_z_up": {"min": low.tolist(), "max": high.tolist()},
              "animation_verified": False, "weft_runtime_verified": False}
    if cfg.get('weight_constraints'):
        report['weight_constraints'] = constrain_weights(obj, rig, points, low, high-low, cfg['weight_constraints'])
    attachment = cfg.get("attachment")
    if attachment:
        # Color plus an authored spatial bound avoids assigning unrelated green details.
        inside = frame_mask(points, low, high-low, attachment["regions"])
        channel = attachment["dominant_channel"]
        others = np.max(colors[:, [i for i in range(3) if i != channel]], axis=1)
        selected = inside & (colors[:, channel] > others*attachment["dominant_ratio"])
        if attachment.get('expand_meters'):
            tree = KDTree(int(selected.sum()))
            for number, index in enumerate(np.flatnonzero(selected)):
                tree.insert(points[index], number)
            tree.balance()
            for index in np.flatnonzero(inside & ~selected):
                selected[index] = tree.find(points[index])[2] <= attachment['expand_meters']
        if not attachment["min_vertices"] <= int(selected.sum()) <= attachment["max_vertices"]:
            raise ValueError("Attachment selector missed its bounded expected region")
        bpy.context.view_layer.objects.active = rig
        bpy.ops.object.select_all(action="DESELECT")
        rig.select_set(True)
        bpy.ops.object.mode_set(mode="EDIT")
        parent = rig.data.edit_bones[attachment["parent_bone"]]
        anchor = rig.data.edit_bones.new(attachment["bone"])
        anchor.head = rig.data.edit_bones[attachment["anchor_from_bone"]].head
        anchor.tail = anchor.head+Vector((0, 0, 3))
        anchor.parent = parent
        bpy.ops.object.mode_set(mode="OBJECT")
        ids = np.flatnonzero(selected).tolist()
        for group in obj.vertex_groups:
            group.remove(ids)
        group = obj.vertex_groups.new(name=attachment["bone"])
        group.add(ids, 1., "REPLACE")
        report["attachment"] = {"bone": attachment["bone"], "parent_bone": attachment["parent_bone"],
                                 "selected_vertices": len(ids), "selector": attachment,
                                 "rest_geometry_changed": False,
                                 "limitation": "Color-selected accessory region; inspect boundary vertices and deformation before acceptance"}
    if cfg.get("groom"):
        if int(mask.sum()) < 1000:
            raise ValueError("Fur mask selected too few source vertices")
        settings = {k: v for k, v in cfg["groom"].items() if k not in ("direction", "comb")}
        groom, groom_report = add_native_groom(obj, rig, settings, base)
        density = groom.vertex_groups.new(name="AuthoredFurDensity")
        density.add(np.flatnonzero(mask).tolist(), 1., "REPLACE")
        system = groom.particle_systems[0]
        system.vertex_group_density = density.name
        system.settings.clump_factor = cfg["fur_clump"]
        bpy.context.view_layer.update()
        if "direction" in cfg["groom"]:
            from blender_comb_groom import comb_native_groom
            groom_report["comb"] = comb_native_groom(groom, cfg["groom"]["direction"], cfg["groom"]["comb"])
        hair_material = groom.data.materials[-1]
        hair = next(n for n in hair_material.node_tree.nodes if n.type == "BSDF_HAIR_PRINCIPLED")
        texture = hair_material.node_tree.nodes.new("ShaderNodeTexImage")
        texture.image = color_node.image
        hair_material.node_tree.links.new(texture.outputs["Color"], hair.inputs["Color"])
        groom_report["material"]["texture_color_sampling"] = True
        groom_report.update(density_selected_vertices=int(mask.sum()), density_excluded_vertices=int((~mask).sum()),
                            clump_factor=cfg["fur_clump"], mask_config=cfg["groom_exclusions"], fur_max_chroma=cfg["fur_max_chroma"])
        report["groom"] = groom_report
        groom_report['limitations'] = [line for line in groom_report['limitations']
                                       if not line.startswith('Material uses the source base-color')]
        if 'native_render_children' in cfg:
            system.settings.rendered_child_count = cfg['native_render_children']
            groom_report['render_children_override_after_comb'] = cfg['native_render_children']
            groom_report['render_fiber_upper_bound'] = cfg['groom']['parents']*(cfg['native_render_children']+1)
            groom_report['native_parameters']['render_children'] = cfg['native_render_children']
    cards = []
    if cfg.get('portable_fur'):
        cards, report['portable_fur'] = portable_coat(obj, rig, mask, colors, cfg['portable_fur'], out)
    if cfg.get('authored_actions'):
        author_actions(rig, cfg['authored_actions'])
        report['motion_origin'] = 'Explicit locally authored actions; source provider clips are not used or relabeled'
    actions = list(bpy.data.actions)
    bindings, owners = imported_bindings(list(bpy.context.scene.objects), actions)
    if cfg.get('authored_actions'):
        bindings = {a.name: [(rig, next(iter(a.slots), None))] for a in actions}
    report['states'] = []
    if cfg.get('export_model'):
        clear_actions(owners, [rig])
        rig.data.pose_position = 'POSE'
        for action in actions:
            action.use_fake_user = True
        bpy.ops.object.select_all(action='DESELECT')
        obj.select_set(True)
        rig.select_set(True)
        for card in cards:
            card.select_set(True)
        bpy.context.view_layer.objects.active = rig
        bpy.ops.export_scene.gltf(filepath=str(out/'model.glb'), export_format='GLB', use_selection=True,
            export_animations=True, export_animation_mode='ACTIONS', export_anim_single_armature=True,
            export_force_sampling=True, export_frame_range=False, export_yup=True, export_skins=True,
            export_normals=True, export_tangents=True, export_image_format='AUTO', export_morph=False,
            export_extras=False, export_materials='EXPORT', export_current_frame=False)
        rig.data.pose_position = 'REST'
    for card in cards:
        card.hide_render = True
    for image in bpy.data.images:
        if image.size[0]:
            image.pack()
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.device, scene.cycles.samples = "CPU", cfg["samples"]
    scene.cycles.use_denoising = cfg.get('portrait_denoise', True)
    scene.render.threads_mode, scene.render.threads = "FIXED", 4
    scene.render.resolution_x = scene.render.resolution_y = cfg["resolution"]
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format, scene.render.image_settings.color_mode = "PNG", "RGB"
    scene.view_settings.view_transform = "AgX"
    world = bpy.data.worlds.new("SoftStudio")
    scene.world, world.use_nodes = world, True
    nodes, links = world.node_tree.nodes, world.node_tree.links
    ambient = nodes.get("Background")
    ambient.inputs["Color"].default_value, ambient.inputs["Strength"].default_value = (.8, .8, .8, 1), .3
    camera_background = nodes.new("ShaderNodeBackground")
    camera_background.inputs["Color"].default_value = (1, 1, 1, 1)
    camera_background.inputs["Strength"].default_value = 6
    path = nodes.new("ShaderNodeLightPath")
    mix = nodes.new("ShaderNodeMixShader")
    links.new(path.outputs["Is Camera Ray"], mix.inputs[0])
    links.new(ambient.outputs[0], mix.inputs[1])
    links.new(camera_background.outputs[0], mix.inputs[2])
    links.new(mix.outputs[0], nodes.get("World Output").inputs[0])
    center, extent = Vector((low+high)/2), float(max(high-low))
    for name, offset, power in (("Key", (-2, -3, 4), 650), ("Fill", (3, -2, 2), 250), ("Rim", (1, 3, 3), 450)):
        data = bpy.data.lights.new(name, "AREA")
        data.energy, data.size = power*extent*extent, 2*extent
        lamp = bpy.data.objects.new(name, data)
        scene.collection.objects.link(lamp)
        lamp.location = center+Vector(offset)*extent
        lamp.rotation_euler = (center-lamp.location).to_track_quat("-Z", "Y").to_euler()
    camera_data = bpy.data.cameras.new("PortraitCamera")
    camera_data.type, camera_data.ortho_scale = "ORTHO", extent*1.14
    camera = bpy.data.objects.new("PortraitCamera", camera_data)
    scene.collection.objects.link(camera)
    scene.camera = camera
    viewpoints = {'front':(0, -3, .04), 'three_quarter':(1.5, -3, .12), 'three_quarter_left':(-1.8, -3, .12)}
    viewpoint = viewpoints[cfg.get('camera_view', 'front')]
    camera.location = center+Vector(viewpoint)*extent
    camera.rotation_euler = (center-camera.location).to_track_quat("-Z", "Y").to_euler()
    if 'camera_view' in cfg:
        basis = camera.rotation_euler.to_matrix()
        right, up = basis@Vector((1, 0, 0)), basis@Vector((0, 1, 0))
        horizontal, vertical = points@np.array(right), points@np.array(up)
        camera_data.ortho_scale = max(float(np.ptp(horizontal)), float(np.ptp(vertical)))*1.2
        camera.location += right*(float((horizontal.min()+horizontal.max())/2)-center.dot(right))
        camera.location += up*(float((vertical.min()+vertical.max())/2)-center.dot(up))
    report['camera'] = {'view':cfg.get('camera_view','front'), 'orthographic_scale':camera_data.ortho_scale}
    scene.render.filepath = str(out/"portrait.png")
    bpy.ops.render.render(write_still=True)
    if cfg.get('sample_states'):
        scene.render.resolution_x = scene.render.resolution_y = 384
        scene.cycles.samples = 8
        scene.cycles.use_denoising = True
        rig.data.pose_position = 'POSE'
        authored = {design['id']: design for design in cfg.get('authored_actions', [])}
        for action in actions:
            bind_action(action, bindings, owners, [rig])
            if authored and action.name not in authored:
                raise ValueError('Authored action has no matching evidence-sampling design')
            start, end = map(float, action.frame_range)
            if end <= start:
                raise ValueError('Static provider action cannot be a completed state')
            folder = out/'renders'/action.name
            folder.mkdir(parents=True)
            sampling = sample_action(authored.get(action.name), (start, end),
                                     scene.render.fps/scene.render.fps_base)
            for index, frame in enumerate(sampling['frames']):
                set_frame(frame)
                scene.render.filepath = str(folder/f'{index:04d}.png')
                bpy.ops.render.render(write_still=True)
            report['states'].append({'id': action.name, **sampling,
                'duration_seconds': (end-start)/(scene.render.fps/scene.render.fps_base), 'loop_verified': False})
        clear_actions(owners, [rig])
        rig.data.pose_position = 'REST'
    scene.render.resolution_x = scene.render.resolution_y = cfg['resolution']
    scene.cycles.samples = cfg['samples']
    bpy.ops.wm.save_as_mainfile(filepath=str(out/"source.blend"), check_existing=False)
    report["source_unchanged"] = hash_file(model) == cfg["source_sha256"]
    write_json(out/"finish-report.json", report)


if __name__ == "__main__":
    args = sys.argv[sys.argv.index("--")+1:]
    finish(args[0], args[1])
