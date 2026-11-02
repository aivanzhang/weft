"""Data-configured texture-driven groom and accessory repair on a generated skin."""

import hashlib
import json
import math
from pathlib import Path
import sys
import time

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


def shared_color_texture(materials):
    selected, signature = None, None
    for material in materials:
        shaders = [n for n in material.node_tree.nodes if n.type == 'BSDF_PRINCIPLED'] if material and material.use_nodes else []
        if len(shaders) != 1 or len(shaders[0].inputs['Base Color'].links) != 1:
            raise ValueError('Texture-driven finish requires one directly textured base-color shader per source material')
        node = shaders[0].inputs['Base Color'].links[0].from_node
        if node.type != 'TEX_IMAGE' or node.image is None:
            raise ValueError('Texture-driven finish requires an explicit base-color image')
        # The bounded sampler reads the active UV layer, not arbitrary shader coordinates.
        if node.inputs['Vector'].links or node.projection != 'FLAT' or node.extension != 'REPEAT':
            raise ValueError('Custom UV routes, projections or addressing are unsupported for texture-driven selection')
        current = node.image
        if selected is not None and current != signature:
            raise ValueError('Heterogeneous base-color textures are unsupported for texture-driven selection')
        selected, signature = node, current
    if selected is None:
        raise ValueError('Texture-driven finish requires a source material')
    return selected


def frame_mask(points, low, span, boxes):
    normalized = (points-low)/np.maximum(span, 1e-8)
    excluded = np.zeros(len(points), dtype=bool)
    for box in boxes:
        center, radius = np.asarray(box["center"]), np.asarray(box["radius"])
        excluded |= np.sum(((normalized-center)/radius)**2, axis=1) < 1
    return excluded


def smooth_surface(obj, mask, iterations, strength, report_dir):
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
    integrity = set_surface_positions(obj, original, result, report_dir)
    return {'iterations': iterations, 'strength': strength,
            'max_vertex_shift_m': float(np.linalg.norm(result-original, axis=1).max()),
            'topology_uv_and_weights_preserved': True, 'authored_not_recovered': True,
            'surface_integrity': integrity}


def set_surface_positions(obj, original, result, report_dir):
    from surface_integrity import enforce_surface_change, SurfaceChangeError
    # Validate the stored geometry: float32 rounding can collapse a small valid face.
    stored = result.astype(np.float32)
    result = stored.astype(np.float64)
    obj.data.calc_loop_triangles()
    triangles = np.array([tuple(t.vertices) for t in obj.data.loop_triangles])
    try:
        integrity = enforce_surface_change(original, result, triangles)
    except SurfaceChangeError as error:
        write_json(report_dir/'surface-integrity.json', error.report)
        raise
    write_json(report_dir/'surface-integrity.json', integrity)
    _, indices, inverse = np.unique(original, axis=0, return_index=True, return_inverse=True)
    points = result[indices]
    obj.data.vertices.foreach_set('co', stored.ravel())
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
    return integrity


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
    return {'changed_vertices': int(changed), 'config': config, 'method': 'Bound provider bone weights to authored ellipsoids; transfer removed weight to the configured fallback bone',
            'authored_not_recovered': True}


def author_actions(rig, designs):
    for design in designs:
        for key in design['keys']:
            if not (set(key['rotations']) | set(key.get('translations_m', {}))) <= set(rig.pose.bones.keys()):
                raise ValueError('Action names an absent bone; refusing silently ignored controls')
    if rig.animation_data:
        rig.animation_data_clear()
    for action in list(bpy.data.actions):
        bpy.data.actions.remove(action)
    rig.animation_data_create()
    bpy.context.scene.render.fps = 24
    translated = {name for design in designs for key in design['keys'] for name in key.get('translations_m', {})}
    for design in designs:
        action = bpy.data.actions.new(design['id'])
        rig.animation_data.action = action
        for key in design['keys']:
            for bone in rig.pose.bones:
                full_basis = (rig.matrix_world @ bone.bone.matrix_local).to_3x3()
                world_basis = full_basis.normalized()
                rotation = Euler([math.radians(x) for x in key['rotations'].get(bone.name, [0, 0, 0])]).to_matrix()
                bone.rotation_mode = 'QUATERNION'
                bone.rotation_quaternion = (world_basis.inverted() @ rotation @ world_basis).to_quaternion()
                # Meter offsets use rest axes and follow the animated parent; retain object scale conversion.
                bone.location = full_basis.inverted() @ Vector(key.get('translations_m', {}).get(bone.name, [0, 0, 0]))
                bone.scale = (1, 1, 1)
                bone.keyframe_insert(data_path='rotation_quaternion', frame=key['second']*24, group=bone.name)
                if bone.name in translated:
                    bone.keyframe_insert(data_path='location', frame=key['second']*24, group=bone.name)
        if design.get('interpolation') == 'LINEAR':
            for curve in action.fcurves:
                for point in curve.keyframe_points:
                    point.interpolation = 'LINEAR'
        action.use_fake_user = True
    rig.animation_data.action = None


def fit_motion_camera(low, high, basis, location, near, far):
    corners = np.array([[x,y,z] for x in (low[0], high[0])
                        for y in (low[1], high[1]) for z in (low[2], high[2])])
    right, up, forward = basis[:, 0], basis[:, 1], -basis[:, 2]
    horizontal, vertical = corners@right, corners@up
    location = np.asarray(location, dtype=float).copy()
    location += right*((horizontal.min()+horizontal.max())/2-location@right)
    location += up*((vertical.min()+vertical.max())/2-location@up)
    scale = max(float(np.ptp(horizontal)), float(np.ptp(vertical)))*1.2
    depths = corners@forward-location@forward
    clearance = max(.05, scale*.05)
    back_shift = max(0., near+clearance-float(depths.min()))
    location -= forward*back_shift
    far = max(far, float(depths.max())+back_shift+clearance)
    return {'location': location.tolist(), 'orthographic_scale': scale, 'clip_end': far,
            'sampled_depth_min_m': float(depths.min())+back_shift,
            'sampled_depth_max_m': float(depths.max())+back_shift, 'camera_back_shift_m': back_shift}


def motion_bounds(obj, rig, actions, bindings, owners, designs):
    """Fit one fixed camera to bounded pose samples; never claim continuous-motion coverage."""
    start = time.monotonic()
    scene = bpy.context.scene
    previous_frame = scene.frame_current + scene.frame_subframe
    low, high = np.full(3, np.inf), np.full(3, -np.inf)
    frames_by_action = {}
    lookup = {design['id']: design for design in designs}
    hidden = [(other, other.hide_get()) for other in scene.objects if other.type == 'MESH' and other != obj]
    try:
        # Bounds need only the base skin, not repeated particle/card dependency work.
        for other, _ in hidden:
            other.hide_set(True)
        rig.data.pose_position = 'POSE'
        for action in actions:
            keys = [key['second'] for key in lookup[action.name]['keys']]
            seconds = sorted(set(keys + [(a+b)/2 for a, b in zip(keys, keys[1:])]))
            if len(seconds) > 63:
                raise ValueError('Motion framing exceeds 63 samples per action')
            bind_action(action, bindings, owners, [rig])
            for second in seconds:
                if time.monotonic()-start > 30:
                    raise ValueError('Motion framing exceeds 30 seconds')
                set_frame(second*scene.render.fps/scene.render.fps_base)
                evaluated = obj.evaluated_get(bpy.context.evaluated_depsgraph_get())
                mesh = evaluated.to_mesh()
                try:
                    points = np.empty((len(mesh.vertices), 3), dtype=np.float64)
                    mesh.vertices.foreach_get('co', points.ravel())
                    matrix = np.asarray(evaluated.matrix_world)
                    points = points@matrix[:3, :3].T + matrix[:3, 3]
                    if not len(points) or not np.isfinite(points).all():
                        raise ValueError('Invalid posed framing geometry')
                    low = np.minimum(low, points.min(axis=0))
                    high = np.maximum(high, points.max(axis=0))
                finally:
                    evaluated.to_mesh_clear()
            frames_by_action[action.name] = seconds
    finally:
        clear_actions(owners, [rig])
        rig.data.pose_position = 'REST'
        set_frame(previous_frame)
        for other, was_hidden in hidden:
            other.hide_set(was_hidden)
    return low, high, {'sample_seconds': frames_by_action,
        'world_bounds_m': {'min': low.tolist(), 'max': high.tolist()},
        'elapsed_seconds': time.monotonic()-start,
        'scope': 'Base skin at every authored key and midpoint, plus explicit fur margin; not continuous bounds or physics'}


def tuft_sampling_context(obj, image, triangles, groom, flow):
    from fur_root_sampling import RootColorSampler
    from groom_flow import prepare_flow

    width, height = image.size
    if (width <= 0 or height <= 0 or width*height > 16_777_216
            or image.colorspace_settings.name != 'sRGB'):
        raise ValueError('Tuft root sampling requires a bounded explicit sRGB image')
    if not 1 <= len(obj.data.vertices) <= 100000 or not 1 <= len(triangles) <= 100000:
        raise ValueError('Tuft sampling requires bounded nonempty emitter geometry')
    vertices = [tuple(v.co) for v in obj.data.vertices]
    low = [min(v[i] for v in vertices) for i in range(3)]
    span = [max(v[i] for v in vertices)-low[i] for i in range(3)]
    # Native grooming copies the entire emitter before applying its density mask.
    prepared = prepare_flow(low, span, groom['direction'], groom['comb'], flow)
    uv = obj.data.uv_layers.active
    if uv is None:
        raise ValueError('Tuft root sampling requires the source active UV layer')
    triangle_uvs = np.asarray([[tuple(uv.data[i].uv) for i in t.loops] for t in triangles], dtype=np.float32)
    pixels = np.empty(width*height*4, dtype=np.float32)
    image.pixels.foreach_get(pixels)
    pixels = pixels.reshape(height, width, 4)
    sampler = RootColorSampler(pixels, triangle_uvs, pixels_are_linear=False)
    return sampler, prepared, {
        'flow_bounds_local': {'min': low, 'span': span}, 'flow_bounds_scope': 'Entire rest emitter, identical to native groom copy',
        'texture_dimensions': [width, height], 'borrowed_pixel_buffer_bytes': pixels.nbytes,
        'source_pixel_interpretation': 'Existing encoded-sRGB image.pixels convention; bottom-up rows',
        'output_color_space': 'linear RGB after per-texel sRGB decode and bilinear filtering',
        'uv_source': 'Actual active UV corners of each selected loop triangle; never first-loop vertex UV',
        'sampling_triangles': len(triangles), 'buffer_lifetime': 'Released when portable_coat returns, before native grooming',
    }


def portable_coat(obj, rig, mask, colors, config, out, *, tuft_image=None, native_groom=None, groom_flow=None):
    from blender_fur_cards import card_geometry, attach_fur_material
    obj.data.calc_loop_triangles()
    tuft = config['settings'].get('profile') == 'tuft-v1'
    sampler, prepared_flow, sampling_report = None, None, None
    if tuft:
        if tuft_image is None or native_groom is None or groom_flow is None or config['settings'].get('sample_root_color') is not True:
            raise ValueError('Tuft coat requires source image, exact root sampling and shared native regional flow')
        triangle_rows = [t for t in obj.data.loop_triangles if all(mask[i] for i in t.vertices)]
        sampler, prepared_flow, sampling_report = tuft_sampling_context(obj, tuft_image, triangle_rows, native_groom, groom_flow)
        triangles = [tuple(t.vertices) for t in triangle_rows]
    else:
        triangles = [tuple(t.vertices) for t in obj.data.loop_triangles if all(mask[i] for i in t.vertices)]
    vertices = [tuple(v.co) for v in obj.data.vertices]
    normal_field = obj.data.attributes.get('AuthoredSurfaceNormal')
    normals = ([tuple(value.vector) for value in normal_field.data] if normal_field is not None
               else [tuple(v.normal) for v in obj.data.vertices])
    names = {g.index: g.name for g in obj.vertex_groups}
    weights = [{names[g.group]: g.weight for g in v.groups if g.weight > 0} for v in obj.data.vertices]
    meshes, reports = [], []
    for index, band in enumerate(config['bands']):
        if tuft:
            selected_indices = [i for i, t in enumerate(triangles)
                                if band['brightness'][0] <= float(colors[list(t)].mean()) < band['brightness'][1]]
            selected = [triangles[i] for i in selected_indices]
        else:
            selected = [t for t in triangles if band['brightness'][0] <= float(colors[list(t)].mean()) < band['brightness'][1]]
        settings = {**config['settings'], 'count': band['count'], 'seed': config['settings']['seed']+index}
        sampled = None
        if tuft:
            data = card_geometry(vertices, selected, weights, settings, normals, profile='tuft-v1',
                                 prepared_flow=prepared_flow,
                                 root_color_sampler=lambda ordinal, bary: sampler(selected_indices[ordinal], bary))
        elif settings.get('sample_root_color', False):
            # Byte sRGB image pixels are encoded values; glTF vertex colors are linear.
            sampled = np.where(colors <= .04045, colors/12.92, ((colors+.055)/1.055)**2.4)
        if not tuft:
            data = card_geometry(vertices, selected, weights, settings, normals,
                                 profile=settings.get('profile', 'legacy'), colors=sampled)
        mesh = bpy.data.meshes.new('PortableCoat_'+str(index))
        mesh.from_pydata(data['vertices'], [], data['faces'])
        mesh.update()
        uv = mesh.uv_layers.new(name='UVMap')
        for loop in mesh.loops:
            uv.data[loop.index].uv = data['uv'][loop.vertex_index]
        mesh.normals_split_custom_set_from_vertices(data['normals'])
        material, report = attach_fur_material(mesh.name,
            {'color': [1,1,1,1] if sampled is not None or tuft else band['color'],
             'fur_atlas': 'tuft-v1' if tuft else 'distance-v1'}, out)
        if sampled is not None or tuft:
            attr = mesh.color_attributes.new(name='FurRootColor', type='FLOAT_COLOR', domain='CORNER')
            for loop in mesh.loops:
                attr.data[loop.index].color = (*data['colors'][loop.vertex_index], 1.)
            nodes, links = material.node_tree.nodes, material.node_tree.links
            color = nodes.new('ShaderNodeVertexColor')
            color.layer_name = 'FurRootColor'
            # glTF recognizes the current Mix node, not legacy MixRGB vertex-color multiplication.
            mix = nodes.new('ShaderNodeMix')
            mix.data_type = 'RGBA'
            mix.blend_type = 'MULTIPLY'
            mix.inputs['Factor'].default_value = 1.
            bsdf = nodes.get('Principled BSDF')
            atlas = bsdf.inputs['Base Color'].links[0].from_socket
            links.new(atlas, mix.inputs[6])
            links.new(color.outputs['Color'], mix.inputs[7])
            links.new(mix.outputs[2], bsdf.inputs['Base Color'])
            report['root_color'] = ('Exact sampled triangle UV, sRGB texels decoded before bilinear interpolation; linear constant per tuft'
                                    if tuft else 'Barycentric source sRGB converted to linear vertex color; constant per ribbon')
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
        reports.append({'cards': band['count'], 'vertices': len(data['vertices']), 'material': report, 'config': band,
                        'geometry_profile': settings.get('profile','legacy'),
                        'sample_root_color': sampled is not None or tuft,
                        **({'sampling': sampling_report, 'root_summary': data['root_summary'],
                            'root_samples': data['root_samples']} if tuft else {})})
    return meshes, reports


def deformation_baseline(obj, rig, rig_route):
    """Bind rest coordinates before grooming; source skins need the same evidence as authored rigs."""
    if not 1 <= len(obj.data.vertices) <= 100000 or not 1 <= len(obj.data.edges) <= 300000:
        raise ValueError('Deformation rest geometry exceeds bounded diagnostic size')
    if rig.data.pose_position != 'REST' or not np.allclose(np.asarray(obj.matrix_world), np.eye(4), atol=1e-7):
        raise ValueError('Deformation baseline requires applied world transforms and an actual rest pose')
    if rig_route not in ('authored', 'source_skin') or not 1 <= len(rig.data.bones) <= 64:
        raise ValueError('Deformation baseline requires a bounded actual rig')
    rest = np.empty((len(obj.data.vertices), 3), dtype='<f8')
    edges = np.empty((len(obj.data.edges), 2), dtype='<i4')
    obj.data.vertices.foreach_get('co', rest.ravel())
    obj.data.edges.foreach_get('vertices', edges.ravel())
    if not np.isfinite(rest).all() or edges.min() < 0 or edges.max() >= len(rest):
        raise ValueError('Invalid deformation rest geometry')
    low, high = rest.min(axis=0), rest.max(axis=0)
    span = high-low
    scale = float(span.max())
    height = float(span[2]) if span[2] > scale*1e-6 else scale
    if not math.isfinite(height) or height <= 0:
        raise ValueError('Deformation baseline has no measurable height')
    return rest, edges, {'schema_version': 1, 'status': 'measured', 'rig_route': rig_route,
        'coordinate_space': 'mesh_local_after_world_transform_applied', 'units': 'meters',
        'rest_sha256': hashlib.sha256(rest.tobytes()).hexdigest(),
        'edges_sha256': hashlib.sha256(edges.tobytes()).hexdigest(),
        'rest_array_encoding': 'little-endian float64 XYZ', 'edge_array_encoding': 'little-endian int32 pairs',
        'vertices': len(rest), 'edges': len(edges), 'bone_names': list(rig.data.bones.keys()),
        'rest_bounds': {'min': low.tolist(), 'max': high.tolist(), 'height': height,
                        'height_basis': 'z_span' if span[2] > scale*1e-6 else 'maximum_span'},
        'pose_samples_measured': 0, 'unchanged_topology_required': True}


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
    transform_error = float(np.max(np.abs(after_apply-points)))
    if transform_error > 1e-6:
        raise ValueError("Applying emitter scale changed the world rest geometry")
    low, high = points.min(axis=0), points.max(axis=0)
    part_rigs = None
    if cfg.get('part_rigs'):
        from blender_part_rigs import apply_part_rigs
        part_rigs = apply_part_rigs(obj, rig, points, low, high, cfg['part_rigs'], cfg['authored_rig'])
    normal_continuity = None
    if cfg.get('normal_continuity'):
        from blender_normal_continuity import apply_normal_continuity
        normal_continuity = apply_normal_continuity(obj, cfg['normal_continuity'])
    base = obj.data.materials[0]
    color_node = shared_color_texture(obj.data.materials)
    if cfg.get('portable_fur', {}).get('settings', {}).get('sample_root_color'):
        if color_node.image.colorspace_settings.name != 'sRGB':
            raise ValueError('Root-colored fur requires an explicit sRGB source texture')
    colors = texture_colors(obj, color_node.image)
    mask = ~frame_mask(points, low, high-low, cfg.get('groom_exclusions', []))
    mask &= colors.max(axis=1)-colors.min(axis=1) < cfg.get('fur_max_chroma', 1)
    excluded_part_vertices = set()
    if cfg.get('groom_excluded_parts'):
        materials = {name for part in cfg['part_rigs'] if part['name'] in cfg['groom_excluded_parts']
                     for name in part['materials']}
        for polygon in obj.data.polygons:
            if obj.data.materials[polygon.material_index].name in materials:
                excluded_part_vertices.update(polygon.vertices)
        # A moving accessory mask must not leave a permanent bald volume on its former support.
        mask[list(excluded_part_vertices)] = False
    smoothing = None
    if cfg.get('smoothing'):
        smoothing = smooth_surface(obj, mask, report_dir=out, **cfg['smoothing'])
        points = np.array([tuple(obj.matrix_world @ v.co) for v in obj.data.vertices])
    if cfg.get('surface_smoothing'):
        from surface_smoothing import refine_positions
        original = np.array([tuple(v.co) for v in obj.data.vertices])
        edges = np.array([tuple(e.vertices) for e in obj.data.edges])
        refined, smoothing = refine_positions(original, edges, cfg['surface_smoothing'])
        smoothing['surface_integrity'] = set_surface_positions(obj, original, refined, out)
        points = np.array([tuple(obj.matrix_world @ v.co) for v in obj.data.vertices])
    for material in obj.data.materials:
        for node in material.node_tree.nodes:
            if node.type == "NORMAL_MAP":
                node.inputs["Strength"].default_value = cfg["normal_strength"]
    surface_profiles = None
    if cfg.get('surface_profiles'):
        from blender_surface_profiles import apply_surface_profiles
        surface_profiles = apply_surface_profiles(obj, colors, points, low, high-low, cfg['surface_profiles'])
    color_corrections = None
    if cfg.get('color_corrections'):
        from blender_color_corrections import apply_color_corrections
        color_corrections, corrected_vertices = apply_color_corrections(
            obj, colors, points, low, high-low, cfg['color_corrections'])
        mask[corrected_vertices] = False
    report = {"schema_version": 1, "source_sha256": cfg["source_sha256"], "blender_version": bpy.app.version_string,
              "normal_strength": cfg["normal_strength"], "groom": None, "attachment": None,
              "smoothing": smoothing, 'authored_rig': authored_rig, 'surface_profiles': surface_profiles,
              'color_corrections': color_corrections, 'part_rigs': part_rigs, 'normal_continuity': normal_continuity,
              'groom_excluded_parts': {'names': cfg.get('groom_excluded_parts', []),
                                       'vertices': len(excluded_part_vertices)},
              "emitter_scale_applied": True, "world_rest_position_max_error": transform_error,
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
    cards = []
    diagnostic_rest, diagnostic_edges = None, None
    if cfg.get('sample_states'):
        from deformation_diagnostics import diagnose_stretched_edges, retain_stretched_edges
        diagnostic_rest, diagnostic_edges, report['deformation_measurement'] = deformation_baseline(
            obj, rig, 'authored' if cfg.get('authored_rig') else 'source_skin')
    if cfg.get('fur_undercoat'):
        from blender_fur_undercoat import apply_undercoat
        from groom_flow import prepare_flow
        local = np.asarray([tuple(vertex.co) for vertex in obj.data.vertices])
        prepared = prepare_flow(local.min(axis=0), np.ptp(local, axis=0),
                                cfg['groom']['direction'], cfg['groom']['comb'], cfg['groom_flow'])
        report['fur_undercoat'] = apply_undercoat(obj, mask, prepared, cfg['fur_undercoat'], out,
            export_vertex_budget=cfg.get('undercoat_export_budget'))
    tuft = cfg.get('portable_fur', {}).get('settings', {}).get('profile') == 'tuft-v1'
    if tuft:
        cards, report['portable_fur'] = portable_coat(obj, rig, mask, colors, cfg['portable_fur'], out,
            tuft_image=color_node.image, native_groom=cfg['groom'], groom_flow=cfg['groom_flow'])
        report['portable_fur_export_budget'] = json.loads((out/'tuft-export-budget.json').read_text())
        report['portable_fur_build_order'] = 'Tuft cards and borrowed root-color buffer completed before native grooming'
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
        if "direction" in cfg["groom"] or normal_continuity:
            from blender_comb_groom import comb_native_groom
            groom_report["comb"] = comb_native_groom(groom, cfg["groom"].get("direction", [0,0,1]),
                                                    cfg["groom"].get("comb", 0), cfg.get("groom_flow"))
        hair_material = groom.data.materials[-1]
        hair = next(n for n in hair_material.node_tree.nodes if n.type == "BSDF_HAIR_PRINCIPLED")
        texture = hair_material.node_tree.nodes.new("ShaderNodeTexImage")
        texture.image = color_node.image
        hair_material.node_tree.links.new(texture.outputs["Color"], hair.inputs["Color"])
        if cfg.get('groom_shader'):
            hair.inputs['Roughness'].default_value = cfg['groom_shader']['roughness']
            hair.inputs['Radial Roughness'].default_value = cfg['groom_shader']['radial_roughness']
            groom_report['material']['shader_roughness'] = cfg['groom_shader']['roughness']
            groom_report['material']['radial_roughness'] = cfg['groom_shader']['radial_roughness']
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
    if cfg.get('portable_fur') and not tuft:
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
    if cfg.get('camera_fit') == 'motion':
        motion_low, motion_high, bounds_report = motion_bounds(obj, rig, actions, bindings, owners, cfg['authored_actions'])
        # The base skin excludes strands/cards; add their configured extent before the framing margin.
        fur_margin = max(cfg.get('groom', {}).get('length', 0)*3,
                         cfg.get('portable_fur', {}).get('settings', {}).get('length', 0))
        motion_low = np.minimum(motion_low, low)-fur_margin
        motion_high = np.maximum(motion_high, high)+fur_margin
        fit = fit_motion_camera(motion_low, motion_high, np.asarray(camera.rotation_euler.to_matrix()),
                                camera.location, camera_data.clip_start, camera_data.clip_end)
        camera.location, camera_data.ortho_scale, camera_data.clip_end = fit['location'], fit['orthographic_scale'], fit['clip_end']
        if camera_data.clip_end < fit['sampled_depth_max_m']:
            raise ValueError('Motion depth exceeds the native camera clip limit')
        report['camera'].update(**fit, fit='motion',
                                fur_margin_m=fur_margin, motion_envelope=bounds_report)
    scene.render.filepath = str(out/"portrait.png")
    bpy.ops.render.render(write_still=True)
    if cfg.get('review_views'):
        (out/'views').mkdir()
        saved_location, saved_rotation = camera.location.copy(), camera.rotation_euler.copy()
        saved_scale = camera_data.ortho_scale
        camera_data.ortho_scale = extent*1.25
        report['review_views'] = []
        for view, offset in (('front', (0, -3, .04)), ('side', (3, 0, .04)),
                             ('back', (0, 3, .04)), ('three_quarter', (1.5, -3, .12))):
            camera.location = center+Vector(offset)*extent
            camera.rotation_euler = (center-camera.location).to_track_quat('-Z', 'Y').to_euler()
            basis = camera.rotation_euler.to_matrix()
            projected = points@np.asarray(basis)
            camera_data.ortho_scale = float(np.ptp(projected[:, :2], axis=0).max())*1.18
            camera_data.ortho_scale += cfg.get('groom', {}).get('length', 0)*6
            for i in (0, 1):
                midpoint = float((projected[:, i].min()+projected[:, i].max())*.5)
                camera.location += basis.col[i]*(midpoint-camera.location.dot(basis.col[i]))
            path = out/'views'/(view+'.png')
            scene.render.filepath = str(path)
            bpy.ops.render.render(write_still=True)
            report['review_views'].append({'view': view, 'path': str(path.relative_to(out)),
                                          'sha256': hash_file(path), 'resolution': cfg['resolution']})
        camera.location, camera.rotation_euler, camera_data.ortho_scale = saved_location, saved_rotation, saved_scale
    if cfg.get('sample_states'):
        scene.render.resolution_x = scene.render.resolution_y = cfg.get('pose_resolution', 384)
        scene.cycles.samples = cfg['samples'] if cfg.get('review_views') else 8
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
            outliers = []
            for index, frame in enumerate(sampling['frames']):
                set_frame(frame)
                if diagnostic_rest is not None:
                    evaluated = obj.evaluated_get(bpy.context.evaluated_depsgraph_get())
                    mesh = evaluated.to_mesh()
                    try:
                        if len(mesh.vertices) != len(diagnostic_rest):
                            raise ValueError('Deformation diagnostics require unchanged body topology')
                        if len(mesh.edges) != len(diagnostic_edges):
                            raise ValueError('Deformation diagnostics require unchanged edge topology')
                        evaluated_edges = np.empty_like(diagnostic_edges)
                        mesh.edges.foreach_get('vertices', evaluated_edges.ravel())
                        if not np.array_equal(evaluated_edges, diagnostic_edges):
                            raise ValueError('Deformation diagnostics require identical edge endpoint ordering')
                        posed = np.empty_like(diagnostic_rest)
                        mesh.vertices.foreach_get('co', posed.ravel())
                        for row in diagnose_stretched_edges(diagnostic_rest, posed, diagnostic_edges):
                            row['second'] = float(frame/(scene.render.fps/scene.render.fps_base))
                            row['endpoint_influences'] = [
                                {obj.vertex_groups[g.group].name: float(g.weight)
                                 for g in obj.data.vertices[v].groups if g.weight > 1e-6}
                                for v in row['endpoint_indices']]
                            outliers.append(row)
                        report['deformation_measurement']['pose_samples_measured'] += 1
                    finally:
                        evaluated.to_mesh_clear()
                scene.render.filepath = str(folder/f'{index:04d}.png')
                bpy.ops.render.render(write_still=True)
            worst = retain_stretched_edges(outliers, report['deformation_measurement']['rest_bounds']['height'])
            report['states'].append({'id': action.name, **sampling,
                'duration_seconds': (end-start)/(scene.render.fps/scene.render.fps_base), 'loop_verified': False,
                'deformation_outliers': worst,
                'deformation_samples_measured': len(sampling['frames']),
                'deformation_rest_sha256': report['deformation_measurement']['rest_sha256'],
                'deformation_diagnostic_scope': 'Sampled edge stretch with original normalized body coordinates and weights; diagnostic only, not visual acceptance or physics'})
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
