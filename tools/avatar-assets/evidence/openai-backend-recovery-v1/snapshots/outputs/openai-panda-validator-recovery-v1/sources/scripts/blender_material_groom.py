"""Material-owned native hair; source geometry and portable exports stay untouched."""

import hashlib
import json
import math
from pathlib import Path
import struct
import time

import numpy as np


POLICY = 'material-owned-native-groom-v1'
MAX_VERTICES, MAX_FACES, MAX_PARTS = 100000, 100000, 32
MAX_PARENTS, MAX_FIBERS, MAX_KEYS = 10000, 240000, 150000
MAX_COMB_SECONDS = 20
NORMAL_TOLERANCE = 1e-4
POINT_NORMAL, FACE_REPAIR = 'AuthoredSurfaceNormal', 'AuthoredNormalRepair'


def _require(value, message):
    if not value:
        raise ValueError('Material groom: '+message)


def _hash(values):
    return hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest()


def _write(path, value):
    encoded = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+'\n').encode()
    _require(len(encoded) <= 1024**2, 'report exceeds1MiB')
    with path.open('xb') as stream:
        stream.write(encoded)


def _rgb(value, label):
    _require(isinstance(value, (list, tuple)) and len(value) == 3 and all(
        type(v) in (int, float) and math.isfinite(v) and 0 <= v <= 1 for v in value),
        'invalid '+label)
    return tuple(float(v) for v in value)


def _capture(obj, rig):
    mesh = obj.data
    _require(obj.type == 'MESH' and obj.mode == 'OBJECT'
             and 1 <= len(mesh.vertices) <= MAX_VERTICES
             and 1 <= len(mesh.polygons) <= MAX_FACES
             and len(mesh.loops) == 3*len(mesh.polygons), 'require a bounded triangular object-mode source')
    _require(not obj.particle_systems and len(obj.modifiers) == 1,
             'source must have only its skin modifier and no particle system')
    modifier = obj.modifiers[0]
    _require(modifier.type == 'ARMATURE' and modifier.object == rig
             and modifier.show_viewport and modifier.show_render and modifier.use_vertex_groups
             and not modifier.use_bone_envelopes and not modifier.use_deform_preserve_volume
             and not modifier.vertex_group, 'require the existing linear vertex-group skin without extra masks')
    matrix = np.asarray(obj.matrix_world, dtype='<f8')
    _require(matrix.shape == (4, 4) and np.isfinite(matrix).all()
             and np.max(abs(matrix-np.eye(4))) <= 1e-7, 'source frame must be applied meter-space identity')
    positions = np.empty((len(mesh.vertices), 3), dtype='<f4')
    mesh.vertices.foreach_get('co', positions.ravel())
    _require(np.isfinite(positions).all(), 'nonfinite source positions')
    faces = np.asarray([tuple(p.vertices) for p in mesh.polygons], dtype='<i4')
    loops = np.asarray([tuple(p.loop_indices) for p in mesh.polygons], dtype='<i4')
    _require(faces.shape == loops.shape == (len(mesh.polygons), 3)
             and np.array_equal(loops.ravel(), np.arange(len(mesh.loops)))
             and faces.min() >= 0 and faces.max() < len(positions), 'invalid source triangle/corner ordering')
    normals = np.asarray([tuple(n.vector) for n in mesh.corner_normals], dtype='<f4')
    _require(normals.shape == (len(mesh.loops), 3) and np.isfinite(normals).all()
             and np.max(abs(np.linalg.norm(normals.astype(float), axis=1)-1)) <= .001,
             'source corner normals must be finite unit vectors')
    slots = np.asarray([p.material_index for p in mesh.polygons], dtype='<i4')
    smooth = np.asarray([p.use_smooth for p in mesh.polygons], dtype=bool)
    _require(1 <= len(mesh.materials) <= MAX_PARTS and all(m is not None for m in mesh.materials) and slots.min() >= 0
             and slots.max() < len(mesh.materials), 'invalid bounded source material slots')
    _require(len(mesh.uv_layers) <= 8, 'source exceeds eight UV layers')
    uvs = []
    for layer in mesh.uv_layers:
        values = np.empty((len(mesh.loops), 2), dtype='<f4')
        layer.data.foreach_get('uv', values.ravel())
        _require(np.isfinite(values).all() and np.max(abs(values)) <= 1e6, 'invalid bounded source UVs')
        uvs.append({'name': layer.name, 'active_render': bool(layer.active_render), 'values': values})
    if uvs:
        _require(0 <= mesh.uv_layers.active_index < len(uvs)
                 and sum(row['active_render'] for row in uvs) == 1, 'ambiguous source UV defaults')
    groups = [group.name for group in obj.vertex_groups]
    bones = {bone.name for bone in rig.data.bones}
    _require(1 <= len(groups) <= 64 and len(set(groups)) == len(groups)
             and 1 <= len(bones) <= 64, 'source skin group/bone count exceeds bounds')
    weights, weight_hash, entries = [], hashlib.sha256(), 0
    for vertex in mesh.vertices:
        row = [(int(g.group), float(g.weight)) for g in vertex.groups]
        entries += len(row)
        _require(len(row) <= 64 and entries <= 400000 and all(
            0 <= index < len(groups) and math.isfinite(weight) and 0 <= weight <= 1 for index, weight in row),
            'source skin entries exceed bounds')
        positive = [(index, weight) for index, weight in row if weight > 0]
        _require(1 <= len(positive) <= 4 and all(groups[index] in bones for index, _ in positive)
                 and abs(sum(weight for _, weight in positive)-1) <= .001,
                 'every original vertex needs1–4 normalized bone influences')
        weight_hash.update(struct.pack('<II', vertex.index, len(row)))
        for index, weight in row:
            weight_hash.update(struct.pack('<If', index, weight))
        weights.append(row)
    authored = {}
    point, repair = mesh.attributes.get(POINT_NORMAL), mesh.attributes.get(FACE_REPAIR)
    _require((point is None) == (repair is None), 'incomplete authored normal attributes')
    if point is not None:
        _require(point.domain == 'POINT' and point.data_type == 'FLOAT_VECTOR'
                 and repair.domain == 'FACE' and repair.data_type == 'BOOLEAN', 'invalid authored normal attribute types')
        values = np.empty((len(positions), 3), dtype='<f4'); point.data.foreach_get('vector', values.ravel())
        flags = np.empty(len(faces), dtype=bool); repair.data.foreach_get('value', flags)
        _require(np.isfinite(values).all(), 'nonfinite authored point normals')
        authored = {POINT_NORMAL: values, FACE_REPAIR: flags}
    signature = {'positions': _hash(positions), 'faces': _hash(faces), 'loops': _hash(loops),
        'corner_normals': _hash(normals), 'material_slots': _hash(slots), 'smooth': _hash(smooth),
        'weights': weight_hash.hexdigest(), 'group_names': groups, 'world_matrix': _hash(matrix),
        'material_names': [m.name if m else None for m in mesh.materials],
        'uvs': [{'name': row['name'], 'sha256': _hash(row['values']), 'active_render': row['active_render']} for row in uvs],
        'active_uv_index': mesh.uv_layers.active_index if uvs else None,
        'authored_normals': {name: _hash(values) for name, values in authored.items()}}
    return {'positions': positions, 'faces': faces, 'loops': loops, 'normals': normals,
            'slots': slots, 'smooth': smooth, 'uvs': uvs, 'groups': groups, 'weights': weights,
            'authored': authored, 'signature': signature,
            'low': positions.min(axis=0).astype(float), 'span': np.ptp(positions.astype(float), axis=0)}


def _source_descriptor(obj, slot, source):
    fields = {'kind', 'image', 'factor', 'linear_rgb', 'material_name', 'uv_layer'}
    _require(isinstance(source, dict) and fields <= set(source) <= fields | {'node'}, 'invalid source color descriptor')
    material = obj.data.materials[slot]
    _require(material is not None and material.name == source['material_name'] and material.use_nodes
             and 1 <= len(material.node_tree.nodes) <= 256, 'source descriptor differs from its actual material')
    shaders = [node for node in material.node_tree.nodes if node.type == 'BSDF_PRINCIPLED']
    _require(len(shaders) == 1, 'require one actual Principled source shader')
    shader = shaders[0]
    outputs = [node for node in material.node_tree.nodes if node.type == 'OUTPUT_MATERIAL' and node.is_active_output]
    _require(len(outputs) == 1 and len(outputs[0].inputs['Surface'].links) == 1
             and outputs[0].inputs['Surface'].links[0].from_node == shader, 'source must directly output its Principled surface')
    _require(not shader.inputs['Alpha'].links and shader.inputs['Alpha'].default_value == 1,
             'emitted faces must be explicitly opaque')
    factor = _rgb(source['factor'], 'linear factor')
    _require(factor == (1., 1., 1.), 'initial material groom supports direct colors, not unverified multiplier graphs')
    color = shader.inputs['Base Color']
    if source['kind'] == 'constant':
        rgb = _rgb(source['linear_rgb'], 'constant linear RGB')
        _require(source['image'] is None and source['uv_layer'] is None and not color.links
                 and tuple(color.default_value[:3]) == rgb, 'constant descriptor differs from actual source color')
        return {'kind': 'constant', 'material_name': material.name, 'factor': list(factor),
                'linear_rgb': list(rgb), 'uv_layer': None}
    _require(source['kind'] == 'image' and source['linear_rgb'] is None and len(color.links) == 1,
             'require a direct image or exact constant source')
    node = color.links[0].from_node
    _require(node.type == 'TEX_IMAGE' and node.image is not None and node.image == source['image']
             and color.links[0].from_socket.name == 'Color' and node.interpolation == 'Linear'
             and node.extension == 'REPEAT' and node.projection == 'FLAT' and not node.inputs['Vector'].links,
             'image descriptor must match direct Linear/REPEAT/FLAT active-UV color')
    uv = obj.data.uv_layers.active
    render = [layer for layer in obj.data.uv_layers if layer.active_render]
    _require(uv is not None and len(render) == 1 and uv.name == render[0].name == source['uv_layer'],
             'sampled and native-render UV layers must agree')
    image = node.image
    _require(image.colorspace_settings.name == 'sRGB' and 0 < image.size[0] <= 4096
             and 0 < image.size[1] <= 4096 and image.size[0]*image.size[1] <= 16777216,
             'source image exceeds the explicit sRGB/16MP profile')
    return {'kind': 'image', 'material_name': material.name, 'image': image.name,
            'dimensions': list(image.size), 'factor': list(factor), 'uv_layer': uv.name,
            'interpolation': node.interpolation, 'extension': node.extension, 'projection': node.projection}


def _subset(obj, rig, captured, face_ids, name):
    import bpy

    mesh = bpy.data.meshes.new(name+'_mesh')
    temporary = None
    try:
        source_loops = captured['loops'][face_ids].ravel()
        mesh.vertices.add(len(captured['positions']))
        mesh.vertices.foreach_set('co', captured['positions'].ravel())
        mesh.loops.add(len(source_loops))
        mesh.loops.foreach_set('vertex_index', captured['faces'][face_ids].ravel())
        mesh.polygons.add(len(face_ids))
        mesh.polygons.foreach_set('loop_start', np.arange(0, len(source_loops), 3, dtype=np.int32))
        mesh.polygons.foreach_set('loop_total', np.full(len(face_ids), 3, dtype=np.int32))
        for material in obj.data.materials:
            mesh.materials.append(material)
        mesh.polygons.foreach_set('material_index', captured['slots'][face_ids])
        mesh.polygons.foreach_set('use_smooth', captured['smooth'][face_ids])
        mesh.update(calc_edges=True)
        for row in captured['uvs']:
            layer = mesh.uv_layers.new(name=row['name'])
            _require(layer.name == row['name'], 'copied UV layer was renamed')
            layer.data.foreach_set('uv', row['values'][source_loops].ravel())
        if captured['uvs']:
            mesh.uv_layers.active_index = captured['signature']['active_uv_index']
            for index, row in enumerate(captured['uvs']):
                mesh.uv_layers[index].active_render = row['active_render']
        if captured['authored']:
            attribute = mesh.attributes.new(name=POINT_NORMAL, type='FLOAT_VECTOR', domain='POINT')
            attribute.data.foreach_set('vector', captured['authored'][POINT_NORMAL].ravel())
            attribute = mesh.attributes.new(name=FACE_REPAIR, type='BOOLEAN', domain='FACE')
            attribute.data.foreach_set('value', captured['authored'][FACE_REPAIR][face_ids])
        # FLOAT_VECTOR avoids the legacy short2 custom-normal storage quantization.
        custom = mesh.attributes.new(name='custom_normal', type='FLOAT_VECTOR', domain='CORNER')
        _require(custom.name == 'custom_normal', 'custom-normal attribute name collision')
        custom.data.foreach_set('vector', captured['normals'][source_loops].ravel())
        mesh.update()
        actual = np.asarray([tuple(n.vector) for n in mesh.corner_normals], dtype='<f4')
        error = float(np.max(abs(actual.astype(float)-captured['normals'][source_loops])))
        _require(np.isfinite(actual).all() and error <= NORMAL_TOLERANCE, 'partition changed source corner normals')
        temporary = bpy.data.objects.new(name, mesh)
        bpy.context.scene.collection.objects.link(temporary)
        temporary.matrix_world = obj.matrix_world.copy()
        temporary.hide_render = True
        for name in captured['groups']:
            group = temporary.vertex_groups.new(name=name)
            _require(group.name == name, 'copied skin group was renamed')
        for vertex, row in enumerate(captured['weights']):
            for group, weight in row:
                temporary.vertex_groups[group].add([vertex], weight, 'REPLACE')
        modifier = temporary.modifiers.new('MaterialFurSkin', 'ARMATURE')
        modifier.object = rig
        modifier.use_vertex_groups, modifier.use_bone_envelopes = True, False
        modifier.use_deform_preserve_volume = False
        return temporary, error
    except BaseException:
        if temporary is not None:
            bpy.data.objects.remove(temporary, do_unlink=True)
        if mesh.users == 0:
            bpy.data.meshes.remove(mesh)
        raise


def _hair_color(groom, source, description, shader_settings):
    material = groom.data.materials[-1]
    hair = next(node for node in material.node_tree.nodes if node.type == 'BSDF_HAIR_PRINCIPLED')
    if description['kind'] == 'image':
        texture = material.node_tree.nodes.new('ShaderNodeTexImage')
        texture.image = source['image']
        texture.interpolation, texture.extension, texture.projection = 'Linear', 'REPEAT', 'FLAT'
        material.node_tree.links.new(texture.outputs['Color'], hair.inputs['Color'])
    else:
        hair.inputs['Color'].default_value = (*description['linear_rgb'], 1.)
    if shader_settings:
        hair.inputs['Roughness'].default_value = shader_settings['roughness']
        hair.inputs['Radial Roughness'].default_value = shader_settings['radial_roughness']
    return {'material': material.name, 'source': description,
            'roughness': float(hair.inputs['Roughness'].default_value),
            'radial_roughness': float(hair.inputs['Radial Roughness'].default_value),
            'cpu_pixel_decode': False, 'native_root_texture_sampling_verified': False}


def _verify_groom_layout(groom, captured, part):
    mesh, ids, loops = groom.data, part['face_ids'], part['loop_ids']
    _require(len(mesh.vertices) == len(captured['positions']) and len(mesh.polygons) == len(ids)
             and len(mesh.loops) == len(loops), 'groom changed copied topology counts')
    _require(np.array_equal(np.asarray([tuple(p.vertices) for p in mesh.polygons]), captured['faces'][ids])
             and np.array_equal(np.asarray([p.material_index for p in mesh.polygons]), captured['slots'][ids])
             and np.array_equal(np.asarray([p.use_smooth for p in mesh.polygons]), captured['smooth'][ids]),
             'groom changed copied faces, material ownership or smooth flags')
    _require(len(mesh.uv_layers) == len(captured['uvs']), 'groom changed copied UV layer count')
    if captured['uvs']:
        _require(mesh.uv_layers.active_index == captured['signature']['active_uv_index'], 'groom changed active UV')
    for layer, row in zip(mesh.uv_layers, captured['uvs'], strict=True):
        values = np.empty((len(loops), 2), dtype='<f4')
        layer.data.foreach_get('uv', values.ravel())
        _require(layer.name == row['name'] and layer.active_render == row['active_render']
                 and np.array_equal(values, row['values'][loops]), 'groom changed copied source UVs')
    if captured['authored']:
        point, face = mesh.attributes.get(POINT_NORMAL), mesh.attributes.get(FACE_REPAIR)
        _require(point is not None and face is not None and point.domain == 'POINT'
                 and point.data_type == 'FLOAT_VECTOR' and face.domain == 'FACE'
                 and face.data_type == 'BOOLEAN', 'groom lost authored normal attributes')
        values = np.empty_like(captured['authored'][POINT_NORMAL]); point.data.foreach_get('vector', values.ravel())
        flags = np.empty(len(ids), dtype=bool); face.data.foreach_get('value', flags)
        _require(np.array_equal(values, captured['authored'][POINT_NORMAL])
                 and np.array_equal(flags, captured['authored'][FACE_REPAIR][ids]), 'groom changed authored normal attributes')
    actual = np.asarray([tuple(n.vector) for n in mesh.corner_normals], dtype='<f4')
    error = float(np.max(abs(actual.astype(float)-captured['normals'][loops])))
    _require(np.isfinite(actual).all() and error <= NORMAL_TOLERANCE, 'groom changed source corner normals')
    return error


def add_material_grooms(obj, rig, cfg, mask, sources, out):
    """Return native-only grooms and custody evidence; all global counts stay fixed."""
    import bpy
    import blender_comb_groom as comb_module
    from blender_native_groom import _skin_signature, add_native_groom, validate_groom_settings
    from material_fur import allocate_counts

    started = time.monotonic()
    requested = cfg.get('material_fur')
    _require(isinstance(requested, dict) and set(requested) == {'version', 'materials'}
             and type(requested['version']) is int and requested['version'] == 1, 'invalid opt-in material_fur schema')
    names = requested['materials']
    _require(isinstance(names, list) and 1 <= len(names) <= MAX_PARTS and all(
        isinstance(name, str) and name and len(name) <= 256 for name in names)
        and len(set(names)) == len(names), 'require unique bounded material names')
    _require(not any(cfg.get(key) for key in ('surface_profiles', 'color_corrections', 'fur_undercoat',
        'normal_continuity', 'attachment', 'weight_constraints')), 'material or skin mutations require an explicit compatible profile')
    _require(rig.type == 'ARMATURE' and rig.data.pose_position == 'REST', 'require the existing rest-pose rig')
    settings = validate_groom_settings(cfg['groom'])
    settings = {key: value for key, value in settings.items() if key not in ('direction', 'comb')}
    parents = settings['parents']
    render_children = cfg.get('native_render_children', settings['children'])
    _require(type(render_children) is int and 0 <= render_children <= 90
             and parents <= MAX_PARENTS and parents*(render_children+1) <= MAX_FIBERS,
             'aggregate native parent/render-fiber limit exceeded')
    clump = cfg.get('fur_clump', .06)
    _require(type(clump) in (int, float) and math.isfinite(clump) and 0 <= clump <= 1, 'invalid clump setting')
    shader_settings = cfg.get('groom_shader')
    if shader_settings:
        _require(isinstance(shader_settings, dict) and set(shader_settings) == {'roughness', 'radial_roughness'}
                 and all(type(v) in (int, float) and math.isfinite(v) and .04 <= v <= 1 for v in shader_settings.values()),
                 'invalid native shader settings')
    captured = _capture(obj, rig)
    skin_before = _skin_signature(obj, rig)
    _require(isinstance(mask, np.ndarray) and mask.dtype == bool and mask.shape == (len(captured['positions']),),
             'require a measured boolean vertex mask')
    _require(isinstance(sources, dict) and 1 <= len(sources) <= MAX_PARTS and all(
        type(slot) is int and 0 <= slot < len(obj.data.materials) for slot in sources), 'invalid material source table')
    material_names = captured['signature']['material_names']
    _require(all(material_names.count(name) == 1 for name in names), 'selected material is missing or ambiguous')
    slots = sorted(material_names.index(name) for name in names)
    _require(set(slots) <= set(sources), 'selected material lacks a source color descriptor')
    descriptions = {slot: _source_descriptor(obj, slot, sources[slot]) for slot in slots}
    images = {sources[slot]['image'].as_pointer(): sources[slot]['image'] for slot in slots
              if sources[slot]['kind'] == 'image'}
    _require(sum(image.size[0]*image.size[1] for image in images.values()) <= 16777216,
             'selected source textures exceed the16MP aggregate limit')
    eligible = mask[captured['faces']].all(axis=1)
    points = captured['positions'].astype(float)
    edges0 = points[captured['faces'][:, 1]]-points[captured['faces'][:, 0]]
    edges1 = points[captured['faces'][:, 2]]-points[captured['faces'][:, 0]]
    areas = np.linalg.norm(np.cross(edges0, edges1), axis=1)*.5
    _require(np.isfinite(areas).all(), 'nonfinite source face areas')
    plans = []
    for slot in slots:
        selected = np.flatnonzero(eligible & (captured['slots'] == slot)).astype('<i4')
        _require(len(selected) > 0 and np.all(areas[selected] > 0), 'selected material has no nondegenerate eligible faces')
        if captured['authored']:
            _require(captured['authored'][FACE_REPAIR][selected].any(),
                     'existing comb helper cannot process a partition with an empty inherited normal-repair mask')
        plans.append({'slot': slot, 'face_ids': selected, 'area_m2': float(areas[selected].sum()),
                      'loop_ids': captured['loops'][selected].ravel().astype('<i4')})
    counts = allocate_counts([part['area_m2'] for part in plans], parents, minimum=16, max_total=MAX_PARENTS)
    _require(sum(counts) == parents and all(count >= 16 for count in counts), 'global allocation changed requested parent count')
    folder = Path(out)/'material-groom'
    _require(Path(out).is_dir() and not Path(out).is_symlink()
             and not folder.exists() and not folder.is_symlink(), 'require fresh scoped material-groom output')
    folder.mkdir(mode=0o700)
    provenance = {'source_vertex_ids': np.arange(len(points), dtype='<i4'),
                  'source_corner_normals': captured['normals']}
    for index, part in enumerate(plans):
        provenance[f'part_{index:02d}_face_ids'] = part['face_ids']
        provenance[f'part_{index:02d}_loop_ids'] = part['loop_ids']
    _require(sum(value.nbytes for value in provenance.values()) < 8*1024**2, 'correspondence arrays exceed8MiB')
    path = folder/'correspondence.npz'
    with path.open('xb') as stream:
        np.savez(stream, **provenance)
    _require(path.stat().st_size <= 8*1024**2, 'correspondence file exceeds8MiB')
    report = {'schema_version': 1, 'kind': 'material_owned_native_grooms', 'policy': POLICY,
        'status': 'preparing', 'native_only': True, 'gltf_export_supported': False, 'quality_approved': False,
        'source_object': obj.name, 'armature': rig.name, 'body_before': captured['signature'],
        'requested_materials': names, 'requested_parents': parents, 'allocated_parents': counts,
        'render_children': render_children, 'render_fiber_upper_bound': parents*(render_children+1),
        'whole_emitter_flow_bounds': {'min': captured['low'].tolist(), 'span': captured['span'].tolist()},
        'source_vertices_per_emitter': len(points), 'retained_native_vertex_instances': len(points)*len(plans),
        'selection': 'Exact owned material AND all three source vertices pass the final mask; no density interpolation across excluded faces.',
        'correspondence': {'path': str(path.relative_to(out)), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
            'vertex_mapping': 'Identity IDs0..N-1 retained in every emitter, including unused vertices'},
        'limits': {'parts': MAX_PARTS, 'source_vertices': MAX_VERTICES, 'source_faces': MAX_FACES,
            'parents': MAX_PARENTS, 'fibers': MAX_FIBERS, 'stored_keys': MAX_KEYS,
            'aggregate_comb_seconds': MAX_COMB_SECONDS, 'normal_max_component_error': NORMAL_TOLERANCE},
        'parts': []}
    grooms, comb_elapsed, stored_keys, actual_parents = [], 0., 0, 0
    try:
        for index, (part, count) in enumerate(zip(plans, counts)):
            temporary, normal_error = _subset(obj, rig, captured, part['face_ids'], f'MaterialFur_{index:02d}')
            mesh = temporary.data
            try:
                _require(_skin_signature(temporary, rig) == skin_before, 'partition changed original vertex IDs, coordinates or skin')
                subset = _capture(temporary, rig)
                _require(np.array_equal(subset['positions'], captured['positions'])
                         and np.array_equal(subset['low'], captured['low']) and np.array_equal(subset['span'], captured['span']),
                         'partition changed the shared whole-emitter flow bounds')
                _require(np.array_equal(subset['faces'], captured['faces'][part['face_ids']])
                         and np.array_equal(subset['slots'], captured['slots'][part['face_ids']])
                         and np.array_equal(subset['smooth'], captured['smooth'][part['face_ids']]), 'partition changed source face identity')
                for original_uv, copied_uv in zip(captured['uvs'], subset['uvs'], strict=True):
                    _require(original_uv['name'] == copied_uv['name'] and original_uv['active_render'] == copied_uv['active_render']
                             and np.array_equal(original_uv['values'][part['loop_ids']], copied_uv['values']), 'partition changed source UVs')
                if captured['authored']:
                    _require(np.array_equal(subset['authored'][POINT_NORMAL], captured['authored'][POINT_NORMAL])
                             and np.array_equal(subset['authored'][FACE_REPAIR], captured['authored'][FACE_REPAIR][part['face_ids']]),
                             'partition changed authored normal attributes')
                groom, native = add_native_groom(temporary, rig, {**settings, 'parents': count}, obj.data.materials[part['slot']])
            finally:
                bpy.data.objects.remove(temporary, do_unlink=True)
                if mesh.users == 0:
                    bpy.data.meshes.remove(mesh)
            grooms.append(groom)
            native['temporary_partition_source'] = native['source_object']
            native['source_object'] = obj.name
            groom['native_groom_source'] = obj.name
            groom['native_groom_source_material'] = descriptions[part['slot']]['material_name']
            groom.hide_render = False
            _require(groom.name.startswith('NativeGroom_') and not groom.show_instancer_for_render
                     and not groom.show_instancer_for_viewport and groom.parent == rig, 'native groom ownership/visibility changed')
            system = groom.particle_systems[0]
            system.settings.clump_factor = clump
            system.settings.rendered_child_count = render_children
            color_report = _hair_color(groom, sources[part['slot']], descriptions[part['slot']], shader_settings)
            bpy.context.view_layer.update()
            remaining = MAX_COMB_SECONDS-comb_elapsed
            _require(remaining > 0, 'aggregate comb deadline exhausted')
            previous_limit = comb_module.MAX_SECONDS
            comb_started = time.monotonic()
            try:
                # This adapter owns the process; restore the helper limit even after rejection.
                comb_module.MAX_SECONDS = min(previous_limit, remaining)
                comb = comb_module.comb_native_groom(groom, cfg['groom'].get('direction', [0, 0, 1]),
                    cfg['groom'].get('comb', 0), cfg.get('groom_flow'))
            finally:
                comb_module.MAX_SECONDS = previous_limit
                comb_elapsed += time.monotonic()-comb_started
            stored_keys += comb['stored_keys']; actual_parents += comb['parents']
            _require(comb_elapsed <= MAX_COMB_SECONDS and stored_keys <= MAX_KEYS and comb['parents'] == count,
                     'aggregate measured comb time/key/parent limit exceeded')
            _require(_skin_signature(groom, rig) == skin_before, 'groom changed inherited source positions or skin')
            normal_error = max(normal_error, _verify_groom_layout(groom, captured, part))
            native['material'] = color_report
            native['settings']['parents'] = count
            native['render_fiber_upper_bound'] = count*(render_children+1)
            native['native_parameters']['render_children'] = render_children
            native['limitations'] = [line for line in native['limitations'] if not line.startswith('Material uses the source base-color')]
            report['parts'].append({'index': index, 'object': groom.name, 'source_material_slot': part['slot'],
                'source_material': descriptions[part['slot']]['material_name'], 'faces': len(part['face_ids']),
                'selected_area_m2': part['area_m2'], 'source_face_ids_sha256': _hash(part['face_ids']),
                'source_loop_ids_sha256': _hash(part['loop_ids']), 'all_source_vertices_retained': True,
                'source_corner_normal_max_component_error': normal_error,
                'whole_emitter_bounds_verified': True, 'parents': count, 'material': color_report,
                'native': native, 'comb': comb})
        after = _capture(obj, rig)['signature']
        _require(after == captured['signature'] and actual_parents == parents,
                 'original body changed or actual global parent count differs')
        report.update(status='native_grooms_pending_render_review', body_after=after, original_body_unchanged=True,
            actual_parents=actual_parents, aggregate_stored_keys=stored_keys, aggregate_comb_seconds=comb_elapsed,
            aggregate_comb_deadline_policy='Sequential calls use the remaining20s helper deadline; the module value is restored in finally.',
            elapsed_seconds=time.monotonic()-started,
            limitations=['Native-only hair; no portable-coat, physical, likeness or appearance approval.',
                'Material partitions keep source faces and all vertex IDs; they are not new source geometry.',
                'Configured interpolated children are an upper bound, not independently counted visible fibers.',
                'No CPU texture decode occurs here; renderer image residency still needs measured resource proof.'])
        _write(folder/'report.json', report)
        return grooms, report
    except BaseException as error:
        report.update(status='unsupported', error={'type': type(error).__name__, 'message': str(error)[:2048]},
                      aggregate_comb_seconds=comb_elapsed, aggregate_stored_keys=stored_keys)
        _write(folder/'rejection.json', report)
        raise
