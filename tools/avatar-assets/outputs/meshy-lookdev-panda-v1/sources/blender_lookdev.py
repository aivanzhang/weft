"""Trusted local visibility/shading probes; never execute scripts stored in a blend."""

import hashlib
import json
from pathlib import Path
import sys
import time

import bpy
import numpy as np


def require(value, message):
    if not value:
        raise ValueError(message)


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write_json(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')


def inspect_dependencies():
    require(not bpy.data.libraries, 'Linked libraries are unsupported')
    require(len(bpy.data.objects) <= 256 and len(bpy.data.materials) <= 128
            and len(bpy.data.images) <= 64, 'Scene exceeds diagnostic data limits')
    images, pixels = [], 0
    for image in bpy.data.images:
        if image.type in ('RENDER_RESULT', 'COMPOSITING'):
            continue
        require(image.source in ('FILE', 'GENERATED') and (image.packed_file or image.packed_files
                or image.source == 'GENERATED'), 'Unpacked/external image dependency is unsupported: '+image.name)
        width, height = image.size
        pixels += width*height
        require(0 < width*height <= 16777216 and pixels <= 67108864, 'Texture decoding exceeds bounds')
        images.append({'name': image.name, 'size': [width, height], 'packed': bool(image.packed_file or image.packed_files),
                       'source': image.source, 'colorspace': image.colorspace_settings.name})
    require(not bpy.data.movieclips and not bpy.data.sounds and not bpy.data.cache_files and not bpy.data.volumes,
            'External media/cache/volume dependencies are unsupported')
    trees = list(bpy.data.node_groups)
    ids = []
    for name in ('objects', 'meshes', 'materials', 'worlds', 'cameras', 'lights', 'scenes', 'armatures', 'particles', 'shape_keys'):
        ids.extend(getattr(bpy.data, name))
    trees.extend(item.node_tree for item in ids if getattr(item, 'node_tree', None))
    total_nodes = 0
    for tree in trees:
        total_nodes += len(tree.nodes)
        require(total_nodes <= 8192, 'Node-tree inspection limit exceeded')
        require(not any(node.bl_idname in ('ShaderNodeScript', 'CompositorNodeOutputFile') for node in tree.nodes),
                'Script nodes and external file-output nodes are unsupported')
    for item in [*ids, *trees]:
        animation = getattr(item, 'animation_data', None)
        require(not animation or not animation.drivers, 'Drivers are unsupported in lookdev diagnostics')
    for obj in bpy.data.objects:
        require(obj.type in ('MESH', 'ARMATURE', 'CAMERA', 'LIGHT', 'EMPTY', 'CURVES'),
                'Unsupported lookdev object type: '+obj.type)
        require(all(mod.type in ('ARMATURE', 'PARTICLE_SYSTEM') for mod in obj.modifiers),
                'Only armature and static particle-hair modifiers are supported')
        for particle in obj.particle_systems:
            require(particle.settings.type == 'HAIR' and not particle.use_hair_dynamics
                    and not particle.point_cache.use_disk_cache, 'Only non-dynamic, non-cached native hair is supported')
    return images


def material_facts(material):
    nodes = material.node_tree.nodes if material.use_nodes else []
    principled = []
    for node in nodes:
        if node.type == 'BSDF_PRINCIPLED':
            principled.append({name: {'value': float(node.inputs[name].default_value), 'linked': node.inputs[name].is_linked}
                               for name in ('Metallic', 'Roughness', 'Alpha')})
    return {'name': material.name, 'use_nodes': material.use_nodes, 'backface_culling': material.use_backface_culling,
            'surface_render_method': getattr(material, 'surface_render_method', None),
            'shaders': [n.bl_idname for n in nodes if n.type.startswith('BSDF')], 'principled': principled,
            'normal_map_strengths': [float(n.inputs['Strength'].default_value) for n in nodes if n.type == 'NORMAL_MAP'],
            'textures': [n.image.name for n in nodes if n.type == 'TEX_IMAGE' and n.image]}


def classify(obj, portable_materials):
    if obj.get('native_groom') or obj.type == 'CURVES' or any(p.settings.type == 'HAIR' for p in obj.particle_systems):
        return 'native_groom'
    if obj.type == 'MESH' and (obj.name.startswith('PortableCoat_')
            or any(m and m.name in portable_materials for m in obj.data.materials)):
        return 'portable_cards'
    return 'base'


def mesh_facts(obj, role):
    mesh = obj.data
    coordinates = np.empty((len(mesh.vertices), 3), dtype=np.float64)
    mesh.vertices.foreach_get('co', coordinates.ravel())
    matrix = np.asarray(obj.matrix_world)
    world = coordinates@matrix[:3, :3].T+matrix[:3, 3]
    indices = np.empty(len(mesh.loops), dtype=np.int32)
    mesh.loops.foreach_get('vertex_index', indices)
    geometry = hashlib.sha256(np.round(world, 6).astype('<f8').tobytes()+indices.astype('<i4').tobytes()).hexdigest()
    return {'name': obj.name, 'role': role, 'vertices': len(mesh.vertices), 'faces': len(mesh.polygons),
            'smooth_faces': sum(p.use_smooth for p in mesh.polygons), 'hide_render': obj.hide_render,
            'show_instancer_for_render': obj.show_instancer_for_render,
            'materials': [m.name if m else None for m in mesh.materials],
            'modifiers': [{'type': m.type, 'show_render': m.show_render} for m in obj.modifiers],
            'world_geometry_ordered_rounded_sha256': geometry,
            'world_bounds': {'min': world.min(axis=0).tolist(), 'max': world.max(axis=0).tolist()}}


def parent_key_hash():
    result = hashlib.sha256()
    keys = 0
    for obj in bpy.data.objects:
        for system in obj.particle_systems:
            for particle in system.particles:
                data = np.asarray([tuple(key.co) for key in particle.hair_keys], dtype='<f8')
                keys += len(data)
                require(keys <= 300000, 'Parent hair-key inspection bound exceeded')
                result.update(data.tobytes())
    return {'keys': keys, 'sha256': result.hexdigest()}


def main(config_path):
    config = json.loads(Path(config_path).read_text())
    source, out = Path(config['source_blend']), Path(config['out'])
    require(digest(source) == config['source_blend_sha256'], 'Blend differs from verified input')
    bpy.context.preferences.filepaths.use_scripts_auto_execute = False
    bpy.ops.wm.open_mainfile(filepath=str(source), load_ui=False, use_scripts=False)
    images = inspect_dependencies()
    scene = bpy.context.scene
    require(scene.camera is not None and scene.camera.type == 'CAMERA', 'Saved camera required')
    require(not scene.sequence_editor, 'Sequence-editor dependencies are unsupported')
    objects = list(scene.objects)
    roles = {obj.name: classify(obj, config['portable_material_names']) for obj in objects}
    meshes = [obj for obj in objects if obj.type == 'MESH']
    require(meshes and sum(len(obj.data.vertices) for obj in meshes) <= 500000, 'Mesh inspection bound exceeded')
    facts = [mesh_facts(obj, roles[obj.name]) for obj in meshes]
    coincident = [{'objects': [a['name'], b['name']], 'same_ordered_geometry_within_rounding_1e_6': True,
                   'saved_hide_render': [a['hide_render'], b['hide_render']],
                   'saved_show_instancer_for_render': [a['show_instancer_for_render'], b['show_instancer_for_render']]}
                  for i, a in enumerate(facts) for b in facts[i+1:]
                  if a['vertices'] == b['vertices'] and a['faces'] == b['faces']
                  and a['world_geometry_ordered_rounded_sha256'] == b['world_geometry_ordered_rounded_sha256']]
    hair = []
    for obj in objects:
        for system in obj.particle_systems:
            s = system.settings
            require(s.count*(s.rendered_child_count+1) <= 600000, 'Native hair exceeds 600000 configured fibers')
            hair.append({'object': obj.name, 'system': system.name, 'parents_configured': s.count,
                         'parents_stored': len(system.particles), 'render_children': s.rendered_child_count,
                         'configured_fiber_upper_bound': s.count*(s.rendered_child_count+1),
                         'length': s.hair_length, 'radius_scale': s.radius_scale, 'clump_factor': s.clump_factor,
                         'show_emitter_surface': obj.show_instancer_for_render, 'dynamics': system.use_hair_dynamics})
    saved_visibility = {obj.name: obj.hide_render for obj in objects}
    saved_materials = {obj.name: list(obj.data.materials) for obj in meshes}
    parents_before = parent_key_hash()
    report = {'schema_version': 1, 'source_blend_sha256': config['source_blend_sha256'],
              'blender_version': bpy.app.version_string, 'pose': {'frame': scene.frame_current, 'subframe': scene.frame_subframe,
              'armatures': {o.name: o.data.pose_position for o in objects if o.type == 'ARMATURE'}},
              'camera': {'name': scene.camera.name, 'type': scene.camera.data.type, 'ortho_scale': scene.camera.data.ortho_scale,
                         'matrix_world': [list(row) for row in scene.camera.matrix_world]},
              'lights': [{'name': o.name, 'type': o.data.type, 'energy': o.data.energy,
                         'matrix_world': [list(row) for row in o.matrix_world]} for o in objects if o.type == 'LIGHT'],
              'world': scene.world.name if scene.world else None, 'meshes': facts,
              'materials': [material_facts(m) for m in bpy.data.materials], 'images': images, 'hair': hair,
              'coincident_geometry_candidates': coincident, 'variants': [],
              'scope': 'Same saved pose/camera/light; static diagnostic only, not a new asset or animation validation',
              'limitations': ['Coincidence checks compare matching vertex order rounded to 1e-6; they do not test arbitrary intersections.',
                             'A copied groom emitter is expected; matching geometry alone does not prove a visible double surface.',
                             'Native CPU rendering of cards does not establish browser-renderer appearance.'],
              'embedded_text_blocks_not_executed': len(bpy.data.texts), 'external_dependencies_loaded': False}
    scene.render.engine = 'CYCLES'
    scene.cycles.device, scene.cycles.samples, scene.cycles.use_denoising = 'CPU', 16, True
    scene.cycles.shading_system = False
    scene.render.threads_mode, scene.render.threads = 'FIXED', 4
    scene.render.resolution_x = scene.render.resolution_y = 640
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = 'PNG'
    scene.render.image_settings.color_mode = 'RGBA' if scene.render.film_transparent else 'RGB'
    scene.render.use_compositing = scene.render.use_sequencer = False
    for variant in config['variants']:
        for obj in objects:
            obj.hide_render = saved_visibility[obj.name]
            if variant not in ('baseline_native', 'native_parents_only') and roles[obj.name] == 'native_groom':
                obj.hide_render = True
            if variant in ('base_only', 'base_matte_no_normal') and roles[obj.name] == 'portable_cards':
                obj.hide_render = True
            if variant == 'portable_only' and roles[obj.name] == 'portable_cards':
                obj.hide_render = False
        for obj in meshes:
            for index, material in enumerate(saved_materials[obj.name]):
                obj.data.materials[index] = material
        if variant == 'native_parents_only':
            for obj in objects:
                for system in obj.particle_systems:
                    system.settings.rendered_child_count = 0
                    system.settings.child_type = 'NONE'
            require(parent_key_hash() == parents_before, 'Child diagnostic changed stored parent keys')
        overrides = []
        if variant == 'base_matte_no_normal':
            copies = {}
            for obj in meshes:
                if roles[obj.name] != 'base':
                    continue
                for index, material in enumerate(obj.data.materials):
                    if material is None:
                        continue
                    if material.name not in copies:
                        clone = material.copy()
                        clone.name = 'LookdevMatte_'+material.name
                        if clone.use_nodes:
                            for node in clone.node_tree.nodes:
                                if node.type == 'BSDF_PRINCIPLED':
                                    for key, value in (('Metallic', 0), ('Roughness', .8)):
                                        for link in list(node.inputs[key].links):
                                            clone.node_tree.links.remove(link)
                                        node.inputs[key].default_value = value
                                if node.type == 'NORMAL_MAP':
                                    for link in list(node.inputs['Strength'].links):
                                        clone.node_tree.links.remove(link)
                                    node.inputs['Strength'].default_value = 0
                        copies[material.name] = clone
                    obj.data.materials[index] = copies[material.name]
                    overrides.append({'object': obj.name, 'original': material.name, 'clone': copies[material.name].name})
        bpy.context.view_layer.update()
        scene.render.filepath = str(out/(variant+'.png'))
        started = time.monotonic()
        bpy.ops.render.render(write_still=True)
        report['variants'].append({'id': variant, 'path': variant+'.png', 'sha256': digest(out/(variant+'.png')),
                                   'seconds': time.monotonic()-started, 'resolution': [640, 640], 'samples': 16, 'denoised': True,
                                   'hidden_objects': [o.name for o in objects if o.hide_render],
                                   'diagnostic_material_overrides': overrides})
        if variant == 'native_parents_only':
            require(parent_key_hash() == parents_before, 'Parent-only render changed stored parent keys')
            report['variants'][-1]['stored_parent_keys_unchanged'] = parents_before
            report['variants'][-1]['diagnostic_hair_override'] = {'rendered_child_count': 0, 'child_type': 'NONE'}
        write_json(out/(variant+'-receipt.json'), report['variants'][-1])
    report['source_unchanged'] = digest(source) == config['source_blend_sha256']
    write_json(out/'native-report.json', report)


if __name__ == '__main__':
    main(sys.argv[sys.argv.index('--')+1])
