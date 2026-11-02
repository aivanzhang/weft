"""Attach bounded source-colored edge fins before the native groom allocates strands."""

import ctypes
import gc
import hashlib
import json
import os
from pathlib import Path
import stat
import sys
import time

import bpy
import numpy as np

from edge_fringe import build_edge_fringe, export_budget, two_tip_atlas
from fringe_color_atlas import build_color_atlas
from groom_flow import prepare_flow, evaluate_prepared_flow


def require(value, message):
    if not value:
        raise ValueError(message)


def phase_memory(out, label):
    """Record current-process RSS; unknown measurements are never reported as zero."""
    require(isinstance(label, str) and 1 <= len(label) <= 128
            and all(c.isalnum() or c in '_.-' for c in label), 'Invalid memory phase label')
    root = Path(out)
    require(root.is_dir() and not root.is_symlink(), 'Memory phases need an existing output directory')
    path = root/'phase-memory.jsonl'
    require(not path.is_symlink(), 'Memory phase log cannot be a symlink')
    previous = b''
    if path.exists():
        require(path.is_file() and path.stat().st_size <= 128*1024, 'Memory phase log exceeds128KiB')
        previous = path.read_bytes()
        require(not previous or previous.endswith(b'\n'), 'Memory phase log has an incomplete record')
    require(previous.count(b'\n') < 256, 'Memory phase log exceeds256records')
    rss, error = None, None
    try:
        if sys.platform == 'darwin':
            lib = ctypes.CDLL('/usr/lib/libproc.dylib', use_errno=True)
            buffer = ctypes.create_string_buffer(96)
            size = lib.proc_pidinfo(os.getpid(), 4, 0, buffer, len(buffer))
            if size < 16:
                raise OSError(ctypes.get_errno(), 'Cannot inspect current-process memory')
            rss = int.from_bytes(buffer.raw[8:16], 'little')
        elif sys.platform.startswith('linux'):
            rss = int(Path('/proc/self/statm').read_text().split()[1])*os.sysconf('SC_PAGE_SIZE')
        else:
            raise OSError('Current-process RSS is unsupported on '+sys.platform)
        if rss <= 0:
            raise OSError('Current-process RSS is not positive')
    except Exception as failure:
        rss, error = None, str(failure)[:160]
    receipt = {'schema_version': 1, 'phase': label, 'pid': os.getpid(),
               'monotonic_seconds': round(time.monotonic(), 6), 'rss_bytes': rss,
               'status': 'measured' if rss is not None else 'unknown', 'error': error,
               'scope': 'Instantaneous current-process RSS; not a peak or measured savings'}
    encoded = (json.dumps(receipt, sort_keys=True, allow_nan=False)+'\n').encode()
    require(len(previous)+len(encoded) <= 128*1024, 'Memory phase log exceeds128KiB')
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND | getattr(os, 'O_NOFOLLOW', 0), 0o600)
    with os.fdopen(descriptor, 'ab') as handle:
        require(stat.S_ISREG(os.fstat(handle.fileno()).st_mode), 'Memory phase log is not a regular file')
        handle.write(encoded)
    return receipt


def _file_hash(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def release_hidden_fringe_buffers(cards, out):
    """Release only reloadable hidden-coat images, retaining their packed PNG bytes."""
    require(isinstance(cards, (list, tuple)) and 1 <= len(cards) <= 4,
            'Expected1..4 hidden fringe coats')
    root = Path(out).resolve()
    images = {}
    for card in cards:
        require(card.type == 'MESH' and card.name.startswith('PortableEdgeFringe')
                and card.hide_render is True and 1 <= len(card.data.materials) <= 4,
                'Only hidden portable fringe coats may release image buffers')
        for material in card.data.materials:
            require(material and material.use_nodes and material.node_tree
                    and 1 <= len(material.node_tree.nodes) <= 32, 'Invalid hidden fringe material')
            found = [node.image for node in material.node_tree.nodes if node.type == 'TEX_IMAGE']
            require(found and all(image is not None for image in found), 'Hidden fringe image is missing')
            for image in found:
                images[image.as_pointer()] = image
    require(1 <= len(images) <= 4, 'Hidden fringe image count exceeds four')
    plans, packed_bytes = [], 0
    for image in images.values():
        require(image.source == 'FILE' and image.packed_file is not None
                and image.colorspace_settings.name == 'sRGB' and image.alpha_mode == 'STRAIGHT',
                'Hidden fringe images must be packed FILE sRGB/STRAIGHT data')
        size = image.packed_file.size
        require(type(size) is int and 0 < size <= 32*1024**2, 'Hidden fringe packed image exceeds32MiB')
        packed_bytes += size
        require(packed_bytes <= 64*1024**2, 'Hidden fringe packed images exceed64MiB')
        path = Path(bpy.path.abspath(image.filepath_raw))
        require(not path.is_symlink() and path.is_file() and path.resolve().is_relative_to(root/'textures')
                and path.suffix.lower() == '.png' and path.stat().st_size == size,
                'Hidden fringe PNG must remain in the current output textures directory')
        packed_hash = hashlib.sha256(image.packed_file.data).hexdigest()
        require(_file_hash(path) == packed_hash, 'Hidden fringe packed PNG differs from its saved bytes')
        plans.append((image, path.resolve(), packed_hash, size))
    before = phase_memory(out, 'hidden_fringe_release_before')
    rows = []
    for image, path, expected, size in plans:
        image.buffers_free()
        # Reading dimensions or pixels here would decode the image again.
        require(hashlib.sha256(image.packed_file.data).hexdigest() == expected
                and _file_hash(path) == expected, 'Freeing fringe buffers changed packed PNG bytes')
        rows.append({'image': image.name, 'path': str(path.relative_to(root)),
                     'packed_bytes': size, 'sha256': expected, 'packed_and_saved_bytes_unchanged': True})
    after = phase_memory(out, 'hidden_fringe_release_after')
    return {'images': rows, 'before': before, 'after': after, 'geometry_materials_and_settings_unchanged': True,
            'scope': 'Decoded buffers released; actual RSS samples are not guaranteed allocator reclamation'}


def source_pixels(image):
    width, height = image.size
    require(0 < width <= 4096 and 0 < height <= 4096 and width*height <= 16_777_216
            and image.colorspace_settings.name == 'sRGB', 'Fringe requires bounded sRGB source pixels')
    raw = np.empty(width*height*4, dtype=np.float32)
    image.pixels.foreach_get(raw)
    raw = raw.reshape(height, width, 4)
    encoded = np.empty((height, width, 4), dtype=np.uint8)
    for start in range(0, height, 64):
        end = min(start+64, height)
        block = raw[start:end]
        require(np.isfinite(block).all() and block.min() >= 0 and block.max() <= 1,
                'Fringe source image contains nonfinite or out-of-range pixels')
        rounded = np.rint(block*255)
        require(np.max(np.abs(rounded-block*255)) < .001,
                'Fringe requires actual encoded8-bit source pixels, not silent precision reduction')
        encoded[height-end:height-start] = rounded[::-1].astype(np.uint8)
    return encoded


def skin_arrays(obj, rig):
    names = list(rig.data.bones.keys())
    require(1 <= len(names) <= 64, 'Fringe requires 1..64 actual bones')
    slots = {name: i for i, name in enumerate(names)}
    groups = {g.index: slots[g.name] for g in obj.vertex_groups if g.name in slots}
    joints = np.zeros((len(obj.data.vertices), 4), dtype=np.uint16)
    weights = np.zeros((len(obj.data.vertices), 4), dtype=np.float64)
    for index, vertex in enumerate(obj.data.vertices):
        active = sorted(((groups[g.group], g.weight) for g in vertex.groups
                         if g.group in groups and g.weight > 0), key=lambda row: (-row[1], row[0]))
        require(1 <= len(active) <= 4 and all(np.isfinite(w) for _, w in active),
                'Fringe never truncates or invents source bone influences')
        total = sum(w for _, w in active)
        require(abs(total-1) <= 1e-6, 'Fringe requires normalized source skin weights')
        for column, (joint, weight) in enumerate(active):
            joints[index, column], weights[index, column] = joint, weight/total
    return names, joints, weights


def part_labels(bone_names, joints, weights, part_report):
    labels = np.zeros(len(joints), dtype=np.int16)
    conflicts = np.zeros(len(joints), dtype=bool)
    for owner, part in enumerate((part_report or {}).get('parts', []), 1):
        names = part['bone_names']
        require(set(names) <= set(bone_names), 'Fringe part proof names an absent bone')
        selected = (np.isin(joints, [bone_names.index(name) for name in names]) & (weights > 0)).any(axis=1)
        conflicts |= selected & (labels != 0)
        labels[selected] = owner
    return labels, conflicts


def atlas_material(pixels, out):
    phase_memory(out, 'atlas_handoff_before')
    height, width = pixels.shape[:2]
    folder = Path(out)/'textures'
    folder.mkdir(exist_ok=True)
    path = folder/'portable-fringe-color-atlas.png'
    require(not path.exists() and not path.is_symlink(), 'Never overwrite a prior fringe atlas')
    temporary = bpy.data.images.new('SourceColoredFringe', width=width, height=height, alpha=True)
    name = temporary.name
    try:
        temporary.colorspace_settings.name = 'sRGB'
        temporary.alpha_mode = 'STRAIGHT'
        native_pixels = pixels[::-1].astype(np.float32)
        native_pixels /= 255
        temporary.pixels.foreach_set(native_pixels.ravel())
        del native_pixels
        temporary.filepath_raw, temporary.file_format = str(path), 'PNG'
        temporary.save()
    finally:
        bpy.data.images.remove(temporary)
    require(path.is_file() and 0 < path.stat().st_size <= 32*1024**2, 'Fringe PNG exceeds32MiB')
    expected_hash = _file_hash(path)
    image = bpy.data.images.load(str(path), check_existing=False)
    image.name = name
    require(image.name == name and image.source == 'FILE', 'Fringe image identity changed during PNG handoff')
    image.colorspace_settings.name = 'sRGB'
    image.alpha_mode = 'STRAIGHT'
    image.pack()
    require(image.packed_file is not None
            and hashlib.sha256(image.packed_file.data).hexdigest() == expected_hash,
            'Packed fringe image differs from the saved PNG')
    image.buffers_free()
    phase_memory(out, 'atlas_handoff_after')
    material = bpy.data.materials.new('SourceColoredFringe')
    material.use_nodes, material.use_backface_culling = True, True
    nodes, links = material.node_tree.nodes, material.node_tree.links
    bsdf = nodes.get('Principled BSDF')
    bsdf.inputs['Metallic'].default_value = 0
    bsdf.inputs['Roughness'].default_value = .92
    texture = nodes.new('ShaderNodeTexImage')
    texture.image, texture.interpolation, texture.extension = image, 'Linear', 'EXTEND'
    links.new(texture.outputs['Color'], bsdf.inputs['Base Color'])
    cutout = nodes.new('ShaderNodeMath')
    cutout.operation, cutout.inputs[1].default_value = 'GREATER_THAN', .18
    links.new(texture.outputs['Alpha'], cutout.inputs[0])
    links.new(cutout.outputs[0], bsdf.inputs['Alpha'])
    return material, {'material': material.name, 'profile': 'edge-fringe-v1',
                      'path': str(path.relative_to(out)), 'dimensions': [width, height],
                      'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                      'alpha_mode': 'MASK', 'alpha_cutoff': .18, 'double_sided': False,
                      'metallic': 0, 'roughness': .92, 'vertex_color_multiplier': False}


def attach_edge_fringe(obj, rig, mask, images, config, native_groom, groom_flow, out,
                       source_budget, *, root_contract, part_report=None, undercoat_report=None):
    started = time.monotonic()
    mesh = obj.data
    mesh.calc_loop_triangles()
    require(1 <= len(mesh.vertices) <= 100000 and 1 <= len(mesh.loop_triangles) <= 100000,
            'Fringe emitter exceeds geometry bounds')
    require(isinstance(mask, np.ndarray) and mask.shape == (len(mesh.vertices),) and mask.dtype == bool,
            'Fringe requires the final measured vertex mask')
    require(rig.data.pose_position == 'REST' and np.allclose(np.asarray(obj.matrix_world), np.eye(4), atol=1e-7),
            'Fringe emitter needs an applied meter-scale rest transform')
    budget = export_budget(max(source_budget['source_export_vertices'], len(mesh.vertices)),
                           max(source_budget['source_export_triangles'], len(mesh.loop_triangles)),
                           config['settings']['max_fins'])
    require(len(mesh.materials) <= 32, 'Fringe supports at most32 emitter materials')
    points = np.empty((len(mesh.vertices), 3), dtype=np.float64)
    mesh.vertices.foreach_get('co', points.ravel())
    faces = np.asarray([tuple(t.vertices) for t in mesh.loop_triangles], dtype=np.int32)
    loops = np.asarray([tuple(t.loops) for t in mesh.loop_triangles], dtype=np.int32)
    materials = np.asarray([t.material_index for t in mesh.loop_triangles], dtype=np.int32)
    corner_normals = np.asarray([tuple(n.vector) for n in mesh.corner_normals], dtype=np.float64)[loops]
    magnitudes = np.linalg.norm(corner_normals, axis=2)
    require(np.isfinite(magnitudes).all() and np.max(np.abs(magnitudes-1)) < .001,
            'Fringe requires finite unit source corner normals')
    corner_normals /= magnitudes[..., None]
    aliases = {row['material']: row['source_material']
               for row in (undercoat_report or {}).get('materials', [])}
    material_names = [m.name if m else '' for m in mesh.materials]
    original_names = [aliases.get(name, name) for name in material_names]
    excluded = set(config['excluded_materials'])
    require(excluded <= set(original_names)|set(material_names), 'A fringe excluded material is absent')
    closure = np.array([original_names[slot] in excluded or material_names[slot] in excluded
                        for slot in materials], dtype=bool)
    bone_names, joints, weights = skin_arrays(obj, rig)
    labels, conflicts = part_labels(bone_names, joints, weights, part_report)
    mixed = (np.ptp(labels[faces], axis=1) != 0) | conflicts[faces].any(axis=1)
    parts = labels[faces[:, 0]].astype(np.int32)
    eligible = mask[faces].all(axis=1) & ~mixed & ~closure
    slots = sorted(map(int, np.unique(materials[eligible])))
    require(slots, 'Fringe has no eligible textured faces')
    image_map = images if isinstance(images, dict) else {slot: images for slot in slots}
    require(set(slots) <= set(image_map), 'An eligible fringe material has no source image')
    sampling, unique_images = [], {}
    for slot in slots:
        image = image_map[slot]
        sampling.append(root_contract(obj, image, [slot]))
        require(0 < image.size[0] <= 4096 and 0 < image.size[1] <= 4096
                and image.colorspace_settings.name == 'sRGB', 'Invalid fringe source texture')
        unique_images[image.as_pointer()] = image
        shader = next(n for n in mesh.materials[slot].node_tree.nodes if n.type == 'BSDF_PRINCIPLED')
        require(not shader.inputs['Alpha'].links and shader.inputs['Alpha'].default_value == 1,
                'Fringe source colors require opaque selected surfaces')
    require(sum(image.size[0]*image.size[1] for image in unique_images.values()) <= 16_777_216,
            'Fringe source textures exceed aggregate16MP limit')
    uv = np.asarray([tuple(v.uv) for v in mesh.uv_layers.active.data], dtype=np.float64)[loops]
    uv[..., 1] = 1-uv[..., 1]
    field = prepare_flow(points.min(axis=0), np.ptp(points, axis=0),
                         native_groom['direction'], native_groom['comb'], groom_flow)
    directions = np.empty_like(points)
    for index, point in enumerate(points):
        if index % 256 == 0:
            require(time.monotonic()-started < 60, 'Fringe adapter exceeded60seconds')
        directions[index] = evaluate_prepared_flow(point, field)[0]
    fins = build_edge_fringe(points, faces, corner_normals, uv, materials, parts, eligible,
        ~eligible, closure, joints, weights, directions, np.asarray(config['design_views'], dtype=float),
        config['settings'], source_export_vertices=budget['source_export_vertices'],
        source_export_triangles=budget['source_export_triangles'])
    pixels = {key: source_pixels(image) for key, image in unique_images.items()}
    textures = {slot: {'pixels': pixels[image_map[slot].as_pointer()], 'factor': np.ones(3)} for slot in slots}
    atlas = build_color_atlas(fins, fins['material_ids'], textures,
                             two_tip_atlas(config['settings']['seed'])[:, :, 3])
    require(atlas['report']['technical_gate_passed'], 'Fringe atlas failed its fixed nominal-tip color/alpha gate')
    material, material_report = atlas_material(atlas['pixels'], Path(out))
    # Blender rejects opposed polygons sharing indices; reserve separate triangle corners.
    corner_ids = fins['triangles'].reshape(-1)
    corner_faces = np.arange(len(corner_ids), dtype=np.int32).reshape(-1, 3)
    output_mesh = bpy.data.meshes.new('PortableEdgeFringe')
    output_mesh.from_pydata(fins['positions'][corner_ids].tolist(), [], corner_faces.tolist())
    output_mesh.update()
    out_uv = atlas['uv'][corner_ids].copy()
    out_uv[:, 1] = 1-out_uv[:, 1]
    layer = output_mesh.uv_layers.new(name='FringeUV')
    for loop in output_mesh.loops:
        layer.data[loop.index].uv = out_uv[loop.vertex_index]
    output_mesh.normals_split_custom_set_from_vertices(fins['normals'][corner_ids].tolist())
    output_mesh.materials.append(material)
    coat = bpy.data.objects.new('PortableEdgeFringe', output_mesh)
    bpy.context.scene.collection.objects.link(coat)
    coat.matrix_world = obj.matrix_world.copy()
    groups = [coat.vertex_groups.new(name=name) for name in bone_names]
    for index, (ids, values) in enumerate(zip(fins['joints'][corner_ids], fins['weights'][corner_ids])):
        for slot, weight in zip(ids, values):
            if weight > 0:
                groups[int(slot)].add([index], float(weight), 'REPLACE')
    coat.modifiers.new('PortableSkin', 'ARMATURE').object = rig
    saved = {key: fins[key] for key in ('positions', 'normals', 'triangles', 'joints', 'weights',
              'root_vertex_ids', 'root_face_ids', 'root_corner_ids', 'root_source_uv', 'material_ids')}
    np.savez_compressed(Path(out)/'fringe-correspondence.npz', **saved, uv_gltf=atlas['uv'],
                        native_vertex_to_helper_vertex=corner_ids, **atlas['permutation'])
    report = {'cards': len(fins['root_vertex_ids']), 'vertices': len(corner_ids),
              'native_topology': 'Independent triangle corners; exact helper positions/normals/UVs/skin duplicated',
              'geometry_profile': 'edge-fringe-v1', 'sample_root_color': True,
              'material': material_report, 'geometry': fins['report'], 'atlas': atlas['report'],
              'budget': budget, 'sampling': sampling, 'config': config,
              'bone_order': bone_names, 'mixed_part_faces_excluded': int(mixed.sum()),
              'explicit_material_faces_excluded': int(closure.sum()),
              'geometry_arrays_npz': 'fringe-correspondence.npz',
              'source_point_sha256': hashlib.sha256(points.astype('<f8').tobytes()).hexdigest(),
              'elapsed_seconds': time.monotonic()-started, 'quality_approved': False,
              'limits': ['Regional direction is sampled; fin length/lean are explicit uniform controls, not native-strand parity.',
                         'MASK filtering and source shading can still differ at small scales; inspect the exported model.',
                         'No dense-fur, Muse-parity, contact repair, physical-hair or fresh-prompt acceptance claim.']}
    (Path(out)/'fringe-report.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    del pixels, textures, atlas, fins, saved, directions, uv, corner_normals
    gc.collect()
    require(time.monotonic()-started < 60, 'Fringe adapter exceeded60seconds')
    return [coat], [report]
