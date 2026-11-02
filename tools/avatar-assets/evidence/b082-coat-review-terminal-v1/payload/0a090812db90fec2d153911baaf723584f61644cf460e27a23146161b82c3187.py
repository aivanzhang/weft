"""Explicit material-owned source colors; never rewrite the body's materials or UVs."""

import hashlib
import math

import numpy as np

from material_fur import constant_selection_rgb, selection_rgb, unique_vertex_sources


POLICY = 'material-owned-fur-v1'
MAX_PIXELS = 16_777_216


def require(value, message):
    if not value:
        raise ValueError(message)


def describe_sources(obj, material_names):
    mesh = obj.data
    require(0 < len(mesh.vertices) <= 100000 and 0 < len(mesh.loops) <= 300000,
            'Material-owned colors require bounded nonempty source geometry')
    require(0 < len(mesh.materials) <= 32 and len(mesh.uv_layers) <= 8,
            'Material-owned colors require bounded materials and UV layers')
    require(all(len(p.vertices) == 3 for p in mesh.polygons), 'Material-owned fur requires source triangles')
    slots = sorted({p.material_index for p in mesh.polygons})
    require(all(0 <= slot < len(mesh.materials) for slot in slots), 'Invalid source material ownership')
    used_names = [mesh.materials[slot].name for slot in slots if mesh.materials[slot]]
    require(len(used_names) == len(slots) and len(set(used_names)) == len(slots),
            'Material-owned colors require distinct named materials')
    require(set(material_names) <= set(used_names), 'A requested fur material has no source faces')
    active = mesh.uv_layers.active
    render_layers = [layer for layer in mesh.uv_layers if layer.active_render]
    sources, rows, images = {}, [], {}
    for slot in slots:
        material = mesh.materials[slot]
        require(material.use_nodes and material.node_tree and 0 < len(material.node_tree.nodes) <= 256,
                'Material-owned colors require bounded node-based source materials')
        nodes = material.node_tree.nodes
        shaders = [node for node in nodes if node.type == 'BSDF_PRINCIPLED']
        outputs = [node for node in nodes if node.type == 'OUTPUT_MATERIAL' and node.is_active_output]
        require(len(shaders) == len(outputs) == 1, 'Require one active Principled source surface')
        shader = shaders[0]
        links = outputs[0].inputs['Surface'].links
        require(len(links) == 1 and links[0].from_node == shader,
                'Mixed or procedural source surfaces are unsupported for material-owned fur')
        if material.name in material_names:
            alpha = shader.inputs['Alpha']
            require(not alpha.links and alpha.default_value == 1,
                    'Material-owned fur requires opaque selected surfaces')
        base = shader.inputs['Base Color']
        descriptor = {'material_name': material.name, 'factor': [1., 1., 1.],
                      'image': None, 'linear_rgb': None, 'uv_layer': None, 'node': None}
        row = {'slot': slot, 'material_name': material.name, 'emits_fur': material.name in material_names}
        if not base.links:
            rgba = tuple(float(value) for value in base.default_value)
            require(len(rgba) == 4 and all(math.isfinite(x) and 0 <= x <= 1 for x in rgba),
                    'Constant source colors must be finite normalized linear RGBA')
            descriptor.update(kind='constant', linear_rgb=list(rgba[:3]))
            row.update(kind='constant', linear_rgba=list(rgba))
        else:
            require(len(base.links) == 1, 'Ambiguous source base-color graph')
            node = base.links[0].from_node
            require(node.type == 'TEX_IMAGE' and node.image is not None,
                    'Only direct image or constant base colors are supported; factors and vertex-color graphs need explicit adapters')
            require(base.links[0].from_socket == node.outputs['Color'],
                    'Base color must use the source image Color output, not alpha')
            require(node.interpolation == 'Linear' and node.extension == 'REPEAT' and node.projection == 'FLAT'
                    and not node.inputs['Vector'].links,
                    'Material-owned images require Linear/REPEAT/FLAT active UVs without coordinate nodes')
            require(active is not None and len(render_layers) == 1 and render_layers[0] == active,
                    'Sampled and render-active source UV layers must match')
            image = node.image
            width, height = tuple(image.size)
            require(0 < width*height <= MAX_PIXELS and max(width, height) <= 4096
                    and image.colorspace_settings.name == 'sRGB',
                    'Material-owned colors require bounded explicit sRGB images')
            packed = image.packed_file
            require(packed is not None and 0 < packed.size <= 64*1024**2,
                    'Material-owned image provenance requires bounded embedded bytes')
            key = image.as_pointer()
            if key not in images:
                images[key] = {'image': image, 'width': width, 'height': height,
                               'packed_sha256': hashlib.sha256(packed.data).hexdigest()}
            descriptor.update(kind='image', image=image, uv_layer=active.name, node=node)
            row.update(kind='image', source_image=image.name, dimensions=[width, height],
                       packed_sha256=images[key]['packed_sha256'], uv_layer=active.name,
                       factor_linear_rgb=[1., 1., 1.], interpolation='Linear', extension='REPEAT')
        sources[slot] = descriptor
        rows.append(row)
    pixels = sum(row['width']*row['height'] for row in images.values())
    require(pixels <= MAX_PIXELS, 'Material-owned source images exceed the aggregate16MP color budget')
    return sources, {'policy': POLICY, 'materials': rows, 'unique_source_images': len(images),
                     'aggregate_source_pixels': pixels, 'maximum_live_cpu_pixel_buffers': 1,
                     'source_materials_changed': False, 'source_uvs_changed': False}


def pixel_buffer(image):
    width, height = tuple(image.size)
    require(0 < width*height <= MAX_PIXELS, 'Source pixel buffer exceeds16MP')
    pixels = np.empty((height, width, 4), dtype=np.float32)
    image.pixels.foreach_get(pixels.ravel())
    return pixels


def sample_selection(obj, sources):
    mesh = obj.data
    ids = np.empty(len(mesh.loops), dtype=np.int32)
    mesh.loops.foreach_get('vertex_index', ids)
    slots = np.empty(len(mesh.loops), dtype=np.int32)
    for polygon in mesh.polygons:
        slots[polygon.loop_start:polygon.loop_start+polygon.loop_total] = polygon.material_index
    uvs = np.zeros((len(mesh.loops), 2), dtype=np.float32)
    if mesh.uv_layers.active is not None:
        mesh.uv_layers.active.data.foreach_get('uv', uvs.ravel())
    first, ownership = unique_vertex_sources(len(mesh.vertices), ids, slots, uvs,
                                            {slot for slot, source in sources.items() if source['kind'] == 'image'})
    vertex_slots, vertex_uvs = slots[first], uvs[first]
    colors = np.empty((len(mesh.vertices), 3), dtype=np.float32)
    by_image, calls, peak_bytes = {}, 0, 0
    for slot, source in sources.items():
        selected = np.flatnonzero(vertex_slots == slot)
        if source['kind'] == 'constant':
            colors[selected] = constant_selection_rgb(source['linear_rgb'], len(selected))
        else:
            by_image.setdefault(source['image'].as_pointer(), []).append((slot, selected))
    for entries in by_image.values():
        pixels = pixel_buffer(sources[entries[0][0]]['image'])
        calls += 1
        peak_bytes = max(peak_bytes, pixels.nbytes)
        for slot, selected in entries:
            colors[selected] = selection_rgb(pixels, vertex_uvs[selected], factor=sources[slot]['factor'])
        del pixels
    return colors, {'policy': 'unique-owner-legacy-nearest-encoded-selection-v1',
                    'ownership': ownership, 'selection_sampling_calls': calls,
                    'sampled_vertices': len(colors), 'maximum_cpu_pixel_buffer_bytes': peak_bytes,
                    'buffer_lifetime': 'One unique image per sequential selection phase; no pixel buffer retained',
                    'scope': 'Representative selection RGB, not exact root color or renderer filtering'}


def context_for_source(obj, cfg):
    names = cfg['material_fur']['materials']
    sources, source_report = describe_sources(obj, names)
    colors, selection = sample_selection(obj, sources)
    return {'policy': POLICY, 'sources': sources}, colors, {
        'status': 'sampled', 'source_materials': source_report, **selection,
        'exact_root_contract': 'Each emitted material is validated separately during portable root sampling'}


def owned_vertex_mask(obj, material_names):
    wanted = set(material_names)
    mask = np.zeros(len(obj.data.vertices), dtype=bool)
    for polygon in obj.data.polygons:
        if obj.data.materials[polygon.material_index].name in wanted:
            mask[list(polygon.vertices)] = True
    return mask
