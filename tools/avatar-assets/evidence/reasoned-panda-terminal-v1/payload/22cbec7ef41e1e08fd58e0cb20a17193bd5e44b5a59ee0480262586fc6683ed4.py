"""Clone material slots for bounded authored color fixes; never repaint source images."""

import numpy as np

try:
    from color_corrections import MAX_CORNERS, MAX_FACES, MAX_VERTICES, select_color_faces, srgb_to_linear, validate_corrections
except ModuleNotFoundError as error:
    if error.name != 'color_corrections':
        raise
    from avatar_assets.color_corrections import MAX_CORNERS, MAX_FACES, MAX_VERTICES, select_color_faces, srgb_to_linear, validate_corrections

try:
    from blender_surface_profiles import _material_contract
except ModuleNotFoundError as error:
    if error.name != 'blender_surface_profiles':
        raise
    from scripts.blender_surface_profiles import _material_contract


MAX_NEW_MATERIALS = 32


def apply_color_corrections(obj, colors, points, low, span, corrections):
    """Return (JSON report, sorted vertex IDs to exclude from subsequent fur sampling)."""
    corrections = validate_corrections(corrections)
    if obj.type != 'MESH' or not 1 <= len(obj.data.vertices) <= MAX_VERTICES:
        raise ValueError('Color corrections require one bounded mesh')
    if not 1 <= len(obj.data.polygons) <= MAX_FACES or len(points) != len(obj.data.vertices):
        raise ValueError('Color correction mesh/point count mismatch')
    if hasattr(obj.data, 'loops') and len(obj.data.loops) > MAX_CORNERS:
        raise ValueError('Color correction mesh exceeds 300000 corners')
    faces = [tuple(p.vertices) for p in obj.data.polygons]
    owners, reports = select_color_faces(faces, colors, points, low, span, corrections)
    sources = list(obj.data.materials)
    slots = np.asarray([p.material_index for p in obj.data.polygons], dtype=np.int64)
    selected = np.flatnonzero(owners >= 0)
    if slots[selected].min() < 0 or slots[selected].max() >= len(sources):
        raise ValueError('Corrected faces reference an absent source material')
    pairs = sorted(set(zip(map(int, owners[selected]), map(int, slots[selected]))))
    if len(pairs) > MAX_NEW_MATERIALS:
        raise ValueError('Color corrections exceed 32 new materials')
    contracts = {slot: _material_contract(sources[slot]) for slot in sorted({slot for _, slot in pairs})}
    copies = []
    for correction_index, slot in pairs:
        correction, original = corrections[correction_index], sources[slot]
        material = original.copy()
        if material is original or material.node_tree is original.node_tree or any(
                node is source for node in material.node_tree.nodes for source in original.node_tree.nodes):
            raise ValueError('Material copy did not isolate the source node graph')
        material.name = f"{original.name}__color_{correction['name']}"
        linear = (*srgb_to_linear(correction['color_srgb']), 1.)
        removed = {'base_color': 0, 'roughness': 0, 'metallic': 0, 'normal_strength': 0}
        normal_maps = 0
        for node in material.node_tree.nodes:
            settings = []
            if node.type == 'BSDF_PRINCIPLED':
                settings = [('Base Color', 'base_color', linear), ('Roughness', 'roughness', correction['roughness']),
                            ('Metallic', 'metallic', 0.)]
            elif node.type == 'NORMAL_MAP':
                settings = [('Strength', 'normal_strength', .1)]
                normal_maps += 1
            for socket_name, setting, value in settings:
                socket = node.inputs[socket_name]
                for link in list(socket.links):
                    material.node_tree.links.remove(link)
                    removed[setting] += 1
                socket.default_value = value
        copied = _material_contract(material)
        if (copied[1] != contracts[slot][1] or _material_contract(original) != contracts[slot] or
                any(copied[1][name][0] is not image for name, (image, _) in contracts[slot][1].items())):
            raise ValueError('Color correction changed a source material or original image/UV routing')
        copies.append((correction_index, slot, material, normal_maps, removed))
    for report, correction in zip(reports, corrections):
        report['settings'] = {'color_srgb': correction['color_srgb'], 'color_linear': list(srgb_to_linear(correction['color_srgb'])),
                              'roughness': correction['roughness'], 'metallic': 0., 'normal_strength': .1}
        report['material_copies'] = []
    assignments = {}
    for correction_index, slot, material, normal_maps, removed in copies:
        target = len(obj.data.materials)
        obj.data.materials.append(material)
        assignments[correction_index, slot] = target
        reports[correction_index]['material_copies'].append({
            'source_slot': slot, 'source_material': sources[slot].name, 'new_slot': target, 'material': material.name,
            'selected_faces': int(np.count_nonzero((owners == correction_index) & (slots == slot))),
            'normal_map_nodes': normal_maps, 'removed_input_links': removed,
            'original_texture_images': [{'node': name, 'image': image.name if image else None,
                                        'dimensions': list(image.size) if image else None}
                                       for name, (image, _) in contracts[slot][1].items()]})
    for index in selected:
        obj.data.polygons[int(index)].material_index = assignments[int(owners[index]), int(slots[index])]
    affected = sorted({vertex for index in selected for vertex in faces[int(index)]})
    report = {'precedence': 'last_matching_correction_wins', 'corrections': reports,
              'new_material_count': len(copies), 'faces_reassigned': len(selected), 'affected_vertices': len(affected),
              'selection': 'Normalized world-bound face centroid inside any ellipsoid AND mean source RGB dominant channel strictly greater than ratio times maximum other channel.',
              'source_materials_unchanged': True, 'geometry_uv_skin_unchanged': True,
              'original_texture_bytes_unchanged': True, 'visible_selected_surface_color_authored': True,
              'texture_scope': 'Original image datablocks and UV routing are reused without pixel/file writes; selected face materials use an authored constant linear RGB instead of their Base Color texture link. This is not texture repainting or recovered hidden-surface color.',
              'fur_exclusion_required': True,
              'fur_exclusion_scope': 'Exclude every returned vertex ID from native and portable fur sampling because the original root-color texture was not changed.'}
    return report, affected
