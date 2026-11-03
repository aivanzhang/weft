"""Refine material sockets without repainting textures or editing skin geometry."""

import numpy as np

try:
    from surface_profiles import validate_profiles
except ModuleNotFoundError as error:
    if error.name != 'surface_profiles':
        raise
    from avatar_assets.surface_profiles import validate_profiles


MAX_VERTICES = 100000
MAX_FACES = 100000
MAX_LOOPS = 300000
MAX_NEW_MATERIALS = 32


def _points(value, name):
    if isinstance(value, np.ndarray):
        if (value.ndim != 2 or value.shape[1] != 3 or not 1 <= len(value) <= MAX_VERTICES or
                value.dtype.kind not in 'fiu'):
            raise ValueError(f'Invalid surface {name} array')
    else:
        if not isinstance(value, (list, tuple)) or not 1 <= len(value) <= MAX_VERTICES:
            raise ValueError(f'Invalid or oversized surface {name}')
        for row in value:
            if not isinstance(row, (list, tuple)) or len(row) != 3 or any(
                    type(number) not in (int, float) for number in row):
                raise ValueError(f'Invalid surface {name} row')
    result = np.asarray(value, dtype=float)
    if not np.isfinite(result).all():
        raise ValueError(f'Surface {name} must be finite')
    return result


def _vector(value):
    if isinstance(value, np.ndarray):
        if value.shape != (3,) or value.dtype.kind not in 'fiu':
            raise ValueError('Invalid surface bounds vector')
    elif not isinstance(value, (list, tuple)) or len(value) != 3 or any(
            type(number) not in (int, float) for number in value):
        raise ValueError('Invalid surface bounds vector')
    result = np.asarray(value, dtype=float)
    if not np.isfinite(result).all():
        raise ValueError('Surface bounds must be finite')
    return result


def _face_indices(faces, vertex_count):
    if isinstance(faces, np.ndarray):
        if (faces.ndim != 2 or not 1 <= len(faces) <= MAX_FACES or not 3 <= faces.shape[1] <= 64 or faces.size > MAX_LOOPS or
                faces.dtype.kind not in 'iu'):
            raise ValueError('Invalid surface face array')
        sizes = np.full(len(faces), faces.shape[1], dtype=np.int64)
        flat = faces.reshape(-1)
    else:
        if not isinstance(faces, (list, tuple)) or not 1 <= len(faces) <= MAX_FACES:
            raise ValueError('Invalid or oversized surface face list')
        sizes, total = [], 0
        for face in faces:
            if not isinstance(face, (tuple, list)) or not 3 <= len(face) <= 64 or any(
                    type(index) is not int or not 0 <= index < vertex_count for index in face):
                raise ValueError('Surface faces require bounded integer vertex indices')
            total += len(face)
            if total > MAX_LOOPS:
                raise ValueError('Surface face corners exceed 300000')
            sizes.append(len(face))
        flat = np.asarray([index for face in faces for index in face], dtype=np.int64)
        sizes = np.asarray(sizes, dtype=np.int64)
    if flat.min() < 0 or flat.max() >= vertex_count:
        raise ValueError('Surface face vertex index is out of bounds')
    starts = np.concatenate(([0], np.cumsum(sizes[:-1])))
    return flat, sizes, starts


def select_surface_faces(faces, colors, points, low, span, profiles):
    """Select by centroid and mean source RGB; enforce final last-match ownership."""
    profiles = validate_profiles(profiles)
    points, colors = _points(points, 'points'), _points(colors, 'colors')
    if points.shape != colors.shape or np.any(colors < 0) or np.any(colors > 1):
        raise ValueError('Source RGB colors must match vertices and stay within 0–1')
    low, span = _vector(low), _vector(span)
    if np.any(span < 0) or not np.any(span > 0):
        raise ValueError('Invalid surface normalization bounds')
    flat, sizes, starts = _face_indices(faces, len(points))
    try:
        with np.errstate(over='raise', invalid='raise', divide='raise'):
            centroids = np.add.reduceat(points[flat], starts, axis=0)/sizes[:, None]
            rgb = np.add.reduceat(colors[flat], starts, axis=0)/sizes[:, None]
            normalized = (centroids-low)/np.maximum(span, 1e-8)
    except FloatingPointError as error:
        raise ValueError('Surface geometry exceeds the finite numerical range') from error
    brightness, chroma = rgb.mean(axis=1), rgb.max(axis=1)-rgb.min(axis=1)
    owners, reports = np.full(len(sizes), -1, dtype=np.int16), []
    for index, profile in enumerate(profiles):
        inside = np.zeros(len(sizes), dtype=bool) if profile['regions'] else np.ones(len(sizes), dtype=bool)
        for region in profile['regions']:
            try:
                with np.errstate(over='raise', invalid='raise', divide='raise'):
                    inside |= np.sum(((normalized-region['center'])/region['radius'])**2, axis=1) <= 1
            except FloatingPointError as error:
                raise ValueError('Surface region exceeds the finite numerical range') from error
        selected = (inside & (brightness >= profile['brightness'][0]) &
                    (brightness < profile['brightness'][1]) & (chroma <= profile['max_chroma']))
        owners[selected] = index
        reports.append({'name': profile['name'], 'matched_faces': int(selected.sum())})
    violations = []
    for index, profile in enumerate(profiles):
        count = int(np.count_nonzero(owners == index))
        reports[index]['selected_faces'] = count
        if not profile['min_faces'] <= count <= profile['max_faces']:
            violations.append(f"{profile['name']}={count} allowed[{profile['min_faces']},{profile['max_faces']}]")
    if violations:
        # Eight validated names/counts fit below 1024 bytes without hiding later failures.
        raise ValueError('Surface profile final face counts outside declared bounds: '+ '; '.join(violations))
    return owners, reports


def _links(socket):
    return sorted((link.from_node.name, link.from_socket.name) for link in socket.links)


def _material_contract(material):
    if material is None or not material.use_nodes or material.node_tree is None:
        raise ValueError('Surface profiles need node-based source materials')
    tree = material.node_tree
    if not 1 <= len(tree.nodes) <= 256 or len(tree.links) > 1024:
        raise ValueError('Source material exceeds bounded node graph limits')
    principled = [node for node in tree.nodes if node.type == 'BSDF_PRINCIPLED']
    if not principled:
        raise ValueError('Surface profiles require a Principled source shader')
    base_color, images, settings = {}, {}, {}
    for node in principled:
        if any(name not in node.inputs for name in ('Base Color', 'Roughness', 'Metallic')):
            raise ValueError('Source shader is missing required material inputs')
        base_color[node.name] = (_links(node.inputs['Base Color']), tuple(node.inputs['Base Color'].default_value))
        for name in ('Roughness', 'Metallic'):
            settings[node.name, name] = (_links(node.inputs[name]), node.inputs[name].default_value)
    for node in tree.nodes:
        if node.type == 'NORMAL_MAP':
            if 'Strength' not in node.inputs:
                raise ValueError('Source normal map is missing its strength input')
            settings[node.name, 'Strength'] = (_links(node.inputs['Strength']), node.inputs['Strength'].default_value)
        if node.type == 'TEX_IMAGE':
            images[node.name] = (node.image, _links(node.inputs['Vector']) if 'Vector' in node.inputs else [])
    return base_color, images, settings


def apply_surface_profiles(obj, colors, points, low, span, profiles):
    """Clone source materials per winning profile/slot; originals remain unchanged."""
    profiles = validate_profiles(profiles)
    if obj.type != 'MESH' or not 1 <= len(obj.data.vertices) <= MAX_VERTICES:
        raise ValueError('Surface profiles require one bounded mesh')
    if not 1 <= len(obj.data.polygons) <= MAX_FACES or len(points) != len(obj.data.vertices):
        raise ValueError('Surface mesh/point count mismatch')
    faces = [tuple(polygon.vertices) for polygon in obj.data.polygons]
    owners, reports = select_surface_faces(faces, colors, points, low, span, profiles)
    sources = list(obj.data.materials)
    slots = np.array([polygon.material_index for polygon in obj.data.polygons], dtype=np.int64)
    selected = np.flatnonzero(owners >= 0)
    if slots[selected].min() < 0 or slots[selected].max() >= len(sources):
        raise ValueError('Selected faces reference an absent source material')
    pairs = sorted(set(zip(map(int, owners[selected]), map(int, slots[selected]))))
    if len(pairs) > MAX_NEW_MATERIALS:
        raise ValueError('Surface profiles exceed 32 new materials')
    contracts = {slot: _material_contract(sources[slot]) for slot in sorted({slot for _, slot in pairs})}
    copies = []
    for profile_index, slot in pairs:
        profile, original = profiles[profile_index], sources[slot]
        material = original.copy()
        if material is original or material.node_tree is original.node_tree:
            raise ValueError('Material copy did not isolate the source node graph')
        material.name = f"{original.name}__surface_{profile['name']}"
        removed = {'roughness': 0, 'metallic': 0, 'normal_strength': 0}
        normal_maps = 0
        for node in material.node_tree.nodes:
            settings = []
            if node.type == 'BSDF_PRINCIPLED':
                settings = [('Roughness', 'roughness'), ('Metallic', 'metallic')]
            elif node.type == 'NORMAL_MAP':
                normal_maps += 1
                settings = [('Strength', 'normal_strength')]
            for socket_name, setting in settings:
                socket = node.inputs[socket_name]
                for link in list(socket.links):
                    material.node_tree.links.remove(link)
                    removed[setting] += 1
                socket.default_value = profile[setting]
        copied = _material_contract(material)
        if (copied[:2] != contracts[slot][:2] or _material_contract(original) != contracts[slot] or
                any(copied[1][name][0] is not image for name, (image, _) in contracts[slot][1].items())):
            raise ValueError('Surface refinement changed a source or base-color/UV texture binding')
        copies.append((profile_index, slot, material, normal_maps, removed))
    for report, profile in zip(reports, profiles):
        report['settings'] = {key: profile[key] for key in ('roughness', 'metallic', 'normal_strength')}
        report['material_copies'] = []
    assignments = {}
    for profile_index, slot, material, normal_maps, removed in copies:
        target = len(obj.data.materials)
        obj.data.materials.append(material)
        assignments[profile_index, slot] = target
        reports[profile_index]['material_copies'].append({
            'source_slot': slot, 'source_material': sources[slot].name,
            'new_slot': target, 'material': material.name,
            'selected_faces': int(np.count_nonzero((owners == profile_index) & (slots == slot))),
            'normal_map_nodes': normal_maps, 'removed_input_links': removed,
            'texture_images': [{'node': name, 'image': image.name if image else None,
                                'dimensions': list(image.size) if image else None}
                               for name, (image, _) in contracts[slot][1].items()]})
    for index in selected:
        obj.data.polygons[int(index)].material_index = assignments[int(owners[index]), int(slots[index])]
    return {'precedence': 'last_matching_profile_wins', 'new_material_count': len(copies),
            'faces_reassigned': len(selected), 'profiles': reports,
            'selection': 'Normalized world-bound face centroids; any region or all when empty. Mean sampled vertex RGB; brightness=channel mean, chroma=max-min. Brightness upper bound is exclusive.',
            'source_materials_unchanged': True, 'geometry_uv_skin_unchanged': True,
            'base_color_textures_preserved': True,
            'texture_scope': 'Existing image datablocks and UV routing are reused unchanged; no repainting, resampling, new resolution or added texture detail.'}
