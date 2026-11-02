"""Apply bounded part skin weights only after material boundaries are split."""

import numpy as np

try:
    from part_rigs import validate_part_rigs, validate_part_points, part_skin_weights, part_rig_config
except ModuleNotFoundError as error:
    if error.name != 'part_rigs':
        raise
    from avatar_assets.part_rigs import validate_part_rigs, validate_part_points, part_skin_weights, part_rig_config


MAX_FACES = 100000
MAX_CORNERS = 300000


def _select_vertices(obj, parts):
    polygons, materials = obj.data.polygons, obj.data.materials
    if not 1 <= len(polygons) <= MAX_FACES or not 1 <= len(materials) <= 64:
        raise ValueError('Part mesh exceeds bounded face/material counts')
    material_names = [material.name if material is not None else None for material in materials]
    owners = {name: index+1 for index, part in enumerate(parts) for name in part['materials']}
    if not set(owners) <= set(material_names):
        raise ValueError('Requested part material is absent from the mesh')
    vertices, face_owners, sizes = [], [], []
    material_faces = {name: 0 for name in owners}
    for polygon in polygons:
        ids, slot = polygon.vertices, polygon.material_index
        if (type(slot) is not int or not 0 <= slot < len(materials) or
                not 3 <= len(ids) <= 64 or len(vertices)+len(ids) > MAX_CORNERS):
            raise ValueError('Invalid or oversized part face indices')
        if any(type(index) is not int or not 0 <= index < len(obj.data.vertices) for index in ids):
            raise ValueError('Part face vertex index is invalid')
        name = material_names[slot]
        owner = owners.get(name, 0)
        if owner:
            material_faces[name] += 1
        vertices.extend(ids)
        face_owners.append(owner)
        sizes.append(len(ids))
    if any(count == 0 for count in material_faces.values()):
        raise ValueError('A requested part material has zero actual face coverage')
    vertex_ids = np.asarray(vertices, dtype=np.int64)
    face_owners = np.asarray(face_owners, dtype=np.int8)
    corner_owners = np.repeat(face_owners, sizes)
    minimum = np.full(len(obj.data.vertices), len(parts)+1, dtype=np.int8)
    maximum = np.full(len(obj.data.vertices), -1, dtype=np.int8)
    np.minimum.at(minimum, vertex_ids, corner_owners)
    np.maximum.at(maximum, vertex_ids, corner_owners)
    shared = (minimum != maximum) & (maximum > 0)
    if np.any(shared):
        raise ValueError(f'Part boundary shares {int(shared.sum())} vertices with another part or outside faces; split the actual mesh first')
    selections = []
    for index, part in enumerate(parts):
        ids = np.flatnonzero(maximum == index+1)
        if not part['min_vertices'] <= len(ids) <= part['max_vertices']:
            raise ValueError(f"Part {part['name']} selected {len(ids)} vertices outside its configured bounds")
        selections.append((ids, {'materials': part['materials'][:],
            'material_slots': [slot for slot, name in enumerate(material_names) if name in part['materials']],
            'faces_by_material': {name: material_faces[name] for name in part['materials']},
            'selected_faces': int(np.count_nonzero(face_owners == index+1))}))
    return selections


def apply_part_rigs(obj, rig, points, low, high, parts, authored_rig):
    """Replace selected vertices' groups; geometry, materials and bone rests stay intact."""
    parts = validate_part_rigs(parts, authored_rig)
    points, low, high = validate_part_points(points, low, high)
    if obj.type != 'MESH' or len(obj.data.vertices) != len(points):
        raise ValueError('Part weighting requires matching mesh and point counts')
    if rig.type != 'ARMATURE':
        raise ValueError('Part weighting requires the actual authored armature')
    available = set(rig.data.bones.keys())
    required = {bone['name'] for part in parts for bone in part_rig_config(part, authored_rig)['bones']}
    if not required <= available:
        raise ValueError('Actual armature is missing configured part bones')
    selections = _select_vertices(obj, parts)
    prepared = []
    # All ownership, counts and field coverage pass before any existing weight is removed.
    for part, (ids, selection) in zip(parts, selections):
        weights, proof = part_skin_weights(points[ids], low, high, part, authored_rig)
        proof.update(selection)
        prepared.append((ids, weights, proof))
    for ids, weights, proof in prepared:
        selected = ids.tolist()
        for group in list(obj.vertex_groups):
            group.remove(selected)
        for column, name in enumerate(proof['bone_names']):
            active = np.flatnonzero(weights[:, column] > 0)
            if not len(active):
                continue
            group = obj.vertex_groups.get(name)
            if group is None:
                group = obj.vertex_groups.new(name=name)
            for local in active:
                group.add([int(ids[local])], float(weights[local, column]), 'REPLACE')
    return {'parts': [proof for _, _, proof in prepared],
            'selected_vertices': sum(len(ids) for ids, _, _ in prepared),
            'max_influences': max(proof['max_influences'] for _, _, proof in prepared),
            'shared_boundary_vertices': 0, 'part_vertices_disjoint': True,
            'geometry_uv_materials_unchanged': True, 'rest_bones_unchanged': True,
            'outside_vertex_weights_unchanged': True, 'authored_not_recovered': True,
            'limits': {'parts': 4, 'vertices': 100000, 'faces': MAX_FACES,
                       'face_corners': MAX_CORNERS, 'influences_per_vertex': 4},
            'scope': 'Exact material-part skin replacement after split-boundary checks; not automatic part segmentation or verified animation'}
