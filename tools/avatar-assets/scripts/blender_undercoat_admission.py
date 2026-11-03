"""Inspect preserved rest mesh/UV ownership; never bake, render or save the loaded scene."""

import hashlib
import json
from pathlib import Path
import sys

import numpy as np

if __package__:
    from .blender_fur_undercoat import _export_budget, plan_material_undercoat, plan_undercoat_uv_layers
else:
    # Blender --python does not add the pinned script folder to its import path.
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from blender_fur_undercoat import _export_budget, plan_material_undercoat, plan_undercoat_uv_layers


POLICY = 'saved-undercoat-admission-v1'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def array_hash(value):
    return hashlib.sha256(value.tobytes(order='C')).hexdigest()


def _raw_mesh(mesh):
    nv, ne, nf, nl = len(mesh.vertices), len(mesh.edges), len(mesh.polygons), len(mesh.loops)
    require(1 <= nv <= 100000 and 1 <= ne <= 300000 and 1 <= nf <= 100000 and nl == nf*3,
            'Raw mesh exceeds vertex/edge/triangle bounds')
    points, edges = np.empty((nv, 3), dtype='<f8'), np.empty((ne, 2), dtype='<i4')
    starts, counts = np.empty(nf, dtype='<i4'), np.empty(nf, dtype='<i4')
    loops, materials = np.empty(nl, dtype='<i4'), np.empty(nf, dtype='<i4')
    mesh.vertices.foreach_get('co', points.ravel())
    mesh.edges.foreach_get('vertices', edges.ravel())
    mesh.polygons.foreach_get('loop_start', starts)
    mesh.polygons.foreach_get('loop_total', counts)
    mesh.polygons.foreach_get('material_index', materials)
    mesh.loops.foreach_get('vertex_index', loops)
    require(np.isfinite(points).all() and np.max(np.abs(points)) <= 10000 and
            edges.min() >= 0 and edges.max() < nv and loops.min() >= 0 and loops.max() < nv and
            np.all(counts == 3) and np.array_equal(starts, np.arange(nf)*3),
            'Require finite, bounded, original triangular polygon order')
    return points, edges, loops.reshape(nf, 3), materials


def verified_mask(body, groom, expected):
    """The density group belongs to a proven exact mesh copy, not a nearest-point transfer."""
    body_arrays, groom_arrays = _raw_mesh(body.data), _raw_mesh(groom.data)
    require(all(np.array_equal(a, b) for a, b in zip(body_arrays[:3], groom_arrays[:3])),
            'Native groom coordinates or topology differ from the original body')
    points, edges, faces, _ = body_arrays
    rest = expected['rest']
    require(len(points) == rest['vertices'] and len(edges) == rest['edges'] and
            len(faces) == expected['triangles'] and array_hash(points) == rest['rest_sha256'] and
            array_hash(edges) == rest['edges_sha256'], 'Saved raw body differs from measured rest geometry')
    group = groom.vertex_groups.get(expected['density_group'])
    require(group is not None, 'Recorded density group is absent')
    mask = np.zeros(len(points), dtype=bool)
    for vertex in groom.data.vertices:
        rows = [g for g in vertex.groups if g.group == group.index]
        require(len(rows) <= 1 and all(np.isfinite(g.weight) and g.weight in (0., 1.) for g in rows),
                'Density mask is not an exact binary vertex selection')
        mask[vertex.index] = bool(rows and rows[0].weight == 1.)
    require(int(mask.sum()) == expected['density_selected_vertices'] and
            int((~mask).sum()) == expected['density_excluded_vertices'], 'Saved density count differs from report')
    selected = mask[faces].all(axis=1)
    require(expected['selected_triangles'] is None or int(selected.sum()) == expected['selected_triangles'],
            'Saved selected triangle count differs from full-band report')
    return body_arrays, mask, selected


def material_seam_capacity(vertex_count, faces, selected, material_ids, export_vertex_budget):
    """Same per-material corner reserve as the planner, before any additional atlas splits."""
    require(type(vertex_count) is int and 1 <= vertex_count <= 100000,
            'Invalid capacity vertex count')
    require(isinstance(faces, np.ndarray) and faces.ndim == 2 and faces.shape[1] == 3 and
            1 <= len(faces) <= 100000 and faces.dtype.kind in 'iu', 'Invalid capacity triangle array')
    require(isinstance(selected, np.ndarray) and selected.shape == (len(faces),) and
            selected.dtype.kind == 'b', 'Invalid capacity selection array')
    require(isinstance(material_ids, np.ndarray) and material_ids.shape == (len(faces),) and
            material_ids.dtype.kind in 'iu', 'Invalid capacity material array')
    require(faces.min() >= 0 and faces.max() < vertex_count and material_ids.min() >= 0 and
            material_ids.max() <= 31, 'Capacity indices exceed bounds')
    budget = _export_budget(export_vertex_budget, vertex_count)
    portable = budget['portable_export_vertices']
    require(portable % 12 == 0 and portable//12 <= 5000, 'Capacity requires the unchanged twelve-corner tuft budget')
    rows, total = [], 0
    for slot in np.unique(material_ids[selected]):
        region = material_ids == slot
        yes = np.bincount(faces[selected & region].ravel().astype(np.intp, copy=False), minlength=vertex_count)
        no = np.bincount(faces[~selected & region].ravel().astype(np.intp, copy=False), minlength=vertex_count)
        reserve = int(np.minimum(yes, no).sum())
        total += reserve
        rows.append({'material_id': int(slot), 'selected_triangles': int((selected & region).sum()),
                     'unselected_triangles': int((~selected & region).sum()),
                     'shared_source_vertices': int(((yes > 0) & (no > 0)).sum()),
                     'boundary_corner_reserve': reserve})
    remainder = 100000-budget['source_export_vertices']-budget['reserve_vertices']-total
    return {'policy': 'material-boundary-corner-capacity-v1', 'limit': 100000,
            'source_export_vertices': budget['source_export_vertices'], 'reserve_vertices': budget['reserve_vertices'],
            'current_cards': portable//12, 'exported_corners_per_card': 12, 'profile_card_limit': 5000,
            'current_portable_export_vertices': portable, 'material_boundary_corner_reserve': total,
            'materials': rows, 'vertices_available_for_cards_before_additional_uv_atlas': remainder,
            'maximum_cards_before_additional_uv_atlas': min(5000, remainder//12) if remainder >= 0 else None,
            'current_vertex_ceiling_before_additional_uv_atlas': sum(budget.values())+total,
            'source_reserve_and_seams_fit_without_cards': remainder >= 0,
            'upper_bound_only': True, 'additional_uv_atlas_cost_included': False, 'admission_verified': False,
            'scope': 'Count ceiling only; additional atlas splits can lower it. No masks, counts or planner policy changed.'}


def plan_arrays(points, faces, uv0, selected, material_ids, settings, budget):
    """Planner rejection is evidence; it is not permission to alter masks or capacity."""
    try:
        plan, payloads = plan_material_undercoat(points, faces, uv0, selected, material_ids, settings,
                                                export_vertex_budget=budget)
        return {'status': 'supported', 'reason': 'Measured UV allocation and unchanged export budget are admitted',
                'plan': plan, 'authored_atlas_regions': sorted(int(k) for k in payloads)}
    except (ValueError, RuntimeError) as error:
        return {'status': 'refused', 'reason': str(error), 'exception_type': type(error).__name__,
                'selection_diagnostic': getattr(error, 'diagnostic', None)}


def inspect_scene(bpy, prepared):
    require(not bpy.data.libraries and len(bpy.data.objects) <= 256, 'Linked or oversized native scene unsupported')
    expected = prepared['expected']
    body, groom, rig = (bpy.context.scene.objects.get(expected[key]) for key in ('body', 'groom', 'rig'))
    require(body is not None and groom is not None and rig is not None and
            body.type == groom.type == 'MESH' and rig.type == 'ARMATURE' and len(rig.data.bones) <= 64,
            'Recorded body/groom/rig identity is absent')
    require(groom.get('native_groom') is True and groom.get('native_groom_source') == body.name,
            'Recorded groom source ownership differs')
    require(len(groom.particle_systems) == 1 and
            groom.particle_systems[0].vertex_group_density == expected['density_group'],
            'Actual particle density does not use the recorded mask')
    for obj in (body, groom):
        require(np.allclose(np.asarray(obj.matrix_world), np.eye(4), rtol=0, atol=1e-7),
                'Admission requires the original applied mesh frame')
        skin = [m for m in obj.modifiers if m.type == 'ARMATURE']
        require(len(skin) == 1 and skin[0].object == rig, 'Body/groom armature ownership differs')
    (points, edges, faces, material_ids), mask, selected = verified_mask(body, groom, expected)
    mesh = body.data
    require(1 <= len(mesh.materials) <= 32 and material_ids.min() >= 0 and material_ids.max() < len(mesh.materials),
            'Body material ownership exceeds bounds')
    materials = []
    for slot in np.unique(material_ids):
        material = mesh.materials[int(slot)]
        require(material is not None, 'Referenced material is absent')
        region = material_ids == slot
        materials.append({'slot': int(slot), 'name': material.name, 'triangles': int(region.sum()),
                          'selected_triangles': int((region & selected).sum())})
    counts = {'vertices': len(points), 'edges': len(edges), 'triangles': len(faces),
              'selected_vertices': int(mask.sum()), 'excluded_vertices': int((~mask).sum()),
              'selected_triangles': int(selected.sum()), 'excluded_triangles': int((~selected).sum())}
    report = {'policy': POLICY, 'settings': prepared['settings'], 'counts': counts, 'materials': materials,
              'export_vertex_budget': prepared['export_vertex_budget'],
              'material_seam_capacity': material_seam_capacity(len(points), faces, selected, material_ids,
                                                               prepared['export_vertex_budget']),
              'geometry': {'rest_sha256': array_hash(points), 'edges_sha256': array_hash(edges),
                           'triangles_sha256': array_hash(faces), 'mask_sha256': array_hash(mask.astype('u1'))},
              'input_writes': False, 'baked': False, 'rendered': False, 'saved_scene': False,
              'quality_approved': False, 'scope': prepared['scope'],
              'not_verified': ['baked pixels', 'tangent frames', 'shader compatibility', 'visible fur', 'animation quality']}
    try:
        allocation = plan_undercoat_uv_layers([layer.name for layer in mesh.uv_layers], mesh.uv_layers.active_index,
            [bool(layer.active_render) for layer in mesh.uv_layers], list(mesh.attributes.keys()))
        layers, uv0 = [], None
        for index, layer in enumerate(mesh.uv_layers):
            require(len(layer.data) == len(mesh.loops), 'UV layer loop count differs')
            values = np.empty((len(mesh.loops), 2), dtype='<f4')
            layer.data.foreach_get('uv', values.ravel())
            require(np.isfinite(values).all() and np.max(np.abs(values)) <= 1e6, 'Original UV is nonfinite or oversized')
            layers.append({'index': index, 'name': layer.name, 'active_render': bool(layer.active_render),
                           'sha256': array_hash(values), 'min': values.min(axis=0).tolist(), 'max': values.max(axis=0).tolist()})
            if index == 0:
                uv0 = values.reshape(len(faces), 3, 2)
        report.update(original_uv_layers=layers, uv_layer_allocation=allocation)
        report.update(plan_arrays(points.astype('<f4'), faces, uv0, selected, material_ids,
                                  prepared['settings'], prepared['export_vertex_budget']))
    except (ValueError, RuntimeError) as error:
        report.update(status='refused', reason=str(error), exception_type=type(error).__name__)
    return report


def main():
    import bpy
    args = sys.argv[sys.argv.index('--')+1:]
    require(len(args) == 2, 'Expected prepared input and new result paths')
    path, target = map(Path, args)
    require(path.is_file() and not path.is_symlink() and path.stat().st_size <= 1024*1024 and
            target.parent.is_dir() and not target.exists() and not target.is_symlink(), 'Invalid diagnostic paths')
    prepared = json.loads(path.read_text())
    require(prepared['policy'] == POLICY, 'Unexpected admission policy')
    blend = prepared['inputs']['source.blend']
    source = Path(blend['path'])
    require(source.is_file() and not source.is_symlink() and source.stat().st_size == blend['bytes'] <= 512*1024*1024
            and digest(source) == blend['sha256'], 'Preserved native scene changed')
    bpy.context.preferences.filepaths.use_scripts_auto_execute = False
    bpy.ops.wm.open_mainfile(filepath=str(source), load_ui=False, use_scripts=False)
    report = inspect_scene(bpy, prepared)
    require(digest(source) == blend['sha256'], 'Diagnostic changed preserved scene')
    report.update(inputs_sha256=digest(path), blender_version=bpy.app.version_string)
    data = json.dumps(report, indent=2, sort_keys=True, allow_nan=False)+'\n'
    require(len(data.encode()) <= 1024*1024, 'Admission report exceeds its bound')
    with target.open('x') as handle:
        handle.write(data)
    print(report['status']+': '+report['reason'])


if __name__ == '__main__':
    main()
