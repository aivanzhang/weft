"""Inspect actual finishing inputs before any fur bake, render or export."""
import json
from pathlib import Path
import sys

import bpy
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE/'sources'))
sys.path.insert(0, str(HERE/'sources/scripts'))
import blender_fur_undercoat as undercoat
from scripts.blender_finish_generated import finish

output = Path(sys.argv[sys.argv.index('--')+1])


class PreflightComplete(Exception):
    pass


def write(name, value):
    (output/name).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+'\n')


def inspect(obj, mask, prepared_flow, settings, out, *, export_vertex_budget):
    mesh = obj.data
    mesh.calc_loop_triangles()
    allocation = undercoat.plan_undercoat_uv_layers([layer.name for layer in mesh.uv_layers],
        mesh.uv_layers.active_index, [layer.active_render for layer in mesh.uv_layers], list(mesh.attributes.keys()))
    vertices = np.asarray([tuple(value.co) for value in mesh.vertices], dtype=np.float32)
    faces = np.asarray([tuple(value.vertices) for value in mesh.loop_triangles], dtype=np.int32)
    loops = np.asarray([tuple(value.loops) for value in mesh.loop_triangles], dtype=np.int32)
    material_ids = np.asarray([value.material_index for value in mesh.loop_triangles], dtype=np.int32)
    uv0 = undercoat._uv_values(mesh, 0, allocation['original_names'][0])[loops]
    uv1 = undercoat._uv_values(mesh, 1, allocation['original_names'][1])[loops]
    selected = mask[faces].all(axis=1)
    frames = []
    for name in allocation['original_names']:
        mesh.calc_tangents(uvmap=name)
        frames.append(np.asarray([(*value.tangent, value.bitangent_sign) for value in mesh.loops], dtype=np.float32))
    basis_rows = []
    for slot, material in enumerate(mesh.materials):
        mapping = [node for node in material.node_tree.nodes if node.type == 'NORMAL_MAP']
        if not mapping or mapping[0].uv_map != allocation['original_names'][1]:
            continue
        ids = loops[material_ids == slot].ravel()
        a, b = frames[0][ids], frames[1][ids]
        usable = (np.linalg.norm(a[:, :3], axis=1) > .99) & (np.linalg.norm(b[:, :3], axis=1) > .99)
        degrees = np.degrees(np.arccos(np.clip(np.sum(a[usable, :3]*b[usable, :3], axis=1), -1, 1)))
        basis_rows.append({'material': material.name, 'corners': len(ids), 'usable_corners': int(usable.sum()),
            'direction_error_degrees_max': float(degrees.max()) if len(degrees) else None,
            'direction_error_degrees_p95': float(np.percentile(degrees, 95)) if len(degrees) else None,
            'handedness_mismatch_corners': int((a[:, 3] != b[:, 3]).sum()),
            'comparison': 'Native Mikk basis from active color UV0 versus actual original-normal UV1; no export yet'})
    write('source-normal-frame-comparison.json', {'schema_version': 1, 'materials': basis_rows,
        'export_orientation_preservation_proven': False, 'quality_accepted': False})
    rows = [{'slot': slot, 'material': material.name,
             'faces': int((material_ids == slot).sum()),
             'selected_faces': int((selected & (material_ids == slot)).sum()),
             'response': undercoat._response_policy(material, settings['roughness'])}
            for slot, material in enumerate(mesh.materials)]
    write('source-selection.json', {'schema_version': 1, 'materials': rows,
        'vertices': len(vertices), 'triangles': len(faces), 'selected_vertices': int(mask.sum()),
        'selected_faces': int(selected.sum()), 'uv_allocation': allocation,
        'export_vertex_budget': export_vertex_budget, 'normal_transfer_requested': False})
    np.savez_compressed(output/'undercoat-inputs.npz', vertices=vertices, faces=faces,
        uv0=uv0, uv1=uv1, selected=selected, material_ids=material_ids,
        vertex_mask=mask, loops=loops, uv0_tangents=frames[0], uv1_tangents=frames[1],
        corner_normals=np.asarray([tuple(value.vector) for value in mesh.corner_normals], dtype=np.float32))
    plan, payloads = undercoat.plan_material_undercoat(vertices, faces, uv0, selected, material_ids,
        settings, export_vertex_budget=export_vertex_budget)
    write('undercoat-plan.json', plan)
    write('result.json', {'schema_version': 1, 'status': 'actual_source_preflight_passed',
        'source_uv_layers': len(mesh.uv_layers), 'authored_uv_index': allocation['authored_index'] if payloads else None,
        'planned_material_atlases': sorted(map(int, payloads)),
        'full_finish_executed': False, 'fur_baked': False, 'model_exported': False,
        'source_normals_retained': True, 'normal_guard_disabled': False, 'quality_accepted': False})
    raise PreflightComplete()


undercoat.apply_undercoat = inspect
try:
    finish(HERE/'finish-config.json', output, prepare_only=True, preserve_skin_source=True)
except PreflightComplete:
    pass
except Exception as error:
    write('result.json', {'schema_version': 1, 'status': 'actual_source_preflight_failed',
        'error': str(error), 'error_type': type(error).__name__, 'full_finish_executed': False,
        'fur_baked': False, 'model_exported': False, 'quality_accepted': False})
    raise
else:
    raise RuntimeError('Expected preflight stop before baking/export')
