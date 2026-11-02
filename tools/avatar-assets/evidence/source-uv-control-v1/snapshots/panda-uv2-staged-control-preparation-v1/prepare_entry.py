"""Experimental real-source control; prove old UV1 normal routes are wholly replaced."""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE/'sources'))
sys.path.insert(0, str(HERE/'sources/scripts'))
import blender_fur_undercoat as undercoat
from scripts.blender_finish_generated import finish
from avatar_assets.three_d import _GLB

out = Path(sys.argv[sys.argv.index('--')+1])
saved = json.loads((HERE/'source-selection.json').read_text())
apply = undercoat.apply_undercoat


def write(name, value):
    (out/name).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+'\n')


def normal_uv(mesh, material):
    shader = undercoat._material_shader(material)
    links = shader.inputs['Normal'].links
    if not links:
        return None
    if len(links) != 1 or links[0].from_node.type != 'NORMAL_MAP':
        raise ValueError('Control cannot verify an indirect normal graph')
    name = links[0].from_node.uv_map
    index = mesh.uv_layers.find(name) if name else mesh.uv_layers.active_index
    if index < 0:
        raise ValueError('Control normal route has no UV layer')
    return index


def checked_undercoat(obj, mask, prepared, settings, directory, *, export_vertex_budget):
    mesh = obj.data
    mesh.calc_loop_triangles()
    selected = np.array([bool(mask[list(face.vertices)].all()) for face in mesh.loop_triangles])
    material_ids = np.array([face.material_index for face in mesh.loop_triangles])
    before = []
    for slot, material in enumerate(mesh.materials):
        owned = np.flatnonzero(material_ids == slot)
        expected = next(row for row in saved['materials'] if row['slot'] == slot)
        if (expected['material'] != material.name or len(owned) != expected['faces']
                or int(selected[owned].sum()) != expected['selected_faces']):
            raise ValueError('Actual source mask differs from the frozen native preflight')
        if len(owned) and normal_uv(mesh, material) not in (None, 0):
            if not selected[owned].all():
                raise ValueError('Unselected original UV1 normals need a separate frame-preservation implementation')
            before.append({'source_material': material.name, 'faces': len(owned),
                'face_ids_sha256': hashlib.sha256(owned.astype('<i4').tobytes()).hexdigest()})
    report = apply(obj, mask, prepared, settings, directory, export_vertex_budget=export_vertex_budget)
    for original in before:
        matches = [row for row in report['materials'] if row['source_material'] == original['source_material']]
        if len(matches) != 1 or matches[0]['selected_faces'] != original['faces'] or matches[0]['normal_texcoord'] != 2:
            raise ValueError('Original UV1 normal faces were not wholly assigned to the new UV2 atlas')
    used = {face.material_index for face in mesh.polygons}
    bindings = [{'material': mesh.materials[index].name, 'normal_uv': normal_uv(mesh, mesh.materials[index])}
                for index in sorted(used)]
    if any(row['normal_uv'] == 1 for row in bindings):
        raise ValueError('A used source normal UV1 route remains before export')
    write('original-normal-coverage.json', {'schema_version': 1, 'original_uv1_materials': before,
        'used_material_bindings_after_bake': bindings, 'all_original_uv1_normal_faces_intentionally_replaced': True,
        'general_source_uv1_frame_preservation': False, 'quality_accepted': False})
    return report


undercoat.apply_undercoat = checked_undercoat
finish(HERE/'finish-config.json', out, prepare_only=True, preserve_skin_source=True)
native = _GLB(out/'model.glb')
bindings = []
for mesh_index, mesh in enumerate(native.doc['meshes']):
    for primitive_index, primitive in enumerate(mesh['primitives']):
        material = native.doc['materials'][primitive['material']]
        normal = material.get('normalTexture')
        uv = normal.get('texCoord', 0) if normal is not None else None
        if uv == 1:
            raise ValueError('Raw export still uses an original UV1 normal map; source frames are not proved')
        bindings.append({'mesh': mesh_index, 'primitive': primitive_index, 'material': material['name'],
                         'normal_uv': uv, 'vertices': native.doc['accessors'][primitive['attributes']['POSITION']]['count']})
write('raw-used-normal-uv.json', {'schema_version': 1, 'raw_sha256': native.hash, 'used_bindings': bindings,
    'used_source_normal_uv1_primitives': 0, 'intentional_undercoat_replacement_not_source_normal_map_preservation': True,
    'source_geometric_normals_retained': True, 'rendered_images': 0, 'quality_accepted': False})
