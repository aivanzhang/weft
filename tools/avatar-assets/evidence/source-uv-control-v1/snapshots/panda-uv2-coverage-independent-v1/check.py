"""Read-only actual-preflight coverage proof; no bake, export or appearance acceptance."""
import hashlib
import json
from pathlib import Path
import sys
import zipfile

import numpy as np

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
sys.path.insert(0, str(ROOT))
from avatar_assets.normalize_generated import _array
from avatar_assets.three_d import _GLB


def require(value, message):
    if not value:
        raise ValueError(message)


def digest(path):
    result = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024*1024), b''):
            result.update(chunk)
    return result.hexdigest()


def write(name, value):
    text = json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+'\n'
    with (OUT/name).open('x') as handle:
        handle.write(text)


pins = json.loads((OUT/'inputs.json').read_text())
for row in pins['files']:
    require(digest(ROOT/row['path']) == row['sha256'], 'Pinned input changed: '+row['path'])
preflight = ROOT/'outputs/panda-uv2-source-preflight-v1'
preparation = ROOT/'outputs/panda-uv2-source-preflight-preparation-v1'
manifest = json.loads((preflight/'manifest.json').read_text())
require(manifest['job']['exit_code'] == 0 and manifest['job']['process_exit_verified']
        and manifest['job']['failure'] is None, 'Native preflight did not exit successfully')
require(digest(preparation/'prepared.json') == manifest['prepared_sha256'], 'Preparation binding differs')
selection = json.loads((preflight/'source-selection.json').read_text())
plan = json.loads((preflight/'undercoat-plan.json').read_text())
config = json.loads((preparation/'finish-config.json').read_text())
require('normal_continuity' not in config and selection['normal_transfer_requested'] is False,
        'This is not the declared source-normal control')
for name in ('undercoat-inputs.npz', 'source-selection.json', 'undercoat-plan.json',
             'source-normal-frame-comparison.json', 'result.json'):
    rows = [row for row in manifest['artifacts'] if row['path'] == name]
    require(len(rows) == 1 and rows[0]['sha256'] == digest(preflight/name)
            and rows[0]['bytes'] == (preflight/name).stat().st_size, 'Native artifact binding differs')

path = preflight/'undercoat-inputs.npz'
with zipfile.ZipFile(path) as archive:
    entries = archive.infolist()
    require(len(entries) == 11 and sum(row.file_size for row in entries) <= 16*1024**2,
            'Saved array decode exceeds the declared bound')
with np.load(path, allow_pickle=False) as source:
    data = {name: source[name] for name in source.files}
shapes = {'vertices': ((51376, 3), '<f4'), 'faces': ((62581, 3), '<i4'),
          'uv0': ((62581, 3, 2), '<f4'), 'uv1': ((62581, 3, 2), '<f4'),
          'selected': ((62581,), '|b1'), 'material_ids': ((62581,), '<i4'),
          'vertex_mask': ((51376,), '|b1'), 'loops': ((62581, 3), '<i4'),
          'uv0_tangents': ((187743, 4), '<f4'), 'uv1_tangents': ((187743, 4), '<f4'),
          'corner_normals': ((187743, 3), '<f4')}
require(set(data) == set(shapes), 'Unexpected saved array inventory')
for name, (shape, dtype) in shapes.items():
    require(data[name].shape == shape and data[name].dtype.str == dtype, 'Saved array format differs: '+name)
    require(np.isfinite(data[name]).all(), 'Nonfinite saved values: '+name)
faces, slots, selected = data['faces'], data['material_ids'], data['selected']
require(faces.min() >= 0 and faces.max() < len(data['vertices']), 'Native faces exceed source vertex range')
require(np.array_equal(np.sort(data['loops'].ravel()), np.arange(187743)), 'Native loop coverage differs')
require(np.array_equal(selected, data['vertex_mask'][faces].all(axis=1)), 'Face selection differs from actual vertex mask')
require(int(selected.sum()) == selection['selected_faces'] == plan['selected_triangles'] == 31722,
        'Selected face count differs')
require(int(data['vertex_mask'].sum()) == selection['selected_vertices'] == 23385, 'Selected vertex count differs')

glb = _GLB(Path(config['source']))
require(glb.hash == config['source_sha256'] == '736ad72d408a43295098f44a29919c6436b37f741b7335f35a2c3f5f2007fb3e',
        'Control source is not the admitted UV repair')
primitives = glb.doc['meshes'][0]['primitives']
native_offset, expected_faces, source_points, primitive_data = 0, [], [], {}
for index, primitive in enumerate(primitives):
    attrs = primitive['attributes']
    points = _array(glb, attrs['POSITION'], 3, 5126, 100000)
    index_spec = glb.doc['accessors'][primitive['indices']]
    indices = _array(glb, primitive['indices'], 1, index_spec['componentType'], 300000).reshape(-1, 3)
    require(np.array_equal(np.unique(indices), np.arange(len(points))), 'Source primitive has unreferenced vertices')
    expected_faces.append(indices+native_offset)
    blender_points = points[:, [0, 2, 1]].copy(); blender_points[:, 1] *= -1
    source_points.append(blender_points)
    primitive_data[index] = (native_offset, indices, points)
    native_offset += len(points)
require(np.array_equal(faces, np.concatenate(expected_faces)), 'Native source face identity/order differs')
all_points = np.concatenate(source_points).astype(np.float64)
low, high = all_points.min(axis=0), all_points.max(axis=0)
offset = np.array([(low[0]+high[0])/2, (low[1]+high[1])/2, low[2]])
factor = config['authored_rig']['height_meters']/(high[2]-low[2])
expected_points = ((all_points-offset)*factor).astype(np.float32)
max_position_error = float(np.max(np.abs(expected_points-data['vertices'])))
require(max_position_error <= 1e-6, 'Native source positions do not match the declared authored scale')

rows = []
source_nondefault_normal_materials = set()
for material in glb.doc['materials']:
    if material.get('normalTexture', {}).get('texCoord', 0) != 0:
        source_nondefault_normal_materials.add(material['name'])
require(source_nondefault_normal_materials == {'body_surface__contact_uv_dark_v1'},
        'Unexpected original nondefault normal route')
for row in selection['materials']:
    slot = row['slot']; region = slots == slot
    require(int(region.sum()) == row['faces'] and int((region & selected).sum()) == row['selected_faces'],
            'Saved material selection count differs')
    if slot not in (4, 5):
        continue
    primitive = primitives[slot]
    require(glb.doc['materials'][primitive['material']]['name'] == row['material'], 'Contact material identity differs')
    expected_ids = primitive_data[slot][1]+primitive_data[slot][0]
    require(np.array_equal(faces[region], expected_ids), 'Contact source triangles do not match native region')
    require(bool(selected[region].all()), 'An original contact face escapes undercoat replacement')
    for channel in (0, 1):
        uv = _array(glb, primitive['attributes']['TEXCOORD_'+str(channel)], 2, 5126, 100000)
        uv = uv[primitive_data[slot][1]].copy(); uv[:, :, 1] = 1-uv[:, :, 1]
        require(np.array_equal(uv, data['uv'+str(channel)][region]), 'Original contact UV changed before bake')
    rows.append({'source_material': row['material'], 'slot': slot, 'faces': int(region.sum()),
                 'selected_faces': int((region & selected).sum()), 'unselected_faces': 0,
                 'normal_uv1_requires_replacement': row['material'] in source_nondefault_normal_materials})

independent_extra = 0
for row in plan['regions']:
    slot = row['region_id']; region = slots == slot; ids = np.flatnonzero(selected & region).astype('<i4')
    require(row['selected_faces'] == len(ids) and row['selection_sha256'] == hashlib.sha256(ids.tobytes()).hexdigest(),
            'Plan is not bound to the exact saved selected faces')
    yes = np.bincount(faces[selected & region].ravel(), minlength=len(data['vertices']))
    no = np.bincount(faces[~selected & region].ravel(), minlength=len(data['vertices']))
    bound = int(np.minimum(yes, no).sum())
    require(bound == row['material_split_corner_bound'], 'Material split corner bound differs')
    independent_extra += bound+row['uv1_extra_vertices']
require(plan['atlas_regions'] == [2, 4, 5] and selection['uv_allocation']['authored_index'] == 2,
        'Contact regions are not assigned the new UV2 atlas')
require(all(row['method'] == ('keep_uv0' if row['region_id'] == 0 else 'authored_uv1')
            for row in plan['regions']), 'Unexpected allocation method')
budget = selection['export_vertex_budget']
require(budget == plan['export_vertex_budget'] == {'source_export_vertices': 51376,
        'portable_export_vertices': 43200, 'reserve_vertices': 4096}, 'Aggregate input budget differs')
ceiling = sum(budget.values())+independent_extra
require(independent_extra == plan['additional_export_vertices'] == 482
        and ceiling == plan['projected_export_vertex_ceiling'] == 99154 and ceiling <= 100000,
        'Projected export ceiling differs or exceeds bound')
frames = data['loops'][slots == 4].ravel()
a, b = data['uv0_tangents'][frames], data['uv1_tangents'][frames]
degrees = np.degrees(np.arccos(np.clip(np.sum(a[:, :3]*b[:, :3], axis=1), -1, 1)))
for row in pins['files']:
    require(digest(ROOT/row['path']) == row['sha256'], 'Input changed during audit: '+row['path'])
write('result.json', {'schema_version': 1, 'status': 'actual_preflight_coverage_verified',
    'contact_regions': rows, 'source_triangle_identity_verified': True,
    'source_uv0_uv1_exact_before_bake': True, 'position_max_error_m': max_position_error,
    'selected_faces_verified': 31722, 'projected_export_vertex_ceiling': ceiling,
    'source_uv1_vs_uv0_frame_hazard': {'corners': len(frames),
        'handedness_mismatches': int((a[:, 3] != b[:, 3]).sum()),
        'direction_error_p95_degrees': float(np.percentile(degrees, 95))},
    'conditional_export_route': 'Every original UV1 normal face is selected for intentional replacement by a newly baked UV2 normal. Require zero used nonauthored nondefault-normal-UV primitive bindings in the actual export.',
    'unused_material_definitions_are_not_used_primitives': True,
    'normal_maps_intentionally_replaced_on_selected_faces': True,
    'original_geometric_normal_transfer_omitted': True,
    'source_frame_preservation_is_not_claimed_for_replaced_normals': True,
    'actual_bake_export_counts_and_target_hashes_still_required': True,
    'full_finish_executed_by_this_check': False, 'public_fringe_enabled': False,
    'quality_accepted': False, 'input_hashes_unchanged': True})
