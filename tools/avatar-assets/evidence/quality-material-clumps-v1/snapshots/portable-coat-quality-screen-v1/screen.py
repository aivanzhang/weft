"""Pinned actual-coat measurements and a no-mesh source-face availability screen."""
import hashlib
import itertools
import json
from pathlib import Path
import struct
import sys
import time
import zipfile

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE/'sources/scripts'))
from fringe_color_atlas import sample_alpha

START = time.monotonic()
PREP = json.loads((HERE/'prepared.json').read_text())
PATHS = {key: ROOT/row['path'] for key, row in PREP['inputs'].items()}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def deadline():
    require(time.monotonic()-START < 27, '27-second internal deadline exceeded')


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def load_npz(path):
    with zipfile.ZipFile(path) as archive:
        require(len(archive.infolist()) <= 16 and sum(row.file_size for row in archive.infolist()) <= 8*1024**2,
                'NPZ entry/decoded byte bound exceeded')
    with np.load(path, allow_pickle=False) as archive:
        return {name: archive[name] for name in archive.files}


def unit(value):
    lengths = np.linalg.norm(value, axis=-1, keepdims=True)
    require(np.all(lengths > 1e-12), 'Degenerate vector')
    return value/lengths


def stats(value):
    value = np.asarray(value, dtype=float)
    require(value.size and np.isfinite(value).all(), 'Invalid statistic')
    return dict(zip(('min', 'p10', 'median', 'p90', 'max'), map(float, np.quantile(value, [0, .1, .5, .9, 1]))))


require(all(sha(PATHS[key]) == row['sha256'] for key, row in PREP['inputs'].items()), 'Input hash changed')
fringe = json.loads(PATHS['fringe_report'].read_text())
inspection = json.loads(PATHS['inspection'].read_text())
skin_info = json.loads(PATHS['skin_report'].read_text())
fins, skin = load_npz(PATHS['fringe_npz']), load_npz(PATHS['skin_npz'])
require(inspection['source_sha256'] == PREP['inputs']['model']['sha256'], 'Preview source differs')
p = fins['positions'].astype(float).reshape(-1, 4, 3)
normal = unit(fins['normals'].astype(float).reshape(-1, 4, 3)[:, :2].sum(axis=1))
uv = fins['uv_gltf'].astype(float).reshape(-1, 4, 2)
count = len(p)
require(count == fringe['cards'] == 3600, 'Unexpected actual root count')
edge, lift = p[:, 1]-p[:, 0], p[:, 2]-p[:, 0]
span, length = np.linalg.norm(edge, axis=1), np.linalg.norm(lift, axis=1)
quad_cross = np.cross(edge, lift)
quad_area = np.linalg.norm(quad_cross, axis=1)
plane_normal = unit(quad_cross)
outward_dot = np.sum(unit(lift)*normal, axis=1)
with Image.open(PATHS['atlas']) as image:
    require(image.format == 'PNG' and image.size == (4096, 1824) and image.mode == 'RGBA', 'Unexpected atlas format')
    alpha = np.asarray(image.getchannel('A')).copy()
require(sha(PATHS['atlas']) == fringe['material']['sha256'], 'Atlas report hash mismatch')
cells = np.empty((count, 28, 60), dtype=np.uint8)
for index, destination in enumerate(fins['destination_for_source_fin']):
    x, y = int(destination % 64)*64+2, int(destination//64)*32+2
    cells[index] = alpha[y:y+28, x:x+60]
mask = cells >= 46
occupancy = mask.mean(axis=(1, 2))
width = mask[:, 14].sum(axis=1)/60*span/2
fraction = .5+np.array([-.0025, .0025])/span[:, None]
tips = uv[:, None, 0]+fraction[..., None]*(uv[:, None, 1]-uv[:, None, 0])
tips[..., 1] = ((uv[:, 0, 1]+uv[:, 2, 1])*.5)[:, None]
mips = []
layer = Image.fromarray(alpha)
for level in range(7):
    values = sample_alpha(np.asarray(layer), tips.reshape(-1, 2))
    mips.append({'level': level, 'dimensions': list(layer.size), 'nominal_tips': len(values),
                 'tips_at_or_above_cutoff': int(np.count_nonzero(values >= .18)),
                 'alpha': stats(values), 'scope': 'BOX mip alpha + bilinear nominal sampling, not measured GPU LOD'})
    layer = layer.resize((max(1, layer.width//2), max(1, layer.height//2)), Image.Resampling.BOX)
deadline()

owner = skin_info['owners'][0]
require(owner['vertex_start'] == owner['triangle_start'] == 0, 'Unexpected native body ownership')
body_points = skin['positions'][:owner['vertex_count']].astype(float)
body_faces = skin['triangles'][:owner['triangle_count']].astype(np.int32)
require(np.array_equal(body_points[fins['root_vertex_ids']], p[:, :2]), 'Root native source IDs differ')
require(np.array_equal(body_faces[fins['root_face_ids'][:, None], fins['root_corner_ids']], fins['root_vertex_ids']),
        'Root source face/corner provenance differs')
all_edges = np.sort(body_faces[:, [[0, 1], [1, 2], [2, 0]]].reshape(-1, 2), axis=1)
face_ids = np.repeat(np.arange(len(body_faces)), 3)
key = all_edges[:, 0].astype(np.int64)*len(body_points)+all_edges[:, 1]
order = np.argsort(key, kind='stable'); keys = key[order]
root_edges = np.sort(fins['root_vertex_ids'], axis=1)
root_keys = root_edges[:, 0].astype(np.int64)*len(body_points)+root_edges[:, 1]
left, right = np.searchsorted(keys, root_keys, side='left'), np.searchsorted(keys, root_keys, side='right')
require(np.all(right-left == 2), 'Retained root lacks exactly two indexed source faces')
pairs = np.stack((face_ids[order[left]], face_ids[order[left+1]]), axis=1)
tri = body_points[body_faces]
face_cross = np.cross(tri[:, 1]-tri[:, 0], tri[:, 2]-tri[:, 0])
face_area = .5*np.linalg.norm(face_cross, axis=1)
same_material = skin['triangle_materials'][pairs[:, 0]] == skin['triangle_materials'][pairs[:, 1]]
require(same_material.all(), 'Paired source faces cross a current material boundary')
pair_area = face_area[pairs].sum(axis=1)
chosen, used_faces = [], set()
for index in np.lexsort((np.arange(count), -pair_area)):
    if not used_faces.intersection(map(int, pairs[index])):
        chosen.append(int(index)); used_faces.update(map(int, pairs[index]))
    if len(chosen) == 2700:
        break
chosen = np.asarray(chosen, dtype=np.int32)
contour = np.setdiff1d(np.arange(count), chosen)
patch_faces = pairs[chosen].ravel()
require(len(np.unique(patch_faces)) == len(patch_faces), 'Patch selection reused source faces')
selectors = {'surface_fin_ids': chosen.tolist(), 'contour_fin_ids': contour.tolist(),
             'surface_source_face_pairs': pairs[chosen].tolist(),
             'selection_policy': 'Largest actual paired-source area first; stable fin-ID tie break; no shared source faces',
             'not_a_mesh': True, 'source_model_sha256': PREP['inputs']['model']['sha256']}
with (HERE/'selection-screen.json').open('x') as stream:
    json.dump(selectors, stream, indent=2, sort_keys=True); stream.write('\n')
boxes = []
for row in skin_info['owners']:
    points = skin['positions'][row['vertex_start']:row['vertex_start']+row['vertex_count']]
    boxes.extend(itertools.product(*zip(points.min(axis=0), points.max(axis=0))))
boxes = np.asarray(boxes)
require(np.max(abs(boxes.min(axis=0)-inspection['bounds_blender_z_up']['min'])) < 2e-7
        and np.max(abs(boxes.max(axis=0)-inspection['bounds_blender_z_up']['max'])) < 2e-7,
        'Measured native rest boxes differ from imported preview bounds')
views = []
for name, offset in (('front', [0, -3, .05]), ('side', [3, 0, .05])):
    back = unit(np.asarray(offset, dtype=float))
    right_axis = unit(np.cross([0., 0, 1.], back)); up = unit(np.cross(back, right_axis))
    basis = np.column_stack((right_axis, up, back))
    scale = float(np.ptp(boxes@basis, axis=0)[:2].max()*1.18)
    edge_screen = np.linalg.norm((edge@basis)[:, :2], axis=1)
    lift_screen = np.linalg.norm((lift@basis)[:, :2], axis=1)
    projected = np.abs(quad_cross@back)*occupancy
    face_on = normal@back > .7
    contour_on = np.abs(normal@back) < .3
    patch_projected = np.abs(face_cross[patch_faces]@back)*.5
    views.append({'view': name, 'orthographic_scale_m': scale,
        'front_facing_surface_roots': int(face_on.sum()), 'contour_oriented_roots': int(contour_on.sum()),
        'plane_facing_camera_abs_dot_for_front_surface': stats(np.abs(plane_normal[face_on]@back)),
        'summed_current_alpha_area_m2_no_occlusion': float(projected.sum()),
        'summed_surface_pair_area_m2_no_occlusion': float(patch_projected.sum()),
        'hypothetical_mixed_alpha_area_m2_at_045_surface_fraction': float(patch_projected.sum()*.45+projected[contour].sum()),
        'hypothetical_mixed_alpha_area_ratio': float((patch_projected.sum()*.45+projected[contour].sum())/projected.sum()),
        'display': [{'pixels': size, 'pixels_per_m': size/scale,
            'half_height_stroke_width_px': stats(width*edge_screen/span*size/scale),
            'lift_height_px': stats(lift_screen*size/scale),
            'estimated_width_minification_lod': stats(np.log2(60/np.maximum(edge_screen*size/scale, 1e-6))),
            'estimated_height_minification_lod': stats(np.log2(28/np.maximum(lift_screen*size/scale, 1e-6)))} for size in (720, 256)]})
deadline()
with PATHS['model'].open('rb') as stream:
    header = stream.read(20); length_json = struct.unpack_from('<I', header, 12)[0]
    require(length_json < 2*1024**2, 'GLB JSON exceeds bound')
    doc = json.loads(stream.read(length_json))
counts = []
for mesh in doc['meshes']:
    counts.append({'vertices': sum(doc['accessors'][prim['attributes']['POSITION']]['count'] for prim in mesh['primitives']),
                   'triangles': sum(doc['accessors'][prim['indices']]['count']//3 for prim in mesh['primitives'])})
result = {'schema_version': 1, 'status': 'measured_not_quality_approved',
    'current': {'roots': count, 'nominal_strands': 2*count, 'eligible_source_area_m2': fringe['geometry']['allocation']['represented_face_area_m2'],
        'span_m': stats(span), 'lift_m': stats(length), 'outward_angle_degrees': stats(np.degrees(np.arccos(np.clip(outward_dot, -1, 1)))),
        'alpha_occupancy_above_018': stats(occupancy), 'half_height_stroke_width_m': stats(width),
        'total_unoccluded_alpha_card_area_m2': float(np.sum(quad_area*occupancy)),
        'atlas_mean_alpha_per_card': stats(cells.mean(axis=(1, 2))/255),
        'atlas_dimensions': [4096, 1824], 'atlas_pixels': int(alpha.size), 'actual_exported_mesh_counts': counts},
    'mip_screen': mips, 'views': views,
    'source_pair_screen': {'roots_with_two_exact_incident_faces': len(pairs), 'same_material_pairs': int(same_material.sum()),
        'selected_nonoverlapping_surface_roots': len(chosen), 'retained_contour_roots': len(contour),
        'selected_unique_source_faces': len(patch_faces), 'actual_source_patch_area_m2': float(face_area[patch_faces].sum()),
        'source_area_fraction': float(face_area[patch_faces].sum()/fringe['geometry']['allocation']['represented_face_area_m2']),
        'candidate_gate': 'pair/topology/data availability only; no source-face mask recertification, offset intersections, skin-pose or appearance proof',
        'proposed_exported_card_corners': int(len(chosen)*6+len(contour)*12),
        'proposed_card_triangles': int(len(chosen)*2+len(contour)*4),
        'projected_vertices_using_actual_base_plus_4096_reserve': int(counts[0]['vertices']+len(chosen)*6+len(contour)*12+4096),
        'projected_triangles': int(counts[0]['triangles']+len(chosen)*2+len(contour)*4)},
    'elapsed_seconds': time.monotonic()-START,
    'limitations': ['Summed projection counts overlapping/hidden cards; it is not visible coverage.',
        'Modeled BOX alpha mips are not measured EEVEE texture derivatives or temporal filtering.',
        'The0.45 surface alpha fraction is a proposed target, not a generated or accepted atlas.',
        'Current normal-map undercoat is surface shading, not dense geometric fur.',
        'No model or native render was produced; source contact folds are not repaired.'],
    'inputs_unchanged': all(sha(PATHS[key]) == row['sha256'] for key, row in PREP['inputs'].items()),
    'native_provider_ai_role_calls': 0, 'quality_approved': False}
with (HERE/'measurements.json').open('x') as stream:
    json.dump(result, stream, indent=2, sort_keys=True); stream.write('\n')
print(json.dumps({'elapsed_seconds': result['elapsed_seconds'], 'current': result['current'],
                  'source_pair_screen': result['source_pair_screen'], 'views': views}, indent=2))
