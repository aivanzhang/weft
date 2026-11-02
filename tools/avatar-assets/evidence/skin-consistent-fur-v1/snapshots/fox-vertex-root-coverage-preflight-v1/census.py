"""Read-only material root capacity, without implementing the proposed selector."""
import hashlib
import json
from pathlib import Path
import resource
import struct
import sys
import time
import zipfile

import numpy as np

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
START = time.monotonic()


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(value):
    return hashlib.sha256(value).hexdigest()


def array_sha(value):
    return sha(np.ascontiguousarray(value).tobytes())


def save(name, value):
    with (OUT/name).open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')


def quantiles(values, weights=None):
    values = np.asarray(values, float)
    if not len(values):
        return None
    qs = [0, .1, .5, .9, .99, 1]
    if weights is None:
        result = np.quantile(values, qs)
    else:
        order = np.argsort(values, kind='stable')
        cumulative = np.cumsum(np.asarray(weights, float)[order])
        result = values[order[np.minimum(np.searchsorted(cumulative, np.asarray(qs)*cumulative[-1]), len(order)-1)]]
    return dict(zip(['min', 'p10', 'p50', 'p90', 'p99', 'max'], map(float, result)))


def glb(path):
    raw = path.read_bytes()
    require(28 <= len(raw) <= 64*1024**2, 'GLB exceeds bound')
    magic, version, length, jsize, kind = struct.unpack_from('<4sIIII', raw)
    require((magic, version, length, kind) == (b'glTF', 2, len(raw), 0x4e4f534a)
            and 0 < jsize <= 2*1024**2, 'Invalid GLB header')
    size, kind = struct.unpack_from('<II', raw, 20+jsize)
    require(kind == 0x004e4942 and size+28+jsize == len(raw), 'Invalid BIN')
    return json.loads(raw[20:20+jsize]), raw[28+jsize:]


def read_array(doc, binary, index):
    a = doc['accessors'][index]
    require(not a.get('sparse') and not a.get('normalized'), 'Unsupported accessor')
    widths = {'SCALAR': 1, 'VEC2': 2, 'VEC3': 3, 'VEC4': 4}
    dtypes = {5121: '<u1', 5123: '<u2', 5125: '<u4', 5126: '<f4'}
    width, dtype = widths[a['type']], np.dtype(dtypes[a['componentType']])
    count = a['count']
    require(type(count) is int and 0 < count <= 300000, 'Accessor count outside bound')
    v = doc['bufferViews'][a['bufferView']]
    offset, stride = v.get('byteOffset', 0)+a.get('byteOffset', 0), v.get('byteStride', width*dtype.itemsize)
    require(v['buffer'] == 0 and stride >= width*dtype.itemsize and
            a.get('byteOffset', 0)+(count-1)*stride+width*dtype.itemsize <= v['byteLength'] and
            v.get('byteOffset', 0)+v['byteLength'] <= len(binary), 'Accessor range invalid')
    result = np.ndarray((count, width), dtype=dtype, buffer=binary,
                        offset=offset, strides=(stride, dtype.itemsize)).copy()
    require(np.isfinite(result).all(), 'Nonfinite source values')
    return result


def components(faces, vertex_count):
    parent = np.arange(vertex_count)

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b, c in faces:
        a, b, c = find(a), find(b), find(c)
        parent[b], parent[c] = a, a
    return np.asarray([find(x) for x in range(vertex_count)])


def main():
    prepared = json.loads((OUT/'prepared.json').read_text())
    for item in prepared['inputs']:
        p = ROOT/item['path']
        require(p.is_file() and not p.is_symlink() and p.stat().st_size == item['bytes']
                and sha(p.read_bytes()) == item['sha256'], 'Input changed: '+item['path'])
    folder = ROOT/'outputs/fox-material-fur-finish-v1'
    cfg = json.loads((folder/'finish-config.json').read_text())
    groom = json.loads((folder/'material-groom/report.json').read_text())
    coat = json.loads((folder/'material-coat.json').read_text())
    doc, binary = glb(Path(cfg['source']))
    require(len(doc['meshes']) == len(doc['skins']) == 1, 'Require the single preserved source mesh/skin')
    # IDs are admitted only if reconstructed arrays match the sealed native signature.
    points, faces, slots, uvs, normals, joints, weights = [], [], [], [], [], [], []
    vstart, fstart, primitive_rows = 0, 0, []
    for pi, primitive in enumerate(doc['meshes'][0]['primitives']):
        attrs = primitive['attributes']
        p = read_array(doc, binary, attrs['POSITION'])
        n = read_array(doc, binary, attrs['NORMAL'])
        uv = read_array(doc, binary, attrs['TEXCOORD_0'])
        j = read_array(doc, binary, attrs['JOINTS_0'])
        w = read_array(doc, binary, attrs['WEIGHTS_0'])
        f = read_array(doc, binary, primitive['indices']).ravel().astype('<i4').reshape(-1, 3)
        require(primitive.get('mode', 4) == 4 and len(p) == len(n) == len(uv) == len(j) == len(w)
                and f.min() >= 0 and f.max() < len(p), 'Invalid source primitive')
        p = np.column_stack((p[:, 0], -p[:, 2], p[:, 1])).astype('<f4')
        n = np.column_stack((n[:, 0], -n[:, 2], n[:, 1])).astype('<f4')
        points.append(p); normals.append(n); uvs.append(uv); joints.append(j); weights.append(w)
        name = doc['materials'][primitive['material']]['name']
        require(groom['body_before']['material_names'].count(name) == 1, 'Ambiguous native material ownership')
        native_slot = groom['body_before']['material_names'].index(name)
        faces.append(f+vstart); slots.extend([native_slot]*len(f))
        primitive_rows.append({'primitive': pi, 'material': name, 'gltf_material_index': primitive['material'],
                               'native_material_slot': native_slot,
                               'vertex_start': vstart, 'vertex_count': len(p),
                               'face_start': fstart, 'face_count': len(f)})
        vstart += len(p); fstart += len(f)
    require(vstart <= 100000 and fstart <= 100000, 'Source geometry exceeds bounds')
    points = np.concatenate(points); faces = np.concatenate(faces).astype('<i4')
    slots = np.asarray(slots, dtype='<i4'); uvs = np.concatenate(uvs)
    normals = np.concatenate(normals); joints = np.concatenate(joints); weights = np.concatenate(weights)
    position_hash = array_sha(points)
    positive_zero_points = points.copy(); positive_zero_points[positive_zero_points == 0] = 0
    zero_hash = array_sha(positive_zero_points)
    binding = {'source_positions_sha256': position_hash, 'zero_canonical_positions_sha256': zero_hash,
               'native_positions_sha256': groom['body_before']['positions'],
               'faces_sha256': array_sha(faces), 'native_faces_sha256': groom['body_before']['faces'],
               'slots_sha256': array_sha(slots), 'native_slots_sha256': groom['body_before']['material_slots'],
               'vertices': len(points), 'triangles': len(faces),
               'coordinate_space': 'Blender Z-up XYZ meters; exact glTF x,y,z -> x,-z,y',
               'node_mesh_bindings': [x for x in doc['nodes'] if 'mesh' in x],
               'primitive_ranges': primitive_rows}
    binding['exact_native_array_binding'] = (groom['body_before']['positions'] in (position_hash, zero_hash)
        and array_sha(faces) == groom['body_before']['faces']
        and array_sha(slots) == groom['body_before']['material_slots']
        and len(points) == groom['source_vertices_per_emitter'])
    save('source-binding.json', binding)
    require(binding['exact_native_array_binding'], 'Native source-ID correspondence not proved; no census claimed')
    if zero_hash == groom['body_before']['positions']:
        points = positive_zero_points
    corr_path = folder/'material-groom/correspondence.npz'
    with zipfile.ZipFile(corr_path) as z:
        require(len(z.namelist()) <= 16 and len(set(z.namelist())) == len(z.namelist())
                and sum(x.file_size for x in z.infolist()) <= 8*1024**2, 'NPZ exceeds bound')
    with np.load(corr_path, allow_pickle=False) as z:
        corr = {k: z[k] for k in z.files if k != 'source_corner_normals'}
    require(np.array_equal(corr['source_vertex_ids'], np.arange(len(points))), 'Native vertex IDs changed')
    low, span = points.min(axis=0).astype(float), np.ptp(points.astype(float), axis=0)
    require(np.all(span > 0), 'Degenerate bounds')
    tri = points[faces].astype(float)
    areas = np.linalg.norm(np.cross(tri[:, 1]-tri[:, 0], tri[:, 2]-tri[:, 0]), axis=1)*.5
    require(np.isfinite(areas).all(), 'Nonfinite source triangle area')
    allocations = coat['allocations'][0]['parts']
    require(coat['requested_cards'] == 3000 and len(coat['allocations']) == 1, 'Expected saved single 3,000-tuft band')
    reports, grid_rows, candidate_sets = [], [], []
    for part, allocation in zip(groom['parts'], allocations):
        index, slot = part['index'], part['source_material_slot']
        ids = corr[f'part_{index:02d}_face_ids']
        loops = corr[f'part_{index:02d}_loop_ids']
        require(array_sha(ids) == part['source_face_ids_sha256'] == allocation['source_triangle_ids_sha256']
                and array_sha(loops) == part['source_loop_ids_sha256']
                and np.array_equal(loops, (ids[:, None]*3+np.arange(3)).ravel())
                and len(ids) == allocation['eligible_triangles'] == part['faces']
                and np.all(slots[ids] == slot), 'Material correspondence mismatch')
        f, a = faces[ids], areas[ids]
        require(np.all(a > 0), 'Degenerate eligible triangle')
        require(abs(float(a.sum())-part['selected_area_m2']) <= 1e-7*part['selected_area_m2'], 'Source areas differ')
        vertices = np.unique(f); candidate_sets.append(vertices)
        count = allocation['cards']; p = points[vertices]
        lump = np.bincount(f.ravel(), weights=np.repeat(a/3, 3), minlength=len(points))[vertices]
        positions, inverse, multiplicity = np.unique(p, axis=0, return_inverse=True, return_counts=True)
        aliased = np.flatnonzero(multiplicity > 1)
        aliases = []
        for group in aliased:
            rows = vertices[inverse == group]
            uv_count = len(np.unique(uvs[rows], axis=0))
            n = normals[rows].astype(float); n /= np.linalg.norm(n, axis=1)[:, None]
            angle = float(np.degrees(np.arccos(np.clip((n@n.T).min(), -1, 1))))
            skin = np.column_stack((joints[rows], weights[rows]))
            aliases.append({'vertices': rows.tolist(), 'uv_variants': uv_count,
                            'max_normal_angle_degrees': angle, 'skin_variants': len(np.unique(skin, axis=0))})
        edges = np.unique(np.sort(np.concatenate((f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]])), axis=1), axis=0)
        lengths = np.linalg.norm(points[edges[:, 1]].astype(float)-points[edges[:, 0]], axis=1)
        centroid_corner = np.linalg.norm(tri[ids]-tri[ids].mean(axis=1)[:, None], axis=2).min(axis=1)
        ids_by_vertex = components(f, len(points))
        labels, counts = np.unique(ids_by_vertex[vertices], return_counts=True)
        component_rows = []
        for label, number in zip(labels, counts):
            mask = ids_by_vertex[f[:, 0]] == label
            area = float(a[mask].sum())
            component_rows.append({'source_vertex_component': int(label), 'vertices': int(number),
                'faces': int(mask.sum()), 'area_m2': area, 'area_share': area/float(a.sum()),
                'fractional_area_quota': count*area/float(a.sum())})
        component_rows.sort(key=lambda x: (-x['area_m2'], x['source_vertex_component']))
        vc = np.minimum(((p.astype(float)-low)/span*6).astype(int), 5)
        fc = np.minimum(((tri[ids].mean(axis=1)-low)/span*6).astype(int), 5)
        cell_v = np.ravel_multi_index(vc.T, (6, 6, 6)); cell_f = np.ravel_multi_index(fc.T, (6, 6, 6))
        vcounts = np.bincount(cell_v, minlength=216); cell_area = np.bincount(cell_f, weights=a, minlength=216)
        quotas = count*cell_area/a.sum()
        grid = [{'cell': list(map(int, np.unravel_index(cell, (6, 6, 6)))), 'vertices': int(vcounts[cell]),
                 'centroid_assigned_area_m2': float(cell_area[cell]), 'fractional_area_quota': float(quotas[cell])}
                for cell in np.flatnonzero((vcounts > 0) | (cell_area > 0))]
        grid_rows.append({'material': part['source_material'], 'cells': grid})
        ideal = count*lump/lump.sum()
        report = {'material': part['source_material'], 'material_slot': slot, 'requested_tufts': count,
            'eligible_faces': len(ids), 'eligible_vertices': len(vertices), 'unique_exact_positions': len(positions),
            'index_capacity_pass': len(vertices) >= count, 'distinct_position_capacity_pass': len(positions) >= count,
            'selected_fraction_of_vertex_capacity': count/len(vertices), 'area_m2': float(a.sum()),
            'nominal_sqrt_area_per_tuft_m': float(np.sqrt(a.sum()/count)),
            'coincident_groups': len(aliased), 'coincident_vertex_instances': int(multiplicity[aliased].sum()),
            'duplicate_position_excess': len(vertices)-len(positions), 'max_position_multiplicity': int(multiplicity.max()),
            'coincident_uv_seam_groups': sum(x['uv_variants'] > 1 for x in aliases),
            'coincident_normal_split_groups_over_1deg': sum(x['max_normal_angle_degrees'] > 1 for x in aliases),
            'coincident_skin_split_groups': sum(x['skin_variants'] > 1 for x in aliases),
            'alias_examples': aliases[:8], 'lumped_area_m2': quantiles(lump),
            'nominal_inclusion_mass_over_one': int((ideal > 1).sum()),
            'maximum_nominal_inclusion_mass': float(ideal.max()),
            'edge_length_m': quantiles(lengths),
            'all_candidate_face_centroid_nearest_own_corner_m': quantiles(centroid_corner, a),
            'indexed_components': component_rows,
            'grid_centroid_area_cells': int((cell_area > 0).sum()),
            'grid_cells_quota_exceeds_vertex_capacity': int((quotas > vcounts).sum()),
            'grid_area_fraction_without_own_cell_vertex': float(cell_area[vcounts == 0].sum()/a.sum())}
        reports.append(report)
        require(time.monotonic()-START < 25, 'Internal 25-second census deadline')
    all_vertices = np.concatenate(candidate_sets)
    all_positions = points[all_vertices]
    _, inverse, multiplicity = np.unique(all_positions, axis=0, return_inverse=True, return_counts=True)
    material_owner = np.concatenate([np.full(len(rows), i) for i, rows in enumerate(candidate_sets)])
    cross_material = sum(len(np.unique(material_owner[inverse == i])) > 1 for i in np.flatnonzero(multiplicity > 1))
    save('spatial-grid.json', {'grid': [6, 6, 6], 'low_blender_xyz_m': low.tolist(), 'span_m': span.tolist(),
        'method': 'Area assigned to source triangle centroid; candidate count by source vertex. Not clipped surface coverage.',
        'materials': grid_rows})
    result = {'schema_version': 1, 'status': 'measured', 'source_binding': binding,
        'unchanged_requested_tuft_count': 3000, 'materials': reports,
        'cross_material_exact_coincident_position_groups': int(cross_material),
        'source_tuft_settings': cfg['portable_fur']['settings'],
        'measurement_scope': 'Capacity, seams and topology only; no new roots selected, images decoded or geometry generated.',
        'limitations': ['Unique vertex indices do not imply unique spatial root positions.',
            'Fractional area quota is a diagnostic target, not the inclusion law of weighted sampling without replacement.',
            'Face-centroid-to-corner distance is an upper bound to the nearest eligible source vertex, not a coverage radius.',
            'Indexed components can be split by UV seams; no welding or anatomy inference was performed.',
            'Counts and surface area do not certify dense fur, silhouette quality, material coverage or Muse parity.'],
        'elapsed_seconds': time.monotonic()-START,
        'self_peak_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform == 'darwin' else 1024)}
    save('report.json', result)
    print(json.dumps({'status': result['status'], 'elapsed_seconds': result['elapsed_seconds'],
        'material_capacity': [{k:r[k] for k in ['material','requested_tufts','eligible_vertices','unique_exact_positions',
            'duplicate_position_excess','nominal_inclusion_mass_over_one','grid_cells_quota_exceeds_vertex_capacity']}
                              for r in reports], 'cross_material_coincident_groups': int(cross_material)}, sort_keys=True))


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        save('failure.json', {'status': 'failed', 'type': type(exc).__name__, 'message': str(exc),
                             'elapsed_seconds': time.monotonic()-START})
        raise
