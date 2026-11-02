"""Read-only cheek reflection screen; no fit, geometry edit, or texture decode."""
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import resource
import time
import zipfile

import numpy as np

from glb_reader import GLB

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
START = time.monotonic()


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write(name, value):
    with (OUT / name).open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')


def unit(value):
    norm = np.linalg.norm(value, axis=-1, keepdims=True)
    require(np.all(norm > 1e-12), 'Undefined direction')
    return value / norm


def blender(value):
    result = np.asarray(value, dtype=np.float64)[..., [0, 2, 1]].copy()
    result[..., 1] *= -1
    return result


def stats(value):
    value = np.asarray(value, dtype=float)
    return {'count': int(value.size), 'min': float(value.min()) if value.size else None,
            'median': float(np.median(value)) if value.size else None,
            'p95': float(np.percentile(value, 95)) if value.size else None,
            'max': float(value.max()) if value.size else None}


def angle(a, b):
    return np.degrees(np.arccos(np.clip(np.sum(unit(a) * unit(b), axis=-1), -1, 1)))


def closest_batch(queries, triangles):
    """Exact closest point on each triangle, including its three finite edges."""
    a, b, c = triangles[:, 0], triangles[:, 1], triangles[:, 2]
    u, v = b - a, c - a
    uu, uv, vv = np.sum(u*u, 1), np.sum(u*v, 1), np.sum(v*v, 1)
    denominator = uu*vv-uv*uv
    require(np.all(denominator > 1e-24), 'Degenerate donor triangle')
    delta = queries[:, None, :] - a
    pu, pv = np.sum(delta*u, 2), np.sum(delta*v, 2)
    x = (vv*pu-uv*pv)/denominator
    y = (uu*pv-uv*pu)/denominator
    bary = np.stack((1-x-y, x, y), axis=2)
    closest = a + x[..., None]*u + y[..., None]*v
    distance = np.sum((queries[:, None, :]-closest)**2, 2)
    distance[np.any(bary < 0, axis=2)] = np.inf
    for i, j in ((0, 1), (1, 2), (2, 0)):
        origin, edge = triangles[:, i], triangles[:, j]-triangles[:, i]
        t = np.clip(np.sum((queries[:, None, :]-origin)*edge, 2)/np.sum(edge*edge, 1), 0, 1)
        point = origin + t[..., None]*edge
        d = np.sum((queries[:, None, :]-point)**2, 2)
        use = d < distance
        distance[use] = d[use]
        closest[use] = point[use]
        row = np.zeros_like(bary)
        row[..., i], row[..., j] = 1-t, t
        bary[use] = row[use]
    return distance, closest, bary


def nearest(queries, triangles, clean):
    all_rows, clean_rows = [], []
    for first in range(0, len(queries), 8):
        require(time.monotonic()-START < 50, 'Internal screen deadline reached')
        distance, points, bary = closest_batch(queries[first:first+8], triangles)
        for mask, rows in ((np.ones(len(triangles), dtype=bool), all_rows), (clean, clean_rows)):
            masked = np.where(mask[None, :], distance, np.inf)
            ids = np.argmin(masked, axis=1)
            for row, index in enumerate(ids):
                rows.append((int(index), float(np.sqrt(masked[row, index])),
                             points[row, index].copy(), bary[row, index].copy()))
    return all_rows, clean_rows


def primitive(model, index):
    p = model.doc['meshes'][0]['primitives'][index]
    require(p.get('mode', 4) == 4 and not p.get('targets'), 'Unsupported mesh primitive')
    points = model.array(p['attributes']['POSITION'])
    normals = model.array(p['attributes']['NORMAL'])
    faces = model.array(p['indices']).reshape(-1, 3)
    require(len(points) == len(normals) and faces.max() < len(points), 'Invalid source geometry')
    return points, normals, faces


def main():
    prep = json.loads((OUT/'prepared.json').read_text())
    for item in prep['inputs'] + prep['scripts']:
        require(sha(ROOT/item['path']) == item['sha256'], 'Pinned input changed: '+item['path'])
    cfg = json.loads((ROOT/prep['config']).read_text())
    candidate = json.loads((ROOT/prep['candidate']).read_text())
    scar = json.loads((ROOT/prep['scar_selection']).read_text())
    require(cfg['authored_rig']['space'] == 'normalized_bounds_blender_z_up', 'Unknown guide axes')
    source = GLB(ROOT/prep['source'])
    old = GLB(ROOT/prep['v7_source'])
    for model in (source, old):
        require(len(model.doc['meshes']) == 1 and len(model.doc['nodes']) == 1, 'Unexpected scene')
        require(not model.doc.get('skins') and not model.doc.get('animations'), 'Unexpected rig')
        node = model.doc['nodes'][0]
        require(node.get('mesh') == 0 and not any(key in node for key in ('matrix', 'translation', 'rotation', 'scale')), 'Nonidentity source frame')
    current = [primitive(source, i) for i in range(len(source.doc['meshes'][0]['primitives']))]
    previous = [primitive(old, i) for i in range(len(old.doc['meshes'][0]['primitives']))]
    require(sum(len(x[0]) for x in current) <= 100000 and sum(len(x[2]) for x in current) <= 100000, 'Geometry cap')
    with zipfile.ZipFile(ROOT/prep['correspondence']) as archive:
        entries = archive.infolist()
        require(len(entries) == 2 and sum(x.file_size for x in entries) < 16*1024**2, 'Correspondence decode cap')
    with np.load(ROOT/prep['correspondence'], allow_pickle=False) as archive:
        face_map = archive['face_map'].copy()
    require(face_map.shape == (62581, 4) and face_map.dtype.kind in 'iu', 'Invalid face map')
    require(len(set(map(tuple, face_map[:, :2]))) == len(face_map) and len(set(map(tuple, face_map[:, 2:]))) == len(face_map), 'Nonunique source face mapping')
    for op in range(len(previous)):
        for cp in range(len(current)):
            rows = face_map[(face_map[:, 0] == op) & (face_map[:, 2] == cp)]
            if not len(rows):
                continue
            for field in (0, 1):
                require(np.array_equal(previous[op][field][previous[op][2][rows[:, 1]]],
                                       current[cp][field][current[cp][2][rows[:, 3]]]), 'Current P/N differs from v7 source corners')
    to_current = {tuple(r[:2]): tuple(map(int, r[2:])) for r in face_map}
    all_points = np.concatenate([blender(x[0]) for x in current])
    low, high = all_points.min(0), all_points.max(0)
    span = high-low
    require(np.all(span > 0), 'Degenerate source bounds')
    # Author annotations establish the axis and hypothesis; no damaged-patch fitting.
    exclusions = cfg['groom_exclusions'][:4]
    eye_a, eye_b = [low+span*np.asarray(x['center']) for x in exclusions[:2]]
    eye_mid = (eye_a+eye_b)*.5
    nose = low+span*np.asarray(exclusions[2]['center'])
    head = next(b for b in cfg['authored_rig']['bones'] if b['name'] == 'Head')
    head_position = low+span*np.asarray(head['head'])
    planes = [
        {'name': 'eye_midpoint_x_plane', 'point': eye_mid, 'normal': np.array([1., 0, 0]), 'primary': True},
        {'name': 'nose_center_x_plane', 'point': nose, 'normal': np.array([1., 0, 0]), 'primary': False},
        {'name': 'head_rig_x_plane', 'point': head_position, 'normal': np.array([1., 0, 0]), 'primary': False},
        {'name': 'eye_pair_perpendicular_plane', 'point': eye_mid, 'normal': unit(eye_b-eye_a), 'primary': False},
    ]
    write('landmarks-and-planes.json', {
        'coordinate_space': 'Blender world XYZ, Z up, meters',
        'gltf_to_blender': '[x,-z,y]', 'bounds': {'min': low.tolist(), 'max': high.tolist()},
        'eyes': [eye_a.tolist(), eye_b.tolist()], 'nose': nose.tolist(), 'head_bone_head': head_position.tolist(),
        'source': prep['config'], 'source_landmark_kind': 'Authored groom exclusions and rig pivots, not recovered anatomical landmarks',
        'whole_model_bbox_x_center_m': float((low[0]+high[0])*.5),
        'planes': [{**p, 'point': p['point'].tolist(), 'normal': p['normal'].tolist(),
                    'nose_signed_offset_m': float(np.dot(nose-p['point'], p['normal']))} for p in planes],
        'assumption': 'Bilateral geometric reflection only. Source asymmetry and accessory-biased bounds remain possible; no exact symmetry or semantic correspondence is asserted.'})

    points, normals, faces, source_ids = [], [], [], []
    for pi in (0, 2):
        p, n, f = previous[pi]
        require(old.doc['materials'][old.doc['meshes'][0]['primitives'][pi]['material']]['name'] in ('body_surface', 'body_cap'), 'Body identity changed')
        faces.append(f+sum(len(x) for x in points))
        points.append(p.copy()); normals.append(n.copy())
        source_ids.extend((pi, i) for i in range(len(f)))
    p_gltf, n_gltf = np.concatenate(points), np.concatenate(normals)
    faces = np.concatenate(faces)
    unique, inverse = np.unique(p_gltf, axis=0, return_inverse=True)
    welded = inverse[faces]
    points, normals, positions = blender(p_gltf), unit(blender(n_gltf)), blender(unique)
    triangles = points[faces]
    cross = np.cross(triangles[:, 1]-triangles[:, 0], triangles[:, 2]-triangles[:, 0])
    area2 = np.linalg.norm(cross, axis=1)
    geometric = cross/np.maximum(area2[:, None], 1e-30)
    edge_faces = defaultdict(list)
    for fid, tri in enumerate(welded):
        for a, b in zip(tri, np.roll(tri, -1)):
            edge_faces[tuple(sorted((int(a), int(b))))].append(fid)
    dihedral = np.zeros(len(faces))
    abnormal = np.zeros(len(faces), dtype=bool)
    for incident in edge_faces.values():
        if len(incident) != 2:
            abnormal[incident] = True
        else:
            a, b = incident
            value = float(angle(geometric[a], geometric[b])) if area2[a] > 1e-12 and area2[b] > 1e-12 else 180.
            dihedral[incident] = np.maximum(dihedral[incident], value)
    custom_geo = np.degrees(np.arccos(np.clip(np.sum(normals[faces]*geometric[:, None, :], 2), -1, 1))).max(1)
    lookup = {key: i for i, key in enumerate(source_ids)}
    deleted = {lookup[(0, 4961)]}
    boundary_sets = [
        ('original_valid_disk', {lookup[tuple(x)] for x in candidate['replacement_source_faces']}, 78),
        ('stopped_scar_domain', {lookup[tuple(x['source_face'])] for x in scar['replacement']}, 65),
    ]
    queries, expected_custom, expected_geometric, metadata, boundaries = [], [], [], [], []
    for name, selected, count in boundary_sets:
        local = defaultdict(list)
        for fid in selected:
            for a, b in zip(welded[fid], np.roll(welded[fid], -1)):
                local[tuple(sorted((int(a), int(b))))].append(fid)
        boundary = sorted(edge for edge, incident in local.items() if len(incident) == 1)
        require(len(boundary) == count, 'Frozen boundary differs')
        if name == 'original_valid_disk':
            expected_ids = candidate['oriented_boundary_body_welded_ids']
            expected_edges = {tuple(sorted((a, b))) for a, b in zip(expected_ids, expected_ids[1:]+expected_ids[:1])}
            require(set(boundary) == expected_edges, 'Original boundary winding/identity changed')
        start = len(queries)
        for ordinal, edge in enumerate(boundary):
            retained = [f for f in edge_faces[edge] if f not in selected|deleted]
            require(len(retained) == 1, 'Ambiguous retained boundary face')
            fid = retained[0]
            corners = [int(np.flatnonzero(welded[fid] == v)[0]) for v in edge]
            expected = normals[faces[fid, corners]]
            for fraction in (0., .5, 1.):
                queries.append(positions[edge[0]]*(1-fraction)+positions[edge[1]]*fraction)
                expected_custom.append(unit(expected[0]*(1-fraction)+expected[1]*fraction))
                expected_geometric.append(geometric[fid])
                metadata.append({'domain': name, 'edge_ordinal': ordinal, 'body_welded_edge': list(edge),
                    'edge_fraction': fraction, 'retained_v7_face': list(source_ids[fid]),
                    'retained_current_face': list(to_current[source_ids[fid]]),
                    'retained_dihedral_degrees': float(dihedral[fid]),
                    'retained_custom_geometric_degrees': float(custom_geo[fid])})
        boundaries.append({'name': name, 'edges': len(boundary), 'vertices': len(set(v for e in boundary for v in e)),
                           'samples': len(queries)-start, 'sample_start': start})
    require(len(queries) == 429, 'Unexpected screen size')
    queries, expected_custom, expected_geometric = map(np.asarray, (queries, expected_custom, expected_geometric))
    write('frozen-boundary-samples.json', {'domains': boundaries, 'samples': [dict(m, position_blender_m=q.tolist()) for m, q in zip(metadata, queries)],
        'scope': 'Exact old selections only. Position alias IDs are used for topology accounting, never mesh welding or a new selection.'})

    head_region = next(r for r in cfg['groom_flow']['regions'] if r['name'] == 'head-short')
    head_center = low+span*np.asarray(head_region['center'])
    head_radius = span*np.asarray(head_region['radius'])
    normalized = (triangles.mean(1)-head_center)/head_radius
    original_body = np.asarray([key[0] == 0 for key in source_ids])
    donor_domain = original_body & np.all(triangles[:, :, 0] > eye_mid[0], 1) & (area2 > 1e-12)
    protected_overlap = np.zeros(len(faces), dtype=bool)
    for region in exclusions:
        transformed = (triangles-(low+span*np.asarray(region['center'])))/(span*np.asarray(region['radius']))
        valid = np.flatnonzero(donor_domain)
        d, _, _ = closest_batch(np.zeros((1, 3)), transformed[valid])
        protected_overlap[valid[d[0] <= 1]] = True
    donors = np.flatnonzero(donor_domain & ~protected_overlap)
    require(0 < len(donors) <= 20000, 'Donor pool cap or empty domain')
    clean = (dihedral[donors] <= 30) & (custom_geo[donors] <= 15) & ~abnormal[donors]
    require(clean.any(), 'No healthy opposite cheek support')
    write('donor-inventory.json', {'domain': 'Original body_surface triangles only, entire opposite X half without a groom-region spatial cutoff; any triangle intersecting eye/nose/muzzle exclusion ellipsoids is excluded.',
        'triangles_before_feature_exclusion': int(donor_domain.sum()), 'feature_excluded_triangles': int((donor_domain&protected_overlap).sum()),
        'donor_faces': len(donors), 'clean_donor_faces': int(clean.sum()),
        'clean_policy': {'maximum_dihedral_degrees': 30, 'maximum_custom_vs_geometric_degrees': 15, 'edge_valence_must_equal': 2},
        'clean_selection_independent_of_target_fit_error': True,
        'geometric_curvature_proxy': 'Maximum adjacent-face dihedral, not a smooth-surface curvature estimate',
        'donor_dihedral_degrees': stats(dihedral[donors]), 'clean_dihedral_degrees': stats(dihedral[donors[clean]]),
        'donor_custom_geometric_degrees': stats(custom_geo[donors]),
        'faces': [{'v7_face': list(source_ids[f]), 'current_face': list(to_current[source_ids[f]]), 'clean': bool(c),
                   'dihedral_degrees': float(dihedral[f]), 'custom_geometric_degrees': float(custom_geo[f])} for f, c in zip(donors, clean)]})

    results = []
    for plane in planes:
        normal, origin = plane['normal'], plane['point']
        reflected = queries-2*np.sum((queries-origin)*normal, axis=1)[:, None]*normal
        nearest_all, nearest_clean = nearest(reflected, triangles[donors], clean)
        plane_rows = []
        for scope, matches in (('all_unprotected_donors', nearest_all), ('clean_donors', nearest_clean)):
            rows = []
            for i, (index, distance, point, bary) in enumerate(matches):
                fid = int(donors[index])
                g = geometric[fid]-2*np.dot(geometric[fid], normal)*normal
                c = unit(np.sum(normals[faces[fid]]*bary[:, None], axis=0))
                c -= 2*np.dot(c, normal)*normal
                geometric_error = float(angle(g, expected_geometric[i]))
                custom_error = float(angle(c, expected_custom[i]))
                rows.append({**metadata[i], 'sample': i, 'donor_v7_face': list(source_ids[fid]),
                    'donor_current_face': list(to_current[source_ids[fid]]), 'donor_barycentric': bary.tolist(),
                    'reflected_query_blender_m': reflected[i].tolist(), 'closest_donor_blender_m': point.tolist(),
                    'distance_m': distance, 'reflected_geometric_normal': g.tolist(), 'reflected_custom_normal': c.tolist(),
                    'geometric_normal_disagreement_degrees': geometric_error,
                    'custom_normal_disagreement_degrees': custom_error,
                    'donor_dihedral_degrees': float(dihedral[fid]), 'donor_clean': bool(clean[index]),
                    'within_15mm': distance <= .015, 'within_45deg_custom': custom_error <= 45,
                    'within_45deg_geometric': geometric_error <= 45,
                    'joint_support': bool(distance <= .015 and custom_error <= 45 and geometric_error <= 45 and clean[index])})
            for domain in boundaries:
                selected = [r for r in rows if r['domain'] == domain['name']]
                edge_pass = [all(r['joint_support'] for r in selected[e:e+3]) for e in range(0, len(selected), 3)]
                plane_rows.append({'domain': domain['name'], 'donor_scope': scope, 'samples': len(selected),
                    'distance_m': stats([r['distance_m'] for r in selected]),
                    'custom_normal_disagreement_degrees': stats([r['custom_normal_disagreement_degrees'] for r in selected]),
                    'geometric_normal_disagreement_degrees': stats([r['geometric_normal_disagreement_degrees'] for r in selected]),
                    'within_15mm_samples': sum(r['within_15mm'] for r in selected),
                    'within_45deg_custom_samples': sum(r['within_45deg_custom'] for r in selected),
                    'within_45deg_geometric_samples': sum(r['within_45deg_geometric'] for r in selected),
                    'joint_support_samples': sum(r['joint_support'] for r in selected),
                    'fully_supported_edges': sum(edge_pass), 'total_edges': len(edge_pass),
                    'unique_matched_donor_triangles': len(set(tuple(r['donor_v7_face']) for r in selected))})
            write(f"matches-{plane['name']}-{scope}.json", {'plane': plane['name'], 'scope': scope, 'rows': rows,
                'correspondence_scope': 'Exact closest triangle point in the declared donor pool, not an anatomical match. Equal-distance adjacent triangles use deterministic source order.'})
        results.append({'plane': plane['name'], 'primary': plane['primary'], 'domains': plane_rows})
    require(all(sha(ROOT/item['path']) == item['sha256'] for item in prep['inputs']), 'Source changed during screen')
    summary = {'schema_version': 1, 'status': 'reflection_reference_screen_only', 'source_sha256': sha(ROOT/prep['source']),
        'v7_source_sha256': sha(ROOT/prep['v7_source']), 'coordinate_space': 'Blender Z-up meters',
        'source_corner_positions_and_normals_exact_v7_to_latest': True, 'source_faces_verified': len(face_map),
        'planes': results, 'donor_faces': len(donors), 'clean_donor_faces': int(clean.sum()),
        'gates_unchanged': {'maximum_distance_m': .015, 'normal_p95_limit_degrees': 45},
        'protected_scope': 'Every source byte is unchanged, including eyes/nose/muzzle, all244 protected white faces, protected0/5795 and0/5812, companion, UVs/images and the forbidden3009-3091 diagonal.',
        'stopped_topology_still_invalid': True, 'new_fit_performed': False, 'new_selection_performed': False,
        'model_modified': False, 'source_files_unchanged': True, 'quality_approved': False,
        'native_jobs': 0, 'provider_calls': 0, 'textures_decoded': 0,
        'elapsed_seconds': time.monotonic()-START, 'self_peak_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        'limits': prep['limits'],
        'limitations': ['Reflection is an authored symmetry assumption, not recovery of unseen geometry or proof of source symmetry.',
            'Nearest points provide reference support only; no closed replacement, displacement, collision, texture, rig, motion or visual validation was attempted.',
            'The prior stopped scar domain remains topologically invalid; reflection does not waive its protected faces or topology limits.',
            'Donor curvature uses finite mesh dihedral, which depends on source tessellation. No smooth curvature field was fitted.']}
    write('summary.json', summary)
    print(json.dumps({'status': summary['status'], 'planes': results, 'donors': len(donors),
                      'clean_donors': int(clean.sum()), 'elapsed_seconds': summary['elapsed_seconds']}))


if __name__ == '__main__':
    # Analytic sanity check precedes source loading inside the same bounded job.
    d, p, b = closest_batch(np.array([[.2, .3, 1.], [2., 0., 0.]]), np.array([[[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]]]))
    require(np.allclose(d[:, 0], [1, 1]) and np.allclose(p[0, 0], [.2, .3, 0]) and np.allclose(b.sum(2), 1), 'Closest triangle oracle failed')
    try:
        main()
    except Exception as error:
        write('failure.json', {'type': type(error).__name__, 'message': str(error), 'elapsed_seconds': time.monotonic()-START,
                              'model_modified': False, 'automatic_retry': False})
        raise
