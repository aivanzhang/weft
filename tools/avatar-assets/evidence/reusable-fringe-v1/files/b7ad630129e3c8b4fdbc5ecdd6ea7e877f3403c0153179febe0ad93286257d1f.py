"""Bounded authored edge fins; no source repair, texture sampling, or quality claim.

Positions are meters. Normals, flow, and fixed design views share that applied
mesh frame. UV corners are passed through unchanged; returned atlas UV is V-down.
Exact coincident positions classify unsafe aliases only; source data is not welded.
"""

import itertools
import math
import time

import numpy as np


MAX_VERTICES = 100000
MAX_FACES = 100000
MAX_FINS = 4800
MAX_CANDIDATES = 32000
FRAME_POLICY = 'edge-fringe-source-surface-v1'
ATLAS_DOMAIN_M = .0208
TIP_CENTER_M = .0025
TIP_RADIUS_M = .0008


def require(condition, message):
    if not condition:
        raise ValueError(message)


def _number(value, low, high, name):
    require(type(value) in (int, float) and math.isfinite(value) and low <= value <= high,
            'Invalid '+name)


def validate_settings(settings):
    """Validate the fixed physical policy without inspecting or mutating a mesh."""
    fields = {'seed', 'min_fins', 'max_fins', 'length_m', 'min_spacing_m', 'min_edge_m', 'max_edge_m', 'comb'}
    require(isinstance(settings, dict) and set(settings) == fields, 'Unexpected fin settings')
    require(type(settings['seed']) is int and 0 <= settings['seed'] <= 2**31-1, 'Invalid fin seed')
    require(type(settings['min_fins']) is int and type(settings['max_fins']) is int and
            1 <= settings['min_fins'] <= settings['max_fins'] <= MAX_FINS, 'Invalid fin count')
    _number(settings['length_m'], .001, .04, 'fin length')
    _number(settings['min_spacing_m'], .0005, .05, 'root spacing')
    _number(settings['min_edge_m'], .0085, .020, 'minimum edge span; supported range is 8.5..20 mm')
    _number(settings['max_edge_m'], settings['min_edge_m'], .020, 'maximum edge span; supported range is 8.5..20 mm')
    _number(settings['comb'], 0, .35, 'fin lean')
    return dict(settings)


def export_budget(source_export_vertices, source_export_triangles, max_fins):
    """Reserve the requested maximum; never reduce a request to fit an export cap."""
    require(type(source_export_vertices) is int and 3 <= source_export_vertices <= MAX_VERTICES,
            'Invalid source export vertex count')
    require(type(source_export_triangles) is int and 1 <= source_export_triangles <= MAX_FACES,
            'Invalid source export triangle count')
    require(type(max_fins) is int and 1 <= max_fins <= MAX_FINS, 'Invalid maximum fin count')
    vertices = source_export_vertices+12*max_fins+4096
    triangles = source_export_triangles+4*max_fins
    require(vertices <= MAX_VERTICES, 'Base plus twelve corners per fin and reserve exceeds 100000 vertices')
    require(triangles <= MAX_FACES, 'Base plus four triangles per fin exceeds 100000 triangles')
    return {'source_export_vertices': source_export_vertices, 'source_export_triangles': source_export_triangles,
            'max_fins': max_fins, 'worst_exported_fin_corners': 12*max_fins, 'reserve_vertices': 4096,
            'projected_vertices': vertices, 'projected_triangles': triangles,
            'vertex_limit': MAX_VERTICES, 'triangle_limit': MAX_FACES}


def _array(value, shape, name, kinds):
    require(isinstance(value, np.ndarray) and value.shape == shape and value.dtype.kind in kinds,
            'Invalid '+name+' array')
    if value.dtype.kind == 'f':
        require(value.dtype.itemsize in (4, 8) and np.isfinite(value).all(), 'Nonfinite/unsupported '+name)
    return value


def _deadline(started):
    require(time.monotonic()-started < 10, 'Fin builder exceeded its ten-second internal deadline')


def _edge_table(points, faces):
    directed = faces[:, [[0, 1], [1, 2], [2, 0]]].reshape(-1, 2)
    corners = np.tile([[0, 1], [1, 2], [2, 0]], (len(faces), 1))
    flipped = directed[:, 0] > directed[:, 1]
    corners[flipped] = corners[flipped, ::-1]
    edges = np.sort(directed, axis=1)
    order = np.lexsort((edges[:, 1], edges[:, 0]))
    sorted_edges = edges[order]
    starts = np.flatnonzero(np.r_[True, np.any(sorted_edges[1:] != sorted_edges[:-1], axis=1)])
    counts = np.diff(np.r_[starts, len(edges)])
    # Reject an indexed root even when its coincident alias is an excluded boundary.
    _, position_ids = np.unique(points, axis=0, return_inverse=True)
    position_pairs = np.sort(position_ids[sorted_edges[starts]], axis=1)
    codes = position_pairs[:, 0].astype(np.int64)*len(points)+position_pairs[:, 1]
    _, groups, group_counts = np.unique(codes, return_inverse=True, return_counts=True)
    shared = np.flatnonzero(counts == 2)
    left, right = order[starts[shared]], order[starts[shared]+1]
    aliases = group_counts[groups[shared]] > 1
    return edges[left], left//3, right//3, corners[left], corners[right], flipped[left] == flipped[right], aliases, counts


def build_edge_fringe(points, faces, corner_normals, uv0, materials, parts, fur_faces,
                      non_fur_faces, closure_faces, joints, weights, flow_directions,
                      design_views, settings, *, source_export_vertices, source_export_triangles):
    """Return detached arrays, exact root correspondence, and a bounded authoring report.

Four vertices per fin: endpoint0, endpoint1, tip0, tip1. Indices contain all
original triangles followed by their reversed copies; material must be single-sided.
Only the supplied positive fur mask is eligible. Part IDs are explicit caller data.
"""
    started = time.monotonic()
    require(isinstance(points, np.ndarray) and points.ndim == 2 and 3 <= len(points) <= MAX_VERTICES,
            'Invalid or oversized source points')
    require(isinstance(faces, np.ndarray) and faces.ndim == 2 and 1 <= len(faces) <= MAX_FACES,
            'Invalid or oversized source faces')
    nv, nf = len(points), len(faces)
    settings = validate_settings(settings)
    budget = export_budget(source_export_vertices, source_export_triangles, settings['max_fins'])
    require(source_export_vertices >= nv and source_export_triangles >= nf, 'Source export counts understate input geometry')
    points = _array(points, (nv, 3), 'points', 'f')
    faces = _array(faces, (nf, 3), 'faces', 'iu')
    require(faces.min() >= 0 and faces.max() < nv and np.max(np.abs(points)) <= 10000, 'Source indices/positions exceed bounds')
    normal = _array(corner_normals, (nf, 3, 3), 'corner normals', 'f').astype(np.float64)
    require(np.all(np.abs(np.linalg.norm(normal, axis=2)-1) < 1e-5), 'Require unit source normals')
    uv0 = _array(uv0, (nf, 3, 2), 'UV0', 'f')
    require(np.all((uv0 >= 0)&(uv0 <= 1)), 'Source UV0 must be in 0..1')
    materials = _array(materials, (nf,), 'material IDs', 'iu')
    parts = _array(parts, (nf,), 'part IDs', 'iu')
    require(materials.min() >= 0 and materials.max() <= 31 and parts.min() >= 0 and parts.max() <= 255,
            'Material/part IDs exceed bounds')
    fur_faces = _array(fur_faces, (nf,), 'fur eligibility', 'b')
    non_fur_faces = _array(non_fur_faces, (nf,), 'non-fur exclusions', 'b')
    closure_faces = _array(closure_faces, (nf,), 'authored closure exclusions', 'b')
    joints = _array(joints, (nv, 4), 'joints', 'iu')
    weights = _array(weights, (nv, 4), 'weights', 'f')
    require(joints.min() >= 0 and joints.max() < 64 and np.all((weights >= 0)&(weights <= 1)) and
            np.all(np.abs(weights.astype(np.float64).sum(axis=1)-1) < 1e-6), 'Invalid four-weight skin')
    flow = _array(flow_directions, (nv, 3), 'flow directions', 'f').astype(np.float64)
    flow_length = np.linalg.norm(flow, axis=1)
    require(np.all(flow_length > 1e-8) and np.isfinite(flow_length).all(), 'Invalid flow direction')
    flow /= flow_length[:, None]
    require(isinstance(design_views, np.ndarray) and design_views.ndim == 2 and 1 <= len(design_views) <= 16,
            'Require 1..16 fixed design views')
    views = _array(design_views, (len(design_views), 3), 'design views', 'f').astype(np.float64)
    view_length = np.linalg.norm(views, axis=1)
    require(np.all(view_length > 1e-8) and np.isfinite(view_length).all(), 'Invalid design view')
    views /= view_length[:, None]

    xyz = points.astype(np.float64)[faces]
    geometry = np.cross(xyz[:, 1]-xyz[:, 0], xyz[:, 2]-xyz[:, 0])
    magnitude = np.linalg.norm(geometry, axis=1)
    require(np.all(magnitude > 1e-12), 'Degenerate source triangle')
    geometry /= magnitude[:, None]
    corner_outward = np.einsum('fvc,fc->fv', normal, geometry) > .05
    endpoints, f0, f1, c0, c1, winding_bad, aliases, counts = _edge_table(points, faces)
    _deadline(started)
    source_normals = normal[f0[:, None], c0]
    n = source_normals.sum(axis=1)+normal[f1[:, None], c1].sum(axis=1)
    n_length = np.linalg.norm(n, axis=1)
    n /= np.maximum(n_length, 1e-12)[:, None]
    edge_vector = points[endpoints[:, 1]].astype(np.float64)-points[endpoints[:, 0]]
    lengths = np.linalg.norm(edge_vector, axis=1)
    source_uv = uv0[f0[:, None], c0]
    reasons = np.full(len(endpoints), '', dtype='<U28')

    def exclude(condition, label):
        reasons[(reasons == '') & condition] = label

    exclude(aliases, 'coincident_edge_alias')
    exclude(~(fur_faces[f0]&fur_faces[f1])|non_fur_faces[f0]|non_fur_faces[f1], 'non_fur')
    exclude(closure_faces[f0]|closure_faces[f1], 'authored_closure')
    exclude(winding_bad, 'inconsistent_winding')
    exclude(materials[f0] != materials[f1], 'material_seam')
    exclude(parts[f0] != parts[f1], 'moving_part_seam')
    exclude(np.any(source_uv != uv0[f1[:, None], c1], axis=(1, 2)), 'UV_seam')
    exclude(np.any(source_normals != normal[f1[:, None], c1], axis=(1, 2)), 'normal_seam')
    exclude((lengths < settings['min_edge_m'])|(lengths > settings['max_edge_m']), 'edge_span')
    exclude((n_length < 1e-8)|~corner_outward[f0].all(axis=1)|~corner_outward[f1].all(axis=1)|
            (np.sum(geometry[f0]*geometry[f1], axis=1) < .5)|
            (np.sum(source_normals[:, 0]*source_normals[:, 1], axis=1) < .5), 'incoherent_normals')
    lean = flow[endpoints].sum(axis=1)
    lean -= n*np.sum(lean*n, axis=1)[:, None]
    lean /= np.maximum(np.linalg.norm(lean, axis=1), 1e-12)[:, None]
    direction = n*(1-settings['comb'])+lean*settings['comb']
    direction /= np.maximum(np.linalg.norm(direction, axis=1), 1e-12)[:, None]
    plane_normal = np.cross(edge_vector, direction)
    plane_length = np.linalg.norm(plane_normal, axis=1)
    exclude(plane_length < 1e-8, 'degenerate_fin_plane')
    plane_normal /= np.maximum(plane_length, 1e-12)[:, None]
    eligible = np.flatnonzero(reasons == '')
    require(len(eligible) >= settings['min_fins'], 'Insufficient safe separated edge roots')
    score = ((1-np.abs(n[eligible]@views.T))*np.abs(plane_normal[eligible]@views.T)).max(axis=1)
    rng = np.random.default_rng(settings['seed'])
    # Share face area among safe incident roots rather than rewarding finer tessellation.
    incident = np.bincount(np.r_[f0[eligible], f1[eligible]], minlength=nf)
    face_share = np.divide(.5*magnitude, incident, out=np.zeros(nf), where=incident > 0)
    area_mass = face_share[f0[eligible]]+face_share[f1[eligible]]
    require(np.all(area_mass > 0), 'Invalid edge area mass')
    priority = -np.log(np.maximum(rng.random(len(eligible)), np.finfo(float).tiny))/(area_mass*(.5+.5*score))
    ranking = eligible[np.lexsort((eligible, priority))]
    roots = points[endpoints].astype(np.float64).mean(axis=1)
    spacing = settings['min_spacing_m']
    buckets, chosen, checked, spacing_rejected = {}, [], 0, 0
    offsets = tuple(itertools.product((-1, 0, 1), repeat=3))
    for index in ranking[:MAX_CANDIDATES]:
        checked += 1
        if checked % 128 == 0:
            _deadline(started)
        root = roots[index]
        cell = tuple(np.floor(root/spacing).astype(np.int64))
        nearby = (other for delta in offsets for other in buckets.get(tuple(cell[i]+delta[i] for i in range(3)), ()))
        if any(np.dot(root-roots[other], root-roots[other]) < spacing**2 for other in nearby):
            spacing_rejected += 1
            continue
        chosen.append(int(index))
        buckets.setdefault(cell, []).append(int(index))
        if len(chosen) == settings['max_fins']:
            break
    require(len(chosen) >= settings['min_fins'], 'Insufficient safe separated edge roots within bounded candidate scan')
    chosen = np.asarray(chosen, dtype=np.int32)
    source_ids = endpoints[chosen]
    root_points = points[source_ids]
    positions = np.concatenate((root_points, root_points+direction[chosen, None]*settings['length_m']), axis=1)
    skin_ids = np.tile(joints[source_ids], (1, 2, 1))
    skin_weights = np.tile(weights[source_ids], (1, 2, 1)).astype(np.float64)
    skin_weights /= skin_weights.sum(axis=2, keepdims=True)
    shading = source_normals[chosen].copy()
    shading /= np.linalg.norm(shading, axis=2, keepdims=True)
    tiles = rng.integers(0, 16, len(chosen))
    half_uv = lengths[chosen]/(2*ATLAS_DOMAIN_M)
    local_uv = np.zeros((len(chosen), 4, 2))
    local_uv[:, [0, 2], 0], local_uv[:, [1, 3], 0] = .5-half_uv[:, None], .5+half_uv[:, None]
    local_uv[:, :2, 1], local_uv[:, 2:, 1] = .985, .015
    require(np.all((local_uv >= .015)&(local_uv <= .985)), 'Physical atlas crop outside tile gutter')
    atlas_uv = (local_uv+np.column_stack((tiles % 4, tiles//4))[:, None])/4
    front = (np.arange(len(chosen))[:, None, None]*4+np.array([[0, 1, 3], [0, 3, 2]])).reshape(-1, 3)
    triangles = np.concatenate((front, front[:, ::-1]))
    result = {'positions': positions.reshape(-1, 3).astype(np.float32), 'triangles': triangles.astype(np.int32),
              'normals': np.tile(shading, (1, 2, 1)).reshape(-1, 3).astype(np.float32),
              'uv': atlas_uv.reshape(-1, 2).astype(np.float32),
              'joints': skin_ids.reshape(-1, 4).astype(np.uint16), 'weights': skin_weights.reshape(-1, 4).astype(np.float32),
              'root_vertex_ids': source_ids.copy(), 'root_face_ids': f0[chosen].copy(),
              'root_corner_ids': c0[chosen].copy(), 'root_source_uv': source_uv[chosen].copy(),
              'material_ids': materials[f0[chosen]].copy(), 'root_centers': roots[chosen].copy(),
              'outward_normals': n[chosen].copy(), 'fin_directions': direction[chosen].copy()}
    result['report'] = {'profile': FRAME_POLICY, 'settings': settings, 'source_vertices': nv, 'source_faces': nf,
        'selected_fins': len(chosen), 'geometric_vertices': len(chosen)*4, 'triangles': len(chosen)*4,
        'export_budget': budget, 'unshared_boundary_edges': int((counts == 1).sum()),
        'nonmanifold_edges': int((counts > 2).sum()), 'coincident_shared_edge_aliases': int(aliases.sum()),
        'exclusions': {str(label): int((reasons == label).sum()) for label in np.unique(reasons) if label},
        'eligible_edges': len(eligible), 'examined_candidates': checked, 'candidate_scan_cap': MAX_CANDIDATES,
        'spacing_rejected': spacing_rejected, 'min_outward_dot': float(np.sum(n[chosen]*direction[chosen], axis=1).min()),
        'elapsed_seconds': time.monotonic()-started, 'camera_billboarding': False, 'source_modified': False,
        'material_requirement': {'alphaMode': 'MASK', 'alphaCutoff': .18, 'doubleSided': False},
        'normal_policy': 'Normalized source endpoint normals repeated at corresponding tips; not fin-plane normals',
        'quality_approved': False,
        'allocation': {'policy': 'eligible-face-area-share-seeded-exponential-v1',
            'represented_face_area_m2': float(np.sum(.5*magnitude[incident > 0])),
            'edge_area_mass_sum_m2': float(area_mass.sum()), 'view_preference_bounds': [.5, 1.],
            'source_edges_not_welded': True},
        'atlas': {'physical_domain_m': ATLAS_DOMAIN_M, 'tip_centers_m': [-TIP_CENTER_M, TIP_CENTER_M],
            'maximum_tip_radius_m': TIP_RADIUS_M, 'visible_groups_max_span_m': .0066},
        'limitations': ['Explicit authored shading, not volumetric or recovered fur.',
                       'Source color/material sampling and renderer culling verification are caller responsibilities.',
                       'No intersection, physical motion, density, or appearance acceptance is inferred.']}
    _deadline(started)
    return result


def two_tip_atlas(seed, size=1024):
    """Return a fixed 20.8 mm-domain white RGBA silhouette atlas, top-down rows."""
    require(type(seed) is int and 0 <= seed <= 2**31-1 and type(size) is int and size == 1024,
            'Invalid bounded fin atlas settings')
    rng = np.random.default_rng(seed)
    tile_size = size//4
    yy, xx = np.mgrid[:tile_size, :tile_size].astype(np.float32)/(tile_size-1)
    height, world_x = 1-yy, (xx-.5)*ATLAS_DOMAIN_M
    result = np.full((size, size, 4), 255, dtype=np.uint8)
    result[:, :, 3] = 0
    for tile in range(16):
        alpha = np.zeros((tile_size, tile_size), dtype=np.float32)
        for center in (-TIP_CENTER_M, TIP_CENTER_M):
            tip = rng.uniform(.92, .96)
            radius = TIP_RADIUS_M*np.maximum(0, 1-height/tip)**.38
            coverage = np.clip((radius-np.abs(world_x-center))*tile_size/ATLAS_DOMAIN_M+.5, 0, 1)
            coverage *= (height > .025)&(height < tip)
            np.maximum(alpha, coverage, out=alpha)
        y, x = tile//4*tile_size, tile % 4*tile_size
        result[y:y+tile_size, x:x+tile_size, 3] = np.rint(alpha*255).astype(np.uint8)
    return result
