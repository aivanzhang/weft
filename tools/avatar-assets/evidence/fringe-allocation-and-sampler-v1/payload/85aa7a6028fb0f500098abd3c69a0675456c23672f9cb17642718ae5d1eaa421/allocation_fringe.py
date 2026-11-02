"""Experimental pure-data edge fins; not an accepted coat or production adapter."""

import itertools
import math
import time

import numpy as np


MAX_VERTICES = 100000
MAX_FACES = 100000
MAX_FINS = 4800
MAX_CANDIDATES = 32000
MAX_TEXTURE_PIXELS = 16777216
FRAME_POLICY = 'area-share-edge-fin-analytic-v1'
ATLAS_DOMAIN_M = .0208
TIP_CENTER_M = .0025
TIP_RADIUS_M = .0008

from root_sampler import repeat_root_colors


def require(value, message):
    if not value:
        raise ValueError(message)


def array(value, shape, name, kinds):
    require(isinstance(value, np.ndarray) and value.shape == shape and value.dtype.kind in kinds,
            'Invalid '+name+' array')
    require(value.dtype.kind != 'f' or np.isfinite(value).all(), 'Nonfinite '+name)
    return value


def number(value, low, high, name):
    require(type(value) in (int, float) and math.isfinite(value) and low <= value <= high,
            'Invalid '+name)
    return value


def settings_checked(value, source_export_vertices, vertices):
    fields = {'seed', 'min_fins', 'max_fins', 'length_m', 'min_spacing_m', 'min_edge_m', 'max_edge_m', 'comb'}
    require(isinstance(value, dict) and set(value) == fields, 'Unexpected fin settings')
    require(type(value['seed']) is int and 0 <= value['seed'] <= 2**31-1, 'Invalid fin seed')
    require(type(value['min_fins']) is int and type(value['max_fins']) is int and
            1 <= value['min_fins'] <= value['max_fins'] <= MAX_FINS, 'Invalid fin count')
    number(value['length_m'], .001, .04, 'fin length')
    number(value['min_spacing_m'], .0005, .05, 'root spacing')
    number(value['min_edge_m'], .001, .02, 'minimum edge span')
    number(value['max_edge_m'], value['min_edge_m'], .03, 'maximum edge span')
    number(value['comb'], 0, .35, 'fin lean')
    require(type(source_export_vertices) is int and vertices <= source_export_vertices <= MAX_VERTICES,
            'Invalid source export budget')
    require(source_export_vertices+6*value['max_fins']+4096 <= MAX_VERTICES,
            'Base plus six corners per fin and reserve exceeds 100000 vertices')
    return dict(value)


def _textures(textures):
    require(isinstance(textures, dict) and 1 <= len(textures) <= 32, 'Invalid source textures')
    seen, pixels = set(), 0
    for slot, record in textures.items():
        require(type(slot) is int and 0 <= slot <= 31 and isinstance(record, dict) and
                set(record) == {'pixels', 'factor'}, 'Invalid texture descriptor')
        image = record['pixels']
        require(isinstance(image, np.ndarray) and image.ndim == 3 and image.shape[2] in (3, 4)
                and image.dtype == np.uint8 and 1 <= image.shape[0] <= 4096 and
                1 <= image.shape[1] <= 4096, 'Source texture exceeds RGB(A)8 bounds')
        if id(image) not in seen:
            seen.add(id(image))
            pixels += image.shape[0]*image.shape[1]
        require(pixels <= MAX_TEXTURE_PIXELS, 'Source textures exceed aggregate 16MP bound')
        factor = array(record['factor'], (3,), 'basecolor factor', 'f')
        require(np.all((factor >= 0)&(factor <= 1)), 'Basecolor factor outside 0..1')


def sample_color(record, uv):
    return repeat_root_colors(record['pixels'], np.asarray(uv)[None], record['factor'])[0]


def build_edge_fringe(points, faces, corner_normals, uv0, materials, parts, fur_faces,
                      non_fur_faces, closure_faces, joints, weights, flow_directions,
                      textures, design_views, settings, *, source_export_vertices):
    """Return detached geometry/correspondence arrays; never weld or mutate source data."""
    started = time.monotonic()
    require(isinstance(points, np.ndarray) and points.ndim == 2 and 1 <= len(points) <= MAX_VERTICES,
            'Invalid or oversized source points')
    require(isinstance(faces, np.ndarray) and faces.ndim == 2 and 1 <= len(faces) <= MAX_FACES,
            'Invalid or oversized source faces')
    nv, nf = len(points), len(faces)
    settings = settings_checked(settings, source_export_vertices, nv)
    points = array(points, (nv, 3), 'points', 'f')
    faces = array(faces, (nf, 3), 'faces', 'iu')
    require(faces.min() >= 0 and faces.max() < nv and np.max(np.abs(points)) <= 10000, 'Source indices or positions exceed bounds')
    normal = array(corner_normals, (nf, 3, 3), 'corner normals', 'f')
    require(np.all(np.abs(np.linalg.norm(normal, axis=2)-1) < 1e-5), 'Require unit source normals')
    uv0 = array(uv0, (nf, 3, 2), 'UV0', 'f')
    require(np.all((uv0 >= 0)&(uv0 <= 1)), 'Source UV0 must be in 0..1')
    materials = array(materials, (nf,), 'material IDs', 'iu')
    parts = array(parts, (nf,), 'part IDs', 'iu')
    require(materials.min() >= 0 and materials.max() <= 31 and parts.min() >= 0 and parts.max() <= 255,
            'Material or part IDs exceed bounds')
    fur_faces = array(fur_faces, (nf,), 'fur eligibility', 'b')
    non_fur_faces = array(non_fur_faces, (nf,), 'non-fur exclusions', 'b')
    closure_faces = array(closure_faces, (nf,), 'authored closure exclusions', 'b')
    joints = array(joints, (nv, 4), 'joints', 'iu')
    weights = array(weights, (nv, 4), 'weights', 'f')
    require(joints.min() >= 0 and joints.max() < 64 and np.all((weights >= 0)&(weights <= 1)) and
            np.all(np.abs(weights.sum(axis=1)-1) < 1e-6), 'Invalid four-weight skin')
    flow = array(flow_directions, (nv, 3), 'flow directions', 'f')
    require(np.all(np.linalg.norm(flow, axis=1) > 1e-8), 'Zero flow direction')
    require(isinstance(design_views, np.ndarray) and design_views.ndim == 2 and
            1 <= len(design_views) <= 16, 'Require 1..16 fixed design views')
    views = array(design_views, (len(design_views), 3), 'design views', 'f').astype(np.float64)
    require(np.all(np.linalg.norm(views, axis=1) > 1e-8), 'Zero design view')
    views /= np.linalg.norm(views, axis=1)[:, None]
    _textures(textures)

    xyz = points[faces]
    geometry = np.cross(xyz[:, 1]-xyz[:, 0], xyz[:, 2]-xyz[:, 0])
    magnitude = np.linalg.norm(geometry, axis=1)
    require(np.all(magnitude > 1e-12), 'Degenerate source triangle')
    geometry /= magnitude[:, None]
    edges = faces[:, [[0, 1], [1, 2], [2, 0]]].reshape(-1, 2)
    corners = np.tile([[0, 1], [1, 2], [2, 0]], (nf, 1))
    flipped = edges[:, 0] > edges[:, 1]
    corners[flipped] = corners[flipped, ::-1]
    edges = np.sort(edges, axis=1)
    order = np.lexsort((edges[:, 1], edges[:, 0]))
    sorted_edges = edges[order]
    starts = np.flatnonzero(np.r_[True, np.any(sorted_edges[1:] != sorted_edges[:-1], axis=1)])
    counts = np.diff(np.r_[starts, len(edges)])
    shared = starts[counts == 2]
    left, right = order[shared], order[shared+1]
    f0, f1 = left//3, right//3
    endpoints = edges[left]
    c0, c1 = corners[left], corners[right]
    n = normal[f0[:, None], c0].sum(axis=1)+normal[f1[:, None], c1].sum(axis=1)
    n_length = np.linalg.norm(n, axis=1)
    n /= np.maximum(n_length, 1e-12)[:, None]
    edge_vector = points[endpoints[:, 1]]-points[endpoints[:, 0]]
    lengths = np.linalg.norm(edge_vector, axis=1)
    source_uv = uv0[f0[:, None], c0]
    reasons = np.full(len(shared), '', dtype='<U28')

    def exclude(condition, label):
        reasons[(reasons == '')&condition] = label

    exclude(~(fur_faces[f0]&fur_faces[f1])|non_fur_faces[f0]|non_fur_faces[f1], 'non_fur')
    exclude(closure_faces[f0]|closure_faces[f1], 'authored_closure')
    exclude(flipped[left] == flipped[right], 'inconsistent_winding')
    exclude(materials[f0] != materials[f1], 'material_seam')
    exclude(parts[f0] != parts[f1], 'moving_part_seam')
    exclude(np.any(source_uv != uv0[f1[:, None], c1], axis=(1, 2)), 'UV_seam')
    exclude((lengths < settings['min_edge_m'])|(lengths > settings['max_edge_m']), 'edge_span')
    exclude((n_length < 1e-8)|(np.sum(n*geometry[f0], axis=1) <= .05)|
            (np.sum(n*geometry[f1], axis=1) <= .05), 'incoherent_normals')
    lean = flow[endpoints].sum(axis=1).astype(np.float64)
    lean -= n*np.sum(lean*n, axis=1)[:, None]
    lean /= np.maximum(np.linalg.norm(lean, axis=1), 1e-12)[:, None]
    direction = n*(1-settings['comb'])+lean*settings['comb']
    direction /= np.maximum(np.linalg.norm(direction, axis=1), 1e-12)[:, None]
    fin_normal = np.cross(edge_vector, direction)
    fin_magnitude = np.linalg.norm(fin_normal, axis=1)
    exclude(fin_magnitude < 1e-8, 'degenerate_fin_plane')
    fin_normal /= np.maximum(fin_magnitude, 1e-12)[:, None]
    eligible = np.flatnonzero(reasons == '')
    score = ((1-np.abs(n[eligible]@views.T))*np.abs(fin_normal[eligible]@views.T)).max(axis=1)
    rng = np.random.default_rng(settings['seed'])
    # Share each representable face's area among its safe incident roots, not its tessellation density.
    incident = np.bincount(np.r_[f0[eligible], f1[eligible]], minlength=nf)
    face_share = np.divide(.5*magnitude, incident, out=np.zeros(nf), where=incident > 0)
    area_mass = face_share[f0[eligible]]+face_share[f1[eligible]]
    require(np.all(area_mass > 0), 'Invalid edge area mass')
    preference = .5+.5*score
    priority = -np.log(np.maximum(rng.random(len(eligible)), np.finfo(float).tiny))/(area_mass*preference)
    ranking = eligible[np.lexsort((eligible, priority))]
    roots = points[endpoints].mean(axis=1)
    spacing = settings['min_spacing_m']
    buckets, chosen, checked, spacing_rejected = {}, [], 0, 0
    offsets = list(itertools.product((-1, 0, 1), repeat=3))
    for index in ranking[:MAX_CANDIDATES]:
        checked += 1
        if checked % 128 == 0:
            require(time.monotonic()-started < 10, 'Fin builder exceeded its ten-second internal deadline')
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
    require(len(chosen) >= settings['min_fins'], 'Insufficient safe separated edge roots')
    chosen = np.asarray(chosen, dtype=np.int32)
    source_ids = endpoints[chosen]
    root_points = points[source_ids]
    positions = np.concatenate((root_points, root_points+direction[chosen, None]*settings['length_m']), axis=1)
    skin_ids = np.tile(joints[source_ids], (1, 2, 1))
    skin_weights = np.tile(weights[source_ids], (1, 2, 1)).astype(np.float64)
    skin_weights /= skin_weights.sum(axis=2)[:, :, None]
    colors = np.empty((len(chosen), 4, 3), dtype=np.float32)
    for ordinal, index in enumerate(chosen):
        slot = int(materials[f0[index]])
        require(slot in textures, 'Selected root has no exact source texture')
        values = np.array([sample_color(textures[slot], uv) for uv in source_uv[index]])
        colors[ordinal] = np.tile(values, (2, 1))
    tiles = rng.integers(0, 16, len(chosen))
    # Crop a physical atlas domain; longer anchors never stretch either visible tip.
    half_uv = lengths[chosen]/(2*ATLAS_DOMAIN_M)
    local_uv = np.zeros((len(chosen), 4, 2))
    local_uv[:, [0, 2], 0] = .5-half_uv[:, None]
    local_uv[:, [1, 3], 0] = .5+half_uv[:, None]
    local_uv[:, [0, 1], 1] = .985
    local_uv[:, [2, 3], 1] = .015
    require(np.all((local_uv >= .015)&(local_uv <= .985)), 'Physical atlas crop outside tile gutter')
    atlas_uv = (local_uv+np.column_stack((tiles % 4, tiles//4))[:, None])/4
    triangles = (np.arange(len(chosen))[:, None, None]*4+np.array([[0, 1, 3], [0, 3, 2]])).reshape(-1, 3)
    result = {'positions': positions.reshape(-1, 3).astype(np.float32), 'triangles': triangles.astype(np.int32),
              'normals': np.repeat(fin_normal[chosen, None], 4, axis=1).reshape(-1, 3).astype(np.float32),
              'uv': atlas_uv.reshape(-1, 2).astype(np.float32), 'colors': colors.reshape(-1, 3),
              'joints': skin_ids.reshape(-1, 4).astype(np.uint16), 'weights': skin_weights.reshape(-1, 4).astype(np.float32),
              'root_vertex_ids': source_ids.copy(), 'root_face_ids': f0[chosen].copy(),
              'root_corner_ids': c0[chosen].copy(), 'root_source_uv': source_uv[chosen].copy(),
              'root_centers': roots[chosen].copy(), 'outward_normals': n[chosen].copy(),
              'fin_directions': direction[chosen].copy()}
    result['report'] = {'profile': FRAME_POLICY, 'settings': settings, 'source_vertices': nv, 'source_faces': nf,
        'selected_fins': len(chosen), 'geometric_vertices': len(chosen)*4, 'triangles': len(chosen)*2,
        'source_export_vertices': source_export_vertices, 'worst_exported_fin_corners': len(chosen)*6,
        'reserve_vertices': 4096, 'projected_vertices': source_export_vertices+len(chosen)*6+4096,
        'unshared_boundary_edges': int((counts == 1).sum()), 'nonmanifold_edges': int((counts > 2).sum()),
        'exclusions': {str(label): int((reasons == label).sum()) for label in np.unique(reasons) if label},
        'eligible_edges': len(eligible), 'examined_candidates': checked, 'candidate_scan_cap': MAX_CANDIDATES,
        'spacing_rejected': spacing_rejected, 'min_outward_dot': float(np.sum(n[chosen]*direction[chosen], axis=1).min()),
        'elapsed_seconds': time.monotonic()-started, 'camera_billboarding': False, 'source_modified': False,
        'material_requirement': 'MASK 0.18, source-colored, geometric fin normals, two-sided; not GPU-verified',
        'quality_approved': False,
        'allocation': {'policy': 'eligible-face-area-share-seeded-exponential-v1',
            'represented_face_area_m2': float(np.sum(.5*magnitude[incident > 0])),
            'edge_area_mass_sum_m2': float(area_mass.sum()),
            'view_preference_bounds': [.5, 1.], 'source_edges_not_welded': True},
        'atlas': {'physical_domain_m': ATLAS_DOMAIN_M, 'tip_centers_m': [-TIP_CENTER_M, TIP_CENTER_M],
            'maximum_tip_radius_m': TIP_RADIUS_M, 'visible_groups_max_span_m': .0066},
        'color_limit': 'Exact endpoint LOD0 samples; interpolation across longer edges is not whole-fin texture matching.'}
    require(time.monotonic()-started < 10, 'Fin builder exceeded its ten-second internal deadline')
    return result


def two_tip_atlas(seed, size=1024):
    require(type(seed) is int and 0 <= seed <= 2**31-1 and size == 1024, 'Invalid bounded fin atlas settings')
    rng = np.random.default_rng(seed)
    tile_size = size//4
    yy, xx = np.mgrid[:tile_size, :tile_size].astype(np.float32)/(tile_size-1)
    height = 1-yy
    world_x = (xx-.5)*ATLAS_DOMAIN_M
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
