"""Bounded, data-only partitioning of a generated mesh; closures are authored assumptions."""

from collections import defaultdict, deque
import hashlib
import io
import json
import math
from pathlib import Path
import re
import shutil
import time

import numpy as np


MAX_VERTICES = 100000
MAX_FACES = 100000
MAX_CAPS = 1000
ROOT = Path(__file__).resolve().parents[1]


def require(value, message):
    if not value:
        raise ValueError(message)


def vector(value, low, high):
    require(isinstance(value, (list, tuple)) and len(value) == 3
            and all(type(x) in (int, float) and math.isfinite(x) and low <= x <= high for x in value),
            'Invalid finite three-coordinate vector')
    return list(map(float, value))


def validate_config(config):
    required = {'schema_version', 'source_sha256', 'part_name', 'height_meters', 'selector',
                'island_face_limit', 'fill_face_limit', 'closure_colors_srgb', 'preview_offset_m', 'rights'}
    require(isinstance(config, dict) and required <= set(config)
            and not set(config)-required-{'split_pinches', 'repair_contact_uvs', 'contact_repair_assumptions',
                                         'contact_color_overrides'}
            and type(config['schema_version']) is int and config['schema_version'] == 1,
            'Invalid partition config fields')
    require(type(config.get('split_pinches', False)) is bool, 'Invalid pinch repair setting')
    require(type(config.get('repair_contact_uvs', False)) is bool, 'Invalid contact UV repair setting')
    if config.get('repair_contact_uvs', False):
        require(isinstance(config.get('contact_repair_assumptions'), str)
                and 1 <= len(config['contact_repair_assumptions']) <= 2048,
                'Contact UV repair requires explicit part-palette and +Y-back assumptions')
    overrides = config.get('contact_color_overrides', [])
    require(isinstance(overrides, list) and len(overrides) <= 4, 'At most four contact-color overrides are supported')
    require(not overrides or config.get('repair_contact_uvs', False), 'Contact overrides require enabled UV repair')
    for override in overrides:
        require(isinstance(override, dict) and set(override) == {'name', 'part', 'min', 'max', 'min_mean',
                'max_chroma', 'min_normal_y', 'palette', 'min_faces', 'max_faces'}, 'Invalid contact override fields')
        require(isinstance(override['name'], str) and re.fullmatch('[A-Za-z][A-Za-z0-9_-]{0,47}', override['name']),
                'Invalid contact override name')
        require(type(override['part']) is int and override['part'] in (0, 1), 'Invalid override part')
        lower, upper = vector(override['min'], -1, 3), vector(override['max'], -1, 3)
        require(all(a < b for a, b in zip(lower, upper)), 'Invalid contact override box')
        for key, low, high in [('min_mean', 0, 1), ('max_chroma', 0, 1), ('min_normal_y', -1, 1)]:
            value = override[key]
            require(type(value) in (float, int) and math.isfinite(value) and low <= value <= high,
                    'Invalid contact override threshold')
        require(type(override['min_faces']) is int and type(override['max_faces']) is int
                and 0 <= override['min_faces'] <= override['max_faces'] <= 1000, 'Invalid contact override face bounds')
        if override['palette'] != 'part':
            vector(override['palette'], 0, 1)
    require(isinstance(config['source_sha256'], str) and re.fullmatch('[0-9a-f]{64}', config['source_sha256']),
            'A pinned source hash is required')
    require(isinstance(config['part_name'], str) and re.fullmatch('[A-Za-z][A-Za-z0-9_-]{0,47}', config['part_name']),
            'Invalid part name')
    require(type(config['height_meters']) in (int, float) and .1 <= config['height_meters'] <= 5,
            'Invalid authored height')
    require(isinstance(config['rights'], str) and 1 <= len(config['rights']) <= 4096,
            'Explicit provenance and authored closure disclosure required')
    for key, limit in [('island_face_limit', 100), ('fill_face_limit', 500)]:
        require(type(config[key]) is int and 0 <= config[key] <= limit, 'Invalid enclosed-island limit')
    colors = config['closure_colors_srgb']
    require(isinstance(colors, list) and len(colors) == 2, 'Require body and detached-part closure colors')
    for color in colors:
        vector(color, 0, 1)
    vector(config['preview_offset_m'], -1, 1)
    selector = config['selector']
    require(isinstance(selector, dict) and set(selector) == {
        'regions', 'dominant_channel', 'dominant_ratio', 'expand_meters', 'min_faces', 'max_faces'},
        'Invalid color/spatial selector')
    require(type(selector['dominant_channel']) is int and 0 <= selector['dominant_channel'] <= 2,
            'Invalid dominant channel')
    require(type(selector['dominant_ratio']) in (int, float) and 1.01 <= selector['dominant_ratio'] <= 5,
            'Invalid color dominance ratio')
    require(type(selector['expand_meters']) in (int, float) and 0 <= selector['expand_meters'] <= .02,
            'Invalid selector expansion')
    require(type(selector['min_faces']) is int and type(selector['max_faces']) is int
            and 1 <= selector['min_faces'] <= selector['max_faces'] <= MAX_FACES,
            'Invalid selected-face limits')
    regions = selector['regions']
    require(isinstance(regions, list) and 1 <= len(regions) <= 4, 'Require 1–4 ellipsoid regions')
    for region in regions:
        require(isinstance(region, dict) and set(region) == {'center', 'radius'}, 'Invalid ellipsoid')
        vector(region['center'], -1, 3)
        vector(region['radius'], .001, 3)
    return json.loads(json.dumps(config, allow_nan=False))


def select_vertices(points, colors, selector):
    points, colors = np.asarray(points, dtype=float), np.asarray(colors, dtype=float)
    require(points.ndim == 2 and points.shape[1:] == (3,) and 1 <= len(points) <= MAX_VERTICES
            and colors.shape == points.shape and np.isfinite(points).all() and np.isfinite(colors).all()
            and ((colors >= 0) & (colors <= 1)).all(), 'Invalid partition point/color arrays')
    span = np.ptp(points, axis=0)
    require(np.all(span > 1e-8), 'Partition geometry has degenerate bounds')
    normalized = (points-points.min(axis=0))/span
    inside = np.zeros(len(points), dtype=bool)
    for region in selector['regions']:
        inside |= (((normalized-region['center'])/region['radius'])**2).sum(axis=1) <= 1
    channel = selector['dominant_channel']
    selected = inside & (colors[:, channel] > selector['dominant_ratio']*np.max(
        colors[:, [i for i in range(3) if i != channel]], axis=1))
    distance = selector['expand_meters']
    if distance and selected.any():
        # Bounded spatial buckets expand from original seeds, never recursively flood the surface.
        buckets = defaultdict(list)
        cells = np.floor(points/distance).astype(np.int64)
        for index in np.flatnonzero(selected):
            buckets[tuple(cells[index])].append(int(index))
        comparisons = 0
        for index in np.flatnonzero(inside & ~selected):
            cell = cells[index]
            found = False
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    for dz in (-1, 0, 1):
                        candidates = buckets.get(tuple(cell+(dx, dy, dz)), [])
                        comparisons += len(candidates)
                        require(comparisons <= 5000000, 'Selector neighborhood work exceeds bound')
                        if candidates and np.any(((points[candidates]-points[index])**2).sum(axis=1) <= distance**2):
                            found = True
            selected[index] = found
    return selected


def contact_color_candidate(part, rgb, centroid_y, normal_y, part_y_mid, *, channel, ratio):
    """Only call inside the bounded contact band; front details and light claws stay protected."""
    dominant = float(rgb[channel])
    others = [float(rgb[i]) for i in range(3) if i != channel]
    if part == 1:
        return (centroid_y >= part_y_mid and normal_y >= -.1 and
                sum(rgb)/3 < .15 and max(rgb)-min(rgb) < .1 and dominant <= ratio*max(others))
    return dominant > ratio*max(others) and dominant-min(others) > .02 and dominant > .05


def contact_override_matches(rule, part, rgb, normalized_position, normal_y):
    return (part == rule['part'] and all(a <= x <= b for x, a, b in
            zip(normalized_position, rule['min'], rule['max'])) and float(np.mean(rgb)) >= rule['min_mean']
            and max(rgb)-min(rgb) <= rule['max_chroma'] and normal_y >= rule['min_normal_y'])


def _edges(faces):
    result = defaultdict(list)
    for fid, face in enumerate(faces):
        for a, b in zip(face, np.roll(face, -1)):
            result[tuple(sorted((int(a), int(b))))].append(fid)
    return result


def _components(mask, adjacency):
    unseen = np.asarray(mask, dtype=bool).copy()
    result = []
    for first in range(len(unseen)):
        if not unseen[first]:
            continue
        unseen[first] = False
        group, queue = [], deque([first])
        while queue:
            face = queue.popleft()
            group.append(face)
            for other in adjacency[face]:
                if unseen[other]:
                    unseen[other] = False
                    queue.append(other)
        result.append(group)
    return sorted(result, key=lambda group: (-len(group), min(group)))


def _orient_faces(faces):
    """Resolve local winding defects without changing any triangle's corner values."""
    edges = _edges(faces)
    adjacency = [[] for _ in faces]
    for edge, incident in edges.items():
        if len(incident) != 2:
            continue
        directions = []
        for fid in incident:
            pairs = set(zip(faces[fid], np.roll(faces[fid], -1)))
            directions.append(edge in pairs)
        parity = int(directions[0] == directions[1])
        left, right = incident
        adjacency[left].append((right, parity))
        adjacency[right].append((left, parity))
    flips = np.full(len(faces), -1, dtype=np.int8)
    for first in range(len(faces)):
        if flips[first] >= 0:
            continue
        flips[first] = 0
        queue, group = deque([first]), []
        while queue:
            face = queue.popleft()
            group.append(face)
            for other, parity in adjacency[face]:
                expected = flips[face] ^ parity
                if flips[other] < 0:
                    flips[other] = expected
                    queue.append(other)
                else:
                    require(flips[other] == expected, 'Unsupported non-orientable source surface')
        if int(flips[group].sum()) > len(group)/2:
            flips[group] ^= 1
    require(int(flips.sum()) <= 64, 'Source winding defects exceed the 64-face local repair bound')
    result = faces.copy()
    mask = flips.astype(bool)
    result[mask] = result[mask][:, [0, 2, 1]]
    return result, mask


def triangulate_boundary_3d(coordinates, boundary_normals=None, *, deadline=None):
    """Minimum-area ordered 3D disk; pinches and degenerate triangles remain unsupported."""
    points = np.asarray(coordinates, dtype=float)
    require(points.ndim == 2 and points.shape[1:] == (3,) and 3 <= len(points) <= 256
            and np.isfinite(points).all(), 'Closure requires 3–256 finite ordered 3D points')
    require(len(np.unique(np.round(points, 8), axis=0)) == len(points),
            'Unsupported pinched boundary: repeated 3D coordinates need local label/fan repair')
    n = len(points)
    normals = None if boundary_normals is None else np.asarray(boundary_normals, dtype=float)
    require(normals is None or (normals.shape == points.shape and np.isfinite(normals).all()),
            'Invalid boundary normals')
    cost = np.full((n, n), np.inf)
    splits = np.full((n, n), -1, dtype=int)
    for i in range(n-1):
        cost[i, i+1] = 0
    for gap in range(2, n):
        require(deadline is None or time.monotonic() <= deadline, 'Closure solver exceeded 30-second budget')
        for i in range(n-gap):
            j = i+gap
            mids = np.arange(i+1, j)
            cross = np.cross(points[mids]-points[i], points[j]-points[i])
            lengths = np.linalg.norm(cross, axis=1)
            areas = lengths*.5
            penalties = np.ones(len(mids))
            if normals is not None:
                expected = normals[mids]+normals[i]+normals[j]
                norm = np.linalg.norm(expected, axis=1)
                dot = np.sum(cross*expected, axis=1)/np.maximum(lengths*norm, 1e-30)
                penalties += .15*(1-np.clip(dot, -1, 1))
            candidates = cost[i, mids]+cost[mids, j]+areas*penalties
            candidates[lengths <= 1e-12] = np.inf
            best = int(np.argmin(candidates))
            cost[i, j], splits[i, j] = candidates[best], mids[best]
    require(math.isfinite(cost[0, -1]), 'No nondegenerate bounded 3D disk triangulation exists')
    triangles, stack = [], [(0, n-1)]
    while stack:
        i, j = stack.pop()
        if j-i < 2:
            continue
        k = int(splits[i, j])
        triangles.append((i, k, j))
        stack.extend([(i, k), (k, j)])
    require(len(triangles) == n-2, 'Incomplete 3D triangulation')
    edges = _edges(np.asarray(triangles))
    boundary = {tuple(sorted((i, (i+1) % n))) for i in range(n)}
    require({e for e, incident in edges.items() if len(incident) == 1} == boundary
            and all(len(v) in (1, 2) for v in edges.values()), '3D closure boundary coverage differs')
    return np.asarray(triangles, dtype=np.int32)


def triangles_intersect_3d(a, b, epsilon=1e-8):
    """Screen proper intersections and coplanar overlap, not mere corner/edge contact."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if np.any(a.max(0) < b.min(0)-epsilon) or np.any(b.max(0) < a.min(0)-epsilon):
        return False
    na, nb = np.cross(a[1]-a[0], a[2]-a[0]), np.cross(b[1]-b[0], b[2]-b[0])
    la, lb = np.linalg.norm(na), np.linalg.norm(nb)
    require(la > 1e-12 and lb > 1e-12, 'Intersection input triangle is degenerate')
    if abs(float(np.dot(na, nb)/(la*lb))) > 1-1e-8 and np.max(np.abs((b-a[0])@na/la)) <= epsilon:
        axes = [i for i in range(3) if i != int(np.argmax(np.abs(na)))]
        aa, bb = a[:, axes], b[:, axes]
        def cross2(x, y):
            return float(x[0]*y[1]-x[1]*y[0])
        def inside(point, triangle):
            signs = [cross2(y-x, point-x) for x, y in zip(triangle, np.roll(triangle, -1, axis=0))]
            return min(signs) > epsilon or max(signs) < -epsilon
        if (inside(aa.mean(0), bb) or inside(bb.mean(0), aa)
                or any(inside(point, bb) for point in aa) or any(inside(point, aa) for point in bb)):
            return True
        for p, q in zip(aa, np.roll(aa, -1, axis=0)):
            for r, s in zip(bb, np.roll(bb, -1, axis=0)):
                den = cross2(q-p, s-r)
                if abs(den) > 1e-15:
                    t, u = cross2(r-p, s-r)/den, cross2(r-p, q-p)/den
                    if epsilon < t < 1-epsilon and epsilon < u < 1-epsilon:
                        return True
        return False
    for source, target in ((a, b), (b, a)):
        e1, e2 = target[1]-target[0], target[2]-target[0]
        for p, q in zip(source, np.roll(source, -1, axis=0)):
            direction = q-p
            h = np.cross(direction, e2)
            denominator = float(np.dot(e1, h))
            if abs(denominator) <= 1e-15:
                continue
            offset = p-target[0]
            u = float(np.dot(offset, h)/denominator)
            cross = np.cross(offset, e1)
            v = float(np.dot(direction, cross)/denominator)
            t = float(np.dot(e2, cross)/denominator)
            if epsilon < t < 1-epsilon and u >= -epsilon and v >= -epsilon and u+v <= 1+epsilon:
                return True
    return False


def closure_intersections(points, faces, caps, labels, *, deadline=None):
    points, faces, caps = np.asarray(points), np.asarray(faces), np.asarray(caps)
    require(len(caps) <= MAX_CAPS and len(faces)+len(caps) <= MAX_FACES, 'Intersection work exceeds geometry limits')
    source_triangles, cap_triangles = points[faces], points[caps]
    source_low, source_high = source_triangles.min(1), source_triangles.max(1)
    cap_low, cap_high = cap_triangles.min(1), cap_triangles.max(1)
    cap_hits, source_hits, pairs = [], [], 0
    for i, triangle in enumerate(cap_triangles):
        require(deadline is None or time.monotonic() <= deadline, 'Closure intersection budget exhausted')
        part = labels[caps[i, 0]]
        candidates = np.flatnonzero(np.all(cap_low <= cap_high[i]+1e-8, axis=1)
                                    & np.all(cap_high >= cap_low[i]-1e-8, axis=1))
        for j in candidates:
            if j <= i or labels[caps[j, 0]] != part or set(caps[i]) & set(caps[j]):
                continue
            pairs += 1
            require(pairs <= 200000, 'Closure intersection candidate bound exceeded')
            if triangles_intersect_3d(triangle, cap_triangles[j]):
                cap_hits.append([i, int(j)])
        candidates = np.flatnonzero(np.all(source_low <= cap_high[i]+1e-8, axis=1)
                                    & np.all(source_high >= cap_low[i]-1e-8, axis=1))
        for j in candidates:
            if labels[faces[j, 0]] != part or set(caps[i]) & set(faces[j]):
                continue
            pairs += 1
            require(pairs <= 200000, 'Closure intersection candidate bound exceeded')
            if triangles_intersect_3d(triangle, source_triangles[j]):
                source_hits.append([i, int(j)])
    return {'cap_cap_intersections': cap_hits, 'cap_source_intersections': source_hits,
            'tested_candidate_pairs': pairs, 'shared_index_corner_or_edge_pairs_excluded': True,
            'opposite_parts_excluded': 'The two closure surfaces coincide at rest and separate with their respective parts',
            'scope': 'Bounded triangle intersection screen, not volumetric or physical validation'}


def split_pinch_loops(points, loops):
    """Split geometric point-touch cycles; zero-area two-edge spurs receive no triangles."""
    pending, simple, aliases, omitted = list(loops), [], {}, []
    while pending:
        loop = pending.pop(0)
        seen = {}
        for position, vertex in enumerate(loop):
            coordinate = tuple(np.round(points[vertex], 8))
            if coordinate in seen:
                first = seen[coordinate]
                keep, discard = loop[first], vertex
                require(np.linalg.norm(points[keep]-points[discard]) <= 1e-8,
                        'Pinch normalization would move an original surface')
                aliases[discard] = keep
                cycles = [loop[first:position], loop[position:]+loop[:first]]
                for cycle in cycles:
                    if len(cycle) >= 3:
                        pending.append(cycle)
                    else:
                        omitted.append(cycle)
                break
            seen[coordinate] = position
        else:
            simple.append(loop)
    def root(vertex):
        seen = set()
        while vertex in aliases:
            require(vertex not in seen, 'Cyclic pinch aliases')
            seen.add(vertex)
            vertex = aliases[vertex]
        return vertex
    aliases = {vertex: root(vertex) for vertex in aliases}
    simple = [[root(vertex) for vertex in loop] for loop in simple]
    return simple, aliases, omitted


def partition_topology(points, faces, selected_vertices, island_limit=40, fill_limit=200, *, split_pinches=False):
    """Preserve every source face while separating labels and nonmanifold vertex fans."""
    points, faces, selected_vertices = np.asarray(points), np.asarray(faces), np.asarray(selected_vertices)
    require(points.ndim == 2 and points.shape[1:] == (3,) and 1 <= len(points) <= MAX_VERTICES
            and np.isfinite(points).all(), 'Invalid source positions')
    require(faces.ndim == 2 and faces.shape[1:] == (3,) and 1 <= len(faces) <= MAX_FACES
            and np.issubdtype(faces.dtype, np.integer) and faces.min() >= 0 and faces.max() < len(points),
            'Invalid source triangles')
    require(selected_vertices.shape == (len(points),) and selected_vertices.dtype == bool,
            'Invalid selected vertex mask')
    require(type(island_limit) is int and 0 <= island_limit <= 100
            and type(fill_limit) is int and 0 <= fill_limit <= 500, 'Invalid island bounds')
    _, inverse = np.unique(np.round(points, 6), axis=0, return_inverse=True)
    welded = inverse[faces]
    require(all(len(set(face)) == 3 for face in welded), 'Welding collapsed a source face')
    edges = _edges(welded)
    adjacency = [set() for _ in faces]
    for incident in edges.values():
        require(len(incident) <= 8, 'Source edge valence exceeds repair profile')
        for face in incident:
            adjacency[face].update(i for i in incident if i != face)
    initial = selected_vertices[faces].all(axis=1)
    foreground = _components(initial, adjacency)
    require(foreground and len(foreground[0]) < len(faces), 'Selector must identify a nonempty proper part')
    labels = np.zeros(len(faces), dtype=bool)
    labels[foreground[0]] = True
    background = _components(~labels, adjacency)
    require(background, 'No body faces remain')
    filled = background[1:]
    require(all(len(group) <= island_limit for group in filled) and sum(map(len, filled)) <= fill_limit,
            'Enclosed surface islands exceed authored cleanup bounds')
    for group in filled:
        labels[group] = True
    # Only two-face edges connect corner fans; triple edges and label cuts become explicit seams.
    parents = np.arange(len(faces)*3, dtype=np.int64)
    def root(index):
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = int(parents[index])
        return index
    def join(a, b):
        a, b = root(a), root(b)
        if a != b:
            parents[max(a, b)] = min(a, b)
    for edge, incident in edges.items():
        if len(incident) != 2 or labels[incident[0]] != labels[incident[1]]:
            continue
        left, right = incident
        for vertex in edge:
            join(left*3+int(np.flatnonzero(welded[left] == vertex)[0]),
                 right*3+int(np.flatnonzero(welded[right] == vertex)[0]))
    roots = np.asarray([root(i) for i in range(len(parents))])
    unique, first, corner_inverse = np.unique(roots, return_index=True, return_inverse=True)
    output_faces = corner_inverse.reshape((-1, 3))
    output_points = points[faces.ravel()[first]].copy()
    vertex_labels = labels[first//3].astype(np.int8)
    require(len(output_points) <= MAX_VERTICES, 'Partition vertex splitting exceeds bound')
    output_faces, flipped = _orient_faces(output_faces)
    source_corners = faces.copy()
    source_corners[flipped] = source_corners[flipped][:, [0, 2, 1]]
    split_edges = _edges(output_faces)
    require(all(len(incident) <= 2 for incident in split_edges.values()),
            'Unsupported: vertex fan splitting did not resolve nonmanifold edges')
    boundary = []
    for face in output_faces:
        for a, b in zip(face, np.roll(face, -1)):
            if len(split_edges[tuple(sorted((int(a), int(b))))]) == 1:
                boundary.append((int(a), int(b)))
    outgoing, incoming = defaultdict(list), defaultdict(list)
    for a, b in boundary:
        outgoing[a].append(b)
        incoming[b].append(a)
    require(set(outgoing) == set(incoming) and all(len(outgoing[v]) == len(incoming[v]) == 1 for v in outgoing),
            'Unsupported: boundary fans are open, branched or inconsistently wound')
    unseen, loops = set(outgoing), []
    while unseen:
        start = min(unseen)
        loop, current = [], start
        while current in unseen:
            unseen.remove(current)
            loop.append(current)
            current = outgoing[current][0]
        require(current == start and len(loop) >= 3, 'Unsupported degenerate boundary loop')
        require(len(set(vertex_labels[loop])) == 1, 'Boundary crosses part labels')
        loops.append(loop[::-1])
    pinch_report = {'enabled': split_pinches, 'aliases_pre_normalization_vertex_ids': {}, 'omitted_zero_area_spurs': []}
    if split_pinches:
        loops, aliases, omitted = split_pinch_loops(output_points, loops)
        require(all(vertex_labels[a] == vertex_labels[b] for a, b in aliases.items()), 'Pinch alias bridges parts')
        remap = np.arange(len(output_points))
        for a, b in aliases.items():
            remap[a] = b
        output_faces = remap[output_faces]
        require(all(len(set(face)) == 3 for face in output_faces), 'Pinch repair collapsed an original face')
        used = np.unique(output_faces)
        compact = np.full(len(output_points), -1, dtype=int)
        compact[used] = np.arange(len(used))
        output_points, vertex_labels = output_points[used], vertex_labels[used]
        output_faces = compact[output_faces]
        loops = [[int(compact[v]) for v in loop] for loop in loops]
        repaired_edges = _edges(output_faces)
        require(all(len(ids) <= 2 for ids in repaired_edges.values()), 'Pinch repair created a nonmanifold edge')
        expected = {edge for edge, ids in repaired_edges.items() if len(ids) == 1}
        actual = [tuple(sorted((a, b))) for loop in loops for a, b in zip(loop, np.roll(loop, -1))]
        require(set(actual) == expected and len(actual) == len(expected), 'Pinch repair changed boundary-edge coverage')
        pinch_report.update(aliases_pre_normalization_vertex_ids=aliases, omitted_zero_area_spurs=omitted,
                            remaining_boundary_loops=len(loops), surface_positions_unchanged=True,
                            limitation='Inherited coincident point-touch geometry is not certified manifold or watertight')
    require(sum(len(loop)-2 for loop in loops) <= MAX_CAPS, 'Closures exceed 1000 triangles')
    return {'points': output_points, 'faces': output_faces, 'vertex_labels': vertex_labels,
            'face_labels': labels.astype(np.int8), 'loops': loops,
            'source_vertex_per_corner': source_corners, 'report': {
                'source_faces': len(faces), 'source_vertices': len(points),
                'welded_positions': len(np.unique(inverse)),
                'source_open_edges': sum(len(v) == 1 for v in edges.values()),
                'source_nonmanifold_edges': sum(len(v) > 2 for v in edges.values()),
                'initial_selected_faces': int(initial.sum()), 'initial_foreground_component_sizes': list(map(len, foreground)),
                'filled_enclosed_island_sizes': list(map(len, filled)),
                'part_faces': int(labels.sum()), 'body_faces': int((~labels).sum()),
                'split_vertices': len(output_points), 'boundary_loops': len(loops),
                'boundary_loop_sizes': list(map(len, loops)),
                'predicted_closure_triangles': sum(len(loop)-2 for loop in loops),
                'pinch_repair': pinch_report,
                'reoriented_source_face_ids': np.flatnonzero(flipped).tolist(),
                'all_original_faces_preserved': True,
                'geometry_position_error_max_m': float(np.abs(output_points[output_faces]-points[source_corners]).max())}}


def prepare(source, config, out):
    from PIL import Image
    from avatar_assets.three_d import _GLB
    config = validate_config(config)
    glb = _GLB(source)
    require(glb.hash == config['source_sha256'], 'Source hash differs from authorized input')
    doc = glb.doc
    require(not doc.get('skins') and not doc.get('animations') and len(doc.get('meshes', [])) == 1,
            'Partitioning requires one unrigged mesh')
    require(len(doc['meshes'][0]['primitives']) == 1, 'Initial partition profile requires one source primitive')
    require(len(doc.get('nodes', [])) == 1 and all(k not in doc['nodes'][0] for k in ('matrix', 'translation', 'rotation', 'scale')),
            'Initial partition profile requires an identity mesh node')
    primitive = doc['meshes'][0]['primitives'][0]
    attrs = primitive['attributes']
    points = np.asarray(glb.accessor(attrs['POSITION'], 'VEC3', (5126,)))[:, [0, 2, 1]]
    points[:, 1] *= -1
    original_low, original_high = points.min(axis=0), points.max(axis=0)
    scale = config['height_meters']/(original_high[2]-original_low[2])
    offset = np.array([(original_high[0]+original_low[0])*.5, (original_high[1]+original_low[1])*.5, original_low[2]])
    points = ((points-offset)*scale).astype(np.float32)
    normals = np.asarray(glb.accessor(attrs['NORMAL'], 'VEC3', (5126,)))[:, [0, 2, 1]]
    normals[:, 1] *= -1
    uv = np.asarray(glb.accessor(attrs['TEXCOORD_0'], 'VEC2', (5126,)))
    faces = np.asarray(glb.accessor(primitive['indices'], 'SCALAR', (5121, 5123, 5125), count_limit=300000), dtype=np.int32).reshape((-1, 3))
    material = doc['materials'][primitive['material']]
    tex = material['pbrMetallicRoughness']['baseColorTexture']
    require(tex.get('texCoord', 0) == 0 and not tex.get('extensions'), 'Require untransformed UV0 base color')
    image = doc['images'][doc['textures'][tex['index']]['source']]
    view = doc['bufferViews'][image['bufferView']]
    start = view.get('byteOffset', 0)
    encoded = glb.binary[start:start+view['byteLength']]
    with Image.open(io.BytesIO(encoded)) as im:
        require(im.width*im.height <= 16777216 and getattr(im, 'n_frames', 1) == 1, 'Source texture exceeds 16MP')
        rgb_image = im.convert('RGB')
        pixels = np.asarray(rgb_image)
        height, width = pixels.shape[:2]
        x = np.minimum(width-1, (np.mod(uv[:, 0], 1)*width).astype(int))
        y = height-1-np.minimum(height-1, (np.mod(1-uv[:, 1], 1)*height).astype(int))
        colors = pixels[y, x].astype(float)/255
        grid = 128
        means = np.asarray(rgb_image.resize((grid, grid), Image.Resampling.BOX)).astype(float)/255
        squared = rgb_image.point([round(value*value/255) for value in range(256)]*3)
        second_moments = np.asarray(squared.resize((grid, grid), Image.Resampling.BOX)).astype(float)/255
        variances = np.maximum(0, second_moments-means**2)
        del squared
        def choose_palette_patch(target):
            distances = ((means-np.asarray(target))**2).sum(axis=2)+4*variances.sum(axis=2)
            candidates = np.argsort(distances.ravel())[:32]
            choices = []
            for candidate in candidates:
                row, column = divmod(int(candidate), grid)
                x0, x1 = int(column*width/grid)+2, int((column+1)*width/grid)-2
                y0, y1 = int(row*height/grid)+2, int((row+1)*height/grid)-2
                require(x1 > x0 and y1 > y0, 'Texture too small for authored palette patches')
                tile = pixels[y0:y1, x0:x1].astype(float)/255
                mean, variance = tile.mean(axis=(0, 1)), tile.var(axis=(0, 1))
                score = float(((mean-target)**2).sum()+variance.sum()*4)
                choices.append((score, x0, x1, y0, y1, mean, variance))
            score, x0, x1, y0, y1, mean, variance = min(choices, key=lambda row: row[0])
            require(float(variance.sum()) < .01, 'No sufficiently uniform source palette patch for closure')
            return {'target_srgb': list(target), 'mean_srgb': mean.tolist(),
                'variance_srgb': variance.tolist(), 'score': score, 'source_pixel_rect': [x0, y0, x1, y1],
                'uv_rect_blender': [(x0+.5)/width, 1-(y1-.5)/height,
                                    (x1-.5)/width, 1-(y0+.5)/height]}
        palette_patches = [choose_palette_patch(target) for target in config['closure_colors_srgb']]
    selected = select_vertices(points, colors, config['selector'])
    topology = partition_topology(points, faces, selected, config['island_face_limit'], config['fill_face_limit'],
                                  split_pinches=config.get('split_pinches', False))
    require(config['selector']['min_faces'] <= topology['report']['part_faces'] <= config['selector']['max_faces'],
            'Selected part face count differs from configured bounds')
    uv[:, 1] = 1-uv[:, 1]
    part_patch = None
    if config.get('repair_contact_uvs', False):
        channel, ratio = config['selector']['dominant_channel'], config['selector']['dominant_ratio']
        dominant = selected & (colors[:, channel] > ratio*np.max(colors[:, [i for i in range(3) if i != channel]], axis=1))
        dominant &= colors[:, channel] > .15
        require(int(dominant.sum()) >= 100, 'Too few dominant-channel source samples for hidden-surface palette')
        part_target = np.median(colors[dominant], axis=0)
        part_patch = choose_palette_patch(part_target)
        part_patch.update(authored_target=f'Measured dominant-channel source palette for {config["part_name"]}; not other-part contact pixels',
                          source_sample_vertices=int(dominant.sum()), dominant_channel=channel, dominant_ratio=ratio)
    boundary_edges = _edges(topology['faces'])
    loop_palettes = []
    for loop in topology['loops']:
        adjacent = sorted({fid for a, b in zip(loop, np.roll(loop, -1))
                           for fid in boundary_edges[tuple(sorted((int(a), int(b))))]})
        samples = colors[topology['source_vertex_per_corner'][adjacent]].reshape((-1, 3))
        target = np.median(samples, axis=0)
        patch = (dict(part_patch) if part_patch is not None and len(loop) >= 8
                 and topology['vertex_labels'][loop[0]] == 1 else choose_palette_patch(target))
        patch.update(boundary_rgb_median=target.tolist(), boundary_rgb_min=samples.min(0).tolist(),
                     boundary_rgb_max=samples.max(0).tolist(), adjacent_original_faces=adjacent)
        loop_palettes.append(patch)
    arrays = {key: topology[key] for key in ('points', 'faces', 'vertex_labels', 'face_labels')}
    arrays.update(uv_per_corner=uv[topology['source_vertex_per_corner']],
                  normal_per_corner=normals[topology['source_vertex_per_corner']])
    arrays['face_material_indices'] = topology['face_labels'].astype(np.int32)
    uv_repair = {'enabled': config.get('repair_contact_uvs', False), 'faces': [],
                 'geometry_changed': False, 'original_image_bytes_changed': False}
    if config.get('repair_contact_uvs', False):
        before = arrays['uv_per_corner'].copy()
        _, inverse = np.unique(np.round(points, 6), axis=0, return_inverse=True)
        source_edges = _edges(inverse[faces])
        face_labels = topology['face_labels']
        contact = {fid for incident in source_edges.values() if len(set(face_labels[incident])) > 1 for fid in incident}
        band = set(contact)
        for incident in source_edges.values():
            if any(fid in contact for fid in incident):
                band.update(incident)
        part_points = points[np.unique(faces[face_labels == 1])]
        y_mid = float((part_points[:, 1].min()+part_points[:, 1].max())*.5)
        body_loops = [(loop, patch) for loop, patch in zip(topology['loops'], loop_palettes)
                      if topology['vertex_labels'][loop[0]] == 0 and set(patch['adjacent_original_faces']) & band]
        require(body_loops, 'No body contact palette available')
        overrides = config.get('contact_color_overrides', [])
        override_patches = [part_patch if rule['palette'] == 'part' else choose_palette_patch(rule['palette'])
                            for rule in overrides]
        override_counts = [0]*len(overrides)
        world_low, world_span = points.min(0), np.ptp(points, axis=0)
        for fid in sorted(band):
            source_ids = topology['source_vertex_per_corner'][fid]
            rgb, center = colors[source_ids].mean(0), points[source_ids].mean(0)
            normal = normals[source_ids].mean(0)
            part = int(face_labels[fid])
            protected = part == 1 and (float(center[1]) < y_mid or float(normal[1]) < -.1)
            matching = [i for i, rule in enumerate(overrides) if not protected
                        and contact_override_matches(rule, part, rgb, (center-world_low)/world_span, float(normal[1]))]
            ordinary = contact_color_candidate(part, rgb, float(center[1]), float(normal[1]), y_mid,
                                               channel=config['selector']['dominant_channel'], ratio=config['selector']['dominant_ratio'])
            if not ordinary and not matching:
                continue
            if matching:
                patch = override_patches[matching[-1]]
                override_counts[matching[-1]] += 1
            elif part == 1:
                patch = part_patch
            else:
                _, patch = min(body_loops, key=lambda item: float(np.sum(
                    (topology['points'][item[0]]-center)**2, axis=1).min()))
            rect = patch['uv_rect_blender']
            replacement = np.array([[.1, .1], [.9, .1], [.1, .9]])
            replacement = replacement*np.array([rect[2]-rect[0], rect[3]-rect[1]])+rect[:2]
            arrays['uv_per_corner'][fid] = replacement
            arrays['face_material_indices'][fid] = 2+part
            uv_repair['faces'].append({'source_face_id': fid, 'part': part, 'original_mean_rgb': rgb.tolist(),
                'reason': 'explicit:'+overrides[matching[-1]]['name'] if matching else 'dominant-contact-fringe',
                'before_uv_sha256': hashlib.sha256(before[fid].astype('<f8').tobytes()).hexdigest(),
                'after_uv_sha256': hashlib.sha256(replacement.astype('<f8').tobytes()).hexdigest(),
                'uv_palette_patch': patch['uv_rect_blender']})
            require(len(uv_repair['faces']) <= 1000, 'Contact UV repair exceeds 1000 original faces')
        for rule, count in zip(overrides, override_counts):
            require(rule['min_faces'] <= count <= rule['max_faces'], 'Contact override selection outside bounds: '+rule['name'])
        uv_repair.update(contact_faces=len(contact), one_ring_band_faces=len(band), companion_front_half_threshold_y=y_mid,
            explicit_overrides=[{'rule': rule, 'selected_faces': count, 'palette_patch': patch}
                                for rule, count, patch in zip(overrides, override_counts, override_patches)],
            before_uv_array_sha256=hashlib.sha256(before.astype('<f8').tobytes()).hexdigest(),
            after_uv_array_sha256=hashlib.sha256(arrays['uv_per_corner'].astype('<f8').tobytes()).hexdigest(),
            protection='Detached-part front half/front-facing surfaces stay protected; light details stay protected except explicit bounded override regions inside the contact band',
            authored_assumption=config['contact_repair_assumptions'],
            space='Calibrated Blender +Z up/-Y front/+Y back; not automatic hidden-texture recovery')
    np.savez_compressed(out/'partition-arrays.npz', **arrays)
    result = {'schema_version': 1, 'source_sha256': glb.hash, 'source_material_index': primitive['material'],
              'basecolor_original_encoded_sha256': hashlib.sha256(encoded).hexdigest(),
              'calibration': {'offset_before_scale': offset.tolist(), 'uniform_scale': scale,
                              'height_meters': config['height_meters'], 'axes': 'Blender +Z up, -Y forward'},
              'palette_patches': palette_patches, 'loop_palette_patches': loop_palettes,
              'contact_uv_repair': uv_repair, 'part_authored_palette_patch': part_patch,
              'selected_source_vertices': int(selected.sum()),
              'topology': topology['report'], 'loops': topology['loops']}
    (out/'partition-plan.json').write_text(json.dumps(result, indent=2, sort_keys=True)+'\n')
    return result


def partition(source, config_path, output):
    from avatar_assets.safety import artifacts, local_file, new_directory, preserve_failure, read_json, sha256, write_json
    from avatar_assets.runner import run_bounded
    source = local_file(source, 64*1024**2)
    config = validate_config(read_json(config_path))
    out = new_directory(output)
    try:
        (out/'sources').mkdir()
        for path in (Path(__file__), ROOT/'scripts/blender_partition_generated.py'):
            shutil.copyfile(path, out/'sources'/path.name)
        shutil.copyfile(source, out/'original.glb')
        write_json(out/'config.json', config)
        prepare(out/'original.glb', config, out)
        exe = ROOT/'.tools/Blender.app/Contents/MacOS/Blender'
        command = [str(exe), '--background', '--factory-startup', '--disable-autoexec', '--offline-mode',
                   '--python-exit-code', '1', '--threads', '2', '--python', str(out/'sources/blender_partition_generated.py'),
                   '--', str(out)]
        job = run_bounded(command, out/'partition.log', out, timeout=120, memory_mb=1024, output_mb=128)
        write_json(out/'job.json', job)
        require(job['exit_code'] == 0 and not job['failure'], 'Native partition failed; retained evidence is not a usable asset')
        native = read_json(out/'native-report.json')
        require(sha256(source) == config['source_sha256'], 'Original input changed')
        write_json(out/'manifest.json', {'schema_version': 1, 'kind': 'locally_authored_mesh_partition',
            'status': native['status'], 'source_sha256': config['source_sha256'], 'source_unchanged': True,
            'provider_calls': 0, 'provider_cost_usd': 0, 'native': native, 'job': job,
            'assumptions': 'New unseen closure surfaces are locally authored; original visible faces/UVs retained. Not rigged, animated or independently rights verified.',
            'rights': config['rights'], 'artifacts': artifacts(out)})
        return {'status': native['status'], 'out': str(out)}
    except Exception as error:
        preserve_failure(out, error)
        raise


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source')
    parser.add_argument('config')
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    print(json.dumps(partition(args.source, args.config, args.out), indent=2))
