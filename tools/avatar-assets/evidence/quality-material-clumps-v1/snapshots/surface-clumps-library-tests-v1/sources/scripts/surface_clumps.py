"""Experimental source-face micro-clumps; no public preset or dense-fur claim."""

import hashlib
import itertools
import time

import numpy as np

from avatar_assets.partition_generated import triangles_intersect_3d
if __package__:
    from .fringe_color_atlas import encode_srgb
    from .groom_flow import PreparedFlow, evaluate_prepared_flow
else:
    from fringe_color_atlas import encode_srgb
    from groom_flow import PreparedFlow, evaluate_prepared_flow


POLICY = 'experimental-paired-surface-clumps-v1'
MAX_SECONDS = 25
MAX_GRID_REFERENCES = 1_000_000
MAX_INTERSECTION_PAIRS = 200_000


class SurfaceClumpError(ValueError):
    def __init__(self, report):
        self.report = report
        super().__init__(report['failure'])


def _array(value, shape, kinds, name):
    if (not isinstance(value, np.ndarray) or value.shape != shape or value.dtype.kind not in kinds
            or value.dtype.hasobject or not np.isfinite(value).all()):
        raise ValueError('Invalid '+name+' array')
    return value


def _unit(value):
    length = np.linalg.norm(value, axis=-1, keepdims=True)
    if np.any(length < 1e-10):
        raise ValueError('Degenerate direction or face')
    return value/length


def _hash(value):
    return hashlib.sha256(value.tobytes(order='C')).hexdigest()


def _grid_cells(low, high, step=.02):
    a, b = np.floor(low/step).astype(np.int64), np.floor(high/step).astype(np.int64)
    if np.prod(b-a+1) > 4096:
        raise ValueError('Triangle exceeds bounded spatial-grid footprint')
    return itertools.product(*(range(int(x), int(y)+1) for x, y in zip(a, b)))


def _intersections(source_triangles, patch_triangles, deadline):
    """Reject proper crossings, including adjacent geometry; root edge contact is allowed."""
    grid, references = {}, 0
    low, high = source_triangles.min(axis=1), source_triangles.max(axis=1)
    for index, (a, b) in enumerate(zip(low, high)):
        if index % 1024 == 0:
            deadline()
        for cell in _grid_cells(a-1e-8, b+1e-8):
            grid.setdefault(cell, []).append(index)
            references += 1
            if references > MAX_GRID_REFERENCES:
                raise ValueError('Spatial-grid reference bound exceeded')
    tested, hits, patch_grid = 0, [], {}
    patch_low, patch_high = patch_triangles.min(axis=1), patch_triangles.max(axis=1)
    for index, triangle in enumerate(patch_triangles):
        deadline()
        cells = list(_grid_cells(patch_low[index]-1e-8, patch_high[index]+1e-8))
        candidates = sorted({j for cell in cells for j in grid.get(cell, ())})
        for kind, ids, triangles, lows, highs in (
            ('source', candidates, source_triangles, low, high),
            ('patch', sorted({j for cell in cells for j in patch_grid.get(cell, ())}),
             patch_triangles, patch_low, patch_high)):
            if not ids:
                continue
            ids = np.asarray(ids)
            overlap = ((lows[ids] <= patch_high[index]+1e-8).all(axis=1)
                       & (highs[ids] >= patch_low[index]-1e-8).all(axis=1))
            for other in ids[overlap]:
                tested += 1
                if tested > MAX_INTERSECTION_PAIRS:
                    raise ValueError('Intersection pair bound exceeded')
                if triangles_intersect_3d(triangle, triangles[other]):
                    hits.append({'patch_triangle': index, 'other_kind': kind, 'other_triangle': int(other),
                                 'patch_points_m': triangle.tolist(), 'other_points_m': triangles[other].tolist()})
                    return {'passed': False, 'complete': False, 'tested_pairs': tested,
                            'grid_references': references, 'witnesses': hits}
        for cell in cells:
            patch_grid.setdefault(cell, []).append(index)
    return {'passed': True, 'complete': True, 'tested_pairs': tested,
            'grid_references': references, 'witnesses': []}


def build_surface_clumps(source, fringe, atlas_rgba, prepared_flow, sample_source_rgb, settings,
                         *, source_export_vertices, source_export_triangles):
    """Keep all root edges; replace explicit IDs with two lifted source-face triangles.

    The trusted sampler accepts (source_material_id, UV[N,2]) and returns linear
    RGB[N,3]; callers bind it to the existing source sampler, never a new repaint.
    Native/export integration and full posed-skin verification remain external.
    """
    started = time.monotonic()
    report = {'schema_version': 1, 'policy': POLICY, 'status': 'unsupported', 'quality_approved': False,
              'body_modified': False, 'roots_reselected': False, 'native_calls': 0}

    def require(condition, message):
        if not condition:
            raise ValueError(message)

    def deadline():
        require(time.monotonic()-started <= MAX_SECONDS, 'Surface-clump25-second deadline exceeded')

    try:
        fields = {'positions', 'triangles', 'corner_normals', 'uv0', 'material_ids', 'part_ids',
                  'fur_faces', 'protected_faces', 'joints', 'weights'}
        require(isinstance(source, dict) and set(source) == fields, 'Exact source-face/skin evidence is required')
        require(isinstance(settings, dict) and set(settings) == {'profile', 'seed', 'lift_m', 'surface_fin_ids'}
                and settings['profile'] == POLICY, 'Unsupported surface-clump settings')
        require(type(settings['seed']) is int and 0 <= settings['seed'] <= 2**31-1
                and type(settings['lift_m']) in (int, float) and np.isfinite(settings['lift_m'])
                and 0 < settings['lift_m'] <= .0012, 'Invalid seed or outward lift')
        require(isinstance(prepared_flow, PreparedFlow) and callable(sample_source_rgb), 'Bounded shared flow and source sampler required')
        points, faces = source['positions'], source['triangles']
        require(isinstance(points, np.ndarray) and points.ndim == 2 and 3 <= len(points) <= 100000
                and isinstance(faces, np.ndarray) and faces.ndim == 2 and 1 <= len(faces) <= 100000,
                'Source geometry exceeds100k bounds')
        nv, nf = len(points), len(faces)
        _array(points, (nv, 3), 'f', 'source positions'); _array(faces, (nf, 3), 'iu', 'source triangles')
        require(np.abs(points).max() <= 10000 and faces.min() >= 0 and faces.max() < nv, 'Invalid source geometry range')
        normals = _array(source['corner_normals'], (nf, 3, 3), 'f', 'corner normals')
        source_uv = _array(source['uv0'], (nf, 3, 2), 'f', 'UV0')
        require(np.max(abs(source_uv)) <= 1e6 and np.max(abs(np.linalg.norm(normals, axis=2)-1)) < .001,
                'Source UV or unit-normal bound exceeded')
        for key in ('material_ids', 'part_ids'):
            _array(source[key], (nf,), 'iu', key)
            require(source[key].min() >= 0 and source[key].max() < 128, 'Invalid source ownership range')
        for key in ('fur_faces', 'protected_faces'):
            _array(source[key], (nf,), 'b', key)
        joints = _array(source['joints'], (nv, 4), 'iu', 'source joints')
        weights = _array(source['weights'], (nv, 4), 'f', 'source weights')
        require(joints.min() >= 0 and joints.max() < 64 and weights.min() >= 0 and weights.max() <= 1
                and np.max(abs(weights.astype(float).sum(axis=1)-1)) <= 1e-6, 'Invalid unpruned four-weight source skin')
        positive = weights > 0
        repeated = (joints[:, :, None] == joints[:, None, :]) & positive[:, :, None] & positive[:, None, :]
        require(not np.any(repeated & np.triu(np.ones((4, 4), dtype=bool), 1)), 'Duplicate positive source influence')
        require(isinstance(fringe, dict) and {'positions', 'normals', 'joints', 'weights', 'uv_gltf',
                'root_vertex_ids', 'root_face_ids', 'root_corner_ids', 'destination_for_source_fin'} <= set(fringe)
                and isinstance(fringe.get('positions'), np.ndarray), 'Frozen fringe arrays required')
        count = len(fringe['positions'])//4
        require(1 <= count <= 4800, 'Root count exceeds1..4800')
        for key, width, kind in (('positions', 3, 'f'), ('normals', 3, 'f'), ('joints', 4, 'iu'), ('weights', 4, 'f'), ('uv_gltf', 2, 'f')):
            _array(fringe[key], (count*4, width), kind, 'fringe '+key)
        roots = _array(fringe['root_vertex_ids'], (count, 2), 'iu', 'root source IDs')
        root_faces = _array(fringe['root_face_ids'], (count,), 'iu', 'root source faces')
        root_corners = _array(fringe['root_corner_ids'], (count, 2), 'iu', 'root source corners')
        require(roots.min() >= 0 and roots.max() < nv and root_faces.min() >= 0 and root_faces.max() < nf
                and root_corners.min() >= 0 and root_corners.max() < 3, 'Root source index outside bounds')
        selected = settings['surface_fin_ids']
        require(isinstance(selected, np.ndarray) and selected.ndim == 1 and 1 <= len(selected) <= count,
                'Explicit bounded surface root IDs required')
        _array(selected, (len(selected),), 'iu', 'selected root IDs')
        require(selected.min() >= 0 and selected.max() < count and len(np.unique(selected)) == len(selected),
                'Duplicate or invalid selected root IDs')
        selected = np.sort(selected.astype(np.int32)); selected_set = set(map(int, selected))
        columns = min(count, 64); rows = (count+columns-1)//columns
        _array(atlas_rgba, (rows*32, columns*64, 4), 'u', 'existing atlas')
        require(atlas_rgba.dtype == np.uint8 and atlas_rgba.shape[0] <= 4096, 'Atlas byte/dimension bound exceeded')
        destination = _array(fringe['destination_for_source_fin'], (count,), 'iu', 'cell permutation')
        require(np.array_equal(np.sort(destination), np.arange(count)), 'Atlas permutation is not exact')
        corners = 6*len(selected)+12*(count-len(selected)); triangle_count = corners//3
        require(type(source_export_vertices) is int and nv <= source_export_vertices <= 100000
                and type(source_export_triangles) is int and nf <= source_export_triangles <= 100000
                and source_export_vertices+corners+4096 <= 100000
                and source_export_triangles+triangle_count <= 100000, 'Whole-model100k export budget exceeded')
        require(np.array_equal(points[roots], fringe['positions'].reshape(count, 4, 3)[:, :2])
                and np.array_equal(faces[root_faces[:, None], root_corners], roots), 'Frozen root positions/ownership differ')
        all_edges = np.sort(faces[:, [[0, 1], [1, 2], [2, 0]]].reshape(-1, 2), axis=1)
        edge_keys = all_edges[:, 0].astype(np.int64)*nv+all_edges[:, 1]
        order = np.argsort(edge_keys, kind='stable'); sorted_keys = edge_keys[order]
        root_edges = np.sort(roots[selected], axis=1)
        wanted = root_edges[:, 0].astype(np.int64)*nv+root_edges[:, 1]
        left, right = np.searchsorted(sorted_keys, wanted, 'left'), np.searchsorted(sorted_keys, wanted, 'right')
        require(np.all(right-left == 2), 'Selected root lacks exactly two indexed source faces')
        pairs = np.stack((order[left]//3, order[left+1]//3), axis=1)
        require(len(np.unique(pairs)) == pairs.size, 'Selected patches overlap source faces')
        require(source['fur_faces'][pairs].all() and not source['protected_faces'][pairs].any()
                and np.all(source['material_ids'][pairs[:, 0]] == source['material_ids'][pairs[:, 1]])
                and np.all(source['part_ids'][pairs[:, 0]] == source['part_ids'][pairs[:, 1]]),
                'Patch crosses protected fur, material or part ownership')
        source_triangles = points[faces].astype(float)
        geometric = _unit(np.cross(source_triangles[:, 1]-source_triangles[:, 0], source_triangles[:, 2]-source_triangles[:, 0]))
        patch_source_faces = pairs.ravel(); patch_ids = faces[patch_source_faces]
        patch_normals = _unit(normals[patch_source_faces].astype(float))
        source_dot = np.sum(patch_normals*geometric[patch_source_faces, None], axis=2)
        require(source_dot.min() >= .5, 'Source corner normals cannot support outward surface patches')
        patch_points = source_triangles[patch_source_faces].copy()
        # Only this patch's edge is fixed; a neighboring selected edge does not own its tip.
        own_roots = np.repeat(roots[selected], 2, axis=0)
        root_mask = (patch_ids[:, :, None] == own_roots[:, None]).any(axis=2)
        patch_points += patch_normals*settings['lift_m']*(~root_mask)[..., None]
        patch_points = patch_points.astype(np.float32).astype(float)
        changed_normals = _unit(np.cross(patch_points[:, 1]-patch_points[:, 0], patch_points[:, 2]-patch_points[:, 0]))
        turns = np.degrees(np.arccos(np.clip(np.sum(changed_normals*geometric[patch_source_faces], axis=1), -1, 1)))
        bad_turns = np.flatnonzero(turns > 15)
        report.update(surface_cards=len(selected), contour_fins=count-len(selected),
            root_edges_preserved=count, source_patch_faces=len(patch_source_faces),
            source_corner_min_outward_dot=float(source_dot.min()), maximum_geometric_turn_degrees=float(turns.max()),
            normal_turn_witnesses=[{'source_face': int(patch_source_faces[i]), 'fin': int(selected[i//2]),
                                   'degrees': float(turns[i])} for i in bad_turns[:16]],
            export_budget={'vertices_with_reserve': source_export_vertices+corners+4096,
                           'triangles': source_export_triangles+triangle_count, 'reserve_vertices': 4096})
        require(not len(bad_turns), 'Offset patch turns more than15degrees from its source face')
        intersection = _intersections(source_triangles, patch_points, deadline)
        report['intersections'] = intersection
        for witness in intersection['witnesses']:
            witness['source_face'] = int(patch_source_faces[witness['patch_triangle']])
            witness['fin'] = int(selected[witness['patch_triangle']//2])
        require(intersection['passed'], 'Offset patch intersects source or another patch')
        deadline()
        output_atlas = atlas_rgba.copy()
        arrays = {k: [] for k in ('positions', 'normals', 'uv', 'joints', 'weights', 'source_vertex_ids')}
        triangle_sources, fin_ranges, alpha_areas, fallback, start = [], [], [], 0, 0
        lookup = {int(fin): index for index, fin in enumerate(selected)}
        rng = np.random.default_rng(settings['seed'])
        yy, xx = np.mgrid[:28, :60]; u, v = (xx.ravel()+.5)/60, (yy.ravel()+.5)/28
        for fin in range(count):
            if fin % 32 == 0:
                deadline()
            if fin not in selected_set:
                ids = fin*4+np.array([0, 1, 3, 0, 3, 2, 3, 1, 0, 2, 3, 0])
                for key, old in (('positions', 'positions'), ('normals', 'normals'), ('uv', 'uv_gltf'),
                                 ('joints', 'joints'), ('weights', 'weights')):
                    arrays[key].append(fringe[old][ids].copy())
                arrays['source_vertex_ids'].append(np.tile(roots[fin], 2)[ids-fin*4])
                triangle_sources.extend([int(root_faces[fin])]*4)
                fin_ranges.append((start, 12)); start += 12; continue
            index = lookup[fin]; pair = pairs[index]; a, b = roots[fin]
            row = int(destination[fin]); x, y = row % columns*64, row//columns*32
            left_uv, right_uv = (x+2)/output_atlas.shape[1], (x+62)/output_atlas.shape[1]
            top_uv, bottom_uv = (y+2)/output_atlas.shape[0], (y+30)/output_atlas.shape[0]
            mid_u, mid_v = (left_uv+right_uv)*.5, (top_uv+bottom_uv)*.5
            vertex_uv = {int(a): (left_uv, mid_v), int(b): (right_uv, mid_v)}
            center = points[[a, b]].astype(float).mean(axis=0)
            outward = _unit(patch_normals[index*2:index*2+2].sum(axis=(0, 1)))
            flow = np.asarray(evaluate_prepared_flow(center, prepared_flow)[0])
            flow -= outward*np.dot(flow, outward)
            if np.linalg.norm(flow) < 1e-8:
                flow = np.cross(outward, points[b]-points[a]); fallback += 1
            flow = _unit(flow); across = _unit(np.cross(outward, flow))
            phase_offset = rng.random()
            cell = np.zeros((28*60, 4), dtype=np.uint8)
            visible = 0; footprint = 0
            for side, face_id in enumerate(pair):
                face = faces[face_id]
                c = int(next(vertex for vertex in face if vertex not in (a, b)))
                vertex_uv[c] = (mid_u, top_uv if side == 0 else bottom_uv)
                wc = 1-2*v if side == 0 else 2*v-1
                wb = u-.5*wc; wa = 1-wb-wc
                belongs = v <= .5 if side == 0 else v > .5
                inside = belongs & (wa >= 0) & (wb >= 0) & (wc >= 0)
                ordered = [int(np.flatnonzero(face == vertex)[0]) for vertex in (a, b, c)]
                bary = np.stack((wa, wb, wc), axis=1)
                clipped = np.maximum(bary, 0); clipped /= clipped.sum(axis=1, keepdims=True)
                q = clipped@source_uv[face_id, ordered]
                color = sample_source_rgb(int(source['material_ids'][face_id]), q[belongs])
                _array(color, (int(belongs.sum()), 3), 'f', 'sampled source color')
                require(color.min() >= 0 and color.max() <= 1, 'Source sampler returned out-of-range linear color')
                cell[belongs, :3] = encode_srgb(color)
                local = bary@points[[a, b, c]].astype(float)-center
                cross_distance = local@across; along = local@flow
                lane = np.floor(cross_distance/.0024)
                phase = np.mod(along/.006+phase_offset+lane*.38196601125, 1)
                radius = .00085*np.maximum(0, 1-phase/.9)**.45
                distance = abs(np.mod(cross_distance+.0012, .0024)-.0012)
                alpha = np.clip((radius-distance)/.00018+.5, 0, 1)
                alpha *= np.clip(np.minimum(wa, wb)/.04, 0, 1)*inside
                cell[inside, 3] = np.rint(alpha[inside]*255).astype(np.uint8)
                visible += int(np.count_nonzero(alpha[inside] >= .18)); footprint += int(inside.sum())
            inner = cell.reshape(28, 60, 4)
            output_atlas[y:y+32, x:x+64] = 0
            output_atlas[y:y+32, x:x+64, :3] = np.pad(inner[..., :3], ((2, 2), (2, 2), (0, 0)), mode='edge')
            output_atlas[y+2:y+30, x+2:x+62, 3] = inner[..., 3]
            source_ids = faces[pair].ravel()
            arrays['positions'].append(patch_points[index*2:index*2+2].reshape(-1, 3).astype(np.float32))
            arrays['normals'].append(patch_normals[index*2:index*2+2].reshape(-1, 3).astype(np.float32))
            arrays['uv'].append(np.asarray([vertex_uv[int(i)] for i in source_ids], dtype=np.float32))
            arrays['joints'].append(joints[source_ids].copy()); arrays['weights'].append(weights[source_ids].copy())
            arrays['source_vertex_ids'].append(source_ids.copy()); triangle_sources.extend(map(int, pair))
            fin_ranges.append((start, 6)); start += 6; alpha_areas.append(visible/footprint)
        result = {key: np.concatenate(value) for key, value in arrays.items()}
        result.update(triangles=np.arange(corners, dtype=np.int32).reshape(-1, 3), pixels=output_atlas,
                      source_face_ids=np.asarray(triangle_sources, dtype=np.int32),
                      fin_vertex_ranges=np.asarray(fin_ranges, dtype=np.int32), surface_fin_ids=selected)
        report.update(status='cpu_component_arrays_built_pending_independent_review', atlas_dimensions=[output_atlas.shape[1], output_atlas.shape[0]],
            atlas_input_sha256=_hash(atlas_rgba), atlas_output_sha256=_hash(output_atlas),
            alpha_fraction_within_source_triangles={'min': min(alpha_areas), 'mean': float(np.mean(alpha_areas)), 'max': max(alpha_areas)},
            flow_tangent_fallback_cards=fallback, lift_m=settings['lift_m'], skin_policy='Exact source vertex rows; no pruning or interpolation',
            alpha_policy='Source-face physical micro-strokes; existing RGB sampler; no root-color extrusion',
            limitations=['Not dense native hair or Muse parity.', 'Rest intersection screen is not a motion collision proof.',
                         'Alpha mip/lighting/tiling and actual appearance remain unverified.'])
        result['report'] = report
        return result
    except ValueError as exc:
        report.update(failure=str(exc), status='unsupported')
        raise SurfaceClumpError(report) from exc
    finally:
        report['elapsed_seconds'] = time.monotonic()-started
