"""Analytical exported-card visibility, not a native render or GPU-quality verdict."""

from io import BytesIO
import itertools
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / 'sources/host'))
from avatar_assets.safety import require, sha256, write_json
from avatar_assets.three_d import _GLB, _world_matrices

START = time.monotonic()
MAX_CANDIDATES = 32_000_000


def deadline():
    require(time.monotonic() - START < 54, 'Analytic audit exceeded its54-second internal deadline')


def stats(values):
    values = np.asarray(values, dtype=float)
    require(values.size and np.isfinite(values).all(), 'Invalid diagnostic distribution')
    return dict(zip(('min', 'p10', 'median', 'p90', 'max'), map(float, np.quantile(values, [0, .1, .5, .9, 1]))))


def unit(value):
    return value / np.linalg.norm(value)


def blender_xyz(points):
    return points[:, [0, 2, 1]] * np.asarray([1, -1, 1])


def load_scene(glb):
    doc, nodes = glb.doc, glb.doc['nodes']
    parents = {child: i for i, node in enumerate(nodes) for child in node.get('children', [])}
    matrices = [np.asarray(value).reshape(4, 4, order='F') for value in _world_matrices(nodes, parents)]
    primitives, boxes, image_cache, count = [], [], {}, 0
    for node_index, node in enumerate(nodes):
        if 'mesh' not in node:
            continue
        matrix = matrices[node_index]
        require(np.allclose(matrix, np.eye(4), atol=1e-7), 'This source needs explicit nonidentity bind-space recovery')
        mesh_points = []
        for primitive in doc['meshes'][node['mesh']]['primitives']:
            attrs = primitive['attributes']
            points = np.asarray(glb.accessor(attrs['POSITION'], 'VEC3', (5126,)), dtype=float)
            points = blender_xyz(points @ matrix[:3, :3].T + matrix[:3, 3])
            indices = np.asarray(glb.accessor(primitive['indices'], 'SCALAR', (5121, 5123, 5125), count_limit=300000), dtype=np.int32).reshape(-1, 3)
            require(len(indices) <= 100000 and indices.max() < len(points), 'Invalid source triangles')
            count += len(points)
            require(count <= 100000, 'Source exceeds100k exported vertices')
            material = doc['materials'][primitive['material']]
            is_card = node.get('name', '').startswith('PortableCoat_')
            row = {'name': node.get('name'), 'material': material['name'], 'points': points,
                'faces': indices, 'card': is_card, 'double_sided': material.get('doubleSided', False)}
            if not is_card:
                require(material.get('alphaMode', 'OPAQUE') == 'OPAQUE' and material.get('doubleSided', False),
                        'Opaque-body depth model requires actual double-sided opaque materials')
                row['fur_surface'] = '__short_pile_adaptive_v1' in material['name']
            else:
                require(material.get('alphaMode') == 'MASK' and not row['double_sided'], 'Expected actual single-sided cutout cards')
                info = material['pbrMetallicRoughness']['baseColorTexture']
                require(info.get('texCoord', 0) == 0, 'Card atlas must use UV0')
                require(material['pbrMetallicRoughness'].get('baseColorFactor', [1, 1, 1, 1])[3] == 1, 'Unexpected material alpha factor')
                uv = np.asarray(glb.accessor(attrs['TEXCOORD_0'], 'VEC2', (5126,)), dtype=float)
                color = np.asarray(glb.accessor(attrs['COLOR_0'], components=(5121, 5123, 5126)), dtype=float)
                require(color.shape[1] == 3 or color.shape[1] == 4 and np.all(color[:, 3] == 1),
                        'Card vertex alpha is neither implicit1 nor explicit constant1')
                texture = doc['textures'][info['index']]
                image_id = texture['source']
                if image_id not in image_cache:
                    image = doc['images'][image_id]
                    view = doc['bufferViews'][image['bufferView']]
                    start = view.get('byteOffset', 0)
                    data = glb.binary[start:start + view['byteLength']]
                    with Image.open(BytesIO(data)) as decoded:
                        require(decoded.format == 'PNG' and decoded.size == (1024, 1024), 'Unexpected card atlas')
                        alpha = decoded.convert('RGBA').getchannel('A')
                        mips = []
                        while True:
                            mips.append(np.asarray(alpha, dtype=np.float32) / 255)
                            if alpha.width == 1:
                                break
                            alpha = alpha.resize((alpha.width // 2, alpha.height // 2), Image.Resampling.BOX)
                    import hashlib
                    image_cache[image_id] = (mips, hashlib.sha256(data).hexdigest())
                mips, image_hash = image_cache[image_id]
                row.update(uv=uv, cutoff=material['alphaCutoff'], mips=mips, atlas_sha256=image_hash,
                           sampler=doc.get('samplers', [{}])[texture.get('sampler', 0)])
            primitives.append(row)
            mesh_points.append(points)
        merged = np.concatenate(mesh_points)
        boxes.extend(itertools.product(*zip(merged.min(axis=0), merged.max(axis=0))))
    return primitives, np.asarray(boxes, dtype=float), count


def camera(boxes, saved_bounds, offset, size=720):
    low, high = boxes.min(axis=0), boxes.max(axis=0)
    require(np.allclose(low, saved_bounds['min'], atol=2e-7, rtol=0)
            and np.allclose(high, saved_bounds['max'], atol=2e-7, rtol=0),
            'CPU rest bounds do not reproduce saved native inspection bounds')
    center, extent = (low + high) / 2, float(np.max(high - low))
    back = unit(np.asarray(offset, dtype=float))
    right = unit(np.cross([0., 0., 1.], back))
    up = unit(np.cross(back, right))
    basis = np.column_stack((right, up, back))
    projected = (boxes - center) @ basis
    scale = float(np.max(np.ptp(projected[:, :2], axis=0)) * 1.18)
    offset_xy = (projected[:, :2].max(axis=0) + projected[:, :2].min(axis=0)) / 2
    location = center + np.asarray(offset) * extent + basis[:, :2] @ offset_xy
    return {'basis': basis, 'center': center, 'offset_xy': offset_xy, 'scale': scale, 'size': size,
        'record': {'location_blender_z_up': location.tolist(), 'basis_columns_blender_z_up': basis.tolist(),
            'orthographic_scale_m': scale, 'resolution': size, 'saved_bounds_reproduced_atol_m': 2e-7,
            'method': 'Recovered immutable preview rest-camera recipe, including mesh-bbox fit and lateral recentering; no source.blend execution.'}}


def project(points, cam):
    view = (points - cam['center']) @ cam['basis']
    xy = (view[:, :2] - cam['offset_xy']) * cam['size'] / cam['scale'] + cam['size'] / 2
    return np.column_stack((xy, view[:, 2]))


def triangle_data(row, cam):
    projected = project(row['points'], cam)[row['faces']]
    xy = projected[:, :, :2]
    a, b = xy[:, 1] - xy[:, 0], xy[:, 2] - xy[:, 0]
    det = a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]
    low = np.maximum(np.ceil(xy.min(axis=1) - .5).astype(int), 0)
    high = np.minimum(np.floor(xy.max(axis=1) - .5).astype(int), cam['size'] - 1)
    candidates = np.maximum(high - low + 1, 0).prod(axis=1)
    require(int(candidates.sum()) <= MAX_CANDIDATES, 'Raster candidate bound exceeded')
    return projected, det, low, high, candidates


def samples(triangle, det, low, high):
    if abs(det) <= 1e-12 or np.any(high < low):
        return None
    x = np.arange(low[0], high[0] + 1, dtype=float) + .5
    y = np.arange(low[1], high[1] + 1, dtype=float) + .5
    a, b = triangle[1, :2] - triangle[0, :2], triangle[2, :2] - triangle[0, :2]
    dx, dy = x[None] - triangle[0, 0], y[:, None] - triangle[0, 1]
    u = (dx * b[1] - dy * b[0]) / det
    v = (dy * a[0] - dx * a[1]) / det
    inside = (u >= -1e-9) & (v >= -1e-9) & (u + v <= 1 + 1e-9)
    yy, xx = np.nonzero(inside)
    if not len(xx):
        return None
    weights = np.column_stack((1 - u[yy, xx] - v[yy, xx], u[yy, xx], v[yy, xx]))
    return yy + low[1], xx + low[0], weights


def bilinear(alpha, uv):
    size = alpha.shape[0]
    p = uv * size - .5
    base = np.floor(p).astype(int)
    fraction = p - base
    x0, y0 = np.clip(base[:, 0], 0, size - 1), np.clip(base[:, 1], 0, size - 1)
    x1, y1 = np.clip(base[:, 0] + 1, 0, size - 1), np.clip(base[:, 1] + 1, 0, size - 1)
    x, y = fraction[:, 0], fraction[:, 1]
    return ((1-x)*(1-y)*alpha[y0, x0] + x*(1-y)*alpha[y0, x1]
            + (1-x)*y*alpha[y1, x0] + x*y*alpha[y1, x1])


def projected_lod(projected, uv):
    a, b = projected[:, 1, :2] - projected[:, 0, :2], projected[:, 2, :2] - projected[:, 0, :2]
    screen = np.stack((a, b), axis=2)
    valid = np.abs(np.linalg.det(screen)) > 1e-12
    gradient = np.zeros_like(screen)
    gradient[valid] = np.stack((uv[:, 1] - uv[:, 0], uv[:, 2] - uv[:, 0]), axis=2)[valid] @ np.linalg.inv(screen[valid])
    footprint = np.max(np.linalg.norm(gradient * 1024, axis=1), axis=1)
    return np.clip(np.log2(np.maximum(1., footprint)), 0, 10), footprint


def alpha_fraction(row, uv_triangle, cache):
    key = (row['atlas_sha256'], row['cutoff'], tuple(sorted(map(tuple, np.round(uv_triangle, 7)))))
    if key not in cache:
        triangle = np.column_stack((uv_triangle * 1024, np.zeros(3)))
        a, b = triangle[1, :2] - triangle[0, :2], triangle[2, :2] - triangle[0, :2]
        det = a[0] * b[1] - a[1] * b[0]
        low = np.maximum(np.ceil(triangle[:, :2].min(axis=0) - .5).astype(int), 0)
        high = np.minimum(np.floor(triangle[:, :2].max(axis=0) - .5).astype(int), 1023)
        hit = samples(triangle, det, low, high)
        require(hit is not None, 'Native atlas triangle has no texel support')
        y, x, _ = hit
        cache[key] = float(np.mean(row['mips'][0][y, x] >= row['cutoff']))
    return cache[key]


def card_widths(row, projected):
    require(len(row['faces']) % 4 == 0, 'Card triangle grouping is not four-per-ribbon')
    widths = []
    for start in range(0, len(row['faces']), 4):
        ids = np.unique(row['faces'][start:start+4])
        positions = row['points'][ids]
        unique, first = np.unique(positions, axis=0, return_index=True)
        require(len(unique) == 6, 'Exported ribbon grouping does not preserve six geometric positions')
        ids = ids[first]
        uv = row['uv'][ids]
        levels = np.unique(uv[:, 1])
        require(len(levels) == 3, 'Ribbon UV rows are not recoverable')
        rows = []
        for level in levels:
            pair = ids[uv[:, 1] == level]
            require(len(pair) == 2, 'Ribbon UV row does not contain two sides')
            rows.append(float(np.linalg.norm(projected[pair[0], :2] - projected[pair[1], :2])))
        widths.append(max(rows))
    return np.asarray(widths)


def audit_view(primitives, cam, name, alpha_cache):
    size = cam['size']
    depth = np.full((size, size), -np.inf)
    owner_fur = np.zeros((size, size), dtype=bool)
    body_candidates, card_candidates = 0, 0
    for row in primitives:
        if row['card']:
            continue
        projected, det, low, high, candidates = triangle_data(row, cam)
        body_candidates += int(candidates.sum())
        for index, triangle in enumerate(projected):
            if index % 512 == 0:
                deadline()
            hit = samples(triangle, det[index], low[index], high[index])
            if hit is None:
                continue
            y, x, bary = hit
            z = bary @ triangle[:, 2]
            closer = z > depth[y, x]
            depth[y[closer], x[closer]] = z[closer]
            owner_fur[y[closer], x[closer]] = row['fur_surface']
    require(body_candidates <= MAX_CANDIDATES, 'Aggregate body raster exceeds32M candidates')
    masks = {key: np.zeros((size, size), dtype=bool) for key in
        ('geometric_front', 'geometric_two_sided', 'alpha_lod0_front', 'alpha_lod0_two_sided', 'alpha_mip_front', 'alpha_mip_two_sided')}
    summaries = []
    for row in primitives:
        if not row['card']:
            continue
        projected, det, low, high, candidates = triangle_data(row, cam)
        card_candidates += int(candidates.sum())
        uv_tri = row['uv'][row['faces']]
        lods, footprints = projected_lod(projected, uv_tri)
        front = det > 0
        areas = np.abs(det) / 2
        fractions = np.asarray([alpha_fraction(row, uv, alpha_cache) for uv in uv_tri])
        positions = project(row['points'], cam)
        widths = card_widths(row, positions)
        card_areas = areas.reshape(-1, 4).sum(axis=1)
        for index, triangle in enumerate(projected):
            if index % 512 == 0:
                deadline()
            hit = samples(triangle, det[index], low[index], high[index])
            if hit is None:
                continue
            y, x, bary = hit
            z = bary @ triangle[:, 2]
            visible = z >= depth[y, x]
            if not visible.any():
                continue
            y, x, bary = y[visible], x[visible], bary[visible]
            uv = bary @ uv_tri[index]
            alpha0 = bilinear(row['mips'][0], uv) >= row['cutoff']
            lod = float(lods[index])
            lo, hi = int(math.floor(lod)), int(math.ceil(lod))
            filtered = bilinear(row['mips'][lo], uv) * (hi-lod if hi != lo else 1)
            if hi != lo:
                filtered += bilinear(row['mips'][hi], uv) * (lod-lo)
            alpham = filtered >= row['cutoff']
            masks['geometric_two_sided'][y, x] = True
            masks['alpha_lod0_two_sided'][y[alpha0], x[alpha0]] = True
            masks['alpha_mip_two_sided'][y[alpham], x[alpham]] = True
            if front[index]:
                masks['geometric_front'][y, x] = True
                masks['alpha_lod0_front'][y[alpha0], x[alpha0]] = True
                masks['alpha_mip_front'][y[alpham], x[alpham]] = True
        summaries.append({'material': row['material'], 'cards': len(widths), 'triangles': len(areas),
            'triangles_front_facing': int(front.sum()), 'triangles_back_facing': int((~front).sum()),
            'projected_area_px2_all_summed': float(areas.sum()), 'projected_area_px2_front_summed': float(areas[front].sum()),
            'projected_alpha_lod0_area_px2_all_summed_upper_bound': float((areas * fractions).sum()),
            'projected_alpha_lod0_area_px2_front_summed_upper_bound': float((areas[front] * fractions[front]).sum()),
            'native_atlas_triangle_alpha_fraction': stats(fractions),
            'triangle_area_px2': stats(areas), 'card_area_px2': stats(card_areas),
            'card_area_below1px2': int((card_areas < 1).sum()), 'card_max_row_width_px': stats(widths),
            'card_width_below1px': int((widths < 1).sum()),
            'approx_ten_fiber_spacing_px': stats(widths / 10),
            'atlas_texels_per_pixel_max_axis': stats(footprints), 'box_mip_lod': stats(lods),
            'display_size_analytic_footprints': [{'size': output, 'card_width_px': stats(widths * output / size),
                'card_width_below1px': int((widths * output / size < 1).sum()),
                'approx_fiber_spacing_px': stats(widths * output / size / 10),
                'box_mip_lod': stats(np.clip(lods + math.log2(size / output), 0, 10))} for output in (360, 180)]})
    require(card_candidates <= MAX_CANDIDATES, 'Aggregate card raster exceeds32M candidates')
    body = np.isfinite(depth)
    counts = {key: {'pixels': int(mask.sum()), 'over_body_pixels': int((mask & body).sum()),
        'over_fur_selected_body_pixels': int((mask & owner_fur).sum()), 'outside_body_silhouette_pixels': int((mask & ~body).sum())}
        for key, mask in masks.items()}
    path = HERE / (name + '-analytic-masks.npz')
    np.savez_compressed(path, body=body, fur_selected_body=owner_fur, **masks)
    return {'view': name, 'camera': cam['record'], 'body_pixels': int(body.sum()), 'fur_selected_body_pixels': int(owner_fur.sum()),
        'cards': summaries, 'visible_pixel_center_counts': counts,
        'two_sided_recovery_alpha_lod0_pixels': int((masks['alpha_lod0_two_sided'] & ~masks['alpha_lod0_front']).sum()),
        'two_sided_recovery_alpha_mip_pixels': int((masks['alpha_mip_two_sided'] & ~masks['alpha_mip_front']).sum()),
        'candidate_pixels': {'body': body_candidates, 'cards': card_candidates},
        'mask_artifact': {'path': path.name, 'sha256': sha256(path), 'scope': 'CPU pixel-center diagnostic masks, not a new render or generated-avatar output'}}


def main():
    inputs = json.loads((HERE / 'inputs.json').read_text())['files']
    for row in inputs:
        require(sha256(Path(row['path'])) == row['sha256'], 'Bound source changed before audit')
    prepared = json.loads((HERE / 'prepared.json').read_text())
    for row in prepared['frozen_files']:
        require(sha256(HERE / row['path']) == row['sha256'], 'Frozen audit source changed')
    model = Path(inputs[0]['path'])
    glb = _GLB(model)
    native = json.loads((HERE / 'snapshots/native-inspection.json').read_text())
    require(native['source_sha256'] == glb.hash, 'Preview is not bound to the measured model')
    primitives, boxes, vertices = load_scene(glb)
    views, alpha_cache = [], {}
    for name, offset in (('front', (0, -3, .05)), ('three-quarter', (-1.6, -3, .1))):
        cam = camera(boxes, native['bounds_blender_z_up'], offset)
        result = audit_view(primitives, cam, name, alpha_cache)
        views.append(result)
        write_json(HERE / (name + '.json'), result)
    for row in inputs:
        require(sha256(Path(row['path'])) == row['sha256'], 'Bound source changed during audit')
    report = {'schema_version': 1, 'status': 'cpu_visibility_audit_complete', 'source_model_sha256': glb.hash,
        'exported_vertices': vertices, 'cards': sum(len(row['faces']) // 4 for row in primitives if row['card']),
        'views': views, 'alpha_integration_cache_entries': len(alpha_cache),
        'atlas': [{'material': row['material'], 'image_sha256': row['atlas_sha256'], 'cutoff': row['cutoff'],
            'sampler': row['sampler'], 'full_resolution_alpha_occupancy': float((row['mips'][0] >= row['cutoff']).mean()),
            'mip_mean_alpha': [float(value.mean()) for value in row['mips']],
            'mip_alpha_occupancy': [float((value >= row['cutoff']).mean()) for value in row['mips']]} for row in primitives if row['card']],
        'method': {'pose': 'Actual mesh bind/rest coordinates, independently matching saved preview rest bounds',
            'projection': 'Recovered saved720px front/three-quarter orthographic preview recipes',
            'culling': 'Geometric triangle winding versus view; source glTF cards are single-sided, opaque body double-sided',
            'occlusion': 'Nearest opaque body triangle at exact pixel centers; cards at same/nearer depth counted, no depth bias',
            'lod0': 'Actual embedded alpha with bilinear sampling and actual material cutoff',
            'mips': 'Analytic BOX pyramid plus trilinear bilinear sampling, derivative max-axis LOD; not measured GPU mip construction/anisotropy',
            'density_upper_bound': 'Summed projected triangle area can overcount overlap; alpha-weighted sum also ignores occlusion',
            'fiber_spacing': 'Whole-ribbon maximum projected row width divided by source10-fiber count; not individually resolved GPU strands'},
        'limitations': ['No GPU/renderer culling, anisotropy, alpha-to-coverage, MSAA, denoising, lighting or shading measurement.',
            'Cycles preview may not enforce material culling like a conformant glTF viewer; this audit reports the contract and counterfactual.',
            'Pixel-center sampling can miss subpixel geometry and depends on phase; continuous area upper bounds are reported separately.',
            'Two static rest views only; no animation or moving-part coat claim.',
            'Source contact seams and undercoat shading are separate from the card-coverage measurement.'],
        'all_inputs_unchanged': True, 'native_jobs': 0, 'provider_calls': 0, 'quality_approved': False,
        'elapsed_seconds': time.monotonic() - START}
    write_json(HERE / 'audit.json', report)
    print(json.dumps({'status': report['status'], 'views': len(views), 'cards': report['cards'], 'elapsed_seconds': report['elapsed_seconds']}))


if __name__ == '__main__':
    main()
